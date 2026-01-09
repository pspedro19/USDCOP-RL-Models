-- =====================================================================
-- USD/COP Trading System - Unified Inference Schema v3.1
-- =====================================================================
--
-- Este script crea la arquitectura simplificada de 3 tablas/vistas:
--   1. usdcop_m5_ohlcv        (ya existe - no modificar)
--   2. macro_indicators_daily (crear - datos macro diarios)
--   3. inference_features_5m  (crear - vista materializada)
--
-- FEATURE SPLIT (13 features totales para modelo PPO v11):
-- ┌─────────────────────────────────────────────────────────────────┐
-- │ CALCULADOS EN SQL (10 features):                                │
-- │   - log_ret_5m, log_ret_1h, log_ret_4h (retornos OHLCV)        │
-- │   - dxy_z, vix_z, embi_z (z-scores macro)                      │
-- │   - dxy_change_1d, brent_change_1d, usdmxn_ret_1h (cambios %)  │
-- │   - rate_spread (yield curve)                                   │
-- ├─────────────────────────────────────────────────────────────────┤
-- │ CALCULADOS EN PYTHON SERVICE (3 features):                      │
-- │   - rsi_9 (RSI period=9)                                        │
-- │   - atr_pct (ATR% period=10)                                    │
-- │   - adx_14 (ADX period=14)                                      │
-- ├─────────────────────────────────────────────────────────────────┤
-- │ ELIMINADOS EN V14 (NO usar):                                    │
-- │   - hour_sin, hour_cos (bajo valor predictivo)                  │
-- │   - bb_position, dxy_mom_5d, vix_regime, brent_vol_5d          │
-- └─────────────────────────────────────────────────────────────────┘
--
-- Autor: Pedro @ Lean Tech Solutions
-- Fecha: 2025-12-16 (v3.1 corregido)
-- SSOT: config/feature_config.json
-- Prerequisito: 01-essential-usdcop-init.sql debe estar ejecutado
-- =====================================================================

-- =====================================================================
-- TABLA 2: macro_indicators_daily
-- =====================================================================
-- Propósito: Almacenar indicadores macroeconómicos diarios
-- Fuentes: TwelveData, FRED, BCRP (EMBI)
-- Actualización: 3 veces/día (7:55, 10:30, 12:00 COT)

DROP TABLE IF EXISTS macro_indicators_daily CASCADE;

CREATE TABLE macro_indicators_daily (
    date            DATE PRIMARY KEY,

    -- Índices principales
    dxy             NUMERIC(10, 4),      -- US Dollar Index (80-130)
    vix             NUMERIC(10, 4),      -- Volatility Index (8-90)
    embi            NUMERIC(10, 4),      -- EMBI Colombia (50-1500)

    -- Commodities
    brent           NUMERIC(10, 4),      -- Brent Crude Oil (20-200)
    wti             NUMERIC(10, 4),      -- WTI Crude Oil (20-200)
    gold            NUMERIC(10, 4),      -- Gold XAU/USD (1000-3000)
    coffee          NUMERIC(10, 4),      -- Coffee C Futures

    -- Tasas de interés USA
    fed_funds       NUMERIC(8, 4),       -- Fed Funds Rate (0-10)
    treasury_2y     NUMERIC(8, 4),       -- Treasury 2Y Yield (0-10)
    treasury_10y    NUMERIC(8, 4),       -- Treasury 10Y Yield (0-10)

    -- Tasas Colombia
    tpm_colombia    NUMERIC(8, 4),       -- Tasa Política Monetaria
    ibr_overnight   NUMERIC(8, 4),       -- IBR Overnight

    -- FX pairs correlacionados
    usdmxn          NUMERIC(10, 4),      -- USD/MXN (15-30)
    usdclp          NUMERIC(10, 4),      -- USD/CLP (600-1200)

    -- Equity
    colcap          NUMERIC(10, 4),      -- COLCAP Index

    -- Metadata
    source          VARCHAR(100),        -- Fuentes combinadas
    is_complete     BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_macro_date_desc ON macro_indicators_daily (date DESC);

-- Comentarios
COMMENT ON TABLE macro_indicators_daily IS 'Indicadores macroeconómicos diarios para USD/COP - SSOT';
COMMENT ON COLUMN macro_indicators_daily.dxy IS 'US Dollar Index - correlación ~+0.8 con USD/COP';
COMMENT ON COLUMN macro_indicators_daily.vix IS 'Volatility Index - flight-to-safety indicator';
COMMENT ON COLUMN macro_indicators_daily.embi IS 'EMBI Colombia - riesgo país emergente';
COMMENT ON COLUMN macro_indicators_daily.is_complete IS 'TRUE cuando tiene todos los datos del día';


-- =====================================================================
-- VISTA 3: inference_features_5m (Vista Materializada)
-- =====================================================================
-- Propósito: Features pre-calculados para inferencia del modelo PPO
-- Definido por: config/feature_config.json
-- Actualización: REFRESH cada 5 minutos durante mercado

DROP MATERIALIZED VIEW IF EXISTS inference_features_5m CASCADE;

CREATE MATERIALIZED VIEW inference_features_5m AS
WITH
-- Paso 1: OHLCV con retornos calculados
ohlcv_with_returns AS (
    SELECT
        o.time AS timestamp,
        o.open,
        o.high,
        o.low,
        o.close,

        -- Retornos logarítmicos multi-timeframe
        LN(o.close / NULLIF(LAG(o.close, 1) OVER w, 0)) AS log_ret_5m,
        LN(o.close / NULLIF(LAG(o.close, 12) OVER w, 0)) AS log_ret_1h,
        LN(o.close / NULLIF(LAG(o.close, 48) OVER w, 0)) AS log_ret_4h,

        -- Retorno raw para reward (SIN normalizar)
        o.close / NULLIF(LAG(o.close, 1) OVER w, 0) - 1 AS _raw_ret_5m,

        -- Datos para cálculos técnicos (RSI, ATR, ADX calculados en Python)
        LAG(o.close, 1) OVER w AS prev_close,
        o.high - o.low AS range_hl,

        -- Rolling stats para técnicos simplificados
        AVG(o.close) OVER (ORDER BY o.time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
        STDDEV(o.close) OVER (ORDER BY o.time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std_20

    FROM usdcop_m5_ohlcv o
    WHERE o.time >= NOW() - INTERVAL '30 days'
    WINDOW w AS (ORDER BY o.time)
),

-- Paso 2: Macro data procesado
macro_processed AS (
    SELECT
        d.date,
        d.dxy,
        d.vix,
        d.embi,
        d.brent,
        d.treasury_2y,
        d.treasury_10y,
        d.usdmxn,

        -- Z-scores con parámetros fijos del entrenamiento
        (d.dxy - 103.0) / NULLIF(5.0, 0) AS dxy_z,
        (d.vix - 20.0) / NULLIF(10.0, 0) AS vix_z,
        (d.embi - 300.0) / NULLIF(100.0, 0) AS embi_z,

        -- Cambios diarios
        (d.dxy - LAG(d.dxy, 1) OVER (ORDER BY d.date)) /
            NULLIF(LAG(d.dxy, 1) OVER (ORDER BY d.date), 0) AS dxy_change_1d,

        (d.brent - LAG(d.brent, 1) OVER (ORDER BY d.date)) /
            NULLIF(LAG(d.brent, 1) OVER (ORDER BY d.date), 0) AS brent_change_1d,

        (d.usdmxn - LAG(d.usdmxn, 1) OVER (ORDER BY d.date)) /
            NULLIF(LAG(d.usdmxn, 1) OVER (ORDER BY d.date), 0) AS usdmxn_ret_1h,

        -- Rate spread
        d.treasury_10y - d.treasury_2y AS rate_spread

    FROM macro_indicators_daily d
    WHERE d.date >= CURRENT_DATE - INTERVAL '60 days'
)

-- Paso 3: JOIN y feature final
SELECT
    owr.timestamp,
    owr.close,

    -- Retornos (clipped)
    LEAST(GREATEST(COALESCE(owr.log_ret_5m, 0), -0.05), 0.05) AS log_ret_5m,
    LEAST(GREATEST(COALESCE(owr.log_ret_1h, 0), -0.05), 0.05) AS log_ret_1h,
    LEAST(GREATEST(COALESCE(owr.log_ret_4h, 0), -0.05), 0.05) AS log_ret_4h,

    -- Raw return para reward
    COALESCE(owr._raw_ret_5m, 0) AS _raw_ret_5m,

    -- Macro Z-scores (clipped a ±4)
    LEAST(GREATEST(COALESCE(m.dxy_z, 0), -4), 4) AS dxy_z,
    LEAST(GREATEST(COALESCE(m.vix_z, 0), -4), 4) AS vix_z,
    LEAST(GREATEST(COALESCE(m.embi_z, 0), -4), 4) AS embi_z,

    -- Macro cambios (clipped)
    LEAST(GREATEST(COALESCE(m.dxy_change_1d, 0), -0.03), 0.03) AS dxy_change_1d,
    LEAST(GREATEST(COALESCE(m.brent_change_1d, 0), -0.10), 0.10) AS brent_change_1d,
    LEAST(GREATEST(COALESCE(m.usdmxn_ret_1h, 0), -0.10), 0.10) AS usdmxn_ret_1h,

    -- Rate spread
    COALESCE(m.rate_spread, 0) AS rate_spread,

    -- NOTA: hour_sin y hour_cos ELIMINADOS en V14 (bajo valor predictivo para FX)
    -- Los 13 features son: log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
    -- dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread, usdmxn_ret_1h
    -- RSI, ATR, ADX se calculan en Python service (no directamente en SQL)

    -- Metadata útil
    DATE(owr.timestamp AT TIME ZONE 'America/Bogota') AS trading_date,
    EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') AS hour_cot,
    EXTRACT(DOW FROM owr.timestamp AT TIME ZONE 'America/Bogota') AS day_of_week

FROM ohlcv_with_returns owr
LEFT JOIN macro_processed m
    ON DATE(owr.timestamp AT TIME ZONE 'America/Bogota') = m.date
WHERE
    -- Solo horario de mercado (8:00-12:55 COT = 13:00-17:55 UTC)
    EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') >= 8
    AND (
        EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') < 13
        OR (EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') = 12
            AND EXTRACT(MINUTE FROM owr.timestamp AT TIME ZONE 'America/Bogota') <= 55)
    )
    -- Solo días de semana
    AND EXTRACT(DOW FROM owr.timestamp AT TIME ZONE 'America/Bogota') BETWEEN 1 AND 5
ORDER BY owr.timestamp DESC;

-- Índice único para REFRESH CONCURRENTLY
CREATE UNIQUE INDEX idx_inf_features_ts ON inference_features_5m (timestamp);

-- Comentarios
COMMENT ON MATERIALIZED VIEW inference_features_5m IS
    'Vista materializada con features para inferencia PPO. Refrescar cada 5 minutos. Definido por config/feature_config.json';


-- =====================================================================
-- TABLA DE LOG DE INFERENCIAS
-- =====================================================================

DROP TABLE IF EXISTS fact_rl_inference_log CASCADE;

CREATE TABLE fact_rl_inference_log (
    id              BIGSERIAL PRIMARY KEY,
    timestamp_utc   TIMESTAMPTZ NOT NULL,
    session_date    DATE NOT NULL,
    bar_number      INTEGER NOT NULL,

    -- Precio y retorno
    close_price     NUMERIC(12, 4),
    raw_return_5m   NUMERIC(10, 8),

    -- Observación (13 features)
    observation     JSONB,

    -- Salida del modelo
    raw_action      NUMERIC(8, 6),       -- [-1, 1]
    position        NUMERIC(8, 6),       -- Después de threshold
    confidence      NUMERIC(5, 4),       -- |raw_action|

    -- Portfolio
    equity_before   NUMERIC(20, 4),
    equity_after    NUMERIC(20, 4),
    pnl_bar         NUMERIC(20, 4),

    -- Metadata
    model_id        VARCHAR(100) NOT NULL,
    latency_ms      INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT valid_bar_number CHECK (bar_number BETWEEN 1 AND 60),
    CONSTRAINT valid_position CHECK (position BETWEEN -1 AND 1)
);

-- Índices
CREATE INDEX idx_inference_log_ts ON fact_rl_inference_log (timestamp_utc DESC);
CREATE INDEX idx_inference_log_session ON fact_rl_inference_log (session_date, bar_number);
CREATE INDEX idx_inference_log_model ON fact_rl_inference_log (model_id);

-- Hypertable para TimescaleDB
SELECT create_hypertable('fact_rl_inference_log', 'timestamp_utc', if_not_exists => TRUE);


-- =====================================================================
-- FUNCIONES DE UTILIDAD
-- =====================================================================

-- Función para verificar si es horario de mercado
CREATE OR REPLACE FUNCTION is_market_open()
RETURNS BOOLEAN AS $$
DECLARE
    now_cot TIMESTAMP;
    hour_cot INTEGER;
    minute_cot INTEGER;
    dow_cot INTEGER;
BEGIN
    now_cot := NOW() AT TIME ZONE 'America/Bogota';
    hour_cot := EXTRACT(HOUR FROM now_cot);
    minute_cot := EXTRACT(MINUTE FROM now_cot);
    dow_cot := EXTRACT(DOW FROM now_cot);

    -- Lunes a Viernes (1-5), 8:00-12:55 COT
    RETURN dow_cot BETWEEN 1 AND 5
           AND (
               (hour_cot >= 8 AND hour_cot < 12)
               OR (hour_cot = 12 AND minute_cot <= 55)
           );
END;
$$ LANGUAGE plpgsql;


-- Función para obtener número de barra del día (1-60)
CREATE OR REPLACE FUNCTION get_bar_number(ts TIMESTAMPTZ DEFAULT NOW())
RETURNS INTEGER AS $$
DECLARE
    ts_cot TIMESTAMP;
    minutes_since_open INTEGER;
BEGIN
    ts_cot := ts AT TIME ZONE 'America/Bogota';
    minutes_since_open := (EXTRACT(HOUR FROM ts_cot) - 8) * 60 + EXTRACT(MINUTE FROM ts_cot);
    RETURN GREATEST(1, LEAST(60, (minutes_since_open / 5) + 1));
END;
$$ LANGUAGE plpgsql;


-- Función para refrescar features (llamar desde DAG)
CREATE OR REPLACE FUNCTION refresh_inference_features()
RETURNS TEXT AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY inference_features_5m;
    RETURN 'Refreshed inference_features_5m at ' || NOW()::TEXT;
END;
$$ LANGUAGE plpgsql;


-- Función para verificar si hoy es festivo Colombia
CREATE OR REPLACE FUNCTION is_colombia_holiday(check_date DATE DEFAULT CURRENT_DATE)
RETURNS BOOLEAN AS $$
BEGIN
    -- Festivos 2025 Colombia
    RETURN check_date IN (
        '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
        '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-04',
        '2025-07-20', '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03',
        '2025-11-17', '2025-11-27', '2025-12-08', '2025-12-25'
    );
END;
$$ LANGUAGE plpgsql;


-- Función para determinar si debe ejecutarse inferencia
CREATE OR REPLACE FUNCTION should_run_inference()
RETURNS TABLE (should_run BOOLEAN, reason TEXT) AS $$
DECLARE
    now_cot TIMESTAMP;
BEGIN
    now_cot := NOW() AT TIME ZONE 'America/Bogota';

    -- Verificar fin de semana
    IF EXTRACT(DOW FROM now_cot) IN (0, 6) THEN
        RETURN QUERY SELECT FALSE, 'Weekend - market closed'::TEXT;
        RETURN;
    END IF;

    -- Verificar festivo
    IF is_colombia_holiday(now_cot::DATE) THEN
        RETURN QUERY SELECT FALSE, 'Colombia holiday - market closed'::TEXT;
        RETURN;
    END IF;

    -- Verificar horario de mercado
    IF NOT is_market_open() THEN
        RETURN QUERY SELECT FALSE, 'Outside market hours (8:00-12:55 COT)'::TEXT;
        RETURN;
    END IF;

    RETURN QUERY SELECT TRUE, 'Market open - proceed with inference'::TEXT;
END;
$$ LANGUAGE plpgsql;


-- =====================================================================
-- GRANTS
-- =====================================================================

-- Crear roles si no existen
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_app') THEN
        CREATE ROLE trading_app WITH LOGIN PASSWORD 'trading_secure_password';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
        CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow_secure_password';
    END IF;
END$$;

-- Permisos lectura
GRANT SELECT ON usdcop_m5_ohlcv TO trading_app;
GRANT SELECT ON macro_indicators_daily TO trading_app;
GRANT SELECT ON inference_features_5m TO trading_app;
GRANT SELECT ON fact_rl_inference_log TO trading_app;

-- Permisos escritura para airflow
GRANT ALL ON macro_indicators_daily TO airflow;
GRANT ALL ON fact_rl_inference_log TO airflow;
GRANT EXECUTE ON FUNCTION refresh_inference_features() TO airflow;
GRANT EXECUTE ON FUNCTION should_run_inference() TO airflow;


-- =====================================================================
-- VERIFICACIÓN
-- =====================================================================

SELECT 'Schema created successfully' AS status, NOW() AS created_at;

-- Verificar tablas creadas
SELECT
    table_name,
    table_type
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('macro_indicators_daily', 'fact_rl_inference_log')
ORDER BY table_name;

-- Verificar vista materializada
SELECT
    matviewname,
    ispopulated
FROM pg_matviews
WHERE matviewname = 'inference_features_5m';

-- Verificar funciones
SELECT
    proname AS function_name,
    pg_get_function_result(oid) AS return_type
FROM pg_proc
WHERE proname IN ('is_market_open', 'get_bar_number', 'refresh_inference_features', 'should_run_inference')
ORDER BY proname;
