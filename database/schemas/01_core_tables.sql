-- ==============================================================================
-- USD/COP RL Trading System - Core Tables Schema
-- Version: 3.1 (Migration V14)
-- ==============================================================================
-- REFERENCIA: ARQUITECTURA_INTEGRAL_V3.md (Sección 4, líneas 297-389)
-- REFERENCIA: MAPEO_MIGRACION_BIDIRECCIONAL.md (Parte 2.2, líneas 334-346)
-- ==============================================================================
-- Este script define las 3 tablas core del sistema de inferencia:
--   1. usdcop_m5_ohlcv      (YA EXISTE - solo documentada)
--   2. macro_indicators_daily (CREAR - 37 variables macro)
--   3. dw.fact_rl_inference (CREAR - log de inferencias en schema DW)
-- ==============================================================================

-- ============================================================================
-- EXTENSIONES REQUERIDAS
-- ============================================================================
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- TABLA 1: usdcop_m5_ohlcv (YA EXISTE)
-- ============================================================================
-- UBICACIÓN ACTUAL: init-scripts/01-essential-usdcop-init.sql
-- TIPO: TimescaleDB Hypertable
-- GRANULARIDAD: 5 minutos
-- HORARIO: Lunes-Viernes, 08:00-12:55 COT (13:00-17:55 UTC)
--
-- Estructura existente:
--   time           TIMESTAMPTZ PRIMARY KEY
--   symbol         TEXT DEFAULT 'USD/COP'
--   open           DECIMAL(12,6)
--   high           DECIMAL(12,6)
--   low            DECIMAL(12,6)
--   close          DECIMAL(12,6)
--   volume         BIGINT
--   source         TEXT DEFAULT 'twelvedata'
--   created_at     TIMESTAMPTZ
--   updated_at     TIMESTAMPTZ
--
-- NOTA: No modificar esta tabla - es SSOT para datos de precios
-- ============================================================================

-- ============================================================================
-- TABLA 2: macro_indicators_daily
-- ============================================================================
-- DESCRIPCIÓN: Indicadores macroeconómicos diarios para features del modelo
-- ACTUALIZACIÓN: 3 veces/día (07:55, 10:30, 12:00 COT)
-- FUENTES: TwelveData, FRED, BCRP, Investing.com
-- REFERENCIA: ARQUITECTURA sección 4 y 21 (37 columnas del diccionario macro)
-- ============================================================================

DROP TABLE IF EXISTS macro_indicators_daily CASCADE;

CREATE TABLE macro_indicators_daily (
    date DATE PRIMARY KEY,

    -- =========================================================================
    -- GRUPO 1: DXY (Dollar Index)
    -- =========================================================================
    dxy                   NUMERIC(10,4),    -- US Dollar Index - Feature: dxy_z, dxy_change_1d

    -- =========================================================================
    -- GRUPO 2: VIX (Volatility Index)
    -- =========================================================================
    vix                   NUMERIC(10,4),    -- CBOE VIX - Feature: vix_z

    -- =========================================================================
    -- GRUPO 3: EMBI (Country Risk)
    -- =========================================================================
    embi                  NUMERIC(10,4),    -- EMBI Colombia - Feature: embi_z

    -- =========================================================================
    -- GRUPO 4: BRENT (Oil Price)
    -- =========================================================================
    brent                 NUMERIC(10,4),    -- Brent Crude Oil - Feature: brent_change_1d

    -- =========================================================================
    -- GRUPO 5: WTI (Oil Price)
    -- =========================================================================
    wti                   NUMERIC(10,4),    -- WTI Crude Oil

    -- =========================================================================
    -- GRUPO 6: GOLD (Precious Metals)
    -- =========================================================================
    gold                  NUMERIC(10,4),    -- Gold Futures (XAU/USD)

    -- =========================================================================
    -- GRUPO 7: COFFEE (Commodities)
    -- =========================================================================
    coffee                NUMERIC(10,4),    -- Coffee Arabica

    -- =========================================================================
    -- GRUPO 8: FED_FUNDS (Policy Rate)
    -- =========================================================================
    fed_funds             NUMERIC(8,4),     -- Fed Funds Rate

    -- =========================================================================
    -- GRUPO 9: TREASURY_2Y (Fixed Income)
    -- =========================================================================
    treasury_2y           NUMERIC(8,4),     -- UST 2Y (DGS2) - Feature: rate_spread

    -- =========================================================================
    -- GRUPO 10: TREASURY_10Y (Fixed Income)
    -- =========================================================================
    treasury_10y          NUMERIC(8,4),     -- UST 10Y (DGS10) - Feature: rate_spread

    -- =========================================================================
    -- GRUPO 11: PRIME_RATE (Policy Rate)
    -- =========================================================================
    prime_rate            NUMERIC(8,4),     -- Prime Rate USA

    -- =========================================================================
    -- GRUPO 12: TPM_COLOMBIA (Policy Rate)
    -- =========================================================================
    tpm_colombia          NUMERIC(8,4),     -- Tasa Política Monetaria BanRep

    -- =========================================================================
    -- GRUPO 13: IBR_OVERNIGHT (Money Market)
    -- =========================================================================
    ibr_overnight         NUMERIC(8,4),     -- IBR Colombia

    -- =========================================================================
    -- GRUPO 14: USDMXN (Exchange Rate)
    -- =========================================================================
    usdmxn                NUMERIC(10,4),    -- USD/MXN - Feature: usdmxn_ret_1h

    -- =========================================================================
    -- GRUPO 15: USDCLP (Exchange Rate)
    -- =========================================================================
    usdclp                NUMERIC(10,4),    -- USD/CLP (peer currency)

    -- =========================================================================
    -- GRUPO 16: BOND_YIELD5Y_COL (Fixed Income)
    -- =========================================================================
    bond_yield5y_col      NUMERIC(8,4),     -- Bono Colombia 5Y

    -- =========================================================================
    -- GRUPO 17: BOND_YIELD10Y_COL (Fixed Income)
    -- =========================================================================
    bond_yield10y_col     NUMERIC(8,4),     -- Bono Colombia 10Y

    -- =========================================================================
    -- GRUPO 18: COLCAP (Equity Index)
    -- =========================================================================
    colcap                NUMERIC(10,4),    -- Índice COLCAP

    -- =========================================================================
    -- GRUPO 19: ITCR (Exchange Rate Index)
    -- =========================================================================
    itcr                  NUMERIC(10,4),    -- Índice Tasa Cambio Real

    -- =========================================================================
    -- GRUPO 20: CCI_COLOMBIA (Sentiment)
    -- =========================================================================
    cci_colombia          NUMERIC(10,4),    -- Índice Confianza Consumidor

    -- =========================================================================
    -- GRUPO 21: ICI_COLOMBIA (Sentiment)
    -- =========================================================================
    ici_colombia          NUMERIC(10,4),    -- Índice Confianza Industrial

    -- =========================================================================
    -- GRUPO 22: IPC_COLOMBIA (Inflation - mensual)
    -- =========================================================================
    ipc_colombia          NUMERIC(10,4),    -- IPC Colombia (mensual)

    -- =========================================================================
    -- GRUPO 23: CPI_USA (Inflation - mensual)
    -- =========================================================================
    cpi_usa               NUMERIC(10,4),    -- CPI USA (mensual)

    -- =========================================================================
    -- GRUPO 24: PCE_USA (Inflation - mensual)
    -- =========================================================================
    pce_usa               NUMERIC(10,4),    -- PCE USA (mensual)

    -- =========================================================================
    -- GRUPO 25: EXPORTS_COL (Trade - mensual)
    -- =========================================================================
    exports_col           NUMERIC(14,2),    -- Exportaciones Colombia

    -- =========================================================================
    -- GRUPO 26: IMPORTS_COL (Trade - mensual)
    -- =========================================================================
    imports_col           NUMERIC(14,2),    -- Importaciones Colombia

    -- =========================================================================
    -- GRUPO 27: TERMS_OF_TRADE (Trade)
    -- =========================================================================
    terms_of_trade        NUMERIC(10,4),    -- Términos de Intercambio

    -- =========================================================================
    -- GRUPO 28: IED_INFLOW (Balance of Payments - trimestral)
    -- =========================================================================
    ied_inflow            NUMERIC(14,2),    -- IED Entrante

    -- =========================================================================
    -- GRUPO 29: IED_OUTFLOW (Balance of Payments - trimestral)
    -- =========================================================================
    ied_outflow           NUMERIC(14,2),    -- IED Saliente

    -- =========================================================================
    -- GRUPO 30: CURRENT_ACCOUNT (Balance of Payments - trimestral)
    -- =========================================================================
    current_account       NUMERIC(14,2),    -- Cuenta Corriente BP

    -- =========================================================================
    -- GRUPO 31: RESERVES_INTL (Balance of Payments)
    -- =========================================================================
    reserves_intl         NUMERIC(14,2),    -- Reservas Internacionales

    -- =========================================================================
    -- GRUPO 32: UNEMPLOYMENT_USA (Labor - mensual)
    -- =========================================================================
    unemployment_usa      NUMERIC(8,4),     -- Desempleo USA

    -- =========================================================================
    -- GRUPO 33: INDUSTRIAL_PROD_USA (Production - mensual)
    -- =========================================================================
    industrial_prod_usa   NUMERIC(10,4),    -- Producción Industrial USA

    -- =========================================================================
    -- GRUPO 34: M2_SUPPLY_USA (Monetary - mensual)
    -- =========================================================================
    m2_supply_usa         NUMERIC(14,2),    -- M2 USA

    -- =========================================================================
    -- GRUPO 35: CONSUMER_SENTIMENT (Sentiment - mensual)
    -- =========================================================================
    consumer_sentiment    NUMERIC(10,4),    -- Michigan Sentiment

    -- =========================================================================
    -- GRUPO 36: GDP_USA (GDP - trimestral)
    -- =========================================================================
    gdp_usa               NUMERIC(14,2),    -- GDP USA Real

    -- =========================================================================
    -- GRUPO 37: USDCOP_SPOT (Reference Rate)
    -- =========================================================================
    usdcop_spot           NUMERIC(10,4),    -- USD/COP Spot (cierre diario)

    -- =========================================================================
    -- METADATA
    -- =========================================================================
    source                VARCHAR(100),                      -- Fuente principal del update
    is_complete           BOOLEAN DEFAULT FALSE,             -- TRUE si todos los campos críticos están llenos
    created_at            TIMESTAMPTZ DEFAULT NOW(),
    updated_at            TIMESTAMPTZ DEFAULT NOW()
);

-- Índices para macro_indicators_daily
CREATE INDEX idx_macro_date ON macro_indicators_daily (date DESC);
CREATE INDEX idx_macro_complete ON macro_indicators_daily (is_complete, date DESC);

-- Índice compuesto para features del modelo (los 7 usados en inferencia)
CREATE INDEX idx_macro_model_features
    ON macro_indicators_daily(date, dxy, vix, embi, brent, usdmxn, treasury_10y, treasury_2y);

-- Trigger para actualizar updated_at
CREATE OR REPLACE FUNCTION update_macro_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_macro_updated_at ON macro_indicators_daily;
CREATE TRIGGER trg_macro_updated_at
    BEFORE UPDATE ON macro_indicators_daily
    FOR EACH ROW
    EXECUTE FUNCTION update_macro_timestamp();

-- Comentarios de documentación
COMMENT ON TABLE macro_indicators_daily IS
'SSOT v3.1: Indicadores macroeconómicos diarios para features del modelo PPO.
37 variables macro actualizadas 3x/día (07:55, 10:30, 12:00 COT).
FEATURES CALCULADOS: dxy_z, vix_z, embi_z, dxy_change_1d, brent_change_1d, rate_spread.
FEATURES EN PYTHON: usdmxn_ret_1h (requiere 12 periodos de lookback).
REF: ARQUITECTURA_INTEGRAL_V3.md sección 4 y 21.';

COMMENT ON COLUMN macro_indicators_daily.dxy IS 'US Dollar Index - Strong positive correlation (~+0.8) with USD/COP';
COMMENT ON COLUMN macro_indicators_daily.vix IS 'CBOE Volatility Index - Flight-to-safety indicator affecting EM currencies';
COMMENT ON COLUMN macro_indicators_daily.embi IS 'EMBI Colombia - Country risk premium spread over US Treasuries';
COMMENT ON COLUMN macro_indicators_daily.brent IS 'Brent Crude Oil - Key for Colombia as oil exporter';
COMMENT ON COLUMN macro_indicators_daily.usdmxn IS 'USD/MXN exchange rate - High correlation with USD/COP (~0.7)';
COMMENT ON COLUMN macro_indicators_daily.is_complete IS 'TRUE when all critical fields (dxy, vix, embi, brent, treasury_10y, treasury_2y) are populated';


-- ============================================================================
-- TABLA 3: dw.fact_rl_inference (Schema DW)
-- ============================================================================
-- DESCRIPCIÓN: Log de cada inferencia del modelo PPO en tiempo real
-- GRANULARIDAD: Una fila por inferencia (cada 5 minutos durante mercado)
-- TIPO: TimescaleDB Hypertable (particionado por día)
-- REFERENCIA: ARQUITECTURA sección 11.2 y init-scripts/11-realtime-inference-tables.sql
-- ============================================================================

-- Crear schema DW si no existe
CREATE SCHEMA IF NOT EXISTS dw;

DROP TABLE IF EXISTS dw.fact_rl_inference CASCADE;

CREATE TABLE dw.fact_rl_inference (
    inference_id          BIGSERIAL PRIMARY KEY,

    -- =========================================================================
    -- TIMING
    -- =========================================================================
    timestamp_utc         TIMESTAMPTZ NOT NULL,              -- Partición hypertable
    timestamp_cot         TIMESTAMPTZ NOT NULL,              -- Hora Colombia

    -- =========================================================================
    -- MODEL IDENTIFICATION
    -- =========================================================================
    model_id              VARCHAR(100) NOT NULL,             -- ej: "ppo_usdcop_v11_fold0"
    model_version         VARCHAR(50),                       -- ej: "11.2"
    fold_id               INT,                               -- Cross-validation fold (0-4)

    -- =========================================================================
    -- OBSERVATION VECTOR (13 features normalizados del modelo V14)
    -- =========================================================================
    -- Features en orden del modelo PPO v11/v14:
    -- [log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,
    --  dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d,
    --  rate_spread, usdmxn_ret_1h]
    -- + 2 features de estado: [position, time_normalized]
    -- TOTAL: 15 dimensiones
    -- NOTA: hour_sin/hour_cos ELIMINADOS en V14 (MAPEO línea 42)
    -- =========================================================================

    -- Feature 1: log_ret_5m
    log_ret_5m            FLOAT,                             -- Logarithmic return 5min

    -- Feature 2: log_ret_1h
    log_ret_1h            FLOAT,                             -- Logarithmic return 1h

    -- Feature 3: log_ret_4h
    log_ret_4h            FLOAT,                             -- Logarithmic return 4h

    -- Feature 4: rsi_9 (calculado en Python)
    rsi_9                 FLOAT,                             -- RSI period 9

    -- Feature 5: atr_pct (calculado en Python)
    atr_pct               FLOAT,                             -- ATR% period 10

    -- Feature 6: adx_14 (calculado en Python)
    adx_14                FLOAT,                             -- ADX period 14

    -- Feature 7: dxy_z
    dxy_z                 FLOAT,                             -- DXY z-score

    -- Feature 8: dxy_change_1d
    dxy_change_1d         FLOAT,                             -- DXY daily change

    -- Feature 9: vix_z
    vix_z                 FLOAT,                             -- VIX z-score

    -- Feature 10: embi_z
    embi_z                FLOAT,                             -- EMBI z-score

    -- Feature 11: brent_change_1d
    brent_change_1d       FLOAT,                             -- Brent daily change

    -- Feature 12: rate_spread
    rate_spread           FLOAT,                             -- Treasury 10Y - 2Y

    -- Feature 13: usdmxn_ret_1h (calculado en Python)
    usdmxn_ret_1h         FLOAT,                             -- USDMXN hourly return

    -- State features (2 adicionales)
    position              FLOAT,                             -- Current position [-1, 1]
    time_normalized       FLOAT,                             -- Normalized time in session

    -- =========================================================================
    -- MODEL OUTPUT
    -- =========================================================================
    action_raw            FLOAT NOT NULL,                    -- [-1, 1] output directo
    action_discretized    VARCHAR(10) NOT NULL
        CHECK (action_discretized IN ('LONG', 'SHORT', 'HOLD')),
    confidence            FLOAT CHECK (confidence >= 0 AND confidence <= 1),

    -- Q-values (si disponibles del modelo)
    q_values              FLOAT[],                           -- [q_long, q_short, q_hold]

    -- =========================================================================
    -- MARKET CONTEXT
    -- =========================================================================
    symbol                VARCHAR(20) DEFAULT 'USD/COP',
    close_price           DECIMAL(12,6) NOT NULL,            -- Precio OHLCV
    raw_return_5m         FLOAT,                             -- Retorno 5min sin normalizar
    spread_bps            DECIMAL(8,4),                      -- Spread en basis points

    -- =========================================================================
    -- PORTFOLIO STATE BEFORE ACTION
    -- =========================================================================
    position_before       FLOAT NOT NULL
        CHECK (position_before >= -1 AND position_before <= 1),
    portfolio_value_before DECIMAL(15,2),
    log_portfolio_before  FLOAT,

    -- =========================================================================
    -- PORTFOLIO STATE AFTER ACTION
    -- =========================================================================
    position_after        FLOAT NOT NULL
        CHECK (position_after >= -1 AND position_after <= 1),
    portfolio_value_after DECIMAL(15,2),
    log_portfolio_after   FLOAT,

    -- =========================================================================
    -- TRANSACTION COSTS
    -- =========================================================================
    position_change       FLOAT,                             -- position_after - position_before
    transaction_cost_bps  DECIMAL(8,4),                      -- Costo en bps
    transaction_cost_usd  DECIMAL(12,4),                     -- Costo en USD

    -- =========================================================================
    -- PERFORMANCE
    -- =========================================================================
    reward                FLOAT,                             -- Reward del step
    cumulative_reward     FLOAT,                             -- Acumulado del día

    -- =========================================================================
    -- METADATA
    -- =========================================================================
    latency_ms            INT,                               -- Latencia de inferencia
    inference_source      VARCHAR(50) DEFAULT 'airflow',     -- 'airflow', 'api', 'manual'
    dag_run_id            VARCHAR(100),                      -- ID del DAG run de Airflow

    created_at            TIMESTAMPTZ DEFAULT NOW()
);

-- Convertir a hypertable (si TimescaleDB está disponible)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'timescaledb') THEN
        PERFORM create_hypertable(
            'dw.fact_rl_inference',
            'timestamp_utc',
            chunk_time_interval => INTERVAL '1 day',
            if_not_exists => TRUE
        );
    END IF;
END $$;

-- Índices optimizados para fact_rl_inference
CREATE INDEX idx_rl_inference_model_time
    ON dw.fact_rl_inference(model_id, timestamp_utc DESC);

CREATE INDEX idx_rl_inference_action
    ON dw.fact_rl_inference(action_discretized);

CREATE INDEX idx_rl_inference_symbol_time
    ON dw.fact_rl_inference(symbol, timestamp_utc DESC);

CREATE INDEX idx_rl_inference_dag_run
    ON dw.fact_rl_inference(dag_run_id);

-- Comentario de documentación
COMMENT ON TABLE dw.fact_rl_inference IS
'Log de inferencias del modelo PPO v11/v14 en tiempo real.
Cada fila = una inferencia (cada 5 min durante mercado 08:00-12:55 COT).
13 features de mercado + 2 features de estado = 15 dimensiones totales.
NOTA: hour_sin/hour_cos ELIMINADOS en V14 (MAPEO línea 42).
REF: ARQUITECTURA_INTEGRAL_V3.md sección 11.2.';


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function: Check if market is currently open
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

    -- Monday to Friday (1-5), 8:00-12:55 COT
    RETURN dow_cot BETWEEN 1 AND 5
           AND (
               (hour_cot >= 8 AND hour_cot < 12)
               OR (hour_cot = 12 AND minute_cot <= 55)
           );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION is_market_open() IS
    'Returns TRUE if current time is within trading hours (Mon-Fri 8:00-12:55 COT)';


-- Function: Get bar number for a given timestamp (1-60)
CREATE OR REPLACE FUNCTION get_bar_number(ts TIMESTAMPTZ DEFAULT NOW())
RETURNS INTEGER AS $$
DECLARE
    ts_cot TIMESTAMP;
    minutes_since_open INTEGER;
BEGIN
    ts_cot := ts AT TIME ZONE 'America/Bogota';
    minutes_since_open := (EXTRACT(HOUR FROM ts_cot) - 8) * 60 + EXTRACT(MINUTE FROM ts_cot);

    -- Return 1-60, clamped to valid range
    RETURN GREATEST(1, LEAST(60, (minutes_since_open / 5) + 1));
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_bar_number(TIMESTAMPTZ) IS
    'Calculates bar number (1-60) for a given timestamp in COT timezone';


-- Function: Check if date is Colombia holiday
CREATE OR REPLACE FUNCTION is_colombia_holiday(check_date DATE DEFAULT CURRENT_DATE)
RETURNS BOOLEAN AS $$
BEGIN
    -- Colombia holidays 2025
    RETURN check_date IN (
        '2025-01-01', '2025-01-06', '2025-03-24', '2025-04-17', '2025-04-18',
        '2025-05-01', '2025-06-02', '2025-06-23', '2025-06-30', '2025-07-04',
        '2025-07-20', '2025-08-07', '2025-08-18', '2025-10-13', '2025-11-03',
        '2025-11-17', '2025-11-27', '2025-12-08', '2025-12-25'
    );
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION is_colombia_holiday(DATE) IS
    'Returns TRUE if date is a Colombia public holiday (2025 calendar)';


-- Function: Should run inference (combines all checks)
CREATE OR REPLACE FUNCTION should_run_inference()
RETURNS TABLE (should_run BOOLEAN, reason TEXT) AS $$
DECLARE
    now_cot TIMESTAMP;
BEGIN
    now_cot := NOW() AT TIME ZONE 'America/Bogota';

    -- Check weekend
    IF EXTRACT(DOW FROM now_cot) IN (0, 6) THEN
        RETURN QUERY SELECT FALSE, 'Weekend - market closed'::TEXT;
        RETURN;
    END IF;

    -- Check holiday
    IF is_colombia_holiday(now_cot::DATE) THEN
        RETURN QUERY SELECT FALSE, 'Colombia holiday - market closed'::TEXT;
        RETURN;
    END IF;

    -- Check market hours
    IF NOT is_market_open() THEN
        RETURN QUERY SELECT FALSE, 'Outside market hours (8:00-12:55 COT)'::TEXT;
        RETURN;
    END IF;

    RETURN QUERY SELECT TRUE, 'Market open - proceed with inference'::TEXT;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION should_run_inference() IS
    'Determines if inference should run based on market hours, weekends, and holidays';


-- ============================================================================
-- GRANTS AND PERMISSIONS
-- ============================================================================

-- Create roles if they don't exist
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'trading_app') THEN
        CREATE ROLE trading_app WITH LOGIN PASSWORD 'trading_secure_password';
    END IF;
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'airflow') THEN
        CREATE ROLE airflow WITH LOGIN PASSWORD 'airflow_secure_password';
    END IF;
END$$;

-- Grant read permissions to trading_app
GRANT SELECT ON macro_indicators_daily TO trading_app;
GRANT SELECT ON dw.fact_rl_inference TO trading_app;

-- Grant read/write permissions to airflow
GRANT ALL ON macro_indicators_daily TO airflow;
GRANT ALL ON dw.fact_rl_inference TO airflow;
GRANT USAGE ON SCHEMA dw TO airflow;
GRANT EXECUTE ON FUNCTION is_market_open() TO airflow;
GRANT EXECUTE ON FUNCTION get_bar_number(TIMESTAMPTZ) TO airflow;
GRANT EXECUTE ON FUNCTION is_colombia_holiday(DATE) TO airflow;
GRANT EXECUTE ON FUNCTION should_run_inference() TO airflow;


-- ============================================================================
-- VERIFICATION
-- ============================================================================

SELECT 'Core tables schema created successfully' AS status, NOW() AS created_at;

-- Verify tables were created
SELECT
    table_schema,
    table_name,
    table_type
FROM information_schema.tables
WHERE (table_schema = 'public' AND table_name = 'macro_indicators_daily')
   OR (table_schema = 'dw' AND table_name = 'fact_rl_inference')
ORDER BY table_schema, table_name;

-- Verify functions were created
SELECT
    proname AS function_name,
    pg_get_function_result(oid) AS return_type
FROM pg_proc
WHERE proname IN ('is_market_open', 'get_bar_number', 'is_colombia_holiday', 'should_run_inference')
ORDER BY proname;

-- Count macro columns (should be 37 + 4 metadata = 41)
SELECT COUNT(*) as column_count
FROM information_schema.columns
WHERE table_name = 'macro_indicators_daily'
  AND table_schema = 'public';

-- Count fact_rl_inference feature columns (should be 13 features + 2 state + other fields)
SELECT COUNT(*) as column_count
FROM information_schema.columns
WHERE table_name = 'fact_rl_inference'
  AND table_schema = 'dw';
