-- ========================================
-- USDCOP TRADING SYSTEM - P0 CRITICAL FIXES
-- Fecha: 2026-01-08
-- ========================================

-- ========================================
-- FIX 1: CORREGIR THRESHOLD MISMATCH
-- Cambiar de 0.30 a 0.10 para alinear con entrenamiento
-- ========================================

-- Verificar valores actuales
SELECT model_id, threshold_long, threshold_short FROM config.models;

-- Aplicar fix
UPDATE config.models
SET
    threshold_long = 0.10,
    threshold_short = -0.10,
    updated_at = NOW()
WHERE model_id IN ('ppo_v1', 'sac_v19_baseline', 'td3_v19_baseline', 'a2c_v19_baseline');

-- Verificar cambio
SELECT model_id, threshold_long, threshold_short, updated_at FROM config.models;

-- ========================================
-- FIX 2: VERIFICAR TRADING_STATE TIENE DATOS
-- ========================================

-- Ver estado actual
SELECT * FROM trading_state;

-- Si está vacío, inicializar con valores por defecto
INSERT INTO trading_state (model_id, position, entry_price, equity, realized_pnl, trade_count, wins, losses, last_updated)
SELECT
    model_id,
    0 as position,
    0.0 as entry_price,
    10000.0 as equity,
    0.0 as realized_pnl,
    0 as trade_count,
    0 as wins,
    0 as losses,
    NOW() as last_updated
FROM config.models
WHERE enabled = true
ON CONFLICT (model_id) DO NOTHING;

-- ========================================
-- FIX 3: CREAR TABLA FACT_FEATURES_5M SI NO EXISTE
-- (Para drift monitor)
-- ========================================

CREATE TABLE IF NOT EXISTS dw.fact_features_5m (
    timestamp TIMESTAMPTZ NOT NULL,
    -- V19 Features
    log_ret_5m DOUBLE PRECISION,
    log_ret_1h DOUBLE PRECISION,
    rsi_9 DOUBLE PRECISION,
    macd_hist DOUBLE PRECISION,
    bb_width DOUBLE PRECISION,
    vol_ratio DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    hour_sin DOUBLE PRECISION,
    hour_cos DOUBLE PRECISION,
    day_of_week_sin DOUBLE PRECISION,
    day_of_week_cos DOUBLE PRECISION,
    dxy_z DOUBLE PRECISION,
    vix_z DOUBLE PRECISION,
    -- Legacy features (for backwards compatibility)
    returns_5m DOUBLE PRECISION,
    returns_15m DOUBLE PRECISION,
    returns_1h DOUBLE PRECISION,
    volatility_5m DOUBLE PRECISION,
    volatility_15m DOUBLE PRECISION,
    rsi_14 DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    sma_ratio_20 DOUBLE PRECISION,
    ema_ratio_12 DOUBLE PRECISION,
    volume_ratio DOUBLE PRECISION,
    atr_14 DOUBLE PRECISION,
    PRIMARY KEY (timestamp)
);

-- Crear índice temporal
CREATE INDEX IF NOT EXISTS idx_fact_features_5m_time
ON dw.fact_features_5m (timestamp DESC);

-- ========================================
-- FIX 4: VERIFICAR MACRO DATA GAPS
-- ========================================

-- Ver últimos registros de macro
SELECT
    fecha,
    fxrt_index_dxy_usa_d_dxy as dxy,
    volt_vix_usa_d_vix as vix
FROM macro_indicators_daily
WHERE fecha > CURRENT_DATE - INTERVAL '14 days'
ORDER BY fecha DESC;

-- Contar NULLs por día
SELECT
    fecha,
    COUNT(*) FILTER (WHERE fxrt_index_dxy_usa_d_dxy IS NULL) as dxy_nulls,
    COUNT(*) FILTER (WHERE volt_vix_usa_d_vix IS NULL) as vix_nulls
FROM macro_indicators_daily
WHERE fecha > CURRENT_DATE - INTERVAL '14 days'
GROUP BY fecha
ORDER BY fecha DESC;

-- ========================================
-- VALIDACIÓN POST-FIX
-- ========================================

-- Confirmar thresholds corregidos
SELECT
    model_id,
    threshold_long,
    threshold_short,
    CASE
        WHEN threshold_long = 0.10 THEN 'CORRECTO'
        ELSE 'INCORRECTO'
    END as status
FROM config.models;

-- Verificar trading_state inicializado
SELECT model_id, equity, trade_count FROM trading_state;

-- Verificar tabla de features existe
SELECT EXISTS (
    SELECT FROM information_schema.tables
    WHERE table_schema = 'dw'
    AND table_name = 'fact_features_5m'
) as features_table_exists;

COMMIT;
