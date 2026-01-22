-- =====================================================
-- USDCOP Trading System - Inference Features (CORRECTED v2.0)
-- Versión: 2.0 - Nombres de columna corregidos
-- =====================================================
--
-- CRITICAL FIX: This script corrects column name mismatches between
-- the original 03-inference-features-views.sql and macro_indicators_daily table.
--
-- Original script used: date, dxy, vix, embi, brent, treasury_2y, treasury_10y
-- Actual columns are:   fecha, fxrt_index_dxy_usa_d_dxy, volt_vix_usa_d_vix, etc.
--
-- ARCHITECTURE:
-- - inference_features_5m: TABLE for Python L1 DAG (SSOT) to insert features
-- - python_features_5m: TABLE for Python-calculated RSI, ATR, ADX, USDMXN
-- - inference_features_5m_sql: MATERIALIZED VIEW (SQL fallback)
-- - inference_features_complete: VIEW that joins Python + SQL with fallback logic
--
-- All stats calculated from training period: 2020-03 to 2025-10
-- =====================================================

-- =============================================================================
-- 1. TABLA para features calculados por Python (L1 DAG - SSOT)
-- =============================================================================
-- This is the PRIMARY source of truth for inference features.
-- The L1 DAG (l1_feature_refresh.py) uses CanonicalFeatureBuilder to calculate
-- and insert features into this table.

CREATE TABLE IF NOT EXISTS inference_features_5m (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    log_ret_5m DOUBLE PRECISION,
    log_ret_1h DOUBLE PRECISION,
    log_ret_4h DOUBLE PRECISION,
    rsi_9 DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    dxy_z DOUBLE PRECISION,
    dxy_change_1d DOUBLE PRECISION,
    vix_z DOUBLE PRECISION,
    embi_z DOUBLE PRECISION,
    brent_change_1d DOUBLE PRECISION,
    rate_spread DOUBLE PRECISION,
    usdmxn_change_1d DOUBLE PRECISION,
    position DOUBLE PRECISION DEFAULT 0.0,
    time_normalized DOUBLE PRECISION,
    builder_version TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inference_features_time
ON inference_features_5m (time DESC);

COMMENT ON TABLE inference_features_5m IS
    'Primary feature store for RL inference. Populated by L1 DAG using CanonicalFeatureBuilder (Python SSOT).';

-- =============================================================================
-- 2. TABLA para features Python (RSI, ATR, ADX, USDMXN)
-- =============================================================================
-- These features require Wilder's EMA calculations which are more accurate in Python.
-- Used as intermediate storage before L1 DAG consolidates into inference_features_5m.

CREATE TABLE IF NOT EXISTS python_features_5m (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    rsi_9 DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    usdmxn_change_1d DOUBLE PRECISION,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_python_features_time
ON python_features_5m (time DESC);

COMMENT ON TABLE python_features_5m IS
    'Python-calculated features (RSI, ATR, ADX with Wilders EMA). Updated by l1_feature_refresh DAG.';

-- =============================================================================
-- 3. VISTA MATERIALIZADA SQL (FALLBACK) - Con nombres CORREGIDOS
-- =============================================================================
-- This is a FALLBACK view that calculates features purely from SQL.
-- It's less accurate than Python for RSI/ATR/ADX but provides basic functionality
-- if the L1 DAG hasn't populated inference_features_5m yet.
--
-- CRITICAL FIX: Uses actual column names from macro_indicators_daily:
-- - fecha (not date)
-- - fxrt_index_dxy_usa_d_dxy (not dxy)
-- - volt_vix_usa_d_vix (not vix)
-- - crsk_spread_embi_col_d_embi (not embi)
-- - comm_oil_brent_glb_d_brent (not brent)
-- - finc_bond_yield2y_usa_d_dgs2 (not treasury_2y)
-- - finc_bond_yield10y_usa_d_ust10y (not treasury_10y)
-- - fxrt_spot_usdmxn_mex_d_usdmxn (not usdmxn)

CREATE MATERIALIZED VIEW IF NOT EXISTS inference_features_5m_sql AS
WITH ohlcv_with_returns AS (
    SELECT
        o.time,
        o.close,
        -- Log returns (clip to [-0.05, 0.05])
        LEAST(GREATEST(
            LN(o.close / NULLIF(LAG(o.close, 1) OVER (ORDER BY o.time), 0)),
            -0.05), 0.05) AS log_ret_5m,
        LEAST(GREATEST(
            LN(o.close / NULLIF(LAG(o.close, 12) OVER (ORDER BY o.time), 0)),
            -0.05), 0.05) AS log_ret_1h,
        LEAST(GREATEST(
            LN(o.close / NULLIF(LAG(o.close, 48) OVER (ORDER BY o.time), 0)),
            -0.05), 0.05) AS log_ret_4h
    FROM usdcop_m5_ohlcv o
    WHERE o.symbol = 'USD/COP'
),
macro_latest AS (
    -- CORRECTED: Use actual column names from macro_indicators_daily
    SELECT DISTINCT ON (fecha)
        fecha AS date,
        fxrt_index_dxy_usa_d_dxy AS dxy,
        volt_vix_usa_d_vix AS vix,
        crsk_spread_embi_col_d_embi AS embi,
        comm_oil_brent_glb_d_brent AS brent,
        finc_bond_yield2y_usa_d_dgs2 AS treasury_2y,
        finc_bond_yield10y_usa_d_ust10y AS treasury_10y,
        fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn
    FROM macro_indicators_daily
    WHERE fxrt_index_dxy_usa_d_dxy IS NOT NULL
    ORDER BY fecha DESC
),
macro_with_changes AS (
    SELECT
        date,
        -- DXY z-score: (dxy - 100.21) / 5.60, clip to [-4, 4]
        -- Stats from training period 2020-03 to 2025-10
        LEAST(GREATEST((dxy - 100.21) / 5.60, -4.0), 4.0) AS dxy_z,
        -- DXY change 1d (clip to [-0.03, 0.03])
        LEAST(GREATEST(
            (dxy - LAG(dxy, 1) OVER (ORDER BY date)) / NULLIF(LAG(dxy, 1) OVER (ORDER BY date), 0),
            -0.03), 0.03) AS dxy_change_1d,
        -- VIX z-score: (vix - 21.16) / 7.89, clip to [-4, 4]
        LEAST(GREATEST((vix - 21.16) / 7.89, -4.0), 4.0) AS vix_z,
        -- EMBI z-score: (embi - 322.01) / 62.68, clip to [-4, 4]
        LEAST(GREATEST((COALESCE(embi, 322.01) - 322.01) / 62.68, -4.0), 4.0) AS embi_z,
        -- Brent change 1d (clip to [-0.10, 0.10])
        LEAST(GREATEST(
            (brent - LAG(brent, 1) OVER (ORDER BY date)) / NULLIF(LAG(brent, 1) OVER (ORDER BY date), 0),
            -0.10), 0.10) AS brent_change_1d,
        -- Rate spread: Sovereign spread (Colombia 10Y - USA 10Y)
        -- Formula: 10.0 (Colombia hardcoded) - treasury_10y, then normalized
        -- Stats from training period: mean=7.03, std=1.41
        LEAST(GREATEST((10.0 - COALESCE(treasury_10y, 4.0) - 7.03) / 1.41, -4.0), 4.0) AS rate_spread,
        -- USDMXN change 1d (clip to [-0.05, 0.05])
        LEAST(GREATEST(
            (usdmxn - LAG(usdmxn, 1) OVER (ORDER BY date)) / NULLIF(LAG(usdmxn, 1) OVER (ORDER BY date), 0),
            -0.05), 0.05) AS usdmxn_change_1d
    FROM macro_latest
)
SELECT
    r.time,
    r.log_ret_5m,
    r.log_ret_1h,
    r.log_ret_4h,
    m.dxy_z,
    m.dxy_change_1d,
    m.vix_z,
    m.embi_z,
    m.brent_change_1d,
    m.rate_spread,
    m.usdmxn_change_1d
FROM ohlcv_with_returns r
LEFT JOIN macro_with_changes m ON DATE(r.time) = m.date
WHERE r.time >= NOW() - INTERVAL '30 days'
ORDER BY r.time DESC;

CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_features_sql_time
ON inference_features_5m_sql (time);

COMMENT ON MATERIALIZED VIEW inference_features_5m_sql IS
    'SQL-calculated features (FALLBACK). Less accurate for RSI/ATR/ADX. Refresh with CONCURRENTLY.';

-- =============================================================================
-- 4. Vista unificada (Python preferido, SQL fallback)
-- =============================================================================
-- This view provides a unified interface that prefers Python-calculated features
-- but falls back to SQL calculations or sensible defaults when Python data is missing.

CREATE OR REPLACE VIEW inference_features_complete AS
SELECT
    COALESCE(py.time, sql.time) AS time,
    COALESCE(py.log_ret_5m, sql.log_ret_5m) AS log_ret_5m,
    COALESCE(py.log_ret_1h, sql.log_ret_1h) AS log_ret_1h,
    COALESCE(py.log_ret_4h, sql.log_ret_4h) AS log_ret_4h,
    COALESCE(py.rsi_9, 50.0) AS rsi_9,           -- Default to neutral RSI
    COALESCE(py.atr_pct, 0.05) AS atr_pct,       -- Default to typical ATR
    COALESCE(py.adx_14, 25.0) AS adx_14,         -- Default to neutral ADX
    COALESCE(py.dxy_z, sql.dxy_z) AS dxy_z,
    COALESCE(py.dxy_change_1d, sql.dxy_change_1d) AS dxy_change_1d,
    COALESCE(py.vix_z, sql.vix_z) AS vix_z,
    COALESCE(py.embi_z, sql.embi_z) AS embi_z,
    COALESCE(py.brent_change_1d, sql.brent_change_1d) AS brent_change_1d,
    COALESCE(py.rate_spread, sql.rate_spread) AS rate_spread,
    COALESCE(py.usdmxn_change_1d, sql.usdmxn_change_1d, 0.0) AS usdmxn_change_1d,
    COALESCE(py.position, 0.0) AS position,
    COALESCE(py.time_normalized, 0.5) AS time_normalized
FROM inference_features_5m py
FULL OUTER JOIN inference_features_5m_sql sql ON py.time = sql.time;

COMMENT ON VIEW inference_features_complete IS
    'All 15 features with Python preferred, SQL fallback, sensible defaults. Use for model inference.';

-- =============================================================================
-- 5. DATA WAREHOUSE SCHEMA FOR INFERENCE RESULTS
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS dw;

-- Agent actions table (stores trading decisions)
CREATE TABLE IF NOT EXISTS dw.fact_agent_actions (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bar_number INTEGER,
    action DOUBLE PRECISION,
    position DOUBLE PRECISION DEFAULT 0.0,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_actions_time
ON dw.fact_agent_actions (timestamp DESC);

-- RL Inference results table
CREATE TABLE IF NOT EXISTS dw.fact_rl_inference (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    bar_number INTEGER,
    action DOUBLE PRECISION,
    observation JSONB,
    model_version TEXT,
    inference_time_ms DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_rl_inference_time
ON dw.fact_rl_inference (timestamp DESC);

-- =============================================================================
-- 6. HELPER FUNCTIONS
-- =============================================================================

-- Function to refresh SQL fallback materialized view
CREATE OR REPLACE FUNCTION refresh_inference_features_sql()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY inference_features_5m_sql;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest complete features (all 15)
CREATE OR REPLACE FUNCTION get_latest_features()
RETURNS TABLE (
    time TIMESTAMPTZ,
    log_ret_5m DOUBLE PRECISION,
    log_ret_1h DOUBLE PRECISION,
    log_ret_4h DOUBLE PRECISION,
    rsi_9 DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    dxy_z DOUBLE PRECISION,
    dxy_change_1d DOUBLE PRECISION,
    vix_z DOUBLE PRECISION,
    embi_z DOUBLE PRECISION,
    brent_change_1d DOUBLE PRECISION,
    rate_spread DOUBLE PRECISION,
    usdmxn_change_1d DOUBLE PRECISION,
    position DOUBLE PRECISION,
    time_normalized DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ifc.time,
        ifc.log_ret_5m,
        ifc.log_ret_1h,
        ifc.log_ret_4h,
        ifc.rsi_9,
        ifc.atr_pct,
        ifc.adx_14,
        ifc.dxy_z,
        ifc.dxy_change_1d,
        ifc.vix_z,
        ifc.embi_z,
        ifc.brent_change_1d,
        ifc.rate_spread,
        ifc.usdmxn_change_1d,
        ifc.position,
        ifc.time_normalized
    FROM inference_features_complete ifc
    ORDER BY ifc.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 7. VERIFICATION AND LOGGING
-- =============================================================================

DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Inference Features Views v2.0 Created Successfully';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'TABLE: inference_features_5m (Python L1 DAG - SSOT)';
    RAISE NOTICE 'TABLE: python_features_5m (RSI, ATR, ADX intermediate)';
    RAISE NOTICE 'MATVIEW: inference_features_5m_sql (SQL fallback)';
    RAISE NOTICE 'VIEW: inference_features_complete (unified with fallback)';
    RAISE NOTICE 'SCHEMA: dw (data warehouse for inference results)';
    RAISE NOTICE 'FUNCTION: refresh_inference_features_sql()';
    RAISE NOTICE 'FUNCTION: get_latest_features()';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'CRITICAL FIX: Column names now match macro_indicators_daily';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;
