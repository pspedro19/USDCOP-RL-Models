-- =====================================================
-- USDCOP Trading System - Inference Features Views
-- Creates materialized views for RL model inference
-- =====================================================
-- VERSION: v15 - Fixed Z-scores (FASE 3)
-- CRITICAL: These normalization formulas MUST match feature_config.json v4.0.0
--
-- Changes in v15:
-- - DXY z-score: Updated stats (mean=100.21, std=5.60) from training period 2020-03 to 2025-10
-- - VIX z-score: Updated stats (mean=21.16, std=7.89) from training period 2020-03 to 2025-10
-- - EMBI z-score: Updated stats (mean=322.01, std=62.68) from training period 2020-03 to 2025-10
-- - Rate spread: Changed to sovereign spread (Colombia 10Y - USA 10Y), normalized with mean=7.03, std=1.41
--
-- All stats calculated from training period: 2020-03 to 2025-10
-- =====================================================

-- =============================================================================
-- 1. SQL FEATURES MATERIALIZED VIEW (9 features from SQL calculations)
-- =============================================================================
-- Features: log_ret_5m, log_ret_1h, log_ret_4h, dxy_z, dxy_change_1d,
--           vix_z, embi_z, brent_change_1d, rate_spread

CREATE MATERIALIZED VIEW IF NOT EXISTS inference_features_5m AS
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
    SELECT DISTINCT ON (date)
        date,
        dxy,
        vix,
        embi,
        brent,
        treasury_2y,
        treasury_10y
    FROM macro_indicators_daily
    ORDER BY date DESC
),
macro_with_changes AS (
    SELECT
        date,
        -- DXY z-score (v15 fixed): (dxy - 100.21) / 5.60, clip to [-4, 4]
        -- Stats from training period 2020-03 to 2025-10
        LEAST(GREATEST((dxy - 100.21) / 5.60, -4.0), 4.0) AS dxy_z,
        -- DXY change 1d (clip to [-0.03, 0.03])
        LEAST(GREATEST(
            (dxy - LAG(dxy, 1) OVER (ORDER BY date)) / NULLIF(LAG(dxy, 1) OVER (ORDER BY date), 0),
            -0.03), 0.03) AS dxy_change_1d,
        -- VIX z-score (v15 fixed): (vix - 21.16) / 7.89, clip to [-4, 4]
        -- Stats from training period 2020-03 to 2025-10
        LEAST(GREATEST((vix - 21.16) / 7.89, -4.0), 4.0) AS vix_z,
        -- EMBI z-score (v15 fixed): (embi - 322.01) / 62.68, clip to [-4, 4]
        -- Stats from training period 2020-03 to 2025-10
        LEAST(GREATEST((embi - 322.01) / 62.68, -4.0), 4.0) AS embi_z,
        -- Brent change 1d (clip to [-0.10, 0.10])
        LEAST(GREATEST(
            (brent - LAG(brent, 1) OVER (ORDER BY date)) / NULLIF(LAG(brent, 1) OVER (ORDER BY date), 0),
            -0.10), 0.10) AS brent_change_1d,
        -- Rate spread (v15 fixed): Sovereign spread (Colombia 10Y - USA 10Y)
        -- Formula: 10.0 (Colombia hardcoded) - treasury_10y, then normalized
        -- Stats from training period: mean=7.03, std=1.41
        LEAST(GREATEST((10.0 - treasury_10y - 7.03) / 1.41, -4.0), 4.0) AS rate_spread
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
    m.rate_spread
FROM ohlcv_with_returns r
LEFT JOIN macro_with_changes m ON DATE(r.time) = m.date
WHERE r.time >= NOW() - INTERVAL '7 days'  -- Keep only recent data for inference
ORDER BY r.time DESC;

-- Create unique index for CONCURRENTLY refresh
CREATE UNIQUE INDEX IF NOT EXISTS idx_inference_features_time
ON inference_features_5m (time);

-- =============================================================================
-- 2. PYTHON FEATURES TABLE (4 features calculated in Python)
-- =============================================================================
-- Features: rsi_9, atr_pct, adx_14, usdmxn_change_1d
-- These are calculated by l1_feature_refresh.py DAG

CREATE TABLE IF NOT EXISTS python_features_5m (
    time TIMESTAMPTZ NOT NULL PRIMARY KEY,
    rsi_9 DOUBLE PRECISION,
    atr_pct DOUBLE PRECISION,
    adx_14 DOUBLE PRECISION,
    usdmxn_change_1d DOUBLE PRECISION,  -- v15: Renamed from usdmxn_ret_1h (daily data)
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_python_features_time
ON python_features_5m (time DESC);

-- =============================================================================
-- 3. UNIFIED INFERENCE FEATURES VIEW (all 13 features)
-- =============================================================================
-- This view joins SQL and Python features for model inference
-- Order matches feature_config.json observation_space.order

CREATE OR REPLACE VIEW inference_features_complete AS
SELECT
    s.time,
    -- In exact order from feature_config.json (SSOT)
    s.log_ret_5m,
    s.log_ret_1h,
    s.log_ret_4h,
    COALESCE(p.rsi_9, 50.0) AS rsi_9,       -- Default to neutral RSI
    COALESCE(p.atr_pct, 0.05) AS atr_pct,   -- Default to typical ATR
    COALESCE(p.adx_14, 25.0) AS adx_14,     -- Default to neutral ADX
    s.dxy_z,
    s.dxy_change_1d,
    s.vix_z,
    s.embi_z,
    s.brent_change_1d,
    s.rate_spread,
    COALESCE(p.usdmxn_change_1d, 0.0) AS usdmxn_change_1d  -- Default to 0 (v15: renamed)
FROM inference_features_5m s
LEFT JOIN python_features_5m p ON s.time = p.time;

-- =============================================================================
-- 4. DATA WAREHOUSE SCHEMA FOR INFERENCE RESULTS
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
-- 5. HELPER FUNCTIONS
-- =============================================================================

-- Function to refresh inference features
CREATE OR REPLACE FUNCTION refresh_inference_features()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY inference_features_5m;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest complete features (all 13)
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
    usdmxn_change_1d DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT * FROM inference_features_complete
    ORDER BY inference_features_complete.time DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS
-- =============================================================================
COMMENT ON MATERIALIZED VIEW inference_features_5m IS
    'SQL-calculated features for RL inference (9 of 13 features). Refresh with CONCURRENTLY.';

COMMENT ON TABLE python_features_5m IS
    'Python-calculated features (rsi_9, atr_pct, adx_14, usdmxn_change_1d). Updated by l1_feature_refresh DAG.';

COMMENT ON VIEW inference_features_complete IS
    'All 13 features in correct order for model.predict(). Joins SQL + Python features.';
