-- ==============================================================================
-- USD/COP Trading System - Inference Features Materialized View
-- Version: 3.1 (Migration V14)
-- ==============================================================================
-- REFERENCIA: ARQUITECTURA_INTEGRAL_V3.md (líneas 340-386)
-- REFERENCIA: MAPEO_MIGRACION_BIDIRECCIONAL.md (líneas 61-66, ERRATA líneas 42-44)
-- ==============================================================================
-- CORRECCIONES APLICADAS (de MAPEO ERRATA):
--   - hour_sin, hour_cos ELIMINADOS (línea 42)
--   - Usar INNER JOIN, no LEFT JOIN (ARQUITECTURA sección 11.2)
--   - usdmxn_ret_1h clip corregido a [-0.10, 0.10] (línea 43)
-- ==============================================================================
-- DISTRIBUCIÓN DE FEATURES (13 totales para modelo PPO v11/v14):
-- ┌─────────────────────────────────────────────────────────────────┐
-- │ CALCULADOS EN SQL (9 features):                                 │
-- │   - log_ret_5m, log_ret_1h, log_ret_4h (logarithmic returns)   │
-- │   - dxy_z, vix_z, embi_z (macro z-scores)                      │
-- │   - dxy_change_1d, brent_change_1d, rate_spread                │
-- ├─────────────────────────────────────────────────────────────────┤
-- │ CALCULADOS EN PYTHON SERVICE (4 features):                      │
-- │   - rsi_9 (RSI period=9)                                        │
-- │   - atr_pct (ATR% period=10)                                    │
-- │   - adx_14 (ADX period=14)                                      │
-- │   - usdmxn_ret_1h (USD/MXN hourly return, clip ±0.10)          │
-- ├─────────────────────────────────────────────────────────────────┤
-- │ ELIMINADOS EN V14 (DO NOT USE):                                 │
-- │   - hour_sin, hour_cos (low predictive value for FX)            │
-- │   - bb_position, dxy_mom_5d, vix_regime, brent_vol_5d          │
-- └─────────────────────────────────────────────────────────────────┘
-- ==============================================================================
-- Prerequisite: 01_core_tables.sql must be executed first
-- Refresh Schedule: Every 5 minutes during market hours
-- ==============================================================================

DROP MATERIALIZED VIEW IF EXISTS inference_features_5m CASCADE;

CREATE MATERIALIZED VIEW inference_features_5m AS
WITH
-- ===========================================================================
-- STEP 1: OHLCV with multi-timeframe returns
-- ===========================================================================
ohlcv_with_returns AS (
    SELECT
        o.time AS timestamp,
        o.open,
        o.high,
        o.low,
        o.close,
        o.volume,

        -- =================================================================
        -- FEATURE 1: log_ret_5m
        -- Formula: ln(close_t / close_t-1)
        -- Clipping: [-0.05, 0.05]
        -- =================================================================
        LN(o.close / NULLIF(LAG(o.close, 1) OVER w, 0)) AS log_ret_5m,

        -- =================================================================
        -- FEATURE 2: log_ret_1h
        -- Formula: ln(close_t / close_t-12)
        -- Clipping: [-0.05, 0.05]
        -- =================================================================
        LN(o.close / NULLIF(LAG(o.close, 12) OVER w, 0)) AS log_ret_1h,

        -- =================================================================
        -- FEATURE 3: log_ret_4h
        -- Formula: ln(close_t / close_t-48)
        -- Clipping: [-0.05, 0.05]
        -- =================================================================
        LN(o.close / NULLIF(LAG(o.close, 48) OVER w, 0)) AS log_ret_4h,

        -- =================================================================
        -- RAW RETURN for reward calculation (NOT normalized)
        -- Used by RL environment to calculate PnL
        -- =================================================================
        o.close / NULLIF(LAG(o.close, 1) OVER w, 0) - 1 AS _raw_ret_5m,

        -- =================================================================
        -- AUXILIARY DATA FOR PYTHON SERVICE (RSI, ATR, ADX)
        -- These fields allow technical indicators calculation in Python
        -- =================================================================
        LAG(o.close, 1) OVER w AS prev_close,
        LAG(o.close, 2) OVER w AS prev_close_2,
        LAG(o.close, 9) OVER w AS prev_close_9,  -- For RSI period 9

        -- High/Low for ATR calculation
        o.high - o.low AS range_hl,
        LAG(o.high, 1) OVER w AS prev_high,
        LAG(o.low, 1) OVER w AS prev_low,

        -- Rolling statistics for volatility-based indicators
        AVG(o.close) OVER (ORDER BY o.time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS sma_20,
        STDDEV(o.close) OVER (ORDER BY o.time ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS std_20

    FROM usdcop_m5_ohlcv o
    WHERE o.time >= NOW() - INTERVAL '30 days'
      AND o.symbol = 'USD/COP'
    WINDOW w AS (ORDER BY o.time)
),

-- ===========================================================================
-- STEP 2: Macro indicators with transformations
-- ===========================================================================
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

        -- =================================================================
        -- FEATURE 7: dxy_z
        -- Formula: (dxy - 103.0) / 5.0
        -- Clipping: [-4, 4]
        -- FIXED parameters from training (DO NOT recalculate)
        -- =================================================================
        (d.dxy - 103.0) / NULLIF(5.0, 0) AS dxy_z,

        -- =================================================================
        -- FEATURE 9: vix_z
        -- Formula: (vix - 20.0) / 10.0
        -- Clipping: [-4, 4]
        -- FIXED parameters from training (DO NOT recalculate)
        -- =================================================================
        (d.vix - 20.0) / NULLIF(10.0, 0) AS vix_z,

        -- =================================================================
        -- FEATURE 10: embi_z
        -- Formula: (embi - 300.0) / 100.0
        -- Clipping: [-4, 4]
        -- FIXED parameters from training (DO NOT recalculate)
        -- =================================================================
        (d.embi - 300.0) / NULLIF(100.0, 0) AS embi_z,

        -- =================================================================
        -- FEATURE 8: dxy_change_1d
        -- Formula: (dxy_t - dxy_t-1) / dxy_t-1
        -- Clipping: [-0.03, 0.03]
        -- =================================================================
        (d.dxy - LAG(d.dxy, 1) OVER (ORDER BY d.date)) /
            NULLIF(LAG(d.dxy, 1) OVER (ORDER BY d.date), 0) AS dxy_change_1d,

        -- =================================================================
        -- FEATURE 11: brent_change_1d
        -- Formula: (brent_t - brent_t-1) / brent_t-1
        -- Clipping: [-0.10, 0.10]
        -- =================================================================
        (d.brent - LAG(d.brent, 1) OVER (ORDER BY d.date)) /
            NULLIF(LAG(d.brent, 1) OVER (ORDER BY d.date), 0) AS brent_change_1d,

        -- =================================================================
        -- USDMXN DATA FOR PYTHON CALCULATION
        -- usdmxn_ret_1h requires 12 periods lookback (calculated in Python)
        -- Here we provide raw values for Python service
        -- REF: MAPEO línea 40-41, periods=12
        -- =================================================================
        LAG(d.usdmxn, 1) OVER (ORDER BY d.date) AS usdmxn_prev,

        -- =================================================================
        -- FEATURE 12: rate_spread
        -- Formula: treasury_10y - treasury_2y
        -- Clipping: None
        -- Positive = normal curve, negative = inverted curve
        -- =================================================================
        d.treasury_10y - d.treasury_2y AS rate_spread

    FROM macro_indicators_daily d
    WHERE d.date >= CURRENT_DATE - INTERVAL '60 days'
      AND d.is_complete = TRUE  -- Only use complete macro data
)

-- ===========================================================================
-- STEP 3: JOIN and final feature calculation
-- ===========================================================================
SELECT
    owr.timestamp,
    owr.close,
    owr.volume,

    -- =========================================================================
    -- SQL-CALCULATED FEATURES (9 features)
    -- =========================================================================

    -- FEATURE 1: log_ret_5m - Clipped to [-0.05, 0.05]
    LEAST(GREATEST(COALESCE(owr.log_ret_5m, 0), -0.05), 0.05) AS log_ret_5m,

    -- FEATURE 2: log_ret_1h - Clipped to [-0.05, 0.05]
    LEAST(GREATEST(COALESCE(owr.log_ret_1h, 0), -0.05), 0.05) AS log_ret_1h,

    -- FEATURE 3: log_ret_4h - Clipped to [-0.05, 0.05]
    LEAST(GREATEST(COALESCE(owr.log_ret_4h, 0), -0.05), 0.05) AS log_ret_4h,

    -- FEATURE 7: dxy_z - Clipped to [-4, 4]
    LEAST(GREATEST(COALESCE(mp.dxy_z, 0), -4), 4) AS dxy_z,

    -- FEATURE 8: dxy_change_1d - Clipped to [-0.03, 0.03]
    LEAST(GREATEST(COALESCE(mp.dxy_change_1d, 0), -0.03), 0.03) AS dxy_change_1d,

    -- FEATURE 9: vix_z - Clipped to [-4, 4]
    LEAST(GREATEST(COALESCE(mp.vix_z, 0), -4), 4) AS vix_z,

    -- FEATURE 10: embi_z - Clipped to [-4, 4]
    LEAST(GREATEST(COALESCE(mp.embi_z, 0), -4), 4) AS embi_z,

    -- FEATURE 11: brent_change_1d - Clipped to [-0.10, 0.10]
    LEAST(GREATEST(COALESCE(mp.brent_change_1d, 0), -0.10), 0.10) AS brent_change_1d,

    -- FEATURE 12: rate_spread - No clipping
    COALESCE(mp.rate_spread, 0) AS rate_spread,

    -- =========================================================================
    -- PYTHON-CALCULATED FEATURES (4 features - NULL placeholders)
    -- =========================================================================
    -- These will be calculated by the Python service using auxiliary data below

    -- FEATURE 4: rsi_9 (calculated in Python)
    NULL::FLOAT AS rsi_9,

    -- FEATURE 5: atr_pct (calculated in Python)
    NULL::FLOAT AS atr_pct,

    -- FEATURE 6: adx_14 (calculated in Python)
    NULL::FLOAT AS adx_14,

    -- FEATURE 13: usdmxn_ret_1h (calculated in Python)
    NULL::FLOAT AS usdmxn_ret_1h,

    -- =========================================================================
    -- RAW RETURN for reward (NOT clipped)
    -- =========================================================================
    COALESCE(owr._raw_ret_5m, 0) AS _raw_ret_5m,

    -- =========================================================================
    -- AUXILIARY DATA FOR PYTHON SERVICE
    -- These columns enable technical indicator calculation in Python
    -- =========================================================================
    -- OHLC data for RSI, ATR, ADX
    owr.open,
    owr.high,
    owr.low,
    owr.prev_close,
    owr.prev_high,
    owr.prev_low,
    owr.range_hl,
    owr.prev_close_9,           -- For RSI period 9

    -- USDMXN data for usdmxn_ret_1h calculation
    mp.usdmxn AS usdmxn_raw,
    mp.usdmxn_prev,

    -- SMA/STD for potential future use
    owr.sma_20,
    owr.std_20,

    -- =========================================================================
    -- METADATA
    -- Useful for filtering, analysis, and debugging
    -- =========================================================================
    DATE(owr.timestamp AT TIME ZONE 'America/Bogota') AS trading_date,
    EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota')::INT AS hour_cot,
    EXTRACT(MINUTE FROM owr.timestamp AT TIME ZONE 'America/Bogota')::INT AS minute_cot,
    EXTRACT(DOW FROM owr.timestamp AT TIME ZONE 'America/Bogota')::INT AS day_of_week,
    get_bar_number(owr.timestamp) AS bar_number

FROM ohlcv_with_returns owr
-- ===========================================================================
-- INNER JOIN (NOT LEFT JOIN) - Only process bars with complete macro data
-- REF: ARQUITECTURA sección 11.2 "Agente 3"
-- REF: MAPEO ERRATA - INNER JOIN avoids inconsistent NULLs
-- ===========================================================================
INNER JOIN macro_processed mp
    ON DATE(owr.timestamp AT TIME ZONE 'America/Bogota') = mp.date

WHERE
    -- Only market hours (8:00-12:55 COT = 13:00-17:55 UTC)
    EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') >= 8
    AND (
        EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') < 13
        OR (EXTRACT(HOUR FROM owr.timestamp AT TIME ZONE 'America/Bogota') = 12
            AND EXTRACT(MINUTE FROM owr.timestamp AT TIME ZONE 'America/Bogota') <= 55)
    )
    -- Only weekdays (Monday=1 to Friday=5)
    AND EXTRACT(DOW FROM owr.timestamp AT TIME ZONE 'America/Bogota') BETWEEN 1 AND 5

ORDER BY owr.timestamp DESC;


-- ===========================================================================
-- INDICES
-- ===========================================================================

-- Unique index REQUIRED for REFRESH MATERIALIZED VIEW CONCURRENTLY
CREATE UNIQUE INDEX idx_inf_features_ts ON inference_features_5m (timestamp);

-- Additional indices for common query patterns
CREATE INDEX idx_inf_features_date ON inference_features_5m (trading_date DESC);
CREATE INDEX idx_inf_features_hour ON inference_features_5m (hour_cot, trading_date);
CREATE INDEX idx_inf_features_bar ON inference_features_5m (bar_number, trading_date);


-- ===========================================================================
-- COMMENTS
-- ===========================================================================

COMMENT ON MATERIALIZED VIEW inference_features_5m IS
'Materialized view with pre-calculated features for RL model inference.
9 features calculated in SQL + 4 calculated in Python = 13 total.
ELIMINATED in V14: hour_sin, hour_cos (MAPEO line 42).
Refresh every 5 minutes during market hours using refresh_inference_features().
Uses INNER JOIN with macro data (not LEFT JOIN) per ERRATA.
Feature definitions: config/feature_config.json';

COMMENT ON COLUMN inference_features_5m.timestamp IS
    'UTC timestamp of the 5-minute bar';

COMMENT ON COLUMN inference_features_5m.log_ret_5m IS
    'FEATURE 1: Logarithmic return over 5 minutes, clipped to [-0.05, 0.05]';

COMMENT ON COLUMN inference_features_5m.log_ret_1h IS
    'FEATURE 2: Logarithmic return over 1 hour (12 bars), clipped to [-0.05, 0.05]';

COMMENT ON COLUMN inference_features_5m.log_ret_4h IS
    'FEATURE 3: Logarithmic return over 4 hours (48 bars), clipped to [-0.05, 0.05]';

COMMENT ON COLUMN inference_features_5m.rsi_9 IS
    'FEATURE 4: RSI period 9 (calculated in Python service)';

COMMENT ON COLUMN inference_features_5m.atr_pct IS
    'FEATURE 5: ATR percentage period 10 (calculated in Python service)';

COMMENT ON COLUMN inference_features_5m.adx_14 IS
    'FEATURE 6: ADX period 14 (calculated in Python service)';

COMMENT ON COLUMN inference_features_5m.dxy_z IS
    'FEATURE 7: DXY z-score: (dxy - 103) / 5, clipped to [-4, 4]';

COMMENT ON COLUMN inference_features_5m.dxy_change_1d IS
    'FEATURE 8: DXY daily percentage change, clipped to [-0.03, 0.03]';

COMMENT ON COLUMN inference_features_5m.vix_z IS
    'FEATURE 9: VIX z-score: (vix - 20) / 10, clipped to [-4, 4]';

COMMENT ON COLUMN inference_features_5m.embi_z IS
    'FEATURE 10: EMBI z-score: (embi - 300) / 100, clipped to [-4, 4]';

COMMENT ON COLUMN inference_features_5m.brent_change_1d IS
    'FEATURE 11: Brent crude oil daily percentage change, clipped to [-0.10, 0.10]';

COMMENT ON COLUMN inference_features_5m.rate_spread IS
    'FEATURE 12: US Treasury yield curve spread: 10Y - 2Y';

COMMENT ON COLUMN inference_features_5m.usdmxn_ret_1h IS
    'FEATURE 13: USD/MXN hourly return, clipped to [-0.10, 0.10] (calculated in Python service)';

COMMENT ON COLUMN inference_features_5m._raw_ret_5m IS
    'Raw 5-minute return (NOT clipped) for reward calculation';

COMMENT ON COLUMN inference_features_5m.bar_number IS
    'Bar number within trading session (1-60). Bar 1 = 08:00, Bar 60 = 12:55 COT';


-- ===========================================================================
-- REFRESH FUNCTION
-- ===========================================================================

CREATE OR REPLACE FUNCTION refresh_inference_features()
RETURNS TEXT AS $$
DECLARE
    row_count INT;
BEGIN
    -- CONCURRENTLY allows queries during refresh
    -- Requires unique index on timestamp
    REFRESH MATERIALIZED VIEW CONCURRENTLY inference_features_5m;

    SELECT COUNT(*) INTO row_count FROM inference_features_5m;

    RETURN format('Refreshed inference_features_5m at %s. Rows: %s',
                  NOW()::TEXT, row_count);
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION refresh_inference_features() IS
'Refresh the inference_features_5m materialized view.
Call from Airflow DAG every 5 minutes during market hours.
Uses CONCURRENTLY to avoid blocking queries.
Returns timestamp and row count for logging.';


-- ===========================================================================
-- GRANTS
-- ===========================================================================

-- Grant read access to trading_app
GRANT SELECT ON inference_features_5m TO trading_app;

-- Grant refresh permission to airflow
GRANT EXECUTE ON FUNCTION refresh_inference_features() TO airflow;


-- ===========================================================================
-- VERIFICATION
-- ===========================================================================

SELECT 'Inference view schema created successfully' AS status, NOW() AS created_at;

-- Verify materialized view was created
SELECT
    schemaname,
    matviewname,
    ispopulated,
    pg_size_pretty(pg_relation_size(matviewname::regclass)) AS size
FROM pg_matviews
WHERE matviewname = 'inference_features_5m';

-- List columns in the view
SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'inference_features_5m'
ORDER BY ordinal_position;

-- Show feature summary
SELECT
    'SQL features: 9' AS feature_group,
    'log_ret_5m, log_ret_1h, log_ret_4h, dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d, rate_spread' AS features
UNION ALL
SELECT
    'Python features: 4' AS feature_group,
    'rsi_9, atr_pct, adx_14, usdmxn_ret_1h' AS features
UNION ALL
SELECT
    'TOTAL: 13 features' AS feature_group,
    'V14 - hour_sin/hour_cos ELIMINATED' AS features;
