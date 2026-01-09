-- =====================================================
-- Migration: Rename usdmxn_ret_1h to usdmxn_change_1d
-- =====================================================
-- Version: V15
-- Date: 2025-12-17
-- Description: Renames the usdmxn feature column to reflect that it's daily data
--              This change aligns with training pipeline DS3_MACRO_CORE column names
-- =====================================================

-- Check if column needs renaming (in case this runs multiple times)
DO $$
BEGIN
    -- Rename column in python_features_5m table
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'python_features_5m'
        AND column_name = 'usdmxn_ret_1h'
    ) THEN
        ALTER TABLE python_features_5m
        RENAME COLUMN usdmxn_ret_1h TO usdmxn_change_1d;

        RAISE NOTICE 'Renamed python_features_5m.usdmxn_ret_1h to usdmxn_change_1d';
    ELSE
        RAISE NOTICE 'Column usdmxn_change_1d already exists or table does not exist';
    END IF;

    -- Rename column in inference_features_5m table (if it exists)
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'inference_features_5m'
        AND column_name = 'usdmxn_ret_1h'
    ) THEN
        ALTER TABLE inference_features_5m
        RENAME COLUMN usdmxn_ret_1h TO usdmxn_change_1d;

        RAISE NOTICE 'Renamed inference_features_5m.usdmxn_ret_1h to usdmxn_change_1d';
    ELSE
        RAISE NOTICE 'inference_features_5m.usdmxn_change_1d already exists or column not found';
    END IF;
END $$;

-- Update any views that reference the old column name
-- Drop and recreate inference_features_complete view
DROP VIEW IF EXISTS inference_features_complete;

CREATE OR REPLACE VIEW inference_features_complete AS
SELECT
    s.time,
    -- In exact order from feature_config.json (SSOT)
    s.log_ret_5m,
    s.log_ret_1h,
    s.log_ret_4h,
    COALESCE(p.rsi_9, 50.0) AS rsi_9,
    COALESCE(p.atr_pct, 0.05) AS atr_pct,
    COALESCE(p.adx_14, 25.0) AS adx_14,
    s.dxy_z,
    s.dxy_change_1d,
    s.vix_z,
    s.embi_z,
    s.brent_change_1d,
    s.rate_spread,
    COALESCE(p.usdmxn_change_1d, 0.0) AS usdmxn_change_1d
FROM inference_features_5m s
LEFT JOIN python_features_5m p ON s.time = p.time;

-- Update function to return new column name
DROP FUNCTION IF EXISTS get_latest_features();

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

-- Update comments
COMMENT ON TABLE python_features_5m IS
    'Python-calculated features (rsi_9, atr_pct, adx_14, usdmxn_change_1d). Updated by l1_feature_refresh DAG.';

COMMENT ON VIEW inference_features_complete IS
    'All 13 features in correct order for model.predict(). Joins SQL + Python features. V15: usdmxn_ret_1h renamed to usdmxn_change_1d.';

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE '===========================================';
    RAISE NOTICE 'Migration 002_rename_usdmxn_column complete';
    RAISE NOTICE 'V15: usdmxn_ret_1h -> usdmxn_change_1d';
    RAISE NOTICE '===========================================';
END $$;
