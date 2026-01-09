-- =====================================================================
-- Migration 001: Initial Setup - USD/COP Trading System
-- =====================================================================
--
-- This migration creates the complete database schema for the USD/COP
-- trading system including:
--   - Core tables (macro_indicators_daily, dw.fact_rl_inference)
--   - Materialized view (inference_features_5m)
--   - Helper functions (market hours, bar numbers, holidays)
--   - Proper indices and constraints
--
-- Author: Pedro @ Lean Tech Solutions
-- Date: 2025-12-16
-- Version: 3.1
--
-- Prerequisites:
--   - PostgreSQL 14+
--   - TimescaleDB extension
--   - usdcop_m5_ohlcv table (from init-scripts/01-essential-usdcop-init.sql)
--
-- Rollback:
--   - See rollback section at end of file
-- =====================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- =====================================================================
-- STEP 1: Execute Core Tables Schema
-- =====================================================================

\echo 'Creating core tables...'
\i database/schemas/01_core_tables.sql

-- =====================================================================
-- STEP 2: Execute Inference View Schema
-- =====================================================================

\echo 'Creating inference view...'
\i database/schemas/02_inference_view.sql

-- =====================================================================
-- STEP 3: Initial Data Population (Optional)
-- =====================================================================

-- Insert sample holiday data for testing (if needed)
-- Uncomment if you want to seed data during migration
/*
\echo 'Seeding initial data...'

-- Example: Insert a test row in macro_indicators_daily
INSERT INTO macro_indicators_daily (
    date, dxy, vix, embi, brent, treasury_2y, treasury_10y, usdmxn,
    source, is_complete
) VALUES (
    CURRENT_DATE - INTERVAL '1 day',
    103.5, 18.2, 295.0, 82.5, 4.25, 4.15, 17.2,
    'test_data', TRUE
) ON CONFLICT (date) DO NOTHING;
*/

-- =====================================================================
-- STEP 4: Verification
-- =====================================================================

\echo 'Verifying migration...'

-- Check that all tables exist
DO $$
DECLARE
    missing_count INT;
BEGIN
    -- Check macro_indicators_daily in public schema
    SELECT COUNT(*) INTO missing_count
    FROM information_schema.tables
    WHERE table_schema = 'public' AND table_name = 'macro_indicators_daily';

    IF missing_count = 0 THEN
        RAISE EXCEPTION 'Missing table: public.macro_indicators_daily';
    END IF;

    -- Check fact_rl_inference in dw schema
    SELECT COUNT(*) INTO missing_count
    FROM information_schema.tables
    WHERE table_schema = 'dw' AND table_name = 'fact_rl_inference';

    IF missing_count = 0 THEN
        RAISE EXCEPTION 'Missing table: dw.fact_rl_inference';
    END IF;

    RAISE NOTICE 'All core tables created successfully';
END$$;

-- Check that materialized view exists
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_matviews
        WHERE matviewname = 'inference_features_5m'
    ) THEN
        RAISE EXCEPTION 'Materialized view inference_features_5m not created';
    END IF;

    RAISE NOTICE 'Inference view created successfully';
END$$;

-- Check that helper functions exist
DO $$
DECLARE
    missing_functions TEXT[];
BEGIN
    SELECT ARRAY_AGG(expected_function)
    INTO missing_functions
    FROM (VALUES
        ('is_market_open'),
        ('get_bar_number'),
        ('is_colombia_holiday'),
        ('should_run_inference'),
        ('refresh_inference_features')
    ) AS expected(expected_function)
    WHERE NOT EXISTS (
        SELECT 1 FROM pg_proc
        WHERE proname = expected_function
    );

    IF missing_functions IS NOT NULL THEN
        RAISE EXCEPTION 'Missing functions: %', missing_functions;
    END IF;

    RAISE NOTICE 'All helper functions created successfully';
END$$;

-- =====================================================================
-- MIGRATION COMPLETE
-- =====================================================================

SELECT
    'Migration 001 completed successfully' AS status,
    NOW() AS completed_at,
    version() AS database_version;

-- Summary of created objects
SELECT
    'Tables' AS object_type,
    COUNT(*) AS count
FROM information_schema.tables
WHERE (table_schema = 'public' AND table_name = 'macro_indicators_daily')
   OR (table_schema = 'dw' AND table_name = 'fact_rl_inference')

UNION ALL

SELECT
    'Materialized Views' AS object_type,
    COUNT(*) AS count
FROM pg_matviews
WHERE matviewname = 'inference_features_5m'

UNION ALL

SELECT
    'Functions' AS object_type,
    COUNT(*) AS count
FROM pg_proc
WHERE proname IN ('is_market_open', 'get_bar_number', 'is_colombia_holiday',
                  'should_run_inference', 'refresh_inference_features');


-- =====================================================================
-- ROLLBACK SCRIPT (for emergencies)
-- =====================================================================
-- To rollback this migration, execute the following commands:
--
-- DROP MATERIALIZED VIEW IF EXISTS inference_features_5m CASCADE;
-- DROP TABLE IF EXISTS dw.fact_rl_inference CASCADE;
-- DROP TABLE IF EXISTS macro_indicators_daily CASCADE;
-- DROP FUNCTION IF EXISTS refresh_inference_features() CASCADE;
-- DROP FUNCTION IF EXISTS should_run_inference() CASCADE;
-- DROP FUNCTION IF EXISTS is_colombia_holiday(DATE) CASCADE;
-- DROP FUNCTION IF EXISTS get_bar_number(TIMESTAMPTZ) CASCADE;
-- DROP FUNCTION IF EXISTS is_market_open() CASCADE;
-- =====================================================================
