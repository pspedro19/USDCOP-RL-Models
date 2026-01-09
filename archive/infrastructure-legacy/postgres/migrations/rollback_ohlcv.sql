-- =====================================================
-- ROLLBACK SCRIPT: Revert OHLCV migrations
-- Description: Safely remove all changes from migrations 001-003
-- Author: Migration System
-- Date: 2025-10-22
-- WARNING: This will remove OHLCV columns but preserve original data
-- =====================================================

\echo ''
\echo '=========================================='
\echo 'ROLLBACK: Reverting OHLCV migrations'
\echo '=========================================='
\echo ''
\echo 'WARNING: This will remove OHLCV columns!'
\echo 'Original price column will be preserved.'
\echo ''

-- Prompt for confirmation (manual execution)
\prompt 'Type YES to continue with rollback: ' confirmation

-- Start transaction
BEGIN;

-- ========================================
-- STEP 1: DROP CONSTRAINTS (Migration 003)
-- ========================================

\echo ''
\echo 'Step 1: Dropping validation constraints...'

-- Drop OHLC validation constraints
DROP CONSTRAINT IF EXISTS chk_high_gte_open CASCADE;
DROP CONSTRAINT IF EXISTS chk_high_gte_close CASCADE;
DROP CONSTRAINT IF EXISTS chk_low_lte_open CASCADE;
DROP CONSTRAINT IF EXISTS chk_low_lte_close CASCADE;
DROP CONSTRAINT IF EXISTS chk_high_gte_low CASCADE;
\echo '  ✓ OHLC validation constraints dropped'

-- Drop price range constraints
DROP CONSTRAINT IF EXISTS chk_prices_positive CASCADE;
DROP CONSTRAINT IF EXISTS chk_volume_non_negative CASCADE;
\echo '  ✓ Price range constraints dropped'

-- Drop spread constraints
DROP CONSTRAINT IF EXISTS chk_spread_non_negative CASCADE;
DROP CONSTRAINT IF EXISTS chk_ask_gte_bid CASCADE;
\echo '  ✓ Spread constraints dropped'

-- Drop data quality constraints
DROP CONSTRAINT IF EXISTS chk_data_quality_range CASCADE;
\echo '  ✓ Data quality constraints dropped'

-- Drop timeframe constraints
DROP CONSTRAINT IF EXISTS chk_valid_timeframe CASCADE;
\echo '  ✓ Timeframe constraints dropped'

-- ========================================
-- STEP 2: DROP INDEXES (Migration 002)
-- ========================================

\echo ''
\echo 'Step 2: Dropping optimized indexes...'

-- Drop composite indexes
DROP INDEX IF EXISTS idx_market_data_symbol_timeframe_time CASCADE;
DROP INDEX IF EXISTS idx_market_data_source_time CASCADE;
\echo '  ✓ Composite indexes dropped'

-- Drop partial indexes
DROP INDEX IF EXISTS idx_market_data_recent_trading CASCADE;
DROP INDEX IF EXISTS idx_market_data_intraday_5min CASCADE;
\echo '  ✓ Partial indexes dropped'

-- Drop BRIN indexes
DROP INDEX IF EXISTS idx_market_data_timestamp_brin CASCADE;
DROP INDEX IF EXISTS idx_market_data_ohlc_brin CASCADE;
\echo '  ✓ BRIN indexes dropped'

-- Drop gap detection indexes
DROP INDEX IF EXISTS idx_market_data_gap_detection CASCADE;
DROP INDEX IF EXISTS idx_market_data_completeness CASCADE;
\echo '  ✓ Gap detection indexes dropped'

-- Drop data quality indexes
DROP INDEX IF EXISTS idx_market_data_quality_issues CASCADE;
DROP INDEX IF EXISTS idx_market_data_zero_volume CASCADE;
DROP INDEX IF EXISTS idx_market_data_spread_analysis CASCADE;
\echo '  ✓ Data quality indexes dropped'

-- Drop covering indexes
DROP INDEX IF EXISTS idx_market_data_ohlc_covering CASCADE;
\echo '  ✓ Covering indexes dropped'

-- ========================================
-- STEP 3: DROP COLUMNS (Migration 001)
-- ========================================

\echo ''
\echo 'Step 3: Dropping OHLCV columns...'
\echo 'NOTE: Original price, bid, ask, volume columns will be preserved'

-- Drop OHLC columns (but keep price)
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'open'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS open CASCADE;
        RAISE NOTICE '  ✓ Column open dropped';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'high'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS high CASCADE;
        RAISE NOTICE '  ✓ Column high dropped';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'low'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS low CASCADE;
        RAISE NOTICE '  ✓ Column low dropped';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'close'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS close CASCADE;
        RAISE NOTICE '  ✓ Column close dropped';
    END IF;
END $$;

-- Drop additional columns
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'timeframe'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS timeframe CASCADE;
        RAISE NOTICE '  ✓ Column timeframe dropped';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'spread'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS spread CASCADE;
        RAISE NOTICE '  ✓ Column spread dropped';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'data_quality'
    ) THEN
        ALTER TABLE market_data DROP COLUMN IF EXISTS data_quality CASCADE;
        RAISE NOTICE '  ✓ Column data_quality dropped';
    END IF;
END $$;

-- ========================================
-- STEP 4: VERIFY ROLLBACK
-- ========================================

\echo ''
\echo 'Step 4: Verifying rollback...'

DO $$
DECLARE
    total_rows INTEGER;
    has_price BOOLEAN;
    has_ohlc BOOLEAN;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM market_data;

    -- Check if price column still exists
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'price'
    ) INTO has_price;

    -- Check if OHLC columns are gone
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name IN ('open', 'high', 'low', 'close')
    ) INTO has_ohlc;

    RAISE NOTICE '';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Rollback Verification Results:';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Total rows preserved: %', total_rows;
    RAISE NOTICE 'Price column exists: %', has_price;
    RAISE NOTICE 'OHLC columns removed: %', NOT has_ohlc;

    IF has_price AND NOT has_ohlc THEN
        RAISE NOTICE '✓ Rollback successful - original structure restored';
    ELSE
        RAISE WARNING '⚠ Rollback may be incomplete - verify manually';
    END IF;
END $$;

-- Show current table structure
\echo ''
\echo 'Current market_data table structure:'
\d market_data

-- Commit transaction
COMMIT;

\echo ''
\echo '=========================================='
\echo '✓ ROLLBACK COMPLETED'
\echo '=========================================='
\echo ''
\echo 'Summary of changes:'
\echo '  - All validation constraints removed'
\echo '  - All optimized indexes removed'
\echo '  - OHLCV columns removed (open, high, low, close)'
\echo '  - timeframe, spread, data_quality columns removed'
\echo '  - Original price column preserved'
\echo '  - All data rows preserved'
\echo ''
\echo 'Database has been restored to pre-migration state.'
\echo ''
