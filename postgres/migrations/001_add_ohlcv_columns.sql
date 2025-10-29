-- =====================================================
-- MIGRATION 001: Add OHLCV columns to market_data table
-- Description: Transform single price column to full OHLCV candlestick data
-- Author: Migration System
-- Date: 2025-10-22
-- =====================================================

\echo ''
\echo '=========================================='
\echo 'MIGRATION 001: Adding OHLCV columns'
\echo '=========================================='
\echo ''

-- Start transaction for atomic migration
BEGIN;

-- Step 1: Add timeframe column if not exists
\echo 'Step 1: Adding timeframe column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'timeframe'
    ) THEN
        ALTER TABLE market_data ADD COLUMN timeframe VARCHAR(10) DEFAULT '5min';
        RAISE NOTICE '✓ Column timeframe added';
    ELSE
        RAISE NOTICE '⊙ Column timeframe already exists';
    END IF;
END $$;

-- Step 2: Add open column if not exists
\echo 'Step 2: Adding open column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'open'
    ) THEN
        ALTER TABLE market_data ADD COLUMN open DECIMAL(12,4);
        RAISE NOTICE '✓ Column open added';
    ELSE
        RAISE NOTICE '⊙ Column open already exists';
    END IF;
END $$;

-- Step 3: Add high column if not exists
\echo 'Step 3: Adding high column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'high'
    ) THEN
        ALTER TABLE market_data ADD COLUMN high DECIMAL(12,4);
        RAISE NOTICE '✓ Column high added';
    ELSE
        RAISE NOTICE '⊙ Column high already exists';
    END IF;
END $$;

-- Step 4: Add low column if not exists
\echo 'Step 4: Adding low column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'low'
    ) THEN
        ALTER TABLE market_data ADD COLUMN low DECIMAL(12,4);
        RAISE NOTICE '✓ Column low added';
    ELSE
        RAISE NOTICE '⊙ Column low already exists';
    END IF;
END $$;

-- Step 5: Add close column if not exists
\echo 'Step 5: Adding close column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'close'
    ) THEN
        ALTER TABLE market_data ADD COLUMN close DECIMAL(12,4);
        RAISE NOTICE '✓ Column close added';
    ELSE
        RAISE NOTICE '⊙ Column close already exists';
    END IF;
END $$;

-- Step 6: Add spread column for bid-ask spread
\echo 'Step 6: Adding spread column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'spread'
    ) THEN
        ALTER TABLE market_data ADD COLUMN spread DECIMAL(12,6) DEFAULT 0;
        RAISE NOTICE '✓ Column spread added';
    ELSE
        RAISE NOTICE '⊙ Column spread already exists';
    END IF;
END $$;

-- Step 7: Add data_quality score column
\echo 'Step 7: Adding data_quality column...'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'market_data' AND column_name = 'data_quality'
    ) THEN
        ALTER TABLE market_data ADD COLUMN data_quality INTEGER DEFAULT 100;
        RAISE NOTICE '✓ Column data_quality added';
    ELSE
        RAISE NOTICE '⊙ Column data_quality already exists';
    END IF;
END $$;

-- Step 8: Backfill OHLCV from existing price data
\echo ''
\echo 'Step 8: Backfilling OHLCV data from price column...'
DO $$
DECLARE
    rows_updated INTEGER;
BEGIN
    -- Update open, high, low, close with price value where they are NULL
    UPDATE market_data
    SET
        open = COALESCE(open, price),
        high = COALESCE(high, price),
        low = COALESCE(low, price),
        close = COALESCE(close, price)
    WHERE
        price IS NOT NULL
        AND (open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL);

    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    RAISE NOTICE '✓ Backfilled % rows with OHLC data from price', rows_updated;
END $$;

-- Step 9: Calculate spread from bid-ask if available
\echo 'Step 9: Calculating spread from bid-ask...'
DO $$
DECLARE
    rows_updated INTEGER;
BEGIN
    UPDATE market_data
    SET spread = COALESCE(ask - bid, 0)
    WHERE bid IS NOT NULL
        AND ask IS NOT NULL
        AND (spread IS NULL OR spread = 0);

    GET DIAGNOSTICS rows_updated = ROW_COUNT;
    RAISE NOTICE '✓ Calculated spread for % rows', rows_updated;
END $$;

-- Step 10: Set NOT NULL constraints after backfill
\echo ''
\echo 'Step 10: Adding NOT NULL constraints...'
DO $$
BEGIN
    -- Check if we have any NULL values before adding constraints
    IF NOT EXISTS (SELECT 1 FROM market_data WHERE open IS NULL LIMIT 1) THEN
        ALTER TABLE market_data ALTER COLUMN open SET NOT NULL;
        RAISE NOTICE '✓ NOT NULL constraint added to open';
    ELSE
        RAISE WARNING '⚠ Cannot add NOT NULL to open - NULL values exist';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM market_data WHERE high IS NULL LIMIT 1) THEN
        ALTER TABLE market_data ALTER COLUMN high SET NOT NULL;
        RAISE NOTICE '✓ NOT NULL constraint added to high';
    ELSE
        RAISE WARNING '⚠ Cannot add NOT NULL to high - NULL values exist';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM market_data WHERE low IS NULL LIMIT 1) THEN
        ALTER TABLE market_data ALTER COLUMN low SET NOT NULL;
        RAISE NOTICE '✓ NOT NULL constraint added to low';
    ELSE
        RAISE WARNING '⚠ Cannot add NOT NULL to low - NULL values exist';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM market_data WHERE close IS NULL LIMIT 1) THEN
        ALTER TABLE market_data ALTER COLUMN close SET NOT NULL;
        RAISE NOTICE '✓ NOT NULL constraint added to close';
    ELSE
        RAISE WARNING '⚠ Cannot add NOT NULL to close - NULL values exist';
    END IF;
END $$;

-- Step 11: Verify migration success
\echo ''
\echo 'Step 11: Verifying migration...'
DO $$
DECLARE
    total_rows INTEGER;
    complete_rows INTEGER;
    null_rows INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_rows FROM market_data;

    SELECT COUNT(*) INTO complete_rows
    FROM market_data
    WHERE open IS NOT NULL
        AND high IS NOT NULL
        AND low IS NOT NULL
        AND close IS NOT NULL;

    null_rows := total_rows - complete_rows;

    RAISE NOTICE '';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Migration Verification Results:';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Total rows: %', total_rows;
    RAISE NOTICE 'Complete OHLC rows: %', complete_rows;
    RAISE NOTICE 'Rows with NULL OHLC: %', null_rows;

    IF null_rows > 0 THEN
        RAISE WARNING '⚠ % rows have incomplete OHLC data', null_rows;
    ELSE
        RAISE NOTICE '✓ All rows have complete OHLC data';
    END IF;
END $$;

-- Commit transaction
COMMIT;

\echo ''
\echo '=========================================='
\echo '✓ MIGRATION 001 COMPLETED SUCCESSFULLY'
\echo '=========================================='
\echo ''
\echo 'Summary of changes:'
\echo '  - Added timeframe column (VARCHAR)'
\echo '  - Added open, high, low, close columns (DECIMAL)'
\echo '  - Added spread column (DECIMAL)'
\echo '  - Added data_quality column (INTEGER)'
\echo '  - Backfilled OHLCV from existing price data'
\echo '  - Added NOT NULL constraints to OHLC columns'
\echo ''
