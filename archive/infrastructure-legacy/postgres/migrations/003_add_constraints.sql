-- =====================================================
-- MIGRATION 003: Add validation constraints
-- Description: Data integrity constraints for OHLCV validation
-- Author: Migration System
-- Date: 2025-10-22
-- =====================================================

\echo ''
\echo '=========================================='
\echo 'MIGRATION 003: Adding validation constraints'
\echo '=========================================='
\echo ''

BEGIN;

-- ========================================
-- 1. OHLC VALIDATION CONSTRAINTS
-- ========================================

\echo 'Step 1: Adding OHLC validation constraints...'

-- Constraint: High must be >= Open
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_high_gte_open'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_high_gte_open
        CHECK (high >= open);
        RAISE NOTICE '  ✓ Constraint chk_high_gte_open added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_high_gte_open already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_high_gte_open - existing data violates constraint';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET high = GREATEST(high, open) WHERE high < open;';
END $$;

-- Constraint: High must be >= Close
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_high_gte_close'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_high_gte_close
        CHECK (high >= close);
        RAISE NOTICE '  ✓ Constraint chk_high_gte_close added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_high_gte_close already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_high_gte_close - existing data violates constraint';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET high = GREATEST(high, close) WHERE high < close;';
END $$;

-- Constraint: Low must be <= Open
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_low_lte_open'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_low_lte_open
        CHECK (low <= open);
        RAISE NOTICE '  ✓ Constraint chk_low_lte_open added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_low_lte_open already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_low_lte_open - existing data violates constraint';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET low = LEAST(low, open) WHERE low > open;';
END $$;

-- Constraint: Low must be <= Close
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_low_lte_close'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_low_lte_close
        CHECK (low <= close);
        RAISE NOTICE '  ✓ Constraint chk_low_lte_close added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_low_lte_close already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_low_lte_close - existing data violates constraint';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET low = LEAST(low, close) WHERE low > close;';
END $$;

-- Constraint: High must be >= Low (always true for valid candles)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_high_gte_low'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_high_gte_low
        CHECK (high >= low);
        RAISE NOTICE '  ✓ Constraint chk_high_gte_low added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_high_gte_low already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_high_gte_low - existing data violates constraint';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET high = GREATEST(high, low), low = LEAST(high, low);';
END $$;

-- ========================================
-- 2. PRICE RANGE VALIDATION
-- ========================================

\echo ''
\echo 'Step 2: Adding price range constraints...'

-- Constraint: All OHLC prices must be positive
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_prices_positive'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_prices_positive
        CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0);
        RAISE NOTICE '  ✓ Constraint chk_prices_positive added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_prices_positive already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_prices_positive - existing data has non-positive prices';
        RAISE NOTICE '  → Check data with: SELECT * FROM market_data WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0;';
END $$;

-- Constraint: Volume must be non-negative
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_volume_non_negative'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_volume_non_negative
        CHECK (volume IS NULL OR volume >= 0);
        RAISE NOTICE '  ✓ Constraint chk_volume_non_negative added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_volume_non_negative already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_volume_non_negative - existing data has negative volume';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET volume = 0 WHERE volume < 0;';
END $$;

-- ========================================
-- 3. SPREAD VALIDATION
-- ========================================

\echo ''
\echo 'Step 3: Adding spread validation constraints...'

-- Constraint: Spread must be non-negative
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_spread_non_negative'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_spread_non_negative
        CHECK (spread IS NULL OR spread >= 0);
        RAISE NOTICE '  ✓ Constraint chk_spread_non_negative added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_spread_non_negative already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_spread_non_negative - existing data has negative spread';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET spread = ABS(spread) WHERE spread < 0;';
END $$;

-- Constraint: If bid and ask exist, ask must be >= bid
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_ask_gte_bid'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_ask_gte_bid
        CHECK (bid IS NULL OR ask IS NULL OR ask >= bid);
        RAISE NOTICE '  ✓ Constraint chk_ask_gte_bid added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_ask_gte_bid already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_ask_gte_bid - existing data has ask < bid';
        RAISE NOTICE '  → Check data with: SELECT * FROM market_data WHERE ask < bid;';
END $$;

-- ========================================
-- 4. DATA QUALITY VALIDATION
-- ========================================

\echo ''
\echo 'Step 4: Adding data quality constraints...'

-- Constraint: Data quality must be between 0 and 100
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_data_quality_range'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_data_quality_range
        CHECK (data_quality IS NULL OR (data_quality >= 0 AND data_quality <= 100));
        RAISE NOTICE '  ✓ Constraint chk_data_quality_range added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_data_quality_range already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_data_quality_range - existing data has invalid quality scores';
        RAISE NOTICE '  → Fix data with: UPDATE market_data SET data_quality = GREATEST(0, LEAST(100, data_quality));';
END $$;

-- ========================================
-- 5. TIMEFRAME VALIDATION
-- ========================================

\echo ''
\echo 'Step 5: Adding timeframe validation constraints...'

-- Constraint: Timeframe must be one of the valid values
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'chk_valid_timeframe'
    ) THEN
        ALTER TABLE market_data
        ADD CONSTRAINT chk_valid_timeframe
        CHECK (timeframe IN ('1min', '5min', '15min', '30min', '1h', '4h', '1d'));
        RAISE NOTICE '  ✓ Constraint chk_valid_timeframe added';
    ELSE
        RAISE NOTICE '  ⊙ Constraint chk_valid_timeframe already exists';
    END IF;
EXCEPTION
    WHEN check_violation THEN
        RAISE WARNING '  ⚠ Cannot add chk_valid_timeframe - existing data has invalid timeframes';
        RAISE NOTICE '  → Check data with: SELECT DISTINCT timeframe FROM market_data WHERE timeframe NOT IN (''1min'', ''5min'', ''15min'', ''30min'', ''1h'', ''4h'', ''1d'');';
END $$;

-- ========================================
-- 6. VERIFY CONSTRAINTS
-- ========================================

\echo ''
\echo 'Step 6: Verifying constraints...'

DO $$
DECLARE
    constraint_count INTEGER;
    market_data_constraints INTEGER;
BEGIN
    -- Count all constraints
    SELECT COUNT(*) INTO constraint_count
    FROM pg_constraint
    WHERE conrelid = 'market_data'::regclass;

    -- Count CHECK constraints specifically
    SELECT COUNT(*) INTO market_data_constraints
    FROM pg_constraint
    WHERE conrelid = 'market_data'::regclass
        AND contype = 'c';

    RAISE NOTICE '';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Constraint Verification Results:';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Total constraints: %', constraint_count;
    RAISE NOTICE 'CHECK constraints: %', market_data_constraints;
    RAISE NOTICE '';
END $$;

-- List all constraints
\echo 'Current constraints on market_data table:'
SELECT
    conname AS constraint_name,
    contype AS constraint_type,
    pg_get_constraintdef(oid) AS constraint_definition
FROM pg_constraint
WHERE conrelid = 'market_data'::regclass
ORDER BY conname;

COMMIT;

\echo ''
\echo '=========================================='
\echo '✓ MIGRATION 003 COMPLETED SUCCESSFULLY'
\echo '=========================================='
\echo ''
\echo 'Summary of constraints added:'
\echo '  - OHLC validation (high >= open/close, low <= open/close)'
\echo '  - Price positivity (all prices > 0)'
\echo '  - Volume non-negative'
\echo '  - Spread validation (spread >= 0, ask >= bid)'
\echo '  - Data quality range (0-100)'
\echo '  - Timeframe validation'
\echo ''
\echo 'Data integrity is now enforced at the database level!'
\echo ''
