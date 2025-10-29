-- =====================================================
-- MIGRATION 002: Add optimized indexes for OHLCV data
-- Description: Time-series optimized indexes for candlestick queries
-- Author: Migration System
-- Date: 2025-10-22
-- =====================================================

\echo ''
\echo '=========================================='
\echo 'MIGRATION 002: Creating optimized indexes'
\echo '=========================================='
\echo ''

BEGIN;

-- ========================================
-- 1. PRIMARY COMPOSITE INDEXES
-- ========================================

\echo 'Step 1: Creating composite indexes...'

-- Composite index for symbol, timeframe, timestamp (main query pattern)
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_timeframe_time
ON market_data (symbol, timeframe, timestamp DESC);

\echo '  ✓ idx_market_data_symbol_timeframe_time'

-- Index for source-based queries
CREATE INDEX IF NOT EXISTS idx_market_data_source_time
ON market_data (source, timestamp DESC);

\echo '  ✓ idx_market_data_source_time'

-- ========================================
-- 2. PARTIAL INDEXES (Hot Data)
-- ========================================

\echo ''
\echo 'Step 2: Creating partial indexes for hot data...'

-- Hot data index: Last 7 days of trading session data
CREATE INDEX IF NOT EXISTS idx_market_data_recent_trading
ON market_data (timestamp DESC, close, volume)
WHERE timestamp >= (NOW() - INTERVAL '7 days');

\echo '  ✓ idx_market_data_recent_trading (last 7 days)'

-- Index for intraday queries (5min timeframe only)
CREATE INDEX IF NOT EXISTS idx_market_data_intraday_5min
ON market_data (timestamp DESC, open, high, low, close, volume)
WHERE timeframe = '5min'
    AND timestamp >= (NOW() - INTERVAL '30 days');

\echo '  ✓ idx_market_data_intraday_5min (last 30 days)'

-- ========================================
-- 3. BRIN INDEXES (Time-series optimization)
-- ========================================

\echo ''
\echo 'Step 3: Creating BRIN indexes for large sequential data...'

-- BRIN index for timestamp (efficient for time-series data)
CREATE INDEX IF NOT EXISTS idx_market_data_timestamp_brin
ON market_data USING BRIN (timestamp)
WITH (pages_per_range = 128);

\echo '  ✓ idx_market_data_timestamp_brin (BRIN)'

-- BRIN index for OHLC values (range queries)
CREATE INDEX IF NOT EXISTS idx_market_data_ohlc_brin
ON market_data USING BRIN (open, high, low, close)
WITH (pages_per_range = 128);

\echo '  ✓ idx_market_data_ohlc_brin (BRIN)'

-- ========================================
-- 4. GAP DETECTION INDEXES
-- ========================================

\echo ''
\echo 'Step 4: Creating indexes for gap detection...'

-- Index optimized for detecting missing timestamps
CREATE INDEX IF NOT EXISTS idx_market_data_gap_detection
ON market_data (symbol, timeframe, timestamp)
WHERE timestamp >= (NOW() - INTERVAL '90 days');

\echo '  ✓ idx_market_data_gap_detection'

-- Index for data completeness checks
CREATE INDEX IF NOT EXISTS idx_market_data_completeness
ON market_data (symbol, timeframe, DATE(timestamp))
WHERE timestamp >= (NOW() - INTERVAL '180 days');

\echo '  ✓ idx_market_data_completeness'

-- ========================================
-- 5. DATA QUALITY INDEXES
-- ========================================

\echo ''
\echo 'Step 5: Creating indexes for data quality monitoring...'

-- Index for data quality issues
CREATE INDEX IF NOT EXISTS idx_market_data_quality_issues
ON market_data (data_quality, timestamp DESC)
WHERE data_quality < 100;

\echo '  ✓ idx_market_data_quality_issues'

-- Index for zero volume detection
CREATE INDEX IF NOT EXISTS idx_market_data_zero_volume
ON market_data (timestamp DESC)
WHERE volume IS NULL OR volume = 0;

\echo '  ✓ idx_market_data_zero_volume'

-- Index for spread analysis
CREATE INDEX IF NOT EXISTS idx_market_data_spread_analysis
ON market_data (spread, timestamp DESC)
WHERE spread > 0;

\echo '  ✓ idx_market_data_spread_analysis'

-- ========================================
-- 6. COVERING INDEXES (Include columns)
-- ========================================

\echo ''
\echo 'Step 6: Creating covering indexes...'

-- Covering index for common OHLC queries
CREATE INDEX IF NOT EXISTS idx_market_data_ohlc_covering
ON market_data (symbol, timestamp DESC)
INCLUDE (open, high, low, close, volume);

\echo '  ✓ idx_market_data_ohlc_covering'

-- ========================================
-- 7. ANALYZE TABLE
-- ========================================

\echo ''
\echo 'Step 7: Analyzing table for query planner statistics...'

ANALYZE market_data;

\echo '  ✓ Table analyzed'

-- ========================================
-- 8. VERIFY INDEXES
-- ========================================

\echo ''
\echo 'Step 8: Verifying index creation...'

DO $$
DECLARE
    index_count INTEGER;
    index_size TEXT;
BEGIN
    -- Count indexes on market_data
    SELECT COUNT(*) INTO index_count
    FROM pg_indexes
    WHERE tablename = 'market_data';

    -- Get total index size
    SELECT pg_size_pretty(SUM(pg_relation_size(indexrelid)))
    INTO index_size
    FROM pg_index
    JOIN pg_class ON pg_class.oid = pg_index.indexrelid
    WHERE pg_index.indrelid = 'market_data'::regclass;

    RAISE NOTICE '';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Index Verification Results:';
    RAISE NOTICE '====================================';
    RAISE NOTICE 'Total indexes on market_data: %', index_count;
    RAISE NOTICE 'Total index size: %', index_size;
    RAISE NOTICE '';
END $$;

-- List all indexes
\echo 'Current indexes on market_data table:'
SELECT
    indexname,
    indexdef
FROM pg_indexes
WHERE tablename = 'market_data'
ORDER BY indexname;

COMMIT;

\echo ''
\echo '=========================================='
\echo '✓ MIGRATION 002 COMPLETED SUCCESSFULLY'
\echo '=========================================='
\echo ''
\echo 'Summary of indexes created:'
\echo '  - Composite indexes for common queries'
\echo '  - Partial indexes for hot data (7-30 days)'
\echo '  - BRIN indexes for time-series efficiency'
\echo '  - Gap detection indexes'
\echo '  - Data quality monitoring indexes'
\echo '  - Covering indexes for OHLC queries'
\echo ''
\echo 'Query performance should be significantly improved!'
\echo ''
