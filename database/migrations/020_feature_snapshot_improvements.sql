-- ============================================================================
-- 020_feature_snapshot_improvements.sql
-- Week 2: Feature Snapshot GIN Index and Expanded View
-- Contract: CTR-004
-- Author: Trading Team
-- Date: 2025-01-17
-- ============================================================================
--
-- This migration adds:
-- 1. GIN index for fast JSONB queries on features_snapshot
-- 2. features_source column for tracking feature origin
-- 3. Expanded view for easier feature analysis
--
-- Prerequisites:
--   - PostgreSQL 14+
--   - trades_history table with features_snapshot JSONB column
--
-- Performance Notes:
--   - GIN index optimizes @>, ?, ?|, ?& operators on JSONB
--   - CONCURRENTLY flag prevents table locks during index creation
--   - Index size ~10-20% of JSONB column size
-- ============================================================================

-- 1. GIN index for features_snapshot JSONB
-- Enables fast queries like:
--   - WHERE features_snapshot @> '{"rsi_9": 30}'
--   - WHERE features_snapshot ? 'log_ret_5m'
--   - WHERE features_snapshot ?& array['rsi_9', 'adx_14']
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_features_snapshot_gin
ON trades_history USING GIN (features_snapshot);

-- 2. Add features_source column to track origin of features
-- Values: 'l1_pipeline', 'feast', 'fallback', 'manual'
ALTER TABLE trades_history
ADD COLUMN IF NOT EXISTS features_source VARCHAR(50) DEFAULT 'l1_pipeline';

-- Add comment for documentation
COMMENT ON COLUMN trades_history.features_source IS
'Source of feature data: l1_pipeline (default), feast, fallback, manual';

-- Index for filtering by source
CREATE INDEX IF NOT EXISTS idx_trades_features_source
ON trades_history(features_source);

-- 3. Create expanded view for easier feature analysis
-- Extracts commonly accessed features from JSONB into columns
CREATE OR REPLACE VIEW trade_features_expanded AS
SELECT
    t.id,
    t.timestamp,
    t.signal,
    t.pnl,
    t.model_id,
    t.model_hash,
    t.features_source,

    -- Extract feature timestamp from snapshot
    (t.features_snapshot->>'timestamp')::timestamp AS feature_timestamp,

    -- Technical indicators (from normalized_features or flat structure)
    COALESCE(
        (t.features_snapshot->'normalized_features'->>'log_ret_5m')::numeric,
        (t.features_snapshot->>'log_ret_5m')::numeric
    ) AS log_ret_5m,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'log_ret_1h')::numeric,
        (t.features_snapshot->>'log_ret_1h')::numeric
    ) AS log_ret_1h,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'log_ret_4h')::numeric,
        (t.features_snapshot->>'log_ret_4h')::numeric
    ) AS log_ret_4h,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'rsi_9')::numeric,
        (t.features_snapshot->>'rsi_9')::numeric
    ) AS rsi_9,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'atr_pct')::numeric,
        (t.features_snapshot->>'atr_pct')::numeric
    ) AS atr_pct,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'adx_14')::numeric,
        (t.features_snapshot->>'adx_14')::numeric
    ) AS adx_14,

    -- Macro features
    COALESCE(
        (t.features_snapshot->'normalized_features'->>'dxy_z')::numeric,
        (t.features_snapshot->>'dxy_z')::numeric
    ) AS dxy_z,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'dxy_change_1d')::numeric,
        (t.features_snapshot->>'dxy_change_1d')::numeric
    ) AS dxy_change_1d,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'vix_z')::numeric,
        (t.features_snapshot->>'vix_z')::numeric
    ) AS vix_z,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'embi_z')::numeric,
        (t.features_snapshot->>'embi_z')::numeric
    ) AS embi_z,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'brent_change_1d')::numeric,
        (t.features_snapshot->>'brent_change_1d')::numeric
    ) AS brent_change_1d,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'rate_spread')::numeric,
        (t.features_snapshot->>'rate_spread')::numeric
    ) AS rate_spread,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'usdmxn_change_1d')::numeric,
        (t.features_snapshot->>'usdmxn_change_1d')::numeric
    ) AS usdmxn_change_1d,

    -- State features
    COALESCE(
        (t.features_snapshot->'normalized_features'->>'position')::numeric,
        (t.features_snapshot->>'position')::numeric
    ) AS position,

    COALESCE(
        (t.features_snapshot->'normalized_features'->>'time_normalized')::numeric,
        (t.features_snapshot->>'time_normalized')::numeric
    ) AS time_normalized,

    -- Metadata
    (t.features_snapshot->>'version') AS snapshot_version,
    (t.features_snapshot->>'bar_idx')::integer AS bar_idx

FROM trades_history t
WHERE t.features_snapshot IS NOT NULL;

-- Comment on view
COMMENT ON VIEW trade_features_expanded IS
'Expanded view of trades with feature columns extracted from features_snapshot JSONB.
Supports both old (flat) and new (normalized_features) schema versions.
Contract: CTR-004';

-- 4. Create index on common query patterns
-- Useful for queries filtering by signal + timestamp
CREATE INDEX IF NOT EXISTS idx_trades_signal_timestamp
ON trades_history(signal, timestamp DESC);

-- 5. Create partial index for trades with features (optimization)
CREATE INDEX IF NOT EXISTS idx_trades_with_features
ON trades_history(timestamp DESC)
WHERE features_snapshot IS NOT NULL;

-- 6. Verification
DO $$
DECLARE
    gin_exists BOOLEAN;
    view_exists BOOLEAN;
    col_exists BOOLEAN;
BEGIN
    -- Check GIN index
    SELECT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE indexname = 'idx_trades_features_snapshot_gin'
    ) INTO gin_exists;

    -- Check view
    SELECT EXISTS (
        SELECT 1 FROM information_schema.views
        WHERE table_name = 'trade_features_expanded'
    ) INTO view_exists;

    -- Check column
    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'trades_history'
          AND column_name = 'features_source'
    ) INTO col_exists;

    IF gin_exists THEN
        RAISE NOTICE 'GIN index idx_trades_features_snapshot_gin created successfully';
    ELSE
        RAISE WARNING 'GIN index idx_trades_features_snapshot_gin NOT created';
    END IF;

    IF view_exists THEN
        RAISE NOTICE 'View trade_features_expanded created successfully';
    ELSE
        RAISE WARNING 'View trade_features_expanded NOT created';
    END IF;

    IF col_exists THEN
        RAISE NOTICE 'Column features_source added successfully';
    ELSE
        RAISE WARNING 'Column features_source NOT added';
    END IF;
END $$;

-- ============================================================================
-- SAMPLE QUERIES (for documentation)
-- ============================================================================
--
-- Query features for a specific trade:
--   SELECT * FROM trade_features_expanded WHERE id = 123;
--
-- Find trades with high RSI:
--   SELECT * FROM trade_features_expanded WHERE rsi_9 > 0.7;
--
-- Find trades using JSONB containment:
--   SELECT * FROM trades_history
--   WHERE features_snapshot @> '{"signal": "LONG"}';
--
-- Check feature source distribution:
--   SELECT features_source, COUNT(*)
--   FROM trades_history
--   GROUP BY features_source;
--
-- ============================================================================

SELECT 'Migration 020_feature_snapshot_improvements.sql completed' AS status;
