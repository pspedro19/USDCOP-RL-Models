-- ============================================================================
-- V20 MIGRATION SCRIPT
-- USDCOP Trading System
-- ============================================================================
--
-- This script applies all database changes required for V20 fixes.
-- Run with: docker exec -it usdcop-postgres-timescale psql -U admin -d usdcop_trading -f /path/to/v20_migration.sql
--
-- From: 09_Documento Maestro Completo.md
-- Date: 2026-01-09
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. UPDATE MODEL THRESHOLDS (P0 CRITICAL)
-- ============================================================================
-- Change thresholds from 0.3 to 0.10 to match training environment

UPDATE config.models
SET
    threshold_long = 0.10,
    threshold_short = -0.10,
    updated_at = NOW()
WHERE model_id IN ('ppo_v1', 'sac_v19_baseline', 'td3_v19_baseline', 'a2c_v19_baseline');

-- Verify the update
DO $$
DECLARE
    wrong_thresholds INTEGER;
BEGIN
    SELECT COUNT(*) INTO wrong_thresholds
    FROM config.models
    WHERE threshold_long != 0.10 OR threshold_short != -0.10;

    IF wrong_thresholds > 0 THEN
        RAISE NOTICE 'WARNING: % models still have incorrect thresholds', wrong_thresholds;
    ELSE
        RAISE NOTICE 'SUCCESS: All model thresholds updated to 0.10/-0.10';
    END IF;
END $$;

-- ============================================================================
-- 2. CREATE TRADING.MODEL_STATE TABLE (StateTracker Persistence)
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS trading;

CREATE TABLE IF NOT EXISTS trading.model_state (
    model_id VARCHAR(100) PRIMARY KEY,
    position DECIMAL(3,1) DEFAULT 0,
    entry_price DECIMAL(12,4) DEFAULT 0,
    unrealized_pnl DECIMAL(12,4) DEFAULT 0,
    realized_pnl DECIMAL(12,4) DEFAULT 0,
    current_equity DECIMAL(12,4) DEFAULT 10000,
    peak_equity DECIMAL(12,4) DEFAULT 10000,
    current_drawdown DECIMAL(5,4) DEFAULT 0,
    trade_count_session INT DEFAULT 0,
    bars_in_position INT DEFAULT 0,
    last_updated TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_model_state_last_updated
ON trading.model_state(last_updated DESC);

-- Comment
COMMENT ON TABLE trading.model_state IS 'V20: Persistent state storage for RL model StateTracker';

-- ============================================================================
-- 3. ALTER CONFIG.MODELS DEFAULT THRESHOLDS
-- ============================================================================
-- Update default values for future inserts

ALTER TABLE config.models
    ALTER COLUMN threshold_long SET DEFAULT 0.10,
    ALTER COLUMN threshold_short SET DEFAULT -0.10;

-- ============================================================================
-- 4. CREATE V20 AUDIT LOG TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS trading.v20_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    model_id VARCHAR(100),
    details JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Log this migration
INSERT INTO trading.v20_audit_log (event_type, details)
VALUES ('V20_MIGRATION', jsonb_build_object(
    'version', '20.0.0',
    'changes', ARRAY[
        'Updated thresholds to 0.10/-0.10',
        'Created trading.model_state table',
        'Updated config.models defaults'
    ],
    'executed_at', NOW()
));

-- ============================================================================
-- 5. VERIFY MACRO DATA COMPLETENESS
-- ============================================================================

DO $$
DECLARE
    null_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO null_count
    FROM macro_indicators_daily
    WHERE fecha > CURRENT_DATE - 7
      AND (fxrt_index_dxy_usa_d_dxy IS NULL OR volt_vix_usa_d_vix IS NULL);

    IF null_count > 0 THEN
        RAISE NOTICE 'WARNING: % recent dates have NULL macro data (DXY/VIX)', null_count;
        RAISE NOTICE 'Run macro_scraper_robust.py to fill missing data';
    ELSE
        RAISE NOTICE 'SUCCESS: No NULL macro data in last 7 days';
    END IF;
END $$;

-- ============================================================================
-- 6. VERIFY TRADING STATE EXISTS
-- ============================================================================

-- Initialize state for existing models if not present
INSERT INTO trading.model_state (model_id, current_equity, peak_equity)
SELECT model_id, 10000.0, 10000.0
FROM config.models
WHERE enabled = TRUE
ON CONFLICT (model_id) DO NOTHING;

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES (Run manually after migration)
-- ============================================================================

-- Check thresholds
-- SELECT model_id, threshold_long, threshold_short FROM config.models;

-- Check model_state table
-- SELECT * FROM trading.model_state;

-- Check audit log
-- SELECT * FROM trading.v20_audit_log ORDER BY created_at DESC LIMIT 5;

-- Check macro data
-- SELECT fecha, fxrt_index_dxy_usa_d_dxy as dxy, volt_vix_usa_d_vix as vix
-- FROM macro_indicators_daily WHERE fecha > CURRENT_DATE - 7 ORDER BY fecha DESC;

-- ============================================================================
-- END OF V20 MIGRATION
-- ============================================================================
