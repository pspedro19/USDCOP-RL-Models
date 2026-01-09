-- ============================================================================
-- V20 MIGRATION SCRIPT (FIXED)
-- USDCOP Trading System
-- ============================================================================
--
-- This script applies all database changes required for V20 fixes.
-- Fixed: Uses JSONB fields instead of non-existent columns
--
-- Date: 2026-01-09
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1. UPDATE MODEL THRESHOLDS IN ENVIRONMENT_CONFIG (JSONB)
-- ============================================================================
-- Add threshold_long and threshold_short to environment_config

UPDATE config.models
SET
    environment_config = environment_config ||
        '{"threshold_long": 0.10, "threshold_short": -0.10}'::jsonb,
    updated_at = NOW()
WHERE model_id IN ('ppo_v1', 'sac_v19_baseline', 'td3_v19_baseline', 'a2c_v19_baseline');

-- Verify the update
DO $$
DECLARE
    updated_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO updated_count
    FROM config.models
    WHERE (environment_config->>'threshold_long')::numeric = 0.10;

    RAISE NOTICE 'SUCCESS: % models updated with threshold 0.10', updated_count;
END $$;

-- ============================================================================
-- 2. CREATE TRADING SCHEMA IF NOT EXISTS
-- ============================================================================

CREATE SCHEMA IF NOT EXISTS trading;

-- ============================================================================
-- 3. CREATE TRADING.MODEL_STATE TABLE (StateTracker Persistence)
-- ============================================================================
-- Only create if it doesn't exist

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables
                   WHERE table_schema = 'trading' AND table_name = 'model_state') THEN
        CREATE TABLE trading.model_state (
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

        CREATE INDEX idx_model_state_last_updated
        ON trading.model_state(last_updated DESC);

        RAISE NOTICE 'SUCCESS: Created trading.model_state table';
    ELSE
        RAISE NOTICE 'INFO: trading.model_state table already exists';
    END IF;
END $$;

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
        'Added thresholds 0.10/-0.10 to environment_config',
        'Created/verified trading.model_state table',
        'Created trading.v20_audit_log table'
    ],
    'executed_at', NOW()
));

-- ============================================================================
-- 5. INITIALIZE MODEL STATE FOR ACTIVE MODELS
-- ============================================================================

INSERT INTO trading.model_state (model_id, current_equity, peak_equity)
SELECT model_id, 10000.0, 10000.0
FROM config.models
WHERE status = 'active'
ON CONFLICT (model_id) DO NOTHING;

-- ============================================================================
-- 6. VERIFY MACRO DATA COMPLETENESS
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
    ELSE
        RAISE NOTICE 'SUCCESS: No NULL macro data in last 7 days';
    END IF;
END $$;

COMMIT;

-- ============================================================================
-- VERIFICATION QUERIES (Results shown below)
-- ============================================================================

-- Show updated thresholds
SELECT
    model_id,
    environment_config->>'threshold_long' as threshold_long,
    environment_config->>'threshold_short' as threshold_short
FROM config.models;

-- Show model_state table
SELECT * FROM trading.model_state;

-- Show audit log
SELECT * FROM trading.v20_audit_log ORDER BY created_at DESC LIMIT 3;

-- ============================================================================
-- END OF V20 MIGRATION
-- ============================================================================
