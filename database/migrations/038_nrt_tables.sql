-- Migration 038: NRT (Near Real-Time) Tables for L1+L5 Integration
-- Created: 2026-02-04
--
-- This migration creates the infrastructure for clean L1+L5 separation:
-- - L1 (Data Layer): Maintains inference_ready_nrt with preprocessed features
-- - L5 (Inference Layer): Reads L1's table, stores signals in inference_signals_nrt
--
-- Key Principle: L5 NEVER does preprocessing. L1 produces a table 100% ready for model.predict()

-- =============================================================================
-- INFERENCE READY TABLE (L1's Output)
-- =============================================================================
-- Contains HISTORICAL data from L2 datasets + NRT updates.
-- 18 market features as array (ready for model.predict)
-- L5 adds 2 state features (position, unrealized_pnl) at inference time.

CREATE TABLE IF NOT EXISTS inference_ready_nrt (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL UNIQUE,
    -- 18 market features as array (already normalized, ready for model.predict)
    -- Feature order matches FEATURE_ORDER[:18] from feature_contract.py
    features FLOAT[18] NOT NULL,
    -- Raw price for signal context and position tracking
    price DECIMAL(12,4) NOT NULL,
    -- Lineage tracking - must match training for correct inference
    feature_order_hash VARCHAR(64) NOT NULL,
    norm_stats_hash VARCHAR(64) NOT NULL,
    -- Source: 'historical' (from L2 datasets) or 'nrt' (real-time updates)
    source VARCHAR(20) DEFAULT 'nrt',
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast timestamp lookups (most recent first)
CREATE INDEX IF NOT EXISTS idx_inference_ready_nrt_ts
    ON inference_ready_nrt(timestamp DESC);

-- Index for source filtering (debugging/auditing)
CREATE INDEX IF NOT EXISTS idx_inference_ready_nrt_source
    ON inference_ready_nrt(source);

-- Comment documentation
COMMENT ON TABLE inference_ready_nrt IS
    'L1 Data Layer output: 18 market features normalized and ready for model.predict(). L5 only reads this table.';

COMMENT ON COLUMN inference_ready_nrt.features IS
    'Array of 18 normalized market features in FEATURE_ORDER. Clip range: [-5.0, 5.0]';

COMMENT ON COLUMN inference_ready_nrt.feature_order_hash IS
    'Hash of FEATURE_ORDER used - must match training for correct inference';

COMMENT ON COLUMN inference_ready_nrt.norm_stats_hash IS
    'Hash of norm_stats.json used - must match training for correct normalization';

COMMENT ON COLUMN inference_ready_nrt.source IS
    'historical = loaded from L2 datasets, nrt = computed in real-time by L1';


-- =============================================================================
-- INFERENCE SIGNALS TABLE (L5's Output)
-- =============================================================================
-- L5 writes inference results here after reading inference_ready_nrt

CREATE TABLE IF NOT EXISTS inference_signals_nrt (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(100) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    -- Trading signal
    signal VARCHAR(20) NOT NULL,  -- LONG, SHORT, HOLD
    raw_action DECIMAL(10,6),     -- Continuous action from model
    confidence DECIMAL(5,4),      -- Signal confidence [0,1]
    -- Context
    price DECIMAL(12,4) NOT NULL,
    -- Performance tracking
    latency_ms DECIMAL(10,3),     -- Inference latency
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT inference_signals_nrt_signal_check
        CHECK (signal IN ('LONG', 'SHORT', 'HOLD'))
);

-- Index for fast model+timestamp lookups
CREATE INDEX IF NOT EXISTS idx_inference_signals_nrt_ts
    ON inference_signals_nrt(model_id, timestamp DESC);

-- Index for recent signals queries
CREATE INDEX IF NOT EXISTS idx_inference_signals_nrt_recent
    ON inference_signals_nrt(created_at DESC);

-- Comment documentation
COMMENT ON TABLE inference_signals_nrt IS
    'L5 Inference Layer output: Trading signals from model.predict(). WebSocket broadcasts from here.';

COMMENT ON COLUMN inference_signals_nrt.raw_action IS
    'Continuous action value from PPO model before discretization';

COMMENT ON COLUMN inference_signals_nrt.latency_ms IS
    'Time from feature read to signal store, in milliseconds';


-- =============================================================================
-- MODEL APPROVAL NOTIFICATION TRIGGER
-- =============================================================================
-- When a model is promoted to 'production', notify L1 and L5 services
-- This trigger fires on model_registry.stage change to 'production'

CREATE OR REPLACE FUNCTION notify_model_approved()
RETURNS TRIGGER AS $$
BEGIN
    -- Only fire when stage changes TO 'production'
    IF NEW.stage = 'production' AND (OLD.stage IS NULL OR OLD.stage != 'production') THEN
        PERFORM pg_notify('model_approved', jsonb_build_object(
            'model_id', NEW.model_id,
            'model_path', NEW.model_path,
            'norm_stats_path', COALESCE(NEW.norm_stats_path, ''),
            'dataset_path', COALESCE(NEW.lineage->>'dataset_path', ''),
            'feature_order_hash', COALESCE(NEW.feature_order_hash, ''),
            'norm_stats_hash', COALESCE(NEW.norm_stats_hash, ''),
            'experiment_name', COALESCE(NEW.experiment_name, ''),
            'approved_by', COALESCE(NEW.approved_by, ''),
            'approved_at', COALESCE(NEW.approved_at::TEXT, NOW()::TEXT)
        )::TEXT);

        -- Log the notification
        RAISE NOTICE 'model_approved notification sent for model: %', NEW.model_id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to model_registry (if not exists)
DROP TRIGGER IF EXISTS trg_notify_model_approved ON model_registry;
CREATE TRIGGER trg_notify_model_approved
    AFTER UPDATE ON model_registry
    FOR EACH ROW EXECUTE FUNCTION notify_model_approved();

COMMENT ON FUNCTION notify_model_approved() IS
    'Sends pg_notify on model_approved channel when model stage changes to production';


-- =============================================================================
-- FEATURES READY NOTIFICATION TRIGGER
-- =============================================================================
-- When L1 inserts a new row into inference_ready_nrt, notify L5 to run inference

CREATE OR REPLACE FUNCTION notify_features_ready()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify('features_ready', jsonb_build_object(
        'timestamp', NEW.timestamp,
        'price', NEW.price,
        'feature_order_hash', NEW.feature_order_hash,
        'source', NEW.source
    )::TEXT);

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Attach trigger to inference_ready_nrt
DROP TRIGGER IF EXISTS trg_notify_features_ready ON inference_ready_nrt;
CREATE TRIGGER trg_notify_features_ready
    AFTER INSERT ON inference_ready_nrt
    FOR EACH ROW EXECUTE FUNCTION notify_features_ready();

COMMENT ON FUNCTION notify_features_ready() IS
    'Sends pg_notify on features_ready channel when L1 inserts new features';


-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- View for latest inference state (useful for dashboard)
CREATE OR REPLACE VIEW v_latest_inference_state AS
SELECT
    s.model_id,
    s.timestamp,
    s.signal,
    s.raw_action,
    s.confidence,
    s.price,
    s.latency_ms,
    s.created_at,
    f.features,
    f.feature_order_hash,
    f.norm_stats_hash
FROM inference_signals_nrt s
JOIN inference_ready_nrt f ON s.timestamp = f.timestamp
WHERE s.id = (SELECT MAX(id) FROM inference_signals_nrt);

COMMENT ON VIEW v_latest_inference_state IS
    'Latest inference signal joined with its features for debugging';


-- View for NRT signal history (last 100)
CREATE OR REPLACE VIEW v_nrt_signal_history AS
SELECT
    s.model_id,
    s.timestamp,
    s.signal,
    s.confidence,
    s.price,
    s.latency_ms,
    f.source as feature_source
FROM inference_signals_nrt s
LEFT JOIN inference_ready_nrt f ON s.timestamp = f.timestamp
ORDER BY s.timestamp DESC
LIMIT 100;

COMMENT ON VIEW v_nrt_signal_history IS
    'Recent 100 NRT signals for dashboard display';


-- =============================================================================
-- CLEANUP FUNCTIONS
-- =============================================================================

-- Function to clean old NRT data (keep last N days)
CREATE OR REPLACE FUNCTION cleanup_nrt_data(days_to_keep INTEGER DEFAULT 30)
RETURNS TABLE(features_deleted BIGINT, signals_deleted BIGINT) AS $$
DECLARE
    cutoff_date TIMESTAMPTZ;
    feat_count BIGINT;
    sig_count BIGINT;
BEGIN
    cutoff_date := NOW() - (days_to_keep || ' days')::INTERVAL;

    -- Delete old features
    DELETE FROM inference_ready_nrt
    WHERE timestamp < cutoff_date AND source = 'nrt';
    GET DIAGNOSTICS feat_count = ROW_COUNT;

    -- Delete old signals
    DELETE FROM inference_signals_nrt
    WHERE timestamp < cutoff_date;
    GET DIAGNOSTICS sig_count = ROW_COUNT;

    RETURN QUERY SELECT feat_count, sig_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_nrt_data(INTEGER) IS
    'Removes NRT data older than specified days. Historical data is preserved.';
