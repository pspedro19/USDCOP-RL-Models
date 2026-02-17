-- =============================================================================
-- Migration: 033_event_triggers.sql
-- Description: PostgreSQL NOTIFY triggers for V7.1 Event-Driven Architecture
-- Author: Trading Team
-- Date: 2026-01-31
-- Contract: CTR-V7-EVENTS
-- =============================================================================

-- =============================================================================
-- OHLCV NOTIFY TRIGGER
-- =============================================================================
-- Fires NOTIFY when new OHLCV bar is inserted

CREATE OR REPLACE FUNCTION notify_new_ohlcv_bar()
RETURNS TRIGGER AS $$
DECLARE
    payload JSONB;
BEGIN
    -- Build payload with essential info only (NOTIFY has 8KB limit)
    payload := jsonb_build_object(
        'event_type', 'new_bar',
        'table', TG_TABLE_NAME,
        'symbol', NEW.symbol,
        'time', NEW.time::TEXT,
        'close', NEW.close,
        'volume', NEW.volume,
        'event_id', md5(NEW.symbol || NEW.time::TEXT),
        'triggered_at', NOW()::TEXT
    );

    -- Send notification
    PERFORM pg_notify('ohlcv_updates', payload::TEXT);

    -- Log for debugging (remove in production if too verbose)
    RAISE DEBUG 'NOTIFY ohlcv_updates: %', payload::TEXT;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on usdcop_m5_ohlcv
DROP TRIGGER IF EXISTS trg_notify_new_ohlcv_bar ON usdcop_m5_ohlcv;
CREATE TRIGGER trg_notify_new_ohlcv_bar
    AFTER INSERT ON usdcop_m5_ohlcv
    FOR EACH ROW
    EXECUTE FUNCTION notify_new_ohlcv_bar();

-- =============================================================================
-- FEATURE NOTIFY TRIGGER
-- =============================================================================
-- Fires NOTIFY when new features are ready

CREATE OR REPLACE FUNCTION notify_features_ready()
RETURNS TRIGGER AS $$
DECLARE
    payload JSONB;
    is_complete BOOLEAN;
BEGIN
    -- Check if critical features are complete
    is_complete := (
        NEW.log_ret_5m IS NOT NULL AND
        NEW.rsi_9 IS NOT NULL AND
        NEW.dxy_z IS NOT NULL
    );

    -- Only notify if features are complete
    IF is_complete THEN
        payload := jsonb_build_object(
            'event_type', 'features_ready',
            'table', TG_TABLE_NAME,
            'time', NEW.time::TEXT,
            'feature_count', 15,
            'is_complete', is_complete,
            'event_id', md5(NEW.time::TEXT || 'features'),
            'triggered_at', NOW()::TEXT
        );

        PERFORM pg_notify('feature_updates', payload::TEXT);
        RAISE DEBUG 'NOTIFY feature_updates: %', payload::TEXT;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger on inference_features_5m
DROP TRIGGER IF EXISTS trg_notify_features_ready ON inference_features_5m;
CREATE TRIGGER trg_notify_features_ready
    AFTER INSERT OR UPDATE ON inference_features_5m
    FOR EACH ROW
    EXECUTE FUNCTION notify_features_ready();

-- =============================================================================
-- HEARTBEAT FUNCTION
-- =============================================================================
-- Used by HeartbeatMonitor to verify NOTIFY system health

CREATE OR REPLACE FUNCTION emit_heartbeat(channel TEXT DEFAULT 'heartbeat')
RETURNS VOID AS $$
DECLARE
    payload JSONB;
BEGIN
    payload := jsonb_build_object(
        'event_type', 'heartbeat',
        'timestamp', NOW()::TEXT,
        'server_id', inet_server_addr()::TEXT
    );

    PERFORM pg_notify(channel, payload::TEXT);
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- DEAD LETTER QUEUE TABLE
-- =============================================================================
-- Stores failed events for retry

CREATE TABLE IF NOT EXISTS event_dead_letter_queue (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    channel VARCHAR(100) NOT NULL,
    payload JSONB NOT NULL,
    error_message TEXT,
    error_count INTEGER DEFAULT 1,
    first_failed_at TIMESTAMPTZ DEFAULT NOW(),
    last_failed_at TIMESTAMPTZ DEFAULT NOW(),
    retry_after TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'pending',  -- pending, processing, resolved, dead
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),

    CONSTRAINT check_status CHECK (status IN ('pending', 'processing', 'resolved', 'dead'))
);

CREATE INDEX IF NOT EXISTS idx_dlq_status ON event_dead_letter_queue(status);
CREATE INDEX IF NOT EXISTS idx_dlq_retry ON event_dead_letter_queue(retry_after)
    WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_dlq_event_type ON event_dead_letter_queue(event_type);

-- =============================================================================
-- EVENT IDEMPOTENCY TABLE
-- =============================================================================
-- Tracks processed events to ensure exactly-once processing

CREATE TABLE IF NOT EXISTS event_processed_log (
    event_id VARCHAR(64) PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    payload_hash VARCHAR(64) NOT NULL,
    processed_at TIMESTAMPTZ DEFAULT NOW(),
    processor_id VARCHAR(100),
    result_status VARCHAR(20) DEFAULT 'success',

    -- Auto-cleanup: keep 7 days
    CONSTRAINT check_result CHECK (result_status IN ('success', 'failed', 'skipped'))
);

CREATE INDEX IF NOT EXISTS idx_processed_at ON event_processed_log(processed_at DESC);

-- Auto-cleanup function for old processed events
CREATE OR REPLACE FUNCTION cleanup_old_processed_events()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM event_processed_log
    WHERE processed_at < NOW() - INTERVAL '7 days';

    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- CIRCUIT BREAKER STATE TABLE
-- =============================================================================
-- Persists circuit breaker state across restarts

CREATE TABLE IF NOT EXISTS circuit_breaker_state (
    circuit_id VARCHAR(100) PRIMARY KEY,
    state VARCHAR(20) DEFAULT 'CLOSED',  -- CLOSED, OPEN, HALF_OPEN
    failure_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_failure_at TIMESTAMPTZ,
    last_success_at TIMESTAMPTZ,
    opened_at TIMESTAMPTZ,
    half_open_at TIMESTAMPTZ,
    reset_timeout_seconds INTEGER DEFAULT 300,
    failure_threshold INTEGER DEFAULT 3,
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT check_state CHECK (state IN ('CLOSED', 'OPEN', 'HALF_OPEN'))
);

-- =============================================================================
-- MONITORING VIEWS
-- =============================================================================

-- View: Current system event health
CREATE OR REPLACE VIEW v_event_system_health AS
SELECT
    'dlq_pending' AS metric,
    (SELECT COUNT(*) FROM event_dead_letter_queue WHERE status = 'pending')::TEXT AS value
UNION ALL
SELECT
    'dlq_dead' AS metric,
    (SELECT COUNT(*) FROM event_dead_letter_queue WHERE status = 'dead')::TEXT AS value
UNION ALL
SELECT
    'events_processed_24h' AS metric,
    (SELECT COUNT(*) FROM event_processed_log WHERE processed_at > NOW() - INTERVAL '24 hours')::TEXT AS value
UNION ALL
SELECT
    'circuit_breakers_open' AS metric,
    (SELECT COUNT(*) FROM circuit_breaker_state WHERE state = 'OPEN')::TEXT AS value;

-- View: DLQ summary by event type
CREATE OR REPLACE VIEW v_dlq_summary AS
SELECT
    event_type,
    status,
    COUNT(*) AS count,
    AVG(error_count) AS avg_retries,
    MIN(first_failed_at) AS oldest_failure,
    MAX(last_failed_at) AS newest_failure
FROM event_dead_letter_queue
GROUP BY event_type, status
ORDER BY event_type, status;

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function: Add event to DLQ
CREATE OR REPLACE FUNCTION dlq_enqueue(
    p_event_id VARCHAR(64),
    p_event_type VARCHAR(50),
    p_channel VARCHAR(100),
    p_payload JSONB,
    p_error_message TEXT
)
RETURNS INTEGER AS $$
DECLARE
    result_id INTEGER;
BEGIN
    INSERT INTO event_dead_letter_queue (
        event_id, event_type, channel, payload, error_message, retry_after
    ) VALUES (
        p_event_id, p_event_type, p_channel, p_payload, p_error_message,
        NOW() + INTERVAL '5 minutes'  -- Exponential backoff handled by processor
    )
    ON CONFLICT (event_id) DO UPDATE SET
        error_count = event_dead_letter_queue.error_count + 1,
        last_failed_at = NOW(),
        error_message = EXCLUDED.error_message,
        retry_after = NOW() + (POWER(2, event_dead_letter_queue.error_count) || ' minutes')::INTERVAL
    RETURNING id INTO result_id;

    -- Mark as dead if too many retries
    UPDATE event_dead_letter_queue
    SET status = 'dead'
    WHERE id = result_id AND error_count >= 5;

    RETURN result_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Check if event was already processed (idempotency)
CREATE OR REPLACE FUNCTION is_event_processed(p_event_id VARCHAR(64))
RETURNS BOOLEAN AS $$
BEGIN
    RETURN EXISTS (
        SELECT 1 FROM event_processed_log WHERE event_id = p_event_id
    );
END;
$$ LANGUAGE plpgsql;

-- Function: Mark event as processed
CREATE OR REPLACE FUNCTION mark_event_processed(
    p_event_id VARCHAR(64),
    p_event_type VARCHAR(50),
    p_payload_hash VARCHAR(64),
    p_processor_id VARCHAR(100) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    INSERT INTO event_processed_log (event_id, event_type, payload_hash, processor_id)
    VALUES (p_event_id, p_event_type, p_payload_hash, p_processor_id)
    ON CONFLICT (event_id) DO NOTHING;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- GRANTS
-- =============================================================================

GRANT SELECT, INSERT, UPDATE, DELETE ON event_dead_letter_queue TO PUBLIC;
GRANT SELECT, INSERT ON event_processed_log TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON circuit_breaker_state TO PUBLIC;
GRANT USAGE ON SEQUENCE event_dead_letter_queue_id_seq TO PUBLIC;

-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON FUNCTION notify_new_ohlcv_bar() IS 'Trigger function that emits NOTIFY on new OHLCV bar insert';
COMMENT ON FUNCTION notify_features_ready() IS 'Trigger function that emits NOTIFY when features are ready';
COMMENT ON FUNCTION emit_heartbeat(TEXT) IS 'Emits heartbeat NOTIFY for system health monitoring';
COMMENT ON TABLE event_dead_letter_queue IS 'Stores failed events for retry (Dead Letter Queue pattern)';
COMMENT ON TABLE event_processed_log IS 'Tracks processed events for idempotency guarantee';
COMMENT ON TABLE circuit_breaker_state IS 'Persists circuit breaker state across restarts';

-- =============================================================================
-- ROLLBACK SCRIPT (save separately as rollback_033_event_triggers.sql)
-- =============================================================================
/*
-- Rollback commands (DO NOT EXECUTE AUTOMATICALLY):

DROP TRIGGER IF EXISTS trg_notify_new_ohlcv_bar ON usdcop_m5_ohlcv;
DROP TRIGGER IF EXISTS trg_notify_features_ready ON inference_features_5m;

DROP FUNCTION IF EXISTS notify_new_ohlcv_bar();
DROP FUNCTION IF EXISTS notify_features_ready();
DROP FUNCTION IF EXISTS emit_heartbeat(TEXT);
DROP FUNCTION IF EXISTS dlq_enqueue(VARCHAR, VARCHAR, VARCHAR, JSONB, TEXT);
DROP FUNCTION IF EXISTS is_event_processed(VARCHAR);
DROP FUNCTION IF EXISTS mark_event_processed(VARCHAR, VARCHAR, VARCHAR, VARCHAR);
DROP FUNCTION IF EXISTS cleanup_old_processed_events();

DROP VIEW IF EXISTS v_event_system_health;
DROP VIEW IF EXISTS v_dlq_summary;

DROP TABLE IF EXISTS event_dead_letter_queue;
DROP TABLE IF EXISTS event_processed_log;
DROP TABLE IF EXISTS circuit_breaker_state;
*/
