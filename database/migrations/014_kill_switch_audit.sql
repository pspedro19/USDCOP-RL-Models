-- =============================================================================
-- Migration: 014_kill_switch_audit.sql
-- =============================================================================
-- Creates the kill switch audit table for tracking all kill switch state changes.
--
-- P1: Kill Switch Audit Logging
--
-- Features:
-- - Complete history of kill switch activations/deactivations
-- - User and source tracking
-- - Reason documentation
-- - Duration tracking
-- - Alert integration support
--
-- Author: Trading Team
-- Date: 2026-01-17
-- =============================================================================

-- Ensure audit schema exists
CREATE SCHEMA IF NOT EXISTS audit;

-- =============================================================================
-- Kill Switch Audit Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS audit.kill_switch_audit (
    -- Primary key
    id SERIAL PRIMARY KEY,

    -- State change
    previous_state BOOLEAN NOT NULL,
    new_state BOOLEAN NOT NULL,
    action VARCHAR(20) NOT NULL CHECK (action IN ('ACTIVATED', 'DEACTIVATED', 'OVERRIDE')),

    -- Reason and context
    reason TEXT NOT NULL,
    severity VARCHAR(20) DEFAULT 'NORMAL' CHECK (severity IN ('LOW', 'NORMAL', 'HIGH', 'CRITICAL')),

    -- Source information
    triggered_by VARCHAR(100) NOT NULL,  -- Username, service name, or 'SYSTEM'
    trigger_source VARCHAR(50) NOT NULL,  -- 'API', 'DAG', 'MANUAL', 'AUTOMATED', 'ALERT'
    correlation_id UUID,  -- Links to related request/event

    -- System state at time of change
    active_trades_count INTEGER DEFAULT 0,
    pending_orders_count INTEGER DEFAULT 0,
    current_drawdown DECIMAL(10, 6),
    current_volatility DECIMAL(10, 6),
    market_status VARCHAR(20),  -- 'OPEN', 'CLOSED', 'PRE_MARKET', etc.

    -- Related alerts/incidents
    alert_id VARCHAR(100),
    incident_id VARCHAR(100),

    -- Environment
    environment VARCHAR(20) DEFAULT 'production',
    hostname VARCHAR(100),

    -- Timing
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Additional context
    metadata JSONB
);

-- =============================================================================
-- Indexes
-- =============================================================================
CREATE INDEX IF NOT EXISTS idx_kill_switch_audit_created ON audit.kill_switch_audit(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_kill_switch_audit_action ON audit.kill_switch_audit(action);
CREATE INDEX IF NOT EXISTS idx_kill_switch_audit_triggered_by ON audit.kill_switch_audit(triggered_by);
CREATE INDEX IF NOT EXISTS idx_kill_switch_audit_severity ON audit.kill_switch_audit(severity);
CREATE INDEX IF NOT EXISTS idx_kill_switch_audit_environment ON audit.kill_switch_audit(environment);

-- Composite index for common queries
CREATE INDEX IF NOT EXISTS idx_kill_switch_audit_action_time
    ON audit.kill_switch_audit(action, created_at DESC);

-- =============================================================================
-- Kill Switch Duration View
-- =============================================================================
CREATE OR REPLACE VIEW audit.kill_switch_duration AS
WITH activations AS (
    SELECT
        id,
        created_at AS activated_at,
        reason,
        triggered_by,
        severity,
        LEAD(created_at) OVER (ORDER BY created_at) AS deactivated_at
    FROM audit.kill_switch_audit
    WHERE action = 'ACTIVATED'
),
deactivations AS (
    SELECT
        created_at AS deactivated_at,
        triggered_by AS deactivated_by,
        reason AS deactivation_reason
    FROM audit.kill_switch_audit
    WHERE action = 'DEACTIVATED'
)
SELECT
    a.id,
    a.activated_at,
    COALESCE(a.deactivated_at, NOW()) AS deactivated_at,
    EXTRACT(EPOCH FROM (COALESCE(a.deactivated_at, NOW()) - a.activated_at)) / 60 AS duration_minutes,
    a.reason AS activation_reason,
    a.triggered_by AS activated_by,
    a.severity,
    CASE
        WHEN a.deactivated_at IS NULL THEN TRUE
        ELSE FALSE
    END AS currently_active
FROM activations a
ORDER BY a.activated_at DESC;

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to log kill switch change
CREATE OR REPLACE FUNCTION audit.log_kill_switch_change(
    p_previous_state BOOLEAN,
    p_new_state BOOLEAN,
    p_reason TEXT,
    p_triggered_by VARCHAR(100),
    p_trigger_source VARCHAR(50) DEFAULT 'API',
    p_severity VARCHAR(20) DEFAULT 'NORMAL',
    p_active_trades INTEGER DEFAULT 0,
    p_metadata JSONB DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    v_action VARCHAR(20);
    v_id INTEGER;
BEGIN
    -- Determine action type
    IF p_previous_state = FALSE AND p_new_state = TRUE THEN
        v_action := 'ACTIVATED';
    ELSIF p_previous_state = TRUE AND p_new_state = FALSE THEN
        v_action := 'DEACTIVATED';
    ELSE
        v_action := 'OVERRIDE';
    END IF;

    INSERT INTO audit.kill_switch_audit (
        previous_state,
        new_state,
        action,
        reason,
        severity,
        triggered_by,
        trigger_source,
        active_trades_count,
        environment,
        hostname,
        metadata
    ) VALUES (
        p_previous_state,
        p_new_state,
        v_action,
        p_reason,
        p_severity,
        p_triggered_by,
        p_trigger_source,
        p_active_trades,
        current_setting('app.environment', true),
        inet_server_addr()::text,
        p_metadata
    )
    RETURNING id INTO v_id;

    -- Log to application log as well
    RAISE NOTICE 'Kill switch %: % by % - %', v_action, p_reason, p_triggered_by, p_severity;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to get current kill switch status
CREATE OR REPLACE FUNCTION audit.get_kill_switch_status()
RETURNS TABLE (
    is_active BOOLEAN,
    last_change_at TIMESTAMPTZ,
    last_change_by VARCHAR(100),
    last_reason TEXT,
    total_activations INTEGER,
    total_duration_minutes DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    WITH latest AS (
        SELECT
            new_state,
            created_at,
            triggered_by,
            reason
        FROM audit.kill_switch_audit
        ORDER BY created_at DESC
        LIMIT 1
    ),
    stats AS (
        SELECT
            COUNT(*) FILTER (WHERE action = 'ACTIVATED') AS activations,
            COALESCE(SUM(
                EXTRACT(EPOCH FROM (
                    COALESCE(
                        LEAD(created_at) OVER (ORDER BY created_at),
                        CASE WHEN new_state = TRUE THEN NOW() ELSE created_at END
                    ) - created_at
                )) / 60
            ) FILTER (WHERE action = 'ACTIVATED'), 0) AS total_minutes
        FROM audit.kill_switch_audit
    )
    SELECT
        COALESCE(l.new_state, FALSE),
        l.created_at,
        l.triggered_by,
        l.reason,
        s.activations::INTEGER,
        s.total_minutes
    FROM latest l
    CROSS JOIN stats s;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Alert Integration Support
-- =============================================================================

-- Table to track kill switch alerts sent
CREATE TABLE IF NOT EXISTS audit.kill_switch_alerts (
    id SERIAL PRIMARY KEY,
    kill_switch_audit_id INTEGER REFERENCES audit.kill_switch_audit(id),
    alert_channel VARCHAR(50) NOT NULL,  -- 'SLACK', 'PAGERDUTY', 'EMAIL', 'SMS'
    alert_sent_at TIMESTAMPTZ DEFAULT NOW(),
    alert_status VARCHAR(20) DEFAULT 'SENT',  -- 'SENT', 'FAILED', 'ACKNOWLEDGED'
    alert_response JSONB,
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_kill_switch_alerts_audit_id
    ON audit.kill_switch_alerts(kill_switch_audit_id);

-- =============================================================================
-- Comments
-- =============================================================================
COMMENT ON TABLE audit.kill_switch_audit IS 'Audit log for all kill switch state changes';
COMMENT ON TABLE audit.kill_switch_alerts IS 'Track alerts sent for kill switch events';
COMMENT ON VIEW audit.kill_switch_duration IS 'View showing duration of each kill switch activation';
COMMENT ON FUNCTION audit.log_kill_switch_change IS 'Log a kill switch state change with full context';
COMMENT ON FUNCTION audit.get_kill_switch_status IS 'Get current kill switch status and statistics';

-- =============================================================================
-- Migration metadata
-- =============================================================================
INSERT INTO public.schema_migrations (version, description, applied_at)
VALUES ('014', 'Create kill switch audit tables', NOW())
ON CONFLICT (version) DO NOTHING;
