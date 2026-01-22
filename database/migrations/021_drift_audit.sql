-- =============================================================================
-- Migration 021: Drift Audit Tables
-- =============================================================================
-- Sprint 3: COMP-88 - Dashboard Improvements
-- Creates tables for tracking drift detection history and alerts.
--
-- Tables:
--   - drift_checks: Individual drift check results
--   - drift_alerts: Triggered drift alerts
--   - drift_reference_stats: Stored reference statistics
--
-- Author: Trading Team
-- Date: 2026-01-17
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: drift_checks
-- Stores individual drift check results for auditing and analysis
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_checks (
    id SERIAL PRIMARY KEY,
    check_id UUID NOT NULL DEFAULT gen_random_uuid(),
    check_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Check type
    check_type VARCHAR(20) NOT NULL CHECK (check_type IN ('univariate', 'multivariate')),

    -- Results summary
    features_checked INTEGER NOT NULL DEFAULT 0,
    features_drifted INTEGER NOT NULL DEFAULT 0,
    overall_drift_score DECIMAL(6,4),
    max_severity VARCHAR(10) CHECK (max_severity IN ('none', 'low', 'medium', 'high')),
    alert_triggered BOOLEAN NOT NULL DEFAULT FALSE,

    -- Detailed results (JSON)
    univariate_results JSONB,
    multivariate_results JSONB,

    -- Context
    model_id VARCHAR(100),
    triggered_by VARCHAR(50) DEFAULT 'scheduled',  -- scheduled, manual, inference

    -- Indexes
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_drift_checks_timestamp ON drift_checks(check_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_drift_checks_check_id ON drift_checks(check_id);
CREATE INDEX IF NOT EXISTS idx_drift_checks_severity ON drift_checks(max_severity) WHERE max_severity IN ('medium', 'high');
CREATE INDEX IF NOT EXISTS idx_drift_checks_model ON drift_checks(model_id);

-- -----------------------------------------------------------------------------
-- Table: drift_alerts
-- Stores triggered drift alerts for investigation
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_alerts (
    id SERIAL PRIMARY KEY,
    alert_id UUID NOT NULL DEFAULT gen_random_uuid(),
    alert_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Alert details
    severity VARCHAR(10) NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    alert_type VARCHAR(30) NOT NULL,  -- univariate_drift, multivariate_drift, combined
    message TEXT NOT NULL,

    -- Related check
    drift_check_id INTEGER REFERENCES drift_checks(id),

    -- Drifted features (JSON array)
    drifted_features JSONB,

    -- Alert state
    state VARCHAR(20) NOT NULL DEFAULT 'active' CHECK (state IN ('active', 'acknowledged', 'resolved', 'suppressed')),
    acknowledged_by VARCHAR(100),
    acknowledged_at TIMESTAMPTZ,
    resolved_by VARCHAR(100),
    resolved_at TIMESTAMPTZ,
    resolution_notes TEXT,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_drift_alerts_timestamp ON drift_alerts(alert_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_drift_alerts_state ON drift_alerts(state) WHERE state = 'active';
CREATE INDEX IF NOT EXISTS idx_drift_alerts_severity ON drift_alerts(severity);

-- -----------------------------------------------------------------------------
-- Table: drift_reference_stats
-- Stores versioned reference statistics for audit trail
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_reference_stats (
    id SERIAL PRIMARY KEY,
    version_id UUID NOT NULL DEFAULT gen_random_uuid(),

    -- Reference data info
    feature_name VARCHAR(100) NOT NULL,
    stat_type VARCHAR(20) NOT NULL,  -- mean, std, min, max, p25, p50, p75
    stat_value DECIMAL(20,10) NOT NULL,

    -- Sample info
    sample_count INTEGER NOT NULL,
    computed_from VARCHAR(200),  -- e.g., "training_data_2026-01-15"

    -- Validity period
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    is_current BOOLEAN NOT NULL DEFAULT TRUE,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100) DEFAULT 'system'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_drift_ref_stats_feature ON drift_reference_stats(feature_name);
CREATE INDEX IF NOT EXISTS idx_drift_ref_stats_current ON drift_reference_stats(is_current) WHERE is_current = TRUE;
CREATE UNIQUE INDEX IF NOT EXISTS idx_drift_ref_stats_current_feature_stat
    ON drift_reference_stats(feature_name, stat_type)
    WHERE is_current = TRUE;

-- -----------------------------------------------------------------------------
-- Table: drift_observations_summary
-- Aggregated observation statistics (not individual observations)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS drift_observations_summary (
    id SERIAL PRIMARY KEY,
    summary_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Time window
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    observation_count INTEGER NOT NULL,

    -- Per-feature statistics (JSON)
    feature_stats JSONB NOT NULL,

    -- Comparison to reference
    drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    drift_score DECIMAL(6,4),

    -- Context
    model_id VARCHAR(100),
    environment VARCHAR(20) DEFAULT 'production',

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for time-based queries
CREATE INDEX IF NOT EXISTS idx_drift_obs_summary_time
    ON drift_observations_summary(summary_timestamp DESC);

-- -----------------------------------------------------------------------------
-- Function: Insert drift check and create alert if needed
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_insert_drift_check_with_alert(
    p_check_type VARCHAR(20),
    p_features_checked INTEGER,
    p_features_drifted INTEGER,
    p_drift_score DECIMAL(6,4),
    p_max_severity VARCHAR(10),
    p_univariate_results JSONB,
    p_multivariate_results JSONB,
    p_model_id VARCHAR(100) DEFAULT NULL,
    p_triggered_by VARCHAR(50) DEFAULT 'scheduled'
) RETURNS TABLE(check_id INTEGER, alert_id INTEGER) AS $$
DECLARE
    v_check_id INTEGER;
    v_alert_id INTEGER;
    v_alert_needed BOOLEAN;
BEGIN
    -- Insert the drift check
    INSERT INTO drift_checks (
        check_type, features_checked, features_drifted,
        overall_drift_score, max_severity, alert_triggered,
        univariate_results, multivariate_results,
        model_id, triggered_by
    ) VALUES (
        p_check_type, p_features_checked, p_features_drifted,
        p_drift_score, p_max_severity,
        p_max_severity IN ('medium', 'high'),
        p_univariate_results, p_multivariate_results,
        p_model_id, p_triggered_by
    ) RETURNING id INTO v_check_id;

    -- Create alert if severity is medium or high
    v_alert_needed := p_max_severity IN ('medium', 'high');

    IF v_alert_needed THEN
        INSERT INTO drift_alerts (
            severity, alert_type, message,
            drift_check_id, drifted_features
        ) VALUES (
            p_max_severity,
            p_check_type || '_drift',
            format('Drift detected: %s/%s features drifted (score: %s)',
                   p_features_drifted, p_features_checked, p_drift_score),
            v_check_id,
            COALESCE(
                p_univariate_results->'drifted_features',
                '[]'::JSONB
            )
        ) RETURNING id INTO v_alert_id;
    END IF;

    RETURN QUERY SELECT v_check_id, v_alert_id;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Function: Get drift history for dashboard
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_get_drift_history(
    p_hours INTEGER DEFAULT 24,
    p_model_id VARCHAR(100) DEFAULT NULL
) RETURNS TABLE(
    check_timestamp TIMESTAMPTZ,
    check_type VARCHAR(20),
    features_drifted INTEGER,
    overall_drift_score DECIMAL(6,4),
    max_severity VARCHAR(10),
    alert_triggered BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        dc.check_timestamp,
        dc.check_type,
        dc.features_drifted,
        dc.overall_drift_score,
        dc.max_severity,
        dc.alert_triggered
    FROM drift_checks dc
    WHERE dc.check_timestamp >= NOW() - (p_hours || ' hours')::INTERVAL
      AND (p_model_id IS NULL OR dc.model_id = p_model_id)
    ORDER BY dc.check_timestamp DESC
    LIMIT 1000;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- View: Active drift alerts
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_active_drift_alerts AS
SELECT
    da.alert_id,
    da.alert_timestamp,
    da.severity,
    da.alert_type,
    da.message,
    da.drifted_features,
    dc.features_checked,
    dc.features_drifted,
    dc.overall_drift_score,
    dc.model_id
FROM drift_alerts da
LEFT JOIN drift_checks dc ON da.drift_check_id = dc.id
WHERE da.state = 'active'
ORDER BY da.alert_timestamp DESC;

-- -----------------------------------------------------------------------------
-- View: Drift summary (last 24 hours)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE VIEW v_drift_summary_24h AS
SELECT
    COUNT(*) AS total_checks,
    SUM(CASE WHEN alert_triggered THEN 1 ELSE 0 END) AS alerts_triggered,
    MAX(overall_drift_score) AS max_drift_score,
    AVG(overall_drift_score) AS avg_drift_score,
    MODE() WITHIN GROUP (ORDER BY max_severity) AS most_common_severity,
    MAX(CASE WHEN max_severity = 'high' THEN check_timestamp END) AS last_high_severity
FROM drift_checks
WHERE check_timestamp >= NOW() - INTERVAL '24 hours';

-- -----------------------------------------------------------------------------
-- Trigger: Update timestamp on alert changes
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION fn_update_drift_alert_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_drift_alert_updated ON drift_alerts;
CREATE TRIGGER trg_drift_alert_updated
    BEFORE UPDATE ON drift_alerts
    FOR EACH ROW
    EXECUTE FUNCTION fn_update_drift_alert_timestamp();

-- -----------------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------------
COMMENT ON TABLE drift_checks IS 'Stores drift detection check results for auditing';
COMMENT ON TABLE drift_alerts IS 'Stores triggered drift alerts for investigation and resolution';
COMMENT ON TABLE drift_reference_stats IS 'Versioned reference statistics for drift comparison';
COMMENT ON TABLE drift_observations_summary IS 'Aggregated observation statistics by time window';

COMMENT ON FUNCTION fn_insert_drift_check_with_alert IS 'Insert drift check and auto-create alert if severity is medium/high';
COMMENT ON FUNCTION fn_get_drift_history IS 'Get drift check history for dashboard visualization';

-- -----------------------------------------------------------------------------
-- Grant permissions (adjust as needed for your setup)
-- -----------------------------------------------------------------------------
-- GRANT SELECT, INSERT ON drift_checks TO inference_api;
-- GRANT SELECT, INSERT, UPDATE ON drift_alerts TO inference_api;
-- GRANT SELECT ON v_active_drift_alerts TO inference_api;
-- GRANT SELECT ON v_drift_summary_24h TO inference_api;
