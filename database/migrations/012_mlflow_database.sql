-- =============================================================================
-- Migration 012: MLflow Database Setup
-- =============================================================================
-- Creates the MLflow database for experiment tracking and model registry.
-- This database is separate from the main trading database for isolation.
--
-- MLflow requires specific tables for:
--   - Experiments and runs tracking
--   - Model registry and versioning
--   - Metrics, parameters, and tags storage
--
-- Author: Trading Team
-- Date: 2026-01-16
-- =============================================================================

-- Create MLflow database if it doesn't exist
-- Note: This needs to be run by superuser or with CREATE DATABASE privileges
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = 'mlflow') THEN
        PERFORM dblink_exec('dbname=postgres', 'CREATE DATABASE mlflow');
    END IF;
EXCEPTION
    WHEN undefined_function THEN
        RAISE NOTICE 'dblink not available. Create mlflow database manually if needed.';
    WHEN OTHERS THEN
        RAISE NOTICE 'MLflow database may already exist or cannot be created: %', SQLERRM;
END $$;

-- =============================================================================
-- MLflow Tracking Tables (created in main database for simplicity)
-- MLflow will auto-create these, but we define them for documentation
-- =============================================================================

-- Create schema for MLflow if using shared database
CREATE SCHEMA IF NOT EXISTS mlflow;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA mlflow TO CURRENT_USER;

-- =============================================================================
-- Custom Model Registry Extension Tables
-- =============================================================================
-- These extend MLflow's model registry with trading-specific metadata

-- Model deployment tracking
CREATE TABLE IF NOT EXISTS mlflow.model_deployments (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version INTEGER NOT NULL,
    stage VARCHAR(50) NOT NULL,  -- None, Staging, Production, Archived
    deployed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    deployed_by VARCHAR(255),
    deployment_environment VARCHAR(50),  -- dev, staging, production
    config JSONB DEFAULT '{}',
    metrics_snapshot JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    notes TEXT,
    UNIQUE(model_name, model_version, deployment_environment)
);

-- Model performance tracking over time
CREATE TABLE IF NOT EXISTS mlflow.model_performance_history (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    model_version INTEGER NOT NULL,
    recorded_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    -- Trading performance metrics
    sharpe_ratio FLOAT,
    sortino_ratio FLOAT,
    max_drawdown FLOAT,
    win_rate FLOAT,
    profit_factor FLOAT,
    total_trades INTEGER,
    total_pnl FLOAT,
    -- Inference metrics
    avg_latency_ms FLOAT,
    p99_latency_ms FLOAT,
    predictions_count INTEGER,
    -- Data quality metrics
    feature_drift_score FLOAT,
    prediction_drift_score FLOAT,
    -- Additional metrics as JSON
    additional_metrics JSONB DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_deployments_model_name
    ON mlflow.model_deployments(model_name);
CREATE INDEX IF NOT EXISTS idx_deployments_stage
    ON mlflow.model_deployments(stage);
CREATE INDEX IF NOT EXISTS idx_deployments_active
    ON mlflow.model_deployments(is_active);
CREATE INDEX IF NOT EXISTS idx_performance_model
    ON mlflow.model_performance_history(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_performance_recorded
    ON mlflow.model_performance_history(recorded_at);

-- =============================================================================
-- Model Promotion Rules Table
-- =============================================================================
-- Defines automatic promotion criteria for models

CREATE TABLE IF NOT EXISTS mlflow.promotion_rules (
    id SERIAL PRIMARY KEY,
    rule_name VARCHAR(255) NOT NULL UNIQUE,
    from_stage VARCHAR(50) NOT NULL,
    to_stage VARCHAR(50) NOT NULL,
    -- Promotion criteria
    min_sharpe_ratio FLOAT,
    min_win_rate FLOAT,
    max_drawdown FLOAT,
    min_trades INTEGER,
    min_validation_days INTEGER,
    -- Additional criteria as JSON
    custom_criteria JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Insert default promotion rules
INSERT INTO mlflow.promotion_rules (rule_name, from_stage, to_stage, min_sharpe_ratio, min_win_rate, max_drawdown, min_trades, min_validation_days)
VALUES
    ('staging_promotion', 'None', 'Staging', 0.5, 0.45, -0.15, 50, 5),
    ('production_promotion', 'Staging', 'Production', 1.0, 0.50, -0.10, 100, 14)
ON CONFLICT (rule_name) DO NOTHING;

-- =============================================================================
-- Model Audit Log
-- =============================================================================
-- Tracks all model lifecycle events for compliance

CREATE TABLE IF NOT EXISTS mlflow.model_audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,  -- created, promoted, demoted, archived, deployed
    model_name VARCHAR(255) NOT NULL,
    model_version INTEGER,
    from_stage VARCHAR(50),
    to_stage VARCHAR(50),
    actor VARCHAR(255),
    reason TEXT,
    metadata JSONB DEFAULT '{}',
    event_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_audit_model
    ON mlflow.model_audit_log(model_name);
CREATE INDEX IF NOT EXISTS idx_audit_event_type
    ON mlflow.model_audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp
    ON mlflow.model_audit_log(event_timestamp);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Function to log model events
CREATE OR REPLACE FUNCTION mlflow.log_model_event(
    p_event_type VARCHAR(50),
    p_model_name VARCHAR(255),
    p_model_version INTEGER,
    p_from_stage VARCHAR(50),
    p_to_stage VARCHAR(50),
    p_actor VARCHAR(255),
    p_reason TEXT DEFAULT NULL,
    p_metadata JSONB DEFAULT '{}'
)
RETURNS INTEGER AS $$
DECLARE
    v_id INTEGER;
BEGIN
    INSERT INTO mlflow.model_audit_log (
        event_type, model_name, model_version,
        from_stage, to_stage, actor, reason, metadata
    )
    VALUES (
        p_event_type, p_model_name, p_model_version,
        p_from_stage, p_to_stage, p_actor, p_reason, p_metadata
    )
    RETURNING id INTO v_id;

    RETURN v_id;
END;
$$ LANGUAGE plpgsql;

-- Function to check promotion eligibility
CREATE OR REPLACE FUNCTION mlflow.check_promotion_eligibility(
    p_model_name VARCHAR(255),
    p_model_version INTEGER,
    p_target_stage VARCHAR(50)
)
RETURNS TABLE (
    eligible BOOLEAN,
    rule_name VARCHAR(255),
    missing_criteria TEXT[]
) AS $$
DECLARE
    v_rule RECORD;
    v_perf RECORD;
    v_missing TEXT[];
BEGIN
    -- Get the latest performance metrics
    SELECT * INTO v_perf
    FROM mlflow.model_performance_history
    WHERE model_name = p_model_name
      AND model_version = p_model_version
    ORDER BY recorded_at DESC
    LIMIT 1;

    -- Check each applicable rule
    FOR v_rule IN
        SELECT * FROM mlflow.promotion_rules
        WHERE to_stage = p_target_stage AND is_active = TRUE
    LOOP
        v_missing := ARRAY[]::TEXT[];

        IF v_perf IS NULL THEN
            RETURN QUERY SELECT FALSE, v_rule.rule_name, ARRAY['No performance data available']::TEXT[];
            CONTINUE;
        END IF;

        IF v_rule.min_sharpe_ratio IS NOT NULL AND COALESCE(v_perf.sharpe_ratio, 0) < v_rule.min_sharpe_ratio THEN
            v_missing := array_append(v_missing, format('Sharpe ratio %.2f < %.2f required', COALESCE(v_perf.sharpe_ratio, 0), v_rule.min_sharpe_ratio));
        END IF;

        IF v_rule.min_win_rate IS NOT NULL AND COALESCE(v_perf.win_rate, 0) < v_rule.min_win_rate THEN
            v_missing := array_append(v_missing, format('Win rate %.2f < %.2f required', COALESCE(v_perf.win_rate, 0), v_rule.min_win_rate));
        END IF;

        IF v_rule.max_drawdown IS NOT NULL AND COALESCE(v_perf.max_drawdown, -1) < v_rule.max_drawdown THEN
            v_missing := array_append(v_missing, format('Max drawdown %.2f < %.2f limit', COALESCE(v_perf.max_drawdown, -1), v_rule.max_drawdown));
        END IF;

        IF v_rule.min_trades IS NOT NULL AND COALESCE(v_perf.total_trades, 0) < v_rule.min_trades THEN
            v_missing := array_append(v_missing, format('Total trades %s < %s required', COALESCE(v_perf.total_trades, 0), v_rule.min_trades));
        END IF;

        RETURN QUERY SELECT (array_length(v_missing, 1) IS NULL OR array_length(v_missing, 1) = 0), v_rule.rule_name, v_missing;
    END LOOP;

    RETURN;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Notification: Migration complete
-- =============================================================================
DO $$
BEGIN
    RAISE NOTICE 'MLflow database migration 012 completed successfully.';
    RAISE NOTICE 'Tables created: model_deployments, model_performance_history, promotion_rules, model_audit_log';
    RAISE NOTICE 'Functions created: log_model_event, check_promotion_eligibility';
END $$;
