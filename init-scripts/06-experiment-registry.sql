-- =============================================================================
-- USDCOP Trading System - Experiment Registry (A/B Testing)
-- =============================================================================
-- Source: database/migrations/022_experiment_registry.sql
-- Copied to init-scripts for Docker auto-initialization
--
-- This script creates the A/B experimentation framework tables:
--   - experiment_runs: Individual experiment run records
--   - experiment_comparisons: A/B comparison results
--   - experiment_deployments: Deployment decisions and rollouts
--
-- CRITICAL: This file includes additional columns not in the original migration
-- to align with the XCom Contracts (CTR-XCOM-001) and Feature Contracts (CTR-FEATURE-001):
--   - dataset_hash (from L2Output)
--   - model_id FK (links to model_registry)
--   - feature_order_hash (from Feature Contract)
--   - norm_stats_hash (from L3Output)
--
-- Author: Trading Team
-- Date: 2026-01-18
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table: experiment_runs
-- Stores individual experiment run results
-- ENHANCED: Added columns for SSOT alignment (dataset_hash, model_id, etc.)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_runs (
    id SERIAL PRIMARY KEY,
    run_uuid UUID NOT NULL DEFAULT gen_random_uuid(),

    -- Experiment identification
    experiment_name VARCHAR(64) NOT NULL,
    experiment_version VARCHAR(20) NOT NULL,
    run_id VARCHAR(64) NOT NULL UNIQUE,

    -- Run status
    status VARCHAR(20) NOT NULL CHECK (status IN ('running', 'success', 'failed', 'cancelled', 'dry_run')),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    duration_seconds DECIMAL(12, 2),

    -- Metrics (JSON for flexibility)
    training_metrics JSONB DEFAULT '{}',
    eval_metrics JSONB DEFAULT '{}',
    backtest_metrics JSONB DEFAULT '{}',

    -- Artifacts
    model_path VARCHAR(500),
    config_path VARCHAR(500),

    -- MLflow integration
    mlflow_run_id VARCHAR(64),
    mlflow_experiment_id VARCHAR(64),

    -- ==========================================================================
    -- SSOT ALIGNMENT: Additional columns from XCom Contracts (CTR-XCOM-001)
    -- ==========================================================================

    -- From L2Output (Dataset traceability)
    dataset_path VARCHAR(500),
    dataset_hash VARCHAR(64),           -- SHA256 of training dataset
    date_range_start DATE,
    date_range_end DATE,

    -- From Feature Contract (CTR-FEATURE-001)
    feature_order_hash VARCHAR(16),     -- Hash of FEATURE_ORDER tuple
    feature_columns JSONB,              -- Array of feature names used

    -- From L3Output (Model traceability)
    norm_stats_hash VARCHAR(64),        -- SHA256 of norm_stats.json
    config_hash VARCHAR(64),            -- SHA256 of training config

    -- Link to model_registry (P0 requirement)
    model_id VARCHAR(50),               -- FK to model_registry.model_id
    model_hash VARCHAR(64),             -- SHA256 of trained model file

    -- DVC integration
    dvc_tag VARCHAR(100),               -- DVC version tag

    -- ==========================================================================
    -- Error tracking
    error TEXT,
    traceback TEXT,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_exp_runs_experiment ON experiment_runs(experiment_name);
CREATE INDEX IF NOT EXISTS idx_exp_runs_status ON experiment_runs(status);
CREATE INDEX IF NOT EXISTS idx_exp_runs_started ON experiment_runs(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_exp_runs_mlflow ON experiment_runs(mlflow_run_id);
CREATE INDEX IF NOT EXISTS idx_exp_runs_model_id ON experiment_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_exp_runs_dataset_hash ON experiment_runs(dataset_hash);
CREATE INDEX IF NOT EXISTS idx_exp_runs_feature_order ON experiment_runs(feature_order_hash);

-- Unique constraint for run tracking
CREATE UNIQUE INDEX IF NOT EXISTS idx_exp_runs_unique_run ON experiment_runs(experiment_name, run_id);

-- -----------------------------------------------------------------------------
-- Table: experiment_comparisons
-- Stores A/B comparison results between experiments
-- ENHANCED: Added sample size and test duration tracking
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_comparisons (
    id SERIAL PRIMARY KEY,
    comparison_uuid UUID NOT NULL DEFAULT gen_random_uuid(),

    -- Experiments being compared
    baseline_experiment VARCHAR(64) NOT NULL,
    baseline_version VARCHAR(20) NOT NULL,
    baseline_run_id VARCHAR(64),

    treatment_experiment VARCHAR(64) NOT NULL,
    treatment_version VARCHAR(20) NOT NULL,
    treatment_run_id VARCHAR(64),

    -- Comparison configuration
    primary_metric VARCHAR(50) NOT NULL DEFAULT 'sharpe_ratio',
    significance_level DECIMAL(4, 3) DEFAULT 0.05,

    -- Results (from ABStatistics.ABTestResult)
    metric_comparisons JSONB NOT NULL,
    statistical_tests JSONB,

    -- ==========================================================================
    -- SSOT ALIGNMENT: Additional tracking for reproducibility
    -- ==========================================================================
    baseline_sample_size INTEGER,       -- Number of trades/observations
    treatment_sample_size INTEGER,
    test_duration_hours DECIMAL(10, 2), -- How long the comparison ran
    assumptions_tested JSONB,           -- Statistical assumptions validation

    -- Decision
    recommendation VARCHAR(50) NOT NULL CHECK (recommendation IN ('deploy_treatment', 'keep_baseline', 'inconclusive', 'needs_more_data')),
    confidence_level DECIMAL(4, 3),

    -- Context
    compared_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    compared_by VARCHAR(100) DEFAULT 'system',
    notes TEXT,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_exp_comp_baseline ON experiment_comparisons(baseline_experiment);
CREATE INDEX IF NOT EXISTS idx_exp_comp_treatment ON experiment_comparisons(treatment_experiment);
CREATE INDEX IF NOT EXISTS idx_exp_comp_date ON experiment_comparisons(compared_at DESC);
CREATE INDEX IF NOT EXISTS idx_exp_comp_recommendation ON experiment_comparisons(recommendation);

-- -----------------------------------------------------------------------------
-- Table: experiment_deployments
-- Tracks deployment decisions and rollouts
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiment_deployments (
    id SERIAL PRIMARY KEY,
    deployment_uuid UUID NOT NULL DEFAULT gen_random_uuid(),

    -- What was deployed
    experiment_name VARCHAR(64) NOT NULL,
    experiment_version VARCHAR(20) NOT NULL,
    run_id VARCHAR(64) NOT NULL,
    model_path VARCHAR(500),

    -- Deployment configuration
    deployment_type VARCHAR(30) NOT NULL CHECK (deployment_type IN ('full', 'canary', 'shadow', 'rollback')),
    traffic_percentage DECIMAL(5, 2) DEFAULT 100.0,
    target_environment VARCHAR(30) NOT NULL CHECK (target_environment IN ('staging', 'production', 'paper_trading')),

    -- From comparison
    comparison_id INTEGER REFERENCES experiment_comparisons(id),

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'deploying', 'deployed', 'rolled_back', 'failed')),
    deployed_at TIMESTAMPTZ,
    rolled_back_at TIMESTAMPTZ,

    -- Who/what triggered
    triggered_by VARCHAR(100) DEFAULT 'manual',
    approval_required BOOLEAN DEFAULT TRUE,
    approved_by VARCHAR(100),
    approved_at TIMESTAMPTZ,

    -- Rollback info
    previous_experiment VARCHAR(64),
    previous_version VARCHAR(20),
    rollback_reason TEXT,

    -- Metadata
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_exp_deploy_experiment ON experiment_deployments(experiment_name);
CREATE INDEX IF NOT EXISTS idx_exp_deploy_status ON experiment_deployments(status);
CREATE INDEX IF NOT EXISTS idx_exp_deploy_env ON experiment_deployments(target_environment);
CREATE INDEX IF NOT EXISTS idx_exp_deploy_date ON experiment_deployments(deployed_at DESC);

-- Get current deployment per environment (only ONE active per environment)
CREATE UNIQUE INDEX IF NOT EXISTS idx_exp_deploy_current
    ON experiment_deployments(target_environment)
    WHERE status = 'deployed';

-- -----------------------------------------------------------------------------
-- Views for common queries
-- -----------------------------------------------------------------------------

-- View: Latest successful run per experiment
CREATE OR REPLACE VIEW v_latest_experiment_runs AS
SELECT DISTINCT ON (experiment_name)
    id,
    experiment_name,
    experiment_version,
    run_id,
    status,
    started_at,
    completed_at,
    duration_seconds,
    backtest_metrics->>'sharpe_ratio' AS sharpe_ratio,
    backtest_metrics->>'total_return' AS total_return,
    backtest_metrics->>'max_drawdown' AS max_drawdown,
    backtest_metrics->>'win_rate' AS win_rate,
    model_path,
    model_id,
    dataset_hash,
    feature_order_hash,
    norm_stats_hash,
    mlflow_run_id
FROM experiment_runs
WHERE status = 'success'
ORDER BY experiment_name, started_at DESC;

-- View: Experiment summary statistics
CREATE OR REPLACE VIEW v_experiment_summary AS
SELECT
    experiment_name,
    COUNT(*) AS total_runs,
    COUNT(*) FILTER (WHERE status = 'success') AS successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed') AS failed_runs,
    AVG(duration_seconds) AS avg_duration_seconds,
    MIN(started_at) AS first_run,
    MAX(started_at) AS last_run,
    AVG((backtest_metrics->>'sharpe_ratio')::FLOAT)
        FILTER (WHERE status = 'success') AS avg_sharpe_ratio,
    MAX((backtest_metrics->>'sharpe_ratio')::FLOAT)
        FILTER (WHERE status = 'success') AS best_sharpe_ratio,
    COUNT(DISTINCT experiment_version) AS version_count,
    COUNT(DISTINCT dataset_hash) AS unique_datasets,
    COUNT(DISTINCT feature_order_hash) AS unique_feature_contracts
FROM experiment_runs
GROUP BY experiment_name
ORDER BY last_run DESC;

-- View: Current deployments with full context
CREATE OR REPLACE VIEW v_current_deployments AS
SELECT
    d.deployment_uuid,
    d.experiment_name,
    d.experiment_version,
    d.run_id,
    d.target_environment,
    d.traffic_percentage,
    d.deployment_type,
    d.deployed_at,
    d.triggered_by,
    r.backtest_metrics->>'sharpe_ratio' AS sharpe_ratio,
    r.model_path,
    r.model_id,
    r.dataset_hash,
    r.feature_order_hash,
    r.norm_stats_hash
FROM experiment_deployments d
LEFT JOIN experiment_runs r ON d.run_id = r.run_id
WHERE d.status = 'deployed'
ORDER BY d.target_environment, d.deployed_at DESC;

-- View: Recent comparisons with winners
CREATE OR REPLACE VIEW v_recent_comparisons AS
SELECT
    comparison_uuid,
    baseline_experiment,
    treatment_experiment,
    primary_metric,
    recommendation,
    confidence_level,
    baseline_sample_size,
    treatment_sample_size,
    test_duration_hours,
    (metric_comparisons->primary_metric->>'baseline_value')::FLOAT AS baseline_value,
    (metric_comparisons->primary_metric->>'treatment_value')::FLOAT AS treatment_value,
    (metric_comparisons->primary_metric->>'relative_difference')::FLOAT AS relative_difference,
    (metric_comparisons->primary_metric->>'is_significant')::BOOLEAN AS is_significant,
    compared_at,
    compared_by
FROM experiment_comparisons
ORDER BY compared_at DESC
LIMIT 50;

-- View: Experiment runs with model registry linkage
CREATE OR REPLACE VIEW v_experiment_model_lineage AS
SELECT
    er.run_id,
    er.experiment_name,
    er.experiment_version,
    er.status,
    er.started_at,
    er.model_id,
    er.model_hash,
    er.dataset_hash,
    er.feature_order_hash,
    er.norm_stats_hash,
    mr.model_version AS registry_model_version,
    mr.model_path AS registry_model_path,
    mr.observation_dim,
    mr.action_space,
    mr.test_sharpe AS registry_test_sharpe,
    mr.status AS registry_status
FROM experiment_runs er
LEFT JOIN model_registry mr ON er.model_id = mr.model_id
WHERE er.status = 'success'
ORDER BY er.started_at DESC;

-- -----------------------------------------------------------------------------
-- Functions
-- -----------------------------------------------------------------------------

-- Function: Get best run for an experiment
CREATE OR REPLACE FUNCTION fn_get_best_experiment_run(
    p_experiment_name VARCHAR(64),
    p_metric VARCHAR(50) DEFAULT 'sharpe_ratio',
    p_higher_is_better BOOLEAN DEFAULT TRUE
) RETURNS TABLE(
    run_id VARCHAR(64),
    experiment_version VARCHAR(20),
    metric_value FLOAT,
    started_at TIMESTAMPTZ,
    model_path VARCHAR(500),
    model_id VARCHAR(50),
    dataset_hash VARCHAR(64),
    feature_order_hash VARCHAR(16)
) AS $$
BEGIN
    IF p_higher_is_better THEN
        RETURN QUERY
        SELECT
            r.run_id,
            r.experiment_version,
            (r.backtest_metrics->>p_metric)::FLOAT AS metric_value,
            r.started_at,
            r.model_path,
            r.model_id,
            r.dataset_hash,
            r.feature_order_hash
        FROM experiment_runs r
        WHERE r.experiment_name = p_experiment_name
          AND r.status = 'success'
          AND r.backtest_metrics ? p_metric
        ORDER BY (r.backtest_metrics->>p_metric)::FLOAT DESC
        LIMIT 1;
    ELSE
        RETURN QUERY
        SELECT
            r.run_id,
            r.experiment_version,
            (r.backtest_metrics->>p_metric)::FLOAT AS metric_value,
            r.started_at,
            r.model_path,
            r.model_id,
            r.dataset_hash,
            r.feature_order_hash
        FROM experiment_runs r
        WHERE r.experiment_name = p_experiment_name
          AND r.status = 'success'
          AND r.backtest_metrics ? p_metric
        ORDER BY (r.backtest_metrics->>p_metric)::FLOAT ASC
        LIMIT 1;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Function: Record deployment with full lineage
CREATE OR REPLACE FUNCTION fn_record_deployment(
    p_experiment_name VARCHAR(64),
    p_experiment_version VARCHAR(20),
    p_run_id VARCHAR(64),
    p_model_path VARCHAR(500),
    p_target_environment VARCHAR(30),
    p_deployment_type VARCHAR(30) DEFAULT 'full',
    p_traffic_percentage DECIMAL(5,2) DEFAULT 100.0,
    p_comparison_id INTEGER DEFAULT NULL,
    p_triggered_by VARCHAR(100) DEFAULT 'manual'
) RETURNS INTEGER AS $$
DECLARE
    v_deployment_id INTEGER;
    v_previous_experiment VARCHAR(64);
    v_previous_version VARCHAR(20);
BEGIN
    -- Get current deployment for rollback info
    SELECT experiment_name, experiment_version
    INTO v_previous_experiment, v_previous_version
    FROM experiment_deployments
    WHERE target_environment = p_target_environment
      AND status = 'deployed'
    LIMIT 1;

    -- Mark previous deployment as rolled back
    UPDATE experiment_deployments
    SET status = 'rolled_back',
        rolled_back_at = NOW(),
        rollback_reason = 'New deployment: ' || p_experiment_name || ' v' || p_experiment_version
    WHERE target_environment = p_target_environment
      AND status = 'deployed';

    -- Insert new deployment
    INSERT INTO experiment_deployments (
        experiment_name,
        experiment_version,
        run_id,
        model_path,
        deployment_type,
        traffic_percentage,
        target_environment,
        comparison_id,
        status,
        deployed_at,
        triggered_by,
        previous_experiment,
        previous_version
    ) VALUES (
        p_experiment_name,
        p_experiment_version,
        p_run_id,
        p_model_path,
        p_deployment_type,
        p_traffic_percentage,
        p_target_environment,
        p_comparison_id,
        'deployed',
        NOW(),
        p_triggered_by,
        v_previous_experiment,
        v_previous_version
    ) RETURNING id INTO v_deployment_id;

    RETURN v_deployment_id;
END;
$$ LANGUAGE plpgsql;

-- Function: Validate experiment SSOT alignment
CREATE OR REPLACE FUNCTION fn_validate_experiment_ssot(
    p_run_id VARCHAR(64),
    p_expected_feature_order_hash VARCHAR(16) DEFAULT NULL
) RETURNS TABLE(
    is_valid BOOLEAN,
    message TEXT,
    actual_feature_order_hash VARCHAR(16),
    has_model_id BOOLEAN,
    has_dataset_hash BOOLEAN,
    has_norm_stats_hash BOOLEAN
) AS $$
DECLARE
    v_run experiment_runs%ROWTYPE;
BEGIN
    SELECT * INTO v_run FROM experiment_runs WHERE run_id = p_run_id;

    IF NOT FOUND THEN
        RETURN QUERY SELECT
            false,
            format('Run %s not found', p_run_id),
            NULL::VARCHAR(16),
            false,
            false,
            false;
        RETURN;
    END IF;

    -- Check SSOT alignment
    IF p_expected_feature_order_hash IS NOT NULL
       AND v_run.feature_order_hash != p_expected_feature_order_hash THEN
        RETURN QUERY SELECT
            false,
            format('Feature order hash mismatch: expected %s, got %s',
                   p_expected_feature_order_hash, v_run.feature_order_hash),
            v_run.feature_order_hash,
            v_run.model_id IS NOT NULL,
            v_run.dataset_hash IS NOT NULL,
            v_run.norm_stats_hash IS NOT NULL;
        RETURN;
    END IF;

    RETURN QUERY SELECT
        true,
        'SSOT alignment validated',
        v_run.feature_order_hash,
        v_run.model_id IS NOT NULL,
        v_run.dataset_hash IS NOT NULL,
        v_run.norm_stats_hash IS NOT NULL;
END;
$$ LANGUAGE plpgsql;

-- -----------------------------------------------------------------------------
-- Triggers
-- -----------------------------------------------------------------------------

-- Trigger: Update timestamp on experiment_runs update
CREATE OR REPLACE FUNCTION fn_update_experiment_runs_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_experiment_runs_updated ON experiment_runs;
CREATE TRIGGER trg_experiment_runs_updated
    BEFORE UPDATE ON experiment_runs
    FOR EACH ROW
    EXECUTE FUNCTION fn_update_experiment_runs_timestamp();

-- Trigger: Update timestamp on experiment_deployments update
CREATE OR REPLACE FUNCTION fn_update_experiment_deployments_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_experiment_deployments_updated ON experiment_deployments;
CREATE TRIGGER trg_experiment_deployments_updated
    BEFORE UPDATE ON experiment_deployments
    FOR EACH ROW
    EXECUTE FUNCTION fn_update_experiment_deployments_timestamp();

-- -----------------------------------------------------------------------------
-- Comments
-- -----------------------------------------------------------------------------
COMMENT ON TABLE experiment_runs IS
    'Stores individual experiment run results with SSOT traceability (dataset_hash, feature_order_hash, model_id)';
COMMENT ON TABLE experiment_comparisons IS
    'Stores A/B comparison results between experiments with sample size tracking';
COMMENT ON TABLE experiment_deployments IS
    'Tracks model deployment decisions, rollouts, and rollbacks';

COMMENT ON COLUMN experiment_runs.dataset_hash IS
    'SHA256 of training dataset from L2Output - enables reproducibility';
COMMENT ON COLUMN experiment_runs.feature_order_hash IS
    'Hash of FEATURE_ORDER tuple from CTR-FEATURE-001 - validates contract alignment';
COMMENT ON COLUMN experiment_runs.norm_stats_hash IS
    'SHA256 of norm_stats.json from L3Output - ensures training/inference parity';
COMMENT ON COLUMN experiment_runs.model_id IS
    'Foreign key to model_registry.model_id - links experiment to deployed model';

COMMENT ON VIEW v_latest_experiment_runs IS 'Latest successful run per experiment with SSOT columns';
COMMENT ON VIEW v_experiment_summary IS 'Summary statistics for each experiment';
COMMENT ON VIEW v_current_deployments IS 'Currently deployed models per environment with full context';
COMMENT ON VIEW v_recent_comparisons IS 'Recent A/B comparisons with results';
COMMENT ON VIEW v_experiment_model_lineage IS 'Experiment runs with model registry linkage for audit trail';

COMMENT ON FUNCTION fn_get_best_experiment_run IS 'Get the best run for an experiment by metric with SSOT columns';
COMMENT ON FUNCTION fn_record_deployment IS 'Record a new deployment and handle previous deployment rollback';
COMMENT ON FUNCTION fn_validate_experiment_ssot IS 'Validate that an experiment run aligns with expected SSOT values';

-- -----------------------------------------------------------------------------
-- Verification
-- -----------------------------------------------------------------------------
DO $$
DECLARE
    v_tables_ok BOOLEAN := true;
BEGIN
    -- Check all tables exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'experiment_runs') THEN
        v_tables_ok := false;
        RAISE WARNING '⚠ experiment_runs table NOT created';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'experiment_comparisons') THEN
        v_tables_ok := false;
        RAISE WARNING '⚠ experiment_comparisons table NOT created';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'experiment_deployments') THEN
        v_tables_ok := false;
        RAISE WARNING '⚠ experiment_deployments table NOT created';
    END IF;

    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    IF v_tables_ok THEN
        RAISE NOTICE 'Experiment Registry Created Successfully';
    ELSE
        RAISE WARNING 'Experiment Registry has ERRORS - check logs';
    END IF;
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'TABLE: experiment_runs (with SSOT columns: dataset_hash, feature_order_hash, model_id)';
    RAISE NOTICE 'TABLE: experiment_comparisons (with sample_size, test_duration)';
    RAISE NOTICE 'TABLE: experiment_deployments';
    RAISE NOTICE 'VIEW: v_latest_experiment_runs';
    RAISE NOTICE 'VIEW: v_experiment_summary';
    RAISE NOTICE 'VIEW: v_current_deployments';
    RAISE NOTICE 'VIEW: v_recent_comparisons';
    RAISE NOTICE 'VIEW: v_experiment_model_lineage';
    RAISE NOTICE 'FUNCTION: fn_get_best_experiment_run()';
    RAISE NOTICE 'FUNCTION: fn_record_deployment()';
    RAISE NOTICE 'FUNCTION: fn_validate_experiment_ssot()';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;
