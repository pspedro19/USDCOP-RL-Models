-- =============================================================================
-- Migration: 025_forecast_experiments.sql
-- =============================================================================
-- Experiment registry for forecasting A/B testing.
-- Tracks experiment runs, comparisons, and deployment decisions.
--
-- Tables:
--   1. bi.forecast_experiment_runs - Individual experiment executions
--   2. bi.forecast_experiment_comparisons - A/B test results
--   3. bi.forecast_experiment_deployments - Deployment tracking
--
-- Author: Trading Team
-- Date: 2026-01-22
-- Contract: CTR-FORECAST-EXPERIMENT-001
-- =============================================================================

-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS bi;

-- =============================================================================
-- TABLE 1: Experiment Runs
-- =============================================================================
-- Stores individual experiment run records with full configuration and results.

CREATE TABLE IF NOT EXISTS bi.forecast_experiment_runs (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    run_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- Experiment Identity
    experiment_name VARCHAR(200) NOT NULL,
    experiment_version VARCHAR(50) NOT NULL DEFAULT '1.0.0',
    run_id VARCHAR(100) NOT NULL,
    parent_run_id VARCHAR(100),  -- For experiment chains

    -- Configuration (SSOT references)
    config_hash VARCHAR(64) NOT NULL,
    config_json JSONB NOT NULL,
    models_included TEXT[] NOT NULL,
    horizons_included INT[] NOT NULL DEFAULT ARRAY[1, 5, 10, 15, 20, 25, 30],

    -- Feature configuration (SSOT)
    feature_contract_version VARCHAR(20),
    feature_columns TEXT[],
    feature_hash VARCHAR(64),

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'success', 'failed', 'cancelled', 'dry_run')),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    duration_seconds INT,

    -- Training Results (per model)
    training_metrics JSONB,
    /*
    {
        "ridge": {"train_rmse": 0.02, "val_rmse": 0.025, "epochs": 100},
        "xgboost_pure": {"train_rmse": 0.018, "val_rmse": 0.022, "n_estimators": 50},
        ...
    }
    */

    -- Backtest Results (per model/horizon)
    backtest_metrics JSONB,
    /*
    {
        "ridge": {
            "1": {"direction_accuracy": 0.55, "rmse": 0.012, "sharpe": 0.8},
            "5": {"direction_accuracy": 0.52, "rmse": 0.025, "sharpe": 0.5},
            ...
        },
        ...
    }
    */

    -- Aggregate Metrics (summary)
    aggregate_metrics JSONB,
    /*
    {
        "avg_direction_accuracy": 0.54,
        "best_model_per_horizon": {"1": "xgboost_pure", "5": "ridge", ...},
        "overall_sharpe": 0.65
    }
    */

    -- Model Artifacts
    model_artifacts_path VARCHAR(500),
    minio_artifacts_uri VARCHAR(500),

    -- MLflow Integration
    mlflow_experiment_id VARCHAR(100),
    mlflow_run_ids JSONB,  -- {"model_id": "run_id", ...}

    -- Dataset Lineage
    dataset_path VARCHAR(500),
    dataset_hash VARCHAR(64),
    dataset_date_range JSONB,  -- {"start": "2020-01-01", "end": "2025-12-31"}

    -- Error Tracking
    error_message TEXT,
    error_traceback TEXT,

    -- Metadata
    created_by VARCHAR(100) DEFAULT 'system',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    UNIQUE(experiment_name, run_id)
);

-- Indexes for experiment_runs
CREATE INDEX IF NOT EXISTS idx_forecast_exp_runs_name
    ON bi.forecast_experiment_runs (experiment_name, experiment_version);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_runs_status
    ON bi.forecast_experiment_runs (status);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_runs_started
    ON bi.forecast_experiment_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_runs_mlflow
    ON bi.forecast_experiment_runs (mlflow_experiment_id);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_runs_config_hash
    ON bi.forecast_experiment_runs (config_hash);

-- Update trigger
CREATE OR REPLACE FUNCTION bi.update_forecast_exp_runs_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_forecast_exp_runs_updated ON bi.forecast_experiment_runs;
CREATE TRIGGER trg_forecast_exp_runs_updated
    BEFORE UPDATE ON bi.forecast_experiment_runs
    FOR EACH ROW
    EXECUTE FUNCTION bi.update_forecast_exp_runs_timestamp();


-- =============================================================================
-- TABLE 2: Experiment Comparisons (A/B Tests)
-- =============================================================================
-- Stores A/B comparison results between experiments.

CREATE TABLE IF NOT EXISTS bi.forecast_experiment_comparisons (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    comparison_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- Experiments Being Compared
    baseline_experiment VARCHAR(200) NOT NULL,
    baseline_version VARCHAR(50) NOT NULL,
    baseline_run_id VARCHAR(100) NOT NULL,
    treatment_experiment VARCHAR(200) NOT NULL,
    treatment_version VARCHAR(50) NOT NULL,
    treatment_run_id VARCHAR(100) NOT NULL,

    -- Primary Metric
    primary_metric VARCHAR(50) NOT NULL DEFAULT 'direction_accuracy',
    significance_level DECIMAL(5,4) NOT NULL DEFAULT 0.05,
    bonferroni_corrected BOOLEAN DEFAULT TRUE,

    -- Per-Horizon Results
    horizon_results JSONB NOT NULL,
    /*
    {
        "1": {
            "baseline_metric": 0.55,
            "treatment_metric": 0.58,
            "metric_difference": 0.03,
            "p_value": 0.02,
            "significant": true,
            "winner": "treatment"
        },
        ...
    }
    */

    -- Statistical Tests Details
    statistical_tests JSONB,
    /*
    {
        "test_method": "mcnemar",
        "aggregate_method": "fisher_combined",
        "n_comparisons": 7,
        "adjusted_alpha": 0.00714
    }
    */

    -- Aggregate Results
    aggregate_p_value DECIMAL(10,8),
    aggregate_significant BOOLEAN,
    treatment_wins INT DEFAULT 0,
    baseline_wins INT DEFAULT 0,
    ties INT DEFAULT 0,

    -- Recommendation
    recommendation VARCHAR(50) NOT NULL
        CHECK (recommendation IN ('deploy_treatment', 'keep_baseline', 'inconclusive', 'needs_more_data')),
    confidence_score DECIMAL(5,4),
    recommendation_reason TEXT,

    -- Warnings
    warnings TEXT[],

    -- Metadata
    compared_at TIMESTAMPTZ DEFAULT NOW(),
    compared_by VARCHAR(100) DEFAULT 'system',

    -- Foreign Keys (soft - experiments may be in different systems)
    CONSTRAINT fk_baseline_run FOREIGN KEY (baseline_experiment, baseline_run_id)
        REFERENCES bi.forecast_experiment_runs(experiment_name, run_id)
        ON DELETE SET NULL,
    CONSTRAINT fk_treatment_run FOREIGN KEY (treatment_experiment, treatment_run_id)
        REFERENCES bi.forecast_experiment_runs(experiment_name, run_id)
        ON DELETE SET NULL
);

-- Indexes for comparisons
CREATE INDEX IF NOT EXISTS idx_forecast_exp_comp_baseline
    ON bi.forecast_experiment_comparisons (baseline_experiment, baseline_run_id);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_comp_treatment
    ON bi.forecast_experiment_comparisons (treatment_experiment, treatment_run_id);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_comp_date
    ON bi.forecast_experiment_comparisons (compared_at DESC);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_comp_recommendation
    ON bi.forecast_experiment_comparisons (recommendation);


-- =============================================================================
-- TABLE 3: Experiment Deployments
-- =============================================================================
-- Tracks which experiments have been deployed to production.

CREATE TABLE IF NOT EXISTS bi.forecast_experiment_deployments (
    -- Primary Key
    id SERIAL PRIMARY KEY,
    deployment_uuid UUID DEFAULT gen_random_uuid() UNIQUE,

    -- Deployed Experiment
    experiment_name VARCHAR(200) NOT NULL,
    experiment_version VARCHAR(50) NOT NULL,
    run_id VARCHAR(100) NOT NULL,

    -- Deployment Details
    deployment_type VARCHAR(20) NOT NULL DEFAULT 'full'
        CHECK (deployment_type IN ('full', 'shadow', 'rollback')),
    target_environment VARCHAR(20) NOT NULL DEFAULT 'production'
        CHECK (target_environment IN ('staging', 'production')),

    -- Status
    status VARCHAR(20) NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'deploying', 'deployed', 'rolled_back', 'failed')),

    -- Comparison Reference
    comparison_id INT REFERENCES bi.forecast_experiment_comparisons(id),

    -- Model Artifacts
    model_artifacts_path VARCHAR(500),
    deployed_models TEXT[],

    -- Timing
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    deployed_at TIMESTAMPTZ,
    rolled_back_at TIMESTAMPTZ,

    -- Approval
    approval_required BOOLEAN DEFAULT FALSE,
    approved_by VARCHAR(100),
    approved_at TIMESTAMPTZ,

    -- Previous Deployment (for rollback)
    previous_experiment VARCHAR(200),
    previous_version VARCHAR(50),
    previous_run_id VARCHAR(100),
    rollback_reason TEXT,

    -- Metadata
    triggered_by VARCHAR(100) DEFAULT 'system',
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT fk_deployed_run FOREIGN KEY (experiment_name, run_id)
        REFERENCES bi.forecast_experiment_runs(experiment_name, run_id)
        ON DELETE SET NULL
);

-- Indexes for deployments
CREATE INDEX IF NOT EXISTS idx_forecast_exp_deploy_exp
    ON bi.forecast_experiment_deployments (experiment_name, run_id);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_deploy_status
    ON bi.forecast_experiment_deployments (status);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_deploy_env
    ON bi.forecast_experiment_deployments (target_environment, status);
CREATE INDEX IF NOT EXISTS idx_forecast_exp_deploy_date
    ON bi.forecast_experiment_deployments (deployed_at DESC);


-- =============================================================================
-- VIEWS
-- =============================================================================

-- View: Latest successful run per experiment
CREATE OR REPLACE VIEW bi.v_forecast_latest_experiment_runs AS
SELECT DISTINCT ON (experiment_name)
    id,
    run_uuid,
    experiment_name,
    experiment_version,
    run_id,
    status,
    started_at,
    completed_at,
    duration_seconds,
    aggregate_metrics,
    mlflow_experiment_id,
    created_at
FROM bi.forecast_experiment_runs
WHERE status = 'success'
ORDER BY experiment_name, completed_at DESC;


-- View: Experiment summary with aggregate stats
CREATE OR REPLACE VIEW bi.v_forecast_experiment_summary AS
SELECT
    experiment_name,
    COUNT(*) as total_runs,
    COUNT(*) FILTER (WHERE status = 'success') as successful_runs,
    COUNT(*) FILTER (WHERE status = 'failed') as failed_runs,
    AVG(duration_seconds) FILTER (WHERE status = 'success') as avg_duration_seconds,
    MAX(completed_at) as last_run_at,
    MAX(experiment_version) as latest_version
FROM bi.forecast_experiment_runs
GROUP BY experiment_name
ORDER BY last_run_at DESC;


-- View: Recent A/B comparisons
CREATE OR REPLACE VIEW bi.v_forecast_recent_comparisons AS
SELECT
    c.comparison_uuid,
    c.baseline_experiment,
    c.baseline_run_id,
    c.treatment_experiment,
    c.treatment_run_id,
    c.primary_metric,
    c.recommendation,
    c.confidence_score,
    c.treatment_wins,
    c.baseline_wins,
    c.ties,
    c.aggregate_p_value,
    c.aggregate_significant,
    c.compared_at
FROM bi.forecast_experiment_comparisons c
ORDER BY c.compared_at DESC
LIMIT 50;


-- View: Current deployed experiments
CREATE OR REPLACE VIEW bi.v_forecast_current_deployments AS
SELECT
    d.deployment_uuid,
    d.experiment_name,
    d.experiment_version,
    d.run_id,
    d.target_environment,
    d.status,
    d.deployed_at,
    r.aggregate_metrics,
    r.backtest_metrics
FROM bi.forecast_experiment_deployments d
LEFT JOIN bi.forecast_experiment_runs r
    ON d.experiment_name = r.experiment_name AND d.run_id = r.run_id
WHERE d.status = 'deployed'
ORDER BY d.deployed_at DESC;


-- =============================================================================
-- FUNCTIONS
-- =============================================================================

-- Function: Get best experiment run by metric
CREATE OR REPLACE FUNCTION bi.fn_get_best_forecast_experiment(
    p_metric_name VARCHAR DEFAULT 'avg_direction_accuracy',
    p_horizon INT DEFAULT NULL
)
RETURNS TABLE (
    experiment_name VARCHAR,
    run_id VARCHAR,
    metric_value DECIMAL,
    completed_at TIMESTAMPTZ
) AS $$
BEGIN
    IF p_horizon IS NULL THEN
        -- Aggregate metric
        RETURN QUERY
        SELECT
            r.experiment_name,
            r.run_id,
            (r.aggregate_metrics->>p_metric_name)::DECIMAL as metric_value,
            r.completed_at
        FROM bi.forecast_experiment_runs r
        WHERE r.status = 'success'
          AND r.aggregate_metrics ? p_metric_name
        ORDER BY metric_value DESC
        LIMIT 1;
    ELSE
        -- Horizon-specific metric (average across models)
        RETURN QUERY
        SELECT
            r.experiment_name,
            r.run_id,
            AVG((model_metrics.value->>p_horizon::text->>p_metric_name)::DECIMAL) as metric_value,
            r.completed_at
        FROM bi.forecast_experiment_runs r,
             jsonb_each(r.backtest_metrics) as model_metrics
        WHERE r.status = 'success'
        GROUP BY r.experiment_name, r.run_id, r.completed_at
        ORDER BY metric_value DESC
        LIMIT 1;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- Function: Record new deployment
CREATE OR REPLACE FUNCTION bi.fn_record_forecast_deployment(
    p_experiment_name VARCHAR,
    p_experiment_version VARCHAR,
    p_run_id VARCHAR,
    p_target_environment VARCHAR DEFAULT 'production',
    p_comparison_id INT DEFAULT NULL,
    p_triggered_by VARCHAR DEFAULT 'system'
)
RETURNS UUID AS $$
DECLARE
    v_deployment_uuid UUID;
    v_previous_exp VARCHAR;
    v_previous_version VARCHAR;
    v_previous_run VARCHAR;
BEGIN
    -- Get current deployment to track for rollback
    SELECT experiment_name, experiment_version, run_id
    INTO v_previous_exp, v_previous_version, v_previous_run
    FROM bi.forecast_experiment_deployments
    WHERE target_environment = p_target_environment
      AND status = 'deployed'
    ORDER BY deployed_at DESC
    LIMIT 1;

    -- Mark previous deployment as rolled back
    UPDATE bi.forecast_experiment_deployments
    SET status = 'rolled_back',
        rolled_back_at = NOW()
    WHERE target_environment = p_target_environment
      AND status = 'deployed';

    -- Insert new deployment
    INSERT INTO bi.forecast_experiment_deployments (
        experiment_name,
        experiment_version,
        run_id,
        target_environment,
        status,
        comparison_id,
        previous_experiment,
        previous_version,
        previous_run_id,
        triggered_by,
        deployed_at
    ) VALUES (
        p_experiment_name,
        p_experiment_version,
        p_run_id,
        p_target_environment,
        'deployed',
        p_comparison_id,
        v_previous_exp,
        v_previous_version,
        v_previous_run,
        p_triggered_by,
        NOW()
    )
    RETURNING deployment_uuid INTO v_deployment_uuid;

    RETURN v_deployment_uuid;
END;
$$ LANGUAGE plpgsql;


-- Function: Rollback to previous deployment
CREATE OR REPLACE FUNCTION bi.fn_rollback_forecast_deployment(
    p_target_environment VARCHAR DEFAULT 'production',
    p_reason TEXT DEFAULT 'Manual rollback',
    p_triggered_by VARCHAR DEFAULT 'system'
)
RETURNS UUID AS $$
DECLARE
    v_current RECORD;
    v_deployment_uuid UUID;
BEGIN
    -- Get current deployment with previous info
    SELECT *
    INTO v_current
    FROM bi.forecast_experiment_deployments
    WHERE target_environment = p_target_environment
      AND status = 'deployed'
    ORDER BY deployed_at DESC
    LIMIT 1;

    IF v_current IS NULL THEN
        RAISE EXCEPTION 'No active deployment found for environment %', p_target_environment;
    END IF;

    IF v_current.previous_experiment IS NULL THEN
        RAISE EXCEPTION 'No previous deployment to rollback to';
    END IF;

    -- Mark current as rolled back
    UPDATE bi.forecast_experiment_deployments
    SET status = 'rolled_back',
        rolled_back_at = NOW(),
        rollback_reason = p_reason
    WHERE id = v_current.id;

    -- Deploy previous
    v_deployment_uuid := bi.fn_record_forecast_deployment(
        v_current.previous_experiment,
        v_current.previous_version,
        v_current.previous_run_id,
        p_target_environment,
        NULL,
        p_triggered_by
    );

    RETURN v_deployment_uuid;
END;
$$ LANGUAGE plpgsql;


-- =============================================================================
-- COMMENTS
-- =============================================================================

COMMENT ON TABLE bi.forecast_experiment_runs IS
    'Forecasting experiment runs with full configuration and results. Contract: CTR-FORECAST-EXPERIMENT-001';

COMMENT ON TABLE bi.forecast_experiment_comparisons IS
    'A/B test comparisons between forecasting experiments with statistical analysis';

COMMENT ON TABLE bi.forecast_experiment_deployments IS
    'Deployment history for forecasting experiments with rollback support';

COMMENT ON VIEW bi.v_forecast_latest_experiment_runs IS
    'Latest successful run for each experiment (for quick lookups)';

COMMENT ON VIEW bi.v_forecast_experiment_summary IS
    'Aggregate statistics per experiment across all runs';

COMMENT ON FUNCTION bi.fn_get_best_forecast_experiment IS
    'Find the best performing experiment by a given metric';

COMMENT ON FUNCTION bi.fn_record_forecast_deployment IS
    'Atomic deployment recording with automatic previous tracking';

COMMENT ON FUNCTION bi.fn_rollback_forecast_deployment IS
    'Rollback to previous deployment with full audit trail';
