-- =====================================================
-- USDCOP Trading System - Model Registry
-- Trazabilidad de Modelos para RL Trading
-- =====================================================
--
-- Source: database/migrations/005_model_registry.sql
-- Copied to init-scripts for Docker auto-initialization
--
-- This table stores metadata about trained models including:
-- - Model identification and versioning
-- - Integrity hashes (SHA256) for model and config files
-- - Training parameters and performance metrics
-- - Deployment status lifecycle
--
-- CLAUDE-T15 | Plan Item: P1-11 | Contrato: CTR-010
-- =====================================================

-- =============================================================================
-- 1. TABLA PRINCIPAL: model_registry
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) UNIQUE NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    model_path TEXT NOT NULL,

    -- MinIO-First Architecture: S3 URIs
    s3_model_uri TEXT,                      -- s3://production/models/{model_id}/policy.zip
    s3_norm_stats_uri TEXT,                 -- s3://production/models/{model_id}/norm_stats.json
    s3_config_uri TEXT,                     -- s3://production/models/{model_id}/config.yaml
    source_experiment_id VARCHAR(100),      -- Original experiment ID
    source_experiment_version VARCHAR(50),  -- Original experiment version

    -- Hashes de integridad (SHA256)
    model_hash VARCHAR(64) NOT NULL,
    norm_stats_hash VARCHAR(64) NOT NULL,
    config_hash VARCHAR(64),
    feature_order_hash VARCHAR(64),         -- Hash of feature order for contract validation

    -- Metadata del modelo
    observation_dim INTEGER NOT NULL DEFAULT 15,
    action_space INTEGER NOT NULL DEFAULT 3,
    feature_order JSONB NOT NULL,

    -- Training info
    training_dataset_id INTEGER,
    training_start_date DATE,
    training_end_date DATE,
    validation_metrics JSONB,
    training_duration_seconds NUMERIC(10,2),
    mlflow_run_id VARCHAR(100),             -- MLflow tracking

    -- Performance metrics (from backtest)
    test_sharpe NUMERIC(8,4),
    test_max_drawdown NUMERIC(8,4),
    test_win_rate NUMERIC(5,4),
    test_total_return NUMERIC(10,6),
    test_total_trades INTEGER,

    -- Reward System Configuration (CTR-REWARD-SNAPSHOT-001)
    reward_contract_id VARCHAR(20),             -- e.g., "v1.0.0" (SSOT reference)
    reward_config_hash VARCHAR(64),             -- SHA256 hash for lineage tracking
    s3_reward_config_uri TEXT,                  -- s3://experiments/{exp}/reward_configs/{ver}/reward_config.json
    reward_weights JSONB,                       -- {"pnl": 0.5, "dsr": 0.3, "sortino": 0.2, ...}
    curriculum_final_phase VARCHAR(20),         -- "phase_1", "phase_2", or "phase_3"
    reward_normalization_enabled BOOLEAN DEFAULT true,
    curriculum_enabled BOOLEAN DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP WITH TIME ZONE,
    retired_at TIMESTAMP WITH TIME ZONE,

    -- Status: registered -> deployed -> retired
    status VARCHAR(20) DEFAULT 'registered',

    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('registered', 'deployed', 'retired'))
);

-- =============================================================================
-- 2. INDICES
-- =============================================================================

CREATE INDEX IF NOT EXISTS idx_model_registry_hash
    ON model_registry(model_hash);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON model_registry(status);

CREATE INDEX IF NOT EXISTS idx_model_registry_version
    ON model_registry(model_version);

CREATE INDEX IF NOT EXISTS idx_model_registry_created
    ON model_registry(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_registry_experiment
    ON model_registry(source_experiment_id);

CREATE INDEX IF NOT EXISTS idx_model_registry_mlflow
    ON model_registry(mlflow_run_id);

CREATE INDEX IF NOT EXISTS idx_model_registry_reward_contract
    ON model_registry(reward_contract_id);

-- =============================================================================
-- 3. COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE model_registry IS
    'Registro de modelos ONNX/ZIP con hashes para verificacion de integridad. CTR-010';

COMMENT ON COLUMN model_registry.model_hash IS
    'SHA256 del archivo modelo (ONNX/ZIP) - verificar que no ha sido modificado';

COMMENT ON COLUMN model_registry.norm_stats_hash IS
    'SHA256 del archivo norm_stats.json - debe coincidir con el usado durante training';

COMMENT ON COLUMN model_registry.feature_order IS
    'Orden exacto de features como array JSON - debe coincidir con FeatureBuilder';

COMMENT ON COLUMN model_registry.status IS
    'Ciclo de vida: registered (nuevo) -> deployed (en produccion) -> retired (obsoleto)';

COMMENT ON COLUMN model_registry.reward_contract_id IS
    'Version of reward contract (SSOT reference, e.g., v1.0.0). Used for reproducibility and A/B testing.';

COMMENT ON COLUMN model_registry.reward_config_hash IS
    'SHA256 hash of reward configuration JSON. Used for lineage tracking and verification.';

COMMENT ON COLUMN model_registry.reward_weights IS
    'JSONB with component weights: {pnl, dsr, sortino, regime_penalty, holding_decay, anti_gaming}';

COMMENT ON COLUMN model_registry.curriculum_final_phase IS
    'Final phase reached during curriculum training: phase_1, phase_2, or phase_3';

-- =============================================================================
-- 4. VISTA: active_models
-- =============================================================================
-- Returns only deployed models, ordered by deployment date (most recent first)

CREATE OR REPLACE VIEW active_models AS
SELECT
    model_id,
    model_version,
    model_path,
    s3_model_uri,
    s3_norm_stats_uri,
    s3_config_uri,
    source_experiment_id,
    source_experiment_version,
    model_hash,
    norm_stats_hash,
    feature_order_hash,
    observation_dim,
    action_space,
    feature_order,
    mlflow_run_id,
    deployed_at,
    test_sharpe,
    test_max_drawdown,
    test_win_rate,
    test_total_return,
    test_total_trades,
    -- Reward system fields
    reward_contract_id,
    reward_config_hash,
    s3_reward_config_uri,
    reward_weights,
    curriculum_final_phase,
    reward_normalization_enabled,
    curriculum_enabled
FROM model_registry
WHERE status = 'deployed'
ORDER BY deployed_at DESC;

COMMENT ON VIEW active_models IS
    'Models currently deployed in production, ordered by deployment date. Includes S3 URIs for MinIO-first architecture and reward configuration tracking.';

-- =============================================================================
-- 5. FUNCION: verify_model_integrity
-- =============================================================================
-- Verifies if a model is registered in the registry
-- Note: Full integrity check requires file system access (done in Python)

CREATE OR REPLACE FUNCTION verify_model_integrity(p_model_id VARCHAR)
RETURNS TABLE (
    is_valid BOOLEAN,
    registered_hash VARCHAR,
    message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        true as is_valid,
        mr.model_hash as registered_hash,
        'Model registered - integrity check requires file access' as message
    FROM model_registry mr
    WHERE mr.model_id = p_model_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT
            false as is_valid,
            null::VARCHAR as registered_hash,
            format('Model %s not found in registry', p_model_id) as message;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION verify_model_integrity IS
    'Check if model exists in registry. Full hash verification done in Python.';

-- =============================================================================
-- 6. FUNCION: get_production_model
-- =============================================================================
-- Returns the currently deployed production model (most recently deployed)

CREATE OR REPLACE FUNCTION get_production_model()
RETURNS TABLE (
    model_id VARCHAR,
    model_version VARCHAR,
    model_path TEXT,
    model_hash VARCHAR,
    norm_stats_hash VARCHAR,
    feature_order JSONB,
    test_sharpe NUMERIC,
    deployed_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        mr.model_id,
        mr.model_version,
        mr.model_path,
        mr.model_hash,
        mr.norm_stats_hash,
        mr.feature_order,
        mr.test_sharpe,
        mr.deployed_at
    FROM model_registry mr
    WHERE mr.status = 'deployed'
    ORDER BY mr.deployed_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_production_model IS
    'Returns the most recently deployed production model';

-- =============================================================================
-- 7. FUNCION: deploy_model
-- =============================================================================
-- Deploys a model (sets status to deployed) and optionally retires the previous one

CREATE OR REPLACE FUNCTION deploy_model(
    p_model_id VARCHAR,
    p_retire_previous BOOLEAN DEFAULT true
)
RETURNS BOOLEAN AS $$
DECLARE
    v_model_exists BOOLEAN;
BEGIN
    -- Check if model exists
    SELECT EXISTS(
        SELECT 1 FROM model_registry WHERE model_id = p_model_id
    ) INTO v_model_exists;

    IF NOT v_model_exists THEN
        RAISE WARNING 'Model % not found in registry', p_model_id;
        RETURN false;
    END IF;

    -- Optionally retire previous deployed models
    IF p_retire_previous THEN
        UPDATE model_registry
        SET status = 'retired',
            retired_at = CURRENT_TIMESTAMP
        WHERE status = 'deployed'
          AND model_id != p_model_id;
    END IF;

    -- Deploy the new model
    UPDATE model_registry
    SET status = 'deployed',
        deployed_at = CURRENT_TIMESTAMP
    WHERE model_id = p_model_id;

    RETURN true;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION deploy_model IS
    'Deploys a model to production, optionally retiring previous models';

-- =============================================================================
-- 8. VERIFICATION AND LOGGING
-- =============================================================================

DO $$
DECLARE
    table_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'model_registry'
    ) INTO table_exists;

    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    IF table_exists THEN
        RAISE NOTICE 'Model Registry Created Successfully';
    ELSE
        RAISE WARNING 'Model Registry NOT Created - Check for errors';
    END IF;
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'TABLE: model_registry (model metadata and hashes)';
    RAISE NOTICE 'VIEW: active_models (deployed models only)';
    RAISE NOTICE 'FUNCTION: verify_model_integrity(model_id)';
    RAISE NOTICE 'FUNCTION: get_production_model()';
    RAISE NOTICE 'FUNCTION: deploy_model(model_id, retire_previous)';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;
