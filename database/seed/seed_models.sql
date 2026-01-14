-- ============================================================================
-- Model Registry Seed Data
-- ============================================================================
-- This script creates the models schema and seeds the initial model registry
-- with PPO Primary (production), PPO Legacy (deprecated), SAC and TD3 (testing)
-- ============================================================================

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS models;

-- ============================================================================
-- Model Registry Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS models.model_registry (
    model_id VARCHAR(50) PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    algorithm VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'testing',
    hyperparameters JSONB,
    policy_config JSONB,
    environment_config JSONB,
    model_path VARCHAR(500) NOT NULL,
    feature_dim INTEGER NOT NULL DEFAULT 15,
    validation_metrics JSONB,
    risk_limits JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    CONSTRAINT valid_status CHECK (status IN ('production', 'testing', 'deprecated', 'training')),
    CONSTRAINT valid_algorithm CHECK (algorithm IN ('PPO', 'SAC', 'TD3', 'A2C', 'DQN'))
);

-- Create index for status queries
CREATE INDEX IF NOT EXISTS idx_model_registry_status
ON models.model_registry(status);

-- Create index for algorithm queries
CREATE INDEX IF NOT EXISTS idx_model_registry_algorithm
ON models.model_registry(algorithm);

-- ============================================================================
-- Model Trades Table (for tracking live performance)
-- ============================================================================
CREATE TABLE IF NOT EXISTS models.model_trades (
    trade_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models.model_registry(model_id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action VARCHAR(10) NOT NULL,
    entry_price NUMERIC(12, 4),
    exit_price NUMERIC(12, 4),
    position_size NUMERIC(10, 4),
    pnl NUMERIC(12, 4),
    fees NUMERIC(10, 4),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_model_trades_model_id
ON models.model_trades(model_id);

CREATE INDEX IF NOT EXISTS idx_model_trades_timestamp
ON models.model_trades(timestamp DESC);

-- ============================================================================
-- Model Signals Table (for tracking predictions)
-- ============================================================================
CREATE TABLE IF NOT EXISTS models.model_signals (
    signal_id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) REFERENCES models.model_registry(model_id),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    action VARCHAR(10) NOT NULL,
    raw_action NUMERIC(8, 6),
    confidence NUMERIC(5, 4),
    features_hash VARCHAR(64),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_model_signals_model_id
ON models.model_signals(model_id);

CREATE INDEX IF NOT EXISTS idx_model_signals_timestamp
ON models.model_signals(timestamp DESC);

-- ============================================================================
-- Trigger for updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION models.update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS update_model_registry_updated_at ON models.model_registry;
CREATE TRIGGER update_model_registry_updated_at
    BEFORE UPDATE ON models.model_registry
    FOR EACH ROW
    EXECUTE FUNCTION models.update_updated_at_column();

-- ============================================================================
-- SEED DATA: PPO Primary (Production Model)
-- ============================================================================
-- Configuration extracted from PRODUCTION_CONFIG.json
-- This is the validated production model with stress tests passed
-- ============================================================================

INSERT INTO models.model_registry (
    model_id,
    model_name,
    algorithm,
    version,
    status,
    hyperparameters,
    policy_config,
    environment_config,
    model_path,
    feature_dim,
    validation_metrics,
    risk_limits
) VALUES (
    'ppo_primary',
    'Model B (Aggressive)',
    'PPO',
    'current',
    'production',
    -- Hyperparameters from PRODUCTION_CONFIG.json
    '{
        "learning_rate": 0.0001,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5
    }'::jsonb,
    -- Policy configuration
    '{
        "type": "MlpPolicy",
        "net_arch": [256, 256],
        "activation_fn": "Tanh"
    }'::jsonb,
    -- Environment configuration
    '{
        "class": "TradingEnvironment",
        "initial_balance": 10000,
        "max_position": 1.0,
        "episode_length": 400,
        "use_vol_scaling": true,
        "use_regime_detection": true,
        "bars_per_day": 56
    }'::jsonb,
    -- Model path
    'models/ppo_primary.zip',
    -- Feature dimension
    21,
    -- Validation results from PRODUCTION_CONFIG.json
    '{
        "stress_tests": {
            "passed": true,
            "pass_rate": 0.60,
            "periods_tested": 5,
            "periods_passed": 3
        },
        "five_fold_cv": {
            "passed": true,
            "mean_sharpe": 2.21,
            "std_sharpe": 2.01,
            "mean_max_dd": 0.002,
            "positive_sharpe_folds": 4
        },
        "validated_date": "2025-12-26",
        "training": {
            "total_timesteps": 80000,
            "seed": 42,
            "device": "cpu"
        },
        "data": {
            "source": "RL_DS3_MACRO_CORE.csv",
            "timeframe": "5min",
            "date_range": {
                "start": "2020-03-02",
                "end": "2025-12-05"
            }
        }
    }'::jsonb,
    -- Risk limits from PRODUCTION_CONFIG.json
    '{
        "max_drawdown_threshold": 0.05,
        "max_hold_percentage": 0.90,
        "min_sharpe_30day": 0.0,
        "long_threshold": 0.1,
        "short_threshold": -0.1
    }'::jsonb
)
ON CONFLICT (model_id) DO UPDATE SET
    model_name = EXCLUDED.model_name,
    hyperparameters = EXCLUDED.hyperparameters,
    policy_config = EXCLUDED.policy_config,
    environment_config = EXCLUDED.environment_config,
    validation_metrics = EXCLUDED.validation_metrics,
    risk_limits = EXCLUDED.risk_limits,
    updated_at = NOW();

-- ============================================================================
-- SEED DATA: PPO Legacy (Deprecated Model)
-- ============================================================================
-- Previous production model, now deprecated
-- ============================================================================

INSERT INTO models.model_registry (
    model_id,
    model_name,
    algorithm,
    version,
    status,
    hyperparameters,
    policy_config,
    environment_config,
    model_path,
    feature_dim,
    validation_metrics,
    risk_limits
) VALUES (
    'ppo_legacy',
    'Model A (Balanced)',
    'PPO',
    'legacy',
    'deprecated',
    '{
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5
    }'::jsonb,
    '{
        "type": "MlpPolicy",
        "net_arch": [128, 128],
        "activation_fn": "Tanh"
    }'::jsonb,
    '{
        "class": "TradingEnvironmentLegacy",
        "initial_balance": 10000,
        "max_position": 1.0,
        "episode_length": 288,
        "use_vol_scaling": false,
        "use_regime_detection": false,
        "bars_per_day": 56
    }'::jsonb,
    'models/ppo_legacy.zip',
    18,
    '{
        "stress_tests": {
            "passed": true,
            "pass_rate": 0.40,
            "periods_tested": 5,
            "periods_passed": 2
        },
        "five_fold_cv": {
            "passed": true,
            "mean_sharpe": 1.45,
            "std_sharpe": 1.82,
            "mean_max_dd": 0.008,
            "positive_sharpe_folds": 3
        },
        "validated_date": "2025-10-15",
        "deprecation_reason": "Superseded by current model with better stress test performance"
    }'::jsonb,
    '{
        "max_drawdown_threshold": 0.08,
        "max_hold_percentage": 0.85,
        "min_sharpe_30day": 0.0
    }'::jsonb
)
ON CONFLICT (model_id) DO UPDATE SET
    status = EXCLUDED.status,
    updated_at = NOW();

-- ============================================================================
-- SEED DATA: SAC (Testing Model)
-- ============================================================================
-- Soft Actor-Critic variant for comparison testing
-- ============================================================================

INSERT INTO models.model_registry (
    model_id,
    model_name,
    algorithm,
    version,
    status,
    hyperparameters,
    policy_config,
    environment_config,
    model_path,
    feature_dim,
    validation_metrics,
    risk_limits
) VALUES (
    'sac_experimental',
    'SAC Experimental',
    'SAC',
    'current',
    'testing',
    '{
        "learning_rate": 0.0003,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto"
    }'::jsonb,
    '{
        "type": "MlpPolicy",
        "net_arch": [256, 256],
        "activation_fn": "ReLU"
    }'::jsonb,
    '{
        "class": "TradingEnvironment",
        "initial_balance": 10000,
        "max_position": 1.0,
        "episode_length": 400,
        "use_vol_scaling": true,
        "use_regime_detection": true,
        "bars_per_day": 56
    }'::jsonb,
    'models/sac_experimental.zip',
    21,
    '{
        "stress_tests": {
            "passed": false,
            "pass_rate": 0.20,
            "periods_tested": 5,
            "periods_passed": 1,
            "note": "Needs more training, unstable in high volatility"
        },
        "five_fold_cv": {
            "passed": true,
            "mean_sharpe": 1.85,
            "std_sharpe": 2.45,
            "mean_max_dd": 0.012,
            "positive_sharpe_folds": 3
        },
        "status": "Under evaluation",
        "training": {
            "total_timesteps": 50000,
            "seed": 42
        }
    }'::jsonb,
    '{
        "max_drawdown_threshold": 0.05,
        "max_hold_percentage": 0.90,
        "min_sharpe_30day": 0.0
    }'::jsonb
)
ON CONFLICT (model_id) DO UPDATE SET
    hyperparameters = EXCLUDED.hyperparameters,
    validation_metrics = EXCLUDED.validation_metrics,
    updated_at = NOW();

-- ============================================================================
-- SEED DATA: TD3 (Testing Model)
-- ============================================================================
-- Twin Delayed DDPG variant for comparison testing
-- ============================================================================

INSERT INTO models.model_registry (
    model_id,
    model_name,
    algorithm,
    version,
    status,
    hyperparameters,
    policy_config,
    environment_config,
    model_path,
    feature_dim,
    validation_metrics,
    risk_limits
) VALUES (
    'td3_experimental',
    'TD3 Experimental',
    'TD3',
    'current',
    'testing',
    '{
        "learning_rate": 0.001,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5
    }'::jsonb,
    '{
        "type": "MlpPolicy",
        "net_arch": [256, 256],
        "activation_fn": "ReLU"
    }'::jsonb,
    '{
        "class": "TradingEnvironment",
        "initial_balance": 10000,
        "max_position": 1.0,
        "episode_length": 400,
        "use_vol_scaling": true,
        "use_regime_detection": true,
        "bars_per_day": 56
    }'::jsonb,
    'models/td3_experimental.zip',
    21,
    '{
        "stress_tests": {
            "passed": false,
            "pass_rate": 0.00,
            "periods_tested": 5,
            "periods_passed": 0,
            "note": "Early stage training, not yet validated"
        },
        "five_fold_cv": {
            "passed": false,
            "mean_sharpe": 0.45,
            "std_sharpe": 1.20,
            "mean_max_dd": 0.025,
            "positive_sharpe_folds": 2
        },
        "status": "Early development",
        "training": {
            "total_timesteps": 25000,
            "seed": 42
        }
    }'::jsonb,
    '{
        "max_drawdown_threshold": 0.05,
        "max_hold_percentage": 0.90,
        "min_sharpe_30day": 0.0
    }'::jsonb
)
ON CONFLICT (model_id) DO UPDATE SET
    hyperparameters = EXCLUDED.hyperparameters,
    validation_metrics = EXCLUDED.validation_metrics,
    updated_at = NOW();

-- ============================================================================
-- Verification Query
-- ============================================================================
-- Run this to verify the seed data was inserted correctly

SELECT
    model_id,
    model_name,
    algorithm,
    version,
    status,
    feature_dim,
    validation_metrics->>'validated_date' as validated_date,
    validation_metrics->'stress_tests'->>'pass_rate' as stress_pass_rate,
    validation_metrics->'five_fold_cv'->>'mean_sharpe' as mean_sharpe,
    created_at
FROM models.model_registry
ORDER BY
    CASE status
        WHEN 'production' THEN 1
        WHEN 'testing' THEN 2
        WHEN 'deprecated' THEN 3
    END,
    created_at DESC;

-- ============================================================================
-- Grant permissions (adjust role names as needed)
-- ============================================================================
-- GRANT USAGE ON SCHEMA models TO trading_api;
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA models TO trading_api;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA models TO trading_api;
