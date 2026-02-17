-- Migration 037: Experiment Contracts Table
-- Stores immutable contracts generated from experiment YAMLs
-- Created: 2026-01-31

-- Experiment contracts (inmutables)
-- Generated from config/experiments/*.yaml
CREATE TABLE IF NOT EXISTS experiment_contracts (
    id SERIAL PRIMARY KEY,

    -- Identity
    contract_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    experiment_version VARCHAR(50) NOT NULL,

    -- Hashes for lineage tracking
    config_hash VARCHAR(64) NOT NULL,           -- sha256(yaml_content)[:16]
    feature_order_hash VARCHAR(64) NOT NULL,    -- from feature_contract.py
    reward_config_hash VARCHAR(64),             -- sha256(reward section)[:16]

    -- Complete frozen config (YAML content as JSON)
    frozen_config JSONB NOT NULL,

    -- Source file path (for reference only)
    yaml_path VARCHAR(512),

    -- Timestamp (immutable after creation)
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiment_contracts_name
    ON experiment_contracts (experiment_name);

CREATE INDEX IF NOT EXISTS idx_experiment_contracts_hash
    ON experiment_contracts (config_hash);

CREATE INDEX IF NOT EXISTS idx_experiment_contracts_created
    ON experiment_contracts (created_at DESC);

CREATE INDEX IF NOT EXISTS idx_experiment_contracts_feature_hash
    ON experiment_contracts (feature_order_hash);

-- Unique constraint on config_hash to prevent duplicate configs
CREATE UNIQUE INDEX IF NOT EXISTS idx_experiment_contracts_config_unique
    ON experiment_contracts (config_hash);

-- Comments
COMMENT ON TABLE experiment_contracts IS
    'Immutable contracts generated from experiment YAML files - provides complete lineage tracking';

COMMENT ON COLUMN experiment_contracts.contract_id IS
    'Unique contract ID in format CTR-EXP-{experiment_name}';

COMMENT ON COLUMN experiment_contracts.config_hash IS
    'SHA256 hash of the full YAML content - uniquely identifies the experiment config';

COMMENT ON COLUMN experiment_contracts.feature_order_hash IS
    'Hash of FEATURE_ORDER from feature_contract.py at time of contract creation';

COMMENT ON COLUMN experiment_contracts.frozen_config IS
    'Complete experiment configuration frozen at contract creation time (immutable)';

-- Prevent updates to immutable contracts
CREATE OR REPLACE FUNCTION prevent_experiment_contract_update()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Experiment contracts are immutable and cannot be updated';
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS prevent_experiment_contract_update_trigger ON experiment_contracts;

CREATE TRIGGER prevent_experiment_contract_update_trigger
    BEFORE UPDATE ON experiment_contracts
    FOR EACH ROW
    EXECUTE FUNCTION prevent_experiment_contract_update();
