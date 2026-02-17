-- Migration 036: Enhanced Model Registry
-- Adds lineage tracking and approval info to model_registry
-- Created: 2026-01-31

-- Check if model_registry exists, create if not
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(255) UNIQUE NOT NULL,
    experiment_name VARCHAR(255) NOT NULL,
    model_path VARCHAR(512) NOT NULL,
    model_hash VARCHAR(64),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add new columns if they don't exist
DO $$
BEGIN
    -- norm_stats columns
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'norm_stats_path') THEN
        ALTER TABLE model_registry ADD COLUMN norm_stats_path VARCHAR(512);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'norm_stats_hash') THEN
        ALTER TABLE model_registry ADD COLUMN norm_stats_hash VARCHAR(64);
    END IF;

    -- Lineage hashes
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'config_hash') THEN
        ALTER TABLE model_registry ADD COLUMN config_hash VARCHAR(64);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'feature_order_hash') THEN
        ALTER TABLE model_registry ADD COLUMN feature_order_hash VARCHAR(64);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'dataset_hash') THEN
        ALTER TABLE model_registry ADD COLUMN dataset_hash VARCHAR(64);
    END IF;

    -- Stage management
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'stage') THEN
        ALTER TABLE model_registry ADD COLUMN stage VARCHAR(20) DEFAULT 'staging';
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'is_active') THEN
        ALTER TABLE model_registry ADD COLUMN is_active BOOLEAN DEFAULT FALSE;
    END IF;

    -- Metrics and lineage JSON
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'metrics') THEN
        ALTER TABLE model_registry ADD COLUMN metrics JSONB;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'lineage') THEN
        ALTER TABLE model_registry ADD COLUMN lineage JSONB;
    END IF;

    -- L4 proposal reference
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'l4_proposal_id') THEN
        ALTER TABLE model_registry ADD COLUMN l4_proposal_id VARCHAR(255);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'l4_recommendation') THEN
        ALTER TABLE model_registry ADD COLUMN l4_recommendation VARCHAR(20);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'l4_confidence') THEN
        ALTER TABLE model_registry ADD COLUMN l4_confidence DECIMAL(5,4);
    END IF;

    -- Approval info (segundo voto)
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'approved_by') THEN
        ALTER TABLE model_registry ADD COLUMN approved_by VARCHAR(255);
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'approved_at') THEN
        ALTER TABLE model_registry ADD COLUMN approved_at TIMESTAMPTZ;
    END IF;

    -- Lifecycle timestamps
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'promoted_at') THEN
        ALTER TABLE model_registry ADD COLUMN promoted_at TIMESTAMPTZ;
    END IF;

    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name = 'model_registry' AND column_name = 'archived_at') THEN
        ALTER TABLE model_registry ADD COLUMN archived_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add constraint for stage values
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'model_registry_stage_check') THEN
        ALTER TABLE model_registry ADD CONSTRAINT model_registry_stage_check
            CHECK (stage IN ('staging', 'production', 'archived', 'canary'));
    END IF;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_model_registry_stage
    ON model_registry (stage);

CREATE INDEX IF NOT EXISTS idx_model_registry_experiment
    ON model_registry (experiment_name);

CREATE INDEX IF NOT EXISTS idx_model_registry_active
    ON model_registry (is_active)
    WHERE is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_model_registry_production
    ON model_registry (stage, is_active)
    WHERE stage = 'production' AND is_active = TRUE;

CREATE INDEX IF NOT EXISTS idx_model_registry_config_hash
    ON model_registry (config_hash);

CREATE INDEX IF NOT EXISTS idx_model_registry_promoted
    ON model_registry (promoted_at DESC);

-- Comments
COMMENT ON COLUMN model_registry.stage IS
    'Model lifecycle stage: staging (after L3), production (after 2 votes), archived, canary';

COMMENT ON COLUMN model_registry.feature_order_hash IS
    'Hash of FEATURE_ORDER used during training - must match L1/L5 for inference';

COMMENT ON COLUMN model_registry.norm_stats_hash IS
    'Hash of norm_stats.json - must match for correct normalization';

COMMENT ON COLUMN model_registry.approved_by IS
    'Email of human approver (segundo voto)';
