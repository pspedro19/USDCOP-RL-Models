-- =============================================================================
-- Migration: 025_lineage_tables.sql
-- Description: Create unified lineage tracking tables
-- Principle: MLflow-First + DVC-Tracked
-- Author: Trading Team
-- Date: 2026-01-22
-- =============================================================================

-- Create ML schema if not exists
CREATE SCHEMA IF NOT EXISTS ml;

-- =============================================================================
-- Lineage Nodes Table
-- =============================================================================
-- Stores individual nodes in the lineage graph (datasets, models, training runs)

CREATE TABLE IF NOT EXISTS ml.lineage_nodes (
    id VARCHAR(255) PRIMARY KEY,
    node_type VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50),

    -- Source references (MLflow-First)
    mlflow_run_id VARCHAR(50),
    mlflow_experiment_id VARCHAR(100),
    mlflow_model_uri TEXT,

    -- DVC references (DVC-Tracked)
    dvc_tag VARCHAR(100),
    dvc_remote VARCHAR(255),

    -- Database references
    db_record_id INTEGER,
    db_table VARCHAR(100),

    -- Hashes for reproducibility
    content_hash VARCHAR(64),
    config_hash VARCHAR(64),
    feature_order_hash VARCHAR(64),

    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for lineage_nodes
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_type ON ml.lineage_nodes(node_type);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_name ON ml.lineage_nodes(name);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_mlflow ON ml.lineage_nodes(mlflow_run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_dvc ON ml.lineage_nodes(dvc_tag);
CREATE INDEX IF NOT EXISTS idx_lineage_nodes_created ON ml.lineage_nodes(created_at DESC);

-- =============================================================================
-- Lineage Edges Table
-- =============================================================================
-- Stores relationships between lineage nodes

CREATE TABLE IF NOT EXISTS ml.lineage_edges (
    id SERIAL PRIMARY KEY,
    source_id VARCHAR(255) NOT NULL REFERENCES ml.lineage_nodes(id) ON DELETE CASCADE,
    target_id VARCHAR(255) NOT NULL REFERENCES ml.lineage_nodes(id) ON DELETE CASCADE,
    relation VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate edges
    UNIQUE(source_id, target_id, relation)
);

-- Indexes for lineage_edges
CREATE INDEX IF NOT EXISTS idx_lineage_edges_source ON ml.lineage_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_lineage_edges_target ON ml.lineage_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_lineage_edges_relation ON ml.lineage_edges(relation);

-- =============================================================================
-- Complete Lineage Records Table
-- =============================================================================
-- Stores complete lineage records for quick querying

CREATE TABLE IF NOT EXISTS ml.lineage_records (
    record_id VARCHAR(255) PRIMARY KEY,
    pipeline VARCHAR(50) NOT NULL,  -- 'rl' or 'forecasting'

    -- Dataset lineage (DVC-Tracked)
    dataset_path TEXT,
    dataset_hash VARCHAR(64),
    dataset_dvc_tag VARCHAR(100),
    dataset_version VARCHAR(50),

    -- Feature lineage
    feature_config_hash VARCHAR(64),
    feature_order_hash VARCHAR(64),
    num_features INTEGER,

    -- Training lineage (MLflow-First)
    mlflow_experiment_id VARCHAR(100),
    mlflow_run_id VARCHAR(50),
    training_config_hash VARCHAR(64),
    training_duration_seconds FLOAT,

    -- Model lineage (MLflow-First)
    model_name VARCHAR(255),
    model_version VARCHAR(50),
    model_stage VARCHAR(50) DEFAULT 'None',
    model_uri TEXT,
    model_hash VARCHAR(64),

    -- Validation lineage
    backtest_id VARCHAR(100),
    validation_metrics JSONB DEFAULT '{}',

    -- Inference tracking
    inference_count INTEGER DEFAULT 0,
    last_inference_at TIMESTAMPTZ,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(100),

    -- Policy compliance
    mlflow_first_compliant BOOLEAN DEFAULT FALSE,
    dvc_tracked_compliant BOOLEAN DEFAULT FALSE
);

-- Indexes for lineage_records
CREATE INDEX IF NOT EXISTS idx_lineage_records_model ON ml.lineage_records(model_name, model_version);
CREATE INDEX IF NOT EXISTS idx_lineage_records_pipeline ON ml.lineage_records(pipeline);
CREATE INDEX IF NOT EXISTS idx_lineage_records_mlflow ON ml.lineage_records(mlflow_run_id);
CREATE INDEX IF NOT EXISTS idx_lineage_records_created ON ml.lineage_records(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_lineage_records_stage ON ml.lineage_records(model_stage);

-- =============================================================================
-- Model Promotion Audit Table
-- =============================================================================
-- Tracks all model promotions for audit trail

CREATE TABLE IF NOT EXISTS ml.model_promotion_audit (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(255) NOT NULL,
    from_version VARCHAR(50),
    to_version VARCHAR(50) NOT NULL,
    from_stage VARCHAR(50),
    to_stage VARCHAR(50) NOT NULL,
    reason TEXT,

    -- Validation results
    validation_passed BOOLEAN NOT NULL,
    validation_errors INTEGER DEFAULT 0,
    validation_warnings INTEGER DEFAULT 0,
    validation_details JSONB DEFAULT '{}',

    -- Lineage reference
    lineage_record_id VARCHAR(255) REFERENCES ml.lineage_records(record_id),

    -- Audit
    promoted_at TIMESTAMPTZ DEFAULT NOW(),
    promoted_by VARCHAR(100),
    approved_by VARCHAR(100),

    -- Policy compliance
    mlflow_first_compliant BOOLEAN DEFAULT FALSE,
    dvc_tracked_compliant BOOLEAN DEFAULT FALSE
);

-- Indexes for promotion audit
CREATE INDEX IF NOT EXISTS idx_promotion_audit_model ON ml.model_promotion_audit(model_name);
CREATE INDEX IF NOT EXISTS idx_promotion_audit_stage ON ml.model_promotion_audit(to_stage);
CREATE INDEX IF NOT EXISTS idx_promotion_audit_date ON ml.model_promotion_audit(promoted_at DESC);

-- =============================================================================
-- Artifact Storage Audit Table
-- =============================================================================
-- Tracks artifact storage for policy compliance

CREATE TABLE IF NOT EXISTS ml.artifact_storage_audit (
    id SERIAL PRIMARY KEY,
    artifact_type VARCHAR(50) NOT NULL,
    artifact_id VARCHAR(255) NOT NULL,

    -- Storage locations
    primary_backend VARCHAR(50) NOT NULL,
    primary_uri TEXT NOT NULL,
    secondary_backend VARCHAR(50),
    secondary_uri TEXT,

    -- Policy compliance
    policy_compliant BOOLEAN DEFAULT TRUE,
    violations JSONB DEFAULT '[]',

    -- Hashes
    content_hash VARCHAR(64),

    -- Audit
    stored_at TIMESTAMPTZ DEFAULT NOW(),
    stored_by VARCHAR(100),

    UNIQUE(artifact_type, artifact_id)
);

-- Indexes for artifact audit
CREATE INDEX IF NOT EXISTS idx_artifact_audit_type ON ml.artifact_storage_audit(artifact_type);
CREATE INDEX IF NOT EXISTS idx_artifact_audit_backend ON ml.artifact_storage_audit(primary_backend);
CREATE INDEX IF NOT EXISTS idx_artifact_audit_compliant ON ml.artifact_storage_audit(policy_compliant);

-- =============================================================================
-- Views
-- =============================================================================

-- View: Complete lineage for models
CREATE OR REPLACE VIEW ml.v_model_lineage AS
SELECT
    lr.record_id,
    lr.pipeline,
    lr.model_name,
    lr.model_version,
    lr.model_stage,
    lr.model_uri,
    lr.dataset_dvc_tag,
    lr.dataset_hash,
    lr.mlflow_run_id,
    lr.mlflow_experiment_id,
    lr.feature_config_hash,
    lr.num_features,
    lr.validation_metrics,
    lr.mlflow_first_compliant,
    lr.dvc_tracked_compliant,
    lr.created_at,
    -- Latest promotion
    pa.to_stage AS latest_promotion_stage,
    pa.promoted_at AS latest_promotion_at,
    pa.validation_passed AS latest_promotion_passed
FROM ml.lineage_records lr
LEFT JOIN LATERAL (
    SELECT * FROM ml.model_promotion_audit pa
    WHERE pa.model_name = lr.model_name
    ORDER BY pa.promoted_at DESC
    LIMIT 1
) pa ON TRUE
ORDER BY lr.created_at DESC;

-- View: Policy compliance summary
CREATE OR REPLACE VIEW ml.v_policy_compliance AS
SELECT
    pipeline,
    COUNT(*) AS total_records,
    SUM(CASE WHEN mlflow_first_compliant THEN 1 ELSE 0 END) AS mlflow_compliant_count,
    SUM(CASE WHEN dvc_tracked_compliant THEN 1 ELSE 0 END) AS dvc_compliant_count,
    SUM(CASE WHEN mlflow_first_compliant AND dvc_tracked_compliant THEN 1 ELSE 0 END) AS fully_compliant_count,
    ROUND(
        100.0 * SUM(CASE WHEN mlflow_first_compliant AND dvc_tracked_compliant THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
        2
    ) AS compliance_pct
FROM ml.lineage_records
GROUP BY pipeline;

-- =============================================================================
-- Functions
-- =============================================================================

-- Function: Update compliance status
CREATE OR REPLACE FUNCTION ml.update_lineage_compliance()
RETURNS TRIGGER AS $$
BEGIN
    -- Check MLflow-First compliance
    NEW.mlflow_first_compliant := (
        NEW.mlflow_run_id IS NOT NULL
        AND NEW.model_uri IS NOT NULL
        AND (NEW.model_uri LIKE 'models:/%' OR NEW.model_uri LIKE 'runs:/%')
    );

    -- Check DVC-Tracked compliance
    NEW.dvc_tracked_compliant := (
        NEW.dataset_dvc_tag IS NOT NULL
        AND NEW.dataset_hash IS NOT NULL
    );

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger: Auto-update compliance on insert/update
DROP TRIGGER IF EXISTS trg_lineage_compliance ON ml.lineage_records;
CREATE TRIGGER trg_lineage_compliance
    BEFORE INSERT OR UPDATE ON ml.lineage_records
    FOR EACH ROW
    EXECUTE FUNCTION ml.update_lineage_compliance();

-- =============================================================================
-- Comments
-- =============================================================================

COMMENT ON TABLE ml.lineage_nodes IS 'Individual nodes in the ML lineage graph';
COMMENT ON TABLE ml.lineage_edges IS 'Relationships between lineage nodes';
COMMENT ON TABLE ml.lineage_records IS 'Complete lineage records for quick querying';
COMMENT ON TABLE ml.model_promotion_audit IS 'Audit trail for model stage promotions';
COMMENT ON TABLE ml.artifact_storage_audit IS 'Audit trail for artifact storage policy compliance';

COMMENT ON COLUMN ml.lineage_records.mlflow_first_compliant IS 'True if model follows MLflow-First principle';
COMMENT ON COLUMN ml.lineage_records.dvc_tracked_compliant IS 'True if dataset follows DVC-Tracked principle';

-- =============================================================================
-- Grant permissions
-- =============================================================================

GRANT USAGE ON SCHEMA ml TO PUBLIC;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA ml TO PUBLIC;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA ml TO PUBLIC;
