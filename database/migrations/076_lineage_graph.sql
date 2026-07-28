-- Migration 076: PostgreSQL lineage graph and typed revisions (BL-24)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS lineage;

CREATE TABLE IF NOT EXISTS lineage.node (
    node_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_type TEXT NOT NULL,
    semantic_hash TEXT NOT NULL CHECK (semantic_hash ~ '^sha256:[0-9a-f]{64}$'),
    bytes_hash TEXT CHECK (bytes_hash IS NULL OR bytes_hash ~ '^sha256:[0-9a-f]{64}$'),
    schema_version TEXT NOT NULL,
    row_count BIGINT CHECK (row_count IS NULL OR row_count >= 0),
    min_event_time TIMESTAMPTZ,
    max_event_time TIMESTAMPTZ,
    quality_status TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('VALID','STALE','INVALIDATED')),
    storage_uri TEXT,
    availability_quality TEXT NOT NULL DEFAULT 'UNKNOWN' CHECK (
        availability_quality IN ('REAL_VINTAGE','RECONSTRUCTED','UNKNOWN')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (node_type, semantic_hash),
    CHECK (
        (min_event_time IS NULL AND max_event_time IS NULL)
        OR (
            min_event_time IS NOT NULL
            AND max_event_time IS NOT NULL
            AND max_event_time >= min_event_time
        )
    )
);

CREATE TABLE IF NOT EXISTS lineage.edge (
    edge_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_node_id UUID NOT NULL REFERENCES lineage.node(node_id),
    target_node_id UUID NOT NULL REFERENCES lineage.node(node_id),
    edge_type TEXT NOT NULL CHECK (
        edge_type IN ('CONSUMED','PRODUCED','DERIVED_FROM','CORRECTED_BY','SUPERSEDES')
    ),
    run_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_node_id, target_node_id, edge_type)
);

CREATE TABLE IF NOT EXISTS lineage.strategy_node (
    strategy_id TEXT NOT NULL,
    node_id UUID NOT NULL REFERENCES lineage.node(node_id),
    role TEXT NOT NULL CHECK (
        role IN ('INPUT','FEATURE','MODEL','SIGNAL','TARGET','EXECUTION','PNL')
    ),
    linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (strategy_id, node_id, role)
);

CREATE TABLE IF NOT EXISTS lineage.revision_event (
    revision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_node_id UUID NOT NULL REFERENCES lineage.node(node_id),
    revised_node_id UUID REFERENCES lineage.node(node_id),
    revision_type TEXT NOT NULL CHECK (
        revision_type IN (
            'LEGITIMATE_RELEASE','PROVIDER_CORRECTION',
            'PIPELINE_ERROR','SCHEMA_REINTERPRETATION'
        )
    ),
    branch TEXT NOT NULL CHECK (branch IN ('as_released','latest_revised')),
    reason TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL,
    actor TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (
        (revision_type = 'LEGITIMATE_RELEASE' AND revised_node_id IS NOT NULL)
        OR revision_type <> 'LEGITIMATE_RELEASE'
    )
);

CREATE OR REPLACE FUNCTION lineage.mark_descendants_stale(p_node_id UUID)
RETURNS INTEGER AS $$
DECLARE
    affected INTEGER;
BEGIN
    WITH RECURSIVE descendants(node_id) AS (
        SELECT target_node_id
        FROM lineage.edge
        WHERE source_node_id = p_node_id
          AND edge_type IN ('CONSUMED','PRODUCED','DERIVED_FROM')
        UNION
        SELECT e.target_node_id
        FROM lineage.edge e
        JOIN descendants d ON e.source_node_id = d.node_id
        WHERE e.edge_type IN ('CONSUMED','PRODUCED','DERIVED_FROM')
    )
    UPDATE lineage.node n
    SET status = 'STALE'
    WHERE n.node_id IN (SELECT node_id FROM descendants)
      AND n.status = 'VALID';
    GET DIAGNOSTICS affected = ROW_COUNT;
    RETURN affected;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION lineage.apply_revision_semantics()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.revision_type IN ('PROVIDER_CORRECTION','PIPELINE_ERROR') THEN
        PERFORM lineage.mark_descendants_stale(NEW.original_node_id);
    ELSIF NEW.revision_type = 'SCHEMA_REINTERPRETATION' THEN
        UPDATE lineage.node SET status = 'INVALIDATED'
        WHERE node_id = NEW.original_node_id AND status <> 'INVALIDATED';
    END IF;
    -- LEGITIMATE_RELEASE intentionally does not invalidate historical vintages.
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_lineage_revision_semantics ON lineage.revision_event;
CREATE TRIGGER trg_lineage_revision_semantics
    AFTER INSERT ON lineage.revision_event
    FOR EACH ROW EXECUTE FUNCTION lineage.apply_revision_semantics();

CREATE INDEX IF NOT EXISTS idx_lineage_edge_source ON lineage.edge(source_node_id);
CREATE INDEX IF NOT EXISTS idx_lineage_edge_target ON lineage.edge(target_node_id);
CREATE INDEX IF NOT EXISTS idx_lineage_node_status ON lineage.node(status, node_type);
