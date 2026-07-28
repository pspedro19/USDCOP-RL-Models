-- Migration 070: FABRIC constitutional control plane (BL-16/17/18)
-- Additive and idempotent. No runtime cutover is performed by this migration.

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS control;

CREATE TABLE IF NOT EXISTS control.strategy_declaration (
    declaration_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    research_state TEXT NOT NULL CHECK (
        research_state IN (
            'DECLARED','SCREENED','DESIGN_RUN','FROZEN',
            'PAPER','CHAMPION','RETIRING','WITHDRAWN'
        )
    ),
    capital_tier TEXT NOT NULL CHECK (
        capital_tier IN ('ZERO','SHADOW','CANARY','FULL','REDUCED','EXIT_ONLY')
    ),
    operational_state TEXT NOT NULL CHECK (
        operational_state IN ('NOMINAL','QUARANTINED')
    ),
    exit_checklist TEXT,
    dag_declared BOOLEAN NOT NULL DEFAULT FALSE,
    surface TEXT NOT NULL CHECK (surface IN ('action','diagnostic')),
    spec_fingerprint TEXT NOT NULL CHECK (spec_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    declared_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    declared_by TEXT NOT NULL,
    transition_evidence JSONB NOT NULL DEFAULT '{}'::jsonb,
    schema_version TEXT NOT NULL DEFAULT '1.0.0',
    UNIQUE (strategy_id, strategy_version),
    CHECK (
        (research_state IN ('DECLARED','SCREENED','DESIGN_RUN','FROZEN') AND capital_tier = 'ZERO')
        OR (research_state = 'PAPER' AND capital_tier IN ('ZERO','SHADOW'))
        OR (research_state = 'CHAMPION' AND capital_tier IN ('ZERO','SHADOW','CANARY','FULL','REDUCED'))
        OR (research_state = 'RETIRING' AND capital_tier = 'EXIT_ONLY')
        OR (research_state = 'WITHDRAWN' AND capital_tier = 'ZERO' AND exit_checklist = 'PASS')
    ),
    CHECK (
        dag_declared = (research_state IN ('FROZEN','PAPER','CHAMPION','RETIRING'))
    ),
    CHECK (
        surface = 'action'
        OR (capital_tier = 'ZERO' AND research_state NOT IN ('CHAMPION','RETIRING'))
    )
);

CREATE TABLE IF NOT EXISTS control.strategy_declaration_event (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    declaration_id UUID NOT NULL REFERENCES control.strategy_declaration(declaration_id),
    strategy_id TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    from_research_state TEXT,
    to_research_state TEXT NOT NULL,
    from_capital_tier TEXT,
    to_capital_tier TEXT NOT NULL,
    from_operational_state TEXT,
    to_operational_state TEXT NOT NULL,
    evidence JSONB NOT NULL,
    actor TEXT NOT NULL,
    event_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS control.strategy_sleeve (
    strategy_id TEXT NOT NULL,
    sleeve_id TEXT NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ,
    mapping_reason TEXT NOT NULL,
    declared_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (strategy_id, sleeve_id, valid_from),
    CHECK (valid_until IS NULL OR valid_until > valid_from)
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_active_sleeve_strategy
    ON control.strategy_sleeve (sleeve_id)
    WHERE valid_until IS NULL;

CREATE OR REPLACE FUNCTION control.enforce_strategy_declaration_transition()
RETURNS TRIGGER AS $$
DECLARE
    allowed BOOLEAN;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.research_state <> 'DECLARED' OR NEW.capital_tier <> 'ZERO' THEN
            RAISE EXCEPTION 'new strategy declarations must start DECLARED/ZERO';
        END IF;
        RETURN NEW;
    END IF;

    allowed := (
        NEW.research_state = OLD.research_state
        OR (OLD.research_state = 'DECLARED' AND NEW.research_state = 'SCREENED')
        OR (OLD.research_state = 'SCREENED' AND NEW.research_state = 'DESIGN_RUN')
        OR (OLD.research_state = 'DESIGN_RUN' AND NEW.research_state = 'FROZEN')
        OR (OLD.research_state = 'FROZEN' AND NEW.research_state = 'PAPER')
        OR (OLD.research_state = 'PAPER' AND NEW.research_state = 'CHAMPION')
        OR (OLD.research_state = 'CHAMPION' AND NEW.research_state = 'RETIRING')
        OR (OLD.research_state = 'RETIRING' AND NEW.research_state = 'WITHDRAWN')
    );
    IF NOT allowed THEN
        RAISE EXCEPTION 'illegal research transition % -> %',
            OLD.research_state, NEW.research_state;
    END IF;
    IF NEW.research_state = 'CHAMPION' AND OLD.research_state <> 'CHAMPION' THEN
        IF COALESCE((NEW.transition_evidence->>'vote2_approved')::BOOLEAN, FALSE) IS NOT TRUE
           OR COALESCE((NEW.transition_evidence->>'oos_positive')::BOOLEAN, FALSE) IS NOT TRUE
           OR COALESCE((NEW.transition_evidence->>'dsr')::NUMERIC, 0) <= 0.95 THEN
            RAISE EXCEPTION
                'CHAMPION requires Vote-2, positive OOS, and trial-aware DSR > 0.95';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION control.audit_strategy_declaration_transition()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO control.strategy_declaration_event (
        declaration_id, strategy_id, strategy_version,
        from_research_state, to_research_state,
        from_capital_tier, to_capital_tier,
        from_operational_state, to_operational_state,
        evidence, actor
    ) VALUES (
        NEW.declaration_id, NEW.strategy_id, NEW.strategy_version,
        CASE WHEN TG_OP = 'UPDATE' THEN OLD.research_state END, NEW.research_state,
        CASE WHEN TG_OP = 'UPDATE' THEN OLD.capital_tier END, NEW.capital_tier,
        CASE WHEN TG_OP = 'UPDATE' THEN OLD.operational_state END, NEW.operational_state,
        NEW.transition_evidence, NEW.declared_by
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_strategy_declaration_transition
    ON control.strategy_declaration;
CREATE TRIGGER trg_strategy_declaration_transition
    BEFORE INSERT OR UPDATE ON control.strategy_declaration
    FOR EACH ROW EXECUTE FUNCTION control.enforce_strategy_declaration_transition();
DROP TRIGGER IF EXISTS trg_strategy_declaration_audit
    ON control.strategy_declaration;
CREATE TRIGGER trg_strategy_declaration_audit
    AFTER INSERT OR UPDATE ON control.strategy_declaration
    FOR EACH ROW EXECUTE FUNCTION control.audit_strategy_declaration_transition();

CREATE TABLE IF NOT EXISTS control.incident (
    incident_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    incident_type TEXT NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('INFO','WARNING','CRITICAL')),
    entity_type TEXT,
    entity_id TEXT,
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL DEFAULT 'OPEN' CHECK (status IN ('OPEN','ACKNOWLEDGED','RESOLVED')),
    resolved_at TIMESTAMPTZ,
    resolution TEXT
);

CREATE TABLE IF NOT EXISTS control.artifact_identity (
    artifact_or_event_id TEXT PRIMARY KEY,
    artifact_type TEXT NOT NULL,
    as_of TIMESTAMPTZ NOT NULL,
    available_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    run_id TEXT NOT NULL,
    spec_fingerprint TEXT NOT NULL CHECK (spec_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    decision_fingerprint TEXT CHECK (
        decision_fingerprint IS NULL OR decision_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    ),
    execution_fingerprint TEXT CHECK (
        execution_fingerprint IS NULL OR execution_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    ),
    derivation_id TEXT NOT NULL CHECK (derivation_id ~ '^sha256:[0-9a-f]{64}$'),
    semantic_hash TEXT NOT NULL CHECK (semantic_hash ~ '^sha256:[0-9a-f]{64}$'),
    bytes_hash TEXT CHECK (bytes_hash IS NULL OR bytes_hash ~ '^sha256:[0-9a-f]{64}$'),
    sleeve_id TEXT,
    forecast_spec_id TEXT,
    family_id TEXT,
    trial_id TEXT,
    code_hash TEXT NOT NULL,
    data_snapshot_id TEXT NOT NULL,
    env TEXT NOT NULL CHECK (env IN ('replay','paper','canary','live','research')),
    schema_version TEXT NOT NULL,
    storage_uri TEXT,
    canonical_writer BOOLEAN NOT NULL DEFAULT FALSE,
    CHECK (available_at >= as_of),
    CHECK (NOT canonical_writer OR bytes_hash = semantic_hash)
);

CREATE INDEX IF NOT EXISTS idx_artifact_identity_derivation
    ON control.artifact_identity (derivation_id);
CREATE INDEX IF NOT EXISTS idx_artifact_identity_decision
    ON control.artifact_identity (decision_fingerprint)
    WHERE decision_fingerprint IS NOT NULL;

CREATE OR REPLACE FUNCTION control.detect_nondeterministic_derivation()
RETURNS TRIGGER AS $$
DECLARE
    prior_hash TEXT;
BEGIN
    SELECT semantic_hash INTO prior_hash
    FROM control.artifact_identity
    WHERE derivation_id = NEW.derivation_id
      AND artifact_or_event_id <> NEW.artifact_or_event_id
    ORDER BY created_at
    LIMIT 1;

    IF prior_hash IS NOT NULL AND prior_hash <> NEW.semantic_hash THEN
        INSERT INTO control.incident (
            incident_type, severity, entity_type, entity_id, details
        ) VALUES (
            'NONDETERMINISTIC_DERIVATION', 'CRITICAL', 'derivation',
            NEW.derivation_id,
            jsonb_build_object(
                'prior_semantic_hash', prior_hash,
                'observed_semantic_hash', NEW.semantic_hash,
                'artifact_or_event_id', NEW.artifact_or_event_id
            )
        );
        RAISE EXCEPTION
            'derivation_id % produced divergent semantic hashes', NEW.derivation_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_artifact_identity_determinism
    ON control.artifact_identity;
CREATE TRIGGER trg_artifact_identity_determinism
    BEFORE INSERT OR UPDATE ON control.artifact_identity
    FOR EACH ROW EXECUTE FUNCTION control.detect_nondeterministic_derivation();

CREATE TABLE IF NOT EXISTS control.metric_event (
    metric_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    event_time TIMESTAMPTZ NOT NULL,
    catalog_version TEXT NOT NULL,
    formula_version TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    strategy_id TEXT,
    asset_id TEXT,
    run_id TEXT,
    environment TEXT CHECK (
        environment IS NULL
        OR environment IN ('backtest','held_out','paper','canary','live')
    ),
    metric_namespace TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DOUBLE PRECISION,
    metric_unit TEXT,
    status TEXT NOT NULL CHECK (
        status IN ('OK','WARNING','CRITICAL','N_A','INSUFFICIENT_SAMPLE')
    ),
    threshold_warning DOUBLE PRECISION,
    threshold_critical DOUBLE PRECISION,
    dimensions JSONB NOT NULL DEFAULT '{}'::jsonb,
    lineage JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        metric_value IS NULL
        OR metric_value NOT IN (
            'NaN'::DOUBLE PRECISION,
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION
        )
    ),
    CHECK (
        threshold_warning IS NULL
        OR threshold_warning NOT IN (
            'NaN'::DOUBLE PRECISION,
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION
        )
    ),
    CHECK (
        threshold_critical IS NULL
        OR threshold_critical NOT IN (
            'NaN'::DOUBLE PRECISION,
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION
        )
    )
);

CREATE TABLE IF NOT EXISTS control.legacy_metric_observation (
    observation_id UUID PRIMARY KEY,
    observed_at TIMESTAMPTZ NOT NULL,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    environment TEXT NOT NULL CHECK (environment = 'backtest'),
    legacy_metric_name TEXT NOT NULL,
    observed_value DOUBLE PRECISION NOT NULL CHECK (
        observed_value NOT IN (
            'NaN'::DOUBLE PRECISION,
            'Infinity'::DOUBLE PRECISION,
            '-Infinity'::DOUBLE PRECISION
        )
    ),
    source_uri TEXT NOT NULL,
    verified_by_metric_engine BOOLEAN NOT NULL DEFAULT FALSE CHECK (
        NOT verified_by_metric_engine
    ),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metric_event_entity_time
    ON control.metric_event (entity_type, entity_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_metric_event_metric_time
    ON control.metric_event (metric_namespace, metric_name, event_time DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_metric_event_semantic_identity
    ON control.metric_event (
        event_time,
        catalog_version,
        formula_version,
        entity_type,
        entity_id,
        COALESCE(run_id, ''),
        COALESCE(environment, ''),
        metric_namespace,
        metric_name,
        dimensions
    );

CREATE OR REPLACE FUNCTION control.block_append_only_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_metric_event_append_only ON control.metric_event;
CREATE TRIGGER trg_metric_event_append_only
    BEFORE UPDATE OR DELETE ON control.metric_event
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_metric_event_no_truncate ON control.metric_event;
CREATE TRIGGER trg_metric_event_no_truncate
    BEFORE TRUNCATE ON control.metric_event
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_legacy_metric_observation_immutable
    ON control.legacy_metric_observation;
CREATE TRIGGER trg_legacy_metric_observation_immutable
    BEFORE UPDATE OR DELETE ON control.legacy_metric_observation
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_legacy_metric_observation_no_truncate
    ON control.legacy_metric_observation;
CREATE TRIGGER trg_legacy_metric_observation_no_truncate
    BEFORE TRUNCATE ON control.legacy_metric_observation
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_strategy_declaration_event_immutable
    ON control.strategy_declaration_event;
CREATE TRIGGER trg_strategy_declaration_event_immutable
    BEFORE UPDATE OR DELETE ON control.strategy_declaration_event
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_strategy_declaration_event_no_truncate
    ON control.strategy_declaration_event;
CREATE TRIGGER trg_strategy_declaration_event_no_truncate
    BEFORE TRUNCATE ON control.strategy_declaration_event
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

COMMENT ON TABLE control.strategy_declaration IS
    'BL-16: 96 nominal state combinations reduced to the 26 constitutional combinations.';
COMMENT ON TABLE control.artifact_identity IS
    'BL-17: resolvable operational spine and canonical hash parity.';
COMMENT ON TABLE control.metric_event IS
    'BL-18: immutable governed metric results with thresholds copied at evaluation time.';
