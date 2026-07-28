-- Migration 079: adversarial integrity remediation for FABRIC v1.
-- This migration is additive: 070..078 remain immutable reviewed history.

ALTER TABLE control.strategy_declaration
    ADD COLUMN IF NOT EXISTS promotion_evidence JSONB NOT NULL DEFAULT '{}'::jsonb;

-- Preserve the best available evidence for declarations created before 079.
UPDATE control.strategy_declaration
SET promotion_evidence = transition_evidence
WHERE research_state = 'CHAMPION'
  AND promotion_evidence = '{}'::jsonb;

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

    IF ROW(
        NEW.strategy_id,
        NEW.strategy_version,
        NEW.spec_fingerprint,
        NEW.surface,
        NEW.declared_at,
        NEW.declared_by,
        NEW.schema_version
    ) IS DISTINCT FROM ROW(
        OLD.strategy_id,
        OLD.strategy_version,
        OLD.spec_fingerprint,
        OLD.surface,
        OLD.declared_at,
        OLD.declared_by,
        OLD.schema_version
    ) THEN
        RAISE EXCEPTION 'strategy identity fields are immutable';
    END IF;
    IF OLD.promotion_evidence <> '{}'::jsonb
       AND NEW.promotion_evidence IS DISTINCT FROM OLD.promotion_evidence THEN
        RAISE EXCEPTION 'promotion evidence is immutable once recorded';
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

    IF OLD.research_state = 'PAPER' AND NEW.research_state = 'CHAMPION' THEN
        IF NEW.capital_tier <> 'CANARY' THEN
            RAISE EXCEPTION 'PAPER -> CHAMPION must enter at CANARY';
        END IF;
        IF jsonb_typeof(NEW.transition_evidence->'dsr') IS DISTINCT FROM 'number'
           OR (NEW.transition_evidence->>'dsr')::NUMERIC <= 0.95
           OR COALESCE(
                (NEW.transition_evidence->>'vote2_approved')::BOOLEAN,
                FALSE
              ) IS NOT TRUE
           OR COALESCE(
                (NEW.transition_evidence->>'oos_positive')::BOOLEAN,
                FALSE
              ) IS NOT TRUE
           OR COALESCE(
                (NEW.transition_evidence->>'novelty_gate_passed')::BOOLEAN,
                FALSE
              ) IS NOT TRUE
           OR COALESCE(NEW.transition_evidence->>'n_trials', '')
                !~ '^[1-9][0-9]*$'
           OR COALESCE(
                NEW.transition_evidence->>'trial_ledger_fingerprint',
                ''
              ) !~ '^sha256:[0-9a-f]{64}$' THEN
            RAISE EXCEPTION
                'CHAMPION requires Vote-2, positive OOS, novelty, DSR > 0.95, '
                'and n_trials committed to the trial ledger';
        END IF;
        NEW.promotion_evidence := NEW.transition_evidence;
    END IF;

    IF NEW.research_state = OLD.research_state
       AND NEW.capital_tier IS DISTINCT FROM OLD.capital_tier THEN
        IF OLD.research_state <> 'CHAMPION' THEN
            RAISE EXCEPTION
                'capital tier cannot change without its declared research transition';
        ELSIF OLD.capital_tier = 'CANARY' AND NEW.capital_tier = 'FULL' THEN
            IF COALESCE(
                    (NEW.transition_evidence->>'canary_minimums_met')::BOOLEAN,
                    FALSE
               ) IS NOT TRUE
               OR COALESCE(
                    (NEW.transition_evidence->>'vote2_approved')::BOOLEAN,
                    FALSE
               ) IS NOT TRUE
               OR COALESCE(
                    NEW.transition_evidence->>'canary_evidence_fingerprint',
                    ''
               ) !~ '^sha256:[0-9a-f]{64}$' THEN
                RAISE EXCEPTION
                    'CANARY -> FULL requires canary evidence and Vote-2';
            END IF;
        ELSIF (
            (OLD.capital_tier = 'FULL' AND NEW.capital_tier = 'REDUCED')
            OR
            (OLD.capital_tier = 'REDUCED' AND NEW.capital_tier = 'FULL')
        ) THEN
            IF COALESCE(
                    (NEW.transition_evidence->>'pnl_rule_triggered')::BOOLEAN,
                    FALSE
               ) IS NOT TRUE
               OR COALESCE(
                    NEW.transition_evidence->>'pnl_rule_fingerprint',
                    ''
               ) !~ '^sha256:[0-9a-f]{64}$' THEN
                RAISE EXCEPTION
                    'FULL <-> REDUCED requires pre-signed PnL evidence';
            END IF;
        ELSE
            RAISE EXCEPTION 'undeclared CHAMPION capital transition % -> %',
                OLD.capital_tier, NEW.capital_tier;
        END IF;
    END IF;

    IF OLD.research_state = 'CHAMPION'
       AND NEW.research_state = 'RETIRING' THEN
        IF NEW.capital_tier <> 'EXIT_ONLY'
           OR COALESCE(
                (NEW.transition_evidence->>'withdrawal_protocol_triggered')::BOOLEAN,
                FALSE
              ) IS NOT TRUE
           OR COALESCE(
                NEW.transition_evidence->>'withdrawal_protocol_fingerprint',
                ''
              ) !~ '^sha256:[0-9a-f]{64}$' THEN
            RAISE EXCEPTION
                'CHAMPION -> RETIRING requires EXIT_ONLY and withdrawal evidence';
        END IF;
    END IF;

    IF NEW.operational_state IS DISTINCT FROM OLD.operational_state THEN
        IF NEW.operational_state = 'QUARANTINED'
           AND NULLIF(
                BTRIM(NEW.transition_evidence->>'quarantine_reason'),
                ''
               ) IS NULL THEN
            RAISE EXCEPTION 'QUARANTINED requires a machine-readable reason';
        ELSIF NEW.operational_state = 'NOMINAL'
           AND COALESCE(
                NEW.transition_evidence->>'recovery_evidence_fingerprint',
                ''
               ) !~ '^sha256:[0-9a-f]{64}$' THEN
            RAISE EXCEPTION 'recovery requires immutable evidence';
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
        CASE WHEN TG_OP = 'UPDATE' THEN OLD.research_state END,
        NEW.research_state,
        CASE WHEN TG_OP = 'UPDATE' THEN OLD.capital_tier END,
        NEW.capital_tier,
        CASE WHEN TG_OP = 'UPDATE' THEN OLD.operational_state END,
        NEW.operational_state,
        NEW.transition_evidence,
        NEW.declared_by
    );
    IF to_regclass('public.audit_log') IS NOT NULL THEN
        EXECUTE
            'INSERT INTO public.audit_log '
            '(action, object_type, object_id, detail) VALUES ($1,$2,$3,$4)'
        USING
            'strategy_transition',
            'control.strategy_declaration',
            NEW.declaration_id::TEXT,
            jsonb_build_object(
                'strategy_id', NEW.strategy_id,
                'strategy_version', NEW.strategy_version,
                'from_research_state',
                    CASE WHEN TG_OP = 'UPDATE' THEN OLD.research_state END,
                'to_research_state', NEW.research_state,
                'from_capital_tier',
                    CASE WHEN TG_OP = 'UPDATE' THEN OLD.capital_tier END,
                'to_capital_tier', NEW.capital_tier,
                'actor', NEW.declared_by,
                'evidence', NEW.transition_evidence
            );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Serialize every derivation before examining prior output. On divergence the
-- attempted artifact is suppressed (RETURN NULL) and the incident can commit.
CREATE OR REPLACE FUNCTION control.detect_nondeterministic_derivation()
RETURNS TRIGGER AS $$
DECLARE
    prior_hash TEXT;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(NEW.derivation_id, 0));
    SELECT semantic_hash INTO prior_hash
    FROM control.artifact_identity
    WHERE derivation_id = NEW.derivation_id
    ORDER BY created_at, artifact_or_event_id
    LIMIT 1;

    IF prior_hash IS NOT NULL AND prior_hash <> NEW.semantic_hash THEN
        INSERT INTO control.incident (
            incident_type, severity, entity_type, entity_id, details
        ) VALUES (
            'NONDETERMINISTIC_DERIVATION',
            'CRITICAL',
            'derivation',
            NEW.derivation_id,
            jsonb_build_object(
                'prior_semantic_hash', prior_hash,
                'observed_semantic_hash', NEW.semantic_hash,
                'artifact_or_event_id', NEW.artifact_or_event_id
            )
        );
        RETURN NULL;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_artifact_identity_determinism
    ON control.artifact_identity;
CREATE TRIGGER trg_artifact_identity_determinism
    BEFORE INSERT ON control.artifact_identity
    FOR EACH ROW EXECUTE FUNCTION control.detect_nondeterministic_derivation();

DROP TRIGGER IF EXISTS trg_artifact_identity_append_only
    ON control.artifact_identity;
CREATE TRIGGER trg_artifact_identity_append_only
    BEFORE UPDATE OR DELETE ON control.artifact_identity
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_artifact_identity_no_truncate
    ON control.artifact_identity;
CREATE TRIGGER trg_artifact_identity_no_truncate
    BEFORE TRUNCATE ON control.artifact_identity
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_forecast_output_append_only
    ON forecast.forecast_output;
CREATE TRIGGER trg_forecast_output_append_only
    BEFORE UPDATE OR DELETE ON forecast.forecast_output
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_forecast_output_no_truncate
    ON forecast.forecast_output;
CREATE TRIGGER trg_forecast_output_no_truncate
    BEFORE TRUNCATE ON forecast.forecast_output
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_forecast_score_append_only
    ON forecast.forecast_score;
CREATE TRIGGER trg_forecast_score_append_only
    BEFORE UPDATE OR DELETE ON forecast.forecast_score
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_forecast_score_no_truncate
    ON forecast.forecast_score;
CREATE TRIGGER trg_forecast_score_no_truncate
    BEFORE TRUNCATE ON forecast.forecast_score
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_model_horizon_result_append_only
    ON forecast.model_horizon_result;
CREATE TRIGGER trg_model_horizon_result_append_only
    BEFORE UPDATE OR DELETE ON forecast.model_horizon_result
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_model_horizon_result_no_truncate
    ON forecast.model_horizon_result;
CREATE TRIGGER trg_model_horizon_result_no_truncate
    BEFORE TRUNCATE ON forecast.model_horizon_result
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_calibration_result_append_only
    ON forecast.calibration_result;
CREATE TRIGGER trg_calibration_result_append_only
    BEFORE UPDATE OR DELETE ON forecast.calibration_result
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_calibration_result_no_truncate
    ON forecast.calibration_result;
CREATE TRIGGER trg_calibration_result_no_truncate
    BEFORE TRUNCATE ON forecast.calibration_result
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

-- The shared RBAC audit table predates the no-TRUNCATE convention. Harden it
-- only when the legacy plan has created it; fabric-v1 remains independently
-- deployable.
DO $$
BEGIN
    IF to_regclass('public.audit_log') IS NOT NULL THEN
        EXECUTE
            'DROP TRIGGER IF EXISTS trg_audit_log_no_truncate '
            'ON public.audit_log';
        EXECUTE
            'CREATE TRIGGER trg_audit_log_no_truncate '
            'BEFORE TRUNCATE ON public.audit_log '
            'FOR EACH STATEMENT '
            'EXECUTE FUNCTION control.block_append_only_mutation()';
    END IF;
END;
$$;

COMMENT ON COLUMN control.strategy_declaration.promotion_evidence IS
    'Immutable Vote-2/DSR/novelty/trial-ledger evidence captured on PAPER->CHAMPION.';
COMMENT ON FUNCTION control.detect_nondeterministic_derivation() IS
    'Serializes derivations, records divergence, and suppresses the conflicting row.';
