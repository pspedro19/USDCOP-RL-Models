-- Migration 077: portfolio snapshot/allocation/target control plane (BL-26/27/30)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE SCHEMA IF NOT EXISTS portfolio;

CREATE TABLE IF NOT EXISTS portfolio.snapshot (
    snapshot_id UUID PRIMARY KEY,
    cutoff_time TIMESTAMPTZ NOT NULL,
    required_sleeves TEXT[] NOT NULL,
    stale_signals TEXT[] NOT NULL DEFAULT '{}',
    missing_sleeves TEXT[] NOT NULL DEFAULT '{}',
    fallback_applied JSONB NOT NULL DEFAULT '{}'::jsonb,
    max_age_by_sleeve JSONB NOT NULL,
    missing_policy_by_sleeve JSONB NOT NULL,
    semantic_hash TEXT NOT NULL UNIQUE CHECK (semantic_hash ~ '^sha256:[0-9a-f]{64}$'),
    run_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (cardinality(required_sleeves) > 0),
    CHECK (array_position(required_sleeves, NULL) IS NULL),
    CHECK (array_position(stale_signals, NULL) IS NULL),
    CHECK (array_position(missing_sleeves, NULL) IS NULL),
    CHECK (jsonb_typeof(fallback_applied) = 'object'),
    CHECK (jsonb_typeof(max_age_by_sleeve) = 'object'),
    CHECK (jsonb_typeof(missing_policy_by_sleeve) = 'object')
);

-- Keep reruns/upgrades fail-closed as well as fresh installs.  An existing
-- populated table cannot infer the policy for accepted sleeves without
-- rewriting evidence, so require an explicit backfill instead.
ALTER TABLE portfolio.snapshot
    ALTER COLUMN snapshot_id DROP DEFAULT;
ALTER TABLE portfolio.snapshot
    ADD COLUMN IF NOT EXISTS missing_policy_by_sleeve JSONB;
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM portfolio.snapshot
        WHERE missing_policy_by_sleeve IS NULL
    ) THEN
        RAISE EXCEPTION
            'portfolio.snapshot requires explicit missing-policy backfill';
    END IF;
END;
$$;
ALTER TABLE portfolio.snapshot
    ALTER COLUMN missing_policy_by_sleeve SET NOT NULL;

CREATE TABLE IF NOT EXISTS portfolio.snapshot_signal (
    snapshot_id UUID NOT NULL REFERENCES portfolio.snapshot(snapshot_id),
    signal_id TEXT,
    sleeve_id TEXT NOT NULL,
    signal_as_of TIMESTAMPTZ NOT NULL,
    signal_available_at TIMESTAMPTZ NOT NULL,
    signal_valid_until TIMESTAMPTZ NOT NULL,
    resolution TEXT NOT NULL CHECK (
        resolution IN (
            'ACCEPTED','FLAT','KEEP_POSITION_UNTIL_EXPIRY',
            'EXIT_ONLY','USE_LAST_VALID_WITH_MAX_AGE'
        )
    ),
    materialized_payload JSONB NOT NULL,
    PRIMARY KEY (snapshot_id, sleeve_id),
    CHECK (signal_available_at >= signal_as_of),
    CHECK (signal_valid_until >= signal_as_of),
    CHECK (jsonb_typeof(materialized_payload) = 'object'),
    CHECK (
        (
            resolution IN ('ACCEPTED','USE_LAST_VALID_WITH_MAX_AGE')
            AND signal_id IS NOT NULL AND btrim(signal_id) <> ''
        )
        OR (
            resolution IN ('FLAT','KEEP_POSITION_UNTIL_EXPIRY','EXIT_ONLY')
            AND signal_id IS NULL
        )
    )
);

CREATE OR REPLACE FUNCTION portfolio.validate_snapshot_contract()
RETURNS TRIGGER AS $$
DECLARE
    v_distinct_sleeves INTEGER;
    v_policy_keys TEXT[];
    v_max_age RECORD;
BEGIN
    SELECT COUNT(DISTINCT sleeve), ARRAY_AGG(DISTINCT sleeve ORDER BY sleeve)
    INTO v_distinct_sleeves, v_policy_keys
    FROM unnest(NEW.required_sleeves) AS sleeve;
    IF v_distinct_sleeves <> cardinality(NEW.required_sleeves) THEN
        RAISE EXCEPTION 'required_sleeves must be unique';
    END IF;
    IF v_policy_keys <> ARRAY(
        SELECT key FROM jsonb_object_keys(NEW.max_age_by_sleeve) AS key
        ORDER BY key
    ) THEN
        RAISE EXCEPTION
            'max_age_by_sleeve keys must exactly match required_sleeves';
    END IF;
    IF v_policy_keys <> ARRAY(
        SELECT key FROM jsonb_object_keys(NEW.missing_policy_by_sleeve) AS key
        ORDER BY key
    ) THEN
        RAISE EXCEPTION
            'missing_policy_by_sleeve keys must exactly match required_sleeves';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM jsonb_each_text(NEW.missing_policy_by_sleeve) AS policy
        WHERE policy.value NOT IN (
            'FLAT', 'KEEP_POSITION_UNTIL_EXPIRY',
            'EXIT_ONLY', 'USE_LAST_VALID_WITH_MAX_AGE'
        )
    ) THEN
        RAISE EXCEPTION 'missing_policy_by_sleeve contains an invalid policy';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM jsonb_each_text(NEW.fallback_applied) AS fallback
        WHERE NEW.missing_policy_by_sleeve ->> fallback.key
              IS DISTINCT FROM fallback.value
    ) THEN
        RAISE EXCEPTION
            'fallback_applied must match missing_policy_by_sleeve';
    END IF;
    IF NEW.snapshot_id IS DISTINCT FROM
       uuid_generate_v5(uuid_ns_url(), NEW.semantic_hash) THEN
        RAISE EXCEPTION
            'snapshot_id must be UUIDv5(namespace_url, semantic_hash)';
    END IF;
    FOR v_max_age IN
        SELECT key, value
        FROM jsonb_each_text(NEW.max_age_by_sleeve)
    LOOP
        BEGIN
            IF v_max_age.value::NUMERIC <= 0
               OR v_max_age.value::NUMERIC IN (
                   'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
               ) THEN
                RAISE EXCEPTION
                    'max_age_by_sleeve.% must be finite positive seconds',
                    v_max_age.key;
            END IF;
        EXCEPTION WHEN invalid_text_representation THEN
            RAISE EXCEPTION
                'max_age_by_sleeve.% must be numeric seconds', v_max_age.key;
        END;
    END LOOP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_snapshot_contract ON portfolio.snapshot;
CREATE TRIGGER trg_snapshot_contract
    BEFORE INSERT ON portfolio.snapshot
    FOR EACH ROW EXECUTE FUNCTION portfolio.validate_snapshot_contract();

CREATE OR REPLACE FUNCTION portfolio.enforce_signal_cutoff()
RETURNS TRIGGER AS $$
DECLARE
    snapshot_cutoff TIMESTAMPTZ;
    snapshot_required_sleeves TEXT[];
    snapshot_fallbacks JSONB;
    snapshot_max_ages JSONB;
    snapshot_missing_policies JSONB;
    expected_fallback TEXT;
BEGIN
    SELECT
        cutoff_time,
        required_sleeves,
        fallback_applied,
        max_age_by_sleeve,
        missing_policy_by_sleeve
    INTO
        snapshot_cutoff,
        snapshot_required_sleeves,
        snapshot_fallbacks,
        snapshot_max_ages,
        snapshot_missing_policies
    FROM portfolio.snapshot
    WHERE snapshot_id = NEW.snapshot_id;
    IF snapshot_cutoff IS NULL THEN
        RAISE EXCEPTION 'unknown portfolio snapshot %', NEW.snapshot_id;
    END IF;
    IF NOT NEW.sleeve_id = ANY(snapshot_required_sleeves) THEN
        RAISE EXCEPTION
            'snapshot sleeve % was not declared as required', NEW.sleeve_id;
    END IF;
    IF NEW.signal_as_of > snapshot_cutoff
       OR NEW.signal_available_at > snapshot_cutoff THEN
        RAISE EXCEPTION 'signal available_at % exceeds snapshot cutoff %',
            NEW.signal_available_at, snapshot_cutoff;
    END IF;
    IF NEW.resolution IN ('ACCEPTED', 'USE_LAST_VALID_WITH_MAX_AGE')
       AND NEW.signal_valid_until < snapshot_cutoff THEN
        RAISE EXCEPTION
            'signal valid_until % precedes snapshot cutoff %',
            NEW.signal_valid_until, snapshot_cutoff;
    END IF;
    IF NEW.resolution IN ('ACCEPTED', 'USE_LAST_VALID_WITH_MAX_AGE')
       AND EXTRACT(EPOCH FROM snapshot_cutoff - NEW.signal_as_of)
           > (snapshot_max_ages ->> NEW.sleeve_id)::NUMERIC THEN
        RAISE EXCEPTION
            'signal as_of exceeds max_age for sleeve %', NEW.sleeve_id;
    END IF;
    expected_fallback := snapshot_fallbacks ->> NEW.sleeve_id;
    IF NEW.resolution = 'ACCEPTED' AND expected_fallback IS NOT NULL THEN
        RAISE EXCEPTION 'accepted sleeve cannot declare a fallback';
    ELSIF NEW.resolution <> 'ACCEPTED'
          AND expected_fallback IS DISTINCT FROM NEW.resolution THEN
        RAISE EXCEPTION
            'materialized fallback does not match snapshot fallback policy';
    END IF;
    IF NEW.resolution <> 'ACCEPTED'
       AND snapshot_missing_policies ->> NEW.sleeve_id
           IS DISTINCT FROM NEW.resolution THEN
        RAISE EXCEPTION
            'materialized fallback does not match declared missing policy';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_snapshot_signal_cutoff ON portfolio.snapshot_signal;
CREATE TRIGGER trg_snapshot_signal_cutoff
    BEFORE INSERT ON portfolio.snapshot_signal
    FOR EACH ROW EXECUTE FUNCTION portfolio.enforce_signal_cutoff();

CREATE TABLE IF NOT EXISTS portfolio.allocation (
    allocation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    snapshot_id UUID NOT NULL REFERENCES portfolio.snapshot(snapshot_id),
    allocator_version TEXT NOT NULL,
    sleeve_id TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    volatility_forecast NUMERIC NOT NULL CHECK (
        volatility_forecast > 0
        AND volatility_forecast NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    target_vol NUMERIC NOT NULL CHECK (
        target_vol > 0
        AND target_vol NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    base_risk_budget NUMERIC NOT NULL,
    multiplier_forward NUMERIC NOT NULL CHECK (multiplier_forward BETWEEN 0 AND 1),
    multiplier_liquidity NUMERIC NOT NULL CHECK (multiplier_liquidity BETWEEN 0 AND 1),
    multiplier_diversification NUMERIC NOT NULL CHECK (
        multiplier_diversification BETWEEN 0.70 AND 1.10
    ),
    multiplier_operations NUMERIC NOT NULL CHECK (multiplier_operations IN (0,1)),
    multiplier_drawdown NUMERIC NOT NULL CHECK (multiplier_drawdown BETWEEN 0 AND 1),
    novelty_max_correlation NUMERIC NOT NULL CHECK (
        novelty_max_correlation BETWEEN -1 AND 1
    ),
    novelty_delta_information_ratio NUMERIC NOT NULL CHECK (
        novelty_delta_information_ratio NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    novelty_correlation_threshold NUMERIC NOT NULL CHECK (
        novelty_correlation_threshold BETWEEN -1 AND 1
    ),
    novelty_delta_ir_threshold NUMERIC NOT NULL CHECK (
        novelty_delta_ir_threshold NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    novelty_gate_passed BOOLEAN NOT NULL,
    final_risk_budget NUMERIC NOT NULL CHECK (final_risk_budget >= 0),
    signed_weight NUMERIC NOT NULL,
    fallback_level INTEGER NOT NULL CHECK (fallback_level BETWEEN 0 AND 4),
    fallback_incident_id UUID REFERENCES control.incident(incident_id),
    constraints_fingerprint TEXT NOT NULL CHECK (
        constraints_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (snapshot_id, allocator_version, sleeve_id),
    UNIQUE (allocation_id, snapshot_id, sleeve_id),
    CHECK (
        base_risk_budget >= 0
        AND base_risk_budget NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    CHECK (
        final_risk_budget NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    CHECK (
        signed_weight NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    CHECK (
        final_risk_budget =
            base_risk_budget
            * multiplier_forward
            * multiplier_liquidity
            * multiplier_diversification
            * multiplier_operations
            * multiplier_drawdown
    ),
    CHECK (ABS(signed_weight) = final_risk_budget),
    CHECK (
        novelty_gate_passed = (
            novelty_max_correlation < novelty_correlation_threshold
            OR novelty_delta_information_ratio > novelty_delta_ir_threshold
        )
    ),
    CHECK ((fallback_level = 0) = (fallback_incident_id IS NULL))
);

CREATE TABLE IF NOT EXISTS portfolio.target (
    target_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_version TEXT NOT NULL,
    snapshot_id UUID NOT NULL REFERENCES portfolio.snapshot(snapshot_id),
    account_id TEXT NOT NULL,
    environment TEXT NOT NULL CHECK (environment IN ('replay','paper','canary','live')),
    allocator_version TEXT NOT NULL,
    valid_from TIMESTAMPTZ NOT NULL,
    valid_until TIMESTAMPTZ NOT NULL,
    rebalance_cutoff TIMESTAMPTZ NOT NULL,
    decision_fingerprint TEXT NOT NULL CHECK (
        decision_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    ),
    constraints_snapshot JSONB NOT NULL,
    infeasibility_fallback INTEGER CHECK (
        infeasibility_fallback BETWEEN 1 AND 4
    ),
    fallback_incident_id UUID REFERENCES control.incident(incident_id),
    semantic_hash TEXT NOT NULL UNIQUE CHECK (
        semantic_hash ~ '^sha256:[0-9a-f]{64}$'
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (target_id, snapshot_id),
    CHECK (valid_until > valid_from),
    CHECK (rebalance_cutoff <= valid_from),
    CHECK (jsonb_typeof(constraints_snapshot) = 'object'),
    CHECK (btrim(target_version) <> ''),
    CHECK (btrim(account_id) <> ''),
    CHECK (btrim(allocator_version) <> ''),
    CHECK (
        (infeasibility_fallback IS NULL AND fallback_incident_id IS NULL)
        OR (
            infeasibility_fallback IS NOT NULL
            AND fallback_incident_id IS NOT NULL
        )
    )
);

CREATE TABLE IF NOT EXISTS portfolio.target_exposure (
    exposure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID NOT NULL,
    snapshot_id UUID NOT NULL,
    allocation_id UUID NOT NULL,
    strategy_id TEXT NOT NULL,
    sleeve_id TEXT NOT NULL,
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    side TEXT NOT NULL CHECK (side IN ('LONG','SHORT','FLAT')),
    risk_budget NUMERIC NOT NULL CHECK (
        risk_budget >= 0
        AND risk_budget NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    target_weight NUMERIC NOT NULL CHECK (
        target_weight NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    currency TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (exposure_id, target_id),
    UNIQUE (target_id, sleeve_id, instrument_id),
    UNIQUE (target_id, allocation_id, sleeve_id, instrument_id),
    FOREIGN KEY (target_id, snapshot_id)
        REFERENCES portfolio.target(target_id, snapshot_id),
    FOREIGN KEY (allocation_id, snapshot_id, sleeve_id)
        REFERENCES portfolio.allocation(allocation_id, snapshot_id, sleeve_id),
    CHECK (btrim(strategy_id) <> ''),
    CHECK (btrim(sleeve_id) <> ''),
    CHECK (btrim(currency) <> ''),
    CHECK (
        (side = 'LONG' AND target_weight > 0)
        OR (side = 'SHORT' AND target_weight < 0)
        OR (side = 'FLAT' AND target_weight = 0 AND risk_budget = 0)
    ),
    CHECK (ABS(target_weight) = risk_budget)
);

CREATE OR REPLACE FUNCTION portfolio.validate_target_cutoff()
RETURNS TRIGGER AS $$
DECLARE
    v_snapshot_cutoff TIMESTAMPTZ;
    v_snapshot_allocator_version TEXT;
    v_materialized_count INTEGER;
BEGIN
    SELECT cutoff_time INTO v_snapshot_cutoff
    FROM portfolio.snapshot
    WHERE snapshot_id = NEW.snapshot_id;
    IF v_snapshot_cutoff IS NULL OR NEW.rebalance_cutoff <> v_snapshot_cutoff THEN
        RAISE EXCEPTION
            'target rebalance_cutoff must equal its snapshot cutoff_time';
    END IF;
    SELECT COUNT(*) INTO v_materialized_count
    FROM portfolio.snapshot_signal
    WHERE snapshot_id = NEW.snapshot_id;
    IF v_materialized_count <> (
        SELECT cardinality(required_sleeves)
        FROM portfolio.snapshot
        WHERE snapshot_id = NEW.snapshot_id
    ) THEN
        RAISE EXCEPTION
            'target requires one materialized input per required sleeve';
    END IF;
    SELECT MIN(allocator_version)
    INTO v_snapshot_allocator_version
    FROM portfolio.allocation
    WHERE snapshot_id = NEW.snapshot_id;
    IF v_snapshot_allocator_version IS NULL
       OR EXISTS (
           SELECT 1
           FROM portfolio.allocation
           WHERE snapshot_id = NEW.snapshot_id
             AND allocator_version <> NEW.allocator_version
       )
       OR v_snapshot_allocator_version <> NEW.allocator_version THEN
        RAISE EXCEPTION
            'target allocator_version must match every snapshot allocation';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_target_cutoff ON portfolio.target;
CREATE TRIGGER trg_target_cutoff
    BEFORE INSERT ON portfolio.target
    FOR EACH ROW EXECUTE FUNCTION portfolio.validate_target_cutoff();

CREATE OR REPLACE FUNCTION portfolio.validate_target_exposure()
RETURNS TRIGGER AS $$
DECLARE
    v_allocation portfolio.allocation%ROWTYPE;
    v_instrument_asset_id TEXT;
BEGIN
    SELECT * INTO v_allocation
    FROM portfolio.allocation
    WHERE allocation_id = NEW.allocation_id
    FOR SHARE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown allocation %', NEW.allocation_id;
    END IF;
    IF NEW.risk_budget <> v_allocation.final_risk_budget
       OR NEW.target_weight <> v_allocation.signed_weight THEN
        RAISE EXCEPTION
            'target exposure must equal its immutable allocation';
    END IF;
    SELECT asset_id INTO v_instrument_asset_id
    FROM reference.instrument
    WHERE instrument_id = NEW.instrument_id;
    IF v_instrument_asset_id IS DISTINCT FROM v_allocation.asset_id THEN
        RAISE EXCEPTION
            'target exposure instrument does not match allocation asset';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_target_exposure_validate
    ON portfolio.target_exposure;
CREATE TRIGGER trg_target_exposure_validate
    BEFORE INSERT ON portfolio.target_exposure
    FOR EACH ROW EXECUTE FUNCTION portfolio.validate_target_exposure();

CREATE OR REPLACE FUNCTION portfolio.assert_target_complete()
RETURNS TRIGGER AS $$
DECLARE
    v_target_id UUID;
    v_required TEXT[];
    v_actual TEXT[];
BEGIN
    v_target_id := CASE
        WHEN TG_TABLE_NAME = 'target' THEN NEW.target_id
        ELSE NEW.target_id
    END;
    SELECT ARRAY(
        SELECT sleeve
        FROM unnest(s.required_sleeves) AS sleeve
        ORDER BY sleeve
    )
    INTO v_required
    FROM portfolio.target t
    JOIN portfolio.snapshot s ON s.snapshot_id = t.snapshot_id
    WHERE t.target_id = v_target_id;

    SELECT ARRAY_AGG(e.sleeve_id ORDER BY e.sleeve_id)
    INTO v_actual
    FROM portfolio.target_exposure e
    WHERE e.target_id = v_target_id;
    IF v_actual IS DISTINCT FROM v_required THEN
        RAISE EXCEPTION
            'portfolio target exposures must exactly cover required sleeves';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_target_complete ON portfolio.target;
CREATE CONSTRAINT TRIGGER trg_target_complete
    AFTER INSERT ON portfolio.target
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION portfolio.assert_target_complete();
DROP TRIGGER IF EXISTS trg_target_exposure_complete
    ON portfolio.target_exposure;
CREATE CONSTRAINT TRIGGER trg_target_exposure_complete
    AFTER INSERT ON portfolio.target_exposure
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION portfolio.assert_target_complete();

CREATE TABLE IF NOT EXISTS portfolio.pretrade_decision (
    decision_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    target_id UUID NOT NULL REFERENCES portfolio.target(target_id),
    exposure_id UUID NOT NULL REFERENCES portfolio.target_exposure(exposure_id),
    account_id TEXT NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL,
    allowed BOOLEAN NOT NULL,
    checks JSONB NOT NULL,
    reason_codes TEXT[] NOT NULL DEFAULT '{}',
    health_snapshot_id TEXT NOT NULL,
    reconciliation_id UUID,
    kill_switch_level TEXT,
    evaluator_version TEXT NOT NULL,
    FOREIGN KEY (exposure_id, target_id)
        REFERENCES portfolio.target_exposure(exposure_id, target_id),
    CHECK (jsonb_typeof(checks) = 'object'),
    CHECK (
        (allowed AND cardinality(reason_codes) = 0)
        OR (NOT allowed AND cardinality(reason_codes) > 0)
    )
);

CREATE TABLE IF NOT EXISTS portfolio.kill_switch_event (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT,
    level TEXT NOT NULL CHECK (level IN ('CLEAR','BLOCK_NEW','CANCEL_OPEN','EXIT_ALL','ACCOUNT_FREEZE')),
    actor TEXT NOT NULL,
    reason TEXT NOT NULL,
    source_reconciliation_id UUID UNIQUE,
    supersedes_event_id UUID UNIQUE REFERENCES portfolio.kill_switch_event(event_id),
    effective_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (expires_at IS NULL OR expires_at > effective_at),
    CHECK (level <> 'CLEAR' OR supersedes_event_id IS NOT NULL)
);

CREATE OR REPLACE FUNCTION portfolio.validate_kill_switch_supersession()
RETURNS TRIGGER AS $$
DECLARE
    prior_account_id TEXT;
BEGIN
    IF NEW.supersedes_event_id IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT account_id INTO prior_account_id
    FROM portfolio.kill_switch_event
    WHERE event_id = NEW.supersedes_event_id;
    IF NOT FOUND OR prior_account_id IS DISTINCT FROM NEW.account_id THEN
        RAISE EXCEPTION
            'kill-switch supersession must reference an event for the same account scope';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_kill_switch_supersession
    ON portfolio.kill_switch_event;
CREATE TRIGGER trg_kill_switch_supersession
    BEFORE INSERT ON portfolio.kill_switch_event
    FOR EACH ROW EXECUTE FUNCTION portfolio.validate_kill_switch_supersession();

CREATE TABLE IF NOT EXISTS portfolio.kill_switch_action (
    action_key TEXT PRIMARY KEY,
    kill_switch_event_id UUID NOT NULL REFERENCES portfolio.kill_switch_event(event_id),
    account_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (action IN ('cancel_open','exit_all')),
    status TEXT NOT NULL DEFAULT 'CLAIMED' CHECK (
        status IN ('CLAIMED','COMPLETED','FAILED','RECONCILIATION_REQUIRED')
    ),
    claim_token UUID NOT NULL DEFAULT gen_random_uuid(),
    attempt_count INTEGER NOT NULL DEFAULT 1 CHECK (attempt_count > 0),
    claimed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    lease_expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '30 seconds'),
    completed_at TIMESTAMPTZ,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (lease_expires_at > claimed_at),
    CHECK ((status = 'COMPLETED') = (completed_at IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS portfolio.kill_switch_action_event (
    action_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    action_key TEXT NOT NULL REFERENCES portfolio.kill_switch_action(action_key),
    claim_token UUID NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN ('CLAIMED','COMPLETED','FAILED','RECONCILIATION_REQUIRED')
    ),
    occurred_at TIMESTAMPTZ NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE OR REPLACE FUNCTION portfolio.claim_kill_switch_action(
    p_action_key TEXT,
    p_kill_switch_event_id UUID,
    p_account_id TEXT,
    p_action TEXT,
    p_claimed_at TIMESTAMPTZ,
    p_lease_seconds INTEGER DEFAULT 30
)
RETURNS UUID AS $$
DECLARE
    v_token UUID := gen_random_uuid();
    v_row portfolio.kill_switch_action%ROWTYPE;
BEGIN
    IF p_lease_seconds < 1 OR p_lease_seconds > 300 THEN
        RAISE EXCEPTION 'kill-switch action lease must be between 1 and 300 seconds';
    END IF;

    INSERT INTO portfolio.kill_switch_action (
        action_key,
        kill_switch_event_id,
        account_id,
        action,
        status,
        claim_token,
        attempt_count,
        claimed_at,
        lease_expires_at
    )
    VALUES (
        p_action_key,
        p_kill_switch_event_id,
        p_account_id,
        p_action,
        'CLAIMED',
        v_token,
        1,
        p_claimed_at,
        p_claimed_at + make_interval(secs => p_lease_seconds)
    )
    ON CONFLICT (action_key) DO NOTHING
    RETURNING * INTO v_row;

    IF FOUND THEN
        INSERT INTO portfolio.kill_switch_action_event (
            action_key, claim_token, status, occurred_at
        )
        VALUES (p_action_key, v_token, 'CLAIMED', p_claimed_at);
        RETURN v_token;
    END IF;

    SELECT * INTO v_row
    FROM portfolio.kill_switch_action
    WHERE action_key = p_action_key
    FOR UPDATE;

    IF v_row.kill_switch_event_id <> p_kill_switch_event_id
       OR v_row.account_id <> p_account_id
       OR v_row.action <> p_action THEN
        RAISE EXCEPTION
            'kill-switch action-key collision with different economic identity';
    END IF;

    IF v_row.status = 'COMPLETED'
       OR (
            v_row.status = 'CLAIMED'
            AND v_row.lease_expires_at > p_claimed_at
       ) THEN
        RETURN NULL;
    END IF;

    UPDATE portfolio.kill_switch_action
    SET
        status = 'CLAIMED',
        claim_token = v_token,
        attempt_count = attempt_count + 1,
        claimed_at = p_claimed_at,
        lease_expires_at = p_claimed_at + make_interval(secs => p_lease_seconds),
        completed_at = NULL
    WHERE action_key = p_action_key;

    INSERT INTO portfolio.kill_switch_action_event (
        action_key, claim_token, status, occurred_at,
        details
    )
    VALUES (
        p_action_key,
        v_token,
        'CLAIMED',
        p_claimed_at,
        jsonb_build_object('attempt_count', v_row.attempt_count + 1)
    );
    RETURN v_token;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION portfolio.complete_kill_switch_action(
    p_action_key TEXT,
    p_claim_token UUID,
    p_completed_at TIMESTAMPTZ,
    p_details JSONB DEFAULT '{}'::jsonb
)
RETURNS BOOLEAN AS $$
DECLARE
    v_updated TEXT;
BEGIN
    UPDATE portfolio.kill_switch_action
    SET
        status = 'COMPLETED',
        completed_at = p_completed_at,
        details = details || COALESCE(p_details, '{}'::jsonb)
    WHERE action_key = p_action_key
      AND claim_token = p_claim_token
      AND status = 'CLAIMED'
    RETURNING action_key INTO v_updated;

    IF v_updated IS NULL THEN
        RETURN FALSE;
    END IF;
    INSERT INTO portfolio.kill_switch_action_event (
        action_key, claim_token, status, occurred_at, details
    )
    VALUES (
        p_action_key, p_claim_token, 'COMPLETED', p_completed_at,
        COALESCE(p_details, '{}'::jsonb)
    );
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION portfolio.require_kill_switch_reconciliation(
    p_action_key TEXT,
    p_claim_token UUID,
    p_occurred_at TIMESTAMPTZ,
    p_reason_code TEXT,
    p_details JSONB DEFAULT '{}'::jsonb
)
RETURNS BOOLEAN AS $$
DECLARE
    v_updated TEXT;
    v_details JSONB;
BEGIN
    v_details := COALESCE(p_details, '{}'::jsonb)
        || jsonb_build_object('reason_code', p_reason_code);
    UPDATE portfolio.kill_switch_action
    SET
        status = 'RECONCILIATION_REQUIRED',
        completed_at = NULL,
        details = details || v_details
    WHERE action_key = p_action_key
      AND claim_token = p_claim_token
      AND status = 'CLAIMED'
    RETURNING action_key INTO v_updated;

    IF v_updated IS NULL THEN
        RETURN FALSE;
    END IF;
    INSERT INTO portfolio.kill_switch_action_event (
        action_key, claim_token, status, occurred_at, details
    )
    VALUES (
        p_action_key, p_claim_token, 'RECONCILIATION_REQUIRED',
        p_occurred_at, v_details
    );
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

REVOKE ALL ON FUNCTION portfolio.claim_kill_switch_action(
    TEXT, UUID, TEXT, TEXT, TIMESTAMPTZ, INTEGER
) FROM PUBLIC;
REVOKE ALL ON FUNCTION portfolio.complete_kill_switch_action(
    TEXT, UUID, TIMESTAMPTZ, JSONB
) FROM PUBLIC;
REVOKE ALL ON FUNCTION portfolio.require_kill_switch_reconciliation(
    TEXT, UUID, TIMESTAMPTZ, TEXT, JSONB
) FROM PUBLIC;

DROP TRIGGER IF EXISTS trg_snapshot_immutable ON portfolio.snapshot;
CREATE TRIGGER trg_snapshot_immutable
    BEFORE UPDATE OR DELETE ON portfolio.snapshot
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_snapshot_no_truncate ON portfolio.snapshot;
CREATE TRIGGER trg_snapshot_no_truncate
    BEFORE TRUNCATE ON portfolio.snapshot
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_snapshot_signal_immutable ON portfolio.snapshot_signal;
CREATE TRIGGER trg_snapshot_signal_immutable
    BEFORE UPDATE OR DELETE ON portfolio.snapshot_signal
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_snapshot_signal_no_truncate ON portfolio.snapshot_signal;
CREATE TRIGGER trg_snapshot_signal_no_truncate
    BEFORE TRUNCATE ON portfolio.snapshot_signal
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_allocation_immutable ON portfolio.allocation;
CREATE TRIGGER trg_allocation_immutable
    BEFORE UPDATE OR DELETE ON portfolio.allocation
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_allocation_no_truncate ON portfolio.allocation;
CREATE TRIGGER trg_allocation_no_truncate
    BEFORE TRUNCATE ON portfolio.allocation
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_target_immutable ON portfolio.target;
CREATE TRIGGER trg_target_immutable
    BEFORE UPDATE OR DELETE ON portfolio.target
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_target_no_truncate ON portfolio.target;
CREATE TRIGGER trg_target_no_truncate
    BEFORE TRUNCATE ON portfolio.target
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_target_exposure_immutable
    ON portfolio.target_exposure;
CREATE TRIGGER trg_target_exposure_immutable
    BEFORE UPDATE OR DELETE ON portfolio.target_exposure
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_target_exposure_no_truncate
    ON portfolio.target_exposure;
CREATE TRIGGER trg_target_exposure_no_truncate
    BEFORE TRUNCATE ON portfolio.target_exposure
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_pretrade_decision_immutable
    ON portfolio.pretrade_decision;
CREATE TRIGGER trg_pretrade_decision_immutable
    BEFORE UPDATE OR DELETE ON portfolio.pretrade_decision
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_pretrade_decision_no_truncate
    ON portfolio.pretrade_decision;
CREATE TRIGGER trg_pretrade_decision_no_truncate
    BEFORE TRUNCATE ON portfolio.pretrade_decision
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

CREATE OR REPLACE FUNCTION portfolio.audit_kill_switch_event()
RETURNS TRIGGER AS $$
BEGIN
    IF to_regclass('public.audit_log') IS NOT NULL THEN
        EXECUTE
            'INSERT INTO public.audit_log '
            '(action, object_type, object_id, detail) VALUES ($1,$2,$3,$4)'
        USING
            CASE WHEN NEW.account_id IS NULL THEN 'kill_global' ELSE 'kill_user' END,
            'portfolio.kill_switch_event',
            NEW.event_id::TEXT,
            jsonb_build_object(
                'account_id', NEW.account_id,
                'level', NEW.level,
                'actor', NEW.actor,
                'reason', NEW.reason,
                'effective_at', NEW.effective_at
            );
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_kill_switch_audit ON portfolio.kill_switch_event;
CREATE TRIGGER trg_kill_switch_audit
    AFTER INSERT ON portfolio.kill_switch_event
    FOR EACH ROW EXECUTE FUNCTION portfolio.audit_kill_switch_event();

DROP TRIGGER IF EXISTS trg_kill_switch_immutable ON portfolio.kill_switch_event;
CREATE TRIGGER trg_kill_switch_immutable
    BEFORE UPDATE OR DELETE ON portfolio.kill_switch_event
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_kill_switch_no_truncate ON portfolio.kill_switch_event;
CREATE TRIGGER trg_kill_switch_no_truncate
    BEFORE TRUNCATE ON portfolio.kill_switch_event
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();

DROP TRIGGER IF EXISTS trg_kill_switch_action_event_immutable
    ON portfolio.kill_switch_action_event;
CREATE TRIGGER trg_kill_switch_action_event_immutable
    BEFORE UPDATE OR DELETE ON portfolio.kill_switch_action_event
    FOR EACH ROW EXECUTE FUNCTION control.block_append_only_mutation();
DROP TRIGGER IF EXISTS trg_kill_switch_action_event_no_truncate
    ON portfolio.kill_switch_action_event;
CREATE TRIGGER trg_kill_switch_action_event_no_truncate
    BEFORE TRUNCATE ON portfolio.kill_switch_action_event
    FOR EACH STATEMENT EXECUTE FUNCTION control.block_append_only_mutation();
