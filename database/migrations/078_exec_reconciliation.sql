-- Migration 078: independent reconciliation and execution-service grants (BL-30)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS exec;

CREATE TABLE IF NOT EXISTS exec.reconciliation_event (
    reconciliation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL,
    instrument_id UUID REFERENCES reference.instrument(instrument_id),
    scope TEXT NOT NULL CHECK (scope IN ('ACCOUNT','INSTRUMENT')),
    phase TEXT NOT NULL CHECK (phase IN ('PRE_OPERATION','INTRADAY','EOD')),
    observed_at TIMESTAMPTZ NOT NULL,
    internal_qty NUMERIC,
    broker_qty NUMERIC,
    internal_cash NUMERIC,
    broker_cash NUMERIC,
    currency TEXT NOT NULL CHECK (currency ~ '^[A-Z]{3}$'),
    status TEXT NOT NULL CHECK (status IN ('RECONCILED','MISMATCH','QUARANTINED')),
    discrepancies JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_snapshot_uri TEXT NOT NULL CHECK (btrim(source_snapshot_uri) <> ''),
    reconciliation_fingerprint TEXT NOT NULL UNIQUE CHECK (
        reconciliation_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    ),
    actor TEXT NOT NULL DEFAULT 'execution_service',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (
            scope = 'ACCOUNT'
            AND instrument_id IS NULL
            AND internal_qty IS NULL
            AND broker_qty IS NULL
            AND internal_cash IS NOT NULL AND broker_cash IS NOT NULL
        )
        OR (
            scope = 'INSTRUMENT'
            AND instrument_id IS NOT NULL
            AND internal_qty IS NOT NULL AND broker_qty IS NOT NULL
            AND internal_cash IS NULL
            AND broker_cash IS NULL
        )
    ),
    CHECK (jsonb_typeof(discrepancies) = 'object'),
    CHECK (btrim(account_id) <> ''),
    CHECK (btrim(actor) <> ''),
    CHECK (
        (internal_qty IS NULL OR internal_qty NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        ))
        AND (broker_qty IS NULL OR broker_qty NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        ))
        AND (internal_cash IS NULL OR internal_cash NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        ))
        AND (broker_cash IS NULL OR broker_cash NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        ))
    )
);

CREATE INDEX IF NOT EXISTS idx_exec_reconciliation_account_time
    ON exec.reconciliation_event (account_id, observed_at DESC);

CREATE OR REPLACE FUNCTION exec.validate_reconciliation_event()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.observed_at > NOW() + INTERVAL '5 minutes' THEN
        RAISE EXCEPTION 'reconciliation observed_at is in the future';
    END IF;

    IF NEW.scope = 'INSTRUMENT'
       AND (NEW.internal_qty IS NULL OR NEW.broker_qty IS NULL) THEN
        RAISE EXCEPTION
            'instrument reconciliation requires internal and broker quantities';
    END IF;
    IF NEW.scope = 'ACCOUNT'
       AND (NEW.internal_cash IS NULL OR NEW.broker_cash IS NULL) THEN
        RAISE EXCEPTION
            'account reconciliation requires internal and broker cash balances';
    END IF;

    IF NEW.status = 'RECONCILED' THEN
        IF NEW.discrepancies <> '{}'::jsonb THEN
            RAISE EXCEPTION 'reconciled status cannot carry discrepancies';
        END IF;
        IF NEW.scope = 'INSTRUMENT'
           AND NEW.internal_qty IS DISTINCT FROM NEW.broker_qty THEN
            RAISE EXCEPTION 'reconciled quantities disagree';
        END IF;
        IF NEW.scope = 'ACCOUNT'
           AND NEW.internal_cash IS DISTINCT FROM NEW.broker_cash THEN
            RAISE EXCEPTION 'reconciled cash balances disagree';
        END IF;
    ELSIF NEW.status = 'MISMATCH'
          AND NEW.discrepancies = '{}'::jsonb THEN
        RAISE EXCEPTION 'mismatch status requires discrepancies';
    ELSIF NEW.status = 'QUARANTINED'
          AND NEW.discrepancies = '{}'::jsonb THEN
        RAISE EXCEPTION 'quarantined status requires discrepancies';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_exec_reconciliation_validate
    ON exec.reconciliation_event;
CREATE TRIGGER trg_exec_reconciliation_validate
    BEFORE INSERT ON exec.reconciliation_event
    FOR EACH ROW EXECUTE FUNCTION exec.validate_reconciliation_event();

CREATE OR REPLACE FUNCTION exec.quarantine_reconciliation_mismatch()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('MISMATCH', 'QUARANTINED') THEN
        INSERT INTO portfolio.kill_switch_event (
            account_id, level, actor, reason, source_reconciliation_id, effective_at
        ) VALUES (
            NEW.account_id, 'BLOCK_NEW', 'reconciliation_guard',
            'broker reconciliation discrepancy: ' || NEW.reconciliation_id::text,
            NEW.reconciliation_id, NOW()
        )
        ON CONFLICT (source_reconciliation_id) DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_exec_reconciliation_guard ON exec.reconciliation_event;
CREATE TRIGGER trg_exec_reconciliation_guard
    AFTER INSERT ON exec.reconciliation_event
    FOR EACH ROW EXECUTE FUNCTION exec.quarantine_reconciliation_mismatch();

DROP TRIGGER IF EXISTS trg_exec_reconciliation_immutable ON exec.reconciliation_event;
CREATE TRIGGER trg_exec_reconciliation_immutable
    BEFORE UPDATE OR DELETE ON exec.reconciliation_event
    FOR EACH ROW EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_exec_reconciliation_no_truncate
    ON exec.reconciliation_event;
CREATE TRIGGER trg_exec_reconciliation_no_truncate
    BEFORE TRUNCATE ON exec.reconciliation_event
    FOR EACH STATEMENT EXECUTE FUNCTION exec.block_event_mutation();

ALTER TABLE exec.order_header
    DROP CONSTRAINT IF EXISTS fk_exec_order_portfolio_target;
ALTER TABLE exec.order_header
    ADD CONSTRAINT fk_exec_order_portfolio_target
    FOREIGN KEY (target_id) REFERENCES portfolio.target(target_id) ON DELETE RESTRICT;
ALTER TABLE exec.order_header
    DROP CONSTRAINT IF EXISTS fk_exec_order_portfolio_exposure;
ALTER TABLE exec.order_header
    ADD CONSTRAINT fk_exec_order_portfolio_exposure
    FOREIGN KEY (target_id, allocation_id, sleeve_id, instrument_id)
    REFERENCES portfolio.target_exposure(
        target_id, allocation_id, sleeve_id, instrument_id
    ) ON DELETE RESTRICT;

CREATE OR REPLACE VIEW exec.v_active_kill_switch_event AS
SELECT
    event_id,
    account_id,
    level,
    CASE level
        WHEN 'CLEAR' THEN 0
        WHEN 'BLOCK_NEW' THEN 1
        WHEN 'CANCEL_OPEN' THEN 2
        WHEN 'EXIT_ALL' THEN 3
        WHEN 'ACCOUNT_FREEZE' THEN 4
    END AS severity_rank,
    actor,
    reason,
    effective_at,
    expires_at,
    created_at
FROM portfolio.kill_switch_event
WHERE effective_at <= NOW()
  AND (expires_at IS NULL OR expires_at > NOW())
  AND NOT EXISTS (
      SELECT 1
      FROM portfolio.kill_switch_event successor
      WHERE successor.supersedes_event_id = portfolio.kill_switch_event.event_id
        AND successor.effective_at <= NOW()
        AND (successor.expires_at IS NULL OR successor.expires_at > NOW())
  );

CREATE OR REPLACE FUNCTION exec.effective_kill_switch(
    p_account_id TEXT,
    p_at TIMESTAMPTZ DEFAULT NOW()
) RETURNS TABLE (
    event_id UUID,
    account_id TEXT,
    level TEXT,
    actor TEXT,
    reason TEXT,
    effective_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ
) AS $$
    SELECT
        e.event_id,
        e.account_id,
        e.level,
        e.actor,
        e.reason,
        e.effective_at,
        e.expires_at
    FROM portfolio.kill_switch_event e
    WHERE (e.account_id = p_account_id OR e.account_id IS NULL)
      AND e.effective_at <= p_at
      AND (e.expires_at IS NULL OR e.expires_at > p_at)
      AND NOT EXISTS (
          SELECT 1
          FROM portfolio.kill_switch_event successor
          WHERE successor.supersedes_event_id = e.event_id
            AND successor.effective_at <= p_at
            AND (successor.expires_at IS NULL OR successor.expires_at > p_at)
      )
    ORDER BY
        CASE e.level
            WHEN 'CLEAR' THEN 0
            WHEN 'BLOCK_NEW' THEN 1
            WHEN 'CANCEL_OPEN' THEN 2
            WHEN 'EXIT_ALL' THEN 3
            WHEN 'ACCOUNT_FREEZE' THEN 4
        END DESC,
        e.effective_at DESC,
        e.created_at DESC
    LIMIT 1;
$$ LANGUAGE SQL STABLE;

COMMENT ON TABLE exec.reconciliation_event IS
    'BL-30 pre-operation, intraday and EOD reconciliation; mismatch blocks openings.';
COMMENT ON FUNCTION exec.effective_kill_switch(TEXT, TIMESTAMPTZ) IS
    'Severity-first account+global kill-switch resolver; weaker events never downgrade stronger controls.';
