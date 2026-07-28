-- Migration 074: common event-sourced execution ledger (BL-21)

CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE SCHEMA IF NOT EXISTS exec;

CREATE TABLE IF NOT EXISTS exec.order_header (
    order_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    client_order_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE CHECK (idempotency_key ~ '^sha256:[0-9a-f]{64}$'),
    account_id TEXT NOT NULL,
    env TEXT NOT NULL CHECK (env IN ('replay','paper','canary','live')),
    executor_type TEXT NOT NULL CHECK (executor_type IN ('deterministic_simulator','broker')),
    broker_id TEXT NOT NULL,
    strategy_id TEXT NOT NULL,
    sleeve_id TEXT NOT NULL,
    target_id UUID NOT NULL,
    target_version TEXT NOT NULL,
    allocation_id UUID NOT NULL,
    instrument_id UUID NOT NULL REFERENCES reference.instrument(instrument_id),
    instrument TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
    qty NUMERIC NOT NULL CHECK (
        qty > 0
        AND qty NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)
    ),
    order_type TEXT NOT NULL,
    limit_price NUMERIC CHECK (
        limit_price IS NULL
        OR (
            limit_price > 0
            AND limit_price NOT IN (
                'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
            )
        )
    ),
    tif TEXT,
    currency TEXT NOT NULL,
    parent_order_id UUID REFERENCES exec.order_header(order_id),
    decision_fingerprint TEXT NOT NULL CHECK (decision_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    execution_fingerprint TEXT NOT NULL CHECK (execution_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    rebalance_cutoff TIMESTAMPTZ NOT NULL,
    submitted_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (env IN ('replay','paper') AND executor_type = 'deterministic_simulator')
        OR (env IN ('canary','live') AND executor_type = 'broker')
    ),
    CHECK (btrim(account_id) <> ''),
    CHECK (btrim(broker_id) <> ''),
    CHECK (btrim(instrument) <> ''),
    CHECK (btrim(currency) <> '')
);

CREATE TABLE IF NOT EXISTS exec.order_status_event (
    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES exec.order_header(order_id),
    event_time TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'CREATED','VALIDATED','SUBMITTED','ACKNOWLEDGED','PARTIALLY_FILLED',
            'FILLED','CANCEL_PENDING','CANCELLED','REJECTED','EXPIRED','QUARANTINED',
            'SUBMIT_UNKNOWN','SIMULATED'
        )
    ),
    reason_code TEXT,
    broker_order_id TEXT,
    actor TEXT NOT NULL,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    CHECK (jsonb_typeof(details) = 'object')
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_order_status_event_time
    ON exec.order_status_event (order_id, event_time);

CREATE OR REPLACE FUNCTION exec.validate_order_status_transition()
RETURNS TRIGGER AS $$
DECLARE
    previous_status TEXT;
    previous_time TIMESTAMPTZ;
    header_submitted_at TIMESTAMPTZ;
BEGIN
    -- The header row is the per-order serialization fence.  Without this lock,
    -- two concurrent writers can both validate against the same prior state.
    SELECT submitted_at
      INTO header_submitted_at
      FROM exec.order_header
     WHERE order_id = NEW.order_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown order_id %', NEW.order_id;
    END IF;

    IF NEW.event_time < header_submitted_at THEN
        RAISE EXCEPTION 'order status event predates order submission';
    END IF;

    SELECT status, event_time
      INTO previous_status, previous_time
      FROM exec.order_status_event
     WHERE order_id = NEW.order_id
     ORDER BY event_time DESC, event_id DESC
     LIMIT 1;

    IF previous_status IS NULL THEN
        IF NEW.status NOT IN (
            'CREATED','VALIDATED','SUBMITTED','REJECTED',
            'SUBMIT_UNKNOWN','SIMULATED','QUARANTINED'
        ) THEN
            RAISE EXCEPTION 'invalid order status transition INITIAL -> %', NEW.status;
        END IF;
        RETURN NEW;
    END IF;

    IF NEW.event_time <= previous_time THEN
        RAISE EXCEPTION 'order status event_time must increase monotonically';
    END IF;

    IF NOT (
        (previous_status = 'CREATED' AND NEW.status IN (
            'VALIDATED','SUBMITTED','REJECTED','SUBMIT_UNKNOWN',
            'SIMULATED','QUARANTINED'
        ))
        OR (previous_status = 'VALIDATED' AND NEW.status IN (
            'SUBMITTED','REJECTED','SUBMIT_UNKNOWN','SIMULATED','QUARANTINED'
        ))
        OR (previous_status = 'SUBMITTED' AND NEW.status IN (
            'ACKNOWLEDGED','PARTIALLY_FILLED','FILLED','CANCEL_PENDING',
            'CANCELLED','REJECTED','EXPIRED','SUBMIT_UNKNOWN','QUARANTINED'
        ))
        OR (previous_status = 'ACKNOWLEDGED' AND NEW.status IN (
            'PARTIALLY_FILLED','FILLED','CANCEL_PENDING','CANCELLED',
            'REJECTED','EXPIRED','QUARANTINED'
        ))
        OR (previous_status = 'PARTIALLY_FILLED' AND NEW.status IN (
            'PARTIALLY_FILLED','FILLED','CANCEL_PENDING',
            'CANCELLED','EXPIRED','QUARANTINED'
        ))
        OR (previous_status = 'CANCEL_PENDING' AND NEW.status IN (
            'PARTIALLY_FILLED','FILLED','CANCELLED','REJECTED',
            'EXPIRED','QUARANTINED'
        ))
        OR (previous_status = 'SUBMIT_UNKNOWN' AND NEW.status IN (
            'SUBMITTED','ACKNOWLEDGED','PARTIALLY_FILLED','FILLED',
            'CANCEL_PENDING','CANCELLED','REJECTED','EXPIRED','QUARANTINED',
            'SUBMIT_UNKNOWN'
        ))
    ) THEN
        RAISE EXCEPTION 'invalid order status transition % -> %',
            previous_status, NEW.status;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_order_status_transition ON exec.order_status_event;
CREATE TRIGGER trg_order_status_transition
    BEFORE INSERT ON exec.order_status_event
    FOR EACH ROW EXECUTE FUNCTION exec.validate_order_status_transition();

CREATE TABLE IF NOT EXISTS exec.order_dispatch (
    order_id UUID PRIMARY KEY REFERENCES exec.order_header(order_id),
    execution_fingerprint TEXT NOT NULL CHECK (
        execution_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    ),
    status TEXT NOT NULL CHECK (
        status IN ('CLAIMED','RECONCILIATION_REQUIRED','COMPLETED')
    ),
    claim_token UUID NOT NULL,
    attempt_count INTEGER NOT NULL CHECK (attempt_count > 0),
    lease_expires_at TIMESTAMPTZ NOT NULL,
    outcome_status TEXT CHECK (
        outcome_status IS NULL
        OR outcome_status IN ('SUBMITTED','REJECTED','SIMULATED')
    ),
    last_reason_code TEXT,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS exec.order_dispatch_event (
    dispatch_event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES exec.order_header(order_id),
    claim_token UUID NOT NULL,
    attempt_count INTEGER NOT NULL CHECK (attempt_count > 0),
    event_time TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL CHECK (
        event_type IN ('CLAIMED','RECONCILIATION_REQUIRED','COMPLETED')
    ),
    outcome_status TEXT CHECK (
        outcome_status IS NULL
        OR outcome_status IN ('SUBMITTED','REJECTED','SIMULATED')
    ),
    reason_code TEXT,
    broker_order_id TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (order_id, attempt_count, event_type),
    CHECK (jsonb_typeof(details) = 'object')
);

CREATE OR REPLACE FUNCTION exec.claim_order_dispatch(
    p_order_id UUID,
    p_execution_fingerprint TEXT,
    p_claimed_at TIMESTAMPTZ,
    p_lease_seconds INTEGER
) RETURNS UUID AS $$
DECLARE
    v_header_fingerprint TEXT;
    v_claim_token UUID := pg_catalog.gen_random_uuid();
    v_row exec.order_dispatch%ROWTYPE;
BEGIN
    IF p_claimed_at IS NULL OR p_lease_seconds <= 0 THEN
        RAISE EXCEPTION 'order dispatch claim requires time and positive lease';
    END IF;
    SELECT execution_fingerprint
      INTO v_header_fingerprint
      FROM exec.order_header
     WHERE order_id = p_order_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown order_id %', p_order_id;
    END IF;
    IF v_header_fingerprint <> p_execution_fingerprint THEN
        RAISE EXCEPTION 'order dispatch execution fingerprint collision';
    END IF;

    INSERT INTO exec.order_dispatch (
        order_id, execution_fingerprint, status, claim_token,
        attempt_count, lease_expires_at, updated_at
    ) VALUES (
        p_order_id, p_execution_fingerprint, 'CLAIMED', v_claim_token,
        1, p_claimed_at + make_interval(secs => p_lease_seconds), p_claimed_at
    )
    ON CONFLICT (order_id) DO NOTHING;
    IF FOUND THEN
        INSERT INTO exec.order_dispatch_event (
            order_id, claim_token, attempt_count, event_time, event_type
        ) VALUES (
            p_order_id, v_claim_token, 1, p_claimed_at, 'CLAIMED'
        );
        RETURN v_claim_token;
    END IF;

    SELECT * INTO v_row
      FROM exec.order_dispatch
     WHERE order_id = p_order_id
     FOR UPDATE;
    IF v_row.execution_fingerprint <> p_execution_fingerprint THEN
        RAISE EXCEPTION 'order dispatch execution fingerprint collision';
    END IF;
    IF v_row.status = 'COMPLETED'
       OR (
           v_row.status = 'CLAIMED'
           AND v_row.lease_expires_at > p_claimed_at
       ) THEN
        RETURN NULL;
    END IF;

    UPDATE exec.order_dispatch
       SET status = 'CLAIMED',
           claim_token = v_claim_token,
           attempt_count = attempt_count + 1,
           lease_expires_at =
               p_claimed_at + make_interval(secs => p_lease_seconds),
           outcome_status = NULL,
           last_reason_code = NULL,
           updated_at = p_claimed_at
     WHERE order_id = p_order_id
     RETURNING * INTO v_row;
    INSERT INTO exec.order_dispatch_event (
        order_id, claim_token, attempt_count, event_time, event_type
    ) VALUES (
        p_order_id, v_claim_token, v_row.attempt_count, p_claimed_at, 'CLAIMED'
    );
    RETURN v_claim_token;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, exec;

CREATE OR REPLACE FUNCTION exec.finalize_order_dispatch(
    p_order_id UUID,
    p_claim_token UUID,
    p_event_time TIMESTAMPTZ,
    p_status TEXT,
    p_reason_code TEXT DEFAULT NULL,
    p_broker_order_id TEXT DEFAULT NULL,
    p_details JSONB DEFAULT '{}'::jsonb
) RETURNS BOOLEAN AS $$
DECLARE
    v_row exec.order_dispatch%ROWTYPE;
BEGIN
    IF p_status NOT IN ('SUBMITTED','REJECTED','SIMULATED') THEN
        RAISE EXCEPTION 'invalid dispatch completion status %', p_status;
    END IF;
    IF jsonb_typeof(p_details) <> 'object' THEN
        RAISE EXCEPTION 'dispatch details must be an object';
    END IF;
    PERFORM 1 FROM exec.order_header
     WHERE order_id = p_order_id
     FOR UPDATE;
    SELECT * INTO v_row
      FROM exec.order_dispatch
     WHERE order_id = p_order_id
     FOR UPDATE;
    IF NOT FOUND
       OR v_row.status <> 'CLAIMED'
       OR v_row.claim_token <> p_claim_token THEN
        RETURN FALSE;
    END IF;

    INSERT INTO exec.order_status_event (
        order_id, event_time, status, reason_code,
        broker_order_id, actor, details
    ) VALUES (
        p_order_id, p_event_time, p_status, p_reason_code,
        p_broker_order_id, 'execution_service', p_details
    );
    UPDATE exec.order_dispatch
       SET status = 'COMPLETED',
           outcome_status = p_status,
           last_reason_code = p_reason_code,
           updated_at = p_event_time
     WHERE order_id = p_order_id;
    INSERT INTO exec.order_dispatch_event (
        order_id, claim_token, attempt_count, event_time, event_type,
        outcome_status, reason_code, broker_order_id, details
    ) VALUES (
        p_order_id, p_claim_token, v_row.attempt_count, p_event_time,
        'COMPLETED', p_status, p_reason_code, p_broker_order_id, p_details
    );
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, exec;

CREATE OR REPLACE FUNCTION exec.require_order_dispatch_reconciliation(
    p_order_id UUID,
    p_claim_token UUID,
    p_event_time TIMESTAMPTZ,
    p_reason_code TEXT,
    p_details JSONB DEFAULT '{}'::jsonb
) RETURNS BOOLEAN AS $$
DECLARE
    v_row exec.order_dispatch%ROWTYPE;
BEGIN
    IF p_reason_code IS NULL OR btrim(p_reason_code) = '' THEN
        RAISE EXCEPTION 'dispatch reconciliation reason is required';
    END IF;
    IF jsonb_typeof(p_details) <> 'object' THEN
        RAISE EXCEPTION 'dispatch details must be an object';
    END IF;
    PERFORM 1 FROM exec.order_header
     WHERE order_id = p_order_id
     FOR UPDATE;
    SELECT * INTO v_row
      FROM exec.order_dispatch
     WHERE order_id = p_order_id
     FOR UPDATE;
    IF NOT FOUND
       OR v_row.status <> 'CLAIMED'
       OR v_row.claim_token <> p_claim_token THEN
        RETURN FALSE;
    END IF;

    INSERT INTO exec.order_status_event (
        order_id, event_time, status, reason_code, actor, details
    ) VALUES (
        p_order_id, p_event_time, 'SUBMIT_UNKNOWN',
        p_reason_code, 'execution_service', p_details
    );
    UPDATE exec.order_dispatch
       SET status = 'RECONCILIATION_REQUIRED',
           outcome_status = NULL,
           last_reason_code = p_reason_code,
           updated_at = p_event_time
     WHERE order_id = p_order_id;
    INSERT INTO exec.order_dispatch_event (
        order_id, claim_token, attempt_count, event_time, event_type,
        reason_code, details
    ) VALUES (
        p_order_id, p_claim_token, v_row.attempt_count, p_event_time,
        'RECONCILIATION_REQUIRED', p_reason_code, p_details
    );
    RETURN TRUE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER
SET search_path = pg_catalog, exec;

REVOKE ALL ON FUNCTION exec.claim_order_dispatch(
    UUID, TEXT, TIMESTAMPTZ, INTEGER
) FROM PUBLIC;
REVOKE ALL ON FUNCTION exec.finalize_order_dispatch(
    UUID, UUID, TIMESTAMPTZ, TEXT, TEXT, TEXT, JSONB
) FROM PUBLIC;
REVOKE ALL ON FUNCTION exec.require_order_dispatch_reconciliation(
    UUID, UUID, TIMESTAMPTZ, TEXT, JSONB
) FROM PUBLIC;
REVOKE ALL ON TABLE exec.order_dispatch FROM PUBLIC;
REVOKE ALL ON TABLE exec.order_dispatch_event FROM PUBLIC;

CREATE TABLE IF NOT EXISTS exec.fill_event (
    fill_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    order_id UUID NOT NULL REFERENCES exec.order_header(order_id),
    fill_time TIMESTAMPTZ NOT NULL,
    qty NUMERIC NOT NULL CHECK (
        qty > 0
        AND qty NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)
    ),
    price NUMERIC NOT NULL CHECK (
        price > 0
        AND price NOT IN ('NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC)
    ),
    commission NUMERIC NOT NULL DEFAULT 0 CHECK (
        commission >= 0
        AND commission NOT IN (
            'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
        )
    ),
    venue TEXT NOT NULL CHECK (btrim(venue) <> ''),
    broker_fill_id TEXT NOT NULL CHECK (btrim(broker_fill_id) <> ''),
    currency TEXT NOT NULL CHECK (btrim(currency) <> ''),
    raw_response_uri TEXT,
    fill_fingerprint TEXT NOT NULL UNIQUE CHECK (fill_fingerprint ~ '^sha256:[0-9a-f]{64}$'),
    UNIQUE (order_id, broker_fill_id)
);

CREATE OR REPLACE FUNCTION exec.validate_fill_event()
RETURNS TRIGGER AS $$
DECLARE
    order_currency TEXT;
    order_submitted_at TIMESTAMPTZ;
BEGIN
    SELECT currency, submitted_at
      INTO order_currency, order_submitted_at
      FROM exec.order_header
     WHERE order_id = NEW.order_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown order_id %', NEW.order_id;
    END IF;
    IF NEW.currency <> order_currency THEN
        RAISE EXCEPTION 'fill currency must match order currency';
    END IF;
    IF NEW.fill_time < order_submitted_at THEN
        RAISE EXCEPTION 'fill event predates order submission';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_fill_event_validate ON exec.fill_event;
CREATE TRIGGER trg_fill_event_validate
    BEFORE INSERT ON exec.fill_event
    FOR EACH ROW EXECUTE FUNCTION exec.validate_fill_event();

CREATE TABLE IF NOT EXISTS exec.fill_correction_event (
    correction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    fill_id UUID NOT NULL REFERENCES exec.fill_event(fill_id),
    event_time TIMESTAMPTZ NOT NULL,
    field TEXT NOT NULL CHECK (
        field IN ('qty','price','commission','venue','broker_fill_id','currency')
    ),
    old_value TEXT NOT NULL,
    new_value TEXT NOT NULL,
    reason TEXT NOT NULL,
    actor TEXT NOT NULL,
    correction_fingerprint TEXT NOT NULL UNIQUE CHECK (
        correction_fingerprint ~ '^sha256:[0-9a-f]{64}$'
    )
);

CREATE OR REPLACE FUNCTION exec.validate_fill_correction()
RETURNS TRIGGER AS $$
DECLARE
    numeric_value NUMERIC;
    current_value TEXT;
    fill_order_id UUID;
    fill_time_value TIMESTAMPTZ;
    order_currency TEXT;
    latest_correction_time TIMESTAMPTZ;
BEGIN
    SELECT order_id, fill_time
      INTO fill_order_id, fill_time_value
      FROM exec.fill_event
     WHERE fill_id = NEW.fill_id
     FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'unknown fill_id %', NEW.fill_id;
    END IF;

    SELECT event_time
      INTO latest_correction_time
      FROM exec.fill_correction_event
     WHERE fill_id = NEW.fill_id
     ORDER BY event_time DESC, correction_id DESC
     LIMIT 1;
    IF NEW.event_time < fill_time_value
       OR (
           latest_correction_time IS NOT NULL
           AND NEW.event_time <= latest_correction_time
       ) THEN
        RAISE EXCEPTION 'fill correction event_time must increase monotonically';
    END IF;

    SELECT c.new_value
      INTO current_value
      FROM exec.fill_correction_event c
     WHERE c.fill_id = NEW.fill_id
       AND c.field = NEW.field
     ORDER BY c.event_time DESC, c.correction_id DESC
     LIMIT 1;
    IF NOT FOUND THEN
        SELECT CASE NEW.field
            WHEN 'qty' THEN fe.qty::TEXT
            WHEN 'price' THEN fe.price::TEXT
            WHEN 'commission' THEN fe.commission::TEXT
            WHEN 'venue' THEN fe.venue
            WHEN 'broker_fill_id' THEN fe.broker_fill_id
            WHEN 'currency' THEN fe.currency
        END
          INTO current_value
          FROM exec.fill_event fe
         WHERE fe.fill_id = NEW.fill_id;
    END IF;

    IF NEW.field IN ('qty','price','commission') THEN
        BEGIN
            numeric_value := NEW.new_value::NUMERIC;
            IF NEW.old_value::NUMERIC <> current_value::NUMERIC THEN
                RAISE EXCEPTION 'fill correction old_value does not match effective value';
            END IF;
        EXCEPTION
            WHEN invalid_text_representation OR numeric_value_out_of_range THEN
            RAISE EXCEPTION 'fill correction % must be numeric', NEW.field;
        END;
        IF numeric_value IN (
               'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC
           )
           OR (NEW.field IN ('qty','price') AND numeric_value <= 0)
           OR (NEW.field = 'commission' AND numeric_value < 0) THEN
            RAISE EXCEPTION 'invalid corrected % value %', NEW.field, NEW.new_value;
        END IF;
    ELSIF NEW.old_value IS DISTINCT FROM current_value THEN
        RAISE EXCEPTION 'fill correction old_value does not match effective value';
    END IF;

    IF NEW.field IN ('venue','broker_fill_id','currency')
       AND btrim(NEW.new_value) = '' THEN
        RAISE EXCEPTION 'corrected % cannot be blank', NEW.field;
    END IF;

    IF NEW.field = 'currency' THEN
        SELECT currency INTO order_currency
          FROM exec.order_header
         WHERE order_id = fill_order_id;
        IF NEW.new_value <> order_currency THEN
            RAISE EXCEPTION 'fill currency must match order currency';
        END IF;
    END IF;

    IF NEW.field = 'broker_fill_id'
       AND EXISTS (
           SELECT 1
             FROM exec.v_effective_fill effective
            WHERE effective.order_id = fill_order_id
              AND effective.fill_id <> NEW.fill_id
              AND effective.broker_fill_id = NEW.new_value
       ) THEN
        RAISE EXCEPTION 'broker_fill_id cannot collide within order';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_fill_correction_validate ON exec.fill_correction_event;
CREATE TRIGGER trg_fill_correction_validate
    BEFORE INSERT ON exec.fill_correction_event
    FOR EACH ROW EXECUTE FUNCTION exec.validate_fill_correction();

CREATE INDEX IF NOT EXISTS idx_order_header_target
    ON exec.order_header (target_id, submitted_at);
CREATE INDEX IF NOT EXISTS idx_order_status_order_time
    ON exec.order_status_event (order_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_fill_order_time
    ON exec.fill_event (order_id, fill_time);

CREATE OR REPLACE VIEW exec.v_effective_fill AS
SELECT
    fe.fill_id,
    fe.order_id,
    fe.fill_time,
    COALESCE((
        SELECT c.new_value::NUMERIC
        FROM exec.fill_correction_event c
        WHERE c.fill_id = fe.fill_id AND c.field = 'qty'
        ORDER BY c.event_time DESC, c.correction_id DESC
        LIMIT 1
    ), fe.qty) AS qty,
    COALESCE((
        SELECT c.new_value::NUMERIC
        FROM exec.fill_correction_event c
        WHERE c.fill_id = fe.fill_id AND c.field = 'price'
        ORDER BY c.event_time DESC, c.correction_id DESC
        LIMIT 1
    ), fe.price) AS price,
    COALESCE((
        SELECT c.new_value::NUMERIC
        FROM exec.fill_correction_event c
        WHERE c.fill_id = fe.fill_id AND c.field = 'commission'
        ORDER BY c.event_time DESC, c.correction_id DESC
        LIMIT 1
    ), fe.commission) AS commission,
    COALESCE((
        SELECT c.new_value
        FROM exec.fill_correction_event c
        WHERE c.fill_id = fe.fill_id AND c.field = 'venue'
        ORDER BY c.event_time DESC, c.correction_id DESC
        LIMIT 1
    ), fe.venue) AS venue,
    COALESCE((
        SELECT c.new_value
        FROM exec.fill_correction_event c
        WHERE c.fill_id = fe.fill_id AND c.field = 'broker_fill_id'
        ORDER BY c.event_time DESC, c.correction_id DESC
        LIMIT 1
    ), fe.broker_fill_id) AS broker_fill_id,
    COALESCE((
        SELECT c.new_value
        FROM exec.fill_correction_event c
        WHERE c.fill_id = fe.fill_id AND c.field = 'currency'
        ORDER BY c.event_time DESC, c.correction_id DESC
        LIMIT 1
    ), fe.currency) AS currency,
    fe.raw_response_uri,
    fe.fill_fingerprint AS base_fill_fingerprint,
    CASE
        WHEN NOT EXISTS (
            SELECT 1
              FROM exec.fill_correction_event c
             WHERE c.fill_id = fe.fill_id
        ) THEN fe.fill_fingerprint
        ELSE 'sha256:' || encode(
            digest(
                fe.fill_fingerprint || E'\x1f' || (
                    SELECT string_agg(
                        c.correction_fingerprint,
                        E'\x1f' ORDER BY c.event_time, c.correction_id
                    )
                      FROM exec.fill_correction_event c
                     WHERE c.fill_id = fe.fill_id
                ),
                'sha256'
            ),
            'hex'
        )
    END AS effective_fill_fingerprint
FROM exec.fill_event fe;

CREATE OR REPLACE VIEW exec.v_order_state AS
SELECT
    h.order_id,
    h.client_order_id,
    h.idempotency_key,
    h.account_id,
    h.env,
    h.executor_type,
    h.broker_id,
    h.strategy_id,
    h.sleeve_id,
    h.target_id,
    h.target_version,
    h.allocation_id,
    h.instrument_id,
    h.instrument,
    h.side,
    h.qty,
    h.order_type,
    h.limit_price,
    h.tif,
    h.currency,
    h.parent_order_id,
    h.decision_fingerprint,
    h.execution_fingerprint,
    h.rebalance_cutoff,
    h.submitted_at,
    h.created_at,
    latest.event_time AS status_time,
    latest.status,
    latest.reason_code,
    latest.broker_order_id,
    COALESCE(f.filled_qty, 0) AS filled_qty,
    f.average_fill_price,
    COALESCE(f.total_commission, 0) AS total_commission
FROM exec.order_header h
LEFT JOIN LATERAL (
    SELECT e.event_time, e.status, e.reason_code, e.broker_order_id
    FROM exec.order_status_event e
    WHERE e.order_id = h.order_id
    ORDER BY e.event_time DESC, e.event_id DESC
    LIMIT 1
) latest ON TRUE
LEFT JOIN LATERAL (
    SELECT
        SUM(fe.qty) AS filled_qty,
        SUM(fe.qty * fe.price) / NULLIF(SUM(fe.qty), 0) AS average_fill_price,
        SUM(fe.commission) AS total_commission
    FROM exec.v_effective_fill fe
    WHERE fe.order_id = h.order_id
) f ON TRUE;

CREATE OR REPLACE FUNCTION exec.block_event_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION '% is event-sourced and append-only', TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_order_header_immutable ON exec.order_header;
CREATE TRIGGER trg_order_header_immutable
    BEFORE UPDATE OR DELETE ON exec.order_header
    FOR EACH ROW EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_order_status_immutable ON exec.order_status_event;
CREATE TRIGGER trg_order_status_immutable
    BEFORE UPDATE OR DELETE ON exec.order_status_event
    FOR EACH ROW EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_fill_event_immutable ON exec.fill_event;
CREATE TRIGGER trg_fill_event_immutable
    BEFORE UPDATE OR DELETE ON exec.fill_event
    FOR EACH ROW EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_fill_correction_immutable ON exec.fill_correction_event;
CREATE TRIGGER trg_fill_correction_immutable
    BEFORE UPDATE OR DELETE ON exec.fill_correction_event
    FOR EACH ROW EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_order_dispatch_event_immutable
    ON exec.order_dispatch_event;
CREATE TRIGGER trg_order_dispatch_event_immutable
    BEFORE UPDATE OR DELETE ON exec.order_dispatch_event
    FOR EACH ROW EXECUTE FUNCTION exec.block_event_mutation();

DROP TRIGGER IF EXISTS trg_order_header_no_truncate ON exec.order_header;
CREATE TRIGGER trg_order_header_no_truncate
    BEFORE TRUNCATE ON exec.order_header
    FOR EACH STATEMENT EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_order_status_no_truncate ON exec.order_status_event;
CREATE TRIGGER trg_order_status_no_truncate
    BEFORE TRUNCATE ON exec.order_status_event
    FOR EACH STATEMENT EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_fill_event_no_truncate ON exec.fill_event;
CREATE TRIGGER trg_fill_event_no_truncate
    BEFORE TRUNCATE ON exec.fill_event
    FOR EACH STATEMENT EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_fill_correction_no_truncate ON exec.fill_correction_event;
CREATE TRIGGER trg_fill_correction_no_truncate
    BEFORE TRUNCATE ON exec.fill_correction_event
    FOR EACH STATEMENT EXECUTE FUNCTION exec.block_event_mutation();
DROP TRIGGER IF EXISTS trg_order_dispatch_event_no_truncate
    ON exec.order_dispatch_event;
CREATE TRIGGER trg_order_dispatch_event_no_truncate
    BEFORE TRUNCATE ON exec.order_dispatch_event
    FOR EACH STATEMENT EXECUTE FUNCTION exec.block_event_mutation();

COMMENT ON VIEW exec.v_order_state IS
    'Current order state is a projection of immutable events, never an UPDATE.';
