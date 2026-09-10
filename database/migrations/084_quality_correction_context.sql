-- Migration 084: reproducible market-quarantine correction context (C027 / BL-40)
-- Additive only: migration 073 is already applied and remains immutable.

ALTER TABLE quality.quarantine_event
    ADD COLUMN IF NOT EXISTS provider_id TEXT,
    ADD COLUMN IF NOT EXISTS provider_symbol TEXT,
    ADD COLUMN IF NOT EXISTS interval_id TEXT REFERENCES reference.bar_interval(interval_id),
    ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS source_uri TEXT,
    ADD COLUMN IF NOT EXISTS context_version SMALLINT;

ALTER TABLE quality.quarantine_event
    DROP CONSTRAINT IF EXISTS quarantine_context_version_valid;
ALTER TABLE quality.quarantine_event
    ADD CONSTRAINT quarantine_context_version_valid
    CHECK (context_version IS NULL OR context_version = 1);

CREATE OR REPLACE FUNCTION quality.require_new_ohlcv_quarantine_context()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.entity_type = 'ohlcv_bar' AND (
        NEW.context_version IS DISTINCT FROM 1
        OR NULLIF(BTRIM(NEW.provider_id), '') IS NULL
        OR NULLIF(BTRIM(NEW.provider_symbol), '') IS NULL
        OR NULLIF(BTRIM(NEW.interval_id), '') IS NULL
        OR NEW.observed_at IS NULL
        OR NULLIF(BTRIM(NEW.source_uri), '') IS NULL
    ) THEN
        RAISE EXCEPTION
            'new ohlcv quarantine requires typed provider/symbol/interval/time/source context';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_new_ohlcv_quarantine_context
    ON quality.quarantine_event;
CREATE TRIGGER trg_new_ohlcv_quarantine_context
    BEFORE INSERT ON quality.quarantine_event
    FOR EACH ROW EXECUTE FUNCTION quality.require_new_ohlcv_quarantine_context();

DO $$
BEGIN
    IF EXISTS (
        SELECT quarantine_id
        FROM quality.correction_event
        GROUP BY quarantine_id
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'quality.correction_event contains multiple corrections for one quarantine';
    END IF;
END;
$$;

CREATE UNIQUE INDEX IF NOT EXISTS uq_quality_correction_quarantine
    ON quality.correction_event (quarantine_id);

ALTER TABLE quality.quarantine_event
    DROP CONSTRAINT IF EXISTS quarantine_corrected_state_consistent;
ALTER TABLE quality.quarantine_event
    ADD CONSTRAINT quarantine_corrected_state_consistent
    CHECK ((status = 'CORRECTED') = (correction_event_id IS NOT NULL)) NOT VALID;

COMMENT ON COLUMN quality.quarantine_event.context_version IS
    'C027 typed replay context; NULL denotes immutable legacy evidence that is not auto-correctable.';
COMMENT ON COLUMN quality.quarantine_event.observed_at IS
    'Original economic observation instant used to re-evaluate provider/date-scoped quality rules.';
