-- Migration 085: DB-owned provenance time for feature availability (C031 / BL-40)
--
-- Existing rows predate this contract and keep created_at NULL. They remain
-- immutable evidence, but consumers cannot treat an unknown insertion time as
-- an authoritative availability measurement.

ALTER TABLE quality.feature_status
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ;

ALTER TABLE quality.feature_status
    ALTER COLUMN created_at SET DEFAULT clock_timestamp();

ALTER TABLE quality.feature_status
    DROP CONSTRAINT IF EXISTS feature_status_observed_not_after_created;
ALTER TABLE quality.feature_status
    ADD CONSTRAINT feature_status_observed_not_after_created
    CHECK (created_at IS NULL OR observed_at <= created_at) NOT VALID;

CREATE OR REPLACE FUNCTION quality.enforce_feature_status_provenance()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Ignore caller input: creation time is owned by the database clock.
    NEW.created_at := clock_timestamp();
    IF NEW.observed_at > NEW.created_at THEN
        RAISE EXCEPTION
            'feature_status observed_at (%) cannot exceed DB creation time (%)',
            NEW.observed_at, NEW.created_at
            USING ERRCODE = '22007';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_feature_status_provenance
    ON quality.feature_status;
CREATE TRIGGER trg_feature_status_provenance
    BEFORE INSERT ON quality.feature_status
    FOR EACH ROW
    EXECUTE FUNCTION quality.enforce_feature_status_provenance();
