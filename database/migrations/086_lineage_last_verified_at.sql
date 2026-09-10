-- Migration 086: distinguish observed checks from revision events (C033 / BL-24)
--
-- Existing lineage nodes predate the verification clock. Their created_at is
-- copied only as an inferred lower bound so the new column can be NOT NULL; it
-- is not evidence that a producer actually re-checked the observation then.
-- New writers advance last_verified_at with their observed, timezone-aware run
-- time even when the semantic hash is unchanged. revision_event remains reserved
-- for actual value/schema changes.

ALTER TABLE lineage.node
    ADD COLUMN IF NOT EXISTS last_verified_at TIMESTAMPTZ;

COMMENT ON COLUMN lineage.node.last_verified_at IS
    'Latest producer check. Legacy values backfilled from created_at are an inferred lower bound, not an observed verification.';

UPDATE lineage.node
SET last_verified_at = created_at
WHERE last_verified_at IS NULL;

ALTER TABLE lineage.node
    ALTER COLUMN last_verified_at SET DEFAULT NOW();

ALTER TABLE lineage.node
    ALTER COLUMN last_verified_at SET NOT NULL;
