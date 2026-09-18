-- Migration 091 — expiring permission grants (the NDA data room).
--
-- WHY: selling research access during a deal requires handing a prospect `research:read`
-- for a few weeks. `rbac_user_overrides` could express the grant but not its END: every
-- override was permanent until a human remembered to clear it. "Remembering to revoke" is
-- the documented failure mode of every data room, and the one thing a buyer's counsel asks
-- about. An access window that closes itself is the control; a calendar reminder is not.
--
-- DESIGN: `expires_at NULL` means "no expiry", so every existing override keeps its exact
-- current meaning and nothing needs backfilling. Expiry is enforced at RESOLUTION time
-- (lib/auth/rbac-resolver) rather than by a sweeper job, so a lapsed grant stops working
-- immediately even if no job has run — the row is evidence of what was granted, never the
-- thing that keeps it alive.
--
-- `nda_reference` records WHICH signed agreement authorised the grant. Access to research
-- internals without a traceable agreement is exactly what the audit trail exists to make
-- impossible to do quietly.

ALTER TABLE rbac_user_overrides
  ADD COLUMN IF NOT EXISTS expires_at TIMESTAMPTZ;

ALTER TABLE rbac_user_overrides
  ADD COLUMN IF NOT EXISTS nda_reference TEXT;

-- A DENY must never carry an expiry: a revocation that silently lapses would re-grant a
-- permission that someone deliberately took away. Expiry is a property of GRANTS only.
ALTER TABLE rbac_user_overrides DROP CONSTRAINT IF EXISTS rbac_override_expiry_grants_only;
ALTER TABLE rbac_user_overrides
  ADD CONSTRAINT rbac_override_expiry_grants_only
  CHECK (expires_at IS NULL OR effect = 'grant');

-- Partial index: the admin console lists expiring grants to show what is open right now.
CREATE INDEX IF NOT EXISTS idx_rbac_overrides_expiring
  ON rbac_user_overrides (expires_at)
  WHERE expires_at IS NOT NULL;

COMMENT ON COLUMN rbac_user_overrides.expires_at IS
  'When a temporary grant stops applying. NULL = permanent. Enforced at resolution time, '
  'not by a sweeper, so a lapsed grant is dead immediately.';
COMMENT ON COLUMN rbac_user_overrides.nda_reference IS
  'Signed agreement that authorises a temporary research grant (data-room access).';
