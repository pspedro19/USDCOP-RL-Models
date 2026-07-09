-- ============================================================================
-- Migration 056: Admin Console — test-traffic flag (CTR-ADMIN-CONSOLE-001, spec C4)
-- ============================================================================
-- Business metrics and the approval-queue counter must exclude test traffic.
-- `is_test` lives on sb_users ONLY: audit_log is append-only (trigger from 055),
-- so audit entries derive their test-ness by JOINing the actor's sb_users row —
-- no ledger mutation, no backfill against the append-only trigger.
--
-- Additive + idempotent. Manual toggle: POST /api/admin/users/:id/flag-test
-- (audited, action 'user_flag_test').
-- ============================================================================

ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS is_test BOOLEAN NOT NULL DEFAULT FALSE;

-- Heuristic backfill (spec C4): obvious QA/test domains. Manual flag wins afterwards.
UPDATE sb_users
SET is_test = TRUE
WHERE is_test = FALSE
  AND (
    email ~* '@test\.com$'
    OR email ~* '@example\.(com|org|net)$'
    OR email ~* '@[^@]*\.local$'
    OR email ~* '@mailinator\.com$'
    OR email ~* '^qa[._+-]'
  );

CREATE INDEX IF NOT EXISTS idx_sb_users_is_test ON sb_users (is_test) WHERE is_test = TRUE;

-- New registrations matching the heuristic are auto-flagged at INSERT time, so the
-- backfill above is not a one-shot: QA signups created tomorrow classify themselves.
-- The trigger only fires on INSERT — a manual admin un-flag (UPDATE) always wins.
CREATE OR REPLACE FUNCTION sb_users_default_is_test() RETURNS trigger AS $$
BEGIN
  IF NEW.is_test IS DISTINCT FROM TRUE AND (
       NEW.email ~* '@test\.com$'
    OR NEW.email ~* '@example\.(com|org|net)$'
    OR NEW.email ~* '@[^@]*\.local$'
    OR NEW.email ~* '@mailinator\.com$'
    OR NEW.email ~* '^qa[._+-]'
  ) THEN
    NEW.is_test := TRUE;
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_sb_users_default_is_test ON sb_users;
CREATE TRIGGER trg_sb_users_default_is_test
  BEFORE INSERT ON sb_users
  FOR EACH ROW EXECUTE FUNCTION sb_users_default_is_test();

COMMENT ON COLUMN sb_users.is_test IS
  'CTR-ADMIN-CONSOLE-001 (C4): test/QA account — excluded from business KPIs and the queue counter; audit view derives is_test via JOIN (audit_log stays append-only).';
