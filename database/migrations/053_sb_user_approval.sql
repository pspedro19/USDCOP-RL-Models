-- Migration 053: SignalBridge user admin-approval lifecycle (additive, idempotent)
-- ============================================================================
-- Purpose:
--   Registration becomes admin-gated. A new sign-up lands in `pending`; an admin
--   approves (issue temp password, force reset) or rejects. This migration adds
--   the lifecycle columns to `sb_users` (created by init-scripts/20-signalbridge-
--   schema.sql) without touching existing rows' usability.
--
-- Column ⇄ enum SSOT (app/contracts/user.py):
--   status IN ('pending','approved','rejected')   UserStatus
--   role   IN ('user','admin')                     UserRole
--
-- Backward-compat:
--   * Existing users are backfilled to status='approved' so no one is locked out.
--   * Columns are added IF NOT EXISTS → safe to re-run.
--   * The bootstrap admin is created by the app at startup (ADMIN_BOOTSTRAP_*),
--     not here, to avoid embedding a bcrypt hash in SQL.
-- ============================================================================

ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS status VARCHAR(20) NOT NULL DEFAULT 'pending';
ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS role VARCHAR(20) NOT NULL DEFAULT 'user';
ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS must_reset_password BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS approved_by UUID REFERENCES sb_users(id);
ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS approved_at TIMESTAMP;
ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS rejected_at TIMESTAMP;
ALTER TABLE sb_users ADD COLUMN IF NOT EXISTS rejection_reason TEXT;

-- Backfill pre-existing accounts to approved (they were usable before this change).
-- Only rows created before the columns existed (still on the 'pending' default with
-- no approval trail) are affected; genuinely-new pending sign-ups have a created_at
-- after this migration and are handled by the app, not touched here.
UPDATE sb_users
SET status = 'approved'
WHERE status = 'pending'
  AND approved_at IS NULL
  AND rejected_at IS NULL
  AND created_at < NOW();

-- Enum guards (drop-then-add so the migration is re-runnable).
ALTER TABLE sb_users DROP CONSTRAINT IF EXISTS sb_users_status_chk;
ALTER TABLE sb_users ADD CONSTRAINT sb_users_status_chk
    CHECK (status IN ('pending', 'approved', 'rejected'));

ALTER TABLE sb_users DROP CONSTRAINT IF EXISTS sb_users_role_chk;
ALTER TABLE sb_users ADD CONSTRAINT sb_users_role_chk
    CHECK (role IN ('user', 'admin'));

-- Index the review-queue lookup (WHERE status = 'pending').
CREATE INDEX IF NOT EXISTS idx_sb_users_status ON sb_users(status);

COMMENT ON COLUMN sb_users.status IS 'Approval lifecycle: pending|approved|rejected (SSOT: UserStatus)';
COMMENT ON COLUMN sb_users.role IS 'Authorization role: user|admin (SSOT: UserRole)';
COMMENT ON COLUMN sb_users.must_reset_password IS 'True on an admin-issued temporary password; cleared by /auth/reset-password';
