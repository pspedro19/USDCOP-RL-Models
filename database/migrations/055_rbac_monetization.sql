-- ============================================================================
-- 055_rbac_monetization.sql — RBAC + monetization + SignalBridge multi-tenant
-- ============================================================================
-- CTR-RBAC-001 (spec: .claude/specs/platform/rbac-monetization.md)
--
-- Adds:
--   1. sb_users.entitlements JSONB — the PLAN (what you paid): assets, delays,
--      execution ceilings, expires_at. Role stays sb_users.role (WHO you are).
--   2. audit_log — append-only ledger for Vote-2/promote/kills/plan/key events.
--   3. user_exchange_keys / user_risk_limits_v2 / user_executions — multi-tenant
--      SignalBridge (per-user keys + limits + append-only executions).
--
-- Additive & idempotent (IF NOT EXISTS everywhere). Never touches strategy logic.
-- Date: 2026-07-06
-- ============================================================================

-- 1) Entitlements on the existing SignalBridge users table --------------------
ALTER TABLE sb_users
    ADD COLUMN IF NOT EXISTS entitlements JSONB NOT NULL DEFAULT '{"plan":"free"}'::jsonb;

COMMENT ON COLUMN sb_users.entitlements IS
    'CTR-RBAC-001: plan/assets/delays/execution ceilings/expires_at. Expired => served as free. '
    'JWT caches {role,plan}; sensitive ops re-validate against THIS column.';

-- Existing admin keeps full access semantics via role; give it the auto plan explicitly.
UPDATE sb_users SET entitlements = jsonb_build_object(
    'plan', 'auto', 'assets', jsonb_build_array('usdcop','xauusd','btcusdt'),
    'forecast_delay_hours', 0, 'analysis_delay_days', 0, 'signals_realtime', true,
    'execution', jsonb_build_object('enabled', true, 'mode', 'paper',
        'paper_weeks_required', 0, 'max_notional_usd', 100000,
        'max_daily_loss_pct', 5.0, 'max_open_positions', 10),
    'expires_at', NULL)
WHERE role = 'admin' AND (entitlements->>'plan') = 'free';

-- 2) Append-only audit log ----------------------------------------------------
CREATE TABLE IF NOT EXISTS audit_log (
    id           BIGSERIAL PRIMARY KEY,
    user_id      UUID,
    action       VARCHAR(64)  NOT NULL,   -- vote2_approve|vote2_reject|promote|kill_global|
                                          -- kill_user|plan_change|key_add|key_remove|
                                          -- live_enable|execution|login_denied
    object_type  VARCHAR(64),
    object_id    VARCHAR(128),
    detail       JSONB,
    ip           VARCHAR(64),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_log_user   ON audit_log (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_log_action ON audit_log (action, created_at DESC);

-- Append-only enforcement: no UPDATE/DELETE, ever (rule beats revocable grants).
CREATE OR REPLACE FUNCTION audit_log_block_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only (CTR-RBAC-001 rule 11)';
END; $$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_audit_log_no_update ON audit_log;
CREATE TRIGGER trg_audit_log_no_update
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_block_mutation();

-- 3) SignalBridge multi-tenant -------------------------------------------------
-- Per-user exchange keys (encrypted by VaultService AES-256-GCM before insert;
-- NEVER plaintext). Keys with withdraw permission are rejected at registration.
CREATE TABLE IF NOT EXISTS user_exchange_keys (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES sb_users(id) ON DELETE CASCADE,
    exchange          VARCHAR(50) NOT NULL,
    api_key_enc       TEXT NOT NULL,
    api_secret_enc    TEXT NOT NULL,
    key_version       INTEGER NOT NULL DEFAULT 1,
    status            VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending|verified|revoked
    last_verified_at  TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, exchange)
);

-- Per-user risk limits (v2: supersedes user_risk_limits for the multi-tenant flow;
-- the old table remains read by RiskBridgeService until cutover).
CREATE TABLE IF NOT EXISTS user_risk_limits_v2 (
    user_id             UUID PRIMARY KEY REFERENCES sb_users(id) ON DELETE CASCADE,
    max_notional_usd    DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_daily_loss_pct  DOUBLE PRECISION NOT NULL DEFAULT 0,
    max_open_positions  INTEGER          NOT NULL DEFAULT 0,
    mode                VARCHAR(10)      NOT NULL DEFAULT 'paper',  -- paper|live
    kill_switch         BOOLEAN          NOT NULL DEFAULT FALSE,
    live_enabled_at     TIMESTAMPTZ,          -- set when paper_weeks_required satisfied
    risk_accepted_at    TIMESTAMPTZ,          -- ToS + risk disclosure acceptance (audited)
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-user executions (append-only mirror for fan-out accounting; sb_executions
-- remains the operational table).
CREATE TABLE IF NOT EXISTS user_executions (
    id          BIGSERIAL PRIMARY KEY,
    user_id     UUID NOT NULL,
    signal_id   VARCHAR(128),
    exchange    VARCHAR(50),
    symbol      VARCHAR(30),
    side        VARCHAR(8),
    qty         DOUBLE PRECISION,
    px          DOUBLE PRECISION,
    status      VARCHAR(20),
    mode        VARCHAR(10) NOT NULL DEFAULT 'paper',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_user_executions_user ON user_executions (user_id, created_at DESC);

-- 4) Role check extended to RBAC roles (CTR-RBAC-001; legacy 'user' kept for back-compat)
ALTER TABLE sb_users DROP CONSTRAINT IF EXISTS sb_users_role_chk;
ALTER TABLE sb_users ADD CONSTRAINT sb_users_role_chk
    CHECK (role::text = ANY (ARRAY['user','admin','developer','subscriber','free']::text[]));

-- 5) Legacy dashboard `users` table compatibility (AuthService fallback queried
--    role/full_name/avatar_url that never existed -> noisy login errors)
ALTER TABLE users ADD COLUMN IF NOT EXISTS role VARCHAR(20) DEFAULT 'user';
ALTER TABLE users ADD COLUMN IF NOT EXISTS full_name VARCHAR(255);
ALTER TABLE users ADD COLUMN IF NOT EXISTS avatar_url TEXT;
UPDATE users SET role='admin' WHERE is_admin = true AND role='user';
