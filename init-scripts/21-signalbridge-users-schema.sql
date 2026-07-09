-- ============================================================================
-- SignalBridge User Schema (sb_* tables) — matches SQLAlchemy ORM in app/models.py
-- ============================================================================
-- Purpose: Create the UUID-based `sb_*` tables the SignalBridge ORM actually
--          uses for registration / login / execution.
--
-- CONTEXT (2026-07-03): main.py skips Base.metadata.create_all and the legacy
-- 20-signalbridge-schema.sql only creates INTEGER-keyed `users`/`signals`/…
-- tables that reference the dashboard `users` table. The ORM (app/models.py)
-- expects `sb_users`, `sb_trading_configs`, `sb_exchange_credentials`,
-- `sb_credential_audit_logs`, `sb_signals`, `sb_executions` with UUID PKs.
-- Without this script those tables never exist and /api/auth/register + /login
-- fail with "relation sb_users does not exist" against a real database.
--
-- This script is idempotent (CREATE TABLE IF NOT EXISTS). Apply on fresh init
-- automatically, or manually on an existing DB:
--   docker exec -i usdcop-postgres-timescale psql -U admin -d usdcop_trading \
--     < init-scripts/21-signalbridge-users-schema.sql
-- ============================================================================

-- Users (SSOT for SignalBridge auth) -----------------------------------------
CREATE TABLE IF NOT EXISTS sb_users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    name            VARCHAR(255),
    is_active       BOOLEAN NOT NULL DEFAULT true,
    is_verified     BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP,
    last_login      TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sb_users_email ON sb_users(email);

-- Per-user trading config ----------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_trading_configs (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                  UUID NOT NULL UNIQUE REFERENCES sb_users(id) ON DELETE CASCADE,
    trading_enabled          BOOLEAN NOT NULL DEFAULT false,
    default_exchange         VARCHAR(50),
    max_position_size        DOUBLE PRECISION DEFAULT 0.1,
    stop_loss_percent        DOUBLE PRECISION DEFAULT 5.0,
    take_profit_percent      DOUBLE PRECISION DEFAULT 10.0,
    use_trailing_stop        BOOLEAN DEFAULT false,
    trailing_stop_percent    DOUBLE PRECISION DEFAULT 2.0,
    allowed_symbols          VARCHAR[] DEFAULT '{}',
    blocked_symbols          VARCHAR[] DEFAULT '{}',
    max_daily_trades         INTEGER DEFAULT 50,
    max_concurrent_positions INTEGER DEFAULT 5,
    created_at               TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at               TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sb_trading_configs_user ON sb_trading_configs(user_id);

-- Encrypted exchange credentials ---------------------------------------------
CREATE TABLE IF NOT EXISTS sb_exchange_credentials (
    id                    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id               UUID NOT NULL REFERENCES sb_users(id) ON DELETE CASCADE,
    exchange              VARCHAR(50) NOT NULL,
    label                 VARCHAR(100) NOT NULL,
    encrypted_api_key     TEXT NOT NULL,
    encrypted_api_secret  TEXT NOT NULL,
    encrypted_passphrase  TEXT,
    key_version           VARCHAR(50) NOT NULL,
    is_testnet            BOOLEAN DEFAULT false,
    is_active             BOOLEAN DEFAULT true,
    is_valid              BOOLEAN DEFAULT true,
    created_at            TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at            TIMESTAMP,
    last_used             TIMESTAMP,
    last_validated        TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_sb_credentials_user ON sb_exchange_credentials(user_id);

-- Credential audit log -------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_credential_audit_logs (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    credential_id UUID NOT NULL REFERENCES sb_exchange_credentials(id) ON DELETE CASCADE,
    action        VARCHAR(50) NOT NULL,
    actor_id      UUID NOT NULL,
    ip_address    VARCHAR(45),
    details       TEXT,
    created_at    TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sb_cred_audit_credential ON sb_credential_audit_logs(credential_id);

-- Signals (UUID-keyed, SignalBridge-native) ----------------------------------
CREATE TABLE IF NOT EXISTS sb_signals (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES sb_users(id) ON DELETE CASCADE,
    symbol          VARCHAR(20) NOT NULL,
    action          INTEGER NOT NULL,
    price           DOUBLE PRECISION,
    quantity        DOUBLE PRECISION,
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    source          VARCHAR(50) DEFAULT 'api',
    signal_metadata JSON DEFAULT '{}',
    is_processed    BOOLEAN NOT NULL DEFAULT false,
    processed_at    TIMESTAMP,
    execution_id    UUID,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sb_signals_user ON sb_signals(user_id);
CREATE INDEX IF NOT EXISTS idx_sb_signals_symbol ON sb_signals(symbol);

-- Executions -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sb_executions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL REFERENCES sb_users(id) ON DELETE CASCADE,
    signal_id          UUID,
    exchange           VARCHAR(50) NOT NULL,
    credential_id      UUID NOT NULL REFERENCES sb_exchange_credentials(id) ON DELETE CASCADE,
    symbol             VARCHAR(20) NOT NULL,
    side               VARCHAR(10) NOT NULL,
    order_type         VARCHAR(30) NOT NULL DEFAULT 'market',
    quantity           DOUBLE PRECISION NOT NULL,
    price              DOUBLE PRECISION,
    stop_loss          DOUBLE PRECISION,
    take_profit        DOUBLE PRECISION,
    status             VARCHAR(20) NOT NULL DEFAULT 'pending',
    exchange_order_id  VARCHAR(100),
    filled_quantity    DOUBLE PRECISION DEFAULT 0.0,
    average_price      DOUBLE PRECISION DEFAULT 0.0,
    commission         DOUBLE PRECISION DEFAULT 0.0,
    commission_asset   VARCHAR(20),
    executed_at        TIMESTAMP,
    error_message      TEXT,
    raw_response       JSON,
    execution_metadata JSON DEFAULT '{}',
    created_at         TIMESTAMP NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sb_executions_user ON sb_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_sb_executions_symbol ON sb_executions(symbol);
CREATE INDEX IF NOT EXISTS idx_sb_executions_status ON sb_executions(status);

-- Permissions ----------------------------------------------------------------
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO admin;

COMMENT ON TABLE sb_users IS 'SignalBridge: authentication SSOT (UUID). Used by /api/auth/register + /login.';
COMMENT ON TABLE sb_trading_configs IS 'SignalBridge: per-user trading config (UUID FK to sb_users).';
COMMENT ON TABLE sb_exchange_credentials IS 'SignalBridge: encrypted exchange API keys (UUID FK to sb_users).';
