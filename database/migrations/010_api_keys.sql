-- Migration: 010_api_keys.sql
-- Purpose: API authentication infrastructure for inference service
-- Contract: CTR-AUTH-001
-- Date: 2026-01-16

-- =============================================================================
-- 1. API KEYS TABLE
-- =============================================================================
-- Stores hashed API keys for authentication

CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,

    -- Key identification (never store raw key!)
    key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 hash of the key
    key_prefix VARCHAR(12) NOT NULL,        -- First 12 chars for identification

    -- Metadata
    name VARCHAR(100) NOT NULL,             -- Human-friendly name for the key
    description TEXT,                       -- Optional description/purpose

    -- Ownership and permissions
    user_id VARCHAR(50),                    -- Owner identifier
    roles TEXT[] DEFAULT ARRAY['trader'],   -- Roles: trader, admin, readonly
    scopes TEXT[] DEFAULT ARRAY['read', 'write'],  -- API scopes

    -- Rate limiting (per-key override)
    rate_limit_per_minute INT DEFAULT 100,  -- Requests per minute
    rate_limit_per_day INT DEFAULT 10000,   -- Requests per day

    -- Status
    is_active BOOLEAN DEFAULT TRUE,         -- Can be deactivated without deletion
    expires_at TIMESTAMPTZ,                 -- Optional expiration date

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by VARCHAR(50),
    last_used_at TIMESTAMPTZ,
    last_used_ip VARCHAR(45),               -- IPv4 or IPv6
    use_count BIGINT DEFAULT 0              -- Total usage count
);

-- Index for fast key lookup (primary authentication path)
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

-- Index for listing active keys by user
CREATE INDEX IF NOT EXISTS idx_api_keys_user_active ON api_keys(user_id, is_active)
    WHERE is_active = TRUE;

-- Index for finding expired keys
CREATE INDEX IF NOT EXISTS idx_api_keys_expires ON api_keys(expires_at)
    WHERE expires_at IS NOT NULL;

-- Table comment
COMMENT ON TABLE api_keys IS 'API keys for authenticating inference service requests. Contract: CTR-AUTH-001';
COMMENT ON COLUMN api_keys.key_hash IS 'SHA-256 hash of the API key (never store raw keys!)';
COMMENT ON COLUMN api_keys.key_prefix IS 'First 12 characters for key identification in logs';
COMMENT ON COLUMN api_keys.roles IS 'Roles granted to this key: trader, admin, readonly';


-- =============================================================================
-- 2. API KEY USAGE LOG
-- =============================================================================
-- Detailed usage tracking for analytics and security

CREATE TABLE IF NOT EXISTS api_key_usage_log (
    id BIGSERIAL PRIMARY KEY,
    key_id INT NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,

    -- Request details
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INT,
    response_time_ms INT,

    -- Client info
    client_ip VARCHAR(45),
    user_agent VARCHAR(500),

    -- Request metadata
    request_id VARCHAR(50),                 -- Correlation ID
    request_size_bytes INT,
    response_size_bytes INT
);

-- Partition by month for efficient queries and cleanup
-- (Note: Requires TimescaleDB or manual partitioning for production)

-- Index for time-series queries
CREATE INDEX IF NOT EXISTS idx_api_usage_log_timestamp
    ON api_key_usage_log(timestamp DESC);

-- Index for per-key queries
CREATE INDEX IF NOT EXISTS idx_api_usage_log_key_id
    ON api_key_usage_log(key_id, timestamp DESC);

-- Comment
COMMENT ON TABLE api_key_usage_log IS 'Detailed API key usage log for analytics and security auditing';


-- =============================================================================
-- 3. RATE LIMIT STATE TABLE (for distributed rate limiting)
-- =============================================================================
-- Stores rate limit state for API keys (alternative to Redis)

CREATE TABLE IF NOT EXISTS api_rate_limit_state (
    key_hash VARCHAR(64) PRIMARY KEY,
    window_start TIMESTAMPTZ NOT NULL,
    request_count INT DEFAULT 0,
    daily_count INT DEFAULT 0,
    daily_reset TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for cleanup of old entries
CREATE INDEX IF NOT EXISTS idx_rate_limit_updated
    ON api_rate_limit_state(updated_at);

COMMENT ON TABLE api_rate_limit_state IS 'Rate limit state tracking for database-backed rate limiting';


-- =============================================================================
-- 4. HELPER FUNCTIONS
-- =============================================================================

-- Function to check if an API key is valid and active
CREATE OR REPLACE FUNCTION is_api_key_valid(p_key_hash VARCHAR(64))
RETURNS TABLE (
    is_valid BOOLEAN,
    user_id VARCHAR(50),
    roles TEXT[],
    rate_limit INT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        CASE
            WHEN ak.id IS NULL THEN FALSE
            WHEN NOT ak.is_active THEN FALSE
            WHEN ak.expires_at IS NOT NULL AND ak.expires_at < NOW() THEN FALSE
            ELSE TRUE
        END,
        ak.user_id,
        ak.roles,
        ak.rate_limit_per_minute
    FROM api_keys ak
    WHERE ak.key_hash = p_key_hash;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION is_api_key_valid IS 'Check if API key hash is valid and return associated metadata';


-- Function to update last_used statistics
CREATE OR REPLACE FUNCTION update_api_key_usage(
    p_key_hash VARCHAR(64),
    p_client_ip VARCHAR(45) DEFAULT NULL
)
RETURNS VOID AS $$
BEGIN
    UPDATE api_keys
    SET
        last_used_at = NOW(),
        last_used_ip = COALESCE(p_client_ip, last_used_ip),
        use_count = use_count + 1
    WHERE key_hash = p_key_hash;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_api_key_usage IS 'Update API key last used timestamp and increment usage count';


-- Function to deactivate expired keys
CREATE OR REPLACE FUNCTION deactivate_expired_api_keys()
RETURNS INT AS $$
DECLARE
    affected_count INT;
BEGIN
    UPDATE api_keys
    SET is_active = FALSE
    WHERE expires_at IS NOT NULL
      AND expires_at < NOW()
      AND is_active = TRUE;

    GET DIAGNOSTICS affected_count = ROW_COUNT;
    RETURN affected_count;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION deactivate_expired_api_keys IS 'Deactivate all expired API keys, returns count of affected keys';


-- =============================================================================
-- 5. VIEW: API KEY SUMMARY
-- =============================================================================

CREATE OR REPLACE VIEW v_api_keys_summary AS
SELECT
    id,
    key_prefix,
    name,
    user_id,
    roles,
    rate_limit_per_minute,
    is_active,
    expires_at,
    created_at,
    last_used_at,
    use_count,
    CASE
        WHEN NOT is_active THEN 'inactive'
        WHEN expires_at IS NOT NULL AND expires_at < NOW() THEN 'expired'
        ELSE 'active'
    END AS status
FROM api_keys
ORDER BY created_at DESC;

COMMENT ON VIEW v_api_keys_summary IS 'Summary view of API keys with computed status';


-- =============================================================================
-- 6. INSERT DEFAULT ADMIN KEY (for initial setup)
-- =============================================================================
-- IMPORTANT: Change this key immediately in production!
-- This is just for initial setup - generate a real key using generate_api_key.py

-- The hash below is for key: "usdcop_INITIAL_SETUP_KEY_CHANGE_ME"
-- SHA-256: echo -n "usdcop_INITIAL_SETUP_KEY_CHANGE_ME" | sha256sum
INSERT INTO api_keys (key_hash, key_prefix, name, user_id, roles, rate_limit_per_minute, description)
VALUES (
    '8a9f8e7d6c5b4a3b2c1d0e9f8a7b6c5d4e3f2a1b0c9d8e7f6a5b4c3d2e1f0a9b',  -- Placeholder hash
    'usdcop_INIT...',
    'Initial Admin Key',
    'admin',
    ARRAY['admin', 'trader'],
    1000,
    'Initial setup key - REPLACE IMMEDIATELY IN PRODUCTION'
)
ON CONFLICT (key_hash) DO NOTHING;
