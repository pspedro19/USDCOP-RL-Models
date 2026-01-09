-- ============================================================================
-- Migration: 001_create_users_table
-- Description: Create users and sessions tables for authentication
-- Date: 2024-12-17
-- ============================================================================

-- Create extension for UUID generation if not exists
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Users Table
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'trader',

    -- Profile info
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),

    -- Security
    email_verified BOOLEAN DEFAULT FALSE,
    email_verified_at TIMESTAMPTZ,
    two_factor_enabled BOOLEAN DEFAULT FALSE,
    two_factor_secret VARCHAR(255),

    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    locked_until TIMESTAMPTZ,
    failed_login_attempts INT DEFAULT 0,
    last_failed_login TIMESTAMPTZ,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    last_login_at TIMESTAMPTZ,

    -- Constraints
    CONSTRAINT valid_role CHECK (role IN ('admin', 'trader', 'viewer', 'api_service'))
);

-- ============================================================================
-- User Sessions Table (for NextAuth)
-- ============================================================================
CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_token VARCHAR(255) UNIQUE NOT NULL,

    -- Session info
    ip_address INET,
    user_agent TEXT,
    device_type VARCHAR(50),

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,
    last_activity_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- API Keys Table (for service-to-service auth)
-- ============================================================================
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,

    -- Key info
    key_hash VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(10) NOT NULL, -- First chars for identification
    name VARCHAR(100) NOT NULL,
    description TEXT,

    -- Permissions
    scopes TEXT[] DEFAULT ARRAY['read'],

    -- Rate limiting
    rate_limit_per_minute INT DEFAULT 60,
    rate_limit_per_day INT DEFAULT 10000,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ
);

-- ============================================================================
-- Login Attempts Table (for rate limiting and security)
-- ============================================================================
CREATE TABLE IF NOT EXISTS login_attempts (
    id SERIAL PRIMARY KEY,
    identifier VARCHAR(255) NOT NULL, -- email or IP
    ip_address INET NOT NULL,
    user_agent TEXT,

    -- Result
    success BOOLEAN NOT NULL,
    failure_reason VARCHAR(100),

    -- Timestamp
    attempted_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Audit Log Table (for security events)
-- ============================================================================
CREATE TABLE IF NOT EXISTS auth_audit_log (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,

    -- Event info
    event_type VARCHAR(50) NOT NULL,
    event_description TEXT,

    -- Context
    ip_address INET,
    user_agent TEXT,
    metadata JSONB,

    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- Indexes
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON user_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_expires ON user_sessions(expires_at);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);

CREATE INDEX IF NOT EXISTS idx_login_attempts_identifier ON login_attempts(identifier);
CREATE INDEX IF NOT EXISTS idx_login_attempts_ip ON login_attempts(ip_address);
CREATE INDEX IF NOT EXISTS idx_login_attempts_time ON login_attempts(attempted_at);

CREATE INDEX IF NOT EXISTS idx_audit_user_id ON auth_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_event_type ON auth_audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_created ON auth_audit_log(created_at);

-- ============================================================================
-- Triggers for updated_at
-- ============================================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- Default Admin User (password: Admin@2024! - CHANGE IN PRODUCTION)
-- Password hash generated with bcrypt, cost factor 12
-- ============================================================================
INSERT INTO users (
    email,
    username,
    password_hash,
    role,
    full_name,
    email_verified,
    is_active
) VALUES (
    'admin@usdcop-trading.local',
    'admin',
    -- bcrypt hash for 'Admin@2024!'
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.G9HJQqYjW1qKHe',
    'admin',
    'System Administrator',
    TRUE,
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- Default Trader User (password: Trader@2024! - CHANGE IN PRODUCTION)
-- ============================================================================
INSERT INTO users (
    email,
    username,
    password_hash,
    role,
    full_name,
    email_verified,
    is_active
) VALUES (
    'trader@usdcop-trading.local',
    'trader',
    -- bcrypt hash for 'Trader@2024!'
    '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi',
    'trader',
    'Default Trader',
    TRUE,
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- ============================================================================
-- Cleanup old login attempts (older than 30 days)
-- ============================================================================
CREATE OR REPLACE FUNCTION cleanup_old_login_attempts()
RETURNS void AS $$
BEGIN
    DELETE FROM login_attempts WHERE attempted_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Function to check if user is locked out
-- ============================================================================
CREATE OR REPLACE FUNCTION is_user_locked(p_user_id UUID)
RETURNS BOOLEAN AS $$
DECLARE
    v_locked_until TIMESTAMPTZ;
BEGIN
    SELECT locked_until INTO v_locked_until FROM users WHERE id = p_user_id;
    RETURN v_locked_until IS NOT NULL AND v_locked_until > NOW();
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Function to record failed login and potentially lock account
-- ============================================================================
CREATE OR REPLACE FUNCTION record_failed_login(p_user_id UUID)
RETURNS void AS $$
DECLARE
    v_attempts INT;
BEGIN
    UPDATE users
    SET
        failed_login_attempts = failed_login_attempts + 1,
        last_failed_login = NOW()
    WHERE id = p_user_id
    RETURNING failed_login_attempts INTO v_attempts;

    -- Lock account after 5 failed attempts for 15 minutes
    IF v_attempts >= 5 THEN
        UPDATE users
        SET locked_until = NOW() + INTERVAL '15 minutes'
        WHERE id = p_user_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Function to record successful login
-- ============================================================================
CREATE OR REPLACE FUNCTION record_successful_login(p_user_id UUID)
RETURNS void AS $$
BEGIN
    UPDATE users
    SET
        failed_login_attempts = 0,
        locked_until = NULL,
        last_login_at = NOW()
    WHERE id = p_user_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON TABLE users IS 'User accounts for authentication';
COMMENT ON TABLE user_sessions IS 'Active user sessions';
COMMENT ON TABLE api_keys IS 'API keys for service authentication';
COMMENT ON TABLE login_attempts IS 'Login attempt history for security monitoring';
COMMENT ON TABLE auth_audit_log IS 'Security audit trail for authentication events';
