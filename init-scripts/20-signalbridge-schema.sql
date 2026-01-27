-- ============================================================================
-- SignalBridge Schema Migration
-- ============================================================================
-- Purpose: Create SignalBridge tables that work with existing users table (integer IDs)
-- Date: 2026-01-22
-- ============================================================================

-- Exchange credentials table
CREATE TABLE IF NOT EXISTS exchange_credentials (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    exchange VARCHAR(50) NOT NULL,
    label VARCHAR(100) NOT NULL,
    encrypted_api_key TEXT NOT NULL,
    encrypted_api_secret TEXT NOT NULL,
    encrypted_passphrase TEXT,
    key_version VARCHAR(50) NOT NULL DEFAULT 'v1',
    is_testnet BOOLEAN NOT NULL DEFAULT false,
    is_active BOOLEAN NOT NULL DEFAULT true,
    is_valid BOOLEAN NOT NULL DEFAULT false,
    last_used TIMESTAMP WITH TIME ZONE,
    last_validated TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_exchange_credentials_user ON exchange_credentials(user_id);
CREATE INDEX IF NOT EXISTS idx_exchange_credentials_exchange ON exchange_credentials(exchange);

-- Trading config table
CREATE TABLE IF NOT EXISTS trading_configs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    trading_mode VARCHAR(20) NOT NULL DEFAULT 'SHADOW',
    max_position_size_usd NUMERIC(10,2) NOT NULL DEFAULT 100.00,
    max_daily_loss_pct NUMERIC(5,2) NOT NULL DEFAULT 2.00,
    max_trades_per_day INTEGER NOT NULL DEFAULT 10,
    cooldown_minutes INTEGER NOT NULL DEFAULT 15,
    enable_short BOOLEAN NOT NULL DEFAULT false,
    kill_switch_active BOOLEAN NOT NULL DEFAULT false,
    kill_switch_reason TEXT,
    kill_switch_activated_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_trading_configs_user ON trading_configs(user_id);

-- Signals table
CREATE TABLE IF NOT EXISTS signals (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    symbol VARCHAR(20) NOT NULL,
    action INTEGER NOT NULL,
    price NUMERIC(20,8),
    quantity NUMERIC(20,8),
    stop_loss NUMERIC(20,8),
    take_profit NUMERIC(20,8),
    source VARCHAR(50) NOT NULL DEFAULT 'api',
    signal_metadata JSONB NOT NULL DEFAULT '{}',
    is_processed BOOLEAN NOT NULL DEFAULT false,
    processed_at TIMESTAMP WITH TIME ZONE,
    execution_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_signals_user ON signals(user_id);
CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
CREATE INDEX IF NOT EXISTS idx_signals_processed ON signals(is_processed);

-- Executions table
CREATE TABLE IF NOT EXISTS executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    signal_id UUID,
    exchange VARCHAR(50) NOT NULL,
    credential_id UUID REFERENCES exchange_credentials(id) ON DELETE SET NULL,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    quantity NUMERIC(20,8) NOT NULL,
    price NUMERIC(20,8),
    stop_loss NUMERIC(20,8),
    take_profit NUMERIC(20,8),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    exchange_order_id VARCHAR(100),
    filled_quantity NUMERIC(20,8) NOT NULL DEFAULT 0,
    average_price NUMERIC(20,8) NOT NULL DEFAULT 0,
    commission NUMERIC(20,8) NOT NULL DEFAULT 0,
    commission_asset VARCHAR(20),
    executed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    raw_response JSONB,
    execution_metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_executions_user ON executions(user_id);
CREATE INDEX IF NOT EXISTS idx_executions_symbol ON executions(symbol);
CREATE INDEX IF NOT EXISTS idx_executions_status ON executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_exchange ON executions(exchange);

-- Add foreign key from signals to executions
ALTER TABLE signals
    DROP CONSTRAINT IF EXISTS signals_execution_id_fkey;
ALTER TABLE signals
    ADD CONSTRAINT signals_execution_id_fkey
    FOREIGN KEY (execution_id) REFERENCES executions(id) ON DELETE SET NULL;

-- User risk limits table
CREATE TABLE IF NOT EXISTS user_risk_limits (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    max_daily_loss_pct NUMERIC(5,2) NOT NULL DEFAULT 2.00,
    max_trades_per_day INTEGER NOT NULL DEFAULT 10,
    max_position_size_usd NUMERIC(10,2) NOT NULL DEFAULT 1000.00,
    cooldown_minutes INTEGER NOT NULL DEFAULT 15,
    enable_short BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_user_risk_limits_user ON user_risk_limits(user_id);

-- Execution audit events table
CREATE TABLE IF NOT EXISTS execution_audit_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES executions(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_execution_audit_execution ON execution_audit_events(execution_id);
CREATE INDEX IF NOT EXISTS idx_execution_audit_type ON execution_audit_events(event_type);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO admin;

COMMENT ON TABLE exchange_credentials IS 'SignalBridge: Encrypted exchange API credentials';
COMMENT ON TABLE trading_configs IS 'SignalBridge: User trading configuration and risk settings';
COMMENT ON TABLE signals IS 'SignalBridge: Trading signals from inference models';
COMMENT ON TABLE executions IS 'SignalBridge: Order executions on exchanges';
COMMENT ON TABLE user_risk_limits IS 'SignalBridge: User-specific risk limits';
COMMENT ON TABLE execution_audit_events IS 'SignalBridge: Audit trail for execution events';
