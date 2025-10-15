-- =====================================================
-- USDCOP Trading System - Essential Database Schema
-- Clean, production-ready initialization script
-- =====================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create essential USDCOP trading tables
-- =====================================================

-- 1. Users table for authentication and user management
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(50),
    last_name VARCHAR(50),
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

-- 2. Market data table for USDCOP price data (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS market_data (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'USDCOP',
    price DECIMAL(12,4) NOT NULL,
    bid DECIMAL(12,4),
    ask DECIMAL(12,4),
    volume BIGINT,
    source TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, symbol)
);

-- 3. Trading metrics table for performance and results (TimescaleDB hypertable)
CREATE TABLE IF NOT EXISTS trading_metrics (
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DECIMAL(15,6),
    metric_type TEXT NOT NULL, -- 'performance', 'risk', 'model_accuracy', etc.
    strategy_name TEXT,
    model_version TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (timestamp, metric_name, metric_type)
);

-- 4. Trading sessions table for tracking active sessions
CREATE TABLE IF NOT EXISTS trading_sessions (
    id BIGSERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    session_start TIMESTAMP WITH TIME ZONE NOT NULL,
    session_end TIMESTAMP WITH TIME ZONE,
    strategy_name TEXT,
    initial_balance DECIMAL(15,2),
    final_balance DECIMAL(15,2),
    total_trades INTEGER DEFAULT 0,
    profitable_trades INTEGER DEFAULT 0,
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'completed', 'stopped'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Convert tables to TimescaleDB hypertables
-- =====================================================
SELECT create_hypertable('market_data', 'timestamp', if_not_exists => TRUE);
SELECT create_hypertable('trading_metrics', 'timestamp', if_not_exists => TRUE);

-- Create optimized indexes
-- =====================================================

-- Market data indexes
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_source ON market_data (source);
CREATE INDEX IF NOT EXISTS idx_market_data_price ON market_data (price);

-- Trading metrics indexes
CREATE INDEX IF NOT EXISTS idx_trading_metrics_name_time ON trading_metrics (metric_name, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_type ON trading_metrics (metric_type);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_strategy ON trading_metrics (strategy_name);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_model ON trading_metrics (model_version);

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_username ON users (username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users (email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users (is_active) WHERE is_active = true;

-- Trading sessions indexes
CREATE INDEX IF NOT EXISTS idx_trading_sessions_user ON trading_sessions (user_id);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_start ON trading_sessions (session_start);
CREATE INDEX IF NOT EXISTS idx_trading_sessions_status ON trading_sessions (status) WHERE status = 'active';

-- Create useful views for data access
-- =====================================================

-- Latest market data view
CREATE OR REPLACE VIEW latest_market_data AS
SELECT DISTINCT ON (symbol)
    symbol,
    timestamp,
    price,
    bid,
    ask,
    volume,
    source,
    created_at
FROM market_data
ORDER BY symbol, timestamp DESC;

-- Daily trading summary view
CREATE OR REPLACE VIEW daily_market_summary AS
SELECT
    DATE(timestamp) as trading_date,
    symbol,
    MIN(price) as low_price,
    MAX(price) as high_price,
    (ARRAY_AGG(price ORDER BY timestamp))[1] as open_price,
    (ARRAY_AGG(price ORDER BY timestamp DESC))[1] as close_price,
    AVG(price) as avg_price,
    COUNT(*) as tick_count,
    SUM(volume) as total_volume
FROM market_data
GROUP BY DATE(timestamp), symbol
ORDER BY trading_date DESC;

-- Performance metrics summary view
CREATE OR REPLACE VIEW metrics_summary AS
SELECT
    strategy_name,
    model_version,
    DATE(timestamp) as date,
    COUNT(*) as metric_count,
    AVG(CASE WHEN metric_type = 'performance' THEN metric_value END) as avg_performance,
    AVG(CASE WHEN metric_type = 'risk' THEN metric_value END) as avg_risk,
    AVG(CASE WHEN metric_type = 'model_accuracy' THEN metric_value END) as avg_accuracy
FROM trading_metrics
WHERE strategy_name IS NOT NULL
GROUP BY strategy_name, model_version, DATE(timestamp)
ORDER BY date DESC;

-- Active trading sessions view
CREATE OR REPLACE VIEW active_sessions AS
SELECT
    s.id,
    u.username,
    s.session_start,
    s.strategy_name,
    s.initial_balance,
    s.total_trades,
    s.profitable_trades,
    CASE
        WHEN s.total_trades > 0 THEN ROUND((s.profitable_trades::DECIMAL / s.total_trades * 100), 2)
        ELSE 0
    END as win_rate_percent
FROM trading_sessions s
JOIN users u ON s.user_id = u.id
WHERE s.status = 'active'
ORDER BY s.session_start DESC;

-- Insert default users
-- =====================================================
INSERT INTO users (username, email, password_hash, first_name, last_name, is_admin)
VALUES
    ('admin', 'admin@usdcop-trading.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LdMpRGP7h4v8E9qTC', 'Admin', 'User', true),
    ('trader1', 'trader1@usdcop-trading.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LdMpRGP7h4v8E9qTC', 'Trader', 'One', false)
ON CONFLICT (username) DO NOTHING;

-- Set up proper permissions
-- =====================================================
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO admin;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO admin;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO airflow;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO airflow;

-- Final verification and success message
-- =====================================================
SELECT
    'USDCOP Trading Database initialized successfully!' as status,
    COUNT(*) as essential_tables_created
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_name IN ('users', 'market_data', 'trading_metrics', 'trading_sessions');

-- Show created hypertables
SELECT
    'TimescaleDB hypertables created:' as info,
    hypertable_name
FROM timescaledb_information.hypertables
WHERE hypertable_name IN ('market_data', 'trading_metrics');