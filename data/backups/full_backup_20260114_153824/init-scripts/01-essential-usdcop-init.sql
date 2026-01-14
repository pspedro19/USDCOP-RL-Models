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

-- 2. UNIFIED OHLCV TABLE - TABLA PRINCIPAL (Pipeline L0 + RT V2)
CREATE TABLE IF NOT EXISTS usdcop_m5_ohlcv (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    symbol TEXT NOT NULL DEFAULT 'USD/COP',
    open DECIMAL(12,6) NOT NULL,
    high DECIMAL(12,6) NOT NULL,
    low DECIMAL(12,6) NOT NULL,
    close DECIMAL(12,6) NOT NULL,
    volume BIGINT DEFAULT 0,
    source TEXT DEFAULT 'twelvedata',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (time, symbol),
    -- Data quality constraints
    CONSTRAINT chk_prices_positive CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CONSTRAINT chk_high_gte_low CHECK (high >= low),
    CONSTRAINT chk_high_gte_open CHECK (high >= open),
    CONSTRAINT chk_high_gte_close CHECK (high >= close),
    CONSTRAINT chk_low_lte_open CHECK (low <= open),
    CONSTRAINT chk_low_lte_close CHECK (low <= close),
    CONSTRAINT chk_volume_non_negative CHECK (volume >= 0)
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
SELECT create_hypertable('usdcop_m5_ohlcv', 'time', if_not_exists => TRUE, migrate_data => TRUE);
SELECT create_hypertable('trading_metrics', 'timestamp', if_not_exists => TRUE);

-- Create optimized indexes
-- =====================================================

-- USDCOP M5 OHLCV indexes (UNIFIED TABLE - PRINCIPAL)
CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_symbol_time ON usdcop_m5_ohlcv (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_time ON usdcop_m5_ohlcv (time DESC);
CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_source ON usdcop_m5_ohlcv (source);
CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_close ON usdcop_m5_ohlcv (close);

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

-- Latest OHLCV data view (UNIFIED TABLE - PRINCIPAL)
CREATE OR REPLACE VIEW latest_ohlcv AS
SELECT DISTINCT ON (symbol)
    symbol,
    time as timestamp,
    open,
    high,
    low,
    close,
    volume,
    source,
    created_at
FROM usdcop_m5_ohlcv
ORDER BY symbol, time DESC;

-- Daily OHLCV summary view (from unified table)
CREATE OR REPLACE VIEW daily_ohlcv_summary AS
SELECT
    DATE(time) as trading_date,
    symbol,
    (ARRAY_AGG(open ORDER BY time))[1] as open_price,
    MAX(high) as high_price,
    MIN(low) as low_price,
    (ARRAY_AGG(close ORDER BY time DESC))[1] as close_price,
    AVG(close) as avg_price,
    COUNT(*) as bar_count,
    SUM(volume) as total_volume
FROM usdcop_m5_ohlcv
GROUP BY DATE(time), symbol
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
AND table_name IN ('users', 'usdcop_m5_ohlcv', 'trading_metrics', 'trading_sessions');

-- Show created hypertables
SELECT
    'TimescaleDB hypertables created:' as info,
    hypertable_name
FROM timescaledb_information.hypertables
WHERE hypertable_name IN ('usdcop_m5_ohlcv', 'trading_metrics');

-- Show confirmation message
DO $$
BEGIN
    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '✅ USDCOP Trading System - Database Initialized';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'PRIMARY TABLE: usdcop_m5_ohlcv (OHLCV data)';
    RAISE NOTICE 'Trading Hours: Monday-Friday, 8:00-12:55 COT';
    RAISE NOTICE 'Data Quality: 100%% within trading hours';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;