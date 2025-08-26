-- ═══════════════════════════════════════════════════════════════════════════════
-- Trading System Database Schema
-- Creates all necessary tables, indexes, and partitions
-- ═══════════════════════════════════════════════════════════════════════════════

-- Connect to trading database
\c trading_db

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS analytics;

-- ═══════════════════════════════════════════════════════════════════════════════
-- MARKET DATA SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- OHLCV data table (partitioned by timeframe)
CREATE TABLE IF NOT EXISTS market_data.ohlcv (
    id BIGSERIAL,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    tick_volume BIGINT,
    spread INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT ohlcv_pkey PRIMARY KEY (symbol, timeframe, timestamp)
) PARTITION BY LIST (timeframe);

-- Create partitions for different timeframes
CREATE TABLE IF NOT EXISTS market_data.ohlcv_m1 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('M1');

CREATE TABLE IF NOT EXISTS market_data.ohlcv_m5 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('M5');

CREATE TABLE IF NOT EXISTS market_data.ohlcv_m15 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('M15');

CREATE TABLE IF NOT EXISTS market_data.ohlcv_m30 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('M30');

CREATE TABLE IF NOT EXISTS market_data.ohlcv_h1 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('H1');

CREATE TABLE IF NOT EXISTS market_data.ohlcv_h4 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('H4');

CREATE TABLE IF NOT EXISTS market_data.ohlcv_d1 PARTITION OF market_data.ohlcv
    FOR VALUES IN ('D1');

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp 
    ON market_data.ohlcv(symbol, timestamp DESC);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS market_data.indicators (
    id BIGSERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    indicator_name VARCHAR(50) NOT NULL,
    indicator_params JSONB,
    value DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(symbol, timeframe, timestamp, indicator_name, indicator_params)
);

CREATE INDEX IF NOT EXISTS idx_indicators_lookup 
    ON market_data.indicators(symbol, timeframe, timestamp, indicator_name);

-- Market events table
CREATE TABLE IF NOT EXISTS market_data.events (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    symbol VARCHAR(20),
    timestamp TIMESTAMPTZ NOT NULL,
    impact VARCHAR(20),
    description TEXT,
    data JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON market_data.events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON market_data.events(symbol);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TRADING SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Trades table
CREATE TABLE IF NOT EXISTS trading.trades (
    id BIGSERIAL PRIMARY KEY,
    trade_id UUID DEFAULT uuid_generate_v4(),
    mt5_ticket BIGINT,
    symbol VARCHAR(20) NOT NULL,
    trade_type VARCHAR(10) NOT NULL CHECK (trade_type IN ('BUY', 'SELL')),
    volume DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    entry_time TIMESTAMPTZ NOT NULL,
    exit_price DECIMAL(20, 8),
    exit_time TIMESTAMPTZ,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    commission DECIMAL(20, 8) DEFAULT 0,
    swap DECIMAL(20, 8) DEFAULT 0,
    profit DECIMAL(20, 8),
    profit_pips DECIMAL(10, 2),
    status VARCHAR(20) NOT NULL DEFAULT 'OPEN',
    close_reason VARCHAR(50),
    strategy_name VARCHAR(100),
    strategy_version VARCHAR(20),
    signal_data JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_status ON trading.trades(symbol, status);
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trading.trades(entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trading.trades(strategy_name, strategy_version);

-- Orders table (pending orders)
CREATE TABLE IF NOT EXISTS trading.orders (
    id BIGSERIAL PRIMARY KEY,
    order_id UUID DEFAULT uuid_generate_v4(),
    mt5_ticket BIGINT,
    symbol VARCHAR(20) NOT NULL,
    order_type VARCHAR(20) NOT NULL,
    volume DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8) NOT NULL,
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    expiration TIMESTAMPTZ,
    status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
    strategy_name VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Trading signals table
CREATE TABLE IF NOT EXISTS trading.signals (
    id BIGSERIAL PRIMARY KEY,
    signal_id UUID DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(20) NOT NULL,
    strength DECIMAL(5, 2),
    entry_price DECIMAL(20, 8),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    strategy_name VARCHAR(100),
    strategy_version VARCHAR(20),
    indicators JSONB,
    metadata JSONB,
    executed BOOLEAN DEFAULT FALSE,
    trade_id UUID,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
    ON trading.signals(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_executed 
    ON trading.signals(executed) WHERE executed = FALSE;

-- Account history table
CREATE TABLE IF NOT EXISTS trading.account_history (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    balance DECIMAL(20, 8) NOT NULL,
    equity DECIMAL(20, 8) NOT NULL,
    margin DECIMAL(20, 8),
    free_margin DECIMAL(20, 8),
    margin_level DECIMAL(10, 2),
    open_positions INTEGER DEFAULT 0,
    total_profit DECIMAL(20, 8),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(timestamp)
);

CREATE INDEX IF NOT EXISTS idx_account_history_timestamp 
    ON trading.account_history(timestamp DESC);

-- ═══════════════════════════════════════════════════════════════════════════════
-- ML MODELS SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Models registry
CREATE TABLE IF NOT EXISTS ml_models.models (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    framework VARCHAR(50),
    parameters JSONB,
    hyperparameters JSONB,
    training_config JSONB,
    file_path VARCHAR(500),
    minio_bucket VARCHAR(100),
    minio_key VARCHAR(500),
    status VARCHAR(20) NOT NULL DEFAULT 'TRAINING',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(name, version)
);

-- Model training history
CREATE TABLE IF NOT EXISTS ml_models.training_history (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL,
    epoch INTEGER,
    batch INTEGER,
    loss DECIMAL(20, 8),
    metrics JSONB,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_history_model 
    ON ml_models.training_history(model_id, timestamp);

-- Model predictions
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    prediction_type VARCHAR(50),
    prediction_value DECIMAL(20, 8),
    confidence DECIMAL(5, 4),
    features JSONB,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_model_symbol 
    ON ml_models.predictions(model_id, symbol, timestamp DESC);

-- Model performance metrics
CREATE TABLE IF NOT EXISTS ml_models.performance_metrics (
    id BIGSERIAL PRIMARY KEY,
    model_id UUID NOT NULL,
    evaluation_date DATE NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(20, 8) NOT NULL,
    dataset VARCHAR(50),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_id, evaluation_date, metric_name, dataset)
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- ANALYTICS SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Daily performance summary
CREATE TABLE IF NOT EXISTS analytics.daily_performance (
    id BIGSERIAL PRIMARY KEY,
    date DATE NOT NULL,
    symbol VARCHAR(20),
    strategy_name VARCHAR(100),
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    total_volume DECIMAL(20, 8),
    gross_profit DECIMAL(20, 8),
    gross_loss DECIMAL(20, 8),
    net_profit DECIMAL(20, 8),
    profit_factor DECIMAL(10, 4),
    win_rate DECIMAL(5, 2),
    avg_win DECIMAL(20, 8),
    avg_loss DECIMAL(20, 8),
    largest_win DECIMAL(20, 8),
    largest_loss DECIMAL(20, 8),
    max_drawdown DECIMAL(10, 4),
    sharpe_ratio DECIMAL(10, 4),
    sortino_ratio DECIMAL(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(date, symbol, strategy_name)
);

CREATE INDEX IF NOT EXISTS idx_daily_performance_date 
    ON analytics.daily_performance(date DESC);

-- Risk metrics table
CREATE TABLE IF NOT EXISTS analytics.risk_metrics (
    id BIGSERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    timeframe VARCHAR(20) NOT NULL,
    value DECIMAL(20, 8) NOT NULL,
    symbol VARCHAR(20),
    strategy_name VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ═══════════════════════════════════════════════════════════════════════════════
-- FUNCTIONS AND TRIGGERS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply update trigger to relevant tables
CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trading.trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON trading.orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_models_updated_at BEFORE UPDATE ON ml_models.models
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ═══════════════════════════════════════════════════════════════════════════════
-- PERMISSIONS
-- ═══════════════════════════════════════════════════════════════════════════════

-- Grant permissions to trading_user
GRANT ALL PRIVILEGES ON SCHEMA market_data TO trading_user;
GRANT ALL PRIVILEGES ON SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON SCHEMA ml_models TO trading_user;
GRANT ALL PRIVILEGES ON SCHEMA analytics TO trading_user;

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA market_data TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO trading_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO trading_user;

GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA market_data TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO trading_user;

-- Grant read-only permissions to analytics_user
GRANT USAGE ON SCHEMA market_data TO analytics_user;
GRANT USAGE ON SCHEMA trading TO analytics_user;
GRANT USAGE ON SCHEMA ml_models TO analytics_user;
GRANT USAGE ON SCHEMA analytics TO analytics_user;

GRANT SELECT ON ALL TABLES IN SCHEMA market_data TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA trading TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA ml_models TO analytics_user;
GRANT SELECT ON ALL TABLES IN SCHEMA analytics TO analytics_user;