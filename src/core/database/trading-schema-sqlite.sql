-- ═══════════════════════════════════════════════════════════════════════════════
-- Trading System Database Schema (SQLite Version)
-- Creates all necessary tables and indexes for SQLite
-- ═══════════════════════════════════════════════════════════════════════════════

-- ═══════════════════════════════════════════════════════════════════════════════
-- MARKET DATA SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- OHLCV data table
CREATE TABLE IF NOT EXISTS market_data_ohlcv (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    tick_volume INTEGER,
    spread INTEGER,
    created_at TEXT DEFAULT (datetime('now'))
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timestamp 
    ON market_data_ohlcv(symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_ohlcv_timeframe 
    ON market_data_ohlcv(timeframe);

-- Technical indicators table
CREATE TABLE IF NOT EXISTS market_data_indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    indicator_name TEXT NOT NULL,
    indicator_params TEXT,  -- JSON as text
    value REAL NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, timeframe, timestamp, indicator_name, indicator_params)
);

CREATE INDEX IF NOT EXISTS idx_indicators_lookup 
    ON market_data_indicators(symbol, timeframe, timestamp, indicator_name);

-- Market events table
CREATE TABLE IF NOT EXISTS market_data_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    symbol TEXT,
    timestamp TEXT NOT NULL,
    impact TEXT,
    description TEXT,
    data TEXT,  -- JSON as text
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON market_data_events(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_events_symbol ON market_data_events(symbol);

-- ═══════════════════════════════════════════════════════════════════════════════
-- TRADING SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Trades table
CREATE TABLE IF NOT EXISTS trading_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT UNIQUE NOT NULL,  -- UUID as text
    symbol TEXT NOT NULL,
    order_type TEXT NOT NULL,  -- 'BUY', 'SELL'
    order_side TEXT NOT NULL,  -- 'LONG', 'SHORT'
    volume REAL NOT NULL,
    price REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    commission REAL DEFAULT 0.0,
    slippage REAL DEFAULT 0.0,
    status TEXT NOT NULL,  -- 'OPEN', 'CLOSED', 'CANCELLED'
    open_time TEXT NOT NULL,
    close_time TEXT,
    pnl REAL,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trading_trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trading_trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trading_trades(open_time);

-- Orders table
CREATE TABLE IF NOT EXISTS trading_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT UNIQUE NOT NULL,
    trade_id TEXT,
    symbol TEXT NOT NULL,
    order_type TEXT NOT NULL,  -- 'MARKET', 'LIMIT', 'STOP'
    order_side TEXT NOT NULL,  -- 'BUY', 'SELL'
    volume REAL NOT NULL,
    price REAL,
    stop_loss REAL,
    take_profit REAL,
    status TEXT NOT NULL,  -- 'PENDING', 'FILLED', 'CANCELLED'
    created_at TEXT DEFAULT (datetime('now')),
    filled_at TEXT,
    FOREIGN KEY (trade_id) REFERENCES trading_trades(trade_id)
);

CREATE INDEX IF NOT EXISTS idx_orders_symbol ON trading_orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_status ON trading_orders(status);

-- Positions table
CREATE TABLE IF NOT EXISTS trading_positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,  -- 'LONG', 'SHORT'
    volume REAL NOT NULL,
    avg_price REAL NOT NULL,
    unrealized_pnl REAL DEFAULT 0.0,
    realized_pnl REAL DEFAULT 0.0,
    stop_loss REAL,
    take_profit REAL,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, side)
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol ON trading_positions(symbol);

-- ═══════════════════════════════════════════════════════════════════════════════
-- ML MODELS SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Model metadata table
CREATE TABLE IF NOT EXISTS ml_models_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    model_type TEXT NOT NULL,  -- 'PPO', 'DQN', 'CUSTOM'
    version TEXT NOT NULL,
    file_path TEXT NOT NULL,
    hyperparameters TEXT,  -- JSON as text
    training_metrics TEXT,  -- JSON as text
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_models_name ON ml_models_metadata(model_name);
CREATE INDEX IF NOT EXISTS idx_models_type ON ml_models_metadata(model_type);

-- Model predictions table
CREATE TABLE IF NOT EXISTS ml_models_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    symbol TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    prediction TEXT NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
    confidence REAL NOT NULL,
    features TEXT,  -- JSON as text
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (model_id) REFERENCES ml_models_metadata(id)
);

CREATE INDEX IF NOT EXISTS idx_predictions_model ON ml_models_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol ON ml_models_predictions(symbol);
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON ml_models_predictions(timestamp);

-- ═══════════════════════════════════════════════════════════════════════════════
-- ANALYTICS SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- Performance metrics table
CREATE TABLE IF NOT EXISTS analytics_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    date TEXT NOT NULL,
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate REAL DEFAULT 0.0,
    total_pnl REAL DEFAULT 0.0,
    max_drawdown REAL DEFAULT 0.0,
    sharpe_ratio REAL DEFAULT 0.0,
    sortino_ratio REAL DEFAULT 0.0,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, timeframe, date)
);

CREATE INDEX IF NOT EXISTS idx_performance_symbol ON analytics_performance(symbol);
CREATE INDEX IF NOT EXISTS idx_performance_date ON analytics_performance(date);

-- Data quality metrics table
CREATE TABLE IF NOT EXISTS analytics_data_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    date TEXT NOT NULL,
    completeness REAL DEFAULT 0.0,
    consistency REAL DEFAULT 0.0,
    integrity REAL DEFAULT 0.0,
    timeliness REAL DEFAULT 0.0,
    accuracy REAL DEFAULT 0.0,
    overall_score REAL DEFAULT 0.0,
    issues TEXT,  -- JSON as text
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(symbol, timeframe, date)
);

CREATE INDEX IF NOT EXISTS idx_quality_symbol ON analytics_data_quality(symbol);
CREATE INDEX IF NOT EXISTS idx_quality_date ON analytics_data_quality(date);

-- ═══════════════════════════════════════════════════════════════════════════════
-- SYSTEM SCHEMA
-- ═══════════════════════════════════════════════════════════════════════════════

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- Insert default configuration
INSERT OR REPLACE INTO system_config (key, value, description) VALUES
    ('database_version', '1.0.0', 'Current database schema version'),
    ('created_at', datetime('now'), 'Database creation timestamp'),
    ('last_updated', datetime('now'), 'Last update timestamp');

-- System logs table
CREATE TABLE IF NOT EXISTS system_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,  -- 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    logger TEXT NOT NULL,
    message TEXT NOT NULL,
    timestamp TEXT DEFAULT (datetime('now')),
    extra_data TEXT  -- JSON as text
);

CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_logger ON system_logs(logger);
