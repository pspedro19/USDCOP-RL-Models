-- =====================================================
-- Trading System Schema
-- =====================================================

\c trading_db

-- Market Data Tables (Bronze Layer)
CREATE TABLE IF NOT EXISTS bronze.market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8),
    tick_volume BIGINT,
    spread DECIMAL(10, 5),
    source VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_market_data_symbol_time ON bronze.market_data(symbol, timestamp);
CREATE INDEX idx_market_data_timeframe ON bronze.market_data(timeframe);

-- Cleaned Data Tables (Silver Layer)
CREATE TABLE IF NOT EXISTS silver.cleaned_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    open DECIMAL(20, 8) NOT NULL,
    high DECIMAL(20, 8) NOT NULL,
    low DECIMAL(20, 8) NOT NULL,
    close DECIMAL(20, 8) NOT NULL,
    volume DECIMAL(20, 8),
    returns DECIMAL(10, 6),
    log_returns DECIMAL(10, 6),
    volatility DECIMAL(10, 6),
    gaps_filled INTEGER DEFAULT 0,
    outliers_removed INTEGER DEFAULT 0,
    quality_score DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_cleaned_data_symbol_time ON silver.cleaned_data(symbol, timestamp);

-- Feature Data Tables (Gold Layer)
CREATE TABLE IF NOT EXISTS gold.features (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    -- Price features
    close DECIMAL(20, 8) NOT NULL,
    returns DECIMAL(10, 6),
    log_returns DECIMAL(10, 6),
    -- Technical indicators
    sma_10 DECIMAL(20, 8),
    sma_20 DECIMAL(20, 8),
    sma_50 DECIMAL(20, 8),
    ema_10 DECIMAL(20, 8),
    ema_20 DECIMAL(20, 8),
    rsi_14 DECIMAL(10, 4),
    macd DECIMAL(10, 6),
    macd_signal DECIMAL(10, 6),
    macd_hist DECIMAL(10, 6),
    bb_upper DECIMAL(20, 8),
    bb_middle DECIMAL(20, 8),
    bb_lower DECIMAL(20, 8),
    atr_14 DECIMAL(10, 6),
    adx_14 DECIMAL(10, 4),
    stoch_k DECIMAL(10, 4),
    stoch_d DECIMAL(10, 4),
    -- Volume indicators
    obv DECIMAL(20, 2),
    volume_sma DECIMAL(20, 2),
    -- Market microstructure
    spread DECIMAL(10, 5),
    bid_ask_imbalance DECIMAL(10, 6),
    -- Labels for ML
    target_return DECIMAL(10, 6),
    target_direction INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, timeframe, timestamp)
);

CREATE INDEX idx_features_symbol_time ON gold.features(symbol, timestamp);

-- Trading Signals Table
CREATE TABLE IF NOT EXISTS gold.signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    signal_type VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    confidence DECIMAL(5, 4),
    predicted_return DECIMAL(10, 6),
    position_size DECIMAL(10, 4),
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    risk_score DECIMAL(5, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_timestamp ON gold.signals(timestamp);
CREATE INDEX idx_signals_symbol ON gold.signals(symbol);

-- Trades Table
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    trade_id VARCHAR(100) UNIQUE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    signal_id INTEGER REFERENCES gold.signals(id),
    direction VARCHAR(10) NOT NULL, -- BUY, SELL
    entry_time TIMESTAMP NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    position_size DECIMAL(20, 8) NOT NULL,
    exit_time TIMESTAMP,
    exit_price DECIMAL(20, 8),
    pnl DECIMAL(20, 8),
    pnl_percentage DECIMAL(10, 4),
    commission DECIMAL(10, 4),
    slippage DECIMAL(10, 4),
    status VARCHAR(20) NOT NULL, -- OPEN, CLOSED, CANCELLED
    stop_loss DECIMAL(20, 8),
    take_profit DECIMAL(20, 8),
    execution_mode VARCHAR(20), -- LIVE, PAPER, BACKTEST
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_trades_status ON trades(status);
CREATE INDEX idx_trades_entry_time ON trades(entry_time);

-- Model Registry Table
CREATE TABLE IF NOT EXISTS models.registry (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    algorithm VARCHAR(50),
    mlflow_run_id VARCHAR(100),
    mlflow_model_uri TEXT,
    training_date TIMESTAMP NOT NULL,
    training_samples INTEGER,
    validation_score DECIMAL(10, 6),
    test_score DECIMAL(10, 6),
    sharpe_ratio DECIMAL(10, 4),
    max_drawdown DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    parameters JSONB,
    metrics JSONB,
    status VARCHAR(20), -- STAGING, PRODUCTION, ARCHIVED
    deployed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(model_name, model_version)
);

CREATE INDEX idx_models_status ON models.registry(status);
CREATE INDEX idx_models_name ON models.registry(model_name);

-- Performance Metrics Table
CREATE TABLE IF NOT EXISTS monitoring.performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 8),
    symbol VARCHAR(20),
    timeframe VARCHAR(10),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_timestamp ON monitoring.performance_metrics(timestamp);
CREATE INDEX idx_metrics_type ON monitoring.performance_metrics(metric_type);

-- System Health Table
CREATE TABLE IF NOT EXISTS monitoring.system_health (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    component VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL, -- HEALTHY, DEGRADED, UNHEALTHY
    cpu_usage DECIMAL(5, 2),
    memory_usage DECIMAL(5, 2),
    disk_usage DECIMAL(5, 2),
    latency_ms INTEGER,
    error_count INTEGER,
    details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_health_timestamp ON monitoring.system_health(timestamp);
CREATE INDEX idx_health_component ON monitoring.system_health(component);

-- Pipeline Runs Table
CREATE TABLE IF NOT EXISTS monitoring.pipeline_runs (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    dag_name VARCHAR(100) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status VARCHAR(20) NOT NULL, -- RUNNING, SUCCESS, FAILED
    total_steps INTEGER,
    completed_steps INTEGER,
    failed_steps INTEGER,
    configuration JSONB,
    results JSONB,
    errors JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pipeline_runs_dag ON monitoring.pipeline_runs(dag_name);
CREATE INDEX idx_pipeline_runs_status ON monitoring.pipeline_runs(status);

-- Create update trigger for updated_at columns
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_trades_updated_at BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();