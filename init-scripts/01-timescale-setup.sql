-- =====================================================
-- TimescaleDB Setup for USDCOP Trading System
-- =====================================================

-- Create the TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Create historical market data table
CREATE TABLE IF NOT EXISTS historical_market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open DECIMAL(15,6) NOT NULL,
    high DECIMAL(15,6) NOT NULL,
    low DECIMAL(15,6) NOT NULL,
    close DECIMAL(15,6) NOT NULL,
    volume BIGINT DEFAULT 0,
    spread DECIMAL(10,6),
    tick_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('historical_market_data', 'time', 
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_historical_symbol_time 
ON historical_market_data (symbol, time DESC);

CREATE INDEX IF NOT EXISTS idx_historical_time 
ON historical_market_data (time DESC);

-- Real-time market data table for current session
CREATE TABLE IF NOT EXISTS realtime_market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    bid DECIMAL(15,6) NOT NULL,
    ask DECIMAL(15,6) NOT NULL,
    last DECIMAL(15,6) NOT NULL,
    volume BIGINT DEFAULT 0,
    change_percent DECIMAL(8,4),
    session_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('realtime_market_data', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_realtime_symbol_session 
ON realtime_market_data (symbol, session_date, time DESC);

CREATE INDEX IF NOT EXISTS idx_realtime_time 
ON realtime_market_data (time DESC);

-- Market sessions table
CREATE TABLE IF NOT EXISTS market_sessions (
    id SERIAL PRIMARY KEY,
    session_date DATE NOT NULL UNIQUE,
    start_time TIMESTAMPTZ NOT NULL,
    end_time TIMESTAMPTZ,
    status VARCHAR(20) DEFAULT 'active',
    total_updates INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Data quality metrics table
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    time TIMESTAMPTZ NOT NULL,
    table_name VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    session_date DATE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('data_quality_metrics', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE
);

-- System health table
CREATE TABLE IF NOT EXISTS system_health (
    time TIMESTAMPTZ NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    response_time_ms INTEGER,
    cpu_usage DECIMAL(5,2),
    memory_usage DECIMAL(5,2),
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable
SELECT create_hypertable('system_health', 'time',
    chunk_time_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create compression policy for historical data (compress data older than 7 days)
SELECT add_compression_policy('historical_market_data', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('data_quality_metrics', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('system_health', INTERVAL '2 days', if_not_exists => TRUE);

-- Create retention policy (keep data for 2 years)
SELECT add_retention_policy('historical_market_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('data_quality_metrics', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('system_health', INTERVAL '30 days', if_not_exists => TRUE);

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_market_data
WITH (timescaledb.continuous) AS
SELECT 
    time_bucket('1 hour', time) AS bucket,
    symbol,
    FIRST(open, time) as open,
    MAX(high) as high,
    MIN(low) as low,
    LAST(close, time) as close,
    SUM(volume) as volume,
    AVG(spread) as avg_spread
FROM historical_market_data
GROUP BY bucket, symbol;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('hourly_market_data',
    start_offset => INTERVAL '2 hours',
    end_offset => INTERVAL '10 minutes',
    schedule_interval => INTERVAL '10 minutes',
    if_not_exists => TRUE
);

-- Insert initial market session if not exists
INSERT INTO market_sessions (session_date, start_time, status)
VALUES (CURRENT_DATE, NOW(), 'active')
ON CONFLICT (session_date) DO NOTHING;

-- Create function to update market session
CREATE OR REPLACE FUNCTION update_market_session()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE market_sessions 
    SET total_updates = total_updates + 1,
        updated_at = NOW()
    WHERE session_date = CURRENT_DATE;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER trigger_update_session_on_realtime_insert
    AFTER INSERT ON realtime_market_data
    FOR EACH ROW EXECUTE FUNCTION update_market_session();

COMMIT;