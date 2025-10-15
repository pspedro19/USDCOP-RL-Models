-- ========================================
-- USDCOP Trading System - Time Series Optimizations
-- Advanced indexes and optimizations for high-performance time-series queries
-- ========================================

\echo '🚀 Creating time-series optimizations...'

-- ========================================
-- PARTITIONING SETUP FOR MARKET_DATA (if needed)
-- Comment: PostgreSQL 10+ native partitioning for large datasets
-- ========================================

-- Create partitioned table for historical data (optional)
-- This would be used if the dataset becomes very large (>10M records)
/*
CREATE TABLE market_data_partitioned (
    LIKE market_data INCLUDING ALL
) PARTITION BY RANGE (datetime);

-- Create monthly partitions (example)
CREATE TABLE market_data_2024_01 PARTITION OF market_data_partitioned
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
*/

-- ========================================
-- SPECIALIZED INDEXES FOR TIME-SERIES QUERIES
-- ========================================

-- BRIN indexes for time-series data (efficient for large sequential data)
CREATE INDEX idx_market_data_datetime_brin ON market_data USING BRIN (datetime);

-- GIN index for JSONB metadata searches
CREATE INDEX idx_pipeline_runs_config_gin ON pipeline_runs USING GIN (config);
CREATE INDEX idx_data_quality_checks_details_gin ON data_quality_checks USING GIN (details);
CREATE INDEX idx_system_metrics_metadata_gin ON system_metrics USING GIN (metadata);

-- Partial indexes for specific query patterns
CREATE INDEX idx_market_data_recent_trading ON market_data (datetime DESC, close)
    WHERE trading_session = true AND datetime >= (NOW() - INTERVAL '7 days');

CREATE INDEX idx_market_data_intraday ON market_data (datetime DESC, open, high, low, close, volume)
    WHERE DATE(datetime) = CURRENT_DATE AND trading_session = true;

CREATE INDEX idx_market_data_by_hour ON market_data (EXTRACT(hour FROM datetime), datetime DESC)
    WHERE trading_session = true;

-- Performance index for gap detection queries
CREATE INDEX idx_market_data_sequence_check ON market_data (symbol, datetime)
    WHERE trading_session = true;

-- Index for OHLC aggregations
CREATE INDEX idx_market_data_ohlc_agg ON market_data (symbol, DATE(datetime), datetime)
    INCLUDE (open, high, low, close, volume);

-- ========================================
-- MATERIALIZED VIEWS FOR COMMON AGGREGATIONS
-- ========================================

-- Daily OHLC summary view
CREATE MATERIALIZED VIEW daily_ohlc_summary AS
SELECT
    symbol,
    DATE(datetime AT TIME ZONE 'America/Bogota') as trading_date,
    MIN(datetime) as session_start,
    MAX(datetime) as session_end,

    -- OHLC data
    (array_agg(open ORDER BY datetime))[1] as open,
    MAX(high) as high,
    MIN(low) as low,
    (array_agg(close ORDER BY datetime DESC))[1] as close,
    SUM(volume) as volume,

    -- Session statistics
    COUNT(*) as data_points,
    ROUND((COUNT(*) * 100.0 / (5 * 60 * 5)), 2) as completeness_pct, -- 5 hours * 60 min * 5min intervals

    -- Price statistics
    STDDEV(close) as price_volatility,
    (MAX(high) - MIN(low)) as trading_range,
    ROUND(((array_agg(close ORDER BY datetime DESC))[1] - (array_agg(open ORDER BY datetime))[1]) * 100.0 / (array_agg(open ORDER BY datetime))[1], 4) as daily_return_pct,

    -- Data quality
    COUNT(DISTINCT source) as source_count,
    array_agg(DISTINCT source) as sources,
    MIN(created_at) as first_recorded,
    MAX(updated_at) as last_updated

FROM market_data
WHERE trading_session = true
GROUP BY symbol, DATE(datetime AT TIME ZONE 'America/Bogota');

-- Index for daily summary
CREATE UNIQUE INDEX idx_daily_ohlc_summary_symbol_date ON daily_ohlc_summary (symbol, trading_date);
CREATE INDEX idx_daily_ohlc_summary_date ON daily_ohlc_summary (trading_date DESC);

-- Hourly trading metrics view
CREATE MATERIALIZED VIEW hourly_trading_metrics AS
SELECT
    symbol,
    DATE(datetime AT TIME ZONE 'America/Bogota') as trading_date,
    EXTRACT(hour FROM datetime AT TIME ZONE 'America/Bogota') as trading_hour,

    -- OHLC for the hour
    (array_agg(open ORDER BY datetime))[1] as open,
    MAX(high) as high,
    MIN(low) as low,
    (array_agg(close ORDER BY datetime DESC))[1] as close,
    SUM(volume) as volume,

    -- Trading activity
    COUNT(*) as data_points,
    ROUND(STDDEV(close), 4) as hourly_volatility,
    ROUND(((array_agg(close ORDER BY datetime DESC))[1] - (array_agg(open ORDER BY datetime))[1]) * 100.0 / (array_agg(open ORDER BY datetime))[1], 4) as hourly_return_pct,

    -- Quality metrics
    ROUND((COUNT(*) * 100.0 / 12), 2) as completeness_pct, -- 12 five-minute intervals per hour
    array_agg(DISTINCT source) as sources

FROM market_data
WHERE trading_session = true
GROUP BY symbol, DATE(datetime AT TIME ZONE 'America/Bogota'), EXTRACT(hour FROM datetime AT TIME ZONE 'America/Bogota');

-- Index for hourly metrics
CREATE UNIQUE INDEX idx_hourly_metrics_symbol_date_hour ON hourly_trading_metrics (symbol, trading_date, trading_hour);
CREATE INDEX idx_hourly_metrics_date_hour ON hourly_trading_metrics (trading_date DESC, trading_hour);

-- API usage efficiency view
CREATE MATERIALIZED VIEW api_usage_efficiency AS
SELECT
    api_key_name,
    DATE(request_datetime) as usage_date,

    -- Usage statistics
    COUNT(*) as total_requests,
    SUM(credits_used) as total_credits,
    COUNT(CASE WHEN success THEN 1 END) as successful_requests,
    ROUND(COUNT(CASE WHEN success THEN 1 END) * 100.0 / COUNT(*), 2) as success_rate,

    -- Performance metrics
    AVG(response_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) as p95_response_time,
    MIN(response_time_ms) as min_response_time,
    MAX(response_time_ms) as max_response_time,

    -- Rate limiting info
    MAX(daily_credits_remaining) as max_credits_remaining,
    MIN(daily_credits_remaining) as min_credits_remaining,

    -- Error analysis
    array_agg(DISTINCT error_message) FILTER (WHERE NOT success) as error_messages,
    array_agg(DISTINCT status_code) as status_codes

FROM api_usage
GROUP BY api_key_name, DATE(request_datetime);

-- Index for API usage efficiency
CREATE UNIQUE INDEX idx_api_efficiency_key_date ON api_usage_efficiency (api_key_name, usage_date);
CREATE INDEX idx_api_efficiency_date ON api_usage_efficiency (usage_date DESC);

-- ========================================
-- FUNCTIONS FOR TIME-SERIES ANALYSIS
-- ========================================

-- Function to get trading session statistics
CREATE OR REPLACE FUNCTION get_trading_session_stats(
    target_date DATE DEFAULT CURRENT_DATE,
    symbol_name VARCHAR DEFAULT 'USDCOP'
)
RETURNS TABLE(
    session_date DATE,
    session_start TIMESTAMPTZ,
    session_end TIMESTAMPTZ,
    total_points INTEGER,
    expected_points INTEGER,
    completeness_pct DECIMAL(5,2),
    avg_spread DECIMAL(8,4),
    trading_range DECIMAL(8,4),
    volume_total BIGINT,
    price_volatility DECIMAL(8,4)
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        target_date,
        MIN(md.datetime) as session_start,
        MAX(md.datetime) as session_end,
        COUNT(*)::INTEGER as total_points,
        ((5 * 60) / 5)::INTEGER as expected_points, -- 5 hours * 60 minutes / 5-minute intervals
        ROUND((COUNT(*) * 100.0 / ((5 * 60) / 5)), 2) as completeness_pct,
        ROUND(AVG(md.high - md.low), 4) as avg_spread,
        ROUND((MAX(md.high) - MIN(md.low)), 4) as trading_range,
        SUM(md.volume) as volume_total,
        ROUND(STDDEV(md.close), 4) as price_volatility
    FROM market_data md
    WHERE md.symbol = symbol_name
    AND DATE(md.datetime AT TIME ZONE 'America/Bogota') = target_date
    AND md.trading_session = true;
END;
$$ LANGUAGE plpgsql;

-- Function for real-time data quality monitoring
CREATE OR REPLACE FUNCTION check_realtime_data_quality(
    lookback_minutes INTEGER DEFAULT 30,
    symbol_name VARCHAR DEFAULT 'USDCOP'
)
RETURNS TABLE(
    check_timestamp TIMESTAMPTZ,
    latest_data_time TIMESTAMPTZ,
    data_age_minutes INTEGER,
    recent_points INTEGER,
    expected_points INTEGER,
    quality_score DECIMAL(5,2),
    issues JSONB
) AS $$
DECLARE
    latest_time TIMESTAMPTZ;
    point_count INTEGER;
    expected_count INTEGER;
    age_minutes INTEGER;
    quality DECIMAL(5,2);
    issues_found JSONB := '[]'::JSONB;
BEGIN
    -- Get latest data timestamp
    SELECT MAX(datetime) INTO latest_time
    FROM market_data
    WHERE symbol = symbol_name;

    -- Calculate age of latest data
    age_minutes := EXTRACT(EPOCH FROM (NOW() - latest_time)) / 60;

    -- Count recent data points
    SELECT COUNT(*) INTO point_count
    FROM market_data
    WHERE symbol = symbol_name
    AND datetime >= (NOW() - (lookback_minutes || ' minutes')::INTERVAL);

    -- Calculate expected points (considering trading hours)
    SELECT COUNT(*) INTO expected_count
    FROM generate_series(
        NOW() - (lookback_minutes || ' minutes')::INTERVAL,
        NOW(),
        '5 minutes'::INTERVAL
    ) ts
    WHERE is_trading_hours(ts);

    -- Calculate quality score
    quality := CASE
        WHEN expected_count = 0 THEN 100.00 -- Outside trading hours
        WHEN age_minutes > 10 THEN 0.00 -- Data too old
        WHEN point_count = 0 THEN 0.00 -- No data
        ELSE ROUND((point_count * 100.0 / expected_count), 2)
    END;

    -- Identify issues
    IF age_minutes > 10 THEN
        issues_found := issues_found || '["Data too old"]'::JSONB;
    END IF;

    IF expected_count > 0 AND point_count < (expected_count * 0.8) THEN
        issues_found := issues_found || '["Missing data points"]'::JSONB;
    END IF;

    RETURN QUERY SELECT
        NOW() as check_timestamp,
        latest_time as latest_data_time,
        age_minutes as data_age_minutes,
        point_count as recent_points,
        expected_count as expected_points,
        quality as quality_score,
        issues_found as issues;
END;
$$ LANGUAGE plpgsql;

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_trading_views()
RETURNS TEXT AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY daily_ohlc_summary;
    REFRESH MATERIALIZED VIEW CONCURRENTLY hourly_trading_metrics;
    REFRESH MATERIALIZED VIEW CONCURRENTLY api_usage_efficiency;

    RETURN 'All trading materialized views refreshed successfully at ' || NOW()::TEXT;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- PERFORMANCE MONITORING FUNCTIONS
-- ========================================

-- Function to analyze query performance
CREATE OR REPLACE FUNCTION analyze_table_performance()
RETURNS TABLE(
    table_name TEXT,
    total_size TEXT,
    index_size TEXT,
    row_count BIGINT,
    last_vacuum TIMESTAMPTZ,
    last_analyze TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        schemaname || '.' || tablename as table_name,
        pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as total_size,
        pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) as index_size,
        n_tup_ins as row_count,
        last_vacuum,
        last_autovacuum as last_analyze
    FROM pg_stat_user_tables
    WHERE schemaname = 'public'
    ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- AUTO-REFRESH SCHEDULE FUNCTION
-- ========================================

-- Function to set up automatic view refresh (to be called by cron/scheduler)
CREATE OR REPLACE FUNCTION setup_view_refresh_schedule()
RETURNS TEXT AS $$
BEGIN
    -- This would typically be called by an external scheduler
    -- For now, it just returns instructions
    RETURN 'To schedule automatic refresh, add this to your scheduler:

    # Refresh views every hour during trading hours (8 AM - 1 PM COT)
    0 8-13 * * 1-5 psql -d usdcop_trading -c "SELECT refresh_trading_views();"

    # Or use pg_cron extension if available:
    # SELECT cron.schedule(''refresh-trading-views'', ''0 8-13 * * 1-5'', ''SELECT refresh_trading_views();'');';
END;
$$ LANGUAGE plpgsql;

-- ========================================
-- CONFIRMAR CREACIÓN DE OPTIMIZACIONES
-- ========================================

\echo '✅ Time-series optimizations created successfully'
\echo '🚀 BRIN indexes for efficient time-range queries'
\echo '📊 Materialized views for common aggregations'
\echo '🔍 Specialized indexes for trading patterns'
\echo '📈 Real-time data quality monitoring functions'
\echo '⚡ Performance analysis and maintenance functions'
\echo '🔄 Automatic view refresh capabilities'