-- ============================================================================
-- TRANSPARENT LOGGING COLUMNS (ModelChat Style)
-- ============================================================================
-- Adds transparency features similar to Alpha Arena's ModelChat
-- Allows auditing and improving LLM decisions
--
-- Date: 2025-10-28
-- ============================================================================

-- Add columns to LLM signals tables for transparency

-- LLM_DEEPSEEK transparency
ALTER TABLE dw.fact_signals_llm_deepseek
ADD COLUMN IF NOT EXISTS prompt_used TEXT,
ADD COLUMN IF NOT EXISTS raw_llm_output TEXT,
ADD COLUMN IF NOT EXISTS tokens_input INTEGER,
ADD COLUMN IF NOT EXISTS tokens_output INTEGER,
ADD COLUMN IF NOT EXISTS cost_usd NUMERIC(8,6),
ADD COLUMN IF NOT EXISTS model_temperature NUMERIC(3,2),
ADD COLUMN IF NOT EXISTS api_call_duration_ms INTEGER,
ADD COLUMN IF NOT EXISTS prompt_hash VARCHAR(64);

COMMENT ON COLUMN dw.fact_signals_llm_deepseek.prompt_used IS 'Exact prompt sent to LLM (for auditing)';
COMMENT ON COLUMN dw.fact_signals_llm_deepseek.raw_llm_output IS 'Raw LLM response before parsing';
COMMENT ON COLUMN dw.fact_signals_llm_deepseek.tokens_input IS 'Input tokens consumed';
COMMENT ON COLUMN dw.fact_signals_llm_deepseek.tokens_output IS 'Output tokens generated';
COMMENT ON COLUMN dw.fact_signals_llm_deepseek.cost_usd IS 'Estimated cost in USD';
COMMENT ON COLUMN dw.fact_signals_llm_deepseek.prompt_hash IS 'SHA256 hash of prompt (deduplication)';


-- LLM_CLAUDE transparency
ALTER TABLE dw.fact_signals_llm_claude
ADD COLUMN IF NOT EXISTS prompt_used TEXT,
ADD COLUMN IF NOT EXISTS raw_llm_output TEXT,
ADD COLUMN IF NOT EXISTS tokens_input INTEGER,
ADD COLUMN IF NOT EXISTS tokens_output INTEGER,
ADD COLUMN IF NOT EXISTS cost_usd NUMERIC(8,6),
ADD COLUMN IF NOT EXISTS model_temperature NUMERIC(3,2),
ADD COLUMN IF NOT EXISTS api_call_duration_ms INTEGER,
ADD COLUMN IF NOT EXISTS prompt_hash VARCHAR(64);


-- ============================================================================
-- API COST TRACKING TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS dw.fact_api_usage (
    usage_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Model info
    model_provider VARCHAR(50) NOT NULL, -- 'deepseek', 'claude', 'openai'
    model_name VARCHAR(100) NOT NULL,
    strategy VARCHAR(50),

    -- Usage
    tokens_input INTEGER NOT NULL,
    tokens_output INTEGER NOT NULL,
    tokens_total INTEGER GENERATED ALWAYS AS (tokens_input + tokens_output) STORED,

    -- Cost
    cost_input_usd NUMERIC(10,6),
    cost_output_usd NUMERIC(10,6),
    cost_total_usd NUMERIC(10,6) GENERATED ALWAYS AS (cost_input_usd + cost_output_usd) STORED,

    -- Performance
    api_latency_ms INTEGER,
    success BOOLEAN DEFAULT TRUE,
    error_message TEXT,

    -- Metadata
    prompt_hash VARCHAR(64),
    cache_hit BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_api_usage_timestamp ON dw.fact_api_usage(timestamp DESC);
CREATE INDEX idx_api_usage_model ON dw.fact_api_usage(model_provider, model_name);
CREATE INDEX idx_api_usage_strategy ON dw.fact_api_usage(strategy);

COMMENT ON TABLE dw.fact_api_usage IS 'LLM API usage tracking (cost and performance monitoring)';


-- ============================================================================
-- COST ANALYSIS VIEWS
-- ============================================================================

-- Daily cost summary
CREATE OR REPLACE VIEW dw.vw_api_cost_daily AS
SELECT
    DATE(timestamp) as date,
    model_provider,
    strategy,
    COUNT(*) as total_calls,
    COUNT(*) FILTER (WHERE success = TRUE) as successful_calls,
    COUNT(*) FILTER (WHERE success = FALSE) as failed_calls,
    SUM(tokens_input) as total_tokens_input,
    SUM(tokens_output) as total_tokens_output,
    SUM(cost_total_usd) as total_cost_usd,
    AVG(api_latency_ms) as avg_latency_ms,
    COUNT(*) FILTER (WHERE cache_hit = TRUE) as cache_hits,
    (COUNT(*) FILTER (WHERE cache_hit = TRUE)::FLOAT / NULLIF(COUNT(*), 0)) * 100 as cache_hit_rate
FROM dw.fact_api_usage
GROUP BY DATE(timestamp), model_provider, strategy
ORDER BY date DESC, total_cost_usd DESC;

COMMENT ON VIEW dw.vw_api_cost_daily IS 'Daily API cost breakdown per model and strategy';


-- Monthly forecast
CREATE OR REPLACE VIEW dw.vw_api_cost_forecast AS
WITH recent_usage AS (
    SELECT
        model_provider,
        AVG(cost_total_usd) as avg_cost_per_call,
        COUNT(*) as calls_last_day
    FROM dw.fact_api_usage
    WHERE timestamp >= NOW() - INTERVAL '1 day'
    GROUP BY model_provider
)
SELECT
    model_provider,
    avg_cost_per_call,
    calls_last_day,
    -- Forecast daily (assuming 288 M5 bars/day during trading hours)
    avg_cost_per_call * 288 as forecast_daily_usd,
    -- Forecast monthly (22 trading days)
    avg_cost_per_call * 288 * 22 as forecast_monthly_usd,
    -- Alpha Arena comparison
    CASE
        WHEN avg_cost_per_call * 288 * 22 < 50 THEN 'AFFORDABLE (<$50/mo)'
        WHEN avg_cost_per_call * 288 * 22 < 100 THEN 'MODERATE ($50-100/mo)'
        ELSE 'EXPENSIVE (>$100/mo)'
    END as cost_category
FROM recent_usage;

COMMENT ON VIEW dw.vw_api_cost_forecast IS 'Monthly API cost forecast based on recent usage';


-- ============================================================================
-- TRANSPARENCY QUERIES (ModelChat Style)
-- ============================================================================

-- Get LLM reasoning for specific trade
CREATE OR REPLACE FUNCTION dw.get_llm_reasoning(
    p_signal_id INTEGER,
    p_strategy VARCHAR(50) DEFAULT 'LLM_DEEPSEEK'
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    signal VARCHAR(20),
    reasoning TEXT,
    raw_output TEXT,
    prompt_used TEXT,
    tokens_used INTEGER,
    cost_usd NUMERIC
) AS $$
BEGIN
    IF p_strategy = 'LLM_DEEPSEEK' THEN
        RETURN QUERY
        SELECT
            s.timestamp,
            s.signal,
            s.reasoning,
            s.raw_llm_output,
            s.prompt_used,
            s.tokens_input + s.tokens_output as tokens_used,
            s.cost_usd
        FROM dw.fact_signals_llm_deepseek s
        WHERE s.signal_id = p_signal_id;

    ELSIF p_strategy = 'LLM_CLAUDE' THEN
        RETURN QUERY
        SELECT
            s.timestamp,
            s.signal,
            s.reasoning,
            s.raw_llm_output,
            s.prompt_used,
            s.tokens_input + s.tokens_output as tokens_used,
            s.cost_usd
        FROM dw.fact_signals_llm_claude s
        WHERE s.signal_id = p_signal_id;
    ELSE
        RAISE EXCEPTION 'Invalid strategy: %', p_strategy;
    END IF;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION dw.get_llm_reasoning IS 'Get full LLM reasoning and transparency data for a signal';


-- ============================================================================
-- SAMPLE QUERIES
-- ============================================================================

/*
-- Daily API costs
SELECT * FROM dw.vw_api_cost_daily
WHERE date >= CURRENT_DATE - INTERVAL '7 days'
ORDER BY date DESC;

-- Monthly forecast
SELECT * FROM dw.vw_api_cost_forecast;

-- Get reasoning for specific signal
SELECT * FROM dw.get_llm_reasoning(123, 'LLM_DEEPSEEK');

-- Find most expensive decisions
SELECT
    signal_id,
    timestamp,
    signal,
    cost_usd,
    tokens_input + tokens_output as tokens_total,
    SUBSTRING(reasoning, 1, 100) as reasoning_preview
FROM dw.fact_signals_llm_deepseek
ORDER BY cost_usd DESC
LIMIT 10;

-- Compare DeepSeek vs Claude costs
SELECT
    model_provider,
    SUM(cost_total_usd) as total_cost,
    COUNT(*) as total_calls,
    AVG(cost_total_usd) as avg_cost_per_call,
    SUM(tokens_total) as total_tokens
FROM dw.fact_api_usage
WHERE timestamp >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY model_provider;
*/
