-- ============================================================================
-- ALPHA ARENA STYLE: Separate Tables per Strategy
-- ============================================================================
-- Date: 2025-10-28
-- Purpose: Independent tracking for RL, ML, and LLM strategies
-- Architecture: Each strategy has its own signals and equity tables
--
-- Strategies:
-- 1. RL_PPO - PPO-LSTM reinforcement learning
-- 2. ML_LGBM - LightGBM multiclass classifier
-- 3. LLM_DEEPSEEK - DeepSeek V3 API (primary) + Claude fallback
-- 4. LLM_CLAUDE - Claude Sonnet 4.5 (optional second LLM)
-- ============================================================================

-- Ensure schema exists
CREATE SCHEMA IF NOT EXISTS dw;

-- ============================================================================
-- SIGNALS TABLES (Separate per Strategy)
-- ============================================================================

-- RL_PPO Signals
CREATE TABLE IF NOT EXISTS dw.fact_signals_rl_ppo (
    signal_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'RL_PPO',

    -- Market data (OHLC)
    open NUMERIC(10,2),
    high NUMERIC(10,2),
    low NUMERIC(10,2),
    close NUMERIC(10,2),
    volume NUMERIC(15,2),

    -- Decision
    signal VARCHAR(20) CHECK (signal IN ('long', 'short', 'flat', 'close', 'hold')),
    size NUMERIC(10,4) CHECK (size >= 0 AND size <= 1),
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),

    -- Execution levels
    entry_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    take_profit NUMERIC(10,2),

    -- Features (17 features for RL)
    features JSONB,

    -- Metadata
    reasoning TEXT,
    model_used VARCHAR(50) DEFAULT 'PPO-LSTM',
    execution_status VARCHAR(20),
    account_balance NUMERIC(12,2),
    account_equity NUMERIC(12,2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_rl_ppo_timestamp ON dw.fact_signals_rl_ppo(timestamp DESC);
CREATE INDEX idx_signals_rl_ppo_signal ON dw.fact_signals_rl_ppo(signal) WHERE signal != 'flat';

COMMENT ON TABLE dw.fact_signals_rl_ppo IS 'RL_PPO strategy trading signals (Alpha Arena style)';


-- ML_LGBM Signals
CREATE TABLE IF NOT EXISTS dw.fact_signals_ml_lgbm (
    signal_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'ML_LGBM',

    -- Market data (OHLC)
    open NUMERIC(10,2),
    high NUMERIC(10,2),
    low NUMERIC(10,2),
    close NUMERIC(10,2),
    volume NUMERIC(15,2),

    -- Decision
    signal VARCHAR(20) CHECK (signal IN ('long', 'short', 'flat', 'close', 'hold')),
    size NUMERIC(10,4) CHECK (size >= 0 AND size <= 1),
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),

    -- Execution levels
    entry_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    take_profit NUMERIC(10,2),

    -- Features (13 features for ML)
    features JSONB,

    -- Metadata
    reasoning TEXT,
    model_used VARCHAR(50) DEFAULT 'LightGBM-Multiclass',
    execution_status VARCHAR(20),
    account_balance NUMERIC(12,2),
    account_equity NUMERIC(12,2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_ml_lgbm_timestamp ON dw.fact_signals_ml_lgbm(timestamp DESC);
CREATE INDEX idx_signals_ml_lgbm_signal ON dw.fact_signals_ml_lgbm(signal) WHERE signal != 'flat';

COMMENT ON TABLE dw.fact_signals_ml_lgbm IS 'ML_LGBM strategy trading signals (Alpha Arena style)';


-- LLM_DEEPSEEK Signals
CREATE TABLE IF NOT EXISTS dw.fact_signals_llm_deepseek (
    signal_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'LLM_DEEPSEEK',

    -- Market data (OHLC)
    open NUMERIC(10,2),
    high NUMERIC(10,2),
    low NUMERIC(10,2),
    close NUMERIC(10,2),
    volume NUMERIC(15,2),

    -- Decision
    signal VARCHAR(20) CHECK (signal IN ('buy_to_enter', 'sell_to_enter', 'hold', 'close')),
    size NUMERIC(10,4) CHECK (size >= 0 AND size <= 1),
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),

    -- Execution levels
    entry_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    take_profit NUMERIC(10,2),

    -- Features (10 Alpha Arena features for LLM)
    features JSONB,

    -- LLM specific
    reasoning TEXT, -- Full LLM reasoning
    invalidation_condition TEXT, -- Trade invalidation trigger
    model_used VARCHAR(50), -- 'deepseek-v3', 'claude-sonnet-4.5', 'fallback'
    api_latency_ms INTEGER, -- API call latency

    -- Metadata
    execution_status VARCHAR(20),
    account_balance NUMERIC(12,2),
    account_equity NUMERIC(12,2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_llm_deepseek_timestamp ON dw.fact_signals_llm_deepseek(timestamp DESC);
CREATE INDEX idx_signals_llm_deepseek_signal ON dw.fact_signals_llm_deepseek(signal) WHERE signal != 'hold';
CREATE INDEX idx_signals_llm_deepseek_model ON dw.fact_signals_llm_deepseek(model_used);

COMMENT ON TABLE dw.fact_signals_llm_deepseek IS 'LLM_DEEPSEEK strategy trading signals (Alpha Arena style - independent)';


-- LLM_CLAUDE Signals (Optional - second LLM for comparison)
CREATE TABLE IF NOT EXISTS dw.fact_signals_llm_claude (
    signal_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'LLM_CLAUDE',

    -- Market data (OHLC)
    open NUMERIC(10,2),
    high NUMERIC(10,2),
    low NUMERIC(10,2),
    close NUMERIC(10,2),
    volume NUMERIC(15,2),

    -- Decision
    signal VARCHAR(20) CHECK (signal IN ('buy_to_enter', 'sell_to_enter', 'hold', 'close')),
    size NUMERIC(10,4) CHECK (size >= 0 AND size <= 1),
    confidence NUMERIC(5,4) CHECK (confidence >= 0 AND confidence <= 1),

    -- Execution levels
    entry_price NUMERIC(10,2),
    stop_loss NUMERIC(10,2),
    take_profit NUMERIC(10,2),

    -- Features (10 Alpha Arena features)
    features JSONB,

    -- LLM specific
    reasoning TEXT,
    invalidation_condition TEXT,
    model_used VARCHAR(50) DEFAULT 'claude-sonnet-4.5',
    api_latency_ms INTEGER,

    -- Metadata
    execution_status VARCHAR(20),
    account_balance NUMERIC(12,2),
    account_equity NUMERIC(12,2),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_signals_llm_claude_timestamp ON dw.fact_signals_llm_claude(timestamp DESC);

COMMENT ON TABLE dw.fact_signals_llm_claude IS 'LLM_CLAUDE strategy trading signals (optional second LLM)';


-- ============================================================================
-- EQUITY CURVES TABLES (Separate per Strategy)
-- ============================================================================

-- RL_PPO Equity Curve
CREATE TABLE IF NOT EXISTS dw.fact_equity_rl_ppo (
    equity_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'RL_PPO',

    -- Account values
    balance NUMERIC(12,2) NOT NULL,
    equity NUMERIC(12,2) NOT NULL,
    margin_used NUMERIC(12,2),
    available_margin NUMERIC(12,2),

    -- Returns
    return_pct NUMERIC(8,4),
    return_cumulative NUMERIC(10,4),

    -- Risk metrics
    sharpe_ratio NUMERIC(6,4),
    sortino_ratio NUMERIC(6,4),
    max_drawdown NUMERIC(6,4),
    win_rate NUMERIC(5,2),
    profit_factor NUMERIC(8,4),

    -- Trading stats
    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    open_positions INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equity_rl_ppo_timestamp ON dw.fact_equity_rl_ppo(timestamp ASC);

COMMENT ON TABLE dw.fact_equity_rl_ppo IS 'RL_PPO equity curve (Alpha Arena style)';


-- ML_LGBM Equity Curve
CREATE TABLE IF NOT EXISTS dw.fact_equity_ml_lgbm (
    equity_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'ML_LGBM',

    balance NUMERIC(12,2) NOT NULL,
    equity NUMERIC(12,2) NOT NULL,
    margin_used NUMERIC(12,2),
    available_margin NUMERIC(12,2),

    return_pct NUMERIC(8,4),
    return_cumulative NUMERIC(10,4),

    sharpe_ratio NUMERIC(6,4),
    sortino_ratio NUMERIC(6,4),
    max_drawdown NUMERIC(6,4),
    win_rate NUMERIC(5,2),
    profit_factor NUMERIC(8,4),

    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    open_positions INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equity_ml_lgbm_timestamp ON dw.fact_equity_ml_lgbm(timestamp ASC);

COMMENT ON TABLE dw.fact_equity_ml_lgbm IS 'ML_LGBM equity curve (Alpha Arena style)';


-- LLM_DEEPSEEK Equity Curve
CREATE TABLE IF NOT EXISTS dw.fact_equity_llm_deepseek (
    equity_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'LLM_DEEPSEEK',

    balance NUMERIC(12,2) NOT NULL,
    equity NUMERIC(12,2) NOT NULL,
    margin_used NUMERIC(12,2),
    available_margin NUMERIC(12,2),

    return_pct NUMERIC(8,4),
    return_cumulative NUMERIC(10,4),

    sharpe_ratio NUMERIC(6,4),
    sortino_ratio NUMERIC(6,4),
    max_drawdown NUMERIC(6,4),
    win_rate NUMERIC(5,2),
    profit_factor NUMERIC(8,4),

    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    open_positions INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equity_llm_deepseek_timestamp ON dw.fact_equity_llm_deepseek(timestamp ASC);

COMMENT ON TABLE dw.fact_equity_llm_deepseek IS 'LLM_DEEPSEEK equity curve (Alpha Arena style)';


-- LLM_CLAUDE Equity Curve (Optional)
CREATE TABLE IF NOT EXISTS dw.fact_equity_llm_claude (
    equity_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    strategy VARCHAR(50) DEFAULT 'LLM_CLAUDE',

    balance NUMERIC(12,2) NOT NULL,
    equity NUMERIC(12,2) NOT NULL,
    margin_used NUMERIC(12,2),
    available_margin NUMERIC(12,2),

    return_pct NUMERIC(8,4),
    return_cumulative NUMERIC(10,4),

    sharpe_ratio NUMERIC(6,4),
    sortino_ratio NUMERIC(6,4),
    max_drawdown NUMERIC(6,4),
    win_rate NUMERIC(5,2),
    profit_factor NUMERIC(8,4),

    total_trades INTEGER,
    winning_trades INTEGER,
    losing_trades INTEGER,
    open_positions INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_equity_llm_claude_timestamp ON dw.fact_equity_llm_claude(timestamp ASC);

COMMENT ON TABLE dw.fact_equity_llm_claude IS 'LLM_CLAUDE equity curve (optional second LLM)';


-- ============================================================================
-- CONSOLIDATED VIEWS (For Easy Comparison)
-- ============================================================================

-- Leaderboard View (Alpha Arena Style)
CREATE OR REPLACE VIEW dw.vw_alpha_arena_leaderboard AS
WITH latest_metrics AS (
    -- RL_PPO latest
    SELECT
        'RL_PPO' as strategy,
        timestamp as last_update,
        equity as current_equity,
        balance as current_balance,
        return_cumulative as total_return_pct,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown as max_drawdown_pct,
        win_rate,
        profit_factor,
        total_trades,
        winning_trades,
        losing_trades,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_equity_rl_ppo

    UNION ALL

    -- ML_LGBM latest
    SELECT
        'ML_LGBM' as strategy,
        timestamp as last_update,
        equity as current_equity,
        balance as current_balance,
        return_cumulative as total_return_pct,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown as max_drawdown_pct,
        win_rate,
        profit_factor,
        total_trades,
        winning_trades,
        losing_trades,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_equity_ml_lgbm

    UNION ALL

    -- LLM_DEEPSEEK latest
    SELECT
        'LLM_DEEPSEEK' as strategy,
        timestamp as last_update,
        equity as current_equity,
        balance as current_balance,
        return_cumulative as total_return_pct,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown as max_drawdown_pct,
        win_rate,
        profit_factor,
        total_trades,
        winning_trades,
        losing_trades,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_equity_llm_deepseek

    UNION ALL

    -- LLM_CLAUDE latest (optional)
    SELECT
        'LLM_CLAUDE' as strategy,
        timestamp as last_update,
        equity as current_equity,
        balance as current_balance,
        return_cumulative as total_return_pct,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown as max_drawdown_pct,
        win_rate,
        profit_factor,
        total_trades,
        winning_trades,
        losing_trades,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_equity_llm_claude
)
SELECT
    ROW_NUMBER() OVER (ORDER BY total_return_pct DESC) as rank,
    strategy,
    last_update,
    current_equity,
    current_balance,
    total_return_pct,
    sharpe_ratio,
    sortino_ratio,
    max_drawdown_pct,
    win_rate,
    profit_factor,
    total_trades,
    winning_trades,
    losing_trades,
    CASE
        WHEN rank() OVER (ORDER BY total_return_pct DESC) = 1 THEN '🥇 WINNER'
        WHEN rank() OVER (ORDER BY total_return_pct DESC) = 2 THEN '🥈 SECOND'
        WHEN rank() OVER (ORDER BY total_return_pct DESC) = 3 THEN '🥉 THIRD'
        ELSE '🔹'
    END as medal
FROM latest_metrics
WHERE rn = 1  -- Only latest row per strategy
ORDER BY total_return_pct DESC;

COMMENT ON VIEW dw.vw_alpha_arena_leaderboard IS 'Alpha Arena leaderboard - strategies ranked by performance';


-- Performance Comparison View
CREATE OR REPLACE VIEW dw.vw_strategy_comparison AS
SELECT
    'RL_PPO' as strategy,
    'RL' as type,
    COUNT(*) as total_signals,
    COUNT(*) FILTER (WHERE signal IN ('long', 'short')) as total_trades,
    AVG(confidence) FILTER (WHERE signal IN ('long', 'short')) as avg_confidence,
    COUNT(*) FILTER (WHERE signal = 'long') as long_signals,
    COUNT(*) FILTER (WHERE signal = 'short') as short_signals,
    COUNT(*) FILTER (WHERE signal = 'flat') as flat_signals
FROM dw.fact_signals_rl_ppo

UNION ALL

SELECT
    'ML_LGBM' as strategy,
    'ML' as type,
    COUNT(*) as total_signals,
    COUNT(*) FILTER (WHERE signal IN ('long', 'short')) as total_trades,
    AVG(confidence) FILTER (WHERE signal IN ('long', 'short')) as avg_confidence,
    COUNT(*) FILTER (WHERE signal = 'long') as long_signals,
    COUNT(*) FILTER (WHERE signal = 'short') as short_signals,
    COUNT(*) FILTER (WHERE signal = 'flat') as flat_signals
FROM dw.fact_signals_ml_lgbm

UNION ALL

SELECT
    'LLM_DEEPSEEK' as strategy,
    'LLM' as type,
    COUNT(*) as total_signals,
    COUNT(*) FILTER (WHERE signal IN ('buy_to_enter', 'sell_to_enter')) as total_trades,
    AVG(confidence) FILTER (WHERE signal IN ('buy_to_enter', 'sell_to_enter')) as avg_confidence,
    COUNT(*) FILTER (WHERE signal = 'buy_to_enter') as long_signals,
    COUNT(*) FILTER (WHERE signal = 'sell_to_enter') as short_signals,
    COUNT(*) FILTER (WHERE signal = 'hold') as flat_signals
FROM dw.fact_signals_llm_deepseek

ORDER BY total_trades DESC;

COMMENT ON VIEW dw.vw_strategy_comparison IS 'Compare trading activity across strategies';


-- Latest Decisions View (Real-time Dashboard)
CREATE OR REPLACE VIEW dw.vw_latest_decisions AS
WITH ranked AS (
    SELECT
        'RL_PPO' as strategy,
        timestamp,
        signal,
        size,
        confidence,
        entry_price,
        stop_loss,
        take_profit,
        reasoning,
        account_equity,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_signals_rl_ppo

    UNION ALL

    SELECT
        'ML_LGBM' as strategy,
        timestamp,
        signal,
        size,
        confidence,
        entry_price,
        stop_loss,
        take_profit,
        reasoning,
        account_equity,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_signals_ml_lgbm

    UNION ALL

    SELECT
        'LLM_DEEPSEEK' as strategy,
        timestamp,
        signal,
        size,
        confidence,
        entry_price,
        stop_loss,
        take_profit,
        reasoning,
        account_equity,
        ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
    FROM dw.fact_signals_llm_deepseek
)
SELECT *
FROM ranked
WHERE rn = 1
ORDER BY strategy;

COMMENT ON VIEW dw.vw_latest_decisions IS 'Latest decision from each strategy (real-time)';


-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to get equity curve for date range
CREATE OR REPLACE FUNCTION dw.get_equity_curve(
    p_strategy VARCHAR(50),
    p_start_date TIMESTAMPTZ DEFAULT NOW() - INTERVAL '30 days',
    p_end_date TIMESTAMPTZ DEFAULT NOW()
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    equity NUMERIC,
    return_pct NUMERIC,
    sharpe_ratio NUMERIC
) AS $$
BEGIN
    IF p_strategy = 'RL_PPO' THEN
        RETURN QUERY
        SELECT e.timestamp, e.equity, e.return_pct, e.sharpe_ratio
        FROM dw.fact_equity_rl_ppo e
        WHERE e.timestamp BETWEEN p_start_date AND p_end_date
        ORDER BY e.timestamp ASC;

    ELSIF p_strategy = 'ML_LGBM' THEN
        RETURN QUERY
        SELECT e.timestamp, e.equity, e.return_pct, e.sharpe_ratio
        FROM dw.fact_equity_ml_lgbm e
        WHERE e.timestamp BETWEEN p_start_date AND p_end_date
        ORDER BY e.timestamp ASC;

    ELSIF p_strategy = 'LLM_DEEPSEEK' THEN
        RETURN QUERY
        SELECT e.timestamp, e.equity, e.return_pct, e.sharpe_ratio
        FROM dw.fact_equity_llm_deepseek e
        WHERE e.timestamp BETWEEN p_start_date AND p_end_date
        ORDER BY e.timestamp ASC;

    ELSE
        RAISE EXCEPTION 'Unknown strategy: %', p_strategy;
    END IF;
END;
$$ LANGUAGE plpgsql;


-- Function to compare strategies
CREATE OR REPLACE FUNCTION dw.compare_strategies()
RETURNS TABLE (
    metric VARCHAR(50),
    rl_ppo NUMERIC,
    ml_lgbm NUMERIC,
    llm_deepseek NUMERIC,
    winner VARCHAR(50)
) AS $$
BEGIN
    RETURN QUERY
    WITH latest AS (
        SELECT * FROM dw.vw_alpha_arena_leaderboard
    )
    SELECT
        'Total Return %' as metric,
        (SELECT total_return_pct FROM latest WHERE strategy = 'RL_PPO') as rl_ppo,
        (SELECT total_return_pct FROM latest WHERE strategy = 'ML_LGBM') as ml_lgbm,
        (SELECT total_return_pct FROM latest WHERE strategy = 'LLM_DEEPSEEK') as llm_deepseek,
        (SELECT strategy FROM latest ORDER BY total_return_pct DESC LIMIT 1) as winner

    UNION ALL

    SELECT
        'Sharpe Ratio',
        (SELECT sharpe_ratio FROM latest WHERE strategy = 'RL_PPO'),
        (SELECT sharpe_ratio FROM latest WHERE strategy = 'ML_LGBM'),
        (SELECT sharpe_ratio FROM latest WHERE strategy = 'LLM_DEEPSEEK'),
        (SELECT strategy FROM latest ORDER BY sharpe_ratio DESC LIMIT 1)

    UNION ALL

    SELECT
        'Max Drawdown %',
        (SELECT max_drawdown_pct FROM latest WHERE strategy = 'RL_PPO'),
        (SELECT max_drawdown_pct FROM latest WHERE strategy = 'ML_LGBM'),
        (SELECT max_drawdown_pct FROM latest WHERE strategy = 'LLM_DEEPSEEK'),
        (SELECT strategy FROM latest ORDER BY max_drawdown_pct ASC LIMIT 1)

    UNION ALL

    SELECT
        'Win Rate %',
        (SELECT win_rate FROM latest WHERE strategy = 'RL_PPO'),
        (SELECT win_rate FROM latest WHERE strategy = 'ML_LGBM'),
        (SELECT win_rate FROM latest WHERE strategy = 'LLM_DEEPSEEK'),
        (SELECT strategy FROM latest ORDER BY win_rate DESC LIMIT 1);
END;
$$ LANGUAGE plpgsql;


-- ============================================================================
-- SAMPLE QUERIES (For Reference)
-- ============================================================================

/*
-- Get leaderboard
SELECT * FROM dw.vw_alpha_arena_leaderboard;

-- Get latest decisions
SELECT * FROM dw.vw_latest_decisions;

-- Compare strategies
SELECT * FROM dw.compare_strategies();

-- Get equity curve for specific strategy
SELECT * FROM dw.get_equity_curve('LLM_DEEPSEEK', NOW() - INTERVAL '7 days', NOW());

-- Find best performing strategy
SELECT strategy, total_return_pct, sharpe_ratio
FROM dw.vw_alpha_arena_leaderboard
WHERE rank = 1;

-- Count signals by type
SELECT strategy, signal, COUNT(*)
FROM (
    SELECT 'RL_PPO' as strategy, signal FROM dw.fact_signals_rl_ppo
    UNION ALL
    SELECT 'ML_LGBM', signal FROM dw.fact_signals_ml_lgbm
    UNION ALL
    SELECT 'LLM_DEEPSEEK', signal FROM dw.fact_signals_llm_deepseek
) combined
GROUP BY strategy, signal
ORDER BY strategy, COUNT(*) DESC;
*/
