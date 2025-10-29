-- ========================================================================
-- DWH FACT TABLES - Kimball Star Schema
-- Sistema USDCOP Trading - All Fact Tables for L0-L6
-- ========================================================================
-- Version: 1.0
-- Date: 2025-10-22
-- Description: Creates all fact tables for the data warehouse
-- ========================================================================

-- ========================================================================
-- LAYER 0 FACTS - Raw Data Ingestion
-- ========================================================================

-- ===================
-- fact_bar_5m - OHLCV bars (TimescaleDB hypertable)
-- ===================
-- NOTE: This is the bridge between public.usdcop_m5_ohlcv and the DWH
-- It references dimension keys for proper star schema queries

CREATE TABLE IF NOT EXISTS dw.fact_bar_5m (
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),
    time_id INT NOT NULL REFERENCES dw.dim_time_5m(time_id),
    ts_utc TIMESTAMPTZ NOT NULL,

    open DECIMAL(12,6) NOT NULL,
    high DECIMAL(12,6) NOT NULL,
    low DECIMAL(12,6) NOT NULL,
    close DECIMAL(12,6) NOT NULL,
    volume BIGINT DEFAULT 0,

    source_id INT REFERENCES dw.dim_source(source_id),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (symbol_id, ts_utc)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('dw.fact_bar_5m', 'ts_utc',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fact_bar_symbol_time ON dw.fact_bar_5m(symbol_id, time_id);
CREATE INDEX IF NOT EXISTS idx_fact_bar_ts ON dw.fact_bar_5m(ts_utc DESC);

COMMENT ON TABLE dw.fact_bar_5m IS 'OHLCV bars at 5-minute granularity (TimescaleDB hypertable)';

-- ===================
-- fact_l0_acquisition - Pipeline run metrics
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_l0_acquisition (
    run_id VARCHAR(100) PRIMARY KEY,
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),
    source_id INT NOT NULL REFERENCES dw.dim_source(source_id),
    execution_date DATE NOT NULL,

    fetch_mode VARCHAR(50) NOT NULL,  -- 'complete_historical', 'recent_incremental'
    date_range_start TIMESTAMPTZ NOT NULL,
    date_range_end TIMESTAMPTZ NOT NULL,

    -- Quality metrics
    rows_fetched INT NOT NULL,
    rows_inserted INT NOT NULL,
    rows_duplicated INT DEFAULT 0,
    rows_rejected INT DEFAULT 0,
    stale_rate_pct DECIMAL(5,2),  -- % OHLC repetidos
    coverage_pct DECIMAL(5,2),    -- % barras esperadas vs obtenidas
    gaps_detected INT DEFAULT 0,

    -- Performance
    duration_sec INT NOT NULL,
    api_calls_count INT DEFAULT 0,
    api_cost_usd DECIMAL(10,6) DEFAULT 0,

    -- GO/NO-GO gate
    quality_passed BOOLEAN NOT NULL,

    -- Lineage
    minio_manifest_path TEXT,
    dag_id VARCHAR(200),
    task_id VARCHAR(200),

    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_fact_l0_exec_date ON dw.fact_l0_acquisition(execution_date DESC);
CREATE INDEX IF NOT EXISTS idx_fact_l0_symbol ON dw.fact_l0_acquisition(symbol_id);
CREATE INDEX IF NOT EXISTS idx_fact_l0_passed ON dw.fact_l0_acquisition(quality_passed);

COMMENT ON TABLE dw.fact_l0_acquisition IS 'L0 pipeline execution metrics (grain: 1 row per DAG run)';

-- ========================================================================
-- LAYER 1 FACTS - Standardization & Quality
-- ========================================================================

-- ===================
-- fact_l1_quality - Daily quality scorecard
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_l1_quality (
    date_cot DATE NOT NULL,
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),

    -- Coverage
    total_episodes INT NOT NULL,
    accepted_episodes INT NOT NULL,
    rejected_episodes INT NOT NULL,

    -- Validations
    grid_300s_ok BOOLEAN NOT NULL,  -- Barras cada 5min exactos
    repeated_ohlc_rate_pct DECIMAL(5,2),  -- % OHLC idénticos consecutivos
    gaps_over_1_interval INT DEFAULT 0,
    coverage_pct DECIMAL(5,2),

    -- GO/NO-GO
    status_passed BOOLEAN NOT NULL,

    -- Lineage
    run_id VARCHAR(100),
    minio_manifest_path TEXT,
    dag_id VARCHAR(200),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (date_cot, symbol_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_l1_date ON dw.fact_l1_quality(date_cot DESC);
CREATE INDEX IF NOT EXISTS idx_fact_l1_passed ON dw.fact_l1_quality(status_passed);

COMMENT ON TABLE dw.fact_l1_quality IS 'L1 daily data quality scorecard (grain: 1 row per date/symbol)';

-- ========================================================================
-- LAYER 2 FACTS - Technical Indicators
-- ========================================================================

-- ===================
-- fact_indicator_5m - Technical indicator values per bar
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m (
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),
    time_id INT NOT NULL REFERENCES dw.dim_time_5m(time_id),
    indicator_id INT NOT NULL REFERENCES dw.dim_indicator(indicator_id),
    ts_utc TIMESTAMPTZ NOT NULL,

    indicator_value DECIMAL(18,6) NOT NULL,
    signal VARCHAR(20),  -- 'buy', 'sell', 'neutral', 'overbought', 'oversold'

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (symbol_id, time_id, indicator_id)
) PARTITION BY RANGE (ts_utc);

-- Create partitions for 2024-2026 (extend as needed)
CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2024 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2025 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');

CREATE TABLE IF NOT EXISTS dw.fact_indicator_5m_2026 PARTITION OF dw.fact_indicator_5m
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_fact_indicator_time ON dw.fact_indicator_5m(time_id);
CREATE INDEX IF NOT EXISTS idx_fact_indicator_symbol ON dw.fact_indicator_5m(symbol_id);
CREATE INDEX IF NOT EXISTS idx_fact_indicator_name ON dw.fact_indicator_5m(indicator_id);

COMMENT ON TABLE dw.fact_indicator_5m IS 'Technical indicator values per bar (grain: symbol/timestamp/indicator)';

-- ===================
-- fact_winsorization - Daily winsorization stats
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_winsorization (
    date_cot DATE NOT NULL,
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),

    winsor_rate_pct DECIMAL(5,2) NOT NULL,
    n_sigma DECIMAL(3,1) DEFAULT 4.0,
    lower_threshold_bps DECIMAL(10,2),
    upper_threshold_bps DECIMAL(10,2),
    outliers_clipped INT DEFAULT 0,

    status_passed BOOLEAN NOT NULL,

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (date_cot, symbol_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_winsor_date ON dw.fact_winsorization(date_cot DESC);

COMMENT ON TABLE dw.fact_winsorization IS 'L2 daily winsorization statistics';

-- ===================
-- fact_hod_baseline - Hour-of-Day baselines
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_hod_baseline (
    hhmm_cot VARCHAR(5) NOT NULL,  -- '08:00', '08:05', ..., '12:55'
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),

    median_ret_log_5m DECIMAL(12,8),
    mad_ret_log_5m DECIMAL(12,8),
    p95_range_bps DECIMAL(10,4),
    p99_range_bps DECIMAL(10,4),

    observations_count INT NOT NULL,

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (hhmm_cot, symbol_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_hod_hhmm ON dw.fact_hod_baseline(hhmm_cot);

COMMENT ON TABLE dw.fact_hod_baseline IS 'L2 hour-of-day baselines for normalization';

-- ========================================================================
-- LAYER 3 FACTS - Feature Engineering
-- ========================================================================

-- ===================
-- fact_forward_ic - Forward information coefficient per feature
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_forward_ic (
    feature_id INT NOT NULL REFERENCES dw.dim_feature(feature_id),
    date_cot DATE NOT NULL,
    split VARCHAR(10) NOT NULL,  -- 'train', 'val', 'test'

    ic DECIMAL(8,6),      -- Information coefficient (correlation)
    pval DECIMAL(8,6),    -- P-value
    is_significant BOOLEAN DEFAULT FALSE,

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (feature_id, date_cot, split)
);

CREATE INDEX IF NOT EXISTS idx_fact_fwd_ic_feature ON dw.fact_forward_ic(feature_id);
CREATE INDEX IF NOT EXISTS idx_fact_fwd_ic_split ON dw.fact_forward_ic(split);
CREATE INDEX IF NOT EXISTS idx_fact_fwd_ic_sig ON dw.fact_forward_ic(is_significant) WHERE is_significant = TRUE;

COMMENT ON TABLE dw.fact_forward_ic IS 'L3 forward information coefficient (feature predictiveness)';

-- ===================
-- fact_leakage_tests - Anti-leakage test results per feature
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_leakage_tests (
    feature_id INT NOT NULL REFERENCES dw.dim_feature(feature_id),
    date_cot DATE NOT NULL,

    same_bar_leak INT DEFAULT 0,      -- 0 = pass, >0 = fail
    gap_leak INT DEFAULT 0,            -- 0 = pass, >0 = fail
    masking_ok BOOLEAN DEFAULT TRUE,   -- TRUE = pass

    status_passed BOOLEAN NOT NULL,

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (feature_id, date_cot)
);

CREATE INDEX IF NOT EXISTS idx_fact_leak_feature ON dw.fact_leakage_tests(feature_id);
CREATE INDEX IF NOT EXISTS idx_fact_leak_passed ON dw.fact_leakage_tests(status_passed);

COMMENT ON TABLE dw.fact_leakage_tests IS 'L3 anti-leakage test results per feature';

-- ===================
-- fact_feature_corr - Feature correlation matrix (long format)
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_feature_corr (
    feature_i_id INT NOT NULL REFERENCES dw.dim_feature(feature_id),
    feature_j_id INT NOT NULL REFERENCES dw.dim_feature(feature_id),
    date_cot DATE NOT NULL,

    correlation DECIMAL(8,6) NOT NULL,  -- Pearson correlation [-1, 1]

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (feature_i_id, feature_j_id, date_cot),
    CHECK (feature_i_id <= feature_j_id)  -- Store only upper triangle
);

CREATE INDEX IF NOT EXISTS idx_fact_corr_date ON dw.fact_feature_corr(date_cot DESC);
CREATE INDEX IF NOT EXISTS idx_fact_corr_features ON dw.fact_feature_corr(feature_i_id, feature_j_id);

COMMENT ON TABLE dw.fact_feature_corr IS 'L3 feature correlation matrix (long format)';

-- ========================================================================
-- LAYER 4 FACTS - RL-Ready Dataset
-- ========================================================================

-- ===================
-- fact_rl_obs_stats - Observation statistics per feature
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_rl_obs_stats (
    feature_id INT NOT NULL REFERENCES dw.dim_feature(feature_id),
    split VARCHAR(10) NOT NULL,  -- 'train', 'val', 'test'
    date_cot DATE NOT NULL,

    clip_rate DECIMAL(8,6),       -- % observations clipped
    abs_max DECIMAL(12,6),        -- Max absolute value
    mean_val DECIMAL(12,6),
    std_val DECIMAL(12,6),
    p50_val DECIMAL(12,6),
    p95_val DECIMAL(12,6),
    p99_val DECIMAL(12,6),

    observations_count INT NOT NULL,

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (feature_id, split, date_cot)
);

CREATE INDEX IF NOT EXISTS idx_fact_rl_obs_feature ON dw.fact_rl_obs_stats(feature_id);
CREATE INDEX IF NOT EXISTS idx_fact_rl_obs_split ON dw.fact_rl_obs_stats(split);

COMMENT ON TABLE dw.fact_rl_obs_stats IS 'L4 RL observation statistics per feature';

-- ===================
-- fact_cost_model_stats - Cost model statistics
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_cost_model_stats (
    cost_model_sk INT NOT NULL REFERENCES dw.dim_cost_model(cost_model_sk),
    split VARCHAR(10) NOT NULL,  -- 'train', 'val', 'test'
    date_cot DATE NOT NULL,

    spread_p50_bps DECIMAL(8,4),
    spread_p95_bps DECIMAL(8,4),
    spread_p99_bps DECIMAL(8,4),
    peg_rate_pct DECIMAL(5,2),

    observations_count INT NOT NULL,

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (cost_model_sk, split, date_cot)
);

CREATE INDEX IF NOT EXISTS idx_fact_cost_model_split ON dw.fact_cost_model_stats(split);

COMMENT ON TABLE dw.fact_cost_model_stats IS 'L4 cost model statistics per split';

-- ===================
-- fact_episode - Episode metadata
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_episode (
    episode_sk INT NOT NULL REFERENCES dw.dim_episode(episode_sk),
    reward_spec_sk INT REFERENCES dw.dim_reward_spec(reward_spec_sk),

    episode_length INT NOT NULL,
    reward_sum DECIMAL(12,6),
    reward_mean DECIMAL(12,6),
    reward_std DECIMAL(12,6),
    reward_zeros_pct DECIMAL(5,2),

    run_id VARCHAR(100),
    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (episode_sk)
);

CREATE INDEX IF NOT EXISTS idx_fact_episode_reward_spec ON dw.fact_episode(reward_spec_sk);

COMMENT ON TABLE dw.fact_episode IS 'L4 episode-level statistics';

-- ========================================================================
-- LAYER 5 FACTS - Model Serving & Signals
-- ========================================================================

-- ===================
-- fact_signal_5m - Trading signals (TimescaleDB hypertable)
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_signal_5m (
    model_sk INT NOT NULL REFERENCES dw.dim_model(model_sk),
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),
    time_id INT NOT NULL REFERENCES dw.dim_time_5m(time_id),
    ts_utc TIMESTAMPTZ NOT NULL,

    action VARCHAR(10) NOT NULL,  -- 'HOLD', 'BUY', 'SELL'
    confidence DECIMAL(5,4),      -- [0, 1]

    q_hold DECIMAL(12,6),
    q_buy DECIMAL(12,6),
    q_sell DECIMAL(12,6),

    epsilon DECIMAL(5,4),         -- Exploration rate (if applicable)
    reason_code VARCHAR(100),     -- 'max_q', 'random_exploration', etc.

    latency_ms INT,               -- Inference latency

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (model_sk, symbol_id, ts_utc)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable('dw.fact_signal_5m', 'ts_utc',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_fact_signal_model ON dw.fact_signal_5m(model_sk);
CREATE INDEX IF NOT EXISTS idx_fact_signal_symbol ON dw.fact_signal_5m(symbol_id);
CREATE INDEX IF NOT EXISTS idx_fact_signal_ts ON dw.fact_signal_5m(ts_utc DESC);

COMMENT ON TABLE dw.fact_signal_5m IS 'L5 trading signals from models (TimescaleDB hypertable)';

-- ===================
-- fact_inference_latency - Daily inference performance
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_inference_latency (
    model_sk INT NOT NULL REFERENCES dw.dim_model(model_sk),
    date_cot DATE NOT NULL,

    latency_p50_ms INT,
    latency_p95_ms INT,
    latency_p99_ms INT,
    e2e_latency_p99_ms INT,      -- End-to-end (fetch + preprocess + inference)

    throughput_eps DECIMAL(10,2), -- Episodes per second

    inference_count INT NOT NULL,

    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (model_sk, date_cot)
);

CREATE INDEX IF NOT EXISTS idx_fact_infer_lat_date ON dw.fact_inference_latency(date_cot DESC);

COMMENT ON TABLE dw.fact_inference_latency IS 'L5 daily inference latency metrics';

-- ========================================================================
-- LAYER 6 FACTS - Backtesting & Performance
-- ========================================================================

-- ===================
-- fact_trade - Individual trades
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_trade (
    run_sk INT NOT NULL REFERENCES dw.dim_backtest_run(run_sk),
    trade_id VARCHAR(100) NOT NULL,

    side VARCHAR(10) NOT NULL,    -- 'LONG', 'SHORT'

    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ NOT NULL,
    duration_bars INT NOT NULL,

    entry_px DECIMAL(12,6) NOT NULL,
    exit_px DECIMAL(12,6) NOT NULL,
    quantity DECIMAL(18,6) DEFAULT 1.0,

    pnl DECIMAL(15,6) NOT NULL,
    pnl_pct DECIMAL(10,6),
    pnl_bps DECIMAL(10,2),

    costs DECIMAL(15,6) DEFAULT 0,  -- Spread + slippage + commission

    reason_entry VARCHAR(100),
    reason_exit VARCHAR(100),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (run_sk, trade_id)
);

CREATE INDEX IF NOT EXISTS idx_fact_trade_run ON dw.fact_trade(run_sk);
CREATE INDEX IF NOT EXISTS idx_fact_trade_entry ON dw.fact_trade(entry_time);
CREATE INDEX IF NOT EXISTS idx_fact_trade_side ON dw.fact_trade(side);

COMMENT ON TABLE dw.fact_trade IS 'L6 individual trades from backtests';

-- ===================
-- fact_perf_daily - Daily performance metrics
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_perf_daily (
    run_sk INT NOT NULL REFERENCES dw.dim_backtest_run(run_sk),
    date_cot DATE NOT NULL,

    daily_return DECIMAL(12,6),
    cumulative_return DECIMAL(12,6),
    equity DECIMAL(18,2),

    trades_count INT DEFAULT 0,
    wins_count INT DEFAULT 0,
    losses_count INT DEFAULT 0,

    daily_pnl DECIMAL(15,6),
    daily_costs DECIMAL(15,6),

    drawdown_pct DECIMAL(10,6),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (run_sk, date_cot)
);

CREATE INDEX IF NOT EXISTS idx_fact_perf_daily_run ON dw.fact_perf_daily(run_sk);
CREATE INDEX IF NOT EXISTS idx_fact_perf_daily_date ON dw.fact_perf_daily(date_cot DESC);

COMMENT ON TABLE dw.fact_perf_daily IS 'L6 daily performance metrics for backtests';

-- ===================
-- fact_perf_summary - Backtest summary statistics
-- ===================
CREATE TABLE IF NOT EXISTS dw.fact_perf_summary (
    run_sk INT NOT NULL REFERENCES dw.dim_backtest_run(run_sk),
    split VARCHAR(10) NOT NULL,  -- 'train', 'val', 'test'

    -- Returns
    total_return DECIMAL(12,6),
    cagr DECIMAL(12,6),
    volatility DECIMAL(12,6),

    -- Risk-adjusted
    sharpe_ratio DECIMAL(8,4),
    sortino_ratio DECIMAL(8,4),
    calmar_ratio DECIMAL(8,4),

    -- Drawdown
    max_drawdown DECIMAL(10,6),
    max_drawdown_duration_days INT,

    -- Trades
    total_trades INT,
    win_rate DECIMAL(5,4),      -- [0, 1]
    profit_factor DECIMAL(8,4),
    avg_win DECIMAL(12,6),
    avg_loss DECIMAL(12,6),

    -- Costs
    total_costs DECIMAL(15,6),
    costs_pct_of_pnl DECIMAL(5,2),

    -- Gates
    is_production_ready BOOLEAN DEFAULT FALSE,

    dag_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (run_sk, split)
);

CREATE INDEX IF NOT EXISTS idx_fact_perf_summ_run ON dw.fact_perf_summary(run_sk);
CREATE INDEX IF NOT EXISTS idx_fact_perf_summ_split ON dw.fact_perf_summary(split);
CREATE INDEX IF NOT EXISTS idx_fact_perf_summ_prod ON dw.fact_perf_summary(is_production_ready)
    WHERE is_production_ready = TRUE;

COMMENT ON TABLE dw.fact_perf_summary IS 'L6 backtest summary statistics';

-- ========================================================================
-- COMPLETION MESSAGE
-- ========================================================================

DO $$
DECLARE
    fact_count INT;
BEGIN
    SELECT COUNT(*) INTO fact_count
    FROM pg_tables
    WHERE schemaname = 'dw'
    AND tablename LIKE 'fact_%';

    RAISE NOTICE '';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '✅ DWH Fact Tables Created Successfully!';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE 'Created % fact tables across all pipeline layers:', fact_count;
    RAISE NOTICE '';
    RAISE NOTICE 'L0 Facts (Raw Ingestion):';
    RAISE NOTICE '  - fact_bar_5m (TimescaleDB hypertable)';
    RAISE NOTICE '  - fact_l0_acquisition';
    RAISE NOTICE '';
    RAISE NOTICE 'L1 Facts (Standardization):';
    RAISE NOTICE '  - fact_l1_quality';
    RAISE NOTICE '';
    RAISE NOTICE 'L2 Facts (Technical Indicators):';
    RAISE NOTICE '  - fact_indicator_5m (partitioned by year)';
    RAISE NOTICE '  - fact_winsorization';
    RAISE NOTICE '  - fact_hod_baseline';
    RAISE NOTICE '';
    RAISE NOTICE 'L3 Facts (Feature Engineering):';
    RAISE NOTICE '  - fact_forward_ic';
    RAISE NOTICE '  - fact_leakage_tests';
    RAISE NOTICE '  - fact_feature_corr';
    RAISE NOTICE '';
    RAISE NOTICE 'L4 Facts (RL-Ready):';
    RAISE NOTICE '  - fact_rl_obs_stats';
    RAISE NOTICE '  - fact_cost_model_stats';
    RAISE NOTICE '  - fact_episode';
    RAISE NOTICE '';
    RAISE NOTICE 'L5 Facts (Model Serving):';
    RAISE NOTICE '  - fact_signal_5m (TimescaleDB hypertable)';
    RAISE NOTICE '  - fact_inference_latency';
    RAISE NOTICE '';
    RAISE NOTICE 'L6 Facts (Backtesting):';
    RAISE NOTICE '  - fact_trade';
    RAISE NOTICE '  - fact_perf_daily';
    RAISE NOTICE '  - fact_perf_summary';
    RAISE NOTICE '';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '  1. Run DAGs to populate fact tables';
    RAISE NOTICE '  2. Create data marts (dm schema) with aggregated views';
    RAISE NOTICE '  3. Set up BI API endpoints to query the DWH';
    RAISE NOTICE '========================================================================';
    RAISE NOTICE '';
END $$;
