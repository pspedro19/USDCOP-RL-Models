-- Multi-Strategy Trading System Tables
-- Compatible with existing DWH schema
-- Date: 2025-10-24

-- ============================================================
-- DIMENSION: Strategy Types
-- ============================================================

CREATE TABLE IF NOT EXISTS dw.dim_strategy (
    strategy_id SERIAL PRIMARY KEY,
    strategy_code VARCHAR(50) UNIQUE NOT NULL,
    strategy_name VARCHAR(200) NOT NULL,
    strategy_type VARCHAR(50) NOT NULL, -- 'RL', 'ML', 'LLM'
    description TEXT,
    config_json JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Insert default strategies
INSERT INTO dw.dim_strategy (strategy_code, strategy_name, strategy_type, description) VALUES
('RL_PPO', 'PPO Reinforcement Learning', 'RL', 'PPO-LSTM agent trained on L4 episodes'),
('ML_LGBM', 'LightGBM Classifier', 'ML', 'LightGBM with meta-labeling and isotonic calibration'),
('LLM_CLAUDE', 'Claude Risk Overlay', 'LLM', 'Claude Sonnet 4.5 with structured output')
ON CONFLICT (strategy_code) DO NOTHING;

-- ============================================================
-- FACT: Strategy Signals (Time-Series)
-- ============================================================

CREATE TABLE IF NOT EXISTS dw.fact_strategy_signals (
    signal_id BIGSERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    timestamp_cot TIMESTAMPTZ NOT NULL,
    strategy_id INT NOT NULL REFERENCES dw.dim_strategy(strategy_id),
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),

    -- Signal details
    signal VARCHAR(20) NOT NULL, -- 'long', 'short', 'flat', 'close'
    side VARCHAR(20), -- 'buy', 'sell', 'hold'
    size DECIMAL(10, 4), -- Position size (0.0 to 1.0)
    confidence DECIMAL(5, 4), -- 0.0 to 1.0

    -- Entry/Exit levels
    entry_price DECIMAL(12, 4),
    stop_loss DECIMAL(12, 4),
    take_profit DECIMAL(12, 4),

    -- Risk metrics
    risk_usd DECIMAL(12, 2),
    notional_usd DECIMAL(12, 2),
    leverage INT DEFAULT 1,

    -- Metadata
    reasoning TEXT,
    invalidation_condition TEXT,
    features_snapshot JSONB,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_signal CHECK (signal IN ('long', 'short', 'flat', 'close')),
    CONSTRAINT valid_size CHECK (size >= 0 AND size <= 1),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1)
);

-- Hypertable for time-series
SELECT create_hypertable(
    'dw.fact_strategy_signals',
    'timestamp_utc',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_strategy_signals_time_strategy
ON dw.fact_strategy_signals (timestamp_utc DESC, strategy_id);

CREATE INDEX IF NOT EXISTS idx_strategy_signals_symbol
ON dw.fact_strategy_signals (symbol_id, timestamp_utc DESC);

-- ============================================================
-- FACT: Strategy Positions (Current State)
-- ============================================================

CREATE TABLE IF NOT EXISTS dw.fact_strategy_positions (
    position_id BIGSERIAL PRIMARY KEY,
    strategy_id INT NOT NULL REFERENCES dw.dim_strategy(strategy_id),
    symbol_id INT NOT NULL REFERENCES dw.dim_symbol(symbol_id),

    -- Position details
    side VARCHAR(20) NOT NULL, -- 'long', 'short'
    quantity DECIMAL(18, 8) NOT NULL,
    entry_price DECIMAL(12, 4) NOT NULL,
    current_price DECIMAL(12, 4),

    -- Risk management
    stop_loss DECIMAL(12, 4),
    take_profit DECIMAL(12, 4),
    leverage INT DEFAULT 1,

    -- P&L
    unrealized_pnl DECIMAL(12, 2),
    realized_pnl DECIMAL(12, 2) DEFAULT 0,

    -- Timing
    entry_time TIMESTAMPTZ NOT NULL,
    exit_time TIMESTAMPTZ,
    holding_time_minutes INT,

    -- Status
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'stopped', 'liquidated'
    exit_reason VARCHAR(50),

    -- Metadata
    entry_signal_id BIGINT REFERENCES dw.fact_strategy_signals(signal_id),
    metadata_json JSONB,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    CONSTRAINT valid_side CHECK (side IN ('long', 'short')),
    CONSTRAINT valid_status CHECK (status IN ('open', 'closed', 'stopped', 'liquidated'))
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_positions_strategy_status
ON dw.fact_strategy_positions (strategy_id, status);

CREATE INDEX IF NOT EXISTS idx_positions_entry_time
ON dw.fact_strategy_positions (entry_time DESC);

-- ============================================================
-- FACT: Strategy Performance (Daily Grain)
-- ============================================================

CREATE TABLE IF NOT EXISTS dw.fact_strategy_performance (
    perf_id SERIAL PRIMARY KEY,
    date_cot DATE NOT NULL,
    strategy_id INT NOT NULL REFERENCES dw.dim_strategy(strategy_id),

    -- Account metrics
    starting_capital DECIMAL(12, 2) NOT NULL,
    ending_capital DECIMAL(12, 2) NOT NULL,
    daily_return_pct DECIMAL(8, 4),
    cumulative_return_pct DECIMAL(10, 4),

    -- Trading activity
    n_trades INT DEFAULT 0,
    n_wins INT DEFAULT 0,
    n_losses INT DEFAULT 0,
    win_rate DECIMAL(5, 4),

    -- P&L
    gross_profit DECIMAL(12, 2) DEFAULT 0,
    gross_loss DECIMAL(12, 2) DEFAULT 0,
    net_profit DECIMAL(12, 2) DEFAULT 0,
    total_fees DECIMAL(12, 2) DEFAULT 0,

    -- Risk metrics
    max_drawdown_pct DECIMAL(8, 4),
    sharpe_ratio DECIMAL(8, 4),
    sortino_ratio DECIMAL(8, 4),
    calmar_ratio DECIMAL(8, 4),

    -- Position metrics
    avg_hold_time_minutes DECIMAL(10, 2),
    max_leverage_used INT,
    avg_position_size DECIMAL(10, 4),

    -- Metadata
    trades_json JSONB,
    metadata_json JSONB,

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW()

    -- Note: UNIQUE constraint removed for hypertable compatibility
    -- Uniqueness enforced at application level
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_perf_date_strategy
ON dw.fact_strategy_performance (date_cot DESC, strategy_id);

-- ============================================================
-- FACT: Equity Curve (Intraday Time-Series)
-- ============================================================

CREATE TABLE IF NOT EXISTS dw.fact_equity_curve (
    equity_id BIGSERIAL PRIMARY KEY,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    timestamp_cot TIMESTAMPTZ NOT NULL,
    strategy_id INT NOT NULL REFERENCES dw.dim_strategy(strategy_id),

    -- Equity metrics
    equity_value DECIMAL(12, 2) NOT NULL,
    cash_balance DECIMAL(12, 2),
    positions_value DECIMAL(12, 2),
    unrealized_pnl DECIMAL(12, 2),

    -- Returns
    return_since_start_pct DECIMAL(10, 4),
    return_daily_pct DECIMAL(8, 4),

    -- Risk
    current_drawdown_pct DECIMAL(8, 4),
    volatility_20 DECIMAL(8, 4),

    -- Audit
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable
SELECT create_hypertable(
    'dw.fact_equity_curve',
    'timestamp_utc',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_equity_time_strategy
ON dw.fact_equity_curve (timestamp_utc DESC, strategy_id);

-- ============================================================
-- VIEW: Portfolio Summary
-- ============================================================

CREATE OR REPLACE VIEW dw.vw_portfolio_summary AS
SELECT
    s.strategy_code,
    s.strategy_name,
    s.strategy_type,

    -- Latest equity
    (SELECT equity_value
     FROM dw.fact_equity_curve ec
     WHERE ec.strategy_id = s.strategy_id
     ORDER BY timestamp_utc DESC LIMIT 1) as current_equity,

    -- Open positions
    (SELECT COUNT(*)
     FROM dw.fact_strategy_positions p
     WHERE p.strategy_id = s.strategy_id
     AND p.status = 'open') as open_positions,

    -- Today's performance
    (SELECT daily_return_pct
     FROM dw.fact_strategy_performance perf
     WHERE perf.strategy_id = s.strategy_id
     AND perf.date_cot = CURRENT_DATE) as today_return_pct,

    -- Total trades
    (SELECT SUM(n_trades)
     FROM dw.fact_strategy_performance perf
     WHERE perf.strategy_id = s.strategy_id) as total_trades,

    -- Performance metrics
    (SELECT sharpe_ratio
     FROM dw.fact_strategy_performance perf
     WHERE perf.strategy_id = s.strategy_id
     ORDER BY date_cot DESC LIMIT 1) as latest_sharpe

FROM dw.dim_strategy s
WHERE s.is_active = TRUE;

-- ============================================================
-- GRANT PERMISSIONS
-- ============================================================

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA dw TO admin;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA dw TO admin;

-- ============================================================
-- COMMENTS
-- ============================================================

COMMENT ON TABLE dw.dim_strategy IS 'Dimension table for trading strategies (RL, ML, LLM)';
COMMENT ON TABLE dw.fact_strategy_signals IS 'Time-series of signals generated by each strategy';
COMMENT ON TABLE dw.fact_strategy_positions IS 'Current and historical positions per strategy';
COMMENT ON TABLE dw.fact_strategy_performance IS 'Daily performance metrics per strategy';
COMMENT ON TABLE dw.fact_equity_curve IS 'Intraday equity curve time-series per strategy';
