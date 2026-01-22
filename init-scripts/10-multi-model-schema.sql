-- =====================================================
-- USDCOP Multi-Model Trading System - Database Schema
-- =====================================================
-- Version: 1.0.0
-- Date: 2025-12-26
-- Author: Pedro @ Lean Tech Solutions
-- Description: Comprehensive schema for multi-model RL trading system
--              supporting PPO, SAC, TD3, A2C algorithms with full
--              configuration, inference tracking, and performance metrics.
-- =====================================================
--
-- CHANGE LOG:
-- v1.0.0 (2025-12-26): Initial production schema
--   - 4 schemas: config, trading, events, metrics
--   - Model configuration with hyperparameters
--   - Feature definitions with normalization stats
--   - Inference results with partitioning support
--   - Trade tracking with PnL and costs
--   - Event streaming for audit trail
--   - Performance metrics aggregation
--   - Pre-seeded with PPO V19 Model B configuration
--   - 13 core features from training dataset
--
-- =====================================================

-- =============================================================================
-- SECTION 1: SCHEMA CREATION
-- =============================================================================

-- Create dedicated schemas for organization
CREATE SCHEMA IF NOT EXISTS config;
COMMENT ON SCHEMA config IS 'Configuration tables for models, features, and system settings';

CREATE SCHEMA IF NOT EXISTS trading;
COMMENT ON SCHEMA trading IS 'Trading operations: inferences, trades, positions';

CREATE SCHEMA IF NOT EXISTS events;
COMMENT ON SCHEMA events IS 'Event streaming and audit trail for compliance';

CREATE SCHEMA IF NOT EXISTS metrics;
COMMENT ON SCHEMA metrics IS 'Performance metrics and aggregated statistics';

-- =============================================================================
-- SECTION 2: CONFIG SCHEMA - Models Configuration Table
-- =============================================================================

-- Drop existing table if migrating (comment out in production)
-- DROP TABLE IF EXISTS config.models CASCADE;

CREATE TABLE IF NOT EXISTS config.models (
    -- Primary identification
    model_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    algorithm VARCHAR(20) NOT NULL,
    version VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'inactive',

    -- UI configuration
    color VARCHAR(7) DEFAULT '#3B82F6',

    -- Hyperparameters (PPO, SAC, TD3, A2C compatible)
    hyperparameters JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Policy network configuration
    policy_config JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Environment configuration
    environment_config JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Backtest performance metrics
    backtest_metrics JSONB DEFAULT '{}'::jsonb,

    -- Model artifacts
    model_path VARCHAR(500),
    framework VARCHAR(50) DEFAULT 'stable-baselines3',

    -- Metadata
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_model_algorithm CHECK (algorithm IN ('PPO', 'SAC', 'TD3', 'A2C', 'DQN', 'DDPG', 'SYNTHETIC')),
    CONSTRAINT chk_model_status CHECK (status IN ('active', 'inactive', 'training', 'deprecated', 'testing')),
    CONSTRAINT chk_model_color CHECK (color ~ '^#[0-9A-Fa-f]{6}$')
);

-- Add comments for documentation
COMMENT ON TABLE config.models IS 'Configuration for all RL trading models including hyperparameters, policy architecture, and performance metrics';
COMMENT ON COLUMN config.models.model_id IS 'Unique identifier for the model (e.g., ppo_v19_model_b)';
COMMENT ON COLUMN config.models.name IS 'Human-readable model name (e.g., Model B Aggressive)';
COMMENT ON COLUMN config.models.algorithm IS 'RL algorithm: PPO, SAC, TD3, A2C, DQN, DDPG';
COMMENT ON COLUMN config.models.version IS 'Model version string (e.g., V19, V20)';
COMMENT ON COLUMN config.models.status IS 'Current status: active, inactive, training, deprecated, testing';
COMMENT ON COLUMN config.models.color IS 'Hex color code for UI visualization';
COMMENT ON COLUMN config.models.hyperparameters IS 'JSON with learning_rate, n_steps, batch_size, n_epochs, gamma, ent_coef, clip_range, gae_lambda, max_grad_norm, vf_coef';
COMMENT ON COLUMN config.models.policy_config IS 'JSON with net_arch, activation_fn, policy type';
COMMENT ON COLUMN config.models.environment_config IS 'JSON with episode_length, max_position, bars_per_day, initial_balance';
COMMENT ON COLUMN config.models.backtest_metrics IS 'JSON with sharpe_ratio, max_drawdown, win_rate, hold_pct, total_return';
COMMENT ON COLUMN config.models.model_path IS 'Path to model artifact file (.zip)';
COMMENT ON COLUMN config.models.framework IS 'ML framework used (stable-baselines3, rllib, etc.)';

-- Index for quick lookups
CREATE INDEX IF NOT EXISTS idx_models_status ON config.models (status);
CREATE INDEX IF NOT EXISTS idx_models_algorithm ON config.models (algorithm);
CREATE INDEX IF NOT EXISTS idx_models_updated ON config.models (updated_at DESC);

-- =============================================================================
-- SECTION 3: CONFIG SCHEMA - Feature Definitions Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS config.feature_definitions (
    -- Primary identification
    feature_id SERIAL PRIMARY KEY,
    feature_name VARCHAR(100) NOT NULL UNIQUE,

    -- Feature categorization
    feature_group VARCHAR(50) NOT NULL,

    -- Normalization statistics (from training dataset)
    normalization_mean DOUBLE PRECISION,
    normalization_std DOUBLE PRECISION,
    clip_min DOUBLE PRECISION,
    clip_max DOUBLE PRECISION,

    -- Data source
    source_table VARCHAR(100),
    source_column VARCHAR(100),
    transformation VARCHAR(200),

    -- Observation space ordering
    observation_order INTEGER NOT NULL,

    -- Compute strategy
    compute_location VARCHAR(20) DEFAULT 'sql',
    python_function VARCHAR(200),
    sql_formula TEXT,

    -- Feature metadata
    lookback_bars INTEGER DEFAULT 1,
    description TEXT,

    -- Status
    is_active BOOLEAN DEFAULT TRUE,

    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    -- Updated 2025-12-26: Added temporal, macro_momentum, macro_volatility for V19 features
    CONSTRAINT chk_feature_group CHECK (feature_group IN (
        'price_returns', 'momentum', 'volatility', 'trend',
        'macro_zscore', 'macro_changes', 'macro_derived', 'macro_momentum', 'macro_volatility',
        'temporal', 'regime', 'state'
    )),
    CONSTRAINT chk_compute_location CHECK (compute_location IN ('sql', 'python', 'hybrid')),
    CONSTRAINT chk_observation_order_unique UNIQUE (observation_order) DEFERRABLE INITIALLY DEFERRED
);

-- Add comments for documentation
COMMENT ON TABLE config.feature_definitions IS 'Feature definitions with normalization statistics for observation space. Order matches model training.';
COMMENT ON COLUMN config.feature_definitions.feature_name IS 'Unique feature identifier (e.g., log_ret_5m, rsi_9)';
COMMENT ON COLUMN config.feature_definitions.feature_group IS 'Category: price_returns, momentum, volatility, trend, macro_zscore, macro_changes, macro_derived, regime, state';
COMMENT ON COLUMN config.feature_definitions.normalization_mean IS 'Mean value from training dataset for z-score normalization';
COMMENT ON COLUMN config.feature_definitions.normalization_std IS 'Standard deviation from training dataset for z-score normalization';
COMMENT ON COLUMN config.feature_definitions.clip_min IS 'Minimum clip value after transformation';
COMMENT ON COLUMN config.feature_definitions.clip_max IS 'Maximum clip value after transformation';
COMMENT ON COLUMN config.feature_definitions.source_table IS 'Source database table (usdcop_m5_ohlcv, macro_indicators_daily)';
COMMENT ON COLUMN config.feature_definitions.source_column IS 'Source column name in the table';
COMMENT ON COLUMN config.feature_definitions.transformation IS 'Transformation type: zscore, pct_change, raw, etc.';
COMMENT ON COLUMN config.feature_definitions.observation_order IS 'Position in observation vector (0-indexed, must match model training)';
COMMENT ON COLUMN config.feature_definitions.compute_location IS 'Where feature is computed: sql, python, hybrid';
COMMENT ON COLUMN config.feature_definitions.python_function IS 'Python function name if compute_location is python';
COMMENT ON COLUMN config.feature_definitions.sql_formula IS 'SQL formula if compute_location is sql';
COMMENT ON COLUMN config.feature_definitions.lookback_bars IS 'Number of historical bars needed for computation';
COMMENT ON COLUMN config.feature_definitions.is_active IS 'Whether feature is currently used in production';

-- Indexes
CREATE INDEX IF NOT EXISTS idx_features_group ON config.feature_definitions (feature_group);
CREATE INDEX IF NOT EXISTS idx_features_order ON config.feature_definitions (observation_order);
CREATE INDEX IF NOT EXISTS idx_features_active ON config.feature_definitions (is_active) WHERE is_active = TRUE;

-- =============================================================================
-- SECTION 4: TRADING SCHEMA - Model Inferences Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS trading.model_inferences (
    -- Primary identification (composite key for partitioning)
    inference_id BIGSERIAL,
    timestamp_utc TIMESTAMPTZ NOT NULL,

    -- Model reference
    model_id VARCHAR(50) NOT NULL,

    -- Action output
    action_raw DOUBLE PRECISION NOT NULL,
    action_discretized VARCHAR(10) NOT NULL,
    confidence DOUBLE PRECISION,

    -- Market context
    price_at_inference NUMERIC(12, 4) NOT NULL,
    spread_pips NUMERIC(8, 2),

    -- Feature snapshot for audit
    features_snapshot JSONB,

    -- Performance metrics
    latency_ms INTEGER,
    preprocessing_ms INTEGER,
    inference_ms INTEGER,

    -- Environment state
    current_position DOUBLE PRECISION DEFAULT 0.0,
    time_normalized DOUBLE PRECISION,
    bar_number INTEGER,

    -- Metadata
    session_id VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Primary key supports time-based partitioning
    PRIMARY KEY (timestamp_utc, inference_id),

    -- Constraints
    CONSTRAINT chk_action_discretized CHECK (action_discretized IN ('LONG', 'SHORT', 'HOLD', 'CLOSE')),
    CONSTRAINT chk_confidence CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
    CONSTRAINT chk_position CHECK (current_position >= -1 AND current_position <= 1),
    CONSTRAINT fk_model_id FOREIGN KEY (model_id) REFERENCES config.models(model_id) ON DELETE RESTRICT
);

-- Add comments for documentation
COMMENT ON TABLE trading.model_inferences IS 'Real-time inference results from all models. Partitioned by timestamp for efficient querying.';
COMMENT ON COLUMN trading.model_inferences.inference_id IS 'Auto-incrementing inference identifier';
COMMENT ON COLUMN trading.model_inferences.timestamp_utc IS 'UTC timestamp of inference execution';
COMMENT ON COLUMN trading.model_inferences.model_id IS 'Reference to config.models';
COMMENT ON COLUMN trading.model_inferences.action_raw IS 'Raw continuous action output from model [-1, 1]';
COMMENT ON COLUMN trading.model_inferences.action_discretized IS 'Discretized action: LONG, SHORT, HOLD, CLOSE';
COMMENT ON COLUMN trading.model_inferences.confidence IS 'Model confidence in action (0-1), derived from action magnitude or ensemble agreement';
COMMENT ON COLUMN trading.model_inferences.price_at_inference IS 'USD/COP price at inference time';
COMMENT ON COLUMN trading.model_inferences.spread_pips IS 'Bid-ask spread in pips';
COMMENT ON COLUMN trading.model_inferences.features_snapshot IS 'JSON snapshot of all 13 features for audit and debugging';
COMMENT ON COLUMN trading.model_inferences.latency_ms IS 'Total end-to-end latency in milliseconds';
COMMENT ON COLUMN trading.model_inferences.preprocessing_ms IS 'Feature preprocessing time in milliseconds';
COMMENT ON COLUMN trading.model_inferences.inference_ms IS 'Model inference time in milliseconds';
COMMENT ON COLUMN trading.model_inferences.current_position IS 'Position state at inference: -1 (short) to 1 (long)';
COMMENT ON COLUMN trading.model_inferences.time_normalized IS 'Normalized time within episode (0 to ~0.983)';
COMMENT ON COLUMN trading.model_inferences.bar_number IS 'Bar number within trading session';
COMMENT ON COLUMN trading.model_inferences.session_id IS 'Trading session identifier for grouping';

-- Convert to hypertable for time-series optimization (if TimescaleDB available)
-- Uncomment the following line if using TimescaleDB:
-- SELECT create_hypertable('trading.model_inferences', 'timestamp_utc', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_inferences_model_time ON trading.model_inferences (model_id, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_inferences_time ON trading.model_inferences (timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_inferences_action ON trading.model_inferences (action_discretized, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_inferences_session ON trading.model_inferences (session_id, timestamp_utc DESC);

-- =============================================================================
-- SECTION 5: TRADING SCHEMA - Model Trades Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS trading.model_trades (
    -- Primary identification
    trade_id BIGSERIAL PRIMARY KEY,

    -- Model reference
    model_id VARCHAR(50) NOT NULL,

    -- Trade timing
    open_time TIMESTAMPTZ NOT NULL,
    close_time TIMESTAMPTZ,

    -- Trade details
    signal VARCHAR(10) NOT NULL,
    entry_price NUMERIC(12, 4) NOT NULL,
    exit_price NUMERIC(12, 4),
    position_size NUMERIC(8, 4) NOT NULL DEFAULT 1.0,

    -- Profit/Loss
    pnl NUMERIC(12, 4),
    pnl_pct NUMERIC(12, 6),
    gross_pnl NUMERIC(12, 4),

    -- Trade duration
    duration_minutes INTEGER,
    duration_bars INTEGER,

    -- Status tracking
    status VARCHAR(20) DEFAULT 'open',
    exit_reason VARCHAR(50),

    -- Confidence and risk
    entry_confidence NUMERIC(4, 3),
    exit_confidence NUMERIC(4, 3),
    max_adverse_excursion NUMERIC(12, 4),
    max_favorable_excursion NUMERIC(12, 4),

    -- Transaction costs
    transaction_costs NUMERIC(8, 4),
    spread_cost NUMERIC(8, 4),
    slippage NUMERIC(8, 4),

    -- Reference to inferences
    entry_inference_id BIGINT,
    exit_inference_id BIGINT,

    -- Metadata
    notes TEXT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Constraints
    CONSTRAINT chk_trade_signal CHECK (signal IN ('LONG', 'SHORT')),
    CONSTRAINT chk_trade_status CHECK (status IN ('open', 'closed', 'cancelled', 'stopped', 'expired')),
    CONSTRAINT chk_position_size CHECK (position_size > 0 AND position_size <= 1),
    CONSTRAINT chk_exit_reason CHECK (exit_reason IS NULL OR exit_reason IN (
        'signal', 'stop_loss', 'take_profit', 'timeout', 'manual', 'system', 'end_of_session'
    )),
    CONSTRAINT fk_trade_model FOREIGN KEY (model_id) REFERENCES config.models(model_id) ON DELETE RESTRICT
);

-- Add comments for documentation
COMMENT ON TABLE trading.model_trades IS 'Completed and open trades for each model with full PnL tracking and cost accounting';
COMMENT ON COLUMN trading.model_trades.trade_id IS 'Unique trade identifier';
COMMENT ON COLUMN trading.model_trades.model_id IS 'Reference to config.models';
COMMENT ON COLUMN trading.model_trades.open_time IS 'UTC timestamp when trade was opened';
COMMENT ON COLUMN trading.model_trades.close_time IS 'UTC timestamp when trade was closed (NULL if open)';
COMMENT ON COLUMN trading.model_trades.signal IS 'Trade direction: LONG or SHORT';
COMMENT ON COLUMN trading.model_trades.entry_price IS 'Price at trade entry';
COMMENT ON COLUMN trading.model_trades.exit_price IS 'Price at trade exit (NULL if open)';
COMMENT ON COLUMN trading.model_trades.position_size IS 'Position size as fraction of capital (0-1)';
COMMENT ON COLUMN trading.model_trades.pnl IS 'Net profit/loss in USD after costs';
COMMENT ON COLUMN trading.model_trades.pnl_pct IS 'Net profit/loss as percentage';
COMMENT ON COLUMN trading.model_trades.gross_pnl IS 'Gross profit/loss before costs';
COMMENT ON COLUMN trading.model_trades.duration_minutes IS 'Trade duration in minutes';
COMMENT ON COLUMN trading.model_trades.duration_bars IS 'Trade duration in 5-minute bars';
COMMENT ON COLUMN trading.model_trades.status IS 'Trade status: open, closed, cancelled, stopped, expired';
COMMENT ON COLUMN trading.model_trades.exit_reason IS 'Reason for exit: signal, stop_loss, take_profit, timeout, manual, system, end_of_session';
COMMENT ON COLUMN trading.model_trades.entry_confidence IS 'Model confidence at entry (0-1)';
COMMENT ON COLUMN trading.model_trades.exit_confidence IS 'Model confidence at exit (0-1)';
COMMENT ON COLUMN trading.model_trades.max_adverse_excursion IS 'Maximum adverse price movement during trade (MAE)';
COMMENT ON COLUMN trading.model_trades.max_favorable_excursion IS 'Maximum favorable price movement during trade (MFE)';
COMMENT ON COLUMN trading.model_trades.transaction_costs IS 'Total transaction costs in pesos';
COMMENT ON COLUMN trading.model_trades.spread_cost IS 'Cost due to bid-ask spread';
COMMENT ON COLUMN trading.model_trades.slippage IS 'Cost due to slippage';

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_trades_model_time ON trading.model_trades (model_id, open_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trading.model_trades (status, model_id);
CREATE INDEX IF NOT EXISTS idx_trades_open_time ON trading.model_trades (open_time DESC);
CREATE INDEX IF NOT EXISTS idx_trades_close_time ON trading.model_trades (close_time DESC) WHERE close_time IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_signal ON trading.model_trades (signal, model_id);

-- =============================================================================
-- SECTION 6: EVENTS SCHEMA - Signals Stream Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS events.signals_stream (
    -- Primary identification
    stream_id BIGSERIAL,
    timestamp_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Event classification
    event_type VARCHAR(50) NOT NULL,
    event_subtype VARCHAR(50),

    -- Source identification
    model_id VARCHAR(50),
    source_system VARCHAR(50) DEFAULT 'trading_system',

    -- Event payload
    payload JSONB NOT NULL,

    -- Event context
    price_at_event NUMERIC(12, 4),
    position_at_event DOUBLE PRECISION,

    -- Correlation and tracing
    correlation_id VARCHAR(100),
    trace_id VARCHAR(100),

    -- Processing status
    processed BOOLEAN DEFAULT FALSE,
    processed_at TIMESTAMPTZ,

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Primary key for partitioning
    PRIMARY KEY (timestamp_utc, stream_id),

    -- Constraints
    CONSTRAINT chk_event_type CHECK (event_type IN (
        'INFERENCE', 'SIGNAL', 'TRADE_OPEN', 'TRADE_CLOSE',
        'POSITION_CHANGE', 'RISK_ALERT', 'SYSTEM_EVENT',
        'MODEL_LOAD', 'MODEL_ERROR', 'DATA_QUALITY', 'HEARTBEAT'
    ))
);

-- Add comments for documentation
COMMENT ON TABLE events.signals_stream IS 'Immutable event stream for audit trail and real-time streaming. Supports compliance and debugging.';
COMMENT ON COLUMN events.signals_stream.stream_id IS 'Auto-incrementing stream identifier';
COMMENT ON COLUMN events.signals_stream.timestamp_utc IS 'UTC timestamp of event';
COMMENT ON COLUMN events.signals_stream.event_type IS 'Event category: INFERENCE, SIGNAL, TRADE_OPEN, TRADE_CLOSE, POSITION_CHANGE, RISK_ALERT, SYSTEM_EVENT, MODEL_LOAD, MODEL_ERROR, DATA_QUALITY, HEARTBEAT';
COMMENT ON COLUMN events.signals_stream.event_subtype IS 'Event subcategory for filtering';
COMMENT ON COLUMN events.signals_stream.model_id IS 'Associated model (if applicable)';
COMMENT ON COLUMN events.signals_stream.source_system IS 'System that generated the event';
COMMENT ON COLUMN events.signals_stream.payload IS 'Full event payload as JSON';
COMMENT ON COLUMN events.signals_stream.price_at_event IS 'USD/COP price at event time';
COMMENT ON COLUMN events.signals_stream.position_at_event IS 'Position state at event time';
COMMENT ON COLUMN events.signals_stream.correlation_id IS 'ID for correlating related events';
COMMENT ON COLUMN events.signals_stream.trace_id IS 'Distributed tracing ID';
COMMENT ON COLUMN events.signals_stream.processed IS 'Whether event has been processed by downstream systems';
COMMENT ON COLUMN events.signals_stream.processed_at IS 'When event was processed';

-- Convert to hypertable (if TimescaleDB available)
-- SELECT create_hypertable('events.signals_stream', 'timestamp_utc', if_not_exists => TRUE, chunk_time_interval => INTERVAL '1 day');

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_events_type_time ON events.signals_stream (event_type, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_events_model_time ON events.signals_stream (model_id, timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_events_time ON events.signals_stream (timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_events_unprocessed ON events.signals_stream (processed, timestamp_utc DESC) WHERE processed = FALSE;
CREATE INDEX IF NOT EXISTS idx_events_correlation ON events.signals_stream (correlation_id) WHERE correlation_id IS NOT NULL;

-- =============================================================================
-- SECTION 7: METRICS SCHEMA - Model Performance Table
-- =============================================================================

CREATE TABLE IF NOT EXISTS metrics.model_performance (
    -- Primary identification
    perf_id BIGSERIAL,
    period_start TIMESTAMPTZ NOT NULL,
    period_end TIMESTAMPTZ NOT NULL,

    -- Aggregation level
    aggregation_type VARCHAR(20) NOT NULL DEFAULT 'daily',

    -- Model reference
    model_id VARCHAR(50) NOT NULL,

    -- Return metrics
    pnl_cumulative NUMERIC(12, 4),
    pnl_period NUMERIC(12, 4),
    return_pct NUMERIC(8, 4),
    return_cumulative_pct NUMERIC(10, 4),

    -- Risk metrics
    sharpe_ratio NUMERIC(8, 4),
    sortino_ratio NUMERIC(8, 4),
    calmar_ratio NUMERIC(8, 4),
    max_drawdown NUMERIC(8, 4),
    current_drawdown NUMERIC(8, 4),
    volatility NUMERIC(8, 4),

    -- Trading activity
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    win_rate NUMERIC(5, 4),

    -- Average metrics
    avg_trade_pnl NUMERIC(12, 4),
    avg_win NUMERIC(12, 4),
    avg_loss NUMERIC(12, 4),
    profit_factor NUMERIC(8, 4),

    -- Position metrics
    avg_position_size NUMERIC(8, 4),
    hold_percentage NUMERIC(5, 4),
    avg_trade_duration_minutes NUMERIC(10, 2),

    -- Inference metrics
    total_inferences INTEGER DEFAULT 0,
    avg_latency_ms NUMERIC(8, 2),

    -- Cost metrics
    total_transaction_costs NUMERIC(12, 4),
    total_slippage NUMERIC(12, 4),

    -- Equity tracking
    starting_equity NUMERIC(15, 2),
    ending_equity NUMERIC(15, 2),
    peak_equity NUMERIC(15, 2),

    -- Metadata
    calculated_at TIMESTAMPTZ DEFAULT NOW(),

    -- Primary key
    PRIMARY KEY (period_start, model_id, aggregation_type),

    -- Constraints
    CONSTRAINT chk_aggregation CHECK (aggregation_type IN ('hourly', 'daily', 'weekly', 'monthly')),
    CONSTRAINT chk_win_rate CHECK (win_rate IS NULL OR (win_rate >= 0 AND win_rate <= 1)),
    CONSTRAINT chk_hold_pct CHECK (hold_percentage IS NULL OR (hold_percentage >= 0 AND hold_percentage <= 1)),
    CONSTRAINT fk_perf_model FOREIGN KEY (model_id) REFERENCES config.models(model_id) ON DELETE CASCADE
);

-- Add comments for documentation
COMMENT ON TABLE metrics.model_performance IS 'Aggregated performance metrics per model at different time granularities (hourly, daily, weekly, monthly)';
COMMENT ON COLUMN metrics.model_performance.perf_id IS 'Auto-incrementing performance record identifier';
COMMENT ON COLUMN metrics.model_performance.period_start IS 'Start of aggregation period';
COMMENT ON COLUMN metrics.model_performance.period_end IS 'End of aggregation period';
COMMENT ON COLUMN metrics.model_performance.aggregation_type IS 'Time granularity: hourly, daily, weekly, monthly';
COMMENT ON COLUMN metrics.model_performance.model_id IS 'Reference to config.models';
COMMENT ON COLUMN metrics.model_performance.pnl_cumulative IS 'Cumulative PnL since model inception';
COMMENT ON COLUMN metrics.model_performance.pnl_period IS 'PnL for this period';
COMMENT ON COLUMN metrics.model_performance.return_pct IS 'Return percentage for this period';
COMMENT ON COLUMN metrics.model_performance.return_cumulative_pct IS 'Cumulative return percentage';
COMMENT ON COLUMN metrics.model_performance.sharpe_ratio IS 'Sharpe ratio (risk-adjusted return)';
COMMENT ON COLUMN metrics.model_performance.sortino_ratio IS 'Sortino ratio (downside risk-adjusted return)';
COMMENT ON COLUMN metrics.model_performance.calmar_ratio IS 'Calmar ratio (return / max drawdown)';
COMMENT ON COLUMN metrics.model_performance.max_drawdown IS 'Maximum drawdown as decimal (e.g., 0.05 = 5%)';
COMMENT ON COLUMN metrics.model_performance.current_drawdown IS 'Current drawdown from peak';
COMMENT ON COLUMN metrics.model_performance.volatility IS 'Return volatility (standard deviation)';
COMMENT ON COLUMN metrics.model_performance.total_trades IS 'Number of trades in period';
COMMENT ON COLUMN metrics.model_performance.winning_trades IS 'Number of winning trades';
COMMENT ON COLUMN metrics.model_performance.losing_trades IS 'Number of losing trades';
COMMENT ON COLUMN metrics.model_performance.win_rate IS 'Win rate as decimal (0-1)';
COMMENT ON COLUMN metrics.model_performance.avg_trade_pnl IS 'Average PnL per trade';
COMMENT ON COLUMN metrics.model_performance.avg_win IS 'Average winning trade PnL';
COMMENT ON COLUMN metrics.model_performance.avg_loss IS 'Average losing trade PnL';
COMMENT ON COLUMN metrics.model_performance.profit_factor IS 'Gross profit / gross loss';
COMMENT ON COLUMN metrics.model_performance.avg_position_size IS 'Average position size (0-1)';
COMMENT ON COLUMN metrics.model_performance.hold_percentage IS 'Percentage of time in position';
COMMENT ON COLUMN metrics.model_performance.avg_trade_duration_minutes IS 'Average trade duration in minutes';
COMMENT ON COLUMN metrics.model_performance.total_inferences IS 'Total model inferences in period';
COMMENT ON COLUMN metrics.model_performance.avg_latency_ms IS 'Average inference latency in milliseconds';
COMMENT ON COLUMN metrics.model_performance.total_transaction_costs IS 'Total transaction costs in period';
COMMENT ON COLUMN metrics.model_performance.total_slippage IS 'Total slippage in period';
COMMENT ON COLUMN metrics.model_performance.starting_equity IS 'Equity at period start';
COMMENT ON COLUMN metrics.model_performance.ending_equity IS 'Equity at period end';
COMMENT ON COLUMN metrics.model_performance.peak_equity IS 'Peak equity during period';

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_perf_model_period ON metrics.model_performance (model_id, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_perf_aggregation ON metrics.model_performance (aggregation_type, period_start DESC);
CREATE INDEX IF NOT EXISTS idx_perf_sharpe ON metrics.model_performance (sharpe_ratio DESC NULLS LAST);

-- =============================================================================
-- SECTION 8: SEED DATA - PPO V19 Model B Configuration
-- =============================================================================

-- =============================================================================
-- PPO PRIMARY - PRODUCTION MODEL (loaded by InferenceEngine)
-- =============================================================================
-- This is the model_id that InferenceEngine loads by default
INSERT INTO config.models (
    model_id,
    name,
    algorithm,
    version,
    status,
    color,
    hyperparameters,
    policy_config,
    environment_config,
    backtest_metrics,
    model_path,
    framework,
    description
) VALUES (
    'ppo_primary',
    'PPO Primary (Production)',
    'PPO',
    'V20',
    'active',
    '#10B981',
    '{
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5
    }'::jsonb,
    '{
        "type": "MlpPolicy",
        "net_arch": [64, 64],
        "activation_fn": "Tanh"
    }'::jsonb,
    '{
        "class": "TradingEnvironmentV19",
        "initial_balance": 10000,
        "max_position": 1.0,
        "episode_length": 1200,
        "use_vol_scaling": true,
        "use_regime_detection": true,
        "bars_per_day": 56,
        "observation_space_dim": 15
    }'::jsonb,
    '{
        "dataset": "RL_DS3_MACRO_CORE.csv",
        "norm_stats": "config/norm_stats.json"
    }'::jsonb,
    '/opt/airflow/models/ppo_v20_production/final_model.zip',
    'stable-baselines3',
    'Primary production model - loaded by InferenceEngine on startup. PPO V20 with 15-dim observation space.'
) ON CONFLICT (model_id) DO UPDATE SET
    hyperparameters = EXCLUDED.hyperparameters,
    policy_config = EXCLUDED.policy_config,
    environment_config = EXCLUDED.environment_config,
    backtest_metrics = EXCLUDED.backtest_metrics,
    model_path = EXCLUDED.model_path,
    status = 'active',
    updated_at = NOW();

-- =============================================================================
-- INVESTOR DEMO MODEL - For investor presentations (SSOT for demo mode)
-- =============================================================================
-- This model generates synthetic trades with optimized metrics for presentations.
-- Select this model_id in the UI to enable demo mode - no separate flag needed (DRY).
INSERT INTO config.models (
    model_id,
    name,
    algorithm,
    version,
    status,
    color,
    hyperparameters,
    policy_config,
    environment_config,
    backtest_metrics,
    model_path,
    framework,
    description
) VALUES (
    'investor_demo',
    'Demo Mode (Investor Presentation)',
    'SYNTHETIC',
    'V1',
    'active',
    '#F59E0B',
    '{
        "target_sharpe": 2.1,
        "target_max_drawdown": -0.095,
        "target_win_rate": 0.61,
        "target_annual_return": 0.32,
        "target_profit_factor": 1.85,
        "trades_per_month": 18
    }'::jsonb,
    '{
        "type": "SyntheticTradeGenerator",
        "avg_win_pct": 0.005,
        "avg_loss_pct": 0.004,
        "max_position_duration_minutes": 300,
        "min_position_duration_minutes": 15
    }'::jsonb,
    '{
        "market_open_hour": 8,
        "market_close_hour": 12,
        "stop_loss_pct": 0.005,
        "take_profit_pct": 0.012
    }'::jsonb,
    '{
        "sharpe_ratio": 2.1,
        "max_drawdown": 0.095,
        "win_rate": 0.61,
        "total_return": 0.32,
        "profit_factor": 1.85,
        "is_synthetic": true
    }'::jsonb,
    NULL,
    'synthetic-generator',
    'Demo model for investor presentations. Generates synthetic trades with target metrics. NOT for live trading.'
) ON CONFLICT (model_id) DO UPDATE SET
    hyperparameters = EXCLUDED.hyperparameters,
    policy_config = EXCLUDED.policy_config,
    backtest_metrics = EXCLUDED.backtest_metrics,
    updated_at = NOW();

-- =============================================================================
-- SECTION 9: SEED DATA - Feature Definitions (30 Total Features)
-- =============================================================================
-- Source: config/features/feature_registry_v19.json
-- Total: 18 market features + 12 state features = 30 observation dimensions
-- Order: State features [0-11] + Market features [12-29]
-- Updated: 2025-12-26 to match training environment exactly
-- =============================================================================

-- Clear existing features for clean re-seed
DELETE FROM config.feature_definitions;

-- Reset sequence
ALTER SEQUENCE config.feature_definitions_feature_id_seq RESTART WITH 1;

-- Temporarily disable the unique constraint on observation_order
ALTER TABLE config.feature_definitions DROP CONSTRAINT IF EXISTS chk_observation_order_unique;

-- =============================================================================
-- STATE FEATURES (12 features, observation_order 0-11)
-- These are computed by the environment at runtime, not from database
-- =============================================================================

INSERT INTO config.feature_definitions (
    feature_name, feature_group, normalization_mean, normalization_std,
    clip_min, clip_max, source_table, source_column, transformation,
    observation_order, compute_location, python_function, sql_formula,
    lookback_bars, description, is_active
) VALUES
-- State Feature 0: Position
(
    'position', 'state', NULL, NULL, -1.0, 1.0,
    NULL, NULL, 'runtime',
    0, 'python', 'env.current_position', NULL,
    0, 'Current position normalized [-1 short, 0 flat, 1 long]', TRUE
),
-- State Feature 1: Unrealized PnL
(
    'unrealized_pnl', 'state', NULL, NULL, -1.0, 1.0,
    NULL, NULL, 'runtime',
    1, 'python', 'clip(unrealized_pnl / 0.05, -1, 1)', NULL,
    0, 'Unrealized PnL normalized by 5% threshold', TRUE
),
-- State Feature 2: Cumulative Return
(
    'cumulative_return', 'state', NULL, NULL, -1.0, 1.0,
    NULL, NULL, 'runtime',
    2, 'python', 'clip(cumulative_return / 0.10, -1, 1)', NULL,
    0, 'Episode cumulative return normalized by 10%', TRUE
),
-- State Feature 3: Current Drawdown
(
    'current_drawdown', 'state', NULL, NULL, -1.0, 0.0,
    NULL, NULL, 'runtime',
    3, 'python', '-current_drawdown / max_drawdown_pct', NULL,
    0, 'Current drawdown as fraction of max allowed (negative)', TRUE
),
-- State Feature 4: Max Drawdown Episode
(
    'max_drawdown_episode', 'state', NULL, NULL, -1.0, 0.0,
    NULL, NULL, 'runtime',
    4, 'python', '-max_drawdown_episode / max_drawdown_pct', NULL,
    0, 'Max drawdown this episode (negative)', TRUE
),
-- State Feature 5: Regime Encoded
(
    'regime_encoded', 'regime', NULL, NULL, -0.5, 1.0,
    NULL, NULL, 'runtime',
    5, 'python', 'encode_regime(volatility_percentile)', NULL,
    0, 'Market regime: crisis=1.0, high_vol=0.5, normal=0.0, low_vol=-0.5', TRUE
),
-- State Feature 6: Session Phase
(
    'session_phase', 'state', NULL, NULL, 0.1, 0.8,
    NULL, NULL, 'runtime',
    6, 'python', 'encode_session_phase(hour)', NULL,
    0, 'Trading session phase: ny_am=0.8, ny_pm=0.6, asia=0.3, london=0.2, closed=0.1', TRUE
),
-- State Feature 7: Volatility Regime
(
    'volatility_regime', 'state', NULL, NULL, 0.0, 1.0,
    NULL, NULL, 'runtime',
    7, 'python', 'volatility_percentile', NULL,
    0, 'Volatility percentile [0=low, 1=high]', TRUE
),
-- State Feature 8: Cost Regime
(
    'cost_regime', 'state', NULL, NULL, 0.0, 1.0,
    NULL, NULL, 'runtime',
    8, 'python', 'curriculum_cost_multiplier', NULL,
    0, 'Curriculum cost multiplier (0=free, 1=full costs)', TRUE
),
-- State Feature 9: Position Duration
(
    'position_duration', 'state', NULL, NULL, 0.0, 1.0,
    NULL, NULL, 'runtime',
    9, 'python', 'min(position_duration / 100, 1.0)', NULL,
    0, 'Bars holding current position (normalized)', TRUE
),
-- State Feature 10: Trade Count Normalized
(
    'trade_count_normalized', 'state', NULL, NULL, 0.0, 1.0,
    NULL, NULL, 'runtime',
    10, 'python', 'min(trade_count / 50, 1.0)', NULL,
    0, 'Trades executed this episode (normalized)', TRUE
),
-- State Feature 11: Time Remaining
(
    'time_remaining', 'state', NULL, NULL, 0.0, 1.0,
    NULL, NULL, 'runtime',
    11, 'python', '1.0 - (current_step / episode_length)', NULL,
    0, 'Fraction of episode remaining', TRUE
),

-- =============================================================================
-- MARKET FEATURES (18 features, observation_order 12-29)
-- These are computed from OHLCV and Macro data
-- =============================================================================

-- Market Feature 12: Log Return 5min
(
    'log_ret_5m', 'price_returns', 0.000002, 0.001138, -0.05, 0.05,
    'usdcop_m5_ohlcv', 'close', 'log_return',
    12, 'sql', NULL, 'LN(close / LAG(close, 1) OVER (ORDER BY time))',
    1, '5-minute logarithmic return', TRUE
),
-- Market Feature 13: Log Return 1h
(
    'log_ret_1h', 'price_returns', 0.000023, 0.003776, -0.05, 0.05,
    'usdcop_m5_ohlcv', 'close', 'log_return',
    13, 'sql', NULL, 'LN(close / LAG(close, 12) OVER (ORDER BY time))',
    12, '1-hour logarithmic return (12 bars)', TRUE
),
-- Market Feature 14: Log Return 4h
(
    'log_ret_4h', 'price_returns', 0.000052, 0.007768, -0.05, 0.05,
    'usdcop_m5_ohlcv', 'close', 'log_return',
    14, 'sql', NULL, 'LN(close / LAG(close, 48) OVER (ORDER BY time))',
    48, '4-hour logarithmic return (48 bars)', TRUE
),
-- Market Feature 15: RSI 9
(
    'rsi_9', 'momentum', 49.27, 23.07, 0.0, 100.0,
    'usdcop_m5_ohlcv', 'close', 'rsi',
    15, 'python', 'calc_rsi(close, period=9)', NULL,
    9, 'Relative Strength Index with 9-period lookback', TRUE
),
-- Market Feature 16: ATR Percent
(
    'atr_pct', 'volatility', 0.062, 0.0446, NULL, NULL,
    'usdcop_m5_ohlcv', 'high,low,close', 'atr_percentage',
    16, 'python', 'calc_atr_pct(high, low, close, period=10)', NULL,
    10, 'Average True Range as percentage of price', TRUE
),
-- Market Feature 17: ADX 14
(
    'adx_14', 'trend', 32.01, 16.36, 0.0, 100.0,
    'usdcop_m5_ohlcv', 'high,low,close', 'adx',
    17, 'python', 'calc_adx(high, low, close, period=14)', NULL,
    14, 'Average Directional Index with 14-period lookback', TRUE
),
-- Market Feature 18: Bollinger Band Position
(
    'bb_position', 'volatility', NULL, NULL, 0.0, 1.0,
    'usdcop_m5_ohlcv', 'close', 'bollinger_position',
    18, 'python', 'calc_bb_position(close, period=20, std_dev=2)', NULL,
    20, 'Position within Bollinger Bands (0=lower, 1=upper)', TRUE
),
-- Market Feature 19: DXY Z-score
(
    'dxy_z', 'macro_zscore', 100.21, 5.60, -4.0, 4.0,
    'macro_indicators_daily', 'dxy', 'zscore_fixed',
    19, 'sql', NULL, '(dxy - 100.21) / 5.60',
    1, 'DXY Dollar Index z-score (fixed stats from 2020-03 to 2025-10)', TRUE
),
-- Market Feature 20: DXY Change 1d
(
    'dxy_change_1d', 'macro_changes', NULL, NULL, -0.03, 0.03,
    'macro_indicators_daily', 'dxy', 'pct_change',
    20, 'sql', NULL, '(dxy - LAG(dxy, 1)) / NULLIF(LAG(dxy, 1), 0)',
    1, 'DXY 1-day percentage change', TRUE
),
-- Market Feature 21: DXY Momentum 5d
(
    'dxy_mom_5d', 'macro_momentum', NULL, NULL, -0.05, 0.05,
    'macro_indicators_daily', 'dxy', 'pct_change',
    21, 'sql', NULL, '(dxy - LAG(dxy, 5)) / NULLIF(LAG(dxy, 5), 0)',
    5, 'DXY 5-day momentum', TRUE
),
-- Market Feature 22: VIX Z-score
(
    'vix_z', 'macro_zscore', 21.16, 7.89, -4.0, 4.0,
    'macro_indicators_daily', 'vix', 'zscore_fixed',
    22, 'sql', NULL, '(vix - 21.16) / 7.89',
    1, 'VIX volatility index z-score (fixed stats from 2020-03 to 2025-10)', TRUE
),
-- Market Feature 23: EMBI Z-score
(
    'embi_z', 'macro_zscore', 322.01, 62.68, -4.0, 4.0,
    'macro_indicators_daily', 'embi', 'zscore_fixed',
    23, 'sql', NULL, '(embi - 322.01) / 62.68',
    1, 'EMBI spread z-score (fixed stats from 2020-03 to 2025-10)', TRUE
),
-- Market Feature 24: Brent Change 1d
(
    'brent_change_1d', 'macro_changes', NULL, NULL, -0.10, 0.10,
    'macro_indicators_daily', 'brent', 'pct_change',
    24, 'sql', NULL, '(brent - LAG(brent, 1)) / NULLIF(LAG(brent, 1), 0)',
    1, 'Brent crude oil 1-day percentage change', TRUE
),
-- Market Feature 25: Brent Volatility 5d
(
    'brent_vol_5d', 'macro_volatility', NULL, NULL, 0.0, 0.05,
    'macro_indicators_daily', 'brent', 'rolling_std',
    25, 'python', 'calc_rolling_std(brent, period=5)', NULL,
    5, '5-day Brent volatility (rolling std)', TRUE
),
-- Market Feature 26: Rate Spread
(
    'rate_spread', 'macro_derived', 7.03, 1.41, NULL, NULL,
    'macro_indicators_daily', 'treasury_10y', 'sovereign_spread',
    26, 'sql', NULL, '((10.0 - treasury_10y) - 7.03) / 1.41',
    1, 'Sovereign spread: Colombia 10Y (10%) minus US 10Y, z-scored', TRUE
),
-- Market Feature 27: USD/MXN Change 1d
(
    'usdmxn_change_1d', 'macro_changes', NULL, NULL, -0.10, 0.10,
    'macro_indicators_daily', 'usdmxn', 'pct_change',
    27, 'sql', NULL, '(usdmxn - LAG(usdmxn, 1)) / NULLIF(LAG(usdmxn, 1), 0)',
    1, 'USD/MXN 1-day percentage change (EM proxy)', TRUE
),
-- Market Feature 28: Hour Sine
(
    'hour_sin', 'temporal', NULL, NULL, -1.0, 1.0,
    'usdcop_m5_ohlcv', 'time', 'cyclical_hour_sin',
    28, 'sql', NULL, 'SIN(2 * PI() * EXTRACT(HOUR FROM time) / 24)',
    0, 'Hour of day (sine encoding for cyclical time)', TRUE
),
-- Market Feature 29: Hour Cosine
(
    'hour_cos', 'temporal', NULL, NULL, -1.0, 1.0,
    'usdcop_m5_ohlcv', 'time', 'cyclical_hour_cos',
    29, 'sql', NULL, 'COS(2 * PI() * EXTRACT(HOUR FROM time) / 24)',
    0, 'Hour of day (cosine encoding for cyclical time)', TRUE
);

-- Re-add the unique constraint
ALTER TABLE config.feature_definitions
ADD CONSTRAINT chk_observation_order_unique UNIQUE (observation_order) DEFERRABLE INITIALLY DEFERRED;

-- =============================================================================
-- SECTION 10: VIEWS FOR COMMON QUERIES
-- =============================================================================

-- View: Latest signals per model
CREATE OR REPLACE VIEW trading.vw_latest_signals AS
SELECT DISTINCT ON (model_id)
    i.model_id,
    m.name AS model_name,
    m.algorithm,
    m.color,
    i.timestamp_utc,
    i.action_discretized AS signal,
    i.action_raw,
    i.confidence,
    i.price_at_inference,
    i.current_position,
    i.latency_ms,
    i.bar_number
FROM trading.model_inferences i
JOIN config.models m ON i.model_id = m.model_id
WHERE m.status = 'active'
ORDER BY model_id, timestamp_utc DESC;

COMMENT ON VIEW trading.vw_latest_signals IS 'Latest inference signal for each active model';

-- View: Daily performance summary
CREATE OR REPLACE VIEW metrics.vw_daily_performance AS
SELECT
    p.model_id,
    m.name AS model_name,
    m.algorithm,
    m.color,
    p.period_start::date AS trading_date,
    p.return_pct,
    p.return_cumulative_pct,
    p.sharpe_ratio,
    p.max_drawdown,
    p.total_trades,
    p.win_rate,
    p.avg_trade_pnl,
    p.hold_percentage,
    p.ending_equity
FROM metrics.model_performance p
JOIN config.models m ON p.model_id = m.model_id
WHERE p.aggregation_type = 'daily'
ORDER BY p.period_start DESC, p.sharpe_ratio DESC;

COMMENT ON VIEW metrics.vw_daily_performance IS 'Daily performance metrics for all models, sorted by date and Sharpe ratio';

-- View: Active model comparison
CREATE OR REPLACE VIEW config.vw_model_comparison AS
SELECT
    m.model_id,
    m.name,
    m.algorithm,
    m.version,
    m.color,
    m.hyperparameters->>'learning_rate' AS learning_rate,
    m.hyperparameters->>'n_steps' AS n_steps,
    m.hyperparameters->>'batch_size' AS batch_size,
    m.policy_config->>'net_arch' AS net_arch,
    m.policy_config->>'activation_fn' AS activation_fn,
    COALESCE((m.backtest_metrics->>'sharpe_ratio')::numeric, 0) AS sharpe_ratio,
    COALESCE((m.backtest_metrics->>'max_drawdown')::numeric, 0) AS max_drawdown,
    COALESCE((m.backtest_metrics->>'win_rate')::numeric, 0) AS win_rate,
    m.status,
    m.updated_at
FROM config.models m
ORDER BY
    CASE WHEN m.status = 'active' THEN 0 ELSE 1 END,
    COALESCE((m.backtest_metrics->>'sharpe_ratio')::numeric, 0) DESC;

COMMENT ON VIEW config.vw_model_comparison IS 'Side-by-side comparison of all models with key hyperparameters and metrics';

-- View: Open trades
CREATE OR REPLACE VIEW trading.vw_open_trades AS
SELECT
    t.trade_id,
    t.model_id,
    m.name AS model_name,
    t.signal,
    t.open_time,
    t.entry_price,
    t.position_size,
    t.entry_confidence,
    EXTRACT(EPOCH FROM (NOW() - t.open_time)) / 60 AS minutes_open,
    t.max_favorable_excursion,
    t.max_adverse_excursion
FROM trading.model_trades t
JOIN config.models m ON t.model_id = m.model_id
WHERE t.status = 'open'
ORDER BY t.open_time DESC;

COMMENT ON VIEW trading.vw_open_trades IS 'Currently open trades across all models';

-- View: Recent events (last 24 hours)
CREATE OR REPLACE VIEW events.vw_recent_events AS
SELECT
    e.stream_id,
    e.timestamp_utc,
    e.event_type,
    e.event_subtype,
    e.model_id,
    m.name AS model_name,
    e.price_at_event,
    e.position_at_event,
    e.payload,
    e.processed
FROM events.signals_stream e
LEFT JOIN config.models m ON e.model_id = m.model_id
WHERE e.timestamp_utc >= NOW() - INTERVAL '24 hours'
ORDER BY e.timestamp_utc DESC;

COMMENT ON VIEW events.vw_recent_events IS 'Events from the last 24 hours for monitoring and debugging';

-- View: Feature configuration summary
CREATE OR REPLACE VIEW config.vw_feature_summary AS
SELECT
    f.observation_order,
    f.feature_name,
    f.feature_group,
    f.compute_location,
    COALESCE(f.normalization_mean::text, 'N/A') AS norm_mean,
    COALESCE(f.normalization_std::text, 'N/A') AS norm_std,
    COALESCE(f.clip_min::text || ' to ' || f.clip_max::text, 'No clip') AS clip_range,
    f.lookback_bars,
    f.is_active
FROM config.feature_definitions f
WHERE f.is_active = TRUE
ORDER BY f.observation_order;

COMMENT ON VIEW config.vw_feature_summary IS 'Active features in observation order with normalization stats';

-- =============================================================================
-- SECTION 11: FUNCTIONS FOR COMMON OPERATIONS
-- =============================================================================

-- Function: Record a new inference
CREATE OR REPLACE FUNCTION trading.record_inference(
    p_model_id VARCHAR(50),
    p_action_raw DOUBLE PRECISION,
    p_action_discretized VARCHAR(10),
    p_confidence DOUBLE PRECISION,
    p_price NUMERIC(12, 4),
    p_features JSONB,
    p_latency_ms INTEGER,
    p_position DOUBLE PRECISION DEFAULT 0.0,
    p_bar_number INTEGER DEFAULT NULL,
    p_session_id VARCHAR(50) DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    v_inference_id BIGINT;
BEGIN
    INSERT INTO trading.model_inferences (
        timestamp_utc, model_id, action_raw, action_discretized,
        confidence, price_at_inference, features_snapshot,
        latency_ms, current_position, bar_number, session_id
    ) VALUES (
        NOW(), p_model_id, p_action_raw, p_action_discretized,
        p_confidence, p_price, p_features,
        p_latency_ms, p_position, p_bar_number, p_session_id
    ) RETURNING inference_id INTO v_inference_id;

    RETURN v_inference_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION trading.record_inference IS 'Records a model inference with all associated data';

-- Function: Open a new trade
CREATE OR REPLACE FUNCTION trading.open_trade(
    p_model_id VARCHAR(50),
    p_signal VARCHAR(10),
    p_entry_price NUMERIC(12, 4),
    p_position_size NUMERIC(8, 4) DEFAULT 1.0,
    p_confidence NUMERIC(4, 3) DEFAULT NULL,
    p_inference_id BIGINT DEFAULT NULL
)
RETURNS BIGINT AS $$
DECLARE
    v_trade_id BIGINT;
BEGIN
    INSERT INTO trading.model_trades (
        model_id, open_time, signal, entry_price,
        position_size, entry_confidence, entry_inference_id, status
    ) VALUES (
        p_model_id, NOW(), p_signal, p_entry_price,
        p_position_size, p_confidence, p_inference_id, 'open'
    ) RETURNING trade_id INTO v_trade_id;

    -- Log trade open event
    INSERT INTO events.signals_stream (event_type, event_subtype, model_id, payload, price_at_event)
    VALUES ('TRADE_OPEN', p_signal, p_model_id,
            jsonb_build_object('trade_id', v_trade_id, 'entry_price', p_entry_price, 'size', p_position_size),
            p_entry_price);

    RETURN v_trade_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION trading.open_trade IS 'Opens a new trade and logs the event';

-- Function: Close a trade
CREATE OR REPLACE FUNCTION trading.close_trade(
    p_trade_id BIGINT,
    p_exit_price NUMERIC(12, 4),
    p_exit_reason VARCHAR(50) DEFAULT 'signal',
    p_exit_confidence NUMERIC(4, 3) DEFAULT NULL,
    p_inference_id BIGINT DEFAULT NULL,
    p_transaction_costs NUMERIC(8, 4) DEFAULT 0
)
RETURNS TABLE (pnl NUMERIC, pnl_pct NUMERIC, duration_minutes INTEGER) AS $$
DECLARE
    v_trade RECORD;
    v_pnl NUMERIC(12, 4);
    v_pnl_pct NUMERIC(12, 6);
    v_gross_pnl NUMERIC(12, 4);
    v_duration INTEGER;
BEGIN
    -- Get trade details
    SELECT * INTO v_trade FROM trading.model_trades WHERE trade_id = p_trade_id;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'Trade % not found', p_trade_id;
    END IF;

    IF v_trade.status != 'open' THEN
        RAISE EXCEPTION 'Trade % is not open (status: %)', p_trade_id, v_trade.status;
    END IF;

    -- Calculate PnL
    IF v_trade.signal = 'LONG' THEN
        v_gross_pnl := (p_exit_price - v_trade.entry_price) * v_trade.position_size;
    ELSE
        v_gross_pnl := (v_trade.entry_price - p_exit_price) * v_trade.position_size;
    END IF;

    v_pnl := v_gross_pnl - p_transaction_costs;
    v_pnl_pct := v_pnl / v_trade.entry_price;
    v_duration := EXTRACT(EPOCH FROM (NOW() - v_trade.open_time)) / 60;

    -- Update trade
    UPDATE trading.model_trades SET
        close_time = NOW(),
        exit_price = p_exit_price,
        pnl = v_pnl,
        pnl_pct = v_pnl_pct,
        gross_pnl = v_gross_pnl,
        duration_minutes = v_duration,
        duration_bars = v_duration / 5,
        status = 'closed',
        exit_reason = p_exit_reason,
        exit_confidence = p_exit_confidence,
        exit_inference_id = p_inference_id,
        transaction_costs = p_transaction_costs,
        updated_at = NOW()
    WHERE trade_id = p_trade_id;

    -- Log trade close event
    INSERT INTO events.signals_stream (event_type, event_subtype, model_id, payload, price_at_event)
    VALUES ('TRADE_CLOSE', p_exit_reason, v_trade.model_id,
            jsonb_build_object('trade_id', p_trade_id, 'exit_price', p_exit_price, 'pnl', v_pnl, 'pnl_pct', v_pnl_pct),
            p_exit_price);

    RETURN QUERY SELECT v_pnl, v_pnl_pct, v_duration;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION trading.close_trade IS 'Closes an open trade with PnL calculation and event logging';

-- Function: Get model current state
CREATE OR REPLACE FUNCTION trading.get_model_state(p_model_id VARCHAR(50))
RETURNS TABLE (
    model_name VARCHAR,
    current_position DOUBLE PRECISION,
    last_signal VARCHAR,
    last_signal_time TIMESTAMPTZ,
    open_trades INTEGER,
    daily_pnl NUMERIC,
    daily_trades INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.name,
        COALESCE(
            (SELECT i.current_position FROM trading.model_inferences i
             WHERE i.model_id = p_model_id ORDER BY timestamp_utc DESC LIMIT 1), 0.0
        ),
        (SELECT i.action_discretized FROM trading.model_inferences i
         WHERE i.model_id = p_model_id ORDER BY timestamp_utc DESC LIMIT 1),
        (SELECT i.timestamp_utc FROM trading.model_inferences i
         WHERE i.model_id = p_model_id ORDER BY timestamp_utc DESC LIMIT 1),
        (SELECT COUNT(*)::INTEGER FROM trading.model_trades t
         WHERE t.model_id = p_model_id AND t.status = 'open'),
        COALESCE(
            (SELECT SUM(t.pnl) FROM trading.model_trades t
             WHERE t.model_id = p_model_id AND t.close_time::date = CURRENT_DATE), 0
        ),
        (SELECT COUNT(*)::INTEGER FROM trading.model_trades t
         WHERE t.model_id = p_model_id AND t.close_time::date = CURRENT_DATE)
    FROM config.models m
    WHERE m.model_id = p_model_id;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION trading.get_model_state IS 'Returns current state for a model including position, signals, and daily PnL';

-- =============================================================================
-- SECTION 12: TRIGGERS FOR AUTOMATIC TIMESTAMPS
-- =============================================================================

-- Function for updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for config.models
DROP TRIGGER IF EXISTS trigger_models_updated_at ON config.models;
CREATE TRIGGER trigger_models_updated_at
    BEFORE UPDATE ON config.models
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for config.feature_definitions
DROP TRIGGER IF EXISTS trigger_features_updated_at ON config.feature_definitions;
CREATE TRIGGER trigger_features_updated_at
    BEFORE UPDATE ON config.feature_definitions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Trigger for trading.model_trades
DROP TRIGGER IF EXISTS trigger_trades_updated_at ON trading.model_trades;
CREATE TRIGGER trigger_trades_updated_at
    BEFORE UPDATE ON trading.model_trades
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- SECTION 13: PERMISSIONS
-- =============================================================================

-- Grant permissions to standard roles (adjust role names as needed)
DO $$
BEGIN
    -- Grant schema access
    GRANT USAGE ON SCHEMA config TO PUBLIC;
    GRANT USAGE ON SCHEMA trading TO PUBLIC;
    GRANT USAGE ON SCHEMA events TO PUBLIC;
    GRANT USAGE ON SCHEMA metrics TO PUBLIC;

    -- Grant table access (read for all, write for appropriate roles)
    GRANT SELECT ON ALL TABLES IN SCHEMA config TO PUBLIC;
    GRANT SELECT ON ALL TABLES IN SCHEMA trading TO PUBLIC;
    GRANT SELECT ON ALL TABLES IN SCHEMA events TO PUBLIC;
    GRANT SELECT ON ALL TABLES IN SCHEMA metrics TO PUBLIC;

    -- Grant sequence usage
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA config TO PUBLIC;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO PUBLIC;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA events TO PUBLIC;
    GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metrics TO PUBLIC;

    -- Grant function execution
    GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA trading TO PUBLIC;

    -- If admin role exists, grant full access
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'admin') THEN
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA config TO admin;
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO admin;
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA events TO admin;
        GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO admin;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA config TO admin;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO admin;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA events TO admin;
        GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO admin;
    END IF;

    -- If airflow role exists, grant necessary access
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'airflow') THEN
        GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA trading TO airflow;
        GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA events TO airflow;
        GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA metrics TO airflow;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA trading TO airflow;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA events TO airflow;
        GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA metrics TO airflow;
    END IF;

EXCEPTION
    WHEN undefined_object THEN
        RAISE NOTICE 'Some roles do not exist, skipping role-specific grants';
END $$;

-- =============================================================================
-- SECTION 14: VERIFICATION AND SUCCESS MESSAGE
-- =============================================================================

-- Verify schema creation
DO $$
DECLARE
    v_schema_count INTEGER;
    v_table_count INTEGER;
    v_model_count INTEGER;
    v_feature_count INTEGER;
BEGIN
    -- Count schemas
    SELECT COUNT(*) INTO v_schema_count
    FROM information_schema.schemata
    WHERE schema_name IN ('config', 'trading', 'events', 'metrics');

    -- Count tables
    SELECT COUNT(*) INTO v_table_count
    FROM information_schema.tables
    WHERE table_schema IN ('config', 'trading', 'events', 'metrics')
    AND table_type = 'BASE TABLE';

    -- Count models
    SELECT COUNT(*) INTO v_model_count FROM config.models;

    -- Count features
    SELECT COUNT(*) INTO v_feature_count FROM config.feature_definitions WHERE is_active = TRUE;

    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'MULTI-MODEL TRADING SCHEMA - INITIALIZATION COMPLETE';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Version: 1.0.0';
    RAISE NOTICE 'Date: 2025-12-26';
    RAISE NOTICE '------------------------------------------------------------';
    RAISE NOTICE 'Schemas created: %', v_schema_count;
    RAISE NOTICE 'Tables created: %', v_table_count;
    RAISE NOTICE 'Models configured: %', v_model_count;
    RAISE NOTICE 'Active features: %', v_feature_count;
    RAISE NOTICE '------------------------------------------------------------';
    RAISE NOTICE 'SCHEMAS: config, trading, events, metrics';
    RAISE NOTICE 'TABLES:';
    RAISE NOTICE '  - config.models (RL model configurations)';
    RAISE NOTICE '  - config.feature_definitions (observation space features)';
    RAISE NOTICE '  - trading.model_inferences (real-time inference results)';
    RAISE NOTICE '  - trading.model_trades (trade tracking with PnL)';
    RAISE NOTICE '  - events.signals_stream (audit trail)';
    RAISE NOTICE '  - metrics.model_performance (aggregated metrics)';
    RAISE NOTICE '------------------------------------------------------------';
    RAISE NOTICE 'PRODUCTION MODEL: PPO V1 (trained 2025-12-26)';
    RAISE NOTICE '  - Algorithm: PPO';
    RAISE NOTICE '  - Network: [256, 256] with Tanh activation';
    RAISE NOTICE '  - Sharpe (validation): 2.91';
    RAISE NOTICE '  - Features: 30 (12 state + 18 market)';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;

-- Final query to show summary
SELECT
    'Multi-Model Schema Ready' AS status,
    (SELECT COUNT(*) FROM config.models) AS total_models,
    (SELECT COUNT(*) FROM config.models WHERE status = 'active') AS active_models,
    (SELECT COUNT(*) FROM config.feature_definitions WHERE is_active = TRUE) AS active_features;
