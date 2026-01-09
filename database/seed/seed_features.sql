-- =============================================================================
-- Feature Definitions Seed for USD/COP RL Trading System V19
-- =============================================================================
-- This script populates the config.feature_definitions table with all features
-- used during V19 model training. These definitions are the Single Source of
-- Truth (SSOT) for feature engineering in production.
--
-- Author: Claude Code
-- Version: 1.0.0
-- Date: 2025-12-26
-- =============================================================================

-- Create schema if not exists
CREATE SCHEMA IF NOT EXISTS config;

-- Drop existing table if recreating
-- DROP TABLE IF EXISTS config.feature_definitions CASCADE;

-- Create feature definitions table
CREATE TABLE IF NOT EXISTS config.feature_definitions (
    id SERIAL PRIMARY KEY,
    name VARCHAR(64) NOT NULL UNIQUE,
    version VARCHAR(16) NOT NULL DEFAULT 'V19',
    feature_group VARCHAR(32) NOT NULL,
    feature_type VARCHAR(16) NOT NULL DEFAULT 'market', -- 'market' or 'state'
    observation_order INTEGER NOT NULL,

    -- Source configuration
    source_column VARCHAR(64),
    source_table VARCHAR(64),

    -- Transformation
    transformation VARCHAR(32) NOT NULL,
    formula TEXT,
    period INTEGER,
    lookback_bars INTEGER,

    -- Normalization
    norm_method VARCHAR(32) NOT NULL DEFAULT 'z_score',
    norm_mean DOUBLE PRECISION,
    norm_std DOUBLE PRECISION,
    clip_min DOUBLE PRECISION,
    clip_max DOUBLE PRECISION,

    -- Metadata
    description TEXT,
    compute_location VARCHAR(16) DEFAULT 'sql', -- 'sql' or 'python'
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for fast lookups
CREATE INDEX IF NOT EXISTS idx_feature_definitions_version
    ON config.feature_definitions(version);
CREATE INDEX IF NOT EXISTS idx_feature_definitions_group
    ON config.feature_definitions(feature_group);
CREATE INDEX IF NOT EXISTS idx_feature_definitions_order
    ON config.feature_definitions(feature_type, observation_order);

-- Clear existing V19 definitions
DELETE FROM config.feature_definitions WHERE version = 'V19';

-- =============================================================================
-- MARKET FEATURES (18 features)
-- =============================================================================

-- Returns Features (3)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula, lookback_bars,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('log_ret_5m', 'V19', 'returns', 'market', 0,
     'close', 'usdcop_m5_ohlcv', 'log_return', 'ln(close / close[-1])', 1,
     'z_score', 2.0e-06, 0.001138, -0.05, 0.05,
     '5-minute log return', 'sql'),

    ('log_ret_1h', 'V19', 'returns', 'market', 1,
     'close', 'usdcop_m5_ohlcv', 'log_return', 'ln(close / close[-12])', 12,
     'z_score', 2.3e-05, 0.003776, -0.05, 0.05,
     '1-hour log return (12 x 5-min bars)', 'sql'),

    ('log_ret_4h', 'V19', 'returns', 'market', 2,
     'close', 'usdcop_m5_ohlcv', 'log_return', 'ln(close / close[-48])', 48,
     'z_score', 5.2e-05, 0.007768, -0.05, 0.05,
     '4-hour log return (48 x 5-min bars)', 'sql');

-- Technical Features (4)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula, period,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('rsi_9', 'V19', 'technical', 'market', 3,
     'close', 'usdcop_m5_ohlcv', 'rsi', 'RSI(close, 9)', 9,
     'z_score', 49.27, 23.07, NULL, NULL,
     'Relative Strength Index 9-period', 'python'),

    ('atr_pct', 'V19', 'technical', 'market', 4,
     'close', 'usdcop_m5_ohlcv', 'atr_percent', '(ATR / close) * 100', 10,
     'z_score', 0.062, 0.0446, NULL, NULL,
     'Average True Range as percentage of close', 'python'),

    ('adx_14', 'V19', 'technical', 'market', 5,
     'close', 'usdcop_m5_ohlcv', 'adx', 'ADX(high, low, close, 14)', 14,
     'z_score', 32.01, 16.36, NULL, NULL,
     'Average Directional Index 14-period', 'python'),

    ('bb_position', 'V19', 'technical', 'market', 6,
     'close', 'usdcop_m5_ohlcv', 'bollinger_position', '(close - bb_lower) / (bb_upper - bb_lower)', 20,
     'clip_only', NULL, NULL, 0.0, 1.0,
     'Position within Bollinger Bands (0=lower, 1=upper)', 'python');

-- Macro Z-score Features (3)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('dxy_z', 'V19', 'macro_zscore', 'market', 7,
     'dxy', 'macro_indicators_daily', 'zscore_fixed', '(dxy - 100.21) / 5.60',
     'z_score', 100.21, 5.60, -4, 4,
     'Dollar Index (DXY) z-score normalized', 'sql'),

    ('vix_z', 'V19', 'macro_zscore', 'market', 10,
     'vix', 'macro_indicators_daily', 'zscore_fixed', '(vix - 21.16) / 7.89',
     'z_score', 21.16, 7.89, -4, 4,
     'VIX z-score normalized', 'sql'),

    ('embi_z', 'V19', 'macro_zscore', 'market', 11,
     'embi', 'macro_indicators_daily', 'zscore_fixed', '(embi - 322.01) / 62.68',
     'z_score', 322.01, 62.68, -4, 4,
     'EMBI spread z-score normalized', 'sql');

-- Macro Change Features (3)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula, lookback_bars,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('dxy_change_1d', 'V19', 'macro_changes', 'market', 8,
     'dxy', 'macro_indicators_daily', 'pct_change', '(dxy - dxy[-1]) / dxy[-1]', 1,
     'clip_only', NULL, NULL, -0.03, 0.03,
     'Daily DXY percent change', 'sql'),

    ('brent_change_1d', 'V19', 'macro_changes', 'market', 12,
     'brent', 'macro_indicators_daily', 'pct_change', '(brent - brent[-1]) / brent[-1]', 1,
     'clip_only', NULL, NULL, -0.10, 0.10,
     'Daily Brent crude oil percent change', 'sql'),

    ('usdmxn_change_1d', 'V19', 'macro_changes', 'market', 15,
     'usdmxn', 'macro_indicators_daily', 'pct_change', '(usdmxn - usdmxn[-1]) / usdmxn[-1]', 1,
     'clip_only', NULL, NULL, -0.10, 0.10,
     'Daily USD/MXN percent change (peso proxy)', 'python');

-- Macro Momentum Feature (1)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula, lookback_bars,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('dxy_mom_5d', 'V19', 'macro_momentum', 'market', 9,
     'dxy', 'macro_indicators_daily', 'pct_change', '(dxy - dxy[-5]) / dxy[-5]', 5,
     'clip_only', NULL, NULL, -0.05, 0.05,
     '5-day DXY momentum', 'sql');

-- Macro Volatility Feature (1)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula, period,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('brent_vol_5d', 'V19', 'macro_volatility', 'market', 13,
     'brent', 'macro_indicators_daily', 'rolling_std', 'rolling_std(pct_change(brent), 5)', 5,
     'clip_only', NULL, NULL, 0.0, 0.05,
     '5-day Brent volatility', 'python');

-- Macro Derived Feature (1)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, source_table, transformation, formula,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('rate_spread', 'V19', 'macro_derived', 'market', 14,
     'treasury_10y', 'macro_indicators_daily', 'derived', '10.0 - treasury_10y',
     'z_score', 7.03, 1.41, NULL, NULL,
     'Sovereign spread: Colombia 10Y (10%) - USA 10Y', 'sql');

-- Temporal Features (2)
INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, transformation, formula,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('hour_sin', 'V19', 'temporal', 'market', 16,
     'timestamp', 'cyclical_hour_sin', 'sin(2 * pi * hour / 24)',
     'none', NULL, NULL, NULL, NULL,
     'Hour of day (sine encoding)', 'python'),

    ('hour_cos', 'V19', 'temporal', 'market', 17,
     'timestamp', 'cyclical_hour_cos', 'cos(2 * pi * hour / 24)',
     'none', NULL, NULL, NULL, NULL,
     'Hour of day (cosine encoding)', 'python');

-- =============================================================================
-- STATE FEATURES (12 features)
-- =============================================================================

INSERT INTO config.feature_definitions
    (name, version, feature_group, feature_type, observation_order,
     source_column, transformation, formula,
     norm_method, norm_mean, norm_std, clip_min, clip_max,
     description, compute_location)
VALUES
    ('position', 'V19', 'portfolio_state', 'state', 0,
     'environment', 'none', 'current_position',
     'none', NULL, NULL, -1.0, 1.0,
     'Current position normalized [-1 short, 0 flat, 1 long]', 'environment'),

    ('unrealized_pnl', 'V19', 'portfolio_state', 'state', 1,
     'environment', 'clip_normalize', 'clip(unrealized_pnl / 0.05, -1, 1)',
     'clip_normalize', NULL, NULL, -1, 1,
     'Unrealized PnL normalized by 5% threshold', 'environment'),

    ('cumulative_return', 'V19', 'portfolio_state', 'state', 2,
     'environment', 'clip_normalize', 'clip(cumulative_return / 0.10, -1, 1)',
     'clip_normalize', NULL, NULL, -1, 1,
     'Episode cumulative return normalized by 10%', 'environment'),

    ('current_drawdown', 'V19', 'risk_state', 'state', 3,
     'environment', 'scale_negative', '-current_drawdown / max_drawdown_pct',
     'scale_negative', NULL, NULL, NULL, NULL,
     'Current drawdown as fraction of max allowed (negative)', 'environment'),

    ('max_drawdown_episode', 'V19', 'risk_state', 'state', 4,
     'environment', 'scale_negative', '-max_drawdown_episode / max_drawdown_pct',
     'scale_negative', NULL, NULL, NULL, NULL,
     'Max drawdown this episode (negative)', 'environment'),

    ('regime_encoded', 'V19', 'market_regime', 'state', 5,
     'environment', 'none', 'regime_encoding',
     'none', NULL, NULL, -0.5, 1.0,
     'Market regime encoded based on volatility percentile', 'environment'),

    ('session_phase', 'V19', 'temporal_state', 'state', 6,
     'environment', 'none', 'session_phase_encoding',
     'none', NULL, NULL, 0.1, 0.8,
     'Trading session phase (liquidity proxy)', 'environment'),

    ('volatility_regime', 'V19', 'risk_state', 'state', 7,
     'environment', 'percentile', 'volatility_percentile',
     'percentile', NULL, NULL, 0.0, 1.0,
     'Volatility percentile [0=low, 1=high]', 'environment'),

    ('cost_regime', 'V19', 'cost_state', 'state', 8,
     'environment', 'none', 'curriculum_cost_multiplier',
     'none', NULL, NULL, 0.0, 1.0,
     'Curriculum cost multiplier (0=free, 1=full costs)', 'environment'),

    ('position_duration', 'V19', 'portfolio_state', 'state', 9,
     'environment', 'min_scale', 'min(position_duration / 100, 1.0)',
     'min_scale', NULL, NULL, 0, 1,
     'Bars holding current position (normalized)', 'environment'),

    ('trade_count_normalized', 'V19', 'portfolio_state', 'state', 10,
     'environment', 'min_scale', 'min(trade_count / 50, 1.0)',
     'min_scale', NULL, NULL, 0, 1,
     'Trades executed this episode (normalized)', 'environment'),

    ('time_remaining', 'V19', 'temporal_state', 'state', 11,
     'environment', 'none', '1.0 - (current_step / episode_length)',
     'none', NULL, NULL, 0.0, 1.0,
     'Fraction of episode remaining', 'environment');

-- =============================================================================
-- ENVIRONMENT CONFIGURATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS config.environment_config (
    id SERIAL PRIMARY KEY,
    version VARCHAR(16) NOT NULL UNIQUE,
    class_name VARCHAR(64) NOT NULL,
    initial_balance DOUBLE PRECISION NOT NULL DEFAULT 10000,
    max_position DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    episode_length INTEGER NOT NULL DEFAULT 400,
    max_drawdown_pct DOUBLE PRECISION NOT NULL DEFAULT 15.0,
    bars_per_day INTEGER NOT NULL DEFAULT 56,
    use_vol_scaling BOOLEAN DEFAULT TRUE,
    use_regime_detection BOOLEAN DEFAULT TRUE,
    volatility_column VARCHAR(64) DEFAULT 'atr_pct',
    return_column VARCHAR(64) DEFAULT 'log_ret_5m',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

DELETE FROM config.environment_config WHERE version = 'V19';

INSERT INTO config.environment_config
    (version, class_name, initial_balance, max_position, episode_length,
     max_drawdown_pct, bars_per_day, use_vol_scaling, use_regime_detection,
     volatility_column, return_column)
VALUES
    ('V19', 'TradingEnvironmentV19', 10000, 1.0, 400,
     15.0, 56, TRUE, TRUE, 'atr_pct', 'log_ret_5m');

-- =============================================================================
-- COST MODEL CONFIGURATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS config.cost_model (
    id SERIAL PRIMARY KEY,
    version VARCHAR(16) NOT NULL UNIQUE,
    class_name VARCHAR(64) NOT NULL,
    base_spread_bps DOUBLE PRECISION NOT NULL DEFAULT 14.0,
    high_vol_spread_bps DOUBLE PRECISION NOT NULL DEFAULT 28.0,
    crisis_spread_bps DOUBLE PRECISION NOT NULL DEFAULT 45.0,
    slippage_bps DOUBLE PRECISION NOT NULL DEFAULT 3.0,
    volatility_threshold_high DOUBLE PRECISION NOT NULL DEFAULT 0.7,
    volatility_threshold_crisis DOUBLE PRECISION NOT NULL DEFAULT 0.9,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

DELETE FROM config.cost_model WHERE version = 'V19';

INSERT INTO config.cost_model
    (version, class_name, base_spread_bps, high_vol_spread_bps, crisis_spread_bps,
     slippage_bps, volatility_threshold_high, volatility_threshold_crisis)
VALUES
    ('V19', 'SETFXCostModel', 14.0, 28.0, 45.0, 3.0, 0.7, 0.9);

-- =============================================================================
-- VOLATILITY SCALER CONFIGURATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS config.volatility_scaler (
    id SERIAL PRIMARY KEY,
    version VARCHAR(16) NOT NULL UNIQUE,
    lookback_window INTEGER NOT NULL DEFAULT 60,
    quantiles DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[0.5, 0.75, 0.9],
    scale_factors DOUBLE PRECISION[] NOT NULL DEFAULT ARRAY[1.0, 0.75, 0.5, 0.25],
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

DELETE FROM config.volatility_scaler WHERE version = 'V19';

INSERT INTO config.volatility_scaler
    (version, lookback_window, quantiles, scale_factors, description)
VALUES
    ('V19', 60, ARRAY[0.5, 0.75, 0.9], ARRAY[1.0, 0.75, 0.5, 0.25],
     'Position sizing: crisis(p90+)=25%, high(p75+)=50%, med(p50+)=75%, normal=100%');

-- =============================================================================
-- HELPER VIEWS
-- =============================================================================

-- View for market features only
CREATE OR REPLACE VIEW config.v_market_features_v19 AS
SELECT
    name,
    observation_order,
    feature_group,
    source_column,
    source_table,
    transformation,
    norm_method,
    norm_mean,
    norm_std,
    clip_min,
    clip_max,
    compute_location
FROM config.feature_definitions
WHERE version = 'V19'
  AND feature_type = 'market'
  AND is_active = TRUE
ORDER BY observation_order;

-- View for state features only
CREATE OR REPLACE VIEW config.v_state_features_v19 AS
SELECT
    name,
    observation_order,
    feature_group,
    transformation,
    formula,
    clip_min,
    clip_max
FROM config.feature_definitions
WHERE version = 'V19'
  AND feature_type = 'state'
  AND is_active = TRUE
ORDER BY observation_order;

-- View for normalization stats (quick reference)
CREATE OR REPLACE VIEW config.v_normalization_stats_v19 AS
SELECT
    name,
    norm_method,
    norm_mean,
    norm_std,
    clip_min,
    clip_max
FROM config.feature_definitions
WHERE version = 'V19'
  AND is_active = TRUE
  AND norm_method != 'none'
ORDER BY feature_type, observation_order;

-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================

-- Verify feature counts
DO $$
DECLARE
    market_count INTEGER;
    state_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO market_count
    FROM config.feature_definitions
    WHERE version = 'V19' AND feature_type = 'market';

    SELECT COUNT(*) INTO state_count
    FROM config.feature_definitions
    WHERE version = 'V19' AND feature_type = 'state';

    IF market_count != 18 THEN
        RAISE EXCEPTION 'Expected 18 market features, got %', market_count;
    END IF;

    IF state_count != 12 THEN
        RAISE EXCEPTION 'Expected 12 state features, got %', state_count;
    END IF;

    RAISE NOTICE 'Feature validation passed: 18 market + 12 state = 30 total';
END $$;

-- Summary
SELECT
    feature_type,
    COUNT(*) as count,
    COUNT(CASE WHEN norm_method = 'z_score' THEN 1 END) as z_score_count,
    COUNT(CASE WHEN norm_method = 'clip_only' THEN 1 END) as clip_only_count,
    COUNT(CASE WHEN norm_method = 'none' THEN 1 END) as none_count
FROM config.feature_definitions
WHERE version = 'V19'
GROUP BY feature_type
ORDER BY feature_type;

COMMENT ON TABLE config.feature_definitions IS
'Single Source of Truth for feature definitions used in V19 model training.
These exact normalization parameters must be used during inference.';
