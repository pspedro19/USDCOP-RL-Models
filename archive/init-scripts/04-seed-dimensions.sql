-- =====================================================
-- Seed Dimension Tables with Initial Data
-- =====================================================

-- Seed dim_symbol
INSERT INTO dw.dim_symbol (symbol_code, base_currency, quote_currency, symbol_type, exchange)
VALUES
    ('USD/COP', 'USD', 'COP', 'forex', 'Colombian Market'),
    ('EUR/COP', 'EUR', 'COP', 'forex', 'Colombian Market'),
    ('BTC/USD', 'BTC', 'USD', 'crypto', 'Binance')
ON CONFLICT (symbol_code) DO UPDATE SET
    updated_at = NOW();

-- Seed dim_source
INSERT INTO dw.dim_source (source_name, source_type, api_endpoint, cost_per_call, rate_limit_per_min)
VALUES
    ('twelvedata', 'api', 'https://api.twelvedata.com', 0.0, 8),
    ('alphavantage', 'api', 'https://www.alphavantage.co', 0.0, 5),
    ('manual', 'file', NULL, 0.0, NULL),
    ('websocket', 'stream', NULL, 0.0, NULL)
ON CONFLICT (source_name) DO UPDATE SET
    updated_at = NOW();

-- Populate dim_time_5m (2020-2030)
SELECT dw.populate_dim_time_5m('2020-01-01'::DATE, '2030-12-31'::DATE);

-- Seed dim_feature (L4 observations)
INSERT INTO dw.dim_feature (feature_name, feature_type, tier, lag_bars, normalization_method, is_trainable)
VALUES
    ('obs_00', 'volatility', 1, 7, 'median_mad', TRUE),
    ('obs_01', 'volatility', 1, 7, 'median_mad', TRUE),
    ('obs_02', 'shape', 1, 7, 'median_mad', TRUE),
    ('obs_03', 'shape', 1, 7, 'median_mad', TRUE),
    ('obs_04', 'momentum', 1, 7, 'median_mad', TRUE),
    ('obs_05', 'volatility', 1, 7, 'median_mad', TRUE),
    ('obs_06', 'trend', 1, 7, 'median_mad', TRUE),
    ('obs_07', 'entropy', 1, 7, 'median_mad', TRUE),
    ('obs_08', 'momentum', 2, 7, 'median_mad', TRUE),
    ('obs_09', 'shape', 2, 7, 'median_mad', TRUE),
    ('obs_10', 'volatility', 2, 7, 'median_mad', TRUE),
    ('obs_11', 'momentum', 2, 7, 'median_mad', TRUE),
    ('obs_12', 'momentum', 2, 7, 'median_mad', TRUE),
    ('obs_13', 'volatility', 2, 7, 'median_mad', TRUE),
    ('obs_14', 'cyclical', 1, 0, 'passthrough', TRUE),
    ('obs_15', 'cyclical', 1, 0, 'passthrough', TRUE),
    ('obs_16', 'cost', 1, 0, 'median_mad', TRUE)
ON CONFLICT (feature_name) DO UPDATE SET
    updated_at = NOW();

-- Seed dim_indicator (common technical indicators)
INSERT INTO dw.dim_indicator (indicator_name, indicator_family, calculation_library, params, interpretation, signal_thresholds)
VALUES
    ('RSI', 'momentum', 'talib', '{"period": 14}', 'Relative Strength Index', '{"overbought": 70, "oversold": 30}'),
    ('MACD', 'momentum', 'talib', '{"fast": 12, "slow": 26, "signal": 9}', 'Moving Average Convergence Divergence', NULL),
    ('EMA_20', 'trend', 'talib', '{"period": 20}', 'Exponential Moving Average 20', NULL),
    ('EMA_50', 'trend', 'talib', '{"period": 50}', 'Exponential Moving Average 50', NULL),
    ('SMA_20', 'trend', 'talib', '{"period": 20}', 'Simple Moving Average 20', NULL),
    ('BB_UPPER', 'volatility', 'talib', '{"period": 20, "std": 2}', 'Bollinger Band Upper', NULL),
    ('BB_LOWER', 'volatility', 'talib', '{"period": 20, "std": 2}', 'Bollinger Band Lower', NULL),
    ('ATR', 'volatility', 'talib', '{"period": 14}', 'Average True Range', NULL),
    ('STOCH', 'momentum', 'talib', '{"k_period": 14, "d_period": 3}', 'Stochastic Oscillator', '{"overbought": 80, "oversold": 20}'),
    ('ADX', 'trend', 'talib', '{"period": 14}', 'Average Directional Index', '{"strong_trend": 25}')
ON CONFLICT (indicator_name) DO UPDATE SET
    params = EXCLUDED.params;

-- Seed default model
INSERT INTO dw.dim_model (
    model_id, model_name, algorithm, architecture, framework, version, is_production, is_current
)
VALUES
    ('rl_baseline_v1.0', 'Baseline PPO Model', 'PPO', 'MLP-64-64', 'stable-baselines3', 'v1.0', FALSE, FALSE),
    ('rl_ppo_v1.2', 'Production PPO v1.2', 'PPO', 'MLP-128-128', 'stable-baselines3', 'v1.2', TRUE, TRUE)
ON CONFLICT DO NOTHING;

-- Success message
DO $$
DECLARE
    symbol_count INT;
    source_count INT;
    time_count INT;
    feature_count INT;
    indicator_count INT;
    model_count INT;
BEGIN
    SELECT COUNT(*) INTO symbol_count FROM dw.dim_symbol;
    SELECT COUNT(*) INTO source_count FROM dw.dim_source;
    SELECT COUNT(*) INTO time_count FROM dw.dim_time_5m;
    SELECT COUNT(*) INTO feature_count FROM dw.dim_feature;
    SELECT COUNT(*) INTO indicator_count FROM dw.dim_indicator;
    SELECT COUNT(*) INTO model_count FROM dw.dim_model;

    RAISE NOTICE '';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '✅ Dimension Tables Seeded Successfully';
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'dim_symbol:     % rows', symbol_count;
    RAISE NOTICE 'dim_source:     % rows', source_count;
    RAISE NOTICE 'dim_time_5m:    % rows (2020-2030)', time_count;
    RAISE NOTICE 'dim_feature:    % rows', feature_count;
    RAISE NOTICE 'dim_indicator:  % rows', indicator_count;
    RAISE NOTICE 'dim_model:      % rows', model_count;
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
END $$;
