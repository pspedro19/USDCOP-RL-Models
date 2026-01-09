-- V20 Model Threshold Update
-- Run this when Docker/Postgres is available
-- Optimized thresholds from backtest grid search: 0.30/-0.30

-- Update V20 model with optimized thresholds
UPDATE multi_model.model_registry
SET config = config || '{"thresholds": {"long": 0.30, "short": -0.30, "optimized": true, "backtest_return_pct": 12.28, "sharpe": 1.19}}'::jsonb,
    updated_at = NOW()
WHERE model_id LIKE 'ppo_v20%';

-- Verify update
SELECT model_id, model_type, status,
       config->>'thresholds' as thresholds,
       config->>'observation_dim' as obs_dim,
       updated_at
FROM multi_model.model_registry
WHERE model_id LIKE 'ppo_v20%';

-- Also insert/update performance metrics
INSERT INTO multi_model.model_performance_metrics (model_id, metric_name, metric_value, recorded_at)
VALUES
    ('ppo_v20_macro', 'backtest_return_pct', 12.28, NOW()),
    ('ppo_v20_macro', 'sharpe_ratio', 1.19, NOW()),
    ('ppo_v20_macro', 'max_drawdown_pct', 7.96, NOW()),
    ('ppo_v20_macro', 'win_rate_pct', 49.2, NOW()),
    ('ppo_v20_macro', 'threshold_long', 0.30, NOW()),
    ('ppo_v20_macro', 'threshold_short', -0.30, NOW())
ON CONFLICT (model_id, metric_name) DO UPDATE
SET metric_value = EXCLUDED.metric_value,
    recorded_at = EXCLUDED.recorded_at;
