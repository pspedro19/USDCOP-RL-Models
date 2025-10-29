-- Add XGB and ENSEMBLE strategies to dimension table
-- Ensures all 5 strategies are available for multi-model dashboard

-- Insert XGB strategy (if not exists)
INSERT INTO dw.dim_strategy (strategy_code, strategy_name, strategy_type, description, config_json)
VALUES (
    'ML_XGB',
    'XGBoost Classifier',
    'ML',
    'XGBoost with meta-labeling and feature engineering',
    '{"algorithm": "XGBoost", "features": ["rsi_14", "macd", "bb_position", "atr_norm", "volume_ratio", "momentum"], "meta_labeling": true, "position_sizing": "kelly"}'::jsonb
)
ON CONFLICT (strategy_code) DO NOTHING;

-- Insert ENSEMBLE strategy (if not exists)
INSERT INTO dw.dim_strategy (strategy_code, strategy_name, strategy_type, description, config_json)
VALUES (
    'ENSEMBLE',
    'Multi-Model Ensemble',
    'ENSEMBLE',
    'Weighted consensus of RL, ML, and LLM strategies',
    '{"weights": {"RL_PPO": 0.3, "ML_LGBM": 0.25, "ML_XGB": 0.25, "LLM_CLAUDE": 0.2}, "method": "weighted_average", "min_consensus": 0.5}'::jsonb
)
ON CONFLICT (strategy_code) DO NOTHING;

-- Verify all 5 strategies exist
DO $$
DECLARE
    strategy_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO strategy_count FROM dw.dim_strategy WHERE is_active = TRUE;

    IF strategy_count >= 5 THEN
        RAISE NOTICE '✅ All 5 strategies present in dimension table';
    ELSE
        RAISE WARNING '⚠️ Only % strategies found. Expected 5.', strategy_count;
    END IF;
END $$;

-- Display all active strategies
SELECT
    strategy_id,
    strategy_code,
    strategy_name,
    strategy_type,
    description
FROM dw.dim_strategy
WHERE is_active = TRUE
ORDER BY strategy_code;
