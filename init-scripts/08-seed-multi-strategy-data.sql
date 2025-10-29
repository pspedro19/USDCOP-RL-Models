-- ============================================================================
-- Script: 08-seed-multi-strategy-data.sql
-- Propósito: Añadir estrategias ML_XGB y ENSEMBLE a dim_strategy
-- Fecha: 2025-10-27
-- ============================================================================

\c usdcop_trading

SET search_path TO dw;

-- Añadir ML_XGB y ENSEMBLE si no existen
INSERT INTO dw.dim_strategy (strategy_code, strategy_name, strategy_type, description, is_active, created_at)
VALUES
  (
    'ML_XGB',
    'XGBoost Classifier',
    'ML',
    'XGBoost gradient boosting classifier for price direction prediction',
    TRUE,
    NOW()
  ),
  (
    'ENSEMBLE',
    'Multi-Model Ensemble',
    'ENSEMBLE',
    'Weighted ensemble of RL, ML, and LLM strategies',
    TRUE,
    NOW()
  )
ON CONFLICT (strategy_code) DO UPDATE SET
  is_active = EXCLUDED.is_active,
  description = EXCLUDED.description;

-- Verificar estrategias activas
SELECT strategy_id, strategy_code, strategy_name, strategy_type, is_active
FROM dw.dim_strategy
WHERE is_active = TRUE
ORDER BY strategy_code;

\echo '✓ Multi-strategy dimension data seeded successfully'
