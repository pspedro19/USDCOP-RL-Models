-- ============================================================================
-- 006_model_auto_register.sql
-- P1-18: Auto-registro de modelos post-entrenamiento
-- CLAUDE-T16 | Contrato: CTR-011
-- ============================================================================
--
-- Este script agrega funcionalidad para:
-- 1. Tracking de metricas de training con cada modelo
-- 2. Vista para listar modelos disponibles en frontend
-- 3. Funcion para obtener modelos con backtest metrics
-- ============================================================================

-- 1. Agregar columnas de training metrics si no existen
ALTER TABLE model_registry
ADD COLUMN IF NOT EXISTS training_duration_seconds NUMERIC(12,2),
ADD COLUMN IF NOT EXISTS total_timesteps BIGINT,
ADD COLUMN IF NOT EXISTS best_mean_reward NUMERIC(12,4),
ADD COLUMN IF NOT EXISTS training_config JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS eval_results JSONB DEFAULT '[]';

-- 2. Agregar columnas de backtest metrics para display en frontend
ALTER TABLE model_registry
ADD COLUMN IF NOT EXISTS backtest_sharpe NUMERIC(8,4),
ADD COLUMN IF NOT EXISTS backtest_max_drawdown NUMERIC(8,4),
ADD COLUMN IF NOT EXISTS backtest_win_rate NUMERIC(5,4),
ADD COLUMN IF NOT EXISTS backtest_total_trades INTEGER,
ADD COLUMN IF NOT EXISTS backtest_hold_percent NUMERIC(5,4),
ADD COLUMN IF NOT EXISTS backtest_test_period VARCHAR(100);

-- 3. Comentarios de documentacion
COMMENT ON COLUMN model_registry.training_duration_seconds IS
'Duracion del entrenamiento en segundos';

COMMENT ON COLUMN model_registry.total_timesteps IS
'Total de timesteps del entrenamiento';

COMMENT ON COLUMN model_registry.best_mean_reward IS
'Mejor reward promedio durante evaluacion';

COMMENT ON COLUMN model_registry.training_config IS
'Configuracion de hiperparametros usados en training (JSONB)';

COMMENT ON COLUMN model_registry.eval_results IS
'Resultados de evaluaciones periodicas durante training (JSONB array)';

COMMENT ON COLUMN model_registry.backtest_sharpe IS
'Sharpe ratio del backtest';

COMMENT ON COLUMN model_registry.backtest_max_drawdown IS
'Max drawdown del backtest (valor negativo, e.g., -0.12 = -12%)';

COMMENT ON COLUMN model_registry.backtest_win_rate IS
'Win rate del backtest (0-1)';

COMMENT ON COLUMN model_registry.backtest_total_trades IS
'Total de trades en backtest';

-- 4. Vista para frontend: modelos disponibles con metricas
CREATE OR REPLACE VIEW public.available_models AS
SELECT
    model_id,
    model_version,
    model_path,
    status,
    created_at,
    deployed_at,
    -- Training metrics
    training_duration_seconds,
    total_timesteps,
    best_mean_reward,
    -- Backtest metrics (prioridad: columnas dedicadas, luego test_* legacy)
    COALESCE(backtest_sharpe, test_sharpe) as sharpe,
    COALESCE(backtest_max_drawdown, test_max_drawdown) as max_drawdown,
    COALESCE(backtest_win_rate, test_win_rate) as win_rate,
    COALESCE(backtest_total_trades, 0) as total_trades,
    COALESCE(backtest_hold_percent, 0) as hold_percent,
    backtest_test_period as test_period,
    -- Metadata
    observation_dim,
    action_space,
    model_hash
FROM model_registry
WHERE status IN ('registered', 'deployed')
ORDER BY
    CASE status WHEN 'deployed' THEN 0 ELSE 1 END,
    created_at DESC;

COMMENT ON VIEW public.available_models IS
'Vista de modelos disponibles para seleccion en frontend. Incluye training y backtest metrics.';

-- 5. Funcion para obtener modelos en formato JSON para API
CREATE OR REPLACE FUNCTION get_available_models_json()
RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_agg(
        jsonb_build_object(
            'id', model_id,
            'name', CONCAT('PPO ', UPPER(model_version)),
            'algorithm', 'PPO',
            'version', UPPER(model_version),
            'status', CASE status WHEN 'deployed' THEN 'production' ELSE 'testing' END,
            'type', 'rl',
            'color', CASE
                WHEN model_version LIKE '%v19%' THEN '#10B981'
                WHEN model_version LIKE '%v20%' THEN '#06B6D4'
                ELSE '#8B5CF6'
            END,
            'description', CONCAT('PPO model ', model_version, ' - ', observation_dim, '-dim observation'),
            'isRealData', true,
            'dbModelId', model_id,
            'modelPath', model_path,
            'modelHash', model_hash,
            'backtest', jsonb_build_object(
                'sharpe', COALESCE(sharpe, 0),
                'maxDrawdown', COALESCE(max_drawdown, 0),
                'winRate', COALESCE(win_rate, 0),
                'totalTrades', COALESCE(total_trades, 0),
                'holdPercent', COALESCE(hold_percent, 0),
                'testPeriod', COALESCE(test_period, '')
            ),
            'training', jsonb_build_object(
                'duration_seconds', training_duration_seconds,
                'total_timesteps', total_timesteps,
                'best_mean_reward', best_mean_reward
            )
        )
        ORDER BY
            CASE status WHEN 'deployed' THEN 0 ELSE 1 END,
            created_at DESC
    )
    INTO result
    FROM available_models;

    RETURN COALESCE(result, '[]'::jsonb);
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_available_models_json() IS
'Retorna modelos disponibles en formato JSON listo para API frontend';

-- 6. Funcion para registrar backtest metrics de un modelo
CREATE OR REPLACE FUNCTION update_model_backtest_metrics(
    p_model_id VARCHAR(100),
    p_sharpe NUMERIC,
    p_max_drawdown NUMERIC,
    p_win_rate NUMERIC,
    p_total_trades INTEGER,
    p_hold_percent NUMERIC DEFAULT 0,
    p_test_period VARCHAR(100) DEFAULT NULL
)
RETURNS BOOLEAN AS $$
BEGIN
    UPDATE model_registry
    SET
        backtest_sharpe = p_sharpe,
        backtest_max_drawdown = p_max_drawdown,
        backtest_win_rate = p_win_rate,
        backtest_total_trades = p_total_trades,
        backtest_hold_percent = p_hold_percent,
        backtest_test_period = p_test_period
    WHERE model_id = p_model_id;

    RETURN FOUND;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION update_model_backtest_metrics IS
'Actualiza metricas de backtest para un modelo registrado';

-- 7. Insertar modelos de produccion existentes si no estan
INSERT INTO model_registry (
    model_id, model_version, model_path, model_hash, norm_stats_hash,
    observation_dim, action_space, status,
    backtest_sharpe, backtest_max_drawdown, backtest_win_rate,
    backtest_total_trades, backtest_hold_percent, backtest_test_period
)
VALUES
    -- PPO V19 Production
    (
        'ppo_v19_prod', 'v19', 'models/ppo_v19.zip', 'pending_hash', 'pending_hash',
        15, 3, 'deployed',
        1.85, -0.123, 0.58,
        2847, 0.15, '2024-01-01 to 2024-12-31'
    ),
    -- PPO V20 Production
    (
        'ppo_v20_prod', 'v20', 'models/ppo_v20.zip', 'pending_hash', 'pending_hash',
        15, 3, 'deployed',
        1.19, -0.0796, 0.492,
        11, 0.0, '2025-01-07 to 2025-12-05'
    )
ON CONFLICT (model_id) DO UPDATE SET
    status = EXCLUDED.status,
    backtest_sharpe = EXCLUDED.backtest_sharpe,
    backtest_max_drawdown = EXCLUDED.backtest_max_drawdown,
    backtest_win_rate = EXCLUDED.backtest_win_rate,
    backtest_total_trades = EXCLUDED.backtest_total_trades,
    backtest_hold_percent = EXCLUDED.backtest_hold_percent,
    backtest_test_period = EXCLUDED.backtest_test_period;

-- 8. Verificacion
DO $$
DECLARE
    model_count INTEGER;
    view_exists BOOLEAN;
    func_exists BOOLEAN;
BEGIN
    -- Contar modelos
    SELECT COUNT(*) INTO model_count FROM model_registry;

    -- Verificar vista
    SELECT EXISTS (
        SELECT 1 FROM information_schema.views
        WHERE table_schema = 'public' AND table_name = 'available_models'
    ) INTO view_exists;

    -- Verificar funcion
    SELECT EXISTS (
        SELECT 1 FROM pg_proc
        WHERE proname = 'get_available_models_json'
    ) INTO func_exists;

    IF model_count >= 2 AND view_exists AND func_exists THEN
        RAISE NOTICE 'Migration 006 completed: % models, view=%, func=%',
            model_count, view_exists, func_exists;
    ELSE
        RAISE WARNING 'Migration 006 incomplete: models=%, view=%, func=%',
            model_count, view_exists, func_exists;
    END IF;
END $$;

SELECT 'Migration 006_model_auto_register.sql completed' as status;
