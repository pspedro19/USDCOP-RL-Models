-- ============================================================================
-- 12-trades-metadata.sql
-- Extensión de trades_history para metadata enriquecida del modelo
-- ============================================================================
--
-- Este script añade columnas JSONB para almacenar:
-- - Confianza del modelo al momento del trade
-- - Probabilidades de acción (HOLD, LONG, SHORT)
-- - Snapshot de features técnicos
-- - Régimen de mercado detectado
-- - Excursiones máximas (MAE/MFE)
--
-- Ejecutar después de 11-paper-trading-tables.sql
-- ============================================================================

-- 1. Añadir columnas de metadata del modelo
ALTER TABLE public.trades_history
ADD COLUMN IF NOT EXISTS entry_confidence NUMERIC(4,3),
ADD COLUMN IF NOT EXISTS exit_confidence NUMERIC(4,3),
ADD COLUMN IF NOT EXISTS model_metadata JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS features_snapshot JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS market_regime VARCHAR(20),
ADD COLUMN IF NOT EXISTS max_adverse_excursion NUMERIC(12,4),
ADD COLUMN IF NOT EXISTS max_favorable_excursion NUMERIC(12,4);

-- 2. Índices para queries eficientes

-- Índice para filtrar por confianza (detectar overfit)
CREATE INDEX IF NOT EXISTS idx_trades_entry_confidence
ON public.trades_history(entry_confidence);

-- Índice para filtrar por régimen de mercado
CREATE INDEX IF NOT EXISTS idx_trades_market_regime
ON public.trades_history(market_regime);

-- Índice GIN para búsquedas en JSONB (model_metadata)
CREATE INDEX IF NOT EXISTS idx_trades_model_metadata
ON public.trades_history USING GIN(model_metadata);

-- Índice GIN para búsquedas en JSONB (features_snapshot)
CREATE INDEX IF NOT EXISTS idx_trades_features_snapshot
ON public.trades_history USING GIN(features_snapshot);

-- Índice compuesto para análisis de overfitting
CREATE INDEX IF NOT EXISTS idx_trades_confidence_pnl
ON public.trades_history(entry_confidence, pnl_usd);

-- 3. Comentarios para documentación
COMMENT ON COLUMN public.trades_history.entry_confidence IS
'Confianza del modelo al entrar (0.0-1.0). Valores > 0.9 con entropy < 0.1 pueden indicar overfit.';

COMMENT ON COLUMN public.trades_history.exit_confidence IS
'Confianza del modelo al salir (0.0-1.0). Opcional, solo si hay señal de salida explícita.';

COMMENT ON COLUMN public.trades_history.model_metadata IS
'JSONB con metadata del modelo: {confidence, action_probs: [hold, long, short], critic_value, entropy, advantage}';

COMMENT ON COLUMN public.trades_history.features_snapshot IS
'JSONB con snapshot de features al momento del trade: {rsi_14, macd_histogram, bb_position, volume_zscore, trend_ema_cross, hour_of_day, day_of_week}';

COMMENT ON COLUMN public.trades_history.market_regime IS
'Régimen de mercado detectado: trending, ranging, volatile, unknown';

COMMENT ON COLUMN public.trades_history.max_adverse_excursion IS
'MAE: Máxima excursión adversa durante el trade (pérdida no realizada máxima)';

COMMENT ON COLUMN public.trades_history.max_favorable_excursion IS
'MFE: Máxima excursión favorable durante el trade (ganancia no realizada máxima)';

-- 4. Vista para análisis de overfitting
CREATE OR REPLACE VIEW public.trades_overfit_analysis AS
SELECT
    id,
    model_id,
    side,
    entry_time,
    pnl_usd,
    pnl_pct,
    entry_confidence,
    (model_metadata->>'entropy')::NUMERIC as entropy,
    (model_metadata->>'critic_value')::NUMERIC as critic_value,
    market_regime,
    -- Flag de posible overfit: alta confianza + baja entropy + pérdida
    CASE
        WHEN entry_confidence > 0.9
         AND (model_metadata->>'entropy')::NUMERIC < 0.1
         AND pnl_usd < 0
        THEN TRUE
        ELSE FALSE
    END as possible_overfit
FROM public.trades_history
WHERE model_metadata IS NOT NULL
  AND model_metadata != '{}'::jsonb;

COMMENT ON VIEW public.trades_overfit_analysis IS
'Vista para análisis de posibles trades con overfit: alta confianza + baja entropy + pérdida';

-- 5. Función para extraer action_probs como array
CREATE OR REPLACE FUNCTION get_action_probs(metadata JSONB)
RETURNS NUMERIC[] AS $$
BEGIN
    IF metadata IS NULL OR metadata = '{}'::jsonb THEN
        RETURN ARRAY[0.33, 0.33, 0.34]::NUMERIC[];
    END IF;

    RETURN ARRAY[
        COALESCE((metadata->'action_probs'->>0)::NUMERIC, 0.33),
        COALESCE((metadata->'action_probs'->>1)::NUMERIC, 0.33),
        COALESCE((metadata->'action_probs'->>2)::NUMERIC, 0.34)
    ];
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION get_action_probs(JSONB) IS
'Extrae action_probs del model_metadata como array [hold, long, short]';

-- 6. Verificación
DO $$
DECLARE
    col_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO col_count
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'trades_history'
      AND column_name IN ('entry_confidence', 'model_metadata', 'features_snapshot',
                          'market_regime', 'max_adverse_excursion', 'max_favorable_excursion');

    IF col_count = 6 THEN
        RAISE NOTICE '✓ Todas las columnas de metadata creadas correctamente';
    ELSE
        RAISE WARNING '⚠ Solo se encontraron % de 6 columnas esperadas', col_count;
    END IF;
END $$;

SELECT 'Migración 12-trades-metadata.sql completada' as status;
