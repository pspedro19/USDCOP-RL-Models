-- ============================================================================
-- 004_model_metadata.sql
-- Model Metadata para trazabilidad de predicciones
-- CLAUDE-T13 | Plan Item: P1-2 | Contrato: CTR-009
-- ============================================================================

-- 1. Agregar columna model_metadata a trades
ALTER TABLE trades ADD COLUMN IF NOT EXISTS model_metadata JSONB;

-- 2. Indices para busquedas eficientes en metadata
CREATE INDEX IF NOT EXISTS idx_trades_model_hash
    ON trades((model_metadata->>'model_hash'));

CREATE INDEX IF NOT EXISTS idx_trades_bid_ask
    ON trades((model_metadata->>'bid_ask_spread'));

CREATE INDEX IF NOT EXISTS idx_trades_confidence
    ON trades((model_metadata->>'confidence'));

CREATE INDEX IF NOT EXISTS idx_trades_entropy
    ON trades((model_metadata->>'entropy'));

CREATE INDEX IF NOT EXISTS idx_trades_model_version
    ON trades((model_metadata->>'model_version'));

-- 3. Indice GIN para busquedas complejas en JSONB
CREATE INDEX IF NOT EXISTS idx_trades_model_metadata_gin
    ON trades USING GIN (model_metadata);

-- 4. Comentarios para documentacion
COMMENT ON COLUMN trades.model_metadata IS
    'Metadata JSONB capturada al momento de cada prediccion. Incluye: model_id, model_hash, observation, raw_features, bid_ask_spread, action_probabilities, confidence, entropy. CTR-009';

-- 5. Vista para analisis de predicciones
CREATE OR REPLACE VIEW prediction_analysis AS
SELECT
    t.id as trade_id,
    t.timestamp,
    t.action,
    t.model_metadata->>'model_id' as model_id,
    t.model_metadata->>'model_version' as model_version,
    (t.model_metadata->>'confidence')::NUMERIC as confidence,
    (t.model_metadata->>'entropy')::NUMERIC as entropy,
    (t.model_metadata->>'bid_ask_spread')::NUMERIC as bid_ask_spread,
    (t.model_metadata->>'market_volatility')::NUMERIC as market_volatility,
    t.model_metadata->'action_probabilities' as action_probabilities,
    t.model_metadata->'observation' as observation,
    t.pnl,
    t.exit_timestamp
FROM trades t
WHERE t.model_metadata IS NOT NULL
ORDER BY t.timestamp DESC;

-- 6. Funcion para analizar drift de confianza
CREATE OR REPLACE FUNCTION analyze_confidence_drift(
    p_model_id VARCHAR,
    p_lookback_days INTEGER DEFAULT 7
) RETURNS TABLE (
    date DATE,
    avg_confidence NUMERIC,
    avg_entropy NUMERIC,
    trade_count BIGINT,
    win_rate NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        DATE(t.timestamp) as date,
        AVG((t.model_metadata->>'confidence')::NUMERIC) as avg_confidence,
        AVG((t.model_metadata->>'entropy')::NUMERIC) as avg_entropy,
        COUNT(*) as trade_count,
        AVG(CASE WHEN t.pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate
    FROM trades t
    WHERE t.model_metadata->>'model_id' = p_model_id
      AND t.timestamp >= CURRENT_DATE - p_lookback_days
      AND t.model_metadata IS NOT NULL
    GROUP BY DATE(t.timestamp)
    ORDER BY date DESC;
END;
$$ LANGUAGE plpgsql;

-- 7. Funcion para detectar bid_ask anomalies
CREATE OR REPLACE FUNCTION detect_bid_ask_anomalies(
    p_threshold_multiplier NUMERIC DEFAULT 3.0
) RETURNS TABLE (
    trade_id INTEGER,
    timestamp TIMESTAMP WITH TIME ZONE,
    bid_ask_spread NUMERIC,
    avg_spread NUMERIC,
    is_anomaly BOOLEAN
) AS $$
DECLARE
    v_avg_spread NUMERIC;
    v_std_spread NUMERIC;
BEGIN
    -- Calcular estadisticas
    SELECT
        AVG((model_metadata->>'bid_ask_spread')::NUMERIC),
        STDDEV((model_metadata->>'bid_ask_spread')::NUMERIC)
    INTO v_avg_spread, v_std_spread
    FROM trades
    WHERE model_metadata->>'bid_ask_spread' IS NOT NULL;

    RETURN QUERY
    SELECT
        t.id as trade_id,
        t.timestamp,
        (t.model_metadata->>'bid_ask_spread')::NUMERIC as bid_ask_spread,
        v_avg_spread as avg_spread,
        ABS((t.model_metadata->>'bid_ask_spread')::NUMERIC - v_avg_spread) >
            (p_threshold_multiplier * COALESCE(v_std_spread, 0)) as is_anomaly
    FROM trades t
    WHERE t.model_metadata->>'bid_ask_spread' IS NOT NULL
    ORDER BY t.timestamp DESC;
END;
$$ LANGUAGE plpgsql;

-- 8. Verificacion de migracion
DO $$
DECLARE
    column_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'trades'
          AND column_name = 'model_metadata'
    ) INTO column_exists;

    IF column_exists THEN
        RAISE NOTICE '✓ model_metadata column added to trades table';
    ELSE
        RAISE WARNING '⚠ model_metadata column NOT added';
    END IF;
END $$;

SELECT 'Migration 004_model_metadata.sql completed' as status;
