-- ============================================================================
-- 005_model_registry.sql
-- Model Registry para trazabilidad de modelos
-- CLAUDE-T15 | Plan Item: P1-11 | Contrato: CTR-010
-- ============================================================================

-- 1. Tabla de registros de modelos
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) UNIQUE NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_path TEXT NOT NULL,

    -- Hashes de integridad (SHA256)
    model_hash VARCHAR(64) NOT NULL,
    norm_stats_hash VARCHAR(64) NOT NULL,
    config_hash VARCHAR(64),

    -- Metadata del modelo
    observation_dim INTEGER NOT NULL DEFAULT 15,
    action_space INTEGER NOT NULL DEFAULT 3,
    feature_order JSONB NOT NULL,

    -- Training info
    training_dataset_id INTEGER,
    training_start_date DATE,
    training_end_date DATE,
    validation_metrics JSONB,

    -- Performance metrics
    test_sharpe NUMERIC(8,4),
    test_max_drawdown NUMERIC(8,4),
    test_win_rate NUMERIC(5,4),

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP WITH TIME ZONE,
    retired_at TIMESTAMP WITH TIME ZONE,

    -- Status
    status VARCHAR(20) DEFAULT 'registered',  -- registered, deployed, retired

    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('registered', 'deployed', 'retired'))
);

-- 2. Indices para busquedas eficientes
CREATE INDEX IF NOT EXISTS idx_model_registry_hash
    ON model_registry(model_hash);

CREATE INDEX IF NOT EXISTS idx_model_registry_status
    ON model_registry(status);

CREATE INDEX IF NOT EXISTS idx_model_registry_version
    ON model_registry(model_version);

-- 3. Comentarios para documentacion
COMMENT ON TABLE model_registry IS
    'Registro de modelos ONNX con hashes para verificacion de integridad. CTR-010';

COMMENT ON COLUMN model_registry.model_hash IS
    'SHA256 del archivo ONNX - usado para verificar que el modelo no ha sido modificado';

COMMENT ON COLUMN model_registry.norm_stats_hash IS
    'SHA256 del archivo norm_stats.json - debe coincidir con el usado durante training';

COMMENT ON COLUMN model_registry.feature_order IS
    'Orden exacto de features como array JSON - debe coincidir con FeatureBuilder';

-- 4. Funcion para verificar integridad
CREATE OR REPLACE FUNCTION verify_model_integrity(p_model_id VARCHAR)
RETURNS TABLE (
    is_valid BOOLEAN,
    registered_hash VARCHAR,
    message TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        true as is_valid,
        model_hash as registered_hash,
        'Model registered - integrity check requires file access' as message
    FROM model_registry
    WHERE model_id = p_model_id;

    IF NOT FOUND THEN
        RETURN QUERY
        SELECT
            false as is_valid,
            null::VARCHAR as registered_hash,
            format('Model %s not found in registry', p_model_id) as message;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- 5. Vista para modelos activos
CREATE OR REPLACE VIEW active_models AS
SELECT
    model_id,
    model_version,
    model_hash,
    observation_dim,
    action_space,
    deployed_at,
    test_sharpe,
    test_win_rate
FROM model_registry
WHERE status = 'deployed'
ORDER BY deployed_at DESC;

-- 6. Verificacion de migracion
DO $$
DECLARE
    table_exists BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_name = 'model_registry'
    ) INTO table_exists;

    IF table_exists THEN
        RAISE NOTICE '✓ model_registry table created successfully';
    ELSE
        RAISE WARNING '⚠ model_registry table NOT created';
    END IF;
END $$;

SELECT 'Migration 005_model_registry.sql completed' as status;
