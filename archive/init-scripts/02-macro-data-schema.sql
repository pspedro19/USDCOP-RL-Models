-- ============================================================================
-- SCHEMA: Macro Economic Data (WTI, DXY, etc.)
-- Versión: 1.0
-- Fecha: 2025-11-05
-- Propósito: Almacenar datos macro para feature engineering en RL system
-- ============================================================================

-- Crear extensión TimescaleDB si no existe (debería estar ya instalada)
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- TABLA: macro_ohlcv
-- ============================================================================

CREATE TABLE IF NOT EXISTS macro_ohlcv (
    -- Timestamp (time-series)
    time TIMESTAMP WITH TIME ZONE NOT NULL,

    -- Símbolo macro (WTI, DXY, etc.)
    symbol TEXT NOT NULL,

    -- Datos OHLCV
    open DECIMAL(12,6) NOT NULL,
    high DECIMAL(12,6) NOT NULL,
    low DECIMAL(12,6) NOT NULL,
    close DECIMAL(12,6) NOT NULL,
    volume BIGINT DEFAULT 0,

    -- Metadata
    source TEXT DEFAULT 'twelvedata',  -- 'twelvedata', 'investing.com_manual', etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Primary key
    PRIMARY KEY (time, symbol),

    -- Constraints para integridad de datos OHLC
    CONSTRAINT chk_macro_high_gte_low CHECK (high >= low),
    CONSTRAINT chk_macro_close_in_range CHECK (close >= low AND close <= high),
    CONSTRAINT chk_macro_open_in_range CHECK (open >= low AND open <= high),
    CONSTRAINT chk_macro_high_gte_open CHECK (high >= open),
    CONSTRAINT chk_macro_high_gte_close CHECK (high >= close),
    CONSTRAINT chk_macro_low_lte_open CHECK (low <= open),
    CONSTRAINT chk_macro_low_lte_close CHECK (low <= close),
    CONSTRAINT chk_macro_positive_prices CHECK (open > 0 AND high > 0 AND low > 0 AND close > 0),
    CONSTRAINT chk_macro_volume_non_negative CHECK (volume >= 0)
);

-- ============================================================================
-- TIMESCALEDB HYPERTABLE
-- ============================================================================

-- Convertir a hypertable (optimización para time-series)
SELECT create_hypertable(
    'macro_ohlcv',
    'time',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 month'  -- Chunks de 1 mes
);

-- ============================================================================
-- ÍNDICES
-- ============================================================================

-- Índice por símbolo y tiempo (para queries por símbolo específico)
CREATE INDEX IF NOT EXISTS idx_macro_symbol_time
ON macro_ohlcv (symbol, time DESC);

-- Índice por source (para auditoría)
CREATE INDEX IF NOT EXISTS idx_macro_source
ON macro_ohlcv (source);

-- Índice por fecha (sin hora) para queries por día
CREATE INDEX IF NOT EXISTS idx_macro_date
ON macro_ohlcv (DATE(time));

-- ============================================================================
-- TRIGGER: Auto-update updated_at
-- ============================================================================

CREATE OR REPLACE FUNCTION update_macro_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_macro_timestamp
BEFORE UPDATE ON macro_ohlcv
FOR EACH ROW
EXECUTE FUNCTION update_macro_updated_at();

-- ============================================================================
-- COMENTARIOS (Documentación)
-- ============================================================================

COMMENT ON TABLE macro_ohlcv IS
'Datos macro económicos (WTI Crude Oil, US Dollar Index, etc.) para feature engineering en USD/COP RL trading system. Almacena datos 1h que se resamplearán a 5min en L3.';

COMMENT ON COLUMN macro_ohlcv.time IS
'Timestamp con timezone (UTC o America/New_York según símbolo)';

COMMENT ON COLUMN macro_ohlcv.symbol IS
'Símbolo macro: WTI (crude oil), DXY (US dollar index), etc.';

COMMENT ON COLUMN macro_ohlcv.open IS
'Precio apertura (open)';

COMMENT ON COLUMN macro_ohlcv.high IS
'Precio máximo (high)';

COMMENT ON COLUMN macro_ohlcv.low IS
'Precio mínimo (low)';

COMMENT ON COLUMN macro_ohlcv.close IS
'Precio cierre (close)';

COMMENT ON COLUMN macro_ohlcv.volume IS
'Volumen negociado';

COMMENT ON COLUMN macro_ohlcv.source IS
'Fuente de datos: twelvedata, investing.com_manual, etc.';

COMMENT ON COLUMN macro_ohlcv.created_at IS
'Timestamp de creación del registro';

COMMENT ON COLUMN macro_ohlcv.updated_at IS
'Timestamp de última actualización';

-- ============================================================================
-- FUNCIONES AUXILIARES
-- ============================================================================

-- Función: Obtener estadísticas de la tabla
CREATE OR REPLACE FUNCTION get_macro_stats()
RETURNS TABLE (
    symbol TEXT,
    record_count BIGINT,
    min_time TIMESTAMP WITH TIME ZONE,
    max_time TIMESTAMP WITH TIME ZONE,
    days_coverage INT,
    source TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        m.symbol,
        COUNT(*)::BIGINT as record_count,
        MIN(m.time) as min_time,
        MAX(m.time) as max_time,
        (EXTRACT(EPOCH FROM (MAX(m.time) - MIN(m.time))) / 86400)::INT as days_coverage,
        m.source
    FROM macro_ohlcv m
    GROUP BY m.symbol, m.source
    ORDER BY m.symbol, m.source;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION get_macro_stats() IS
'Obtiene estadísticas de cobertura de datos macro por símbolo';

-- Función: Detectar gaps en datos
CREATE OR REPLACE FUNCTION detect_macro_gaps(p_symbol TEXT, p_interval INTERVAL DEFAULT '1 hour')
RETURNS TABLE (
    gap_start TIMESTAMP WITH TIME ZONE,
    gap_end TIMESTAMP WITH TIME ZONE,
    gap_duration INTERVAL
) AS $$
BEGIN
    RETURN QUERY
    WITH time_diffs AS (
        SELECT
            time,
            LEAD(time) OVER (ORDER BY time) as next_time,
            LEAD(time) OVER (ORDER BY time) - time as diff
        FROM macro_ohlcv
        WHERE symbol = p_symbol
    )
    SELECT
        time as gap_start,
        next_time as gap_end,
        diff as gap_duration
    FROM time_diffs
    WHERE diff > p_interval * 1.5  -- 1.5x el intervalo esperado
    ORDER BY time;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION detect_macro_gaps(TEXT, INTERVAL) IS
'Detecta gaps en datos macro para un símbolo específico';

-- ============================================================================
-- DATOS INICIALES / SEED (Opcional)
-- ============================================================================

-- Insertar registro de prueba (comentado, descomentar si se necesita)
-- INSERT INTO macro_ohlcv (time, symbol, open, high, low, close, volume, source)
-- VALUES
--     (NOW(), 'WTI', 75.00, 75.50, 74.50, 75.20, 1000000, 'test'),
--     (NOW(), 'DXY', 103.00, 103.20, 102.80, 103.10, 500000, 'test')
-- ON CONFLICT (time, symbol) DO NOTHING;

-- ============================================================================
-- GRANTS (Permisos)
-- ============================================================================

-- Dar permisos al usuario de la aplicación
GRANT SELECT, INSERT, UPDATE ON TABLE macro_ohlcv TO usdcop;
GRANT EXECUTE ON FUNCTION get_macro_stats() TO usdcop;
GRANT EXECUTE ON FUNCTION detect_macro_gaps(TEXT, INTERVAL) TO usdcop;

-- ============================================================================
-- VERIFICACIÓN
-- ============================================================================

-- Verificar que la tabla fue creada correctamente
DO $$
DECLARE
    v_table_exists BOOLEAN;
    v_is_hypertable BOOLEAN;
BEGIN
    -- Verificar tabla existe
    SELECT EXISTS (
        SELECT FROM information_schema.tables
        WHERE table_name = 'macro_ohlcv'
    ) INTO v_table_exists;

    -- Verificar es hypertable
    SELECT EXISTS (
        SELECT FROM timescaledb_information.hypertables
        WHERE hypertable_name = 'macro_ohlcv'
    ) INTO v_is_hypertable;

    IF v_table_exists AND v_is_hypertable THEN
        RAISE NOTICE '✅ Tabla macro_ohlcv creada correctamente como hypertable';
    ELSIF v_table_exists THEN
        RAISE WARNING '⚠️  Tabla macro_ohlcv creada pero NO es hypertable';
    ELSE
        RAISE EXCEPTION '❌ Error: Tabla macro_ohlcv NO fue creada';
    END IF;
END;
$$;

-- Mostrar estadísticas iniciales (debería estar vacía)
SELECT * FROM get_macro_stats();

-- ============================================================================
-- FIN DEL SCHEMA
-- ============================================================================

-- Imprimir resumen
\echo '============================================'
\echo 'Schema macro_ohlcv creado exitosamente'
\echo '============================================'
\echo ''
\echo 'Tabla creada: macro_ohlcv'
\echo 'Tipo: TimescaleDB hypertable'
\echo 'Índices: 3 (symbol_time, source, date)'
\echo 'Funciones auxiliares: 2 (get_macro_stats, detect_macro_gaps)'
\echo ''
\echo 'Próximo paso:'
\echo '  1. Ejecutar: python scripts/verify_twelvedata_macro.py'
\echo '  2. Si OK: Crear DAG L0 macro'
\echo '  3. Si FALLA: Usar upload_macro_manual.py'
\echo '============================================'
