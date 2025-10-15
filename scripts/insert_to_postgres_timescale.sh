#!/bin/bash
# Script para insertar dataset unificado a PostgreSQL/TimescaleDB (manejo especial para hypertables)

echo "🐘 INSERCIÓN DE DATASET UNIFICADO A POSTGRESQL/TIMESCALEDB"
echo "=========================================================="

# Verificar que el archivo unificado existe
echo "🔍 Verificando dataset unificado en MinIO..."
if ! docker exec usdcop-minio test -f /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv; then
    echo "❌ Error: Archivo LATEST.csv no encontrado"
    echo "💡 Ejecuta primero: scripts/extract_and_fuse_real.sh"
    exit 1
fi

# Crear directorio temporal
TEMP_DIR="/tmp/timescale_insert_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"
echo "📁 Directorio temporal: $TEMP_DIR"

# Descargar archivo desde MinIO
echo "📥 Descargando dataset unificado desde MinIO..."
UNIFIED_FILE="$TEMP_DIR/usdcop_unified_data.csv"

if docker exec usdcop-minio cat /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv > "$UNIFIED_FILE"; then
    SIZE=$(wc -c < "$UNIFIED_FILE")
    LINES=$(wc -l < "$UNIFIED_FILE")
    echo "✅ Descargado: $LINES líneas, $SIZE bytes"
else
    echo "❌ Error descargando archivo"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Limpiar archivo CSV
echo "🧹 Limpiando archivo CSV..."
CLEAN_FILE="$TEMP_DIR/usdcop_clean_data.csv"
sed 's/^[^a-zA-Z0-9,_-]*//g' "$UNIFIED_FILE" | grep -E '^(time|[0-9])' > "$CLEAN_FILE"

CLEAN_LINES=$(wc -l < "$CLEAN_FILE")
echo "✅ Archivo limpio: $CLEAN_LINES líneas"

# Verificar conexión PostgreSQL
echo "🔌 Verificando conexión a PostgreSQL/TimescaleDB..."
if ! docker exec usdcop-postgres-timescale pg_isready -U admin > /dev/null 2>&1; then
    echo "❌ PostgreSQL no disponible"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Verificar si la tabla existe y es una hypertable
echo "🔍 Verificando estado actual de la tabla market_data..."

CHECK_TABLE="
DO \$\$
BEGIN
    -- Verificar si la tabla existe
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'market_data') THEN
        RAISE NOTICE 'Tabla market_data ya existe';

        -- Verificar si es una hypertable
        IF EXISTS (SELECT FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data') THEN
            RAISE NOTICE 'market_data es una hypertable de TimescaleDB';
        ELSE
            RAISE NOTICE 'market_data es una tabla regular';
        END IF;
    ELSE
        RAISE NOTICE 'Tabla market_data no existe, será creada';
    END IF;
END \$\$;

-- Mostrar estadísticas actuales si existe
SELECT
    CASE
        WHEN EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'market_data')
        THEN (SELECT COUNT(*)::text FROM market_data)
        ELSE '0 (tabla no existe)'
    END as registros_actuales;
"

docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$CHECK_TABLE"

# Crear/modificar tabla para TimescaleDB
echo "🗃️ Preparando tabla market_data para TimescaleDB..."

SQL_PREPARE_TABLE="
-- Crear tabla si no existe (compatible con TimescaleDB)
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume NUMERIC DEFAULT 0,
    timestamp_cot TIMESTAMPTZ,
    hour_cot INTEGER,
    weekday INTEGER,
    source VARCHAR(100) DEFAULT 'unknown',
    timezone VARCHAR(50) DEFAULT 'America/Bogota',
    batch_id VARCHAR(200),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convertir a hypertable si no lo es ya (TimescaleDB)
SELECT CASE
    WHEN NOT EXISTS (SELECT FROM timescaledb_information.hypertables WHERE hypertable_name = 'market_data')
    THEN create_hypertable('market_data', 'time', if_not_exists => TRUE)::text
    ELSE 'Hypertable ya existe'::text
END as hypertable_status;

-- Crear índices compatibles con TimescaleDB
CREATE INDEX IF NOT EXISTS idx_market_data_time
ON market_data (time);

CREATE INDEX IF NOT EXISTS idx_market_data_source
ON market_data (source, time);

CREATE INDEX IF NOT EXISTS idx_market_data_created_at
ON market_data (created_at);
"

if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$SQL_PREPARE_TABLE"; then
    echo "✅ Tabla market_data preparada como hypertable"
else
    echo "❌ Error preparando tabla"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copiar archivo al contenedor PostgreSQL
echo "📤 Copiando archivo al contenedor PostgreSQL..."
POSTGRES_FILE="/tmp/usdcop_data_insert.csv"

if docker cp "$CLEAN_FILE" "usdcop-postgres-timescale:$POSTGRES_FILE"; then
    echo "✅ Archivo copiado"
else
    echo "❌ Error copiando archivo"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Script SQL para inserción
SQL_INSERT="
-- Crear tabla temporal
CREATE TEMP TABLE temp_import (
    time TIMESTAMPTZ,
    open NUMERIC,
    high NUMERIC,
    low NUMERIC,
    close NUMERIC,
    volume NUMERIC,
    timestamp_cot TIMESTAMPTZ,
    hour_cot INTEGER,
    weekday INTEGER,
    source VARCHAR(100),
    timezone VARCHAR(50),
    batch_id VARCHAR(200)
);

-- Importar datos desde CSV
\\COPY temp_import FROM '$POSTGRES_FILE' DELIMITER ',' CSV HEADER;

-- Verificar datos importados
SELECT
    COUNT(*) as registros_importados,
    MIN(time) as fecha_inicial,
    MAX(time) as fecha_final
FROM temp_import;

-- Insertar datos evitando duplicados (usando INSERT ... ON CONFLICT para TimescaleDB)
INSERT INTO market_data (
    time, open, high, low, close, volume,
    timestamp_cot, hour_cot, weekday, source, timezone, batch_id
)
SELECT
    time, open, high, low, close, volume,
    timestamp_cot, hour_cot, weekday, source, timezone, batch_id
FROM temp_import
ON CONFLICT DO NOTHING;

-- Obtener estadísticas finales
SELECT
    COUNT(*) as total_registros,
    MIN(time) as fecha_inicial,
    MAX(time) as fecha_final,
    COUNT(DISTINCT source) as fuentes_unicas
FROM market_data;

-- Mostrar muestra de datos
SELECT
    time, open, high, low, close, source
FROM market_data
ORDER BY time
LIMIT 5;
"

# Ejecutar inserción
echo "🚀 Insertando datos a TimescaleDB..."
echo "⏳ Procesando $CLEAN_LINES registros..."

if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$SQL_INSERT"; then
    echo "✅ Datos insertados exitosamente"
else
    echo "❌ Error en inserción"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Verificación adicional con consultas de rendimiento de TimescaleDB
echo "📊 Verificación final con estadísticas de TimescaleDB..."

TIMESCALE_STATS="
-- Estadísticas de la hypertable
SELECT
    schemaname,
    tablename,
    attname as column_name,
    n_distinct,
    most_common_vals[1:3] as sample_values
FROM pg_stats
WHERE tablename = 'market_data'
AND attname IN ('time', 'source', 'close')
ORDER BY attname;

-- Información de chunks (TimescaleDB)
SELECT
    hypertable_name,
    chunk_name,
    range_start,
    range_end
FROM timescaledb_information.chunks
WHERE hypertable_name = 'market_data'
ORDER BY range_start
LIMIT 5;

-- Resumen por días
SELECT
    DATE(time) as fecha,
    COUNT(*) as registros,
    MIN(low) as minimo,
    MAX(high) as maximo,
    AVG(close)::numeric(10,2) as promedio_cierre
FROM market_data
GROUP BY DATE(time)
ORDER BY fecha
LIMIT 10;
"

echo "📈 ESTADÍSTICAS DETALLADAS:"
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$TIMESCALE_STATS"

# Limpiar archivos temporales
rm -rf "$TEMP_DIR"
docker exec usdcop-postgres-timescale rm -f "$POSTGRES_FILE"
echo "🧹 Limpieza completada"

echo ""
echo "🎉 INSERCIÓN A TIMESCALEDB COMPLETADA"
echo "====================================="
echo "📊 Dataset histórico USDCOP disponible en TimescaleDB"
echo "🗃️ Tabla: market_data (hypertable)"
echo "⚡ Optimizada para consultas de series de tiempo"
echo ""
echo "🔗 Para consultar:"
echo "   docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading"
echo ""
echo "💡 Consultas optimizadas para TimescaleDB:"
echo "   -- Datos por rango de tiempo"
echo "   SELECT * FROM market_data WHERE time >= '2020-01-01' AND time < '2020-02-01';"
echo ""
echo "   -- Agregaciones por hora"
echo "   SELECT time_bucket('1 hour', time) as hour, AVG(close) FROM market_data GROUP BY hour;"
echo ""
echo "   -- Últimos precios"
echo "   SELECT * FROM market_data ORDER BY time DESC LIMIT 10;"
echo "====================================="