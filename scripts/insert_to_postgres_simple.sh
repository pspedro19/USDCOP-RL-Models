#!/bin/bash
# Script simple para insertar el dataset unificado a PostgreSQL

echo "🐘 INSERCIÓN DE DATASET UNIFICADO A POSTGRESQL"
echo "==============================================="

# Verificar que el archivo unificado existe en MinIO
echo "🔍 Verificando dataset unificado en MinIO..."
if ! docker exec usdcop-minio test -f /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv; then
    echo "❌ Error: Archivo LATEST.csv no encontrado en MinIO"
    echo "💡 Ejecuta primero: scripts/extract_and_fuse_real.sh"
    exit 1
fi

# Crear directorio temporal
TEMP_DIR="/tmp/postgres_insert_$(date +%Y%m%d_%H%M%S)"
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
    echo "❌ Error descargando archivo desde MinIO"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Limpiar archivo CSV (quitar caracteres de control al inicio)
echo "🧹 Limpiando archivo CSV..."
CLEAN_FILE="$TEMP_DIR/usdcop_clean_data.csv"

# Remover caracteres de control y mantener solo el CSV válido
sed 's/^[^a-zA-Z0-9,_-]*//g' "$UNIFIED_FILE" | grep -E '^(time|[0-9])' > "$CLEAN_FILE"

CLEAN_LINES=$(wc -l < "$CLEAN_FILE")
echo "✅ Archivo limpio creado: $CLEAN_LINES líneas"

# Mostrar muestra del archivo limpio
echo "📋 Muestra del archivo limpio:"
head -n 3 "$CLEAN_FILE"

# Verificar conexión a PostgreSQL
echo "🔌 Verificando conexión a PostgreSQL..."
if ! docker exec usdcop-postgres-timescale pg_isready -U admin > /dev/null 2>&1; then
    echo "❌ Error: PostgreSQL no está disponible"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "✅ PostgreSQL disponible"

# Crear tabla si no existe
echo "🗃️ Verificando/creando tabla market_data..."

SQL_CREATE_TABLE="
-- Crear tabla market_data si no existe
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
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

-- Crear índice único en time si no existe
CREATE UNIQUE INDEX IF NOT EXISTS idx_market_data_time_unique
ON market_data (time);

-- Crear índices adicionales
CREATE INDEX IF NOT EXISTS idx_market_data_source
ON market_data (source);

CREATE INDEX IF NOT EXISTS idx_market_data_created_at
ON market_data (created_at);

-- Mostrar información de la tabla
SELECT COUNT(*) as current_records FROM market_data;
"

# Ejecutar creación de tabla
if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$SQL_CREATE_TABLE"; then
    echo "✅ Tabla market_data verificada/creada"
else
    echo "❌ Error creando tabla market_data"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Copiar archivo limpio al contenedor de PostgreSQL
echo "📤 Copiando archivo al contenedor PostgreSQL..."
POSTGRES_FILE="/tmp/usdcop_data_to_insert.csv"

if docker cp "$CLEAN_FILE" "usdcop-postgres-timescale:$POSTGRES_FILE"; then
    echo "✅ Archivo copiado al contenedor PostgreSQL"
else
    echo "❌ Error copiando archivo al contenedor"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Preparar script SQL para inserción
SQL_INSERT="
-- Crear tabla temporal para la inserción
CREATE TEMP TABLE temp_market_data (
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
\\COPY temp_market_data FROM '$POSTGRES_FILE' DELIMITER ',' CSV HEADER;

-- Mostrar estadísticas de datos importados
SELECT
    COUNT(*) as imported_records,
    MIN(time) as earliest_date,
    MAX(time) as latest_date,
    COUNT(DISTINCT source) as unique_sources
FROM temp_market_data;

-- Insertar datos únicos (evitar duplicados basado en timestamp)
INSERT INTO market_data (
    time, open, high, low, close, volume,
    timestamp_cot, hour_cot, weekday, source, timezone, batch_id
)
SELECT DISTINCT
    time, open, high, low, close, volume,
    timestamp_cot, hour_cot, weekday, source, timezone, batch_id
FROM temp_market_data t
WHERE NOT EXISTS (
    SELECT 1 FROM market_data m
    WHERE m.time = t.time
)
ORDER BY time;

-- Mostrar estadísticas finales
SELECT
    COUNT(*) as total_records_after_insert,
    MIN(time) as earliest_date,
    MAX(time) as latest_date,
    COUNT(DISTINCT source) as unique_sources
FROM market_data;

-- Mostrar muestra de datos insertados
SELECT
    time, open, high, low, close, volume, source
FROM market_data
ORDER BY time
LIMIT 5;
"

# Ejecutar inserción
echo "🚀 Insertando datos a PostgreSQL..."
echo "⏳ Esto puede tomar unos momentos..."

if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$SQL_INSERT"; then
    echo "✅ Datos insertados exitosamente"
else
    echo "❌ Error en la inserción de datos"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Verificar inserción final
echo "🔍 Verificación final..."
FINAL_CHECK="
SELECT
    'Total registros' as estadistica,
    COUNT(*)::text as valor
FROM market_data
UNION ALL
SELECT
    'Rango de fechas',
    MIN(time)::text || ' → ' || MAX(time)::text
FROM market_data
UNION ALL
SELECT
    'Fuentes únicas',
    string_agg(DISTINCT source, ', ')
FROM market_data;
"

echo "📊 ESTADÍSTICAS FINALES:"
docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$FINAL_CHECK"

# Limpiar archivos temporales
rm -rf "$TEMP_DIR"
docker exec usdcop-postgres-timescale rm -f "$POSTGRES_FILE"
echo "🧹 Archivos temporales limpiados"

echo ""
echo "🎉 INSERCIÓN COMPLETADA EXITOSAMENTE"
echo "========================================"
echo "📊 Dataset histórico USDCOP insertado en PostgreSQL"
echo "🗃️ Tabla: market_data"
echo "🔗 Para consultar datos:"
echo "   docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading"
echo "   SELECT COUNT(*) FROM market_data;"
echo ""
echo "💡 Consultas útiles:"
echo "   -- Ver registros por fecha"
echo "   SELECT DATE(time), COUNT(*) FROM market_data GROUP BY DATE(time) ORDER BY DATE(time);"
echo ""
echo "   -- Ver rango de precios por día"
echo "   SELECT DATE(time), MIN(low), MAX(high), AVG(close) FROM market_data GROUP BY DATE(time);"
echo "========================================"