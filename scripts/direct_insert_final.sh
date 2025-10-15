#!/bin/bash
# Script final directo para insertar datos a la tabla market_data existente

echo "🐘 INSERCIÓN DIRECTA A MARKET_DATA (TIMESCALEDB)"
echo "================================================"

# Verificar archivo unificado
if ! docker exec usdcop-minio test -f /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv; then
    echo "❌ Archivo LATEST.csv no encontrado"
    exit 1
fi

# Crear directorio temporal
TEMP_DIR="/tmp/direct_insert_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"

# Descargar y limpiar archivo
echo "📥 Descargando y preparando datos..."
UNIFIED_FILE="$TEMP_DIR/data.csv"
CLEAN_FILE="$TEMP_DIR/clean_data.csv"

docker exec usdcop-minio cat /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv > "$UNIFIED_FILE"
sed 's/^[^a-zA-Z0-9,_-]*//g' "$UNIFIED_FILE" | grep -E '^(time|[0-9])' > "$CLEAN_FILE"

LINES=$(wc -l < "$CLEAN_FILE")
echo "✅ Datos preparados: $LINES líneas"

# Mostrar muestra
echo "📋 Muestra de datos:"
head -n 3 "$CLEAN_FILE"

# Verificar PostgreSQL
if ! docker exec usdcop-postgres-timescale pg_isready -U admin > /dev/null 2>&1; then
    echo "❌ PostgreSQL no disponible"
    exit 1
fi

# Verificar tabla existente
echo "🔍 Verificando tabla market_data..."
TABLE_INFO=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM market_data;" 2>/dev/null || echo "0")
echo "📊 Registros actuales en market_data: $TABLE_INFO"

# Copiar archivo limpio al contenedor
POSTGRES_FILE="/tmp/insert_data.csv"
docker cp "$CLEAN_FILE" "usdcop-postgres-timescale:$POSTGRES_FILE"

# Inserción directa con COPY
echo "🚀 Insertando datos directamente..."

DIRECT_SQL="
-- Verificar estructura de la tabla
\\d market_data

-- Crear tabla temporal para importación
CREATE TEMP TABLE temp_data_import (
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

-- Importar datos
\\COPY temp_data_import FROM '$POSTGRES_FILE' DELIMITER ',' CSV HEADER;

-- Mostrar estadísticas de importación
SELECT COUNT(*) as registros_importados FROM temp_data_import;

-- Insertar en market_data principal
INSERT INTO market_data (
    time, open, high, low, close, volume,
    timestamp_cot, hour_cot, weekday, source, timezone, batch_id, created_at
)
SELECT
    time, open, high, low, close, volume,
    timestamp_cot, hour_cot, weekday, source, timezone, batch_id, NOW()
FROM temp_data_import
WHERE NOT EXISTS (
    SELECT 1 FROM market_data m WHERE m.time = temp_data_import.time
);

-- Estadísticas finales
SELECT
    COUNT(*) as total_registros,
    MIN(time) as fecha_inicial,
    MAX(time) as fecha_final
FROM market_data;

-- Muestra de datos insertados
SELECT time, open, high, low, close, source
FROM market_data
ORDER BY time
LIMIT 5;
"

if docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "$DIRECT_SQL"; then
    echo "✅ Inserción completada exitosamente"

    # Verificación final
    echo "📊 Verificación final:"
    FINAL_COUNT=$(docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -t -c "SELECT COUNT(*) FROM market_data;")
    echo "📈 Total registros en market_data: $FINAL_COUNT"

    # Consulta de muestra
    echo "📋 Muestra de registros por fecha:"
    docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading -c "
        SELECT DATE(time) as fecha, COUNT(*) as registros
        FROM market_data
        GROUP BY DATE(time)
        ORDER BY fecha
        LIMIT 10;
    "

else
    echo "❌ Error en la inserción"
fi

# Limpiar
rm -rf "$TEMP_DIR"
docker exec usdcop-postgres-timescale rm -f "$POSTGRES_FILE" 2>/dev/null || true

echo ""
echo "🎉 PROCESO FINALIZADO"
echo "===================="
echo "📊 Datos históricos USDCOP disponibles en PostgreSQL"
echo "🗃️ Tabla: market_data"
echo "🔗 Conexión: docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading"
echo "===================="