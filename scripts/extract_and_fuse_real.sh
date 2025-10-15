#!/bin/bash
# Script para extraer y fusionar los datos reales de MinIO

echo "🚀 EXTRACCIÓN Y FUSIÓN DE DATOS HISTÓRICOS REALES USDCOP"
echo "========================================================="

# Crear directorio temporal
TEMP_DIR="/tmp/usdcop_real_fusion_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"
echo "📁 Directorio temporal: $TEMP_DIR"

# Obtener lista de archivos CSV disponibles (limit to avoid overwhelming)
echo "🔍 Obteniendo lista de archivos CSV disponibles..."
AVAILABLE_FILES=$(docker exec usdcop-minio mc ls --recursive /data/00-raw-usdcop-marketdata/ | grep -E "premium_data.*\.csv.*part\.1" | head -20)

if [ -z "$AVAILABLE_FILES" ]; then
    echo "❌ No se encontraron archivos CSV válidos"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "📊 Archivos encontrados:"
echo "$AVAILABLE_FILES" | nl

# Extraer rutas de archivos
echo "📥 Descargando archivos CSV..."
COUNTER=0
DOWNLOADED=0

while IFS= read -r line; do
    if [[ $line =~ part\.1$ ]]; then
        # Extraer la ruta del archivo
        FILE_PATH=$(echo "$line" | awk '{print $NF}')
        COUNTER=$((COUNTER + 1))

        # Limitar a los primeros 10 archivos para no sobrecargar
        if [ $COUNTER -gt 10 ]; then
            break
        fi

        LOCAL_FILE="$TEMP_DIR/data_$COUNTER.csv"

        echo "📥 [$COUNTER] Descargando: $(basename "$(dirname "$FILE_PATH")")"

        if docker exec usdcop-minio cat "/data/00-raw-usdcop-marketdata/$FILE_PATH" > "$LOCAL_FILE" 2>/dev/null; then
            if [ -s "$LOCAL_FILE" ]; then
                LINES=$(wc -l < "$LOCAL_FILE")
                SIZE=$(wc -c < "$LOCAL_FILE")
                echo "   ✅ Descargado: $LINES líneas, $SIZE bytes"
                DOWNLOADED=$((DOWNLOADED + 1))
            else
                echo "   ⚠️ Archivo vacío"
                rm -f "$LOCAL_FILE"
            fi
        else
            echo "   ❌ Error en descarga"
            rm -f "$LOCAL_FILE"
        fi
    fi
done <<< "$AVAILABLE_FILES"

if [ $DOWNLOADED -eq 0 ]; then
    echo "❌ No se descargaron archivos válidos"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "✅ Descargados $DOWNLOADED archivos exitosamente"

# Verificar estructura de datos
echo "🔍 Verificando estructura de datos..."
FIRST_FILE=$(find "$TEMP_DIR" -name "*.csv" | head -n 1)

if [ -n "$FIRST_FILE" ]; then
    echo "📋 Header del primer archivo:"
    head -n 1 "$FIRST_FILE"
    echo ""
    echo "📋 Muestra de datos:"
    head -n 3 "$FIRST_FILE" | tail -n 2
fi

# Crear archivo unificado
echo "🔄 Fusionando archivos CSV..."
UNIFIED_FILE="$TEMP_DIR/usdcop_unified_historical.csv"

# Escribir header usando el primer archivo
if [ -n "$FIRST_FILE" ]; then
    head -n 1 "$FIRST_FILE" > "$UNIFIED_FILE"
    echo "📝 Header escrito: $(head -n 1 "$FIRST_FILE")"
fi

# Agregar datos de todos los archivos (sin duplicar headers)
TOTAL_LINES=0
for CSV_FILE in "$TEMP_DIR"/data_*.csv; do
    if [ -f "$CSV_FILE" ]; then
        LINES_BEFORE=$(wc -l < "$UNIFIED_FILE")

        # Agregar todas las líneas excepto el header
        tail -n +2 "$CSV_FILE" >> "$UNIFIED_FILE"

        LINES_AFTER=$(wc -l < "$UNIFIED_FILE")
        ADDED_LINES=$((LINES_AFTER - LINES_BEFORE))
        echo "📊 $(basename "$CSV_FILE"): +$ADDED_LINES líneas"
        TOTAL_LINES=$((TOTAL_LINES + ADDED_LINES))
    fi
done

# Verificar archivo unificado
if [ ! -f "$UNIFIED_FILE" ] || [ ! -s "$UNIFIED_FILE" ]; then
    echo "❌ Error: No se pudo crear el archivo unificado"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo ""
echo "📊 ESTADÍSTICAS DEL DATASET UNIFICADO:"
TOTAL_LINES_WITH_HEADER=$(wc -l < "$UNIFIED_FILE")
DATA_LINES=$((TOTAL_LINES_WITH_HEADER - 1))
FILE_SIZE=$(du -h "$UNIFIED_FILE" | cut -f1)

echo "   📈 Total líneas (con header): $TOTAL_LINES_WITH_HEADER"
echo "   📊 Registros de datos: $DATA_LINES"
echo "   💾 Tamaño archivo: $FILE_SIZE"

# Mostrar muestra de datos
echo ""
echo "📋 MUESTRA DEL DATASET UNIFICADO:"
echo "Header:"
head -n 1 "$UNIFIED_FILE"
echo ""
echo "Primeros 3 registros:"
head -n 4 "$UNIFIED_FILE" | tail -n 3
echo ""
echo "Últimos 3 registros:"
tail -n 3 "$UNIFIED_FILE"

# Verificar rango de fechas (aproximado usando herramientas básicas)
echo ""
echo "📅 ANÁLISIS DE FECHAS:"
# Extraer timestamps del segundo campo (asumiendo que time está en columna 1)
echo "Primera fecha en archivo:"
head -n 2 "$UNIFIED_FILE" | tail -n 1 | cut -d',' -f1

echo "Última fecha en archivo:"
tail -n 1 "$UNIFIED_FILE" | cut -d',' -f1

# Subir archivo a MinIO
echo ""
echo "⬆️ Subiendo dataset unificado a MinIO..."

# Crear directorio UNIFIED_COMPLETE en MinIO si no existe
docker exec usdcop-minio mkdir -p /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/

# Generar nombre con timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_NAME="usdcop_unified_historical_$TIMESTAMP.csv"

# Copiar archivo unificado a MinIO
if docker cp "$UNIFIED_FILE" "usdcop-minio:/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/$FINAL_NAME"; then
    echo "✅ Archivo subido: UNIFIED_COMPLETE/$FINAL_NAME"

    # Crear enlace LATEST
    docker exec usdcop-minio cp "/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/$FINAL_NAME" "/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv"
    echo "🔗 Enlace LATEST creado: UNIFIED_COMPLETE/LATEST.csv"

    # Crear archivo de metadata simple
    METADATA_FILE="$TEMP_DIR/metadata.json"
    cat > "$METADATA_FILE" << EOF
{
    "dataset_name": "USDCOP_M5_Historical_Unified",
    "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "total_records": $DATA_LINES,
    "total_files_processed": $DOWNLOADED,
    "timeframe": "5min",
    "symbol": "USDCOP",
    "source": "twelvedata",
    "files": {
        "latest_csv": "UNIFIED_COMPLETE/LATEST.csv",
        "timestamped_csv": "UNIFIED_COMPLETE/$FINAL_NAME"
    }
}
EOF

    # Subir metadata
    docker cp "$METADATA_FILE" "usdcop-minio:/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/metadata.json"
    echo "📝 Metadata subida: UNIFIED_COMPLETE/metadata.json"

else
    echo "❌ Error subiendo archivo a MinIO"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Limpiar directorio temporal
rm -rf "$TEMP_DIR"
echo "🧹 Directorio temporal limpiado"

echo ""
echo "🎉 PROCESO COMPLETADO EXITOSAMENTE"
echo "================================================"
echo "📊 Dataset unificado creado con $DATA_LINES registros"
echo "📁 Archivos disponibles en MinIO:"
echo "   • s3://00-raw-usdcop-marketdata/UNIFIED_COMPLETE/$FINAL_NAME"
echo "   • s3://00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv"
echo "   • s3://00-raw-usdcop-marketdata/UNIFIED_COMPLETE/metadata.json"
echo ""
echo "🔗 Para acceder al dataset unificado:"
echo "   docker exec usdcop-minio cat /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv"
echo ""
echo "🐘 Para insertar a PostgreSQL (próximo paso):"
echo "   Usar script: scripts/insert_to_postgres_simple.sh"
echo "================================================"