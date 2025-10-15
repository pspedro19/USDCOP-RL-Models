#!/bin/bash
# Script simple para descargar y fusionar datos históricos usando solo herramientas del sistema

echo "🚀 FUSIÓN SIMPLE DE DATOS HISTÓRICOS USDCOP"
echo "============================================="

# Crear directorio temporal
TEMP_DIR="/tmp/usdcop_fusion_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEMP_DIR"
echo "📁 Directorio temporal: $TEMP_DIR"

# Lista de archivos conocidos para descargar (primeros archivos de cada mes)
declare -a FILES=(
    "usdcop_m5__01_l0_acquire_sync_incremental/market=usdcop/timeframe=m5/source=twelvedata/date=2020-01-01/run_id=usdcop_m5__01_l0_acquire_sync_incremental_2025-09-19_7933bce8/premium_data_20250920_220057.csv"
    "usdcop_m5__01_l0_acquire_sync_incremental/market=usdcop/timeframe=m5/source=twelvedata/date=2020-02-01/run_id=usdcop_m5__01_l0_acquire_sync_incremental_2025-09-19_7933bce8/premium_data_20250920_220108.csv"
    "usdcop_m5__01_l0_acquire_sync_incremental/market=usdcop/timeframe=m5/source=twelvedata/date=2020-03-01/run_id=usdcop_m5__01_l0_acquire_sync_incremental_2025-09-19_7933bce8/premium_data_20250920_220120.csv"
    "usdcop_m5__01_l0_acquire_sync_incremental/market=usdcop/timeframe=m5/source=twelvedata/date=2020-04-01/run_id=usdcop_m5__01_l0_acquire_sync_incremental_2025-09-19_7933bce8/premium_data_20250920_220133.csv"
    "usdcop_m5__01_l0_acquire_sync_incremental/market=usdcop/timeframe=m5/source=twelvedata/date=2020-05-01/run_id=usdcop_m5__01_l0_acquire_sync_incremental_2025-09-19_7933bce8/premium_data_20250920_220149.csv"
)

# Descargar archivos
echo "📥 Descargando archivos CSV desde MinIO..."
DOWNLOADED=0

for i in "${!FILES[@]}"; do
    FILE="${FILES[$i]}"
    LOCAL_FILE="$TEMP_DIR/sample_$((i+1)).csv"

    echo "📥 [$((i+1))/${#FILES[@]}] Descargando: $(basename "$FILE")"

    # Descargar usando docker exec
    if docker exec usdcop-minio cat "/data/00-raw-usdcop-marketdata/$FILE" > "$LOCAL_FILE" 2>/dev/null; then
        if [ -s "$LOCAL_FILE" ]; then
            SIZE=$(wc -c < "$LOCAL_FILE")
            echo "   ✅ Descargado: $SIZE bytes"
            DOWNLOADED=$((DOWNLOADED + 1))
        else
            echo "   ⚠️ Archivo vacío"
            rm -f "$LOCAL_FILE"
        fi
    else
        echo "   ❌ Error en descarga"
        rm -f "$LOCAL_FILE"
    fi
done

if [ $DOWNLOADED -eq 0 ]; then
    echo "❌ No se descargaron archivos válidos"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "✅ Descargados $DOWNLOADED archivos exitosamente"

# Verificar estructura de los archivos
echo "🔍 Verificando estructura de archivos..."
FIRST_FILE=$(find "$TEMP_DIR" -name "*.csv" | head -n 1)

if [ -n "$FIRST_FILE" ]; then
    echo "📋 Estructura del primer archivo:"
    head -n 3 "$FIRST_FILE"
fi

# Fusionar archivos CSV (usando herramientas básicas)
echo "🔄 Fusionando archivos CSV..."
UNIFIED_FILE="$TEMP_DIR/usdcop_unified_historical.csv"

# Escribir header (usando el primer archivo)
if [ -n "$FIRST_FILE" ]; then
    head -n 1 "$FIRST_FILE" > "$UNIFIED_FILE"
    echo "📝 Header escrito al archivo unificado"
fi

# Agregar contenido de todos los archivos (sin headers)
TOTAL_LINES=0
for CSV_FILE in "$TEMP_DIR"/*.csv; do
    if [ -f "$CSV_FILE" ]; then
        LINES_BEFORE=$(wc -l < "$UNIFIED_FILE")
        tail -n +2 "$CSV_FILE" >> "$UNIFIED_FILE"  # Omitir header
        LINES_AFTER=$(wc -l < "$UNIFIED_FILE")
        ADDED_LINES=$((LINES_AFTER - LINES_BEFORE))
        echo "📊 Agregadas $ADDED_LINES líneas desde $(basename "$CSV_FILE")"
        TOTAL_LINES=$((TOTAL_LINES + ADDED_LINES))
    fi
done

if [ ! -f "$UNIFIED_FILE" ] || [ ! -s "$UNIFIED_FILE" ]; then
    echo "❌ Error: No se pudo crear el archivo unificado"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# Mostrar estadísticas
echo "📊 ESTADÍSTICAS DEL ARCHIVO UNIFICADO:"
echo "   Total líneas: $(wc -l < "$UNIFIED_FILE")"
echo "   Tamaño: $(du -h "$UNIFIED_FILE" | cut -f1)"

# Mostrar muestra de datos
echo "📋 MUESTRA DE DATOS (primeras 5 líneas):"
head -n 5 "$UNIFIED_FILE"

echo ""
echo "📋 MUESTRA DE DATOS (últimas 5 líneas):"
tail -n 5 "$UNIFIED_FILE"

# Copiar archivo a MinIO
echo "⬆️ Subiendo archivo unificado a MinIO..."

# Crear directorio en MinIO si no existe
docker exec usdcop-minio mkdir -p /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/

# Copiar archivo unificado
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
FINAL_NAME="usdcop_unified_historical_$TIMESTAMP.csv"

if docker cp "$UNIFIED_FILE" "usdcop-minio:/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/$FINAL_NAME"; then
    echo "✅ Archivo subido: UNIFIED_COMPLETE/$FINAL_NAME"

    # Crear enlace LATEST
    docker exec usdcop-minio cp "/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/$FINAL_NAME" "/data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv"
    echo "🔗 Enlace LATEST creado: UNIFIED_COMPLETE/LATEST.csv"
else
    echo "❌ Error subiendo archivo a MinIO"
fi

# Limpiar directorio temporal
rm -rf "$TEMP_DIR"
echo "🧹 Directorio temporal limpiado"

echo ""
echo "🎉 PROCESO COMPLETADO EXITOSAMENTE"
echo "📁 Archivos disponibles en MinIO:"
echo "   - s3://00-raw-usdcop-marketdata/UNIFIED_COMPLETE/$FINAL_NAME"
echo "   - s3://00-raw-usdcop-marketdata/UNIFIED_COMPLETE/LATEST.csv"
echo ""
echo "💡 Para insertar a PostgreSQL, ejecuta:"
echo "   docker exec usdcop-postgres-timescale psql -U admin -d usdcop_trading"
echo "   \\COPY market_data(time,open,high,low,close,volume,source) FROM '/path/to/csv' DELIMITER ',' CSV HEADER;"