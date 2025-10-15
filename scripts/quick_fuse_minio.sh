#!/bin/bash
# Script rápido para fusionar datos históricos usando mc y herramientas del sistema

echo "🚀 FUSIÓN RÁPIDA DE DATOS HISTÓRICOS USDCOP"
echo "=============================================="

# Crear directorio temporal
TEMP_DIR="/tmp/usdcop_fusion"
mkdir -p "$TEMP_DIR"

echo "📁 Directorio temporal: $TEMP_DIR"

# Descargar todos los archivos CSV
echo "⬇️ Descargando archivos CSV desde MinIO..."
docker exec usdcop-minio mc cp --recursive /data/00-raw-usdcop-marketdata/ "$TEMP_DIR/" > /dev/null 2>&1

# Encontrar archivos CSV válidos (no metadatos)
echo "🔍 Buscando archivos CSV válidos..."
CSV_FILES=$(docker exec usdcop-minio find "$TEMP_DIR" -name "*.csv" -type f | grep -v xl.meta | head -10)

if [ -z "$CSV_FILES" ]; then
    echo "❌ No se encontraron archivos CSV válidos"
    exit 1
fi

echo "📊 Archivos CSV encontrados:"
echo "$CSV_FILES"

# Crear un archivo Python temporal para fusión
FUSION_SCRIPT="$TEMP_DIR/quick_fusion.py"

cat > "$FUSION_SCRIPT" << 'EOF'
import pandas as pd
import glob
import sys
import os
from datetime import datetime

print("🔄 Iniciando fusión de archivos CSV...")

# Buscar archivos CSV
csv_files = []
for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        if file.endswith('.csv') and 'xl.meta' not in file:
            csv_files.append(os.path.join(root, file))

print(f"📂 Encontrados {len(csv_files)} archivos CSV")

all_dfs = []
total_records = 0

for i, csv_file in enumerate(csv_files[:10], 1):  # Limitar a 10 archivos para prueba
    try:
        print(f"📖 [{i}/{min(len(csv_files), 10)}] Leyendo: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)

        # Renombrar columnas si es necesario
        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'time'})

        # Verificar columnas requeridas
        if 'time' in df.columns and 'close' in df.columns:
            print(f"   ✅ {len(df)} registros válidos")
            all_dfs.append(df)
            total_records += len(df)
        else:
            print(f"   ⚠️ Columnas faltantes: {list(df.columns)}")

    except Exception as e:
        print(f"   ❌ Error: {e}")

if not all_dfs:
    print("❌ No se encontraron datos válidos")
    sys.exit(1)

print(f"🔄 Fusionando {len(all_dfs)} DataFrames con {total_records:,} registros...")

# Fusionar todos los DataFrames
unified_df = pd.concat(all_dfs, ignore_index=True)

# Eliminar duplicados si existe columna 'time'
if 'time' in unified_df.columns:
    unified_df['time'] = pd.to_datetime(unified_df['time'])
    initial_count = len(unified_df)
    unified_df = unified_df.drop_duplicates(subset=['time']).sort_values('time')
    final_count = len(unified_df)
    print(f"✂️ Eliminados {initial_count - final_count:,} duplicados")

# Guardar resultado
output_file = f"{sys.argv[1]}/usdcop_unified_data.csv"
unified_df.to_csv(output_file, index=False)

print(f"💾 Dataset unificado guardado: {output_file}")
print(f"📊 Total registros finales: {len(unified_df):,}")

if 'time' in unified_df.columns:
    print(f"📅 Rango: {unified_df['time'].min()} → {unified_df['time'].max()}")

print("✅ Fusión completada exitosamente")
EOF

# Ejecutar script de fusión
echo "🐍 Ejecutando script de fusión..."
docker exec usdcop-minio python3 "$FUSION_SCRIPT" "$TEMP_DIR"

# Verificar resultado
UNIFIED_FILE="$TEMP_DIR/usdcop_unified_data.csv"
if docker exec usdcop-minio test -f "$UNIFIED_FILE"; then
    echo "✅ Archivo unificado creado exitosamente"

    # Mostrar estadísticas del archivo
    echo "📊 Estadísticas del archivo unificado:"
    docker exec usdcop-minio wc -l "$UNIFIED_FILE"
    docker exec usdcop-minio ls -lh "$UNIFIED_FILE"

    # Copiar archivo unificado a un location accesible
    docker exec usdcop-minio cp "$UNIFIED_FILE" /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE_DATA.csv
    echo "📁 Archivo copiado a: /data/00-raw-usdcop-marketdata/UNIFIED_COMPLETE_DATA.csv"

else
    echo "❌ Error: No se pudo crear el archivo unificado"
    exit 1
fi

echo "🎉 PROCESO COMPLETADO"