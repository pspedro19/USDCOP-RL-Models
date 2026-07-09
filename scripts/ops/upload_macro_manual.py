#!/usr/bin/env python3
"""
Script para cargar datos macro manualmente desde investing.com CSV
Usar solo si TwelveData API no está disponible

Uso:
    python scripts/upload_macro_manual.py --file wti_data.csv --symbol WTI
    python scripts/upload_macro_manual.py --file dxy_data.csv --symbol DXY

Formato CSV esperado (investing.com):
    Date,Price,Open,High,Low,Vol.,Change %
    Nov 05, 2025,75.23,75.10,75.50,74.80,123.45K,-0.05%
"""

import os
import sys
import argparse
import pandas as pd
import psycopg2
from datetime import datetime
from minio import Minio
import io

# Colores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_success(msg):
    print(f"{GREEN}✅ {msg}{RESET}")

def print_error(msg):
    print(f"{RED}❌ {msg}{RESET}")

def print_warning(msg):
    print(f"{YELLOW}⚠️  {msg}{RESET}")

def print_info(msg):
    print(f"{BLUE}ℹ️  {msg}{RESET}")

def print_header(msg):
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{msg.center(60)}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'usdcop_db'),
    'user': os.getenv('POSTGRES_USER', 'usdcop'),
    'password': os.getenv('POSTGRES_PASSWORD')
}

MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
    'secure': os.getenv('MINIO_SECURE', 'False').lower() == 'true'
}

MINIO_BUCKET = '00-raw-macro-marketdata'

VALID_SYMBOLS = ['WTI', 'DXY']

# ============================================================================
# FUNCIONES
# ============================================================================

def parse_investing_csv(file_path, symbol):
    """
    Parsear CSV de investing.com

    Formato esperado:
        Date,Price,Open,High,Low,Vol.,Change %
        Nov 05, 2025,75.23,75.10,75.50,74.80,123.45K,-0.05%

    Args:
        file_path: Path al archivo CSV
        symbol: Símbolo (WTI o DXY)

    Returns:
        DataFrame con columnas: time, symbol, open, high, low, close, volume, source
    """

    print_info(f"Leyendo CSV: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print_error(f"Error leyendo CSV: {e}")
        return None

    print_info(f"Registros encontrados: {len(df)}")

    # Verificar columnas requeridas
    required_cols = ['Date', 'Price', 'Open', 'High', 'Low']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print_error(f"Columnas faltantes: {', '.join(missing_cols)}")
        print_info("Columnas encontradas: " + ", ".join(df.columns))
        return None

    # Renombrar columnas
    df = df.rename(columns={
        'Date': 'date',
        'Price': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Vol.': 'volume'
    })

    # Convertir fecha (investing.com usa formato "Nov 05, 2025")
    try:
        df['time'] = pd.to_datetime(df['date'], format='%b %d, %Y')
    except:
        # Intentar otros formatos
        try:
            df['time'] = pd.to_datetime(df['date'])
        except Exception as e:
            print_error(f"Error parseando fechas: {e}")
            print_info("Ejemplo de fecha: " + str(df['date'].iloc[0]))
            return None

    # Convertir precios
    for col in ['open', 'high', 'low', 'close']:
        try:
            df[col] = df[col].astype(str).str.replace(',', '').astype(float)
        except Exception as e:
            print_error(f"Error convirtiendo columna {col}: {e}")
            return None

    # Limpiar volumen (123.45K → 123450, 1.5M → 1500000)
    def clean_volume(vol_str):
        if pd.isna(vol_str) or vol_str == '-' or vol_str == '':
            return 0

        vol_str = str(vol_str).replace(',', '')

        try:
            if 'K' in vol_str:
                return int(float(vol_str.replace('K', '')) * 1000)
            elif 'M' in vol_str:
                return int(float(vol_str.replace('M', '')) * 1000000)
            elif 'B' in vol_str:
                return int(float(vol_str.replace('B', '')) * 1000000000)
            else:
                return int(float(vol_str))
        except:
            return 0

    if 'volume' in df.columns:
        df['volume'] = df['volume'].apply(clean_volume)
    else:
        df['volume'] = 0

    # Añadir metadata
    df['symbol'] = symbol
    df['source'] = 'investing.com_manual'

    # Seleccionar columnas finales
    df_final = df[['time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source']]

    # Ordenar por fecha (más antigua primero)
    df_final = df_final.sort_values('time')

    # Eliminar duplicados
    df_final = df_final.drop_duplicates(subset=['time', 'symbol'])

    print_success(f"CSV parseado correctamente:")
    print(f"  - Registros: {len(df_final)}")
    print(f"  - Rango: {df_final['time'].min().date()} → {df_final['time'].max().date()}")
    print(f"  - Último precio: {df_final['close'].iloc[-1]:.2f}")

    return df_final

def upload_to_postgresql(df):
    """
    Subir datos a PostgreSQL tabla macro_ohlcv

    Args:
        df: DataFrame con datos

    Returns:
        int: Número de registros insertados/actualizados
    """

    print_info("Conectando a PostgreSQL...")

    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()
    except Exception as e:
        print_error(f"Error conectando a PostgreSQL: {e}")
        return 0

    print_success("Conectado a PostgreSQL")

    insert_count = 0
    error_count = 0

    # Upsert (INSERT ... ON CONFLICT UPDATE)
    query = """
    INSERT INTO macro_ohlcv (time, symbol, open, high, low, close, volume, source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (time, symbol) DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        source = EXCLUDED.source,
        updated_at = NOW()
    """

    print_info(f"Insertando {len(df)} registros...")

    for idx, row in df.iterrows():
        try:
            cursor.execute(query, (
                row['time'],
                row['symbol'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                row['source']
            ))
            insert_count += 1

            # Progress bar
            if (idx + 1) % 100 == 0:
                print(f"  Progreso: {idx + 1}/{len(df)} registros...", end='\r')

        except Exception as e:
            error_count += 1
            if error_count <= 5:  # Solo mostrar primeros 5 errores
                print_error(f"Error en registro {idx}: {e}")

    # Commit
    try:
        conn.commit()
        print()  # Nueva línea después del progress bar
        print_success(f"Insertados/actualizados {insert_count} registros")

        if error_count > 0:
            print_warning(f"{error_count} registros con errores (omitidos)")

    except Exception as e:
        conn.rollback()
        print_error(f"Error en commit: {e}")
        return 0

    finally:
        cursor.close()
        conn.close()

    return insert_count

def upload_to_minio(df, symbol):
    """
    Subir datos a MinIO bucket

    Args:
        df: DataFrame con datos
        symbol: Símbolo (WTI o DXY)

    Returns:
        bool: True si éxito
    """

    print_info("Conectando a MinIO...")

    try:
        client = Minio(
            MINIO_CONFIG['endpoint'],
            access_key=MINIO_CONFIG['access_key'],
            secret_key=MINIO_CONFIG['secret_key'],
            secure=MINIO_CONFIG['secure']
        )
    except Exception as e:
        print_error(f"Error conectando a MinIO: {e}")
        return False

    print_success("Conectado a MinIO")

    # Crear bucket si no existe
    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)
            print_success(f"Bucket {MINIO_BUCKET} creado")
    except Exception as e:
        print_error(f"Error verificando bucket: {e}")
        return False

    # Nombre del archivo
    date_range = f"{df['time'].min().strftime('%Y%m%d')}_{df['time'].max().strftime('%Y%m%d')}"
    filename = f"macro_{symbol}_manual_{date_range}.parquet"
    object_path = f"{symbol}/manual/{filename}"

    print_info(f"Subiendo a MinIO: {object_path}")

    try:
        # Convertir a parquet en memoria
        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False, engine='pyarrow', compression='snappy')
        parquet_buffer.seek(0)

        # Subir
        client.put_object(
            bucket_name=MINIO_BUCKET,
            object_name=object_path,
            data=parquet_buffer,
            length=parquet_buffer.getbuffer().nbytes,
            content_type='application/octet-stream'
        )

        file_size_kb = parquet_buffer.getbuffer().nbytes / 1024

        print_success(f"Archivo subido: {object_path} ({file_size_kb:.2f} KB)")

        return True

    except Exception as e:
        print_error(f"Error subiendo a MinIO: {e}")
        return False

def verify_data(symbol):
    """
    Verificar datos en PostgreSQL

    Args:
        symbol: Símbolo a verificar
    """

    print_info("Verificando datos en PostgreSQL...")

    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()

        query = """
        SELECT
            COUNT(*) as record_count,
            MIN(time) as min_time,
            MAX(time) as max_time,
            AVG(close) as avg_close,
            MIN(close) as min_close,
            MAX(close) as max_close
        FROM macro_ohlcv
        WHERE symbol = %s
        """

        cursor.execute(query, (symbol,))
        result = cursor.fetchone()

        if result and result[0] > 0:
            print_success("Datos verificados:")
            print(f"  - Registros totales: {result[0]}")
            print(f"  - Rango temporal: {result[1]} → {result[2]}")
            print(f"  - Precio promedio: {result[3]:.2f}")
            print(f"  - Precio mín/máx: {result[4]:.2f} / {result[5]:.2f}")
        else:
            print_warning(f"Sin datos para {symbol} en PostgreSQL")

        cursor.close()
        conn.close()

    except Exception as e:
        print_error(f"Error verificando datos: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Upload macro data from investing.com CSV to PostgreSQL + MinIO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/upload_macro_manual.py --file ~/Downloads/WTI_Historical_Data.csv --symbol WTI
  python scripts/upload_macro_manual.py --file ~/Downloads/DXY_Historical_Data.csv --symbol DXY

Pasos para obtener datos de investing.com:
  1. WTI: https://www.investing.com/commodities/crude-oil-historical-data
  2. DXY: https://www.investing.com/indices/usdollar-historical-data
  3. Seleccionar rango de fechas (desde 2002-01-01)
  4. Descargar CSV
  5. Ejecutar este script
        """
    )

    parser.add_argument('--file', required=True, help='Path al archivo CSV de investing.com')
    parser.add_argument('--symbol', required=True, choices=VALID_SYMBOLS, help='Símbolo: WTI o DXY')
    parser.add_argument('--skip-minio', action='store_true', help='No subir a MinIO (solo PostgreSQL)')

    args = parser.parse_args()

    print_header(f"UPLOAD MACRO DATA - {args.symbol}")

    # Verificar que archivo existe
    if not os.path.exists(args.file):
        print_error(f"Archivo no encontrado: {args.file}")
        return 1

    # Parsear CSV
    df = parse_investing_csv(args.file, args.symbol)

    if df is None or len(df) == 0:
        print_error("No se pudieron parsear datos del CSV")
        return 1

    # Upload a PostgreSQL
    print_header("SUBIENDO A POSTGRESQL")
    inserted = upload_to_postgresql(df)

    if inserted == 0:
        print_error("No se pudieron insertar datos en PostgreSQL")
        return 1

    # Upload a MinIO (opcional)
    if not args.skip_minio:
        print_header("SUBIENDO A MINIO")
        minio_success = upload_to_minio(df, args.symbol)

        if not minio_success:
            print_warning("Falló la subida a MinIO (datos están en PostgreSQL)")

    # Verificar datos
    print_header("VERIFICACIÓN")
    verify_data(args.symbol)

    # Resumen final
    print_header("✅ PROCESO COMPLETADO")
    print_success(f"{args.symbol}: {inserted} registros cargados")

    return 0

if __name__ == '__main__':
    sys.exit(main())
