"""
DAG: usdcop_m5__01b_l0_macro_acquire
Propósito: Adquirir datos macro (WTI, DXY) desde TwelveData API
Pipeline: L0 (Data Acquisition)
Versión: 1.0
Fecha: 2025-11-05

Este DAG descarga datos macro económicos (WTI Crude Oil, US Dollar Index) desde
TwelveData API y los almacena en:
  1. PostgreSQL tabla: macro_ohlcv
  2. MinIO bucket: 00-raw-macro-marketdata

Los datos se descargan en intervalos de 1 hora y se usarán en L3 para features macro.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os
import sys
import logging
import pandas as pd
import requests
import psycopg2
from psycopg2.extras import execute_values
from minio import Minio
from minio.error import S3Error
import io

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

DAG_ID = 'usdcop_m5__01b_l0_macro_acquire'

# Símbolos macro a descargar
MACRO_SYMBOLS = {
    'CL': {
        'name': 'WTI Crude Oil',
        'twelvedata_symbol': 'CL',  # WTI Crude Oil futures
        'interval': '1h',
        'timezone': 'America/New_York'
    },
    'DXY': {
        'name': 'US Dollar Index',
        'twelvedata_symbol': 'DXY',  # US Dollar Index
        'interval': '1h',
        'timezone': 'America/New_York'
    }
}

# TwelveData API keys (rotar en caso de rate limit)
TWELVEDATA_API_KEYS = [
    os.getenv('TWELVEDATA_API_KEY_G1'),
    os.getenv('TWELVEDATA_API_KEY_G2'),
    os.getenv('TWELVEDATA_API_KEY_G3')
]

# PostgreSQL
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'usdcop_db'),
    'user': os.getenv('POSTGRES_USER', 'usdcop'),
    'password': os.getenv('POSTGRES_PASSWORD')
}

# MinIO
MINIO_CONFIG = {
    'endpoint': os.getenv('MINIO_ENDPOINT', 'localhost:9000'),
    'access_key': os.getenv('MINIO_ACCESS_KEY', 'minioadmin'),
    'secret_key': os.getenv('MINIO_SECRET_KEY', 'minioadmin'),
    'secure': os.getenv('MINIO_SECURE', 'False').lower() == 'true'
}

MINIO_BUCKET = '00-raw-macro-marketdata'

# ============================================================================
# DAG DEFAULT ARGS
# ============================================================================

default_args = {
    'owner': 'pipeline_engineer',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2002, 1, 1),  # Datos históricos desde 2002
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_postgres_connection():
    """Crear conexión a PostgreSQL"""
    return psycopg2.connect(**POSTGRES_CONFIG)

def get_minio_client():
    """Crear cliente MinIO"""
    return Minio(
        MINIO_CONFIG['endpoint'],
        access_key=MINIO_CONFIG['access_key'],
        secret_key=MINIO_CONFIG['secret_key'],
        secure=MINIO_CONFIG['secure']
    )

def ensure_minio_bucket(bucket_name):
    """Crear bucket MinIO si no existe"""
    try:
        client = get_minio_client()
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            logging.info(f"✅ Bucket {bucket_name} creado")
        else:
            logging.info(f"ℹ️  Bucket {bucket_name} ya existe")
    except S3Error as e:
        logging.error(f"❌ Error creando bucket: {e}")
        raise

# ============================================================================
# TASK 1: Fetch Macro Data from TwelveData
# ============================================================================

def fetch_macro_data_twelvedata(**context):
    """
    Descargar datos macro de TwelveData API

    Returns:
        dict: Datos descargados por símbolo
    """
    execution_date = context['ds']  # YYYY-MM-DD
    execution_dt = datetime.strptime(execution_date, '%Y-%m-%d')

    logging.info(f"📅 Execution date: {execution_date}")

    # Definir ventana de descarga (día completo en UTC)
    start_datetime = execution_dt.replace(hour=0, minute=0, second=0)
    end_datetime = execution_dt.replace(hour=23, minute=59, second=59)

    all_data = {}
    download_summary = []

    for symbol_key, symbol_config in MACRO_SYMBOLS.items():
        symbol_data = []
        symbol_twelvedata = symbol_config['twelvedata_symbol']
        symbol_name = symbol_config['name']

        logging.info(f"\n{'='*60}")
        logging.info(f"📊 Descargando: {symbol_name} ({symbol_twelvedata})")
        logging.info(f"{'='*60}")

        # Intentar con cada API key (rotar si falla)
        for api_key_idx, api_key in enumerate(TWELVEDATA_API_KEYS):
            if api_key is None:
                continue

            try:
                logging.info(f"🔑 Intentando con API key #{api_key_idx + 1}: {api_key[:8]}...")

                url = "https://api.twelvedata.com/time_series"
                params = {
                    'symbol': symbol_twelvedata,
                    'interval': symbol_config['interval'],
                    'apikey': api_key,
                    'start_date': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'timezone': symbol_config['timezone'],
                    'outputsize': 5000,
                    'format': 'JSON',
                    'type': 'Commodity' if symbol_key == 'CL' else 'Index'
                }

                response = requests.get(url, params=params, timeout=30)

                if response.status_code != 200:
                    logging.warning(f"⚠️  HTTP {response.status_code}, probando siguiente key...")
                    continue

                data = response.json()

                # Verificar errores de API
                if 'status' in data and data['status'] == 'error':
                    logging.warning(f"⚠️  API Error: {data.get('message')}, probando siguiente key...")
                    continue

                # Verificar que tiene datos
                if 'values' not in data or len(data['values']) == 0:
                    logging.warning(f"⚠️  Sin datos para {execution_date}, probando siguiente key...")
                    continue

                # Procesar datos
                df = pd.DataFrame(data['values'])

                # Renombrar columnas
                df = df.rename(columns={'datetime': 'time'})

                # Convertir tipos
                df['time'] = pd.to_datetime(df['time'])
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)

                # Volume puede ser string o ausente
                if 'volume' in df.columns:
                    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0).astype(int)
                else:
                    df['volume'] = 0

                # Añadir metadata
                df['symbol'] = symbol_key
                df['source'] = 'twelvedata'

                # Filtrar solo fechas del día solicitado
                df = df[df['time'].dt.date == execution_dt.date()]

                if len(df) == 0:
                    logging.warning(f"⚠️  Sin datos para {execution_date} después de filtrar")
                    continue

                # Ordenar por tiempo
                df = df.sort_values('time')

                logging.info(f"✅ Descargados {len(df)} registros para {symbol_name}")
                logging.info(f"   Rango: {df['time'].min()} → {df['time'].max()}")
                logging.info(f"   Último precio: {df['close'].iloc[-1]:.2f}")

                symbol_data.append(df)

                download_summary.append({
                    'symbol': symbol_key,
                    'name': symbol_name,
                    'records': len(df),
                    'success': True,
                    'api_key_used': api_key_idx + 1
                })

                break  # Éxito, no intentar más keys

            except Exception as e:
                logging.warning(f"⚠️  Error con API key #{api_key_idx + 1}: {e}")
                continue

        # Verificar si se obtuvo data para este símbolo
        if not symbol_data:
            logging.error(f"❌ No se pudo obtener datos para {symbol_name}")
            download_summary.append({
                'symbol': symbol_key,
                'name': symbol_name,
                'records': 0,
                'success': False,
                'api_key_used': None
            })
        else:
            # Combinar todos los DataFrames del símbolo
            df_combined = pd.concat(symbol_data, ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['time', 'symbol'])
            df_combined = df_combined.sort_values('time')

            all_data[symbol_key] = df_combined

    # Resumen de descarga
    logging.info(f"\n{'='*60}")
    logging.info("📋 RESUMEN DE DESCARGA")
    logging.info(f"{'='*60}")

    for summary in download_summary:
        status = "✅" if summary['success'] else "❌"
        logging.info(f"{status} {summary['name']}: {summary['records']} registros")

    # Guardar en XCom para siguiente task
    context['task_instance'].xcom_push(key='macro_data', value=all_data)
    context['task_instance'].xcom_push(key='download_summary', value=download_summary)

    # Verificar que al menos 1 símbolo tuvo éxito
    success_count = sum(1 for s in download_summary if s['success'])

    if success_count == 0:
        raise ValueError("❌ No se pudo descargar datos de ningún símbolo")

    logging.info(f"\n✅ Descarga completada: {success_count}/{len(MACRO_SYMBOLS)} símbolos exitosos")

    return success_count

# ============================================================================
# TASK 2: Insert to PostgreSQL
# ============================================================================

def insert_to_postgresql(**context):
    """
    Insertar datos en PostgreSQL tabla macro_ohlcv

    Returns:
        dict: Estadísticas de inserción
    """
    # Obtener datos de XCom
    all_data = context['task_instance'].xcom_pull(task_ids='fetch_macro_data', key='macro_data')

    if not all_data:
        logging.warning("⚠️  Sin datos para insertar")
        return {'total_inserts': 0, 'total_updates': 0}

    conn = get_postgres_connection()
    cursor = conn.cursor()

    insert_stats = {}
    total_inserts = 0
    total_updates = 0

    for symbol_key, df in all_data.items():
        logging.info(f"\n📝 Insertando {symbol_key}: {len(df)} registros...")

        inserts = 0
        updates = 0

        # Upsert (INSERT ... ON CONFLICT UPDATE)
        query = """
        INSERT INTO macro_ohlcv (time, symbol, open, high, low, close, volume, source)
        VALUES %s
        ON CONFLICT (time, symbol) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            source = EXCLUDED.source,
            updated_at = NOW()
        """

        # Preparar valores
        values = [
            (
                row['time'],
                row['symbol'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                row['source']
            )
            for _, row in df.iterrows()
        ]

        # Ejecutar batch insert
        execute_values(cursor, query, values)

        # Contar inserts vs updates (aproximado por cambios)
        inserts = cursor.rowcount
        total_inserts += inserts

        logging.info(f"✅ {symbol_key}: {inserts} registros upserted")

        insert_stats[symbol_key] = {
            'records': len(df),
            'upserts': inserts
        }

    # Commit
    conn.commit()
    cursor.close()
    conn.close()

    # Resumen
    logging.info(f"\n{'='*60}")
    logging.info("📊 RESUMEN DE INSERCIÓN")
    logging.info(f"{'='*60}")
    for symbol_key, stats in insert_stats.items():
        logging.info(f"✅ {symbol_key}: {stats['upserts']} upserts")

    logging.info(f"\n✅ Total upserts: {total_inserts}")

    # Guardar en XCom
    context['task_instance'].xcom_push(key='insert_stats', value=insert_stats)

    return {'total_upserts': total_inserts, 'symbols': insert_stats}

# ============================================================================
# TASK 3: Export to MinIO
# ============================================================================

def export_to_minio(**context):
    """
    Exportar datos a MinIO bucket

    Returns:
        dict: Archivos exportados
    """
    execution_date = context['ds']  # YYYY-MM-DD

    # Obtener datos de XCom
    all_data = context['task_instance'].xcom_pull(task_ids='fetch_macro_data', key='macro_data')

    if not all_data:
        logging.warning("⚠️  Sin datos para exportar")
        return {'files_exported': 0}

    # Asegurar que bucket existe
    ensure_minio_bucket(MINIO_BUCKET)

    client = get_minio_client()

    export_summary = []

    for symbol_key, df in all_data.items():
        # Nombre del archivo
        filename = f"macro_{symbol_key}_{execution_date}.parquet"
        object_path = f"{symbol_key}/{filename}"

        logging.info(f"\n📤 Exportando {symbol_key} a MinIO...")

        try:
            # Convertir a parquet en memoria
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False, engine='pyarrow', compression='snappy')
            parquet_buffer.seek(0)

            # Subir a MinIO
            client.put_object(
                bucket_name=MINIO_BUCKET,
                object_name=object_path,
                data=parquet_buffer,
                length=parquet_buffer.getbuffer().nbytes,
                content_type='application/octet-stream'
            )

            file_size_kb = parquet_buffer.getbuffer().nbytes / 1024

            logging.info(f"✅ Exportado: {object_path} ({file_size_kb:.2f} KB)")

            export_summary.append({
                'symbol': symbol_key,
                'object_path': object_path,
                'size_kb': file_size_kb,
                'records': len(df)
            })

        except Exception as e:
            logging.error(f"❌ Error exportando {symbol_key}: {e}")
            continue

    # Resumen
    logging.info(f"\n{'='*60}")
    logging.info("📦 RESUMEN DE EXPORTACIÓN")
    logging.info(f"{'='*60}")
    logging.info(f"Bucket: {MINIO_BUCKET}")
    for summary in export_summary:
        logging.info(f"✅ {summary['symbol']}: {summary['object_path']} ({summary['size_kb']:.2f} KB)")

    logging.info(f"\n✅ Total archivos exportados: {len(export_summary)}")

    # Guardar en XCom
    context['task_instance'].xcom_push(key='export_summary', value=export_summary)

    return {'files_exported': len(export_summary), 'details': export_summary}

# ============================================================================
# TASK 4: Validate Data Quality
# ============================================================================

def validate_data_quality(**context):
    """
    Validar calidad de datos insertados

    Returns:
        dict: Reporte de validación
    """
    execution_date = context['ds']

    conn = get_postgres_connection()
    cursor = conn.cursor()

    validation_report = {}

    for symbol_key in MACRO_SYMBOLS.keys():
        logging.info(f"\n🔍 Validando {symbol_key}...")

        # Query para obtener estadísticas del día
        query = """
        SELECT
            COUNT(*) as record_count,
            MIN(time) as min_time,
            MAX(time) as max_time,
            AVG(close) as avg_close,
            STDDEV(close) as stddev_close,
            MIN(close) as min_close,
            MAX(close) as max_close,
            SUM(CASE WHEN open = close THEN 1 ELSE 0 END) as flat_bars_count
        FROM macro_ohlcv
        WHERE symbol = %s
          AND DATE(time) = %s
        """

        cursor.execute(query, (symbol_key, execution_date))
        result = cursor.fetchone()

        if result and result[0] > 0:
            validation_report[symbol_key] = {
                'records': result[0],
                'min_time': str(result[1]),
                'max_time': str(result[2]),
                'avg_close': float(result[3]) if result[3] else None,
                'stddev_close': float(result[4]) if result[4] else None,
                'min_close': float(result[5]) if result[5] else None,
                'max_close': float(result[6]) if result[6] else None,
                'flat_bars': result[7],
                'status': 'OK'
            }

            logging.info(f"✅ {symbol_key}: {result[0]} registros válidos")
            logging.info(f"   Rango de precios: {result[5]:.2f} - {result[6]:.2f}")
            logging.info(f"   Barras planas: {result[7]} ({result[7]/result[0]*100:.1f}%)")

        else:
            validation_report[symbol_key] = {
                'records': 0,
                'status': 'NO_DATA'
            }
            logging.warning(f"⚠️  {symbol_key}: Sin datos para {execution_date}")

    cursor.close()
    conn.close()

    # Resumen final
    logging.info(f"\n{'='*60}")
    logging.info("✅ VALIDACIÓN COMPLETADA")
    logging.info(f"{'='*60}")

    success_count = sum(1 for v in validation_report.values() if v['status'] == 'OK')
    logging.info(f"Símbolos validados: {success_count}/{len(MACRO_SYMBOLS)}")

    # Guardar en XCom
    context['task_instance'].xcom_push(key='validation_report', value=validation_report)

    return validation_report

# ============================================================================
# DAG DEFINITION
# ============================================================================

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Adquirir datos macro (WTI, DXY) desde TwelveData API',
    schedule_interval='@daily',  # Ejecutar diariamente
    catchup=True,  # Importante: rellenar datos históricos
    max_active_runs=1,  # Solo 1 ejecución a la vez
    tags=['L0', 'macro', 'acquisition', 'twelvedata'],
) as dag:

    # Task 1: Fetch data
    task_fetch = PythonOperator(
        task_id='fetch_macro_data',
        python_callable=fetch_macro_data_twelvedata,
        provide_context=True,
        execution_timeout=timedelta(minutes=10),
    )

    # Task 2: Insert to PostgreSQL
    task_insert = PythonOperator(
        task_id='insert_to_postgresql',
        python_callable=insert_to_postgresql,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
    )

    # Task 3: Export to MinIO
    task_export = PythonOperator(
        task_id='export_to_minio',
        python_callable=export_to_minio,
        provide_context=True,
        execution_timeout=timedelta(minutes=5),
    )

    # Task 4: Validate
    task_validate = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality,
        provide_context=True,
        execution_timeout=timedelta(minutes=2),
    )

    # Dependencias
    task_fetch >> task_insert >> task_export >> task_validate

# ============================================================================
# FIN DEL DAG
# ============================================================================
