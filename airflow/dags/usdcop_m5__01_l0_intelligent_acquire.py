"""
DAG: usdcop_m5__01_l0_intelligent_acquire
==========================================
Layer: L0 - INTELLIGENT ACQUIRE (AUTO-DETECT GAPS)

🚀 NUEVA FUNCIONALIDAD INTELIGENTE:
✅ Detecta automáticamente datos faltantes desde última fecha hasta HOY
✅ Primera ejecución: Descarga TODO el histórico desde 2020
✅ Ejecuciones posteriores: Solo datos faltantes (incremental)
✅ Inserción automática a PostgreSQL (usdcop_m5_ohlcv) + CSV unificado
✅ Ejecuta cada 5 minutos en horarios de trading (8AM-2PM COT)
✅ Usa 20 API keys (4 grupos) para máxima capacidad

TABLA UNIFICADA:
- Escribe en: usdcop_m5_ohlcv (misma tabla que Servicio RT V2)
- Symbol: 'USD/COP' (formato estándar)
- ON CONFLICT: GREATEST/LEAST para high/low, suma volume

HORARIOS DE TRADING (ALIGNED WITH L1 PREMIUM WINDOW):
- Lunes a Viernes: 8:00 AM - 12:55 PM COT (UTC-5)
- Frecuencia: Cada 5 minutos
- Barras esperadas: 60 por día (premium trading window)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import numpy as np
import json
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Optional, Tuple
import logging
import uuid
import io
import hashlib
import requests
import time
import pytz
from dateutil.relativedelta import relativedelta
import psycopg2
from psycopg2.extras import execute_values
import os
import sys

# Add utils directory to path
sys.path.insert(0, os.path.dirname(__file__))
from utils.dwh_helper import DWHHelper, get_dwh_connection

# Timezone definitions
COT_TIMEZONE = pytz.timezone('America/Bogota')
UTC_TIMEZONE = pytz.UTC

# DAG Constants
DAG_ID = 'usdcop_m5__01_l0_intelligent_acquire'
BUCKET_OUTPUT = '00-raw-usdcop-marketdata'

# Business Hours Configuration - Colombian market (ALIGNED WITH L1)
BUSINESS_HOURS_START = 8  # 8 AM COT
BUSINESS_HOURS_END = 13   # 1 PM COT (premium window end - data until 12:55 PM)
BARS_PER_HOUR = 12        # For 5-minute frequency
EXPECTED_BARS_PER_DAY = 60  # Premium window: 8:00-12:55 = 60 bars (aligned with L1)

# Historical data range (if no data exists)
START_DATE = datetime(2020, 1, 1, tzinfo=COT_TIMEZONE)
BATCH_SIZE_DAYS = 7  # Process 1 week at a time for better API management

# PostgreSQL Configuration
POSTGRES_CONFIG = {
    'host': 'usdcop-postgres-timescale',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

def ensure_timezone_aware(timestamp, target_tz='America/Bogota'):
    """Ensure timestamp is timezone-aware"""
    if timestamp is None:
        return None

    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)

    if hasattr(timestamp, 'tz') and timestamp.tz is None:
        target_pytz = pytz.timezone(target_tz)
        return timestamp.tz_localize(target_pytz)
    elif not hasattr(timestamp, 'tz'):
        timestamp = pd.Timestamp(timestamp)
        target_pytz = pytz.timezone(target_tz)
        return timestamp.tz_localize(target_pytz)

    return timestamp

def get_postgres_connection():
    """Get PostgreSQL connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logging.error(f"❌ PostgreSQL connection failed: {e}")
        return None

def check_existing_data(**context):
    """Check what data already exists and determine what needs to be fetched"""

    logging.info("🔍 INTELLIGENT GAP DETECTION - Checking existing data...")

    gap_info = {
        'has_data': False,
        'last_date': None,
        'fetch_mode': 'unknown',
        'date_ranges': [],
        'total_records': 0
    }

    try:
        # Check PostgreSQL first (UNIFIED TABLE: usdcop_m5_ohlcv)
        conn = get_postgres_connection()
        if conn:
            with conn.cursor() as cursor:
                # Check if we have USD/COP data in unified table
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_records,
                        MIN(time) as first_date,
                        MAX(time) as last_date
                    FROM usdcop_m5_ohlcv
                    WHERE symbol = 'USD/COP'
                """)

                result = cursor.fetchone()
                total_records, first_date, last_date = result

                gap_info['total_records'] = total_records or 0
                gap_info['has_data'] = (total_records or 0) > 0

                if gap_info['has_data']:
                    gap_info['last_date'] = last_date
                    logging.info(f"✅ Found {total_records:,} existing USD/COP records in usdcop_m5_ohlcv")
                    logging.info(f"📅 Date range: {first_date} → {last_date}")
                else:
                    logging.info("📝 No existing USD/COP data found - will fetch complete historical")

            conn.close()

        # Determine fetch strategy
        current_time = datetime.now(COT_TIMEZONE)

        if not gap_info['has_data']:
            # No data exists - fetch everything from 2020
            gap_info['fetch_mode'] = 'complete_historical'
            gap_info['date_ranges'] = calculate_historical_batches(START_DATE, current_time)
            logging.info(f"📊 COMPLETE HISTORICAL MODE: {len(gap_info['date_ranges'])} batches from 2020")

        else:
            # Data exists - NEW: Use comprehensive gap detection to find ALL missing days
            logging.info("🔍 Scanning for ALL historical gaps (including intermediate missing days)...")

            try:
                from utils.comprehensive_gap_detector import find_all_missing_days

                # Detect ALL missing trading days from 2020 to today
                missing_ranges = find_all_missing_days(
                    start_date=START_DATE,
                    end_date=current_time,
                    postgres_config=POSTGRES_CONFIG,
                    symbol='USD/COP'
                )

                if missing_ranges:
                    # Found intermediate gaps - backfill them
                    gap_info['fetch_mode'] = 'fill_historical_gaps'
                    gap_info['date_ranges'] = [
                        (start.isoformat(), end.isoformat())
                        for start, end in missing_ranges
                    ]
                    logging.info(f"🔧 FILL HISTORICAL GAPS MODE: {len(gap_info['date_ranges'])} gap ranges found")
                    logging.info(f"📦 Will backfill {sum((end - start).days + 1 for start, end in missing_ranges)} trading days")

                else:
                    # No historical gaps found - check if we need recent data
                    logging.info("✅ NO GAPS FOUND - All trading days have data")

                    last_date_pg = gap_info['last_date']

                    # Convert PostgreSQL datetime to timezone-aware datetime
                    if hasattr(last_date_pg, 'astimezone'):
                        last_date_aware = last_date_pg.astimezone(COT_TIMEZONE)
                    else:
                        last_date_aware = last_date_pg.replace(tzinfo=UTC_TIMEZONE).astimezone(COT_TIMEZONE)

                    # Calculate hours since last data
                    time_diff = current_time - last_date_aware
                    hours_gap = time_diff.total_seconds() / 3600

                    logging.info(f"⏰ Time gap since last data: {hours_gap:.1f} hours")

                    if hours_gap <= 0.5:
                        gap_info['fetch_mode'] = 'up_to_date'
                        gap_info['date_ranges'] = []
                        logging.info("✅ Data is completely up-to-date - no fetch needed")
                    else:
                        # Recent data needed (less than 30 minutes old data missing)
                        gap_info['fetch_mode'] = 'recent_incremental'
                        start_fetch = last_date_aware - timedelta(hours=2)  # Small overlap
                        gap_info['date_ranges'] = calculate_incremental_batches(start_fetch, current_time)
                        logging.info(f"🔄 RECENT INCREMENTAL: {len(gap_info['date_ranges'])} daily batches")

            except ImportError as ie:
                # Fallback if comprehensive_gap_detector not available
                logging.warning(f"⚠️  Comprehensive gap detector not available: {ie}")
                logging.info("Falling back to last_date-based gap detection")

                last_date_pg = gap_info['last_date']
                if hasattr(last_date_pg, 'astimezone'):
                    last_date_aware = last_date_pg.astimezone(COT_TIMEZONE)
                else:
                    last_date_aware = last_date_pg.replace(tzinfo=UTC_TIMEZONE).astimezone(COT_TIMEZONE)

                time_diff = current_time - last_date_aware
                hours_gap = time_diff.total_seconds() / 3600
                days_gap = hours_gap / 24

                if hours_gap <= 0.5:
                    gap_info['fetch_mode'] = 'up_to_date'
                    gap_info['date_ranges'] = []
                elif days_gap <= 7:
                    gap_info['fetch_mode'] = 'recent_incremental'
                    start_fetch = last_date_aware - timedelta(hours=2)
                    gap_info['date_ranges'] = calculate_incremental_batches(start_fetch, current_time)
                else:
                    gap_info['fetch_mode'] = 'large_incremental'
                    gap_info['date_ranges'] = calculate_historical_batches(last_date_aware, current_time)

        # Push info to XCom
        context['ti'].xcom_push(key='gap_info', value=gap_info)

        logging.info("="*60)
        logging.info("🎯 GAP DETECTION SUMMARY")
        logging.info(f"📊 Existing records: {gap_info['total_records']:,}")
        logging.info(f"🔄 Fetch mode: {gap_info['fetch_mode']}")
        logging.info(f"📦 Batches to process: {len(gap_info['date_ranges'])}")
        logging.info("="*60)

        return gap_info

    except Exception as e:
        logging.error(f"❌ Error in gap detection: {e}")
        # Fallback - assume complete historical fetch needed
        gap_info['fetch_mode'] = 'complete_historical'
        gap_info['date_ranges'] = calculate_historical_batches(START_DATE, datetime.now(COT_TIMEZONE))
        context['ti'].xcom_push(key='gap_info', value=gap_info)
        return gap_info

def calculate_historical_batches(start_date, end_date):
    """Calculate batches for complete historical data fetch"""

    batches = []
    current_date = start_date

    while current_date < end_date:
        batch_end = min(current_date + timedelta(days=BATCH_SIZE_DAYS), end_date)
        batches.append((current_date.isoformat(), batch_end.isoformat()))
        current_date = batch_end

    logging.info(f"📊 Historical batches: {len(batches)} × {BATCH_SIZE_DAYS} days each")
    return batches

def calculate_incremental_batches(start_date, end_date):
    """Calculate batches for incremental data fetch (daily)"""

    batches = []
    current_date = start_date

    # For incremental, use daily batches
    batch_size = timedelta(days=1)

    while current_date < end_date:
        batch_end = min(current_date + batch_size, end_date)
        batches.append((current_date.isoformat(), batch_end.isoformat()))
        current_date = batch_end

    logging.info(f"🔄 Daily incremental batches: {len(batches)}")
    return batches

def calculate_weekly_batches(start_date, end_date):
    """Calculate weekly batches for medium gaps"""

    batches = []
    current_date = start_date

    batch_size = timedelta(weeks=1)

    while current_date < end_date:
        batch_end = min(current_date + batch_size, end_date)
        batches.append((current_date.isoformat(), batch_end.isoformat()))
        current_date = batch_end

    logging.info(f"📅 Weekly batches: {len(batches)}")
    return batches

def calculate_monthly_batches(start_date, end_date):
    """Calculate monthly batches for large gaps"""

    batches = []
    current_date = start_date

    while current_date < end_date:
        # Calculate next month
        if current_date.month == 12:
            next_month = current_date.replace(year=current_date.year + 1, month=1, day=1)
        else:
            next_month = current_date.replace(month=current_date.month + 1, day=1)

        batch_end = min(next_month, end_date)
        batches.append((current_date.isoformat(), batch_end.isoformat()))
        current_date = batch_end

    logging.info(f"📆 Monthly batches: {len(batches)}")
    return batches

def is_business_hours(timestamp_cot):
    """
    Check if timestamp is within business hours (8:00 AM - 12:55 PM COT, Mon-Fri)
    ALIGNED WITH L1 PREMIUM WINDOW
    """
    if timestamp_cot.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False

    hour = timestamp_cot.hour
    minute = timestamp_cot.minute

    # Before market open (before 8:00 AM)
    if hour < BUSINESS_HOURS_START:
        return False

    # After market close (hour 13 or later)
    if hour >= BUSINESS_HOURS_END:
        return False

    # Hour 12: only accept until minute 55 (12:55 PM)
    if hour == 12 and minute > 55:
        return False

    return True

def generate_business_hours_timestamps(start_date, end_date):
    """Generate expected timestamps for business hours only"""

    timestamps = []
    current = start_date

    while current <= end_date:
        if is_business_hours(current):
            timestamps.append(current)
        current += timedelta(minutes=5)

    return timestamps

def load_api_keys():
    """Load all available TwelveData API keys"""

    api_keys = []

    # Load all groups of API keys
    for group in ['G1', 'G2', 'G3']:
        for i in range(1, 9):  # Up to 8 keys per group
            key = os.environ.get(f'API_KEY_{group}_{i}')
            if key and key.strip() and key != 'your_twelvedata_api_key_here':
                api_keys.append(key.strip())

    # Fallback to legacy keys if needed
    if not api_keys:
        for i in range(1, 9):
            key = os.environ.get(f'TWELVEDATA_API_KEY_{i}')
            if key and key.strip():
                api_keys.append(key.strip())

    logging.info(f"✅ Loaded {len(api_keys)} TwelveData API keys")
    return api_keys

def fetch_data_intelligent(**context):
    """Intelligent data fetching based on gap analysis"""

    gap_info = context['ti'].xcom_pull(key='gap_info')

    if not gap_info or gap_info['fetch_mode'] == 'up_to_date':
        logging.info("✅ Data is up-to-date - no fetching needed")
        return {'status': 'up_to_date', 'records': 0}

    if not gap_info['date_ranges']:
        logging.info("📝 No date ranges to process")
        return {'status': 'no_ranges', 'records': 0}

    logging.info(f"🚀 Starting intelligent fetch: {gap_info['fetch_mode']}")
    logging.info(f"📦 Processing {len(gap_info['date_ranges'])} batches")

    # Load API keys
    api_keys = load_api_keys()
    if not api_keys:
        raise ValueError("No TwelveData API keys available")

    # Fetch data
    all_data = []
    current_key_index = 0
    calls_per_key = 0
    max_calls_per_key = 10  # Conservative limit

    s3_hook = S3Hook(aws_conn_id='minio_conn')

    for batch_index, (start_str, end_str) in enumerate(gap_info['date_ranges']):
        # Parse timestamps - they may already have timezone info
        start_dt = pd.to_datetime(start_str)
        if start_dt.tz is None:
            start_dt = start_dt.tz_localize(COT_TIMEZONE)
        else:
            start_dt = start_dt.tz_convert(COT_TIMEZONE)

        end_dt = pd.to_datetime(end_str)
        if end_dt.tz is None:
            end_dt = end_dt.tz_localize(COT_TIMEZONE)
        else:
            end_dt = end_dt.tz_convert(COT_TIMEZONE)

        logging.info(f"📥 Batch {batch_index + 1}/{len(gap_info['date_ranges'])}: {start_dt.date()} → {end_dt.date()}")

        # Rate limiting
        if batch_index > 0:
            delay = 8
            logging.info(f"⏳ Rate limiting: {delay}s...")
            time.sleep(delay)

        # Rotate API key if needed
        if calls_per_key >= max_calls_per_key and current_key_index < len(api_keys) - 1:
            current_key_index += 1
            calls_per_key = 0
            logging.info(f"🔄 Rotating to API key {current_key_index + 1}/{len(api_keys)}")

        current_api_key = api_keys[current_key_index]

        # Fetch data from TwelveData
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': 'USD/COP',
                'interval': '5min',
                'apikey': current_api_key,
                'start_date': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'end_date': end_dt.strftime('%Y-%m-%d %H:%M:%S'),
                'timezone': 'America/Bogota',
                'outputsize': 5000,
                'format': 'JSON'
            }

            logging.info(f"📡 API call with key {current_key_index + 1}/{len(api_keys)}")
            response = requests.get(url, params=params, timeout=30)
            calls_per_key += 1

            if response.status_code == 200:
                data = response.json()

                if 'values' in data and data['values']:
                    df = pd.DataFrame(data['values'])
                    df['datetime'] = pd.to_datetime(df['datetime'])

                    # Ensure timezone awareness
                    if df['datetime'].dt.tz is None:
                        df['datetime'] = df['datetime'].dt.tz_localize('America/Bogota')

                    df = df.rename(columns={'datetime': 'time'})

                    # Convert price columns
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                    if 'volume' not in df.columns:
                        df['volume'] = 0
                    else:
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)

                    # Filter for business hours only
                    df_business = df[df['time'].apply(lambda x: is_business_hours(x))]

                    # Add metadata
                    df_business['source'] = 'twelvedata'
                    df_business['batch_id'] = f"intelligent_batch_{batch_index}"

                    # Add additional Colombian timezone fields
                    df_business['timestamp_cot'] = df_business['time'].dt.tz_convert('America/Bogota')
                    df_business['hour_cot'] = df_business['timestamp_cot'].dt.hour
                    df_business['weekday'] = df_business['timestamp_cot'].dt.weekday
                    df_business['timezone'] = 'America/Bogota'

                    all_data.append(df_business)

                    logging.info(f"✅ Downloaded {len(df_business)} business-hours bars")

                elif 'code' in data and data['code'] == 429:
                    logging.error(f"⚠️ Rate limit hit on key {current_key_index + 1}")
                    if current_key_index < len(api_keys) - 1:
                        current_key_index += 1
                        calls_per_key = 0
                        logging.info(f"🔄 Switching to backup key {current_key_index + 1}")
                    else:
                        logging.error("❌ All API keys exhausted!")
                        break
                else:
                    logging.warning(f"⚠️ No data for period {start_dt.date()} → {end_dt.date()}")

            else:
                logging.error(f"❌ API request failed: {response.status_code}")

        except Exception as e:
            logging.error(f"❌ Error fetching data for batch {batch_index}: {e}")
            continue

    # Process and save results
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)
        df_final = df_combined.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)

        logging.info(f"✅ Total data fetched: {len(df_final):,} records")

        # Save to unified dataset
        save_result = save_unified_dataset(df_final, s3_hook, context)

        # Insert to PostgreSQL
        postgres_result = insert_to_postgres(df_final)

        result = {
            'status': 'success',
            'records': len(df_final),
            'fetch_mode': gap_info['fetch_mode'],
            'saved_to_s3': save_result,
            'saved_to_postgres': postgres_result
        }

        # Push results to XCom
        context['ti'].xcom_push(key='fetch_result', value=result)

        return result

    else:
        logging.error("❌ No data was fetched")
        return {'status': 'failed', 'records': 0}

def save_unified_dataset(df, s3_hook, context):
    """ROBUSTO: Actualizar CSV unificado SIN crear archivos extra"""

    try:
        if len(df) == 0:
            logging.info("📝 No new data to save to CSV")
            return True

        # 1. SOLO actualizar el archivo LATEST.csv principal (sin timestamps)
        latest_key = f"UNIFIED_COMPLETE/LATEST.csv"

        # 2. Leer CSV existente si existe
        existing_df = None
        try:
            existing_csv = s3_hook.read_key(latest_key, bucket_name=BUCKET_OUTPUT)
            if existing_csv:
                existing_df = pd.read_csv(io.StringIO(existing_csv))
                logging.info(f"📄 Found existing CSV with {len(existing_df):,} records")
        except Exception as e:
            logging.info(f"📝 No existing CSV found, creating new one")

        # 3. Combinar datos existentes + nuevos (sin duplicados)
        if existing_df is not None:
            # Combinar y eliminar duplicados por timestamp
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df['time'] = pd.to_datetime(combined_df['time'])
            final_df = combined_df.sort_values('time').drop_duplicates(subset=['time']).reset_index(drop=True)
            logging.info(f"🔄 Combined data: {len(existing_df):,} existing + {len(df):,} new = {len(final_df):,} total")
        else:
            final_df = df.copy()
            logging.info(f"📝 New CSV with {len(final_df):,} records")

        # 4. Guardar CSV actualizado
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        s3_hook.load_string(
            string_data=csv_data,
            bucket_name=BUCKET_OUTPUT,
            key=latest_key,
            replace=True
        )

        # 5. Actualizar metadata simple
        metadata = {
            'last_update': datetime.now().isoformat(),
            'total_records': len(final_df),
            'date_range': {
                'start': str(final_df['time'].min()),
                'end': str(final_df['time'].max())
            },
            'source': 'robust_pipeline',
            'new_records_added': len(df)
        }

        metadata_key = f"UNIFIED_COMPLETE/metadata.json"
        s3_hook.load_string(
            string_data=json.dumps(metadata, indent=2, default=str),
            bucket_name=BUCKET_OUTPUT,
            key=metadata_key,
            replace=True
        )

        logging.info(f"💾 ROBUSTO: Updated CSV with {len(df):,} new records")
        logging.info(f"📊 Total CSV records: {len(final_df):,}")
        logging.info(f"🔗 CSV: s3://{BUCKET_OUTPUT}/{latest_key}")

        return True

    except Exception as e:
        logging.error(f"❌ Error updating CSV: {e}")
        return False

def insert_to_postgres(df):
    """Insert data to PostgreSQL usdcop_m5_ohlcv table (UNIFIED TABLE)"""

    try:
        conn = get_postgres_connection()
        if not conn:
            return False

        # Prepare data for insertion - ensure timezone-aware timestamps
        data_tuples = []
        for _, row in df.iterrows():
            # Ensure timestamp is timezone-aware
            timestamp = row['time']
            if pd.isna(timestamp):
                continue

            if hasattr(timestamp, 'tz') and timestamp.tz is None:
                timestamp = timestamp.tz_localize('America/Bogota')
            elif not hasattr(timestamp, 'tz'):
                timestamp = pd.Timestamp(timestamp).tz_localize('America/Bogota')

            # Extract OHLCV values
            open_price = float(row.get('open', row['close']))
            high_price = float(row.get('high', row['close']))
            low_price = float(row.get('low', row['close']))
            close_price = float(row['close'])

            data_tuples.append((
                timestamp,                        # time (timezone-aware)
                'USD/COP',                       # symbol (mantener consistencia con RT service)
                open_price,                      # open
                high_price,                      # high
                low_price,                       # low
                close_price,                     # close
                int(row.get('volume', 0)),       # volume
                row.get('source', 'twelvedata')  # source
            ))

        if not data_tuples:
            logging.warning("⚠️ No valid data tuples to insert")
            return False

        insert_sql = """
        INSERT INTO usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume, source)
        VALUES %s
        ON CONFLICT (time, symbol) DO UPDATE SET
            open = EXCLUDED.open,
            high = GREATEST(usdcop_m5_ohlcv.high, EXCLUDED.high),
            low = LEAST(usdcop_m5_ohlcv.low, EXCLUDED.low),
            close = EXCLUDED.close,
            volume = usdcop_m5_ohlcv.volume + EXCLUDED.volume,
            source = CASE
                WHEN EXCLUDED.source = 'twelvedata' THEN EXCLUDED.source
                ELSE usdcop_m5_ohlcv.source
            END,
            updated_at = NOW()
        """

        with conn.cursor() as cursor:
            # Execute in smaller batches to avoid memory issues
            batch_size = 500
            for i in range(0, len(data_tuples), batch_size):
                batch = data_tuples[i:i + batch_size]
                execute_values(
                    cursor, insert_sql, batch,
                    template=None, page_size=batch_size
                )
            conn.commit()

        conn.close()

        logging.info(f"🐘 Inserted {len(data_tuples):,} records to usdcop_m5_ohlcv (UNIFIED TABLE)")
        return True

    except Exception as e:
        logging.error(f"❌ PostgreSQL insertion failed: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def generate_final_report(**context):
    """Generate final execution report"""

    gap_info = context['ti'].xcom_pull(key='gap_info')
    fetch_result = context['ti'].xcom_pull(key='fetch_result') or {}

    report = {
        'execution_time': datetime.now().isoformat(),
        'gap_analysis': gap_info,
        'fetch_result': fetch_result,
        'status': fetch_result.get('status', 'unknown')
    }

    # Save report
    try:
        s3_hook = S3Hook(aws_conn_id='minio_conn')
        report_key = f"REPORTS/intelligent_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        s3_hook.load_string(
            string_data=json.dumps(report, indent=2, default=str),
            bucket_name=BUCKET_OUTPUT,
            key=report_key,
            replace=True
        )

        logging.info(f"📋 Report saved: s3://{BUCKET_OUTPUT}/{report_key}")

    except Exception as e:
        logging.error(f"❌ Error saving report: {e}")

    logging.info("="*60)
    logging.info("🎉 INTELLIGENT PIPELINE EXECUTION COMPLETED")
    logging.info(f"📊 Mode: {gap_info.get('fetch_mode', 'unknown')}")
    logging.info(f"📈 Records processed: {fetch_result.get('records', 0):,}")
    logging.info(f"✅ Status: {fetch_result.get('status', 'unknown')}")
    logging.info("="*60)

    return report

def export_to_minio(**context):
    """
    📦 EXPORT PostgreSQL → MinIO
    Creates L0 parquet files in MinIO from PostgreSQL data with contracts for L1
    """
    logging.info("="*60)
    logging.info("📦 EXPORTING PostgreSQL → MinIO + CONTRACTS")
    logging.info("="*60)

    try:
        s3_hook = S3Hook(aws_conn_id='minio_conn')

        # Read ALL data from PostgreSQL
        conn = get_postgres_connection()
        if not conn:
            raise ValueError("Cannot connect to PostgreSQL")

        query = """
        SELECT
            time,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            source
        FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
        ORDER BY time
        """

        df = pd.read_sql(query, conn)
        conn.close()

        logging.info(f"📊 Loaded {len(df):,} rows from PostgreSQL")

        if len(df) == 0:
            logging.warning("⚠️ No data in PostgreSQL to export")
            return {'status': 'no_data'}

        # Prepare for MinIO
        df['time_utc'] = pd.to_datetime(df['time'], utc=True)
        latest_date = df['time_utc'].max().strftime('%Y-%m-%d')
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save parquet with L0 structure
        output_key = f"{DAG_ID}/market=usdcop/timeframe=m5/source=twelvedata/date={latest_date}/run_id={run_id}/premium_data.parquet"

        parquet_buffer = io.BytesIO()
        df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
        parquet_buffer.seek(0)

        s3_hook.load_bytes(
            bytes_data=parquet_buffer.read(),
            bucket_name=BUCKET_OUTPUT,
            key=output_key,
            replace=True
        )

        logging.info(f"✅ Parquet saved: s3://{BUCKET_OUTPUT}/{output_key}")

        # Generate contract for L1
        contract = {
            'dag_id': DAG_ID,
            'run_id': run_id,
            'execution_date': latest_date,
            'total_records': len(df),
            'date_range': {
                'start': str(df['time_utc'].min()),
                'end': str(df['time_utc'].max())
            },
            'columns': list(df.columns),
            'source': 'twelvedata',
            'export_timestamp': datetime.now().isoformat(),
            'ready_for_l1': True
        }

        contract_key = f"{DAG_ID}/market=usdcop/timeframe=m5/source=twelvedata/date={latest_date}/run_id={run_id}/contract.json"
        s3_hook.load_string(
            string_data=json.dumps(contract, indent=2, default=str),
            bucket_name=BUCKET_OUTPUT,
            key=contract_key,
            replace=True
        )

        logging.info(f"✅ Contract saved: s3://{BUCKET_OUTPUT}/{contract_key}")
        logging.info(f"📦 Export Complete: {len(df):,} rows → MinIO")

        return {
            'status': 'success',
            'records_exported': len(df),
            'output_key': output_key,
            'contract_key': contract_key
        }

    except Exception as e:
        logging.error(f"❌ MinIO export failed: {e}")
        raise

def load_to_dwh(**context):
    """
    Load L0 data to Data Warehouse (Kimball model).

    This function:
    1. Inserts OHLCV bars into fact_bar_5m
    2. Inserts pipeline metrics into fact_l0_acquisition
    3. Logs operations to audit log
    """
    logging.info("="*60)
    logging.info("🏛️ LOADING DATA TO DWH (KIMBALL)")
    logging.info("="*60)

    try:
        # Get data from XCom
        gap_info = context['ti'].xcom_pull(key='gap_info')
        fetch_result = context['ti'].xcom_pull(key='fetch_result') or {}

        if not fetch_result or fetch_result.get('status') != 'success':
            logging.warning("⚠️ No successful fetch result, skipping DWH load")
            return {'status': 'skipped', 'reason': 'no_data'}

        # Get DAG run information
        dag_run = context['dag_run']
        run_id = f"L0_{dag_run.run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution_date = dag_run.execution_date

        # Connect to DWH
        conn = get_dwh_connection()
        dwh = DWHHelper(conn)

        # Step 1: Get or insert dimensions
        logging.info("📊 Step 1: Managing dimensions...")
        symbol_id = dwh.get_or_insert_dim_symbol(
            symbol_code='USD/COP',
            base_currency='USD',
            quote_currency='COP',
            symbol_type='forex',
            exchange='Colombian Market'
        )

        source_id = dwh.get_or_insert_dim_source(
            source_name='twelvedata',
            source_type='api',
            api_endpoint='https://api.twelvedata.com',
            cost_per_call=0.0,
            rate_limit_per_min=8
        )

        logging.info(f"✅ Dimensions: symbol_id={symbol_id}, source_id={source_id}")

        # Step 2: Load bars from PostgreSQL public.usdcop_m5_ohlcv into fact_bar_5m
        logging.info("📊 Step 2: Loading bars into fact_bar_5m...")

        # Determine date range from gap_info
        if gap_info and 'date_ranges' in gap_info and gap_info['date_ranges']:
            date_ranges = gap_info['date_ranges']
            # date_ranges is a list of tuples: [(start_str, end_str), ...]
            start_date = pd.to_datetime(date_ranges[0][0])  # First tuple, first element
            end_date = pd.to_datetime(date_ranges[-1][1])  # Last tuple, second element
        else:
            # Fallback: last 7 days
            end_date = datetime.now(UTC_TIMEZONE)
            start_date = end_date - timedelta(days=7)

        # Fetch bars from public.usdcop_m5_ohlcv
        query_bars = """
            SELECT time as ts_utc, open, high, low, close, volume
            FROM usdcop_m5_ohlcv
            WHERE symbol = 'USD/COP'
            AND time >= %s AND time <= %s
            ORDER BY time
        """

        with conn.cursor() as cur:
            cur.execute(query_bars, (start_date, end_date))
            rows = cur.fetchall()

        if rows:
            bars = [
                {
                    'ts_utc': row[0],
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': int(row[5]) if row[5] else 0
                }
                for row in rows
            ]

            rows_inserted = dwh.insert_fact_bars_bulk(
                bars=bars,
                symbol_id=symbol_id,
                source_id=source_id
            )
            logging.info(f"✅ Inserted {rows_inserted} bars into fact_bar_5m")
        else:
            logging.warning("⚠️ No bars found to insert into fact_bar_5m")
            rows_inserted = 0

        # Step 3: Insert L0 acquisition metrics into fact_l0_acquisition
        logging.info("📊 Step 3: Loading metrics into fact_l0_acquisition...")

        metrics = {
            'fetch_mode': gap_info.get('fetch_mode', 'unknown'),
            'date_range_start': start_date,
            'date_range_end': end_date,
            'rows_fetched': fetch_result.get('records', 0),
            'rows_inserted': rows_inserted,
            'rows_duplicated': 0,
            'rows_rejected': 0,
            'stale_rate_pct': fetch_result.get('stale_rate', 0.0) * 100,
            'coverage_pct': fetch_result.get('coverage', 0.0) * 100,
            'gaps_detected': len(gap_info.get('date_ranges', [])),
            'duration_sec': fetch_result.get('duration_sec', 0),
            'api_calls_count': fetch_result.get('api_calls', 0),
            'api_cost_usd': 0.0,  # TwelveData is free
            'quality_passed': fetch_result.get('status') == 'success',
            'minio_manifest_path': f"s3://{BUCKET_OUTPUT}/REPORTS/"
        }

        dwh.insert_fact_l0_acquisition(
            run_id=run_id,
            symbol_id=symbol_id,
            source_id=source_id,
            execution_date=execution_date,
            metrics=metrics,
            dag_id=DAG_ID,
            task_id='load_to_dwh'
        )

        logging.info(f"✅ Inserted L0 acquisition metrics for run_id: {run_id}")

        # Close connections
        dwh.close()

        result = {
            'status': 'success',
            'run_id': run_id,
            'symbol_id': symbol_id,
            'source_id': source_id,
            'bars_inserted': rows_inserted,
            'metrics_inserted': True
        }

        logging.info("="*60)
        logging.info("🎉 DWH LOAD COMPLETED SUCCESSFULLY")
        logging.info(f"📊 Bars inserted: {rows_inserted}")
        logging.info(f"📊 Run ID: {run_id}")
        logging.info("="*60)

        return result

    except Exception as e:
        logging.error(f"❌ DWH load failed: {e}")
        import traceback
        logging.error(traceback.format_exc())

        return {
            'status': 'failed',
            'error': str(e)
        }

# DAG Definition
default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L0 Intelligent Acquire - Detects ALL gaps (including intermediate) and fetches via TwelveData API',
    schedule_interval=None,  # MANUAL EXECUTION - Detects gaps in trading hours when triggered
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'intelligent', 'usdcop', 'm5', 'manual', 'gap-filling', 'comprehensive', 'dwh']
)

with dag:
    # Task 1: Check existing data and determine gaps
    task_check_gaps = PythonOperator(
        task_id='check_existing_data_gaps',
        python_callable=check_existing_data,
        provide_context=True
    )

    # Task 2: Intelligent data fetching
    task_fetch_data = PythonOperator(
        task_id='fetch_missing_data_intelligent',
        python_callable=fetch_data_intelligent,
        provide_context=True
    )

    # Task 3: Generate final report
    task_generate_report = PythonOperator(
        task_id='generate_execution_report',
        python_callable=generate_final_report,
        provide_context=True
    )

    # Task 4: Load to Data Warehouse (Kimball)
    task_load_dwh = PythonOperator(
        task_id='load_to_dwh',
        python_callable=load_to_dwh,
        provide_context=True
    )

    # Task 5: Export PostgreSQL to MinIO (for L1 consumption)
    task_export_minio = PythonOperator(
        task_id='export_to_minio',
        python_callable=export_to_minio,
        provide_context=True
    )

    # Task dependencies
    task_check_gaps >> task_fetch_data >> task_generate_report >> task_load_dwh >> task_export_minio