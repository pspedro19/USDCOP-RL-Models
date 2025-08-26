"""
DAG: usdcop_m5__01_l0_acquire_sync_incremental
================================================
Layer: L0 - ACQUIRE (RAW DATA)
Bucket: 00-raw-usdcop-marketdata

Responsabilidad:
- Download maximum historical data from 2020 onwards
- TwelveData: UTC-5 (Colombia) timezone tracking  
- 6-month batch processing for optimal quality
- Validate 12 bars/hour (5-min frequency)
- Generate detailed quality reports

IMPORTANT: This DAG focuses on RAW data acquisition with timezone preservation
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
from dateutil.relativedelta import relativedelta

# Helper function to convert numpy types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

try:
    from data_sources import TwelveDataClient
except ImportError:
    logging.warning("Data source clients not available, using fallback")
    TwelveDataClient = None

try:
    from utils import DetailedLogger
except ImportError:
    logging.warning("DetailedLogger not available, using standard logging")
    DetailedLogger = None

try:
    from utils.data_cache_manager import DataCacheManager
except ImportError:
    logging.warning("DataCacheManager not available, will fetch all data")
    DataCacheManager = None

try:
    from utils.pipeline_config import get_bucket
except ImportError:
    logging.warning("Pipeline config not available, using defaults")
    def get_bucket(layer):
        return '00-raw-usdcop-marketdata'

try:
    from airflow.models import Variable
except ImportError:
    logging.warning("Airflow Variable not available in this context")
    Variable = None

import os

# DAG Constants
DAG_ID = 'usdcop_m5__01_l0_acquire_sync_incremental'
BUCKET_OUTPUT = get_bucket('l0_acquire')

# Business Hours Configuration - Colombian market
BUSINESS_HOURS_START = 8  # 8 AM COT
BUSINESS_HOURS_END = 14   # 2 PM COT (Colombian market close)
BARS_PER_HOUR = 12        # For 5-minute frequency
EXPECTED_BARS_PER_DAY = (BUSINESS_HOURS_END - BUSINESS_HOURS_START) * BARS_PER_HOUR  # 72 bars

# Data quality thresholds
MIN_COMPLETENESS = 80  # Minimum acceptable completeness %
MIN_RECORDS_PER_BATCH = 50  # Minimum records to consider batch valid

# Historical data range
START_DATE = datetime(2020, 1, 1)  # Start from 2020 for comprehensive backtesting
BATCH_SIZE_MONTHS = 1       # Process 1 month at a time for better quality

# Colombian holidays (2021-2024 sample)
COLOMBIAN_HOLIDAYS = [
    '2021-01-01', '2021-01-11', '2021-03-22', '2021-04-01', '2021-04-02',
    '2021-05-01', '2021-05-17', '2021-06-07', '2021-06-14', '2021-07-05',
    '2021-07-20', '2021-08-07', '2021-08-16', '2021-10-18', '2021-11-01',
    '2021-11-15', '2021-12-08', '2021-12-25',
    # Add more years as needed
]

# Timezone configuration
TWELVEDATA_TIMEZONE = "UTC-5"  # Colombia time (COT)

# API Configuration 
TWELVEDATA_CONFIG = {
    'api_key': 'demo_key',  # Should be from Airflow Variable
    'symbol': 'USD/COP',
    'interval': '5min',
    'timezone': TWELVEDATA_TIMEZONE,
    'outputsize': 5000  # Max bars per request
}

def generate_run_id(**context):
    """Generate a unique run ID for this execution"""
    run_id = f"{DAG_ID}_{context['ds']}_{uuid.uuid4().hex[:8]}"
    context['ti'].xcom_push(key='run_id', value=run_id)
    logging.info(f"Generated run_id: {run_id}")
    return run_id

def calculate_date_batches(**context) -> List[Tuple[datetime, datetime]]:
    """Calculate date batches for incremental data fetching"""
    
    # For incremental loads, fetch last 7 days
    # For initial/backfill loads, fetch from START_DATE
    execution_date = context['execution_date']
    
    # Check if this is a backfill run
    is_backfill = context.get('dag_run').conf.get('backfill', False) if context.get('dag_run') else False
    
    if is_backfill:
        # Full historical load from 2020
        start_date = START_DATE
        end_date = execution_date
        logging.info(f"Backfill mode: Fetching from {start_date} to {end_date}")
    else:
        # Incremental load - last 7 days
        end_date = execution_date
        start_date = end_date - timedelta(days=7)
        logging.info(f"Incremental mode: Fetching from {start_date} to {end_date}")
    
    # Create monthly batches
    date_batches = []
    current_date = start_date
    
    while current_date < end_date:
        batch_end = min(current_date + relativedelta(months=BATCH_SIZE_MONTHS), end_date)
        date_batches.append((current_date, batch_end))
        current_date = batch_end
    
    logging.info(f"Created {len(date_batches)} date batches")
    for i, (batch_start, batch_end) in enumerate(date_batches, 1):
        logging.info(f"  Batch {i}: {batch_start.date()} to {batch_end.date()}")
    
    # Push to XCom for other tasks
    context['ti'].xcom_push(key='date_batches', value=[(str(s), str(e)) for s, e in date_batches])
    
    return date_batches

def fetch_twelvedata_historical(**context):
    """Fetch TwelveData historical data with proper rate limiting"""
    
    run_id = context['ti'].xcom_pull(key='run_id')
    date_batches = context['ti'].xcom_pull(key='date_batches')
    
    if not date_batches:
        logging.warning("No date batches to process")
        return {}
    
    # Convert string dates back to datetime
    date_batches = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in date_batches]
    
    # UPDATED API KEYS - 8 fresh keys for 6400 total daily credits
    api_keys = [
        "3656827e648a4c6fa2c4e2e7935c4fb8",  # Key 1: 800 credits
        "24fa6e96005d44bd929bffa20ae79403",  # Key 2: 800 credits
        "95cb0e0ad93949a4a30274046fcf0d10",  # Key 3: 800 credits
        "8329ac93e9f24d9ca943bdcb9df12801",  # Key 4: 800 credits
        "19c7657a44f6462fa4a91b12caaaf854",  # Key 5: 800 credits
        "df609df02abc43dd928914f21a1ab5e1",  # Key 6: 800 credits
        "d2aea0ac87504d8d9e44ac5ff54bf1c0",  # Key 7: 800 credits
        "6c62fab8415346189b3abedc7dd48d9c"   # Key 8: 800 credits
    ]
    
    # Track API key usage
    current_key_index = 0
    calls_per_key = 0
    max_calls_per_key = 8  # Conservative limit per key
    
    all_data = []
    quality_reports = []
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    for batch_index, (batch_start, batch_end) in enumerate(date_batches):
        batch_start_str = batch_start.strftime('%Y-%m-%d %H:%M:%S')
        batch_end_str = batch_end.strftime('%Y-%m-%d %H:%M:%S')
        
        logging.info(f"Fetching TwelveData for {batch_start.date()} to {batch_end.date()}")
        
        # Rate limiting - wait 8 seconds between API calls (except first)
        if batch_index > 0:  # Use batch_index from enumerate
            delay = 8
            logging.info(f"  ⏳ Rate limiting: waiting {delay} seconds before API call...")
            time.sleep(delay)
        
        # Rotate API key if needed
        if calls_per_key >= max_calls_per_key and current_key_index < len(api_keys) - 1:
            current_key_index += 1
            calls_per_key = 0
            logging.info(f"  🔄 Rotating to API key {current_key_index + 1}/{len(api_keys)}")
        
        current_api_key = api_keys[current_key_index]
        
        # Fetch data from TwelveData API
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': 'USD/COP',
                'interval': '5min',
                'apikey': current_api_key,
                'start_date': batch_start_str,
                'end_date': batch_end_str,
                'timezone': 'America/Bogota',
                'outputsize': 5000,
                'format': 'JSON'
            }
            
            logging.info(f"  📡 Making API call with key {current_key_index + 1}/{len(api_keys)}")
            response = requests.get(url, params=params, timeout=30)
            calls_per_key += 1
            
            if response.status_code == 200:
                data = response.json()
                
                if 'values' in data and data['values']:
                    df = pd.DataFrame(data['values'])
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.rename(columns={'datetime': 'time'})
                    
                    # Convert price columns to float
                    for col in ['open', 'high', 'low', 'close']:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    if 'volume' in df.columns:
                        df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
                    else:
                        df['volume'] = 0
                    
                    # Add metadata
                    df['source'] = 'twelvedata'
                    df['timezone'] = TWELVEDATA_TIMEZONE
                    df['batch_id'] = f"{run_id}_batch_{batch_index}"
                    
                    logging.info(f"  ✅ Downloaded {len(df)} bars from TwelveData")
                    
                    # Calculate quality metrics
                    quality = calculate_quality_metrics(df, batch_start, batch_end)
                    quality_reports.append(quality)
                    
                    # Save batch to S3
                    save_batch_to_s3(df, batch_start, batch_end, run_id, s3_hook, context)
                    
                    all_data.append(df)
                    
                elif 'code' in data and data['code'] == 429:
                    logging.error(f"  ⚠️ Rate limit hit on key {current_key_index + 1}")
                    if current_key_index < len(api_keys) - 1:
                        current_key_index += 1
                        calls_per_key = 0
                        logging.info(f"  🔄 Switching to backup key {current_key_index + 1}")
                    else:
                        logging.error("  ❌ All API keys exhausted!")
                        break
                else:
                    logging.warning(f"  ⚠️ No data returned for this period")
                    
            else:
                logging.error(f"  ❌ API request failed: {response.status_code}")
                logging.error(f"  Response: {response.text[:500]}")
                
        except Exception as e:
            logging.error(f"  ❌ Error fetching TwelveData: {e}")
            continue
    
    # Combine all data
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df = final_df.sort_values('time').drop_duplicates(subset=['time'])
        
        # Generate comprehensive report
        overall_quality = generate_overall_quality_report(final_df, quality_reports)
        
        # Push results to XCom
        context['ti'].xcom_push(key='twelvedata_records', value=len(final_df))
        context['ti'].xcom_push(key='twelvedata_quality', value=overall_quality)
        context['ti'].xcom_push(key='twelvedata_data_sample', value=final_df.head(1000).to_json())
        
        logging.info(f"✅ TwelveData: Fetched {len(final_df):,} total records")
        logging.info(f"   Completeness: {overall_quality.get('overall_completeness', 0):.1f}%")
        
        return {
            'status': 'success',
            'records': len(final_df),
            'quality': overall_quality
        }
    else:
        logging.error("❌ No data fetched from TwelveData")
        raise ValueError("No data available from TwelveData API. Pipeline requires real data.")

def calculate_quality_metrics(df, batch_start, batch_end):
    """Calculate quality metrics for a data batch"""
    
    # Calculate expected vs actual bars
    business_days = pd.bdate_range(start=batch_start, end=batch_end, freq='B')
    colombian_holidays_in_range = [
        pd.Timestamp(h) for h in COLOMBIAN_HOLIDAYS 
        if batch_start <= pd.Timestamp(h) <= batch_end
    ]
    trading_days = [d for d in business_days if d not in colombian_holidays_in_range]
    expected_bars = len(trading_days) * EXPECTED_BARS_PER_DAY
    
    # Filter for premium hours (8am-2pm COT)
    df['hour'] = pd.to_datetime(df['time']).dt.hour
    premium_df = df[(df['hour'] >= BUSINESS_HOURS_START) & (df['hour'] < BUSINESS_HOURS_END)]
    
    actual_premium_bars = len(premium_df)
    completeness = (actual_premium_bars / expected_bars * 100) if expected_bars > 0 else 0
    
    # Detect gaps
    if len(premium_df) > 1:
        time_diffs = premium_df['time'].diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        gap_count = len(gaps)
    else:
        gap_count = 0
    
    return {
        'batch_start': str(batch_start),
        'batch_end': str(batch_end),
        'expected_bars': expected_bars,
        'actual_bars': len(df),
        'premium_bars': actual_premium_bars,
        'completeness': completeness,
        'gap_count': gap_count,
        'trading_days': len(trading_days)
    }

def save_batch_to_s3(df, batch_start, batch_end, run_id, s3_hook, context):
    """Save data batch to S3"""
    
    # Generate file path
    year = batch_start.year
    month = f"{batch_start.month:02d}"
    day = f"{batch_start.day:02d}"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    file_key = f"{DAG_ID}/market=usdcop/timeframe=m5/source=twelvedata/year={year}/month={month}/day={day}/data_{timestamp}.parquet"
    
    # Convert to parquet
    table = pa.Table.from_pandas(df)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression='snappy')
    buffer.seek(0)
    
    # Upload to S3
    s3_hook.load_bytes(
        bytes_data=buffer.getvalue(),
        bucket_name=BUCKET_OUTPUT,
        key=file_key,
        replace=True
    )
    
    logging.info(f"  💾 Saved {len(df)} records to s3://{BUCKET_OUTPUT}/{file_key}")
    
    # Also save as CSV for debugging
    csv_key = file_key.replace('.parquet', '.csv')
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    
    s3_hook.load_string(
        string_data=csv_buffer.getvalue(),
        bucket_name=BUCKET_OUTPUT,
        key=csv_key,
        replace=True
    )

def generate_overall_quality_report(df, batch_reports):
    """Generate comprehensive quality report"""
    
    if DetailedLogger:
        logger = DetailedLogger("L0_ACQUIRE")
        
        # Log overall statistics
        logger.log_section("L0 ACQUISITION SUMMARY")
        logger.log_metric("Total Records", f"{len(df):,}")
        
        # Calculate overall completeness
        total_expected = sum(r['expected_bars'] for r in batch_reports)
        total_premium = sum(r['premium_bars'] for r in batch_reports)
        overall_completeness = (total_premium / total_expected * 100) if total_expected > 0 else 0
        
        logger.log_metric("Overall Completeness", f"{overall_completeness:.1f}%")
        logger.log_metric("Total Gaps Detected", sum(r['gap_count'] for r in batch_reports))
        
        # Log batch details
        logger.log_section("BATCH DETAILS")
        for i, report in enumerate(batch_reports, 1):
            logger.log_metric(f"Batch {i}", f"{report['premium_bars']}/{report['expected_bars']} bars ({report['completeness']:.1f}%)")
    
    # Return quality report
    return {
        'overall_completeness': overall_completeness if 'overall_completeness' in locals() else 0,
        'total_records': len(df),
        'batch_count': len(batch_reports),
        'batch_reports': batch_reports,
        'date_range': {
            'start': str(df['time'].min()) if len(df) > 0 else None,
            'end': str(df['time'].max()) if len(df) > 0 else None
        }
    }

def generate_consolidated_report(**context):
    """Generate final consolidated report"""
    
    run_id = context['ti'].xcom_pull(key='run_id')
    
    # Get TwelveData results
    twelvedata_records = context['ti'].xcom_pull(key='twelvedata_records') or 0
    twelvedata_quality = context['ti'].xcom_pull(key='twelvedata_quality') or {}
    
    # Generate final report
    final_report = {
        'run_id': run_id,
        'execution_date': context['ds'],
        'status': 'SUCCESS' if twelvedata_records > 0 else 'FAILED',
        'data_sources': {
            'twelvedata': {
                'records': twelvedata_records,
                'completeness': twelvedata_quality.get('overall_completeness', 0),
                'quality': twelvedata_quality
            }
        },
        'total_records': twelvedata_records,
        'overall_completeness': twelvedata_quality.get('overall_completeness', 0)
    }
    
    # Save report to S3
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    report_key = f"{DAG_ID}/_quality/date={context['ds']}/quality_report_{run_id}.json"
    
    s3_hook.load_string(
        string_data=json.dumps(final_report, indent=2, default=convert_numpy_types),
        bucket_name=BUCKET_OUTPUT,
        key=report_key,
        replace=True
    )
    
    # Write READY signal for downstream
    ready_key = f"{DAG_ID}/_control/date={context['ds']}/run_id={run_id}/READY"
    s3_hook.load_string(
        string_data=json.dumps({
            'ready': True,
            'timestamp': datetime.now().isoformat(),
            'records': twelvedata_records
        }),
        bucket_name=BUCKET_OUTPUT,
        key=ready_key,
        replace=True
    )
    
    logging.info("="*50)
    logging.info("L0 ACQUISITION COMPLETE")
    logging.info(f"TwelveData: {twelvedata_records:,} records ({twelvedata_quality.get('overall_completeness', 0):.1f}% complete)")
    logging.info(f"Total: {twelvedata_records:,} records")
    logging.info(f"Report saved to: s3://{BUCKET_OUTPUT}/{report_key}")
    logging.info(f"Ready signal: s3://{BUCKET_OUTPUT}/{ready_key}")
    logging.info("="*50)
    
    return final_report

# DAG Definition
default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='L0 Acquire - Incremental sync from TwelveData API',
    schedule_interval='0 1 * * *',  # Daily at 1 AM UTC
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'acquire', 'usdcop', 'm5', 'twelvedata']
)

with dag:
    # Task 1: Generate run ID
    task_generate_run_id = PythonOperator(
        task_id='generate_run_id',
        python_callable=generate_run_id,
        provide_context=True
    )
    
    # Task 2: Calculate date batches
    task_calculate_batches = PythonOperator(
        task_id='calculate_date_batches',
        python_callable=calculate_date_batches,
        provide_context=True
    )
    
    # Task 3: Fetch TwelveData historical data
    task_fetch_twelvedata = PythonOperator(
        task_id='fetch_twelvedata_historical',
        python_callable=fetch_twelvedata_historical,
        provide_context=True,
        retries=2,
        retry_delay=timedelta(minutes=5)
    )
    
    # Task 4: Generate consolidated report
    task_generate_report = PythonOperator(
        task_id='generate_consolidated_report',
        python_callable=generate_consolidated_report,
        provide_context=True
    )
    
    # Task dependencies
    task_generate_run_id >> task_calculate_batches
    task_calculate_batches >> task_fetch_twelvedata
    task_fetch_twelvedata >> task_generate_report