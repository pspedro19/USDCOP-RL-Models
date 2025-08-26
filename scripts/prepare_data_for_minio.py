"""
Script to prepare and upload existing data to MinIO for L0->L1 pipeline integration
"""

import os
import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
import json
import boto3
from botocore.client import Config
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MinIO configuration
MINIO_ENDPOINT = 'localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
MINIO_USE_SSL = False

# Data paths
PROJECT_ROOT = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL")
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "processed/silver/SILVER_PREMIUM_ONLY_20250819_171008.csv"

# Target bucket and structure
BUCKET_L0 = "ds-usdcop-acquire"
DAG_ID = "usdcop_m5__01_l0_acquire"

def connect_to_minio():
    """Connect to MinIO"""
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=f'http://{MINIO_ENDPOINT}',
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            use_ssl=MINIO_USE_SSL
        )
        logger.info("Connected to MinIO successfully")
        return s3_client
    except Exception as e:
        logger.error(f"Failed to connect to MinIO: {e}")
        return None

def ensure_bucket_exists(s3_client, bucket_name):
    """Ensure bucket exists in MinIO"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"Bucket {bucket_name} exists")
    except:
        try:
            s3_client.create_bucket(Bucket=bucket_name)
            logger.info(f"Created bucket {bucket_name}")
        except Exception as e:
            logger.error(f"Failed to create bucket {bucket_name}: {e}")

def process_premium_csv():
    """Process the premium CSV file and convert to parquet"""
    logger.info(f"Processing CSV file: {CSV_FILE}")
    
    if not CSV_FILE.exists():
        logger.error(f"CSV file not found: {CSV_FILE}")
        return None
    
    # Read CSV
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Loaded {len(df)} rows from CSV")
    
    # Ensure required columns
    required_cols = ['time', 'open', 'high', 'low', 'close']
    
    # Check for datetime/time column (might be named differently)
    time_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
    if time_cols and 'time' not in df.columns:
        df['time'] = df[time_cols[0]]
    
    # Ensure time is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    # Add volume if not present
    if 'volume' not in df.columns:
        df['volume'] = 0.0
    
    # Add source
    df['source'] = 'twelvedata'
    
    logger.info(f"Columns in dataframe: {df.columns.tolist()}")
    
    return df

def split_data_by_date(df):
    """Split data by date for daily partitioning"""
    df['date'] = pd.to_datetime(df['time']).dt.date
    
    daily_data = {}
    for date in df['date'].unique():
        daily_df = df[df['date'] == date].copy()
        daily_df = daily_df.drop(columns=['date'])
        date_str = date.strftime('%Y-%m-%d')
        daily_data[date_str] = daily_df
        logger.info(f"Date {date_str}: {len(daily_df)} records")
    
    return daily_data

def upload_to_minio(s3_client, bucket_name, key, data):
    """Upload data to MinIO"""
    try:
        if isinstance(data, pd.DataFrame):
            # Convert to parquet
            table = pa.Table.from_pandas(data)
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer, compression='snappy')
            body = buffer.getvalue().to_pybytes()
        elif isinstance(data, str):
            body = data.encode('utf-8')
        elif isinstance(data, bytes):
            body = data
        else:
            body = json.dumps(data, default=str).encode('utf-8')
        
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=body
        )
        logger.info(f"Uploaded to {bucket_name}/{key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload {key}: {e}")
        return False

def create_ready_signal(s3_client, bucket_name, date_str, run_id):
    """Create READY signal for the pipeline"""
    ready_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'date': date_str,
        'run_id': run_id,
        'source': 'manual_upload',
        'status': 'READY'
    }
    
    ready_key = f"{DAG_ID}/_control/date={date_str}/run_id={run_id}/READY"
    return upload_to_minio(s3_client, bucket_name, ready_key, ready_data)

def main():
    """Main execution"""
    logger.info("Starting data preparation for MinIO")
    
    # Connect to MinIO
    s3_client = connect_to_minio()
    if not s3_client:
        logger.error("Failed to connect to MinIO. Is MinIO running?")
        logger.info("Start MinIO with: docker-compose up minio")
        return
    
    # Ensure bucket exists
    ensure_bucket_exists(s3_client, BUCKET_L0)
    
    # Process CSV file
    df = process_premium_csv()
    if df is None:
        return
    
    # Split by date
    daily_data = split_data_by_date(df)
    
    # Upload each day's data
    for date_str, daily_df in daily_data.items():
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
        # Upload parquet file
        parquet_key = (
            f"{DAG_ID}/market=usdcop/timeframe=m5/"
            f"source=twelvedata/date={date_str}/data.parquet"
        )
        
        if upload_to_minio(s3_client, BUCKET_L0, parquet_key, daily_df):
            # Also upload as CSV for backup
            csv_key = (
                f"{DAG_ID}/market=usdcop/timeframe=m5/"
                f"source=twelvedata/date={date_str}/data.csv"
            )
            csv_data = daily_df.to_csv(index=False)
            upload_to_minio(s3_client, BUCKET_L0, csv_key, csv_data)
            
            # Create READY signal
            create_ready_signal(s3_client, BUCKET_L0, date_str, run_id)
            
            # Create quality report
            quality_report = {
                'date': date_str,
                'records': len(daily_df),
                'completeness': 1.0,
                'source': 'twelvedata',
                'columns': daily_df.columns.tolist()
            }
            report_key = f"{DAG_ID}/_reports/date={date_str}/quality_report.json"
            upload_to_minio(s3_client, BUCKET_L0, report_key, quality_report)
    
    logger.info(f"Uploaded {len(daily_data)} days of data to MinIO")
    
    # List uploaded files
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_L0, Prefix=DAG_ID)
        if 'Contents' in response:
            logger.info(f"\nUploaded files in {BUCKET_L0}:")
            for obj in response['Contents'][:10]:  # Show first 10
                logger.info(f"  - {obj['Key']}")
    except Exception as e:
        logger.error(f"Failed to list objects: {e}")

if __name__ == "__main__":
    main()