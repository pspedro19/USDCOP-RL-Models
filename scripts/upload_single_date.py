#!/usr/bin/env python3
"""
Upload data for a single date to MinIO for testing L1 pipeline
"""

import pandas as pd
import boto3
from botocore.client import Config
import json
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L0 = "ds-usdcop-acquire"
DAG_ID = "usdcop_m5__01_l0_acquire"

# Data path
CSV_FILE = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\processed\silver\SILVER_PREMIUM_ONLY_20250819_171008.csv")
TARGET_DATE = "2020-01-03"

def upload_date_to_minio():
    """Upload data for specific date"""
    
    # Connect to MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    logger.info("Connected to MinIO")
    
    # Read CSV
    df = pd.read_csv(CSV_FILE)
    logger.info(f"Loaded {len(df)} total rows")
    
    # Filter for target date
    df['date'] = pd.to_datetime(df['time']).dt.date.astype(str)
    df_date = df[df['date'] == TARGET_DATE].copy()
    logger.info(f"Found {len(df_date)} rows for {TARGET_DATE}")
    
    if len(df_date) == 0:
        logger.error(f"No data found for {TARGET_DATE}")
        return False
    
    # Ensure required columns
    df_date['source'] = 'twelvedata'
    if 'volume' not in df_date.columns:
        df_date['volume'] = 0.0
    
    # Upload as parquet
    parquet_key = (
        f"{DAG_ID}/market=usdcop/timeframe=m5/"
        f"source=twelvedata/date={TARGET_DATE}/data.parquet"
    )
    
    # Convert to parquet bytes
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    table = pa.Table.from_pandas(df_date)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=parquet_key,
        Body=buffer.getvalue().to_pybytes()
    )
    logger.info(f"Uploaded parquet to {parquet_key}")
    
    # Also upload as CSV for backup
    csv_key = (
        f"{DAG_ID}/market=usdcop/timeframe=m5/"
        f"source=twelvedata/date={TARGET_DATE}/data.csv"
    )
    
    csv_data = df_date.to_csv(index=False)
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=csv_key,
        Body=csv_data.encode('utf-8')
    )
    logger.info(f"Uploaded CSV to {csv_key}")
    
    # Create READY signal
    ready_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'date': TARGET_DATE,
        'run_id': datetime.now().strftime("%Y%m%d%H%M%S"),
        'source': 'manual_upload',
        'status': 'READY',
        'records': len(df_date)
    }
    
    ready_key = f"{DAG_ID}/_control/date={TARGET_DATE}/run_id={ready_data['run_id']}/READY"
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=ready_key,
        Body=json.dumps(ready_data).encode('utf-8')
    )
    logger.info(f"Created READY signal at {ready_key}")
    
    # Create quality report
    quality_report = {
        'date': TARGET_DATE,
        'records': len(df_date),
        'completeness': 1.0,
        'source': 'twelvedata',
        'columns': df_date.columns.tolist(),
        'timestamp': datetime.utcnow().isoformat()
    }
    
    report_key = f"{DAG_ID}/_reports/date={TARGET_DATE}/quality_report.json"
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=report_key,
        Body=json.dumps(quality_report, default=str).encode('utf-8')
    )
    logger.info(f"Created quality report at {report_key}")
    
    # List uploaded files
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_L0,
        Prefix=f"{DAG_ID}/market=usdcop/timeframe=m5/source=twelvedata/date={TARGET_DATE}/"
    )
    
    if 'Contents' in response:
        logger.info(f"\nUploaded files for {TARGET_DATE}:")
        for obj in response['Contents']:
            logger.info(f"  - {obj['Key']}")
    
    return True

if __name__ == "__main__":
    success = upload_date_to_minio()
    if success:
        logger.info(f"\n✅ Data for {TARGET_DATE} uploaded successfully!")
        logger.info("You can now trigger the L1 pipeline for this date.")
    else:
        logger.error("\n❌ Upload failed")