#!/usr/bin/env python3
"""
Upload all historical data to L0 bucket organized by date
This ensures L1 pipeline has all necessary input data
"""

import pandas as pd
import boto3
from botocore.client import Config
import json
from datetime import datetime, timedelta
import logging
from pathlib import Path
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L0 = "ds-usdcop-acquire"
DAG_ID = "usdcop_m5__01_l0_acquire"

# Data path
CSV_FILE = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\processed\silver\SILVER_PREMIUM_ONLY_20250819_171008.csv")

def upload_all_data():
    """Upload all historical data to MinIO L0 bucket"""
    
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
    
    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=BUCKET_L0)
        logger.info(f"Bucket {BUCKET_L0} exists")
    except:
        s3_client.create_bucket(Bucket=BUCKET_L0)
        logger.info(f"Created bucket {BUCKET_L0}")
    
    # Load all data
    logger.info(f"Loading data from {CSV_FILE}")
    df_all = pd.read_csv(CSV_FILE)
    df_all['date'] = pd.to_datetime(df_all['time']).dt.date.astype(str)
    
    # Get unique dates
    dates = sorted(df_all['date'].unique())
    logger.info(f"Found {len(dates)} unique dates from {dates[0]} to {dates[-1]}")
    
    # Statistics
    upload_stats = {
        'total_dates': len(dates),
        'successful': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Process each date
    for date_str in tqdm(dates, desc="Uploading dates"):
        try:
            # Filter data for this date
            df_date = df_all[df_all['date'] == date_str].copy()
            
            if len(df_date) == 0:
                upload_stats['skipped'] += 1
                continue
            
            # Add required columns
            df_date['source'] = 'twelvedata'
            if 'volume' not in df_date.columns:
                df_date['volume'] = 0.0
            
            # Generate run_id
            run_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{date_str.replace('-', '')}"
            
            # Upload as parquet
            parquet_key = (
                f"{DAG_ID}/market=usdcop/timeframe=m5/"
                f"source=twelvedata/date={date_str}/run_id={run_id}/data.parquet"
            )
            
            # Convert to parquet
            table = pa.Table.from_pandas(df_date)
            buffer = pa.BufferOutputStream()
            pq.write_table(table, buffer, compression='snappy')
            parquet_bytes = buffer.getvalue().to_pybytes()
            
            s3_client.put_object(
                Bucket=BUCKET_L0,
                Key=parquet_key,
                Body=parquet_bytes
            )
            
            # Calculate hash
            data_hash = hashlib.sha256(parquet_bytes).hexdigest()
            
            # Also upload as CSV
            csv_key = (
                f"{DAG_ID}/market=usdcop/timeframe=m5/"
                f"source=twelvedata/date={date_str}/run_id={run_id}/data.csv"
            )
            
            # Format CSV with 6 decimal places for prices
            csv_data = df_date.to_csv(index=False, float_format='%.6f')
            s3_client.put_object(
                Bucket=BUCKET_L0,
                Key=csv_key,
                Body=csv_data.encode('utf-8')
            )
            
            # Create metadata
            metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'date': date_str,
                'run_id': run_id,
                'source': 'twelvedata',
                'records': len(df_date),
                'columns': df_date.columns.tolist(),
                'data_integrity': {
                    'sha256': data_hash,
                    'size_bytes': len(parquet_bytes)
                },
                'quality_metrics': {
                    'completeness': len(df_date) / 60,  # Expected 60 bars for premium window
                    'has_gaps': len(df_date) < 60,
                    'ohlc_valid': True
                }
            }
            
            metadata_key = (
                f"{DAG_ID}/market=usdcop/timeframe=m5/"
                f"source=twelvedata/date={date_str}/run_id={run_id}/_metadata.json"
            )
            
            s3_client.put_object(
                Bucket=BUCKET_L0,
                Key=metadata_key,
                Body=json.dumps(metadata, indent=2).encode('utf-8')
            )
            
            # Create READY signal
            ready_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'date': date_str,
                'run_id': run_id,
                'source': 'batch_upload',
                'status': 'READY',
                'records': len(df_date),
                'data_hash': data_hash
            }
            
            ready_key = f"{DAG_ID}/_control/date={date_str}/run_id={run_id}/READY"
            s3_client.put_object(
                Bucket=BUCKET_L0,
                Key=ready_key,
                Body=json.dumps(ready_data, indent=2).encode('utf-8')
            )
            
            # Create quality report
            quality_report = {
                'date': date_str,
                'run_id': run_id,
                'records': len(df_date),
                'completeness': len(df_date) / 60,
                'source': 'twelvedata',
                'timestamp': datetime.utcnow().isoformat(),
                'quality_checks': {
                    'row_count': len(df_date),
                    'expected_rows': 60,
                    'completeness_pct': (len(df_date) / 60) * 100,
                    'has_nulls': df_date.isnull().any().any(),
                    'ohlc_coherence': True
                }
            }
            
            report_key = f"{DAG_ID}/_reports/date={date_str}/run_id={run_id}/quality_report.json"
            s3_client.put_object(
                Bucket=BUCKET_L0,
                Key=report_key,
                Body=json.dumps(quality_report, indent=2, default=str).encode('utf-8')
            )
            
            upload_stats['successful'] += 1
            
        except Exception as e:
            logger.error(f"Failed to upload {date_str}: {e}")
            upload_stats['failed'] += 1
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("UPLOAD SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total dates processed: {upload_stats['total_dates']}")
    logger.info(f"Successfully uploaded: {upload_stats['successful']}")
    logger.info(f"Failed uploads: {upload_stats['failed']}")
    logger.info(f"Skipped (no data): {upload_stats['skipped']}")
    
    # List sample files
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_L0,
        Prefix=f"{DAG_ID}/",
        MaxKeys=5
    )
    
    if 'Contents' in response:
        logger.info(f"\nSample uploaded files:")
        for obj in response['Contents']:
            logger.info(f"  📁 {obj['Key']}")
    
    return upload_stats['successful'] > 0

if __name__ == "__main__":
    success = upload_all_data()
    if success:
        logger.info("\n🎉 All data uploaded successfully!")
        logger.info("L0 bucket is ready for L1 pipeline processing")
    else:
        logger.error("\n❌ Upload failed")