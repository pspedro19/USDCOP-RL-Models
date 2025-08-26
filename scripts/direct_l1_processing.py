#!/usr/bin/env python3
"""
Direct L1 processing without Airflow
Processes L0 data directly to L1 standardized format
"""

import pandas as pd
import numpy as np
import boto3
from botocore.client import Config
import json
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timezone
import logging
from pathlib import Path
import io
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L0 = "ds-usdcop-acquire"
BUCKET_L1 = "ds-usdcop-standardize"

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)

def connect_to_minio():
    """Connect to MinIO"""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )

def process_single_date(s3_client, date_str):
    """Process a single date from L0 to L1"""
    try:
        # Read L0 data
        prefix = f"usdcop_m5__01_l0_acquire/market=usdcop/timeframe=m5/source=twelvedata/date={date_str}/"
        
        response = s3_client.list_objects_v2(Bucket=BUCKET_L0, Prefix=prefix)
        if 'Contents' not in response:
            return {'date': date_str, 'status': 'no_data'}
        
        # Find parquet file
        parquet_key = None
        for obj in response['Contents']:
            if 'data.parquet' in obj['Key']:
                parquet_key = obj['Key']
                break
        
        if not parquet_key:
            return {'date': date_str, 'status': 'no_parquet'}
        
        # Read parquet data
        obj = s3_client.get_object(Bucket=BUCKET_L0, Key=parquet_key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Standardize the data
        df = standardize_data(df, date_str)
        
        if df is None or len(df) == 0:
            return {'date': date_str, 'status': 'empty_data'}
        
        # Save to L1
        save_to_l1(s3_client, df, date_str)
        
        return {'date': date_str, 'status': 'success', 'rows': len(df)}
        
    except Exception as e:
        logger.error(f"Error processing {date_str}: {e}")
        return {'date': date_str, 'status': 'error', 'error': str(e)}

def standardize_data(df, date_str):
    """Standardize L0 data to L1 format"""
    try:
        # Ensure required columns
        if 'time' not in df.columns:
            return None
        
        # Parse time and filter for premium window (08:00-12:55 COT)
        df['time_utc'] = pd.to_datetime(df['time'])
        df['time_cot'] = df['time_utc'].dt.tz_localize('UTC').dt.tz_convert('America/Bogota')
        
        # Filter premium window
        df['hour_cot'] = df['time_cot'].dt.hour
        df['minute_cot'] = df['time_cot'].dt.minute
        df = df[(df['hour_cot'] >= 8) & (df['hour_cot'] < 13)].copy()
        
        if df['hour_cot'].max() == 12:
            df = df[~((df['hour_cot'] == 12) & (df['minute_cot'] > 55))].copy()
        
        if len(df) == 0:
            return None
        
        # Add episode structure
        df['episode_id'] = date_str
        df['t_in_episode'] = range(len(df))
        df['is_terminal'] = False
        if len(df) > 0:
            df.loc[df.index[-1], 'is_terminal'] = True
        
        # Add quality flags
        df['ohlc_valid'] = (
            (df['high'] >= df[['open', 'close']].max(axis=1)) &
            (df[['open', 'close']].min(axis=1) >= df['low'])
        )
        
        # Check for stale data
        if len(df) > 1:
            df['price_change'] = df['close'].diff().abs()
            df['is_stale'] = df['price_change'] == 0
            df.loc[df.index[0], 'is_stale'] = False
        else:
            df['is_stale'] = False
        
        # Select final columns (13 columns as specified)
        final_columns = [
            'episode_id', 't_in_episode', 'is_terminal',
            'time_utc', 'time_cot', 'hour_cot', 'minute_cot',
            'open', 'high', 'low', 'close',
            'ohlc_valid', 'is_stale'
        ]
        
        df_final = df[final_columns].copy()
        
        return df_final
        
    except Exception as e:
        logger.error(f"Error standardizing data for {date_str}: {e}")
        return None

def save_to_l1(s3_client, df, date_str):
    """Save standardized data to L1 bucket"""
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    base_path = f"usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date={date_str}/run_id={run_id}"
    
    # Save parquet
    table = pa.Table.from_pandas(df)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    parquet_bytes = buffer.getvalue().to_pybytes()
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key=f"{base_path}/standardized_data.parquet",
        Body=parquet_bytes
    )
    
    # Calculate hash
    data_hash = hashlib.sha256(parquet_bytes).hexdigest()
    
    # Save CSV with 6 decimal precision
    df_csv = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].map(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    # Convert datetime columns to string
    for col in df_csv.columns:
        if pd.api.types.is_datetime64_any_dtype(df_csv[col]):
            df_csv[col] = df_csv[col].astype(str)
    
    csv_data = df_csv.to_csv(index=False)
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key=f"{base_path}/standardized_data.csv",
        Body=csv_data.encode('utf-8')
    )
    
    # Save metadata
    metadata = {
        'dataset_version': 'v2.0',
        'date': date_str,
        'run_id': run_id,
        'rows': len(df),
        'rows_expected': 60,
        'completeness_pct': (len(df) / 60) * 100,
        'data_hash': data_hash,
        'columns': df.columns.tolist(),
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    }
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key=f"{base_path}/_metadata.json",
        Body=json.dumps(metadata, indent=2, cls=NumpyEncoder).encode('utf-8')
    )
    
    # Save READY signal
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key=f"{base_path}/_control/READY",
        Body=json.dumps({'timestamp': datetime.utcnow().isoformat(), 'status': 'READY'}).encode('utf-8')
    )

def process_all_dates():
    """Process all L0 dates to L1"""
    s3_client = connect_to_minio()
    
    # Get all L0 dates
    logger.info("Getting all L0 dates...")
    paginator = s3_client.get_paginator('list_objects_v2')
    dates = set()
    
    for page in paginator.paginate(
        Bucket=BUCKET_L0,
        Prefix="usdcop_m5__01_l0_acquire/market=usdcop/timeframe=m5/source=twelvedata/date="
    ):
        if 'Contents' in page:
            for obj in page['Contents']:
                if 'date=' in obj['Key'] and 'data.parquet' in obj['Key']:
                    date = obj['Key'].split('date=')[1].split('/')[0]
                    if len(date) == 10:
                        dates.add(date)
    
    dates = sorted(list(dates))
    logger.info(f"Found {len(dates)} dates to process")
    
    if not dates:
        logger.error("No dates found in L0")
        return False
    
    # Process each date
    results = {
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    logger.info(f"Processing {len(dates)} dates...")
    for date_str in tqdm(dates, desc="Processing L1"):
        # Check if already processed
        check_key = f"usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date={date_str}/"
        response = s3_client.list_objects_v2(Bucket=BUCKET_L1, Prefix=check_key, MaxKeys=1)
        
        if 'Contents' in response:
            # Check if standardized_data.parquet exists
            skip = False
            for obj in response['Contents']:
                if 'standardized_data.parquet' in obj['Key']:
                    results['skipped'] += 1
                    skip = True
                    break
            if skip:
                continue
        
        # Process the date
        result = process_single_date(s3_client, date_str)
        
        if result['status'] == 'success':
            results['success'] += 1
        else:
            results['failed'] += 1
            if result['status'] == 'error':
                logger.warning(f"Failed {date_str}: {result.get('error', 'Unknown error')}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("L1 PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total dates: {len(dates)}")
    logger.info(f"✅ Successfully processed: {results['success']}")
    logger.info(f"❌ Failed: {results['failed']}")
    logger.info(f"⏭️ Skipped (already processed): {results['skipped']}")
    
    return results['success'] > 0

if __name__ == "__main__":
    success = process_all_dates()
    if success:
        logger.info("\n🎉 Direct L1 processing completed!")
        logger.info("Now run consolidate_l1_dataset.py to create the final consolidated dataset")
    else:
        logger.error("\n❌ Direct L1 processing failed")