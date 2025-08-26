#!/usr/bin/env python3
"""
Fix L1 processing - properly upload all dates to MinIO
"""

import pandas as pd
import numpy as np
import boto3
from botocore.client import Config
import json
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import logging
from pathlib import Path
import io

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

def get_unique_l0_dates(s3_client):
    """Get unique dates from L0 bucket"""
    dates = {}  # date -> best key
    
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(
        Bucket=BUCKET_L0,
        Prefix="usdcop_m5__01_l0_acquire/market=usdcop/timeframe=m5/source=twelvedata/date="
    ):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            if 'data.parquet' not in obj['Key']:
                continue
                
            # Extract date
            if 'date=' in obj['Key']:
                date = obj['Key'].split('date=')[1].split('/')[0]
                if len(date) == 10:
                    # Prefer keys with run_id (more specific)
                    if date not in dates or 'run_id=' in obj['Key']:
                        dates[date] = obj['Key']
    
    return dates

def process_and_upload(s3_client, date_str, l0_key):
    """Process L0 data and upload to L1"""
    try:
        # Read L0 data
        obj = s3_client.get_object(Bucket=BUCKET_L0, Key=l0_key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
        
        logger.info(f"  Processing {date_str}: {len(df)} rows in L0")
        
        # Parse time columns
        df['time_utc'] = pd.to_datetime(df['time'])
        
        # Handle timezone
        if df['time_utc'].dt.tz is None:
            df['time_utc'] = df['time_utc'].dt.tz_localize('UTC')
        else:
            df['time_utc'] = df['time_utc'].dt.tz_convert('UTC')
        
        df['time_cot'] = df['time_utc'].dt.tz_convert('America/Bogota')
        df['hour_cot'] = df['time_cot'].dt.hour
        df['minute_cot'] = df['time_cot'].dt.minute
        
        # Filter for premium window (08:00-12:55 COT)
        df_premium = df[(df['hour_cot'] >= 8) & (df['hour_cot'] < 13)].copy()
        if len(df_premium) > 0 and df_premium['hour_cot'].max() == 12:
            df_premium = df_premium[~((df_premium['hour_cot'] == 12) & (df_premium['minute_cot'] > 55))].copy()
        
        if len(df_premium) == 0:
            logger.warning(f"    No data in premium window for {date_str}")
            return False
        
        # Create standardized structure
        df_std = pd.DataFrame()
        df_std['episode_id'] = date_str
        df_std['t_in_episode'] = range(len(df_premium))
        df_std['is_terminal'] = False
        if len(df_std) > 0:
            df_std.loc[df_std.index[-1], 'is_terminal'] = True
        
        df_std['time_utc'] = df_premium['time_utc'].values
        df_std['time_cot'] = df_premium['time_cot'].values
        df_std['hour_cot'] = df_premium['hour_cot'].values
        df_std['minute_cot'] = df_premium['minute_cot'].values
        
        # OHLC data
        df_std['open'] = df_premium['open'].values
        df_std['high'] = df_premium['high'].values
        df_std['low'] = df_premium['low'].values
        df_std['close'] = df_premium['close'].values
        
        # Quality flags
        df_std['ohlc_valid'] = (
            (df_std['high'] >= df_std[['open', 'close']].max(axis=1)) &
            (df_std[['open', 'close']].min(axis=1) >= df_std['low'])
        )
        
        # Stale detection
        if len(df_std) > 1:
            df_std['is_stale'] = (df_std['close'].diff().abs() == 0)
            df_std.loc[0, 'is_stale'] = False
        else:
            df_std['is_stale'] = False
        
        # Upload to L1
        run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        base_path = f"usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date={date_str}/run_id={run_id}"
        
        # Save parquet
        table = pa.Table.from_pandas(df_std)
        buffer = pa.BufferOutputStream()
        pq.write_table(table, buffer, compression='snappy')
        parquet_bytes = buffer.getvalue().to_pybytes()
        
        s3_client.put_object(
            Bucket=BUCKET_L1,
            Key=f"{base_path}/standardized_data.parquet",
            Body=parquet_bytes
        )
        
        # Save CSV
        df_csv = df_std.copy()
        for col in ['open', 'high', 'low', 'close']:
            df_csv[col] = df_csv[col].map(lambda x: f'{x:.6f}' if pd.notna(x) else '')
        
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
            'rows': len(df_std),
            'rows_expected': 60,
            'completeness_pct': (len(df_std) / 60) * 100,
            'data_hash': hashlib.sha256(parquet_bytes).hexdigest(),
            'columns': df_std.columns.tolist(),
            'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
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
            Body=json.dumps({'timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'), 'status': 'READY'}).encode('utf-8')
        )
        
        logger.info(f"    ✓ Uploaded {len(df_std)} rows to L1")
        return True
        
    except Exception as e:
        logger.error(f"    ✗ Error processing {date_str}: {e}")
        return False

def main():
    """Main execution"""
    logger.info("Starting L1 processing fix")
    
    s3_client = connect_to_minio()
    
    # Get all unique L0 dates
    l0_dates = get_unique_l0_dates(s3_client)
    logger.info(f"Found {len(l0_dates)} unique dates in L0")
    
    # Check what's already in L1
    l1_dates = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    for page in paginator.paginate(
        Bucket=BUCKET_L1,
        Prefix="usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date="
    ):
        if 'Contents' in page:
            for obj in page['Contents']:
                if 'standardized_data.parquet' in obj['Key'] and 'date=' in obj['Key']:
                    date = obj['Key'].split('date=')[1].split('/')[0]
                    if len(date) == 10:
                        l1_dates.add(date)
    
    logger.info(f"Found {len(l1_dates)} dates already in L1")
    
    # Process missing dates
    to_process = set(l0_dates.keys()) - l1_dates
    logger.info(f"Need to process {len(to_process)} dates")
    
    if not to_process:
        logger.info("All dates already processed!")
        return
    
    # Process in batches
    success_count = 0
    fail_count = 0
    
    for i, date_str in enumerate(sorted(to_process), 1):
        logger.info(f"\n[{i}/{len(to_process)}] Processing {date_str}")
        
        if process_and_upload(s3_client, date_str, l0_dates[date_str]):
            success_count += 1
        else:
            fail_count += 1
        
        # Progress update every 100
        if i % 100 == 0:
            logger.info(f"Progress: {i}/{len(to_process)} processed, {success_count} success, {fail_count} failed")
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("L1 PROCESSING FIX COMPLETE")
    logger.info("="*60)
    logger.info(f"Total processed: {len(to_process)}")
    logger.info(f"✓ Success: {success_count}")
    logger.info(f"✗ Failed: {fail_count}")
    
    # Verify final state
    l1_dates_final = set()
    for page in paginator.paginate(
        Bucket=BUCKET_L1,
        Prefix="usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date="
    ):
        if 'Contents' in page:
            for obj in page['Contents']:
                if 'standardized_data.parquet' in obj['Key'] and 'date=' in obj['Key']:
                    date = obj['Key'].split('date=')[1].split('/')[0]
                    if len(date) == 10:
                        l1_dates_final.add(date)
    
    logger.info(f"\nFinal L1 state: {len(l1_dates_final)} dates with data")

if __name__ == "__main__":
    main()