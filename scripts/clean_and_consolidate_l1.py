#!/usr/bin/env python3
"""
Clean L1 bucket and generate ONLY the final consolidated dataset
Produces exactly 3 files in MinIO as specified
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

def clean_l1_bucket(s3_client):
    """Clean all existing files in L1 bucket"""
    logger.info("Cleaning L1 bucket...")
    
    # List all objects
    paginator = s3_client.get_paginator('list_objects_v2')
    objects_to_delete = []
    
    try:
        for page in paginator.paginate(Bucket=BUCKET_L1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects_to_delete.append({'Key': obj['Key']})
    except Exception as e:
        logger.error(f"Error listing objects in bucket {BUCKET_L1}: {e}")
        raise
    
    if objects_to_delete:
        logger.info(f"Found {len(objects_to_delete)} objects to delete")
        
        # Show some examples of what will be deleted
        logger.info("Sample objects to be deleted:")
        for i, obj in enumerate(objects_to_delete[:5]):
            logger.info(f"  {i+1}. {obj['Key']}")
        if len(objects_to_delete) > 5:
            logger.info(f"  ... and {len(objects_to_delete) - 5} more objects")
        
        # Delete in batches of 1000 (AWS limit)
        deleted_count = 0
        try:
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i+1000]
                response = s3_client.delete_objects(
                    Bucket=BUCKET_L1,
                    Delete={'Objects': batch}
                )
                
                # Check for errors in the response
                if 'Errors' in response and response['Errors']:
                    for error in response['Errors']:
                        logger.error(f"Failed to delete {error['Key']}: {error['Message']}")
                
                if 'Deleted' in response:
                    deleted_count += len(response['Deleted'])
                    
            logger.info(f"Successfully deleted {deleted_count} objects from L1 bucket")
            
        except Exception as e:
            logger.error(f"Error during batch deletion: {e}")
            raise
        
        # Verify the bucket is now empty
        verification_objects = []
        for page in paginator.paginate(Bucket=BUCKET_L1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    verification_objects.append(obj['Key'])
        
        if verification_objects:
            logger.warning(f"WARNING: {len(verification_objects)} objects still remain after cleanup:")
            for obj in verification_objects:
                logger.warning(f"  {obj}")
        else:
            logger.info("SUCCESS: L1 bucket is now completely empty")
            
    else:
        logger.info("L1 bucket is already empty")

def process_l0_data(s3_client):
    """Process all L0 data and create consolidated dataset"""
    logger.info("Processing L0 data...")
    
    # Get all L0 dates and their data
    all_data = []
    dates_processed = []
    quality_records = []
    
    # Find all unique dates in L0
    dates_data = {}
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
                    if date not in dates_data or 'run_id=' in obj['Key']:
                        dates_data[date] = obj['Key']
    
    logger.info(f"Found {len(dates_data)} dates in L0 to process")
    
    # Process each date
    for date_str in tqdm(sorted(dates_data.keys()), desc="Processing dates"):
        try:
            # Read L0 data
            obj = s3_client.get_object(Bucket=BUCKET_L0, Key=dates_data[date_str])
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            
            # Parse time and convert to COT
            df['time_utc'] = pd.to_datetime(df['time'])
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
                continue
            
            # Create standardized structure (13 columns exactly)
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
            
            # Add to collection
            all_data.append(df_std)
            dates_processed.append(date_str)
            
            # Calculate quality metrics for this day
            n_rows = len(df_std)
            n_stale = df_std['is_stale'].sum()
            stale_rate = (n_stale / n_rows * 100) if n_rows > 0 else 0
            
            # OHLC violations
            ohlc_violations = (~df_std['ohlc_valid']).sum()
            
            # Gap analysis
            n_gaps = 60 - n_rows
            max_gap_bars = n_gaps  # Simplified
            
            # Quality flag
            completeness_pct = (n_rows / 60) * 100
            if n_rows == 60 and ohlc_violations == 0 and stale_rate <= 1:
                quality_flag = 'OK'
            elif n_rows >= 59 and ohlc_violations == 0 and stale_rate <= 2:
                quality_flag = 'WARN'
            else:
                quality_flag = 'FAIL'
            
            quality_record = {
                'date': date_str,
                'rows_expected': 60,
                'rows_found': n_rows,
                'completeness_pct': round(completeness_pct, 2),
                'n_stale': int(n_stale),
                'stale_rate': round(stale_rate, 2),
                'n_gaps': n_gaps,
                'max_gap_bars': max_gap_bars,
                'ohlc_violations': int(ohlc_violations),
                'quality_flag': quality_flag
            }
            quality_records.append(quality_record)
            
        except Exception as e:
            logger.warning(f"Error processing {date_str}: {e}")
            continue
    
    return all_data, dates_processed, quality_records

def create_and_upload_consolidated(s3_client, all_data, dates_processed, quality_records):
    """Create consolidated dataset and upload to MinIO"""
    
    if not all_data:
        logger.error("No data to consolidate!")
        return False
    
    logger.info(f"Consolidating {len(all_data)} days of data...")
    
    # 1. Create consolidated dataframe
    df_consolidated = pd.concat(all_data, ignore_index=True)
    
    # Sort by time_utc (handle timezone issues)
    df_consolidated['time_utc_str'] = df_consolidated['time_utc'].astype(str)
    df_consolidated = df_consolidated.sort_values('time_utc_str').reset_index(drop=True)
    df_consolidated = df_consolidated.drop('time_utc_str', axis=1)
    
    logger.info(f"Consolidated dataset: {len(df_consolidated)} rows, {len(df_consolidated.columns)} columns")
    
    # Generate run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get UTC window
    first_date = min(dates_processed)
    last_date = max(dates_processed)
    
    # Convert timestamps for UTC window
    time_utc_series = pd.to_datetime(df_consolidated['time_utc'].astype(str))
    utc_start = time_utc_series.min().strftime('%Y-%m-%dT%H:%M:%SZ')
    utc_end = time_utc_series.max().strftime('%Y-%m-%dT%H:%M:%SZ')
    
    # FILE 1: standardized_data.parquet
    logger.info("Uploading standardized_data.parquet...")
    table = pa.Table.from_pandas(df_consolidated)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    parquet_bytes = buffer.getvalue().to_pybytes()
    parquet_hash = hashlib.sha256(parquet_bytes).hexdigest()
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="standardized_data.parquet",
        Body=parquet_bytes
    )
    
    # FILE 2: standardized_data.csv (with 6 decimal precision)
    logger.info("Uploading standardized_data.csv...")
    df_csv = df_consolidated.copy()
    
    # Format price columns with exactly 6 decimals
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df_csv[col] = df_csv[col].map(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    # Convert datetime columns to string
    for col in df_csv.columns:
        if pd.api.types.is_datetime64_any_dtype(df_csv[col]):
            df_csv[col] = df_csv[col].astype(str)
    
    csv_data = df_csv.to_csv(index=False)
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="standardized_data.csv",
        Body=csv_data.encode('utf-8')
    )
    
    # FILE 3: _reports/daily_quality_60.csv
    logger.info("Uploading _reports/daily_quality_60.csv...")
    quality_df = pd.DataFrame(quality_records)
    quality_csv = quality_df.to_csv(index=False, float_format='%.2f')
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="_reports/daily_quality_60.csv",
        Body=quality_csv.encode('utf-8')
    )
    
    # FILE 4: _metadata.json (required fields only)
    logger.info("Uploading _metadata.json...")
    metadata = {
        "dataset_version": "v1.0",
        "run_id": run_id,
        "date_cot": last_date,  # Last date processed
        "utc_window": [utc_start, utc_end],
        "rows": len(df_consolidated),
        "price_unit": "COP per USD",
        "price_precision": 6,
        "source": "twelvedata",
        "created_ts": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "parquet_sha256": parquet_hash  # Optional but recommended
    }
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="_metadata.json",
        Body=json.dumps(metadata, indent=2, cls=NumpyEncoder).encode('utf-8')
    )
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CONSOLIDATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Files uploaded to MinIO bucket '{BUCKET_L1}':")
    logger.info(f"  1. standardized_data.parquet ({len(df_consolidated)} rows × {len(df_consolidated.columns)} columns)")
    logger.info(f"  2. standardized_data.csv (prices with 6 decimals)")
    logger.info(f"  3. _reports/daily_quality_60.csv ({len(quality_records)} days)")
    logger.info(f"  4. _metadata.json")
    logger.info(f"\nDataset specifications:")
    logger.info(f"  - Total rows: {len(df_consolidated):,}")
    logger.info(f"  - Days processed: {len(dates_processed)}")
    logger.info(f"  - Date range: {first_date} to {last_date}")
    logger.info(f"  - Columns (13): {list(df_consolidated.columns)}")
    logger.info(f"  - SHA256: {parquet_hash[:32]}...")
    
    # Quality summary
    ok_days = sum(1 for r in quality_records if r['quality_flag'] == 'OK')
    warn_days = sum(1 for r in quality_records if r['quality_flag'] == 'WARN')
    fail_days = sum(1 for r in quality_records if r['quality_flag'] == 'FAIL')
    
    logger.info(f"\nQuality summary:")
    logger.info(f"  - OK days: {ok_days} ({ok_days/len(quality_records)*100:.1f}%)")
    logger.info(f"  - WARN days: {warn_days} ({warn_days/len(quality_records)*100:.1f}%)")
    logger.info(f"  - FAIL days: {fail_days} ({fail_days/len(quality_records)*100:.1f}%)")
    
    return True

def main():
    """Main execution"""
    logger.info("Starting L1 clean and consolidate process")
    
    s3_client = connect_to_minio()
    
    # Step 1: Clean L1 bucket
    clean_l1_bucket(s3_client)
    
    # Step 2: Process all L0 data
    all_data, dates_processed, quality_records = process_l0_data(s3_client)
    
    if not all_data:
        logger.error("No data found to process!")
        return False
    
    # Step 3: Create and upload consolidated dataset
    success = create_and_upload_consolidated(s3_client, all_data, dates_processed, quality_records)
    
    if success:
        logger.info("\n✅ SUCCESS: L1 consolidation complete with exactly 4 files in MinIO")
    else:
        logger.error("\n❌ FAILED: L1 consolidation failed")
    
    return success

if __name__ == "__main__":
    main()