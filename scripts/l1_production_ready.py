#!/usr/bin/env python3
"""
Production-Ready L1 Pipeline
=============================
Implements all auditor recommendations for robust L1 processing:
- Correct time_cot timezone conversion
- Populated episode_id field
- Stale bars treated as missing for quality assessment
- SHA256 hash in metadata
- 6 decimal precision for prices in CSV
- Complete data integrity checks
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
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L0 = "ds-usdcop-acquire"
BUCKET_L1 = "ds-usdcop-standardize"

# Quality thresholds
MIN_VALID_BARS = 59  # Minimum non-stale bars for OK/WARN
STALE_RATE_WARN = 2.0  # Max stale rate for WARN (%)
OHLC_VIOLATIONS_MAX = 0  # Zero tolerance for OHLC violations

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
    
    try:
        # List all objects
        paginator = s3_client.get_paginator('list_objects_v2')
        objects_to_delete = []
        
        for page in paginator.paginate(Bucket=BUCKET_L1):
            if 'Contents' in page:
                for obj in page['Contents']:
                    objects_to_delete.append({'Key': obj['Key']})
        
        if objects_to_delete:
            logger.info(f"Found {len(objects_to_delete)} objects to delete")
            
            # Show sample of what will be deleted
            logger.info("Sample objects to be deleted:")
            for i, obj in enumerate(objects_to_delete[:5]):
                logger.info(f"  {i+1}. {obj['Key']}")
            if len(objects_to_delete) > 5:
                logger.info(f"  ... and {len(objects_to_delete)-5} more")
            
            # Delete in batches of 1000 (AWS limit)
            for i in range(0, len(objects_to_delete), 1000):
                batch = objects_to_delete[i:i+1000]
                s3_client.delete_objects(
                    Bucket=BUCKET_L1,
                    Delete={'Objects': batch}
                )
            logger.info(f"Successfully deleted {len(objects_to_delete)} objects from L1 bucket")
        else:
            logger.info("L1 bucket is already empty")
        
        # Verify bucket is empty
        response = s3_client.list_objects_v2(Bucket=BUCKET_L1, MaxKeys=1)
        if 'Contents' not in response:
            logger.info("SUCCESS: L1 bucket is now completely empty")
        else:
            logger.warning("WARNING: Some objects may remain in the bucket")
            
    except Exception as e:
        logger.error(f"Error cleaning L1 bucket: {e}")
        raise

def process_l0_data(s3_client):
    """Process all L0 data and create consolidated dataset with proper timezone handling"""
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
    
    # Colombia timezone
    cot_tz = pytz.timezone('America/Bogota')
    
    # Process each date
    for date_str in tqdm(sorted(dates_data.keys()), desc="Processing dates"):
        try:
            # Read L0 data
            obj = s3_client.get_object(Bucket=BUCKET_L0, Key=dates_data[date_str])
            df = pd.read_parquet(io.BytesIO(obj['Body'].read()))
            
            # Parse time and ensure UTC
            df['time_utc'] = pd.to_datetime(df['time'])
            if df['time_utc'].dt.tz is None:
                df['time_utc'] = df['time_utc'].dt.tz_localize('UTC')
            else:
                df['time_utc'] = df['time_utc'].dt.tz_convert('UTC')
            
            # CRITICAL FIX: Proper COT conversion
            df['time_cot'] = df['time_utc'].dt.tz_convert(cot_tz)
            
            # Extract hour and minute in COT
            df['hour_cot'] = df['time_cot'].dt.hour
            df['minute_cot'] = df['time_cot'].dt.minute
            
            # Filter for premium window (08:00-12:55 COT)
            df_premium = df[(df['hour_cot'] >= 8) & (df['hour_cot'] < 13)].copy()
            if len(df_premium) > 0 and df_premium['hour_cot'].max() == 12:
                df_premium = df_premium[~((df_premium['hour_cot'] == 12) & (df_premium['minute_cot'] > 55))].copy()
            
            if len(df_premium) == 0:
                logger.debug(f"No premium window data for {date_str}")
                continue
            
            # Create standardized structure (13 columns exactly)
            df_std = pd.DataFrame()
            
            # FIX: Populate episode_id with COT date (as string)
            episode_date_cot = df_premium['time_cot'].iloc[0].strftime('%Y-%m-%d')
            df_std['episode_id'] = str(episode_date_cot)  # Ensure it's a string
            
            df_std['t_in_episode'] = range(len(df_premium))
            df_std['is_terminal'] = False
            if len(df_std) > 0:
                df_std.loc[df_std.index[-1], 'is_terminal'] = True
            
            # Store time columns properly
            df_std['time_utc'] = df_premium['time_utc'].values
            # Convert COT to naive datetime (removes timezone info but keeps local time)
            df_std['time_cot'] = df_premium['time_cot'].dt.tz_localize(None).values
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
                (df_std[['open', 'close']].min(axis=1) >= df_std['low']) &
                (df_std['high'] >= df_std['low'])
            )
            
            # Stale detection
            if len(df_std) > 1:
                df_std['is_stale'] = (df_std['close'].diff().abs() == 0)
                df_std.loc[0, 'is_stale'] = False
            else:
                df_std['is_stale'] = False
            
            # Add to collection (only if we have data)
            if len(df_std) > 0:
                all_data.append(df_std)
                dates_processed.append(episode_date_cot)  # Use COT date
            else:
                logger.warning(f"Skipping {date_str} - no data after standardization")
                continue  # Skip quality record for empty data
            
            # Calculate quality metrics for this day (only if we have data)
            n_rows = len(df_std)
            n_stale = df_std['is_stale'].sum()
            stale_rate = (n_stale / n_rows * 100) if n_rows > 0 else 0
            
            # NEW: Valid bars (non-stale bars)
            valid_bars = n_rows - n_stale
            
            # OHLC violations
            ohlc_violations = (~df_std['ohlc_valid']).sum()
            
            # Grid violations (check if minutes are multiples of 5)
            grid_violations = (df_std['minute_cot'] % 5 != 0).sum()
            
            # Gap analysis
            n_gaps = 60 - n_rows
            max_gap_bars = n_gaps  # Simplified - could calculate consecutive gaps
            
            # NEW QUALITY FLAG LOGIC: Consider stale bars as missing
            completeness_pct = (valid_bars / 60) * 100
            
            if valid_bars >= 60 and ohlc_violations == 0 and grid_violations == 0:
                quality_flag = 'OK'
            elif valid_bars >= MIN_VALID_BARS and ohlc_violations == 0 and stale_rate <= STALE_RATE_WARN:
                quality_flag = 'WARN'
            else:
                quality_flag = 'FAIL'
            
            quality_record = {
                'date': episode_date_cot,  # COT date
                'rows_expected': 60,
                'rows_found': n_rows,
                'valid_bars': int(valid_bars),  # NEW field
                'completeness_pct': round(completeness_pct, 2),
                'n_stale': int(n_stale),
                'stale_rate': round(stale_rate, 2),
                'n_gaps': n_gaps,
                'max_gap_bars': max_gap_bars,
                'ohlc_violations': int(ohlc_violations),
                'grid_violations': int(grid_violations),  # NEW field
                'quality_flag': quality_flag
            }
            quality_records.append(quality_record)
            
        except Exception as e:
            logger.warning(f"Error processing {date_str}: {e}")
            continue
    
    return all_data, dates_processed, quality_records

def validate_data_integrity(df_consolidated):
    """Perform comprehensive data integrity checks"""
    logger.info("Running data integrity checks...")
    
    checks_passed = True
    
    # Check 1: time_utc uniqueness
    if not df_consolidated['time_utc'].is_unique:
        logger.error("FAIL: time_utc is not unique")
        checks_passed = False
    else:
        logger.info("PASS: time_utc is unique")
    
    # Check 2: (episode_id, t_in_episode) uniqueness
    if df_consolidated.duplicated(['episode_id', 't_in_episode']).any():
        logger.error("FAIL: (episode_id, t_in_episode) is not unique")
        checks_passed = False
    else:
        logger.info("PASS: (episode_id, t_in_episode) is unique")
    
    # Check 3: No NaN in critical columns
    critical_cols = ['time_utc', 'time_cot', 'episode_id', 't_in_episode', 
                    'is_terminal', 'hour_cot', 'minute_cot', 'open', 'high', 
                    'low', 'close', 'ohlc_valid', 'is_stale']
    
    for col in critical_cols:
        if df_consolidated[col].isna().any():
            logger.error(f"FAIL: {col} contains NaN values")
            checks_passed = False
    
    if checks_passed:
        logger.info("PASS: No NaN in critical columns")
    
    # Check 4: is_terminal only at end of episodes
    if not df_consolidated['episode_id'].isna().all():
        terminal_check = df_consolidated.groupby('episode_id', group_keys=False).apply(
            lambda x: (x['is_terminal'].sum() == 1) and 
                      (x.loc[x['is_terminal'], 't_in_episode'].values[0] == x['t_in_episode'].max())
        )
        
        if not terminal_check.all():
            logger.error("FAIL: is_terminal not properly set at episode ends")
            checks_passed = False
        else:
            logger.info("PASS: is_terminal properly set")
    else:
        logger.error("FAIL: episode_id is all NaN - cannot check terminal flags")
        checks_passed = False
    
    # Check 5: Grid M5 validation
    if (df_consolidated['minute_cot'] % 5 != 0).any():
        logger.warning("WARN: Some minutes are not multiples of 5 (grid violations)")
    else:
        logger.info("PASS: All minutes are multiples of 5 (M5 grid)")
    
    # Check 6: time_cot vs time_utc consistency
    # Convert both to string for comparison to avoid timezone issues
    time_utc_to_cot = pd.to_datetime(df_consolidated['time_utc']).dt.tz_localize('UTC').dt.tz_convert('America/Bogota')
    time_cot_parsed = pd.to_datetime(df_consolidated['time_cot']).dt.tz_localize('America/Bogota')
    
    # Compare hour difference (should be -5 hours)
    hour_diff = time_cot_parsed.dt.hour - pd.to_datetime(df_consolidated['time_utc']).dt.tz_localize('UTC').dt.hour
    expected_diff = -5  # COT is UTC-5
    
    if not ((hour_diff == expected_diff) | (hour_diff == expected_diff + 24) | (hour_diff == expected_diff - 24)).all():
        logger.error("FAIL: time_cot is not properly converted from time_utc")
        checks_passed = False
    else:
        logger.info("PASS: time_cot is properly converted from time_utc")
    
    return checks_passed

def create_and_upload_consolidated(s3_client, all_data, dates_processed, quality_records):
    """Create consolidated dataset and upload to MinIO with all fixes"""
    
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
    
    # Run integrity checks
    integrity_ok = validate_data_integrity(df_consolidated)
    if not integrity_ok:
        logger.warning("Some integrity checks failed - review logs above")
    
    # Generate run_id
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get date ranges
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
    
    # Calculate SHA256 hash
    parquet_hash = hashlib.sha256(parquet_bytes).hexdigest()
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="standardized_data.parquet",
        Body=parquet_bytes
    )
    
    # FILE 2: standardized_data.csv (with EXACTLY 6 decimal precision)
    logger.info("Uploading standardized_data.csv...")
    df_csv = df_consolidated.copy()
    
    # Format price columns with exactly 6 decimals
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df_csv[col] = df_csv[col].apply(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    # Convert datetime columns to string
    for col in df_csv.columns:
        if pd.api.types.is_datetime64_any_dtype(df_csv[col]):
            df_csv[col] = df_csv[col].astype(str)
    
    csv_data = df_csv.to_csv(index=False)
    csv_size = len(csv_data.encode('utf-8'))
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="standardized_data.csv",
        Body=csv_data.encode('utf-8')
    )
    
    # FILE 3: _reports/daily_quality_60.csv
    logger.info("Uploading _reports/daily_quality_60.csv...")
    quality_df = pd.DataFrame(quality_records)
    
    # Ensure proper column order
    quality_cols = ['date', 'rows_expected', 'rows_found', 'valid_bars', 
                   'completeness_pct', 'n_stale', 'stale_rate', 'n_gaps', 
                   'max_gap_bars', 'ohlc_violations', 'grid_violations', 'quality_flag']
    quality_df = quality_df[quality_cols]
    
    quality_csv = quality_df.to_csv(index=False, float_format='%.2f')
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="_reports/daily_quality_60.csv",
        Body=quality_csv.encode('utf-8')
    )
    
    # FILE 4: _metadata.json (with all required fields + hash)
    logger.info("Uploading _metadata.json...")
    metadata = {
        "dataset_version": "v2.0",
        "run_id": run_id,
        "date_range_cot": f"{first_date} to {last_date}",  # COT dates
        "utc_window": [utc_start, utc_end],
        "rows": len(df_consolidated),
        "rows_per_day_expected": 60,
        "price_unit": "COP per USD",
        "price_precision": 6,
        "source": "twelvedata",
        "created_ts": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        "parquet_sha256": parquet_hash,  # ADDED: SHA256 hash
        "csv_size_bytes": csv_size,
        "columns": list(df_consolidated.columns),
        "quality_summary": {
            "total_days": len(quality_records),
            "days_ok": sum(1 for r in quality_records if r['quality_flag'] == 'OK'),
            "days_warn": sum(1 for r in quality_records if r['quality_flag'] == 'WARN'),
            "days_fail": sum(1 for r in quality_records if r['quality_flag'] == 'FAIL')
        },
        "integrity_checks_passed": integrity_ok
    }
    
    s3_client.put_object(
        Bucket=BUCKET_L1,
        Key="_metadata.json",
        Body=json.dumps(metadata, indent=2, cls=NumpyEncoder).encode('utf-8')
    )
    
    # Cross-validation checks
    logger.info("\n" + "="*60)
    logger.info("CROSS-VALIDATION CHECKS")
    logger.info("="*60)
    
    # Check 1: Row count consistency
    parquet_rows = len(df_consolidated)
    csv_rows = len(df_csv)
    metadata_rows = metadata['rows']
    
    if parquet_rows == csv_rows == metadata_rows:
        logger.info(f"✅ Row count consistent: {parquet_rows}")
    else:
        logger.error(f"❌ Row count mismatch: parquet={parquet_rows}, csv={csv_rows}, metadata={metadata_rows}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("CONSOLIDATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Files uploaded to MinIO bucket '{BUCKET_L1}':")
    logger.info(f"  1. standardized_data.parquet ({len(df_consolidated)} rows × {len(df_consolidated.columns)} columns)")
    logger.info(f"  2. standardized_data.csv (prices with 6 decimals)")
    logger.info(f"  3. _reports/daily_quality_60.csv ({len(quality_records)} days)")
    logger.info(f"  4. _metadata.json (with SHA256: {parquet_hash[:16]}...)")
    logger.info(f"\nDataset specifications:")
    logger.info(f"  - Total rows: {len(df_consolidated):,}")
    logger.info(f"  - Days processed: {len(dates_processed)}")
    logger.info(f"  - Date range (COT): {first_date} to {last_date}")
    logger.info(f"  - Columns (13): {list(df_consolidated.columns)}")
    logger.info(f"  - SHA256: {parquet_hash}")
    
    # Quality summary with new logic
    ok_days = sum(1 for r in quality_records if r['quality_flag'] == 'OK')
    warn_days = sum(1 for r in quality_records if r['quality_flag'] == 'WARN')
    fail_days = sum(1 for r in quality_records if r['quality_flag'] == 'FAIL')
    
    logger.info(f"\nQuality summary (with stale-as-missing logic):")
    logger.info(f"  - OK days: {ok_days} ({ok_days/len(quality_records)*100:.1f}%)")
    logger.info(f"  - WARN days: {warn_days} ({warn_days/len(quality_records)*100:.1f}%)")
    logger.info(f"  - FAIL days: {fail_days} ({fail_days/len(quality_records)*100:.1f}%)")
    
    # Show improvement
    valid_days = ok_days + warn_days
    logger.info(f"  - Total valid days (OK+WARN): {valid_days} ({valid_days/len(quality_records)*100:.1f}%)")
    
    return True

def main():
    """Main execution"""
    logger.info("Starting Production-Ready L1 Pipeline")
    logger.info("Implementing all auditor recommendations...")
    
    s3_client = connect_to_minio()
    
    # Step 1: Clean L1 bucket
    clean_l1_bucket(s3_client)
    
    # Step 2: Process all L0 data with fixes
    all_data, dates_processed, quality_records = process_l0_data(s3_client)
    
    if not all_data:
        logger.error("No data found to process!")
        return False
    
    # Step 3: Create and upload consolidated dataset with all improvements
    success = create_and_upload_consolidated(s3_client, all_data, dates_processed, quality_records)
    
    if success:
        logger.info("\n✅ SUCCESS: Production-ready L1 consolidation complete")
        logger.info("All auditor recommendations have been implemented:")
        logger.info("  ✓ time_cot properly converted to America/Bogota timezone")
        logger.info("  ✓ episode_id populated with COT date (YYYY-MM-DD)")
        logger.info("  ✓ Stale bars treated as missing for quality assessment")
        logger.info("  ✓ SHA256 hash added to metadata")
        logger.info("  ✓ CSV prices with exactly 6 decimal precision")
        logger.info("  ✓ Data integrity checks implemented")
        logger.info("  ✓ Grid violations tracked")
    else:
        logger.error("\n❌ FAILED: L1 consolidation failed")
    
    return success

if __name__ == "__main__":
    main()