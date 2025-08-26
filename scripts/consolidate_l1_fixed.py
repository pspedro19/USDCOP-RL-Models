#!/usr/bin/env python3
"""
Consolidate L1 Pipeline Output - FIXED VERSION
==============================================
Generates EXACTLY 4 files as required:
1. standardized_data.parquet (and .csv)
2. _reports/daily_quality_60.csv
3. _metadata.json

With all required columns and proper fixes.
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
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L1 = "ds-usdcop-standardize"
OUTPUT_DIR = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "_reports").mkdir(parents=True, exist_ok=True)

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

def fix_time_columns(df):
    """Fix time_cot and episode_id issues"""
    df = df.copy()
    
    # Fix time_utc - ensure it's UTC timezone-aware
    if 'time_utc' in df.columns:
        df['time_utc'] = pd.to_datetime(df['time_utc'])
        if df['time_utc'].dt.tz is None:
            df['time_utc'] = df['time_utc'].dt.tz_localize('UTC')
        else:
            df['time_utc'] = df['time_utc'].dt.tz_convert('UTC')
    
    # Fix time_cot - properly convert to COT (America/Bogota)
    if 'time_utc' in df.columns:
        # Convert UTC to COT (UTC-5)
        df['time_cot'] = df['time_utc'].dt.tz_convert('America/Bogota')
        # Keep as timezone-naive in COT for storage
        df['time_cot'] = df['time_cot'].dt.tz_localize(None)
        
        # Recalculate hour_cot and minute_cot from corrected time_cot
        df['hour_cot'] = df['time_cot'].dt.hour
        df['minute_cot'] = df['time_cot'].dt.minute
    
    # Fix episode_id - should be YYYY-MM-DD in COT
    if 'time_cot' in df.columns:
        df['episode_id'] = df['time_cot'].dt.strftime('%Y-%m-%d')
    
    # Fix t_in_episode - recalculate per episode
    if 'episode_id' in df.columns:
        df = df.sort_values(['episode_id', 'time_utc']).reset_index(drop=True)
        df['t_in_episode'] = df.groupby('episode_id').cumcount()
        
        # Fix is_terminal - only last bar of each episode
        df['is_terminal'] = False
        last_indices = df.groupby('episode_id').tail(1).index
        df.loc[last_indices, 'is_terminal'] = True
    
    return df

def calculate_gaps(df_day):
    """Calculate consecutive gaps in a day's data"""
    if len(df_day) == 60:
        return 0, 0
    
    # Create expected t_in_episode range (0-59)
    expected = set(range(60))
    actual = set(df_day['t_in_episode'].values) if 't_in_episode' in df_day.columns else set()
    missing = sorted(expected - actual)
    
    if not missing:
        return 0, 0
    
    # Calculate max consecutive gap
    max_gap = 1
    current_gap = 1
    for i in range(1, len(missing)):
        if missing[i] == missing[i-1] + 1:
            current_gap += 1
            max_gap = max(max_gap, current_gap)
        else:
            current_gap = 1
    
    return len(missing), max_gap

def collect_all_l1_data(s3_client):
    """Collect all processed L1 data from MinIO"""
    logger.info("Collecting all L1 processed data...")
    
    # Find all parquet files in L1
    prefix = "usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date="
    
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_L1, Prefix=prefix)
    
    all_data = []
    dates_processed = []
    quality_records = []
    
    for page in pages:
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            key = obj['Key']
            
            # Process only standardized_data.parquet files
            if 'standardized_data.parquet' in key:
                # Extract date from path
                parts = key.split('/')
                date_part = None
                for part in parts:
                    if part.startswith('date='):
                        date_part = part.replace('date=', '')
                        break
                
                if not date_part:
                    continue
                
                logger.info(f"Reading data for {date_part}...")
                
                try:
                    # Read parquet from S3
                    response = s3_client.get_object(Bucket=BUCKET_L1, Key=key)
                    df = pd.read_parquet(io.BytesIO(response['Body'].read()))
                    
                    # Fix time columns and episode_id
                    df = fix_time_columns(df)
                    
                    # Ensure we have exactly 13 required columns
                    required_cols = [
                        'episode_id', 't_in_episode', 'is_terminal',
                        'time_utc', 'time_cot', 'hour_cot', 'minute_cot',
                        'open', 'high', 'low', 'close',
                        'ohlc_valid', 'is_stale'
                    ]
                    
                    # Check if all required columns exist
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        logger.warning(f"Missing columns for {date_part}: {missing_cols}")
                        # Add missing columns with defaults
                        for col in missing_cols:
                            if col == 'ohlc_valid':
                                # Calculate OHLC validity
                                df['ohlc_valid'] = (
                                    (df['high'] >= df[['open', 'close']].max(axis=1)) &
                                    (df[['open', 'close']].min(axis=1) >= df['low'])
                                )
                            elif col == 'is_stale':
                                # Detect stale bars (O=H=L=C)
                                df['is_stale'] = (
                                    (df['open'] == df['high']) &
                                    (df['open'] == df['low']) &
                                    (df['open'] == df['close'])
                                )
                            else:
                                df[col] = None
                    
                    # Select ONLY the 13 required columns
                    df = df[required_cols].copy()
                    
                    # Add to collection
                    all_data.append(df)
                    dates_processed.append(date_part)
                    
                    # Calculate quality metrics for this day
                    n_rows = len(df)
                    n_stale = df['is_stale'].sum() if 'is_stale' in df.columns else 0
                    stale_rate = (n_stale / n_rows * 100) if n_rows > 0 else 0
                    
                    # Calculate gaps
                    n_gaps, max_gap_bars = calculate_gaps(df)
                    
                    # Check OHLC violations
                    ohlc_violations = (~df['ohlc_valid']).sum() if 'ohlc_valid' in df.columns else 0
                    
                    # Determine quality flag using proper thresholds
                    completeness_pct = (n_rows / 60) * 100
                    
                    # Apply the rules from the requirements
                    if n_rows == 60 and ohlc_violations == 0 and stale_rate <= 2.0:
                        quality_flag = 'OK'
                    elif n_rows >= 59 and ohlc_violations == 0 and stale_rate <= 2.0:
                        quality_flag = 'WARN'
                    else:
                        quality_flag = 'FAIL'
                    
                    # Create quality record with EXACTLY 10 required columns
                    quality_record = {
                        'date': date_part,
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
                    
                    logger.info(f"  - {date_part}: {n_rows} rows, quality={quality_flag}, stale_rate={stale_rate:.2f}%")
                    
                except Exception as e:
                    logger.error(f"Error reading {key}: {e}")
                    continue
    
    return all_data, dates_processed, quality_records

def create_consolidated_dataset(all_data, dates_processed):
    """Create the consolidated dataset with exactly 13 columns"""
    if not all_data:
        logger.error("No data to consolidate!")
        return None
    
    logger.info(f"Consolidating {len(all_data)} days of data...")
    
    # Concatenate all dataframes
    df_consolidated = pd.concat(all_data, ignore_index=True)
    
    # Sort by time_utc
    df_consolidated = df_consolidated.sort_values('time_utc').reset_index(drop=True)
    
    # Verify data integrity
    logger.info("Verifying data integrity...")
    
    # Check unique time_utc
    n_unique_times = df_consolidated['time_utc'].nunique()
    n_total_rows = len(df_consolidated)
    if n_unique_times != n_total_rows:
        logger.warning(f"time_utc not unique! {n_unique_times} unique vs {n_total_rows} total")
    else:
        logger.info(f"✅ time_utc is unique ({n_unique_times} values)")
    
    # Check episode integrity
    episode_key_unique = df_consolidated[['episode_id', 't_in_episode']].drop_duplicates().shape[0] == n_total_rows
    if not episode_key_unique:
        logger.warning("(episode_id, t_in_episode) not unique!")
    else:
        logger.info("✅ (episode_id, t_in_episode) is unique")
    
    # OHLC coherence check
    ohlc_violations = (~df_consolidated['ohlc_valid']).sum()
    logger.info(f"OHLC violations: {ohlc_violations} ({ohlc_violations/n_total_rows*100:.2f}%)")
    
    logger.info(f"Consolidated dataset: {len(df_consolidated)} total rows ({len(dates_processed)} days)")
    
    return df_consolidated

def save_outputs(df_consolidated, quality_records, dates_processed):
    """Save EXACTLY 4 files as required"""
    
    run_id = f"L1_CONSOLIDATED_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Save standardized_data.parquet
    parquet_path = OUTPUT_DIR / "standardized_data.parquet"
    table = pa.Table.from_pandas(df_consolidated)
    pq.write_table(table, parquet_path, compression='snappy')
    logger.info(f"✅ Saved: {parquet_path}")
    
    # Calculate SHA256 hash
    with open(parquet_path, 'rb') as f:
        parquet_hash = hashlib.sha256(f.read()).hexdigest()
    
    # 2. Save standardized_data.csv with 6 decimal precision for prices
    csv_path = OUTPUT_DIR / "standardized_data.csv"
    df_csv = df_consolidated.copy()
    
    # Convert datetime columns to string for CSV
    df_csv['time_utc'] = df_csv['time_utc'].astype(str)
    df_csv['time_cot'] = df_csv['time_cot'].astype(str)
    
    # Format price columns with EXACTLY 6 decimals
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df_csv[col] = df_csv[col].apply(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved: {csv_path}")
    
    # 3. Save _reports/daily_quality_60.csv (EXACTLY 10 columns)
    quality_df = pd.DataFrame(quality_records)
    
    # Ensure we have exactly the 10 required columns in the correct order
    required_quality_cols = [
        'date', 'rows_expected', 'rows_found', 'completeness_pct',
        'n_stale', 'stale_rate', 'n_gaps', 'max_gap_bars',
        'ohlc_violations', 'quality_flag'
    ]
    quality_df = quality_df[required_quality_cols]
    
    quality_path = OUTPUT_DIR / "_reports" / "daily_quality_60.csv"
    quality_df.to_csv(quality_path, index=False, float_format='%.2f')
    logger.info(f"✅ Saved: {quality_path}")
    
    # 4. Save _metadata.json (EXACTLY 9 required fields + optional)
    first_date = min(dates_processed) if dates_processed else None
    last_date = max(dates_processed) if dates_processed else None
    
    # Get UTC window from data
    utc_min = df_consolidated['time_utc'].min()
    utc_max = df_consolidated['time_utc'].max()
    
    # Format UTC window properly
    utc_window = [
        pd.Timestamp(utc_min).strftime('%Y-%m-%dT%H:%M:%SZ'),
        pd.Timestamp(utc_max).strftime('%Y-%m-%dT%H:%M:%SZ')
    ]
    
    # Create metadata with EXACTLY the required 9 keys
    metadata = {
        "dataset_version": "v1.0",
        "run_id": run_id,
        "date_cot": last_date,  # Latest date processed
        "utc_window": utc_window,
        "rows": len(df_consolidated),
        "price_unit": "COP per USD",
        "price_precision": 6,
        "source": "twelvedata",
        "created_ts": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        # Optional but recommended
        "parquet_sha256": parquet_hash,
        # Additional useful info
        "date_range_cot": f"{first_date} to {last_date}",
        "days_processed": len(dates_processed),
        "quality_summary": {
            "total_days": len(quality_records),
            "days_ok": sum(1 for r in quality_records if r['quality_flag'] == 'OK'),
            "days_warn": sum(1 for r in quality_records if r['quality_flag'] == 'WARN'),
            "days_fail": sum(1 for r in quality_records if r['quality_flag'] == 'FAIL')
        }
    }
    
    metadata_path = OUTPUT_DIR / "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved: {metadata_path}")
    
    return {
        'parquet_path': parquet_path,
        'csv_path': csv_path,
        'quality_path': quality_path,
        'metadata_path': metadata_path,
        'total_rows': len(df_consolidated),
        'days_processed': len(dates_processed)
    }

def verify_outputs(output_info, df_consolidated, quality_records):
    """Verify all outputs meet requirements"""
    logger.info("\n" + "="*70)
    logger.info("VERIFYING OUTPUT REQUIREMENTS")
    logger.info("="*70)
    
    errors = []
    
    # 1. Verify file count (exactly 4 files)
    files = list(OUTPUT_DIR.glob("*")) + list((OUTPUT_DIR / "_reports").glob("*"))
    actual_files = [f for f in files if f.is_file()]
    if len(actual_files) != 4:
        errors.append(f"Expected 4 files, found {len(actual_files)}")
    else:
        logger.info("✅ File count: 4 files")
    
    # 2. Verify standardized_data columns (exactly 13)
    if df_consolidated.shape[1] != 13:
        errors.append(f"Expected 13 columns, found {df_consolidated.shape[1]}")
    else:
        logger.info("✅ Column count: 13 columns")
    
    # 3. Verify daily_quality_60 columns (exactly 10)
    quality_df = pd.read_csv(output_info['quality_path'])
    if quality_df.shape[1] != 10:
        errors.append(f"Quality report expected 10 columns, found {quality_df.shape[1]}")
    else:
        logger.info("✅ Quality report: 10 columns")
    
    # 4. Verify metadata has required fields
    with open(output_info['metadata_path'], 'r') as f:
        metadata = json.load(f)
    
    required_metadata_fields = [
        "dataset_version", "run_id", "date_cot", "utc_window",
        "rows", "price_unit", "price_precision", "source", "created_ts"
    ]
    
    missing_fields = [f for f in required_metadata_fields if f not in metadata]
    if missing_fields:
        errors.append(f"Metadata missing fields: {missing_fields}")
    else:
        logger.info("✅ Metadata: All 9 required fields present")
    
    # 5. Verify data integrity
    # Check time_utc uniqueness
    if df_consolidated['time_utc'].nunique() != len(df_consolidated):
        errors.append("time_utc is not unique")
    else:
        logger.info("✅ time_utc is unique")
    
    # Check (episode_id, t_in_episode) uniqueness
    if df_consolidated[['episode_id', 't_in_episode']].drop_duplicates().shape[0] != len(df_consolidated):
        errors.append("(episode_id, t_in_episode) is not unique")
    else:
        logger.info("✅ (episode_id, t_in_episode) is unique")
    
    # 6. Verify CSV price precision
    csv_df = pd.read_csv(output_info['csv_path'])
    sample_price = csv_df['close'].iloc[0] if len(csv_df) > 0 else ""
    if sample_price and '.' in str(sample_price):
        decimals = len(str(sample_price).split('.')[1])
        if decimals != 6:
            errors.append(f"CSV prices should have 6 decimals, found {decimals}")
        else:
            logger.info("✅ CSV prices: 6 decimal precision")
    
    if errors:
        logger.error("\n❌ ERRORS FOUND:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    else:
        logger.info("\n✅ ALL REQUIREMENTS MET!")
        return True

def print_summary(df_consolidated, quality_records, output_info):
    """Print final summary"""
    print("\n" + "="*70)
    print("L1 CONSOLIDATED DATASET - FINAL SUMMARY")
    print("="*70)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  - Total rows: {output_info['total_rows']:,}")
    print(f"  - Days processed: {output_info['days_processed']}")
    print(f"  - Columns: 13 (exactly as required)")
    
    print(f"\n📁 OUTPUT FILES (4 files only):")
    print(f"  1. standardized_data.parquet ({output_info['total_rows']:,} rows, 13 columns)")
    print(f"  2. standardized_data.csv (prices with 6 decimals)")
    print(f"  3. _reports/daily_quality_60.csv ({len(quality_records)} days, 10 columns)")
    print(f"  4. _metadata.json (9 required fields)")
    
    print(f"\n✅ QUALITY SUMMARY:")
    ok_days = sum(1 for r in quality_records if r['quality_flag'] == 'OK')
    warn_days = sum(1 for r in quality_records if r['quality_flag'] == 'WARN')
    fail_days = sum(1 for r in quality_records if r['quality_flag'] == 'FAIL')
    
    print(f"  - OK days: {ok_days} ({ok_days/len(quality_records)*100:.1f}%)")
    print(f"  - WARN days: {warn_days} ({warn_days/len(quality_records)*100:.1f}%)")
    print(f"  - FAIL days: {fail_days} ({fail_days/len(quality_records)*100:.1f}%)")
    
    # Show required columns
    print(f"\n📋 REQUIRED COLUMNS (13):")
    required_cols = [
        'episode_id', 't_in_episode', 'is_terminal',
        'time_utc', 'time_cot', 'hour_cot', 'minute_cot',
        'open', 'high', 'low', 'close',
        'ohlc_valid', 'is_stale'
    ]
    for i, col in enumerate(required_cols, 1):
        print(f"  {i:2}. {col}")
    
    # Show sample of data
    if df_consolidated is not None and len(df_consolidated) > 0:
        print(f"\n📋 SAMPLE DATA (first row):")
        first_row = df_consolidated.iloc[0]
        print(f"  episode_id: {first_row['episode_id']}")
        print(f"  time_utc: {first_row['time_utc']}")
        print(f"  time_cot: {first_row['time_cot']}")
        print(f"  close: {first_row['close']:.6f}")
    
    print("\n" + "="*70)
    print("✅ L1 CONSOLIDATION COMPLETE - Exactly 4 files generated")
    print("="*70)

def main():
    """Main execution"""
    logger.info("Starting L1 dataset consolidation (FIXED VERSION)...")
    
    # Connect to MinIO
    s3_client = connect_to_minio()
    
    # Collect all L1 data
    all_data, dates_processed, quality_records = collect_all_l1_data(s3_client)
    
    if not all_data:
        logger.error("No data found to consolidate!")
        return False
    
    # Create consolidated dataset
    df_consolidated = create_consolidated_dataset(all_data, dates_processed)
    
    if df_consolidated is None:
        logger.error("Failed to create consolidated dataset!")
        return False
    
    # Save all outputs (exactly 4 files)
    output_info = save_outputs(df_consolidated, quality_records, dates_processed)
    
    # Verify outputs meet requirements
    verification_passed = verify_outputs(output_info, df_consolidated, quality_records)
    
    # Print summary
    print_summary(df_consolidated, quality_records, output_info)
    
    if not verification_passed:
        logger.error("⚠️ Some requirements were not met. Please review the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("Consolidation failed!")
        exit(1)