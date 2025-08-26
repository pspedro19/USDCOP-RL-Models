#!/usr/bin/env python3
"""
Run the audit-ready L1 pipeline locally and upload to MinIO
===============================================
This script runs all audit features and uploads the 5 required files to MinIO
"""

import pandas as pd
import numpy as np
import json
import hashlib
import boto3
from botocore.client import Config
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MinIO configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'airflow'
MINIO_SECRET_KEY = 'airflow'
BUCKET = "ds-usdcop-standardize"

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

def apply_mode_b_padding(df):
    """Apply Mode B: Pad single missing slots with placeholder rows"""
    logger.info("Applying Mode B padding for single missing slots...")
    
    padded_episodes = 0
    all_data = []
    
    # Filter out None/NaN episode_ids
    valid_episodes = df['episode_id'].dropna().unique()
    for episode_id in sorted(valid_episodes):
        df_day = df[df['episode_id'] == episode_id].copy()
        
        if len(df_day) == 59:
            # Single missing slot - apply Mode B
            logger.info(f"  Mode B: Padding {episode_id} (59 bars -> 60)")
            
            # Create time grid for the day
            start_time = pd.Timestamp(episode_id + ' 08:00:00').tz_localize('America/Bogota')
            time_grid = pd.date_range(start=start_time, periods=60, freq='5min')
            
            # Find missing slot
            df_day['time_cot_ts'] = pd.to_datetime(df_day['time_cot'])
            existing_times = set(df_day['time_cot_ts'].dt.floor('5min'))
            all_times = set(time_grid.tz_localize(None))
            missing_times = all_times - existing_times
            
            if len(missing_times) == 1:
                missing_time = list(missing_times)[0]
                
                # Create placeholder row
                placeholder = pd.DataFrame({
                    'episode_id': [episode_id],
                    't_in_episode': [None],
                    'is_terminal': [False],
                    'time_utc': [missing_time.tz_localize('America/Bogota').tz_convert('UTC')],
                    'time_cot': [missing_time],
                    'hour_cot': [missing_time.hour],
                    'minute_cot': [missing_time.minute],
                    'open': [np.nan],
                    'high': [np.nan],
                    'low': [np.nan],
                    'close': [np.nan],
                    'ohlc_valid': [False],
                    'is_stale': [False],
                    'is_missing': [True]
                })
                
                # Combine and sort
                df_day = pd.concat([df_day, placeholder], ignore_index=True)
                df_day = df_day.sort_values('time_cot').reset_index(drop=True)
                df_day['t_in_episode'] = range(60)
                df_day.loc[59, 'is_terminal'] = True
                
                padded_episodes += 1
            
            # Clean up temp column
            if 'time_cot_ts' in df_day.columns:
                df_day = df_day.drop('time_cot_ts', axis=1)
        
        # Add is_missing column if not exists
        if 'is_missing' not in df_day.columns:
            df_day['is_missing'] = False
        
        all_data.append(df_day)
    
    logger.info(f"Mode B padding complete: {padded_episodes} episodes padded")
    return pd.concat(all_data, ignore_index=True)

def calculate_enhanced_quality(df):
    """Calculate enhanced quality metrics with fail reasons and stale burst"""
    logger.info("Calculating enhanced quality metrics...")
    
    quality_rows = []
    
    # Filter out None/NaN episode_ids
    valid_episodes = df['episode_id'].dropna().unique()
    for episode_id in sorted(valid_episodes):
        df_day = df[df['episode_id'] == episode_id]
        
        # Basic metrics
        n_rows = len(df_day)
        n_missing = df_day['is_missing'].sum() if 'is_missing' in df_day.columns else 0
        n_actual = n_rows - n_missing
        n_stale = df_day['is_stale'].sum() if n_actual > 0 else 0
        stale_rate = (n_stale / n_actual * 100) if n_actual > 0 else 0
        
        # OHLC violations
        if 'is_missing' in df_day.columns:
            # Handle NaN values
            is_missing_mask = df_day['is_missing'].fillna(False).astype(bool)
            ohlc_valid_mask = df_day['ohlc_valid'].fillna(True).astype(bool)
            ohlc_violations = (~ohlc_valid_mask & ~is_missing_mask).sum()
        else:
            ohlc_violations = (~df_day['ohlc_valid'].fillna(True)).sum()
        
        # Stale burst detection
        stale_burst_max = 0
        if n_stale > 0:
            current_burst = 0
            for is_stale in df_day['is_stale'].values:
                if is_stale:
                    current_burst += 1
                    stale_burst_max = max(stale_burst_max, current_burst)
                else:
                    current_burst = 0
        
        # Determine quality flag and fail reason
        if n_actual < 59:
            quality_flag = 'FAIL'
            if n_actual == 0:
                fail_reason = 'NO_DATA'
            elif n_actual < 30:
                fail_reason = 'INSUFFICIENT_BARS_SEVERE'
            else:
                fail_reason = 'INSUFFICIENT_BARS'
        elif stale_rate > 2:
            quality_flag = 'FAIL'
            if stale_rate > 10:
                fail_reason = 'HIGH_STALE_RATE_SEVERE'
            else:
                fail_reason = 'HIGH_STALE_RATE'
        elif stale_rate > 1:
            quality_flag = 'WARN'
            fail_reason = 'MODERATE_STALE_RATE'
        else:
            quality_flag = 'OK'
            fail_reason = 'PASS'
        
        # Handle Mode B padded episodes
        if n_missing == 1 and n_rows == 60:
            fail_reason = 'SINGLE_MISSING_PADDED'
            if quality_flag == 'FAIL':
                quality_flag = 'WARN'  # Downgrade from FAIL to WARN
        
        quality_rows.append({
            'date': episode_id,
            'rows_expected': 60,
            'rows_found': n_actual,
            'rows_padded': n_missing,
            'completeness_pct': n_actual / 60 * 100,
            'n_stale': int(n_stale),
            'stale_rate': round(stale_rate, 2),
            'stale_burst_max': int(stale_burst_max),
            'n_gaps': 60 - n_actual,
            'max_gap_bars': 60 - n_actual,
            'ohlc_violations': int(ohlc_violations),
            'quality_flag': quality_flag,
            'fail_reason': fail_reason
        })
    
    return pd.DataFrame(quality_rows)

def main():
    """Run the complete audit pipeline"""
    logger.info("=" * 70)
    logger.info("AUDIT-READY L1 PIPELINE")
    logger.info("=" * 70)
    
    # Load existing L1 data
    base = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")
    df = pd.read_parquet(base / "standardized_data.parquet")
    logger.info(f"Loaded {len(df):,} rows from existing L1 data")
    
    # Apply Mode B padding
    df_padded = apply_mode_b_padding(df)
    logger.info(f"After padding: {len(df_padded):,} rows")
    
    # Calculate enhanced quality metrics
    quality_df = calculate_enhanced_quality(df_padded)
    logger.info(f"Quality report generated for {len(quality_df)} days")
    
    # Quality summary
    ok_count = (quality_df['quality_flag'] == 'OK').sum()
    warn_count = (quality_df['quality_flag'] == 'WARN').sum()
    fail_count = (quality_df['quality_flag'] == 'FAIL').sum()
    
    logger.info(f"Quality summary: OK={ok_count}, WARN={warn_count}, FAIL={fail_count}")
    
    # Create clean subset (OK and WARN only)
    ok_warn_episodes = quality_df[quality_df['quality_flag'].isin(['OK', 'WARN'])]['date'].values
    df_clean = df_padded[df_padded['episode_id'].isin(ok_warn_episodes)]
    logger.info(f"Clean subset: {len(df_clean):,} rows ({len(ok_warn_episodes)} episodes)")
    
    # Generate metadata with SHA256 hash
    data_str = df_padded.to_json(orient='records', date_format='iso')
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()
    
    metadata = {
        "dataset_version": "v1.0-audit",
        "run_id": "audit_test_20250821",
        "date_cot": pd.Timestamp.now(tz='America/Bogota').strftime('%Y-%m-%d'),
        "utc_window": f"{df_padded['time_utc'].min()} to {df_padded['time_utc'].max()}",
        "rows": len(df_padded),
        "rows_clean_subset": len(df_clean),
        "price_unit": "COP",
        "price_precision": 6,
        "source": "Premium consolidated with Mode B padding",
        "created_ts": pd.Timestamp.now().isoformat(),
        "data_hash": data_hash,
        "mode_b_padded_episodes": len(df_padded[df_padded['is_missing'] == True]['episode_id'].unique()) if 'is_missing' in df_padded.columns else 0,
        "quality_summary": {
            "ok": int(ok_count),
            "warn": int(warn_count),
            "fail": int(fail_count)
        }
    }
    
    # Save files locally
    output_dir = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_audit")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "_reports").mkdir(exist_ok=True)
    
    logger.info("\nSaving audit files locally...")
    
    # 1. Main parquet file
    df_padded.to_parquet(output_dir / "standardized_data.parquet", index=False)
    logger.info(f"  ✓ standardized_data.parquet ({len(df_padded):,} rows)")
    
    # 2. Main CSV file
    df_csv = df_padded.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].round(6)
    df_csv.to_csv(output_dir / "standardized_data.csv", index=False)
    logger.info(f"  ✓ standardized_data.csv")
    
    # 3. Enhanced quality report
    quality_df.to_csv(output_dir / "_reports" / "daily_quality_60.csv", index=False)
    logger.info(f"  ✓ _reports/daily_quality_60.csv ({len(quality_df)} days)")
    
    # 4. Metadata with hash
    with open(output_dir / "_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  ✓ _metadata.json (with SHA256 hash)")
    
    # 5. Clean subset (NEW FILE)
    df_clean.to_parquet(output_dir / "standardized_data_OK_WARNS.parquet", index=False)
    logger.info(f"  ✓ standardized_data_OK_WARNS.parquet ({len(df_clean):,} rows)")
    
    # Upload to MinIO
    logger.info("\nUploading to MinIO...")
    s3_client = connect_to_minio()
    
    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=BUCKET)
    except:
        s3_client.create_bucket(Bucket=BUCKET)
        logger.info(f"Created bucket {BUCKET}")
    
    # Upload files
    prefix = "usdcop_m5__02_l1_standardize_audit/consolidated/"
    
    files_to_upload = [
        ("standardized_data.parquet", f"{prefix}standardized_data.parquet"),
        ("standardized_data.csv", f"{prefix}standardized_data.csv"),
        ("_reports/daily_quality_60.csv", f"{prefix}_reports/daily_quality_60.csv"),
        ("_metadata.json", f"{prefix}_metadata.json"),
        ("standardized_data_OK_WARNS.parquet", f"{prefix}standardized_data_OK_WARNS.parquet"),
    ]
    
    for local_file, s3_key in files_to_upload:
        local_path = output_dir / local_file
        with open(local_path, 'rb') as f:
            s3_client.put_object(Bucket=BUCKET, Key=s3_key, Body=f)
        logger.info(f"  ✓ Uploaded to s3://{BUCKET}/{s3_key}")
    
    logger.info("\n" + "=" * 70)
    logger.info("AUDIT PIPELINE COMPLETE")
    logger.info(f"All 5 files saved and uploaded successfully")
    logger.info("=" * 70)
    
    # Print fail reason breakdown
    logger.info("\nFail Reason Breakdown:")
    for reason, count in quality_df['fail_reason'].value_counts().items():
        logger.info(f"  {reason}: {count} days")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)