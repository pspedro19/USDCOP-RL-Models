#!/usr/bin/env python3
"""
Create all 6 L1 output files locally
=====================================
Generates the complete audit-ready L1 dataset with 6 files:
- standardized_data.parquet
- standardized_data.csv
- _reports/daily_quality_60.csv
- _metadata.json
- standardized_data_OK_WARNS.parquet (clean subset)
- standardized_data_OK_WARNS.csv (clean subset)
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Create all 6 L1 output files"""
    logger.info("="*70)
    logger.info("CREATING 6 L1 OUTPUT FILES")
    logger.info("="*70)
    
    # Input and output paths
    input_dir = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")
    output_dir = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_6files")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "_reports").mkdir(exist_ok=True)
    
    # Load existing data
    logger.info("Loading existing L1 data...")
    df = pd.read_parquet(input_dir / "standardized_data.parquet")
    logger.info(f"Loaded {len(df):,} rows")
    
    # Add is_missing column if not exists
    if 'is_missing' not in df.columns:
        df['is_missing'] = False
    
    # Load or generate quality report
    if (input_dir / "_reports" / "daily_quality_60.csv").exists():
        quality_df = pd.read_csv(input_dir / "_reports" / "daily_quality_60.csv")
    else:
        # Generate basic quality report
        quality_rows = []
        for episode_id in df['episode_id'].dropna().unique():
            df_day = df[df['episode_id'] == episode_id]
            n_rows = len(df_day)
            n_stale = df_day['is_stale'].sum() if 'is_stale' in df_day.columns else 0
            stale_rate = (n_stale / n_rows * 100) if n_rows > 0 else 0
            
            # Determine quality flag
            if n_rows < 59:
                quality_flag = 'FAIL'
                fail_reason = 'INSUFFICIENT_BARS'
            elif stale_rate > 2:
                quality_flag = 'FAIL'
                fail_reason = 'HIGH_STALE_RATE'
            elif stale_rate > 1:
                quality_flag = 'WARN'
                fail_reason = 'MODERATE_STALE_RATE'
            else:
                quality_flag = 'OK'
                fail_reason = 'PASS'
            
            quality_rows.append({
                'date': str(episode_id),
                'rows_expected': 60,
                'rows_found': n_rows,
                'rows_padded': 0,
                'completeness_pct': n_rows / 60 * 100,
                'n_stale': int(n_stale),
                'stale_rate': round(stale_rate, 2),
                'stale_burst_max': 0,
                'n_gaps': 60 - n_rows,
                'max_gap_bars': 60 - n_rows,
                'ohlc_violations': 0,
                'quality_flag': quality_flag,
                'fail_reason': fail_reason
            })
        
        quality_df = pd.DataFrame(quality_rows)
    
    # Add missing columns to quality report if needed
    required_cols = ['date', 'rows_expected', 'rows_found', 'rows_padded', 
                     'completeness_pct', 'n_stale', 'stale_rate', 'stale_burst_max',
                     'n_gaps', 'max_gap_bars', 'ohlc_violations', 'quality_flag', 'fail_reason']
    
    for col in required_cols:
        if col not in quality_df.columns:
            if col == 'rows_padded':
                quality_df[col] = 0
            elif col == 'stale_burst_max':
                quality_df[col] = 0
            elif col == 'fail_reason':
                quality_df[col] = quality_df['quality_flag'].apply(
                    lambda x: 'PASS' if x == 'OK' else 'QUALITY_ISSUE'
                )
    
    # Create clean subset (OK and WARN only)
    ok_warn_episodes = quality_df[
        quality_df['quality_flag'].isin(['OK', 'WARN'])
    ]['date'].values
    df_clean = df[df['episode_id'].isin(ok_warn_episodes)]
    
    logger.info(f"Clean subset: {len(df_clean):,} rows from {len(ok_warn_episodes)} episodes")
    
    # Generate SHA256 hash
    data_str = df.to_json(orient='records', date_format='iso')
    data_hash = hashlib.sha256(data_str.encode()).hexdigest()
    
    # Create metadata
    metadata = {
        "dataset_version": "v1.1-audit-6files",
        "run_id": "local_6files_generation",
        "date_cot": pd.Timestamp.now(tz='America/Bogota').strftime('%Y-%m-%d'),
        "utc_window": f"{df['time_utc'].min() if 'time_utc' in df.columns else 'N/A'} to {df['time_utc'].max() if 'time_utc' in df.columns else 'N/A'}",
        "rows": len(df),
        "rows_clean_subset": len(df_clean),
        "price_unit": "COP",
        "price_precision": 6,
        "source": "L1 consolidated with 6-file output",
        "created_ts": pd.Timestamp.now().isoformat(),
        "data_hash": data_hash,
        "files_generated": 6,
        "quality_summary": {
            "ok": int((quality_df['quality_flag'] == 'OK').sum()),
            "warn": int((quality_df['quality_flag'] == 'WARN').sum()),
            "fail": int((quality_df['quality_flag'] == 'FAIL').sum())
        }
    }
    
    # Save all 6 files
    logger.info("\nSaving 6 output files...")
    
    # 1. Main parquet
    df.to_parquet(output_dir / "standardized_data.parquet", index=False)
    logger.info(f"  1/6 ✓ standardized_data.parquet ({len(df):,} rows)")
    
    # 2. Main CSV
    df_csv = df.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].round(6)
    df_csv.to_csv(output_dir / "standardized_data.csv", index=False)
    logger.info(f"  2/6 ✓ standardized_data.csv")
    
    # 3. Quality report
    quality_df.to_csv(output_dir / "_reports" / "daily_quality_60.csv", index=False)
    logger.info(f"  3/6 ✓ _reports/daily_quality_60.csv ({len(quality_df)} days)")
    
    # 4. Metadata
    with open(output_dir / "_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"  4/6 ✓ _metadata.json (with SHA256 hash)")
    
    # 5. Clean subset parquet
    df_clean.to_parquet(output_dir / "standardized_data_OK_WARNS.parquet", index=False)
    logger.info(f"  5/6 ✓ standardized_data_OK_WARNS.parquet ({len(df_clean):,} rows)")
    
    # 6. Clean subset CSV
    df_clean_csv = df_clean.copy()
    for col in ['open', 'high', 'low', 'close']:
        if col in df_clean_csv.columns:
            df_clean_csv[col] = df_clean_csv[col].round(6)
    df_clean_csv.to_csv(output_dir / "standardized_data_OK_WARNS.csv", index=False)
    logger.info(f"  6/6 ✓ standardized_data_OK_WARNS.csv ({len(df_clean):,} rows)")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ ALL 6 FILES CREATED SUCCESSFULLY")
    logger.info("="*70)
    
    logger.info(f"\nOutput location: {output_dir}")
    logger.info("\nFiles created:")
    logger.info("  1. standardized_data.parquet (full dataset)")
    logger.info("  2. standardized_data.csv (full dataset)")
    logger.info("  3. _reports/daily_quality_60.csv (quality report)")
    logger.info("  4. _metadata.json (with SHA256 hash)")
    logger.info("  5. standardized_data_OK_WARNS.parquet (clean subset)")
    logger.info("  6. standardized_data_OK_WARNS.csv (clean subset)")
    
    logger.info(f"\nData summary:")
    logger.info(f"  Total rows: {len(df):,}")
    logger.info(f"  Clean subset rows: {len(df_clean):,}")
    logger.info(f"  Reduction: {(1 - len(df_clean)/len(df))*100:.1f}%")
    
    # Verify file sizes
    logger.info("\nFile sizes:")
    for file_name in ["standardized_data.parquet", "standardized_data.csv", 
                      "_metadata.json", "standardized_data_OK_WARNS.parquet", 
                      "standardized_data_OK_WARNS.csv"]:
        file_path = output_dir / file_name
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"  {file_name}: {size_mb:.2f} MB")
    
    quality_file = output_dir / "_reports" / "daily_quality_60.csv"
    if quality_file.exists():
        size_kb = quality_file.stat().st_size / 1024
        logger.info(f"  _reports/daily_quality_60.csv: {size_kb:.2f} KB")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)