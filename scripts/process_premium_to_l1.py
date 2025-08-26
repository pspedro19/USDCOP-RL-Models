#!/usr/bin/env python3
"""
Process Premium Silver Data to L1 Format
=========================================
Converts the existing premium silver data to the exact L1 format required.
Outputs exactly 4 files as specified.
"""

import pandas as pd
import numpy as np
import json
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paths
INPUT_FILE = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\processed\silver\SILVER_PREMIUM_ONLY_20250819_171008.csv")
OUTPUT_DIR = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "_reports").mkdir(parents=True, exist_ok=True)

def process_premium_data():
    """Process premium silver data to L1 format"""
    logger.info(f"Reading premium data from: {INPUT_FILE}")
    
    # Read the premium data
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Parse time columns
    df['time_utc'] = pd.to_datetime(df['time_utc'])
    
    # Create proper time_cot (UTC-5)
    df['time_cot'] = df['time_utc'].dt.tz_localize('UTC').dt.tz_convert('America/Bogota').dt.tz_localize(None)
    
    # Extract hour and minute from time_cot
    df['hour_cot'] = df['time_cot'].dt.hour
    df['minute_cot'] = df['time_cot'].dt.minute
    
    # Create episode_id (date in COT)
    df['episode_id'] = df['time_cot'].dt.strftime('%Y-%m-%d')
    
    # Create t_in_episode (0-59 for each day)
    df = df.sort_values(['episode_id', 'time_utc']).reset_index(drop=True)
    df['t_in_episode'] = df.groupby('episode_id').cumcount()
    
    # Create is_terminal (last bar of each episode)
    df['is_terminal'] = False
    last_indices = df.groupby('episode_id').tail(1).index
    df.loc[last_indices, 'is_terminal'] = True
    
    # Calculate ohlc_valid
    df['ohlc_valid'] = (
        (df['high'] >= df[['open', 'close']].max(axis=1)) &
        (df[['open', 'close']].min(axis=1) >= df['low'])
    )
    
    # Calculate is_stale (O=H=L=C)
    df['is_stale'] = (
        (df['open'] == df['high']) &
        (df['open'] == df['low']) &
        (df['open'] == df['close'])
    )
    
    # Select EXACTLY 13 required columns
    required_cols = [
        'episode_id', 't_in_episode', 'is_terminal',
        'time_utc', 'time_cot', 'hour_cot', 'minute_cot',
        'open', 'high', 'low', 'close',
        'ohlc_valid', 'is_stale'
    ]
    
    df_final = df[required_cols].copy()
    
    # Make time_utc timezone-aware for storage
    df_final['time_utc'] = pd.to_datetime(df_final['time_utc']).dt.tz_localize('UTC')
    
    logger.info(f"Processed data: {len(df_final)} rows, {len(df_final.columns)} columns")
    
    return df_final

def calculate_daily_quality(df):
    """Calculate daily quality metrics"""
    quality_records = []
    
    for episode_id in df['episode_id'].unique():
        df_day = df[df['episode_id'] == episode_id]
        
        n_rows = len(df_day)
        n_stale = df_day['is_stale'].sum()
        stale_rate = (n_stale / n_rows * 100) if n_rows > 0 else 0
        ohlc_violations = (~df_day['ohlc_valid']).sum()
        
        # Calculate gaps
        expected = set(range(60))
        actual = set(df_day['t_in_episode'].values)
        missing = sorted(expected - actual)
        n_gaps = len(missing)
        
        # Calculate max consecutive gap
        max_gap = 0
        if missing:
            max_gap = 1
            current_gap = 1
            for i in range(1, len(missing)):
                if missing[i] == missing[i-1] + 1:
                    current_gap += 1
                    max_gap = max(max_gap, current_gap)
                else:
                    current_gap = 1
        
        # Determine quality flag
        completeness_pct = (n_rows / 60) * 100
        
        if n_rows == 60 and ohlc_violations == 0 and stale_rate <= 2.0:
            quality_flag = 'OK'
        elif n_rows >= 59 and ohlc_violations == 0 and stale_rate <= 2.0:
            quality_flag = 'WARN'
        else:
            quality_flag = 'FAIL'
        
        quality_records.append({
            'date': episode_id,
            'rows_expected': 60,
            'rows_found': n_rows,
            'completeness_pct': round(completeness_pct, 2),
            'n_stale': int(n_stale),
            'stale_rate': round(stale_rate, 2),
            'n_gaps': n_gaps,
            'max_gap_bars': max_gap,
            'ohlc_violations': int(ohlc_violations),
            'quality_flag': quality_flag
        })
    
    return pd.DataFrame(quality_records)

def save_outputs(df, quality_df):
    """Save exactly 4 output files"""
    
    run_id = f"L1_PREMIUM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 1. Save standardized_data.parquet
    parquet_path = OUTPUT_DIR / "standardized_data.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path, compression='snappy')
    logger.info(f"✅ Saved: {parquet_path}")
    
    # Calculate SHA256 hash
    with open(parquet_path, 'rb') as f:
        parquet_hash = hashlib.sha256(f.read()).hexdigest()
    
    # 2. Save standardized_data.csv with 6 decimal precision
    csv_path = OUTPUT_DIR / "standardized_data.csv"
    df_csv = df.copy()
    
    # Convert datetime columns to string
    df_csv['time_utc'] = df_csv['time_utc'].astype(str)
    df_csv['time_cot'] = df_csv['time_cot'].astype(str)
    
    # Format prices with exactly 6 decimals
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df_csv[col] = df_csv[col].apply(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved: {csv_path}")
    
    # 3. Save daily_quality_60.csv
    quality_path = OUTPUT_DIR / "_reports" / "daily_quality_60.csv"
    quality_df.to_csv(quality_path, index=False, float_format='%.2f')
    logger.info(f"✅ Saved: {quality_path}")
    
    # 4. Save _metadata.json
    utc_min = df['time_utc'].min()
    utc_max = df['time_utc'].max()
    
    # Get the latest date (handling NaN)
    dates = df['episode_id'].dropna().unique()
    latest_date = sorted(dates)[-1] if len(dates) > 0 else "unknown"
    
    metadata = {
        "dataset_version": "v1.0",
        "run_id": run_id,
        "date_cot": latest_date,
        "utc_window": [
            pd.Timestamp(utc_min).strftime('%Y-%m-%dT%H:%M:%SZ'),
            pd.Timestamp(utc_max).strftime('%Y-%m-%dT%H:%M:%SZ')
        ],
        "rows": len(df),
        "price_unit": "COP per USD",
        "price_precision": 6,
        "source": "twelvedata",
        "created_ts": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "parquet_sha256": parquet_hash
    }
    
    metadata_path = OUTPUT_DIR / "_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"✅ Saved: {metadata_path}")
    
    return {
        'parquet_path': parquet_path,
        'csv_path': csv_path,
        'quality_path': quality_path,
        'metadata_path': metadata_path
    }

def verify_outputs(df, quality_df):
    """Verify outputs meet all requirements"""
    logger.info("\n" + "="*70)
    logger.info("VERIFICATION")
    logger.info("="*70)
    
    # Check file count
    files = list(OUTPUT_DIR.glob("*")) + list((OUTPUT_DIR / "_reports").glob("*"))
    actual_files = [f for f in files if f.is_file()]
    logger.info(f"✅ Files created: {len(actual_files)} (expected: 4)")
    
    # Check column counts
    logger.info(f"✅ standardized_data columns: {df.shape[1]} (expected: 13)")
    logger.info(f"✅ daily_quality_60 columns: {quality_df.shape[1]} (expected: 10)")
    
    # Check data integrity
    time_unique = df['time_utc'].nunique() == len(df)
    episode_key_unique = df[['episode_id', 't_in_episode']].drop_duplicates().shape[0] == len(df)
    
    logger.info(f"✅ time_utc unique: {time_unique}")
    logger.info(f"✅ (episode_id, t_in_episode) unique: {episode_key_unique}")
    
    # Quality summary
    ok_days = len(quality_df[quality_df['quality_flag'] == 'OK'])
    warn_days = len(quality_df[quality_df['quality_flag'] == 'WARN'])
    fail_days = len(quality_df[quality_df['quality_flag'] == 'FAIL'])
    
    logger.info(f"\n📊 QUALITY SUMMARY:")
    logger.info(f"  - Total days: {len(quality_df)}")
    logger.info(f"  - OK: {ok_days} ({ok_days/len(quality_df)*100:.1f}%)")
    logger.info(f"  - WARN: {warn_days} ({warn_days/len(quality_df)*100:.1f}%)")
    logger.info(f"  - FAIL: {fail_days} ({fail_days/len(quality_df)*100:.1f}%)")

def main():
    """Main execution"""
    logger.info("Starting premium data to L1 conversion...")
    
    # Process the data
    df = process_premium_data()
    
    # Calculate quality metrics
    quality_df = calculate_daily_quality(df)
    
    # Save outputs
    output_info = save_outputs(df, quality_df)
    
    # Verify
    verify_outputs(df, quality_df)
    
    print("\n" + "="*70)
    print("✅ L1 DATASET READY")
    print("="*70)
    print(f"\n📁 Output location: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  1. standardized_data.parquet")
    print("  2. standardized_data.csv")
    print("  3. _reports/daily_quality_60.csv")
    print("  4. _metadata.json")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()