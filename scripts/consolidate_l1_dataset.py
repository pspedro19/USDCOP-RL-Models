#!/usr/bin/env python3
"""
Consolidate L1 Pipeline Output into Final Dataset
==================================================
Genera el dataset consolidado completo con TODOS los días procesados
según especificación estricta del auditor.

Output:
1. standardized_data.parquet/csv - Dataset completo (60 filas × N días)
2. daily_quality_60.csv - Una fila por día con KPIs
3. _metadata.json - Metadata global del dataset
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
BUCKET_L1 = "ds-usdcop-standardize"
OUTPUT_DIR = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
                    
                    # Ensure required columns exist
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
                        continue
                    
                    # Select only required columns (13 columns)
                    df = df[required_cols].copy()
                    
                    # Add to collection
                    all_data.append(df)
                    dates_processed.append(date_part)
                    
                    # Calculate quality metrics for this day
                    n_rows = len(df)
                    n_stale = df['is_stale'].sum() if 'is_stale' in df.columns else 0
                    stale_rate = (n_stale / n_rows * 100) if n_rows > 0 else 0
                    
                    # Check OHLC violations
                    ohlc_violations = 0
                    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                        violations = (
                            (df['high'] < df['open']) | 
                            (df['high'] < df['close']) |
                            (df['low'] > df['open']) |
                            (df['low'] > df['close']) |
                            (df['high'] < df['low'])
                        ).sum()
                        ohlc_violations = int(violations)
                    
                    # Determine quality flag
                    completeness_pct = (n_rows / 60) * 100
                    if n_rows == 60 and ohlc_violations == 0 and stale_rate <= 1:
                        quality_flag = 'OK'
                    elif n_rows >= 59 and ohlc_violations == 0 and stale_rate <= 2:
                        quality_flag = 'WARN'
                    else:
                        quality_flag = 'FAIL'
                    
                    # Gap analysis
                    n_gaps = 60 - n_rows
                    max_gap_bars = n_gaps  # Simplified - should calculate consecutive gaps
                    
                    quality_record = {
                        'date': date_part,
                        'rows_expected': 60,
                        'rows_found': n_rows,
                        'completeness_pct': round(completeness_pct, 2),
                        'n_stale': int(n_stale),
                        'stale_rate': round(stale_rate, 2),
                        'n_gaps': n_gaps,
                        'max_gap_bars': max_gap_bars,
                        'ohlc_violations': ohlc_violations,
                        'quality_flag': quality_flag
                    }
                    quality_records.append(quality_record)
                    
                    logger.info(f"  - {date_part}: {n_rows} rows, quality={quality_flag}")
                    
                except Exception as e:
                    logger.error(f"Error reading {key}: {e}")
                    continue
    
    return all_data, dates_processed, quality_records

def create_consolidated_dataset(all_data, dates_processed):
    """Create the consolidated dataset"""
    if not all_data:
        logger.error("No data to consolidate!")
        return None
    
    logger.info(f"Consolidating {len(all_data)} days of data...")
    
    # Concatenate all dataframes
    df_consolidated = pd.concat(all_data, ignore_index=True)
    
    # Sort by time_utc (ensure consistent timezone handling)
    if 'time_utc' in df_consolidated.columns:
        # Convert time_utc to string first to avoid timezone comparison issues
        df_consolidated['time_utc_str'] = df_consolidated['time_utc'].astype(str)
        df_consolidated = df_consolidated.sort_values('time_utc_str').reset_index(drop=True)
        df_consolidated = df_consolidated.drop('time_utc_str', axis=1)
    
    # Verify data integrity
    logger.info("Verifying data integrity...")
    
    # Check unique time_utc
    if 'time_utc' in df_consolidated.columns:
        n_unique_times = df_consolidated['time_utc'].nunique()
        n_total_rows = len(df_consolidated)
        if n_unique_times != n_total_rows:
            logger.warning(f"time_utc not unique! {n_unique_times} unique vs {n_total_rows} total")
    
    # Check episode integrity
    if 'episode_id' in df_consolidated.columns and 't_in_episode' in df_consolidated.columns:
        episode_key_unique = df_consolidated[['episode_id', 't_in_episode']].drop_duplicates().shape[0] == n_total_rows
        if not episode_key_unique:
            logger.warning("(episode_id, t_in_episode) not unique!")
    
    # OHLC coherence check
    if all(col in df_consolidated.columns for col in ['open', 'high', 'low', 'close']):
        ohlc_coherent = (
            (df_consolidated['high'] >= df_consolidated[['open', 'close']].max(axis=1)) &
            (df_consolidated[['open', 'close']].min(axis=1) >= df_consolidated['low'])
        ).all()
        logger.info(f"OHLC coherence: {'PASS' if ohlc_coherent else 'FAIL'}")
    
    logger.info(f"Consolidated dataset: {len(df_consolidated)} total rows ({len(dates_processed)} days)")
    
    return df_consolidated

def save_outputs(df_consolidated, quality_records, dates_processed):
    """Save all outputs according to specification"""
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
    
    # Format price columns with 6 decimals
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df_csv.columns:
            df_csv[col] = df_csv[col].map(lambda x: f'{x:.6f}' if pd.notna(x) else '')
    
    # Convert datetime columns to string for CSV
    for col in df_csv.columns:
        if pd.api.types.is_datetime64_any_dtype(df_csv[col]):
            df_csv[col] = df_csv[col].astype(str)
    
    df_csv.to_csv(csv_path, index=False)
    logger.info(f"✅ Saved: {csv_path}")
    
    # 3. Save daily_quality_60.csv
    quality_df = pd.DataFrame(quality_records)
    quality_path = OUTPUT_DIR / "_reports" / "daily_quality_60.csv"
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(quality_path, index=False, float_format='%.2f')
    logger.info(f"✅ Saved: {quality_path}")
    
    # 4. Save _metadata.json
    first_date = min(dates_processed) if dates_processed else None
    last_date = max(dates_processed) if dates_processed else None
    
    # Get UTC window from data
    if 'time_utc' in df_consolidated.columns and len(df_consolidated) > 0:
        try:
            # Convert to string first to avoid timezone issues
            time_utc_str = df_consolidated['time_utc'].astype(str)
            utc_start_str = time_utc_str.min()
            utc_end_str = time_utc_str.max()
            utc_window = [utc_start_str, utc_end_str]
        except:
            utc_window = ["Unknown", "Unknown"]
    else:
        utc_window = ["Unknown", "Unknown"]
    
    metadata = {
        "dataset_version": "v2.0",
        "run_id": run_id,
        "consolidation_date": datetime.now().strftime('%Y-%m-%d'),
        "date_range": f"{first_date} to {last_date}",
        "days_processed": len(dates_processed),
        "utc_window": utc_window,
        "rows": len(df_consolidated),
        "rows_per_day_expected": 60,
        "price_unit": "COP per USD",
        "price_precision": 6,
        "source": "twelvedata",
        "created_ts": datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        "parquet_sha256": parquet_hash,
        "columns": list(df_consolidated.columns),
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

def print_summary(df_consolidated, quality_records, output_info):
    """Print final summary"""
    print("\n" + "="*70)
    print("L1 CONSOLIDATED DATASET - FINAL SUMMARY")
    print("="*70)
    
    print(f"\n📊 DATASET STATISTICS:")
    print(f"  - Total rows: {output_info['total_rows']:,}")
    print(f"  - Days processed: {output_info['days_processed']}")
    print(f"  - Expected rows: {output_info['days_processed'] * 60:,}")
    print(f"  - Completeness: {(output_info['total_rows'] / (output_info['days_processed'] * 60) * 100):.2f}%")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  1. {output_info['parquet_path'].name} ({output_info['total_rows']:,} rows, 13 columns)")
    print(f"  2. {output_info['csv_path'].name} (prices with 6 decimals)")
    print(f"  3. {output_info['quality_path'].name} ({len(quality_records)} days)")
    print(f"  4. {output_info['metadata_path'].name}")
    
    print(f"\n✅ QUALITY SUMMARY:")
    ok_days = sum(1 for r in quality_records if r['quality_flag'] == 'OK')
    warn_days = sum(1 for r in quality_records if r['quality_flag'] == 'WARN')
    fail_days = sum(1 for r in quality_records if r['quality_flag'] == 'FAIL')
    
    print(f"  - OK days: {ok_days} ({ok_days/len(quality_records)*100:.1f}%)")
    print(f"  - WARN days: {warn_days} ({warn_days/len(quality_records)*100:.1f}%)")
    print(f"  - FAIL days: {fail_days} ({fail_days/len(quality_records)*100:.1f}%)")
    
    # Show sample of data
    if df_consolidated is not None and len(df_consolidated) > 0:
        print(f"\n📋 SAMPLE DATA (first 3 rows):")
        print(df_consolidated[['episode_id', 't_in_episode', 'time_utc', 'open', 'close']].head(3))
        
        print(f"\n📋 SAMPLE DATA (last 3 rows):")
        print(df_consolidated[['episode_id', 't_in_episode', 'time_utc', 'open', 'close', 'is_terminal']].tail(3))
    
    print("\n" + "="*70)
    print("✅ CONSOLIDATION COMPLETE - Dataset ready for L2 processing")
    print("="*70)

def main():
    """Main execution"""
    logger.info("Starting L1 dataset consolidation...")
    
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
    
    # Save all outputs
    output_info = save_outputs(df_consolidated, quality_records, dates_processed)
    
    # Print summary
    print_summary(df_consolidated, quality_records, output_info)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        logger.error("Consolidation failed!")
        exit(1)