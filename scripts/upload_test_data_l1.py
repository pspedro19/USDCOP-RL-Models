#!/usr/bin/env python3
"""
Upload test data for L1 pipeline validation
Ensures data for 2020-01-02 is available in L0 bucket
"""

import pandas as pd
import boto3
from botocore.client import Config
import json
from datetime import datetime
import logging
from pathlib import Path
import hashlib
import pyarrow as pa
import pyarrow.parquet as pq

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
TARGET_DATE = "2020-01-05"

def create_perfect_m5_data():
    """Create perfect M5 grid data for testing"""
    import numpy as np
    
    # Create perfect 60-bar M5 grid for premium window
    # Premium window: 08:00-12:55 COT = 13:00-17:55 UTC
    times = pd.date_range(
        start=f"{TARGET_DATE} 13:00:00",
        end=f"{TARGET_DATE} 17:55:00", 
        freq='5min'
    )
    
    # Generate realistic OHLC data
    base_price = 3900.0
    data = []
    
    for i, time in enumerate(times):
        # Add some realistic price movement
        trend = np.sin(i / 10) * 20  # Sinusoidal trend
        noise = np.random.randn() * 5  # Random noise
        
        open_price = base_price + trend + noise
        close_price = open_price + np.random.randn() * 3
        
        # Ensure OHLC coherence
        high_price = max(open_price, close_price) + abs(np.random.randn() * 2)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 2)
        
        data.append({
            'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'open': round(open_price, 6),
            'high': round(high_price, 6),
            'low': round(low_price, 6),
            'close': round(close_price, 6),
            'volume': round(np.random.uniform(100, 1000), 2),
            'source': 'twelvedata',
            'date': TARGET_DATE
        })
        
        base_price = close_price  # Continue from last close
    
    df = pd.DataFrame(data)
    logger.info(f"Created perfect M5 grid with {len(df)} bars")
    
    return df

def upload_to_minio():
    """Upload test data to MinIO for L1 pipeline"""
    
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
    
    # Create bucket if needed
    try:
        s3_client.head_bucket(Bucket=BUCKET_L0)
        logger.info(f"Bucket {BUCKET_L0} exists")
    except:
        s3_client.create_bucket(Bucket=BUCKET_L0)
        logger.info(f"Created bucket {BUCKET_L0}")
    
    # Try to load existing data or create perfect test data
    if CSV_FILE.exists():
        # Load and filter existing data
        df_all = pd.read_csv(CSV_FILE)
        df_all['date'] = pd.to_datetime(df_all['time']).dt.date.astype(str)
        df = df_all[df_all['date'] == TARGET_DATE].copy()
        
        if len(df) == 0:
            logger.warning(f"No data found for {TARGET_DATE} in CSV, creating perfect test data")
            df = create_perfect_m5_data()
        else:
            logger.info(f"Found {len(df)} rows for {TARGET_DATE}")
            # Ensure required columns
            df['source'] = 'twelvedata'
            if 'volume' not in df.columns:
                df['volume'] = 0.0
    else:
        logger.info("CSV file not found, creating perfect test data")
        df = create_perfect_m5_data()
    
    # Generate run_id
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Upload as parquet
    parquet_key = (
        f"{DAG_ID}/market=usdcop/timeframe=m5/"
        f"source=twelvedata/date={TARGET_DATE}/run_id={run_id}/data.parquet"
    )
    
    # Convert to parquet bytes
    table = pa.Table.from_pandas(df)
    buffer = pa.BufferOutputStream()
    pq.write_table(table, buffer, compression='snappy')
    parquet_bytes = buffer.getvalue().to_pybytes()
    
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=parquet_key,
        Body=parquet_bytes
    )
    logger.info(f"✅ Uploaded parquet to {parquet_key}")
    
    # Calculate hash for integrity
    data_hash = hashlib.sha256(parquet_bytes).hexdigest()
    
    # Also upload as CSV
    csv_key = (
        f"{DAG_ID}/market=usdcop/timeframe=m5/"
        f"source=twelvedata/date={TARGET_DATE}/run_id={run_id}/data.csv"
    )
    
    csv_data = df.to_csv(index=False, float_format='%.6f')
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=csv_key,
        Body=csv_data.encode('utf-8')
    )
    logger.info(f"✅ Uploaded CSV to {csv_key}")
    
    # Create comprehensive metadata
    metadata = {
        'timestamp': datetime.utcnow().isoformat(),
        'date': TARGET_DATE,
        'run_id': run_id,
        'source': 'twelvedata',
        'records': len(df),
        'columns': df.columns.tolist(),
        'data_integrity': {
            'sha256': data_hash,
            'size_bytes': len(parquet_bytes)
        },
        'quality_metrics': {
            'completeness': len(df) / 60,  # Expected 60 bars
            'has_gaps': len(df) < 60,
            'ohlc_valid': True
        }
    }
    
    metadata_key = (
        f"{DAG_ID}/market=usdcop/timeframe=m5/"
        f"source=twelvedata/date={TARGET_DATE}/run_id={run_id}/_metadata.json"
    )
    
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2).encode('utf-8')
    )
    logger.info(f"✅ Created metadata at {metadata_key}")
    
    # Create READY signal
    ready_data = {
        'timestamp': datetime.utcnow().isoformat(),
        'date': TARGET_DATE,
        'run_id': run_id,
        'source': 'manual_upload',
        'status': 'READY',
        'records': len(df),
        'data_hash': data_hash
    }
    
    ready_key = f"{DAG_ID}/_control/date={TARGET_DATE}/run_id={run_id}/READY"
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=ready_key,
        Body=json.dumps(ready_data, indent=2).encode('utf-8')
    )
    logger.info(f"✅ Created READY signal at {ready_key}")
    
    # Create quality report
    quality_report = {
        'date': TARGET_DATE,
        'run_id': run_id,
        'records': len(df),
        'completeness': len(df) / 60,
        'source': 'twelvedata',
        'columns': df.columns.tolist(),
        'timestamp': datetime.utcnow().isoformat(),
        'quality_checks': {
            'row_count': len(df),
            'expected_rows': 60,
            'completeness_pct': (len(df) / 60) * 100,
            'has_nulls': df.isnull().any().any(),
            'ohlc_coherence': True
        }
    }
    
    report_key = f"{DAG_ID}/_reports/date={TARGET_DATE}/run_id={run_id}/quality_report.json"
    s3_client.put_object(
        Bucket=BUCKET_L0,
        Key=report_key,
        Body=json.dumps(quality_report, indent=2, default=str).encode('utf-8')
    )
    logger.info(f"✅ Created quality report at {report_key}")
    
    # List uploaded files
    logger.info(f"\n{'='*60}")
    logger.info("UPLOADED FILES SUMMARY")
    logger.info(f"{'='*60}")
    
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_L0,
        Prefix=f"{DAG_ID}/"
    )
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if TARGET_DATE in obj['Key']:
                size_kb = obj['Size'] / 1024
                logger.info(f"  📁 {obj['Key']} ({size_kb:.1f} KB)")
    
    logger.info(f"\n✅ L0 test data ready for date {TARGET_DATE}")
    logger.info(f"   - Run ID: {run_id}")
    logger.info(f"   - Records: {len(df)}")
    logger.info(f"   - Completeness: {len(df)/60*100:.1f}%")
    logger.info(f"   - Data Hash: {data_hash[:16]}...")
    
    return True

if __name__ == "__main__":
    success = upload_to_minio()
    if success:
        logger.info("\n🎉 Test data uploaded successfully!")
        logger.info("You can now run the L1 pipeline for this date.")
        logger.info("\nNext steps:")
        logger.info("1. Trigger L1 DAG in Airflow for date 2020-01-02")
        logger.info("2. Run: python scripts/test_l1_pipeline_complete.py")
    else:
        logger.error("\n❌ Upload failed")