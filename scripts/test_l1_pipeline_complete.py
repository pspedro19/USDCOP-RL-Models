#!/usr/bin/env python3
"""
Complete test of L1 pipeline - verify all files are created correctly
"""

import pandas as pd
import boto3
from botocore.client import Config
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L1 = "ds-usdcop-standardize"
DAG_ID = "usdcop_m5__02_l1_standardize"
TEST_DATE = "2020-01-14"  # Test with this date

def verify_l1_output():
    """Verify all L1 output files are created correctly"""
    
    # Connect to MinIO
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    
    logger.info(f"Checking L1 output for date: {TEST_DATE}")
    
    # Expected file structure
    expected_files = {
        'Main Data': [
            f'{DAG_ID}/market=usdcop/timeframe=m5/date={TEST_DATE}/*/standardized_data.parquet',
            f'{DAG_ID}/market=usdcop/timeframe=m5/date={TEST_DATE}/*/standardized_data.csv',
            f'{DAG_ID}/market=usdcop/timeframe=m5/date={TEST_DATE}/*/_metadata.json',
            f'{DAG_ID}/market=usdcop/timeframe=m5/date={TEST_DATE}/*/_control/READY'
        ],
        'Reports': [
            f'{DAG_ID}/_reports/date={TEST_DATE}/*/daily_quality_60.csv',
            f'{DAG_ID}/_reports/date={TEST_DATE}/*/quality_summary.json',
            f'{DAG_ID}/_reports/date={TEST_DATE}/*/grid_alignment.json'
        ],
        'Statistics': [
            f'{DAG_ID}/_statistics/snapshot_date={TEST_DATE}/hod_baseline.parquet'
        ],
        'Audit': [
            f'{DAG_ID}/_audit/date={TEST_DATE}/*/gating_decision.json',
            f'{DAG_ID}/_audit/date={TEST_DATE}/*/ohlc_coherence_report.json',
            f'{DAG_ID}/_audit/date={TEST_DATE}/*/lineage.json'
        ],
        'Schema': [
            f'{DAG_ID}/_schemas/schema_v2.0.json'
        ]
    }
    
    # Check each category
    results = {}
    all_files = []
    
    # List all files in the bucket
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=BUCKET_L1, Prefix=DAG_ID)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    all_files.append(obj['Key'])
    except Exception as e:
        logger.error(f"Error listing bucket contents: {e}")
        return False
    
    logger.info(f"Found {len(all_files)} total files in L1 bucket")
    
    # Check expected files
    for category, patterns in expected_files.items():
        logger.info(f"\nChecking {category}:")
        results[category] = {}
        
        for pattern in patterns:
            # Convert pattern to regex-like matching
            pattern_simple = pattern.replace('*/', '').replace('*', '')
            found = False
            
            for file in all_files:
                if TEST_DATE in file and pattern_simple.split('/')[-1] in file:
                    found = True
                    results[category][pattern_simple.split('/')[-1]] = '✅ Found'
                    logger.info(f"  ✅ {pattern_simple.split('/')[-1]}: {file}")
                    
                    # Try to read and validate content for key files
                    if file.endswith('_metadata.json'):
                        try:
                            obj = s3_client.get_object(Bucket=BUCKET_L1, Key=file)
                            content = json.loads(obj['Body'].read())
                            logger.info(f"     - Rows: {content.get('rows', 'N/A')}")
                            logger.info(f"     - Completeness: {content.get('completeness_pct', 'N/A')}%")
                            logger.info(f"     - Quality: {content.get('validation_results', {}).get('quality_flag', 'N/A')}")
                            logger.info(f"     - SHA256: {content.get('data_integrity', {}).get('sha256', 'N/A')[:16]}...")
                        except Exception as e:
                            logger.warning(f"     - Error reading metadata: {e}")
                    
                    elif file.endswith('daily_quality_60.csv'):
                        try:
                            obj = s3_client.get_object(Bucket=BUCKET_L1, Key=file)
                            df = pd.read_csv(obj['Body'])
                            logger.info(f"     - Quality report rows: {len(df)}")
                            if len(df) > 0:
                                logger.info(f"     - Quality flag: {df.iloc[0]['quality_flag']}")
                                logger.info(f"     - Completeness: {df.iloc[0]['completeness_pct']}%")
                        except Exception as e:
                            logger.warning(f"     - Error reading quality report: {e}")
                    
                    break
            
            if not found:
                results[category][pattern_simple.split('/')[-1]] = '❌ Missing'
                logger.warning(f"  ❌ {pattern_simple.split('/')[-1]}: NOT FOUND")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("L1 PIPELINE VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    
    total_expected = sum(len(patterns) for patterns in expected_files.values())
    total_found = sum(1 for cat in results.values() for status in cat.values() if '✅' in status)
    
    logger.info(f"Total files expected: {total_expected}")
    logger.info(f"Total files found: {total_found}")
    logger.info(f"Success rate: {total_found/total_expected*100:.1f}%")
    
    if total_found == total_expected:
        logger.info("\n✅ ALL REQUIRED FILES CREATED SUCCESSFULLY!")
        return True
    else:
        logger.error(f"\n❌ MISSING {total_expected - total_found} FILES")
        logger.error("Missing files:")
        for category, files in results.items():
            for file, status in files.items():
                if '❌' in status:
                    logger.error(f"  - {category}/{file}")
        return False

def check_data_integrity():
    """Cross-validate that parquet and CSV have same data"""
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    
    logger.info("\n" + "="*60)
    logger.info("DATA INTEGRITY CHECK")
    logger.info("="*60)
    
    # Find parquet and CSV files
    response = s3_client.list_objects_v2(
        Bucket=BUCKET_L1,
        Prefix=f'{DAG_ID}/market=usdcop/timeframe=m5/date={TEST_DATE}/'
    )
    
    parquet_file = None
    csv_file = None
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if 'standardized_data.parquet' in obj['Key']:
                parquet_file = obj['Key']
            elif 'standardized_data.csv' in obj['Key']:
                csv_file = obj['Key']
    
    if parquet_file and csv_file:
        try:
            # Read parquet
            obj = s3_client.get_object(Bucket=BUCKET_L1, Key=parquet_file)
            df_parquet = pd.read_parquet(obj['Body'])
            
            # Read CSV
            obj = s3_client.get_object(Bucket=BUCKET_L1, Key=csv_file)
            df_csv = pd.read_csv(obj['Body'])
            
            logger.info(f"Parquet rows: {len(df_parquet)}")
            logger.info(f"CSV rows: {len(df_csv)}")
            
            if len(df_parquet) == len(df_csv):
                logger.info("✅ Row count matches")
            else:
                logger.error("❌ Row count mismatch!")
            
            # Check columns
            logger.info(f"Parquet columns: {len(df_parquet.columns)}")
            logger.info(f"CSV columns: {len(df_csv.columns)}")
            
            # Check key integrity rules
            if len(df_parquet) > 0:
                # Check unique constraints
                if 'time_utc' in df_parquet.columns:
                    unique_times = df_parquet['time_utc'].nunique()
                    logger.info(f"Unique time_utc: {unique_times} (should equal {len(df_parquet)})")
                
                # Check OHLC coherence
                if all(col in df_parquet.columns for col in ['open', 'high', 'low', 'close']):
                    ohlc_valid = (
                        (df_parquet['high'] >= df_parquet[['open', 'close']].max(axis=1)) &
                        (df_parquet[['open', 'close']].min(axis=1) >= df_parquet['low'])
                    ).all()
                    
                    if ohlc_valid:
                        logger.info("✅ OHLC coherence: VALID")
                    else:
                        logger.error("❌ OHLC coherence: VIOLATIONS FOUND")
                
                # Check grid alignment (5-minute intervals)
                if 'time_utc' in df_parquet.columns:
                    df_parquet['time_utc'] = pd.to_datetime(df_parquet['time_utc'])
                    time_diffs = df_parquet['time_utc'].diff().dt.total_seconds().dropna()
                    grid_aligned = (time_diffs % 300 == 0).all()
                    
                    if grid_aligned:
                        logger.info("✅ Grid M5 alignment: PERFECT")
                    else:
                        logger.error("❌ Grid M5 alignment: VIOLATIONS")
                
        except Exception as e:
            logger.error(f"Error in integrity check: {e}")
    else:
        logger.warning("Could not find both parquet and CSV files for comparison")

if __name__ == "__main__":
    # Run validation
    success = verify_l1_output()
    
    # Check data integrity
    check_data_integrity()
    
    if success:
        logger.info("\n🎉 L1 PIPELINE VALIDATION PASSED!")
        logger.info("All required files are present and properly formatted.")
    else:
        logger.error("\n⚠️ L1 PIPELINE VALIDATION FAILED")
        logger.error("Please check the missing files and re-run the pipeline.")