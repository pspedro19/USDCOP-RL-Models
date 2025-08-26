#!/usr/bin/env python3
"""
Run L1 standardization pipeline for all dates in batch
Processes L0 data and generates standardized L1 outputs
"""

import subprocess
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
import boto3
from botocore.client import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L0 = "ds-usdcop-acquire"
BUCKET_L1 = "ds-usdcop-standardize"
DAG_ID = "usdcop_m5__02_l1_standardize"

def get_available_dates():
    """Get all dates available in L0 bucket"""
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    
    # List all objects in L0
    dates = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=BUCKET_L0, Prefix="usdcop_m5__01_l0_acquire/")
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                # Extract date from key
                if 'date=' in obj['Key']:
                    date_part = obj['Key'].split('date=')[1].split('/')[0]
                    if len(date_part) == 10:  # YYYY-MM-DD format
                        dates.add(date_part)
    
    return sorted(list(dates))

def trigger_dag_for_date(date_str):
    """Trigger L1 DAG for a specific date"""
    # Use the date directly as execution date
    trigger_date_str = date_str
    
    cmd = [
        'docker', 'exec', 'usdcop-airflow-webserver',
        'airflow', 'dags', 'trigger', DAG_ID,
        '--exec-date', trigger_date_str
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            logger.info(f"✅ Triggered DAG for {trigger_date_str} (processes {date_str})")
            return True
        else:
            if "already exists" in result.stderr:
                logger.warning(f"⚠️ DAG run already exists for {trigger_date_str}")
                return True
            else:
                logger.error(f"❌ Failed to trigger DAG for {trigger_date_str}: {result.stderr}")
                return False
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Timeout triggering DAG for {trigger_date_str}")
        return False
    except Exception as e:
        logger.error(f"❌ Error triggering DAG for {trigger_date_str}: {e}")
        return False

def wait_for_dag_completion(date_str, max_wait=60):
    """Wait for DAG run to complete"""
    trigger_date_str = date_str
    
    cmd = [
        'docker', 'exec', 'usdcop-airflow-webserver',
        'airflow', 'dags', 'state', DAG_ID, trigger_date_str
    ]
    
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                state = result.stdout.strip()
                if state in ['success', 'failed']:
                    return state
            time.sleep(2)
        except:
            pass
    
    return 'timeout'

def check_l1_output(date_str):
    """Check if L1 output exists for a date"""
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    
    # Check for standardized data
    prefix = f"usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date={date_str}/"
    
    try:
        response = s3_client.list_objects_v2(Bucket=BUCKET_L1, Prefix=prefix, MaxKeys=1)
        return 'Contents' in response
    except:
        return False

def run_batch_pipeline():
    """Run L1 pipeline for all available dates"""
    logger.info("Starting batch L1 pipeline processing")
    
    # Get available dates
    dates = get_available_dates()
    logger.info(f"Found {len(dates)} dates in L0 bucket")
    
    if not dates:
        logger.error("No dates found in L0 bucket")
        return
    
    # Process statistics
    stats = {
        'total': len(dates),
        'processed': 0,
        'skipped': 0,
        'failed': 0
    }
    
    # Process all dates
    dates_to_process = dates
    logger.info(f"Processing all {len(dates_to_process)} dates")
    
    for i, date_str in enumerate(dates_to_process, 1):
        logger.info(f"\n[{i}/{len(dates_to_process)}] Processing {date_str}")
        
        # Check if already processed
        if check_l1_output(date_str):
            logger.info(f"  ⏭️ Already processed, skipping")
            stats['skipped'] += 1
            continue
        
        # Trigger DAG
        if trigger_dag_for_date(date_str):
            # Wait for completion
            state = wait_for_dag_completion(date_str, max_wait=30)
            
            if state == 'success':
                logger.info(f"  ✅ Successfully processed")
                stats['processed'] += 1
            elif state == 'failed':
                logger.error(f"  ❌ Processing failed")
                stats['failed'] += 1
            else:
                logger.warning(f"  ⏱️ Processing timeout")
                stats['failed'] += 1
        else:
            stats['failed'] += 1
        
        # Small delay between triggers
        time.sleep(1)
    
    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("BATCH PROCESSING SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total dates: {stats['total']}")
    logger.info(f"Processed: {stats['processed']}")
    logger.info(f"Skipped: {stats['skipped']}")
    logger.info(f"Failed: {stats['failed']}")
    
    return stats['processed'] > 0

if __name__ == "__main__":
    # Wait for L0 upload to complete
    logger.info("Waiting for L0 data upload to complete...")
    time.sleep(10)
    
    success = run_batch_pipeline()
    if success:
        logger.info("\n🎉 Batch L1 processing completed!")
        logger.info("Check MinIO L1 bucket for standardized outputs")
    else:
        logger.error("\n❌ Batch processing failed")