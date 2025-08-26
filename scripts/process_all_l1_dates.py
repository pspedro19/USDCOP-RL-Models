#!/usr/bin/env python3
"""
Process all L0 dates through L1 pipeline efficiently
Triggers DAG runs for all dates and monitors progress
"""

import subprocess
import pandas as pd
import logging
from pathlib import Path
import time
import boto3
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET_L0 = "ds-usdcop-acquire"
BUCKET_L1 = "ds-usdcop-standardize"
DAG_ID = "usdcop_m5__02_l1_standardize"
BATCH_SIZE = 20  # Process 20 dates in parallel

def get_all_l0_dates():
    """Get all dates available in L0 bucket"""
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    
    dates = set()
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(
        Bucket=BUCKET_L0, 
        Prefix="usdcop_m5__01_l0_acquire/market=usdcop/timeframe=m5/source=twelvedata/date="
    )
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if 'date=' in obj['Key'] and 'data.parquet' in obj['Key']:
                    date_part = obj['Key'].split('date=')[1].split('/')[0]
                    if len(date_part) == 10:
                        dates.add(date_part)
    
    return sorted(list(dates))

def check_l1_exists(date_str):
    """Check if L1 output already exists for a date"""
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        use_ssl=False
    )
    
    prefix = f"usdcop_m5__02_l1_standardize/market=usdcop/timeframe=m5/date={date_str}/"
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=BUCKET_L1, 
            Prefix=prefix,
            MaxKeys=1
        )
        # Check if standardized_data.parquet exists
        if 'Contents' in response:
            for obj in response['Contents']:
                if 'standardized_data.parquet' in obj['Key']:
                    return True
        return False
    except:
        return False

def trigger_dag_run(date_str):
    """Trigger a single DAG run for a date"""
    cmd = [
        'docker', 'exec', 'usdcop-airflow-webserver',
        'airflow', 'dags', 'trigger', DAG_ID,
        '--exec-date', date_str
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return {'date': date_str, 'status': 'triggered'}
        elif "already exists" in result.stderr:
            return {'date': date_str, 'status': 'already_running'}
        else:
            return {'date': date_str, 'status': 'failed', 'error': result.stderr}
    except Exception as e:
        return {'date': date_str, 'status': 'error', 'error': str(e)}

def wait_for_batch_completion(dates, timeout=300):
    """Wait for a batch of DAG runs to complete"""
    start_time = time.time()
    pending = set(dates)
    completed = {}
    
    while pending and (time.time() - start_time < timeout):
        for date_str in list(pending):
            cmd = [
                'docker', 'exec', 'usdcop-airflow-webserver',
                'airflow', 'dags', 'state', DAG_ID, date_str
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    state = result.stdout.strip()
                    if state in ['success', 'failed']:
                        completed[date_str] = state
                        pending.remove(date_str)
            except:
                pass
        
        if pending:
            time.sleep(2)
    
    # Mark remaining as timeout
    for date_str in pending:
        completed[date_str] = 'timeout'
    
    return completed

def process_all_dates():
    """Process all L0 dates through L1 pipeline"""
    logger.info("Starting comprehensive L1 pipeline processing")
    
    # Get all available dates
    all_dates = get_all_l0_dates()
    logger.info(f"Found {len(all_dates)} dates in L0 bucket")
    
    if not all_dates:
        logger.error("No dates found in L0 bucket")
        return False
    
    logger.info(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    
    # Filter out already processed dates
    dates_to_process = []
    already_processed = []
    
    logger.info("Checking which dates are already processed...")
    for date_str in all_dates:
        if check_l1_exists(date_str):
            already_processed.append(date_str)
        else:
            dates_to_process.append(date_str)
    
    logger.info(f"Already processed: {len(already_processed)} dates")
    logger.info(f"To process: {len(dates_to_process)} dates")
    
    if not dates_to_process:
        logger.info("All dates already processed!")
        return True
    
    # Process in batches
    total_batches = (len(dates_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Processing in {total_batches} batches of {BATCH_SIZE} dates each")
    
    stats = {
        'total': len(dates_to_process),
        'success': 0,
        'failed': 0,
        'timeout': 0
    }
    
    for batch_num in range(total_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(dates_to_process))
        batch_dates = dates_to_process[start_idx:end_idx]
        
        logger.info(f"\n[Batch {batch_num+1}/{total_batches}] Processing {len(batch_dates)} dates")
        
        # Trigger all DAGs in batch
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(trigger_dag_run, date): date for date in batch_dates}
            
            for future in as_completed(futures):
                result = future.result()
                if result['status'] in ['triggered', 'already_running']:
                    logger.info(f"  ✓ {result['date']}: {result['status']}")
                else:
                    logger.warning(f"  ✗ {result['date']}: {result['status']}")
        
        # Wait for batch completion
        logger.info(f"  Waiting for batch completion...")
        results = wait_for_batch_completion(batch_dates, timeout=60)
        
        for date_str, state in results.items():
            if state == 'success':
                stats['success'] += 1
                logger.info(f"  ✅ {date_str}: SUCCESS")
            elif state == 'failed':
                stats['failed'] += 1
                logger.error(f"  ❌ {date_str}: FAILED")
            else:
                stats['timeout'] += 1
                logger.warning(f"  ⏱️ {date_str}: TIMEOUT")
        
        # Small delay between batches
        if batch_num < total_batches - 1:
            time.sleep(5)
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("L1 PIPELINE PROCESSING COMPLETE")
    logger.info("="*60)
    logger.info(f"Total dates processed: {stats['total']}")
    logger.info(f"✅ Successful: {stats['success']}")
    logger.info(f"❌ Failed: {stats['failed']}")
    logger.info(f"⏱️ Timeout: {stats['timeout']}")
    logger.info(f"⏭️ Already processed: {len(already_processed)}")
    
    # Save processing report
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_dates_in_l0': len(all_dates),
        'already_processed': len(already_processed),
        'processed_in_run': stats['total'],
        'successful': stats['success'],
        'failed': stats['failed'],
        'timeout': stats['timeout'],
        'date_range': {
            'first': all_dates[0],
            'last': all_dates[-1]
        }
    }
    
    report_path = Path("data/L1_consolidated/_reports/l1_processing_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"\n📊 Processing report saved to: {report_path}")
    
    return stats['success'] > 0

if __name__ == "__main__":
    success = process_all_dates()
    if success:
        logger.info("\n🎉 L1 pipeline processing completed!")
        logger.info("Now run consolidate_l1_dataset.py to create the final consolidated dataset")
    else:
        logger.error("\n❌ L1 pipeline processing failed")