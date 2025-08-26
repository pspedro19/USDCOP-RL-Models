#!/usr/bin/env python3
"""
Test MinIO connection and create required buckets
"""

import boto3
from botocore.client import Config
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MinIO configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'

def test_minio_connection():
    """Test MinIO connection and create buckets"""
    
    try:
        # Create S3 client
        s3_client = boto3.client(
            's3',
            endpoint_url=MINIO_ENDPOINT,
            aws_access_key_id=MINIO_ACCESS_KEY,
            aws_secret_access_key=MINIO_SECRET_KEY,
            config=Config(signature_version='s3v4'),
            use_ssl=False
        )
        
        logger.info("Connected to MinIO successfully")
        
        # List existing buckets
        response = s3_client.list_buckets()
        logger.info(f"Existing buckets: {[b['Name'] for b in response['Buckets']]}")
        
        # Create required buckets
        buckets_to_create = [
            'ds-usdcop-acquire',
            'ds-usdcop-standardize',
            'ds-usdcop-prepare',
            'ds-usdcop-features',
            'ds-usdcop-train',
            'ds-usdcop-predict'
        ]
        
        for bucket in buckets_to_create:
            try:
                s3_client.head_bucket(Bucket=bucket)
                logger.info(f"Bucket {bucket} already exists")
            except:
                try:
                    s3_client.create_bucket(Bucket=bucket)
                    logger.info(f"Created bucket {bucket}")
                except Exception as e:
                    logger.error(f"Failed to create bucket {bucket}: {e}")
        
        # Create a test READY signal for L0
        test_ready = {
            'timestamp': datetime.utcnow().isoformat(),
            'date': '2020-01-02',
            'run_id': 'test_20250821',
            'source': 'manual_test',
            'status': 'READY'
        }
        
        # Upload test READY signal
        ready_key = "usdcop_m5__01_l0_acquire/market=usdcop/timeframe=m5/source=twelvedata/date=2020-01-02/_control/READY"
        
        s3_client.put_object(
            Bucket='ds-usdcop-acquire',
            Key=ready_key,
            Body=json.dumps(test_ready).encode('utf-8')
        )
        logger.info(f"Created test READY signal at {ready_key}")
        
        # List files in acquire bucket
        response = s3_client.list_objects_v2(
            Bucket='ds-usdcop-acquire',
            MaxKeys=10
        )
        
        if 'Contents' in response:
            logger.info("\nFiles in ds-usdcop-acquire:")
            for obj in response['Contents']:
                logger.info(f"  - {obj['Key']}")
        
        return True
        
    except Exception as e:
        logger.error(f"MinIO test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_minio_connection()
    if success:
        logger.info("\n✅ MinIO is ready for L1 pipeline!")
        logger.info("Buckets created and test data uploaded.")
    else:
        logger.error("\n❌ MinIO setup failed. Check credentials and connection.")