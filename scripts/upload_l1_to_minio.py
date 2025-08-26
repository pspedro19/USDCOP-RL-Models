#!/usr/bin/env python3
"""
Upload L1 Consolidated Files to MinIO
======================================
Uploads the 4 required L1 files to MinIO for Airflow DAG access
"""

import boto3
from botocore.client import Config
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
MINIO_ENDPOINT = 'http://localhost:9000'
MINIO_ACCESS_KEY = 'minioadmin'
MINIO_SECRET_KEY = 'minioadmin'
BUCKET = "ds-usdcop-standardize"
LOCAL_DIR = Path(r"C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP_Trading_RL\data\L1_consolidated")

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

def upload_files():
    """Upload L1 files to MinIO"""
    s3_client = connect_to_minio()
    
    # Ensure bucket exists
    try:
        s3_client.head_bucket(Bucket=BUCKET)
        logger.info(f"Bucket {BUCKET} exists")
    except:
        s3_client.create_bucket(Bucket=BUCKET)
        logger.info(f"Created bucket {BUCKET}")
    
    # Files to upload
    files_to_upload = [
        ("standardized_data.parquet", "usdcop_m5__02_l1_consolidate/consolidated/standardized_data.parquet"),
        ("standardized_data.csv", "usdcop_m5__02_l1_consolidate/consolidated/standardized_data.csv"),
        ("_reports/daily_quality_60.csv", "usdcop_m5__02_l1_consolidate/consolidated/_reports/daily_quality_60.csv"),
        ("_metadata.json", "usdcop_m5__02_l1_consolidate/consolidated/_metadata.json"),
        # Also upload to premium_data for easier access
        ("standardized_data.parquet", "premium_data/standardized_data.parquet"),
        ("standardized_data.csv", "premium_data/standardized_data.csv"),
    ]
    
    uploaded_count = 0
    
    for local_file, s3_key in files_to_upload:
        local_path = LOCAL_DIR / local_file
        
        if not local_path.exists():
            logger.warning(f"File not found: {local_path}")
            continue
        
        try:
            # Upload file
            with open(local_path, 'rb') as f:
                s3_client.put_object(
                    Bucket=BUCKET,
                    Key=s3_key,
                    Body=f
                )
            
            logger.info(f"✅ Uploaded: {local_file} -> s3://{BUCKET}/{s3_key}")
            uploaded_count += 1
            
            # Verify upload
            response = s3_client.head_object(Bucket=BUCKET, Key=s3_key)
            size = response['ContentLength']
            logger.info(f"   Size: {size:,} bytes")
            
        except Exception as e:
            logger.error(f"Failed to upload {local_file}: {e}")
    
    # Also upload to the acquire bucket for the DAG to find
    try:
        s3_client.head_bucket(Bucket="ds-usdcop-acquire")
    except:
        s3_client.create_bucket(Bucket="ds-usdcop-acquire")
        logger.info("Created bucket ds-usdcop-acquire")
    
    # Upload premium data to acquire bucket
    premium_files = [
        ("standardized_data.parquet", "premium_data/consolidated_premium.parquet"),
        ("standardized_data.csv", "premium_data/consolidated_premium.csv"),
    ]
    
    for local_file, s3_key in premium_files:
        local_path = LOCAL_DIR / local_file
        
        if local_path.exists():
            try:
                with open(local_path, 'rb') as f:
                    s3_client.put_object(
                        Bucket="ds-usdcop-acquire",
                        Key=s3_key,
                        Body=f
                    )
                logger.info(f"✅ Also uploaded to: s3://ds-usdcop-acquire/{s3_key}")
                uploaded_count += 1
            except Exception as e:
                logger.error(f"Failed to upload to acquire bucket: {e}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Upload complete: {uploaded_count} files uploaded to MinIO")
    logger.info(f"{'='*70}")
    
    # List files in bucket
    logger.info("\nFiles in MinIO bucket:")
    response = s3_client.list_objects_v2(Bucket=BUCKET, Prefix="usdcop_m5__02_l1_consolidate/")
    
    if 'Contents' in response:
        for obj in response['Contents']:
            logger.info(f"  - {obj['Key']} ({obj['Size']:,} bytes)")
    
    return uploaded_count

def main():
    """Main execution"""
    logger.info("Starting L1 data upload to MinIO...")
    
    # Check local files exist
    local_files = list(LOCAL_DIR.glob("*")) + list((LOCAL_DIR / "_reports").glob("*"))
    logger.info(f"Found {len(local_files)} local files")
    
    # Upload to MinIO
    uploaded = upload_files()
    
    if uploaded > 0:
        logger.info("\n✅ SUCCESS: L1 data is now available in MinIO")
        logger.info("You can now run the Airflow DAG: usdcop_m5__02_l1_consolidate")
    else:
        logger.error("\n❌ ERROR: No files were uploaded")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        exit(1)