#!/usr/bin/env python3
"""
Initialize MinIO buckets for USDCOP Trading Pipeline
Creates all required buckets according to the data flow
"""

from minio import Minio
from minio.error import S3Error
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MinIO configuration
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin"

# Buckets according to data flow L0 -> L1 -> L2 -> L3 -> L4 -> L5
BUCKETS = [
    # L0 - Acquire Layer (Raw data from sources)
    {
        "name": "ds-usdcop-acquire",
        "description": "L0 - Raw data from MT5 and TwelveData",
        "folders": [
            "usdcop_m5__01_l0_acquire_sync_incremental/",
            "usdcop_m5__01_l0_acquire_sync_incremental/_metadata/",
            "usdcop_m5__01_l0_acquire_sync_incremental/_quality/",
            "usdcop_m5__01_l0_acquire_sync_incremental/_sync/"
        ]
    },
    
    # L1 - Standardize Layer (Normalized and merged data)
    {
        "name": "ds-usdcop-standardize",
        "description": "L1 - Standardized data with canonical merge",
        "folders": [
            "usdcop_m5__02_l1_standardize_time_sessions/",
            "usdcop_m5__02_l1_standardize_time_sessions/_reports/",
            "usdcop_m5__02_l1_standardize_time_sessions/_quality/",
            "usdcop_m5__02_l1_standardize_time_sessions/_audit/"
        ]
    },
    
    # L2 - Prepare Layer (Cleaned and prepared data)
    {
        "name": "ds-usdcop-prepare",
        "description": "L2 - Prepared data with premium filtering",
        "folders": [
            "usdcop_m5__03_l2_prepare/",
            "usdcop_m5__03_l2_prepare/_reports/",
            "usdcop_m5__03_l2_prepare/_quality/",
            "usdcop_m5__03_l2_prepare/_stats/"
        ]
    },
    
    # L3 - Feature Layer (Engineered features)
    {
        "name": "ds-usdcop-feature",
        "description": "L3 - Feature engineering with anti-leakage",
        "folders": [
            "usdcop_m5__04_l3_feature/",
            "usdcop_m5__04_l3_feature/_reports/",
            "usdcop_m5__04_l3_feature/_quality/",
            "usdcop_m5__04_l3_feature/_stats/"
        ]
    },
    
    # L4 - ML Ready Layer (Training-ready datasets)
    {
        "name": "ds-usdcop-mlready",
        "description": "L4 - ML-ready datasets with train/val/test splits",
        "folders": [
            "usdcop_m5__05_l4_mlready/",
            "usdcop_m5__05_l4_mlready/train/",
            "usdcop_m5__05_l4_mlready/validation/",
            "usdcop_m5__05_l4_mlready/test/",
            "usdcop_m5__05_l4_mlready/_scalers/",
            "usdcop_m5__05_l4_mlready/_reports/"
        ]
    },
    
    # L5 - Serving Layer (Model predictions)
    {
        "name": "ds-usdcop-serving",
        "description": "L5 - Model serving and predictions",
        "folders": [
            "usdcop_m5__06_l5_serving/",
            "usdcop_m5__06_l5_serving/predictions/",
            "usdcop_m5__06_l5_serving/monitoring/",
            "usdcop_m5__06_l5_serving/_reports/"
        ]
    },
    
    # Additional buckets for models and backups
    {
        "name": "ds-usdcop-models",
        "description": "Trained models storage",
        "folders": [
            "rl_models/",
            "lstm_models/",
            "ensemble_models/",
            "model_artifacts/"
        ]
    },
    {
        "name": "ds-usdcop-backups",
        "description": "Backup storage",
        "folders": [
            "daily/",
            "weekly/",
            "monthly/"
        ]
    }
]

def create_buckets():
    """Create all required MinIO buckets"""
    
    # Initialize MinIO client
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    logging.info(f"Connecting to MinIO at {MINIO_ENDPOINT}")
    
    # Check connection
    try:
        buckets = client.list_buckets()
        existing_buckets = [b.name for b in buckets]
        logging.info(f"Found {len(existing_buckets)} existing buckets")
    except Exception as e:
        logging.error(f"Failed to connect to MinIO: {e}")
        sys.exit(1)
    
    # Create each bucket
    for bucket_config in BUCKETS:
        bucket_name = bucket_config["name"]
        
        try:
            if bucket_name in existing_buckets:
                logging.info(f"✓ Bucket '{bucket_name}' already exists")
            else:
                client.make_bucket(bucket_name)
                logging.info(f"✓ Created bucket '{bucket_name}' - {bucket_config['description']}")
            
            # Create folder structure (by uploading empty .keep files)
            for folder in bucket_config["folders"]:
                keep_file = f"{folder}.keep"
                try:
                    # Check if .keep file already exists
                    client.stat_object(bucket_name, keep_file)
                    logging.debug(f"  Folder {folder} already exists")
                except S3Error as e:
                    if e.code == 'NoSuchKey':
                        # Create .keep file
                        from io import BytesIO
                        data = BytesIO(b"")
                        client.put_object(
                            bucket_name,
                            keep_file,
                            data,
                            length=0
                        )
                        logging.info(f"  → Created folder: {folder}")
                    else:
                        raise e
                        
        except S3Error as e:
            logging.error(f"Error with bucket '{bucket_name}': {e}")
            continue
    
    # List final state
    logging.info("\n" + "="*60)
    logging.info("BUCKET STRUCTURE CREATED:")
    logging.info("="*60)
    
    buckets = client.list_buckets()
    for i, bucket in enumerate(buckets, 1):
        logging.info(f"{i}. {bucket.name}")
        
        # Count objects in each bucket
        try:
            objects = list(client.list_objects(bucket.name))
            if objects:
                logging.info(f"   └─ {len(objects)} objects")
        except:
            pass
    
    logging.info("="*60)
    logging.info("✅ All buckets created successfully!")
    logging.info(f"Access MinIO Console at: http://{MINIO_ENDPOINT.replace('9000', '9001')}")
    logging.info(f"Credentials: {MINIO_ACCESS_KEY} / {MINIO_SECRET_KEY}")

if __name__ == "__main__":
    create_buckets()