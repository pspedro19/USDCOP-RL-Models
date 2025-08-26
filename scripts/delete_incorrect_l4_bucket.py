#!/usr/bin/env python3
"""
Delete Incorrect L4 Bucket from MinIO
======================================
Removes the incorrectly named bucket 'ds-usdcop-rlready' 
and ensures the correct bucket '04-l4-ds-usdcop-rlready' exists.
"""

import boto3
from botocore.client import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_incorrect_bucket():
    """Delete incorrect L4 bucket and create correct one"""
    
    # MinIO connection
    s3_client = boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='airflow',
        aws_secret_access_key='airflow',
        config=Config(signature_version='s3v4')
    )
    
    incorrect_bucket = 'ds-usdcop-rlready'
    correct_bucket = '04-l4-ds-usdcop-rlready'
    
    try:
        # Check if incorrect bucket exists
        logger.info(f"Checking for incorrect bucket: {incorrect_bucket}")
        try:
            s3_client.head_bucket(Bucket=incorrect_bucket)
            bucket_exists = True
            logger.info(f"Found incorrect bucket: {incorrect_bucket}")
        except:
            bucket_exists = False
            logger.info(f"Incorrect bucket does not exist: {incorrect_bucket}")
        
        if bucket_exists:
            # List objects in incorrect bucket
            logger.info(f"Listing objects in {incorrect_bucket}...")
            response = s3_client.list_objects_v2(Bucket=incorrect_bucket)
            
            if 'Contents' in response:
                object_count = len(response['Contents'])
                logger.warning(f"Found {object_count} objects in {incorrect_bucket}")
                
                # Optional: List first few objects for verification
                for i, obj in enumerate(response['Contents'][:5]):
                    logger.info(f"  - {obj['Key']} ({obj['Size']} bytes)")
                if object_count > 5:
                    logger.info(f"  ... and {object_count - 5} more objects")
                
                # Delete all objects
                logger.info(f"Deleting all objects from {incorrect_bucket}...")
                for obj in response['Contents']:
                    s3_client.delete_object(Bucket=incorrect_bucket, Key=obj['Key'])
                logger.info(f"Deleted {object_count} objects")
            else:
                logger.info(f"No objects found in {incorrect_bucket}")
            
            # Delete the bucket
            logger.info(f"Deleting bucket: {incorrect_bucket}")
            s3_client.delete_bucket(Bucket=incorrect_bucket)
            logger.info(f"✅ Successfully deleted incorrect bucket: {incorrect_bucket}")
        
        # Ensure correct bucket exists
        logger.info(f"Checking for correct bucket: {correct_bucket}")
        try:
            s3_client.head_bucket(Bucket=correct_bucket)
            logger.info(f"✅ Correct bucket already exists: {correct_bucket}")
        except:
            logger.info(f"Creating correct bucket: {correct_bucket}")
            s3_client.create_bucket(Bucket=correct_bucket)
            logger.info(f"✅ Successfully created correct bucket: {correct_bucket}")
        
        # List all buckets for verification
        logger.info("\nCurrent MinIO buckets:")
        response = s3_client.list_buckets()
        for bucket in response['Buckets']:
            logger.info(f"  - {bucket['Name']}")
        
        logger.info("\n✅ Cleanup completed successfully!")
        logger.info(f"   Incorrect bucket '{incorrect_bucket}' has been removed")
        logger.info(f"   Correct bucket '{correct_bucket}' is ready for use")
        
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise

if __name__ == "__main__":
    cleanup_incorrect_bucket()