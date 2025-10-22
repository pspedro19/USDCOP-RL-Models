#!/usr/bin/env python3
"""
MinIO Manifest Reader
=====================
Helper functions to read manifest files from MinIO buckets

Usage:
    from minio_manifest_reader import read_manifest_from_minio, read_parquet_from_minio

    # Read manifest
    manifest = read_manifest_from_minio('05-l5-ds-usdcop-serving', 'l5')

    # Read parquet file
    file_key = manifest['files'][0]['file_key']
    data_bytes = read_parquet_from_minio('05-l5-ds-usdcop-serving', file_key)
"""

import boto3
from botocore.client import Config
import json
import logging
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)

# MinIO configuration from environment or defaults
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
MINIO_REGION = os.getenv('MINIO_REGION', 'us-east-1')

# MinIO configuration
MINIO_CONFIG = {
    'endpoint_url': MINIO_ENDPOINT,
    'aws_access_key_id': MINIO_ACCESS_KEY,
    'aws_secret_access_key': MINIO_SECRET_KEY,
    'config': Config(signature_version='s3v4'),
    'region_name': MINIO_REGION
}

def get_minio_client():
    """
    Get boto3 S3 client configured for MinIO

    Returns:
        boto3.client: S3 client instance
    """
    try:
        client = boto3.client('s3', **MINIO_CONFIG)
        logger.info(f"MinIO client created successfully for endpoint: {MINIO_ENDPOINT}")
        return client
    except Exception as e:
        logger.error(f"Error creating MinIO client: {e}")
        raise


def read_manifest_from_minio(bucket: str, layer: str) -> Optional[Dict[str, Any]]:
    """
    Read manifest file from MinIO bucket

    Args:
        bucket: Bucket name (e.g., '01-l1-ds-usdcop-standardize')
        layer: Layer name (e.g., 'l1', 'l2', 'l3', etc.)

    Returns:
        Manifest dict or None if not found

    Example:
        >>> manifest = read_manifest_from_minio('05-l5-ds-usdcop-serving', 'l5')
        >>> print(manifest['data_contract_version'])
        '2.0'
    """
    try:
        s3_client = get_minio_client()
        manifest_key = f'_meta/{layer}_latest.json'

        logger.info(f"Reading manifest from {bucket}/{manifest_key}")
        response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
        manifest_data = json.loads(response['Body'].read().decode('utf-8'))

        logger.info(f"Successfully read manifest from {bucket}/{manifest_key}")
        logger.debug(f"Manifest keys: {list(manifest_data.keys())}")

        return manifest_data

    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"Manifest not found: {bucket}/{manifest_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading manifest from {bucket}/{manifest_key}: {e}")
        return None


def get_latest_file_from_manifest(manifest: Dict[str, Any]) -> Optional[str]:
    """
    Extract the latest file path from manifest

    Args:
        manifest: Manifest dictionary

    Returns:
        File key/path or None

    Example:
        >>> manifest = read_manifest_from_minio('05-l5-ds-usdcop-serving', 'l5')
        >>> file_key = get_latest_file_from_manifest(manifest)
        >>> print(file_key)
        'policy_USDCOP_v1.0.onnx'
    """
    try:
        files = manifest.get('files', [])
        if not files:
            logger.warning("Manifest has no files")
            return None

        # Return the first file (usually the latest)
        file_info = files[0]
        file_key = file_info.get('file_key') or file_info.get('path') or file_info.get('name')

        logger.info(f"Latest file from manifest: {file_key}")
        return file_key

    except Exception as e:
        logger.error(f"Error extracting file from manifest: {e}")
        return None


def get_all_files_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract all file information from manifest

    Args:
        manifest: Manifest dictionary

    Returns:
        List of file dictionaries with metadata
    """
    try:
        files = manifest.get('files', [])
        logger.info(f"Found {len(files)} files in manifest")
        return files
    except Exception as e:
        logger.error(f"Error extracting files from manifest: {e}")
        return []


def read_file_from_minio(bucket: str, file_key: str) -> Optional[bytes]:
    """
    Read any file from MinIO

    Args:
        bucket: Bucket name
        file_key: File path in bucket

    Returns:
        File bytes or None

    Example:
        >>> data = read_file_from_minio('05-l5-ds-usdcop-serving', 'policy.onnx')
        >>> print(f"Read {len(data)} bytes")
        Read 524288 bytes
    """
    try:
        s3_client = get_minio_client()
        logger.info(f"Reading file from {bucket}/{file_key}")

        response = s3_client.get_object(Bucket=bucket, Key=file_key)
        file_bytes = response['Body'].read()

        logger.info(f"Successfully read {len(file_bytes)} bytes from {bucket}/{file_key}")
        return file_bytes

    except s3_client.exceptions.NoSuchKey:
        logger.warning(f"File not found: {bucket}/{file_key}")
        return None
    except Exception as e:
        logger.error(f"Error reading file from {bucket}/{file_key}: {e}")
        return None


def read_parquet_from_minio(bucket: str, file_key: str) -> Optional[bytes]:
    """
    Read parquet file from MinIO (alias for read_file_from_minio)

    Args:
        bucket: Bucket name
        file_key: File path in bucket

    Returns:
        File bytes or None
    """
    return read_file_from_minio(bucket, file_key)


def list_bucket_files(bucket: str, prefix: str = '') -> List[str]:
    """
    List all files in a MinIO bucket with optional prefix

    Args:
        bucket: Bucket name
        prefix: Optional prefix to filter files

    Returns:
        List of file keys
    """
    try:
        s3_client = get_minio_client()
        logger.info(f"Listing files in {bucket} with prefix '{prefix}'")

        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        if 'Contents' not in response:
            logger.warning(f"No files found in {bucket}/{prefix}")
            return []

        files = [obj['Key'] for obj in response['Contents']]
        logger.info(f"Found {len(files)} files in {bucket}/{prefix}")

        return files

    except Exception as e:
        logger.error(f"Error listing files in {bucket}: {e}")
        return []


def get_manifest_metadata(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key metadata from manifest

    Args:
        manifest: Manifest dictionary

    Returns:
        Dictionary with key metadata fields
    """
    try:
        metadata = {
            'data_contract_version': manifest.get('data_contract_version'),
            'run_id': manifest.get('run_id'),
            'timestamp': manifest.get('timestamp'),
            'validation_status': manifest.get('validation_status'),
            'file_count': len(manifest.get('files', [])),
            'statistics': manifest.get('statistics', {})
        }

        return metadata

    except Exception as e:
        logger.error(f"Error extracting manifest metadata: {e}")
        return {}


# Convenience functions for specific layers
def read_l5_manifest():
    """Read L5 (serving) manifest"""
    return read_manifest_from_minio('05-l5-ds-usdcop-serving', 'l5')


def read_l6_manifest():
    """Read L6 (backtest) manifest"""
    return read_manifest_from_minio('usdcop-l6-backtest', 'l6')


def read_l1_manifest():
    """Read L1 (standardize) manifest"""
    return read_manifest_from_minio('01-l1-ds-usdcop-standardize', 'l1')


def read_l2_manifest():
    """Read L2 (prepare) manifest"""
    return read_manifest_from_minio('02-l2-ds-usdcop-prepare', 'l2')


def read_l3_manifest():
    """Read L3 (feature) manifest"""
    return read_manifest_from_minio('03-l3-ds-usdcop-feature', 'l3')


def read_l4_manifest():
    """Read L4 (RL ready) manifest"""
    return read_manifest_from_minio('04-l4-ds-usdcop-rlready', 'l4')


if __name__ == '__main__':
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("MinIO Manifest Reader - Test")
    print("=" * 60)

    # Test L5 manifest reading
    print("\nTesting L5 manifest...")
    l5_manifest = read_l5_manifest()
    if l5_manifest:
        print(f"✅ L5 manifest loaded successfully")
        print(f"   - Data contract version: {l5_manifest.get('data_contract_version')}")
        print(f"   - Run ID: {l5_manifest.get('run_id')}")
        print(f"   - Files: {len(l5_manifest.get('files', []))}")
        print(f"   - Validation status: {l5_manifest.get('validation_status')}")
    else:
        print("❌ L5 manifest not found (DAG may not have run yet)")

    # Test L6 manifest reading
    print("\nTesting L6 manifest...")
    l6_manifest = read_l6_manifest()
    if l6_manifest:
        print(f"✅ L6 manifest loaded successfully")
        print(f"   - Data contract version: {l6_manifest.get('data_contract_version')}")
        print(f"   - Run ID: {l6_manifest.get('run_id')}")
        print(f"   - Files: {len(l6_manifest.get('files', []))}")
    else:
        print("❌ L6 manifest not found (DAG may not have run yet)")

    print("\n" + "=" * 60)
    print("Test complete. Module is ready to use.")
    print("=" * 60)
