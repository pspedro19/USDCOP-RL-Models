#!/usr/bin/env python3
"""
Example: How to Write Manifests from Airflow DAGs
==================================================
This script shows how pipeline DAGs should write manifests
to enable the API to discover and serve their outputs.

Integration Points:
- Airflow DAG completes L4 processing
- DAG writes manifest to MinIO
- API automatically discovers latest run via manifest
- Frontend gets real data without hardcoding paths
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path

import boto3
from botocore.client import Config


# ==========================================
# S3/MinIO CLIENT SETUP
# ==========================================
def get_s3_client():
    """Create S3 client configured for MinIO"""
    return boto3.client(
        's3',
        endpoint_url='http://localhost:9000',
        aws_access_key_id='minioadmin',
        aws_secret_access_key='minioadmin123',
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )


# ==========================================
# MANIFEST WRITER
# ==========================================
def write_manifest(
    s3_client,
    bucket: str,
    layer: str,
    run_id: str,
    files: list,
    status: str = "success",
    metadata: dict = None
):
    """
    Write run manifest and update latest pointer

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket name (e.g., 'usdcop')
        layer: Layer name (e.g., 'l4')
        run_id: Run identifier (e.g., '2025-10-20' or '20251020_083045')
        files: List of file metadata dicts
        status: 'success', 'failed', or 'running'
        metadata: Additional metadata dict

    Example:
        files = [
            {
                "name": "replay_dataset.parquet",
                "path": "l4/2025-10-20/replay_dataset.parquet",
                "size_bytes": 12345678,
                "row_count": 50000,
                "checksum": "sha256:abc123..."
            },
            {
                "name": "env_spec.json",
                "path": "l4/2025-10-20/env_spec.json",
                "size_bytes": 1234,
                "row_count": None,
                "checksum": "sha256:def456..."
            }
        ]
    """
    now = datetime.utcnow().isoformat() + "Z"

    # Calculate dataset hash (based on file checksums)
    files_str = json.dumps([f["checksum"] for f in files], sort_keys=True)
    dataset_hash = "sha256:" + hashlib.sha256(files_str.encode()).hexdigest()[:16]

    # Build run manifest
    manifest = {
        "run_id": run_id,
        "layer": layer,
        "path": f"{layer}/{run_id}/",
        "dataset_hash": dataset_hash,
        "started_at": metadata.get("started_at", now) if metadata else now,
        "completed_at": now,
        "status": status,
        "files": files,
        "metadata": metadata or {}
    }

    # Write run manifest
    run_key = f"_meta/{layer}_{run_id}_run.json"
    s3_client.put_object(
        Bucket=bucket,
        Key=run_key,
        Body=json.dumps(manifest, indent=2),
        ContentType="application/json"
    )

    print(f"✓ Written run manifest: s3://{bucket}/{run_key}")

    # Update latest pointer (only if success)
    if status == "success":
        latest = {
            "run_id": run_id,
            "layer": layer,
            "path": f"{layer}/{run_id}/",
            "dataset_hash": dataset_hash,
            "updated_at": now
        }

        latest_key = f"_meta/{layer}_latest.json"
        s3_client.put_object(
            Bucket=bucket,
            Key=latest_key,
            Body=json.dumps(latest, indent=2),
            ContentType="application/json"
        )

        print(f"✓ Updated latest pointer: s3://{bucket}/{latest_key}")

    return manifest


# ==========================================
# FILE METADATA HELPER
# ==========================================
def create_file_metadata(
    s3_client,
    bucket: str,
    key: str,
    row_count: int = None
) -> dict:
    """
    Create file metadata dict with size and checksum

    Args:
        s3_client: Boto3 S3 client
        bucket: S3 bucket
        key: S3 key (path)
        row_count: Number of rows (for parquet files)

    Returns:
        {
            "name": "filename.parquet",
            "path": "l4/2025-10-20/filename.parquet",
            "size_bytes": 12345,
            "row_count": 1000,
            "checksum": "sha256:abc..."
        }
    """
    # Get object metadata
    response = s3_client.head_object(Bucket=bucket, Key=key)
    size_bytes = response['ContentLength']
    etag = response['ETag'].strip('"')

    # Use ETag as checksum (MD5 for single-part uploads)
    checksum = f"md5:{etag}"

    return {
        "name": Path(key).name,
        "path": key,
        "size_bytes": size_bytes,
        "row_count": row_count,
        "checksum": checksum
    }


# ==========================================
# EXAMPLE USAGE IN AIRFLOW DAG
# ==========================================
def example_l4_dag_task():
    """
    Example of how L4 DAG would write manifest after processing
    """
    print("=" * 80)
    print("EXAMPLE: L4 DAG Task - Writing Manifest")
    print("=" * 80)

    s3_client = get_s3_client()
    bucket = "usdcop"
    layer = "l4"
    run_id = datetime.utcnow().strftime("%Y-%m-%d")

    # Simulate: DAG has written these files to MinIO
    output_files = [
        f"{layer}/{run_id}/replay_dataset.parquet",
        f"{layer}/{run_id}/env_spec.json",
        f"{layer}/{run_id}/reward_spec.json",
        f"{layer}/{run_id}/split_spec.json",
        f"{layer}/{run_id}/obs_clip_rates.json",
        f"{layer}/{run_id}/metadata.json"
    ]

    # Create file metadata for each
    files_metadata = []

    for key in output_files:
        # In real DAG: check if file actually exists
        try:
            s3_client.head_object(Bucket=bucket, Key=key)

            # For parquet files, you could read row count
            if key.endswith('.parquet'):
                # row_count = read_parquet_row_count(bucket, key)
                row_count = 50000  # Example
            else:
                row_count = None

            metadata = create_file_metadata(s3_client, bucket, key, row_count)
            files_metadata.append(metadata)

        except s3_client.exceptions.NoSuchKey:
            print(f"⚠ File not found (skipping): {key}")
            continue

    # Write manifest
    if files_metadata:
        manifest = write_manifest(
            s3_client=s3_client,
            bucket=bucket,
            layer=layer,
            run_id=run_id,
            files=files_metadata,
            status="success",
            metadata={
                "started_at": "2025-10-20T08:00:00Z",
                "pipeline": "usdcop_m5__05_l4_rlready",
                "airflow_dag_id": "usdcop_m5__05_l4_rlready",
                "airflow_run_id": "scheduled__2025-10-20T05:00:00+00:00",
                "total_episodes": 250,
                "train_episodes": 175,
                "val_episodes": 37,
                "test_episodes": 38
            }
        )

        print("\n✅ Manifest written successfully!")
        print(f"\nAPI can now read this data via:")
        print(f"  GET /api/pipeline/l4/contract")
        print(f"  GET /api/pipeline/l4/quality-check")
        print(f"  GET /api/pipeline/l4/quality-check?run_id={run_id}")

    else:
        print("❌ No files found to include in manifest")


# ==========================================
# INTEGRATION WITH AIRFLOW
# ==========================================
"""
In your Airflow DAG (e.g., usdcop_m5__05_l4_rlready.py):

from airflow import DAG
from airflow.operators.python import PythonOperator
from scripts.write_manifest_example import write_manifest, create_file_metadata
import boto3

def l4_processing(**context):
    # 1. Your existing L4 processing logic
    df = process_l4_data()

    # 2. Write outputs to MinIO
    output_path = f"l4/{run_id}/replay_dataset.parquet"
    df.to_parquet(f"s3://usdcop/{output_path}")

    # ... write other files (env_spec.json, etc.)

    # 3. Create manifest
    s3_client = boto3.client('s3', endpoint_url='http://minio:9000', ...)

    files = [
        create_file_metadata(s3_client, "usdcop", f"l4/{run_id}/replay_dataset.parquet", len(df)),
        create_file_metadata(s3_client, "usdcop", f"l4/{run_id}/env_spec.json"),
        # ... other files
    ]

    write_manifest(
        s3_client=s3_client,
        bucket="usdcop",
        layer="l4",
        run_id=run_id,
        files=files,
        status="success"
    )

    # 4. API now automatically discovers this run!

with DAG('usdcop_m5__05_l4_rlready', ...) as dag:

    process_task = PythonOperator(
        task_id='process_l4',
        python_callable=l4_processing
    )
"""


# ==========================================
# CLI USAGE
# ==========================================
if __name__ == "__main__":
    """
    Run this script to see example output:

    python scripts/write_manifest_example.py
    """
    example_l4_dag_task()

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    # Show how to read the manifest
    s3_client = get_s3_client()
    bucket = "usdcop"

    try:
        # Read latest.json
        response = s3_client.get_object(Bucket=bucket, Key="_meta/l4_latest.json")
        latest = json.loads(response['Body'].read())

        print("\nLatest L4 manifest:")
        print(json.dumps(latest, indent=2))

        # Read full run manifest
        run_id = latest["run_id"]
        response = s3_client.get_object(Bucket=bucket, Key=f"_meta/l4_{run_id}_run.json")
        run_manifest = json.loads(response['Body'].read())

        print(f"\nFull run manifest for {run_id}:")
        print(json.dumps(run_manifest, indent=2))

    except Exception as e:
        print(f"\n⚠ Could not read manifest (may not exist yet): {e}")
        print("\nTo create test data, first upload some files to MinIO:")
        print("  aws --endpoint-url http://localhost:9000 s3 cp test.parquet s3://usdcop/l4/2025-10-20/")
