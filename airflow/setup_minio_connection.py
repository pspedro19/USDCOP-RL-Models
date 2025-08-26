#!/usr/bin/env python3
"""
Setup MinIO connection in Airflow
This script creates the necessary connection for Airflow to communicate with MinIO
"""

import json
import subprocess
import sys

def create_minio_connection():
    """Create MinIO connection in Airflow"""
    
    # MinIO connection configuration
    minio_conn = {
        "conn_type": "aws",
        "host": "http://minio:9000",  # Internal Docker network endpoint
        "login": "minioadmin",  # MinIO access key
        "password": "minioadmin123",  # MinIO secret key
        "extra": json.dumps({
            "endpoint_url": "http://minio:9000",
            "aws_access_key_id": "minioadmin",
            "aws_secret_access_key": "minioadmin123",
            "region_name": "us-east-1",  # Default region for MinIO
            "signature_version": "s3v4",
            "config": {
                "retries": {
                    "max_attempts": 3,
                    "mode": "standard"
                }
            }
        })
    }
    
    # Build the airflow CLI command (without -it for Windows)
    cmd = [
        "docker", "exec", "trading-airflow-webserver",
        "airflow", "connections", "add", "minio_conn",
        "--conn-type", minio_conn["conn_type"],
        "--conn-host", minio_conn["host"],
        "--conn-login", minio_conn["login"],
        "--conn-password", minio_conn["password"],
        "--conn-extra", minio_conn["extra"]
    ]
    
    try:
        # First, try to delete existing connection if it exists
        delete_cmd = [
            "docker", "exec", "trading-airflow-webserver",
            "airflow", "connections", "delete", "minio_conn"
        ]
        subprocess.run(delete_cmd, capture_output=True, text=True)
        print("Removed existing MinIO connection (if any)")
        
        # Create new connection
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("[SUCCESS] MinIO connection created successfully!")
        print(result.stdout)
        
        # Test the connection
        test_cmd = [
            "docker", "exec", "trading-airflow-webserver",
            "airflow", "connections", "get", "minio_conn"
        ]
        test_result = subprocess.run(test_cmd, capture_output=True, text=True)
        print("\n[INFO] Connection details:")
        print(test_result.stdout)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Error creating MinIO connection: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_bucket_if_not_exists():
    """Create the required MinIO bucket if it doesn't exist"""
    
    bucket_name = "00-raw-usdcop-marketdata"
    
    # Check if bucket exists
    check_cmd = [
        "docker", "exec", "trading-minio",
        "mc", "ls", "local/" + bucket_name
    ]
    
    result = subprocess.run(check_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        # Bucket doesn't exist, create it
        create_cmd = [
            "docker", "exec", "trading-minio",
            "mc", "mb", "local/" + bucket_name
        ]
        
        try:
            subprocess.run(create_cmd, capture_output=True, text=True, check=True)
            print(f"[SUCCESS] Created bucket: {bucket_name}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error creating bucket: {e}")
            return False
    else:
        print(f"[SUCCESS] Bucket already exists: {bucket_name}")
    
    # Set bucket policy to allow read/write
    policy_cmd = [
        "docker", "exec", "trading-minio",
        "mc", "anonymous", "set", "download", "local/" + bucket_name
    ]
    
    try:
        subprocess.run(policy_cmd, capture_output=True, text=True)
        print(f"[SUCCESS] Set bucket policy for: {bucket_name}")
    except:
        pass  # Policy setting is optional
    
    return True

def test_minio_connection():
    """Test MinIO connection from Airflow"""
    
    test_script = """
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import json
from datetime import datetime

try:
    # Initialize S3Hook with MinIO connection
    s3_hook = S3Hook(aws_conn_id='minio_conn')
    
    # Test: List buckets
    buckets = s3_hook.get_conn().list_buckets()
    print("[SUCCESS] Connection successful!")
    print(f"Available buckets: {[b['Name'] for b in buckets['Buckets']]}")
    
    # Test: Write a test file
    test_data = {
        'test': 'MinIO connection working',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    s3_hook.load_string(
        string_data=json.dumps(test_data),
        key='_test/connection_test.json',
        bucket_name='00-raw-usdcop-marketdata',
        replace=True
    )
    print("[SUCCESS] Successfully wrote test file to MinIO")
    
    # Test: Read the test file back
    content = s3_hook.read_key(
        key='_test/connection_test.json',
        bucket_name='00-raw-usdcop-marketdata'
    )
    print(f"[SUCCESS] Successfully read test file: {content}")
    
except Exception as e:
    print(f"[ERROR] Connection test failed: {e}")
    """
    
    # Save test script to temp file
    with open('/tmp/test_minio.py', 'w') as f:
        f.write(test_script)
    
    # Copy to container and execute
    copy_cmd = [
        "docker", "cp", "/tmp/test_minio.py",
        "trading-airflow-webserver:/tmp/test_minio.py"
    ]
    subprocess.run(copy_cmd, capture_output=True)
    
    exec_cmd = [
        "docker", "exec", "trading-airflow-webserver",
        "python", "/tmp/test_minio.py"
    ]
    
    result = subprocess.run(exec_cmd, capture_output=True, text=True)
    print("\n[TEST] Connection test results:")
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def main():
    print("Setting up MinIO connection for Airflow")
    print("=" * 50)
    
    # Step 1: Create MinIO connection in Airflow
    if not create_minio_connection():
        print("Failed to create MinIO connection")
        sys.exit(1)
    
    # Step 2: Create bucket if it doesn't exist
    if not create_bucket_if_not_exists():
        print("Failed to create bucket")
        sys.exit(1)
    
    # Step 3: Test the connection
    test_minio_connection()
    
    print("\n" + "=" * 50)
    print("[SUCCESS] MinIO setup complete!")
    print("\nConnection details:")
    print("  - Connection ID: minio_conn")
    print("  - Endpoint: http://minio:9000")
    print("  - Bucket: 00-raw-usdcop-marketdata")
    print("  - Access Key: minioadmin")
    print("\nYou can now run the L0 pipeline and data will be saved to MinIO")

if __name__ == "__main__":
    main()