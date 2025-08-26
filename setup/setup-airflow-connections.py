#!/usr/bin/env python3
"""
Setup Airflow connections for USDCOP Trading Pipeline
"""

import subprocess
import json
import sys

def run_command(cmd):
    """Execute a command and return output"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

def setup_minio_connection():
    """Setup MinIO connection in Airflow"""
    
    print("Setting up MinIO connection...")
    
    # First, try to delete existing connection
    cmd_delete = 'docker exec airflow-webserver airflow connections delete minio_conn 2>/dev/null'
    run_command(cmd_delete)
    
    # Connection configuration for MinIO
    conn_config = {
        "aws_access_key_id": "minioadmin",
        "aws_secret_access_key": "minioadmin", 
        "endpoint_url": "http://airflow-minio:9000",
        "region_name": "us-east-1"
    }
    
    # Create the connection
    conn_extra = json.dumps(conn_config).replace('"', '\\"')
    
    cmd_add = f'''docker exec airflow-webserver airflow connections add minio_conn \
        --conn-type aws \
        --conn-extra '{conn_extra}' '''
    
    stdout, stderr, returncode = run_command(cmd_add)
    
    if returncode == 0:
        print("✅ MinIO connection created successfully")
        return True
    else:
        print(f"❌ Failed to create MinIO connection: {stderr}")
        return False

def setup_postgres_connection():
    """Setup PostgreSQL connection in Airflow"""
    
    print("Setting up PostgreSQL connection...")
    
    # Delete existing connection if any
    cmd_delete = 'docker exec airflow-webserver airflow connections delete postgres_trading 2>/dev/null'
    run_command(cmd_delete)
    
    # Create PostgreSQL connection
    cmd_add = '''docker exec airflow-webserver airflow connections add postgres_trading \
        --conn-type postgres \
        --conn-host airflow-postgres \
        --conn-login trading \
        --conn-password trading123 \
        --conn-schema trading_db \
        --conn-port 5432'''
    
    stdout, stderr, returncode = run_command(cmd_add)
    
    if returncode == 0:
        print("✅ PostgreSQL connection created successfully")
        return True
    else:
        print(f"❌ Failed to create PostgreSQL connection: {stderr}")
        return False

def verify_connections():
    """Verify all connections are working"""
    
    print("\nVerifying connections...")
    
    # Test MinIO connection with a Python script inside the container
    test_script = '''
import boto3
from botocore.client import Config

# MinIO configuration
s3_client = boto3.client(
    's3',
    endpoint_url='http://airflow-minio:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# List buckets
try:
    response = s3_client.list_buckets()
    print(f"Found {len(response['Buckets'])} buckets in MinIO")
    for bucket in response['Buckets'][:3]:
        print(f"  - {bucket['Name']}")
    print("MinIO connection: OK")
except Exception as e:
    print(f"MinIO connection: FAILED - {e}")
'''
    
    # Execute test inside container
    cmd = f'''docker exec airflow-webserver python -c "{test_script}"'''
    stdout, stderr, returncode = run_command(cmd)
    
    if stdout:
        print(stdout)
    if stderr and "WARNING" not in stderr:
        print(f"Error: {stderr}")
    
    return returncode == 0

def main():
    """Main setup function"""
    
    print("="*60)
    print("AIRFLOW CONNECTIONS SETUP")
    print("="*60)
    
    # Setup connections
    minio_ok = setup_minio_connection()
    postgres_ok = setup_postgres_connection()
    
    # Verify
    verify_ok = verify_connections()
    
    print("\n" + "="*60)
    print("SETUP SUMMARY")
    print("="*60)
    print(f"MinIO Connection: {'✅ OK' if minio_ok else '❌ FAILED'}")
    print(f"PostgreSQL Connection: {'✅ OK' if postgres_ok else '❌ FAILED'}")
    print(f"Connection Test: {'✅ PASSED' if verify_ok else '❌ FAILED'}")
    
    if minio_ok and postgres_ok:
        print("\n✅ All connections configured successfully!")
        print("\nYou can now:")
        print("1. Trigger DAG execution from Airflow UI")
        print("2. Monitor pipeline progress in real-time")
        print("3. Check data in MinIO buckets")
        return 0
    else:
        print("\n❌ Some connections failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())