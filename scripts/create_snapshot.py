#!/usr/bin/env python3
"""
Create Snapshot of Critical Tables
==================================

Exports critical database tables to Parquet format for reproducible deployments.

Usage:
    python create_snapshot.py --version 2026.01.22
    python create_snapshot.py --version $(date +%Y.%m.%d) --upload-to-minio

Contract: CTR-BACKUP-001
Version: 2.0.0
"""

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add paths for imports
sys.path.insert(0, '/opt/airflow')
sys.path.insert(0, '/opt/airflow/dags')

import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor


# =============================================================================
# CONFIGURATION
# =============================================================================

CRITICAL_TABLES = [
    {
        'name': 'usdcop_m5_ohlcv',
        'query': '''
            SELECT time, symbol, open, high, low, close, volume, source
            FROM usdcop_m5_ohlcv
            ORDER BY time
        ''',
        'date_column': 'time',
        'required': True,
    },
    {
        'name': 'macro_indicators_daily',
        'query': '''
            SELECT *
            FROM macro_indicators_daily
            ORDER BY fecha
        ''',
        'date_column': 'fecha',
        'required': True,
    },
    {
        'name': 'trades_history',
        'query': '''
            SELECT *
            FROM trades_history
            ORDER BY id
        ''',
        'date_column': None,
        'required': False,
    },
]

DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'postgres'),
    'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
    'user': os.environ.get('POSTGRES_USER', 'admin'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'admin123'),
}


# =============================================================================
# FUNCTIONS
# =============================================================================

def get_db_connection():
    """Get database connection with retry logic."""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            print(f"  Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(5)
            else:
                raise


def calculate_sha256(filepath: Path) -> str:
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def export_table(conn, table_config: dict, output_dir: Path) -> dict:
    """Export a single table to Parquet format."""
    table_name = table_config['name']
    query = table_config['query']
    date_column = table_config['date_column']
    required = table_config['required']

    print(f"\n[{table_name}] Exporting...")

    try:
        # Read data
        df = pd.read_sql(query, conn)
        rows = len(df)

        if rows == 0:
            if required:
                raise ValueError(f"Table {table_name} is empty but marked as required")
            print(f"  WARNING: Table is empty, skipping")
            return None

        # Calculate date range
        date_range = None
        if date_column and date_column in df.columns:
            min_date = df[date_column].min()
            max_date = df[date_column].max()
            date_range = [str(min_date)[:10], str(max_date)[:10]]

        # Export to Parquet
        output_file = output_dir / f"{table_name}.parquet"
        df.to_parquet(output_file, compression='snappy', index=False)

        # Calculate hash
        file_size = output_file.stat().st_size
        sha256 = calculate_sha256(output_file)

        print(f"  Rows: {rows:,}")
        print(f"  Date range: {date_range}")
        print(f"  File size: {file_size / 1024 / 1024:.2f} MB")
        print(f"  SHA256: {sha256[:16]}...")

        return {
            'file': f"{table_name}.parquet",
            'rows': rows,
            'date_range': date_range,
            'size_bytes': file_size,
            'sha256': sha256,
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        if required:
            raise
        return None


def create_manifest(version: str, tables_info: dict, output_dir: Path) -> dict:
    """Create manifest.json with snapshot metadata."""
    manifest = {
        'version': version,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'created_by': 'create_snapshot.py',
        'tables': tables_info,
        'compatibility': {
            'min_schema_version': '2.0',
            'postgres_version': '15',
            'timescaledb_version': '2.x',
        },
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def upload_to_minio(output_dir: Path, version: str):
    """Upload snapshot to MinIO."""
    try:
        from minio import Minio

        minio_client = Minio(
            os.environ.get('MINIO_ENDPOINT', 'minio:9000'),
            access_key=os.environ.get('MINIO_ACCESS_KEY', 'minioadmin'),
            secret_key=os.environ.get('MINIO_SECRET_KEY', 'minioadmin'),
            secure=False,
        )

        bucket = 'seeds'

        # Ensure bucket exists
        if not minio_client.bucket_exists(bucket):
            minio_client.make_bucket(bucket)
            print(f"  Created bucket: {bucket}")

        # Upload files
        for file_path in output_dir.iterdir():
            object_name = f"v{version}/{file_path.name}"
            minio_client.fput_object(bucket, object_name, str(file_path))
            print(f"  Uploaded: {object_name}")

        # Update 'latest' reference
        latest_file = output_dir / 'LATEST_VERSION'
        with open(latest_file, 'w') as f:
            f.write(version)
        minio_client.fput_object(bucket, 'LATEST_VERSION', str(latest_file))

        # Copy files to latest/
        for file_path in output_dir.iterdir():
            if file_path.name != 'LATEST_VERSION':
                object_name = f"latest/{file_path.name}"
                minio_client.fput_object(bucket, object_name, str(file_path))

        print(f"\n  Uploaded to minio://{bucket}/v{version}/")
        print(f"  Updated minio://{bucket}/latest/")

    except ImportError:
        print("  WARNING: minio package not installed, skipping upload")
    except Exception as e:
        print(f"  WARNING: MinIO upload failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Create snapshot of critical tables')
    parser.add_argument('--version', required=True, help='Snapshot version (e.g., 2026.01.22)')
    parser.add_argument('--output-dir', default='/opt/airflow/seeds', help='Output directory')
    parser.add_argument('--upload-to-minio', action='store_true', help='Upload to MinIO after export')
    args = parser.parse_args()

    print('=' * 70)
    print(f'CREATING SNAPSHOT v{args.version}')
    print('=' * 70)

    # Create output directory
    output_dir = Path(args.output_dir) / f"v{args.version}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")

    # Connect to database
    print("\nConnecting to database...")
    conn = get_db_connection()
    print("  Connected")

    # Export tables
    tables_info = {}
    for table_config in CRITICAL_TABLES:
        result = export_table(conn, table_config, output_dir)
        if result:
            tables_info[table_config['name']] = result

    conn.close()

    # Create manifest
    print("\nCreating manifest...")
    manifest = create_manifest(args.version, tables_info, output_dir)
    print(f"  Created manifest.json")

    # Upload to MinIO
    if args.upload_to_minio:
        print("\nUploading to MinIO...")
        upload_to_minio(output_dir, args.version)

    # Summary
    print('\n' + '=' * 70)
    print('SNAPSHOT COMPLETE')
    print('=' * 70)
    print(f"\nVersion: {args.version}")
    print(f"Location: {output_dir}")
    print(f"Tables exported: {len(tables_info)}")
    for name, info in tables_info.items():
        print(f"  - {name}: {info['rows']:,} rows ({info['size_bytes'] / 1024 / 1024:.2f} MB)")

    return manifest


if __name__ == '__main__':
    main()
