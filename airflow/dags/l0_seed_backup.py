"""
DAG: l0_seed_backup
===================

Backup semanal de tablas críticas a MinIO para reproducibilidad.

Schedule: Domingos 02:00 UTC
Retention: 4 snapshots (1 mes)

Tables backed up:
- usdcop_m5_ohlcv (OHLCV 5-min data)
- macro_indicators_daily (Macro indicators)

Output:
- MinIO bucket: seeds/v{YYYY.MM.DD}/
- MinIO bucket: seeds/latest/ (symlink-like copy)

Contract: CTR-BACKUP-DAG-001
Version: 2.0.0
"""

import hashlib
import json
import os
import sys
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

# Add paths
sys.path.insert(0, '/opt/airflow')
sys.path.insert(0, '/opt/airflow/dags')

import pandas as pd
import psycopg2


# =============================================================================
# CONFIGURATION
# =============================================================================

DAG_ID = 'l0_seed_backup'

TABLES_TO_BACKUP = [
    {
        'name': 'usdcop_m5_ohlcv',
        'query': '''
            SELECT time, symbol, open, high, low, close, volume, source
            FROM usdcop_m5_ohlcv
            ORDER BY time
        ''',
        'date_column': 'time',
    },
    {
        'name': 'macro_indicators_daily',
        'query': 'SELECT * FROM macro_indicators_daily ORDER BY fecha',
        'date_column': 'fecha',
    },
]

DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'postgres'),
    'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
    'user': os.environ.get('POSTGRES_USER', 'admin'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'admin123'),
}

MINIO_CONFIG = {
    'endpoint': os.environ.get('MINIO_ENDPOINT', 'minio:9000'),
    'access_key': os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin'),
    'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY', 'minioadmin123'),
    'bucket': 'seeds',
    'secure': False,
}

RETENTION_COUNT = 4  # Keep 4 weekly snapshots


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def export_tables_to_parquet(**context) -> Dict[str, Any]:
    """Export tables to Parquet files."""
    import logging
    logger = logging.getLogger(__name__)

    version = datetime.utcnow().strftime('%Y.%m.%d')
    output_dir = Path(f'/tmp/seeds/v{version}')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating snapshot v{version}")

    conn = psycopg2.connect(**DB_CONFIG)
    tables_info = {}

    for table_config in TABLES_TO_BACKUP:
        table_name = table_config['name']
        query = table_config['query']
        date_col = table_config['date_column']

        logger.info(f"Exporting {table_name}...")

        try:
            df = pd.read_sql(query, conn)
            rows = len(df)

            if rows == 0:
                logger.warning(f"{table_name} is empty, skipping")
                continue

            # Date range
            date_range = None
            if date_col and date_col in df.columns:
                date_range = [
                    str(df[date_col].min())[:10],
                    str(df[date_col].max())[:10],
                ]

            # Export
            output_file = output_dir / f'{table_name}.parquet'
            df.to_parquet(output_file, compression='snappy', index=False)

            # Hash
            sha256 = hashlib.sha256()
            with open(output_file, 'rb') as f:
                for block in iter(lambda: f.read(4096), b''):
                    sha256.update(block)

            tables_info[table_name] = {
                'file': f'{table_name}.parquet',
                'rows': rows,
                'date_range': date_range,
                'size_bytes': output_file.stat().st_size,
                'sha256': sha256.hexdigest(),
            }

            logger.info(f"  {table_name}: {rows:,} rows exported")

        except Exception as e:
            logger.error(f"Failed to export {table_name}: {e}")
            raise

    conn.close()

    # Create manifest
    manifest = {
        'version': version,
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'created_by': DAG_ID,
        'tables': tables_info,
    }

    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)

    # Push to XCom
    context['ti'].xcom_push(key='version', value=version)
    context['ti'].xcom_push(key='output_dir', value=str(output_dir))
    context['ti'].xcom_push(key='manifest', value=manifest)

    return manifest


def upload_to_minio(**context) -> Dict[str, Any]:
    """Upload snapshot to MinIO."""
    import logging
    from minio import Minio

    logger = logging.getLogger(__name__)

    version = context['ti'].xcom_pull(key='version')
    output_dir = Path(context['ti'].xcom_pull(key='output_dir'))

    logger.info(f"Uploading v{version} to MinIO")

    client = Minio(
        MINIO_CONFIG['endpoint'],
        access_key=MINIO_CONFIG['access_key'],
        secret_key=MINIO_CONFIG['secret_key'],
        secure=MINIO_CONFIG['secure'],
    )

    bucket = MINIO_CONFIG['bucket']

    # Create bucket if not exists
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info(f"Created bucket: {bucket}")

    # Upload versioned files
    for file_path in output_dir.iterdir():
        object_name = f'v{version}/{file_path.name}'
        client.fput_object(bucket, object_name, str(file_path))
        logger.info(f"  Uploaded: {object_name}")

    # Update latest/
    for file_path in output_dir.iterdir():
        object_name = f'latest/{file_path.name}'
        client.fput_object(bucket, object_name, str(file_path))

    # Update LATEST_VERSION
    latest_file = output_dir / 'LATEST_VERSION'
    with open(latest_file, 'w') as f:
        f.write(version)
    client.fput_object(bucket, 'LATEST_VERSION', str(latest_file))

    logger.info(f"Updated seeds/latest/ to v{version}")

    return {'uploaded_version': version}


def cleanup_old_snapshots(**context) -> Dict[str, Any]:
    """Remove old snapshots beyond retention count."""
    import logging
    from minio import Minio

    logger = logging.getLogger(__name__)

    client = Minio(
        MINIO_CONFIG['endpoint'],
        access_key=MINIO_CONFIG['access_key'],
        secret_key=MINIO_CONFIG['secret_key'],
        secure=MINIO_CONFIG['secure'],
    )

    bucket = MINIO_CONFIG['bucket']

    # List all version prefixes
    versions = set()
    for obj in client.list_objects(bucket, prefix='v', recursive=False):
        # Extract version from path like 'v2026.01.22/'
        version = obj.object_name.strip('/').replace('v', '')
        if version and '.' in version:
            versions.add(version)

    # Sort by date (versions are YYYY.MM.DD)
    sorted_versions = sorted(versions, reverse=True)
    logger.info(f"Found {len(sorted_versions)} snapshots: {sorted_versions}")

    # Delete old versions
    deleted = []
    if len(sorted_versions) > RETENTION_COUNT:
        to_delete = sorted_versions[RETENTION_COUNT:]
        for version in to_delete:
            prefix = f'v{version}/'
            objects = list(client.list_objects(bucket, prefix=prefix, recursive=True))
            for obj in objects:
                client.remove_object(bucket, obj.object_name)
            deleted.append(version)
            logger.info(f"  Deleted snapshot: v{version}")

    logger.info(f"Cleanup complete. Deleted {len(deleted)} old snapshots")

    return {'deleted_versions': deleted, 'kept_versions': sorted_versions[:RETENTION_COUNT]}


def generate_report(**context) -> str:
    """Generate backup report."""
    import logging
    logger = logging.getLogger(__name__)

    manifest = context['ti'].xcom_pull(key='manifest')
    version = manifest['version']

    report = []
    report.append("=" * 60)
    report.append(f"SEED BACKUP REPORT - v{version}")
    report.append("=" * 60)
    report.append(f"Created: {manifest['created_at']}")
    report.append("")

    for table_name, info in manifest['tables'].items():
        report.append(f"Table: {table_name}")
        report.append(f"  Rows: {info['rows']:,}")
        report.append(f"  Date range: {info['date_range']}")
        report.append(f"  Size: {info['size_bytes'] / 1024 / 1024:.2f} MB")
        report.append("")

    report.append("=" * 60)
    report.append("Backup stored in:")
    report.append(f"  minio://seeds/v{version}/")
    report.append(f"  minio://seeds/latest/")

    report_text = '\n'.join(report)
    logger.info(report_text)

    return report_text


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'usdcop-trading',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Weekly backup of critical tables to MinIO',
    schedule_interval='0 2 * * 0',  # Sundays 02:00 UTC
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=['l0', 'backup', 'seeds', 'minio'],
) as dag:

    start = EmptyOperator(task_id='start')

    export = PythonOperator(
        task_id='export_tables_to_parquet',
        python_callable=export_tables_to_parquet,
        execution_timeout=timedelta(minutes=30),
    )

    upload = PythonOperator(
        task_id='upload_to_minio',
        python_callable=upload_to_minio,
        execution_timeout=timedelta(minutes=15),
    )

    cleanup = PythonOperator(
        task_id='cleanup_old_snapshots',
        python_callable=cleanup_old_snapshots,
        execution_timeout=timedelta(minutes=10),
    )

    report = PythonOperator(
        task_id='generate_report',
        python_callable=generate_report,
    )

    end = EmptyOperator(task_id='end')

    start >> export >> upload >> cleanup >> report >> end
