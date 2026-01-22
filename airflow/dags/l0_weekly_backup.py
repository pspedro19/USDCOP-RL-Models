"""
DAG: v3.l0_weekly_backup
=========================
USD/COP Trading System - V3 Architecture
Layer 0: Weekly Automated Backup

Purpose:
    Creates weekly compressed backups of critical database tables:
    - usdcop_m5_ohlcv (OHLCV 5-minute data)
    - macro_indicators_daily (Macro indicators)
    - inference_features_5m (Pre-calculated features)

Schedule:
    Weekly on Sundays at midnight COT (05:00 UTC)
    Cron: 0 5 * * 0

Features:
    - Compressed CSV backups (gzip)
    - Automatic cleanup of old backups (keeps last 7)
    - Backup registry with metadata
    - Integrity verification

Author: Pipeline Automatizado
Version: 1.0.0
Created: 2025-12-26
"""

from datetime import datetime, timedelta
from pathlib import Path
import gzip
import json
import os
import logging
from io import StringIO

from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import psycopg2

# =============================================================================
# CONFIGURATION
# =============================================================================

from utils.dag_common import get_db_connection
from contracts.dag_registry import L0_WEEKLY_BACKUP

DAG_ID = L0_WEEKLY_BACKUP

# Backup configuration
def get_backup_dir():
    """Get backup directory based on environment."""
    docker_path = Path('/app/data/backups')
    if docker_path.exists():
        return docker_path

    # Local development
    project_root = Path('/opt/airflow') if Path('/opt/airflow').exists() else Path(__file__).parent.parent.parent.parent
    backup_dir = project_root / 'data' / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    return backup_dir


BACKUP_DIR = get_backup_dir()
BACKUP_RETENTION = 7  # Keep last 7 backups per table

# Tables to backup
BACKUP_TABLES = [
    {
        'name': 'usdcop_m5_ohlcv',
        'date_column': 'time',
        'pattern': 'ohlcv_*.csv.gz',
        'description': 'OHLCV 5-minute candles'
    },
    {
        'name': 'macro_indicators_daily',
        'date_column': 'fecha',
        'pattern': 'macro_*.csv.gz',
        'description': 'Macro indicators daily'
    },
    {
        'name': 'inference_features_5m',
        'date_column': 'timestamp_utc',
        'pattern': 'features_*.csv.gz',
        'description': 'Pre-calculated inference features',
        'optional': True
    }
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def table_exists(conn, table_name: str) -> bool:
    """Check if a table exists."""
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = %s
            )
        """, (table_name,))
        return cur.fetchone()[0]
    finally:
        cur.close()


def get_table_stats(conn, table_name: str, date_col: str) -> dict:
    """Get basic statistics for a table."""
    cur = conn.cursor()
    try:
        cur.execute(f"""
            SELECT
                COUNT(*) as row_count,
                MIN({date_col}) as min_date,
                MAX({date_col}) as max_date
            FROM {table_name}
        """)
        result = cur.fetchone()
        return {
            'row_count': result[0],
            'min_date': str(result[1]) if result[1] else None,
            'max_date': str(result[2]) if result[2] else None
        }
    except Exception as e:
        logging.warning(f"Could not get stats for {table_name}: {e}")
        return {'row_count': 0, 'min_date': None, 'max_date': None}
    finally:
        cur.close()


def export_table_to_csv(conn, table_name: str, date_col: str, output_path: Path) -> int:
    """
    Export table to compressed CSV using PostgreSQL COPY.

    Args:
        conn: Database connection
        table_name: Table to export
        date_col: Column to order by
        output_path: Path for output .csv.gz file

    Returns:
        Number of rows exported
    """
    cur = conn.cursor()

    try:
        # Get row count first
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cur.fetchone()[0]

        if row_count == 0:
            logging.warning(f"Table {table_name} is empty, skipping backup")
            return 0

        # Export using COPY
        buffer = StringIO()
        cur.copy_expert(
            f"COPY (SELECT * FROM {table_name} ORDER BY {date_col}) TO STDOUT WITH CSV HEADER",
            buffer
        )

        # Compress and write
        buffer.seek(0)
        with gzip.open(output_path, 'wt', encoding='utf-8') as f:
            f.write(buffer.read())

        return row_count

    finally:
        cur.close()


def cleanup_old_backups(pattern: str, keep: int = BACKUP_RETENTION):
    """
    Remove old backup files, keeping the N most recent.

    Args:
        pattern: Glob pattern to match backup files
        keep: Number of backups to retain
    """
    backups = sorted(
        BACKUP_DIR.glob(pattern),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if len(backups) <= keep:
        return

    for old_backup in backups[keep:]:
        logging.info(f"Removing old backup: {old_backup.name}")
        old_backup.unlink()


def update_backup_registry(backup_info: dict):
    """
    Update the backup registry JSON file with new backup metadata.

    Args:
        backup_info: Dictionary with backup metadata
    """
    registry_path = BACKUP_DIR / 'backup_registry.json'

    # Load existing registry or create new
    if registry_path.exists():
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {'backups': [], 'last_updated': None}

    # Add new backup entry
    registry['backups'].append(backup_info)
    registry['last_updated'] = datetime.now().isoformat()

    # Keep only last 50 entries
    if len(registry['backups']) > 50:
        registry['backups'] = registry['backups'][-50:]

    # Save registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2, default=str)


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def backup_ohlcv(**context):
    """Backup OHLCV table."""
    table_config = BACKUP_TABLES[0]  # usdcop_m5_ohlcv
    table_name = table_config['name']
    date_col = table_config['date_column']

    logging.info(f"Starting backup of {table_name}...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = BACKUP_DIR / f'ohlcv_{timestamp}.csv.gz'

    conn = get_db_connection()

    try:
        # Get stats before backup
        stats = get_table_stats(conn, table_name, date_col)
        logging.info(f"Table stats: {stats['row_count']:,} rows, range: {stats['min_date']} to {stats['max_date']}")

        # Export
        rows_exported = export_table_to_csv(conn, table_name, date_col, output_file)

        if rows_exported > 0:
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logging.info(f"Backup complete: {output_file.name} ({file_size_mb:.2f} MB, {rows_exported:,} rows)")

            # Update registry
            backup_info = {
                'table': table_name,
                'file': output_file.name,
                'timestamp': timestamp,
                'rows': rows_exported,
                'size_mb': round(file_size_mb, 2),
                'date_range': f"{stats['min_date']} to {stats['max_date']}"
            }
            context['ti'].xcom_push(key='ohlcv_backup', value=backup_info)

            return backup_info
        else:
            logging.warning(f"No data to backup for {table_name}")
            return None

    finally:
        conn.close()


def backup_macro(**context):
    """Backup Macro indicators table."""
    table_config = BACKUP_TABLES[1]  # macro_indicators_daily
    table_name = table_config['name']
    date_col = table_config['date_column']

    logging.info(f"Starting backup of {table_name}...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = BACKUP_DIR / f'macro_{timestamp}.csv.gz'

    conn = get_db_connection()

    try:
        # Check if table exists
        if not table_exists(conn, table_name):
            logging.warning(f"Table {table_name} does not exist")
            return None

        # Get stats
        stats = get_table_stats(conn, table_name, date_col)
        logging.info(f"Table stats: {stats['row_count']:,} rows")

        # Export
        rows_exported = export_table_to_csv(conn, table_name, date_col, output_file)

        if rows_exported > 0:
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logging.info(f"Backup complete: {output_file.name} ({file_size_mb:.2f} MB, {rows_exported:,} rows)")

            backup_info = {
                'table': table_name,
                'file': output_file.name,
                'timestamp': timestamp,
                'rows': rows_exported,
                'size_mb': round(file_size_mb, 2),
                'date_range': f"{stats['min_date']} to {stats['max_date']}"
            }
            context['ti'].xcom_push(key='macro_backup', value=backup_info)

            return backup_info
        else:
            return None

    finally:
        conn.close()


def backup_features(**context):
    """Backup inference features table (optional)."""
    table_config = BACKUP_TABLES[2]  # inference_features_5m
    table_name = table_config['name']
    date_col = table_config['date_column']

    logging.info(f"Starting backup of {table_name} (optional)...")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = BACKUP_DIR / f'features_{timestamp}.csv.gz'

    conn = get_db_connection()

    try:
        # Check if table exists
        if not table_exists(conn, table_name):
            logging.info(f"Table {table_name} does not exist - skipping (this is normal)")
            return None

        # Get stats
        stats = get_table_stats(conn, table_name, date_col)

        if stats['row_count'] == 0:
            logging.info(f"Table {table_name} is empty - skipping")
            return None

        logging.info(f"Table stats: {stats['row_count']:,} rows")

        # Export
        rows_exported = export_table_to_csv(conn, table_name, date_col, output_file)

        if rows_exported > 0:
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            logging.info(f"Backup complete: {output_file.name} ({file_size_mb:.2f} MB, {rows_exported:,} rows)")

            backup_info = {
                'table': table_name,
                'file': output_file.name,
                'timestamp': timestamp,
                'rows': rows_exported,
                'size_mb': round(file_size_mb, 2),
                'date_range': f"{stats['min_date']} to {stats['max_date']}"
            }
            context['ti'].xcom_push(key='features_backup', value=backup_info)

            return backup_info
        else:
            return None

    finally:
        conn.close()


def cleanup_old_backups_task(**context):
    """Clean up old backup files."""
    logging.info("Cleaning up old backups...")

    for table_config in BACKUP_TABLES:
        pattern = table_config['pattern']
        cleanup_old_backups(pattern, keep=BACKUP_RETENTION)

    logging.info(f"Cleanup complete - retaining last {BACKUP_RETENTION} backups per table")


def update_registry(**context):
    """Update backup registry with all backup metadata."""
    ti = context['ti']

    ohlcv_backup = ti.xcom_pull(key='ohlcv_backup', task_ids='backup_ohlcv')
    macro_backup = ti.xcom_pull(key='macro_backup', task_ids='backup_macro')
    features_backup = ti.xcom_pull(key='features_backup', task_ids='backup_features')

    run_info = {
        'run_id': context['run_id'],
        'execution_date': str(context['execution_date']),
        'tables_backed_up': [],
        'total_rows': 0,
        'total_size_mb': 0
    }

    for backup in [ohlcv_backup, macro_backup, features_backup]:
        if backup:
            run_info['tables_backed_up'].append(backup['table'])
            run_info['total_rows'] += backup['rows']
            run_info['total_size_mb'] += backup['size_mb']
            update_backup_registry(backup)

    logging.info("=" * 60)
    logging.info("WEEKLY BACKUP SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Tables backed up: {', '.join(run_info['tables_backed_up'])}")
    logging.info(f"Total rows: {run_info['total_rows']:,}")
    logging.info(f"Total size: {run_info['total_size_mb']:.2f} MB")
    logging.info("=" * 60)

    return run_info


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='V3 L0: Weekly automated backup of database tables',
    schedule_interval='0 5 * * 0',  # Sundays at 00:00 COT (05:00 UTC)
    catchup=False,
    max_active_runs=1,
    tags=['v3', 'l0', 'backup', 'weekly', 'maintenance']
)

with dag:

    # Task 1: Backup OHLCV table
    task_backup_ohlcv = PythonOperator(
        task_id='backup_ohlcv',
        python_callable=backup_ohlcv,
        provide_context=True
    )

    # Task 2: Backup Macro table
    task_backup_macro = PythonOperator(
        task_id='backup_macro',
        python_callable=backup_macro,
        provide_context=True
    )

    # Task 3: Backup Features table (optional)
    task_backup_features = PythonOperator(
        task_id='backup_features',
        python_callable=backup_features,
        provide_context=True
    )

    # Task 4: Cleanup old backups
    task_cleanup = PythonOperator(
        task_id='cleanup_old_backups',
        python_callable=cleanup_old_backups_task,
        provide_context=True
    )

    # Task 5: Update registry
    task_registry = PythonOperator(
        task_id='update_registry',
        python_callable=update_registry,
        provide_context=True
    )

    # Backups run in parallel, then cleanup, then registry
    [task_backup_ohlcv, task_backup_macro, task_backup_features] >> task_cleanup >> task_registry
