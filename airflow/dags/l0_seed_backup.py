"""
DAG: core_l0_05_seed_backup
============================
USD/COP Trading System - L0 Automated Seed Backup

Purpose:
    Daily backup of critical DB tables (OHLCV 5-min + macro daily) to parquet
    files under data/backups/seeds/. These backups are read by init-scripts on
    docker-compose up, eliminating the data gap between last seed export and now.

Schedule:
    0 18 * * * (18:00 UTC = 13:00 COT, after market close + all L0 updates)

Flow:
    health_check -> export_ohlcv_backup -> export_macro_backup
                 -> write_manifest -> validate_backups

Output files (inside /opt/airflow/data/backups/seeds/):
    usdcop_m5_ohlcv_backup.parquet
    macro_indicators_daily_backup.parquet
    backup_manifest.json

Author: Pedro @ Lean Tech Solutions
Version: 1.0.0
Created: 2026-02-17
Contract: CTR-L0-SEED-BACKUP-001
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import json
import hashlib
import logging
import os
from pathlib import Path

from utils.dag_common import get_db_connection
from contracts.dag_registry import CORE_L0_SEED_BACKUP, get_dag_tags

# =============================================================================
# CONFIGURATION
# =============================================================================

DAG_ID = CORE_L0_SEED_BACKUP
BACKUP_DIR = Path('/opt/airflow/data/backups/seeds')


# =============================================================================
# TASK FUNCTIONS
# =============================================================================

def health_check(**context):
    """Verify DB connection and that both tables have data."""
    logging.info("=" * 60)
    logging.info("SEED BACKUP - HEALTH CHECK")
    logging.info("=" * 60)

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Check OHLCV
        cur.execute("SELECT COUNT(*) FROM usdcop_m5_ohlcv")
        ohlcv_count = cur.fetchone()[0]
        logging.info(f"usdcop_m5_ohlcv: {ohlcv_count:,} rows")

        if ohlcv_count == 0:
            raise ValueError("usdcop_m5_ohlcv is empty — nothing to back up")

        # Check macro
        cur.execute("SELECT COUNT(*) FROM macro_indicators_daily")
        macro_count = cur.fetchone()[0]
        logging.info(f"macro_indicators_daily: {macro_count:,} rows")

        if macro_count == 0:
            logging.warning("macro_indicators_daily is empty — will skip macro backup")

        return {'ohlcv_count': ohlcv_count, 'macro_count': macro_count}

    finally:
        cur.close()
        conn.close()


def export_ohlcv_backup(**context):
    """Export full OHLCV table to parquet with atomic write."""
    logging.info("=" * 60)
    logging.info("EXPORTING OHLCV BACKUP")
    logging.info("=" * 60)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT time, symbol, open, high, low, close, volume, source
            FROM usdcop_m5_ohlcv
            ORDER BY symbol, time
        """)
        rows = cur.fetchall()

        if not rows:
            raise ValueError("No OHLCV data returned from query")

        df = pd.DataFrame(rows, columns=[
            'time', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'source'
        ])

        # Ensure tz-aware America/Bogota timestamps
        df['time'] = pd.to_datetime(df['time'], utc=True).dt.tz_convert('America/Bogota')

        # Per-symbol counts
        for sym in df['symbol'].unique():
            count = (df['symbol'] == sym).sum()
            logging.info(f"  {sym}: {count:,} rows")

        # Atomic write: write to .tmp then rename
        final_path = BACKUP_DIR / 'usdcop_m5_ohlcv_backup.parquet'
        tmp_path = BACKUP_DIR / 'usdcop_m5_ohlcv_backup.parquet.tmp'

        df.to_parquet(tmp_path, index=False)
        tmp_path.rename(final_path)

        size_bytes = final_path.stat().st_size
        logging.info(f"Exported {len(df):,} rows -> {final_path} ({size_bytes:,} bytes)")

        # Compute file hash for manifest
        sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()

        symbols = sorted(df['symbol'].unique().tolist())
        date_min = str(df['time'].min().date())
        date_max = str(df['time'].max().date())

        return {
            'rows': len(df),
            'symbols': symbols,
            'date_range': [date_min, date_max],
            'size_bytes': size_bytes,
            'sha256': sha256,
        }

    finally:
        cur.close()
        conn.close()


def export_macro_backup(**context):
    """Export full macro_indicators_daily table to parquet with atomic write."""
    logging.info("=" * 60)
    logging.info("EXPORTING MACRO BACKUP")
    logging.info("=" * 60)

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get column names
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'macro_indicators_daily'
            ORDER BY ordinal_position
        """)
        columns = [row[0] for row in cur.fetchall()]

        cur.execute("SELECT * FROM macro_indicators_daily ORDER BY fecha")
        rows = cur.fetchall()

        if not rows:
            logging.warning("No macro data returned — skipping macro backup")
            return {'rows': 0, 'skipped': True}

        df = pd.DataFrame(rows, columns=columns)

        # Atomic write
        final_path = BACKUP_DIR / 'macro_indicators_daily_backup.parquet'
        tmp_path = BACKUP_DIR / 'macro_indicators_daily_backup.parquet.tmp'

        df.to_parquet(tmp_path, index=False)
        tmp_path.rename(final_path)

        size_bytes = final_path.stat().st_size
        logging.info(f"Exported {len(df):,} rows -> {final_path} ({size_bytes:,} bytes)")

        sha256 = hashlib.sha256(final_path.read_bytes()).hexdigest()

        date_min = str(df['fecha'].min())
        date_max = str(df['fecha'].max())

        return {
            'rows': len(df),
            'date_range': [date_min, date_max],
            'size_bytes': size_bytes,
            'sha256': sha256,
        }

    finally:
        cur.close()
        conn.close()


def write_manifest(**context):
    """Write backup_manifest.json with timestamp, row counts, and file hashes."""
    logging.info("=" * 60)
    logging.info("WRITING BACKUP MANIFEST")
    logging.info("=" * 60)

    ti = context['ti']
    ohlcv_result = ti.xcom_pull(task_ids='export_ohlcv_backup')
    macro_result = ti.xcom_pull(task_ids='export_macro_backup')

    manifest = {
        'backup_timestamp': datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ'),
        'ohlcv': {
            'file': 'usdcop_m5_ohlcv_backup.parquet',
            'rows': ohlcv_result.get('rows', 0),
            'symbols': ohlcv_result.get('symbols', []),
            'date_range': ohlcv_result.get('date_range', []),
            'size_bytes': ohlcv_result.get('size_bytes', 0),
            'sha256': ohlcv_result.get('sha256', ''),
        },
        'macro': {
            'file': 'macro_indicators_daily_backup.parquet',
            'rows': macro_result.get('rows', 0),
            'date_range': macro_result.get('date_range', []),
            'size_bytes': macro_result.get('size_bytes', 0),
            'sha256': macro_result.get('sha256', ''),
            'skipped': macro_result.get('skipped', False),
        },
    }

    manifest_path = BACKUP_DIR / 'backup_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logging.info(f"Manifest written: {manifest_path}")
    logging.info(f"  OHLCV: {manifest['ohlcv']['rows']:,} rows")
    logging.info(f"  Macro: {manifest['macro']['rows']:,} rows")

    return manifest


def validate_backups(**context):
    """Re-read parquets and verify row counts match manifest."""
    logging.info("=" * 60)
    logging.info("VALIDATING BACKUPS")
    logging.info("=" * 60)

    manifest_path = BACKUP_DIR / 'backup_manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    errors = []

    # Validate OHLCV
    ohlcv_path = BACKUP_DIR / manifest['ohlcv']['file']
    if ohlcv_path.exists():
        df = pd.read_parquet(ohlcv_path)
        expected = manifest['ohlcv']['rows']
        actual = len(df)
        if actual != expected:
            errors.append(f"OHLCV row mismatch: expected {expected}, got {actual}")
        if ohlcv_path.stat().st_size == 0:
            errors.append("OHLCV backup file is empty (0 bytes)")
        logging.info(f"OHLCV: {actual:,} rows (expected {expected:,}) — OK")
    else:
        errors.append(f"OHLCV backup file missing: {ohlcv_path}")

    # Validate macro
    if not manifest['macro'].get('skipped', False):
        macro_path = BACKUP_DIR / manifest['macro']['file']
        if macro_path.exists():
            df = pd.read_parquet(macro_path)
            expected = manifest['macro']['rows']
            actual = len(df)
            if actual != expected:
                errors.append(f"Macro row mismatch: expected {expected}, got {actual}")
            if macro_path.stat().st_size == 0:
                errors.append("Macro backup file is empty (0 bytes)")
            logging.info(f"Macro: {actual:,} rows (expected {expected:,}) — OK")
        else:
            errors.append(f"Macro backup file missing: {macro_path}")
    else:
        logging.info("Macro backup was skipped (empty table)")

    if errors:
        for err in errors:
            logging.error(f"VALIDATION FAILED: {err}")
        raise ValueError(f"Backup validation failed: {'; '.join(errors)}")

    logging.info("All backup validations PASSED")
    return {'status': 'ok', 'errors': []}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'trading-system',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='L0: Daily DB backup to parquet for startup restore',
    schedule_interval='0 18 * * *',  # 18:00 UTC = 13:00 COT, after market close
    start_date=datetime(2026, 2, 17),
    catchup=False,
    max_active_runs=1,
    tags=get_dag_tags(DAG_ID),
) as dag:

    t_health = PythonOperator(
        task_id='health_check',
        python_callable=health_check,
    )

    t_ohlcv = PythonOperator(
        task_id='export_ohlcv_backup',
        python_callable=export_ohlcv_backup,
    )

    t_macro = PythonOperator(
        task_id='export_macro_backup',
        python_callable=export_macro_backup,
    )

    t_manifest = PythonOperator(
        task_id='write_manifest',
        python_callable=write_manifest,
    )

    t_validate = PythonOperator(
        task_id='validate_backups',
        python_callable=validate_backups,
    )

    t_health >> t_ohlcv >> t_macro >> t_manifest >> t_validate
