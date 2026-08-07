#!/usr/bin/env python3
"""
Seed Database from MinIO/S3 with Fallback Chain
================================================

This script loads seed data into PostgreSQL with intelligent fallback:

Priority Chain:
1. MinIO bucket: seeds/latest/
2. Local DVC: seeds/latest/ (if dvc pull worked)
3. Legacy backups: data/backups/ or data/pipeline/

Features:
- Idempotent: Skips tables that already have data
- Validates data after loading
- Triggers backfill DAG for updates to current date

Contract: CTR-SEED-INIT-001
Version: 2.0.0
"""

import glob
import gzip
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import psycopg2
import yaml

from src.data_quality.macro_scale import (
    MacroScaleError,
    manifiesto_backup_2026_06,
    validate_and_repair_macro_scale,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Database config
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'postgres'),
    'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
    'user': os.environ.get('POSTGRES_USER', 'admin'),
    'password': os.environ.get('POSTGRES_PASSWORD', ''),
}

# MinIO config
MINIO_CONFIG = {
    'endpoint': os.environ.get('MINIO_ENDPOINT', 'minio:9000'),
    'access_key': os.environ.get('AWS_ACCESS_KEY_ID', 'minioadmin'),
    'secret_key': os.environ.get('AWS_SECRET_ACCESS_KEY', ''),
    'bucket': 'seeds',
    'secure': False,
}

# Seed sources configuration
SEED_SOURCES = [
    {
        'name': 'ohlcv',
        'table': 'usdcop_m5_ohlcv',
        'required': True,
        'minio_path': 'latest/usdcop_m5_ohlcv.parquet',
        'backup_paths': [
            '/app/data/backups/seeds/usdcop_m5_ohlcv_backup.parquet',
            '/opt/airflow/data/backups/seeds/usdcop_m5_ohlcv_backup.parquet',
        ],
        'local_paths': [
            '/opt/airflow/seeds/latest/fx_multi_m5_ohlcv.parquet',
            '/app/seeds/latest/fx_multi_m5_ohlcv.parquet',
            '/opt/airflow/seeds/latest/usdcop_m5_ohlcv.parquet',
            '/app/seeds/latest/usdcop_m5_ohlcv.parquet',
        ],
        'legacy_paths': [
            '/app/data/backups/usdcop_m5_ohlcv_*.csv.gz',
            '/opt/airflow/data/backups/usdcop_m5_ohlcv_*.csv.gz',
        ],
        'min_rows': 50000,
        'date_column': 'time',
        'conflict_columns': ['time', 'symbol'],
    },
    {
        'name': 'macro',
        'table': 'macro_indicators_daily',
        'required': True,
        'minio_path': 'latest/macro_indicators_daily.parquet',
        'backup_paths': [
            '/app/data/backups/seeds/macro_indicators_daily_backup.parquet',
            '/opt/airflow/data/backups/seeds/macro_indicators_daily_backup.parquet',
        ],
        'local_paths': [
            '/opt/airflow/seeds/latest/macro_indicators_daily.parquet',
            '/app/seeds/latest/macro_indicators_daily.parquet',
        ],
        'legacy_paths': [
            '/app/data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet',
            '/opt/airflow/data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet',
        ],
        'min_rows': 5000,
        'date_column': 'fecha',
    },
    {
        'name': 'trades_history',
        'table': 'trades_history',
        'required': False,  # Optional - Investor mode replay data
        'minio_path': 'latest/trades_history.csv.gz',
        'local_paths': [
            '/opt/airflow/seeds/latest/trades_history.csv.gz',
            '/app/seeds/latest/trades_history.csv.gz',
        ],
        'legacy_paths': [],
        'min_rows': 0,
        'date_column': 'timestamp',
        'json_columns': ['model_metadata'],  # Handle JSONB columns specially
    },
]


@dataclass(frozen=True)
class SeedArtifact:
    """A parsed seed tied to the exact bytes and origin that produced it."""

    frame: pd.DataFrame
    source_uri: str
    sha256: str


def _parse_artifact(payload: bytes, source_uri: str) -> SeedArtifact:
    data = BytesIO(payload)
    if source_uri.endswith('.parquet'):
        frame = pd.read_parquet(data)
    elif source_uri.endswith('.csv.gz'):
        frame = pd.read_csv(data, compression='gzip')
    elif source_uri.endswith('.csv'):
        frame = pd.read_csv(data)
    else:
        raise ValueError(f"Unknown seed format: {source_uri}")
    return SeedArtifact(frame, source_uri, hashlib.sha256(payload).hexdigest())


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection(max_retries: int = 10, retry_delay: int = 5):
    """Connect to PostgreSQL with retry logic."""
    for attempt in range(max_retries):
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            return conn
        except Exception as e:
            print(f"  DB connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise


def table_has_data(conn, table_name: str) -> Tuple[bool, int]:
    """Check if table exists and has data."""
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        return count > 0, count
    finally:
        cursor.close()


def insert_dataframe(conn, df: pd.DataFrame, table_name: str, batch_size: int = 10000,
                     json_columns: List[str] = None, conflict_columns: List[str] = None,
                     commit: bool = True, required_target_columns=None):
    """Insert DataFrame into table using batch UPSERT.

    Args:
        conflict_columns: Explicit list of columns for ON CONFLICT clause.
            If None, auto-detects a single PK column (legacy behavior).
            Use ['time', 'symbol'] for usdcop_m5_ohlcv composite PK.
    """
    if df.empty:
        return 0

    # Resilience to schema drift: a backup parquet may carry columns that the
    # freshly-initialized table does not have (schema evolved after the dump).
    # Insert only the intersection so restore never aborts on an extra column.
    _c = conn.cursor()
    _c.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = %s",
        (table_name.split('.')[-1],),
    )
    target_cols = {r[0] for r in _c.fetchall()}
    _c.close()
    if target_cols:
        missing_required = sorted(set(required_target_columns or ()) - target_cols)
        if missing_required:
            raise RuntimeError(
                f"Required columns absent in {table_name}: {missing_required}"
            )
        extra = [c for c in df.columns if c not in target_cols]
        if extra:
            print(f"    Note: dropping {len(extra)} col(s) absent in {table_name}: "
                  f"{extra[:5]}{'...' if len(extra) > 5 else ''}")
            df = df[[c for c in df.columns if c in target_cols]]
        if df.empty or len(df.columns) == 0:
            return 0

    # Handle JSON columns - convert to JSON strings for PostgreSQL JSONB
    if json_columns:
        for col in json_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None and not isinstance(x, str) else x)

    cursor = conn.cursor()
    columns = list(df.columns)
    columns_str = ', '.join(columns)
    placeholders = ', '.join(['%s'] * len(columns))

    # Determine conflict columns (composite PK support)
    if conflict_columns is None:
        # Legacy auto-detect: single PK column
        pk_col = columns[0]
        if 'time' in columns:
            pk_col = 'time'
        elif 'fecha' in columns:
            pk_col = 'fecha'
        elif 'id' in columns:
            pk_col = 'id'
        elif 'timestamp' in columns:
            pk_col = 'timestamp'
        conflict_cols = [pk_col]
    else:
        conflict_cols = conflict_columns

    conflict_str = ', '.join(conflict_cols)

    # Build UPSERT query
    update_cols = [c for c in columns if c not in conflict_cols]
    update_str = ', '.join([f"{c} = EXCLUDED.{c}" for c in update_cols])

    if update_str:
        query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_str}) DO UPDATE SET {update_str}
        """
    else:
        query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT ({conflict_str}) DO NOTHING
        """

    # Batch insert
    rows_inserted = 0
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i + batch_size]
        data = [tuple(row) for row in batch.itertuples(index=False, name=None)]
        cursor.executemany(query, data)
        rows_inserted += len(data)
        if rows_inserted % 50000 == 0:
            print(f"    Inserted {rows_inserted:,} rows...")

    if commit:
        conn.commit()
    return rows_inserted


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_from_minio(minio_path: str) -> Optional[SeedArtifact]:
    """Load Parquet file from MinIO."""
    try:
        from minio import Minio

        client = Minio(
            MINIO_CONFIG['endpoint'],
            access_key=MINIO_CONFIG['access_key'],
            secret_key=MINIO_CONFIG['secret_key'],
            secure=MINIO_CONFIG['secure'],
        )

        bucket = MINIO_CONFIG['bucket']
        if not client.bucket_exists(bucket):
            print(f"    MinIO bucket '{bucket}' not found")
            return None

        # Download to memory
        response = client.get_object(bucket, minio_path)
        try:
            payload = response.read()
        finally:
            response.close()
            response.release_conn()
        artifact = _parse_artifact(payload, f"s3://{bucket}/{minio_path}")
        print(f"    Loaded from MinIO: {minio_path} ({len(artifact.frame):,} rows)")
        return artifact

    except Exception as e:
        print(f"    MinIO load failed: {e}")
        return None


def load_from_local(paths: List[str]) -> Optional[SeedArtifact]:
    """Load from local file paths."""
    for path_pattern in paths:
        # Handle glob patterns
        if '*' in path_pattern:
            matches = sorted(glob.glob(path_pattern))
            if matches:
                path = matches[-1]  # Most recent
            else:
                continue
        else:
            path = path_pattern

        if not os.path.exists(path):
            continue

        try:
            artifact = _parse_artifact(Path(path).read_bytes(), str(Path(path).resolve()))
            print(f"    Loaded from local: {path} ({len(artifact.frame):,} rows)")
            return artifact

        except Exception as e:
            print(f"    Failed to load {path}: {e}")

    return None


def load_seed_data(source: dict) -> Optional[SeedArtifact]:
    """Load seed data using priority chain.

    Priority:
    1. Daily backup parquets (freshest — written by l0_seed_backup DAG)
    2. Local Git LFS seeds (available immediately after clone)
    3. MinIO bucket (may be empty on fresh deployment)
    4. Legacy CSV/parquet backups (fallback)
    """
    name = source['name']
    print(f"\n  Attempting to load: {name}")

    # 1. Try daily backup parquets FIRST (freshest data)
    backup_paths = source.get('backup_paths', [])
    if backup_paths:
        print(f"    [1/4] Trying daily backup parquets...")
        artifact = load_from_local(backup_paths)
        if artifact is not None:
            return artifact

    # 2. Try local Git LFS paths (available after git clone)
    print(f"    [2/4] Trying local Git LFS paths...")
    artifact = load_from_local(source['local_paths'])
    if artifact is not None:
        return artifact

    # 3. Try MinIO (may have data on existing deployments)
    print(f"    [3/4] Trying MinIO: {source['minio_path']}")
    artifact = load_from_minio(source['minio_path'])
    if artifact is not None:
        return artifact

    # 4. Try legacy paths
    print(f"    [4/4] Trying legacy paths...")
    artifact = load_from_local(source['legacy_paths'])
    if artifact is not None:
        return artifact

    return None


# =============================================================================
# VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame, source: dict) -> bool:
    """Validate loaded data meets requirements."""
    name = source['name']
    min_rows = source.get('min_rows', 0)
    date_col = source.get('date_column')

    # Check row count
    if len(df) < min_rows:
        print(f"    VALIDATION FAILED: {name} has {len(df)} rows (minimum: {min_rows})")
        return False

    # Check date range
    if date_col and date_col in df.columns:
        min_date = df[date_col].min()
        max_date = df[date_col].max()
        print(f"    Date range: {min_date} to {max_date}")

    print(f"    VALIDATION PASSED: {len(df):,} rows")
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    print('=' * 70)
    print('SEED DATABASE FROM MINIO (with fallback)')
    print('=' * 70)
    print(f"Started: {datetime.utcnow().isoformat()}Z")

    # Connect to database
    print("\n[1/3] Connecting to database...")
    conn = get_db_connection()
    print("  Connected")

    # Process each seed source
    print("\n[2/3] Loading seed data...")
    results = {}

    for source in SEED_SOURCES:
        name = source['name']
        table = source['table']
        required = source['required']

        print(f"\n{'='*50}")
        print(f"Processing: {name} -> {table}")
        print('=' * 50)

        # Check if table already has data
        has_data, row_count = table_has_data(conn, table)
        if has_data:
            print(f"  Table already has {row_count:,} rows, skipping")
            results[name] = {'status': 'skipped', 'rows': row_count}
            continue

        # Load data
        artifact = load_seed_data(source)
        if artifact is None:
            if required:
                print(f"  ERROR: Failed to load required seed: {name}")
                results[name] = {'status': 'failed', 'rows': 0}
            else:
                print(f"  WARNING: Optional seed not found: {name}")
                results[name] = {'status': 'not_found', 'rows': 0}
            continue

        df = artifact.frame
        print(f"    Source SHA-256: {artifact.sha256} ({artifact.source_uri})")

        if name == 'macro':
            required_columns = {'fecha', *manifiesto_backup_2026_06().columnas_vigiladas}
            missing = sorted(required_columns - set(df.columns))
            if missing:
                print(f"  ERROR: macro seed missing required columns: {missing}")
                results[name] = {'status': 'validation_failed', 'rows': len(df)}
                continue
            try:
                df, scale_report = validate_and_repair_macro_scale(
                    df, manifiesto_backup_2026_06()
                )
            except MacroScaleError as exc:
                print(f"  ERROR: macro scale validation failed: {exc}")
                results[name] = {'status': 'validation_failed', 'rows': len(df)}
                continue
            print(f"    Macro scale audit: {json.dumps(scale_report, sort_keys=True)}")

        # Validate
        if not validate_data(df, source):
            results[name] = {'status': 'validation_failed', 'rows': len(df)}
            continue

        # Insert
        print(f"\n  Inserting into {table}...")
        json_cols = source.get('json_columns', [])
        conflict_cols = source.get('conflict_columns', None)
        if name == 'macro':
            try:
                lock_cursor = conn.cursor()
                lock_cursor.execute(
                    "LOCK TABLE macro_indicators_daily IN SHARE ROW EXCLUSIVE MODE"
                )
                lock_cursor.execute(
                    "SELECT COUNT(*) FROM macro_indicators_daily"
                )
                locked_count = lock_cursor.fetchone()[0]
                lock_cursor.close()
                if locked_count:
                    conn.rollback()
                    print(f"  Table became non-empty ({locked_count:,} rows), skipping")
                    results[name] = {'status': 'skipped', 'rows': locked_count}
                    continue
                rows = insert_dataframe(
                    conn, df, table, json_columns=json_cols,
                    conflict_columns=conflict_cols, commit=False,
                    required_target_columns=required_columns,
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
        else:
            rows = insert_dataframe(conn, df, table, json_columns=json_cols,
                                    conflict_columns=conflict_cols)
        print(f"  Inserted {rows:,} rows")
        results[name] = {'status': 'success', 'rows': rows}

    conn.close()

    # Summary
    print("\n" + "=" * 70)
    print("SEEDING COMPLETE")
    print("=" * 70)

    for name, result in results.items():
        status = result['status']
        rows = result['rows']
        icon = '✓' if status in ['success', 'skipped'] else '✗'
        print(f"  {icon} {name}: {status} ({rows:,} rows)")

    print(f"\nCompleted: {datetime.utcnow().isoformat()}Z")
    required_failures = {
        source['name']
        for source in SEED_SOURCES
        if source['required']
        and results.get(source['name'], {}).get('status') not in {'success', 'skipped'}
    }
    if required_failures:
        print(f"Required seeds failed: {sorted(required_failures)}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
