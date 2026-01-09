#!/usr/bin/env python3
"""
Complete Database Backup Script - Pre-V20 Migration
====================================================

Creates comprehensive backups of all critical tables before V20 migration.

Usage:
    python scripts/backup_all_tables.py

Or via Docker:
    docker exec -it usdcop-postgres-timescale python /opt/airflow/scripts/backup_all_tables.py

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

import os
import sys
import gzip
import json
import subprocess
from datetime import datetime
from pathlib import Path

# Backup configuration
BACKUP_DIR = Path("/opt/airflow/data/backups") if os.path.exists("/opt/airflow") else Path("data/backups")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_SUBDIR = BACKUP_DIR / f"pre_v20_migration_{TIMESTAMP}"

# Database connection
DB_HOST = os.environ.get("POSTGRES_HOST", "postgres")
DB_PORT = os.environ.get("POSTGRES_PORT", "5432")
DB_NAME = os.environ.get("POSTGRES_DB", "usdcop_trading")
DB_USER = os.environ.get("POSTGRES_USER", "admin")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "admin123")

# Tables to backup (organized by importance)
CRITICAL_TABLES = [
    # Trading data
    ("public", "usdcop_m5_ohlcv", "OHLCV market data"),
    ("public", "macro_indicators_daily", "Macro economic indicators"),

    # Configuration
    ("config", "models", "Model configurations"),

    # Trading state & history
    ("public", "trading_state", "Current trading state"),
    ("public", "trades_history", "Trade history"),
    ("public", "equity_snapshots", "Equity curve snapshots"),

    # Inference & signals
    ("trading", "model_inferences", "Model inference results"),
    ("dw", "fact_strategy_signals", "Strategy signals"),
    ("dw", "fact_rl_inference", "RL inference facts"),
    ("dw", "dim_strategy", "Strategy dimensions"),

    # Events & logs
    ("events", "trading_events", "Trading events log"),
]

# Additional tables to check and backup if they exist
OPTIONAL_TABLES = [
    ("public", "paper_trades", "Paper trading records"),
    ("public", "model_state", "Model state backup"),
    ("trading", "model_state", "Trading model state"),
    ("public", "risk_events", "Risk events"),
    ("dw", "fact_trades", "Trade facts"),
]


def get_connection_string():
    """Get PostgreSQL connection string."""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def run_psql_command(sql: str, return_output: bool = False):
    """Execute SQL command via psql."""
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD

    cmd = [
        "psql",
        "-h", DB_HOST,
        "-p", DB_PORT,
        "-U", DB_USER,
        "-d", DB_NAME,
        "-t",  # Tuples only (no headers)
        "-A",  # Unaligned output
        "-c", sql
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=300)
        if return_output:
            return result.stdout.strip()
        return result.returncode == 0
    except Exception as e:
        print(f"  ERROR executing SQL: {e}")
        return None if return_output else False


def table_exists(schema: str, table: str) -> bool:
    """Check if a table exists."""
    sql = f"""
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = '{schema}' AND table_name = '{table}'
    );
    """
    result = run_psql_command(sql, return_output=True)
    return result == 't'


def get_row_count(schema: str, table: str) -> int:
    """Get row count for a table."""
    sql = f"SELECT COUNT(*) FROM {schema}.{table};"
    result = run_psql_command(sql, return_output=True)
    try:
        return int(result)
    except:
        return -1


def backup_table_to_csv(schema: str, table: str, output_path: Path) -> bool:
    """Backup a table to compressed CSV."""
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD

    # Use COPY command to export to CSV
    sql = f"COPY {schema}.{table} TO STDOUT WITH CSV HEADER;"

    cmd = [
        "psql",
        "-h", DB_HOST,
        "-p", DB_PORT,
        "-U", DB_USER,
        "-d", DB_NAME,
        "-c", sql
    ]

    try:
        # Run psql and capture output
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=600)

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.decode()}")
            return False

        # Compress and save
        with gzip.open(output_path, 'wb') as f:
            f.write(result.stdout)

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def backup_schema_ddl(output_path: Path) -> bool:
    """Backup database schema (DDL only, no data)."""
    env = os.environ.copy()
    env["PGPASSWORD"] = DB_PASSWORD

    cmd = [
        "pg_dump",
        "-h", DB_HOST,
        "-p", DB_PORT,
        "-U", DB_USER,
        "-d", DB_NAME,
        "--schema-only",  # DDL only
        "--no-owner",
        "--no-privileges"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, env=env, timeout=120)

        if result.returncode != 0:
            print(f"  ERROR: {result.stderr.decode()}")
            return False

        with gzip.open(output_path, 'wb') as f:
            f.write(result.stdout)

        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def create_backup_manifest(backup_dir: Path, results: list) -> Path:
    """Create a manifest file with backup details."""
    manifest = {
        "timestamp": TIMESTAMP,
        "backup_dir": str(backup_dir),
        "database": DB_NAME,
        "host": DB_HOST,
        "purpose": "Pre-V20 Migration Backup",
        "tables": []
    }

    for schema, table, description, success, rows, file_path in results:
        manifest["tables"].append({
            "schema": schema,
            "table": table,
            "description": description,
            "success": success,
            "rows": rows,
            "file": str(file_path) if file_path else None
        })

    manifest["summary"] = {
        "total_tables": len(results),
        "successful": sum(1 for r in results if r[3]),
        "failed": sum(1 for r in results if not r[3]),
        "total_rows": sum(r[4] for r in results if r[4] > 0)
    }

    manifest_path = backup_dir / "MANIFEST.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def main():
    print("=" * 70)
    print("USDCOP Trading System - Complete Database Backup")
    print(f"Timestamp: {TIMESTAMP}")
    print("=" * 70)

    # Create backup directory
    BACKUP_SUBDIR.mkdir(parents=True, exist_ok=True)
    print(f"\nBackup directory: {BACKUP_SUBDIR}")

    results = []

    # 1. Backup schema DDL first
    print("\n[1/3] Backing up database schema (DDL)...")
    schema_path = BACKUP_SUBDIR / "schema_ddl.sql.gz"
    if backup_schema_ddl(schema_path):
        print(f"  ✓ Schema DDL saved to {schema_path.name}")
    else:
        print(f"  ✗ Failed to backup schema DDL")

    # 2. Backup critical tables
    print("\n[2/3] Backing up CRITICAL tables...")
    print("-" * 70)

    for schema, table, description in CRITICAL_TABLES:
        full_name = f"{schema}.{table}"

        if not table_exists(schema, table):
            print(f"  SKIP: {full_name} (does not exist)")
            results.append((schema, table, description, False, 0, None))
            continue

        rows = get_row_count(schema, table)
        output_file = BACKUP_SUBDIR / f"{schema}_{table}.csv.gz"

        print(f"  Backing up {full_name} ({rows:,} rows)...", end=" ", flush=True)

        if backup_table_to_csv(schema, table, output_file):
            size_kb = output_file.stat().st_size / 1024
            print(f"✓ ({size_kb:.1f} KB)")
            results.append((schema, table, description, True, rows, output_file))
        else:
            print("✗ FAILED")
            results.append((schema, table, description, False, rows, None))

    # 3. Backup optional tables (if they exist)
    print("\n[3/3] Backing up OPTIONAL tables...")
    print("-" * 70)

    for schema, table, description in OPTIONAL_TABLES:
        full_name = f"{schema}.{table}"

        if not table_exists(schema, table):
            print(f"  SKIP: {full_name} (does not exist)")
            continue

        rows = get_row_count(schema, table)
        output_file = BACKUP_SUBDIR / f"{schema}_{table}.csv.gz"

        print(f"  Backing up {full_name} ({rows:,} rows)...", end=" ", flush=True)

        if backup_table_to_csv(schema, table, output_file):
            size_kb = output_file.stat().st_size / 1024
            print(f"✓ ({size_kb:.1f} KB)")
            results.append((schema, table, description, True, rows, output_file))
        else:
            print("✗ FAILED")
            results.append((schema, table, description, False, rows, None))

    # Create manifest
    manifest_path = create_backup_manifest(BACKUP_SUBDIR, results)

    # Summary
    print("\n" + "=" * 70)
    print("BACKUP SUMMARY")
    print("=" * 70)

    successful = sum(1 for r in results if r[3])
    failed = sum(1 for r in results if not r[3])
    total_rows = sum(r[4] for r in results if r[4] > 0)

    print(f"  Tables backed up: {successful}/{len(results)}")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Failed: {failed}")
    print(f"  Backup location: {BACKUP_SUBDIR}")
    print(f"  Manifest: {manifest_path}")

    # List files
    print("\nBackup files:")
    for f in sorted(BACKUP_SUBDIR.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name} ({size_kb:.1f} KB)")

    print("\n" + "=" * 70)
    if failed == 0:
        print("✓ BACKUP COMPLETED SUCCESSFULLY")
        print("  Safe to proceed with V20 migration")
    else:
        print("⚠ BACKUP COMPLETED WITH WARNINGS")
        print(f"  {failed} tables could not be backed up")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
