#!/usr/bin/env python3
"""
Database Seeding Script - USDCOP Trading System
================================================

Este script carga datos iniciales en PostgreSQL.
Intenta usar parquet (LFS) primero, luego CSV como fallback.

Uso:
    python scripts/seed_database.py

Variables de entorno:
    DATABASE_URL: postgresql://user:pass@host:port/db

Si no se especifica DATABASE_URL, usa valores por defecto.
"""

import os
import sys
import gzip
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import create_engine, text


def get_database_url():
    """Get database connection URL from environment or defaults."""
    return os.getenv(
        "DATABASE_URL",
        "postgresql://trading:trading@localhost:5432/usdcop"
    )


def find_latest_backup(pattern: str, backup_dir: Path) -> Path:
    """Find the most recent backup file matching pattern."""
    files = list(backup_dir.glob(f"{pattern}*.csv.gz"))
    if not files:
        return None
    return max(files, key=lambda f: f.stat().st_mtime)


def load_data(name: str, parquet_path: Path, backup_dir: Path) -> pd.DataFrame:
    """
    Load data from parquet (preferred) or CSV backup (fallback).

    Args:
        name: Data source name for logging
        parquet_path: Path to parquet file
        backup_dir: Directory with CSV backups

    Returns:
        DataFrame with loaded data
    """
    # Try parquet first (LFS)
    if parquet_path.exists():
        try:
            # Check if it's an LFS pointer (small text file)
            size = parquet_path.stat().st_size
            if size < 1000:  # LFS pointers are ~130 bytes
                content = parquet_path.read_text()[:50]
                if "version https://git-lfs" in content:
                    print(f"  [!] {name}: LFS pointer detected, trying CSV backup...")
                    raise ValueError("LFS file not downloaded")

            df = pd.read_parquet(parquet_path)
            print(f"  [OK] {name}: Loaded from parquet ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"  [!] {name}: Parquet failed ({e}), trying CSV backup...")

    # Fallback to CSV
    csv_backup = find_latest_backup(name, backup_dir)
    if csv_backup:
        try:
            df = pd.read_csv(csv_backup, compression='gzip')
            print(f"  [OK] {name}: Loaded from CSV backup ({len(df)} rows)")
            return df
        except Exception as e:
            print(f"  [ERROR] {name}: CSV backup failed: {e}")
            return None

    print(f"  [ERROR] {name}: No data source available!")
    return None


def seed_database():
    """Main seeding function."""
    print("=" * 60)
    print("USDCOP Trading System - Database Seeding")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Paths
    seeds_dir = PROJECT_ROOT / "seeds" / "latest"
    backup_dir = PROJECT_ROOT / "data" / "backups"

    # Database connection
    db_url = get_database_url()
    print(f"Database: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print()

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[OK] Database connection successful")
    except Exception as e:
        print(f"[ERROR] Database connection failed: {e}")
        print("\nMake sure PostgreSQL is running:")
        print("  docker-compose up -d postgres")
        sys.exit(1)

    print()
    print("Loading data sources...")
    print("-" * 40)

    # Load OHLCV 5-min
    df_ohlcv = load_data(
        "usdcop_m5_ohlcv",
        seeds_dir / "usdcop_m5_ohlcv.parquet",
        backup_dir
    )

    # Load macro daily
    df_macro = load_data(
        "macro_indicators_daily",
        seeds_dir / "macro_indicators_daily.parquet",
        backup_dir
    )

    # Load USDCOP daily
    df_daily = load_data(
        "usdcop_daily_ohlcv",
        seeds_dir / "usdcop_daily_ohlcv.parquet",
        backup_dir
    )

    print()
    print("Inserting into database...")
    print("-" * 40)

    # Insert data
    try:
        if df_ohlcv is not None:
            # Clear existing data to avoid duplicates
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM usdcop_m5_ohlcv WHERE TRUE"))
                conn.commit()
            df_ohlcv.to_sql("usdcop_m5_ohlcv", engine, if_exists="append", index=False)
            print(f"  [OK] usdcop_m5_ohlcv: {len(df_ohlcv)} rows inserted")

        if df_macro is not None:
            with engine.connect() as conn:
                conn.execute(text("DELETE FROM macro_indicators_daily WHERE TRUE"))
                conn.commit()
            df_macro.to_sql("macro_indicators_daily", engine, if_exists="append", index=False)
            print(f"  [OK] macro_indicators_daily: {len(df_macro)} rows inserted")

        if df_daily is not None:
            # Create bi schema if not exists
            with engine.connect() as conn:
                conn.execute(text("CREATE SCHEMA IF NOT EXISTS bi"))
                conn.execute(text("DELETE FROM bi.dim_daily_usdcop WHERE TRUE"))
                conn.commit()
            df_daily.to_sql("dim_daily_usdcop", engine, schema="bi", if_exists="append", index=False)
            print(f"  [OK] bi.dim_daily_usdcop: {len(df_daily)} rows inserted")

    except Exception as e:
        print(f"  [ERROR] Database insert failed: {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("Database seeding completed successfully!")
    print("=" * 60)

    # Summary
    print()
    print("Summary:")
    with engine.connect() as conn:
        for table in ["usdcop_m5_ohlcv", "macro_indicators_daily"]:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            count = result.scalar()
            print(f"  - {table}: {count:,} rows")

        result = conn.execute(text("SELECT COUNT(*) FROM bi.dim_daily_usdcop"))
        count = result.scalar()
        print(f"  - bi.dim_daily_usdcop: {count:,} rows")


if __name__ == "__main__":
    seed_database()
