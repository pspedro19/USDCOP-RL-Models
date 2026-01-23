#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backfill Daily USDCOP OHLCV from Investing.com
==============================================

Script para hacer backfill completo de datos OHLCV diarios oficiales
desde Investing.com hacia la tabla bi.dim_daily_usdcop.

Este script es la fuente de verdad para datos diarios de forecasting.
NO usar datos resampled de 5-min para forecasting.

Usage:
    # Full backfill (2015-present)
    python scripts/backfill_daily_ohlcv_investing.py --full

    # Incremental update (last 30 days)
    python scripts/backfill_daily_ohlcv_investing.py

    # Custom date range
    python scripts/backfill_daily_ohlcv_investing.py --start 2020-01-01 --end 2024-12-31

    # Export to parquet (for seeds/latest)
    python scripts/backfill_daily_ohlcv_investing.py --export-parquet

Contract: CTR-BACKFILL-DAILY-001
Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Add scraper path
import sys
scraper_path = PROJECT_ROOT / "data/pipeline/02_scrapers/02_custom"
sys.path.insert(0, str(scraper_path))

from scraper_usdcop_investing import (
    USDCOPInvestingScraper,
    fetch_usdcop_daily,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Full backfill start date (Investing.com has data from ~2010)
FULL_BACKFILL_START = "2015-01-01"

# Default incremental days
DEFAULT_INCREMENTAL_DAYS = 30

# Output paths
PARQUET_OUTPUT = PROJECT_ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
CSV_OUTPUT = PROJECT_ROOT / "data/forecasting/usdcop_daily_ohlcv.csv"


# =============================================================================
# DATABASE FUNCTIONS
# =============================================================================

def get_db_connection_string() -> str:
    """Get database connection string from environment."""
    return os.environ.get("DATABASE_URL", "")


def get_latest_date_in_db(connection_string: str) -> str:
    """Get the latest date in bi.dim_daily_usdcop."""
    import psycopg2

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT MAX(date)::text
                    FROM bi.dim_daily_usdcop
                """)
                result = cur.fetchone()
                return result[0] if result and result[0] else None
    except Exception as e:
        logger.warning(f"Could not get latest date from DB: {e}")
        return None


def insert_to_database(
    df: pd.DataFrame,
    connection_string: str,
    mode: str = "upsert"
) -> int:
    """
    Insert data into bi.dim_daily_usdcop.

    Args:
        df: DataFrame with OHLCV data
        connection_string: PostgreSQL connection string
        mode: 'upsert' (default) or 'replace'

    Returns:
        Number of rows affected
    """
    import psycopg2
    from psycopg2.extras import execute_values

    if df.empty:
        logger.warning("Empty DataFrame, nothing to insert")
        return 0

    # Ensure date is proper format
    df = df.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    if mode == "replace":
        # Truncate and insert
        delete_sql = """
            DELETE FROM bi.dim_daily_usdcop
            WHERE date >= %s AND date <= %s
        """
        min_date = df['date'].min()
        max_date = df['date'].max()
    else:
        delete_sql = None

    insert_sql = """
        INSERT INTO bi.dim_daily_usdcop (date, open, high, low, close, volume, source)
        VALUES %s
        ON CONFLICT (date) DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            source = EXCLUDED.source,
            updated_at = NOW()
    """

    # Prepare values
    values = []
    for _, row in df.iterrows():
        values.append((
            row['date'],
            float(row['open']) if pd.notna(row['open']) else None,
            float(row['high']) if pd.notna(row['high']) else None,
            float(row['low']) if pd.notna(row['low']) else None,
            float(row['close']) if pd.notna(row['close']) else None,
            int(row['volume']) if pd.notna(row['volume']) else 0,
            row.get('source', 'investing')
        ))

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                if delete_sql:
                    cur.execute(delete_sql, (min_date, max_date))
                    logger.info(f"Deleted existing data from {min_date} to {max_date}")

                execute_values(cur, insert_sql, values, page_size=500)
                rows_affected = len(values)

            conn.commit()

        logger.info(f"Inserted/updated {rows_affected} rows")
        return rows_affected

    except Exception as e:
        logger.error(f"Database insert failed: {e}")
        raise


def ensure_table_exists(connection_string: str):
    """Ensure bi.dim_daily_usdcop table exists."""
    import psycopg2

    create_sql = """
        CREATE SCHEMA IF NOT EXISTS bi;

        CREATE TABLE IF NOT EXISTS bi.dim_daily_usdcop (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            open DECIMAL(12,4) NOT NULL,
            high DECIMAL(12,4) NOT NULL,
            low DECIMAL(12,4) NOT NULL,
            close DECIMAL(12,4) NOT NULL,
            volume BIGINT DEFAULT 0,
            source VARCHAR(50) DEFAULT 'investing',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_daily_usdcop_date
            ON bi.dim_daily_usdcop (date DESC);
    """

    try:
        with psycopg2.connect(connection_string) as conn:
            with conn.cursor() as cur:
                cur.execute(create_sql)
            conn.commit()
        logger.info("Table bi.dim_daily_usdcop ensured")
    except Exception as e:
        logger.error(f"Failed to create table: {e}")
        raise


# =============================================================================
# DATA VALIDATION
# =============================================================================

def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate OHLCV data quality.

    Returns:
        Dict with validation results
    """
    results = {
        'is_valid': True,
        'total_rows': len(df),
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    if df.empty:
        results['is_valid'] = False
        results['errors'].append("Empty DataFrame")
        return results

    # Check for required columns
    required = ['date', 'open', 'high', 'low', 'close']
    missing = [col for col in required if col not in df.columns]
    if missing:
        results['is_valid'] = False
        results['errors'].append(f"Missing columns: {missing}")
        return results

    # Check for NaN in critical columns
    for col in required:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            results['warnings'].append(f"{col}: {nan_count} NaN values")

    # Check OHLC relationships
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    ).sum()

    if invalid_ohlc > 0:
        results['warnings'].append(f"{invalid_ohlc} rows with invalid OHLC relationships")

    # Check price range
    out_of_range = ((df['close'] < 2000) | (df['close'] > 7000)).sum()
    if out_of_range > 0:
        results['warnings'].append(f"{out_of_range} rows with close outside 2000-7000 range")

    # Check for duplicates
    duplicates = df['date'].duplicated().sum()
    if duplicates > 0:
        results['warnings'].append(f"{duplicates} duplicate dates")

    # Check for gaps (missing trading days)
    df_sorted = df.sort_values('date')
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    # Calculate expected trading days (exclude weekends)
    date_range = pd.date_range(
        df_sorted['date'].min(),
        df_sorted['date'].max(),
        freq='B'  # Business days
    )
    expected_days = len(date_range)
    actual_days = len(df_sorted)

    if actual_days < expected_days * 0.9:  # Allow 10% tolerance for holidays
        missing_pct = (1 - actual_days / expected_days) * 100
        results['warnings'].append(
            f"Possible data gaps: {actual_days} rows vs {expected_days} expected ({missing_pct:.1f}% missing)"
        )

    # Statistics
    results['stats'] = {
        'date_range': (
            str(df['date'].min()),
            str(df['date'].max())
        ),
        'close_range': (
            float(df['close'].min()),
            float(df['close'].max())
        ),
        'close_mean': float(df['close'].mean()),
        'close_std': float(df['close'].std()),
        'trading_days': actual_days,
    }

    return results


def fill_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill gaps in daily data using forward fill.

    Only fills gaps of 1-2 days (holidays).
    Larger gaps are left as-is (may indicate real data issues).
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    # Create full date range (business days only)
    full_range = pd.date_range(
        df.index.min(),
        df.index.max(),
        freq='B'
    )

    # Reindex and forward fill (max 2 days)
    df = df.reindex(full_range)
    df = df.ffill(limit=2)

    # Reset index
    df = df.reset_index().rename(columns={'index': 'date'})

    # Drop remaining NaN rows
    df = df.dropna(subset=['close'])

    return df


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def export_to_parquet(df: pd.DataFrame, output_path: Path):
    """Export DataFrame to parquet file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(output_path, index=False, compression='snappy')
    logger.info(f"Exported {len(df)} rows to {output_path}")


def export_to_csv(df: pd.DataFrame, output_path: Path):
    """Export DataFrame to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(df)} rows to {output_path}")


# =============================================================================
# MAIN BACKFILL FUNCTION
# =============================================================================

def run_backfill(
    start_date: str,
    end_date: str,
    to_database: bool = True,
    to_parquet: bool = False,
    to_csv: bool = False,
    fill_data_gaps: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run the backfill process.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        to_database: Insert into bi.dim_daily_usdcop
        to_parquet: Export to parquet file
        to_csv: Export to CSV file
        fill_data_gaps: Fill missing trading days
        verbose: Print progress

    Returns:
        DataFrame with fetched data
    """
    logger.info("=" * 60)
    logger.info("USDCOP Daily OHLCV Backfill")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date} to {end_date}")

    # 1. Fetch data from Investing.com
    logger.info("\n1. Fetching data from Investing.com...")
    scraper = USDCOPInvestingScraper(delay_between_requests=2.0)

    def progress(chunk, total):
        if verbose:
            print(f"   Progress: {chunk}/{total} chunks", end='\r')

    df = scraper.fetch_ohlcv(start_date, end_date, progress_callback=progress)
    print()  # New line after progress

    if df.empty:
        logger.error("No data fetched!")
        return df

    logger.info(f"   Fetched {len(df)} rows")

    # 2. Validate data
    logger.info("\n2. Validating data...")
    validation = validate_data(df)

    if validation['errors']:
        for error in validation['errors']:
            logger.error(f"   ERROR: {error}")
    if validation['warnings']:
        for warning in validation['warnings']:
            logger.warning(f"   WARNING: {warning}")

    logger.info(f"   Date range: {validation['stats']['date_range']}")
    logger.info(f"   Close range: {validation['stats']['close_range']}")

    if not validation['is_valid']:
        logger.error("Data validation failed!")
        return df

    # 3. Fill gaps if requested
    if fill_data_gaps:
        logger.info("\n3. Filling data gaps...")
        original_len = len(df)
        df = fill_gaps(df)
        filled = len(df) - original_len
        logger.info(f"   Filled {filled} gap days")

    # 4. Insert into database
    if to_database:
        logger.info("\n4. Inserting into database...")
        connection_string = get_db_connection_string()

        if not connection_string:
            logger.warning("   DATABASE_URL not set, skipping database insert")
        else:
            try:
                ensure_table_exists(connection_string)
                rows = insert_to_database(df, connection_string)
                logger.info(f"   Inserted/updated {rows} rows")
            except Exception as e:
                logger.error(f"   Database insert failed: {e}")

    # 5. Export to files
    if to_parquet:
        logger.info("\n5. Exporting to parquet...")
        export_to_parquet(df, PARQUET_OUTPUT)

    if to_csv:
        logger.info("\n6. Exporting to CSV...")
        export_to_csv(df, CSV_OUTPUT)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backfill USDCOP daily OHLCV from Investing.com"
    )

    parser.add_argument(
        '--full',
        action='store_true',
        help=f'Full backfill from {FULL_BACKFILL_START}'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=DEFAULT_INCREMENTAL_DAYS,
        help=f'Number of days for incremental update (default: {DEFAULT_INCREMENTAL_DAYS})'
    )
    parser.add_argument(
        '--no-db',
        action='store_true',
        help='Skip database insert'
    )
    parser.add_argument(
        '--export-parquet',
        action='store_true',
        help=f'Export to {PARQUET_OUTPUT}'
    )
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help=f'Export to {CSV_OUTPUT}'
    )
    parser.add_argument(
        '--no-fill-gaps',
        action='store_true',
        help='Skip gap filling'
    )

    args = parser.parse_args()

    # Determine date range
    if args.full:
        start_date = FULL_BACKFILL_START
    elif args.start:
        start_date = args.start
    else:
        # Incremental: check last date in DB or use default days
        connection_string = get_db_connection_string()
        if connection_string:
            last_date = get_latest_date_in_db(connection_string)
            if last_date:
                # Start from day after last date
                start_date = (
                    datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                logger.info(f"Incremental update from {last_date}")
            else:
                start_date = (
                    datetime.now() - timedelta(days=args.days)
                ).strftime("%Y-%m-%d")
        else:
            start_date = (
                datetime.now() - timedelta(days=args.days)
            ).strftime("%Y-%m-%d")

    end_date = args.end

    # Run backfill
    run_backfill(
        start_date=start_date,
        end_date=end_date,
        to_database=not args.no_db,
        to_parquet=args.export_parquet,
        to_csv=args.export_csv,
        fill_data_gaps=not args.no_fill_gaps,
    )


if __name__ == '__main__':
    main()
