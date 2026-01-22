#!/usr/bin/env python
"""
Backfill Script for Core CPI (CPILFESL)
=======================================

One-shot script to backfill historical Core CPI data from FRED API.

Problem: P0-1 in DATASET_QUALITY_ISSUES.md
- CPILFESL only has data from 2024-02-01
- Need to backfill from 2020-01-01

Usage:
    python scripts/backfill_fred_cpilfesl.py

    # Dry run (no database changes):
    python scripts/backfill_fred_cpilfesl.py --dry-run

    # Custom date range:
    python scripts/backfill_fred_cpilfesl.py --start-date 2020-01-01 --end-date 2024-01-31

Version: 1.0.0
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_fred_api_key() -> Optional[str]:
    """Get FRED API key from environment or Vault."""
    # Try environment variable
    api_key = os.environ.get('FRED_API_KEY')
    if api_key:
        return api_key

    # Try Vault
    try:
        from src.shared.secrets.vault_client import get_vault_client
        vault = get_vault_client()
        return vault.get_secret('trading/fred', 'api_key')
    except Exception:
        pass

    return None


def get_database_connection():
    """Get PostgreSQL connection."""
    import psycopg2

    # Try environment variables
    conn_params = {
        'host': os.environ.get('POSTGRES_HOST', 'localhost'),
        'port': int(os.environ.get('POSTGRES_PORT', 5432)),
        'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        'user': os.environ.get('POSTGRES_USER', 'admin'),
        'password': os.environ.get('POSTGRES_PASSWORD', 'admin123'),
    }

    return psycopg2.connect(**conn_params)


def fetch_fred_cpilfesl(api_key: str, start_date: str, end_date: Optional[str] = None):
    """
    Fetch CPILFESL data from FRED API.

    Args:
        api_key: FRED API key
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        pandas DataFrame with date and value columns
    """
    from fredapi import Fred
    import pandas as pd

    fred = Fred(api_key=api_key)

    logger.info(f"Fetching CPILFESL from {start_date} to {end_date or 'today'}...")

    data = fred.get_series(
        'CPILFESL',
        observation_start=start_date,
        observation_end=end_date
    )

    if data is None or data.empty:
        raise ValueError("No data returned from FRED API")

    # Convert to DataFrame
    df = pd.DataFrame({
        'fecha': data.index,
        'value': data.values
    })

    # Ensure date format
    df['fecha'] = pd.to_datetime(df['fecha']).dt.strftime('%Y-%m-%d')

    logger.info(f"Fetched {len(df)} records")
    logger.info(f"Date range: {df['fecha'].min()} to {df['fecha'].max()}")

    return df


def upsert_to_database(df, conn, dry_run: bool = False):
    """
    Upsert CPILFESL data to macro_indicators_daily table.

    Args:
        df: DataFrame with fecha and value columns
        conn: Database connection
        dry_run: If True, don't commit changes
    """
    cursor = conn.cursor()

    column_name = 'infl_cpi_core_usa_m_cpilfesl'

    upsert_sql = f"""
    INSERT INTO macro_indicators_daily (fecha, {column_name}, updated_at)
    VALUES (%s, %s, NOW())
    ON CONFLICT (fecha)
    DO UPDATE SET
        {column_name} = EXCLUDED.{column_name},
        updated_at = NOW()
    WHERE macro_indicators_daily.{column_name} IS NULL
       OR macro_indicators_daily.{column_name} != EXCLUDED.{column_name}
    """

    inserted = 0
    updated = 0

    for _, row in df.iterrows():
        fecha = row['fecha']
        value = float(row['value'])

        if dry_run:
            logger.info(f"[DRY-RUN] Would upsert: {fecha} = {value}")
            inserted += 1
            continue

        try:
            cursor.execute(upsert_sql, (fecha, value))
            if cursor.rowcount > 0:
                updated += 1
        except Exception as e:
            logger.error(f"Error upserting {fecha}: {e}")

    if not dry_run:
        conn.commit()
        logger.info(f"Committed {updated} rows to database")
    else:
        logger.info(f"[DRY-RUN] Would have upserted {inserted} rows")

    cursor.close()
    return updated if not dry_run else inserted


def verify_backfill(conn, start_date: str):
    """Verify the backfill was successful."""
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            MIN(fecha) as min_date,
            MAX(fecha) as max_date,
            COUNT(*) as total_rows,
            COUNT(infl_cpi_core_usa_m_cpilfesl) as non_null_rows,
            ROUND(100.0 * COUNT(infl_cpi_core_usa_m_cpilfesl) / COUNT(*), 1) as completeness_pct
        FROM macro_indicators_daily
        WHERE fecha >= %s
    """, (start_date,))

    result = cursor.fetchone()
    cursor.close()

    logger.info("=" * 60)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Date range: {result[0]} to {result[1]}")
    logger.info(f"Total rows: {result[2]}")
    logger.info(f"CPILFESL non-null rows: {result[3]}")
    logger.info(f"Completeness: {result[4]}%")
    logger.info("=" * 60)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Backfill Core CPI (CPILFESL) from FRED API'
    )
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date (YYYY-MM-DD), default: 2020-01-01'
    )
    parser.add_argument(
        '--end-date',
        default=None,
        help='End date (YYYY-MM-DD), default: today'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be done without making changes'
    )
    parser.add_argument(
        '--skip-verify',
        action='store_true',
        help='Skip verification step'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CPILFESL BACKFILL SCRIPT")
    logger.info("=" * 60)
    logger.info(f"Start date: {args.start_date}")
    logger.info(f"End date: {args.end_date or 'today'}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info("=" * 60)

    # Get API key
    api_key = get_fred_api_key()
    if not api_key:
        logger.error("FRED_API_KEY not found in environment or Vault")
        logger.error("Set FRED_API_KEY environment variable and retry")
        sys.exit(1)

    logger.info("FRED API key found")

    # Fetch data from FRED
    try:
        df = fetch_fred_cpilfesl(api_key, args.start_date, args.end_date)
    except Exception as e:
        logger.error(f"Failed to fetch from FRED: {e}")
        sys.exit(1)

    # Connect to database
    try:
        conn = get_database_connection()
        logger.info("Connected to database")
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    # Upsert data
    try:
        rows_affected = upsert_to_database(df, conn, dry_run=args.dry_run)
        logger.info(f"Upsert complete: {rows_affected} rows affected")
    except Exception as e:
        logger.error(f"Failed to upsert data: {e}")
        conn.close()
        sys.exit(1)

    # Verify
    if not args.skip_verify and not args.dry_run:
        verify_backfill(conn, args.start_date)

    conn.close()
    logger.info("Backfill complete!")


if __name__ == '__main__':
    main()
