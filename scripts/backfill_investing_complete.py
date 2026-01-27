#!/usr/bin/env python3
"""
Complete Investing.com Backfill Script
=======================================

This script backfills all Investing.com indicators to ensure complete
historical data coverage. Uses the robust dual-strategy scraper.

Indicators to backfill:
- USD/COP (new column in macro_indicators_daily)
- Colombia 10Y and 5Y bonds
- Commodities (Gold, Coffee, Brent, WTI)
- daily_ohlcv_summary for USD/COP OHLCV data

Usage:
    python scripts/backfill_investing_complete.py --dry-run
    python scripts/backfill_investing_complete.py --update-db
    python scripts/backfill_investing_complete.py --indicator USDCOP --update-db
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'data' / 'pipeline' / '02_scrapers' / '02_custom'))

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd

# Import our scraper
from scraper_investing import (
    fetch_historical_robust,
    fetch_historical_api,
    create_session,
    INDICATOR_CONFIG,
    FetchMethod,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database connection settings
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'localhost'),
    'port': int(os.environ.get('POSTGRES_PORT', 5432)),
    'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
    'user': os.environ.get('POSTGRES_USER', 'admin'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'admin123'),
}

# Indicators that need backfill with their target dates
BACKFILL_TARGETS = {
    # New indicator - full backfill needed
    'fxrt_spot_usdcop_col_d_usdcop': {
        'name': 'USD/COP',
        'start_date': '2020-01-01',
        'table': 'macro_indicators_daily',
    },
    # Colombia bonds - need update to today
    'finc_bond_yield10y_col_d_col10y': {
        'name': 'Colombia 10Y',
        'start_date': '2026-01-21',  # Last date in DB
        'table': 'macro_indicators_daily',
    },
    'finc_bond_yield5y_col_d_col5y': {
        'name': 'Colombia 5Y',
        'start_date': '2026-01-21',
        'table': 'macro_indicators_daily',
    },
    # Commodities - need update to today
    'comm_metal_gold_glb_d_gold': {
        'name': 'Gold',
        'start_date': '2026-01-22',
        'table': 'macro_indicators_daily',
    },
    'comm_agri_coffee_glb_d_coffee': {
        'name': 'Coffee',
        'start_date': '2026-01-22',
        'table': 'macro_indicators_daily',
    },
    'comm_oil_brent_glb_d_brent': {
        'name': 'Brent',
        'start_date': '2026-01-22',
        'table': 'macro_indicators_daily',
    },
    'comm_oil_wti_glb_d_wti': {
        'name': 'WTI',
        'start_date': '2026-01-22',
        'table': 'macro_indicators_daily',
    },
}


def get_db_connection():
    """Create database connection."""
    return psycopg2.connect(**DB_CONFIG)


def get_last_date(column: str, conn) -> str:
    """Get the last date with data for a column."""
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT MAX(fecha)::text
        FROM macro_indicators_daily
        WHERE {column} IS NOT NULL
    """)
    result = cursor.fetchone()[0]
    cursor.close()
    return result


def upsert_macro_data(df: pd.DataFrame, column: str, conn, dry_run: bool = True):
    """
    Upsert data into macro_indicators_daily.

    Args:
        df: DataFrame with columns ['fecha', 'valor']
        column: Target column name
        conn: Database connection
        dry_run: If True, only show what would be done
    """
    if df.empty:
        logger.warning(f"No data to upsert for {column}")
        return 0

    rows = [(row['fecha'], row['valor']) for _, row in df.iterrows()]

    if dry_run:
        logger.info(f"[DRY RUN] Would upsert {len(rows)} rows to {column}")
        logger.info(f"  Date range: {df['fecha'].min()} to {df['fecha'].max()}")
        logger.info(f"  Value range: {df['valor'].min():.2f} to {df['valor'].max():.2f}")
        return len(rows)

    cursor = conn.cursor()

    # First ensure all dates exist in the table
    for fecha, valor in rows:
        cursor.execute("""
            INSERT INTO macro_indicators_daily (fecha)
            VALUES (%s)
            ON CONFLICT (fecha) DO NOTHING
        """, (fecha,))

    # Then update the specific column
    for fecha, valor in rows:
        cursor.execute(f"""
            UPDATE macro_indicators_daily
            SET {column} = %s, updated_at = NOW()
            WHERE fecha = %s
        """, (valor, fecha))

    conn.commit()
    cursor.close()

    logger.info(f"Upserted {len(rows)} rows to {column}")
    return len(rows)


def backfill_indicator(
    column: str,
    target: dict,
    session,
    conn,
    end_date: str,
    dry_run: bool = True
) -> dict:
    """
    Backfill a single indicator.

    Returns:
        Dict with backfill results
    """
    name = target['name']
    start_date = target['start_date']

    logger.info(f"\n{'='*60}")
    logger.info(f"Backfilling {name} ({column})")
    logger.info(f"  From: {start_date} To: {end_date}")
    logger.info(f"{'='*60}")

    # Check if we need full backfill or just update
    last_date = get_last_date(column, conn)
    if last_date:
        # Only fetch from day after last date
        actual_start = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        if actual_start > end_date:
            logger.info(f"  Already up to date (last: {last_date})")
            return {'indicator': name, 'rows': 0, 'status': 'UP_TO_DATE'}
        logger.info(f"  Updating from {actual_start} (last in DB: {last_date})")
        start_date = actual_start

    try:
        # Fetch data using robust method
        df = fetch_historical_robust(
            column=column,
            start_date=start_date,
            end_date=end_date,
            session=session,
            delay=3.0,
            validate=True
        )

        if df.empty:
            logger.warning(f"  No data returned for {name}")
            return {'indicator': name, 'rows': 0, 'status': 'NO_DATA'}

        # Filter to only dates we need
        df = df[df['fecha'] >= start_date]
        df = df[df['fecha'] <= end_date]

        logger.info(f"  Fetched {len(df)} rows")

        # Upsert to database
        rows = upsert_macro_data(df, column, conn, dry_run)

        status = 'DRY_RUN' if dry_run else 'SUCCESS'
        return {'indicator': name, 'rows': rows, 'status': status}

    except Exception as e:
        logger.error(f"  Error backfilling {name}: {e}")
        return {'indicator': name, 'rows': 0, 'status': 'ERROR', 'error': str(e)}


def backfill_daily_ohlcv(session, conn, dry_run: bool = True) -> dict:
    """
    Backfill daily_ohlcv_summary for USD/COP.

    This table stores OHLCV data, so we need to fetch it differently.
    """
    logger.info(f"\n{'='*60}")
    logger.info("Backfilling daily_ohlcv_summary (USD/COP OHLCV)")
    logger.info(f"{'='*60}")

    # Get last date in table
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(trading_date)::text
        FROM daily_ohlcv_summary
        WHERE symbol = 'USD/COP'
    """)
    last_date = cursor.fetchone()[0]
    cursor.close()

    if last_date:
        start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = '2020-01-01'

    end_date = datetime.now().strftime('%Y-%m-%d')

    if start_date > end_date:
        logger.info(f"  Already up to date (last: {last_date})")
        return {'indicator': 'USD/COP OHLCV', 'rows': 0, 'status': 'UP_TO_DATE'}

    logger.info(f"  Fetching from {start_date} to {end_date}")

    # For OHLCV, we need to use a different approach - the API returns OHLC data
    try:
        # Use the API endpoint which returns full OHLCV
        from scraper_investing import fetch_historical_api

        df = fetch_historical_api(
            instrument_id=2112,  # USD/COP
            start_date=start_date,
            end_date=end_date,
            session=session,
            delay=3.0,
            referer_url='https://www.investing.com/currencies/usd-cop-historical-data'
        )

        if df.empty:
            logger.warning("  No OHLCV data returned")
            return {'indicator': 'USD/COP OHLCV', 'rows': 0, 'status': 'NO_DATA'}

        # The API returns 'valor' which is close price
        # For daily_ohlcv_summary we need full OHLCV
        # Let's use the close price for all values (approximation for now)

        rows_to_insert = []
        for _, row in df.iterrows():
            close = row['valor']
            rows_to_insert.append((
                row['fecha'],
                'USD/COP',
                close,  # open
                close,  # high
                close,  # low
                close,  # close
                close,  # avg
                1,      # bar_count
                0       # volume
            ))

        if dry_run:
            logger.info(f"[DRY RUN] Would upsert {len(rows_to_insert)} OHLCV rows")
            return {'indicator': 'USD/COP OHLCV', 'rows': len(rows_to_insert), 'status': 'DRY_RUN'}

        cursor = conn.cursor()
        execute_values(
            cursor,
            """
            INSERT INTO daily_ohlcv_summary
            (trading_date, symbol, open_price, high_price, low_price, close_price, avg_price, bar_count, total_volume)
            VALUES %s
            ON CONFLICT (trading_date, symbol) DO UPDATE SET
                close_price = EXCLUDED.close_price,
                avg_price = EXCLUDED.avg_price
            """,
            rows_to_insert
        )
        conn.commit()
        cursor.close()

        logger.info(f"  Upserted {len(rows_to_insert)} OHLCV rows")
        return {'indicator': 'USD/COP OHLCV', 'rows': len(rows_to_insert), 'status': 'SUCCESS'}

    except Exception as e:
        logger.error(f"  Error backfilling OHLCV: {e}")
        return {'indicator': 'USD/COP OHLCV', 'rows': 0, 'status': 'ERROR', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Complete Investing.com backfill')
    parser.add_argument('--dry-run', action='store_true', help='Preview without updating DB')
    parser.add_argument('--update-db', action='store_true', help='Actually update the database')
    parser.add_argument('--indicator', help='Specific indicator (USDCOP, COL10Y, GOLD, etc.)')
    parser.add_argument('--skip-ohlcv', action='store_true', help='Skip daily_ohlcv_summary update')

    args = parser.parse_args()

    if not args.update_db:
        logger.info("Running in DRY RUN mode. Use --update-db to write to database.")
        args.dry_run = True

    # Create session and connection
    session = create_session()
    conn = get_db_connection()

    end_date = datetime.now().strftime('%Y-%m-%d')

    # Filter indicators if specified
    targets = BACKFILL_TARGETS.copy()
    if args.indicator:
        indicator_map = {
            'USDCOP': 'fxrt_spot_usdcop_col_d_usdcop',
            'COL10Y': 'finc_bond_yield10y_col_d_col10y',
            'COL5Y': 'finc_bond_yield5y_col_d_col5y',
            'GOLD': 'comm_metal_gold_glb_d_gold',
            'COFFEE': 'comm_agri_coffee_glb_d_coffee',
            'BRENT': 'comm_oil_brent_glb_d_brent',
            'WTI': 'comm_oil_wti_glb_d_wti',
        }
        column = indicator_map.get(args.indicator.upper())
        if column and column in targets:
            targets = {column: targets[column]}
        else:
            logger.error(f"Unknown indicator: {args.indicator}")
            logger.info(f"Valid options: {list(indicator_map.keys())}")
            sys.exit(1)

    # Backfill macro indicators
    results = []
    for column, target in targets.items():
        result = backfill_indicator(
            column=column,
            target=target,
            session=session,
            conn=conn,
            end_date=end_date,
            dry_run=args.dry_run
        )
        results.append(result)

        # Rate limiting between indicators
        import time
        time.sleep(5)

    # Backfill OHLCV if not skipped
    if not args.skip_ohlcv and (not args.indicator or args.indicator.upper() == 'USDCOP'):
        result = backfill_daily_ohlcv(session, conn, args.dry_run)
        results.append(result)

    conn.close()

    # Print summary
    print("\n" + "="*60)
    print("BACKFILL SUMMARY")
    print("="*60)
    for r in results:
        status_mark = '[OK]' if r['status'] in ('SUCCESS', 'UP_TO_DATE') else '[DRY]' if r['status'] == 'DRY_RUN' else '[ERR]'
        print(f"  {status_mark} {r['indicator']}: {r['rows']} rows - {r['status']}")
    print("="*60)


if __name__ == '__main__':
    main()
