#!/usr/bin/env python3
"""
Historical Backfill: FRED/TwelveData → Investing.com Migration
==============================================================

Regenerates historical data from 2020-01-01 to today for indicators
migrated from FRED and TwelveData to Investing.com as primary source.

Indicators Migrated:
    - DXY (Dollar Index) - from FRED DTWEXBGS
    - VIX (Volatility Index) - from FRED VIXCLS
    - US 10Y Treasury Yield - from FRED DGS10
    - US 2Y Treasury Yield - from FRED DGS2
    - USD/MXN - from TwelveData
    - USD/CLP - from TwelveData

Features:
    - Uses AJAX endpoint for efficient historical fetching
    - Checkpointing to resume if interrupted
    - Rate limiting with exponential backoff
    - Validates data ranges before inserting
    - Updates both seeds parquet and PostgreSQL

Usage:
    python scripts/backfill_investing_migration.py --dry-run
    python scripts/backfill_investing_migration.py --update-db
    python scripts/backfill_investing_migration.py --update-seeds
    python scripts/backfill_investing_migration.py --indicator DXY --update-db
    python scripts/backfill_investing_migration.py --resume --update-db

Contract: CTR-L0-BACKFILL-001
Version: 1.0.0
Created: 2026-01-23
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'data' / 'pipeline' / '02_scrapers' / '02_custom'))

# Import the enhanced scraper
try:
    from scraper_investing import (
        create_session,
        fetch_historical_ajax_chunked,
        fetch_historical_from_url,
        fetch_historical_api,
        INVESTING_PAIR_IDS,
        INVESTING_URLS,
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(PROJECT_ROOT / 'data' / 'pipeline' / '02_scrapers' / '02_custom'))
    from scraper_investing import (
        create_session,
        fetch_historical_ajax_chunked,
        fetch_historical_from_url,
        fetch_historical_api,
        INVESTING_PAIR_IDS,
        INVESTING_URLS,
    )

# Try to import psycopg2 for database operations
try:
    import psycopg2
    from psycopg2.extras import execute_values
    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Indicators to migrate from FRED/TwelveData to Investing.com
# For FX pairs, use HTML scraping (use_html=True) instead of AJAX due to AJAX returning wrong values
INDICATORS_TO_MIGRATE: List[Dict[str, Any]] = [
    {
        'column': 'fxrt_index_dxy_usa_d_dxy',
        'pair_id': 8827,
        'name': 'DXY',
        'url': 'https://www.investing.com/indices/usdollar-historical-data',
        'use_html': False,  # AJAX works correctly for indices
        'description': 'Dollar Index',
        'expected_range': [80, 130],
    },
    {
        'column': 'volt_vix_usa_d_vix',
        'pair_id': 44336,
        'name': 'VIX',
        'url': 'https://www.investing.com/indices/volatility-s-p-500-historical-data',
        'use_html': False,  # AJAX works correctly for indices
        'description': 'Volatility Index',
        'expected_range': [9, 90],
    },
    {
        'column': 'finc_bond_yield10y_usa_d_ust10y',
        'pair_id': 23705,
        'name': 'US10Y',
        'url': 'https://www.investing.com/rates-bonds/u.s.-10-year-bond-yield-historical-data',
        'use_html': False,  # AJAX works correctly for bonds
        'description': 'US 10Y Treasury Yield',
        'expected_range': [0, 12],
    },
    {
        'column': 'finc_bond_yield2y_usa_d_dgs2',
        'pair_id': 23701,
        'name': 'US2Y',
        'url': 'https://www.investing.com/rates-bonds/u.s.-2-year-bond-yield-historical-data',
        'use_html': False,  # AJAX works correctly for bonds
        'description': 'US 2Y Treasury Yield',
        'expected_range': [0, 12],
    },
    {
        'column': 'fxrt_spot_usdmxn_mex_d_usdmxn',
        'pair_id': 8827,  # Old AJAX pair_id (not used for FX)
        'instrument_id': 39,  # New API instrument ID - returns correct USD/MXN values
        'name': 'USDMXN',
        'url': 'https://es.investing.com/currencies/usd-mxn-historical-data',
        'use_api': True,  # Use new API endpoint
        'description': 'USD/MXN Exchange Rate',
        'expected_range': [12, 30],
    },
    {
        'column': 'fxrt_spot_usdclp_chl_d_usdclp',
        'pair_id': 8827,  # Old AJAX pair_id (not used for FX)
        'instrument_id': 2110,  # New API instrument ID - returns correct USD/CLP values
        'name': 'USDCLP',
        'url': 'https://es.investing.com/currencies/usd-clp-historical-data',
        'use_api': True,  # Use new API endpoint
        'description': 'USD/CLP Exchange Rate',
        'expected_range': [600, 1200],
    },
]

# Paths
CHECKPOINT_FILE = PROJECT_ROOT / 'outputs' / 'backfill_checkpoint.json'
SEEDS_PATH = PROJECT_ROOT / 'seeds' / 'latest' / 'macro_indicators_daily.parquet'
MANIFEST_PATH = PROJECT_ROOT / 'seeds' / 'latest' / 'manifest.json'

# Rate limiting
RATE_LIMIT_SECONDS = 5
INTER_INDICATOR_DELAY = 10


# =============================================================================
# Database Operations
# =============================================================================

def get_db_connection():
    """Get PostgreSQL connection from environment variables."""
    if not PSYCOPG2_AVAILABLE:
        raise ImportError("psycopg2 not installed. Install with: pip install psycopg2-binary")

    return psycopg2.connect(
        host=os.environ.get('POSTGRES_HOST', 'localhost'),
        port=int(os.environ.get('POSTGRES_PORT', 5432)),
        database=os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        user=os.environ.get('POSTGRES_USER', 'admin'),
        password=os.environ.get('POSTGRES_PASSWORD', 'admin123')
    )


def upsert_to_database(
    df: pd.DataFrame,
    column: str,
    batch_size: int = 500
) -> int:
    """
    Upsert data to PostgreSQL using ON CONFLICT.

    Args:
        df: DataFrame with columns ['fecha', column]
        column: Database column name
        batch_size: Rows per batch

    Returns:
        Number of rows upserted
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    total_upserted = 0

    try:
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]

            values = [
                (row['fecha'], row[column])
                for _, row in batch.iterrows()
            ]

            query = f"""
                INSERT INTO macro_indicators_daily (fecha, {column}, updated_at)
                VALUES %s
                ON CONFLICT (fecha) DO UPDATE SET
                    {column} = EXCLUDED.{column},
                    updated_at = NOW()
            """

            execute_values(
                cursor,
                query,
                values,
                template="(%s, %s, NOW())"
            )

            conn.commit()
            total_upserted += len(batch)
            logger.info(f"    Upserted batch {i//batch_size + 1}: {len(batch)} rows")

    finally:
        cursor.close()
        conn.close()

    return total_upserted


# =============================================================================
# Seeds Operations
# =============================================================================

def load_seeds() -> pd.DataFrame:
    """Load the current seeds parquet file."""
    if not SEEDS_PATH.exists():
        raise FileNotFoundError(f"Seeds file not found: {SEEDS_PATH}")

    df = pd.read_parquet(SEEDS_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def update_seeds(
    seeds_df: pd.DataFrame,
    new_data: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    """
    Update seeds DataFrame with new data for a column.

    Args:
        seeds_df: Original seeds DataFrame
        new_data: New data with ['fecha', column]
        column: Column to update

    Returns:
        Updated seeds DataFrame
    """
    # Ensure fecha is datetime
    new_data = new_data.copy()
    new_data['fecha'] = pd.to_datetime(new_data['fecha'])

    # Create lookup dict
    lookup = dict(zip(new_data['fecha'], new_data[column]))

    # Update values
    updated_count = 0
    for idx, row in seeds_df.iterrows():
        fecha = row['fecha']
        if fecha in lookup:
            old_value = row[column]
            new_value = lookup[fecha]

            if pd.isna(old_value) or abs(old_value - new_value) > 0.001:
                seeds_df.at[idx, column] = new_value
                updated_count += 1

    logger.info(f"  Updated {updated_count} values in seeds for {column}")
    return seeds_df


def save_seeds(df: pd.DataFrame) -> str:
    """
    Save updated seeds and regenerate manifest.

    Returns:
        SHA256 hash of the new file
    """
    # Save parquet
    df.to_parquet(SEEDS_PATH, index=False)
    logger.info(f"  Saved: {SEEDS_PATH}")

    # Calculate new hash
    with open(SEEDS_PATH, 'rb') as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    # Update manifest
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH, 'r') as f:
            manifest = json.load(f)

        manifest['tables']['macro_indicators_daily']['sha256'] = file_hash
        manifest['tables']['macro_indicators_daily']['size_bytes'] = SEEDS_PATH.stat().st_size
        manifest['tables']['macro_indicators_daily']['rows'] = len(df)
        manifest['tables']['macro_indicators_daily']['date_range'] = [
            df['fecha'].min().strftime('%Y-%m-%d'),
            df['fecha'].max().strftime('%Y-%m-%d')
        ]
        manifest['version'] = datetime.now(timezone.utc).strftime('%Y.%m.%d')
        manifest['created_at'] = datetime.now(timezone.utc).isoformat()
        manifest['created_by'] = 'backfill_investing_migration'

        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"  Updated: {MANIFEST_PATH}")

    return file_hash


# =============================================================================
# Checkpointing
# =============================================================================

def load_checkpoint() -> Dict[str, Any]:
    """Load checkpoint from file."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: Dict[str, Any]):
    """Save checkpoint to file."""
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)


# =============================================================================
# Validation
# =============================================================================

def validate_data(
    df: pd.DataFrame,
    column: str,
    expected_range: List[float]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate data is within expected range.

    Args:
        df: DataFrame with column to validate
        column: Column name
        expected_range: [min, max] expected values

    Returns:
        Tuple of (valid_df, warnings)
    """
    warnings = []
    min_val, max_val = expected_range

    # Check for out of range values
    out_of_range = df[
        (df[column] < min_val) | (df[column] > max_val)
    ]

    if len(out_of_range) > 0:
        warnings.append(
            f"{len(out_of_range)} values outside range [{min_val}, {max_val}]"
        )
        # Log some examples
        for _, row in out_of_range.head(5).iterrows():
            warnings.append(f"  {row['fecha']}: {row[column]}")

    # Remove invalid values
    valid_df = df[
        (df[column] >= min_val) & (df[column] <= max_val)
    ].copy()

    return valid_df, warnings


# =============================================================================
# Main Backfill Logic
# =============================================================================

def backfill_indicator(
    indicator: Dict[str, Any],
    start_date: str,
    end_date: str,
    session,
    dry_run: bool = True,
    update_db: bool = False,
    update_seeds_flag: bool = False,
    seeds_df: Optional[pd.DataFrame] = None
) -> Dict[str, Any]:
    """
    Backfill a single indicator from Investing.com.

    Args:
        indicator: Indicator configuration dict
        start_date: Start date YYYY-MM-DD
        end_date: End date YYYY-MM-DD
        session: CloudScraper session
        dry_run: Preview only, no changes
        update_db: Update PostgreSQL
        update_seeds_flag: Update seeds parquet
        seeds_df: Seeds DataFrame (if updating seeds)

    Returns:
        Dict with backfill statistics
    """
    column = indicator['column']
    pair_id = indicator.get('pair_id')
    instrument_id = indicator.get('instrument_id')
    name = indicator['name']
    url = indicator.get('url')
    use_api = indicator.get('use_api', False)
    description = indicator['description']
    expected_range = indicator['expected_range']

    logger.info(f"\n{'='*60}")
    logger.info(f"BACKFILLING: {name} ({description})")
    logger.info(f"{'='*60}")
    logger.info(f"  Column: {column}")
    logger.info(f"  URL: {url}")
    if use_api:
        logger.info(f"  Method: New API (instrument_id={instrument_id})")
    else:
        logger.info(f"  Method: AJAX Endpoint (pair_id={pair_id})")
    logger.info(f"  Date Range: {start_date} to {end_date}")

    result = {
        'indicator': name,
        'column': column,
        'rows': 0,
        'status': 'PENDING',
        'warnings': [],
    }

    try:
        # Choose scraping method based on indicator config
        if use_api and instrument_id:
            # Use new API endpoint (for FX pairs with correct values)
            logger.info(f"  Fetching from Investing.com API (instrument_id={instrument_id})...")
            df = fetch_historical_api(
                instrument_id=instrument_id,
                start_date=start_date,
                end_date=end_date,
                session=session,
                delay=RATE_LIMIT_SECONDS,
                referer_url=url
            )
        else:
            # Use AJAX endpoint (for indices and bonds)
            logger.info(f"  Fetching from Investing.com AJAX endpoint...")
            df = fetch_historical_ajax_chunked(
                pair_id=pair_id,
                start_date=start_date,
                end_date=end_date,
                session=session,
                chunk_days=365,
                delay=RATE_LIMIT_SECONDS,
                referer_url=url
            )

        if df.empty:
            logger.error(f"  No data fetched for {name}")
            result['status'] = 'FAILED'
            result['error'] = 'No data returned from Investing.com'
            return result

        logger.info(f"  Fetched {len(df)} rows")
        logger.info(f"  Date range: {df['fecha'].min()} to {df['fecha'].max()}")

        # Rename column
        df = df.rename(columns={'valor': column})

        # Validate data
        logger.info(f"  Validating data range [{expected_range[0]}, {expected_range[1]}]...")
        df, warnings = validate_data(df, column, expected_range)
        result['warnings'] = warnings

        if warnings:
            for w in warnings[:3]:
                logger.warning(f"    {w}")

        result['rows'] = len(df)

        if dry_run:
            logger.info(f"  [DRY RUN] Would update {len(df)} rows")
            logger.info(f"\n  Preview (first 10 rows):")
            print(df.head(10).to_string(index=False))
            logger.info(f"\n  Preview (last 10 rows):")
            print(df.tail(10).to_string(index=False))
            result['status'] = 'DRY_RUN'
            return result

        # Update database
        if update_db:
            logger.info(f"  Updating PostgreSQL...")
            try:
                rows_upserted = upsert_to_database(df, column)
                logger.info(f"  [DB] Upserted {rows_upserted} rows")
            except Exception as e:
                logger.error(f"  [DB] Error: {e}")
                result['warnings'].append(f"DB error: {str(e)}")

        # Update seeds
        if update_seeds_flag and seeds_df is not None:
            logger.info(f"  Updating seeds parquet...")
            update_seeds(seeds_df, df, column)

        result['status'] = 'SUCCESS'
        logger.info(f"  Backfill completed: {len(df)} rows")

    except Exception as e:
        logger.error(f"  Error backfilling {name}: {e}")
        result['status'] = 'ERROR'
        result['error'] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Backfill macro data from Investing.com (FRED/TwelveData migration)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Preview what would be done
    python scripts/backfill_investing_migration.py --dry-run

    # Backfill all indicators to database
    python scripts/backfill_investing_migration.py --update-db

    # Backfill specific indicator
    python scripts/backfill_investing_migration.py --indicator DXY --update-db

    # Update both database and seeds
    python scripts/backfill_investing_migration.py --update-db --update-seeds

    # Resume from checkpoint
    python scripts/backfill_investing_migration.py --resume --update-db
        """
    )
    parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date (YYYY-MM-DD). Default: 2020-01-01'
    )
    parser.add_argument(
        '--end-date',
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--indicator',
        help='Specific indicator name (DXY, VIX, US10Y, US2Y, USDMXN, USDCLP)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview without making changes'
    )
    parser.add_argument(
        '--update-db',
        action='store_true',
        help='Update PostgreSQL database'
    )
    parser.add_argument(
        '--update-seeds',
        action='store_true',
        help='Update seeds parquet file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip data range validation'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dry_run and not args.update_db and not args.update_seeds:
        logger.info("No action specified. Running in dry-run mode.")
        args.dry_run = True

    print("\n" + "="*70)
    print("INVESTING.COM MIGRATION BACKFILL")
    print("="*70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"Date Range: {args.start_date} to {args.end_date}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    if args.update_db:
        print("  - Will update PostgreSQL")
    if args.update_seeds:
        print("  - Will update seeds parquet")

    # Load checkpoint if resuming
    checkpoint = load_checkpoint() if args.resume else {}
    if args.resume and checkpoint:
        logger.info(f"Resuming from checkpoint with {len(checkpoint)} completed indicators")

    # Filter indicators
    indicators = INDICATORS_TO_MIGRATE.copy()
    if args.indicator:
        indicators = [i for i in indicators if i['name'] == args.indicator.upper()]
        if not indicators:
            available = [i['name'] for i in INDICATORS_TO_MIGRATE]
            logger.error(f"Unknown indicator: {args.indicator}")
            logger.error(f"Available: {', '.join(available)}")
            sys.exit(1)

    # Load seeds if updating
    seeds_df = None
    if args.update_seeds:
        logger.info(f"\nLoading seeds from: {SEEDS_PATH}")
        seeds_df = load_seeds()
        logger.info(f"  Rows in seeds: {len(seeds_df):,}")

    # Create session
    session = create_session()

    # Process each indicator
    results = []
    for indicator in indicators:
        name = indicator['name']

        # Skip if already completed in checkpoint
        if checkpoint.get(name, {}).get('status') == 'SUCCESS' and args.resume:
            logger.info(f"\nSkipping {name} (already completed in checkpoint)")
            continue

        try:
            result = backfill_indicator(
                indicator=indicator,
                start_date=args.start_date,
                end_date=args.end_date,
                session=session,
                dry_run=args.dry_run,
                update_db=args.update_db,
                update_seeds_flag=args.update_seeds,
                seeds_df=seeds_df
            )
            results.append(result)

            # Save checkpoint
            checkpoint[name] = result
            save_checkpoint(checkpoint)

        except Exception as e:
            logger.error(f"Failed to backfill {name}: {e}")
            results.append({
                'indicator': name,
                'rows': 0,
                'status': 'ERROR',
                'error': str(e)
            })

        # Delay between indicators
        if indicator != indicators[-1]:
            logger.info(f"\nWaiting {INTER_INDICATOR_DELAY}s before next indicator...")
            time.sleep(INTER_INDICATOR_DELAY)

    # Save seeds if updated
    if args.update_seeds and seeds_df is not None and not args.dry_run:
        logger.info("\nSaving updated seeds...")
        file_hash = save_seeds(seeds_df)
        logger.info(f"  New SHA256: {file_hash[:16]}...")

    # Print summary
    print("\n" + "="*70)
    print("BACKFILL SUMMARY")
    print("="*70)

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    total_rows = sum(r['rows'] for r in results)

    for r in results:
        status_emoji = {
            'SUCCESS': '[OK]',
            'DRY_RUN': '[DRY]',
            'FAILED': '[FAIL]',
            'ERROR': '[ERR]',
        }.get(r['status'], '[?]')

        print(f"  {status_emoji} {r['indicator']}: {r['rows']:,} rows - {r['status']}")
        if r.get('error'):
            print(f"      Error: {r['error']}")
        if r.get('warnings') and len(r['warnings']) > 0:
            print(f"      Warnings: {len(r['warnings'])}")

    print("-"*70)
    print(f"Total: {success_count}/{len(results)} succeeded, {total_rows:,} rows processed")
    print("="*70)
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")

    if args.dry_run:
        print("\n[DRY RUN] No changes were made.")
        print("Run with --update-db or --update-seeds to apply changes.")

    if args.update_seeds and not args.dry_run:
        print("\nNext steps:")
        print("  1. Verify seeds: python -c \"import pandas as pd; print(pd.read_parquet('seeds/latest/macro_indicators_daily.parquet').tail())\"")
        print("  2. Commit: git add seeds/latest/ && git commit -m 'Backfill Investing.com migration'")
        print("  3. Push: git push")


if __name__ == '__main__':
    main()
