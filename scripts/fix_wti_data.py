#!/usr/bin/env python3
"""
Fix WTI Oil Data - Complete Historical Backfill
================================================

This script corrects ALL historical WTI oil data that was corrupted by TwelveData
returning Colgate-Palmolive stock prices (NYSE:CL) instead of WTI crude oil.

Root Cause:
    The symbol "CL" in TwelveData is interpreted as NYSE:CL (Colgate-Palmolive)
    instead of crude oil. The correct symbols are:
    - WTI/USD (spot price) - FIXED in l0_macro_sources.yaml
    - CL=F (Yahoo Finance futures symbol)

Data Sources (in priority order):
    1. Yahoo Finance (CL=F) - Most reliable, full history from 2000
    2. Investing.com (cloudscraper) - Fallback for recent data only

This script:
    1. Fetches COMPLETE historical WTI prices from Yahoo Finance
    2. Updates the seeds parquet file with correct values
    3. Optionally updates the PostgreSQL database
    4. Regenerates the manifest with new hash

Usage:
    python scripts/fix_wti_data.py --dry-run           # Show comparison only
    python scripts/fix_wti_data.py --update-seeds      # Update seeds parquet
    python scripts/fix_wti_data.py --update-db         # Update PostgreSQL
    python scripts/fix_wti_data.py --update-seeds --update-db  # Both

Created: 2026-01-22
Author: Claude Code
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import yfinance as yf

try:
    import cloudscraper
    from bs4 import BeautifulSoup
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# Configuration
WTI_COLUMN = "comm_oil_wti_glb_d_wti"
SEEDS_PATH = Path("seeds/latest/macro_indicators_daily.parquet")
MANIFEST_PATH = Path("seeds/latest/manifest.json")

# Yahoo Finance symbol for WTI Crude Oil Futures
YAHOO_WTI_SYMBOL = "CL=F"

# Investing.com configuration
INVESTING_WTI_URL = "https://www.investing.com/commodities/crude-oil-historical-data"
INVESTING_PAIR_ID = 8833  # WTI Crude Oil pair ID


def fetch_wti_investing_com(start_date: str = "2020-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Fetch WTI historical data from Investing.com using cloudscraper.

    Uses the HistoricalDataAjax endpoint with POST request and date filters.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with columns: [fecha, comm_oil_wti_glb_d_wti]
    """
    if not CLOUDSCRAPER_AVAILABLE:
        raise ImportError("cloudscraper and beautifulsoup4 required. Install with: pip install cloudscraper beautifulsoup4")

    print(f"\n{'='*60}")
    print("FETCHING WTI DATA FROM INVESTING.COM")
    print(f"{'='*60}")
    print(f"URL: {INVESTING_WTI_URL}")
    print(f"Pair ID: {INVESTING_PAIR_ID}")

    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Convert dates to MM/DD/YYYY format for Investing.com
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    scraper = cloudscraper.create_scraper(
        browser={'browser': 'chrome', 'platform': 'windows', 'mobile': False}
    )

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'X-Requested-With': 'XMLHttpRequest',
        'Content-Type': 'application/x-www-form-urlencoded',
    }

    # Fetch data year by year to avoid timeouts
    current_year = start_dt.year
    end_year = end_dt.year

    for year in range(current_year, end_year + 1):
        year_start = f"{max(start_dt, datetime(year, 1, 1)).strftime('%m/%d/%Y')}"
        year_end = f"{min(end_dt, datetime(year, 12, 31)).strftime('%m/%d/%Y')}"

        print(f"\n  Fetching {year}: {year_start} to {year_end}")

        # POST payload for HistoricalDataAjax
        payload = {
            'curr_id': INVESTING_PAIR_ID,
            'smlID': str(random.randint(1000000, 99999999)),
            'header': 'Crude Oil WTI Futures Historical Data',
            'st_date': year_start,
            'end_date': year_end,
            'interval_sec': 'Daily',
            'sort_col': 'date',
            'sort_ord': 'DESC',
            'action': 'historical_data'
        }

        try:
            # First, visit the main page to get cookies
            scraper.get(INVESTING_WTI_URL, headers=headers, timeout=30)
            time.sleep(1)

            # Then POST to the AJAX endpoint
            ajax_url = "https://www.investing.com/instruments/HistoricalDataAjax"
            response = scraper.post(ajax_url, data=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table')

                if table:
                    rows = table.find_all('tr')[1:]  # Skip header
                    year_count = 0

                    for row in rows:
                        cols = row.find_all('td')
                        if len(cols) >= 2:
                            try:
                                date_str = cols[0].get_text(strip=True)
                                price_str = cols[1].get_text(strip=True).replace(',', '')

                                fecha = pd.to_datetime(date_str)
                                price = float(price_str)

                                all_data.append({
                                    'fecha': fecha,
                                    WTI_COLUMN: price
                                })
                                year_count += 1
                            except (ValueError, AttributeError):
                                continue

                    print(f"    -> {year_count} records")
                else:
                    print(f"    -> No table found in response")

                    # Fallback: Try scraping the visible HTML page
                    print(f"    -> Trying HTML page scrape fallback...")
                    page_data = _scrape_investing_html_page(scraper, headers)
                    if page_data:
                        all_data.extend(page_data)
                        print(f"    -> Got {len(page_data)} records from HTML page")

            else:
                print(f"    -> HTTP {response.status_code}")

            # Rate limiting
            time.sleep(2)

        except Exception as e:
            print(f"    -> Error: {e}")

    if not all_data:
        # Final fallback: scrape the visible HTML page
        print("\n  AJAX failed, trying direct HTML scrape...")
        all_data = _scrape_investing_html_page(scraper, headers)

    if not all_data:
        raise ValueError("Could not fetch any data from Investing.com")

    # Create DataFrame
    df = pd.DataFrame(all_data)
    df['fecha'] = pd.to_datetime(df['fecha']).dt.normalize()
    df = df.drop_duplicates(subset='fecha', keep='last')
    df = df.sort_values('fecha').reset_index(drop=True)

    print(f"\nResults:")
    print(f"  Total records: {len(df):,}")
    print(f"  Date range: {df['fecha'].min().date()} to {df['fecha'].max().date()}")
    print(f"  Price range: ${df[WTI_COLUMN].min():.2f} - ${df[WTI_COLUMN].max():.2f}")

    return df


def _scrape_investing_html_page(scraper, headers) -> list:
    """
    Fallback: Scrape the visible HTML table from Investing.com page.

    Returns list of dicts with fecha and WTI price.
    """
    data = []

    try:
        response = scraper.get(INVESTING_WTI_URL, headers=headers, timeout=30)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Try different table selectors
            table = soup.find('table', class_='freeze-column-w-1')
            if not table:
                table = soup.find('table', {'data-test': 'historical-data-table'})
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = max(tables, key=lambda t: len(str(t)))

            if table:
                rows = table.find_all('tr')[1:100]  # Get up to 100 rows

                for row in rows:
                    cols = row.find_all('td')
                    if len(cols) >= 2:
                        try:
                            date_str = cols[0].get_text(strip=True)
                            price_str = cols[1].get_text(strip=True).replace(',', '')

                            fecha = pd.to_datetime(date_str)
                            price = float(price_str)

                            data.append({
                                'fecha': fecha,
                                WTI_COLUMN: price
                            })
                        except (ValueError, AttributeError):
                            continue

    except Exception as e:
        print(f"    HTML scrape error: {e}")

    return data


def fetch_wti_yahoo_finance(start_date: str = "2000-01-01") -> pd.DataFrame:
    """
    Fetch complete WTI historical data from Yahoo Finance.

    Args:
        start_date: Start date for historical data (YYYY-MM-DD)

    Returns:
        DataFrame with columns: [fecha, comm_oil_wti_glb_d_wti]
    """
    print(f"\n{'='*60}")
    print("FETCHING WTI DATA FROM YAHOO FINANCE")
    print(f"{'='*60}")
    print(f"Symbol: {YAHOO_WTI_SYMBOL} (WTI Crude Oil Futures)")
    print(f"Start date: {start_date}")

    try:
        ticker = yf.Ticker(YAHOO_WTI_SYMBOL)
        df = ticker.history(start=start_date, end=None)

        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")

        # Process the data
        df = df.reset_index()
        df = df.rename(columns={'Date': 'fecha', 'Close': WTI_COLUMN})
        df = df[['fecha', WTI_COLUMN]]

        # Convert timezone-aware datetime to date
        df['fecha'] = pd.to_datetime(df['fecha']).dt.tz_localize(None).dt.normalize()

        # Remove duplicates and sort
        df = df.drop_duplicates(subset='fecha', keep='last')
        df = df.sort_values('fecha').reset_index(drop=True)

        print(f"\nResults:")
        print(f"  Total records: {len(df):,}")
        print(f"  Date range: {df['fecha'].min().date()} to {df['fecha'].max().date()}")
        print(f"  Price range: ${df[WTI_COLUMN].min():.2f} - ${df[WTI_COLUMN].max():.2f}")
        print(f"  Current price: ${df[WTI_COLUMN].iloc[-1]:.2f}")

        return df

    except Exception as e:
        print(f"\nERROR: Failed to fetch from Yahoo Finance: {e}")
        raise


def load_seeds() -> pd.DataFrame:
    """Load the current seeds parquet file."""
    if not SEEDS_PATH.exists():
        raise FileNotFoundError(f"Seeds file not found: {SEEDS_PATH}")

    df = pd.read_parquet(SEEDS_PATH)
    df['fecha'] = pd.to_datetime(df['fecha'])
    return df


def show_comparison(seeds_df: pd.DataFrame, wti_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Show comparison between corrupted and correct WTI values.

    Returns:
        Tuple of (comparison_df, stats_dict)
    """
    print(f"\n{'='*60}")
    print("COMPARISON: Corrupted (Colgate-Palmolive) vs Correct (WTI Oil)")
    print(f"{'='*60}")

    # Merge on fecha
    comparison = pd.merge(
        seeds_df[['fecha', WTI_COLUMN]].rename(columns={WTI_COLUMN: 'corrupted'}),
        wti_df[['fecha', WTI_COLUMN]].rename(columns={WTI_COLUMN: 'correct'}),
        on='fecha',
        how='inner'
    )

    # Calculate differences
    comparison['diff'] = comparison['corrupted'] - comparison['correct']
    comparison['diff_pct'] = (comparison['diff'] / comparison['correct'] * 100).round(1)

    # Filter to rows where both have data
    comparison = comparison.dropna()

    # Show last 20 records
    print("\nLast 20 records with both values:")
    print(comparison.tail(20).to_string(index=False))

    # Statistics
    stats = {
        'total_rows_with_both': len(comparison),
        'avg_error_usd': comparison['diff'].mean(),
        'avg_error_pct': comparison['diff_pct'].mean(),
        'max_error_usd': comparison['diff'].max(),
        'min_error_usd': comparison['diff'].min(),
    }

    print(f"\nStatistics:")
    print(f"  Rows with both values: {stats['total_rows_with_both']:,}")
    print(f"  Average error: ${stats['avg_error_usd']:.2f} ({stats['avg_error_pct']:.1f}%)")
    print(f"  Max error: ${stats['max_error_usd']:.2f}")
    print(f"  Min error: ${stats['min_error_usd']:.2f}")

    return comparison, stats


def update_seeds(seeds_df: pd.DataFrame, wti_df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the seeds DataFrame with correct WTI values.

    Args:
        seeds_df: Original seeds DataFrame
        wti_df: Correct WTI data from Yahoo Finance

    Returns:
        Updated seeds DataFrame
    """
    print(f"\n{'='*60}")
    print("UPDATING SEEDS WITH CORRECT WTI DATA")
    print(f"{'='*60}")

    # Create a copy
    updated_df = seeds_df.copy()

    # Create lookup dict from wti_df
    wti_lookup = dict(zip(wti_df['fecha'], wti_df[WTI_COLUMN]))

    # Count updates
    updated_count = 0
    new_count = 0

    for idx, row in updated_df.iterrows():
        fecha = row['fecha']
        if fecha in wti_lookup:
            old_value = row[WTI_COLUMN]
            new_value = wti_lookup[fecha]

            if pd.isna(old_value):
                new_count += 1
            elif abs(old_value - new_value) > 0.01:  # Different value
                updated_count += 1

            updated_df.at[idx, WTI_COLUMN] = new_value

    print(f"\nResults:")
    print(f"  Values corrected: {updated_count:,}")
    print(f"  New values added: {new_count:,}")
    print(f"  Total WTI values: {updated_df[WTI_COLUMN].notna().sum():,}")

    return updated_df


def save_seeds(df: pd.DataFrame) -> str:
    """
    Save updated seeds and regenerate manifest.

    Returns:
        SHA256 hash of the new file
    """
    print(f"\n{'='*60}")
    print("SAVING UPDATED SEEDS")
    print(f"{'='*60}")

    # Save parquet
    df.to_parquet(SEEDS_PATH, index=False)
    print(f"  Saved: {SEEDS_PATH}")

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

        # Update date range
        manifest['tables']['macro_indicators_daily']['date_range'] = [
            df['fecha'].min().strftime('%Y-%m-%d'),
            df['fecha'].max().strftime('%Y-%m-%d')
        ]

        manifest['version'] = datetime.now(timezone.utc).strftime('%Y.%m.%d')
        manifest['created_at'] = datetime.now(timezone.utc).isoformat()
        manifest['created_by'] = 'fix_wti_data_script'

        with open(MANIFEST_PATH, 'w') as f:
            json.dump(manifest, f, indent=2)
        print(f"  Updated: {MANIFEST_PATH}")

    print(f"  New SHA256: {file_hash[:16]}...")

    return file_hash


def update_database(wti_df: pd.DataFrame) -> int:
    """
    Update the PostgreSQL database with correct WTI data.

    Returns:
        Number of rows updated
    """
    import psycopg2

    print(f"\n{'='*60}")
    print("UPDATING DATABASE")
    print(f"{'='*60}")

    db_config = {
        'host': os.environ.get('POSTGRES_HOST', 'localhost'),
        'database': os.environ.get('POSTGRES_DB', 'usdcop_trading'),
        'user': os.environ.get('POSTGRES_USER', 'admin'),
        'password': os.environ.get('POSTGRES_PASSWORD', 'admin123'),
        'port': os.environ.get('POSTGRES_PORT', '5432'),
    }

    print(f"  Host: {db_config['host']}:{db_config['port']}")
    print(f"  Database: {db_config['database']}")

    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()

        # Update in batches
        updated = 0
        batch_size = 1000

        for i in range(0, len(wti_df), batch_size):
            batch = wti_df.iloc[i:i+batch_size]

            for _, row in batch.iterrows():
                cursor.execute(f"""
                    UPDATE macro_indicators_daily
                    SET {WTI_COLUMN} = %s, updated_at = NOW()
                    WHERE fecha = %s
                """, (float(row[WTI_COLUMN]), row['fecha'].date()))
                updated += cursor.rowcount

            conn.commit()
            print(f"  Processed {min(i+batch_size, len(wti_df)):,}/{len(wti_df):,} rows...")

        cursor.close()
        conn.close()

        print(f"\n  Total rows updated: {updated:,}")
        return updated

    except Exception as e:
        print(f"\n  ERROR: Database update failed: {e}")
        print(f"  Make sure PostgreSQL is running and credentials are correct")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Fix corrupted WTI oil data in seeds and database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/fix_wti_data.py --dry-run
    python scripts/fix_wti_data.py --update-seeds
    python scripts/fix_wti_data.py --update-seeds --update-db
    python scripts/fix_wti_data.py --update-seeds --source investing
    python scripts/fix_wti_data.py --update-seeds --source yahoo
        """
    )
    parser.add_argument('--update-seeds', action='store_true',
                       help='Update seeds parquet file with correct WTI data')
    parser.add_argument('--update-db', action='store_true',
                       help='Update PostgreSQL database with correct WTI data')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show comparison without making any changes')
    parser.add_argument('--start-date', default='2000-01-01',
                       help='Start date for historical data (default: 2000-01-01)')
    parser.add_argument('--source', choices=['yahoo', 'investing', 'both'], default='yahoo',
                       help='Data source: yahoo (recommended), investing, or both (default: yahoo)')
    args = parser.parse_args()

    if not args.update_seeds and not args.update_db and not args.dry_run:
        parser.print_help()
        print("\nError: Specify at least one action (--update-seeds, --update-db, or --dry-run)")
        sys.exit(1)

    print("=" * 70)
    print("FIX WTI OIL DATA - COMPLETE HISTORICAL BACKFILL")
    print("=" * 70)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print(f"\nRoot Cause: TwelveData symbol 'CL' = Colgate-Palmolive, NOT crude oil")
    print(f"Data Source: {args.source.upper()}")

    # Fetch correct WTI data based on source selection
    wti_df = None

    if args.source in ['yahoo', 'both']:
        print(f"\n>>> Using Yahoo Finance (CL=F)")
        try:
            wti_df = fetch_wti_yahoo_finance(start_date=args.start_date)
        except Exception as e:
            print(f"Yahoo Finance failed: {e}")
            if args.source == 'yahoo':
                raise

    if args.source in ['investing', 'both']:
        print(f"\n>>> Using Investing.com")
        try:
            investing_df = fetch_wti_investing_com(start_date=args.start_date)

            if wti_df is None:
                wti_df = investing_df
            else:
                # Merge: prefer Yahoo for older data, Investing for recent data
                print("\n>>> Merging data sources (Yahoo + Investing.com)")
                # Use Investing.com for dates not in Yahoo
                existing_dates = set(wti_df['fecha'])
                new_data = investing_df[~investing_df['fecha'].isin(existing_dates)]
                if len(new_data) > 0:
                    wti_df = pd.concat([wti_df, new_data], ignore_index=True)
                    wti_df = wti_df.sort_values('fecha').reset_index(drop=True)
                    print(f"  Added {len(new_data)} new records from Investing.com")

        except Exception as e:
            print(f"Investing.com failed: {e}")
            if args.source == 'investing':
                raise

    if wti_df is None or wti_df.empty:
        print("\nERROR: Could not fetch WTI data from any source")
        sys.exit(1)

    # Load current seeds
    print(f"\nLoading seeds from: {SEEDS_PATH}")
    seeds_df = load_seeds()
    print(f"  Rows in seeds: {len(seeds_df):,}")
    print(f"  WTI values present: {seeds_df[WTI_COLUMN].notna().sum():,}")

    # Show comparison
    comparison, stats = show_comparison(seeds_df, wti_df)

    if args.dry_run:
        print(f"\n{'='*70}")
        print("[DRY RUN] No changes made")
        print("="*70)
        return

    # Update seeds
    if args.update_seeds:
        updated_seeds = update_seeds(seeds_df, wti_df)
        save_seeds(updated_seeds)

    # Update database
    if args.update_db:
        update_database(wti_df)

    print(f"\n{'='*70}")
    print("COMPLETED SUCCESSFULLY")
    print(f"{'='*70}")
    print(f"Finished: {datetime.now(timezone.utc).isoformat()}")

    if args.update_seeds:
        print(f"\nNext steps:")
        print(f"  1. Verify seeds: python -c \"import pandas as pd; df=pd.read_parquet('{SEEDS_PATH}'); print(df[['fecha','{WTI_COLUMN}']].tail(10))\"")
        print(f"  2. Commit changes: git add seeds/latest/ && git commit -m 'Fix WTI data corruption'")
        print(f"  3. Push to repo: git push")


if __name__ == '__main__':
    main()
