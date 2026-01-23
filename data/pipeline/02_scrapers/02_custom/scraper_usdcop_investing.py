# -*- coding: utf-8 -*-
"""
Scraper USDCOP Daily OHLCV - Investing.com
==========================================

Obtiene datos OHLCV diarios OFICIALES de USD/COP desde Investing.com.

Esta es la fuente OFICIAL para el pipeline de Forecasting.
NO usar resample de 5-min para forecasting.

URL: https://www.investing.com/currencies/usd-cop-historical-data

Contract: CTR-SCRAPER-USDCOP-001
Version: 1.0.0

Usage:
    from scraper_usdcop_investing import USDCOPInvestingScraper

    scraper = USDCOPInvestingScraper()
    df = scraper.fetch_ohlcv("2020-01-01", "2025-12-31")
"""

import pandas as pd
import numpy as np
import cloudscraper
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
import logging
import time
import re

logger = logging.getLogger(__name__)


class USDCOPInvestingScraper:
    """
    Scraper para datos OHLCV diarios de USD/COP desde Investing.com.

    Attributes:
        BASE_URL: URL base para datos históricos
        PAIR_ID: ID del par en Investing.com (para API calls si es necesario)

    Example:
        >>> scraper = USDCOPInvestingScraper()
        >>> df = scraper.fetch_ohlcv("2024-01-01", "2024-12-31")
        >>> print(df.columns)
        Index(['date', 'open', 'high', 'low', 'close', 'volume', 'change_pct'])
    """

    BASE_URL = "https://www.investing.com/currencies/usd-cop-historical-data"
    PAIR_ID = "2124"  # Investing.com internal ID for USD/COP

    # Column mapping from Investing.com to standard names
    COLUMN_MAP = {
        'Date': 'date',
        'Price': 'close',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Vol.': 'volume',
        'Change %': 'change_pct',
    }

    def __init__(self, max_retries: int = 3, delay_between_requests: float = 1.0):
        """
        Initialize scraper.

        Args:
            max_retries: Maximum retry attempts per request
            delay_between_requests: Delay in seconds between requests
        """
        self.max_retries = max_retries
        self.delay = delay_between_requests
        self.scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.investing.com/currencies/usd-cop',
            'Connection': 'keep-alive',
        }

    def fetch_ohlcv(
        self,
        start_date: str,
        end_date: str,
        progress_callback: callable = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            progress_callback: Optional callback for progress updates

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, change_pct

        Note:
            For large date ranges, this will make multiple requests
            to handle Investing.com's pagination.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        logger.info(f"Fetching USDCOP OHLCV from {start_date} to {end_date}")

        all_data = []

        # Investing.com returns ~60-90 days per page, so we chunk by year
        current_start = start
        chunk_count = 0
        total_chunks = ((end - start).days // 365) + 1

        while current_start < end:
            chunk_end = min(current_start + timedelta(days=365), end)

            chunk_count += 1
            if progress_callback:
                progress_callback(chunk_count, total_chunks)

            logger.info(f"Fetching chunk {chunk_count}: {current_start.date()} to {chunk_end.date()}")

            df_chunk = self._fetch_chunk(
                current_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d")
            )

            if df_chunk is not None and len(df_chunk) > 0:
                all_data.append(df_chunk)
                logger.info(f"  Got {len(df_chunk)} rows")
            else:
                logger.warning(f"  No data for this chunk")

            current_start = chunk_end + timedelta(days=1)

            # Delay between requests to be respectful
            if current_start < end:
                time.sleep(self.delay)

        if not all_data:
            logger.warning("No data fetched from Investing.com")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)

        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)

        # Validate data
        df = self._validate_and_clean(df)

        logger.info(f"Total rows fetched: {len(df)}")
        return df

    def _fetch_chunk(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Fetch a single chunk of data."""
        for attempt in range(self.max_retries):
            try:
                # Method 1: Direct page scraping
                df = self._scrape_historical_page()
                if df is not None and len(df) > 0:
                    # Filter to date range
                    df['date'] = pd.to_datetime(df['date'])
                    mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                    return df[mask].copy()

                # Method 2: Try AJAX endpoint (if direct fails)
                # Investing.com uses AJAX for loading more data
                df = self._fetch_via_ajax(start_date, end_date)
                if df is not None:
                    return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * 2)

        return None

    def _scrape_historical_page(self) -> Optional[pd.DataFrame]:
        """Scrape the main historical data page."""
        try:
            response = self.scraper.get(
                self.BASE_URL,
                headers=self.headers,
                timeout=30
            )

            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find the historical data table
            # Investing.com uses different table classes
            table = None

            # Try different selectors
            selectors = [
                ('table', {'data-test': 'historical-data-table'}),
                ('table', {'class': 'freeze-column-w-1'}),
                ('table', {'class': 'genTbl'}),
                ('table', {'id': 'curr_table'}),
            ]

            for tag, attrs in selectors:
                table = soup.find(tag, attrs)
                if table:
                    break

            # Fallback: find largest table
            if not table:
                tables = soup.find_all('table')
                if tables:
                    table = max(tables, key=lambda t: len(t.find_all('tr')))

            if not table:
                logger.warning("No table found on page")
                return None

            # Parse table
            return self._parse_table(table)

        except Exception as e:
            logger.error(f"Error scraping page: {e}")
            return None

    def _parse_table(self, table) -> pd.DataFrame:
        """Parse HTML table into DataFrame."""
        rows = table.find_all('tr')

        if len(rows) < 2:
            return None

        # Get headers
        header_row = rows[0]
        headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]

        # Parse data rows
        data = []
        for row in rows[1:]:
            cols = row.find_all('td')
            if len(cols) >= 5:  # At minimum: Date, Price, Open, High, Low
                row_data = {}
                for i, col in enumerate(cols):
                    if i < len(headers):
                        row_data[headers[i]] = col.get_text(strip=True)
                if row_data:
                    data.append(row_data)

        if not data:
            return None

        df = pd.DataFrame(data)

        # Rename columns to standard names
        df = df.rename(columns=self.COLUMN_MAP)

        # Convert data types
        df = self._convert_types(df)

        return df

    def _convert_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert column types to proper formats."""
        df = df.copy()

        # Date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                # Remove commas and convert
                df[col] = df[col].astype(str).str.replace(',', '').str.replace(' ', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Volume (may have K, M suffixes)
        if 'volume' in df.columns:
            df['volume'] = self._parse_volume(df['volume'])

        # Change percentage
        if 'change_pct' in df.columns:
            df['change_pct'] = df['change_pct'].astype(str).str.replace('%', '').str.replace(',', '')
            df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')

        return df

    def _parse_volume(self, volume_series: pd.Series) -> pd.Series:
        """Parse volume with K/M suffixes."""
        def parse_vol(v):
            if pd.isna(v) or v == '-' or v == '':
                return 0
            v = str(v).upper().replace(',', '').strip()
            if v.endswith('K'):
                return float(v[:-1]) * 1000
            elif v.endswith('M'):
                return float(v[:-1]) * 1000000
            elif v.endswith('B'):
                return float(v[:-1]) * 1000000000
            else:
                try:
                    return float(v)
                except:
                    return 0

        return volume_series.apply(parse_vol).astype(int)

    def _fetch_via_ajax(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch data via Investing.com's AJAX endpoint.

        This is used when the page doesn't have all historical data.
        """
        try:
            # Investing.com AJAX endpoint for historical data
            ajax_url = "https://www.investing.com/instruments/HistoricalDataAjax"

            # Convert dates to required format
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")

            payload = {
                'curr_id': self.PAIR_ID,
                'smlID': '1540783',  # May need to be dynamic
                'header': 'USD/COP Historical Data',
                'st_date': start.strftime("%m/%d/%Y"),
                'end_date': end.strftime("%m/%d/%Y"),
                'interval_sec': 'Daily',
                'sort_col': 'date',
                'sort_ord': 'DESC',
                'action': 'historical_data',
            }

            ajax_headers = self.headers.copy()
            ajax_headers['X-Requested-With'] = 'XMLHttpRequest'
            ajax_headers['Content-Type'] = 'application/x-www-form-urlencoded'

            response = self.scraper.post(
                ajax_url,
                data=payload,
                headers=ajax_headers,
                timeout=30
            )

            if response.status_code != 200:
                return None

            # Parse the HTML response
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table')

            if table:
                return self._parse_table(table)

            return None

        except Exception as e:
            logger.debug(f"AJAX fetch failed: {e}")
            return None

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data."""
        df = df.copy()

        # Remove rows with invalid dates
        df = df.dropna(subset=['date'])

        # Remove rows with all NaN prices
        price_cols = ['open', 'high', 'low', 'close']
        df = df.dropna(subset=price_cols, how='all')

        # Forward fill missing OHLC values (rare but possible)
        df[price_cols] = df[price_cols].ffill()

        # Validate OHLC relationships
        # High should be >= Open, Close, Low
        # Low should be <= Open, Close, High
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )

        if invalid_mask.any():
            logger.warning(f"Found {invalid_mask.sum()} rows with invalid OHLC relationships")
            # Fix by adjusting high/low
            df.loc[invalid_mask, 'high'] = df.loc[invalid_mask, ['open', 'high', 'low', 'close']].max(axis=1)
            df.loc[invalid_mask, 'low'] = df.loc[invalid_mask, ['open', 'high', 'low', 'close']].min(axis=1)

        # Validate price range (USDCOP typically 3000-6000)
        valid_range = (df['close'] >= 2000) & (df['close'] <= 7000)
        if not valid_range.all():
            logger.warning(f"Found {(~valid_range).sum()} rows with prices outside expected range")

        # Ensure volume is non-negative
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)
        else:
            df['volume'] = 0

        # Add source column
        df['source'] = 'investing'

        # Final sort
        df = df.sort_values('date').reset_index(drop=True)

        return df

    def fetch_latest(self, n_days: int = 30) -> pd.DataFrame:
        """
        Fetch the latest N days of data.

        Args:
            n_days: Number of days to fetch

        Returns:
            DataFrame with OHLCV data
        """
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=n_days)).strftime("%Y-%m-%d")
        return self.fetch_ohlcv(start_date, end_date)

    def test_connection(self) -> Tuple[bool, str]:
        """
        Test if scraper can connect to Investing.com.

        Returns:
            Tuple of (success, message)
        """
        try:
            df = self.fetch_latest(7)
            if df is not None and len(df) > 0:
                return True, f"Success: fetched {len(df)} rows"
            else:
                return False, "No data returned"
        except Exception as e:
            return False, str(e)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fetch_usdcop_daily(
    start_date: str = "2020-01-01",
    end_date: str = None
) -> pd.DataFrame:
    """
    Convenience function to fetch USDCOP daily OHLCV.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today

    Returns:
        DataFrame with OHLCV data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    scraper = USDCOPInvestingScraper()
    return scraper.fetch_ohlcv(start_date, end_date)


def update_daily_ohlcv_table(
    connection_string: str,
    start_date: str = None
) -> int:
    """
    Update bi.dim_daily_usdcop table with latest data from Investing.com.

    Args:
        connection_string: PostgreSQL connection string
        start_date: Start date for backfill (None = last 30 days)

    Returns:
        Number of rows inserted/updated
    """
    import psycopg2
    from psycopg2.extras import execute_values

    # Determine date range
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch data
    scraper = USDCOPInvestingScraper()
    df = scraper.fetch_ohlcv(start_date, end_date)

    if df.empty:
        logger.warning("No data to insert")
        return 0

    # Insert into database
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
    values = [
        (
            row['date'].date(),
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            int(row['volume']),
            'investing'
        )
        for _, row in df.iterrows()
    ]

    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            execute_values(cur, insert_sql, values)
            rows_affected = cur.rowcount
        conn.commit()

    logger.info(f"Inserted/updated {rows_affected} rows")
    return rows_affected


# =============================================================================
# CLI / TEST
# =============================================================================

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("=" * 60)
    print("USDCOP Investing.com Scraper Test")
    print("=" * 60)

    scraper = USDCOPInvestingScraper()

    # Test connection
    print("\n1. Testing connection...")
    success, msg = scraper.test_connection()
    print(f"   Result: {msg}")

    if success:
        # Fetch last 30 days
        print("\n2. Fetching last 30 days...")
        df = scraper.fetch_latest(30)

        if not df.empty:
            print(f"   Rows: {len(df)}")
            print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
            print(f"   Columns: {df.columns.tolist()}")
            print("\n   Sample data:")
            print(df.head(10).to_string(index=False))

            print("\n   Statistics:")
            print(f"   Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
            print(f"   Average close: {df['close'].mean():.2f}")
        else:
            print("   No data returned")
    else:
        print("   Connection test failed")

    print("\n" + "=" * 60)
