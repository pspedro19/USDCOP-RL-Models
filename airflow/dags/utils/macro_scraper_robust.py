"""
Robust Macro Data Scraper with Fallbacks
=========================================

V20 Implementation: Macro data scraper with retry logic and multiple fallback sources.
Ensures macro data (DXY, VIX, etc.) is always available for model inference.

From: 09_Documento Maestro Completo.md Section 6.6

Features:
- Exponential backoff retry logic
- FRED API fallback for DXY
- Yahoo Finance fallback for VIX
- Database fill for missing dates

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import json

logger = logging.getLogger(__name__)

# API Keys from environment
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '')


class RobustMacroScraper:
    """
    Macro data scraper with retry logic and fallbacks.

    Provides reliable access to macro indicators even when primary
    sources are unavailable.

    Supported indicators:
    - DXY (Dollar Index): Primary source + FRED fallback
    - VIX (Volatility Index): Primary source + Yahoo fallback
    - Brent Oil: Primary source
    - EMBI Colombia: Primary source
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: int = 60,
        timeout: int = 30
    ):
        """
        Initialize the robust scraper.

        Args:
            max_retries: Maximum retry attempts per source
            retry_delay: Base delay between retries (seconds)
            timeout: Request timeout (seconds)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.fred_api_key = FRED_API_KEY
        self.alpha_vantage_key = ALPHA_VANTAGE_KEY

        # Track successful sources for logging
        self._last_sources: Dict[str, str] = {}

    def fetch_with_retry(
        self,
        fetch_func,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Execute fetch function with exponential backoff.

        Args:
            fetch_func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Result from fetch_func or None if all retries failed
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                result = fetch_func(*args, **kwargs)
                if result is not None:
                    return result
            except Exception as e:
                last_error = e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. "
                    f"Retrying in {wait_time}s"
                )
                time.sleep(wait_time)

        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        return None

    # =========================================================================
    # DXY (Dollar Index)
    # =========================================================================

    def fetch_dxy(self, date: datetime) -> Optional[float]:
        """
        Fetch DXY with fallback to FRED API.

        Args:
            date: Date to fetch DXY for

        Returns:
            DXY value or None if unavailable
        """
        # Try primary source first
        dxy = self.fetch_with_retry(self._fetch_dxy_primary, date)
        if dxy is not None:
            self._last_sources['dxy'] = 'primary'
            return dxy

        # Fallback: FRED API
        logger.info("Using FRED API fallback for DXY")
        dxy = self._fetch_dxy_fred(date)
        if dxy is not None:
            self._last_sources['dxy'] = 'fred'
            return dxy

        # Last resort: Use previous day's value
        logger.warning("All DXY sources failed, will use forward-fill")
        self._last_sources['dxy'] = 'forward_fill'
        return None

    def _fetch_dxy_primary(self, date: datetime) -> Optional[float]:
        """Primary source for DXY (placeholder - implement actual source)."""
        # This would connect to your primary data provider
        # For now, return None to trigger fallback
        return None

    def _fetch_dxy_fred(self, date: datetime) -> Optional[float]:
        """
        Fallback: FRED API for DXY (Trade Weighted U.S. Dollar Index).

        Series: DTWEXBGS (Broad, Goods and Services)
        """
        if not self.fred_api_key:
            logger.warning("FRED API key not configured")
            return None

        try:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": "DTWEXBGS",
                "api_key": self.fred_api_key,
                "file_type": "json",
                "observation_start": date.strftime("%Y-%m-%d"),
                "observation_end": date.strftime("%Y-%m-%d"),
            }

            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()

            if data.get("observations"):
                value = data["observations"][0].get("value")
                if value and value != ".":
                    logger.info(f"FRED DXY for {date.date()}: {value}")
                    return float(value)

        except requests.exceptions.RequestException as e:
            logger.error(f"FRED API request error: {e}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"FRED API parse error: {e}")

        return None

    # =========================================================================
    # VIX (Volatility Index)
    # =========================================================================

    def fetch_vix(self, date: datetime) -> Optional[float]:
        """
        Fetch VIX with fallback to Yahoo Finance.

        Args:
            date: Date to fetch VIX for

        Returns:
            VIX value or None if unavailable
        """
        # Try primary source first
        vix = self.fetch_with_retry(self._fetch_vix_primary, date)
        if vix is not None:
            self._last_sources['vix'] = 'primary'
            return vix

        # Fallback: Yahoo Finance
        logger.info("Using Yahoo Finance fallback for VIX")
        vix = self._fetch_vix_yahoo(date)
        if vix is not None:
            self._last_sources['vix'] = 'yahoo'
            return vix

        logger.warning("All VIX sources failed, will use forward-fill")
        self._last_sources['vix'] = 'forward_fill'
        return None

    def _fetch_vix_primary(self, date: datetime) -> Optional[float]:
        """Primary source for VIX (placeholder - implement actual source)."""
        return None

    def _fetch_vix_yahoo(self, date: datetime) -> Optional[float]:
        """
        Fallback: Yahoo Finance for VIX.

        Uses yfinance-style API endpoint.
        """
        try:
            # Yahoo Finance API endpoint
            start_ts = int(date.timestamp())
            end_ts = int((date + timedelta(days=1)).timestamp())

            url = f"https://query1.finance.yahoo.com/v8/finance/chart/%5EVIX"
            params = {
                "period1": start_ts,
                "period2": end_ts,
                "interval": "1d",
            }

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }

            response = requests.get(
                url, params=params, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Parse Yahoo response
            result = data.get("chart", {}).get("result", [])
            if result:
                closes = result[0].get("indicators", {}).get("quote", [{}])[0].get("close", [])
                if closes and closes[0] is not None:
                    logger.info(f"Yahoo VIX for {date.date()}: {closes[0]}")
                    return float(closes[0])

        except requests.exceptions.RequestException as e:
            logger.error(f"Yahoo Finance request error: {e}")
        except (KeyError, IndexError, ValueError, TypeError) as e:
            logger.error(f"Yahoo Finance parse error: {e}")

        return None

    # =========================================================================
    # Database Operations
    # =========================================================================

    def fill_missing_macro(
        self,
        conn,
        days_back: int = 7,
        forward_fill: bool = True
    ) -> Dict[str, int]:
        """
        Fill missing macro data for recent days.

        Args:
            conn: Database connection (psycopg2)
            days_back: Number of days to check for missing data
            forward_fill: Whether to use forward-fill for still-missing values

        Returns:
            Dict with counts of filled values per indicator
        """
        filled_counts = {"dxy": 0, "vix": 0, "forward_filled": 0}

        # Find dates with missing macro data
        query = """
            SELECT fecha FROM macro_indicators_daily
            WHERE fecha > CURRENT_DATE - %s
              AND (fxrt_index_dxy_usa_d_dxy IS NULL OR volt_vix_usa_d_vix IS NULL)
            ORDER BY fecha
        """

        with conn.cursor() as cur:
            cur.execute(query, (days_back,))
            missing_dates = [row[0] for row in cur.fetchall()]

        logger.info(f"Found {len(missing_dates)} dates with missing macro data")

        # Try to fill each missing date
        for date in missing_dates:
            dt = datetime.combine(date, datetime.min.time())
            dxy = self.fetch_dxy(dt)
            vix = self.fetch_vix(dt)

            if dxy is not None or vix is not None:
                update_query = """
                    UPDATE macro_indicators_daily SET
                        fxrt_index_dxy_usa_d_dxy = COALESCE(%s, fxrt_index_dxy_usa_d_dxy),
                        volt_vix_usa_d_vix = COALESCE(%s, volt_vix_usa_d_vix),
                        updated_at = NOW()
                    WHERE fecha = %s
                """
                with conn.cursor() as cur:
                    cur.execute(update_query, (dxy, vix, date))
                    conn.commit()

                if dxy is not None:
                    filled_counts["dxy"] += 1
                if vix is not None:
                    filled_counts["vix"] += 1

                logger.info(f"Updated macro for {date}: DXY={dxy}, VIX={vix}")

        # Forward-fill remaining NULLs
        if forward_fill:
            filled_counts["forward_filled"] = self._forward_fill_nulls(conn, days_back)

        return filled_counts

    def _forward_fill_nulls(self, conn, days_back: int) -> int:
        """
        Forward-fill remaining NULL values using previous day's data.

        Args:
            conn: Database connection
            days_back: Days to check

        Returns:
            Count of forward-filled values
        """
        forward_fill_query = """
            UPDATE macro_indicators_daily m
            SET
                fxrt_index_dxy_usa_d_dxy = COALESCE(
                    m.fxrt_index_dxy_usa_d_dxy,
                    (SELECT fxrt_index_dxy_usa_d_dxy
                     FROM macro_indicators_daily
                     WHERE fecha < m.fecha AND fxrt_index_dxy_usa_d_dxy IS NOT NULL
                     ORDER BY fecha DESC LIMIT 1)
                ),
                volt_vix_usa_d_vix = COALESCE(
                    m.volt_vix_usa_d_vix,
                    (SELECT volt_vix_usa_d_vix
                     FROM macro_indicators_daily
                     WHERE fecha < m.fecha AND volt_vix_usa_d_vix IS NOT NULL
                     ORDER BY fecha DESC LIMIT 1)
                ),
                updated_at = NOW()
            WHERE fecha > CURRENT_DATE - %s
              AND (fxrt_index_dxy_usa_d_dxy IS NULL OR volt_vix_usa_d_vix IS NULL)
        """

        with conn.cursor() as cur:
            cur.execute(forward_fill_query, (days_back,))
            rows_affected = cur.rowcount
            conn.commit()

        if rows_affected > 0:
            logger.info(f"Forward-filled {rows_affected} macro indicator rows")

        return rows_affected

    def get_last_sources(self) -> Dict[str, str]:
        """Get the last source used for each indicator."""
        return self._last_sources.copy()


# Convenience function for Airflow tasks
def fill_missing_macro_data(conn, days_back: int = 7) -> Dict[str, int]:
    """
    Convenience function to fill missing macro data.

    Usage in Airflow DAG:
        from utils.macro_scraper_robust import fill_missing_macro_data
        fill_missing_macro_data(conn, days_back=7)
    """
    scraper = RobustMacroScraper()
    return scraper.fill_missing_macro(conn, days_back)


if __name__ == "__main__":
    # Test the scraper
    logging.basicConfig(level=logging.INFO)

    scraper = RobustMacroScraper()

    # Test DXY fetch
    test_date = datetime.now() - timedelta(days=1)
    print(f"\nTesting DXY fetch for {test_date.date()}:")
    dxy = scraper.fetch_dxy(test_date)
    print(f"  Result: {dxy}")
    print(f"  Source: {scraper.get_last_sources().get('dxy', 'unknown')}")

    # Test VIX fetch
    print(f"\nTesting VIX fetch for {test_date.date()}:")
    vix = scraper.fetch_vix(test_date)
    print(f"  Result: {vix}")
    print(f"  Source: {scraper.get_last_sources().get('vix', 'unknown')}")

    print("\nMacro scraper test complete!")
