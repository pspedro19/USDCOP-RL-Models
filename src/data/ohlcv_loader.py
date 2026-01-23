"""
Unified OHLCV Loader - SSOT for USD/COP price data.

Contract: CTR-DATA-004
Version: 3.0.0

ARCHITECTURE 10/10:
====================

This loader provides TWO distinct data paths:

1. RL Pipeline (5-minute):
   - Source: usdcop_m5_ohlcv (TwelveData API)
   - Method: load_5min()
   - Use case: Intraday trading, reinforcement learning

2. Forecasting Pipeline (daily):
   - Source: bi.dim_daily_usdcop (Investing.com OFFICIAL values)
   - Method: load_daily()
   - Use case: Daily forecasting with official close prices

CRITICAL: Forecasting should ALWAYS use load_daily() for official values,
NOT resample_to_daily() which is only for fallback/validation.

Data Lineage:
-------------
    Investing.com → bi.dim_daily_usdcop → load_daily() → Forecasting
    TwelveData    → usdcop_m5_ohlcv     → load_5min()  → RL Training

Usage:
    from src.data import UnifiedOHLCVLoader

    loader = UnifiedOHLCVLoader()

    # For RL (5-min) - TwelveData
    df_5min = loader.load_5min("2024-01-01", "2024-12-31")

    # For Forecasting (daily) - Investing.com OFFICIAL
    df_daily = loader.load_daily("2024-01-01", "2024-12-31")
"""

import os
import gzip
import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

import pandas as pd
import numpy as np

from .calendar import TradingCalendar, filter_market_hours
from .contracts import OHLCV_COLUMNS, OHLCV_REQUIRED, validate_ohlcv_columns

logger = logging.getLogger(__name__)


class UnifiedOHLCVLoader:
    """
    Unified OHLCV loader for RL and Forecasting pipelines.

    Loads from PostgreSQL as primary source, with CSV fallback.
    Supports both 5-min (RL) and daily (Forecasting) frequencies.

    Attributes:
        connection_string: PostgreSQL connection string
        fallback_csv: If True, fall back to CSV when DB unavailable
        csv_path: Path to CSV backup file

    Example:
        >>> loader = UnifiedOHLCVLoader()
        >>> df = loader.load_5min("2024-01-01", "2024-06-30")
        >>> print(f"Loaded {len(df)} bars")
        Loaded 15000 bars
    """

    # Default paths
    DEFAULT_CSV_PATH = Path("seeds/latest/usdcop_m5_ohlcv.parquet")
    DEFAULT_CSV_GZ_PATH = Path("data/backups/usdcop_m5_ohlcv_latest.csv.gz")

    def __init__(
        self,
        connection_string: Optional[str] = None,
        fallback_csv: bool = True,
        csv_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize OHLCV loader.

        Args:
            connection_string: PostgreSQL connection string.
                If None, uses DATABASE_URL environment variable.
            fallback_csv: If True, fall back to CSV when DB unavailable.
            csv_path: Path to CSV/Parquet backup file.
        """
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.fallback_csv = fallback_csv
        self.csv_path = Path(csv_path) if csv_path else None
        self.calendar = TradingCalendar()

    def load_5min(
        self,
        start_date: str,
        end_date: str,
        filter_market_hours: bool = True
    ) -> pd.DataFrame:
        """
        Load 5-minute OHLCV data.

        Primary: PostgreSQL (usdcop_m5_ohlcv)
        Fallback: Parquet/CSV backup

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            filter_market_hours: If True, filter to 13:00-18:00 UTC

        Returns:
            DataFrame with columns: time, open, high, low, close, volume

        Raises:
            ValueError: If no data source available
        """
        df = None

        # Try DB first
        if self.connection_string:
            try:
                df = self._load_from_db(start_date, end_date)
                logger.info(f"Loaded {len(df)} rows from PostgreSQL")
            except Exception as e:
                logger.warning(f"DB load failed: {e}")
                if not self.fallback_csv:
                    raise

        # Fallback to CSV/Parquet
        if df is None and self.fallback_csv:
            df = self._load_from_file(start_date, end_date)
            logger.info(f"Loaded {len(df)} rows from file backup")

        if df is None:
            raise ValueError(
                "No data source available. Set DATABASE_URL or provide CSV path."
            )

        # Validate columns
        is_valid, missing = validate_ohlcv_columns(df.columns.tolist())
        if not is_valid:
            raise ValueError(f"Missing required OHLCV columns: {missing}")

        # Filter to market hours if requested
        if filter_market_hours:
            df = self.calendar.filter_market_hours(df, datetime_col='time')

        return df

    def _load_from_db(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load from PostgreSQL database."""
        import psycopg2

        query = """
            SELECT
                time,
                open,
                high,
                low,
                close,
                COALESCE(volume, 0) as volume
            FROM usdcop_m5_ohlcv
            WHERE time >= %s AND time <= %s
            ORDER BY time ASC
        """

        with psycopg2.connect(self.connection_string) as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=(f"{start_date} 00:00:00", f"{end_date} 23:59:59")
            )

        df['time'] = pd.to_datetime(df['time'])
        return df

    def _load_from_file(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Load from CSV or Parquet backup file."""
        # Determine file path
        path = self._find_backup_file()

        if path is None:
            raise FileNotFoundError("No backup file found")

        # Load based on extension
        if path.suffix == '.parquet':
            df = pd.read_parquet(path)
        elif path.suffix == '.gz':
            with gzip.open(path, 'rt') as f:
                df = pd.read_csv(f)
        elif path.suffix == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Normalize column names
        df = self._normalize_columns(df)

        # Convert time column
        df['time'] = pd.to_datetime(df['time'])

        # Filter to date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        df = df[(df['time'] >= start) & (df['time'] < end)]

        return df

    def load_daily(
        self,
        start_date: str,
        end_date: str,
        source: str = None
    ) -> pd.DataFrame:
        """
        Load OFFICIAL daily OHLCV data from bi.dim_daily_usdcop.

        This is the PRIMARY method for Forecasting pipeline.
        Uses official Investing.com daily close prices (NOT resampled from 5-min).

        Primary: PostgreSQL (bi.dim_daily_usdcop)
        Fallback: Parquet backup

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Optional filter by source ('investing', 'twelvedata', etc.)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, source

        Raises:
            ValueError: If no data source available
        """
        df = None

        # Try DB first (bi.dim_daily_usdcop)
        if self.connection_string:
            try:
                df = self._load_daily_from_db(start_date, end_date, source)
                logger.info(f"Loaded {len(df)} daily rows from bi.dim_daily_usdcop")
            except Exception as e:
                logger.warning(f"Daily DB load failed: {e}")
                if not self.fallback_csv:
                    raise

        # Fallback to parquet backup
        if df is None and self.fallback_csv:
            df = self._load_daily_from_file(start_date, end_date)
            if df is not None:
                logger.info(f"Loaded {len(df)} daily rows from file backup")

        # Last resort: resample from 5-min (NOT recommended for production)
        if df is None and self.fallback_csv:
            logger.warning(
                "No official daily data available. "
                "Falling back to 5-min resample (NOT recommended for production)."
            )
            df_5min = self.load_5min(start_date, end_date, filter_market_hours=True)
            df = self.resample_to_daily(df_5min)
            df['source'] = 'resampled_5min'

        if df is None:
            raise ValueError(
                "No daily data source available. "
                "Check bi.dim_daily_usdcop or provide daily parquet."
            )

        return df

    def _load_daily_from_db(
        self,
        start_date: str,
        end_date: str,
        source: str = None
    ) -> pd.DataFrame:
        """Load official daily data from bi.dim_daily_usdcop."""
        import psycopg2

        query = """
            SELECT
                date,
                open,
                high,
                low,
                close,
                COALESCE(volume, 0) as volume,
                source
            FROM bi.dim_daily_usdcop
            WHERE date >= %s AND date <= %s
        """
        params = [start_date, end_date]

        if source:
            query += " AND source = %s"
            params.append(source)

        query += " ORDER BY date ASC"

        with psycopg2.connect(self.connection_string) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_daily_from_file(
        self,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load daily data from parquet backup."""
        # Check for daily-specific backup file
        project_root = Path(__file__).parent.parent.parent
        daily_paths = [
            project_root / "seeds/latest/usdcop_daily_ohlcv.parquet",
            project_root / "data/forecasting/daily_ohlcv.parquet",
        ]

        for path in daily_paths:
            if path.exists():
                df = pd.read_parquet(path)

                # Normalize column names
                df.columns = df.columns.str.lower()
                if 'fecha' in df.columns:
                    df = df.rename(columns={'fecha': 'date'})

                df['date'] = pd.to_datetime(df['date'])

                # Filter date range
                start = pd.Timestamp(start_date)
                end = pd.Timestamp(end_date)
                df = df[(df['date'] >= start) & (df['date'] <= end)]

                if 'source' not in df.columns:
                    df['source'] = 'parquet_backup'

                return df

        return None

    def _find_backup_file(self) -> Optional[Path]:
        """Find available backup file."""
        # Check user-specified path first
        if self.csv_path and self.csv_path.exists():
            return self.csv_path

        # Check default paths
        project_root = Path(__file__).parent.parent.parent

        candidates = [
            project_root / self.DEFAULT_CSV_PATH,
            project_root / self.DEFAULT_CSV_GZ_PATH,
        ]

        for path in candidates:
            if path.exists():
                return path

        return None

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase standard."""
        column_mapping = {
            'Time': 'time',
            'TIME': 'time',
            'datetime': 'time',
            'Datetime': 'time',
            'Open': 'open',
            'OPEN': 'open',
            'High': 'high',
            'HIGH': 'high',
            'Low': 'low',
            'LOW': 'low',
            'Close': 'close',
            'CLOSE': 'close',
            'Volume': 'volume',
            'VOLUME': 'volume',
        }

        df = df.rename(columns=column_mapping)

        # Ensure volume exists
        if 'volume' not in df.columns:
            df['volume'] = 0

        return df

    def resample_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample 5-minute OHLCV to daily.

        WARNING: This method is for FALLBACK/VALIDATION only.
        For production forecasting, use load_daily() which returns
        OFFICIAL Investing.com daily close prices.

        Aggregation rules:
        - open: first of day
        - high: max of day
        - low: min of day
        - close: last of day
        - volume: sum of day

        Args:
            df: DataFrame with 5-min OHLCV data

        Returns:
            DataFrame with daily OHLCV data

        Note:
            Prefer load_daily() for Forecasting pipeline.
            This method may not match official daily close prices.
        """
        import warnings
        warnings.warn(
            "resample_to_daily() is deprecated for production use. "
            "Use load_daily() for official Investing.com values.",
            DeprecationWarning,
            stacklevel=2
        )
        df = df.copy()

        # Ensure time is datetime
        df['time'] = pd.to_datetime(df['time'])

        # Extract date
        df['date'] = df['time'].dt.date

        # Resample
        daily = df.groupby('date').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).reset_index()

        # Convert date back to datetime
        daily['date'] = pd.to_datetime(daily['date'])

        return daily

    def get_latest_timestamp(self) -> Optional[pd.Timestamp]:
        """
        Get the latest timestamp in the database.

        Returns:
            Latest timestamp or None if no data
        """
        if not self.connection_string:
            return None

        try:
            import psycopg2

            query = "SELECT MAX(time) as latest FROM usdcop_m5_ohlcv"

            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(query)
                    result = cur.fetchone()
                    return pd.Timestamp(result[0]) if result[0] else None
        except Exception as e:
            logger.warning(f"Failed to get latest timestamp: {e}")
            return None

    def validate_data_quality(self, df: pd.DataFrame) -> dict:
        """
        Validate OHLCV data quality.

        Checks:
        - No NaN in required columns
        - OHLC relationships (high >= low, etc.)
        - Price range (3000 < close < 6000 for COP)
        - No duplicate timestamps

        Args:
            df: DataFrame to validate

        Returns:
            Dict with validation results
        """
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check NaN
        for col in OHLCV_REQUIRED:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    results['errors'].append(f"{col} has {nan_count} NaN values")
                    results['is_valid'] = False

        # Check OHLC relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            high_low_violations = (df['high'] < df['low']).sum()
            if high_low_violations > 0:
                results['errors'].append(
                    f"{high_low_violations} rows with high < low"
                )
                results['is_valid'] = False

        # Check price range
        if 'close' in df.columns:
            out_of_range = ((df['close'] < 3000) | (df['close'] > 6000)).sum()
            if out_of_range > 0:
                results['warnings'].append(
                    f"{out_of_range} rows with close outside 3000-6000 range"
                )

        # Check duplicates
        if 'time' in df.columns:
            duplicates = df['time'].duplicated().sum()
            if duplicates > 0:
                results['warnings'].append(f"{duplicates} duplicate timestamps")

        # Stats
        results['stats'] = {
            'rows': len(df),
            'date_range': (
                df['time'].min().isoformat() if 'time' in df.columns else None,
                df['time'].max().isoformat() if 'time' in df.columns else None
            ),
            'price_range': (
                float(df['close'].min()) if 'close' in df.columns else None,
                float(df['close'].max()) if 'close' in df.columns else None
            )
        }

        return results
