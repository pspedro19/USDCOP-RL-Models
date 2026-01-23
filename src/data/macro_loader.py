"""
Unified Macro Loader - SSOT for macroeconomic indicators.

Contract: CTR-DATA-005
Version: 2.0.0

This loader provides:
- Loading daily macro from PostgreSQL (v_macro_unified or macro_indicators_daily)
- Fallback to Parquet backup when DB is unavailable
- Resampling daily macro to 5-min with forward-fill for RL pipeline
- Column mapping from DB names to friendly names

Usage:
    from src.data import UnifiedMacroLoader

    loader = UnifiedMacroLoader()

    # For Forecasting (daily - no resample needed)
    df_daily = loader.load_daily("2024-01-01", "2024-12-31")

    # For RL (5-min with ffill)
    df_5min = loader.load_5min("2024-01-01", "2024-12-31")
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union, List, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from .calendar import TradingCalendar
from .contracts import (
    MACRO_DB_TO_FRIENDLY,
    RL_MACRO_COLUMNS,
    FORECASTING_MACRO_COLUMNS,
)
from .safe_merge import FFILL_LIMIT_5MIN, safe_ffill

logger = logging.getLogger(__name__)


class UnifiedMacroLoader:
    """
    Unified macro data loader for RL and Forecasting pipelines.

    Loads from PostgreSQL as primary source (v_macro_unified view or
    macro_indicators_daily table), with Parquet fallback.

    For RL:
        - Resamples daily macro to 5-min grid
        - Applies forward-fill with limit (12 hours max)
        - Uses merge_asof to avoid lookahead bias

    For Forecasting:
        - Returns daily macro as-is
        - No resampling needed

    Attributes:
        connection_string: PostgreSQL connection string
        fallback_parquet: If True, fall back to Parquet when DB unavailable
        parquet_path: Path to Parquet backup file

    Example:
        >>> loader = UnifiedMacroLoader()
        >>> df_daily = loader.load_daily("2024-01-01", "2024-06-30")
        >>> print(f"Loaded {len(df_daily)} days of macro data")
        Loaded 125 days of macro data

        >>> df_5min = loader.load_5min("2024-01-01", "2024-06-30")
        >>> print(f"Loaded {len(df_5min)} 5-min bars with macro")
        Loaded 15000 5-min bars with macro
    """

    # Default paths
    DEFAULT_PARQUET_PATH = Path("seeds/latest/macro_indicators_daily.parquet")

    # View name with friendly column names
    VIEW_NAME = "v_macro_unified"

    # Table name (fallback if view doesn't exist)
    TABLE_NAME = "macro_indicators_daily"

    def __init__(
        self,
        connection_string: Optional[str] = None,
        fallback_parquet: bool = True,
        parquet_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize macro loader.

        Args:
            connection_string: PostgreSQL connection string.
                If None, uses DATABASE_URL environment variable.
            fallback_parquet: If True, fall back to Parquet when DB unavailable.
            parquet_path: Path to Parquet backup file.
        """
        self.connection_string = connection_string or os.getenv("DATABASE_URL")
        self.fallback_parquet = fallback_parquet
        self.parquet_path = Path(parquet_path) if parquet_path else None
        self.calendar = TradingCalendar()

    def load_daily(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        use_friendly_names: bool = True
    ) -> pd.DataFrame:
        """
        Load daily macro data.

        Primary: PostgreSQL (v_macro_unified or macro_indicators_daily)
        Fallback: Parquet backup

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Specific columns to load (friendly names).
                If None, loads all columns.
            use_friendly_names: If True, use friendly column names (dxy, vix, etc.)

        Returns:
            DataFrame with daily macro indicators

        Raises:
            ValueError: If no data source available
        """
        df = None

        # Try DB first
        if self.connection_string:
            try:
                df = self._load_from_db(start_date, end_date, use_friendly_names)
                logger.info(f"Loaded {len(df)} rows from PostgreSQL")
            except Exception as e:
                logger.warning(f"DB load failed: {e}")
                if not self.fallback_parquet:
                    raise

        # Fallback to Parquet
        if df is None and self.fallback_parquet:
            df = self._load_from_parquet(start_date, end_date, use_friendly_names)
            logger.info(f"Loaded {len(df)} rows from Parquet backup")

        if df is None:
            raise ValueError(
                "No data source available. Set DATABASE_URL or provide Parquet path."
            )

        # Filter columns if specified
        if columns is not None:
            # Keep date column + requested columns
            date_col = 'date' if 'date' in df.columns else 'fecha'
            cols_to_keep = [date_col] + [c for c in columns if c in df.columns]
            df = df[cols_to_keep]

        return df

    def load_5min(
        self,
        start_date: str,
        end_date: str,
        columns: Optional[List[str]] = None,
        ffill_limit: int = FFILL_LIMIT_5MIN
    ) -> pd.DataFrame:
        """
        Load macro data resampled to 5-minute frequency.

        Process:
        1. Load daily macro data
        2. Generate 5-min grid for trading hours
        3. Merge with forward-fill (limited to avoid stale data)

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            columns: Specific columns to load (friendly names).
                If None, loads RL default columns.
            ffill_limit: Maximum bars for forward-fill (default: 144 = 12 hours)

        Returns:
            DataFrame with 5-min macro indicators (forward-filled)
        """
        # Default to RL columns if not specified
        if columns is None:
            columns = list(RL_MACRO_COLUMNS)

        # Load daily macro
        df_daily = self.load_daily(
            start_date,
            end_date,
            columns=columns,
            use_friendly_names=True
        )

        # Normalize date column name
        date_col = 'date' if 'date' in df_daily.columns else 'fecha'
        df_daily = df_daily.rename(columns={date_col: 'date'})
        df_daily['date'] = pd.to_datetime(df_daily['date'])

        # Generate 5-min grid
        grid = self.calendar.generate_5min_grid(start_date, end_date)
        df_5min = pd.DataFrame({'time': grid})

        # Add date column for merge
        df_5min['date'] = df_5min['time'].dt.date

        # Merge with daily macro (assign start-of-day value to all bars)
        df_daily['date_for_merge'] = df_daily['date'].dt.date
        df_5min = df_5min.merge(
            df_daily.drop(columns=['date']),
            left_on='date',
            right_on='date_for_merge',
            how='left'
        )
        df_5min = df_5min.drop(columns=['date_for_merge'], errors='ignore')

        # Forward-fill with limit
        macro_cols = [c for c in columns if c in df_5min.columns]
        df_5min = safe_ffill(df_5min, columns=macro_cols, limit=ffill_limit)

        # Calculate derived columns
        df_5min = self._add_derived_columns(df_5min)

        return df_5min

    def _load_from_db(
        self,
        start_date: str,
        end_date: str,
        use_friendly_names: bool = True
    ) -> pd.DataFrame:
        """Load from PostgreSQL database."""
        import psycopg2

        # Try view first (has friendly names)
        if use_friendly_names:
            try:
                query = f"""
                    SELECT * FROM {self.VIEW_NAME}
                    WHERE fecha >= %s AND fecha <= %s
                    ORDER BY fecha ASC
                """
                with psycopg2.connect(self.connection_string) as conn:
                    df = pd.read_sql_query(query, conn, params=(start_date, end_date))

                # Rename fecha to date for consistency
                if 'fecha' in df.columns:
                    df = df.rename(columns={'fecha': 'date'})

                return df
            except Exception as e:
                logger.warning(f"View {self.VIEW_NAME} not found, using table: {e}")

        # Fallback to table with manual column mapping
        query = f"""
            SELECT * FROM {self.TABLE_NAME}
            WHERE fecha >= %s AND fecha <= %s
            ORDER BY fecha ASC
        """

        with psycopg2.connect(self.connection_string) as conn:
            df = pd.read_sql_query(query, conn, params=(start_date, end_date))

        # Apply column mapping if requested
        if use_friendly_names:
            df = self._apply_friendly_names(df)

        return df

    def _load_from_parquet(
        self,
        start_date: str,
        end_date: str,
        use_friendly_names: bool = True
    ) -> pd.DataFrame:
        """Load from Parquet backup file."""
        path = self._find_parquet_file()

        if path is None:
            raise FileNotFoundError("No Parquet backup file found")

        df = pd.read_parquet(path)

        # Normalize date column
        date_col = None
        for col in ['fecha', 'date', 'Date', 'FECHA']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            raise ValueError("No date column found in Parquet file")

        df[date_col] = pd.to_datetime(df[date_col])

        # Filter to date range
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[(df[date_col] >= start) & (df[date_col] <= end)]

        # Apply column mapping if requested
        if use_friendly_names:
            df = self._apply_friendly_names(df)

        # Rename to 'date' for consistency
        if date_col != 'date':
            df = df.rename(columns={date_col: 'date'})

        return df

    def _find_parquet_file(self) -> Optional[Path]:
        """Find available Parquet backup file."""
        # Check user-specified path first
        if self.parquet_path and self.parquet_path.exists():
            return self.parquet_path

        # Check default path
        project_root = Path(__file__).parent.parent.parent
        default_path = project_root / self.DEFAULT_PARQUET_PATH

        if default_path.exists():
            return default_path

        return None

    def _apply_friendly_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply friendly column names using mapping."""
        # Only rename columns that exist
        rename_map = {
            k: v for k, v in MACRO_DB_TO_FRIENDLY.items()
            if k in df.columns
        }

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated/derived columns for RL."""
        df = df.copy()

        # Rate spread (Colombia 10Y - US 10Y)
        if 'col10y' in df.columns and 'ust10y' in df.columns:
            df['rate_spread'] = df['col10y'] - df['ust10y']

        # DXY change (1-day percentage change)
        if 'dxy' in df.columns:
            df['dxy_change_1d'] = df['dxy'].pct_change(1)

        # VIX regime (1=low, 2=normal, 3=high)
        if 'vix' in df.columns:
            df['vix_regime'] = pd.cut(
                df['vix'],
                bins=[0, 20, 30, 100],
                labels=[1, 2, 3]
            ).astype(float)

        return df

    def load_for_forecasting(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Load macro data optimized for Forecasting pipeline.

        Loads only DXY and WTI with 1-day lag applied.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with date, dxy_close_lag1, oil_close_lag1
        """
        df = self.load_daily(
            start_date,
            end_date,
            columns=list(FORECASTING_MACRO_COLUMNS),
            use_friendly_names=True
        )

        # Apply 1-day lag (avoid lookahead)
        df['dxy_close_lag1'] = df['dxy'].shift(1)
        df['oil_close_lag1'] = df['wti'].shift(1)

        # Keep only lagged columns
        date_col = 'date' if 'date' in df.columns else 'fecha'
        return df[[date_col, 'dxy_close_lag1', 'oil_close_lag1']]

    def get_available_columns(self) -> List[str]:
        """
        Get list of available macro columns.

        Returns:
            List of column names (friendly names)
        """
        return list(MACRO_DB_TO_FRIENDLY.values())

    def validate_coverage(
        self,
        start_date: str,
        end_date: str
    ) -> Tuple[float, List[str]]:
        """
        Validate macro data coverage for date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Tuple of (coverage_ratio, missing_dates)
        """
        try:
            df = self.load_daily(start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to load data for coverage check: {e}")
            return 0.0, []

        # Get expected trading days
        expected_days = self.calendar.get_trading_days(start_date, end_date)

        # Get actual days in data
        date_col = 'date' if 'date' in df.columns else 'fecha'
        actual_days = set(pd.to_datetime(df[date_col]).dt.date)

        # Find missing days
        expected_set = set(d.date() for d in expected_days)
        missing = sorted(expected_set - actual_days)

        coverage = len(actual_days) / len(expected_set) if expected_set else 0.0

        return coverage, [str(d) for d in missing]
