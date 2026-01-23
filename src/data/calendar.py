"""
Trading Calendar - COT Market Hours and Holidays.

Contract: CTR-DATA-003
Version: 2.0.0

This module handles:
- Colombian market hours (8:00am - 1:00pm COT)
- US market holidays
- Colombian market holidays
- Grid generation for 5-min resampling
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import List, Optional, Tuple
from functools import lru_cache


# =============================================================================
# MARKET HOURS (COT = Colombia Time = UTC-5)
# =============================================================================

# Colombian market hours
MARKET_OPEN_COT = time(8, 0)    # 8:00 AM COT = 13:00 UTC
MARKET_CLOSE_COT = time(13, 0)  # 1:00 PM COT = 18:00 UTC

# In UTC (for 5-min data processing)
MARKET_OPEN_UTC = time(13, 0)   # 13:00 UTC
MARKET_CLOSE_UTC = time(18, 0)  # 18:00 UTC

# Last bar of the day (5-min candle that ends at close)
LAST_BAR_UTC = time(17, 55)     # 17:55 UTC (candle closes at 18:00)


# =============================================================================
# US HOLIDAYS (NYSE Closed)
# =============================================================================

US_HOLIDAYS_2024 = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # MLK Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
]

US_HOLIDAYS_2025 = [
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
]

US_HOLIDAYS_2026 = [
    "2026-01-01",  # New Year's Day
    "2026-01-19",  # MLK Day
    "2026-02-16",  # Presidents Day
    "2026-04-03",  # Good Friday
    "2026-05-25",  # Memorial Day
    "2026-06-19",  # Juneteenth
    "2026-07-03",  # Independence Day (observed)
    "2026-09-07",  # Labor Day
    "2026-11-26",  # Thanksgiving
    "2026-12-25",  # Christmas
]


# =============================================================================
# COLOMBIAN HOLIDAYS
# =============================================================================

COL_HOLIDAYS_2024 = [
    "2024-01-01",  # Ano Nuevo
    "2024-01-08",  # Dia de los Reyes Magos
    "2024-03-25",  # San Jose
    "2024-03-28",  # Jueves Santo
    "2024-03-29",  # Viernes Santo
    "2024-05-01",  # Dia del Trabajo
    "2024-05-13",  # Ascension del Senor
    "2024-06-03",  # Corpus Christi
    "2024-06-10",  # Sagrado Corazon
    "2024-07-01",  # San Pedro y San Pablo
    "2024-07-20",  # Dia de la Independencia
    "2024-08-07",  # Batalla de Boyaca
    "2024-08-19",  # Asuncion de la Virgen
    "2024-10-14",  # Dia de la Raza
    "2024-11-04",  # Todos los Santos
    "2024-11-11",  # Independencia de Cartagena
    "2024-12-08",  # Inmaculada Concepcion
    "2024-12-25",  # Navidad
]

COL_HOLIDAYS_2025 = [
    "2025-01-01",  # Ano Nuevo
    "2025-01-06",  # Dia de los Reyes Magos
    "2025-03-24",  # San Jose
    "2025-04-17",  # Jueves Santo
    "2025-04-18",  # Viernes Santo
    "2025-05-01",  # Dia del Trabajo
    "2025-06-02",  # Ascension del Senor
    "2025-06-23",  # Corpus Christi
    "2025-06-30",  # Sagrado Corazon
    "2025-06-30",  # San Pedro y San Pablo (mismo dia)
    "2025-07-20",  # Dia de la Independencia
    "2025-08-07",  # Batalla de Boyaca
    "2025-08-18",  # Asuncion de la Virgen
    "2025-10-13",  # Dia de la Raza
    "2025-11-03",  # Todos los Santos
    "2025-11-17",  # Independencia de Cartagena
    "2025-12-08",  # Inmaculada Concepcion
    "2025-12-25",  # Navidad
]

COL_HOLIDAYS_2026 = [
    "2026-01-01",  # Ano Nuevo
    "2026-01-12",  # Dia de los Reyes Magos
    "2026-03-23",  # San Jose
    "2026-04-02",  # Jueves Santo
    "2026-04-03",  # Viernes Santo
    "2026-05-01",  # Dia del Trabajo
    "2026-05-18",  # Ascension del Senor
    "2026-06-08",  # Corpus Christi
    "2026-06-15",  # Sagrado Corazon
    "2026-06-29",  # San Pedro y San Pablo
    "2026-07-20",  # Dia de la Independencia
    "2026-08-07",  # Batalla de Boyaca
    "2026-08-17",  # Asuncion de la Virgen
    "2026-10-12",  # Dia de la Raza
    "2026-11-02",  # Todos los Santos
    "2026-11-16",  # Independencia de Cartagena
    "2026-12-08",  # Inmaculada Concepcion
    "2026-12-25",  # Navidad
]


class TradingCalendar:
    """
    Trading calendar for USD/COP market.

    Handles:
    - Market hours (8:00-13:00 COT / 13:00-18:00 UTC)
    - US and Colombian holidays
    - 5-min grid generation for resampling

    Usage:
        calendar = TradingCalendar()

        # Check if date is trading day
        if calendar.is_trading_day(date):
            ...

        # Generate 5-min grid
        grid = calendar.generate_5min_grid("2024-01-02", "2024-01-05")

        # Filter DataFrame to market hours
        df = calendar.filter_market_hours(df)
    """

    def __init__(self):
        """Initialize calendar with holidays."""
        self._us_holidays = self._parse_holidays(
            US_HOLIDAYS_2024 + US_HOLIDAYS_2025 + US_HOLIDAYS_2026
        )
        self._col_holidays = self._parse_holidays(
            COL_HOLIDAYS_2024 + COL_HOLIDAYS_2025 + COL_HOLIDAYS_2026
        )
        self._all_holidays = self._us_holidays.union(self._col_holidays)

    def _parse_holidays(self, holidays: List[str]) -> set:
        """Parse list of holiday strings to set of dates."""
        return {pd.Timestamp(h).date() for h in holidays}

    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if date is a trading day.

        A trading day is:
        - Monday through Friday
        - Not a US or Colombian holiday

        Args:
            date: Date to check

        Returns:
            True if trading day, False otherwise
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)

        d = date.date() if hasattr(date, 'date') else date

        # Weekday check (0=Mon, 6=Sun)
        if d.weekday() >= 5:
            return False

        # Holiday check
        if d in self._all_holidays:
            return False

        return True

    def is_market_hours(self, timestamp: datetime) -> bool:
        """
        Check if timestamp is within market hours.

        Market hours: 13:00-18:00 UTC (8:00-13:00 COT)

        Args:
            timestamp: Timestamp to check (UTC)

        Returns:
            True if within market hours, False otherwise
        """
        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)

        # Get time component
        t = timestamp.time()

        # Check if within market hours
        return MARKET_OPEN_UTC <= t < MARKET_CLOSE_UTC

    def filter_market_hours(
        self,
        df: pd.DataFrame,
        datetime_col: str = 'time'
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include market hours.

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column

        Returns:
            Filtered DataFrame
        """
        df = df.copy()

        # Ensure datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Extract components
        dow = df[datetime_col].dt.dayofweek
        hour = df[datetime_col].dt.hour
        minute = df[datetime_col].dt.minute

        # Filter: Mon-Fri, 13:00-17:55 UTC
        mask = (
            (dow < 5) &  # Monday through Friday
            (
                ((hour >= 13) & (hour < 17)) |  # 13:00-16:59
                ((hour == 17) & (minute <= 55))  # 17:00-17:55
            )
        )

        return df[mask].reset_index(drop=True)

    def generate_5min_grid(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DatetimeIndex:
        """
        Generate 5-minute timestamp grid for trading hours.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DatetimeIndex with 5-min timestamps for all trading hours

        Example:
            >>> calendar = TradingCalendar()
            >>> grid = calendar.generate_5min_grid("2024-01-02", "2024-01-03")
            >>> len(grid)  # 2 days * 60 bars/day = 120
            120
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        timestamps = []
        for date in dates:
            if not self.is_trading_day(date):
                continue

            # Generate 5-min bars for this day
            # 13:00 to 17:55 UTC = 60 bars
            for hour in range(13, 18):
                max_minute = 55 if hour == 17 else 55
                for minute in range(0, max_minute + 1, 5):
                    ts = pd.Timestamp(
                        year=date.year,
                        month=date.month,
                        day=date.day,
                        hour=hour,
                        minute=minute,
                        tz='UTC'
                    )
                    timestamps.append(ts)

        return pd.DatetimeIndex(timestamps)

    def get_trading_days(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DatetimeIndex:
        """
        Get list of trading days in date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DatetimeIndex with trading days only
        """
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        return pd.DatetimeIndex([d for d in dates if self.is_trading_day(d)])

    def next_trading_day(self, date: datetime) -> pd.Timestamp:
        """
        Get next trading day after given date.

        Args:
            date: Reference date

        Returns:
            Next trading day
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)

        next_day = date + timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)

        return pd.Timestamp(next_day)

    def prev_trading_day(self, date: datetime) -> pd.Timestamp:
        """
        Get previous trading day before given date.

        Args:
            date: Reference date

        Returns:
            Previous trading day
        """
        if isinstance(date, str):
            date = pd.Timestamp(date)

        prev_day = date - timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)

        return pd.Timestamp(prev_day)


# =============================================================================
# MODULE-LEVEL HELPERS
# =============================================================================

@lru_cache(maxsize=1)
def get_calendar() -> TradingCalendar:
    """Get cached TradingCalendar instance."""
    return TradingCalendar()


def is_trading_day(date: datetime) -> bool:
    """Convenience function to check if date is trading day."""
    return get_calendar().is_trading_day(date)


def filter_market_hours(df: pd.DataFrame, datetime_col: str = 'time') -> pd.DataFrame:
    """Convenience function to filter DataFrame to market hours."""
    return get_calendar().filter_market_hours(df, datetime_col)
