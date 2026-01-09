"""
Trading Calendar Validation for USD/COP (DAG Utils Version)
============================================================
Simplified version for Airflow DAGs.

V20 FIX: US holidays now included by default for USD/COP trading.
USD/COP trading is affected by US market holidays (reduced liquidity).

Author: Pedro @ Lean Tech Solutions / Claude Code
Version: 2.0.0
Date: 2026-01-09
"""

from datetime import datetime, date, timedelta
from typing import Union, Optional, Set
import logging

import pytz

logger = logging.getLogger(__name__)

# Colombian market timezone
COT_TZ = pytz.timezone('America/Bogota')
UTC_TZ = pytz.UTC

# Trading days (Monday=0, Friday=4)
WEEKEND_DAYS = {5, 6}  # Sat-Sun

# Default market hours for USD/COP (8:00 AM - 12:55 PM COT)
DEFAULT_MARKET_OPEN_HOUR = 8
DEFAULT_MARKET_CLOSE_HOUR = 12

# V20 FIX: Explicit US Federal Holidays 2025-2027 (fallback if holidays package unavailable)
# These are days when US markets are closed, affecting USD liquidity
US_FEDERAL_HOLIDAYS: Set[date] = {
    # 2025
    date(2025, 1, 1),    # New Year's Day
    date(2025, 1, 20),   # MLK Day
    date(2025, 2, 17),   # Presidents Day
    date(2025, 5, 26),   # Memorial Day
    date(2025, 6, 19),   # Juneteenth
    date(2025, 7, 4),    # Independence Day
    date(2025, 9, 1),    # Labor Day
    date(2025, 10, 13),  # Columbus Day
    date(2025, 11, 11),  # Veterans Day
    date(2025, 11, 27),  # Thanksgiving
    date(2025, 12, 25),  # Christmas
    # 2026
    date(2026, 1, 1),    # New Year's Day
    date(2026, 1, 19),   # MLK Day
    date(2026, 2, 16),   # Presidents Day
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 10, 12),  # Columbus Day
    date(2026, 11, 11),  # Veterans Day
    date(2026, 11, 26),  # Thanksgiving
    date(2026, 12, 25),  # Christmas
    # 2027
    date(2027, 1, 1),    # New Year's Day
    date(2027, 1, 18),   # MLK Day
    date(2027, 2, 15),   # Presidents Day
    date(2027, 5, 31),   # Memorial Day
    date(2027, 6, 18),   # Juneteenth (observed)
    date(2027, 7, 5),    # Independence Day (observed)
    date(2027, 9, 6),    # Labor Day
    date(2027, 10, 11),  # Columbus Day
    date(2027, 11, 11),  # Veterans Day
    date(2027, 11, 25),  # Thanksgiving
    date(2027, 12, 24),  # Christmas (observed)
}


class TradingCalendar:
    """
    Trading calendar for USD/COP validation.

    V20 FIX: US holidays now included by default (affects USD liquidity).
    """

    def __init__(
        self,
        timezone: str = 'America/Bogota',
        include_us_holidays: bool = True,  # V20 FIX: Changed default from False to True
        market_open_hour: int = DEFAULT_MARKET_OPEN_HOUR,
        market_close_hour: int = DEFAULT_MARKET_CLOSE_HOUR
    ):
        self.timezone = pytz.timezone(timezone)
        self.include_us_holidays = include_us_holidays
        self.market_open_hour = market_open_hour
        self.market_close_hour = market_close_hour

        # Try to import colombian holidays
        try:
            from colombian_holidays import is_holiday as col_is_holiday
            self._col_is_holiday = col_is_holiday
        except ImportError:
            logger.warning("colombian-holidays not installed")
            self._col_is_holiday = None

        # Try to import US holidays (use package if available, fallback to static list)
        self.us_holidays = None
        if include_us_holidays:
            try:
                import holidays
                self.us_holidays = holidays.US(years=range(2020, 2030))
                logger.info("US holidays loaded from holidays package")
            except ImportError:
                logger.info("Using static US holiday list (holidays package not installed)")
                self.us_holidays = US_FEDERAL_HOLIDAYS

    def is_weekend(self, dt: Union[datetime, date]) -> bool:
        return dt.weekday() in WEEKEND_DAYS

    def is_colombian_holiday(self, dt: Union[datetime, date]) -> bool:
        if self._col_is_holiday is None:
            return False
        check_date = dt.date() if isinstance(dt, datetime) else dt
        try:
            return self._col_is_holiday(check_date)
        except:
            return False

    def is_us_holiday(self, dt: Union[datetime, date]) -> bool:
        """Check if date is a US federal holiday (affects USD liquidity)."""
        if not self.us_holidays:
            return False
        check_date = dt.date() if isinstance(dt, datetime) else dt
        # Works with both holidays package (dict) and static set
        return check_date in self.us_holidays

    def is_trading_day(self, dt: Union[datetime, date]) -> bool:
        if self.is_weekend(dt):
            return False
        if self.is_colombian_holiday(dt):
            return False
        if self.is_us_holiday(dt):
            return False
        return True

    def get_violation_reason(self, dt: Union[datetime, date]) -> Optional[str]:
        if self.is_weekend(dt):
            return f"Weekend ({dt.strftime('%A')})"
        if self.is_colombian_holiday(dt):
            return "Colombian Holiday"
        if self.is_us_holiday(dt):
            return "US Holiday"
        return None

    def get_next_trading_day(self, dt: Union[datetime, date], skip_current: bool = False) -> date:
        check_date = dt.date() if isinstance(dt, datetime) else dt
        if skip_current:
            check_date += timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)
        raise ValueError(f"Could not find trading day within 30 days of {dt}")

    def get_previous_trading_day(self, dt: Union[datetime, date], skip_current: bool = False) -> date:
        check_date = dt.date() if isinstance(dt, datetime) else dt
        if skip_current:
            check_date -= timedelta(days=1)
        for _ in range(30):
            if self.is_trading_day(check_date):
                return check_date
            check_date -= timedelta(days=1)
        raise ValueError(f"Could not find trading day within 30 days before {dt}")


# Global singleton
_calendar = None


def get_calendar() -> TradingCalendar:
    global _calendar
    if _calendar is None:
        _calendar = TradingCalendar()
    return _calendar


def is_trading_day(dt: Union[datetime, date]) -> bool:
    return get_calendar().is_trading_day(dt)


# For backwards compatibility
trading_calendar = get_calendar()
