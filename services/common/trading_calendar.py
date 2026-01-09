"""
Trading Calendar Validation for USD/COP
========================================
Validates that data only includes valid trading days:
- Excludes Colombian holidays (using colombian-holidays library)
- Excludes weekends (Saturday, Sunday)
- Excludes US holidays (for macro data)

CRITICAL REQUIREMENT:
    The entire project must NOT use weekends or Colombian holidays in:
    - Data loading to database
    - Training data
    - Validation data
    - Test data
    - Production inference

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
Version: 2.0.0
"""

from datetime import datetime, date, timedelta
from typing import List, Tuple, Union, Optional, Dict, Set
import logging

import pandas as pd
import pytz

try:
    from colombian_holidays import is_holiday as col_is_holiday, list_holidays as col_list_holidays
    HAS_COLOMBIAN_HOLIDAYS = True
except ImportError:
    HAS_COLOMBIAN_HOLIDAYS = False
    logging.warning(
        "colombian-holidays library not installed. "
        "Install with: pip install colombian-holidays"
    )

try:
    import holidays
    HAS_US_HOLIDAYS = True
except ImportError:
    HAS_US_HOLIDAYS = False
    logging.warning("holidays library not installed for US holiday detection")


logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Colombian market timezone
COT_TZ = pytz.timezone('America/Bogota')
UTC_TZ = pytz.UTC

# Trading days (Monday=0, Friday=4)
TRADING_WEEKDAYS = {0, 1, 2, 3, 4}  # Mon-Fri
WEEKEND_DAYS = {5, 6}  # Sat-Sun

# Default market hours for USD/COP (8:00 AM - 12:55 PM COT)
DEFAULT_MARKET_OPEN_HOUR = 8
DEFAULT_MARKET_OPEN_MINUTE = 0
DEFAULT_MARKET_CLOSE_HOUR = 12
DEFAULT_MARKET_CLOSE_MINUTE = 55


# =============================================================================
# MAIN TRADING CALENDAR CLASS
# =============================================================================

class TradingCalendar:
    """
    Validates trading days for USD/COP market.

    Features:
        - Weekend detection (Saturday/Sunday)
        - Colombian holiday detection (using colombian-holidays library)
        - US holiday detection (optional, for macro data)
        - Timezone-aware validation (Colombia = UTC-5)
        - DataFrame filtering and validation
        - Holiday range queries

    Usage:
        >>> cal = TradingCalendar()
        >>> cal.is_trading_day(datetime(2025, 1, 1))  # New Year
        False
        >>> cal.is_trading_day(datetime(2025, 1, 6))  # Reyes Magos
        False
        >>> cal.is_trading_day(datetime(2025, 1, 7))  # Tuesday
        True

        # Filter DataFrame to only trading days
        >>> df_clean = cal.filter_trading_days(df, date_col='timestamp')

        # Validate no holidays/weekends exist
        >>> is_valid, violations = cal.validate_no_holidays(df, date_col='timestamp')
        >>> if not is_valid:
        ...     print(f"Found {len(violations)} invalid dates!")
    """

    def __init__(
        self,
        timezone: str = 'America/Bogota',
        include_us_holidays: bool = False,
        market_open_hour: int = DEFAULT_MARKET_OPEN_HOUR,
        market_close_hour: int = DEFAULT_MARKET_CLOSE_HOUR
    ):
        """
        Initialize trading calendar.

        Args:
            timezone: Market timezone (default: 'America/Bogota' = UTC-5)
            include_us_holidays: If True, also check US holidays (for macro data)
            market_open_hour: Market opening hour in local time (default: 8)
            market_close_hour: Market closing hour in local time (default: 12)
        """
        self.timezone = pytz.timezone(timezone)
        self.include_us_holidays = include_us_holidays
        self.market_open_hour = market_open_hour
        self.market_close_hour = market_close_hour

        # Initialize US holidays if requested
        self.us_holidays = None
        if include_us_holidays and HAS_US_HOLIDAYS:
            # US holidays that affect macro data (NYSE calendar)
            self.us_holidays = holidays.US(years=range(2015, 2030))

        # Cache for Colombian holidays (year -> set of dates)
        self._col_holiday_cache: Dict[int, Set[date]] = {}

        if not HAS_COLOMBIAN_HOLIDAYS:
            logger.warning(
                "Colombian holidays library not available. "
                "Holiday validation will be incomplete!"
            )

    def _get_col_holidays_for_year(self, year: int) -> Set[date]:
        """
        Get Colombian holidays for a specific year (cached).

        Args:
            year: Year to get holidays for

        Returns:
            Set of date objects representing Colombian holidays
        """
        if year not in self._col_holiday_cache:
            if HAS_COLOMBIAN_HOLIDAYS:
                try:
                    # Get all holidays for the year
                    holiday_list = col_list_holidays(year)
                    self._col_holiday_cache[year] = {
                        h[0] if isinstance(h, tuple) else h
                        for h in holiday_list
                    }
                    logger.debug(f"Loaded {len(self._col_holiday_cache[year])} Colombian holidays for {year}")
                except Exception as e:
                    logger.error(f"Error loading Colombian holidays for {year}: {e}")
                    self._col_holiday_cache[year] = set()
            else:
                self._col_holiday_cache[year] = set()

        return self._col_holiday_cache[year]

    def is_weekend(self, dt: Union[datetime, date]) -> bool:
        """
        Check if date is a weekend (Saturday or Sunday).

        Args:
            dt: Date or datetime to check

        Returns:
            True if weekend, False otherwise
        """
        weekday = dt.weekday()
        return weekday in WEEKEND_DAYS

    def is_colombian_holiday(self, dt: Union[datetime, date]) -> bool:
        """
        Check if date is a Colombian holiday.

        Args:
            dt: Date or datetime to check

        Returns:
            True if Colombian holiday, False otherwise
        """
        if not HAS_COLOMBIAN_HOLIDAYS:
            return False

        # Convert to date if datetime
        check_date = dt.date() if isinstance(dt, datetime) else dt

        # Check using library
        try:
            return col_is_holiday(check_date)
        except Exception as e:
            logger.error(f"Error checking Colombian holiday for {check_date}: {e}")
            # Fallback to cache
            holidays_set = self._get_col_holidays_for_year(check_date.year)
            return check_date in holidays_set

    def is_us_holiday(self, dt: Union[datetime, date]) -> bool:
        """
        Check if date is a US holiday (for macro data validation).

        Args:
            dt: Date or datetime to check

        Returns:
            True if US holiday, False otherwise
        """
        if not self.include_us_holidays or not HAS_US_HOLIDAYS:
            return False

        check_date = dt.date() if isinstance(dt, datetime) else dt
        return check_date in self.us_holidays

    def is_trading_day(self, dt: Union[datetime, date]) -> bool:
        """
        Check if date is a valid trading day.

        A valid trading day is:
        - Not a weekend (Sat/Sun)
        - Not a Colombian holiday
        - Not a US holiday (if include_us_holidays=True)

        Args:
            dt: Date or datetime to check

        Returns:
            True if valid trading day, False otherwise
        """
        # Weekend check
        if self.is_weekend(dt):
            return False

        # Colombian holiday check
        if self.is_colombian_holiday(dt):
            return False

        # US holiday check (optional)
        if self.is_us_holiday(dt):
            return False

        return True

    def get_violation_reason(self, dt: Union[datetime, date]) -> Optional[str]:
        """
        Get reason why date is not a trading day.

        Args:
            dt: Date or datetime to check

        Returns:
            String describing violation, or None if valid trading day
        """
        if self.is_weekend(dt):
            day_name = dt.strftime('%A')
            return f"Weekend ({day_name})"

        if self.is_colombian_holiday(dt):
            return "Colombian Holiday"

        if self.is_us_holiday(dt):
            return "US Holiday"

        return None

    def filter_trading_days(
        self,
        df: pd.DataFrame,
        date_col: str = 'timestamp',
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Filter DataFrame to only include valid trading days.

        Removes:
        - Weekends (Saturday, Sunday)
        - Colombian holidays
        - US holidays (if include_us_holidays=True)

        Args:
            df: DataFrame to filter
            date_col: Name of date/timestamp column
            inplace: If True, modify DataFrame in place

        Returns:
            Filtered DataFrame (or original if inplace=True)

        Example:
            >>> cal = TradingCalendar()
            >>> df_clean = cal.filter_trading_days(df, date_col='timestamp')
            >>> print(f"Removed {len(df) - len(df_clean)} non-trading days")
        """
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in DataFrame")

        # Create mask for valid trading days
        mask = df[date_col].apply(self.is_trading_day)

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.info(f"Filtering out {n_removed:,} non-trading days ({n_removed/len(df)*100:.2f}%)")

        if inplace:
            df.drop(df.index[~mask], inplace=True)
            return df
        else:
            return df[mask].copy()

    def validate_no_holidays(
        self,
        df: pd.DataFrame,
        date_col: str = 'timestamp'
    ) -> Tuple[bool, List[Dict[str, Union[datetime, str]]]]:
        """
        Validate that DataFrame contains no holidays or weekends.

        Args:
            df: DataFrame to validate
            date_col: Name of date/timestamp column

        Returns:
            Tuple of (is_valid, violations)
            - is_valid: True if no violations found
            - violations: List of dicts with 'date' and 'reason' for each violation

        Example:
            >>> cal = TradingCalendar()
            >>> is_valid, violations = cal.validate_no_holidays(df)
            >>> if not is_valid:
            ...     print("VALIDATION FAILED!")
            ...     for v in violations[:5]:
            ...         print(f"  {v['date']}: {v['reason']}")
        """
        if date_col not in df.columns:
            raise ValueError(f"Column '{date_col}' not found in DataFrame")

        violations = []

        # Check each unique date
        unique_dates = df[date_col].unique()
        for dt in unique_dates:
            reason = self.get_violation_reason(dt)
            if reason:
                violations.append({
                    'date': dt,
                    'reason': reason
                })

        is_valid = len(violations) == 0

        if not is_valid:
            logger.warning(
                f"Validation FAILED: Found {len(violations)} dates that are not trading days"
            )
        else:
            logger.info(
                f"Validation PASSED: All {len(unique_dates)} unique dates are valid trading days"
            )

        return is_valid, violations

    def get_holidays_in_range(
        self,
        start: date,
        end: date,
        include_weekends: bool = True
    ) -> List[Tuple[date, str]]:
        """
        Get all holidays and weekends in date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)
            include_weekends: If True, include Saturday/Sunday in results

        Returns:
            List of (date, reason) tuples

        Example:
            >>> cal = TradingCalendar()
            >>> holidays = cal.get_holidays_in_range(
            ...     date(2025, 1, 1),
            ...     date(2025, 1, 31)
            ... )
            >>> for dt, reason in holidays:
            ...     print(f"{dt}: {reason}")
        """
        result = []
        current = start

        while current <= end:
            reason = self.get_violation_reason(current)
            if reason:
                # Optionally filter out weekends
                if include_weekends or not self.is_weekend(current):
                    result.append((current, reason))

            current += timedelta(days=1)

        return result

    def get_trading_days_in_range(
        self,
        start: date,
        end: date
    ) -> List[date]:
        """
        Get all valid trading days in date range.

        Args:
            start: Start date (inclusive)
            end: End date (inclusive)

        Returns:
            List of date objects representing valid trading days

        Example:
            >>> cal = TradingCalendar()
            >>> trading_days = cal.get_trading_days_in_range(
            ...     date(2025, 1, 1),
            ...     date(2025, 1, 31)
            ... )
            >>> print(f"Found {len(trading_days)} trading days in January 2025")
        """
        result = []
        current = start

        while current <= end:
            if self.is_trading_day(current):
                result.append(current)
            current += timedelta(days=1)

        return result

    def get_next_trading_day(
        self,
        dt: Union[datetime, date],
        skip_current: bool = False
    ) -> date:
        """
        Get next valid trading day after (or including) given date.

        Args:
            dt: Reference date
            skip_current: If True, skip current date even if it's a trading day

        Returns:
            Next valid trading day

        Example:
            >>> cal = TradingCalendar()
            >>> next_day = cal.get_next_trading_day(date(2025, 1, 1))  # New Year
            >>> print(next_day)  # First trading day of 2025
        """
        check_date = dt.date() if isinstance(dt, datetime) else dt

        if skip_current:
            check_date += timedelta(days=1)

        # Safety limit: search max 30 days ahead
        for _ in range(30):
            if self.is_trading_day(check_date):
                return check_date
            check_date += timedelta(days=1)

        raise ValueError(f"Could not find trading day within 30 days of {dt}")

    def get_previous_trading_day(
        self,
        dt: Union[datetime, date],
        skip_current: bool = False
    ) -> date:
        """
        Get previous valid trading day before (or including) given date.

        Args:
            dt: Reference date
            skip_current: If True, skip current date even if it's a trading day

        Returns:
            Previous valid trading day
        """
        check_date = dt.date() if isinstance(dt, datetime) else dt

        if skip_current:
            check_date -= timedelta(days=1)

        # Safety limit: search max 30 days back
        for _ in range(30):
            if self.is_trading_day(check_date):
                return check_date
            check_date -= timedelta(days=1)

        raise ValueError(f"Could not find trading day within 30 days before {dt}")

    def generate_report(
        self,
        df: pd.DataFrame,
        date_col: str = 'timestamp'
    ) -> str:
        """
        Generate validation report for DataFrame.

        Args:
            df: DataFrame to analyze
            date_col: Name of date/timestamp column

        Returns:
            Multi-line string report

        Example:
            >>> cal = TradingCalendar()
            >>> report = cal.generate_report(df)
            >>> print(report)
        """
        is_valid, violations = self.validate_no_holidays(df, date_col)

        lines = []
        lines.append("="*70)
        lines.append("TRADING CALENDAR VALIDATION REPORT")
        lines.append("="*70)
        lines.append(f"DataFrame Shape: {df.shape}")
        lines.append(f"Date Column: {date_col}")

        if date_col in df.columns:
            unique_dates = df[date_col].unique()
            lines.append(f"Unique Dates: {len(unique_dates)}")
            lines.append(f"Date Range: {df[date_col].min()} to {df[date_col].max()}")

        lines.append("")
        lines.append(f"Status: {'PASS' if is_valid else 'FAIL'}")
        lines.append(f"Violations: {len(violations)}")

        if violations:
            lines.append("")
            lines.append("First 10 violations:")
            for v in violations[:10]:
                dt = v['date']
                reason = v['reason']
                lines.append(f"  - {dt}: {reason}")

            if len(violations) > 10:
                lines.append(f"  ... and {len(violations) - 10} more")

        lines.append("="*70)

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global instance for convenience
_default_calendar = None


def get_calendar(
    timezone: str = 'America/Bogota',
    include_us_holidays: bool = False
) -> TradingCalendar:
    """
    Get singleton trading calendar instance.

    Args:
        timezone: Market timezone
        include_us_holidays: Include US holidays

    Returns:
        TradingCalendar instance
    """
    global _default_calendar
    if _default_calendar is None:
        _default_calendar = TradingCalendar(
            timezone=timezone,
            include_us_holidays=include_us_holidays
        )
    return _default_calendar


def is_trading_day(dt: Union[datetime, date]) -> bool:
    """Convenience: Check if date is a trading day."""
    return get_calendar().is_trading_day(dt)


def filter_trading_days(df: pd.DataFrame, date_col: str = 'timestamp') -> pd.DataFrame:
    """Convenience: Filter DataFrame to trading days only."""
    return get_calendar().filter_trading_days(df, date_col)


def validate_no_holidays(df: pd.DataFrame, date_col: str = 'timestamp') -> Tuple[bool, List[Dict]]:
    """Convenience: Validate DataFrame contains no holidays/weekends."""
    return get_calendar().validate_no_holidays(df, date_col)


def validate_and_filter(
    df: pd.DataFrame,
    date_col: str = 'timestamp',
    raise_on_violations: bool = False
) -> pd.DataFrame:
    """
    Validate and filter DataFrame in one step.

    Args:
        df: DataFrame to process
        date_col: Date column name
        raise_on_violations: If True, raise exception on violations

    Returns:
        Filtered DataFrame

    Raises:
        ValueError: If violations found and raise_on_violations=True
    """
    cal = get_calendar()
    is_valid, violations = cal.validate_no_holidays(df, date_col)

    if not is_valid and raise_on_violations:
        raise ValueError(
            f"Found {len(violations)} non-trading days in DataFrame. "
            f"First violation: {violations[0]['date']} ({violations[0]['reason']})"
        )

    return cal.filter_trading_days(df, date_col)


# =============================================================================
# AIRFLOW DAG INTEGRATION
# =============================================================================

def validate_dag_execution_date(execution_date: datetime) -> bool:
    """
    Validate that Airflow DAG execution date is a trading day.

    Use this in DAG tasks to skip execution on non-trading days.

    Args:
        execution_date: Airflow execution_date from context

    Returns:
        True if trading day, False otherwise

    Example:
        In your Airflow task:
        >>> def my_task(**context):
        ...     if not validate_dag_execution_date(context['execution_date']):
        ...         logging.info("Skipping: not a trading day")
        ...         return
        ...     # ... rest of task
    """
    return get_calendar().is_trading_day(execution_date)


def should_skip_dag_task(**context) -> str:
    """
    Airflow BranchPythonOperator function to skip non-trading days.

    Returns:
        'skip_task' if not a trading day, 'continue_task' otherwise

    Example:
        >>> from airflow.operators.python import BranchPythonOperator
        >>>
        >>> branch_task = BranchPythonOperator(
        ...     task_id='check_trading_day',
        ...     python_callable=should_skip_dag_task,
        ...     provide_context=True
        ... )
    """
    execution_date = context.get('execution_date')
    if execution_date and not validate_dag_execution_date(execution_date):
        logger.info(f"Skipping: {execution_date.date()} is not a trading day")
        return 'skip_task'
    return 'continue_task'


# =============================================================================
# TRAINING/BACKTESTING INTEGRATION
# =============================================================================

def validate_training_data(
    df: pd.DataFrame,
    date_col: str = 'timestamp',
    data_type: str = 'training'
) -> pd.DataFrame:
    """
    Validate and clean data for ML training/testing.

    CRITICAL: Ensures no holidays/weekends leak into model training.

    Args:
        df: DataFrame to validate
        date_col: Date column name
        data_type: Description for logging (e.g., 'training', 'validation', 'test')

    Returns:
        Cleaned DataFrame with only trading days

    Raises:
        ValueError: If DataFrame is empty after filtering

    Example:
        In run.py:
        >>> from services.common.trading_calendar import validate_training_data
        >>>
        >>> df_train = validate_training_data(df_train, data_type='training')
        >>> df_test = validate_training_data(df_test, data_type='test')
    """
    logger.info(f"Validating {data_type} data: {len(df):,} rows")

    cal = get_calendar()
    is_valid, violations = cal.validate_no_holidays(df, date_col)

    if not is_valid:
        logger.warning(
            f"{data_type.upper()} DATA CONTAMINATION DETECTED: "
            f"Found {len(violations)} non-trading days"
        )

        # Log details of violations
        for v in violations[:5]:
            logger.warning(f"  - {v['date']}: {v['reason']}")
        if len(violations) > 5:
            logger.warning(f"  ... and {len(violations) - 5} more violations")

    # Filter to trading days only
    df_clean = cal.filter_trading_days(df, date_col)

    if len(df_clean) == 0:
        raise ValueError(f"No valid trading days found in {data_type} data!")

    n_removed = len(df) - len(df_clean)
    if n_removed > 0:
        logger.info(
            f"Removed {n_removed:,} non-trading days from {data_type} data "
            f"({n_removed/len(df)*100:.2f}%)"
        )

    return df_clean


# =============================================================================
# PRODUCTION INFERENCE VALIDATION
# =============================================================================

def validate_inference_time(dt: Optional[datetime] = None) -> Tuple[bool, str]:
    """
    Validate that current time is valid for production inference.

    Checks:
    - Is it a trading day?
    - Are we within market hours?

    Args:
        dt: Datetime to check (default: now in COT timezone)

    Returns:
        Tuple of (is_valid, message)

    Example:
        >>> is_valid, msg = validate_inference_time()
        >>> if not is_valid:
        ...     logger.warning(f"Skipping inference: {msg}")
        ...     return
    """
    if dt is None:
        dt = datetime.now(COT_TZ)
    elif dt.tzinfo is None:
        dt = COT_TZ.localize(dt)

    cal = get_calendar()

    # Check if trading day
    if not cal.is_trading_day(dt):
        reason = cal.get_violation_reason(dt)
        return False, f"Not a trading day: {reason}"

    # Check market hours
    hour = dt.hour
    minute = dt.minute

    market_open = cal.market_open_hour * 60 + DEFAULT_MARKET_OPEN_MINUTE
    market_close = cal.market_close_hour * 60 + DEFAULT_MARKET_CLOSE_MINUTE
    current_minutes = hour * 60 + minute

    if current_minutes < market_open:
        return False, f"Before market open (market opens at {cal.market_open_hour}:00)"

    if current_minutes > market_close:
        return False, f"After market close (market closes at {cal.market_close_hour}:55)"

    return True, "Valid inference time"


# =============================================================================
# LEGACY COMPATIBILITY (for existing code)
# =============================================================================

# Singleton instance for backward compatibility
trading_calendar = get_calendar()


if __name__ == '__main__':
    """
    Test/demo script for trading calendar.

    Run with: python -m services.common.trading_calendar
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("USD/COP TRADING CALENDAR - VALIDATION TEST")
    print("="*70 + "\n")

    # Check library availability
    print("Library Status:")
    print(f"  colombian-holidays: {'✓ Installed' if HAS_COLOMBIAN_HOLIDAYS else '✗ Missing'}")
    print(f"  holidays (US):      {'✓ Installed' if HAS_US_HOLIDAYS else '✗ Missing'}")
    print()

    if not HAS_COLOMBIAN_HOLIDAYS:
        print("ERROR: colombian-holidays not installed!")
        print("Install with: pip install colombian-holidays")
        sys.exit(1)

    # Create calendar
    cal = TradingCalendar()

    # Test dates
    test_dates = [
        (date(2025, 1, 1), "New Year's Day"),
        (date(2025, 1, 6), "Epiphany (Reyes Magos)"),
        (date(2025, 1, 7), "Tuesday - should be trading day"),
        (date(2025, 1, 11), "Saturday - weekend"),
        (date(2025, 3, 24), "Monday before Holy Week"),
        (date(2025, 5, 1), "Labor Day"),
        (date(2025, 7, 20), "Independence Day"),
        (date(2025, 12, 25), "Christmas"),
    ]

    print("Testing specific dates:")
    print(f"{'Date':<15} {'Expected':<30} {'Is Trading?':<15} {'Reason'}")
    print("-" * 80)

    for dt, description in test_dates:
        is_trading = cal.is_trading_day(dt)
        reason = cal.get_violation_reason(dt) or "Valid trading day"
        status = "✗ No" if not is_trading else "✓ Yes"
        print(f"{dt.strftime('%Y-%m-%d'):<15} {description:<30} {status:<15} {reason}")

    print()

    # Get holidays in January 2025
    print("Colombian holidays in January 2025:")
    holidays_jan = cal.get_holidays_in_range(
        date(2025, 1, 1),
        date(2025, 1, 31),
        include_weekends=False
    )
    for dt, reason in holidays_jan:
        print(f"  - {dt}: {reason}")

    print()

    # Count trading days
    trading_days_jan = cal.get_trading_days_in_range(
        date(2025, 1, 1),
        date(2025, 1, 31)
    )
    print(f"Trading days in January 2025: {len(trading_days_jan)} out of 31 days")

    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")
