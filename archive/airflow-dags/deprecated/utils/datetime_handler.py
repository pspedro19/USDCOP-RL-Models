"""
USDCOP Trading System - Unified Datetime Handler
=================================================
Comprehensive datetime handling solution to fix ALL timezone-aware/naive mixing issues.

CRITICAL ISSUES ADDRESSED:
1. Mixed timezone-aware and timezone-naive datetime comparisons
2. Inconsistent timezone handling across pipeline stages
3. Business hours filtering with proper Colombian timezone support
4. TwelveData API timezone conversion standardization
5. Pandas datetime operations with proper timezone awareness

SYSTEM REQUIREMENTS:
- Colombian market hours: 8am-2pm COT (UTC-5)
- TwelveData API returns data in America/Bogota timezone
- 5-minute forex data for USD/COP pair
- Business days filtering with Colombian holidays
"""

import pandas as pd
import pytz
import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
import logging
from functools import wraps
import holidays

logger = logging.getLogger(__name__)

class UnifiedDatetimeHandler:
    """
    Unified datetime handling for USDCOP trading pipeline.
    Ensures consistent timezone-aware operations throughout the system.
    """

    # Timezone definitions
    COT_TZ = pytz.timezone('America/Bogota')  # Colombia Time (UTC-5)
    UTC_TZ = pytz.UTC

    # Market hours in COT
    MARKET_OPEN_HOUR = 8   # 8:00 AM COT
    MARKET_CLOSE_HOUR = 14  # 2:00 PM COT (market closes at 2 PM)

    # Colombian holidays cache
    _holidays_cache = {}

    @classmethod
    def ensure_timezone_aware(cls, dt: Union[datetime, pd.Timestamp, pd.Series],
                             assume_tz: str = 'America/Bogota') -> Union[datetime, pd.Timestamp, pd.Series]:
        """
        Ensure datetime object is timezone-aware.

        Args:
            dt: Datetime object (can be naive or aware)
            assume_tz: Timezone to assume if datetime is naive

        Returns:
            Timezone-aware datetime object
        """
        if isinstance(dt, pd.Series):
            # Handle pandas Series
            if dt.dt.tz is None:
                # Series is naive, localize to assumed timezone
                logger.debug(f"Localizing naive Series to {assume_tz}")
                return dt.dt.tz_localize(assume_tz)
            else:
                # Already timezone-aware
                return dt

        elif isinstance(dt, (datetime, pd.Timestamp)):
            # Handle single datetime objects
            if dt.tzinfo is None:
                # Naive datetime, localize to assumed timezone
                logger.debug(f"Localizing naive datetime to {assume_tz}")
                if assume_tz == 'America/Bogota':
                    return cls.COT_TZ.localize(dt)
                elif assume_tz == 'UTC':
                    return cls.UTC_TZ.localize(dt)
                else:
                    tz = pytz.timezone(assume_tz)
                    return tz.localize(dt)
            else:
                # Already timezone-aware
                return dt
        else:
            raise TypeError(f"Unsupported datetime type: {type(dt)}")

    @classmethod
    def convert_to_cot(cls, dt: Union[datetime, pd.Timestamp, pd.Series]) -> Union[datetime, pd.Timestamp, pd.Series]:
        """
        Convert datetime to Colombian time (COT).

        Args:
            dt: Timezone-aware datetime object

        Returns:
            Datetime in COT timezone
        """
        # First ensure it's timezone-aware
        dt_aware = cls.ensure_timezone_aware(dt)

        if isinstance(dt_aware, pd.Series):
            return dt_aware.dt.tz_convert('America/Bogota')
        else:
            return dt_aware.astimezone(cls.COT_TZ)

    @classmethod
    def convert_to_utc(cls, dt: Union[datetime, pd.Timestamp, pd.Series]) -> Union[datetime, pd.Timestamp, pd.Series]:
        """
        Convert datetime to UTC.

        Args:
            dt: Timezone-aware datetime object

        Returns:
            Datetime in UTC timezone
        """
        # First ensure it's timezone-aware
        dt_aware = cls.ensure_timezone_aware(dt)

        if isinstance(dt_aware, pd.Series):
            return dt_aware.dt.tz_convert('UTC')
        else:
            return dt_aware.astimezone(cls.UTC_TZ)

    @classmethod
    def standardize_dataframe_timestamps(cls, df: pd.DataFrame,
                                       timestamp_cols: List[str] = None,
                                       assume_tz: str = 'America/Bogota') -> pd.DataFrame:
        """
        Standardize all timestamp columns in a DataFrame to be timezone-aware.

        Args:
            df: DataFrame with timestamp columns
            timestamp_cols: List of timestamp column names (if None, auto-detect)
            assume_tz: Timezone to assume for naive timestamps

        Returns:
            DataFrame with timezone-aware timestamps
        """
        df = df.copy()

        # Auto-detect timestamp columns if not provided
        if timestamp_cols is None:
            timestamp_cols = []
            for col in df.columns:
                if 'time' in col.lower() or 'date' in col.lower():
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        timestamp_cols.append(col)

        logger.info(f"Standardizing timezone for columns: {timestamp_cols}")

        for col in timestamp_cols:
            if col in df.columns:
                # Convert to datetime first if not already
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    df[col] = pd.to_datetime(df[col])

                # Ensure timezone awareness
                df[col] = cls.ensure_timezone_aware(df[col], assume_tz)

                logger.debug(f"Column {col}: timezone = {df[col].dt.tz}")

        return df

    @classmethod
    def add_timezone_columns(cls, df: pd.DataFrame,
                           base_col: str = 'timestamp') -> pd.DataFrame:
        """
        Add standardized timezone columns to DataFrame.

        Args:
            df: DataFrame with base timestamp column
            base_col: Name of the base timestamp column

        Returns:
            DataFrame with additional timezone columns
        """
        df = df.copy()

        if base_col not in df.columns:
            raise KeyError(f"Base timestamp column '{base_col}' not found")

        # Ensure base column is timezone-aware
        df[base_col] = cls.ensure_timezone_aware(df[base_col])

        # Add standardized columns
        df['timestamp_utc'] = cls.convert_to_utc(df[base_col])
        df['timestamp_cot'] = cls.convert_to_cot(df[base_col])
        df['hour_cot'] = df['timestamp_cot'].dt.hour
        df['minute_cot'] = df['timestamp_cot'].dt.minute
        df['weekday'] = df['timestamp_cot'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['date_cot'] = df['timestamp_cot'].dt.date

        return df

    @classmethod
    def is_premium_hours(cls, dt: Union[datetime, pd.Series],
                        include_weekends: bool = False) -> Union[bool, pd.Series]:
        """
        Check if datetime is within premium trading hours (8am-2pm COT, Mon-Fri).

        Args:
            dt: Datetime in COT timezone
            include_weekends: Whether to include weekends

        Returns:
            Boolean or boolean Series indicating premium hours
        """
        # Convert to COT if not already
        dt_cot = cls.convert_to_cot(dt)

        if isinstance(dt_cot, pd.Series):
            # Handle Series
            hour_check = (dt_cot.dt.hour >= cls.MARKET_OPEN_HOUR) & (dt_cot.dt.hour < cls.MARKET_CLOSE_HOUR)
            if include_weekends:
                return hour_check
            else:
                weekday_check = dt_cot.dt.dayofweek < 5  # Monday=0, Friday=4
                return hour_check & weekday_check
        else:
            # Handle single datetime
            hour_check = cls.MARKET_OPEN_HOUR <= dt_cot.hour < cls.MARKET_CLOSE_HOUR
            if include_weekends:
                return hour_check
            else:
                weekday_check = dt_cot.weekday() < 5
                return hour_check and weekday_check

    @classmethod
    def get_colombian_holidays(cls, year: int) -> set:
        """
        Get Colombian holidays for a given year.

        Args:
            year: Year to get holidays for

        Returns:
            Set of date objects representing Colombian holidays
        """
        if year not in cls._holidays_cache:
            try:
                co_holidays = holidays.Colombia(years=year)
                cls._holidays_cache[year] = set(co_holidays.keys())
                logger.debug(f"Loaded {len(cls._holidays_cache[year])} Colombian holidays for {year}")
            except Exception as e:
                logger.warning(f"Could not load Colombian holidays for {year}: {e}")
                # Fallback to basic holidays
                cls._holidays_cache[year] = {
                    date(year, 1, 1),   # New Year
                    date(year, 5, 1),   # Labor Day
                    date(year, 7, 20),  # Independence Day
                    date(year, 8, 7),   # Battle of Boyacá
                    date(year, 12, 25), # Christmas
                }

        return cls._holidays_cache[year]

    @classmethod
    def is_business_day(cls, dt: Union[datetime, pd.Series, date]) -> Union[bool, pd.Series]:
        """
        Check if datetime is a Colombian business day (excludes holidays).

        Args:
            dt: Datetime or date object

        Returns:
            Boolean indicating if it's a business day
        """
        if isinstance(dt, pd.Series):
            # Handle Series
            dt_cot = cls.convert_to_cot(dt) if hasattr(dt.iloc[0], 'tzinfo') else dt
            dates = dt_cot.dt.date

            result = pd.Series(True, index=dt.index)

            # Check weekends
            weekends = dt_cot.dt.dayofweek >= 5  # Saturday=5, Sunday=6
            result = result & ~weekends

            # Check holidays for each year
            years = dt_cot.dt.year.unique()
            for year in years:
                year_holidays = cls.get_colombian_holidays(year)
                year_mask = dt_cot.dt.year == year
                holiday_mask = dates.isin(year_holidays)
                result = result & ~(year_mask & holiday_mask)

            return result

        else:
            # Handle single datetime/date
            if isinstance(dt, datetime):
                dt_cot = cls.convert_to_cot(dt)
                check_date = dt_cot.date()
                weekday = dt_cot.weekday()
            else:  # date object
                check_date = dt
                weekday = dt.weekday()

            # Check weekend
            if weekday >= 5:  # Saturday=5, Sunday=6
                return False

            # Check holidays
            year_holidays = cls.get_colombian_holidays(check_date.year)
            return check_date not in year_holidays

    @classmethod
    def filter_business_hours(cls, df: pd.DataFrame,
                             timestamp_col: str = 'timestamp',
                             include_holidays: bool = False) -> pd.DataFrame:
        """
        Filter DataFrame to include only business hours data.

        Args:
            df: DataFrame with timestamp column
            timestamp_col: Name of timestamp column
            include_holidays: Whether to include holidays

        Returns:
            Filtered DataFrame with only business hours data
        """
        df_filtered = df.copy()

        # Ensure timestamp column is timezone-aware
        df_filtered = cls.standardize_dataframe_timestamps(df_filtered, [timestamp_col])

        # Add timezone columns if not present
        if 'timestamp_cot' not in df_filtered.columns:
            df_filtered = cls.add_timezone_columns(df_filtered, timestamp_col)

        # Filter for premium hours
        premium_mask = cls.is_premium_hours(df_filtered['timestamp_cot'])
        df_filtered = df_filtered[premium_mask].copy()

        # Filter for business days (exclude holidays)
        if not include_holidays:
            business_mask = cls.is_business_day(df_filtered['timestamp_cot'])
            df_filtered = df_filtered[business_mask].copy()

        logger.info(f"Filtered from {len(df)} to {len(df_filtered)} rows (business hours only)")

        return df_filtered

    @classmethod
    def calculate_time_differences(cls, timestamps: pd.Series,
                                 expected_interval_minutes: int = 5) -> pd.Series:
        """
        Calculate time differences with proper timezone handling.

        Args:
            timestamps: Series of timestamps
            expected_interval_minutes: Expected interval in minutes

        Returns:
            Series of time differences in minutes
        """
        # Ensure timezone awareness
        timestamps_aware = cls.ensure_timezone_aware(timestamps)

        # Sort timestamps
        timestamps_sorted = timestamps_aware.sort_values()

        # Calculate differences
        time_diffs = timestamps_sorted.diff()

        # Convert to minutes
        diff_minutes = time_diffs.dt.total_seconds() / 60

        return diff_minutes

    @classmethod
    def generate_expected_timestamps(cls, start_time: datetime, end_time: datetime,
                                   interval_minutes: int = 5,
                                   business_hours_only: bool = True,
                                   exclude_holidays: bool = True) -> List[datetime]:
        """
        Generate expected timestamps for a given period.

        Args:
            start_time: Start datetime (will be converted to COT)
            end_time: End datetime (will be converted to COT)
            interval_minutes: Interval in minutes
            business_hours_only: Only include business hours
            exclude_holidays: Exclude Colombian holidays

        Returns:
            List of expected timestamps in COT
        """
        # Ensure timezone awareness and convert to COT
        start_cot = cls.convert_to_cot(cls.ensure_timezone_aware(start_time))
        end_cot = cls.convert_to_cot(cls.ensure_timezone_aware(end_time))

        timestamps = []
        current = start_cot

        while current <= end_cot:
            include_timestamp = True

            if business_hours_only:
                if not cls.is_premium_hours(current):
                    include_timestamp = False

            if exclude_holidays:
                if not cls.is_business_day(current):
                    include_timestamp = False

            if include_timestamp:
                timestamps.append(current)

            current += timedelta(minutes=interval_minutes)

        return timestamps

def timezone_safe(func):
    """
    Decorator to ensure all datetime operations in a function are timezone-safe.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed with timezone-safe operations")
            return result
        except Exception as e:
            if "timezone" in str(e).lower() or "naive" in str(e).lower():
                logger.error(f"Timezone-related error in {func.__name__}: {e}")
                logger.error("Hint: Use UnifiedDatetimeHandler methods to ensure timezone consistency")
            raise
    return wrapper

# Convenience functions for common operations
def ensure_timezone_aware(dt, assume_tz='America/Bogota'):
    """Convenience function for timezone awareness"""
    return UnifiedDatetimeHandler.ensure_timezone_aware(dt, assume_tz)

def convert_to_cot(dt):
    """Convenience function to convert to Colombian time"""
    return UnifiedDatetimeHandler.convert_to_cot(dt)

def convert_to_utc(dt):
    """Convenience function to convert to UTC"""
    return UnifiedDatetimeHandler.convert_to_utc(dt)

def is_premium_hours(dt):
    """Convenience function to check premium hours"""
    return UnifiedDatetimeHandler.is_premium_hours(dt)

def is_business_day(dt):
    """Convenience function to check business day"""
    return UnifiedDatetimeHandler.is_business_day(dt)

def standardize_dataframe_timestamps(df, timestamp_cols=None):
    """Convenience function to standardize DataFrame timestamps"""
    return UnifiedDatetimeHandler.standardize_dataframe_timestamps(df, timestamp_cols)

# Example usage and testing
if __name__ == "__main__":
    # Test the datetime handler
    import pandas as pd

    # Create test data with mixed timezone awareness
    test_data = pd.DataFrame({
        'timestamp': [
            datetime(2024, 1, 15, 9, 0),  # Naive datetime
            datetime(2024, 1, 15, 9, 5),  # Naive datetime
            datetime(2024, 1, 15, 9, 10), # Naive datetime
        ],
        'price': [4200.0, 4201.5, 4199.8]
    })

    print("Original data:")
    print(test_data.dtypes)
    print(test_data['timestamp'].dt.tz)

    # Standardize timestamps
    handler = UnifiedDatetimeHandler()
    test_data_fixed = handler.standardize_dataframe_timestamps(test_data)

    print("\nAfter standardization:")
    print(test_data_fixed.dtypes)
    print(test_data_fixed['timestamp'].dt.tz)

    # Add timezone columns
    test_data_enhanced = handler.add_timezone_columns(test_data_fixed)

    print("\nWith timezone columns:")
    print(test_data_enhanced.columns.tolist())
    print(test_data_enhanced[['timestamp', 'timestamp_cot', 'timestamp_utc', 'hour_cot']])

    # Test premium hours
    premium_mask = handler.is_premium_hours(test_data_enhanced['timestamp_cot'])
    print(f"\nPremium hours mask: {premium_mask.tolist()}")

    # Test business day
    business_mask = handler.is_business_day(test_data_enhanced['timestamp_cot'])
    print(f"Business day mask: {business_mask.tolist()}")

    print("\n✅ UnifiedDatetimeHandler test completed successfully!")