"""
Trading Hours Filter for USD/COP Market Data
=============================================

This module provides filtering and validation for Colombian trading hours.
It ensures data is filtered to valid market sessions using SSOT configuration.

Contract ID: P0-07
Author: Trading Team
Version: 2.1.0
Created: 2026-01-17
Updated: 2026-01-17

Design Patterns:
    - Protocol: ITradingHoursFilter interface for dependency injection
    - Factory: TradingHoursFilterFactory for creating market-specific filters
    - SSOT: All config loaded from config/trading_calendar.json
    - Immutable: Frozen dataclasses prevent runtime config changes

Architecture:
    config/trading_calendar.json  <-- SINGLE SOURCE OF TRUTH
           |
           v
    TradingCalendarConfig        <-- Immutable frozen dataclass
           |
           v
    TradingHoursFilterFactory    <-- Factory for creating filters
           |
           +---> ColombianTradingHoursFilter (default)
           +---> USMarketTradingHoursFilter (for macro data)
           +---> CustomTradingHoursFilter (configurable)

Usage:
    # Using factory (recommended)
    >>> from src.features.trading_hours_filter import get_trading_hours_filter
    >>> filter = get_trading_hours_filter()  # Loads from SSOT config
    >>> df_filtered = filter.filter(df, time_col='timestamp')

    # Direct instantiation with config
    >>> from src.features.trading_hours_filter import (
    ...     TradingHoursFilter,
    ...     load_trading_calendar_config
    ... )
    >>> config = load_trading_calendar_config()
    >>> filter = TradingHoursFilter(config)

Integration:
    Works with existing TradingCalendar from services.common.trading_calendar
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import pytz

# Import centralized constants from SSOT
from src.core.constants import (
    TRADING_TIMEZONE,
    TRADING_START_HOUR,
    TRADING_END_HOUR,
    UTC_OFFSET_BOGOTA,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS - SSOT References
# =============================================================================

# Path to the Single Source of Truth configuration file
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
TRADING_CALENDAR_CONFIG_PATH = CONFIG_DIR / "trading_calendar.json"

# Re-export SSOT constants for backward compatibility
DEFAULT_TIMEZONE = TRADING_TIMEZONE
DEFAULT_START_HOUR = TRADING_START_HOUR
DEFAULT_END_HOUR = TRADING_END_HOUR

# Weekday indices (Monday=0, Friday=4) - immutable
TRADING_WEEKDAYS: FrozenSet[int] = frozenset({0, 1, 2, 3, 4})
WEEKEND_DAYS: FrozenSet[int] = frozenset({5, 6})

# Named constants for hours (no magic numbers)
HOUR_MIN = 0
HOUR_MAX = 24
MINUTES_PER_HOUR = 60


# =============================================================================
# ENUMS
# =============================================================================

class MarketType(str, Enum):
    """Market types for filter factory."""
    COLOMBIA = "colombia"
    USA = "usa"
    CUSTOM = "custom"


class HolidayType(str, Enum):
    """Holiday classification types."""
    COLOMBIAN = "colombia"
    US = "usa"
    CUSTOM = "custom"


# =============================================================================
# PROTOCOL INTERFACE
# =============================================================================

@runtime_checkable
class ITradingHoursFilter(Protocol):
    """
    Protocol interface for trading hours filtering.

    Implementations must provide:
    - filter(): Filter DataFrame to trading hours
    - is_trading_time(): Check single timestamp validity
    - get_session(): Get current session configuration

    This enables dependency injection and testing with mock implementations.
    """

    def filter(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to valid trading hours.

        Args:
            df: DataFrame to filter
            time_col: Name of timestamp column

        Returns:
            Filtered DataFrame (copy, original not modified)
        """
        ...

    def is_trading_time(
        self,
        timestamp: Union[datetime, pd.Timestamp],
    ) -> bool:
        """
        Check if timestamp is within trading hours.

        Args:
            timestamp: Datetime to check

        Returns:
            True if valid trading time
        """
        ...

    def get_session(self) -> TradingSession:
        """
        Get trading session configuration.

        Returns:
            TradingSession with hours and timezone
        """
        ...

    @property
    def timezone(self) -> str:
        """Market timezone string."""
        ...

    @property
    def holidays(self) -> FrozenSet[str]:
        """Set of all holiday dates (YYYY-MM-DD format)."""
        ...


# =============================================================================
# IMMUTABLE CONFIG DATACLASSES (Frozen = cannot be modified after creation)
# =============================================================================

@dataclass(frozen=True)
class TradingSession:
    """
    Immutable trading session window configuration.

    Represents a time window during which trading is active.
    All validation happens at construction time (fail-fast).

    Attributes:
        start_time: Session start time (e.g., time(8, 0) for 08:00)
        end_time: Session end time (e.g., time(13, 0) for 13:00)
        timezone: Timezone string (e.g., "America/Bogota")
    """
    start_time: time
    end_time: time
    timezone: str

    def __post_init__(self) -> None:
        """Validate session parameters at construction time."""
        if self.start_time >= self.end_time:
            raise ValueError(
                f"start_time ({self.start_time}) must be before "
                f"end_time ({self.end_time})"
            )
        # Validate timezone exists
        try:
            pytz.timezone(self.timezone)
        except pytz.UnknownTimeZoneError as e:
            raise ValueError(f"Invalid timezone: {self.timezone}") from e

    def is_within_session(
        self,
        timestamp: Union[datetime, pd.Timestamp],
    ) -> bool:
        """
        Check if timestamp falls within this trading session.

        Handles timezone conversion automatically. If the timestamp is
        timezone-naive, it is assumed to be in UTC.

        Args:
            timestamp: Datetime or pandas Timestamp to check

        Returns:
            True if timestamp is within session hours
        """
        tz = pytz.timezone(self.timezone)

        # Convert to datetime if needed
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        if timestamp.tzinfo is None:
            # Assume UTC for naive timestamps
            timestamp = pytz.UTC.localize(timestamp)

        local_dt = timestamp.astimezone(tz)
        local_time = local_dt.time()

        return self.start_time <= local_time < self.end_time

    def get_session_bounds(
        self,
        date_obj: date,
    ) -> Tuple[datetime, datetime]:
        """
        Get absolute datetime bounds for session on given date.

        Args:
            date_obj: Date to get session bounds for

        Returns:
            Tuple of (session_start, session_end) as timezone-aware datetimes
        """
        tz = pytz.timezone(self.timezone)
        start_dt = tz.localize(datetime.combine(date_obj, self.start_time))
        end_dt = tz.localize(datetime.combine(date_obj, self.end_time))
        return start_dt, end_dt


@dataclass(frozen=True)
class TradingCalendarConfig:
    """
    Immutable trading calendar configuration - SSOT.

    Loaded from config/trading_calendar.json and cannot be modified
    after creation. All trading-related code should use this config
    instead of hardcoding values.

    Attributes:
        timezone: Market timezone (e.g., "America/Bogota")
        start_hour: Market open hour in local time
        start_minute: Market open minute in local time
        end_hour: Market close hour in local time
        end_minute: Market close minute in local time
        holidays_colombia: Frozen set of Colombian holidays
        holidays_usa: Frozen set of US holidays
        bars_per_session: Number of 5-min bars per trading session
        bar_duration_minutes: Duration of each bar in minutes
    """
    timezone: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int
    holidays_colombia: FrozenSet[str]
    holidays_usa: FrozenSet[str]
    bars_per_session: int
    bar_duration_minutes: int

    def __post_init__(self) -> None:
        """Validate configuration at construction time."""
        # Validate timezone
        try:
            pytz.timezone(self.timezone)
        except pytz.UnknownTimeZoneError as e:
            raise ValueError(f"Invalid timezone: {self.timezone}") from e

        # Validate hours
        if not (HOUR_MIN <= self.start_hour < HOUR_MAX):
            raise ValueError(
                f"start_hour must be {HOUR_MIN}-{HOUR_MAX - 1}, "
                f"got {self.start_hour}"
            )
        if not (HOUR_MIN <= self.end_hour <= HOUR_MAX):
            raise ValueError(
                f"end_hour must be {HOUR_MIN}-{HOUR_MAX}, "
                f"got {self.end_hour}"
            )

        # Validate start is before end
        start_minutes = self.start_hour * MINUTES_PER_HOUR + self.start_minute
        end_minutes = self.end_hour * MINUTES_PER_HOUR + self.end_minute
        if start_minutes >= end_minutes:
            raise ValueError(
                f"Start time ({self.start_hour:02d}:{self.start_minute:02d}) "
                f"must be before end time ({self.end_hour:02d}:{self.end_minute:02d})"
            )

        logger.debug(
            f"TradingCalendarConfig validated: tz={self.timezone}, "
            f"hours={self.start_hour:02d}:{self.start_minute:02d}-"
            f"{self.end_hour:02d}:{self.end_minute:02d}, "
            f"holidays_co={len(self.holidays_colombia)}, "
            f"holidays_us={len(self.holidays_usa)}"
        )

    @property
    def session(self) -> TradingSession:
        """Get TradingSession from this config."""
        return TradingSession(
            start_time=time(self.start_hour, self.start_minute),
            end_time=time(self.end_hour, self.end_minute),
            timezone=self.timezone,
        )

    @property
    def all_holidays(self) -> FrozenSet[str]:
        """Combined set of all holidays."""
        return self.holidays_colombia | self.holidays_usa

    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "timezone": self.timezone,
            "start_hour": self.start_hour,
            "start_minute": self.start_minute,
            "end_hour": self.end_hour,
            "end_minute": self.end_minute,
            "holidays_colombia": sorted(self.holidays_colombia),
            "holidays_usa": sorted(self.holidays_usa),
            "bars_per_session": self.bars_per_session,
            "bar_duration_minutes": self.bar_duration_minutes,
        }


# =============================================================================
# CONFIG LOADING FUNCTIONS
# =============================================================================

class ConfigLoadError(Exception):
    """Raised when configuration cannot be loaded."""
    pass


def load_trading_calendar_config(
    config_path: Optional[Union[str, Path]] = None,
) -> TradingCalendarConfig:
    """
    Load trading calendar configuration from JSON file (SSOT).

    This function loads the Single Source of Truth configuration from
    config/trading_calendar.json. All trading hours, timezone, and
    holiday data should come from this source.

    Args:
        config_path: Path to config file (default: config/trading_calendar.json)

    Returns:
        Immutable TradingCalendarConfig instance

    Raises:
        ConfigLoadError: If config file not found or invalid
        ValueError: If config validation fails

    Example:
        >>> config = load_trading_calendar_config()
        >>> print(config.timezone)
        'America/Bogota'
        >>> print(config.start_hour)
        8
    """
    if config_path is None:
        config_path = TRADING_CALENDAR_CONFIG_PATH

    path = Path(config_path)
    if not path.exists():
        raise ConfigLoadError(
            f"Trading calendar config not found: {path}. "
            f"Expected SSOT file at: {TRADING_CALENDAR_CONFIG_PATH}"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigLoadError(
            f"Invalid JSON in trading calendar config: {e}"
        ) from e

    # Parse market hours
    market_hours = data.get("market_hours", {})
    timezone = market_hours.get("timezone", "America/Bogota")

    # Parse start/end times from local time strings
    start_local = market_hours.get("start_local", "08:00")
    end_local = market_hours.get("end_local", "12:55")

    start_parts = start_local.split(":")
    end_parts = end_local.split(":")

    start_hour = int(start_parts[0])
    start_minute = int(start_parts[1]) if len(start_parts) > 1 else 0
    end_hour = int(end_parts[0])
    end_minute = int(end_parts[1]) if len(end_parts) > 1 else 0

    # Collect all holidays from all years in config
    holidays_colombia: set[str] = set()
    holidays_usa: set[str] = set()

    for key, value in data.items():
        if key.startswith("holidays_") and "_colombia" in key:
            if isinstance(value, list):
                holidays_colombia.update(value)
        elif key.startswith("holidays_") and "_usa" in key:
            if isinstance(value, list):
                holidays_usa.update(value)

    # Parse session info
    session_info = data.get("session_info", {})
    bars_per_session = session_info.get("bars_per_session", 60)
    bar_duration_minutes = session_info.get("bar_duration_minutes", 5)

    logger.info(
        f"Loaded trading calendar config from {path}: "
        f"tz={timezone}, hours={start_hour:02d}:{start_minute:02d}-"
        f"{end_hour:02d}:{end_minute:02d}, "
        f"holidays_co={len(holidays_colombia)}, "
        f"holidays_us={len(holidays_usa)}"
    )

    return TradingCalendarConfig(
        timezone=timezone,
        start_hour=start_hour,
        start_minute=start_minute,
        end_hour=end_hour,
        end_minute=end_minute,
        holidays_colombia=frozenset(holidays_colombia),
        holidays_usa=frozenset(holidays_usa),
        bars_per_session=bars_per_session,
        bar_duration_minutes=bar_duration_minutes,
    )


# =============================================================================
# MAIN FILTER CLASS
# =============================================================================

class TradingHoursFilter:
    """
    Filter DataFrames to Colombian trading hours.

    Implements ITradingHoursFilter protocol. Provides comprehensive
    filtering functionality using SSOT configuration:
    - Convert timezone-naive timestamps to timezone-aware
    - Filter to weekdays only (Monday-Friday)
    - Filter to trading hours from SSOT config
    - Filter out holidays from SSOT config
    - Log retention statistics

    Attributes:
        config: Immutable TradingCalendarConfig (SSOT)

    Example:
        >>> config = load_trading_calendar_config()
        >>> filter = TradingHoursFilter(config)
        >>> df_filtered = filter.filter(df, time_col='timestamp')
        >>> print(f"Retained {len(df_filtered)}/{len(df)} rows")
    """

    def __init__(
        self,
        config: Optional[TradingCalendarConfig] = None,
        *,  # Force keyword arguments after this
        include_us_holidays: bool = True,
    ) -> None:
        """
        Initialize TradingHoursFilter.

        Args:
            config: TradingCalendarConfig instance. If None, loads from SSOT.
            include_us_holidays: Whether to exclude US holidays (default: True)

        Raises:
            ConfigLoadError: If config cannot be loaded
            ValueError: If config is invalid
        """
        if config is None:
            config = load_trading_calendar_config()

        self._config = config
        self._include_us_holidays = include_us_holidays

        # Compute timezone object once
        self._tz = pytz.timezone(config.timezone)

        # Compute holiday set based on options
        if include_us_holidays:
            self._holidays = config.all_holidays
        else:
            self._holidays = config.holidays_colombia

        # Create session object
        self._session = config.session

        logger.debug(
            f"TradingHoursFilter initialized: "
            f"timezone={config.timezone}, "
            f"hours={config.start_hour:02d}:{config.start_minute:02d}-"
            f"{config.end_hour:02d}:{config.end_minute:02d}, "
            f"total_holidays={len(self._holidays)}"
        )

    @property
    def timezone(self) -> str:
        """Market timezone string."""
        return self._config.timezone

    @property
    def holidays(self) -> FrozenSet[str]:
        """Set of all holiday dates (YYYY-MM-DD format)."""
        return self._holidays

    @property
    def config(self) -> TradingCalendarConfig:
        """Get immutable configuration."""
        return self._config

    def get_session(self) -> TradingSession:
        """Get trading session configuration."""
        return self._session

    def filter(
        self,
        df: pd.DataFrame,
        time_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        Filter DataFrame to Colombian trading hours.

        Applies the following filters in order:
        1. Convert to timezone-aware if needed (assumes UTC for naive)
        2. Filter to weekdays (Monday-Friday)
        3. Filter to trading hours from SSOT config
        4. Filter out holidays from SSOT config

        Args:
            df: DataFrame to filter
            time_col: Name of timestamp column (default: 'timestamp')

        Returns:
            Filtered DataFrame (copy, original not modified)

        Raises:
            ValueError: If time_col not found in DataFrame
        """
        if df.empty:
            logger.info("Empty DataFrame provided, returning empty result")
            return df.copy()

        if time_col not in df.columns:
            raise ValueError(
                f"Column '{time_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        original_len = len(df)
        logger.info(f"Filtering DataFrame with {original_len:,} rows")

        # Make a copy to avoid modifying original
        df_filtered = df.copy()

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df_filtered[time_col]):
            df_filtered[time_col] = pd.to_datetime(df_filtered[time_col])

        # Step 1: Convert to timezone-aware if needed
        ts_series = df_filtered[time_col]
        if ts_series.dt.tz is None:
            df_filtered[time_col] = ts_series.dt.tz_localize("UTC")
            logger.debug("Converted timezone-naive timestamps to UTC")

        # Convert to local timezone for filtering
        ts_local = df_filtered[time_col].dt.tz_convert(self._tz)

        # Step 2: Filter weekdays
        weekday_mask = ts_local.dt.weekday.isin(TRADING_WEEKDAYS)
        df_filtered = df_filtered[weekday_mask]
        after_weekday = len(df_filtered)
        logger.debug(f"After weekday filter: {after_weekday:,} rows")

        if df_filtered.empty:
            self._log_retention(original_len, 0, "all rows filtered (weekends)")
            return df_filtered

        # Recompute local timestamps after filtering
        ts_local = df_filtered[time_col].dt.tz_convert(self._tz)

        # Step 3: Filter trading hours using SSOT config
        start_minutes = (
            self._config.start_hour * MINUTES_PER_HOUR +
            self._config.start_minute
        )
        end_minutes = (
            self._config.end_hour * MINUTES_PER_HOUR +
            self._config.end_minute
        )

        current_minutes = (
            ts_local.dt.hour * MINUTES_PER_HOUR +
            ts_local.dt.minute
        )
        hour_mask = (current_minutes >= start_minutes) & (current_minutes < end_minutes)
        df_filtered = df_filtered[hour_mask]
        after_hours = len(df_filtered)
        logger.debug(f"After trading hours filter: {after_hours:,} rows")

        if df_filtered.empty:
            self._log_retention(
                original_len, 0, "all rows filtered (outside trading hours)"
            )
            return df_filtered

        # Recompute local timestamps after filtering
        ts_local = df_filtered[time_col].dt.tz_convert(self._tz)

        # Step 4: Filter holidays using SSOT config
        date_str_series = ts_local.dt.strftime("%Y-%m-%d")
        holiday_mask = ~date_str_series.isin(self._holidays)
        df_filtered = df_filtered[holiday_mask]
        after_holidays = len(df_filtered)
        logger.debug(f"After holiday filter: {after_holidays:,} rows")

        # Log retention rate
        self._log_retention(original_len, after_holidays)

        return df_filtered.reset_index(drop=True)

    def _log_retention(
        self,
        original: int,
        final: int,
        reason: Optional[str] = None,
    ) -> None:
        """Log filtering retention statistics."""
        if original == 0:
            rate = 0.0
        else:
            rate = (final / original) * 100

        removed = original - final

        if reason:
            logger.info(
                f"Retention: {final:,}/{original:,} ({rate:.1f}%) - {reason}"
            )
        else:
            logger.info(
                f"Retention: {final:,}/{original:,} ({rate:.1f}%) - "
                f"removed {removed:,} rows"
            )

    def is_trading_time(
        self,
        timestamp: Union[datetime, pd.Timestamp],
    ) -> bool:
        """
        Check if a single timestamp is within trading hours.

        Validates:
        - Is a weekday (Monday-Friday)
        - Within trading hours from SSOT config
        - Not a holiday from SSOT config

        Args:
            timestamp: Datetime or pandas Timestamp to check

        Returns:
            True if timestamp is valid trading time, False otherwise
        """
        # Convert to datetime if needed
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Handle timezone-naive timestamps
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)

        # Convert to local timezone
        local_dt = timestamp.astimezone(self._tz)

        # Check weekday
        if local_dt.weekday() not in TRADING_WEEKDAYS:
            return False

        # Check trading hours using SSOT config
        current_minutes = (
            local_dt.hour * MINUTES_PER_HOUR +
            local_dt.minute
        )
        start_minutes = (
            self._config.start_hour * MINUTES_PER_HOUR +
            self._config.start_minute
        )
        end_minutes = (
            self._config.end_hour * MINUTES_PER_HOUR +
            self._config.end_minute
        )

        if not (start_minutes <= current_minutes < end_minutes):
            return False

        # Check holidays
        date_str = local_dt.strftime("%Y-%m-%d")
        if date_str in self._holidays:
            return False

        return True

    def add_holiday(
        self,
        holiday_date: Union[str, date],
        holiday_type: HolidayType = HolidayType.CUSTOM,
    ) -> None:
        """
        Add a custom holiday to the filter.

        Note: This modifies the internal holiday set but does NOT modify
        the SSOT config file. For permanent changes, update the config file.

        Args:
            holiday_date: Date as YYYY-MM-DD string or date object
            holiday_type: Type of holiday for logging
        """
        if isinstance(holiday_date, date):
            date_str = holiday_date.strftime("%Y-%m-%d")
        else:
            date_str = holiday_date

        # Convert to mutable set, add, convert back
        holidays_set = set(self._holidays)
        holidays_set.add(date_str)
        self._holidays = frozenset(holidays_set)

        logger.info(f"Added holiday: {date_str} (type={holiday_type.value})")

    def to_dict(self) -> Dict[str, Any]:
        """Export filter configuration as dictionary."""
        return {
            "timezone": self._config.timezone,
            "start_hour": self._config.start_hour,
            "start_minute": self._config.start_minute,
            "end_hour": self._config.end_hour,
            "end_minute": self._config.end_minute,
            "include_us_holidays": self._include_us_holidays,
            "total_holidays": len(self._holidays),
        }

    def __repr__(self) -> str:
        """String representation of filter."""
        return (
            f"TradingHoursFilter("
            f"timezone='{self._config.timezone}', "
            f"hours={self._config.start_hour:02d}:{self._config.start_minute:02d}-"
            f"{self._config.end_hour:02d}:{self._config.end_minute:02d}, "
            f"holidays={len(self._holidays)})"
        )


# =============================================================================
# FACTORY PATTERN
# =============================================================================

class TradingHoursFilterFactory:
    """
    Factory for creating market-specific trading hours filters.

    Implements Factory Pattern to create filters with appropriate
    configuration for different markets:
    - Colombia: Uses SSOT config with Colombian holidays
    - USA: Uses US market hours and NYSE holidays
    - Custom: User-provided configuration

    Example:
        >>> factory = TradingHoursFilterFactory()
        >>> col_filter = factory.create(MarketType.COLOMBIA)
        >>> us_filter = factory.create(MarketType.USA)
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Initialize factory.

        Args:
            config_path: Path to SSOT config file (default: config/trading_calendar.json)
        """
        self._config_path = config_path

    def create(
        self,
        market_type: MarketType = MarketType.COLOMBIA,
        *,
        include_us_holidays: bool = True,
        custom_config: Optional[TradingCalendarConfig] = None,
    ) -> TradingHoursFilter:
        """
        Create a trading hours filter for specified market.

        Args:
            market_type: Market type to create filter for
            include_us_holidays: Whether to include US holidays (for COLOMBIA)
            custom_config: Custom config (for CUSTOM market type)

        Returns:
            Configured TradingHoursFilter instance

        Raises:
            ValueError: If custom_config missing for CUSTOM market type
        """
        if market_type == MarketType.CUSTOM:
            if custom_config is None:
                raise ValueError(
                    "custom_config required for MarketType.CUSTOM"
                )
            return TradingHoursFilter(
                config=custom_config,
                include_us_holidays=include_us_holidays,
            )

        if market_type == MarketType.USA:
            # US market uses different hours and only US holidays
            # This would need a separate config file in production
            logger.warning(
                "US market filter not fully implemented. "
                "Using Colombia config with US holidays only."
            )
            config = load_trading_calendar_config(self._config_path)
            return TradingHoursFilter(
                config=config,
                include_us_holidays=True,
            )

        # Default: Colombia market
        config = load_trading_calendar_config(self._config_path)
        return TradingHoursFilter(
            config=config,
            include_us_holidays=include_us_holidays,
        )


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_default_filter: Optional[TradingHoursFilter] = None


def get_trading_hours_filter(
    *,
    force_reload: bool = False,
) -> TradingHoursFilter:
    """
    Get singleton trading hours filter instance.

    Loads configuration from SSOT (config/trading_calendar.json) on first
    call and caches the instance for subsequent calls.

    Args:
        force_reload: If True, reload config and create new instance

    Returns:
        TradingHoursFilter instance

    Example:
        >>> filter = get_trading_hours_filter()
        >>> df_filtered = filter.filter(df)
    """
    global _default_filter

    if _default_filter is None or force_reload:
        _default_filter = TradingHoursFilter()
        logger.info("Created default TradingHoursFilter from SSOT config")

    return _default_filter


def filter_to_trading_hours(
    df: pd.DataFrame,
    time_col: str = "timestamp",
) -> pd.DataFrame:
    """
    Convenience: Filter DataFrame to trading hours using default filter.

    Args:
        df: DataFrame to filter
        time_col: Name of timestamp column

    Returns:
        Filtered DataFrame
    """
    return get_trading_hours_filter().filter(df, time_col)


def is_trading_time(
    timestamp: Union[datetime, pd.Timestamp],
) -> bool:
    """
    Convenience: Check if timestamp is trading time using default filter.

    Args:
        timestamp: Datetime to check

    Returns:
        True if valid trading time
    """
    return get_trading_hours_filter().is_trading_time(timestamp)


# =============================================================================
# INTEGRATION WITH services.common.trading_calendar
# =============================================================================

def create_filter_from_trading_calendar() -> TradingHoursFilter:
    """
    Create TradingHoursFilter using TradingCalendar from services.common.

    This provides integration with the existing trading calendar
    implementation for backward compatibility.

    Returns:
        TradingHoursFilter instance

    Note:
        Falls back to SSOT config if TradingCalendar import fails.
    """
    try:
        from services.common.trading_calendar import TradingCalendar, get_calendar

        # Get the existing calendar instance
        calendar = get_calendar()

        # Load SSOT config as base
        config = load_trading_calendar_config()

        # Create filter with SSOT config
        # The TradingCalendar class provides holiday checking that
        # can be used for validation
        filter_instance = TradingHoursFilter(config=config)

        logger.info(
            "Created TradingHoursFilter integrated with "
            "services.common.trading_calendar"
        )
        return filter_instance

    except ImportError:
        logger.warning(
            "services.common.trading_calendar not available. "
            "Using standalone SSOT config."
        )
        return TradingHoursFilter()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Protocol
    "ITradingHoursFilter",
    # Config classes
    "TradingCalendarConfig",
    "TradingSession",
    # Main class
    "TradingHoursFilter",
    # Factory
    "TradingHoursFilterFactory",
    "MarketType",
    "HolidayType",
    # Config loading
    "load_trading_calendar_config",
    "ConfigLoadError",
    # Convenience functions
    "get_trading_hours_filter",
    "filter_to_trading_hours",
    "is_trading_time",
    # Integration
    "create_filter_from_trading_calendar",
    # Constants (from SSOT)
    "DEFAULT_TIMEZONE",
    "DEFAULT_START_HOUR",
    "DEFAULT_END_HOUR",
    "TRADING_WEEKDAYS",
    "WEEKEND_DAYS",
    "TRADING_CALENDAR_CONFIG_PATH",
]


# =============================================================================
# MAIN (for testing)
# =============================================================================

if __name__ == "__main__":
    """
    Test/demo script for TradingHoursFilter.

    Run with: python -m src.features.trading_hours_filter
    """
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    print("\n" + "=" * 70)
    print("TRADING HOURS FILTER - TEST (SSOT Version)")
    print("=" * 70 + "\n")

    # Load SSOT config
    print("Loading SSOT configuration...")
    try:
        config = load_trading_calendar_config()
        print(f"  Config path: {TRADING_CALENDAR_CONFIG_PATH}")
        print(f"  Timezone: {config.timezone}")
        print(f"  Trading hours: {config.start_hour:02d}:{config.start_minute:02d} - "
              f"{config.end_hour:02d}:{config.end_minute:02d}")
        print(f"  Colombian holidays: {len(config.holidays_colombia)}")
        print(f"  US holidays: {len(config.holidays_usa)}")
        print(f"  Bars per session: {config.bars_per_session}")
    except ConfigLoadError as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print()

    # Create filter
    filter_obj = get_trading_hours_filter()
    print(f"Filter: {filter_obj}\n")

    # Test is_trading_time
    test_times = [
        (datetime(2025, 1, 15, 15, 30, tzinfo=pytz.UTC), "Wed 10:30 Bogota"),
        (datetime(2025, 1, 18, 15, 30, tzinfo=pytz.UTC), "Sat 10:30 Bogota"),
        (datetime(2025, 1, 1, 15, 30, tzinfo=pytz.UTC), "New Year's Day"),
        (datetime(2025, 1, 6, 15, 30, tzinfo=pytz.UTC), "Epiphany (CO holiday)"),
        (datetime(2025, 7, 4, 15, 30, tzinfo=pytz.UTC), "July 4th (US holiday)"),
        (datetime(2025, 1, 15, 5, 30, tzinfo=pytz.UTC), "Wed 00:30 Bogota (before market)"),
        (datetime(2025, 1, 15, 23, 30, tzinfo=pytz.UTC), "Wed 18:30 Bogota (after market)"),
    ]

    print("Testing is_trading_time():")
    print(f"{'Timestamp (UTC)':<35} {'Description':<30} {'Valid?'}")
    print("-" * 75)

    for dt, desc in test_times:
        is_valid = filter_obj.is_trading_time(dt)
        status = "Yes" if is_valid else "No"
        print(f"{str(dt):<35} {desc:<30} {status}")

    print("\n")

    # Test TradingSession
    session = filter_obj.get_session()
    print(f"Trading Session: {session.start_time} - {session.end_time} ({session.timezone})")

    # Test Factory
    print("\n" + "-" * 70)
    print("Testing Factory Pattern...")
    print("-" * 70 + "\n")

    factory = TradingHoursFilterFactory()
    col_filter = factory.create(MarketType.COLOMBIA)
    print(f"Colombia filter: {col_filter}")

    # Test DataFrame filtering
    print("\n" + "-" * 70)
    print("Testing DataFrame filtering...")
    print("-" * 70 + "\n")

    # Create sample DataFrame
    dates = pd.date_range(
        start="2025-01-01 00:00:00",
        end="2025-01-31 23:59:59",
        freq="5min",
        tz="UTC"
    )
    sample_df = pd.DataFrame({
        "timestamp": dates,
        "price": np.random.randn(len(dates)) * 10 + 4500,
    })

    print(f"Sample DataFrame: {len(sample_df):,} rows")
    print(f"Date range: {sample_df['timestamp'].min()} to {sample_df['timestamp'].max()}")

    # Filter
    filtered_df = filter_obj.filter(sample_df, time_col="timestamp")

    print(f"\nFiltered DataFrame: {len(filtered_df):,} rows")
    retention = len(filtered_df) / len(sample_df) * 100
    print(f"Retention rate: {retention:.1f}%")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70 + "\n")
