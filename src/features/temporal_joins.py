"""
Temporal Joins Module for USD/COP RL Trading System.
=====================================================

This module provides utilities for performing point-in-time correct temporal joins
between price data and macro indicators. It ensures no lookahead bias in feature
construction by using backward-looking joins.

Contract ID: CTR-TEMPORAL-001
Audit Ref: P0-06

Key Features:
- Point-in-time correct joins using pd.merge_asof
- Lookahead validation to prevent data leakage
- Support for multiple macro data sources
- Comprehensive logging and statistics

Usage:
    from src.features.temporal_joins import (
        merge_price_with_macro,
        validate_no_lookahead,
        join_multiple_sources,
        TemporalJoinConfig,
    )

    # Simple merge
    merged_df = merge_price_with_macro(
        price_df=ohlcv_data,
        macro_df=macro_data,
        price_time_col="timestamp",
        macro_time_col="date",
        tolerance="3D"
    )

    # Validate no lookahead
    is_valid = validate_no_lookahead(merged_df, "timestamp", "macro_date")

    # Multiple sources
    config = TemporalJoinConfig.default()
    merged_all = join_multiple_sources(
        price_df=ohlcv_data,
        macro_dfs={"dxy": dxy_df, "vix": vix_df, "embi": embi_df},
        config=config
    )

Author: Trading Team
Version: 1.1.0
Date: 2026-01-17
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Final, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

# Import centralized constants from SSOT
from src.core.constants import (
    CTR_TEMPORAL_001,
    DEFAULT_MERGE_TOLERANCE,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS (Module-specific, derived from SSOT)
# =============================================================================

# Default tolerance values for temporal joins (derived from SSOT)
DEFAULT_DAILY_TOLERANCE: Final[str] = DEFAULT_MERGE_TOLERANCE  # 3 days for daily macro data
DEFAULT_INTRADAY_TOLERANCE: Final[str] = "1h"  # 1 hour for intraday data

# Valid join directions
VALID_JOIN_DIRECTIONS: Final[Tuple[str, ...]] = ("backward", "forward", "nearest")

# Standard time column names for auto-detection
STANDARD_TIME_COLUMNS: Final[Tuple[str, ...]] = (
    "timestamp",
    "date",
    "time",
    "datetime",
)

# Valid fill methods for missing data
# NOTE: bfill is PROHIBITED - it creates lookahead bias by using future data
VALID_FILL_METHODS: Final[Tuple[str, ...]] = ("ffill", "interpolate")

# Lookahead violation sample size for logging
LOOKAHEAD_VIOLATION_SAMPLE_SIZE: Final[int] = 5

# Contract ID reference (imported from SSOT)
CONTRACT_ID: Final[str] = CTR_TEMPORAL_001


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================


class TemporalJoinError(ValueError):
    """Base exception for temporal join operations.

    Contract ID: CTR-TEMPORAL-001

    Attributes:
        message: Human-readable error description
        context: Optional dictionary with additional context
    """

    def __init__(self, message: str, context: Optional[Dict] = None):
        self.message = message
        self.context = context or {}
        super().__init__(message)


class MissingColumnError(TemporalJoinError):
    """Raised when a required column is missing from a DataFrame."""
    pass


class DatetimeConversionError(TemporalJoinError):
    """Raised when datetime conversion fails."""
    pass


class LookaheadBiasError(TemporalJoinError):
    """Raised when lookahead bias is detected in joined data."""
    pass


class InvalidJoinConfigError(TemporalJoinError):
    """Raised when join configuration is invalid."""
    pass


class EmptyDataFrameError(TemporalJoinError):
    """Raised when an empty DataFrame is provided where data is required."""
    pass


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class TemporalJoinConfig:
    """
    Configuration for temporal joins between price and macro data.

    This is an immutable dataclass that specifies how to perform
    point-in-time correct joins.

    Contract ID: CTR-TEMPORAL-001

    Attributes:
        tolerance: Maximum time difference for asof join (e.g., '3D', '1h')
        direction: Join direction ('backward', 'forward', 'nearest')
        price_time_col: Column name for price timestamp
        macro_time_col: Column name for macro timestamp
        allow_exact_matches: Whether to allow exact timestamp matches
        suffix_left: Suffix for left DataFrame columns on conflict
        suffix_right: Suffix for right DataFrame columns on conflict

    Example:
        >>> config = TemporalJoinConfig(
        ...     tolerance='3D',
        ...     direction='backward',
        ...     price_time_col='timestamp',
        ...     macro_time_col='date'
        ... )
        >>> config.tolerance
        '3D'

    Raises:
        InvalidJoinConfigError: If direction is not one of VALID_JOIN_DIRECTIONS
    """

    tolerance: str = DEFAULT_DAILY_TOLERANCE
    direction: str = "backward"
    price_time_col: str = "timestamp"
    macro_time_col: str = "date"
    allow_exact_matches: bool = True
    suffix_left: str = ""
    suffix_right: str = "_macro"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.direction not in VALID_JOIN_DIRECTIONS:
            raise InvalidJoinConfigError(
                f"Invalid direction '{self.direction}'. "
                f"Must be one of: {VALID_JOIN_DIRECTIONS}",
                context={"provided_direction": self.direction}
            )

    @classmethod
    def default(cls) -> TemporalJoinConfig:
        """
        Create default configuration for trading use case.

        The default uses backward joins with 3-day tolerance, which is
        appropriate for daily macro data joined to intraday price data.

        Returns:
            TemporalJoinConfig with default settings

        Example:
            >>> config = TemporalJoinConfig.default()
            >>> config.direction
            'backward'
        """
        return cls(
            tolerance=DEFAULT_DAILY_TOLERANCE,
            direction="backward",
            price_time_col="timestamp",
            macro_time_col="date",
            allow_exact_matches=True,
        )

    @classmethod
    def for_intraday(cls) -> TemporalJoinConfig:
        """
        Create configuration for intraday data joins.

        Uses 1-hour tolerance for high-frequency data.

        Returns:
            TemporalJoinConfig for intraday joins

        Example:
            >>> config = TemporalJoinConfig.for_intraday()
            >>> config.tolerance
            '1h'
        """
        return cls(
            tolerance=DEFAULT_INTRADAY_TOLERANCE,
            direction="backward",
            price_time_col="timestamp",
            macro_time_col="timestamp",
            allow_exact_matches=True,
        )

    @classmethod
    def for_daily_macro(cls) -> TemporalJoinConfig:
        """
        Create configuration for daily macro data.

        Uses 3-day tolerance to handle weekends and holidays.

        Returns:
            TemporalJoinConfig for daily macro data

        Example:
            >>> config = TemporalJoinConfig.for_daily_macro()
            >>> config.macro_time_col
            'date'
        """
        return cls(
            tolerance=DEFAULT_DAILY_TOLERANCE,
            direction="backward",
            price_time_col="timestamp",
            macro_time_col="date",
            allow_exact_matches=True,
        )

    def to_dict(self) -> Dict[str, Union[str, bool]]:
        """Convert configuration to dictionary."""
        return {
            "tolerance": self.tolerance,
            "direction": self.direction,
            "price_time_col": self.price_time_col,
            "macro_time_col": self.macro_time_col,
            "allow_exact_matches": self.allow_exact_matches,
            "suffix_left": self.suffix_left,
            "suffix_right": self.suffix_right,
        }


# =============================================================================
# JOIN STATISTICS
# =============================================================================


@dataclass(frozen=True)
class JoinStatistics:
    """
    Immutable statistics from a temporal join operation.

    Contract ID: CTR-TEMPORAL-001

    This is a frozen dataclass because join statistics represent
    a point-in-time snapshot and should not be modified after creation.

    Attributes:
        total_rows: Total rows in result
        matched_rows: Rows with successful macro match
        null_rows: Rows with null macro values (no match found)
        match_rate: Percentage of rows with matches (0.0 to 1.0)
        null_columns: Dictionary mapping column names to null counts
        source_name: Name of the macro source (if applicable)

    Example:
        >>> stats = JoinStatistics(
        ...     total_rows=100,
        ...     matched_rows=95,
        ...     null_rows=5,
        ...     match_rate=0.95,
        ...     null_columns={"dxy": 5},
        ...     source_name="dxy"
        ... )
        >>> str(stats)
        'JoinStatistics (dxy): total=100, matched=95, null=5, match_rate=95.00%'
    """

    total_rows: int
    matched_rows: int
    null_rows: int
    match_rate: float
    null_columns: Tuple[Tuple[str, int], ...] = ()  # Frozen-compatible tuple
    source_name: Optional[str] = None

    def __str__(self) -> str:
        """Human-readable string representation."""
        source_info = f" ({self.source_name})" if self.source_name else ""
        return (
            f"JoinStatistics{source_info}: "
            f"total={self.total_rows}, "
            f"matched={self.matched_rows}, "
            f"null={self.null_rows}, "
            f"match_rate={self.match_rate:.2%}"
        )

    def get_null_columns_dict(self) -> Dict[str, int]:
        """Convert null_columns tuple to dictionary for easier access."""
        return dict(self.null_columns)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


def _ensure_datetime(
    df: pd.DataFrame,
    time_col: str,
    df_name: str = "DataFrame"
) -> pd.DataFrame:
    """
    Ensure the time column is datetime type.

    Contract ID: CTR-TEMPORAL-001

    Args:
        df: DataFrame to process
        time_col: Name of the time column
        df_name: Name for error messages

    Returns:
        DataFrame with datetime column

    Raises:
        MissingColumnError: If column doesn't exist in DataFrame
        DatetimeConversionError: If datetime conversion fails

    Example:
        >>> df = pd.DataFrame({"timestamp": ["2024-01-01"]})
        >>> result = _ensure_datetime(df, "timestamp", "price_df")
        >>> pd.api.types.is_datetime64_any_dtype(result["timestamp"])
        True
    """
    if time_col not in df.columns:
        raise MissingColumnError(
            f"{df_name} missing required column: '{time_col}'",
            context={"df_name": df_name, "column": time_col, "available": list(df.columns)}
        )

    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        try:
            df[time_col] = pd.to_datetime(df[time_col], utc=True)
            logger.debug(f"Converted {df_name}.{time_col} to datetime")
        except Exception as e:
            raise DatetimeConversionError(
                f"Failed to convert {df_name}.{time_col} to datetime: {e}",
                context={"df_name": df_name, "column": time_col, "error": str(e)}
            )

    return df


def _compute_join_statistics(
    result_df: pd.DataFrame,
    original_rows: int,
    macro_columns: List[str],
    source_name: Optional[str] = None
) -> JoinStatistics:
    """
    Compute statistics for a join operation.

    Contract ID: CTR-TEMPORAL-001

    Args:
        result_df: Merged DataFrame
        original_rows: Number of rows before join
        macro_columns: List of macro column names
        source_name: Optional source identifier

    Returns:
        JoinStatistics with computed metrics (immutable)

    Example:
        >>> df = pd.DataFrame({"price": [100, 101], "dxy": [102.5, np.nan]})
        >>> stats = _compute_join_statistics(df, 2, ["dxy"], "dxy")
        >>> stats.match_rate
        0.5
    """
    total_rows = len(result_df)

    # Count nulls per column (build as list of tuples for frozen dataclass)
    null_columns_list: List[Tuple[str, int]] = []
    total_nulls = 0
    for col in macro_columns:
        if col in result_df.columns:
            null_count = int(result_df[col].isna().sum())
            null_columns_list.append((col, null_count))
            total_nulls = max(total_nulls, null_count)

    matched_rows = total_rows - total_nulls
    match_rate = matched_rows / total_rows if total_rows > 0 else 0.0

    return JoinStatistics(
        total_rows=total_rows,
        matched_rows=matched_rows,
        null_rows=total_nulls,
        match_rate=match_rate,
        null_columns=tuple(null_columns_list),
        source_name=source_name
    )


def merge_price_with_macro(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    price_time_col: str = "timestamp",
    macro_time_col: str = "date",
    tolerance: str = DEFAULT_DAILY_TOLERANCE,
    direction: str = "backward",
    suffixes: Tuple[str, str] = ("", "_macro"),
) -> pd.DataFrame:
    """
    Merge price data with macro data using point-in-time correct temporal joins.

    Contract ID: CTR-TEMPORAL-001

    This function performs a temporal join that ensures no lookahead bias by
    using pd.merge_asof with backward direction. For each price row, it finds
    the most recent macro row that occurred on or before the price timestamp.

    Args:
        price_df: DataFrame with price/OHLCV data. Must contain price_time_col.
        macro_df: DataFrame with macro indicator data. Must contain macro_time_col.
        price_time_col: Name of timestamp column in price_df. Default: 'timestamp'
        macro_time_col: Name of timestamp column in macro_df. Default: 'date'
        tolerance: Maximum time difference for join. Default: DEFAULT_DAILY_TOLERANCE
        direction: Join direction ('backward', 'forward', 'nearest'). Default: 'backward'
        suffixes: Tuple of suffixes for overlapping columns. Default: ('', '_macro')

    Returns:
        pd.DataFrame: Merged DataFrame with price and macro columns.
            - Preserves all price_df rows
            - Adds macro columns with suffix if overlapping
            - Null values where no macro match found within tolerance

    Raises:
        MissingColumnError: If required time columns are missing
        DatetimeConversionError: If datetime conversion fails
        TemporalJoinError: If merge_asof fails for any reason

    Example:
        >>> price_df = pd.DataFrame({
        ...     'timestamp': pd.to_datetime(['2024-01-02 10:00', '2024-01-02 11:00']),
        ...     'close': [4000.0, 4010.0]
        ... })
        >>> macro_df = pd.DataFrame({
        ...     'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
        ...     'dxy': [102.5, 102.8]
        ... })
        >>> merged = merge_price_with_macro(price_df, macro_df)
        >>> merged['dxy'].tolist()
        [102.8, 102.8]  # Both get Jan 2nd value (backward join)

    Notes:
        - Both DataFrames are sorted by their time columns before merging
        - The direction='backward' ensures no future data is used (no lookahead)
        - Use validate_no_lookahead() to verify the join is correct
    """
    logger.debug(
        f"Merging price data ({len(price_df)} rows) with "
        f"macro data ({len(macro_df)} rows)"
    )

    # Ensure datetime types
    price_df = _ensure_datetime(price_df, price_time_col, "price_df")
    macro_df = _ensure_datetime(macro_df, macro_time_col, "macro_df")

    # Sort both dataframes by time (required for merge_asof)
    price_sorted = price_df.sort_values(price_time_col).reset_index(drop=True)
    macro_sorted = macro_df.sort_values(macro_time_col).reset_index(drop=True)

    # Get macro columns for statistics
    macro_columns = [
        col for col in macro_df.columns
        if col != macro_time_col
    ]

    # Parse tolerance
    tolerance_td = pd.Timedelta(tolerance)

    # Perform merge_asof
    try:
        result = pd.merge_asof(
            price_sorted,
            macro_sorted,
            left_on=price_time_col,
            right_on=macro_time_col,
            direction=direction,
            tolerance=tolerance_td,
            suffixes=suffixes,
        )
    except Exception as e:
        logger.error(f"merge_asof failed: {e}")
        raise TemporalJoinError(
            f"Temporal join failed: {e}",
            context={
                "price_rows": len(price_df),
                "macro_rows": len(macro_df),
                "tolerance": tolerance,
                "direction": direction,
                "error": str(e),
            }
        )

    # Compute and log statistics
    stats = _compute_join_statistics(
        result,
        len(price_df),
        macro_columns
    )

    logger.info(
        f"Temporal join complete: {stats.total_rows} rows, "
        f"{stats.match_rate:.1%} match rate, "
        f"{stats.null_rows} rows with null macro values"
    )

    if stats.null_columns:
        for col, null_count in stats.null_columns:
            if null_count > 0:
                null_pct = null_count / stats.total_rows * 100
                logger.warning(
                    f"Column '{col}' has {null_count} nulls ({null_pct:.1f}%)"
                )

    return result


def validate_no_lookahead(
    df: pd.DataFrame,
    price_time_col: str = "timestamp",
    macro_time_col: str = "date",
    raise_on_violation: bool = False,
) -> bool:
    """
    Validate that no lookahead bias exists in merged DataFrame.

    Contract ID: CTR-TEMPORAL-001

    This function checks that for every row, the macro timestamp is less than
    or equal to the price timestamp. If any violations are found, it logs
    detailed error messages.

    Args:
        df: Merged DataFrame from merge_price_with_macro()
        price_time_col: Name of price timestamp column
        macro_time_col: Name of macro timestamp column
        raise_on_violation: If True, raise LookaheadBiasError on detection

    Returns:
        bool: True if no lookahead bias detected, False otherwise

    Raises:
        MissingColumnError: If required columns are missing
        LookaheadBiasError: If raise_on_violation=True and bias detected

    Example:
        >>> merged_df = merge_price_with_macro(price_df, macro_df)
        >>> is_valid = validate_no_lookahead(merged_df, 'timestamp', 'date')
        >>> if not is_valid:
        ...     raise LookaheadBiasError("Lookahead bias detected!")

    Notes:
        - Null macro timestamps are skipped (no match = no lookahead)
        - Use this function after every temporal join to ensure data integrity
        - Critical for preventing data leakage in ML training
    """
    # Validate columns exist
    if price_time_col not in df.columns:
        raise MissingColumnError(
            f"Missing price time column: '{price_time_col}'",
            context={"column": price_time_col, "available": list(df.columns)}
        )
    if macro_time_col not in df.columns:
        raise MissingColumnError(
            f"Missing macro time column: '{macro_time_col}'",
            context={"column": macro_time_col, "available": list(df.columns)}
        )

    # Skip rows where macro timestamp is null (no match)
    valid_mask = df[macro_time_col].notna()
    valid_df = df[valid_mask]

    if len(valid_df) == 0:
        logger.warning("No valid macro matches to validate")
        return True

    # Ensure datetime types for comparison
    price_times = pd.to_datetime(valid_df[price_time_col])
    macro_times = pd.to_datetime(valid_df[macro_time_col])

    # Check for lookahead: macro_time should be <= price_time
    lookahead_mask = macro_times > price_times
    lookahead_count = lookahead_mask.sum()

    if lookahead_count > 0:
        # Log detailed violations
        violations = valid_df[lookahead_mask]
        logger.error(
            f"LOOKAHEAD BIAS DETECTED: {lookahead_count} violations found!"
        )

        # Log first few violations (use constant for sample size)
        sample_size = min(LOOKAHEAD_VIOLATION_SAMPLE_SIZE, len(violations))
        for idx, row in violations.head(sample_size).iterrows():
            logger.error(
                f"  Row {idx}: price_time={row[price_time_col]}, "
                f"macro_time={row[macro_time_col]} "
                f"(macro is {(row[macro_time_col] - row[price_time_col]).total_seconds() / 3600:.1f}h ahead)"
            )

        if len(violations) > sample_size:
            logger.error(f"  ... and {len(violations) - sample_size} more violations")

        if raise_on_violation:
            raise LookaheadBiasError(
                f"Lookahead bias detected: {lookahead_count} violations",
                context={
                    "violation_count": int(lookahead_count),
                    "total_rows": len(valid_df),
                    "sample_violations": violations.head(sample_size).to_dict(),
                }
            )

        return False

    logger.info(
        f"No lookahead bias detected in {len(valid_df)} matched rows"
    )
    return True


def join_multiple_sources(
    price_df: pd.DataFrame,
    macro_dfs: Dict[str, pd.DataFrame],
    config: Optional[TemporalJoinConfig] = None,
) -> pd.DataFrame:
    """
    Join price data with multiple macro data sources sequentially.

    Contract ID: CTR-TEMPORAL-001

    This function performs temporal joins with multiple macro sources,
    tracking which source each macro column came from. It ensures
    point-in-time correctness across all joins.

    Args:
        price_df: DataFrame with price/OHLCV data
        macro_dfs: Dictionary mapping source names to DataFrames
            Example: {"dxy": dxy_df, "vix": vix_df, "embi": embi_df}
        config: TemporalJoinConfig or None for defaults

    Returns:
        pd.DataFrame: Merged DataFrame with all macro sources.
            - Columns from each source are prefixed with source name
            - Metadata about source origin is preserved in column names

    Raises:
        EmptyDataFrameError: If price_df is empty or macro_dfs is empty

    Example:
        >>> config = TemporalJoinConfig.default()
        >>> merged = join_multiple_sources(
        ...     price_df=ohlcv_data,
        ...     macro_dfs={
        ...         "dxy": dxy_df,      # DXY index data
        ...         "vix": vix_df,      # VIX volatility data
        ...         "embi": embi_df,    # EMBI spread data
        ...     },
        ...     config=config
        ... )
        >>> # Result contains: dxy_value, vix_value, embi_value, etc.

    Notes:
        - Each source is joined sequentially to preserve independence
        - Source names are used as prefixes for non-time columns
        - All joins use the same tolerance and direction from config
    """
    if len(price_df) == 0:
        raise EmptyDataFrameError(
            "price_df cannot be empty",
            context={"shape": price_df.shape}
        )

    if not macro_dfs:
        raise EmptyDataFrameError(
            "macro_dfs cannot be empty",
            context={"provided_sources": list(macro_dfs.keys()) if macro_dfs else []}
        )

    # Use default config if not provided
    if config is None:
        config = TemporalJoinConfig.default()

    logger.info(
        f"Joining price data with {len(macro_dfs)} macro sources: "
        f"{list(macro_dfs.keys())}"
    )

    # Start with price data
    result = price_df.copy()
    all_stats: List[JoinStatistics] = []
    source_columns: Dict[str, List[str]] = {}

    # Join each source sequentially
    for source_name, macro_df in macro_dfs.items():
        logger.debug(f"Joining source: {source_name}")

        # Get non-time columns from this source
        macro_time_col = config.macro_time_col
        if macro_time_col not in macro_df.columns:
            # Try standard alternatives from SSOT constant
            for alt_col in STANDARD_TIME_COLUMNS:
                if alt_col in macro_df.columns:
                    macro_time_col = alt_col
                    break
            else:
                logger.warning(
                    f"Source '{source_name}' missing time column, skipping. "
                    f"Expected one of: {STANDARD_TIME_COLUMNS}"
                )
                continue

        # Rename macro columns with source prefix (except time column)
        macro_renamed = macro_df.copy()
        column_mapping = {}
        new_columns = []

        for col in macro_df.columns:
            if col != macro_time_col:
                new_col_name = f"{source_name}_{col}"
                column_mapping[col] = new_col_name
                new_columns.append(new_col_name)

        macro_renamed = macro_renamed.rename(columns=column_mapping)

        # Perform temporal join
        try:
            result = merge_price_with_macro(
                price_df=result,
                macro_df=macro_renamed,
                price_time_col=config.price_time_col,
                macro_time_col=macro_time_col,
                tolerance=config.tolerance,
                direction=config.direction,
                suffixes=(config.suffix_left, f"_{source_name}"),
            )

            # Track source columns
            source_columns[source_name] = new_columns

            # Compute statistics for this source
            stats = _compute_join_statistics(
                result,
                len(price_df),
                new_columns,
                source_name=source_name
            )
            all_stats.append(stats)

            logger.info(f"  {stats}")

        except Exception as e:
            logger.error(f"Failed to join source '{source_name}': {e}")
            continue

    # Log summary
    total_sources = len(macro_dfs)
    successful_sources = len(all_stats)
    logger.info(
        f"Multi-source join complete: "
        f"{successful_sources}/{total_sources} sources joined successfully"
    )

    # Add metadata as DataFrame attributes (optional, for tracking)
    result.attrs["temporal_join_sources"] = source_columns
    result.attrs["temporal_join_config"] = config.to_dict()

    return result


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_join_statistics(
    merged_df: pd.DataFrame,
    macro_columns: List[str],
    source_name: Optional[str] = None
) -> JoinStatistics:
    """
    Compute join statistics for an already-merged DataFrame.

    Contract ID: CTR-TEMPORAL-001

    Args:
        merged_df: DataFrame after temporal join
        macro_columns: List of macro column names to analyze
        source_name: Optional source identifier

    Returns:
        JoinStatistics with computed metrics (immutable)

    Example:
        >>> merged = merge_price_with_macro(price_df, macro_df)
        >>> stats = get_join_statistics(merged, ['dxy', 'vix'], 'macro_data')
        >>> print(f"Match rate: {stats.match_rate:.1%}")
    """
    return _compute_join_statistics(
        merged_df,
        len(merged_df),
        macro_columns,
        source_name
    )


class FillStrategy(Protocol):
    """
    Protocol for fill strategy implementations (Strategy Pattern).

    Contract ID: CTR-TEMPORAL-001

    This protocol defines the interface for different null-filling strategies
    used after temporal joins. Implementations should handle a single column.

    Example:
        >>> class CustomFillStrategy:
        ...     def fill(self, series: pd.Series, limit: Optional[int]) -> pd.Series:
        ...         return series.fillna(0)
    """

    def fill(self, series: pd.Series, limit: Optional[int] = None) -> pd.Series:
        """Fill null values in a series."""
        ...


class ForwardFillStrategy:
    """Forward fill strategy - propagates last valid observation forward."""

    def fill(self, series: pd.Series, limit: Optional[int] = None) -> pd.Series:
        return series.ffill(limit=limit)


# NOTE: BackwardFillStrategy is INTENTIONALLY OMITTED
# Backward fill (bfill) is PROHIBITED because it creates lookahead bias
# by using future data to fill past values, violating temporal causality.
# See: CTR-TEMPORAL-001 compliance requirements


class InterpolateFillStrategy:
    """Linear interpolation fill strategy."""

    def fill(self, series: pd.Series, limit: Optional[int] = None) -> pd.Series:
        return series.interpolate(method="linear", limit=limit)


# Strategy registry for string-based lookup
# NOTE: bfill is INTENTIONALLY EXCLUDED - it violates temporal causality
_FILL_STRATEGIES: Dict[str, FillStrategy] = {
    "ffill": ForwardFillStrategy(),
    "interpolate": InterpolateFillStrategy(),
}


def fill_missing_macro(
    df: pd.DataFrame,
    macro_columns: List[str],
    method: str = "ffill",
    limit: Optional[int] = None,
    strategy: Optional[FillStrategy] = None,
) -> pd.DataFrame:
    """
    Fill missing macro values after temporal join.

    Contract ID: CTR-TEMPORAL-001

    This function supports both string-based method selection (for simplicity)
    and Strategy pattern injection (for extensibility and testing).

    Args:
        df: DataFrame with potential null macro values
        macro_columns: Columns to fill
        method: Fill method name from VALID_FILL_METHODS ('ffill', 'interpolate')
            NOTE: 'bfill' is PROHIBITED - it violates temporal causality
        limit: Maximum number of consecutive fills
        strategy: Optional FillStrategy implementation (overrides method if provided)

    Returns:
        DataFrame with filled values

    Raises:
        InvalidJoinConfigError: If method is not in VALID_FILL_METHODS

    Example:
        >>> merged = merge_price_with_macro(price_df, macro_df)
        >>> filled = fill_missing_macro(merged, ['dxy', 'vix'], method='ffill')

        # Using custom strategy (dependency injection)
        >>> custom_strategy = ForwardFillStrategy()
        >>> filled = fill_missing_macro(merged, ['dxy'], strategy=custom_strategy)
    """
    result = df.copy()

    # Use provided strategy or look up by method name
    if strategy is None:
        if method not in VALID_FILL_METHODS:
            raise InvalidJoinConfigError(
                f"Unknown fill method: '{method}'. Must be one of: {VALID_FILL_METHODS}",
                context={"provided_method": method}
            )
        strategy = _FILL_STRATEGIES[method]

    for col in macro_columns:
        if col not in result.columns:
            logger.warning(f"Column '{col}' not found, skipping fill")
            continue

        null_before = result[col].isna().sum()
        result[col] = strategy.fill(result[col], limit=limit)
        null_after = result[col].isna().sum()
        filled = null_before - null_after

        if filled > 0:
            logger.debug(f"Filled {filled} nulls in '{col}' using {type(strategy).__name__}")

    return result


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Main functions
    "merge_price_with_macro",
    "validate_no_lookahead",
    "join_multiple_sources",
    # Configuration
    "TemporalJoinConfig",
    # Statistics
    "JoinStatistics",
    "get_join_statistics",
    # Utilities
    "fill_missing_macro",
    # Strategy pattern interfaces
    "FillStrategy",
    "ForwardFillStrategy",
    "InterpolateFillStrategy",
    # Constants (SSOT)
    "DEFAULT_DAILY_TOLERANCE",
    "DEFAULT_INTRADAY_TOLERANCE",
    "VALID_JOIN_DIRECTIONS",
    "STANDARD_TIME_COLUMNS",
    "VALID_FILL_METHODS",
    # Exceptions
    "TemporalJoinError",
    "MissingColumnError",
    "DatetimeConversionError",
    "LookaheadBiasError",
    "InvalidJoinConfigError",
    "EmptyDataFrameError",
]
