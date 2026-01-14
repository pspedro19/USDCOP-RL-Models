"""
Centralized Gap Handler (Phase 14)
==================================

Unified gap handling for both training and inference.
Single source of truth for how to handle missing data and gaps.

Strategy: forward_then_zero
1. Forward fill for gaps <= max_gap_minutes
2. Fill with 0.0 for larger gaps (safe default)
3. Log all gaps for monitoring

This ensures IDENTICAL behavior between training and inference.

Author: Trading Team
Version: 1.0.0
Created: 2025-01-14
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GapConfig:
    """
    Configuration for gap handling - MUST match training.

    CRITICAL: These values should be identical in training and inference
    to ensure feature parity.
    """
    warmup_bars: int = 14  # Minimum bars before valid features
    max_gap_minutes: int = 30  # Gaps larger than this trigger special handling
    fill_strategy: str = "forward_then_zero"  # forward_fill, zero, interpolate
    log_gaps: bool = True  # Log detected gaps
    expected_frequency_minutes: int = 5  # Expected bar frequency (USDCOP = 5min)


@dataclass
class GapStatistics:
    """Statistics about detected gaps."""
    total_gaps: int = 0
    filled_forward: int = 0
    filled_zero: int = 0
    largest_gap_minutes: float = 0.0
    gap_timestamps: List[str] = field(default_factory=list)


class GapHandler:
    """
    Unified gap handler for training and inference.

    Strategy: forward_then_zero
    1. Forward fill for gaps <= max_gap_minutes
    2. Fill with 0.0 for larger gaps (safe default)
    3. Log all gaps for monitoring

    This ensures IDENTICAL behavior between training and inference.

    Usage:
        handler = GapHandler()

        # For DataFrame processing
        df = handler.handle_gaps(df)

        # For validation
        has_issues = handler.validate_data(df)

        # Get statistics
        stats = handler.get_stats()
    """

    def __init__(self, config: Optional[GapConfig] = None):
        """
        Initialize the gap handler.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or GapConfig()
        self._stats = GapStatistics()

        logger.info(
            f"GapHandler initialized: warmup={self.config.warmup_bars}, "
            f"max_gap={self.config.max_gap_minutes}min, "
            f"strategy={self.config.fill_strategy}"
        )

    def handle_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle gaps in OHLCV data.

        Args:
            df: DataFrame with datetime index and OHLCV columns

        Returns:
            DataFrame with gaps handled consistently
        """
        if df.empty:
            return df

        df = df.copy()

        # Detect gaps
        gaps = self._detect_gaps(df)

        # Log gaps
        if self.config.log_gaps and len(gaps) > 0:
            self._log_gaps(gaps)

        # Fill gaps based on strategy
        if self.config.fill_strategy == "forward_then_zero":
            df = self._forward_then_zero(df, gaps)
        elif self.config.fill_strategy == "zero":
            df = df.fillna(0.0)
        elif self.config.fill_strategy == "forward_fill":
            df = df.ffill()
        elif self.config.fill_strategy == "interpolate":
            df = df.interpolate(method='linear', limit_direction='forward')
        else:
            raise ValueError(f"Unknown fill strategy: {self.config.fill_strategy}")

        return df

    def _detect_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect gaps in the data.

        Args:
            df: DataFrame with datetime index

        Returns:
            DataFrame with gap information (start, duration_minutes)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime
            if 'time' in df.columns:
                df = df.set_index('time')
            elif 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                logger.warning("Cannot detect gaps: no datetime index")
                return pd.DataFrame()

        if len(df) < 2:
            return pd.DataFrame()

        # Calculate time differences
        time_diff = df.index.to_series().diff()

        # Expected frequency
        expected_freq = pd.Timedelta(minutes=self.config.expected_frequency_minutes)

        # Find gaps (time diff > expected * 1.5 for some tolerance)
        gap_mask = time_diff > expected_freq * 1.5

        if not gap_mask.any():
            return pd.DataFrame()

        gaps = pd.DataFrame({
            'start': df.index[gap_mask],
            'duration_minutes': (time_diff[gap_mask].dt.total_seconds() / 60).values
        })

        return gaps

    def _forward_then_zero(
        self,
        df: pd.DataFrame,
        gaps: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Fill strategy: forward fill for small gaps, zero for large gaps.

        Args:
            df: DataFrame to process
            gaps: DataFrame with gap information

        Returns:
            DataFrame with gaps filled
        """
        # First pass: forward fill everything
        df = df.ffill()

        # Second pass: zero out large gaps
        for _, gap in gaps.iterrows():
            if gap['duration_minutes'] > self.config.max_gap_minutes:
                # Find rows in the gap and fill calculated features with zero
                gap_start = gap['start']
                gap_duration = pd.Timedelta(minutes=gap['duration_minutes'])

                # Only zero out calculated features, not OHLCV
                feature_cols = [
                    c for c in df.columns
                    if c not in ['open', 'high', 'low', 'close', 'volume', '_is_warmup']
                ]

                if feature_cols:
                    mask = (df.index >= gap_start) & (df.index < gap_start + gap_duration)
                    df.loc[mask, feature_cols] = 0.0

                self._stats.filled_zero += 1
                logger.warning(
                    f"Large gap ({gap['duration_minutes']:.0f}min) at {gap_start}: "
                    f"zeroing calculated features"
                )
            else:
                self._stats.filled_forward += 1

        # Final pass: fill any remaining NaN with 0
        df = df.fillna(0.0)

        return df

    def _log_gaps(self, gaps: pd.DataFrame) -> None:
        """Log detected gaps for monitoring."""
        self._stats.total_gaps += len(gaps)

        for _, gap in gaps.iterrows():
            # Track largest gap
            if gap['duration_minutes'] > self._stats.largest_gap_minutes:
                self._stats.largest_gap_minutes = gap['duration_minutes']

            # Track gap timestamps (limit to 100)
            if len(self._stats.gap_timestamps) < 100:
                self._stats.gap_timestamps.append(str(gap['start']))

            # Log based on severity
            severity = "WARNING" if gap['duration_minutes'] > self.config.max_gap_minutes else "INFO"
            log_func = logger.warning if severity == "WARNING" else logger.info

            log_func(
                f"Gap detected: {gap['start']} - {gap['duration_minutes']:.0f} minutes"
            )

    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality before processing.

        Args:
            df: DataFrame to validate

        Returns:
            Dict with validation results
        """
        issues = []

        # Check for datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' not in df.columns and 'timestamp' not in df.columns:
                issues.append("No datetime index or time/timestamp column")

        # Check for required OHLCV columns
        required = ['open', 'high', 'low', 'close']
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"Missing OHLCV columns: {missing}")

        # Check for sufficient data
        if len(df) < self.config.warmup_bars:
            issues.append(
                f"Insufficient data: {len(df)} rows < warmup {self.config.warmup_bars}"
            )

        # Check for NaN in OHLCV
        if 'close' in df.columns:
            nan_pct = df['close'].isna().mean() * 100
            if nan_pct > 0:
                issues.append(f"Close column has {nan_pct:.1f}% NaN values")

        # Detect gaps
        gaps = self._detect_gaps(df)
        large_gaps = gaps[gaps['duration_minutes'] > self.config.max_gap_minutes]
        if len(large_gaps) > 0:
            issues.append(
                f"{len(large_gaps)} gaps > {self.config.max_gap_minutes}min detected"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "rows": len(df),
            "gaps_detected": len(gaps),
            "large_gaps": len(large_gaps),
        }

    def mark_warmup_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark the warmup period in the DataFrame.

        Args:
            df: DataFrame to mark

        Returns:
            DataFrame with '_is_warmup' column
        """
        df = df.copy()
        df['_is_warmup'] = False

        if len(df) > self.config.warmup_bars:
            df.iloc[:self.config.warmup_bars, df.columns.get_loc('_is_warmup')] = True

        return df

    def get_stats(self) -> Dict[str, Any]:
        """Get gap handling statistics."""
        return {
            "total_gaps": self._stats.total_gaps,
            "filled_forward": self._stats.filled_forward,
            "filled_zero": self._stats.filled_zero,
            "largest_gap_minutes": self._stats.largest_gap_minutes,
            "recent_gap_timestamps": self._stats.gap_timestamps[-10:],
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = GapStatistics()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_gap_handler: Optional[GapHandler] = None


def get_gap_handler(config: Optional[GapConfig] = None) -> GapHandler:
    """Get or create the global gap handler instance."""
    global _gap_handler
    if _gap_handler is None:
        _gap_handler = GapHandler(config)
    return _gap_handler


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def handle_gaps(df: pd.DataFrame, config: Optional[GapConfig] = None) -> pd.DataFrame:
    """
    Convenience function to handle gaps in a DataFrame.

    Args:
        df: DataFrame with OHLCV data
        config: Optional configuration

    Returns:
        DataFrame with gaps handled
    """
    handler = GapHandler(config)
    return handler.handle_gaps(df)


def validate_ohlcv_data(
    df: pd.DataFrame,
    config: Optional[GapConfig] = None
) -> Dict[str, Any]:
    """
    Validate OHLCV data for gaps and quality issues.

    Args:
        df: DataFrame to validate
        config: Optional configuration

    Returns:
        Validation results dictionary
    """
    handler = GapHandler(config)
    return handler.validate_data(df)
