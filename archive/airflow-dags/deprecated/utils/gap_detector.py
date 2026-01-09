"""
USDCOP Trading System - Intelligent Gap Detection
=================================================

Provides sophisticated gap detection for incremental data updates with:
- Business hours awareness (L-V 8:00-12:55 COT)
- 5-minute interval gap detection
- Holiday handling for Colombian market
- Efficient date range calculations for API calls

GAP DETECTION FEATURES:
- Missing period identification in business hours
- Optimal batch sizing for API requests
- Gap severity assessment
- Incremental update range calculation
- Market context awareness

BUSINESS RULES:
- Colombian market hours: 8:00 AM - 12:55 PM COT (Monday-Friday)
- Expected intervals: 5 minutes
- Colombian holidays are excluded from gap detection
- WeekendS are excluded from business hour calculations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum
import bisect
import holidays

from .datetime_handler import UnifiedDatetimeHandler
from .db_manager import DatabaseManager

logger = logging.getLogger(__name__)

class GapSeverity(Enum):
    """Enumeration of gap severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class GapType(Enum):
    """Enumeration of gap types"""
    MISSING_DATA = "missing_data"
    DELAYED_DATA = "delayed_data"
    PARTIAL_DATA = "partial_data"
    BUSINESS_HOUR_GAP = "business_hour_gap"
    WEEKEND_GAP = "weekend_gap"
    HOLIDAY_GAP = "holiday_gap"

@dataclass
class DataGap:
    """Data class representing a detected gap"""
    gap_id: str
    gap_type: GapType
    severity: GapSeverity
    start_time: datetime
    end_time: datetime
    expected_points: int
    missing_points: int
    duration_minutes: int
    business_hours_only: bool
    context: Dict[str, Any]

@dataclass
class IncrementalRange:
    """Data class for incremental update ranges"""
    range_id: str
    start_time: datetime
    end_time: datetime
    expected_points: int
    priority: int
    batch_size: int
    api_calls_required: int
    estimated_duration_minutes: int

class GapDetector:
    """
    Intelligent gap detection system for USDCOP trading data.
    Provides business-hours-aware gap detection and incremental update planning.
    """

    def __init__(self,
                 interval_minutes: int = 5,
                 tolerance_minutes: int = 2,
                 max_api_calls_per_batch: int = 100,
                 db_manager: DatabaseManager = None):
        """
        Initialize gap detector with configuration.

        Args:
            interval_minutes: Expected data interval in minutes
            tolerance_minutes: Tolerance for considering data as missing
            max_api_calls_per_batch: Maximum API calls per batch for incremental updates
            db_manager: Database manager for historical data queries
        """
        self.interval_minutes = interval_minutes
        self.tolerance_minutes = tolerance_minutes
        self.max_api_calls_per_batch = max_api_calls_per_batch
        self.db_manager = db_manager

        # Initialize datetime handler for timezone operations
        self.datetime_handler = UnifiedDatetimeHandler()

        # Business hours configuration
        self.market_open_hour = 8   # 8:00 AM COT
        self.market_close_hour = 12  # 12:00 PM COT (closes at 12:55, but we use 12 for simplicity)
        self.market_close_minute = 55  # Market closes at 12:55 PM

        # Gap severity thresholds (in minutes)
        self.severity_thresholds = {
            GapSeverity.LOW: 15,      # 15 minutes or less
            GapSeverity.MEDIUM: 60,   # 1 hour or less
            GapSeverity.HIGH: 240,    # 4 hours or less
            GapSeverity.CRITICAL: float('inf')  # More than 4 hours
        }

        logger.info(f"GapDetector initialized - Interval: {interval_minutes}min, Tolerance: {tolerance_minutes}min")

    def detect_gaps(self,
                   timestamps: Union[pd.Series, List[datetime]],
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   business_hours_only: bool = True) -> List[DataGap]:
        """
        Detect gaps in timestamp data.

        Args:
            timestamps: Series or list of timestamps to analyze
            start_time: Start of analysis period (optional)
            end_time: End of analysis period (optional)
            business_hours_only: Only detect gaps during business hours

        Returns:
            List of detected gaps
        """
        try:
            # Convert to pandas Series if needed
            if isinstance(timestamps, list):
                timestamps = pd.Series(timestamps)

            # Ensure timezone awareness
            timestamps = self.datetime_handler.ensure_timezone_aware(timestamps)

            # Convert to COT timezone
            timestamps_cot = self.datetime_handler.convert_to_cot(timestamps)

            # Sort timestamps
            timestamps_sorted = timestamps_cot.sort_values().reset_index(drop=True)

            # Determine analysis period
            if start_time is None:
                start_time = timestamps_sorted.min()
            else:
                start_time = self.datetime_handler.convert_to_cot(
                    self.datetime_handler.ensure_timezone_aware(start_time)
                )

            if end_time is None:
                end_time = timestamps_sorted.max()
            else:
                end_time = self.datetime_handler.convert_to_cot(
                    self.datetime_handler.ensure_timezone_aware(end_time)
                )

            logger.info(f"Detecting gaps from {start_time} to {end_time}")

            # Generate expected timestamps
            expected_timestamps = self._generate_expected_timestamps(
                start_time, end_time, business_hours_only
            )

            if not expected_timestamps:
                logger.warning("No expected timestamps generated")
                return []

            # Find missing timestamps
            missing_timestamps = self._find_missing_timestamps(
                timestamps_sorted, expected_timestamps
            )

            # Group consecutive missing timestamps into gaps
            gaps = self._group_consecutive_gaps(missing_timestamps, business_hours_only)

            logger.info(f"✅ Detected {len(gaps)} gaps in data")
            return gaps

        except Exception as e:
            logger.error(f"❌ Error detecting gaps: {e}")
            return []

    def calculate_missing_periods(self,
                                 start_time: datetime,
                                 end_time: datetime,
                                 existing_data: Optional[pd.DataFrame] = None,
                                 business_hours_only: bool = True) -> List[Tuple[datetime, datetime]]:
        """
        Calculate missing periods that need to be fetched.

        Args:
            start_time: Start of period to check
            end_time: End of period to check
            existing_data: DataFrame with existing data (optional)
            business_hours_only: Only consider business hours

        Returns:
            List of (start, end) tuples for missing periods
        """
        try:
            # Ensure timezone awareness and convert to COT
            start_cot = self.datetime_handler.convert_to_cot(
                self.datetime_handler.ensure_timezone_aware(start_time)
            )
            end_cot = self.datetime_handler.convert_to_cot(
                self.datetime_handler.ensure_timezone_aware(end_time)
            )

            # Generate expected timestamps for the period
            expected_timestamps = self._generate_expected_timestamps(
                start_cot, end_cot, business_hours_only
            )

            # If no existing data provided, all expected timestamps are missing
            if existing_data is None or existing_data.empty:
                return self._group_consecutive_periods(expected_timestamps)

            # Extract timestamps from existing data
            timestamp_col = self._find_timestamp_column(existing_data)
            if timestamp_col is None:
                logger.warning("No timestamp column found in existing data")
                return self._group_consecutive_periods(expected_timestamps)

            existing_timestamps = self.datetime_handler.convert_to_cot(
                self.datetime_handler.ensure_timezone_aware(existing_data[timestamp_col])
            )

            # Find missing timestamps
            missing_timestamps = []
            existing_set = set(existing_timestamps.dt.floor(f'{self.interval_minutes}min'))

            for expected_ts in expected_timestamps:
                # Floor to interval boundary for comparison
                expected_floored = expected_ts.floor(f'{self.interval_minutes}min')
                if expected_floored not in existing_set:
                    missing_timestamps.append(expected_ts)

            # Group consecutive missing timestamps into periods
            missing_periods = self._group_consecutive_periods(missing_timestamps)

            logger.info(f"✅ Found {len(missing_periods)} missing periods")
            return missing_periods

        except Exception as e:
            logger.error(f"❌ Error calculating missing periods: {e}")
            return []

    def get_incremental_ranges(self,
                              missing_periods: List[Tuple[datetime, datetime]],
                              max_points_per_call: int = 5000,
                              priority_business_hours: bool = True) -> List[IncrementalRange]:
        """
        Generate optimized incremental update ranges.

        Args:
            missing_periods: List of missing period tuples
            max_points_per_call: Maximum data points per API call
            priority_business_hours: Prioritize business hours data

        Returns:
            List of incremental ranges optimized for API calls
        """
        try:
            if not missing_periods:
                return []

            ranges = []
            range_counter = 1

            for period_start, period_end in missing_periods:
                # Calculate expected points for this period
                expected_points = self._calculate_expected_points(
                    period_start, period_end, business_hours_only=True
                )

                if expected_points == 0:
                    continue

                # Determine if period needs to be split
                if expected_points <= max_points_per_call:
                    # Single range
                    api_calls = 1
                    batch_size = expected_points
                    priority = self._calculate_priority(period_start, period_end, priority_business_hours)

                    incremental_range = IncrementalRange(
                        range_id=f"range_{range_counter:03d}",
                        start_time=period_start,
                        end_time=period_end,
                        expected_points=expected_points,
                        priority=priority,
                        batch_size=batch_size,
                        api_calls_required=api_calls,
                        estimated_duration_minutes=self._estimate_duration(api_calls, expected_points)
                    )
                    ranges.append(incremental_range)
                    range_counter += 1

                else:
                    # Split into multiple ranges
                    sub_ranges = self._split_period_into_ranges(
                        period_start, period_end, max_points_per_call, range_counter, priority_business_hours
                    )
                    ranges.extend(sub_ranges)
                    range_counter += len(sub_ranges)

            # Sort by priority (higher priority first)
            ranges.sort(key=lambda r: r.priority, reverse=True)

            logger.info(f"✅ Generated {len(ranges)} incremental ranges")
            return ranges

        except Exception as e:
            logger.error(f"❌ Error generating incremental ranges: {e}")
            return []

    def analyze_gap_patterns(self,
                            gaps: List[DataGap],
                            lookback_days: int = 30) -> Dict[str, Any]:
        """
        Analyze gap patterns to identify systemic issues.

        Args:
            gaps: List of detected gaps
            lookback_days: Number of days to analyze

        Returns:
            Analysis results dictionary
        """
        try:
            if not gaps:
                return {"total_gaps": 0, "analysis": "No gaps detected"}

            analysis = {
                "total_gaps": len(gaps),
                "gap_types": {},
                "severity_distribution": {},
                "time_patterns": {},
                "duration_stats": {},
                "recommendations": []
            }

            # Analyze gap types
            for gap in gaps:
                gap_type = gap.gap_type.value
                severity = gap.severity.value

                analysis["gap_types"][gap_type] = analysis["gap_types"].get(gap_type, 0) + 1
                analysis["severity_distribution"][severity] = analysis["severity_distribution"].get(severity, 0) + 1

            # Analyze time patterns
            gap_hours = [gap.start_time.hour for gap in gaps]
            gap_days = [gap.start_time.weekday() for gap in gaps]

            analysis["time_patterns"] = {
                "most_common_hour": max(set(gap_hours), key=gap_hours.count) if gap_hours else None,
                "most_common_day": max(set(gap_days), key=gap_days.count) if gap_days else None,
                "hour_distribution": {h: gap_hours.count(h) for h in set(gap_hours)},
                "day_distribution": {d: gap_days.count(d) for d in set(gap_days)}
            }

            # Analyze duration statistics
            durations = [gap.duration_minutes for gap in gaps]
            if durations:
                analysis["duration_stats"] = {
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "total_missing_minutes": sum(durations)
                }

            # Generate recommendations
            analysis["recommendations"] = self._generate_gap_recommendations(analysis)

            logger.info(f"✅ Analyzed {len(gaps)} gaps")
            return analysis

        except Exception as e:
            logger.error(f"❌ Error analyzing gap patterns: {e}")
            return {"error": str(e)}

    def validate_data_completeness(self,
                                  df: pd.DataFrame,
                                  start_time: datetime,
                                  end_time: datetime,
                                  business_hours_only: bool = True) -> Dict[str, Any]:
        """
        Validate data completeness for a given period.

        Args:
            df: DataFrame with timestamp data
            start_time: Start of validation period
            end_time: End of validation period
            business_hours_only: Only validate business hours

        Returns:
            Validation results dictionary
        """
        try:
            # Find timestamp column
            timestamp_col = self._find_timestamp_column(df)
            if timestamp_col is None:
                return {"error": "No timestamp column found"}

            # Convert timestamps to COT
            timestamps = self.datetime_handler.convert_to_cot(
                self.datetime_handler.ensure_timezone_aware(df[timestamp_col])
            )

            # Filter to analysis period
            start_cot = self.datetime_handler.convert_to_cot(
                self.datetime_handler.ensure_timezone_aware(start_time)
            )
            end_cot = self.datetime_handler.convert_to_cot(
                self.datetime_handler.ensure_timezone_aware(end_time)
            )

            filtered_timestamps = timestamps[
                (timestamps >= start_cot) & (timestamps <= end_cot)
            ]

            # Generate expected timestamps
            expected_timestamps = self._generate_expected_timestamps(
                start_cot, end_cot, business_hours_only
            )

            # Calculate completeness
            expected_count = len(expected_timestamps)
            actual_count = len(filtered_timestamps)
            completeness_score = actual_count / expected_count if expected_count > 0 else 0

            # Detect gaps
            gaps = self.detect_gaps(filtered_timestamps, start_cot, end_cot, business_hours_only)

            validation_results = {
                "period": {
                    "start": start_cot.isoformat(),
                    "end": end_cot.isoformat(),
                    "business_hours_only": business_hours_only
                },
                "completeness": {
                    "score": round(completeness_score, 4),
                    "percentage": round(completeness_score * 100, 2),
                    "expected_points": expected_count,
                    "actual_points": actual_count,
                    "missing_points": expected_count - actual_count
                },
                "gaps": {
                    "total_gaps": len(gaps),
                    "gap_summary": self._summarize_gaps(gaps)
                },
                "quality_assessment": self._assess_data_quality(completeness_score, gaps)
            }

            logger.info(f"✅ Data completeness: {validation_results['completeness']['percentage']:.1f}%")
            return validation_results

        except Exception as e:
            logger.error(f"❌ Error validating data completeness: {e}")
            return {"error": str(e)}

    def _generate_expected_timestamps(self,
                                     start_time: datetime,
                                     end_time: datetime,
                                     business_hours_only: bool) -> List[datetime]:
        """Generate expected timestamps for the given period."""
        try:
            if business_hours_only:
                return self.datetime_handler.generate_expected_timestamps(
                    start_time, end_time,
                    interval_minutes=self.interval_minutes,
                    business_hours_only=True,
                    exclude_holidays=True
                )
            else:
                timestamps = []
                current = start_time
                while current <= end_time:
                    timestamps.append(current)
                    current += timedelta(minutes=self.interval_minutes)
                return timestamps

        except Exception as e:
            logger.error(f"Error generating expected timestamps: {e}")
            return []

    def _find_missing_timestamps(self,
                                actual_timestamps: pd.Series,
                                expected_timestamps: List[datetime]) -> List[datetime]:
        """Find missing timestamps by comparing actual vs expected."""
        # Convert to sets for efficient comparison
        # Floor timestamps to interval boundaries for comparison
        actual_set = set(
            ts.floor(f'{self.interval_minutes}min') for ts in actual_timestamps
        )

        missing = []
        for expected_ts in expected_timestamps:
            expected_floored = expected_ts.floor(f'{self.interval_minutes}min')
            if expected_floored not in actual_set:
                missing.append(expected_ts)

        return missing

    def _group_consecutive_gaps(self,
                               missing_timestamps: List[datetime],
                               business_hours_only: bool) -> List[DataGap]:
        """Group consecutive missing timestamps into gaps."""
        if not missing_timestamps:
            return []

        gaps = []
        gap_counter = 1

        # Sort timestamps
        missing_sorted = sorted(missing_timestamps)

        # Group consecutive timestamps
        current_group = [missing_sorted[0]]

        for i in range(1, len(missing_sorted)):
            prev_ts = missing_sorted[i-1]
            curr_ts = missing_sorted[i]

            # Check if timestamps are consecutive (allowing for business hour breaks)
            expected_next = prev_ts + timedelta(minutes=self.interval_minutes)

            if business_hours_only:
                # Skip non-business hour gaps
                if not self._is_consecutive_in_business_hours(prev_ts, curr_ts):
                    # Start new group
                    if current_group:
                        gap = self._create_gap_from_group(current_group, gap_counter, business_hours_only)
                        gaps.append(gap)
                        gap_counter += 1
                    current_group = [curr_ts]
                else:
                    current_group.append(curr_ts)
            else:
                # Simple consecutive check
                if curr_ts <= expected_next + timedelta(minutes=self.tolerance_minutes):
                    current_group.append(curr_ts)
                else:
                    # Start new group
                    if current_group:
                        gap = self._create_gap_from_group(current_group, gap_counter, business_hours_only)
                        gaps.append(gap)
                        gap_counter += 1
                    current_group = [curr_ts]

        # Handle last group
        if current_group:
            gap = self._create_gap_from_group(current_group, gap_counter, business_hours_only)
            gaps.append(gap)

        return gaps

    def _create_gap_from_group(self,
                              timestamp_group: List[datetime],
                              gap_id: int,
                              business_hours_only: bool) -> DataGap:
        """Create a DataGap object from a group of consecutive missing timestamps."""
        start_time = min(timestamp_group)
        end_time = max(timestamp_group) + timedelta(minutes=self.interval_minutes)
        missing_points = len(timestamp_group)

        # Calculate expected points for the gap period
        expected_points = self._calculate_expected_points(start_time, end_time, business_hours_only)

        # Calculate duration
        duration_minutes = (end_time - start_time).total_seconds() / 60

        # Determine gap type and severity
        gap_type = self._classify_gap_type(start_time, end_time, missing_points)
        severity = self._classify_gap_severity(duration_minutes, missing_points)

        return DataGap(
            gap_id=f"gap_{gap_id:03d}",
            gap_type=gap_type,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            expected_points=expected_points,
            missing_points=missing_points,
            duration_minutes=int(duration_minutes),
            business_hours_only=business_hours_only,
            context={
                "interval_minutes": self.interval_minutes,
                "tolerance_minutes": self.tolerance_minutes
            }
        )

    def _is_consecutive_in_business_hours(self, prev_ts: datetime, curr_ts: datetime) -> bool:
        """Check if two timestamps are consecutive considering business hours."""
        # If same day, check simple interval
        if prev_ts.date() == curr_ts.date():
            expected_next = prev_ts + timedelta(minutes=self.interval_minutes)
            return curr_ts <= expected_next + timedelta(minutes=self.tolerance_minutes)

        # Different days - check if it's next business day start
        next_business_day = self._get_next_business_day(prev_ts.date())
        if curr_ts.date() == next_business_day:
            # Check if curr_ts is at market open
            market_open = curr_ts.replace(hour=self.market_open_hour, minute=0, second=0, microsecond=0)
            return abs((curr_ts - market_open).total_seconds()) <= self.tolerance_minutes * 60

        return False

    def _get_next_business_day(self, current_date: date) -> date:
        """Get the next business day after the given date."""
        next_date = current_date + timedelta(days=1)
        while next_date.weekday() >= 5 or not self.datetime_handler.is_business_day(next_date):
            next_date += timedelta(days=1)
        return next_date

    def _classify_gap_type(self, start_time: datetime, end_time: datetime, missing_points: int) -> GapType:
        """Classify the type of gap based on timing and context."""
        # Check if gap is during weekend
        if start_time.weekday() >= 5 or end_time.weekday() >= 5:
            return GapType.WEEKEND_GAP

        # Check if gap is during holiday
        if not self.datetime_handler.is_business_day(start_time):
            return GapType.HOLIDAY_GAP

        # Check if gap is during business hours
        if self.datetime_handler.is_premium_hours(start_time):
            return GapType.BUSINESS_HOUR_GAP

        # Default to missing data
        return GapType.MISSING_DATA

    def _classify_gap_severity(self, duration_minutes: float, missing_points: int) -> GapSeverity:
        """Classify gap severity based on duration and impact."""
        for severity, threshold in self.severity_thresholds.items():
            if duration_minutes <= threshold:
                return severity
        return GapSeverity.CRITICAL

    def _calculate_expected_points(self,
                                  start_time: datetime,
                                  end_time: datetime,
                                  business_hours_only: bool) -> int:
        """Calculate expected number of data points for a time period."""
        expected_timestamps = self._generate_expected_timestamps(
            start_time, end_time, business_hours_only
        )
        return len(expected_timestamps)

    def _group_consecutive_periods(self, timestamps: List[datetime]) -> List[Tuple[datetime, datetime]]:
        """Group consecutive timestamps into periods."""
        if not timestamps:
            return []

        periods = []
        sorted_timestamps = sorted(timestamps)

        period_start = sorted_timestamps[0]
        period_end = sorted_timestamps[0]

        for i in range(1, len(sorted_timestamps)):
            curr_ts = sorted_timestamps[i]
            expected_next = period_end + timedelta(minutes=self.interval_minutes)

            if curr_ts <= expected_next + timedelta(minutes=self.tolerance_minutes):
                # Consecutive - extend current period
                period_end = curr_ts
            else:
                # Gap - close current period and start new one
                periods.append((period_start, period_end + timedelta(minutes=self.interval_minutes)))
                period_start = curr_ts
                period_end = curr_ts

        # Add final period
        periods.append((period_start, period_end + timedelta(minutes=self.interval_minutes)))

        return periods

    def _split_period_into_ranges(self,
                                 period_start: datetime,
                                 period_end: datetime,
                                 max_points_per_call: int,
                                 start_range_id: int,
                                 priority_business_hours: bool) -> List[IncrementalRange]:
        """Split a large period into smaller ranges."""
        ranges = []
        current_start = period_start
        range_counter = start_range_id

        while current_start < period_end:
            # Calculate range end based on max points
            max_duration_minutes = max_points_per_call * self.interval_minutes

            # For business hours only, we need to be more careful about range sizing
            if priority_business_hours:
                # Find next business hour boundary
                current_end = min(
                    current_start + timedelta(minutes=max_duration_minutes),
                    period_end
                )
            else:
                current_end = min(
                    current_start + timedelta(minutes=max_duration_minutes),
                    period_end
                )

            expected_points = self._calculate_expected_points(
                current_start, current_end, business_hours_only=True
            )

            if expected_points > 0:
                priority = self._calculate_priority(current_start, current_end, priority_business_hours)

                incremental_range = IncrementalRange(
                    range_id=f"range_{range_counter:03d}",
                    start_time=current_start,
                    end_time=current_end,
                    expected_points=expected_points,
                    priority=priority,
                    batch_size=expected_points,
                    api_calls_required=1,
                    estimated_duration_minutes=self._estimate_duration(1, expected_points)
                )
                ranges.append(incremental_range)
                range_counter += 1

            current_start = current_end

        return ranges

    def _calculate_priority(self, start_time: datetime, end_time: datetime, priority_business_hours: bool) -> int:
        """Calculate priority for an incremental range."""
        base_priority = 50

        if priority_business_hours:
            # Higher priority for business hours
            if self.datetime_handler.is_premium_hours(start_time):
                base_priority += 30

        # Higher priority for recent data
        hours_old = (datetime.now() - end_time).total_seconds() / 3600
        if hours_old < 24:
            base_priority += 20
        elif hours_old < 168:  # 1 week
            base_priority += 10

        # Higher priority for smaller gaps (easier to fill)
        duration_hours = (end_time - start_time).total_seconds() / 3600
        if duration_hours < 1:
            base_priority += 15
        elif duration_hours < 4:
            base_priority += 10

        return min(base_priority, 100)

    def _estimate_duration(self, api_calls: int, expected_points: int) -> int:
        """Estimate duration for fetching data."""
        # Base time per API call + processing time
        base_time_per_call = 2  # 2 minutes per API call
        processing_time = expected_points * 0.001  # 1ms per point

        return int(api_calls * base_time_per_call + processing_time)

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find the timestamp column in a DataFrame."""
        timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'dt']

        for col in df.columns:
            if col.lower() in timestamp_candidates:
                return col
            if 'time' in col.lower() or 'date' in col.lower():
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    return col

        return None

    def _summarize_gaps(self, gaps: List[DataGap]) -> Dict[str, Any]:
        """Create a summary of detected gaps."""
        if not gaps:
            return {"total": 0}

        summary = {
            "total": len(gaps),
            "by_severity": {},
            "by_type": {},
            "total_missing_points": sum(gap.missing_points for gap in gaps),
            "total_duration_minutes": sum(gap.duration_minutes for gap in gaps)
        }

        for gap in gaps:
            severity = gap.severity.value
            gap_type = gap.gap_type.value

            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            summary["by_type"][gap_type] = summary["by_type"].get(gap_type, 0) + 1

        return summary

    def _assess_data_quality(self, completeness_score: float, gaps: List[DataGap]) -> Dict[str, str]:
        """Assess overall data quality based on completeness and gaps."""
        # Determine overall quality level
        if completeness_score >= 0.98:
            quality_level = "excellent"
        elif completeness_score >= 0.95:
            quality_level = "good"
        elif completeness_score >= 0.90:
            quality_level = "acceptable"
        elif completeness_score >= 0.80:
            quality_level = "poor"
        else:
            quality_level = "critical"

        # Check for critical gaps
        critical_gaps = [gap for gap in gaps if gap.severity == GapSeverity.CRITICAL]
        if critical_gaps:
            quality_level = "critical"

        # Generate assessment message
        assessment = {
            "level": quality_level,
            "score": round(completeness_score * 100, 1),
            "message": self._generate_quality_message(quality_level, completeness_score, gaps)
        }

        return assessment

    def _generate_quality_message(self, quality_level: str, completeness_score: float, gaps: List[DataGap]) -> str:
        """Generate a quality assessment message."""
        messages = {
            "excellent": "Data quality is excellent with minimal gaps.",
            "good": "Data quality is good with minor gaps that don't significantly impact analysis.",
            "acceptable": "Data quality is acceptable but monitoring for improvement is recommended.",
            "poor": "Data quality is poor with significant gaps that may impact analysis.",
            "critical": "Data quality is critical with major gaps requiring immediate attention."
        }

        base_message = messages.get(quality_level, "Data quality assessment unavailable.")

        # Add gap-specific information
        if gaps:
            critical_gaps = len([g for g in gaps if g.severity == GapSeverity.CRITICAL])
            if critical_gaps > 0:
                base_message += f" {critical_gaps} critical gaps detected."

        return base_message

    def _generate_gap_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on gap analysis."""
        recommendations = []

        # Check for high gap frequency
        total_gaps = analysis.get("total_gaps", 0)
        if total_gaps > 10:
            recommendations.append("Consider investigating data source reliability due to high gap frequency.")

        # Check for patterns in gap timing
        time_patterns = analysis.get("time_patterns", {})
        if time_patterns.get("most_common_hour") is not None:
            hour = time_patterns["most_common_hour"]
            recommendations.append(f"Gaps commonly occur at hour {hour}:00. Check data source status at this time.")

        # Check severity distribution
        severity_dist = analysis.get("severity_distribution", {})
        if severity_dist.get("critical", 0) > 0:
            recommendations.append("Critical gaps detected. Prioritize immediate data recovery.")

        # Check for long duration gaps
        duration_stats = analysis.get("duration_stats", {})
        if duration_stats.get("max_duration", 0) > 240:  # More than 4 hours
            recommendations.append("Long-duration gaps detected. Consider implementing more frequent data checks.")

        if not recommendations:
            recommendations.append("Data quality appears stable. Continue regular monitoring.")

        return recommendations


# Convenience functions
def get_gap_detector(db_manager: DatabaseManager = None) -> GapDetector:
    """Get a gap detector instance with standard configuration."""
    return GapDetector(db_manager=db_manager)


# Example usage and testing
if __name__ == "__main__":
    # Test gap detection functionality
    import pandas as pd
    from datetime import datetime, timedelta

    try:
        # Create test data with intentional gaps
        base_time = datetime(2024, 1, 15, 8, 0)  # Start at market open
        timestamps = []

        # Add normal data for first hour
        for i in range(12):  # 12 x 5min = 1 hour
            timestamps.append(base_time + timedelta(minutes=i * 5))

        # Add gap (missing 30 minutes)
        gap_start = base_time + timedelta(hours=1)
        gap_end = gap_start + timedelta(minutes=30)

        # Resume data after gap
        for i in range(12):  # Another hour of data
            timestamps.append(gap_end + timedelta(minutes=i * 5))

        test_data = pd.DataFrame({
            'timestamp': timestamps,
            'close': [4200.0 + i * 0.1 for i in range(len(timestamps))]
        })

        print(f"Created test data with {len(timestamps)} points")

        # Test gap detector
        detector = GapDetector()

        # Detect gaps
        gaps = detector.detect_gaps(
            test_data['timestamp'],
            start_time=base_time,
            end_time=base_time + timedelta(hours=3)
        )

        print(f"Detected {len(gaps)} gaps:")
        for gap in gaps:
            print(f"  - {gap.gap_id}: {gap.start_time} to {gap.end_time} ({gap.severity.value})")

        # Test missing periods calculation
        missing_periods = detector.calculate_missing_periods(
            start_time=base_time,
            end_time=base_time + timedelta(hours=3),
            existing_data=test_data
        )

        print(f"Missing periods: {len(missing_periods)}")
        for start, end in missing_periods:
            print(f"  - {start} to {end}")

        # Test incremental ranges
        incremental_ranges = detector.get_incremental_ranges(missing_periods)

        print(f"Incremental ranges: {len(incremental_ranges)}")
        for range_obj in incremental_ranges:
            print(f"  - {range_obj.range_id}: {range_obj.expected_points} points, priority {range_obj.priority}")

        # Test data completeness validation
        validation = detector.validate_data_completeness(
            test_data,
            start_time=base_time,
            end_time=base_time + timedelta(hours=3)
        )

        print(f"Data completeness: {validation['completeness']['percentage']}%")

        print("✅ GapDetector test completed successfully!")

    except Exception as e:
        print(f"❌ GapDetector test failed: {e}")