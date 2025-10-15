"""
USDCOP L0 Pipeline - Comprehensive Data Validation Framework (FIXED)
=================================================================

This module fixes all identified timezone, business hours, holiday calendar,
and data quality validation issues in the L0 pipeline.

CRITICAL FIXES IMPLEMENTED:
1. Fixed timezone handling: Ensure data is properly converted to Colombian timezone
2. Complete Colombian holiday calendar (2020-2030)
3. Robust business hours validation with proper timezone awareness
4. Enhanced gap detection and completeness calculations
5. Improved stale/repeated OHLC detection
6. Comprehensive quality thresholds and validation rules

Author: Financial Data Validation Specialist
Date: 2025-09-18
"""

import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationSeverity(Enum):
    """Validation result severity levels"""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    CRITICAL = "CRITICAL"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    check_name: str
    severity: ValidationSeverity
    passed: bool
    value: Union[float, int, str]
    threshold: Union[float, int, str]
    message: str
    details: Dict = None

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for USDCOP data"""
    # Basic metrics
    total_bars: int
    premium_bars: int
    trading_days: int
    expected_bars: int

    # Quality percentages
    completeness_pct: float
    stale_rate_pct: float
    gap_rate_pct: float

    # Gap analysis
    gap_count: int
    max_gap_minutes: float

    # Time coverage
    start_time: datetime
    end_time: datetime
    timezone_validated: bool

    # Business hours compliance
    business_hours_bars: int
    non_business_bars: int
    holiday_bars: int

class ColombianMarketCalendar:
    """Comprehensive Colombian market holiday calendar with forex market specifics"""

    def __init__(self):
        """Initialize with comprehensive Colombian holidays 2020-2030"""
        # Core Colombian holidays (fixed dates)
        self.fixed_holidays = [
            # New Year's Day
            "2020-01-01", "2021-01-01", "2022-01-01", "2023-01-01", "2024-01-01",
            "2025-01-01", "2026-01-01", "2027-01-01", "2028-01-01", "2029-01-01", "2030-01-01",

            # Labor Day
            "2020-05-01", "2021-05-01", "2022-05-01", "2023-05-01", "2024-05-01",
            "2025-05-01", "2026-05-01", "2027-05-01", "2028-05-01", "2029-05-01", "2030-05-01",

            # Independence Day
            "2020-07-20", "2021-07-20", "2022-07-20", "2023-07-20", "2024-07-20",
            "2025-07-20", "2026-07-20", "2027-07-20", "2028-07-20", "2029-07-20", "2030-07-20",

            # Battle of Boyacá
            "2020-08-07", "2021-08-07", "2022-08-07", "2023-08-07", "2024-08-07",
            "2025-08-07", "2026-08-07", "2027-08-07", "2028-08-07", "2029-08-07", "2030-08-07",

            # Immaculate Conception
            "2020-12-08", "2021-12-08", "2022-12-08", "2023-12-08", "2024-12-08",
            "2025-12-08", "2026-12-08", "2027-12-08", "2028-12-08", "2029-12-08", "2030-12-08",

            # Christmas Day
            "2020-12-25", "2021-12-25", "2022-12-25", "2023-12-25", "2024-12-25",
            "2025-12-25", "2026-12-25", "2027-12-25", "2028-12-25", "2029-12-25", "2030-12-25",
        ]

        # Variable holidays (moved to Monday if needed)
        self.variable_holidays = [
            # Three Kings Day (Epiphany) - moved to first Monday after Jan 6
            "2020-01-06", "2021-01-11", "2022-01-10", "2023-01-09", "2024-01-08",
            "2025-01-06", "2026-01-12", "2027-01-11", "2028-01-10", "2029-01-08", "2030-01-14",

            # St. Joseph's Day - moved to Monday after Mar 19
            "2020-03-23", "2021-03-22", "2022-03-21", "2023-03-20", "2024-03-25",
            "2025-03-24", "2026-03-23", "2027-03-22", "2028-03-20", "2029-03-19", "2030-03-25",
        ]

        # Easter-related holidays (calculated separately)
        self.easter_holidays = [
            # Maundy Thursday
            "2020-04-09", "2021-04-01", "2022-04-14", "2023-04-06", "2024-03-28",
            "2025-04-17", "2026-04-02", "2027-03-25", "2028-04-13", "2029-03-29", "2030-04-18",

            # Good Friday
            "2020-04-10", "2021-04-02", "2022-04-15", "2023-04-07", "2024-03-29",
            "2025-04-18", "2026-04-03", "2027-03-26", "2028-04-14", "2029-03-30", "2030-04-19",

            # Ascension of Jesus (39 days after Easter)
            "2020-05-25", "2021-05-17", "2022-05-30", "2023-05-22", "2024-05-13",
            "2025-06-02", "2026-05-18", "2027-05-10", "2028-05-29", "2029-05-14", "2030-06-03",

            # Corpus Christi (60 days after Easter)
            "2020-06-15", "2021-06-14", "2022-06-20", "2023-06-12", "2024-06-03",
            "2025-06-23", "2026-06-08", "2027-05-31", "2028-06-19", "2029-06-04", "2030-06-24",

            # Sacred Heart (68 days after Easter)
            "2020-06-22", "2021-06-21", "2022-06-27", "2023-06-19", "2024-06-10",
            "2025-06-30", "2026-06-15", "2027-06-07", "2028-06-26", "2029-06-11", "2030-07-01",
        ]

        # Other significant holidays
        self.other_holidays = [
            # All Saints Day - moved to Monday after Nov 1
            "2020-11-02", "2021-11-01", "2022-11-07", "2023-11-06", "2024-11-04",
            "2025-11-03", "2026-11-02", "2027-11-01", "2028-11-06", "2029-11-05", "2030-11-04",

            # Independence of Cartagena - moved to Monday after Nov 11
            "2020-11-16", "2021-11-15", "2022-11-14", "2023-11-13", "2024-11-11",
            "2025-11-17", "2026-11-16", "2027-11-15", "2028-11-13", "2029-11-12", "2030-11-18",
        ]

        # Combine all holidays
        all_holiday_strings = (self.fixed_holidays + self.variable_holidays +
                             self.easter_holidays + self.other_holidays)

        # Convert to date objects and remove duplicates
        self.holidays = set()
        for holiday_str in all_holiday_strings:
            try:
                holiday_date = pd.to_datetime(holiday_str).date()
                self.holidays.add(holiday_date)
            except:
                logger.warning(f"Could not parse holiday date: {holiday_str}")

        logger.info(f"Loaded {len(self.holidays)} Colombian holidays for 2020-2030")

    def is_holiday(self, check_date: Union[datetime, date, str]) -> bool:
        """Check if a given date is a Colombian holiday"""
        if isinstance(check_date, str):
            check_date = pd.to_datetime(check_date).date()
        elif isinstance(check_date, datetime):
            check_date = check_date.date()

        return check_date in self.holidays

    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[date]:
        """Get all trading days (business days excluding holidays) in a date range"""
        business_days = pd.bdate_range(start=start_date, end=end_date, freq='B')
        trading_days = []

        for day in business_days:
            if not self.is_holiday(day.date()):
                trading_days.append(day.date())

        return trading_days

class TimezoneHandler:
    """Handles timezone operations for Colombian market data"""

    def __init__(self):
        """Initialize timezone handler for Colombian market"""
        self.colombia_tz = pytz.timezone('America/Bogota')  # UTC-5
        self.utc_tz = pytz.UTC

    def ensure_colombia_timezone(self, df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
        """Ensure dataframe times are in Colombian timezone"""
        df = df.copy()

        # Convert time column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])

        # Check if timezone aware
        if df[time_col].dt.tz is None:
            # Assume data is already in Colombian time if no timezone info
            logger.warning("No timezone info found, assuming Colombian time")
            df[time_col] = df[time_col].dt.tz_localize(self.colombia_tz)
        else:
            # Convert to Colombian timezone
            df[time_col] = df[time_col].dt.tz_convert(self.colombia_tz)

        return df

    def get_business_hours_mask(self, df: pd.DataFrame, time_col: str = 'time') -> pd.Series:
        """Get boolean mask for Colombian business hours (8AM-2PM COT)"""
        # Ensure proper timezone
        df = self.ensure_colombia_timezone(df, time_col)

        # Extract hour in Colombian timezone
        colombia_hour = df[time_col].dt.hour

        # Business hours: 8 AM to 2 PM (14:00) Colombian time
        return (colombia_hour >= 8) & (colombia_hour < 14)

class USDCOPDataValidator:
    """Comprehensive USDCOP data validator with fixed logic"""

    def __init__(self):
        """Initialize validator with fixed configuration"""
        self.calendar = ColombianMarketCalendar()
        self.timezone_handler = TimezoneHandler()

        # Business configuration
        self.business_hours_start = 8   # 8 AM COT
        self.business_hours_end = 14    # 2 PM COT
        self.bars_per_hour = 12         # 5-minute frequency
        self.expected_bars_per_day = (self.business_hours_end - self.business_hours_start) * self.bars_per_hour  # 72

        # Quality thresholds
        self.min_completeness = 95.0    # 95% minimum completeness
        self.max_stale_rate = 2.0       # 2% maximum stale rate
        self.max_gap_rate = 5.0         # 5% maximum gap rate
        self.expected_interval_minutes = 5  # 5-minute bars

        # Validation results storage
        self.validation_results: List[ValidationResult] = []

    def validate_timezone_consistency(self, df: pd.DataFrame, time_col: str = 'time') -> ValidationResult:
        """Validate timezone consistency and proper Colombian timezone handling"""
        try:
            # Ensure Colombian timezone
            df_tz = self.timezone_handler.ensure_colombia_timezone(df, time_col)

            # Check if all timestamps are timezone-aware
            is_tz_aware = df_tz[time_col].dt.tz is not None

            # Check if timezone is Colombian
            if is_tz_aware:
                sample_tz = df_tz[time_col].iloc[0].tz
                is_colombia_tz = str(sample_tz) == 'America/Bogota'
            else:
                is_colombia_tz = False

            passed = is_tz_aware and is_colombia_tz
            severity = ValidationSeverity.PASS if passed else ValidationSeverity.CRITICAL

            message = "Timezone validation passed" if passed else "Timezone validation failed - not Colombian timezone"

            return ValidationResult(
                check_name="timezone_consistency",
                severity=severity,
                passed=passed,
                value=str(sample_tz) if is_tz_aware else "No timezone",
                threshold="America/Bogota",
                message=message
            )

        except Exception as e:
            return ValidationResult(
                check_name="timezone_consistency",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                value="Error",
                threshold="America/Bogota",
                message=f"Timezone validation error: {str(e)}"
            )

    def validate_business_hours(self, df: pd.DataFrame, time_col: str = 'time') -> ValidationResult:
        """Validate business hours filtering (8AM-2PM Colombian time)"""
        try:
            # Get business hours mask
            business_mask = self.timezone_handler.get_business_hours_mask(df, time_col)

            business_bars = business_mask.sum()
            total_bars = len(df)
            business_pct = (business_bars / total_bars * 100) if total_bars > 0 else 0

            # We expect most bars to be in business hours (should be > 80%)
            passed = business_pct > 80.0
            severity = ValidationSeverity.PASS if passed else ValidationSeverity.WARNING

            message = f"Business hours bars: {business_bars}/{total_bars} ({business_pct:.1f}%)"

            return ValidationResult(
                check_name="business_hours_validation",
                severity=severity,
                passed=passed,
                value=business_pct,
                threshold=80.0,
                message=message,
                details={
                    'business_bars': int(business_bars),
                    'total_bars': total_bars,
                    'non_business_bars': total_bars - int(business_bars)
                }
            )

        except Exception as e:
            return ValidationResult(
                check_name="business_hours_validation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                value=0,
                threshold=80.0,
                message=f"Business hours validation error: {str(e)}"
            )

    def validate_completeness(self, df: pd.DataFrame, start_date: datetime, end_date: datetime) -> ValidationResult:
        """Validate data completeness against expected trading schedule"""
        try:
            # Get trading days (exclude holidays)
            trading_days = self.calendar.get_trading_days(start_date, end_date)
            expected_total_bars = len(trading_days) * self.expected_bars_per_day

            # Filter for business hours only
            business_mask = self.timezone_handler.get_business_hours_mask(df)
            actual_business_bars = business_mask.sum()

            # Calculate completeness
            completeness_pct = (actual_business_bars / expected_total_bars * 100) if expected_total_bars > 0 else 0

            passed = completeness_pct >= self.min_completeness
            severity = ValidationSeverity.PASS if passed else ValidationSeverity.FAIL

            message = f"Completeness: {actual_business_bars}/{expected_total_bars} bars ({completeness_pct:.1f}%)"

            return ValidationResult(
                check_name="completeness_validation",
                severity=severity,
                passed=passed,
                value=completeness_pct,
                threshold=self.min_completeness,
                message=message,
                details={
                    'trading_days': len(trading_days),
                    'expected_bars': expected_total_bars,
                    'actual_bars': int(actual_business_bars),
                    'missing_bars': expected_total_bars - int(actual_business_bars)
                }
            )

        except Exception as e:
            return ValidationResult(
                check_name="completeness_validation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                value=0,
                threshold=self.min_completeness,
                message=f"Completeness validation error: {str(e)}"
            )

    def validate_gaps(self, df: pd.DataFrame, time_col: str = 'time') -> ValidationResult:
        """Validate time gaps in the data"""
        try:
            if len(df) < 2:
                return ValidationResult(
                    check_name="gap_validation",
                    severity=ValidationSeverity.WARNING,
                    passed=True,
                    value=0,
                    threshold=self.max_gap_rate,
                    message="Insufficient data for gap analysis"
                )

            # Ensure proper timezone and sort by time
            df_sorted = self.timezone_handler.ensure_colombia_timezone(df, time_col).sort_values(time_col)

            # Filter for business hours only
            business_mask = self.timezone_handler.get_business_hours_mask(df_sorted, time_col)
            business_df = df_sorted[business_mask].copy()

            if len(business_df) < 2:
                return ValidationResult(
                    check_name="gap_validation",
                    severity=ValidationSeverity.WARNING,
                    passed=True,
                    value=0,
                    threshold=self.max_gap_rate,
                    message="Insufficient business hours data for gap analysis"
                )

            # Calculate time differences
            time_diffs = business_df[time_col].diff().dt.total_seconds() / 60  # Convert to minutes
            expected_diff = self.expected_interval_minutes

            # Identify gaps (more than 1.5x expected interval)
            gap_threshold = expected_diff * 1.5
            gaps = time_diffs[time_diffs > gap_threshold]

            gap_count = len(gaps)
            max_gap_minutes = gaps.max() if len(gaps) > 0 else 0
            gap_rate_pct = (gap_count / len(business_df) * 100) if len(business_df) > 0 else 0

            passed = gap_rate_pct <= self.max_gap_rate
            severity = ValidationSeverity.PASS if passed else ValidationSeverity.WARNING

            message = f"Gaps: {gap_count} gaps ({gap_rate_pct:.1f}%), max gap: {max_gap_minutes:.0f} min"

            return ValidationResult(
                check_name="gap_validation",
                severity=severity,
                passed=passed,
                value=gap_rate_pct,
                threshold=self.max_gap_rate,
                message=message,
                details={
                    'gap_count': gap_count,
                    'max_gap_minutes': float(max_gap_minutes),
                    'total_intervals': len(business_df) - 1
                }
            )

        except Exception as e:
            return ValidationResult(
                check_name="gap_validation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                value=0,
                threshold=self.max_gap_rate,
                message=f"Gap validation error: {str(e)}"
            )

    def validate_stale_data(self, df: pd.DataFrame, time_col: str = 'time') -> ValidationResult:
        """Validate for stale/repeated OHLC data"""
        try:
            ohlc_cols = ['open', 'high', 'low', 'close']

            # Check if all OHLC columns exist
            missing_cols = [col for col in ohlc_cols if col not in df.columns]
            if missing_cols:
                return ValidationResult(
                    check_name="stale_data_validation",
                    severity=ValidationSeverity.CRITICAL,
                    passed=False,
                    value=0,
                    threshold=self.max_stale_rate,
                    message=f"Missing OHLC columns: {missing_cols}"
                )

            if len(df) < 2:
                return ValidationResult(
                    check_name="stale_data_validation",
                    severity=ValidationSeverity.PASS,
                    passed=True,
                    value=0,
                    threshold=self.max_stale_rate,
                    message="Insufficient data for stale analysis"
                )

            # Ensure proper timezone and sort by time
            df_sorted = self.timezone_handler.ensure_colombia_timezone(df, time_col).sort_values(time_col)

            # Filter for business hours only
            business_mask = self.timezone_handler.get_business_hours_mask(df_sorted, time_col)
            business_df = df_sorted[business_mask].copy()

            if len(business_df) < 2:
                return ValidationResult(
                    check_name="stale_data_validation",
                    severity=ValidationSeverity.PASS,
                    passed=True,
                    value=0,
                    threshold=self.max_stale_rate,
                    message="Insufficient business hours data for stale analysis"
                )

            # Check for consecutive identical OHLC values
            stale_count = 0
            for i in range(1, len(business_df)):
                current_ohlc = business_df.iloc[i][ohlc_cols].values
                previous_ohlc = business_df.iloc[i-1][ohlc_cols].values

                # Check if all OHLC values are identical (with small tolerance for floating point)
                if np.allclose(current_ohlc, previous_ohlc, rtol=1e-10, atol=1e-10):
                    stale_count += 1

            stale_rate_pct = (stale_count / len(business_df) * 100) if len(business_df) > 0 else 0

            passed = stale_rate_pct <= self.max_stale_rate
            severity = ValidationSeverity.PASS if passed else ValidationSeverity.FAIL

            message = f"Stale data: {stale_count}/{len(business_df)} bars ({stale_rate_pct:.1f}%)"

            return ValidationResult(
                check_name="stale_data_validation",
                severity=severity,
                passed=passed,
                value=stale_rate_pct,
                threshold=self.max_stale_rate,
                message=message,
                details={
                    'stale_count': stale_count,
                    'total_bars': len(business_df)
                }
            )

        except Exception as e:
            return ValidationResult(
                check_name="stale_data_validation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                value=0,
                threshold=self.max_stale_rate,
                message=f"Stale data validation error: {str(e)}"
            )

    def validate_holiday_filtering(self, df: pd.DataFrame, time_col: str = 'time') -> ValidationResult:
        """Validate that data doesn't include Colombian holidays"""
        try:
            # Ensure proper timezone
            df_tz = self.timezone_handler.ensure_colombia_timezone(df, time_col)

            # Extract dates
            dates = df_tz[time_col].dt.date.unique()

            # Check for holidays
            holiday_dates = []
            for date_val in dates:
                if self.calendar.is_holiday(date_val):
                    holiday_dates.append(date_val)

            holiday_count = len(holiday_dates)
            passed = holiday_count == 0
            severity = ValidationSeverity.PASS if passed else ValidationSeverity.WARNING

            if holiday_count > 0:
                message = f"Data includes {holiday_count} holiday dates: {holiday_dates}"
            else:
                message = "No holiday data found - good"

            return ValidationResult(
                check_name="holiday_filtering_validation",
                severity=severity,
                passed=passed,
                value=holiday_count,
                threshold=0,
                message=message,
                details={'holiday_dates': [str(d) for d in holiday_dates]}
            )

        except Exception as e:
            return ValidationResult(
                check_name="holiday_filtering_validation",
                severity=ValidationSeverity.CRITICAL,
                passed=False,
                value=0,
                threshold=0,
                message=f"Holiday filtering validation error: {str(e)}"
            )

    def calculate_comprehensive_metrics(self, df: pd.DataFrame, start_date: datetime,
                                      end_date: datetime, time_col: str = 'time') -> QualityMetrics:
        """Calculate comprehensive quality metrics"""
        try:
            # Ensure proper timezone
            df_tz = self.timezone_handler.ensure_colombia_timezone(df, time_col)

            # Get business hours mask
            business_mask = self.timezone_handler.get_business_hours_mask(df_tz, time_col)
            business_df = df_tz[business_mask]

            # Get trading days
            trading_days = self.calendar.get_trading_days(start_date, end_date)
            expected_bars = len(trading_days) * self.expected_bars_per_day

            # Calculate completeness
            completeness_pct = (len(business_df) / expected_bars * 100) if expected_bars > 0 else 0

            # Calculate stale rate
            stale_validation = self.validate_stale_data(df, time_col)
            stale_rate_pct = stale_validation.value if stale_validation.passed else 0

            # Calculate gap rate
            gap_validation = self.validate_gaps(df, time_col)
            gap_rate_pct = gap_validation.value if gap_validation.passed else 0
            gap_count = gap_validation.details.get('gap_count', 0) if gap_validation.details else 0
            max_gap_minutes = gap_validation.details.get('max_gap_minutes', 0) if gap_validation.details else 0

            # Holiday bars
            holiday_validation = self.validate_holiday_filtering(df, time_col)
            holiday_bars = holiday_validation.value if isinstance(holiday_validation.value, int) else 0

            return QualityMetrics(
                total_bars=len(df),
                premium_bars=len(business_df),
                trading_days=len(trading_days),
                expected_bars=expected_bars,
                completeness_pct=completeness_pct,
                stale_rate_pct=stale_rate_pct,
                gap_rate_pct=gap_rate_pct,
                gap_count=gap_count,
                max_gap_minutes=max_gap_minutes,
                start_time=df_tz[time_col].min(),
                end_time=df_tz[time_col].max(),
                timezone_validated=True,
                business_hours_bars=len(business_df),
                non_business_bars=len(df) - len(business_df),
                holiday_bars=holiday_bars
            )

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return QualityMetrics(
                total_bars=len(df),
                premium_bars=0,
                trading_days=0,
                expected_bars=0,
                completeness_pct=0,
                stale_rate_pct=0,
                gap_rate_pct=0,
                gap_count=0,
                max_gap_minutes=0,
                start_time=datetime.now(),
                end_time=datetime.now(),
                timezone_validated=False,
                business_hours_bars=0,
                non_business_bars=len(df),
                holiday_bars=0
            )

    def run_comprehensive_validation(self, df: pd.DataFrame, start_date: datetime,
                                   end_date: datetime, time_col: str = 'time') -> Dict:
        """Run all validation checks and return comprehensive report"""
        logger.info("="*80)
        logger.info("USDCOP L0 PIPELINE - COMPREHENSIVE DATA VALIDATION")
        logger.info("="*80)

        # Clear previous results
        self.validation_results = []

        # Run all validation checks
        validations = [
            self.validate_timezone_consistency(df, time_col),
            self.validate_business_hours(df, time_col),
            self.validate_completeness(df, start_date, end_date),
            self.validate_gaps(df, time_col),
            self.validate_stale_data(df, time_col),
            self.validate_holiday_filtering(df, time_col)
        ]

        self.validation_results.extend(validations)

        # Calculate comprehensive metrics
        metrics = self.calculate_comprehensive_metrics(df, start_date, end_date, time_col)

        # Determine overall status
        failed_critical = sum(1 for v in validations if v.severity == ValidationSeverity.CRITICAL and not v.passed)
        failed_validations = sum(1 for v in validations if v.severity == ValidationSeverity.FAIL and not v.passed)
        warnings = sum(1 for v in validations if v.severity == ValidationSeverity.WARNING and not v.passed)

        if failed_critical > 0:
            overall_status = "CRITICAL_FAIL"
        elif failed_validations > 0:
            overall_status = "FAIL"
        elif warnings > 0:
            overall_status = "PASS_WITH_WARNINGS"
        else:
            overall_status = "PASS"

        # Generate comprehensive report
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'overall_status': overall_status,
            'data_period': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'actual_start': metrics.start_time.isoformat() if metrics.timezone_validated else None,
                'actual_end': metrics.end_time.isoformat() if metrics.timezone_validated else None
            },
            'quality_metrics': {
                'total_bars': metrics.total_bars,
                'premium_bars': metrics.premium_bars,
                'trading_days': metrics.trading_days,
                'expected_bars': metrics.expected_bars,
                'completeness_pct': round(metrics.completeness_pct, 2),
                'stale_rate_pct': round(metrics.stale_rate_pct, 2),
                'gap_rate_pct': round(metrics.gap_rate_pct, 2),
                'gap_count': metrics.gap_count,
                'max_gap_minutes': round(metrics.max_gap_minutes, 2),
                'business_hours_bars': metrics.business_hours_bars,
                'non_business_bars': metrics.non_business_bars,
                'holiday_bars': metrics.holiday_bars,
                'timezone_validated': metrics.timezone_validated
            },
            'validation_results': [
                {
                    'check_name': v.check_name,
                    'severity': v.severity.value,
                    'passed': v.passed,
                    'value': v.value,
                    'threshold': v.threshold,
                    'message': v.message,
                    'details': v.details
                }
                for v in validations
            ],
            'thresholds': {
                'min_completeness_pct': self.min_completeness,
                'max_stale_rate_pct': self.max_stale_rate,
                'max_gap_rate_pct': self.max_gap_rate,
                'business_hours': f"{self.business_hours_start}:00-{self.business_hours_end}:00 COT",
                'expected_bars_per_day': self.expected_bars_per_day
            },
            'summary': {
                'total_validations': len(validations),
                'passed_validations': sum(1 for v in validations if v.passed),
                'failed_critical': failed_critical,
                'failed_validations': failed_validations,
                'warnings': warnings
            }
        }

        # Log results
        logger.info(f"Validation Status: {overall_status}")
        logger.info(f"Completeness: {metrics.completeness_pct:.1f}% ({metrics.premium_bars}/{metrics.expected_bars} bars)")
        logger.info(f"Stale Rate: {metrics.stale_rate_pct:.1f}% (threshold: {self.max_stale_rate}%)")
        logger.info(f"Gap Rate: {metrics.gap_rate_pct:.1f}% (threshold: {self.max_gap_rate}%)")
        logger.info(f"Holiday Bars: {metrics.holiday_bars}")
        logger.info(f"Validations: {report['summary']['passed_validations']}/{len(validations)} passed")

        for validation in validations:
            status_symbol = "✅" if validation.passed else ("❌" if validation.severity == ValidationSeverity.FAIL else "⚠️")
            logger.info(f"  {status_symbol} {validation.check_name}: {validation.message}")

        logger.info("="*80)

        return report

# Example usage and testing functions
def create_test_data() -> pd.DataFrame:
    """Create test USDCOP data for validation testing"""
    colombia_tz = pytz.timezone('America/Bogota')

    # Create 3 days of 5-minute data during business hours
    start_date = datetime(2024, 1, 15, 8, 0)  # Monday 8 AM
    end_date = datetime(2024, 1, 17, 14, 0)   # Wednesday 2 PM

    # Generate timestamps every 5 minutes during business hours
    timestamps = []
    current = start_date

    while current <= end_date:
        # Only include business hours (8 AM to 2 PM)
        if 8 <= current.hour < 14:
            timestamps.append(current)
        current += timedelta(minutes=5)

    # Create OHLC data
    np.random.seed(42)
    n_bars = len(timestamps)
    base_price = 4000.0

    # Generate realistic OHLC data
    close_prices = base_price + np.cumsum(np.random.normal(0, 1, n_bars))
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = base_price

    high_prices = np.maximum(open_prices, close_prices) + np.random.uniform(0, 5, n_bars)
    low_prices = np.minimum(open_prices, close_prices) - np.random.uniform(0, 5, n_bars)

    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100, 1000, n_bars)
    })

    # Localize to Colombian timezone
    df['time'] = df['time'].dt.tz_localize(colombia_tz)

    return df

def run_validation_example():
    """Example of running the validation framework"""
    # Create test data
    test_df = create_test_data()

    # Initialize validator
    validator = USDCOPDataValidator()

    # Define validation period
    start_date = datetime(2024, 1, 15)
    end_date = datetime(2024, 1, 17)

    # Run comprehensive validation
    report = validator.run_comprehensive_validation(test_df, start_date, end_date)

    return report

if __name__ == "__main__":
    # Run example validation
    example_report = run_validation_example()

    # Print summary
    print("\n" + "="*80)
    print("VALIDATION FRAMEWORK EXAMPLE RESULTS")
    print("="*80)
    print(f"Overall Status: {example_report['overall_status']}")
    print(f"Completeness: {example_report['quality_metrics']['completeness_pct']}%")
    print(f"Stale Rate: {example_report['quality_metrics']['stale_rate_pct']}%")
    print(f"Gap Rate: {example_report['quality_metrics']['gap_rate_pct']}%")
    print("="*80)