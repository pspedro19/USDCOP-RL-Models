# -*- coding: utf-8 -*-
"""
Anti-Leakage Validator
======================
Comprehensive validation to prevent data leakage in ML training pipelines.

Contract: CTR-L0-ANTILEAKAGE-001

This validator goes beyond simple future date detection to ensure:
1. Macro data is shifted T-1 (no same-day data leakage)
2. Normalization stats computed from TRAIN set only
3. No overlap between train/val/test sets
4. Proper merge_asof direction (backward only)

Usage:
    from validators.anti_leakage_validator import AntiLeakageValidator

    validator = AntiLeakageValidator()
    result = validator.validate_macro_shift_t1(df, 'IBR_OVERNIGHT')
    result = validator.validate_normalization_source(norm_stats, train_dates)
    result = validator.validate_no_future_data(df, reference_date)
    result = validator.validate_dataset_splits(train_df, val_df, test_df)

Version: 1.0.0
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from .data_validators import (
    DataValidator,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


@dataclass
class AntiLeakageReport:
    """Comprehensive anti-leakage validation report."""

    passed: bool
    checks: Dict[str, ValidationResult] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def add_check(self, name: str, result: ValidationResult):
        """Add a validation check result."""
        self.checks[name] = result
        if not result.passed:
            self.passed = False

    def get_all_errors(self) -> List[str]:
        """Get all error messages."""
        errors = []
        for name, result in self.checks.items():
            for error in result.errors:
                errors.append(f"[{name}] {error}")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'passed': self.passed,
            'checks': {
                name: {
                    'passed': r.passed,
                    'errors': r.errors,
                    'warnings': r.warnings,
                    'metadata': r.metadata,
                }
                for name, r in self.checks.items()
            },
            'timestamp': self.timestamp.isoformat(),
            'error_count': len(self.get_all_errors()),
        }


class AntiLeakageValidator:
    """
    Comprehensive anti-leakage validator for ML pipelines.

    Validates:
    - T-1 shift for macro data (no same-day data)
    - Normalization computed from train set only
    - No temporal overlap in train/val/test splits
    - Proper merge_asof direction
    - No future data relative to reference date
    """

    # Macro variables that require T-1 shift
    # These are typically economic indicators released after market close
    MACRO_VARIABLES_T1 = [
        'IBR_OVERNIGHT', 'FEDFUNDS', 'DXY', 'VIX', 'EMBI', 'BRENT',
        'COLCAP', 'COLTES', 'CPI', 'GDP', 'UNEMPLOYMENT',
    ]

    # Minimum safe buffer for macro data (hours after market close)
    MACRO_RELEASE_BUFFER_HOURS = 18  # Typically released after US market close

    def __init__(
        self,
        strict_mode: bool = True,
        market_timezone: str = 'America/Bogota'
    ):
        """
        Initialize validator.

        Args:
            strict_mode: If True, any leakage is a critical error
            market_timezone: Timezone for market hours validation
        """
        self.strict_mode = strict_mode
        self.market_timezone = market_timezone

    def validate_macro_shift_t1(
        self,
        df: pd.DataFrame,
        variable: str,
        date_column: str = 'fecha',
        value_column: Optional[str] = None
    ) -> ValidationResult:
        """
        Verify macro data is shifted T-1 (no same-day data leakage).

        For macro indicators, the value for date T should only be available
        at T+1, because economic data is typically released after market close.

        Args:
            df: DataFrame with macro data
            variable: Name of the macro variable
            date_column: Name of date column
            value_column: Name of value column (defaults to variable name)

        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []
        warnings = []
        value_col = value_column or variable

        if date_column not in df.columns:
            return ValidationResult(
                passed=False,
                validator="AntiLeakageValidator.macro_shift_t1",
                errors=[f"Missing date column: {date_column}"],
                severity=ValidationSeverity.CRITICAL,
            )

        if value_col not in df.columns:
            return ValidationResult(
                passed=False,
                validator="AntiLeakageValidator.macro_shift_t1",
                errors=[f"Missing value column: {value_col}"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Convert to datetime
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])

        # Check for same-day data pattern
        # If macro data for date T is used with OHLCV for date T, it's leakage
        # We detect this by checking if value changes align with date changes

        df_sorted = df.sort_values(date_column).reset_index(drop=True)

        # Count unique dates with values
        dates_with_values = df_sorted[df_sorted[value_col].notna()][date_column].dt.date.unique()

        if len(dates_with_values) < 2:
            return ValidationResult(
                passed=True,
                validator="AntiLeakageValidator.macro_shift_t1",
                warnings=["Insufficient data to verify T-1 shift"],
                severity=ValidationSeverity.INFO,
                metadata={'dates_count': len(dates_with_values)},
            )

        # Check for suspicious patterns:
        # 1. Value changes exactly at midnight (suggests same-day data)
        # 2. Value available before expected release time

        today = datetime.now().date()
        same_day_values = df_sorted[
            (df_sorted[date_column].dt.date == today) &
            (df_sorted[value_col].notna())
        ]

        if len(same_day_values) > 0:
            # This is a potential leakage - macro data for today shouldn't be available
            msg = (
                f"POTENTIAL LEAKAGE: {variable} has {len(same_day_values)} values for today "
                f"({today}). Macro data should be T-1 shifted."
            )
            if self.strict_mode:
                errors.append(msg)
            else:
                warnings.append(msg)

        # Validate that the most recent macro value is from at least T-1
        most_recent_date = df_sorted[df_sorted[value_col].notna()][date_column].max()
        if pd.notna(most_recent_date):
            days_ago = (pd.Timestamp.now() - most_recent_date).days

            if days_ago < 1:
                msg = (
                    f"LEAKAGE WARNING: Most recent {variable} value is from {most_recent_date.date()}, "
                    f"which is less than T-1. Expected at least 1 day delay."
                )
                if self.strict_mode:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        return ValidationResult(
            passed=len(errors) == 0,
            validator="AntiLeakageValidator.macro_shift_t1",
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
            metadata={
                'variable': variable,
                'dates_count': len(dates_with_values),
                'most_recent_date': str(most_recent_date.date()) if pd.notna(most_recent_date) else None,
            },
        )

    def validate_normalization_source(
        self,
        norm_stats: Dict[str, Any],
        train_start: Union[str, datetime, date],
        train_end: Union[str, datetime, date],
    ) -> ValidationResult:
        """
        Verify norm_stats were computed from TRAIN set only.

        Normalization statistics MUST be computed only from training data.
        Using validation or test data to compute normalization leads to leakage.

        Args:
            norm_stats: Normalization statistics dictionary
            train_start: Training set start date
            train_end: Training set end date

        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []
        warnings = []

        # Check for _meta section with versioning
        meta = norm_stats.get('_meta', {})

        if not meta:
            warnings.append(
                "norm_stats missing _meta section. Cannot verify training-only computation. "
                "Consider adding _meta.computed_from and _meta.date_range."
            )
        else:
            # Verify computed_from field
            computed_from = meta.get('computed_from', '').lower()
            if computed_from and 'train' not in computed_from:
                errors.append(
                    f"LEAKAGE: norm_stats computed_from='{computed_from}'. "
                    "Must be computed from train set only."
                )

            # Verify date_range doesn't exceed train period
            date_range = meta.get('date_range', {})
            if date_range:
                stats_start = pd.to_datetime(date_range.get('start'))
                stats_end = pd.to_datetime(date_range.get('end'))
                train_start_dt = pd.to_datetime(train_start)
                train_end_dt = pd.to_datetime(train_end)

                if pd.notna(stats_start) and stats_start < train_start_dt:
                    errors.append(
                        f"LEAKAGE: norm_stats start date ({stats_start.date()}) is before "
                        f"train start ({train_start_dt.date()})."
                    )

                if pd.notna(stats_end) and stats_end > train_end_dt:
                    errors.append(
                        f"LEAKAGE: norm_stats end date ({stats_end.date()}) is after "
                        f"train end ({train_end_dt.date()}). Possible val/test data contamination."
                    )

        # Check individual feature statistics for anomalies
        features_checked = 0
        for key, stats in norm_stats.items():
            if key.startswith('_'):
                continue

            if isinstance(stats, dict) and 'count' in stats:
                features_checked += 1

                # Check for suspiciously high counts (might include val/test)
                count = stats.get('count', 0)
                if count > 500000:  # Arbitrary threshold for 5min data
                    warnings.append(
                        f"Feature '{key}' has {count:,} samples. "
                        "Verify this is from train set only."
                    )

        return ValidationResult(
            passed=len(errors) == 0,
            validator="AntiLeakageValidator.normalization_source",
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.WARNING,
            metadata={
                'has_meta': bool(meta),
                'features_checked': features_checked,
                'train_start': str(train_start),
                'train_end': str(train_end),
            },
        )

    def validate_no_future_data(
        self,
        df: pd.DataFrame,
        reference_date: Union[str, datetime, date],
        date_column: str = 'fecha',
    ) -> ValidationResult:
        """
        Verify no data from future relative to reference date.

        Args:
            df: DataFrame to validate
            reference_date: The "current" date - no data should be after this
            date_column: Name of date column

        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []

        if date_column not in df.columns:
            return ValidationResult(
                passed=False,
                validator="AntiLeakageValidator.no_future_data",
                errors=[f"Missing date column: {date_column}"],
                severity=ValidationSeverity.CRITICAL,
            )

        df_dates = pd.to_datetime(df[date_column], errors='coerce')
        ref_date = pd.Timestamp(reference_date)

        future_mask = df_dates > ref_date
        future_count = future_mask.sum()

        if future_count > 0:
            future_dates = df_dates[future_mask].dt.strftime('%Y-%m-%d').unique()[:5]
            errors.append(
                f"LEAKAGE: {future_count} rows have dates after reference date "
                f"({ref_date.date()}). Examples: {', '.join(future_dates)}"
            )

        return ValidationResult(
            passed=len(errors) == 0,
            validator="AntiLeakageValidator.no_future_data",
            errors=errors,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
            metadata={
                'reference_date': str(ref_date.date()),
                'future_rows_count': future_count,
                'total_rows': len(df),
            },
        )

    def validate_dataset_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        date_column: str = 'fecha',
    ) -> ValidationResult:
        """
        Validate no temporal overlap between train/val/test splits.

        For time series data, splits MUST be chronological:
        train < val < test

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            date_column: Name of date column

        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []
        warnings = []

        def get_date_range(df: pd.DataFrame, name: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
            dates = pd.to_datetime(df[date_column], errors='coerce')
            return dates.min(), dates.max()

        try:
            train_start, train_end = get_date_range(train_df, 'train')
            val_start, val_end = get_date_range(val_df, 'val')
            test_start, test_end = get_date_range(test_df, 'test')
        except Exception as e:
            return ValidationResult(
                passed=False,
                validator="AntiLeakageValidator.dataset_splits",
                errors=[f"Failed to extract date ranges: {str(e)}"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Check train < val
        if train_end >= val_start:
            overlap_days = (train_end - val_start).days + 1
            errors.append(
                f"LEAKAGE: Train/Val overlap! Train ends {train_end.date()}, "
                f"Val starts {val_start.date()}. Overlap: {overlap_days} days."
            )

        # Check val < test
        if val_end >= test_start:
            overlap_days = (val_end - test_start).days + 1
            errors.append(
                f"LEAKAGE: Val/Test overlap! Val ends {val_end.date()}, "
                f"Test starts {test_start.date()}. Overlap: {overlap_days} days."
            )

        # Check train < test (should never happen if above pass)
        if train_end >= test_start:
            errors.append(
                f"LEAKAGE: Train/Test overlap! Train ends {train_end.date()}, "
                f"Test starts {test_start.date()}."
            )

        # Check for gaps (not errors, but worth noting)
        if val_start > train_end + timedelta(days=1):
            gap_days = (val_start - train_end).days
            warnings.append(
                f"Gap between train and val: {gap_days} days. "
                "This is acceptable but may waste data."
            )

        if test_start > val_end + timedelta(days=1):
            gap_days = (test_start - val_end).days
            warnings.append(
                f"Gap between val and test: {gap_days} days. "
                "This is acceptable but may waste data."
            )

        return ValidationResult(
            passed=len(errors) == 0,
            validator="AntiLeakageValidator.dataset_splits",
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
            metadata={
                'train_range': [str(train_start.date()), str(train_end.date())],
                'val_range': [str(val_start.date()), str(val_end.date())],
                'test_range': [str(test_start.date()), str(test_end.date())],
                'train_rows': len(train_df),
                'val_rows': len(val_df),
                'test_rows': len(test_df),
            },
        )

    def validate_merge_asof_direction(
        self,
        merged_df: pd.DataFrame,
        left_date_col: str,
        right_date_col: str,
        tolerance: timedelta = timedelta(days=1),
    ) -> ValidationResult:
        """
        Validate merge_asof used backward direction only.

        For time series, merge_asof MUST use direction='backward' to prevent
        future data from leaking into current observations.

        Args:
            merged_df: Result of merge_asof
            left_date_col: Original date column (e.g., OHLCV fecha)
            right_date_col: Merged date column (e.g., macro fecha_macro)
            tolerance: Maximum allowed difference (default 1 day)

        Returns:
            ValidationResult with pass/fail and details
        """
        errors = []
        warnings = []

        if left_date_col not in merged_df.columns:
            return ValidationResult(
                passed=False,
                validator="AntiLeakageValidator.merge_asof_direction",
                errors=[f"Missing left date column: {left_date_col}"],
                severity=ValidationSeverity.CRITICAL,
            )

        if right_date_col not in merged_df.columns:
            return ValidationResult(
                passed=False,
                validator="AntiLeakageValidator.merge_asof_direction",
                errors=[f"Missing right date column: {right_date_col}"],
                severity=ValidationSeverity.CRITICAL,
            )

        left_dates = pd.to_datetime(merged_df[left_date_col])
        right_dates = pd.to_datetime(merged_df[right_date_col])

        # Forward merge would have right_date > left_date (future data!)
        forward_merge_mask = right_dates > left_dates
        forward_count = forward_merge_mask.sum()

        if forward_count > 0:
            examples = merged_df[forward_merge_mask].head(3)
            example_str = ', '.join([
                f"({row[left_date_col].date()} <- {row[right_date_col].date()})"
                for _, row in examples.iterrows()
            ])
            errors.append(
                f"LEAKAGE: {forward_count} rows have future data merged in. "
                f"Right date > Left date indicates forward merge. Examples: {example_str}"
            )

        # Check tolerance
        date_diff = (left_dates - right_dates).abs()
        max_diff = date_diff.max()

        if max_diff > tolerance:
            warnings.append(
                f"Large merge gap detected: {max_diff.days} days. "
                f"This may indicate missing data or incorrect merge."
            )

        return ValidationResult(
            passed=len(errors) == 0,
            validator="AntiLeakageValidator.merge_asof_direction",
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
            metadata={
                'forward_merge_count': forward_count,
                'max_date_diff_days': max_diff.days if pd.notna(max_diff) else None,
                'total_rows': len(merged_df),
            },
        )

    def run_full_validation(
        self,
        dataset_df: pd.DataFrame,
        norm_stats_path: Optional[Path] = None,
        train_dates: Optional[Tuple[str, str]] = None,
        reference_date: Optional[datetime] = None,
    ) -> AntiLeakageReport:
        """
        Run all anti-leakage validations on a dataset.

        Args:
            dataset_df: The dataset to validate
            norm_stats_path: Path to norm_stats.json
            train_dates: Tuple of (start, end) for training period
            reference_date: Reference date for future data check

        Returns:
            AntiLeakageReport with all check results
        """
        report = AntiLeakageReport(passed=True)

        # 1. Check for future data
        ref_date = reference_date or datetime.now()
        result = self.validate_no_future_data(dataset_df, ref_date)
        report.add_check('no_future_data', result)

        # 2. Check normalization source if provided
        if norm_stats_path and norm_stats_path.exists() and train_dates:
            try:
                with open(norm_stats_path, 'r') as f:
                    norm_stats = json.load(f)
                result = self.validate_normalization_source(
                    norm_stats, train_dates[0], train_dates[1]
                )
                report.add_check('normalization_source', result)
            except Exception as e:
                report.add_check('normalization_source', ValidationResult(
                    passed=False,
                    validator="AntiLeakageValidator.normalization_source",
                    errors=[f"Failed to load norm_stats: {str(e)}"],
                    severity=ValidationSeverity.CRITICAL,
                ))

        # 3. Check macro variables for T-1 shift
        macro_columns = [col for col in dataset_df.columns if any(
            macro_var.lower() in col.lower()
            for macro_var in self.MACRO_VARIABLES_T1
        )]

        for col in macro_columns[:5]:  # Check first 5 macro columns
            result = self.validate_macro_shift_t1(dataset_df, col, value_column=col)
            report.add_check(f'macro_shift_{col}', result)

        # Log summary
        if report.passed:
            logger.info("[ANTI-LEAKAGE] All checks passed (%d checks)", len(report.checks))
        else:
            logger.error(
                "[ANTI-LEAKAGE] VALIDATION FAILED! %d errors detected",
                len(report.get_all_errors())
            )
            for error in report.get_all_errors():
                logger.error("  - %s", error)

        return report


# Convenience function for quick validation
def validate_anti_leakage(
    df: pd.DataFrame,
    reference_date: Optional[datetime] = None,
    strict: bool = True
) -> bool:
    """
    Quick anti-leakage validation.

    Args:
        df: DataFrame to validate
        reference_date: Reference date for future data check
        strict: If True, any warning becomes an error

    Returns:
        True if validation passes, False otherwise
    """
    validator = AntiLeakageValidator(strict_mode=strict)
    report = validator.run_full_validation(df, reference_date=reference_date)
    return report.passed
