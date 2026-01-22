# backend/src/data/alignment_validator.py
"""
Data Alignment Validation Module.

Provides comprehensive validation for data alignment between different sources,
ensuring column consistency, value integrity, and target calculation correctness.

Follows Single Responsibility Principle - only handles alignment validation.
"""

import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging

from ..core.config import PipelineConfig, HORIZONS
from ..core.exceptions import DataValidationError

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'


class ValidationCategory(Enum):
    """Categories of validation checks."""
    COLUMN_ALIGNMENT = 'column_alignment'
    VALUE_CONSISTENCY = 'value_consistency'
    TARGET_CALCULATION = 'target_calculation'
    DATA_QUALITY = 'data_quality'
    TEMPORAL_INTEGRITY = 'temporal_integrity'


@dataclass
class ValidationIssue:
    """
    Single validation issue found during alignment checks.
    """
    category: ValidationCategory
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    affected_rows: int = 0
    affected_columns: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.category.value}: {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'category': self.category.value,
            'severity': self.severity.value,
            'message': self.message,
            'details': self.details,
            'affected_rows': self.affected_rows,
            'affected_columns': self.affected_columns,
        }


@dataclass
class ColumnReport:
    """
    Report for column alignment validation.
    """
    df1_name: str
    df2_name: str
    df1_columns: List[str]
    df2_columns: List[str]
    common_columns: List[str]
    only_in_df1: List[str]
    only_in_df2: List[str]
    columns_matched: bool
    dtype_mismatches: Dict[str, Tuple[str, str]] = field(default_factory=dict)

    @property
    def match_percentage(self) -> float:
        """Percentage of columns that match."""
        total = len(set(self.df1_columns) | set(self.df2_columns))
        if total == 0:
            return 100.0
        return (len(self.common_columns) / total) * 100

    def summary(self) -> str:
        lines = [
            f"Column Alignment Report: {self.df1_name} vs {self.df2_name}",
            "-" * 50,
            f"  {self.df1_name} columns: {len(self.df1_columns)}",
            f"  {self.df2_name} columns: {len(self.df2_columns)}",
            f"  Common columns: {len(self.common_columns)}",
            f"  Only in {self.df1_name}: {len(self.only_in_df1)}",
            f"  Only in {self.df2_name}: {len(self.only_in_df2)}",
            f"  Match percentage: {self.match_percentage:.1f}%",
            f"  Dtype mismatches: {len(self.dtype_mismatches)}",
            f"  Status: {'MATCHED' if self.columns_matched else 'MISMATCHED'}",
        ]
        return "\n".join(lines)


@dataclass
class ValueReport:
    """
    Report for value consistency validation in overlapping data.
    """
    n_overlap_rows: int
    n_columns_checked: int
    n_mismatches: int
    mismatch_rate: float
    max_absolute_diff: float
    max_relative_diff: float
    mean_absolute_diff: float
    problematic_columns: List[str] = field(default_factory=list)
    sample_mismatches: List[Dict] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines = [
            "Value Consistency Report",
            "-" * 50,
            f"  Overlap rows checked: {self.n_overlap_rows}",
            f"  Columns checked: {self.n_columns_checked}",
            f"  Total mismatches: {self.n_mismatches}",
            f"  Mismatch rate: {self.mismatch_rate:.2%}",
            f"  Max absolute diff: {self.max_absolute_diff:.6f}",
            f"  Max relative diff: {self.max_relative_diff:.2%}",
            f"  Mean absolute diff: {self.mean_absolute_diff:.6f}",
            f"  Problematic columns: {len(self.problematic_columns)}",
            f"  Status: {'PASSED' if self.passed else 'FAILED'}",
        ]
        return "\n".join(lines)


@dataclass
class TargetReport:
    """
    Report for target calculation validation.
    """
    horizons_checked: List[int]
    n_rows_checked: int
    correct_calculations: int
    incorrect_calculations: int
    missing_targets: int
    accuracy_rate: float
    issues_by_horizon: Dict[int, int] = field(default_factory=dict)
    sample_errors: List[Dict] = field(default_factory=list)
    passed: bool = True

    def summary(self) -> str:
        lines = [
            "Target Calculation Report",
            "-" * 50,
            f"  Horizons checked: {self.horizons_checked}",
            f"  Rows checked: {self.n_rows_checked}",
            f"  Correct calculations: {self.correct_calculations}",
            f"  Incorrect calculations: {self.incorrect_calculations}",
            f"  Missing targets: {self.missing_targets}",
            f"  Accuracy rate: {self.accuracy_rate:.2%}",
            f"  Status: {'PASSED' if self.passed else 'FAILED'}",
        ]
        return "\n".join(lines)


@dataclass
class AlignmentReport:
    """
    Complete alignment validation report.
    """
    generated_at: datetime = field(default_factory=datetime.now)
    column_report: Optional[ColumnReport] = None
    value_report: Optional[ValueReport] = None
    target_report: Optional[TargetReport] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    overall_passed: bool = True
    error_message: Optional[str] = None

    @property
    def n_critical(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.CRITICAL)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == ValidationSeverity.WARNING)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "ALIGNMENT VALIDATION REPORT",
            "=" * 60,
            f"Generated at: {self.generated_at.isoformat()}",
            "",
        ]

        if self.column_report:
            lines.append(self.column_report.summary())
            lines.append("")

        if self.value_report:
            lines.append(self.value_report.summary())
            lines.append("")

        if self.target_report:
            lines.append(self.target_report.summary())
            lines.append("")

        lines.extend([
            "Issues Summary:",
            f"  Critical: {self.n_critical}",
            f"  Errors: {self.n_errors}",
            f"  Warnings: {self.n_warnings}",
            "",
            f"OVERALL STATUS: {'PASSED' if self.overall_passed else 'FAILED'}",
        ])

        if self.error_message:
            lines.append(f"Error: {self.error_message}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            'generated_at': self.generated_at.isoformat(),
            'overall_passed': self.overall_passed,
            'n_critical': self.n_critical,
            'n_errors': self.n_errors,
            'n_warnings': self.n_warnings,
            'error_message': self.error_message,
            'issues': [i.to_dict() for i in self.issues],
        }


class AlignmentValidator:
    """
    Validates data alignment between different sources.

    Responsibilities:
    - Validate column alignment between DataFrames
    - Validate value consistency in overlapping date ranges
    - Validate target calculations (forward returns)
    - Generate comprehensive alignment reports

    Usage:
        validator = AlignmentValidator(config)
        col_report = validator.validate_columns_match(df1, df2)
        val_report = validator.validate_values_in_overlap(df1, df2, 'date')
        tgt_report = validator.validate_target_calculation(df, 'Close', [1, 5, 10])
        full_report = validator.generate_report()
    """

    # Tolerance for floating point comparisons
    ABSOLUTE_TOLERANCE = 1e-6
    RELATIVE_TOLERANCE = 1e-4

    # Columns that are expected to differ (e.g., auto-generated IDs)
    IGNORE_COLUMNS = {'id', 'created_at', 'updated_at', 'row_id', 'index'}

    def __init__(self, config: PipelineConfig = None):
        """
        Initialize AlignmentValidator.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        self._issues: List[ValidationIssue] = []
        self._column_report: Optional[ColumnReport] = None
        self._value_report: Optional[ValueReport] = None
        self._target_report: Optional[TargetReport] = None

    def validate_columns_match(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        df1_name: str = 'df1',
        df2_name: str = 'df2',
        strict: bool = False
    ) -> ColumnReport:
        """
        Validate that columns match between two DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame
            df1_name: Name for df1 in reports
            df2_name: Name for df2 in reports
            strict: If True, require exact column match

        Returns:
            ColumnReport with detailed column alignment information
        """
        logger.info(f"Validating columns: {df1_name} vs {df2_name}")

        cols1 = set(df1.columns) - self.IGNORE_COLUMNS
        cols2 = set(df2.columns) - self.IGNORE_COLUMNS

        common = cols1 & cols2
        only_in_1 = cols1 - cols2
        only_in_2 = cols2 - cols1

        # Check dtype mismatches for common columns
        dtype_mismatches = {}
        for col in common:
            dtype1 = str(df1[col].dtype)
            dtype2 = str(df2[col].dtype)
            if dtype1 != dtype2:
                # Check if they're compatible (e.g., float64 vs float32)
                if not self._dtypes_compatible(dtype1, dtype2):
                    dtype_mismatches[col] = (dtype1, dtype2)

        # Determine if columns match
        if strict:
            columns_matched = len(only_in_1) == 0 and len(only_in_2) == 0
        else:
            # Non-strict: allow extra columns in either DataFrame
            columns_matched = len(common) > 0

        report = ColumnReport(
            df1_name=df1_name,
            df2_name=df2_name,
            df1_columns=list(df1.columns),
            df2_columns=list(df2.columns),
            common_columns=sorted(common),
            only_in_df1=sorted(only_in_1),
            only_in_df2=sorted(only_in_2),
            columns_matched=columns_matched,
            dtype_mismatches=dtype_mismatches,
        )

        # Add issues
        if only_in_1:
            self._issues.append(ValidationIssue(
                category=ValidationCategory.COLUMN_ALIGNMENT,
                severity=ValidationSeverity.WARNING,
                message=f"Columns only in {df1_name}: {len(only_in_1)}",
                details={'columns': list(only_in_1)[:10]},
                affected_columns=list(only_in_1),
            ))

        if only_in_2:
            self._issues.append(ValidationIssue(
                category=ValidationCategory.COLUMN_ALIGNMENT,
                severity=ValidationSeverity.WARNING,
                message=f"Columns only in {df2_name}: {len(only_in_2)}",
                details={'columns': list(only_in_2)[:10]},
                affected_columns=list(only_in_2),
            ))

        if dtype_mismatches:
            self._issues.append(ValidationIssue(
                category=ValidationCategory.COLUMN_ALIGNMENT,
                severity=ValidationSeverity.INFO,
                message=f"Dtype mismatches in {len(dtype_mismatches)} columns",
                details={'mismatches': dtype_mismatches},
                affected_columns=list(dtype_mismatches.keys()),
            ))

        self._column_report = report
        logger.info(f"Column validation: {len(common)} common, {len(only_in_1)} only in df1, {len(only_in_2)} only in df2")

        return report

    def validate_values_in_overlap(
        self,
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        date_col: str = 'date',
        columns: List[str] = None,
        tolerance_abs: float = None,
        tolerance_rel: float = None
    ) -> ValueReport:
        """
        Validate that values match in overlapping date ranges.

        Args:
            df1: First DataFrame (historical)
            df2: Second DataFrame (recent)
            date_col: Name of date column (or 'index' to use DataFrame index)
            columns: Specific columns to validate (default: all numeric)
            tolerance_abs: Absolute tolerance for comparison
            tolerance_rel: Relative tolerance for comparison

        Returns:
            ValueReport with detailed value consistency information
        """
        logger.info("Validating values in overlap")

        tolerance_abs = tolerance_abs or self.ABSOLUTE_TOLERANCE
        tolerance_rel = tolerance_rel or self.RELATIVE_TOLERANCE

        # Get dates from index or column
        if date_col == 'index' or date_col not in df1.columns:
            dates1 = set(df1.index.date if hasattr(df1.index, 'date') else df1.index)
            dates2 = set(df2.index.date if hasattr(df2.index, 'date') else df2.index)
            use_index = True
        else:
            dates1 = set(pd.to_datetime(df1[date_col]).dt.date)
            dates2 = set(pd.to_datetime(df2[date_col]).dt.date)
            use_index = False

        overlap_dates = sorted(dates1 & dates2)

        if not overlap_dates:
            logger.info("No overlapping dates found")
            report = ValueReport(
                n_overlap_rows=0,
                n_columns_checked=0,
                n_mismatches=0,
                mismatch_rate=0.0,
                max_absolute_diff=0.0,
                max_relative_diff=0.0,
                mean_absolute_diff=0.0,
                passed=True
            )
            self._value_report = report
            return report

        # Determine columns to check
        common_cols = set(df1.columns) & set(df2.columns) - self.IGNORE_COLUMNS
        if columns:
            check_cols = [c for c in columns if c in common_cols]
        else:
            # Check all numeric columns
            check_cols = [
                c for c in common_cols
                if pd.api.types.is_numeric_dtype(df1[c]) and pd.api.types.is_numeric_dtype(df2[c])
            ]

        # Compare values
        mismatches = []
        abs_diffs = []
        rel_diffs = []
        problematic_cols = set()

        for dt in overlap_dates:
            # Get rows for this date
            if use_index:
                if hasattr(df1.index, 'date'):
                    mask1 = df1.index.date == dt
                    mask2 = df2.index.date == dt
                else:
                    mask1 = df1.index == dt
                    mask2 = df2.index == dt
            else:
                mask1 = pd.to_datetime(df1[date_col]).dt.date == dt
                mask2 = pd.to_datetime(df2[date_col]).dt.date == dt

            row1 = df1[mask1]
            row2 = df2[mask2]

            if row1.empty or row2.empty:
                continue

            for col in check_cols:
                try:
                    val1 = row1[col].iloc[0]
                    val2 = row2[col].iloc[0]

                    # Skip NaN comparisons
                    if pd.isna(val1) and pd.isna(val2):
                        continue

                    # Handle NaN mismatch
                    if pd.isna(val1) != pd.isna(val2):
                        mismatches.append({
                            'date': dt,
                            'column': col,
                            'val1': val1,
                            'val2': val2,
                            'abs_diff': None,
                            'rel_diff': None,
                        })
                        problematic_cols.add(col)
                        continue

                    # Calculate differences
                    abs_diff = abs(float(val1) - float(val2))
                    rel_diff = abs_diff / abs(float(val1)) if val1 != 0 else abs_diff

                    abs_diffs.append(abs_diff)
                    rel_diffs.append(rel_diff)

                    # Check tolerance
                    if abs_diff > tolerance_abs and rel_diff > tolerance_rel:
                        mismatches.append({
                            'date': str(dt),
                            'column': col,
                            'val1': float(val1),
                            'val2': float(val2),
                            'abs_diff': abs_diff,
                            'rel_diff': rel_diff,
                        })
                        problematic_cols.add(col)

                except (TypeError, ValueError):
                    continue

        # Calculate statistics
        n_mismatches = len(mismatches)
        total_comparisons = len(overlap_dates) * len(check_cols)
        mismatch_rate = n_mismatches / total_comparisons if total_comparisons > 0 else 0.0

        max_abs = max(abs_diffs) if abs_diffs else 0.0
        max_rel = max(rel_diffs) if rel_diffs else 0.0
        mean_abs = np.mean(abs_diffs) if abs_diffs else 0.0

        # Determine pass/fail
        passed = mismatch_rate < 0.05  # Less than 5% mismatches

        report = ValueReport(
            n_overlap_rows=len(overlap_dates),
            n_columns_checked=len(check_cols),
            n_mismatches=n_mismatches,
            mismatch_rate=mismatch_rate,
            max_absolute_diff=max_abs,
            max_relative_diff=max_rel,
            mean_absolute_diff=mean_abs,
            problematic_columns=sorted(problematic_cols),
            sample_mismatches=mismatches[:10],  # First 10 mismatches
            passed=passed,
        )

        if n_mismatches > 0:
            severity = ValidationSeverity.ERROR if n_mismatches > 10 else ValidationSeverity.WARNING
            self._issues.append(ValidationIssue(
                category=ValidationCategory.VALUE_CONSISTENCY,
                severity=severity,
                message=f"Found {n_mismatches} value mismatches ({mismatch_rate:.2%})",
                details={'sample_mismatches': mismatches[:5]},
                affected_rows=len(overlap_dates),
                affected_columns=sorted(problematic_cols),
            ))

        self._value_report = report
        logger.info(f"Value validation: {n_mismatches} mismatches in {len(overlap_dates)} overlap rows")

        return report

    def validate_target_calculation(
        self,
        df: pd.DataFrame,
        price_col: str,
        horizons: List[int] = None,
        tolerance: float = 1e-8
    ) -> TargetReport:
        """
        Validate that target columns (forward returns) are calculated correctly.

        Target formula: target_Hd = log(price[t+H] / price[t])

        Args:
            df: DataFrame with price and target columns
            price_col: Name of the price column
            horizons: List of horizons to check (default: from config)
            tolerance: Tolerance for floating point comparison

        Returns:
            TargetReport with detailed target validation information
        """
        horizons = horizons or self.config.horizons
        logger.info(f"Validating target calculations for horizons: {horizons}")

        if price_col not in df.columns:
            self._issues.append(ValidationIssue(
                category=ValidationCategory.TARGET_CALCULATION,
                severity=ValidationSeverity.CRITICAL,
                message=f"Price column '{price_col}' not found",
                details={'available_columns': list(df.columns)[:20]},
            ))
            report = TargetReport(
                horizons_checked=[],
                n_rows_checked=0,
                correct_calculations=0,
                incorrect_calculations=0,
                missing_targets=0,
                accuracy_rate=0.0,
                passed=False,
            )
            self._target_report = report
            return report

        prices = df[price_col]
        n_rows = len(df)
        correct = 0
        incorrect = 0
        missing = 0
        issues_by_horizon = {}
        sample_errors = []

        for h in horizons:
            target_col = f'target_{h}d'
            issues_by_horizon[h] = 0

            if target_col not in df.columns:
                missing += n_rows
                self._issues.append(ValidationIssue(
                    category=ValidationCategory.TARGET_CALCULATION,
                    severity=ValidationSeverity.WARNING,
                    message=f"Target column '{target_col}' not found",
                    affected_columns=[target_col],
                ))
                continue

            # Calculate expected targets
            expected = np.log(prices.shift(-h) / prices)
            actual = df[target_col]

            # Compare values
            for i in range(len(df) - h):
                if pd.isna(expected.iloc[i]) or pd.isna(actual.iloc[i]):
                    continue

                diff = abs(expected.iloc[i] - actual.iloc[i])
                if diff <= tolerance:
                    correct += 1
                else:
                    incorrect += 1
                    issues_by_horizon[h] += 1
                    if len(sample_errors) < 10:
                        sample_errors.append({
                            'horizon': h,
                            'row': i,
                            'expected': float(expected.iloc[i]),
                            'actual': float(actual.iloc[i]),
                            'diff': diff,
                        })

        total_checked = correct + incorrect
        accuracy_rate = correct / total_checked if total_checked > 0 else 0.0
        passed = accuracy_rate >= 0.99  # 99% accuracy required

        report = TargetReport(
            horizons_checked=horizons,
            n_rows_checked=total_checked,
            correct_calculations=correct,
            incorrect_calculations=incorrect,
            missing_targets=missing,
            accuracy_rate=accuracy_rate,
            issues_by_horizon=issues_by_horizon,
            sample_errors=sample_errors,
            passed=passed,
        )

        if incorrect > 0:
            self._issues.append(ValidationIssue(
                category=ValidationCategory.TARGET_CALCULATION,
                severity=ValidationSeverity.ERROR if incorrect > 10 else ValidationSeverity.WARNING,
                message=f"Found {incorrect} incorrect target calculations",
                details={'accuracy_rate': accuracy_rate, 'issues_by_horizon': issues_by_horizon},
                affected_rows=incorrect,
            ))

        self._target_report = report
        logger.info(f"Target validation: {correct} correct, {incorrect} incorrect ({accuracy_rate:.2%} accuracy)")

        return report

    def generate_report(self) -> AlignmentReport:
        """
        Generate comprehensive alignment report from all validations performed.

        Returns:
            AlignmentReport with all validation results
        """
        # Determine overall pass/fail
        has_critical = any(i.severity == ValidationSeverity.CRITICAL for i in self._issues)
        has_errors = sum(1 for i in self._issues if i.severity == ValidationSeverity.ERROR)

        overall_passed = not has_critical and has_errors < 5

        if self._column_report and not self._column_report.columns_matched:
            overall_passed = False
        if self._value_report and not self._value_report.passed:
            overall_passed = False
        if self._target_report and not self._target_report.passed:
            overall_passed = False

        error_message = None
        if not overall_passed:
            if has_critical:
                error_message = f"Critical issues found: {self.n_critical}"
            elif has_errors >= 5:
                error_message = f"Too many errors: {has_errors}"
            else:
                error_message = "Validation checks failed"

        report = AlignmentReport(
            column_report=self._column_report,
            value_report=self._value_report,
            target_report=self._target_report,
            issues=self._issues.copy(),
            overall_passed=overall_passed,
            error_message=error_message,
        )

        return report

    def clear(self):
        """Clear all stored validation results."""
        self._issues = []
        self._column_report = None
        self._value_report = None
        self._target_report = None

    @property
    def n_critical(self) -> int:
        return sum(1 for i in self._issues if i.severity == ValidationSeverity.CRITICAL)

    def _dtypes_compatible(self, dtype1: str, dtype2: str) -> bool:
        """Check if two dtypes are compatible for comparison."""
        # Numeric types are compatible with each other
        numeric_types = {'float64', 'float32', 'int64', 'int32', 'int16', 'int8'}
        if dtype1 in numeric_types and dtype2 in numeric_types:
            return True
        # Object and string are compatible
        if dtype1 in {'object', 'string'} and dtype2 in {'object', 'string'}:
            return True
        return dtype1 == dtype2


def validate_alignment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    price_col: str = 'Close',
    horizons: List[int] = None,
    config: PipelineConfig = None
) -> AlignmentReport:
    """
    Convenience function to perform full alignment validation.

    Args:
        df1: First DataFrame (historical)
        df2: Second DataFrame (recent)
        price_col: Name of price column for target validation
        horizons: List of horizons to validate
        config: Pipeline configuration

    Returns:
        AlignmentReport with all validation results
    """
    validator = AlignmentValidator(config)

    # Run all validations
    validator.validate_columns_match(df1, df2, 'historical', 'recent')
    validator.validate_values_in_overlap(df1, df2)

    # Validate targets on the first DataFrame if it has them
    target_cols = [c for c in df1.columns if c.startswith('target_')]
    if target_cols and price_col in df1.columns:
        validator.validate_target_calculation(df1, price_col, horizons)

    return validator.generate_report()
