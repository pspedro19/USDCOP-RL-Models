# -*- coding: utf-8 -*-
"""
L0 Data Validators
==================
Validation strategies for L0 macro data quality assurance.

Contract: CTR-L0-VALIDATOR-001

Design Patterns:
- Strategy Pattern: Interchangeable validators
- Chain of Responsibility: ValidationPipeline processes multiple validators
- Template Method: Base class defines validate interface

SOLID Principles:
- Single Responsibility: Each validator checks one aspect
- Open/Closed: New validators via inheritance
- Liskov Substitution: All validators are interchangeable
- Interface Segregation: Minimal validator interface
- Dependency Inversion: Depends on abstractions (DataValidator)

Version: 1.0.0
"""

import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation errors."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    passed: bool
    validator: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    severity: ValidationSeverity = ValidationSeverity.WARNING
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.passed and not self.errors:
            self.errors = ["Validation failed (no specific message)"]


@dataclass
class ValidationReport:
    """Aggregated validation report from multiple validators."""
    variable: str
    results: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.results:
            self.overall_passed = all(r.passed for r in self.results)

    def get_all_errors(self) -> List[str]:
        """Get all error messages from all validators."""
        errors = []
        for result in self.results:
            for error in result.errors:
                errors.append(f"[{result.validator}] {error}")
        return errors

    def get_all_warnings(self) -> List[str]:
        """Get all warning messages from all validators."""
        warnings = []
        for result in self.results:
            for warning in result.warnings:
                warnings.append(f"[{result.validator}] {warning}")
        return warnings

    def get_critical_failures(self) -> List[ValidationResult]:
        """Get only critical failures."""
        return [
            r for r in self.results
            if not r.passed and r.severity == ValidationSeverity.CRITICAL
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'variable': self.variable,
            'overall_passed': self.overall_passed,
            'timestamp': self.timestamp.isoformat(),
            'errors': self.get_all_errors(),
            'warnings': self.get_all_warnings(),
            'results_count': len(self.results),
            'passed_count': sum(1 for r in self.results if r.passed),
            'failed_count': sum(1 for r in self.results if not r.passed),
            'metadata': self.metadata,
        }


class DataValidator(ABC):
    """Abstract base class for data validators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Validator name for reporting."""
        pass

    @abstractmethod
    def validate(self, df: pd.DataFrame, variable: str) -> ValidationResult:
        """
        Validate data for a specific variable.

        Args:
            df: DataFrame with 'fecha' column and variable column
            variable: Column name to validate

        Returns:
            ValidationResult with pass/fail and error messages
        """
        pass


class SchemaValidator(DataValidator):
    """
    Validates column types and NOT NULL constraints.

    Checks:
    - 'fecha' column exists and is datetime
    - Variable column exists and is numeric
    - No NULL values in critical columns
    """

    @property
    def name(self) -> str:
        return "SchemaValidator"

    def validate(self, df: pd.DataFrame, variable: str) -> ValidationResult:
        errors = []
        warnings = []

        # Check DataFrame is not empty
        if df is None or df.empty:
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=["DataFrame is empty or None"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Check 'fecha' column exists
        if 'fecha' not in df.columns:
            errors.append("Missing 'fecha' column")
        else:
            # Check 'fecha' is datetime type
            if not pd.api.types.is_datetime64_any_dtype(df['fecha']):
                try:
                    pd.to_datetime(df['fecha'])
                    warnings.append("'fecha' column is not datetime type but can be converted")
                except Exception:
                    errors.append("'fecha' column is not datetime and cannot be converted")

        # Check variable column exists
        if variable not in df.columns:
            errors.append(f"Missing '{variable}' column")
        else:
            # Check variable is numeric
            if not pd.api.types.is_numeric_dtype(df[variable]):
                errors.append(f"'{variable}' column is not numeric type")

            # Check for NULLs
            null_count = df[variable].isna().sum()
            total_count = len(df)
            if null_count > 0:
                null_pct = (null_count / total_count) * 100
                if null_pct > 50:
                    errors.append(f"{null_count} NULL values ({null_pct:.1f}%) in '{variable}'")
                else:
                    warnings.append(f"{null_count} NULL values ({null_pct:.1f}%) in '{variable}'")

        return ValidationResult(
            passed=len(errors) == 0,
            validator=self.name,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
            metadata={'variable': variable, 'row_count': len(df)},
        )


class RangeValidator(DataValidator):
    """
    Validates values are within expected ranges.

    Loads ranges from config/l0_macro_sources.yaml validation.ranges section.
    """

    def __init__(self, ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        """
        Initialize with optional custom ranges.

        Args:
            ranges: Dict mapping variable -> (min, max) tuple
                    If None, loads from l0_macro_sources.yaml
        """
        self._ranges = ranges or self._load_ranges_from_config()

    def _load_ranges_from_config(self) -> Dict[str, Tuple[float, float]]:
        """Load validation ranges from l0_macro_sources.yaml."""
        # Try multiple paths
        config_paths = [
            Path('/opt/airflow/config/l0_macro_sources.yaml'),
            Path(__file__).parent.parent.parent.parent / 'config' / 'l0_macro_sources.yaml',
        ]

        for config_path in config_paths:
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)

                    ranges_config = config.get('validation', {}).get('ranges', {})
                    ranges = {}
                    for var, range_list in ranges_config.items():
                        if isinstance(range_list, list) and len(range_list) == 2:
                            ranges[var] = (float(range_list[0]), float(range_list[1]))
                    logger.info(f"Loaded {len(ranges)} validation ranges from config")
                    return ranges
                except Exception as e:
                    logger.warning(f"Failed to load ranges from {config_path}: {e}")

        logger.warning("No validation ranges loaded from config")
        return {}

    @property
    def name(self) -> str:
        return "RangeValidator"

    def validate(self, df: pd.DataFrame, variable: str) -> ValidationResult:
        errors = []
        warnings = []

        if variable not in df.columns:
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=[f"Column '{variable}' not found in DataFrame"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Get range for this variable
        if variable not in self._ranges:
            return ValidationResult(
                passed=True,
                validator=self.name,
                warnings=[f"No range defined for '{variable}', skipping range validation"],
                severity=ValidationSeverity.INFO,
                metadata={'variable': variable, 'has_range': False},
            )

        min_val, max_val = self._ranges[variable]
        values = df[variable].dropna()

        if len(values) == 0:
            return ValidationResult(
                passed=True,
                validator=self.name,
                warnings=["No non-null values to validate"],
                severity=ValidationSeverity.INFO,
            )

        # Check for out-of-range values
        below_min = values[values < min_val]
        above_max = values[values > max_val]
        out_of_range_count = len(below_min) + len(above_max)

        if out_of_range_count > 0:
            total_count = len(values)
            out_of_range_pct = (out_of_range_count / total_count) * 100

            # Get examples of bad values
            examples = []
            if len(below_min) > 0:
                examples.append(f"below {min_val}: {below_min.min():.2f}")
            if len(above_max) > 0:
                examples.append(f"above {max_val}: {above_max.max():.2f}")

            message = (
                f"{out_of_range_count} values ({out_of_range_pct:.1f}%) outside range "
                f"[{min_val}, {max_val}] for '{variable}'. Examples: {', '.join(examples)}"
            )

            if out_of_range_pct > 10:
                errors.append(message)
            else:
                warnings.append(message)

        return ValidationResult(
            passed=len(errors) == 0,
            validator=self.name,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.WARNING if errors else ValidationSeverity.INFO,
            metadata={
                'variable': variable,
                'range': [min_val, max_val],
                'out_of_range_count': out_of_range_count,
                'total_count': len(values),
            },
        )


class CompletenessValidator(DataValidator):
    """
    Validates data completeness (percentage of non-null values).

    Thresholds by frequency:
    - daily: 95% minimum
    - monthly: 98% minimum
    - quarterly: 99% minimum
    """

    THRESHOLDS = {
        'daily': 0.95,
        'monthly': 0.98,
        'quarterly': 0.99,
    }

    def __init__(self, frequency: str = 'daily', custom_threshold: Optional[float] = None):
        """
        Initialize with frequency or custom threshold.

        Args:
            frequency: 'daily', 'monthly', or 'quarterly'
            custom_threshold: Override threshold (0.0 to 1.0)
        """
        self.frequency = frequency
        self.threshold = custom_threshold or self.THRESHOLDS.get(frequency, 0.95)

    @property
    def name(self) -> str:
        return "CompletenessValidator"

    def _infer_frequency(self, variable: str) -> str:
        """Infer frequency from variable name."""
        if '_m_' in variable:
            return 'monthly'
        elif '_q_' in variable:
            return 'quarterly'
        elif '_a_' in variable:
            return 'annual'
        return 'daily'

    def validate(self, df: pd.DataFrame, variable: str) -> ValidationResult:
        errors = []
        warnings = []

        if variable not in df.columns:
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=[f"Column '{variable}' not found in DataFrame"],
                severity=ValidationSeverity.CRITICAL,
            )

        total_count = len(df)
        if total_count == 0:
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=["DataFrame is empty"],
                severity=ValidationSeverity.CRITICAL,
            )

        non_null_count = df[variable].notna().sum()
        completeness = non_null_count / total_count

        # Use variable-inferred frequency if not set
        freq = self._infer_frequency(variable) if self.frequency == 'daily' else self.frequency
        threshold = self.THRESHOLDS.get(freq, self.threshold)

        metadata = {
            'variable': variable,
            'completeness': completeness,
            'threshold': threshold,
            'frequency': freq,
            'total_rows': total_count,
            'non_null_rows': non_null_count,
        }

        if completeness < threshold:
            message = (
                f"Completeness {completeness:.1%} below threshold {threshold:.1%} "
                f"for '{variable}' ({non_null_count}/{total_count} rows)"
            )
            errors.append(message)

        return ValidationResult(
            passed=len(errors) == 0,
            validator=self.name,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.WARNING if errors else ValidationSeverity.INFO,
            metadata=metadata,
        )


class LeakageValidator(DataValidator):
    """
    Detects future data leakage.

    Checks if any dates in the data are in the future.
    This is CRITICAL for ML models to prevent look-ahead bias.
    """

    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize with optional reference date.

        Args:
            reference_date: Date to use as "now". Defaults to current date.
        """
        self.reference_date = reference_date or datetime.now()

    @property
    def name(self) -> str:
        return "LeakageValidator"

    def validate(self, df: pd.DataFrame, variable: str) -> ValidationResult:
        errors = []
        warnings = []

        if 'fecha' not in df.columns:
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=["Missing 'fecha' column, cannot check for leakage"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Convert fecha to datetime if needed
        fecha_col = pd.to_datetime(df['fecha'], errors='coerce')
        reference = pd.Timestamp(self.reference_date.date())

        # Find future dates
        future_mask = fecha_col.dt.date > reference.date()
        future_count = future_mask.sum()

        if future_count > 0:
            future_dates = fecha_col[future_mask].dt.strftime('%Y-%m-%d').unique()[:5]
            errors.append(
                f"LEAKAGE DETECTED: {future_count} rows with future dates. "
                f"Examples: {', '.join(future_dates)}"
            )

        return ValidationResult(
            passed=len(errors) == 0,
            validator=self.name,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.CRITICAL if errors else ValidationSeverity.INFO,
            metadata={
                'variable': variable,
                'reference_date': self.reference_date.strftime('%Y-%m-%d'),
                'future_rows_count': future_count,
            },
        )


class FreshnessValidator(DataValidator):
    """
    Validates data freshness.

    Checks if the most recent data is within expected recency window.
    """

    # Max days of staleness by frequency
    STALENESS_THRESHOLDS = {
        'daily': 3,      # Max 3 days old
        'monthly': 45,   # Max 45 days old
        'quarterly': 120,  # Max 120 days old
    }

    def __init__(
        self,
        reference_date: Optional[datetime] = None,
        max_staleness_days: Optional[int] = None
    ):
        """
        Initialize with optional parameters.

        Args:
            reference_date: Date to use as "now". Defaults to current date.
            max_staleness_days: Override staleness threshold
        """
        self.reference_date = reference_date or datetime.now()
        self.max_staleness_days = max_staleness_days

    @property
    def name(self) -> str:
        return "FreshnessValidator"

    def _infer_frequency(self, variable: str) -> str:
        """Infer frequency from variable name."""
        if '_m_' in variable:
            return 'monthly'
        elif '_q_' in variable:
            return 'quarterly'
        elif '_a_' in variable:
            return 'annual'
        return 'daily'

    def validate(self, df: pd.DataFrame, variable: str) -> ValidationResult:
        errors = []
        warnings = []

        if 'fecha' not in df.columns:
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=["Missing 'fecha' column"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Get most recent date
        fecha_col = pd.to_datetime(df['fecha'], errors='coerce')
        most_recent = fecha_col.max()

        if pd.isna(most_recent):
            return ValidationResult(
                passed=False,
                validator=self.name,
                errors=["No valid dates found"],
                severity=ValidationSeverity.CRITICAL,
            )

        # Calculate staleness
        reference = pd.Timestamp(self.reference_date)
        staleness_days = (reference - most_recent).days

        # Get threshold
        freq = self._infer_frequency(variable)
        threshold = self.max_staleness_days or self.STALENESS_THRESHOLDS.get(freq, 7)

        metadata = {
            'variable': variable,
            'most_recent_date': most_recent.strftime('%Y-%m-%d'),
            'reference_date': self.reference_date.strftime('%Y-%m-%d'),
            'staleness_days': staleness_days,
            'threshold_days': threshold,
            'frequency': freq,
        }

        if staleness_days > threshold:
            message = (
                f"Data is {staleness_days} days stale (threshold: {threshold} days). "
                f"Most recent date: {most_recent.strftime('%Y-%m-%d')}"
            )
            warnings.append(message)

        return ValidationResult(
            passed=True,  # Staleness is a warning, not a hard failure
            validator=self.name,
            errors=errors,
            warnings=warnings,
            severity=ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO,
            metadata=metadata,
        )


class ValidationPipeline:
    """
    Orchestrates multiple validators.

    Runs all validators and aggregates results into a single report.
    """

    def __init__(
        self,
        validators: Optional[List[DataValidator]] = None,
        fail_fast: bool = False
    ):
        """
        Initialize pipeline with validators.

        Args:
            validators: List of validators to run. Defaults to all standard validators.
            fail_fast: If True, stop on first failure.
        """
        self.validators = validators or [
            SchemaValidator(),
            RangeValidator(),
            CompletenessValidator(),
            LeakageValidator(),
            FreshnessValidator(),
        ]
        self.fail_fast = fail_fast

    def validate(
        self,
        df: pd.DataFrame,
        variable: str,
        **kwargs
    ) -> ValidationReport:
        """
        Run all validators and return aggregated report.

        Args:
            df: DataFrame to validate
            variable: Column name to validate
            **kwargs: Additional context passed to validators

        Returns:
            ValidationReport with all results
        """
        results = []
        overall_passed = True

        for validator in self.validators:
            try:
                result = validator.validate(df, variable)
                results.append(result)

                if not result.passed:
                    overall_passed = False
                    if self.fail_fast:
                        break

            except Exception as e:
                logger.error(f"Validator {validator.name} failed with error: {e}")
                results.append(ValidationResult(
                    passed=False,
                    validator=validator.name,
                    errors=[f"Validator exception: {str(e)}"],
                    severity=ValidationSeverity.CRITICAL,
                ))
                overall_passed = False
                if self.fail_fast:
                    break

        report = ValidationReport(
            variable=variable,
            results=results,
            overall_passed=overall_passed,
            metadata=kwargs,
        )

        # Log summary
        if overall_passed:
            logger.info(f"[VALIDATION] {variable}: PASSED ({len(results)} checks)")
        else:
            error_count = sum(len(r.errors) for r in results)
            logger.warning(f"[VALIDATION] {variable}: FAILED ({error_count} errors)")
            for error in report.get_all_errors():
                logger.warning(f"  - {error}")

        return report

    def validate_batch(
        self,
        df: pd.DataFrame,
        variables: List[str]
    ) -> Dict[str, ValidationReport]:
        """
        Validate multiple variables in one DataFrame.

        Args:
            df: DataFrame containing all variables
            variables: List of column names to validate

        Returns:
            Dict mapping variable name to ValidationReport
        """
        reports = {}
        for variable in variables:
            reports[variable] = self.validate(df, variable)
        return reports
