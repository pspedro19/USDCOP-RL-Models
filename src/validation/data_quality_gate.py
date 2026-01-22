"""
Data Quality Gate - L2 → L3 Validation
=======================================

This module provides comprehensive data quality validation between
L2 (preprocessing) and L3 (training) pipeline stages.

Validates:
- NaN/null percentages
- Row counts
- Feature counts and order
- Value ranges
- Date monotonicity (no future leakage)
- Distribution checks
- Duplicate detection

Author: Trading Team
Version: 1.0.0
Date: 2026-01-18
Contract: CTR-DQ-001
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


class ValidationSeverity(str, Enum):
    """Severity level for validation failures."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class DataQualityReport:
    """Complete data quality validation report."""
    dataset_path: str
    validated_at: datetime
    overall_status: ValidationStatus
    checks: List[ValidationCheck]
    summary: Dict[str, Any]

    @property
    def passed(self) -> bool:
        return self.overall_status == ValidationStatus.PASSED

    @property
    def failed_checks(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == ValidationStatus.FAILED]

    @property
    def warning_checks(self) -> List[ValidationCheck]:
        return [c for c in self.checks if c.status == ValidationStatus.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "validated_at": self.validated_at.isoformat(),
            "overall_status": self.overall_status.value,
            "passed": self.passed,
            "total_checks": len(self.checks),
            "passed_checks": len([c for c in self.checks if c.status == ValidationStatus.PASSED]),
            "failed_checks": len(self.failed_checks),
            "warning_checks": len(self.warning_checks),
            "checks": [c.to_dict() for c in self.checks],
            "summary": self.summary,
        }


class DataQualityError(Exception):
    """Raised when data quality validation fails."""

    def __init__(self, message: str, report: DataQualityReport):
        super().__init__(message)
        self.report = report


class DataQualityGate:
    """
    Data Quality Gate for L2 → L3 validation.

    Ensures training data meets quality standards before
    passing to the training pipeline.

    Usage:
        gate = DataQualityGate()
        report = gate.validate(dataset_path)
        if not report.passed:
            raise DataQualityError("Validation failed", report)
    """

    # Default configuration
    DEFAULT_CONFIG = {
        "max_nan_percentage": 0.05,
        "min_row_count": 100000,
        "expected_feature_count": 15,
        "require_monotonic_dates": True,
        "max_zscore": 10.0,
        "allow_duplicate_timestamps": False,
        "feature_ranges": {
            "rsi_9": {"min": 0, "max": 100},
            "atr_pct": {"min": 0, "max": None},
            "adx_14": {"min": 0, "max": 100},
            "log_ret_5m": {"min": -0.10, "max": 0.10},
            "log_ret_1h": {"min": -0.20, "max": 0.20},
            "log_ret_4h": {"min": -0.30, "max": 0.30},
        },
    }

    # Expected feature order (from SSOT)
    EXPECTED_FEATURES = [
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data quality gate.

        Args:
            config: Override default configuration
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._load_safeguards_config()

    def _load_safeguards_config(self):
        """Load configuration from deployment_safeguards.yaml if available."""
        try:
            config_path = Path(__file__).parent.parent.parent / "config" / "deployment_safeguards.yaml"
            if config_path.exists():
                with open(config_path) as f:
                    safeguards = yaml.safe_load(f)
                    if "data_quality" in safeguards and "l2_to_l3_gate" in safeguards["data_quality"]:
                        gate_config = safeguards["data_quality"]["l2_to_l3_gate"]["validations"]
                        self.config.update(gate_config)
                        logger.info("[DQ] Loaded config from deployment_safeguards.yaml")
        except Exception as e:
            logger.warning(f"[DQ] Could not load safeguards config: {e}")

    def validate(
        self,
        dataset_path: str,
        strict: bool = True,
    ) -> DataQualityReport:
        """
        Validate dataset quality.

        Args:
            dataset_path: Path to the dataset (CSV or Parquet)
            strict: If True, raise exception on failure

        Returns:
            DataQualityReport with validation results

        Raises:
            DataQualityError: If strict=True and validation fails
        """
        logger.info(f"[DQ] Starting validation: {dataset_path}")

        # Load dataset
        path = Path(dataset_path)
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        checks: List[ValidationCheck] = []

        # Run all validation checks
        checks.append(self._check_row_count(df))
        checks.append(self._check_feature_count(df))
        checks.append(self._check_feature_order(df))
        checks.append(self._check_nan_percentage(df))
        checks.append(self._check_duplicate_timestamps(df))
        checks.append(self._check_date_monotonicity(df))
        checks.append(self._check_value_ranges(df))
        checks.append(self._check_zscore_bounds(df))
        checks.append(self._check_infinite_values(df))
        checks.append(self._check_constant_features(df))

        # Determine overall status
        failed = any(c.status == ValidationStatus.FAILED for c in checks)
        warnings = any(c.status == ValidationStatus.WARNING for c in checks)

        if failed:
            overall_status = ValidationStatus.FAILED
        elif warnings:
            overall_status = ValidationStatus.WARNING
        else:
            overall_status = ValidationStatus.PASSED

        # Build summary
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "nan_percentage": df.isna().sum().sum() / df.size * 100,
            "memory_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "date_range": {
                "start": str(df["time"].min()) if "time" in df.columns else None,
                "end": str(df["time"].max()) if "time" in df.columns else None,
            },
        }

        report = DataQualityReport(
            dataset_path=str(dataset_path),
            validated_at=datetime.now(),
            overall_status=overall_status,
            checks=checks,
            summary=summary,
        )

        # Log results
        self._log_report(report)

        # Raise if strict mode and failed
        if strict and not report.passed:
            failed_names = [c.name for c in report.failed_checks]
            raise DataQualityError(
                f"Data quality validation failed: {failed_names}",
                report,
            )

        return report

    def _check_row_count(self, df: pd.DataFrame) -> ValidationCheck:
        """Check minimum row count."""
        min_rows = self.config.get("min_row_count", 100000)
        actual_rows = len(df)
        passed = actual_rows >= min_rows

        return ValidationCheck(
            name="row_count",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Row count: {actual_rows:,} (min: {min_rows:,})",
            details={"actual": actual_rows, "minimum": min_rows},
        )

    def _check_feature_count(self, df: pd.DataFrame) -> ValidationCheck:
        """Check expected feature count."""
        expected = self.config.get("expected_feature_count", 15)
        # Exclude time column
        feature_cols = [c for c in df.columns if c not in ["time", "timestamp", "date"]]
        actual = len(feature_cols)
        passed = actual >= expected

        return ValidationCheck(
            name="feature_count",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"Feature count: {actual} (expected: {expected})",
            details={"actual": actual, "expected": expected, "columns": feature_cols},
        )

    def _check_feature_order(self, df: pd.DataFrame) -> ValidationCheck:
        """Check feature order matches SSOT contract."""
        feature_cols = [c for c in df.columns if c not in ["time", "timestamp", "date"]]

        # Check if expected features are present
        missing = set(self.EXPECTED_FEATURES) - set(feature_cols)
        extra = set(feature_cols) - set(self.EXPECTED_FEATURES) - {"position", "time_normalized"}

        if missing:
            return ValidationCheck(
                name="feature_order",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Missing features: {missing}",
                details={"missing": list(missing), "extra": list(extra)},
            )

        # Check order for the first 13 features (market features)
        market_features = feature_cols[:13] if len(feature_cols) >= 13 else feature_cols
        expected_order = self.EXPECTED_FEATURES[:len(market_features)]
        order_matches = market_features == expected_order

        if not order_matches:
            return ValidationCheck(
                name="feature_order",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message="Feature order differs from SSOT contract",
                details={"actual": market_features, "expected": expected_order},
            )

        return ValidationCheck(
            name="feature_order",
            status=ValidationStatus.PASSED,
            severity=ValidationSeverity.INFO,
            message="Feature order matches SSOT contract",
            details={"features": market_features},
        )

    def _check_nan_percentage(self, df: pd.DataFrame) -> ValidationCheck:
        """Check NaN percentage is below threshold."""
        max_nan = self.config.get("max_nan_percentage", 0.05)

        # Calculate per-column and total
        nan_per_column = df.isna().sum()
        total_nan = nan_per_column.sum()
        total_cells = df.size
        nan_pct = total_nan / total_cells

        passed = nan_pct <= max_nan

        # Find columns with highest NaN
        worst_columns = nan_per_column.nlargest(5).to_dict()

        return ValidationCheck(
            name="nan_percentage",
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            severity=ValidationSeverity.ERROR if not passed else ValidationSeverity.INFO,
            message=f"NaN percentage: {nan_pct*100:.2f}% (max: {max_nan*100:.1f}%)",
            details={
                "nan_percentage": nan_pct,
                "max_allowed": max_nan,
                "total_nan_cells": int(total_nan),
                "worst_columns": worst_columns,
            },
        )

    def _check_duplicate_timestamps(self, df: pd.DataFrame) -> ValidationCheck:
        """Check for duplicate timestamps."""
        allow_duplicates = self.config.get("allow_duplicate_timestamps", False)

        if "time" not in df.columns:
            return ValidationCheck(
                name="duplicate_timestamps",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.INFO,
                message="No 'time' column found",
                details={},
            )

        duplicates = df["time"].duplicated().sum()

        if duplicates == 0:
            return ValidationCheck(
                name="duplicate_timestamps",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="No duplicate timestamps found",
                details={"duplicate_count": 0},
            )

        if allow_duplicates:
            return ValidationCheck(
                name="duplicate_timestamps",
                status=ValidationStatus.WARNING,
                severity=ValidationSeverity.WARNING,
                message=f"Found {duplicates:,} duplicate timestamps (allowed)",
                details={"duplicate_count": duplicates},
            )

        return ValidationCheck(
            name="duplicate_timestamps",
            status=ValidationStatus.FAILED,
            severity=ValidationSeverity.ERROR,
            message=f"Found {duplicates:,} duplicate timestamps",
            details={"duplicate_count": duplicates},
        )

    def _check_date_monotonicity(self, df: pd.DataFrame) -> ValidationCheck:
        """Check that dates are monotonically increasing (no future leakage)."""
        require_monotonic = self.config.get("require_monotonic_dates", True)

        if "time" not in df.columns:
            return ValidationCheck(
                name="date_monotonicity",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.INFO,
                message="No 'time' column found",
                details={},
            )

        # Convert to datetime if needed
        time_col = pd.to_datetime(df["time"])
        is_monotonic = time_col.is_monotonic_increasing

        if is_monotonic:
            return ValidationCheck(
                name="date_monotonicity",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="Timestamps are monotonically increasing",
                details={"is_monotonic": True},
            )

        # Find violations
        violations = (time_col.diff() < pd.Timedelta(0)).sum()

        status = ValidationStatus.FAILED if require_monotonic else ValidationStatus.WARNING
        severity = ValidationSeverity.CRITICAL if require_monotonic else ValidationSeverity.WARNING

        return ValidationCheck(
            name="date_monotonicity",
            status=status,
            severity=severity,
            message=f"Found {violations:,} timestamp ordering violations (potential future leakage)",
            details={"is_monotonic": False, "violations": int(violations)},
        )

    def _check_value_ranges(self, df: pd.DataFrame) -> ValidationCheck:
        """Check feature values are within expected ranges."""
        feature_ranges = self.config.get("feature_ranges", {})
        violations = []

        for feature, bounds in feature_ranges.items():
            if feature not in df.columns:
                continue

            col = df[feature]
            min_val = bounds.get("min")
            max_val = bounds.get("max")

            if min_val is not None:
                below_min = (col < min_val).sum()
                if below_min > 0:
                    violations.append({
                        "feature": feature,
                        "violation": "below_min",
                        "count": int(below_min),
                        "threshold": min_val,
                        "actual_min": float(col.min()),
                    })

            if max_val is not None:
                above_max = (col > max_val).sum()
                if above_max > 0:
                    violations.append({
                        "feature": feature,
                        "violation": "above_max",
                        "count": int(above_max),
                        "threshold": max_val,
                        "actual_max": float(col.max()),
                    })

        if not violations:
            return ValidationCheck(
                name="value_ranges",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="All features within expected ranges",
                details={"checked_features": list(feature_ranges.keys())},
            )

        return ValidationCheck(
            name="value_ranges",
            status=ValidationStatus.WARNING,
            severity=ValidationSeverity.WARNING,
            message=f"Found {len(violations)} range violations",
            details={"violations": violations},
        )

    def _check_zscore_bounds(self, df: pd.DataFrame) -> ValidationCheck:
        """Check that no features have extreme z-scores."""
        max_zscore = self.config.get("max_zscore", 10.0)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        extreme_values = {}

        for col in numeric_cols:
            if col in ["time", "timestamp"]:
                continue

            series = df[col].dropna()
            if len(series) == 0:
                continue

            mean = series.mean()
            std = series.std()

            if std == 0:
                continue

            zscores = np.abs((series - mean) / std)
            extreme_count = (zscores > max_zscore).sum()

            if extreme_count > 0:
                extreme_values[col] = {
                    "count": int(extreme_count),
                    "max_zscore": float(zscores.max()),
                    "pct": extreme_count / len(series) * 100,
                }

        if not extreme_values:
            return ValidationCheck(
                name="zscore_bounds",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message=f"No values exceed {max_zscore} sigma",
                details={"max_allowed_zscore": max_zscore},
            )

        total_extreme = sum(v["count"] for v in extreme_values.values())

        return ValidationCheck(
            name="zscore_bounds",
            status=ValidationStatus.WARNING,
            severity=ValidationSeverity.WARNING,
            message=f"Found {total_extreme:,} extreme values (>{max_zscore} sigma)",
            details={"extreme_by_feature": extreme_values},
        )

    def _check_infinite_values(self, df: pd.DataFrame) -> ValidationCheck:
        """Check for infinite values."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        inf_counts = {}
        for col in numeric_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = int(inf_count)

        if not inf_counts:
            return ValidationCheck(
                name="infinite_values",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="No infinite values found",
                details={},
            )

        total_inf = sum(inf_counts.values())

        return ValidationCheck(
            name="infinite_values",
            status=ValidationStatus.FAILED,
            severity=ValidationSeverity.CRITICAL,
            message=f"Found {total_inf:,} infinite values",
            details={"inf_by_column": inf_counts},
        )

    def _check_constant_features(self, df: pd.DataFrame) -> ValidationCheck:
        """Check for constant features (zero variance)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        constant_cols = []
        for col in numeric_cols:
            if col in ["time", "timestamp"]:
                continue
            if df[col].nunique() <= 1:
                constant_cols.append(col)

        if not constant_cols:
            return ValidationCheck(
                name="constant_features",
                status=ValidationStatus.PASSED,
                severity=ValidationSeverity.INFO,
                message="No constant features found",
                details={},
            )

        return ValidationCheck(
            name="constant_features",
            status=ValidationStatus.WARNING,
            severity=ValidationSeverity.WARNING,
            message=f"Found {len(constant_cols)} constant features",
            details={"constant_columns": constant_cols},
        )

    def _log_report(self, report: DataQualityReport):
        """Log validation report."""
        status_emoji = {
            ValidationStatus.PASSED: "[OK]",
            ValidationStatus.WARNING: "[WARN]",
            ValidationStatus.FAILED: "[FAIL]",
        }

        logger.info("=" * 60)
        logger.info(f"[DQ] DATA QUALITY REPORT: {status_emoji.get(report.overall_status, '?')}")
        logger.info("=" * 60)
        logger.info(f"  Dataset: {report.dataset_path}")
        logger.info(f"  Rows: {report.summary['total_rows']:,}")
        logger.info(f"  NaN%: {report.summary['nan_percentage']:.2f}%")

        for check in report.checks:
            emoji = status_emoji.get(check.status, "?")
            logger.info(f"  {emoji} {check.name}: {check.message}")

        if report.failed_checks:
            logger.error(f"  FAILED CHECKS: {[c.name for c in report.failed_checks]}")

        logger.info("=" * 60)


# =============================================================================
# Convenience function for DAG usage
# =============================================================================

def validate_dataset_for_training(
    dataset_path: str,
    strict: bool = True,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Validate dataset before training.

    This function is designed to be called from Airflow DAGs.

    Args:
        dataset_path: Path to dataset file
        strict: If True, raise on failure
        config: Optional configuration overrides

    Returns:
        Validation report as dictionary

    Raises:
        DataQualityError: If validation fails and strict=True
    """
    gate = DataQualityGate(config=config)
    report = gate.validate(dataset_path, strict=strict)
    return report.to_dict()
