# backend/src/monitoring/quality_report.py
"""
Data Quality Report structures.

Provides dataclasses for comprehensive quality reporting.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class DriftSeverity(Enum):
    """Severity levels for drift detection."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(Enum):
    """Types of data quality issues."""
    MISSING_VALUES = "missing_values"
    OUTLIERS = "outliers"
    DISTRIBUTION_DRIFT = "distribution_drift"
    STALE_DATA = "stale_data"
    RANGE_VIOLATION = "range_violation"
    SCHEMA_MISMATCH = "schema_mismatch"
    DUPLICATES = "duplicates"


@dataclass
class DriftReport:
    """
    Report for distribution drift analysis.

    Attributes:
        column: Column name analyzed
        baseline_mean: Mean of baseline distribution
        baseline_std: Std of baseline distribution
        current_mean: Mean of current distribution
        current_std: Std of current distribution
        drift_score: Statistical measure of drift (KS statistic or similar)
        p_value: P-value from statistical test
        is_drifted: Whether drift is detected
        severity: Severity level of drift
    """
    column: str
    baseline_mean: float
    baseline_std: float
    current_mean: float
    current_std: float
    drift_score: float
    p_value: float
    is_drifted: bool
    severity: DriftSeverity = DriftSeverity.NONE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "current_mean": self.current_mean,
            "current_std": self.current_std,
            "drift_score": self.drift_score,
            "p_value": self.p_value,
            "is_drifted": self.is_drifted,
            "severity": self.severity.value
        }


@dataclass
class OutlierReport:
    """
    Report for outlier detection.

    Attributes:
        column: Column name analyzed
        n_outliers: Number of outliers detected
        outlier_indices: Indices of outlier rows
        outlier_values: Values of outliers
        lower_bound: Lower threshold for outliers
        upper_bound: Upper threshold for outliers
        outlier_percentage: Percentage of data that are outliers
    """
    column: str
    n_outliers: int
    outlier_indices: List[int]
    outlier_values: List[float]
    lower_bound: float
    upper_bound: float
    outlier_percentage: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "n_outliers": self.n_outliers,
            "outlier_indices": self.outlier_indices[:10],  # Limit for readability
            "outlier_values": self.outlier_values[:10],
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "outlier_percentage": self.outlier_percentage
        }


@dataclass
class RangeReport:
    """
    Report for value range validation.

    Attributes:
        column: Column name analyzed
        expected_min: Expected minimum value
        expected_max: Expected maximum value
        actual_min: Actual minimum value
        actual_max: Actual maximum value
        violations_below: Count of values below expected min
        violations_above: Count of values above expected max
        is_valid: Whether all values are within range
    """
    column: str
    expected_min: float
    expected_max: float
    actual_min: float
    actual_max: float
    violations_below: int
    violations_above: int
    is_valid: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "expected_min": self.expected_min,
            "expected_max": self.expected_max,
            "actual_min": self.actual_min,
            "actual_max": self.actual_max,
            "violations_below": self.violations_below,
            "violations_above": self.violations_above,
            "is_valid": self.is_valid
        }


@dataclass
class MissingValuesReport:
    """
    Report for missing values analysis.

    Attributes:
        column: Column name analyzed
        missing_count: Number of missing values
        missing_percentage: Percentage of missing values
        threshold: Threshold for acceptable missing percentage
        exceeds_threshold: Whether missing rate exceeds threshold
    """
    column: str
    missing_count: int
    missing_percentage: float
    threshold: float
    exceeds_threshold: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "column": self.column,
            "missing_count": self.missing_count,
            "missing_percentage": self.missing_percentage,
            "threshold": self.threshold,
            "exceeds_threshold": self.exceeds_threshold
        }


@dataclass
class QualityReport:
    """
    Comprehensive data quality report.

    Aggregates all quality checks into a single report with
    an overall quality score and actionable recommendations.

    Attributes:
        timestamp: When the report was generated
        n_rows: Number of rows in dataset
        n_cols: Number of columns in dataset
        missing_pct: Overall missing value percentage
        outlier_count: Total number of outliers detected
        drift_detected: Whether distribution drift was detected
        stale_data: Whether data is stale (older than threshold)
        overall_score: Quality score from 0-100
        issues: List of detected issues
        recommendations: List of actionable recommendations
        missing_reports: Detailed missing value reports per column
        outlier_reports: Detailed outlier reports per column
        drift_reports: Detailed drift reports per column
        range_reports: Detailed range reports per column
        data_freshness_days: Age of most recent data in days
        execution_time_ms: Time taken to generate report
    """
    timestamp: datetime
    n_rows: int
    n_cols: int
    missing_pct: float
    outlier_count: int
    drift_detected: bool
    stale_data: bool
    overall_score: float  # 0-100
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    missing_reports: List[MissingValuesReport] = field(default_factory=list)
    outlier_reports: List[OutlierReport] = field(default_factory=list)
    drift_reports: List[DriftReport] = field(default_factory=list)
    range_reports: List[RangeReport] = field(default_factory=list)
    data_freshness_days: Optional[float] = None
    execution_time_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "missing_pct": self.missing_pct,
            "outlier_count": self.outlier_count,
            "drift_detected": self.drift_detected,
            "stale_data": self.stale_data,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "data_freshness_days": self.data_freshness_days,
            "execution_time_ms": self.execution_time_ms,
            "details": {
                "missing_reports": [r.to_dict() for r in self.missing_reports],
                "outlier_reports": [r.to_dict() for r in self.outlier_reports],
                "drift_reports": [r.to_dict() for r in self.drift_reports],
                "range_reports": [r.to_dict() for r in self.range_reports]
            }
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def is_passing(self, min_score: float = 70.0) -> bool:
        """
        Check if quality report passes minimum threshold.

        Args:
            min_score: Minimum acceptable score (default 70)

        Returns:
            True if overall_score >= min_score
        """
        return self.overall_score >= min_score

    def get_critical_issues(self) -> List[str]:
        """Get list of critical issues that should block pipeline."""
        critical = []

        if self.stale_data:
            critical.append("CRITICAL: Data is stale - exceeds freshness threshold")

        if self.missing_pct > 0.20:  # More than 20% missing
            critical.append(f"CRITICAL: High missing rate ({self.missing_pct*100:.1f}%)")

        for drift_report in self.drift_reports:
            if drift_report.severity == DriftSeverity.CRITICAL:
                critical.append(
                    f"CRITICAL: Severe drift in column '{drift_report.column}'"
                )

        return critical

    def get_summary(self) -> str:
        """Get human-readable summary of the report."""
        status = "PASS" if self.is_passing() else "FAIL"
        lines = [
            f"=== Data Quality Report ===",
            f"Status: {status}",
            f"Score: {self.overall_score:.1f}/100",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"",
            f"Dataset: {self.n_rows:,} rows x {self.n_cols} columns",
            f"Missing: {self.missing_pct*100:.2f}%",
            f"Outliers: {self.outlier_count:,}",
            f"Drift Detected: {'Yes' if self.drift_detected else 'No'}",
            f"Stale Data: {'Yes' if self.stale_data else 'No'}",
        ]

        if self.data_freshness_days is not None:
            lines.append(f"Data Age: {self.data_freshness_days:.1f} days")

        if self.issues:
            lines.append(f"\nIssues ({len(self.issues)}):")
            for issue in self.issues[:5]:  # Show first 5
                lines.append(f"  - {issue}")
            if len(self.issues) > 5:
                lines.append(f"  ... and {len(self.issues) - 5} more")

        if self.recommendations:
            lines.append(f"\nRecommendations:")
            for rec in self.recommendations[:3]:  # Show first 3
                lines.append(f"  - {rec}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.get_summary()


@dataclass
class QualityCheckResult:
    """
    Result of a single quality check.

    Used for intermediate results before aggregating into QualityReport.
    """
    check_name: str
    passed: bool
    score: float  # 0-100
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    issue_type: Optional[IssueType] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "check_name": self.check_name,
            "passed": self.passed,
            "score": self.score,
            "message": self.message,
            "details": self.details,
            "issue_type": self.issue_type.value if self.issue_type else None
        }
