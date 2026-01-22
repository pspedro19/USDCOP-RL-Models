# backend/src/monitoring/__init__.py
"""
Data Quality Monitoring Module.

This module provides comprehensive data quality monitoring capabilities
for ML pipelines including:

- Missing value detection
- Distribution drift detection
- Outlier detection
- Data freshness validation
- Value range validation
- Comprehensive quality reporting

Example Usage:
    >>> from monitoring import DataQualityMonitor, QualityReport
    >>>
    >>> monitor = DataQualityMonitor(
    ...     missing_threshold=0.05,
    ...     outlier_z_threshold=3.0,
    ...     freshness_max_days=7
    ... )
    >>>
    >>> report = monitor.generate_quality_report(df)
    >>> if not report.is_passing():
    ...     raise DataQualityError(report.issues)
"""

from .quality_report import (
    QualityReport,
    DriftReport,
    OutlierReport,
    RangeReport,
    MissingValuesReport,
    DriftSeverity,
    IssueType,
    QualityCheckResult
)

from .data_quality import (
    DataQualityMonitor,
    DataQualityError
)


__all__ = [
    # Main classes
    "DataQualityMonitor",
    "DataQualityError",

    # Report types
    "QualityReport",
    "DriftReport",
    "OutlierReport",
    "RangeReport",
    "MissingValuesReport",
    "QualityCheckResult",

    # Enums
    "DriftSeverity",
    "IssueType",
]

__version__ = "1.0.0"
