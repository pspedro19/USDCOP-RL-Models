# backend/src/monitoring/data_quality.py
"""
Data Quality Monitor for ML Pipeline.

Provides comprehensive data quality checks including:
- Missing values detection
- Distribution drift detection
- Outlier detection
- Data freshness validation
- Value range validation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
import time

import numpy as np
import pandas as pd
from scipy import stats

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


logger = logging.getLogger(__name__)


class DataQualityMonitor:
    """
    Comprehensive data quality monitoring for ML pipelines.

    This class provides methods to check various aspects of data quality
    including missing values, distribution drift, outliers, data freshness,
    and value ranges. It generates detailed reports with actionable insights.

    Example:
        >>> monitor = DataQualityMonitor()
        >>> report = monitor.generate_quality_report(df)
        >>> if not report.is_passing():
        ...     print(report.get_summary())
        ...     raise DataQualityError(report.issues)
    """

    def __init__(
        self,
        missing_threshold: float = 0.05,
        outlier_z_threshold: float = 3.0,
        drift_p_threshold: float = 0.05,
        freshness_max_days: int = 7
    ):
        """
        Initialize DataQualityMonitor with default thresholds.

        Args:
            missing_threshold: Maximum acceptable missing value percentage (default 5%)
            outlier_z_threshold: Z-score threshold for outlier detection (default 3)
            drift_p_threshold: P-value threshold for drift detection (default 0.05)
            freshness_max_days: Maximum acceptable data age in days (default 7)
        """
        self.missing_threshold = missing_threshold
        self.outlier_z_threshold = outlier_z_threshold
        self.drift_p_threshold = drift_p_threshold
        self.freshness_max_days = freshness_max_days

    def check_missing_values(
        self,
        df: pd.DataFrame,
        threshold: float = None,
        columns: List[str] = None
    ) -> Dict[str, MissingValuesReport]:
        """
        Check for missing values in the dataset.

        Args:
            df: DataFrame to check
            threshold: Override default threshold (percentage, e.g., 0.05 for 5%)
            columns: Specific columns to check (default: all columns)

        Returns:
            Dictionary mapping column names to MissingValuesReport

        Example:
            >>> reports = monitor.check_missing_values(df, threshold=0.10)
            >>> for col, report in reports.items():
            ...     if report.exceeds_threshold:
            ...         print(f"Column {col} has {report.missing_percentage:.2%} missing")
        """
        threshold = threshold or self.missing_threshold
        columns = columns or df.columns.tolist()

        reports = {}

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df) if len(df) > 0 else 0.0

            reports[col] = MissingValuesReport(
                column=col,
                missing_count=int(missing_count),
                missing_percentage=float(missing_pct),
                threshold=threshold,
                exceeds_threshold=missing_pct > threshold
            )

        return reports

    def check_distribution_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        columns: List[str],
        method: str = "ks"
    ) -> List[DriftReport]:
        """
        Check for distribution drift between baseline and current data.

        Uses statistical tests to detect if the distribution of values
        has changed significantly between two datasets.

        Args:
            baseline_df: Reference/baseline DataFrame
            current_df: Current DataFrame to compare
            columns: Columns to check for drift
            method: Statistical test to use ('ks' for Kolmogorov-Smirnov,
                   'psi' for Population Stability Index)

        Returns:
            List of DriftReport for each column

        Example:
            >>> drift_reports = monitor.check_distribution_drift(
            ...     baseline_df=training_data,
            ...     current_df=inference_data,
            ...     columns=['feature1', 'feature2', 'feature3']
            ... )
            >>> for report in drift_reports:
            ...     if report.is_drifted:
            ...         print(f"Drift detected in {report.column}")
        """
        reports = []

        for col in columns:
            if col not in baseline_df.columns or col not in current_df.columns:
                logger.warning(f"Column '{col}' not found in one of the DataFrames")
                continue

            baseline_values = baseline_df[col].dropna().values
            current_values = current_df[col].dropna().values

            if len(baseline_values) < 10 or len(current_values) < 10:
                logger.warning(f"Insufficient data for drift detection in '{col}'")
                continue

            # Calculate basic statistics
            baseline_mean = float(np.mean(baseline_values))
            baseline_std = float(np.std(baseline_values))
            current_mean = float(np.mean(current_values))
            current_std = float(np.std(current_values))

            # Perform statistical test
            if method == "ks":
                statistic, p_value = stats.ks_2samp(baseline_values, current_values)
            elif method == "psi":
                statistic = self._calculate_psi(baseline_values, current_values)
                # PSI doesn't have a p-value, use threshold-based approach
                p_value = 0.01 if statistic > 0.25 else (0.05 if statistic > 0.1 else 0.5)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Determine drift and severity
            is_drifted = p_value < self.drift_p_threshold
            severity = self._determine_drift_severity(statistic, p_value, method)

            reports.append(DriftReport(
                column=col,
                baseline_mean=baseline_mean,
                baseline_std=baseline_std,
                current_mean=current_mean,
                current_std=current_std,
                drift_score=float(statistic),
                p_value=float(p_value),
                is_drifted=is_drifted,
                severity=severity
            ))

        return reports

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).

        PSI measures how much the distribution has shifted:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.25: Moderate change
        - PSI >= 0.25: Significant change

        Args:
            expected: Expected/baseline distribution
            actual: Actual/current distribution
            n_bins: Number of bins for histogram

        Returns:
            PSI value
        """
        # Create bins based on expected distribution
        _, bin_edges = np.histogram(expected, bins=n_bins)

        # Calculate frequencies
        expected_freq, _ = np.histogram(expected, bins=bin_edges)
        actual_freq, _ = np.histogram(actual, bins=bin_edges)

        # Convert to proportions
        expected_pct = expected_freq / len(expected)
        actual_pct = actual_freq / len(actual)

        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)

    def _determine_drift_severity(
        self,
        statistic: float,
        p_value: float,
        method: str
    ) -> DriftSeverity:
        """Determine severity level based on test results."""
        if method == "ks":
            if p_value >= 0.1:
                return DriftSeverity.NONE
            elif p_value >= 0.05:
                return DriftSeverity.LOW
            elif p_value >= 0.01:
                return DriftSeverity.MEDIUM
            elif p_value >= 0.001:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL
        elif method == "psi":
            if statistic < 0.1:
                return DriftSeverity.NONE
            elif statistic < 0.15:
                return DriftSeverity.LOW
            elif statistic < 0.25:
                return DriftSeverity.MEDIUM
            elif statistic < 0.4:
                return DriftSeverity.HIGH
            else:
                return DriftSeverity.CRITICAL
        return DriftSeverity.NONE

    def check_outliers(
        self,
        df: pd.DataFrame,
        columns: List[str],
        z_threshold: float = None,
        method: str = "zscore"
    ) -> List[OutlierReport]:
        """
        Detect outliers in specified columns.

        Supports multiple outlier detection methods:
        - zscore: Standard z-score method
        - iqr: Interquartile range method
        - mad: Median Absolute Deviation method

        Args:
            df: DataFrame to check
            columns: Columns to check for outliers
            z_threshold: Z-score threshold (default from init)
            method: Detection method ('zscore', 'iqr', 'mad')

        Returns:
            List of OutlierReport for each column

        Example:
            >>> outlier_reports = monitor.check_outliers(
            ...     df, columns=['price', 'volume'], z_threshold=3.0
            ... )
            >>> total_outliers = sum(r.n_outliers for r in outlier_reports)
        """
        z_threshold = z_threshold or self.outlier_z_threshold
        reports = []

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            values = df[col].dropna()

            if len(values) < 3:
                logger.warning(f"Insufficient data for outlier detection in '{col}'")
                continue

            if method == "zscore":
                z_scores = np.abs(stats.zscore(values))
                outlier_mask = z_scores > z_threshold
                mean = values.mean()
                std = values.std()
                lower_bound = mean - z_threshold * std
                upper_bound = mean + z_threshold * std

            elif method == "iqr":
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outlier_mask = (values < lower_bound) | (values > upper_bound)

            elif method == "mad":
                median = values.median()
                mad = np.median(np.abs(values - median))
                # Use modified z-score with MAD
                modified_z = 0.6745 * (values - median) / (mad + 1e-10)
                outlier_mask = np.abs(modified_z) > z_threshold
                lower_bound = median - z_threshold * mad / 0.6745
                upper_bound = median + z_threshold * mad / 0.6745

            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_indices = values[outlier_mask].index.tolist()
            outlier_values = values[outlier_mask].tolist()

            reports.append(OutlierReport(
                column=col,
                n_outliers=int(outlier_mask.sum()),
                outlier_indices=outlier_indices,
                outlier_values=outlier_values,
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                outlier_percentage=float(outlier_mask.sum() / len(values)) if len(values) > 0 else 0.0
            ))

        return reports

    def check_data_freshness(
        self,
        df: pd.DataFrame,
        date_col: str,
        max_days: int = None,
        reference_date: datetime = None
    ) -> Tuple[bool, float]:
        """
        Check if data is fresh (not stale).

        Args:
            df: DataFrame to check
            date_col: Name of the date column
            max_days: Maximum acceptable age in days
            reference_date: Reference date for comparison (default: now)

        Returns:
            Tuple of (is_fresh: bool, age_in_days: float)

        Example:
            >>> is_fresh, age = monitor.check_data_freshness(
            ...     df, date_col='Date', max_days=7
            ... )
            >>> if not is_fresh:
            ...     raise StaleDataError(f"Data is {age:.1f} days old")
        """
        max_days = max_days or self.freshness_max_days
        reference_date = reference_date or datetime.now()

        if date_col not in df.columns:
            logger.warning(f"Date column '{date_col}' not found")
            return False, float('inf')

        # Convert to datetime if needed
        dates = pd.to_datetime(df[date_col], errors='coerce')
        valid_dates = dates.dropna()

        if len(valid_dates) == 0:
            logger.warning("No valid dates found in date column")
            return False, float('inf')

        # Get most recent date
        latest_date = valid_dates.max()

        # Handle timezone-aware datetimes
        if latest_date.tzinfo is not None:
            if reference_date.tzinfo is None:
                reference_date = reference_date.replace(tzinfo=latest_date.tzinfo)
        else:
            if hasattr(reference_date, 'tzinfo') and reference_date.tzinfo is not None:
                reference_date = reference_date.replace(tzinfo=None)

        # Calculate age in days
        age_delta = reference_date - latest_date.to_pydatetime()
        age_days = age_delta.total_seconds() / (24 * 3600)

        is_fresh = age_days <= max_days

        return is_fresh, float(age_days)

    def check_value_ranges(
        self,
        df: pd.DataFrame,
        expected_ranges: Dict[str, Tuple[float, float]]
    ) -> List[RangeReport]:
        """
        Check if values fall within expected ranges.

        Args:
            df: DataFrame to check
            expected_ranges: Dictionary mapping column names to (min, max) tuples

        Returns:
            List of RangeReport for each column

        Example:
            >>> ranges = {
            ...     'price': (1000, 6000),
            ...     'volume': (0, 1e9),
            ...     'return': (-0.1, 0.1)
            ... }
            >>> range_reports = monitor.check_value_ranges(df, ranges)
            >>> for report in range_reports:
            ...     if not report.is_valid:
            ...         print(f"Range violation in {report.column}")
        """
        reports = []

        for col, (expected_min, expected_max) in expected_ranges.items():
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame")
                continue

            values = df[col].dropna()

            if len(values) == 0:
                reports.append(RangeReport(
                    column=col,
                    expected_min=expected_min,
                    expected_max=expected_max,
                    actual_min=float('nan'),
                    actual_max=float('nan'),
                    violations_below=0,
                    violations_above=0,
                    is_valid=True
                ))
                continue

            actual_min = float(values.min())
            actual_max = float(values.max())
            violations_below = int((values < expected_min).sum())
            violations_above = int((values > expected_max).sum())
            is_valid = violations_below == 0 and violations_above == 0

            reports.append(RangeReport(
                column=col,
                expected_min=expected_min,
                expected_max=expected_max,
                actual_min=actual_min,
                actual_max=actual_max,
                violations_below=violations_below,
                violations_above=violations_above,
                is_valid=is_valid
            ))

        return reports

    def generate_quality_report(
        self,
        df: pd.DataFrame,
        date_col: str = "Date",
        baseline_df: pd.DataFrame = None,
        expected_ranges: Dict[str, Tuple[float, float]] = None,
        numeric_columns: List[str] = None
    ) -> QualityReport:
        """
        Generate a comprehensive quality report for the dataset.

        This method runs all quality checks and aggregates results into
        a single report with an overall quality score.

        Args:
            df: DataFrame to analyze
            date_col: Name of the date column for freshness check
            baseline_df: Baseline DataFrame for drift detection (optional)
            expected_ranges: Expected value ranges for range check (optional)
            numeric_columns: Columns to check for outliers/drift (default: auto-detect)

        Returns:
            QualityReport with comprehensive quality assessment

        Example:
            >>> report = monitor.generate_quality_report(
            ...     df,
            ...     date_col='Date',
            ...     baseline_df=training_data,
            ...     expected_ranges={'price': (1000, 6000)}
            ... )
            >>> print(report.get_summary())
            >>> if not report.is_passing(min_score=75):
            ...     raise DataQualityError(report.issues)
        """
        start_time = time.time()

        # Initialize tracking
        issues = []
        recommendations = []
        check_scores = []

        # Auto-detect numeric columns if not specified
        if numeric_columns is None:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # 1. Check missing values
        missing_reports_dict = self.check_missing_values(df)
        missing_reports = list(missing_reports_dict.values())

        total_missing_cells = sum(r.missing_count for r in missing_reports)
        total_cells = len(df) * len(df.columns)
        overall_missing_pct = total_missing_cells / total_cells if total_cells > 0 else 0

        # Score missing values (0-100)
        missing_score = max(0, 100 - (overall_missing_pct * 500))  # 20% missing = 0 score
        check_scores.append(missing_score)

        high_missing_cols = [r.column for r in missing_reports if r.exceeds_threshold]
        if high_missing_cols:
            issues.append(
                f"High missing values in columns: {', '.join(high_missing_cols[:5])}"
            )
            recommendations.append(
                "Consider imputation strategies or investigate data collection issues"
            )

        # 2. Check outliers
        outlier_reports = self.check_outliers(df, numeric_columns)
        total_outliers = sum(r.n_outliers for r in outlier_reports)

        # Score outliers (0-100)
        outlier_pct = total_outliers / (len(df) * len(numeric_columns)) if len(df) > 0 and len(numeric_columns) > 0 else 0
        outlier_score = max(0, 100 - (outlier_pct * 1000))  # 10% outliers = 0 score
        check_scores.append(outlier_score)

        high_outlier_cols = [r.column for r in outlier_reports if r.n_outliers > len(df) * 0.05]
        if high_outlier_cols:
            issues.append(
                f"High outlier rate in columns: {', '.join(high_outlier_cols[:5])}"
            )
            recommendations.append(
                "Review outliers for data quality issues or consider robust scaling"
            )

        # 3. Check distribution drift (if baseline provided)
        drift_reports = []
        drift_detected = False

        if baseline_df is not None:
            drift_columns = [c for c in numeric_columns if c in baseline_df.columns]
            drift_reports = self.check_distribution_drift(
                baseline_df, df, drift_columns
            )
            drift_detected = any(r.is_drifted for r in drift_reports)

            # Score drift (0-100)
            drifted_count = sum(1 for r in drift_reports if r.is_drifted)
            drift_score = max(0, 100 - (drifted_count / len(drift_columns) * 100)) if drift_columns else 100
            check_scores.append(drift_score)

            if drift_detected:
                drifted_cols = [r.column for r in drift_reports if r.is_drifted]
                issues.append(
                    f"Distribution drift detected in: {', '.join(drifted_cols[:5])}"
                )
                recommendations.append(
                    "Consider retraining models or investigating data source changes"
                )
        else:
            check_scores.append(100)  # No baseline = no drift check, full score

        # 4. Check data freshness
        stale_data = False
        data_freshness_days = None

        if date_col in df.columns:
            is_fresh, age_days = self.check_data_freshness(df, date_col)
            stale_data = not is_fresh
            data_freshness_days = age_days

            # Score freshness (0-100)
            freshness_score = max(0, 100 - (age_days / self.freshness_max_days * 100)) if age_days <= self.freshness_max_days * 2 else 0
            check_scores.append(freshness_score)

            if stale_data:
                issues.append(
                    f"Data is stale: {age_days:.1f} days old (max: {self.freshness_max_days})"
                )
                recommendations.append(
                    "Update data source or adjust freshness requirements"
                )
        else:
            check_scores.append(100)  # No date column = no freshness check

        # 5. Check value ranges (if provided)
        range_reports = []

        if expected_ranges:
            range_reports = self.check_value_ranges(df, expected_ranges)

            # Score ranges (0-100)
            valid_ranges = sum(1 for r in range_reports if r.is_valid)
            range_score = (valid_ranges / len(range_reports) * 100) if range_reports else 100
            check_scores.append(range_score)

            invalid_ranges = [r.column for r in range_reports if not r.is_valid]
            if invalid_ranges:
                issues.append(
                    f"Range violations in columns: {', '.join(invalid_ranges[:5])}"
                )
                recommendations.append(
                    "Verify data transformations and input validation"
                )

        # Calculate overall score (weighted average)
        weights = [0.25, 0.20, 0.25, 0.20, 0.10]  # missing, outliers, drift, freshness, ranges
        weights = weights[:len(check_scores)]
        weights = [w / sum(weights) for w in weights]  # Normalize

        overall_score = sum(s * w for s, w in zip(check_scores, weights))

        # Add general recommendations based on score
        if overall_score < 50:
            recommendations.append(
                "CRITICAL: Data quality is poor. Investigate data pipeline issues."
            )
        elif overall_score < 70:
            recommendations.append(
                "WARNING: Data quality needs attention before production use."
            )

        execution_time = (time.time() - start_time) * 1000  # ms

        return QualityReport(
            timestamp=datetime.now(),
            n_rows=len(df),
            n_cols=len(df.columns),
            missing_pct=overall_missing_pct,
            outlier_count=total_outliers,
            drift_detected=drift_detected,
            stale_data=stale_data,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations,
            missing_reports=missing_reports,
            outlier_reports=outlier_reports,
            drift_reports=drift_reports,
            range_reports=range_reports,
            data_freshness_days=data_freshness_days,
            execution_time_ms=execution_time
        )

    def validate_for_inference(
        self,
        df: pd.DataFrame,
        min_score: float = 70.0,
        date_col: str = "Date",
        baseline_df: pd.DataFrame = None
    ) -> Tuple[bool, QualityReport]:
        """
        Validate data quality for inference pipeline.

        Convenience method that generates a report and checks if it passes
        the minimum score threshold.

        Args:
            df: DataFrame to validate
            min_score: Minimum acceptable quality score
            date_col: Date column name
            baseline_df: Baseline for drift detection

        Returns:
            Tuple of (is_valid: bool, report: QualityReport)

        Example:
            >>> is_valid, report = monitor.validate_for_inference(df, min_score=80)
            >>> if not is_valid:
            ...     logger.error(f"Data quality check failed: {report.issues}")
            ...     raise DataQualityError(report.get_summary())
        """
        report = self.generate_quality_report(
            df,
            date_col=date_col,
            baseline_df=baseline_df
        )

        is_valid = report.is_passing(min_score)

        # Log results
        if is_valid:
            logger.info(f"Data quality check PASSED with score {report.overall_score:.1f}")
        else:
            logger.warning(f"Data quality check FAILED with score {report.overall_score:.1f}")
            for issue in report.issues:
                logger.warning(f"  Issue: {issue}")

        return is_valid, report


class DataQualityError(Exception):
    """Exception raised when data quality checks fail."""

    def __init__(self, message: str, report: QualityReport = None):
        super().__init__(message)
        self.report = report

    def __str__(self):
        if self.report:
            return f"{self.args[0]}\n\n{self.report.get_summary()}"
        return self.args[0]
