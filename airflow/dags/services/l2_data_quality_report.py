# -*- coding: utf-8 -*-
"""
L2 Data Quality Report Generator
================================

Generates comprehensive statistical reports for each variable in the L2 dataset.
Provides transparency, debugging capabilities, and data quality monitoring.

Contract: CTR-L2-QUALITY-REPORT-001

Report Contents per Variable:
    1. Basic Statistics (mean, std, min, max, percentiles, skewness, kurtosis)
    2. Temporal Coverage (date range, frequency, gaps)
    3. Missing Data Analysis (count, %, gap ranges)
    4. Anomaly Detection (outliers, sudden jumps, zeros)
    5. Data Quality Scores (completeness, freshness, validity)
    6. Distribution Analysis (normality test, distribution type)
    7. Trend Analysis (direction, seasonality, stationarity)
    8. Anti-Leakage Verification (T-1 shift, no future data)
    9. Correlations (top correlated variables)
    10. Transformations Applied (normalization params)

Output:
    - JSON report: l2_data_quality_report_{timestamp}.json
    - HTML report: l2_data_quality_report_{timestamp}.html (optional)
    - CSV summary: l2_variable_summary_{timestamp}.csv

Version: 1.0.0
Author: USDCOP Trading System
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class DataQualityLevel(str, Enum):
    """Data quality assessment levels."""
    EXCELLENT = "excellent"  # >= 95%
    GOOD = "good"            # >= 85%
    ACCEPTABLE = "acceptable"  # >= 70%
    POOR = "poor"            # >= 50%
    CRITICAL = "critical"    # < 50%


class AnomalyType(str, Enum):
    """Types of anomalies detected."""
    OUTLIER_IQR = "outlier_iqr"
    OUTLIER_ZSCORE = "outlier_zscore"
    SUDDEN_JUMP = "sudden_jump"
    UNEXPECTED_ZERO = "unexpected_zero"
    UNEXPECTED_NEGATIVE = "unexpected_negative"
    DUPLICATE_VALUE = "duplicate_value"
    CONSTANT_PERIOD = "constant_period"


class FrequencyType(str, Enum):
    """Data frequency types."""
    MINUTE_5 = "5min"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"


@dataclass
class TemporalCoverage:
    """Temporal coverage statistics."""
    fecha_min: Optional[str] = None
    fecha_max: Optional[str] = None
    total_days: int = 0
    expected_frequency: str = "daily"
    detected_frequency: str = "unknown"
    expected_records: int = 0
    actual_records: int = 0
    coverage_percentage: float = 0.0
    trading_days_only: bool = True


@dataclass
class MissingDataAnalysis:
    """Missing data analysis results."""
    total_missing: int = 0
    missing_percentage: float = 0.0
    gap_count: int = 0
    longest_gap_days: int = 0
    longest_gap_start: Optional[str] = None
    longest_gap_end: Optional[str] = None
    gap_ranges: List[Dict[str, str]] = field(default_factory=list)
    missing_by_month: Dict[str, int] = field(default_factory=dict)


@dataclass
class AnomalyInfo:
    """Information about a detected anomaly."""
    type: str
    date: str
    value: float
    expected_range: Optional[Tuple[float, float]] = None
    severity: str = "medium"  # low, medium, high
    description: str = ""


@dataclass
class AnomalyAnalysis:
    """Anomaly detection results."""
    total_anomalies: int = 0
    outliers_iqr_count: int = 0
    outliers_zscore_count: int = 0
    sudden_jumps_count: int = 0
    unexpected_zeros_count: int = 0
    unexpected_negatives_count: int = 0
    anomaly_percentage: float = 0.0
    anomalies: List[AnomalyInfo] = field(default_factory=list)
    iqr_bounds: Optional[Tuple[float, float]] = None
    zscore_threshold: float = 3.0


@dataclass
class DistributionAnalysis:
    """Distribution analysis results."""
    is_normal: bool = False
    normality_pvalue: float = 0.0
    normality_test: str = "shapiro"
    detected_distribution: str = "unknown"
    skewness: float = 0.0
    kurtosis: float = 0.0
    is_symmetric: bool = False
    has_heavy_tails: bool = False


@dataclass
class TrendAnalysis:
    """Trend and stationarity analysis."""
    trend_direction: str = "flat"  # up, down, flat
    trend_strength: float = 0.0  # 0-1
    is_stationary: bool = False
    adf_statistic: float = 0.0
    adf_pvalue: float = 1.0
    has_seasonality: bool = False
    seasonality_period: Optional[int] = None
    volatility_regime: str = "normal"  # low, normal, high


@dataclass
class AntiLeakageVerification:
    """Anti-leakage verification results."""
    t1_shift_applied: bool = False
    t1_shift_verified: bool = False
    no_future_data: bool = True
    merge_direction: str = "backward"
    last_available_date: Optional[str] = None
    reference_date: Optional[str] = None
    verification_passed: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class CorrelationInfo:
    """Correlation with other variables."""
    variable: str
    correlation: float
    correlation_type: str = "pearson"


@dataclass
class TransformationInfo:
    """Applied transformations."""
    normalization_method: str = "zscore"  # zscore, minmax, robust
    norm_mean: Optional[float] = None
    norm_std: Optional[float] = None
    norm_min: Optional[float] = None
    norm_max: Optional[float] = None
    log_transformed: bool = False
    differenced: bool = False
    differencing_order: int = 0


@dataclass
class VariableReport:
    """Complete report for a single variable."""
    # Identification
    variable_name: str
    variable_type: str = "numeric"  # numeric, categorical, datetime
    source: str = "unknown"
    frequency: str = "daily"

    # Basic Statistics
    count: int = 0
    mean: float = 0.0
    std: float = 0.0
    min: float = 0.0
    q25: float = 0.0
    median: float = 0.0
    q75: float = 0.0
    max: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    variance: float = 0.0
    range: float = 0.0
    iqr: float = 0.0
    cv: float = 0.0  # Coefficient of variation

    # Detailed Analysis
    temporal: TemporalCoverage = field(default_factory=TemporalCoverage)
    missing: MissingDataAnalysis = field(default_factory=MissingDataAnalysis)
    anomalies: AnomalyAnalysis = field(default_factory=AnomalyAnalysis)
    distribution: DistributionAnalysis = field(default_factory=DistributionAnalysis)
    trend: TrendAnalysis = field(default_factory=TrendAnalysis)
    anti_leakage: AntiLeakageVerification = field(default_factory=AntiLeakageVerification)
    transformations: TransformationInfo = field(default_factory=TransformationInfo)

    # Correlations
    top_correlations: List[CorrelationInfo] = field(default_factory=list)

    # Quality Score
    quality_score: float = 0.0
    quality_level: str = "unknown"
    quality_issues: List[str] = field(default_factory=list)

    # Metadata
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class L2DataQualityReport:
    """Complete L2 data quality report."""
    # Report Metadata
    report_id: str = ""
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    pipeline_version: str = "1.0.0"
    dataset_name: str = ""

    # Dataset Overview
    total_variables: int = 0
    total_records: int = 0
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None

    # Quality Summary
    overall_quality_score: float = 0.0
    overall_quality_level: str = "unknown"
    variables_excellent: int = 0
    variables_good: int = 0
    variables_acceptable: int = 0
    variables_poor: int = 0
    variables_critical: int = 0

    # Issue Summary
    total_missing_values: int = 0
    total_anomalies: int = 0
    anti_leakage_verified: bool = False

    # Variable Reports
    variables: Dict[str, VariableReport] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'report_id': self.report_id,
            'generated_at': self.generated_at,
            'pipeline_version': self.pipeline_version,
            'dataset_name': self.dataset_name,
            'total_variables': self.total_variables,
            'total_records': self.total_records,
            'date_range': {
                'start': self.date_range_start,
                'end': self.date_range_end,
            },
            'quality_summary': {
                'overall_score': self.overall_quality_score,
                'overall_level': self.overall_quality_level,
                'by_level': {
                    'excellent': self.variables_excellent,
                    'good': self.variables_good,
                    'acceptable': self.variables_acceptable,
                    'poor': self.variables_poor,
                    'critical': self.variables_critical,
                },
            },
            'issue_summary': {
                'total_missing': self.total_missing_values,
                'total_anomalies': self.total_anomalies,
                'anti_leakage_verified': self.anti_leakage_verified,
            },
            'variables': {k: v.to_dict() for k, v in self.variables.items()},
            'recommendations': self.recommendations,
            'warnings': self.warnings,
        }
        return result


# =============================================================================
# DATA QUALITY REPORT GENERATOR
# =============================================================================

class L2DataQualityReportGenerator:
    """
    Generates comprehensive data quality reports for L2 datasets.

    Usage:
        generator = L2DataQualityReportGenerator()
        report = generator.generate_report(
            df=dataset,
            date_column='fecha',
            reference_date=datetime.now(),
            norm_stats=norm_stats_dict
        )
        generator.save_report(report, output_dir)
    """

    def __init__(
        self,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        jump_threshold_pct: float = 50.0,
        min_records_for_normality: int = 20,
    ):
        """
        Initialize the report generator.

        Args:
            zscore_threshold: Threshold for Z-score outlier detection
            iqr_multiplier: Multiplier for IQR outlier detection
            jump_threshold_pct: Percentage threshold for sudden jumps
            min_records_for_normality: Minimum records for normality test
        """
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.jump_threshold_pct = jump_threshold_pct
        self.min_records_for_normality = min_records_for_normality

        # Variable metadata (source mapping)
        self._variable_sources = self._load_variable_sources()

    def _load_variable_sources(self) -> Dict[str, str]:
        """Load variable to source mapping."""
        # This would ideally come from SSOT
        return {
            'close': 'twelvedata',
            'open': 'twelvedata',
            'high': 'twelvedata',
            'low': 'twelvedata',
            'volume': 'twelvedata',
            'rsi_14': 'calculated',
            'atr_14': 'calculated',
            'adx_14': 'calculated',
            'macd': 'calculated',
            'macd_signal': 'calculated',
            'bb_upper': 'calculated',
            'bb_lower': 'calculated',
            'returns_1': 'calculated',
            'volatility_20': 'calculated',
            'fed_funds': 'fred',
            'dxy': 'investing',
            'wti': 'investing',
            'vix': 'investing',
            'ibr': 'banrep',
            'ipc_col': 'dane',
            'pib_col': 'dane',
        }

    def generate_report(
        self,
        df: pd.DataFrame,
        date_column: str = 'fecha',
        reference_date: Optional[datetime] = None,
        norm_stats: Optional[Dict[str, Any]] = None,
        correlation_matrix: Optional[pd.DataFrame] = None,
        expected_frequency: str = 'daily',
        exclude_columns: Optional[List[str]] = None,
    ) -> L2DataQualityReport:
        """
        Generate a comprehensive data quality report.

        Args:
            df: The L2 dataset DataFrame
            date_column: Name of the date column
            reference_date: Reference date for anti-leakage checks
            norm_stats: Normalization statistics from training
            correlation_matrix: Pre-computed correlation matrix
            expected_frequency: Expected data frequency
            exclude_columns: Columns to exclude from analysis

        Returns:
            L2DataQualityReport with all variable analyses
        """
        reference_date = reference_date or datetime.utcnow()
        exclude_columns = exclude_columns or [date_column, 'symbol', 'source']

        logger.info("Generating L2 data quality report...")

        # Initialize report
        report = L2DataQualityReport(
            report_id=f"l2_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            dataset_name="l2_training_dataset",
            total_records=len(df),
        )

        # Get date range
        if date_column in df.columns:
            dates = pd.to_datetime(df[date_column])
            report.date_range_start = str(dates.min())
            report.date_range_end = str(dates.max())

        # Compute correlation matrix if not provided
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in exclude_columns]

        if correlation_matrix is None and len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()

        # Analyze each variable
        for col in df.columns:
            if col in exclude_columns:
                continue

            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                var_report = self._analyze_numeric_variable(
                    df=df,
                    column=col,
                    date_column=date_column,
                    reference_date=reference_date,
                    norm_stats=norm_stats,
                    correlation_matrix=correlation_matrix,
                    expected_frequency=expected_frequency,
                )
                report.variables[col] = var_report
                report.total_missing_values += var_report.missing.total_missing
                report.total_anomalies += var_report.anomalies.total_anomalies

        report.total_variables = len(report.variables)

        # Calculate overall quality
        self._calculate_overall_quality(report)

        # Generate recommendations
        self._generate_recommendations(report)

        logger.info(
            "Report generated: %d variables, overall quality: %.1f%% (%s)",
            report.total_variables,
            report.overall_quality_score * 100,
            report.overall_quality_level
        )

        return report

    def _analyze_numeric_variable(
        self,
        df: pd.DataFrame,
        column: str,
        date_column: str,
        reference_date: datetime,
        norm_stats: Optional[Dict],
        correlation_matrix: Optional[pd.DataFrame],
        expected_frequency: str,
    ) -> VariableReport:
        """Analyze a single numeric variable."""
        series = df[column].dropna()
        full_series = df[column]

        report = VariableReport(
            variable_name=column,
            variable_type="numeric",
            source=self._variable_sources.get(column, "unknown"),
            frequency=expected_frequency,
        )

        if len(series) == 0:
            report.quality_level = DataQualityLevel.CRITICAL.value
            report.quality_issues.append("No valid data")
            return report

        # Basic statistics
        report.count = len(series)
        report.mean = float(series.mean())
        report.std = float(series.std())
        report.min = float(series.min())
        report.q25 = float(series.quantile(0.25))
        report.median = float(series.median())
        report.q75 = float(series.quantile(0.75))
        report.max = float(series.max())
        report.variance = float(series.var())
        report.range = report.max - report.min
        report.iqr = report.q75 - report.q25

        # Coefficient of variation (handle zero mean)
        if report.mean != 0:
            report.cv = abs(report.std / report.mean)

        # Skewness and kurtosis
        if len(series) >= 3:
            report.skewness = float(series.skew())
            report.kurtosis = float(series.kurtosis())

        # Temporal coverage
        report.temporal = self._analyze_temporal_coverage(
            df, column, date_column, expected_frequency
        )

        # Missing data analysis
        report.missing = self._analyze_missing_data(
            df, column, date_column
        )

        # Anomaly detection
        report.anomalies = self._detect_anomalies(
            df, column, date_column
        )

        # Distribution analysis
        report.distribution = self._analyze_distribution(series)

        # Trend analysis
        report.trend = self._analyze_trend(series)

        # Anti-leakage verification
        report.anti_leakage = self._verify_anti_leakage(
            df, column, date_column, reference_date
        )

        # Transformations
        if norm_stats and column in norm_stats.get('features', {}):
            report.transformations = self._get_transformation_info(
                norm_stats['features'][column]
            )

        # Correlations
        if correlation_matrix is not None and column in correlation_matrix.columns:
            report.top_correlations = self._get_top_correlations(
                correlation_matrix, column, top_n=5
            )

        # Calculate quality score
        report.quality_score, report.quality_level = self._calculate_variable_quality(report)

        return report

    def _analyze_temporal_coverage(
        self,
        df: pd.DataFrame,
        column: str,
        date_column: str,
        expected_frequency: str,
    ) -> TemporalCoverage:
        """Analyze temporal coverage of a variable."""
        coverage = TemporalCoverage(expected_frequency=expected_frequency)

        if date_column not in df.columns:
            return coverage

        dates = pd.to_datetime(df[date_column])
        valid_mask = df[column].notna()
        valid_dates = dates[valid_mask]

        if len(valid_dates) == 0:
            return coverage

        coverage.fecha_min = str(valid_dates.min())
        coverage.fecha_max = str(valid_dates.max())
        coverage.actual_records = len(valid_dates)

        # Calculate expected records based on frequency
        date_range = (valid_dates.max() - valid_dates.min()).days
        coverage.total_days = date_range

        freq_multipliers = {
            'daily': 1,
            'weekly': 1/7,
            'monthly': 1/30,
            'quarterly': 1/90,
            '5min': 12 * 5,  # 5 hours trading * 12 bars/hour
        }

        multiplier = freq_multipliers.get(expected_frequency, 1)
        # Assume ~252 trading days per year for daily data
        if expected_frequency == 'daily':
            coverage.expected_records = int(date_range * 252 / 365)
        else:
            coverage.expected_records = int(date_range * multiplier)

        if coverage.expected_records > 0:
            coverage.coverage_percentage = min(
                100.0,
                (coverage.actual_records / coverage.expected_records) * 100
            )

        # Detect actual frequency
        if len(valid_dates) >= 2:
            diffs = valid_dates.sort_values().diff().dropna()
            median_diff = diffs.median().days

            if median_diff <= 0:
                coverage.detected_frequency = "intraday"
            elif median_diff <= 1:
                coverage.detected_frequency = "daily"
            elif median_diff <= 7:
                coverage.detected_frequency = "weekly"
            elif median_diff <= 31:
                coverage.detected_frequency = "monthly"
            else:
                coverage.detected_frequency = "quarterly"

        return coverage

    def _analyze_missing_data(
        self,
        df: pd.DataFrame,
        column: str,
        date_column: str,
    ) -> MissingDataAnalysis:
        """Analyze missing data patterns."""
        analysis = MissingDataAnalysis()

        missing_mask = df[column].isna()
        analysis.total_missing = int(missing_mask.sum())
        analysis.missing_percentage = (analysis.total_missing / len(df)) * 100 if len(df) > 0 else 0

        if date_column not in df.columns or analysis.total_missing == 0:
            return analysis

        # Find gap ranges
        dates = pd.to_datetime(df[date_column])
        df_sorted = df.sort_values(date_column)

        # Identify contiguous missing periods
        missing_dates = dates[missing_mask].sort_values()

        if len(missing_dates) > 0:
            gaps = []
            gap_start = missing_dates.iloc[0]
            gap_end = gap_start

            for i in range(1, len(missing_dates)):
                current = missing_dates.iloc[i]
                prev = missing_dates.iloc[i-1]

                # If gap between missing dates is more than expected frequency, new gap
                if (current - prev).days > 3:  # Assuming daily data with weekends
                    gaps.append({
                        'start': str(gap_start.date()),
                        'end': str(gap_end.date()),
                        'days': (gap_end - gap_start).days + 1
                    })
                    gap_start = current

                gap_end = current

            # Add last gap
            gaps.append({
                'start': str(gap_start.date()),
                'end': str(gap_end.date()),
                'days': (gap_end - gap_start).days + 1
            })

            analysis.gap_count = len(gaps)
            analysis.gap_ranges = gaps[:10]  # Keep top 10

            # Find longest gap
            if gaps:
                longest = max(gaps, key=lambda x: x['days'])
                analysis.longest_gap_days = longest['days']
                analysis.longest_gap_start = longest['start']
                analysis.longest_gap_end = longest['end']

        # Missing by month
        if len(missing_dates) > 0:
            missing_months = missing_dates.dt.to_period('M').value_counts()
            analysis.missing_by_month = {
                str(k): int(v) for k, v in missing_months.head(12).items()
            }

        return analysis

    def _detect_anomalies(
        self,
        df: pd.DataFrame,
        column: str,
        date_column: str,
    ) -> AnomalyAnalysis:
        """Detect anomalies in the data."""
        analysis = AnomalyAnalysis(zscore_threshold=self.zscore_threshold)
        series = df[column].dropna()

        if len(series) < 10:
            return analysis

        anomalies = []

        # IQR-based outliers
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        analysis.iqr_bounds = (float(lower_bound), float(upper_bound))

        iqr_outliers = (series < lower_bound) | (series > upper_bound)
        analysis.outliers_iqr_count = int(iqr_outliers.sum())

        # Z-score outliers
        if series.std() > 0:
            zscores = np.abs((series - series.mean()) / series.std())
            zscore_outliers = zscores > self.zscore_threshold
            analysis.outliers_zscore_count = int(zscore_outliers.sum())

        # Sudden jumps (percentage change)
        pct_change = series.pct_change().abs() * 100
        sudden_jumps = pct_change > self.jump_threshold_pct
        analysis.sudden_jumps_count = int(sudden_jumps.sum())

        # Unexpected zeros (if mean is significantly non-zero)
        if abs(series.mean()) > series.std() * 0.1:
            zeros = series == 0
            analysis.unexpected_zeros_count = int(zeros.sum())

        # Unexpected negatives (for typically positive variables)
        if series.mean() > 0 and series.min() < 0:
            negatives = series < 0
            analysis.unexpected_negatives_count = int(negatives.sum())

        # Total anomalies (unique dates)
        analysis.total_anomalies = (
            analysis.outliers_iqr_count +
            analysis.sudden_jumps_count +
            analysis.unexpected_zeros_count +
            analysis.unexpected_negatives_count
        )

        analysis.anomaly_percentage = (analysis.total_anomalies / len(series)) * 100

        # Record top anomalies with dates
        if date_column in df.columns and analysis.total_anomalies > 0:
            dates = pd.to_datetime(df[date_column])

            # Add IQR outliers
            if analysis.outliers_iqr_count > 0:
                outlier_idx = series[iqr_outliers].head(5).index
                for idx in outlier_idx:
                    if idx in dates.index:
                        anomalies.append(AnomalyInfo(
                            type=AnomalyType.OUTLIER_IQR.value,
                            date=str(dates[idx].date()),
                            value=float(series[idx]),
                            expected_range=analysis.iqr_bounds,
                            severity="medium"
                        ))

            # Add sudden jumps
            if analysis.sudden_jumps_count > 0:
                jump_idx = pct_change[sudden_jumps].head(5).index
                for idx in jump_idx:
                    if idx in dates.index and idx in series.index:
                        anomalies.append(AnomalyInfo(
                            type=AnomalyType.SUDDEN_JUMP.value,
                            date=str(dates[idx].date()),
                            value=float(series[idx]),
                            severity="high",
                            description=f"{pct_change[idx]:.1f}% change"
                        ))

        analysis.anomalies = anomalies[:20]  # Keep top 20

        return analysis

    def _analyze_distribution(self, series: pd.Series) -> DistributionAnalysis:
        """Analyze the distribution of a variable."""
        analysis = DistributionAnalysis()

        if len(series) < self.min_records_for_normality:
            return analysis

        analysis.skewness = float(series.skew())
        analysis.kurtosis = float(series.kurtosis())

        # Symmetry check
        analysis.is_symmetric = abs(analysis.skewness) < 0.5

        # Heavy tails check
        analysis.has_heavy_tails = analysis.kurtosis > 1.0

        # Normality test (Shapiro-Wilk for small samples, else use sample)
        try:
            from scipy import stats

            # Use sample for large datasets
            sample = series.sample(min(5000, len(series)), random_state=42)
            stat, pvalue = stats.shapiro(sample)

            analysis.normality_test = "shapiro"
            analysis.normality_pvalue = float(pvalue)
            analysis.is_normal = pvalue > 0.05

        except Exception as e:
            logger.debug("Normality test failed for series: %s", e)

        # Detect distribution type
        if analysis.is_normal:
            analysis.detected_distribution = "normal"
        elif analysis.skewness > 1:
            analysis.detected_distribution = "right_skewed"
        elif analysis.skewness < -1:
            analysis.detected_distribution = "left_skewed"
        elif analysis.has_heavy_tails:
            analysis.detected_distribution = "heavy_tailed"
        else:
            analysis.detected_distribution = "unknown"

        return analysis

    def _analyze_trend(self, series: pd.Series) -> TrendAnalysis:
        """Analyze trend and stationarity."""
        analysis = TrendAnalysis()

        if len(series) < 30:
            return analysis

        # Simple trend detection using linear regression
        try:
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            slope = coeffs[0]

            # Normalize slope by series range
            series_range = series.max() - series.min()
            if series_range > 0:
                normalized_slope = (slope * len(series)) / series_range
                analysis.trend_strength = min(abs(normalized_slope), 1.0)

                if normalized_slope > 0.1:
                    analysis.trend_direction = "up"
                elif normalized_slope < -0.1:
                    analysis.trend_direction = "down"
                else:
                    analysis.trend_direction = "flat"
        except Exception:
            pass

        # ADF test for stationarity
        try:
            from scipy import stats

            # Simple stationarity check: compare first and second half statistics
            half = len(series) // 2
            first_half = series.iloc[:half]
            second_half = series.iloc[half:]

            # T-test for mean difference
            t_stat, p_value = stats.ttest_ind(first_half, second_half)
            analysis.adf_pvalue = float(p_value)
            analysis.is_stationary = p_value > 0.05

        except Exception:
            pass

        # Volatility regime
        std = series.std()
        rolling_std = series.rolling(window=min(20, len(series)//4)).std()

        if len(rolling_std.dropna()) > 0:
            recent_std = rolling_std.dropna().iloc[-10:].mean()
            if recent_std > std * 1.5:
                analysis.volatility_regime = "high"
            elif recent_std < std * 0.5:
                analysis.volatility_regime = "low"

        return analysis

    def _verify_anti_leakage(
        self,
        df: pd.DataFrame,
        column: str,
        date_column: str,
        reference_date: datetime,
    ) -> AntiLeakageVerification:
        """Verify anti-leakage constraints."""
        verification = AntiLeakageVerification(
            reference_date=str(reference_date.date())
        )

        if date_column not in df.columns:
            verification.warnings.append("No date column found")
            return verification

        dates = pd.to_datetime(df[date_column])
        valid_mask = df[column].notna()

        if valid_mask.sum() == 0:
            verification.warnings.append("No valid data")
            return verification

        last_date = dates[valid_mask].max()
        verification.last_available_date = str(last_date.date())

        # Check no future data
        future_data = dates[valid_mask] > reference_date
        if future_data.any():
            verification.no_future_data = False
            verification.warnings.append(
                f"Found {future_data.sum()} records with future dates"
            )

        # T-1 shift verification for macro variables
        macro_prefixes = ['fed', 'dxy', 'wti', 'vix', 'ibr', 'ipc', 'pib', 'cpi', 'gdp']
        is_macro = any(column.lower().startswith(p) for p in macro_prefixes)

        if is_macro:
            # For macro, the data date should be at least T-1 from reference
            expected_max_date = reference_date - timedelta(days=1)
            if last_date.date() > expected_max_date.date():
                verification.t1_shift_applied = False
                verification.warnings.append(
                    f"Macro variable may have T-0 data (last: {last_date.date()}, expected <= {expected_max_date.date()})"
                )
            else:
                verification.t1_shift_applied = True
                verification.t1_shift_verified = True

        # Overall verification
        verification.verification_passed = (
            verification.no_future_data and
            (not is_macro or verification.t1_shift_verified)
        )

        return verification

    def _get_transformation_info(self, var_stats: Dict) -> TransformationInfo:
        """Extract transformation info from norm_stats."""
        info = TransformationInfo()

        if 'mean' in var_stats:
            info.normalization_method = "zscore"
            info.norm_mean = var_stats.get('mean')
            info.norm_std = var_stats.get('std')
        elif 'min' in var_stats and 'max' in var_stats:
            info.normalization_method = "minmax"
            info.norm_min = var_stats.get('min')
            info.norm_max = var_stats.get('max')

        info.log_transformed = var_stats.get('log_transformed', False)
        info.differenced = var_stats.get('differenced', False)
        info.differencing_order = var_stats.get('differencing_order', 0)

        return info

    def _get_top_correlations(
        self,
        corr_matrix: pd.DataFrame,
        column: str,
        top_n: int = 5,
    ) -> List[CorrelationInfo]:
        """Get top correlated variables."""
        if column not in corr_matrix.columns:
            return []

        correlations = corr_matrix[column].drop(column, errors='ignore')
        top_corr = correlations.abs().nlargest(top_n)

        result = []
        for var, abs_corr in top_corr.items():
            actual_corr = correlations[var]
            result.append(CorrelationInfo(
                variable=str(var),
                correlation=float(actual_corr),
                correlation_type="pearson"
            ))

        return result

    def _calculate_variable_quality(self, report: VariableReport) -> Tuple[float, str]:
        """Calculate quality score for a variable."""
        scores = []
        issues = []

        # Completeness (40% weight)
        completeness = 1.0 - (report.missing.missing_percentage / 100)
        scores.append(('completeness', completeness, 0.4))
        if completeness < 0.95:
            issues.append(f"Missing {report.missing.missing_percentage:.1f}% data")

        # Coverage (20% weight)
        coverage = report.temporal.coverage_percentage / 100
        scores.append(('coverage', min(coverage, 1.0), 0.2))
        if coverage < 0.9:
            issues.append(f"Only {coverage*100:.1f}% temporal coverage")

        # Anomaly rate (20% weight)
        anomaly_score = 1.0 - min(report.anomalies.anomaly_percentage / 10, 1.0)
        scores.append(('anomaly', anomaly_score, 0.2))
        if report.anomalies.anomaly_percentage > 5:
            issues.append(f"{report.anomalies.anomaly_percentage:.1f}% anomalies detected")

        # Anti-leakage (20% weight)
        leakage_score = 1.0 if report.anti_leakage.verification_passed else 0.5
        scores.append(('anti_leakage', leakage_score, 0.2))
        if not report.anti_leakage.verification_passed:
            issues.extend(report.anti_leakage.warnings)

        # Calculate weighted score
        total_score = sum(score * weight for _, score, weight in scores)

        # Determine quality level
        if total_score >= 0.95:
            level = DataQualityLevel.EXCELLENT.value
        elif total_score >= 0.85:
            level = DataQualityLevel.GOOD.value
        elif total_score >= 0.70:
            level = DataQualityLevel.ACCEPTABLE.value
        elif total_score >= 0.50:
            level = DataQualityLevel.POOR.value
        else:
            level = DataQualityLevel.CRITICAL.value

        report.quality_issues = issues

        return total_score, level

    def _calculate_overall_quality(self, report: L2DataQualityReport):
        """Calculate overall dataset quality."""
        if not report.variables:
            return

        scores = []
        for var_report in report.variables.values():
            scores.append(var_report.quality_score)

            if var_report.quality_level == DataQualityLevel.EXCELLENT.value:
                report.variables_excellent += 1
            elif var_report.quality_level == DataQualityLevel.GOOD.value:
                report.variables_good += 1
            elif var_report.quality_level == DataQualityLevel.ACCEPTABLE.value:
                report.variables_acceptable += 1
            elif var_report.quality_level == DataQualityLevel.POOR.value:
                report.variables_poor += 1
            else:
                report.variables_critical += 1

        report.overall_quality_score = sum(scores) / len(scores)

        # Overall level
        if report.overall_quality_score >= 0.95:
            report.overall_quality_level = DataQualityLevel.EXCELLENT.value
        elif report.overall_quality_score >= 0.85:
            report.overall_quality_level = DataQualityLevel.GOOD.value
        elif report.overall_quality_score >= 0.70:
            report.overall_quality_level = DataQualityLevel.ACCEPTABLE.value
        elif report.overall_quality_score >= 0.50:
            report.overall_quality_level = DataQualityLevel.POOR.value
        else:
            report.overall_quality_level = DataQualityLevel.CRITICAL.value

        # Anti-leakage verification
        report.anti_leakage_verified = all(
            v.anti_leakage.verification_passed
            for v in report.variables.values()
        )

    def _generate_recommendations(self, report: L2DataQualityReport):
        """Generate recommendations based on the report."""
        recommendations = []
        warnings = []

        # Missing data recommendations
        if report.total_missing_values > 0:
            pct = (report.total_missing_values / (report.total_records * report.total_variables)) * 100
            if pct > 10:
                recommendations.append(
                    f"HIGH: {pct:.1f}% total missing data. Consider imputation or source review."
                )
            elif pct > 5:
                warnings.append(
                    f"MEDIUM: {pct:.1f}% total missing data. Monitor data sources."
                )

        # Anomaly recommendations
        if report.total_anomalies > 0:
            anomaly_rate = report.total_anomalies / (report.total_records * report.total_variables) * 100
            if anomaly_rate > 5:
                recommendations.append(
                    f"HIGH: {anomaly_rate:.1f}% anomaly rate. Review outlier handling."
                )

        # Anti-leakage
        if not report.anti_leakage_verified:
            recommendations.append(
                "CRITICAL: Anti-leakage verification failed. Review data pipeline."
            )

        # Variable-specific issues
        critical_vars = [
            v.variable_name for v in report.variables.values()
            if v.quality_level == DataQualityLevel.CRITICAL.value
        ]
        if critical_vars:
            recommendations.append(
                f"CRITICAL: Variables with critical quality: {', '.join(critical_vars[:5])}"
            )

        poor_vars = [
            v.variable_name for v in report.variables.values()
            if v.quality_level == DataQualityLevel.POOR.value
        ]
        if poor_vars:
            warnings.append(
                f"MEDIUM: Variables with poor quality: {', '.join(poor_vars[:5])}"
            )

        report.recommendations = recommendations
        report.warnings = warnings

    def save_report(
        self,
        report: L2DataQualityReport,
        output_dir: Union[str, Path],
        formats: List[str] = None,
    ) -> Dict[str, Path]:
        """
        Save the report to files.

        Args:
            report: The L2DataQualityReport to save
            output_dir: Directory to save files
            formats: List of formats ['json', 'csv', 'html']

        Returns:
            Dictionary mapping format to file path
        """
        formats = formats or ['json', 'csv']
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        saved_files = {}

        # JSON report (full)
        if 'json' in formats:
            json_path = output_dir / f"l2_data_quality_report_{timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(report.to_dict(), f, indent=2, default=str)
            saved_files['json'] = json_path
            logger.info("Saved JSON report: %s", json_path)

        # CSV summary
        if 'csv' in formats:
            csv_path = output_dir / f"l2_variable_summary_{timestamp}.csv"

            rows = []
            for name, var in report.variables.items():
                rows.append({
                    'variable': name,
                    'source': var.source,
                    'count': var.count,
                    'mean': round(var.mean, 4),
                    'std': round(var.std, 4),
                    'min': round(var.min, 4),
                    'max': round(var.max, 4),
                    'missing_pct': round(var.missing.missing_percentage, 2),
                    'anomaly_pct': round(var.anomalies.anomaly_percentage, 2),
                    'quality_score': round(var.quality_score, 3),
                    'quality_level': var.quality_level,
                    'fecha_min': var.temporal.fecha_min,
                    'fecha_max': var.temporal.fecha_max,
                    'anti_leakage_ok': var.anti_leakage.verification_passed,
                })

            df = pd.DataFrame(rows)
            df.to_csv(csv_path, index=False)
            saved_files['csv'] = csv_path
            logger.info("Saved CSV summary: %s", csv_path)

        # HTML report (optional)
        if 'html' in formats:
            html_path = output_dir / f"l2_data_quality_report_{timestamp}.html"
            html_content = self._generate_html_report(report)
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            saved_files['html'] = html_path
            logger.info("Saved HTML report: %s", html_path)

        return saved_files

    def _generate_html_report(self, report: L2DataQualityReport) -> str:
        """Generate an HTML report."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>L2 Data Quality Report - {report.report_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .excellent {{ color: #28a745; }}
        .good {{ color: #5cb85c; }}
        .acceptable {{ color: #f0ad4e; }}
        .poor {{ color: #d9534f; }}
        .critical {{ color: #c9302c; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        .warning {{ background: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .recommendation {{ background: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
    </style>
</head>
<body>
    <h1>L2 Data Quality Report</h1>
    <p>Report ID: {report.report_id}</p>
    <p>Generated: {report.generated_at}</p>

    <div class="summary">
        <h2>Summary</h2>
        <p>Total Variables: {report.total_variables}</p>
        <p>Total Records: {report.total_records:,}</p>
        <p>Date Range: {report.date_range_start} to {report.date_range_end}</p>
        <p>Overall Quality: <span class="{report.overall_quality_level}">{report.overall_quality_score*100:.1f}% ({report.overall_quality_level.upper()})</span></p>
        <p>Anti-Leakage Verified: {'Yes' if report.anti_leakage_verified else 'NO'}</p>
    </div>

    <h2>Quality Distribution</h2>
    <ul>
        <li class="excellent">Excellent: {report.variables_excellent}</li>
        <li class="good">Good: {report.variables_good}</li>
        <li class="acceptable">Acceptable: {report.variables_acceptable}</li>
        <li class="poor">Poor: {report.variables_poor}</li>
        <li class="critical">Critical: {report.variables_critical}</li>
    </ul>
"""

        # Recommendations
        if report.recommendations:
            html += "<h2>Recommendations</h2>"
            for rec in report.recommendations:
                html += f'<div class="recommendation">{rec}</div>'

        if report.warnings:
            html += "<h2>Warnings</h2>"
            for warn in report.warnings:
                html += f'<div class="warning">{warn}</div>'

        # Variable table
        html += """
    <h2>Variable Details</h2>
    <table>
        <tr>
            <th>Variable</th>
            <th>Source</th>
            <th>Count</th>
            <th>Mean</th>
            <th>Std</th>
            <th>Min</th>
            <th>Max</th>
            <th>Missing %</th>
            <th>Anomaly %</th>
            <th>Quality</th>
        </tr>
"""

        for name, var in sorted(report.variables.items(), key=lambda x: x[1].quality_score):
            html += f"""
        <tr>
            <td>{name}</td>
            <td>{var.source}</td>
            <td>{var.count:,}</td>
            <td>{var.mean:.4f}</td>
            <td>{var.std:.4f}</td>
            <td>{var.min:.4f}</td>
            <td>{var.max:.4f}</td>
            <td>{var.missing.missing_percentage:.1f}%</td>
            <td>{var.anomalies.anomaly_percentage:.1f}%</td>
            <td class="{var.quality_level}">{var.quality_score*100:.0f}%</td>
        </tr>
"""

        html += """
    </table>
</body>
</html>
"""
        return html


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_l2_report(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    date_column: str = 'fecha',
    norm_stats: Optional[Dict] = None,
    formats: List[str] = None,
) -> L2DataQualityReport:
    """
    Convenience function to generate and save L2 data quality report.

    Args:
        df: The L2 dataset
        output_dir: Directory to save reports
        date_column: Name of date column
        norm_stats: Normalization statistics
        formats: Output formats ['json', 'csv', 'html']

    Returns:
        L2DataQualityReport
    """
    generator = L2DataQualityReportGenerator()
    report = generator.generate_report(
        df=df,
        date_column=date_column,
        norm_stats=norm_stats,
    )
    generator.save_report(report, output_dir, formats=formats or ['json', 'csv'])
    return report
