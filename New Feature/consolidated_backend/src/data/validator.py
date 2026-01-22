# usdcop_forecasting_clean/backend/src/data/validator.py
"""
Data validation module.

Follows Single Responsibility Principle - only handles validation.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

from ..core.config import PipelineConfig

logger = logging.getLogger(__name__)


@dataclass
class DataReport:
    """
    Data quality report.

    Contains all validation results in a structured format.
    """
    n_rows: int
    n_cols: int
    date_range: Tuple[str, str]
    missing_pct: float
    price_range: Tuple[float, float]
    return_stats: dict = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues."""
        return len(self.issues) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if there are warnings."""
        return len(self.warnings) > 0

    @property
    def is_valid(self) -> bool:
        """Check if data passes all validations."""
        return not self.has_critical_issues


class DataValidator:
    """
    Validates data quality for regression pipeline.

    Checks:
    - Minimum sample size
    - Missing data percentage
    - Price validity (positive, no outliers)
    - Date ordering
    - Return distribution
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

    def validate(
        self,
        df: pd.DataFrame,
        prices: pd.Series
    ) -> DataReport:
        """
        Perform all validations.

        Args:
            df: DataFrame with data
            prices: Price series

        Returns:
            DataReport with all validation results
        """
        issues = []
        warnings = []

        # Basic stats
        n_rows = len(df)
        n_cols = len(df.columns)
        date_range = (
            df.index[0].strftime('%Y-%m-%d'),
            df.index[-1].strftime('%Y-%m-%d')
        )
        missing_pct = df.isnull().mean().mean()
        price_range = (float(prices.min()), float(prices.max()))

        # Calculate return statistics
        returns = np.log(prices / prices.shift(1)).dropna()
        return_stats = {
            'mean': float(returns.mean()),
            'std': float(returns.std()),
            'skew': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'min': float(returns.min()),
            'max': float(returns.max())
        }

        # Validation checks
        # 1. Minimum samples
        if n_rows < self.config.min_samples:
            issues.append(
                f"Insufficient data: {n_rows} < {self.config.min_samples} required"
            )

        # 2. Missing data
        if missing_pct > self.config.max_missing_pct:
            issues.append(
                f"Too many missing values: {missing_pct:.2%} > {self.config.max_missing_pct:.2%}"
            )
        elif missing_pct > 0.01:
            warnings.append(f"Missing values present: {missing_pct:.2%}")

        # 3. Price validity
        if (prices <= 0).any():
            issues.append("Found non-positive prices")

        # Check for extreme outliers in prices
        price_zscore = np.abs((prices - prices.mean()) / prices.std())
        if (price_zscore > 10).any():
            warnings.append("Extreme price outliers detected (|z| > 10)")

        # 4. Date ordering
        if not df.index.is_monotonic_increasing:
            issues.append("Dates are not monotonically increasing")

        # 5. Return distribution
        if abs(return_stats['skew']) > 3:
            warnings.append(f"High return skewness: {return_stats['skew']:.2f}")

        if return_stats['kurtosis'] > 20:
            warnings.append(f"High return kurtosis: {return_stats['kurtosis']:.2f}")

        # 6. Check for gaps in data
        date_diffs = pd.Series(df.index).diff().dropna()
        median_diff = date_diffs.median()
        large_gaps = date_diffs[date_diffs > median_diff * 5]
        if len(large_gaps) > 0:
            warnings.append(f"Found {len(large_gaps)} large gaps in time series")

        return DataReport(
            n_rows=n_rows,
            n_cols=n_cols,
            date_range=date_range,
            missing_pct=missing_pct,
            price_range=price_range,
            return_stats=return_stats,
            issues=issues,
            warnings=warnings
        )

    def print_report(self, report: DataReport):
        """Print formatted validation report."""
        print("\n" + "=" * 60)
        print("DATA QUALITY REPORT")
        print("=" * 60)
        print(f"Rows: {report.n_rows:,}")
        print(f"Columns: {report.n_cols}")
        print(f"Period: {report.date_range[0]} to {report.date_range[1]}")
        print(f"Missing: {report.missing_pct:.2%}")
        print(f"Price range: {report.price_range[0]:,.2f} - {report.price_range[1]:,.2f}")

        print("\nReturn Statistics:")
        for key, value in report.return_stats.items():
            print(f"  {key}: {value:.6f}")

        if report.issues:
            print("\nCRITICAL ISSUES:")
            for issue in report.issues:
                print(f"  [X] {issue}")

        if report.warnings:
            print("\nWARNINGS:")
            for warning in report.warnings:
                print(f"  [!] {warning}")

        if report.is_valid:
            print("\nStatus: VALID")
        else:
            print("\nStatus: INVALID - Critical issues found")

        print("=" * 60)
