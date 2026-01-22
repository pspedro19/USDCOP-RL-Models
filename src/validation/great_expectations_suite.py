"""
Great Expectations Data Validation Suite
=========================================

Comprehensive data validation for feature pipelines using Great Expectations.
Validates feature data quality, schema, and distribution constraints.

P1: Data Quality Validation

Features:
- Column type validation
- Null percentage thresholds
- Value range validation (RSI: 0-100, etc.)
- Distribution checks
- Freshness validation
- Custom feature-specific expectations

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Feature Expectations Configuration
# =============================================================================

@dataclass
class FeatureExpectation:
    """Configuration for a single feature's validation expectations."""
    name: str
    dtype: str  # "float64", "int64", "datetime64[ns]", etc.
    nullable: bool = True
    max_null_percentage: float = 5.0  # Maximum allowed null percentage
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    must_be_positive: bool = False
    must_be_negative: bool = False
    expected_mean_range: Optional[Tuple[float, float]] = None
    expected_std_range: Optional[Tuple[float, float]] = None


# Default expectations for common trading features
DEFAULT_FEATURE_EXPECTATIONS = {
    # Technical indicators
    "rsi_14": FeatureExpectation(
        name="rsi_14",
        dtype="float64",
        max_null_percentage=5.0,
        min_value=0.0,
        max_value=100.0,
        expected_mean_range=(30.0, 70.0),
    ),
    "macd": FeatureExpectation(
        name="macd",
        dtype="float64",
        max_null_percentage=5.0,
    ),
    "macd_signal": FeatureExpectation(
        name="macd_signal",
        dtype="float64",
        max_null_percentage=5.0,
    ),
    "bb_upper": FeatureExpectation(
        name="bb_upper",
        dtype="float64",
        max_null_percentage=5.0,
        must_be_positive=True,
    ),
    "bb_lower": FeatureExpectation(
        name="bb_lower",
        dtype="float64",
        max_null_percentage=5.0,
        must_be_positive=True,
    ),
    "atr_14": FeatureExpectation(
        name="atr_14",
        dtype="float64",
        max_null_percentage=5.0,
        must_be_positive=True,
    ),

    # Log returns
    "log_ret_5m": FeatureExpectation(
        name="log_ret_5m",
        dtype="float64",
        max_null_percentage=1.0,
        min_value=-0.5,  # 50% move limit
        max_value=0.5,
        expected_mean_range=(-0.001, 0.001),
    ),
    "log_ret_1h": FeatureExpectation(
        name="log_ret_1h",
        dtype="float64",
        max_null_percentage=5.0,
        min_value=-1.0,
        max_value=1.0,
    ),
    "log_ret_1d": FeatureExpectation(
        name="log_ret_1d",
        dtype="float64",
        max_null_percentage=5.0,
        min_value=-1.0,
        max_value=1.0,
    ),

    # Price features
    "close": FeatureExpectation(
        name="close",
        dtype="float64",
        max_null_percentage=0.0,  # No nulls allowed
        must_be_positive=True,
        min_value=1000.0,  # USD/COP minimum
        max_value=10000.0,  # USD/COP maximum
    ),

    # Volume
    "volume": FeatureExpectation(
        name="volume",
        dtype="float64",
        max_null_percentage=5.0,
        min_value=0.0,
    ),

    # Macro indicators
    "dxy": FeatureExpectation(
        name="dxy",
        dtype="float64",
        max_null_percentage=10.0,
        min_value=50.0,
        max_value=200.0,
    ),
    "vix": FeatureExpectation(
        name="vix",
        dtype="float64",
        max_null_percentage=10.0,
        min_value=5.0,
        max_value=100.0,
    ),
}


# =============================================================================
# Validation Result Classes
# =============================================================================

@dataclass
class ExpectationResult:
    """Result of a single expectation check."""
    expectation_type: str
    feature_name: str
    success: bool
    observed_value: Any
    expected_value: Any
    details: str


@dataclass
class ValidationResult:
    """Complete validation result for a dataset."""
    timestamp: str
    success: bool
    total_expectations: int
    failed_expectations: int
    passed_expectations: int
    results: List[ExpectationResult]
    summary: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "success": self.success,
            "total_expectations": self.total_expectations,
            "failed_expectations": self.failed_expectations,
            "passed_expectations": self.passed_expectations,
            "pass_rate": self.passed_expectations / self.total_expectations if self.total_expectations > 0 else 0,
            "results": [
                {
                    "expectation_type": r.expectation_type,
                    "feature_name": r.feature_name,
                    "success": r.success,
                    "observed_value": str(r.observed_value),
                    "expected_value": str(r.expected_value),
                    "details": r.details,
                }
                for r in self.results
            ],
            "summary": self.summary,
        }


# =============================================================================
# Feature Validator
# =============================================================================

class FeatureValidator:
    """
    Validates feature data against configured expectations.

    This validator provides Great Expectations-style validation without
    requiring the full Great Expectations library, making it lighter
    and easier to integrate into existing pipelines.

    Usage:
        validator = FeatureValidator(expectations=DEFAULT_FEATURE_EXPECTATIONS)
        result = validator.validate(df)

        if not result.success:
            for r in result.results:
                if not r.success:
                    logger.warning(f"Validation failed: {r.details}")
    """

    def __init__(
        self,
        expectations: Optional[Dict[str, FeatureExpectation]] = None,
        strict_mode: bool = False,
        max_freshness_hours: float = 24.0,
    ):
        """
        Initialize the feature validator.

        Args:
            expectations: Dictionary of feature expectations
            strict_mode: If True, fail on any unexpected column
            max_freshness_hours: Maximum age of data in hours
        """
        self.expectations = expectations or DEFAULT_FEATURE_EXPECTATIONS
        self.strict_mode = strict_mode
        self.max_freshness_hours = max_freshness_hours
        self._results: List[ExpectationResult] = []

    def _add_result(
        self,
        expectation_type: str,
        feature_name: str,
        success: bool,
        observed: Any,
        expected: Any,
        details: str,
    ) -> None:
        """Add a validation result."""
        self._results.append(ExpectationResult(
            expectation_type=expectation_type,
            feature_name=feature_name,
            success=success,
            observed_value=observed,
            expected_value=expected,
            details=details,
        ))

    def _validate_column_exists(self, df: pd.DataFrame, column: str) -> bool:
        """Check if column exists in dataframe."""
        exists = column in df.columns
        self._add_result(
            expectation_type="column_exists",
            feature_name=column,
            success=exists,
            observed=column in df.columns,
            expected=True,
            details=f"Column '{column}' {'exists' if exists else 'does not exist'}",
        )
        return exists

    def _validate_column_dtype(
        self,
        df: pd.DataFrame,
        column: str,
        expected_dtype: str,
    ) -> bool:
        """Validate column data type."""
        if column not in df.columns:
            return False

        actual_dtype = str(df[column].dtype)

        # Allow some flexibility in dtype matching
        dtype_matches = (
            actual_dtype == expected_dtype or
            (expected_dtype == "float64" and actual_dtype in ["float32", "float64"]) or
            (expected_dtype == "int64" and actual_dtype in ["int32", "int64"])
        )

        self._add_result(
            expectation_type="column_dtype",
            feature_name=column,
            success=dtype_matches,
            observed=actual_dtype,
            expected=expected_dtype,
            details=f"Column '{column}' dtype is {actual_dtype}, expected {expected_dtype}",
        )
        return dtype_matches

    def _validate_null_percentage(
        self,
        df: pd.DataFrame,
        column: str,
        max_percentage: float,
    ) -> bool:
        """Validate null percentage is below threshold."""
        if column not in df.columns:
            return False

        null_count = df[column].isna().sum()
        total_count = len(df)
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0

        success = null_percentage <= max_percentage

        self._add_result(
            expectation_type="null_percentage",
            feature_name=column,
            success=success,
            observed=f"{null_percentage:.2f}%",
            expected=f"<= {max_percentage}%",
            details=f"Column '{column}' has {null_percentage:.2f}% nulls ({null_count}/{total_count})",
        )
        return success

    def _validate_value_range(
        self,
        df: pd.DataFrame,
        column: str,
        min_value: Optional[float],
        max_value: Optional[float],
    ) -> bool:
        """Validate values are within expected range."""
        if column not in df.columns:
            return False

        series = df[column].dropna()
        if len(series) == 0:
            self._add_result(
                expectation_type="value_range",
                feature_name=column,
                success=True,
                observed="No data",
                expected=f"[{min_value}, {max_value}]",
                details=f"Column '{column}' has no non-null values to check",
            )
            return True

        actual_min = series.min()
        actual_max = series.max()

        min_ok = min_value is None or actual_min >= min_value
        max_ok = max_value is None or actual_max <= max_value
        success = min_ok and max_ok

        self._add_result(
            expectation_type="value_range",
            feature_name=column,
            success=success,
            observed=f"[{actual_min:.4f}, {actual_max:.4f}]",
            expected=f"[{min_value}, {max_value}]",
            details=f"Column '{column}' range is [{actual_min:.4f}, {actual_max:.4f}]",
        )
        return success

    def _validate_positive_values(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> bool:
        """Validate all values are positive."""
        if column not in df.columns:
            return False

        series = df[column].dropna()
        if len(series) == 0:
            return True

        negative_count = (series < 0).sum()
        success = negative_count == 0

        self._add_result(
            expectation_type="positive_values",
            feature_name=column,
            success=success,
            observed=f"{negative_count} negative values",
            expected="0 negative values",
            details=f"Column '{column}' has {negative_count} negative values",
        )
        return success

    def _validate_mean_range(
        self,
        df: pd.DataFrame,
        column: str,
        expected_range: Tuple[float, float],
    ) -> bool:
        """Validate mean is within expected range."""
        if column not in df.columns:
            return False

        series = df[column].dropna()
        if len(series) == 0:
            return True

        actual_mean = series.mean()
        success = expected_range[0] <= actual_mean <= expected_range[1]

        self._add_result(
            expectation_type="mean_range",
            feature_name=column,
            success=success,
            observed=f"{actual_mean:.4f}",
            expected=f"[{expected_range[0]}, {expected_range[1]}]",
            details=f"Column '{column}' mean is {actual_mean:.4f}",
        )
        return success

    def _validate_freshness(
        self,
        df: pd.DataFrame,
        timestamp_column: str = "timestamp",
    ) -> bool:
        """Validate data freshness."""
        if timestamp_column not in df.columns:
            self._add_result(
                expectation_type="freshness",
                feature_name=timestamp_column,
                success=False,
                observed="Column not found",
                expected="timestamp column exists",
                details=f"Timestamp column '{timestamp_column}' not found",
            )
            return False

        try:
            timestamps = pd.to_datetime(df[timestamp_column])
            max_timestamp = timestamps.max()

            if pd.isna(max_timestamp):
                success = False
                details = "No valid timestamps found"
            else:
                now = datetime.now(timezone.utc)
                if max_timestamp.tzinfo is None:
                    max_timestamp = max_timestamp.replace(tzinfo=timezone.utc)

                age_hours = (now - max_timestamp).total_seconds() / 3600
                success = age_hours <= self.max_freshness_hours
                details = f"Most recent data is {age_hours:.1f} hours old"

            self._add_result(
                expectation_type="freshness",
                feature_name=timestamp_column,
                success=success,
                observed=f"{age_hours:.1f} hours" if success else details,
                expected=f"<= {self.max_freshness_hours} hours",
                details=details,
            )
            return success

        except Exception as e:
            self._add_result(
                expectation_type="freshness",
                feature_name=timestamp_column,
                success=False,
                observed=str(e),
                expected="Valid timestamps",
                details=f"Error checking freshness: {e}",
            )
            return False

    def validate(
        self,
        df: pd.DataFrame,
        check_freshness: bool = True,
        timestamp_column: str = "timestamp",
    ) -> ValidationResult:
        """
        Validate a dataframe against all configured expectations.

        Args:
            df: DataFrame to validate
            check_freshness: Whether to check data freshness
            timestamp_column: Column containing timestamps

        Returns:
            ValidationResult with all expectation results
        """
        self._results = []

        # Dataset-level checks
        if len(df) == 0:
            self._add_result(
                expectation_type="row_count",
                feature_name="dataset",
                success=False,
                observed=0,
                expected="> 0",
                details="Dataset is empty",
            )
        else:
            self._add_result(
                expectation_type="row_count",
                feature_name="dataset",
                success=True,
                observed=len(df),
                expected="> 0",
                details=f"Dataset has {len(df)} rows",
            )

        # Check freshness
        if check_freshness and timestamp_column in df.columns:
            self._validate_freshness(df, timestamp_column)

        # Feature-level checks
        for feature_name, expectation in self.expectations.items():
            # Column exists
            if not self._validate_column_exists(df, feature_name):
                continue

            # Data type
            self._validate_column_dtype(df, feature_name, expectation.dtype)

            # Null percentage
            self._validate_null_percentage(
                df, feature_name, expectation.max_null_percentage
            )

            # Value range
            if expectation.min_value is not None or expectation.max_value is not None:
                self._validate_value_range(
                    df, feature_name, expectation.min_value, expectation.max_value
                )

            # Positive values
            if expectation.must_be_positive:
                self._validate_positive_values(df, feature_name)

            # Mean range
            if expectation.expected_mean_range is not None:
                self._validate_mean_range(
                    df, feature_name, expectation.expected_mean_range
                )

        # Compile results
        failed = [r for r in self._results if not r.success]
        passed = [r for r in self._results if r.success]

        return ValidationResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            success=len(failed) == 0,
            total_expectations=len(self._results),
            failed_expectations=len(failed),
            passed_expectations=len(passed),
            results=self._results,
            summary={
                "row_count": len(df),
                "column_count": len(df.columns),
                "features_validated": len(self.expectations),
                "failed_features": [r.feature_name for r in failed],
            },
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def validate_features(
    df: pd.DataFrame,
    expectations: Optional[Dict[str, FeatureExpectation]] = None,
    raise_on_failure: bool = False,
) -> ValidationResult:
    """
    Convenience function to validate features.

    Args:
        df: DataFrame to validate
        expectations: Custom expectations (uses defaults if None)
        raise_on_failure: If True, raise exception on validation failure

    Returns:
        ValidationResult

    Raises:
        ValueError: If raise_on_failure=True and validation fails
    """
    validator = FeatureValidator(expectations=expectations)
    result = validator.validate(df)

    if raise_on_failure and not result.success:
        failed_details = [
            f"{r.feature_name}: {r.details}"
            for r in result.results
            if not r.success
        ]
        raise ValueError(
            f"Feature validation failed with {result.failed_expectations} errors:\n"
            + "\n".join(failed_details[:10])  # Show first 10 errors
        )

    return result


def create_validator_for_inference() -> FeatureValidator:
    """Create a validator configured for inference features (15 dimensions)."""
    inference_expectations = {
        "log_ret_5m": DEFAULT_FEATURE_EXPECTATIONS["log_ret_5m"],
        "rsi_14": DEFAULT_FEATURE_EXPECTATIONS["rsi_14"],
        "macd": DEFAULT_FEATURE_EXPECTATIONS["macd"],
        "macd_signal": DEFAULT_FEATURE_EXPECTATIONS["macd_signal"],
        "bb_upper": DEFAULT_FEATURE_EXPECTATIONS["bb_upper"],
        "bb_lower": DEFAULT_FEATURE_EXPECTATIONS["bb_lower"],
        "atr_14": DEFAULT_FEATURE_EXPECTATIONS["atr_14"],
        "dxy": DEFAULT_FEATURE_EXPECTATIONS["dxy"],
        "vix": DEFAULT_FEATURE_EXPECTATIONS["vix"],
    }

    return FeatureValidator(
        expectations=inference_expectations,
        max_freshness_hours=2.0,  # Tighter freshness for inference
    )


__all__ = [
    "FeatureExpectation",
    "ExpectationResult",
    "ValidationResult",
    "FeatureValidator",
    "DEFAULT_FEATURE_EXPECTATIONS",
    "validate_features",
    "create_validator_for_inference",
]
