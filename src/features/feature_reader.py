"""
Feature Reader Module
=====================

Read and validate features from the feature store for inference.

This module provides a clean interface for reading features with:
- Staleness detection (max age validation)
- Feature order validation
- Missing feature handling
- Caching support

Components:
- FeatureReader: Main class for reading features
- FeatureReadResult: Container for read results
- FeatureValidationError: Raised on validation failures

Author: USD/COP Trading System
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import numpy as np


# Expected feature order for model input
EXPECTED_FEATURE_ORDER = [
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    "rsi_9",
    "atr_pct",
    "adx_14",
    "dxy_z",
    "dxy_change_1d",
    "vix_z",
    "embi_z",
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d",
]


@dataclass
class FeatureReadResult:
    """Result container for feature read operations."""

    # Feature data
    features: Optional[Dict[str, float]] = None
    feature_vector: Optional[np.ndarray] = None

    # Metadata
    timestamp: Optional[datetime] = None
    age_seconds: float = 0.0

    # Validation status
    is_valid: bool = False
    is_stale: bool = False
    missing_features: List[str] = field(default_factory=list)
    extra_features: List[str] = field(default_factory=list)

    # Error info
    error: Optional[str] = None


class FeatureValidationError(Exception):
    """Raised when feature validation fails."""

    def __init__(self, message: str, missing: Optional[List[str]] = None):
        super().__init__(message)
        self.missing = missing or []


class FeatureReader:
    """Reader for features from the feature store.

    This class provides a clean interface for reading and validating
    features before model inference.

    Attributes:
        expected_features: List of expected feature names in order
        max_age_seconds: Maximum allowed age for features
        strict_order: Whether to enforce strict feature ordering

    Example:
        reader = FeatureReader(max_age_seconds=300)
        result = reader.get_latest_features()
        if result.is_valid and not result.is_stale:
            model.predict(result.feature_vector)
    """

    def __init__(
        self,
        expected_features: Optional[List[str]] = None,
        max_age_seconds: float = 300.0,
        strict_order: bool = True,
        default_values: Optional[Dict[str, float]] = None,
    ):
        """Initialize the feature reader.

        Args:
            expected_features: List of expected feature names
            max_age_seconds: Maximum age in seconds for valid features
            strict_order: Enforce feature order matching
            default_values: Default values for missing features
        """
        self.expected_features = expected_features or EXPECTED_FEATURE_ORDER.copy()
        self.max_age_seconds = max_age_seconds
        self.strict_order = strict_order
        self.default_values = default_values or {}

        # Internal state
        self._feature_cache: Optional[Dict[str, float]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._feature_store: Dict[str, Dict[str, Any]] = {}

    def set_feature_store(self, store: Dict[str, Dict[str, Any]]) -> None:
        """Set the feature store to read from.

        Args:
            store: Dictionary mapping feature names to their values and metadata
        """
        self._feature_store = store

    def add_features(
        self,
        features: Dict[str, float],
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add features to the internal store (for testing).

        Args:
            features: Dictionary of feature name to value
            timestamp: Timestamp for the features
        """
        ts = timestamp or datetime.now()
        for name, value in features.items():
            self._feature_store[name] = {
                'value': value,
                'timestamp': ts,
            }

    def clear_features(self) -> None:
        """Clear all stored features."""
        self._feature_store.clear()
        self._feature_cache = None
        self._cache_timestamp = None

    def get_latest_features(
        self,
        timestamp: Optional[datetime] = None
    ) -> FeatureReadResult:
        """Get the latest features from the store.

        Args:
            timestamp: Reference timestamp for age calculation (default: now)

        Returns:
            FeatureReadResult with features and validation status
        """
        result = FeatureReadResult()
        ref_time = timestamp or datetime.now()

        # Check if store is empty
        if not self._feature_store:
            result.error = "No features available"
            result.is_valid = False
            return result

        # Extract features
        features: Dict[str, float] = {}
        oldest_timestamp: Optional[datetime] = None

        for feature_name in self.expected_features:
            if feature_name in self._feature_store:
                entry = self._feature_store[feature_name]
                features[feature_name] = entry['value']

                # Track oldest timestamp
                feature_ts = entry.get('timestamp')
                if feature_ts is not None:
                    if oldest_timestamp is None or feature_ts < oldest_timestamp:
                        oldest_timestamp = feature_ts
            elif feature_name in self.default_values:
                features[feature_name] = self.default_values[feature_name]

        # Validate features
        validation = self._validate_features(features)
        result.missing_features = validation['missing']
        result.extra_features = validation['extra']

        if validation['missing']:
            result.error = f"Missing features: {validation['missing']}"
            result.is_valid = False
            return result

        # Calculate age
        if oldest_timestamp:
            result.timestamp = oldest_timestamp
            result.age_seconds = (ref_time - oldest_timestamp).total_seconds()
            result.is_stale = result.age_seconds > self.max_age_seconds

        # Build feature vector
        result.features = features
        result.feature_vector = self._build_feature_vector(features)
        result.is_valid = True

        return result

    def validate_feature_order(
        self,
        feature_names: List[str]
    ) -> Tuple[bool, List[str]]:
        """Validate that feature names match expected order.

        Args:
            feature_names: List of feature names to validate

        Returns:
            Tuple of (is_valid, list_of_differences)
        """
        differences: List[str] = []

        if len(feature_names) != len(self.expected_features):
            differences.append(
                f"Length mismatch: got {len(feature_names)}, "
                f"expected {len(self.expected_features)}"
            )

        for i, (got, expected) in enumerate(
            zip(feature_names, self.expected_features)
        ):
            if got != expected:
                differences.append(
                    f"Position {i}: got '{got}', expected '{expected}'"
                )

        # Check for extra features
        expected_set = set(self.expected_features)
        got_set = set(feature_names)
        extra = got_set - expected_set
        missing = expected_set - got_set

        if extra:
            differences.append(f"Unexpected features: {sorted(extra)}")
        if missing:
            differences.append(f"Missing features: {sorted(missing)}")

        return len(differences) == 0, differences

    def check_max_age(
        self,
        feature_timestamp: datetime,
        reference_time: Optional[datetime] = None
    ) -> Tuple[bool, float]:
        """Check if features are within max age limit.

        Args:
            feature_timestamp: Timestamp of the features
            reference_time: Reference time for comparison (default: now)

        Returns:
            Tuple of (is_valid, age_in_seconds)
        """
        ref_time = reference_time or datetime.now()
        age_seconds = (ref_time - feature_timestamp).total_seconds()
        is_valid = age_seconds <= self.max_age_seconds
        return is_valid, age_seconds

    def _validate_features(
        self,
        features: Dict[str, float]
    ) -> Dict[str, List[str]]:
        """Internal validation of feature dictionary.

        Args:
            features: Dictionary of features to validate

        Returns:
            Dictionary with 'missing' and 'extra' feature lists
        """
        got_set = set(features.keys())
        expected_set = set(self.expected_features)

        return {
            'missing': sorted(expected_set - got_set),
            'extra': sorted(got_set - expected_set),
        }

    def _build_feature_vector(
        self,
        features: Dict[str, float]
    ) -> np.ndarray:
        """Build numpy array from features in correct order.

        Args:
            features: Dictionary of feature name to value

        Returns:
            Numpy array with features in expected order
        """
        vector = np.zeros(len(self.expected_features), dtype=np.float32)

        for i, name in enumerate(self.expected_features):
            if name in features:
                vector[i] = features[name]
            elif name in self.default_values:
                vector[i] = self.default_values[name]
            # else: remains 0.0

        return vector


__all__ = [
    'FeatureReader',
    'FeatureReadResult',
    'FeatureValidationError',
    'EXPECTED_FEATURE_ORDER',
]
