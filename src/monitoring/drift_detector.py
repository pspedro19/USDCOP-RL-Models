"""
Feature Drift Detection
=======================

Detects feature drift using statistical tests.
Monitors feature distributions over time and alerts when significant
drift is detected.

MLOps-4: Feature Drift Detection
FEAST-19: Multivariate Drift Detection

Features:
- KS test for univariate continuous features
- Multivariate drift detection:
  - Maximum Mean Discrepancy (MMD)
  - Wasserstein distance
  - PCA + reconstruction error
- Sliding window comparison
- Per-feature drift tracking
- Prometheus metrics integration
- Configurable thresholds

Author: Trading Team
Version: 2.0.0
Date: 2026-01-17

CHANGELOG v2.0.0:
- ADDED: Multivariate drift detection methods (FEAST-19)
- ADDED: MMD (Maximum Mean Discrepancy) test
- ADDED: Wasserstein distance calculation
- ADDED: PCA reconstruction error method
- ADDED: Aggregate drift scoring
"""

import json
import logging
from dataclasses import dataclass, asdict
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone

import numpy as np
from scipy import stats

try:
    from prometheus_client import Gauge, Counter
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)


# =============================================================================
# Prometheus Metrics
# =============================================================================

if PROMETHEUS_AVAILABLE:
    DRIFT_SCORE = Gauge(
        'feature_drift_score',
        'Drift score (KS statistic) for each feature',
        ['feature_name']
    )
    DRIFT_PVALUE = Gauge(
        'feature_drift_pvalue',
        'P-value from drift test for each feature',
        ['feature_name']
    )
    DRIFT_ALERT = Gauge(
        'feature_drift_alert',
        'Drift alert active (1) or not (0) for each feature',
        ['feature_name']
    )
    DRIFT_CHECK_TOTAL = Counter(
        'feature_drift_checks_total',
        'Total number of drift checks performed'
    )
    FEATURES_DRIFTED = Gauge(
        'features_drifted_count',
        'Number of features currently in drift state'
    )
else:
    class MockMetric:
        def labels(self, *args, **kwargs):
            return self
        def inc(self, *args, **kwargs):
            pass
        def set(self, *args, **kwargs):
            pass

    DRIFT_SCORE = MockMetric()
    DRIFT_PVALUE = MockMetric()
    DRIFT_ALERT = MockMetric()
    DRIFT_CHECK_TOTAL = MockMetric()
    FEATURES_DRIFTED = MockMetric()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FeatureStats:
    """Statistics for a single feature from reference data."""
    name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    q25: float
    q75: float
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureStats':
        return cls(**data)


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""
    feature_name: str
    reference_mean: float
    current_mean: float
    reference_std: float
    current_std: float
    ks_statistic: float
    p_value: float
    is_drifted: bool
    drift_severity: str  # "none", "low", "medium", "high"
    timestamp: str


@dataclass
class DriftReport:
    """Complete drift detection report."""
    timestamp: str
    features_checked: int
    features_drifted: int
    drift_results: List[DriftResult]
    overall_drift_score: float
    alert_active: bool


# =============================================================================
# Reference Stats Manager
# =============================================================================

class ReferenceStatsManager:
    """
    Manages reference statistics for drift detection.
    Can load from file or compute from data.
    """

    def __init__(self, stats_path: Optional[str] = None):
        self.stats_path = stats_path
        self.feature_stats: Dict[str, FeatureStats] = {}
        self._raw_data: Dict[str, np.ndarray] = {}

        if stats_path and Path(stats_path).exists():
            self.load_stats(stats_path)

    def load_stats(self, path: str) -> bool:
        """Load reference statistics from JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)

            for feature_name, stats_dict in data.get("features", {}).items():
                self.feature_stats[feature_name] = FeatureStats.from_dict(stats_dict)

            # Also store raw data if available
            if "raw_data" in data:
                for feature_name, values in data["raw_data"].items():
                    self._raw_data[feature_name] = np.array(values)

            logger.info(f"Loaded reference stats for {len(self.feature_stats)} features from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load reference stats: {e}")
            return False

    def save_stats(self, path: str) -> bool:
        """Save reference statistics to JSON file."""
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "features": {
                    name: stats.to_dict()
                    for name, stats in self.feature_stats.items()
                },
                "raw_data": {
                    name: arr.tolist()
                    for name, arr in self._raw_data.items()
                }
            }

            with open(path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved reference stats to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save reference stats: {e}")
            return False

    def compute_stats(
        self,
        feature_name: str,
        values: np.ndarray,
        store_raw: bool = True
    ) -> FeatureStats:
        """Compute and store statistics for a feature."""
        values = np.asarray(values).flatten()
        values = values[~np.isnan(values)]  # Remove NaNs

        if len(values) == 0:
            raise ValueError(f"No valid values for feature {feature_name}")

        stats = FeatureStats(
            name=feature_name,
            mean=float(np.mean(values)),
            std=float(np.std(values)),
            min=float(np.min(values)),
            max=float(np.max(values)),
            median=float(np.median(values)),
            q25=float(np.percentile(values, 25)),
            q75=float(np.percentile(values, 75)),
            sample_size=len(values),
        )

        self.feature_stats[feature_name] = stats

        if store_raw:
            # Store sample for KS test (limit size for memory)
            max_samples = 10000
            if len(values) > max_samples:
                indices = np.random.choice(len(values), max_samples, replace=False)
                values = values[indices]
            self._raw_data[feature_name] = values

        return stats

    def compute_all_stats(
        self,
        data: Dict[str, np.ndarray],
        store_raw: bool = True
    ) -> None:
        """Compute stats for all features in a dictionary."""
        for feature_name, values in data.items():
            try:
                self.compute_stats(feature_name, values, store_raw)
            except Exception as e:
                logger.warning(f"Could not compute stats for {feature_name}: {e}")

    def get_reference_data(self, feature_name: str) -> Optional[np.ndarray]:
        """Get raw reference data for a feature."""
        return self._raw_data.get(feature_name)

    def get_stats(self, feature_name: str) -> Optional[FeatureStats]:
        """Get statistics for a feature."""
        return self.feature_stats.get(feature_name)

    @property
    def feature_names(self) -> List[str]:
        """Get list of feature names with reference stats."""
        return list(self.feature_stats.keys())


# =============================================================================
# Feature Drift Detector
# =============================================================================

class FeatureDriftDetector:
    """
    Detect feature drift using the Kolmogorov-Smirnov test.

    Compares current feature distributions against reference distributions
    computed from training or baseline data.

    Usage:
        detector = FeatureDriftDetector(
            reference_stats_path="reference_stats.json",
            p_value_threshold=0.01
        )

        # Add observations during inference
        detector.add_observation(features)

        # Check for drift periodically
        drift_results = detector.check_drift()
        for result in drift_results:
            if result.is_drifted:
                logger.warning(f"Drift detected in {result.feature_name}")
    """

    # Severity thresholds based on KS statistic
    SEVERITY_LOW = 0.1
    SEVERITY_MEDIUM = 0.2
    SEVERITY_HIGH = 0.3

    def __init__(
        self,
        reference_stats_path: Optional[str] = None,
        p_value_threshold: float = 0.01,
        window_size: int = 1000,
        min_samples: int = 100
    ):
        """
        Initialize the drift detector.

        Args:
            reference_stats_path: Path to JSON file with reference statistics
            p_value_threshold: P-value threshold for drift detection (lower = more sensitive)
            window_size: Size of sliding window for current observations
            min_samples: Minimum samples before drift can be detected
        """
        self.p_value_threshold = p_value_threshold
        self.window_size = window_size
        self.min_samples = min_samples

        # Reference statistics
        self.reference_manager = ReferenceStatsManager(reference_stats_path)

        # Current observation windows (per feature)
        self._windows: Dict[str, deque] = {}
        self._drifted_features: set = set()
        self._last_check_results: List[DriftResult] = []

        logger.info(
            f"FeatureDriftDetector initialized: "
            f"p_value_threshold={p_value_threshold}, "
            f"window_size={window_size}, "
            f"reference_features={len(self.reference_manager.feature_names)}"
        )

    def set_reference_data(self, data: Dict[str, np.ndarray]) -> None:
        """
        Set reference data for drift detection.

        Args:
            data: Dictionary mapping feature names to arrays of reference values
        """
        self.reference_manager.compute_all_stats(data, store_raw=True)
        logger.info(f"Reference data set for {len(data)} features")

    def load_reference_stats(self, path: str) -> bool:
        """Load reference statistics from file."""
        return self.reference_manager.load_stats(path)

    def save_reference_stats(self, path: str) -> bool:
        """Save current reference statistics to file."""
        return self.reference_manager.save_stats(path)

    def add_observation(self, features: Dict[str, float]) -> None:
        """
        Add a single observation to the sliding windows.

        Args:
            features: Dictionary mapping feature names to values
        """
        for feature_name, value in features.items():
            if feature_name not in self._windows:
                self._windows[feature_name] = deque(maxlen=self.window_size)

            # Skip NaN values
            if not np.isnan(value):
                self._windows[feature_name].append(float(value))

    def add_batch(self, features_batch: List[Dict[str, float]]) -> None:
        """
        Add a batch of observations.

        Args:
            features_batch: List of feature dictionaries
        """
        for features in features_batch:
            self.add_observation(features)

    def _get_drift_severity(self, ks_statistic: float) -> str:
        """Classify drift severity based on KS statistic."""
        if ks_statistic < self.SEVERITY_LOW:
            return "none"
        elif ks_statistic < self.SEVERITY_MEDIUM:
            return "low"
        elif ks_statistic < self.SEVERITY_HIGH:
            return "medium"
        else:
            return "high"

    def check_drift_single(self, feature_name: str) -> Optional[DriftResult]:
        """
        Check drift for a single feature.

        Args:
            feature_name: Name of the feature to check

        Returns:
            DriftResult or None if check cannot be performed
        """
        # Get current window
        if feature_name not in self._windows:
            return None

        current_values = np.array(list(self._windows[feature_name]))

        if len(current_values) < self.min_samples:
            return None

        # Get reference data
        reference_data = self.reference_manager.get_reference_data(feature_name)
        reference_stats = self.reference_manager.get_stats(feature_name)

        if reference_data is None or reference_stats is None:
            return None

        # Run KS test
        try:
            ks_stat, p_value = stats.ks_2samp(reference_data, current_values)
        except Exception as e:
            logger.warning(f"KS test failed for {feature_name}: {e}")
            return None

        # Determine drift
        is_drifted = p_value < self.p_value_threshold
        severity = self._get_drift_severity(ks_stat)

        result = DriftResult(
            feature_name=feature_name,
            reference_mean=reference_stats.mean,
            current_mean=float(np.mean(current_values)),
            reference_std=reference_stats.std,
            current_std=float(np.std(current_values)),
            ks_statistic=float(ks_stat),
            p_value=float(p_value),
            is_drifted=is_drifted,
            drift_severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        # Update metrics
        DRIFT_SCORE.labels(feature_name=feature_name).set(ks_stat)
        DRIFT_PVALUE.labels(feature_name=feature_name).set(p_value)
        DRIFT_ALERT.labels(feature_name=feature_name).set(1 if is_drifted else 0)

        # Track drifted features
        if is_drifted:
            self._drifted_features.add(feature_name)
        else:
            self._drifted_features.discard(feature_name)

        return result

    def check_drift(self) -> List[DriftResult]:
        """
        Check drift for all features with sufficient data.

        Returns:
            List of DriftResult for each checked feature
        """
        DRIFT_CHECK_TOTAL.inc()

        results = []

        for feature_name in self.reference_manager.feature_names:
            result = self.check_drift_single(feature_name)
            if result is not None:
                results.append(result)

        # Update global metrics
        drifted_count = sum(1 for r in results if r.is_drifted)
        FEATURES_DRIFTED.set(drifted_count)

        self._last_check_results = results

        return results

    def get_drifted_features(self) -> List[str]:
        """Get list of features currently in drift state."""
        return list(self._drifted_features)

    def get_drift_report(self) -> DriftReport:
        """
        Generate comprehensive drift report.

        Returns:
            DriftReport with all drift information
        """
        results = self.check_drift()

        drifted_count = sum(1 for r in results if r.is_drifted)

        # Calculate overall drift score (average KS statistic)
        if results:
            overall_score = np.mean([r.ks_statistic for r in results])
        else:
            overall_score = 0.0

        return DriftReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            features_checked=len(results),
            features_drifted=drifted_count,
            drift_results=results,
            overall_drift_score=float(overall_score),
            alert_active=drifted_count > 0,
        )

    def reset_windows(self) -> None:
        """Clear all observation windows."""
        self._windows.clear()
        self._drifted_features.clear()
        logger.info("Drift detector windows reset")

    @property
    def status(self) -> Dict[str, Any]:
        """Get current detector status."""
        window_sizes = {
            name: len(window)
            for name, window in self._windows.items()
        }

        return {
            "reference_features": len(self.reference_manager.feature_names),
            "monitored_features": len(self._windows),
            "drifted_features": list(self._drifted_features),
            "drifted_count": len(self._drifted_features),
            "p_value_threshold": self.p_value_threshold,
            "window_size": self.window_size,
            "min_samples": self.min_samples,
            "window_fill_levels": window_sizes,
        }


# =============================================================================
# Factory Functions
# =============================================================================

def create_drift_detector(
    reference_stats_path: Optional[str] = None,
    p_value_threshold: float = 0.01,
    window_size: int = 1000
) -> FeatureDriftDetector:
    """
    Factory function to create a configured FeatureDriftDetector.

    Args:
        reference_stats_path: Path to reference statistics file
        p_value_threshold: P-value threshold for drift detection
        window_size: Sliding window size

    Returns:
        Configured FeatureDriftDetector instance
    """
    return FeatureDriftDetector(
        reference_stats_path=reference_stats_path,
        p_value_threshold=p_value_threshold,
        window_size=window_size,
    )


def compute_reference_stats_from_dataframe(
    df,
    feature_columns: List[str],
    output_path: str
) -> ReferenceStatsManager:
    """
    Compute reference statistics from a pandas DataFrame.

    Args:
        df: pandas DataFrame with feature data
        feature_columns: List of column names to use as features
        output_path: Path to save reference statistics

    Returns:
        ReferenceStatsManager with computed statistics
    """
    manager = ReferenceStatsManager()

    data = {}
    for col in feature_columns:
        if col in df.columns:
            data[col] = df[col].values

    manager.compute_all_stats(data)
    manager.save_stats(output_path)

    return manager


# =============================================================================
# MULTIVARIATE DRIFT DETECTION (FEAST-19)
# =============================================================================

@dataclass
class MultivariateDriftResult:
    """Result of multivariate drift detection."""
    timestamp: str
    method: str  # "mmd", "wasserstein", "pca_reconstruction"
    score: float
    threshold: float
    is_drifted: bool
    drift_severity: str
    details: Dict[str, Any]


class MultivariateDriftDetector:
    """
    Multivariate drift detection using multiple statistical methods.

    FEAST-19: Detects drift in the joint distribution of features,
    not just individual features.

    Methods:
    - MMD (Maximum Mean Discrepancy): Kernel-based comparison of distributions
    - Wasserstein Distance: Optimal transport distance
    - PCA Reconstruction Error: Detect changes in feature correlations

    Usage:
        detector = MultivariateDriftDetector(n_features=15)
        detector.set_reference_data(training_observations)

        # During inference
        detector.add_observation(current_observation)

        # Check for drift
        result = detector.check_multivariate_drift()
        if result.is_drifted:
            logger.warning(f"Multivariate drift detected: {result.method}")
    """

    # Severity thresholds
    SEVERITY_THRESHOLDS = {
        "mmd": {"low": 0.05, "medium": 0.1, "high": 0.2},
        "wasserstein": {"low": 0.5, "medium": 1.0, "high": 2.0},
        "pca_reconstruction": {"low": 1.5, "medium": 2.0, "high": 3.0},
    }

    def __init__(
        self,
        n_features: int = 15,
        window_size: int = 500,
        min_samples: int = 100,
        mmd_threshold: float = 0.1,
        wasserstein_threshold: float = 1.0,
        pca_variance_threshold: float = 0.95,
    ):
        """
        Initialize multivariate drift detector.

        Args:
            n_features: Number of features in observation
            window_size: Size of sliding window
            min_samples: Minimum samples before drift can be detected
            mmd_threshold: Threshold for MMD test
            wasserstein_threshold: Threshold for Wasserstein distance
            pca_variance_threshold: Variance to retain in PCA
        """
        self.n_features = n_features
        self.window_size = window_size
        self.min_samples = min_samples
        self.mmd_threshold = mmd_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.pca_variance_threshold = pca_variance_threshold

        # Reference data
        self._reference_data: Optional[np.ndarray] = None
        self._reference_mean: Optional[np.ndarray] = None
        self._reference_cov: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None

        # Current window
        self._current_window: deque = deque(maxlen=window_size)

        logger.info(
            f"MultivariateDriftDetector initialized: n_features={n_features}, "
            f"window_size={window_size}"
        )

    def set_reference_data(self, data: np.ndarray) -> None:
        """
        Set reference data for drift detection.

        Args:
            data: 2D array of shape (n_samples, n_features)
        """
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError(f"Expected 2D array, got {data.ndim}D")

        if data.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {data.shape[1]}"
            )

        # Remove NaN rows
        valid_mask = ~np.any(np.isnan(data), axis=1)
        data = data[valid_mask]

        if len(data) < self.min_samples:
            raise ValueError(
                f"Need at least {self.min_samples} samples, got {len(data)}"
            )

        self._reference_data = data
        self._reference_mean = np.mean(data, axis=0)
        self._reference_cov = np.cov(data.T)

        # Fit PCA
        self._fit_pca(data)

        logger.info(f"Reference data set: {data.shape[0]} samples")

    def _fit_pca(self, data: np.ndarray) -> None:
        """Fit PCA on reference data."""
        # Center data
        self._pca_mean = np.mean(data, axis=0)
        centered = data - self._pca_mean

        # SVD
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # Calculate explained variance
            explained_variance = (S ** 2) / (len(data) - 1)
            total_variance = explained_variance.sum()
            explained_variance_ratio = explained_variance / total_variance

            # Select components
            cumsum = np.cumsum(explained_variance_ratio)
            n_components = np.searchsorted(cumsum, self.pca_variance_threshold) + 1
            n_components = min(n_components, len(S))

            self._pca_components = Vt[:n_components]
            self._pca_explained_variance = explained_variance[:n_components]

            logger.info(
                f"PCA fitted: {n_components} components, "
                f"{cumsum[n_components-1]:.2%} variance explained"
            )

        except np.linalg.LinAlgError as e:
            logger.warning(f"PCA fitting failed: {e}")
            self._pca_components = None

    def add_observation(self, observation: np.ndarray) -> None:
        """
        Add a single observation to the current window.

        Args:
            observation: 1D array of shape (n_features,)
        """
        observation = np.asarray(observation).flatten()

        if len(observation) != self.n_features:
            logger.warning(
                f"Expected {self.n_features} features, got {len(observation)}"
            )
            return

        if np.any(np.isnan(observation)):
            return

        self._current_window.append(observation)

    def add_batch(self, observations: np.ndarray) -> None:
        """Add batch of observations."""
        observations = np.asarray(observations)
        if observations.ndim == 1:
            self.add_observation(observations)
        else:
            for obs in observations:
                self.add_observation(obs)

    def _get_current_data(self) -> Optional[np.ndarray]:
        """Get current window as numpy array."""
        if len(self._current_window) < self.min_samples:
            return None
        return np.array(list(self._current_window))

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """Compute RBF kernel mean."""
        # Efficient computation using broadcasting
        X_sqnorm = np.sum(X ** 2, axis=1)
        Y_sqnorm = np.sum(Y ** 2, axis=1)
        XY = X @ Y.T

        K = np.exp(-gamma * (
            X_sqnorm[:, np.newaxis] + Y_sqnorm[np.newaxis, :] - 2 * XY
        ))
        return K.mean()

    def compute_mmd(self) -> Optional[MultivariateDriftResult]:
        """
        Compute Maximum Mean Discrepancy between reference and current data.

        MMD is a kernel-based distance measure between distributions.

        Returns:
            MultivariateDriftResult or None if insufficient data
        """
        current_data = self._get_current_data()
        if current_data is None or self._reference_data is None:
            return None

        # Subsample reference for efficiency
        ref_sample_size = min(len(self._reference_data), 500)
        ref_indices = np.random.choice(
            len(self._reference_data), ref_sample_size, replace=False
        )
        ref_sample = self._reference_data[ref_indices]

        # Compute MMD
        # MMD^2 = E[k(X,X')] + E[k(Y,Y')] - 2*E[k(X,Y)]
        gamma = 1.0 / self.n_features  # Bandwidth

        k_xx = self._rbf_kernel(ref_sample, ref_sample, gamma)
        k_yy = self._rbf_kernel(current_data, current_data, gamma)
        k_xy = self._rbf_kernel(ref_sample, current_data, gamma)

        mmd_squared = k_xx + k_yy - 2 * k_xy
        mmd = np.sqrt(max(0, mmd_squared))

        # Determine drift
        is_drifted = mmd > self.mmd_threshold
        severity = self._get_severity("mmd", mmd)

        return MultivariateDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            method="mmd",
            score=float(mmd),
            threshold=self.mmd_threshold,
            is_drifted=is_drifted,
            drift_severity=severity,
            details={
                "k_xx": float(k_xx),
                "k_yy": float(k_yy),
                "k_xy": float(k_xy),
                "ref_samples": ref_sample_size,
                "current_samples": len(current_data),
            },
        )

    def compute_wasserstein(self) -> Optional[MultivariateDriftResult]:
        """
        Compute approximate Wasserstein distance.

        Uses sliced Wasserstein distance for efficiency.

        Returns:
            MultivariateDriftResult or None if insufficient data
        """
        current_data = self._get_current_data()
        if current_data is None or self._reference_data is None:
            return None

        # Sliced Wasserstein: project to 1D and compute W1
        n_projections = 50
        distances = []

        for _ in range(n_projections):
            # Random projection direction
            direction = np.random.randn(self.n_features)
            direction /= np.linalg.norm(direction)

            # Project
            ref_proj = self._reference_data @ direction
            curr_proj = current_data @ direction

            # 1D Wasserstein (sorted quantile comparison)
            try:
                w1 = stats.wasserstein_distance(ref_proj, curr_proj)
                distances.append(w1)
            except Exception:
                pass

        if not distances:
            return None

        # Average over projections
        avg_wasserstein = np.mean(distances)

        is_drifted = avg_wasserstein > self.wasserstein_threshold
        severity = self._get_severity("wasserstein", avg_wasserstein)

        return MultivariateDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            method="wasserstein",
            score=float(avg_wasserstein),
            threshold=self.wasserstein_threshold,
            is_drifted=is_drifted,
            drift_severity=severity,
            details={
                "n_projections": n_projections,
                "min_distance": float(min(distances)),
                "max_distance": float(max(distances)),
                "current_samples": len(current_data),
            },
        )

    def compute_pca_reconstruction_error(self) -> Optional[MultivariateDriftResult]:
        """
        Compute PCA reconstruction error for drift detection.

        High reconstruction error indicates the feature correlations
        have changed from the reference period.

        Returns:
            MultivariateDriftResult or None if insufficient data
        """
        current_data = self._get_current_data()
        if current_data is None or self._pca_components is None:
            return None

        # Center current data
        centered = current_data - self._pca_mean

        # Project to PCA space and reconstruct
        projected = centered @ self._pca_components.T
        reconstructed = projected @ self._pca_components

        # Compute reconstruction error
        errors = np.sqrt(np.sum((centered - reconstructed) ** 2, axis=1))
        mean_error = np.mean(errors)

        # Compare to reference error
        ref_centered = self._reference_data - self._pca_mean
        ref_projected = ref_centered @ self._pca_components.T
        ref_reconstructed = ref_projected @ self._pca_components
        ref_errors = np.sqrt(np.sum((ref_centered - ref_reconstructed) ** 2, axis=1))
        ref_mean_error = np.mean(ref_errors)
        ref_std_error = np.std(ref_errors)

        # Z-score of current error relative to reference
        if ref_std_error > 0:
            z_score = (mean_error - ref_mean_error) / ref_std_error
        else:
            z_score = 0.0

        # Threshold based on z-score
        pca_threshold = 2.0  # 2 standard deviations
        is_drifted = z_score > pca_threshold
        severity = self._get_severity("pca_reconstruction", z_score)

        return MultivariateDriftResult(
            timestamp=datetime.now(timezone.utc).isoformat(),
            method="pca_reconstruction",
            score=float(z_score),
            threshold=pca_threshold,
            is_drifted=is_drifted,
            drift_severity=severity,
            details={
                "mean_reconstruction_error": float(mean_error),
                "reference_mean_error": float(ref_mean_error),
                "reference_std_error": float(ref_std_error),
                "n_components": len(self._pca_components),
                "current_samples": len(current_data),
            },
        )

    def _get_severity(self, method: str, score: float) -> str:
        """Get severity level based on score."""
        thresholds = self.SEVERITY_THRESHOLDS.get(method, {})
        if score >= thresholds.get("high", float("inf")):
            return "high"
        elif score >= thresholds.get("medium", float("inf")):
            return "medium"
        elif score >= thresholds.get("low", float("inf")):
            return "low"
        return "none"

    def check_multivariate_drift(
        self,
        methods: Optional[List[str]] = None
    ) -> Dict[str, MultivariateDriftResult]:
        """
        Run all multivariate drift detection methods.

        Args:
            methods: List of methods to run. Default: all methods

        Returns:
            Dictionary mapping method name to result
        """
        if methods is None:
            methods = ["mmd", "wasserstein", "pca_reconstruction"]

        results = {}

        if "mmd" in methods:
            result = self.compute_mmd()
            if result:
                results["mmd"] = result

        if "wasserstein" in methods:
            result = self.compute_wasserstein()
            if result:
                results["wasserstein"] = result

        if "pca_reconstruction" in methods:
            result = self.compute_pca_reconstruction_error()
            if result:
                results["pca_reconstruction"] = result

        return results

    def get_aggregate_drift_score(self) -> Dict[str, Any]:
        """
        Get aggregate drift score combining all methods.

        Returns:
            Dictionary with aggregate metrics
        """
        results = self.check_multivariate_drift()

        if not results:
            return {
                "status": "insufficient_data",
                "methods_checked": 0,
                "any_drifted": False,
            }

        any_drifted = any(r.is_drifted for r in results.values())
        severities = [r.drift_severity for r in results.values()]

        # Highest severity
        severity_order = ["none", "low", "medium", "high"]
        max_severity = max(severities, key=lambda s: severity_order.index(s))

        return {
            "status": "checked",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "methods_checked": len(results),
            "any_drifted": any_drifted,
            "max_severity": max_severity,
            "method_results": {
                method: {
                    "score": r.score,
                    "threshold": r.threshold,
                    "is_drifted": r.is_drifted,
                    "severity": r.drift_severity,
                }
                for method, r in results.items()
            },
        }

    def reset(self) -> None:
        """Reset current window."""
        self._current_window.clear()

    @property
    def status(self) -> Dict[str, Any]:
        """Get detector status."""
        return {
            "n_features": self.n_features,
            "window_size": self.window_size,
            "current_samples": len(self._current_window),
            "has_reference": self._reference_data is not None,
            "reference_samples": len(self._reference_data) if self._reference_data is not None else 0,
            "pca_fitted": self._pca_components is not None,
        }


def create_multivariate_drift_detector(
    n_features: int = 15,
    window_size: int = 500,
) -> MultivariateDriftDetector:
    """
    Factory function to create MultivariateDriftDetector.

    Args:
        n_features: Number of features
        window_size: Sliding window size

    Returns:
        Configured MultivariateDriftDetector
    """
    return MultivariateDriftDetector(
        n_features=n_features,
        window_size=window_size,
    )
