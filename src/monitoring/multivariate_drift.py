"""
Multivariate Drift Detection
=============================

Extends the basic drift detector with multivariate methods that capture
correlations between features.

P1: Multivariate Drift Detection

Methods:
- Maximum Mean Discrepancy (MMD)
- Wasserstein distance (Earth Mover's Distance)
- PCA + Reconstruction Error
- Multivariate KS test

Author: Trading Team
Version: 1.0.0
Date: 2026-01-17
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class MultivariateDriftResult:
    """Result of multivariate drift detection."""
    method: str
    statistic: float
    p_value: Optional[float]
    threshold: float
    is_drifted: bool
    drift_severity: str  # "none", "low", "medium", "high"
    feature_contributions: Optional[Dict[str, float]]  # Per-feature contribution
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "method": self.method,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "is_drifted": self.is_drifted,
            "drift_severity": self.drift_severity,
            "feature_contributions": self.feature_contributions,
            "timestamp": self.timestamp,
        }


@dataclass
class MultivariateDriftReport:
    """Complete multivariate drift report."""
    timestamp: str
    overall_is_drifted: bool
    results: List[MultivariateDriftResult]
    pca_variance_explained: Optional[float]
    correlation_change: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "overall_is_drifted": self.overall_is_drifted,
            "results": [r.to_dict() for r in self.results],
            "pca_variance_explained": self.pca_variance_explained,
            "correlation_change": self.correlation_change,
        }


# =============================================================================
# Multivariate Drift Detector
# =============================================================================

class MultivariateDriftDetector:
    """
    Multivariate drift detection using multiple methods.

    This detector captures drift in feature correlations and joint
    distributions, which univariate tests may miss.

    Usage:
        detector = MultivariateDriftDetector()

        # Set reference data
        detector.set_reference(reference_features)

        # Check current data for drift
        report = detector.detect_drift(current_features)

        if report.overall_is_drifted:
            logger.warning("Multivariate drift detected!")
    """

    # Severity thresholds
    SEVERITY_LOW = 0.1
    SEVERITY_MEDIUM = 0.2
    SEVERITY_HIGH = 0.3

    def __init__(
        self,
        mmd_threshold: float = 0.1,
        wasserstein_threshold: float = 0.5,
        pca_threshold: float = 2.0,  # Standard deviations
        n_permutations: int = 100,
        n_pca_components: int = 5,
    ):
        """
        Initialize the multivariate drift detector.

        Args:
            mmd_threshold: Threshold for MMD statistic
            wasserstein_threshold: Threshold for Wasserstein distance
            pca_threshold: Threshold for PCA reconstruction error (in std devs)
            n_permutations: Number of permutations for MMD significance test
            n_pca_components: Number of PCA components to use
        """
        self.mmd_threshold = mmd_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.pca_threshold = pca_threshold
        self.n_permutations = n_permutations
        self.n_pca_components = n_pca_components

        # Reference data
        self._reference_data: Optional[np.ndarray] = None
        self._reference_mean: Optional[np.ndarray] = None
        self._reference_std: Optional[np.ndarray] = None
        self._reference_correlation: Optional[np.ndarray] = None
        self._pca_components: Optional[np.ndarray] = None
        self._pca_mean: Optional[np.ndarray] = None
        self._reconstruction_error_mean: Optional[float] = None
        self._reconstruction_error_std: Optional[float] = None
        self._feature_names: List[str] = []

    def set_reference(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        """
        Set reference data for drift detection.

        Args:
            data: Reference feature matrix (n_samples, n_features)
            feature_names: Optional list of feature names
        """
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        self._reference_data = data
        self._reference_mean = np.mean(data, axis=0)
        self._reference_std = np.std(data, axis=0)
        self._reference_correlation = np.corrcoef(data.T)

        # Fit PCA on reference data
        self._fit_pca(data)

        self._feature_names = feature_names or [f"feature_{i}" for i in range(data.shape[1])]

        logger.info(
            f"Reference data set: {data.shape[0]} samples, "
            f"{data.shape[1]} features"
        )

    def _fit_pca(self, data: np.ndarray) -> None:
        """Fit PCA on reference data."""
        # Center the data
        self._pca_mean = np.mean(data, axis=0)
        centered = data - self._pca_mean

        # Compute PCA using SVD
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)

            # Select top components
            n_components = min(self.n_pca_components, data.shape[1])
            self._pca_components = Vt[:n_components].T

            # Compute reconstruction errors for reference
            projected = centered @ self._pca_components
            reconstructed = projected @ self._pca_components.T
            errors = np.sum((centered - reconstructed) ** 2, axis=1)

            self._reconstruction_error_mean = np.mean(errors)
            self._reconstruction_error_std = np.std(errors)

            variance_explained = np.sum(S[:n_components] ** 2) / np.sum(S ** 2)
            logger.info(f"PCA fitted: {n_components} components, "
                       f"{variance_explained:.1%} variance explained")

        except Exception as e:
            logger.warning(f"PCA fitting failed: {e}")
            self._pca_components = None

    def _get_severity(self, value: float, threshold: float) -> str:
        """Classify drift severity."""
        ratio = value / threshold if threshold > 0 else 0
        if ratio < self.SEVERITY_LOW:
            return "none"
        elif ratio < self.SEVERITY_MEDIUM:
            return "low"
        elif ratio < self.SEVERITY_HIGH:
            return "medium"
        else:
            return "high"

    def compute_mmd(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        kernel: str = "rbf",
        gamma: Optional[float] = None,
    ) -> Tuple[float, Optional[float]]:
        """
        Compute Maximum Mean Discrepancy between two distributions.

        MMD measures the distance between distributions in a reproducing
        kernel Hilbert space (RKHS).

        Args:
            reference: Reference samples
            current: Current samples
            kernel: Kernel type ('rbf' or 'linear')
            gamma: RBF kernel parameter (auto if None)

        Returns:
            Tuple of (mmd_statistic, p_value)
        """
        if gamma is None:
            # Use median heuristic for gamma
            combined = np.vstack([reference, current])
            pairwise_dists = cdist(combined, combined, 'euclidean')
            gamma = 1.0 / (np.median(pairwise_dists) ** 2 + 1e-8)

        n = len(reference)
        m = len(current)

        if kernel == "rbf":
            # Compute RBF kernel matrices
            K_xx = np.exp(-gamma * cdist(reference, reference, 'sqeuclidean'))
            K_yy = np.exp(-gamma * cdist(current, current, 'sqeuclidean'))
            K_xy = np.exp(-gamma * cdist(reference, current, 'sqeuclidean'))
        else:
            # Linear kernel
            K_xx = reference @ reference.T
            K_yy = current @ current.T
            K_xy = reference @ current.T

        # Unbiased MMD^2 estimate
        mmd_squared = (
            (np.sum(K_xx) - np.trace(K_xx)) / (n * (n - 1)) +
            (np.sum(K_yy) - np.trace(K_yy)) / (m * (m - 1)) -
            2 * np.sum(K_xy) / (n * m)
        )

        mmd = np.sqrt(max(mmd_squared, 0))

        # Permutation test for p-value
        p_value = self._mmd_permutation_test(reference, current, mmd, gamma)

        return mmd, p_value

    def _mmd_permutation_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        observed_mmd: float,
        gamma: float,
    ) -> float:
        """Compute p-value via permutation test."""
        combined = np.vstack([reference, current])
        n = len(reference)
        count_greater = 0

        for _ in range(self.n_permutations):
            np.random.shuffle(combined)
            perm_ref = combined[:n]
            perm_cur = combined[n:]

            perm_mmd, _ = self.compute_mmd(perm_ref, perm_cur, gamma=gamma)
            if perm_mmd >= observed_mmd:
                count_greater += 1

        return (count_greater + 1) / (self.n_permutations + 1)

    def compute_wasserstein(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Compute 1D Wasserstein distance (averaged over features).

        For multivariate data, computes the average Wasserstein distance
        across all features.

        Args:
            reference: Reference samples
            current: Current samples

        Returns:
            Average Wasserstein distance
        """
        n_features = reference.shape[1]
        distances = []

        for i in range(n_features):
            ref_1d = reference[:, i]
            cur_1d = current[:, i]

            # Normalize by standard deviation for comparability
            std = np.std(ref_1d)
            if std > 0:
                ref_1d = ref_1d / std
                cur_1d = cur_1d / std

            distance = stats.wasserstein_distance(ref_1d, cur_1d)
            distances.append(distance)

        return np.mean(distances)

    def compute_pca_reconstruction_error(
        self,
        data: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute PCA reconstruction error.

        High reconstruction error indicates the current data has different
        structure than the reference data.

        Args:
            data: Current feature matrix

        Returns:
            Tuple of (mean_error, z_score)
        """
        if self._pca_components is None:
            return 0.0, 0.0

        # Center using reference mean
        centered = data - self._pca_mean

        # Project and reconstruct
        projected = centered @ self._pca_components
        reconstructed = projected @ self._pca_components.T
        errors = np.sum((centered - reconstructed) ** 2, axis=1)

        mean_error = np.mean(errors)

        # Z-score relative to reference
        if self._reconstruction_error_std > 0:
            z_score = (mean_error - self._reconstruction_error_mean) / self._reconstruction_error_std
        else:
            z_score = 0.0

        return mean_error, z_score

    def compute_correlation_change(
        self,
        current: np.ndarray,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute change in feature correlations.

        Args:
            current: Current feature matrix

        Returns:
            Tuple of (overall_change, per_feature_change)
        """
        if self._reference_correlation is None:
            return 0.0, {}

        current_correlation = np.corrcoef(current.T)

        # Frobenius norm of difference (normalized)
        diff = current_correlation - self._reference_correlation
        overall_change = np.linalg.norm(diff, 'fro') / diff.size

        # Per-feature correlation changes
        per_feature = {}
        for i, name in enumerate(self._feature_names):
            feature_corr_change = np.mean(np.abs(diff[i, :]))
            per_feature[name] = float(feature_corr_change)

        return float(overall_change), per_feature

    def detect_drift(
        self,
        current_data: np.ndarray,
    ) -> MultivariateDriftReport:
        """
        Detect multivariate drift using multiple methods.

        Args:
            current_data: Current feature matrix

        Returns:
            MultivariateDriftReport with results from all methods
        """
        if self._reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        if current_data.ndim == 1:
            current_data = current_data.reshape(-1, 1)

        results = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. Maximum Mean Discrepancy
        mmd, mmd_p = self.compute_mmd(self._reference_data, current_data)
        mmd_drifted = mmd > self.mmd_threshold

        results.append(MultivariateDriftResult(
            method="mmd",
            statistic=float(mmd),
            p_value=float(mmd_p) if mmd_p else None,
            threshold=self.mmd_threshold,
            is_drifted=mmd_drifted,
            drift_severity=self._get_severity(mmd, self.mmd_threshold),
            feature_contributions=None,
            timestamp=timestamp,
        ))

        # 2. Wasserstein Distance
        wasserstein = self.compute_wasserstein(self._reference_data, current_data)
        wasserstein_drifted = wasserstein > self.wasserstein_threshold

        results.append(MultivariateDriftResult(
            method="wasserstein",
            statistic=float(wasserstein),
            p_value=None,
            threshold=self.wasserstein_threshold,
            is_drifted=wasserstein_drifted,
            drift_severity=self._get_severity(wasserstein, self.wasserstein_threshold),
            feature_contributions=None,
            timestamp=timestamp,
        ))

        # 3. PCA Reconstruction Error
        pca_error, pca_z = self.compute_pca_reconstruction_error(current_data)
        pca_drifted = abs(pca_z) > self.pca_threshold

        results.append(MultivariateDriftResult(
            method="pca_reconstruction",
            statistic=float(pca_z),
            p_value=None,
            threshold=self.pca_threshold,
            is_drifted=pca_drifted,
            drift_severity=self._get_severity(abs(pca_z), self.pca_threshold),
            feature_contributions=None,
            timestamp=timestamp,
        ))

        # 4. Correlation Change
        corr_change, per_feature_corr = self.compute_correlation_change(current_data)
        corr_threshold = 0.1  # 10% correlation change
        corr_drifted = corr_change > corr_threshold

        results.append(MultivariateDriftResult(
            method="correlation_change",
            statistic=float(corr_change),
            p_value=None,
            threshold=corr_threshold,
            is_drifted=corr_drifted,
            drift_severity=self._get_severity(corr_change, corr_threshold),
            feature_contributions=per_feature_corr,
            timestamp=timestamp,
        ))

        # Overall drift decision (majority vote)
        drifted_count = sum(1 for r in results if r.is_drifted)
        overall_drifted = drifted_count >= 2  # At least 2 methods agree

        return MultivariateDriftReport(
            timestamp=timestamp,
            overall_is_drifted=overall_drifted,
            results=results,
            pca_variance_explained=None,  # Could compute if needed
            correlation_change=float(corr_change),
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def create_multivariate_detector(
    reference_data: np.ndarray,
    feature_names: Optional[List[str]] = None,
) -> MultivariateDriftDetector:
    """
    Create and initialize a multivariate drift detector.

    Args:
        reference_data: Reference feature matrix
        feature_names: Optional feature names

    Returns:
        Initialized MultivariateDriftDetector
    """
    detector = MultivariateDriftDetector()
    detector.set_reference(reference_data, feature_names)
    return detector


def quick_multivariate_check(
    reference: np.ndarray,
    current: np.ndarray,
) -> Dict[str, Any]:
    """
    Quick multivariate drift check.

    Args:
        reference: Reference feature matrix
        current: Current feature matrix

    Returns:
        Dictionary with drift detection results
    """
    detector = MultivariateDriftDetector(n_permutations=50)  # Faster
    detector.set_reference(reference)
    report = detector.detect_drift(current)

    return {
        "is_drifted": report.overall_is_drifted,
        "mmd_drifted": report.results[0].is_drifted if report.results else False,
        "wasserstein_drifted": report.results[1].is_drifted if len(report.results) > 1 else False,
        "pca_drifted": report.results[2].is_drifted if len(report.results) > 2 else False,
        "correlation_change": report.correlation_change,
    }


__all__ = [
    "MultivariateDriftResult",
    "MultivariateDriftReport",
    "MultivariateDriftDetector",
    "create_multivariate_detector",
    "quick_multivariate_check",
]
