"""
Unit Tests for Feature Drift Detector
=====================================

Tests for the FeatureDriftDetector class which detects
feature drift using Kolmogorov-Smirnov statistical test.

MLOps-4: Feature Drift Detection Testing

Author: Trading Team
Date: 2025-01-14
"""

import json
import tempfile
from pathlib import Path
import pytest
import numpy as np

# Add src to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.monitoring.drift_detector import (
    FeatureDriftDetector,
    DriftResult,
    DriftReport,
    FeatureStats,
    ReferenceStatsManager,
    create_drift_detector,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_reference_data():
    """Generate sample reference data from normal distribution."""
    np.random.seed(42)
    return {
        "feature_1": np.random.normal(0, 1, 1000),
        "feature_2": np.random.normal(5, 2, 1000),
        "feature_3": np.random.uniform(-1, 1, 1000),
    }


@pytest.fixture
def sample_current_data_no_drift():
    """Generate current data similar to reference (no drift)."""
    np.random.seed(123)
    return {
        "feature_1": np.random.normal(0, 1, 500),
        "feature_2": np.random.normal(5, 2, 500),
        "feature_3": np.random.uniform(-1, 1, 500),
    }


@pytest.fixture
def sample_current_data_with_drift():
    """Generate current data with significant drift."""
    np.random.seed(456)
    return {
        "feature_1": np.random.normal(2, 1, 500),      # Mean shifted from 0 to 2
        "feature_2": np.random.normal(5, 5, 500),      # Std changed from 2 to 5
        "feature_3": np.random.uniform(0.5, 1.5, 500), # Range shifted
    }


@pytest.fixture
def detector_with_reference(sample_reference_data):
    """Create a drift detector with reference data set."""
    detector = FeatureDriftDetector(
        p_value_threshold=0.01,
        window_size=1000,
        min_samples=100
    )
    detector.set_reference_data(sample_reference_data)
    return detector


# =============================================================================
# Test: FeatureStats
# =============================================================================

class TestFeatureStats:
    """Tests for FeatureStats dataclass."""

    def test_feature_stats_creation(self):
        """Test FeatureStats creation."""
        stats = FeatureStats(
            name="test_feature",
            mean=0.5,
            std=1.2,
            min=-3.0,
            max=4.0,
            median=0.4,
            q25=-0.5,
            q75=1.5,
            sample_size=1000,
        )

        assert stats.name == "test_feature"
        assert stats.mean == 0.5
        assert stats.std == 1.2
        assert stats.sample_size == 1000

    def test_feature_stats_to_dict(self):
        """Test FeatureStats serialization."""
        stats = FeatureStats(
            name="test_feature",
            mean=0.5,
            std=1.2,
            min=-3.0,
            max=4.0,
            median=0.4,
            q25=-0.5,
            q75=1.5,
            sample_size=1000,
        )

        d = stats.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_feature"
        assert d["mean"] == 0.5

    def test_feature_stats_from_dict(self):
        """Test FeatureStats deserialization."""
        data = {
            "name": "test_feature",
            "mean": 0.5,
            "std": 1.2,
            "min": -3.0,
            "max": 4.0,
            "median": 0.4,
            "q25": -0.5,
            "q75": 1.5,
            "sample_size": 1000,
        }

        stats = FeatureStats.from_dict(data)
        assert stats.name == "test_feature"
        assert stats.mean == 0.5


# =============================================================================
# Test: ReferenceStatsManager
# =============================================================================

class TestReferenceStatsManager:
    """Tests for ReferenceStatsManager."""

    def test_compute_stats(self):
        """Test computing statistics for a feature."""
        manager = ReferenceStatsManager()

        values = np.random.normal(0, 1, 1000)
        stats = manager.compute_stats("test_feature", values)

        assert stats.name == "test_feature"
        assert abs(stats.mean) < 0.1  # Should be close to 0
        assert abs(stats.std - 1.0) < 0.1  # Should be close to 1
        assert stats.sample_size == 1000

    def test_compute_stats_with_nans(self):
        """Test computing statistics with NaN values."""
        manager = ReferenceStatsManager()

        values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0])
        stats = manager.compute_stats("test_feature", values)

        assert stats.sample_size == 5  # NaNs removed
        assert stats.mean == 3.0

    def test_save_and_load_stats(self, sample_reference_data):
        """Test saving and loading reference stats."""
        manager = ReferenceStatsManager()
        manager.compute_all_stats(sample_reference_data)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Save
            success = manager.save_stats(temp_path)
            assert success

            # Load into new manager
            new_manager = ReferenceStatsManager(temp_path)

            assert len(new_manager.feature_stats) == 3
            assert "feature_1" in new_manager.feature_stats

            # Check values match
            assert abs(
                new_manager.feature_stats["feature_1"].mean -
                manager.feature_stats["feature_1"].mean
            ) < 0.001

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_get_reference_data(self, sample_reference_data):
        """Test retrieving raw reference data."""
        manager = ReferenceStatsManager()
        manager.compute_all_stats(sample_reference_data, store_raw=True)

        data = manager.get_reference_data("feature_1")
        assert data is not None
        assert len(data) > 0


# =============================================================================
# Test: FeatureDriftDetector Initialization
# =============================================================================

class TestDriftDetectorInitialization:
    """Tests for FeatureDriftDetector initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        detector = FeatureDriftDetector()

        assert detector.p_value_threshold == 0.01
        assert detector.window_size == 1000
        assert detector.min_samples == 100

    def test_custom_initialization(self):
        """Test custom initialization parameters."""
        detector = FeatureDriftDetector(
            p_value_threshold=0.05,
            window_size=500,
            min_samples=50
        )

        assert detector.p_value_threshold == 0.05
        assert detector.window_size == 500
        assert detector.min_samples == 50

    def test_factory_function(self):
        """Test create_drift_detector factory function."""
        detector = create_drift_detector(
            p_value_threshold=0.02,
            window_size=2000
        )

        assert isinstance(detector, FeatureDriftDetector)
        assert detector.p_value_threshold == 0.02
        assert detector.window_size == 2000


# =============================================================================
# Test: Adding Observations
# =============================================================================

class TestAddingObservations:
    """Tests for adding observations to the detector."""

    def test_add_single_observation(self, detector_with_reference):
        """Test adding a single observation."""
        detector = detector_with_reference

        detector.add_observation({
            "feature_1": 0.5,
            "feature_2": 5.2,
            "feature_3": 0.0,
        })

        assert len(detector._windows["feature_1"]) == 1
        assert len(detector._windows["feature_2"]) == 1

    def test_add_batch_observations(self, detector_with_reference):
        """Test adding a batch of observations."""
        detector = detector_with_reference

        batch = [
            {"feature_1": 0.5, "feature_2": 5.0},
            {"feature_1": 0.6, "feature_2": 5.1},
            {"feature_1": 0.4, "feature_2": 4.9},
        ]

        detector.add_batch(batch)

        assert len(detector._windows["feature_1"]) == 3
        assert len(detector._windows["feature_2"]) == 3

    def test_window_size_limit(self, detector_with_reference):
        """Test that window size is respected."""
        detector = detector_with_reference
        detector.window_size = 10  # Set small window for test

        # Add more than window size
        for i in range(20):
            detector.add_observation({"feature_1": float(i)})

        # Window should be at max size
        assert len(detector._windows["feature_1"]) == 10
        # Should contain latest values
        assert list(detector._windows["feature_1"])[-1] == 19.0

    def test_nan_values_skipped(self, detector_with_reference):
        """Test that NaN values are skipped."""
        detector = detector_with_reference

        detector.add_observation({"feature_1": 1.0})
        detector.add_observation({"feature_1": np.nan})
        detector.add_observation({"feature_1": 2.0})

        # NaN should not be added
        assert len(detector._windows["feature_1"]) == 2
        assert np.nan not in list(detector._windows["feature_1"])


# =============================================================================
# Test: Drift Detection - No Drift
# =============================================================================

class TestDriftDetectionNoDrift:
    """Tests for drift detection when no drift is present."""

    def test_no_drift_detected(self, detector_with_reference, sample_current_data_no_drift):
        """Test that no drift is detected with similar distributions."""
        detector = detector_with_reference

        # Add current data similar to reference
        for i in range(len(sample_current_data_no_drift["feature_1"])):
            detector.add_observation({
                "feature_1": sample_current_data_no_drift["feature_1"][i],
                "feature_2": sample_current_data_no_drift["feature_2"][i],
                "feature_3": sample_current_data_no_drift["feature_3"][i],
            })

        results = detector.check_drift()

        # Should have results for all features
        assert len(results) == 3

        # Most should not be drifted
        drifted_count = sum(1 for r in results if r.is_drifted)
        assert drifted_count == 0, f"Unexpected drift detected: {[r for r in results if r.is_drifted]}"

    def test_drift_result_structure(self, detector_with_reference, sample_current_data_no_drift):
        """Test DriftResult structure is correct."""
        detector = detector_with_reference

        for i in range(200):
            detector.add_observation({
                "feature_1": sample_current_data_no_drift["feature_1"][i],
            })

        result = detector.check_drift_single("feature_1")

        assert isinstance(result, DriftResult)
        assert result.feature_name == "feature_1"
        assert isinstance(result.ks_statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.is_drifted, bool)
        assert 0 <= result.ks_statistic <= 1
        assert 0 <= result.p_value <= 1


# =============================================================================
# Test: Drift Detection - With Drift
# =============================================================================

class TestDriftDetectionWithDrift:
    """Tests for drift detection when drift is present."""

    def test_drift_detected_mean_shift(self, detector_with_reference):
        """Test drift detection with mean shift."""
        detector = detector_with_reference

        # Add data with shifted mean (0 -> 3)
        np.random.seed(789)
        for _ in range(500):
            detector.add_observation({
                "feature_1": np.random.normal(3, 1),  # Shifted from mean=0 to mean=3
            })

        result = detector.check_drift_single("feature_1")

        assert result is not None
        assert result.is_drifted, f"Expected drift, got p={result.p_value}, ks={result.ks_statistic}"
        assert result.ks_statistic > 0.2  # Significant KS statistic

    def test_drift_detected_variance_change(self, detector_with_reference):
        """Test drift detection with variance change."""
        detector = detector_with_reference

        # Add data with changed variance (std 1 -> std 5)
        np.random.seed(101)
        for _ in range(500):
            detector.add_observation({
                "feature_1": np.random.normal(0, 5),  # Same mean, much larger std
            })

        result = detector.check_drift_single("feature_1")

        assert result is not None
        assert result.is_drifted

    def test_multiple_features_drifted(self, detector_with_reference, sample_current_data_with_drift):
        """Test detection when multiple features drift."""
        detector = detector_with_reference

        # Add drifted data
        for i in range(len(sample_current_data_with_drift["feature_1"])):
            detector.add_observation({
                "feature_1": sample_current_data_with_drift["feature_1"][i],
                "feature_2": sample_current_data_with_drift["feature_2"][i],
                "feature_3": sample_current_data_with_drift["feature_3"][i],
            })

        results = detector.check_drift()
        drifted = [r for r in results if r.is_drifted]

        # At least one feature should be drifted
        assert len(drifted) >= 1, f"Expected drift, results: {[r.feature_name for r in results]}"

    def test_get_drifted_features(self, detector_with_reference, sample_current_data_with_drift):
        """Test get_drifted_features method."""
        detector = detector_with_reference

        for i in range(len(sample_current_data_with_drift["feature_1"])):
            detector.add_observation({
                "feature_1": sample_current_data_with_drift["feature_1"][i],
            })

        detector.check_drift()
        drifted = detector.get_drifted_features()

        assert isinstance(drifted, list)
        # feature_1 should be drifted
        assert "feature_1" in drifted


# =============================================================================
# Test: Drift Severity
# =============================================================================

class TestDriftSeverity:
    """Tests for drift severity classification."""

    def test_severity_none(self, detector_with_reference):
        """Test 'none' severity for low KS statistic."""
        detector = detector_with_reference

        severity = detector._get_drift_severity(0.05)
        assert severity == "none"

    def test_severity_low(self, detector_with_reference):
        """Test 'low' severity."""
        detector = detector_with_reference

        severity = detector._get_drift_severity(0.15)
        assert severity == "low"

    def test_severity_medium(self, detector_with_reference):
        """Test 'medium' severity."""
        detector = detector_with_reference

        severity = detector._get_drift_severity(0.25)
        assert severity == "medium"

    def test_severity_high(self, detector_with_reference):
        """Test 'high' severity."""
        detector = detector_with_reference

        severity = detector._get_drift_severity(0.4)
        assert severity == "high"


# =============================================================================
# Test: Drift Report
# =============================================================================

class TestDriftReport:
    """Tests for DriftReport generation."""

    def test_drift_report_structure(self, detector_with_reference, sample_current_data_no_drift):
        """Test DriftReport structure."""
        detector = detector_with_reference

        for i in range(200):
            detector.add_observation({
                "feature_1": sample_current_data_no_drift["feature_1"][i],
                "feature_2": sample_current_data_no_drift["feature_2"][i],
            })

        report = detector.get_drift_report()

        assert isinstance(report, DriftReport)
        assert isinstance(report.timestamp, str)
        assert isinstance(report.features_checked, int)
        assert isinstance(report.features_drifted, int)
        assert isinstance(report.drift_results, list)
        assert isinstance(report.overall_drift_score, float)
        assert isinstance(report.alert_active, bool)

    def test_drift_report_consistency(self, detector_with_reference, sample_current_data_no_drift):
        """Test drift report values are consistent."""
        detector = detector_with_reference

        for i in range(200):
            detector.add_observation({
                "feature_1": sample_current_data_no_drift["feature_1"][i],
            })

        report = detector.get_drift_report()

        # Drifted count should match results
        drifted_in_results = sum(1 for r in report.drift_results if r.is_drifted)
        assert report.features_drifted == drifted_in_results

        # Alert active should match drifted count
        assert report.alert_active == (report.features_drifted > 0)


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_check_drift_insufficient_samples(self, detector_with_reference):
        """Test drift check with insufficient samples."""
        detector = detector_with_reference

        # Add fewer than min_samples
        for i in range(50):  # min_samples is 100
            detector.add_observation({"feature_1": float(i)})

        result = detector.check_drift_single("feature_1")
        assert result is None  # Cannot check with insufficient data

    def test_check_drift_no_reference(self):
        """Test drift check without reference data."""
        detector = FeatureDriftDetector()

        for i in range(200):
            detector.add_observation({"feature_1": float(i)})

        result = detector.check_drift_single("feature_1")
        assert result is None  # No reference to compare against

    def test_check_drift_unknown_feature(self, detector_with_reference):
        """Test drift check for unknown feature."""
        detector = detector_with_reference

        for i in range(200):
            detector.add_observation({"unknown_feature": float(i)})

        result = detector.check_drift_single("unknown_feature")
        assert result is None  # No reference for this feature

    def test_reset_windows(self, detector_with_reference):
        """Test resetting observation windows."""
        detector = detector_with_reference

        for i in range(200):
            detector.add_observation({"feature_1": float(i)})

        assert len(detector._windows["feature_1"]) == 200

        detector.reset_windows()

        assert len(detector._windows) == 0
        assert len(detector._drifted_features) == 0

    def test_status_property(self, detector_with_reference, sample_current_data_no_drift):
        """Test status property."""
        detector = detector_with_reference

        for i in range(200):
            detector.add_observation({
                "feature_1": sample_current_data_no_drift["feature_1"][i],
            })

        status = detector.status

        assert isinstance(status, dict)
        assert "reference_features" in status
        assert "monitored_features" in status
        assert "drifted_features" in status
        assert "p_value_threshold" in status


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for drift detection workflow."""

    def test_full_workflow(self):
        """Test full drift detection workflow."""
        # 1. Create reference data
        np.random.seed(42)
        reference_data = {
            "price_change": np.random.normal(0, 0.01, 1000),
            "volume": np.random.lognormal(10, 1, 1000),
            "rsi": np.random.uniform(30, 70, 1000),
        }

        # 2. Create detector and set reference
        detector = FeatureDriftDetector(
            p_value_threshold=0.01,
            window_size=500,
            min_samples=100
        )
        detector.set_reference_data(reference_data)

        # 3. Simulate normal trading (no drift)
        np.random.seed(123)
        for _ in range(200):
            detector.add_observation({
                "price_change": np.random.normal(0, 0.01),
                "volume": np.random.lognormal(10, 1),
                "rsi": np.random.uniform(30, 70),
            })

        results_normal = detector.check_drift()
        assert all(not r.is_drifted for r in results_normal), "Unexpected drift in normal phase"

        # 4. Simulate market regime change (drift)
        np.random.seed(456)
        detector.reset_windows()  # Reset for new test

        for _ in range(200):
            detector.add_observation({
                "price_change": np.random.normal(0.05, 0.03),  # Shifted up with more volatility
                "volume": np.random.lognormal(12, 2),          # Higher volume
                "rsi": np.random.uniform(60, 90),              # RSI shifted up
            })

        results_drift = detector.check_drift()
        drifted_count = sum(1 for r in results_drift if r.is_drifted)

        assert drifted_count >= 1, "Expected drift after regime change"

    def test_save_and_load_workflow(self, sample_reference_data):
        """Test saving reference stats and loading in new detector."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            # Create and save reference stats
            detector1 = FeatureDriftDetector()
            detector1.set_reference_data(sample_reference_data)
            detector1.save_reference_stats(temp_path)

            # Create new detector and load stats
            detector2 = create_drift_detector(
                reference_stats_path=temp_path,
                p_value_threshold=0.01
            )

            assert len(detector2.reference_manager.feature_names) == 3

            # Add data and check drift works
            np.random.seed(789)
            for _ in range(200):
                detector2.add_observation({
                    "feature_1": np.random.normal(0, 1),
                })

            result = detector2.check_drift_single("feature_1")
            assert result is not None

        finally:
            Path(temp_path).unlink(missing_ok=True)


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
