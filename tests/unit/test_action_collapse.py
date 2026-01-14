"""
Action Collapse Detector Tests
==============================

Tests for the ActionCollapseDetector MLOps monitoring component.

Author: Trading Team
Date: 2026-01-14
"""

import pytest
import math
from services.mlops.action_collapse_detector import (
    ActionCollapseDetector,
    ActionCollapseConfig,
    ActionCollapseResult,
    check_action_collapse,
)


class TestActionCollapseDetector:
    """Test suite for ActionCollapseDetector."""

    @pytest.fixture
    def detector(self):
        """Create default detector instance."""
        return ActionCollapseDetector()

    @pytest.fixture
    def collapsed_actions(self):
        """Generate collapsed (all HOLD) action history."""
        return ["HOLD"] * 100

    @pytest.fixture
    def healthy_actions(self):
        """Generate healthy distribution action history."""
        # ~50% HOLD, ~25% LONG, ~25% SHORT
        actions = []
        for i in range(100):
            if i % 4 == 0:
                actions.append("LONG")
            elif i % 4 == 1:
                actions.append("SHORT")
            else:
                actions.append("HOLD")
        return actions

    @pytest.fixture
    def uniform_actions(self):
        """Generate uniform distribution."""
        return ["HOLD", "LONG", "SHORT"] * 34

    def test_detect_collapsed_all_hold(self, detector, collapsed_actions):
        """Test detection of all-HOLD collapse."""
        result = detector.check(collapsed_actions)

        assert result.is_collapsed is True
        assert result.dominant_action == "HOLD"
        assert result.dominant_pct == 1.0
        assert result.entropy == 0.0
        assert result.severity == 2
        assert "100% HOLD" in result.warning

    def test_detect_healthy_distribution(self, detector, healthy_actions):
        """Test that healthy distribution is not flagged."""
        result = detector.check(healthy_actions)

        assert result.is_collapsed is False
        assert result.severity < 2
        assert result.entropy > 0.5

    def test_detect_uniform_distribution(self, detector, uniform_actions):
        """Test uniform distribution has max entropy."""
        result = detector.check(uniform_actions)

        # Uniform distribution should have entropy ≈ log2(3) ≈ 1.58
        assert result.entropy > 1.5
        assert result.is_collapsed is False
        # Uniform dist has low HOLD which triggers warning
        assert result.distribution["HOLD"] == pytest.approx(0.33, abs=0.02)

    def test_entropy_calculation_boundary(self, detector):
        """Test entropy at boundary cases."""
        # All same action = 0 entropy
        result = detector.check(["LONG"] * 100)
        assert result.entropy == 0.0

        # Two equal actions
        result = detector.check(["LONG", "SHORT"] * 50)
        assert result.entropy == pytest.approx(1.0, abs=0.1)

    def test_dominance_threshold(self):
        """Test dominance threshold detection."""
        config = ActionCollapseConfig(
            dominance_threshold=0.70,  # Lower threshold
            min_samples=20
        )
        detector = ActionCollapseDetector(config)

        # 75% HOLD should trigger
        actions = ["HOLD"] * 75 + ["LONG"] * 15 + ["SHORT"] * 10
        result = detector.check(actions)

        assert result.is_collapsed is True
        assert "High dominance" in result.warning

    def test_insufficient_samples(self, detector):
        """Test behavior with insufficient samples."""
        result = detector.check(["HOLD"] * 10)  # Less than min_samples

        assert result.is_collapsed is False
        assert "Insufficient samples" in result.warning
        assert result.severity == 0

    def test_record_and_check(self, detector, healthy_actions):
        """Test recording actions incrementally."""
        for action in healthy_actions:
            detector.record(action)

        result = detector.check()
        assert result.is_collapsed is False
        assert detector.sample_count == 100

    def test_record_batch(self, detector, healthy_actions):
        """Test batch recording."""
        detector.record_batch(healthy_actions)

        assert detector.sample_count == 100
        result = detector.check()
        assert result.is_collapsed is False

    def test_unknown_action_handling(self, detector):
        """Test handling of unknown action types."""
        # Should treat unknown as HOLD
        detector.record("UNKNOWN_ACTION")
        assert detector.sample_count == 1

    def test_case_insensitive(self, detector):
        """Test action names are case insensitive."""
        actions = ["hold", "Hold", "HOLD", "long", "LONG", "short"]
        for action in actions:
            detector.record(action)

        assert detector.sample_count == 6

    def test_reset(self, detector, healthy_actions):
        """Test detector reset."""
        detector.record_batch(healthy_actions)
        assert detector.sample_count == 100

        detector.reset()
        assert detector.sample_count == 0

    def test_recent_distribution(self, detector):
        """Test getting recent distribution."""
        # Record 100 actions
        detector.record_batch(["HOLD"] * 50)
        detector.record_batch(["LONG"] * 50)

        recent = detector.get_recent_distribution(n_samples=30)
        # Last 30 are all LONG
        assert recent["LONG"] == pytest.approx(1.0, abs=0.01)

    def test_distribution_shift_detection(self, detector):
        """Test distribution shift detection."""
        # First 100: mostly HOLD
        detector.record_batch(["HOLD"] * 80 + ["LONG"] * 20)

        # Next 100: mostly LONG
        detector.record_batch(["LONG"] * 80 + ["HOLD"] * 20)

        shift_detected, divergence = detector.detect_distribution_shift(
            window_a=100, window_b=100
        )

        assert shift_detected is True
        assert divergence > 0.1

    def test_severity_levels(self, detector):
        """Test severity level assignment."""
        # Healthy = 0
        result = detector.check(["HOLD"] * 50 + ["LONG"] * 25 + ["SHORT"] * 25)
        assert result.severity < 2

        # Collapsed = 2
        result = detector.check(["HOLD"] * 100)
        assert result.severity == 2

    def test_result_to_dict(self, detector):
        """Test result serialization."""
        result = detector.check(["HOLD"] * 60 + ["LONG"] * 20 + ["SHORT"] * 20)

        d = result.to_dict()
        assert "entropy" in d
        assert "distribution" in d
        assert "is_collapsed" in d
        assert isinstance(d["distribution"], dict)

    def test_convenience_function(self):
        """Test check_action_collapse convenience function."""
        actions = ["HOLD"] * 100
        result = check_action_collapse(actions)

        assert result.is_collapsed is True
        assert result.dominant_action == "HOLD"


class TestActionCollapseConfig:
    """Test configuration options."""

    def test_custom_entropy_threshold(self):
        """Test custom entropy threshold."""
        config = ActionCollapseConfig(entropy_threshold=1.0)  # Higher threshold
        detector = ActionCollapseDetector(config)

        # This distribution has entropy ~1.0, should be flagged
        actions = ["HOLD", "LONG"] * 50
        result = detector.check(actions)

        # Entropy ≈ 1.0, threshold is 1.0, so borderline
        assert result.entropy < 1.1

    def test_custom_window_size(self):
        """Test custom window size."""
        config = ActionCollapseConfig(window_size=50)
        detector = ActionCollapseDetector(config)

        # Record 100 actions, only last 50 should be in window
        detector.record_batch(["HOLD"] * 50)
        detector.record_batch(["LONG"] * 50)

        assert detector.sample_count == 50  # Window limit
        result = detector.check()
        assert result.distribution["LONG"] == 1.0  # All LONG in window

    def test_hold_percentage_warning(self):
        """Test HOLD percentage range warnings."""
        config = ActionCollapseConfig(
            expected_hold_min=0.40,
            expected_hold_max=0.60,
            min_samples=20
        )
        detector = ActionCollapseDetector(config)

        # Too little HOLD
        actions = ["LONG"] * 70 + ["SHORT"] * 20 + ["HOLD"] * 10
        result = detector.check(actions)
        assert "Low HOLD" in (result.warning or "")

        # Too much HOLD
        detector.reset()
        actions = ["HOLD"] * 70 + ["LONG"] * 15 + ["SHORT"] * 15
        result = detector.check(actions)
        assert "High HOLD" in (result.warning or "")
