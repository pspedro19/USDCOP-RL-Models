"""Tests for confidence_scorer — dynamic sizing based on model agreement + magnitude."""

import pytest
from src.forecasting.confidence_scorer import (
    ConfidenceConfig,
    ConfidenceScore,
    ConfidenceTier,
    score_confidence,
)


@pytest.fixture
def config():
    return ConfidenceConfig()


class TestConfidenceTiers:
    def test_high_confidence_short(self, config):
        """Both models agree tightly + large magnitude → HIGH."""
        score = score_confidence(
            ridge_pred=-0.015,
            br_pred=-0.0155,    # agreement = 0.0005 < 0.001
            direction=-1,
            config=config,
        )
        assert score.tier == ConfidenceTier.HIGH
        assert score.agreement < config.agreement_tight
        assert score.magnitude > config.magnitude_high
        assert score.sizing_multiplier == 2.0
        assert score.skip_trade is False

    def test_medium_confidence_short(self, config):
        """Moderate agreement + moderate magnitude → MEDIUM."""
        score = score_confidence(
            ridge_pred=-0.007,
            br_pred=-0.004,     # agreement = 0.003 < 0.005
            direction=-1,
            config=config,
        )
        assert score.tier == ConfidenceTier.MEDIUM
        assert score.sizing_multiplier == 1.5

    def test_low_confidence_short(self, config):
        """Large divergence → LOW."""
        score = score_confidence(
            ridge_pred=-0.002,
            br_pred=0.008,      # agreement = 0.01 > 0.005
            direction=-1,
            config=config,
        )
        assert score.tier == ConfidenceTier.LOW
        assert score.sizing_multiplier == 1.0
        assert score.skip_trade is False

    def test_high_confidence_long(self, config):
        """HIGH confidence LONG gets 1.0x (conservative)."""
        score = score_confidence(
            ridge_pred=0.015,
            br_pred=0.0155,
            direction=1,
            config=config,
        )
        assert score.tier == ConfidenceTier.HIGH
        assert score.sizing_multiplier == 1.0  # Capped at 1.0x for LONGs

    def test_medium_confidence_long(self, config):
        """MEDIUM confidence LONG gets 0.5x (exploratory)."""
        score = score_confidence(
            ridge_pred=0.007,
            br_pred=0.004,
            direction=1,
            config=config,
        )
        assert score.tier == ConfidenceTier.MEDIUM
        assert score.sizing_multiplier == 0.5

    def test_low_confidence_long_skipped(self, config):
        """LOW confidence LONG → SKIP (0.0x)."""
        score = score_confidence(
            ridge_pred=0.002,
            br_pred=-0.003,     # Models disagree on direction
            direction=1,
            config=config,
        )
        assert score.tier == ConfidenceTier.LOW
        assert score.sizing_multiplier == 0.0
        assert score.skip_trade is True


class TestEdgeCases:
    def test_above_tight_threshold(self, config):
        """Agreement above tight threshold → MEDIUM (not HIGH)."""
        score = score_confidence(
            ridge_pred=-0.015,
            br_pred=-0.0135,    # agreement = 0.0015 > 0.001
            direction=-1,
            config=config,
        )
        # magnitude = 0.01425 > 0.01, agreement = 0.0015 > 0.001 → MEDIUM
        assert score.tier == ConfidenceTier.MEDIUM

    def test_zero_predictions(self, config):
        """Both models predict zero → LOW."""
        score = score_confidence(
            ridge_pred=0.0,
            br_pred=0.0,
            direction=-1,
            config=config,
        )
        assert score.tier == ConfidenceTier.LOW

    def test_large_divergence_high_magnitude(self, config):
        """Models diverge widely but magnitude is high → still LOW."""
        score = score_confidence(
            ridge_pred=-0.020,
            br_pred=0.010,      # agreement = 0.03, magnitude = 0.005
            direction=-1,
            config=config,
        )
        assert score.tier == ConfidenceTier.LOW

    def test_agreement_values(self, config):
        """Verify agreement and magnitude are computed correctly."""
        score = score_confidence(
            ridge_pred=-0.010,
            br_pred=-0.006,
            direction=-1,
            config=config,
        )
        assert abs(score.agreement - 0.004) < 1e-10
        assert abs(score.magnitude - 0.008) < 1e-10


class TestCustomConfig:
    def test_aggressive_long_config(self):
        """Custom config can make LONGs more aggressive."""
        config = ConfidenceConfig(
            long_high=1.5,
            long_medium=1.0,
            long_low=0.5,
        )
        score = score_confidence(
            ridge_pred=0.015,
            br_pred=0.0155,
            direction=1,
            config=config,
        )
        assert score.sizing_multiplier == 1.5

    def test_short_only_config(self):
        """Config that skips all LONGs."""
        config = ConfidenceConfig(
            long_high=0.0,
            long_medium=0.0,
            long_low=0.0,
        )
        score = score_confidence(
            ridge_pred=0.020,
            br_pred=0.021,
            direction=1,
            config=config,
        )
        assert score.skip_trade is True
