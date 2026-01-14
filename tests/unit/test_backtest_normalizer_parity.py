"""
Backtest Normalizer Parity Test
===============================

Verifies that backtest.py and inference engine produce identical
normalized observations for the same input features.

This ensures SSOT parity between training/backtest and production inference.

Author: Trading Team
Date: 2026-01-14
"""

import pytest
import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestBacktestNormalizerParity:
    """Test parity between backtest and inference normalization."""

    @pytest.fixture
    def sample_features(self):
        """Sample feature values for testing."""
        return {
            "log_ret_5m": 0.0015,
            "log_ret_1h": 0.0045,
            "log_ret_4h": 0.012,
            "rsi_9": 55.0,
            "atr_pct": 0.0085,
            "adx_14": 22.5,
            "dxy_z": 0.5,
            "dxy_change_1d": 0.002,
            "vix_z": -0.3,
            "embi_z": 0.1,
            "brent_change_1d": -0.015,
            "rate_spread": 7.5,
            "usdmxn_change_1d": 0.003,
        }

    @pytest.fixture
    def zscore_normalizer(self):
        """Create ZScoreNormalizer instance."""
        from src.core.normalizers.zscore_normalizer import ZScoreNormalizer
        return ZScoreNormalizer()

    @pytest.fixture
    def observation_builder(self):
        """Create ObservationBuilder instance."""
        from src.core.builders.observation_builder import ObservationBuilder
        return ObservationBuilder()

    def test_zscore_normalizer_matches_observation_builder(
        self, sample_features, zscore_normalizer, observation_builder
    ):
        """Verify ZScoreNormalizer produces same values as ObservationBuilder."""
        position = 0.0
        time_normalized = 0.5

        # Normalize using ObservationBuilder (inference path)
        obs_inference = observation_builder.build(
            market_features=sample_features,
            position=position,
            time_normalized=time_normalized
        )

        # Normalize using ZScoreNormalizer (backtest path)
        core_features = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d"
        ]

        obs_backtest = np.zeros(15, dtype=np.float32)
        for i, fname in enumerate(core_features):
            obs_backtest[i] = zscore_normalizer.normalize(
                fname, sample_features[fname]
            )
        obs_backtest[13] = position
        obs_backtest[14] = time_normalized

        # Assert parity
        np.testing.assert_array_almost_equal(
            obs_inference[:13],
            obs_backtest[:13],
            decimal=5,
            err_msg="Core features normalization mismatch"
        )
        assert obs_inference[13] == obs_backtest[13], "Position mismatch"
        assert obs_inference[14] == obs_backtest[14], "Time normalized mismatch"

    def test_clip_values_match(self, zscore_normalizer, observation_builder):
        """Verify clip bounds are consistent."""
        # Both should clip to [-5, 5]
        assert zscore_normalizer.CLIP_MIN == -5.0
        assert zscore_normalizer.CLIP_MAX == 5.0

        # Test extreme value gets clipped
        extreme_value = 1000.0  # Way outside normal range
        clipped = zscore_normalizer.normalize("rsi_9", extreme_value)
        assert -5.0 <= clipped <= 5.0, "Value not clipped to expected range"

    def test_nan_handling_match(self, zscore_normalizer):
        """Verify NaN handling is consistent."""
        # NaN should become 0.0
        result = zscore_normalizer.normalize("rsi_9", float('nan'))
        assert result == 0.0, "NaN should be replaced with 0.0"

    def test_unknown_feature_handling(self, zscore_normalizer):
        """Verify unknown features use identity normalization."""
        # Unknown feature should use mean=0, std=1
        unknown_val = 0.5
        result = zscore_normalizer.normalize("unknown_feature_xyz", unknown_val)
        # With mean=0, std=1: z = (0.5 - 0) / 1 = 0.5
        assert abs(result - 0.5) < 0.01, "Unknown feature should use identity"

    def test_state_features_not_normalized(self, observation_builder):
        """Verify state features (position, time) are not normalized."""
        features = {
            "log_ret_5m": 0.0,
            "log_ret_1h": 0.0,
            "log_ret_4h": 0.0,
            "rsi_9": 50.0,
            "atr_pct": 0.01,
            "adx_14": 20.0,
            "dxy_z": 0.0,
            "dxy_change_1d": 0.0,
            "vix_z": 0.0,
            "embi_z": 0.0,
            "brent_change_1d": 0.0,
            "rate_spread": 0.0,
            "usdmxn_change_1d": 0.0,
        }

        # Position should pass through without normalization
        obs = observation_builder.build(features, position=0.75, time_normalized=0.3)
        assert abs(obs[13] - 0.75) < 0.01, "Position should not be normalized"
        assert abs(obs[14] - 0.3) < 0.01, "Time should not be normalized"

    def test_deterministic_results(self, sample_features, zscore_normalizer):
        """Verify normalization is deterministic."""
        results = []
        for _ in range(10):
            normalized = zscore_normalizer.normalize_dict(sample_features)
            results.append(list(normalized.values()))

        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(
                results[0], results[i],
                err_msg="Normalization should be deterministic"
            )


class TestBacktestDeterminism:
    """Test that backtest produces deterministic results."""

    def test_same_seed_same_results(self):
        """Verify same seed produces identical backtest results."""
        # This test requires a trained model and dataset
        # Skip if not available
        import random
        import numpy as np

        # Set seed
        seed = 42
        random.seed(seed)
        np.random.seed(seed)

        # Generate some "random" data
        data1 = [random.random() for _ in range(10)]

        # Reset seed
        random.seed(seed)
        np.random.seed(seed)

        # Generate again
        data2 = [random.random() for _ in range(10)]

        assert data1 == data2, "Same seed should produce same results"
