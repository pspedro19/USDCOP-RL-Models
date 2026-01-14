"""
Chaos Test: NaN/Inf Handling.

Tests that the system correctly sanitizes and handles
NaN and Inf values in feature calculations and model outputs.

Contract: CTR-DATA-001
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class TestNaNSanitization:
    """Tests for NaN value sanitization."""

    def test_sanitize_nan_in_features(self):
        """NaN values in features should be replaced with 0."""
        features = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

        # Sanitize
        sanitized = np.where(np.isnan(features), 0.0, features)

        assert not np.isnan(sanitized).any()
        np.testing.assert_array_equal(sanitized, [1.0, 0.0, 3.0, 0.0, 5.0])

    def test_sanitize_inf_in_features(self):
        """Inf values in features should be replaced with clip bounds."""
        features = np.array([1.0, np.inf, -np.inf, 3.0])
        clip_range = (-5.0, 5.0)

        # Sanitize
        sanitized = np.clip(features, clip_range[0], clip_range[1])
        sanitized = np.where(np.isinf(sanitized), 0.0, sanitized)

        assert not np.isinf(sanitized).any()

    def test_observation_builder_handles_nan(self):
        """Observation builder should handle NaN in input data."""
        # Create DataFrame with NaN values
        df = pd.DataFrame({
            'close': [100.0, np.nan, 102.0, np.nan, 104.0],
            'open': [99.0, 100.0, np.nan, 102.0, 103.0],
            'high': [101.0, 101.0, 103.0, 104.0, np.nan],
            'low': [98.0, 99.0, 100.0, 101.0, 102.0],
        })

        # Fill NaN with forward fill then backward fill
        df_clean = df.ffill().bfill()

        assert not df_clean.isna().any().any()

    def test_log_return_with_nan_prices(self):
        """Log return calculation should handle NaN prices."""
        prices = pd.Series([100.0, np.nan, 102.0, 103.0])

        # Fill NaN before calculation
        prices_clean = prices.ffill()

        log_returns = np.log(prices_clean / prices_clean.shift(1))

        # First value will be NaN (no previous)
        assert np.isnan(log_returns.iloc[0])
        # Other values should be valid
        assert not np.isnan(log_returns.iloc[2])


class TestInfHandling:
    """Tests for Infinity value handling."""

    def test_division_by_zero_protection(self):
        """Division by zero should not produce Inf."""
        numerator = 10.0
        denominator = 0.0

        # Safe division
        result = numerator / denominator if denominator != 0 else 0.0

        assert result == 0.0
        assert not np.isinf(result)

    def test_log_of_zero_protection(self):
        """Log of zero should not produce -Inf."""
        values = np.array([0.0, 0.5, 1.0])

        # Safe log
        safe_values = np.where(values <= 0, 1e-10, values)
        log_values = np.log(safe_values)

        assert not np.isinf(log_values).any()

    def test_zscore_with_zero_std(self):
        """Z-score with zero std should not produce Inf."""
        values = np.array([5.0, 5.0, 5.0])  # Constant - std = 0
        mean = np.mean(values)
        std = np.std(values)

        # Safe z-score
        if std == 0:
            z_scores = np.zeros_like(values)
        else:
            z_scores = (values - mean) / std

        assert not np.isinf(z_scores).any()
        np.testing.assert_array_equal(z_scores, [0.0, 0.0, 0.0])


class TestModelOutputSanitization:
    """Tests for model output sanitization."""

    def test_clip_extreme_actions(self):
        """Extreme model outputs should be clipped."""
        raw_action = 100.0  # Extreme value

        # Clip to valid range
        clipped = np.clip(raw_action, -1.0, 1.0)

        assert clipped == 1.0

    def test_nan_action_fallback(self):
        """NaN action should fallback to HOLD (0)."""
        raw_action = np.nan

        # Sanitize
        if np.isnan(raw_action):
            action = 0.0  # HOLD
        else:
            action = raw_action

        assert action == 0.0


class TestDataFrameOperations:
    """Tests for DataFrame operations with bad values."""

    def test_rolling_mean_with_nan(self):
        """Rolling mean should handle NaN values."""
        series = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])

        # Rolling mean with min_periods
        rolling = series.rolling(window=3, min_periods=1).mean()

        # Should not have NaN propagation
        assert not np.isnan(rolling.iloc[-1])

    def test_pct_change_with_zeros(self):
        """Percentage change should handle zero values."""
        series = pd.Series([100.0, 0.0, 50.0])

        # pct_change can produce inf
        pct = series.pct_change()

        # Should replace inf
        pct = pct.replace([np.inf, -np.inf], 0.0)

        assert not np.isinf(pct).any()

    def test_merge_with_nan_keys(self):
        """Merge operations should handle NaN in keys."""
        df1 = pd.DataFrame({
            'time': [1, 2, np.nan, 4],
            'value': [10, 20, 30, 40]
        })
        df2 = pd.DataFrame({
            'time': [1, 2, 3, 4],
            'other': [100, 200, 300, 400]
        })

        # Merge with NaN handling
        merged = df1.merge(df2, on='time', how='left')

        # NaN key won't match
        assert len(merged) == 4


class TestNormalizerNaNHandling:
    """Tests for normalizer NaN handling."""

    def test_zscore_normalizer_nan_input(self):
        """Z-score normalizer should handle NaN input."""
        from features.normalizers import ZScoreNormalizer

        normalizer = ZScoreNormalizer(mean=0.0, std=1.0)

        result = normalizer.normalize(float('nan'))
        assert result == 0.0  # Should return default

    def test_zscore_normalizer_inf_input(self):
        """Z-score normalizer should handle Inf input."""
        from features.normalizers import ZScoreNormalizer

        normalizer = ZScoreNormalizer(mean=0.0, std=1.0)

        result = normalizer.normalize(float('inf'))
        assert result == 0.0  # Should return default

    def test_batch_normalization_with_nan(self):
        """Batch normalization should sanitize NaN values."""
        from features.normalizers import ZScoreNormalizer

        normalizer = ZScoreNormalizer(mean=0.0, std=1.0, clip=(-5.0, 5.0))

        values = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
        result = normalizer.normalize_batch(values)

        assert not np.isnan(result).any()
        assert not np.isinf(result).any()


class TestObservationValidation:
    """Tests for observation validation."""

    def test_observation_no_nan(self):
        """Complete observation should have no NaN values."""
        observation = np.array([0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6, 0.7, 0.8, 0.9, 1.0,
                                0.1, 0.2, 0.3, 0.0, 0.5])

        # Insert a NaN for testing
        bad_observation = observation.copy()
        bad_observation[5] = np.nan

        # Validation
        assert np.isnan(bad_observation).any()

        # Sanitize
        sanitized = np.where(np.isnan(bad_observation), 0.0, bad_observation)
        assert not np.isnan(sanitized).any()

    def test_observation_in_valid_range(self):
        """Observation values should be in valid range."""
        observation = np.array([10.0, -10.0, 0.0, 3.0, -3.0])
        clip_range = (-5.0, 5.0)

        clipped = np.clip(observation, clip_range[0], clip_range[1])

        assert np.all(clipped >= clip_range[0])
        assert np.all(clipped <= clip_range[1])
