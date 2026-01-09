"""
Unit Tests for Feature Builder / Feature Calculator
=====================================================

Tests feature calculation logic ensuring:
- Correct formulas
- Proper dimensionality
- Normalization ranges
- Time normalization boundary (0.983, NOT 1.0)

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-16
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.mark.unit
class TestFeatureCalculation:
    """Test individual feature calculation methods"""

    def test_rsi_calculation(self, feature_calculator, sample_ohlcv_df):
        """Test RSI calculation produces values in [0, 100]"""
        rsi = feature_calculator.calc_rsi(sample_ohlcv_df['close'], period=9)

        # RSI should be between 0 and 100
        assert rsi.dropna().min() >= 0, "RSI minimum below 0"
        assert rsi.dropna().max() <= 100, "RSI maximum above 100"

        # Should have NaN for first period-1 values
        assert rsi.isna().sum() >= 8, "RSI should have NaN for warmup period"

    def test_atr_calculation(self, feature_calculator, sample_ohlcv_df):
        """Test ATR calculation produces positive values"""
        atr = feature_calculator.calc_atr(
            sample_ohlcv_df['high'],
            sample_ohlcv_df['low'],
            sample_ohlcv_df['close'],
            period=10
        )

        # ATR should be positive
        assert (atr.dropna() >= 0).all(), "ATR contains negative values"

        # Should have NaN for first period values
        assert atr.isna().sum() >= 9, "ATR should have NaN for warmup period"

    def test_atr_pct_calculation(self, feature_calculator, sample_ohlcv_df):
        """Test ATR% calculation is ATR/close * 100"""
        atr_pct = feature_calculator.calc_atr_pct(
            sample_ohlcv_df['high'],
            sample_ohlcv_df['low'],
            sample_ohlcv_df['close'],
            period=10
        )

        # ATR% should be positive and typically < 10%
        assert (atr_pct.dropna() >= 0).all(), "ATR% contains negative values"
        assert atr_pct.dropna().max() < 20, "ATR% suspiciously high"

    def test_adx_calculation(self, feature_calculator, sample_ohlcv_df):
        """Test ADX calculation produces values in [0, 100]"""
        adx = feature_calculator.calc_adx(
            sample_ohlcv_df['high'],
            sample_ohlcv_df['low'],
            sample_ohlcv_df['close'],
            period=14
        )

        # ADX should be between 0 and 100
        assert adx.dropna().min() >= 0, "ADX minimum below 0"
        assert adx.dropna().max() <= 100, "ADX maximum above 100"

        # Should have NaN for warmup period
        assert adx.isna().sum() >= 13, "ADX should have NaN for warmup period"

    def test_log_return_calculation(self, feature_calculator):
        """Test log return calculation"""
        close = pd.Series([100, 101, 99, 102, 100])
        log_ret_1 = feature_calculator.calc_log_return(close, periods=1)

        # Verify formula: log(close[t] / close[t-1])
        expected = np.log(101 / 100)
        assert abs(log_ret_1.iloc[1] - expected) < 1e-10, "Log return formula incorrect"

        # First value should be NaN
        assert pd.isna(log_ret_1.iloc[0]), "First log return should be NaN"

    def test_pct_change_clipping(self, feature_calculator):
        """Test percentage change with clipping"""
        series = pd.Series([100, 120, 80, 150])  # Large changes

        # Test clipping to [-0.1, 0.1]
        pct = feature_calculator.calc_pct_change(series, periods=1, clip_range=(-0.1, 0.1))

        # All values should be within clip range
        assert (pct.dropna() >= -0.1).all(), "Values below clip minimum"
        assert (pct.dropna() <= 0.1).all(), "Values above clip maximum"

        # Large positive change should be clipped to 0.1
        assert pct.iloc[1] == 0.1, "Positive clip not applied correctly"

        # Large negative change should be clipped to -0.1
        assert pct.iloc[2] == -0.1, "Negative clip not applied correctly"


@pytest.mark.unit
class TestObservationSpace:
    """Test observation space construction"""

    def test_observation_dimension(self, feature_calculator):
        """Observation must be exactly 15 dimensions"""
        # Create sample features
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        assert obs.shape == (15,), f"Expected 15 dimensions, got {obs.shape[0]}"

    def test_time_normalized_range(self, feature_calculator):
        """time_normalized should be in [0, 0.983], NOT [0, 1]"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        # Test various steps
        for step_count in [0, 29, 59]:
            obs = feature_calculator.build_observation(
                features=test_features,
                position=0.0,
                step_count=step_count,
                episode_length=60
            )

            time_normalized = obs[-1]  # Last element is time_normalized

            # Verify formula: step_count / episode_length
            expected = step_count / 60
            assert abs(time_normalized - expected) < 1e-6, \
                f"time_normalized mismatch at step {step_count}"

            # Verify range
            assert 0 <= time_normalized <= 0.983, \
                f"time_normalized out of range: {time_normalized}"

        # Bar 60 (step 59) should be 0.983, NOT 1.0
        obs_final = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=59,
            episode_length=60
        )
        assert abs(obs_final[-1] - 0.983) < 0.001, \
            f"Final time_normalized should be ~0.983, got {obs_final[-1]}"

    def test_position_in_observation(self, feature_calculator):
        """Test position is correctly included in observation"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        for pos in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            obs = feature_calculator.build_observation(
                features=test_features,
                position=pos,
                step_count=30,
                episode_length=60
            )

            # Position is second-to-last element
            assert abs(obs[-2] - pos) < 1e-6, f"Position mismatch: expected {pos}, got {obs[-2]}"

    def test_observation_clipping(self, feature_calculator):
        """Test observation is clipped to [-5, 5]"""
        # Create features with extreme values
        test_features = pd.Series({f: 100.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # All values should be clipped to [-5, 5]
        assert (obs >= -5).all(), "Observation contains values below -5"
        assert (obs <= 5).all(), "Observation contains values above 5"
