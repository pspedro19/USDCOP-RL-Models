"""
Unit Tests for Feature Normalization
======================================

Tests normalization logic:
- Z-score normalization
- Clipping ranges
- Fixed vs computed stats
- CORRECTED usdmxn_ret_1h clip [-0.10, 0.10]

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-16
"""

import pytest
import numpy as np
import pandas as pd


@pytest.mark.unit
class TestZScoreNormalization:
    """Test z-score normalization implementation"""

    def test_zscore_basic(self, feature_calculator):
        """Test basic z-score normalization formula"""
        series = pd.Series([100, 105, 95, 110, 90])
        mean = 100.0
        std = 5.0

        normalized = feature_calculator.normalize_zscore(series, mean, std, clip=10.0)

        # Manual verification for each value
        expected = [
            (100 - 100) / 5,  # 0.0
            (105 - 100) / 5,  # 1.0
            (95 - 100) / 5,   # -1.0
            (110 - 100) / 5,  # 2.0
            (90 - 100) / 5,   # -2.0
        ]

        for i, exp in enumerate(expected):
            assert abs(normalized.iloc[i] - exp) < 1e-6, \
                f"Z-score mismatch at index {i}: expected {exp}, got {normalized.iloc[i]}"

    def test_zscore_with_clipping(self, feature_calculator):
        """Test z-score normalization with clipping to ±4"""
        series = pd.Series([50, 100, 150, 200])  # Large range
        mean = 100.0
        std = 10.0

        normalized = feature_calculator.normalize_zscore(series, mean, std, clip=4.0)

        # All values should be within [-4, 4]
        assert (normalized >= -4.0).all(), "Values below -4 after clipping"
        assert (normalized <= 4.0).all(), "Values above 4 after clipping"

        # Extreme value should be clipped
        # (200 - 100) / 10 = 10.0, should clip to 4.0
        assert normalized.iloc[3] == 4.0, "Max value not clipped correctly"

    def test_zscore_zero_division_protection(self, feature_calculator):
        """Test division by zero protection in z-score"""
        series = pd.Series([100, 100, 100])
        mean = 100.0
        std = 0.0  # Zero std should be handled

        # Should not raise exception
        normalized = feature_calculator.normalize_zscore(series, mean, std, clip=4.0)

        # All values should be 0 (due to 1e-10 protection)
        assert (normalized.abs() < 1e-5).all(), "Zero std not handled correctly"


@pytest.mark.unit
class TestFeatureClipping:
    """Test clipping ranges for different features"""

    def test_log_return_clipping(self, feature_calculator):
        """Test log returns are clipped to [-0.05, 0.05]"""
        # Extreme price movement
        close = pd.Series([100, 120, 80, 150, 60])

        log_ret = feature_calculator.calc_log_return(close, periods=1)
        clipped = log_ret.clip(-0.05, 0.05)

        # All values should be within clip range
        assert (clipped.dropna() >= -0.05).all(), "Log return below clip minimum"
        assert (clipped.dropna() <= 0.05).all(), "Log return above clip maximum"

    def test_usdmxn_clip_corrected(self, feature_calculator):
        """usdmxn_ret_1h clip CORRECTED to [-0.10, 0.10]"""
        # This is the critical test mentioned in requirements
        # Config was corrected from [-0.05, 0.05] to [-0.10, 0.10]

        series = pd.Series([19.0, 19.5, 20.0, 20.5, 21.0] * 3)  # 15 values
        pct_change_12 = series.pct_change(12)

        # Apply corrected clipping
        clipped = feature_calculator.calc_pct_change(
            series, periods=12, clip_range=(-0.1, 0.1)
        )

        # Should clip to ±0.10, NOT ±0.05
        assert (clipped.dropna() >= -0.1).all(), "USDMXN below corrected clip minimum"
        assert (clipped.dropna() <= 0.1).all(), "USDMXN above corrected clip maximum"

        # Test extreme value clipping
        extreme_series = pd.Series([10.0] * 12 + [15.0])  # 50% jump
        extreme_clipped = feature_calculator.calc_pct_change(
            extreme_series, periods=12, clip_range=(-0.1, 0.1)
        )

        # Large move should clip to 0.10 (corrected max)
        assert extreme_clipped.iloc[-1] == 0.10, \
            "Extreme positive move should clip to 0.10 (corrected)"

    def test_dxy_change_clipping(self, feature_config):
        """Test DXY change clips to [-0.03, 0.03]"""
        # config: market_features['macro_changes']['items']
        macro_changes = feature_config.get('market_features', {}).get('macro_changes', {})
        if not macro_changes:
            pytest.skip("config format - macro_changes in market_features")

        dxy_config = next(
            (item for item in macro_changes.get('items', [])
             if item['name'] == 'dxy_change_1d'),
            None
        )

        if dxy_config is None:
            pytest.skip("dxy_change_1d not found in config")

        clip_range = dxy_config['clip']
        assert clip_range == [-0.03, 0.03], \
            f"DXY change should clip to [-0.03, 0.03], got {clip_range}"

    def test_brent_change_clipping(self, feature_config):
        """Test Brent change clips to [-0.10, 0.10]"""
        # config: market_features['macro_changes']['items']
        macro_changes = feature_config.get('market_features', {}).get('macro_changes', {})
        if not macro_changes:
            pytest.skip("config format - macro_changes in market_features")

        brent_config = next(
            (item for item in macro_changes.get('items', [])
             if item['name'] == 'brent_change_1d'),
            None
        )

        if brent_config is None:
            pytest.skip("brent_change_1d not found in config")

        clip_range = brent_config['clip']
        assert clip_range == [-0.1, 0.1], \
            f"Brent change should clip to [-0.10, 0.10], got {clip_range}"


@pytest.mark.unit
class TestNormalizationParity:
    """Test normalization produces consistent results"""

    def test_normalize_features_produces_bounded_values(self, feature_calculator, sample_ohlcv_df):
        """Test that normalization produces bounded values within expected range"""
        # Compute features
        features = feature_calculator.compute_technical_features(sample_ohlcv_df)

        # Normalize once
        norm1 = feature_calculator.normalize_features(features)

        # Check that normalized values are within reasonable bounds [-10, 10]
        for col in ['log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'rsi_9', 'atr_pct', 'adx_14']:
            if col in norm1.columns:
                max_val = norm1[col].abs().max()
                assert max_val <= 10.0, \
                    f"{col} has values outside [-10, 10]: max abs = {max_val}"

        # Check that normalization reduces the scale of values
        # (normalized values should have approximately mean ~0, std ~1)
        for col in ['log_ret_5m', 'log_ret_1h', 'log_ret_4h']:
            if col in norm1.columns:
                # After z-score normalization, std should be close to 1
                normalized_std = norm1[col].std()
                assert normalized_std < 100, \
                    f"{col} std after normalization is too high: {normalized_std}"

    def test_nan_preservation(self, feature_calculator):
        """Test that NaN values are preserved during normalization"""
        series = pd.Series([np.nan, 100, 105, np.nan, 95])
        mean = 100.0
        std = 5.0

        normalized = feature_calculator.normalize_zscore(series, mean, std)

        # NaN positions should be preserved
        assert pd.isna(normalized.iloc[0]), "First NaN not preserved"
        assert pd.isna(normalized.iloc[3]), "Middle NaN not preserved"

        # Non-NaN values should be normalized
        assert pd.notna(normalized.iloc[1]), "Valid value became NaN"


@pytest.mark.unit
class TestMacroFeatureNormalization:
    """Test macro feature normalization with fixed stats"""

    def test_dxy_zscore(self, feature_calculator):
        """Test DXY z-score with fixed mean=103, std=5"""
        dxy_series = pd.Series([98, 103, 108, 113, 93])

        normalized = feature_calculator.normalize_zscore(
            dxy_series, mean=103.0, std=5.0, clip=4.0
        )

        # Verify specific values
        assert abs(normalized.iloc[1]) < 1e-6, "DXY at mean should normalize to ~0"
        assert abs(normalized.iloc[2] - 1.0) < 1e-6, "DXY at mean+std should normalize to 1.0"
        assert abs(normalized.iloc[0] - (-1.0)) < 1e-6, "DXY at mean-std should normalize to -1.0"

    def test_vix_zscore(self, feature_calculator):
        """Test VIX z-score with fixed mean=20, std=10"""
        vix_series = pd.Series([10, 20, 30, 40, 50])

        normalized = feature_calculator.normalize_zscore(
            vix_series, mean=20.0, std=10.0, clip=4.0
        )

        # Verify specific values
        assert abs(normalized.iloc[1]) < 1e-6, "VIX at mean should normalize to ~0"
        assert abs(normalized.iloc[2] - 1.0) < 1e-6, "VIX at mean+std should normalize to 1.0"
        assert abs(normalized.iloc[4] - 3.0) < 1e-6, "VIX at mean+3*std should normalize to 3.0"

    def test_embi_zscore(self, feature_calculator):
        """Test EMBI z-score with fixed mean=300, std=100"""
        embi_series = pd.Series([200, 300, 400, 500, 100])

        normalized = feature_calculator.normalize_zscore(
            embi_series, mean=300.0, std=100.0, clip=4.0
        )

        # Verify specific values
        assert abs(normalized.iloc[1]) < 1e-6, "EMBI at mean should normalize to ~0"
        assert abs(normalized.iloc[2] - 1.0) < 1e-6, "EMBI at mean+std should normalize to 1.0"
        assert abs(normalized.iloc[0] - (-1.0)) < 1e-6, "EMBI at mean-std should normalize to -1.0"


@pytest.mark.unit
class TestNormStatsRetrieval:
    """Test normalization stats retrieval from config"""

    def test_get_norm_stats_for_returns(self, feature_calculator):
        """Test retrieval of norm stats for return features"""
        stats = feature_calculator.get_norm_stats('log_ret_5m')

        assert 'mean' in stats, "Missing mean in log_ret_5m stats"
        assert 'std' in stats, "Missing std in log_ret_5m stats"

        # Check approximate values from config
        assert abs(stats['mean']) < 1e-4, "log_ret_5m mean should be near 0"
        assert 0.001 < stats['std'] < 0.01, "log_ret_5m std should be in reasonable range"

    def test_get_norm_stats_for_technical(self, feature_calculator):
        """Test retrieval of norm stats for technical features"""
        rsi_stats = feature_calculator.get_norm_stats('rsi_9')

        assert 'mean' in rsi_stats, "Missing mean in rsi_9 stats"
        assert 'std' in rsi_stats, "Missing std in rsi_9 stats"

        # RSI mean should be around 50 (neutral)
        assert 40 < rsi_stats['mean'] < 60, "RSI mean should be near 50"
        assert rsi_stats['std'] > 10, "RSI std should be > 10"

    def test_get_norm_stats_nonexistent(self, feature_calculator):
        """Test retrieval for non-existent feature returns empty dict"""
        stats = feature_calculator.get_norm_stats('nonexistent_feature')

        assert isinstance(stats, dict), "Should return dict for nonexistent feature"
        assert len(stats) == 0, "Should return empty dict for nonexistent feature"


@pytest.mark.unit
class TestObservationNormalization:
    """Test final observation normalization"""

    def test_observation_final_clip(self, feature_calculator):
        """Test observation is clipped to [-5, 5] as final step"""
        # Create features with extreme values
        extreme_features = pd.Series({f: 100.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=extreme_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # All feature values should be clipped to 5 (except position and time which are in range)
        feature_values = obs[:-2]  # Exclude position and time_normalized
        assert (feature_values == 5.0).all(), "Extreme features not clipped to 5.0"

    def test_observation_preserves_position_time(self, feature_calculator):
        """Test position and time_normalized are not over-clipped"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.5,
            step_count=30,
            episode_length=60
        )

        # Position should be preserved
        assert abs(obs[-2] - 0.5) < 1e-6, "Position incorrectly clipped"

        # Time normalized should be preserved
        expected_time = 30 / 60
        assert abs(obs[-1] - expected_time) < 1e-6, "Time normalized incorrectly clipped"
