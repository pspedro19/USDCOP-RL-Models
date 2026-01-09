"""
Integration Tests: Feature Builder
==================================

Tests the feature construction pipeline for the trading system.

Tests cover:
- Observation dimension validation (15)
- Feature order validation
- Normalization ranges
- Missing data handling
- Technical indicator calculations

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-26
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import json


# Expected feature order from feature_config.json
EXPECTED_FEATURE_ORDER = [
    'log_ret_5m',
    'log_ret_1h',
    'log_ret_4h',
    'rsi_9',
    'atr_pct',
    'adx_14',
    'dxy_z',
    'dxy_change_1d',
    'vix_z',
    'embi_z',
    'brent_change_1d',
    'rate_spread',
    'usdmxn_change_1d'
]

EXPECTED_OBS_DIM = 15
EXPECTED_FEATURE_COUNT = 13


@pytest.fixture
def feature_builder():
    """Create feature builder instance"""
    try:
        import sys
        src_path = Path(__file__).parent.parent.parent / 'src'
        sys.path.insert(0, str(src_path))

        from core.services.feature_builder import FeatureBuilder
        return FeatureBuilder()
    except ImportError:
        pytest.skip("FeatureBuilder not available")


@pytest.fixture
def sample_ohlcv_100bars() -> pd.DataFrame:
    """Generate 100 bars of OHLCV data for testing"""
    np.random.seed(42)

    base_price = 4250.0
    n_bars = 100

    # Generate random walk for close prices
    returns = np.random.normal(0, 0.001, n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    # Generate other prices around close
    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    open_prices = close_prices * (1 + np.random.normal(0, 0.0003, n_bars))

    return pd.DataFrame({
        'time': pd.date_range('2025-12-01 08:00:00', periods=n_bars, freq='5min'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100, 10000, n_bars)
    })


@pytest.fixture
def sample_macro_data() -> pd.DataFrame:
    """Generate macro data for testing"""
    dates = pd.date_range('2025-11-30', periods=5, freq='D')

    return pd.DataFrame({
        'date': dates,
        'dxy': [103.5, 103.2, 103.8, 104.0, 103.7],
        'vix': [18.5, 19.2, 17.8, 20.1, 18.9],
        'embi': [320.0, 325.0, 318.0, 330.0, 322.0],
        'brent': [75.0, 74.5, 76.0, 75.5, 75.8],
        'treasury_2y': [4.5, 4.6, 4.4, 4.5, 4.55],
        'treasury_10y': [4.2, 4.3, 4.1, 4.25, 4.22],
        'usdmxn': [17.2, 17.3, 17.1, 17.25, 17.22]
    })


@pytest.fixture
def feature_config_path() -> Path:
    """Path to feature config file"""
    return Path(__file__).parent.parent.parent / 'config' / 'feature_config.json'


@pytest.fixture
def loaded_feature_config(feature_config_path) -> Dict[str, Any]:
    """Load feature config"""
    if not feature_config_path.exists():
        pytest.skip(f"Config not found: {feature_config_path}")

    with open(feature_config_path, 'r') as f:
        return json.load(f)


@pytest.mark.integration
class TestObservationDimension:
    """Tests for observation dimension (15)"""

    def test_observation_dimension_is_15(self, feature_builder):
        """Observation has correct dimension (15)"""
        obs_dim = feature_builder.get_observation_dim()
        assert obs_dim == EXPECTED_OBS_DIM, \
            f"Expected {EXPECTED_OBS_DIM}-dim observation, got {obs_dim}"

    def test_observation_dimension_in_config(self, loaded_feature_config):
        """Config specifies 15-dim observation"""
        config_dim = loaded_feature_config['observation_space']['dimension']
        assert config_dim == EXPECTED_OBS_DIM, \
            f"Config specifies {config_dim}-dim, expected {EXPECTED_OBS_DIM}"

    def test_observation_breakdown(self, loaded_feature_config):
        """Observation is 13 features + 2 state variables"""
        order = loaded_feature_config['observation_space']['order']
        additional = loaded_feature_config['observation_space']['additional_in_env']

        assert len(order) == EXPECTED_FEATURE_COUNT, \
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(order)}"
        assert len(additional) == 2, \
            f"Expected 2 additional state vars, got {len(additional)}"
        assert 'position' in additional
        assert 'time_normalized' in additional


@pytest.mark.integration
class TestFeatureOrder:
    """Tests for feature order"""

    def test_feature_order_matches_config(self, feature_builder):
        """Features are in correct order"""
        feature_order = feature_builder.get_feature_order()

        assert len(feature_order) == EXPECTED_FEATURE_COUNT, \
            f"Expected {EXPECTED_FEATURE_COUNT} features, got {len(feature_order)}"

        for i, expected_feat in enumerate(EXPECTED_FEATURE_ORDER):
            assert feature_order[i] == expected_feat, \
                f"Feature {i}: expected '{expected_feat}', got '{feature_order[i]}'"

    def test_feature_order_from_config(self, loaded_feature_config):
        """Config feature order matches expected"""
        config_order = loaded_feature_config['observation_space']['order']

        for i, expected_feat in enumerate(EXPECTED_FEATURE_ORDER):
            assert config_order[i] == expected_feat, \
                f"Config feature {i}: expected '{expected_feat}', got '{config_order[i]}'"

    def test_no_eliminated_features_in_order(self, feature_builder):
        """Eliminated features are not in feature order"""
        eliminated = ['hour_sin', 'hour_cos', 'bb_position', 'dxy_mom_5d',
                     'vix_regime', 'brent_vol_5d', 'sma_ratio', 'macd_hist']

        feature_order = feature_builder.get_feature_order()

        for feat in eliminated:
            assert feat not in feature_order, \
                f"Eliminated feature '{feat}' found in feature order"


@pytest.mark.integration
class TestNormalization:
    """Tests for feature normalization"""

    def test_normalization_produces_expected_ranges(self, feature_builder, sample_ohlcv_100bars):
        """Normalization produces expected ranges"""
        # Build features (normalize=True by default)
        features = feature_builder.build_batch(sample_ohlcv_100bars)

        # Check normalized technical features
        for col in ['rsi_9', 'atr_pct', 'adx_14']:
            if col in features.columns:
                values = features[col].dropna()
                if len(values) > 0:
                    # After z-score normalization, should be mostly within [-4, 4]
                    within_range = ((values >= -4) & (values <= 4)).mean()
                    assert within_range > 0.90, \
                        f"{col}: Only {within_range*100:.1f}% within [-4, 4]"

    def test_log_returns_clipped(self, feature_builder, sample_ohlcv_100bars):
        """Log returns are properly clipped"""
        features = feature_builder.build_batch(sample_ohlcv_100bars, normalize=True)

        for col in ['log_ret_5m', 'log_ret_1h', 'log_ret_4h']:
            if col in features.columns:
                values = features[col].dropna()
                if len(values) > 0:
                    # After normalization and clipping
                    assert values.min() >= -5, f"{col} below -5"
                    assert values.max() <= 5, f"{col} above 5"

    def test_zscore_normalization_stats(self, feature_builder):
        """Z-score normalization uses correct stats"""
        norm_stats = feature_builder._norm_stats

        # Check some key features have stats defined
        key_features = ['log_ret_5m', 'rsi_9', 'atr_pct']

        for feat in key_features:
            assert feat in norm_stats, f"Missing norm stats for {feat}"
            assert 'mean' in norm_stats[feat], f"Missing mean for {feat}"
            assert 'std' in norm_stats[feat], f"Missing std for {feat}"
            assert norm_stats[feat]['std'] > 0, f"Std must be positive for {feat}"

    def test_normalization_preserves_nan_handling(self, feature_builder):
        """Normalization handles NaN values correctly"""
        value_with_nan = float('nan')
        normalized = feature_builder.normalize_feature('log_ret_5m', value_with_nan)

        # NaN should be converted to 0.0
        assert normalized == 0.0, "NaN should normalize to 0.0"


@pytest.mark.integration
class TestHandlesMissingData:
    """Tests for handling missing data"""

    def test_handles_nan_in_features(self, feature_builder):
        """Properly handles NaN values in feature dict"""
        features = {
            'log_ret_5m': 0.001,
            'log_ret_1h': np.nan,  # NaN
            'log_ret_4h': 0.002,
            'rsi_9': 55.0,
            'atr_pct': 0.08,
            'adx_14': 25.0,
            'dxy_z': -0.5,
            'dxy_change_1d': 0.01,
            'vix_z': 0.3,
            'embi_z': 0.1,
            'brent_change_1d': 0.02,
            'rate_spread': -0.5,
            'usdmxn_change_1d': 0.003
        }

        obs = feature_builder.build_observation(
            features,
            position=0.5,
            bar_number=30
        )

        # Observation should be valid (no NaN)
        assert not np.any(np.isnan(obs)), "Observation contains NaN"

        # NaN feature should be converted to 0
        # log_ret_1h is at index 1
        assert obs[1] == 0.0, "NaN feature should be 0.0"

    def test_handles_all_nan_features(self, feature_builder):
        """Handles case where all features are NaN"""
        features = {feat: np.nan for feat in EXPECTED_FEATURE_ORDER}

        obs = feature_builder.build_observation(
            features,
            position=0.0,
            bar_number=1
        )

        # Should produce valid observation with zeros
        assert obs.shape == (EXPECTED_OBS_DIM,)
        assert not np.any(np.isnan(obs))

        # All features should be 0, only position and time may differ
        for i in range(EXPECTED_FEATURE_COUNT):
            assert obs[i] == 0.0

    def test_handles_inf_in_features(self, feature_builder):
        """Properly handles Inf values"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}
        features['log_ret_5m'] = float('inf')

        obs = feature_builder.build_observation(
            features,
            position=0.5,
            bar_number=30
        )

        # Observation should be clipped (no inf)
        assert not np.any(np.isinf(obs)), "Observation contains Inf"

        # Inf should be clipped to max (5.0)
        assert obs[0] <= 5.0, "Inf should be clipped to 5.0"

    def test_missing_feature_raises_error(self, feature_builder):
        """Missing required feature raises appropriate error"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}
        del features['log_ret_5m']  # Remove required feature

        with pytest.raises(Exception):  # Should raise FeatureMissingError or KeyError
            feature_builder.build_observation(
                features,
                position=0.5,
                bar_number=30
            )


@pytest.mark.integration
class TestTechnicalIndicators:
    """Tests for technical indicator calculations"""

    def test_rsi_range(self, feature_builder, sample_ohlcv_100bars):
        """RSI is in valid range [0, 100]"""
        close = sample_ohlcv_100bars['close']
        rsi = feature_builder.calc_rsi(close, period=9)

        valid_rsi = rsi.dropna()
        assert (valid_rsi >= 0).all(), "RSI below 0"
        assert (valid_rsi <= 100).all(), "RSI above 100"

    def test_rsi_period_is_9(self, feature_builder):
        """RSI uses period 9 (not 14)"""
        assert feature_builder.RSI_PERIOD == 9, \
            f"RSI period should be 9, got {feature_builder.RSI_PERIOD}"

    def test_atr_period_is_10(self, feature_builder):
        """ATR uses period 10 (not 14)"""
        assert feature_builder.ATR_PERIOD == 10, \
            f"ATR period should be 10, got {feature_builder.ATR_PERIOD}"

    def test_adx_period_is_14(self, feature_builder):
        """ADX uses period 14"""
        assert feature_builder.ADX_PERIOD == 14, \
            f"ADX period should be 14, got {feature_builder.ADX_PERIOD}"

    def test_atr_pct_positive(self, feature_builder, sample_ohlcv_100bars):
        """ATR% is always positive"""
        high = sample_ohlcv_100bars['high']
        low = sample_ohlcv_100bars['low']
        close = sample_ohlcv_100bars['close']

        atr_pct = feature_builder.calc_atr_pct(high, low, close, period=10)
        valid_atr = atr_pct.dropna()

        assert (valid_atr >= 0).all(), "ATR% should be non-negative"

    def test_adx_range(self, feature_builder, sample_ohlcv_100bars):
        """ADX is in valid range [0, 100]"""
        high = sample_ohlcv_100bars['high']
        low = sample_ohlcv_100bars['low']
        close = sample_ohlcv_100bars['close']

        adx = feature_builder.calc_adx(high, low, close, period=14)
        valid_adx = adx.dropna()

        assert (valid_adx >= 0).all(), "ADX below 0"
        assert (valid_adx <= 100).all(), "ADX above 100"

    def test_log_returns_calculation(self, feature_builder, sample_ohlcv_100bars):
        """Log returns are calculated correctly"""
        close = sample_ohlcv_100bars['close']

        log_ret = feature_builder.calc_log_return(close, periods=1)

        # Manual verification for first valid value
        expected = np.log(close.iloc[1] / close.iloc[0])
        actual = log_ret.iloc[1]

        assert np.isclose(expected, actual, atol=1e-10), \
            f"Log return mismatch: expected {expected}, got {actual}"


@pytest.mark.integration
class TestTimeNormalized:
    """Tests for time_normalized calculation"""

    def test_time_normalized_range(self, feature_builder):
        """time_normalized is in [0, 0.983] range"""
        # First bar
        t1 = feature_builder.calculate_time_normalized(1)
        assert t1 == 0.0, f"Bar 1 should be 0.0, got {t1}"

        # Last bar
        t60 = feature_builder.calculate_time_normalized(60)
        expected = 59 / 60  # 0.9833...
        assert np.isclose(t60, expected, atol=1e-6), \
            f"Bar 60 should be ~{expected:.4f}, got {t60}"

    def test_time_normalized_never_reaches_1(self, feature_builder):
        """time_normalized never equals 1.0"""
        for bar in range(1, 61):
            t = feature_builder.calculate_time_normalized(bar)
            assert t < 1.0, f"Bar {bar}: time_normalized = {t} >= 1.0"

    def test_time_normalized_formula(self, feature_builder):
        """time_normalized follows (bar_number - 1) / episode_length formula"""
        for bar in [1, 15, 30, 45, 60]:
            expected = (bar - 1) / 60
            actual = feature_builder.calculate_time_normalized(bar)
            assert np.isclose(actual, expected, atol=1e-10), \
                f"Bar {bar}: expected {expected}, got {actual}"


@pytest.mark.integration
class TestObservationConstruction:
    """Tests for observation vector construction"""

    def test_observation_shape(self, feature_builder):
        """Observation has correct shape (15,)"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}

        obs = feature_builder.build_observation(
            features,
            position=0.0,
            bar_number=30
        )

        assert obs.shape == (EXPECTED_OBS_DIM,), \
            f"Expected shape ({EXPECTED_OBS_DIM},), got {obs.shape}"

    def test_observation_dtype(self, feature_builder):
        """Observation has correct dtype (float32)"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}

        obs = feature_builder.build_observation(
            features,
            position=0.0,
            bar_number=30
        )

        assert obs.dtype == np.float32, \
            f"Expected dtype float32, got {obs.dtype}"

    def test_observation_clipped_to_5(self, feature_builder):
        """Observation values are clipped to [-5, 5]"""
        features = {feat: 10.0 for feat in EXPECTED_FEATURE_ORDER}  # All extreme values

        obs = feature_builder.build_observation(
            features,
            position=0.5,
            bar_number=30
        )

        assert obs.min() >= -5.0, f"Observation below -5: {obs.min()}"
        assert obs.max() <= 5.0, f"Observation above 5: {obs.max()}"

    def test_position_in_observation(self, feature_builder):
        """Position is correctly placed in observation"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}

        test_positions = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for pos in test_positions:
            obs = feature_builder.build_observation(
                features,
                position=pos,
                bar_number=30
            )

            # Position is at index 13 (after 13 features)
            assert obs[13] == pos, \
                f"Position mismatch: expected {pos}, got {obs[13]}"

    def test_time_normalized_in_observation(self, feature_builder):
        """time_normalized is correctly placed in observation"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}

        bar = 30
        expected_time = (bar - 1) / 60

        obs = feature_builder.build_observation(
            features,
            position=0.0,
            bar_number=bar
        )

        # time_normalized is at index 14 (last position)
        assert np.isclose(obs[14], expected_time, atol=1e-6), \
            f"time_normalized mismatch: expected {expected_time}, got {obs[14]}"


@pytest.mark.integration
class TestBatchProcessing:
    """Tests for batch feature processing"""

    def test_build_batch_returns_dataframe(self, feature_builder, sample_ohlcv_100bars):
        """build_batch returns DataFrame with features"""
        result = feature_builder.build_batch(sample_ohlcv_100bars)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv_100bars)

    def test_build_batch_has_technical_features(self, feature_builder, sample_ohlcv_100bars):
        """Batch result includes technical features"""
        result = feature_builder.build_batch(sample_ohlcv_100bars)

        technical_features = ['rsi_9', 'atr_pct', 'adx_14']
        for feat in technical_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_build_batch_has_log_returns(self, feature_builder, sample_ohlcv_100bars):
        """Batch result includes log returns"""
        result = feature_builder.build_batch(sample_ohlcv_100bars)

        return_features = ['log_ret_5m', 'log_ret_1h', 'log_ret_4h']
        for feat in return_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_build_batch_with_macro(self, feature_builder, sample_ohlcv_100bars, sample_macro_data):
        """Batch result includes macro features when provided"""
        result = feature_builder.build_batch(sample_ohlcv_100bars, sample_macro_data)

        macro_features = ['dxy_z', 'vix_z', 'embi_z']
        for feat in macro_features:
            if feat in result.columns:
                assert result[feat].notna().any(), f"Macro feature {feat} is all NaN"


@pytest.mark.integration
class TestValidation:
    """Tests for observation validation"""

    def test_validate_valid_observation(self, feature_builder):
        """Valid observation passes validation"""
        features = {feat: 0.0 for feat in EXPECTED_FEATURE_ORDER}

        obs = feature_builder.build_observation(
            features,
            position=0.0,
            bar_number=30
        )

        # Should not raise
        is_valid = feature_builder.validate_observation(obs)
        assert is_valid

    def test_validate_wrong_dimension_raises(self, feature_builder):
        """Wrong dimension observation raises error"""
        wrong_obs = np.zeros(10, dtype=np.float32)

        with pytest.raises(Exception):  # ObservationDimensionError
            feature_builder.validate_observation(wrong_obs)

    def test_validate_nan_raises(self, feature_builder):
        """Observation with NaN raises validation error"""
        obs = np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
        obs[0] = np.nan

        with pytest.raises(Exception):  # ValidationError
            feature_builder.validate_observation(obs)

    def test_validate_out_of_range_raises(self, feature_builder):
        """Observation out of [-5, 5] raises validation error"""
        obs = np.zeros(EXPECTED_OBS_DIM, dtype=np.float32)
        obs[0] = 10.0  # Out of range

        with pytest.raises(Exception):  # ValidationError
            feature_builder.validate_observation(obs, check_range=True)
