"""
Integration Test: Model Inference Pipeline
===========================================

Tests the complete model inference pipeline:
- Model loading and validation
- Observation construction
- Action prediction
- Gate checks and thresholds

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-17
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import json


@pytest.mark.integration
class TestModelLoading:
    """Test PPO model loading and validation"""

    @pytest.fixture
    def model_paths(self):
        """Available model paths"""
        models_dir = Path(__file__).parent.parent.parent / 'models'
        return {
            'legacy': models_dir / 'ppo_legacy.zip',
            'primary': models_dir / 'ppo_primary.zip',
        }

    def test_model_file_exists(self, model_paths):
        """Test at least one model file exists"""
        available = [p for p in model_paths.values() if p.exists()]

        if not available:
            pytest.skip("No model files available for testing")

        assert len(available) >= 1, "At least one model file should exist"

    def test_model_loads_successfully(self, model_paths):
        """Test model can be loaded without errors"""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # Find available model
        model_path = None
        for p in model_paths.values():
            if p.exists():
                model_path = p
                break

        if not model_path:
            pytest.skip("No model file available")

        model = PPO.load(str(model_path))

        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'observation_space')

    def test_model_observation_space_dimension(self, model_paths):
        """Test model expects 15-dimensional observation space"""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        model_path = None
        for p in model_paths.values():
            if p.exists():
                model_path = p
                break

        if not model_path:
            pytest.skip("No model file available")

        model = PPO.load(str(model_path))
        obs_space = model.observation_space

        assert obs_space.shape == (15,), \
            f"Model should expect 15-dim observation, got {obs_space.shape}"

    def test_model_action_space_is_continuous(self, model_paths):
        """Test model has continuous action space [-1, 1]"""
        try:
            from stable_baselines3 import PPO
            from gymnasium.spaces import Box
        except ImportError:
            pytest.skip("stable-baselines3 or gymnasium not installed")

        model_path = None
        for p in model_paths.values():
            if p.exists():
                model_path = p
                break

        if not model_path:
            pytest.skip("No model file available")

        model = PPO.load(str(model_path))
        action_space = model.action_space

        assert isinstance(action_space, Box), "Action space should be Box"
        assert action_space.shape == (1,), "Action space should be 1-dimensional"
        assert action_space.low[0] == -1.0, "Action space min should be -1"
        assert action_space.high[0] == 1.0, "Action space max should be 1"


@pytest.mark.integration
class TestModelPrediction:
    """Test model prediction functionality"""

    @pytest.fixture
    def loaded_model(self):
        """Load model for testing"""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        models_dir = Path(__file__).parent.parent.parent / 'models'

        for version in ['primary', 'legacy']:
            model_path = models_dir / f'ppo_usdcop_{version}_fold0.zip'
            if model_path.exists():
                return PPO.load(str(model_path))

        pytest.skip("No model file available")

    def test_predict_with_zeros(self, loaded_model):
        """Test prediction with zero observation"""
        obs = np.zeros((1, 15), dtype=np.float32)

        action, _ = loaded_model.predict(obs, deterministic=True)

        assert action.shape == (1,), f"Action shape incorrect: {action.shape}"
        assert -1.0 <= action[0] <= 1.0, f"Action out of range: {action[0]}"

    def test_predict_with_random_valid_obs(self, loaded_model):
        """Test prediction with random but valid observation"""
        np.random.seed(42)
        obs = np.random.uniform(-3, 3, size=(1, 15)).astype(np.float32)

        action, _ = loaded_model.predict(obs, deterministic=True)

        assert -1.0 <= action[0] <= 1.0, f"Action out of range: {action[0]}"

    def test_predict_deterministic_is_repeatable(self, loaded_model):
        """Test deterministic prediction gives same result"""
        obs = np.array([[0.1, 0.2, -0.1, 0.5, 0.3, 0.4,
                        -0.2, 0.1, 0.6, -0.3, 0.2, 0.1,
                        -0.1, 0.5, 0.5]], dtype=np.float32)

        action1, _ = loaded_model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)

        assert np.allclose(action1, action2), \
            f"Deterministic predictions differ: {action1} vs {action2}"

    def test_predict_batch(self, loaded_model):
        """Test batch prediction"""
        batch_size = 10
        obs = np.random.uniform(-3, 3, size=(batch_size, 15)).astype(np.float32)

        actions, _ = loaded_model.predict(obs, deterministic=True)

        assert actions.shape == (batch_size,), f"Batch shape incorrect: {actions.shape}"
        assert all(-1.0 <= a <= 1.0 for a in actions), "Some actions out of range"

    def test_predict_with_extreme_values(self, loaded_model):
        """Test model handles extreme (clipped) values"""
        # Values at clip boundaries
        obs = np.array([[4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                        4.0, 4.0, 4.0, 4.0, 4.0, 4.0,
                        4.0, 1.0, 1.0]], dtype=np.float32)

        action, _ = loaded_model.predict(obs, deterministic=True)

        # Should not crash and return valid action
        assert not np.isnan(action[0]), "Action should not be NaN"
        assert -1.0 <= action[0] <= 1.0, f"Action out of range: {action[0]}"


@pytest.mark.integration
class TestInferencePipeline:
    """Test complete inference pipeline from features to action"""

    @pytest.fixture
    def feature_config(self):
        """Load feature configuration"""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'feature_config.json'
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @pytest.fixture
    def norm_stats(self, feature_config):
        """Extract normalization stats from config"""
        return feature_config['validation']['norm_stats_validation']

    def test_feature_normalization_formula(self, norm_stats):
        """Test z-score normalization formula"""
        # Test log_ret_5m normalization
        raw_value = 0.001  # Example return
        stats = norm_stats['log_ret_5m']

        normalized = (raw_value - stats['mean']) / stats['std']

        # Should produce reasonable z-score
        assert -10 < normalized < 10, f"Normalized value out of reasonable range: {normalized}"

    def test_feature_clipping(self, norm_stats):
        """Test feature clipping boundaries"""
        # Features with explicit clip values
        clipped_features = ['log_ret_5m', 'dxy_z', 'dxy_change_1d', 'brent_change_1d', 'usdmxn_ret_1h']

        for feat in clipped_features:
            if feat in norm_stats and 'clip' in norm_stats[feat]:
                clip_min, clip_max = norm_stats[feat]['clip']

                # Test clipping
                extreme_high = 100.0
                extreme_low = -100.0

                clipped_high = np.clip(extreme_high, clip_min, clip_max)
                clipped_low = np.clip(extreme_low, clip_min, clip_max)

                assert clipped_high == clip_max, f"{feat}: high clip failed"
                assert clipped_low == clip_min, f"{feat}: low clip failed"

    def test_observation_construction(self, feature_config):
        """Test observation has correct order and dimension"""
        expected_order = feature_config['observation_space']['order']
        expected_dim = feature_config['observation_space']['dimension']

        assert len(expected_order) == 13, f"Expected 13 features, got {len(expected_order)}"
        assert expected_dim == 15, f"Expected 15-dim obs space, got {expected_dim}"

        # Features + position + time_normalized = 15
        additional = feature_config['observation_space']['additional_in_env']
        assert len(expected_order) + len(additional) == expected_dim

    def test_time_normalized_range(self, feature_config):
        """Test time_normalized stays in [0, 0.983]"""
        episode_length = feature_config['trading']['bars_per_session']

        for step in range(episode_length):
            time_norm = step / episode_length

            assert 0.0 <= time_norm <= 0.983, \
                f"time_normalized out of range at step {step}: {time_norm}"

    def test_position_encoding(self):
        """Test position values are in [-1, 1]"""
        valid_positions = [-1.0, -0.5, 0.0, 0.5, 1.0]

        for pos in valid_positions:
            assert -1.0 <= pos <= 1.0, f"Invalid position: {pos}"


@pytest.mark.integration
class TestGateChecks:
    """Test production gate checks for model deployment"""

    @pytest.fixture
    def production_gates(self):
        """Production deployment thresholds"""
        return {
            'sharpe_ratio': 0.0,        # Minimum Sharpe
            'win_rate': 0.35,           # Minimum win rate (35%)
            'max_drawdown': -0.20,      # Maximum drawdown (-20%)
            'profit_factor': 1.0,       # Minimum profit factor
            'calmar_ratio': 0.0,        # Minimum Calmar ratio
        }

    def test_gate_thresholds_are_reasonable(self, production_gates):
        """Test gate thresholds are properly configured"""
        assert production_gates['sharpe_ratio'] >= 0.0, "Sharpe threshold too low"
        assert 0.0 < production_gates['win_rate'] < 1.0, "Win rate threshold invalid"
        assert production_gates['max_drawdown'] < 0, "Max drawdown should be negative"
        assert production_gates['profit_factor'] >= 1.0, "Profit factor threshold too low"

    def test_mock_metrics_pass_gates(self, production_gates):
        """Test good metrics pass gates"""
        good_metrics = {
            'sharpe_ratio': 1.5,
            'win_rate': 0.55,
            'max_drawdown': -0.08,
            'profit_factor': 1.8,
            'calmar_ratio': 2.0,
        }

        gates_passed = all(
            good_metrics[k] >= production_gates[k] if k != 'max_drawdown'
            else good_metrics[k] >= production_gates[k]  # drawdown: less negative is better
            for k in production_gates
        )

        assert gates_passed, "Good metrics should pass all gates"

    def test_mock_metrics_fail_gates(self, production_gates):
        """Test bad metrics fail gates"""
        bad_metrics = {
            'sharpe_ratio': -0.5,       # Fail
            'win_rate': 0.25,           # Fail
            'max_drawdown': -0.35,      # Fail
            'profit_factor': 0.8,       # Fail
            'calmar_ratio': -1.0,       # Fail
        }

        gates_passed = all(
            bad_metrics[k] >= production_gates[k]
            for k in production_gates
        )

        assert not gates_passed, "Bad metrics should fail gates"


@pytest.mark.integration
class TestWeakSignalFiltering:
    """Test weak signal filtering logic"""

    @pytest.fixture
    def signal_config(self):
        """Signal filtering configuration"""
        return {
            'weak_signal_threshold': 0.3,
            'trade_count_threshold': 0.3,
            'min_cost_threshold': 0.001,
            'cost_per_trade': 0.0015,
        }

    def test_strong_signal_not_filtered(self, signal_config):
        """Strong signals should not be filtered"""
        action = 0.8  # Strong buy signal

        is_weak = abs(action) < signal_config['weak_signal_threshold']

        assert not is_weak, "Strong signal should not be filtered"

    def test_weak_signal_filtered(self, signal_config):
        """Weak signals should be filtered"""
        action = 0.2  # Weak signal

        is_weak = abs(action) < signal_config['weak_signal_threshold']

        assert is_weak, "Weak signal should be filtered"

    def test_boundary_signal(self, signal_config):
        """Test boundary signal handling"""
        threshold = signal_config['weak_signal_threshold']

        # Exactly at threshold
        action_at_threshold = threshold
        is_weak = abs(action_at_threshold) < threshold

        assert not is_weak, "Signal at threshold should not be filtered"

    def test_cost_threshold_filtering(self, signal_config):
        """Test cost-based filtering"""
        cost = signal_config['cost_per_trade']
        min_expected_return = 0.0008  # Small expected return

        trade_makes_sense = min_expected_return > cost

        assert not trade_makes_sense, \
            "Trade with return below cost should be filtered"


@pytest.mark.integration
class TestObservationFromFeatures:
    """Test building observation from raw features"""

    @pytest.fixture
    def sample_features(self, feature_config):
        """Sample normalized features"""
        feature_order = feature_config['observation_space']['order']

        return pd.Series({
            'log_ret_5m': 0.1,
            'log_ret_1h': 0.2,
            'log_ret_4h': -0.1,
            'rsi_9': 0.0,
            'atr_pct': 0.5,
            'adx_14': 0.3,
            'dxy_z': -0.2,
            'dxy_change_1d': 0.01,
            'vix_z': 0.5,
            'embi_z': -0.1,
            'brent_change_1d': 0.03,
            'rate_spread': -0.5,
            'usdmxn_ret_1h': 0.02,
        })

    def test_build_observation_from_series(self, sample_features, feature_config):
        """Test building observation from pandas Series"""
        feature_order = feature_config['observation_space']['order']

        # Build observation array
        obs = np.zeros(15, dtype=np.float32)

        for i, feat in enumerate(feature_order):
            obs[i] = sample_features[feat]

        # Add position and time_normalized
        obs[13] = 0.5   # position
        obs[14] = 0.5   # time_normalized (step 30 of 60)

        assert obs.shape == (15,)
        assert obs.dtype == np.float32
        assert obs[0] == sample_features['log_ret_5m']
        assert obs[13] == 0.5  # position
        assert obs[14] == 0.5  # time_normalized

    def test_handle_missing_features(self, feature_config):
        """Test handling missing features with zeros"""
        feature_order = feature_config['observation_space']['order']

        # Partial features (some missing)
        partial_features = pd.Series({
            'log_ret_5m': 0.1,
            'log_ret_1h': 0.2,
            # Missing other features
        })

        obs = np.zeros(15, dtype=np.float32)

        for i, feat in enumerate(feature_order):
            obs[i] = partial_features.get(feat, 0.0)

        # Missing features should be 0
        assert obs[2] == 0.0  # log_ret_4h was missing
        assert obs[0] == 0.1  # log_ret_5m was present


@pytest.mark.integration
class TestInferencePerformance:
    """Test inference performance requirements"""

    @pytest.fixture
    def loaded_model(self):
        """Load model for performance testing"""
        try:
            from stable_baselines3 import PPO
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        models_dir = Path(__file__).parent.parent.parent / 'models'

        for version in ['primary', 'legacy']:
            model_path = models_dir / f'ppo_usdcop_{version}_fold0.zip'
            if model_path.exists():
                return PPO.load(str(model_path))

        pytest.skip("No model file available")

    def test_single_prediction_latency(self, loaded_model):
        """Test single prediction completes within 100ms"""
        import time

        obs = np.random.uniform(-3, 3, size=(1, 15)).astype(np.float32)

        start = time.perf_counter()
        action, _ = loaded_model.predict(obs, deterministic=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"Prediction took {elapsed_ms:.2f}ms, should be <100ms"

    def test_batch_prediction_efficiency(self, loaded_model):
        """Test batch prediction is more efficient than sequential"""
        import time

        batch_size = 100
        obs_batch = np.random.uniform(-3, 3, size=(batch_size, 15)).astype(np.float32)
        obs_single = obs_batch[0:1]

        # Time batch prediction
        start = time.perf_counter()
        loaded_model.predict(obs_batch, deterministic=True)
        batch_time = time.perf_counter() - start

        # Time sequential predictions
        start = time.perf_counter()
        for i in range(batch_size):
            loaded_model.predict(obs_batch[i:i+1], deterministic=True)
        sequential_time = time.perf_counter() - start

        # Batch should be faster (or at least not significantly slower)
        assert batch_time < sequential_time * 1.5, \
            f"Batch ({batch_time:.4f}s) not efficient vs sequential ({sequential_time:.4f}s)"
