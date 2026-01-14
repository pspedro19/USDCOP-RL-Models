"""
Integration Test: Observation Space
====================================

Tests observation space compatibility with PPO model:
- 15 dimensions exactly
- Correct feature ordering
- Proper value ranges
- Model compatibility (if model available)

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-16
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.mark.integration
class TestObservationDimension:
    """Test observation space has correct dimensions"""

    def test_observation_is_15_dimensions(self, feature_calculator):
        """Observation must be EXACTLY 15 dimensions for model compatibility"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        assert obs.shape == (15,), \
            f"CRITICAL: Observation must be 15-dim for model compatibility, got {obs.shape[0]}"

    def test_feature_count_is_13(self, feature_calculator):
        """Feature order should contain exactly 13 features"""
        assert len(feature_calculator.feature_order) == 13, \
            f"Expected 13 features, got {len(feature_calculator.feature_order)}"

    def test_state_variables_are_2(self, feature_config):
        """State variables (position, time_normalized) should be 2"""
        additional = feature_config['observation_space']['additional_in_env']

        assert len(additional) == 2, \
            f"Expected 2 state variables, got {len(additional)}"
        assert 'position' in additional, "Missing position state variable"
        assert 'time_normalized' in additional, "Missing time_normalized state variable"


@pytest.mark.integration
class TestObservationOrdering:
    """Test observation maintains correct feature ordering"""

    def test_feature_order_consistent(self, feature_calculator, feature_config):
        """Test feature ordering is consistent with config"""
        expected_order = feature_config['observation_space']['order']
        actual_order = feature_calculator.feature_order

        assert actual_order == expected_order, \
            f"Feature order mismatch.\nConfig: {expected_order}\nCalculator: {actual_order}"

    def test_observation_feature_positions(self, feature_calculator):
        """Test that features appear in correct positions in observation"""
        # Create features with distinct values
        test_features = pd.Series({
            feature_calculator.feature_order[i]: float(i)
            for i in range(13)
        })

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # Check each feature is in its correct position
        for i, feat_name in enumerate(feature_calculator.feature_order):
            expected_val = test_features[feat_name]
            actual_val = obs[i]

            assert abs(actual_val - expected_val) < 1e-5, \
                f"Feature {feat_name} not in position {i}. Expected {expected_val}, got {actual_val}"

    def test_position_and_time_at_end(self, feature_calculator):
        """Test position and time_normalized are last two elements"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.75,
            step_count=45,
            episode_length=60
        )

        # Position should be second-to-last
        assert abs(obs[-2] - 0.75) < 1e-6, \
            f"Position should be at index -2, got value {obs[-2]}"

        # Time normalized should be last
        expected_time = 45 / 60
        assert abs(obs[-1] - expected_time) < 1e-6, \
            f"Time normalized should be at index -1, got value {obs[-1]}"


@pytest.mark.integration
class TestObservationValueRanges:
    """Test observation values are within expected ranges"""

    def test_observation_clipped_to_bounds(self, feature_calculator):
        """Test observation is clipped to [-5, 5] for numerical stability"""
        # Create features with extreme values
        extreme_features = pd.Series({f: 1000.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=extreme_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # All values should be within [-5, 5]
        assert (obs >= -5).all(), f"Observation has values below -5: {obs[obs < -5]}"
        assert (obs <= 5).all(), f"Observation has values above 5: {obs[obs > 5]}"

    def test_position_range(self, feature_calculator):
        """Test position values are in [-1, 1]"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        for pos in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            obs = feature_calculator.build_observation(
                features=test_features,
                position=pos,
                step_count=30,
                episode_length=60
            )

            assert -1.0 <= obs[-2] <= 1.0, \
                f"Position {pos} out of range [-1, 1]: {obs[-2]}"

    def test_time_normalized_range(self, feature_calculator):
        """Test time_normalized is in [0, 0.983]"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        # Test boundary values
        test_steps = [0, 1, 29, 30, 59]

        for step in test_steps:
            obs = feature_calculator.build_observation(
                features=test_features,
                position=0.0,
                step_count=step,
                episode_length=60
            )

            time_norm = obs[-1]

            # Should be in [0, 0.983]
            assert 0.0 <= time_norm <= 0.983, \
                f"time_normalized at step {step} out of range: {time_norm}"

            # Verify exact formula
            expected = step / 60
            assert abs(time_norm - expected) < 1e-6, \
                f"time_normalized formula incorrect at step {step}"


@pytest.mark.integration
class TestModelCompatibility:
    """Test observation space is compatible with PPO model"""

    def test_model_observation_space_matches(self):
        """Test model expects 15-dim observation space"""
        from pathlib import Path

        models_dir = Path(__file__).parent.parent.parent / 'models'
        # Try primary first, fallback to legacy
        model_path = models_dir / 'ppo_primary.zip'
        if not model_path.exists():
            model_path = models_dir / 'ppo_legacy.zip'

        if not model_path.exists():
            pytest.skip("Model not available for testing")

        try:
            from stable_baselines3 import PPO

            model = PPO.load(str(model_path))
            obs_space = model.observation_space

            assert obs_space.shape[0] == 15, \
                f"Model expects {obs_space.shape[0]} dims, but we provide 15"

        except ImportError:
            pytest.skip("stable-baselines3 not installed")

    def test_model_can_predict(self, feature_calculator):
        """Test model can make predictions with our observations"""
        from pathlib import Path

        models_dir = Path(__file__).parent.parent.parent / 'models'
        # Try primary first, fallback to legacy
        model_path = models_dir / 'ppo_primary.zip'
        if not model_path.exists():
            model_path = models_dir / 'ppo_legacy.zip'

        if not model_path.exists():
            pytest.skip("Model not available for testing")

        try:
            from stable_baselines3 import PPO

            model = PPO.load(str(model_path))

            # Create test observation
            test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})
            obs = feature_calculator.build_observation(
                features=test_features,
                position=0.0,
                step_count=30,
                episode_length=60
            )

            # Reshape for model (needs batch dimension)
            obs_batch = obs.reshape(1, -1)

            # Model should be able to predict without errors
            action, _states = model.predict(obs_batch, deterministic=True)

            # Action should be in [-1, 1] range
            assert -1.0 <= action[0] <= 1.0, \
                f"Model action out of range: {action[0]}"

        except ImportError:
            pytest.skip("stable-baselines3 not installed")
        except Exception as e:
            pytest.fail(f"Model prediction failed: {e}")


@pytest.mark.integration
class TestObservationBatch:
    """Test batch observation construction"""

    def test_multiple_observations(self, feature_calculator, sample_ohlcv_df):
        """Test building observations for multiple timesteps"""
        features = feature_calculator.compute_technical_features(sample_ohlcv_df)

        # Build observations for first 10 valid rows
        observations = []
        for i in range(50, 60):  # Skip warmup period
            row = features.iloc[i]

            feature_values = pd.Series({
                feat: row.get(feat, 0.0) for feat in feature_calculator.feature_order
            })

            obs = feature_calculator.build_observation(
                features=feature_values,
                position=0.0,
                step_count=i - 50,
                episode_length=60
            )

            observations.append(obs)

        # All observations should be 15-dim
        assert all(obs.shape == (15,) for obs in observations), \
            "Not all observations are 15-dimensional"

        # Convert to array
        obs_array = np.array(observations)
        assert obs_array.shape == (10, 15), \
            f"Batch shape incorrect: {obs_array.shape}"

    def test_observation_time_progression(self, feature_calculator):
        """Test time_normalized progresses correctly across episode"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        time_values = []
        for step in range(60):
            obs = feature_calculator.build_observation(
                features=test_features,
                position=0.0,
                step_count=step,
                episode_length=60
            )
            time_values.append(obs[-1])

        # Time should be monotonically increasing
        assert all(time_values[i] < time_values[i+1] for i in range(59)), \
            "time_normalized not monotonically increasing"

        # First should be 0, last should be 0.983
        assert abs(time_values[0] - 0.0) < 1e-6, "First time should be 0.0"
        assert abs(time_values[-1] - 0.983) < 0.001, "Last time should be ~0.983"


@pytest.mark.integration
class TestNaNHandling:
    """Test handling of NaN values in observations"""

    def test_nan_features_replaced_with_zero(self, feature_calculator):
        """Test NaN features are replaced with 0.0"""
        # Create features with some NaN values
        test_features = pd.Series({
            feature_calculator.feature_order[0]: np.nan,
            feature_calculator.feature_order[1]: 0.5,
            feature_calculator.feature_order[2]: np.nan,
        })

        # Fill remaining with zeros
        for feat in feature_calculator.feature_order[3:]:
            test_features[feat] = 0.0

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # NaN values should be replaced with 0
        assert obs[0] == 0.0, "NaN at position 0 not replaced with 0"
        assert abs(obs[1] - 0.5) < 1e-6, "Valid value at position 1 changed"
        assert obs[2] == 0.0, "NaN at position 2 not replaced with 0"

        # No NaN should remain in observation
        assert not np.isnan(obs).any(), "Observation contains NaN values"


@pytest.mark.integration
class TestObservationConsistency:
    """Test observation construction is deterministic and consistent"""

    def test_same_inputs_same_output(self, feature_calculator):
        """Test same inputs always produce same observation"""
        test_features = pd.Series({f: 0.5 for f in feature_calculator.feature_order})

        obs1 = feature_calculator.build_observation(
            features=test_features,
            position=0.3,
            step_count=25,
            episode_length=60
        )

        obs2 = feature_calculator.build_observation(
            features=test_features,
            position=0.3,
            step_count=25,
            episode_length=60
        )

        # Should be identical
        assert np.allclose(obs1, obs2, atol=1e-10), \
            "Same inputs produced different observations"

    def test_observation_dtype(self, feature_calculator):
        """Test observation has correct dtype (float32 for model)"""
        test_features = pd.Series({f: 0.0 for f in feature_calculator.feature_order})

        obs = feature_calculator.build_observation(
            features=test_features,
            position=0.0,
            step_count=30,
            episode_length=60
        )

        # Should be float32 for model compatibility
        assert obs.dtype == np.float32, \
            f"Observation dtype should be float32, got {obs.dtype}"
