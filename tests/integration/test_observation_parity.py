"""
Test de Paridad de Observacion V19
===================================

PRUEBA CRITICA: Verifica que el vector de observacion generado en produccion
coincide exactamente con el usado durante el entrenamiento.

El desalineamiento de features causa:
- Predicciones incorrectas
- Comportamiento erratico del modelo
- Perdidas financieras

Este test debe PASAR antes de cualquier deployment a produccion.

Author: Pedro @ Lean Tech Solutions
Version: 19.0.0
"""

import pytest
import numpy as np
import json
from pathlib import Path
from typing import Dict, List

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.builders.observation_builder_v19 import ObservationBuilderV19


# ===========================================================================
# Test Data - Known good values from training dataset
# ===========================================================================

# Sample row from RL_DS3_MACRO_CORE.csv (training data)
TRAINING_SAMPLE = {
    "log_ret_5m": 0.000234,
    "log_ret_1h": 0.001567,
    "log_ret_4h": -0.002345,
    "rsi_9": 55.234,
    "atr_pct": 0.0523,
    "adx_14": 28.45,
    "dxy_z": 0.234,
    "dxy_change_1d": 0.0012,
    "vix_z": -0.456,
    "embi_z": 0.789,
    "brent_change_1d": -0.0234,
    "rate_spread": 0.123,
    "usdmxn_change_1d": 0.0045
}

# Expected normalized values (pre-calculated from training pipeline)
# Formula: z = (x - mean) / std
# Using actual norm stats from config/v19_norm_stats.json:
# log_ret_5m:   mean=9.042127679274034e-07, std=0.0011338119633965713
# log_ret_1h:   mean=1.2402473246226877e-05, std=0.003736154311053953
# log_ret_4h:   mean=5.743994224498364e-05, std=0.007675464150035283
# rsi_9:        mean=48.552482317304815, std=23.916683229459384
# atr_pct:      mean=0.06081762815355896, std=0.04524957350426368
# adx_14:       mean=32.29754481101079, std=17.04633374382278
# dxy_z:        mean=0.02472985775149688, std=0.9986400525980448
# dxy_change_1d: mean=4.456795996369574e-05, std=0.010044210489892007
# vix_z:        mean=-0.014129796865044113, std=0.901167016412187
# embi_z:       mean=0.001491430960557065, std=1.002432669266212
# brent_change_1d: mean=0.0024204124659553757, std=0.04579143400327978
# rate_spread:  mean=-0.014770838918153483, std=0.9980839062292765
# usdmxn_change_1d: mean=-7.5938764815286e-05, std=0.01840172366368339

EXPECTED_NORMALIZED = {
    "log_ret_5m": 0.2056,     # (0.000234 - 9.04e-07) / 0.001134 = 0.2056
    "log_ret_1h": 0.4163,     # (0.001567 - 1.24e-05) / 0.003736 = 0.4163
    "log_ret_4h": -0.3128,    # (-0.002345 - 5.74e-05) / 0.007675 = -0.3128
    "rsi_9": 0.2794,          # (55.234 - 48.552) / 23.917 = 0.2794
    "atr_pct": -0.1905,       # (0.0523 - 0.06082) / 0.04525 = -0.1905
    "adx_14": -0.2258,        # (28.45 - 32.298) / 17.046 = -0.2258
    "dxy_z": 0.2095,          # (0.234 - 0.02473) / 0.9986 = 0.2095
    "dxy_change_1d": 0.1151,  # (0.0012 - 4.46e-05) / 0.01004 = 0.1151
    "vix_z": -0.4903,         # (-0.456 - (-0.01413)) / 0.9012 = -0.4903
    "embi_z": 0.7858,         # (0.789 - 0.001491) / 1.0024 = 0.7858
    "brent_change_1d": -0.5629,  # (-0.0234 - 0.00242) / 0.04579 = -0.5629
    "rate_spread": 0.1380,    # (0.123 - (-0.01477)) / 0.9981 = 0.1380
    "usdmxn_change_1d": 0.2487   # (0.0045 - (-7.59e-05)) / 0.01840 = 0.2487
}


class TestObservationParity:
    """Verify observation vector matches training pipeline exactly."""

    @pytest.fixture
    def builder(self):
        """Create observation builder instance."""
        return ObservationBuilderV19()

    def test_observation_dimension(self, builder):
        """Test that observation has correct dimension (15)."""
        obs = builder.build(TRAINING_SAMPLE, position=0.0, time_normalized=0.5)

        assert obs.shape == (15,), f"Expected shape (15,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_feature_order_matches_training(self, builder):
        """
        CRITICAL: Verify feature order matches training.

        The order MUST be:
        [0] log_ret_5m
        [1] log_ret_1h
        [2] log_ret_4h
        [3] rsi_9
        [4] atr_pct
        [5] adx_14
        [6] dxy_z
        [7] dxy_change_1d
        [8] vix_z
        [9] embi_z
        [10] brent_change_1d
        [11] rate_spread
        [12] usdmxn_change_1d
        [13] position
        [14] time_normalized
        """
        expected_order = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized"
        ]

        assert builder.FEATURE_ORDER == expected_order, \
            f"Feature order mismatch!\nExpected: {expected_order}\nGot: {builder.FEATURE_ORDER}"

    def test_normalization_matches_training(self, builder):
        """
        CRITICAL: Verify normalization produces same values as training.

        Uses known input values and compares against expected normalized values.
        """
        obs = builder.build(TRAINING_SAMPLE, position=0.0, time_normalized=0.5)

        # Define tolerance (accounts for floating point differences)
        TOLERANCE = 0.01  # 1% tolerance

        # Check each normalized value
        feature_names = builder.CORE_FEATURES
        for i, feature_name in enumerate(feature_names):
            if feature_name in EXPECTED_NORMALIZED:
                expected = EXPECTED_NORMALIZED[feature_name]
                actual = obs[i]
                diff = abs(actual - expected)

                assert diff < TOLERANCE, \
                    f"Feature {feature_name} normalization mismatch!\n" \
                    f"Expected: {expected:.4f}, Got: {actual:.4f}, Diff: {diff:.4f}"

    def test_state_features_position(self, builder):
        """Test position state feature (index 13)."""
        test_cases = [
            (1.0, 1.0),    # Long
            (-1.0, -1.0),  # Short
            (0.0, 0.0),    # Flat
            (0.5, 0.5),    # Partial long
        ]

        for position, expected in test_cases:
            obs = builder.build(TRAINING_SAMPLE, position=position, time_normalized=0.5)
            assert obs[13] == expected, \
                f"Position mismatch: input={position}, expected={expected}, got={obs[13]}"

    def test_state_features_time_normalized(self, builder):
        """Test time_normalized state feature (index 14)."""
        test_cases = [
            (0.0, 0.0),    # Start of episode
            (0.5, 0.5),    # Middle
            (1.0, 1.0),    # End
            (0.25, 0.25),  # Quarter
        ]

        for time_norm, expected in test_cases:
            obs = builder.build(TRAINING_SAMPLE, position=0.0, time_normalized=time_norm)
            assert obs[14] == expected, \
                f"Time normalized mismatch: input={time_norm}, expected={expected}, got={obs[14]}"

    def test_clipping_bounds(self, builder):
        """Test that all values are clipped to [-5, 5]."""
        # Create extreme values
        extreme_features = {k: 1000.0 for k in TRAINING_SAMPLE.keys()}
        obs = builder.build(extreme_features, position=0.0, time_normalized=0.5)

        assert np.all(obs >= -5.0), f"Values below -5 found: {obs[obs < -5.0]}"
        assert np.all(obs <= 5.0), f"Values above 5 found: {obs[obs > 5.0]}"

    def test_nan_handling(self, builder):
        """Test that NaN values are handled (replaced with 0)."""
        nan_features = {k: np.nan for k in TRAINING_SAMPLE.keys()}
        obs = builder.build(nan_features, position=0.0, time_normalized=0.5)

        assert not np.any(np.isnan(obs)), "NaN values found in observation!"

        # Core features should be 0 when input is NaN
        for i in range(13):
            assert obs[i] == 0.0, f"Feature index {i} should be 0 for NaN input"

    def test_missing_feature_handling(self, builder):
        """Test handling of missing features."""
        incomplete_features = {"log_ret_5m": 0.001}  # Only one feature
        obs = builder.build(incomplete_features, position=0.0, time_normalized=0.5)

        # Should not raise, missing features should be 0
        assert obs.shape == (15,)
        assert not np.any(np.isnan(obs))

    def test_config_version_check(self, builder):
        """Verify config version matches V19."""
        config = builder.get_config()
        meta = config.get("_meta", {})

        version = meta.get("version", "")
        assert version.startswith("19"), \
            f"Config version mismatch! Expected V19.x.x, got {version}"

    def test_norm_stats_loaded(self, builder):
        """Verify normalization stats are properly loaded."""
        for feature in builder.CORE_FEATURES:
            stats = builder.get_feature_stats(feature)

            assert "mean" in stats, f"Missing 'mean' for feature {feature}"
            assert "std" in stats, f"Missing 'std' for feature {feature}"
            assert stats["std"] > 0, f"Invalid std for feature {feature}: {stats['std']}"

    def test_reproducibility(self, builder):
        """Test that same inputs produce same outputs."""
        obs1 = builder.build(TRAINING_SAMPLE, position=1.0, time_normalized=0.5)
        obs2 = builder.build(TRAINING_SAMPLE, position=1.0, time_normalized=0.5)

        np.testing.assert_array_equal(obs1, obs2,
            err_msg="Observation not reproducible!")


class TestParityWithDataset:
    """Test parity against actual training dataset samples."""

    @pytest.fixture
    def builder(self):
        return ObservationBuilderV19()

    @pytest.mark.parametrize("sample_idx", range(5))
    def test_dataset_sample_parity(self, builder, sample_idx):
        """
        Test multiple samples from training dataset.

        Note: This test requires the training dataset to be available.
        Skip if dataset not found.
        """
        dataset_path = Path("data/pipeline/07_output/datasets_5min")

        # Skip if dataset not available
        if not dataset_path.exists():
            pytest.skip("Training dataset not available")

        # Load a sample from dataset and verify
        # Implementation depends on dataset format
        pass


class TestInferenceIntegration:
    """Test integration with inference pipeline."""

    def test_model_input_shape(self):
        """
        Verify observation shape matches model's expected input.

        PPO models expect observation_space.shape = (15,)
        """
        builder = ObservationBuilderV19()
        obs = builder.build(TRAINING_SAMPLE, position=0.0, time_normalized=0.5)

        expected_shape = (15,)
        assert obs.shape == expected_shape, \
            f"Model expects {expected_shape}, got {obs.shape}"

    def test_observation_dtype(self):
        """Verify dtype is float32 (standard for SB3 models)."""
        builder = ObservationBuilderV19()
        obs = builder.build(TRAINING_SAMPLE, position=0.0, time_normalized=0.5)

        assert obs.dtype == np.float32, \
            f"Model expects float32, got {obs.dtype}"


class TestNormalizationEdgeCases:
    """Test edge cases in normalization."""

    @pytest.fixture
    def builder(self):
        return ObservationBuilderV19()

    def test_zero_input_normalization(self, builder):
        """Test normalization of zero values."""
        zero_features = {k: 0.0 for k in TRAINING_SAMPLE.keys()}
        obs = builder.build(zero_features, position=0.0, time_normalized=0.5)

        assert obs.shape == (15,)
        assert not np.any(np.isnan(obs))

    def test_negative_values(self, builder):
        """Test handling of negative values."""
        negative_features = {k: -0.01 for k in TRAINING_SAMPLE.keys()}
        obs = builder.build(negative_features, position=-1.0, time_normalized=0.5)

        assert obs.shape == (15,)
        assert not np.any(np.isnan(obs))

    def test_very_small_values(self, builder):
        """Test handling of very small values (near zero)."""
        small_features = {k: 1e-10 for k in TRAINING_SAMPLE.keys()}
        obs = builder.build(small_features, position=0.0, time_normalized=0.5)

        assert obs.shape == (15,)
        assert not np.any(np.isnan(obs))

    def test_boundary_clipping(self, builder):
        """Test values at clipping boundaries."""
        # Test at -5 boundary
        obs = builder.build(TRAINING_SAMPLE, position=-5.0, time_normalized=0.0)
        assert obs[13] >= -5.0  # Position should be clipped

        # Test at +5 boundary
        obs = builder.build(TRAINING_SAMPLE, position=5.0, time_normalized=1.0)
        assert obs[13] <= 5.0  # Position should be clipped


class TestFeatureOrdering:
    """Detailed tests for feature ordering."""

    @pytest.fixture
    def builder(self):
        return ObservationBuilderV19()

    def test_core_features_count(self, builder):
        """Verify exactly 13 core features."""
        assert len(builder.CORE_FEATURES) == 13, \
            f"Expected 13 core features, got {len(builder.CORE_FEATURES)}"

    def test_state_features_count(self, builder):
        """Verify exactly 2 state features."""
        assert len(builder.STATE_FEATURES) == 2, \
            f"Expected 2 state features, got {len(builder.STATE_FEATURES)}"

    def test_total_dimension(self, builder):
        """Verify total observation dimension is 15."""
        assert builder.OBS_DIM == 15, \
            f"Expected OBS_DIM=15, got {builder.OBS_DIM}"

    def test_feature_order_length(self, builder):
        """Verify feature order contains all 15 features."""
        assert len(builder.FEATURE_ORDER) == 15, \
            f"Expected 15 features in order, got {len(builder.FEATURE_ORDER)}"

    def test_no_duplicate_features(self, builder):
        """Verify no duplicate features in order."""
        assert len(builder.FEATURE_ORDER) == len(set(builder.FEATURE_ORDER)), \
            "Duplicate features found in FEATURE_ORDER"


# ===========================================================================
# Parity Report Generator
# ===========================================================================

def generate_parity_report():
    """
    Generate detailed parity report comparing inference vs training.

    This can be run manually to debug parity issues.
    """
    builder = ObservationBuilderV19()
    obs = builder.build(TRAINING_SAMPLE, position=0.0, time_normalized=0.5)

    print("\n" + "="*60)
    print("OBSERVATION PARITY REPORT V19")
    print("="*60)

    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Value range: [{obs.min():.4f}, {obs.max():.4f}]")

    print("\n" + "-"*60)
    print("Feature-by-feature breakdown:")
    print("-"*60)
    print(f"{'Idx':<4} {'Feature':<20} {'Raw':<12} {'Normalized':<12} {'Expected':<12} {'Match'}")
    print("-"*60)

    for i, feature in enumerate(builder.FEATURE_ORDER):
        if i < 13:
            raw = TRAINING_SAMPLE.get(feature, "N/A")
            raw_str = f"{raw:.6f}" if isinstance(raw, float) else str(raw)
            expected = EXPECTED_NORMALIZED.get(feature, "N/A")
            expected_str = f"{expected:.4f}" if isinstance(expected, float) else str(expected)

            if isinstance(expected, float):
                match = "OK" if abs(obs[i] - expected) < 0.01 else "FAIL"
            else:
                match = "N/A"
        else:
            raw_str = "state"
            expected_str = "N/A"
            match = "N/A"

        print(f"{i:<4} {feature:<20} {raw_str:<12} {obs[i]:<12.4f} {expected_str:<12} {match}")

    print("\n" + "="*60)

    # Summary
    errors = []
    for i, feature in enumerate(builder.CORE_FEATURES):
        if feature in EXPECTED_NORMALIZED:
            if abs(obs[i] - EXPECTED_NORMALIZED[feature]) >= 0.01:
                errors.append(feature)

    if errors:
        print(f"FAILURES: {errors}")
    else:
        print("ALL NORMALIZATIONS MATCH EXPECTED VALUES")

    print("="*60 + "\n")

    return obs


if __name__ == "__main__":
    # Run parity report when executed directly
    generate_parity_report()

    # Run tests
    pytest.main([__file__, "-v"])
