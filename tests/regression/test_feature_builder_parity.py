"""
Feature Builder Parity Regression Tests
=========================================
P0-4: Verify SSOT parity across Training, Inference, and Backtest contexts.

These tests ensure that:
1. CanonicalFeatureBuilder produces identical results in all contexts
2. Feature order matches the contract exactly (15 features)
3. Normalization statistics are applied consistently
4. Hash verification prevents norm_stats drift
5. Observations never contain NaN or Inf

Contract: CTR-PARITY-001
Author: Trading Team
Created: 2025-01-16

Run with:
    pytest tests/regression/test_feature_builder_parity.py -v
    pytest tests/regression/test_feature_builder_parity.py::TestFeatureBuilderParity -v
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest

# Import from feature_store
from src.feature_store import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
)
from src.feature_store.builders import (
    CanonicalFeatureBuilder,
    BuilderContext,
    NormStatsNotFoundError,
    ObservationDimensionError,
    FeatureCalculationError,
    IFeatureBuilder,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_norm_stats() -> Dict[str, Dict[str, float]]:
    """Sample normalization statistics matching production config."""
    return {
        "log_ret_5m": {"mean": 9.04e-07, "std": 0.001134},
        "log_ret_1h": {"mean": 1.24e-05, "std": 0.003736},
        "log_ret_4h": {"mean": 5.74e-05, "std": 0.007675},
        "rsi_9": {"mean": 48.55, "std": 23.92},
        "atr_pct": {"mean": 0.0608, "std": 0.0452},
        "adx_14": {"mean": 32.30, "std": 17.05},
        "dxy_z": {"mean": 0.0247, "std": 0.999},
        "dxy_change_1d": {"mean": 4.46e-05, "std": 0.0100},
        "vix_z": {"mean": -0.0141, "std": 0.901},
        "embi_z": {"mean": 0.00149, "std": 1.002},
        "brent_change_1d": {"mean": 0.00242, "std": 0.0458},
        "rate_spread": {"mean": -0.0148, "std": 0.998},
        "usdmxn_change_1d": {"mean": -7.59e-05, "std": 0.0184},
    }


@pytest.fixture
def norm_stats_file(sample_norm_stats, tmp_path) -> Path:
    """Create temporary norm_stats file."""
    file_path = tmp_path / "norm_stats.json"
    with open(file_path, "w") as f:
        json.dump(sample_norm_stats, f)
    return file_path


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # Deterministic
    n_bars = 100

    # Generate realistic USDCOP prices
    base_price = 4100.0
    returns = np.random.normal(0, 0.001, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    # Create OHLCV DataFrame
    df = pd.DataFrame({
        "open": prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        "high": prices * (1 + np.random.uniform(0, 0.002, n_bars)),
        "low": prices * (1 - np.random.uniform(0, 0.002, n_bars)),
        "close": prices,
        "volume": np.random.uniform(1000, 10000, n_bars),
    })

    # Ensure high >= low >= close logic
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


@pytest.fixture
def sample_macro() -> pd.DataFrame:
    """Create sample macro data for testing."""
    np.random.seed(42)
    n_bars = 100

    return pd.DataFrame({
        "dxy": np.random.uniform(100, 105, n_bars),
        "vix": np.random.uniform(15, 25, n_bars),
        "embi": np.random.uniform(280, 350, n_bars),
        "brent": np.random.uniform(70, 90, n_bars),
        "usdmxn": np.random.uniform(16, 18, n_bars),
        "treasury_10y": np.random.uniform(3.5, 4.5, n_bars),
    })


@pytest.fixture
def training_builder(norm_stats_file) -> CanonicalFeatureBuilder:
    """Create builder for training context."""
    return CanonicalFeatureBuilder.for_training(
        norm_stats_path=str(norm_stats_file)
    )


@pytest.fixture
def inference_builder(norm_stats_file, training_builder) -> CanonicalFeatureBuilder:
    """Create builder for inference context with expected hash."""
    return CanonicalFeatureBuilder.for_inference(
        norm_stats_path=str(norm_stats_file),
        expected_hash=training_builder.get_norm_stats_hash(),
    )


@pytest.fixture
def backtest_builder(norm_stats_file, training_builder) -> CanonicalFeatureBuilder:
    """Create builder for backtest context with expected hash."""
    return CanonicalFeatureBuilder.for_backtest(
        norm_stats_path=str(norm_stats_file),
        expected_hash=training_builder.get_norm_stats_hash(),
    )


# =============================================================================
# TEST CLASS: FEATURE BUILDER PARITY
# =============================================================================

class TestFeatureBuilderParity:
    """
    Test suite for verifying SSOT parity across all contexts.

    CRITICAL: These tests must pass before any deployment.
    """

    def test_feature_order_matches_contract(self, training_builder):
        """Verify feature order matches SSOT contract exactly."""
        builder_order = training_builder.get_feature_order()

        assert len(builder_order) == 15, f"Expected 15 features, got {len(builder_order)}"
        assert tuple(builder_order) == FEATURE_ORDER, (
            f"Feature order mismatch:\nBuilder: {builder_order}\nContract: {FEATURE_ORDER}"
        )

    def test_observation_dim_matches_contract(self, training_builder):
        """Verify observation dimension matches contract."""
        assert training_builder.get_observation_dim() == OBSERVATION_DIM
        assert training_builder.get_observation_dim() == 15

    def test_training_inference_parity(
        self,
        training_builder,
        inference_builder,
        sample_ohlcv,
        sample_macro
    ):
        """
        CRITICAL: Training and inference must produce IDENTICAL observations.

        This is the most important parity test. Any difference here means
        the model will behave differently in production vs training.
        """
        bar_idx = 50  # After warmup
        position = 0.5
        time_normalized = 0.6

        # Build observations in both contexts
        obs_training = training_builder.build_observation(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=position,
            bar_idx=bar_idx,
            time_normalized=time_normalized,
        )

        obs_inference = inference_builder.build_observation(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=position,
            bar_idx=bar_idx,
            time_normalized=time_normalized,
        )

        # Must be exactly equal
        np.testing.assert_array_almost_equal(
            obs_training,
            obs_inference,
            decimal=6,
            err_msg="Training and inference observations differ!"
        )

    def test_training_backtest_parity(
        self,
        training_builder,
        backtest_builder,
        sample_ohlcv,
        sample_macro
    ):
        """
        CRITICAL: Training and backtest must produce IDENTICAL observations.

        Backtest validation depends on exact parity with training.
        """
        bar_idx = 50
        position = -0.3
        time_normalized = 0.4

        obs_training = training_builder.build_observation(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=position,
            bar_idx=bar_idx,
            time_normalized=time_normalized,
        )

        obs_backtest = backtest_builder.build_observation(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=position,
            bar_idx=bar_idx,
            time_normalized=time_normalized,
        )

        np.testing.assert_array_almost_equal(
            obs_training,
            obs_backtest,
            decimal=6,
            err_msg="Training and backtest observations differ!"
        )

    def test_all_contexts_parity(
        self,
        training_builder,
        inference_builder,
        backtest_builder,
        sample_ohlcv,
        sample_macro
    ):
        """Verify all three contexts produce identical results."""
        test_cases = [
            {"bar_idx": 20, "position": 0.0, "time_normalized": 0.2},
            {"bar_idx": 50, "position": 1.0, "time_normalized": 0.5},
            {"bar_idx": 80, "position": -1.0, "time_normalized": 0.8},
        ]

        for case in test_cases:
            obs_train = training_builder.build_observation(
                sample_ohlcv, sample_macro, **case
            )
            obs_infer = inference_builder.build_observation(
                sample_ohlcv, sample_macro, **case
            )
            obs_back = backtest_builder.build_observation(
                sample_ohlcv, sample_macro, **case
            )

            np.testing.assert_array_almost_equal(
                obs_train, obs_infer, decimal=6,
                err_msg=f"Training vs Inference mismatch at {case}"
            )
            np.testing.assert_array_almost_equal(
                obs_train, obs_back, decimal=6,
                err_msg=f"Training vs Backtest mismatch at {case}"
            )

    def test_norm_stats_hash_consistency(
        self,
        training_builder,
        inference_builder,
        backtest_builder
    ):
        """Verify all contexts use same norm_stats (via hash)."""
        training_hash = training_builder.get_norm_stats_hash()
        inference_hash = inference_builder.get_norm_stats_hash()
        backtest_hash = backtest_builder.get_norm_stats_hash()

        assert training_hash == inference_hash, "Training/Inference hash mismatch"
        assert training_hash == backtest_hash, "Training/Backtest hash mismatch"


# =============================================================================
# TEST CLASS: OBSERVATION INVARIANTS
# =============================================================================

class TestObservationInvariants:
    """Test that observation invariants are always maintained."""

    def test_observation_shape_is_15(self, training_builder, sample_ohlcv, sample_macro):
        """Observation shape must always be (15,)."""
        obs = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=50
        )

        assert obs.shape == (15,), f"Expected shape (15,), got {obs.shape}"

    def test_observation_dtype_is_float32(self, training_builder, sample_ohlcv, sample_macro):
        """Observation dtype must be float32."""
        obs = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=50
        )

        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_observation_no_nan(self, training_builder, sample_ohlcv, sample_macro):
        """Observation must never contain NaN."""
        obs = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=50
        )

        assert not np.isnan(obs).any(), f"Observation contains NaN: {obs}"

    def test_observation_no_inf(self, training_builder, sample_ohlcv, sample_macro):
        """Observation must never contain Inf."""
        obs = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=50
        )

        assert not np.isinf(obs).any(), f"Observation contains Inf: {obs}"

    def test_normalized_features_in_clip_range(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """Normalized features (first 13) must be in [-5, 5]."""
        obs = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=50
        )

        # First 13 features are normalized
        normalized = obs[:13]

        assert (normalized >= -5.0).all(), f"Features below -5.0: {normalized}"
        assert (normalized <= 5.0).all(), f"Features above 5.0: {normalized}"

    def test_position_in_range(self, training_builder, sample_ohlcv, sample_macro):
        """Position (index 13) must be in [-1, 1]."""
        for pos in [-1.5, -1.0, 0.0, 1.0, 1.5]:
            obs = training_builder.build_observation(
                sample_ohlcv, sample_macro,
                position=pos,
                bar_idx=50,
                time_normalized=0.5
            )

            # Position should be clipped to [-1, 1]
            assert -1.0 <= obs[13] <= 1.0, f"Position {obs[13]} out of range for input {pos}"

    def test_time_normalized_in_range(self, training_builder, sample_ohlcv, sample_macro):
        """Time normalized (index 14) must be in [0, 1]."""
        for progress in [-0.1, 0.0, 0.5, 1.0, 1.1]:
            obs = training_builder.build_observation(
                sample_ohlcv, sample_macro,
                position=0.0,
                bar_idx=50,
                time_normalized=progress
            )

            # Should be clipped to [0, 1]
            assert 0.0 <= obs[14] <= 1.0, f"Time {obs[14]} out of range for input {progress}"


# =============================================================================
# TEST CLASS: DETERMINISM
# =============================================================================

class TestDeterminism:
    """Test that feature calculation is deterministic."""

    def test_same_input_same_output(self, training_builder, sample_ohlcv, sample_macro):
        """Same input must always produce same output."""
        params = {
            "ohlcv": sample_ohlcv,
            "macro": sample_macro,
            "position": 0.5,
            "bar_idx": 50,
            "time_normalized": 0.5,
        }

        # Build multiple times
        obs1 = training_builder.build_observation(**params)
        obs2 = training_builder.build_observation(**params)
        obs3 = training_builder.build_observation(**params)

        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(obs2, obs3)

    def test_different_bar_idx_different_output(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """Different bar indices should produce different outputs."""
        obs_50 = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=50
        )
        obs_60 = training_builder.build_observation(
            sample_ohlcv, sample_macro, position=0.0, bar_idx=60
        )

        # At least technical features should differ
        assert not np.allclose(obs_50[:6], obs_60[:6]), \
            "Technical features should differ for different bars"


# =============================================================================
# TEST CLASS: ERROR HANDLING
# =============================================================================

class TestErrorHandling:
    """Test error handling and fail-fast behavior."""

    def test_norm_stats_not_found_raises(self):
        """Missing norm_stats file should raise NormStatsNotFoundError."""
        with pytest.raises(NormStatsNotFoundError):
            CanonicalFeatureBuilder.for_training(
                norm_stats_path="/nonexistent/path/norm_stats.json"
            )

    def test_warmup_violation_raises(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """Bar index below warmup should raise ValueError."""
        with pytest.raises(ValueError, match="warmup_bars"):
            training_builder.build_observation(
                sample_ohlcv, sample_macro,
                position=0.0,
                bar_idx=5  # Below warmup (14)
            )

    def test_empty_norm_stats_raises(self, tmp_path):
        """Empty norm_stats should raise ValueError."""
        empty_file = tmp_path / "empty_norm_stats.json"
        with open(empty_file, "w") as f:
            json.dump({}, f)

        with pytest.raises(ValueError, match="cannot be empty"):
            CanonicalFeatureBuilder.for_training(norm_stats_path=str(empty_file))

    def test_missing_features_raises(self, tmp_path):
        """Missing required features should raise ValueError."""
        incomplete_stats = {
            "log_ret_5m": {"mean": 0.0, "std": 1.0},
            # Missing other features
        }

        file_path = tmp_path / "incomplete_stats.json"
        with open(file_path, "w") as f:
            json.dump(incomplete_stats, f)

        with pytest.raises(ValueError, match="missing required features"):
            CanonicalFeatureBuilder.for_training(norm_stats_path=str(file_path))


# =============================================================================
# TEST CLASS: EXPORT AND AUDIT
# =============================================================================

class TestExportAndAudit:
    """Test export and audit functionality."""

    def test_export_snapshot_structure(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """Verify export_snapshot returns complete snapshot."""
        snapshot = training_builder.export_snapshot(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=0.5,
            bar_idx=50,
            time_normalized=0.6,
        )

        # Verify structure
        assert hasattr(snapshot, "timestamp")
        assert hasattr(snapshot, "bar_idx")
        assert hasattr(snapshot, "context")
        assert hasattr(snapshot, "raw_features")
        assert hasattr(snapshot, "normalized_features")
        assert hasattr(snapshot, "observation")
        assert hasattr(snapshot, "norm_stats_hash")
        assert hasattr(snapshot, "calculation_time_ms")

        # Verify values
        assert snapshot.bar_idx == 50
        assert snapshot.context == "training"
        assert len(snapshot.observation) == 15
        assert len(snapshot.raw_features) == 13  # Market features
        assert len(snapshot.normalized_features) == 15  # All features

    def test_to_json_contract(self, training_builder):
        """Verify to_json_contract returns valid contract."""
        contract = training_builder.to_json_contract()

        assert contract["version"] == "1.0.0"
        assert contract["observation_dim"] == 15
        assert len(contract["feature_order"]) == 15
        assert "norm_stats_hash" in contract
        assert "warmup_bars" in contract
        assert "technical_periods" in contract


# =============================================================================
# TEST CLASS: INTERFACE COMPLIANCE
# =============================================================================

class TestInterfaceCompliance:
    """Test that CanonicalFeatureBuilder implements IFeatureBuilder."""

    def test_implements_build_observation(self, training_builder):
        """Must implement build_observation method."""
        assert hasattr(training_builder, "build_observation")
        assert callable(training_builder.build_observation)

    def test_implements_get_feature_order(self, training_builder):
        """Must implement get_feature_order method."""
        assert hasattr(training_builder, "get_feature_order")
        assert callable(training_builder.get_feature_order)

        order = training_builder.get_feature_order()
        assert isinstance(order, list)
        assert len(order) == 15

    def test_implements_get_observation_dim(self, training_builder):
        """Must implement get_observation_dim method."""
        assert hasattr(training_builder, "get_observation_dim")
        assert callable(training_builder.get_observation_dim)

        dim = training_builder.get_observation_dim()
        assert isinstance(dim, int)
        assert dim == 15

    def test_implements_get_norm_stats_hash(self, training_builder):
        """Must implement get_norm_stats_hash method."""
        assert hasattr(training_builder, "get_norm_stats_hash")
        assert callable(training_builder.get_norm_stats_hash)

        hash_val = training_builder.get_norm_stats_hash()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA-256


# =============================================================================
# TEST CLASS: BATCH PROCESSING
# =============================================================================

class TestBatchProcessing:
    """Test batch feature building for training/backtest."""

    def test_build_batch_returns_dataframe(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """build_batch should return DataFrame."""
        result = training_builder.build_batch(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            normalize=True
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_ohlcv)

    def test_build_batch_contains_features(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """build_batch result should contain all features."""
        result = training_builder.build_batch(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            normalize=True
        )

        # Check for key features
        expected_features = [
            "log_ret_5m", "log_ret_1h", "rsi_9", "atr_pct", "adx_14"
        ]
        for feat in expected_features:
            assert feat in result.columns, f"Missing feature: {feat}"

    def test_build_batch_normalized_in_range(
        self, training_builder, sample_ohlcv, sample_macro
    ):
        """Normalized batch features should be clipped."""
        result = training_builder.build_batch(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            normalize=True
        )

        for feat in ["log_ret_5m", "rsi_9", "atr_pct"]:
            if feat in result.columns:
                valid_rows = result[feat].dropna()
                assert (valid_rows >= -5.0).all(), f"{feat} has values < -5"
                assert (valid_rows <= 5.0).all(), f"{feat} has values > 5"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
