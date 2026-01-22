"""
Unit Tests for Model Router (Shadow Mode)
=========================================

Tests for the ModelRouter class which executes champion and shadow
models in parallel for A/B testing and gradual model rollout.

MLOps-3: Shadow Mode Testing

Author: Trading Team
Date: 2025-01-14
"""

import time
from pathlib import Path
import pytest
import numpy as np

# Add src to path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from src.inference.model_router import (
    ModelRouter,
    ModelWrapper,
    PredictionResult,
    RouterPrediction,
    create_model_router,
)


# =============================================================================
# Mock Models for Testing
# =============================================================================

class MockChampionModel:
    """Mock champion model that predicts based on input."""

    def __init__(self, bias: float = 0.0):
        self.bias = bias
        self.predict_count = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        self.predict_count += 1
        # Simple prediction: mean of observation + bias
        action = float(np.mean(observation)) + self.bias
        action = np.clip(action, -1.0, 1.0)
        return np.array([action]), None


class MockShadowModel:
    """Mock shadow model with slightly different behavior."""

    def __init__(self, bias: float = 0.1):
        self.bias = bias
        self.predict_count = 0

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        self.predict_count += 1
        # Different prediction logic
        action = float(np.mean(observation)) + self.bias
        action = np.clip(action, -1.0, 1.0)
        return np.array([action]), None


class MockDivergentModel:
    """Mock model that always disagrees with champion."""

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        # Always predict opposite direction
        base = float(np.mean(observation))
        action = -base if base >= 0 else abs(base)
        action = np.clip(action, -1.0, 1.0)
        return np.array([action]), None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_champion():
    """Create mock champion model."""
    return MockChampionModel(bias=0.0)


@pytest.fixture
def mock_shadow():
    """Create mock shadow model."""
    return MockShadowModel(bias=0.05)


@pytest.fixture
def router_with_local_models(mock_champion, mock_shadow):
    """Create router with local mock models."""
    router = ModelRouter(
        mlflow_uri="http://localhost:5000",  # Won't be used
        enable_shadow=True
    )
    router.load_local_models(
        champion_model=mock_champion,
        champion_version="v1.0",
        shadow_model=mock_shadow,
        shadow_version="v1.1"
    )
    return router


@pytest.fixture
def router_champion_only(mock_champion):
    """Create router with only champion model."""
    router = ModelRouter(
        mlflow_uri="http://localhost:5000",
        enable_shadow=False
    )
    router.load_local_models(
        champion_model=mock_champion,
        champion_version="v1.0"
    )
    return router


@pytest.fixture
def sample_observation():
    """Create sample observation array."""
    return np.array([0.1, 0.2, -0.1, 0.0, 0.3])


# =============================================================================
# Test: ModelWrapper
# =============================================================================

class TestModelWrapper:
    """Tests for ModelWrapper class."""

    def test_wrapper_creation(self, mock_champion):
        """Test creating a ModelWrapper."""
        wrapper = ModelWrapper(
            model=mock_champion,
            name="test_model",
            version="v1.0",
            stage="Production"
        )

        assert wrapper.name == "test_model"
        assert wrapper.version == "v1.0"
        assert wrapper.stage == "Production"

    def test_wrapper_predict(self, mock_champion, sample_observation):
        """Test prediction through wrapper."""
        wrapper = ModelWrapper(
            model=mock_champion,
            name="test_model",
            version="v1.0",
            stage="Production"
        )

        action, confidence = wrapper.predict(sample_observation)

        assert isinstance(action, float)
        assert isinstance(confidence, float)
        assert -1.0 <= action <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_wrapper_get_signal_long(self):
        """Test signal conversion for LONG."""
        wrapper = ModelWrapper(
            model=MockChampionModel(),
            name="test",
            version="v1",
            stage="Production",
            threshold_long=0.33,
            threshold_short=-0.33
        )

        signal = wrapper.get_signal(0.5)
        assert signal == "LONG"

    def test_wrapper_get_signal_short(self):
        """Test signal conversion for SHORT."""
        wrapper = ModelWrapper(
            model=MockChampionModel(),
            name="test",
            version="v1",
            stage="Production",
            threshold_long=0.33,
            threshold_short=-0.33
        )

        signal = wrapper.get_signal(-0.5)
        assert signal == "SHORT"

    def test_wrapper_get_signal_hold(self):
        """Test signal conversion for HOLD."""
        wrapper = ModelWrapper(
            model=MockChampionModel(),
            name="test",
            version="v1",
            stage="Production",
            threshold_long=0.33,
            threshold_short=-0.33
        )

        signal = wrapper.get_signal(0.1)
        assert signal == "HOLD"

    def test_wrapper_info(self, mock_champion):
        """Test info property."""
        wrapper = ModelWrapper(
            model=mock_champion,
            name="test_model",
            version="v1.0",
            stage="Production"
        )

        info = wrapper.info

        assert info["name"] == "test_model"
        assert info["version"] == "v1.0"
        assert info["stage"] == "Production"
        assert "loaded_at" in info


# =============================================================================
# Test: ModelRouter Initialization
# =============================================================================

class TestModelRouterInitialization:
    """Tests for ModelRouter initialization."""

    def test_default_initialization(self):
        """Test default initialization (no MLflow connection)."""
        router = ModelRouter(
            mlflow_uri="http://localhost:5000",
            enable_shadow=True
        )

        assert router.mlflow_uri == "http://localhost:5000"
        assert router.enable_shadow is True
        # Champion won't be loaded without MLflow
        assert not router.is_ready

    def test_shadow_disabled(self):
        """Test initialization with shadow disabled."""
        router = ModelRouter(
            mlflow_uri="http://localhost:5000",
            enable_shadow=False
        )

        assert router.enable_shadow is False

    def test_factory_function(self):
        """Test create_model_router factory function."""
        router = create_model_router(
            mlflow_uri="http://localhost:5001",
            enable_shadow=True,
            model_name="test_model"
        )

        assert isinstance(router, ModelRouter)
        assert router.mlflow_uri == "http://localhost:5001"
        assert router.model_name == "test_model"


# =============================================================================
# Test: Loading Local Models
# =============================================================================

class TestLoadingLocalModels:
    """Tests for loading local models."""

    def test_load_champion_only(self, mock_champion):
        """Test loading only champion model."""
        router = ModelRouter(enable_shadow=False)
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1.0"
        )

        assert router.is_ready
        assert router.champion is not None
        assert router.shadow is None

    def test_load_both_models(self, mock_champion, mock_shadow):
        """Test loading both champion and shadow."""
        router = ModelRouter(enable_shadow=True)
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1.0",
            shadow_model=mock_shadow,
            shadow_version="v1.1"
        )

        assert router.is_ready
        assert router.champion is not None
        assert router.shadow is not None
        assert router.champion.version == "v1.0"
        assert router.shadow.version == "v1.1"


# =============================================================================
# Test: Predictions
# =============================================================================

class TestPredictions:
    """Tests for model predictions."""

    def test_predict_champion_only(self, router_champion_only, sample_observation):
        """Test prediction with champion only."""
        result = router_champion_only.predict(sample_observation)

        assert isinstance(result, RouterPrediction)
        assert result.champion is not None
        assert result.shadow is None
        assert result.champion_used is True
        assert result.agree is True  # No shadow to disagree

    def test_predict_with_shadow(self, router_with_local_models, sample_observation):
        """Test prediction with both models."""
        result = router_with_local_models.predict(sample_observation)

        assert result.champion is not None
        assert result.shadow is not None
        assert isinstance(result.divergence, float)
        assert result.champion_used is True

    def test_prediction_result_structure(self, router_with_local_models, sample_observation):
        """Test PredictionResult structure."""
        result = router_with_local_models.predict(sample_observation)
        champion = result.champion

        assert isinstance(champion, PredictionResult)
        assert isinstance(champion.action, float)
        assert isinstance(champion.confidence, float)
        assert champion.signal in ["LONG", "SHORT", "HOLD"]
        assert isinstance(champion.latency_ms, float)
        assert champion.latency_ms >= 0
        assert isinstance(champion.timestamp, str)

    def test_prediction_not_ready(self):
        """Test prediction when router not ready raises error."""
        router = ModelRouter(enable_shadow=False)

        with pytest.raises(RuntimeError, match="Champion model not loaded"):
            router.predict(np.array([0.1, 0.2]))


# =============================================================================
# Test: Agreement Tracking
# =============================================================================

class TestAgreementTracking:
    """Tests for agreement tracking between models."""

    def test_agreement_with_similar_models(self, mock_champion):
        """Test agreement when models produce similar results."""
        router = ModelRouter(enable_shadow=True)

        # Use same model for both (should always agree)
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1.0",
            shadow_model=mock_champion,  # Same model
            shadow_version="v1.0-copy"
        )

        # Make predictions
        for _ in range(10):
            obs = np.random.randn(5)
            result = router.predict(obs)
            assert result.agree is True

        assert router.agreement_rate == 1.0

    def test_disagreement_with_divergent_models(self, mock_champion):
        """Test disagreement tracking with divergent models."""
        router = ModelRouter(enable_shadow=True)

        # Use divergent shadow model
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1.0",
            shadow_model=MockDivergentModel(),
            shadow_version="divergent"
        )

        # Make predictions with varied inputs
        agreements = []
        for _ in range(20):
            obs = np.random.randn(5) * 0.5  # Larger values for more divergence
            result = router.predict(obs)
            agreements.append(result.agree)

        # Should have some disagreements
        agreement_rate = sum(agreements) / len(agreements)
        assert agreement_rate < 1.0, "Expected some disagreements"

    def test_divergence_calculation(self, router_with_local_models, sample_observation):
        """Test divergence calculation."""
        result = router_with_local_models.predict(sample_observation)

        # Divergence should be the absolute difference
        expected_divergence = abs(result.champion.action - result.shadow.action)
        assert abs(result.divergence - expected_divergence) < 0.001

    def test_recent_agreement_rate(self, router_with_local_models):
        """Test recent agreement rate calculation."""
        # Make many predictions to fill the window
        for _ in range(50):
            obs = np.random.randn(5)
            router_with_local_models.predict(obs)

        recent_rate = router_with_local_models.recent_agreement_rate
        assert 0.0 <= recent_rate <= 1.0


# =============================================================================
# Test: Status and Statistics
# =============================================================================

class TestStatusAndStatistics:
    """Tests for status and statistics methods."""

    def test_get_status(self, router_with_local_models, sample_observation):
        """Test get_status method."""
        # Make some predictions first
        for _ in range(10):
            router_with_local_models.predict(sample_observation)

        status = router_with_local_models.get_status()

        assert isinstance(status, dict)
        assert status["ready"] is True
        assert status["champion"] is not None
        assert status["shadow"] is not None
        assert status["shadow_enabled"] is True
        assert "statistics" in status
        assert status["statistics"]["total_predictions"] == 10

    def test_is_ready_property(self, router_with_local_models, router_champion_only):
        """Test is_ready property."""
        assert router_with_local_models.is_ready is True
        assert router_champion_only.is_ready is True

        empty_router = ModelRouter(enable_shadow=False)
        assert empty_router.is_ready is False


# =============================================================================
# Test: Concurrent Execution
# =============================================================================

class TestConcurrentExecution:
    """Tests for concurrent model execution."""

    def test_parallel_execution(self, mock_champion, mock_shadow):
        """Test that models are executed in parallel."""
        # Create models that track call order
        call_times = {"champion": None, "shadow": None}

        class TimedChampion:
            def predict(self, obs, deterministic=True):
                call_times["champion"] = time.time()
                time.sleep(0.01)  # 10ms delay
                return np.array([0.5]), None

        class TimedShadow:
            def predict(self, obs, deterministic=True):
                call_times["shadow"] = time.time()
                time.sleep(0.01)  # 10ms delay
                return np.array([0.4]), None

        router = ModelRouter(enable_shadow=True)
        router.load_local_models(
            champion_model=TimedChampion(),
            champion_version="v1",
            shadow_model=TimedShadow(),
            shadow_version="v1"
        )

        start = time.time()
        result = router.predict(np.array([0.1]))
        elapsed = time.time() - start

        # If parallel, should take ~10-15ms (one delay + overhead)
        # If sequential, would take ~20ms+
        assert elapsed < 0.025, f"Expected parallel execution, took {elapsed*1000:.1f}ms"

    def test_both_models_called(self, mock_champion, mock_shadow):
        """Test that both models are called during prediction."""
        router = ModelRouter(enable_shadow=True)
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1",
            shadow_model=mock_shadow,
            shadow_version="v1"
        )

        initial_champion_count = mock_champion.predict_count
        initial_shadow_count = mock_shadow.predict_count

        router.predict(np.array([0.1, 0.2]))

        assert mock_champion.predict_count == initial_champion_count + 1
        assert mock_shadow.predict_count == initial_shadow_count + 1


# =============================================================================
# Test: Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""

    def test_shadow_failure_continues(self, mock_champion):
        """Test that champion continues if shadow fails."""
        class FailingShadow:
            def predict(self, obs, deterministic=True):
                raise RuntimeError("Shadow model error")

        router = ModelRouter(enable_shadow=True)
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1",
            shadow_model=FailingShadow(),
            shadow_version="fail"
        )

        # Should still get champion result
        result = router.predict(np.array([0.1, 0.2]))

        assert result.champion is not None
        # Shadow might be None or have error

    def test_shutdown(self, router_with_local_models):
        """Test router shutdown."""
        # Should not raise
        router_with_local_models.shutdown()


# =============================================================================
# Test: Reload
# =============================================================================

class TestReload:
    """Tests for model reloading."""

    def test_reload_local_models(self, mock_champion, mock_shadow):
        """Test reloading with new local models."""
        router = ModelRouter(enable_shadow=True)
        router.load_local_models(
            champion_model=mock_champion,
            champion_version="v1.0",
            shadow_model=mock_shadow,
            shadow_version="v1.0"
        )

        assert router.champion.version == "v1.0"

        # Load new models
        new_champion = MockChampionModel(bias=0.1)
        new_shadow = MockShadowModel(bias=0.2)

        router.load_local_models(
            champion_model=new_champion,
            champion_version="v2.0",
            shadow_model=new_shadow,
            shadow_version="v2.0"
        )

        assert router.champion.version == "v2.0"
        assert router.shadow.version == "v2.0"


# =============================================================================
# Test: Integration
# =============================================================================

class TestIntegration:
    """Integration tests for ModelRouter."""

    def test_full_workflow(self):
        """Test complete shadow mode workflow."""
        # 1. Create router
        router = ModelRouter(enable_shadow=True)

        # 2. Load models
        champion = MockChampionModel(bias=0.0)
        shadow = MockShadowModel(bias=0.1)

        router.load_local_models(
            champion_model=champion,
            champion_version="v1.0",
            shadow_model=shadow,
            shadow_version="v1.1"
        )

        assert router.is_ready

        # 3. Make predictions
        predictions = []
        for _ in range(100):
            obs = np.random.randn(10)
            result = router.predict(obs)
            predictions.append(result)

        # 4. Check statistics
        status = router.get_status()
        assert status["statistics"]["total_predictions"] == 100

        # 5. Check agreement tracking
        agreement_rate = router.agreement_rate
        assert 0.0 <= agreement_rate <= 1.0

        # 6. Verify all predictions have required fields
        for pred in predictions:
            assert pred.champion is not None
            assert pred.shadow is not None
            assert pred.champion_used is True

        # 7. Shutdown
        router.shutdown()


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
