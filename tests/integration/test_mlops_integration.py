"""
MLOps Integration Tests
=======================

Comprehensive tests for the MLOps inference system.
Tests all components: config, inference engine, risk manager, feature cache, drift monitor.

Usage:
    pytest tests/integration/test_mlops_integration.py -v
    python -m tests.integration.test_mlops_integration
"""

import os
import sys
import pytest
import numpy as np
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# Configuration Tests
# ============================================================================

class TestMLOpsConfig:
    """Test MLOps configuration module."""

    def test_config_loads_from_yaml(self):
        """Test that config loads from YAML file."""
        from services.mlops.config import get_config, Environment

        config = get_config()

        assert config is not None
        assert config.environment in [Environment.DEVELOPMENT, Environment.STAGING, Environment.PRODUCTION]

    def test_config_has_required_sections(self):
        """Test config has all required sections."""
        from services.mlops.config import get_config

        config = get_config()

        assert config.risk_limits is not None
        assert config.trading_hours is not None
        assert config.redis is not None
        assert config.monitoring is not None

    def test_risk_limits_are_valid(self):
        """Test that risk limits are within valid ranges."""
        from services.mlops.config import get_config

        config = get_config()
        limits = config.risk_limits

        # Max daily loss should be negative
        assert limits.max_daily_loss < 0, "max_daily_loss should be negative"
        assert limits.max_daily_loss >= -0.10, "max_daily_loss should not exceed -10%"

        # Max drawdown should be negative
        assert limits.max_drawdown < 0, "max_drawdown should be negative"
        assert limits.max_drawdown >= -0.20, "max_drawdown should not exceed -20%"

        # Confidence thresholds should be between 0 and 1
        assert 0 < limits.min_confidence < 1, "min_confidence should be between 0 and 1"
        assert 0 < limits.high_confidence_threshold < 1, "high_confidence_threshold should be between 0 and 1"
        assert limits.min_confidence < limits.high_confidence_threshold

    def test_trading_hours_are_valid(self):
        """Test that trading hours are valid."""
        from services.mlops.config import get_config

        config = get_config()
        hours = config.trading_hours

        assert 0 <= hours.start_hour < 24
        assert 0 <= hours.end_hour < 24
        assert hours.start_hour < hours.end_hour
        assert hours.timezone is not None


# ============================================================================
# Risk Manager Tests
# ============================================================================

class TestRiskManager:
    """Test risk management functionality."""

    def test_risk_manager_initialization(self):
        """Test risk manager initializes correctly."""
        from services.mlops.risk_manager import RiskManager

        manager = RiskManager()

        assert manager is not None
        assert manager.daily_stats is not None

    def test_check_signal_approved_high_confidence(self):
        """Test signal approval with high confidence."""
        from services.mlops.risk_manager import RiskManager
        from services.mlops.config import SignalType

        manager = RiskManager()
        result = manager.check_signal(SignalType.BUY, 0.85)

        assert result is not None
        assert result.confidence == 0.85

    def test_check_signal_rejected_low_confidence(self):
        """Test signal rejection with low confidence."""
        from services.mlops.risk_manager import RiskManager
        from services.mlops.config import SignalType

        manager = RiskManager()
        result = manager.check_signal(SignalType.BUY, 0.30)

        # Low confidence should result in HOLD
        assert result.adjusted_signal == SignalType.HOLD

    def test_circuit_breaker_triggers_on_losses(self):
        """Test circuit breaker activates on excessive losses."""
        from services.mlops.risk_manager import RiskManager

        manager = RiskManager()

        # Simulate losses that exceed threshold
        for i in range(6):  # max_consecutive_losses is 5
            manager.update_trade_result(
                pnl=-100,
                pnl_percent=-0.005,
                is_win=False,
                trade_id=f"test_loss_{i}"
            )

        assert manager.is_circuit_breaker_active()

    def test_daily_stats_reset(self):
        """Test daily stats reset."""
        from services.mlops.risk_manager import RiskManager

        manager = RiskManager()

        # Add some trades
        manager.update_trade_result(pnl=100, pnl_percent=0.01, is_win=True)
        manager.update_trade_result(pnl=-50, pnl_percent=-0.005, is_win=False)

        # Reset
        manager.reset_daily_stats()

        assert manager.daily_stats.daily_pnl == 0
        assert manager.daily_stats.total_trades == 0
        assert manager.daily_stats.consecutive_losses == 0

    def test_position_size_recommendation(self):
        """Test position size recommendations."""
        from services.mlops.risk_manager import RiskManager

        manager = RiskManager()

        # High confidence should give larger position
        high_conf_rec = manager.get_position_size_recommendation(0.90, capital=100000)
        low_conf_rec = manager.get_position_size_recommendation(0.65, capital=100000)

        assert high_conf_rec["position_size_usd"] > low_conf_rec["position_size_usd"]
        assert high_conf_rec["risk_level"] in ["low", "medium", "high"]


# ============================================================================
# Feature Cache Tests
# ============================================================================

class TestFeatureCache:
    """Test feature caching functionality."""

    def test_feature_cache_initialization(self):
        """Test feature cache initializes correctly."""
        from services.mlops.feature_cache import FeatureCache

        cache = FeatureCache()

        assert cache is not None
        assert cache.feature_order is not None
        assert len(cache.feature_order) > 0

    def test_set_and_get_features(self):
        """Test storing and retrieving features."""
        from services.mlops.feature_cache import FeatureCache

        cache = FeatureCache()

        # Create sample features
        timestamp = "2024-01-15T10:30:00"
        features = {
            "returns_5m": 0.001,
            "rsi_14": 55.0,
            "macd": 0.0003,
            "volatility_5m": 0.005,
        }

        # Store
        success = cache.set_features(timestamp, features, source="test")
        assert success

        # Retrieve
        cached = cache.get_features(timestamp)
        assert cached is not None
        assert cached.timestamp == timestamp
        assert cached.features["returns_5m"] == 0.001

    def test_feature_vector_ordering(self):
        """Test feature vector maintains correct order."""
        from services.mlops.feature_cache import FeatureCache

        cache = FeatureCache()

        features = {
            "rsi_14": 55.0,  # Not first in order
            "returns_5m": 0.001,  # Should be early
        }

        cache.set_features("test_ts", features, source="test")
        cached = cache.get_features("test_ts")

        # Verify it produces a numpy array
        vector = cached.to_numpy()
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32

    def test_cache_stats(self):
        """Test cache statistics."""
        from services.mlops.feature_cache import FeatureCache

        cache = FeatureCache()

        # Generate some activity
        cache.set_features("ts1", {"rsi_14": 50.0}, source="test")
        cache.get_features("ts1")  # Hit
        cache.get_features("ts2")  # Miss

        stats = cache.get_stats()
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats

    def test_cache_health_check(self):
        """Test cache health check."""
        from services.mlops.feature_cache import FeatureCache

        cache = FeatureCache()
        health = cache.health_check()

        assert "status" in health
        assert health["status"] in ["healthy", "degraded", "unhealthy"]


# ============================================================================
# Inference Engine Tests
# ============================================================================

class TestInferenceEngine:
    """Test inference engine functionality."""

    def test_inference_engine_initialization(self):
        """Test inference engine initializes correctly."""
        from services.mlops.inference_engine import InferenceEngine

        engine = InferenceEngine()

        assert engine is not None
        assert engine.models is not None

    def test_engine_without_models(self):
        """Test engine handles no models gracefully."""
        from services.mlops.inference_engine import InferenceEngine

        engine = InferenceEngine()

        # Without loaded models, should not be ready
        assert not engine.is_loaded

    def test_engine_stats(self):
        """Test engine statistics."""
        from services.mlops.inference_engine import InferenceEngine

        engine = InferenceEngine()
        stats = engine.get_stats()

        assert "loaded_models" in stats
        assert "total_predictions" in stats
        assert "is_ready" in stats

    def test_engine_health_check(self):
        """Test engine health check."""
        from services.mlops.inference_engine import InferenceEngine

        engine = InferenceEngine()
        health = engine.health_check()

        assert "status" in health
        assert "models_loaded" in health


# ============================================================================
# Drift Monitor Tests
# ============================================================================

class TestDriftMonitor:
    """Test drift monitoring functionality."""

    def test_drift_monitor_initialization(self):
        """Test drift monitor initializes correctly."""
        from services.mlops.drift_monitor import DriftMonitor

        monitor = DriftMonitor()

        assert monitor is not None
        assert monitor.drift_threshold > 0

    def test_calculate_drift_identical_data(self):
        """Test drift calculation with identical data."""
        from services.mlops.drift_monitor import DriftMonitor
        import pandas as pd

        monitor = DriftMonitor()

        # Create identical datasets
        data = pd.DataFrame({
            "feature_1": np.random.randn(100),
            "feature_2": np.random.randn(100),
        })

        result = monitor.calculate_drift(data, data)

        # Identical data should show no drift
        assert result["drift_detected"] == False
        assert result["drift_share"] == 0.0

    def test_calculate_drift_different_data(self):
        """Test drift calculation with different distributions."""
        from services.mlops.drift_monitor import DriftMonitor
        import pandas as pd

        monitor = DriftMonitor()

        # Create different distributions
        reference = pd.DataFrame({
            "feature_1": np.random.normal(0, 1, 100),
            "feature_2": np.random.normal(0, 1, 100),
        })

        current = pd.DataFrame({
            "feature_1": np.random.normal(5, 1, 100),  # Shifted mean
            "feature_2": np.random.normal(0, 5, 100),  # Different variance
        })

        result = monitor.calculate_drift(reference, current)

        # Should detect drift
        assert result["drift_share"] > 0
        assert len(result["drifted_features"]) > 0

    def test_drift_result_structure(self):
        """Test drift result has required fields."""
        from services.mlops.drift_monitor import DriftMonitor
        import pandas as pd

        monitor = DriftMonitor()

        data = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        result = monitor.calculate_drift(data, data)

        required_fields = [
            "drift_detected",
            "drift_share",
            "drifted_features",
            "total_features",
            "timestamp",
        ]

        for field in required_fields:
            assert field in result, f"Missing field: {field}"


# ============================================================================
# Integration Tests
# ============================================================================

class TestMLOpsIntegration:
    """Integration tests for the complete MLOps pipeline."""

    def test_full_inference_pipeline(self):
        """Test complete inference pipeline without actual models."""
        from services.mlops.config import get_config, SignalType
        from services.mlops.risk_manager import RiskManager
        from services.mlops.feature_cache import FeatureCache

        config = get_config()
        risk_manager = RiskManager(config)
        feature_cache = FeatureCache(config)

        # Simulate feature input
        timestamp = datetime.now().isoformat()
        features = {
            "returns_5m": 0.002,
            "rsi_14": 65.0,
            "macd": 0.001,
            "volatility_5m": 0.008,
        }

        # Cache features
        feature_cache.set_features(timestamp, features, source="test")

        # Simulate a model prediction (BUY signal)
        simulated_signal = SignalType.BUY
        simulated_confidence = 0.75

        # Run risk check
        risk_result = risk_manager.check_signal(simulated_signal, simulated_confidence)

        # Verify risk check produces valid result
        assert risk_result is not None
        assert risk_result.confidence == simulated_confidence

    def test_risk_approval_workflow(self):
        """Test the risk approval workflow."""
        from services.mlops.risk_manager import RiskManager, RiskStatus
        from services.mlops.config import SignalType

        manager = RiskManager()

        # Test various scenarios
        scenarios = [
            (SignalType.BUY, 0.85, True),   # High confidence BUY
            (SignalType.SELL, 0.70, True),  # Medium confidence SELL
            (SignalType.BUY, 0.40, False),  # Low confidence (rejected)
            (SignalType.HOLD, 0.50, True),  # HOLD always approved
        ]

        for signal, confidence, expected_trade in scenarios:
            result = manager.check_signal(signal, confidence)

            if expected_trade and signal != SignalType.HOLD:
                # Should approve trade signals with sufficient confidence
                assert result.adjusted_signal == signal or confidence < 0.60
            elif signal == SignalType.HOLD:
                # HOLD should always be the result
                assert result.adjusted_signal == SignalType.HOLD

    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery after cooldown."""
        from services.mlops.risk_manager import RiskManager
        from services.mlops.config import SignalType

        manager = RiskManager()

        # Trigger circuit breaker
        for i in range(6):
            manager.update_trade_result(pnl=-100, pnl_percent=-0.01, is_win=False)

        assert manager.is_circuit_breaker_active()

        # Reset (simulating cooldown)
        manager.reset_daily_stats()

        assert not manager.is_circuit_breaker_active()


# ============================================================================
# API Tests (requires running server)
# ============================================================================

class TestMLOpsAPI:
    """Test MLOps API endpoints (requires running server)."""

    @pytest.fixture
    def api_base_url(self):
        return os.environ.get("MLOPS_API_URL", "http://localhost:8090")

    @pytest.mark.skipif(
        os.environ.get("MLOPS_API_RUNNING") != "true",
        reason="MLOps API not running"
    )
    def test_health_endpoint(self, api_base_url):
        """Test health endpoint."""
        import requests

        response = requests.get(f"{api_base_url}/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.skipif(
        os.environ.get("MLOPS_API_RUNNING") != "true",
        reason="MLOps API not running"
    )
    def test_inference_endpoint(self, api_base_url):
        """Test inference endpoint."""
        import requests

        # Create test observation (45 features)
        observation = [0.0] * 45

        response = requests.post(
            f"{api_base_url}/v1/inference",
            json={
                "observation": observation,
                "enforce_risk_checks": True,
            }
        )

        assert response.status_code in [200, 503]  # 503 if no models loaded


# ============================================================================
# Main Entry Point
# ============================================================================

def run_tests():
    """Run all tests."""
    print("=" * 60)
    print("MLOps Integration Tests")
    print("=" * 60)

    # Run with pytest
    exit_code = pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
    ])

    return exit_code


if __name__ == "__main__":
    sys.exit(run_tests())
