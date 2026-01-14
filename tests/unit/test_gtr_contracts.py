"""
Tests para contratos GTR (Gemini).
GEMINI-T1 a T11 | Contratos: GTR-001 a GTR-009

Tests de:
- GTR-001: ONNX Converter
- GTR-002: Circuit Breakers
- GTR-003: Drift Detection
- GTR-004: Risk Engine
- GTR-005: MacroDataSource
- GTR-006: Dataset Registry
- GTR-007: Config Loader
- GTR-008: Equity Filter (TypeScript - covered separately)
- GTR-009: Paper Trading Validator
"""

import pytest
import sys
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


# =============================================================================
# GTR-002: Circuit Breakers Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.risk.circuit_breakers not implemented. Risk checks are in src/risk/checks/")
class TestCircuitBreakers:
    """Tests para Circuit Breakers (GTR-002)."""

    @pytest.fixture
    def config(self):
        from lib.risk.circuit_breakers import CircuitBreakerConfig
        return CircuitBreakerConfig(max_clipped_features=3)

    @pytest.fixture
    def feature_breaker(self, config):
        from lib.risk.circuit_breakers import FeatureCircuitBreaker
        return FeatureCircuitBreaker(config)

    @pytest.fixture
    def master_breaker(self):
        from lib.risk.circuit_breakers import MasterCircuitBreaker
        return MasterCircuitBreaker()

    def test_valid_observation_passes(self, feature_breaker):
        """Observation valida DEBE pasar."""
        obs = np.random.randn(15).astype(np.float32)
        obs = np.clip(obs, -4.0, 4.0)  # Dentro de rango

        is_valid, reason = feature_breaker.check(obs)

        assert is_valid
        assert reason is None

    def test_nan_triggers_trip(self, feature_breaker):
        """NaN en observation DEBE disparar breaker."""
        obs = np.random.randn(15).astype(np.float32)
        obs[5] = np.nan

        is_valid, reason = feature_breaker.check(obs)

        assert not is_valid
        assert "NaN" in reason

    def test_inf_triggers_trip(self, feature_breaker):
        """Inf en observation DEBE disparar breaker."""
        obs = np.random.randn(15).astype(np.float32)
        obs[5] = np.inf

        is_valid, reason = feature_breaker.check(obs)

        assert not is_valid
        assert "Inf" in reason

    def test_too_many_clipped_triggers_trip(self, feature_breaker):
        """Muchas features en limites DEBE disparar breaker."""
        obs = np.ones(15, dtype=np.float32) * 5.0  # Todas en limite

        is_valid, reason = feature_breaker.check(obs)

        assert not is_valid
        assert "clipping" in reason

    def test_wrong_shape_triggers_trip(self, feature_breaker):
        """Shape incorrecto DEBE disparar breaker."""
        obs = np.random.randn(10).astype(np.float32)  # Wrong shape

        is_valid, reason = feature_breaker.check(obs)

        assert not is_valid
        assert "Shape" in reason

    def test_master_initial_state_allows_trading(self, master_breaker):
        """Estado inicial DEBE permitir trading."""
        can_trade, _ = master_breaker.can_trade()
        assert can_trade

    def test_master_breaker_blocks_on_bad_obs(self, master_breaker):
        """Master breaker DEBE bloquear obs con NaN."""
        bad_obs = np.ones(15) * np.nan
        master_breaker.check_observation(bad_obs)

        can_trade, reason = master_breaker.can_trade()

        assert not can_trade
        assert "OPEN" in reason

    def test_callback_called_on_trip(self, master_breaker):
        """Callback DEBE ser llamado cuando breaker se dispara."""
        trip_reasons = []
        master_breaker.register_callback(lambda r: trip_reasons.append(r))

        bad_obs = np.ones(15) * np.nan
        master_breaker.check_observation(bad_obs)

        assert len(trip_reasons) == 1
        assert "Feature" in trip_reasons[0]

    def test_status_returns_all_breakers(self, master_breaker):
        """get_status DEBE retornar estado de todos los breakers."""
        status = master_breaker.get_status()

        assert "can_trade" in status
        assert "feature_breaker" in status
        assert "latency_breaker" in status
        assert "loss_breaker" in status


@pytest.mark.skip(reason="Module lib.risk.circuit_breakers not implemented")
class TestLatencyCircuitBreaker:
    """Tests para LatencyCircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        from lib.risk.circuit_breakers import LatencyCircuitBreaker, CircuitBreakerConfig
        config = CircuitBreakerConfig(
            latency_threshold_ms=50.0,
            latency_window_size=20,
            latency_breach_pct=0.1
        )
        return LatencyCircuitBreaker(config)

    def test_normal_latency_passes(self, breaker):
        """Latencia normal DEBE pasar."""
        for _ in range(20):
            is_healthy, _ = breaker.record_latency(10.0)

        assert is_healthy

    def test_high_latency_triggers_trip(self, breaker):
        """Latencia alta sostenida DEBE disparar breaker."""
        # Fill buffer with normal latencies
        for _ in range(15):
            breaker.record_latency(10.0)

        # Add high latencies (>10% of window)
        for _ in range(5):
            is_healthy, reason = breaker.record_latency(100.0)

        assert not is_healthy
        assert "degradada" in reason


@pytest.mark.skip(reason="Module lib.risk.circuit_breakers not implemented")
class TestLossCircuitBreaker:
    """Tests para LossCircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        from lib.risk.circuit_breakers import LossCircuitBreaker, CircuitBreakerConfig
        config = CircuitBreakerConfig(
            max_consecutive_losses=3,
            max_drawdown_pct=0.1
        )
        return LossCircuitBreaker(config)

    def test_wins_dont_trigger(self, breaker):
        """Ganancias NO deben disparar breaker."""
        equity = 10000
        for _ in range(10):
            equity += 100
            is_healthy, _ = breaker.record_trade(pnl=100, equity=equity)

        assert is_healthy

    def test_consecutive_losses_trigger(self, breaker):
        """Perdidas consecutivas DEBEN disparar breaker."""
        equity = 10000
        for i in range(4):
            equity -= 100
            is_healthy, reason = breaker.record_trade(pnl=-100, equity=equity)

        assert not is_healthy
        assert "consecutivas" in reason

    def test_drawdown_triggers(self, breaker):
        """Drawdown excesivo DEBE disparar breaker."""
        breaker.record_trade(pnl=0, equity=10000)  # Set peak

        is_healthy, reason = breaker.record_trade(pnl=-1500, equity=8500)

        assert not is_healthy
        assert "Drawdown" in reason


# =============================================================================
# GTR-003: Drift Detection Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.risk.drift_detection not implemented")
class TestDriftDetection:
    """Tests para Drift Detection (GTR-003)."""

    @pytest.fixture
    def config(self):
        from lib.risk.drift_detection import DriftConfig
        return DriftConfig(
            reference_window_size=100,
            detection_window_size=20,
            ks_threshold=0.1,
            psi_threshold=0.2
        )

    @pytest.fixture
    def detector_with_reference(self, config):
        from lib.risk.drift_detection import FeatureDriftDetector
        np.random.seed(42)
        reference = np.random.randn(100)
        return FeatureDriftDetector(
            feature_name="test_feature",
            config=config,
            reference_data=reference
        )

    @pytest.fixture
    def monitor(self):
        from lib.risk.drift_detection import DriftMonitor, DriftConfig
        config = DriftConfig(
            reference_window_size=50,
            detection_window_size=10
        )
        return DriftMonitor(config=config)

    def test_no_drift_on_same_distribution(self, detector_with_reference):
        """Misma distribucion NO debe generar alerta."""
        np.random.seed(43)

        alerts = []
        for i in range(30):
            value = np.random.randn()
            alert = detector_with_reference.update(value, bar_idx=i)
            if alert:
                alerts.append(alert)

        assert len(alerts) == 0

    def test_drift_detected_on_mean_shift(self, detector_with_reference):
        """Cambio de media DEBE generar alerta."""
        alerts = []
        for i in range(50):
            value = np.random.randn() + 3.0  # Mean shift
            alert = detector_with_reference.update(value, bar_idx=i)
            if alert:
                alerts.append(alert)

        assert len(alerts) > 0
        assert alerts[0].drift_type == "covariate"

    def test_monitor_processes_all_features(self, monitor):
        """Monitor DEBE procesar todas las 15 features."""
        assert len(monitor.detectors) == 15

    def test_monitor_update_accepts_valid_observation(self, monitor):
        """update() DEBE aceptar observation valida."""
        obs = np.random.randn(15).astype(np.float32)

        alerts = monitor.update(obs)

        assert isinstance(alerts, list)

    def test_monitor_update_rejects_wrong_shape(self, monitor):
        """update() DEBE rechazar shape incorrecto."""
        bad_obs = np.random.randn(10)

        with pytest.raises(ValueError, match="Shape"):
            monitor.update(bad_obs)


# =============================================================================
# GTR-004: Risk Engine Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.risk.engine not implemented")
class TestRiskEngine:
    """Tests para Risk Engine (GTR-004)."""

    @pytest.fixture
    def engine(self):
        from lib.risk.engine import RiskEngine, RiskEngineConfig
        config = RiskEngineConfig(
            model_path="models/test.onnx",  # Will use mock
            min_confidence=0.6
        )
        return RiskEngine(config)

    def test_evaluate_returns_trade_decision(self, engine):
        """evaluate DEBE retornar TradeDecision."""
        from lib.risk.engine import TradeDecision
        obs = np.random.randn(15).astype(np.float32)
        obs = np.clip(obs, -4, 4)

        decision = engine.evaluate(obs)

        assert isinstance(decision, TradeDecision)
        assert decision.action in [0, 1, 2]
        assert 0 <= decision.confidence <= 1

    def test_blocks_on_bad_observation(self, engine):
        """DEBE bloquear observations invalidas."""
        bad_obs = np.ones(15) * np.nan

        decision = engine.evaluate(bad_obs)

        assert not decision.can_execute
        assert "Circuit breaker" in decision.block_reason

    def test_get_status_returns_complete_info(self, engine):
        """get_status DEBE retornar info completa."""
        obs = np.random.randn(15).astype(np.float32)
        obs = np.clip(obs, -4, 4)
        engine.evaluate(obs)

        status = engine.get_status()

        assert "evaluation_count" in status
        assert "blocked_count" in status
        assert "circuit_breaker" in status
        assert "drift_monitor" in status


# =============================================================================
# GTR-006: Dataset Registry Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.data.dataset_registry not implemented")
class TestDatasetRegistry:
    """Tests para Dataset Registry (GTR-006)."""

    @pytest.fixture
    def registry(self):
        from lib.data.dataset_registry import DatasetRegistry
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "test_registry.json"
            yield DatasetRegistry(registry_path=registry_path)

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=100),
            "value": range(100),
            "feature": np.random.randn(100),
        })

    def test_register_creates_metadata(self, registry, sample_df):
        """register DEBE crear metadata con checksums."""
        metadata = registry.register(sample_df, "test_dataset", "1.0")

        assert metadata.checksum_sha256 is not None
        assert len(metadata.checksum_sha256) == 64
        assert metadata.row_count == 100

    def test_verify_passes_for_same_data(self, registry, sample_df):
        """verify DEBE pasar para mismo DataFrame."""
        registry.register(sample_df, "test_dataset", "1.0")

        result = registry.verify(sample_df, "test_dataset", "1.0")

        assert result is True

    def test_verify_fails_for_modified_data(self, registry, sample_df):
        """verify DEBE fallar para datos modificados."""
        registry.register(sample_df, "test_dataset", "1.0")

        modified_df = sample_df.copy()
        modified_df["value"] = modified_df["value"] + 1

        with pytest.raises(ValueError, match="checksum mismatch"):
            registry.verify(modified_df, "test_dataset", "1.0")


# =============================================================================
# GTR-007: Config Loader Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.config.loader not implemented")
class TestConfigLoader:
    """Tests para Config Loader (GTR-007)."""

    def test_load_config_returns_current_config(self):
        """load_config DEBE retornar config actual."""
        from lib.config.loader import load_config

        config = load_config("current")

        assert config.model.version == "current"
        assert config.model.observation_dim == 15

    def test_get_training_params_returns_dict(self):
        """get_training_params DEBE retornar dict con params."""
        from lib.config.loader import get_training_params

        params = get_training_params("current")

        assert "learning_rate" in params
        assert "n_steps" in params
        assert "batch_size" in params

    def test_config_has_all_sections(self):
        """Config DEBE tener todas las secciones."""
        from lib.config.loader import load_config

        config = load_config("current")

        assert config.model is not None
        assert config.training is not None
        assert config.thresholds is not None
        assert config.features is not None
        assert config.trading is not None
        assert config.risk is not None


# =============================================================================
# GTR-009: Paper Trading Validator Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.validation.paper_trading_validator not implemented")
class TestPaperTradingValidator:
    """Tests para Paper Trading Validator (GTR-009)."""

    @pytest.fixture
    def backtest_results(self):
        from lib.validation.paper_trading_validator import create_synthetic_results
        return create_synthetic_results(sharpe=1.5, trades=100)

    @pytest.fixture
    def paper_results_similar(self):
        from lib.validation.paper_trading_validator import create_synthetic_results
        return create_synthetic_results(sharpe=1.4, trades=95)

    @pytest.fixture
    def paper_results_different(self):
        from lib.validation.paper_trading_validator import create_synthetic_results
        return create_synthetic_results(sharpe=0.5, trades=50)

    def test_similar_results_pass(self, backtest_results, paper_results_similar):
        """Resultados similares DEBEN pasar."""
        from lib.validation.paper_trading_validator import PaperTradingValidator

        validator = PaperTradingValidator(backtest_results, paper_results_similar)
        report = validator.validate()

        # Puede tener warnings pero no deberia tener critical failures
        # (dependiendo de la varianza aleatoria)
        assert isinstance(report.results, list)

    def test_different_results_detected(self, backtest_results, paper_results_different):
        """Divergencias significativas DEBEN ser detectadas."""
        from lib.validation.paper_trading_validator import PaperTradingValidator

        validator = PaperTradingValidator(backtest_results, paper_results_different)
        report = validator.validate()

        # Should have either critical failures or warnings
        assert len(report.warnings) > 0 or len(report.critical_failures) > 0

    def test_report_has_all_fields(self, backtest_results, paper_results_similar):
        """Report DEBE tener todos los campos."""
        from lib.validation.paper_trading_validator import PaperTradingValidator

        validator = PaperTradingValidator(backtest_results, paper_results_similar)
        report = validator.validate()

        assert hasattr(report, "start_date")
        assert hasattr(report, "end_date")
        assert hasattr(report, "results")
        assert hasattr(report, "overall_passed")
        assert hasattr(report, "warnings")
        assert hasattr(report, "critical_failures")


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.skip(reason="Module lib.risk.engine not implemented")
class TestContractIntegration:
    """Tests de integracion entre contratos."""

    def test_risk_engine_uses_circuit_breakers(self):
        """Risk Engine DEBE usar Circuit Breakers."""
        from lib.risk.engine import RiskEngine, RiskEngineConfig

        config = RiskEngineConfig(model_path="test.onnx")
        engine = RiskEngine(config)

        assert engine.circuit_breaker is not None
        assert hasattr(engine.circuit_breaker, "check_observation")

    def test_risk_engine_uses_drift_monitor(self):
        """Risk Engine DEBE usar Drift Monitor."""
        from lib.risk.engine import RiskEngine, RiskEngineConfig

        config = RiskEngineConfig(model_path="test.onnx")
        engine = RiskEngine(config)

        assert engine.drift_monitor is not None
        assert hasattr(engine.drift_monitor, "update")
