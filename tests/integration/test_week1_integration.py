"""
Week 1 Integration Tests
========================

Comprehensive integration tests for Week 1 deliverables:
- TradingFlags: Environment-controlled trading flags
- FeatureReader: Feature store reading and validation
- UnifiedBacktestEngine: Backtesting with slippage
- SmokeTest: Pre-deployment model validation

Author: USD/COP Trading System
Version: 1.0.0
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Week 1 imports - TradingFlags from SSOT (src.config.trading_flags)
from src.config.trading_flags import (
    TradingFlags,
    get_trading_flags,
    reload_trading_flags,
    reset_trading_flags_cache,
    TradingMode,
)

# Backward compatibility aliases for tests
def load_trading_flags(force_reload: bool = False) -> TradingFlags:
    """Load trading flags with optional force reload."""
    if force_reload:
        return reload_trading_flags()
    return get_trading_flags()

def reload_flags() -> TradingFlags:
    """Force reload trading flags."""
    return reload_trading_flags()

def clear_flags_cache() -> None:
    """Clear the trading flags cache."""
    reset_trading_flags_cache()

def validate_for_production():
    """Validate flags for production."""
    flags = get_trading_flags()
    return flags.validate_for_production()

def get_current_flags() -> TradingFlags:
    """Get current trading flags."""
    return get_trading_flags()

class TradingDisabledError(Exception):
    """Raised when trading is attempted while disabled."""
    pass

from src.features.feature_reader import (
    FeatureReader,
    FeatureReadResult,
    FeatureValidationError,
    EXPECTED_FEATURE_ORDER,
)

from src.validation.backtest_engine import (
    SignalType,
    BacktestConfig,
    BacktestMetrics,
    BacktestResult,
    UnifiedBacktestEngine,
    create_backtest_engine,
)

from src.validation.smoke_test import (
    ValidationStatus,
    ValidationCheck,
    SmokeTestConfig,
    SmokeTestResult,
    SmokeTest,
    run_smoke_test,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for backtesting (100 bars, 5min freq)."""
    np.random.seed(42)
    n_bars = 100

    # Generate realistic price movement
    base_price = 4200.0
    returns = np.random.normal(0.0001, 0.002, n_bars)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    dates = pd.date_range(start='2025-01-01', periods=n_bars, freq='5min')
    high_noise = np.abs(np.random.normal(0.001, 0.0005, n_bars))
    low_noise = np.abs(np.random.normal(0.001, 0.0005, n_bars))

    data = pd.DataFrame({
        'open': prices * (1 - np.random.uniform(0, 0.001, n_bars)),
        'high': prices * (1 + high_noise),
        'low': prices * (1 - low_noise),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_bars),
    }, index=dates)

    return data


@pytest.fixture
def sample_features():
    """Generate sample feature dictionary."""
    return {
        "log_ret_5m": 0.001,
        "log_ret_1h": 0.002,
        "log_ret_4h": -0.001,
        "rsi_9": 55.0,
        "atr_pct": 0.05,
        "adx_14": 25.0,
        "dxy_z": 0.5,
        "dxy_change_1d": 0.002,
        "vix_z": -0.3,
        "embi_z": 0.1,
        "brent_change_1d": 0.01,
        "rate_spread": 0.2,
        "usdmxn_change_1d": 0.005,
    }


@pytest.fixture
def mock_model():
    """Create a mock model for smoke testing."""
    class MockModel:
        def __init__(self, output_value=0.5, deterministic=True):
            self.output_value = output_value
            self.deterministic = deterministic
            self.call_count = 0

        def predict(self, x):
            self.call_count += 1
            if self.deterministic:
                return np.array([self.output_value])
            else:
                return np.array([self.output_value + np.random.normal(0, 0.1)])

    return MockModel()


@pytest.fixture
def invalid_mock_model():
    """Create a mock model that fails for smoke testing."""
    class InvalidModel:
        def predict(self, x):
            return None  # Invalid output

    return InvalidModel()


@pytest.fixture
def sample_input():
    """Generate sample model input."""
    return np.random.randn(1, 15).astype(np.float32)


# ===========================================================================
# TestTradingFlags
# ===========================================================================

class TestTradingFlags:
    """Tests for TradingFlags environment-controlled flags."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_flags_cache()

    def test_flags_from_env(self, monkeypatch):
        """Test loading flags from environment variables."""
        # Set environment variables
        monkeypatch.setenv('TRADING_ENABLED', 'true')
        monkeypatch.setenv('KILL_SWITCH', 'false')
        monkeypatch.setenv('PAPER_MODE', 'true')
        monkeypatch.setenv('MAX_POSITION_SIZE', '0.5')
        monkeypatch.setenv('MAX_DAILY_LOSS_PCT', '3.0')

        # Clear cache and reload
        clear_flags_cache()
        flags = load_trading_flags()

        assert flags.trading_enabled is True
        assert flags.kill_switch is False
        assert flags.paper_mode is True
        assert flags.max_position_size == 0.5
        assert flags.max_daily_loss_pct == 3.0
        assert flags.source == "environment"

    def test_kill_switch_blocks_trading(self, monkeypatch):
        """Test that kill switch stops all trading."""
        monkeypatch.setenv('KILL_SWITCH', 'true')
        monkeypatch.setenv('TRADING_ENABLED', 'true')

        clear_flags_cache()
        flags = load_trading_flags()

        # Kill switch should block trading
        assert flags.kill_switch is True
        assert flags.is_trading_allowed() is False

        # can_execute_trade should return False with reason
        can_trade, reason = flags.can_execute_trade()
        assert can_trade is False
        assert "kill switch" in reason.lower()

    def test_production_validation(self, monkeypatch):
        """Test validate_for_production returns errors when appropriate."""
        # Set problematic flags
        monkeypatch.setenv('TRADING_ENABLED', 'false')
        monkeypatch.setenv('KILL_SWITCH', 'true')
        monkeypatch.setenv('PAPER_MODE', 'true')
        monkeypatch.setenv('MAX_POSITION_SIZE', '0')

        clear_flags_cache()
        is_valid, errors = validate_for_production()

        # Should fail validation
        assert is_valid is False
        assert len(errors) > 0

        # Check for expected errors
        error_text = ' '.join(errors).lower()
        assert 'kill switch' in error_text or 'trading' in error_text

    def test_paper_trading_mode(self, monkeypatch):
        """Test paper mode prevents live execution."""
        monkeypatch.setenv('PAPER_MODE', 'true')
        monkeypatch.setenv('TRADING_ENABLED', 'true')
        monkeypatch.setenv('KILL_SWITCH', 'false')

        clear_flags_cache()
        flags = load_trading_flags()

        assert flags.is_paper_mode() is True
        assert flags.is_live_mode() is False

        # Trading should still be allowed (just in paper mode)
        assert flags.is_trading_allowed() is True

    def test_reload_flags(self, monkeypatch):
        """Test cache clear and reload works."""
        # First load with paper mode
        monkeypatch.setenv('PAPER_MODE', 'true')
        clear_flags_cache()
        flags1 = load_trading_flags()
        assert flags1.paper_mode is True

        # Change environment and reload
        monkeypatch.setenv('PAPER_MODE', 'false')
        flags2 = reload_flags()

        assert flags2.paper_mode is False
        # Verify reload happened (loaded_at should be >= first load)
        assert flags2.loaded_at >= flags1.loaded_at
        # Most importantly, the value changed
        assert flags1.paper_mode != flags2.paper_mode

    def test_default_values(self):
        """Test default values when no environment set."""
        clear_flags_cache()
        flags = TradingFlags()  # Direct instantiation with defaults

        assert flags.trading_enabled is True
        assert flags.kill_switch is False
        assert flags.paper_mode is True
        assert flags.max_position_size == 1.0
        assert flags.max_daily_loss_pct == 5.0


# ===========================================================================
# TestFeatureReader
# ===========================================================================

class TestFeatureReader:
    """Tests for FeatureReader feature store reading."""

    @pytest.fixture
    def reader(self):
        """Create FeatureReader instance."""
        return FeatureReader(max_age_seconds=300.0)

    def test_get_latest_features_returns_none_when_empty(self, reader):
        """Test that empty store returns invalid result."""
        result = reader.get_latest_features()

        assert result.is_valid is False
        assert result.features is None
        assert result.error is not None
        assert "no features" in result.error.lower()

    def test_feature_order_validation(self, reader, sample_features):
        """Test feature order validation works correctly."""
        # Validate correct order
        is_valid, diffs = reader.validate_feature_order(EXPECTED_FEATURE_ORDER)
        assert is_valid is True
        assert len(diffs) == 0

        # Validate wrong order
        wrong_order = EXPECTED_FEATURE_ORDER.copy()
        wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]
        is_valid, diffs = reader.validate_feature_order(wrong_order)
        assert is_valid is False
        assert len(diffs) > 0

        # Validate missing feature
        missing = EXPECTED_FEATURE_ORDER[:-1]  # Remove last
        is_valid, diffs = reader.validate_feature_order(missing)
        assert is_valid is False
        assert any('missing' in d.lower() for d in diffs)

    def test_max_age_validation(self, reader):
        """Test max age validation works correctly."""
        now = datetime.now()

        # Fresh features should pass
        fresh_time = now - timedelta(seconds=100)
        is_valid, age = reader.check_max_age(fresh_time, now)
        assert is_valid is True
        assert age < 300

        # Stale features should fail
        stale_time = now - timedelta(seconds=400)
        is_valid, age = reader.check_max_age(stale_time, now)
        assert is_valid is False
        assert age > 300

    def test_get_latest_features_with_data(self, reader, sample_features):
        """Test reading features when data is available."""
        # Add features to internal store
        reader.add_features(sample_features, datetime.now())

        result = reader.get_latest_features()

        assert result.is_valid is True
        assert result.features is not None
        assert result.feature_vector is not None
        assert result.feature_vector.shape == (len(EXPECTED_FEATURE_ORDER),)
        assert len(result.missing_features) == 0

    def test_missing_features_detected(self, reader):
        """Test that missing features are properly detected."""
        # Add only some features
        partial_features = {
            "log_ret_5m": 0.001,
            "log_ret_1h": 0.002,
        }
        reader.add_features(partial_features, datetime.now())

        result = reader.get_latest_features()

        assert result.is_valid is False
        assert len(result.missing_features) > 0
        assert "rsi_9" in result.missing_features

    def test_stale_features_flagged(self, reader, sample_features):
        """Test that stale features are properly flagged."""
        # Add old features
        old_time = datetime.now() - timedelta(seconds=600)
        reader.add_features(sample_features, old_time)

        result = reader.get_latest_features()

        assert result.is_valid is True  # Still valid, just stale
        assert result.is_stale is True
        assert result.age_seconds > 300

    def test_clear_features(self, reader, sample_features):
        """Test clearing features works."""
        reader.add_features(sample_features, datetime.now())
        reader.clear_features()

        result = reader.get_latest_features()
        assert result.is_valid is False


# ===========================================================================
# TestUnifiedBacktestEngine
# ===========================================================================

class TestUnifiedBacktestEngine:
    """Tests for UnifiedBacktestEngine backtesting."""

    @pytest.fixture
    def engine(self):
        """Create backtest engine with default config."""
        return create_backtest_engine(
            initial_capital=10000.0,
            slippage_pct=0.001,
            commission_pct=0.0005,
        )

    def test_backtest_config_defaults(self):
        """Test BacktestConfig has sensible defaults."""
        config = BacktestConfig()

        assert config.initial_capital == 10000.0
        assert config.slippage_pct == 0.001
        assert config.commission_pct == 0.0005
        assert config.max_drawdown_pct == 20.0
        assert config.prevent_lookahead is True

    def test_metrics_calculation(self, engine, sample_ohlcv_data):
        """Test that metrics are calculated correctly."""
        # Simple always-long strategy for predictable metrics
        def always_long(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 50:
                return SignalType.CLOSE
            return SignalType.HOLD

        result = engine.run(sample_ohlcv_data, always_long)

        # Should have metrics
        assert result.metrics is not None
        assert isinstance(result.metrics.total_return_pct, float)
        assert isinstance(result.metrics.sharpe_ratio, float)
        assert isinstance(result.metrics.max_drawdown_pct, float)

        # Should have trades
        assert len(result.trades) >= 1  # At least one trade

        # Should have equity curve
        assert len(result.equity_curve) > 0

    def test_no_lookahead_bias(self, sample_ohlcv_data):
        """Test that lookahead bias is prevented."""
        engine = create_backtest_engine(prevent_lookahead=True)

        data_lengths = []

        def record_data_length(data, bar):
            data_lengths.append(len(data))
            return SignalType.HOLD

        engine.run(sample_ohlcv_data, record_data_length)

        # With lookahead prevention, visible data length should increase
        # Starting from warmup_bars (0 by default), each bar should only see up to that bar
        for i, length in enumerate(data_lengths):
            expected_length = i + 1  # bar 0 sees 1 row, bar 1 sees 2 rows, etc
            assert length == expected_length, \
                f"At bar {i}, expected {expected_length} rows but saw {length}"

    def test_slippage_applied(self, sample_ohlcv_data):
        """Test that slippage is applied to trades."""
        # Create engine with significant slippage
        engine = create_backtest_engine(
            slippage_pct=0.01,  # 1% slippage
            commission_pct=0.0,  # No commission for clear test
        )

        def quick_trade(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 11:
                return SignalType.CLOSE
            return SignalType.HOLD

        result = engine.run(sample_ohlcv_data, quick_trade)

        # Should have exactly one trade
        assert len(result.trades) == 1

        trade = result.trades[0]
        # Slippage should be recorded
        assert trade.slippage_cost > 0

        # Entry price should be worse than close price (for long)
        actual_close_at_entry = sample_ohlcv_data['close'].iloc[10]
        assert trade.entry_price > actual_close_at_entry

    def test_commission_applied(self, sample_ohlcv_data):
        """Test that commission is applied to trades."""
        engine = create_backtest_engine(
            slippage_pct=0.0,  # No slippage for clear test
            commission_pct=0.01,  # 1% commission
        )

        def quick_trade(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 11:
                return SignalType.CLOSE
            return SignalType.HOLD

        result = engine.run(sample_ohlcv_data, quick_trade)

        assert len(result.trades) == 1
        trade = result.trades[0]
        assert trade.commission_cost > 0

    def test_long_and_short_trades(self, sample_ohlcv_data):
        """Test both long and short trades work."""
        engine = create_backtest_engine()

        def alternating_strategy(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 20:
                return SignalType.CLOSE
            elif bar == 30:
                return SignalType.SHORT
            elif bar == 40:
                return SignalType.CLOSE
            return SignalType.HOLD

        result = engine.run(sample_ohlcv_data, alternating_strategy)

        assert len(result.trades) == 2
        assert result.trades[0].direction == SignalType.LONG
        assert result.trades[1].direction == SignalType.SHORT

    def test_position_tracking(self, sample_ohlcv_data):
        """Test position is tracked correctly."""
        engine = create_backtest_engine()

        def long_strategy(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 50:
                return SignalType.CLOSE
            return SignalType.HOLD

        result = engine.run(sample_ohlcv_data, long_strategy)

        # Positions should be tracked
        assert len(result.positions) > 0

        # Should have 0s before entry, 1s during position, 0s after exit
        assert result.positions[9] == 0  # Before entry
        assert result.positions[11] == 1  # After entry
        assert result.positions[51] == 0  # After exit


# ===========================================================================
# TestSmokeTest
# ===========================================================================

class TestSmokeTest:
    """Tests for SmokeTest pre-deployment validation."""

    def test_smoke_test_fails_for_invalid_model(self, invalid_mock_model, sample_input):
        """Test smoke test fails when model returns invalid output."""
        config = SmokeTestConfig(
            check_model_loadable=True,
            check_model_output_valid=True,
            check_inference_latency=False,  # Skip for speed
        )
        smoke_test = SmokeTest(config)

        result = smoke_test.run(invalid_mock_model, sample_input)

        assert result.passed is False
        assert result.failed_checks > 0
        assert any('output' in check.name.lower() for check in result.checks
                   if check.status == ValidationStatus.FAILED)

    def test_smoke_test_reports_duration(self, mock_model, sample_input):
        """Test smoke test reports test duration."""
        config = SmokeTestConfig(
            check_model_loadable=True,
            check_model_output_valid=True,
            check_inference_latency=True,
            inference_test_runs=5,
        )
        smoke_test = SmokeTest(config)

        result = smoke_test.run(mock_model, sample_input)

        # Should have timing info
        assert result.total_duration_ms > 0
        assert result.start_time is not None
        assert result.end_time is not None

        # Individual checks should have duration
        for check in result.checks:
            assert check.duration_ms >= 0

    def test_smoke_test_checks_all_validations(self, mock_model, sample_input):
        """Test smoke test runs all configured validations."""
        config = SmokeTestConfig(
            check_model_loadable=True,
            check_model_output_valid=True,
            check_model_deterministic=True,
            check_inference_latency=True,
            check_output_range=True,
            inference_warmup_runs=2,
            inference_test_runs=3,
        )
        smoke_test = SmokeTest(config)

        result = smoke_test.run(mock_model, sample_input)

        # Should have run multiple checks
        assert len(result.checks) >= 4

        # Check names should be present
        check_names = [c.name for c in result.checks]
        assert 'model_loadable' in check_names
        assert 'model_output_valid' in check_names
        assert 'model_deterministic' in check_names
        assert 'inference_latency' in check_names

    def test_smoke_test_pass_for_valid_model(self, mock_model, sample_input):
        """Test smoke test passes for valid model."""
        result = run_smoke_test(mock_model, sample_input, max_latency_ms=1000.0)

        assert result.passed is True
        assert result.status == "passed"
        assert result.failed_checks == 0

    def test_smoke_test_latency_check(self, sample_input):
        """Test latency check fails for slow model."""
        import time

        class SlowModel:
            def predict(self, x):
                time.sleep(0.2)  # 200ms delay
                return np.array([0.5])

        config = SmokeTestConfig(
            check_model_loadable=True,
            check_model_output_valid=True,
            check_inference_latency=True,
            max_inference_latency_ms=50.0,  # 50ms limit
            inference_warmup_runs=1,
            inference_test_runs=2,
        )
        smoke_test = SmokeTest(config)
        result = smoke_test.run(SlowModel(), sample_input)

        # Should fail latency check
        latency_check = next(
            (c for c in result.checks if c.name == 'inference_latency'),
            None
        )
        assert latency_check is not None
        assert latency_check.status == ValidationStatus.FAILED

    def test_smoke_test_feature_check(self, mock_model, sample_input, sample_features):
        """Test feature availability check."""
        config = SmokeTestConfig(
            check_features_available=True,
            required_features=['log_ret_5m', 'rsi_9'],
        )
        smoke_test = SmokeTest(config)

        # Test with feature provider
        def feature_provider():
            return sample_features

        result = smoke_test.run(mock_model, sample_input, feature_provider)

        # Should pass feature check
        feature_check = next(
            (c for c in result.checks if c.name == 'features_available'),
            None
        )
        assert feature_check is not None
        assert feature_check.status == ValidationStatus.PASSED

    def test_smoke_test_fail_fast(self, invalid_mock_model, sample_input):
        """Test fail_fast stops on first failure."""
        config = SmokeTestConfig(
            check_model_loadable=True,
            check_model_output_valid=True,
            check_inference_latency=True,
            fail_fast=True,
        )
        smoke_test = SmokeTest(config)

        result = smoke_test.run(invalid_mock_model, sample_input)

        # Should have failed
        assert result.passed is False

        # Should not have run all checks due to fail_fast
        # (depends on which check fails first)
        assert result.failed_checks >= 1


# ===========================================================================
# Integration Tests - Cross-Component
# ===========================================================================

class TestWeek1Integration:
    """Integration tests combining multiple Week 1 components."""

    def test_flags_affect_backtest(self, monkeypatch, sample_ohlcv_data):
        """Test that trading flags can influence backtest behavior."""
        monkeypatch.setenv('PAPER_MODE', 'true')
        monkeypatch.setenv('KILL_SWITCH', 'false')

        clear_flags_cache()
        flags = load_trading_flags()

        # In paper mode, we can still backtest
        assert flags.is_paper_mode() is True

        engine = create_backtest_engine()

        def simple_strategy(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 20:
                return SignalType.CLOSE
            return SignalType.HOLD

        # Should be able to run backtest
        result = engine.run(sample_ohlcv_data, simple_strategy)
        assert result is not None
        assert len(result.trades) > 0

    def test_feature_reader_with_smoke_test(self, mock_model, sample_input, sample_features):
        """Test FeatureReader integrates with SmokeTest."""
        reader = FeatureReader()
        reader.add_features(sample_features, datetime.now())

        config = SmokeTestConfig(
            check_features_available=True,
            required_features=EXPECTED_FEATURE_ORDER[:5],
        )
        smoke_test = SmokeTest(config)

        # Feature provider using FeatureReader
        def feature_provider():
            result = reader.get_latest_features()
            return result.features if result.is_valid else {}

        result = smoke_test.run(mock_model, sample_input, feature_provider)

        # Feature check should pass
        feature_check = next(
            (c for c in result.checks if c.name == 'features_available'),
            None
        )
        assert feature_check is not None
        assert feature_check.status == ValidationStatus.PASSED

    def test_end_to_end_validation_workflow(
        self,
        monkeypatch,
        mock_model,
        sample_input,
        sample_features,
        sample_ohlcv_data,
    ):
        """Test complete Week 1 validation workflow."""
        # Step 1: Check trading flags
        monkeypatch.setenv('TRADING_ENABLED', 'true')
        monkeypatch.setenv('KILL_SWITCH', 'false')
        monkeypatch.setenv('PAPER_MODE', 'true')

        clear_flags_cache()
        flags = load_trading_flags()
        assert flags.is_trading_allowed() is True

        # Step 2: Validate features are available
        reader = FeatureReader()
        reader.add_features(sample_features, datetime.now())
        feature_result = reader.get_latest_features()
        assert feature_result.is_valid is True

        # Step 3: Run smoke test on model
        smoke_result = run_smoke_test(mock_model, sample_input)
        assert smoke_result.passed is True

        # Step 4: Run backtest
        engine = create_backtest_engine()

        def strategy(data, bar):
            if bar == 10:
                return SignalType.LONG
            elif bar == 50:
                return SignalType.CLOSE
            return SignalType.HOLD

        backtest_result = engine.run(sample_ohlcv_data, strategy)
        assert backtest_result.metrics is not None

        # Workflow complete - all validations passed
        print(f"Total return: {backtest_result.metrics.total_return_pct:.2f}%")
        print(f"Sharpe ratio: {backtest_result.metrics.sharpe_ratio:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
