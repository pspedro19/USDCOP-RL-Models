"""
V19 Integration Tests
=====================

Tests full integration of V19 components:
- ObservationBuilderV19
- StateTracker
- RiskManager
- PaperTrader
- ModelMonitor

Author: Pedro @ Lean Tech Solutions
Version: 19.0.0
"""

import pytest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.builders.observation_builder_v19 import ObservationBuilderV19
from src.core.state.state_tracker import StateTracker, ModelState
from src.risk.risk_manager import RiskManager, RiskLimits
from src.trading.paper_trader import PaperTrader
from src.monitoring.model_monitor import ModelMonitor


# ===========================================================================
# Test Data
# ===========================================================================

SAMPLE_MARKET_FEATURES = {
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
    "usdmxn_change_1d": 0.005
}


class TestV19Components:
    """Test all V19 components exist and have expected interface."""

    def test_observation_builder_exists(self):
        """Test ObservationBuilderV19 instantiation and OBS_DIM."""
        builder = ObservationBuilderV19()
        assert builder.OBS_DIM == 15
        assert len(builder.CORE_FEATURES) == 13
        assert len(builder.STATE_FEATURES) == 2

    def test_state_tracker_exists(self):
        """Test StateTracker instantiation and interface."""
        tracker = StateTracker()
        assert hasattr(tracker, 'get_state_features')
        assert hasattr(tracker, 'update_position')
        assert hasattr(tracker, 'get_or_create')

    def test_risk_manager_exists(self):
        """Test RiskManager instantiation and interface."""
        rm = RiskManager()
        assert hasattr(rm, 'validate_signal')
        assert hasattr(rm, 'record_trade_result')
        assert hasattr(rm, 'get_status')

    def test_paper_trader_exists(self):
        """Test PaperTrader instantiation and interface."""
        pt = PaperTrader()
        assert hasattr(pt, 'execute_signal')
        assert hasattr(pt, 'get_statistics')
        assert hasattr(pt, 'get_equity_curve')

    def test_model_monitor_exists(self):
        """Test ModelMonitor instantiation and interface."""
        mm = ModelMonitor()
        assert hasattr(mm, 'get_health_status')
        assert hasattr(mm, 'record_action')
        assert hasattr(mm, 'check_action_drift')


class TestObservationBuilderV19:
    """Detailed tests for ObservationBuilderV19."""

    @pytest.fixture
    def builder(self):
        return ObservationBuilderV19()

    def test_build_returns_correct_shape(self, builder):
        """Test observation shape is (15,)."""
        obs = builder.build(SAMPLE_MARKET_FEATURES, position=0.0, time_normalized=0.5)
        assert obs.shape == (15,)

    def test_build_returns_float32(self, builder):
        """Test observation dtype is float32."""
        obs = builder.build(SAMPLE_MARKET_FEATURES, position=0.0, time_normalized=0.5)
        assert obs.dtype == np.float32

    def test_feature_order_is_correct(self, builder):
        """Test feature order matches V19 specification."""
        expected = [
            "log_ret_5m", "log_ret_1h", "log_ret_4h",
            "rsi_9", "atr_pct", "adx_14",
            "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
            "brent_change_1d", "rate_spread", "usdmxn_change_1d",
            "position", "time_normalized"
        ]
        assert builder.FEATURE_ORDER == expected

    def test_normalization_applied(self, builder):
        """Test that normalization is applied to features."""
        obs = builder.build(SAMPLE_MARKET_FEATURES, position=0.0, time_normalized=0.5)

        # Raw rsi_9 = 55.0, mean = 48.55, std = 23.92
        # Expected z-score approx (55 - 48.55) / 23.92 = 0.27
        assert abs(obs[3] - 0.27) < 0.1, f"RSI normalization unexpected: {obs[3]}"

    def test_clipping_applied(self, builder):
        """Test that values are clipped to [-5, 5]."""
        extreme = {k: 1e6 for k in SAMPLE_MARKET_FEATURES}
        obs = builder.build(extreme, position=0.0, time_normalized=0.5)

        assert obs.min() >= -5.0
        assert obs.max() <= 5.0


class TestStateTracker:
    """Detailed tests for StateTracker."""

    @pytest.fixture
    def tracker(self):
        return StateTracker(initial_equity=10000.0)

    def test_get_state_features_new_model(self, tracker):
        """Test state features for new model (no position)."""
        position, time_norm = tracker.get_state_features("test_model", current_bar=30, total_bars=60)

        assert position == 0.0, "New model should have no position"
        assert time_norm == pytest.approx(0.483, rel=0.01), f"Time norm should be ~0.483, got {time_norm}"

    def test_update_position_long(self, tracker):
        """Test updating position to long."""
        state = tracker.update_position("test_model", new_position=1.0, current_price=4200.0)

        assert state.position == 1.0
        assert state.entry_price == 4200.0
        assert state.trade_count_session == 1

    def test_update_position_short(self, tracker):
        """Test updating position to short."""
        state = tracker.update_position("test_model", new_position=-1.0, current_price=4200.0)

        assert state.position == -1.0
        assert state.entry_price == 4200.0

    def test_update_position_flat(self, tracker):
        """Test closing position to flat."""
        # First go long
        tracker.update_position("test_model", new_position=1.0, current_price=4200.0)

        # Then close
        state = tracker.update_position("test_model", new_position=0.0, current_price=4250.0)

        assert state.position == 0.0
        assert state.entry_price == 0.0
        assert state.trade_count_session == 2  # Open + close

    def test_model_state_dataclass(self):
        """Test ModelState dataclass initialization."""
        state = ModelState(model_id="test")

        assert state.model_id == "test"
        assert state.position == 0.0
        assert state.current_equity == 10000.0


class TestRiskManager:
    """Detailed tests for RiskManager."""

    @pytest.fixture
    def risk_manager(self):
        limits = RiskLimits(
            max_drawdown_pct=15.0,
            max_daily_loss_pct=5.0,
            max_trades_per_day=20
        )
        return RiskManager(limits)

    def test_validate_signal_allowed(self, risk_manager):
        """Test signal validation when allowed."""
        allowed, reason = risk_manager.validate_signal("LONG", current_drawdown_pct=5.0)

        assert allowed is True
        assert "allowed" in reason.lower()

    def test_validate_signal_kill_switch(self, risk_manager):
        """Test kill switch triggers on high drawdown."""
        allowed, reason = risk_manager.validate_signal("LONG", current_drawdown_pct=20.0)

        assert allowed is False
        assert "kill switch" in reason.lower()

    def test_validate_signal_close_always_allowed(self, risk_manager):
        """Test CLOSE signals are always allowed."""
        # Trigger kill switch first
        risk_manager.validate_signal("LONG", current_drawdown_pct=20.0)

        # Close should still be allowed
        allowed, reason = risk_manager.validate_signal("CLOSE", current_drawdown_pct=20.0)
        assert allowed is True

    def test_record_trade_result(self, risk_manager):
        """Test recording trade results."""
        risk_manager.record_trade_result(pnl_pct=0.5, signal="LONG")

        status = risk_manager.get_status()
        assert status["trade_count_today"] == 1
        assert status["daily_pnl_pct"] > 0


class TestPaperTrader:
    """Detailed tests for PaperTrader."""

    @pytest.fixture
    def trader(self):
        return PaperTrader(initial_capital=10000.0)

    def test_execute_long_signal(self, trader):
        """Test executing a LONG signal."""
        trade = trader.execute_signal("model_1", "LONG", current_price=4200.0)

        assert trade is not None
        assert trade.direction == "LONG"
        assert trade.entry_price == 4200.0
        assert trade.status == "open"

    def test_execute_short_signal(self, trader):
        """Test executing a SHORT signal."""
        trade = trader.execute_signal("model_1", "SHORT", current_price=4200.0)

        assert trade is not None
        assert trade.direction == "SHORT"

    def test_execute_close_signal(self, trader):
        """Test executing a CLOSE signal."""
        # First open a position
        trader.execute_signal("model_1", "LONG", current_price=4200.0)

        # Then close
        trade = trader.execute_signal("model_1", "CLOSE", current_price=4250.0)

        assert trade is not None
        assert trade.status == "closed"
        assert trade.pnl > 0  # Price went up, long should profit

    def test_hold_signal_no_action(self, trader):
        """Test HOLD signal does nothing."""
        trade = trader.execute_signal("model_1", "HOLD", current_price=4200.0)

        assert trade is None

    def test_get_statistics(self, trader):
        """Test getting trading statistics."""
        stats = trader.get_statistics()

        assert "total_trades" in stats
        assert "win_rate" in stats
        assert "total_pnl" in stats


class TestModelMonitor:
    """Detailed tests for ModelMonitor."""

    @pytest.fixture
    def monitor(self):
        return ModelMonitor(window_size=100)

    def test_record_action(self, monitor):
        """Test recording actions."""
        for i in range(10):
            monitor.record_action(float(i) / 10)

        assert len(monitor.action_history) == 10

    def test_record_pnl(self, monitor):
        """Test recording PnL values."""
        for i in range(10):
            monitor.record_pnl(float(i))

        assert len(monitor.pnl_history) == 10

    def test_get_health_status_healthy(self, monitor):
        """Test health status when healthy."""
        # Record some varied actions
        for i in range(50):
            monitor.record_action(np.sin(i / 10))

        health = monitor.get_health_status()

        assert health["status"] == "healthy"
        assert "stuck_behavior" in health
        assert "rolling_sharpe" in health

    def test_check_stuck_behavior(self, monitor):
        """Test stuck behavior detection."""
        # Record same action repeatedly
        for _ in range(50):
            monitor.record_action(0.5)

        stuck = monitor.check_stuck_behavior()
        assert stuck is True

    def test_rolling_sharpe(self, monitor):
        """Test rolling Sharpe calculation."""
        # Record some PnL values
        for i in range(50):
            monitor.record_pnl(0.1 if i % 2 == 0 else -0.05)

        sharpe = monitor.get_rolling_sharpe()
        assert isinstance(sharpe, float)


class TestIntegrationFlow:
    """Test complete inference flow with all components."""

    @pytest.fixture
    def components(self):
        return {
            'builder': ObservationBuilderV19(),
            'tracker': StateTracker(),
            'risk': RiskManager(),
            'trader': PaperTrader(),
            'monitor': ModelMonitor()
        }

    def test_full_inference_flow(self, components):
        """Simulate one complete inference cycle."""
        model_id = "test_model"

        # 1. Get state features
        position, time_norm = components['tracker'].get_state_features(
            model_id, current_bar=30, total_bars=60
        )
        assert position == 0.0  # New model, no position
        assert time_norm == pytest.approx(0.483, rel=0.01)

        # 2. Build observation
        obs = components['builder'].build(SAMPLE_MARKET_FEATURES, position, time_norm)
        assert obs.shape == (15,)

        # 3. Simulate model action
        mock_action = 0.5  # Would come from model
        components['monitor'].record_action(mock_action)

        # 4. Validate with risk manager
        allowed, reason = components['risk'].validate_signal("LONG", 5.0)
        assert allowed is True

        # 5. Execute paper trade
        trade = components['trader'].execute_signal(
            model_id, "LONG", current_price=4200.0
        )
        assert trade is not None
        assert trade.direction == "LONG"

        # 6. Update state
        state = components['tracker'].update_position(model_id, 1.0, 4200.0)
        assert state.position == 1.0

        # 7. Check monitor health
        health = components['monitor'].get_health_status()
        assert health['status'] == 'healthy'

    def test_full_trade_cycle(self, components):
        """Test opening and closing a position."""
        model_id = "test_model"

        # Open long position
        components['trader'].execute_signal(model_id, "LONG", current_price=4200.0)
        components['tracker'].update_position(model_id, 1.0, 4200.0)

        # Get state after opening
        position, time_norm = components['tracker'].get_state_features(
            model_id, current_bar=40, total_bars=60
        )
        assert position == 1.0

        # Build observation with position
        obs = components['builder'].build(SAMPLE_MARKET_FEATURES, position, time_norm)
        assert obs[13] == 1.0  # Position in observation

        # Close position
        trade = components['trader'].execute_signal(model_id, "CLOSE", current_price=4250.0)
        components['tracker'].update_position(model_id, 0.0, 4250.0)

        # Verify closed
        assert trade.status == "closed"
        assert trade.pnl > 0  # Profit from long

        # Record result
        components['risk'].record_trade_result(trade.pnl_pct, "LONG")

        # Get final state
        position, _ = components['tracker'].get_state_features(model_id, 50, 60)
        assert position == 0.0

    def test_risk_blocks_after_losses(self, components):
        """Test risk manager blocking after consecutive losses."""
        # Configure risk manager with tight limits
        components['risk'] = RiskManager(RiskLimits(cooldown_after_losses=2))

        # Record consecutive losses
        components['risk'].record_trade_result(-1.0, "LONG")
        components['risk'].record_trade_result(-1.0, "LONG")

        # Check if cooldown is active
        status = components['risk'].get_status()
        assert status["cooldown_active"] is True

    def test_monitor_detects_stuck_model(self, components):
        """Test monitor detecting stuck behavior."""
        # Record repetitive actions
        for _ in range(50):
            components['monitor'].record_action(0.5)

        health = components['monitor'].get_health_status()
        assert health["stuck_behavior"] is True
        assert health["status"] in ("warning", "critical")


class TestComponentInteroperability:
    """Test that components work together correctly."""

    def test_observation_to_model_input(self):
        """Test observation can be used as model input."""
        builder = ObservationBuilderV19()
        obs = builder.build(SAMPLE_MARKET_FEATURES, position=0.0, time_normalized=0.5)

        # Verify it can be reshaped for batch inference
        batch_obs = obs.reshape(1, -1)
        assert batch_obs.shape == (1, 15)

    def test_state_tracker_to_observation(self):
        """Test state tracker output feeds into observation builder."""
        tracker = StateTracker()
        builder = ObservationBuilderV19()

        # Get state features
        position, time_norm = tracker.get_state_features("model_1", 30, 60)

        # Build observation
        obs = builder.build(SAMPLE_MARKET_FEATURES, position, time_norm)

        # Verify state features are in correct positions
        assert obs[13] == position
        assert abs(obs[14] - time_norm) < 0.001

    def test_paper_trader_to_risk_manager(self):
        """Test paper trader results feed into risk manager."""
        trader = PaperTrader()
        risk = RiskManager()

        # Execute trade
        trader.execute_signal("model_1", "LONG", 4200.0)
        trade = trader.execute_signal("model_1", "CLOSE", 4250.0)

        # Record result in risk manager
        risk.record_trade_result(trade.pnl_pct, trade.direction)

        # Verify tracking
        status = risk.get_status()
        assert status["trade_count_today"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
