"""
Unit tests for Stop Strategies (Phase 2).

Tests fixed percentage stops, ATR dynamic stops, and the factory.
"""

import pytest
from unittest.mock import MagicMock


# ============================================================================
# Fixed percentage stops
# ============================================================================


class TestFixedPctStopStrategy:
    """Test FixedPctStopStrategy behavior."""

    def test_stop_loss_triggered(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy(stop_loss_pct=-0.04, take_profit_pct=0.04)
        result = strategy.check_stop(unrealized_pnl_pct=-0.05, bars_held=10)
        assert result == "stop_loss"

    def test_take_profit_triggered(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy(stop_loss_pct=-0.04, take_profit_pct=0.04)
        result = strategy.check_stop(unrealized_pnl_pct=0.05, bars_held=10)
        assert result == "take_profit"

    def test_no_stop_in_range(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy(stop_loss_pct=-0.04, take_profit_pct=0.04)
        result = strategy.check_stop(unrealized_pnl_pct=0.02, bars_held=10)
        assert result is None

    def test_exact_sl_boundary(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy(stop_loss_pct=-0.04, take_profit_pct=0.04)
        # At exactly -4%, should NOT trigger (< not <=)
        result = strategy.check_stop(unrealized_pnl_pct=-0.04, bars_held=10)
        assert result is None

    def test_exact_tp_boundary(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy(stop_loss_pct=-0.04, take_profit_pct=0.04)
        # At exactly +4%, SHOULD trigger (>= for TP)
        result = strategy.check_stop(unrealized_pnl_pct=0.04, bars_held=10)
        assert result == "take_profit"

    def test_on_position_open_close_noop(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy()
        strategy.on_position_open()  # Should not raise
        strategy.on_position_close()  # Should not raise

    def test_asymmetric_stops(self):
        from src.training.environments.stop_strategies import FixedPctStopStrategy

        strategy = FixedPctStopStrategy(stop_loss_pct=-0.025, take_profit_pct=0.06)
        assert strategy.check_stop(-0.03, 5) == "stop_loss"
        assert strategy.check_stop(0.05, 5) is None
        assert strategy.check_stop(0.06, 5) == "take_profit"


# ============================================================================
# ATR dynamic stops
# ============================================================================


class TestATRDynamicStopStrategy:
    """Test ATRDynamicStopStrategy behavior."""

    def test_basic_atr_stops(self):
        from src.training.environments.stop_strategies import ATRDynamicStopStrategy

        strategy = ATRDynamicStopStrategy(
            sl_atr_multiplier=2.0,
            tp_atr_multiplier=4.0,
            min_sl_pct=-0.01,
            max_sl_pct=-0.10,
            min_tp_pct=0.01,
            max_tp_pct=0.15,
        )

        # Open position with ATR = 1% → SL = -2%, TP = +4%
        strategy.on_position_open(atr_pct=0.01)

        # Within range
        assert strategy.check_stop(-0.01, 5) is None
        assert strategy.check_stop(0.03, 5) is None

        # SL triggered at -2.1%
        assert strategy.check_stop(-0.021, 5) == "stop_loss"

        # TP triggered at +4%
        assert strategy.check_stop(0.04, 5) == "take_profit"

    def test_atr_sl_clamped_to_min(self):
        from src.training.environments.stop_strategies import ATRDynamicStopStrategy

        # Very small ATR → SL would be tiny, clamped to min_sl_pct
        strategy = ATRDynamicStopStrategy(
            sl_atr_multiplier=2.0,
            min_sl_pct=-0.01,
            max_sl_pct=-0.08,
        )
        strategy.on_position_open(atr_pct=0.001)  # raw SL = -0.002
        # Should be clamped to -0.01
        assert strategy._current_sl == -0.01

    def test_atr_sl_clamped_to_max(self):
        from src.training.environments.stop_strategies import ATRDynamicStopStrategy

        # Very large ATR → SL would be huge, clamped to max_sl_pct
        strategy = ATRDynamicStopStrategy(
            sl_atr_multiplier=2.5,
            min_sl_pct=-0.01,
            max_sl_pct=-0.08,
        )
        strategy.on_position_open(atr_pct=0.10)  # raw SL = -0.25
        assert strategy._current_sl == -0.08

    def test_no_stop_before_position_open(self):
        from src.training.environments.stop_strategies import ATRDynamicStopStrategy

        strategy = ATRDynamicStopStrategy()
        # Before on_position_open, levels are None → no stop
        assert strategy.check_stop(-0.5, 100) is None

    def test_position_close_resets_levels(self):
        from src.training.environments.stop_strategies import ATRDynamicStopStrategy

        strategy = ATRDynamicStopStrategy()
        strategy.on_position_open(atr_pct=0.01)
        assert strategy._current_sl is not None

        strategy.on_position_close()
        assert strategy._current_sl is None
        assert strategy._current_tp is None


# ============================================================================
# Factory
# ============================================================================


class TestCreateStopStrategy:
    """Test create_stop_strategy() factory."""

    def test_default_is_fixed_pct(self):
        from src.training.environments.stop_strategies import (
            create_stop_strategy,
            FixedPctStopStrategy,
        )

        config = MagicMock(spec=[])  # No attributes
        strategy = create_stop_strategy(config)
        assert isinstance(strategy, FixedPctStopStrategy)

    def test_fixed_pct_from_config(self):
        from src.training.environments.stop_strategies import (
            create_stop_strategy,
            FixedPctStopStrategy,
        )

        config = MagicMock()
        config.stop_mode = "fixed_pct"
        config.stop_loss_pct = -0.03
        config.take_profit_pct = 0.05

        strategy = create_stop_strategy(config)
        assert isinstance(strategy, FixedPctStopStrategy)
        assert strategy._sl_pct == -0.03
        assert strategy._tp_pct == 0.05

    def test_atr_dynamic_from_config(self):
        from src.training.environments.stop_strategies import (
            create_stop_strategy,
            ATRDynamicStopStrategy,
        )

        config = MagicMock()
        config.stop_mode = "atr_dynamic"
        config.atr_stop = {
            "sl_atr_multiplier": 3.0,
            "tp_atr_multiplier": 6.0,
            "atr_lookback": 20,
            "min_sl_pct": -0.02,
            "max_sl_pct": -0.10,
        }

        strategy = create_stop_strategy(config)
        assert isinstance(strategy, ATRDynamicStopStrategy)
        assert strategy._sl_mult == 3.0
        assert strategy._tp_mult == 6.0
