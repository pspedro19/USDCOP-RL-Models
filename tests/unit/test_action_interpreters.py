"""
Unit tests for Action Interpreters (Phase 3).

Tests threshold-3 and zone-5 interpreters and the factory.
"""

import pytest
from unittest.mock import MagicMock


# ============================================================================
# Threshold (3-zone) interpreter
# ============================================================================


class TestThresholdInterpreter:
    """Test ThresholdInterpreter (V21.5b default)."""

    def test_long_above_threshold(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter(threshold_long=0.35, threshold_short=-0.35)
        action, size = interp.interpret(0.7)
        assert action == TradingAction.LONG
        assert size == 1.0

    def test_short_below_threshold(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter(threshold_long=0.35, threshold_short=-0.35)
        action, size = interp.interpret(-0.5)
        assert action == TradingAction.SHORT
        assert size == 1.0

    def test_hold_in_dead_zone(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter(threshold_long=0.35, threshold_short=-0.35)
        action, size = interp.interpret(0.0)
        assert action == TradingAction.HOLD
        assert size == 0.0

    def test_boundary_long(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter(threshold_long=0.35, threshold_short=-0.35)
        # At exactly 0.35, should be HOLD (> not >=)
        action, size = interp.interpret(0.35)
        assert action == TradingAction.HOLD

    def test_boundary_short(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter(threshold_long=0.35, threshold_short=-0.35)
        # At exactly -0.35, should be HOLD (< not <=)
        action, size = interp.interpret(-0.35)
        assert action == TradingAction.HOLD

    def test_extreme_values(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter()
        action, size = interp.interpret(1.0)
        assert action == TradingAction.LONG
        assert size == 1.0

        action, size = interp.interpret(-1.0)
        assert action == TradingAction.SHORT
        assert size == 1.0

    def test_custom_thresholds(self):
        from src.training.environments.action_interpreters import ThresholdInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ThresholdInterpreter(threshold_long=0.5, threshold_short=-0.5)
        # 0.4 would be LONG with default 0.35 but HOLD with 0.5
        action, size = interp.interpret(0.4)
        assert action == TradingAction.HOLD


# ============================================================================
# Zone (5-zone) interpreter
# ============================================================================


class TestZoneInterpreter:
    """Test ZoneInterpreter (5-zone with variable sizing)."""

    def test_full_long(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter()
        action, size = interp.interpret(0.7)
        assert action == TradingAction.LONG
        assert size == 1.0

    def test_half_long(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter()
        action, size = interp.interpret(0.3)
        assert action == TradingAction.LONG
        assert size == 0.5

    def test_hold(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter()
        action, size = interp.interpret(0.0)
        assert action == TradingAction.HOLD
        assert size == 0.0

    def test_half_short(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter()
        action, size = interp.interpret(-0.3)
        assert action == TradingAction.SHORT
        assert size == 0.5

    def test_full_short(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter()
        action, size = interp.interpret(-0.7)
        assert action == TradingAction.SHORT
        assert size == 1.0

    def test_zone_boundaries(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter(
            full_long_threshold=0.6,
            half_long_threshold=0.2,
            half_short_threshold=-0.2,
            full_short_threshold=-0.6,
        )

        # At full_long boundary (>=)
        action, size = interp.interpret(0.6)
        assert action == TradingAction.LONG
        assert size == 1.0

        # Just below full_long → half_long
        action, size = interp.interpret(0.59)
        assert action == TradingAction.LONG
        assert size == 0.5

        # At half_long boundary (>=)
        action, size = interp.interpret(0.2)
        assert action == TradingAction.LONG
        assert size == 0.5

        # Just below half_long → HOLD
        action, size = interp.interpret(0.19)
        assert action == TradingAction.HOLD
        assert size == 0.0

        # At half_short boundary (<=)
        action, size = interp.interpret(-0.2)
        assert action == TradingAction.SHORT
        assert size == 0.5

        # At full_short boundary (<=)
        action, size = interp.interpret(-0.6)
        assert action == TradingAction.SHORT
        assert size == 1.0

    def test_custom_thresholds(self):
        from src.training.environments.action_interpreters import ZoneInterpreter
        from src.training.environments.trading_env import TradingAction

        interp = ZoneInterpreter(
            full_long_threshold=0.8,
            half_long_threshold=0.4,
            half_short_threshold=-0.4,
            full_short_threshold=-0.8,
        )

        # 0.5 → half long (between 0.4 and 0.8)
        action, size = interp.interpret(0.5)
        assert action == TradingAction.LONG
        assert size == 0.5


# ============================================================================
# Factory
# ============================================================================


class TestCreateActionInterpreter:
    """Test create_action_interpreter() factory."""

    def test_default_is_threshold_3(self):
        from src.training.environments.action_interpreters import (
            create_action_interpreter,
            ThresholdInterpreter,
        )

        config = MagicMock(spec=[])  # No attributes
        interp = create_action_interpreter(config)
        assert isinstance(interp, ThresholdInterpreter)

    def test_threshold_3_from_config(self):
        from src.training.environments.action_interpreters import (
            create_action_interpreter,
            ThresholdInterpreter,
        )

        config = MagicMock()
        config.action_interpretation = "threshold_3"
        config.threshold_long = 0.45
        config.threshold_short = -0.45

        interp = create_action_interpreter(config)
        assert isinstance(interp, ThresholdInterpreter)
        assert interp._threshold_long == 0.45
        assert interp._threshold_short == -0.45

    def test_zone_5_from_config(self):
        from src.training.environments.action_interpreters import (
            create_action_interpreter,
            ZoneInterpreter,
        )

        config = MagicMock()
        config.action_interpretation = "zone_5"
        config.zone_5_config = {
            "full_long_threshold": 0.7,
            "half_long_threshold": 0.3,
            "half_short_threshold": -0.3,
            "full_short_threshold": -0.7,
        }

        interp = create_action_interpreter(config)
        assert isinstance(interp, ZoneInterpreter)
        assert interp._full_long == 0.7
        assert interp._half_long == 0.3

    def test_zone_5_default_thresholds(self):
        from src.training.environments.action_interpreters import (
            create_action_interpreter,
            ZoneInterpreter,
        )

        config = MagicMock()
        config.action_interpretation = "zone_5"
        config.zone_5_config = {}  # Empty dict → use defaults

        interp = create_action_interpreter(config)
        assert isinstance(interp, ZoneInterpreter)
        assert interp._full_long == 0.6  # Default
        assert interp._half_long == 0.2  # Default
