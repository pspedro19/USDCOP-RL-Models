"""
Action Interpreters - Continuous Action → Trading Signal
=========================================================
Strategy pattern for interpreting continuous [-1, 1] actions.

Supports:
- threshold_3: 3-zone (LONG/HOLD/SHORT), fixed size=1.0 (V21.5b default)
- zone_5: 5-zone with variable position sizing

Usage:
    from src.training.environments.action_interpreters import create_action_interpreter

    interpreter = create_action_interpreter(config)
    action, size = interpreter.interpret(0.7)
    # -> (TradingAction.LONG, 1.0)

Contract: CTR-ACTION-INTERPRETER-001
Version: 1.0.0
Date: 2026-02-12
"""

import logging
from abc import ABC, abstractmethod
from typing import Tuple

logger = logging.getLogger(__name__)


class ActionInterpreter(ABC):
    """Abstract action interpreter."""

    @abstractmethod
    def interpret(self, raw_action: float) -> Tuple:
        """Interpret raw continuous action value.

        Args:
            raw_action: Continuous value in [-1, 1]

        Returns:
            Tuple of (TradingAction, position_size)
            position_size is in [0.0, 1.0]
        """
        ...


class ThresholdInterpreter(ActionInterpreter):
    """3-zone threshold interpreter (V21.5b default behavior).

    Zones:
        action > threshold_long  -> LONG, size=1.0
        action < threshold_short -> SHORT, size=1.0
        otherwise                -> HOLD, size=0.0
    """

    def __init__(self, threshold_long: float = 0.35, threshold_short: float = -0.35):
        self._threshold_long = threshold_long
        self._threshold_short = threshold_short

    def interpret(self, raw_action: float) -> Tuple:
        from src.training.environments.trading_env import TradingAction

        if raw_action > self._threshold_long:
            return TradingAction.LONG, 1.0
        elif raw_action < self._threshold_short:
            return TradingAction.SHORT, 1.0
        else:
            return TradingAction.HOLD, 0.0


class ZoneInterpreter(ActionInterpreter):
    """5-zone interpreter with variable position sizing.

    Zones (from high to low):
        action >= full_long   -> LONG, size=1.0
        action >= half_long   -> LONG, size=0.5
        action <= full_short  -> SHORT, size=1.0
        action <= half_short  -> SHORT, size=0.5
        otherwise             -> HOLD, size=0.0

    Args:
        full_long_threshold: Threshold for full long (default 0.6)
        half_long_threshold: Threshold for half long (default 0.2)
        half_short_threshold: Threshold for half short (default -0.2)
        full_short_threshold: Threshold for full short (default -0.6)
    """

    def __init__(
        self,
        full_long_threshold: float = 0.6,
        half_long_threshold: float = 0.2,
        half_short_threshold: float = -0.2,
        full_short_threshold: float = -0.6,
    ):
        self._full_long = full_long_threshold
        self._half_long = half_long_threshold
        self._half_short = half_short_threshold
        self._full_short = full_short_threshold

    def interpret(self, raw_action: float) -> Tuple:
        from src.training.environments.trading_env import TradingAction

        if raw_action >= self._full_long:
            return TradingAction.LONG, 1.0
        elif raw_action >= self._half_long:
            return TradingAction.LONG, 0.5
        elif raw_action <= self._full_short:
            return TradingAction.SHORT, 1.0
        elif raw_action <= self._half_short:
            return TradingAction.SHORT, 0.5
        else:
            return TradingAction.HOLD, 0.0


def create_action_interpreter(config) -> ActionInterpreter:
    """Factory: create action interpreter from config.

    Args:
        config: TradingEnvConfig or PipelineConfig

    Returns:
        ActionInterpreter instance
    """
    interp_type = getattr(config, "action_interpretation", "threshold_3")

    if interp_type == "zone_5":
        zone_cfg = getattr(config, "zone_5_config", {})
        if isinstance(zone_cfg, dict) and zone_cfg:
            return ZoneInterpreter(
                full_long_threshold=zone_cfg.get("full_long_threshold", 0.6),
                half_long_threshold=zone_cfg.get("half_long_threshold", 0.2),
                half_short_threshold=zone_cfg.get("half_short_threshold", -0.2),
                full_short_threshold=zone_cfg.get("full_short_threshold", -0.6),
            )
        return ZoneInterpreter()

    # Default: threshold_3
    threshold_long = getattr(config, "threshold_long", 0.35)
    threshold_short = getattr(config, "threshold_short", -0.35)
    return ThresholdInterpreter(
        threshold_long=threshold_long,
        threshold_short=threshold_short,
    )
