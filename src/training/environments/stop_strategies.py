"""
Stop Strategies - Fixed % and ATR Dynamic Stops
================================================
Strategy pattern for position stop-loss and take-profit logic.

Usage:
    from src.training.environments.stop_strategies import create_stop_strategy

    strategy = create_stop_strategy(config)
    strategy.on_position_open(atr_pct=0.012)
    result = strategy.check_stop(unrealized_pnl_pct=-0.03, bars_held=10)

Contract: CTR-STOP-STRATEGY-001
Version: 1.0.0
Date: 2026-02-12
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class StopStrategy(ABC):
    """Abstract stop-loss/take-profit strategy."""

    @abstractmethod
    def check_stop(self, unrealized_pnl_pct: float, bars_held: int) -> Optional[str]:
        """Check if a stop condition is met.

        Args:
            unrealized_pnl_pct: Unrealized PnL as decimal (e.g., -0.03 = -3%)
            bars_held: Number of bars the position has been held

        Returns:
            Close reason string ("stop_loss", "take_profit") or None
        """
        ...

    @abstractmethod
    def on_position_open(self, **context) -> None:
        """Called when a new position is opened.

        Args:
            **context: Entry context (e.g., atr_pct for ATR mode)
        """
        ...

    @abstractmethod
    def on_position_close(self) -> None:
        """Called when a position is closed (reset per-position state)."""
        ...


class FixedPctStopStrategy(StopStrategy):
    """Fixed percentage stop-loss and take-profit (current V21.5b behavior).

    Args:
        stop_loss_pct: Stop-loss threshold as negative decimal (e.g., -0.04)
        take_profit_pct: Take-profit threshold as positive decimal (e.g., 0.04)
    """

    def __init__(self, stop_loss_pct: float = -0.04, take_profit_pct: float = 0.04):
        self._sl_pct = stop_loss_pct
        self._tp_pct = take_profit_pct

    def check_stop(self, unrealized_pnl_pct: float, bars_held: int) -> Optional[str]:
        if unrealized_pnl_pct < self._sl_pct:
            logger.info(
                f"[STOP-LOSS] Fixed: {unrealized_pnl_pct:.2%} < {self._sl_pct:.2%}"
            )
            return "stop_loss"
        if unrealized_pnl_pct >= self._tp_pct:
            logger.info(
                f"[TAKE-PROFIT] Fixed: {unrealized_pnl_pct:.2%} >= {self._tp_pct:.2%}"
            )
            return "take_profit"
        return None

    def on_position_open(self, **context) -> None:
        pass  # Fixed stops don't need entry context

    def on_position_close(self) -> None:
        pass


class ATRDynamicStopStrategy(StopStrategy):
    """ATR-based dynamic stop-loss and take-profit.

    Stop levels are computed as N * ATR at entry time.

    Args:
        sl_atr_multiplier: SL = entry_atr * multiplier (default 2.5)
        tp_atr_multiplier: TP = entry_atr * multiplier (default 5.0)
        atr_lookback: ATR lookback period (for reference, actual ATR passed at entry)
        min_sl_pct: Minimum SL floor (e.g., -0.01 = 1%)
        max_sl_pct: Maximum SL cap (e.g., -0.08 = 8%)
        min_tp_pct: Minimum TP floor (e.g., 0.01 = 1%)
        max_tp_pct: Maximum TP cap (e.g., 0.10 = 10%)
    """

    def __init__(
        self,
        sl_atr_multiplier: float = 2.5,
        tp_atr_multiplier: float = 5.0,
        atr_lookback: int = 14,
        min_sl_pct: float = -0.01,
        max_sl_pct: float = -0.08,
        min_tp_pct: float = 0.01,
        max_tp_pct: float = 0.10,
    ):
        self._sl_mult = sl_atr_multiplier
        self._tp_mult = tp_atr_multiplier
        self._atr_lookback = atr_lookback
        self._min_sl = min_sl_pct  # Negative
        self._max_sl = max_sl_pct  # Negative (more negative = wider)
        self._min_tp = min_tp_pct  # Positive
        self._max_tp = max_tp_pct  # Positive

        # Per-position state
        self._current_sl: Optional[float] = None
        self._current_tp: Optional[float] = None

    def on_position_open(self, **context) -> None:
        """Compute stop levels from entry ATR.

        Args:
            atr_pct: ATR as decimal percentage at entry time
        """
        atr_pct = context.get("atr_pct", 0.01)

        # SL = -N * ATR, clamped to [max_sl, min_sl] (both negative)
        raw_sl = -self._sl_mult * atr_pct
        self._current_sl = max(self._max_sl, min(self._min_sl, raw_sl))

        # TP = N * ATR, clamped to [min_tp, max_tp]
        raw_tp = self._tp_mult * atr_pct
        self._current_tp = max(self._min_tp, min(self._max_tp, raw_tp))

        logger.debug(
            f"[ATR-STOP] Entry ATR={atr_pct:.4f} -> "
            f"SL={self._current_sl:.4f}, TP={self._current_tp:.4f}"
        )

    def on_position_close(self) -> None:
        self._current_sl = None
        self._current_tp = None

    def check_stop(self, unrealized_pnl_pct: float, bars_held: int) -> Optional[str]:
        if self._current_sl is None or self._current_tp is None:
            return None

        if unrealized_pnl_pct < self._current_sl:
            logger.info(
                f"[STOP-LOSS] ATR: {unrealized_pnl_pct:.2%} < {self._current_sl:.2%}"
            )
            return "stop_loss"
        if unrealized_pnl_pct >= self._current_tp:
            logger.info(
                f"[TAKE-PROFIT] ATR: {unrealized_pnl_pct:.2%} >= {self._current_tp:.2%}"
            )
            return "take_profit"
        return None


def create_stop_strategy(config) -> StopStrategy:
    """Factory: create stop strategy from PipelineConfig.

    Args:
        config: PipelineConfig (or TradingEnvConfig for direct use)

    Returns:
        StopStrategy instance
    """
    # Support both PipelineConfig and TradingEnvConfig
    stop_mode = getattr(config, "stop_mode", "fixed_pct")

    if stop_mode == "atr_dynamic":
        atr_cfg = getattr(config, "atr_stop", {})
        if isinstance(atr_cfg, dict):
            return ATRDynamicStopStrategy(
                sl_atr_multiplier=atr_cfg.get("sl_atr_multiplier", 2.5),
                tp_atr_multiplier=atr_cfg.get("tp_atr_multiplier", 5.0),
                atr_lookback=atr_cfg.get("atr_lookback", 14),
                min_sl_pct=atr_cfg.get("min_sl_pct", -0.01),
                max_sl_pct=atr_cfg.get("max_sl_pct", -0.08),
                min_tp_pct=atr_cfg.get("min_tp_pct", 0.01),
                max_tp_pct=atr_cfg.get("max_tp_pct", 0.10),
            )
        return ATRDynamicStopStrategy()

    # Default: fixed_pct
    sl = getattr(config, "stop_loss_pct", -0.04)
    tp = getattr(config, "take_profit_pct", 0.04)
    return FixedPctStopStrategy(stop_loss_pct=sl, take_profit_pct=tp)
