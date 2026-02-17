"""
Close Reason Detector - V22 P2 Reward Shaping
==============================================
Applies PnL multipliers based on HOW a position was closed.

CRITICAL: Uses MULTIPLIERS on base PnL reward, NOT fixed penalties.
A SL close at -4% hurts more than SL at -2% (proportional to PnL).

Contract: CTR-CLOSE-REASON-001
Version: 1.0.0
Date: 2026-02-06
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CloseReasonDetector:
    """
    Applies PnL multipliers based on HOW a position was closed.

    Close reasons:
    - stop_loss: Mechanical SL triggered → amplify loss (discourage)
    - take_profit: Mechanical TP triggered → slight bonus
    - agent_close: Agent used CLOSE action → reward smart exits
    - agent_reverse: Agent reversed position → treat like agent_close
    - timeout: Max duration exceeded → penalize indecision
    - trailing_stop: Trailing stop triggered → neutral
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}
        self.sl_multiplier = config.get("stop_loss_mult", 1.5)
        self.tp_multiplier = config.get("take_profit_mult", 1.2)
        self.agent_close_win_mult = config.get("agent_close_win", 1.1)
        self.agent_close_loss_mult = config.get("agent_close_loss", 0.7)
        self.timeout_mult = config.get("timeout_mult", 0.8)
        self.trailing_stop_mult = config.get("trailing_stop_mult", 1.0)
        self._stats = {
            "stop_loss": 0,
            "take_profit": 0,
            "agent_close": 0,
            "agent_reverse": 0,
            "timeout": 0,
            "trailing_stop": 0,
        }

    def shape_reward(
        self,
        base_reward: float,
        close_reason: str,
        pnl: float,
    ) -> float:
        """
        Apply multiplier to base_reward based on close_reason.

        Args:
            base_reward: The PnL-based reward before shaping
            close_reason: One of "stop_loss", "take_profit", "agent_close",
                         "timeout", "agent_reverse", "trailing_stop"
            pnl: Raw PnL percentage (needed to determine win/loss for agent_close)

        Returns:
            Shaped reward (base_reward * multiplier)
        """
        if close_reason in self._stats:
            self._stats[close_reason] += 1

        if close_reason == "stop_loss":
            return base_reward * self.sl_multiplier
        elif close_reason == "take_profit":
            return base_reward * self.tp_multiplier
        elif close_reason == "agent_close":
            if pnl > 0:
                return base_reward * self.agent_close_win_mult
            else:
                return base_reward * self.agent_close_loss_mult
        elif close_reason == "timeout":
            return base_reward * self.timeout_mult
        elif close_reason == "agent_reverse":
            if pnl > 0:
                return base_reward * self.agent_close_win_mult
            else:
                return base_reward * self.agent_close_loss_mult
        elif close_reason == "trailing_stop":
            return base_reward * self.trailing_stop_mult

        return base_reward

    def get_delta(
        self,
        base_reward: float,
        close_reason: str,
        pnl: float,
    ) -> float:
        """
        Get the reward adjustment (delta) from shaping.

        Returns shaped_reward - base_reward.
        """
        shaped = self.shape_reward(base_reward, close_reason, pnl)
        return shaped - base_reward

    def reset(self) -> None:
        """Reset statistics for new episode."""
        pass  # Keep lifetime stats

    def get_stats(self) -> Dict[str, Any]:
        """Get close reason statistics."""
        return {f"close_reason_{k}": v for k, v in self._stats.items()}
