"""
Drawdown Penalty Reward Component
===================================
Penalizes the agent when portfolio drawdown exceeds a configurable threshold.

Tracks peak equity (cumulative return) and applies a penalty proportional
to how far below the peak the current equity sits.

Usage:
    component = DrawdownPenaltyComponent(threshold=0.05, max_penalty=0.5)
    penalty = component.calculate(cumulative_return=0.03, peak_return=0.08)
    # drawdown = 0.08 - 0.03 = 0.05, which equals threshold → penalty starts

Contract: CTR-REWARD-DRAWDOWN-001
Version: 1.0.0
Date: 2026-02-12
"""

import logging
from typing import Any, Dict

from .base import RewardComponent, ComponentType

logger = logging.getLogger(__name__)


class DrawdownPenaltyComponent(RewardComponent):
    """Penalizes drawdown beyond a configurable threshold.

    Tracks peak cumulative return internally.  When the current cumulative
    return drops below (peak - threshold), a penalty is applied that scales
    linearly from 0 at the threshold to max_penalty at 2x threshold.

    Args:
        threshold: Drawdown percentage before penalty kicks in (default 0.05 = 5%)
        max_penalty: Maximum penalty value (default 0.5)
    """

    def __init__(
        self,
        threshold: float = 0.05,
        max_penalty: float = 0.5,
    ):
        super().__init__()
        self._threshold = threshold
        self._max_penalty = max_penalty
        self._peak_return = 0.0
        self._cumulative_return = 0.0

    @property
    def name(self) -> str:
        return "drawdown_penalty"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        cumulative_return: float = 0.0,
        **kwargs,
    ) -> float:
        """Calculate drawdown penalty.

        Args:
            cumulative_return: Current cumulative return of the episode

        Returns:
            Negative penalty value (0 if drawdown < threshold)
        """
        self._cumulative_return = cumulative_return

        # Update peak
        if cumulative_return > self._peak_return:
            self._peak_return = cumulative_return

        # Calculate drawdown from peak
        drawdown = self._peak_return - cumulative_return

        # No penalty if below threshold
        if drawdown <= self._threshold:
            self._update_stats(0.0)
            return 0.0

        # Linear penalty from threshold to 2x threshold
        excess = drawdown - self._threshold
        penalty_ratio = min(excess / max(self._threshold, 1e-8), 1.0)
        penalty = -self._max_penalty * penalty_ratio

        self._update_stats(penalty)
        return penalty

    def reset(self) -> None:
        """Reset for new episode."""
        self._peak_return = 0.0
        self._cumulative_return = 0.0
        self._stats.reset()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "threshold": self._threshold,
            "max_penalty": self._max_penalty,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats["drawdown_peak_return"] = self._peak_return
        stats["drawdown_current"] = self._peak_return - self._cumulative_return
        return stats
