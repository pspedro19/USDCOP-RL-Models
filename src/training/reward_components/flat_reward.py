"""
Flat Reward Component - Phase 3 Anti-Reward-Hacking.

CRITICAL FIX: flat_reward_weight=0.3 caused REWARD HACKING where HOLD
gave MORE reward than profitable trades. This caused the agent to learn
"always HOLD" instead of "trade when profitable".

PHASE 3 Changes:
- flat_reward_weight reduced 6x (0.3 -> 0.05)
- decay_enabled=True by default
- decay_half_life=12 bars (1 hour)
- decay_max=0.9 (90% reduction after extended HOLD)

Problem Solved:
--------------
When the agent is FLAT:
  - pnl_pct = 0 (no exposure = no PnL)
  - DSR/Sortino see zero returns (no variance to optimize)
  - Without flat_reward: Model has no incentive to HOLD vs. trade
  - WITH flat_reward but NO decay: Model learns to ALWAYS HOLD (reward hacking)

Solution:
---------
DIRECTION-NEUTRAL counterfactual reward WITH DECAY:
  If position=0 and market moved:
      reward = abs(market_return) * scale * decay_factor
  Where decay_factor decreases with consecutive HOLD bars.

Contract: CTR-REWARD-FLAT-003 (v3 - Anti-reward-hacking)
Author: Trading Team
Version: 3.0.0
Created: 2026-02-02
Updated: 2026-02-02 - PHASE 3 anti-reward-hacking with decay
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np

from .base import RewardComponent, ComponentType


@dataclass
class FlatRewardConfig:
    """Configuration for FlatReward component (v2 - Direction-neutral)."""
    # Scale factor for counterfactual reward
    scale: float = 50.0  # Scale to make comparable to PnL reward

    # Only reward if market moved more than this threshold
    min_move_threshold: float = 0.0001  # 0.01% minimum move

    # Multiplier for loss avoidance reward (direction-neutral)
    loss_avoidance_mult: float = 1.0   # Reward for avoiding loss in ANY direction

    # Decay for consecutive flat bars (to prevent always holding)
    decay_enabled: bool = False
    decay_half_life: int = 48  # Start decaying after 4 hours
    decay_max: float = 0.5  # Maximum decay (50% reduction)

    # History window for tracking counterfactual
    history_window: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "min_move_threshold": self.min_move_threshold,
            "loss_avoidance_mult": self.loss_avoidance_mult,
            "decay_enabled": self.decay_enabled,
            "decay_half_life": self.decay_half_life,
            "decay_max": self.decay_max,
            "history_window": self.history_window,
        }


class FlatReward(RewardComponent):
    """
    Direction-neutral counterfactual reward for FLAT positions.

    Provides positive signal when agent stays flat during ANY market movement.
    This addresses the fundamental issue where HOLD receives zero reward signal.

    IMPORTANT: This is DIRECTION-NEUTRAL to avoid LONG/SHORT bias.

    Reward Logic (v2 - Symmetric):
    ------------------------------
    1. If position=0 and market moved (UP or DOWN):
       - Agent avoided a loss from the WRONG direction
       - Market DOWN: Would have lost if LONG
       - Market UP: Would have lost if SHORT
       - Reward = abs(market_return) * scale * loss_avoidance_mult

    2. If position != 0:
       - Component returns 0 (PnL component handles actual positions)

    Why Direction-Neutral?
    ----------------------
    The previous v1 implementation only rewarded avoiding LONG losses,
    which created a systematic bias toward SHORT positions. The model
    learned "when in doubt, SHORT" because:
    - FLAT during market DOWN: Got reward (avoided LONG loss)
    - FLAT during market UP: Got 0 or penalty (missed LONG gain)

    This v2 treats both directions equally, allowing the model to learn
    unbiased LONG/SHORT decisions based solely on market features.

    Example:
    --------
        >>> flat_reward = FlatReward(scale=50.0)
        >>> # Market dropped 0.1%, agent was flat
        >>> reward = flat_reward.calculate(position=0, market_return=-0.001)
        >>> print(reward)  # 0.05 (avoided LONG loss)
        >>>
        >>> # Market rose 0.1%, agent was flat
        >>> reward = flat_reward.calculate(position=0, market_return=0.001)
        >>> print(reward)  # 0.05 (avoided SHORT loss) - SAME reward
    """

    def __init__(
        self,
        scale: float = 50.0,
        min_move_threshold: float = 0.0001,
        loss_avoidance_mult: float = 1.0,
        decay_enabled: bool = True,    # PHASE3: default True for anti-hacking
        decay_half_life: int = 12,     # PHASE3: 1 hour (12 bars * 5min)
        decay_max: float = 0.9,        # PHASE3: 90% max reduction for anti-hacking
        history_window: int = 10,
    ):
        """
        Initialize FlatReward component (v3 - Anti-Reward-Hacking).

        PHASE 3 FIX: decay_enabled=True by default to prevent reward hacking
        where HOLD gives more reward than profitable trades.

        Args:
            scale: Scale factor for counterfactual reward
            min_move_threshold: Minimum market move to trigger reward
            loss_avoidance_mult: Multiplier for loss avoidance reward (any direction)
            decay_enabled: Enable decay for consecutive flat bars (PHASE3: True)
            decay_half_life: Bars until reward decays by 50% (PHASE3: 12 = 1 hour)
            decay_max: Maximum decay factor (PHASE3: 0.9 = 90% reduction)
            history_window: Window for tracking returns
        """
        super().__init__()

        self._scale = scale
        self._min_move_threshold = min_move_threshold
        self._loss_avoidance_mult = loss_avoidance_mult
        self._decay_enabled = decay_enabled
        self._decay_half_life = decay_half_life
        self._decay_max = decay_max
        self._history_window = history_window

        # State tracking
        self._consecutive_flat_bars = 0
        self._return_history: List[float] = []

        # Statistics
        self._total_loss_avoided = 0.0
        self._flat_bar_count = 0

    @property
    def name(self) -> str:
        return "flat_reward"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.BONUS

    def calculate(
        self,
        position: int,
        market_return: float,
        **kwargs
    ) -> float:
        """
        Calculate counterfactual reward for FLAT position.

        Args:
            position: Current position (-1, 0, 1)
            market_return: Raw market return this bar (not z-scored!)
            **kwargs: Additional arguments (ignored)

        Returns:
            Reward value (positive for avoided losses, 0 or negative for missed gains)
        """
        if not self._enabled:
            return 0.0

        # Track return history
        self._return_history.append(market_return)
        if len(self._return_history) > self._history_window:
            self._return_history.pop(0)

        # Only applies when FLAT
        if position != 0:
            self._consecutive_flat_bars = 0
            return 0.0

        # Count flat bars
        self._consecutive_flat_bars += 1
        self._flat_bar_count += 1

        # Check if market moved enough
        if abs(market_return) < self._min_move_threshold:
            return 0.0

        # Calculate counterfactual reward - DIRECTION NEUTRAL
        # When flat, the agent avoided a loss from the WRONG direction:
        #   - If market DOWN: avoided loss from LONG
        #   - If market UP: avoided loss from SHORT
        # Both directions are treated equally - no LONG/SHORT bias
        reward = abs(market_return) * self._scale * self._loss_avoidance_mult
        self._total_loss_avoided += abs(market_return)

        # Apply decay for extended flat periods
        if self._decay_enabled and self._consecutive_flat_bars > 0:
            decay_factor = 1.0 - min(
                self._decay_max,
                self._decay_max * (1 - 0.5 ** (self._consecutive_flat_bars / self._decay_half_life))
            )
            reward *= decay_factor

        # Update stats
        self._update_stats(reward)

        return reward

    def reset(self) -> None:
        """Reset for new episode."""
        self._consecutive_flat_bars = 0
        self._return_history = []
        self._stats.reset()

    def reset_position(self) -> None:
        """Reset when position changes."""
        self._consecutive_flat_bars = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        base_stats = super().get_stats()
        base_stats.update({
            f"{self.name}_loss_avoided": self._total_loss_avoided,
            f"{self.name}_flat_bar_count": self._flat_bar_count,
            f"{self.name}_avg_return_in_window": np.mean(self._return_history) if self._return_history else 0.0,
        })
        return base_stats

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        base_config = super().get_config()
        base_config.update({
            "scale": self._scale,
            "min_move_threshold": self._min_move_threshold,
            "loss_avoidance_mult": self._loss_avoidance_mult,
            "decay_enabled": self._decay_enabled,
            "decay_half_life": self._decay_half_life,
            "version": "2.0.0_direction_neutral",
        })
        return base_config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FlatReward",
    "FlatRewardConfig",
]
