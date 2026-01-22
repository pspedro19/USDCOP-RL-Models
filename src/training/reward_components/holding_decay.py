"""
Holding Time Decay Component.

Implements exponential penalty for holding positions over time.
This discourages the agent from holding positions indefinitely,
which is important for:
1. Gap risk (overnight/weekend gaps)
2. Avoiding missed opportunities
3. Preventing position "sticking"

Contract: CTR-REWARD-HOLDING-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Formula:
    decay = max_penalty * (1 - exp(-lambda * holding_bars))

Where lambda controls decay speed (higher = faster penalty accumulation).
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple

from .base import RewardComponent, ComponentType, IDecayModel


class HoldingDecay(RewardComponent):
    """
    Exponential decay penalty for position holding time.

    The penalty grows exponentially with holding time:
    - At t=0: penalty = 0
    - At t→∞: penalty → max_penalty

    This creates incentive to close positions rather than hold indefinitely,
    while allowing reasonable holding periods without excessive penalty.

    The decay rate is calibrated so that:
    - At half_life bars: penalty = max_penalty * 0.5
    - At 2*half_life bars: penalty = max_penalty * 0.75

    Usage:
        >>> decay = HoldingDecay(half_life_bars=48, max_penalty=0.3)
        >>> penalty = decay.calculate(holding_bars=24)  # ~0.15
        >>> penalty = decay.calculate(holding_bars=96)  # ~0.26
    """

    def __init__(
        self,
        half_life_bars: int = 48,
        max_penalty: float = 0.3,
        flat_threshold: int = 0,
        enable_overnight_boost: bool = True,
        overnight_multiplier: float = 1.5,
    ):
        """
        Initialize holding decay.

        Args:
            half_life_bars: Bars until penalty reaches 50% of max
                           (48 bars = 4 hours at 5min timeframe)
            max_penalty: Maximum penalty value (asymptote)
            flat_threshold: Bars before penalty starts (grace period)
            enable_overnight_boost: Apply extra penalty for overnight holds
            overnight_multiplier: Multiplier for overnight positions
        """
        super().__init__()
        self._half_life_bars = half_life_bars
        self._max_penalty = max_penalty
        self._flat_threshold = flat_threshold
        self._enable_overnight_boost = enable_overnight_boost
        self._overnight_multiplier = overnight_multiplier

        # Calculate lambda from half-life: ln(2) / half_life
        self._lambda = np.log(2) / half_life_bars if half_life_bars > 0 else 0.01

        # State tracking
        self._current_holding_bars = 0
        self._is_overnight = False
        self._total_penalty_applied = 0.0
        self._penalty_count = 0

    @property
    def name(self) -> str:
        return "holding_decay"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        holding_bars: Optional[int] = None,
        is_overnight: bool = False,
        has_position: bool = True,
        **kwargs
    ) -> float:
        """
        Calculate holding decay penalty.

        Args:
            holding_bars: Number of bars holding position (if None, uses internal state)
            is_overnight: Whether current bar is overnight
            has_position: Whether agent has an open position

        Returns:
            Penalty value (negative float, or 0 if no position)
        """
        if not self._enabled:
            return 0.0

        # No penalty if no position
        if not has_position:
            self._current_holding_bars = 0
            return 0.0

        # Use provided or internal counter
        if holding_bars is not None:
            bars = holding_bars
        else:
            self._current_holding_bars += 1
            bars = self._current_holding_bars

        # Grace period
        if bars <= self._flat_threshold:
            return 0.0

        # Effective holding time after grace period
        effective_bars = bars - self._flat_threshold

        # Exponential decay formula: penalty = max * (1 - exp(-λt))
        decay_factor = 1.0 - np.exp(-self._lambda * effective_bars)
        penalty = self._max_penalty * decay_factor

        # Overnight boost
        if is_overnight and self._enable_overnight_boost:
            penalty *= self._overnight_multiplier
            self._is_overnight = True

        # Tracking
        self._total_penalty_applied += penalty
        self._penalty_count += 1

        result = -penalty  # Negative for penalty

        self._update_stats(result)
        return result

    def get_decay_at_bars(self, bars: int) -> float:
        """
        Calculate decay factor at specific bar count (for analysis).

        Args:
            bars: Number of holding bars

        Returns:
            Decay factor [0, 1] where 1 = max penalty
        """
        if bars <= self._flat_threshold:
            return 0.0
        effective_bars = bars - self._flat_threshold
        return 1.0 - np.exp(-self._lambda * effective_bars)

    def get_penalty_schedule(self, max_bars: int = 100) -> Dict[int, float]:
        """
        Get penalty schedule for visualization.

        Args:
            max_bars: Maximum bars to calculate

        Returns:
            Dict mapping bars -> penalty value
        """
        return {
            bars: self._max_penalty * self.get_decay_at_bars(bars)
            for bars in range(0, max_bars + 1, max(1, max_bars // 20))
        }

    def reset(self) -> None:
        """Reset state for new episode."""
        self._current_holding_bars = 0
        self._is_overnight = False

    def reset_position(self) -> None:
        """Reset holding counter when position is closed."""
        self._current_holding_bars = 0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "half_life_bars": self._half_life_bars,
            "max_penalty": self._max_penalty,
            "flat_threshold": self._flat_threshold,
            "enable_overnight_boost": self._enable_overnight_boost,
            "overnight_multiplier": self._overnight_multiplier,
            "lambda": self._lambda,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "holding_current_bars": self._current_holding_bars,
            "holding_is_overnight": self._is_overnight,
            "holding_total_penalty": self._total_penalty_applied,
            "holding_penalty_count": self._penalty_count,
            "holding_avg_penalty": (
                self._total_penalty_applied / self._penalty_count
                if self._penalty_count > 0 else 0.0
            ),
        })
        return stats

    @property
    def current_holding_bars(self) -> int:
        """Current bars in position."""
        return self._current_holding_bars

    @property
    def current_decay_factor(self) -> float:
        """Current decay factor based on holding time."""
        return self.get_decay_at_bars(self._current_holding_bars)


class GapRiskPenalty(RewardComponent):
    """
    Penalty specifically for gap risk exposure.

    Applies extra penalty when position is held through:
    - Market close (overnight gaps)
    - Weekend
    - Holidays

    This complements HoldingDecay by adding time-specific risk awareness.
    """

    def __init__(
        self,
        overnight_penalty: float = 0.1,
        weekend_penalty: float = 0.25,
        holiday_penalty: float = 0.15,
    ):
        """
        Initialize gap risk penalty.

        Args:
            overnight_penalty: Penalty for overnight holding
            weekend_penalty: Penalty for weekend holding
            holiday_penalty: Penalty for holiday holding
        """
        super().__init__()
        self._overnight_penalty = overnight_penalty
        self._weekend_penalty = weekend_penalty
        self._holiday_penalty = holiday_penalty

        # State
        self._overnight_count = 0
        self._weekend_count = 0
        self._holiday_count = 0

    @property
    def name(self) -> str:
        return "gap_risk"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        has_position: bool = True,
        is_overnight: bool = False,
        is_weekend: bool = False,
        is_holiday: bool = False,
        **kwargs
    ) -> float:
        """
        Calculate gap risk penalty.

        Args:
            has_position: Whether agent has open position
            is_overnight: Whether approaching overnight
            is_weekend: Whether approaching weekend
            is_holiday: Whether approaching holiday

        Returns:
            Penalty value (negative if risky, 0 otherwise)
        """
        if not self._enabled or not has_position:
            return 0.0

        penalty = 0.0

        # Weekend takes priority (highest risk)
        if is_weekend:
            penalty = self._weekend_penalty
            self._weekend_count += 1
        elif is_holiday:
            penalty = self._holiday_penalty
            self._holiday_count += 1
        elif is_overnight:
            penalty = self._overnight_penalty
            self._overnight_count += 1

        result = -penalty

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset state for new episode."""
        pass  # Stateless within episode

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "overnight_penalty": self._overnight_penalty,
            "weekend_penalty": self._weekend_penalty,
            "holiday_penalty": self._holiday_penalty,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "gap_overnight_count": self._overnight_count,
            "gap_weekend_count": self._weekend_count,
            "gap_holiday_count": self._holiday_count,
        })
        return stats


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["HoldingDecay", "GapRiskPenalty"]
