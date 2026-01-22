"""
Anti-Gaming Components.

Components to prevent RL agents from "gaming" the reward system
through degenerate behaviors:

1. InactivityTracker: Penalizes excessive flatness (no trading)
2. ChurnTracker: Penalizes excessive trading (churning)
3. ActionCorrelationTracker: Detects action sequence gaming
4. BiasDetector: Detects directional bias (always long/short)

Contract: CTR-REWARD-ANTIGAMING-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Background:
    RL agents can learn to exploit reward structure through:
    - Never trading (if costs dominate)
    - Trading every bar (if base reward is positive)
    - Alternating actions to maximize some metric
    - Strong directional bias

These components detect and penalize such behaviors.
"""

import numpy as np
from collections import deque, Counter
from typing import Dict, Any, List, Optional
from enum import Enum

from .base import RewardComponent, ComponentType


class InactivityTracker(RewardComponent):
    """
    Tracks and penalizes excessive inactivity.

    Detects when agent is "flat" (no position) for too long,
    which may indicate the agent learned to avoid all trading
    to minimize transaction costs.

    Applies graduated penalty:
    - First N bars: no penalty (reasonable waiting)
    - After N bars: increasing penalty

    Usage:
        >>> tracker = InactivityTracker(grace_period=12)
        >>> penalty = tracker.calculate(position=0)  # Flat
        >>> # After 12 bars flat, penalty starts
    """

    def __init__(
        self,
        grace_period: int = 12,
        max_penalty: float = 0.2,
        penalty_growth_rate: float = 0.01,
    ):
        """
        Initialize inactivity tracker.

        Args:
            grace_period: Bars of flatness before penalty starts
            max_penalty: Maximum penalty value
            penalty_growth_rate: Penalty increase per bar after grace
        """
        super().__init__()
        self._grace_period = grace_period
        self._max_penalty = max_penalty
        self._penalty_growth_rate = penalty_growth_rate

        # State
        self._flat_bars = 0
        self._total_flat_bars = 0
        self._inactivity_events = 0

    @property
    def name(self) -> str:
        return "inactivity"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        position: int = 0,
        **kwargs
    ) -> float:
        """
        Calculate inactivity penalty.

        Args:
            position: Current position (-1, 0, 1)

        Returns:
            Penalty value (negative if inactive too long)
        """
        if not self._enabled:
            return 0.0

        if position == 0:
            self._flat_bars += 1
            self._total_flat_bars += 1
        else:
            # Reset on position entry
            if self._flat_bars > self._grace_period:
                self._inactivity_events += 1
            self._flat_bars = 0
            return 0.0

        # No penalty during grace period
        if self._flat_bars <= self._grace_period:
            return 0.0

        # Calculate graduated penalty
        excess_bars = self._flat_bars - self._grace_period
        penalty = min(excess_bars * self._penalty_growth_rate, self._max_penalty)

        result = -penalty

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset state for new episode."""
        self._flat_bars = 0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "grace_period": self._grace_period,
            "max_penalty": self._max_penalty,
            "penalty_growth_rate": self._penalty_growth_rate,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "inactivity_current_flat_bars": self._flat_bars,
            "inactivity_total_flat_bars": self._total_flat_bars,
            "inactivity_events": self._inactivity_events,
        })
        return stats

    @property
    def flat_bars(self) -> int:
        """Current consecutive flat bars."""
        return self._flat_bars


class ChurnTracker(RewardComponent):
    """
    Tracks and penalizes excessive trading (churning).

    Detects when agent trades too frequently, which may indicate:
    - Attempting to game the reward structure
    - Poor exploration leading to unstable policy
    - Overfitting to noise

    Uses rolling window to track trade frequency.

    Usage:
        >>> tracker = ChurnTracker(window=20, max_trades=10)
        >>> penalty = tracker.calculate(action_is_trade=True)
    """

    def __init__(
        self,
        window_size: int = 20,
        max_trades_in_window: int = 10,
        base_penalty: float = 0.1,
        excess_trade_penalty: float = 0.02,
    ):
        """
        Initialize churn tracker.

        Args:
            window_size: Rolling window for counting trades
            max_trades_in_window: Max trades before penalty applies
            base_penalty: Base penalty when threshold exceeded
            excess_trade_penalty: Additional penalty per excess trade
        """
        super().__init__()
        self._window_size = window_size
        self._max_trades = max_trades_in_window
        self._base_penalty = base_penalty
        self._excess_trade_penalty = excess_trade_penalty

        # State: deque of booleans (True = traded)
        self._trade_history: deque = deque(maxlen=window_size)
        self._total_trades = 0

    @property
    def name(self) -> str:
        return "churn"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        action_is_trade: bool = False,
        **kwargs
    ) -> float:
        """
        Calculate churn penalty.

        Args:
            action_is_trade: Whether this action opened/closed position

        Returns:
            Penalty value (negative if churning)
        """
        if not self._enabled:
            return 0.0

        self._trade_history.append(action_is_trade)
        if action_is_trade:
            self._total_trades += 1

        # Count trades in window
        trades_in_window = sum(self._trade_history)

        # No penalty if under threshold
        if trades_in_window <= self._max_trades:
            return 0.0

        # Calculate penalty
        excess_trades = trades_in_window - self._max_trades
        penalty = self._base_penalty + (excess_trades * self._excess_trade_penalty)

        result = -penalty

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset state for new episode."""
        self._trade_history.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "window_size": self._window_size,
            "max_trades_in_window": self._max_trades,
            "base_penalty": self._base_penalty,
            "excess_trade_penalty": self._excess_trade_penalty,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "churn_trades_in_window": sum(self._trade_history),
            "churn_total_trades": self._total_trades,
            "churn_rate": (
                sum(self._trade_history) / len(self._trade_history)
                if len(self._trade_history) > 0 else 0.0
            ),
        })
        return stats

    @property
    def current_churn_rate(self) -> float:
        """Current trade frequency in window."""
        if len(self._trade_history) == 0:
            return 0.0
        return sum(self._trade_history) / len(self._trade_history)


class ActionCorrelationTracker(RewardComponent):
    """
    Detects suspicious action correlations.

    Watches for patterns like:
    - Alternating buy/sell (gaming some metric)
    - Repeating action sequences
    - Actions correlated with irrelevant features

    High correlation with predictable patterns = gaming.

    Usage:
        >>> tracker = ActionCorrelationTracker()
        >>> penalty = tracker.calculate(action=1, prev_action=-1)
    """

    def __init__(
        self,
        window_size: int = 20,
        alternation_threshold: float = 0.8,
        alternation_penalty: float = 0.15,
        repetition_penalty: float = 0.1,
    ):
        """
        Initialize action correlation tracker.

        Args:
            window_size: Window for detecting patterns
            alternation_threshold: Fraction of alternations to trigger
            alternation_penalty: Penalty for detected alternation
            repetition_penalty: Penalty for detected repetition
        """
        super().__init__()
        self._window_size = window_size
        self._alternation_threshold = alternation_threshold
        self._alternation_penalty = alternation_penalty
        self._repetition_penalty = repetition_penalty

        # State
        self._action_history: deque = deque(maxlen=window_size)
        self._alternation_count = 0
        self._pattern_detections = 0

    @property
    def name(self) -> str:
        return "action_correlation"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        action: int,
        **kwargs
    ) -> float:
        """
        Calculate action correlation penalty.

        Args:
            action: Current action (-1, 0, 1)

        Returns:
            Penalty value (negative if gaming detected)
        """
        if not self._enabled:
            return 0.0

        # Store action
        self._action_history.append(action)

        if len(self._action_history) < 4:
            return 0.0

        # Check for alternation pattern
        actions = list(self._action_history)
        alternations = 0
        for i in range(1, len(actions)):
            if actions[i] != actions[i-1] and actions[i] != 0 and actions[i-1] != 0:
                alternations += 1

        non_flat_transitions = sum(
            1 for i in range(1, len(actions))
            if actions[i] != 0 or actions[i-1] != 0
        )

        if non_flat_transitions == 0:
            return 0.0

        alternation_rate = alternations / non_flat_transitions

        penalty = 0.0

        # Detect alternation gaming
        if alternation_rate > self._alternation_threshold:
            penalty = self._alternation_penalty
            self._pattern_detections += 1

        # Detect strict repetition (same action many times)
        action_counts = Counter(actions)
        most_common_count = action_counts.most_common(1)[0][1]
        repetition_rate = most_common_count / len(actions)

        if repetition_rate > 0.9:  # >90% same action
            penalty = max(penalty, self._repetition_penalty)
            self._pattern_detections += 1

        result = -penalty

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset state for new episode."""
        self._action_history.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "window_size": self._window_size,
            "alternation_threshold": self._alternation_threshold,
            "alternation_penalty": self._alternation_penalty,
            "repetition_penalty": self._repetition_penalty,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        if len(self._action_history) > 0:
            action_counts = Counter(self._action_history)
            stats.update({
                "action_correlation_detections": self._pattern_detections,
                "action_distribution": dict(action_counts),
            })
        return stats


class BiasDetector(RewardComponent):
    """
    Detects directional bias in trading.

    Tracks long vs short position time to detect if agent
    has learned a degenerate "always long" or "always short" policy.

    A healthy agent should have balanced directional exposure
    (unless market has strong trend, which should be detected separately).

    Usage:
        >>> detector = BiasDetector(imbalance_threshold=0.8)
        >>> penalty = detector.calculate(position=1)
        >>> # If >80% of time in one direction, penalty applies
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.75,
        bias_penalty: float = 0.1,
        min_samples: int = 50,
    ):
        """
        Initialize bias detector.

        Args:
            imbalance_threshold: Threshold for detecting bias (0.75 = 75%)
            bias_penalty: Penalty for detected bias
            min_samples: Minimum samples before detection
        """
        super().__init__()
        self._imbalance_threshold = imbalance_threshold
        self._bias_penalty = bias_penalty
        self._min_samples = min_samples

        # State
        self._long_bars = 0
        self._short_bars = 0
        self._flat_bars = 0
        self._total_bars = 0
        self._bias_detected = False

    @property
    def name(self) -> str:
        return "bias_detector"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        position: int = 0,
        **kwargs
    ) -> float:
        """
        Calculate bias penalty.

        Args:
            position: Current position (-1, 0, 1)

        Returns:
            Penalty value (negative if biased)
        """
        if not self._enabled:
            return 0.0

        # Update counts
        self._total_bars += 1
        if position > 0:
            self._long_bars += 1
        elif position < 0:
            self._short_bars += 1
        else:
            self._flat_bars += 1

        # Need minimum samples
        if self._total_bars < self._min_samples:
            return 0.0

        # Calculate directional imbalance (excluding flat)
        directional_bars = self._long_bars + self._short_bars
        if directional_bars == 0:
            # All flat - handled by InactivityTracker
            return 0.0

        long_ratio = self._long_bars / directional_bars
        short_ratio = self._short_bars / directional_bars

        # Detect bias
        max_ratio = max(long_ratio, short_ratio)
        self._bias_detected = max_ratio > self._imbalance_threshold

        if not self._bias_detected:
            return 0.0

        # Graduated penalty based on severity
        excess = max_ratio - self._imbalance_threshold
        penalty = self._bias_penalty * (1 + excess * 2)

        result = -penalty

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset state for new episode."""
        self._long_bars = 0
        self._short_bars = 0
        self._flat_bars = 0
        self._total_bars = 0
        self._bias_detected = False

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "imbalance_threshold": self._imbalance_threshold,
            "bias_penalty": self._bias_penalty,
            "min_samples": self._min_samples,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()

        directional_bars = self._long_bars + self._short_bars
        long_ratio = self._long_bars / directional_bars if directional_bars > 0 else 0.0
        short_ratio = self._short_bars / directional_bars if directional_bars > 0 else 0.0

        stats.update({
            "bias_long_bars": self._long_bars,
            "bias_short_bars": self._short_bars,
            "bias_flat_bars": self._flat_bars,
            "bias_long_ratio": long_ratio,
            "bias_short_ratio": short_ratio,
            "bias_detected": self._bias_detected,
            "bias_direction": "LONG" if long_ratio > short_ratio else "SHORT",
        })
        return stats

    @property
    def is_biased(self) -> bool:
        """Check if bias is currently detected."""
        return self._bias_detected

    @property
    def dominant_direction(self) -> str:
        """Get the dominant trading direction."""
        if self._long_bars > self._short_bars:
            return "LONG"
        elif self._short_bars > self._long_bars:
            return "SHORT"
        return "BALANCED"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "InactivityTracker",
    "ChurnTracker",
    "ActionCorrelationTracker",
    "BiasDetector",
]
