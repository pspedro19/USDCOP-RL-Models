"""
Oil Correlation Tracker Component.

Tracks the correlation between WTI oil prices and USD/COP.
Colombia is a major oil exporter, so oil prices significantly
impact the Colombian Peso.

Contract: CTR-REWARD-OIL-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Background:
    - Colombia exports ~800k bbl/day of crude oil
    - Oil revenue accounts for ~40% of exports
    - Higher oil prices → stronger COP (lower USD/COP)
    - Correlation typically ranges from -0.3 to -0.7

When oil correlation breaks down, it may signal:
    1. Other macro factors dominating
    2. Local political events
    3. Central bank intervention
    4. Regime change

This component provides:
    - Rolling correlation tracking
    - Correlation breakdown detection
    - Signal for potential regime shifts
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from .base import RewardComponent, ComponentType


class OilCorrelationState(Enum):
    """State of oil-COP correlation."""
    STRONG_NEGATIVE = "STRONG_NEGATIVE"  # Normal: high oil → strong COP
    WEAK = "WEAK"                         # Correlation breaking down
    POSITIVE = "POSITIVE"                 # Anomaly: positive correlation
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class OilCorrelationTracker(RewardComponent):
    """
    Tracks rolling correlation between oil returns and USD/COP returns.

    Typical behavior:
    - Negative correlation (oil up → COP up → USD/COP down)
    - Correlation breakdown signals potential regime change

    When correlation weakens significantly, applies penalty to discourage
    trading during unstable correlation regime.

    Usage:
        >>> tracker = OilCorrelationTracker(window=20)
        >>> status = tracker.update(oil_return=0.02, usdcop_return=-0.01)
        >>> if status == OilCorrelationState.WEAK:
        ...     # Be cautious - normal relationship broken
        ...     penalty = tracker.get_penalty()
    """

    # Expected negative correlation
    EXPECTED_CORRELATION = -0.5

    def __init__(
        self,
        window_size: int = 20,
        strong_threshold: float = -0.3,
        weak_threshold: float = -0.1,
        breakdown_penalty: float = 0.1,
        min_samples: int = 10,
    ):
        """
        Initialize oil correlation tracker.

        Args:
            window_size: Rolling window for correlation
            strong_threshold: Correlation below this is "strong" (e.g., -0.3)
            weak_threshold: Correlation above this is "weak" (e.g., -0.1)
            breakdown_penalty: Penalty when correlation breaks down
            min_samples: Minimum samples before detecting correlation
        """
        super().__init__()
        self._window_size = window_size
        self._strong_threshold = strong_threshold
        self._weak_threshold = weak_threshold
        self._breakdown_penalty = breakdown_penalty
        self._min_samples = min_samples

        # State
        self._oil_returns: deque = deque(maxlen=window_size)
        self._usdcop_returns: deque = deque(maxlen=window_size)
        self._current_state = OilCorrelationState.INSUFFICIENT_DATA
        self._current_correlation = 0.0

        # Statistics
        self._breakdown_count = 0
        self._total_updates = 0

    @property
    def name(self) -> str:
        return "oil_correlation"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DETECTOR

    def calculate(
        self,
        oil_return: float,
        usdcop_return: float,
        **kwargs
    ) -> float:
        """
        Calculate oil correlation adjustment.

        Args:
            oil_return: WTI oil return (decimal)
            usdcop_return: USD/COP return (decimal)

        Returns:
            Penalty value (negative if correlation broken)
        """
        if not self._enabled:
            return 0.0

        self.update(oil_return, usdcop_return)

        # Apply penalty based on state
        if self._current_state == OilCorrelationState.WEAK:
            result = -self._breakdown_penalty * 0.5
        elif self._current_state == OilCorrelationState.POSITIVE:
            result = -self._breakdown_penalty  # Full penalty for anomaly
        else:
            result = 0.0

        self._update_stats(result)
        return result

    def update(
        self,
        oil_return: float,
        usdcop_return: float,
        **kwargs
    ) -> OilCorrelationState:
        """
        Update correlation tracker with new observation.

        Args:
            oil_return: WTI oil return
            usdcop_return: USD/COP return

        Returns:
            Current correlation state
        """
        self._total_updates += 1
        self._oil_returns.append(oil_return)
        self._usdcop_returns.append(usdcop_return)

        # Need minimum samples
        if len(self._oil_returns) < self._min_samples:
            self._current_state = OilCorrelationState.INSUFFICIENT_DATA
            return self._current_state

        # Calculate correlation
        oil_array = np.array(self._oil_returns)
        cop_array = np.array(self._usdcop_returns)

        # Avoid division by zero
        oil_std = np.std(oil_array)
        cop_std = np.std(cop_array)

        if oil_std < 1e-10 or cop_std < 1e-10:
            self._current_correlation = 0.0
            self._current_state = OilCorrelationState.WEAK
            return self._current_state

        # Pearson correlation
        self._current_correlation = np.corrcoef(oil_array, cop_array)[0, 1]

        # Handle NaN
        if np.isnan(self._current_correlation):
            self._current_correlation = 0.0
            self._current_state = OilCorrelationState.WEAK
            return self._current_state

        # Classify state
        prev_state = self._current_state

        if self._current_correlation < self._strong_threshold:
            self._current_state = OilCorrelationState.STRONG_NEGATIVE
        elif self._current_correlation > 0:
            self._current_state = OilCorrelationState.POSITIVE
        elif self._current_correlation > self._weak_threshold:
            self._current_state = OilCorrelationState.WEAK
        else:
            self._current_state = OilCorrelationState.STRONG_NEGATIVE

        # Track breakdowns
        if prev_state == OilCorrelationState.STRONG_NEGATIVE and \
           self._current_state in (OilCorrelationState.WEAK, OilCorrelationState.POSITIVE):
            self._breakdown_count += 1

        return self._current_state

    def get_penalty(self) -> float:
        """Get current penalty based on correlation state."""
        if self._current_state == OilCorrelationState.WEAK:
            return self._breakdown_penalty * 0.5
        elif self._current_state == OilCorrelationState.POSITIVE:
            return self._breakdown_penalty
        return 0.0

    def get_oil_signal(self, oil_return: float) -> float:
        """
        Get directional signal based on oil movement.

        Args:
            oil_return: Current oil return

        Returns:
            Expected USD/COP direction signal [-1, 1]
            Positive = expect USD/COP to rise
            Negative = expect USD/COP to fall
        """
        if self._current_state == OilCorrelationState.INSUFFICIENT_DATA:
            return 0.0

        # Normal: negative correlation → oil up means COP up (USD/COP down)
        if self._current_state == OilCorrelationState.STRONG_NEGATIVE:
            # Scale signal by correlation strength
            signal = -np.sign(oil_return) * min(abs(self._current_correlation), 1.0)
            return float(signal)

        # Correlation broken - no signal
        return 0.0

    def reset(self) -> None:
        """Reset state for new episode."""
        self._oil_returns.clear()
        self._usdcop_returns.clear()
        self._current_state = OilCorrelationState.INSUFFICIENT_DATA
        self._current_correlation = 0.0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "window_size": self._window_size,
            "strong_threshold": self._strong_threshold,
            "weak_threshold": self._weak_threshold,
            "breakdown_penalty": self._breakdown_penalty,
            "min_samples": self._min_samples,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "oil_correlation_current": self._current_correlation,
            "oil_correlation_state": self._current_state.value,
            "oil_breakdown_count": self._breakdown_count,
            "oil_sample_count": len(self._oil_returns),
            "oil_total_updates": self._total_updates,
        })
        return stats

    @property
    def correlation(self) -> float:
        """Current rolling correlation."""
        return self._current_correlation

    @property
    def state(self) -> OilCorrelationState:
        """Current correlation state."""
        return self._current_state

    @property
    def is_normal(self) -> bool:
        """Check if correlation is in normal (strong negative) range."""
        return self._current_state == OilCorrelationState.STRONG_NEGATIVE


class OilMomentumSignal(RewardComponent):
    """
    Generates trading signals based on oil momentum.

    When oil has strong momentum, and correlation is normal,
    provides a directional signal for USD/COP trading.

    This is used as a reward shaping term to encourage
    trades aligned with oil fundamentals.
    """

    def __init__(
        self,
        momentum_window: int = 5,
        signal_threshold: float = 0.02,
        alignment_bonus: float = 0.05,
        misalignment_penalty: float = 0.03,
    ):
        """
        Initialize oil momentum signal.

        Args:
            momentum_window: Window for momentum calculation
            signal_threshold: Minimum oil momentum to generate signal
            alignment_bonus: Bonus for trades aligned with oil signal
            misalignment_penalty: Penalty for trades against oil signal
        """
        super().__init__()
        self._momentum_window = momentum_window
        self._signal_threshold = signal_threshold
        self._alignment_bonus = alignment_bonus
        self._misalignment_penalty = misalignment_penalty

        # State
        self._oil_prices: deque = deque(maxlen=momentum_window + 1)
        self._current_momentum = 0.0

    @property
    def name(self) -> str:
        return "oil_momentum"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.SIGNAL

    def calculate(
        self,
        oil_price: float,
        position: int = 0,
        correlation_is_normal: bool = True,
        **kwargs
    ) -> float:
        """
        Calculate alignment bonus/penalty.

        Args:
            oil_price: Current WTI oil price
            position: Current position (-1, 0, 1)
            correlation_is_normal: Whether oil-COP correlation is normal

        Returns:
            Bonus (positive) or penalty (negative) for alignment
        """
        if not self._enabled:
            return 0.0

        # Update momentum
        self._oil_prices.append(oil_price)

        if len(self._oil_prices) < self._momentum_window:
            return 0.0

        # Calculate momentum (returns over window)
        prices = list(self._oil_prices)
        self._current_momentum = (prices[-1] - prices[0]) / prices[0]

        # No signal if correlation broken or flat
        if not correlation_is_normal or position == 0:
            return 0.0

        # No signal if momentum below threshold
        if abs(self._current_momentum) < self._signal_threshold:
            return 0.0

        # Oil up → COP up → USD/COP down → should be SHORT (position = -1)
        # Oil down → COP down → USD/COP up → should be LONG (position = 1)
        expected_position = -1 if self._current_momentum > 0 else 1

        if position == expected_position:
            result = self._alignment_bonus
        else:
            result = -self._misalignment_penalty

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset state for new episode."""
        self._oil_prices.clear()
        self._current_momentum = 0.0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "momentum_window": self._momentum_window,
            "signal_threshold": self._signal_threshold,
            "alignment_bonus": self._alignment_bonus,
            "misalignment_penalty": self._misalignment_penalty,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "oil_momentum_current": self._current_momentum,
            "oil_momentum_sample_count": len(self._oil_prices),
        })
        return stats

    @property
    def momentum(self) -> float:
        """Current oil momentum."""
        return self._current_momentum


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "OilCorrelationTracker",
    "OilCorrelationState",
    "OilMomentumSignal",
]
