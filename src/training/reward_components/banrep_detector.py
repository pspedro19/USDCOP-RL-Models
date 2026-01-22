"""
Banco de la República Intervention Detector.

Detects potential central bank interventions in USD/COP based on:
- Volatility spikes (z-score > threshold)
- Rapid price reversals (V-shape patterns)
- Abnormal spread widening

Contract: CTR-REWARD-BANREP-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Note: This is a proxy-based detector since actual intervention data
is not available in real-time. The Banrep typically intervenes to
defend the Colombian Peso during rapid depreciation.
"""

import numpy as np
from collections import deque
from enum import Enum
from typing import Dict, Any, Optional

from .base import RewardComponent, ComponentType


class InterventionStatus(Enum):
    """Status of potential Banrep intervention."""
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"


class BanrepInterventionDetector(RewardComponent):
    """
    Detector for Banco de la República interventions.

    The Banrep intervenes in the FX market through:
    1. Direct USD sales when COP depreciates rapidly
    2. Forward contracts
    3. Options auctions

    This detector uses proxy signals:
    1. Volatility spike (z-score > threshold)
    2. Price reversal after spike
    3. Abnormal intraday range

    When intervention is detected:
    - Apply penalty to discourage trading during uncertainty
    - Enter cooldown period

    Usage:
        >>> detector = BanrepInterventionDetector()
        >>> status = detector.update(volatility=0.05)
        >>> if status == InterventionStatus.CONFIRMED:
        ...     penalty = detector.get_penalty_multiplier()
    """

    def __init__(
        self,
        volatility_spike_zscore: float = 3.0,
        volatility_baseline_window: int = 100,
        intervention_penalty: float = 0.5,
        cooldown_bars: int = 24,
        reversal_threshold: float = 0.02,
        min_history: int = 50,
    ):
        """
        Initialize Banrep detector.

        Args:
            volatility_spike_zscore: Z-score threshold for spike detection
            volatility_baseline_window: Window for baseline volatility calculation
            intervention_penalty: Penalty multiplier during intervention
            cooldown_bars: Bars to remain in cooldown after intervention
            reversal_threshold: Price reversal threshold for confirmation
            min_history: Minimum history before detection activates
        """
        super().__init__()
        self._volatility_spike_zscore = volatility_spike_zscore
        self._volatility_baseline_window = volatility_baseline_window
        self._intervention_penalty = intervention_penalty
        self._cooldown_bars = cooldown_bars
        self._reversal_threshold = reversal_threshold
        self._min_history = min_history

        # State
        self._volatility_history: deque = deque(maxlen=volatility_baseline_window)
        self._price_history: deque = deque(maxlen=10)
        self._current_status = InterventionStatus.NORMAL
        self._cooldown_counter = 0
        self._intervention_count = 0
        self._last_spike_bar = 0
        self._total_bars = 0

    @property
    def name(self) -> str:
        return "banrep_detector"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DETECTOR

    def calculate(
        self,
        volatility: float,
        price_change: float = 0.0,
        **kwargs
    ) -> float:
        """
        Calculate intervention penalty.

        Args:
            volatility: Current volatility measure
            price_change: Current bar price change

        Returns:
            Penalty value (negative for risky conditions)
        """
        if not self._enabled:
            return 0.0

        self.update(volatility, price_change)

        penalty = self.get_penalty_multiplier()
        result = -penalty if penalty > 0 else 0.0

        self._update_stats(result)
        return result

    def update(
        self,
        volatility: float,
        price_change: float = 0.0,
        **kwargs
    ) -> InterventionStatus:
        """
        Update detector with new observation.

        Args:
            volatility: Current volatility (ATR or similar)
            price_change: Price change this bar (for reversal detection)

        Returns:
            Current intervention status
        """
        self._total_bars += 1
        self._volatility_history.append(volatility)
        self._price_history.append(price_change)

        # Handle cooldown
        if self._current_status == InterventionStatus.COOLDOWN:
            self._cooldown_counter -= 1
            if self._cooldown_counter <= 0:
                self._current_status = InterventionStatus.NORMAL
            return self._current_status

        # Need minimum history
        if len(self._volatility_history) < self._min_history:
            return self._current_status

        # Calculate volatility z-score
        vol_array = np.array(list(self._volatility_history)[:-1])  # Exclude current
        vol_mean = np.mean(vol_array)
        vol_std = np.std(vol_array) + 1e-8
        vol_zscore = (volatility - vol_mean) / vol_std

        # Detect spike
        if vol_zscore > self._volatility_spike_zscore:
            self._last_spike_bar = self._total_bars

            # Check for reversal pattern (V-shape)
            if len(self._price_history) >= 3:
                recent_prices = list(self._price_history)[-3:]
                # V-shape: down then up, or up then down
                is_reversal = (
                    (recent_prices[0] < 0 and recent_prices[-1] > 0) or
                    (recent_prices[0] > 0 and recent_prices[-1] < 0)
                )
                reversal_magnitude = abs(sum(recent_prices))

                if is_reversal and reversal_magnitude > self._reversal_threshold:
                    self._current_status = InterventionStatus.CONFIRMED
                    self._intervention_count += 1
                    self._cooldown_counter = self._cooldown_bars
                    self._current_status = InterventionStatus.COOLDOWN
                else:
                    self._current_status = InterventionStatus.SUSPECTED
            else:
                self._current_status = InterventionStatus.SUSPECTED

        elif self._current_status == InterventionStatus.SUSPECTED:
            # Clear suspicion if no spike in last 5 bars
            if self._total_bars - self._last_spike_bar > 5:
                self._current_status = InterventionStatus.NORMAL

        return self._current_status

    def get_penalty_multiplier(self) -> float:
        """
        Get penalty multiplier based on current status.

        Returns:
            Penalty as positive float (0.0 = no penalty)
        """
        if self._current_status in (InterventionStatus.CONFIRMED, InterventionStatus.COOLDOWN):
            return self._intervention_penalty
        elif self._current_status == InterventionStatus.SUSPECTED:
            return self._intervention_penalty * 0.5
        return 0.0

    def reset(self) -> None:
        """Reset state for new episode."""
        self._volatility_history.clear()
        self._price_history.clear()
        self._current_status = InterventionStatus.NORMAL
        self._cooldown_counter = 0
        self._last_spike_bar = 0
        self._total_bars = 0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "volatility_spike_zscore": self._volatility_spike_zscore,
            "volatility_baseline_window": self._volatility_baseline_window,
            "intervention_penalty": self._intervention_penalty,
            "cooldown_bars": self._cooldown_bars,
            "reversal_threshold": self._reversal_threshold,
            "min_history": self._min_history,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "banrep_status": self._current_status.value,
            "banrep_intervention_count": self._intervention_count,
            "banrep_cooldown_remaining": self._cooldown_counter,
            "banrep_total_bars": self._total_bars,
            "banrep_history_size": len(self._volatility_history),
        })

        if len(self._volatility_history) >= self._min_history:
            vol_array = np.array(self._volatility_history)
            stats.update({
                "banrep_vol_mean": float(np.mean(vol_array)),
                "banrep_vol_std": float(np.std(vol_array)),
            })

        return stats

    @property
    def intervention_count(self) -> int:
        """Total interventions detected."""
        return self._intervention_count

    @property
    def status(self) -> str:
        """Current status as string."""
        return self._current_status.value

    @property
    def is_intervention_active(self) -> bool:
        """Check if intervention is currently active."""
        return self._current_status in (
            InterventionStatus.CONFIRMED,
            InterventionStatus.COOLDOWN,
            InterventionStatus.SUSPECTED,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["BanrepInterventionDetector", "InterventionStatus"]
