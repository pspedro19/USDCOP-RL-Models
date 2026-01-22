"""
Market Regime Detector Component.

Detects market regime (LOW_VOL, NORMAL, HIGH_VOL, CRISIS) based on
volatility levels using percentile-based classification.

Contract: CTR-REWARD-REGIME-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple

from .base import (
    RewardComponent,
    ComponentType,
    MarketRegime,
    IRegimeDetector,
)


class StableRegimeDetector(RewardComponent):
    """
    Market regime detector with stability filtering.

    Uses volatility percentiles to classify market conditions:
    - LOW_VOL: volatility < 25th percentile
    - NORMAL: 25th - 75th percentile
    - HIGH_VOL: > 75th percentile
    - CRISIS: > 75th percentile * crisis_multiplier

    Includes stability filter to prevent rapid regime switching.

    Usage:
        >>> detector = StableRegimeDetector()
        >>> regime = detector.update(volatility=0.02)
        >>> print(regime)  # MarketRegime.NORMAL
    """

    def __init__(
        self,
        low_vol_percentile: int = 25,
        high_vol_percentile: int = 75,
        crisis_multiplier: float = 1.5,
        min_stability: int = 3,
        history_window: int = 500,
        smoothing_window: int = 5,
    ):
        """
        Initialize regime detector.

        Args:
            low_vol_percentile: Percentile threshold for LOW_VOL (default 25)
            high_vol_percentile: Percentile threshold for HIGH_VOL (default 75)
            crisis_multiplier: Multiplier above high_vol for CRISIS (default 1.5)
            min_stability: Minimum bars in regime before change (default 3)
            history_window: Window for percentile calculation (default 500)
            smoothing_window: Smoothing window for volatility (default 5)
        """
        super().__init__()
        self._low_vol_percentile = low_vol_percentile
        self._high_vol_percentile = high_vol_percentile
        self._crisis_multiplier = crisis_multiplier
        self._min_stability = min_stability
        self._history_window = history_window
        self._smoothing_window = smoothing_window

        # State
        self._volatility_history: deque = deque(maxlen=history_window)
        self._smoothing_buffer: deque = deque(maxlen=smoothing_window)
        self._current_regime = MarketRegime.NORMAL
        self._regime_bars = 0
        self._proposed_regime = MarketRegime.NORMAL
        self._proposed_bars = 0

        # Regime change tracking
        self._regime_changes = 0
        self._last_regime_change_bar = 0
        self._total_bars = 0

    @property
    def name(self) -> str:
        return "regime_detector"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.DETECTOR

    @property
    def current_regime(self) -> MarketRegime:
        """Get current detected regime."""
        return self._current_regime

    def calculate(
        self,
        volatility: float,
        **kwargs
    ) -> float:
        """
        Calculate regime penalty/bonus.

        Args:
            volatility: Current volatility measure (e.g., ATR %)

        Returns:
            Regime-based adjustment (negative for risky regimes)
        """
        if not self._enabled:
            return 0.0

        # Update regime
        self.update(volatility)

        # Return penalty based on regime
        if self._current_regime == MarketRegime.LOW_VOL:
            result = 0.05  # Small bonus for calm markets
        elif self._current_regime == MarketRegime.NORMAL:
            result = 0.0
        elif self._current_regime == MarketRegime.HIGH_VOL:
            result = -0.1  # Moderate penalty
        else:  # CRISIS
            result = -0.25  # Heavy penalty

        self._update_stats(result)
        return result

    def update(self, volatility: float, **kwargs) -> MarketRegime:
        """
        Update regime detection with new volatility observation.

        Args:
            volatility: Current volatility measure

        Returns:
            Current market regime
        """
        self._total_bars += 1

        # Add to smoothing buffer
        self._smoothing_buffer.append(volatility)

        # Use smoothed volatility
        smoothed_vol = np.mean(self._smoothing_buffer)
        self._volatility_history.append(smoothed_vol)

        # Need enough history for percentile calculation
        if len(self._volatility_history) < 50:
            return self._current_regime

        # Calculate thresholds
        vol_array = np.array(self._volatility_history)
        low_threshold = np.percentile(vol_array, self._low_vol_percentile)
        high_threshold = np.percentile(vol_array, self._high_vol_percentile)
        crisis_threshold = high_threshold * self._crisis_multiplier

        # Classify current volatility
        if smoothed_vol < low_threshold:
            detected_regime = MarketRegime.LOW_VOL
        elif smoothed_vol > crisis_threshold:
            detected_regime = MarketRegime.CRISIS
        elif smoothed_vol > high_threshold:
            detected_regime = MarketRegime.HIGH_VOL
        else:
            detected_regime = MarketRegime.NORMAL

        # Stability filter
        if detected_regime == self._proposed_regime:
            self._proposed_bars += 1
        else:
            self._proposed_regime = detected_regime
            self._proposed_bars = 1

        # Only change regime after stability period
        if self._proposed_bars >= self._min_stability:
            if self._proposed_regime != self._current_regime:
                self._regime_changes += 1
                self._last_regime_change_bar = self._total_bars
                self._current_regime = self._proposed_regime
                self._regime_bars = 0

        self._regime_bars += 1

        return self._current_regime

    def reset(self) -> None:
        """Reset state for new episode."""
        self._volatility_history.clear()
        self._smoothing_buffer.clear()
        self._current_regime = MarketRegime.NORMAL
        self._regime_bars = 0
        self._proposed_regime = MarketRegime.NORMAL
        self._proposed_bars = 0

    def get_regime_penalties(self) -> Dict[MarketRegime, float]:
        """Get default penalty multipliers for each regime."""
        return {
            MarketRegime.LOW_VOL: 0.0,
            MarketRegime.NORMAL: 0.0,
            MarketRegime.HIGH_VOL: 0.25,
            MarketRegime.CRISIS: 0.6,
        }

    def get_regime_cost_multipliers(self) -> Dict[MarketRegime, float]:
        """Get cost multipliers for each regime."""
        return {
            MarketRegime.LOW_VOL: 0.8,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOL: 1.3,
            MarketRegime.CRISIS: 2.0,
        }

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "low_vol_percentile": self._low_vol_percentile,
            "high_vol_percentile": self._high_vol_percentile,
            "crisis_multiplier": self._crisis_multiplier,
            "min_stability": self._min_stability,
            "history_window": self._history_window,
            "smoothing_window": self._smoothing_window,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "regime_current": self._current_regime.value,
            "regime_bars": self._regime_bars,
            "regime_changes": self._regime_changes,
            "regime_total_bars": self._total_bars,
            "regime_history_size": len(self._volatility_history),
        })

        if len(self._volatility_history) >= 50:
            vol_array = np.array(self._volatility_history)
            stats.update({
                "regime_vol_mean": float(np.mean(vol_array)),
                "regime_vol_std": float(np.std(vol_array)),
                "regime_vol_p25": float(np.percentile(vol_array, 25)),
                "regime_vol_p75": float(np.percentile(vol_array, 75)),
            })

        return stats

    def get_thresholds(self) -> Optional[Tuple[float, float, float]]:
        """
        Get current volatility thresholds.

        Returns:
            Tuple of (low_threshold, high_threshold, crisis_threshold)
            or None if not enough history
        """
        if len(self._volatility_history) < 50:
            return None

        vol_array = np.array(self._volatility_history)
        low = np.percentile(vol_array, self._low_vol_percentile)
        high = np.percentile(vol_array, self._high_vol_percentile)
        crisis = high * self._crisis_multiplier

        return (low, high, crisis)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["StableRegimeDetector", "MarketRegime"]
