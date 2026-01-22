"""
Sortino Ratio Component.

Implements rolling Sortino ratio calculation focusing on downside risk.
Sortino penalizes only negative returns, unlike Sharpe which penalizes all volatility.

Contract: CTR-REWARD-SORTINO-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional

from .base import RewardComponent, ComponentType, safe_divide


class SortinoCalculator(RewardComponent):
    """
    Rolling Sortino Ratio calculator.

    Sortino Ratio = (Mean Return - Target Return) / Downside Deviation

    Where Downside Deviation only considers returns below target.
    This is more appropriate for trading where we care about downside risk,
    not upside volatility.

    Reference:
        Sortino, F. A., & Van Der Meer, R. (1991).
        Downside risk. Journal of Portfolio Management, 17(4), 27-31.
    """

    def __init__(
        self,
        window_size: int = 20,
        target_return: float = 0.0,
        min_samples: int = 5,
        scale: float = 1.0,
        annualization_factor: float = 1.0,
    ):
        """
        Initialize Sortino calculator.

        Args:
            window_size: Rolling window for calculation
            target_return: Minimum acceptable return (usually 0 or risk-free rate)
            min_samples: Minimum samples before producing signal
            scale: Output scaling factor
            annualization_factor: Factor for annualizing (1.0 = no annualization)
        """
        super().__init__()
        self._window_size = window_size
        self._target_return = target_return
        self._min_samples = min_samples
        self._scale = scale
        self._annualization_factor = annualization_factor

        # State
        self._returns: deque = deque(maxlen=window_size)

    @property
    def name(self) -> str:
        return "sortino"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.RISK_METRIC

    def calculate(
        self,
        return_pct: float,
        **kwargs
    ) -> float:
        """
        Calculate Sortino contribution.

        Args:
            return_pct: Return percentage for current step

        Returns:
            Normalized Sortino signal
        """
        if not self._enabled:
            return 0.0

        self._returns.append(return_pct)

        # Need minimum samples
        if len(self._returns) < self._min_samples:
            return 0.0

        # Calculate Sortino
        returns_array = np.array(self._returns)
        mean_return = np.mean(returns_array)

        # Downside returns (only those below target)
        downside_returns = returns_array[returns_array < self._target_return]

        if len(downside_returns) == 0:
            # No downside = infinite Sortino, cap it
            sortino = 2.0  # Cap at reasonable value
        else:
            # Downside deviation
            downside_sq = (downside_returns - self._target_return) ** 2
            downside_std = np.sqrt(np.mean(downside_sq))

            # Sortino ratio
            excess_return = mean_return - self._target_return
            sortino = safe_divide(excess_return, downside_std, default=0.0)

            # Apply annualization
            sortino *= np.sqrt(self._annualization_factor)

        # Normalize with tanh for bounded output
        result = np.tanh(sortino / 2.0) * self._scale

        self._update_stats(result)
        return float(result)

    def update(self, return_pct: float) -> float:
        """Alias for calculate() for backward compatibility."""
        return self.calculate(return_pct=return_pct)

    def reset(self) -> None:
        """Reset state for new episode."""
        self._returns.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "window_size": self._window_size,
            "target_return": self._target_return,
            "min_samples": self._min_samples,
            "scale": self._scale,
            "annualization_factor": self._annualization_factor,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()

        if len(self._returns) >= self._min_samples:
            returns_array = np.array(self._returns)
            downside = returns_array[returns_array < self._target_return]

            stats.update({
                "sortino_mean_return": float(np.mean(returns_array)),
                "sortino_downside_count": len(downside),
                "sortino_raw_ratio": self._get_raw_sortino(),
                "sortino_sample_count": len(self._returns),
            })

        return stats

    def _get_raw_sortino(self) -> float:
        """Calculate raw (unnormalized) Sortino ratio."""
        if len(self._returns) < self._min_samples:
            return 0.0

        returns_array = np.array(self._returns)
        mean_return = np.mean(returns_array)
        downside_returns = returns_array[returns_array < self._target_return]

        if len(downside_returns) == 0:
            return float('inf')

        downside_sq = (downside_returns - self._target_return) ** 2
        downside_std = np.sqrt(np.mean(downside_sq))

        excess_return = mean_return - self._target_return
        return safe_divide(excess_return, downside_std, default=0.0)

    @property
    def raw_sortino(self) -> float:
        """Current raw Sortino ratio."""
        return self._get_raw_sortino()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["SortinoCalculator"]
