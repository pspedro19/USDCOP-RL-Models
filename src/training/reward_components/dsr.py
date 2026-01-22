"""
Differential Sharpe Ratio (DSR) Component.

Implements incremental Sharpe Ratio calculation for RL rewards.
Based on Moody & Saffell (2001) "Learning to Trade via Direct Reinforcement".

Contract: CTR-REWARD-DSR-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19
"""

import numpy as np
from typing import Dict, Any, Optional

from .base import RewardComponent, ComponentType, safe_divide


class DifferentialSharpeRatio(RewardComponent):
    """
    Differential Sharpe Ratio calculator.

    DSR provides a gradient-friendly Sharpe ratio signal for RL training.
    Uses exponential moving averages for numerical stability.

    Formula:
        A_t = A_{t-1} + η * (R_t - A_{t-1})
        B_t = B_{t-1} + η * (R_t² - B_{t-1})
        DSR_t = (B_{t-1} * ΔA_t - 0.5 * A_{t-1} * ΔB_t) / (B_{t-1} - A_{t-1}²)^{3/2}

    Where:
        R_t = return at time t
        A_t = EMA of returns
        B_t = EMA of squared returns
        η = learning rate (adaptation speed)

    Reference:
        Moody, J., & Saffell, M. (2001). Learning to trade via direct reinforcement.
        IEEE transactions on neural Networks, 12(4), 875-889.
    """

    def __init__(
        self,
        eta: float = 0.01,
        min_samples: int = 10,
        scale: float = 1.0,
    ):
        """
        Initialize DSR calculator.

        Args:
            eta: Learning rate for EMA updates (0.01 = slow, 0.1 = fast)
            min_samples: Minimum samples before producing DSR signal
            scale: Output scaling factor
        """
        super().__init__()
        self._eta = eta
        self._min_samples = min_samples
        self._scale = scale

        # State variables
        self._A: float = 0.0  # EMA of returns
        self._B: float = 0.0  # EMA of squared returns
        self._count: int = 0

    @property
    def name(self) -> str:
        return "dsr"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.RISK_METRIC

    def calculate(
        self,
        return_pct: float,
        **kwargs
    ) -> float:
        """
        Calculate DSR contribution.

        Args:
            return_pct: Return percentage for current step

        Returns:
            DSR value (positive = improving Sharpe, negative = declining)
        """
        if not self._enabled:
            return 0.0

        self._count += 1

        # Store previous values
        A_prev = self._A
        B_prev = self._B

        # Update EMAs
        self._A = A_prev + self._eta * (return_pct - A_prev)
        self._B = B_prev + self._eta * (return_pct ** 2 - B_prev)

        # Need minimum samples for stability
        if self._count < self._min_samples:
            return 0.0

        # Calculate deltas
        delta_A = self._A - A_prev
        delta_B = self._B - B_prev

        # Variance term
        variance = B_prev - A_prev ** 2

        # Avoid numerical issues
        if variance <= 1e-10:
            dsr = 0.0
        else:
            # DSR formula
            numerator = B_prev * delta_A - 0.5 * A_prev * delta_B
            denominator = variance ** 1.5
            dsr = safe_divide(numerator, denominator, default=0.0)

        # Scale output
        result = dsr * self._scale

        # Clip to reasonable range
        result = float(np.clip(result, -1.0, 1.0))

        self._update_stats(result)
        return result

    def update(self, return_pct: float) -> float:
        """Alias for calculate() for backward compatibility."""
        return self.calculate(return_pct=return_pct)

    def reset(self) -> None:
        """Reset state for new episode."""
        self._A = 0.0
        self._B = 0.0
        self._count = 0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "eta": self._eta,
            "min_samples": self._min_samples,
            "scale": self._scale,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "dsr_A": self._A,
            "dsr_B": self._B,
            "dsr_count": self._count,
            "dsr_implied_sharpe": self._get_implied_sharpe(),
        })
        return stats

    def _get_implied_sharpe(self) -> float:
        """Calculate implied Sharpe ratio from current EMA values."""
        if self._count < self._min_samples:
            return 0.0

        variance = self._B - self._A ** 2
        if variance <= 1e-10:
            return 0.0

        std = np.sqrt(variance)
        return safe_divide(self._A, std, default=0.0)

    @property
    def implied_sharpe(self) -> float:
        """Current implied Sharpe ratio."""
        return self._get_implied_sharpe()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["DifferentialSharpeRatio"]
