"""
Execution Alpha Reward Component
==================================
Measures RL improvement over naive open-to-close execution.

When the forecasting pipeline provides a daily direction signal,
this component computes the alpha the RL agent generates by
optimizing intraday execution timing.

benchmark = forecast_direction * (current_price - day_open) / day_open
alpha = actual_rl_pnl - benchmark

3 cases:
1. RL traded and closed: alpha = rl_pnl - benchmark
2. RL has open position at day end: alpha = mark_to_market - benchmark
3. RL did NOT trade: penalty = -0.1 * |benchmark|

Contract: CTR-REWARD-EXEC-ALPHA-001
Version: 1.0.0
Date: 2026-02-15
"""

import logging
from typing import Any, Dict

from .base import RewardComponent, ComponentType

logger = logging.getLogger(__name__)


class ExecutionAlphaComponent(RewardComponent):
    """Measures RL execution alpha vs naive forecast-following benchmark.

    The benchmark return is what a naive agent would earn by:
    - Going LONG at session open if forecast_direction = +1
    - Going SHORT at session open if forecast_direction = -1
    - Doing nothing if forecast_direction = 0

    The RL agent's actual PnL is compared to this benchmark.
    Positive alpha = RL adds value through timing. Negative = RL detracts.

    Args:
        penalty_no_trade: Penalty multiplier when RL doesn't trade but benchmark exists
        scale: Scaling factor for the alpha signal (default 100, similar to PnL scaling)
    """

    def __init__(
        self,
        penalty_no_trade: float = 0.1,
        scale: float = 100.0,
    ):
        super().__init__()
        self._penalty_no_trade = penalty_no_trade
        self._scale = scale
        self._cumulative_alpha = 0.0
        self._n_calculations = 0

    @property
    def name(self) -> str:
        return "execution_alpha"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.RISK_METRIC

    def calculate(
        self,
        rl_pnl_pct: float = 0.0,
        forecast_direction: int = 0,
        session_open_price: float = 0.0,
        current_price: float = 0.0,
        has_position: bool = False,
        is_day_end: bool = False,
        **kwargs,
    ) -> float:
        """Calculate execution alpha.

        Args:
            rl_pnl_pct: RL agent's actual PnL percentage this step
            forecast_direction: Daily forecast direction (-1, 0, +1)
            session_open_price: Price at session open
            current_price: Current bar's price
            has_position: Whether RL has an open position
            is_day_end: Whether this is the last bar of the session

        Returns:
            Scaled alpha value (positive = RL beats benchmark)
        """
        # No benchmark when forecast is neutral
        if forecast_direction == 0 or session_open_price <= 0:
            self._update_stats(0.0)
            return 0.0

        # Benchmark: naive open-to-current return * direction
        benchmark_pnl = forecast_direction * (current_price - session_open_price) / session_open_price

        if has_position:
            # Case 1 & 2: RL is trading → alpha = rl_pnl - benchmark
            alpha = rl_pnl_pct - benchmark_pnl
        else:
            # Case 3: RL didn't trade → penalty proportional to missed opportunity
            alpha = -self._penalty_no_trade * abs(benchmark_pnl)

        scaled_alpha = alpha * self._scale

        self._cumulative_alpha += alpha
        self._n_calculations += 1

        self._update_stats(scaled_alpha)
        return scaled_alpha

    def reset(self) -> None:
        """Reset for new episode."""
        self._cumulative_alpha = 0.0
        self._n_calculations = 0
        self._stats.reset()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "penalty_no_trade": self._penalty_no_trade,
            "scale": self._scale,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats["exec_alpha_cumulative"] = self._cumulative_alpha
        stats["exec_alpha_mean"] = (
            self._cumulative_alpha / max(self._n_calculations, 1)
        )
        stats["exec_alpha_n_calculations"] = self._n_calculations
        return stats
