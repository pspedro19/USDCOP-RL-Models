"""
Market Impact Model (Almgren-Chriss, 2001).

Implements non-linear slippage based on:
- Permanent impact: sqrt(Q) * sigma
- Temporary impact: sqrt(Q/V)

Where Q = order size, V = volume, sigma = volatility

Contract: CTR-REWARD-IMPACT-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Reference:
    Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
    Journal of Risk, 3, 5-40.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

from .base import RewardComponent, ComponentType, ICostModel, MarketRegime


@dataclass
class MarketImpactResult:
    """Detailed result from market impact calculation."""
    total_cost_bps: float
    spread_cost_bps: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    volatility_impact_bps: float

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "total_cost_bps": self.total_cost_bps,
            "spread_cost_bps": self.spread_cost_bps,
            "permanent_impact_bps": self.permanent_impact_bps,
            "temporary_impact_bps": self.temporary_impact_bps,
            "volatility_impact_bps": self.volatility_impact_bps,
        }


class AlmgrenChrissImpactModel(RewardComponent):
    """
    Market impact model based on Almgren-Chriss (2001).

    Total Impact = spread + permanent + temporary + volatility

    Where:
        - spread = spread_base * regime_multiplier
        - permanent = η * sign(Q) * |Q/ADV|^0.5
        - temporary = γ * |Q/ADV|^0.5 / sqrt(T)
        - volatility = λ * σ * |Q/ADV|^0.5

    The sqrt scaling captures the empirical observation that market impact
    grows sublinearly with order size - doubling order size increases
    impact by ~1.4x, not 2x.

    USD/COP Specifics:
        - Spread varies by hour (wider at open/close)
        - Spread increases in HIGH_VOL/CRISIS regimes
        - Lower liquidity than major pairs → higher impact coefficients
    """

    # Default spread by hour (UTC) for USD/COP
    # Colombia trading hours: 8:00-13:00 COT = 13:00-18:00 UTC
    DEFAULT_SPREAD_BY_HOUR: Dict[int, int] = {
        13: 120,  # 8:00 COT - Pre-market, wide spread
        14: 90,   # 9:00 COT - Apertura
        15: 75,   # 10:00 COT - Peak liquidity
        16: 80,   # 11:00 COT
        17: 95,   # 12:00 COT
        18: 130,  # 13:00 COT - Close, wide spread
    }

    # Cost multipliers by regime
    DEFAULT_REGIME_MULTIPLIERS: Dict[str, float] = {
        "LOW_VOL": 0.8,
        "NORMAL": 1.0,
        "HIGH_VOL": 1.3,
        "CRISIS": 2.0,
    }

    def __init__(
        self,
        spread_by_hour: Optional[Dict[int, int]] = None,
        regime_multipliers: Optional[Dict[str, float]] = None,
        permanent_impact_coef: float = 0.1,
        temporary_impact_coef: float = 0.3,
        volatility_impact_coef: float = 0.15,
        adv_base_usd: float = 50_000_000.0,
        typical_order_fraction: float = 0.001,
        default_spread_bps: int = 100,
    ):
        """
        Initialize Almgren-Chriss impact model.

        Args:
            spread_by_hour: Spread in bps by hour UTC
            regime_multipliers: Cost multipliers by regime
            permanent_impact_coef: η - permanent impact coefficient
            temporary_impact_coef: γ - temporary impact coefficient
            volatility_impact_coef: λ - volatility sensitivity
            adv_base_usd: Average Daily Volume in USD (for USD/COP ~50M)
            typical_order_fraction: Typical order as fraction of ADV
            default_spread_bps: Default spread when hour not in table
        """
        super().__init__()

        self._spread_by_hour = spread_by_hour or self.DEFAULT_SPREAD_BY_HOUR.copy()
        self._regime_multipliers = regime_multipliers or self.DEFAULT_REGIME_MULTIPLIERS.copy()
        self._permanent_impact_coef = permanent_impact_coef
        self._temporary_impact_coef = temporary_impact_coef
        self._volatility_impact_coef = volatility_impact_coef
        self._adv_base_usd = adv_base_usd
        self._typical_order_fraction = typical_order_fraction
        self._default_spread_bps = default_spread_bps

        # Last calculation result for detailed analysis
        self._last_result: Optional[MarketImpactResult] = None

    @property
    def name(self) -> str:
        return "market_impact"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.PENALTY

    def calculate(
        self,
        hour_utc: int = 15,
        regime: str = "NORMAL",
        volatility: float = 0.01,
        order_size_fraction: Optional[float] = None,
        execution_time_bars: int = 1,
        **kwargs
    ) -> float:
        """
        Calculate total market impact as percentage cost.

        Args:
            hour_utc: Current hour in UTC
            regime: Market regime string
            volatility: Current volatility (decimal, e.g., 0.01 = 1%)
            order_size_fraction: Order size as fraction of ADV
            execution_time_bars: Bars to execute order

        Returns:
            Total cost as negative percentage (for reward subtraction)
        """
        if not self._enabled:
            return 0.0

        result = self.calculate_impact(
            hour_utc=hour_utc,
            regime=regime,
            volatility=volatility,
            order_size_fraction=order_size_fraction,
            execution_time_bars=execution_time_bars,
        )

        # Convert bps to percentage (negative for cost)
        cost_pct = -result.total_cost_bps / 10000.0

        self._update_stats(cost_pct)
        return cost_pct

    def calculate_impact(
        self,
        hour_utc: int,
        regime: str = "NORMAL",
        volatility: float = 0.01,
        order_size_fraction: Optional[float] = None,
        execution_time_bars: int = 1,
    ) -> MarketImpactResult:
        """
        Calculate detailed market impact breakdown.

        Args:
            hour_utc: Current hour in UTC
            regime: Market regime string
            volatility: Current volatility (decimal)
            order_size_fraction: Order size as fraction of ADV
            execution_time_bars: Bars to execute order

        Returns:
            MarketImpactResult with full cost breakdown
        """
        if order_size_fraction is None:
            order_size_fraction = self._typical_order_fraction

        # 1. Base spread by hour
        spread_bps = self._spread_by_hour.get(hour_utc, self._default_spread_bps)

        # 2. Regime multiplier
        regime_mult = self._regime_multipliers.get(regime, 1.0)
        spread_cost = spread_bps * regime_mult

        # 3. Permanent impact: η * sqrt(Q/ADV) - converts to bps
        sqrt_order = np.sqrt(order_size_fraction)
        permanent_impact = self._permanent_impact_coef * sqrt_order * 10000

        # 4. Temporary impact: γ * sqrt(Q/ADV) / sqrt(T)
        sqrt_time = np.sqrt(max(execution_time_bars, 1))
        temporary_impact = (self._temporary_impact_coef * sqrt_order / sqrt_time) * 10000

        # 5. Volatility impact: λ * σ * sqrt(Q/ADV)
        volatility_impact = self._volatility_impact_coef * volatility * sqrt_order * 10000

        # Total
        total_cost = spread_cost + permanent_impact + temporary_impact + volatility_impact

        self._last_result = MarketImpactResult(
            total_cost_bps=total_cost,
            spread_cost_bps=spread_cost,
            permanent_impact_bps=permanent_impact,
            temporary_impact_bps=temporary_impact,
            volatility_impact_bps=volatility_impact,
        )

        return self._last_result

    def calculate_cost_pct(
        self,
        hour_utc: int,
        regime: str = "NORMAL",
        volatility: float = 0.01,
    ) -> float:
        """
        Simplified method returning total cost as percentage.

        Args:
            hour_utc: Current hour in UTC
            regime: Market regime string
            volatility: Current volatility

        Returns:
            Total cost as positive percentage
        """
        result = self.calculate_impact(
            hour_utc=hour_utc,
            regime=regime,
            volatility=volatility,
        )
        return result.total_cost_bps / 10000.0

    def reset(self) -> None:
        """Reset state (stateless model, but interface compliance)."""
        self._last_result = None

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "permanent_impact_coef": self._permanent_impact_coef,
            "temporary_impact_coef": self._temporary_impact_coef,
            "volatility_impact_coef": self._volatility_impact_coef,
            "adv_base_usd": self._adv_base_usd,
            "typical_order_fraction": self._typical_order_fraction,
            "default_spread_bps": self._default_spread_bps,
            "spread_by_hour": self._spread_by_hour,
            "regime_multipliers": self._regime_multipliers,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()

        if self._last_result:
            stats.update({
                "impact_last_total_bps": self._last_result.total_cost_bps,
                "impact_last_spread_bps": self._last_result.spread_cost_bps,
                "impact_last_permanent_bps": self._last_result.permanent_impact_bps,
                "impact_last_temporary_bps": self._last_result.temporary_impact_bps,
                "impact_last_volatility_bps": self._last_result.volatility_impact_bps,
            })

        return stats

    @property
    def last_result(self) -> Optional[MarketImpactResult]:
        """Get last calculation result."""
        return self._last_result


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["AlmgrenChrissImpactModel", "MarketImpactResult"]
