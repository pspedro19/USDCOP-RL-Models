"""
Reward Component Base Classes (DRY).

Provides abstract base classes and protocols for all reward components.
Ensures consistent interface across DSR, Sortino, regime detection, etc.

Contract: CTR-REWARD-BASE-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable
import numpy as np


# =============================================================================
# ENUMS
# =============================================================================


class MarketRegime(Enum):
    """Market regime classification."""
    LOW_VOL = "LOW_VOL"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOL"
    CRISIS = "CRISIS"


class ComponentType(Enum):
    """Type of reward component."""
    RISK_METRIC = "risk_metric"      # DSR, Sortino, etc.
    PENALTY = "penalty"              # Costs, decay, anti-gaming
    BONUS = "bonus"                  # Consistency, regime bonuses
    DETECTOR = "detector"            # Regime, intervention detection
    NORMALIZER = "normalizer"        # Reward normalization
    TRANSFORM = "transform"          # PnL transforms


# =============================================================================
# BASE DATACLASSES
# =============================================================================


@dataclass
class ComponentResult:
    """Result from a reward component calculation."""
    value: float
    component_name: str
    component_type: ComponentType
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "value": self.value,
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "details": self.details,
        }


@dataclass
class ComponentStats:
    """Statistics for a reward component."""
    call_count: int = 0
    sum_value: float = 0.0
    min_value: float = float('inf')
    max_value: float = float('-inf')
    last_value: float = 0.0

    def update(self, value: float) -> None:
        """Update statistics with new value."""
        self.call_count += 1
        self.sum_value += value
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)
        self.last_value = value

    @property
    def mean_value(self) -> float:
        """Average value."""
        if self.call_count == 0:
            return 0.0
        return self.sum_value / self.call_count

    def reset(self) -> None:
        """Reset statistics."""
        self.call_count = 0
        self.sum_value = 0.0
        self.min_value = float('inf')
        self.max_value = float('-inf')
        self.last_value = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "call_count": self.call_count,
            "sum_value": self.sum_value,
            "mean_value": self.mean_value,
            "min_value": self.min_value if self.min_value != float('inf') else None,
            "max_value": self.max_value if self.max_value != float('-inf') else None,
            "last_value": self.last_value,
        }


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================


class RewardComponent(ABC):
    """
    Abstract base class for all reward components (DRY principle).

    All reward components must implement:
    - name: Component identifier
    - component_type: Type classification
    - calculate(): Core calculation logic
    - reset(): State reset for new episode

    Optional overrides:
    - get_stats(): Component statistics
    - get_config(): Configuration dict
    """

    def __init__(self):
        """Initialize component with statistics tracking."""
        self._stats = ComponentStats()
        self._enabled = True

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Component name for logging and identification.

        Returns:
            Unique string identifier (e.g., "dsr", "sortino", "market_impact")
        """
        pass

    @property
    @abstractmethod
    def component_type(self) -> ComponentType:
        """
        Component type classification.

        Returns:
            ComponentType enum value
        """
        pass

    @abstractmethod
    def calculate(self, **kwargs) -> float:
        """
        Calculate component contribution to reward.

        Args:
            **kwargs: Component-specific parameters

        Returns:
            Float value to add/subtract from reward
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """
        Reset internal state for new episode.

        Called at the start of each training episode.
        Should reset any accumulated state (e.g., running statistics).
        """
        pass

    @property
    def enabled(self) -> bool:
        """Whether component is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable/disable component."""
        self._enabled = value

    def get_stats(self) -> Dict[str, Any]:
        """
        Get component statistics for logging.

        Returns:
            Dictionary with statistics (call_count, mean, min, max, etc.)
        """
        return {
            f"{self.name}_stats": self._stats.to_dict(),
        }

    def get_config(self) -> Dict[str, Any]:
        """
        Get component configuration.

        Returns:
            Dictionary with component configuration
        """
        return {
            "name": self.name,
            "type": self.component_type.value,
            "enabled": self._enabled,
        }

    def _update_stats(self, value: float) -> None:
        """Update internal statistics (call from calculate())."""
        self._stats.update(value)

    def reset_stats(self) -> None:
        """Reset statistics (separate from episode reset)."""
        self._stats.reset()


# =============================================================================
# PROTOCOLS (Interface Segregation Principle)
# =============================================================================


@runtime_checkable
class IRewardCalculator(Protocol):
    """Protocol for reward calculators."""

    def calculate(
        self,
        pnl_pct: float,
        position_change: int,
        bars_held: int,
        current_drawdown: float,
        **kwargs
    ) -> Tuple[float, Any]:
        """Calculate total reward with breakdown."""
        ...

    def reset(self) -> None:
        """Reset for new episode."""
        ...


@runtime_checkable
class IRegimeDetector(Protocol):
    """Protocol for market regime detection."""

    def update(self, volatility: float, **kwargs) -> MarketRegime:
        """Update and return current regime."""
        ...

    def reset(self) -> None:
        """Reset detector state."""
        ...

    @property
    def current_regime(self) -> MarketRegime:
        """Get current detected regime."""
        ...


@runtime_checkable
class IRewardNormalizer(Protocol):
    """Protocol for reward normalization."""

    def normalize(self, reward: float, update_stats: bool = True) -> float:
        """Normalize reward value."""
        ...

    def reset(self) -> None:
        """Reset normalizer state."""
        ...


@runtime_checkable
class ICostModel(Protocol):
    """Protocol for transaction cost models."""

    def calculate_cost(
        self,
        hour_utc: int,
        regime: str,
        volatility: float,
        **kwargs
    ) -> float:
        """Calculate transaction cost."""
        ...


@runtime_checkable
class IDecayModel(Protocol):
    """Protocol for time/holding decay models."""

    def calculate_decay(self, bars_held: int, **kwargs) -> float:
        """Calculate decay penalty."""
        ...

    def reset(self) -> None:
        """Reset decay state."""
        ...


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def clip_reward(
    reward: float,
    min_reward: float = -1.0,
    max_reward: float = 1.0,
) -> float:
    """
    Clip reward to valid range.

    Args:
        reward: Raw reward value
        min_reward: Minimum allowed value
        max_reward: Maximum allowed value

    Returns:
        Clipped reward value
    """
    return float(np.clip(reward, min_reward, max_reward))


def safe_divide(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    epsilon: float = 1e-8,
) -> float:
    """
    Safe division with epsilon to avoid divide by zero.

    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if denominator is ~0
        epsilon: Small value for stability

    Returns:
        Division result or default
    """
    if abs(denominator) < epsilon:
        return default
    return numerator / denominator


def exponential_decay(
    base_rate: float,
    decay_factor: float,
    steps: int,
    max_value: float = 1.0,
) -> float:
    """
    Calculate exponential decay penalty.

    penalty = base_rate * (decay_factor ^ steps - 1)

    Args:
        base_rate: Base penalty rate
        decay_factor: Exponential growth factor (e.g., 1.02 = 2% per step)
        steps: Number of steps held
        max_value: Maximum penalty cap

    Returns:
        Decay penalty value
    """
    penalty = base_rate * (decay_factor ** steps - 1)
    return min(penalty, max_value)


def z_score(value: float, mean: float, std: float, epsilon: float = 1e-8) -> float:
    """
    Calculate z-score.

    Args:
        value: Raw value
        mean: Mean for normalization
        std: Standard deviation
        epsilon: Stability constant

    Returns:
        Z-score
    """
    return (value - mean) / (std + epsilon)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MarketRegime",
    "ComponentType",
    # Dataclasses
    "ComponentResult",
    "ComponentStats",
    # Base class
    "RewardComponent",
    # Protocols
    "IRewardCalculator",
    "IRegimeDetector",
    "IRewardNormalizer",
    "ICostModel",
    "IDecayModel",
    # Helpers
    "clip_reward",
    "safe_divide",
    "exponential_decay",
    "z_score",
]
