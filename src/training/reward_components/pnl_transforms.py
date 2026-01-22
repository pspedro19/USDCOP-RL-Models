"""
PnL Transformation Components.

Transform raw PnL into more stable reward signals for RL training.
Raw PnL is often noisy and can lead to unstable training.

Contract: CTR-REWARD-PNLTRANSFORM-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Transforms available:
1. LogPnL: log(1 + pnl) for compressing outliers
2. AsymmetricPnL: Different scaling for wins vs losses
3. ClippedPnL: Hard clipping to prevent extreme values
4. RankPnL: Convert to percentile ranks (robust to outliers)
5. ZScorePnL: Normalize using rolling z-score
"""

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Callable

from .base import RewardComponent, ComponentType, safe_divide


class LogPnLTransform(RewardComponent):
    """
    Log transformation for PnL values.

    Formula: sign(pnl) * log(1 + |pnl| * scale)

    Properties:
    - Preserves sign
    - Compresses large positive/negative values
    - Near-linear for small values

    This is useful when PnL has occasional large outliers
    that could destabilize training.
    """

    def __init__(
        self,
        scale: float = 100.0,
        min_pnl: float = -0.1,
        max_pnl: float = 0.1,
    ):
        """
        Initialize log transform.

        Args:
            scale: Scaling factor before log (higher = more compression)
            min_pnl: Minimum PnL clip (before transform)
            max_pnl: Maximum PnL clip (before transform)
        """
        super().__init__()
        self._scale = scale
        self._min_pnl = min_pnl
        self._max_pnl = max_pnl

    @property
    def name(self) -> str:
        return "log_pnl"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.TRANSFORM

    def calculate(
        self,
        pnl: float,
        **kwargs
    ) -> float:
        """
        Apply log transformation to PnL.

        Args:
            pnl: Raw PnL value (as decimal, e.g., 0.01 = 1%)

        Returns:
            Transformed PnL value
        """
        if not self._enabled:
            return pnl

        # Clip extreme values first
        clipped = np.clip(pnl, self._min_pnl, self._max_pnl)

        # Log transform preserving sign
        sign = np.sign(clipped)
        magnitude = np.abs(clipped)
        transformed = sign * np.log(1.0 + magnitude * self._scale)

        self._update_stats(transformed)
        return float(transformed)

    def reset(self) -> None:
        """Reset state (stateless transform)."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "scale": self._scale,
            "min_pnl": self._min_pnl,
            "max_pnl": self._max_pnl,
        })
        return config


class AsymmetricPnLTransform(RewardComponent):
    """
    Asymmetric transformation for gains vs losses.

    Applies different multipliers:
    - Gains scaled by win_multiplier
    - Losses scaled by loss_multiplier

    This is based on prospect theory: losses hurt more than
    equivalent gains feel good. Setting loss_multiplier > win_multiplier
    makes the agent more loss-averse.

    Reference: Kahneman & Tversky (1979). Prospect Theory.
    """

    def __init__(
        self,
        win_multiplier: float = 1.0,
        loss_multiplier: float = 2.0,
        neutral_zone: float = 0.0001,
    ):
        """
        Initialize asymmetric transform.

        Args:
            win_multiplier: Multiplier for positive PnL
            loss_multiplier: Multiplier for negative PnL
            neutral_zone: Small PnL values treated as zero
        """
        super().__init__()
        self._win_multiplier = win_multiplier
        self._loss_multiplier = loss_multiplier
        self._neutral_zone = neutral_zone

    @property
    def name(self) -> str:
        return "asymmetric_pnl"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.TRANSFORM

    def calculate(
        self,
        pnl: float,
        **kwargs
    ) -> float:
        """
        Apply asymmetric transformation.

        Args:
            pnl: Raw PnL value

        Returns:
            Transformed PnL
        """
        if not self._enabled:
            return pnl

        # Neutral zone
        if abs(pnl) < self._neutral_zone:
            return 0.0

        # Apply asymmetric scaling
        if pnl > 0:
            transformed = pnl * self._win_multiplier
        else:
            transformed = pnl * self._loss_multiplier

        self._update_stats(transformed)
        return float(transformed)

    def reset(self) -> None:
        """Reset state (stateless transform)."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "win_multiplier": self._win_multiplier,
            "loss_multiplier": self._loss_multiplier,
            "neutral_zone": self._neutral_zone,
        })
        return config


class ClippedPnLTransform(RewardComponent):
    """
    Hard clipping transformation.

    Simply clips PnL to a fixed range.
    Useful when extreme outliers can destabilize training.
    """

    def __init__(
        self,
        min_value: float = -0.05,
        max_value: float = 0.05,
    ):
        """
        Initialize clipper.

        Args:
            min_value: Minimum allowed value
            max_value: Maximum allowed value
        """
        super().__init__()
        self._min_value = min_value
        self._max_value = max_value
        self._clip_count = 0

    @property
    def name(self) -> str:
        return "clipped_pnl"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.TRANSFORM

    def calculate(
        self,
        pnl: float,
        **kwargs
    ) -> float:
        """
        Apply clipping.

        Args:
            pnl: Raw PnL value

        Returns:
            Clipped PnL
        """
        if not self._enabled:
            return pnl

        if pnl < self._min_value or pnl > self._max_value:
            self._clip_count += 1

        clipped = float(np.clip(pnl, self._min_value, self._max_value))

        self._update_stats(clipped)
        return clipped

    def reset(self) -> None:
        """Reset state."""
        pass

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "min_value": self._min_value,
            "max_value": self._max_value,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "clip_count": self._clip_count,
        })
        return stats


class RankPnLTransform(RewardComponent):
    """
    Rank transformation - converts PnL to percentile rank.

    Properties:
    - Output is always in [0, 1] (or [-1, 1] if centered)
    - Robust to outliers
    - Non-parametric

    Useful when PnL distribution is unknown or has heavy tails.
    """

    def __init__(
        self,
        window_size: int = 100,
        center: bool = True,
        min_samples: int = 20,
    ):
        """
        Initialize rank transform.

        Args:
            window_size: Window for rank calculation
            center: If True, output in [-1, 1] instead of [0, 1]
            min_samples: Minimum samples before ranking
        """
        super().__init__()
        self._window_size = window_size
        self._center = center
        self._min_samples = min_samples

        # State
        self._pnl_history: deque = deque(maxlen=window_size)

    @property
    def name(self) -> str:
        return "rank_pnl"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.TRANSFORM

    def calculate(
        self,
        pnl: float,
        **kwargs
    ) -> float:
        """
        Calculate percentile rank of PnL.

        Args:
            pnl: Raw PnL value

        Returns:
            Percentile rank (0-1 or -1 to 1 if centered)
        """
        if not self._enabled:
            return pnl

        self._pnl_history.append(pnl)

        if len(self._pnl_history) < self._min_samples:
            return pnl  # Pass through until enough samples

        # Calculate rank
        history_array = np.array(self._pnl_history)
        rank = np.sum(history_array < pnl) / len(history_array)

        # Center to [-1, 1] if requested
        if self._center:
            rank = 2 * rank - 1

        self._update_stats(rank)
        return float(rank)

    def reset(self) -> None:
        """Reset state for new episode."""
        self._pnl_history.clear()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "window_size": self._window_size,
            "center": self._center,
            "min_samples": self._min_samples,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "rank_sample_count": len(self._pnl_history),
        })
        return stats


class ZScorePnLTransform(RewardComponent):
    """
    Z-score transformation using rolling statistics.

    Formula: (pnl - rolling_mean) / rolling_std

    Properties:
    - Centers around 0
    - Scales to unit variance
    - Adapts to changing market conditions

    This is the recommended transform for most RL applications.
    """

    def __init__(
        self,
        window_size: int = 100,
        min_samples: int = 20,
        clip_zscore: float = 3.0,
        decay: float = 0.99,
    ):
        """
        Initialize z-score transform.

        Args:
            window_size: Window for statistics
            min_samples: Minimum samples before z-scoring
            clip_zscore: Clip output to [-clip, +clip]
            decay: EMA decay for running statistics
        """
        super().__init__()
        self._window_size = window_size
        self._min_samples = min_samples
        self._clip_zscore = clip_zscore
        self._decay = decay

        # State - EMA statistics
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    @property
    def name(self) -> str:
        return "zscore_pnl"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.TRANSFORM

    def calculate(
        self,
        pnl: float,
        update_stats: bool = True,
        **kwargs
    ) -> float:
        """
        Calculate z-score of PnL.

        Args:
            pnl: Raw PnL value
            update_stats: Whether to update running statistics

        Returns:
            Z-score (clipped)
        """
        if not self._enabled:
            return pnl

        if update_stats:
            self._count += 1
            if self._count == 1:
                self._mean = pnl
                self._var = 0.0
            else:
                delta = pnl - self._mean
                self._mean = self._decay * self._mean + (1 - self._decay) * pnl
                self._var = self._decay * self._var + (1 - self._decay) * (delta ** 2)

        # Return raw during warmup
        if self._count < self._min_samples:
            return pnl

        # Calculate z-score
        std = np.sqrt(self._var + 1e-8)
        zscore = (pnl - self._mean) / std

        # Clip
        clipped = float(np.clip(zscore, -self._clip_zscore, self._clip_zscore))

        self._update_stats(clipped)
        return clipped

    def reset(self) -> None:
        """Reset state for new episode."""
        # Optionally keep statistics across episodes for stability
        pass

    def hard_reset(self) -> None:
        """Full reset of statistics."""
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "window_size": self._window_size,
            "min_samples": self._min_samples,
            "clip_zscore": self._clip_zscore,
            "decay": self._decay,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "zscore_running_mean": self._mean,
            "zscore_running_std": np.sqrt(self._var + 1e-8),
            "zscore_sample_count": self._count,
        })
        return stats


class CompositePnLTransform(RewardComponent):
    """
    Composite transform that chains multiple transforms.

    Example: Log → Asymmetric → Clip

    Order matters - transforms are applied sequentially.
    """

    def __init__(
        self,
        transforms: Optional[list] = None,
    ):
        """
        Initialize composite transform.

        Args:
            transforms: List of transform components to chain
        """
        super().__init__()
        self._transforms = transforms or []

    @property
    def name(self) -> str:
        return "composite_pnl"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.TRANSFORM

    def add_transform(self, transform: RewardComponent) -> None:
        """Add a transform to the chain."""
        self._transforms.append(transform)

    def calculate(
        self,
        pnl: float,
        **kwargs
    ) -> float:
        """
        Apply all transforms in sequence.

        Args:
            pnl: Raw PnL value

        Returns:
            Fully transformed PnL
        """
        if not self._enabled:
            return pnl

        result = pnl
        for transform in self._transforms:
            result = transform.calculate(result, **kwargs)

        self._update_stats(result)
        return result

    def reset(self) -> None:
        """Reset all transforms."""
        for transform in self._transforms:
            transform.reset()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "transforms": [t.name for t in self._transforms],
            "transform_configs": [t.get_config() for t in self._transforms],
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        for i, transform in enumerate(self._transforms):
            for key, value in transform.get_stats().items():
                stats[f"transform_{i}_{key}"] = value
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_default_pnl_transform() -> CompositePnLTransform:
    """
    Create the recommended default PnL transform chain.

    Returns:
        CompositePnLTransform with: Clip → ZScore → Asymmetric
    """
    composite = CompositePnLTransform()
    composite.add_transform(ClippedPnLTransform(min_value=-0.1, max_value=0.1))
    composite.add_transform(ZScorePnLTransform(window_size=100, clip_zscore=3.0))
    composite.add_transform(AsymmetricPnLTransform(win_multiplier=1.0, loss_multiplier=1.5))
    return composite


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "LogPnLTransform",
    "AsymmetricPnLTransform",
    "ClippedPnLTransform",
    "RankPnLTransform",
    "ZScorePnLTransform",
    "CompositePnLTransform",
    "create_default_pnl_transform",
]
