"""
Reward Normalizer (FinRL-Meta Style).

Stabilizes PPO training using running mean/std normalization.
Based on FinRL-Meta's approach for financial RL.

Contract: CTR-REWARD-NORMALIZER-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Reference:
    Liu, X. Y., et al. (2022). FinRL-Meta: Market environments and benchmarks
    for data-driven financial reinforcement learning.
    Advances in Neural Information Processing Systems, 35.
"""

import numpy as np
from typing import Dict, Any, Optional

from .base import RewardComponent, ComponentType, IRewardNormalizer


class RunningMeanStd:
    """
    Running mean and standard deviation with exponential decay.

    Uses Welford's online algorithm with EMA for stability.
    """

    def __init__(self, decay: float = 0.99, epsilon: float = 1e-8):
        """
        Initialize running statistics.

        Args:
            decay: Decay factor for EMA (0.99 = slow adaptation)
            epsilon: Stability constant
        """
        self.decay = decay
        self.epsilon = epsilon
        self._mean = 0.0
        self._var = 1.0  # Start with unit variance
        self._count = 0

    def update(self, x: float) -> None:
        """
        Update statistics with new observation.

        Args:
            x: New value to incorporate
        """
        self._count += 1

        if self._count == 1:
            self._mean = x
            self._var = 0.0
        else:
            delta = x - self._mean
            # EMA update
            self._mean = self.decay * self._mean + (1 - self.decay) * x
            self._var = self.decay * self._var + (1 - self.decay) * (delta ** 2)

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def std(self) -> float:
        """Current running standard deviation."""
        return np.sqrt(self._var + self.epsilon)

    @property
    def var(self) -> float:
        """Current running variance."""
        return self._var

    @property
    def count(self) -> int:
        """Number of observations."""
        return self._count

    def reset(self) -> None:
        """Full reset of statistics."""
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "mean": self._mean,
            "std": self.std,
            "var": self._var,
            "count": self._count,
        }


class RewardNormalizer(RewardComponent):
    """
    Normalizes rewards using running mean/std.

    Formula:
        reward_normalized = (reward - running_mean) / (running_std + epsilon)

    This stabilizes PPO training by:
    1. Centering rewards around zero
    2. Scaling to unit variance
    3. Preventing reward scale drift

    Features:
    - Warmup period before normalizing
    - Clipping to prevent extreme values
    - Optional per-episode reset

    Reference: FinRL-Meta (Liu et al., 2022)
    """

    def __init__(
        self,
        decay: float = 0.99,
        epsilon: float = 1e-8,
        clip_range: float = 10.0,
        warmup_steps: int = 1000,
        per_episode_reset: bool = False,
    ):
        """
        Initialize reward normalizer.

        Args:
            decay: EMA decay factor (higher = slower adaptation)
            epsilon: Numerical stability constant
            clip_range: Clip normalized rewards to [-clip_range, clip_range]
            warmup_steps: Steps before normalization kicks in
            per_episode_reset: Whether to reset stats per episode
        """
        super().__init__()
        self._decay = decay
        self._epsilon = epsilon
        self._clip_range = clip_range
        self._warmup_steps = warmup_steps
        self._per_episode_reset = per_episode_reset

        self._running_stats = RunningMeanStd(decay=decay, epsilon=epsilon)

    @property
    def name(self) -> str:
        return "reward_normalizer"

    @property
    def component_type(self) -> ComponentType:
        return ComponentType.NORMALIZER

    def calculate(
        self,
        reward: float,
        update_stats: bool = True,
        **kwargs
    ) -> float:
        """
        Normalize reward value.

        Args:
            reward: Raw reward to normalize
            update_stats: Whether to update running statistics

        Returns:
            Normalized (and clipped) reward
        """
        if not self._enabled:
            return reward

        return self.normalize(reward, update_stats)

    def normalize(self, reward: float, update_stats: bool = True) -> float:
        """
        Normalize reward using running statistics.

        Args:
            reward: Raw reward value
            update_stats: Whether to update running mean/std

        Returns:
            Normalized reward value
        """
        if update_stats:
            self._running_stats.update(reward)

        # During warmup, return raw reward
        if self._running_stats.count < self._warmup_steps:
            return reward

        # Normalize
        normalized = (reward - self._running_stats.mean) / self._running_stats.std

        # Clip to prevent extreme values
        clipped = float(np.clip(normalized, -self._clip_range, self._clip_range))

        self._update_stats(clipped)
        return clipped

    def reset(self) -> None:
        """
        Reset for new episode.

        Only resets if per_episode_reset is True.
        Otherwise, maintains global statistics across episodes.
        """
        if self._per_episode_reset:
            self._running_stats.reset()

    def hard_reset(self) -> None:
        """Force full reset of all statistics."""
        self._running_stats.reset()

    def get_config(self) -> Dict[str, Any]:
        """Get component configuration."""
        config = super().get_config()
        config.update({
            "decay": self._decay,
            "epsilon": self._epsilon,
            "clip_range": self._clip_range,
            "warmup_steps": self._warmup_steps,
            "per_episode_reset": self._per_episode_reset,
        })
        return config

    def get_stats(self) -> Dict[str, Any]:
        """Get component statistics."""
        stats = super().get_stats()
        stats.update({
            "normalizer_mean": self._running_stats.mean,
            "normalizer_std": self._running_stats.std,
            "normalizer_count": self._running_stats.count,
            "normalizer_in_warmup": self._running_stats.count < self._warmup_steps,
        })
        return stats

    @property
    def stats(self) -> Dict[str, float]:
        """Convenience property for logging."""
        return {
            "reward_running_mean": self._running_stats.mean,
            "reward_running_std": self._running_stats.std,
            "reward_normalizer_count": self._running_stats.count,
        }

    @property
    def is_warmed_up(self) -> bool:
        """Check if warmup period is complete."""
        return self._running_stats.count >= self._warmup_steps


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["RewardNormalizer", "RunningMeanStd"]
