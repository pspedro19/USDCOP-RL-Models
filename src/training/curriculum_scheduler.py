"""
Curriculum Scheduler for Phased RL Training.

Implements curriculum learning for USD/COP trading:
- Phase 1: NORMAL regime only, reduced costs
- Phase 2: NORMAL + HIGH_VOL, moderate costs
- Phase 3: All regimes including CRISIS, full costs

This gradual exposure helps the agent learn stable behavior before
handling extreme market conditions.

Contract: CTR-CURRICULUM-SCHEDULER-001
Author: Trading Team
Version: 1.0.0
Created: 2026-01-19

Reference:
    Bengio, Y., et al. (2009). Curriculum learning.
    Proceedings of the 26th annual international conference on machine learning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.training.config import CurriculumConfig, RewardConfig
from src.training.reward_components.base import MarketRegime


class CurriculumPhase(Enum):
    """Training phase in curriculum."""
    PHASE_1 = "phase_1"  # NORMAL only
    PHASE_2 = "phase_2"  # NORMAL + HIGH_VOL
    PHASE_3 = "phase_3"  # All regimes


@dataclass
class PhaseConfig:
    """Configuration for a curriculum phase."""
    phase: CurriculumPhase
    name: str
    description: str

    # Regime filtering
    allowed_regimes: Tuple[MarketRegime, ...]

    # Cost scaling
    cost_multiplier: float

    # Transition criteria
    start_step: int
    end_step: int

    # Optional: Early transition criteria
    min_sharpe_for_advance: Optional[float] = None
    min_win_rate_for_advance: Optional[float] = None

    @property
    def is_regime_allowed(self) -> Callable[[MarketRegime], bool]:
        """Check if a regime is allowed in this phase."""
        return lambda regime: regime in self.allowed_regimes


@dataclass
class CurriculumState:
    """Current state of curriculum learning."""
    current_phase: CurriculumPhase = CurriculumPhase.PHASE_1
    current_step: int = 0
    phase_start_step: int = 0

    # Performance tracking per phase
    phase_rewards: List[float] = field(default_factory=list)
    phase_sharpe: float = 0.0
    phase_win_rate: float = 0.0

    # Phase transition history
    transitions: List[Tuple[int, CurriculumPhase, CurriculumPhase]] = field(default_factory=list)

    def record_transition(self, from_phase: CurriculumPhase, to_phase: CurriculumPhase) -> None:
        """Record a phase transition."""
        self.transitions.append((self.current_step, from_phase, to_phase))
        self.phase_start_step = self.current_step
        self.phase_rewards = []


class CurriculumScheduler:
    """
    Scheduler for curriculum learning phases.

    Manages transitions between training phases based on:
    1. Step count (primary)
    2. Performance metrics (optional early advancement)

    Usage:
        >>> scheduler = CurriculumScheduler(config)
        >>> for step in range(total_steps):
        ...     phase_config = scheduler.get_current_config()
        ...     if phase_config.is_regime_allowed(current_regime):
        ...         # Train on this sample
        ...         scheduler.step(reward)
    """

    def __init__(
        self,
        config: CurriculumConfig,
        callback_on_phase_change: Optional[Callable[[CurriculumPhase, CurriculumPhase], None]] = None,
    ):
        """
        Initialize curriculum scheduler.

        Args:
            config: Curriculum configuration
            callback_on_phase_change: Optional callback when phase changes
        """
        self._config = config
        self._enabled = config.enabled
        self._callback = callback_on_phase_change

        # Phase configurations
        self._phases = self._build_phases(config)

        # State
        self._state = CurriculumState()

    def _build_phases(self, config: CurriculumConfig) -> Dict[CurriculumPhase, PhaseConfig]:
        """Build phase configurations from config."""
        return {
            CurriculumPhase.PHASE_1: PhaseConfig(
                phase=CurriculumPhase.PHASE_1,
                name="Foundation",
                description="Learning in NORMAL regime with reduced costs",
                allowed_regimes=(MarketRegime.NORMAL, MarketRegime.LOW_VOL),
                cost_multiplier=config.phase_1_cost_mult,
                start_step=0,
                end_step=config.phase_1_steps,
            ),
            CurriculumPhase.PHASE_2: PhaseConfig(
                phase=CurriculumPhase.PHASE_2,
                name="Adaptation",
                description="Adding HIGH_VOL regime with moderate costs",
                allowed_regimes=(MarketRegime.NORMAL, MarketRegime.LOW_VOL, MarketRegime.HIGH_VOL),
                cost_multiplier=config.phase_2_cost_mult,
                start_step=config.phase_1_steps,
                end_step=config.phase_2_steps,
            ),
            CurriculumPhase.PHASE_3: PhaseConfig(
                phase=CurriculumPhase.PHASE_3,
                name="Full Training",
                description="All regimes including CRISIS with full costs",
                allowed_regimes=(MarketRegime.NORMAL, MarketRegime.LOW_VOL,
                                 MarketRegime.HIGH_VOL, MarketRegime.CRISIS),
                cost_multiplier=config.phase_3_cost_mult,
                start_step=config.phase_2_steps,
                end_step=config.phase_3_steps,
            ),
        }

    @property
    def current_phase(self) -> CurriculumPhase:
        """Get current curriculum phase."""
        return self._state.current_phase

    @property
    def current_config(self) -> PhaseConfig:
        """Get current phase configuration."""
        return self._phases[self._state.current_phase]

    @property
    def cost_multiplier(self) -> float:
        """Get current cost multiplier."""
        if not self._enabled:
            return 1.0
        return self.current_config.cost_multiplier

    @property
    def allowed_regimes(self) -> Tuple[MarketRegime, ...]:
        """Get currently allowed regimes."""
        if not self._enabled:
            return (MarketRegime.NORMAL, MarketRegime.LOW_VOL,
                    MarketRegime.HIGH_VOL, MarketRegime.CRISIS)
        return self.current_config.allowed_regimes

    def is_regime_allowed(self, regime: MarketRegime) -> bool:
        """Check if a regime is allowed in current phase."""
        if not self._enabled:
            return True
        return regime in self.allowed_regimes

    def step(self, reward: Optional[float] = None) -> bool:
        """
        Advance curriculum by one step.

        Args:
            reward: Optional reward from this step (for tracking)

        Returns:
            True if phase changed, False otherwise
        """
        self._state.current_step += 1

        if reward is not None:
            self._state.phase_rewards.append(reward)

        if not self._enabled:
            return False

        # Check for phase transition
        current = self._state.current_phase
        new_phase = self._determine_phase(self._state.current_step)

        if new_phase != current:
            self._transition_to_phase(new_phase)
            return True

        return False

    def _determine_phase(self, step: int) -> CurriculumPhase:
        """Determine which phase based on step count."""
        if step < self._config.phase_1_steps:
            return CurriculumPhase.PHASE_1
        elif step < self._config.phase_2_steps:
            return CurriculumPhase.PHASE_2
        else:
            return CurriculumPhase.PHASE_3

    def _transition_to_phase(self, new_phase: CurriculumPhase) -> None:
        """Handle phase transition."""
        old_phase = self._state.current_phase

        # Update state
        self._state.record_transition(old_phase, new_phase)
        self._state.current_phase = new_phase

        # Call callback if provided
        if self._callback:
            self._callback(old_phase, new_phase)

    def get_progress(self) -> Dict[str, Any]:
        """Get curriculum progress information."""
        current_config = self.current_config

        # Calculate phase progress
        phase_start = current_config.start_step
        phase_end = current_config.end_step
        phase_duration = phase_end - phase_start

        if phase_duration > 0:
            phase_progress = (self._state.current_step - phase_start) / phase_duration
            phase_progress = min(1.0, max(0.0, phase_progress))
        else:
            phase_progress = 1.0

        return {
            "current_phase": self._state.current_phase.value,
            "phase_name": current_config.name,
            "phase_description": current_config.description,
            "current_step": self._state.current_step,
            "phase_progress": phase_progress,
            "cost_multiplier": current_config.cost_multiplier,
            "allowed_regimes": [r.value for r in current_config.allowed_regimes],
            "transitions": len(self._state.transitions),
            "enabled": self._enabled,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics."""
        stats = self.get_progress()

        # Add phase-specific stats
        if self._state.phase_rewards:
            import numpy as np
            rewards = np.array(self._state.phase_rewards)
            stats.update({
                "phase_mean_reward": float(np.mean(rewards)),
                "phase_std_reward": float(np.std(rewards)),
                "phase_total_steps": len(rewards),
            })

        return stats

    def reset(self) -> None:
        """Reset curriculum state (for new training run)."""
        self._state = CurriculumState()

    def should_skip_sample(self, regime: MarketRegime) -> bool:
        """
        Check if a training sample should be skipped.

        In early phases, we skip samples from disallowed regimes.
        This creates a curated training set.

        Args:
            regime: Detected market regime

        Returns:
            True if sample should be skipped
        """
        if not self._enabled:
            return False
        return not self.is_regime_allowed(regime)

    def get_phase_for_step(self, step: int) -> CurriculumPhase:
        """Get which phase applies at a given step."""
        return self._determine_phase(step)


class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """
    Adaptive curriculum scheduler with performance-based transitions.

    Extends base scheduler with:
    - Early advancement if performance is good
    - Delayed advancement if performance is poor
    - Automatic difficulty adjustment
    """

    def __init__(
        self,
        config: CurriculumConfig,
        advance_sharpe_threshold: float = 0.5,
        advance_win_rate_threshold: float = 0.52,
        min_phase_steps: int = 10000,
        callback_on_phase_change: Optional[Callable[[CurriculumPhase, CurriculumPhase], None]] = None,
    ):
        """
        Initialize adaptive scheduler.

        Args:
            config: Curriculum configuration
            advance_sharpe_threshold: Sharpe threshold for early advancement
            advance_win_rate_threshold: Win rate threshold for early advancement
            min_phase_steps: Minimum steps before considering advancement
            callback_on_phase_change: Optional callback
        """
        super().__init__(config, callback_on_phase_change)

        self._advance_sharpe = advance_sharpe_threshold
        self._advance_win_rate = advance_win_rate_threshold
        self._min_phase_steps = min_phase_steps

        # Additional tracking
        self._phase_trades = 0
        self._phase_wins = 0
        self._trade_returns: List[float] = []

    def record_trade(self, pnl: float) -> None:
        """
        Record a completed trade for performance tracking.

        Args:
            pnl: Trade PnL
        """
        self._trade_returns.append(pnl)
        self._phase_trades += 1
        if pnl > 0:
            self._phase_wins += 1

        # Update phase metrics
        if self._phase_trades > 0:
            self._state.phase_win_rate = self._phase_wins / self._phase_trades

    def step(self, reward: Optional[float] = None) -> bool:
        """
        Advance with potential early advancement check.

        Args:
            reward: Optional reward

        Returns:
            True if phase changed
        """
        self._state.current_step += 1

        if reward is not None:
            self._state.phase_rewards.append(reward)

        if not self._enabled:
            return False

        # Check standard transition
        current = self._state.current_phase
        step_based_phase = self._determine_phase(self._state.current_step)

        # Check early advancement
        if self._can_advance_early(current):
            next_phase = self._get_next_phase(current)
            if next_phase and next_phase != current:
                self._transition_to_phase(next_phase)
                self._reset_phase_tracking()
                return True

        # Standard transition
        if step_based_phase != current:
            self._transition_to_phase(step_based_phase)
            self._reset_phase_tracking()
            return True

        return False

    def _can_advance_early(self, current_phase: CurriculumPhase) -> bool:
        """Check if agent can advance early based on performance."""
        # Can't advance from final phase
        if current_phase == CurriculumPhase.PHASE_3:
            return False

        # Need minimum steps in phase
        steps_in_phase = self._state.current_step - self._state.phase_start_step
        if steps_in_phase < self._min_phase_steps:
            return False

        # Need minimum trades
        if self._phase_trades < 20:
            return False

        # Check performance criteria
        sharpe = self._calculate_phase_sharpe()
        win_rate = self._state.phase_win_rate

        return sharpe >= self._advance_sharpe and win_rate >= self._advance_win_rate

    def _calculate_phase_sharpe(self) -> float:
        """Calculate Sharpe ratio for current phase."""
        if len(self._trade_returns) < 5:
            return 0.0

        import numpy as np
        returns = np.array(self._trade_returns)
        mean = np.mean(returns)
        std = np.std(returns)

        if std < 1e-8:
            return 0.0

        return mean / std

    def _get_next_phase(self, current: CurriculumPhase) -> Optional[CurriculumPhase]:
        """Get the next phase after current."""
        phase_order = [CurriculumPhase.PHASE_1, CurriculumPhase.PHASE_2, CurriculumPhase.PHASE_3]
        idx = phase_order.index(current)

        if idx < len(phase_order) - 1:
            return phase_order[idx + 1]
        return None

    def _reset_phase_tracking(self) -> None:
        """Reset phase-specific tracking."""
        self._phase_trades = 0
        self._phase_wins = 0
        self._trade_returns = []

    def get_stats(self) -> Dict[str, Any]:
        """Get extended statistics."""
        stats = super().get_stats()
        stats.update({
            "phase_trades": self._phase_trades,
            "phase_wins": self._phase_wins,
            "phase_win_rate": self._state.phase_win_rate,
            "phase_sharpe": self._calculate_phase_sharpe(),
            "can_advance_early": self._can_advance_early(self._state.current_phase),
        })
        return stats


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_curriculum_scheduler(
    reward_config: RewardConfig,
    adaptive: bool = False,
    callback: Optional[Callable[[CurriculumPhase, CurriculumPhase], None]] = None,
) -> CurriculumScheduler:
    """
    Create a curriculum scheduler from reward configuration.

    Args:
        reward_config: RewardConfig with curriculum settings
        adaptive: If True, use AdaptiveCurriculumScheduler
        callback: Optional phase change callback

    Returns:
        CurriculumScheduler or AdaptiveCurriculumScheduler
    """
    if adaptive:
        return AdaptiveCurriculumScheduler(
            config=reward_config.curriculum,
            callback_on_phase_change=callback,
        )
    else:
        return CurriculumScheduler(
            config=reward_config.curriculum,
            callback_on_phase_change=callback,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "CurriculumPhase",
    "PhaseConfig",
    "CurriculumState",
    "CurriculumScheduler",
    "AdaptiveCurriculumScheduler",
    "create_curriculum_scheduler",
]
