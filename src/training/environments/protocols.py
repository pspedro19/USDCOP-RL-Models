"""
Trading Environment Protocols - Interface Segregation.

This module defines Protocol interfaces for all trading environment components.
Following Interface Segregation Principle (ISP) from SOLID.

Design Patterns:
- Strategy Pattern: All components are interchangeable strategies
- Protocol Pattern: Python 3.8+ structural subtyping

SOLID Principles:
- Interface Segregation: Minimal, focused interfaces
- Dependency Inversion: High-level modules depend on abstractions

Usage:
    class MyRewardCalculator:
        def calculate(self, action, returns, position) -> float:
            return returns * (1 if position else 0.5)

    # Works because it implements RewardCalculator protocol
    env = TradingEnvironment(reward_calculator=MyRewardCalculator())
"""

from typing import Dict, List, Optional, Protocol, Tuple, Any, runtime_checkable
import numpy as np


# =============================================================================
# CORE PROTOCOLS
# =============================================================================

@runtime_checkable
class RewardCalculator(Protocol):
    """
    Protocol for reward calculation strategies.

    Implementations can use different reward shaping strategies:
    - Simple PnL-based rewards
    - Risk-adjusted rewards (Sharpe-like)
    - Penalized rewards (for excessive trading)
    """

    def calculate(
        self,
        action: int,
        returns: float,
        position: float,
        portfolio_value: float,
        drawdown: float,
        step: int,
        total_steps: int,
    ) -> float:
        """
        Calculate reward for the current step.

        Args:
            action: Discrete action taken (0=HOLD, 1=LONG, 2=SHORT)
            returns: Log return for this step
            position: Current position (-1, 0, 1)
            portfolio_value: Current portfolio value
            drawdown: Current drawdown percentage
            step: Current step number
            total_steps: Total steps in episode

        Returns:
            Reward value (typically in range [-1, 1])
        """
        ...


@runtime_checkable
class RiskManager(Protocol):
    """
    Protocol for risk management strategies.

    Implementations control position sizing, stop-losses,
    and circuit breakers.
    """

    def should_reduce_position(
        self,
        drawdown: float,
        volatility: float,
        consecutive_losses: int,
    ) -> bool:
        """Check if position should be reduced due to risk."""
        ...

    def get_position_scale(
        self,
        volatility: float,
        drawdown: float,
    ) -> float:
        """Get position size multiplier (0.0 to 1.0)."""
        ...

    def is_blocked(self) -> bool:
        """Check if trading is blocked (circuit breaker)."""
        ...

    def record_trade_result(self, pnl: float) -> None:
        """Record trade result for risk tracking."""
        ...


@runtime_checkable
class PositionTracker(Protocol):
    """
    Protocol for position tracking.

    Tracks current position state and provides state features
    for the observation space.
    """

    def update(self, action: int, price: float, timestamp: Any) -> None:
        """Update position based on action."""
        ...

    def get_position(self) -> float:
        """Get current position as float (-1, 0, 1)."""
        ...

    def get_state_features(self) -> Tuple[float, float]:
        """Get state features: (position, time_normalized)."""
        ...

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Get unrealized P&L at current price."""
        ...

    def reset(self) -> None:
        """Reset position state for new episode."""
        ...


@runtime_checkable
class StateObserver(Protocol):
    """
    Protocol for observation building.

    Constructs the observation vector from market features
    and position state.
    """

    def get_observation(
        self,
        market_features: np.ndarray,
        position: float,
        time_normalized: float,
    ) -> np.ndarray:
        """
        Build complete observation vector.

        Args:
            market_features: Array of market features (13 dims)
            position: Current position (-1, 0, 1)
            time_normalized: Normalized time in session (0-1)

        Returns:
            Complete observation array (15 dims)
        """
        ...

    def get_feature_names(self) -> List[str]:
        """Get names of all features in order."""
        ...

    @property
    def observation_dim(self) -> int:
        """Get total observation dimension."""
        ...


@runtime_checkable
class TransactionCostModel(Protocol):
    """
    Protocol for transaction cost modeling.

    Models trading costs including spread, slippage, and fees.
    """

    def calculate_cost(
        self,
        size: float,
        price: float,
        volatility: float,
    ) -> float:
        """
        Calculate total transaction cost.

        Args:
            size: Position size
            price: Current price
            volatility: Current volatility (e.g., ATR%)

        Returns:
            Total cost in currency units
        """
        ...


@runtime_checkable
class ActionDiscretizer(Protocol):
    """
    Protocol for action discretization.

    Converts continuous model output to discrete trading actions.
    """

    def discretize(self, continuous_action: float) -> int:
        """
        Convert continuous action to discrete action.

        Args:
            continuous_action: Model output (typically -1 to 1)

        Returns:
            Discrete action (0=HOLD, 1=LONG, 2=SHORT)
        """
        ...

    def get_thresholds(self) -> Tuple[float, float]:
        """Get (short_threshold, long_threshold)."""
        ...


@runtime_checkable
class EpisodeTerminator(Protocol):
    """
    Protocol for episode termination logic.

    Determines when an episode should end.
    """

    def should_terminate(
        self,
        step: int,
        max_steps: int,
        drawdown: float,
        max_drawdown: float,
    ) -> Tuple[bool, str]:
        """
        Check if episode should terminate.

        Returns:
            Tuple of (should_terminate, reason)
        """
        ...


# =============================================================================
# DATA PROTOCOLS
# =============================================================================

@runtime_checkable
class MarketDataProvider(Protocol):
    """
    Protocol for market data access.

    Provides OHLCV and derived features.
    """

    def get_bar(self, index: int) -> Dict[str, float]:
        """Get OHLCV data for a specific bar."""
        ...

    def get_features(self, index: int) -> np.ndarray:
        """Get feature vector for a specific bar."""
        ...

    def get_price(self, index: int) -> float:
        """Get close price for a specific bar."""
        ...

    def __len__(self) -> int:
        """Get total number of bars."""
        ...


@runtime_checkable
class NormalizationProvider(Protocol):
    """
    Protocol for feature normalization.

    Provides normalization parameters and methods.
    """

    def normalize(self, feature_name: str, value: float) -> float:
        """Normalize a single feature value."""
        ...

    def normalize_batch(self, features: Dict[str, float]) -> np.ndarray:
        """Normalize all features and return array in order."""
        ...

    def get_stats(self, feature_name: str) -> Dict[str, float]:
        """Get normalization stats (mean, std) for a feature."""
        ...


# =============================================================================
# COMPOSITE PROTOCOLS
# =============================================================================

@runtime_checkable
class TradingEnvironmentComponents(Protocol):
    """
    Composite protocol for all environment components.

    Used for dependency injection in TradingEnvironment.
    """

    @property
    def reward_calculator(self) -> RewardCalculator:
        ...

    @property
    def risk_manager(self) -> RiskManager:
        ...

    @property
    def position_tracker(self) -> PositionTracker:
        ...

    @property
    def state_observer(self) -> StateObserver:
        ...

    @property
    def transaction_cost_model(self) -> TransactionCostModel:
        ...


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Observation type: 1D numpy array of floats
Observation = np.ndarray

# Action type: continuous float or discrete int
ContinuousAction = float
DiscreteAction = int

# Step result type
StepResult = Tuple[Observation, float, bool, bool, Dict[str, Any]]

# Reset result type
ResetResult = Tuple[Observation, Dict[str, Any]]
