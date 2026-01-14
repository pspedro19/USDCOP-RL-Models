"""
USD/COP RL Trading System - Core Interfaces (Protocols)
========================================================

This module defines Protocol interfaces for the main trading system components.
Using Protocol instead of ABC allows structural subtyping (duck typing with
type checking), meaning existing implementations are automatically compatible
without explicit inheritance.

DESIGN PRINCIPLES:
1. Minimal interface: Only essential methods required by consumers
2. Backward compatible: Existing classes work without modification
3. Optional methods: Extended functionality through separate protocols
4. Type safety: Full typing support for mypy/pyright

EXISTING IMPLEMENTATIONS:
- SortinoRewardFunction implements IRewardFunction
- RegimeDetector implements IRegimeDetector
- RiskManager implements IRiskManager
- SETFXCostModel implements ICostModel

Author: Claude Code
Version: 1.0.0
Date: 2025-12-26
"""

from typing import (
    Protocol,
    Dict,
    Tuple,
    List,
    Any,
    Optional,
    Union,
    TypeVar,
    runtime_checkable,
)
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# SUPPORTING TYPES
# =============================================================================

@dataclass
class RewardResult:
    """
    Standard result from reward calculation.

    Attributes:
        reward: The scalar reward value
        components: Breakdown of reward components for debugging/logging
    """
    reward: float
    components: Dict[str, float]


@dataclass
class RiskUpdateResult:
    """
    Standard result from risk manager update.

    Attributes:
        status: Current risk status (ACTIVE, REDUCED, PAUSED, etc.)
        position_multiplier: Factor to apply to positions [0.0, 1.0]
        alerts: List of new alerts generated this update
    """
    status: str
    position_multiplier: float
    alerts: List[Any]


# =============================================================================
# IREWARDFUNCTION PROTOCOL
# =============================================================================

@runtime_checkable
class IRewardFunction(Protocol):
    """
    Protocol for reward functions in the RL trading system.

    A reward function calculates the reward signal given trading state.
    It should track historical returns for ratio-based metrics (Sharpe, Sortino).

    COMPATIBLE IMPLEMENTATIONS:
    - SortinoRewardFunction
    - HybridSharpesortinoReward
    - Any custom reward with reset() and calculate() methods

    Example:
        ```python
        def train_with_reward(reward_fn: IRewardFunction):
            reward_fn.reset(initial_balance=10000)

            for step in episode:
                reward, components = reward_fn.calculate(
                    portfolio_return=step_return,
                    market_return=market_return,
                    portfolio_value=portfolio_value,
                    position=current_position,
                    prev_position=previous_position,
                )
                # Use reward for training...
        ```

    DESIGN NOTE:
    - reset() and calculate() are REQUIRED
    - get_stats() is optional but recommended
    - volatility_percentile and transaction_cost in calculate() have defaults
      to maintain backward compatibility with simpler reward functions
    """

    def reset(self, initial_balance: float = 10000) -> None:
        """
        Reset reward function state for a new episode.

        Should clear all historical data and reinitialize any internal buffers.

        Args:
            initial_balance: Starting portfolio value for the episode.
                           Used for drawdown and return calculations.
        """
        ...

    def calculate(
        self,
        portfolio_return: float,
        market_return: float,
        portfolio_value: float,
        position: float,
        prev_position: float,
        volatility_percentile: float = 0.5,
        transaction_cost: float = 0.0,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for current trading step.

        Args:
            portfolio_return: Portfolio return this step (decimal, e.g., 0.001 = 0.1%)
            market_return: Raw market return this step (for reference/alpha calc)
            portfolio_value: Current portfolio value in currency units
            position: Current position after action [-1.0, 1.0]
            prev_position: Position before this step's action
            volatility_percentile: Current volatility percentile [0.0, 1.0]
                                  (optional, defaults to 0.5 for neutral)
            transaction_cost: Cost incurred this step (decimal)
                            (optional, defaults to 0.0)

        Returns:
            Tuple of:
            - reward (float): Scalar reward value (typically scaled/clipped)
            - components (Dict[str, float]): Breakdown of reward components
              e.g., {'pnl': 0.5, 'sortino': 0.3, 'cost': -0.1, 'total': 0.7}

        Note:
            The components dict should always include 'total' key.
            Additional keys depend on implementation (sortino, sharpe, drawdown, etc.)
        """
        ...


class IRewardFunctionWithStats(IRewardFunction, Protocol):
    """
    Extended protocol for reward functions with statistics tracking.

    This is an optional extension - implementations can provide
    get_stats() for detailed monitoring during training.
    """

    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics about reward function state.

        Returns:
            Dictionary with aggregated statistics, typically including:
            - total_mean, total_std: Reward distribution stats
            - sortino_current: Current Sortino ratio
            - cumulative_cost: Total transaction costs
            - Component-specific means/stds
        """
        ...


# =============================================================================
# IREGIMEDETECTOR PROTOCOL
# =============================================================================

@runtime_checkable
class IRegimeDetector(Protocol):
    """
    Protocol for market regime detection.

    A regime detector classifies market conditions (NORMAL, VOLATILE, CRISIS)
    based on macroeconomic indicators and volatility measures.

    COMPATIBLE IMPLEMENTATIONS:
    - RegimeDetector

    Example:
        ```python
        def adjust_for_regime(detector: IRegimeDetector, obs: np.ndarray):
            vix_z = obs[VIX_IDX]
            embi_z = obs[EMBI_IDX]
            vol_pct = obs[VOL_IDX] * 100  # Scale to 0-100

            regime = detector.detect_regime(vix_z, embi_z, vol_pct)
            multiplier = detector.get_position_multiplier(regime)

            return regime, multiplier
        ```

    REGIME DEFINITIONS:
    - "NORMAL": Standard market conditions, full position sizing
    - "VOLATILE": Elevated volatility, reduced position sizing (typically 50%)
    - "CRISIS": Extreme conditions, minimal or no trading (typically 0%)
    """

    def detect_regime(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float,
        return_probabilities: bool = False
    ) -> Union[str, Tuple[str, Dict[str, float]]]:
        """
        Detect current market regime based on indicators.

        Args:
            vix_z: VIX z-score (standardized, mean=0, std=1)
            embi_z: EMBI+ z-score (standardized)
            vol_pct: Realized volatility percentile (0-100 scale)
            return_probabilities: If True, also return regime probabilities

        Returns:
            If return_probabilities=False:
                regime (str): One of "NORMAL", "VOLATILE", "CRISIS"

            If return_probabilities=True:
                Tuple of:
                - regime (str): Detected regime
                - probabilities (Dict[str, float]): Probability for each regime
                  e.g., {"NORMAL": 0.2, "VOLATILE": 0.7, "CRISIS": 0.1}

        Note:
            Probabilities should sum to 1.0 (or close to it).
        """
        ...

    def get_position_multiplier(self, regime: str) -> float:
        """
        Get position sizing multiplier for given regime.

        Args:
            regime: Market regime string ("NORMAL", "VOLATILE", or "CRISIS")

        Returns:
            Multiplier in range [0.0, 1.0]:
            - NORMAL: typically 1.0 (100% position)
            - VOLATILE: typically 0.5 (50% position)
            - CRISIS: typically 0.0 (no trading)

        Note:
            Unknown regimes should return the most conservative multiplier (0.0).
        """
        ...


class IRegimeDetectorWithHistory(IRegimeDetector, Protocol):
    """
    Extended protocol for regime detectors with history tracking.

    Optional extension for detectors that maintain regime history
    for analysis and statistics.
    """

    def reset_history(self) -> None:
        """Clear regime detection history."""
        ...

    def get_regime_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about detected regimes.

        Returns:
            Dictionary containing:
            - total_observations: Number of regime detections
            - distribution: Regime counts/percentages
            - avg_indicators: Average indicators per regime
            - transitions: Regime transition counts
        """
        ...

    def get_regime_probs(
        self,
        vix_z: float,
        embi_z: float,
        vol_pct: float
    ) -> Dict[str, float]:
        """
        Calculate probabilistic regime classification.

        Args:
            vix_z: VIX z-score
            embi_z: EMBI+ z-score
            vol_pct: Volatility percentile (0-100)

        Returns:
            Dictionary with probabilities for each regime.
            Keys: "CRISIS", "VOLATILE", "NORMAL"
            Values sum to 1.0
        """
        ...


# =============================================================================
# IRISKMANAGER PROTOCOL
# =============================================================================

@runtime_checkable
class IRiskManager(Protocol):
    """
    Protocol for risk management with kill switches.

    A risk manager monitors trading metrics in real-time and can
    pause/reduce trading when limits are breached.

    COMPATIBLE IMPLEMENTATIONS:
    - RiskManager

    Example:
        ```python
        def trading_loop(risk_mgr: IRiskManager, model, env):
            risk_mgr.reset(initial_value=10000)

            for step in episode:
                if not risk_mgr.should_trade():
                    action = [0.0]  # HOLD
                else:
                    action = model.predict(obs)
                    action *= risk_mgr.get_position_multiplier()

                obs, reward, done, info = env.step(action)

                status, mult, alerts = risk_mgr.update(
                    portfolio_value=info['portfolio'],
                    step_return=info['step_return'],
                    action=float(action[0]),
                )

                for alert in alerts:
                    log_alert(alert)
        ```

    STATUS VALUES:
    - ACTIVE: Normal trading
    - REDUCED: Position sizing reduced (typically 50%)
    - PAUSED: Trading halted
    - MANUAL_REVIEW: Requires human intervention
    """

    def reset(self, initial_value: float = 10000.0) -> None:
        """
        Reset risk manager for new episode/trading session.

        Args:
            initial_value: Starting portfolio value
        """
        ...

    def update(
        self,
        portfolio_value: float,
        step_return: float,
        action: float,
    ) -> Tuple[Any, float, List[Any]]:
        """
        Update risk manager with new trading data.

        This is the main method called each step to track metrics
        and check risk limits.

        Args:
            portfolio_value: Current portfolio value
            step_return: Return this step (decimal)
            action: Action taken [-1.0, 1.0]

        Returns:
            Tuple of:
            - status: Current risk status (RiskStatus enum or string)
            - position_multiplier: Factor to apply to positions [0.0, 1.0]
            - alerts: List of new Alert objects generated this update
        """
        ...

    def should_trade(self) -> bool:
        """
        Check if trading is currently allowed.

        Returns:
            True if trading is allowed (status != PAUSED)
            False if trading should be halted
        """
        ...

    def get_position_multiplier(self) -> float:
        """
        Get current position sizing multiplier based on risk status.

        Returns:
            Multiplier in range [0.0, 1.0]:
            - ACTIVE: 1.0
            - REDUCED: 0.5
            - PAUSED: 0.0
        """
        ...


class IRiskManagerWithReporting(IRiskManager, Protocol):
    """
    Extended protocol for risk managers with detailed reporting.

    Optional extension for risk managers that provide detailed
    state reporting and metrics.
    """

    def get_report(self) -> Dict[str, Any]:
        """
        Get comprehensive risk management report.

        Returns:
            Dictionary containing:
            - status: Current risk status
            - current_value: Current portfolio value
            - peak_value: Peak portfolio value
            - drawdown: Current drawdown
            - rolling_sharpe: Rolling Sharpe ratio
            - hold_pct: Percentage of HOLD actions
            - consecutive_losses: Current losing streak
            - total_alerts: Total alert count
            - recent_alerts: List of recent alerts
        """
        ...

    def get_current_drawdown(self) -> float:
        """
        Get current drawdown from peak.

        Returns:
            Drawdown as decimal (e.g., 0.05 = 5% drawdown)
        """
        ...

    def force_resume(self) -> None:
        """
        Force resume trading from PAUSED status.

        This should be used sparingly and only with human confirmation.
        Logs an alert indicating manual override.
        """
        ...


# =============================================================================
# ICOSTMODEL PROTOCOL
# =============================================================================

@runtime_checkable
class ICostModel(Protocol):
    """
    Protocol for transaction cost modeling.

    A cost model calculates trading costs based on market conditions
    and position changes.

    COMPATIBLE IMPLEMENTATIONS:
    - SETFXCostModel

    Example:
        ```python
        def calculate_net_return(
            cost_model: ICostModel,
            gross_return: float,
            position_change: float,
            volatility_percentile: float
        ) -> float:
            cost = cost_model.get_cost(volatility_percentile, position_change)
            return gross_return - cost
        ```

    COST COMPONENTS:
    - Spread: Bid-ask spread (varies by volatility regime)
    - Slippage: Market impact from order execution
    - Total cost = (spread + slippage) * |position_change|
    """

    def get_cost(
        self,
        volatility_percentile: float,
        position_change: float
    ) -> float:
        """
        Calculate total transaction cost.

        Args:
            volatility_percentile: Current volatility percentile [0.0, 1.0]
                                  Used to determine spread regime
            position_change: Absolute change in position [0.0, 2.0]
                           (e.g., going from -1 to +1 = 2.0 change)

        Returns:
            Cost as decimal (e.g., 0.0020 = 20 basis points)

        Note:
            Cost should be proportional to position_change.
            Cost should increase with volatility.
        """
        ...


class ICostModelWithDetails(ICostModel, Protocol):
    """
    Extended protocol for cost models with detailed cost breakdown.

    Optional extension for cost models that can provide
    detailed cost information (spread, slippage breakdown).
    """

    def get_cost_bps(self, volatility_percentile: float) -> float:
        """
        Get current spread in basis points.

        Args:
            volatility_percentile: Current volatility percentile [0.0, 1.0]

        Returns:
            Spread in basis points (e.g., 14.0 = 14 bps)

        Note:
            This returns only the spread component, not total cost.
            Multiply by 1e-4 to convert to decimal for calculations.
        """
        ...


# =============================================================================
# TYPE ALIASES FOR CONVENIENCE
# =============================================================================

# For type hints in function signatures
RewardFunction = IRewardFunction
RegimeDetector = IRegimeDetector
RiskManager = IRiskManager
CostModel = ICostModel


# =============================================================================
# VERIFICATION FUNCTIONS
# =============================================================================

def verify_reward_function(obj: Any) -> bool:
    """
    Verify an object implements IRewardFunction protocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements required methods

    Example:
        ```python
        reward_fn = SortinoRewardFunction()
        assert verify_reward_function(reward_fn)
        ```
    """
    return isinstance(obj, IRewardFunction)


def verify_regime_detector(obj: Any) -> bool:
    """
    Verify an object implements IRegimeDetector protocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements required methods
    """
    return isinstance(obj, IRegimeDetector)


def verify_risk_manager(obj: Any) -> bool:
    """
    Verify an object implements IRiskManager protocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements required methods
    """
    return isinstance(obj, IRiskManager)


def verify_cost_model(obj: Any) -> bool:
    """
    Verify an object implements ICostModel protocol.

    Args:
        obj: Object to check

    Returns:
        True if object implements required methods
    """
    return isinstance(obj, ICostModel)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == '__main__':
    """
    Test protocol compatibility with existing implementations.
    """
    print("=" * 70)
    print("CORE INTERFACES - Protocol Verification")
    print("=" * 70)

    # Test imports
    print("\n1. Testing imports from existing modules...")

    try:
        from ..sortino_reward import SortinoRewardFunction
        print("  [OK] SortinoRewardFunction imported")

        reward_fn = SortinoRewardFunction()
        is_compatible = verify_reward_function(reward_fn)
        print(f"  [{'OK' if is_compatible else 'FAIL'}] SortinoRewardFunction implements IRewardFunction: {is_compatible}")
    except ImportError as e:
        print(f"  [SKIP] Could not import SortinoRewardFunction: {e}")

    try:
        from ..regime_detector import RegimeDetector as RD
        print("  [OK] RegimeDetector imported")

        detector = RD()
        is_compatible = verify_regime_detector(detector)
        print(f"  [{'OK' if is_compatible else 'FAIL'}] RegimeDetector implements IRegimeDetector: {is_compatible}")
    except ImportError as e:
        print(f"  [SKIP] Could not import RegimeDetector: {e}")

    try:
        from ..risk_manager import RiskManager as RM
        print("  [OK] RiskManager imported")

        risk_mgr = RM()
        is_compatible = verify_risk_manager(risk_mgr)
        print(f"  [{'OK' if is_compatible else 'FAIL'}] RiskManager implements IRiskManager: {is_compatible}")
    except ImportError as e:
        print(f"  [SKIP] Could not import RiskManager: {e}")

    try:
        from ..environment_v19 import SETFXCostModel
        print("  [OK] SETFXCostModel imported")

        cost_model = SETFXCostModel()
        is_compatible = verify_cost_model(cost_model)
        print(f"  [{'OK' if is_compatible else 'FAIL'}] SETFXCostModel implements ICostModel: {is_compatible}")
    except ImportError as e:
        print(f"  [SKIP] Could not import SETFXCostModel: {e}")

    print("\n" + "=" * 70)
    print("Protocol verification complete!")
    print("=" * 70)
