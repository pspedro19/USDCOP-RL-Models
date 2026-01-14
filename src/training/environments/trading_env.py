"""
Trading Environment - Professional Implementation
=================================================
Gymnasium-based trading environment for PPO training.

SOLID Principles:
- Single Responsibility: Only simulates trading
- Open/Closed: Extensible via reward strategies
- Liskov Substitution: Follows gym.Env interface
- Interface Segregation: Minimal required interface
- Dependency Inversion: Depends on abstractions (strategies)

Design Patterns:
- Strategy Pattern: Pluggable reward calculation
- Template Method: Standard step/reset flow
- Factory Method: Created via EnvironmentFactory

Clean Code:
- Descriptive naming
- Small focused methods
- No magic numbers (all configurable)
- Comprehensive docstrings
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Protocol
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Type Definitions
# =============================================================================

class TradingAction(Enum):
    """Trading action types"""
    SHORT = -1
    HOLD = 0
    LONG = 1


@dataclass
class Position:
    """Current position state"""
    side: TradingAction = TradingAction.HOLD
    size: float = 0.0
    entry_price: float = 0.0
    entry_bar: int = 0
    unrealized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return self.side == TradingAction.HOLD

    @property
    def is_long(self) -> bool:
        return self.side == TradingAction.LONG

    @property
    def is_short(self) -> bool:
        return self.side == TradingAction.SHORT


@dataclass
class PortfolioState:
    """Portfolio state tracking"""
    balance: float
    equity: float
    peak_equity: float
    position: Position
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.equity) / self.peak_equity

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0.0
        return self.winning_trades / self.trades_count


@dataclass
class StepResult:
    """Result of a single step"""
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


# =============================================================================
# Reward Strategy Protocol (Strategy Pattern)
# =============================================================================

class RewardStrategy(Protocol):
    """Protocol for reward calculation strategies"""

    def calculate(
        self,
        pnl: float,
        portfolio: PortfolioState,
        position_changed: bool,
        market_return: float,
        step_count: int,
        episode_length: int,
    ) -> float:
        """Calculate reward for current step"""
        ...


# =============================================================================
# Environment Configuration
# =============================================================================

@dataclass
class TradingEnvConfig:
    """
    Trading environment configuration.

    All parameters are explicit - no magic numbers.
    """
    # Episode settings
    # Training uses longer episodes (1200 bars = ~20 days) for diverse experience
    # Production inference uses shorter episodes (400 bars) per feature_config.json
    episode_length: int = 1200  # ~20 days @ 60 bars/day (training default)
    warmup_bars: int = 14  # Bars needed for technical indicators

    # Portfolio settings - values from config/trading_config.yaml SSOT
    initial_balance: float = 10_000.0
    transaction_cost_bps: float = 75.0  # From SSOT: costs.transaction_cost_bps
    slippage_bps: float = 15.0  # From SSOT: costs.slippage_bps

    # Risk management
    max_drawdown_pct: float = 15.0  # Stop episode at 15% drawdown
    max_position_duration: int = 288  # Max bars in position (~1 day)

    # Action thresholds - from config/trading_config.yaml SSOT
    threshold_long: float = 0.33  # From SSOT: thresholds.long
    threshold_short: float = -0.33  # From SSOT: thresholds.short

    # Circuit breaker for consecutive losses
    max_consecutive_losses: int = 5  # Stop trading after 5 consecutive losses
    cooldown_bars_after_losses: int = 12  # 1 hour cooldown (12 x 5min)

    # Volatility filter
    enable_volatility_filter: bool = True
    max_atr_multiplier: float = 2.0  # Force HOLD if ATR > 2x historical mean

    # Observation settings
    observation_dim: int = 15
    clip_range: Tuple[float, float] = (-5.0, 5.0)

    # Feature configuration
    core_features: List[str] = field(default_factory=lambda: [
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    ])
    state_features: List[str] = field(default_factory=lambda: [
        "position", "time_normalized"
    ])

    def __post_init__(self):
        """Validate configuration"""
        total_features = len(self.core_features) + len(self.state_features)
        if total_features != self.observation_dim:
            raise ValueError(
                f"Feature count mismatch: {len(self.core_features)} core + "
                f"{len(self.state_features)} state = {total_features}, "
                f"but observation_dim = {self.observation_dim}"
            )

    @property
    def transaction_cost(self) -> float:
        return self.transaction_cost_bps / 10_000

    @property
    def slippage(self) -> float:
        return self.slippage_bps / 10_000

    @property
    def max_drawdown(self) -> float:
        return self.max_drawdown_pct / 100


# =============================================================================
# Reward Strategy Adapter (Integrates RewardCalculator)
# =============================================================================

class RewardStrategyAdapter:
    """
    Adapter that wraps RewardCalculator to work with TradingEnvironment.

    This adapter ensures the full reward function is used,
    including intratrade DD penalty and time decay.
    """

    def __init__(self):
        from ..reward_calculator import RewardCalculator, RewardConfig
        self._calculator = RewardCalculator(RewardConfig())
        self._bars_held = 0
        self._consecutive_wins = 0
        self._max_intratrade_dd = 0.0
        self._last_position_is_flat = True

    def calculate(
        self,
        pnl: float,
        portfolio: "PortfolioState",
        position_changed: bool,
        market_return: float,
        step_count: int,
        episode_length: int,
    ) -> float:
        """Calculate reward using RewardCalculator."""
        initial_balance = portfolio.balance - portfolio.total_pnl

        # Calculate PnL percentage
        pnl_pct = pnl / initial_balance if initial_balance > 0 else 0.0

        # Track position state
        is_flat = portfolio.position.is_flat

        # Determine position change for reward calculator
        if position_changed:
            if is_flat:
                position_change = -1  # Closed position
            else:
                position_change = 1  # Opened position
        else:
            position_change = 0

        # Update bars held
        if is_flat:
            self._bars_held = 0
            self._max_intratrade_dd = 0.0
        else:
            self._bars_held += 1
            # Track intratrade drawdown
            if pnl_pct < 0:
                self._max_intratrade_dd = max(self._max_intratrade_dd, abs(pnl_pct))

        # Update consecutive wins
        if position_change == -1:  # Position just closed
            if pnl > 0:
                self._consecutive_wins += 1
            else:
                self._consecutive_wins = 0

        # Call reward calculator
        reward, _ = self._calculator.calculate(
            pnl_pct=pnl_pct,
            position_change=position_change,
            bars_held=self._bars_held,
            consecutive_wins=self._consecutive_wins,
            current_drawdown=portfolio.drawdown,
            intratrade_drawdown=self._max_intratrade_dd,
        )

        self._last_position_is_flat = is_flat

        return reward

    def reset(self):
        """Reset internal state for new episode."""
        self._bars_held = 0
        self._consecutive_wins = 0
        self._max_intratrade_dd = 0.0
        self._last_position_is_flat = True
        self._calculator.reset()


# =============================================================================
# Default Reward Strategy (Legacy - basic implementation)
# =============================================================================

class DefaultRewardStrategy:
    """
    Default reward calculation strategy.

    Reward components:
    1. Scaled PnL (primary signal)
    2. Asymmetric loss penalty (risk aversion)
    3. Trade frequency penalty (avoid overtrading)
    4. Profit bonus (encourage good trades)
    """

    def __init__(
        self,
        pnl_scale: float = 100.0,
        loss_multiplier: float = 2.0,  # Increased from 1.5 for more asymmetry
        trade_penalty: float = 0.01,
        profit_bonus: float = 0.0,  # Disabled (was counterproductive)
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.pnl_scale = pnl_scale
        self.loss_multiplier = loss_multiplier
        self.trade_penalty = trade_penalty
        self.profit_bonus = profit_bonus
        self.clip_range = clip_range

    def calculate(
        self,
        pnl: float,
        portfolio: PortfolioState,
        position_changed: bool,
        market_return: float,
        step_count: int,
        episode_length: int,
    ) -> float:
        """Calculate reward using default strategy"""
        initial_balance = portfolio.balance - portfolio.total_pnl

        # 1. Base reward: scaled PnL
        if initial_balance > 0:
            reward = (pnl / initial_balance) * self.pnl_scale
        else:
            reward = 0.0

        # 2. Asymmetric loss penalty
        if reward < 0:
            reward *= self.loss_multiplier

        # 3. Trade frequency penalty
        if position_changed:
            reward -= self.trade_penalty

        # 4. Profit bonus for good trades
        if not portfolio.position.is_flat and pnl > 0:
            reward += self.profit_bonus

        # Clip reward
        return float(np.clip(reward, self.clip_range[0], self.clip_range[1]))


# =============================================================================
# Observation Builder (for environment)
# =============================================================================

class EnvObservationBuilder:
    """
    Builds observations for the trading environment.

    Separate from inference ObservationBuilder for:
    - Training-specific optimizations (vectorized)
    - Direct numpy access (no pandas overhead)
    """

    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, float]],
        core_features: List[str],
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.norm_stats = norm_stats
        self.core_features = core_features
        self.clip_range = clip_range
        self.observation_dim = len(core_features) + 2  # +2 for state features

    def build(
        self,
        feature_values: np.ndarray,
        position: float,
        time_normalized: float,
    ) -> np.ndarray:
        """
        Build observation vector.

        Args:
            feature_values: Raw feature values (len = core_features)
            position: Current position (-1 to 1)
            time_normalized: Session progress (0 to 1)

        Returns:
            Normalized observation array
        """
        obs = np.zeros(self.observation_dim, dtype=np.float32)

        # Normalize core features
        for i, (fname, value) in enumerate(zip(self.core_features, feature_values)):
            obs[i] = self._normalize(fname, value)

        # State features (not normalized)
        obs[-2] = np.clip(position, -1.0, 1.0)
        obs[-1] = np.clip(time_normalized, 0.0, 1.0)

        # Final clip and NaN handling
        obs = np.clip(obs, self.clip_range[0], self.clip_range[1])
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.clip_range[1], neginf=self.clip_range[0])

        return obs

    def _normalize(self, feature: str, value: float) -> float:
        """Z-score normalize a feature"""
        if feature not in self.norm_stats:
            return float(np.clip(value, self.clip_range[0], self.clip_range[1]))

        stats = self.norm_stats[feature]
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        if std < 1e-8:
            std = 1.0

        z = (value - mean) / std
        return float(np.clip(z, self.clip_range[0], self.clip_range[1]))


# =============================================================================
# Trading Environment
# =============================================================================

class TradingEnvironment(gym.Env):
    """
    Professional trading environment for RL training.

    Features:
    - Configurable observation space (default 15-dim)
    - Pluggable reward strategies
    - Risk management (max drawdown, position duration)
    - Transaction costs and slippage
    - Episode management with random starts

    Usage:
        config = TradingEnvConfig(episode_length=1200)
        env = TradingEnvironment(
            df=training_data,
            norm_stats=norm_stats,
            config=config,
        )

        obs, info = env.reset()
        for _ in range(1000):
            action = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        df: pd.DataFrame,
        norm_stats: Dict[str, Dict[str, float]],
        config: TradingEnvConfig,
        reward_strategy: Optional[RewardStrategy] = None,
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV and feature data
            norm_stats: Normalization statistics {feature: {mean, std}}
            config: Environment configuration
            reward_strategy: Optional custom reward strategy
        """
        super().__init__()

        self.config = config
        # Use RewardStrategyAdapter by default for full reward function
        self.reward_strategy = reward_strategy or RewardStrategyAdapter()

        # Validate and prepare data
        self._validate_dataframe(df)
        self.df = df.reset_index(drop=True)
        self.n_bars = len(df)

        # Extract feature matrix for fast access
        self.feature_matrix = df[config.core_features].values.astype(np.float32)
        self.returns = df["log_ret_5m"].values.astype(np.float32)

        # Observation builder
        self.obs_builder = EnvObservationBuilder(
            norm_stats=norm_stats,
            core_features=config.core_features,
            clip_range=config.clip_range,
        )

        # Action space: continuous [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=config.clip_range[0],
            high=config.clip_range[1],
            shape=(config.observation_dim,),
            dtype=np.float32,
        )

        # State (initialized in reset)
        self._portfolio: Optional[PortfolioState] = None
        self._current_idx: int = 0
        self._start_idx: int = 0
        self._step_count: int = 0

        # Circuit breaker state
        self._consecutive_losses: int = 0
        self._cooldown_until_step: int = 0

        # Historical ATR for volatility filter
        if config.enable_volatility_filter and "atr_pct" in df.columns:
            self._historical_atr_mean = df["atr_pct"].mean()
            self._historical_atr_std = df["atr_pct"].std()
        else:
            self._historical_atr_mean = 0.01  # Default fallback
            self._historical_atr_std = 0.005

        logger.info(
            f"TradingEnvironment initialized: "
            f"{self.n_bars} bars, {config.observation_dim} dims, "
            f"episode_length={config.episode_length}, "
            f"thresholds={config.threshold_long}/{config.threshold_short}, "
            f"costs={config.transaction_cost_bps}bps"
        )

    def _validate_dataframe(self, df: pd.DataFrame) -> None:
        """Validate DataFrame has required columns"""
        missing = [f for f in self.config.core_features if f not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing required features: {missing}")

        if "log_ret_5m" not in df.columns:
            raise ValueError("DataFrame must have 'log_ret_5m' column for returns")

        if len(df) < self.config.episode_length + self.config.warmup_bars:
            raise ValueError(
                f"DataFrame too short: {len(df)} rows, need at least "
                f"{self.config.episode_length + self.config.warmup_bars}"
            )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Optional reset options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Random episode start (with margin for episode length)
        max_start = self.n_bars - self.config.episode_length - self.config.warmup_bars
        self._start_idx = self.np_random.integers(self.config.warmup_bars, max(self.config.warmup_bars + 1, max_start))
        self._current_idx = self._start_idx
        self._step_count = 0

        # Initialize portfolio
        self._portfolio = PortfolioState(
            balance=self.config.initial_balance,
            equity=self.config.initial_balance,
            peak_equity=self.config.initial_balance,
            position=Position(),
        )

        # Reset circuit breaker state
        self._consecutive_losses = 0
        self._cooldown_until_step = 0

        # Reset reward strategy if it has reset method
        if hasattr(self.reward_strategy, 'reset'):
            self.reward_strategy.reset()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.

        Args:
            action: Action array of shape (1,) with value in [-1, 1]

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action_value = float(action[0])

        # Map continuous action to discrete signal
        target_action = self._map_action(action_value)

        # Execute trade if position changes
        position_changed, trade_cost = self._execute_action(target_action)

        # Move to next bar
        self._current_idx += 1
        self._step_count += 1

        # Calculate PnL from market return
        market_return = self.returns[self._current_idx]
        position_value = self._portfolio.position.side.value  # -1, 0, or 1
        gross_pnl = position_value * market_return * self._portfolio.balance
        net_pnl = gross_pnl - trade_cost

        # Update portfolio
        self._update_portfolio(net_pnl)

        # Calculate reward
        reward = self.reward_strategy.calculate(
            pnl=net_pnl,
            portfolio=self._portfolio,
            position_changed=position_changed,
            market_return=market_return,
            step_count=self._step_count,
            episode_length=self.config.episode_length,
        )

        # Check termination
        terminated = self._check_termination()
        truncated = False

        observation = self._get_observation()
        info = self._get_info()
        info["action_value"] = action_value
        info["position_changed"] = position_changed
        info["pnl"] = net_pnl

        return observation, reward, terminated, truncated, info

    def _map_action(self, action_value: float) -> TradingAction:
        """Map continuous action to discrete trading action"""
        if action_value > self.config.threshold_long:
            return TradingAction.LONG
        elif action_value < self.config.threshold_short:
            return TradingAction.SHORT
        else:
            return TradingAction.HOLD

    def _execute_action(self, target_action: TradingAction) -> Tuple[bool, float]:
        """
        Execute trading action.

        Features:
        - Circuit breaker: Force HOLD after consecutive losses
        - Volatility filter: Force HOLD in extreme volatility

        Returns:
            Tuple of (position_changed, trade_cost)
        """
        current_side = self._portfolio.position.side

        # Circuit breaker - force HOLD during cooldown
        if self._is_in_cooldown() and target_action != TradingAction.HOLD:
            logger.debug(f"Circuit breaker active: forcing HOLD (cooldown until step {self._cooldown_until_step})")
            return False, 0.0

        # Volatility filter - force HOLD in extreme volatility
        if self.config.enable_volatility_filter and target_action != TradingAction.HOLD:
            current_atr = self._get_current_atr()
            if current_atr > self.config.max_atr_multiplier * self._historical_atr_mean:
                logger.debug(f"Volatility filter active: ATR {current_atr:.4f} > {self.config.max_atr_multiplier}x mean")
                return False, 0.0

        if target_action == current_side:
            return False, 0.0

        # Calculate position change magnitude
        old_value = abs(current_side.value)
        new_value = abs(target_action.value)
        change_magnitude = abs(new_value - old_value) + min(old_value, new_value) * 2

        # Calculate cost (transaction + slippage)
        cost = change_magnitude * (
            self.config.transaction_cost + self.config.slippage
        ) * self._portfolio.balance

        # Update position
        self._portfolio.position = Position(
            side=target_action,
            size=1.0 if target_action != TradingAction.HOLD else 0.0,
            entry_price=0.0,  # Price tracking not needed for log returns
            entry_bar=self._current_idx,
        )

        if target_action != TradingAction.HOLD or current_side != TradingAction.HOLD:
            self._portfolio.trades_count += 1

        return True, cost

    def _update_portfolio(self, pnl: float) -> None:
        """Update portfolio state after step"""
        self._portfolio.balance += pnl
        self._portfolio.equity = self._portfolio.balance
        self._portfolio.total_pnl += pnl

        # Update peak and max drawdown
        if self._portfolio.equity > self._portfolio.peak_equity:
            self._portfolio.peak_equity = self._portfolio.equity

        current_dd = self._portfolio.drawdown
        if current_dd > self._portfolio.max_drawdown:
            self._portfolio.max_drawdown = current_dd

        # Track winning/losing trades
        if pnl > 0 and not self._portfolio.position.is_flat:
            self._portfolio.winning_trades += 1
        elif pnl < 0 and not self._portfolio.position.is_flat:
            self._portfolio.losing_trades += 1

        # Update circuit breaker state
        self._update_circuit_breaker(pnl)

    def _update_circuit_breaker(self, pnl: float) -> None:
        """Track consecutive losses for circuit breaker."""
        if pnl < 0:
            self._consecutive_losses += 1
            # Trigger cooldown after threshold
            if self._consecutive_losses >= self.config.max_consecutive_losses:
                self._cooldown_until_step = self._step_count + self.config.cooldown_bars_after_losses
                logger.warning(
                    f"Circuit breaker triggered: {self._consecutive_losses} consecutive losses. "
                    f"Cooldown for {self.config.cooldown_bars_after_losses} bars."
                )
                self._consecutive_losses = 0  # Reset counter
        else:
            # Reset counter on any non-loss
            self._consecutive_losses = 0

    def _is_in_cooldown(self) -> bool:
        """Check if currently in cooldown period."""
        return self._step_count < self._cooldown_until_step

    def _get_current_atr(self) -> float:
        """Get current ATR value for volatility filter."""
        if self._current_idx < len(self.feature_matrix):
            # atr_pct is at index 4 in core_features
            atr_idx = self.config.core_features.index("atr_pct") if "atr_pct" in self.config.core_features else -1
            if atr_idx >= 0:
                return float(self.feature_matrix[self._current_idx, atr_idx])
        return self._historical_atr_mean  # Fallback to mean

    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Max drawdown reached
        if self._portfolio.drawdown > self.config.max_drawdown:
            return True

        # Episode length reached
        if self._step_count >= self.config.episode_length:
            return True

        # End of data
        if self._current_idx >= self.n_bars - 1:
            return True

        # Max position duration
        if not self._portfolio.position.is_flat:
            bars_in_position = self._current_idx - self._portfolio.position.entry_bar
            if bars_in_position > self.config.max_position_duration:
                return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Build current observation"""
        feature_values = self.feature_matrix[self._current_idx]
        position = float(self._portfolio.position.side.value)
        time_normalized = self._step_count / self.config.episode_length

        return self.obs_builder.build(
            feature_values=feature_values,
            position=position,
            time_normalized=time_normalized,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary"""
        return {
            "equity": self._portfolio.equity,
            "balance": self._portfolio.balance,
            "position": self._portfolio.position.side.value,
            "trades": self._portfolio.trades_count,
            "drawdown": self._portfolio.drawdown,
            "max_drawdown": self._portfolio.max_drawdown,
            "total_pnl": self._portfolio.total_pnl,
            "win_rate": self._portfolio.win_rate,
            "step": self._step_count,
            "bar_idx": self._current_idx,
            "episode_return": (self._portfolio.equity / self.config.initial_balance) - 1,
        }

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render environment state"""
        if mode == "human":
            info = self._get_info()
            print(
                f"Step {info['step']:4d} | "
                f"Equity: ${info['equity']:,.2f} | "
                f"Position: {info['position']:+d} | "
                f"Trades: {info['trades']:3d} | "
                f"DD: {info['drawdown']*100:.1f}%"
            )
        return None

    def close(self) -> None:
        """Clean up resources"""
        pass
