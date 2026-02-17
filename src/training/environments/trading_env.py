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
# PIPELINE SSOT INTEGRATION (unified single source of truth)
# =============================================================================
# Load feature configuration from pipeline_ssot.yaml

_PIPELINE_CONFIG = None

def _load_pipeline_config():
    """Load pipeline config from SSOT (cached)."""
    global _PIPELINE_CONFIG
    if _PIPELINE_CONFIG is not None:
        return _PIPELINE_CONFIG
    try:
        from src.config.pipeline_config import load_pipeline_config
        _PIPELINE_CONFIG = load_pipeline_config()
        return _PIPELINE_CONFIG
    except (ImportError, FileNotFoundError):
        return None


def _get_default_core_features() -> List[str]:
    """Get core (market) features from pipeline SSOT or fallback."""
    config = _load_pipeline_config()
    if config:
        return [f.name for f in config.get_market_features()]
    # Fallback: 18 market features matching pipeline_ssot.yaml
    return [
        "log_ret_5m", "log_ret_1h", "log_ret_4h", "log_ret_1d",
        "rsi_9", "rsi_21", "volatility_pct", "trend_z",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "gold_change_1d", "rate_spread_z",
        "rate_spread_change", "usdmxn_change_1d", "yield_curve_z",
    ]


def _get_default_state_features() -> List[str]:
    """Get state features from pipeline SSOT or fallback."""
    config = _load_pipeline_config()
    if config:
        return [f.name for f in config.get_state_features()]
    # Fallback - V22: 9 state features (V21 5 + 4 temporal)
    return [
        "position", "unrealized_pnl", "sl_proximity", "tp_proximity", "bars_held",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]


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
    cumulative_return: float = 0.0  # Cumulative log return since entry

    @property
    def is_flat(self) -> bool:
        return self.side == TradingAction.HOLD

    @property
    def is_long(self) -> bool:
        return self.side == TradingAction.LONG

    @property
    def is_short(self) -> bool:
        return self.side == TradingAction.SHORT

    @property
    def unrealized_pnl_pct(self) -> float:
        """Return unrealized PnL as percentage (from cumulative log return)."""
        # Convert log return to percentage: exp(r) - 1
        return np.exp(self.cumulative_return) - 1.0 if self.cumulative_return != 0 else 0.0


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
    episode_length: int = 2400  # V21: ~40 trading days (was 1200)
    warmup_bars: int = 14  # Bars needed for technical indicators

    # Portfolio settings - OPTIMIZED FOR MEXC/BINANCE USDT/COP
    # MEXC: 0% maker, 0.05% taker = 0.05% round-trip (18x cheaper than retail)
    initial_balance: float = 10_000.0
    transaction_cost_bps: float = 2.5   # MEXC: 0.025% per side
    slippage_bps: float = 2.5           # Minimal slippage on liquid pair

    # Risk management
    max_drawdown_pct: float = 15.0  # Stop episode at 15% drawdown
    max_position_duration: int = 864  # V21: Was 576 → 3 days (allow trades to develop)
    stop_loss_pct: float = -0.04  # V21: Was -0.025 → wider, fewer false stops
    stop_loss_penalty: float = 0.3  # Penalty for hitting stop loss
    take_profit_pct: float = 0.04  # V21: Was 0.03 → symmetric 1:1 R:R
    take_profit_bonus: float = 0.5  # Bonus for hitting take profit
    exit_bonus: float = 0.2  # Bonus for voluntary profitable exit

    # Action thresholds - EXP-B-001: Lower for more trades
    # Creates 100% HOLD zone for balanced trading
    threshold_long: float = 0.35   # V21: was 0.50 (narrower HOLD zone → more trades)
    threshold_short: float = -0.35  # V21: was -0.50

    # Circuit breaker for consecutive losses - DISABLED for training
    circuit_breaker_enabled: bool = False  # SSOT: circuit_breaker.enabled
    max_consecutive_losses: int = 999  # SSOT: circuit_breaker.max_consecutive_losses
    cooldown_bars_after_losses: int = 0  # SSOT: circuit_breaker.cooldown_bars

    # Volatility filter
    enable_volatility_filter: bool = True
    max_atr_multiplier: float = 2.0  # Force HOLD if ATR > 2x historical mean

    # Observation settings - V22: 28 features (19 market + 9 state)
    observation_dim: int = 28
    clip_range: Tuple[float, float] = (-5.0, 5.0)

    # V22 P2: Action space configuration
    action_type: str = "discrete"  # "discrete" or "continuous"
    n_actions: int = 4  # HOLD=0, BUY=1, SELL=2, CLOSE=3

    # Trailing Stop - V21: DISABLED by default (was cutting winners early)
    trailing_stop_enabled: bool = False  # V21: Was True → disabled
    trailing_stop_activation_pct: float = 0.015
    trailing_stop_trail_factor: float = 0.5
    trailing_stop_min_trail_pct: float = 0.007  # V21: match SSOT (was 0.005)
    trailing_stop_bonus: float = 0.25

    # V21: Reward delivery interval (accumulate and deliver every N bars)
    # 25 bars = ~2 hours → 2-3 reward signals per session (59 bars/day)
    reward_interval: int = 25

    # V21.1: Minimum hold period before agent can voluntarily exit
    # Prevents premature exits that create tiny wins (avg +0.74%) vs large losses (-2.12%)
    # 25 bars = ~2 hours — aligns with reward_interval for coherent decision-making
    # SL/TP/max_duration still override (risk management always fires)
    min_hold_bars: int = 25

    # EXP-RL-EXECUTOR: Forecast-constrained mode
    # When True + forecast_signals provided, blocks trades against forecast direction
    forecast_constrained: bool = False

    # EXP-SWING-001: Decision interval — bars between agent decisions
    # 1 = every bar (default, backward compatible), 59 = once per trading day
    # SL/TP/trailing stop are still checked every bar within the interval
    decision_interval: int = 1

    # Stop mode strategy: "fixed_pct" (default) or "atr_dynamic"
    stop_mode: str = "fixed_pct"
    atr_stop: Dict[str, Any] = field(default_factory=dict)

    # Action interpretation: "threshold_3" (default) or "zone_5"
    action_interpretation: str = "threshold_3"
    zone_5_config: Dict[str, Any] = field(default_factory=dict)

    # Feature configuration - reads from experiment SSOT when available
    core_features: List[str] = field(default_factory=lambda: _get_default_core_features())
    state_features: List[str] = field(default_factory=lambda: _get_default_state_features())

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
# Reward Strategy Adapter (Integrates ModularRewardCalculator)
# =============================================================================

class RewardStrategyAdapter:
    """
    Adapter that wraps ModularRewardCalculator to work with TradingEnvironment.

    This adapter integrates the full reward system including:
    - DSR (Differential Sharpe Ratio)
    - Sortino Ratio
    - Regime-based penalties
    - Market impact (Almgren-Chriss)
    - Holding decay (gap risk)
    - Anti-gaming penalties
    - Curriculum learning

    Contract: CTR-REWARD-SNAPSHOT-001
    """

    def __init__(
        self,
        reward_config: Optional["RewardConfig"] = None,
        enable_curriculum: bool = True,
    ):
        """
        Initialize reward strategy adapter.

        Args:
            reward_config: Optional RewardConfig for customization
            enable_curriculum: Enable curriculum learning phases
        """
        from ..reward_calculator import ModularRewardCalculator
        from ..config import RewardConfig

        config = reward_config or RewardConfig()
        self._calculator = ModularRewardCalculator(
            config=config,
            enable_curriculum=enable_curriculum,
        )

        # State tracking
        self._bars_held = 0
        self._consecutive_wins = 0
        self._max_intratrade_dd = 0.0
        self._last_position_is_flat = True
        self._last_action: int = 0

        # Track volatility for regime detection
        self._volatility_window: List[float] = []
        self._volatility_window_size = 20

    def calculate(
        self,
        pnl: float,
        portfolio: "PortfolioState",
        position_changed: bool,
        market_return: float,
        step_count: int,
        episode_length: int,
        volatility: Optional[float] = None,
        hour_utc: int = 12,
        is_overnight: bool = False,
        oil_return: Optional[float] = None,
        spread_bid_ask: float = 0.0,
        close_reason: str = "",
        forecast_direction: int = 0,
        session_open_price: float = 0.0,
        current_price: float = 0.0,
    ) -> float:
        """
        Calculate reward using ModularRewardCalculator.

        Args:
            pnl: PnL in currency units
            portfolio: Current portfolio state
            position_changed: Whether position changed this step
            market_return: Log return of underlying
            step_count: Current step in episode
            episode_length: Total episode length
            volatility: Optional ATR or volatility measure
            hour_utc: Hour in UTC (0-23)
            is_overnight: Whether this is an overnight position
            oil_return: Optional oil price return (for correlation tracking)
            spread_bid_ask: Bid-ask spread in bps

        Returns:
            Calculated reward
        """
        initial_balance = portfolio.balance - portfolio.total_pnl

        # Calculate PnL percentage
        pnl_pct = pnl / initial_balance if initial_balance > 0 else 0.0

        # Track position state
        is_flat = portfolio.position.is_flat
        position = 0 if is_flat else (1 if portfolio.position.is_long else -1)

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

        # Estimate volatility if not provided
        if volatility is None:
            self._volatility_window.append(abs(market_return))
            if len(self._volatility_window) > self._volatility_window_size:
                self._volatility_window.pop(0)
            volatility = np.std(self._volatility_window) if len(self._volatility_window) > 1 else 0.01

        # Calculate action (discrete: -1, 0, 1)
        action = position

        # Call modular reward calculator
        reward, breakdown = self._calculator.calculate(
            pnl_pct=pnl_pct,
            position=position,
            position_change=position_change,
            holding_bars=self._bars_held,
            volatility=volatility,
            hour_utc=hour_utc,
            is_overnight=is_overnight,
            oil_return=oil_return,
            price_change=pnl_pct,  # Use PnL as price change proxy
            close_reason=close_reason,
            forecast_direction=forecast_direction,
            session_open_price=session_open_price,
            current_price=current_price,
            has_position=(position != 0),
        )

        self._last_position_is_flat = is_flat
        self._last_action = action

        return reward

    def reset(self):
        """Reset internal state for new episode."""
        self._bars_held = 0
        self._consecutive_wins = 0
        self._max_intratrade_dd = 0.0
        self._last_position_is_flat = True
        self._last_action = 0
        self._volatility_window = []
        self._calculator.reset()

    @property
    def curriculum_phase(self) -> Optional[str]:
        """Get current curriculum phase."""
        return self._calculator.get_curriculum_phase()

    @property
    def curriculum_stats(self) -> Optional[Dict[str, Any]]:
        """Get curriculum statistics."""
        return self._calculator.get_curriculum_stats()

    def get_reward_breakdown(self) -> Optional[Dict[str, float]]:
        """Get the breakdown of the last reward calculation."""
        last_breakdown = getattr(self._calculator, '_last_breakdown', None)
        if last_breakdown:
            return last_breakdown.to_dict()
        return None


class ModularRewardStrategyAdapter(RewardStrategyAdapter):
    """
    Extended adapter with full modular reward system support.

    Provides additional hooks for:
    - Curriculum learning callbacks
    - Reward breakdown logging
    - Regime tracking
    """

    def __init__(
        self,
        reward_config: Optional["RewardConfig"] = None,
        enable_curriculum: bool = True,
        on_phase_change: Optional[callable] = None,
    ):
        """
        Initialize modular reward strategy adapter.

        Args:
            reward_config: Optional RewardConfig
            enable_curriculum: Enable curriculum learning
            on_phase_change: Callback for curriculum phase changes
        """
        super().__init__(reward_config, enable_curriculum)
        self._on_phase_change = on_phase_change
        self._regime_history: List[str] = []

    def step(self) -> bool:
        """
        Advance curriculum by one step.

        Returns:
            True if curriculum phase changed
        """
        old_phase = self.curriculum_phase
        self._calculator.step_curriculum()
        new_phase = self.curriculum_phase

        if old_phase != new_phase and self._on_phase_change:
            self._on_phase_change(old_phase, new_phase)
            return True
        return False

    def get_regime_distribution(self) -> Dict[str, int]:
        """Get distribution of regimes seen during training."""
        from collections import Counter
        return dict(Counter(self._regime_history))


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
        loss_multiplier: float = 1.2,  # FIX: 2.0→1.2 (less aggressive loss penalty)
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

    V21: Expanded to 5 state features (position, unrealized_pnl,
         sl_proximity, tp_proximity, bars_held).
    """

    def __init__(
        self,
        norm_stats: Dict[str, Dict[str, float]],
        core_features: List[str],
        state_features: List[str],
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ):
        self.norm_stats = norm_stats
        self.core_features = core_features
        self.state_features = state_features
        self.clip_range = clip_range
        self.n_state_features = len(state_features)
        self.observation_dim = len(core_features) + self.n_state_features

    def build(
        self,
        feature_values: np.ndarray,
        state_values: Dict[str, float],
    ) -> np.ndarray:
        """
        Build observation vector.

        Args:
            feature_values: Raw feature values (len = core_features)
            state_values: Dict of state feature values
                         (position, unrealized_pnl, sl_proximity, tp_proximity, bars_held)

        Returns:
            Normalized observation array
        """
        obs = np.zeros(self.observation_dim, dtype=np.float32)

        # Normalize core features
        for i, (fname, value) in enumerate(zip(self.core_features, feature_values)):
            obs[i] = self._normalize(fname, value)

        # State features (V21: 5 features after market features)
        n_core = len(self.core_features)
        for j, sf_name in enumerate(self.state_features):
            val = state_values.get(sf_name, 0.0)
            obs[n_core + j] = float(np.clip(val, -1.0, 1.0))

        # Final clip and NaN handling
        obs = np.clip(obs, self.clip_range[0], self.clip_range[1])
        obs = np.nan_to_num(obs, nan=0.0, posinf=self.clip_range[1], neginf=self.clip_range[0])

        return obs

    def _normalize(self, feature: str, value: float) -> float:
        """Normalize a feature value for observation.

        Uses the 'method' field from norm_stats to decide whether to normalize.
        Features already normalized by L2 (method=zscore/minmax/none) are just clipped.
        Only features without method metadata get z-score normalized here.
        """
        # Skip features ending in _z (pre-z-scored by calculator)
        if feature.endswith('_z'):
            return float(np.clip(value, self.clip_range[0], self.clip_range[1]))

        if feature not in self.norm_stats:
            return float(np.clip(value, self.clip_range[0], self.clip_range[1]))

        stats = self.norm_stats[feature]
        method = stats.get("method", "unknown")

        # If L2 already normalized this feature, just clip (no re-normalization)
        if method in ("zscore", "minmax", "none"):
            return float(np.clip(value, self.clip_range[0], self.clip_range[1]))

        # Fallback for features without method metadata
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
        stop_strategy: Optional["StopStrategy"] = None,
        action_interpreter: Optional["ActionInterpreter"] = None,
        forecast_signals: Optional[Dict[str, Tuple[int, float]]] = None,
    ):
        """
        Initialize trading environment.

        Args:
            df: DataFrame with OHLCV and feature data
            norm_stats: Normalization statistics {feature: {mean, std}}
            config: Environment configuration
            reward_strategy: Optional custom reward strategy
            stop_strategy: Optional stop strategy (default: FixedPctStopStrategy from config)
            action_interpreter: Optional action interpreter (default: ThresholdInterpreter)
            forecast_signals: Optional dict mapping date string -> (direction, leverage).
                              When provided, adds 3 forecast state features to observation
                              and enables direction constraint if config.forecast_constrained=True.
        """
        super().__init__()

        self.config = config
        # Use RewardStrategyAdapter by default for full reward function
        self.reward_strategy = reward_strategy or RewardStrategyAdapter()

        # Stop strategy (defaults to FixedPctStopStrategy from config)
        if stop_strategy is None:
            from src.training.environments.stop_strategies import create_stop_strategy
            self._stop_strategy = create_stop_strategy(config)
        else:
            self._stop_strategy = stop_strategy

        # Action interpreter (defaults to ThresholdInterpreter from config)
        if action_interpreter is None:
            from src.training.environments.action_interpreters import create_action_interpreter
            self._action_interpreter = create_action_interpreter(config)
        else:
            self._action_interpreter = action_interpreter

        # Validate and prepare data
        self._validate_dataframe(df)
        self.df = df.reset_index(drop=True)
        self.n_bars = len(df)

        # Extract feature matrix for fast access
        self.feature_matrix = df[config.core_features].values.astype(np.float32)

        # CRITICAL FIX (v2.1): Use raw returns for PnL calculation
        # raw_log_ret_* contains actual return magnitudes (~0.0001 per bar)
        # log_ret_* is z-scored and would cause 100x-1000x inflated PnL
        raw_ret_col = next((c for c in df.columns if c.startswith("raw_log_ret_")), None)
        if raw_ret_col:
            self.returns = df[raw_ret_col].values.astype(np.float32)
            logger.info(f"[ENV] Using {raw_ret_col} for PnL calculation (correct)")
        else:
            # Backwards compatibility: find first log_ret feature
            first_ret_col = next((c for c in df.columns if c.startswith("log_ret_")), None)
            if first_ret_col:
                self.returns = df[first_ret_col].values.astype(np.float32)
                logger.warning(f"[ENV] raw_log_ret_* not found, using {first_ret_col} (may be normalized!)")
            else:
                raise ValueError("No raw_log_ret_* or log_ret_* column found in DataFrame")

        # Observation builder (V21: pass state_features for expanded obs)
        self.obs_builder = EnvObservationBuilder(
            norm_stats=norm_stats,
            core_features=config.core_features,
            state_features=config.state_features,
            clip_range=config.clip_range,
        )

        # Action space: V22 supports both discrete and continuous
        if config.action_type == "discrete":
            # V22 P2: Discrete(4) — HOLD=0, BUY=1, SELL=2, CLOSE=3
            self.action_space = spaces.Discrete(config.n_actions)
        else:
            # V21 legacy: continuous [-1, 1]
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

        # EXP-B-001: Trailing stop state
        self._trailing_stop_active: bool = False
        self._peak_unrealized_pnl: float = 0.0
        self._trailing_stop_triggered_count: int = 0

        # V21: Reward accumulation buffer (also reset in reset())
        self._reward_accumulator: float = 0.0
        self._bars_since_reward: int = 0

        # EXP-RL-EXECUTOR: Forecast signal integration
        self._forecast_signals = forecast_signals
        self._session_open_price: float = 0.0
        self._current_session_date: str = ""

        # Historical volatility for volatility filter
        # Support both new (volatility_pct) and legacy (atr_pct) column names
        volatility_col = None
        for col in ["volatility_pct", "atr_pct"]:
            if col in df.columns:
                volatility_col = col
                break

        if config.enable_volatility_filter and volatility_col:
            self._historical_atr_mean = df[volatility_col].mean()
            self._historical_atr_std = df[volatility_col].std()
            self._volatility_column = volatility_col
        else:
            self._historical_atr_mean = 0.01  # Default fallback
            self._historical_atr_std = 0.005
            self._volatility_column = None

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

        has_ret = any(c.startswith("log_ret_") or c.startswith("raw_log_ret_") for c in df.columns)
        if not has_ret:
            raise ValueError("DataFrame must have a 'log_ret_*' or 'raw_log_ret_*' column for returns")

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

        # EXP-B-001: Reset trailing stop state
        self._trailing_stop_active = False
        self._peak_unrealized_pnl = 0.0

        # V21: Reset reward accumulation buffer
        self._reward_accumulator = 0.0
        self._bars_since_reward = 0

        # V22 P2: Close reason tracking
        self._last_close_reason: str = ""

        # Action interpreter: position size from last action
        self._last_position_size: float = 1.0

        # Reset reward strategy if it has reset method
        if hasattr(self.reward_strategy, 'reset'):
            self.reward_strategy.reset()

        # EXP-RL-EXECUTOR: Reset session tracking
        self._session_open_price = 0.0
        self._current_session_date = ""
        self._update_session_tracking()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one decision step, advancing decision_interval bars.

        EXP-SWING-001: When decision_interval > 1, the agent's action is applied
        on the first bar, then HOLD is auto-applied for the remaining bars.
        SL/TP/trailing stops are checked every bar (risk management always fires).
        Rewards are accumulated across all bars in the interval.

        Backward compatible: decision_interval=1 delegates directly to _step_single().
        """
        if self.config.decision_interval <= 1:
            return self._step_single(action)

        total_reward = 0.0
        info = {}
        bars_processed = 0

        for bar_idx in range(self.config.decision_interval):
            if bar_idx == 0:
                # First bar: apply the agent's actual action
                obs, reward, terminated, truncated, info = self._step_single(action)
            else:
                # Subsequent bars: auto-HOLD (maintain current position)
                if self.config.action_type == "discrete":
                    hold_action = 0  # HOLD in Discrete(4)
                else:
                    hold_action = np.array([0.0], dtype=np.float32)  # HOLD in continuous
                obs, reward, terminated, truncated, info = self._step_single(hold_action)

            total_reward += reward
            bars_processed = bar_idx + 1

            if terminated or truncated:
                break

        info['decision_bars_processed'] = bars_processed
        return obs, total_reward, terminated, truncated, info

    def _step_single(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one single-bar step in the environment.

        Args:
            action: int for Discrete(4) or np.ndarray for continuous Box(1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # V22: Handle both discrete and continuous actions
        close_requested = False
        if self.config.action_type == "discrete":
            action_value = int(action)
            target_action, close_requested = self._map_discrete_action(action_value)
        else:
            action_value = float(action[0])
            target_action = self._map_action(action_value)

        stop_loss_triggered = False
        stop_loss_penalty = 0.0
        take_profit_triggered = False
        take_profit_bonus = 0.0
        exit_bonus = 0.0

        # Track pre-trade position for exit bonus calculation
        was_in_position = not self._portfolio.position.is_flat
        pre_trade_unrealized_pnl = self._portfolio.position.unrealized_pnl_pct if was_in_position else 0.0

        # V22 FIX #1: Enforce min_hold_bars on ALL exit actions (CLOSE + reversals)
        # This guard runs BEFORE any action processing to prevent premature exits
        if was_in_position and self.config.action_type == "discrete":
            bars_held = self._current_idx - self._portfolio.position.entry_bar
            if bars_held < self.config.min_hold_bars:
                current_side = self._portfolio.position.side
                # Block CLOSE action
                if close_requested:
                    close_requested = False
                    target_action = current_side  # Force maintain current position
                # Block reversals (BUY while SHORT, SELL while LONG)
                elif target_action != current_side and target_action != TradingAction.HOLD:
                    target_action = current_side  # Force maintain current position

        # EXP-RL-EXECUTOR: Enforce forecast direction constraint
        # Block trades against forecast direction when forecast_constrained=True
        if self.config.forecast_constrained and self._forecast_signals:
            forecast_dir, _ = self._get_current_forecast()
            if forecast_dir != 0:
                # Block LONG when forecast is SHORT
                if forecast_dir == -1 and target_action == TradingAction.LONG:
                    target_action = TradingAction.HOLD
                # Block SHORT when forecast is LONG
                elif forecast_dir == 1 and target_action == TradingAction.SHORT:
                    target_action = TradingAction.HOLD

        # V22 P2: Handle CLOSE action (flatten position)
        if close_requested and was_in_position:
            self._last_close_reason = "agent_close"
            flatten_cost = self._execute_flatten()
            position_changed = True
            trade_cost = flatten_cost
        else:
            # Execute trade if position changes
            # V22 P2: Track agent_reverse close reason
            if was_in_position and target_action != self._portfolio.position.side and target_action != TradingAction.HOLD:
                self._last_close_reason = "agent_reverse"
            position_changed, trade_cost = self._execute_action(target_action)

        # Move to next bar
        self._current_idx += 1
        self._step_count += 1

        # EXP-RL-EXECUTOR: Track session open price for execution alpha
        self._update_session_tracking()

        # Calculate PnL from market return (scaled by position size for zone_5)
        market_return = self.returns[self._current_idx]
        position_value = self._portfolio.position.side.value  # -1, 0, or 1
        position_size = self._portfolio.position.size  # 0.0, 0.5, or 1.0
        gross_pnl = position_value * position_size * market_return * self._portfolio.balance
        net_pnl = gross_pnl - trade_cost

        # PHASE 3: Track cumulative return for stop-loss (scaled by size)
        if not self._portfolio.position.is_flat:
            # Accumulate signed return based on position direction and size
            self._portfolio.position.cumulative_return += position_value * position_size * market_return

        # PHASE 3: Check stop-loss BEFORE updating portfolio
        if self._check_stop_loss():
            stop_loss_triggered = True
            stop_loss_penalty = self.config.stop_loss_penalty
            self._last_close_reason = "stop_loss"
            flatten_cost = self._execute_flatten()
            net_pnl -= flatten_cost

        # PHASE 4: Check take-profit BEFORE updating portfolio
        if not stop_loss_triggered and self._check_take_profit():
            take_profit_triggered = True
            take_profit_bonus = self.config.take_profit_bonus
            self._last_close_reason = "take_profit"
            flatten_cost = self._execute_flatten()
            net_pnl -= flatten_cost

        # EXP-B-001: Check trailing stop (after take-profit, before normal action)
        trailing_stop_triggered = False
        trailing_stop_bonus = 0.0
        if not stop_loss_triggered and not take_profit_triggered:
            if self._check_trailing_stop():
                trailing_stop_triggered = True
                trailing_stop_bonus = self.config.trailing_stop_bonus
                self._last_close_reason = "trailing_stop"
                flatten_cost = self._execute_flatten()
                net_pnl -= flatten_cost
                self._trailing_stop_triggered_count += 1
                logger.info(f"[TRAILING-STOP] Position closed")

        # PHASE 4: Check for voluntary profitable exit (agent chose to exit with profit)
        if was_in_position and position_changed and target_action == TradingAction.HOLD:
            if pre_trade_unrealized_pnl > 0:
                exit_bonus = self.config.exit_bonus * min(1.0, pre_trade_unrealized_pnl / 0.02)  # Scale by profit
                logger.debug(f"[EXIT-BONUS] Voluntary profitable exit: +{exit_bonus:.3f}")

        # Update portfolio
        self._update_portfolio(net_pnl)

        # Get additional context for reward calculation
        volatility = self._get_current_atr()

        # Get hour from timestamp if available
        hour_utc = 12  # Default
        if hasattr(self.df, 'index') and hasattr(self.df.index, 'hour'):
            try:
                hour_utc = self.df.index[self._current_idx].hour
            except (IndexError, AttributeError):
                pass
        elif 'datetime' in self.df.columns:
            try:
                hour_utc = pd.to_datetime(self.df.iloc[self._current_idx]['datetime']).hour
            except (KeyError, IndexError, TypeError):
                pass

        # Check if overnight position (after market close or before open)
        is_overnight = hour_utc < 8 or hour_utc >= 22

        # Get oil return if available
        oil_return = None
        if 'brent_change_1d' in self.df.columns:
            try:
                oil_return = float(self.df.iloc[self._current_idx]['brent_change_1d'])
            except (KeyError, IndexError, TypeError):
                pass

        # Calculate reward using the strategy
        # Handle both old and new strategy interfaces
        if hasattr(self.reward_strategy, 'calculate'):
            import inspect
            sig = inspect.signature(self.reward_strategy.calculate)
            params = sig.parameters

            if 'volatility' in params:
                # New modular reward strategy — V22: pass close_reason for shaping
                # EXP-RL-EXECUTOR: pass forecast context for execution alpha
                fc_kwargs = {}
                if self._forecast_signals:
                    fc_dir, _ = self._get_current_forecast()
                    fc_kwargs["forecast_direction"] = fc_dir
                    fc_kwargs["session_open_price"] = self._session_open_price
                    if 'close' in self.df.columns:
                        fc_kwargs["current_price"] = float(self.df.iloc[self._current_idx]['close'])

                reward = self.reward_strategy.calculate(
                    pnl=net_pnl,
                    portfolio=self._portfolio,
                    position_changed=position_changed,
                    market_return=market_return,
                    step_count=self._step_count,
                    episode_length=self.config.episode_length,
                    volatility=volatility,
                    hour_utc=hour_utc,
                    is_overnight=is_overnight,
                    oil_return=oil_return,
                    close_reason=self._last_close_reason,
                    **fc_kwargs,
                )
            else:
                # Legacy reward strategy
                reward = self.reward_strategy.calculate(
                    pnl=net_pnl,
                    portfolio=self._portfolio,
                    position_changed=position_changed,
                    market_return=market_return,
                    step_count=self._step_count,
                    episode_length=self.config.episode_length,
                )
        else:
            reward = 0.0

        # FIX 8: Removed all reward bonuses/penalties for stop-loss, take-profit,
        # trailing stop, and exit. These mechanisms still EXECUTE (risk management)
        # but don't add extra reward. The PnL impact of closing is already captured
        # by the ModularRewardCalculator's PnL component. The bonuses were 5-50x
        # larger than per-step PnL reward, causing the agent to optimize for
        # triggering bonuses rather than trading profitably.

        # V21 FIX 5: Reward interval delivery (accumulate, deliver every N bars)
        # With 59 bars/session and interval=25: 2-3 reward signals per day
        # Filters 5-min noise while giving PPO enough gradient signal
        self._reward_accumulator += reward
        self._bars_since_reward += 1

        position_just_closed = (
            stop_loss_triggered or take_profit_triggered or trailing_stop_triggered
            or (was_in_position and self._portfolio.position.is_flat)
        )

        if self._bars_since_reward >= self.config.reward_interval or position_just_closed:
            # Deliver accumulated reward
            reward = float(np.clip(self._reward_accumulator, -2.0, 2.0))
            self._reward_accumulator = 0.0
            self._bars_since_reward = 0
        else:
            # Suppress reward — accumulate silently
            reward = 0.0

        # Check termination
        terminated = self._check_termination()
        truncated = False

        observation = self._get_observation()
        info = self._get_info()
        info["action_value"] = action_value
        info["position_changed"] = position_changed
        info["pnl"] = net_pnl
        info["stop_loss_triggered"] = stop_loss_triggered
        info["take_profit_triggered"] = take_profit_triggered
        info["trailing_stop_triggered"] = trailing_stop_triggered
        info["exit_bonus"] = exit_bonus
        info["overtime_penalty"] = 0.0  # FIX 8: bonuses/penalties removed from reward
        info["close_reason"] = self._last_close_reason  # V22 P2

        # EXP-RL-EXECUTOR: Add forecast context for execution alpha
        if self._forecast_signals:
            fc_dir, fc_lev = self._get_current_forecast()
            info["forecast_direction"] = fc_dir
            info["forecast_leverage"] = fc_lev
            info["session_open_price"] = self._session_open_price
            if 'close' in self.df.columns:
                info["current_price"] = float(self.df.iloc[self._current_idx]['close'])

        # Reset close reason after reporting
        if self._last_close_reason:
            self._last_close_reason = ""

        return observation, reward, terminated, truncated, info

    def _map_action(self, action_value: float) -> TradingAction:
        """Map continuous action to trading action via action interpreter."""
        action, size = self._action_interpreter.interpret(action_value)
        # Store size for PnL scaling (used by _execute_action)
        self._last_position_size = size
        return action

    def _map_discrete_action(self, action: int) -> Tuple[TradingAction, bool]:
        """
        V22 P2: Map Discrete(4) action to TradingAction + close flag.

        Actions:
            0 = HOLD: Stay flat or maintain current position
            1 = BUY: Open long or maintain long (reverse from short)
            2 = SELL: Open short or maintain short (reverse from long)
            3 = CLOSE: Flatten to HOLD

        Returns:
            Tuple of (target_action, close_requested)
        """
        if action == 0:  # HOLD
            if self._portfolio.position.is_flat:
                return TradingAction.HOLD, False
            else:
                return self._portfolio.position.side, False
        elif action == 1:  # BUY
            return TradingAction.LONG, False
        elif action == 2:  # SELL
            return TradingAction.SHORT, False
        elif action == 3:  # CLOSE
            return TradingAction.HOLD, True
        return TradingAction.HOLD, False

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

        # V21.1: Minimum hold period — prevent premature voluntary exits/reversals
        # SL/TP/max_duration are checked separately and always fire (before _execute_action)
        if current_side != TradingAction.HOLD and target_action != current_side:
            bars_held = self._current_idx - self._portfolio.position.entry_bar
            if bars_held < self.config.min_hold_bars:
                return False, 0.0  # Ignore exit/reversal signal, keep position

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

        # FIX 4: Track trade outcome on position closure (not per step)
        if current_side != TradingAction.HOLD and target_action != current_side:
            trade_pnl = self._portfolio.position.unrealized_pnl_pct
            if trade_pnl > 0:
                self._portfolio.winning_trades += 1
            elif trade_pnl < 0:
                self._portfolio.losing_trades += 1

        # Calculate position change magnitude
        old_value = abs(current_side.value)
        new_value = abs(target_action.value)
        change_magnitude = abs(new_value - old_value) + min(old_value, new_value) * 2

        # Calculate cost (transaction + slippage)
        cost = change_magnitude * (
            self.config.transaction_cost + self.config.slippage
        ) * self._portfolio.balance

        # Update position (size from action interpreter, default 1.0)
        position_size = self._last_position_size if target_action != TradingAction.HOLD else 0.0
        self._portfolio.position = Position(
            side=target_action,
            size=position_size,
            entry_price=0.0,  # Price tracking not needed for log returns
            entry_bar=self._current_idx,
        )

        if target_action != TradingAction.HOLD or current_side != TradingAction.HOLD:
            self._portfolio.trades_count += 1

        # Notify stop strategy when opening a new position
        if target_action != TradingAction.HOLD and current_side == TradingAction.HOLD:
            self._stop_strategy.on_position_open(atr_pct=self._get_current_atr())
        elif target_action == TradingAction.HOLD:
            self._stop_strategy.on_position_close()

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

        # FIX 4: Win/loss tracking moved to _execute_action() (per trade closure)

        # Update circuit breaker state
        self._update_circuit_breaker(pnl)

    def _update_circuit_breaker(self, pnl: float) -> None:
        """Track consecutive losses for circuit breaker."""
        # Skip if circuit breaker is disabled (for training)
        if not self.config.circuit_breaker_enabled:
            return

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

    def _check_stop_loss(self) -> bool:
        """Check if position hit stop-loss via stop strategy.

        Delegates to self._stop_strategy.check_stop() which supports
        both fixed % and ATR-dynamic modes.
        """
        if self._portfolio.position.is_flat:
            return False

        unrealized_pnl_pct = self._portfolio.position.unrealized_pnl_pct
        bars_held = self._current_idx - self._portfolio.position.entry_bar
        result = self._stop_strategy.check_stop(unrealized_pnl_pct, bars_held)
        return result == "stop_loss"

    def _check_take_profit(self) -> bool:
        """Check if position hit take-profit via stop strategy.

        Delegates to self._stop_strategy.check_stop() which supports
        both fixed % and ATR-dynamic modes.
        """
        if self._portfolio.position.is_flat:
            return False

        unrealized_pnl_pct = self._portfolio.position.unrealized_pnl_pct
        bars_held = self._current_idx - self._portfolio.position.entry_bar
        result = self._stop_strategy.check_stop(unrealized_pnl_pct, bars_held)
        return result == "take_profit"

    def _check_trailing_stop(self) -> bool:
        """
        EXP-B-001: Check if trailing stop triggered.

        Trailing stop dynamically locks in profits by:
        1. Activating once unrealized PnL exceeds activation_pct
        2. Tracking peak unrealized PnL
        3. Triggering exit if price drops below trail level

        Returns:
            True if trailing stop triggered and should flatten
        """
        if not self.config.trailing_stop_enabled:
            return False

        if self._portfolio.position.is_flat:
            # Reset trailing stop state when flat
            self._trailing_stop_active = False
            self._peak_unrealized_pnl = 0.0
            return False

        unrealized_pnl_pct = self._portfolio.position.unrealized_pnl_pct

        # Update peak unrealized PnL
        if unrealized_pnl_pct > self._peak_unrealized_pnl:
            self._peak_unrealized_pnl = unrealized_pnl_pct

        # Check activation threshold
        if self._peak_unrealized_pnl < self.config.trailing_stop_activation_pct:
            return False

        # Now trailing stop is active
        self._trailing_stop_active = True

        # Calculate trail level
        trail_level = self._peak_unrealized_pnl * self.config.trailing_stop_trail_factor
        trail_level = max(trail_level, self.config.trailing_stop_min_trail_pct)

        # Check if dropped below trail level
        if unrealized_pnl_pct < trail_level:
            logger.info(
                f"[TRAILING-STOP] Triggered: Peak={self._peak_unrealized_pnl:.2%}, "
                f"Trail={trail_level:.2%}, Current={unrealized_pnl_pct:.2%}"
            )
            return True

        return False

    def _execute_flatten(self) -> float:
        """
        Force flatten position (for stop-loss/take-profit/trailing stop).

        Returns:
            Trade cost incurred
        """
        if self._portfolio.position.is_flat:
            return 0.0

        # FIX 4: Track trade outcome on forced closure
        trade_pnl = self._portfolio.position.unrealized_pnl_pct
        if trade_pnl > 0:
            self._portfolio.winning_trades += 1
        elif trade_pnl < 0:
            self._portfolio.losing_trades += 1

        # Calculate cost to flatten
        cost = (
            self.config.transaction_cost + self.config.slippage
        ) * self._portfolio.balance

        # Reset position
        self._portfolio.position = Position(
            side=TradingAction.HOLD,
            size=0.0,
            entry_price=0.0,
            entry_bar=self._current_idx,
        )
        self._portfolio.trades_count += 1

        # FIX 9: Reset trailing stop state on flatten to prevent thrashing.
        # Without this, peak_unrealized_pnl from the old trade persists and
        # immediately triggers trailing stop on the next entry.
        self._trailing_stop_active = False
        self._peak_unrealized_pnl = 0.0

        # Notify stop strategy that position is closed
        self._stop_strategy.on_position_close()

        logger.debug(f"[FLATTEN] Position flattened, cost: ${cost:.2f}")
        return cost

    def _get_current_atr(self) -> float:
        """Get current volatility value for volatility filter."""
        if self._current_idx < len(self.feature_matrix):
            # Support both volatility_pct (new) and atr_pct (legacy)
            for col_name in ["volatility_pct", "atr_pct"]:
                if col_name in self.config.core_features:
                    vol_idx = self.config.core_features.index(col_name)
                    return float(self.feature_matrix[self._current_idx, vol_idx])
        return self._historical_atr_mean  # Fallback to mean

    def _calculate_overtime_penalty(self) -> float:
        """
        Calculate penalty for holding position beyond max_position_duration.

        PHASE 3 FIX: Instead of terminating episode (forcing exit at random price),
        apply an increasing penalty to discourage extended holding while letting
        the agent decide when to exit.

        Returns:
            Overtime penalty (0 if within duration limit)
        """
        if self._portfolio.position.is_flat:
            return 0.0

        bars_in_position = self._current_idx - self._portfolio.position.entry_bar
        if bars_in_position <= self.config.max_position_duration:
            return 0.0

        # Apply increasing penalty: 0.01 per bar over limit, max 0.5
        overtime_bars = bars_in_position - self.config.max_position_duration
        overtime_penalty = min(0.5, overtime_bars * 0.01)

        if overtime_bars == 1:  # Log only on first overtime bar
            logger.debug(
                f"[OVERTIME] Position held {bars_in_position} bars "
                f"(limit: {self.config.max_position_duration}), penalty: {overtime_penalty:.3f}"
            )

        return overtime_penalty

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

        # PHASE 3 FIX: Removed max_position_duration termination
        # Now handled via overtime penalty in step() - agent decides when to exit
        # This prevents forced exit at potentially bad prices

        return False

    def _get_observation(self) -> np.ndarray:
        """Build current observation with V21 risk visibility features."""
        feature_values = self.feature_matrix[self._current_idx]
        position = float(self._portfolio.position.side.value)

        # Unrealized PnL (from cumulative_return)
        if not self._portfolio.position.is_flat:
            unrealized_pnl = float(np.clip(
                self._portfolio.position.unrealized_pnl_pct, -1.0, 1.0
            ))
        else:
            unrealized_pnl = 0.0

        # V21 FIX 2: Risk visibility features
        if not self._portfolio.position.is_flat:
            unrealized_pnl_raw = self._portfolio.position.unrealized_pnl_pct

            # Feature 21: Stop-loss proximity (0=at stop, 1=far from stop)
            sl_distance = unrealized_pnl_raw - self.config.stop_loss_pct
            sl_proximity = float(np.clip(sl_distance / abs(self.config.stop_loss_pct), 0.0, 2.0) / 2.0)

            # Feature 22: Take-profit proximity (0=far, 1=at TP)
            tp_proximity = float(np.clip(unrealized_pnl_raw / self.config.take_profit_pct, 0.0, 1.0))

            # Feature 23: Bars in position (normalized 0-1)
            bars_in_pos = self._current_idx - self._portfolio.position.entry_bar
            bars_held = float(np.clip(bars_in_pos / self.config.max_position_duration, 0.0, 1.0))
        else:
            sl_proximity = 1.0  # Safe when flat
            tp_proximity = 0.0
            bars_held = 0.0

        # V22 P1: Temporal features from current bar's timestamp
        hour_sin = 0.0
        hour_cos = 1.0
        dow_sin = 0.0
        dow_cos = 1.0
        if hasattr(self.df, 'index') and hasattr(self.df.index, 'hour'):
            try:
                ts = self.df.index[self._current_idx]
                # USDCOP trades 8:00-12:55 COT. After L2 tz-strip, hours are 8-12.
                # Auto-detect: if hour >= 13, assume pre-fix UTC labels; else COT.
                session_start = 8.0 if ts.hour < 13 else 13.0
                hour_frac = (ts.hour + ts.minute / 60.0 - session_start) / 5.0
                hour_frac = max(0.0, min(1.0, hour_frac))
                dow_frac = ts.dayofweek / 5.0  # CRITICAL: /5.0 for 5 trading days, NOT /7.0
                hour_sin = float(np.sin(2 * np.pi * hour_frac))
                hour_cos = float(np.cos(2 * np.pi * hour_frac))
                dow_sin = float(np.sin(2 * np.pi * dow_frac))
                dow_cos = float(np.cos(2 * np.pi * dow_frac))
            except (IndexError, AttributeError):
                pass
        elif 'datetime' in self.df.columns:
            try:
                ts = pd.to_datetime(self.df.iloc[self._current_idx]['datetime'])
                session_start = 8.0 if ts.hour < 13 else 13.0
                hour_frac = (ts.hour + ts.minute / 60.0 - session_start) / 5.0
                hour_frac = max(0.0, min(1.0, hour_frac))
                dow_frac = ts.dayofweek / 5.0  # CRITICAL: /5.0 for 5 trading days
                hour_sin = float(np.sin(2 * np.pi * hour_frac))
                hour_cos = float(np.cos(2 * np.pi * hour_frac))
                dow_sin = float(np.sin(2 * np.pi * dow_frac))
                dow_cos = float(np.cos(2 * np.pi * dow_frac))
            except (KeyError, IndexError, TypeError):
                pass

        state_values = {
            "position": position,
            "unrealized_pnl": unrealized_pnl,
            "sl_proximity": sl_proximity,
            "tp_proximity": tp_proximity,
            "bars_held": bars_held,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "dow_sin": dow_sin,
            "dow_cos": dow_cos,
        }

        # EXP-RL-EXECUTOR: Append 3 forecast features when forecast_signals provided
        if self._forecast_signals:
            fc_dir, fc_lev = self._get_current_forecast()
            state_values["forecast_direction"] = float(fc_dir)  # -1, 0, +1
            # Normalize leverage: [0.5, 2.0] -> [0, 1]
            state_values["forecast_leverage_norm"] = float(np.clip((fc_lev - 0.5) / 1.5, 0.0, 1.0))
            # Intraday progress: position of current bar within trading session [0, 1]
            intraday_progress = 0.5  # Default
            try:
                if hasattr(self.df, 'index') and hasattr(self.df.index, 'hour'):
                    ts = self.df.index[self._current_idx]
                    # COT session 8:00-12:55. Auto-detect offset.
                    hour_min = ts.hour + ts.minute / 60.0
                    sess_start = 8.0 if ts.hour < 13 else 13.0
                    intraday_progress = float(np.clip((hour_min - sess_start) / 5.0, 0.0, 1.0))
                elif 'datetime' in self.df.columns:
                    ts = pd.to_datetime(self.df.iloc[self._current_idx]['datetime'])
                    hour_min = ts.hour + ts.minute / 60.0
                    sess_start = 8.0 if ts.hour < 13 else 13.0
                    intraday_progress = float(np.clip((hour_min - sess_start) / 5.0, 0.0, 1.0))
            except (IndexError, AttributeError, TypeError):
                pass
            state_values["intraday_progress"] = intraday_progress

        return self.obs_builder.build(
            feature_values=feature_values,
            state_values=state_values,
        )

    def _get_current_forecast(self) -> Tuple[int, float]:
        """Get forecast direction and leverage for current bar's date.

        Returns:
            Tuple of (direction, leverage). (0, 1.0) if no signal.
        """
        if not self._forecast_signals:
            return 0, 1.0

        date_str = self._get_session_date()
        if not date_str:
            return 0, 1.0

        return self._forecast_signals.get(date_str, (0, 1.0))

    def _get_session_date(self) -> str:
        """Get date string for the current bar."""
        try:
            if hasattr(self.df, 'index') and hasattr(self.df.index, 'date'):
                return str(self.df.index[self._current_idx].date())
            elif 'datetime' in self.df.columns:
                return str(pd.to_datetime(self.df.iloc[self._current_idx]['datetime']).date())
        except (IndexError, AttributeError, TypeError):
            pass
        return ""

    def _update_session_tracking(self) -> None:
        """Track session open price for execution alpha benchmark."""
        if not self._forecast_signals:
            return

        date_str = self._get_session_date()
        if date_str and date_str != self._current_session_date:
            # New session — record open price
            self._current_session_date = date_str
            if 'close' in self.df.columns:
                self._session_open_price = float(self.df.iloc[self._current_idx]['close'])
            elif hasattr(self.df, 'index'):
                # Use first bar's feature as proxy
                self._session_open_price = 0.0

    def _get_info(self) -> Dict[str, Any]:
        """Get current info dictionary"""
        info = {
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

        # Add curriculum phase info if available
        if hasattr(self.reward_strategy, 'curriculum_phase'):
            info["curriculum_phase"] = self.reward_strategy.curriculum_phase

        # Add reward breakdown if available
        if hasattr(self.reward_strategy, 'get_reward_breakdown'):
            breakdown = self.reward_strategy.get_reward_breakdown()
            if breakdown:
                info["reward_breakdown"] = breakdown

        return info

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
