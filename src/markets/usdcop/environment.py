"""
USDCOP Trading Environment - Advanced Gymnasium Environment for RL
==================================================================
Production-ready trading environment with realistic costs, risk management,
and reward shaping optimized for Forex trading.

Key Features:
- Realistic trading costs: variable spread, commission, slippage, swaps
- Advanced position management with dynamic SL/TP
- Risk-based position sizing and portfolio management
- Sophisticated reward shaping to prevent overfitting
- Multiple observation modes (windowed/flat)
- Comprehensive trade tracking and analytics
- Compatible with Stable-Baselines3 and Ray RLlib

Author: USDCOP Trading System
Version: 4.0.0
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import logging
from enum import IntEnum
import json
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


# =====================================================
# TYPES AND ENUMS
# =====================================================

class Actions(IntEnum):
    """Available actions in the environment"""
    HOLD = 0
    BUY = 1      # Open long or close short
    SELL = 2     # Open short or close long  
    CLOSE = 3    # Close any position


class PositionType(IntEnum):
    """Position types"""
    NONE = 0
    LONG = 1
    SHORT = -1


class ExecutionMode(IntEnum):
    """Price execution modes"""
    CLOSE = 0      # Execute at current close (less realistic)
    NEXT_OPEN = 1  # Execute at next bar open (realistic)
    NEXT_CLOSE = 2 # Execute at next bar close


# =====================================================
# CONFIGURATION DATACLASSES
# =====================================================

@dataclass
class TradingCosts:
    """Trading costs configuration"""
    # Basic costs
    point: float = 0.01                # Price point value
    default_spread_points: float = 2.0  # Default spread in points
    use_spread_column: bool = True      # Use spread from data if available
    commission_pct: float = 0.00005     # Commission percentage (0.5 pips per million)
    slippage_points: float = 1.0        # Average slippage in points
    
    # Financing costs
    swap_long_daily_pct: float = -0.00005   # Daily swap for long positions
    swap_short_daily_pct: float = -0.00003  # Daily swap for short positions
    
    # Advanced costs
    market_impact_factor: float = 0.0001    # Price impact for large orders
    rejection_probability: float = 0.01      # Order rejection probability
    
    def calculate_entry_cost(self, price: float, volume: float, 
                           is_buy: bool, spread: Optional[float] = None) -> Dict[str, float]:
        """Calculate detailed entry costs"""
        spread_points = spread if spread is not None else self.default_spread_points
        
        # Individual cost components
        spread_cost = spread_points * self.point * volume
        commission = price * volume * self.commission_pct
        slippage_cost = self.slippage_points * self.point * volume
        market_impact = self.market_impact_factor * volume * price * np.sqrt(volume)
        
        total = spread_cost + commission + slippage_cost + market_impact
        
        return {
            'spread': spread_cost,
            'commission': commission,
            'slippage': slippage_cost,
            'market_impact': market_impact,
            'total': total
        }


@dataclass
class RiskParameters:
    """Risk management parameters"""
    # Position limits
    max_position_size: float = 1.0      # Maximum position size in lots
    max_positions: int = 1              # Maximum simultaneous positions
    
    # Risk limits
    max_drawdown_pct: float = 0.20      # Maximum allowed drawdown
    risk_per_trade_pct: float = 0.02    # Risk per trade (2%)
    
    # Stop loss / Take profit
    stop_loss_pips: float = 50.0        # Default SL in pips
    take_profit_pips: float = 100.0     # Default TP in pips
    use_atr_stops: bool = True          # Use ATR-based stops
    atr_sl_multiplier: float = 2.0      # ATR multiplier for SL
    atr_tp_multiplier: float = 3.0      # ATR multiplier for TP
    trailing_stop_pips: float = 0.0     # Trailing stop distance (0 = disabled)
    
    # Trade frequency limits
    max_trades_per_day: int = 10        # Daily trade limit
    min_bars_between_trades: int = 5    # Minimum bars between trades
    max_holding_bars: int = 1440        # Maximum position holding time (5 days on M5)
    
    # Margin and leverage
    leverage: float = 100.0             # Account leverage
    margin_call_level: float = 0.5      # Margin call at 50% margin level
    stop_out_level: float = 0.2         # Stop out at 20% margin level


@dataclass
class RewardConfig:
    """Reward calculation configuration"""
    # Base reward
    reward_scaling: float = 1.0         # Global reward scale
    use_log_returns: bool = False       # Use log returns for reward
    
    # Trade outcome modifiers
    win_bonus_factor: float = 1.5       # Multiply reward for winning trades
    loss_penalty_factor: float = 2.0    # Multiply penalty for losing trades
    breakeven_reward: float = -0.001    # Small penalty for breakeven trades
    
    # Risk penalties
    drawdown_penalty_weight: float = 0.5     # Weight for drawdown penalty
    large_loss_penalty_threshold: float = 0.05  # Extra penalty for losses > 5%
    
    # Behavior shaping
    holding_penalty_per_bar: float = 0.0001  # Penalty for holding positions
    no_trade_penalty: float = 0.0002         # Penalty for not trading when flat
    overtrading_penalty: float = 0.01        # Penalty for excessive trading
    
    # Advanced shaping
    sharpe_bonus_weight: float = 0.1         # Bonus for high Sharpe ratio
    consistency_bonus_weight: float = 0.05   # Bonus for consistent profits
    
    # Clipping
    clip_reward: Optional[float] = 0.1       # Clip absolute reward value


@dataclass
class ObservationConfig:
    """Observation space configuration"""
    # Feature selection
    price_features: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume'
    ])
    
    technical_features: List[str] = field(default_factory=lambda: [
        'rsi_14', 'macd', 'macd_signal', 'bb_upper', 'bb_middle', 'bb_lower',
        'atr_14', 'adx_14', 'cci_14', 'roc_10', 'williams_r'
    ])
    
    custom_features: List[str] = field(default_factory=list)
    
    # Observation window
    lookback_window: int = 50           # Historical bars to include
    use_windowed_obs: bool = True       # Use 2D windowed observation
    
    # Normalization
    normalize_obs: bool = True          # Normalize observations
    normalization_method: str = 'zscore'  # 'zscore', 'minmax', 'robust'
    clip_obs: float = 10.0              # Clip normalized values
    
    # Additional context
    include_time_features: bool = True   # Include time-based features
    include_account_state: bool = True   # Include account metrics
    include_market_state: bool = True    # Include market microstructure


@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""
    # Account settings
    initial_balance: float = 10000.0
    account_currency: str = 'USD'
    
    # Symbol settings
    symbol: str = 'USDCOP'
    lot_size: float = 100000            # Standard lot size
    min_lot: float = 0.01               # Minimum lot size
    lot_step: float = 0.01              # Lot size increment
    
    # Execution settings
    execution_mode: ExecutionMode = ExecutionMode.NEXT_OPEN
    use_tick_data: bool = False         # Use tick-level execution
    
    # Component configs
    costs: TradingCosts = field(default_factory=TradingCosts)
    risk: RiskParameters = field(default_factory=RiskParameters)
    reward: RewardConfig = field(default_factory=RewardConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    
    # Episode settings
    max_episode_steps: Optional[int] = None
    random_start_steps: int = 100       # Random initial steps for exploration
    warmup_steps: int = 50              # Steps before trading allowed
    
    # Misc settings
    seed: Optional[int] = None
    render_mode: Optional[str] = None
    save_trades: bool = True            # Save trade history
    calculate_metrics: bool = True      # Calculate performance metrics


# =====================================================
# TRADE TRACKING
# =====================================================

@dataclass
class Trade:
    """Detailed trade information"""
    # Entry info
    entry_time: pd.Timestamp
    entry_price: float
    entry_spread: float
    
    # Exit info
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_spread: Optional[float] = None
    
    # Trade details
    position_type: PositionType = PositionType.NONE
    volume: float = 0.0
    
    # Costs
    entry_costs: Dict[str, float] = field(default_factory=dict)
    exit_costs: Dict[str, float] = field(default_factory=dict)
    swap_costs: float = 0.0
    total_costs: float = 0.0
    
    # Performance
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    pnl_pips: float = 0.0
    return_pct: float = 0.0
    
    # Risk metrics
    max_profit: float = 0.0
    max_loss: float = 0.0
    max_profit_pips: float = 0.0
    max_loss_pips: float = 0.0
    
    # Trade lifecycle
    duration_bars: int = 0
    exit_reason: Optional[str] = None
    
    # Metadata
    trade_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)


# =====================================================
# MAIN ENVIRONMENT CLASS
# =====================================================

class USDCOPTradingEnvironment(gym.Env):
    """
    Advanced trading environment for USDCOP with realistic market dynamics
    """
    
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array']}
    
    def __init__(self, 
                 data: pd.DataFrame,
                 config: Optional[EnvironmentConfig] = None,
                 training_mode: bool = True):
        """
        Initialize the trading environment
        
        Args:
            data: DataFrame with market data (OHLCV + features)
            config: Environment configuration
            training_mode: If True, use random episode starts
        """
        super().__init__()
        
        self.config = config or EnvironmentConfig()
        self.training_mode = training_mode
        
        # Set random seed
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            self._rng = np.random.RandomState(self.config.seed)
        else:
            self._rng = np.random.RandomState()
        
        # Prepare market data
        self._prepare_data(data)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Actions))
        self._define_observation_space()
        
        # Initialize components
        self._init_normalizer()
        self._init_metrics()
        
        # State variables
        self._episode_start_step = 0
        self._current_step = 0
        
        logger.info(f"Environment initialized with {len(self.data)} bars")
        logger.info(f"Features: {self._get_all_features()}")
    
    def _prepare_data(self, data: pd.DataFrame) -> None:
        """Prepare and validate market data"""
        # Validate required columns
        required = {'open', 'high', 'low', 'close'}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Ensure datetime index
        if 'time' in data.columns:
            data['time'] = pd.to_datetime(data['time'])
            data = data.set_index('time')
        
        # Sort by time
        data = data.sort_index()
        
        # Add volume if missing
        if 'volume' not in data.columns:
            data['volume'] = 1000  # Default volume
        
        # Add spread if missing and not using tick data
        if 'spread' not in data.columns and not self.config.costs.use_spread_column:
            data['spread'] = self.config.costs.default_spread_points
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log1p(data['returns'])
        
        # Add basic features if missing
        if 'atr_14' not in data.columns:
            high_low = data['high'] - data['low']
            high_close = abs(data['high'] - data['close'].shift())
            low_close = abs(data['low'] - data['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            data['atr_14'] = true_range.rolling(14).mean()
        
        # Add time features if requested
        if self.config.observation.include_time_features:
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['day_of_month'] = data.index.day
            data['is_london_session'] = ((data.index.hour >= 7) & (data.index.hour < 16)).astype(int)
            data['is_ny_session'] = ((data.index.hour >= 12) & (data.index.hour < 21)).astype(int)
            data['is_asian_session'] = ((data.index.hour >= 23) | (data.index.hour < 8)).astype(int)
        
        # Store prepared data
        self.data = data.fillna(method='ffill').fillna(0)
        
        # Pre-calculate execution prices to avoid look-ahead bias
        self._prepare_execution_prices()
    
    def _prepare_execution_prices(self) -> None:
        """Pre-calculate execution prices based on execution mode"""
        if self.config.execution_mode == ExecutionMode.CLOSE:
            self.data['exec_price'] = self.data['close']
            self.data['exec_high'] = self.data['high']
            self.data['exec_low'] = self.data['low']
        elif self.config.execution_mode == ExecutionMode.NEXT_OPEN:
            self.data['exec_price'] = self.data['open'].shift(-1)
            self.data['exec_high'] = self.data['high'].shift(-1)
            self.data['exec_low'] = self.data['low'].shift(-1)
        else:  # NEXT_CLOSE
            self.data['exec_price'] = self.data['close'].shift(-1)
            self.data['exec_high'] = self.data['high'].shift(-1)
            self.data['exec_low'] = self.data['low'].shift(-1)
        
        # Remove last row(s) with NaN execution prices
        self.data = self.data[:-1]
    
    def _get_all_features(self) -> List[str]:
        """Get all feature columns to use"""
        features = []
        
        # Price features
        for feat in self.config.observation.price_features:
            if feat in self.data.columns:
                features.append(feat)
        
        # Technical features
        for feat in self.config.observation.technical_features:
            if feat in self.data.columns:
                features.append(feat)
        
        # Custom features
        for feat in self.config.observation.custom_features:
            if feat in self.data.columns:
                features.append(feat)
        
        # Time features
        if self.config.observation.include_time_features:
            time_features = ['hour', 'day_of_week', 'is_london_session', 
                           'is_ny_session', 'is_asian_session']
            features.extend([f for f in time_features if f in self.data.columns])
        
        return features
    
    def _define_observation_space(self) -> None:
        """Define the observation space"""
        features = self._get_all_features()
        n_features = len(features)
        
        # Base market features
        if self.config.observation.use_windowed_obs:
            # 2D observation: (lookback_window, n_features)
            market_shape = (self.config.observation.lookback_window, n_features)
        else:
            # 1D observation: flattened
            market_shape = (self.config.observation.lookback_window * n_features,)
        
        # Additional state features
        n_position_info = 8  # position, pnl, entry_price, bars_held, etc.
        n_account_info = 6   # balance, equity, margin, drawdown, etc.
        n_market_info = 5    # spread, volatility, session, etc.
        
        # Total observation size
        if self.config.observation.use_windowed_obs:
            # Combine 2D market data with 1D state data
            total_shape = (market_shape[0], market_shape[1] + n_position_info + n_account_info + n_market_info)
        else:
            total_shape = (market_shape[0] + n_position_info + n_account_info + n_market_info,)
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=total_shape,
            dtype=np.float32
        )
        
        # Store feature info
        self._features = features
        self._n_features = n_features
    
    def _init_normalizer(self) -> None:
        """Initialize feature normalizer"""
        if not self.config.observation.normalize_obs:
            return
        
        features_data = self.data[self._features]
        
        if self.config.observation.normalization_method == 'zscore':
            self._feature_mean = features_data.mean()
            self._feature_std = features_data.std() + 1e-8
        elif self.config.observation.normalization_method == 'minmax':
            self._feature_min = features_data.min()
            self._feature_max = features_data.max()
            self._feature_range = self._feature_max - self._feature_min + 1e-8
        elif self.config.observation.normalization_method == 'robust':
            self._feature_median = features_data.median()
            self._feature_mad = features_data.mad() + 1e-8
    
    def _init_metrics(self) -> None:
        """Initialize performance tracking"""
        self.metrics = {
            'episode_returns': [],
            'episode_lengths': [],
            'trade_returns': [],
            'win_rates': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'trade_counts': []
        }
    
    def reset(self, *, seed: Optional[int] = None, 
              options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to initial state
        
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset account state
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance
        self.equity = self.balance
        self.margin_used = 0.0
        self.margin_free = self.balance
        self.margin_level = np.inf
        
        # Reset position state
        self.position = PositionType.NONE
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.position_pnl = 0.0
        self.position_pnl_pips = 0.0
        self.position_max_profit = 0.0
        self.position_max_loss = 0.0
        self.bars_held = 0
        
        # Reset risk metrics
        self.peak_equity = self.balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0
        self.bars_since_last_trade = 0
        
        # Reset trade tracking
        self.current_trade = None
        self.trade_history = []
        self.equity_curve = [self.balance]
        self.balance_curve = [self.balance]
        self.returns_curve = []
        
        # Reset counters
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_spread_cost = 0.0
        self.total_swap = 0.0
        
        # Select episode start
        if self.training_mode:
            # Random start for training
            min_start = self.config.observation.lookback_window + self.config.warmup_steps
            max_start = len(self.data) - self.config.random_start_steps - 1000
            self._episode_start_step = self._rng.randint(min_start, max(min_start + 1, max_start))
        else:
            # Sequential start for testing
            self._episode_start_step = self.config.observation.lookback_window
        
        self._current_step = self._episode_start_step
        self._max_step = len(self.data) - 1
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation: New observation
            reward: Reward from action
            terminated: Whether episode ended
            truncated: Whether episode was cut short
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        
        action = Actions(action)
        
        # Get current market data
        current_bar = self.data.iloc[self._current_step]
        current_time = self.data.index[self._current_step]
        current_price = current_bar['close']
        exec_price = current_bar['exec_price']
        
        # Initialize step info
        step_pnl = 0.0
        step_costs = {}
        trade_opened = False
        trade_closed = False
        action_valid = True
        
        # Update counters
        self.bars_since_last_trade += 1
        if self.position != PositionType.NONE:
            self.bars_held += 1
        
        # Reset daily counter (simplified - every 288 bars for M5)
        if self._current_step % 288 == 0:
            self.trades_today = 0
        
        # Execute action
        if action == Actions.BUY:
            if self.position == PositionType.NONE:
                # Open long position
                success, costs = self._open_position(PositionType.LONG, exec_price, current_bar)
                if success:
                    trade_opened = True
                    step_costs = costs
                else:
                    action_valid = False
            elif self.position == PositionType.SHORT:
                # Close short and open long
                close_pnl, close_costs = self._close_position(exec_price, current_bar, 'reverse_to_long')
                step_pnl += close_pnl
                step_costs.update(close_costs)
                trade_closed = True
                
                success, open_costs = self._open_position(PositionType.LONG, exec_price, current_bar)
                if success:
                    trade_opened = True
                    step_costs.update(open_costs)
        
        elif action == Actions.SELL:
            if self.position == PositionType.NONE:
                # Open short position
                success, costs = self._open_position(PositionType.SHORT, exec_price, current_bar)
                if success:
                    trade_opened = True
                    step_costs = costs
                else:
                    action_valid = False
            elif self.position == PositionType.LONG:
                # Close long and open short
                close_pnl, close_costs = self._close_position(exec_price, current_bar, 'reverse_to_short')
                step_pnl += close_pnl
                step_costs.update(close_costs)
                trade_closed = True
                
                success, open_costs = self._open_position(PositionType.SHORT, exec_price, current_bar)
                if success:
                    trade_opened = True
                    step_costs.update(open_costs)
        
        elif action == Actions.CLOSE:
            if self.position != PositionType.NONE:
                # Close current position
                close_pnl, close_costs = self._close_position(exec_price, current_bar, 'manual_close')
                step_pnl += close_pnl
                step_costs = close_costs
                trade_closed = True
            else:
                action_valid = False
        
        # Update position P&L if still open
        if self.position != PositionType.NONE:
            self._update_position_pnl(current_price)
            
            # Check stop loss / take profit
            sl_hit, tp_hit = self._check_stops(current_bar)
            
            if sl_hit or tp_hit:
                reason = 'stop_loss' if sl_hit else 'take_profit'
                close_pnl, close_costs = self._close_position(exec_price, current_bar, reason)
                step_pnl += close_pnl
                step_costs.update(close_costs)
                trade_closed = True
            
            # Check max holding period
            elif self.bars_held >= self.config.risk.max_holding_bars:
                close_pnl, close_costs = self._close_position(exec_price, current_bar, 'max_holding_time')
                step_pnl += close_pnl
                step_costs.update(close_costs)
                trade_closed = True
            
            # Apply holding costs
            if self.position != PositionType.NONE:
                swap = self._calculate_swap()
                self.total_swap += swap
                step_costs['swap'] = swap
        
        # Update account state
        self._update_account_state()
        
        # Calculate reward
        reward = self._calculate_reward(
            step_pnl=step_pnl,
            step_costs=step_costs,
            action=action,
            action_valid=action_valid,
            trade_opened=trade_opened,
            trade_closed=trade_closed
        )
        
        # Store metrics
        self.equity_curve.append(self.equity)
        self.balance_curve.append(self.balance)
        if len(self.equity_curve) > 1:
            ret = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.returns_curve.append(ret)
        
        # Advance time
        self._current_step += 1
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = False
        
        if self.config.max_episode_steps is not None:
            if self._current_step - self._episode_start_step >= self.config.max_episode_steps:
                truncated = True
        
        if self._current_step >= self._max_step:
            truncated = True
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        # Add step-specific info
        info.update({
            'step_pnl': step_pnl,
            'step_costs': step_costs,
            'action': action.name,
            'action_valid': action_valid,
            'trade_opened': trade_opened,
            'trade_closed': trade_closed
        })
        
        return obs, reward, terminated, truncated, info
    
    def _open_position(self, position_type: PositionType, 
                      price: float, bar: pd.Series) -> Tuple[bool, Dict[str, float]]:
        """Open a new position"""
        # Check if we can open a position
        if not self._can_open_position():
            return False, {}
        
        # Calculate position size
        position_size = self._calculate_position_size(bar)
        if position_size < self.config.min_lot:
            return False, {}
        
        # Get spread
        spread = bar.get('spread', self.config.costs.default_spread_points)
        
        # Calculate entry costs
        is_buy = position_type == PositionType.LONG
        costs = self.config.costs.calculate_entry_cost(price, position_size, is_buy, spread)
        
        # Check if we have enough margin
        required_margin = (price * position_size) / self.config.risk.leverage
        if required_margin > self.margin_free:
            return False, {}
        
        # Apply costs to get actual entry price
        if is_buy:
            entry_price = price + (spread * self.config.costs.point / 2) + \
                         (self.config.costs.slippage_points * self.config.costs.point)
        else:
            entry_price = price - (spread * self.config.costs.point / 2) - \
                         (self.config.costs.slippage_points * self.config.costs.point)
        
        # Check for order rejection
        if self._rng.random() < self.config.costs.rejection_probability:
            return False, {}
        
        # Open position
        self.position = position_type
        self.position_size = position_size
        self.entry_price = entry_price
        self.entry_time = self.data.index[self._current_step]
        self.position_pnl = -costs['total']
        self.position_pnl_pips = 0.0
        self.position_max_profit = 0.0
        self.position_max_loss = -costs['total']
        self.bars_held = 0
        
        # Update account
        self.balance -= costs['total']
        self.margin_used = required_margin
        self.margin_free = self.balance - self.margin_used
        
        # Update counters
        self.trades_today += 1
        self.bars_since_last_trade = 0
        self.total_commission += costs['commission']
        self.total_spread_cost += costs['spread']
        self.total_slippage += costs['slippage']
        
        # Create trade record
        self.current_trade = Trade(
            entry_time=self.entry_time,
            entry_price=entry_price,
            entry_spread=spread,
            position_type=position_type,
            volume=position_size,
            entry_costs=costs,
            trade_id=f"{self.symbol}_{self.entry_time.strftime('%Y%m%d_%H%M%S')}"
        )
        
        logger.debug(f"Opened {position_type.name} position: size={position_size:.2f}, "
                    f"price={entry_price:.4f}, costs={costs['total']:.2f}")
        
        return True, costs
    
    def _close_position(self, price: float, bar: pd.Series, 
                       reason: str) -> Tuple[float, Dict[str, float]]:
        """Close current position"""
        if self.position == PositionType.NONE or self.current_trade is None:
            return 0.0, {}
        
        # Get spread
        spread = bar.get('spread', self.config.costs.default_spread_points)
        
        # Calculate exit price with spread and slippage
        if self.position == PositionType.LONG:
            # Selling at bid
            exit_price = price - (spread * self.config.costs.point / 2) - \
                        (self.config.costs.slippage_points * self.config.costs.point)
        else:
            # Buying at ask
            exit_price = price + (spread * self.config.costs.point / 2) + \
                        (self.config.costs.slippage_points * self.config.costs.point)
        
        # Calculate raw P&L
        if self.position == PositionType.LONG:
            price_diff = exit_price - self.entry_price
        else:
            price_diff = self.entry_price - exit_price
        
        gross_pnl = price_diff * self.position_size
        pnl_pips = price_diff / self.config.costs.point
        
        # Calculate exit costs
        is_buy = self.position == PositionType.SHORT
        costs = self.config.costs.calculate_entry_cost(price, self.position_size, is_buy, spread)
        
        # Calculate total costs including swap
        total_swap = self._calculate_swap() * self.bars_held
        total_costs = costs['total'] + abs(total_swap)
        
        # Net P&L
        net_pnl = gross_pnl - total_costs
        
        # Update trade record
        self.current_trade.exit_time = self.data.index[self._current_step]
        self.current_trade.exit_price = exit_price
        self.current_trade.exit_spread = spread
        self.current_trade.exit_costs = costs
        self.current_trade.swap_costs = total_swap
        self.current_trade.total_costs = self.current_trade.entry_costs['total'] + total_costs
        self.current_trade.gross_pnl = gross_pnl
        self.current_trade.net_pnl = net_pnl
        self.current_trade.pnl_pips = pnl_pips
        self.current_trade.return_pct = net_pnl / (self.entry_price * self.position_size) * 100
        self.current_trade.duration_bars = self.bars_held
        self.current_trade.exit_reason = reason
        self.current_trade.max_profit = self.position_max_profit
        self.current_trade.max_loss = self.position_max_loss
        
        # Add to history
        self.trade_history.append(self.current_trade)
        
        # Update account
        self.balance += net_pnl
        self.margin_used = 0.0
        self.margin_free = self.balance
        
        # Update statistics
        self.total_trades += 1
        self.total_pnl += net_pnl
        
        if net_pnl > 0:
            self.winning_trades += 1
            self.consecutive_losses = 0
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
        
        # Reset position
        self.position = PositionType.NONE
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_time = None
        self.position_pnl = 0.0
        self.position_pnl_pips = 0.0
        self.position_max_profit = 0.0
        self.position_max_loss = 0.0
        self.bars_held = 0
        self.current_trade = None
        
        logger.debug(f"Closed position: pnl={net_pnl:.2f}, pips={pnl_pips:.1f}, "
                    f"reason={reason}")
        
        return net_pnl, costs
    
    def _update_position_pnl(self, current_price: float) -> None:
        """Update unrealized P&L for open position"""
        if self.position == PositionType.NONE:
            return
        
        # Calculate current P&L
        if self.position == PositionType.LONG:
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price
        
        # Gross P&L (before costs)
        self.position_pnl = price_diff * self.position_size
        self.position_pnl_pips = price_diff / self.config.costs.point
        
        # Track max profit/loss
        self.position_max_profit = max(self.position_max_profit, self.position_pnl)
        self.position_max_loss = min(self.position_max_loss, self.position_pnl)
        
        # Update current trade tracking
        if self.current_trade:
            self.current_trade.max_profit = self.position_max_profit
            self.current_trade.max_loss = self.position_max_loss
            self.current_trade.max_profit_pips = self.position_max_profit / (self.config.costs.point * self.position_size)
            self.current_trade.max_loss_pips = self.position_max_loss / (self.config.costs.point * self.position_size)
    
    def _check_stops(self, bar: pd.Series) -> Tuple[bool, bool]:
        """Check if stop loss or take profit is hit"""
        if self.position == PositionType.NONE:
            return False, False
        
        # Get execution high/low for stop checking
        exec_high = bar['exec_high']
        exec_low = bar['exec_low']
        
        # Calculate stop levels
        if self.config.risk.use_atr_stops and 'atr_14' in bar:
            atr = bar['atr_14']
            sl_distance = atr * self.config.risk.atr_sl_multiplier
            tp_distance = atr * self.config.risk.atr_tp_multiplier
        else:
            sl_distance = self.config.risk.stop_loss_pips * self.config.costs.point
            tp_distance = self.config.risk.take_profit_pips * self.config.costs.point
        
        # Check stops based on position type
        if self.position == PositionType.LONG:
            sl_price = self.entry_price - sl_distance
            tp_price = self.entry_price + tp_distance
            
            # Trailing stop
            if self.config.risk.trailing_stop_pips > 0 and self.position_max_profit > 0:
                trailing_sl = self.entry_price + self.position_max_profit / self.position_size - \
                             (self.config.risk.trailing_stop_pips * self.config.costs.point)
                sl_price = max(sl_price, trailing_sl)
            
            sl_hit = exec_low <= sl_price
            tp_hit = exec_high >= tp_price
            
        else:  # SHORT
            sl_price = self.entry_price + sl_distance
            tp_price = self.entry_price - tp_distance
            
            # Trailing stop
            if self.config.risk.trailing_stop_pips > 0 and self.position_max_profit > 0:
                trailing_sl = self.entry_price - self.position_max_profit / self.position_size + \
                             (self.config.risk.trailing_stop_pips * self.config.costs.point)
                sl_price = min(sl_price, trailing_sl)
            
            sl_hit = exec_high >= sl_price
            tp_hit = exec_low <= tp_price
        
        return sl_hit, tp_hit
    
    def _calculate_position_size(self, bar: pd.Series) -> float:
        """Calculate position size based on risk management"""
        # Risk-based position sizing
        risk_amount = self.equity * self.config.risk.risk_per_trade_pct
        
        # Get stop distance
        if self.config.risk.use_atr_stops and 'atr_14' in bar:
            stop_distance = bar['atr_14'] * self.config.risk.atr_sl_multiplier
        else:
            stop_distance = self.config.risk.stop_loss_pips * self.config.costs.point
        
        # Calculate size
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
        else:
            position_size = self.config.risk.max_position_size
        
        # Apply limits
        position_size = min(position_size, self.config.risk.max_position_size)
        position_size = max(position_size, self.config.min_lot)
        
        # Round to lot step
        position_size = round(position_size / self.config.lot_step) * self.config.lot_step
        
        return position_size
    
    def _calculate_swap(self) -> float:
        """Calculate swap cost for one period"""
        if self.position == PositionType.NONE:
            return 0.0
        
        if self.position == PositionType.LONG:
            swap_rate = self.config.costs.swap_long_daily_pct
        else:
            swap_rate = self.config.costs.swap_short_daily_pct
        
        # Convert to period rate (assuming M5 bars, 288 per day)
        period_rate = swap_rate / 288
        
        return abs(self.position_size * self.entry_price * period_rate)
    
    def _can_open_position(self) -> bool:
        """Check if we can open a new position"""
        # Check warmup period
        if self._current_step - self._episode_start_step < self.config.warmup_steps:
            return False
        
        # Check daily limit
        if self.trades_today >= self.config.risk.max_trades_per_day:
            return False
        
        # Check minimum bars between trades
        if self.bars_since_last_trade < self.config.risk.min_bars_between_trades:
            return False
        
        # Check margin requirements
        if self.margin_level < 200:  # 200% margin level minimum
            return False
        
        # Check drawdown limit
        if self.current_drawdown > self.config.risk.max_drawdown_pct * 0.8:  # 80% of max
            return False
        
        return True
    
    def _update_account_state(self) -> None:
        """Update account metrics"""
        # Update equity
        self.equity = self.balance + self.position_pnl
        
        # Update peak and drawdown
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - self.equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
        
        # Update margin level
        if self.margin_used > 0:
            self.margin_level = (self.equity / self.margin_used) * 100
        else:
            self.margin_level = np.inf
    
    def _calculate_reward(self, step_pnl: float, step_costs: Dict[str, float],
                         action: Actions, action_valid: bool,
                         trade_opened: bool, trade_closed: bool) -> float:
        """Calculate step reward with sophisticated shaping"""
        # Base reward is the P&L
        if self.config.reward.use_log_returns and step_pnl != 0:
            reward = np.sign(step_pnl) * np.log1p(abs(step_pnl) / self.config.initial_balance)
        else:
            reward = step_pnl / self.config.initial_balance
        
        # Scale base reward
        reward *= self.config.reward.reward_scaling
        
        # Invalid action penalty
        if not action_valid:
            reward -= 0.01
        
        # Trade outcome modifiers
        if trade_closed and self.trade_history:
            last_trade = self.trade_history[-1]
            
            if last_trade.net_pnl > 0:
                # Win bonus
                reward *= self.config.reward.win_bonus_factor
            elif last_trade.net_pnl < 0:
                # Loss penalty
                reward *= self.config.reward.loss_penalty_factor
                
                # Extra penalty for large losses
                if abs(last_trade.return_pct) > self.config.reward.large_loss_penalty_threshold * 100:
                    reward -= 0.05
            else:
                # Breakeven penalty
                reward += self.config.reward.breakeven_reward
        
        # Holding penalty
        if self.position != PositionType.NONE:
            reward -= self.config.reward.holding_penalty_per_bar * abs(self.position_size)
        elif action == Actions.HOLD:
            # No trade penalty when flat
            reward -= self.config.reward.no_trade_penalty
        
        # Overtrading penalty
        if self.trades_today > self.config.risk.max_trades_per_day * 0.8:
            reward -= self.config.reward.overtrading_penalty
        
        # Risk penalties
        if self.current_drawdown > 0:
            reward -= self.config.reward.drawdown_penalty_weight * self.current_drawdown
        
        # Performance bonuses (only if we have enough history)
        if len(self.returns_curve) > 20:
            # Sharpe ratio bonus
            returns = np.array(self.returns_curve[-20:])
            if returns.std() > 0:
                sharpe = np.sqrt(252 * 288) * returns.mean() / returns.std()  # Annualized
                reward += self.config.reward.sharpe_bonus_weight * np.clip(sharpe, -1, 1) / 10
            
            # Consistency bonus
            if len([r for r in returns if r > 0]) > len(returns) * 0.6:  # 60%+ positive
                reward += self.config.reward.consistency_bonus_weight
        
        # Clip reward
        if self.config.reward.clip_reward is not None:
            reward = np.clip(reward, -self.config.reward.clip_reward, self.config.reward.clip_reward)
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Account blown
        if self.balance <= 0 or self.equity <= 0:
            return True
        
        # Margin call
        if self.margin_level < self.config.risk.margin_call_level * 100:
            return True
        
        # Stop out
        if self.margin_level < self.config.risk.stop_out_level * 100:
            return True
        
        # Max drawdown exceeded
        if self.current_drawdown > self.config.risk.max_drawdown_pct:
            return True
        
        # Severe consecutive losses
        if self.consecutive_losses > 10:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        # Get market features window
        start_idx = max(0, self._current_step - self.config.observation.lookback_window + 1)
        end_idx = self._current_step + 1
        
        window_data = self.data.iloc[start_idx:end_idx][self._features].values
        
        # Pad if necessary
        if len(window_data) < self.config.observation.lookback_window:
            pad_size = self.config.observation.lookback_window - len(window_data)
            padding = np.repeat(window_data[0:1], pad_size, axis=0)
            window_data = np.vstack([padding, window_data])
        
        # Normalize features
        if self.config.observation.normalize_obs:
            window_data = self._normalize_features(window_data)
        
        # Position information
        position_info = np.array([
            float(self.position),
            self.position_size / self.config.risk.max_position_size,
            self.position_pnl / self.config.initial_balance if self.position != PositionType.NONE else 0,
            self.position_pnl_pips / 100 if self.position != PositionType.NONE else 0,
            self.bars_held / 100 if self.position != PositionType.NONE else 0,
            self.position_max_profit / self.config.initial_balance if self.position != PositionType.NONE else 0,
            self.position_max_loss / self.config.initial_balance if self.position != PositionType.NONE else 0,
            float(self.consecutive_losses) / 10
        ])
        
        # Account information
        account_info = np.array([
            self.balance / self.config.initial_balance,
            self.equity / self.config.initial_balance,
            self.margin_free / self.config.initial_balance,
            np.clip(self.margin_level / 1000, 0, 10) if self.margin_level != np.inf else 10,
            self.current_drawdown,
            self.total_pnl / self.config.initial_balance
        ])
        
        # Market state information
        current_bar = self.data.iloc[self._current_step]
        market_info = np.array([
            current_bar.get('spread', self.config.costs.default_spread_points) / 10,
            current_bar.get('atr_14', 0) / current_bar['close'] if 'atr_14' in current_bar else 0.001,
            self.trades_today / self.config.risk.max_trades_per_day,
            self.bars_since_last_trade / 100,
            self.total_trades / 100
        ])
        
        # Combine based on observation mode
        if self.config.observation.use_windowed_obs:
            # Stack additional info as extra features
            n_additional = len(position_info) + len(account_info) + len(market_info)
            additional_info = np.concatenate([position_info, account_info, market_info])
            additional_window = np.repeat(additional_info.reshape(1, -1), 
                                        self.config.observation.lookback_window, axis=0)
            
            observation = np.hstack([window_data, additional_window])
        else:
            # Flatten everything
            observation = np.concatenate([
                window_data.flatten(),
                position_info,
                account_info,
                market_info
            ])
        
        return observation.astype(np.float32)
    
    def _normalize_features(self, data: np.ndarray) -> np.ndarray:
        """Normalize feature data"""
        if self.config.observation.normalization_method == 'zscore':
            normalized = (data - self._feature_mean.values) / self._feature_std.values
        elif self.config.observation.normalization_method == 'minmax':
            normalized = (data - self._feature_min.values) / self._feature_range.values
        elif self.config.observation.normalization_method == 'robust':
            normalized = (data - self._feature_median.values) / self._feature_mad.values
        else:
            normalized = data
        
        # Clip values
        normalized = np.clip(normalized, -self.config.observation.clip_obs, 
                           self.config.observation.clip_obs)
        
        return normalized
    
    def _get_info(self) -> Dict[str, Any]:
        """Get environment info dictionary"""
        info = {
            # Account state
            'balance': float(self.balance),
            'equity': float(self.equity),
            'margin_free': float(self.margin_free),
            'margin_used': float(self.margin_used),
            'margin_level': float(self.margin_level) if self.margin_level != np.inf else 999999,
            
            # Position state
            'position': self.position.name,
            'position_size': float(self.position_size),
            'entry_price': float(self.entry_price) if self.entry_price else None,
            'position_pnl': float(self.position_pnl),
            'position_pnl_pips': float(self.position_pnl_pips),
            'bars_held': int(self.bars_held),
            
            # Risk metrics
            'current_drawdown': float(self.current_drawdown),
            'max_drawdown': float(self.max_drawdown),
            'consecutive_losses': int(self.consecutive_losses),
            
            # Trading statistics
            'total_trades': int(self.total_trades),
            'winning_trades': int(self.winning_trades),
            'losing_trades': int(self.losing_trades),
            'win_rate': float(self.winning_trades / self.total_trades) if self.total_trades > 0 else 0.0,
            'total_pnl': float(self.total_pnl),
            
            # Costs breakdown
            'total_commission': float(self.total_commission),
            'total_spread_cost': float(self.total_spread_cost),
            'total_slippage': float(self.total_slippage),
            'total_swap': float(self.total_swap),
            
            # Episode info
            'current_step': int(self._current_step),
            'episode_length': int(self._current_step - self._episode_start_step),
            
            # Market info
            'current_price': float(self.data.iloc[self._current_step]['close']),
            'current_time': str(self.data.index[self._current_step])
        }
        
        # Add performance metrics if available
        if self.trade_history:
            metrics = self.calculate_performance_metrics()
            info['performance_metrics'] = metrics
        
        return info
    
    def render(self, mode='human') -> Optional[np.ndarray]:
        """Render environment state"""
        if mode == 'human':
            self._render_human()
        elif mode == 'ansi':
            return self._render_ansi()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
    
    def _render_human(self) -> None:
        """Human-readable rendering to console"""
        current_bar = self.data.iloc[self._current_step]
        
        print(f"\n{'='*60}")
        print(f"Step: {self._current_step} | Time: {self.data.index[self._current_step]}")
        print(f"Price: {current_bar['close']:.4f} | Spread: {current_bar.get('spread', 'N/A')}")
        print(f"{'-'*60}")
        print(f"Position: {self.position.name} | Size: {self.position_size:.2f}")
        if self.position != PositionType.NONE:
            print(f"Entry: {self.entry_price:.4f} | P&L: ${self.position_pnl:.2f} ({self.position_pnl_pips:.1f} pips)")
            print(f"Max Profit: ${self.position_max_profit:.2f} | Max Loss: ${self.position_max_loss:.2f}")
        print(f"{'-'*60}")
        print(f"Balance: ${self.balance:.2f} | Equity: ${self.equity:.2f}")
        print(f"Margin: ${self.margin_used:.2f} | Free: ${self.margin_free:.2f} | Level: {self.margin_level:.1f}%")
        print(f"Drawdown: {self.current_drawdown:.2%} | Max DD: {self.max_drawdown:.2%}")
        print(f"{'-'*60}")
        print(f"Trades: {self.total_trades} | Win Rate: {self.winning_trades/max(1,self.total_trades):.1%}")
        print(f"Total P&L: ${self.total_pnl:.2f}")
        print(f"{'='*60}")
    
    def _render_ansi(self) -> str:
        """ANSI rendering for text-based environments"""
        # Similar to human but returns string
        lines = []
        current_bar = self.data.iloc[self._current_step]
        
        lines.append(f"Step: {self._current_step} | Price: {current_bar['close']:.4f}")
        lines.append(f"Position: {self.position.name} | P&L: ${self.position_pnl:.2f}")
        lines.append(f"Equity: ${self.equity:.2f} | DD: {self.current_drawdown:.2%}")
        lines.append(f"Trades: {self.total_trades} | Win Rate: {self.winning_trades/max(1,self.total_trades):.1%}")
        
        return '\n'.join(lines)
    
    def _render_rgb_array(self) -> np.ndarray:
        """RGB array rendering for video recording"""
        # Create a simple visualization
        # This is a placeholder - implement proper visualization if needed
        img = np.zeros((400, 600, 3), dtype=np.uint8)
        # Add visualization code here
        return img
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {}
        
        # Extract trade returns
        returns = [t.return_pct / 100 for t in self.trade_history]
        pnls = [t.net_pnl for t in self.trade_history]
        
        # Win/loss analysis
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Calculate metrics
        metrics = {
            'total_return': (self.equity - self.config.initial_balance) / self.config.initial_balance,
            'win_rate': len(wins) / len(self.trade_history) if self.trade_history else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else float('inf'),
            'expectancy': np.mean(pnls) if pnls else 0,
            'avg_trade_duration': np.mean([t.duration_bars for t in self.trade_history]),
            'max_consecutive_wins': self._max_consecutive(pnls, True),
            'max_consecutive_losses': self._max_consecutive(pnls, False),
        }
        
        # Sharpe ratio
        if len(self.returns_curve) > 1:
            returns_array = np.array(self.returns_curve)
            if returns_array.std() > 0:
                # Annualized Sharpe (288 M5 bars per day, 252 trading days)
                metrics['sharpe_ratio'] = np.sqrt(252 * 288) * returns_array.mean() / returns_array.std()
            else:
                metrics['sharpe_ratio'] = 0.0
        
        # Sortino ratio
        if len(self.returns_curve) > 1:
            negative_returns = [r for r in self.returns_curve if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    metrics['sortino_ratio'] = np.sqrt(252 * 288) * np.mean(self.returns_curve) / downside_std
                else:
                    metrics['sortino_ratio'] = 0.0
        
        # Calmar ratio
        if self.max_drawdown > 0:
            annual_return = metrics['total_return'] * (252 * 288 / len(self.equity_curve))
            metrics['calmar_ratio'] = annual_return / self.max_drawdown
        
        return metrics
    
    def _max_consecutive(self, values: List[float], positive: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_count = 0
        current_count = 0
        
        for v in values:
            if (positive and v > 0) or (not positive and v < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history as list of dictionaries"""
        return [asdict(trade) for trade in self.trade_history]
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame"""
        if not self.equity_curve:
            return pd.DataFrame()
        
        times = self.data.index[self._episode_start_step:self._episode_start_step + len(self.equity_curve)]
        
        return pd.DataFrame({
            'time': times,
            'equity': self.equity_curve,
            'balance': self.balance_curve,
            'returns': [0] + self.returns_curve  # Prepend 0 for first period
        })
    
    def save_episode_data(self, filepath: str) -> None:
        """Save episode data for analysis"""
        episode_data = {
            'config': asdict(self.config),
            'metrics': self.calculate_performance_metrics(),
            'trades': self.get_trade_history(),
            'equity_curve': self.get_equity_curve().to_dict('records'),
            'final_state': self._get_info()
        }
        
        with open(filepath, 'w') as f:
            json.dump(episode_data, f, indent=2, default=str)
    
    @property
    def symbol(self) -> str:
        """Get trading symbol"""
        return self.config.symbol


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def create_env_fn(data: pd.DataFrame, 
                  config: Optional[EnvironmentConfig] = None,
                  training_mode: bool = True) -> Callable:
    """Factory function for creating environments (useful for vectorization)"""
    def _init():
        return USDCOPTradingEnvironment(data.copy(), config, training_mode)
    return _init


def make_env(data: pd.DataFrame,
             **kwargs) -> USDCOPTradingEnvironment:
    """Convenience function to create environment with custom config"""
    # Create config from kwargs
    config_dict = {}
    
    # Parse nested config options
    for key, value in kwargs.items():
        if '__' in key:
            # Handle nested configs like 'costs__spread_points'
            parts = key.split('__')
            if len(parts) == 2:
                section, param = parts
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][param] = value
        else:
            config_dict[key] = value
    
    # Create config objects
    costs = TradingCosts(**config_dict.get('costs', {}))
    risk = RiskParameters(**config_dict.get('risk', {}))
    reward = RewardConfig(**config_dict.get('reward', {}))
    observation = ObservationConfig(**config_dict.get('observation', {}))
    
    # Create main config
    config = EnvironmentConfig(
        costs=costs,
        risk=risk,
        reward=reward,
        observation=observation,
        **{k: v for k, v in config_dict.items() 
           if k not in ['costs', 'risk', 'reward', 'observation']}
    )
    
    return USDCOPTradingEnvironment(data, config)


# =====================================================
# TESTING
# =====================================================

def test_environment():
    """Comprehensive environment test"""
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    print("Creating test data...")
    dates = pd.date_range('2024-01-01', '2024-02-01', freq='5min')
    
    # Realistic price simulation
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.001, len(dates))
    price = 4000 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': price * (1 + np.random.normal(0, 0.0001, len(dates))),
        'high': price * (1 + np.abs(np.random.normal(0, 0.0005, len(dates)))),
        'low': price * (1 - np.abs(np.random.normal(0, 0.0005, len(dates)))),
        'close': price,
        'volume': np.random.randint(100, 1000, len(dates)),
        'spread': np.random.uniform(1, 5, len(dates)),
        
        # Technical indicators
        'rsi_14': 50 + 20 * np.sin(np.linspace(0, 10, len(dates))) + np.random.normal(0, 5, len(dates)),
        'macd': np.random.normal(0, 0.01, len(dates)),
        'macd_signal': np.random.normal(0, 0.01, len(dates)),
        'atr_14': np.abs(np.random.normal(20, 5, len(dates))),
        'bb_upper': price * (1 + 0.02),
        'bb_middle': price,
        'bb_lower': price * (1 - 0.02)
    }, index=dates)
    
    # Create environment with custom config
    print("Creating environment...")
    config = EnvironmentConfig(
        initial_balance=10000,
        symbol='USDCOP',
        execution_mode=ExecutionMode.NEXT_OPEN,
        costs=TradingCosts(
            point=0.0001,
            default_spread_points=3.0,
            commission_pct=0.00005,
            slippage_points=1.0
        ),
        risk=RiskParameters(
            max_position_size=1.0,
            stop_loss_pips=30,
            take_profit_pips=60,
            use_atr_stops=True,
            max_drawdown_pct=0.15
        ),
        reward=RewardConfig(
            reward_scaling=1.0,
            win_bonus_factor=1.2,
            loss_penalty_factor=1.5,
            holding_penalty_per_bar=0.0001
        ),
        observation=ObservationConfig(
            lookback_window=30,
            use_windowed_obs=False,
            normalize_obs=True
        )
    )
    
    env = USDCOPTradingEnvironment(data, config, training_mode=False)
    
    # Test reset
    print("\nTesting reset...")
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Test random episode
    print("\nRunning test episode...")
    done = False
    step = 0
    total_reward = 0
    
    while not done and step < 500:
        # Semi-random action (bias towards trading)
        if step < 10:
            action = 0  # Hold initially
        else:
            action = env.action_space.sample()
            if env.position == PositionType.NONE and np.random.random() < 0.3:
                action = np.random.choice([1, 2])  # Buy or Sell
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        # Render every 50 steps
        if step % 50 == 0:
            print(f"\nStep {step}:")
            env.render()
        
        step += 1
    
    # Final statistics
    print("\n" + "="*60)
    print("EPISODE COMPLETE")
    print("="*60)
    
    metrics = env.calculate_performance_metrics()
    print("\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nTotal steps: {step}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final equity: ${env.equity:.2f}")
    print(f"Total return: {(env.equity - config.initial_balance) / config.initial_balance:.2%}")
    
    # Save episode data
    print("\nSaving episode data...")
    env.save_episode_data('test_episode.json')
    print("Episode data saved to test_episode.json")
    
    # Test trade history
    if env.trade_history:
        print(f"\nSample trades ({len(env.trade_history)} total):")
        for i, trade in enumerate(env.trade_history[:3]):
            print(f"\nTrade {i+1}:")
            print(f"  Type: {trade.position_type.name}")
            print(f"  Entry: {trade.entry_price:.4f} -> Exit: {trade.exit_price:.4f}")
            print(f"  P&L: ${trade.net_pnl:.2f} ({trade.pnl_pips:.1f} pips)")
            print(f"  Duration: {trade.duration_bars} bars")
            print(f"  Exit reason: {trade.exit_reason}")


# =====================================================
# MAIN ENTRY POINT
# =====================================================

if __name__ == "__main__":
    test_environment()