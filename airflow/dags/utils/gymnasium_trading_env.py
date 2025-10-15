"""
Gymnasium-compatible Trading Environment for L5 Production
==========================================================
Proper Gymnasium environment that works with stable-baselines3
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, Tuple, Dict, Any

# Use dependency handler for gymnasium
from .dependency_handler import GYMNASIUM_HANDLER

# Check if gymnasium is available
HAS_GYMNASIUM = GYMNASIUM_HANDLER.is_available

if HAS_GYMNASIUM:
    gym = GYMNASIUM_HANDLER.get_module()
    from gymnasium import spaces
else:
    gym = None
    # Create minimal gymnasium-compatible interface for fallback
    
    class Env:
        """Minimal Gymnasium Env interface"""
        metadata = {"render_modes": []}
        
        def reset(self, *, seed=None, options=None):
            raise NotImplementedError
            
        def step(self, action):
            raise NotImplementedError
    
    class spaces:
        @staticmethod
        def Box(low, high, shape, dtype):
            class BoxSpace:
                def __init__(self):
                    self.low = np.array([low] * shape[0] if np.isscalar(low) else low, dtype=dtype)
                    self.high = np.array([high] * shape[0] if np.isscalar(high) else high, dtype=dtype)
                    self.shape = shape
                    self.dtype = dtype
            return BoxSpace()
        
        @staticmethod
        def Discrete(n):
            class DiscreteSpace:
                def __init__(self):
                    self.n = n
                    self.shape = ()
                    self.dtype = np.int64
            return DiscreteSpace()
    
    class gym:
        Env = Env

logger = logging.getLogger(__name__)

class USDCOPTradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for USDCOP
    Conforms to L4/L5 contracts: 17 observations in [-5, 5], 3 discrete actions
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(self, mode: str = "train", transaction_cost: float = 0.001):
        super().__init__()
        
        self.mode = mode
        self.transaction_cost = transaction_cost
        
        # Define action and observation spaces per L4/L5 contract
        self.action_space = spaces.Discrete(3)  # 0=hold, 1=buy, 2=sell
        self.observation_space = spaces.Box(
            low=-5.0, 
            high=5.0, 
            shape=(17,), 
            dtype=np.float32
        )
        
        # Load data
        self._load_data(mode)
        
        # Trading state
        self.current_step = 0
        self.position = 0  # -1=short, 0=flat, 1=long
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        logger.info(f"Created {mode} Gymnasium environment with {len(self.data)} steps")
    
    def _load_data(self, mode: str):
        """Load preprocessed L4 data"""
        try:
            path = f"/tmp/{mode}_df.parquet"
            self.data = pd.read_parquet(path)
            
            # CRITICAL FIX: Extract ret_forward_1 for proper reward calculation
            # This is the L4 pre-computed forward return that should be used for rewards
            if 'ret_forward_1' in self.data.columns:
                self.forward_returns = self.data['ret_forward_1'].values
                logger.info(f"Loaded {len(self.forward_returns)} forward returns from L4 data")
            else:
                logger.warning("ret_forward_1 not found in L4 data - will use fallback pricing")
                self.forward_returns = None
            
            # Extract prices (for position tracking only, not reward calculation)
            if 'Close' in self.data.columns:
                self.prices = self.data['Close'].values
            elif 'close' in self.data.columns:
                self.prices = self.data['close'].values
            elif 'mid' in self.data.columns:
                self.prices = self.data['mid'].values
            else:
                # Fallback to first numeric column
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns
                self.prices = self.data[numeric_cols[0]].values if len(numeric_cols) > 0 else np.ones(len(self.data))
            
            # Extract or create observation matrix
            # Look for normalized features from L4
            obs_cols = []
            
            # Try standard feature names first
            standard_features = [
                'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h',
                'rsi_14', 'rsi_28', 'macd_signal', 'macd_diff',
                'bb_position', 'volume_ratio', 'spread',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'is_london', 'is_ny'
            ]
            
            for feat in standard_features:
                if feat in self.data.columns:
                    obs_cols.append(feat)
            
            # If we don't have enough features, use any numeric columns
            if len(obs_cols) < 17:
                numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols:
                    if col not in obs_cols and len(obs_cols) < 17:
                        obs_cols.append(col)
            
            # If still not enough, pad with zeros
            while len(obs_cols) < 17:
                pad_col = f'pad_{len(obs_cols)}'
                self.data[pad_col] = 0.0
                obs_cols.append(pad_col)
            
            # Create observation matrix
            self.obs_matrix = self.data[obs_cols[:17]].values.astype(np.float32)
            
            # Handle NaN and Inf values
            self.obs_matrix = np.nan_to_num(self.obs_matrix, nan=0.0, posinf=5.0, neginf=-5.0)
            
            # Clip to observation space bounds
            self.obs_matrix = np.clip(self.obs_matrix, -5.0, 5.0)
            
        except Exception as e:
            logger.warning(f"Could not load {mode} data: {e}. Using synthetic data.")
            # Create synthetic data for testing
            n_steps = 1000
            price_returns = np.random.randn(n_steps) * 0.0001  # Small random returns
            self.data = pd.DataFrame({
                'Close': 4000 + np.random.randn(n_steps).cumsum() * 10,
                'ret_forward_1': price_returns  # Synthetic forward returns centered at 0
            })
            self.prices = self.data['Close'].values
            self.forward_returns = self.data['ret_forward_1'].values
            self.obs_matrix = np.random.randn(n_steps, 17).astype(np.float32)
            self.obs_matrix = np.clip(self.obs_matrix, -5.0, 5.0)
            logger.info(f"Created synthetic forward returns with mean={np.mean(self.forward_returns):.6f}")
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment (Gymnasium API)
        Returns: (observation, info)
        """
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        self.action_counts = {'buy': 0, 'sell': 0, 'hold': 0}  # Faithful tracking
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step (Gymnasium API)
        Returns: (observation, reward, terminated, truncated, info)
        """
        # Track action counts based on ACTUAL position changes
        old_position = self.position
        
        # Get current price
        current_price = self.prices[self.current_step]
        
        # Calculate reward based on action and position
        reward = self._calculate_reward(action, current_price)
        
        # Update position based on action
        self._update_position(action, current_price)
        
        # 100% faithful trade counting based on position changes
        if old_position != self.position:
            if self.position == 1:  # Moved to long
                self.action_counts['buy'] += 1
            elif self.position == -1:  # Moved to short
                self.action_counts['sell'] += 1
        elif self.position is not None:  # Position unchanged and exists
            self.action_counts['hold'] += 1
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False  # Could add max step truncation if needed
        
        # Get next observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        if self.current_step >= len(self.obs_matrix):
            return np.zeros(17, dtype=np.float32)
        
        return self.obs_matrix[self.current_step]
    
    def _calculate_reward(self, action: int, current_price: float) -> float:
        """
        Calculate reward based on L4 forward returns (ret_forward_1)
        CRITICAL FIX: Use position * ret_forward_1 for pure signal reward
        This ensures sanity check gets clean log returns without trading logic contamination
        """
        # IMPORTANT: The reward should be based on ret_forward_1 from L4 data
        # This is the log(mid_t1/mid_t) that sanity check expects
        if self.forward_returns is not None and self.current_step < len(self.forward_returns):
            forward_return = self.forward_returns[self.current_step]
            
            # Handle NaN values (end of episodes)
            if pd.isna(forward_return):
                reward = 0.0
            else:
                # Calculate position for reward computation
                # Map actions to positions: 0=hold/flat, 1=long, 2=short
                if action == 0:  # Hold - maintain current position
                    position_for_reward = self.position
                elif action == 1:  # Buy signal - go long
                    position_for_reward = 1
                elif action == 2:  # Sell signal - go short  
                    position_for_reward = -1
                else:
                    position_for_reward = 0
                
                # Pure signal reward: position * forward_return
                # NO transaction costs here - those are handled by SentinelTradingEnv wrapper
                reward = position_for_reward * forward_return
            
        else:
            # Fallback to old logic if ret_forward_1 not available
            logger.warning("Using fallback reward calculation - ret_forward_1 not available")
            reward = 0.0
            
            if action == 1 and self.position <= 0:  # Buy signal
                if self.position < 0:  # Close short position
                    profit = (self.entry_price - current_price) / self.entry_price
                    profit -= self.transaction_cost * 2  # Entry + exit costs
                    reward = profit
                    self.trades.append({'type': 'close_short', 'profit': profit})
                # Cost of opening long
                reward -= self.transaction_cost
                
            elif action == 2 and self.position >= 0:  # Sell signal
                if self.position > 0:  # Close long position
                    profit = (current_price - self.entry_price) / self.entry_price
                    profit -= self.transaction_cost * 2  # Entry + exit costs
                    reward = profit
                    self.trades.append({'type': 'close_long', 'profit': profit})
                # Cost of opening short
                reward -= self.transaction_cost
            
            elif action == 0:  # Hold
                # Small negative reward to encourage action when appropriate
                reward = -0.0001
        
        self.total_reward += reward
        return reward
    
    def _update_position(self, action: int, current_price: float):
        """Update position based on action"""
        if action == 1 and self.position <= 0:  # Buy
            self.position = 1
            self.entry_price = current_price
        elif action == 2 and self.position >= 0:  # Sell
            self.position = -1
            self.entry_price = current_price
        # action == 0 means hold, no position change
    
    def _get_info(self) -> Dict[str, Any]:
        """Get info dictionary"""
        return {
            'position': self.position,
            'total_reward': self.total_reward,
            'n_trades': len(self.trades),
            'current_step': self.current_step,
            'current_price': self.prices[min(self.current_step, len(self.prices)-1)],
            'action_counts': self.action_counts.copy()  # Faithful action counts
        }
    
    def render(self, mode: str = "human"):
        """Render the environment (optional)"""
        if mode == "human":
            price = self.prices[min(self.current_step, len(self.prices)-1)]
            print(f"Step: {self.current_step}, Price: {price:.2f}, "
                  f"Position: {self.position}, Total Reward: {self.total_reward:.4f}")

def create_gym_env(mode: str = "train", **kwargs) -> USDCOPTradingEnv:
    """
    Factory function to create Gymnasium-compatible environment
    Used everywhere in the DAG (train/test/gates)
    """
    return USDCOPTradingEnv(mode=mode, **kwargs)