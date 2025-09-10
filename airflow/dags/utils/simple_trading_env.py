"""
Simplified Trading Environment for L5 Production
================================================
Minimal trading environment that works without gym/gymnasium
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger(__name__)

class SimpleTradingEnv:
    """
    Simplified trading environment compatible with stable-baselines3
    Uses vectorized operations for efficiency
    """
    
    def __init__(self, data_df: pd.DataFrame = None, mode: str = "train"):
        self.mode = mode
        
        # Load data
        if data_df is not None:
            self.data = data_df
        else:
            # Load from default location
            if mode == "train":
                self.data = pd.read_parquet("/tmp/train_df.parquet")
            else:
                self.data = pd.read_parquet("/tmp/test_df.parquet")
        
        # Convert to numpy for speed
        self.prices = self.data[['Close', 'High', 'Low', 'Open']].values if 'Close' in self.data.columns else self.data.values[:, :4]
        self.n_steps = len(self.data)
        
        # Trading parameters
        self.transaction_cost = 0.001
        self.position_size = 0.1
        
        # State variables
        self.current_step = 0
        self.position = 0  # -1: short, 0: flat, 1: long
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        logger.info(f"Created {mode} environment with {self.n_steps} steps")
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.total_reward = 0
        self.trades = []
        
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute trading action
        Returns: (observation, reward, terminated, truncated, info)
        """
        
        # Get current price
        current_price = self.prices[self.current_step, 0]  # Close price
        
        # Calculate reward based on action
        reward = 0.0
        
        if action == 1 and self.position <= 0:  # Buy signal
            if self.position < 0:  # Close short
                profit = (self.entry_price - current_price) * self.position_size
                profit -= self.transaction_cost * 2
                reward = profit
                self.trades.append(profit)
            
            # Open long
            self.position = 1
            self.entry_price = current_price
            reward -= self.transaction_cost
            
        elif action == 2 and self.position >= 0:  # Sell signal
            if self.position > 0:  # Close long
                profit = (current_price - self.entry_price) * self.position_size
                profit -= self.transaction_cost * 2
                reward = profit
                self.trades.append(profit)
            
            # Open short
            self.position = -1
            self.entry_price = current_price
            reward -= self.transaction_cost
        
        # Hold position generates small negative reward (opportunity cost)
        elif action == 0:
            reward = -0.0001
        
        self.total_reward += reward
        self.current_step += 1
        
        # Check if done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False
        
        # Get next state
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'position': self.position,
            'total_reward': self.total_reward,
            'n_trades': len(self.trades),
            'current_price': current_price
        }
        
        return next_state, reward, terminated, truncated, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_step >= self.n_steps - 1:
            return np.zeros(17, dtype=np.float32)
        
        # Simple feature engineering
        idx = self.current_step
        current_price = self.prices[idx, 0]
        
        # Price features
        returns_1 = (self.prices[idx, 0] / self.prices[max(0, idx-1), 0] - 1) if idx > 0 else 0
        returns_5 = (self.prices[idx, 0] / self.prices[max(0, idx-5), 0] - 1) if idx > 5 else 0
        returns_20 = (self.prices[idx, 0] / self.prices[max(0, idx-20), 0] - 1) if idx > 20 else 0
        
        # Volatility
        if idx > 20:
            recent_returns = np.diff(self.prices[max(0, idx-20):idx+1, 0]) / self.prices[max(0, idx-20):idx, 0]
            volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0
        else:
            volatility = 0
        
        # Simple technical indicators
        if idx > 14:
            sma_14 = np.mean(self.prices[idx-14:idx+1, 0])
            rsi_indicator = (current_price - sma_14) / (sma_14 + 1e-8)
        else:
            rsi_indicator = 0
        
        # Position encoding
        position_long = 1 if self.position > 0 else 0
        position_short = 1 if self.position < 0 else 0
        
        # Time features
        progress = idx / self.n_steps
        
        # Create feature vector (17 features to match expected)
        features = np.array([
            returns_1, returns_5, returns_20,  # 3 return features
            volatility, rsi_indicator,  # 2 technical features
            position_long, position_short,  # 2 position features
            progress,  # 1 time feature
            0, 0, 0, 0, 0, 0, 0, 0, 0  # 9 padding features to reach 17
        ], dtype=np.float32)
        
        # Clip to [-5, 5] range
        features = np.clip(features, -5, 5)
        
        return features

def create_simple_env_wrapper(mode: str = "train") -> Any:
    """
    Create environment wrapper that mimics gymnasium interface
    """
    env = SimpleTradingEnv(mode=mode)
    
    # Add gymnasium-like attributes
    class EnvWrapper:
        def __init__(self, env):
            self.env = env
            self.observation_space = type('Space', (), {
                'shape': (17,),
                'dtype': np.float32,
                'low': np.array([-5.0] * 17, dtype=np.float32),
                'high': np.array([5.0] * 17, dtype=np.float32),
                'sample': lambda: np.random.uniform(-5, 5, 17).astype(np.float32)
            })()
            self.action_space = type('Space', (), {
                'n': 3,
                'shape': (),
                'dtype': np.int64,
                'sample': lambda: np.random.randint(0, 3)
            })()
            
            # Gymnasium compatibility
            self.spec = type('Spec', (), {'id': 'SimpleTradingEnv-v0'})()
            self.metadata = {'render_modes': []}
            self.render_mode = None
        
        def reset(self, seed=None, options=None):
            obs = self.env.reset()
            info = {}
            return obs, info
        
        def step(self, action):
            return self.env.step(action)
        
        def render(self):
            pass
        
        def close(self):
            pass
    
    return EnvWrapper(env)