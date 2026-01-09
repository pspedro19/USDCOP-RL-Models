"""
Trading Environment for L5 Production
=====================================
Creates trading environment using L4 data from MinIO
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

# Try to import gymnasium (newer) or gym (older)
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        # Create minimal replacements if neither is available
        class spaces:
            @staticmethod
            def Box(low, high, shape, dtype):
                return {"type": "Box", "low": low, "high": high, "shape": shape, "dtype": dtype}
            
            @staticmethod
            def Discrete(n):
                return {"type": "Discrete", "n": n}
        
        class gym:
            class Env:
                pass

logger = logging.getLogger(__name__)

class USDCOPTradingEnv:
    """
    Trading environment for USDCOP with L4 data
    Compatible with stable-baselines3
    """
    
    def __init__(
        self, 
        data_path: str = None,
        mode: str = "train",
        transaction_cost: float = 0.001,
        position_size: float = 0.1
    ):
        # No need for super().__init__() since we're not inheriting from gym.Env
        
        self.mode = mode
        self.transaction_cost = transaction_cost
        self.position_size = position_size
        
        # Load data based on mode
        if data_path:
            self.df = pd.read_parquet(data_path)
        else:
            # Default paths for L4 data
            if mode == "train":
                self.df = pd.read_parquet("/tmp/train_df.parquet")
            elif mode == "test":
                self.df = pd.read_parquet("/tmp/test_df.parquet")
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        # Setup spaces (compatible with stable-baselines3)
        n_features = 17  # Based on L4 feature engineering
        
        # Create observation space
        try:
            self.observation_space = spaces.Box(
                low=-5.0, 
                high=5.0, 
                shape=(n_features,), 
                dtype=np.float32
            )
        except:
            # Fallback for when gym is not available
            self.observation_space = type('Space', (), {
                'shape': (n_features,),
                'dtype': np.float32,
                'low': np.array([-5.0] * n_features, dtype=np.float32),
                'high': np.array([5.0] * n_features, dtype=np.float32)
            })()
        
        # Action space: 0=hold, 1=buy, 2=sell
        try:
            self.action_space = spaces.Discrete(3)
        except:
            # Fallback for when gym is not available
            self.action_space = type('Space', (), {
                'n': 3,
                'shape': (),
                'dtype': np.int64
            })()
        
        # Initialize state
        self.reset()
        
        logger.info(f"Created {mode} environment with {len(self.df)} timesteps")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 0
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []
        
        # Get initial observation
        obs = self._get_observation()
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        
        # Get current price
        current_row = self.df.iloc[self.current_step]
        current_price = current_row.get('close', current_row.get('Close', 0))
        
        # Calculate reward
        reward = 0
        
        # Execute action
        if action == 1 and self.position <= 0:  # Buy
            if self.position < 0:  # Close short
                profit = (self.entry_price - current_price) * self.position_size
                profit -= self.transaction_cost * self.position_size * 2
                reward = profit
                self.total_profit += profit
                self.trades.append({
                    'type': 'close_short',
                    'profit': profit,
                    'price': current_price
                })
            
            # Open long
            self.position = 1
            self.entry_price = current_price
            reward -= self.transaction_cost * self.position_size
            
        elif action == 2 and self.position >= 0:  # Sell
            if self.position > 0:  # Close long
                profit = (current_price - self.entry_price) * self.position_size
                profit -= self.transaction_cost * self.position_size * 2
                reward = profit
                self.total_profit += profit
                self.trades.append({
                    'type': 'close_long',
                    'profit': profit,
                    'price': current_price
                })
            
            # Open short
            self.position = -1
            self.entry_price = current_price
            reward -= self.transaction_cost * self.position_size
        
        # Move to next step
        self.current_step += 1
        
        # Check if done
        done = self.current_step >= len(self.df) - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'position': self.position,
            'total_profit': self.total_profit,
            'n_trades': len(self.trades),
            'current_price': current_price
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation from data"""
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        row = self.df.iloc[self.current_step]
        
        # Extract features (adjust based on your L4 feature names)
        feature_cols = [
            'returns_5m', 'returns_15m', 'returns_1h', 'returns_4h',
            'rsi_14', 'rsi_28', 'macd_signal', 'macd_diff',
            'bb_position', 'volume_ratio', 'spread',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_london', 'is_ny'
        ]
        
        # Try to get features, use defaults if not found
        features = []
        for col in feature_cols:
            if col in row:
                features.append(row[col])
            else:
                # Try alternate naming
                alt_col = col.replace('_', '')
                if alt_col in row:
                    features.append(row[alt_col])
                else:
                    features.append(0.0)
        
        # Ensure we have the right number of features
        if len(features) < self.observation_space.shape[0]:
            features.extend([0.0] * (self.observation_space.shape[0] - len(features)))
        elif len(features) > self.observation_space.shape[0]:
            features = features[:self.observation_space.shape[0]]
        
        obs = np.array(features, dtype=np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs
    
    def render(self, mode='human'):
        """Render the environment (optional)"""
        if mode == 'human':
            print(f"Step: {self.current_step}, Position: {self.position}, "
                  f"Total Profit: {self.total_profit:.4f}")

def create_trading_env(
    mode: str = "train",
    data_path: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> USDCOPTradingEnv:
    """
    Factory function to create trading environment
    
    Args:
        mode: "train" or "test"
        data_path: Optional path to data file
        config: Optional configuration dict
    
    Returns:
        Configured trading environment
    """
    
    # Load config if provided
    if config is None:
        config = {}
    
    # Try to load from saved specs if they exist
    try:
        if os.path.exists("/tmp/reward_spec.json"):
            with open("/tmp/reward_spec.json", 'r') as f:
                reward_spec = json.load(f)
                config['transaction_cost'] = reward_spec.get('transaction_cost', 0.001)
                config['position_size'] = reward_spec.get('position_size', 0.1)
    except Exception as e:
        logger.warning(f"Could not load reward spec: {e}")
    
    try:
        if os.path.exists("/tmp/cost_model.json"):
            with open("/tmp/cost_model.json", 'r') as f:
                cost_model = json.load(f)
                config['transaction_cost'] = cost_model.get('base_cost', 0.001)
    except Exception as e:
        logger.warning(f"Could not load cost model: {e}")
    
    # Create environment
    env = USDCOPTradingEnv(
        data_path=data_path,
        mode=mode,
        transaction_cost=config.get('transaction_cost', 0.001),
        position_size=config.get('position_size', 0.1)
    )
    
    return env