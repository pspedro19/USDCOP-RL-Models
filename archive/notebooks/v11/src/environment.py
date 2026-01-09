"""
USD/COP RL Trading System V11 - Trading Environment
====================================================

Key V11 Fix: Uses RAW returns for portfolio calculation,
NOT the normalized log_ret_5m feature.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TradingEnvV11(gym.Env):
    """
    Trading Environment V11 - Raw returns for portfolio.

    Key fix: Use RAW returns from _raw_ret_5m column,
    NOT the normalized log_ret_5m feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with features and raw returns
    features : list
        List of feature column names for model input
    episode_length : int
        Number of steps per episode
    initial_balance : float
        Starting portfolio value
    cost : float
        Transaction cost per trade (as fraction)
    raw_ret_col : str
        Column name for raw (unnormalized) returns
    """

    def __init__(
        self,
        df,
        features,
        episode_length=288,
        initial_balance=10000,
        cost=0.0003,
        raw_ret_col='_raw_ret_5m'
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.features = features
        self.episode_length = episode_length
        self.initial_balance = initial_balance
        self.cost = cost
        self.raw_ret_col = raw_ret_col

        # Log-space tracking (numerically stable)
        self.initial_log_balance = np.log(initial_balance)
        self.log_portfolio_clip_min = self.initial_log_balance - 10
        self.log_portfolio_clip_max = self.initial_log_balance + 10

        # Observation: features + position + normalized time
        obs_dim = len(features) + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: continuous position from -1 (full short) to +1 (full long)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)

        max_start = max(1, len(self.df) - self.episode_length - 2)
        self.start_idx = self.np_random.integers(0, max_start)
        self.step_count = 0
        self.position = 0.0

        # Log-space portfolio
        self.log_portfolio = self.initial_log_balance
        self.log_portfolio_prev = self.initial_log_balance

        self.trades = 0

        return self._obs(), {}

    def _obs(self):
        """Get observation (normalized features for model)."""
        idx = min(self.start_idx + self.step_count, len(self.df) - 1)

        feat = self.df.iloc[idx][self.features].values.astype(np.float32)
        feat = np.nan_to_num(feat, 0)

        state = np.array([
            self.position,
            self.step_count / self.episode_length
        ], dtype=np.float32)

        return np.clip(np.concatenate([feat, state]), -5, 5)

    def _get_raw_return(self):
        """
        V11 FIX: Get RAW return from _raw_ret_5m column.
        NOT the normalized feature!
        """
        idx = min(self.start_idx + self.step_count + 1, len(self.df) - 1)

        if self.raw_ret_col in self.df.columns:
            return float(self.df.iloc[idx][self.raw_ret_col])

        # Fallback (should not happen)
        return 0.0

    def step(self, action):
        """Execute one step in the environment."""
        self.log_portfolio_prev = self.log_portfolio
        prev_pos = self.position

        # Parse action
        if hasattr(action, '__len__'):
            new_pos = float(np.clip(action[0], -1, 1))
        else:
            new_pos = float(np.clip(action, -1, 1))

        # Transaction cost (in log-space)
        pos_change = abs(new_pos - prev_pos)
        if pos_change > 0.1:
            cost_factor = 1.0 - pos_change * self.cost
            self.log_portfolio += np.log(max(cost_factor, 0.001))
            self.trades += 1

        self.position = new_pos

        # V11 FIX: Use RAW return, not normalized!
        mkt_ret = self._get_raw_return()

        # Portfolio update in log-space
        self.log_portfolio += self.position * mkt_ret

        # Clip to prevent extreme values
        self.log_portfolio = np.clip(
            self.log_portfolio,
            self.log_portfolio_clip_min,
            self.log_portfolio_clip_max
        )

        self.step_count += 1

        # REWARD: log-return scaled for learning
        reward = (self.log_portfolio - self.log_portfolio_prev) * 100

        # Convert to real portfolio for info
        portfolio = np.exp(self.log_portfolio)

        # Episode termination
        done = self.log_portfolio <= self.log_portfolio_clip_min + 0.1
        truncated = self.step_count >= self.episode_length

        info = {
            'portfolio': portfolio,
            'log_portfolio': self.log_portfolio,
            'trades': self.trades,
            'position': self.position,
            'market_return': mkt_ret
        }

        return self._obs(), reward, done, truncated, info
