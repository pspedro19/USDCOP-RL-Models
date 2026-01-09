#!/usr/bin/env python3
"""
V20 Training Script - Production Parity
========================================

This script trains a PPO model with EXACT parity to production:
- 15-dimensional observation space (13 core + 2 state)
- Same features as ObservationBuilderV19
- Compatible with production inference pipeline

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# V19 Production Features (MUST match ObservationBuilderV19)
V19_CORE_FEATURES = [
    "log_ret_5m",
    "log_ret_1h",
    "log_ret_4h",
    "rsi_9",
    "atr_pct",
    "adx_14",
    "dxy_z",
    "dxy_change_1d",
    "vix_z",
    "embi_z",
    "brent_change_1d",
    "rate_spread",
    "usdmxn_change_1d"
]
V19_STATE_FEATURES = ["position", "time_normalized"]
V19_OBS_DIM = 15  # 13 core + 2 state

# Configuration
CONFIG = {
    "dataset_path": "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv",
    "norm_stats_path": "config/v19_norm_stats.json",
    "output_dir": "models/ppo_v20_production",
    "total_timesteps": 500_000,
    "episode_length": 1200,  # 20 days @ 60 bars/day
    "initial_balance": 10_000,
    "max_drawdown_pct": 15.0,
    "threshold_long": 0.10,
    "threshold_short": -0.10,
    "ppo_config": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # Entropy for exploration
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "network_arch": [64, 64],
}


class ProductionParityEnv(gym.Env):
    """
    Trading Environment with Production Parity.

    Ensures 15-dimensional observation space matching production exactly.
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        df: pd.DataFrame,
        norm_stats: Dict,
        initial_balance: float = 10_000,
        episode_length: int = 1200,
        max_drawdown_pct: float = 15.0,
        transaction_cost_bps: float = 25.0,
        threshold_long: float = 0.10,
        threshold_short: float = -0.10,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.norm_stats = norm_stats
        self.initial_balance = initial_balance
        self.episode_length = episode_length
        self.max_drawdown = max_drawdown_pct / 100
        self.transaction_cost = transaction_cost_bps / 10000
        self.threshold_long = threshold_long
        self.threshold_short = threshold_short

        # Validate features
        missing = [f for f in V19_CORE_FEATURES if f not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing features: {missing}")

        # Extract feature matrix for fast access
        self.features = df[V19_CORE_FEATURES].values.astype(np.float32)
        self.returns = df['log_ret_5m'].values.astype(np.float32)
        self.n_bars = len(df)

        # Action space: continuous [-1, 1] -> maps to {short, hold, long}
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space: 15 dimensions (production parity)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=(V19_OBS_DIM,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random start point (with margin for episode)
        max_start = self.n_bars - self.episode_length - 1
        self.start_idx = self.np_random.integers(0, max(1, max_start))
        self.current_idx = self.start_idx

        # Portfolio state
        self.balance = self.initial_balance
        self.position = 0.0  # -1=short, 0=flat, 1=long
        self.entry_price = 0.0
        self.equity = self.initial_balance
        self.peak_equity = self.initial_balance

        # Episode tracking
        self.step_count = 0
        self.trades = 0
        self.episode_return = 0.0

        return self._get_observation(), {}

    def _get_observation(self) -> np.ndarray:
        """Build 15-dimensional observation (production parity)."""
        obs = np.zeros(V19_OBS_DIM, dtype=np.float32)

        # Core market features (indices 0-12)
        raw_features = self.features[self.current_idx]
        for i, (fname, value) in enumerate(zip(V19_CORE_FEATURES, raw_features)):
            obs[i] = self._normalize(fname, value)

        # State features (indices 13-14)
        obs[13] = self.position  # Already in [-1, 1]
        obs[14] = self.step_count / self.episode_length  # time_normalized

        # Clip and handle NaN
        obs = np.clip(obs, -5.0, 5.0)
        obs = np.nan_to_num(obs, nan=0.0)

        return obs

    def _normalize(self, feature: str, value: float) -> float:
        """Z-score normalize using production stats."""
        if feature not in self.norm_stats:
            return np.clip(value, -5.0, 5.0)

        stats = self.norm_stats[feature]
        mean = stats.get('mean', 0.0)
        std = stats.get('std', 1.0)
        if std < 1e-8:
            std = 1.0

        z = (value - mean) / std
        return np.clip(z, -5.0, 5.0)

    def step(self, action: np.ndarray):
        action_value = float(action[0])

        # Map continuous action to discrete signal using thresholds
        if action_value > self.threshold_long:
            target_position = 1.0  # Long
        elif action_value < self.threshold_short:
            target_position = -1.0  # Short
        else:
            target_position = 0.0  # Hold/Flat

        # Calculate position change and cost
        position_change = abs(target_position - self.position)
        cost = position_change * self.transaction_cost * self.balance

        if position_change > 0:
            self.trades += 1

        # Update position
        old_position = self.position
        self.position = target_position

        # Move to next bar
        self.current_idx += 1
        self.step_count += 1

        # Calculate PnL from market return
        market_return = self.returns[self.current_idx]
        pnl = self.position * market_return * self.balance - cost

        self.balance += pnl
        self.equity = self.balance
        self.peak_equity = max(self.peak_equity, self.equity)
        self.episode_return = (self.equity / self.initial_balance) - 1

        # Calculate drawdown
        drawdown = (self.peak_equity - self.equity) / self.peak_equity if self.peak_equity > 0 else 0

        # Reward function (simplified for stability)
        reward = self._compute_reward(pnl, position_change, market_return)

        # Termination conditions
        terminated = (
            drawdown > self.max_drawdown or
            self.step_count >= self.episode_length or
            self.current_idx >= self.n_bars - 1
        )

        truncated = False

        info = {
            'equity': self.equity,
            'position': self.position,
            'trades': self.trades,
            'drawdown': drawdown,
            'episode_return': self.episode_return,
            'action_value': action_value,
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _compute_reward(self, pnl: float, position_change: float, market_return: float) -> float:
        """
        Compute reward with production-aligned logic.
        """
        # Base: scaled PnL
        reward = pnl / self.initial_balance * 100

        # Asymmetric penalty for losses
        if reward < 0:
            reward *= 1.5

        # Small penalty for excessive trading
        if position_change > 0:
            reward -= 0.01

        # Bonus for profitable trades
        if self.position != 0 and pnl > 0:
            reward += 0.02

        return np.clip(reward, -5.0, 5.0)


class ActionDistributionCallback(BaseCallback):
    """Monitor action distribution during training."""

    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.actions = []

    def _on_step(self) -> bool:
        self.actions.append(self.locals['actions'][0])

        if self.n_calls % self.log_freq == 0:
            actions = np.array(self.actions[-self.log_freq:])

            # Calculate distribution
            long_pct = np.mean(actions > 0.1) * 100
            short_pct = np.mean(actions < -0.1) * 100
            hold_pct = np.mean(np.abs(actions) <= 0.1) * 100

            print(f"\n[Step {self.n_calls}] Action Distribution:")
            print(f"  LONG:  {long_pct:.1f}%")
            print(f"  HOLD:  {hold_pct:.1f}%")
            print(f"  SHORT: {short_pct:.1f}%")

            # Warning if collapsed
            if hold_pct > 80:
                print("  WARNING: High HOLD rate - model may need more exploration")
            elif long_pct > 60 or short_pct > 60:
                print("  WARNING: Directional bias detected")

        return True


def load_norm_stats(path: str) -> Dict:
    """Load normalization statistics."""
    with open(path, 'r') as f:
        return json.load(f)


def make_env(df: pd.DataFrame, norm_stats: Dict, config: Dict):
    """Factory function for vectorized environments."""
    def _init():
        return ProductionParityEnv(
            df=df,
            norm_stats=norm_stats,
            initial_balance=config['initial_balance'],
            episode_length=config['episode_length'],
            max_drawdown_pct=config['max_drawdown_pct'],
            threshold_long=config['threshold_long'],
            threshold_short=config['threshold_short'],
        )
    return _init


def train():
    """Main training function."""
    print("="*70)
    print("V20 TRAINING - PRODUCTION PARITY")
    print("="*70)
    print(f"Observation dimensions: {V19_OBS_DIM} (same as production)")
    print(f"Features: {V19_CORE_FEATURES}")
    print(f"State: {V19_STATE_FEATURES}")
    print("="*70)

    # Load data
    dataset_path = PROJECT_ROOT / CONFIG['dataset_path']
    print(f"\nLoading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"  Rows: {len(df):,}")

    # Load normalization stats
    norm_stats_path = PROJECT_ROOT / CONFIG['norm_stats_path']
    print(f"\nLoading norm stats: {norm_stats_path}")
    norm_stats = load_norm_stats(norm_stats_path)

    # Split data
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    print(f"\nData splits:")
    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    # Create environments
    print("\nCreating environments...")
    train_env = DummyVecEnv([make_env(train_df, norm_stats, CONFIG)])
    eval_env = DummyVecEnv([make_env(val_df, norm_stats, CONFIG)])

    # Verify observation space
    obs = train_env.reset()
    print(f"\nEnvironment observation space: {train_env.observation_space.shape}")
    print(f"Initial observation shape: {obs.shape}")
    assert obs.shape[1] == V19_OBS_DIM, f"Expected {V19_OBS_DIM} dims, got {obs.shape[1]}"
    print("  PARITY CHECK PASSED!")

    # Create output directory
    output_dir = PROJECT_ROOT / CONFIG['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nTraining device: {device}")

    # Create model
    print("\nCreating PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        **CONFIG['ppo_config'],
        policy_kwargs={"net_arch": CONFIG['network_arch']},
        tensorboard_log=str(output_dir / "tensorboard"),
        device=device,
        verbose=1,
    )

    # Callbacks
    callbacks = [
        ActionDistributionCallback(log_freq=10000),
        EvalCallback(
            eval_env,
            best_model_save_path=str(output_dir),
            log_path=str(output_dir / "eval_logs"),
            eval_freq=25000,
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    # Train
    print(f"\nStarting training for {CONFIG['total_timesteps']:,} steps...")
    print("-"*70)

    model.learn(
        total_timesteps=CONFIG['total_timesteps'],
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    model_path = output_dir / "final_model.zip"
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save config
    config_path = output_dir / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(CONFIG, f, indent=2)
    print(f"Config saved to: {config_path}")

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)

    return model, output_dir


if __name__ == "__main__":
    train()
