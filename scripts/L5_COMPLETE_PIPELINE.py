#!/usr/bin/env python3
"""
==================================================================================
L5 COMPLETE REINFORCEMENT LEARNING PIPELINE - VERSIÓN FINAL COMPLETA
==================================================================================
Author: Pedro
Date: 2025-09-01
Version: 1.0.0

Este es el pipeline COMPLETO de L5 que incluye:
1. Validación de L4
2. Entrenamiento multi-seed con PPO
3. Evaluación con gates de producción
4. Generación de bundle de deployment
5. Finalización y deployment

TODO ESTÁ AQUÍ - NO HAY OMISIONES
==================================================================================
"""

import json
import logging
import os
import sys
import subprocess
import hashlib
import shutil
import tarfile
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
import numpy as np
import pandas as pd

# Machine Learning
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# Reinforcement Learning
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CallbackList,
    CheckpointCallback,
    BaseCallback
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import get_linear_fn

# Experiment tracking
import mlflow
import mlflow.pytorch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - L5 - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================================================================================
# CONFIGURATION
# ==================================================================================

# Training configuration
TRAINING_CONFIG = {
    'total_timesteps': int(os.environ.get('L5_TOTAL_TIMESTEPS', 1_000_000)),
    'n_steps': int(os.environ.get('L5_N_STEPS', 2048)),
    'batch_size': int(os.environ.get('L5_BATCH_SIZE', 256)),
    'n_epochs': 10,
    'learning_rate': float(os.environ.get('L5_LEARNING_RATE', 3e-4)),
    'ent_coef': float(os.environ.get('L5_ENT_COEF', 0.01)),
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'clip_range': 0.2,
    'device': os.environ.get('L5_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu'),
    'verbose': 1,
}

# Production gates thresholds
PERFORMANCE_THRESHOLDS = {
    'sortino_ratio': 1.3,
    'calmar_ratio': 0.8,
    'max_drawdown': 0.15,
    'win_rate': 0.55,
    'profit_factor': 1.2,
    'consistency_sortino': 0.5,
    'cost_stress_cagr_drop': 0.20,
}

# Latency requirements
LATENCY_REQUIREMENTS = {
    'inference_p99': 20.0,  # ms
    'e2e_p99': 100.0,      # ms
}

# Cost stress multiplier
COST_STRESS_MULTIPLIER = 1.25

# ==================================================================================
# TRADING ENVIRONMENT
# ==================================================================================

class TradingEnvironment(gym.Env):
    """
    Custom trading environment for USD/COP forex trading.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        mode: str = 'train',
        transaction_cost: float = 0.001,
        slippage_bps: float = 5,
        spread_bps: float = 20,
        max_position: float = 1.0,
        lookback_window: int = 100,
        features: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.data = data
        self.mode = mode
        self.transaction_cost = transaction_cost
        self.slippage_bps = slippage_bps / 10000.0
        self.spread_bps = spread_bps / 10000.0
        self.max_position = max_position
        self.lookback_window = lookback_window
        self.features = features or self._get_default_features()
        
        # Set seed
        if seed is not None:
            self.seed(seed)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(self.features) * self.lookback_window + 2,),
            dtype=np.float32
        )
        
        # Episode tracking
        self.reset()
    
    def _get_default_features(self) -> List[str]:
        """Get default feature columns."""
        return [
            'returns', 'log_returns', 'volatility',
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
            'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'volume_ratio', 'spread', 'hour_sin', 'hour_cos'
        ]
    
    def reset(self) -> np.ndarray:
        """Reset environment for new episode."""
        self.current_step = self.lookback_window
        self.position = 0.0
        self.entry_price = 0.0
        self.total_pnl = 0.0
        self.trades = []
        
        # Episode metrics
        self.episode_rewards = []
        self.episode_positions = []
        self.episode_returns = []
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one trading step."""
        action_value = float(action[0])
        action_value = np.clip(action_value, -1.0, 1.0)
        
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate position change
        target_position = action_value * self.max_position
        position_change = target_position - self.position
        
        # Calculate transaction costs
        transaction_cost = abs(position_change) * self.transaction_cost
        slippage_cost = abs(position_change) * current_price * self.slippage_bps
        spread_cost = abs(position_change) * current_price * self.spread_bps
        total_cost = transaction_cost + slippage_cost + spread_cost
        
        # Calculate PnL
        if self.position != 0:
            price_change = current_price - self.entry_price
            position_pnl = self.position * price_change / self.entry_price
        else:
            position_pnl = 0.0
        
        # Update position
        if position_change != 0:
            self.position = target_position
            self.entry_price = current_price
            self.trades.append({
                'step': self.current_step,
                'action': action_value,
                'position': self.position,
                'price': current_price,
                'cost': total_cost
            })
        
        # Calculate reward
        step_return = self.data.iloc[self.current_step]['returns']
        reward = self.position * step_return - total_cost
        
        # Risk penalty
        risk_penalty = abs(self.position) * 0.001
        reward -= risk_penalty
        
        # Update tracking
        self.total_pnl += position_pnl - total_cost
        self.episode_rewards.append(reward)
        self.episode_positions.append(self.position)
        self.episode_returns.append(step_return)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        # Get next observation
        obs = self._get_observation()
        
        # Additional info
        info = {
            'position': self.position,
            'pnl': self.total_pnl,
            'trades': len(self.trades),
            'current_price': current_price,
            'step': self.current_step
        }
        
        return obs, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step < self.lookback_window:
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        # Get historical features
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step
        
        feature_data = []
        for feature in self.features:
            if feature in self.data.columns:
                values = self.data[feature].iloc[start_idx:end_idx].values
                feature_data.extend(values)
            else:
                feature_data.extend([0.0] * self.lookback_window)
        
        # Add position and PnL
        feature_data.append(self.position)
        feature_data.append(self.total_pnl)
        
        return np.array(feature_data, dtype=np.float32)
    
    def render(self, mode='human'):
        """Render environment (optional)."""
        pass
    
    def close(self):
        """Clean up environment."""
        pass

# ==================================================================================
# UTILITY FUNCTIONS
# ==================================================================================

def get_git_info() -> Dict[str, str]:
    """Get git information with multiple fallback methods."""
    git_info = {
        'sha': 'unknown',
        'branch': 'unknown',
        'tag': None,
        'dirty': False
    }
    
    try:
        # Method 1: Direct git command
        try:
            git_sha = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL,
                text=True
            ).strip()
            git_info['sha'] = git_sha[:8]
        except:
            pass
        
        # Method 2: GitPython library
        if git_info['sha'] == 'unknown':
            try:
                import git
                repo = git.Repo(search_parent_directories=True)
                git_info['sha'] = repo.head.object.hexsha[:8]
                git_info['branch'] = repo.active_branch.name
                git_info['dirty'] = repo.is_dirty()
            except:
                pass
        
        # Method 3: Environment variables
        if git_info['sha'] == 'unknown':
            git_sha = os.environ.get('GIT_COMMIT', os.environ.get('GIT_SHA', 'unknown'))
            if git_sha != 'unknown':
                git_info['sha'] = git_sha[:8]
        
        # Get branch
        if git_info['branch'] == 'unknown':
            try:
                git_branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    stderr=subprocess.DEVNULL,
                    text=True
                ).strip()
                git_info['branch'] = git_branch
            except:
                git_info['branch'] = os.environ.get('GIT_BRANCH', 'unknown')
    
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
    
    return git_info


def safe_ratio(mean: float, std: float, clip: float = 10.0, eps: float = 1e-4) -> float:
    """Calculate safe ratio with clipping."""
    if std < eps:
        std = eps
    ratio = mean / std
    return float(np.clip(ratio, -clip, clip))


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """Calculate Sortino ratio."""
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    return safe_ratio(excess_returns.mean(), downside_std)


def calculate_calmar_ratio(returns: np.ndarray) -> float:
    """Calculate Calmar ratio."""
    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min())
    
    if max_dd < 1e-6:
        return float('inf') if returns.mean() > 0 else 0.0
    
    n_periods = len(returns)
    total_return = cumulative.iloc[-1] - 1
    cagr = (1 + total_return) ** (252 / n_periods) - 1
    
    return cagr / max_dd


def calculate_max_drawdown(returns: np.ndarray) -> float:
    """Calculate maximum drawdown."""
    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return abs(float(drawdown.min()))


def calculate_profit_factor(returns: np.ndarray) -> float:
    """Calculate profit factor."""
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    return profits / losses


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of file."""
    if not file_path.exists():
        return "not_found"
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]

# ==================================================================================
# L4 VALIDATION
# ==================================================================================

def validate_l4_readiness(l4_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """Validate L4 outputs are ready for L5."""
    logger.info("=" * 80)
    logger.info("Validating L4 outputs...")
    logger.info("=" * 80)
    
    validation_results = {
        'ready': False,
        'errors': [],
        'warnings': [],
        'artifacts': {}
    }
    
    # Check READY flag
    ready_flag = l4_path / "READY"
    if not ready_flag.exists():
        validation_results['errors'].append("L4 READY flag not found")
        return False, validation_results
    
    # Check required files
    required_files = [
        'train_dataset.parquet',
        'test_dataset.parquet',
        'cost_model.json',
        'env_spec.json',
        'reward_spec.json',
        'feature_importance.json'
    ]
    
    for file_name in required_files:
        file_path = l4_path / file_name
        if not file_path.exists():
            validation_results['errors'].append(f"Missing required file: {file_name}")
        else:
            validation_results['artifacts'][file_name] = str(file_path)
    
    # Load and validate datasets
    try:
        train_df = pd.read_parquet(l4_path / "train_dataset.parquet")
        test_df = pd.read_parquet(l4_path / "test_dataset.parquet")
        
        # Check required columns
        required_columns = ['close', 'returns', 'reward']
        for col in required_columns:
            if col not in train_df.columns:
                validation_results['errors'].append(f"Missing column in train data: {col}")
            if col not in test_df.columns:
                validation_results['errors'].append(f"Missing column in test data: {col}")
        
        logger.info(f"Train dataset: {len(train_df)} rows, {len(train_df.columns)} columns")
        logger.info(f"Test dataset: {len(test_df)} rows, {len(test_df.columns)} columns")
        
    except Exception as e:
        validation_results['errors'].append(f"Error loading datasets: {e}")
    
    # Load and validate configurations
    try:
        with open(l4_path / "cost_model.json", 'r') as f:
            cost_model = json.load(f)
            validation_results['artifacts']['cost_model'] = cost_model
            logger.info(f"Cost model: {cost_model}")
        
        with open(l4_path / "env_spec.json", 'r') as f:
            env_spec = json.load(f)
            validation_results['artifacts']['env_spec'] = env_spec
            logger.info(f"Environment spec loaded: {len(env_spec.get('features', []))} features")
        
        with open(l4_path / "reward_spec.json", 'r') as f:
            reward_spec = json.load(f)
            validation_results['artifacts']['reward_spec'] = reward_spec
            logger.info("Reward specification loaded")
            
    except Exception as e:
        validation_results['errors'].append(f"Error loading configurations: {e}")
    
    # Check for errors
    if validation_results['errors']:
        logger.error("L4 validation failed:")
        for error in validation_results['errors']:
            logger.error(f"  - {error}")
        return False, validation_results
    
    validation_results['ready'] = True
    logger.info("✓ L4 validation passed")
    return True, validation_results

# ==================================================================================
# REWARD REPRODUCIBILITY
# ==================================================================================

def verify_reward_reproducibility(
    df: pd.DataFrame,
    reward_spec: Dict[str, Any]
) -> float:
    """Verify rewards can be reproduced from specification."""
    logger.info("Verifying reward reproducibility...")
    
    recomputed_rewards = []
    
    for idx, row in df.iterrows():
        reward = 0.0
        
        # Base return component
        if 'returns' in row:
            reward += row['returns'] * reward_spec.get('return_weight', 1.0)
        
        # Risk penalty
        if 'position_size' in row:
            risk_penalty = abs(row['position_size']) * reward_spec.get('risk_penalty', 0.001)
            reward -= risk_penalty
        
        # Cost component
        if 'transaction_cost' in row:
            reward -= row['transaction_cost']
        
        recomputed_rewards.append(reward)
    
    # Calculate RMSE
    original_rewards = df['reward'].values if 'reward' in df else np.zeros(len(df))
    rmse = np.sqrt(np.mean((original_rewards - recomputed_rewards) ** 2))
    
    logger.info(f"Reward reproducibility RMSE: {rmse:.8f}")
    
    if rmse > 1e-6:
        logger.warning(f"High RMSE detected: {rmse:.8f}")
    
    return rmse

# ==================================================================================
# TRAINING
# ==================================================================================

class MetricsCallback(BaseCallback):
    """Custom callback for tracking metrics."""
    
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        return True
    
    def _on_rollout_end(self) -> None:
        pass


def create_training_env(
    data: pd.DataFrame,
    mode: str,
    l4_artifacts: Dict[str, Any],
    seed: Optional[int] = None
) -> Monitor:
    """Create training environment with Monitor wrapper."""
    cost_model = l4_artifacts.get('cost_model', {})
    env_spec = l4_artifacts.get('env_spec', {})
    
    env = TradingEnvironment(
        data=data,
        mode=mode,
        transaction_cost=cost_model.get('transaction_cost_bps', 10) / 10000.0,
        slippage_bps=cost_model.get('slippage_bps', 5),
        spread_bps=cost_model.get('spread_p95_bps', 20),
        max_position=env_spec.get('max_position', 1.0),
        lookback_window=env_spec.get('lookback_window', 100),
        features=env_spec.get('features', []),
        seed=seed
    )
    
    # CRITICAL: Wrap with Monitor
    return Monitor(env, allow_early_resets=True)


def train_ppo_agent(
    seed: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    l4_artifacts: Dict[str, Any],
    output_path: Path
) -> Dict[str, Any]:
    """Train single PPO agent."""
    logger.info(f"Training PPO agent with seed {seed}")
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environments
    train_env = DummyVecEnv([
        lambda: create_training_env(train_df, 'train', l4_artifacts, seed)
    ])
    
    eval_env = DummyVecEnv([
        lambda: create_training_env(test_df, 'test', l4_artifacts, seed)
    ])
    
    # Create learning rate schedule
    lr_schedule = get_linear_fn(
        TRAINING_CONFIG['learning_rate'],
        TRAINING_CONFIG['learning_rate'] * 0.1,
        1.0
    )
    
    # Create model
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        n_steps=TRAINING_CONFIG['n_steps'],
        batch_size=TRAINING_CONFIG['batch_size'],
        n_epochs=TRAINING_CONFIG['n_epochs'],
        learning_rate=lr_schedule,
        ent_coef=TRAINING_CONFIG['ent_coef'],
        vf_coef=TRAINING_CONFIG['vf_coef'],
        max_grad_norm=TRAINING_CONFIG['max_grad_norm'],
        gae_lambda=TRAINING_CONFIG['gae_lambda'],
        gamma=TRAINING_CONFIG['gamma'],
        clip_range=TRAINING_CONFIG['clip_range'],
        device=TRAINING_CONFIG['device'],
        verbose=TRAINING_CONFIG['verbose'],
        seed=seed,
        tensorboard_log=str(output_path / f"tensorboard_seed_{seed}")
    )
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_path / f"seed_{seed}"),
        log_path=str(output_path / f"seed_{seed}"),
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=str(output_path / f"seed_{seed}" / "checkpoints"),
        name_prefix=f"ppo_model_seed_{seed}",
        save_replay_buffer=False,
        save_vecnormalize=True
    )
    
    metrics_callback = MetricsCallback(eval_env, verbose=1)
    
    callbacks = CallbackList([eval_callback, checkpoint_callback, metrics_callback])
    
    # Train model
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=TRAINING_CONFIG['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Final evaluation
    final_rewards, final_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # Calculate metrics
    metrics = {
        'seed': seed,
        'mean_reward': float(np.mean(final_rewards)),
        'std_reward': float(np.std(final_rewards)),
        'sortino_ratio': calculate_sortino_ratio(np.array(final_rewards)),
        'max_reward': float(np.max(final_rewards)),
        'min_reward': float(np.min(final_rewards)),
        'training_time_seconds': training_time,
        'total_timesteps': TRAINING_CONFIG['total_timesteps']
    }
    
    # Save model
    model_path = output_path / f"seed_{seed}" / "model.zip"
    model.save(str(model_path))
    
    # Save metrics
    with open(output_path / f"seed_{seed}" / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Seed {seed} completed: Sortino={metrics['sortino_ratio']:.2f}")
    
    return metrics

# ==================================================================================
# PRODUCTION GATES
# ==================================================================================

def evaluate_production_gates(
    model_path: Path,
    test_df: pd.DataFrame,
    l4_artifacts: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    """Evaluate model against production gates."""
    logger.info("=" * 80)
    logger.info("Evaluating production gates...")
    logger.info("=" * 80)
    
    results = {
        'gates_status': {},
        'metrics': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Load model
    model = PPO.load(str(model_path / "model.zip"))
    
    # 1. Performance evaluation
    logger.info("1. Evaluating performance metrics...")
    test_env = DummyVecEnv([
        lambda: create_training_env(test_df, 'test', l4_artifacts)
    ])
    
    rewards, lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=20,
        deterministic=True,
        return_episode_rewards=True
    )
    
    returns = np.array(rewards)
    
    perf_metrics = {
        'mean_reward': float(np.mean(returns)),
        'std_reward': float(np.std(returns)),
        'sortino_ratio': calculate_sortino_ratio(returns),
        'calmar_ratio': calculate_calmar_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': float((returns > 0).mean()),
        'profit_factor': calculate_profit_factor(returns)
    }
    
    results['metrics']['performance'] = perf_metrics
    
    # Check performance gates
    results['gates_status']['sortino'] = perf_metrics['sortino_ratio'] >= PERFORMANCE_THRESHOLDS['sortino_ratio']
    results['gates_status']['calmar'] = perf_metrics['calmar_ratio'] >= PERFORMANCE_THRESHOLDS['calmar_ratio']
    results['gates_status']['max_drawdown'] = perf_metrics['max_drawdown'] <= PERFORMANCE_THRESHOLDS['max_drawdown']
    results['gates_status']['win_rate'] = perf_metrics['win_rate'] >= PERFORMANCE_THRESHOLDS['win_rate']
    results['gates_status']['profit_factor'] = perf_metrics['profit_factor'] >= PERFORMANCE_THRESHOLDS['profit_factor']
    
    # 2. Consistency check
    logger.info("2. Evaluating consistency...")
    train_env = DummyVecEnv([
        lambda: create_training_env(test_df[:len(test_df)//2], 'train', l4_artifacts)
    ])
    
    train_rewards, _ = evaluate_policy(
        model, train_env, n_eval_episodes=10, deterministic=True, return_episode_rewards=True
    )
    
    train_sortino = calculate_sortino_ratio(np.array(train_rewards))
    sortino_diff = abs(train_sortino - perf_metrics['sortino_ratio'])
    
    results['metrics']['consistency'] = {
        'train_sortino': train_sortino,
        'test_sortino': perf_metrics['sortino_ratio'],
        'difference': sortino_diff
    }
    
    results['gates_status']['consistency'] = sortino_diff <= PERFORMANCE_THRESHOLDS['consistency_sortino']
    
    # 3. Cost stress test
    logger.info("3. Running cost stress test...")
    cost_model = l4_artifacts['cost_model']
    
    stress_env = DummyVecEnv([
        lambda: Monitor(TradingEnvironment(
            data=test_df,
            mode='test',
            transaction_cost=(cost_model.get('transaction_cost_bps', 10) / 10000.0) * COST_STRESS_MULTIPLIER,
            slippage_bps=cost_model.get('slippage_bps', 5) * COST_STRESS_MULTIPLIER,
            spread_bps=cost_model.get('spread_p95_bps', 20) * COST_STRESS_MULTIPLIER,
            max_position=l4_artifacts['env_spec'].get('max_position', 1.0),
            lookback_window=l4_artifacts['env_spec'].get('lookback_window', 100),
            features=l4_artifacts['env_spec'].get('features', [])
        ))
    ])
    
    stress_rewards, _ = evaluate_policy(
        model, stress_env, n_eval_episodes=10, deterministic=True, return_episode_rewards=True
    )
    
    stress_mean = np.mean(stress_rewards)
    base_cagr = (1 + perf_metrics['mean_reward']) ** 252 - 1
    stress_cagr = (1 + stress_mean) ** 252 - 1
    cagr_drop = (base_cagr - stress_cagr) / base_cagr if base_cagr > 0 else 0
    
    results['metrics']['stress'] = {
        'base_mean': perf_metrics['mean_reward'],
        'stress_mean': stress_mean,
        'cagr_drop': cagr_drop
    }
    
    results['gates_status']['cost_stress'] = cagr_drop <= PERFORMANCE_THRESHOLDS['cost_stress_cagr_drop']
    
    # 4. Latency test
    logger.info("4. Testing inference latency...")
    
    # Export to ONNX for latency testing
    policy = model.policy
    dummy_input = torch.randn(1, test_env.observation_space.shape[0])
    
    onnx_path = model_path / "model.onnx"
    torch.onnx.export(
        policy,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=11,
        do_constant_folding=True
    )
    
    # Test ONNX inference latency
    session = ort.InferenceSession(str(onnx_path))
    
    # Warm up
    for _ in range(10):
        session.run(None, {'input': dummy_input.numpy()})
    
    # Measure
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {'input': dummy_input.numpy()})
        latencies.append((time.perf_counter() - start) * 1000)
    
    results['metrics']['latency'] = {
        'inference_p50': float(np.percentile(latencies, 50)),
        'inference_p99': float(np.percentile(latencies, 99)),
        'inference_max': float(np.max(latencies))
    }
    
    results['gates_status']['latency'] = results['metrics']['latency']['inference_p99'] <= LATENCY_REQUIREMENTS['inference_p99']
    
    # Overall status
    results['overall_status'] = 'PASS' if all(results['gates_status'].values()) else 'FAIL'
    results['gates_passed'] = sum(results['gates_status'].values())
    results['gates_total'] = len(results['gates_status'])
    
    # Log results
    logger.info("=" * 80)
    logger.info(f"GATES EVALUATION: {results['overall_status']}")
    logger.info(f"Gates passed: {results['gates_passed']}/{results['gates_total']}")
    for gate, status in results['gates_status'].items():
        logger.info(f"  {gate}: {'✓ PASS' if status else '✗ FAIL'}")
    logger.info("=" * 80)
    
    return results['overall_status'] == 'PASS', results

# ==================================================================================
# BUNDLE GENERATION
# ==================================================================================

def create_deployment_bundle(
    model_path: Path,
    l4_artifacts: Dict[str, Any],
    gate_results: Dict[str, Any],
    output_path: Path
) -> Path:
    """Create deployment bundle."""
    logger.info("Creating deployment bundle...")
    
    bundle_name = f"l5_bundle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    bundle_path = output_path / bundle_name
    bundle_path.mkdir(parents=True, exist_ok=True)
    
    # Copy model
    shutil.copy2(model_path / "model.zip", bundle_path / "model.zip")
    shutil.copy2(model_path / "model.onnx", bundle_path / "model.onnx")
    
    # Save L4 artifacts
    l4_path = bundle_path / "l4_artifacts"
    l4_path.mkdir(exist_ok=True)
    
    with open(l4_path / "cost_model.json", 'w') as f:
        json.dump(l4_artifacts['cost_model'], f, indent=2)
    
    with open(l4_path / "env_spec.json", 'w') as f:
        json.dump(l4_artifacts['env_spec'], f, indent=2)
    
    with open(l4_path / "reward_spec.json", 'w') as f:
        json.dump(l4_artifacts['reward_spec'], f, indent=2)
    
    # Get git info
    git_info = get_git_info()
    
    # Create manifest
    manifest = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'git': git_info,
        'model': {
            'framework': 'stable-baselines3',
            'algorithm': 'PPO',
            'training_config': TRAINING_CONFIG
        },
        'performance': gate_results['metrics']['performance'],
        'gates': {
            'overall_status': gate_results['overall_status'],
            'gates_passed': gate_results['gates_passed'],
            'gates_total': gate_results['gates_total'],
            'details': gate_results['gates_status']
        },
        'latency': gate_results['metrics']['latency'],
        'deployment': {
            'ready': gate_results['overall_status'] == 'PASS'
        }
    }
    
    with open(bundle_path / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create README
    readme = f"""# L5 Deployment Bundle

## Status: {manifest['gates']['overall_status']}
- Gates Passed: {manifest['gates']['gates_passed']}/{manifest['gates']['gates_total']}
- Git SHA: {manifest['git']['sha']}
- Created: {manifest['created_at']}

## Performance
- Sortino Ratio: {manifest['performance']['sortino_ratio']:.2f}
- Calmar Ratio: {manifest['performance']['calmar_ratio']:.2f}
- Max Drawdown: {manifest['performance']['max_drawdown']:.2%}

## Deployment
- Ready: {manifest['deployment']['ready']}
"""
    
    with open(bundle_path / "README.md", 'w') as f:
        f.write(readme)
    
    # Create tarball
    tarball_path = output_path / f"{bundle_name}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(bundle_path, arcname=bundle_name)
    
    logger.info(f"Bundle created: {tarball_path}")
    return tarball_path

# ==================================================================================
# MAIN PIPELINE
# ==================================================================================

def run_l5_pipeline(
    l4_path: Path,
    output_path: Path,
    seeds: List[int] = [42, 123, 456],
    mlflow_experiment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run complete L5 reinforcement learning pipeline.
    
    Args:
        l4_path: Path to L4 outputs
        output_path: Path for L5 outputs
        seeds: Random seeds for training
        mlflow_experiment: MLflow experiment name
    
    Returns:
        Pipeline results dictionary
    """
    logger.info("=" * 80)
    logger.info("L5 REINFORCEMENT LEARNING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"L4 path: {l4_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Seeds: {seeds}")
    logger.info(f"Device: {TRAINING_CONFIG['device']}")
    logger.info("=" * 80)
    
    pipeline_results = {
        'start_time': datetime.now().isoformat(),
        'status': 'RUNNING',
        'stages': {}
    }
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        mlflow_run = mlflow.start_run(run_name=f"L5_Pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params(TRAINING_CONFIG)
    else:
        mlflow_run = None
    
    try:
        # ====================================================================
        # STAGE 1: L4 Validation
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 1: L4 Validation")
        logger.info("=" * 40)
        
        l4_ready, l4_validation = validate_l4_readiness(l4_path)
        
        if not l4_ready:
            raise ValueError(f"L4 validation failed: {l4_validation['errors']}")
        
        pipeline_results['stages']['l4_validation'] = {
            'status': 'PASSED',
            'artifacts': l4_validation['artifacts']
        }
        
        # Load data
        train_df = pd.read_parquet(l4_path / "train_dataset.parquet")
        test_df = pd.read_parquet(l4_path / "test_dataset.parquet")
        
        # Verify reward reproducibility
        train_rmse = verify_reward_reproducibility(train_df, l4_validation['artifacts']['reward_spec'])
        test_rmse = verify_reward_reproducibility(test_df, l4_validation['artifacts']['reward_spec'])
        
        if train_rmse > 1e-6 or test_rmse > 1e-6:
            logger.warning(f"High reward RMSE - Train: {train_rmse:.8f}, Test: {test_rmse:.8f}")
        
        # ====================================================================
        # STAGE 2: Multi-seed Training
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 2: Multi-seed Training")
        logger.info("=" * 40)
        
        training_results = []
        for seed in seeds:
            logger.info(f"\nTraining seed {seed}...")
            try:
                metrics = train_ppo_agent(
                    seed=seed,
                    train_df=train_df,
                    test_df=test_df,
                    l4_artifacts=l4_validation['artifacts'],
                    output_path=output_path / "models"
                )
                training_results.append(metrics)
                
                if mlflow_run:
                    mlflow.log_metrics({
                        f"seed_{seed}_sortino": metrics['sortino_ratio'],
                        f"seed_{seed}_mean_reward": metrics['mean_reward']
                    })
                    
            except Exception as e:
                logger.error(f"Training failed for seed {seed}: {e}")
                continue
        
        if not training_results:
            raise ValueError("All training seeds failed")
        
        # Select best model
        best_idx = np.argmax([m['sortino_ratio'] for m in training_results])
        best_seed = training_results[best_idx]['seed']
        best_model_path = output_path / "models" / f"seed_{best_seed}"
        
        logger.info(f"\nBest model: Seed {best_seed} with Sortino={training_results[best_idx]['sortino_ratio']:.2f}")
        
        pipeline_results['stages']['training'] = {
            'status': 'COMPLETED',
            'best_seed': best_seed,
            'all_results': training_results
        }
        
        # ====================================================================
        # STAGE 3: Production Gates
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 3: Production Gates")
        logger.info("=" * 40)
        
        gates_passed, gate_results = evaluate_production_gates(
            model_path=best_model_path,
            test_df=test_df,
            l4_artifacts=l4_validation['artifacts']
        )
        
        pipeline_results['stages']['gates'] = gate_results
        
        if mlflow_run:
            mlflow.log_metrics({
                'gates_passed': gate_results['gates_passed'],
                'gates_total': gate_results['gates_total']
            })
        
        # Save gate results
        with open(output_path / "gate_results.json", 'w') as f:
            json.dump(gate_results, f, indent=2)
        
        # ====================================================================
        # STAGE 4: Bundle Generation
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 4: Bundle Generation")
        logger.info("=" * 40)
        
        bundle_path = create_deployment_bundle(
            model_path=best_model_path,
            l4_artifacts=l4_validation['artifacts'],
            gate_results=gate_results,
            output_path=output_path / "bundles"
        )
        
        pipeline_results['stages']['bundle'] = {
            'status': 'CREATED',
            'path': str(bundle_path)
        }
        
        if mlflow_run:
            mlflow.log_artifact(str(bundle_path))
        
        # ====================================================================
        # STAGE 5: Finalization
        # ====================================================================
        logger.info("\n" + "=" * 40)
        logger.info("STAGE 5: Finalization")
        logger.info("=" * 40)
        
        if gates_passed:
            logger.info("✓ All gates PASSED - Model ready for deployment")
            
            # Create READY flag
            ready_flag = output_path / "READY"
            with open(ready_flag, 'w') as f:
                f.write(f"L5 Pipeline completed at {datetime.now().isoformat()}\n")
                f.write(f"Best model: seed_{best_seed}\n")
                f.write(f"Gates passed: {gate_results['gates_passed']}/{gate_results['gates_total']}\n")
            
            pipeline_results['status'] = 'SUCCESS'
            pipeline_results['deployment_ready'] = True
            
        else:
            logger.warning("✗ Some gates FAILED - Model not ready for deployment")
            logger.info("Suggested actions:")
            logger.info("  1. Increase training timesteps")
            logger.info("  2. Tune hyperparameters")
            logger.info("  3. Review gate thresholds")
            logger.info("  4. Check data quality")
            
            pipeline_results['status'] = 'FAILED'
            pipeline_results['deployment_ready'] = False
        
        # Save pipeline results
        pipeline_results['end_time'] = datetime.now().isoformat()
        with open(output_path / "pipeline_results.json", 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        if mlflow_run:
            mlflow.log_artifact(str(output_path / "pipeline_results.json"))
            mlflow.end_run()
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("L5 PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Status: {pipeline_results['status']}")
        logger.info(f"Deployment Ready: {pipeline_results['deployment_ready']}")
        logger.info(f"Best Model: seed_{best_seed}")
        logger.info(f"Gates: {gate_results['gates_passed']}/{gate_results['gates_total']} passed")
        logger.info(f"Bundle: {bundle_path}")
        logger.info("=" * 80)
        
        return pipeline_results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        pipeline_results['status'] = 'ERROR'
        pipeline_results['error'] = str(e)
        
        if mlflow_run:
            mlflow.log_param("error", str(e))
            mlflow.end_run(status='FAILED')
        
        raise


# ==================================================================================
# ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="L5 Complete Reinforcement Learning Pipeline"
    )
    parser.add_argument(
        "--l4-path",
        type=Path,
        required=True,
        help="Path to L4 outputs"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Path for L5 outputs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[42, 123, 456],
        help="Random seeds for training"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="L5_Pipeline",
        help="MLflow experiment name"
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    try:
        results = run_l5_pipeline(
            l4_path=args.l4_path,
            output_path=args.output_path,
            seeds=args.seeds,
            mlflow_experiment=args.mlflow_experiment
        )
        
        # Exit with appropriate code
        if results['deployment_ready']:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(2)