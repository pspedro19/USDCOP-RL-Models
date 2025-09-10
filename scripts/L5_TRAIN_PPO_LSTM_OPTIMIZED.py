#!/usr/bin/env python3
"""
L5 PPO-LSTM Training with Optimized Hyperparameters
Based on best practices from academic research for USD/COP trading
"""

import os
import sys
import json
import time
import logging
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback, CallbackList
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed

# Optional: Use sb3-contrib if available
try:
    from sb3_contrib import RecurrentPPO
    HAS_RECURRENT_PPO = True
except ImportError:
    HAS_RECURRENT_PPO = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LSTM FEATURE EXTRACTOR
# ============================================================================

class LSTMExtractor(BaseFeaturesExtractor):
    """
    LSTM feature extractor for temporal dependencies in trading
    """
    def __init__(self, observation_space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        n_input = observation_space.shape[0]
        
        # 2-layer LSTM for better temporal modeling
        self.lstm = nn.LSTM(
            input_size=n_input,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.1  # Regularization
        )
        
        # Projection layer
        self.linear = nn.Sequential(
            nn.Linear(256, features_dim),
            nn.ReLU(),
            nn.LayerNorm(features_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Add time dimension if needed
        if len(observations.shape) == 2:
            observations = observations.unsqueeze(1)
        
        lstm_out, _ = self.lstm(observations)
        
        # Take last timestep output
        if lstm_out.shape[1] > 1:
            lstm_out = lstm_out[:, -1, :]
        else:
            lstm_out = lstm_out.squeeze(1)
            
        return self.linear(lstm_out)

# ============================================================================
# CURRICULUM LEARNING FOR COSTS
# ============================================================================

class CurriculumCostWrapper:
    """
    Gradually introduce trading costs to help policy learn direction first
    """
    def __init__(self, env, initial_factor=0.0, final_factor=1.0, 
                 warmup_steps=50000, schedule='linear'):
        self.env = env
        self.initial_factor = initial_factor
        self.final_factor = final_factor
        self.warmup_steps = warmup_steps
        self.schedule = schedule
        self.current_step = 0
        
    def get_cost_factor(self):
        if self.current_step >= self.warmup_steps:
            return self.final_factor
            
        progress = self.current_step / self.warmup_steps
        
        if self.schedule == 'linear':
            return self.initial_factor + (self.final_factor - self.initial_factor) * progress
        elif self.schedule == 'cosine':
            cosine_factor = 0.5 * (1 + np.cos(np.pi * (1 - progress)))
            return self.initial_factor + (self.final_factor - self.initial_factor) * (1 - cosine_factor)
        else:
            return self.initial_factor
            
    def step(self, action):
        # Store original cost model
        original_cost_model = getattr(self.env, 'cost_model', None)
        
        # Apply curriculum factor
        factor = self.get_cost_factor()
        if original_cost_model and factor < 1.0:
            scaled_cost_model = {
                k: v * factor if isinstance(v, (int, float)) else v
                for k, v in original_cost_model.items()
            }
            self.env.cost_model = scaled_cost_model
            
        # Step environment
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1
        
        # Restore original cost model
        if original_cost_model:
            self.env.cost_model = original_cost_model
            
        # Add curriculum info
        if info:
            info['curriculum_cost_factor'] = factor
            
        return obs, reward, done, info

# ============================================================================
# OPTIMIZED PPO CONFIGURATIONS
# ============================================================================

def get_ppo_lstm_config() -> Dict[str, Any]:
    """
    PPO + LSTM configuration based on best practices
    Critical: Adam betas (0.99, 0.99) prevent policy collapse
    """
    return {
        # Core PPO parameters
        'learning_rate': 3e-4,
        'n_steps': 480,  # ~8 hours of 5-min bars context
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.995,  # For long horizons
        'gae_lambda': 0.95,
        'clip_range': 0.1,  # Reduced for stability
        'clip_range_vf': None,
        
        # Critical for preventing collapse
        'ent_coef': 0.03,  # HIGH - prevents policy collapse
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        
        # Optimizer settings - CRITICAL
        'optimizer_class': torch.optim.Adam,
        'optimizer_kwargs': {
            'betas': (0.99, 0.99),  # CRITICAL - prevents collapse
            'eps': 1e-5,
            'weight_decay': 0.001
        },
        
        # Policy network architecture
        'policy_kwargs': {
            'features_extractor_class': LSTMExtractor,
            'features_extractor_kwargs': {'features_dim': 512},
            'net_arch': {
                'pi': [256, 128],  # Policy network
                'vf': [256, 128]   # Value network
            },
            'activation_fn': nn.ReLU,
            'ortho_init': True,
            'normalize_images': False
        },
        
        # Training settings
        'tensorboard_log': './tensorboard_logs/',
        'verbose': 1,
        'seed': None,  # Set per seed
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

def get_ppo_mlp_baseline_config() -> Dict[str, Any]:
    """
    Standard PPO-MLP for baseline comparison
    """
    return {
        'learning_rate': 3e-4,
        'n_steps': 480,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.995,
        'gae_lambda': 0.95,
        'clip_range': 0.1,
        'ent_coef': 0.02,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'optimizer_kwargs': {
            'betas': (0.99, 0.99),
            'eps': 1e-5,
            'weight_decay': 0.001
        },
        'policy_kwargs': {
            'net_arch': {
                'pi': [512, 256, 128],
                'vf': [512, 256, 128]
            },
            'activation_fn': nn.ReLU,
            'ortho_init': True
        },
        'verbose': 1,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

# ============================================================================
# QR-DQN CONFIGURATION (OPTIONAL)
# ============================================================================

def get_qrdqn_config() -> Dict[str, Any]:
    """
    Quantile Regression DQN for risk-aware trading
    """
    try:
        from sb3_contrib import QRDQN
        return {
            'learning_rate': 1e-4,
            'buffer_size': 100_000,
            'learning_starts': 10_000,
            'batch_size': 32,
            'tau': 0.005,
            'gamma': 0.995,
            'train_freq': (4, "step"),
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.2,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
            'n_quantiles': 51,  # Full distribution
            'policy_kwargs': {
                'net_arch': [512, 256, 128]
            },
            'verbose': 1,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    except ImportError:
        logger.warning("QR-DQN not available. Install sb3-contrib for this model.")
        return None

# ============================================================================
# TRAINING CALLBACKS
# ============================================================================

class TradingMetricsCallback(BaseCallback):
    """
    Track trading-specific metrics during training
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.action_counts = {0: 0, 1: 0, 2: 0}
        
    def _on_step(self) -> bool:
        # Track actions
        if 'actions' in self.locals:
            actions = self.locals['actions']
            for action in actions.flatten():
                self.action_counts[int(action)] = self.action_counts.get(int(action), 0) + 1
                
        # Log episode metrics
        if 'dones' in self.locals and self.locals['dones'][0]:
            if 'infos' in self.locals:
                info = self.locals['infos'][0]
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                    
                    # Log to tensorboard
                    self.logger.record('rollout/ep_reward', info['episode']['r'])
                    self.logger.record('rollout/ep_length', info['episode']['l'])
                    
        # Log action distribution every 10k steps
        if self.num_timesteps % 10000 == 0:
            total_actions = sum(self.action_counts.values())
            if total_actions > 0:
                for action, count in self.action_counts.items():
                    ratio = count / total_actions
                    action_name = ['sell', 'hold', 'buy'][action]
                    self.logger.record(f'actions/{action_name}_ratio', ratio)
                    
        return True

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_ppo_optimized(
    train_env,
    test_env,
    seed: int = 42,
    total_timesteps: int = 1_000_000,
    model_type: str = 'ppo_lstm',
    use_curriculum: bool = True,
    save_path: Optional[str] = None
) -> Tuple[Any, Dict[str, float]]:
    """
    Train PPO model with optimized configuration
    
    Args:
        train_env: Training environment
        test_env: Test environment for evaluation
        seed: Random seed
        total_timesteps: Total training steps (minimum 500k, recommended 1M)
        model_type: 'ppo_lstm', 'ppo_mlp', or 'qrdqn'
        use_curriculum: Whether to use curriculum learning for costs
        save_path: Where to save the model
        
    Returns:
        Trained model and metrics dictionary
    """
    
    # Set seeds
    set_random_seed(seed)
    
    # Wrap environment with curriculum if requested
    if use_curriculum:
        logger.info("Applying curriculum learning for trading costs")
        train_env = CurriculumCostWrapper(
            train_env,
            initial_factor=0.0,
            final_factor=1.0,
            warmup_steps=int(total_timesteps * 0.05),  # 5% warmup
            schedule='cosine'
        )
    
    # Wrap in vectorized environment
    train_env = DummyVecEnv([lambda: Monitor(train_env)])
    test_env = DummyVecEnv([lambda: Monitor(test_env)])
    
    # Select configuration
    if model_type == 'ppo_lstm':
        config = get_ppo_lstm_config()
        model = PPO('MlpPolicy', train_env, **config)
        logger.info("Training PPO with LSTM feature extractor")
        
    elif model_type == 'ppo_mlp':
        config = get_ppo_mlp_baseline_config()
        model = PPO('MlpPolicy', train_env, **config)
        logger.info("Training baseline PPO-MLP")
        
    elif model_type == 'qrdqn':
        config = get_qrdqn_config()
        if config is None:
            raise ValueError("QR-DQN not available. Install sb3-contrib.")
        from sb3_contrib import QRDQN
        model = QRDQN('MlpPolicy', train_env, **config)
        logger.info("Training QR-DQN for risk-aware trading")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    if save_path:
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=save_path,
            name_prefix=f"{model_type}_seed_{seed}"
        )
        callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        test_env,
        best_model_save_path=save_path if save_path else './best_model/',
        log_path=save_path if save_path else './logs/',
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    # Trading metrics callback
    metrics_callback = TradingMetricsCallback()
    callbacks.append(metrics_callback)
    
    # Combine callbacks
    callback = CallbackList(callbacks)
    
    # Train model
    logger.info(f"Starting training for {total_timesteps} timesteps")
    start_time = time.time()
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=100,
        tb_log_name=f"{model_type}_seed_{seed}",
        reset_num_timesteps=True,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate final performance
    logger.info("Evaluating final model performance...")
    rewards = []
    lengths = []
    
    for _ in range(100):  # 100 test episodes
        obs = test_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    # Calculate metrics
    metrics = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'mean_length': np.mean(lengths),
        'training_time': training_time,
        'total_timesteps': total_timesteps,
        'model_type': model_type,
        'seed': seed
    }
    
    # Calculate Sortino ratio (simplified)
    if len(rewards) > 1:
        returns = np.diff(rewards)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns)
            if downside_std > 0:
                metrics['sortino'] = np.mean(returns) / downside_std
            else:
                metrics['sortino'] = np.inf if np.mean(returns) > 0 else 0
        else:
            metrics['sortino'] = np.inf if np.mean(returns) > 0 else 0
    
    logger.info(f"Final metrics: {metrics}")
    
    # Save final model
    if save_path:
        final_path = os.path.join(save_path, f"{model_type}_seed_{seed}_final.zip")
        model.save(final_path)
        logger.info(f"Model saved to {final_path}")
    
    return model, metrics

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # This would be called from your Airflow DAG
    # Example usage:
    
    # Import your environment creation function
    # from utils.gymnasium_trading_env import create_gym_env
    
    # Create environments
    # train_env = create_gym_env(mode="train")
    # test_env = create_gym_env(mode="test")
    
    # Train models with different seeds and configurations
    seeds = [42, 123, 456]
    
    for seed in seeds:
        logger.info(f"Training with seed {seed}")
        
        # Train PPO-LSTM (main model)
        # model, metrics = train_ppo_optimized(
        #     train_env=train_env,
        #     test_env=test_env,
        #     seed=seed,
        #     total_timesteps=1_000_000,  # 1M steps
        #     model_type='ppo_lstm',
        #     use_curriculum=True,
        #     save_path=f'./models/seed_{seed}/'
        # )
        
        logger.info(f"Seed {seed} completed with Sortino: {metrics.get('sortino', 0):.3f}")
    
    logger.info("All training completed!")