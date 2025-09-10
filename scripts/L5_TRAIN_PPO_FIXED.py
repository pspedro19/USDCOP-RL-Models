"""
L5 PPO Training with Monitor Wrapper - VERSIÓN CORREGIDA
========================================================
Fixes:
1. Monitor wrapper for proper episode tracking
2. Higher n_steps and batch_size for better learning
3. Learning rate schedule
4. Entropy coefficient for exploration
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime

import mlflow
import mlflow.pytorch
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Training configuration
TRAINING_CONFIG = {
    'total_timesteps': 1_000_000,  # 1M steps per seed
    'n_steps': 2048,               # Increased from default
    'batch_size': 256,             # Increased from default
    'n_epochs': 10,
    'learning_rate': 3e-4,         # Initial LR for schedule
    'ent_coef': 0.01,             # Entropy for exploration
    'vf_coef': 0.5,
    'max_grad_norm': 0.5,
    'gae_lambda': 0.95,
    'gamma': 0.99,
    'clip_range': 0.2,
    'device': 'cuda' if os.environ.get('USE_GPU', 'false').lower() == 'true' else 'cpu',
    'verbose': 1,
    'seed': None,  # Will be set per run
}

# Evaluation configuration
EVAL_CONFIG = {
    'n_eval_episodes': 10,
    'eval_freq': 10000,
    'deterministic': True,
}


class MetricsCallback(BaseCallback):
    """
    Custom callback to track and log additional metrics.
    """
    def __init__(self, eval_env, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Log training metrics periodically
        if self.n_calls % 1000 == 0:
            if hasattr(self.training_env, 'get_attr'):
                try:
                    episode_rewards = self.training_env.get_attr('episode_rewards')
                    if episode_rewards and len(episode_rewards[0]) > 0:
                        mean_reward = np.mean([np.mean(r[-100:]) for r in episode_rewards])
                        self.logger.record('train/mean_reward_100ep', mean_reward)
                except:
                    pass
        return True
    
    def _on_rollout_end(self) -> None:
        # Log PPO specific metrics
        if hasattr(self.model, 'logger'):
            # These are automatically logged by SB3, but we can add custom ones
            pass


def create_trading_env(
    mode: str,
    data_path: Path,
    l4_artifacts: Dict[str, Any],
    seed: Optional[int] = None
):
    """
    Create trading environment with Monitor wrapper.
    """
    from your_env_module import TradingEnv  # Replace with actual import
    
    cost_model = l4_artifacts.get('cost_model', {})
    env_spec = l4_artifacts.get('env_spec', {})
    
    # Load data
    if mode == 'train':
        data = pd.read_parquet(data_path / "train_dataset.parquet")
    else:
        data = pd.read_parquet(data_path / "test_dataset.parquet")
    
    # Create base environment
    env = TradingEnv(
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
    
    # CRITICAL: Wrap with Monitor for proper episode tracking
    return Monitor(env, allow_early_resets=True)


def train_ppo_agent(
    seed: int,
    l4_path: Path,
    output_path: Path,
    mlflow_run_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a single PPO agent with given seed.
    """
    logger.info(f"Starting PPO training with seed {seed}")
    
    # Set seeds
    np.random.seed(seed)
    
    # Load L4 artifacts
    with open(l4_path / "cost_model.json", 'r') as f:
        cost_model = json.load(f)
    with open(l4_path / "env_spec.json", 'r') as f:
        env_spec = json.load(f)
    with open(l4_path / "reward_spec.json", 'r') as f:
        reward_spec = json.load(f)
    
    l4_artifacts = {
        'cost_model': cost_model,
        'env_spec': env_spec,
        'reward_spec': reward_spec
    }
    
    # Create environments WITH Monitor wrapper
    logger.info("Creating training environment with Monitor wrapper...")
    train_env = DummyVecEnv([
        lambda: create_trading_env('train', l4_path, l4_artifacts, seed)
    ])
    
    logger.info("Creating evaluation environment with Monitor wrapper...")
    eval_env = DummyVecEnv([
        lambda: create_trading_env('test', l4_path, l4_artifacts, seed)
    ])
    
    # Create model with learning rate schedule
    lr_schedule = get_linear_fn(
        TRAINING_CONFIG['learning_rate'],
        TRAINING_CONFIG['learning_rate'] * 0.1,  # End at 10% of initial
        1.0  # Fraction of training
    )
    
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
        eval_freq=EVAL_CONFIG['eval_freq'],
        n_eval_episodes=EVAL_CONFIG['n_eval_episodes'],
        deterministic=EVAL_CONFIG['deterministic'],
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
    logger.info(f"Training for {TRAINING_CONFIG['total_timesteps']} timesteps...")
    start_time = datetime.now()
    
    model.learn(
        total_timesteps=TRAINING_CONFIG['total_timesteps'],
        callback=callbacks,
        progress_bar=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_rewards, final_lengths = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=20,
        deterministic=True,
        return_episode_rewards=True
    )
    
    # Calculate final metrics
    final_metrics = {
        'seed': seed,
        'mean_reward': float(np.mean(final_rewards)),
        'std_reward': float(np.std(final_rewards)),
        'sortino_ratio': calculate_sortino_ratio(np.array(final_rewards)),
        'max_reward': float(np.max(final_rewards)),
        'min_reward': float(np.min(final_rewards)),
        'training_time_seconds': training_time,
        'total_timesteps': TRAINING_CONFIG['total_timesteps'],
        'final_learning_rate': model.learning_rate if not callable(model.learning_rate) else model.learning_rate(1.0),
    }
    
    # Save model
    model_path = output_path / f"seed_{seed}" / "model.zip"
    model.save(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = output_path / f"seed_{seed}" / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    # Log to MLflow if available
    if mlflow_run_id:
        with mlflow.start_run(run_id=mlflow_run_id, nested=True):
            mlflow.log_metrics({
                f"seed_{seed}_mean_reward": final_metrics['mean_reward'],
                f"seed_{seed}_sortino": final_metrics['sortino_ratio'],
                f"seed_{seed}_training_time": final_metrics['training_time_seconds'],
            })
            mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(metrics_path))
    
    return final_metrics


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sortino ratio for downside risk assessment.
    """
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    downside_std = np.sqrt(np.mean(downside_returns ** 2))
    
    if downside_std < 1e-6:
        return float('inf') if excess_returns.mean() > 0 else 0.0
    
    return float(excess_returns.mean() / downside_std)


def train_multi_seed(
    seeds: list,
    l4_path: Path,
    output_path: Path,
    mlflow_experiment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train multiple PPO agents with different seeds.
    """
    logger.info(f"Starting multi-seed training with seeds: {seeds}")
    
    # Setup MLflow
    if mlflow_experiment:
        mlflow.set_experiment(mlflow_experiment)
        run = mlflow.start_run()
        mlflow_run_id = run.info.run_id
    else:
        mlflow_run_id = None
    
    # Train each seed
    all_metrics = []
    for seed in seeds:
        try:
            metrics = train_ppo_agent(seed, l4_path, output_path, mlflow_run_id)
            all_metrics.append(metrics)
            logger.info(f"Seed {seed} completed: mean_reward={metrics['mean_reward']:.2f}, sortino={metrics['sortino_ratio']:.2f}")
        except Exception as e:
            logger.error(f"Training failed for seed {seed}: {e}")
            continue
    
    # Find best seed
    if all_metrics:
        best_seed_idx = np.argmax([m['sortino_ratio'] for m in all_metrics])
        best_seed = all_metrics[best_seed_idx]['seed']
        
        # Create symlink to best model
        best_model_link = output_path / "best_model"
        best_model_source = output_path / f"seed_{best_seed}"
        
        if best_model_link.exists():
            best_model_link.unlink()
        best_model_link.symlink_to(best_model_source)
        
        logger.info(f"Best seed: {best_seed} with Sortino ratio: {all_metrics[best_seed_idx]['sortino_ratio']:.2f}")
        
        # Save summary
        summary = {
            'seeds': seeds,
            'best_seed': best_seed,
            'all_metrics': all_metrics,
            'training_config': TRAINING_CONFIG,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_path / "training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Log to MLflow
        if mlflow_run_id:
            mlflow.log_metric("best_seed", best_seed)
            mlflow.log_metric("best_sortino", all_metrics[best_seed_idx]['sortino_ratio'])
            mlflow.log_artifact(str(output_path / "training_summary.json"))
            mlflow.end_run()
    
    return summary if all_metrics else {}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="L5 PPO Multi-seed Training")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument("--l4-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--mlflow-experiment", type=str, default="L5_PPO_Training")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_path.mkdir(parents=True, exist_ok=True)
    
    # Run training
    summary = train_multi_seed(
        seeds=args.seeds,
        l4_path=args.l4_path,
        output_path=args.output_path,
        mlflow_experiment=args.mlflow_experiment
    )
    
    logger.info("Training complete!")
    logger.info(json.dumps(summary, indent=2))