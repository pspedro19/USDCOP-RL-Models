#!/usr/bin/env python
"""
PPOV1 Simple Training Script
============================
Entrenamiento simple basado en PRODUCTION_CONFIG.json
Genera modelos .zip y metricas JSON.
"""

import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# =============================================================================
# CONFIGURATION (from PRODUCTION_CONFIG.json)
# =============================================================================

DATA_PATH = Path(__file__).parent.parent.parent / "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv"
TOTAL_TIMESTEPS = 80_000
SEED = 42

PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.05,
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
}

ENV_CONFIG = {
    "initial_balance": 10_000,
    "max_position": 1.0,
    "episode_length": 400,
    "max_drawdown_pct": 15.0,
    "use_vol_scaling": True,
    "use_regime_detection": True,
    "bars_per_day": 56,
}


def main():
    print("=" * 70)
    print("PPOV1 - SIMPLE TRAINING SCRIPT")
    print("=" * 70)

    # 1. Load Data
    print(f"\n[1/5] Loading data...")
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
    df = df.rename(columns={'timestamp': 'datetime'})
    df = df.sort_values('datetime').reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows")

    # 2. Preprocess
    print("\n[2/5] Preprocessing...")
    from preprocessing import DataPreprocessor, DataQualityConfig

    config = DataQualityConfig()
    preprocessor = DataPreprocessor(config)
    df_clean = preprocessor.fit_transform(df)

    # Train/Test split (80/20)
    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split_idx].reset_index(drop=True)
    df_test = df_clean.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    # 3. Create Environment
    print("\n[3/5] Creating TradingEnvironmentV19...")
    from environment_v19 import TradingEnvironmentV19
    from rewards import SymmetricCurriculumReward

    reward_fn = SymmetricCurriculumReward(
        total_timesteps=TOTAL_TIMESTEPS,
        config=None
    )

    env = TradingEnvironmentV19(
        df=df_train,
        initial_balance=ENV_CONFIG["initial_balance"],
        max_position=ENV_CONFIG["max_position"],
        episode_length=ENV_CONFIG["episode_length"],
        max_drawdown_pct=ENV_CONFIG["max_drawdown_pct"],
        use_vol_scaling=ENV_CONFIG["use_vol_scaling"],
        use_regime_detection=ENV_CONFIG["use_regime_detection"],
        reward_function=reward_fn,
    )

    print(f"  Environment: {type(env).__name__}")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space}")
    print(f"  Episode length: {ENV_CONFIG['episode_length']}")

    # 4. Train PPO
    print("\n[4/5] Training PPO...")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env():
        r = SymmetricCurriculumReward(total_timesteps=TOTAL_TIMESTEPS)
        e = TradingEnvironmentV19(
            df=df_train,
            initial_balance=ENV_CONFIG["initial_balance"],
            max_position=ENV_CONFIG["max_position"],
            episode_length=ENV_CONFIG["episode_length"],
            max_drawdown_pct=ENV_CONFIG["max_drawdown_pct"],
            use_vol_scaling=ENV_CONFIG["use_vol_scaling"],
            use_regime_detection=ENV_CONFIG["use_regime_detection"],
            reward_function=r,
        )
        return Monitor(e)

    vec_env = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        vec_env,
        **PPO_CONFIG,
        policy_kwargs={"net_arch": [256, 256]},
        seed=SEED,
        verbose=1,
    )

    print(f"\n  Training {TOTAL_TIMESTEPS:,} timesteps...")
    print("-" * 70)

    start_time = datetime.now()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    train_time = (datetime.now() - start_time).total_seconds()

    print("-" * 70)
    print(f"  Completed in {train_time:.1f}s")

    # 5. Evaluate
    print("\n[5/5] Evaluating on test set (10 episodes)...")

    test_reward_fn = SymmetricCurriculumReward(total_timesteps=TOTAL_TIMESTEPS)
    test_env = TradingEnvironmentV19(
        df=df_test,
        initial_balance=ENV_CONFIG["initial_balance"],
        max_position=ENV_CONFIG["max_position"],
        episode_length=ENV_CONFIG["episode_length"],
        max_drawdown_pct=ENV_CONFIG["max_drawdown_pct"],
        use_vol_scaling=ENV_CONFIG["use_vol_scaling"],
        use_regime_detection=ENV_CONFIG["use_regime_detection"],
        reward_function=test_reward_fn,
    )

    n_eval_episodes = 10
    all_rewards = []
    all_actions = []
    all_returns = []

    for ep in range(n_eval_episodes):
        obs, _ = test_env.reset()
        episode_reward = 0
        episode_actions = []
        episode_returns = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_actions.append(action[0])

            # Get step return
            if 'step_return' in info:
                episode_returns.append(info['step_return'])

        all_rewards.append(episode_reward)
        all_actions.extend(episode_actions)
        all_returns.extend(episode_returns)

    # Calculate metrics
    actions = np.array(all_actions)
    returns = np.array(all_returns) if all_returns else np.array([0])

    # Action distribution (continuous: >0.1=LONG, <-0.1=SHORT, else=HOLD)
    long_pct = np.mean(actions > 0.1) * 100
    short_pct = np.mean(actions < -0.1) * 100
    hold_pct = np.mean(np.abs(actions) <= 0.1) * 100

    # Sharpe (annualized)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 56)
    else:
        sharpe = 0

    # Max DD
    cumret = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumret)
    drawdowns = running_max - cumret
    max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    # Win rate
    win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0

    # Mean PnL
    mean_pnl = np.mean(returns) * 100 if len(returns) > 0 else 0

    print(f"\n{'='*70}")
    print("RESULTS - PPOV1")
    print(f"{'='*70}")
    print(f"  Episodes evaluated: {n_eval_episodes}")
    print(f"  Mean Reward:     {np.mean(all_rewards):.2f}")
    print(f"  Sharpe Ratio:    {sharpe:.2f}")
    print(f"  Max Drawdown:    {max_dd*100:.2f}%")
    print(f"  Win Rate:        {win_rate:.1f}%")
    print(f"  Mean PnL/step:   {mean_pnl:.4f}%")
    print(f"  Actions: LONG={long_pct:.1f}% HOLD={hold_pct:.1f}% SHORT={short_pct:.1f}%")

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / f"ppo_v1_{timestamp}.zip"
    model.save(str(model_path))
    print(f"\n  Model saved: {model_path.name}")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "PPOV1",
        "total_timesteps": TOTAL_TIMESTEPS,
        "train_time_seconds": train_time,
        "n_eval_episodes": n_eval_episodes,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "mean_pnl_pct": float(mean_pnl),
        "mean_reward": float(np.mean(all_rewards)),
        "action_distribution": {
            "long": float(long_pct),
            "hold": float(hold_pct),
            "short": float(short_pct),
        },
        "ppo_config": PPO_CONFIG,
        "env_config": ENV_CONFIG,
    }

    results_path = output_dir / f"results_v1_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {results_path.name}")

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
