#!/usr/bin/env python
"""
Multi-Seed Ensemble Training Script
====================================
Entrena múltiples modelos con diferentes seeds para obtener
métricas robustas con intervalos de confianza.

Beneficios:
- Reduce varianza del entrenamiento RL
- Reporta mean ± std en lugar de un solo run
- Ensemble para producción con mayor estabilidad
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
# CONFIGURATION
# =============================================================================

DATA_PATH = Path(__file__).parent.parent.parent / "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv"

# Ensemble configuration
N_SEEDS = 5  # Número de modelos en el ensemble
BASE_SEED = 42
TIMESTEPS_PER_MODEL = 50_000  # 50K por modelo = 250K total
N_EVAL_EPISODES = 10

# PPO Configuration (same as single model)
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
    "policy_kwargs": {"net_arch": [256, 256]},
}

ENV_CONFIG = {
    "initial_balance": 10_000,
    "max_position": 1.0,
    "episode_length": 400,
    "max_drawdown_pct": 15.0,
    "use_vol_scaling": True,
    "use_regime_detection": True,
}


def main():
    print("=" * 70)
    print("USD/COP RL - MULTI-SEED ENSEMBLE TRAINING")
    print("=" * 70)
    print(f"  Seeds: {N_SEEDS} (base={BASE_SEED})")
    print(f"  Timesteps per model: {TIMESTEPS_PER_MODEL:,}")
    print(f"  Total timesteps: {N_SEEDS * TIMESTEPS_PER_MODEL:,}")

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

    split_idx = int(len(df_clean) * 0.8)
    df_train = df_clean.iloc[:split_idx].reset_index(drop=True)
    df_test = df_clean.iloc[split_idx:].reset_index(drop=True)
    print(f"  Train: {len(df_train):,} | Test: {len(df_test):,}")

    # 3. Train Ensemble
    print("\n[3/5] Training Multi-Seed Ensemble...")
    from environment_v19 import TradingEnvironmentV19
    from rewards.symmetric_curriculum import SymmetricCurriculumReward
    from multi_seed_ensemble import MultiSeedEnsemble
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    def make_env():
        r = SymmetricCurriculumReward(total_timesteps=TIMESTEPS_PER_MODEL)
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
        return DummyVecEnv([lambda: Monitor(e)])

    # Create and train ensemble
    start_time = datetime.now()

    ensemble = MultiSeedEnsemble(
        n_seeds=N_SEEDS,
        base_seed=BASE_SEED,
        aggregation='mean',
    )

    ensemble.train(
        env_factory=make_env,
        model_params=PPO_CONFIG,
        total_timesteps=TIMESTEPS_PER_MODEL,
        verbose=1,
    )

    train_time = (datetime.now() - start_time).total_seconds()
    print(f"\n  Training completed in {train_time:.1f}s")

    # 4. Evaluate Each Model Individually + Ensemble
    print(f"\n[4/5] Evaluating on test set ({N_EVAL_EPISODES} episodes per model)...")

    # Create test environment factory
    def make_test_env():
        test_r = SymmetricCurriculumReward(total_timesteps=TIMESTEPS_PER_MODEL)
        return TradingEnvironmentV19(
            df=df_test,
            initial_balance=ENV_CONFIG["initial_balance"],
            max_position=ENV_CONFIG["max_position"],
            episode_length=ENV_CONFIG["episode_length"],
            max_drawdown_pct=ENV_CONFIG["max_drawdown_pct"],
            use_vol_scaling=ENV_CONFIG["use_vol_scaling"],
            use_regime_detection=ENV_CONFIG["use_regime_detection"],
            reward_function=test_r,
        )

    # Evaluate each individual model
    individual_results = []

    for i, (model, seed) in enumerate(zip(ensemble.models, ensemble.seeds)):
        print(f"\n  Evaluating model {i+1}/{N_SEEDS} (seed={seed})...")
        model_rewards = []
        model_returns = []
        model_actions = []

        for ep in range(N_EVAL_EPISODES):
            test_env = make_test_env()
            obs, _ = test_env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                model_actions.append(float(action[0]))
                if 'step_return' in info:
                    model_returns.append(info['step_return'])

            model_rewards.append(episode_reward)

        # Calculate metrics for this model
        returns = np.array(model_returns) if model_returns else np.array([0])
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 56)
        else:
            sharpe = 0

        cumret = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumret)
        drawdowns = running_max - cumret
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0

        actions = np.array(model_actions)
        long_pct = np.mean(actions > 0.1) * 100
        short_pct = np.mean(actions < -0.1) * 100
        hold_pct = np.mean(np.abs(actions) <= 0.1) * 100

        individual_results.append({
            'seed': seed,
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'win_rate': float(win_rate),
            'mean_reward': float(np.mean(model_rewards)),
            'long_pct': float(long_pct),
            'short_pct': float(short_pct),
            'hold_pct': float(hold_pct),
        })

        print(f"    Sharpe: {sharpe:.2f}, DD: {max_dd*100:.2f}%, WR: {win_rate:.1f}%")

    # Evaluate Ensemble (aggregated predictions)
    print(f"\n  Evaluating ENSEMBLE (aggregated)...")
    ensemble_rewards = []
    ensemble_returns = []
    ensemble_actions = []
    ensemble_agreements = []

    for ep in range(N_EVAL_EPISODES):
        test_env = make_test_env()
        obs, _ = test_env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Use ensemble prediction with confidence
            action, confidence, should_reduce = ensemble.get_action_with_confidence(
                obs, deterministic=True
            )

            # Track agreement
            agreement = ensemble.get_agreement_score(obs)
            ensemble_agreements.append(agreement)

            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            ensemble_actions.append(float(action[0]))
            if 'step_return' in info:
                ensemble_returns.append(info['step_return'])

        ensemble_rewards.append(episode_reward)

    # Calculate ensemble metrics
    returns = np.array(ensemble_returns) if ensemble_returns else np.array([0])
    if len(returns) > 1 and np.std(returns) > 0:
        ensemble_sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 56)
    else:
        ensemble_sharpe = 0

    cumret = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumret)
    drawdowns = running_max - cumret
    ensemble_max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

    ensemble_win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0

    actions = np.array(ensemble_actions)
    ensemble_long = np.mean(actions > 0.1) * 100
    ensemble_short = np.mean(actions < -0.1) * 100
    ensemble_hold = np.mean(np.abs(actions) <= 0.1) * 100

    mean_agreement = np.mean(ensemble_agreements)

    # 5. Summary
    print(f"\n{'='*70}")
    print("RESULTS - MULTI-SEED ENSEMBLE")
    print(f"{'='*70}")

    # Individual model statistics
    sharpes = [r['sharpe'] for r in individual_results]
    max_dds = [r['max_dd'] for r in individual_results]
    win_rates = [r['win_rate'] for r in individual_results]

    print(f"\n  Individual Models ({N_SEEDS} seeds):")
    print(f"  {'-'*50}")
    print(f"  Sharpe Ratio:   {np.mean(sharpes):>6.2f} ± {np.std(sharpes):.2f}  (range: {np.min(sharpes):.2f} to {np.max(sharpes):.2f})")
    print(f"  Max Drawdown:   {np.mean(max_dds)*100:>6.2f}% ± {np.std(max_dds)*100:.2f}%")
    print(f"  Win Rate:       {np.mean(win_rates):>6.1f}% ± {np.std(win_rates):.1f}%")

    print(f"\n  Ensemble (Aggregated):")
    print(f"  {'-'*50}")
    print(f"  Sharpe Ratio:   {ensemble_sharpe:>6.2f}")
    print(f"  Max Drawdown:   {ensemble_max_dd*100:>6.2f}%")
    print(f"  Win Rate:       {ensemble_win_rate:>6.1f}%")
    print(f"  Mean Agreement: {mean_agreement:>6.1%}")
    print(f"  Actions: LONG={ensemble_long:.1f}% HOLD={ensemble_hold:.1f}% SHORT={ensemble_short:.1f}%")

    # Variance Reduction
    individual_mean = np.mean(sharpes)
    print(f"\n  Variance Analysis:")
    print(f"  {'-'*50}")
    print(f"  Individual Sharpe Std:  {np.std(sharpes):.2f}")
    print(f"  Ensemble vs Mean:       {ensemble_sharpe - individual_mean:+.2f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save ensemble
    ensemble_dir = output_dir / f"ensemble_{timestamp}"
    ensemble.save(str(ensemble_dir))
    print(f"\n  Ensemble saved: {ensemble_dir.name}/")

    # Save metrics
    results = {
        "timestamp": datetime.now().isoformat(),
        "version": "ENSEMBLE_V1",
        "config": {
            "n_seeds": N_SEEDS,
            "base_seed": BASE_SEED,
            "timesteps_per_model": TIMESTEPS_PER_MODEL,
            "total_timesteps": N_SEEDS * TIMESTEPS_PER_MODEL,
            "n_eval_episodes": N_EVAL_EPISODES,
        },
        "individual_results": individual_results,
        "individual_summary": {
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "sharpe_min": float(np.min(sharpes)),
            "sharpe_max": float(np.max(sharpes)),
            "max_dd_mean": float(np.mean(max_dds)),
            "max_dd_std": float(np.std(max_dds)),
            "win_rate_mean": float(np.mean(win_rates)),
            "win_rate_std": float(np.std(win_rates)),
        },
        "ensemble_results": {
            "sharpe_ratio": float(ensemble_sharpe),
            "max_drawdown": float(ensemble_max_dd),
            "win_rate": float(ensemble_win_rate),
            "mean_agreement": float(mean_agreement),
            "mean_reward": float(np.mean(ensemble_rewards)),
            "action_distribution": {
                "long": float(ensemble_long),
                "hold": float(ensemble_hold),
                "short": float(ensemble_short),
            },
        },
        "train_time_seconds": train_time,
        "ppo_config": PPO_CONFIG,
        "env_config": ENV_CONFIG,
    }

    results_path = output_dir / f"ensemble_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved: {results_path.name}")

    print(f"\n{'='*70}")
    print("ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*70}")

    # Final recommendation
    print(f"\n  RECOMMENDATION:")
    if ensemble_sharpe > individual_mean:
        print(f"  Ensemble ({ensemble_sharpe:.2f}) outperforms average individual model ({individual_mean:.2f})")
        print(f"  -> Use ensemble for production")
    else:
        print(f"  Ensemble ({ensemble_sharpe:.2f}) vs average individual ({individual_mean:.2f})")
        print(f"  -> Consider using best individual model (Sharpe={np.max(sharpes):.2f})")


if __name__ == "__main__":
    main()
