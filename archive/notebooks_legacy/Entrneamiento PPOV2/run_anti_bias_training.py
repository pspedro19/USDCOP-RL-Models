#!/usr/bin/env python
"""
Anti-Bias Training: Entrenamiento con protecciones agresivas contra sesgo direccional
======================================================================================

Problema identificado:
- Los modelos aprenden sesgo LONG o SHORT aleatorio según seed
- El sesgo coincide (o no) con la tendencia del test period
- Esto causa varianza extrema en Sharpe (-4.54 a +5.06)

Solución implementada:
1. SymmetryTracker con penalización FUERTE (10x más agresiva)
2. max_directional_bias reducido a 10% (era 30%)
3. Data augmentation: flip de retornos 50% del tiempo
4. Regime features en observation space
5. Régimen detector activo
"""

import sys
import os

# Force unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

print("Script starting...", flush=True)

import json
import warnings
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass

print("Basic imports done", flush=True)

import numpy as np
import pandas as pd

print("NumPy/Pandas imported", flush=True)

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# =============================================================================
# CONFIGURATION - ANTI-BIAS AGGRESSIVA
# =============================================================================

DATA_PATH = Path(__file__).parent.parent.parent / "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv"

# Ensemble más pequeño pero efectivo
N_SEEDS = 5
BASE_SEED = 42
TIMESTEPS_PER_MODEL = 80_000  # Más timesteps para convergencia
N_EVAL_EPISODES = 10

PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "ent_coef": 0.08,  # Mayor entropía para exploración
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
    "use_regime_detection": True,  # ACTIVADO
}

# Anti-bias config AGRESIVA
def create_anti_bias_config():
    """Crea config con parámetros anti-sesgo agresivos."""
    from rewards.symmetric_curriculum import RewardConfig

    # Crear config base y modificar parámetros anti-sesgo
    config = RewardConfig(
        # Fases del curriculum
        phase_boundaries=(0.30, 0.60),
        transition_target_cost_bps=10.0,
        realistic_min_cost_bps=25.0,
        realistic_max_cost_bps=36.0,

        # ANTI-BIAS: Configuración AGRESIVA
        symmetry_window=60,
        max_directional_bias=0.10,  # ERA 0.30, ahora 0.10 (3x más estricto)
        symmetry_penalty_scale=20.0,  # ERA 2.0, ahora 20.0 (10x más fuerte)

        # Patológicos
        max_trades_per_bar=0.05,
        overtrading_lookback=120,
        overtrading_penalty=0.5,
        max_hold_duration=36,
        inactivity_penalty=0.3,
        reversal_threshold=5,
        churning_penalty=0.4,

        # Sortino
        sortino_window=60,
        sortino_mar=0.0,
        reward_scale=100.0,
        clip_range=(-5.0, 5.0),
    )
    return config


class DataAugmenter:
    """
    Augmenta datos flipeando retornos para eliminar sesgo direccional.
    50% del tiempo flipea todos los retornos (LONG<->SHORT equivalentes).
    """

    def __init__(self, flip_probability: float = 0.5):
        self.flip_probability = flip_probability
        self.return_columns = ['log_ret_5m', 'log_ret_1h', 'log_ret_4h']

    def augment(self, df: pd.DataFrame, seed: int = None) -> pd.DataFrame:
        """Retorna df con retornos potencialmente flipeados."""
        if seed is not None:
            np.random.seed(seed)

        if np.random.random() < self.flip_probability:
            df_aug = df.copy()
            for col in self.return_columns:
                if col in df_aug.columns:
                    df_aug[col] = -df_aug[col]  # Flip sign
            return df_aug
        return df




def main():
    print("=" * 70)
    print("ANTI-BIAS TRAINING - Protecciones Agresivas")
    print("=" * 70)
    print(f"  max_directional_bias: 0.10 (era 0.30)")
    print(f"  symmetry_penalty_scale: 20.0 (era 2.0)")
    print(f"  Data augmentation: 50% flip")
    print(f"  Regime detection: ENABLED")
    print(f"  Seeds: {N_SEEDS}")
    print(f"  Timesteps/model: {TIMESTEPS_PER_MODEL:,}")

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

    # 3. Train Models with Anti-Bias
    print("\n[3/5] Training with Anti-Bias protections...")
    from environment_v19 import TradingEnvironmentV19
    from rewards.symmetric_curriculum import SymmetricCurriculumReward
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    augmenter = DataAugmenter(flip_probability=0.5)
    models = []
    seeds = [BASE_SEED + i * 1000 for i in range(N_SEEDS)]

    start_time = datetime.now()

    for i, seed in enumerate(seeds):
        print(f"\n  Training model {i+1}/{N_SEEDS} (seed={seed})...")

        # Augment data for this seed
        df_train_aug = augmenter.augment(df_train, seed=seed)
        flip_applied = "FLIPPED" if id(df_train_aug) != id(df_train) else "ORIGINAL"
        print(f"    Data: {flip_applied}")

        # Create env with anti-bias reward
        def make_env(df_data=df_train_aug):
            anti_bias_cfg = create_anti_bias_config()
            r = SymmetricCurriculumReward(
                total_timesteps=TIMESTEPS_PER_MODEL,
                config=anti_bias_cfg,
            )
            e = TradingEnvironmentV19(
                df=df_data,
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
            seed=seed,
            verbose=0,
        )

        model.learn(total_timesteps=TIMESTEPS_PER_MODEL, progress_bar=True)
        models.append((model, seed))
        print(f"    Model {i+1} completed")

    train_time = (datetime.now() - start_time).total_seconds()
    print(f"\n  Total training time: {train_time:.1f}s")

    # 4. Evaluate
    print(f"\n[4/5] Evaluating on test set...")

    results = []

    for i, (model, seed) in enumerate(models):
        print(f"\n  Evaluating model {i+1}/{N_SEEDS} (seed={seed})...")

        model_rewards = []
        model_returns = []
        model_actions = []

        for ep in range(N_EVAL_EPISODES):
            # Create fresh test env
            test_r = SymmetricCurriculumReward(
                total_timesteps=TIMESTEPS_PER_MODEL,
                config=create_anti_bias_config(),
            )
            test_env = TradingEnvironmentV19(
                df=df_test,
                initial_balance=ENV_CONFIG["initial_balance"],
                max_position=ENV_CONFIG["max_position"],
                episode_length=ENV_CONFIG["episode_length"],
                max_drawdown_pct=ENV_CONFIG["max_drawdown_pct"],
                use_vol_scaling=ENV_CONFIG["use_vol_scaling"],
                use_regime_detection=ENV_CONFIG["use_regime_detection"],
                reward_function=test_r,
            )

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

        # Calculate metrics
        returns = np.array(model_returns) if model_returns else np.array([0])
        actions = np.array(model_actions)

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 56)
        else:
            sharpe = 0

        cumret = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumret)
        drawdowns = running_max - cumret
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0

        win_rate = np.mean(returns > 0) * 100 if len(returns) > 0 else 0

        # Action distribution
        long_pct = np.mean(actions > 0.1) * 100
        short_pct = np.mean(actions < -0.1) * 100
        hold_pct = np.mean(np.abs(actions) <= 0.1) * 100

        # Directional bias
        directional_bias = np.mean(actions)

        results.append({
            'seed': seed,
            'sharpe': float(sharpe),
            'max_dd': float(max_dd),
            'win_rate': float(win_rate),
            'mean_reward': float(np.mean(model_rewards)),
            'long_pct': float(long_pct),
            'short_pct': float(short_pct),
            'hold_pct': float(hold_pct),
            'directional_bias': float(directional_bias),
        })

        print(f"    Sharpe: {sharpe:.2f}, Bias: {directional_bias:+.3f}, "
              f"L/H/S: {long_pct:.0f}/{hold_pct:.0f}/{short_pct:.0f}%")

    # 5. Summary
    print(f"\n{'='*70}")
    print("ANTI-BIAS TRAINING RESULTS")
    print(f"{'='*70}")

    sharpes = [r['sharpe'] for r in results]
    biases = [r['directional_bias'] for r in results]

    print(f"\n  Individual Models ({N_SEEDS} seeds):")
    print(f"  {'-'*50}")
    print(f"  Sharpe:    {np.mean(sharpes):>6.2f} +/- {np.std(sharpes):.2f}  (range: {np.min(sharpes):.2f} to {np.max(sharpes):.2f})")
    print(f"  Bias:      {np.mean(biases):>+6.3f} +/- {np.std(biases):.3f}  (range: {np.min(biases):+.3f} to {np.max(biases):+.3f})")

    # Compare vs baseline
    print(f"\n  Comparacion vs Baseline (sin anti-bias):")
    print(f"  {'-'*50}")
    print(f"  Baseline Sharpe Std:    3.68")
    print(f"  Anti-Bias Sharpe Std:   {np.std(sharpes):.2f}")
    print(f"  Reduccion varianza:     {(1 - np.std(sharpes)/3.68)*100:.1f}%")

    print(f"\n  Baseline Bias Range:    -0.311 to +0.070")
    print(f"  Anti-Bias Bias Range:   {np.min(biases):+.3f} to {np.max(biases):+.3f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Save models
    for i, (model, seed) in enumerate(models):
        model_path = output_dir / f"anti_bias_model_seed_{seed}_{timestamp}.zip"
        model.save(str(model_path))

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "version": "ANTI_BIAS_V1",
        "config": {
            "n_seeds": N_SEEDS,
            "timesteps_per_model": TIMESTEPS_PER_MODEL,
            "max_directional_bias": 0.10,
            "symmetry_penalty_scale": 20.0,
            "data_augmentation": "50% flip",
        },
        "results": results,
        "summary": {
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": float(np.std(sharpes)),
            "bias_mean": float(np.mean(biases)),
            "bias_std": float(np.std(biases)),
        },
        "comparison_vs_baseline": {
            "baseline_sharpe_std": 3.68,
            "anti_bias_sharpe_std": float(np.std(sharpes)),
            "variance_reduction_pct": float((1 - np.std(sharpes)/3.68)*100),
        },
        "train_time_seconds": train_time,
    }

    results_path = output_dir / f"anti_bias_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved: {results_path.name}")

    print(f"\n{'='*70}")
    print("ANTI-BIAS TRAINING COMPLETE")
    print(f"{'='*70}")

    if np.std(sharpes) < 2.0:
        print(f"\n  SUCCESS: Varianza reducida significativamente")
        print(f"  El anti-bias esta funcionando")
    else:
        print(f"\n  WARNING: Varianza aun alta")
        print(f"  Considerar aumentar symmetry_penalty_scale")


if __name__ == "__main__":
    main()
