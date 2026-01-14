#!/usr/bin/env python3
"""
USD/COP RL Trading System - Config Comparison Experiment
=========================================================

Ejecuta 3 configuraciones en paralelo para encontrar el balance optimo:

A) SIN anti-overfit (config original) - Baseline
B) Config BALANCEADA - Recomendada
C) Anti-overfit + RegimeDetector - Alternativa

USO:
    python run_config_comparison.py --config A   # Solo config A
    python run_config_comparison.py --config B   # Solo config B
    python run_config_comparison.py --config C   # Solo config C
    python run_config_comparison.py --all        # Las 3 secuencialmente

Author: Claude Code
Date: 2025-12-25
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from environment_v19 import TradingEnvironmentV19

# Try to import RegimeDetector
try:
    from regime_detector import RegimeDetector
    HAS_REGIME_DETECTOR = True
except ImportError:
    HAS_REGIME_DETECTOR = False
    print("[WARN] RegimeDetector not available")


# =============================================================================
# CONFIGURATIONS
# =============================================================================

CONFIG_A = {
    "name": "A_original",
    "description": "Config original SIN anti-overfit (baseline)",
    "model": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "ent_coef": 0.05,
        "clip_range": 0.2,
        "policy_kwargs": {"net_arch": [64, 64]},
    },
    "training": {
        "timesteps": 100_000,
    },
    "environment": {
        "use_vol_scaling": True,
        "use_regime_detection": False,
        "protection_mode": "vol_only",
    },
}

CONFIG_B = {
    "name": "B_balanced",
    "description": "Config BALANCEADA (recomendada)",
    "model": {
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 64,  # Factor of 2048
        "n_epochs": 7,
        "gamma": 0.97,
        "ent_coef": 0.08,
        "clip_range": 0.15,
        "policy_kwargs": {"net_arch": [64, 32]},
    },
    "training": {
        "timesteps": 75_000,
    },
    "environment": {
        "use_vol_scaling": True,
        "use_regime_detection": False,
        "protection_mode": "vol_only",
    },
}

CONFIG_C = {
    "name": "C_regime",
    "description": "Anti-overfit + RegimeDetector",
    "model": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 5,
        "gamma": 0.97,
        "ent_coef": 0.10,
        "clip_range": 0.1,
        "policy_kwargs": {"net_arch": [32, 32]},
    },
    "training": {
        "timesteps": 75_000,
    },
    "environment": {
        "use_vol_scaling": True,
        "use_regime_detection": True,
        "protection_mode": "min",
    },
}

# CONFIG B_v3: Balanced + RegimeDetector (RECOMMENDED by 3 agents)
CONFIG_B_V3 = {
    "name": "B_v3_regime",
    "description": "Balanced [48,32] + RegimeDetector + ent_coef=0.05",
    "model": {
        "learning_rate": 1e-4,
        "n_steps": 2048,
        "batch_size": 128,  # Factor of 2048
        "n_epochs": 4,
        "gamma": 0.95,
        "ent_coef": 0.05,  # PHASE 2: Reduced from 0.10 to 0.05
        "clip_range": 0.1,
        "policy_kwargs": {"net_arch": [48, 32]},  # Sweet spot
    },
    "training": {
        "timesteps": 60_000,
    },
    "environment": {
        "use_vol_scaling": True,
        "use_regime_detection": True,  # ACTIVATED!
        "protection_mode": "min",  # MIN logic (the fix!)
    },
}

CONFIGS = {"A": CONFIG_A, "B": CONFIG_B, "C": CONFIG_C, "B3": CONFIG_B_V3}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(timeframe: str = "5min") -> pd.DataFrame:
    """Load dataset."""
    data_dir = ROOT.parent.parent / "data" / "pipeline" / "07_output" / "datasets_5min"

    if timeframe == "5min":
        path = data_dir / "RL_DS3_MACRO_CORE.csv"
    else:
        path = data_dir / "RL_DS3_MACRO_CORE_15MIN.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Ensure required columns
    if 'close_return' not in df.columns and 'log_ret_5m' in df.columns:
        df['close_return'] = df['log_ret_5m']

    return df


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Dict,
    fold_num: int,
    seed: int = 42,
) -> Dict:
    """Run single fold with given config."""

    np.random.seed(seed)

    # Feature columns
    exclude = ['timestamp', 'date', 'time', 'symbol', 'close_return', 'log_ret_5m']
    feature_cols = [c for c in train_df.columns if c not in exclude
                    and train_df[c].dtype in ['float64', 'float32', 'int64']]

    # Episode length - CORRECTED: actual bars/day is ~56 for 5-min data
    bars_per_day = 56  # Actual bars per trading day (not 288 which assumes 24h)
    episode_length = min(bars_per_day * 10, len(train_df) - 10)

    # Environment config
    env_config = config.get("environment", {})

    # Create regime detector if needed
    regime_detector = None
    if env_config.get("use_regime_detection", False) and HAS_REGIME_DETECTOR:
        regime_detector = RegimeDetector()

    # Create environments
    train_env = TradingEnvironmentV19(
        df=train_df,
        initial_balance=10000,
        max_position=1.0,
        episode_length=episode_length,
        feature_columns=feature_cols,
        use_vol_scaling=env_config.get("use_vol_scaling", True),
        use_regime_detection=env_config.get("use_regime_detection", False),
        regime_detector=regime_detector,
        protection_mode=env_config.get("protection_mode", "min"),
        verbose=0,
    )

    test_env = TradingEnvironmentV19(
        df=test_df,
        initial_balance=10000,
        max_position=1.0,
        episode_length=min(episode_length, len(test_df) - 10),
        feature_columns=feature_cols,
        use_vol_scaling=env_config.get("use_vol_scaling", True),
        use_regime_detection=env_config.get("use_regime_detection", False),
        regime_detector=regime_detector,
        protection_mode=env_config.get("protection_mode", "min"),
        verbose=0,
    )

    # Create model
    model_config = config["model"]
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=model_config["learning_rate"],
        n_steps=model_config["n_steps"],
        batch_size=model_config["batch_size"],
        n_epochs=model_config["n_epochs"],
        gamma=model_config["gamma"],
        ent_coef=model_config["ent_coef"],
        clip_range=model_config["clip_range"],
        policy_kwargs=model_config.get("policy_kwargs", {}),
        seed=seed,
        verbose=0,
        device="cpu",  # CPU is faster for MLP policies
    )

    # Train
    timesteps = config["training"]["timesteps"]
    model.learn(total_timesteps=timesteps, progress_bar=False)

    # Evaluate on train
    train_sharpe = evaluate_model(model, train_env, n_episodes=3)

    # Evaluate on test
    test_results = evaluate_model_detailed(model, test_env, n_episodes=5)

    return {
        "fold": fold_num,
        "train_sharpe": train_sharpe,
        "test_sharpe": test_results["sharpe"],
        "test_max_dd": test_results["max_dd"],
        "test_return": test_results["total_return"],
        "pct_long": test_results["pct_long"],
        "pct_short": test_results["pct_short"],
        "pct_hold": test_results["pct_hold"],
    }


def evaluate_model(model, env, n_episodes: int = 3, bars_per_day: int = 56) -> float:
    """Quick evaluation - return Sharpe.

    NOTE: Uses CORRECT annualization factor based on actual bars per day.
    Actual data has ~56.5 bars/day, NOT 288 (which would be 24h continuous).
    """
    all_returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_returns = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_returns.append(info.get('step_return', 0))

        all_returns.extend(episode_returns)

    if len(all_returns) < 2:
        return 0.0

    returns = np.array(all_returns)
    # CORRECTED: Use actual bars per year (252 * 56.5 ≈ 14,238)
    bars_per_year = 252 * bars_per_day
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(bars_per_year)
    return sharpe


def evaluate_model_detailed(model, env, n_episodes: int = 5, bars_per_day: int = 56) -> Dict:
    """Detailed evaluation.

    NOTE: Uses CORRECT annualization factor based on actual bars per day.
    """
    all_returns = []
    all_actions = []
    portfolio_values = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_portfolio = [10000]

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            all_returns.append(info.get('step_return', 0))
            all_actions.append(action[0])
            # FIX: Environment uses 'portfolio', not 'portfolio_value'
            episode_portfolio.append(info.get('portfolio', episode_portfolio[-1]))

        portfolio_values.extend(episode_portfolio)

    returns = np.array(all_returns)
    actions = np.array(all_actions)

    # FIX: Sharpe must use DAILY returns, not per-bar
    n_bars = len(returns)
    n_days = n_bars // bars_per_day

    if n_days >= 2:
        # Aggregate to daily returns
        daily_returns = returns[:n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
        # Annualize using sqrt(252) for daily data
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown (with protection against div/0)
    portfolio = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-10) * 100
    max_dd = np.max(drawdown)

    # Total return
    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0] * 100

    # Action distribution - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
    ACTION_THRESHOLD = 0.10
    n_long = np.sum(actions > ACTION_THRESHOLD)
    n_short = np.sum(actions < -ACTION_THRESHOLD)
    n_hold = np.sum(np.abs(actions) <= ACTION_THRESHOLD)
    total = len(actions)

    return {
        "sharpe": sharpe,
        "max_dd": max_dd,
        "total_return": total_return,
        "pct_long": n_long / total * 100 if total > 0 else 0,
        "pct_short": n_short / total * 100 if total > 0 else 0,
        "pct_hold": n_hold / total * 100 if total > 0 else 0,
    }


def run_walkforward(config: Dict, n_folds: int = 5) -> Dict:
    """Run walk-forward validation with given config."""

    print(f"\n{'='*70}")
    print(f"  CONFIG: {config['name']}")
    print(f"  {config['description']}")
    print(f"{'='*70}\n")

    # Load data
    df = load_data("5min")
    print(f"  Loaded {len(df):,} rows")

    # Setup folds (expanding window)
    bars_per_day = 288
    train_days = 60
    test_days = 20
    gap_days = 5

    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    gap_bars = gap_days * bars_per_day

    fold_results = []

    for fold in range(n_folds):
        print(f"\n  FOLD {fold + 1}/{n_folds}")
        print(f"  {'-'*40}")

        # Expanding window
        train_end = train_bars + fold * test_bars
        test_start = train_end + gap_bars
        test_end = test_start + test_bars

        if test_end > len(df):
            print(f"  [SKIP] Not enough data")
            break

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        print(f"  Train: 0 - {train_end:,} ({len(train_df):,} bars)")
        print(f"  Test:  {test_start:,} - {test_end:,} ({len(test_df):,} bars)")

        result = run_fold(train_df, test_df, config, fold + 1)
        fold_results.append(result)

        print(f"\n  Results:")
        print(f"    Train Sharpe: {result['train_sharpe']:.2f}")
        print(f"    Test Sharpe:  {result['test_sharpe']:.2f}")
        print(f"    Test MaxDD:   {result['test_max_dd']:.1f}%")
        print(f"    Actions:      L:{result['pct_long']:.0f}% S:{result['pct_short']:.0f}% H:{result['pct_hold']:.0f}%")

    # Summary
    if fold_results:
        test_sharpes = [r['test_sharpe'] for r in fold_results]
        train_sharpes = [r['train_sharpe'] for r in fold_results]
        hold_pcts = [r['pct_hold'] for r in fold_results]

        summary = {
            "config_name": config['name'],
            "description": config['description'],
            "n_folds": len(fold_results),
            "avg_test_sharpe": np.mean(test_sharpes),
            "std_test_sharpe": np.std(test_sharpes),
            "avg_train_sharpe": np.mean(train_sharpes),
            "avg_hold_pct": np.mean(hold_pcts),
            "overfit_gap": np.mean(train_sharpes) - np.mean(test_sharpes),
            "fold_results": fold_results,
        }

        print(f"\n  {'='*40}")
        print(f"  SUMMARY: {config['name']}")
        print(f"  {'='*40}")
        print(f"  Avg Test Sharpe:  {summary['avg_test_sharpe']:.2f} +/- {summary['std_test_sharpe']:.2f}")
        print(f"  Avg Train Sharpe: {summary['avg_train_sharpe']:.2f}")
        print(f"  Overfit Gap:      {summary['overfit_gap']:.2f}")
        print(f"  Avg HOLD %:       {summary['avg_hold_pct']:.0f}%")

        return summary

    return {"error": "No folds completed"}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Config Comparison Experiment')
    parser.add_argument('--config', '-c', choices=['A', 'B', 'C', 'B3'],
                        help='Run specific config')
    parser.add_argument('--all', action='store_true',
                        help='Run all configs sequentially')
    parser.add_argument('--folds', '-f', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='Output directory')

    args = parser.parse_args()

    output_dir = ROOT / args.output
    output_dir.mkdir(exist_ok=True)

    results = {}

    if args.all:
        configs_to_run = ['A', 'B', 'C']
    elif args.config:
        configs_to_run = [args.config]
    else:
        print("Usage: python run_config_comparison.py --config A|B|C")
        print("       python run_config_comparison.py --all")
        return

    for config_name in configs_to_run:
        config = CONFIGS[config_name]
        result = run_walkforward(config, n_folds=args.folds)
        results[config_name] = result

        # Save individual result
        output_file = output_dir / f"comparison_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n  Saved: {output_file}")

    # Print comparison table
    if len(results) > 1:
        print("\n" + "="*70)
        print("  COMPARISON TABLE")
        print("="*70)
        print(f"\n  {'Config':<15} {'Sharpe':<12} {'HOLD %':<10} {'Overfit Gap':<12} {'Verdict'}")
        print(f"  {'-'*60}")

        for name, res in results.items():
            if 'error' not in res:
                sharpe = res['avg_test_sharpe']
                hold = res['avg_hold_pct']
                gap = res['overfit_gap']

                # Verdict
                if sharpe > 0.8 and 20 <= hold <= 50 and gap < 2.0:
                    verdict = "GOOD"
                elif sharpe > 0.5 and hold < 80:
                    verdict = "OK"
                else:
                    verdict = "POOR"

                print(f"  {name:<15} {sharpe:<12.2f} {hold:<10.0f} {gap:<12.2f} {verdict}")

    print("\n" + "="*70)
    print("  DONE")
    print("="*70)


if __name__ == "__main__":
    main()
