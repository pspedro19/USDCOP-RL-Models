#!/usr/bin/env python3
"""
USD/COP RL Trading System - Ensemble Training
==============================================

Entrena un ensemble de 2 modelos PPO con diferentes configuraciones:
- Model A (70%): ent_coef=0.10, conservador, baja varianza
- Model B (30%): ent_coef=0.05, agresivo, alta performance

OBJETIVO (basado en analisis de 5 agentes):
- Sharpe esperado: ~1.95 (vs 1.56 solo A, 2.66 solo B)
- Varianza esperada: ~1.8 (vs 2.05 solo A, 3.37 solo B)
- Mejor balance riesgo/retorno

USO:
    python run_ensemble_training.py                    # Default 5 folds
    python run_ensemble_training.py --folds 3          # 3 folds
    python run_ensemble_training.py --weights 0.6 0.4  # Custom weights

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

# Ensemble imports
from ensemble import (
    EnsemblePredictor,
    MODEL_A_CONFIG,
    MODEL_B_CONFIG,
    DEFAULT_ENSEMBLE_CONFIGS,
)

# Try to import RegimeDetector
try:
    from regime_detector import RegimeDetector
    HAS_REGIME_DETECTOR = True
except ImportError:
    HAS_REGIME_DETECTOR = False
    print("[WARN] RegimeDetector not available")


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
# TRAINING FUNCTIONS
# =============================================================================

def create_environment(
    df: pd.DataFrame,
    feature_cols: List[str],
    env_config: Dict,
    episode_length: int,
) -> TradingEnvironmentV19:
    """Create trading environment with given config."""

    # Create regime detector if needed
    regime_detector = None
    if env_config.get("use_regime_detection", False) and HAS_REGIME_DETECTOR:
        regime_detector = RegimeDetector()

    env = TradingEnvironmentV19(
        df=df,
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

    return env


def train_single_model(
    train_df: pd.DataFrame,
    model_config: Dict,
    env_config: Dict,
    feature_cols: List[str],
    episode_length: int,
    timesteps: int = 60_000,
    seed: int = 42,
) -> PPO:
    """Train a single PPO model."""

    np.random.seed(seed)

    # Create environment
    train_env = create_environment(train_df, feature_cols, env_config, episode_length)

    # Create model
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
        device="cpu",
    )

    # Train
    model.learn(total_timesteps=timesteps, progress_bar=False)

    return model


def train_ensemble_fold(
    train_df: pd.DataFrame,
    ensemble_configs: List,
    feature_cols: List[str],
    episode_length: int,
    timesteps: int = 60_000,
    seed: int = 42,
) -> EnsemblePredictor:
    """Train ensemble for a single fold."""

    ensemble = EnsemblePredictor()

    for i, config in enumerate(ensemble_configs):
        print(f"      Training {config.name} (weight={config.weight:.0%})...")

        model = train_single_model(
            train_df=train_df,
            model_config=config.model_params,
            env_config=config.env_config,
            feature_cols=feature_cols,
            episode_length=episode_length,
            timesteps=timesteps,
            seed=seed + i,  # Different seed per model
        )

        ensemble.add_model(model, config.weight, config.name)

    return ensemble


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_ensemble(
    ensemble: EnsemblePredictor,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    episode_length: int,
    env_config: Dict,
    n_episodes: int = 5,
    bars_per_day: int = 56,
    fixed_start_indices: Optional[List[int]] = None,
) -> Dict:
    """Evaluate ensemble on test data.

    FIX: Use fixed_start_indices to ensure all models are evaluated on SAME data.
    """

    test_env = create_environment(test_df, feature_cols, env_config, episode_length)

    all_returns = []
    all_actions = []
    all_agreements = []
    portfolio_values = []

    # FIX BUG #3: Use fixed indices to ensure comparable evaluation
    if fixed_start_indices is None:
        # Generate deterministic indices based on data length
        max_start = len(test_df) - episode_length - 1
        step_size = max(1, max_start // n_episodes)
        fixed_start_indices = [i * step_size for i in range(n_episodes)]

    for start_idx in fixed_start_indices:
        obs, _ = test_env.reset(options={'start_idx': start_idx})
        done = False
        episode_portfolio = [10000]

        while not done:
            # Ensemble prediction
            action, _ = ensemble.predict(obs, deterministic=True)

            # Agreement score
            agreement = ensemble.get_agreement_score(obs)
            all_agreements.append(agreement)

            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated

            all_returns.append(info.get('step_return', 0))
            all_actions.append(action[0])
            # FIX BUG #1: Environment uses 'portfolio', not 'portfolio_value'
            episode_portfolio.append(info.get('portfolio', episode_portfolio[-1]))

        portfolio_values.extend(episode_portfolio)

    returns = np.array(all_returns)
    actions = np.array(all_actions)

    # FIX BUG #2: Sharpe must use DAILY returns, not per-bar
    # Aggregate per-bar returns to daily returns FIRST, then annualize
    n_bars = len(returns)
    n_days = n_bars // bars_per_day

    if n_days >= 2:
        # Aggregate to daily returns
        daily_returns = returns[:n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
        # Annualize using sqrt(252) for daily data
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    else:
        # Fallback for insufficient data
        sharpe = 0.0

    # Max Drawdown (with protection against div/0)
    portfolio = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-10) * 100
    max_dd = np.max(drawdown)

    # Total return
    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0] * 100

    # Action distribution
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
        "avg_agreement": np.mean(all_agreements),
        "std_actions": np.std(actions),
    }


def evaluate_individual_models(
    ensemble: EnsemblePredictor,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    episode_length: int,
    env_config: Dict,
    n_episodes: int = 5,
    bars_per_day: int = 56,
    fixed_start_indices: Optional[List[int]] = None,
) -> List[Dict]:
    """Evaluate each model in ensemble individually.

    FIX: Use same fixed_start_indices as ensemble evaluation for fair comparison.
    """

    # FIX BUG #3: Use fixed indices for comparable evaluation
    if fixed_start_indices is None:
        max_start = len(test_df) - episode_length - 1
        step_size = max(1, max_start // n_episodes)
        fixed_start_indices = [i * step_size for i in range(n_episodes)]

    results = []

    for i, model in enumerate(ensemble.models):
        test_env = create_environment(test_df, feature_cols, env_config, episode_length)

        all_returns = []
        all_actions = []

        for start_idx in fixed_start_indices:
            obs, _ = test_env.reset(options={'start_idx': start_idx})
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = test_env.step(action)
                done = terminated or truncated

                all_returns.append(info.get('step_return', 0))
                all_actions.append(action[0])

        returns = np.array(all_returns)

        # FIX BUG #2: Aggregate to daily returns before annualizing
        n_bars = len(returns)
        n_days = n_bars // bars_per_day

        if n_days >= 2:
            daily_returns = returns[:n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
            sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe = 0.0

        results.append({
            "model_idx": i,
            "weight": ensemble.weights[i],
            "sharpe": sharpe,
        })

    return results


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================

def _calculate_model_correlation(
    ensemble: EnsemblePredictor,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    episode_length: int,
    env_config: Dict,
    n_samples: int = 200,
) -> float:
    """Calculate correlation between model actions.

    Returns correlation coefficient. Negative means models disagree!
    """
    if len(ensemble.models) < 2:
        return 1.0

    test_env = create_environment(test_df, feature_cols, env_config, episode_length)
    obs, _ = test_env.reset(options={'start_idx': 0})

    actions_a = []
    actions_b = []

    for _ in range(n_samples):
        act_a, _ = ensemble.models[0].predict(obs, deterministic=True)
        act_b, _ = ensemble.models[1].predict(obs, deterministic=True)

        actions_a.append(float(act_a[0]))
        actions_b.append(float(act_b[0]))

        # Step with ensemble action
        action_e, _ = ensemble.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = test_env.step(action_e)

        if term or trunc:
            obs, _ = test_env.reset(options={'start_idx': 0})

    if len(actions_a) < 2:
        return 0.0

    corr = np.corrcoef(actions_a, actions_b)[0, 1]
    return corr if not np.isnan(corr) else 0.0


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def run_walkforward_ensemble(
    n_folds: int = 5,
    timesteps: int = 60_000,
    weights: Optional[Tuple[float, float]] = None,
) -> Dict:
    """Run walk-forward validation with ensemble."""

    print("\n" + "="*70)
    print("  ENSEMBLE TRAINING: 70% Model A + 30% Model B")
    print("  Strategy: Combine stability (A) with alpha (B)")
    print("="*70 + "\n")

    # Load data
    df = load_data("5min")
    print(f"  Loaded {len(df):,} rows")

    # Feature columns
    exclude = ['timestamp', 'date', 'time', 'symbol', 'close_return', 'log_ret_5m']
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'float32', 'int64']]

    # Episode config
    bars_per_day = 56
    episode_length = min(bars_per_day * 10, len(df) // 10)

    # Fold config
    bars_per_day_raw = 288
    train_days = 60
    test_days = 20
    gap_days = 5

    train_bars = train_days * bars_per_day_raw
    test_bars = test_days * bars_per_day_raw
    gap_bars = gap_days * bars_per_day_raw

    # Custom weights if provided
    ensemble_configs = DEFAULT_ENSEMBLE_CONFIGS.copy()
    if weights:
        ensemble_configs[0] = MODEL_A_CONFIG
        ensemble_configs[0] = EnsembleConfig(
            name=MODEL_A_CONFIG.name,
            weight=weights[0],
            model_params=MODEL_A_CONFIG.model_params,
            env_config=MODEL_A_CONFIG.env_config,
        )
        ensemble_configs[1] = EnsembleConfig(
            name=MODEL_B_CONFIG.name,
            weight=weights[1],
            model_params=MODEL_B_CONFIG.model_params,
            env_config=MODEL_B_CONFIG.env_config,
        )

    fold_results = []
    all_individual_results = []

    for fold in range(n_folds):
        print(f"\n  FOLD {fold + 1}/{n_folds}")
        print(f"  {'-'*50}")

        # Expanding window
        train_end = train_bars + fold * test_bars
        test_start = train_end + gap_bars
        test_end = test_start + test_bars

        if test_end > len(df):
            print(f"  [SKIP] Not enough data")
            break

        train_df = df.iloc[:train_end].copy()
        test_df = df.iloc[test_start:test_end].copy()

        print(f"    Train: 0 - {train_end:,} ({len(train_df):,} bars)")
        print(f"    Test:  {test_start:,} - {test_end:,} ({len(test_df):,} bars)")

        # Train ensemble
        print(f"\n    Training ensemble...")
        ensemble = train_ensemble_fold(
            train_df=train_df,
            ensemble_configs=ensemble_configs,
            feature_cols=feature_cols,
            episode_length=episode_length,
            timesteps=timesteps,
            seed=42 + fold,
        )

        # Evaluate ensemble
        print(f"    Evaluating ensemble...")
        env_config = ensemble_configs[0].env_config
        eval_episode_length = min(episode_length, len(test_df) - 10)

        # FIX BUG #3: Calculate FIXED indices ONCE and use for ALL evaluations
        n_eval_episodes = 5
        max_start = len(test_df) - eval_episode_length - 1
        step_size = max(1, max_start // n_eval_episodes)
        fixed_start_indices = [i * step_size for i in range(n_eval_episodes)]

        ensemble_results = evaluate_ensemble(
            ensemble=ensemble,
            test_df=test_df,
            feature_cols=feature_cols,
            episode_length=eval_episode_length,
            env_config=env_config,
            fixed_start_indices=fixed_start_indices,  # SAME indices for all
        )

        # Evaluate individual models with SAME indices
        individual_results = evaluate_individual_models(
            ensemble=ensemble,
            test_df=test_df,
            feature_cols=feature_cols,
            episode_length=eval_episode_length,
            env_config=env_config,
            fixed_start_indices=fixed_start_indices,  # SAME indices for all
        )
        all_individual_results.append(individual_results)

        # FIX: Calculate model correlation to detect anti-correlation
        model_correlation = _calculate_model_correlation(
            ensemble, test_df, feature_cols, eval_episode_length, env_config
        )

        # Store results
        fold_result = {
            "fold": fold + 1,
            "ensemble_sharpe": ensemble_results["sharpe"],
            "ensemble_max_dd": ensemble_results["max_dd"],
            "ensemble_return": ensemble_results["total_return"],
            "pct_hold": ensemble_results["pct_hold"],
            "pct_long": ensemble_results["pct_long"],
            "pct_short": ensemble_results["pct_short"],
            "avg_agreement": ensemble_results["avg_agreement"],
            "model_a_sharpe": individual_results[0]["sharpe"],
            "model_b_sharpe": individual_results[1]["sharpe"],
            "model_correlation": model_correlation,  # NEW: Track correlation
        }
        fold_results.append(fold_result)

        # Print results
        print(f"\n    Results:")
        print(f"      Ensemble Sharpe:   {ensemble_results['sharpe']:.2f}")
        print(f"      Model A Sharpe:    {individual_results[0]['sharpe']:.2f} (weight={individual_results[0]['weight']:.0%})")
        print(f"      Model B Sharpe:    {individual_results[1]['sharpe']:.2f} (weight={individual_results[1]['weight']:.0%})")
        print(f"      Max Drawdown:      {ensemble_results['max_dd']:.1f}%")
        print(f"      Actions:           L:{ensemble_results['pct_long']:.0f}% S:{ensemble_results['pct_short']:.0f}% H:{ensemble_results['pct_hold']:.0f}%")
        print(f"      Model Agreement:   {ensemble_results['avg_agreement']:.1%}")
        print(f"      Model Correlation: {model_correlation:.2f}")

        # WARNING if models are anti-correlated
        if model_correlation < 0:
            print(f"      [!] WARNING: Negative correlation! Ensemble may hurt performance.")

    # Summary
    if fold_results:
        ensemble_sharpes = [r['ensemble_sharpe'] for r in fold_results]
        model_a_sharpes = [r['model_a_sharpe'] for r in fold_results]
        model_b_sharpes = [r['model_b_sharpe'] for r in fold_results]
        hold_pcts = [r['pct_hold'] for r in fold_results]
        agreements = [r['avg_agreement'] for r in fold_results]

        summary = {
            "strategy": "Ensemble 70% A + 30% B",
            "n_folds": len(fold_results),
            "ensemble": {
                "avg_sharpe": np.mean(ensemble_sharpes),
                "std_sharpe": np.std(ensemble_sharpes),
                "min_sharpe": np.min(ensemble_sharpes),
                "max_sharpe": np.max(ensemble_sharpes),
            },
            "model_a": {
                "avg_sharpe": np.mean(model_a_sharpes),
                "std_sharpe": np.std(model_a_sharpes),
            },
            "model_b": {
                "avg_sharpe": np.mean(model_b_sharpes),
                "std_sharpe": np.std(model_b_sharpes),
            },
            "avg_hold_pct": np.mean(hold_pcts),
            "avg_agreement": np.mean(agreements),
            "fold_results": fold_results,
        }

        print("\n" + "="*70)
        print("  ENSEMBLE SUMMARY")
        print("="*70)
        print(f"\n  Ensemble Performance:")
        print(f"    Avg Sharpe:  {summary['ensemble']['avg_sharpe']:.2f} +/- {summary['ensemble']['std_sharpe']:.2f}")
        print(f"    Range:       [{summary['ensemble']['min_sharpe']:.2f}, {summary['ensemble']['max_sharpe']:.2f}]")
        print(f"\n  Individual Models:")
        print(f"    Model A:     {summary['model_a']['avg_sharpe']:.2f} +/- {summary['model_a']['std_sharpe']:.2f}")
        print(f"    Model B:     {summary['model_b']['avg_sharpe']:.2f} +/- {summary['model_b']['std_sharpe']:.2f}")
        print(f"\n  Behavior:")
        print(f"    Avg HOLD %:  {summary['avg_hold_pct']:.0f}%")
        print(f"    Agreement:   {summary['avg_agreement']:.1%}")

        # Comparison with expected
        print("\n" + "-"*50)
        print("  COMPARISON VS EXPECTED (from agent analysis):")
        print("-"*50)
        print(f"    Expected Sharpe:  ~1.95")
        print(f"    Actual Sharpe:    {summary['ensemble']['avg_sharpe']:.2f}")
        print(f"    Expected Var:     ~1.8")
        print(f"    Actual Std:       {summary['ensemble']['std_sharpe']:.2f}")

        # Verdict
        improvement_vs_a = (summary['ensemble']['avg_sharpe'] - summary['model_a']['avg_sharpe']) / (summary['model_a']['avg_sharpe'] + 1e-8) * 100
        variance_reduction = (summary['model_b']['std_sharpe'] - summary['ensemble']['std_sharpe']) / (summary['model_b']['std_sharpe'] + 1e-8) * 100

        print(f"\n  VERDICT:")
        print(f"    Sharpe improvement vs A: {improvement_vs_a:+.1f}%")
        print(f"    Variance reduction vs B: {variance_reduction:+.1f}%")

        if summary['ensemble']['avg_sharpe'] > 1.5 and summary['ensemble']['std_sharpe'] < 2.5:
            print(f"    STATUS: SUCCESS - Good risk/return balance")
        else:
            print(f"    STATUS: REVIEW - May need weight adjustment")

        return summary

    return {"error": "No folds completed"}


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ensemble Training')
    parser.add_argument('--folds', '-f', type=int, default=5,
                        help='Number of folds (default: 5)')
    parser.add_argument('--timesteps', '-t', type=int, default=60_000,
                        help='Training timesteps per model (default: 60000)')
    parser.add_argument('--weights', '-w', nargs=2, type=float, default=None,
                        help='Custom weights for Model A and B (e.g., --weights 0.6 0.4)')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                        help='Output directory')

    args = parser.parse_args()

    output_dir = ROOT / args.output
    output_dir.mkdir(exist_ok=True)

    # Run ensemble training
    weights = tuple(args.weights) if args.weights else None
    results = run_walkforward_ensemble(
        n_folds=args.folds,
        timesteps=args.timesteps,
        weights=weights,
    )

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f"ensemble_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results saved: {output_file}")
    print("\n" + "="*70)
    print("  DONE")
    print("="*70)


if __name__ == "__main__":
    main()
