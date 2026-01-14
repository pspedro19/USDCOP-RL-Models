#!/usr/bin/env python3
"""
USD/COP RL Trading System - Walk-Forward Validation V2
=======================================================

MEJORAS V2.1 (15-Fold + Multi-Seed + Robust Statistics):

1. 15-Fold Support para 5min:
   - Dataset 5min: 84,671 filas (soporta 15+ folds)
   - Auto-ajuste de configuración por timeframe
   - Train: 60 días, Test: 20 días, Gap: 5 días (5min)

2. Multi-Seed Training:
   - Entrenar cada fold con múltiples seeds (default: 3)
   - Promediar resultados por fold
   - Reducir varianza de estimación

3. Estadísticas Robustas:
   - IC 95% con bootstrap (10,000 samples)
   - Coeficiente de Variación (CV)
   - Test de normalidad Shapiro-Wilk
   - Comparación con benchmark (Sharpe = 1.0)

4. GO/NO-GO Automático:
   - GO: Sharpe > 1.0, IC positivo, CV < 80%
   - MARGINAL: Sharpe > 0.5, CV < 100%
   - NO-GO: Sharpe < 0.5 o CV > 150%

5. Hiperparametros anti-overfitting:
   - Red mas pequena (32,32) en vez de (64,64)
   - Entropy coefficient alto (0.15) para mas exploracion
   - Learning rate bajo (1e-4) para convergencia suave
   - Early stopping por fold

Author: Claude Code
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse

warnings.filterwarnings('ignore')

# Setup paths
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "config"))

import numpy as np
import pandas as pd
from scipy import stats

# Imports
from environment_v19 import TradingEnvironmentV19
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """Early stopping basado en reward promedio."""

    def __init__(self, check_freq: int = 5000, patience: int = 3, min_improvement: float = 0.01):
        super().__init__()
        self.check_freq = check_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.episode_rewards = []

    def _on_step(self) -> bool:
        # Collect episode rewards
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep['r'] for ep in self.model.ep_info_buffer]
            self.episode_rewards.extend(rewards)

        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 10:
            mean_reward = np.mean(self.episode_rewards[-50:])

            if mean_reward > self.best_mean_reward + self.min_improvement:
                self.best_mean_reward = mean_reward
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.patience:
                print(f" [Early stop at {self.n_calls} steps]", end="")
                return False

        return True


def calculate_metrics(returns: np.ndarray, portfolios: List[List[float]],
                      bars_per_day: int) -> Dict:
    """Calcular metricas de trading."""
    n_days = len(returns) // bars_per_day

    if n_days < 2:
        return {
            'sharpe': 0.0, 'sortino': 0.0, 'max_dd': 0.0,
            'total_return': 0.0, 'win_rate': 0.0
        }

    # Aggregate to daily
    daily_returns = returns[:n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)

    # Sharpe
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)

    # Sortino
    downside = daily_returns[daily_returns < 0]
    sortino = np.mean(daily_returns) / (np.std(downside) + 1e-8) * np.sqrt(252) if len(downside) > 0 else 0

    # Max Drawdown
    max_dds = []
    for pv in portfolios:
        pv = np.array(pv)
        peak = np.maximum.accumulate(pv)
        dd = (peak - pv) / (peak + 1e-10)
        max_dds.append(np.max(dd))
    max_dd = np.mean(max_dds) * 100

    # Total return
    total_return = (np.mean([pv[-1] for pv in portfolios]) / 10000 - 1) * 100

    # Win rate
    win_rate = (daily_returns > 0).mean() * 100

    return {
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'total_return': float(total_return),
        'win_rate': float(win_rate),
    }


def evaluate_model(model, env, n_episodes: int = 5, bars_per_day: int = 20) -> Dict:
    """Evaluar modelo en environment."""
    all_returns = []
    all_actions = []
    all_portfolios = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_returns = []
        episode_actions = []
        portfolio_values = [10000]
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_returns.append(info.get('step_return', 0))
            episode_actions.append(float(action[0] if hasattr(action, '__len__') else action))
            portfolio_values.append(info.get('portfolio', portfolio_values[-1]))

            done = terminated or truncated

        all_returns.extend(episode_returns)
        all_actions.extend(episode_actions)
        all_portfolios.append(portfolio_values)

    metrics = calculate_metrics(np.array(all_returns), all_portfolios, bars_per_day)

    # Action distribution
    actions = np.array(all_actions)
    # Action classification - THRESHOLD 0.10 (was 0.15, changed to account for regime multiplier)
    ACTION_THRESHOLD = 0.10
    metrics['pct_long'] = float((actions > ACTION_THRESHOLD).mean() * 100)
    metrics['pct_short'] = float((actions < -ACTION_THRESHOLD).mean() * 100)
    metrics['pct_hold'] = float((np.abs(actions) < ACTION_THRESHOLD).mean() * 100)

    return metrics


def bootstrap_confidence_interval(data: np.ndarray, n_bootstrap: int = 10000,
                                  confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcular intervalo de confianza con bootstrap.

    Returns:
        (lower_bound, upper_bound)
    """
    bootstrap_means = []
    n_samples = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_means.append(np.mean(sample))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return lower, upper


def train_fold_with_seeds(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    n_seeds: int,
    timesteps_per_fold: int,
    bars_per_day: int,
    use_anti_overfit: bool,
    fold_num: int,
) -> Dict:
    """
    Entrenar un fold con múltiples seeds y promediar resultados.

    Returns:
        Dict con métricas agregadas
    """
    seed_results = []

    for seed_idx in range(n_seeds):
        seed = 42 + fold_num * 100 + seed_idx

        print(f"    Seed {seed_idx + 1}/{n_seeds} (seed={seed})...", end=" ", flush=True)

        # Set seed
        np.random.seed(seed)

        # Create environments
        episode_length = min(bars_per_day * 10, len(train_df) - 10)

        train_env = TradingEnvironmentV19(
            df=train_df,
            initial_balance=10000,
            max_position=1.0,
            episode_length=episode_length,
            feature_columns=feature_cols,
            verbose=0,
        )

        test_env = TradingEnvironmentV19(
            df=test_df,
            initial_balance=10000,
            max_position=1.0,
            episode_length=min(episode_length, len(test_df) - 10),
            feature_columns=feature_cols,
            verbose=0,
        )

        # Create model with seed
        if use_anti_overfit:
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=1e-4,
                n_steps=2048,
                batch_size=128,
                n_epochs=5,
                gamma=0.99,
                ent_coef=0.15,
                clip_range=0.1,
                max_grad_norm=0.3,
                policy_kwargs={"net_arch": [32, 32]},
                seed=seed,
                verbose=0,
            )
            early_stop = EarlyStoppingCallback(check_freq=10000, patience=3)
            callbacks = [early_stop]
        else:
            model = PPO(
                policy="MlpPolicy",
                env=train_env,
                learning_rate=3e-4,
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                ent_coef=0.05,
                clip_range=0.2,
                policy_kwargs={"net_arch": [64, 64]},
                seed=seed,
                verbose=0,
            )
            callbacks = []

        # Train
        train_start_time = datetime.now()
        model.learn(total_timesteps=timesteps_per_fold, callback=callbacks, progress_bar=False)
        train_duration = (datetime.now() - train_start_time).total_seconds()

        # Evaluate
        test_metrics = evaluate_model(model, test_env, n_episodes=5, bars_per_day=bars_per_day)
        train_metrics = evaluate_model(model, train_env, n_episodes=3, bars_per_day=bars_per_day)

        seed_results.append({
            'seed': seed,
            'train_sharpe': train_metrics['sharpe'],
            'test_sharpe': test_metrics['sharpe'],
            'test_sortino': test_metrics['sortino'],
            'test_max_dd': test_metrics['max_dd'],
            'test_return': test_metrics['total_return'],
            'test_win_rate': test_metrics['win_rate'],
            'pct_long': test_metrics['pct_long'],
            'pct_short': test_metrics['pct_short'],
            'pct_hold': test_metrics['pct_hold'],
            'train_duration': train_duration,
        })

        print(f"Sharpe={test_metrics['sharpe']:.3f}")

    # Aggregate results across seeds
    aggregated = {
        'train_sharpe': np.mean([r['train_sharpe'] for r in seed_results]),
        'test_sharpe': np.mean([r['test_sharpe'] for r in seed_results]),
        'test_sharpe_std': np.std([r['test_sharpe'] for r in seed_results]),
        'test_sortino': np.mean([r['test_sortino'] for r in seed_results]),
        'test_max_dd': np.mean([r['test_max_dd'] for r in seed_results]),
        'test_return': np.mean([r['test_return'] for r in seed_results]),
        'test_win_rate': np.mean([r['test_win_rate'] for r in seed_results]),
        'pct_long': np.mean([r['pct_long'] for r in seed_results]),
        'pct_short': np.mean([r['pct_short'] for r in seed_results]),
        'pct_hold': np.mean([r['pct_hold'] for r in seed_results]),
        'train_duration': np.mean([r['train_duration'] for r in seed_results]),
        'seed_results': seed_results,
    }

    return aggregated


def calculate_robust_statistics(test_sharpes: List[float]) -> Dict:
    """
    Calcular estadísticas robustas con bootstrap.

    Returns:
        Dict con métricas estadísticas
    """
    sharpes = np.array(test_sharpes)
    n_folds = len(sharpes)

    # Basic stats
    mean_sharpe = np.mean(sharpes)
    std_sharpe = np.std(sharpes, ddof=1)  # Sample std

    # Coefficient of Variation (CV)
    cv = (std_sharpe / abs(mean_sharpe)) * 100 if mean_sharpe != 0 else np.inf

    # Bootstrap CI (95%)
    ci_lower, ci_upper = bootstrap_confidence_interval(sharpes, n_bootstrap=10000, confidence=0.95)

    # Shapiro-Wilk test for normality (si n >= 3)
    if n_folds >= 3:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sharpes)
            is_normal = shapiro_p > 0.05
        except:
            shapiro_stat, shapiro_p, is_normal = None, None, None
    else:
        shapiro_stat, shapiro_p, is_normal = None, None, None

    # T-test vs benchmark (Sharpe = 1.0)
    benchmark_sharpe = 1.0
    if n_folds >= 2:
        t_stat, t_p = stats.ttest_1samp(sharpes, benchmark_sharpe)
        beats_benchmark = mean_sharpe > benchmark_sharpe and t_p < 0.05
    else:
        t_stat, t_p = None, None
        beats_benchmark = mean_sharpe > benchmark_sharpe

    return {
        'mean': mean_sharpe,
        'std': std_sharpe,
        'cv': cv,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        'ci_95_width': ci_upper - ci_lower,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'is_normal': is_normal,
        't_stat': t_stat,
        't_p': t_p,
        'beats_benchmark': beats_benchmark,
        'benchmark_sharpe': benchmark_sharpe,
    }


def make_go_nogo_decision(stats: Dict, max_dd: float) -> Tuple[str, str, List[Tuple]]:
    """
    Tomar decisión GO/NO-GO basada en estadísticas robustas.

    Returns:
        (decision, recommendation, checks)
    """
    mean_sharpe = stats['mean']
    cv = stats['cv']
    ci_lower = stats['ci_95_lower']
    ci_width = stats['ci_95_width']
    beats_benchmark = stats['beats_benchmark']

    checks = []

    # Check 1: Mean Sharpe > 1.0 (GO threshold)
    check1 = mean_sharpe > 1.0
    checks.append(("Mean Sharpe > 1.0", check1, f"{mean_sharpe:.3f}"))

    # Check 2: IC 95% completamente positivo
    check2 = ci_lower > 0.0
    checks.append(("CI 95% Lower > 0.0", check2, f"{ci_lower:.3f}"))

    # Check 3: Coefficient of Variation < 80%
    check3 = cv < 80
    checks.append(("CV < 80%", check3, f"{cv:.1f}%"))

    # Check 4: IC 95% estrecho (< 1.0)
    check4 = ci_width < 1.0
    checks.append(("CI Width < 1.0", check4, f"{ci_width:.3f}"))

    # Check 5: Max Drawdown < 25%
    check5 = max_dd < 25
    checks.append(("Max DD < 25%", check5, f"{max_dd:.1f}%"))

    # Check 6: Beats benchmark (Sharpe = 1.0)
    check6 = beats_benchmark
    checks.append(("Beats Benchmark (p<0.05)", check6, f"{'Yes' if check6 else 'No'}"))

    # Decision logic
    n_passed = sum(1 for _, passed, _ in checks if passed)
    n_total = len(checks)

    if n_passed == n_total and mean_sharpe > 1.0:
        decision = "GO"
        recommendation = "✓ Excelente. Proceder a producción (500k-1M steps)"
    elif n_passed >= 5 and mean_sharpe > 0.8 and cv < 100:
        decision = "CONDITIONAL GO"
        recommendation = "✓ Bueno. Proceder con monitoreo continuo"
    elif n_passed >= 4 and mean_sharpe > 0.5 and cv < 150:
        decision = "MARGINAL"
        recommendation = "⚠ Marginal. Considerar ajustes antes de producción"
    else:
        decision = "NO-GO"
        recommendation = "✗ NO proceder. Mejorar modelo o features"

    return decision, recommendation, checks


def run_walk_forward_validation_v2(
    data_path: str,
    timeframe: str = "5min",
    n_folds: int = 15,
    n_seeds: int = 3,
    timesteps_per_fold: int = 100_000,
    bars_per_day: Optional[int] = None,
    use_anti_overfit: bool = True,
):
    """
    Walk-Forward Validation V2 con 15 folds y multi-seed.

    MEJORAS V2.1:
    - Soporta 15 folds con dataset 5min (84,671 filas)
    - Multi-seed training para reducir varianza
    - Estadísticas robustas con bootstrap CI 95%
    - GO/NO-GO automático

    Args:
        data_path: Path al dataset
        timeframe: "5min" o "15min"
        n_folds: Número de folds (15 para 5min, 5 para 15min)
        n_seeds: Número de seeds por fold (default: 3)
        timesteps_per_fold: Timesteps de entrenamiento por fold
        bars_per_day: Barras por día (auto-detectado si None)
        use_anti_overfit: Usar configuración anti-overfitting
    """

    print("=" * 80)
    print(f"{'WALK-FORWARD VALIDATION V2.1 (15-Fold + Multi-Seed)':^80}")
    print(f"{'Validación Robusta para Producción':^80}")
    print("=" * 80)

    # Auto-detect bars_per_day
    if bars_per_day is None:
        bars_per_day = 60 if timeframe == "5min" else 20

    # Load data
    print(f"\n[1/5] Loading data...")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows ({timeframe})")

    # Prepare features
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Add required columns
    if 'close_return' not in df.columns:
        df['close_return'] = df['close'].pct_change().fillna(0)
    if 'volatility_pct' not in df.columns:
        df['volatility_pct'] = df['close_return'].rolling(20).std().fillna(0.01)

    # Fill NaN
    for col in feature_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(0)

    print(f"  Features: {len(feature_cols)}")

    # Expanding window fold calculation
    n_samples = len(df)

    # Configuración dinámica por timeframe
    if timeframe == "5min":
        # 5min: 60 bars/day
        # Train: 60 days = 3,600 bars
        # Test: 20 days = 1,200 bars
        # Gap: 5 days = 300 bars
        train_days = 60
        test_days = 20
        gap_days = 5
    else:
        # 15min: 20 bars/day
        # Train: 90 days = 1,800 bars
        # Test: 30 days = 600 bars
        # Gap: 10 days = 200 bars
        train_days = 90
        test_days = 30
        gap_days = 10

    train_bars = train_days * bars_per_day
    test_bars = test_days * bars_per_day
    gap_bars = gap_days * bars_per_day

    # Calculate minimum samples needed
    min_samples_needed = train_bars + gap_bars + test_bars * n_folds

    if n_samples < min_samples_needed:
        print(f"\n  WARNING: Dataset too small for {n_folds} folds")
        print(f"    Available: {n_samples:,} bars")
        print(f"    Required:  {min_samples_needed:,} bars")
        print(f"    Reducing folds to fit...")
        n_folds = max(3, (n_samples - train_bars - gap_bars) // test_bars)
        print(f"    New n_folds: {n_folds}")

    print(f"\n[2/5] Setting up {n_folds} EXPANDING folds with {n_seeds} seeds each...")
    print(f"  Train: {train_days} days ({train_bars:,} bars)")
    print(f"  Test:  {test_days} days ({test_bars:,} bars)")
    print(f"  Gap:   {gap_days} days ({gap_bars} bars)")
    print(f"  Total seeds to train: {n_folds * n_seeds}")

    if use_anti_overfit:
        print(f"\n  ANTI-OVERFIT CONFIG:")
        print(f"    - Network: [32, 32] (smaller)")
        print(f"    - Entropy: 0.15 (higher exploration)")
        print(f"    - LR: 1e-4 (slower learning)")
        print(f"    - Early stopping: enabled")

    # Run Walk-Forward
    print(f"\n[3/5] Running Walk-Forward Validation...")
    print("-" * 80)

    fold_results = []

    for fold in range(n_folds):
        print(f"\n  FOLD {fold + 1}/{n_folds}")
        print("  " + "-" * 40)

        # EXPANDING: train usa desde inicio hasta test_start
        test_start = train_bars + gap_bars + fold * test_bars
        test_end = test_start + test_bars

        if test_end > n_samples:
            print(f"  [SKIP] Not enough data for fold {fold + 1}")
            break

        train_start = 0
        train_end = test_start - gap_bars

        print(f"  Train: {train_start:,} - {train_end:,} ({train_end - train_start:,} bars)")
        print(f"  Test:  {test_start:,} - {test_end:,} ({test_end - test_start:,} bars)")

        # Create datasets
        train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
        test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

        # Train with multiple seeds
        fold_result = train_fold_with_seeds(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            n_seeds=n_seeds,
            timesteps_per_fold=timesteps_per_fold,
            bars_per_day=bars_per_day,
            use_anti_overfit=use_anti_overfit,
            fold_num=fold,
        )

        # Add metadata
        fold_result['fold'] = fold + 1
        fold_result['train_size'] = train_end - train_start
        fold_result['test_size'] = test_end - test_start

        fold_results.append(fold_result)

        # Print fold results
        overfit_flag = "[OVERFIT]" if fold_result['train_sharpe'] > fold_result['test_sharpe'] + 1 else ""
        print(f"\n  Fold Results (avg across {n_seeds} seeds):")
        print(f"    Train Sharpe: {fold_result['train_sharpe']:.3f}")
        print(f"    Test Sharpe:  {fold_result['test_sharpe']:.3f} ± {fold_result['test_sharpe_std']:.3f}  {overfit_flag}")
        print(f"    Test MaxDD:   {fold_result['test_max_dd']:.1f}%")
        print(f"    Actions:      L:{fold_result['pct_long']:.0f}% S:{fold_result['pct_short']:.0f}% H:{fold_result['pct_hold']:.0f}%")

    # Summary
    print("\n" + "=" * 80)
    print(f"{'WALK-FORWARD VALIDATION V2.1 RESULTS':^80}")
    print("=" * 80)

    if not fold_results:
        print("\nERROR: No folds completed")
        return None

    # Calculate statistics
    test_sharpes = [r['test_sharpe'] for r in fold_results]
    train_sharpes = [r['train_sharpe'] for r in fold_results]
    test_max_dds = [r['test_max_dd'] for r in fold_results]

    avg_train_sharpe = np.mean(train_sharpes)
    avg_max_dd = np.mean(test_max_dds)
    max_max_dd = np.max(test_max_dds)
    overfit_gap = avg_train_sharpe - np.mean(test_sharpes)

    # ROBUST STATISTICS
    print(f"\n[4/5] Calculating Robust Statistics...")
    robust_stats = calculate_robust_statistics(test_sharpes)

    # Per-fold table
    print(f"\n{'Fold':>6} {'Train':>8} {'Test':>12} {'Test Sharpe':>14} {'±Std':>8} {'MaxDD':>10} {'Status':>10}")
    print(f"{'':>6} {'Size':>8} {'Size':>12} {'':>14} {'':>8} {'':>10} {'':>10}")
    print("-" * 78)

    for r in fold_results:
        status = "OK" if r['test_sharpe'] > 0 else "WARN"
        if r['train_sharpe'] > r['test_sharpe'] + 1.5:
            status = "OVERFIT"
        print(f"{r['fold']:>6} {r['train_size']:>8,} {r['test_size']:>12,} "
              f"{r['test_sharpe']:>14.3f} {r['test_sharpe_std']:>8.3f} "
              f"{r['test_max_dd']:>9.1f}% {status:>10}")

    print("-" * 78)

    # Robust statistics summary
    print(f"\n{'ROBUST STATISTICS (Bootstrap CI 95%, n=10,000)':^78}")
    print("-" * 78)
    print(f"  Mean Sharpe:          {robust_stats['mean']:>8.3f}")
    print(f"  Std Dev:              {robust_stats['std']:>8.3f}")
    print(f"  Coefficient of Var:   {robust_stats['cv']:>8.1f}%")
    print(f"  CI 95%:               [{robust_stats['ci_95_lower']:.3f}, {robust_stats['ci_95_upper']:.3f}]")
    print(f"  CI Width:             {robust_stats['ci_95_width']:>8.3f}")

    if robust_stats['shapiro_p'] is not None:
        normality = "Yes ✓" if robust_stats['is_normal'] else "No ✗"
        print(f"  Normal Distribution:  {normality:>8} (p={robust_stats['shapiro_p']:.3f})")

    if robust_stats['t_p'] is not None:
        beats = "Yes ✓" if robust_stats['beats_benchmark'] else "No ✗"
        print(f"  Beats Benchmark:      {beats:>8} (Sharpe > 1.0, p={robust_stats['t_p']:.3f})")

    print(f"  Avg Max DD:           {avg_max_dd:>8.1f}%")
    print(f"  Max Max DD:           {max_max_dd:>8.1f}%")
    print(f"  Overfit Gap:          {overfit_gap:>8.3f}")

    # GO/NO-GO Decision
    print("\n" + "=" * 80)
    print(f"{'GO / NO-GO DECISION':^80}")
    print("=" * 80)

    print(f"\n[5/5] Making GO/NO-GO Decision...")
    decision, recommendation, checks = make_go_nogo_decision(robust_stats, max_max_dd)

    print(f"\n{'Check':<35} {'Status':>10} {'Value':>15}")
    print("-" * 65)

    for check_name, passed, value in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:<35} {status:>10} {value:>15}")

    # Final decision
    n_passed = sum(1 for _, passed, _ in checks if passed)
    n_total = len(checks)

    print("\n" + "-" * 65)
    print(f"\n  CHECKS PASSED: {n_passed}/{n_total}")
    print(f"\n  DECISION: {decision}")
    print(f"\n  RECOMMENDATION:")
    print(f"    {recommendation}")

    print("\n" + "=" * 80)

    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.1_15fold_multiseed',
        'timeframe': timeframe,
        'config': {
            'n_folds': n_folds,
            'folds_completed': len(fold_results),
            'n_seeds': n_seeds,
            'timesteps_per_fold': timesteps_per_fold,
            'bars_per_day': bars_per_day,
            'train_days': train_days,
            'test_days': test_days,
            'gap_days': gap_days,
            'use_anti_overfit': use_anti_overfit,
            'window_type': 'expanding',
        },
        'anti_overfit_params': {
            'net_arch': [32, 32],
            'ent_coef': 0.15,
            'learning_rate': 1e-4,
            'clip_range': 0.1,
            'early_stopping': True,
        } if use_anti_overfit else None,
        'fold_results': fold_results,
        'robust_statistics': {
            'mean_sharpe': robust_stats['mean'],
            'std_sharpe': robust_stats['std'],
            'cv_percent': robust_stats['cv'],
            'ci_95_lower': robust_stats['ci_95_lower'],
            'ci_95_upper': robust_stats['ci_95_upper'],
            'ci_95_width': robust_stats['ci_95_width'],
            'shapiro_wilk_p': robust_stats['shapiro_p'],
            'is_normal': robust_stats['is_normal'],
            't_test_p': robust_stats['t_p'],
            'beats_benchmark': robust_stats['beats_benchmark'],
            'benchmark_sharpe': robust_stats['benchmark_sharpe'],
        },
        'summary': {
            'avg_test_sharpe': robust_stats['mean'],
            'std_test_sharpe': robust_stats['std'],
            'min_test_sharpe': np.min(test_sharpes),
            'max_test_sharpe': np.max(test_sharpes),
            'avg_train_sharpe': avg_train_sharpe,
            'avg_max_dd': avg_max_dd,
            'max_max_dd': max_max_dd,
            'overfit_gap': overfit_gap,
        },
        'decision': {
            'status': decision,
            'checks_passed': n_passed,
            'checks_total': n_total,
            'recommendation': recommendation,
            'checks': [{'name': name, 'passed': passed, 'value': value}
                      for name, passed, value in checks],
        }
    }

    # Save to appropriate path
    if timeframe == "5min" and n_folds >= 10:
        output_filename = f"walkforward_5min_15folds.json"
    else:
        output_filename = f"walkforward_{timeframe}_v2.json"

    output_path = ROOT / "outputs" / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Create visualization script suggestion
    print(f"\nNext steps:")
    print(f"  1. Review results: cat {output_path}")
    print(f"  2. Visualize distribution: python plot_walkforward_results.py {output_filename}")
    if decision == "GO" or decision == "CONDITIONAL GO":
        print(f"  3. Train production model: python train_v19_production.py --timeframe {timeframe}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Walk-Forward Validation V2.1 - 15 Folds + Multi-Seed + Robust Stats'
    )
    parser.add_argument('--timeframe', type=str, default='5min',
                       choices=['5min', '15min'],
                       help='Timeframe (default: 5min for 15 folds)')
    parser.add_argument('--folds', type=int, default=None,
                       help='Number of folds (default: 15 for 5min, 5 for 15min)')
    parser.add_argument('--seeds', type=int, default=3,
                       help='Number of seeds per fold (default: 3)')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Timesteps per fold (default: 100,000)')
    parser.add_argument('--no-anti-overfit', action='store_true',
                       help='Disable anti-overfit config')

    args = parser.parse_args()

    # Auto-select folds based on timeframe
    if args.folds is None:
        args.folds = 15 if args.timeframe == '5min' else 5

    # Select data path
    if args.timeframe == '15min':
        data_path = ROOT / "../../data/pipeline/07_output/datasets_15min/RL_DS3_MACRO_CORE_15MIN.csv"
    else:
        data_path = ROOT / "../../data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv"

    if not data_path.exists():
        print(f"ERROR: Dataset not found at {data_path}")
        print(f"Please generate dataset first.")
        sys.exit(1)

    run_walk_forward_validation_v2(
        data_path=str(data_path),
        timeframe=args.timeframe,
        n_folds=args.folds,
        n_seeds=args.seeds,
        timesteps_per_fold=args.timesteps,
        use_anti_overfit=not args.no_anti_overfit,
    )


if __name__ == "__main__":
    main()
