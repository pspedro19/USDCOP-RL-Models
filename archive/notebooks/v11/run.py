#!/usr/bin/env python
"""
USD/COP RL Trading System V11 - Main Execution Script
======================================================

Walk-forward training and backtesting with professional reporting.

Usage:
    python run.py                    # Run all folds
    python run.py --fold 0           # Run single fold
    python run.py --folds 0 1 2      # Run specific folds
    python run.py --quick            # Quick test (50k steps)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import gc
import time
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.monitor import Monitor

# Local imports
from config import (
    DATA_PATH, LOGS_DIR, MODELS_DIR, OUTPUTS_DIR,
    FEATURES_FOR_MODEL, RAW_RETURN_COL, COST_PER_TRADE,
    CONFIG, PPO_CONFIG, WALK_FORWARD_FOLDS, DEVICE
)
from src import (
    TradingEnvV11, BacktestReporter,
    EntropyScheduler, PositionMonitor,
    calculate_norm_stats, normalize_df_v11, analyze_regime_raw,
    Logger
)
from src.utils import calculate_wfe, classify_wfe, load_and_prepare_data

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)


def run_backtest_with_report(model, df, features, initial_balance):
    """Run backtest with professional reporting."""
    reporter = BacktestReporter(initial_balance)

    env = DummyVecEnv([lambda: Monitor(TradingEnvV11(
        df, features, len(df), initial_balance, COST_PER_TRADE, RAW_RETURN_COL
    ))])
    env = VecNormalize(env, norm_obs=False, norm_reward=False, clip_obs=5.0)
    env.training = False

    obs = env.reset()
    n_steps = min(len(df) - 10, 50000)

    for i in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)

        reporter.update(
            position=infos[0]['position'],
            portfolio=infos[0]['portfolio'],
            market_return=infos[0]['market_return']
        )

        if dones[0]:
            break

    reporter.finalize()
    env.close()

    return reporter


def run_fold(fold, df, features, log, timesteps=None):
    """Run training and backtesting for a single fold."""
    fold_id = fold['fold_id']
    timesteps = timesteps or CONFIG['timesteps_per_fold']

    log.separator()
    log.log(f"FOLD {fold_id}: {fold['train_start']} to {fold['train_end']}")
    log.separator()

    # Create data splits
    train_mask = (df.index >= fold['train_start']) & (df.index <= fold['train_end'])
    test_mask = (df.index >= fold['test_start']) & (df.index <= fold['test_end'])

    df_train_raw = df[train_mask].copy()
    df_test_raw = df[test_mask].copy()

    if len(df_train_raw) < 1000 or len(df_test_raw) < 100:
        log.log("SKIP: Insufficient data")
        return None

    log.log(f"Train: {len(df_train_raw):,} rows")
    log.log(f"Test: {len(df_test_raw):,} rows")

    # Analyze regimes (RAW data)
    train_regime = analyze_regime_raw(df_train_raw, "Train", log)
    test_regime = analyze_regime_raw(df_test_raw, "Test", log)

    # V11: Normalize with raw returns preserved
    norm_stats = calculate_norm_stats(df_train_raw, features)
    df_train = normalize_df_v11(df_train_raw, norm_stats, features, RAW_RETURN_COL)
    df_test = normalize_df_v11(df_test_raw, norm_stats, features, RAW_RETURN_COL)

    # Verify raw returns
    log.log(f"Raw ret stats (train): mean={df_train[RAW_RETURN_COL].mean():.8f}")
    log.log(f"Normalized log_ret_5m (train): mean={df_train['log_ret_5m'].mean():.4f}")

    # Create environment
    def make_env(data, ep_len):
        def _init():
            return Monitor(TradingEnvV11(
                data, features, ep_len,
                CONFIG['initial_balance'], COST_PER_TRADE, RAW_RETURN_COL
            ))
        return _init

    train_env = DummyVecEnv([make_env(df_train, CONFIG['episode_length'])])
    train_env = VecNormalize(
        train_env, norm_obs=False, norm_reward=True,
        clip_obs=5.0, clip_reward=10.0
    )

    # Create model
    model = PPO(
        'MlpPolicy', train_env,
        **PPO_CONFIG,
        verbose=0,
        device=DEVICE,
        seed=42 + fold_id
    )

    # Train
    log.log(f"Training {timesteps:,} steps...")
    train_start = time.time()

    callbacks = CallbackList([
        EntropyScheduler(0.05, 0.01),
        PositionMonitor(100000, log)
    ])

    model.learn(
        total_timesteps=timesteps,
        callback=callbacks,
        progress_bar=True
    )

    train_time = time.time() - train_start
    log.log(f"Training time: {train_time/60:.1f} min")

    train_env.close()

    # Backtest
    log.log("Backtesting...")
    train_reporter = run_backtest_with_report(
        model, df_train, features, CONFIG['initial_balance']
    )
    test_reporter = run_backtest_with_report(
        model, df_test, features, CONFIG['initial_balance']
    )

    train_metrics = train_reporter.get_metrics()
    test_metrics = test_reporter.get_metrics()

    # Calculate WFE
    wfe = calculate_wfe(train_metrics['sharpe'], test_metrics['sharpe'])

    # Log results
    log.blank()
    log.log(f"RESULTS FOLD {fold_id}:")
    log.log(f"{'Metric':<20} {'Train':>12} {'Test':>12}")
    log.log("-" * 46)
    log.log(f"{'Return':<20} {train_metrics['total_return']:>+11.2f}% {test_metrics['total_return']:>+11.2f}%")
    log.log(f"{'Sharpe':<20} {train_metrics['sharpe']:>12.2f} {test_metrics['sharpe']:>12.2f}")
    log.log(f"{'Max DD':<20} {train_metrics['max_drawdown']:>11.2f}% {test_metrics['max_drawdown']:>11.2f}%")
    log.log(f"{'Win Rate':<20} {train_metrics['win_rate']:>11.1f}% {test_metrics['win_rate']:>11.1f}%")
    log.log(f"{'Profit Factor':<20} {train_metrics['profit_factor']:>12.2f} {test_metrics['profit_factor']:>12.2f}")
    log.log(f"{'Mean Position':<20} {train_metrics['mean_position']:>+12.3f} {test_metrics['mean_position']:>+12.3f}")
    log.log(f"{'Trades':<20} {train_metrics['n_trades']:>12} {test_metrics['n_trades']:>12}")
    log.log("-" * 46)
    log.log(f"{'WFE':<20} {wfe:>12.1%}")

    # Save model
    model_path = MODELS_DIR / f'ppo_usdcop_v11_fold{fold_id}.zip'
    model.save(model_path)
    log.log(f"Model saved: {model_path}")

    # Build result
    result = {
        'fold_id': fold_id,
        'train_return': train_metrics['total_return'],
        'train_sharpe': train_metrics['sharpe'],
        'train_max_dd': train_metrics['max_drawdown'],
        'train_win_rate': train_metrics['win_rate'],
        'train_pf': train_metrics['profit_factor'],
        'train_mean_pos': train_metrics['mean_position'],
        'test_return': test_metrics['total_return'],
        'test_sharpe': test_metrics['sharpe'],
        'test_max_dd': test_metrics['max_drawdown'],
        'test_win_rate': test_metrics['win_rate'],
        'test_pf': test_metrics['profit_factor'],
        'test_mean_pos': test_metrics['mean_position'],
        'wfe': wfe,
        'train_regime': train_regime['regime'],
        'test_regime': test_regime['regime'],
    }

    # Cleanup
    del model
    gc.collect()
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    return result


def create_visualizations(results_df, avg_metrics, log):
    """Create and save visualizations."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    x = results_df['fold_id'].values
    width = 0.35

    # 1. Sharpe by fold
    ax = axes[0, 0]
    ax.bar(x - width/2, results_df['train_sharpe'], width, label='Train', color='#2ecc71')
    ax.bar(x + width/2, results_df['test_sharpe'], width, label='Test', color='#e74c3c')
    ax.axhline(0, color='black')
    ax.axhline(1, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Sharpe')
    ax.set_title('Sharpe Ratio by Fold', fontweight='bold')
    ax.legend()

    # 2. WFE
    ax = axes[0, 1]
    colors = [
        '#2ecc71' if w >= 0.5 else '#f39c12' if w >= 0.3 else '#e74c3c'
        for w in results_df['wfe']
    ]
    ax.bar(results_df['fold_id'], results_df['wfe'], color=colors)
    ax.axhline(0.5, color='green', linestyle='--', label='Target 50%')
    ax.axhline(0.3, color='orange', linestyle='--', label='Min 30%')
    ax.set_xlabel('Fold')
    ax.set_ylabel('WFE')
    ax.set_title('Walk-Forward Efficiency', fontweight='bold')
    ax.legend()

    # 3. Returns
    ax = axes[0, 2]
    ax.bar(x - width/2, results_df['train_return'], width, label='Train', color='#2ecc71')
    ax.bar(x + width/2, results_df['test_return'], width, label='Test', color='#e74c3c')
    ax.axhline(0, color='black')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Return (%)')
    ax.set_title('Returns by Fold', fontweight='bold')
    ax.legend()

    # 4. Win Rate
    ax = axes[1, 0]
    ax.bar(x - width/2, results_df['train_win_rate'], width, label='Train', color='#2ecc71')
    ax.bar(x + width/2, results_df['test_win_rate'], width, label='Test', color='#e74c3c')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Fold', fontweight='bold')
    ax.legend()

    # 5. Position Bias
    ax = axes[1, 1]
    ax.bar(x - width/2, results_df['train_mean_pos'], width, label='Train', color='#2ecc71')
    ax.bar(x + width/2, results_df['test_mean_pos'], width, label='Test', color='#e74c3c')
    ax.axhline(0, color='black')
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-0.3, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Fold')
    ax.set_ylabel('Mean Position')
    ax.set_title('Position Bias', fontweight='bold')
    ax.set_ylim(-1, 1)
    ax.legend()

    # 6. Summary
    ax = axes[1, 2]
    ax.axis('off')
    summary = f"""
V11 SUMMARY
{'='*30}

CRITICAL FIX APPLIED:
- V10 Bug: Used normalized returns
- V11 Fix: Uses RAW returns for portfolio

Folds: {len(results_df)}
Avg Train Return: {avg_metrics['train_return']:+.2f}%
Avg Test Return: {avg_metrics['test_return']:+.2f}%
Avg Train Sharpe: {avg_metrics['train_sharpe']:.2f}
Avg Test Sharpe: {avg_metrics['test_sharpe']:.2f}
Avg WFE: {avg_metrics['wfe']:.1%}

Classification: {avg_metrics['wfe_class']}
"""
    ax.text(
        0.1, 0.9, summary,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.suptitle(
        'USD/COP RL V11 - Walk-Forward Results (RAW RETURNS FIX)',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.tight_layout()

    output_path = OUTPUTS_DIR / 'results_v11.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.log(f"Saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='USD/COP RL V11 Training')
    parser.add_argument('--fold', type=int, help='Run single fold')
    parser.add_argument('--folds', type=int, nargs='+', help='Run specific folds')
    parser.add_argument('--quick', action='store_true', help='Quick test (50k steps)')
    args = parser.parse_args()

    # Initialize logger
    log = Logger(LOGS_DIR)
    log.header("USD/COP RL TRADING SYSTEM V11")
    log.log(f"Device: {DEVICE}")
    log.log(f"Session: {log.session}")

    # Load data
    log.subheader("LOADING DATA")
    log.log(f"Data path: {DATA_PATH}")

    if not DATA_PATH.exists():
        log.log(f"ERROR: Data file not found!", "ERROR")
        return

    df, features = load_and_prepare_data(str(DATA_PATH), FEATURES_FOR_MODEL)
    log.log(f"Shape: {df.shape[0]:,} x {df.shape[1]}")
    log.log(f"Features: {len(features)}/{len(FEATURES_FOR_MODEL)}")
    log.log(f"Range: {df.index.min()} to {df.index.max()}")

    # Determine which folds to run
    if args.fold is not None:
        folds_to_run = [f for f in WALK_FORWARD_FOLDS if f['fold_id'] == args.fold]
    elif args.folds:
        folds_to_run = [f for f in WALK_FORWARD_FOLDS if f['fold_id'] in args.folds]
    else:
        folds_to_run = WALK_FORWARD_FOLDS

    timesteps = 50_000 if args.quick else None

    # Run folds
    log.header("WALK-FORWARD TRAINING")
    all_results = []

    for fold in folds_to_run:
        result = run_fold(fold, df, features, log, timesteps)
        if result:
            all_results.append(result)

    if not all_results:
        log.log("No results to aggregate", "WARN")
        return

    # Aggregate results
    log.header("AGGREGATE RESULTS")

    results_df = pd.DataFrame(all_results)

    avg_metrics = {
        'train_return': results_df['train_return'].mean(),
        'test_return': results_df['test_return'].mean(),
        'train_sharpe': results_df['train_sharpe'].mean(),
        'test_sharpe': results_df['test_sharpe'].mean(),
        'wfe': results_df['wfe'].mean(),
    }
    avg_metrics['wfe_class'] = classify_wfe(avg_metrics['wfe'])

    # Print summary
    print("\n" + "="*80)
    print("USD/COP RL V11 - WALK-FORWARD RESULTS".center(80))
    print("="*80)

    print(f"\n{'Fold':<6} {'Train Ret':>10} {'Test Ret':>10} {'Train Shp':>10} {'Test Shp':>10} {'WFE':>8}")
    print("-"*60)

    for _, row in results_df.iterrows():
        print(
            f"{row['fold_id']:<6} "
            f"{row['train_return']:>+9.2f}% "
            f"{row['test_return']:>+9.2f}% "
            f"{row['train_sharpe']:>10.2f} "
            f"{row['test_sharpe']:>10.2f} "
            f"{row['wfe']:>7.1%}"
        )

    print("-"*60)
    print(f"\nAvg Train Return: {avg_metrics['train_return']:+.2f}%")
    print(f"Avg Test Return: {avg_metrics['test_return']:+.2f}%")
    print(f"Avg WFE: {avg_metrics['wfe']:.1%}")
    print(f"Classification: {avg_metrics['wfe_class']}")
    print("="*80)

    # Create visualizations
    create_visualizations(results_df, avg_metrics, log)

    # Save results
    results_path = OUTPUTS_DIR / 'results_v11.csv'
    results_df.to_csv(results_path, index=False)
    log.log(f"Results saved: {results_path}")

    log.header("V11 COMPLETE")
    log.log(f"Avg Test Return: {avg_metrics['test_return']:+.2f}%")
    log.log(f"Classification: {avg_metrics['wfe_class']}")


if __name__ == '__main__':
    main()
