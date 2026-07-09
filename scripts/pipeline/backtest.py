#!/usr/bin/env python3
"""
Backtest Validation Script
==============================

Runs out-of-sample backtest on a trained PPO model.
Calculates key metrics: Win rate, Sharpe, Max DD, HOLD%.

Author: Pedro @ Lean Tech Solutions / Claude Code
Date: 2026-01-09
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple


def set_seed(seed: int = 42) -> None:
    """Set random seeds for deterministic backtest results."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO

# SSOT Config Helper
from src.config.config_helper import get_thresholds, get_costs

# Normalizer from core (SSOT parity with inference)
from src.core.normalizers.zscore_normalizer import ZScoreNormalizer

# Core Features for RL Models
CORE_FEATURES = [
    "log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z", "brent_change_1d",
    "rate_spread", "usdmxn_change_1d"
]
OBS_DIM = 15


def load_norm_stats(path: str) -> Dict:
    """Load normalization statistics."""
    with open(path, 'r') as f:
        return json.load(f)


def normalize_observation(
    features: np.ndarray,
    position: float,
    time_norm: float,
    normalizer: ZScoreNormalizer
) -> np.ndarray:
    """
    Normalize observation to match production ObservationBuilder.

    Uses ZScoreNormalizer for SSOT parity with inference engine.
    """
    obs = np.zeros(OBS_DIM, dtype=np.float32)

    # Normalize core features using ZScoreNormalizer (SSOT parity)
    for i, (fname, value) in enumerate(zip(CORE_FEATURES, features)):
        obs[i] = normalizer.normalize(fname, float(value))

    # State features (not normalized)
    obs[13] = np.clip(position, -1.0, 1.0)
    obs[14] = np.clip(time_norm, 0.0, 1.0)

    return np.nan_to_num(obs, nan=0.0)


def run_backtest(
    model_path: str,
    dataset_path: str,
    norm_stats_path: str,
    test_split: float = 0.85,
    threshold_long: float = None,  # From SSOT via get_thresholds()
    threshold_short: float = None,  # From SSOT via get_thresholds()
    transaction_cost_bps: float = None,  # From SSOT via get_costs()
    initial_balance: float = 10_000,
    verbose: bool = True
) -> Dict:
    """
    Run backtest on out-of-sample data.

    Args:
        model_path: Path to trained model (.zip)
        dataset_path: Path to dataset CSV
        norm_stats_path: Path to normalization stats JSON
        test_split: Fraction of data to use for test (after this point)
        threshold_long: Threshold for long signal
        threshold_short: Threshold for short signal
        transaction_cost_bps: Transaction cost in basis points
        initial_balance: Starting balance
        verbose: Print progress

    Returns:
        Dictionary with backtest metrics
    """
    # Load SSOT defaults if not explicitly provided
    if threshold_long is None or threshold_short is None:
        ssot_long, ssot_short = get_thresholds()
        threshold_long = threshold_long if threshold_long is not None else ssot_long
        threshold_short = threshold_short if threshold_short is not None else ssot_short
    if transaction_cost_bps is None:
        transaction_cost_bps, _ = get_costs()

    if verbose:
        print("="*70)
        print("BACKTEST VALIDATION")
        print("="*70)

    # Load model
    if verbose:
        print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path, device='cpu')

    # Load data
    if verbose:
        print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Create normalizer (SSOT parity with inference engine)
    normalizer = ZScoreNormalizer(stats_path=norm_stats_path)

    # Get test data (out-of-sample)
    test_start = int(len(df) * test_split)
    test_df = df.iloc[test_start:].reset_index(drop=True)

    if verbose:
        print(f"Test data: {len(test_df):,} rows ({(1-test_split)*100:.0f}% of data)")
        print(f"Date range: {test_df['timestamp'].iloc[0]} to {test_df['timestamp'].iloc[-1]}")

    # Extract features
    features = test_df[CORE_FEATURES].values.astype(np.float32)
    returns = test_df['log_ret_5m'].values.astype(np.float32)

    # Backtest variables
    balance = initial_balance
    position = 0.0
    peak_balance = initial_balance
    transaction_cost = transaction_cost_bps / 10000

    # Tracking
    equity_curve = [balance]
    positions = []
    trades = []
    actions_taken = []

    n_bars = len(test_df)
    n_long = 0
    n_short = 0
    n_hold = 0

    if verbose:
        print(f"\nRunning backtest on {n_bars:,} bars...")

    for i in range(1, n_bars):
        # Build observation (SSOT parity with inference)
        obs = normalize_observation(
            features=features[i-1],
            position=position,
            time_norm=i / n_bars,
            normalizer=normalizer
        )

        # Get model prediction
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        action_value = float(action[0])
        actions_taken.append(action_value)

        # Map action to target position using thresholds
        if action_value > threshold_long:
            target_position = 1.0  # Long
            n_long += 1
        elif action_value < threshold_short:
            target_position = -1.0  # Short
            n_short += 1
        else:
            target_position = 0.0  # Hold/Flat
            n_hold += 1

        # Calculate costs
        position_change = abs(target_position - position)
        cost = position_change * transaction_cost * balance

        if position_change > 0:
            trades.append({
                'bar': i,
                'from_pos': position,
                'to_pos': target_position,
                'action': action_value,
                'balance_before': balance,
                'cost': cost
            })

        # Update position
        old_position = position
        position = target_position
        positions.append(position)

        # Calculate PnL from current bar's return
        market_return = returns[i]
        pnl = position * market_return * balance - cost
        balance += pnl

        # Track
        peak_balance = max(peak_balance, balance)
        equity_curve.append(balance)

    # Calculate metrics
    final_equity = balance
    total_return = (final_equity / initial_balance) - 1
    drawdown_curve = [(peak_balance - e) / peak_balance for e, peak_balance in
                      zip(equity_curve, [max(equity_curve[:i+1]) for i in range(len(equity_curve))])]
    max_drawdown = max(drawdown_curve)

    # Calculate Sharpe ratio (annualized)
    returns_series = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = np.mean(returns_series) / np.std(returns_series) * np.sqrt(252 * 60) if np.std(returns_series) > 0 else 0

    # Win rate (profitable trades)
    winning_trades = sum(1 for i in range(1, len(equity_curve)) if equity_curve[i] > equity_curve[i-1])
    win_rate = winning_trades / (len(equity_curve) - 1) if len(equity_curve) > 1 else 0

    # Action distribution
    total_actions = n_long + n_short + n_hold
    pct_long = n_long / total_actions * 100
    pct_short = n_short / total_actions * 100
    pct_hold = n_hold / total_actions * 100

    metrics = {
        'initial_balance': initial_balance,
        'final_equity': final_equity,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown * 100,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'total_trades': len(trades),
        'n_bars': n_bars,
        'trades_per_day': len(trades) / (n_bars / 60),  # Assuming 60 bars/day
        'action_distribution': {
            'long_pct': pct_long,
            'short_pct': pct_short,
            'hold_pct': pct_hold,
        },
        'equity_curve': equity_curve,
    }

    if verbose:
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)
        print(f"  Initial Balance:  ${initial_balance:,.2f}")
        print(f"  Final Equity:     ${final_equity:,.2f}")
        print(f"  Total Return:     {total_return*100:+.2f}%")
        print(f"  Max Drawdown:     {max_drawdown*100:.2f}%")
        print(f"  Sharpe Ratio:     {sharpe:.2f}")
        print(f"  Win Rate:         {win_rate*100:.1f}%")
        print(f"  Total Trades:     {len(trades)}")
        print(f"  Trades/Day:       {metrics['trades_per_day']:.1f}")
        print(f"\nAction Distribution:")
        print(f"  LONG:  {pct_long:.1f}%")
        print(f"  HOLD:  {pct_hold:.1f}%")
        print(f"  SHORT: {pct_short:.1f}%")

        # Acceptance criteria check
        print("\n" + "-"*70)
        print("ACCEPTANCE CRITERIA CHECK:")
        checks = [
            ('Win Rate >= 30%', win_rate >= 0.30),
            ('Sharpe >= 0.5', sharpe >= 0.5),
            ('Max DD <= 20%', max_drawdown <= 0.20),
            ('HOLD % >= 15%', pct_hold >= 15),
            ('HOLD % <= 80%', pct_hold <= 80),
        ]

        all_passed = True
        for name, passed in checks:
            status = "PASS" if passed else "FAIL"
            print(f"  [{status}] {name}")
            if not passed:
                all_passed = False

        print("-"*70)
        if all_passed:
            print("BACKTEST PASSED ALL CRITERIA!")
        else:
            print("BACKTEST FAILED SOME CRITERIA")
        print("="*70)

    return metrics


def main():
    """Main backtest function."""
    import argparse

    parser = argparse.ArgumentParser(description='BACKTEST VALIDATION')
    parser.add_argument('--model', type=str,
                       default='models/ppo_production/final_model.zip',
                       help='Path to model .zip file')
    parser.add_argument('--dataset', type=str,
                       default='data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv',
                       help='Path to dataset')
    parser.add_argument('--norm-stats', type=str,
                       default='config/norm_stats.json',
                       help='Path to normalization stats')
    parser.add_argument('--test-split', type=float, default=0.85,
                       help='Test split fraction')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON path for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for deterministic results')

    args = parser.parse_args()

    # Set seed for deterministic backtest (FASE 8)
    set_seed(args.seed)

    # Resolve paths
    model_path = PROJECT_ROOT / args.model
    dataset_path = PROJECT_ROOT / args.dataset
    norm_stats_path = PROJECT_ROOT / args.norm_stats

    # Try best_model if final not found
    if not model_path.exists():
        best = model_path.parent / "best_model.zip"
        if best.exists():
            model_path = best
        else:
            print(f"ERROR: Model not found at {model_path}")
            sys.exit(1)

    # Run backtest
    metrics = run_backtest(
        model_path=str(model_path),
        dataset_path=str(dataset_path),
        norm_stats_path=str(norm_stats_path),
        test_split=args.test_split,
        verbose=True
    )

    # Save results if output specified
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove equity curve from saved results (too large)
        metrics_to_save = {k: v for k, v in metrics.items() if k != 'equity_curve'}
        metrics_to_save['timestamp'] = datetime.now().isoformat()

        with open(output_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
