#!/usr/bin/env python3
"""
USD/COP RL Trading System - Stress Test Runner
===============================================

Ejecuta stress tests en Model B para validar comportamiento en crisis.

Periodos de crisis a testear:
1. COVID_Crash (Feb-Apr 2020): MaxDD < 30%, Sharpe > -2.0
2. Fed_Hikes_2022 (Mar-Dec 2022): MaxDD < 25%, Sharpe > -1.0
3. Petro_Election (May-Aug 2022): MaxDD < 20%, Sharpe > -0.5
4. LatAm_Selloff (Sep-Nov 2022): MaxDD < 20%, Sharpe > -0.5
5. Banking_Crisis_2023 (Mar-Apr 2023): MaxDD < 20%, Sharpe > -1.0

Author: Claude Code
Date: 2025-12-26
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "validation"))

from stable_baselines3 import PPO

# Environment
from environment_v19 import TradingEnvironmentV19

# Model config
from ensemble import MODEL_B_CONFIG


def load_data():
    """Load the full dataset."""
    data_dir = ROOT.parent.parent / "data" / "pipeline" / "07_output" / "datasets_5min"
    path = data_dir / "RL_DS3_MACRO_CORE.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Ensure timestamp column
    if 'timestamp' not in df.columns and 'date' in df.columns:
        df['timestamp'] = pd.to_datetime(df['date'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure close_return
    if 'close_return' not in df.columns and 'log_ret_5m' in df.columns:
        df['close_return'] = df['log_ret_5m']

    return df


def get_feature_columns(df):
    """Get feature columns from dataframe."""
    exclude = ['timestamp', 'date', 'time', 'symbol', 'close_return', 'log_ret_5m']
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'float32', 'int64']]
    return feature_cols


def create_environment(df, feature_cols, episode_length=None):
    """Create trading environment."""
    if episode_length is None:
        episode_length = min(400, len(df) - 10)

    return TradingEnvironmentV19(
        df=df,
        initial_balance=10000,
        max_position=1.0,
        episode_length=episode_length,
        feature_columns=feature_cols,
        use_vol_scaling=True,
        use_regime_detection=False,  # Test without regime first
        verbose=0,
    )


def train_model_b(train_df, feature_cols, episode_length, timesteps=60_000):
    """Train Model B with its config."""
    print("  Training Model B (ent_coef=0.05)...")

    env = create_environment(train_df, feature_cols, episode_length)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=MODEL_B_CONFIG.model_params["learning_rate"],
        n_steps=MODEL_B_CONFIG.model_params["n_steps"],
        batch_size=MODEL_B_CONFIG.model_params["batch_size"],
        n_epochs=MODEL_B_CONFIG.model_params["n_epochs"],
        gamma=MODEL_B_CONFIG.model_params["gamma"],
        ent_coef=MODEL_B_CONFIG.model_params["ent_coef"],
        clip_range=MODEL_B_CONFIG.model_params["clip_range"],
        policy_kwargs=MODEL_B_CONFIG.model_params.get("policy_kwargs", {}),
        seed=42,
        verbose=0,
        device="cpu",
    )

    model.learn(total_timesteps=timesteps, progress_bar=False)
    print("  Model B trained.")

    return model


def evaluate_on_period(model, period_df, feature_cols, period_name, bars_per_day=56):
    """Evaluate model on a specific crisis period."""
    if len(period_df) < bars_per_day * 5:
        return {
            'period': period_name,
            'data_available': False,
            'n_bars': len(period_df),
        }

    episode_length = min(400, len(period_df) - 10)
    env = create_environment(period_df, feature_cols, episode_length)

    obs, _ = env.reset(options={'start_idx': 0})
    done = False

    returns = []
    actions = []
    portfolio = [10000]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        returns.append(info.get('step_return', 0))
        actions.append(float(action[0]))
        portfolio.append(info.get('portfolio', portfolio[-1]))

    returns = np.array(returns)
    actions = np.array(actions)
    portfolio = np.array(portfolio)

    # Calculate metrics
    n_bars = len(returns)
    n_days = n_bars // bars_per_day

    # Sharpe (daily)
    if n_days >= 2:
        daily_returns = returns[:n_days * bars_per_day].reshape(n_days, bars_per_day).sum(axis=1)
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max Drawdown
    peak = np.maximum.accumulate(portfolio)
    drawdown = (peak - portfolio) / (peak + 1e-10) * 100
    max_dd = np.max(drawdown)

    # Total return
    total_return = (portfolio[-1] - portfolio[0]) / portfolio[0] * 100

    # Action distribution
    ACTION_THRESHOLD = 0.10
    pct_long = (actions > ACTION_THRESHOLD).mean() * 100
    pct_short = (actions < -ACTION_THRESHOLD).mean() * 100
    pct_hold = (np.abs(actions) <= ACTION_THRESHOLD).mean() * 100

    return {
        'period': period_name,
        'data_available': True,
        'n_bars': n_bars,
        'n_days': n_days,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'total_return': total_return,
        'pct_long': pct_long,
        'pct_short': pct_short,
        'pct_hold': pct_hold,
    }


def run_stress_tests():
    """Run all stress tests."""
    print("\n" + "=" * 70)
    print("  STRESS TESTS - Model B")
    print("  Testing behavior during crisis periods")
    print("=" * 70)

    # Load data
    df = load_data()
    print(f"\n  Loaded {len(df):,} rows")

    # Check date range
    if 'timestamp' in df.columns:
        min_date = df['timestamp'].min()
        max_date = df['timestamp'].max()
        print(f"  Date range: {min_date} to {max_date}")

    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")

    # Define crisis periods
    crisis_periods = [
        {
            'name': 'COVID_Crash',
            'start': '2020-02-20',
            'end': '2020-04-30',
            'max_dd_threshold': 30.0,
            'min_sharpe_threshold': -2.0,
        },
        {
            'name': 'Fed_Hikes_2022',
            'start': '2022-03-01',
            'end': '2022-12-31',
            'max_dd_threshold': 25.0,
            'min_sharpe_threshold': -1.0,
        },
        {
            'name': 'Petro_Election',
            'start': '2022-05-15',
            'end': '2022-08-15',
            'max_dd_threshold': 20.0,
            'min_sharpe_threshold': -0.5,
        },
        {
            'name': 'LatAm_Selloff',
            'start': '2022-09-01',
            'end': '2022-11-30',
            'max_dd_threshold': 20.0,
            'min_sharpe_threshold': -0.5,
        },
        {
            'name': 'Banking_Crisis_2023',
            'start': '2023-03-01',
            'end': '2023-04-30',
            'max_dd_threshold': 20.0,
            'min_sharpe_threshold': -1.0,
        },
    ]

    # Train Model B on first 60 days (before any crisis)
    print("\n  Training Model B on pre-crisis data...")
    train_end = 60 * 288  # 60 days * 288 bars/day (5min)
    train_df = df.iloc[:min(train_end, len(df) // 3)].copy()

    episode_length = min(400, len(train_df) - 10)
    model = train_model_b(train_df, feature_cols, episode_length, timesteps=60_000)

    # Test on each crisis period
    print("\n  Testing on crisis periods...")
    print("-" * 70)

    results = []

    for period in crisis_periods:
        print(f"\n  {period['name']}:")
        print(f"    Period: {period['start']} to {period['end']}")

        # Filter data for this period
        if 'timestamp' in df.columns:
            mask = (df['timestamp'] >= period['start']) & (df['timestamp'] <= period['end'])
            period_df = df[mask].copy().reset_index(drop=True)
        else:
            # If no timestamp, skip
            print(f"    [SKIP] No timestamp column")
            results.append({
                'period': period['name'],
                'data_available': False,
                'passed': False,
            })
            continue

        print(f"    Bars available: {len(period_df):,}")

        if len(period_df) < 280:  # Less than 1 day
            print(f"    [SKIP] Not enough data")
            results.append({
                'period': period['name'],
                'data_available': False,
                'passed': False,
            })
            continue

        # Evaluate
        result = evaluate_on_period(model, period_df, feature_cols, period['name'])

        if not result['data_available']:
            print(f"    [SKIP] Evaluation failed")
            results.append(result)
            continue

        # Check pass/fail
        passed = True
        failures = []

        if result['max_dd'] > period['max_dd_threshold']:
            passed = False
            failures.append(f"MaxDD {result['max_dd']:.1f}% > {period['max_dd_threshold']:.1f}%")

        if result['sharpe'] < period['min_sharpe_threshold']:
            passed = False
            failures.append(f"Sharpe {result['sharpe']:.2f} < {period['min_sharpe_threshold']:.2f}")

        if result['pct_hold'] > 95:
            passed = False
            failures.append("100% HOLD (model collapsed)")

        result['passed'] = passed
        result['failures'] = failures
        results.append(result)

        # Print result
        status = "PASS" if passed else "FAIL"
        print(f"    [{status}] Sharpe: {result['sharpe']:.2f}, MaxDD: {result['max_dd']:.1f}%")
        print(f"           Actions: L:{result['pct_long']:.0f}% S:{result['pct_short']:.0f}% H:{result['pct_hold']:.0f}%")

        if failures:
            for f in failures:
                print(f"           - {f}")

    # Summary
    print("\n" + "=" * 70)
    print("  STRESS TEST SUMMARY")
    print("=" * 70)

    valid_results = [r for r in results if r.get('data_available', False)]
    passed_results = [r for r in valid_results if r.get('passed', False)]

    print(f"\n  Periods tested: {len(valid_results)}/{len(crisis_periods)}")
    print(f"  Periods passed: {len(passed_results)}/{len(valid_results)}")

    if valid_results:
        sharpes = [r['sharpe'] for r in valid_results]
        max_dds = [r['max_dd'] for r in valid_results]

        print(f"\n  Crisis Performance:")
        print(f"    Mean Sharpe: {np.mean(sharpes):.2f}")
        print(f"    Mean MaxDD:  {np.mean(max_dds):.1f}%")
        print(f"    Worst Sharpe: {np.min(sharpes):.2f}")
        print(f"    Worst MaxDD:  {np.max(max_dds):.1f}%")

    # Overall verdict
    pass_rate = len(passed_results) / len(valid_results) if valid_results else 0
    overall_passed = pass_rate >= 0.6  # At least 60% pass

    print(f"\n  OVERALL: {'PASSED' if overall_passed else 'FAILED'}")
    print(f"  Pass rate: {pass_rate:.0%}")

    if overall_passed:
        print("\n  Model B is ROBUST in crisis conditions.")
        print("  Proceed to 5-fold validation.")
    else:
        print("\n  Model B shows WEAKNESS in crisis conditions.")
        print("  Consider enabling regime detection before deployment.")

    print("\n" + "=" * 70)

    return {
        'overall_passed': overall_passed,
        'pass_rate': pass_rate,
        'results': results,
    }


if __name__ == "__main__":
    run_stress_tests()
