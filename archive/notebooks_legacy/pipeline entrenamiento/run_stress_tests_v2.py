#!/usr/bin/env python3
"""
USD/COP RL Trading System - Stress Test V2 (with Regime Detection)
===================================================================

Compares Model B performance WITH vs WITHOUT regime detection on crisis periods.

Hypothesis: Regime detection should help during PROLONGED crises (Fed Hikes, LatAm Selloff)
where the model collapsed to HOLD without it.

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


def create_environment(df, feature_cols, episode_length=None, use_regime=False):
    """Create trading environment with optional regime detection."""
    if episode_length is None:
        episode_length = min(400, len(df) - 10)

    return TradingEnvironmentV19(
        df=df,
        initial_balance=10000,
        max_position=1.0,
        episode_length=episode_length,
        feature_columns=feature_cols,
        use_vol_scaling=True,
        use_regime_detection=use_regime,  # Key difference!
        verbose=0,
    )


def train_model_b(train_df, feature_cols, episode_length, use_regime=False, timesteps=60_000):
    """Train Model B with optional regime detection."""
    env = create_environment(train_df, feature_cols, episode_length, use_regime=use_regime)

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

    return model


def evaluate_on_period(model, period_df, feature_cols, period_name, use_regime=False, bars_per_day=56):
    """Evaluate model on a specific crisis period."""
    if len(period_df) < bars_per_day * 5:
        return {
            'period': period_name,
            'data_available': False,
            'n_bars': len(period_df),
        }

    episode_length = min(400, len(period_df) - 10)
    env = create_environment(period_df, feature_cols, episode_length, use_regime=use_regime)

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


def run_comparative_stress_tests():
    """Run stress tests comparing WITH vs WITHOUT regime detection."""
    print("\n" + "=" * 70)
    print("  COMPARATIVE STRESS TESTS - With vs Without Regime Detection")
    print("=" * 70)

    # Load data
    df = load_data()
    print(f"\n  Loaded {len(df):,} rows")

    # Check for regime detection features
    has_vix = 'vix_z' in df.columns
    has_embi = 'embi_z' in df.columns
    print(f"  VIX feature available: {has_vix}")
    print(f"  EMBI feature available: {has_embi}")

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

    # Train 60 days pre-crisis data
    train_end = 60 * 288  # 60 days * 288 bars/day
    train_df = df.iloc[:min(train_end, len(df) // 3)].copy()
    episode_length = min(400, len(train_df) - 10)

    # Train Model B WITHOUT regime detection
    print("\n  Training Model B WITHOUT regime detection...")
    model_no_regime = train_model_b(train_df, feature_cols, episode_length, use_regime=False, timesteps=60_000)
    print("    Done.")

    # Train Model B WITH regime detection
    print("\n  Training Model B WITH regime detection...")
    model_with_regime = train_model_b(train_df, feature_cols, episode_length, use_regime=True, timesteps=60_000)
    print("    Done.")

    # Compare on each crisis period
    print("\n  Testing on crisis periods...")
    print("-" * 70)

    results_comparison = []

    for period in crisis_periods:
        print(f"\n  {period['name']}:")
        print(f"    Period: {period['start']} to {period['end']}")

        # Filter data for this period
        if 'timestamp' in df.columns:
            mask = (df['timestamp'] >= period['start']) & (df['timestamp'] <= period['end'])
            period_df = df[mask].copy().reset_index(drop=True)
        else:
            print(f"    [SKIP] No timestamp column")
            continue

        print(f"    Bars available: {len(period_df):,}")

        if len(period_df) < 280:
            print(f"    [SKIP] Not enough data")
            continue

        # Evaluate WITHOUT regime detection
        result_no = evaluate_on_period(model_no_regime, period_df, feature_cols, period['name'], use_regime=False)

        # Evaluate WITH regime detection
        result_with = evaluate_on_period(model_with_regime, period_df, feature_cols, period['name'], use_regime=True)

        if not result_no['data_available'] or not result_with['data_available']:
            print(f"    [SKIP] Evaluation failed")
            continue

        # Check improvements
        sharpe_improvement = result_with['sharpe'] - result_no['sharpe']
        hold_reduction = result_no['pct_hold'] - result_with['pct_hold']

        # Check pass/fail
        def check_pass(result, period):
            passed = True
            if result['max_dd'] > period['max_dd_threshold']:
                passed = False
            if result['sharpe'] < period['min_sharpe_threshold']:
                passed = False
            if result['pct_hold'] > 95:
                passed = False
            return passed

        passed_no = check_pass(result_no, period)
        passed_with = check_pass(result_with, period)

        status_no = "PASS" if passed_no else "FAIL"
        status_with = "PASS" if passed_with else "FAIL"

        print(f"\n    WITHOUT Regime Detection:")
        print(f"      [{status_no}] Sharpe: {result_no['sharpe']:.2f}, MaxDD: {result_no['max_dd']:.1f}%")
        print(f"             Actions: L:{result_no['pct_long']:.0f}% S:{result_no['pct_short']:.0f}% H:{result_no['pct_hold']:.0f}%")

        print(f"\n    WITH Regime Detection:")
        print(f"      [{status_with}] Sharpe: {result_with['sharpe']:.2f}, MaxDD: {result_with['max_dd']:.1f}%")
        print(f"             Actions: L:{result_with['pct_long']:.0f}% S:{result_with['pct_short']:.0f}% H:{result_with['pct_hold']:.0f}%")

        # Show improvement
        if sharpe_improvement > 0:
            print(f"\n    [+] Regime Detection IMPROVED Sharpe by {sharpe_improvement:.2f}")
        else:
            print(f"\n    [-] Regime Detection REDUCED Sharpe by {-sharpe_improvement:.2f}")

        if hold_reduction > 0:
            print(f"    [+] Regime Detection REDUCED HOLD by {hold_reduction:.0f}%")
        elif hold_reduction < 0:
            print(f"    [-] Regime Detection INCREASED HOLD by {-hold_reduction:.0f}%")

        results_comparison.append({
            'period': period['name'],
            'sharpe_no_regime': result_no['sharpe'],
            'sharpe_with_regime': result_with['sharpe'],
            'sharpe_improvement': sharpe_improvement,
            'hold_no_regime': result_no['pct_hold'],
            'hold_with_regime': result_with['pct_hold'],
            'hold_reduction': hold_reduction,
            'passed_no_regime': passed_no,
            'passed_with_regime': passed_with,
        })

    # Summary
    print("\n" + "=" * 70)
    print("  COMPARATIVE SUMMARY")
    print("=" * 70)

    if results_comparison:
        print(f"\n  {'Period':<20} | {'No Regime':<12} | {'With Regime':<12} | {'Improvement'}")
        print(f"  {'-'*20}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

        for r in results_comparison:
            status_no = "PASS" if r['passed_no_regime'] else "FAIL"
            status_with = "PASS" if r['passed_with_regime'] else "FAIL"
            print(f"  {r['period']:<20} | {status_no} Sh:{r['sharpe_no_regime']:>6.2f} | {status_with} Sh:{r['sharpe_with_regime']:>6.2f} | {r['sharpe_improvement']:>+6.2f}")

        # Count passes
        pass_no = sum(1 for r in results_comparison if r['passed_no_regime'])
        pass_with = sum(1 for r in results_comparison if r['passed_with_regime'])

        print(f"\n  Pass Rate WITHOUT Regime: {pass_no}/{len(results_comparison)}")
        print(f"  Pass Rate WITH Regime:    {pass_with}/{len(results_comparison)}")

        # Calculate averages
        avg_improvement = np.mean([r['sharpe_improvement'] for r in results_comparison])
        avg_hold_reduction = np.mean([r['hold_reduction'] for r in results_comparison])

        print(f"\n  Average Sharpe Improvement: {avg_improvement:+.2f}")
        print(f"  Average HOLD Reduction:     {avg_hold_reduction:+.1f}%")

        # Verdict
        if pass_with > pass_no:
            print("\n  VERDICT: Regime Detection HELPS during crises")
            print("           --> Enable for production")
        elif pass_with < pass_no:
            print("\n  VERDICT: Regime Detection HURTS during crises")
            print("           --> Keep disabled")
        else:
            if avg_improvement > 0:
                print("\n  VERDICT: Same pass rate, but Sharpe improved")
                print("           --> Consider enabling for production")
            else:
                print("\n  VERDICT: Same pass rate, Sharpe similar/worse")
                print("           --> Keep disabled for simplicity")

    print("\n" + "=" * 70)

    return results_comparison


if __name__ == "__main__":
    run_comparative_stress_tests()
