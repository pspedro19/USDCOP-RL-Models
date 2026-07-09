"""
Analyze Model Action Distribution
=================================
Diagnose why a model generates few trades by analyzing action values.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta
import psycopg2

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from paper_trading
from paper_trading import (
    load_ohlcv_data, load_macro_data, FeatureCalculator,
    MARKET_FEATURES, CONFIG, MODEL_PATH
)

def analyze_actions():
    """Analyze what action values the model outputs."""
    from stable_baselines3 import PPO

    print("=" * 70)
    print("ANALYZING MODEL ACTION DISTRIBUTION")
    print("=" * 70)

    # Load data
    ohlcv_df = load_ohlcv_data(CONFIG["out_of_sample_start"], CONFIG["out_of_sample_end"])
    macro_df = load_macro_data(CONFIG["out_of_sample_start"], CONFIG["out_of_sample_end"])

    # Build features
    feature_calc = FeatureCalculator()
    features_df = feature_calc.build_features(ohlcv_df, macro_df)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}")
    model = PPO.load(str(MODEL_PATH))
    print(f"Observation space: {model.observation_space.shape}")

    # Calculate feature stats
    feature_means = {f: float(features_df[f].mean()) for f in MARKET_FEATURES if f in features_df.columns}
    feature_stds = {f: float(features_df[f].std()) + 1e-8 for f in MARKET_FEATURES if f in features_df.columns}

    # Collect all action values
    actions = []
    warmup = 50
    episode_length = len(features_df) - warmup
    position = 0.0

    for i in range(warmup, len(features_df)):
        row = features_df.iloc[i]
        current_step = i - warmup

        # Build observation (15-dim)
        obs = np.zeros(15, dtype=np.float32)
        for j, feat in enumerate(MARKET_FEATURES):
            raw = float(row.get(feat, 0.0))
            mean = feature_means.get(feat, 0.0)
            std = feature_stds.get(feat, 1.0)
            if std < 1e-8:
                std = 1.0
            normalized = (raw - mean) / std
            obs[j] = np.clip(normalized, -5, 5)
        obs[13] = position
        obs[14] = max(0, 1.0 - current_step / episode_length)
        obs = np.nan_to_num(obs, nan=0.0)

        action, _ = model.predict(obs, deterministic=True)
        action_value = float(action[0])
        actions.append(action_value)

    actions = np.array(actions)

    print("\n" + "=" * 70)
    print("ACTION VALUE STATISTICS")
    print("=" * 70)
    print(f"Total bars analyzed: {len(actions)}")
    print(f"Min action: {actions.min():.4f}")
    print(f"Max action: {actions.max():.4f}")
    print(f"Mean action: {actions.mean():.4f}")
    print(f"Std action: {actions.std():.4f}")

    print("\n" + "=" * 70)
    print("ACTION DISTRIBUTION (PERCENTILES)")
    print("=" * 70)
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(actions, p)
        print(f"  {p}th percentile: {val:.4f}")

    print("\n" + "=" * 70)
    print("TRADE SIGNALS BY THRESHOLD")
    print("=" * 70)

    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for thresh in thresholds:
        long_signals = np.sum(actions > thresh)
        short_signals = np.sum(actions < -thresh)
        total_signals = long_signals + short_signals
        print(f"  Threshold +/-{thresh:.2f}: {long_signals} LONG, {short_signals} SHORT, {total_signals} total")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    # Find threshold that gives ~4 trades per day
    # Period has about 8 trading days, so ~32 trades total
    target_trades = 32

    best_thresh = 0.10
    for thresh in np.arange(0.05, 0.35, 0.01):
        total = np.sum(actions > thresh) + np.sum(actions < -thresh)
        if total <= target_trades:
            best_thresh = thresh
            break

    print(f"For ~4 trades/day (32 total), use threshold: +/-{best_thresh:.2f}")

    # Also show what happens with different thresholds
    print("\nTrade simulation with different thresholds:")
    for thresh in [0.10, 0.15, 0.20]:
        # Count position changes
        trades = 0
        prev_pos = "FLAT"
        for a in actions:
            if a > thresh:
                new_pos = "LONG"
            elif a < -thresh:
                new_pos = "SHORT"
            else:
                new_pos = "FLAT"
            if new_pos != prev_pos and new_pos != "FLAT":
                trades += 1
            prev_pos = new_pos
        print(f"  Threshold +/-{thresh:.2f}: ~{trades} trades in the period")

    return actions, best_thresh


if __name__ == "__main__":
    actions, recommended_thresh = analyze_actions()
