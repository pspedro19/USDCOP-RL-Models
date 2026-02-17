#!/usr/bin/env python3
"""
EXP-ENS-V215b-001: Ensemble Backtest with Real TradingEnvironment
==================================================================
Runs L4 backtest using 5 V21.5b seed models with majority vote,
through the real TradingEnvironment (with SL/TP, min_hold_bars, etc.).

Unlike ensemble_backtest.py (simplified loop), this uses the ACTUAL
environment so results are directly comparable to single-model L4.

Usage:
    python scripts/ensemble_backtest_env.py
    python scripts/ensemble_backtest_env.py --model-dir models/v215b_ensemble
    python scripts/ensemble_backtest_env.py --model-dir models/v215b_ensemble --kelly
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_ensemble_models(model_dir: Path) -> List:
    """Load all seed models from ensemble directory."""
    from stable_baselines3 import PPO

    models = []
    seed_dirs = sorted(model_dir.glob("seed_*"))

    if not seed_dirs:
        # Fallback: try loading directly from timestamped dirs
        logger.warning(f"No seed_* dirs in {model_dir}, looking for best_model.zip files")
        for p in sorted(model_dir.glob("*/best_model.zip")):
            models.append(PPO.load(str(p)))
            logger.info(f"  Loaded: {p}")
        return models

    for seed_dir in seed_dirs:
        best = seed_dir / "best_model.zip"
        if best.exists():
            models.append(PPO.load(str(best)))
            logger.info(f"  Loaded: {best}")
        else:
            logger.warning(f"  Missing: {best}")

    return models


def ensemble_predict(models: List, obs: np.ndarray, threshold_long: float, threshold_short: float) -> tuple:
    """
    Majority vote across models for continuous action space.

    Maps each model's continuous action to signal:
      action > threshold_long  → +1 (LONG)
      action < threshold_short → -1 (SHORT)
      else                     →  0 (HOLD)

    Returns: (consensus_action_value, confidence, vote_details)
      - consensus_action_value: float in [-1, 0, 1] for env.step()
      - confidence: fraction of models agreeing (0-1)
      - vote_details: dict with individual votes
    """
    signals = []
    raw_actions = []

    for model in models:
        action, _ = model.predict(obs, deterministic=True)
        val = float(action[0]) if hasattr(action, '__len__') else float(action)
        raw_actions.append(val)

        if val > threshold_long:
            signals.append(1)
        elif val < threshold_short:
            signals.append(-1)
        else:
            signals.append(0)

    # Majority vote
    from collections import Counter
    votes = Counter(signals)
    majority_signal, majority_count = votes.most_common(1)[0]
    confidence = majority_count / len(models)

    # Require minimum consensus (3/5)
    min_consensus = 3
    if majority_count < min_consensus:
        majority_signal = 0  # HOLD if no consensus

    # Map signal back to continuous action for the environment
    # Use the mean raw action of agreeing models for magnitude
    if majority_signal != 0:
        agreeing_actions = [a for a, s in zip(raw_actions, signals) if s == majority_signal]
        action_value = float(np.mean(agreeing_actions))
    else:
        action_value = 0.0

    return action_value, confidence, {
        "signals": signals,
        "raw_actions": [round(a, 4) for a in raw_actions],
        "majority_signal": majority_signal,
        "majority_count": majority_count,
    }


def run_ensemble_backtest(
    model_dir: Path,
    use_kelly: bool = False,
) -> Dict[str, Any]:
    """Run ensemble backtest through real TradingEnvironment."""
    from src.config.pipeline_config import load_pipeline_config
    from src.training.environments.trading_env import TradingEnvironment

    # Import shared utilities from pipeline runner
    from scripts.run_ssot_pipeline import (
        create_env_config,
        calculate_backtest_metrics,
        validate_gates,
        log_backtest_results,
        get_backtest_margin,
    )

    config = load_pipeline_config()

    # Load dataset (val + test combined)
    logger.info("Loading datasets...")
    val_df = pd.read_parquet(PROJECT_ROOT / config.paths.val_file)
    test_df = pd.read_parquet(PROJECT_ROOT / config.paths.test_file)
    backtest_df = pd.concat([val_df, test_df]).sort_index()
    backtest_df = backtest_df[~backtest_df.index.duplicated(keep="first")]

    norm_stats_path = PROJECT_ROOT / config.paths.norm_stats_file
    with open(norm_stats_path) as f:
        norm_stats = json.load(f)

    logger.info(f"  Combined: {len(backtest_df)} rows ({backtest_df.index[0]} → {backtest_df.index[-1]})")

    # Load ensemble models
    logger.info(f"Loading ensemble from {model_dir}...")
    models = load_ensemble_models(model_dir)
    if len(models) == 0:
        logger.error("No models loaded!")
        return {}
    logger.info(f"  Loaded {len(models)} models")

    # Get feature columns from SSOT
    feature_cols = [f.name for f in config.get_market_features()]

    # Create backtest environment
    bt_env_cfg = create_env_config(
        config, feature_cols,
        len(backtest_df) - get_backtest_margin(),
        stage="backtest",
    )
    env = TradingEnvironment(df=backtest_df, norm_stats=norm_stats, config=bt_env_cfg)

    # Run backtest with ensemble predictions
    obs, _ = env.reset()
    done = False
    equity_curve = [config.backtest.initial_capital]
    step = 0

    threshold_long = config.backtest.threshold_long
    threshold_short = config.backtest.threshold_short

    while not done:
        action_value, confidence, votes = ensemble_predict(
            models, obs, threshold_long, threshold_short,
        )

        # Kelly position sizing (optional)
        if use_kelly:
            ps = config.position_sizing
            kelly_frac = ps.base_fraction
            if ps.consensus_scaling:
                kelly_frac *= confidence
            kelly_frac = np.clip(kelly_frac, ps.min_fraction, ps.max_fraction)
            # Scale action magnitude by Kelly fraction
            action_value *= kelly_frac / ps.max_fraction

        # Step environment with consensus action
        action = np.array([action_value], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        equity_curve.append(float(info.get("equity", info.get("balance", equity_curve[-1]))))
        step += 1

        if step % 5000 == 0:
            logger.info(f"  Step {step}, Equity: ${equity_curve[-1]:.2f}")

    # Calculate metrics using shared function
    metrics = calculate_backtest_metrics(equity_curve, env, stage="ENSEMBLE-L4")

    # Monthly returns
    idx = backtest_df.index[:len(equity_curve)]
    equity_series = pd.Series(equity_curve, index=idx)
    monthly = equity_series.resample("ME").last()
    monthly_returns = monthly.pct_change().dropna() * 100
    metrics["monthly_returns"] = {
        d.strftime("%Y-%m"): round(float(r), 2) for d, r in monthly_returns.items()
    }

    # Log results
    log_backtest_results(metrics, "ENSEMBLE L4")

    logger.info("  Monthly Returns:")
    for month, ret in metrics["monthly_returns"].items():
        logger.info(f"    {month}: {ret:+7.2f}%")

    # Validate gates
    passed, gate_results = validate_gates(metrics, config)
    metrics["gates_passed"] = bool(passed)
    metrics["gate_results"] = gate_results
    metrics["ensemble_info"] = {
        "n_models": len(models),
        "min_consensus": 3,
        "use_kelly": use_kelly,
        "model_dir": str(model_dir),
    }

    # Save results
    results_dir = PROJECT_ROOT / "results/backtests"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "ensemble_v215b_backtest.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save equity curve
    equity_df = pd.DataFrame({"equity": equity_curve}, index=idx)
    equity_df.to_parquet(results_dir / "equity_curve_ensemble_v215b.parquet")

    logger.info(f"Results saved: {output_path}")
    logger.info(f"L4 GATES: {'PASSED' if passed else 'FAILED'}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="EXP-ENS-V215b-001: Ensemble Backtest")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/v215b_ensemble",
        help="Directory with seed model subdirectories",
    )
    parser.add_argument(
        "--kelly",
        action="store_true",
        help="Enable Half-Kelly position sizing",
    )
    args = parser.parse_args()

    model_dir = PROJECT_ROOT / args.model_dir

    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        logger.info("Create it first with symlinks to V21.5b seed models.")
        return 1

    result = run_ensemble_backtest(model_dir, use_kelly=args.kelly)

    if not result:
        return 1

    # Compare vs V21.5b baselines
    logger.info("")
    logger.info("=" * 60)
    logger.info("COMPARISON vs V21.5b SINGLE-MODEL BASELINES")
    logger.info("=" * 60)

    seed_results_path = PROJECT_ROOT / "results/backtests/l4_per_seed_comparison_v21_5b.json"
    if seed_results_path.exists():
        with open(seed_results_path) as f:
            seed_results = json.load(f)

        mean_single = np.mean([v["total_return_pct"] for v in seed_results.values()])
        best_single = max(v["total_return_pct"] for v in seed_results.values())

        ens_ret = result["total_return_pct"]
        logger.info(f"  Ensemble return:      {ens_ret:+.2f}%")
        logger.info(f"  V21.5b mean (5 seeds): {mean_single:+.2f}%")
        logger.info(f"  V21.5b best seed:      {best_single:+.2f}%")
        logger.info(f"  vs mean: {ens_ret - mean_single:+.2f}pp")
        logger.info(f"  vs best: {ens_ret - best_single:+.2f}pp")

    return 0


if __name__ == "__main__":
    sys.exit(main())
