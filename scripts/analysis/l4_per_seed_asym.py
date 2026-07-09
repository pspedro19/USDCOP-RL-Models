"""
L4 Per-Seed Backtest for EXP-ASYM-001 (Fase 1: Asymmetric SL/TP)

Runs L4 backtest on all 5 seeds and produces a comparison table.
Uses the same L2 dataset and pipeline config as the training run.

Usage:
    python scripts/l4_per_seed_asym.py
"""

import json
import logging
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Seed -> model directory mapping (from EXP-ASYM-001 training)
SEED_MODEL_MAP = {
    42: "ppo_ssot_20260211_224521",
    123: "ppo_ssot_20260212_011640",
    456: "ppo_ssot_20260212_034629",
    789: "ppo_ssot_20260212_061540",
    1337: "ppo_ssot_20260212_090002",
}


def run_l4_for_seed(seed: int, model_dir_name: str) -> dict:
    """Run L4 backtest for a single seed."""
    from src.config.pipeline_config import load_pipeline_config
    from src.training.environments.trading_env import TradingEnvironment
    from stable_baselines3 import PPO

    # Import helpers from pipeline runner
    from scripts.run_ssot_pipeline import (
        create_env_config,
        run_backtest_loop,
        calculate_backtest_metrics,
        validate_gates,
        get_backtest_margin,
    )

    config = load_pipeline_config()

    # Load model
    model_dir = PROJECT_ROOT / "models" / model_dir_name
    model_path = model_dir / "best_model.zip"
    if not model_path.exists():
        model_path = model_dir / "final_model.zip"

    logger.info(f"\n{'='*60}")
    logger.info(f"SEED {seed}: Loading model from {model_path}")
    model = PPO.load(str(model_path), device="cpu")

    # Load norm_stats from THIS model's directory
    norm_stats_path = model_dir / "norm_stats.json"
    with open(norm_stats_path, "r") as f:
        norm_stats = json.load(f)

    # Load val + test data (combined for full 2025 backtest)
    l2_dir = PROJECT_ROOT / "data" / "pipeline" / "07_output" / "5min"
    val_df = pd.read_parquet(l2_dir / "DS_production_val.parquet")
    test_df = pd.read_parquet(l2_dir / "DS_production_test.parquet")
    backtest_df = pd.concat([val_df, test_df]).sort_index()
    backtest_df = backtest_df[~backtest_df.index.duplicated(keep="first")]

    logger.info(f"  Data: {len(backtest_df)} rows (V:{len(val_df)} + T:{len(test_df)})")

    # Get feature columns from norm_stats (exclude _meta key)
    feature_cols = [k for k in norm_stats.keys() if k != "_meta"]

    # Create environment
    bt_env_cfg = create_env_config(
        config, feature_cols, len(backtest_df) - get_backtest_margin(), stage="backtest"
    )
    env = TradingEnvironment(df=backtest_df, norm_stats=norm_stats, config=bt_env_cfg)

    # Run backtest
    equity_curve = run_backtest_loop(model, env, config.backtest.initial_capital, log_progress=False)
    metrics = calculate_backtest_metrics(equity_curve, env, stage="L4-TEST")

    # Monthly returns
    equity_arr = np.array(equity_curve)
    dates = backtest_df.index[: len(equity_curve)]
    if len(dates) < len(equity_curve):
        dates = backtest_df.index  # fallback
    equity_series = pd.Series(equity_arr[: len(dates)], index=dates[: len(equity_arr)])
    monthly = equity_series.resample("ME").last()
    monthly_returns = ((monthly / monthly.shift(1)) - 1) * 100
    monthly_dict = {str(k.strftime("%Y-%m")): round(v, 2) for k, v in monthly_returns.items() if not np.isnan(v)}
    metrics["monthly_returns"] = monthly_dict

    # Gates
    passed, gate_results = validate_gates(metrics, config)
    metrics["gates_passed"] = bool(passed)
    metrics["seed"] = seed

    # Avg bars per trade
    portfolio = env._portfolio
    if hasattr(portfolio, "trade_durations") and portfolio.trade_durations:
        metrics["avg_bars_per_trade"] = round(np.mean(portfolio.trade_durations), 1)
    else:
        total_closed = metrics["n_trades"]
        if total_closed > 0:
            metrics["avg_bars_per_trade"] = round(len(backtest_df) / total_closed, 1)

    logger.info(f"  SEED {seed}: Return={metrics['total_return_pct']:+.2f}%, "
                f"Sharpe={metrics['sharpe_ratio']:.3f}, WR={metrics['win_rate_pct']:.1f}%, "
                f"PF={metrics['profit_factor']:.3f}, MaxDD={metrics['max_drawdown_pct']:.2f}%, "
                f"Trades={metrics['n_trades']}")

    return metrics


def main():
    logger.info("=" * 70)
    logger.info("EXP-ASYM-001: Per-Seed L4 Backtest (SL=-2.5%, TP=+6%)")
    logger.info("=" * 70)

    all_results = {}
    for seed, model_dir_name in SEED_MODEL_MAP.items():
        try:
            metrics = run_l4_for_seed(seed, model_dir_name)
            all_results[seed] = metrics
        except Exception as e:
            logger.error(f"SEED {seed} FAILED: {e}")
            import traceback
            traceback.print_exc()
            all_results[seed] = {"error": str(e), "seed": seed}

    # Summary table
    logger.info("\n" + "=" * 100)
    logger.info("EXP-ASYM-001 RESULTS SUMMARY")
    logger.info("=" * 100)
    logger.info(f"{'Seed':>6} | {'Return%':>9} | {'Sharpe':>8} | {'WR%':>6} | {'PF':>6} | {'MaxDD%':>8} | {'Trades':>7} | {'AvgBars':>8} | {'Gates':>6}")
    logger.info("-" * 100)

    returns = []
    seeds_positive = 0
    for seed in [42, 123, 456, 789, 1337]:
        m = all_results.get(seed, {})
        if "error" in m:
            logger.info(f"{seed:>6} | ERROR: {m['error']}")
            continue

        ret = m.get("total_return_pct", 0)
        returns.append(ret)
        if ret > 0:
            seeds_positive += 1

        logger.info(
            f"{seed:>6} | {ret:>+8.2f}% | {m.get('sharpe_ratio', 0):>+7.3f} | "
            f"{m.get('win_rate_pct', 0):>5.1f} | {m.get('profit_factor', 0):>5.3f} | "
            f"{m.get('max_drawdown_pct', 0):>7.2f} | {m.get('n_trades', 0):>7d} | "
            f"{m.get('avg_bars_per_trade', 0):>7.1f} | "
            f"{'PASS' if m.get('gates_passed') else 'FAIL':>6}"
        )

    logger.info("-" * 100)

    if returns:
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 0
        logger.info(f"Mean return: {mean_ret:+.2f}% ± {std_ret:.2f}%")
        logger.info(f"Seeds positive: {seeds_positive}/{len(returns)}")

        # Bootstrap CI
        n_bootstrap = 10000
        rng = np.random.default_rng(42)
        boot_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(returns, size=len(returns), replace=True)
            boot_means.append(np.mean(sample))
        ci_low = np.percentile(boot_means, 2.5)
        ci_high = np.percentile(boot_means, 97.5)
        logger.info(f"Bootstrap 95% CI: [{ci_low:+.2f}%, {ci_high:+.2f}%]")
        logger.info(f"CI excludes zero: {'YES' if ci_low > 0 else 'NO'}")

        # t-test
        if len(returns) > 1 and std_ret > 0:
            from scipy import stats
            t_stat, p_val = stats.ttest_1samp(returns, 0)
            logger.info(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")
            logger.info(f"Statistically significant (p<0.05): {'YES' if p_val < 0.05 else 'NO'}")

        # vs V21.5b baseline
        v215b_mean = 2.51
        logger.info(f"\nvs V21.5b baseline ({v215b_mean:+.2f}%): {mean_ret - v215b_mean:+.2f}pp")
        logger.info(f"vs Buy-and-hold (-14.66%): {mean_ret - (-14.66):+.2f}pp")

    # Save results
    output_dir = PROJECT_ROOT / "results" / "backtests"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "l4_per_seed_exp_asym_001.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "experiment_id": "EXP-ASYM-001",
                "description": "Fase 1: Asymmetric SL/TP (SL=-2.5%, TP=+6%)",
                "config_version": "4.1.0",
                "timestamp": datetime.now().isoformat(),
                "per_seed": {str(s): all_results[s] for s in [42, 123, 456, 789, 1337] if s in all_results},
                "summary": {
                    "mean_return_pct": float(np.mean(returns)) if returns else None,
                    "std_return_pct": float(np.std(returns, ddof=1)) if len(returns) > 1 else None,
                    "seeds_positive": seeds_positive,
                    "n_seeds": len(returns),
                },
            },
            f,
            indent=2,
            default=str,
        )
    logger.info(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
