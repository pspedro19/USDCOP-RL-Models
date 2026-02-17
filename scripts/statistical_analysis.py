#!/usr/bin/env python3
"""
EXP-INFRA-001: Statistical Baselines for V21.5b
=================================================
Pre-Fase analysis: buy-and-hold, random agent, bootstrap CI, long/short breakdown.

Usage:
    python scripts/statistical_analysis.py
    python scripts/statistical_analysis.py --seed-results results/backtests/l4_per_seed_comparison_v21_5b.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

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


# =============================================================================
# 1. Buy-and-Hold Return
# =============================================================================

def calculate_buy_and_hold(ohlcv_path: Path, val_start: str, test_end: str) -> dict:
    """Calculate buy-and-hold USDCOP return for the OOS period."""
    df = pd.read_parquet(ohlcv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    mask = (df.index >= val_start) & (df.index <= test_end)
    oos = df.loc[mask]

    if len(oos) == 0:
        logger.error("No data in OOS period")
        return {}

    first_close = oos["close"].iloc[0]
    last_close = oos["close"].iloc[-1]
    bnh_return = (last_close / first_close - 1) * 100

    logger.info("=" * 60)
    logger.info("1. BUY-AND-HOLD USDCOP")
    logger.info("=" * 60)
    logger.info(f"  Period: {oos.index[0]} → {oos.index[-1]}")
    logger.info(f"  First close: {first_close:.2f}")
    logger.info(f"  Last close:  {last_close:.2f}")
    logger.info(f"  Return: {bnh_return:+.2f}%")
    logger.info(f"  Bars: {len(oos)}")

    return {
        "period_start": str(oos.index[0]),
        "period_end": str(oos.index[-1]),
        "first_close": float(first_close),
        "last_close": float(last_close),
        "return_pct": round(float(bnh_return), 4),
        "n_bars": len(oos),
    }


# =============================================================================
# 2. Random Agent Baseline (1000 simulations)
# =============================================================================

def simulate_random_agents(
    ohlcv_path: Path,
    val_start: str,
    test_end: str,
    n_sims: int = 1000,
    sl_pct: float = -0.04,
    tp_pct: float = 0.04,
    min_hold: int = 25,
    cost_bps: float = 1.0,
    seed: int = 42,
) -> dict:
    """Simulate random agents with same SL/TP/costs as V21.5b."""
    rng = np.random.RandomState(seed)

    df = pd.read_parquet(ohlcv_path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

    mask = (df.index >= val_start) & (df.index <= test_end)
    oos = df.loc[mask]
    prices = oos["close"].values

    cost_rate = cost_bps / 10000.0
    sim_returns = []

    for sim in range(n_sims):
        capital = 10000.0
        position = 0  # -1, 0, 1
        entry_price = 0.0
        entry_bar = 0
        n_trades = 0
        wins = 0

        for i in range(1, len(prices)):
            if position != 0:
                bars_held = i - entry_bar
                pnl_pct = (prices[i] / entry_price - 1) * position

                # Check SL/TP
                hit_sl = pnl_pct <= sl_pct
                hit_tp = pnl_pct >= tp_pct
                can_exit = bars_held >= min_hold

                if hit_sl or hit_tp or (can_exit and rng.random() < 0.02):
                    # Close position
                    trade_pnl = pnl_pct * capital
                    cost = abs(1) * cost_rate * capital * 2  # round-trip
                    capital += trade_pnl - cost
                    n_trades += 1
                    if trade_pnl > 0:
                        wins += 1
                    position = 0
            else:
                # Random entry: ~3% chance per bar (matches ~350 trades in 14000 bars)
                if rng.random() < 0.025:
                    position = rng.choice([-1, 1])
                    entry_price = prices[i]
                    entry_bar = i
                    # Entry cost
                    capital -= cost_rate * capital

        total_return = (capital / 10000.0 - 1) * 100
        sim_returns.append(total_return)

    sim_returns = np.array(sim_returns)

    logger.info("=" * 60)
    logger.info("2. RANDOM AGENT BASELINE")
    logger.info("=" * 60)
    logger.info(f"  Simulations: {n_sims}")
    logger.info(f"  Config: SL={sl_pct*100:.1f}%, TP={tp_pct*100:.1f}%, min_hold={min_hold}, cost={cost_bps}bps")
    logger.info(f"  Mean return: {np.mean(sim_returns):+.2f}%")
    logger.info(f"  Median return: {np.median(sim_returns):+.2f}%")
    logger.info(f"  Std: {np.std(sim_returns):.2f}%")
    logger.info(f"  5th-95th percentile: [{np.percentile(sim_returns, 5):+.2f}%, {np.percentile(sim_returns, 95):+.2f}%]")
    logger.info(f"  % positive: {(sim_returns > 0).mean()*100:.1f}%")

    return {
        "n_simulations": n_sims,
        "config": {
            "sl_pct": sl_pct,
            "tp_pct": tp_pct,
            "min_hold_bars": min_hold,
            "cost_bps": cost_bps,
        },
        "mean_return_pct": round(float(np.mean(sim_returns)), 4),
        "median_return_pct": round(float(np.median(sim_returns)), 4),
        "std_return_pct": round(float(np.std(sim_returns)), 4),
        "pct_positive": round(float((sim_returns > 0).mean() * 100), 2),
        "percentile_5": round(float(np.percentile(sim_returns, 5)), 4),
        "percentile_25": round(float(np.percentile(sim_returns, 25)), 4),
        "percentile_75": round(float(np.percentile(sim_returns, 75)), 4),
        "percentile_95": round(float(np.percentile(sim_returns, 95)), 4),
    }


# =============================================================================
# 3. Bootstrap CI on V21.5b per-seed returns
# =============================================================================

def bootstrap_ci(
    seed_results: dict,
    n_bootstrap: int = 10000,
    ci_level: float = 0.95,
    rng_seed: int = 42,
) -> dict:
    """Bootstrap 95% CI on mean return across seeds."""
    rng = np.random.RandomState(rng_seed)

    returns = np.array([v["total_return_pct"] for v in seed_results.values()])
    seeds = list(seed_results.keys())
    n_seeds = len(returns)

    # Bootstrap mean returns
    boot_means = np.array([
        np.mean(rng.choice(returns, size=n_seeds, replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci_level
    ci_low = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_high = float(np.percentile(boot_means, (1 - alpha / 2) * 100))
    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns, ddof=1))

    # t-test: H0 = mean return is 0
    t_stat = mean_ret / (std_ret / np.sqrt(n_seeds)) if std_ret > 0 else 0
    from scipy import stats as sp_stats
    p_value = float(2 * sp_stats.t.sf(abs(t_stat), df=n_seeds - 1))

    significant = ci_low > 0  # CI excludes zero on the low end

    logger.info("=" * 60)
    logger.info("3. BOOTSTRAP CI ON V21.5b PER-SEED RETURNS")
    logger.info("=" * 60)
    logger.info(f"  Per-seed returns:")
    for s, r in zip(seeds, returns):
        logger.info(f"    Seed {s}: {r:+.2f}%")
    logger.info(f"  Mean: {mean_ret:+.2f}% ± {std_ret:.2f}%")
    logger.info(f"  Seeds positive: {(returns > 0).sum()}/{n_seeds}")
    logger.info(f"  Bootstrap {ci_level*100:.0f}% CI: [{ci_low:+.2f}%, {ci_high:+.2f}%]")
    logger.info(f"  t-test: t={t_stat:.3f}, p={p_value:.4f}")
    logger.info(f"  CI excludes zero: {'YES' if significant else 'NO'}")
    logger.info(f"  Statistically significant: {'YES' if p_value < 0.05 else 'NO'}")

    return {
        "per_seed_returns": {s: round(float(r), 4) for s, r in zip(seeds, returns)},
        "mean_return_pct": round(mean_ret, 4),
        "std_return_pct": round(std_ret, 4),
        "seeds_positive": int((returns > 0).sum()),
        "n_seeds": n_seeds,
        "bootstrap_ci_low": round(ci_low, 4),
        "bootstrap_ci_high": round(ci_high, 4),
        "ci_level": ci_level,
        "n_bootstrap": n_bootstrap,
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "ci_excludes_zero": significant,
        "statistically_significant_at_005": p_value < 0.05,
    }


# =============================================================================
# 4. Long vs Short Breakdown (per-seed from backtest re-run)
# =============================================================================

def analyze_long_short(seed_results: dict) -> dict:
    """
    Analyze long/short breakdown from per-seed results.

    Note: The existing per-seed results don't split by direction.
    This section provides what we CAN derive and flags what needs
    a fresh backtest run to obtain.
    """
    logger.info("=" * 60)
    logger.info("4. LONG vs SHORT ANALYSIS")
    logger.info("=" * 60)

    # Aggregate across seeds
    total_trades = sum(v["n_trades"] for v in seed_results.values())
    total_winning = sum(v["winning_trades"] for v in seed_results.values())
    total_losing = sum(v["losing_trades"] for v in seed_results.values())
    avg_wr = np.mean([v["win_rate_pct"] for v in seed_results.values()])
    avg_pf = np.mean([v["profit_factor"] for v in seed_results.values()])
    avg_bars = np.mean([v.get("avg_bars_per_trade", 0) for v in seed_results.values()])

    logger.info(f"  Aggregate across 5 seeds:")
    logger.info(f"    Total trades: {total_trades}")
    logger.info(f"    Winning: {total_winning} ({total_winning/total_trades*100:.1f}%)")
    logger.info(f"    Losing: {total_losing} ({total_losing/total_trades*100:.1f}%)")
    logger.info(f"    Avg WR: {avg_wr:.1f}%")
    logger.info(f"    Avg PF: {avg_pf:.3f}")
    logger.info(f"    Avg bars/trade: {avg_bars:.1f}")
    logger.info(f"  NOTE: Long/Short split requires fresh backtest with trade logging.")
    logger.info(f"        Use ensemble_backtest_env.py for detailed trade analysis.")

    return {
        "total_trades_all_seeds": total_trades,
        "total_winning": total_winning,
        "total_losing": total_losing,
        "aggregate_win_rate_pct": round(float(avg_wr), 2),
        "aggregate_profit_factor": round(float(avg_pf), 4),
        "avg_bars_per_trade": round(float(avg_bars), 1),
        "note": "Long/short split not available from existing results. Need fresh backtest with trade logging.",
    }


# =============================================================================
# 5. Trade Duration Analysis
# =============================================================================

def analyze_trade_duration(seed_results: dict) -> dict:
    """Analyze trade durations from per-seed data."""
    logger.info("=" * 60)
    logger.info("5. TRADE DURATION ANALYSIS")
    logger.info("=" * 60)

    avg_bars_list = [v.get("avg_bars_per_trade", 0) for v in seed_results.values()]
    seeds = list(seed_results.keys())

    for s, avg in zip(seeds, avg_bars_list):
        hours = avg * 5 / 60  # 5-min bars to hours
        logger.info(f"  Seed {s}: avg {avg:.1f} bars ({hours:.1f} hours)")

    overall_avg = np.mean(avg_bars_list)
    min_hold = 25  # from config
    logger.info(f"  Overall avg: {overall_avg:.1f} bars ({overall_avg * 5 / 60:.1f} hours)")
    logger.info(f"  min_hold_bars: {min_hold} ({min_hold * 5 / 60:.1f} hours)")
    logger.info(f"  Avg/min_hold ratio: {overall_avg / min_hold:.2f}x")

    clustered_at_min = overall_avg < min_hold * 1.5
    if clustered_at_min:
        logger.warning("  WARNING: Trades clustered near min_hold — agent exits ASAP")
    else:
        logger.info("  Trades NOT clustered at min_hold — agent holds beyond minimum")

    return {
        "per_seed_avg_bars": {s: round(float(a), 1) for s, a in zip(seeds, avg_bars_list)},
        "overall_avg_bars": round(float(overall_avg), 1),
        "overall_avg_hours": round(float(overall_avg * 5 / 60), 1),
        "min_hold_bars": min_hold,
        "avg_to_min_hold_ratio": round(float(overall_avg / min_hold), 2),
        "clustered_at_min_hold": bool(clustered_at_min),
    }


# =============================================================================
# 6. V21.5b vs Baselines Comparison
# =============================================================================

def compare_vs_baselines(
    seed_results: dict,
    bnh_result: dict,
    random_result: dict,
    bootstrap_result: dict,
) -> dict:
    """Compare V21.5b against all baselines."""
    logger.info("=" * 60)
    logger.info("6. V21.5b vs BASELINES COMPARISON")
    logger.info("=" * 60)

    mean_ret = bootstrap_result["mean_return_pct"]
    bnh_ret = bnh_result["return_pct"]
    random_mean = random_result["mean_return_pct"]
    random_p95 = random_result["percentile_95"]

    beats_bnh = mean_ret > bnh_ret
    beats_random_mean = mean_ret > random_mean
    beats_random_p95 = mean_ret > random_p95

    logger.info(f"  V21.5b mean return:     {mean_ret:+.2f}%")
    logger.info(f"  Buy-and-hold return:    {bnh_ret:+.2f}%")
    logger.info(f"  Random agent mean:      {random_mean:+.2f}%")
    logger.info(f"  Random agent 95th pctl: {random_p95:+.2f}%")
    logger.info(f"")
    logger.info(f"  V21.5b vs B&H:          {mean_ret - bnh_ret:+.2f}pp ({'BEATS' if beats_bnh else 'LOSES'})")
    logger.info(f"  V21.5b vs Random mean:   {mean_ret - random_mean:+.2f}pp ({'BEATS' if beats_random_mean else 'LOSES'})")
    logger.info(f"  V21.5b vs Random 95th:   {mean_ret - random_p95:+.2f}pp ({'BEATS' if beats_random_p95 else 'LOSES'})")

    # Gate checks
    logger.info(f"")
    logger.info(f"  GATE CHECKS:")
    logger.info(f"    Beats random agent mean? {'PASS' if beats_random_mean else 'FAIL'}")
    logger.info(f"    Beats random agent 95th pctl? {'PASS' if beats_random_p95 else 'FAIL → agent may be lucky'}")
    logger.info(f"    Bootstrap CI excludes zero? {'PASS' if bootstrap_result['ci_excludes_zero'] else 'FAIL → not significant'}")

    return {
        "v215b_mean_return": mean_ret,
        "buy_and_hold_return": bnh_ret,
        "random_agent_mean": random_mean,
        "random_agent_p95": random_p95,
        "beats_buy_and_hold": beats_bnh,
        "beats_random_mean": beats_random_mean,
        "beats_random_p95": beats_random_p95,
        "excess_vs_bnh_pp": round(float(mean_ret - bnh_ret), 4),
        "excess_vs_random_pp": round(float(mean_ret - random_mean), 4),
        "all_gates_pass": beats_random_mean and bootstrap_result["ci_excludes_zero"],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="EXP-INFRA-001: Statistical Analysis")
    parser.add_argument(
        "--seed-results",
        type=str,
        default="results/backtests/l4_per_seed_comparison_v21_5b.json",
        help="Path to per-seed comparison JSON",
    )
    parser.add_argument(
        "--n-sims",
        type=int,
        default=1000,
        help="Number of random agent simulations",
    )
    args = parser.parse_args()

    ohlcv_path = PROJECT_ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet"
    seed_results_path = PROJECT_ROOT / args.seed_results

    # Load SSOT config for date ranges
    from src.config.pipeline_config import load_pipeline_config
    config = load_pipeline_config()
    val_start = config.date_ranges.val_start
    test_end = config.date_ranges.test_end

    # Load per-seed results
    with open(seed_results_path) as f:
        seed_results = json.load(f)

    logger.info("=" * 70)
    logger.info("EXP-INFRA-001: STATISTICAL BASELINES FOR V21.5b")
    logger.info("=" * 70)
    logger.info(f"  OOS period: {val_start} → {test_end}")
    logger.info(f"  Seeds: {list(seed_results.keys())}")

    # 1. Buy-and-hold
    bnh = calculate_buy_and_hold(ohlcv_path, val_start, test_end)

    # 2. Random agent baseline
    env_cfg = config.environment
    random_baseline = simulate_random_agents(
        ohlcv_path, val_start, test_end,
        n_sims=args.n_sims,
        sl_pct=env_cfg.stop_loss_pct,
        tp_pct=env_cfg.take_profit_pct,
        min_hold=env_cfg.min_hold_bars,
        cost_bps=env_cfg.slippage_bps,
    )

    # 3. Bootstrap CI
    bootstrap = bootstrap_ci(seed_results)

    # 4. Long/Short breakdown
    long_short = analyze_long_short(seed_results)

    # 5. Trade duration
    duration = analyze_trade_duration(seed_results)

    # 6. Comparison
    comparison = compare_vs_baselines(seed_results, bnh, random_baseline, bootstrap)

    # Save results
    output_dir = PROJECT_ROOT / "results/statistical_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": "EXP-INFRA-001",
        "model_version": "V21.5b",
        "buy_and_hold": bnh,
        "random_agent_baseline": random_baseline,
        "bootstrap_ci": bootstrap,
        "long_short_analysis": long_short,
        "trade_duration": duration,
        "comparison": comparison,
    }

    output_path = output_dir / "summary.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY SAVED")
    logger.info("=" * 70)
    logger.info(f"  Output: {output_path}")

    # Final verdict
    logger.info("")
    logger.info("=" * 70)
    logger.info("FINAL VERDICT")
    logger.info("=" * 70)
    if comparison["all_gates_pass"]:
        logger.info("  ALL GATES PASS — V21.5b has a real edge. Proceed to Fase 1.")
    else:
        if not comparison["beats_random_mean"]:
            logger.warning("  FAIL: V21.5b does NOT beat random agent mean. Replantear todo.")
        if not bootstrap["ci_excludes_zero"]:
            logger.warning("  FAIL: Bootstrap CI includes zero. Edge not statistically significant.")
        logger.warning("  Review results before proceeding.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
