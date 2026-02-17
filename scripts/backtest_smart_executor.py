"""
Smart Executor Backtest — Trailing Stop on 5-min Bars
=====================================================

Validates the trailing-stop executor against hold-to-close baseline.
Reuses walk-forward + vol-targeting from vol_target_backtest.py.

Workflow:
  1. Load daily OHLCV + features, run walk-forward (9 models, 5 folds)
  2. Apply vol-targeting to get direction + leverage per day
  3. Load 5-min OHLCV for intraday simulation
  4. For each signal day: simulate trailing stop on next-day 5-min bars
  5. Compare hold vs trail returns, sensitivity grid, Phase 1 gate

Usage:
    python scripts/backtest_smart_executor.py
    python scripts/backtest_smart_executor.py --activation 0.002 --trail 0.003 --hard-stop 0.015
    python scripts/backtest_smart_executor.py --skip-sensitivity

Data sources:
    - seeds/latest/usdcop_daily_ohlcv.parquet  (daily, for forecasting)
    - seeds/latest/usdcop_m5_ohlcv.parquet     (5-min, for trailing stop)
    - data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet (DXY+WTI)

@version 1.0.0
"""

import argparse
import json
import logging
import sys
import time
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse from vol_target_backtest (same project, not duplicating)
from scripts.vol_target_backtest import (
    BacktestMetrics,
    FoldPredictions,
    VolTargetConfig,
    apply_vol_targeting,
    bootstrap_ci,
    compute_metrics,
    load_data,
    run_walk_forward,
)
from src.execution.trailing_stop import (
    TrailingState,
    TrailingStopConfig,
    TrailingStopTracker,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# 5-MIN DATA LOADING
# =============================================================================

def load_m5_data() -> Tuple[pd.DataFrame, dict]:
    """
    Load 5-min OHLCV for USD/COP, add date column, group by date.

    Returns:
        (m5_df, m5_grouped_dict)
        m5_grouped_dict: {date -> DataFrame of 5-min bars for that day}
    """
    logger.info("Loading 5-min OHLCV...")
    path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
    m5 = pd.read_parquet(path)

    # Filter to USD/COP only
    m5 = m5[m5["symbol"] == "USD/COP"].copy()

    # Strip timezone for date extraction, keep original time for reference
    m5["time_local"] = m5["time"].dt.tz_localize(None)
    m5["date"] = m5["time_local"].dt.date

    # Sort by time within each day
    m5 = m5.sort_values("time_local").reset_index(drop=True)

    # Group by date into dict for O(1) lookup
    m5_grouped = {date: group for date, group in m5.groupby("date")}

    logger.info(f"  5-min: {len(m5)} bars, {len(m5_grouped)} days, "
                f"{m5['date'].min()} to {m5['date'].max()}")

    return m5, m5_grouped


def get_next_trading_day(signal_date, sorted_dates: list):
    """
    Find first trading day strictly after signal_date using bisect.
    Returns None if no subsequent trading day exists.
    """
    idx = bisect_right(sorted_dates, signal_date)
    if idx < len(sorted_dates):
        return sorted_dates[idx]
    return None


# =============================================================================
# TRAILING STOP SIMULATION
# =============================================================================

@dataclass
class TradeDetail:
    """Record of a single day's trailing stop simulation."""
    signal_date: str
    execution_date: str
    direction: int
    leverage: float
    entry_price: float
    exit_price: float
    exit_reason: str
    hold_return: float       # direction * leverage * log(close_D1/close_D)
    trail_return: float      # direction * leverage * pnl_pct
    n_bars: int
    exit_bar_idx: Optional[int]
    peak_price: float


def simulate_trailing_stops(
    dates: np.ndarray,
    directions: np.ndarray,
    leverages: np.ndarray,
    actual_returns: np.ndarray,
    daily_df: pd.DataFrame,
    m5_grouped: dict,
    trail_config: TrailingStopConfig,
) -> Tuple[np.ndarray, np.ndarray, List[TradeDetail]]:
    """
    Simulate trailing stops for each signal day.

    For each signal on day D:
      - Entry price = daily close[D] (consistent with paper trading)
      - Next trading day D+1: run trailing stop on 5-min bars
      - Hold return = direction * leverage * actual_return[D] (log return close-to-close)
      - Trail return = direction * leverage * tracker.pnl_pct

    Returns:
        (hold_returns, trail_returns, trade_details)
    """
    # Build daily date -> close lookup
    daily_dates = pd.to_datetime(daily_df["date"]).dt.date.values
    daily_closes = daily_df["close"].values
    daily_close_map = dict(zip(daily_dates, daily_closes))

    # Sorted list of dates that have 5-min data
    sorted_m5_dates = sorted(m5_grouped.keys())

    n = len(dates)
    hold_returns = np.zeros(n)
    trail_returns = np.zeros(n)
    trade_details = []

    n_triggered = 0
    n_expired = 0
    n_hard_stop = 0
    n_no_m5 = 0

    for i in range(n):
        signal_date = pd.Timestamp(dates[i]).date()
        direction = int(directions[i])
        leverage = leverages[i]
        actual_ret = actual_returns[i]

        # Hold return (same for both strategies)
        hold_ret = direction * leverage * actual_ret
        hold_returns[i] = hold_ret

        # Get entry price = close[signal_date]
        entry_price = daily_close_map.get(signal_date)
        if entry_price is None:
            # Fallback: no daily close for this date, use hold return
            trail_returns[i] = hold_ret
            continue

        # Find next trading day with 5-min data
        next_day = get_next_trading_day(signal_date, sorted_m5_dates)
        if next_day is None or next_day not in m5_grouped:
            trail_returns[i] = hold_ret
            n_no_m5 += 1
            continue

        bars = m5_grouped[next_day]
        if len(bars) == 0:
            trail_returns[i] = hold_ret
            n_no_m5 += 1
            continue

        # Direction 0 = flat, no position
        if direction == 0:
            trail_returns[i] = 0.0
            hold_returns[i] = 0.0
            continue

        # Run trailing stop
        tracker = TrailingStopTracker(entry_price, direction, trail_config)

        for bar_idx, (_, bar) in enumerate(bars.iterrows()):
            state = tracker.update(bar["high"], bar["low"], bar["close"], bar_idx)
            if state == TrailingState.TRIGGERED:
                break

        # If not triggered, expire at session close
        if tracker.state not in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            tracker.expire(bars.iloc[-1]["close"])

        # Trail return
        trail_pnl = tracker.pnl_pct if tracker.pnl_pct is not None else 0.0
        trail_ret = leverage * trail_pnl  # direction already in pnl_pct
        trail_returns[i] = trail_ret

        # Track stats
        if tracker.exit_reason == "trailing_stop":
            n_triggered += 1
        elif tracker.exit_reason == "hard_stop":
            n_hard_stop += 1
        else:
            n_expired += 1

        trade_details.append(TradeDetail(
            signal_date=str(signal_date),
            execution_date=str(next_day),
            direction=direction,
            leverage=round(leverage, 4),
            entry_price=round(entry_price, 2),
            exit_price=round(tracker.exit_price, 2) if tracker.exit_price else 0.0,
            exit_reason=tracker.exit_reason or "unknown",
            hold_return=round(hold_ret, 6),
            trail_return=round(trail_ret, 6),
            n_bars=len(bars),
            exit_bar_idx=tracker.exit_bar_idx,
            peak_price=round(tracker.peak_price, 2),
        ))

    total_trades = n_triggered + n_expired + n_hard_stop
    if total_trades > 0:
        logger.info(f"  Trailing stops: {n_triggered} triggered ({n_triggered/total_trades:.0%}), "
                    f"{n_hard_stop} hard stops ({n_hard_stop/total_trades:.0%}), "
                    f"{n_expired} expired ({n_expired/total_trades:.0%}), "
                    f"{n_no_m5} days without 5-min data")

    return hold_returns, trail_returns, trade_details


# =============================================================================
# SENSITIVITY ANALYSIS
# =============================================================================

def run_sensitivity(
    dates: np.ndarray,
    directions: np.ndarray,
    leverages: np.ndarray,
    actual_returns: np.ndarray,
    daily_df: pd.DataFrame,
    m5_grouped: dict,
) -> pd.DataFrame:
    """
    Grid search over activation_pct x trail_pct with fixed hard_stop=0.015.
    Returns DataFrame with one row per config.
    """
    activation_grid = [0.001, 0.0015, 0.002, 0.003]
    trail_grid = [0.002, 0.0025, 0.003, 0.004, 0.005]

    results = []
    total_combos = len(activation_grid) * len(trail_grid)
    logger.info(f"Sensitivity analysis: {total_combos} combinations")

    for act in activation_grid:
        for trail in trail_grid:
            config = TrailingStopConfig(
                activation_pct=act,
                trail_pct=trail,
                hard_stop_pct=0.015,
            )
            hold_rets, trail_rets, details = simulate_trailing_stops(
                dates, directions, leverages, actual_returns,
                daily_df, m5_grouped, config,
            )

            # Quick metrics
            cum_hold = np.prod(1 + hold_rets) - 1
            cum_trail = np.prod(1 + trail_rets) - 1
            delta = cum_trail - cum_hold

            trail_std = np.std(trail_rets, ddof=1)
            trail_sharpe = (np.mean(trail_rets) / trail_std * np.sqrt(252)) if trail_std > 0 else 0

            hold_std = np.std(hold_rets, ddof=1)
            hold_sharpe = (np.mean(hold_rets) / hold_std * np.sqrt(252)) if hold_std > 0 else 0

            n_triggered = sum(1 for d in details if d.exit_reason == "trailing_stop")
            trigger_rate = n_triggered / len(details) if details else 0

            results.append({
                "activation_pct": act,
                "trail_pct": trail,
                "hold_return_pct": round(cum_hold * 100, 2),
                "trail_return_pct": round(cum_trail * 100, 2),
                "delta_pct": round(delta * 100, 2),
                "hold_sharpe": round(hold_sharpe, 3),
                "trail_sharpe": round(trail_sharpe, 3),
                "trigger_rate_pct": round(trigger_rate * 100, 1),
            })

    return pd.DataFrame(results)


# =============================================================================
# PER-YEAR ANALYSIS
# =============================================================================

def per_year_analysis(
    dates: np.ndarray,
    hold_returns: np.ndarray,
    trail_returns: np.ndarray,
) -> Dict:
    """Compute per-year comparison metrics."""
    df = pd.DataFrame({
        "date": pd.to_datetime(dates),
        "hold": hold_returns,
        "trail": trail_returns,
    })
    df["year"] = df["date"].dt.year

    years = sorted(df["year"].unique())
    per_year = {}

    for year in years:
        mask = df["year"] == year
        h = df.loc[mask, "hold"].values
        t = df.loc[mask, "trail"].values

        cum_hold = (np.prod(1 + h) - 1) * 100
        cum_trail = (np.prod(1 + t) - 1) * 100

        h_std = np.std(h, ddof=1)
        t_std = np.std(t, ddof=1)
        h_sharpe = (np.mean(h) / h_std * np.sqrt(252)) if h_std > 0 else 0
        t_sharpe = (np.mean(t) / t_std * np.sqrt(252)) if t_std > 0 else 0

        # Max drawdown
        h_cum = np.cumprod(1 + h)
        t_cum = np.cumprod(1 + t)
        h_peak = np.maximum.accumulate(h_cum)
        t_peak = np.maximum.accumulate(t_cum)
        h_maxdd = float(np.min((h_cum - h_peak) / h_peak)) * 100
        t_maxdd = float(np.min((t_cum - t_peak) / t_peak)) * 100

        per_year[str(year)] = {
            "n_days": int(mask.sum()),
            "hold_return_pct": round(cum_hold, 2),
            "trail_return_pct": round(cum_trail, 2),
            "delta_pct": round(cum_trail - cum_hold, 2),
            "hold_sharpe": round(h_sharpe, 3),
            "trail_sharpe": round(t_sharpe, 3),
            "hold_maxdd_pct": round(h_maxdd, 2),
            "trail_maxdd_pct": round(t_maxdd, 2),
            "trail_wins": cum_trail > cum_hold,
        }

    return per_year


# =============================================================================
# GATE EVALUATION
# =============================================================================

def gate_phase1(
    per_year: Dict,
    hold_returns: np.ndarray,
    trail_returns: np.ndarray,
    trade_details: List[TradeDetail],
) -> Dict:
    """
    Evaluate Phase 1 gate criteria.

    Criteria:
      1. Trail > Hold in >= 4/5 years
      2. Paired t-test p < 0.10 on daily deltas
      3. Trail activation rate > 20% (trailing stop fires meaningfully)
      4. MaxDD not worse: trail_maxdd <= 1.10 * hold_maxdd (per overall period)
    """
    # 1. Trail > Hold per year
    years_trail_wins = sum(1 for y in per_year.values() if y["trail_wins"])
    total_years = len(per_year)

    # 2. Paired t-test on daily deltas
    daily_deltas = trail_returns - hold_returns
    t_stat, p_value = stats.ttest_rel(trail_returns, hold_returns)

    # 3. Activation rate
    n_triggered = sum(1 for d in trade_details if d.exit_reason == "trailing_stop")
    n_hard_stop = sum(1 for d in trade_details if d.exit_reason == "hard_stop")
    total_trades = len(trade_details)
    activation_rate = (n_triggered + n_hard_stop) / total_trades if total_trades > 0 else 0

    # 4. MaxDD comparison (overall)
    h_cum = np.cumprod(1 + hold_returns)
    t_cum = np.cumprod(1 + trail_returns)
    h_peak = np.maximum.accumulate(h_cum)
    t_peak = np.maximum.accumulate(t_cum)
    hold_maxdd = abs(float(np.min((h_cum - h_peak) / h_peak)))
    trail_maxdd = abs(float(np.min((t_cum - t_peak) / t_peak)))
    maxdd_ratio = trail_maxdd / hold_maxdd if hold_maxdd > 0 else 1.0

    # Gate checks
    check_years = years_trail_wins >= min(4, total_years)
    check_pvalue = p_value < 0.10
    check_activation = activation_rate > 0.20
    check_maxdd = maxdd_ratio <= 1.10

    all_pass = check_years and check_pvalue and check_activation and check_maxdd

    return {
        "passed": all_pass,
        "criteria": {
            "trail_wins_per_year": {
                "value": f"{years_trail_wins}/{total_years}",
                "threshold": f">= {min(4, total_years)}/{total_years}",
                "passed": check_years,
            },
            "paired_ttest_p": {
                "value": round(p_value, 4),
                "t_stat": round(float(t_stat), 4),
                "threshold": "< 0.10",
                "passed": check_pvalue,
            },
            "activation_rate": {
                "value": round(activation_rate * 100, 1),
                "n_triggered": n_triggered,
                "n_hard_stop": n_hard_stop,
                "n_expired": total_trades - n_triggered - n_hard_stop,
                "threshold": "> 20%",
                "passed": check_activation,
            },
            "maxdd_ratio": {
                "trail_maxdd_pct": round(trail_maxdd * 100, 2),
                "hold_maxdd_pct": round(hold_maxdd * 100, 2),
                "ratio": round(maxdd_ratio, 3),
                "threshold": "<= 1.10",
                "passed": check_maxdd,
            },
        },
        "summary_stats": {
            "mean_daily_delta": round(float(np.mean(daily_deltas)), 6),
            "median_daily_delta": round(float(np.median(daily_deltas)), 6),
            "delta_positive_pct": round(float(np.mean(daily_deltas > 0)) * 100, 1),
        },
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smart Executor Backtest — Trailing Stop")
    parser.add_argument("--activation", type=float, default=0.002, help="Activation pct (default 0.002)")
    parser.add_argument("--trail", type=float, default=0.003, help="Trail pct (default 0.003)")
    parser.add_argument("--hard-stop", type=float, default=0.015, help="Hard stop pct (default 0.015)")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Vol-target level (default 0.15)")
    parser.add_argument("--max-leverage", type=float, default=2.0, help="Max leverage (default 2.0)")
    parser.add_argument("--n-folds", type=int, default=5, help="Walk-forward folds (default 5)")
    parser.add_argument("--skip-sensitivity", action="store_true", help="Skip sensitivity grid")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    trail_config = TrailingStopConfig(
        activation_pct=args.activation,
        trail_pct=args.trail,
        hard_stop_pct=args.hard_stop,
    )
    vol_config = VolTargetConfig(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
    )

    print("=" * 70)
    print("  USDCOP Smart Executor Backtest — Trailing Stop")
    print("  Phase 1: Core Logic + Backtest Validation")
    print("=" * 70)
    print(f"  Trail config: activation={trail_config.activation_pct:.3%}, "
          f"trail={trail_config.trail_pct:.3%}, hard_stop={trail_config.hard_stop_pct:.3%}")
    print(f"  Vol-target:   tv={vol_config.target_vol:.0%}, ml={vol_config.max_leverage:.1f}x")

    start_time = time.time()

    # ─── Step 1: Load data ───
    print("\n" + "=" * 70)
    print("  STEP 1: Load Data")
    print("=" * 70)

    daily_df = load_data()
    m5_df, m5_grouped = load_m5_data()

    # ─── Step 2: Walk-forward (reuse from vol_target_backtest) ───
    print("\n" + "=" * 70)
    print("  STEP 2: Walk-Forward (9 models, 5 folds, H=1)")
    print("=" * 70)

    fold_predictions, model_metrics = run_walk_forward(
        daily_df, n_folds=args.n_folds, ensemble_strategy="top_3",
    )

    # ─── Step 3: Apply vol-targeting ───
    print("\n" + "=" * 70)
    print("  STEP 3: Apply Vol-Targeting")
    print("=" * 70)

    all_dates = np.concatenate([f.dates for f in fold_predictions])
    all_preds = np.concatenate([f.predicted_returns for f in fold_predictions])
    all_actuals = np.concatenate([f.actual_returns for f in fold_predictions])

    _, vol_leverages, _, valid_mask = apply_vol_targeting(
        all_dates, all_preds, all_actuals, vol_config,
    )

    # Filter to valid (post-warmup) dates
    valid_dates = all_dates[valid_mask]
    valid_preds = all_preds[valid_mask]
    valid_actuals = all_actuals[valid_mask]
    valid_leverages = vol_leverages[valid_mask]
    valid_directions = np.sign(valid_preds)

    logger.info(f"  Valid days after vol warmup: {len(valid_dates)} "
                f"({pd.Timestamp(valid_dates[0]).date()} to {pd.Timestamp(valid_dates[-1]).date()})")

    # ─── Step 4: Trailing stop simulation ───
    print("\n" + "=" * 70)
    print("  STEP 4: Trailing Stop Simulation")
    print("=" * 70)

    hold_returns, trail_returns, trade_details = simulate_trailing_stops(
        valid_dates, valid_directions, valid_leverages, valid_actuals,
        daily_df, m5_grouped, trail_config,
    )

    # ─── Step 5: Per-year analysis ───
    print("\n" + "=" * 70)
    print("  STEP 5: Per-Year Comparison")
    print("=" * 70)

    per_year = per_year_analysis(valid_dates, hold_returns, trail_returns)

    print(f"\n  {'Year':<6} {'Hold%':>8} {'Trail%':>8} {'Delta':>8} "
          f"{'H_Sharpe':>9} {'T_Sharpe':>9} {'H_MaxDD':>8} {'T_MaxDD':>8} {'Win':>5}")
    print(f"  {'-' * 72}")

    for year, data in sorted(per_year.items()):
        win_str = "YES" if data["trail_wins"] else "no"
        print(f"  {year:<6} {data['hold_return_pct']:>+8.2f} {data['trail_return_pct']:>+8.2f} "
              f"{data['delta_pct']:>+8.2f} "
              f"{data['hold_sharpe']:>9.3f} {data['trail_sharpe']:>9.3f} "
              f"{data['hold_maxdd_pct']:>8.2f} {data['trail_maxdd_pct']:>8.2f} "
              f"{win_str:>5}")

    # Totals
    cum_hold = (np.prod(1 + hold_returns) - 1) * 100
    cum_trail = (np.prod(1 + trail_returns) - 1) * 100
    h_std = np.std(hold_returns, ddof=1)
    t_std = np.std(trail_returns, ddof=1)
    h_sharpe = (np.mean(hold_returns) / h_std * np.sqrt(252)) if h_std > 0 else 0
    t_sharpe = (np.mean(trail_returns) / t_std * np.sqrt(252)) if t_std > 0 else 0

    print(f"  {'-' * 72}")
    print(f"  {'TOTAL':<6} {cum_hold:>+8.2f} {cum_trail:>+8.2f} {cum_trail - cum_hold:>+8.2f} "
          f"{h_sharpe:>9.3f} {t_sharpe:>9.3f}")

    # ─── Step 6: Gate evaluation ───
    print("\n" + "=" * 70)
    print("  STEP 6: GATE Phase 1 Evaluation")
    print("=" * 70)

    gate = gate_phase1(per_year, hold_returns, trail_returns, trade_details)

    for criterion, data in gate["criteria"].items():
        status = "PASS" if data["passed"] else "FAIL"
        if criterion == "trail_wins_per_year":
            print(f"  [{status}] Trail > Hold per year: {data['value']} (need {data['threshold']})")
        elif criterion == "paired_ttest_p":
            print(f"  [{status}] Paired t-test: t={data['t_stat']:.3f}, p={data['value']:.4f} (need {data['threshold']})")
        elif criterion == "activation_rate":
            print(f"  [{status}] Activation rate: {data['value']:.1f}% "
                  f"({data['n_triggered']} trail + {data['n_hard_stop']} hard stop + {data['n_expired']} expired) "
                  f"(need {data['threshold']})")
        elif criterion == "maxdd_ratio":
            print(f"  [{status}] MaxDD ratio: trail={data['trail_maxdd_pct']:.2f}% / hold={data['hold_maxdd_pct']:.2f}% "
                  f"= {data['ratio']:.3f} (need {data['threshold']})")

    print(f"\n  Mean daily delta:   {gate['summary_stats']['mean_daily_delta']:.6f}")
    print(f"  Median daily delta: {gate['summary_stats']['median_daily_delta']:.6f}")
    print(f"  Delta positive:     {gate['summary_stats']['delta_positive_pct']:.1f}% of days")

    gate_result = "PASS" if gate["passed"] else "FAIL"
    print(f"\n  >>> GATE Phase 1: {gate_result} <<<")

    # ─── Step 7: Sensitivity analysis (optional) ───
    sensitivity_results = None
    if not args.skip_sensitivity:
        print("\n" + "=" * 70)
        print("  STEP 7: Sensitivity Analysis")
        print("=" * 70)

        sensitivity_df = run_sensitivity(
            valid_dates, valid_directions, valid_leverages, valid_actuals,
            daily_df, m5_grouped,
        )

        print(f"\n  {'Act%':>6} {'Trail%':>7} {'Hold':>8} {'Trail':>8} {'Delta':>8} "
              f"{'H_Shp':>7} {'T_Shp':>7} {'Trig%':>7}")
        print(f"  {'-' * 62}")

        for _, row in sensitivity_df.iterrows():
            print(f"  {row['activation_pct']:.3%} {row['trail_pct']:.3%} "
                  f"{row['hold_return_pct']:>+8.2f} {row['trail_return_pct']:>+8.2f} "
                  f"{row['delta_pct']:>+8.2f} {row['hold_sharpe']:>7.3f} "
                  f"{row['trail_sharpe']:>7.3f} {row['trigger_rate_pct']:>7.1f}")

        # Best config by delta
        best_idx = sensitivity_df["delta_pct"].idxmax()
        best = sensitivity_df.loc[best_idx]
        print(f"\n  Best config: activation={best['activation_pct']:.3%}, trail={best['trail_pct']:.3%} "
              f"-> delta={best['delta_pct']:+.2f}pp, trail_sharpe={best['trail_sharpe']:.3f}")

        sensitivity_results = sensitivity_df.to_dict("records")

    # ─── Step 8: Save results ───
    duration = time.time() - start_time

    results = {
        "experiment": "Smart Executor Backtest — Trailing Stop Phase 1",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "duration_seconds": round(duration, 1),
        "config": {
            "trail": asdict(trail_config),
            "vol_target": asdict(vol_config),
            "n_folds": args.n_folds,
            "ensemble_strategy": "top_3",
        },
        "overall": {
            "hold_return_pct": round(cum_hold, 2),
            "trail_return_pct": round(cum_trail, 2),
            "delta_pct": round(cum_trail - cum_hold, 2),
            "hold_sharpe": round(h_sharpe, 3),
            "trail_sharpe": round(t_sharpe, 3),
            "n_days": len(valid_dates),
        },
        "per_year": per_year,
        "gate_phase1": gate,
        "sensitivity": sensitivity_results,
        "exit_reason_counts": {
            "trailing_stop": sum(1 for d in trade_details if d.exit_reason == "trailing_stop"),
            "hard_stop": sum(1 for d in trade_details if d.exit_reason == "hard_stop"),
            "session_close": sum(1 for d in trade_details if d.exit_reason == "session_close"),
            "total": len(trade_details),
        },
    }

    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = str(output_dir / "smart_executor_backtest.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results saved to: {output_path}")
    print(f"  Total time: {duration:.1f}s")


if __name__ == "__main__":
    main()
