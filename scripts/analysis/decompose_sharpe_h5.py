"""
Sharpe Decomposition: LINEAR-ONLY H=5 OOS 2025
================================================

Decomposes the Sharpe 3.461 into:
  1. Sharpe per WEEKLY SIGNAL (N=44): each observation = cumulative PnL of all
     sub-trades within that 5-day holding window. This is the TRUE Sharpe.
  2. Sharpe per SUB-TRADE (N=total sub-trades): inflated because sub-trades
     within a week are correlated (same direction, same market regime).
  3. Average sub-trades per week.
  4. Distribution of sub-trades per week (how many weeks have 1, 2, 3 sub-trades).

Also reports the correlation between sub-trade returns within the same week
to quantify the inflation factor.

@date 2026-02-16
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import get_horizon_config
from src.forecasting.vol_targeting import (
    VolTargetConfig, compute_realized_vol,
)
from src.execution.trailing_stop import TrailingStopConfig, TrailingStopTracker, TrailingState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG (same as diagnose_h5_ensemble.py)
# =============================================================================

INITIAL_CAPITAL = 10_000.0
SLIPPAGE_BPS = 1.0
HORIZON = 5

VOL_CONFIG = VolTargetConfig(
    target_vol=0.15, max_leverage=2.0, min_leverage=0.5,
    vol_lookback=21, vol_floor=0.05,
)

TRAIL_CONFIG = TrailingStopConfig(activation_pct=0.004, trail_pct=0.003, hard_stop_pct=0.04)

TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
OOS_END = pd.Timestamp("2025-12-31")

LINEAR_IDS = ["ridge", "bayesian_ridge"]


# =============================================================================
# DATA LOADING (same as validate_linear_walkforward.py)
# =============================================================================

def load_daily_data() -> pd.DataFrame:
    logger.info("Loading daily OHLCV...")
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)

    logger.info("Loading macro (DXY, WTI, VIX, EMBI)...")
    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }
    df_macro_sub = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_sub.rename(columns=macro_cols, inplace=True)
    df_macro_sub = df_macro_sub.sort_values("date")
    for col in macro_cols.values():
        df_macro_sub[col] = df_macro_sub[col].shift(1)

    df = pd.merge_asof(df_ohlcv.sort_values("date"), df_macro_sub.sort_values("date"),
                        on="date", direction="backward")
    df = _build_features(df)
    df["target_5d"] = np.log(df["close"].shift(-HORIZON) / df["close"])

    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    df = df[feature_mask].reset_index(drop=True)
    return df


def load_5min_data() -> pd.DataFrame:
    logger.info("Loading 5-min OHLCV...")
    m5_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
    df_m5 = pd.read_parquet(m5_path)
    if "symbol" in df_m5.columns:
        df_m5 = df_m5[df_m5["symbol"] == "USD/COP"].copy()
    df_m5 = df_m5.reset_index()
    if "time" in df_m5.columns:
        df_m5.rename(columns={"time": "timestamp"}, inplace=True)
    df_m5["timestamp"] = pd.to_datetime(df_m5["timestamp"])
    if df_m5["timestamp"].dt.tz is not None:
        df_m5["timestamp"] = df_m5["timestamp"].dt.tz_localize(None)
    df_m5["date"] = df_m5["timestamp"].dt.normalize()
    df_m5 = df_m5[df_m5["date"] >= pd.Timestamp("2024-12-01")].copy()
    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    return df_m5[["timestamp", "date", "open", "high", "low", "close"]].copy()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss_s = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss_s.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)
    for c in ["dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1"]:
        df[c] = df[c].ffill()
    return df


# =============================================================================
# MODEL TRAINING + PREDICTION (LINEAR-ONLY)
# =============================================================================

def predict_linear_oos(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Monthly re-training, LINEAR-ONLY (ridge + bayesian_ridge)."""
    target_col = "target_5d"
    df_train_all = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_oos = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].copy()
    df_oos = df_oos[df_oos[target_col].notna()]

    months = sorted(df_oos["date"].dt.to_period("M").unique())
    all_rows = []

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        if month_idx == 0:
            df_month_train = df_train_all[df_train_all[target_col].notna()].copy()
        else:
            df_month_train = df[(df["date"] >= TRAIN_START) & (df["date"] < month_start)].copy()
            df_month_train = df_month_train[df_month_train[target_col].notna()]

        df_month_oos = df_oos[(df_oos["date"] >= month_start) & (df_oos["date"] <= month_end)]

        if len(df_month_oos) == 0 or len(df_month_train) < 200:
            continue

        horizon_config = get_horizon_config(HORIZON)
        X_train = df_month_train[feature_cols].values.astype(np.float64)
        y_train = df_month_train[target_col].values.astype(np.float64)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        trained = {}
        for model_id in LINEAR_IDS:
            try:
                model = ModelFactory.create(model_id, params=None, horizon=HORIZON)
                model.fit(X_scaled, y_train)
                trained[model_id] = model
            except Exception:
                pass

        for _, row in df_month_oos.iterrows():
            X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)
            X_day_scaled = scaler.transform(X_day)

            preds = []
            for model_id in trained:
                try:
                    pred = trained[model_id].predict(X_day_scaled)[0]
                    preds.append(pred)
                except Exception:
                    pass

            if len(preds) == 0:
                continue

            ensemble_pred = np.mean(preds)

            # Vol-targeting
            idx_list = df.index[df["date"] == row["date"]]
            realized_vol = 0.10
            if len(idx_list) > 0:
                idx = idx_list[0]
                if idx >= VOL_CONFIG.vol_lookback:
                    recent = df.iloc[idx - VOL_CONFIG.vol_lookback:idx]["return_1d"].values
                    realized_vol = compute_realized_vol(recent, VOL_CONFIG.vol_lookback)

            leverage = np.clip(
                VOL_CONFIG.target_vol / max(realized_vol, VOL_CONFIG.vol_floor),
                VOL_CONFIG.min_leverage, VOL_CONFIG.max_leverage,
            )

            all_rows.append({
                "date": row["date"],
                "close": row["close"],
                "actual_return": row[target_col],
                "ensemble_pred": ensemble_pred,
                "direction": 1 if ensemble_pred > 0 else -1,
                "realized_vol": realized_vol,
                "leverage": leverage,
            })

    return pd.DataFrame(all_rows)


def select_weekly_trades(df: pd.DataFrame, h: int = 5) -> List[int]:
    df = df.sort_values("date").reset_index(drop=True)
    selected = [0]
    last = 0
    for i in range(1, len(df)):
        if i - last >= h:
            selected.append(i)
            last = i
    return selected


# =============================================================================
# TRAILING STOP WITH SUB-TRADE TRACKING
# =============================================================================

def simulate_trailing_detailed(trades: pd.DataFrame, df_m5: pd.DataFrame,
                                config: TrailingStopConfig) -> Tuple[pd.DataFrame, List[List[Dict]]]:
    """
    Like simulate_trailing but also returns individual sub-trade details per week.

    Returns:
      - trades_df with week_pnl, n_subtrades
      - all_subtrades: list of lists, one per trade. Each inner list has dicts:
        {entry_price, exit_price, pnl, reason, day_start, day_end}
    """
    m5_dates = sorted(df_m5["date"].unique())
    slip = SLIPPAGE_BPS / 10_000

    trail_results = []
    all_subtrades = []

    for _, trade in trades.iterrows():
        signal_date = trade["date"]
        direction = int(trade["direction"])

        future_dates = [d for d in m5_dates if d > signal_date]
        if len(future_dates) == 0:
            trail_results.append({"week_pnl": None, "reason": "no_bars", "n_subtrades": 0})
            all_subtrades.append([])
            continue

        holding_dates = future_dates[:HORIZON]
        week_pnl = 0.0
        last_reason = "no_bars"
        day_idx = 0
        need_entry = True
        week_subtrades = []
        subtrade_entry_day = None

        while day_idx < len(holding_dates):
            day = holding_dates[day_idx]
            bars = df_m5[df_m5["date"] == day].sort_values("timestamp")
            if len(bars) == 0:
                day_idx += 1
                continue

            if need_entry:
                entry_price = float(bars.iloc[0]["open"])
                slipped_entry = entry_price * (1 + slip) if direction == 1 else entry_price * (1 - slip)
                tracker = TrailingStopTracker(entry_price=slipped_entry, direction=direction, config=config)
                need_entry = False
                subtrade_entry_day = day_idx

            triggered = False
            for bar_idx, (_, bar) in enumerate(bars.iterrows()):
                state = tracker.update(
                    bar_high=float(bar["high"]), bar_low=float(bar["low"]),
                    bar_close=float(bar["close"]), bar_idx=bar_idx,
                )
                if state == TrailingState.TRIGGERED:
                    triggered = True
                    break

            if triggered:
                exit_price = tracker.exit_price
                exit_price = exit_price * (1 - slip) if direction == 1 else exit_price * (1 + slip)
                sub_pnl = direction * (exit_price - slipped_entry) / slipped_entry
                week_pnl += sub_pnl
                last_reason = tracker.exit_reason

                week_subtrades.append({
                    "entry_price": slipped_entry,
                    "exit_price": exit_price,
                    "pnl": sub_pnl,
                    "reason": tracker.exit_reason,
                    "day_start": subtrade_entry_day,
                    "day_end": day_idx,
                })

                if day_idx < len(holding_dates) - 1:
                    need_entry = True
                    day_idx += 1
                    continue
                else:
                    break
            else:
                if day_idx == len(holding_dates) - 1:
                    last_close = float(bars.iloc[-1]["close"])
                    exit_price = last_close * (1 - slip) if direction == 1 else last_close * (1 + slip)
                    sub_pnl = direction * (exit_price - slipped_entry) / slipped_entry
                    week_pnl += sub_pnl
                    last_reason = "week_end"

                    week_subtrades.append({
                        "entry_price": slipped_entry,
                        "exit_price": exit_price,
                        "pnl": sub_pnl,
                        "reason": "week_end",
                        "day_start": subtrade_entry_day if subtrade_entry_day is not None else day_idx,
                        "day_end": day_idx,
                    })

            day_idx += 1

        trail_results.append({"week_pnl": week_pnl, "reason": last_reason,
                              "n_subtrades": len(week_subtrades)})
        all_subtrades.append(week_subtrades)

    trades = trades.copy()
    tr_df = pd.DataFrame(trail_results)
    trades["week_pnl"] = tr_df["week_pnl"].values
    trades["exit_reason"] = tr_df["reason"].values
    trades["n_subtrades"] = tr_df["n_subtrades"].values
    return trades, all_subtrades


# =============================================================================
# SHARPE COMPUTATION
# =============================================================================

def compute_sharpe(returns: np.ndarray, periods_per_year: float) -> Dict:
    """Compute annualized Sharpe and related stats."""
    n = len(returns)
    if n < 2:
        return {"sharpe": 0, "mean_ret": 0, "std_ret": 0, "ann_ret": 0, "ann_vol": 0, "n": n}

    total_ret = np.prod(1 + returns) - 1
    ann_ret = (1 + total_ret) ** (periods_per_year / n) - 1
    ann_vol = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    t_stat, p_val = stats.ttest_1samp(returns, 0)

    return {
        "sharpe": round(sharpe, 3),
        "mean_ret_pct": round(np.mean(returns) * 100, 4),
        "std_ret_pct": round(np.std(returns, ddof=1) * 100, 4),
        "ann_ret_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "total_ret_pct": round(total_ret * 100, 2),
        "n": n,
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("  SHARPE DECOMPOSITION: LINEAR-ONLY H=5 OOS 2025")
    print("  Reported Sharpe: 3.461 (from diagnose_h5_ensemble.py)")
    print("  Question: Is this real or inflated by correlated sub-trades?")
    print("=" * 100)

    # 1. Load data + predict
    df = load_daily_data()
    df_m5 = load_5min_data()
    feature_cols = list(FEATURE_COLUMNS)

    logger.info("\n--- Generating LINEAR-ONLY predictions (monthly re-training) ---")
    predictions = predict_linear_oos(df, feature_cols)
    logger.info(f"  Predictions: {len(predictions)} days")

    # 2. Select weekly trades
    weekly_idx = select_weekly_trades(predictions)
    trades = predictions.iloc[weekly_idx].copy().reset_index(drop=True)
    logger.info(f"  Weekly trades: {len(trades)}")

    # 3. Trailing stop with sub-trade tracking
    logger.info("\n--- Simulating trailing stop with sub-trade tracking ---")
    trades_with_trail, all_subtrades = simulate_trailing_detailed(trades, df_m5, TRAIL_CONFIG)

    # 4. Compute weekly returns (leveraged)
    valid = trades_with_trail[trades_with_trail["week_pnl"].notna()].copy()
    weekly_rets = valid["week_pnl"].values * valid["leverage"].values
    n_weekly = len(weekly_rets)

    # 5. Compute sub-trade returns (leveraged)
    all_subtrade_rets = []
    subtrade_counts = []
    for i, subtrades in enumerate(all_subtrades):
        if i >= len(valid):
            break
        lev = valid.iloc[i]["leverage"]
        n_sub = len(subtrades)
        subtrade_counts.append(n_sub)
        for st in subtrades:
            all_subtrade_rets.append(st["pnl"] * lev)

    subtrade_rets = np.array(all_subtrade_rets)
    n_subtrades_total = len(subtrade_rets)

    # 6. Compute both Sharpe values
    trades_per_year = 252 / HORIZON  # ~50.4 weekly signals per year

    weekly_sharpe = compute_sharpe(weekly_rets, trades_per_year)

    # For sub-trades, estimate periods per year:
    # If avg 1.5 sub-trades per week, then ~75.6 sub-trades/year
    avg_subtrades = n_subtrades_total / n_weekly if n_weekly > 0 else 1
    subtrades_per_year = trades_per_year * avg_subtrades
    subtrade_sharpe = compute_sharpe(subtrade_rets, subtrades_per_year)

    # 7. Print results
    print("\n" + "=" * 100)
    print("  SHARPE DECOMPOSITION RESULTS")
    print("=" * 100)

    print(f"\n  1. WEEKLY SIGNAL SHARPE (N={n_weekly}, the TRUE metric)")
    print(f"     Sharpe:      {weekly_sharpe['sharpe']:+.3f}")
    print(f"     Ann. Return: {weekly_sharpe['ann_ret_pct']:+.2f}%")
    print(f"     Ann. Vol:    {weekly_sharpe['ann_vol_pct']:.2f}%")
    print(f"     Total Ret:   {weekly_sharpe['total_ret_pct']:+.2f}%")
    print(f"     Mean/week:   {weekly_sharpe['mean_ret_pct']:+.4f}%")
    print(f"     Std/week:    {weekly_sharpe['std_ret_pct']:.4f}%")
    print(f"     t-stat:      {weekly_sharpe['t_stat']:.3f}")
    print(f"     p-value:     {weekly_sharpe['p_value']:.4f}")

    print(f"\n  2. SUB-TRADE SHARPE (N={n_subtrades_total}, INFLATED if sub-trades correlated)")
    print(f"     Sharpe:      {subtrade_sharpe['sharpe']:+.3f}")
    print(f"     Ann. Return: {subtrade_sharpe['ann_ret_pct']:+.2f}%")
    print(f"     Ann. Vol:    {subtrade_sharpe['ann_vol_pct']:.2f}%")
    print(f"     Total Ret:   {subtrade_sharpe['total_ret_pct']:+.2f}%")
    print(f"     Mean/trade:  {subtrade_sharpe['mean_ret_pct']:+.4f}%")
    print(f"     Std/trade:   {subtrade_sharpe['std_ret_pct']:.4f}%")
    print(f"     t-stat:      {subtrade_sharpe['t_stat']:.3f}")
    print(f"     p-value:     {subtrade_sharpe['p_value']:.4f}")

    # 8. Sub-trade distribution
    print(f"\n  3. SUB-TRADE DISTRIBUTION")
    print(f"     Total weekly signals:  {n_weekly}")
    print(f"     Total sub-trades:      {n_subtrades_total}")
    print(f"     Avg sub-trades/week:   {avg_subtrades:.2f}")
    print(f"     Median sub-trades/wk:  {np.median(subtrade_counts):.0f}")

    # Distribution
    from collections import Counter
    counts = Counter(subtrade_counts)
    print(f"\n     Distribution:")
    for k in sorted(counts.keys()):
        pct = counts[k] / n_weekly * 100
        bar = "#" * counts[k]
        print(f"       {k} sub-trade(s): {counts[k]:>3} weeks ({pct:>5.1f}%)  {bar}")

    # 9. Exit reason distribution across all sub-trades
    all_reasons = []
    for subtrades in all_subtrades:
        for st in subtrades:
            all_reasons.append(st["reason"])
    reason_counts = Counter(all_reasons)
    print(f"\n  4. EXIT REASON DISTRIBUTION (all sub-trades)")
    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        pct = count / n_subtrades_total * 100
        print(f"       {reason:<20}: {count:>3} ({pct:>5.1f}%)")

    # 10. Correlation analysis
    # For weeks with exactly 2 sub-trades, compute correlation of their returns
    pairs_first = []
    pairs_second = []
    for subtrades in all_subtrades:
        if len(subtrades) == 2:
            pairs_first.append(subtrades[0]["pnl"])
            pairs_second.append(subtrades[1]["pnl"])

    print(f"\n  5. INTRA-WEEK SUB-TRADE CORRELATION")
    if len(pairs_first) >= 3:
        corr, corr_p = stats.pearsonr(pairs_first, pairs_second)
        print(f"     Weeks with exactly 2 sub-trades: {len(pairs_first)}")
        print(f"     Correlation (1st vs 2nd):         {corr:+.3f} (p={corr_p:.3f})")
        if abs(corr) < 0.3:
            print(f"     -> Low correlation: sub-trades are approximately independent")
        elif abs(corr) < 0.6:
            print(f"     -> Moderate correlation: some Sharpe inflation from sub-trade counting")
        else:
            print(f"     -> HIGH correlation: Sharpe per sub-trade is significantly inflated")
    else:
        print(f"     Not enough weeks with exactly 2 sub-trades ({len(pairs_first)}) for correlation analysis")

    # 11. Compare win streaks (are returns positively autocorrelated?)
    if n_subtrades_total > 5:
        autocorr = np.corrcoef(subtrade_rets[:-1], subtrade_rets[1:])[0, 1]
        print(f"\n     Sub-trade return autocorrelation (lag-1): {autocorr:+.3f}")
        if abs(autocorr) < 0.15:
            print(f"     -> Negligible autocorrelation: no serial dependence concern")
        else:
            print(f"     -> Notable autocorrelation: returns are serially dependent")

    # 12. Verdict
    print("\n" + "=" * 100)
    print("  VERDICT")
    print("=" * 100)

    sharpe_ratio = weekly_sharpe["sharpe"] / subtrade_sharpe["sharpe"] if subtrade_sharpe["sharpe"] != 0 else 1
    print(f"\n  Weekly Sharpe / Sub-trade Sharpe = {weekly_sharpe['sharpe']:.3f} / {subtrade_sharpe['sharpe']:.3f} = {sharpe_ratio:.2f}")

    if abs(weekly_sharpe["sharpe"] - subtrade_sharpe["sharpe"]) < 0.5:
        print("  -> Sharpe is CONSISTENT between weekly and sub-trade levels")
        print("  -> The 3.46 Sharpe is NOT inflated by sub-trade counting")
    elif weekly_sharpe["sharpe"] > subtrade_sharpe["sharpe"]:
        print("  -> Weekly Sharpe HIGHER than sub-trade Sharpe (aggregation helps)")
        print("  -> The 3.46 Sharpe is CONSERVATIVE (not inflated)")
    else:
        inflation = subtrade_sharpe["sharpe"] - weekly_sharpe["sharpe"]
        print(f"  -> Sub-trade Sharpe exceeds weekly by {inflation:.3f}")
        print(f"  -> Some inflation from correlated sub-trades within weeks")
        print(f"  -> TRUE Sharpe (weekly): {weekly_sharpe['sharpe']:.3f}")

    print(f"\n  Weekly p-value: {weekly_sharpe['p_value']:.4f} {'< 0.05 SIGNIFICANT' if weekly_sharpe['p_value'] < 0.05 else '>= 0.05 NOT SIGNIFICANT'}")

    # 13. Save JSON
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "h5_sharpe_decomposition.json"
    json_data = {
        "weekly_sharpe": weekly_sharpe,
        "subtrade_sharpe": subtrade_sharpe,
        "subtrade_distribution": {
            "total_weekly_signals": n_weekly,
            "total_subtrades": n_subtrades_total,
            "avg_subtrades_per_week": round(avg_subtrades, 2),
            "distribution": {str(k): v for k, v in sorted(counts.items())},
        },
        "exit_reasons": {k: v for k, v in sorted(reason_counts.items(), key=lambda x: -x[1])},
        "config": {
            "ensemble": "LINEAR-ONLY (ridge + bayesian_ridge)",
            "trail": {"activation": TRAIL_CONFIG.activation_pct, "trail": TRAIL_CONFIG.trail_pct,
                       "hard_stop": TRAIL_CONFIG.hard_stop_pct},
            "vol_target": {"target_vol": VOL_CONFIG.target_vol, "max_lev": VOL_CONFIG.max_leverage},
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
