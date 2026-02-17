"""
Walk-Forward Validation: LINEAR-ONLY (Ridge + BayesianRidge) on H=5
====================================================================

3 expanding folds:
  Fold 1: Train 2020-2021 -> Test 2022
  Fold 2: Train 2020-2022 -> Test 2023
  Fold 3: Train 2020-2023 -> Test 2024

Pass criteria: Sharpe > 1.0 in at least 2/3 years.

Uses same config as diagnose_h5_ensemble.py:
  - Monthly re-training within each test year
  - Vol-targeting: tv=0.15, max_lev=2.0
  - Trailing stop: activation=0.40%, trail=0.30%, hard=4.00% (re-entry)
  - Ensemble: mean(ridge, bayesian_ridge)

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
# CONFIG
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

LINEAR_IDS = ["ridge", "bayesian_ridge"]  # Exactly what user requested

FOLDS = [
    {"name": "2022", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"name": "2023", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"name": "2024", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
]


# =============================================================================
# DATA LOADING
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
    logger.info(f"  Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
    return df


def load_5min_data(start_date: str = "2021-12-01") -> pd.DataFrame:
    """Load 5-min OHLCV for trailing stop simulation."""
    logger.info(f"Loading 5-min OHLCV from {start_date}...")
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
    df_m5 = df_m5[df_m5["date"] >= pd.Timestamp(start_date)].copy()
    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    logger.info(f"  5-min: {len(df_m5)} bars from {df_m5['date'].min().date()} to {df_m5['date'].max().date()}")
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
# MODEL TRAINING
# =============================================================================

def train_linear_models(df_train: pd.DataFrame, feature_cols: List[str],
                        target_col: str = "target_5d") -> Tuple[Dict, StandardScaler]:
    """Train only Ridge + BayesianRidge."""
    horizon_config = get_horizon_config(HORIZON)

    X_train = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train[target_col].values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    trained = {}
    for model_id in LINEAR_IDS:
        try:
            model = ModelFactory.create(model_id, params=None, horizon=HORIZON)
            model.fit(X_scaled, y_train)
            trained[model_id] = model
        except Exception as e:
            logger.warning(f"  Failed to train {model_id}: {e}")

    return trained, scaler


# =============================================================================
# PREDICTION + ENSEMBLE
# =============================================================================

def predict_fold(df: pd.DataFrame, feature_cols: List[str],
                 test_start: str, test_end: str) -> pd.DataFrame:
    """
    Monthly re-training within the test year. LINEAR-ONLY ensemble.
    Returns DataFrame with date, close, actual_return, ensemble_pred, direction, leverage.
    """
    target_col = "target_5d"
    df_test = df[(df["date"] >= pd.Timestamp(test_start)) &
                 (df["date"] <= pd.Timestamp(test_end))].copy()
    df_test = df_test[df_test[target_col].notna()]

    months = sorted(df_test["date"].dt.to_period("M").unique())
    all_rows = []

    for month in months:
        month_start = month.start_time

        # Expanding window: train on all data from TRAIN_START to month start
        df_month_train = df[(df["date"] >= TRAIN_START) & (df["date"] < month_start)].copy()
        df_month_train = df_month_train[df_month_train[target_col].notna()]

        df_month_test = df_test[(df_test["date"] >= month_start) &
                                (df_test["date"] <= month.end_time)]

        if len(df_month_test) == 0 or len(df_month_train) < 200:
            continue

        trained, scaler = train_linear_models(df_month_train, feature_cols)

        for _, row in df_month_test.iterrows():
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
            if len(idx_list) > 0:
                idx = idx_list[0]
                if idx >= VOL_CONFIG.vol_lookback:
                    recent = df.iloc[idx - VOL_CONFIG.vol_lookback:idx]["return_1d"].values
                    realized_vol = compute_realized_vol(recent, VOL_CONFIG.vol_lookback)
                else:
                    realized_vol = 0.10
            else:
                realized_vol = 0.10

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
# TRAILING STOP (same as diagnose_h5_ensemble.py)
# =============================================================================

def simulate_trailing(trades: pd.DataFrame, df_m5: pd.DataFrame,
                      config: TrailingStopConfig) -> pd.DataFrame:
    m5_dates = sorted(df_m5["date"].unique())
    slip = SLIPPAGE_BPS / 10_000

    trail_results = []
    for _, trade in trades.iterrows():
        signal_date = trade["date"]
        direction = int(trade["direction"])

        future_dates = [d for d in m5_dates if d > signal_date]
        if len(future_dates) == 0:
            trail_results.append({"week_pnl": None, "reason": "no_bars", "n_subtrades": 0})
            continue

        holding_dates = future_dates[:HORIZON]
        week_pnl = 0.0
        last_reason = "no_bars"
        day_idx = 0
        need_entry = True
        n_subtrades = 0

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
                n_subtrades += 1

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
                    n_subtrades += 1

            day_idx += 1

        trail_results.append({"week_pnl": week_pnl, "reason": last_reason, "n_subtrades": n_subtrades})

    trades = trades.copy()
    tr_df = pd.DataFrame(trail_results)
    trades["week_pnl"] = tr_df["week_pnl"].values
    trades["exit_reason"] = tr_df["reason"].values
    trades["n_subtrades"] = tr_df["n_subtrades"].values
    return trades


# =============================================================================
# STATS
# =============================================================================

def compute_fold_stats(trades: pd.DataFrame) -> Dict:
    valid = trades[trades["week_pnl"].notna()].copy()
    if len(valid) == 0:
        return {"n_trades": 0}

    rets = valid["week_pnl"].values * valid["leverage"].values
    n = len(rets)

    # DA
    actual_signs = np.sign(valid["actual_return"].values)
    pred_dirs = valid["direction"].values
    da = (pred_dirs == actual_signs).sum() / n * 100

    # Total return (compounding)
    total_ret = np.prod(1 + rets) - 1

    # Annualized stats
    trades_per_year = 252 / HORIZON
    ann_ret = (1 + total_ret) ** (trades_per_year / n) - 1 if n > 0 else 0
    vol = np.std(rets, ddof=1) * np.sqrt(trades_per_year) if n > 1 else 1e-10
    sharpe = ann_ret / vol if vol > 0 else 0

    # Win rate, profit factor
    wins = np.sum(rets > 0)
    wr = wins / n * 100
    pos_sum = np.sum(rets[rets > 0])
    neg_sum = np.abs(np.sum(rets[rets < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")

    # Max drawdown
    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0

    # LONG/SHORT breakdown
    long_mask = valid["direction"].values == 1
    short_mask = valid["direction"].values == -1
    n_long = int(np.sum(long_mask))
    n_short = int(np.sum(short_mask))
    long_wr = np.sum(rets[long_mask] > 0) / n_long * 100 if n_long > 0 else 0
    short_wr = np.sum(rets[short_mask] > 0) / n_short * 100 if n_short > 0 else 0

    # t-test
    t_stat, p_val = stats.ttest_1samp(rets, 0) if n > 1 else (0, 1)

    # Bootstrap CI
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(10_000):
        sample = rng.choice(rets, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)

    return {
        "n_trades": n,
        "da_pct": round(da, 1),
        "n_long": n_long,
        "n_short": n_short,
        "long_wr_pct": round(long_wr, 1),
        "short_wr_pct": round(short_wr, 1),
        "total_return_pct": round(total_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "bootstrap_ci_lo": round(ci_lo * 100, 3),
        "bootstrap_ci_hi": round(ci_hi * 100, 3),
        "final_equity": round(INITIAL_CAPITAL * (1 + total_ret), 2),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("  WALK-FORWARD VALIDATION: LINEAR-ONLY (Ridge + BayesianRidge) H=5")
    print("  3 Expanding Folds: 2022, 2023, 2024")
    print("  Pass criteria: Sharpe > 1.0 in at least 2/3 years")
    print("=" * 100)

    # 1. Load data
    df = load_daily_data()
    df_m5 = load_5min_data(start_date="2021-12-01")
    feature_cols = list(FEATURE_COLUMNS)

    # 2. Run each fold
    fold_results = {}
    all_fold_trades = {}

    for fold in FOLDS:
        fold_name = fold["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"  FOLD: Test {fold_name} (Train 2020-{fold['train_end'][:4]})")
        logger.info(f"{'='*60}")

        # Predict
        predictions = predict_fold(df, feature_cols, fold["test_start"], fold["test_end"])
        if len(predictions) == 0:
            logger.warning(f"  No predictions for fold {fold_name}")
            fold_results[fold_name] = {"n_trades": 0}
            continue

        logger.info(f"  Predictions: {len(predictions)} days, "
                     f"L={int((predictions['direction']==1).sum())}, "
                     f"S={int((predictions['direction']==-1).sum())}")

        # Select weekly trades
        weekly_idx = select_weekly_trades(predictions)
        trades = predictions.iloc[weekly_idx].copy().reset_index(drop=True)
        logger.info(f"  Weekly trades: {len(trades)}")

        # Trailing stop
        trades_with_trail = simulate_trailing(trades, df_m5, TRAIL_CONFIG)
        all_fold_trades[fold_name] = trades_with_trail

        # Stats
        fold_stats = compute_fold_stats(trades_with_trail)
        fold_results[fold_name] = fold_stats
        logger.info(f"  Return: {fold_stats.get('total_return_pct', 0):+.2f}%, "
                     f"Sharpe: {fold_stats.get('sharpe', 0):+.3f}, "
                     f"p={fold_stats.get('p_value', 1):.4f}")

    # 3. Print results
    print("\n" + "=" * 110)
    print("  WALK-FORWARD RESULTS: LINEAR-ONLY (Ridge + BayesianRidge) H=5")
    print("  Training: Monthly expanding window | Trail: 0.40/0.30/4.00 | Vol-target: 0.15")
    print("=" * 110)

    header = (f"  {'Fold':<8} {'Train':<20} {'Trades':>6} {'DA%':>6} {'L/S':>7} "
              f"{'LONG WR':>8} {'SHORT WR':>9} {'Return%':>9} {'Sharpe':>7} "
              f"{'WR%':>6} {'PF':>6} {'p-val':>7} {'$10K->':>8}")
    print(header)
    print("  " + "-" * 106)

    for fold in FOLDS:
        name = fold["name"]
        s = fold_results.get(name, {})
        if s.get("n_trades", 0) == 0:
            print(f"  {name:<8} {'---':<20} {'N/A':>6}")
            continue
        train_label = f"2020-{fold['train_end'][:4]}"
        ls = f"{s['n_long']}/{s['n_short']}"
        sharpe_marker = " *" if s["sharpe"] > 1.0 else ""
        print(f"  {name:<8} {train_label:<20} {s['n_trades']:>6} {s['da_pct']:>5.1f}% {ls:>7} "
              f"{s['long_wr_pct']:>7.1f}% {s['short_wr_pct']:>8.1f}% "
              f"{s['total_return_pct']:>+8.2f}% {s['sharpe']:>+6.3f}{sharpe_marker} "
              f"{s['win_rate_pct']:>5.1f}% {s['profit_factor']:>5.3f} {s['p_value']:>7.4f} "
              f"${s['final_equity']:>7.0f}")

    # 4. Combined 2025 reference (from previous results)
    print()
    print(f"  {'2025':.<8} {'2020-2024 (ref)':.<20} {'44':>6} {'59.1':>5}% {'13/31':>7} "
          f"{'69.2':>7}% {'71.0':>8}% {'+32.85':>8}% {'+3.461':>6} "
          f"{'70.5':>5}% {'2.923':>5} {'0.0077':>7} {'$13285':>8}")

    # 5. Bootstrap CIs
    print("\n  Bootstrap 95% CI:")
    for fold in FOLDS:
        name = fold["name"]
        s = fold_results.get(name, {})
        if s.get("n_trades", 0) == 0:
            continue
        ci = f"[{s['bootstrap_ci_lo']:+.3f}%, {s['bootstrap_ci_hi']:+.3f}%]"
        excludes = "Excludes 0" if s["bootstrap_ci_lo"] > 0 else "INCLUDES 0"
        print(f"  {name}: {ci}  {excludes}")

    # 6. Pass/Fail verdict
    n_pass = sum(1 for f in FOLDS if fold_results.get(f["name"], {}).get("sharpe", 0) > 1.0)
    n_total = len(FOLDS)

    print("\n" + "=" * 110)
    print(f"  VERDICT: {n_pass}/{n_total} folds with Sharpe > 1.0")
    if n_pass >= 2:
        print("  >>> PASS: LINEAR-ONLY is robust across multiple years")
    else:
        print("  >>> FAIL: LINEAR-ONLY does NOT generalize (possible 2025 overfitting)")
    print("=" * 110)

    # 7. Buy-and-hold comparison per year
    print("\n  BUY & HOLD USDCOP comparison:")
    for fold in FOLDS:
        name = fold["name"]
        fold_df = df[(df["date"] >= pd.Timestamp(fold["test_start"])) &
                     (df["date"] <= pd.Timestamp(fold["test_end"]))].copy()
        if len(fold_df) > 0:
            bh_ret = (fold_df.iloc[-1]["close"] / fold_df.iloc[0]["close"] - 1) * 100
            strat_ret = fold_results.get(name, {}).get("total_return_pct", 0)
            alpha = strat_ret - bh_ret
            print(f"  {name}: B&H={bh_ret:+.1f}%, Strategy={strat_ret:+.1f}%, Alpha={alpha:+.1f}pp")

    # 8. Save JSON
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "h5_linear_walkforward.json"
    json_data = {
        "folds": {name: fold_results.get(name, {}) for name in [f["name"] for f in FOLDS]},
        "config": {
            "ensemble": "LINEAR-ONLY (ridge + bayesian_ridge)",
            "trail": {"activation": TRAIL_CONFIG.activation_pct, "trail": TRAIL_CONFIG.trail_pct,
                       "hard_stop": TRAIL_CONFIG.hard_stop_pct},
            "vol_target": {"target_vol": VOL_CONFIG.target_vol, "max_lev": VOL_CONFIG.max_leverage},
            "horizon": HORIZON,
        },
        "verdict": {
            "n_pass": n_pass,
            "n_total": n_total,
            "pass": n_pass >= 2,
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"\n  Saved: {json_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
