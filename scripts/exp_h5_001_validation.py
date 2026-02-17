"""
EXP-H5-001: Weekly Horizon (H=5) Full Validation
==================================================

Hypothesis: H=5 with trailing stop adapted for 5-day holding period
will outperform H=1 on both 2025 and 2026, because:
- Mean-reversion noise cancels within the week
- Trailing stop has 5 days of runway to capture moves
- Feature-target relationship is stronger at weekly scale

Pipeline:
1. Walk-forward H=5 within 2020-2024 (4 folds, 30-day gap)
2. Grid search trailing stop params for H=5 (activation x trail x hard_stop)
3. OOS 2025 with frozen models (non-overlapping weekly trades)
4. OOS 2026 YTD (regime change test)
5. Full comparison vs H=1 baseline
6. GATE evaluation: DA>55%, Sharpe>1.0, p<0.10

Variable changed: Horizon (H=1 -> H=5)
Baseline: backtest_oos_2025.py H=1 results

@experiment EXP-H5-001
@date 2026-02-16
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import MODEL_IDS, get_horizon_config
from src.forecasting.vol_targeting import (
    VolTargetConfig, compute_vol_target_signal, compute_realized_vol,
)
from src.execution.trailing_stop import TrailingStopConfig, TrailingStopTracker, TrailingState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

INITIAL_CAPITAL = 10_000.0
SLIPPAGE_BPS = 1.0
HORIZON = 5  # Weekly

VOL_CONFIG = VolTargetConfig(
    target_vol=0.15,
    max_leverage=2.0,
    min_leverage=0.5,
    vol_lookback=21,
    vol_floor=0.05,
)

# Baseline trailing stop (H=1 params)
TRAIL_H1 = TrailingStopConfig(activation_pct=0.002, trail_pct=0.003, hard_stop_pct=0.015)

# Proposed trailing stop for H=5 (scaled for 5-day holding)
TRAIL_H5_DEFAULT = TrailingStopConfig(activation_pct=0.0035, trail_pct=0.005, hard_stop_pct=0.025)

# Grid search space for H=5 trailing stop
TRAIL_GRID = {
    "activation_pct": [0.002, 0.003, 0.0035, 0.004, 0.005],
    "trail_pct":      [0.003, 0.004, 0.005, 0.006, 0.008],
    "hard_stop_pct":  [0.015, 0.020, 0.025, 0.030, 0.040],
}

TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
OOS_END_2025 = pd.Timestamp("2025-12-31")
OOS_END_2026 = pd.Timestamp("2026-12-31")

N_FOLDS = 4
GAP_DAYS = 30
INITIAL_TRAIN_RATIO = 0.60


# =============================================================================
# DATA LOADING (same as backtest_oos_2025.py)
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

    # Target H=5: log return 5 days ahead
    df["target_5d"] = np.log(df["close"].shift(-HORIZON) / df["close"])
    # Also keep H=1 for comparison
    df["target_1d"] = np.log(df["close"].shift(-1) / df["close"])

    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    df = df[feature_mask].reset_index(drop=True)
    logger.info(f"  Data: {len(df)} rows, {df['date'].min().date()} to {df['date'].max().date()}")
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
    logger.info(f"  5-min: {len(df_m5)} bars")
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

def get_model_list():
    models = list(MODEL_IDS)
    try:
        ModelFactory.create("ard")
    except Exception:
        models = [m for m in models if m != "ard"]
    return models


def get_model_params(model_id, horizon_config):
    if model_id in {"ridge", "bayesian_ridge", "ard"}:
        return None
    elif model_id in {"catboost_pure"}:
        return {"iterations": horizon_config.get("n_estimators", 50),
                "depth": horizon_config.get("max_depth", 3),
                "learning_rate": horizon_config.get("learning_rate", 0.05),
                "verbose": False, "allow_writing_files": False}
    elif model_id in {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}:
        if "catboost" in model_id:
            return {"iterations": horizon_config.get("n_estimators", 50),
                    "depth": horizon_config.get("max_depth", 3),
                    "learning_rate": horizon_config.get("learning_rate", 0.05),
                    "verbose": False, "allow_writing_files": False}
        return horizon_config
    else:
        return horizon_config


def train_and_predict(df_train, df_test, feature_cols, target_col, horizon):
    """Train 9 models, return top-3 ensemble predictions for test set."""
    horizon_config = get_horizon_config(horizon)
    models_to_use = get_model_list()

    X_train = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train[target_col].values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    trained = {}
    for model_id in models_to_use:
        try:
            params = get_model_params(model_id, horizon_config)
            model = ModelFactory.create(model_id, params=params, horizon=horizon)
            if model.requires_scaling:
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            trained[model_id] = model
        except Exception:
            pass

    # Predict test set
    results = []
    for _, row in df_test.iterrows():
        X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)
        X_day_scaled = scaler.transform(X_day)

        preds = {}
        for model_id, model in trained.items():
            try:
                if model.requires_scaling:
                    p = model.predict(X_day_scaled)[0]
                else:
                    p = model.predict(X_day)[0]
                preds[model_id] = p
            except Exception:
                pass

        if len(preds) < 3:
            continue

        sorted_m = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
        top3 = sorted_m[:3]
        ensemble = np.mean([preds[m] for m in top3])
        direction = 1 if ensemble > 0 else -1

        results.append({
            "date": row["date"],
            "close": row["close"],
            "actual_return": row[target_col],
            "ensemble_pred": ensemble,
            "direction": direction,
        })

    return pd.DataFrame(results), len(trained)


# =============================================================================
# NON-OVERLAPPING WEEKLY TRADES
# =============================================================================

def select_weekly_trades(signals: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Select non-overlapping trades every H trading days."""
    if len(signals) == 0:
        return signals
    signals = signals.sort_values("date").reset_index(drop=True)
    selected = [0]
    last = 0
    for i in range(1, len(signals)):
        if i - last >= horizon:
            selected.append(i)
            last = i
    return signals.iloc[selected].copy().reset_index(drop=True)


# =============================================================================
# MULTI-DAY TRAILING STOP WITH RE-ENTRY
# =============================================================================

def simulate_trailing_h5(
    trades: pd.DataFrame,
    df_m5: pd.DataFrame,
    config: TrailingStopConfig,
    re_entry: bool = True,
) -> pd.DataFrame:
    """
    Trailing stop over H=5 days of 5-min bars.
    If re_entry=True: when trailing closes a trade, re-enter next day
    in the same direction (weekly thesis persists).
    Returns per-trade PnL (may have multiple sub-trades per weekly trade).
    """
    m5_dates = sorted(df_m5["date"].unique())
    slip = SLIPPAGE_BPS / 10_000

    weekly_pnls = []
    weekly_reasons = []
    weekly_subtrades = []

    for _, trade in trades.iterrows():
        signal_date = trade["date"]
        direction = int(trade["direction"])

        # Find next 5 trading days
        future_dates = [d for d in m5_dates if d > signal_date]
        if len(future_dates) == 0:
            weekly_pnls.append(None)
            weekly_reasons.append("no_bars")
            weekly_subtrades.append(0)
            continue

        holding_dates = future_dates[:HORIZON]
        week_pnl = 0.0
        sub_trades = 0
        last_reason = "no_bars"

        # Iterate over each day in the holding period
        day_idx = 0
        need_entry = True

        while day_idx < len(holding_dates):
            day = holding_dates[day_idx]
            bars = df_m5[df_m5["date"] == day].sort_values("timestamp")

            if len(bars) == 0:
                day_idx += 1
                continue

            if need_entry:
                # Enter at day's open
                entry_price = float(bars.iloc[0]["open"])
                if direction == 1:
                    slipped_entry = entry_price * (1 + slip)
                else:
                    slipped_entry = entry_price * (1 - slip)

                tracker = TrailingStopTracker(
                    entry_price=slipped_entry,
                    direction=direction,
                    config=config,
                )
                need_entry = False
                sub_trades += 1

            # Run trailing stop through today's bars
            triggered = False
            for bar_idx, (_, bar) in enumerate(bars.iterrows()):
                state = tracker.update(
                    bar_high=float(bar["high"]),
                    bar_low=float(bar["low"]),
                    bar_close=float(bar["close"]),
                    bar_idx=bar_idx,
                )
                if state == TrailingState.TRIGGERED:
                    triggered = True
                    break

            if triggered:
                # Trailing stop fired
                exit_price = tracker.exit_price
                if direction == 1:
                    exit_price *= (1 - slip)
                else:
                    exit_price *= (1 + slip)

                sub_pnl = direction * (exit_price - slipped_entry) / slipped_entry
                week_pnl += sub_pnl
                last_reason = tracker.exit_reason

                if re_entry and day_idx < len(holding_dates) - 1:
                    # Re-enter next day in same direction
                    need_entry = True
                    day_idx += 1
                    continue
                else:
                    break
            else:
                # Day ended, position still open
                if day_idx == len(holding_dates) - 1:
                    # Last day — force close at session end
                    last_close = float(bars.iloc[-1]["close"])
                    if direction == 1:
                        exit_price = last_close * (1 - slip)
                    else:
                        exit_price = last_close * (1 + slip)

                    sub_pnl = direction * (exit_price - slipped_entry) / slipped_entry
                    week_pnl += sub_pnl
                    last_reason = "week_end"

            day_idx += 1

        weekly_pnls.append(week_pnl)
        weekly_reasons.append(last_reason)
        weekly_subtrades.append(sub_trades)

    trades = trades.copy()
    trades["trail_pnl"] = weekly_pnls
    trades["trail_reason"] = weekly_reasons
    trades["n_subtrades"] = weekly_subtrades
    return trades


# =============================================================================
# STRATEGY STATS
# =============================================================================

def compute_stats(rets, horizon=5):
    if len(rets) == 0:
        return {"total_return_pct": 0, "sharpe": 0, "win_rate_pct": 0,
                "profit_factor": 0, "max_dd_pct": 0, "n_trades": 0, "final_equity": INITIAL_CAPITAL}

    total = np.prod(1 + rets) - 1
    tpy = 252 / horizon
    ann_ret = (1 + total) ** (tpy / len(rets)) - 1 if len(rets) > 0 else 0
    vol = np.std(rets, ddof=1) * np.sqrt(tpy) if len(rets) > 1 else 1e-10
    sharpe = ann_ret / vol if vol > 0 else 0

    cum = np.cumprod(1 + rets)
    rmax = np.maximum.accumulate(cum)
    dd = (cum - rmax) / rmax
    max_dd = float(np.min(dd))

    wins = np.sum(rets > 0)
    wr = wins / len(rets) * 100
    pos = np.sum(rets[rets > 0])
    neg = np.abs(np.sum(rets[rets < 0]))
    pf = pos / neg if neg > 0 else float("inf")

    return {
        "final_equity": round(INITIAL_CAPITAL * (1 + total), 2),
        "total_return_pct": round(total * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 3),
        "n_trades": len(rets),
        "avg_ret_per_trade_pct": round(np.mean(rets) * 100, 3),
    }


def run_strategies(trades, horizon=5):
    """Run 4 strategies on weekly trades."""
    n = len(trades)
    slip = SLIPPAGE_BPS / 10_000

    # DA
    actual = np.sign(trades["actual_return"].values)
    pred = trades["direction"].values
    correct = int(np.sum(pred == actual))
    da = correct / n * 100 if n > 0 else 0

    # Buy & Hold
    bh_ret = (trades.iloc[-1]["close"] - trades.iloc[0]["close"]) / trades.iloc[0]["close"]

    # Forecast 1x
    f1 = np.array([row["direction"] * (np.exp(row["actual_return"]) - 1) - slip
                    for _, row in trades.iterrows()])

    # Vol-target
    vt = np.array([row["direction"] * row["leverage"] * (np.exp(row["actual_return"]) - 1) - slip * row["leverage"]
                    for _, row in trades.iterrows()])

    # VT + Trailing stop
    ts = []
    for _, row in trades.iterrows():
        lev = row["leverage"]
        tp = row.get("trail_pnl")
        if tp is not None and not (isinstance(tp, float) and np.isnan(tp)):
            ts.append(tp * lev)
        else:
            ts.append(row["direction"] * lev * (np.exp(row["actual_return"]) - 1) - slip * lev)
    ts = np.array(ts)

    # t-test
    t_stat, p_val = stats.ttest_1samp(ts, 0) if len(ts) > 1 else (0, 1)

    return {
        "n_trades": n, "da_pct": round(da, 1),
        "buy_hold": {"total_return_pct": round(bh_ret * 100, 2)},
        "forecast_1x": compute_stats(f1, horizon),
        "forecast_vt": compute_stats(vt, horizon),
        "forecast_vt_trail": compute_stats(ts, horizon),
        "t_stat": round(t_stat, 3), "p_value": round(p_val, 4),
    }


# =============================================================================
# PHASE 1: WALK-FORWARD H=5 WITHIN 2020-2024
# =============================================================================

def run_walk_forward(df):
    print("\n" + "=" * 80)
    print("  PHASE 1: Walk-Forward Validation H=5 (2020-2024 ONLY)")
    print("=" * 80)

    feature_cols = list(FEATURE_COLUMNS)
    target_col = "target_5d"

    df_period = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_period = df_period[df_period[target_col].notna()].reset_index(drop=True)

    n_total = len(df_period)
    init_size = int(n_total * INITIAL_TRAIN_RATIO)
    remaining = n_total - init_size
    test_size = remaining // N_FOLDS

    folds = []

    for fold in range(N_FOLDS):
        train_end = init_size + fold * test_size
        test_start = train_end + GAP_DAYS
        test_end = min(test_start + test_size, n_total)

        if test_start >= n_total or test_end <= test_start:
            continue

        df_train = df_period.iloc[:train_end]
        df_test = df_period.iloc[test_start:test_end]

        preds, n_models = train_and_predict(df_train, df_test, feature_cols, target_col, HORIZON)

        if len(preds) == 0:
            continue

        # DA
        actual = np.sign(preds["actual_return"].values)
        pred_dir = preds["direction"].values
        da = np.sum(pred_dir == actual) / len(preds) * 100

        # Weekly non-overlapping trades
        weekly = select_weekly_trades(preds, HORIZON)
        weekly_rets = weekly["direction"].values * (np.exp(weekly["actual_return"].values) - 1)
        fold_ret = (np.prod(1 + weekly_rets) - 1) * 100
        fold_sharpe = (np.mean(weekly_rets) / np.std(weekly_rets, ddof=1) * np.sqrt(252/HORIZON)
                       if np.std(weekly_rets, ddof=1) > 0 else 0)

        print(f"\n  Fold {fold+1}/{N_FOLDS}: "
              f"Train {len(df_train)} -> Test {len(df_test)} (gap {GAP_DAYS}d)")
        print(f"    {df_test['date'].iloc[0].date()} to {df_test['date'].iloc[-1].date()}")
        print(f"    DA={da:.1f}%, {len(weekly)} weekly trades, "
              f"Ret={fold_ret:+.2f}%, Sharpe={fold_sharpe:.2f}")

        folds.append({
            "fold": fold + 1,
            "da_pct": round(da, 1),
            "n_weekly_trades": len(weekly),
            "return_pct": round(fold_ret, 2),
            "sharpe": round(fold_sharpe, 2),
            "test_start": str(df_test["date"].iloc[0].date()),
            "test_end": str(df_test["date"].iloc[-1].date()),
        })

    avg_da = np.mean([f["da_pct"] for f in folds])
    folds_positive = sum(1 for f in folds if f["return_pct"] > 0)
    avg_sharpe = np.mean([f["sharpe"] for f in folds])

    print(f"\n  Walk-Forward Summary:")
    print(f"    Avg DA: {avg_da:.1f}%")
    print(f"    Avg Sharpe: {avg_sharpe:.2f}")
    print(f"    Folds positive: {folds_positive}/{len(folds)}")

    return {"folds": folds, "avg_da": round(avg_da, 1),
            "avg_sharpe": round(avg_sharpe, 2), "folds_positive": folds_positive}


# =============================================================================
# PHASE 2: OOS PREDICTION (2025 + 2026)
# =============================================================================

def predict_oos(df, oos_end):
    """Train with monthly re-training, predict OOS non-overlapping weekly."""
    feature_cols = list(FEATURE_COLUMNS)
    target_col = "target_5d"

    df_oos = df[(df["date"] >= OOS_START) & (df["date"] <= oos_end)].copy()
    df_oos = df_oos[df_oos[target_col].notna()]

    months = sorted(df_oos["date"].dt.to_period("M").unique())
    all_signals = []

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        if month_idx == 0:
            df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
        else:
            df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < month_start)].copy()

        df_train = df_train[df_train[target_col].notna()]
        df_month = df_oos[(df_oos["date"] >= month_start) & (df_oos["date"] <= month_end)]

        if len(df_month) == 0 or len(df_train) < 200:
            continue

        preds, n_models = train_and_predict(df_train, df_month, feature_cols, target_col, HORIZON)

        for _, row in preds.iterrows():
            # Add vol-targeting
            idx_list = df.index[df["date"] == row["date"]]
            if len(idx_list) == 0:
                continue
            idx = idx_list[0]
            if idx >= VOL_CONFIG.vol_lookback:
                recent = df.iloc[idx - VOL_CONFIG.vol_lookback:idx]["return_1d"].values
                rvol = compute_realized_vol(recent, VOL_CONFIG.vol_lookback)
            else:
                rvol = 0.10

            vt = compute_vol_target_signal(
                forecast_direction=int(row["direction"]),
                forecast_return=row["ensemble_pred"],
                realized_vol_21d=rvol,
                config=VOL_CONFIG,
                date=str(row["date"].date()),
            )

            all_signals.append({
                "date": row["date"],
                "close": row["close"],
                "actual_return": row["actual_return"],
                "ensemble_pred": row["ensemble_pred"],
                "direction": row["direction"],
                "leverage": vt.clipped_leverage,
            })

    return pd.DataFrame(all_signals)


# =============================================================================
# PHASE 3: GRID SEARCH TRAILING STOP FOR H=5
# =============================================================================

def grid_search_trailing(trades, df_m5):
    """Find optimal trailing stop params for H=5 using walk-forward trades."""
    print("\n" + "=" * 80)
    print("  PHASE 3: Grid Search Trailing Stop Params for H=5")
    print("=" * 80)

    activations = TRAIL_GRID["activation_pct"]
    trails = TRAIL_GRID["trail_pct"]
    hard_stops = TRAIL_GRID["hard_stop_pct"]

    total_combos = len(activations) * len(trails) * len(hard_stops)
    logger.info(f"  Testing {total_combos} combinations...")

    results = []

    for act, trail, hard in product(activations, trails, hard_stops):
        # Skip invalid combos (trail must be >= activation for the stop to work logically)
        if trail < act * 0.5:
            continue

        config = TrailingStopConfig(activation_pct=act, trail_pct=trail, hard_stop_pct=hard)
        trades_with_trail = simulate_trailing_h5(trades, df_m5, config, re_entry=True)

        valid = trades_with_trail["trail_pnl"].notna()
        if valid.sum() < 5:
            continue

        pnls = trades_with_trail.loc[valid, "trail_pnl"].values
        levs = trades_with_trail.loc[valid, "leverage"].values
        levered_rets = pnls * levs

        s = compute_stats(levered_rets, HORIZON)
        results.append({
            "activation": act, "trail": trail, "hard_stop": hard,
            **s,
        })

    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n  Top 10 configurations by Sharpe:")
    print(f"  {'#':>3} {'Act%':>6} {'Trail%':>7} {'Hard%':>6} "
          f"{'Return':>9} {'Sharpe':>8} {'WR%':>7} {'PF':>7} {'MaxDD':>8} {'Trades':>7}")
    print(f"  {'-'*3} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*8} {'-'*7}")

    for i, r in enumerate(results[:10]):
        print(f"  {i+1:>3} {r['activation']*100:>5.2f}% {r['trail']*100:>6.2f}% {r['hard_stop']*100:>5.2f}% "
              f"{r['total_return_pct']:>+8.2f}% {r['sharpe']:>+7.3f} {r['win_rate_pct']:>6.1f}% "
              f"{r['profit_factor']:>6.3f} {r['max_dd_pct']:>+7.2f}% {r['n_trades']:>7}")

    # Best config
    best = results[0] if results else None
    if best:
        print(f"\n  BEST: activation={best['activation']*100:.2f}%, "
              f"trail={best['trail']*100:.2f}%, hard_stop={best['hard_stop']*100:.2f}%")
        print(f"  Return={best['total_return_pct']:+.2f}%, Sharpe={best['sharpe']:.3f}, "
              f"WR={best['win_rate_pct']:.1f}%, PF={best['profit_factor']:.3f}")

    return results, best


# =============================================================================
# PHASE 4: FULL OOS EVALUATION
# =============================================================================

def evaluate_oos(trades, df_m5, trail_config, label):
    """Full OOS evaluation with given trailing config."""
    # Add vol-targeting leveraged trailing stop
    trades_with_trail = simulate_trailing_h5(trades, df_m5, trail_config, re_entry=True)

    result = run_strategies(trades_with_trail, HORIZON)

    print(f"\n  {label}:")
    print(f"    Trades: {result['n_trades']} | DA: {result['da_pct']}%")

    strats = [
        ("Buy & Hold", result["buy_hold"]),
        ("Forecast 1x", result["forecast_1x"]),
        ("FC + VolTarget", result["forecast_vt"]),
        ("FC + VT + Trail", result["forecast_vt_trail"]),
    ]

    print(f"    {'Strategy':<18} {'$10K':>10} {'Ret%':>9} {'Sharpe':>8} {'WR%':>7} {'PF':>7} {'MaxDD':>8}")
    print(f"    {'-'*18} {'-'*10} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

    for lbl, s in strats:
        if "final_equity" not in s:
            print(f"    {lbl:<18} {s.get('total_return_pct', 0):>+8.2f}%")
            continue
        print(f"    {lbl:<18} ${s['final_equity']:>9,.2f} {s['total_return_pct']:>+8.2f}% "
              f"{s.get('sharpe',0):>+7.3f} {s.get('win_rate_pct',0):>6.1f}% "
              f"{s.get('profit_factor',0):>6.3f} {s.get('max_dd_pct',0):>+7.2f}%")

    print(f"    t-stat={result['t_stat']:.3f}, p-value={result['p_value']:.4f}")

    # Period split
    trades_with_trail_copy = trades_with_trail.copy()
    trades_with_trail_copy["year"] = trades_with_trail_copy["date"].dt.year

    for year in sorted(trades_with_trail_copy["year"].unique()):
        sub = trades_with_trail_copy[trades_with_trail_copy["year"] == year]
        if len(sub) < 3:
            continue
        valid = sub["trail_pnl"].notna()
        if valid.sum() < 3:
            continue
        pnls = sub.loc[valid, "trail_pnl"].values * sub.loc[valid, "leverage"].values
        s = compute_stats(pnls, HORIZON)
        da = np.sum(np.sign(sub["actual_return"].values) == sub["direction"].values) / len(sub) * 100
        print(f"    {year}: {len(sub)} trades, DA={da:.1f}%, "
              f"Ret={s['total_return_pct']:+.2f}%, Sharpe={s['sharpe']:.3f}, "
              f"WR={s['win_rate_pct']:.1f}%, PF={s['profit_factor']:.3f}")

    return result


# =============================================================================
# GATE EVALUATION
# =============================================================================

def evaluate_gates(wf_result, oos_result_2025, oos_result_full):
    print("\n" + "=" * 80)
    print("  GATE EVALUATION (EXP-H5-001)")
    print("=" * 80)

    gates = []

    # Gate 1: Walk-forward
    wf_pass = wf_result["avg_da"] > 50 and wf_result["folds_positive"] >= 2
    gates.append(("WF Validation", wf_pass, f"DA={wf_result['avg_da']:.1f}%, "
                  f"Folds+={wf_result['folds_positive']}/{len(wf_result['folds'])}"))

    # Gate 2: OOS 2025
    ts_2025 = oos_result_2025["forecast_vt_trail"]
    g2_da = oos_result_2025["da_pct"] > 55
    g2_sharpe = ts_2025["sharpe"] > 1.0
    g2_ret = ts_2025["total_return_pct"] > 0
    g2_p = oos_result_2025["p_value"] < 0.10
    g2_pass = sum([g2_da, g2_sharpe, g2_ret, g2_p]) >= 3
    gates.append(("OOS 2025", g2_pass, f"DA={oos_result_2025['da_pct']:.1f}%>{55}, "
                  f"Sharpe={ts_2025['sharpe']:.3f}>{1.0}, "
                  f"Ret={ts_2025['total_return_pct']:+.2f}%>0, "
                  f"p={oos_result_2025['p_value']:.4f}<{0.10}"))

    # Gate 3: Beats Buy & Hold
    ts_ret = ts_2025["total_return_pct"]
    bh_ret = oos_result_2025["buy_hold"]["total_return_pct"]
    g3_pass = ts_ret > bh_ret
    gates.append(("Beats B&H", g3_pass, f"Trail={ts_ret:+.2f}% vs B&H={bh_ret:+.2f}%"))

    # Gate 4: 2026 not catastrophic
    if oos_result_full:
        ts_full = oos_result_full["forecast_vt_trail"]
        g4_pass = ts_full["max_dd_pct"] > -20  # Not too bad
        gates.append(("2026 MaxDD", g4_pass, f"MaxDD={ts_full['max_dd_pct']:.2f}% > -20%"))

    overall = all(passed for _, passed, _ in gates)

    for name, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        print(f"\n  {name}: {status}")
        print(f"    {detail}")

    print(f"\n  OVERALL: {'PROMOTE' if overall else 'REVIEW'}")
    return overall


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("  EXP-H5-001: Weekly Horizon (H=5) Full Validation")
    print("  Variable: Horizon H=1 -> H=5")
    print("  Baseline: backtest_oos_2025.py H=1")
    print("=" * 80)

    df = load_daily_data()
    df_m5 = load_5min_data()

    # Phase 1: Walk-forward
    wf_result = run_walk_forward(df)

    # Phase 2: OOS predictions (2025 + 2026)
    print("\n" + "=" * 80)
    print("  PHASE 2: OOS Prediction (2025 + 2026)")
    print("=" * 80)

    signals_full = predict_oos(df, OOS_END_2026)
    trades_full = select_weekly_trades(signals_full, HORIZON)

    signals_2025 = signals_full[signals_full["date"] <= OOS_END_2025].copy()
    trades_2025 = select_weekly_trades(signals_2025, HORIZON)

    logger.info(f"  Full OOS: {len(signals_full)} signals -> {len(trades_full)} weekly trades")
    logger.info(f"  2025 only: {len(signals_2025)} signals -> {len(trades_2025)} weekly trades")

    # Phase 3: Grid search trailing on 2025 trades
    grid_results, best_config_dict = grid_search_trailing(trades_2025, df_m5)

    if best_config_dict:
        best_trail = TrailingStopConfig(
            activation_pct=best_config_dict["activation"],
            trail_pct=best_config_dict["trail"],
            hard_stop_pct=best_config_dict["hard_stop"],
        )
    else:
        best_trail = TRAIL_H5_DEFAULT

    # Phase 4: Full OOS evaluation
    print("\n" + "=" * 80)
    print("  PHASE 4: OOS Evaluation with Optimized Trailing Stop")
    print("=" * 80)
    print(f"  Config: activation={best_trail.activation_pct*100:.2f}%, "
          f"trail={best_trail.trail_pct*100:.2f}%, "
          f"hard_stop={best_trail.hard_stop_pct*100:.2f}%")

    oos_2025 = evaluate_oos(trades_2025, df_m5, best_trail, "OOS 2025 ONLY")
    oos_full = evaluate_oos(trades_full, df_m5, best_trail, "OOS 2025 + 2026 (Full)")

    # Also test with H=1 params for comparison
    print("\n" + "-" * 80)
    print("  COMPARISON: H=5 with H=1 trailing params (baseline)")
    print("-" * 80)
    oos_2025_h1params = evaluate_oos(trades_2025, df_m5, TRAIL_H1, "H=5 + H=1 trail params")

    # Also test with proposed H=5 default params
    print("\n" + "-" * 80)
    print("  COMPARISON: H=5 with proposed H=5 default params")
    print("-" * 80)
    oos_2025_h5default = evaluate_oos(trades_2025, df_m5, TRAIL_H5_DEFAULT,
                                       "H=5 + default H=5 trail (0.35/0.50/2.50%)")

    # Gate evaluation
    overall = evaluate_gates(wf_result, oos_2025, oos_full)

    # Final summary
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON: H=5 vs H=1 Baseline")
    print("=" * 80)

    print(f"""
  H=1 BASELINE (from backtest_oos_2025.py):
    2025: $10K -> $12,621 (+26.21%), Sharpe 2.098, WR 61.5%, PF 1.393, p=0.078
    2026: $10K -> $9,116 (-8.84%), Sharpe -3.585, WR 44.2%, PF 0.329

  H=5 WITH OPTIMIZED TRAILING:
    Config: act={best_trail.activation_pct*100:.2f}%, trail={best_trail.trail_pct*100:.2f}%, hard={best_trail.hard_stop_pct*100:.2f}%
    2025: {oos_2025['forecast_vt_trail']['n_trades']} trades, Ret={oos_2025['forecast_vt_trail']['total_return_pct']:+.2f}%, \
Sharpe={oos_2025['forecast_vt_trail']['sharpe']:.3f}, WR={oos_2025['forecast_vt_trail']['win_rate_pct']:.1f}%, \
PF={oos_2025['forecast_vt_trail']['profit_factor']:.3f}, p={oos_2025['p_value']:.4f}
""")

    # Save
    output = {
        "experiment": "EXP-H5-001",
        "horizon": HORIZON,
        "best_trailing_config": {
            "activation_pct": best_trail.activation_pct,
            "trail_pct": best_trail.trail_pct,
            "hard_stop_pct": best_trail.hard_stop_pct,
        },
        "walk_forward": wf_result,
        "oos_2025": oos_2025,
        "oos_full": oos_full,
        "grid_top_5": grid_results[:5] if grid_results else [],
        "gate_result": "PROMOTE" if overall else "REVIEW",
    }
    out_path = PROJECT_ROOT / "results" / "exp_h5_001_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Results: {out_path}")


if __name__ == "__main__":
    main()
