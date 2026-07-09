"""
H=5 Ensemble Diagnostic: 4 Strategies Comparison
==================================================

Diagnosis: Boosting models (XGB, LGB, CatBoost) produce CONSTANT predictions
for H=5, dominating the "top-3 by |magnitude|" ensemble and drowning out the
linear models (Ridge, BayesianRidge, ARD) that actually react to features.

4 Strategies:
  1. BASELINE:      Top-3 by |prediction| magnitude (current production)
  2. LINEAR-ONLY:   Mean of ridge + bayesian_ridge + ard
  3. NON-COLLAPSED: Mean of all models with rolling pred_std > 0.001
  4. SMART-FILTER:  Non-collapsed pool → top-3 by rolling DA (12 weeks)

Architecture: Single prediction pass (all 9 models), 4 ensemble selections.
Training: Monthly expanding window (2020-2024 → 2025 OOS).
Trailing stop: activation=0.40%, trail=0.30%, hard=4.00% (re-entry enabled).

@date 2026-02-16
"""

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

LINEAR_IDS = {"ridge", "bayesian_ridge", "ard"}
BOOSTING_IDS = {"xgboost_pure", "lightgbm_pure", "catboost_pure"}
HYBRID_IDS = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}

COLLAPSE_THRESHOLD = 0.001  # Models with pred_std < this are "collapsed"
WARMUP_WEEKS = 12


# =============================================================================
# DATA LOADING (from diagnose_h5_root_cause.py)
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
# MODEL TRAINING (from diagnose_h5_root_cause.py)
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


def train_models_for_horizon(df_train, feature_cols, target_col="target_5d"):
    """Train 9 models, return trained models + scaler."""
    horizon_config = get_horizon_config(HORIZON)
    models_to_use = get_model_list()

    X_train = df_train[feature_cols].values.astype(np.float64)
    y_train = df_train[target_col].values.astype(np.float64)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    trained = {}
    for model_id in models_to_use:
        try:
            params = get_model_params(model_id, horizon_config)
            model = ModelFactory.create(model_id, params=params, horizon=HORIZON)
            if model.requires_scaling:
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            trained[model_id] = model
        except Exception:
            pass

    return trained, scaler


# =============================================================================
# SECTION 4: OOS PREDICTION — PER-MODEL STORAGE
# =============================================================================

def predict_oos_all_models(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Monthly re-training (expanding window). Returns DataFrame with ALL per-model
    predictions for each OOS day.

    NOTE: Monthly retraining ~= production weekly retrain with ~3 week lag.
    Sufficient for ensemble comparison diagnostic.

    Returns columns: date, close, actual_return, {model_id}_pred, realized_vol, leverage
    """
    target_col = "target_5d"
    df_train_all = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_oos = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].copy()
    df_oos = df_oos[df_oos[target_col].notna()]

    months = sorted(df_oos["date"].dt.to_period("M").unique())
    all_rows = []

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        # Expanding window: train on all data up to this month
        if month_idx == 0:
            df_month_train = df_train_all[df_train_all[target_col].notna()].copy()
        else:
            df_month_train = df[(df["date"] >= TRAIN_START) & (df["date"] < month_start)].copy()
            df_month_train = df_month_train[df_month_train[target_col].notna()]

        df_month_oos = df_oos[(df_oos["date"] >= month_start) & (df_oos["date"] <= month_end)]

        if len(df_month_oos) == 0 or len(df_month_train) < 200:
            continue

        trained, scaler = train_models_for_horizon(df_month_train, feature_cols)
        model_ids = sorted(trained.keys())

        for _, row in df_month_oos.iterrows():
            X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)
            X_day_scaled = scaler.transform(X_day)

            row_data = {
                "date": row["date"],
                "close": row["close"],
                "actual_return": row[target_col],
            }

            for model_id in model_ids:
                try:
                    model = trained[model_id]
                    if model.requires_scaling:
                        pred = model.predict(X_day_scaled)[0]
                    else:
                        pred = model.predict(X_day)[0]
                    row_data[f"{model_id}_pred"] = pred
                except Exception:
                    row_data[f"{model_id}_pred"] = np.nan

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
            row_data["realized_vol"] = realized_vol
            row_data["leverage"] = leverage

            all_rows.append(row_data)

    result = pd.DataFrame(all_rows)
    logger.info(f"  Predictions: {len(result)} days, models: {model_ids}")
    return result


# =============================================================================
# SECTION 5: WEEKLY SELECTION + ENSEMBLE STRATEGIES
# =============================================================================

def select_weekly_trades(df: pd.DataFrame, h: int = 5) -> List[int]:
    """Return indices of non-overlapping weekly trades (every h trading days)."""
    df = df.sort_values("date").reset_index(drop=True)
    selected = [0]
    last = 0
    for i in range(1, len(df)):
        if i - last >= h:
            selected.append(i)
            last = i
    return selected


def get_model_type(model_id: str) -> str:
    if model_id in LINEAR_IDS:
        return "LINEAR"
    elif model_id in BOOSTING_IDS:
        return "BOOSTING"
    elif model_id in HYBRID_IDS:
        return "HYBRID"
    return "UNKNOWN"


def run_all_strategies(
    predictions_df: pd.DataFrame,
    model_ids: List[str],
) -> Dict[str, pd.DataFrame]:
    """
    Apply 4 ensemble strategies iteratively over weekly trades.

    Strategies 3-4 need rolling history, so we process sequentially.
    First WARMUP_WEEKS weeks fallback to BASELINE for strategies 3-4.

    Returns {strategy_name: trades_df with ensemble_pred, direction, models_selected}
    """
    predictions_df = predictions_df.sort_values("date").reset_index(drop=True)
    weekly_indices = select_weekly_trades(predictions_df)
    weekly_df = predictions_df.iloc[weekly_indices].copy().reset_index(drop=True)
    n_weeks = len(weekly_df)

    pred_cols = [f"{m}_pred" for m in model_ids]
    available_ids = [m for m in model_ids if f"{m}_pred" in weekly_df.columns]

    # Storage for each strategy
    strategies = ["BASELINE", "LINEAR-ONLY", "NON-COLLAPSED", "SMART-FILTER"]
    results = {s: [] for s in strategies}

    # Rolling history for variance/DA filters
    pred_history = []  # List of dicts {model_id: pred_value}
    actual_history = []  # List of actual direction signs

    for week_idx in range(n_weeks):
        row = weekly_df.iloc[week_idx]
        actual_dir = 1 if row["actual_return"] > 0 else -1

        # Get per-model predictions for this week
        preds = {}
        for m in available_ids:
            v = row[f"{m}_pred"]
            if not np.isnan(v):
                preds[m] = v

        if len(preds) < 2:
            for s in strategies:
                results[s].append(None)
            continue

        # --- Strategy 1: BASELINE (top-3 by |magnitude|) ---
        sorted_mag = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
        top3_baseline = sorted_mag[:3]
        ens_baseline = np.mean([preds[m] for m in top3_baseline])

        # --- Strategy 2: LINEAR-ONLY ---
        linear_avail = [m for m in LINEAR_IDS if m in preds]
        if len(linear_avail) > 0:
            ens_linear = np.mean([preds[m] for m in linear_avail])
        else:
            ens_linear = ens_baseline
            linear_avail = top3_baseline

        # --- Strategies 3-4: Need rolling history ---
        in_warmup = week_idx < WARMUP_WEEKS

        if in_warmup:
            # Fallback to baseline during warmup
            ens_noncollapsed = ens_baseline
            sel_noncollapsed = top3_baseline
            ens_smart = ens_baseline
            sel_smart = top3_baseline
        else:
            # Compute rolling std per model (last WARMUP_WEEKS entries)
            recent_preds = pred_history[-WARMUP_WEEKS:]
            model_std = {}
            for m in preds:
                vals = [h.get(m) for h in recent_preds if m in h]
                if len(vals) >= 3:
                    model_std[m] = np.std(vals)
                else:
                    model_std[m] = 0.0

            # Non-collapsed = models with std > threshold
            non_collapsed = [m for m in preds if model_std.get(m, 0) > COLLAPSE_THRESHOLD]

            if len(non_collapsed) == 0:
                # All collapsed — fallback to all
                non_collapsed = list(preds.keys())

            # --- Strategy 3: NON-COLLAPSED = mean of all active models ---
            ens_noncollapsed = np.mean([preds[m] for m in non_collapsed])
            sel_noncollapsed = non_collapsed

            # --- Strategy 4: SMART-FILTER = non-collapsed pool, top-3 by rolling DA ---
            recent_actuals = actual_history[-WARMUP_WEEKS:]
            model_da = {}
            for m in non_collapsed:
                correct = 0
                total = 0
                for h_idx, h_preds in enumerate(recent_preds):
                    if m in h_preds and h_idx < len(recent_actuals):
                        m_dir = 1 if h_preds[m] > 0 else -1
                        if m_dir == recent_actuals[h_idx]:
                            correct += 1
                        total += 1
                model_da[m] = correct / total if total > 0 else 0.0

            sorted_by_da = sorted(non_collapsed, key=lambda m: model_da.get(m, 0), reverse=True)
            top3_smart = sorted_by_da[:3]
            if len(top3_smart) == 0:
                top3_smart = top3_baseline
            ens_smart = np.mean([preds[m] for m in top3_smart])
            sel_smart = top3_smart

        # Store results for each strategy
        base_data = {
            "date": row["date"],
            "close": row["close"],
            "actual_return": row["actual_return"],
            "realized_vol": row["realized_vol"],
            "leverage": row["leverage"],
        }

        for s, ens, sel in [
            ("BASELINE", ens_baseline, top3_baseline),
            ("LINEAR-ONLY", ens_linear, linear_avail),
            ("NON-COLLAPSED", ens_noncollapsed, sel_noncollapsed),
            ("SMART-FILTER", ens_smart, sel_smart),
        ]:
            d = base_data.copy()
            d["ensemble_pred"] = ens
            d["direction"] = 1 if ens > 0 else -1
            d["models_selected"] = ",".join(sorted(sel))
            results[s].append(d)

        # Update rolling history
        pred_history.append(preds)
        actual_history.append(actual_dir)

    # Build DataFrames (skip None entries)
    output = {}
    for s in strategies:
        valid = [r for r in results[s] if r is not None]
        output[s] = pd.DataFrame(valid) if valid else pd.DataFrame()

    return output


# =============================================================================
# SECTION 6: TRAILING STOP (from diagnose_h5_root_cause.py, with re-entry)
# =============================================================================

def simulate_trailing(trades: pd.DataFrame, df_m5: pd.DataFrame, config: TrailingStopConfig) -> pd.DataFrame:
    """Trailing stop with re-entry over H=5 trading days of 5-min bars."""
    m5_dates = sorted(df_m5["date"].unique())
    slip = SLIPPAGE_BPS / 10_000

    trail_results = []
    for _, trade in trades.iterrows():
        signal_date = trade["date"]
        direction = int(trade["direction"])

        future_dates = [d for d in m5_dates if d > signal_date]
        if len(future_dates) == 0:
            trail_results.append({"week_pnl": None, "reason": "no_bars"})
            continue

        holding_dates = future_dates[:HORIZON]
        week_pnl = 0.0
        last_reason = "no_bars"
        day_idx = 0
        need_entry = True

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

                # Re-entry if days remain
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

            day_idx += 1

        trail_results.append({"week_pnl": week_pnl, "reason": last_reason})

    trades = trades.copy()
    tr_df = pd.DataFrame(trail_results)
    trades["week_pnl"] = tr_df["week_pnl"].values
    trades["exit_reason"] = tr_df["reason"].values
    return trades


# =============================================================================
# SECTION 7: STATS COMPUTATION
# =============================================================================

def compute_strategy_stats(trades: pd.DataFrame) -> Dict:
    """Compute stats including LONG/SHORT breakdown, p-value, bootstrap CI."""
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

    # Bootstrap CI (10000 samples)
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
# SECTION 8: COLLAPSE DIAGNOSTIC
# =============================================================================

def diagnose_collapse(predictions_df: pd.DataFrame, model_ids: List[str]) -> pd.DataFrame:
    """Per-model prediction statistics across full OOS period."""
    rows = []
    for m in model_ids:
        col = f"{m}_pred"
        if col not in predictions_df.columns:
            continue
        vals = predictions_df[col].dropna().values
        if len(vals) == 0:
            continue
        std_val = np.std(vals)
        rows.append({
            "model": m,
            "type": get_model_type(m),
            "mean": round(np.mean(vals), 6),
            "std": round(std_val, 6),
            "min": round(np.min(vals), 6),
            "max": round(np.max(vals), 6),
            "n_preds": len(vals),
            "status": "COLLAPSED" if std_val < COLLAPSE_THRESHOLD else "ACTIVE",
        })
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 9: OUTPUT
# =============================================================================

def print_comparison_table(all_stats: Dict[str, Dict]):
    print("\n" + "=" * 100)
    print("  ENSEMBLE COMPARISON: H=5 OOS 2025 (Train 2020-2024, Trail 0.40/0.30/4.00)")
    print("=" * 100)

    header = (f"  {'Strategy':<18} {'Trades':>6} {'DA%':>6} {'L/S':>7} "
              f"{'LONG WR':>8} {'SHORT WR':>9} {'Return%':>9} {'Sharpe':>7} "
              f"{'WR%':>6} {'PF':>6} {'p-val':>7}")
    print(header)
    print("  " + "-" * 96)

    for name, s in all_stats.items():
        if s["n_trades"] == 0:
            continue
        ls = f"{s['n_long']}/{s['n_short']}"
        print(f"  {name:<18} {s['n_trades']:>6} {s['da_pct']:>5.1f}% {ls:>7} "
              f"{s['long_wr_pct']:>7.1f}% {s['short_wr_pct']:>8.1f}% "
              f"{s['total_return_pct']:>+8.2f}% {s['sharpe']:>+6.3f} "
              f"{s['win_rate_pct']:>5.1f}% {s['profit_factor']:>5.3f} {s['p_value']:>7.4f}")

    print()
    for name, s in all_stats.items():
        if s["n_trades"] == 0:
            continue
        ci = f"[{s['bootstrap_ci_lo']:+.3f}%, {s['bootstrap_ci_hi']:+.3f}%]"
        print(f"  {name:<18} Bootstrap 95% CI: {ci}  "
              f"{'Excludes 0' if s['bootstrap_ci_lo'] > 0 else 'INCLUDES 0'}")


def print_collapse_table(collapse_df: pd.DataFrame):
    print("\n" + "=" * 90)
    print("  PER-MODEL PREDICTION STATISTICS (full OOS 2025)")
    print("=" * 90)
    print(f"  {'Model':<22} {'Type':<10} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10} {'Status':<12}")
    print("  " + "-" * 86)
    for _, row in collapse_df.iterrows():
        print(f"  {row['model']:<22} {row['type']:<10} {row['mean']:>+10.6f} {row['std']:>10.6f} "
              f"{row['min']:>+10.6f} {row['max']:>+10.6f} {row['status']:<12}")

    n_collapsed = (collapse_df["status"] == "COLLAPSED").sum()
    n_active = (collapse_df["status"] == "ACTIVE").sum()
    print(f"\n  Summary: {n_active} ACTIVE, {n_collapsed} COLLAPSED out of {len(collapse_df)} models")


def print_weekly_selection(all_trades: Dict[str, pd.DataFrame]):
    """Show which models each strategy selected per week."""
    print("\n" + "=" * 120)
    print("  WEEKLY MODEL SELECTION")
    print("=" * 120)

    # Get all dates
    baseline_df = all_trades.get("BASELINE")
    if baseline_df is None or len(baseline_df) == 0:
        return

    # Abbreviation map
    abbr = {
        "ridge": "R", "bayesian_ridge": "BR", "ard": "A",
        "xgboost_pure": "XGB", "lightgbm_pure": "LGB", "catboost_pure": "CAT",
        "hybrid_xgboost": "hXGB", "hybrid_lightgbm": "hLGB", "hybrid_catboost": "hCAT",
    }

    def shorten(models_str):
        models = models_str.split(",")
        return ",".join([abbr.get(m, m) for m in models])

    print(f"  {'Week':<12} {'Dir':>4} {'BASELINE':<22} {'LINEAR-ONLY':<18} "
          f"{'NON-COLLAPSED':<30} {'SMART-FILTER':<30}")
    print("  " + "-" * 116)

    for i in range(len(baseline_df)):
        date_str = baseline_df.iloc[i]["date"].strftime("%Y-%m-%d")

        cols = []
        for s in ["BASELINE", "LINEAR-ONLY", "NON-COLLAPSED", "SMART-FILTER"]:
            df = all_trades.get(s)
            if df is not None and i < len(df):
                sel = shorten(df.iloc[i].get("models_selected", ""))
                d = "L" if df.iloc[i]["direction"] == 1 else "S"
                cols.append((d, sel))
            else:
                cols.append(("?", "?"))

        # Use baseline direction for the Dir column
        dir_str = cols[0][0]

        print(f"  {date_str:<12} {dir_str:>4} {cols[0][1]:<22} {cols[1][1]:<18} "
              f"{cols[2][1]:<30} {cols[3][1]:<30}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("=" * 100)
    print("  H=5 ENSEMBLE DIAGNOSTIC: 4 STRATEGIES")
    print("  Hypothesis: Boosting models collapsed -> ensemble stuck on constant predictions")
    print("=" * 100)

    # 1. Load data
    df = load_daily_data()
    df_m5 = load_5min_data()
    feature_cols = list(FEATURE_COLUMNS)

    # 2. Single prediction pass (all 9 models, all OOS days)
    logger.info("\n--- Generating per-model predictions (monthly re-training) ---")
    predictions_df = predict_oos_all_models(df, feature_cols)
    model_ids = sorted([
        c.replace("_pred", "") for c in predictions_df.columns if c.endswith("_pred")
    ])
    logger.info(f"  Models: {model_ids}")
    logger.info(f"  OOS days: {len(predictions_df)}")

    # 3. Run 4 ensemble strategies over weekly trades
    logger.info("\n--- Running 4 ensemble strategies ---")
    all_trades = run_all_strategies(predictions_df, model_ids)

    for name, trades_df in all_trades.items():
        if len(trades_df) == 0:
            logger.warning(f"  {name}: No trades!")
            continue
        n_long = (trades_df["direction"] == 1).sum()
        n_short = (trades_df["direction"] == -1).sum()
        logger.info(f"  {name}: {len(trades_df)} trades (L={n_long}, S={n_short})")

    # 4. Apply trailing stop to each strategy
    logger.info("\n--- Simulating trailing stop (re-entry, 0.40/0.30/4.00) ---")
    all_stats = {}
    for name, trades_df in all_trades.items():
        if len(trades_df) == 0:
            all_stats[name] = {"n_trades": 0}
            continue
        trades_with_trail = simulate_trailing(trades_df, df_m5, TRAIL_CONFIG)
        all_stats[name] = compute_strategy_stats(trades_with_trail)
        all_trades[name] = trades_with_trail  # Update with trail results

    # 5. Collapse diagnostic
    collapse_df = diagnose_collapse(predictions_df, model_ids)

    # 6. Print results
    print_comparison_table(all_stats)
    print_collapse_table(collapse_df)
    print_weekly_selection(all_trades)

    # 7. Key insight: L/S ratio comparison
    print("\n" + "=" * 100)
    print("  KEY DIAGNOSTIC: LONG/SHORT RATIO COMPARISON")
    print("=" * 100)
    for name, s in all_stats.items():
        if s["n_trades"] == 0:
            continue
        pct_long = s["n_long"] / s["n_trades"] * 100
        print(f"  {name:<18}: {s['n_long']}L / {s['n_short']}S ({pct_long:.0f}% LONG)")

    baseline_ls = all_stats.get("BASELINE", {}).get("n_long", 0)
    linear_ls = all_stats.get("LINEAR-ONLY", {}).get("n_long", 0)
    if baseline_ls > 0 and linear_ls > 0:
        bl_total = all_stats["BASELINE"]["n_trades"]
        li_total = all_stats["LINEAR-ONLY"]["n_trades"]
        bl_pct = baseline_ls / bl_total * 100
        li_pct = linear_ls / li_total * 100
        print(f"\n  Baseline: {bl_pct:.0f}% LONG, Linear-only: {li_pct:.0f}% LONG")
        if abs(bl_pct - li_pct) > 15:
            print("  >>> SIGNIFICANT DIRECTION DIFFERENCE: Linear models change direction,")
            print("      confirming boosting collapse is driving the baseline's directional bias.")
        else:
            print("  >>> Similar L/S ratios: boosting and linear models agree on direction.")

    # 8. Warmup caveat
    print(f"\n  NOTE: Strategies 3-4 fall back to BASELINE for first {WARMUP_WEEKS} weeks.")
    n_post = all_stats.get("BASELINE", {}).get("n_trades", 0) - WARMUP_WEEKS
    print(f"  Effective comparison window: {n_post} trades (post-warmup).")

    # 9. Save outputs
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # CSV: per-model predictions per week
    weekly_indices = select_weekly_trades(predictions_df.sort_values("date").reset_index(drop=True))
    weekly_preds = predictions_df.sort_values("date").reset_index(drop=True).iloc[weekly_indices].copy()
    csv_path = output_dir / "h5_per_model_predictions.csv"
    weekly_preds.to_csv(csv_path, index=False)
    print(f"\n  Saved per-model predictions: {csv_path}")

    # JSON: full results
    json_path = output_dir / "h5_ensemble_comparison.json"
    json_data = {
        "strategies": {k: v for k, v in all_stats.items()},
        "collapse": collapse_df.to_dict(orient="records"),
        "config": {
            "trail": {"activation": TRAIL_CONFIG.activation_pct, "trail": TRAIL_CONFIG.trail_pct,
                       "hard_stop": TRAIL_CONFIG.hard_stop_pct},
            "vol_target": {"target_vol": VOL_CONFIG.target_vol, "max_lev": VOL_CONFIG.max_leverage},
            "warmup_weeks": WARMUP_WEEKS,
            "collapse_threshold": COLLAPSE_THRESHOLD,
        },
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"  Saved full results: {json_path}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
