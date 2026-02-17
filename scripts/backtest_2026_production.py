"""
Production Replay: 2026 YTD Backtest + Dashboard Images
========================================================

Runs the EXACT production pipeline (forecast + vol-target + trailing stop)
on 2026 data and generates PNG images + JSON for the Production dashboard.

Adapted from backtest_2025_10k.py — same engine, different year.

Output:
    usdcop-trading-dashboard/public/data/production/
    ├── equity_curve_2026.png
    ├── strategy_comparison_2026.png
    ├── monthly_pnl_2026.png
    ├── trade_distribution_2026.png
    ├── daily_returns_2026.png
    ├── trades/
    │   ├── buy_and_hold.json
    │   ├── forecast_1x.json
    │   ├── forecast_vol_target.json
    │   └── forecast_vt_trailing.json
    └── summary.json

Usage:
    python scripts/backtest_2026_production.py

@version 1.0.0
@date 2026-02-15
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import MODEL_IDS, get_horizon_config, MODEL_DEFINITIONS
from src.forecasting.vol_targeting import (
    VolTargetConfig, compute_vol_target_signal, compute_realized_vol,
)
from src.execution.trailing_stop import TrailingStopConfig, TrailingStopTracker, TrailingState

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIG (same as 2025 backtest)
# =============================================================================

INITIAL_CAPITAL = 10_000.0
SLIPPAGE_BPS = 1.0
VOL_CONFIG = VolTargetConfig(
    target_vol=0.15,
    max_leverage=2.0,
    min_leverage=0.5,
    vol_lookback=21,
    vol_floor=0.05,
)
TRAILING_CONFIG = TrailingStopConfig(
    activation_pct=0.002,
    trail_pct=0.003,
    hard_stop_pct=0.015,
)

YEAR = 2026
TRAIN_CUTOFF = pd.Timestamp(f"{YEAR}-01-01")

OUTPUT_DIR = (
    PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "data" / "production"
)
TRADES_DIR = OUTPUT_DIR / "trades"

# Dark theme colors
S = {
    "bg": "#0f172a",
    "text": "#e2e8f0",
    "grid": "#1e293b",
    "accent": "#a855f7",
    "green": "#10b981",
    "red": "#ef4444",
    "blue": "#3b82f6",
    "amber": "#f59e0b",
    "cyan": "#06b6d4",
}


# =============================================================================
# DATA LOADING (verbatim from backtest_2025_10k.py)
# =============================================================================

def load_daily_data() -> pd.DataFrame:
    """Load daily OHLCV + macro, build 19 SSOT features."""
    logger.info("Loading daily OHLCV...")
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path)

    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)
    logger.info(
        f"  OHLCV: {len(df_ohlcv)} rows, "
        f"{df_ohlcv['date'].min().date()} to {df_ohlcv['date'].max().date()}"
    )

    logger.info("Loading macro data (DXY, WTI)...")
    macro_path = (
        PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
        / "MACRO_DAILY_CLEAN.parquet"
    )
    df_macro = pd.read_parquet(macro_path).reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
    }
    df_macro_subset = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_subset.rename(columns=macro_cols, inplace=True)
    df_macro_subset = df_macro_subset.sort_values("date")
    df_macro_subset["dxy_close_lag1"] = df_macro_subset["dxy_close_lag1"].shift(1)
    df_macro_subset["oil_close_lag1"] = df_macro_subset["oil_close_lag1"].shift(1)

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_subset.sort_values("date"),
        on="date",
        direction="backward",
    )

    df = _build_features(df)
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    target_mask = df["target_return_1d"].notna()
    df = df[feature_mask & target_mask].reset_index(drop=True)

    logger.info(f"  After cleanup: {len(df)} rows with complete features")
    return df


def load_5min_data() -> pd.DataFrame:
    """Load 5-min OHLCV for intraday trailing stop simulation."""
    logger.info("Loading 5-min OHLCV for trailing stop...")
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
    df_m5 = df_m5[df_m5["date"] >= pd.Timestamp("2025-12-01")].copy()
    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"  5-min: {len(df_m5)} bars, "
        f"{df_m5['timestamp'].min()} to {df_m5['timestamp'].max()}"
    )
    return df_m5[["timestamp", "date", "open", "high", "low", "close"]].copy()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 19 SSOT features from raw OHLCV + macro."""
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
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()
    return df


# =============================================================================
# REUSE: train_and_predict_daily, simulate_trailing_stop, compute_stats, run_backtest
# Imported inline from backtest_2025_10k.py logic
# =============================================================================

def train_and_predict_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Train 9 models, predict daily with monthly expanding window."""
    from sklearn.preprocessing import StandardScaler

    feature_cols = list(FEATURE_COLUMNS)
    horizon_config = get_horizon_config(1)

    df_year = df[df["date"] >= TRAIN_CUTOFF].copy()
    df_pre = df[df["date"] < TRAIN_CUTOFF].copy()

    logger.info(f"Training set: {len(df_pre)} rows (up to {df_pre['date'].max().date()})")
    logger.info(f"Trading period: {len(df_year)} rows ({df_year['date'].iloc[0].date()} to {df_year['date'].iloc[-1].date()})")

    boosting_models = {"xgboost_pure", "lightgbm_pure"}
    catboost_models = {"catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    models_to_use = list(MODEL_IDS)
    try:
        ModelFactory.create("ard")
    except Exception:
        models_to_use = [m for m in models_to_use if m != "ard"]

    results = []
    months = sorted(df_year["date"].dt.to_period("M").unique())
    logger.info(f"Processing {len(months)} months with monthly re-training...")

    for month in months:
        month_start = month.start_time
        month_end = month.end_time

        df_train = df[df["date"] < month_start]
        df_month = df_year[
            (df_year["date"] >= month_start) & (df_year["date"] <= month_end)
        ]

        if len(df_month) == 0 or len(df_train) < 200:
            continue

        X_train = df_train[feature_cols].values.astype(np.float64)
        y_train = df_train["target_return_1d"].values.astype(np.float64)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        trained_models = {}
        for model_id in models_to_use:
            try:
                if model_id in linear_models:
                    params = None
                elif model_id in catboost_models:
                    params = {
                        "iterations": horizon_config.get("n_estimators", 50),
                        "depth": horizon_config.get("max_depth", 3),
                        "learning_rate": horizon_config.get("learning_rate", 0.05),
                        "l2_leaf_reg": horizon_config.get("reg_alpha", 0.5),
                        "verbose": False,
                        "allow_writing_files": False,
                    }
                elif model_id in hybrid_models:
                    if "catboost" in model_id:
                        params = {
                            "iterations": horizon_config.get("n_estimators", 50),
                            "depth": horizon_config.get("max_depth", 3),
                            "learning_rate": horizon_config.get("learning_rate", 0.05),
                            "verbose": False,
                            "allow_writing_files": False,
                        }
                    else:
                        params = horizon_config
                else:
                    params = horizon_config

                model = ModelFactory.create(model_id, params=params, horizon=1)
                if model.requires_scaling:
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
                trained_models[model_id] = model
            except Exception as e:
                logger.warning(f"  Failed to train {model_id}: {e}")

        for _, row in df_month.iterrows():
            X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)
            X_day_scaled = scaler.transform(X_day)

            preds = {}
            for model_id, model in trained_models.items():
                try:
                    if model.requires_scaling:
                        pred = model.predict(X_day_scaled)[0]
                    else:
                        pred = model.predict(X_day)[0]
                    preds[model_id] = pred
                except Exception:
                    pass

            if len(preds) < 3:
                continue

            sorted_models = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
            top3 = sorted_models[:3]
            ensemble_pred = np.mean([preds[m] for m in top3])
            direction = 1 if ensemble_pred > 0 else -1

            idx = df.index[df["date"] == row["date"]][0]
            if idx >= VOL_CONFIG.vol_lookback:
                recent_returns = df.iloc[idx - VOL_CONFIG.vol_lookback : idx]["return_1d"].values
                realized_vol = compute_realized_vol(recent_returns, VOL_CONFIG.vol_lookback)
            else:
                realized_vol = 0.10

            vt_signal = compute_vol_target_signal(
                forecast_direction=direction,
                forecast_return=ensemble_pred,
                realized_vol_21d=realized_vol,
                config=VOL_CONFIG,
                date=str(row["date"].date()),
            )

            results.append({
                "date": row["date"],
                "close": row["close"],
                "actual_return": row["target_return_1d"],
                "ensemble_pred": ensemble_pred,
                "direction": direction,
                "realized_vol": realized_vol,
                "leverage": vt_signal.clipped_leverage,
                "top3_models": top3,
                "n_models": len(preds),
            })

        logger.info(f"  Month {month}: {len(df_month)} days, {len(trained_models)} models")

    return pd.DataFrame(results)


def simulate_trailing_stop(predictions: pd.DataFrame, df_m5: pd.DataFrame, df_daily: pd.DataFrame) -> pd.DataFrame:
    """Simulate trailing stop on T+1 5-min bars (verbatim from 2025 backtest)."""
    logger.info("Simulating trailing stop on 5-min bars...")
    m5_dates = sorted(df_m5["date"].unique())

    trail_exit_prices, trail_exit_reasons, trail_pnl_pcts = [], [], []
    trail_bar_counts, trail_peak_prices = [], []

    for _, row in predictions.iterrows():
        signal_date = row["date"]
        entry_price = row["close"]
        direction = int(row["direction"])

        next_dates = [d for d in m5_dates if d > signal_date]
        if not next_dates:
            trail_exit_prices.append(None)
            trail_exit_reasons.append("no_bars")
            trail_pnl_pcts.append(None)
            trail_bar_counts.append(0)
            trail_peak_prices.append(entry_price)
            continue

        t1_date = next_dates[0]
        bars_t1 = df_m5[df_m5["date"] == t1_date].sort_values("timestamp")

        if len(bars_t1) == 0:
            trail_exit_prices.append(None)
            trail_exit_reasons.append("no_bars")
            trail_pnl_pcts.append(None)
            trail_bar_counts.append(0)
            trail_peak_prices.append(entry_price)
            continue

        slip = SLIPPAGE_BPS / 10_000
        slipped_entry = entry_price * (1 + slip) if direction == 1 else entry_price * (1 - slip)

        tracker = TrailingStopTracker(entry_price=slipped_entry, direction=direction, config=TRAILING_CONFIG)

        for bar_idx, (_, bar) in enumerate(bars_t1.iterrows()):
            state = tracker.update(bar_high=float(bar["high"]), bar_low=float(bar["low"]), bar_close=float(bar["close"]), bar_idx=bar_idx)
            if state == TrailingState.TRIGGERED:
                break

        if tracker.state not in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            tracker.expire(float(bars_t1.iloc[-1]["close"]))

        exit_price = tracker.exit_price
        if exit_price is not None:
            exit_price = exit_price * (1 - slip) if direction == 1 else exit_price * (1 + slip)

        raw_pnl = direction * (exit_price - slipped_entry) / slipped_entry if exit_price and slipped_entry > 0 else None

        trail_exit_prices.append(exit_price)
        trail_exit_reasons.append(tracker.exit_reason)
        trail_pnl_pcts.append(raw_pnl)
        trail_bar_counts.append(tracker.exit_bar_idx + 1 if tracker.exit_bar_idx is not None else len(bars_t1))
        trail_peak_prices.append(tracker.peak_price)

    predictions = predictions.copy()
    predictions["trail_exit_price"] = trail_exit_prices
    predictions["trail_exit_reason"] = trail_exit_reasons
    predictions["trail_pnl_unlevered"] = trail_pnl_pcts
    predictions["trail_bar_count"] = trail_bar_counts
    predictions["trail_peak_price"] = trail_peak_prices
    return predictions


def compute_stats(rets: np.ndarray) -> Dict:
    """Compute strategy statistics."""
    if len(rets) == 0:
        return {}
    total_ret = np.prod(1 + rets) - 1
    ann_ret = (1 + total_ret) ** (252 / max(len(rets), 1)) - 1
    vol = np.std(rets) * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))
    wins = np.sum(rets > 0)
    win_rate = wins / len(rets) * 100
    pos_sum = np.sum(rets[rets > 0])
    neg_sum = np.abs(np.sum(rets[rets < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")
    return {
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "volatility_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(pf, 3),
        "trading_days": len(rets),
    }


def run_backtest(predictions: pd.DataFrame) -> Dict:
    """Run 4 strategies and return full results."""
    n_days = len(predictions)
    slip = SLIPPAGE_BPS / 10_000
    cap = INITIAL_CAPITAL

    # Buy & Hold
    first_close = predictions.iloc[0]["close"]
    last_close = predictions.iloc[-1]["close"]
    bh_return = (last_close - first_close) / first_close

    # Forecast 1x
    f1_rets = []
    for _, row in predictions.iterrows():
        r = row["direction"] * (np.exp(row["actual_return"]) - 1) - slip
        f1_rets.append(r)
    f1_rets = np.array(f1_rets)

    # Vol-target
    vt_rets = []
    for _, row in predictions.iterrows():
        lev = row["leverage"]
        r = row["direction"] * lev * (np.exp(row["actual_return"]) - 1) - slip * lev
        vt_rets.append(r)
    vt_rets = np.array(vt_rets)

    # VT + Trailing
    ts_rets = []
    ts_exits = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}
    for _, row in predictions.iterrows():
        lev = row["leverage"]
        trail_pnl = row.get("trail_pnl_unlevered")
        reason = row.get("trail_exit_reason", "no_bars")
        if trail_pnl is not None and not np.isnan(trail_pnl):
            r = trail_pnl * lev
        else:
            r = row["direction"] * lev * (np.exp(row["actual_return"]) - 1) - slip * lev
        ts_rets.append(r)
        if reason in ts_exits:
            ts_exits[reason] += 1
    ts_rets = np.array(ts_rets)

    # Equity curves
    bh_eq = cap * np.cumprod(1 + np.full(n_days, bh_return / n_days))
    f1_eq = cap * np.cumprod(1 + f1_rets)
    vt_eq = cap * np.cumprod(1 + vt_rets)
    ts_eq = cap * np.cumprod(1 + ts_rets)

    # Direction accuracy
    actual_rets = predictions["actual_return"].values
    correct = int(np.sum(predictions["direction"].values == np.sign(actual_rets)))
    da = correct / n_days * 100

    # Stats
    f1_stats = compute_stats(f1_rets)
    vt_stats = compute_stats(vt_rets)
    ts_stats = compute_stats(ts_rets)

    # Monthly
    pred_m = predictions.copy()
    pred_m["month"] = pred_m["date"].dt.to_period("M")
    pred_m["f1_ret"] = f1_rets
    pred_m["vt_ret"] = vt_rets
    pred_m["ts_ret"] = ts_rets

    monthly = pred_m.groupby("month").agg(
        days=("date", "count"),
        f1_sum=("f1_ret", "sum"),
        vt_sum=("vt_ret", "sum"),
        ts_sum=("ts_ret", "sum"),
    ).reset_index()

    # Statistical tests
    t_ts, p_ts = stats.ttest_1samp(ts_rets, 0) if len(ts_rets) > 1 else (0, 1)

    # Bootstrap
    rng = np.random.RandomState(42)
    boot = [np.mean(ts_rets[rng.choice(n_days, n_days, replace=True)]) for _ in range(10_000)]
    ci_lo = float(np.percentile(boot, 2.5) * 252 * 100)
    ci_hi = float(np.percentile(boot, 97.5) * 252 * 100)

    return {
        "year": YEAR,
        "initial_capital": cap,
        "n_trading_days": n_days,
        "direction_accuracy_pct": round(da, 1),
        "strategies": {
            "buy_and_hold": {
                "final_equity": round(cap * (1 + bh_return), 2),
                "total_return_pct": round(bh_return * 100, 2),
            },
            "forecast_1x": {"final_equity": round(float(f1_eq[-1]), 2), **f1_stats},
            "forecast_vol_target": {
                "final_equity": round(float(vt_eq[-1]), 2), **vt_stats,
                "avg_leverage": round(float(predictions["leverage"].mean()), 2),
            },
            "forecast_vt_trailing": {
                "final_equity": round(float(ts_eq[-1]), 2), **ts_stats,
                "avg_leverage": round(float(predictions["leverage"].mean()), 2),
                "exit_reasons": ts_exits,
            },
        },
        "statistical_tests": {
            "t_stat": round(float(t_ts), 3),
            "p_value": round(float(p_ts), 4),
            "bootstrap_95ci_ann": [round(ci_lo, 2), round(ci_hi, 2)],
            "significant": float(p_ts) < 0.05,
        },
        "monthly": {
            "months": [str(m) for m in monthly["month"]],
            "days": monthly["days"].tolist(),
            "f1_pct": (monthly["f1_sum"] * 100).round(2).tolist(),
            "vt_pct": (monthly["vt_sum"] * 100).round(2).tolist(),
            "ts_pct": (monthly["ts_sum"] * 100).round(2).tolist(),
        },
        "equity_curves": {
            "dates": [str(d.date()) for d in predictions["date"]],
            "buy_and_hold": (cap * (1 + np.cumsum(np.full(n_days, bh_return / n_days)))).tolist(),
            "forecast_1x": f1_eq.tolist(),
            "forecast_vol_target": vt_eq.tolist(),
            "forecast_vt_trailing": ts_eq.tolist(),
        },
    }


# =============================================================================
# IMAGE GENERATION
# =============================================================================

def _setup_fig(figsize=(14, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=S["bg"])
    ax.set_facecolor(S["bg"])
    ax.tick_params(colors=S["text"], labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(S["grid"])
    ax.grid(True, alpha=0.15, color=S["grid"])
    return fig, ax


def generate_equity_curve_png(results: Dict, predictions: pd.DataFrame):
    """Equity curve: 4 strategies overlaid."""
    fig, ax = _setup_fig(figsize=(14, 6))

    dates = pd.to_datetime(results["equity_curves"]["dates"])
    ax.plot(dates, results["equity_curves"]["buy_and_hold"], color="#64748b", lw=1, alpha=0.7, label="Buy & Hold")
    ax.plot(dates, results["equity_curves"]["forecast_1x"], color=S["blue"], lw=1.2, label="Forecast 1x")
    ax.plot(dates, results["equity_curves"]["forecast_vol_target"], color=S["amber"], lw=1.5, label="Forecast + VT")
    ax.plot(dates, results["equity_curves"]["forecast_vt_trailing"], color=S["green"], lw=2.5, label="Forecast + VT + Trail")

    ax.axhline(INITIAL_CAPITAL, color=S["text"], lw=0.5, ls=":", alpha=0.3)
    ax.set_title(f"Equity Curve — 2026 YTD ($10K)", color=S["text"], fontsize=14, fontweight="bold")
    ax.set_ylabel("Equity (USD)", color=S["text"], fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate(rotation=45)
    ax.legend(fontsize=9, loc="upper left", facecolor=S["bg"], edgecolor=S["grid"], labelcolor=S["text"])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curve_2026.png", dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


def generate_strategy_comparison_png(results: Dict):
    """Bar chart comparing 4 strategies."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=S["bg"])

    strats = ["buy_and_hold", "forecast_1x", "forecast_vol_target", "forecast_vt_trailing"]
    labels = ["B&H", "Forecast\n1x", "Forecast\n+VT", "Forecast\n+VT+Trail"]
    colors = ["#64748b", S["blue"], S["amber"], S["green"]]

    metrics = ["total_return_pct", "sharpe", "profit_factor", "max_dd_pct"]
    metric_labels = ["Return (%)", "Sharpe Ratio", "Profit Factor", "Max DD (%)"]

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        ax.set_facecolor(S["bg"])
        ax.tick_params(colors=S["text"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(S["grid"])

        vals = []
        for s in strats:
            v = results["strategies"][s].get(metric, 0)
            vals.append(v)

        bar_colors = []
        for j, v in enumerate(vals):
            if metric == "max_dd_pct":
                bar_colors.append(S["red"])
            elif v > 0:
                bar_colors.append(colors[j])
            else:
                bar_colors.append(S["red"])

        bars = ax.bar(labels, vals, color=bar_colors, alpha=0.85, edgecolor=S["grid"])
        ax.set_title(mlabel, color=S["text"], fontsize=10, fontweight="bold")
        ax.axhline(0, color=S["text"], lw=0.5, alpha=0.3)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=7, color=S["text"])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "strategy_comparison_2026.png", dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


def generate_monthly_pnl_png(results: Dict):
    """Monthly P&L bars for the best strategy."""
    fig, ax = _setup_fig(figsize=(12, 5))

    months = results["monthly"]["months"]
    ts_pct = results["monthly"]["ts_pct"]
    colors_bar = [S["green"] if v > 0 else S["red"] for v in ts_pct]

    x = range(len(months))
    bars = ax.bar(x, ts_pct, color=colors_bar, alpha=0.85, edgecolor=S["grid"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.axhline(0, color=S["text"], lw=0.5, alpha=0.3)
    ax.set_title("Monthly P&L — Forecast + VT + Trail (%)", color=S["text"], fontsize=13, fontweight="bold")
    ax.set_ylabel("Return (%)", color=S["text"])

    for bar, v in zip(bars, ts_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=9, color=S["text"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "monthly_pnl_2026.png", dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


def generate_trade_distribution_png(predictions: pd.DataFrame):
    """Histogram of per-trade PnL."""
    fig, ax = _setup_fig(figsize=(10, 5))

    pnls = predictions["trail_pnl_unlevered"].dropna() * 100
    if len(pnls) == 0:
        plt.close(fig)
        return

    bins = np.linspace(pnls.min(), pnls.max(), 30)
    ax.hist(pnls[pnls > 0], bins=bins, color=S["green"], alpha=0.7, label="Wins", edgecolor=S["grid"])
    ax.hist(pnls[pnls <= 0], bins=bins, color=S["red"], alpha=0.7, label="Losses", edgecolor=S["grid"])

    ax.axvline(0, color=S["text"], lw=1, ls="--", alpha=0.5)
    ax.axvline(pnls.mean(), color=S["amber"], lw=1.5, ls="--", alpha=0.8, label=f"Mean: {pnls.mean():.3f}%")

    ax.set_title("Trade P&L Distribution (Unlevered %)", color=S["text"], fontsize=13, fontweight="bold")
    ax.set_xlabel("P&L (%)", color=S["text"])
    ax.set_ylabel("Count", color=S["text"])
    ax.legend(fontsize=9, facecolor=S["bg"], edgecolor=S["grid"], labelcolor=S["text"])

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "trade_distribution_2026.png", dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


def generate_daily_returns_png(predictions: pd.DataFrame, ts_rets: np.ndarray):
    """Daily returns scatter + cumulative line."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=S["bg"], height_ratios=[1, 1])

    dates = predictions["date"].values

    for ax in [ax1, ax2]:
        ax.set_facecolor(S["bg"])
        ax.tick_params(colors=S["text"], labelsize=8)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(S["grid"])
        ax.grid(True, alpha=0.15, color=S["grid"])

    # Top: scatter
    colors_scatter = [S["green"] if r > 0 else S["red"] for r in ts_rets]
    ax1.scatter(dates, ts_rets * 100, c=colors_scatter, s=20, alpha=0.7, zorder=3)
    ax1.axhline(0, color=S["text"], lw=0.5, alpha=0.3)
    ax1.set_title("Daily Returns — VT + Trailing Stop (%)", color=S["text"], fontsize=12, fontweight="bold")
    ax1.set_ylabel("Return (%)", color=S["text"])

    # Bottom: cumulative
    cum_ret = np.cumprod(1 + ts_rets) - 1
    ax2.fill_between(dates, cum_ret * 100, 0, where=cum_ret >= 0, color=S["green"], alpha=0.3)
    ax2.fill_between(dates, cum_ret * 100, 0, where=cum_ret < 0, color=S["red"], alpha=0.3)
    ax2.plot(dates, cum_ret * 100, color=S["green"], lw=1.5)
    ax2.axhline(0, color=S["text"], lw=0.5, alpha=0.3)
    ax2.set_title("Cumulative Return (%)", color=S["text"], fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cum. Return (%)", color=S["text"])
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "daily_returns_2026.png", dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


# =============================================================================
# TRADE EXPORT (same as backtest_2025_10k.py)
# =============================================================================

def export_trades(predictions: pd.DataFrame, results: Dict):
    """Export trade JSONs for dashboard."""
    TRADES_DIR.mkdir(parents=True, exist_ok=True)
    cap = INITIAL_CAPITAL
    slip = SLIPPAGE_BPS / 10_000
    dates = predictions["date"].values

    def _ts(d):
        return pd.Timestamp(d).strftime("%Y-%m-%dT12:30:00-05:00")

    def _summary(trades):
        wins = [t for t in trades if t["pnl_usd"] > 0]
        losses = [t for t in trades if t["pnl_usd"] <= 0]
        total_pnl = sum(t["pnl_usd"] for t in trades)
        wr = len(wins) / len(trades) * 100 if trades else 0
        eq = [cap]
        for t in trades:
            eq.append(eq[-1] + t["pnl_usd"])
        peak, max_dd = eq[0], 0
        for e in eq:
            peak = max(peak, e)
            max_dd = max(max_dd, (peak - e) / peak if peak > 0 else 0)
        rets = [t["pnl_usd"] / t["equity_at_entry"] for t in trades if t["equity_at_entry"] > 0]
        sharpe = float(np.mean(rets) / np.std(rets, ddof=1) * np.sqrt(252)) if len(rets) > 1 and np.std(rets) > 0 else 0
        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(wr, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / cap * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
        }

    def _save(path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    # Build trades for each strategy
    for strat_key, strat_name in [
        ("buy_and_hold", "Buy & Hold USDCOP"),
        ("forecast_1x", "Forecast 1x"),
        ("forecast_vol_target", "Forecast + Vol-Target"),
        ("forecast_vt_trailing", "Forecast + VT + Trail"),
    ]:
        trades = []
        equity = cap

        for i, (_, row) in enumerate(predictions.iterrows()):
            direction = int(row["direction"])
            lev = float(row.get("leverage", 1.0))
            entry_close = float(row["close"])

            if strat_key == "buy_and_hold":
                daily_ret = np.exp(row["actual_return"]) - 1
            elif strat_key == "forecast_1x":
                daily_ret = direction * (np.exp(row["actual_return"]) - 1) - slip
            elif strat_key == "forecast_vol_target":
                daily_ret = direction * lev * (np.exp(row["actual_return"]) - 1) - slip * lev
            else:
                trail_pnl = row.get("trail_pnl_unlevered")
                reason = row.get("trail_exit_reason", "session_close")
                if trail_pnl is not None and not np.isnan(trail_pnl):
                    daily_ret = float(trail_pnl) * lev
                else:
                    daily_ret = direction * lev * (np.exp(row["actual_return"]) - 1) - slip * lev
                    reason = "session_close"

            pnl_usd = equity * daily_ret
            eq_before = equity
            equity += pnl_usd

            exit_close = entry_close * np.exp(float(row["actual_return"]))
            if strat_key == "forecast_vt_trailing":
                trail_exit = row.get("trail_exit_price")
                if trail_exit is not None and not np.isnan(trail_exit):
                    exit_close = float(trail_exit)

            trades.append({
                "trade_id": i + 1,
                "timestamp": _ts(row["date"]),
                "side": "LONG" if (direction == 1 or strat_key == "buy_and_hold") else "SHORT",
                "entry_price": round(entry_close, 2),
                "exit_price": round(exit_close, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_pct": round(daily_ret * 100, 3),
                "exit_reason": reason if strat_key == "forecast_vt_trailing" else "session_close",
                "equity_at_entry": round(eq_before, 2),
                "equity_at_exit": round(equity, 2),
                "leverage": round(lev, 2) if strat_key not in ("buy_and_hold", "forecast_1x") else 1.0,
            })

        _save(TRADES_DIR / f"{strat_key}.json", {
            "strategy_name": strat_name,
            "initial_capital": cap,
            "date_range": {"start": str(pd.Timestamp(dates[0]).date()), "end": str(pd.Timestamp(dates[-1]).date())},
            "trades": trades,
            "summary": _summary(trades),
        })

    logger.info(f"  Trade JSONs saved to {TRADES_DIR}")


# =============================================================================
# GATE EVALUATION & APPROVAL STATE
# =============================================================================

# Paper-trading gates: lenient thresholds (2025 backtest proved p=0.0178)
GATES = {
    "min_return_pct": (-15.0, "Retorno Minimo"),
    "min_sharpe_ratio": (-5.0, "Sharpe Minimo"),
    "max_drawdown_pct": (20.0, "Max Drawdown"),
    "min_trades": (30, "Trades Minimos"),
    "min_win_rate": (35.0, "Win Rate Minimo"),
}


def evaluate_gates(results: Dict) -> List[Dict]:
    """Evaluate 5 backtest gates against the best strategy."""
    best = results["strategies"]["forecast_vt_trailing"]
    gate_results = []

    for gate_key, (threshold, label) in GATES.items():
        if gate_key == "min_return_pct":
            value = best.get("total_return_pct", -100)
            passed = value >= threshold
        elif gate_key == "min_sharpe_ratio":
            value = best.get("sharpe", -100)
            passed = value >= threshold
        elif gate_key == "max_drawdown_pct":
            value = abs(best.get("max_dd_pct", -100))
            passed = value <= threshold
        elif gate_key == "min_trades":
            value = best.get("trading_days", 0)
            passed = value >= threshold
        elif gate_key == "min_win_rate":
            value = best.get("win_rate_pct", 0)
            passed = value >= threshold
        else:
            continue

        gate_results.append({
            "gate": gate_key,
            "label": label,
            "passed": bool(passed),
            "value": round(float(value), 2) if isinstance(value, float) else int(value),
            "threshold": threshold,
        })

    return gate_results


def update_approval_state(gate_results: List[Dict]) -> None:
    """Create or update approval_state.json preserving existing approval status."""
    approval_path = OUTPUT_DIR / "approval_state.json"

    passed = sum(1 for g in gate_results if g["passed"])
    total = len(gate_results)
    confidence = passed / total if total > 0 else 0

    if confidence >= 0.8:
        recommendation = "PROMOTE"
    elif confidence >= 0.6:
        recommendation = "REVIEW"
    else:
        recommendation = "REJECT"

    now = pd.Timestamp.now().isoformat()

    # Check existing state — preserve user's vote
    existing_fields = {}
    existing_created_at = now
    if approval_path.exists():
        try:
            with open(approval_path) as f:
                existing = json.load(f)
            existing_created_at = existing.get("created_at", now)
            existing_status = existing.get("status", "PENDING_APPROVAL")
            if existing_status in ("APPROVED", "LIVE"):
                existing_fields = {
                    "status": existing_status,
                    "approved_by": existing.get("approved_by"),
                    "approved_at": existing.get("approved_at"),
                    "reviewer_notes": existing.get("reviewer_notes"),
                }
            elif existing_status == "REJECTED":
                existing_fields = {
                    "status": existing_status,
                    "rejected_by": existing.get("rejected_by"),
                    "rejected_at": existing.get("rejected_at"),
                    "rejection_reason": existing.get("rejection_reason"),
                }
        except (json.JSONDecodeError, OSError):
            pass

    state = {
        "status": existing_fields.get("status", "PENDING_APPROVAL"),
        "strategy": "forecast_vt_trailing",
        "backtest_recommendation": recommendation,
        "backtest_confidence": round(confidence, 2),
        "gates": gate_results,
        **{k: v for k, v in existing_fields.items() if k != "status"},
        "created_at": existing_created_at,
        "last_updated": now,
    }

    with open(approval_path, "w") as f:
        json.dump(state, f, indent=2, default=str)

    logger.info(f"  Approval state: {state['status']} (recommendation={recommendation}, confidence={confidence:.0%})")
    for g in gate_results:
        status = "PASS" if g["passed"] else "FAIL"
        logger.info(f"    [{status}] {g['label']}: {g['value']} (threshold: {g['threshold']})")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TRADES_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df_daily = load_daily_data()
    df_m5 = load_5min_data()

    # Check we have 2026 data
    max_date = df_daily["date"].max()
    if max_date < pd.Timestamp("2026-01-05"):
        logger.error(f"Daily data only goes to {max_date.date()}. Run backfill_2026_data.py first!")
        sys.exit(1)

    # Train + predict
    predictions = train_and_predict_daily(df_daily)
    if len(predictions) == 0:
        logger.error("No predictions generated for 2026!")
        sys.exit(1)
    logger.info(f"Generated {len(predictions)} daily predictions for 2026")

    # Trailing stop
    predictions = simulate_trailing_stop(predictions, df_m5, df_daily)

    # Backtest
    results = run_backtest(predictions)

    # Compute ts_rets for images
    ts_rets = []
    slip = SLIPPAGE_BPS / 10_000
    for _, row in predictions.iterrows():
        lev = row["leverage"]
        trail_pnl = row.get("trail_pnl_unlevered")
        if trail_pnl is not None and not np.isnan(trail_pnl):
            ts_rets.append(trail_pnl * lev)
        else:
            ts_rets.append(row["direction"] * lev * (np.exp(row["actual_return"]) - 1) - slip * lev)
    ts_rets = np.array(ts_rets)

    # Print results
    best = results["strategies"]["forecast_vt_trailing"]
    bh = results["strategies"]["buy_and_hold"]
    print(f"\n{'='*60}")
    print(f"  2026 YTD PRODUCTION REPLAY")
    print(f"{'='*60}")
    print(f"  Trading days: {results['n_trading_days']}")
    print(f"  DA: {results['direction_accuracy_pct']}%")
    print(f"  B&H:            ${bh['final_equity']:>10,.2f}  ({bh['total_return_pct']:+.2f}%)")
    print(f"  VT+Trail:       ${best['final_equity']:>10,.2f}  ({best['total_return_pct']:+.2f}%)")
    print(f"  Sharpe:         {best['sharpe']}")
    print(f"  PF:             {best['profit_factor']}")
    print(f"  Max DD:         {best['max_dd_pct']}%")
    print(f"  p-value:        {results['statistical_tests']['p_value']}")
    print(f"  95% CI (ann.):  {results['statistical_tests']['bootstrap_95ci_ann']}")
    print(f"{'='*60}\n")

    # Generate images
    logger.info("Generating PNG images...")
    generate_equity_curve_png(results, predictions)
    generate_strategy_comparison_png(results)
    generate_monthly_pnl_png(results)
    generate_trade_distribution_png(predictions)
    generate_daily_returns_png(predictions, ts_rets)

    # Export trades
    logger.info("Exporting trade JSONs...")
    export_trades(predictions, results)

    # Evaluate gates & update approval state
    logger.info("Evaluating gates...")
    gate_results = evaluate_gates(results)
    update_approval_state(gate_results)

    # Add gates to results for summary.json
    passed_count = sum(1 for g in gate_results if g["passed"])
    total_gates = len(gate_results)
    results["gates"] = gate_results
    results["backtest_recommendation"] = "PROMOTE" if passed_count / total_gates >= 0.8 else ("REVIEW" if passed_count / total_gates >= 0.6 else "REJECT")
    results["backtest_confidence"] = round(passed_count / total_gates, 2)

    # Save summary
    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "year": YEAR,
        **results,
    }
    # Remove equity_curves from summary (too large for JSON cards)
    summary_light = {k: v for k, v in summary.items() if k != "equity_curves"}
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary_light, f, indent=2, default=str)
    logger.info(f"Summary saved to {OUTPUT_DIR / 'summary.json'}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s")
