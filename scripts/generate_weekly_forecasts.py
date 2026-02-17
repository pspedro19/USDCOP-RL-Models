"""
Generate Weekly Forecast Dashboard Data (CSV + PNG)
====================================================

Generates bi_dashboard_unified.csv + backtest/forward forecast PNG images
for the ForecastingDashboard component. Supports retroactive multi-week
generation with actual price verification.

Usage:
    # Generate last 7 weeks retroactively (default)
    python scripts/generate_weekly_forecasts.py

    # Custom number of weeks
    python scripts/generate_weekly_forecasts.py --num-weeks 10

    # Single specific week
    python scripts/generate_weekly_forecasts.py --week 2025-W50

Data Sources (parquet-based, same as L5a DAG):
    - seeds/latest/usdcop_daily_ohlcv.parquet (daily OHLCV)
    - data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet (DXY + WTI)

Output:
    usdcop-trading-dashboard/public/forecasting/
    ├── bi_dashboard_unified.csv
    ├── backtest_{model}_h{horizon}.png            (latest week only)
    ├── forward_{model}_{week}.png                 (per week)
    ├── forward_consensus_{week}.png               (per week)
    └── forward_ensemble_{key}_{week}.png          (per week)

Version: 3.0.0
Date: 2026-02-15
"""

import argparse
import csv
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.contracts import (
    HORIZONS,
    MODEL_IDS,
    MODEL_DEFINITIONS,
    HORIZON_CATEGORIES,
    get_horizon_config,
)
from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "forecasting"

# Model classifications (same as L5a DAG)
LINEAR_MODELS = {"ridge", "bayesian_ridge", "ard"}
CATBOOST_MODELS = {"catboost_pure"}
HYBRID_MODELS = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
BOOSTING_MODELS = {"xgboost_pure", "lightgbm_pure"}

MODEL_NAMES = {
    "ridge": "Ridge",
    "bayesian_ridge": "Bayesian Ridge",
    "ard": "ARD",
    "xgboost_pure": "XGBoost",
    "lightgbm_pure": "LightGBM",
    "catboost_pure": "CatBoost",
    "hybrid_xgboost": "Hybrid XGBoost",
    "hybrid_lightgbm": "Hybrid LightGBM",
    "hybrid_catboost": "Hybrid CatBoost",
}

MODEL_TYPES = {
    "ridge": "linear",
    "bayesian_ridge": "linear",
    "ard": "linear",
    "xgboost_pure": "boosting",
    "lightgbm_pure": "boosting",
    "catboost_pure": "boosting",
    "hybrid_xgboost": "hybrid",
    "hybrid_lightgbm": "hybrid",
    "hybrid_catboost": "hybrid",
}

# Plot style (dark, matching dashboard)
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
    "pink": "#ec4899",
}

# Model colors for multi-model plots
MODEL_COLORS = [
    "#a855f7", "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
    "#ec4899", "#06b6d4", "#f97316", "#84cc16",
]

CSV_COLUMNS = [
    "record_id", "view_type", "model_id", "model_name", "model_type",
    "horizon_days", "horizon_label", "horizon_category",
    "inference_week", "inference_year", "inference_date",
    "direction_accuracy", "rmse", "mae", "r2",
    "sharpe", "profit_factor", "max_drawdown", "total_return",
    "wf_direction_accuracy", "model_avg_direction_accuracy", "model_avg_rmse",
    "is_best_overall_model", "is_best_for_this_horizon", "best_da_for_this_horizon",
    "image_path", "image_backtest", "generated_at", "image_forecast",
]


# =============================================================================
# DATA LOADING (mirrors L5a DAG pattern)
# =============================================================================

def load_full_dataset() -> pd.DataFrame:
    """Load full daily OHLCV + macro dataset, build 21 SSOT features.

    Uses shared ForecastingDatasetLoader (DB-first with parquet fallback).
    All 4 macro columns: DXY, WTI, VIX, EMBI.
    """
    cfg = ForecastingSSOTConfig.load()
    loader = ForecastingDatasetLoader(cfg, project_root=PROJECT_ROOT)
    df, _ = loader.load_dataset()
    return df


# =============================================================================
# MODEL PARAMS ROUTING (same as L5a DAG)
# =============================================================================

def _get_model_params(model_id: str, horizon_config: dict) -> Optional[dict]:
    """Get model-specific params. Linear->None, CatBoost->translated."""
    if model_id in LINEAR_MODELS:
        return None

    if model_id in CATBOOST_MODELS:
        return {
            "iterations": horizon_config.get("n_estimators", 50),
            "depth": horizon_config.get("max_depth", 3),
            "learning_rate": horizon_config.get("learning_rate", 0.05),
            "l2_leaf_reg": horizon_config.get("reg_alpha", 0.5),
            "verbose": False,
            "allow_writing_files": False,
        }

    if model_id in HYBRID_MODELS:
        if "catboost" in model_id:
            return {
                "iterations": horizon_config.get("n_estimators", 50),
                "depth": horizon_config.get("max_depth", 3),
                "learning_rate": horizon_config.get("learning_rate", 0.05),
                "verbose": False,
                "allow_writing_files": False,
            }
        return horizon_config

    return horizon_config


# =============================================================================
# WALK-FORWARD VALIDATION
# =============================================================================

def walk_forward_validate(
    X: np.ndarray,
    y: np.ndarray,
    model_id: str,
    params: Optional[dict],
    horizon: int,
    n_folds: int = 3,
) -> Dict[str, Any]:
    """Walk-forward validation returning full metrics + predicted/actual arrays."""
    n = len(X)
    initial_train = int(n * 0.6)
    step = (n - initial_train) // n_folds

    all_preds = []
    all_actuals = []

    for fold in range(n_folds):
        train_end = initial_train + fold * step
        test_start = train_end
        test_end = min(test_start + step, n)

        if test_end <= test_start or train_end < 50:
            continue

        model = ModelFactory.create(model_id, params=params, horizon=horizon)
        model.fit(X[:train_end], y[:train_end])
        preds = model.predict(X[test_start:test_end])
        actuals = y[test_start:test_end]

        all_preds.extend(preds.tolist())
        all_actuals.extend(actuals.tolist())

    if not all_preds:
        return {
            "da": 0.5, "rmse": 0.0, "mae": 0.0, "r2": 0.0,
            "sharpe": 0.0, "pf": 1.0, "max_drawdown": 0.0, "total_return": 0.0,
            "predicted_returns": [], "actual_returns": [],
        }

    preds_arr = np.array(all_preds)
    actuals_arr = np.array(all_actuals)

    correct = np.sum(np.sign(preds_arr) == np.sign(actuals_arr))
    da = correct / len(actuals_arr)

    rmse = float(np.sqrt(np.mean((preds_arr - actuals_arr) ** 2)))
    mae = float(np.mean(np.abs(preds_arr - actuals_arr)))
    ss_res = np.sum((actuals_arr - preds_arr) ** 2)
    ss_tot = np.sum((actuals_arr - np.mean(actuals_arr)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    signals = np.sign(preds_arr)
    trade_returns = signals * actuals_arr

    periods_per_year = 252 / max(horizon, 1)
    mean_ret = np.mean(trade_returns)
    std_ret = np.std(trade_returns)
    sharpe = float(mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    gains = trade_returns[trade_returns > 0].sum()
    losses = abs(trade_returns[trade_returns < 0].sum())
    pf = float(gains / losses) if losses > 0 else (2.0 if gains > 0 else 1.0)

    equity = np.cumprod(1 + trade_returns)
    total_return = float(equity[-1] - 1) if len(equity) > 0 else 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_drawdown = float(abs(drawdown.min()))

    return {
        "da": da, "rmse": rmse, "mae": mae, "r2": r2,
        "sharpe": sharpe, "pf": pf, "max_drawdown": max_drawdown,
        "total_return": total_return,
        "predicted_returns": all_preds, "actual_returns": all_actuals,
    }


# =============================================================================
# WEEK DISCOVERY
# =============================================================================

def find_last_n_weeks(df: pd.DataFrame, n: int = 7) -> List[Tuple[str, pd.Timestamp]]:
    """Find the last n weeks in the dataset, returning (week_label, last_trading_day)."""
    dates = df["date"].sort_values().unique()

    # Group by ISO week, keep last trading day per week
    weeks = {}
    for d in dates:
        ts = pd.Timestamp(d)
        year, week_num, _ = ts.isocalendar()
        week_key = f"{year}-W{week_num:02d}"
        weeks[week_key] = ts

    sorted_weeks = sorted(weeks.items(), key=lambda x: x[1])
    return sorted_weeks[-n:]


def get_target_dates_and_actuals(
    df_full: pd.DataFrame,
    inference_date: pd.Timestamp,
    base_price: float,
) -> Dict[int, Dict]:
    """For each horizon, find the target trading date and actual price (if available)."""
    df_after = df_full[df_full["date"] > inference_date].sort_values("date")

    result = {}
    for h in HORIZONS:
        if h <= len(df_after):
            target_date = pd.Timestamp(df_after["date"].iloc[h - 1])
            actual_price = float(df_after["close"].iloc[h - 1])
            actual_return = np.log(actual_price / base_price)
            result[h] = {"date": target_date, "actual_price": actual_price, "actual_return": actual_return}
        else:
            # Estimate: H trading days ≈ H * 7/5 calendar days
            cal_days = int(h * 7 / 5) + 1
            est_date = inference_date + pd.Timedelta(days=cal_days)
            while est_date.dayofweek >= 5:
                est_date += pd.Timedelta(days=1)
            result[h] = {"date": est_date, "actual_price": None, "actual_return": None}

    return result


# =============================================================================
# TRAIN MODELS + GENERATE DATA FOR ONE WEEK
# =============================================================================

def generate_week_data(
    df_full: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    week_label: str,
    is_latest_week: bool,
) -> Tuple[List[Dict], Dict[str, Dict[int, float]], Dict, float, pd.Timestamp]:
    """
    Train 9 models x 7 horizons for one week, run walk-forward.

    Returns:
        csv_rows, raw_predictions, wf_results, base_price, inference_date
    """
    from sklearn.preprocessing import StandardScaler

    inference_year = cutoff_date.year
    df = df_full[df_full["date"] <= cutoff_date].copy()
    feature_cols = list(FEATURE_COLUMNS)
    feature_mask = df[feature_cols].notna().all(axis=1)
    df_clean = df[feature_mask].reset_index(drop=True)

    if len(df_clean) < 100:
        raise ValueError(f"Not enough data for {week_label}: {len(df_clean)} rows")

    X = df_clean[feature_cols].values.astype(np.float64)
    base_price = float(df_clean["close"].iloc[-1])
    inference_date = df_clean["date"].iloc[-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_latest_scaled = X_scaled[-1:].copy()
    X_latest_raw = X[-1:].copy()

    logger.info(f"  {week_label}: {len(df_clean)} rows, base={base_price:.2f}, inf_date={inference_date.date()}")

    now_str = datetime.now().isoformat()
    inf_date_str = str(inference_date.date())
    week_suffix = week_label.replace("-", "_")

    # Collect results
    all_metrics: Dict[str, Dict[int, Dict]] = {}
    raw_predictions: Dict[str, Dict[int, float]] = {}
    wf_results: Dict[str, Dict[int, Dict]] = {}

    for model_id in MODEL_IDS:
        all_metrics[model_id] = {}
        raw_predictions[model_id] = {}
        wf_results[model_id] = {}

        for horizon in HORIZONS:
            try:
                future_price = df_clean["close"].shift(-horizon)
                y = np.log(future_price / df_clean["close"]).values
                valid_mask = ~np.isnan(y)
                y_valid = y[valid_mask]

                if len(y_valid) < 50:
                    continue

                h_config = get_horizon_config(horizon)
                params = _get_model_params(model_id, h_config)

                model_tmp = ModelFactory.create(model_id, params=params, horizon=horizon)
                use_scaled = model_tmp.requires_scaling
                X_wf = X_scaled[valid_mask] if use_scaled else X[valid_mask]

                wf = walk_forward_validate(X_wf, y_valid, model_id, params, horizon)
                wf_results[model_id][horizon] = wf
                all_metrics[model_id][horizon] = {
                    "da": wf["da"], "rmse": wf["rmse"], "mae": wf["mae"], "r2": wf["r2"],
                    "sharpe": wf["sharpe"], "pf": wf["pf"],
                    "max_drawdown": wf["max_drawdown"], "total_return": wf["total_return"],
                }

                model = ModelFactory.create(model_id, params=params, horizon=horizon)
                if use_scaled:
                    model.fit(X_scaled[valid_mask], y_valid)
                    pred_return = float(model.predict(X_latest_scaled)[0])
                else:
                    model.fit(X[valid_mask], y_valid)
                    pred_return = float(model.predict(X_latest_raw)[0])

                raw_predictions[model_id][horizon] = pred_return

            except Exception as e:
                logger.warning(f"    {model_id} H={horizon}: {e}")

    # Model averages
    model_avg_da = {}
    model_avg_rmse = {}
    for model_id in MODEL_IDS:
        das = [m["da"] for m in all_metrics[model_id].values()]
        rmses = [m["rmse"] for m in all_metrics[model_id].values()]
        model_avg_da[model_id] = float(np.mean(das)) if das else 0.0
        model_avg_rmse[model_id] = float(np.mean(rmses)) if rmses else 0.0

    best_overall = max(model_avg_da, key=model_avg_da.get) if model_avg_da else None

    best_per_horizon: Dict[int, Tuple[str, float]] = {}
    for horizon in HORIZONS:
        best_da, best_model = -1.0, None
        for model_id in MODEL_IDS:
            if horizon in all_metrics[model_id]:
                da = all_metrics[model_id][horizon]["da"]
                if da > best_da:
                    best_da, best_model = da, model_id
        if best_model is not None:
            best_per_horizon[horizon] = (best_model, best_da)

    # Build CSV rows
    csv_rows = []
    for model_id in MODEL_IDS:
        for horizon in HORIZONS:
            if horizon not in all_metrics[model_id]:
                continue

            m = all_metrics[model_id][horizon]
            h_cat = HORIZON_CATEGORIES.get(horizon, "short")
            is_best = model_id == best_overall
            is_best_h = best_per_horizon.get(horizon, (None, 0))[0] == model_id
            best_da_h = best_per_horizon.get(horizon, (None, 0))[1]

            # Forward forecast row (all weeks)
            csv_rows.append({
                "record_id": f"FF_{model_id}_h{horizon}_{week_suffix}",
                "view_type": "forward_forecast",
                "model_id": model_id,
                "model_name": MODEL_NAMES.get(model_id, model_id),
                "model_type": MODEL_TYPES.get(model_id, "unknown"),
                "horizon_days": horizon,
                "horizon_label": f"H={horizon}",
                "horizon_category": h_cat,
                "inference_week": week_label,
                "inference_year": inference_year,
                "inference_date": inf_date_str,
                "direction_accuracy": m["da"],
                "rmse": 0.0,
                "mae": 0.0,
                "r2": 0.0,
                "sharpe": m["sharpe"],
                "profit_factor": m["pf"],
                "max_drawdown": m["max_drawdown"],
                "total_return": m["total_return"],
                "wf_direction_accuracy": m["da"],
                "model_avg_direction_accuracy": model_avg_da[model_id],
                "model_avg_rmse": model_avg_rmse[model_id],
                "is_best_overall_model": is_best,
                "is_best_for_this_horizon": is_best_h,
                "best_da_for_this_horizon": best_da_h,
                "image_path": f"forward_{model_id}_{week_suffix}.png",
                "image_backtest": f"backtest_{model_id}_h{horizon}.png",
                "generated_at": now_str,
                "image_forecast": f"forward_{model_id}_{week_suffix}.png",
            })

            # Backtest row (latest week only)
            if is_latest_week:
                csv_rows.append({
                    "record_id": f"BT_{model_id}_h{horizon}",
                    "view_type": "backtest",
                    "model_id": model_id,
                    "model_name": MODEL_NAMES.get(model_id, model_id),
                    "model_type": MODEL_TYPES.get(model_id, "unknown"),
                    "horizon_days": horizon,
                    "horizon_label": f"H={horizon}",
                    "horizon_category": h_cat,
                    "inference_week": week_label,
                    "inference_year": inference_year,
                    "inference_date": inf_date_str,
                    "direction_accuracy": m["da"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "r2": m["r2"],
                    "sharpe": m["sharpe"],
                    "profit_factor": m["pf"],
                    "max_drawdown": m["max_drawdown"],
                    "total_return": m["total_return"],
                    "wf_direction_accuracy": m["da"],
                    "model_avg_direction_accuracy": model_avg_da[model_id],
                    "model_avg_rmse": model_avg_rmse[model_id],
                    "is_best_overall_model": is_best,
                    "is_best_for_this_horizon": is_best_h,
                    "best_da_for_this_horizon": best_da_h,
                    "image_path": f"backtest_{model_id}_h{horizon}.png",
                    "image_backtest": f"backtest_{model_id}_h{horizon}.png",
                    "generated_at": now_str,
                    "image_forecast": f"forward_{model_id}_{week_suffix}.png",
                })

    return csv_rows, raw_predictions, wf_results, base_price, inference_date


# =============================================================================
# IMAGE GENERATION — Backtest (same as before, latest week only)
# =============================================================================

def _setup_dark_fig(figsize=(12, 6)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=S["bg"])
    ax.set_facecolor(S["bg"])
    ax.tick_params(colors=S["text"], labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(S["grid"])
    ax.grid(True, alpha=0.15, color=S["grid"])
    return fig, ax


def _style_ax(ax):
    """Apply dark style to an axis."""
    ax.set_facecolor(S["bg"])
    ax.tick_params(colors=S["text"], labelsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(S["grid"])
    ax.grid(True, alpha=0.15, color=S["grid"])


def generate_backtest_image(wf_result: Dict, model_id: str, horizon: int, output_dir: Path):
    """Generate backtest PNG: scatter + equity curve (latest week only)."""
    preds = np.array(wf_result["predicted_returns"])
    actuals = np.array(wf_result["actual_returns"])
    if len(preds) < 5:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=S["bg"])

    # Left: Predicted vs Actual scatter
    ax = axes[0]
    _style_ax(ax)
    correct = np.sign(preds) == np.sign(actuals)
    ax.scatter(actuals[correct], preds[correct], c=S["green"], s=18, alpha=0.7, label="Correct", zorder=3)
    ax.scatter(actuals[~correct], preds[~correct], c=S["red"], s=18, alpha=0.7, label="Wrong", zorder=3)
    lims = [min(actuals.min(), preds.min()), max(actuals.max(), preds.max())]
    ax.plot(lims, lims, "--", color=S["accent"], alpha=0.5, lw=1)
    ax.axhline(0, color="#64748b", lw=0.5, alpha=0.5)
    ax.axvline(0, color="#64748b", lw=0.5, alpha=0.5)
    ax.set_title(f"Predicted vs Actual (DA={wf_result['da']*100:.1f}%)",
                 color=S["text"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Actual Return", color=S["text"], fontsize=9)
    ax.set_ylabel("Predicted Return", color=S["text"], fontsize=9)
    ax.legend(fontsize=8, loc="upper left", facecolor=S["bg"],
              edgecolor=S["grid"], labelcolor=S["text"])

    # Right: Equity curve
    ax2 = axes[1]
    _style_ax(ax2)
    signals = np.sign(preds)
    trade_rets = signals * actuals
    equity = np.cumprod(1 + trade_rets) * 100
    bh_equity = np.cumprod(1 + actuals) * 100
    ax2.plot(equity, color=S["accent"], lw=1.5, label="Strategy")
    ax2.plot(bh_equity, color=S["blue"], lw=1, alpha=0.5, label="Buy & Hold")
    ax2.axhline(100, color="#64748b", lw=0.5, ls="--", alpha=0.5)
    ax2.set_title(
        f"Equity (Return={wf_result['total_return']*100:.1f}%, Sharpe={wf_result['sharpe']:.2f})",
        color=S["text"], fontsize=11, fontweight="bold")
    ax2.set_xlabel("Trade #", color=S["text"], fontsize=9)
    ax2.set_ylabel("Equity (base=100)", color=S["text"], fontsize=9)
    ax2.legend(fontsize=8, loc="upper left", facecolor=S["bg"],
               edgecolor=S["grid"], labelcolor=S["text"])

    model_name = MODEL_NAMES.get(model_id, model_id)
    fig.suptitle(f"{model_name} — H={horizon} Day Backtest",
                 color=S["text"], fontsize=13, fontweight="bold", y=1.02)

    plt.tight_layout()
    fig.savefig(output_dir / f"backtest_{model_id}_h{horizon}.png",
                dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


# =============================================================================
# IMAGE GENERATION — Forward Forecast (date-based, with historical + actuals)
# =============================================================================

def generate_forward_forecast_image(
    predictions: Dict[int, float],
    base_price: float,
    model_id: str,
    inference_date: pd.Timestamp,
    output_dir: Path,
    week_suffix: str,
    df_full: pd.DataFrame,
    target_info: Dict[int, Dict],
    title_override: str = "",
    filename_override: str = "",
):
    """
    Forward forecast PNG with ~1 month historical context, date X-axis,
    and actual prices where available.
    """
    horizons_sorted = sorted([h for h in predictions if h in target_info])
    if not horizons_sorted:
        return

    fig, ax = _setup_dark_fig(figsize=(14, 6))

    # --- Historical prices (~25 trading days before inference_date) ---
    df_hist = df_full[df_full["date"] <= inference_date].tail(25).copy()
    ax.plot(df_hist["date"].values, df_hist["close"].values,
            color=S["text"], lw=1.5, alpha=0.7, label="Precio Historico", zorder=2)

    # --- Actual prices after inference (if available, draw as continuation) ---
    last_actual_h = 0
    for h in horizons_sorted:
        if target_info[h]["actual_price"] is not None:
            last_actual_h = h

    if last_actual_h > 0:
        # Get actual close prices between inference_date and the last actual
        last_target_date = target_info[last_actual_h]["date"]
        df_actual = df_full[
            (df_full["date"] > inference_date) & (df_full["date"] <= last_target_date)
        ].sort_values("date")
        if len(df_actual) > 0:
            # Connect from base price
            actual_dates = pd.concat([
                pd.Series([inference_date]),
                df_actual["date"]
            ]).values
            actual_prices = np.concatenate([[base_price], df_actual["close"].values])
            ax.plot(actual_dates, actual_prices,
                    color=S["cyan"], lw=1.2, alpha=0.6, label="Precio Real", zorder=3)

    # --- Base price marker ---
    ax.scatter(inference_date, base_price, color=S["amber"], s=120, zorder=10,
               marker="D", edgecolors="white", linewidths=1)
    ax.annotate(f"Base: ${base_price:,.0f}",
                (inference_date, base_price),
                textcoords="offset points", xytext=(-10, -22),
                ha="center", fontsize=8, color=S["amber"], fontweight="bold")

    # --- Predicted prices (dashed line + dots) ---
    pred_dates = [inference_date]
    pred_prices = [base_price]
    for h in horizons_sorted:
        pred_dates.append(target_info[h]["date"])
        pred_prices.append(base_price * np.exp(predictions[h]))

    ax.plot(pred_dates, pred_prices, color=S["accent"], lw=2, ls="--",
            alpha=0.9, marker="o", markersize=5, zorder=7, label="Prediccion")

    # Annotate predicted prices
    for h in horizons_sorted:
        pred_price = base_price * np.exp(predictions[h])
        ret_pct = predictions[h] * 100
        target_date = target_info[h]["date"]
        color = S["green"] if ret_pct > 0 else S["red"]

        ax.scatter(target_date, pred_price, color=color, s=50, zorder=8,
                   edgecolors="white", linewidths=0.5)

        # Annotation: price + return %
        label = f"${pred_price:,.0f}\n{ret_pct:+.2f}%"
        ax.annotate(label, (target_date, pred_price),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=6.5, color=color, fontweight="bold")

    # --- Actual prices at target dates (if available) ---
    has_actuals = False
    for h in horizons_sorted:
        actual_price = target_info[h]["actual_price"]
        if actual_price is not None:
            has_actuals = True
            target_date = target_info[h]["date"]
            pred_price = base_price * np.exp(predictions[h])
            # Color: green if prediction direction was correct
            actual_dir = 1 if actual_price > base_price else -1
            pred_dir = 1 if pred_price > base_price else -1
            marker_color = S["green"] if actual_dir == pred_dir else S["red"]

            ax.scatter(target_date, actual_price, color=marker_color, s=70, zorder=9,
                       marker="x", linewidths=2)
            ax.annotate(f"${actual_price:,.0f}",
                        (target_date, actual_price),
                        textcoords="offset points", xytext=(0, -14),
                        ha="center", fontsize=6.5, color=S["cyan"], fontstyle="italic")

    if has_actuals:
        ax.scatter([], [], color=S["cyan"], marker="x", s=50, label="Precio Real (x)")

    # --- Reference line ---
    ax.axhline(base_price, color=S["amber"], lw=0.5, ls=":", alpha=0.3)

    # --- Title ---
    title = title_override or MODEL_NAMES.get(model_id, model_id)
    ax.set_title(f"{title} — Forecast desde {inference_date.date()}",
                 color=S["text"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Fecha", color=S["text"], fontsize=10)
    ax.set_ylabel("Precio USD/COP", color=S["text"], fontsize=10)

    # --- Date formatting ---
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    fig.autofmt_xdate(rotation=45, ha="right")

    ax.legend(fontsize=8, loc="upper left", facecolor=S["bg"],
              edgecolor=S["grid"], labelcolor=S["text"])

    plt.tight_layout()
    filename = filename_override or f"forward_{model_id}_{week_suffix}.png"
    fig.savefig(output_dir / filename, dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


def generate_consensus_image(
    all_predictions: Dict[str, Dict[int, float]],
    base_price: float,
    inference_date: pd.Timestamp,
    output_dir: Path,
    week_suffix: str,
    df_full: pd.DataFrame,
    target_info: Dict[int, Dict],
):
    """All models consensus with historical context, dates, and actuals."""
    fig, ax = _setup_dark_fig(figsize=(14, 7))

    # Historical prices
    df_hist = df_full[df_full["date"] <= inference_date].tail(25).copy()
    ax.plot(df_hist["date"].values, df_hist["close"].values,
            color=S["text"], lw=1.5, alpha=0.7, label="Historico", zorder=2)

    # Each model as thin line
    for i, (model_id, h_preds) in enumerate(all_predictions.items()):
        horizons_sorted = sorted([h for h in h_preds if h in target_info])
        if not horizons_sorted:
            continue
        dates = [inference_date] + [target_info[h]["date"] for h in horizons_sorted]
        prices = [base_price] + [base_price * np.exp(h_preds[h]) for h in horizons_sorted]
        color = MODEL_COLORS[i % len(MODEL_COLORS)]
        ax.plot(dates, prices, lw=0.8, alpha=0.35, marker=".", markersize=3,
                color=color, label=MODEL_NAMES.get(model_id, model_id))

    # Consensus (average)
    consensus_dates = [inference_date]
    consensus_prices = [base_price]
    min_prices = [base_price]
    max_prices = [base_price]

    for h in HORIZONS:
        if h not in target_info:
            continue
        vals = [all_predictions[m][h] for m in all_predictions if h in all_predictions[m]]
        if vals:
            consensus_dates.append(target_info[h]["date"])
            consensus_prices.append(base_price * np.exp(np.mean(vals)))
            min_prices.append(base_price * np.exp(min(vals)))
            max_prices.append(base_price * np.exp(max(vals)))

    ax.plot(consensus_dates, consensus_prices, color=S["accent"], lw=3,
            marker="o", markersize=8, zorder=10, label="CONSENSO (Promedio)")
    ax.fill_between(consensus_dates, min_prices, max_prices,
                    alpha=0.1, color=S["accent"])

    # Actual prices at target dates
    has_actuals = False
    for h in HORIZONS:
        if h in target_info and target_info[h]["actual_price"] is not None:
            has_actuals = True
            ax.scatter(target_info[h]["date"], target_info[h]["actual_price"],
                       color=S["cyan"], marker="x", s=80, zorder=11, linewidths=2)

    if has_actuals:
        # Actual price line
        actual_dates = [inference_date]
        actual_prices_line = [base_price]
        for h in HORIZONS:
            if h in target_info and target_info[h]["actual_price"] is not None:
                actual_dates.append(target_info[h]["date"])
                actual_prices_line.append(target_info[h]["actual_price"])
        ax.plot(actual_dates, actual_prices_line, color=S["cyan"], lw=1.5,
                ls="-", alpha=0.7, zorder=9, label="Precio Real")

    # Base marker
    ax.scatter(inference_date, base_price, color=S["amber"], s=120, zorder=12,
               marker="D", edgecolors="white", linewidths=1)

    ax.axhline(base_price, color=S["amber"], lw=0.5, ls=":", alpha=0.3)

    ax.set_title(f"Consenso — Forecast desde {inference_date.date()}",
                 color=S["text"], fontsize=13, fontweight="bold")
    ax.set_xlabel("Fecha", color=S["text"], fontsize=10)
    ax.set_ylabel("Precio USD/COP", color=S["text"], fontsize=10)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=1))
    fig.autofmt_xdate(rotation=45, ha="right")

    ax.legend(fontsize=7, loc="upper left", ncol=2, facecolor=S["bg"],
              edgecolor=S["grid"], labelcolor=S["text"])

    plt.tight_layout()
    fig.savefig(output_dir / f"forward_consensus_{week_suffix}.png",
                dpi=120, bbox_inches="tight", facecolor=S["bg"])
    plt.close(fig)


# =============================================================================
# CSV WRITING
# =============================================================================

def write_csv(rows: List[Dict], output_path: Path):
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV written: {output_path} ({len(rows)} rows)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate weekly forecast data (CSV + PNG)")
    parser.add_argument("--week", type=str, help="Single week (e.g., 2025-W50)")
    parser.add_argument("--num-weeks", type=int, default=7, help="Number of retroactive weeks (default: 7)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading full dataset...")
    t0 = time.time()
    df_full = load_full_dataset()
    logger.info(f"Dataset loaded: {len(df_full)} rows in {time.time()-t0:.1f}s")

    # Determine weeks
    if args.week:
        # Single week mode
        parts = args.week.replace("W", "").split("-")
        year, week_num = int(parts[0]), int(parts[1])
        jan4 = datetime(year, 1, 4)
        monday_w1 = jan4 - timedelta(days=jan4.weekday())
        friday = monday_w1 + timedelta(weeks=week_num - 1, days=4)
        cutoff = df_full[df_full["date"] <= pd.Timestamp(friday).normalize()]["date"].max()
        if pd.isna(cutoff):
            logger.error(f"No data for {args.week}")
            sys.exit(1)
        weeks = [(args.week, cutoff)]
    else:
        # Retroactive: last N weeks of data
        weeks = find_last_n_weeks(df_full, n=args.num_weeks)

    logger.info(f"Generating {len(weeks)} weeks: {weeks[0][0]} to {weeks[-1][0]}")

    all_csv_rows = []
    total_images = 0

    for i, (week_label, cutoff_date) in enumerate(weeks):
        is_latest = (i == len(weeks) - 1)
        week_suffix = week_label.replace("-", "_")

        t1 = time.time()
        csv_rows, raw_preds, wf_results, base_price, inf_date = generate_week_data(
            df_full, cutoff_date, week_label, is_latest_week=is_latest
        )
        logger.info(f"  Models trained in {time.time()-t1:.1f}s ({len(csv_rows)} CSV rows)")
        all_csv_rows.extend(csv_rows)

        # Target dates and actuals for this week
        target_info = get_target_dates_and_actuals(df_full, inf_date, base_price)

        # --- Forward forecast images (all weeks) ---
        for model_id in MODEL_IDS:
            if raw_preds.get(model_id):
                generate_forward_forecast_image(
                    raw_preds[model_id], base_price, model_id, inf_date,
                    OUTPUT_DIR, week_suffix, df_full, target_info,
                )
                total_images += 1

        # Consensus image
        generate_consensus_image(
            raw_preds, base_price, inf_date, OUTPUT_DIR,
            week_suffix, df_full, target_info,
        )
        total_images += 1

        # Ensemble images
        model_scores = {}
        for mid, h_preds in raw_preds.items():
            if h_preds:
                model_scores[mid] = np.mean([abs(v) for v in h_preds.values()])
        sorted_models = sorted(model_scores.keys(), key=lambda m: model_scores[m], reverse=True)

        ensemble_configs = [
            ("best_of_breed", "Ensemble: Best of Breed", None),
            ("top_3", "Ensemble: Top 3", sorted_models[:3]),
            ("top_6_mean", "Ensemble: Top 6", sorted_models[:6]),
        ]

        for ens_key, ens_label, model_subset in ensemble_configs:
            ens_preds = {}
            for h in HORIZONS:
                if ens_key == "best_of_breed":
                    best_val, best_abs = None, -1
                    for m in raw_preds:
                        if h in raw_preds[m] and abs(raw_preds[m][h]) > best_abs:
                            best_abs = abs(raw_preds[m][h])
                            best_val = raw_preds[m][h]
                    if best_val is not None:
                        ens_preds[h] = best_val
                else:
                    vals = [raw_preds[m][h] for m in model_subset if h in raw_preds.get(m, {})]
                    if vals:
                        ens_preds[h] = float(np.mean(vals))

            if ens_preds:
                generate_forward_forecast_image(
                    ens_preds, base_price, f"ensemble_{ens_key}", inf_date,
                    OUTPUT_DIR, week_suffix, df_full, target_info,
                    title_override=ens_label,
                    filename_override=f"forward_ensemble_{ens_key}_{week_suffix}.png",
                )
                total_images += 1

        # --- Backtest images (latest week only) ---
        if is_latest:
            logger.info("  Generating backtest images (latest week)...")
            for model_id in MODEL_IDS:
                for horizon in HORIZONS:
                    if horizon in wf_results.get(model_id, {}):
                        wf = wf_results[model_id][horizon]
                        if len(wf["predicted_returns"]) >= 5:
                            generate_backtest_image(wf, model_id, horizon, OUTPUT_DIR)
                            total_images += 1

    # Write combined CSV
    csv_path = OUTPUT_DIR / "bi_dashboard_unified.csv"
    write_csv(all_csv_rows, csv_path)

    # Summary
    bt_rows = sum(1 for r in all_csv_rows if r["view_type"] == "backtest")
    ff_rows = sum(1 for r in all_csv_rows if r["view_type"] == "forward_forecast")
    total_time = time.time() - t0

    logger.info(f"\nDone in {total_time:.1f}s!")
    logger.info(f"  Weeks: {len(weeks)} ({weeks[0][0]} to {weeks[-1][0]})")
    logger.info(f"  CSV: {bt_rows} backtest + {ff_rows} forward_forecast = {len(all_csv_rows)} rows")
    logger.info(f"  Images: {total_images} total")
    logger.info(f"  Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
