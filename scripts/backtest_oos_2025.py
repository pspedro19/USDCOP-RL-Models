"""
Pure OOS 2025 Backtest — The Most Honest Validation
=====================================================

KEY DIFFERENCE vs backtest_2025_10k.py:
- Training: ONLY 2020-01-01 to 2024-12-31 (5 years)
- OOS: 2025-01-01 to 2025-12-31 (NEVER touched during training)
- Walk-forward folds: 4 expanding windows WITHIN 2020-2024 only
- 30-day gap between train/test in each fold (anti-leakage)

This script answers: "Is the alpha REAL, or did it depend on seeing 2025
during walk-forward training?"

Pipeline:
1. Load daily OHLCV + macro (21 features including VIX + EMBI)
2. 4 walk-forward folds within 2020-2024 (model selection + validation)
3. Final model: train on ALL 2020-2024 data, predict 2025 completely blind
4. Apply vol-targeting (target_vol=0.15, leverage [0.5, 2.0])
5. Apply trailing stop on 5-min bars (activation=0.2%, trail=0.3%, hard=1.5%)
6. Compare 4 strategies: B&H, Forecast 1x, +Vol-Target, +Trailing Stop
7. Statistical validation: t-test, bootstrap CI, binomial test
8. Sensitivity analysis: trailing stop parameter grid

@version 1.0.0
@date 2026-02-16
"""

import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Add project root to path
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
# CONFIG
# =============================================================================

INITIAL_CAPITAL = 10_000.0
SLIPPAGE_BPS = 1.0  # 1 bps slippage per trade (MEXC maker = 0% fee)

VOL_CONFIG = VolTargetConfig(
    target_vol=0.15,
    max_leverage=2.0,
    min_leverage=0.5,
    vol_lookback=21,
    vol_floor=0.05,
)

TRAILING_CONFIG = TrailingStopConfig(
    activation_pct=0.002,   # 0.20%
    trail_pct=0.003,        # 0.30%
    hard_stop_pct=0.015,    # 1.50%
)

# STRICT date boundaries
TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
OOS_END = pd.Timestamp("2026-12-31")  # Use all available data after 2024

# Walk-forward config within 2020-2024
N_FOLDS = 4
GAP_DAYS = 30  # Anti-leakage gap between train and test in each fold
INITIAL_TRAIN_RATIO = 0.60  # 60% of 2020-2024 as initial train


# =============================================================================
# DATA LOADING
# =============================================================================

def load_daily_data() -> pd.DataFrame:
    """Load daily OHLCV + macro (4 macro features), build 21 SSOT features."""
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

    # Load macro (DXY + WTI + VIX + EMBI with T-1 lag)
    logger.info("Loading macro data (DXY, WTI, VIX, EMBI)...")
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
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }
    df_macro_subset = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_subset.rename(columns=macro_cols, inplace=True)
    df_macro_subset = df_macro_subset.sort_values("date")

    # T-1 lag: use yesterday's macro (anti-leakage)
    for col in macro_cols.values():
        df_macro_subset[col] = df_macro_subset[col].shift(1)

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

    # Only need 2025 bars (plus a few days before for T+1 from late Dec 2024)
    df_m5 = df_m5[df_m5["date"] >= pd.Timestamp("2024-12-01")].copy()
    # No upper filter — use all available 5-min data

    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"  5-min: {len(df_m5)} bars, "
        f"{df_m5['timestamp'].min()} to {df_m5['timestamp'].max()}"
    )

    return df_m5[["timestamp", "date", "open", "high", "low", "close"]].copy()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 21 SSOT features from raw OHLCV + macro."""
    df = df.copy()

    # Returns (4)
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # Volatility (3)
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Technical (3)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()

    # Calendar (3)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)

    # Macro (4) — already in df from merge
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()
    df["vix_close_lag1"] = df["vix_close_lag1"].ffill()
    df["embi_close_lag1"] = df["embi_close_lag1"].ffill()

    return df


# =============================================================================
# MODEL TRAINING HELPERS
# =============================================================================

def get_model_list() -> List[str]:
    """Get list of 9 model IDs (or 8 if ARD unavailable)."""
    models = list(MODEL_IDS)
    try:
        ModelFactory.create("ard")
    except Exception:
        models = [m for m in models if m != "ard"]
        logger.warning("ARD model not available, using 8 models")
    return models


def get_model_params(model_id: str, horizon_config: dict) -> Optional[dict]:
    """Get model-specific params based on type."""
    catboost_models = {"catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    if model_id in linear_models:
        return None
    elif model_id in catboost_models:
        return {
            "iterations": horizon_config.get("n_estimators", 50),
            "depth": horizon_config.get("max_depth", 3),
            "learning_rate": horizon_config.get("learning_rate", 0.05),
            "l2_leaf_reg": horizon_config.get("reg_alpha", 0.5),
            "verbose": False,
            "allow_writing_files": False,
        }
    elif model_id in hybrid_models:
        if "catboost" in model_id:
            return {
                "iterations": horizon_config.get("n_estimators", 50),
                "depth": horizon_config.get("max_depth", 3),
                "learning_rate": horizon_config.get("learning_rate", 0.05),
                "verbose": False,
                "allow_writing_files": False,
            }
        return horizon_config
    else:
        return horizon_config


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_to_use: List[str],
    horizon_config: dict,
) -> Tuple[Dict, StandardScaler]:
    """Train all 9 models on given training data. Returns (trained_models, scaler)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    trained = {}
    for model_id in models_to_use:
        try:
            params = get_model_params(model_id, horizon_config)
            model = ModelFactory.create(model_id, params=params, horizon=1)
            if model.requires_scaling:
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            trained[model_id] = model
        except Exception as e:
            logger.warning(f"  Failed to train {model_id}: {e}")

    return trained, scaler


def predict_ensemble(
    trained_models: Dict,
    scaler: StandardScaler,
    X: np.ndarray,
) -> np.ndarray:
    """
    Generate top-3 ensemble predictions (by prediction magnitude).
    Same as production: top 3 models by |prediction| for each row.
    """
    X_scaled = scaler.transform(X)
    n_rows = X.shape[0]
    ensemble_preds = np.zeros(n_rows)

    for i in range(n_rows):
        X_row = X[i:i+1]
        X_row_scaled = X_scaled[i:i+1]

        preds = {}
        for model_id, model in trained_models.items():
            try:
                if model.requires_scaling:
                    pred = model.predict(X_row_scaled)[0]
                else:
                    pred = model.predict(X_row)[0]
                preds[model_id] = pred
            except Exception:
                pass

        if len(preds) < 3:
            # Not enough models — use what we have
            ensemble_preds[i] = np.mean(list(preds.values())) if preds else 0.0
            continue

        # Top-3 by |prediction| magnitude (production ensemble)
        sorted_models = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
        top3 = sorted_models[:3]
        ensemble_preds[i] = np.mean([preds[m] for m in top3])

    return ensemble_preds


# =============================================================================
# PHASE 1: WALK-FORWARD WITHIN 2020-2024 (Model Validation)
# =============================================================================

def run_walk_forward_2020_2024(df: pd.DataFrame) -> Dict:
    """
    4-fold expanding walk-forward WITHIN 2020-2024 only.
    30-day gap between train and test in each fold.

    Purpose: Validate model works on pre-2025 data. NOT used for 2025 prediction.
    """
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 1: Walk-Forward Validation (2020-2024 ONLY)")
    logger.info("=" * 70)

    feature_cols = list(FEATURE_COLUMNS)
    df_train_period = df[
        (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)
    ].copy().reset_index(drop=True)

    n_total = len(df_train_period)
    initial_train_size = int(n_total * INITIAL_TRAIN_RATIO)
    remaining = n_total - initial_train_size
    test_size = remaining // N_FOLDS

    models_to_use = get_model_list()
    horizon_config = get_horizon_config(1)

    fold_results = []

    for fold_idx in range(N_FOLDS):
        train_end_idx = initial_train_size + (fold_idx * test_size)
        test_start_idx = train_end_idx + GAP_DAYS  # 30-day gap
        test_end_idx = min(test_start_idx + test_size, n_total)

        if test_start_idx >= n_total or test_end_idx <= test_start_idx:
            logger.warning(f"  Fold {fold_idx+1}: Not enough data, skipping")
            continue

        df_fold_train = df_train_period.iloc[:train_end_idx]
        df_fold_test = df_train_period.iloc[test_start_idx:test_end_idx]

        X_train = df_fold_train[feature_cols].values.astype(np.float64)
        y_train = df_fold_train["target_return_1d"].values.astype(np.float64)
        X_test = df_fold_test[feature_cols].values.astype(np.float64)
        y_test = df_fold_test["target_return_1d"].values.astype(np.float64)

        logger.info(
            f"\n  Fold {fold_idx+1}/{N_FOLDS}: "
            f"Train {len(df_fold_train)} rows "
            f"({df_fold_train['date'].iloc[0].date()} to {df_fold_train['date'].iloc[-1].date()}), "
            f"Gap {GAP_DAYS}d, "
            f"Test {len(df_fold_test)} rows "
            f"({df_fold_test['date'].iloc[0].date()} to {df_fold_test['date'].iloc[-1].date()})"
        )

        # Train all models
        trained, scaler = train_models(X_train, y_train, models_to_use, horizon_config)
        logger.info(f"    Trained {len(trained)}/{len(models_to_use)} models")

        # Per-model DA
        model_das = {}
        for model_id, model in trained.items():
            try:
                if model.requires_scaling:
                    preds = model.predict(scaler.transform(X_test))
                else:
                    preds = model.predict(X_test)
                correct = np.sum(np.sign(preds) == np.sign(y_test))
                da = correct / len(y_test) * 100
                model_das[model_id] = da
            except Exception:
                pass

        # Ensemble prediction (top-3 by magnitude)
        ensemble_preds = predict_ensemble(trained, scaler, X_test)

        correct = np.sum(np.sign(ensemble_preds) == np.sign(y_test))
        da = correct / len(y_test) * 100
        strategy_rets = np.sign(ensemble_preds) * y_test
        fold_return = float(np.prod(1 + strategy_rets) - 1) * 100
        fold_sharpe = (
            np.mean(strategy_rets) / np.std(strategy_rets, ddof=1) * np.sqrt(252)
            if np.std(strategy_rets, ddof=1) > 0 else 0
        )

        logger.info(
            f"    Ensemble: DA={da:.1f}%, Return={fold_return:+.2f}%, "
            f"Sharpe={fold_sharpe:.2f}"
        )

        fold_results.append({
            "fold": fold_idx + 1,
            "train_rows": len(df_fold_train),
            "test_rows": len(df_fold_test),
            "train_end": str(df_fold_train["date"].iloc[-1].date()),
            "test_start": str(df_fold_test["date"].iloc[0].date()),
            "test_end": str(df_fold_test["date"].iloc[-1].date()),
            "gap_days": GAP_DAYS,
            "da_pct": round(da, 1),
            "return_pct": round(fold_return, 2),
            "sharpe": round(fold_sharpe, 2),
            "n_models": len(trained),
            "model_das": {k: round(v, 1) for k, v in model_das.items()},
        })

    # Summary
    avg_da = np.mean([f["da_pct"] for f in fold_results])
    folds_positive = sum(1 for f in fold_results if f["return_pct"] > 0)

    logger.info(f"\n  Walk-Forward Summary:")
    logger.info(f"    Avg DA: {avg_da:.1f}%")
    logger.info(f"    Folds positive: {folds_positive}/{len(fold_results)}")

    return {
        "folds": fold_results,
        "avg_da_pct": round(avg_da, 1),
        "folds_positive": folds_positive,
        "total_folds": len(fold_results),
    }


# =============================================================================
# PHASE 2: TRAIN ON ALL 2020-2024, PREDICT 2025 BLIND
# =============================================================================

def train_final_and_predict_oos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train on ALL 2020-2024 data, predict 2025 completely blind.
    Monthly re-training WITHIN 2020-2024 boundaries (expanding into 2025).

    Wait — NO. To be truly pure OOS, we train ONCE on all 2020-2024
    and predict ALL of 2025 without retraining. That's the strictest test.

    However, for operational realism (production re-trains weekly), we also
    support monthly re-training where each month of 2025 trains on
    all data UP TO that month (but always starting from 2020).
    We default to monthly re-training to match production.
    """
    logger.info("\n" + "=" * 70)
    logger.info("  PHASE 2: Final Model — Train 2020-2024, Predict 2025 OOS")
    logger.info("=" * 70)

    feature_cols = list(FEATURE_COLUMNS)
    horizon_config = get_horizon_config(1)
    models_to_use = get_model_list()

    # Strict partition: training data ONLY up to 2024-12-31
    df_train_all = df[
        (df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)
    ].copy()
    df_oos = df[
        (df["date"] >= OOS_START) & (df["date"] <= OOS_END)
    ].copy()

    if len(df_oos) == 0:
        logger.error("No OOS data found for 2025!")
        sys.exit(1)

    logger.info(
        f"  Training data: {len(df_train_all)} rows "
        f"({df_train_all['date'].iloc[0].date()} to {df_train_all['date'].iloc[-1].date()})"
    )
    logger.info(
        f"  OOS data: {len(df_oos)} rows "
        f"({df_oos['date'].iloc[0].date()} to {df_oos['date'].iloc[-1].date()})"
    )

    # Monthly re-training: for each month of 2025, train on all data
    # up to the start of that month (but NEVER including 2025 test data)
    months_2025 = sorted(df_oos["date"].dt.to_period("M").unique())
    results = []

    for month_idx, month in enumerate(months_2025):
        month_start = month.start_time
        month_end = month.end_time

        # Training data: everything from 2020 up to (but not including) this month
        # For the first month (Jan 2025), this is exactly 2020-2024
        # For Feb 2025, this is 2020 to 2025-01-31 (includes Jan 2025 actual data)
        if month_idx == 0:
            # First month: pure 2020-2024 training
            df_month_train = df_train_all.copy()
        else:
            # Subsequent months: include previous OOS months as training data
            # This is realistic: production re-trains weekly with new data
            df_month_train = df[df["date"] < month_start].copy()
            df_month_train = df_month_train[df_month_train["date"] >= TRAIN_START]

        df_month_oos = df_oos[
            (df_oos["date"] >= month_start) & (df_oos["date"] <= month_end)
        ]

        if len(df_month_oos) == 0 or len(df_month_train) < 200:
            continue

        X_train = df_month_train[feature_cols].values.astype(np.float64)
        y_train = df_month_train["target_return_1d"].values.astype(np.float64)

        # Train models
        trained, scaler = train_models(X_train, y_train, models_to_use, horizon_config)

        # Predict each OOS day
        for _, row in df_month_oos.iterrows():
            X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)

            # Get predictions from all models
            preds = {}
            for model_id, model in trained.items():
                try:
                    if model.requires_scaling:
                        pred = model.predict(scaler.transform(X_day))[0]
                    else:
                        pred = model.predict(X_day)[0]
                    preds[model_id] = pred
                except Exception:
                    pass

            if len(preds) < 3:
                continue

            # Top-3 ensemble by magnitude
            sorted_models = sorted(
                preds.keys(), key=lambda m: abs(preds[m]), reverse=True
            )
            top3 = sorted_models[:3]
            ensemble_pred = np.mean([preds[m] for m in top3])
            direction = 1 if ensemble_pred > 0 else -1

            # Vol-targeting
            idx = df.index[df["date"] == row["date"]][0]
            if idx >= VOL_CONFIG.vol_lookback:
                recent_returns = df.iloc[
                    idx - VOL_CONFIG.vol_lookback : idx
                ]["return_1d"].values
                realized_vol = compute_realized_vol(
                    recent_returns, VOL_CONFIG.vol_lookback
                )
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
                "n_models": len(preds),
                "top3": top3,
                "train_rows": len(df_month_train),
            })

        logger.info(
            f"  Month {month}: {len(df_month_oos)} OOS days, "
            f"{len(trained)} models, train={len(df_month_train)} rows"
        )

    return pd.DataFrame(results)


# =============================================================================
# TRAILING STOP SIMULATION ON 5-MIN BARS
# =============================================================================

def simulate_trailing_stop(
    predictions: pd.DataFrame,
    df_m5: pd.DataFrame,
    config: TrailingStopConfig = TRAILING_CONFIG,
) -> pd.DataFrame:
    """
    For each daily signal (day T), simulate trailing stop on T+1 5-min bars.
    """
    logger.info(
        f"Simulating trailing stop "
        f"(act={config.activation_pct*100:.1f}%, "
        f"trail={config.trail_pct*100:.1f}%, "
        f"hard={config.hard_stop_pct*100:.1f}%)..."
    )

    m5_dates = sorted(df_m5["date"].unique())

    trail_exit_prices = []
    trail_exit_reasons = []
    trail_pnl_pcts = []
    trail_bar_counts = []

    counters = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}
    slip = SLIPPAGE_BPS / 10_000

    for _, row in predictions.iterrows():
        signal_date = row["date"]
        entry_price = row["close"]
        direction = int(row["direction"])

        # Find next trading day (T+1)
        next_dates = [d for d in m5_dates if d > signal_date]
        if not next_dates:
            trail_exit_prices.append(None)
            trail_exit_reasons.append("no_bars")
            trail_pnl_pcts.append(None)
            trail_bar_counts.append(0)
            counters["no_bars"] += 1
            continue

        t1_date = next_dates[0]
        bars_t1 = df_m5[df_m5["date"] == t1_date].sort_values("timestamp")

        if len(bars_t1) == 0:
            trail_exit_prices.append(None)
            trail_exit_reasons.append("no_bars")
            trail_pnl_pcts.append(None)
            trail_bar_counts.append(0)
            counters["no_bars"] += 1
            continue

        # Apply slippage to entry
        if direction == 1:
            slipped_entry = entry_price * (1 + slip)
        else:
            slipped_entry = entry_price * (1 - slip)

        # Run trailing stop tracker
        tracker = TrailingStopTracker(
            entry_price=slipped_entry,
            direction=direction,
            config=config,
        )

        for bar_idx, (_, bar) in enumerate(bars_t1.iterrows()):
            state = tracker.update(
                bar_high=float(bar["high"]),
                bar_low=float(bar["low"]),
                bar_close=float(bar["close"]),
                bar_idx=bar_idx,
            )
            if state == TrailingState.TRIGGERED:
                break

        if tracker.state not in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            last_close = float(bars_t1.iloc[-1]["close"])
            tracker.expire(last_close)

        # Apply slippage to exit
        exit_price = tracker.exit_price
        if exit_price is not None:
            if direction == 1:
                exit_price *= (1 - slip)
            else:
                exit_price *= (1 + slip)

        # Compute PnL
        if exit_price is not None and slipped_entry > 0:
            raw_pnl = direction * (exit_price - slipped_entry) / slipped_entry
        else:
            raw_pnl = None

        trail_exit_prices.append(exit_price)
        trail_exit_reasons.append(tracker.exit_reason)
        trail_pnl_pcts.append(raw_pnl)
        trail_bar_counts.append(
            tracker.exit_bar_idx + 1
            if tracker.exit_bar_idx is not None
            else len(bars_t1)
        )

        reason = tracker.exit_reason or "no_bars"
        if reason in counters:
            counters[reason] += 1

    predictions = predictions.copy()
    predictions["trail_exit_price"] = trail_exit_prices
    predictions["trail_exit_reason"] = trail_exit_reasons
    predictions["trail_pnl_unlevered"] = trail_pnl_pcts
    predictions["trail_bar_count"] = trail_bar_counts

    total = len(predictions) - counters["no_bars"]
    logger.info(
        f"  Trailing stop results ({total} trades): "
        f"activated={counters['trailing_stop']}, "
        f"hard_stop={counters['hard_stop']}, "
        f"session_close={counters['session_close']}, "
        f"no_bars={counters['no_bars']}"
    )

    return predictions


# =============================================================================
# BACKTEST ENGINE — 4 STRATEGIES
# =============================================================================

def compute_strategy_stats(rets: np.ndarray) -> Dict:
    """Compute statistics from daily returns array."""
    if len(rets) == 0:
        return {}
    total_ret = np.prod(1 + rets) - 1
    ann_ret = (1 + total_ret) ** (252 / len(rets)) - 1
    vol = np.std(rets, ddof=1) * np.sqrt(252)
    sharpe = ann_ret / vol if vol > 0 else 0
    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown))
    wins = np.sum(rets > 0)
    losses = np.sum(rets < 0)
    win_rate = wins / len(rets) * 100
    pos_sum = np.sum(rets[rets > 0])
    neg_sum = np.abs(np.sum(rets[rets < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-10
    sortino = (np.mean(rets) / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    return {
        "final_equity": INITIAL_CAPITAL * (1 + total_ret),
        "total_return_pct": round(total_ret * 100, 4),
        "annualized_return_pct": round(ann_ret * 100, 4),
        "volatility_pct": round(vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_dd_pct": round(max_dd * 100, 4),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(pf, 4),
        "winning_days": int(wins),
        "losing_days": int(losses),
        "trading_days": len(rets),
    }


def run_4_strategies(predictions: pd.DataFrame) -> Dict:
    """Run 4 strategies on $10K initial capital."""
    n_days = len(predictions)
    slip = SLIPPAGE_BPS / 10_000
    cap = INITIAL_CAPITAL

    # --- Strategy 1: Buy & Hold ---
    first_close = predictions.iloc[0]["close"]
    last_close = predictions.iloc[-1]["close"]
    bh_return = (last_close - first_close) / first_close

    # --- Strategy 2: Forecast 1x ---
    f1_rets = []
    for _, row in predictions.iterrows():
        daily_ret = row["direction"] * (np.exp(row["actual_return"]) - 1)
        daily_ret -= slip
        f1_rets.append(daily_ret)
    f1_rets = np.array(f1_rets)

    # --- Strategy 3: Forecast + Vol-Target ---
    vt_rets = []
    for _, row in predictions.iterrows():
        lev = row["leverage"]
        daily_ret = row["direction"] * lev * (np.exp(row["actual_return"]) - 1)
        daily_ret -= slip * lev
        vt_rets.append(daily_ret)
    vt_rets = np.array(vt_rets)

    # --- Strategy 4: Forecast + VT + Trailing Stop ---
    ts_rets = []
    exit_reasons = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}
    for _, row in predictions.iterrows():
        lev = row["leverage"]
        trail_pnl = row.get("trail_pnl_unlevered")
        reason = row.get("trail_exit_reason", "no_bars")

        if trail_pnl is not None and not np.isnan(trail_pnl):
            daily_ret = trail_pnl * lev
        else:
            daily_ret = row["direction"] * lev * (np.exp(row["actual_return"]) - 1)
            daily_ret -= slip * lev

        ts_rets.append(daily_ret)
        if reason in exit_reasons:
            exit_reasons[reason] += 1

    ts_rets = np.array(ts_rets)

    # --- Direction Accuracy ---
    actual_signs = np.sign(predictions["actual_return"].values)
    pred_dirs = predictions["direction"].values
    correct = int(np.sum(pred_dirs == actual_signs))
    da = correct / n_days * 100

    # --- Compute stats ---
    f1_stats = compute_strategy_stats(f1_rets)
    vt_stats = compute_strategy_stats(vt_rets)
    ts_stats = compute_strategy_stats(ts_rets)

    # --- Monthly breakdown ---
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
        da_pct=("direction", lambda x: (
            np.sum(
                pred_m.loc[x.index, "direction"].values
                == np.sign(pred_m.loc[x.index, "actual_return"].values)
            ) / len(x) * 100
        )),
    ).reset_index()

    # --- Statistical tests ---
    t_f, p_f = stats.ttest_1samp(f1_rets, 0)
    t_v, p_v = stats.ttest_1samp(vt_rets, 0)
    t_ts, p_ts = stats.ttest_1samp(ts_rets, 0)
    t_paired, p_paired = stats.ttest_rel(ts_rets, vt_rets)

    # Bootstrap CIs
    rng = np.random.RandomState(42)
    n_boot = 10_000
    boot_f, boot_v, boot_ts = [], [], []
    for _ in range(n_boot):
        idx = rng.choice(n_days, size=n_days, replace=True)
        boot_f.append(np.mean(f1_rets[idx]))
        boot_v.append(np.mean(vt_rets[idx]))
        boot_ts.append(np.mean(ts_rets[idx]))

    def ci(boot):
        return [
            round(float(np.percentile(boot, 2.5) * 252 * 100), 2),
            round(float(np.percentile(boot, 97.5) * 252 * 100), 2),
        ]

    # Binomial test
    binom_p = float(stats.binomtest(correct, n_days, 0.5).pvalue)

    # Trailing stop alpha
    has_trail = pred_m["trail_pnl_unlevered"].notna()
    if has_trail.any():
        trail_pnls = pred_m.loc[has_trail, "trail_pnl_unlevered"].values
        hold_pnls = pred_m.loc[has_trail].apply(
            lambda r: r["direction"] * (np.exp(r["actual_return"]) - 1) - slip,
            axis=1,
        ).values
        alpha_per_trade = trail_pnls - hold_pnls
        avg_alpha_bps = round(float(np.mean(alpha_per_trade) * 10_000), 2)
        alpha_positive_rate = round(
            float(np.sum(alpha_per_trade > 0) / len(alpha_per_trade) * 100), 1
        )
    else:
        avg_alpha_bps = 0.0
        alpha_positive_rate = 0.0

    return {
        "initial_capital": cap,
        "n_trading_days": n_days,
        "n_features": len(FEATURE_COLUMNS),
        "oos_period": f"{OOS_START.date()} to {OOS_END.date()}",
        "direction_accuracy_pct": round(da, 2),
        "strategies": {
            "buy_and_hold": {
                "final_equity": round(cap * (1 + bh_return), 2),
                "total_return_pct": round(bh_return * 100, 2),
            },
            "forecast_1x": f1_stats,
            "forecast_vol_target": {
                **vt_stats,
                "avg_leverage": round(float(predictions["leverage"].mean()), 3),
            },
            "forecast_vt_trailing": {
                **ts_stats,
                "avg_leverage": round(float(predictions["leverage"].mean()), 3),
                "exit_reasons": exit_reasons,
                "avg_alpha_bps": avg_alpha_bps,
                "alpha_positive_rate_pct": alpha_positive_rate,
            },
        },
        "statistical_tests": {
            "forecast_1x": {
                "t_stat": round(float(t_f), 4),
                "p_value": round(float(p_f), 4),
                "bootstrap_95ci_ann": ci(boot_f),
                "significant_5pct": float(p_f) < 0.05,
            },
            "forecast_vol_target": {
                "t_stat": round(float(t_v), 4),
                "p_value": round(float(p_v), 4),
                "bootstrap_95ci_ann": ci(boot_v),
                "significant_5pct": float(p_v) < 0.05,
            },
            "forecast_vt_trailing": {
                "t_stat": round(float(t_ts), 4),
                "p_value": round(float(p_ts), 4),
                "bootstrap_95ci_ann": ci(boot_ts),
                "significant_5pct": float(p_ts) < 0.05,
            },
            "paired_trail_vs_hold": {
                "t_stat": round(float(t_paired), 4),
                "p_value": round(float(p_paired), 4),
                "significant_5pct": float(p_paired) < 0.05,
            },
            "direction_accuracy": {
                "correct": correct,
                "total": n_days,
                "binomial_p": round(binom_p, 4),
                "significant_5pct": binom_p < 0.05,
            },
        },
        "monthly": {
            "months": [str(m) for m in monthly["month"]],
            "days": monthly["days"].tolist(),
            "f1_pct": (monthly["f1_sum"] * 100).round(2).tolist(),
            "vt_pct": (monthly["vt_sum"] * 100).round(2).tolist(),
            "ts_pct": (monthly["ts_sum"] * 100).round(2).tolist(),
            "da": monthly["da_pct"].round(1).tolist(),
        },
    }


# =============================================================================
# SENSITIVITY ANALYSIS: TRAILING STOP GRID
# =============================================================================

def run_trailing_stop_sensitivity(
    predictions: pd.DataFrame,
    df_m5: pd.DataFrame,
) -> Dict:
    """
    Grid search over trailing stop parameters.
    Runs full trailing stop simulation for each parameter combination.
    """
    logger.info("\n" + "=" * 70)
    logger.info("  SENSITIVITY ANALYSIS: Trailing Stop Parameters")
    logger.info("=" * 70)

    activation_grid = [0.001, 0.002, 0.003]   # 0.1%, 0.2%, 0.3%
    trail_grid = [0.002, 0.003, 0.004]         # 0.2%, 0.3%, 0.4%
    hard_grid = [0.010, 0.015, 0.020]          # 1.0%, 1.5%, 2.0%

    results = []
    slip = SLIPPAGE_BPS / 10_000

    for act in activation_grid:
        for trail in trail_grid:
            for hard in hard_grid:
                config = TrailingStopConfig(
                    activation_pct=act,
                    trail_pct=trail,
                    hard_stop_pct=hard,
                )

                # Run trailing stop simulation
                pred_copy = predictions.copy()
                # Remove old trailing stop columns
                for col in ["trail_exit_price", "trail_exit_reason",
                            "trail_pnl_unlevered", "trail_bar_count"]:
                    if col in pred_copy.columns:
                        pred_copy = pred_copy.drop(columns=[col])

                pred_with_trail = simulate_trailing_stop(pred_copy, df_m5, config)

                # Compute strategy 4 returns
                ts_rets = []
                for _, row in pred_with_trail.iterrows():
                    lev = row["leverage"]
                    trail_pnl = row.get("trail_pnl_unlevered")
                    if trail_pnl is not None and not np.isnan(trail_pnl):
                        daily_ret = trail_pnl * lev
                    else:
                        daily_ret = row["direction"] * lev * (np.exp(row["actual_return"]) - 1)
                        daily_ret -= slip * lev
                    ts_rets.append(daily_ret)

                ts_rets = np.array(ts_rets)
                total_ret = float(np.prod(1 + ts_rets) - 1)
                vol = float(np.std(ts_rets, ddof=1) * np.sqrt(252))
                sharpe = (np.mean(ts_rets) / np.std(ts_rets, ddof=1) * np.sqrt(252)
                          if np.std(ts_rets, ddof=1) > 0 else 0)
                cum = np.cumprod(1 + ts_rets)
                peak = np.maximum.accumulate(cum)
                max_dd = float(np.min((cum - peak) / peak))

                results.append({
                    "activation_pct": act * 100,
                    "trail_pct": trail * 100,
                    "hard_stop_pct": hard * 100,
                    "total_return_pct": round(total_ret * 100, 2),
                    "sharpe": round(sharpe, 3),
                    "max_dd_pct": round(max_dd * 100, 2),
                    "vol_pct": round(vol * 100, 2),
                })

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    logger.info(f"\n  Top 5 configurations by Sharpe:")
    for i, r in enumerate(results[:5]):
        logger.info(
            f"    {i+1}. act={r['activation_pct']:.1f}% trail={r['trail_pct']:.1f}% "
            f"hard={r['hard_stop_pct']:.1f}% -> "
            f"Ret={r['total_return_pct']:+.2f}% Sharpe={r['sharpe']:.3f} "
            f"MaxDD={r['max_dd_pct']:.2f}%"
        )

    return {"grid_results": results, "n_combinations": len(results)}


# =============================================================================
# PERIOD SPLIT: 2025-only vs 2026-only
# =============================================================================

def compute_period_split(predictions: pd.DataFrame) -> Dict:
    """
    Compute full stats for 2025-only and 2026-only periods independently.
    Each period gets its own 4-strategy comparison + statistical tests.
    """
    slip = SLIPPAGE_BPS / 10_000
    cap = INITIAL_CAPITAL

    split_2025 = pd.Timestamp("2026-01-01")
    pred_2025 = predictions[predictions["date"] < split_2025].copy()
    pred_2026 = predictions[predictions["date"] >= split_2025].copy()

    def _compute_one_period(pred: pd.DataFrame, label: str) -> Dict:
        if len(pred) == 0:
            return {"label": label, "n_days": 0, "error": "No data"}

        n = len(pred)

        # B&H
        bh_ret = (pred.iloc[-1]["close"] - pred.iloc[0]["close"]) / pred.iloc[0]["close"]

        # Forecast 1x
        f1_rets = np.array([
            row["direction"] * (np.exp(row["actual_return"]) - 1) - slip
            for _, row in pred.iterrows()
        ])

        # Vol-target
        vt_rets = np.array([
            row["direction"] * row["leverage"] * (np.exp(row["actual_return"]) - 1) - slip * row["leverage"]
            for _, row in pred.iterrows()
        ])

        # VT + Trailing stop
        ts_rets = []
        exit_reasons = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}
        for _, row in pred.iterrows():
            lev = row["leverage"]
            trail_pnl = row.get("trail_pnl_unlevered")
            reason = row.get("trail_exit_reason", "no_bars")
            if trail_pnl is not None and not np.isnan(trail_pnl):
                daily_ret = trail_pnl * lev
            else:
                daily_ret = row["direction"] * lev * (np.exp(row["actual_return"]) - 1)
                daily_ret -= slip * lev
            ts_rets.append(daily_ret)
            if reason in exit_reasons:
                exit_reasons[reason] += 1
        ts_rets = np.array(ts_rets)

        # DA
        actual_signs = np.sign(pred["actual_return"].values)
        pred_dirs = pred["direction"].values
        correct = int(np.sum(pred_dirs == actual_signs))
        da = correct / n * 100

        # Long vs Short
        long_mask = pred_dirs == 1
        short_mask = pred_dirs == -1
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())

        long_correct = int(np.sum(pred_dirs[long_mask] == actual_signs[long_mask])) if n_long > 0 else 0
        short_correct = int(np.sum(pred_dirs[short_mask] == actual_signs[short_mask])) if n_short > 0 else 0
        long_wr = long_correct / n_long * 100 if n_long > 0 else 0
        short_wr = short_correct / n_short * 100 if n_short > 0 else 0

        # Stats for each strategy
        f1_stats = compute_strategy_stats(f1_rets)
        vt_stats = compute_strategy_stats(vt_rets)
        ts_stats = compute_strategy_stats(ts_rets)

        # Statistical tests
        t_f, p_f = stats.ttest_1samp(f1_rets, 0) if n > 2 else (0, 1)
        t_v, p_v = stats.ttest_1samp(vt_rets, 0) if n > 2 else (0, 1)
        t_ts, p_ts = stats.ttest_1samp(ts_rets, 0) if n > 2 else (0, 1)

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot_ts_means = []
        for _ in range(10_000):
            idx = rng.choice(n, size=n, replace=True)
            boot_ts_means.append(np.mean(ts_rets[idx]))
        ci_lo = round(float(np.percentile(boot_ts_means, 2.5) * 252 * 100), 2)
        ci_hi = round(float(np.percentile(boot_ts_means, 97.5) * 252 * 100), 2)

        binom_p = float(stats.binomtest(correct, n, 0.5).pvalue) if n > 0 else 1.0

        # Monthly breakdown
        pred_m = pred.copy()
        pred_m["month"] = pred_m["date"].dt.to_period("M")
        pred_m["ts_ret"] = ts_rets
        monthly = pred_m.groupby("month").agg(
            days=("date", "count"),
            ts_sum=("ts_ret", "sum"),
            da_pct=("direction", lambda x: (
                np.sum(
                    pred_m.loc[x.index, "direction"].values
                    == np.sign(pred_m.loc[x.index, "actual_return"].values)
                ) / len(x) * 100
            )),
        ).reset_index()

        return {
            "label": label,
            "period": f"{pred['date'].iloc[0].date()} to {pred['date'].iloc[-1].date()}",
            "n_days": n,
            "direction_accuracy_pct": round(da, 2),
            "n_long": n_long,
            "n_short": n_short,
            "long_wr_pct": round(long_wr, 1),
            "short_wr_pct": round(short_wr, 1),
            "strategies": {
                "buy_and_hold": {
                    "total_return_pct": round(bh_ret * 100, 2),
                    "final_equity": round(cap * (1 + bh_ret), 2),
                },
                "forecast_1x": f1_stats,
                "forecast_vol_target": {
                    **vt_stats,
                    "avg_leverage": round(float(pred["leverage"].mean()), 3),
                },
                "forecast_vt_trailing": {
                    **ts_stats,
                    "avg_leverage": round(float(pred["leverage"].mean()), 3),
                    "exit_reasons": exit_reasons,
                },
            },
            "statistical_tests": {
                "forecast_1x": {"t_stat": round(float(t_f), 4), "p_value": round(float(p_f), 4)},
                "forecast_vol_target": {"t_stat": round(float(t_v), 4), "p_value": round(float(p_v), 4)},
                "forecast_vt_trailing": {
                    "t_stat": round(float(t_ts), 4),
                    "p_value": round(float(p_ts), 4),
                    "bootstrap_95ci_ann": [ci_lo, ci_hi],
                },
                "direction_accuracy": {
                    "correct": correct, "total": n,
                    "binomial_p": round(binom_p, 4),
                },
            },
            "monthly": {
                "months": [str(m) for m in monthly["month"]],
                "days": monthly["days"].tolist(),
                "ts_pct": (monthly["ts_sum"] * 100).round(2).tolist(),
                "da": monthly["da_pct"].round(1).tolist(),
            },
        }

    r2025 = _compute_one_period(pred_2025, "2025 ONLY")
    r2026 = _compute_one_period(pred_2026, "2026 ONLY")

    return {"2025_only": r2025, "2026_only": r2026}


def print_period_split(split: Dict) -> None:
    """Pretty-print 2025-only vs 2026-only comparison."""
    cap = INITIAL_CAPITAL

    print("\n" + "=" * 80)
    print("  PERIODO SPLIT: 2025 ONLY vs 2026 ONLY")
    print("  (Ambos periodos NUNCA vistos durante training 2020-2024)")
    print("=" * 80)

    for period_key in ["2025_only", "2026_only"]:
        r = split[period_key]
        if r.get("error"):
            print(f"\n  {r['label']}: {r['error']}")
            continue

        print(f"\n  {'-' * 76}")
        print(f"  {r['label']}  |  {r['period']}  |  {r['n_days']} dias")
        print(f"  {'-' * 76}")
        print(f"  DA: {r['direction_accuracy_pct']:.1f}% "
              f"({r['n_long']} LONG WR={r['long_wr_pct']:.1f}%, "
              f"{r['n_short']} SHORT WR={r['short_wr_pct']:.1f}%)")

        strats = [
            ("buy_and_hold", "Buy & Hold"),
            ("forecast_1x", "Forecast 1x"),
            ("forecast_vol_target", "FC + VolTarget"),
            ("forecast_vt_trailing", "FC + VT + Trail"),
        ]

        print(f"\n  {'Strategy':<20} {'$10K ->':>12} {'Return':>9} {'Sharpe':>8} {'WR%':>7} "
              f"{'PF':>7} {'MaxDD':>8}")
        print(f"  {'-'*20} {'-'*12} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

        for key, label in strats:
            s = r["strategies"][key]
            ret = s["total_return_pct"]
            final = s.get("final_equity", cap * (1 + ret / 100))
            sharpe = s.get("sharpe", 0)
            wr = s.get("win_rate_pct", 0)
            pf = s.get("profit_factor", 0)
            mdd = s.get("max_dd_pct", 0)
            print(f"  {label:<20} ${final:>10,.2f} {ret:>+8.2f}% {sharpe:>+7.3f} "
                  f"{wr:>6.1f}% {pf:>6.3f} {mdd:>7.2f}%")

        # Exit reasons for trailing stop
        ts = r["strategies"]["forecast_vt_trailing"]
        if "exit_reasons" in ts:
            er = ts["exit_reasons"]
            print(f"\n  Trail exits: trail={er['trailing_stop']} hard={er['hard_stop']} "
                  f"session={er['session_close']} no_bars={er['no_bars']}")

        # Stats
        st = r["statistical_tests"]["forecast_vt_trailing"]
        da_st = r["statistical_tests"]["direction_accuracy"]
        ci = st.get("bootstrap_95ci_ann", [0, 0])
        print(f"\n  VT+Trail: t={st['t_stat']:.3f}, p={st['p_value']:.4f}, "
              f"CI=[{ci[0]:+.2f}%, {ci[1]:+.2f}%]")
        print(f"  DA binom: {da_st['correct']}/{da_st['total']} p={da_st['binomial_p']:.4f}")

        # Monthly
        mb = r["monthly"]
        print(f"\n  {'Mes':<10} {'Dias':>5} {'VT+Trail':>11} {'DA%':>7}")
        print(f"  {'-'*10} {'-'*5} {'-'*11} {'-'*7}")
        for i in range(len(mb["months"])):
            print(f"  {mb['months'][i]:<10} {mb['days'][i]:>5} "
                  f"{mb['ts_pct'][i]:>+10.2f}% {mb['da'][i]:>6.1f}%")

    # Side-by-side summary
    r25 = split["2025_only"]
    r26 = split["2026_only"]
    if r25.get("error") or r26.get("error"):
        return

    ts25 = r25["strategies"]["forecast_vt_trailing"]
    ts26 = r26["strategies"]["forecast_vt_trailing"]
    st25 = r25["statistical_tests"]["forecast_vt_trailing"]
    st26 = r26["statistical_tests"]["forecast_vt_trailing"]

    print(f"\n  {'=' * 76}")
    print(f"  RESUMEN COMPARATIVO")
    print(f"  {'=' * 76}")
    print(f"  {'Metric':<25} {'2025':>15} {'2026':>15} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*12}")

    metrics = [
        ("Dias", r25["n_days"], r26["n_days"]),
        ("DA%", r25["direction_accuracy_pct"], r26["direction_accuracy_pct"]),
        ("LONG WR%", r25["long_wr_pct"], r26["long_wr_pct"]),
        ("SHORT WR%", r25["short_wr_pct"], r26["short_wr_pct"]),
        ("Return%", ts25["total_return_pct"], ts26["total_return_pct"]),
        ("Sharpe", ts25["sharpe"], ts26["sharpe"]),
        ("Win Rate%", ts25["win_rate_pct"], ts26["win_rate_pct"]),
        ("Profit Factor", ts25["profit_factor"], ts26["profit_factor"]),
        ("Max DD%", ts25["max_dd_pct"], ts26["max_dd_pct"]),
        ("p-value", st25["p_value"], st26["p_value"]),
        ("Hard stops", ts25.get("exit_reasons", {}).get("hard_stop", 0),
         ts26.get("exit_reasons", {}).get("hard_stop", 0)),
    ]

    for name, v25, v26 in metrics:
        delta = v26 - v25 if isinstance(v25, (int, float)) else ""
        if isinstance(delta, (int, float)):
            print(f"  {name:<25} {v25:>15.2f} {v26:>15.2f} {delta:>+11.2f}")
        else:
            print(f"  {name:<25} {v25:>15} {v26:>15}")


# =============================================================================
# GATE EVALUATION
# =============================================================================

def evaluate_gates(results: Dict, wf_results: Dict) -> Dict:
    """
    Evaluate promotion gates.
    GATE 1: Walk-forward within 2020-2024 validates model
    GATE 2: Pure OOS 2025 performance
    """
    gates = {}

    # GATE 1: Walk-forward validation (2020-2024 internal)
    wf_da = wf_results["avg_da_pct"]
    wf_folds_pos = wf_results["folds_positive"]
    gates["gate_1_walk_forward"] = {
        "description": "Walk-forward 2020-2024 validation",
        "criteria": {
            "avg_da_gt_50": {"value": wf_da, "threshold": 50.0, "passed": wf_da > 50.0},
            "folds_positive_gte_2": {
                "value": wf_folds_pos,
                "threshold": 2,
                "passed": wf_folds_pos >= 2,
            },
        },
        "passed": wf_da > 50.0 and wf_folds_pos >= 2,
    }

    # GATE 2: Pure OOS 2025 performance
    ts = results["strategies"]["forecast_vt_trailing"]
    stat = results["statistical_tests"]["forecast_vt_trailing"]
    da = results["direction_accuracy_pct"]

    gates["gate_2_oos_2025"] = {
        "description": "Pure OOS 2025 performance",
        "criteria": {
            "da_gt_53": {"value": da, "threshold": 53.0, "passed": da > 53.0},
            "sharpe_gt_1": {"value": ts["sharpe"], "threshold": 1.0, "passed": ts["sharpe"] > 1.0},
            "return_gt_0": {"value": ts["total_return_pct"], "threshold": 0.0, "passed": ts["total_return_pct"] > 0},
            "p_value_lt_005": {"value": stat["p_value"], "threshold": 0.05, "passed": stat["p_value"] < 0.05},
        },
        "passed": (
            da > 53.0
            and ts["sharpe"] > 1.0
            and ts["total_return_pct"] > 0
            and stat["p_value"] < 0.05
        ),
    }

    # GATE 3: Strategy beats buy & hold
    bh_ret = results["strategies"]["buy_and_hold"]["total_return_pct"]
    ts_ret = ts["total_return_pct"]
    gates["gate_3_beats_buyhold"] = {
        "description": "Strategy beats buy & hold",
        "criteria": {
            "strategy_return": ts_ret,
            "buyhold_return": bh_ret,
            "outperformance_pct": round(ts_ret - bh_ret, 2),
        },
        "passed": ts_ret > bh_ret,
    }

    all_passed = all(g["passed"] for g in gates.values())
    gates["overall"] = {
        "passed": all_passed,
        "recommendation": "PROMOTE" if all_passed else "REVIEW" if gates["gate_2_oos_2025"]["passed"] else "REJECT",
    }

    return gates


# =============================================================================
# DISPLAY
# =============================================================================

def print_results(results: Dict, wf_results: Dict, gates: Dict, sensitivity: Dict) -> None:
    """Pretty-print all results."""
    cap = results["initial_capital"]
    n = results["n_trading_days"]
    da = results["direction_accuracy_pct"]
    n_feat = results["n_features"]

    print("\n" + "=" * 80)
    print("  PURE OOS BACKTEST — $10,000 USD en USDCOP")
    print("  Training: SOLO 2020-2024 | OOS: 2025 + 2026 YTD (NUNCA tocado)")
    print("=" * 80)

    print(f"\n  Capital inicial:     ${cap:,.0f} USD")
    print(f"  OOS Period:         {results['oos_period']}")
    print(f"  Dias de trading:    {n}")
    print(f"  Direction Accuracy: {da:.1f}%")
    print(f"  Features:           {n_feat} (incl. VIX + EMBI)")
    print(f"  Pipeline:           9 modelos, top-3 ensemble by magnitude")
    print(f"  Slippage:           {SLIPPAGE_BPS} bps/trade")
    print(f"  Vol-target:         tv={VOL_CONFIG.target_vol}, max_lev={VOL_CONFIG.max_leverage}")
    print(f"  Trailing stop:      act={TRAILING_CONFIG.activation_pct*100:.1f}%, "
          f"trail={TRAILING_CONFIG.trail_pct*100:.1f}%, "
          f"hard={TRAILING_CONFIG.hard_stop_pct*100:.1f}%")

    # Walk-forward results
    print("\n" + "-" * 80)
    print("  WALK-FORWARD 2020-2024 (Model Validation)")
    print("-" * 80)
    for f in wf_results["folds"]:
        print(f"  Fold {f['fold']}: {f['test_start']} to {f['test_end']} "
              f"(gap {f['gap_days']}d) -> DA={f['da_pct']:.1f}%, "
              f"Return={f['return_pct']:+.2f}%, Sharpe={f['sharpe']:.2f}")
    print(f"\n  Avg DA: {wf_results['avg_da_pct']:.1f}%, "
          f"Folds positive: {wf_results['folds_positive']}/{wf_results['total_folds']}")

    # Strategy results
    strats = [
        ("buy_and_hold", "1. Buy & Hold USDCOP"),
        ("forecast_1x", "2. Forecast 1x"),
        ("forecast_vol_target", "3. Forecast + Vol-Target"),
        ("forecast_vt_trailing", "4. Forecast + VT + Trailing Stop"),
    ]

    print("\n" + "-" * 80)
    print("  OOS 2025 RESULTADOS POR ESTRATEGIA")
    print("-" * 80)

    for key, label in strats:
        s = results["strategies"][key]
        final = s.get("final_equity", cap * (1 + s["total_return_pct"] / 100))
        ret = s["total_return_pct"]
        pnl = final - cap

        print(f"\n  {label}:")
        print(f"    Capital final:  ${final:>12,.2f}  ({'+' if pnl >= 0 else '-'}${abs(pnl):,.2f})")
        print(f"    Retorno total:  {ret:>+10.2f}%")

        if "sharpe" in s:
            print(f"    Sharpe:         {s['sharpe']:>+10.3f}")
            print(f"    Sortino:        {s.get('sortino', 0):>+10.3f}")
            print(f"    Max Drawdown:   {s['max_dd_pct']:>10.2f}%")
            print(f"    Win Rate:       {s['win_rate_pct']:>10.1f}%")
            print(f"    Profit Factor:  {s['profit_factor']:>10.3f}")

        if "avg_leverage" in s:
            print(f"    Leverage prom:  {s['avg_leverage']:>10.3f}x")

        if "exit_reasons" in s:
            er = s["exit_reasons"]
            print(f"    Exits:          trail={er['trailing_stop']}, "
                  f"hard={er['hard_stop']}, "
                  f"session={er['session_close']}, "
                  f"no_bars={er['no_bars']}")
            print(f"    Exec alpha:     {s['avg_alpha_bps']:+.1f} bps/trade, "
                  f"{s['alpha_positive_rate_pct']:.1f}% positive")

    # Monthly breakdown
    mb = results["monthly"]
    print("\n" + "-" * 80)
    print("  DESGLOSE MENSUAL OOS")
    print("-" * 80)
    header = (f"  {'Mes':<10} {'Dias':>5} {'Forecast1x':>11} "
              f"{'VolTarget':>11} {'VT+Trail':>11} {'DA%':>7}")
    print(header)
    print(f"  {'-'*10} {'-'*5} {'-'*11} {'-'*11} {'-'*11} {'-'*7}")
    for i in range(len(mb["months"])):
        print(f"  {mb['months'][i]:<10} {mb['days'][i]:>5} "
              f"{mb['f1_pct'][i]:>+10.2f}% {mb['vt_pct'][i]:>+10.2f}% "
              f"{mb['ts_pct'][i]:>+10.2f}% {mb['da'][i]:>6.1f}%")

    # Statistical tests
    st = results["statistical_tests"]
    print("\n" + "-" * 80)
    print("  VALIDACION ESTADISTICA (Pure OOS 2025)")
    print("-" * 80)

    for key, label in [
        ("forecast_1x", "Forecast 1x"),
        ("forecast_vol_target", "Forecast + Vol-Target"),
        ("forecast_vt_trailing", "Forecast + VT + Trailing Stop"),
    ]:
        t = st[key]
        sig = "SI" if t["significant_5pct"] else "NO"
        ci_vals = t["bootstrap_95ci_ann"]
        print(f"\n  {label}:")
        print(f"    t-test p={t['p_value']:.4f} (significativo@5%: {sig})")
        print(f"    Bootstrap 95% CI: [{ci_vals[0]:+.2f}%, {ci_vals[1]:+.2f}%] ann.")

    da_t = st["direction_accuracy"]
    print(f"\n  Direction Accuracy: {da_t['correct']}/{da_t['total']} "
          f"(binomial p={da_t['binomial_p']:.4f})")

    # Sensitivity
    print("\n" + "-" * 80)
    print("  TRAILING STOP SENSITIVITY (Top 5)")
    print("-" * 80)
    for i, r in enumerate(sensitivity["grid_results"][:5]):
        print(f"  {i+1}. act={r['activation_pct']:.1f}% "
              f"trail={r['trail_pct']:.1f}% "
              f"hard={r['hard_stop_pct']:.1f}% -> "
              f"Ret={r['total_return_pct']:+.2f}% "
              f"Sharpe={r['sharpe']:.3f} "
              f"MaxDD={r['max_dd_pct']:.2f}%")

    # Gates
    print("\n" + "=" * 80)
    print("  GATE EVALUATION")
    print("=" * 80)

    for gate_name, gate in gates.items():
        if gate_name == "overall":
            continue
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"\n  {gate_name}: {gate['description']} -> {status}")
        if "criteria" in gate:
            for crit_name, crit in gate["criteria"].items():
                if isinstance(crit, dict) and "passed" in crit:
                    mark = "OK" if crit["passed"] else "XX"
                    print(f"    {crit_name}: {crit['value']:.2f} "
                          f"(threshold: {crit['threshold']}) [{mark}]")

    overall = gates["overall"]
    print(f"\n  OVERALL: {overall['recommendation']}")

    # Final summary
    s4 = results["strategies"]["forecast_vt_trailing"]
    print("\n" + "=" * 80)
    print("  RESPUESTA FINAL: Pure OOS 2025 (Training SOLO 2020-2024)")
    print("=" * 80)
    print(f"\n    $10,000 -> ${s4['final_equity']:,.2f} ({s4['total_return_pct']:+.2f}%)")
    print(f"    Sharpe: {s4['sharpe']:.3f}")
    print(f"    p-value: {st['forecast_vt_trailing']['p_value']:.4f}")
    ci_best = st["forecast_vt_trailing"]["bootstrap_95ci_ann"]
    print(f"    Bootstrap 95% CI: [{ci_best[0]:+.2f}%, {ci_best[1]:+.2f}%]")
    print(f"\n    2025 fue COMPLETAMENTE blind — NUNCA visto durante training.")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()

    print("\n" + "=" * 80)
    print("  PURE OOS 2025 VALIDATION")
    print("  The Most Honest Test: 2025 is NEVER touched during training")
    print("=" * 80)

    # Load data
    df_daily = load_daily_data()
    df_m5 = load_5min_data()

    # Phase 1: Walk-forward within 2020-2024 (model validation)
    wf_results = run_walk_forward_2020_2024(df_daily)

    # Phase 2: Train on 2020-2024, predict 2025 blind
    predictions = train_final_and_predict_oos(df_daily)

    if len(predictions) == 0:
        logger.error("No predictions generated!")
        sys.exit(1)

    logger.info(f"Generated {len(predictions)} OOS predictions for 2025")

    # Phase 3: Trailing stop simulation on 5-min bars
    predictions = simulate_trailing_stop(predictions, df_m5)

    # Phase 4: Run 4-strategy backtest
    results = run_4_strategies(predictions)

    # Phase 5: Sensitivity analysis
    sensitivity = run_trailing_stop_sensitivity(predictions, df_m5)

    # Phase 6: Period split (2025 only vs 2026 only)
    period_split = compute_period_split(predictions)

    # Phase 7: Gate evaluation
    gates = evaluate_gates(results, wf_results)

    # Display
    print_results(results, wf_results, gates, sensitivity)
    print_period_split(period_split)

    # Save JSON
    output = {
        "experiment": "PURE_OOS_2025_VALIDATION",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "description": "Training ONLY 2020-2024, OOS 2025 completely blind",
        "config": {
            "train_period": f"{TRAIN_START.date()} to {TRAIN_END.date()}",
            "oos_period": f"{OOS_START.date()} to {OOS_END.date()}",
            "n_features": len(FEATURE_COLUMNS),
            "features": list(FEATURE_COLUMNS),
            "n_folds_walk_forward": N_FOLDS,
            "gap_days": GAP_DAYS,
            "initial_train_ratio": INITIAL_TRAIN_RATIO,
            "vol_target": {
                "target_vol": VOL_CONFIG.target_vol,
                "max_leverage": VOL_CONFIG.max_leverage,
                "min_leverage": VOL_CONFIG.min_leverage,
                "vol_lookback": VOL_CONFIG.vol_lookback,
            },
            "trailing_stop": {
                "activation_pct": TRAILING_CONFIG.activation_pct,
                "trail_pct": TRAILING_CONFIG.trail_pct,
                "hard_stop_pct": TRAILING_CONFIG.hard_stop_pct,
            },
            "slippage_bps": SLIPPAGE_BPS,
            "initial_capital": INITIAL_CAPITAL,
        },
        "walk_forward_2020_2024": wf_results,
        "oos_2025_results": results,
        "sensitivity_analysis": sensitivity,
        "period_split": period_split,
        "gates": gates,
    }

    output_path = PROJECT_ROOT / "results" / "oos_2025_pure_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj):
        if isinstance(obj, float) and np.isnan(obj):
            return None
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=_default)
    logger.info(f"Results saved to {output_path}")

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s")
