"""
Backtest: $10K USD Investment in USDCOP Forecasting Strategy (2025)
===================================================================

Evidence-based backtest using the EXACT same pipeline as production:
- Same 21 SSOT features
- Same 9 models (3 linear + 3 boosting + 3 hybrid)
- Same walk-forward methodology (train on pre-2025, predict each day of 2025)
- Same vol-targeting module (target_vol=0.15, max_lev=2.0)
- Same trailing stop tracker (activation=0.2%, trail=0.3%, hard_stop=1.5%)

Compares 4 strategies:
1. Buy & Hold USDCOP
2. Forecast-only (1x fixed leverage, direction from ensemble)
3. Forecast + Vol-Target (dynamic leverage via vol-targeting)
4. Forecast + Vol-Target + Trailing Stop (intraday 5-min monitoring)

@version 2.0.0
@date 2026-02-15
"""

import json
import logging
import sys
import time
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
    activation_pct=0.002,   # 0.20% favorable move to arm
    trail_pct=0.003,        # 0.30% drawback from peak triggers exit
    hard_stop_pct=0.015,    # 1.50% adverse = unconditional exit
)

# 2025 trading period
YEAR = 2025
TRAIN_CUTOFF = pd.Timestamp(f"{YEAR}-01-01")


# =============================================================================
# DATA LOADING
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

    # Load macro (DXY + WTI with T-1 lag)
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
        "VOLT_VIX_USA_D_VIX": "vix_close_lag1",
        "CRSK_SPREAD_EMBI_COL_D_EMBI": "embi_close_lag1",
    }
    df_macro_subset = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_subset.rename(columns=macro_cols, inplace=True)
    df_macro_subset = df_macro_subset.sort_values("date")
    df_macro_subset["dxy_close_lag1"] = df_macro_subset["dxy_close_lag1"].shift(1)
    df_macro_subset["oil_close_lag1"] = df_macro_subset["oil_close_lag1"].shift(1)
    df_macro_subset["vix_close_lag1"] = df_macro_subset["vix_close_lag1"].shift(1)
    df_macro_subset["embi_close_lag1"] = df_macro_subset["embi_close_lag1"].shift(1)

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

    # Filter for USD/COP only
    if "symbol" in df_m5.columns:
        df_m5 = df_m5[df_m5["symbol"] == "USD/COP"].copy()

    df_m5 = df_m5.reset_index()
    if "time" in df_m5.columns:
        df_m5.rename(columns={"time": "timestamp"}, inplace=True)

    df_m5["timestamp"] = pd.to_datetime(df_m5["timestamp"])
    if df_m5["timestamp"].dt.tz is not None:
        df_m5["timestamp"] = df_m5["timestamp"].dt.tz_localize(None)

    # Extract date for matching with daily signals
    df_m5["date"] = df_m5["timestamp"].dt.normalize()

    # Filter 2025 only (need T+1 bars for signals from late 2024 too)
    df_m5 = df_m5[df_m5["date"] >= pd.Timestamp("2024-12-01")].copy()

    df_m5 = df_m5.sort_values("timestamp").reset_index(drop=True)
    logger.info(
        f"  5-min: {len(df_m5)} bars, "
        f"{df_m5['timestamp'].min()} to {df_m5['timestamp'].max()}"
    )

    return df_m5[["timestamp", "date", "open", "high", "low", "close"]].copy()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 19 SSOT features from raw OHLCV + macro."""
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

    # Macro (4) — already in df
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()
    df["vix_close_lag1"] = df["vix_close_lag1"].ffill()
    df["embi_close_lag1"] = df["embi_close_lag1"].ffill()

    return df


# =============================================================================
# MODEL TRAINING + DAILY PREDICTION
# =============================================================================

def train_and_predict_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train 9 models on pre-2025 data, then predict every day in 2025.
    Uses expanding window: re-trains every month to incorporate new data.
    """
    feature_cols = list(FEATURE_COLUMNS)
    horizon_config = get_horizon_config(1)

    df_2025 = df[df["date"] >= TRAIN_CUTOFF].copy()
    df_pre2025 = df[df["date"] < TRAIN_CUTOFF].copy()

    logger.info(
        f"Training set: {len(df_pre2025)} rows "
        f"(up to {df_pre2025['date'].max().date()})"
    )
    logger.info(
        f"Trading period: {len(df_2025)} rows "
        f"({df_2025['date'].iloc[0].date()} to {df_2025['date'].iloc[-1].date()})"
    )

    boosting_models = {"xgboost_pure", "lightgbm_pure"}
    catboost_models = {"catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    models_to_use = list(MODEL_IDS)
    try:
        ModelFactory.create("ard")
    except Exception:
        models_to_use = [m for m in models_to_use if m != "ard"]
        logger.warning("ARD model not available, using 8 models")

    results = []

    months = sorted(df_2025["date"].dt.to_period("M").unique())
    logger.info(f"Processing {len(months)} months with monthly re-training...")

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        df_train = df[df["date"] < month_start]
        df_month = df_2025[
            (df_2025["date"] >= month_start) & (df_2025["date"] <= month_end)
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

            sorted_models = sorted(
                preds.keys(), key=lambda m: abs(preds[m]), reverse=True
            )
            top3 = sorted_models[:3]
            ensemble_pred = np.mean([preds[m] for m in top3])
            direction = 1 if ensemble_pred > 0 else -1

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
            })

        logger.info(
            f"  Month {month}: {len(df_month)} days, "
            f"{len(trained_models)} models, train={len(df_train)} rows"
        )

    return pd.DataFrame(results)


# =============================================================================
# TRAILING STOP SIMULATION ON 5-MIN BARS
# =============================================================================

def simulate_trailing_stop(
    predictions: pd.DataFrame,
    df_m5: pd.DataFrame,
    df_daily: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each daily signal (day T), simulate trailing stop on T+1 5-min bars.

    Entry: close of day T (from daily predictions).
    Monitor: all 5-min bars on the NEXT trading day (T+1).
    Exit: trailing stop trigger OR session close.

    Returns predictions with trailing stop columns added.
    """
    logger.info("Simulating trailing stop on 5-min bars...")

    # Build a sorted list of unique trading dates in 5-min data
    m5_dates = sorted(df_m5["date"].unique())
    # Build a lookup: given a signal date T, find T+1 trading date
    daily_dates = sorted(df_daily[df_daily["date"] >= TRAIN_CUTOFF]["date"].unique())

    trail_exit_prices = []
    trail_exit_reasons = []
    trail_pnl_pcts = []
    trail_bar_counts = []
    trail_peak_prices = []

    n_activated = 0
    n_hard_stop = 0
    n_session_close = 0
    n_no_bars = 0

    for _, row in predictions.iterrows():
        signal_date = row["date"]
        entry_price = row["close"]
        direction = int(row["direction"])

        # Find next trading day (T+1) in the 5-min data
        next_dates = [d for d in m5_dates if d > signal_date]
        if not next_dates:
            # No T+1 bars available - fall back to hold-to-close
            trail_exit_prices.append(None)
            trail_exit_reasons.append("no_bars")
            trail_pnl_pcts.append(None)
            trail_bar_counts.append(0)
            trail_peak_prices.append(entry_price)
            n_no_bars += 1
            continue

        t1_date = next_dates[0]
        bars_t1 = df_m5[df_m5["date"] == t1_date].sort_values("timestamp")

        if len(bars_t1) == 0:
            trail_exit_prices.append(None)
            trail_exit_reasons.append("no_bars")
            trail_pnl_pcts.append(None)
            trail_bar_counts.append(0)
            trail_peak_prices.append(entry_price)
            n_no_bars += 1
            continue

        # Apply slippage to entry price
        slip = SLIPPAGE_BPS / 10_000
        if direction == 1:
            slipped_entry = entry_price * (1 + slip)
        else:
            slipped_entry = entry_price * (1 - slip)

        # Run trailing stop tracker
        tracker = TrailingStopTracker(
            entry_price=slipped_entry,
            direction=direction,
            config=TRAILING_CONFIG,
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

        # If not triggered, expire at session close
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
        trail_peak_prices.append(tracker.peak_price)

        if tracker.exit_reason == "trailing_stop":
            n_activated += 1
        elif tracker.exit_reason == "hard_stop":
            n_hard_stop += 1
        elif tracker.exit_reason == "session_close":
            n_session_close += 1

    predictions = predictions.copy()
    predictions["trail_exit_price"] = trail_exit_prices
    predictions["trail_exit_reason"] = trail_exit_reasons
    predictions["trail_pnl_unlevered"] = trail_pnl_pcts
    predictions["trail_bar_count"] = trail_bar_counts
    predictions["trail_peak_price"] = trail_peak_prices

    total = len(predictions) - n_no_bars
    logger.info(
        f"  Trailing stop results ({total} trades): "
        f"activated={n_activated} ({n_activated/max(total,1)*100:.1f}%), "
        f"hard_stop={n_hard_stop}, session_close={n_session_close}, "
        f"no_bars={n_no_bars}"
    )

    return predictions


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def compute_stats(rets: np.ndarray) -> Dict:
    """Compute strategy statistics from daily returns array."""
    if len(rets) == 0:
        return {}
    total_ret = np.prod(1 + rets) - 1
    ann_ret = (1 + total_ret) ** (252 / len(rets)) - 1
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
        "total_return_pct": total_ret * 100,
        "annualized_return_pct": ann_ret * 100,
        "volatility_pct": vol * 100,
        "sharpe": sharpe,
        "max_dd_pct": max_dd * 100,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "trading_days": len(rets),
    }


def run_backtest(predictions: pd.DataFrame) -> Dict:
    """
    Simulate 4 strategies on $10K initial capital.
    """
    n_days = len(predictions)
    slippage_mult = SLIPPAGE_BPS / 10_000
    cap = INITIAL_CAPITAL

    # --- Strategy 1: Buy & Hold USDCOP ---
    first_close = predictions.iloc[0]["close"]
    last_close = predictions.iloc[-1]["close"]
    bh_return = (last_close - first_close) / first_close
    bh_final = cap * (1 + bh_return)

    # --- Strategy 2: Forecast 1x (direction only, no leverage) ---
    f1_equity = cap
    f1_rets = []
    for _, row in predictions.iterrows():
        daily_ret = row["direction"] * (np.exp(row["actual_return"]) - 1)
        daily_ret -= slippage_mult
        f1_equity *= (1 + daily_ret)
        f1_rets.append(daily_ret)
    f1_rets = np.array(f1_rets)

    # --- Strategy 3: Forecast + Vol-Target (hold-to-close) ---
    vt_equity = cap
    vt_rets = []
    for _, row in predictions.iterrows():
        lev = row["leverage"]
        daily_ret = row["direction"] * lev * (np.exp(row["actual_return"]) - 1)
        daily_ret -= slippage_mult * lev
        vt_equity *= (1 + daily_ret)
        vt_rets.append(daily_ret)
    vt_rets = np.array(vt_rets)

    # --- Strategy 4: Forecast + Vol-Target + Trailing Stop ---
    ts_equity = cap
    ts_rets = []
    ts_exit_reasons = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}

    for _, row in predictions.iterrows():
        lev = row["leverage"]
        trail_pnl = row.get("trail_pnl_unlevered")
        reason = row.get("trail_exit_reason", "no_bars")

        if trail_pnl is not None and not np.isnan(trail_pnl):
            # Use trailing stop PnL (already includes entry/exit slippage)
            daily_ret = trail_pnl * lev
        else:
            # Fallback: same as vol-target (hold-to-close)
            daily_ret = row["direction"] * lev * (np.exp(row["actual_return"]) - 1)
            daily_ret -= slippage_mult * lev

        ts_equity *= (1 + daily_ret)
        ts_rets.append(daily_ret)

        if reason in ts_exit_reasons:
            ts_exit_reasons[reason] += 1

    ts_rets = np.array(ts_rets)

    # --- Direction Accuracy ---
    actual_rets = predictions["actual_return"].values
    correct_dirs = int(np.sum(predictions["direction"].values == np.sign(actual_rets)))
    da = correct_dirs / n_days * 100

    # --- Stats ---
    f1_stats = compute_stats(f1_rets)
    vt_stats = compute_stats(vt_rets)
    ts_stats = compute_stats(ts_rets)

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

    # --- Statistical validation ---
    t_stat_f, p_f = stats.ttest_1samp(f1_rets, 0)
    t_stat_v, p_v = stats.ttest_1samp(vt_rets, 0)
    t_stat_ts, p_ts = stats.ttest_1samp(ts_rets, 0)

    # Paired t-test: trailing stop vs hold-to-close (vol-target)
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
        return (
            float(np.percentile(boot, 2.5) * 252 * 100),
            float(np.percentile(boot, 97.5) * 252 * 100),
        )

    ci_f = ci(boot_f)
    ci_v = ci(boot_v)
    ci_ts = ci(boot_ts)

    binom_p = stats.binomtest(correct_dirs, n_days, 0.5).pvalue

    # Trailing stop alpha: average per-trade improvement
    has_trail = pred_m["trail_pnl_unlevered"].notna()
    if has_trail.any():
        # Compare trail PnL vs hold-to-close PnL (both unlevered)
        trail_pnls = pred_m.loc[has_trail, "trail_pnl_unlevered"].values
        hold_pnls = pred_m.loc[has_trail].apply(
            lambda r: r["direction"] * (np.exp(r["actual_return"]) - 1) - slippage_mult,
            axis=1,
        ).values
        alpha_per_trade = trail_pnls - hold_pnls
        avg_alpha = float(np.mean(alpha_per_trade) * 100)
        alpha_positive_rate = float(np.sum(alpha_per_trade > 0) / len(alpha_per_trade) * 100)
    else:
        avg_alpha = 0.0
        alpha_positive_rate = 0.0

    return {
        "initial_capital": cap,
        "n_trading_days": n_days,
        "direction_accuracy_pct": da,
        "strategies": {
            "buy_and_hold": {
                "final_equity": bh_final,
                "total_return_pct": bh_return * 100,
            },
            "forecast_1x": {
                "final_equity": f1_equity,
                **f1_stats,
            },
            "forecast_vol_target": {
                "final_equity": vt_equity,
                **vt_stats,
                "avg_leverage": float(predictions["leverage"].mean()),
            },
            "forecast_vt_trailing": {
                "final_equity": ts_equity,
                **ts_stats,
                "avg_leverage": float(predictions["leverage"].mean()),
                "exit_reasons": ts_exit_reasons,
                "avg_alpha_bps": avg_alpha * 100,
                "alpha_positive_rate_pct": alpha_positive_rate,
            },
        },
        "statistical_tests": {
            "forecast_1x": {
                "t_stat": float(t_stat_f), "p_value": float(p_f),
                "bootstrap_95ci_ann": list(ci_f),
                "significant": p_f < 0.05,
            },
            "forecast_vol_target": {
                "t_stat": float(t_stat_v), "p_value": float(p_v),
                "bootstrap_95ci_ann": list(ci_v),
                "significant": p_v < 0.05,
            },
            "forecast_vt_trailing": {
                "t_stat": float(t_stat_ts), "p_value": float(p_ts),
                "bootstrap_95ci_ann": list(ci_ts),
                "significant": p_ts < 0.05,
            },
            "paired_trail_vs_hold": {
                "t_stat": float(t_paired), "p_value": float(p_paired),
                "significant": p_paired < 0.05,
                "description": "Trail better than hold-to-close?",
            },
            "direction_accuracy": {
                "correct": correct_dirs, "total": n_days,
                "binomial_p": float(binom_p), "significant": binom_p < 0.05,
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
# DISPLAY
# =============================================================================

def print_results(results: Dict) -> None:
    """Pretty-print backtest results."""
    cap = results["initial_capital"]
    n = results["n_trading_days"]
    da = results["direction_accuracy_pct"]

    print("\n" + "=" * 80)
    print("  BACKTEST: $10,000 USD en USDCOP — 4 Estrategias (2025)")
    print("=" * 80)
    print(f"\n  Capital inicial:     ${cap:,.0f} USD")
    print(f"  Dias de trading:    {n}")
    print(f"  Direction Accuracy: {da:.1f}%")
    print(f"  Pipeline:           9 modelos, 19 features, H=1, top-3 ensemble")
    print(f"  Slippage:           {SLIPPAGE_BPS} bps/trade (MEXC maker 0%)")
    print(f"  Vol-target:         tv={VOL_CONFIG.target_vol}, max_lev={VOL_CONFIG.max_leverage}")
    print(f"  Trailing stop:      act={TRAILING_CONFIG.activation_pct*100:.1f}%, "
          f"trail={TRAILING_CONFIG.trail_pct*100:.1f}%, "
          f"hard={TRAILING_CONFIG.hard_stop_pct*100:.1f}%")

    strats = [
        ("buy_and_hold", "1. Buy & Hold USDCOP"),
        ("forecast_1x", "2. Forecast 1x"),
        ("forecast_vol_target", "3. Forecast + Vol-Target"),
        ("forecast_vt_trailing", "4. Forecast + VT + Trailing Stop"),
    ]

    print("\n" + "-" * 80)
    print("  RESULTADOS POR ESTRATEGIA")
    print("-" * 80)

    for key, label in strats:
        s = results["strategies"][key]
        final = s["final_equity"]
        ret = s.get("total_return_pct", (final / cap - 1) * 100)
        pnl = final - cap
        sign = "+" if pnl >= 0 else "-"

        print(f"\n  {label}:")
        print(f"    Capital final:  ${final:>12,.2f}  ({sign}${abs(pnl):,.2f})")
        print(f"    Retorno total:  {ret:>+10.2f}%")

        if "sharpe" in s:
            print(f"    Sharpe:         {s['sharpe']:>+10.3f}")
            print(f"    Max Drawdown:   {s['max_dd_pct']:>10.2f}%")
            print(f"    Win Rate:       {s['win_rate_pct']:>10.1f}%")
            print(f"    Profit Factor:  {s['profit_factor']:>10.3f}")

        if "avg_leverage" in s:
            print(f"    Leverage prom:  {s['avg_leverage']:>10.2f}x")

        if "exit_reasons" in s:
            er = s["exit_reasons"]
            print(f"    Exits:          trail={er['trailing_stop']}, "
                  f"hard={er['hard_stop']}, "
                  f"session={er['session_close']}, "
                  f"no_bars={er['no_bars']}")
            print(f"    Exec alpha:     {s['avg_alpha_bps']:+.1f} bps/trade avg, "
                  f"{s['alpha_positive_rate_pct']:.1f}% positive")

    # Monthly breakdown
    mb = results["monthly"]
    print("\n" + "-" * 80)
    print("  DESGLOSE MENSUAL")
    print("-" * 80)
    header = (f"  {'Mes':<10} {'Dias':>5} {'Forecast1x':>11} "
              f"{'VolTarget':>11} {'VT+Trail':>11} {'DA%':>7}")
    print(header)
    print(f"  {'-'*10} {'-'*5} {'-'*11} {'-'*11} {'-'*11} {'-'*7}")

    for i in range(len(mb["months"])):
        m = mb["months"][i]
        d = mb["days"][i]
        f1 = mb["f1_pct"][i]
        vt = mb["vt_pct"][i]
        ts = mb["ts_pct"][i]
        da_m = mb["da"][i]
        print(f"  {m:<10} {d:>5} {f1:>+10.2f}% {vt:>+10.2f}% {ts:>+10.2f}% {da_m:>6.1f}%")

    # Stats
    st = results["statistical_tests"]
    print("\n" + "-" * 80)
    print("  VALIDACION ESTADISTICA")
    print("-" * 80)

    for key, label in [
        ("forecast_1x", "Forecast 1x"),
        ("forecast_vol_target", "Forecast + Vol-Target"),
        ("forecast_vt_trailing", "Forecast + VT + Trailing Stop"),
    ]:
        t = st[key]
        sig = "SI" if t["significant"] else "NO"
        ci_vals = t["bootstrap_95ci_ann"]
        print(f"\n  {label}:")
        print(f"    t-test p={t['p_value']:.4f} (significativo: {sig})")
        print(f"    Bootstrap 95% CI: [{ci_vals[0]:+.2f}%, {ci_vals[1]:+.2f}%] ann.")

    pt = st["paired_trail_vs_hold"]
    sig_p = "SI" if pt["significant"] else "NO"
    print(f"\n  Trail vs Hold (paired t-test):")
    print(f"    p={pt['p_value']:.4f} (trail mejor que hold: {sig_p})")

    da_t = st["direction_accuracy"]
    sig_da = "SI" if da_t["significant"] else "NO"
    print(f"\n  Direction Accuracy: {da_t['correct']}/{da_t['total']} "
          f"(binomial p={da_t['binomial_p']:.4f}, sig: {sig_da})")

    # Final answer
    s1 = results["strategies"]["buy_and_hold"]
    s2 = results["strategies"]["forecast_1x"]
    s3 = results["strategies"]["forecast_vol_target"]
    s4 = results["strategies"]["forecast_vt_trailing"]

    def fmt(s):
        ret = s.get("total_return_pct", (s["final_equity"] / cap - 1) * 100)
        return f"${s['final_equity']:>12,.2f}  ({ret:+.2f}%)"

    print("\n" + "=" * 80)
    print("  RESPUESTA FINAL: $10,000 USD invertidos el 1 de enero de 2025")
    print("=" * 80)
    print(f"\n    1. Buy & Hold USDCOP:            {fmt(s1)}")
    print(f"    2. Forecast 1x:                  {fmt(s2)}")
    print(f"    3. Forecast + Vol-Target:         {fmt(s3)}")
    print(f"    4. Forecast + VT + Trailing Stop: {fmt(s4)}")

    best = max(
        [("Forecast 1x", s2), ("Forecast + VT", s3), ("Forecast + VT + Trail", s4)],
        key=lambda x: x[1]["final_equity"],
    )
    print(f"\n  Mejor estrategia: {best[0]} "
          f"(+${best[1]['final_equity'] - cap:,.2f})")
    print(f"\n  Nota: Walk-forward expandible (re-entrena mensual),")
    print(f"  trailing stop sobre barras 5-min reales,")
    print(f"  {SLIPPAGE_BPS} bps slippage, 0% comision MEXC maker.")
    print("=" * 80)


# =============================================================================
# DASHBOARD TRADE EXPORT
# =============================================================================

def export_trades_for_dashboard(predictions: pd.DataFrame, results: Dict) -> None:
    """
    Export trade-level JSON files for dashboard replay.

    Produces 4 files in usdcop-trading-dashboard/public/data/forecast-trades/:
      buy_and_hold.json, forecast_1x.json, forecast_vol_target.json,
      forecast_vt_trailing.json

    Each file matches the BacktestTrade contract from backtest.contract.ts.
    """
    out_dir = (
        PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "data"
        / "forecast-trades"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    slippage_mult = SLIPPAGE_BPS / 10_000
    cap = INITIAL_CAPITAL
    dates = predictions["date"].values
    start_date = str(pd.Timestamp(dates[0]).date())
    end_date = str(pd.Timestamp(dates[-1]).date())

    def _make_timestamp(date_val, hour: int = 17, minute: int = 55) -> str:
        """Create ISO timestamp from a date at given hour:minute UTC."""
        ts = pd.Timestamp(date_val)
        return ts.strftime(f"%Y-%m-%dT{hour:02d}:{minute:02d}:00Z")

    def _build_summary(trades: List[Dict]) -> Dict:
        """Build BacktestSummary from list of trade dicts."""
        wins = [t for t in trades if t["pnl_usd"] > 0]
        losses = [t for t in trades if t["pnl_usd"] <= 0]
        total_pnl = sum(t["pnl_usd"] for t in trades)
        wr = len(wins) / len(trades) * 100 if trades else 0

        # Equity curve for max drawdown
        equity_curve = [cap]
        running = cap
        for t in trades:
            running += t["pnl_usd"]
            equity_curve.append(running)
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe from daily returns
        daily_rets = [t["pnl_usd"] / t["equity_at_entry"] for t in trades if t["equity_at_entry"] > 0]
        if len(daily_rets) > 1:
            mean_r = float(np.mean(daily_rets))
            std_r = float(np.std(daily_rets, ddof=1))
            sharpe = mean_r / std_r * np.sqrt(252) if std_r > 0 else 0
        else:
            sharpe = 0

        # Avg duration
        avg_dur = (
            sum(t["duration_minutes"] for t in trades) / len(trades)
            if trades else 0
        )

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(wr, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_pnl / cap * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 3),
            "avg_trade_duration_minutes": round(avg_dur),
        }

    # ── Strategy 1: Buy & Hold ─────────────────────────────────────────
    bh_trades: List[Dict] = []
    bh_equity = cap
    first_close = float(predictions.iloc[0]["close"])

    # Split into monthly trades for visible equity curve shape
    pred_copy = predictions.copy()
    pred_copy["month"] = pred_copy["date"].dt.to_period("M")
    months = pred_copy.groupby("month")

    for month_idx, (month, grp) in enumerate(months):
        entry_date = grp.iloc[0]["date"]
        exit_date = grp.iloc[-1]["date"]
        entry_close = float(grp.iloc[0]["close"])
        exit_close = float(grp.iloc[-1]["close"])
        monthly_ret = (exit_close - entry_close) / entry_close
        pnl_usd = bh_equity * monthly_ret
        eq_before = bh_equity
        bh_equity += pnl_usd

        bh_trades.append({
            "trade_id": month_idx + 1,
            "model_id": "fc_buy_hold",
            "timestamp": _make_timestamp(entry_date),
            "entry_time": _make_timestamp(entry_date),
            "exit_time": _make_timestamp(exit_date),
            "side": "LONG",
            "entry_price": round(entry_close, 2),
            "exit_price": round(exit_close, 2),
            "pnl": round(pnl_usd, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_percent": round(monthly_ret * 100, 3),
            "pnl_pct": round(monthly_ret * 100, 3),
            "status": "closed",
            "duration_minutes": int(len(grp)) * 1440,
            "exit_reason": "month_end",
            "equity_at_entry": round(eq_before, 2),
            "equity_at_exit": round(bh_equity, 2),
            "entry_confidence": 50.0,
            "exit_confidence": 50.0,
        })

    _save_strategy_file(out_dir / "buy_and_hold.json", {
        "model_id": "fc_buy_hold",
        "strategy_name": "Buy & Hold USDCOP",
        "initial_capital": cap,
        "date_range": {"start": start_date, "end": end_date},
        "trades": bh_trades,
        "summary": _build_summary(bh_trades),
    })

    # ── Strategy 2: Forecast 1x ────────────────────────────────────────
    f1_trades: List[Dict] = []
    f1_equity = cap

    for i, (_, row) in enumerate(predictions.iterrows()):
        direction = int(row["direction"])
        daily_ret = direction * (np.exp(row["actual_return"]) - 1) - slippage_mult
        pnl_usd = f1_equity * daily_ret
        eq_before = f1_equity
        f1_equity += pnl_usd
        entry_close = float(row["close"])
        # Approximate exit price from return
        exit_close = entry_close * np.exp(float(row["actual_return"]))
        confidence = abs(float(row["ensemble_pred"])) * 100

        f1_trades.append({
            "trade_id": i + 1,
            "model_id": "fc_forecast_1x",
            "timestamp": _make_timestamp(row["date"]),
            "entry_time": _make_timestamp(row["date"]),
            "exit_time": _make_timestamp(row["date"], 17, 55),
            "side": "LONG" if direction == 1 else "SHORT",
            "entry_price": round(entry_close, 2),
            "exit_price": round(exit_close, 2),
            "pnl": round(pnl_usd, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_percent": round(daily_ret * 100, 3),
            "pnl_pct": round(daily_ret * 100, 3),
            "status": "closed",
            "duration_minutes": 1440,
            "exit_reason": "session_close",
            "equity_at_entry": round(eq_before, 2),
            "equity_at_exit": round(f1_equity, 2),
            "entry_confidence": round(min(confidence, 99.9), 1),
            "exit_confidence": None,
        })

    _save_strategy_file(out_dir / "forecast_1x.json", {
        "model_id": "fc_forecast_1x",
        "strategy_name": "Forecast 1x (Direction Only)",
        "initial_capital": cap,
        "date_range": {"start": start_date, "end": end_date},
        "trades": f1_trades,
        "summary": _build_summary(f1_trades),
    })

    # ── Strategy 3: Forecast + Vol-Target ──────────────────────────────
    vt_trades: List[Dict] = []
    vt_equity = cap

    for i, (_, row) in enumerate(predictions.iterrows()):
        direction = int(row["direction"])
        lev = float(row["leverage"])
        daily_ret = direction * lev * (np.exp(row["actual_return"]) - 1) - slippage_mult * lev
        pnl_usd = vt_equity * daily_ret
        eq_before = vt_equity
        vt_equity += pnl_usd
        entry_close = float(row["close"])
        exit_close = entry_close * np.exp(float(row["actual_return"]))
        confidence = abs(float(row["ensemble_pred"])) * 100

        vt_trades.append({
            "trade_id": i + 1,
            "model_id": "fc_forecast_vt",
            "timestamp": _make_timestamp(row["date"]),
            "entry_time": _make_timestamp(row["date"]),
            "exit_time": _make_timestamp(row["date"], 17, 55),
            "side": "LONG" if direction == 1 else "SHORT",
            "entry_price": round(entry_close, 2),
            "exit_price": round(exit_close, 2),
            "pnl": round(pnl_usd, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_percent": round(daily_ret * 100, 3),
            "pnl_pct": round(daily_ret * 100, 3),
            "status": "closed",
            "duration_minutes": 1440,
            "exit_reason": "session_close",
            "equity_at_entry": round(eq_before, 2),
            "equity_at_exit": round(vt_equity, 2),
            "entry_confidence": round(min(confidence, 99.9), 1),
            "exit_confidence": None,
        })

    _save_strategy_file(out_dir / "forecast_vol_target.json", {
        "model_id": "fc_forecast_vt",
        "strategy_name": "Forecast + Vol-Target",
        "initial_capital": cap,
        "date_range": {"start": start_date, "end": end_date},
        "trades": vt_trades,
        "summary": _build_summary(vt_trades),
    })

    # ── Strategy 4: Forecast + VT + Trailing Stop ──────────────────────
    ts_trades: List[Dict] = []
    ts_equity = cap

    for i, (_, row) in enumerate(predictions.iterrows()):
        direction = int(row["direction"])
        lev = float(row["leverage"])
        trail_pnl = row.get("trail_pnl_unlevered")
        reason = row.get("trail_exit_reason", "no_bars")
        bar_count = int(row.get("trail_bar_count", 0))

        if trail_pnl is not None and not np.isnan(trail_pnl):
            daily_ret = float(trail_pnl) * lev
        else:
            daily_ret = direction * lev * (np.exp(row["actual_return"]) - 1)
            daily_ret -= slippage_mult * lev
            reason = "session_close"

        pnl_usd = ts_equity * daily_ret
        eq_before = ts_equity
        ts_equity += pnl_usd
        entry_close = float(row["close"])

        trail_exit = row.get("trail_exit_price")
        if trail_exit is not None and not np.isnan(trail_exit):
            exit_close = float(trail_exit)
        else:
            exit_close = entry_close * np.exp(float(row["actual_return"]))

        confidence = abs(float(row["ensemble_pred"])) * 100
        dur_minutes = bar_count * 5 if bar_count > 0 else 1440

        ts_trades.append({
            "trade_id": i + 1,
            "model_id": "fc_forecast_vt_trail",
            "timestamp": _make_timestamp(row["date"]),
            "entry_time": _make_timestamp(row["date"]),
            "exit_time": _make_timestamp(row["date"], 17, 55),
            "side": "LONG" if direction == 1 else "SHORT",
            "entry_price": round(entry_close, 2),
            "exit_price": round(exit_close, 2),
            "pnl": round(pnl_usd, 2),
            "pnl_usd": round(pnl_usd, 2),
            "pnl_percent": round(daily_ret * 100, 3),
            "pnl_pct": round(daily_ret * 100, 3),
            "status": "closed",
            "duration_minutes": dur_minutes,
            "exit_reason": reason if reason else "session_close",
            "equity_at_entry": round(eq_before, 2),
            "equity_at_exit": round(ts_equity, 2),
            "entry_confidence": round(min(confidence, 99.9), 1),
            "exit_confidence": None,
        })

    _save_strategy_file(out_dir / "forecast_vt_trailing.json", {
        "model_id": "fc_forecast_vt_trail",
        "strategy_name": "Forecast + VT + Trailing Stop",
        "initial_capital": cap,
        "date_range": {"start": start_date, "end": end_date},
        "trades": ts_trades,
        "summary": _build_summary(ts_trades),
    })

    logger.info(
        f"Dashboard trade files exported to {out_dir}/ "
        f"({len(bh_trades)} + {len(f1_trades)} + {len(vt_trades)} + {len(ts_trades)} trades)"
    )


def _save_strategy_file(path: Path, data: Dict) -> None:
    """Save strategy JSON with NaN/None handling."""
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
        return str(obj)

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_default)
    logger.info(f"  Saved {path.name}: {len(data['trades'])} trades")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    t0 = time.time()

    # Load data
    df_daily = load_daily_data()
    df_m5 = load_5min_data()

    # Train models and predict daily
    predictions = train_and_predict_daily(df_daily)

    if len(predictions) == 0:
        logger.error("No predictions generated!")
        sys.exit(1)

    logger.info(f"Generated {len(predictions)} daily predictions for 2025")

    # Simulate trailing stop on 5-min bars
    predictions = simulate_trailing_stop(predictions, df_m5, df_daily)

    # Run backtest
    results = run_backtest(predictions)

    # Display
    print_results(results)

    # Save JSON
    output_path = PROJECT_ROOT / "results" / "backtest_2025_10k_4strat.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        serializable = json.loads(json.dumps(results, default=str))
        json.dump(serializable, f, indent=2)
    logger.info(f"Results saved to {output_path}")

    # Export trade-level JSON for dashboard replay
    export_trades_for_dashboard(predictions, results)

    elapsed = time.time() - t0
    logger.info(f"Total time: {elapsed:.1f}s")
