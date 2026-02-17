"""
Multi-Horizon Backtest Comparison: H=1 (Daily), H=5 (Weekly), H=20 (Monthly)
=============================================================================

Trains on 2020-2024, predicts 2025 pure OOS for each horizon.
For each horizon H:
  - Target: ln(close[t+H] / close[t])
  - Signal: every H trading days (non-overlapping trades)
  - Hold: H days
  - Trailing stop: on 5-min bars over the full H-day holding period
  - Vol-targeting: same config, adjusted for horizon

@version 1.0.0
@date 2026-02-16
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
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

TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
OOS_END = pd.Timestamp("2025-12-31")

HORIZONS = [1, 5, 20]  # Daily, Weekly, Monthly


# =============================================================================
# DATA LOADING
# =============================================================================

def load_daily_data() -> pd.DataFrame:
    """Load daily OHLCV + macro, build 21 features."""
    logger.info("Loading daily OHLCV...")
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path).reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)
    logger.info(f"  OHLCV: {len(df_ohlcv)} rows, {df_ohlcv['date'].min().date()} to {df_ohlcv['date'].max().date()}")

    # Macro
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

    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_sub.sort_values("date"),
        on="date", direction="backward",
    )

    df = _build_features(df)

    # Create targets for all horizons
    for h in HORIZONS:
        df[f"target_{h}d"] = np.log(df["close"].shift(-h) / df["close"])

    # Filter complete rows
    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    df = df[feature_mask].reset_index(drop=True)
    logger.info(f"  After cleanup: {len(df)} rows")
    return df


def load_5min_data() -> pd.DataFrame:
    """Load 5-min bars for trailing stop."""
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
    logger.info(f"  5-min: {len(df_m5)} bars, {df_m5['timestamp'].min()} to {df_m5['timestamp'].max()}")
    return df_m5[["timestamp", "date", "open", "high", "low", "close"]].copy()


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build 21 SSOT features."""
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
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14d"] = 100 - (100 / (1 + rs))
    df["ma_ratio_20d"] = df["close"] / df["close"].rolling(20).mean()
    df["ma_ratio_50d"] = df["close"] / df["close"].rolling(50).mean()
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["is_month_end"] = pd.to_datetime(df["date"]).dt.is_month_end.astype(int)
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()
    df["vix_close_lag1"] = df["vix_close_lag1"].ffill()
    df["embi_close_lag1"] = df["embi_close_lag1"].ffill()
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_model_list() -> List[str]:
    models = list(MODEL_IDS)
    try:
        ModelFactory.create("ard")
    except Exception:
        models = [m for m in models if m != "ard"]
    return models


def get_model_params(model_id: str, horizon_config: dict) -> Optional[dict]:
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
            "verbose": False, "allow_writing_files": False,
        }
    elif model_id in hybrid_models:
        if "catboost" in model_id:
            return {
                "iterations": horizon_config.get("n_estimators", 50),
                "depth": horizon_config.get("max_depth", 3),
                "learning_rate": horizon_config.get("learning_rate", 0.05),
                "verbose": False, "allow_writing_files": False,
            }
        return horizon_config
    else:
        return horizon_config


def train_models(X_train, y_train, models_to_use, horizon_config, horizon):
    """Train all models for a given horizon."""
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
        except Exception as e:
            pass
    return trained, scaler


# =============================================================================
# PREDICT OOS FOR A GIVEN HORIZON
# =============================================================================

def predict_oos_horizon(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Train on 2020-2024, predict 2025 OOS for a specific horizon.
    Monthly re-training. Signal every H days (non-overlapping).
    """
    feature_cols = list(FEATURE_COLUMNS)
    target_col = f"target_{horizon}d"
    horizon_config = get_horizon_config(horizon)
    models_to_use = get_model_list()

    df_train_all = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_oos = df[(df["date"] >= OOS_START) & (df["date"] <= OOS_END)].copy()

    # Filter rows with valid target
    df_train_all = df_train_all[df_train_all[target_col].notna()]
    df_oos = df_oos[df_oos[target_col].notna()]

    logger.info(f"\n  H={horizon}: Train {len(df_train_all)} rows, OOS {len(df_oos)} rows")

    # Monthly re-training
    months = sorted(df_oos["date"].dt.to_period("M").unique())
    all_signals = []

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        if month_idx == 0:
            df_month_train = df_train_all.copy()
        else:
            df_month_train = df[(df["date"] >= TRAIN_START) & (df["date"] < month_start)].copy()
            df_month_train = df_month_train[df_month_train[target_col].notna()]

        df_month_oos = df_oos[(df_oos["date"] >= month_start) & (df_oos["date"] <= month_end)]

        if len(df_month_oos) == 0 or len(df_month_train) < 200:
            continue

        X_train = df_month_train[feature_cols].values.astype(np.float64)
        y_train = df_month_train[target_col].values.astype(np.float64)

        trained, scaler = train_models(X_train, y_train, models_to_use, horizon_config, horizon)

        # Predict each OOS day (signals generated every day, but trades are non-overlapping)
        for _, row in df_month_oos.iterrows():
            X_day = row[feature_cols].values.astype(np.float64).reshape(1, -1)
            X_day_scaled = scaler.transform(X_day)

            preds = {}
            for model_id, model in trained.items():
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

            # Vol-targeting
            idx_list = df.index[df["date"] == row["date"]]
            if len(idx_list) == 0:
                continue
            idx = idx_list[0]
            if idx >= VOL_CONFIG.vol_lookback:
                recent = df.iloc[idx - VOL_CONFIG.vol_lookback:idx]["return_1d"].values
                realized_vol = compute_realized_vol(recent, VOL_CONFIG.vol_lookback)
            else:
                realized_vol = 0.10

            vt_signal = compute_vol_target_signal(
                forecast_direction=direction,
                forecast_return=ensemble_pred,
                realized_vol_21d=realized_vol,
                config=VOL_CONFIG,
                date=str(row["date"].date()),
            )

            all_signals.append({
                "date": row["date"],
                "close": row["close"],
                "actual_return_Hd": row[target_col],
                "ensemble_pred": ensemble_pred,
                "direction": direction,
                "realized_vol": realized_vol,
                "leverage": vt_signal.clipped_leverage,
            })

    return pd.DataFrame(all_signals)


# =============================================================================
# NON-OVERLAPPING TRADE SELECTION
# =============================================================================

def select_non_overlapping_trades(signals: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    From daily signals, select non-overlapping trades every H days.
    First trade on first OOS day, next trade H trading days later, etc.
    """
    if len(signals) == 0:
        return signals

    signals = signals.sort_values("date").reset_index(drop=True)
    selected = [0]  # First trade
    last_trade_idx = 0

    for i in range(1, len(signals)):
        # Count trading days since last trade
        days_since = i - last_trade_idx
        if days_since >= horizon:
            selected.append(i)
            last_trade_idx = i

    result = signals.iloc[selected].copy().reset_index(drop=True)
    logger.info(f"  H={horizon}: {len(result)} non-overlapping trades from {len(signals)} signals")
    return result


# =============================================================================
# MULTI-DAY TRAILING STOP
# =============================================================================

def simulate_trailing_stop_multi_day(
    trades: pd.DataFrame,
    df_m5: pd.DataFrame,
    horizon: int,
    config: TrailingStopConfig = TRAILING_CONFIG,
) -> pd.DataFrame:
    """
    Trailing stop over H trading days of 5-min bars.
    Entry: open of T+1. Exit: trailing stop trigger or last bar of T+H.
    """
    m5_dates = sorted(df_m5["date"].unique())
    slip = SLIPPAGE_BPS / 10_000

    trail_pnl = []
    trail_reasons = []
    trail_bars = []
    counters = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}

    for _, row in trades.iterrows():
        signal_date = row["date"]
        direction = int(row["direction"])

        # Find next H trading days after signal
        future_dates = [d for d in m5_dates if d > signal_date]
        if len(future_dates) == 0:
            trail_pnl.append(None)
            trail_reasons.append("no_bars")
            trail_bars.append(0)
            counters["no_bars"] += 1
            continue

        # Take up to H trading days of 5-min bars
        holding_dates = future_dates[:horizon]
        bars = df_m5[df_m5["date"].isin(holding_dates)].sort_values("timestamp")

        if len(bars) == 0:
            trail_pnl.append(None)
            trail_reasons.append("no_bars")
            trail_bars.append(0)
            counters["no_bars"] += 1
            continue

        # Entry at first bar's open
        entry_price = float(bars.iloc[0]["open"])
        if direction == 1:
            slipped_entry = entry_price * (1 + slip)
        else:
            slipped_entry = entry_price * (1 - slip)

        # Run trailing stop across all H days
        tracker = TrailingStopTracker(
            entry_price=slipped_entry,
            direction=direction,
            config=config,
        )

        for bar_idx, (_, bar) in enumerate(bars.iterrows()):
            state = tracker.update(
                bar_high=float(bar["high"]),
                bar_low=float(bar["low"]),
                bar_close=float(bar["close"]),
                bar_idx=bar_idx,
            )
            if state == TrailingState.TRIGGERED:
                break

        if tracker.state not in (TrailingState.TRIGGERED, TrailingState.EXPIRED):
            last_close = float(bars.iloc[-1]["close"])
            tracker.expire(last_close)

        exit_price = tracker.exit_price
        if exit_price is not None:
            if direction == 1:
                exit_price *= (1 - slip)
            else:
                exit_price *= (1 + slip)

        if exit_price is not None and slipped_entry > 0:
            pnl = direction * (exit_price - slipped_entry) / slipped_entry
        else:
            pnl = None

        trail_pnl.append(pnl)
        reason = tracker.exit_reason or "no_bars"
        trail_reasons.append(reason)
        trail_bars.append(tracker.exit_bar_idx + 1 if tracker.exit_bar_idx is not None else len(bars))
        if reason in counters:
            counters[reason] += 1

    trades = trades.copy()
    trades["trail_pnl"] = trail_pnl
    trades["trail_reason"] = trail_reasons
    trades["trail_bars"] = trail_bars

    total = len(trades) - counters["no_bars"]
    logger.info(
        f"  H={horizon} trailing stop ({total} trades): "
        f"trail={counters['trailing_stop']}, hard={counters['hard_stop']}, "
        f"expire={counters['session_close']}, no_bars={counters['no_bars']}"
    )
    return trades


# =============================================================================
# STRATEGY STATS
# =============================================================================

def compute_stats(rets: np.ndarray, horizon: int) -> Dict:
    """Compute stats from per-trade returns, annualized for horizon."""
    if len(rets) == 0:
        return {"final_equity": INITIAL_CAPITAL, "total_return_pct": 0, "sharpe": 0,
                "max_dd_pct": 0, "win_rate_pct": 0, "profit_factor": 0, "n_trades": 0}

    total_ret = np.prod(1 + rets) - 1
    trades_per_year = 252 / horizon
    ann_ret = (1 + total_ret) ** (trades_per_year / len(rets)) - 1 if len(rets) > 0 else 0
    vol = np.std(rets, ddof=1) * np.sqrt(trades_per_year) if len(rets) > 1 else 1e-10
    sharpe = ann_ret / vol if vol > 0 else 0

    cumulative = np.cumprod(1 + rets)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_dd = float(np.min(drawdown)) if len(drawdown) > 0 else 0

    wins = np.sum(rets > 0)
    wr = wins / len(rets) * 100
    pos_sum = np.sum(rets[rets > 0])
    neg_sum = np.abs(np.sum(rets[rets < 0]))
    pf = pos_sum / neg_sum if neg_sum > 0 else float("inf")

    return {
        "final_equity": round(INITIAL_CAPITAL * (1 + total_ret), 2),
        "total_return_pct": round(total_ret * 100, 2),
        "annualized_return_pct": round(ann_ret * 100, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(wr, 1),
        "profit_factor": round(pf, 3),
        "n_trades": len(rets),
        "avg_return_per_trade_pct": round(np.mean(rets) * 100, 3),
    }


def run_strategies_for_horizon(trades: pd.DataFrame, horizon: int) -> Dict:
    """Run 4 strategies for a given horizon on non-overlapping trades."""
    slip = SLIPPAGE_BPS / 10_000
    n = len(trades)

    if n == 0:
        return {"n_trades": 0}

    # DA
    actual_signs = np.sign(trades["actual_return_Hd"].values)
    pred_dirs = trades["direction"].values
    correct = int(np.sum(pred_dirs == actual_signs))
    da = correct / n * 100

    # Strategy 1: Forecast 1x (hold-to-close, no leverage)
    f1_rets = []
    for _, row in trades.iterrows():
        ret = row["direction"] * (np.exp(row["actual_return_Hd"]) - 1) - slip
        f1_rets.append(ret)
    f1_rets = np.array(f1_rets)

    # Strategy 2: Forecast + Vol-Target
    vt_rets = []
    for _, row in trades.iterrows():
        lev = row["leverage"]
        ret = row["direction"] * lev * (np.exp(row["actual_return_Hd"]) - 1) - slip * lev
        vt_rets.append(ret)
    vt_rets = np.array(vt_rets)

    # Strategy 3: Forecast + VT + Trailing Stop
    ts_rets = []
    exit_reasons = {"trailing_stop": 0, "hard_stop": 0, "session_close": 0, "no_bars": 0}
    for _, row in trades.iterrows():
        lev = row["leverage"]
        trail_pnl = row.get("trail_pnl")

        if trail_pnl is not None and not (isinstance(trail_pnl, float) and np.isnan(trail_pnl)):
            ret = trail_pnl * lev
        else:
            ret = row["direction"] * lev * (np.exp(row["actual_return_Hd"]) - 1) - slip * lev

        ts_rets.append(ret)
        reason = row.get("trail_reason", "no_bars")
        if reason in exit_reasons:
            exit_reasons[reason] += 1
    ts_rets = np.array(ts_rets)

    # Buy & Hold
    first_close = trades.iloc[0]["close"]
    last_close = trades.iloc[-1]["close"]
    bh_ret = (last_close - first_close) / first_close

    # Statistical test on VT+Trail
    t_stat, p_val = stats.ttest_1samp(ts_rets, 0) if len(ts_rets) > 1 else (0, 1)

    # Long vs Short breakdown
    long_mask = trades["direction"].values == 1
    short_mask = trades["direction"].values == -1
    n_long = int(np.sum(long_mask))
    n_short = int(np.sum(short_mask))
    long_wr = np.sum(ts_rets[long_mask] > 0) / n_long * 100 if n_long > 0 else 0
    short_wr = np.sum(ts_rets[short_mask] > 0) / n_short * 100 if n_short > 0 else 0

    return {
        "horizon": horizon,
        "n_trades": n,
        "da_pct": round(da, 1),
        "n_long": n_long,
        "n_short": n_short,
        "long_wr_pct": round(long_wr, 1),
        "short_wr_pct": round(short_wr, 1),
        "buy_hold": {
            "total_return_pct": round(bh_ret * 100, 2),
            "final_equity": round(INITIAL_CAPITAL * (1 + bh_ret), 2),
        },
        "forecast_1x": compute_stats(f1_rets, horizon),
        "forecast_vt": compute_stats(vt_rets, horizon),
        "forecast_vt_trail": compute_stats(ts_rets, horizon),
        "exit_reasons": exit_reasons,
        "t_stat": round(t_stat, 3),
        "p_value": round(p_val, 4),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()

    df = load_daily_data()
    df_m5 = load_5min_data()

    results = {}

    for h in HORIZONS:
        h_label = {1: "DAILY", 5: "WEEKLY", 20: "MONTHLY"}[h]
        logger.info(f"\n{'='*70}")
        logger.info(f"  HORIZON H={h} ({h_label})")
        logger.info(f"{'='*70}")

        # 1. Predict OOS (all days get a prediction)
        signals = predict_oos_horizon(df, h)

        if len(signals) == 0:
            logger.warning(f"  H={h}: No signals generated!")
            continue

        # 2. Select non-overlapping trades (every H days)
        trades = select_non_overlapping_trades(signals, h)

        if len(trades) == 0:
            logger.warning(f"  H={h}: No trades after filtering!")
            continue

        # 3. Trailing stop on 5-min bars over H days
        trades = simulate_trailing_stop_multi_day(trades, df_m5, h)

        # 4. Run 4 strategies
        r = run_strategies_for_horizon(trades, h)
        results[h] = r

        logger.info(
            f"  H={h} result: {r['n_trades']} trades, DA={r['da_pct']}%, "
            f"VT+Trail: ${r['forecast_vt_trail']['final_equity']:,.0f} "
            f"({r['forecast_vt_trail']['total_return_pct']:+.2f}%), "
            f"Sharpe={r['forecast_vt_trail']['sharpe']:.3f}, p={r['p_value']:.4f}"
        )

    # ==========================================================================
    # PRINT RESULTS
    # ==========================================================================

    print("\n" + "=" * 90)
    print("  MULTI-HORIZON COMPARISON: OOS 2025 (Training SOLO 2020-2024)")
    print("=" * 90)

    # Overview table
    print(f"\n  {'Horizon':<10} {'Trades':>7} {'DA%':>7} {'B&H':>10} {'FC 1x':>10} "
          f"{'FC+VT':>10} {'FC+VT+TS':>10} {'Sharpe':>8} {'p-val':>8}")
    print(f"  {'-'*10} {'-'*7} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*8} {'-'*8}")

    for h in HORIZONS:
        if h not in results:
            continue
        r = results[h]
        h_label = {1: "H=1 (D)", 5: "H=5 (W)", 20: "H=20 (M)"}[h]
        bh = r["buy_hold"]["total_return_pct"]
        f1 = r["forecast_1x"]["total_return_pct"]
        vt = r["forecast_vt"]["total_return_pct"]
        ts = r["forecast_vt_trail"]["total_return_pct"]
        sh = r["forecast_vt_trail"]["sharpe"]
        pv = r["p_value"]
        print(f"  {h_label:<10} {r['n_trades']:>7} {r['da_pct']:>6.1f}% "
              f"{bh:>+9.2f}% {f1:>+9.2f}% {vt:>+9.2f}% {ts:>+9.2f}% "
              f"{sh:>+7.3f} {pv:>8.4f}")

    # Detailed per-horizon
    for h in HORIZONS:
        if h not in results:
            continue
        r = results[h]
        h_label = {1: "DAILY (H=1)", 5: "WEEKLY (H=5)", 20: "MONTHLY (H=20)"}[h]

        print(f"\n  {'-'*80}")
        print(f"  {h_label}")
        print(f"  {'-'*80}")
        print(f"  Trades: {r['n_trades']} | DA: {r['da_pct']}%")
        print(f"  Long: {r['n_long']} (WR={r['long_wr_pct']:.1f}%) | "
              f"Short: {r['n_short']} (WR={r['short_wr_pct']:.1f}%)")

        strats = [
            ("Buy & Hold", r["buy_hold"]),
            ("Forecast 1x", r["forecast_1x"]),
            ("FC + VolTarget", r["forecast_vt"]),
            ("FC + VT + Trail", r["forecast_vt_trail"]),
        ]

        print(f"\n  {'Strategy':<18} {'$10K ->':>12} {'Return':>9} {'Ann.Ret':>9} "
              f"{'Sharpe':>8} {'WR%':>7} {'PF':>7} {'MaxDD':>8}")
        print(f"  {'-'*18} {'-'*12} {'-'*9} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*8}")

        for label, s in strats:
            if "final_equity" not in s:
                continue
            eq = s["final_equity"]
            ret = s["total_return_pct"]
            ann = s.get("annualized_return_pct", ret)
            sh = s.get("sharpe", 0)
            wr = s.get("win_rate_pct", 0)
            pf = s.get("profit_factor", 0)
            mdd = s.get("max_dd_pct", 0)
            print(f"  {label:<18} ${eq:>10,.2f} {ret:>+8.2f}% {ann:>+8.2f}% "
                  f"{sh:>+7.3f} {wr:>6.1f}% {pf:>6.3f} {mdd:>+7.2f}%")

        ex = r["exit_reasons"]
        print(f"\n  Exits: trail={ex['trailing_stop']} hard={ex['hard_stop']} "
              f"expire={ex['session_close']} no_bars={ex['no_bars']}")
        print(f"  t-stat={r['t_stat']:.3f}, p-value={r['p_value']:.4f}")

    # Final comparison
    print(f"\n  {'='*80}")
    print(f"  BEST STRATEGY (FC + VT + Trailing Stop) COMPARISON")
    print(f"  {'='*80}")
    print(f"\n  {'Metric':<25} {'H=1 (Daily)':>15} {'H=5 (Weekly)':>15} {'H=20 (Monthly)':>15}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*15}")

    metrics = [
        ("Trades", "n_trades", "forecast_vt_trail"),
        ("DA%", "da_pct", None),
        ("Return%", "total_return_pct", "forecast_vt_trail"),
        ("Annualized%", "annualized_return_pct", "forecast_vt_trail"),
        ("Sharpe", "sharpe", "forecast_vt_trail"),
        ("Win Rate%", "win_rate_pct", "forecast_vt_trail"),
        ("Profit Factor", "profit_factor", "forecast_vt_trail"),
        ("Max DD%", "max_dd_pct", "forecast_vt_trail"),
        ("Avg Ret/Trade%", "avg_return_per_trade_pct", "forecast_vt_trail"),
        ("p-value", "p_value", None),
    ]

    for label, key, strat in metrics:
        vals = []
        for h in HORIZONS:
            if h not in results:
                vals.append("N/A")
                continue
            r = results[h]
            if strat:
                v = r[strat].get(key, "N/A")
            else:
                v = r.get(key, "N/A")

            if isinstance(v, float):
                vals.append(f"{v:>13.3f}")
            elif isinstance(v, int):
                vals.append(f"{v:>13d}")
            else:
                vals.append(f"{str(v):>13}")
        print(f"  {label:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

    # Save results
    output_path = PROJECT_ROOT / "results" / "multi_horizon_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for h, r in results.items():
        serializable[str(h)] = r
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    elapsed = time.time() - t0
    logger.info(f"\nTotal time: {elapsed:.1f}s")
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
