"""
H=5 Root Cause Analysis
========================
Why does H=5 get p=0.111? Is the trailing stop creating illusory alpha?

Diagnostic sections:
1. FORECAST QUALITY: Is H=5 direction accuracy real or noise? Compare H=1 vs H=5
2. TRAILING STOP DISSECTION: Is the trail creating alpha or just getting lucky?
3. GRID SEARCH OVERFITTING: Does the optimized config degrade on held-out folds?
4. TRADE DECOMPOSITION: Where do the returns come from (monthly, by direction)?
5. RE-ENTRY CONTRIBUTION: How much alpha comes from re-entry vs initial position?
6. SAMPLE SIZE POWER ANALYSIS: How many trades do we need for p<0.05?
7. REGIME STABILITY: Is weekly autocorrelation stable across periods?
8. NULL HYPOTHESIS: Random direction + same trailing stop = what return?

@date 2026-02-16
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

# Best config from grid search
TRAIL_BEST = TrailingStopConfig(activation_pct=0.004, trail_pct=0.003, hard_stop_pct=0.04)

TRAIN_START = pd.Timestamp("2020-01-01")
TRAIN_END = pd.Timestamp("2024-12-31")
OOS_START = pd.Timestamp("2025-01-01")
OOS_END = pd.Timestamp("2026-12-31")


# =============================================================================
# DATA LOADING (shared with exp_h5_001)
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
# MODEL TRAINING (shared)
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


def train_models(df_train, feature_cols, target_col, horizon):
    """Train 9 models, return trained models + scaler."""
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

    return trained, scaler


def predict_ensemble(trained, scaler, feature_row, feature_cols):
    """Get top-3 ensemble prediction for a single row."""
    X_day = feature_row[feature_cols].values.astype(np.float64).reshape(1, -1)
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
        return None, None, {}

    sorted_m = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
    top3 = sorted_m[:3]
    ensemble = np.mean([preds[m] for m in top3])
    direction = 1 if ensemble > 0 else -1
    return ensemble, direction, preds


def select_weekly_trades(signals, horizon=5):
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
# TRAILING STOP SIMULATION
# =============================================================================

def simulate_trailing(trades, df_m5, config, re_entry=True):
    m5_dates = sorted(df_m5["date"].unique())
    slip = SLIPPAGE_BPS / 10_000

    results = []
    for _, trade in trades.iterrows():
        signal_date = trade["date"]
        direction = int(trade["direction"])

        future_dates = [d for d in m5_dates if d > signal_date]
        if len(future_dates) == 0:
            results.append({"week_pnl": None, "reason": "no_bars", "subtrades": 0,
                           "n_reentries": 0, "trail_fires": 0, "hard_fires": 0,
                           "first_sub_pnl": None, "reentry_pnl": 0.0,
                           "max_favorable": 0.0, "max_adverse": 0.0})
            continue

        holding_dates = future_dates[:HORIZON]
        week_pnl = 0.0
        sub_trades = 0
        last_reason = "no_bars"
        n_reentries = 0
        trail_fires = 0
        hard_fires = 0
        first_sub_pnl = None
        reentry_pnl = 0.0
        max_fav = 0.0
        max_adv = 0.0

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
                if direction == 1:
                    slipped_entry = entry_price * (1 + slip)
                else:
                    slipped_entry = entry_price * (1 - slip)
                tracker = TrailingStopTracker(entry_price=slipped_entry, direction=direction, config=config)
                need_entry = False
                sub_trades += 1

            triggered = False
            for bar_idx, (_, bar) in enumerate(bars.iterrows()):
                # Track MFE/MAE
                mid = float(bar["close"])
                unrealized = direction * (mid - slipped_entry) / slipped_entry
                max_fav = max(max_fav, unrealized)
                max_adv = min(max_adv, unrealized)

                state = tracker.update(bar_high=float(bar["high"]), bar_low=float(bar["low"]),
                                       bar_close=float(bar["close"]), bar_idx=bar_idx)
                if state == TrailingState.TRIGGERED:
                    triggered = True
                    break

            if triggered:
                exit_price = tracker.exit_price
                if direction == 1:
                    exit_price *= (1 - slip)
                else:
                    exit_price *= (1 + slip)
                sub_pnl = direction * (exit_price - slipped_entry) / slipped_entry
                week_pnl += sub_pnl

                reason = tracker.exit_reason
                if "hard" in str(reason).lower():
                    hard_fires += 1
                else:
                    trail_fires += 1

                if first_sub_pnl is None:
                    first_sub_pnl = sub_pnl
                else:
                    reentry_pnl += sub_pnl

                last_reason = reason

                if re_entry and day_idx < len(holding_dates) - 1:
                    need_entry = True
                    n_reentries += 1
                    day_idx += 1
                    continue
                else:
                    break
            else:
                if day_idx == len(holding_dates) - 1:
                    last_close = float(bars.iloc[-1]["close"])
                    if direction == 1:
                        exit_price = last_close * (1 - slip)
                    else:
                        exit_price = last_close * (1 + slip)
                    sub_pnl = direction * (exit_price - slipped_entry) / slipped_entry
                    week_pnl += sub_pnl
                    last_reason = "week_end"
                    if first_sub_pnl is None:
                        first_sub_pnl = sub_pnl

            day_idx += 1

        results.append({
            "week_pnl": week_pnl,
            "reason": last_reason,
            "subtrades": sub_trades,
            "n_reentries": n_reentries,
            "trail_fires": trail_fires,
            "hard_fires": hard_fires,
            "first_sub_pnl": first_sub_pnl if first_sub_pnl is not None else 0.0,
            "reentry_pnl": reentry_pnl,
            "max_favorable": max_fav,
            "max_adverse": max_adv,
        })

    r_df = pd.DataFrame(results)
    for col in r_df.columns:
        trades[col] = r_df[col].values
    return trades


# =============================================================================
# DIAGNOSTIC SECTIONS
# =============================================================================

def diag_1_forecast_quality(df, feature_cols):
    """Compare H=1 vs H=5 direction accuracy using walk-forward."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 1: FORECAST QUALITY - H=1 vs H=5")
    print("="*80)

    # Train on 2020-2024, test on 2025
    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_test = df[(df["date"] >= OOS_START) & (df["date"] <= pd.Timestamp("2025-12-31"))].copy()
    df_2026 = df[df["date"] >= pd.Timestamp("2026-01-01")].copy()

    print(f"\nTrain: {len(df_train)} days ({df_train['date'].min().date()} -> {df_train['date'].max().date()})")
    print(f"Test 2025: {len(df_test)} days")
    print(f"Test 2026: {len(df_2026)} days")

    # Train H=1 and H=5 models
    for h, target_col in [(1, "target_1d"), (5, "target_5d")]:
        trained_h, scaler_h = train_models(df_train, feature_cols, target_col, h)
        print(f"\n--- H={h} ({len(trained_h)} models trained) ---")

        for period_name, df_period in [("2025", df_test), ("2026", df_2026)]:
            if len(df_period) == 0:
                print(f"  {period_name}: no data")
                continue

            predictions = []
            per_model_preds = {m: [] for m in trained_h.keys()}

            for _, row in df_period.iterrows():
                if pd.isna(row[target_col]):
                    continue
                ens, direction, all_preds = predict_ensemble(trained_h, scaler_h, row, feature_cols)
                if ens is None:
                    continue
                actual_dir = 1 if row[target_col] > 0 else -1
                predictions.append({
                    "date": row["date"],
                    "pred_dir": direction,
                    "actual_dir": actual_dir,
                    "pred_mag": abs(ens),
                    "actual_mag": abs(row[target_col]),
                    "pred_val": ens,
                    "actual_val": row[target_col],
                })
                for m, p in all_preds.items():
                    per_model_preds[m].append({
                        "pred_dir": 1 if p > 0 else -1,
                        "actual_dir": actual_dir,
                    })

            preds_df = pd.DataFrame(predictions)
            if len(preds_df) == 0:
                print(f"  {period_name}: no predictions")
                continue

            # Overall DA
            da = (preds_df["pred_dir"] == preds_df["actual_dir"]).mean() * 100
            n = len(preds_df)
            # Binomial CI
            se = np.sqrt(da/100 * (1-da/100) / n) * 100
            # Binomial test vs 50%
            n_correct = int((preds_df["pred_dir"] == preds_df["actual_dir"]).sum())
            binom_p = stats.binom_test(n_correct, n, 0.5)

            # Direction bias
            n_long = int((preds_df["pred_dir"] == 1).sum())
            long_pct = n_long / n * 100

            # Correlation pred vs actual
            corr = np.corrcoef(preds_df["pred_val"], preds_df["actual_val"])[0, 1]

            print(f"\n  {period_name} (n={n}):")
            print(f"    DA = {da:.1f}% +/- {se:.1f}% (binomial p={binom_p:.4f})")
            print(f"    Pred-actual correlation: {corr:.4f}")
            print(f"    Direction bias: {long_pct:.1f}% LONG, {100-long_pct:.1f}% SHORT")
            print(f"    Avg |prediction|: {preds_df['pred_mag'].mean():.6f}")
            print(f"    Avg |actual|:     {preds_df['actual_mag'].mean():.6f}")

            # Per-model DA
            print(f"\n    Per-model DA ({period_name}):")
            for m, mp in sorted(per_model_preds.items()):
                if len(mp) == 0:
                    continue
                mp_df = pd.DataFrame(mp)
                mda = (mp_df["pred_dir"] == mp_df["actual_dir"]).mean() * 100
                print(f"      {m:25s}: {mda:.1f}% ({len(mp_df)} pred)")

            # Weekly sampling DA (for H=5 comparison fairness)
            if h == 1:
                weekly = preds_df.iloc[::5]
                wda = (weekly["pred_dir"] == weekly["actual_dir"]).mean() * 100
                print(f"\n    H=1 weekly-sampled DA (every 5th day): {wda:.1f}% ({len(weekly)} trades)")


def diag_2_trailing_stop_dissection(trades_df):
    """Break down where trailing stop returns come from."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 2: TRAILING STOP DISSECTION")
    print("="*80)

    valid = trades_df[trades_df["week_pnl"].notna()].copy()
    if len(valid) == 0:
        print("  No valid trades")
        return

    # 2a: Exit reason breakdown
    print("\n--- Exit Reason Breakdown ---")
    reasons = valid["reason"].value_counts()
    for r, count in reasons.items():
        subset = valid[valid["reason"] == r]
        avg_pnl = subset["week_pnl"].mean() * 100
        wr = (subset["week_pnl"] > 0).mean() * 100
        print(f"  {str(r):20s}: {count:3d} trades, avg PnL={avg_pnl:+.3f}%, WR={wr:.1f}%")

    # 2b: Re-entry contribution
    print("\n--- Re-entry Contribution ---")
    total_pnl = valid["week_pnl"].sum()
    first_pnl = valid["first_sub_pnl"].sum()
    reentry_total = valid["reentry_pnl"].sum()
    n_with_reentry = (valid["n_reentries"] > 0).sum()

    print(f"  Total PnL:         {total_pnl*100:.3f}%")
    print(f"  First sub-trade:   {first_pnl*100:.3f}% ({first_pnl/total_pnl*100:.1f}% of total)" if total_pnl != 0 else "  First sub-trade: N/A")
    print(f"  Re-entry PnL:      {reentry_total*100:.3f}% ({reentry_total/total_pnl*100:.1f}% of total)" if total_pnl != 0 else "  Re-entry PnL: N/A")
    print(f"  Trades with re-entry: {n_with_reentry}/{len(valid)} ({n_with_reentry/len(valid)*100:.1f}%)")

    # 2c: MFE/MAE analysis (max favorable / max adverse excursion)
    print("\n--- MFE/MAE Analysis ---")
    mfe = valid["max_favorable"].values * 100
    mae = valid["max_adverse"].values * 100
    print(f"  Avg MFE: {np.mean(mfe):.3f}%  (median: {np.median(mfe):.3f}%)")
    print(f"  Avg MAE: {np.mean(mae):.3f}%  (median: {np.median(mae):.3f}%)")

    # MFE/MAE by winning vs losing
    winners = valid[valid["week_pnl"] > 0]
    losers = valid[valid["week_pnl"] <= 0]
    if len(winners) > 0:
        print(f"  Winners MFE: {winners['max_favorable'].mean()*100:.3f}%, MAE: {winners['max_adverse'].mean()*100:.3f}%")
    if len(losers) > 0:
        print(f"  Losers  MFE: {losers['max_favorable'].mean()*100:.3f}%, MAE: {losers['max_adverse'].mean()*100:.3f}%")

    # 2d: Trail stop fires vs hard stop fires
    print("\n--- Stop Type Analysis ---")
    print(f"  Trail fires: {valid['trail_fires'].sum()}")
    print(f"  Hard fires:  {valid['hard_fires'].sum()}")
    print(f"  Week end:    {(valid['reason'] == 'week_end').sum()}")

    # 2e: Correct vs wrong direction breakdown
    print("\n--- Direction Accuracy vs Trail Outcome ---")
    if "actual_return" in valid.columns:
        correct = valid[np.sign(valid["actual_return"]) == valid["direction"]]
        wrong = valid[np.sign(valid["actual_return"]) != valid["direction"]]
        print(f"  Correct direction ({len(correct)} trades):")
        if len(correct) > 0:
            print(f"    Avg week PnL: {correct['week_pnl'].mean()*100:+.3f}%")
            print(f"    Trail captured: {correct['week_pnl'].mean()/correct['actual_return'].abs().mean()*100:.1f}% of move")
        print(f"  Wrong direction ({len(wrong)} trades):")
        if len(wrong) > 0:
            print(f"    Avg week PnL: {wrong['week_pnl'].mean()*100:+.3f}%")
            print(f"    Avg actual return (abs): {wrong['actual_return'].abs().mean()*100:.3f}%")


def diag_3_grid_search_overfitting(df, df_m5, feature_cols):
    """Test if grid-optimized config works on held-out data."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 3: GRID SEARCH OVERFITTING TEST")
    print("="*80)
    print("Training on 2020-2023, grid search on 2024, test on 2025")

    # Split: train 2020-2023, grid_val 2024, test 2025
    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= pd.Timestamp("2023-12-31"))].copy()
    df_grid_val = df[(df["date"] >= pd.Timestamp("2024-01-01")) & (df["date"] <= pd.Timestamp("2024-12-31"))].copy()
    df_test_oos = df[(df["date"] >= pd.Timestamp("2025-01-01")) & (df["date"] <= pd.Timestamp("2025-12-31"))].copy()

    print(f"  Train: {len(df_train)} days (2020-2023)")
    print(f"  Grid validation: {len(df_grid_val)} days (2024)")
    print(f"  True OOS: {len(df_test_oos)} days (2025)")

    trained_h5, scaler_h5 = train_models(df_train, feature_cols, "target_5d", 5)
    print(f"  Models trained: {len(trained_h5)}")

    # Predict on grid_val (2024) and test (2025) separately
    for period_name, df_period in [("2024 (grid val)", df_grid_val), ("2025 (true OOS)", df_test_oos)]:
        predictions = []
        for _, row in df_period.iterrows():
            if pd.isna(row["target_5d"]):
                continue
            ens, direction, _ = predict_ensemble(trained_h5, scaler_h5, row, feature_cols)
            if ens is None:
                continue
            predictions.append({"date": row["date"], "close": row["close"],
                               "actual_return": row["target_5d"], "ensemble_pred": ens, "direction": direction})

        preds_df = pd.DataFrame(predictions)
        weekly = select_weekly_trades(preds_df, 5)

        # Add vol targeting
        vol = compute_realized_vol(df[df["date"] <= weekly["date"].max()]["close"], VOL_CONFIG.vol_lookback)
        for idx in weekly.index:
            d = weekly.loc[idx, "date"]
            v_before = vol[vol.index <= d]
            if len(v_before) > 0:
                weekly.loc[idx, "leverage"] = np.clip(VOL_CONFIG.target_vol / max(v_before.iloc[-1], VOL_CONFIG.vol_floor),
                                                       VOL_CONFIG.min_leverage, VOL_CONFIG.max_leverage)
            else:
                weekly.loc[idx, "leverage"] = 1.0

        # DA
        actual_dir = np.sign(weekly["actual_return"])
        da = (weekly["direction"] == actual_dir).mean() * 100

        # Test with the "optimized" trailing params
        weekly_trail = simulate_trailing(weekly, df_m5, TRAIL_BEST, re_entry=True)
        valid = weekly_trail[weekly_trail["week_pnl"].notna()]

        if len(valid) > 0:
            rets = valid["week_pnl"].values * valid["leverage"].values
            total_ret = np.prod(1 + rets) - 1
            wr = (rets > 0).mean() * 100
            n = len(rets)
            tpy = 252 / 5
            ann = (1 + total_ret) ** (tpy / n) - 1 if n > 0 else 0
            vol_ann = np.std(rets, ddof=1) * np.sqrt(tpy) if n > 1 else 1
            sharpe = ann / vol_ann if vol_ann > 0 else 0
        else:
            total_ret = 0; wr = 0; sharpe = 0; n = 0

        print(f"\n  {period_name}: n={n}, DA={da:.1f}%, Return={total_ret*100:+.2f}%, Sharpe={sharpe:.3f}, WR={wr:.1f}%")

    # Also test: what if we use DEFAULT trail params (not grid-optimized)?
    print("\n--- Comparison: Optimized vs Default trail on 2025 OOS ---")
    configs_to_test = [
        ("Grid-best (0.40/0.30/4.00)", TRAIL_BEST),
        ("Default H5 (0.35/0.50/2.50)", TrailingStopConfig(activation_pct=0.0035, trail_pct=0.005, hard_stop_pct=0.025)),
        ("Conservative (0.30/0.40/2.00)", TrailingStopConfig(activation_pct=0.003, trail_pct=0.004, hard_stop_pct=0.020)),
        ("No trail (hold H=5)", None),
    ]

    # Re-predict 2025 with 2020-2023 training
    predictions_2025 = []
    for _, row in df_test_oos.iterrows():
        if pd.isna(row["target_5d"]):
            continue
        ens, direction, _ = predict_ensemble(trained_h5, scaler_h5, row, feature_cols)
        if ens is None:
            continue
        predictions_2025.append({"date": row["date"], "close": row["close"],
                                 "actual_return": row["target_5d"], "ensemble_pred": ens, "direction": direction})

    preds_2025 = pd.DataFrame(predictions_2025)
    weekly_2025 = select_weekly_trades(preds_2025, 5)

    vol = compute_realized_vol(df[df["date"] <= weekly_2025["date"].max()]["close"], VOL_CONFIG.vol_lookback)
    for idx in weekly_2025.index:
        d = weekly_2025.loc[idx, "date"]
        v_before = vol[vol.index <= d]
        if len(v_before) > 0:
            weekly_2025.loc[idx, "leverage"] = np.clip(VOL_CONFIG.target_vol / max(v_before.iloc[-1], VOL_CONFIG.vol_floor),
                                                        VOL_CONFIG.min_leverage, VOL_CONFIG.max_leverage)
        else:
            weekly_2025.loc[idx, "leverage"] = 1.0

    for config_name, config in configs_to_test:
        if config is None:
            # Pure hold-to-expiry
            slip = SLIPPAGE_BPS / 10_000
            rets = np.array([r["direction"] * r["leverage"] * (np.exp(r["actual_return"]) - 1) - slip * r["leverage"]
                            for _, r in weekly_2025.iterrows()])
        else:
            wt = simulate_trailing(weekly_2025.copy(), df_m5, config, re_entry=True)
            valid = wt[wt["week_pnl"].notna()]
            rets = valid["week_pnl"].values * valid["leverage"].values

        if len(rets) > 0:
            total_ret = np.prod(1 + rets) - 1
            wr = (rets > 0).mean() * 100
            n = len(rets)
            tpy = 252 / 5
            ann = (1 + total_ret) ** (tpy / n) - 1 if n > 0 else 0
            vol_ann = np.std(rets, ddof=1) * np.sqrt(tpy) if n > 1 else 1
            sharpe = ann / vol_ann if vol_ann > 0 else 0
        else:
            total_ret = 0; wr = 0; sharpe = 0; n = 0

        print(f"  {config_name:40s}: Return={total_ret*100:+.2f}%, Sharpe={sharpe:.3f}, WR={wr:.1f}% (n={n})")


def diag_4_monthly_decomposition(trades_df):
    """Where do the returns come from month by month?"""
    print("\n" + "="*80)
    print("DIAGNOSTIC 4: MONTHLY DECOMPOSITION")
    print("="*80)

    valid = trades_df[trades_df["week_pnl"].notna()].copy()
    if len(valid) == 0:
        print("  No valid trades")
        return

    valid["month"] = valid["date"].dt.to_period("M")

    print(f"\n{'Month':>10s} | {'Trades':>6s} | {'DA%':>5s} | {'VT+Trail%':>10s} | {'WR%':>5s} | {'Exits':>30s}")
    print("-" * 80)

    for month, group in valid.groupby("month"):
        n = len(group)
        actual_dir = np.sign(group["actual_return"])
        da = (group["direction"] == actual_dir).mean() * 100
        pnl = group["week_pnl"].sum() * 100
        wr = (group["week_pnl"] > 0).mean() * 100

        exits = group["reason"].value_counts().to_dict()
        exit_str = ", ".join([f"{k}:{v}" for k, v in sorted(exits.items())])

        print(f"  {str(month):>10s} | {n:6d} | {da:5.1f} | {pnl:+10.3f} | {wr:5.1f} | {exit_str}")

    # Concentration
    monthly_pnl = valid.groupby("month")["week_pnl"].sum()
    if len(monthly_pnl) > 0:
        best_month = monthly_pnl.idxmax()
        best_val = monthly_pnl.max() * 100
        total = monthly_pnl.sum() * 100
        concentration = best_val / total * 100 if total != 0 else 0
        print(f"\n  Best month: {best_month} ({best_val:+.3f}%)")
        print(f"  Total PnL: {total:+.3f}%")
        print(f"  Concentration: best month = {concentration:.1f}% of total")
        print(f"  Months positive: {(monthly_pnl > 0).sum()}/{len(monthly_pnl)}")


def diag_5_power_analysis(trades_df):
    """How many trades do we need for statistical significance?"""
    print("\n" + "="*80)
    print("DIAGNOSTIC 5: POWER ANALYSIS")
    print("="*80)

    valid = trades_df[trades_df["week_pnl"].notna()].copy()
    if len(valid) == 0:
        print("  No valid trades")
        return

    rets = valid["week_pnl"].values
    if "leverage" in valid.columns:
        rets = rets * valid["leverage"].values

    mu = np.mean(rets)
    sigma = np.std(rets, ddof=1)
    n = len(rets)

    print(f"\n  Current: n={n}, mean={mu*100:.4f}%, std={sigma*100:.4f}%")
    t_stat = mu / (sigma / np.sqrt(n)) if sigma > 0 else 0
    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
    print(f"  t-stat={t_stat:.3f}, p={p_val:.4f}")

    # What n needed for p<0.05 at this effect size?
    if sigma > 0 and mu > 0:
        effect_size = mu / sigma
        n_needed_05 = int(np.ceil((stats.norm.ppf(0.975) / effect_size) ** 2))
        n_needed_10 = int(np.ceil((stats.norm.ppf(0.90) / effect_size) ** 2))
        print(f"\n  Effect size (d): {effect_size:.4f}")
        print(f"  Trades needed for p<0.05 (two-sided): ~{n_needed_05}")
        print(f"  Trades needed for p<0.10 (two-sided): ~{n_needed_10}")
        print(f"  At 44 trades/year: need {n_needed_05/44:.1f} years for p<0.05")
        print(f"  At 52 trades/year (w/ 2026): need {n_needed_05/52:.1f} years for p<0.05")

    # Bootstrap
    print("\n--- Bootstrap Analysis (10,000 samples) ---")
    n_boot = 10_000
    boot_means = []
    for _ in range(n_boot):
        sample = np.random.choice(rets, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.array(boot_means)
    ci_lo = np.percentile(boot_means, 2.5) * 100
    ci_hi = np.percentile(boot_means, 97.5) * 100
    p_boot = np.mean(boot_means <= 0)
    print(f"  Bootstrap 95% CI: [{ci_lo:+.3f}%, {ci_hi:+.3f}%]")
    print(f"  Bootstrap p(mean<=0): {p_boot:.4f}")
    print(f"  CI excludes zero: {'YES' if ci_lo > 0 else 'NO'}")


def diag_6_weekly_autocorrelation(df):
    """Is weekly return autocorrelation stable across periods?"""
    print("\n" + "="*80)
    print("DIAGNOSTIC 6: WEEKLY AUTOCORRELATION REGIME")
    print("="*80)

    # Compute weekly returns
    df_sorted = df.sort_values("date").copy()
    weekly_idx = list(range(0, len(df_sorted), 5))
    weekly_rets = []
    for i in range(len(weekly_idx) - 1):
        start_idx = weekly_idx[i]
        end_idx = weekly_idx[i+1]
        start_close = df_sorted.iloc[start_idx]["close"]
        end_close = df_sorted.iloc[end_idx]["close"]
        weekly_rets.append({
            "date": df_sorted.iloc[start_idx]["date"],
            "return": np.log(end_close / start_close),
        })
    wr_df = pd.DataFrame(weekly_rets)

    periods = [
        ("Training (2020-2024)", (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31"))),
        ("2020-2022", (pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"))),
        ("2023-2024", (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31"))),
        ("2025 H1", (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"))),
        ("2025 H2", (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31"))),
        ("2026 YTD", (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31"))),
    ]

    print(f"\n{'Period':>20s} | {'AC(1)':>8s} | {'AC(2)':>8s} | {'Mean%':>8s} | {'Std%':>8s} | {'N':>5s} | {'Regime':>15s}")
    print("-" * 90)

    for name, (start, end) in periods:
        sub = wr_df[(wr_df["date"] >= start) & (wr_df["date"] <= end)]["return"].values
        if len(sub) < 5:
            print(f"  {name:>20s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s} | {'N/A':>8s} | {len(sub):5d} | {'insufficient':>15s}")
            continue

        ac1 = np.corrcoef(sub[:-1], sub[1:])[0, 1] if len(sub) > 2 else 0
        ac2 = np.corrcoef(sub[:-2], sub[2:])[0, 1] if len(sub) > 3 else 0
        mean_r = np.mean(sub) * 100
        std_r = np.std(sub) * 100
        regime = "MOMENTUM" if ac1 > 0.1 else ("MEAN-REV" if ac1 < -0.1 else "RANDOM")

        print(f"  {name:>20s} | {ac1:+8.4f} | {ac2:+8.4f} | {mean_r:+8.3f} | {std_r:8.3f} | {len(sub):5d} | {regime:>15s}")


def diag_7_null_hypothesis(trades_df, df_m5):
    """Random direction + same trailing stop = what return?"""
    print("\n" + "="*80)
    print("DIAGNOSTIC 7: NULL HYPOTHESIS - RANDOM DIRECTIONS")
    print("="*80)
    print("If trailing stop creates alpha regardless of direction,")
    print("random directions should also be profitable.")

    valid = trades_df[trades_df["week_pnl"].notna()].copy()
    if len(valid) == 0:
        print("  No valid trades")
        return

    n_simulations = 500
    n_trades = len(valid)
    random_returns = []

    for sim in range(n_simulations):
        random_dirs = np.random.choice([-1, 1], size=n_trades)
        random_trades = valid.copy()
        random_trades["direction"] = random_dirs

        # Simulate trailing
        rand_result = simulate_trailing(random_trades, df_m5, TRAIL_BEST, re_entry=True)
        rand_valid = rand_result[rand_result["week_pnl"].notna()]

        if len(rand_valid) > 0:
            rets = rand_valid["week_pnl"].values
            if "leverage" in rand_valid.columns:
                rets = rets * rand_valid["leverage"].values
            total_ret = np.prod(1 + rets) - 1
            random_returns.append(total_ret)

    random_returns = np.array(random_returns)
    actual_ret = np.prod(1 + valid["week_pnl"].values * valid.get("leverage", pd.Series([1]*len(valid))).values) - 1

    print(f"\n  Simulations: {n_simulations}")
    print(f"  Actual strategy return: {actual_ret*100:+.2f}%")
    print(f"  Random mean return:     {np.mean(random_returns)*100:+.2f}% +/- {np.std(random_returns)*100:.2f}%")
    print(f"  Random median return:   {np.median(random_returns)*100:+.2f}%")
    print(f"  Random 5th percentile:  {np.percentile(random_returns, 5)*100:+.2f}%")
    print(f"  Random 95th percentile: {np.percentile(random_returns, 95)*100:+.2f}%")
    print(f"  Random > 0:             {(random_returns > 0).mean()*100:.1f}%")

    # How often does random beat our strategy?
    pct_random_beats = (random_returns >= actual_ret).mean() * 100
    print(f"\n  Random beats strategy:  {pct_random_beats:.1f}% of simulations")
    print(f"  -> p-value (empirical):  {pct_random_beats/100:.4f}")

    if pct_random_beats < 5:
        print("  CONCLUSION: Strategy significantly beats random (p<0.05)")
    elif pct_random_beats < 10:
        print("  CONCLUSION: Strategy marginally beats random (p<0.10)")
    else:
        print("  CONCLUSION: CANNOT distinguish from random")
        print("  >>> TRAILING STOP MAY BE CREATING ILLUSORY ALPHA <<<")


def diag_8_h1_vs_h5_fair_comparison(df, df_m5, feature_cols):
    """Fair comparison: both H=1 and H=5 with same training data and optimal trails."""
    print("\n" + "="*80)
    print("DIAGNOSTIC 8: FAIR H=1 vs H=5 COMPARISON")
    print("="*80)
    print("Same training data (2020-2024), OOS 2025, each with its own optimized trail.")

    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_test = df[(df["date"] >= OOS_START) & (df["date"] <= pd.Timestamp("2025-12-31"))].copy()

    results = {}
    for h, target_col, trail_config, label in [
        (1, "target_1d", TrailingStopConfig(activation_pct=0.002, trail_pct=0.003, hard_stop_pct=0.015), "H=1 (prod trail)"),
        (5, "target_5d", TRAIL_BEST, "H=5 (grid-best trail)"),
        (5, "target_5d", TrailingStopConfig(activation_pct=0.0035, trail_pct=0.005, hard_stop_pct=0.025), "H=5 (default trail)"),
        (5, "target_5d", None, "H=5 (no trail, hold-to-expiry)"),
    ]:
        trained_h, scaler_h = train_models(df_train, feature_cols, target_col, h)

        predictions = []
        for _, row in df_test.iterrows():
            if pd.isna(row[target_col]):
                continue
            ens, direction, _ = predict_ensemble(trained_h, scaler_h, row, feature_cols)
            if ens is None:
                continue
            predictions.append({"date": row["date"], "close": row["close"],
                               "actual_return": row[target_col], "ensemble_pred": ens, "direction": direction})

        preds_df = pd.DataFrame(predictions)
        weekly = select_weekly_trades(preds_df, h)

        # Vol target
        vol = compute_realized_vol(df[df["date"] <= weekly["date"].max()]["close"], VOL_CONFIG.vol_lookback)
        for idx in weekly.index:
            d = weekly.loc[idx, "date"]
            v_before = vol[vol.index <= d]
            if len(v_before) > 0:
                weekly.loc[idx, "leverage"] = np.clip(VOL_CONFIG.target_vol / max(v_before.iloc[-1], VOL_CONFIG.vol_floor),
                                                       VOL_CONFIG.min_leverage, VOL_CONFIG.max_leverage)
            else:
                weekly.loc[idx, "leverage"] = 1.0

        da = (np.sign(weekly["actual_return"]) == weekly["direction"]).mean() * 100

        if trail_config is not None:
            weekly_trail = simulate_trailing(weekly.copy(), df_m5, trail_config, re_entry=(h == 5))
            valid = weekly_trail[weekly_trail["week_pnl"].notna()]
            rets = valid["week_pnl"].values * valid["leverage"].values
        else:
            slip = SLIPPAGE_BPS / 10_000
            rets = np.array([r["direction"] * r["leverage"] * (np.exp(r["actual_return"]) - 1) - slip * r["leverage"]
                            for _, r in weekly.iterrows()])

        if len(rets) > 0:
            total_ret = np.prod(1 + rets) - 1
            wr = (rets > 0).mean() * 100
            n = len(rets)
            tpy = 252 / h
            ann = (1 + total_ret) ** (tpy / n) - 1 if n > 0 else 0
            vol_ann = np.std(rets, ddof=1) * np.sqrt(tpy) if n > 1 else 1
            sharpe = ann / vol_ann if vol_ann > 0 else 0
            t_stat = np.mean(rets) / (np.std(rets, ddof=1) / np.sqrt(n)) if np.std(rets) > 0 else 0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-1))
        else:
            total_ret = wr = sharpe = p_val = 0; n = 0

        results[label] = {"n": n, "da": da, "return": total_ret, "sharpe": sharpe, "wr": wr, "p": p_val}

    print(f"\n{'Strategy':>35s} | {'N':>4s} | {'DA%':>5s} | {'Return%':>8s} | {'Sharpe':>7s} | {'WR%':>5s} | {'p-val':>7s}")
    print("-" * 85)
    for label, r in results.items():
        print(f"  {label:>35s} | {r['n']:4d} | {r['da']:5.1f} | {r['return']*100:+8.2f} | {r['sharpe']:7.3f} | {r['wr']:5.1f} | {r['p']:7.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    print("="*80)
    print("H=5 ROOT CAUSE ANALYSIS")
    print("="*80)

    df = load_daily_data()
    df_m5 = load_5min_data()
    feature_cols = list(FEATURE_COLUMNS)

    # --- Train models and get OOS predictions (2020-2024 -> 2025) ---
    print("\n--- Training models (2020-2024) for OOS evaluation ---")
    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)].copy()
    df_oos = df[df["date"] >= OOS_START].copy()

    trained_h5, scaler_h5 = train_models(df_train, feature_cols, "target_5d", 5)
    print(f"  Trained {len(trained_h5)} models")

    # Get all OOS predictions
    predictions = []
    for _, row in df_oos.iterrows():
        if pd.isna(row["target_5d"]):
            continue
        ens, direction, all_preds = predict_ensemble(trained_h5, scaler_h5, row, feature_cols)
        if ens is None:
            continue
        predictions.append({
            "date": row["date"], "close": row["close"],
            "actual_return": row["target_5d"],
            "ensemble_pred": ens, "direction": direction,
        })

    all_preds_df = pd.DataFrame(predictions)
    weekly = select_weekly_trades(all_preds_df, 5)
    print(f"  OOS predictions: {len(all_preds_df)} days -> {len(weekly)} weekly trades")

    # Add vol targeting
    vol = compute_realized_vol(df[df["date"] <= weekly["date"].max()]["close"], VOL_CONFIG.vol_lookback)
    for idx in weekly.index:
        d = weekly.loc[idx, "date"]
        v_before = vol[vol.index <= d]
        if len(v_before) > 0:
            weekly.loc[idx, "leverage"] = np.clip(
                VOL_CONFIG.target_vol / max(v_before.iloc[-1], VOL_CONFIG.vol_floor),
                VOL_CONFIG.min_leverage, VOL_CONFIG.max_leverage)
        else:
            weekly.loc[idx, "leverage"] = 1.0

    # Simulate trailing
    weekly = simulate_trailing(weekly, df_m5, TRAIL_BEST, re_entry=True)

    # Split 2025 vs 2026
    w_2025 = weekly[weekly["date"] < pd.Timestamp("2026-01-01")].copy()
    w_2026 = weekly[weekly["date"] >= pd.Timestamp("2026-01-01")].copy()
    print(f"  2025: {len(w_2025)} trades, 2026: {len(w_2026)} trades")

    # --- Run all diagnostics ---
    diag_1_forecast_quality(df, feature_cols)
    diag_2_trailing_stop_dissection(w_2025)
    print("\n  --- 2026 Trailing Stop Dissection ---")
    diag_2_trailing_stop_dissection(w_2026)
    diag_3_grid_search_overfitting(df, df_m5, feature_cols)
    diag_4_monthly_decomposition(w_2025)
    print("\n  --- 2026 Monthly ---")
    diag_4_monthly_decomposition(w_2026)
    diag_5_power_analysis(w_2025)
    diag_6_weekly_autocorrelation(df)
    diag_7_null_hypothesis(w_2025, df_m5)
    diag_8_h1_vs_h5_fair_comparison(df, df_m5, feature_cols)

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETE ({elapsed:.0f}s)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
