"""
Daily Production Runner
========================

Standalone script designed to run daily at 12:30 PM COT via Windows Task Scheduler.
Performs the complete daily production cycle:

1. Fetch today's daily OHLCV bar (TwelveData)
2. Update DXY + WTI from FRED
3. Build features, train 9 models, generate H=1 signal
4. Compute vol-target leverage
5. Output signal JSON
6. Update production trades + summary
7. Regenerate production PNGs
8. If Sunday: regenerate weekly forecast PNGs

Usage:
    python scripts/run_daily_production.py              # Normal run
    python scripts/run_daily_production.py --dry-run    # Signal only, no trades
    python scripts/run_daily_production.py --force-png  # Force PNG regeneration

Scheduling (Windows Task Scheduler):
    schtasks /create /tn "USDCOP_Daily_Production" /tr "python C:\\path\\to\\scripts\\run_daily_production.py" /sc daily /st 12:30

@version 1.0.0
@date 2026-02-15
"""

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# =============================================================================
# PATHS
# =============================================================================

DAILY_OHLCV_PATH = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
M5_OHLCV_PATH = PROJECT_ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
MACRO_PATH = (
    PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output"
    / "MACRO_DAILY_CLEAN.parquet"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "data" / "production"
)
SIGNAL_PATH = OUTPUT_DIR / "latest_signal.json"
TRADES_PATH = OUTPUT_DIR / "trades" / "forecast_vt_trailing.json"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
APPROVAL_PATH = OUTPUT_DIR / "approval_state.json"
EXECUTOR_CONFIG_PATH = PROJECT_ROOT / "config" / "execution" / "smart_executor_v1.yaml"

# =============================================================================
# API CONFIG
# =============================================================================

TWELVEDATA_BASE = "https://api.twelvedata.com"
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

TD_KEYS = [os.getenv(f"TWELVEDATA_API_KEY_{i}") for i in range(1, 9)]
TD_KEYS = [k for k in TD_KEYS if k]
FRED_KEY = os.getenv("FRED_API_KEY", "")
_td_idx = 0


def _next_key() -> str:
    global _td_idx
    key = TD_KEYS[_td_idx % len(TD_KEYS)]
    _td_idx += 1
    return key


# =============================================================================
# DIRECTION FILTER
# =============================================================================

def load_direction_filter() -> str:
    """Load direction filter from smart_executor config.

    Returns "all", "short_only", or "long_only".
    """
    try:
        import yaml
        with open(EXECUTOR_CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)
        mode = cfg.get("direction_filter", {}).get("mode", "all")
        if mode in ("all", "short_only", "long_only"):
            return mode
        return "all"
    except Exception:
        return "all"


def should_skip_signal(direction: int, direction_filter: str) -> bool:
    """Return True if signal should be skipped based on direction filter."""
    if direction_filter == "short_only" and direction == 1:
        return True
    if direction_filter == "long_only" and direction == -1:
        return True
    return False


# =============================================================================
# APPROVAL CHECK
# =============================================================================

def check_approval() -> bool:
    """Check if strategy is approved for live trading.

    Returns True if status is APPROVED or LIVE, False otherwise.
    When not approved, the daily runner should operate in dry-run mode.
    """
    if not APPROVAL_PATH.exists():
        logger.warning("No approval_state.json found — running in DRY-RUN mode")
        return False

    try:
        with open(APPROVAL_PATH) as f:
            state = json.load(f)

        status = state.get("status", "PENDING_APPROVAL")

        if status in ("APPROVED", "LIVE"):
            approved_by = state.get("approved_by", "unknown")
            approved_at = state.get("approved_at", "unknown")
            logger.info(f"Strategy APPROVED by {approved_by} on {approved_at}")
            return True
        else:
            logger.warning(f"Strategy status is {status} — running in DRY-RUN mode")
            return False

    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Error reading approval state: {e} — running in DRY-RUN mode")
        return False


# =============================================================================
# STEP 1: FETCH TODAY'S DAILY BAR
# =============================================================================

def fetch_today_daily() -> Optional[pd.DataFrame]:
    """Fetch latest daily bar from TwelveData and append to parquet."""
    today = datetime.now().strftime("%Y-%m-%d")
    yesterday = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")

    logger.info(f"Fetching daily OHLCV: {yesterday} → {today}")

    params = {
        "symbol": "USD/COP",
        "interval": "1day",
        "start_date": yesterday,
        "end_date": today,
        "timezone": "America/Bogota",
        "outputsize": 10,
        "apikey": _next_key(),
    }

    try:
        resp = requests.get(f"{TWELVEDATA_BASE}/time_series", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "values" not in data:
            logger.warning(f"No values from TwelveData: {data.get('message', '')}")
            return None

        rows = []
        for v in data["values"]:
            close = float(v["close"])
            if 3000 <= close <= 6000:
                rows.append({
                    "time": pd.Timestamp(v["datetime"]),
                    "open": float(v["open"]),
                    "high": float(v["high"]),
                    "low": float(v["low"]),
                    "close": close,
                    "volume": 0.0,
                    "symbol": "USD/COP",
                })

        if not rows:
            logger.warning("No valid bars fetched")
            return None

        df_new = pd.DataFrame(rows)
        df_new["time"] = pd.to_datetime(df_new["time"]).dt.tz_localize("America/Bogota")

        # Append to existing
        if DAILY_OHLCV_PATH.exists():
            df_existing = pd.read_parquet(DAILY_OHLCV_PATH)
            if df_existing.index.name == "time":
                df_existing = df_existing.reset_index()
            df_existing["time"] = pd.to_datetime(df_existing["time"])
            if df_existing["time"].dt.tz is None:
                df_existing["time"] = df_existing["time"].dt.tz_localize("America/Bogota")

            existing_dates = set(df_existing["time"].dt.normalize().dt.date)
            mask = ~df_new["time"].dt.normalize().dt.date.isin(existing_dates)
            df_append = df_new[mask]

            if len(df_append) > 0:
                common = [c for c in df_existing.columns if c in df_append.columns]
                df_combined = pd.concat([df_existing[common], df_append[common]], ignore_index=True)
                df_combined = df_combined.sort_values("time").drop_duplicates(subset=["time"], keep="last")
                df_combined = df_combined.set_index("time")
                df_combined.to_parquet(DAILY_OHLCV_PATH)
                logger.info(f"  Appended {len(df_append)} new daily bars")
            else:
                logger.info("  No new daily bars to append")

        return df_new

    except Exception as e:
        logger.error(f"Failed to fetch daily OHLCV: {e}")
        return None


# =============================================================================
# STEP 2: UPDATE MACRO (DXY + WTI)
# =============================================================================

def update_macro() -> None:
    """Fetch latest DXY + WTI from FRED."""
    if not FRED_KEY:
        logger.warning("No FRED_API_KEY, skipping macro update")
        return

    start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    end = datetime.now().strftime("%Y-%m-%d")

    for series_id, col_name in [("DTWEXBGS", "FXRT_INDEX_DXY_USA_D_DXY"), ("DCOILWTICO", "COMM_OIL_WTI_GLB_D_WTI")]:
        try:
            params = {
                "series_id": series_id,
                "api_key": FRED_KEY,
                "file_type": "json",
                "observation_start": start,
                "observation_end": end,
            }
            resp = requests.get(FRED_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            values = [(pd.Timestamp(o["date"]), float(o["value"]))
                      for o in data.get("observations", []) if o["value"] != "."]

            if not values or not MACRO_PATH.exists():
                continue

            df_macro = pd.read_parquet(MACRO_PATH)
            date_col = "__date__"
            df_macro = df_macro.reset_index()
            # After reset_index on unnamed DatetimeIndex, first col is "index"
            actual_first = df_macro.columns[0]
            df_macro = df_macro.rename(columns={actual_first: date_col})
            df_macro[date_col] = pd.to_datetime(df_macro[date_col]).dt.tz_localize(None).dt.normalize()

            for d, v in values:
                d = d.normalize()
                mask = df_macro[date_col] == d
                if mask.any():
                    df_macro.loc[mask, col_name] = v
                else:
                    new_row = {date_col: d, col_name: v}
                    df_macro = pd.concat([df_macro, pd.DataFrame([new_row])], ignore_index=True)

            df_macro = df_macro.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
            if col_name in df_macro.columns:
                df_macro[col_name] = df_macro[col_name].ffill()
            df_macro = df_macro.set_index(date_col)
            df_macro.index.name = None
            df_macro.to_parquet(MACRO_PATH)
            logger.info(f"  Updated {col_name}: {len(values)} observations")

        except Exception as e:
            logger.warning(f"Failed to update {series_id}: {e}")


# =============================================================================
# STEP 3-4: GENERATE SIGNAL
# =============================================================================

def generate_signal() -> Optional[Dict]:
    """Build features, train models, generate today's signal."""
    from sklearn.preprocessing import StandardScaler
    from src.forecasting.data_contracts import FEATURE_COLUMNS
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.contracts import MODEL_IDS, get_horizon_config
    from src.forecasting.vol_targeting import (
        VolTargetConfig, compute_vol_target_signal, compute_realized_vol,
    )

    # Load data
    df_ohlcv = pd.read_parquet(DAILY_OHLCV_PATH)
    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    # Load macro (unnamed DatetimeIndex — first col after reset_index is "index")
    df_macro = pd.read_parquet(MACRO_PATH).reset_index()
    actual_first = df_macro.columns[0]
    df_macro = df_macro.rename(columns={actual_first: "date"})
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_map = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
    }
    df_m = df_macro[["date"] + list(macro_map.keys())].copy()
    df_m.rename(columns=macro_map, inplace=True)
    df_m = df_m.sort_values("date")
    df_m["dxy_close_lag1"] = df_m["dxy_close_lag1"].shift(1)
    df_m["oil_close_lag1"] = df_m["oil_close_lag1"].shift(1)

    df = pd.merge_asof(df_ohlcv.sort_values("date"), df_m.sort_values("date"), on="date", direction="backward")

    # Build features
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
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()

    # Build target for training
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    feature_cols = list(FEATURE_COLUMNS)
    valid = df[feature_cols].notna().all(axis=1) & df["target_return_1d"].notna()
    df_clean = df[valid].reset_index(drop=True)

    if len(df_clean) < 200:
        logger.error(f"Not enough data: {len(df_clean)} rows")
        return None

    # Train on all data except last row, predict last
    X = df_clean[feature_cols].values.astype(np.float64)
    y = df_clean["target_return_1d"].values.astype(np.float64)

    # For prediction: use ALL features including today (last row with features)
    df_all_features = df[df[feature_cols].notna().all(axis=1)].reset_index(drop=True)
    X_latest = df_all_features[feature_cols].iloc[-1:].values.astype(np.float64)
    base_price = float(df_all_features["close"].iloc[-1])
    inference_date = df_all_features["date"].iloc[-1]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_latest_scaled = scaler.transform(X_latest)

    # Train models
    h_config = get_horizon_config(1)
    linear_models = {"ridge", "bayesian_ridge", "ard"}
    catboost_models = {"catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}

    preds = {}
    for model_id in MODEL_IDS:
        try:
            if model_id in linear_models:
                params = None
            elif model_id in catboost_models:
                params = {"iterations": h_config.get("n_estimators", 50), "depth": h_config.get("max_depth", 3),
                          "learning_rate": h_config.get("learning_rate", 0.05), "verbose": False, "allow_writing_files": False}
            elif model_id in hybrid_models and "catboost" in model_id:
                params = {"iterations": h_config.get("n_estimators", 50), "depth": h_config.get("max_depth", 3),
                          "learning_rate": h_config.get("learning_rate", 0.05), "verbose": False, "allow_writing_files": False}
            else:
                params = h_config

            model = ModelFactory.create(model_id, params=params, horizon=1)
            if model.requires_scaling:
                model.fit(X_scaled, y)
                pred = float(model.predict(X_latest_scaled)[0])
            else:
                model.fit(X, y)
                pred = float(model.predict(X_latest)[0])
            preds[model_id] = pred
        except Exception as e:
            logger.warning(f"  {model_id}: {e}")

    if len(preds) < 3:
        logger.error(f"Only {len(preds)} models succeeded, need >= 3")
        return None

    # Top-3 ensemble
    sorted_models = sorted(preds.keys(), key=lambda m: abs(preds[m]), reverse=True)
    top3 = sorted_models[:3]
    ensemble_pred = float(np.mean([preds[m] for m in top3]))
    direction = 1 if ensemble_pred > 0 else -1

    # Vol-target
    vol_config = VolTargetConfig(target_vol=0.15, max_leverage=2.0, min_leverage=0.5, vol_lookback=21, vol_floor=0.05)
    recent_rets = df_all_features["return_1d"].iloc[-21:].values
    realized_vol = compute_realized_vol(recent_rets, 21)
    vt = compute_vol_target_signal(
        forecast_direction=direction,
        forecast_return=ensemble_pred,
        realized_vol_21d=realized_vol,
        config=vol_config,
        date=str(inference_date.date()),
    )

    predicted_price = base_price * np.exp(ensemble_pred)

    signal = {
        "date": str(inference_date.date()),
        "direction": direction,
        "leverage": round(vt.clipped_leverage, 2),
        "ensemble_pred": round(ensemble_pred, 6),
        "top3_models": top3,
        "base_price": round(base_price, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_return_pct": round(ensemble_pred * 100, 4),
        "realized_vol_21d": round(realized_vol, 4),
        "n_models": len(preds),
        "all_predictions": {k: round(v, 6) for k, v in preds.items()},
        "generated_at": datetime.now().isoformat(),
    }

    logger.info(f"  Signal: dir={direction}, lev={vt.clipped_leverage:.2f}, "
                f"pred={ensemble_pred*100:+.4f}%, price={base_price:.0f}→{predicted_price:.0f}")

    return signal


# =============================================================================
# STEP 5: SAVE SIGNAL
# =============================================================================

def save_signal(signal: Dict) -> None:
    """Save latest signal JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SIGNAL_PATH, "w") as f:
        json.dump(signal, f, indent=2)
    logger.info(f"  Signal saved: {SIGNAL_PATH}")


# =============================================================================
# STEP 7: REGENERATE PRODUCTION PNGs
# =============================================================================

def regenerate_production_pngs() -> None:
    """Run backtest_2026_production.py to regenerate images."""
    script = PROJECT_ROOT / "scripts" / "backtest_2026_production.py"
    if not script.exists():
        logger.warning("backtest_2026_production.py not found, skipping PNG regeneration")
        return

    logger.info("Regenerating production PNGs...")
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("  Production PNGs regenerated successfully")
        else:
            logger.warning(f"  PNG regeneration failed: {result.stderr[-500:]}")
    except subprocess.TimeoutExpired:
        logger.warning("  PNG regeneration timed out (10 min)")
    except Exception as e:
        logger.warning(f"  PNG regeneration error: {e}")


# =============================================================================
# STEP 8: REGENERATE WEEKLY FORECASTS (Sunday only)
# =============================================================================

def regenerate_weekly_forecasts() -> None:
    """Run generate_weekly_forecasts.py if today is Sunday."""
    if datetime.now().weekday() != 6:  # 6 = Sunday
        return

    script = PROJECT_ROOT / "scripts" / "generate_weekly_forecasts.py"
    if not script.exists():
        logger.warning("generate_weekly_forecasts.py not found")
        return

    logger.info("Sunday detected — regenerating weekly forecast PNGs...")
    try:
        result = subprocess.run(
            [sys.executable, str(script), "--num-weeks", "7"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            logger.info("  Weekly forecasts regenerated")
        else:
            logger.warning(f"  Weekly forecast regen failed: {result.stderr[-500:]}")
    except Exception as e:
        logger.warning(f"  Weekly forecast error: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Daily production runner")
    parser.add_argument("--dry-run", action="store_true", help="Generate signal only, no trades/PNGs")
    parser.add_argument("--force-png", action="store_true", help="Force PNG regeneration")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch (use existing)")
    args = parser.parse_args()

    # Check approval status — override to dry-run if not approved
    is_approved = check_approval()
    if not is_approved and not args.dry_run:
        logger.info("  Overriding to DRY-RUN mode (strategy not approved)")
        args.dry_run = True

    # Load direction filter from config
    direction_filter = load_direction_filter()

    logger.info("=" * 60)
    logger.info("  DAILY PRODUCTION RUNNER")
    logger.info(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    logger.info(f"  Approved: {is_approved}")
    logger.info(f"  Direction filter: {direction_filter}")
    logger.info("=" * 60)

    t0 = time.time()

    # Step 1: Fetch daily bar
    if not args.skip_fetch:
        logger.info("\n[Step 1] Fetching daily OHLCV...")
        fetch_today_daily()

        # Step 2: Update macro
        logger.info("\n[Step 2] Updating macro (DXY + WTI)...")
        update_macro()
    else:
        logger.info("\n[Steps 1-2] Skipped (--skip-fetch)")

    # Step 3-4: Generate signal
    logger.info("\n[Step 3-4] Generating signal...")
    signal = generate_signal()

    if signal:
        # Direction filter check
        if should_skip_signal(signal["direction"], direction_filter):
            dir_str = "LONG" if signal["direction"] == 1 else "SHORT"
            logger.info(
                f"\n[FILTERED] {dir_str} signal skipped (direction_filter={direction_filter})"
            )
            signal["filtered"] = True
            signal["filter_reason"] = direction_filter

        # Step 5: Save signal (always save, even if filtered — for tracking)
        logger.info("\n[Step 5] Saving signal...")
        save_signal(signal)

        if signal.get("filtered"):
            logger.info("  Signal was filtered — skipping trade execution and PNGs")
        elif not args.dry_run:
            # Step 7: Regenerate PNGs
            if args.force_png or datetime.now().hour >= 12:
                logger.info("\n[Step 7] Regenerating production PNGs...")
                regenerate_production_pngs()

            # Step 8: Weekly forecasts (Sunday)
            logger.info("\n[Step 8] Checking for Sunday weekly forecast regen...")
            regenerate_weekly_forecasts()
    else:
        logger.error("Signal generation failed!")

    elapsed = time.time() - t0
    logger.info(f"\nDone in {elapsed:.1f}s")

    # Print signal summary
    if signal:
        dir_label = 'LONG' if signal['direction'] == 1 else 'SHORT'
        filtered = signal.get('filtered', False)
        print(f"\n{'='*50}")
        print(f"  TODAY'S SIGNAL: {signal['date']}")
        if filtered:
            print(f"  *** FILTERED ({direction_filter}) — not executed ***")
        print(f"{'='*50}")
        print(f"  Direction:  {dir_label}{' [FILTERED]' if filtered else ''}")
        print(f"  Leverage:   {signal['leverage']:.2f}x")
        print(f"  Prediction: {signal['predicted_return_pct']:+.4f}%")
        print(f"  Price:      ${signal['base_price']:,.0f} → ${signal['predicted_price']:,.0f}")
        print(f"  Top 3:      {', '.join(signal['top3_models'])}")
        print(f"  Models:     {signal['n_models']}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
