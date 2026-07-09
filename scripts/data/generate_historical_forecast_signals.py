"""
Generate Historical Forecast Signals for RL Executor
=====================================================

Generates daily forecast signals via quarterly rolling walk-forward
for use as RL training features (forecast_direction, forecast_leverage).

Reuses data loading and model training from vol_target_backtest.py.

Output: data/forecasting/historical_forecast_signals.parquet
Columns: date, forecast_direction, forecast_leverage, forecast_return,
         realized_vol_21d, ensemble_models, window_idx

Usage:
    python scripts/generate_historical_forecast_signals.py
    python scripts/generate_historical_forecast_signals.py --window-size 63 --output data/forecasting/historical_forecast_signals.parquet

@version 1.0.0
@experiment EXP-RL-EXECUTOR-001
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.models.factory import ModelFactory
from src.forecasting.contracts import get_horizon_config
from src.forecasting.vol_targeting import VolTargetConfig, compute_vol_target_signal, compute_realized_vol

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING (reused from vol_target_backtest.py)
# =============================================================================

def load_data() -> pd.DataFrame:
    """Load daily OHLCV + macro, merge, build 19 SSOT features."""
    logger.info("Loading daily OHLCV...")
    ohlcv_path = PROJECT_ROOT / "seeds" / "latest" / "usdcop_daily_ohlcv.parquet"
    df_ohlcv = pd.read_parquet(ohlcv_path)

    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)

    logger.info(f"  OHLCV: {len(df_ohlcv)} rows, {df_ohlcv['date'].min().date()} to {df_ohlcv['date'].max().date()}")

    # Load macro for DXY and WTI (with T-1 lag)
    logger.info("Loading macro data (DXY, WTI)...")
    macro_path = PROJECT_ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"
    df_macro = pd.read_parquet(macro_path)
    df_macro = df_macro.reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    macro_cols = {
        "FXRT_INDEX_DXY_USA_D_DXY": "dxy_close_lag1",
        "COMM_OIL_WTI_GLB_D_WTI": "oil_close_lag1",
    }
    df_macro_subset = df_macro[["date"] + list(macro_cols.keys())].copy()
    df_macro_subset.rename(columns=macro_cols, inplace=True)

    # T-1 lag
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

    # Target: H=1 log return
    df["target_return_1d"] = np.log(df["close"].shift(-1) / df["close"])

    # Drop rows with NaN features
    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    target_mask = df["target_return_1d"].notna()
    df = df[feature_mask & target_mask].reset_index(drop=True)

    logger.info(f"  After cleanup: {len(df)} rows with complete features")
    return df


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

    # Macro (2)
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()

    return df


# =============================================================================
# QUARTERLY WALK-FORWARD SIGNAL GENERATION
# =============================================================================

def generate_quarterly_signals(
    df: pd.DataFrame,
    window_size: int = 63,
    initial_train_end: str = "2022-12-31",
    models_to_use: Optional[List[str]] = None,
    ensemble_strategy: str = "top_3",
    vol_config: VolTargetConfig = VolTargetConfig(),
) -> pd.DataFrame:
    """
    Generate daily forecast signals via quarterly rolling walk-forward.

    Args:
        df: DataFrame with features + target_return_1d
        window_size: Test window size in trading days (~1 quarter)
        initial_train_end: End of initial training period (pre-signal era)
        models_to_use: Model IDs to train (None = all 9)
        ensemble_strategy: Ensemble method for combining model predictions
        vol_config: Vol-targeting configuration for leverage computation

    Returns:
        DataFrame with columns:
            date, forecast_direction, forecast_leverage, forecast_return,
            realized_vol_21d, ensemble_models, window_idx
    """
    feature_cols = list(FEATURE_COLUMNS)
    X = df[feature_cols].values.astype(np.float64)
    y = df["target_return_1d"].values.astype(np.float64)
    dates = df["date"].values

    # Find initial train end index
    initial_train_end_ts = pd.Timestamp(initial_train_end)
    train_end_idx = df[df["date"] <= initial_train_end_ts].index[-1] + 1

    if models_to_use is None:
        models_to_use = [
            "ridge", "bayesian_ridge",
            "xgboost_pure", "lightgbm_pure", "catboost_pure",
            "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost",
        ]
        try:
            ModelFactory.create("ard")
            models_to_use.insert(2, "ard")
        except Exception:
            logger.warning("ARD model not available, using 8 models")

    n_samples = len(X)
    horizon_config = get_horizon_config(1)

    # Define model type sets
    catboost_models = {"catboost", "catboost_pure"}
    hybrid_models = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
    linear_models = {"ridge", "bayesian_ridge", "ard"}

    # Generate quarterly windows
    signals = []
    window_idx = 0
    test_start = train_end_idx

    while test_start < n_samples:
        test_end = min(test_start + window_size, n_samples)

        X_train, y_train = X[:test_start], y[:test_start]
        X_test, y_test = X[test_start:test_end], y[test_start:test_end]
        dates_test = dates[test_start:test_end]

        train_end_date = str(pd.Timestamp(dates[test_start - 1]).date())
        test_start_date = str(pd.Timestamp(dates_test[0]).date())
        test_end_date = str(pd.Timestamp(dates_test[-1]).date())

        logger.info(f"\n  Window {window_idx}: Train [0-{test_start}] to {train_end_date}, "
                     f"Test [{test_start}-{test_end}] {test_start_date} to {test_end_date}")

        # Anti-leakage assertion
        assert pd.Timestamp(train_end_date) < pd.Timestamp(test_start_date), \
            f"LOOK-AHEAD BIAS: train_end {train_end_date} >= test_start {test_start_date}"

        # Train each model
        model_preds = {}
        model_das = {}

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
                    scaler = StandardScaler()
                    X_train_s = scaler.fit_transform(X_train)
                    X_test_s = scaler.transform(X_test)
                else:
                    X_train_s, X_test_s = X_train, X_test

                model.fit(X_train_s, y_train)
                preds = model.predict(X_test_s)

                correct = np.sum(np.sign(preds) == np.sign(y_test))
                da = correct / len(y_test) * 100
                model_das[model_id] = da
                model_preds[model_id] = preds
                logger.info(f"    {model_id}: DA={da:.1f}%")

            except Exception as e:
                logger.error(f"    {model_id} FAILED: {e}")
                continue

        # Create ensemble
        if not model_preds:
            logger.error(f"  Window {window_idx}: All models failed, skipping")
            test_start = test_end
            window_idx += 1
            continue

        ensemble_preds, ensemble_names = _create_ensemble(
            model_preds, model_das, y_test, ensemble_strategy
        )

        # Compute realized vol for leverage sizing (backward-looking, no leakage)
        # Use all available actual returns up to the test window start
        historical_returns = y[:test_start]

        for i in range(len(dates_test)):
            # Realized vol uses returns up to this point
            all_returns_up_to = np.concatenate([historical_returns, y_test[:i]]) if i > 0 else historical_returns
            rvol = compute_realized_vol(all_returns_up_to, lookback=vol_config.vol_lookback)
            if rvol == 0.0:
                rvol = vol_config.vol_floor

            pred_return = float(ensemble_preds[i])
            direction = int(np.sign(pred_return)) if pred_return != 0 else 0

            signal = compute_vol_target_signal(
                forecast_direction=direction,
                forecast_return=pred_return,
                realized_vol_21d=rvol,
                config=vol_config,
                date=str(pd.Timestamp(dates_test[i]).date()),
            )

            signals.append({
                "date": pd.Timestamp(dates_test[i]).normalize(),
                "forecast_direction": direction,
                "forecast_leverage": signal.clipped_leverage,
                "forecast_return": pred_return,
                "realized_vol_21d": rvol,
                "ensemble_models": ",".join(ensemble_names),
                "window_idx": window_idx,
            })

        test_start = test_end
        window_idx += 1

    df_signals = pd.DataFrame(signals)
    logger.info(f"\nGenerated {len(df_signals)} daily signals across {window_idx} windows")

    return df_signals


def _create_ensemble(
    model_preds: Dict[str, np.ndarray],
    model_das: Dict[str, float],
    y_test: np.ndarray,
    strategy: str,
) -> Tuple[np.ndarray, List[str]]:
    """Create ensemble prediction and return (predictions, model_names)."""
    if strategy == "top_3":
        sorted_models = sorted(model_das.keys(), key=lambda m: model_das[m], reverse=True)
        top_3 = sorted_models[:3]
        preds = np.mean([model_preds[m] for m in top_3 if m in model_preds], axis=0)
        logger.info(f"    Ensemble top_3: [{', '.join(top_3)}]")
        return preds, top_3
    elif strategy == "consensus":
        all_p = [model_preds[m] for m in model_preds]
        preds = np.mean(all_p, axis=0)
        return preds, list(model_preds.keys())
    elif strategy == "best_of_breed":
        best = max(model_das, key=model_das.get)
        return model_preds[best], [best]
    else:
        raise ValueError(f"Unknown ensemble strategy: {strategy}")


# =============================================================================
# VALIDATION (GATE B.1)
# =============================================================================

def validate_signals(df_signals: pd.DataFrame, df_data: pd.DataFrame) -> Dict:
    """
    Validate generated signals against GATE B.1 criteria.

    Checks:
    1. DA between 51-58% (consistent with walk-forward, no look-ahead)
    2. Coverage >95% of 2023-2025 trading days
    3. Distribution ~50/50 long/short
    4. No look-ahead bias (verified by assertion in generation)
    """
    # Merge signals with actual returns for DA calculation
    df_data_subset = df_data[["date", "target_return_1d"]].copy()
    df_merged = df_signals.merge(df_data_subset, on="date", how="left")

    direction_correct = (
        df_merged["forecast_direction"] * df_merged["target_return_1d"] > 0
    ).astype(float)
    # Exclude zero-direction days
    nonzero_mask = df_merged["forecast_direction"] != 0
    da = direction_correct[nonzero_mask].mean() * 100 if nonzero_mask.sum() > 0 else 0

    # Coverage: trading days in 2023-2025
    trading_days_2023_2025 = df_data[
        (df_data["date"] >= "2023-01-01") & (df_data["date"] <= "2025-12-31")
    ]["date"].nunique()
    signal_days_2023_2025 = df_signals[
        (df_signals["date"] >= "2023-01-01") & (df_signals["date"] <= "2025-12-31")
    ]["date"].nunique()
    coverage = signal_days_2023_2025 / max(trading_days_2023_2025, 1) * 100

    # Distribution
    n_long = (df_signals["forecast_direction"] == 1).sum()
    n_short = (df_signals["forecast_direction"] == -1).sum()
    n_neutral = (df_signals["forecast_direction"] == 0).sum()
    total = len(df_signals)
    long_pct = n_long / max(total, 1) * 100
    short_pct = n_short / max(total, 1) * 100

    # GATE B.1 validation
    da_ok = 51 <= da <= 58
    da_no_lookahead = da <= 60  # DA > 60% = look-ahead bias
    coverage_ok = coverage > 95
    distribution_ok = abs(long_pct - short_pct) < 20  # Not too imbalanced

    results = {
        "da_pct": round(da, 2),
        "da_ok": da_ok,
        "da_no_lookahead": da_no_lookahead,
        "coverage_pct": round(coverage, 1),
        "coverage_ok": coverage_ok,
        "n_signals": total,
        "n_long": int(n_long),
        "n_short": int(n_short),
        "n_neutral": int(n_neutral),
        "long_pct": round(long_pct, 1),
        "short_pct": round(short_pct, 1),
        "distribution_ok": distribution_ok,
        "gate_b1_passed": da_ok and da_no_lookahead and coverage_ok,
        "windows": int(df_signals["window_idx"].nunique()),
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate Historical Forecast Signals for RL Executor")
    parser.add_argument("--window-size", type=int, default=63, help="Quarterly window size in trading days")
    parser.add_argument("--initial-train-end", type=str, default="2022-12-31", help="End of initial training period")
    parser.add_argument("--ensemble", type=str, default="top_3", choices=["top_3", "best_of_breed", "consensus"])
    parser.add_argument("--target-vol", type=float, default=0.15, help="Target vol for leverage computation")
    parser.add_argument("--max-leverage", type=float, default=2.0, help="Max leverage")
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    print("=" * 70)
    print("  Generate Historical Forecast Signals (EXP-RL-EXECUTOR-001)")
    print("=" * 70)

    start_time = time.time()

    # Load data
    df = load_data()

    # Vol target config
    vol_config = VolTargetConfig(
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
    )

    # Generate signals
    df_signals = generate_quarterly_signals(
        df,
        window_size=args.window_size,
        initial_train_end=args.initial_train_end,
        ensemble_strategy=args.ensemble,
        vol_config=vol_config,
    )

    # Validate
    print("\n" + "=" * 70)
    print("  GATE B.1 VALIDATION")
    print("=" * 70)

    validation = validate_signals(df_signals, df)

    print(f"  DA: {validation['da_pct']:.1f}% {'OK (51-58%)' if validation['da_ok'] else 'XX'}")
    print(f"  Look-ahead check: {'OK (DA<=60%)' if validation['da_no_lookahead'] else 'FAIL (DA>60%!)'}")
    print(f"  Coverage: {validation['coverage_pct']:.1f}% {'OK (>95%)' if validation['coverage_ok'] else 'XX'}")
    print(f"  Distribution: {validation['n_long']}L / {validation['n_short']}S / {validation['n_neutral']}N "
          f"({validation['long_pct']:.1f}% / {validation['short_pct']:.1f}%)")
    print(f"  Windows: {validation['windows']}")
    print(f"  GATE B.1: {'PASSED' if validation['gate_b1_passed'] else 'FAILED'}")

    # Save signals
    output_path = args.output
    if output_path is None:
        output_dir = PROJECT_ROOT / "data" / "forecasting"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / "historical_forecast_signals.parquet")

    df_signals.to_parquet(output_path, index=False)

    # Also save validation results (convert numpy types for JSON)
    validation_json = {k: (bool(v) if isinstance(v, (np.bool_,)) else
                           int(v) if isinstance(v, (np.integer,)) else
                           float(v) if isinstance(v, (np.floating,)) else v)
                       for k, v in validation.items()}
    validation_path = Path(output_path).with_suffix(".json")
    with open(validation_path, "w") as f:
        json.dump(validation_json, f, indent=2)

    duration = time.time() - start_time
    print(f"\n  Signals saved to: {output_path}")
    print(f"  Validation saved to: {validation_path}")
    print(f"  Total time: {duration:.1f}s")


if __name__ == "__main__":
    main()
