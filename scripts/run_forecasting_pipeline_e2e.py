#!/usr/bin/env python3
"""
Forecasting Pipeline End-to-End Execution
==========================================

Executes the complete forecasting pipeline:
1. L0: Data acquisition (USDCOP daily from yfinance)
2. L1: Feature engineering
3. Data quality analysis
4. L3: Model training (9 models × 7 horizons)
5. L5: Inference simulation
6. Results report

This script runs LOCALLY without Airflow/PostgreSQL dependencies.

Usage:
    python scripts/run_forecasting_pipeline_e2e.py
    python scripts/run_forecasting_pipeline_e2e.py --days 365
    python scripts/run_forecasting_pipeline_e2e.py --quick  # Fewer models

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import argparse
import json
import logging
import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS FROM CONTRACTS (SSOT)
# =============================================================================

from src.forecasting.contracts import (
    HORIZONS,
    MODEL_IDS,
    MODEL_DEFINITIONS,
    HORIZON_CATEGORIES,
    get_horizon_config,
    FORECASTING_CONTRACT_VERSION,
)

from src.forecasting.data_contracts import (
    FEATURE_COLUMNS,
    TARGET_HORIZONS,
    TARGET_COLUMN,
    DATA_CONTRACT_VERSION,
)

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "forecasting" / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# =============================================================================
# L0: DATA ACQUISITION
# =============================================================================

def fetch_usdcop_data(days: int = 365 * 3) -> pd.DataFrame:
    """
    L0: Fetch daily USDCOP data from yfinance.

    Contract: CTR-FORECAST-DATA-001
    """
    logger.info("=" * 60)
    logger.info("L0: DATA ACQUISITION")
    logger.info("=" * 60)

    try:
        import yfinance as yf

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        logger.info(f"Fetching USDCOP from {start_date.date()} to {end_date.date()}")

        ticker = yf.Ticker("USDCOP=X")
        df = ticker.history(start=start_date, end=end_date, interval="1d")

        if df.empty:
            raise ValueError("No data returned from yfinance")

        # Rename columns to match contract
        df = df.reset_index()
        df = df.rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Remove timezone info
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.sort_values("date").reset_index(drop=True)

        # Validate data
        df = validate_ohlcv(df)

        logger.info(f"✅ L0 Complete: {len(df)} records fetched")
        logger.info(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"   Price range: {df['close'].min():.2f} - {df['close'].max():.2f} COP")

        return df[["date", "open", "high", "low", "close", "volume"]]

    except ImportError:
        logger.error("yfinance not installed. Install with: pip install yfinance")
        raise
    except Exception as e:
        logger.error(f"L0 Failed: {e}")
        raise


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLCV data according to contract."""
    initial_count = len(df)

    # Remove rows with missing data
    df = df.dropna(subset=["date", "open", "high", "low", "close"])

    # Validate price range (3000-6000 COP for USDCOP)
    df = df[(df["close"] >= 3000) & (df["close"] <= 6000)]

    # Validate OHLC consistency
    df = df[
        (df["high"] >= df["low"]) &
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"])
    ]

    removed = initial_count - len(df)
    if removed > 0:
        logger.warning(f"   Removed {removed} invalid records during validation")

    return df


# =============================================================================
# L1: FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    L1: Build forecasting features from OHLCV data.

    Contract: CTR-FORECAST-DATA-001
    Features: 19 columns as defined in FEATURE_COLUMNS
    """
    logger.info("=" * 60)
    logger.info("L1: FEATURE ENGINEERING")
    logger.info("=" * 60)

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # === RETURNS ===
    df["return_1d"] = df["close"].pct_change(1)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # === VOLATILITY ===
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # === TECHNICAL INDICATORS ===
    # RSI (14-day)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 0.0001)
    df["rsi_14d"] = 100 - (100 / (1 + rs))

    # Moving Average Ratios
    df["ma_20d"] = df["close"].rolling(20).mean()
    df["ma_50d"] = df["close"].rolling(50).mean()
    df["ma_ratio_20d"] = df["close"] / df["ma_20d"]
    df["ma_ratio_50d"] = df["close"] / df["ma_50d"]

    # === CALENDAR FEATURES ===
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # === MACRO PLACEHOLDERS (lagged to avoid lookahead) ===
    # In production, these would come from external sources
    df["dxy_close_lag1"] = 104.5 + np.random.randn(len(df)) * 0.5  # Placeholder
    df["oil_close_lag1"] = 75.0 + np.random.randn(len(df)) * 2.0   # Placeholder

    # === TARGET COLUMNS (future prices) ===
    for horizon in TARGET_HORIZONS:
        df[f"target_{horizon}d"] = df["close"].shift(-horizon)
        df[f"target_return_{horizon}d"] = df["close"].pct_change(horizon).shift(-horizon)

    # Drop rows with NaN from rolling calculations
    min_lookback = 50  # MA50 needs 50 days
    df = df.iloc[min_lookback:].reset_index(drop=True)

    logger.info(f"✅ L1 Complete: {len(df)} rows with features")
    logger.info(f"   Features: {len([c for c in df.columns if c not in ['date'] and not c.startswith('target')])}")
    logger.info(f"   Targets: {len([c for c in df.columns if c.startswith('target')])}")

    return df


# =============================================================================
# DATA QUALITY ANALYSIS
# =============================================================================

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform comprehensive data quality analysis.

    Checks:
    - Missing values
    - Data types
    - Value ranges
    - Outliers
    - Stationarity indicators
    - Feature correlations
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY ANALYSIS")
    logger.info("=" * 60)

    report = {
        "timestamp": datetime.now().isoformat(),
        "contract_version": DATA_CONTRACT_VERSION,
        "total_rows": len(df),
        "date_range": {
            "start": str(df["date"].min().date()),
            "end": str(df["date"].max().date()),
            "trading_days": len(df),
        },
        "missing_values": {},
        "statistics": {},
        "outliers": {},
        "quality_score": 0.0,
    }

    # === MISSING VALUES ===
    logger.info("\n📊 Missing Values Analysis:")
    feature_cols = [c for c in df.columns if c != "date" and not c.startswith("target")]

    for col in feature_cols:
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        report["missing_values"][col] = {
            "count": int(null_count),
            "percentage": round(null_pct, 2),
        }
        if null_count > 0:
            logger.info(f"   {col}: {null_count} ({null_pct:.1f}%)")

    total_nulls = sum(v["count"] for v in report["missing_values"].values())
    if total_nulls == 0:
        logger.info("   ✅ No missing values in features")

    # === STATISTICS ===
    logger.info("\n📊 Feature Statistics:")
    stats_cols = ["close", "return_1d", "volatility_20d", "rsi_14d"]

    for col in stats_cols:
        if col in df.columns:
            stats = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "skewness": float(df[col].skew()),
                "kurtosis": float(df[col].kurtosis()),
            }
            report["statistics"][col] = stats
            logger.info(f"   {col}:")
            logger.info(f"      Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            logger.info(f"      Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

    # === OUTLIER DETECTION (IQR method) ===
    logger.info("\n📊 Outlier Analysis (IQR method):")
    outlier_cols = ["return_1d", "return_5d", "volatility_20d"]

    for col in outlier_cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower) | (df[col] > upper)][col]

            report["outliers"][col] = {
                "count": len(outliers),
                "percentage": round(len(outliers) / len(df) * 100, 2),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
            logger.info(f"   {col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")

    # === TARGET COVERAGE ===
    logger.info("\n📊 Target Coverage:")
    for horizon in TARGET_HORIZONS:
        target_col = f"target_return_{horizon}d"
        if target_col in df.columns:
            valid = df[target_col].notna().sum()
            pct = valid / len(df) * 100
            logger.info(f"   H{horizon:2d}: {valid:4d} samples ({pct:.1f}%)")

    # === CORRELATION ANALYSIS ===
    logger.info("\n📊 Feature-Target Correlations (1d return):")
    if "target_return_1d" in df.columns:
        correlations = {}
        for col in feature_cols[:10]:  # Top 10 features
            if col in df.columns and df[col].notna().sum() > 10:
                corr = df[col].corr(df["target_return_1d"])
                correlations[col] = float(corr) if not np.isnan(corr) else 0.0

        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        for col, corr in sorted_corr[:5]:
            logger.info(f"   {col}: {corr:.4f}")
        report["correlations"] = correlations

    # === QUALITY SCORE ===
    # Calculate overall quality score (0-100)
    null_score = 100 - (total_nulls / (len(df) * len(feature_cols)) * 100)
    outlier_score = 100 - sum(v["percentage"] for v in report["outliers"].values()) / len(report["outliers"]) if report["outliers"] else 100
    coverage_score = df[[f"target_return_{h}d" for h in TARGET_HORIZONS[:3]]].notna().mean().mean() * 100

    quality_score = (null_score * 0.4 + outlier_score * 0.3 + coverage_score * 0.3)
    report["quality_score"] = round(quality_score, 2)

    logger.info(f"\n📊 QUALITY SCORE: {quality_score:.1f}/100")
    if quality_score >= 90:
        logger.info("   ✅ Excellent data quality")
    elif quality_score >= 75:
        logger.info("   ⚠️ Good data quality with minor issues")
    else:
        logger.info("   ❌ Data quality needs attention")

    return report


# =============================================================================
# L3: MODEL TRAINING
# =============================================================================

def train_models(
    df: pd.DataFrame,
    models_to_train: Optional[List[str]] = None,
    horizons_to_train: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    L3: Train forecasting models using walk-forward validation.

    Contract: CTR-FORECASTING-001
    """
    logger.info("=" * 60)
    logger.info("L3: MODEL TRAINING")
    logger.info("=" * 60)

    from src.forecasting.models import ModelFactory
    from src.forecasting.evaluation.walk_forward import WalkForwardValidator
    from src.forecasting.evaluation.metrics import Metrics

    models_to_train = models_to_train or list(MODEL_IDS)
    horizons_to_train = horizons_to_train or list(HORIZONS)

    total_combinations = len(models_to_train) * len(horizons_to_train)
    logger.info(f"Training {len(models_to_train)} models × {len(horizons_to_train)} horizons = {total_combinations} combinations")

    # Prepare features
    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
    ]

    # Filter to available columns
    available_features = [c for c in feature_cols if c in df.columns]
    logger.info(f"Using {len(available_features)} features")

    X = df[available_features].values

    # Initialize validator
    validator = WalkForwardValidator(
        n_folds=3,
        initial_train_ratio=0.6,
        gap=5,  # 5-day gap to avoid lookahead
    )

    results = {
        "models_trained": 0,
        "total_combinations": total_combinations,
        "metrics": {},
        "best_per_horizon": {},
        "training_time_seconds": 0,
        "feature_count": len(available_features),
    }

    start_time = time.time()
    trained = 0

    for horizon in horizons_to_train:
        target_col = f"target_return_{horizon}d"

        if target_col not in df.columns:
            logger.warning(f"Target column {target_col} not found, skipping H{horizon}")
            continue

        # Get valid samples (non-NaN target)
        valid_mask = df[target_col].notna()
        X_valid = X[valid_mask]
        y_valid = df.loc[valid_mask, target_col].values

        if len(y_valid) < 100:
            logger.warning(f"H{horizon}: Only {len(y_valid)} samples, skipping")
            continue

        results["metrics"][horizon] = {}
        best_da = 0
        best_model = None

        for model_id in models_to_train:
            try:
                # Create model
                model = ModelFactory.create(model_id, params=get_horizon_config(horizon))

                # Run walk-forward validation
                wf_result = validator.validate(model, X_valid, y_valid, horizon=horizon)

                # Store metrics
                da = wf_result.avg_metrics.direction_accuracy
                rmse = wf_result.avg_metrics.rmse

                results["metrics"][horizon][model_id] = {
                    "direction_accuracy": round(da, 2),
                    "rmse": round(rmse, 6),
                    "mae": round(wf_result.avg_metrics.mae, 6),
                    "samples": len(y_valid),
                }

                # Track best
                if da > best_da:
                    best_da = da
                    best_model = model_id

                trained += 1

            except Exception as e:
                logger.warning(f"H{horizon}/{model_id}: {str(e)[:50]}")
                continue

        if best_model:
            results["best_per_horizon"][horizon] = {
                "model": best_model,
                "direction_accuracy": round(best_da, 2),
            }

        # Progress
        pct = trained / total_combinations * 100
        logger.info(f"   H{horizon:2d}: Best={best_model} (DA={best_da:.1f}%) - {pct:.0f}% complete")

    results["models_trained"] = trained
    results["training_time_seconds"] = round(time.time() - start_time, 2)

    logger.info(f"\n✅ L3 Complete: {trained}/{total_combinations} models trained")
    logger.info(f"   Training time: {results['training_time_seconds']:.1f}s")

    return results


# =============================================================================
# L5: INFERENCE SIMULATION
# =============================================================================

def run_inference(
    df: pd.DataFrame,
    training_results: Dict[str, Any],
) -> Dict[str, Any]:
    """
    L5: Run inference simulation on latest data.

    Contract: CTR-FORECASTING-001
    """
    logger.info("=" * 60)
    logger.info("L5: INFERENCE SIMULATION")
    logger.info("=" * 60)

    from src.forecasting.models import ModelFactory

    # Get latest row for inference
    latest = df.iloc[-1]
    inference_date = latest["date"]
    current_price = latest["close"]

    logger.info(f"Inference date: {inference_date.date()}")
    logger.info(f"Current price: {current_price:.2f} COP")

    # Prepare features
    feature_cols = [
        "close", "open", "high", "low",
        "return_1d", "return_5d", "return_10d", "return_20d",
        "volatility_5d", "volatility_10d", "volatility_20d",
        "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
        "day_of_week", "month", "is_month_end",
        "dxy_close_lag1", "oil_close_lag1",
    ]

    available_features = [c for c in feature_cols if c in df.columns]
    X_latest = df[available_features].iloc[-1:].values

    predictions = []

    for horizon, horizon_results in training_results.get("metrics", {}).items():
        best_model_id = training_results["best_per_horizon"].get(horizon, {}).get("model")

        if not best_model_id:
            continue

        try:
            # Create and "train" model on all data for final prediction
            model = ModelFactory.create(best_model_id)

            target_col = f"target_return_{horizon}d"
            valid_mask = df[target_col].notna()
            X_train = df.loc[valid_mask, available_features].values
            y_train = df.loc[valid_mask, target_col].values

            model.fit(X_train, y_train)

            # Predict
            pred_return = model.predict(X_latest)[0]
            pred_price = current_price * (1 + pred_return)
            direction = "UP" if pred_return > 0.0001 else "DOWN"

            predictions.append({
                "horizon": horizon,
                "model_id": best_model_id,
                "inference_date": str(inference_date.date()),
                "target_date": str((inference_date + timedelta(days=horizon)).date()),
                "base_price": round(current_price, 2),
                "predicted_price": round(pred_price, 2),
                "predicted_return_pct": round(pred_return * 100, 4),
                "direction": direction,
            })

        except Exception as e:
            logger.warning(f"H{horizon} inference failed: {e}")

    # Calculate consensus
    consensus = {}
    for horizon in HORIZONS:
        h_preds = [p for p in predictions if p["horizon"] == horizon]
        if h_preds:
            bullish = sum(1 for p in h_preds if p["direction"] == "UP")
            bearish = len(h_preds) - bullish
            avg_return = np.mean([p["predicted_return_pct"] for p in h_preds])
            consensus[horizon] = {
                "direction": "UP" if bullish > bearish else "DOWN",
                "bullish_count": bullish,
                "bearish_count": bearish,
                "avg_return_pct": round(avg_return, 4),
            }

    logger.info(f"\n✅ L5 Complete: {len(predictions)} predictions generated")

    # Show predictions
    logger.info("\n📈 FORECASTS:")
    logger.info("-" * 70)
    logger.info(f"{'Horizon':<10} {'Model':<20} {'Direction':<10} {'Return %':<12} {'Price':<12}")
    logger.info("-" * 70)

    for p in sorted(predictions, key=lambda x: x["horizon"]):
        logger.info(
            f"H{p['horizon']:<9} {p['model_id']:<20} {p['direction']:<10} "
            f"{p['predicted_return_pct']:>+10.2f}% {p['predicted_price']:>10.2f}"
        )

    return {
        "inference_date": str(inference_date.date()),
        "current_price": current_price,
        "predictions": predictions,
        "consensus": consensus,
    }


# =============================================================================
# FINAL REPORT
# =============================================================================

def generate_report(
    data_df: pd.DataFrame,
    quality_report: Dict[str, Any],
    training_results: Dict[str, Any],
    inference_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Generate final comprehensive report."""
    logger.info("=" * 60)
    logger.info("FINAL REPORT")
    logger.info("=" * 60)

    report = {
        "pipeline_version": FORECASTING_CONTRACT_VERSION,
        "execution_timestamp": datetime.now().isoformat(),
        "data": {
            "rows": len(data_df),
            "date_range": quality_report["date_range"],
            "quality_score": quality_report["quality_score"],
        },
        "training": {
            "models_trained": training_results["models_trained"],
            "best_models": training_results["best_per_horizon"],
            "training_time_seconds": training_results["training_time_seconds"],
        },
        "inference": {
            "inference_date": inference_results["inference_date"],
            "current_price": inference_results["current_price"],
            "predictions_count": len(inference_results["predictions"]),
            "consensus": inference_results["consensus"],
        },
    }

    # Summary table
    logger.info("\n📊 PERFORMANCE SUMMARY BY HORIZON:")
    logger.info("-" * 60)
    logger.info(f"{'Horizon':<10} {'Best Model':<20} {'DA %':<10} {'Prediction':<15}")
    logger.info("-" * 60)

    for horizon in sorted(training_results["best_per_horizon"].keys()):
        best = training_results["best_per_horizon"][horizon]
        pred = next((p for p in inference_results["predictions"] if p["horizon"] == horizon), None)
        pred_str = f"{pred['direction']} ({pred['predicted_return_pct']:+.2f}%)" if pred else "N/A"

        logger.info(f"H{horizon:<9} {best['model']:<20} {best['direction_accuracy']:<10.1f} {pred_str:<15}")

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\n📁 Report saved to: {report_path}")

    return report


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Forecasting Pipeline E2E")
    parser.add_argument("--days", type=int, default=365*2, help="Days of historical data")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer models)")
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("FORECASTING PIPELINE - END TO END EXECUTION")
    logger.info("=" * 60)
    logger.info(f"Contract version: {FORECASTING_CONTRACT_VERSION}")
    logger.info(f"Data contract: {DATA_CONTRACT_VERSION}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Select models for quick mode
    if args.quick:
        models_to_train = ["ridge", "xgboost_pure", "lightgbm_pure"]
        horizons_to_train = [1, 5, 10, 20]
        logger.info(f"Quick mode: {len(models_to_train)} models, {len(horizons_to_train)} horizons")
    else:
        models_to_train = list(MODEL_IDS)
        horizons_to_train = list(HORIZONS)

    try:
        # L0: Data acquisition
        df_raw = fetch_usdcop_data(days=args.days)

        # L1: Feature engineering
        df_features = build_features(df_raw)

        # Data quality analysis
        quality_report = analyze_data_quality(df_features)

        # L3: Model training
        training_results = train_models(
            df_features,
            models_to_train=models_to_train,
            horizons_to_train=horizons_to_train,
        )

        # L5: Inference
        inference_results = run_inference(df_features, training_results)

        # Final report
        report = generate_report(
            df_features,
            quality_report,
            training_results,
            inference_results,
        )

        total_time = time.time() - start_time
        logger.info(f"\n✅ PIPELINE COMPLETE in {total_time:.1f}s")

        return 0

    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
