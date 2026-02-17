"""
DAG: forecast_h1_l5_daily_inference
====================================
Daily forecasting inference: load pre-trained models from L5a, generate
fresh H=1 predictions using today's features, persist to bi.fact_forecasts.

Architecture:
    L5a (Sunday) -> trains 9 models -> saves .pkl to MODELS_DIR
                                            |
    L5b (Mon-Fri) -> loads .pkl from MODELS_DIR
                                            |
    seeds/usdcop_daily_ohlcv.parquet + MACRO_DAILY_CLEAN.parquet
                     + bi.dim_daily_usdcop (DB extension)
                                            |
                                            v
                              load_models_and_features()
                                            |
                                            v
                              generate_predictions()
                                            |
                                            v
                              persist_forecasts()
                                            |
                                            v
                              inference_summary()

Why daily?
    L5a trains models weekly (Sunday). But L5c vol-targeting runs DAILY and
    needs fresh predictions based on TODAY's features (latest close, returns,
    vol, macro). Without L5b, L5c would use stale Sunday predictions.

    L5b takes ~1 second: just loads 9 .pkl files + scaler, builds features,
    predicts, and UPSERTs.

Schedule: Mon-Fri at 18:00 UTC (13:00 COT), 30 min before L5c (13:30 COT)
Output: 9 rows in bi.fact_forecasts per inference_date (one per model, H=1)
Depends on: forecast_h1_l3_weekly_training (L3, models must exist)
Downstream: forecast_h1_l5_vol_targeting reads these predictions

Author: Trading Team
Version: 1.0.0
Date: 2026-02-15
Contract: FC-H1-L5-001
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

from contracts.dag_registry import (
    FORECAST_H1_L5_DAILY_INFERENCE,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H1_L5_DAILY_INFERENCE
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

# Project root in Docker
PROJECT_ROOT = Path('/opt/airflow')

# Models directory (same as L5a output)
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'forecasting' / 'weekly_models' / 'latest'

# 21 SSOT features (from src/forecasting/data_contracts.py)
FEATURE_COLUMNS = (
    "close", "open", "high", "low",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
    "day_of_week", "month", "is_month_end",
    "dxy_close_lag1", "oil_close_lag1",
    "vix_close_lag1", "embi_close_lag1",
)

# 9 models (same as L5a)
MODEL_IDS = (
    "ridge", "bayesian_ridge", "ard",
    "xgboost_pure", "lightgbm_pure", "catboost_pure",
    "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost",
)

# Colombia holidays 2026
COLOMBIA_HOLIDAYS_2026 = {
    "2026-01-01", "2026-01-12", "2026-03-23", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-18", "2026-06-08", "2026-06-15", "2026-06-29",
    "2026-07-20", "2026-08-07", "2026-08-17", "2026-10-12", "2026-11-02",
    "2026-11-16", "2026-12-08", "2026-12-25",
}


# =============================================================================
# TASK 1: CHECK MARKET DAY
# =============================================================================

def check_market_day(**context) -> bool:
    """
    Check if today is a trading day. Skip holidays and weekends.
    Override with dag_run.conf: {"force_run": true}
    """
    conf = context.get('dag_run', None)
    if conf and conf.conf and conf.conf.get('force_run', False):
        logger.info("[L5b] Force run enabled, skipping market day check")
        return True

    today = datetime.utcnow().date()
    today_str = today.strftime("%Y-%m-%d")

    if today.weekday() >= 5:
        logger.info(f"[L5b] {today_str} is a weekend, skipping")
        return False

    if today_str in COLOMBIA_HOLIDAYS_2026:
        logger.info(f"[L5b] {today_str} is a Colombia holiday, skipping")
        return False

    logger.info(f"[L5b] {today_str} is a trading day, proceeding")
    return True


# =============================================================================
# TASK 2: LOAD MODELS AND BUILD FEATURES
# =============================================================================

def load_models_and_features(**context) -> Dict[str, Any]:
    """
    1. Verify pre-trained .pkl models exist (from L5a weekly training).
    2. Load daily OHLCV + macro, build 19 SSOT features.
    3. Return paths + latest features for prediction.
    """
    import numpy as np
    import pandas as pd
    import joblib

    # --- Verify models exist ---
    scaler_path = MODELS_DIR / 'scaler.pkl'
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"[L5b] No scaler.pkl at {scaler_path}. "
            f"Run L5a (weekly training) first."
        )

    available_models = []
    for model_id in MODEL_IDS:
        model_path = MODELS_DIR / f'{model_id}_h1.pkl'
        if model_path.exists():
            available_models.append(model_id)
        else:
            logger.warning(f"[L5b] Model not found: {model_path}")

    if not available_models:
        raise FileNotFoundError(
            f"[L5b] No trained models found in {MODELS_DIR}. "
            f"Run L5a first."
        )

    logger.info(f"[L5b] Found {len(available_models)}/{len(MODEL_IDS)} models in {MODELS_DIR}")

    # --- Load OHLCV ---
    ohlcv_path = PROJECT_ROOT / 'seeds' / 'latest' / 'usdcop_daily_ohlcv.parquet'
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"[L5b] Daily OHLCV not found: {ohlcv_path}")

    df_ohlcv = pd.read_parquet(ohlcv_path)
    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.rename(columns={"time": "date"}, inplace=True)
    df_ohlcv["date"] = pd.to_datetime(df_ohlcv["date"]).dt.tz_localize(None).dt.normalize()
    df_ohlcv = df_ohlcv[["date", "open", "high", "low", "close"]].copy()
    df_ohlcv = df_ohlcv.sort_values("date").reset_index(drop=True)

    logger.info(
        f"[L5b] OHLCV: {len(df_ohlcv)} rows, "
        f"{df_ohlcv['date'].iloc[0].date()} to {df_ohlcv['date'].iloc[-1].date()}"
    )

    # Try to extend with latest DB data
    last_parquet_date = df_ohlcv["date"].iloc[-1]
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT date, open, high, low, close
            FROM bi.dim_daily_usdcop
            WHERE date > %s
            ORDER BY date ASC
        """, (last_parquet_date.date(),))
        db_rows = cur.fetchall()
        conn.close()

        if db_rows:
            df_db = pd.DataFrame(db_rows, columns=["date", "open", "high", "low", "close"])
            df_db["date"] = pd.to_datetime(df_db["date"])
            df_ohlcv = pd.concat([df_ohlcv, df_db], ignore_index=True)
            df_ohlcv = df_ohlcv.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            logger.info(f"[L5b] Extended with {len(db_rows)} DB rows -> {len(df_ohlcv)} total")
    except Exception as e:
        logger.warning(f"[L5b] Could not extend from DB (using parquet only): {e}")

    # --- Load Macro (DXY + WTI) ---
    macro_path = PROJECT_ROOT / 'data' / 'pipeline' / '04_cleaning' / 'output' / 'MACRO_DAILY_CLEAN.parquet'
    if not macro_path.exists():
        raise FileNotFoundError(f"[L5b] Macro data not found: {macro_path}")

    df_macro = pd.read_parquet(macro_path)
    df_macro = df_macro.reset_index()
    df_macro.rename(columns={df_macro.columns[0]: "date"}, inplace=True)
    df_macro["date"] = pd.to_datetime(df_macro["date"]).dt.tz_localize(None).dt.normalize()

    # DXY + WTI + VIX + EMBI with T-1 lag (anti-leakage)
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

    # Merge
    df = pd.merge_asof(
        df_ohlcv.sort_values("date"),
        df_macro_subset.sort_values("date"),
        on="date",
        direction="backward",
    )

    # --- Build Features ---
    df = _build_features(df)

    # Filter rows with all features present (latest row for prediction)
    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    df_pred = df[feature_mask].reset_index(drop=True)

    if len(df_pred) == 0:
        raise ValueError("[L5b] No valid feature rows after cleanup")

    latest_date = str(df_pred["date"].iloc[-1].date())
    latest_close = float(df_pred["close"].iloc[-1])

    # Save pred features temp parquet
    pred_path = str(PROJECT_ROOT / 'outputs' / 'forecasting' / 'l5b_pred_features_temp.parquet')
    Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
    df_pred.to_parquet(pred_path, index=False)

    result = {
        "pred_features_path": pred_path,
        "models_dir": str(MODELS_DIR),
        "available_models": available_models,
        "n_models": len(available_models),
        "latest_date": latest_date,
        "latest_close": latest_close,
        "n_pred_rows": len(df_pred),
    }

    context['ti'].xcom_push(key='data', value=result)
    return result


def _build_features(df):
    """Build 19 SSOT features from raw OHLCV + macro (identical to L5a)."""
    import numpy as np
    import pandas as pd

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

    # Macro (4) - already merged, just ffill gaps
    df["dxy_close_lag1"] = df["dxy_close_lag1"].ffill()
    df["oil_close_lag1"] = df["oil_close_lag1"].ffill()
    df["vix_close_lag1"] = df["vix_close_lag1"].ffill()
    df["embi_close_lag1"] = df["embi_close_lag1"].ffill()

    return df


# =============================================================================
# TASK 3: GENERATE PREDICTIONS
# =============================================================================

def generate_predictions(**context) -> Dict[str, Any]:
    """
    Generate H=1 predictions for today using pre-trained models.
    Loads .pkl models from L5a's MODELS_DIR (no training).
    """
    import numpy as np
    import pandas as pd
    import joblib

    ti = context['ti']
    data = ti.xcom_pull(key='data', task_ids='load_models_and_features')

    if not data:
        raise ValueError("[L5b] No data from load_models_and_features")

    # Load prediction features
    df_pred = pd.read_parquet(data["pred_features_path"])
    feature_cols = list(FEATURE_COLUMNS)

    # Latest features for prediction
    X_latest = df_pred[feature_cols].iloc[-1:].values.astype(np.float64)
    latest_date = data["latest_date"]
    latest_close = data["latest_close"]

    # Load scaler
    models_dir = Path(data["models_dir"])
    scaler = joblib.load(models_dir / 'scaler.pkl')
    X_latest_scaled = scaler.transform(X_latest)

    # Compute ISO week/year
    inf_date = pd.Timestamp(latest_date)
    iso_cal = inf_date.isocalendar()
    inference_week = iso_cal[1]
    inference_year = iso_cal[0]

    # Target date = inference_date + 1 business day
    target_date = inf_date + pd.offsets.BDay(1)
    target_date_str = str(target_date.date())

    predictions = []
    for model_id in data["available_models"]:
        try:
            model_path = models_dir / f'{model_id}_h1.pkl'
            model = joblib.load(model_path)

            if model.requires_scaling:
                pred_return = float(model.predict(X_latest_scaled)[0])
            else:
                pred_return = float(model.predict(X_latest)[0])

            predicted_price = latest_close * np.exp(pred_return)
            direction = "UP" if pred_return > 0 else "DOWN"
            signal = 1 if pred_return > 0.001 else (-1 if pred_return < -0.001 else 0)

            predictions.append({
                "model_id": model_id,
                "horizon_id": 1,
                "inference_date": latest_date,
                "inference_week": inference_week,
                "inference_year": inference_year,
                "target_date": target_date_str,
                "base_price": latest_close,
                "predicted_price": round(predicted_price, 4),
                "predicted_return_pct": round(pred_return * 100, 4),
                "direction": direction,
                "signal": signal,
            })

            logger.info(
                f"[L5b] {model_id}: return={pred_return:+.6f}, "
                f"price={predicted_price:.2f}, dir={direction}"
            )

        except Exception as e:
            logger.error(f"[L5b] Prediction failed for {model_id}: {e}")

    logger.info(f"[L5b] Generated {len(predictions)} predictions for {latest_date}")

    result = {
        "inference_date": latest_date,
        "inference_week": inference_week,
        "inference_year": inference_year,
        "latest_close": latest_close,
        "n_predictions": len(predictions),
        "predictions": predictions,
    }
    context['ti'].xcom_push(key='predictions', value=result)
    return result


# =============================================================================
# TASK 4: PERSIST FORECASTS
# =============================================================================

def persist_forecasts(**context) -> Dict[str, Any]:
    """
    UPSERT predictions into bi.fact_forecasts.
    Same schema as L5a — ON CONFLICT updates prediction fields.
    """
    ti = context['ti']
    pred_data = ti.xcom_pull(key='predictions', task_ids='generate_predictions')

    if not pred_data or not pred_data.get("predictions"):
        logger.warning("[L5b] No predictions to persist")
        return {"persisted": 0}

    conn = get_db_connection()
    try:
        cur = conn.cursor()
        persisted = 0

        for pred in pred_data["predictions"]:
            cur.execute("""
                INSERT INTO bi.fact_forecasts
                (inference_date, inference_week, inference_year, target_date,
                 model_id, horizon_id, base_price, predicted_price,
                 predicted_return_pct, direction, signal)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (inference_date, model_id, horizon_id)
                DO UPDATE SET
                    predicted_price = EXCLUDED.predicted_price,
                    predicted_return_pct = EXCLUDED.predicted_return_pct,
                    direction = EXCLUDED.direction,
                    signal = EXCLUDED.signal,
                    base_price = EXCLUDED.base_price,
                    target_date = EXCLUDED.target_date
            """, (
                pred["inference_date"],
                pred["inference_week"],
                pred["inference_year"],
                pred["target_date"],
                pred["model_id"],
                pred["horizon_id"],
                pred["base_price"],
                pred["predicted_price"],
                pred["predicted_return_pct"],
                pred["direction"],
                pred["signal"],
            ))
            persisted += 1

        conn.commit()
        logger.info(f"[L5b] Persisted {persisted} forecasts to bi.fact_forecasts")

        return {"persisted": persisted, "inference_date": pred_data["inference_date"]}

    finally:
        conn.close()


# =============================================================================
# TASK 5: INFERENCE SUMMARY
# =============================================================================

def inference_summary(**context) -> None:
    """Log structured summary of the daily inference run."""
    ti = context['ti']

    data = ti.xcom_pull(key='data', task_ids='load_models_and_features')
    pred_data = ti.xcom_pull(key='predictions', task_ids='generate_predictions')

    logger.info("=" * 60)
    logger.info("[L5b] DAILY INFERENCE SUMMARY")
    logger.info("=" * 60)

    if data:
        logger.info(f"  Models loaded:    {data.get('n_models', '?')}/{len(MODEL_IDS)}")
        logger.info(f"  Latest date:      {data.get('latest_date', '?')}")
        logger.info(f"  Latest close:     {data.get('latest_close', '?'):.2f}")

    if pred_data:
        logger.info(f"  Predictions:      {pred_data.get('n_predictions', 0)} for H=1")
        logger.info(f"  Inference date:   {pred_data.get('inference_date', '?')}")

        # Count directions
        ups = sum(1 for p in pred_data.get("predictions", []) if p["direction"] == "UP")
        downs = len(pred_data.get("predictions", [])) - ups
        consensus = "LONG" if ups > downs else "SHORT" if downs > ups else "MIXED"
        logger.info(f"  Consensus:        {consensus} ({ups} UP / {downs} DOWN)")

        for p in pred_data.get("predictions", []):
            dir_str = "LONG" if p["direction"] == "UP" else "SHORT"
            logger.info(
                f"    {p['model_id']:25s} {dir_str:5s} "
                f"ret={p['predicted_return_pct']:+.4f}% "
                f"price={p['predicted_price']:.2f}"
            )

    logger.info("=" * 60)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 15),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=5),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Daily forecasting inference: load L5a models + fresh features -> predictions',
    schedule_interval='0 18 * * 1-5',  # 18:00 UTC = 13:00 COT (30 min before L5c)
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_check_market = ShortCircuitOperator(
        task_id='check_market_day',
        python_callable=check_market_day,
        provide_context=True,
    )

    t_load = PythonOperator(
        task_id='load_models_and_features',
        python_callable=load_models_and_features,
        provide_context=True,
    )

    t_predict = PythonOperator(
        task_id='generate_predictions',
        python_callable=generate_predictions,
        provide_context=True,
    )

    t_persist = PythonOperator(
        task_id='persist_forecasts',
        python_callable=persist_forecasts,
        provide_context=True,
    )

    t_summary = PythonOperator(
        task_id='inference_summary',
        python_callable=inference_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # DAG flow: check market -> load -> predict -> persist -> summary
    t_check_market >> t_load >> t_predict >> t_persist >> t_summary
