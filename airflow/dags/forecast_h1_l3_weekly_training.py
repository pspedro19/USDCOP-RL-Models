"""
DAG: forecast_h1_l3_weekly_training
=====================================
Weekly forecasting training: train 9 models for H=1, generate predictions,
persist to bi.fact_forecasts.

Architecture:
    seeds/usdcop_daily_ohlcv.parquet + MACRO_DAILY_CLEAN.parquet
                  |
                  v
        load_and_build_features()
                  |
                  v
        train_models()  --> save .pkl to models_dir
                  |
                  v
        generate_predictions()
                  |
                  v
        persist_forecasts()  --> UPSERT into bi.fact_forecasts
                  |
                  v
        training_summary()

Data Sources (parquet-based, no DB required for training):
    - seeds/latest/usdcop_daily_ohlcv.parquet (daily OHLCV)
    - data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet (DXY + WTI)

Why train weekly?
    Models train on ALL historical data in ~3.5 seconds (9 models, H=1).
    Walk-forward validation (5 folds) confirmed DA=55.9%, Sharpe=2.109, p=0.002.
    Weekly retraining captures the latest data without overhead.

Schedule: Every Sunday at 06:00 UTC (01:00 COT)
Output: 9 rows in bi.fact_forecasts per inference_date (one per model, H=1)
Downstream: forecast_h1_l5_vol_targeting reads these predictions daily

Author: Trading Team
Version: 1.0.0
Date: 2026-02-15
Contract: FC-H1-L3-001
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import math
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
sys.path.insert(0, '/opt/airflow')

# =============================================================================
# IMPORTS FROM SSOT
# =============================================================================

from contracts.dag_registry import (
    FORECAST_H1_L3_WEEKLY_TRAINING,
    get_dag_tags,
)
from utils.dag_common import get_db_connection
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader
from src.forecasting.data_contracts import FEATURE_COLUMNS
from src.forecasting.contracts import HORIZON_CONFIGS

DAG_ID = FORECAST_H1_L3_WEEKLY_TRAINING
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

# Project root in Docker
PROJECT_ROOT = Path('/opt/airflow')

# Models output directory (overwritten each week)
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'forecasting' / 'weekly_models' / 'latest'

# 9 models to train (validated in FC-SIZE-001 walk-forward)
MODEL_IDS = (
    "ridge", "bayesian_ridge", "ard",
    "xgboost_pure", "lightgbm_pure", "catboost_pure",
    "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost",
)

# Horizon-specific hyperparameters for H=1 (short horizon)
H1_CONFIG = HORIZON_CONFIGS.get("short", {
    "n_estimators": 50,
    "max_depth": 3,
    "learning_rate": 0.05,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "min_samples_leaf": 20,
})

# Model type classification (for param routing)
LINEAR_MODELS = {"ridge", "bayesian_ridge", "ard"}
CATBOOST_MODELS = {"catboost_pure"}
HYBRID_MODELS = {"hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"}
BOOSTING_MODELS = {"xgboost_pure", "lightgbm_pure"}


# =============================================================================
# TASK 1: LOAD AND BUILD FEATURES
# =============================================================================

def load_and_build_features(**context) -> Dict[str, Any]:
    """
    Load daily OHLCV + macro, build 21 SSOT features.
    Uses shared ForecastingDatasetLoader (parquet base + DB extension).
    """
    import numpy as np
    import pandas as pd

    cfg = ForecastingSSOTConfig.load()
    loader = ForecastingDatasetLoader(cfg, project_root=PROJECT_ROOT)
    df, feature_cols = loader.load_dataset_with_db_extension(
        target_horizon=1,
        db_conn_func=get_db_connection,
    )

    # Drop NaN (warmup period + missing target)
    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    target_mask = df["target_return_1d"].notna()
    df_clean = df[feature_mask & target_mask].reset_index(drop=True)

    logger.info(
        f"[L5a] After cleanup: {len(df_clean)} rows, "
        f"{df_clean['date'].iloc[0].date()} to {df_clean['date'].iloc[-1].date()}"
    )

    # Save temp parquet for downstream tasks
    temp_path = str(PROJECT_ROOT / 'outputs' / 'forecasting' / 'l5a_features_temp.parquet')
    Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(temp_path, index=False)

    # Also save the FULL df (including last row without target) for prediction
    df_pred = df[feature_mask].reset_index(drop=True)
    pred_path = str(PROJECT_ROOT / 'outputs' / 'forecasting' / 'l5a_pred_features_temp.parquet')
    df_pred.to_parquet(pred_path, index=False)

    result = {
        "features_path": temp_path,
        "pred_features_path": pred_path,
        "n_rows": len(df_clean),
        "n_pred_rows": len(df_pred),
        "latest_date": str(df_pred["date"].iloc[-1].date()),
        "latest_close": float(df_pred["close"].iloc[-1]),
    }

    context['ti'].xcom_push(key='data', value=result)
    return result


# =============================================================================
# TASK 2: TRAIN MODELS
# =============================================================================

def train_models(**context) -> Dict[str, Any]:
    """
    Train 9 models for H=1 on all available data. Save .pkl files.
    Same models + params as validated walk-forward (FC-SIZE-001).
    Training takes ~3.5 seconds total.
    """
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler

    ti = context['ti']
    data = ti.xcom_pull(key='data', task_ids='load_and_build_features')
    if not data:
        raise ValueError("[L5a] No data from load_and_build_features")

    df = pd.read_parquet(data["features_path"])
    feature_cols = list(FEATURE_COLUMNS)
    X = df[feature_cols].values.astype(np.float64)
    y = df["target_return_1d"].values.astype(np.float64)

    logger.info(f"[L5a] Training on {len(X)} samples, {len(feature_cols)} features")

    # Lazy import to avoid DAG parse issues
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.models.factory import ModelFactory
    from src.forecasting.contracts import get_horizon_config

    horizon_config = get_horizon_config(1)  # H=1 = "short"

    # Prepare output directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Fit a scaler on ALL training data (for linear/hybrid models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler alongside models
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')

    trained = {}
    for model_id in MODEL_IDS:
        try:
            params = _get_model_params(model_id, horizon_config)
            model = ModelFactory.create(model_id, params=params, horizon=1)

            # Scale for linear/hybrid models
            if model.requires_scaling:
                model.fit(X_scaled, y)
            else:
                model.fit(X, y)

            # Save model
            model_path = MODELS_DIR / f'{model_id}_h1.pkl'
            joblib.dump(model, model_path)

            # Quick in-sample DA (not for validation — just sanity check)
            if model.requires_scaling:
                preds = model.predict(X_scaled)
            else:
                preds = model.predict(X)
            correct = np.sum(np.sign(preds) == np.sign(y))
            in_sample_da = correct / len(y) * 100

            trained[model_id] = {
                "path": str(model_path),
                "in_sample_da": round(in_sample_da, 1),
            }
            logger.info(f"[L5a] Trained {model_id}: in-sample DA={in_sample_da:.1f}%")

        except Exception as e:
            logger.error(f"[L5a] Failed to train {model_id}: {e}")
            trained[model_id] = {"path": None, "error": str(e)}

    # Save feature columns for reference
    with open(MODELS_DIR / 'feature_cols.json', 'w') as f:
        json.dump(feature_cols, f)

    n_success = sum(1 for v in trained.values() if v.get("path"))
    logger.info(f"[L5a] Trained {n_success}/{len(MODEL_IDS)} models")

    result = {
        "models_dir": str(MODELS_DIR),
        "models_trained": n_success,
        "model_details": trained,
    }
    context['ti'].xcom_push(key='training', value=result)
    return result


def _get_model_params(model_id: str, horizon_config: dict) -> Optional[dict]:
    """Get model-specific params for training."""
    if model_id in LINEAR_MODELS:
        return None  # Use model defaults

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

    # XGBoost, LightGBM
    return horizon_config


# =============================================================================
# TASK 3: GENERATE PREDICTIONS
# =============================================================================

def generate_predictions(**context) -> Dict[str, Any]:
    """
    Generate H=1 predictions for the latest date using trained models.
    """
    import numpy as np
    import pandas as pd
    import joblib

    ti = context['ti']
    data = ti.xcom_pull(key='data', task_ids='load_and_build_features')
    training = ti.xcom_pull(key='training', task_ids='train_models')

    if not data or not training:
        raise ValueError("[L5a] Missing upstream data")

    # Load prediction features (includes latest row without target)
    df_pred = pd.read_parquet(data["pred_features_path"])
    feature_cols = list(FEATURE_COLUMNS)

    # Latest features for prediction
    X_latest = df_pred[feature_cols].iloc[-1:].values.astype(np.float64)
    latest_date = str(df_pred["date"].iloc[-1].date())
    latest_close = float(df_pred["close"].iloc[-1])

    # Load scaler
    models_dir = Path(training["models_dir"])
    scaler = joblib.load(models_dir / 'scaler.pkl')
    X_latest_scaled = scaler.transform(X_latest)

    # Compute ISO week/year for the inference date
    inf_date = pd.Timestamp(latest_date)
    iso_cal = inf_date.isocalendar()
    inference_week = iso_cal[1]
    inference_year = iso_cal[0]

    # Target date = inference_date + 1 business day
    target_date = inf_date + pd.offsets.BDay(1)
    target_date_str = str(target_date.date())

    predictions = []
    for model_id in MODEL_IDS:
        detail = training["model_details"].get(model_id, {})
        model_path = detail.get("path")
        if not model_path:
            continue

        try:
            model = joblib.load(model_path)

            # Use scaled or raw features depending on model type
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
                f"[L5a] {model_id}: return={pred_return:+.6f}, "
                f"price={predicted_price:.2f}, dir={direction}"
            )

        except Exception as e:
            logger.error(f"[L5a] Prediction failed for {model_id}: {e}")

    logger.info(f"[L5a] Generated {len(predictions)} predictions for {latest_date}")

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
    ON CONFLICT updates prediction fields, preserving created_at.
    """
    ti = context['ti']
    pred_data = ti.xcom_pull(key='predictions', task_ids='generate_predictions')

    if not pred_data or not pred_data.get("predictions"):
        logger.warning("[L5a] No predictions to persist")
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
        logger.info(f"[L5a] Persisted {persisted} forecasts to bi.fact_forecasts")

        return {"persisted": persisted, "inference_date": pred_data["inference_date"]}

    finally:
        conn.close()


# =============================================================================
# TASK 5: TRAINING SUMMARY
# =============================================================================

def training_summary(**context) -> None:
    """Log structured summary of the weekly inference run."""
    ti = context['ti']

    data = ti.xcom_pull(key='data', task_ids='load_and_build_features')
    training = ti.xcom_pull(key='training', task_ids='train_models')
    pred_data = ti.xcom_pull(key='predictions', task_ids='generate_predictions')

    logger.info("=" * 60)
    logger.info("[L5a] WEEKLY INFERENCE SUMMARY")
    logger.info("=" * 60)

    if data:
        logger.info(f"  Data rows:        {data.get('n_rows', '?')}")
        logger.info(f"  Latest date:      {data.get('latest_date', '?')}")
        logger.info(f"  Latest close:     {data.get('latest_close', '?'):.2f}")

    if training:
        logger.info(f"  Models trained:   {training.get('models_trained', '?')}/{len(MODEL_IDS)}")
        for mid, detail in training.get("model_details", {}).items():
            if detail.get("path"):
                logger.info(f"    {mid:25s} DA_in={detail.get('in_sample_da', '?')}%")
            else:
                logger.info(f"    {mid:25s} FAILED: {detail.get('error', '?')}")

    if pred_data:
        logger.info(f"  Predictions:      {pred_data.get('n_predictions', 0)} for H=1")
        logger.info(f"  Inference date:   {pred_data.get('inference_date', '?')}")
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
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=15),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Weekly forecasting: train 9 models H=1 + persist to bi.fact_forecasts',
    schedule_interval='0 6 * * 0',  # Sunday 06:00 UTC = 01:00 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_load = PythonOperator(
        task_id='load_and_build_features',
        python_callable=load_and_build_features,
        provide_context=True,
    )

    t_train = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
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
        task_id='training_summary',
        python_callable=training_summary,
        provide_context=True,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # DAG flow: load -> train -> predict -> persist -> summary
    t_load >> t_train >> t_predict >> t_persist >> t_summary
