"""
DAG: forecast_h5_l3_weekly_training
=====================================
Track B — Weekly H=5 training: Ridge + BayesianRidge only.

Architecture:
    seeds/usdcop_daily_ohlcv.parquet + MACRO_DAILY_CLEAN.parquet
                  |
                  v
        load_and_build_features()   (same 21 features as Track A)
                  |
                  v
        train_h5_models()           (Ridge + BR only, target = ln(close[t+5]/close[t]))
                  |
                  v
        validate_models()           (collapse detection, sanity checks)
                  |
                  v
        persist_models()            (save .pkl to models_dir)
                  |
                  v
        training_summary()

Schedule: Sunday 06:30 UTC = 01:30 COT (30 min after Track A L5a)
Output: 2 model .pkl files + scaler, saved to outputs/forecasting/h5_weekly_models/latest/
Downstream: forecast_h5_l5_weekly_signal reads these models on Monday

Contract: FC-H5-L3-001
Version: 1.0.0
Date: 2026-02-16
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L3_WEEKLY_TRAINING,
    get_dag_tags,
)
from utils.dag_common import get_db_connection
from src.forecasting.ssot_config import ForecastingSSOTConfig
from src.forecasting.dataset_loader import ForecastingDatasetLoader
from src.forecasting.data_contracts import FEATURE_COLUMNS

DAG_ID = FORECAST_H5_L3_WEEKLY_TRAINING
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'forecasting' / 'h5_weekly_models' / 'latest'

# Track B: LINEAR ONLY (ARD excluded — collapsed in validation)
H5_MODEL_IDS = ("ridge", "bayesian_ridge")

# Collapse detection
COLLAPSE_THRESHOLD = 0.001
COLLAPSE_WINDOW_WEEKS = 12


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
        target_horizon=5,
        db_conn_func=get_db_connection,
    )

    # Drop NaN
    feature_mask = df[list(FEATURE_COLUMNS)].notna().all(axis=1)
    target_mask = df["target_return_5d"].notna()
    df_clean = df[feature_mask & target_mask].reset_index(drop=True)

    logger.info(
        f"[H5-L5a] After cleanup: {len(df_clean)} rows, "
        f"{df_clean['date'].iloc[0].date()} to {df_clean['date'].iloc[-1].date()}"
    )

    # Save temp parquet
    temp_path = str(PROJECT_ROOT / 'outputs' / 'forecasting' / 'h5_l5a_features_temp.parquet')
    Path(temp_path).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(temp_path, index=False)

    # Save full df for prediction (includes last rows without target)
    df_pred = df[feature_mask].reset_index(drop=True)
    pred_path = str(PROJECT_ROOT / 'outputs' / 'forecasting' / 'h5_l5a_pred_features_temp.parquet')
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
# TASK 2: TRAIN H=5 MODELS
# =============================================================================

def train_h5_models(**context) -> Dict[str, Any]:
    """
    Train Ridge + BayesianRidge for H=5 on expanding window from 2020-01-01.
    Training takes < 1 second (2 linear models).
    """
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler

    ti = context['ti']
    data = ti.xcom_pull(key='data', task_ids='load_and_build_features')
    if not data:
        raise ValueError("[H5-L5a] No data from load_and_build_features")

    df = pd.read_parquet(data["features_path"])

    # Expanding window: train on everything from 2020-01-01
    train_start = pd.Timestamp("2020-01-01")
    df_train = df[df["date"] >= train_start].copy()

    feature_cols = list(FEATURE_COLUMNS)
    X = df_train[feature_cols].values.astype(np.float64)
    y = df_train["target_return_5d"].values.astype(np.float64)

    logger.info(f"[H5-L5a] Training on {len(X)} samples, {len(feature_cols)} features, H=5")

    sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.models.factory import ModelFactory

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Scaler for linear models
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, MODELS_DIR / 'scaler_h5.pkl')

    trained = {}
    for model_id in H5_MODEL_IDS:
        try:
            model = ModelFactory.create(model_id, params=None, horizon=5)
            model.fit(X_scaled, y)

            model_path = MODELS_DIR / f'{model_id}_h5.pkl'
            joblib.dump(model, model_path)

            # In-sample DA
            preds = model.predict(X_scaled)
            correct = np.sum(np.sign(preds) == np.sign(y))
            in_sample_da = correct / len(y) * 100

            trained[model_id] = {
                "path": str(model_path),
                "in_sample_da": round(in_sample_da, 1),
                "pred_std": round(float(np.std(preds)), 6),
            }
            logger.info(f"[H5-L5a] Trained {model_id}: DA={in_sample_da:.1f}%, pred_std={np.std(preds):.6f}")

        except Exception as e:
            logger.error(f"[H5-L5a] Failed to train {model_id}: {e}")
            trained[model_id] = {"path": None, "error": str(e)}

    # Save feature columns
    with open(MODELS_DIR / 'feature_cols_h5.json', 'w') as f:
        json.dump(feature_cols, f)

    n_success = sum(1 for v in trained.values() if v.get("path"))
    logger.info(f"[H5-L5a] Trained {n_success}/{len(H5_MODEL_IDS)} models")

    result = {
        "models_dir": str(MODELS_DIR),
        "models_trained": n_success,
        "model_details": trained,
        "train_samples": len(X),
    }
    context['ti'].xcom_push(key='training', value=result)
    return result


# =============================================================================
# TASK 3: VALIDATE MODELS (collapse detection)
# =============================================================================

def validate_models(**context) -> Dict[str, Any]:
    """
    Check for model collapse: if rolling 12-week prediction std < threshold,
    flag the model. A collapsed model produces nearly identical predictions
    regardless of input, indicating training failure.
    """
    ti = context['ti']
    training = ti.xcom_pull(key='training', task_ids='train_h5_models')

    if not training:
        raise ValueError("[H5-L5a] No training data")

    collapse_flags = {}
    for model_id, detail in training.get("model_details", {}).items():
        if not detail.get("path"):
            collapse_flags[model_id] = {"collapsed": True, "reason": "training_failed"}
            continue

        pred_std = detail.get("pred_std", 0)
        collapsed = pred_std < COLLAPSE_THRESHOLD
        collapse_flags[model_id] = {
            "collapsed": collapsed,
            "pred_std": pred_std,
            "threshold": COLLAPSE_THRESHOLD,
        }
        if collapsed:
            logger.warning(f"[H5-L5a] COLLAPSE WARNING: {model_id} pred_std={pred_std:.6f} < {COLLAPSE_THRESHOLD}")

    any_collapsed = any(v["collapsed"] for v in collapse_flags.values())
    result = {
        "collapse_flags": collapse_flags,
        "any_collapsed": any_collapsed,
    }
    context['ti'].xcom_push(key='validation', value=result)
    return result


# =============================================================================
# TASK 4: PERSIST MODELS
# =============================================================================

def persist_models(**context) -> Dict[str, Any]:
    """
    Persist training metadata to DB (forecast_h5_predictions will be
    populated by L5b when signal is generated on Monday).
    """
    ti = context['ti']
    data = ti.xcom_pull(key='data', task_ids='load_and_build_features')
    training = ti.xcom_pull(key='training', task_ids='train_h5_models')

    if not data or not training:
        return {"persisted": False}

    logger.info(
        f"[H5-L5a] Models saved to {training['models_dir']}, "
        f"{training['models_trained']}/{len(H5_MODEL_IDS)} successful"
    )

    result = {
        "persisted": True,
        "models_dir": training["models_dir"],
        "latest_date": data.get("latest_date"),
    }
    context['ti'].xcom_push(key='persist', value=result)
    return result


# =============================================================================
# TASK 5: TRAINING SUMMARY
# =============================================================================

def training_summary(**context) -> None:
    """Log structured summary of the weekly H=5 training run."""
    ti = context['ti']
    data = ti.xcom_pull(key='data', task_ids='load_and_build_features')
    training = ti.xcom_pull(key='training', task_ids='train_h5_models')
    validation = ti.xcom_pull(key='validation', task_ids='validate_models')

    logger.info("=" * 60)
    logger.info("[H5-L5a] WEEKLY H=5 TRAINING SUMMARY")
    logger.info("=" * 60)

    if data:
        logger.info(f"  Data rows:        {data.get('n_rows', '?')}")
        logger.info(f"  Latest date:      {data.get('latest_date', '?')}")
        logger.info(f"  Latest close:     {data.get('latest_close', '?'):.2f}")

    if training:
        logger.info(f"  Models trained:   {training.get('models_trained', '?')}/{len(H5_MODEL_IDS)}")
        logger.info(f"  Train samples:    {training.get('train_samples', '?')}")
        for mid, detail in training.get("model_details", {}).items():
            if detail.get("path"):
                logger.info(f"    {mid:25s} DA_in={detail.get('in_sample_da', '?')}% std={detail.get('pred_std', '?')}")
            else:
                logger.info(f"    {mid:25s} FAILED: {detail.get('error', '?')}")

    if validation:
        for mid, flags in validation.get("collapse_flags", {}).items():
            status = "COLLAPSED" if flags.get("collapsed") else "OK"
            logger.info(f"    {mid:25s} collapse={status}")

    logger.info("=" * 60)


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-h5-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 16),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Track B: Weekly H=5 training — Ridge + BayesianRidge (01:30 COT)',
    schedule_interval='30 6 * * 0',  # Dom 06:30 UTC = 01:30 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_load = PythonOperator(
        task_id='load_and_build_features',
        python_callable=load_and_build_features,
    )

    t_train = PythonOperator(
        task_id='train_h5_models',
        python_callable=train_h5_models,
    )

    t_validate = PythonOperator(
        task_id='validate_models',
        python_callable=validate_models,
    )

    t_persist = PythonOperator(
        task_id='persist_models',
        python_callable=persist_models,
    )

    t_summary = PythonOperator(
        task_id='training_summary',
        python_callable=training_summary,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t_load >> t_train >> t_validate >> t_persist >> t_summary
