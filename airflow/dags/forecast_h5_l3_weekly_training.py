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


# =============================================================================
# MLFLOW TRACKING HELPERS (non-blocking)
# =============================================================================

MLFLOW_EXPERIMENT_NAME = "h5_weekly_training"


def _get_git_commit_hash() -> Optional[str]:
    """Return current git HEAD commit hash, or None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as e:  # pragma: no cover - best-effort
        logger.debug(f"[H5-L3] Could not resolve git commit: {e}")
    return None


def _mlflow_safe_start_run(run_name: str, experiment_name: str = MLFLOW_EXPERIMENT_NAME):
    """
    Start an MLflow run if the server is reachable. Returns (mlflow_module, active_run)
    on success or (None, None) if MLflow is unavailable. Non-blocking: never raises.
    """
    try:
        import mlflow  # type: ignore
        import os
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        active_run = mlflow.start_run(run_name=run_name)
        logger.info(
            f"[H5-L3] MLflow run started: experiment={experiment_name}, "
            f"run_name={run_name}, run_id={active_run.info.run_id}"
        )
        return mlflow, active_run
    except Exception as e:
        logger.warning(
            f"[H5-L3] MLflow tracking unavailable (non-blocking): {type(e).__name__}: {e}"
        )
        return None, None


def _mlflow_safe_end_run(mlflow_mod, status: str = "FINISHED") -> None:
    """End active MLflow run, swallowing errors."""
    if mlflow_mod is None:
        return
    try:
        mlflow_mod.end_run(status=status)
    except Exception as e:  # pragma: no cover
        logger.warning(f"[H5-L3] MLflow end_run failed (non-blocking): {e}")


def _mlflow_safe_call(mlflow_mod, method: str, *args, **kwargs) -> None:
    """Call an MLflow logging method, swallowing errors."""
    if mlflow_mod is None:
        return
    try:
        getattr(mlflow_mod, method)(*args, **kwargs)
    except Exception as e:  # pragma: no cover
        logger.warning(f"[H5-L3] MLflow {method} failed (non-blocking): {e}")


# =============================================================================
# TASK 0: DATA FRESHNESS GATE
# =============================================================================

def validate_data_freshness(**context):
    """Gate: verify L0 OHLCV + macro data are fresh before training."""
    from utils.data_quality import validate_training_data_freshness

    result = validate_training_data_freshness(ohlcv_max_age=3, macro_max_age=7)
    logger.info(f"[H5-L3] Data freshness OK: {result}")
    context['ti'].xcom_push(key='data_freshness', value=result)

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

    MLflow tracking is wrapped around the entire training block. If the MLflow
    server is unreachable, training proceeds normally and a warning is logged.
    """
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

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

    # ------------------------------------------------------------------
    # MLflow: start run (non-blocking)
    # ------------------------------------------------------------------
    train_start_date = df_train["date"].iloc[0].date() if len(df_train) else None
    train_end_date = df_train["date"].iloc[-1].date() if len(df_train) else None
    iso_cal = pd.Timestamp(train_end_date).isocalendar() if train_end_date else None
    iso_week = int(iso_cal[1]) if iso_cal is not None else 0
    iso_year = int(iso_cal[0]) if iso_cal is not None else 0
    run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_name = f"h5_weekly_W{iso_week:02d}_{iso_year}_{run_timestamp}"

    mlflow_mod, mlflow_run = _mlflow_safe_start_run(run_name)

    # Log high-level params + tags (safe no-op if mlflow_mod is None)
    _mlflow_safe_call(mlflow_mod, "log_params", {
        "horizon": 5,
        "model_ids": ",".join(H5_MODEL_IDS),
        "n_models": len(H5_MODEL_IDS),
        "feature_count": len(feature_cols),
        "train_samples": int(len(X)),
        "train_window_start": str(train_start_date) if train_start_date else "",
        "train_window_end": str(train_end_date) if train_end_date else "",
        "iso_week": iso_week,
        "iso_year": iso_year,
        "expanding_window_from": "2020-01-01",
        "target": "ln(close[t+5]/close[t])",
        "seed": 42,
    })
    _mlflow_safe_call(mlflow_mod, "set_tag", "pipeline", "h5_weekly")
    _mlflow_safe_call(mlflow_mod, "set_tag", "dag_id", DAG_ID)
    _mlflow_safe_call(mlflow_mod, "set_tag", "contract", "FC-H5-L3-001")
    git_sha = _get_git_commit_hash()
    if git_sha:
        _mlflow_safe_call(mlflow_mod, "set_tag", "git_commit", git_sha)
        _mlflow_safe_call(mlflow_mod, "set_tag", "mlflow.source.git.commit", git_sha)

    try:
        # Scaler for linear models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler_path = MODELS_DIR / 'scaler_h5.pkl'
        joblib.dump(scaler, scaler_path)

        trained = {}
        for model_id in H5_MODEL_IDS:
            try:
                model = ModelFactory.create(model_id, params=None, horizon=5)
                model.fit(X_scaled, y)

                model_path = MODELS_DIR / f'{model_id}_h5.pkl'
                joblib.dump(model, model_path)

                # In-sample DA + R^2
                preds = model.predict(X_scaled)
                correct = np.sum(np.sign(preds) == np.sign(y))
                in_sample_da = correct / len(y) * 100
                try:
                    in_sample_r2 = float(r2_score(y, preds))
                except Exception:
                    in_sample_r2 = float("nan")

                trained[model_id] = {
                    "path": str(model_path),
                    "in_sample_da": round(in_sample_da, 1),
                    "pred_std": round(float(np.std(preds)), 6),
                    "in_sample_r2": round(in_sample_r2, 6)
                    if not np.isnan(in_sample_r2) else None,
                }
                logger.info(
                    f"[H5-L5a] Trained {model_id}: DA={in_sample_da:.1f}%, "
                    f"R2={in_sample_r2:.4f}, pred_std={np.std(preds):.6f}"
                )

                # MLflow: per-model metrics
                _mlflow_safe_call(mlflow_mod, "log_metric", f"{model_id}_in_sample_da_pct", float(in_sample_da))
                _mlflow_safe_call(mlflow_mod, "log_metric", f"{model_id}_pred_std", float(np.std(preds)))
                if not np.isnan(in_sample_r2):
                    _mlflow_safe_call(mlflow_mod, "log_metric", f"{model_id}_in_sample_r2", float(in_sample_r2))

                # MLflow: per-model artifact
                _mlflow_safe_call(mlflow_mod, "log_artifact", str(model_path), artifact_path="models")

            except Exception as e:
                logger.error(f"[H5-L5a] Failed to train {model_id}: {e}")
                trained[model_id] = {"path": None, "error": str(e)}
                _mlflow_safe_call(mlflow_mod, "set_tag", f"{model_id}_error", str(e)[:250])

        # Save feature columns
        feature_cols_path = MODELS_DIR / 'feature_cols_h5.json'
        with open(feature_cols_path, 'w') as f:
            json.dump(feature_cols, f)

        n_success = sum(1 for v in trained.values() if v.get("path"))
        logger.info(f"[H5-L5a] Trained {n_success}/{len(H5_MODEL_IDS)} models")

        # MLflow: aggregate metrics + supporting artifacts
        _mlflow_safe_call(mlflow_mod, "log_metric", "models_trained", float(n_success))
        _mlflow_safe_call(mlflow_mod, "log_metric", "models_total", float(len(H5_MODEL_IDS)))
        _mlflow_safe_call(mlflow_mod, "log_artifact", str(scaler_path), artifact_path="models")
        _mlflow_safe_call(mlflow_mod, "log_artifact", str(feature_cols_path), artifact_path="models")

        result = {
            "models_dir": str(MODELS_DIR),
            "models_trained": n_success,
            "model_details": trained,
            "train_samples": len(X),
            "mlflow_run_id": mlflow_run.info.run_id if mlflow_run is not None else None,
        }
        context['ti'].xcom_push(key='training', value=result)
        _mlflow_safe_end_run(mlflow_mod, status="FINISHED")
        return result

    except Exception:
        _mlflow_safe_end_run(mlflow_mod, status="FAILED")
        raise


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

    t_data_gate = PythonOperator(
        task_id='validate_data_freshness',
        python_callable=validate_data_freshness,
    )

    t_data_gate >> t_load >> t_train >> t_validate >> t_persist >> t_summary
