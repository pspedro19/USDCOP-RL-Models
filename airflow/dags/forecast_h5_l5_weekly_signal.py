"""
DAG: forecast_h5_l5_weekly_signal
====================================
Track B — Monday signal generation: load Ridge + BR models, predict H=5,
compute ensemble mean -> direction + confidence scoring.

Schedule: Monday 13:15 UTC = 08:15 COT (15 min after Track A L5b)
Input: Models from H5-L5a (.pkl files)
Output: Rows in forecast_h5_predictions + forecast_h5_signals (with confidence)
Downstream: forecast_h5_l5_vol_targeting reads the signal

Contract: FC-H5-L5-001
Version: 2.0.0 (Smart Simple v1.0 — confidence scoring)
Date: 2026-02-16
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict
import json
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H5_L5_WEEKLY_SIGNAL,
    get_dag_tags,
)
from utils.dag_common import get_db_connection

DAG_ID = FORECAST_H5_L5_WEEKLY_SIGNAL
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path('/opt/airflow')
MODELS_DIR = PROJECT_ROOT / 'outputs' / 'forecasting' / 'h5_weekly_models' / 'latest'

FEATURE_COLUMNS = (
    "close", "open", "high", "low",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
    "day_of_week", "month", "is_month_end",
    "dxy_close_lag1", "oil_close_lag1",
    "vix_close_lag1", "embi_close_lag1",
)

H5_MODEL_IDS = ("ridge", "bayesian_ridge")
COLLAPSE_THRESHOLD = 0.001


# =============================================================================
# TASK 1: CHECK MARKET DAY
# =============================================================================

def check_market_day(**context) -> bool:
    """Skip if Monday is a Colombian holiday."""
    import pandas as pd
    today = pd.Timestamp(context['ds'])
    if today.dayofweek != 0:
        logger.info(f"[H5-L5b] Not Monday ({today.day_name()}), skipping")
        return False
    logger.info(f"[H5-L5b] Monday {today.date()}, proceeding")
    return True


# =============================================================================
# TASK 2: LOAD MODELS AND GENERATE SIGNAL
# =============================================================================

def generate_signal(**context) -> Dict[str, Any]:
    """
    Load trained H5 models, predict on latest features, compute ensemble mean.
    """
    import numpy as np
    import pandas as pd
    import joblib

    # Load latest features (built by L5a on Sunday)
    pred_path = PROJECT_ROOT / 'outputs' / 'forecasting' / 'h5_l5a_pred_features_temp.parquet'
    if not pred_path.exists():
        raise FileNotFoundError(f"[H5-L5b] Prediction features not found: {pred_path}")

    df_pred = pd.read_parquet(pred_path)
    feature_cols = list(FEATURE_COLUMNS)

    X_latest = df_pred[feature_cols].iloc[-1:].values.astype(np.float64)
    latest_date = str(df_pred["date"].iloc[-1].date())
    latest_close = float(df_pred["close"].iloc[-1])

    # Load scaler
    scaler = joblib.load(MODELS_DIR / 'scaler_h5.pkl')
    X_latest_scaled = scaler.transform(X_latest)

    # Monday signal date
    signal_date = pd.Timestamp(context['ds'])
    iso_cal = signal_date.isocalendar()
    inference_week = iso_cal[1]
    inference_year = iso_cal[0]

    # Target date = signal_date + 5 business days
    target_date = signal_date + pd.offsets.BDay(5)

    # Predict with each model
    predictions = []
    pred_returns = []

    for model_id in H5_MODEL_IDS:
        model_path = MODELS_DIR / f'{model_id}_h5.pkl'
        if not model_path.exists():
            logger.error(f"[H5-L5b] Model not found: {model_path}")
            continue

        model = joblib.load(model_path)
        pred_return = float(model.predict(X_latest_scaled)[0])
        predicted_price = latest_close * np.exp(pred_return)
        direction = "UP" if pred_return > 0 else "DOWN"

        predictions.append({
            "model_id": model_id,
            "horizon_id": 5,
            "inference_date": latest_date,
            "inference_week": inference_week,
            "inference_year": inference_year,
            "target_date": str(target_date.date()),
            "base_price": latest_close,
            "predicted_price": round(predicted_price, 4),
            "predicted_return_pct": round(pred_return * 100, 4),
            "direction": direction,
        })
        pred_returns.append(pred_return)

        logger.info(
            f"[H5-L5b] {model_id}: return={pred_return:+.6f}, "
            f"price={predicted_price:.2f}, dir={direction}"
        )

    if not pred_returns:
        raise ValueError("[H5-L5b] No valid predictions generated")

    # Ensemble: mean of 2 model predictions
    ensemble_return = float(np.mean(pred_returns))
    ensemble_direction = 1 if ensemble_return > 0 else -1

    logger.info(
        f"[H5-L5b] ENSEMBLE: return={ensemble_return:+.6f}, "
        f"direction={'LONG' if ensemble_direction == 1 else 'SHORT'}"
    )

    # Confidence scoring (Smart Simple v1.0)
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.forecasting.confidence_scorer import (
        ConfidenceConfig,
        score_confidence,
    )

    import yaml
    conf_path = PROJECT_ROOT / 'config' / 'execution' / 'smart_simple_v1.yaml'
    with open(conf_path) as f:
        ss_config = yaml.safe_load(f)
    cc = ss_config.get("confidence", {})

    conf_config = ConfidenceConfig(
        agreement_tight=cc.get("agreement_tight", 0.001),
        agreement_loose=cc.get("agreement_loose", 0.005),
        magnitude_high=cc.get("magnitude_high", 0.010),
        magnitude_medium=cc.get("magnitude_medium", 0.005),
        short_high=cc.get("short", {}).get("HIGH", 2.0),
        short_medium=cc.get("short", {}).get("MEDIUM", 1.5),
        short_low=cc.get("short", {}).get("LOW", 1.0),
        long_high=cc.get("long", {}).get("HIGH", 1.0),
        long_medium=cc.get("long", {}).get("MEDIUM", 0.5),
        long_low=cc.get("long", {}).get("LOW", 0.0),
    )

    confidence = score_confidence(
        ridge_pred=pred_returns[0],
        br_pred=pred_returns[1],
        direction=ensemble_direction,
        config=conf_config,
    )

    dir_str = "LONG" if ensemble_direction == 1 else "SHORT"
    logger.info(
        f"[H5-L5b] CONFIDENCE: tier={confidence.tier.value}, "
        f"agreement={confidence.agreement:.6f}, "
        f"magnitude={confidence.magnitude:.6f}, "
        f"sizing_mult={confidence.sizing_multiplier:.2f}, "
        f"skip={confidence.skip_trade}"
    )

    result = {
        "signal_date": str(signal_date.date()),
        "inference_date": latest_date,
        "inference_week": inference_week,
        "inference_year": inference_year,
        "target_date": str(target_date.date()),
        "latest_close": latest_close,
        "predictions": predictions,
        "ensemble_return": ensemble_return,
        "direction": ensemble_direction,
        "confidence_tier": confidence.tier.value,
        "confidence_agreement": confidence.agreement,
        "confidence_magnitude": confidence.magnitude,
        "sizing_multiplier": confidence.sizing_multiplier,
        "skip_trade": confidence.skip_trade,
    }
    context['ti'].xcom_push(key='signal', value=result)
    return result


# =============================================================================
# TASK 3: PERSIST PREDICTIONS + SIGNAL
# =============================================================================

def persist_signal(**context) -> Dict[str, Any]:
    """
    UPSERT per-model predictions to forecast_h5_predictions
    and ensemble signal to forecast_h5_signals.
    """
    ti = context['ti']
    signal_data = ti.xcom_pull(key='signal', task_ids='generate_signal')

    if not signal_data:
        return {"persisted": 0}

    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Persist per-model predictions
        for pred in signal_data["predictions"]:
            cur.execute("""
                INSERT INTO forecast_h5_predictions
                (inference_date, inference_week, inference_year, target_date,
                 model_id, horizon_id, base_price, predicted_price,
                 predicted_return_pct, direction)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (inference_date, model_id, horizon_id)
                DO UPDATE SET
                    predicted_price = EXCLUDED.predicted_price,
                    predicted_return_pct = EXCLUDED.predicted_return_pct,
                    direction = EXCLUDED.direction,
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
            ))

        # Persist ensemble signal + confidence (vol-targeting added by L5c)
        cur.execute("""
            INSERT INTO forecast_h5_signals
            (signal_date, inference_date, inference_week, inference_year,
             ensemble_return, direction,
             confidence_tier, confidence_agreement, confidence_magnitude,
             sizing_multiplier, skip_trade)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (signal_date)
            DO UPDATE SET
                ensemble_return = EXCLUDED.ensemble_return,
                direction = EXCLUDED.direction,
                confidence_tier = EXCLUDED.confidence_tier,
                confidence_agreement = EXCLUDED.confidence_agreement,
                confidence_magnitude = EXCLUDED.confidence_magnitude,
                sizing_multiplier = EXCLUDED.sizing_multiplier,
                skip_trade = EXCLUDED.skip_trade
        """, (
            signal_data["signal_date"],
            signal_data["inference_date"],
            signal_data["inference_week"],
            signal_data["inference_year"],
            signal_data["ensemble_return"],
            signal_data["direction"],
            signal_data.get("confidence_tier"),
            signal_data.get("confidence_agreement"),
            signal_data.get("confidence_magnitude"),
            signal_data.get("sizing_multiplier"),
            signal_data.get("skip_trade", False),
        ))

        conn.commit()
        logger.info(
            f"[H5-L5b] Persisted {len(signal_data['predictions'])} predictions "
            f"+ 1 signal for {signal_data['signal_date']}"
        )

        return {"persisted": len(signal_data["predictions"]) + 1}

    finally:
        conn.close()


# =============================================================================
# TASK 4: NOTIFY
# =============================================================================

def notify_signal(**context) -> None:
    """Log the final signal for downstream DAGs."""
    ti = context['ti']
    signal_data = ti.xcom_pull(key='signal', task_ids='generate_signal')

    if signal_data:
        dir_str = "LONG" if signal_data["direction"] == 1 else "SHORT"
        skip_str = " [SKIP]" if signal_data.get("skip_trade") else ""
        logger.info(
            f"[H5-L5b] SIGNAL: {dir_str} for week "
            f"{signal_data['inference_year']}-W{signal_data['inference_week']:02d}, "
            f"ensemble_return={signal_data['ensemble_return']:+.6f}, "
            f"confidence={signal_data.get('confidence_tier', '?')}, "
            f"sizing={signal_data.get('sizing_multiplier', '?')}{skip_str}"
        )


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    'owner': 'forecast-h5-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 16),
    'retries': 2,
    'retry_delay': timedelta(minutes=3),
    'execution_timeout': timedelta(minutes=10),
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='Track B: Monday H=5 signal — Ridge+BR ensemble mean (08:15 COT)',
    schedule_interval='15 13 * * 1',  # Lun 13:15 UTC = 08:15 COT
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_check = ShortCircuitOperator(
        task_id='check_market_day',
        python_callable=check_market_day,
    )

    t_signal = PythonOperator(
        task_id='generate_signal',
        python_callable=generate_signal,
    )

    t_persist = PythonOperator(
        task_id='persist_signal',
        python_callable=persist_signal,
    )

    t_notify = PythonOperator(
        task_id='notify_signal',
        python_callable=notify_signal,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    t_check >> t_signal >> t_persist >> t_notify
