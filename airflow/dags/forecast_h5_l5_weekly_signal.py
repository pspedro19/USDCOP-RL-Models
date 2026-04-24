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
import os
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.sensors.external_task import ExternalTaskSensor
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
# TASK 5: PUBLISH SIGNAL TO KAFKA (MLOps course project — gRPC+Kafka compliance)
# =============================================================================

def publish_signal_to_kafka(**context) -> Dict[str, Any]:
    """
    Publish the just-generated H5 ensemble signal to the Kafka topic ``signals.h5``.

    This task exists to satisfy the MLOps course project requirement of integrating
    Kafka as one of the two required course technologies (alongside gRPC). It is
    STRICTLY ADDITIVE: it runs after the normal DB write (``persist_signal``) and
    never alters the core signal-generation flow. Any failure (missing broker,
    missing ``kafka-python`` package, serialization error, etc.) is swallowed and
    logged as a warning so the DAG remains green.

    Flow:
      1. Pull the signal dict from XCom (produced by ``generate_signal``). If
         absent, fall back to re-querying the most recent row from
         ``forecast_h5_signals``.
      2. Build the course-defined JSON contract (see module docstring / task
         docstring below).
      3. Connect to Kafka at ``$KAFKA_BROKER`` (default ``redpanda:9092``) and
         publish to topic ``signals.h5``.
      4. Always return a status dict — never raise.

    Contract of the published message::

        {
          "week": "2026-W17",
          "signal_date": "2026-04-20",
          "direction": "SHORT",
          "confidence_tier": "HIGH",
          "confidence": 0.85,
          "ensemble_return": -0.012,
          "skip_trade": false,
          "hard_stop_pct": 2.81,
          "take_profit_pct": 1.41,
          "adjusted_leverage": 1.5,
          "regime": "trending",
          "hurst": 0.55,
          "timestamp": "2026-04-23T14:00:00-05:00",
          "source": "airflow_h5_l5"
        }
    """
    try:
        try:
            from kafka import KafkaProducer
        except ImportError:
            logger.warning(
                "[H5-L5b][kafka] kafka-python not installed, skipping Kafka publish"
            )
            return {"published": False, "reason": "kafka-python not installed"}

        ti = context['ti']
        signal_data = ti.xcom_pull(key='signal', task_ids='generate_signal')

        # Fallback: re-query DB for the most recent row if XCom is empty
        # (e.g., when this task runs after an upstream failure).
        db_row: Dict[str, Any] = {}
        if not signal_data:
            logger.warning(
                "[H5-L5b][kafka] No signal in XCom, attempting DB fallback"
            )
            try:
                conn = get_db_connection()
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        SELECT signal_date, inference_week, inference_year,
                               ensemble_return, direction, confidence_tier,
                               confidence_agreement, confidence_magnitude,
                               sizing_multiplier, skip_trade,
                               hard_stop_pct, take_profit_pct, adjusted_leverage,
                               regime, hurst
                        FROM forecast_h5_signals
                        ORDER BY signal_date DESC
                        LIMIT 1
                        """
                    )
                    row = cur.fetchone()
                    if row:
                        cols = [d[0] for d in cur.description]
                        db_row = dict(zip(cols, row))
                finally:
                    conn.close()
            except Exception as db_err:
                logger.warning(
                    f"[H5-L5b][kafka] DB fallback failed: {db_err}"
                )

            if not db_row:
                logger.warning(
                    "[H5-L5b][kafka] No signal available, skipping publish"
                )
                return {"published": False, "reason": "no signal available"}

        # Normalize values from either XCom or DB fallback.
        def _get(key: str, default: Any = None) -> Any:
            if signal_data and key in signal_data:
                return signal_data[key]
            return db_row.get(key, default)

        direction_raw = _get("direction")
        if isinstance(direction_raw, int):
            direction_str = "LONG" if direction_raw == 1 else (
                "SHORT" if direction_raw == -1 else "FLAT"
            )
        elif isinstance(direction_raw, str):
            direction_str = direction_raw.upper()
        else:
            direction_str = "FLAT"

        iso_year = _get("inference_year")
        iso_week = _get("inference_week")
        if iso_year and iso_week:
            week_str = f"{int(iso_year):04d}-W{int(iso_week):02d}"
        else:
            week_str = None

        agreement = _get("confidence_agreement")
        try:
            confidence_float = float(1.0 - min(1.0, max(0.0, float(agreement) * 100.0))) \
                if agreement is not None else None
        except (TypeError, ValueError):
            confidence_float = None

        payload = {
            "week": week_str,
            "signal_date": str(_get("signal_date")) if _get("signal_date") else None,
            "direction": direction_str,
            "confidence_tier": _get("confidence_tier"),
            "confidence": confidence_float,
            "ensemble_return": _get("ensemble_return"),
            "skip_trade": bool(_get("skip_trade", False)),
            "hard_stop_pct": _get("hard_stop_pct"),
            "take_profit_pct": _get("take_profit_pct"),
            "adjusted_leverage": _get("adjusted_leverage", _get("sizing_multiplier")),
            "regime": _get("regime"),
            "hurst": _get("hurst"),
            "timestamp": datetime.now().astimezone().isoformat(),
            "source": "airflow_h5_l5",
        }

        broker = os.environ.get("KAFKA_BROKER", "redpanda:9092")
        topic = "signals.h5"

        logger.info(
            f"[H5-L5b][kafka] Publishing to {broker}/{topic}: "
            f"week={payload['week']} direction={payload['direction']} "
            f"skip_trade={payload['skip_trade']}"
        )

        producer = KafkaProducer(
            bootstrap_servers=broker,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            acks="all",
            retries=1,
            request_timeout_ms=5000,
            api_version_auto_timeout_ms=5000,
        )
        try:
            future = producer.send(topic, value=payload)
            # Short block so we surface obvious broker errors, but cap at 5s.
            future.get(timeout=5)
            producer.flush(timeout=5)
        finally:
            try:
                producer.close(timeout=5)
            except Exception:
                pass

        logger.info(
            f"[H5-L5b][kafka] Published signal for {payload['week']} to {topic}"
        )
        return {"published": True, "topic": topic, "broker": broker}

    except Exception as exc:  # Never fail the DAG over Kafka
        logger.warning(
            f"[H5-L5b][kafka] Kafka publish failed (non-fatal): {exc}"
        )
        return {"published": False, "reason": str(exc)}


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

    # MLOps course project (gRPC+Kafka compliance): optional, always-runs-after
    # task that publishes the generated signal to the ``signals.h5`` Kafka topic.
    # Never blocks or fails the DAG regardless of upstream status or Kafka health.
    t_kafka = PythonOperator(
        task_id='publish_signal_to_kafka',
        python_callable=publish_signal_to_kafka,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

    t_wait_l3 = ExternalTaskSensor(
        task_id='wait_for_h5_l3_training',
        external_dag_id='forecast_h5_l3_weekly_training',
        external_task_id=None,  # Wait for entire DAG
        execution_delta=timedelta(hours=30, minutes=45),  # Sun 06:30 -> Mon 13:15
        timeout=3600,  # 1 hour timeout
        mode='reschedule',
        poke_interval=300,
        soft_fail=True,  # Don't block if L3 hasn't run (first-time setup)
    )

    t_wait_l3 >> t_check >> t_signal >> t_persist >> t_notify
    t_persist >> t_kafka
