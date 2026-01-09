"""
MLOps Drift Monitoring DAG
==========================

Scheduled DAG for monitoring data drift in production.
Runs hourly during trading hours to detect feature distribution changes.

Features:
- Loads reference data from training period
- Compares against recent production data
- Generates alerts on significant drift
- Stores drift reports in MinIO
"""

from datetime import datetime, timedelta
from typing import Dict, Any

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# Default Arguments
# ============================================================================

default_args = {
    'owner': 'mlops',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ============================================================================
# Task Functions
# ============================================================================

def load_reference_data(**context) -> Dict[str, Any]:
    """Load reference data from database (last 30 days of training data)."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    logger.info("Loading reference data...")

    conn_string = "postgresql://postgres:postgres@postgres:5432/usdcop"

    query = """
    SELECT
        returns_5m, returns_15m, returns_1h,
        volatility_5m, volatility_15m,
        rsi_14, macd, macd_signal,
        sma_ratio_20, ema_ratio_12,
        volume_ratio, atr_14
    FROM fact_features_5m
    WHERE timestamp >= NOW() - INTERVAL '30 days'
      AND timestamp < NOW() - INTERVAL '1 day'
    ORDER BY timestamp
    """

    try:
        with psycopg2.connect(conn_string) as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} reference samples")

        # Store in XCom
        return {
            "num_samples": len(df),
            "columns": list(df.columns),
            "reference_data_json": df.to_json(orient='records'),
        }

    except Exception as e:
        logger.error(f"Failed to load reference data: {e}")
        raise


def load_current_data(**context) -> Dict[str, Any]:
    """Load current production data (last hour)."""
    import psycopg2

    logger.info("Loading current data...")

    conn_string = "postgresql://postgres:postgres@postgres:5432/usdcop"

    query = """
    SELECT
        returns_5m, returns_15m, returns_1h,
        volatility_5m, volatility_15m,
        rsi_14, macd, macd_signal,
        sma_ratio_20, ema_ratio_12,
        volume_ratio, atr_14
    FROM fact_features_5m
    WHERE timestamp >= NOW() - INTERVAL '1 hour'
    ORDER BY timestamp
    """

    try:
        with psycopg2.connect(conn_string) as conn:
            df = pd.read_sql(query, conn)

        logger.info(f"Loaded {len(df)} current samples")

        return {
            "num_samples": len(df),
            "columns": list(df.columns),
            "current_data_json": df.to_json(orient='records'),
        }

    except Exception as e:
        logger.error(f"Failed to load current data: {e}")
        raise


def check_drift(**context) -> Dict[str, Any]:
    """Check for data drift between reference and current data."""
    from scipy import stats

    ti = context['ti']

    # Get data from XCom
    ref_result = ti.xcom_pull(task_ids='load_reference_data')
    cur_result = ti.xcom_pull(task_ids='load_current_data')

    if not ref_result or not cur_result:
        raise ValueError("Missing data from upstream tasks")

    reference_df = pd.read_json(ref_result['reference_data_json'], orient='records')
    current_df = pd.read_json(cur_result['current_data_json'], orient='records')

    logger.info(f"Checking drift: {len(reference_df)} ref vs {len(current_df)} current samples")

    if len(current_df) < 5:
        logger.warning("Insufficient current data for drift detection")
        return {
            "drift_detected": False,
            "drift_share": 0.0,
            "message": "Insufficient data",
            "drifted_features": [],
        }

    # Check drift for each feature
    drifted_features = []
    feature_scores = {}
    drift_threshold = 0.05  # p-value threshold

    for col in reference_df.columns:
        try:
            ref_values = reference_df[col].dropna().values
            cur_values = current_df[col].dropna().values

            if len(ref_values) > 0 and len(cur_values) > 0:
                statistic, p_value = stats.ks_2samp(ref_values, cur_values)
                feature_scores[col] = {
                    "ks_statistic": float(statistic),
                    "p_value": float(p_value),
                    "drift_detected": p_value < drift_threshold,
                }

                if p_value < drift_threshold:
                    drifted_features.append(col)
                    logger.warning(f"Drift detected in {col}: KS={statistic:.4f}, p={p_value:.4f}")

        except Exception as e:
            logger.error(f"Error checking drift for {col}: {e}")

    drift_share = len(drifted_features) / len(reference_df.columns) if reference_df.columns.any() else 0
    drift_detected = drift_share > 0.15  # 15% threshold

    result = {
        "timestamp": datetime.now().isoformat(),
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "drifted_features": drifted_features,
        "feature_scores": feature_scores,
        "reference_samples": len(reference_df),
        "current_samples": len(current_df),
    }

    logger.info(f"Drift check complete: drift_share={drift_share:.2%}, detected={drift_detected}")

    return result


def store_drift_report(**context) -> None:
    """Store drift report in Redis and MinIO."""
    import redis
    import json

    ti = context['ti']
    drift_result = ti.xcom_pull(task_ids='check_drift')

    if not drift_result:
        logger.warning("No drift result to store")
        return

    # Store in Redis
    try:
        r = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)

        # Store latest
        r.set("drift:latest", json.dumps(drift_result))

        # Add to history
        r.lpush("drift:log", json.dumps(drift_result))
        r.ltrim("drift:log", 0, 999)

        # Set alert if drift detected
        if drift_result.get("drift_detected"):
            r.setex("drift:alert", 3600, json.dumps(drift_result))
            logger.warning("🚨 DRIFT ALERT SET")

        logger.info("Drift report stored in Redis")

    except Exception as e:
        logger.error(f"Failed to store in Redis: {e}")


def send_alert_if_drift(**context) -> None:
    """Send alert if significant drift detected."""
    ti = context['ti']
    drift_result = ti.xcom_pull(task_ids='check_drift')

    if not drift_result:
        return

    if drift_result.get("drift_detected"):
        drift_share = drift_result.get("drift_share", 0)
        drifted = drift_result.get("drifted_features", [])

        alert_msg = f"""
🚨 DATA DRIFT ALERT

Time: {datetime.now().isoformat()}
Drift Share: {drift_share:.1%}
Drifted Features: {', '.join(drifted[:5])}{'...' if len(drifted) > 5 else ''}

Action Required: Review model performance and consider retraining.
        """

        logger.warning(alert_msg)

        # TODO: Send to Slack/email/PagerDuty
        # slack_webhook.send(alert_msg)

    else:
        logger.info("No significant drift detected - no alert needed")


# ============================================================================
# DAG Definition
# ============================================================================

with DAG(
    dag_id='mlops_drift_monitor',
    default_args=default_args,
    description='Monitor data drift for ML models',
    schedule_interval='0 9-12 * * 1-5',  # Hourly during trading hours, Mon-Fri
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['mlops', 'monitoring', 'drift'],
    max_active_runs=1,
) as dag:

    # Start
    start = EmptyOperator(task_id='start')

    # Load data
    load_ref = PythonOperator(
        task_id='load_reference_data',
        python_callable=load_reference_data,
    )

    load_cur = PythonOperator(
        task_id='load_current_data',
        python_callable=load_current_data,
    )

    # Check drift
    check = PythonOperator(
        task_id='check_drift',
        python_callable=check_drift,
    )

    # Store results
    store = PythonOperator(
        task_id='store_drift_report',
        python_callable=store_drift_report,
    )

    # Alert if needed
    alert = PythonOperator(
        task_id='send_alert_if_drift',
        python_callable=send_alert_if_drift,
    )

    # End
    end = EmptyOperator(
        task_id='end',
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Task dependencies
    start >> [load_ref, load_cur] >> check >> store >> alert >> end
