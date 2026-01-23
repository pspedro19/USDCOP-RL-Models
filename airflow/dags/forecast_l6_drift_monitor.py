"""
Forecast L6: Feature and Model Drift Monitor
=============================================

Monitors forecasting features and model predictions for drift.
Triggers alerts or retraining when drift is detected.

Schedule: Daily at 9:00 AM (after L1 features refresh)
Depends: forecast_l1_01_daily_features
Triggers: Alerts on drift detection

Drift Types Monitored:
    1. Feature Drift: Statistical changes in input features
    2. Concept Drift: Changes in target variable distribution
    3. Prediction Drift: Changes in model prediction patterns
    4. Performance Drift: Degradation in model accuracy

Author: Trading Team
Date: 2026-01-22
Contract: CTR-FORECAST-DRIFT-001
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import from SSOT
from airflow.dags.contracts.dag_registry import (
    FORECAST_L6_DRIFT_MONITOR,
    FORECAST_L1_DAILY_FEATURES,
    get_dag_tags,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DRIFT DETECTION CONFIGURATION
# =============================================================================

DRIFT_CONFIG = {
    # Feature drift thresholds
    "feature_drift": {
        "z_score_threshold": 3.0,       # Z-score for outlier detection
        "ks_stat_threshold": 0.2,       # Kolmogorov-Smirnov statistic
        "psi_threshold": 0.2,           # Population Stability Index
    },
    # Model performance thresholds
    "performance_drift": {
        "da_drop_threshold": 0.05,      # 5% drop in direction accuracy
        "rmse_increase_threshold": 0.5, # 50% increase in RMSE
    },
    # Reference window
    "reference_window_days": 30,
    "comparison_window_days": 7,
    # Alert settings
    "alert_on_feature_drift": True,
    "alert_on_performance_drift": True,
}

# Features to monitor for drift
MONITORED_FEATURES = [
    "close",
    "return_1d",
    "return_5d",
    "volatility_20d",
    "rsi_14d",
    "ma_ratio_20d",
]


# =============================================================================
# DRIFT DETECTION FUNCTIONS
# =============================================================================

def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI).

    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change
    """
    # Create bins from reference distribution
    breakpoints = np.linspace(reference.min(), reference.max(), bins + 1)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    # Calculate proportions in each bin
    ref_counts = np.histogram(reference, bins=breakpoints)[0]
    curr_counts = np.histogram(current, bins=breakpoints)[0]

    # Avoid division by zero
    ref_pct = (ref_counts + 0.0001) / (len(reference) + 0.0001 * bins)
    curr_pct = (curr_counts + 0.0001) / (len(current) + 0.0001 * bins)

    # Calculate PSI
    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))

    return psi


def calculate_ks_statistic(reference: pd.Series, current: pd.Series) -> float:
    """Calculate Kolmogorov-Smirnov statistic."""
    from scipy import stats
    statistic, _ = stats.ks_2samp(reference.dropna(), current.dropna())
    return statistic


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: List[str],
    thresholds: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    """Detect drift in specified features."""
    drift_results = {}

    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue

        ref_data = reference_df[feature].dropna()
        curr_data = current_df[feature].dropna()

        if len(ref_data) < 10 or len(curr_data) < 3:
            continue

        # Calculate drift metrics
        psi = calculate_psi(ref_data, curr_data)
        ks_stat = calculate_ks_statistic(ref_data, curr_data)

        # Z-score of current mean vs reference
        ref_mean = ref_data.mean()
        ref_std = ref_data.std()
        curr_mean = curr_data.mean()
        z_score = (curr_mean - ref_mean) / ref_std if ref_std > 0 else 0

        # Determine if drift detected
        drift_detected = (
            psi > thresholds["psi_threshold"] or
            ks_stat > thresholds["ks_stat_threshold"] or
            abs(z_score) > thresholds["z_score_threshold"]
        )

        drift_results[feature] = {
            "psi": psi,
            "ks_statistic": ks_stat,
            "z_score": z_score,
            "ref_mean": ref_mean,
            "curr_mean": curr_mean,
            "drift_detected": drift_detected,
        }

        if drift_detected:
            logger.warning(f"DRIFT DETECTED in {feature}: PSI={psi:.3f}, KS={ks_stat:.3f}, Z={z_score:.2f}")

    return drift_results


# =============================================================================
# AIRFLOW TASKS
# =============================================================================

def task_load_reference_data(**context) -> Dict[str, Any]:
    """Load reference period data for comparison."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    config = DRIFT_CONFIG

    end_date = (datetime.now() - timedelta(days=config["comparison_window_days"])).strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=config["reference_window_days"])).strftime("%Y-%m-%d")

    query = """
    SELECT *
    FROM bi.v_forecasting_features
    WHERE date BETWEEN %s AND %s
    ORDER BY date
    """

    df = hook.get_pandas_df(query, parameters=(start_date, end_date))
    logger.info(f"Loaded {len(df)} reference rows ({start_date} to {end_date})")

    context["ti"].xcom_push(key="reference_data", value=df.to_json(orient="records", date_format="iso"))
    return {"reference_rows": len(df), "date_range": [start_date, end_date]}


def task_load_current_data(**context) -> Dict[str, Any]:
    """Load current period data for drift detection."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    config = DRIFT_CONFIG

    start_date = (datetime.now() - timedelta(days=config["comparison_window_days"])).strftime("%Y-%m-%d")

    query = """
    SELECT *
    FROM bi.v_forecasting_features
    WHERE date >= %s
    ORDER BY date
    """

    df = hook.get_pandas_df(query, parameters=(start_date,))
    logger.info(f"Loaded {len(df)} current rows (from {start_date})")

    context["ti"].xcom_push(key="current_data", value=df.to_json(orient="records", date_format="iso"))
    return {"current_rows": len(df)}


def task_detect_feature_drift(**context) -> Dict[str, Any]:
    """Run feature drift detection."""
    ti = context["ti"]
    config = DRIFT_CONFIG

    reference_json = ti.xcom_pull(task_ids="load_reference_data", key="reference_data")
    current_json = ti.xcom_pull(task_ids="load_current_data", key="current_data")

    reference_df = pd.read_json(reference_json, orient="records")
    current_df = pd.read_json(current_json, orient="records")

    if len(reference_df) == 0 or len(current_df) == 0:
        logger.warning("Insufficient data for drift detection")
        return {"status": "insufficient_data"}

    drift_results = detect_feature_drift(
        reference_df,
        current_df,
        MONITORED_FEATURES,
        config["feature_drift"],
    )

    # Count drifted features
    drifted_count = sum(1 for r in drift_results.values() if r.get("drift_detected", False))

    ti.xcom_push(key="feature_drift_results", value=drift_results)
    ti.xcom_push(key="drifted_feature_count", value=drifted_count)

    logger.info(f"Feature drift detection: {drifted_count}/{len(drift_results)} features drifted")
    return {"total_features": len(drift_results), "drifted_features": drifted_count}


def task_detect_performance_drift(**context) -> Dict[str, Any]:
    """Check for performance drift in recent predictions."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    config = DRIFT_CONFIG

    # Get recent backtest metrics
    query = """
    SELECT
        model_id,
        horizon,
        direction_accuracy,
        rmse,
        backtest_date
    FROM bi.fact_forecast_backtest
    WHERE backtest_date >= CURRENT_DATE - INTERVAL '60 days'
    ORDER BY backtest_date DESC
    """

    try:
        df = hook.get_pandas_df(query)
    except Exception as e:
        logger.warning(f"Could not load backtest metrics: {e}")
        return {"status": "no_data"}

    if len(df) < 2:
        return {"status": "insufficient_data"}

    # Compare recent vs historical
    df["backtest_date"] = pd.to_datetime(df["backtest_date"])
    recent = df[df["backtest_date"] >= df["backtest_date"].max() - timedelta(days=7)]
    historical = df[df["backtest_date"] < df["backtest_date"].max() - timedelta(days=7)]

    if len(recent) == 0 or len(historical) == 0:
        return {"status": "insufficient_data"}

    recent_da = recent["direction_accuracy"].mean()
    historical_da = historical["direction_accuracy"].mean()
    da_change = recent_da - historical_da

    recent_rmse = recent["rmse"].mean()
    historical_rmse = historical["rmse"].mean()
    rmse_change = (recent_rmse - historical_rmse) / historical_rmse if historical_rmse > 0 else 0

    # Detect drift
    da_drift = da_change < -config["performance_drift"]["da_drop_threshold"]
    rmse_drift = rmse_change > config["performance_drift"]["rmse_increase_threshold"]

    result = {
        "recent_da": recent_da,
        "historical_da": historical_da,
        "da_change": da_change,
        "da_drift_detected": da_drift,
        "recent_rmse": recent_rmse,
        "historical_rmse": historical_rmse,
        "rmse_change_pct": rmse_change,
        "rmse_drift_detected": rmse_drift,
        "any_drift": da_drift or rmse_drift,
    }

    context["ti"].xcom_push(key="performance_drift_results", value=result)

    if da_drift:
        logger.warning(f"PERFORMANCE DRIFT: DA dropped from {historical_da:.1%} to {recent_da:.1%}")
    if rmse_drift:
        logger.warning(f"PERFORMANCE DRIFT: RMSE increased by {rmse_change:.1%}")

    return result


def task_check_drift_severity(**context) -> str:
    """Branch based on drift severity."""
    ti = context["ti"]

    feature_drift = ti.xcom_pull(task_ids="detect_feature_drift", key="drifted_feature_count") or 0
    performance_drift = ti.xcom_pull(task_ids="detect_performance_drift", key="performance_drift_results") or {}

    # Determine severity
    severe_drift = (
        feature_drift >= 3 or
        performance_drift.get("da_drift_detected", False) or
        performance_drift.get("rmse_drift_detected", False)
    )

    moderate_drift = feature_drift >= 1

    if severe_drift:
        return "trigger_alert"
    elif moderate_drift:
        return "log_warning"
    else:
        return "no_action"


def task_trigger_alert(**context) -> Dict[str, Any]:
    """Send alert for severe drift."""
    ti = context["ti"]

    feature_results = ti.xcom_pull(task_ids="detect_feature_drift", key="feature_drift_results") or {}
    performance_results = ti.xcom_pull(task_ids="detect_performance_drift", key="performance_drift_results") or {}

    # Build alert message
    drifted_features = [f for f, r in feature_results.items() if r.get("drift_detected", False)]

    alert_msg = f"""
    FORECASTING DRIFT ALERT
    =======================
    Time: {datetime.now().isoformat()}

    Feature Drift:
    - Drifted features: {', '.join(drifted_features) if drifted_features else 'None'}

    Performance Drift:
    - DA Change: {performance_results.get('da_change', 0):.1%}
    - RMSE Change: {performance_results.get('rmse_change_pct', 0):.1%}

    Recommended Action: Review and consider retraining models
    """

    logger.error(alert_msg)

    # Persist alert
    hook = PostgresHook(postgres_conn_id="postgres_default")
    hook.run("""
        CREATE TABLE IF NOT EXISTS bi.forecast_drift_alerts (
            id SERIAL PRIMARY KEY,
            alert_date TIMESTAMPTZ DEFAULT NOW(),
            alert_type VARCHAR(50),
            drifted_features TEXT[],
            da_change DECIMAL(10,6),
            rmse_change DECIMAL(10,6),
            severity VARCHAR(20)
        );

        INSERT INTO bi.forecast_drift_alerts
        (alert_type, drifted_features, da_change, rmse_change, severity)
        VALUES ('DRIFT_DETECTED', %s, %s, %s, 'SEVERE')
    """, parameters=(
        drifted_features,
        performance_results.get("da_change", 0),
        performance_results.get("rmse_change_pct", 0),
    ))

    return {"alert_sent": True, "drifted_features": drifted_features}


def task_log_warning(**context) -> Dict[str, Any]:
    """Log warning for moderate drift."""
    ti = context["ti"]
    feature_results = ti.xcom_pull(task_ids="detect_feature_drift", key="feature_drift_results") or {}

    drifted = [f for f, r in feature_results.items() if r.get("drift_detected", False)]
    logger.warning(f"Moderate drift detected in features: {drifted}")

    return {"warning_logged": True, "features": drifted}


def task_persist_drift_metrics(**context) -> Dict[str, Any]:
    """Persist drift metrics to PostgreSQL."""
    hook = PostgresHook(postgres_conn_id="postgres_default")
    ti = context["ti"]

    feature_results = ti.xcom_pull(task_ids="detect_feature_drift", key="feature_drift_results") or {}
    drifted_count = ti.xcom_pull(task_ids="detect_feature_drift", key="drifted_feature_count") or 0

    # Ensure table exists
    hook.run("""
        CREATE TABLE IF NOT EXISTS bi.forecast_drift_history (
            id SERIAL PRIMARY KEY,
            check_date DATE NOT NULL,
            feature_name VARCHAR(100),
            psi DECIMAL(10,6),
            ks_statistic DECIMAL(10,6),
            z_score DECIMAL(10,6),
            drift_detected BOOLEAN,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS idx_drift_history_date
            ON bi.forecast_drift_history (check_date DESC);
    """)

    check_date = datetime.now().strftime("%Y-%m-%d")

    for feature, metrics in feature_results.items():
        hook.run("""
            INSERT INTO bi.forecast_drift_history
            (check_date, feature_name, psi, ks_statistic, z_score, drift_detected)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, parameters=(
            check_date,
            feature,
            metrics.get("psi", 0),
            metrics.get("ks_statistic", 0),
            metrics.get("z_score", 0),
            metrics.get("drift_detected", False),
        ))

    logger.info(f"Persisted drift metrics for {len(feature_results)} features")
    return {"status": "success", "features_checked": len(feature_results), "drifted": drifted_count}


# =============================================================================
# DAG DEFINITION
# =============================================================================

default_args = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "email_on_failure": True,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id=FORECAST_L6_DRIFT_MONITOR,
    default_args=default_args,
    description="Monitor forecasting features and models for drift",
    schedule_interval="0 9 * * 1-5",  # 9 AM on weekdays
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=get_dag_tags(FORECAST_L6_DRIFT_MONITOR),
    doc_md=__doc__,
) as dag:

    load_reference = PythonOperator(
        task_id="load_reference_data",
        python_callable=task_load_reference_data,
    )

    load_current = PythonOperator(
        task_id="load_current_data",
        python_callable=task_load_current_data,
    )

    detect_feature = PythonOperator(
        task_id="detect_feature_drift",
        python_callable=task_detect_feature_drift,
    )

    detect_performance = PythonOperator(
        task_id="detect_performance_drift",
        python_callable=task_detect_performance_drift,
    )

    check_severity = BranchPythonOperator(
        task_id="check_drift_severity",
        python_callable=task_check_drift_severity,
    )

    alert = PythonOperator(
        task_id="trigger_alert",
        python_callable=task_trigger_alert,
    )

    warning = PythonOperator(
        task_id="log_warning",
        python_callable=task_log_warning,
    )

    no_action = DummyOperator(task_id="no_action")

    persist = PythonOperator(
        task_id="persist_drift_metrics",
        python_callable=task_persist_drift_metrics,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # Task dependencies
    [load_reference, load_current] >> detect_feature >> detect_performance >> check_severity
    check_severity >> [alert, warning, no_action] >> persist
