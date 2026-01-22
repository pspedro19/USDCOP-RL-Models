"""
DAG: v3.alert_monitor
Latency & Data Integrity Monitoring

SOLID Compliance:
- SRP: Each function monitors one aspect
- OCP: Thresholds configurable via feature_config.json
- DIP: Uses shared utilities from dag_common

All thresholds read from config/feature_config.json (SSOT).
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import psycopg2
import os, logging, json

# DRY: Using shared utilities
from utils.dag_common import get_db_connection, load_feature_config

from contracts.dag_registry import L6_ALERT_MONITOR

DAG_ID = L6_ALERT_MONITOR

# Load config with thresholds (SSOT - no hardcoded values)
CONFIG = load_feature_config(raise_on_error=False)
MONITORING = CONFIG.get("monitoring", {})

# Extract thresholds from config (with safe defaults from SSOT)
OHLCV_LAG_THRESHOLD_SECONDS = MONITORING.get("ohlcv_lag_threshold_seconds", 600)
MACRO_AGE_THRESHOLD_HOURS = MONITORING.get("macro_age_threshold_hours", 24)
EXPECTED_INFERENCES_PER_HOUR = MONITORING.get("expected_inferences_per_hour", 12)
MIN_OHLCV_ROWS = MONITORING.get("min_ohlcv_rows", 1000)
MIN_MACRO_ROWS = MONITORING.get("min_macro_rows", 50)


def monitor_ohlcv_latency(**ctx):
    """Monitor OHLCV data latency. Threshold from config."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT MAX(time) as latest_time,
                   EXTRACT(EPOCH FROM (NOW() - MAX(time))) as lag_seconds
            FROM usdcop_m5_ohlcv
        """)
        result = cur.fetchone()
        if result:
            latest, lag = result
            logging.info(f"OHLCV: {latest}, Lag: {lag}s")
            # Use config threshold (SSOT)
            if lag and lag > OHLCV_LAG_THRESHOLD_SECONDS:
                logging.warning(
                    f"OHLCV lag exceeds threshold ({OHLCV_LAG_THRESHOLD_SECONDS}s): {lag}s"
                )
            ctx["ti"].xcom_push(key="ohlcv_lag_seconds", value=float(lag) if lag else None)
        return {"status": "ok", "threshold": OHLCV_LAG_THRESHOLD_SECONDS}
    finally:
        cur.close()
        conn.close()


def monitor_macro_latency(**ctx):
    """Monitor macro data freshness. Threshold from config."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT date,
                   EXTRACT(EPOCH FROM (NOW() - date::timestamp + interval '3 hours')) / 3600 as age_hours
            FROM macro_indicators_daily
            ORDER BY date DESC LIMIT 1
        """)
        result = cur.fetchone()
        if result:
            date, age = result
            logging.info(f"Macro: {date}, Age: {age}h")
            # Use config threshold (SSOT)
            if age and age > MACRO_AGE_THRESHOLD_HOURS:
                logging.warning(
                    f"Macro data older than threshold ({MACRO_AGE_THRESHOLD_HOURS}h): {age}h"
                )
            ctx["ti"].xcom_push(key="macro_age_hours", value=float(age) if age else None)
        return {"status": "ok", "threshold": MACRO_AGE_THRESHOLD_HOURS}
    finally:
        cur.close()
        conn.close()


def monitor_inference_health(**ctx):
    """Monitor RL inference frequency. Threshold from config."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT COUNT(*) as inference_count,
                   MAX(timestamp) as latest_inference
            FROM dw.fact_rl_inference
            WHERE timestamp > NOW() - INTERVAL '1 hour'
        """)
        result = cur.fetchone()
        if result:
            count, latest = result
            logging.info(f"Inferences last hour: {count}")
            # Use config threshold (SSOT)
            if count < EXPECTED_INFERENCES_PER_HOUR:
                logging.warning(
                    f"Low inference count: {count} (expected ~{EXPECTED_INFERENCES_PER_HOUR})"
                )
            ctx["ti"].xcom_push(key="inference_count_1h", value=count)
        return {"status": "ok", "threshold": EXPECTED_INFERENCES_PER_HOUR}
    finally:
        cur.close()
        conn.close()


def check_data_integrity(**ctx):
    """Check data integrity. Thresholds from config."""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        alerts = []

        cur.execute("SELECT COUNT(*) FROM usdcop_m5_ohlcv")
        ohlcv_count = cur.fetchone()[0]
        # Use config threshold (SSOT)
        if ohlcv_count < MIN_OHLCV_ROWS:
            alerts.append(f"Low OHLCV count: {ohlcv_count} (min: {MIN_OHLCV_ROWS})")

        cur.execute("SELECT COUNT(*) FROM macro_indicators_daily")
        macro_count = cur.fetchone()[0]
        # Use config threshold (SSOT)
        if macro_count < MIN_MACRO_ROWS:
            alerts.append(f"Low macro count: {macro_count} (min: {MIN_MACRO_ROWS})")

        if alerts:
            logging.warning("Data integrity alerts: " + "; ".join(alerts))

        ctx["ti"].xcom_push(key="alerts", value=alerts)
        return {
            "status": "ok",
            "alerts": alerts,
            "thresholds": {
                "min_ohlcv_rows": MIN_OHLCV_ROWS,
                "min_macro_rows": MIN_MACRO_ROWS
            }
        }
    finally:
        cur.close()
        conn.close()


default_args = {
    "owner": "trading-team",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=5)
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description="V3 Alert Monitor: Latency & data integrity checks (config-driven)",
    schedule_interval="*/15 13-17 * * 1-5",
    catchup=False,
    max_active_runs=1,
    tags=["v3", "monitoring", "alerts"]
)

with dag:
    ohlcv_check = PythonOperator(
        task_id="monitor_ohlcv",
        python_callable=monitor_ohlcv_latency,
        provide_context=True
    )
    macro_check = PythonOperator(
        task_id="monitor_macro",
        python_callable=monitor_macro_latency,
        provide_context=True
    )
    inference_check = PythonOperator(
        task_id="monitor_inference",
        python_callable=monitor_inference_health,
        provide_context=True
    )
    integrity_check = PythonOperator(
        task_id="check_integrity",
        python_callable=check_data_integrity,
        provide_context=True
    )

    [ohlcv_check, macro_check, inference_check] >> integrity_check
