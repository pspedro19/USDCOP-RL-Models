"""
Daily Reconciliation DAG — Compare internal signals vs exchange fills.

Schedule: Mon-Fri at 20:00 UTC (15:00 COT), after market close.
Compares H5 and H1 executions in our DB against fills from SignalBridge/MEXC.

Flags: missed fills, slippage > 2 bps, quantity mismatches.
Results written to reconciliation_runs + reconciliation_items tables.
"""

import logging
import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

default_args = {
    "owner": "trading-team",
    "depends_on_past": False,
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _get_db_conn():
    """Get a psycopg2 connection to the trading database."""
    import psycopg2

    return psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=os.environ.get("POSTGRES_PASSWORD", ""),
    )


def reconcile_pipeline(**context):
    """Run reconciliation for a specific pipeline (h5 or h1)."""
    import sys

    sys.path.insert(0, "/opt/airflow")

    from src.reconciliation.engine import ReconciliationEngine

    pipeline = context["params"].get("pipeline", "h5")
    exchange = context["params"].get("exchange", "mexc")
    signalbridge_url = os.environ.get(
        "SIGNALBRIDGE_URL", "http://signalbridge-api:8000"
    )

    # Reconcile yesterday's trades (today's market hasn't closed yet)
    execution_date = context["ds"]
    from datetime import date as date_type

    recon_date = date_type.fromisoformat(execution_date)

    conn = _get_db_conn()
    try:
        engine = ReconciliationEngine(
            db_conn=conn,
            signalbridge_url=signalbridge_url,
            slippage_threshold_bps=2.0,
        )
        result = engine.reconcile(
            run_date=recon_date,
            pipeline=pipeline,
            exchange=exchange,
        )

        logger.info(
            f"Reconciliation {pipeline} {recon_date}: "
            f"checked={result.signals_checked}, "
            f"matches={result.matches}, "
            f"discrepancies={result.discrepancies}, "
            f"missed={result.missed_fills}"
        )

        # Push metrics for downstream alerting
        context["ti"].xcom_push(key="signals_checked", value=result.signals_checked)
        context["ti"].xcom_push(key="discrepancies", value=result.discrepancies)
        context["ti"].xcom_push(key="missed_fills", value=result.missed_fills)

    finally:
        conn.close()


def alert_discrepancies(**context):
    """Alert if there are discrepancies in reconciliation."""
    h5_disc = context["ti"].xcom_pull(
        task_ids="reconcile_h5", key="discrepancies"
    ) or 0
    h1_disc = context["ti"].xcom_pull(
        task_ids="reconcile_h1", key="discrepancies"
    ) or 0

    total = h5_disc + h1_disc
    if total > 0:
        logger.warning(
            f"RECONCILIATION ALERT: {total} discrepancies found "
            f"(H5={h5_disc}, H1={h1_disc}). Check reconciliation_items table."
        )
    else:
        logger.info("Reconciliation clean — zero discrepancies.")


with DAG(
    dag_id="reconciliation_daily",
    default_args=default_args,
    description="Daily reconciliation: internal signals vs exchange fills",
    schedule_interval="0 20 * * 1-5",  # Mon-Fri 20:00 UTC = 15:00 COT
    start_date=datetime(2026, 3, 15),
    catchup=False,
    max_active_runs=1,
    tags=["reconciliation", "risk", "monitoring"],
) as dag:

    reconcile_h5 = PythonOperator(
        task_id="reconcile_h5",
        python_callable=reconcile_pipeline,
        params={"pipeline": "h5", "exchange": "mexc"},
    )

    reconcile_h1 = PythonOperator(
        task_id="reconcile_h1",
        python_callable=reconcile_pipeline,
        params={"pipeline": "h1", "exchange": "mexc"},
    )

    alert = PythonOperator(
        task_id="alert_discrepancies",
        python_callable=alert_discrepancies,
        trigger_rule="all_done",  # Run even if a reconcile task fails
    )

    [reconcile_h5, reconcile_h1] >> alert
