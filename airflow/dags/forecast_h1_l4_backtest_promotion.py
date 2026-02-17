"""
DAG: forecast_h1_l4_backtest_promotion
=======================================
H1 Daily Pipeline — L4 Backtest Promotion DAG (placeholder).

This DAG is a placeholder for the H1 daily pipeline's backtest promotion workflow.
The H1 pipeline currently uses scripts/generate_weekly_forecasts.py for training
and the /forecasting page for model comparison. A dedicated L4 backtest+export
workflow (similar to H5's train_and_export_smart_simple.py) is planned but not
yet implemented.

For now, use the manual workflow:
    python scripts/generate_weekly_forecasts.py       # Train + export to /forecasting
    python scripts/backtest_smart_executor.py          # H1-specific backtest

Schedule: None (manual trigger)

Author: Trading Team
Version: 0.1.0 (placeholder)
Date: 2026-02-17
Contract: FC-H1-L4-001
"""

from datetime import datetime, timedelta
import logging
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, '/opt/airflow')

from contracts.dag_registry import (
    FORECAST_H1_L4_BACKTEST_PROMOTION,
    get_dag_tags,
)

DAG_ID = FORECAST_H1_L4_BACKTEST_PROMOTION
DAG_TAGS_LIST = get_dag_tags(DAG_ID)

logger = logging.getLogger(__name__)


def placeholder_task(**context) -> None:
    """
    Placeholder: H1 L4 backtest promotion not yet implemented.
    Use scripts/generate_weekly_forecasts.py manually.
    """
    logger.info("=" * 60)
    logger.info("[H1-L4] PLACEHOLDER — H1 backtest promotion not yet implemented")
    logger.info("[H1-L4] Manual workflow:")
    logger.info("[H1-L4]   1. python scripts/generate_weekly_forecasts.py")
    logger.info("[H1-L4]   2. python scripts/backtest_smart_executor.py")
    logger.info("[H1-L4]   3. Review results on /forecasting page")
    logger.info("=" * 60)


default_args = {
    'owner': 'forecast-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2026, 2, 17),
    'retries': 0,
}

with DAG(
    dag_id=DAG_ID,
    default_args=default_args,
    description='H1 Daily — L4 backtest promotion (placeholder, use manual scripts)',
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
    tags=DAG_TAGS_LIST,
) as dag:

    t_placeholder = PythonOperator(
        task_id='placeholder',
        python_callable=placeholder_task,
        provide_context=True,
    )
