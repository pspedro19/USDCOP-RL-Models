"""
Analysis L8 Daily Generation (SDD-10)
========================================
Mon-Fri 14:00 COT: Generate AI daily analysis + macro snapshots + chart PNGs.
On Friday: Also generate weekly summary.
Exports JSON to public/data/analysis/ for the dashboard.

DAG ID: analysis_l8_daily_generation
Schedule: 0 19 * * 1-5  (UTC = 14:00 COT)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

DEFAULT_ARGS = {
    "owner": "usdcop",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=20),
}


def _generate_daily_analysis(**context):
    """Generate daily AI analysis for today."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.analysis.weekly_generator import WeeklyAnalysisGenerator

    generator = WeeklyAnalysisGenerator(dry_run=False)
    today = date.today()

    record = generator.generate_for_date(today)

    logger.info(
        f"Daily analysis for {today}: "
        f"headline='{record.headline}', "
        f"sentiment={record.sentiment}, "
        f"tokens={record.llm_tokens_used}, "
        f"cost=${record.llm_cost_usd:.4f}"
    )

    context["ti"].xcom_push(key="analysis_date", value=str(today))
    context["ti"].xcom_push(key="sentiment", value=record.sentiment)


def _check_if_friday(**context):
    """Branch: if Friday, also generate weekly summary."""
    today = date.today()
    if today.weekday() == 4:  # Friday
        return "generate_weekly_summary"
    return "skip_weekly"


def _generate_weekly_summary(**context):
    """Generate weekly AI summary (Friday only)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.analysis.weekly_generator import WeeklyAnalysisGenerator

    generator = WeeklyAnalysisGenerator(dry_run=False)
    today = date.today()
    iso_cal = today.isocalendar()
    year, week = iso_cal[0], iso_cal[1]

    export = generator.generate_for_week(year, week)

    logger.info(
        f"Weekly analysis for {year}-W{week:02d}: "
        f"daily_entries={len(export.daily_entries)}, "
        f"macro_vars={len(export.macro_snapshots)}, "
        f"charts={len(export.macro_charts)}, "
        f"total_cost=${generator.llm.total_cost:.4f}"
    )


def _export_json(**context):
    """Ensure JSON files are in the dashboard public directory."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    # The weekly_generator already exports to the correct directory.
    # This task validates the files exist.
    project_root = Path(__file__).resolve().parents[2]
    analysis_dir = project_root / "usdcop-trading-dashboard" / "public" / "data" / "analysis"

    index_file = analysis_dir / "analysis_index.json"
    if index_file.exists():
        import json
        with open(index_file) as f:
            data = json.load(f)
        logger.info(f"Analysis index: {len(data.get('weeks', []))} weeks available")
    else:
        logger.warning("analysis_index.json not found — weekly generation may not have run yet")


with DAG(
    dag_id="analysis_l8_daily_generation",
    default_args=DEFAULT_ARGS,
    description="Analysis: daily AI analysis + weekly summary on Fridays",
    schedule="0 19 * * 1-5",  # UTC 19:00 = COT 14:00
    start_date=days_ago(1),
    catchup=False,
    tags=["analysis", "l8", "llm", "sdd-10"],
    max_active_runs=1,
) as dag:

    wait_for_news = ExternalTaskSensor(
        task_id='wait_for_news_pipeline',
        external_dag_id='news_daily_pipeline',
        external_task_id=None,  # Wait for entire DAG
        execution_delta=timedelta(hours=1),  # News runs at 18:00, analysis at 19:00
        timeout=1800,  # 30 min timeout
        mode='reschedule',
        poke_interval=120,
        soft_fail=True,  # Don't block analysis if news DAG hasn't run yet
    )

    daily = PythonOperator(
        task_id="generate_daily_analysis",
        python_callable=_generate_daily_analysis,
    )

    check_friday = BranchPythonOperator(
        task_id="check_if_friday",
        python_callable=_check_if_friday,
    )

    weekly = PythonOperator(
        task_id="generate_weekly_summary",
        python_callable=_generate_weekly_summary,
    )

    skip_weekly = EmptyOperator(task_id="skip_weekly")

    export = PythonOperator(
        task_id="export_json",
        python_callable=_export_json,
        trigger_rule="none_failed_min_one_success",
    )

    wait_for_news >> daily >> check_friday >> [weekly, skip_weekly] >> export
