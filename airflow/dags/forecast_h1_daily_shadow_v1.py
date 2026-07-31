"""Prospective daily-origin H1 shadow with weekly model retraining."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path("/opt/airflow")
GENERATOR = PROJECT_ROOT / "scripts/pipeline/generate_usdcop_h1_daily_shadow_v1.py"
EVALUATOR = PROJECT_ROOT / "scripts/validation/evaluate_usdcop_h1_daily_shadow_v1.py"
INDEX = (
    PROJECT_ROOT
    / "usdcop-trading-dashboard/public/forecasting/usdcop/h1_daily_shadow_index.json"
)
EXPECTED_EXPERIMENT = "usdcop_h1_daily_shadow_v1"


def _run(script: Path, label: str, timeout_seconds: int) -> None:
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    for line in result.stdout.splitlines()[-30:]:
        logger.info("[%s] %s", label, line)
    for line in result.stderr.splitlines()[-20:]:
        logger.warning("[%s:err] %s", label, line)
    if result.returncode != 0:
        raise RuntimeError(f"{label} exited {result.returncode}")


def run_daily_shadow() -> None:
    _run(GENERATOR, "h1-daily-shadow", 15 * 60)


def verify_daily_shadow() -> None:
    if not INDEX.exists():
        raise FileNotFoundError(f"Daily H1 index missing: {INDEX}")
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    if document.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise RuntimeError("Dashboard index is not the registered daily H1 shadow")
    if document.get("registration_valid") is not True:
        raise RuntimeError("Daily H1 registration is invalid")
    for field in ("signal_authorized", "capital_authorized", "promotion_review_eligible"):
        if document.get(field) is not False:
            raise RuntimeError(f"Daily H1 shadow must keep {field}=false")
    records = document.get("records", [])
    origins = [str(item["origin_date"]) for item in records]
    if len(origins) != len(set(origins)):
        raise RuntimeError("Daily H1 shadow contains duplicate origins")
    if origins != sorted(origins):
        raise RuntimeError("Daily H1 shadow origins are not chronological")
    if any(origin < "2026-07-27" for origin in origins):
        raise RuntimeError("Daily H1 shadow contains a pre-registration origin")
    if any(item.get("signal_authorized") is not False for item in records):
        raise RuntimeError("A daily shadow record authorized a signal")
    if any(item.get("capital_authorized") is not False for item in records):
        raise RuntimeError("A daily shadow record authorized capital")
    logger.info(
        "[h1-daily-shadow] verified status=%s records=%s head=%s",
        document.get("status"),
        len(records),
        document.get("prediction_chain_head"),
    )


def run_daily_evaluation() -> None:
    _run(EVALUATOR, "h1-daily-shadow-evaluation", 5 * 60)


DEFAULT_ARGS = {
    "owner": "forecast-research",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=25),
}


with DAG(
    dag_id="forecast_h1_daily_shadow_v1",
    default_args=DEFAULT_ARGS,
    description="Daily H1 prospective shadow; weekly refit, current-session commits only",
    schedule_interval="35 20 * * 1-5",  # 15:35 America/Bogota, after the 15:00 seed backup
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=True,
    tags=["forecasting", "research", "shadow", "usdcop", "h1", "daily", "pit"],
) as dag:
    commit_or_mature = PythonOperator(
        task_id="commit_or_mature_daily_h1_shadow",
        python_callable=run_daily_shadow,
    )
    verify = PythonOperator(
        task_id="verify_daily_h1_shadow_integrity",
        python_callable=verify_daily_shadow,
    )
    evaluate = PythonOperator(
        task_id="evaluate_daily_h1_shadow_prospectively",
        python_callable=run_daily_evaluation,
    )
    commit_or_mature >> verify >> evaluate
