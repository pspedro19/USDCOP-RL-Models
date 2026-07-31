"""Prospective fail-closed USD/COP H1 regime shadow v2.

Runs after the local market close on weekdays.  Only Friday may append a new
prediction; other weekdays can append the next-session outcome.  Missed Friday
predictions are never reconstructed.
"""
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
SCRIPT = PROJECT_ROOT / "scripts/pipeline/generate_usdcop_h1_regime_shadow_v2.py"
EVALUATOR = PROJECT_ROOT / "scripts/validation/evaluate_usdcop_h1_shadow_v2.py"
INDEX = (
    PROJECT_ROOT
    / "usdcop-trading-dashboard/public/forecasting/usdcop/h1_regime_shadow_index.json"
)
EXPECTED_EXPERIMENT = "usdcop_h1_regime_shadow_v2"


def run_shadow_generator() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=15 * 60,
    )
    for line in result.stdout.splitlines()[-30:]:
        logger.info("[h1-shadow-v2] %s", line)
    for line in result.stderr.splitlines()[-20:]:
        logger.warning("[h1-shadow-v2:err] %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"H1 regime shadow v2 exited {result.returncode}")


def verify_shadow_v2() -> None:
    if not INDEX.exists():
        raise FileNotFoundError(f"Shadow v2 index missing: {INDEX}")
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    if document.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise RuntimeError("Dashboard index is not H1 regime shadow v2")
    if document.get("registration_valid") is not True:
        raise RuntimeError("H1 shadow v2 registration is invalid")
    if document.get("signal_authorized") is not False:
        raise RuntimeError("H1 shadow v2 must not authorize signals")
    if document.get("capital_authorized") is not False:
        raise RuntimeError("H1 shadow v2 must not authorize capital")
    if document.get("promotion_review_eligible") is not False:
        raise RuntimeError("Generator cannot self-authorize a promotion review")
    records = document.get("records", [])
    weeks = [str(item["iso_week"]) for item in records]
    if len(weeks) != len(set(weeks)):
        raise RuntimeError("H1 shadow v2 has duplicate committed weeks")
    if any(week < "2026-W31" for week in weeks):
        raise RuntimeError("H1 shadow v2 contains a pre-registration week")
    if any(item.get("signal_authorized") is not False for item in records):
        raise RuntimeError("A shadow record authorized a signal")
    if any(item.get("capital_authorized") is not False for item in records):
        raise RuntimeError("A shadow record authorized capital")
    logger.info(
        "[h1-shadow-v2] verified status=%s records=%s prediction_head=%s",
        document.get("status"),
        len(records),
        document.get("prediction_chain_head"),
    )


def run_prospective_evaluation() -> None:
    result = subprocess.run(
        [sys.executable, str(EVALUATOR)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=5 * 60,
    )
    for line in result.stdout.splitlines()[-30:]:
        logger.info("[h1-shadow-v2-evaluation] %s", line)
    for line in result.stderr.splitlines()[-20:]:
        logger.warning("[h1-shadow-v2-evaluation:err] %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"H1 regime shadow v2 evaluation exited {result.returncode}")


DEFAULT_ARGS = {
    "owner": "forecast-research",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=20),
}


with DAG(
    dag_id="forecast_h1_regime_shadow_v2",
    default_args=DEFAULT_ARGS,
    description="PIT snapshot + hash-chain USD/COP H1 shadow; never retrocommits",
    schedule_interval="30 20 * * 1-5",  # 15:30 America/Bogota, weekdays
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["forecasting", "research", "shadow", "usdcop", "h1", "pit", "v2"],
) as dag:
    generate = PythonOperator(
        task_id="commit_or_mature_h1_shadow_v2",
        python_callable=run_shadow_generator,
    )
    verify = PythonOperator(
        task_id="verify_h1_shadow_v2_integrity",
        python_callable=verify_shadow_v2,
    )
    evaluate = PythonOperator(
        task_id="evaluate_h1_shadow_v2_prospectively",
        python_callable=run_prospective_evaluation,
    )
    generate >> verify >> evaluate
