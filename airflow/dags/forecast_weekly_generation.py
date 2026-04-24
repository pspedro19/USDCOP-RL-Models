"""
DAG: forecast_weekly_generation
=================================
Weekly generation of forecasting dashboard data (CSV + PNGs).

Runs scripts/generate_weekly_forecasts.py every Monday at 09:00 COT (14:00 UTC)
after the weekend H1-L3 + H5-L3 training DAGs have completed. Populates
/forecasting page with the current ISO week's forward forecasts from the
9-model zoo (Ridge/BR/ARD/XGBoost/LightGBM/CatBoost + hybrids).

Output:
    usdcop-trading-dashboard/public/forecasting/
        bi_dashboard_unified.csv    (126 rows: 63 backtest + 63 forward)
        backtest_{model}_h{horizon}.png  (63 PNGs)
        forward_{model}_{week}.png       (9 + ensembles per week)

Schedule: 0 14 * * 1  (Mon 14:00 UTC = Mon 09:00 COT)

Contract: FC-WEEKLY-GEN-001
Version: 1.0.0
Date: 2026-04-16
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/airflow")
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "generate_weekly_forecasts.py"
OUTPUT_DIR = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "forecasting"

DEFAULT_ARGS = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=30),
}


def _current_iso_week() -> str:
    """Return current ISO week as YYYY-WNN."""
    today = date.today()
    iso = today.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _run_generate_weekly_forecasts(**context):
    """Invoke generate_weekly_forecasts.py for the current ISO week.

    The script:
      1. Loads full dataset (OHLCV + macro from parquets)
      2. Trains 9 models × 7 horizons (walk-forward)
      3. Writes bi_dashboard_unified.csv (126 rows)
      4. Generates 63 backtest PNGs + ~12 forward PNGs per week
    """
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT_PATH}")

    current_week = _current_iso_week()
    logger.info(f"Generating forecasting for {current_week}")
    logger.info(f"Script: {SCRIPT_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--week",
        current_week,
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=25 * 60,
    )

    if result.stdout:
        for line in result.stdout.splitlines()[-50:]:
            logger.info(f"[forecast-gen] {line}")
    if result.stderr:
        for line in result.stderr.splitlines()[-20:]:
            logger.warning(f"[forecast-gen:err] {line}")

    if result.returncode != 0:
        raise RuntimeError(
            f"generate_weekly_forecasts.py exited {result.returncode} for {current_week}"
        )

    context["ti"].xcom_push(key="week", value=current_week)
    logger.info(f"[forecast-gen] Forecasting DONE for {current_week}")


def _verify_outputs(**context):
    """Validate that CSV + PNGs were written for this week."""
    ti = context["ti"]
    week = ti.xcom_pull(task_ids="generate_forecasts", key="week") or _current_iso_week()
    year, wk = week.split("-W")
    week_suffix = f"{year}_W{wk}"

    csv_path = OUTPUT_DIR / "bi_dashboard_unified.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV missing: {csv_path}")

    csv_size = csv_path.stat().st_size
    logger.info(f"[forecast-gen] CSV: {csv_path.name} ({csv_size} bytes)")

    # Check forward PNGs
    consensus_png = OUTPUT_DIR / f"forward_consensus_{week_suffix}.png"
    ridge_png = OUTPUT_DIR / f"forward_ridge_{week_suffix}.png"
    ensemble_png = OUTPUT_DIR / f"forward_ensemble_top_3_{week_suffix}.png"

    missing = [p for p in [consensus_png, ridge_png, ensemble_png] if not p.exists()]
    if missing:
        logger.warning(f"[forecast-gen] Missing expected PNGs: {[p.name for p in missing]}")
    else:
        logger.info(
            f"[forecast-gen] Forward PNGs OK: {consensus_png.name}, "
            f"{ridge_png.name}, {ensemble_png.name}"
        )

    # Count total W* PNGs for this week
    w_pngs = list(OUTPUT_DIR.glob(f"forward_*_{week_suffix}.png"))
    logger.info(f"[forecast-gen] Total forward PNGs for {week_suffix}: {len(w_pngs)}")


with DAG(
    dag_id="forecast_weekly_generation",
    default_args=DEFAULT_ARGS,
    description="Weekly forecasting dashboard generation (9 models × 7 horizons)",
    schedule="0 14 * * 1",  # Mon 14:00 UTC = Mon 09:00 COT (post-training)
    start_date=days_ago(1),
    catchup=False,
    tags=["forecasting", "weekly", "dashboard", "fc-weekly-gen-001"],
    max_active_runs=1,
) as dag:

    generate = PythonOperator(
        task_id="generate_forecasts",
        python_callable=_run_generate_weekly_forecasts,
    )

    verify = PythonOperator(
        task_id="verify_outputs",
        python_callable=_verify_outputs,
    )

    generate >> verify
