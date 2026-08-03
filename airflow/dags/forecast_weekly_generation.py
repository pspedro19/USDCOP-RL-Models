"""
DAG: forecast_weekly_generation
=================================
Weekly generation of forecasting dashboard data (CSV + PNGs).

Runs scripts/pipeline/generate_weekly_forecasts.py every Monday at 09:00 COT (14:00 UTC)
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

import json
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
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "pipeline" / "generate_weekly_forecasts.py"
DIRECTIONAL_SCRIPT_PATH = (
    PROJECT_ROOT / "scripts" / "pipeline" / "generate_usdcop_directional_replay.py"
)
OUTPUT_DIR = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "forecasting"

# Whole-year coverage: regenerate the last N ISO weeks each run so /forecasting shows the
# year (not just the current week). NOTE: generate_weekly_forecasts.py OVERWRITES the CSV each
# run, so a single `--week` invocation would collapse the page to ONE week — `--num-weeks` keeps
# the whole window. Cost scales ~linearly (each week retrains 7 horizons × ~7 models, walk-forward);
# raise toward 52 for a full calendar year at higher runtime (bump the timeouts below to match).
NUM_WEEKS = 30

DEFAULT_ARGS = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=60),
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
    logger.info(f"Generating whole-year forecasting (last {NUM_WEEKS} weeks, current={current_week})")
    logger.info(f"Script: {SCRIPT_PATH}")
    logger.info(f"Output: {OUTPUT_DIR}")

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--num-weeks",
        str(NUM_WEEKS),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=55 * 60,
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


def _run_generate_directional_replay(**context):
    """Publish the complete frozen-2025 / expanding-2026 causal replay."""
    if not DIRECTIONAL_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Script not found: {DIRECTIONAL_SCRIPT_PATH}")
    result = subprocess.run(
        [sys.executable, str(DIRECTIONAL_SCRIPT_PATH)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=15 * 60,
    )
    if result.stdout:
        for line in result.stdout.splitlines()[-40:]:
            logger.info(f"[directional-replay] {line}")
    if result.stderr:
        for line in result.stderr.splitlines()[-20:]:
            logger.warning(f"[directional-replay:err] {line}")
    if result.returncode != 0:
        raise RuntimeError(f"directional replay exited {result.returncode}")


def _verify_outputs(**context):
    """Validate that the legacy model-zoo CSV + PNGs were written."""
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



def _verify_directional_outputs(**context):
    """Validate complete contiguous coverage and all directional images."""
    directional_index = OUTPUT_DIR / "usdcop" / "directional_replay_index.json"
    if not directional_index.exists():
        raise FileNotFoundError(f"Directional replay missing: {directional_index}")
    document = json.loads(directional_index.read_text(encoding="utf-8"))
    latest = document.get("latest_week")
    weeks = document.get("weeks", [])
    if not latest or not weeks:
        raise RuntimeError("Directional replay has no latest_week/weeks")
    if weeks[0].get("iso_week") != "2025-W01" or weeks[-1].get("iso_week") != latest:
        raise RuntimeError("Directional replay coverage is not contiguous from 2025-W01")
    latest_year, latest_number = map(int, latest.replace("W", "").split("-"))
    expected_weeks = sum(
        date(year, 12, 28).isocalendar().week
        for year in range(2025, latest_year)
    ) + latest_number
    if len(weeks) != expected_weeks:
        raise RuntimeError(
            f"Directional replay expected {expected_weeks} weeks, got {len(weeks)}"
        )
    if any(len(item.get("horizons", [])) != 7 for item in weeks):
        raise RuntimeError("Directional replay has a week without all seven horizons")
    missing_images = [
        item["image_path"]
        for item in weeks
        if not (OUTPUT_DIR / item["image_path"]).exists()
    ]
    if missing_images:
        raise FileNotFoundError(f"Directional charts missing: {missing_images[:5]}")
    logger.info(
        "[directional-replay] Contract %s: %s weeks through %s",
        document.get("contract_hash"), len(weeks), latest,
    )


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

    directional = PythonOperator(
        task_id="generate_usdcop_directional_replay",
        python_callable=_run_generate_directional_replay,
    )

    verify = PythonOperator(
        task_id="verify_model_zoo_outputs",
        python_callable=_verify_outputs,
    )

    verify_directional = PythonOperator(
        task_id="verify_usdcop_directional_replay",
        python_callable=_verify_directional_outputs,
    )

    generate >> verify
    directional >> verify_directional
