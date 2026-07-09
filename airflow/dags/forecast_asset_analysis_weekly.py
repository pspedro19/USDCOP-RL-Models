"""
DAG: forecast_asset_analysis_weekly
===================================
Weekly generation of MULTI-ASSET (Gold/BTC) weekly+daily analysis for the
/analysis page. Mirrors `forecast_weekly_generation` (which does COP forecasting)
but drives the science-stack analysis via
`scripts/pipeline/generate_asset_analysis.py`.

USD/COP analysis is produced separately by `generate_weekly_analysis.py`
(macro LangGraph pipeline) — this DAG covers only the assets declared in
`config/analysis/analysis_assets.yaml` (currently xauusd, btcusdt). Adding an
asset there makes it flow through here automatically (no DAG change).

Output (per asset in the SSOT):
    usdcop-trading-dashboard/public/data/analysis/<asset>/
        weekly_YYYY_WXX.json
        analysis_index.json
        upcoming_events.json

Data is REAL: daily OHLCV seeds → technicals, published strategy trades → signal,
Google News (primary) / GDELT (fallback) → news.

Schedule: 20 14 * * 1  (Mon 14:20 UTC = Mon 09:20 COT) — after the COP forecasting
generation (14:00) so the two dashboard-data jobs don't collide.

Contract: FC-ASSET-ANALYSIS-001
Version: 1.0.0
Date: 2026-07-05
"""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/opt/airflow")
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "pipeline" / "generate_asset_analysis.py"
CONFIG_PATH = PROJECT_ROOT / "config" / "analysis" / "analysis_assets.yaml"
ANALYSIS_DIR = PROJECT_ROOT / "usdcop-trading-dashboard" / "public" / "data" / "analysis"

DEFAULT_ARGS = {
    "owner": "forecast-pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=30),
}


def _configured_assets() -> list[str]:
    """Assets declared in the SSOT (drives what this DAG generates)."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return sorted((cfg.get("assets") or {}).keys())
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not read %s: %s", CONFIG_PATH, e)
        return ["xauusd", "btcusdt"]


def _run_asset_analysis(**context):
    """Invoke generate_asset_analysis.py for all configured assets, current year."""
    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Script not found: {SCRIPT_PATH}")

    year = date.today().year
    logger.info("Generating multi-asset analysis for %s (assets: %s)", year, _configured_assets())

    cmd = [sys.executable, str(SCRIPT_PATH), "--all-assets", "--year", str(year)]
    result = subprocess.run(
        cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=25 * 60,
    )
    for line in (result.stdout or "").splitlines()[-60:]:
        logger.info("[asset-analysis] %s", line)
    for line in (result.stderr or "").splitlines()[-20:]:
        logger.warning("[asset-analysis:err] %s", line)

    if result.returncode != 0:
        raise RuntimeError(f"generate_asset_analysis.py exited {result.returncode}")

    context["ti"].xcom_push(key="year", value=year)
    logger.info("[asset-analysis] DONE for %s", year)


def _verify_outputs(**context):
    """Validate each configured asset has an index + at least one weekly file."""
    problems = []
    for asset in _configured_assets():
        adir = ANALYSIS_DIR / asset
        idx = adir / "analysis_index.json"
        weeklies = list(adir.glob("weekly_*.json"))
        if not idx.exists():
            problems.append(f"{asset}: missing analysis_index.json")
        if not weeklies:
            problems.append(f"{asset}: no weekly_*.json files")
        logger.info("[asset-analysis] %s: %d weekly files, index=%s",
                    asset, len(weeklies), idx.exists())
    if problems:
        raise RuntimeError("Verification failed: " + "; ".join(problems))
    logger.info("[asset-analysis] verify OK for %s", _configured_assets())


with DAG(
    dag_id="forecast_asset_analysis_weekly",
    default_args=DEFAULT_ARGS,
    description="Weekly multi-asset (Gold/BTC) analysis generation for /analysis",
    schedule="20 14 * * 1",  # Mon 14:20 UTC = Mon 09:20 COT (after COP forecasting)
    start_date=days_ago(1),
    catchup=False,
    tags=["forecasting", "analysis", "multi-asset", "weekly", "dashboard",
          "fc-asset-analysis-001"],
    max_active_runs=1,
) as dag:

    generate = PythonOperator(
        task_id="generate_asset_analysis",
        python_callable=_run_asset_analysis,
    )
    verify = PythonOperator(
        task_id="verify_outputs",
        python_callable=_verify_outputs,
    )
    generate >> verify
