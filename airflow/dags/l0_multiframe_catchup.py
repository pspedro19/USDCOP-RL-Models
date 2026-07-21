"""L0 multi-timeframe catch-up: keep every native layer alive (CTR-MKT-CANON-001).

Root cause this closes (quality plan 2026-07-22, Fase 3/4): the max-history backfill was a
one-shot — XAU/BTC M5 had NO scheduled ingestion at all (the '21,560 XAU bars' were a July-3
manual load, not a feed), native 1h/4h aged from the moment they landed, and the 1h/4h
matviews were only refreshed by hand.

Hourly, cheap (~30 TwelveData credits/run): pulls each series from its own last bar via
`backfill_max_history.catchup()` — M5 (XAU/COP-gaps/MXN/BRL via TwelveData, BTC via Binance),
native 1h/4h/1month, daily continuity (incl. SPY + BTC daily), then REFRESH of the derived
matviews. Every window logs to market_ingestion_manifest; available_at stamped at insert.
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, "/opt/airflow/dags")
from utils.run_status import honest_leaf  # noqa: E402

PROJECT_ROOT = Path("/opt/airflow")
DAG_ID = "l0_multiframe_catchup"


@honest_leaf
def run_catchup(**_):
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.ops.backfill_max_history", "--phase", "catchup"],
        cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=3000)
    print(proc.stdout[-4000:])
    if proc.returncode != 0:
        print(proc.stderr[-4000:])
        raise RuntimeError(f"catchup exit {proc.returncode}")


with DAG(
    dag_id=DAG_ID,
    description="Catch-up horario multi-timeframe (M5/1h/4h/daily/monthly) + refresh matviews",
    schedule_interval="20 * * * *",   # hh:20 — clear of the */5 realtime and hourly macro slots
    start_date=datetime(2026, 7, 20),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "data-eng", "retries": 1, "retry_delay": timedelta(minutes=10)},
    tags=["l0", "ohlcv", "catchup", "CTR-MKT-CANON-001"],
) as dag:
    PythonOperator(task_id="multiframe_catchup", python_callable=run_catchup)
