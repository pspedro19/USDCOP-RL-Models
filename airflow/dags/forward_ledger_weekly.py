"""Weekly forward-ledger recording — the graduation evidence, on a clock.

Contract: CTR-FORWARD-TRACK-001

The forward ledger is the organ every withdrawal protocol depends on: graduation criterion #4
(|paper − replay| divergence) is unevaluable without a weekly series, and until today the
ledger was fed BY HAND. A hand-fed evidence stream has a failure mode no test catches: nobody
runs it for six weeks, and in November there is no series to judge — the 26-week window ends
with nothing, which is exactly the state the whole system was in before 2026-07-21.

Schedule: Friday 19:00 UTC (14:00 COT), after the COP session closes (12:55 COT) and after
the H5 L6 weekly monitor (14:30 COT Friday) has had its slot — the week is over, the paper
YTD snapshots are final for the week.

The recorder is append-only and idempotent per (strategy, week): re-running the DAG cannot
rewrite history, and a corrected week is a NEW row (--amend, deliberately NOT used here —
corrections are a human decision, not a cron's).
"""
from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

REPO = Path("/opt/airflow")  # dags mount root; scripts/ is a sibling mount

default_args = {
    "owner": "trading-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _record(**context):
    """Run the recorder for the ISO week of the logical date."""
    logical = context["logical_date"]
    y, w, _ = logical.isocalendar()
    week = f"{y}-W{int(w):02d}"
    cmd = [sys.executable, "-m", "scripts.analysis.forward_tracker", "--record",
           "--week", week]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, timeout=1800)
    print(r.stdout)
    if r.returncode != 0:
        # The recorder's own exit code is the verdict; do not soften it.
        raise RuntimeError(f"forward_tracker failed rc={r.returncode}: {r.stderr[-800:]}")
    # Absent legs are named in stdout (FALTA: paper/replay). They are legitimate states, but
    # a week where EVERY strategy is missing both sides means the recorder ran against a dead
    # environment, and that must fail loudly rather than append four hollow rows.
    if r.stdout.count("FALTA: paper+replay") >= 4:
        raise RuntimeError("every strategy missing both paper and replay -- dead environment?")


with DAG(
    dag_id="forward_ledger_weekly",
    description="Append one immutable forward-evidence row per (champion, week)",
    schedule_interval="0 19 * * 5",   # Fri 19:00 UTC = 14:00 COT, post-session, post-L6
    start_date=datetime(2026, 7, 14),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["forward", "evidence", "all-assets"],
) as dag:
    PythonOperator(task_id="record_week", python_callable=_record)
