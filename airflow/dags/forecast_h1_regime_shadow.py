"""Prospective USD/COP H1 regime-shadow ledger — **WITHDRAWN EXPERIMENT (v1)**.

Runs Friday at 15:30 America/Bogota, after the daily seed backup. The task is
research-only and fails closed if the pre-registered contract hash changes.

**This experiment was retired.** `.claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md`
records it withdrawn with 0 predictions and 0 outcomes (it did not reproduce the
researched model — `class_weight=None` live vs `balanced` — consumed mutable history and
pinned no code/runtime hashes). It is in `DEPRECATED_DAGS`.

The module is kept on disk because its generator and contract are the evidence trail of a
pre-registered experiment, and the constitution's §1 forbids deleting the record of a look.
But `usdcop_h1_regime_shadow_v2` now **owns the same ledger index this module writes**, and
the two DAGs carry identical tags — so a single "unpause the shadow DAGs" would let a
retired experiment overwrite the prospective chain that §5 makes the only clean judge.
The generator therefore refuses to run unless the ledger still belongs to v1. See
`assert_ledger_is_ours()`.
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
SCRIPT = PROJECT_ROOT / "scripts/pipeline/generate_usdcop_h1_regime_shadow.py"
INDEX = (
    PROJECT_ROOT
    / "usdcop-trading-dashboard/public/forecasting/usdcop/h1_regime_shadow_index.json"
)


EXPERIMENT_ID = "usdcop_h1_regime_shadow_v1"


def assert_ledger_is_ours() -> None:
    """Refuse to write if the ledger index no longer belongs to this experiment.

    Fail-closed on purpose, and deliberately BEFORE the subprocess: the generator writes
    the index and `verify_shadow_contract` only checks the document it just produced for
    internal consistency, so by the time verification runs the overwrite has happened. A
    missing `experiment_id` counts as foreign — an unlabelled ledger is not proof of
    ownership.
    """
    if not INDEX.exists():
        return  # nothing to overwrite; the generator bootstraps its own file
    try:
        document = json.loads(INDEX.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"Refusing to run: cannot read the ledger at {INDEX} to prove ownership ({exc})"
        ) from exc
    owner = document.get("experiment_id")
    if owner != EXPERIMENT_ID:
        raise RuntimeError(
            f"REFUSING TO RUN: {INDEX.name} belongs to experiment {owner!r}, not "
            f"{EXPERIMENT_ID!r}. This DAG is a WITHDRAWN experiment (0 predictions, 0 "
            "outcomes) and running it would overwrite the live prospective ledger of its "
            "successor. If you meant to run the shadow ledger, unpause "
            "forecast_h1_regime_shadow_v2 instead."
        )


def run_shadow_generator() -> None:
    assert_ledger_is_ours()
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Shadow generator missing: {SCRIPT}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=15 * 60,
    )
    if result.stdout:
        for line in result.stdout.splitlines()[-30:]:
            logger.info("[h1-regime-shadow] %s", line)
    if result.stderr:
        for line in result.stderr.splitlines()[-20:]:
            logger.warning("[h1-regime-shadow:err] %s", line)
    if result.returncode != 0:
        raise RuntimeError(f"H1 regime shadow exited {result.returncode}")


def verify_shadow_contract() -> None:
    if not INDEX.exists():
        raise FileNotFoundError(f"Shadow index missing: {INDEX}")
    document = json.loads(INDEX.read_text(encoding="utf-8"))
    if document.get("signal_authorized") is not False:
        raise RuntimeError("H1 regime shadow must not authorize signals")
    if document.get("capital_authorized") is not False:
        raise RuntimeError("H1 regime shadow must not authorize capital")
    if document.get("registration_valid") is not True:
        raise RuntimeError("H1 regime shadow registration is invalid")
    records = document.get("records", [])
    weeks = [str(item["iso_week"]) for item in records]
    if len(weeks) != len(set(weeks)):
        raise RuntimeError("H1 regime shadow contains duplicate ISO weeks")
    if any(week < "2026-W31" for week in weeks):
        raise RuntimeError("H1 regime shadow contains a pre-registration prediction")
    hashes = {str(item["contract_sha256"]) for item in records}
    if hashes and hashes != {str(document["contract_sha256"])}:
        raise RuntimeError("H1 regime shadow record hash does not match the index")
    logger.info(
        "[h1-regime-shadow] verified status=%s records=%s data_cutoff=%s",
        document.get("status"), len(records), document.get("data_cutoff"),
    )


DEFAULT_ARGS = {
    "owner": "forecast-research",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=20),
}


with DAG(
    dag_id="forecast_h1_regime_shadow",
    default_args=DEFAULT_ARGS,
    description="Immutable prospective USD/COP H1 regime shadow (no capital authorization)",
    schedule_interval="30 20 * * 5",  # Friday 15:30 America/Bogota
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["forecasting", "research", "shadow", "usdcop", "h1"],
) as dag:
    generate = PythonOperator(
        task_id="commit_h1_regime_shadow",
        python_callable=run_shadow_generator,
    )
    verify = PythonOperator(
        task_id="verify_h1_regime_shadow",
        python_callable=verify_shadow_contract,
    )
    generate >> verify
