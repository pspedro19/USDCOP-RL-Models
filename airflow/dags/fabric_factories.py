"""Airflow adapter for FABRIC's data, ACTION and DIAGNOSTIC DAG factories."""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import yaml
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from src.orchestration.factories import DagSpec, build_all_specs

try:  # Airflow 3
    from airflow.sdk import Asset as _Dataset
except ImportError:  # Airflow 2
    from airflow.datasets import Dataset as _Dataset

PROJECT_ROOT = Path("/opt/airflow")
CONFIG_PATH = PROJECT_ROOT / "config" / "assets" / "fabric_factories.yaml"
logger = logging.getLogger(__name__)


def _run(script: str, args: tuple[str, ...]) -> None:
    path = PROJECT_ROOT / script
    if not path.is_file():
        raise FileNotFoundError(path)
    result = subprocess.run(
        [sys.executable, str(path), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=60 * 60,
    )
    if result.returncode:
        raise RuntimeError(f"{script} failed with exit code {result.returncode}: {result.stderr[-3000:]}")


def _build(spec: DagSpec) -> DAG:
    inputs = [_Dataset(uri) for uri in spec.consumes]
    outputs = [_Dataset(uri) for uri in spec.produces]
    dag = DAG(
        dag_id=spec.dag_id,
        schedule=spec.schedule,
        start_date=days_ago(1),
        catchup=False,
        max_active_runs=1,
        tags=list(spec.tags),
        default_args={"owner": "fabric", "depends_on_past": False},
    )
    with dag:
        previous = None
        for index, task_spec in enumerate(spec.tasks):
            task = PythonOperator(
                task_id=task_spec.task_id,
                python_callable=_run,
                op_kwargs={"script": task_spec.callable_path, "args": task_spec.args},
                pool=task_spec.pool,
                retries=task_spec.retries,
                execution_timeout=timedelta(minutes=task_spec.timeout_minutes),
                inlets=inputs if index == 0 else None,
                outlets=outputs if index == len(spec.tasks) - 1 else None,
            )
            if previous is not None:
                previous >> task
            previous = task
    return dag


try:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        _config = yaml.safe_load(handle) or {}
    for _spec in build_all_specs(_config):
        globals()[_spec.dag_id] = _build(_spec)
except Exception as exc:  # Airflow must expose the parse failure without hiding other DAGs.
    logger.exception("FABRIC factory configuration rejected: %s", exc)
    raise
