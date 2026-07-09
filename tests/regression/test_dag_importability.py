"""Regression: every Airflow DAG must import with ZERO errors.

Closes the CI gap surfaced by the E2E audit — nothing previously asserted that the DAG files
actually *load* (only `test_dag_registry_deprecated.py` checks the ID registry, not loadability).
This is the pytest equivalent of `airflow dags list-import-errors` returning empty.

It is **skipped** when Airflow is not importable (e.g. a plain dev checkout without the Airflow
stack) so local `pytest` stays green; it is ACTIVE inside the Airflow container and in any CI job
that installs `airflow/requirements.txt`, where it becomes a hard gate.

Run in-container:
    docker exec usdcop-airflow-scheduler python -m pytest \
        /opt/airflow/project/tests/regression/test_dag_importability.py -q
(or rely on `airflow dags list-import-errors` which this mirrors).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

# Guard on the concrete submodule we use — some envs ship a partial `airflow` namespace
# (top-level importable) without `airflow.models`, which would slip past a bare importorskip.
pytest.importorskip("airflow.models.dagbag", reason="Airflow (full) not installed in this environment")

# Prefer the container's canonical DAG folder; fall back to the repo path for CI checkouts.
_CANDIDATES = [
    Path("/opt/airflow/dags"),
    Path(__file__).resolve().parents[2] / "airflow" / "dags",
]
DAG_FOLDER = next((p for p in _CANDIDATES if p.is_dir()), None)


@pytest.mark.skipif(DAG_FOLDER is None, reason="No airflow/dags folder found")
def test_all_dags_import_without_errors():
    # DAG files self-insert '/opt/airflow' onto sys.path; mirror that so a container run resolves
    # `contracts.*` / `src.*` exactly as the scheduler does.
    for root in ("/opt/airflow", "/opt/airflow/dags", str(DAG_FOLDER)):
        if os.path.isdir(root) and root not in os.sys.path:
            os.sys.path.insert(0, root)

    from airflow.models.dagbag import DagBag

    dagbag = DagBag(dag_folder=str(DAG_FOLDER), include_examples=False)

    assert not dagbag.import_errors, (
        "DAG import errors detected:\n"
        + "\n".join(f"  {f}: {err.splitlines()[-1]}" for f, err in dagbag.import_errors.items())
    )
    # sanity: the registry declares ~41 dag_ids; we should have loaded a comparable number
    assert len(dagbag.dags) >= 30, f"only {len(dagbag.dags)} DAGs loaded (expected ~41)"
