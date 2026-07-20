"""A DAG whose critical path failed must not report success.

Contract: CTR-DQ-OPS-001

Airflow derives a DAG-run's state from its LEAF tasks. A reporting task with
``trigger_rule='all_done'`` is meant to run even when upstream failed — but when it is also
the leaf, its success becomes the run's success.

Observed on 2026-07-20: `core_l0_03_macro_backfill` finished **success** while `health_check`
had failed and every task on the critical path was `upstream_failed`. Only `send_report` ran.
The Airflow UI showed green; macro stayed 13 days stale; the H5 training gate kept blocking
with no visible cause. That is the worst failure mode an orchestrator has — silent.

`utils/run_status.fail_if_upstream_failed` re-raises after the report is emitted, so the
notification still goes out and the run goes red.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DAGS = ROOT / "airflow" / "dags"

ALL_DONE = re.compile(r"trigger_rule\s*=\s*['\"]all_done['\"]")


def _dag_files() -> list[Path]:
    return sorted(p for p in DAGS.glob("*.py") if p.name != "__init__.py")


def test_helper_exists():
    assert (DAGS / "utils" / "run_status.py").is_file(), (
        "utils/run_status.py is the shared guard for all_done leaf tasks"
    )


@pytest.mark.parametrize(
    "path",
    [p for p in _dag_files() if ALL_DONE.search(p.read_text(encoding="utf-8", errors="replace"))],
    ids=lambda p: p.name,
)
def test_all_done_tasks_do_not_mask_failure(path: Path):
    """Any DAG using an `all_done` task must also guard its run state.

    This does not forbid `all_done` — reporting on failure is exactly what it is for.
    It requires that such a DAG imports the guard, so the report cannot silently become
    the run's verdict.
    """
    src = path.read_text(encoding="utf-8", errors="replace")
    assert "fail_if_upstream_failed" in src, (
        f"{path.name} has a task with trigger_rule='all_done' but never calls "
        "fail_if_upstream_failed(context). If that task is a leaf, a fully failed run "
        "will report success — which is how a 13-day-stale macro table went unnoticed."
    )


def test_guard_detects_failed_upstream():
    """The helper must actually raise; a no-op guard is worse than none."""
    import importlib.util

    # Load by path: importing `utils.run_status` as a package pulls in utils/__init__,
    # which imports dag_common and demands POSTGRES_PASSWORD. The guard itself has no
    # such dependency and must stay testable without a live environment.
    spec = importlib.util.spec_from_file_location(
        "run_status", DAGS / "utils" / "run_status.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fail_if_upstream_failed, upstream_failures = mod.fail_if_upstream_failed, mod.upstream_failures

    class TI:
        def __init__(self, task_id, state):
            self.task_id = task_id
            self.state = state

    class DagRun:
        def __init__(self, tis):
            self._tis = tis

        def get_task_instances(self):
            return self._tis

    me = TI("send_report", "running")

    healthy = {"dag_run": DagRun([TI("a", "success"), me]), "task_instance": me}
    assert upstream_failures(healthy) == []
    fail_if_upstream_failed(healthy)  # must not raise

    broken = {
        "dag_run": DagRun([TI("health_check", "failed"), TI("extract", "upstream_failed"), me]),
        "task_instance": me,
    }
    assert upstream_failures(broken) == ["extract", "health_check"]
    with pytest.raises(RuntimeError, match="report emitted"):
        fail_if_upstream_failed(broken)
