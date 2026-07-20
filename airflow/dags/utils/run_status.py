"""Stop `all_done` reporting tasks from painting a failed DAG green.

Contract: CTR-DQ-OPS-001

Airflow decides a DAG-run's state from its LEAF tasks. A reporting task with
``trigger_rule='all_done'`` runs even when everything upstream failed — which is correct,
you want the failure reported — but if it is also the leaf, its success becomes the run's
success.

Observed 2026-07-20: `core_l0_03_macro_backfill` reported **success** while `health_check`
had failed and every task on the critical path was `upstream_failed`. Only `send_report`
ran. An operator reading the Airflow UI would conclude macro was refreshed; it was not, and
the H5 training gate kept blocking on 13-day-old macro data with no visible cause.

Same shape in `l0_macro_update.py` (`daily_summary`) and `reconciliation_daily.py`
(`alert_discrepancies`).

Usage — call at the END of the reporting callable, after the report is emitted:

    from utils.run_status import fail_if_upstream_failed
    def send_report(**context):
        ...emit the report...
        fail_if_upstream_failed(context)   # keeps the report, restores an honest state
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# States that mean "this task did not do its job".
_BAD = {"failed", "upstream_failed"}


def upstream_failures(context) -> list[str]:
    """Task ids in this DAG run that failed or were skipped by a failure."""
    dag_run = context.get("dag_run")
    if dag_run is None:
        return []
    me = getattr(context.get("task_instance"), "task_id", None)
    return sorted(
        ti.task_id for ti in dag_run.get_task_instances()
        if ti.task_id != me and ti.state in _BAD
    )


def fail_if_upstream_failed(context, *, task_name: str | None = None) -> None:
    """Raise if anything else in this run failed, so the DAG state stays honest.

    The report has already been emitted by the time this runs, so nothing is lost: the
    operator still gets the notification AND a red run instead of a green one that hides it.
    """
    failed = upstream_failures(context)
    if not failed:
        return
    label = task_name or getattr(context.get("task_instance"), "task_id", "report")
    msg = (
        f"[{label}] report emitted, but {len(failed)} task(s) failed in this run: "
        f"{', '.join(failed)}. Failing so the DAG run is not marked success — a green run "
        "over a dead critical path is worse than a red one, because nobody investigates it."
    )
    logger.error(msg)
    raise RuntimeError(msg)
