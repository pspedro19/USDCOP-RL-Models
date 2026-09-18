"""Deterministic Airflow-to-COT session date conversion."""
from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

BOGOTA = ZoneInfo("America/Bogota")


def session_date_from_context(context: dict) -> str:
    """Return the session date represented by a DagRun's interval end.

    ``data_interval_end`` is the scheduling contract. ``logical_date`` is only a
    compatibility fallback for old/manual runs and must not win when the interval
    is present. Naive timestamps are interpreted as UTC and are therefore explicit.
    """
    value = context.get("data_interval_end")
    if value is None:
        value = context.get("logical_date") or context.get("execution_date")
    if value is None:
        raise ValueError("Airflow context lacks data_interval_end/logical_date")
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime):
        raise TypeError("Airflow interval must be datetime or ISO string")
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(BOGOTA).date().isoformat()

