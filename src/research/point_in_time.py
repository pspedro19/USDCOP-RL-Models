"""Point-in-time reads whose cutoff is enforced by the read layer.

Screening callers cannot opt out.  The result is checked again after retrieval
so an incorrect repository query fails the job rather than merely logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Awaitable, Callable, Iterable, Mapping, Sequence


class ResearchEnvironment(StrEnum):
    IDEATION = "ideation"
    SCREENING = "screening"
    VALIDATION = "validation"
    REPLAY = "replay"
    PAPER = "paper"
    PRODUCTION = "production"


class PointInTimeViolation(RuntimeError):
    pass


def _utc(value: datetime | str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PointInTimeViolation("cutoff and available_at must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def assert_available_at(
    rows: Iterable[Mapping[str, Any]],
    *,
    cutoff: datetime | str,
    available_at_field: str = "available_at",
) -> list[Mapping[str, Any]]:
    boundary = _utc(cutoff)
    materialized = list(rows)
    violations: list[str] = []
    for index, row in enumerate(materialized):
        if available_at_field not in row or row[available_at_field] is None:
            violations.append(f"row {index}: missing {available_at_field}")
            continue
        if _utc(row[available_at_field]) > boundary:
            violations.append(
                f"row {index}: {_utc(row[available_at_field]).isoformat()} > {boundary.isoformat()}"
            )
    if violations:
        sample = "; ".join(violations[:5])
        raise PointInTimeViolation(f"look-ahead blocked ({len(violations)} rows): {sample}")
    return materialized


def read_point_in_time(
    reader: Callable[..., Iterable[Mapping[str, Any]]],
    *,
    cutoff: datetime | str,
    environment: ResearchEnvironment | str,
    available_at_field: str = "available_at",
    **reader_kwargs: Any,
) -> list[Mapping[str, Any]]:
    """Read data and impose the cutoff for screening at both input and output."""
    env = ResearchEnvironment(environment)
    boundary = _utc(cutoff)
    if env is ResearchEnvironment.SCREENING:
        reader_kwargs["cutoff"] = boundary
        reader_kwargs["available_at_field"] = available_at_field
    rows = list(reader(**reader_kwargs))
    if env is ResearchEnvironment.SCREENING:
        return assert_available_at(
            rows, cutoff=boundary, available_at_field=available_at_field
        )
    return rows


async def read_point_in_time_async(
    reader: Callable[..., Awaitable[Sequence[Mapping[str, Any]]]],
    *,
    cutoff: datetime | str,
    environment: ResearchEnvironment | str,
    available_at_field: str = "available_at",
    **reader_kwargs: Any,
) -> list[Mapping[str, Any]]:
    env = ResearchEnvironment(environment)
    boundary = _utc(cutoff)
    if env is ResearchEnvironment.SCREENING:
        reader_kwargs["cutoff"] = boundary
        reader_kwargs["available_at_field"] = available_at_field
    rows = list(await reader(**reader_kwargs))
    if env is ResearchEnvironment.SCREENING:
        return assert_available_at(rows, cutoff=boundary, available_at_field=available_at_field)
    return rows


@dataclass(frozen=True)
class PointInTimeSQL:
    sql: str
    parameters: Mapping[str, Any]


def bounded_select_sql(
    *,
    table: str,
    cutoff: datetime | str,
    columns: Sequence[str] = ("*",),
    available_at_field: str = "available_at",
    additional_where: str | None = None,
) -> PointInTimeSQL:
    """Construct the mandatory SQL predicate used by screening repositories."""
    identifiers = [*table.split("."), available_at_field, *columns]
    if any(not token.replace("_", "").isalnum() and token != "*" for token in identifiers):
        raise ValueError("unsafe SQL identifier")
    predicate = f"{available_at_field} <= :pit_cutoff"
    if additional_where:
        predicate = f"({additional_where}) AND {predicate}"
    return PointInTimeSQL(
        sql=f"SELECT {', '.join(columns)} FROM {table} WHERE {predicate}",
        parameters={"pit_cutoff": _utc(cutoff)},
    )
