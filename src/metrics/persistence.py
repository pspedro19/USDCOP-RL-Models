"""Append-only PostgreSQL persistence for governed metric events (BL-18)."""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Protocol

from src.identity.canonical import CanonicalizationError, canonical_json_bytes
from src.metrics.errors import MetricContractError

if TYPE_CHECKING:
    from src.metrics.engine import MetricEvent


class MetricEventConnection(Protocol):
    async def fetchrow(self, query: str, *args: object) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True, slots=True)
class PersistMetricEventResult:
    metric_event_id: str
    inserted: bool


_INSERT_OR_LOAD = """
WITH attempted AS (
    INSERT INTO control.metric_event (
        metric_event_id, event_time, catalog_version, formula_version,
        entity_type, entity_id, strategy_id, asset_id, run_id, environment,
        metric_namespace, metric_name, metric_value, metric_unit, status,
        threshold_warning, threshold_critical, dimensions, lineage
    ) VALUES (
        $1::uuid, $2::timestamptz, $3, $4,
        $5, $6, $7, $8, $9, $10,
        $11, $12, $13, $14, $15,
        $16, $17, $18::jsonb, $19::jsonb
    )
    ON CONFLICT (metric_event_id) DO NOTHING
    RETURNING metric_event_id, event_time, catalog_version, formula_version,
        entity_type, entity_id, strategy_id, asset_id, run_id, environment,
        metric_namespace, metric_name, metric_value, metric_unit, status,
        threshold_warning, threshold_critical, dimensions, lineage, TRUE AS inserted
), stored AS (
    SELECT metric_event_id, event_time, catalog_version, formula_version,
        entity_type, entity_id, strategy_id, asset_id, run_id, environment,
        metric_namespace, metric_name, metric_value, metric_unit, status,
        threshold_warning, threshold_critical, dimensions, lineage, FALSE AS inserted
    FROM control.metric_event
    WHERE metric_event_id = $1::uuid
)
SELECT * FROM attempted
UNION ALL
SELECT * FROM stored
LIMIT 1
"""


def _finite_optional(name: str, value: float | None) -> None:
    if value is not None and not math.isfinite(value):
        raise MetricContractError(f"{name} must be finite or null")


def _json_payload(name: str, value: Mapping[str, Any]) -> str:
    try:
        canonical_json_bytes(value)
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (CanonicalizationError, TypeError, ValueError) as exc:
        raise MetricContractError(f"{name} is not finite canonical JSON: {exc}") from exc


def _event_values(event: MetricEvent) -> tuple[object, ...]:
    try:
        uuid.UUID(event.metric_event_id)
    except (AttributeError, TypeError, ValueError) as exc:
        raise MetricContractError("metric_event_id must be a UUID") from exc
    _finite_optional("metric_value", event.metric_value)
    _finite_optional("threshold_warning", event.threshold_warning)
    _finite_optional("threshold_critical", event.threshold_critical)
    return (
        event.metric_event_id,
        event.event_time,
        event.catalog_version,
        event.formula_version,
        event.entity_type,
        event.entity_id,
        event.strategy_id,
        event.asset_id,
        event.run_id,
        event.environment,
        event.metric_namespace,
        event.metric_name,
        event.metric_value,
        event.metric_unit,
        event.status,
        event.threshold_warning,
        event.threshold_critical,
        _json_payload("dimensions", event.dimensions),
        _json_payload("lineage", event.lineage),
    )


def _comparable_record(event: MetricEvent) -> dict[str, Any]:
    record = event.to_record()
    record["event_time"] = event.event_time.replace("Z", "+00:00")
    return record


def _normalize_stored(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result.pop("inserted", None)
    result["metric_event_id"] = str(result["metric_event_id"])
    event_time = result["event_time"]
    result["event_time"] = (
        event_time.isoformat() if hasattr(event_time, "isoformat") else str(event_time)
    )
    result["dimensions"] = _stored_json("dimensions", result["dimensions"])
    result["lineage"] = _stored_json("lineage", result["lineage"])
    return result


def _stored_json(name: str, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MetricContractError(f"stored {name} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise MetricContractError(f"stored {name} must be a JSON object")
    return dict(value)


async def persist_metric_event(
    connection: MetricEventConnection, event: MetricEvent
) -> PersistMetricEventResult:
    """Insert once; an existing UUID is accepted only for byte-equivalent semantics."""

    row = await connection.fetchrow(_INSERT_OR_LOAD, *_event_values(event))
    if row is None:
        raise MetricContractError("metric_event insert returned no durable row")
    if _normalize_stored(row) != _comparable_record(event):
        raise MetricContractError(
            f"metric_event_id collision with different payload: {event.metric_event_id}"
        )
    return PersistMetricEventResult(
        metric_event_id=event.metric_event_id, inserted=bool(row["inserted"])
    )
