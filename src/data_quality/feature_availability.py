"""Cutoff-aware measurement and immutable persistence of feature availability."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.data_quality.rules import QualityRuleSet

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
FEATURE_AVAILABILITY_REGISTRY = (
    Path(__file__).resolve().parents[2] / "config/quality/feature_availability.yaml"
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    feature_id: str
    table: str
    column: str
    time_column: str
    require_variation: bool = True
    source_enabled: bool = True
    instrument_id: str | None = None


@dataclass(frozen=True, slots=True)
class FeatureMeasurement:
    feature_id: str
    instrument_id: str | None
    status: str
    reason_code: str
    observed_at: datetime
    details: dict[str, Any]


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SQL identifier")
    return value


def _load_registry(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if raw.get("version") != "1.1.0" or not isinstance(raw.get("features"), list):
        raise ValueError("feature availability registry requires version 1.1.0 and features")
    max_age_hours = raw.get("max_age_hours")
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, int) or max_age_hours <= 0:
        raise ValueError("feature availability registry requires positive max_age_hours")
    max_publish_lag_minutes = raw.get("max_publish_lag_minutes")
    if (
        isinstance(max_publish_lag_minutes, bool)
        or not isinstance(max_publish_lag_minutes, int)
        or max_publish_lag_minutes <= 0
    ):
        raise ValueError(
            "feature availability registry requires positive max_publish_lag_minutes"
        )
    return raw


def load_feature_max_age(path: str | Path) -> timedelta:
    return timedelta(hours=_load_registry(path)["max_age_hours"])


def load_feature_publish_lag(path: str | Path) -> timedelta:
    return timedelta(minutes=_load_registry(path)["max_publish_lag_minutes"])


def load_feature_specs(path: str | Path) -> tuple[FeatureSpec, ...]:
    raw = _load_registry(path)
    specs: list[FeatureSpec] = []
    seen: set[tuple[str, str | None]] = set()
    for item in raw["features"]:
        if not isinstance(item, dict):
            raise ValueError("feature entries must be mappings")
        unknown = set(item) - {
            "feature_id", "table", "column", "time_column", "require_variation",
            "source_enabled", "instrument_id",
        }
        if unknown:
            raise ValueError(f"unknown feature fields: {sorted(unknown)}")
        feature_id = item.get("feature_id")
        if not isinstance(feature_id, str) or not feature_id.strip():
            raise ValueError("feature_id is required")
        spec = FeatureSpec(
            feature_id=feature_id.strip(),
            table=_identifier(item.get("table"), "table"),
            column=_identifier(item.get("column"), "column"),
            time_column=_identifier(item.get("time_column"), "time_column"),
            require_variation=item.get("require_variation", True) is True,
            source_enabled=item.get("source_enabled", True) is True,
            instrument_id=item.get("instrument_id"),
        )
        key = (spec.feature_id, spec.instrument_id)
        if key in seen:
            raise ValueError(f"duplicate feature scope: {key}")
        seen.add(key)
        specs.append(spec)
    return tuple(specs)


def _utc_cutoff(cutoff: datetime) -> datetime:
    if not isinstance(cutoff, datetime) or cutoff.tzinfo is None:
        raise ValueError("cutoff must be timezone-aware")
    return cutoff.astimezone(UTC)


def news_feature_cutoff(end: date) -> datetime:
    """Cutoff of the final daily news run consumed by the 19:00 UTC analysis."""
    if not isinstance(end, date):
        raise ValueError("news window end must be a date")
    return datetime.combine(end, time(hour=18), tzinfo=UTC)


def measure_feature(cursor, spec: FeatureSpec, cutoff: datetime) -> FeatureMeasurement:
    observed_at = _utc_cutoff(cutoff)
    table = _identifier(spec.table, "table")
    column = _identifier(spec.column, "column")
    time_column = _identifier(spec.time_column, "time_column")
    cursor.execute(
        """SELECT column_name FROM information_schema.columns
           WHERE table_schema = 'public' AND table_name = %s
             AND column_name IN (%s, %s)""",
        (table, column, time_column),
    )
    present = {row[0] for row in cursor.fetchall()}
    required = {column, time_column}
    if present != required:
        total = non_null = distinct = 0
        measured = False
        missing = sorted(required - present)
    else:
        cursor.execute(
            f'SELECT COUNT(*), COUNT("{column}"), '
            f'COUNT(DISTINCT "{column}") FROM "{table}" '
            f'WHERE "{time_column}" <= %s',
            (observed_at,),
        )
        total, non_null, distinct = (int(value) for value in cursor.fetchone())
        measured = non_null > 0
        missing = []
    decision = QualityRuleSet.feature_status(
        feature_id=spec.feature_id,
        measured=measured,
        all_values_identical=spec.require_variation and measured and distinct <= 1,
        source_enabled=spec.source_enabled,
    )
    details = {
        "cutoff": observed_at.isoformat(),
        "table": table,
        "column": column,
        "time_column": time_column,
        "total_rows": total,
        "non_null_rows": non_null,
        "distinct_values": distinct,
        "require_variation": spec.require_variation,
    }
    if missing:
        details["missing_columns"] = missing
    return FeatureMeasurement(
        feature_id=spec.feature_id,
        instrument_id=spec.instrument_id,
        status=decision.status,
        reason_code=decision.rule_id or "feature.available",
        observed_at=observed_at,
        details=details,
    )


def persist_measurement(cursor, measurement: FeatureMeasurement) -> None:
    payload = json.dumps(measurement.details, sort_keys=True, separators=(",", ":"))
    cursor.execute(
        """INSERT INTO quality.feature_status
           (feature_id, instrument_id, status, reason_code, observed_at, details)
           VALUES (%s, %s, %s, %s, %s, %s::jsonb)
           ON CONFLICT DO NOTHING""",
        (
            measurement.feature_id, measurement.instrument_id, measurement.status,
            measurement.reason_code, measurement.observed_at, payload,
        ),
    )
    cursor.execute(
        """SELECT status, reason_code, details
           FROM quality.feature_status
           WHERE feature_id = %s AND instrument_id IS NOT DISTINCT FROM %s
             AND observed_at = %s""",
        (measurement.feature_id, measurement.instrument_id, measurement.observed_at),
    )
    row = cursor.fetchone()
    if row is None:
        raise RuntimeError("feature status insert was not observable")
    stored_details = row[2] if isinstance(row[2], dict) else json.loads(row[2])
    if (row[0], row[1], stored_details) != (
        measurement.status, measurement.reason_code, measurement.details
    ):
        raise RuntimeError("immutable feature status collision")


def measure_and_persist(
    connection, specs: Iterable[FeatureSpec], cutoff: datetime
) -> tuple[FeatureMeasurement, ...]:
    with connection.cursor() as cursor:
        measurements = tuple(measure_feature(cursor, spec, cutoff) for spec in specs)
        for measurement in measurements:
            persist_measurement(cursor, measurement)
    return measurements
