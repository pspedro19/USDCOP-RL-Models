"""Causal feature-snapshot boundary for policy evaluation (BL-45 R3)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any


class FeatureSnapshotError(ValueError):
    """Feature observations cannot prove they existed by the decision cutoff."""


def _aware_datetime(value: Any, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise FeatureSnapshotError(f"{field} must be a valid ISO-8601 timestamp") from exc
    else:
        raise FeatureSnapshotError(f"{field} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise FeatureSnapshotError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def resolve_feature_snapshot(
    observations: Mapping[str, Mapping[str, Any]],
    *,
    decision_cutoff: datetime | str,
) -> dict[str, Any]:
    """Validate point-in-time evidence, then project only feature values.

    Metadata stays at the read boundary. Policy contracts continue receiving the
    existing ``{feature_name: value}`` shape only after every observation proves
    ``available_at <= decision_cutoff``.
    """

    if not isinstance(observations, Mapping) or not observations:
        raise FeatureSnapshotError("feature observations must be a non-empty mapping")
    cutoff = _aware_datetime(decision_cutoff, field="decision_cutoff")
    resolved: dict[str, Any] = {}
    for name in sorted(observations):
        if not isinstance(name, str) or not name.strip():
            raise FeatureSnapshotError("feature names must be non-empty strings")
        observation = observations[name]
        if not isinstance(observation, Mapping):
            raise FeatureSnapshotError(f"feature {name!r} must be an observation mapping")
        if "available_at" not in observation:
            raise FeatureSnapshotError(f"feature {name!r} is missing available_at")
        if "value" not in observation:
            raise FeatureSnapshotError(f"feature {name!r} is missing value")
        available_at = _aware_datetime(
            observation["available_at"], field=f"feature {name!r} available_at"
        )
        if available_at > cutoff:
            raise FeatureSnapshotError(
                f"feature {name!r} available_at exceeds decision_cutoff"
            )
        resolved[name] = observation["value"]
    return resolved
