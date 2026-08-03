from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.orchestration.feature_snapshot import FeatureSnapshotError, resolve_feature_snapshot


CUTOFF = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)


def test_resolver_projects_values_deterministically_after_causal_validation() -> None:
    observations = {
        "spread": {"value": 0.02, "available_at": "2026-01-05T14:59:00Z"},
        "close": {"value": 4200.0, "available_at": "2026-01-05T14:58:00Z"},
    }
    assert resolve_feature_snapshot(observations, decision_cutoff=CUTOFF) == {
        "close": 4200.0,
        "spread": 0.02,
    }


def test_available_at_equal_to_cutoff_is_inclusive() -> None:
    result = resolve_feature_snapshot(
        {"close": {"value": 1, "available_at": CUTOFF}},
        decision_cutoff=CUTOFF,
    )
    assert result == {"close": 1}


def test_observation_after_cutoff_fails_closed() -> None:
    with pytest.raises(FeatureSnapshotError, match="exceeds decision_cutoff"):
        resolve_feature_snapshot(
            {"close": {"value": 1, "available_at": "2026-01-05T15:00:01Z"}},
            decision_cutoff=CUTOFF,
        )


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        ({"value": 1}, "missing available_at"),
        ({"available_at": "2026-01-05T14:00:00Z"}, "missing value"),
        ({"value": 1, "available_at": "not-a-clock"}, "valid ISO-8601"),
        ({"value": 1, "available_at": datetime(2026, 1, 5)}, "timezone-aware"),
    ],
)
def test_incomplete_or_invalid_observation_fails_closed(
    observation: dict[str, object], message: str
) -> None:
    with pytest.raises(FeatureSnapshotError, match=message):
        resolve_feature_snapshot({"close": observation}, decision_cutoff=CUTOFF)


def test_invalid_decision_clock_fails_closed() -> None:
    with pytest.raises(FeatureSnapshotError, match="decision_cutoff must be timezone-aware"):
        resolve_feature_snapshot(
            {"close": {"value": 1, "available_at": CUTOFF}},
            decision_cutoff=datetime(2026, 1, 5),
        )
