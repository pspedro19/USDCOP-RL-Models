from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.metrics.errors import MetricContractError
from src.metrics.persistence import persist_metric_event


class _Event:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


def _event(**changes: object) -> _Event:
    values = {
        "metric_event_id": "dd49d8dd-9604-5f3b-ac7a-4f30adce8574",
        "event_time": "2026-08-03T16:00:00Z",
        "catalog_version": "1.0.0",
        "formula_version": "1.0.0",
        "entity_type": "strategy",
        "entity_id": "s1",
        "strategy_id": "s1",
        "asset_id": "usdcop",
        "run_id": "run-1",
        "environment": "paper",
        "metric_namespace": "strategy",
        "metric_name": "sharpe",
        "metric_value": 0.25,
        "metric_unit": "ratio",
        "status": "OK",
        "threshold_warning": 0.0,
        "threshold_critical": -1.0,
        "dimensions": {"window": "26w", "n_observations": 26},
        "lineage": {"source": "paper-ledger"},
    }
    values.update(changes)
    return _Event(**values)


class _Connection:
    def __init__(
        self,
        *,
        inserted: bool = True,
        mutation: dict[str, object] | None = None,
        json_as_text: bool = False,
    ):
        self.inserted = inserted
        self.mutation = mutation or {}
        self.json_as_text = json_as_text
        self.query = ""
        self.args: tuple[object, ...] = ()

    async def fetchrow(self, query: str, *args: object):
        self.query = query
        self.args = args
        event = _event().to_record()
        event["metric_event_id"] = event["metric_event_id"]
        event["event_time"] = datetime(2026, 8, 3, 16, tzinfo=timezone.utc)
        if self.json_as_text:
            event["dimensions"] = '{"n_observations":26,"window":"26w"}'
            event["lineage"] = '{"source":"paper-ledger"}'
        event.update(self.mutation)
        return event | {"inserted": self.inserted}


@pytest.mark.asyncio
async def test_metric_event_sink_is_parameterized_and_idempotent() -> None:
    connection = _Connection(inserted=False)
    result = await persist_metric_event(connection, _event())

    assert result.inserted is False
    assert result.metric_event_id == _event().metric_event_id
    assert "ON CONFLICT (metric_event_id) DO NOTHING" in connection.query
    assert "$18::jsonb" in connection.query and "$19::jsonb" in connection.query
    assert _event().metric_event_id not in connection.query
    assert connection.args[0] == _event().metric_event_id


@pytest.mark.asyncio
async def test_metric_event_sink_rejects_uuid_collision_with_different_payload() -> None:
    with pytest.raises(MetricContractError, match="collision"):
        await persist_metric_event(
            _Connection(inserted=False, mutation={"metric_value": 99.0}), _event()
        )


@pytest.mark.asyncio
async def test_metric_event_sink_rejects_non_finite_json_before_sql() -> None:
    connection = _Connection()
    with pytest.raises(MetricContractError, match="finite canonical JSON"):
        await persist_metric_event(connection, _event(dimensions={"bad": float("nan")}))
    assert connection.query == ""


@pytest.mark.asyncio
async def test_metric_event_sink_accepts_asyncpg_default_jsonb_text() -> None:
    result = await persist_metric_event(
        _Connection(inserted=False, json_as_text=True), _event()
    )
    assert result.inserted is False


@pytest.mark.asyncio
async def test_metric_event_sink_rejects_invalid_stored_jsonb() -> None:
    with pytest.raises(MetricContractError, match="stored dimensions is invalid JSON"):
        await persist_metric_event(
            _Connection(mutation={"dimensions": "not-json"}), _event()
        )
