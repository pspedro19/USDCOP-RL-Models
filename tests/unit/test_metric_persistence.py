from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.metrics.errors import MetricContractError
from src.metrics.persistence import persist_metric_event, persist_metric_event_dbapi


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
    assert "ON CONFLICT DO NOTHING" in connection.query
    assert "dimensions = $18::jsonb" in connection.query
    assert "$18::jsonb" in connection.query and "$19::jsonb" in connection.query
    assert _event().metric_event_id not in connection.query
    assert connection.args[0] == _event().metric_event_id
    assert connection.args[1] == datetime(2026, 8, 3, 16, tzinfo=timezone.utc)


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
@pytest.mark.parametrize("event_time", ["not-a-timestamp", "2026-08-03T16:00:00"])
async def test_metric_event_sink_rejects_invalid_or_naive_event_time(
    event_time: str,
) -> None:
    connection = _Connection()
    with pytest.raises(MetricContractError, match="event_time"):
        await persist_metric_event(connection, _event(event_time=event_time))
    assert connection.query == ""


@pytest.mark.asyncio
async def test_metric_event_sink_accepts_asyncpg_default_jsonb_text() -> None:
    result = await persist_metric_event(
        _Connection(inserted=False, json_as_text=True), _event()
    )
    assert result.inserted is False


@pytest.mark.asyncio
async def test_metric_event_replay_accepts_equivalent_non_utc_offset() -> None:
    result = await persist_metric_event(
        _Connection(inserted=False),
        _event(event_time="2026-08-03T11:00:00-05:00"),
    )
    assert result.inserted is False


@pytest.mark.asyncio
async def test_metric_event_sink_rejects_invalid_stored_jsonb() -> None:
    with pytest.raises(MetricContractError, match="stored dimensions is invalid JSON"):
        await persist_metric_event(
            _Connection(mutation={"dimensions": "not-json"}), _event()
        )


class _Description:
    def __init__(self, name: str) -> None:
        self.name = name


class _DbApiCursor:
    def __init__(self, *, mutation: dict[str, object] | None = None) -> None:
        self.query = ""
        self.args: tuple[object, ...] = ()
        row = _event().to_record() | {"inserted": True}
        row["event_time"] = datetime(2026, 8, 3, 16, tzinfo=timezone.utc)
        row.update(mutation or {})
        self._row = row
        self.description = [_Description(name) for name in row]

    def execute(self, query: str, args: tuple[object, ...]) -> None:
        self.query = query
        self.args = args

    def fetchone(self) -> tuple[object, ...]:
        return tuple(self._row.values())


def test_dbapi_sink_is_transaction_neutral_and_parameterized() -> None:
    cursor = _DbApiCursor()
    result = persist_metric_event_dbapi(cursor, _event())

    assert result.inserted is True
    assert "ON CONFLICT DO NOTHING" in cursor.query
    assert "%s::jsonb" in cursor.query
    assert len(cursor.args) == 30


def test_dbapi_sink_translates_semantic_identity_collision() -> None:
    cursor = _DbApiCursor(
        mutation={"metric_event_id": "2f24cb93-4d0a-56c0-a273-0cf15fa7a366"}
    )
    with pytest.raises(MetricContractError, match="semantic identity"):
        persist_metric_event_dbapi(cursor, _event())
