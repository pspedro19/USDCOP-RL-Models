from __future__ import annotations

import os
import uuid

import asyncpg
import pytest

from src.metrics.errors import MetricContractError
from src.metrics.persistence import persist_metric_event


class _Event:
    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def to_record(self) -> dict[str, object]:
        return dict(self.__dict__)


def _event(event_id: str, *, metric_value: float = 0.25) -> _Event:
    return _Event(
        metric_event_id=event_id,
        event_time="2026-08-03T17:00:00Z",
        catalog_version="1.0.0",
        formula_version="1.0.0",
        entity_type="strategy",
        entity_id=f"bl18-integration-{event_id}",
        strategy_id="bl18-integration",
        asset_id="usdcop",
        run_id="postgres-real",
        environment="paper",
        metric_namespace="strategy",
        metric_name="sharpe",
        metric_value=metric_value,
        metric_unit="ratio",
        status="OK",
        threshold_warning=0.0,
        threshold_critical=-1.0,
        dimensions={"window": "26w", "n_observations": 26},
        lineage={"source": "postgres-integration"},
    )


@pytest.mark.asyncio
async def test_metric_event_insert_replay_and_collision_against_postgres() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        pytest.skip("DATABASE_URL not provided; PostgreSQL integration not executed")
    connection = await asyncpg.connect(database_url)
    try:
        event_id = str(uuid.uuid4())
        inserted = await persist_metric_event(connection, _event(event_id))
        replay = await persist_metric_event(connection, _event(event_id))
        assert inserted.inserted is True
        assert replay.inserted is False
        with pytest.raises(MetricContractError, match="collision"):
            await persist_metric_event(
                connection, _event(event_id, metric_value=0.99)
            )
    finally:
        await connection.close()
