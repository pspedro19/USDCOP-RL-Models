from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.forecasting.dataset_loader import (
    DatasetProvenance,
    SourceProvenance,
    _frame_semantic_hash,
)
from src.lineage.paper_writer import persist_paper_lineage


IDS = {
    "paper_signal": "10000000-0000-0000-0000-000000000001",
    "data_snapshot": "20000000-0000-0000-0000-000000000002",
    "bar_l0": "30000000-0000-0000-0000-000000000003",
}


class Cursor:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple]] = []
        self._row = None

    def execute(self, query: str, params: tuple) -> None:
        self.executions.append((query, params))
        if "INSERT INTO lineage.node" in query:
            self._row = (IDS[params[0]],)
        else:
            self._row = None

    def fetchone(self):
        return self._row


def _dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "open": [4000.0, 4010.0],
        "high": [4020.0, 4030.0],
        "low": [3990.0, 4000.0],
        "close": [4015.0, 4025.0],
        "feature": [0.1, 0.2],
    })


def _provenance(frame: pd.DataFrame) -> DatasetProvenance:
    columns = ("date", "open", "high", "low", "close", "feature")
    source = SourceProvenance("parquet", "file:///seed.parquet", "sha256:" + "a" * 64, 2)
    return DatasetProvenance(
        _frame_semantic_hash(frame, list(columns)),
        columns,
        2,
        "2026-01-05T00:00:00",
        "2026-01-06T00:00:00",
        source,
        source,
    )


def test_writer_persists_three_content_nodes_two_edges_and_strategy_links() -> None:
    frame = _dataset()
    cursor = Cursor()
    declaration = persist_paper_lineage(
        cursor,
        strategy_id="smart_simple_v11",
        trade={"timestamp": "2026-01-05T09:00:00-05:00", "side": "SHORT"},
        dataset=frame,
        provenance=_provenance(frame),
        run_id="paper-ledger:2026-01-05",
        verified_at=datetime(2026, 1, 7, tzinfo=timezone.utc),
    )

    assert declaration.timestamp == "2026-01-05T09:00:00-05:00"
    assert declaration.signal_node_id == IDS["paper_signal"]
    node_calls = [call for call in cursor.executions if "INSERT INTO lineage.node" in call[0]]
    assert [call[1][0] for call in node_calls] == ["paper_signal", "data_snapshot", "bar_l0"]
    assert all(call[1][1] == call[1][2] for call in node_calls), "canonical bytes/hash diverged"
    edge_calls = [call for call in cursor.executions if "INSERT INTO lineage.edge" in call[0]]
    assert [call[1][2] for call in edge_calls] == ["DERIVED_FROM", "CONSUMED"]
    strategy_calls = [call for call in cursor.executions if "INSERT INTO lineage.strategy_node" in call[0]]
    assert [call[1][2] for call in strategy_calls] == ["SIGNAL", "INPUT", "INPUT"]


def test_writer_rejects_dataset_that_no_longer_matches_snapshot() -> None:
    frame = _dataset()
    provenance = _provenance(frame)
    frame.loc[0, "close"] += 1
    try:
        persist_paper_lineage(
            Cursor(),
            strategy_id="smart_simple_v11",
            trade={"timestamp": "2026-01-05T09:00:00-05:00", "side": "SHORT"},
            dataset=frame,
            provenance=provenance,
            run_id="run",
            verified_at=datetime.now(timezone.utc),
        )
    except ValueError as exc:
        assert "does not match" in str(exc)
    else:
        raise AssertionError("mutated dataset was accepted")


def test_writer_requires_exactly_one_entry_bar() -> None:
    frame = _dataset().iloc[1:].copy()
    provenance = _provenance(frame)
    try:
        persist_paper_lineage(
            Cursor(),
            strategy_id="smart_simple_v11",
            trade={"timestamp": "2026-01-05T09:00:00-05:00", "side": "SHORT"},
            dataset=frame,
            provenance=provenance,
            run_id="run",
            verified_at=datetime.now(timezone.utc),
        )
    except ValueError as exc:
        assert "exactly one L0 bar; got 0" in str(exc)
    else:
        raise AssertionError("missing entry bar was accepted")
