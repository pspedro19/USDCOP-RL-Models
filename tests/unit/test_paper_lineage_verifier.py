from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from src.identity.canonical import semantic_hash
from src.lineage.paper_path import PaperPathStatus, verify_paper_path


IDS = (
    "10000000-0000-0000-0000-000000000001",
    "20000000-0000-0000-0000-000000000002",
    "30000000-0000-0000-0000-000000000003",
)


TIMESTAMP = "2026-01-05T09:00:00-05:00"
SIDE = "SHORT"
SIGNAL_HASH = semantic_hash(
    {"strategy_id": "smart_simple_v11", "timestamp": TIMESTAMP, "side": SIDE}
)


def _node(node_id: str, node_type: str, fill: str, *, digest: str | None = None) -> tuple:
    return (
        node_id,
        node_type,
        digest or "sha256:" + fill * 64,
        "1",
        "PASS",
        "VALID",
        None,
        None,
        "REAL_VINTAGE",
        1,
        None,
        None,
    )


NODES = [
    _node(IDS[0], "paper_signal", "1", digest=SIGNAL_HASH),
    _node(IDS[1], "data_snapshot", "2"),
    _node(IDS[2], "bar_l0", "3"),
]
EDGES = [(IDS[0], IDS[1], "DERIVED_FROM"), (IDS[1], IDS[2], "CONSUMED")]


class Cursor:
    def __init__(self, nodes: list[tuple], edges: list[tuple]) -> None:
        self._nodes = nodes
        self._edges = edges
        self._rows: list[tuple] = []

    def execute(self, query: str, params: tuple[str, ...]) -> None:
        self._rows = self._nodes if "FROM lineage.node" in query else self._edges

    def fetchall(self) -> list[tuple]:
        return self._rows

    def close(self) -> None:
        pass


class Connection:
    def __init__(self, nodes: list[tuple] | None = None, edges: list[tuple] | None = None) -> None:
        self.nodes = NODES if nodes is None else nodes
        self.edges = EDGES if edges is None else edges
        self.cursor_calls = 0

    def cursor(self) -> Cursor:
        self.cursor_calls += 1
        return Cursor(self.nodes, self.edges)


class BrokenConnection:
    def cursor(self) -> Cursor:
        raise RuntimeError("database unavailable")


def _ledger(lineage: object = ..., *, trades: list[dict] | None = None) -> dict:
    strategy: dict = {"trades": trades if trades is not None else [
        {"timestamp": TIMESTAMP, "side": SIDE}
    ]}
    if lineage is not ...:
        strategy["lineage"] = lineage
    return {"strategies": {"smart_simple_v11": strategy}}


def _ids() -> dict[str, str]:
    return {
        "timestamp": TIMESTAMP,
        **dict(zip(("signal_node_id", "snapshot_node_id", "bar_l0_node_id"), IDS, strict=True)),
    }


def test_real_ledger_without_lineage_ids_is_absent_not_resolved() -> None:
    connection = Connection()
    result = verify_paper_path(connection, _ledger(), strategy_id="smart_simple_v11")
    assert result.status is PaperPathStatus.ABSENT
    assert result.verified is False
    assert result.node_ids == ()
    assert connection.cursor_calls == 0


def test_partial_declaration_is_broken_not_absent() -> None:
    declaration = _ids()
    declaration.pop("snapshot_node_id")
    result = verify_paper_path(Connection(), _ledger(declaration), strategy_id="smart_simple_v11")
    assert result.status is PaperPathStatus.BROKEN
    assert "snapshot_node_id" in result.detail


def test_lineage_timestamp_matching_zero_trade_rows_is_broken() -> None:
    result = verify_paper_path(
        Connection(), _ledger(_ids(), trades=[]), strategy_id="smart_simple_v11"
    )
    assert result.status is PaperPathStatus.BROKEN
    assert "exactly one trade row; got 0" in result.detail


def test_lineage_timestamp_matching_two_trade_rows_is_broken() -> None:
    duplicate = {"timestamp": TIMESTAMP, "side": SIDE}
    result = verify_paper_path(
        Connection(),
        _ledger(_ids(), trades=[duplicate, dict(duplicate)]),
        strategy_id="smart_simple_v11",
    )
    assert result.status is PaperPathStatus.BROKEN
    assert "exactly one trade row; got 2" in result.detail


def test_signal_node_hash_must_bind_strategy_timestamp_and_side() -> None:
    wrong_signal = _node(IDS[0], "paper_signal", "f")
    result = verify_paper_path(
        Connection(nodes=[wrong_signal, *NODES[1:]]),
        _ledger(_ids()),
        strategy_id="smart_simple_v11",
    )
    assert result.status is PaperPathStatus.BROKEN
    assert "does not identify the declared trade row" in result.detail


def test_unique_persisted_golden_path_resolves() -> None:
    result = verify_paper_path(Connection(), _ledger(_ids()), strategy_id="smart_simple_v11")
    assert result.status is PaperPathStatus.RESOLVED
    assert result.verified is True
    assert result.node_ids == IDS


def test_missing_intermediate_edge_is_broken() -> None:
    result = verify_paper_path(
        Connection(edges=[EDGES[0]]), _ledger(_ids()), strategy_id="smart_simple_v11"
    )
    assert result.status is PaperPathStatus.BROKEN
    assert "no lineage path" in result.detail


def test_ambiguous_direct_edge_is_broken() -> None:
    edges = [*EDGES, (IDS[0], IDS[2], "DERIVED_FROM")]
    result = verify_paper_path(Connection(edges=edges), _ledger(_ids()), strategy_id="smart_simple_v11")
    assert result.status is PaperPathStatus.BROKEN
    assert "ambiguous lineage path" in result.detail


def test_declared_node_missing_from_persistence_is_broken() -> None:
    result = verify_paper_path(
        Connection(nodes=NODES[:2]), _ledger(_ids()), strategy_id="smart_simple_v11"
    )
    assert result.status is PaperPathStatus.BROKEN
    assert IDS[2] in result.detail


def test_wrong_persisted_node_type_is_broken() -> None:
    wrong = [NODES[0], _node(IDS[1], "model_snapshot", "2"), NODES[2]]
    result = verify_paper_path(Connection(nodes=wrong), _ledger(_ids()), strategy_id="smart_simple_v11")
    assert result.status is PaperPathStatus.BROKEN
    assert "wrong lineage node types" in result.detail


def test_persistence_failure_is_reported_as_broken() -> None:
    result = verify_paper_path(BrokenConnection(), _ledger(_ids()), strategy_id="smart_simple_v11")
    assert result.status is PaperPathStatus.BROKEN
    assert result.verified is False
    assert result.detail == "database unavailable"


def test_cli_exit_codes_keep_absent_distinct_from_success() -> None:
    from scripts.diagnostics.verify_paper_lineage import EXIT_CODES

    assert EXIT_CODES[PaperPathStatus.RESOLVED] == 0
    assert EXIT_CODES[PaperPathStatus.BROKEN] != 0
    assert EXIT_CODES[PaperPathStatus.ABSENT] != 0
    assert EXIT_CODES[PaperPathStatus.BROKEN] != EXIT_CODES[PaperPathStatus.ABSENT]


def test_cli_reports_real_absence_with_coverage_zero_and_exit_two(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(_ledger()), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/diagnostics/verify_paper_lineage.py",
            "--ledger",
            str(ledger),
            "--strategy-id",
            "smart_simple_v11",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 2
    assert report["status"] == "ABSENT"
    assert report["coverage"] == 0
    assert report["verified"] is False
