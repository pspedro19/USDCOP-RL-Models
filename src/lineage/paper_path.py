"""Fail-closed verification of a paper signal's persisted golden lineage path."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from src.lineage.graph import (
    EdgeType,
    LineageEdge,
    LineageNode,
    LineagePathError,
    resolve_unique_path,
)


class PaperPathStatus(StrEnum):
    """Three non-overlapping outcomes; ABSENT is deliberately not success."""

    RESOLVED = "RESOLVED"
    BROKEN = "BROKEN"
    ABSENT = "ABSENT"


@dataclass(frozen=True, slots=True)
class PaperPathResult:
    status: PaperPathStatus
    strategy_id: str
    detail: str
    node_ids: tuple[str, ...] = ()

    @property
    def verified(self) -> bool:
        return self.status is PaperPathStatus.RESOLVED


_ID_FIELDS = ("signal_node_id", "snapshot_node_id", "bar_l0_node_id")
_EXPECTED_TYPES = ("paper_signal", "data_snapshot", "bar_l0")


def _declared_ids(
    ledger: Mapping[str, Any], strategy_id: str
) -> tuple[PaperPathStatus | None, tuple[str, str, str] | None, str]:
    strategies = ledger.get("strategies")
    strategy = strategies.get(strategy_id) if isinstance(strategies, Mapping) else None
    if not isinstance(strategy, Mapping):
        return PaperPathStatus.ABSENT, None, f"strategy {strategy_id!r} is absent from ledger"

    lineage = strategy.get("lineage")
    if lineage is None:
        return (
            PaperPathStatus.ABSENT,
            None,
            f"strategy {strategy_id!r} declares no persisted lineage ids",
        )
    if not isinstance(lineage, Mapping):
        return PaperPathStatus.BROKEN, None, "lineage declaration must be an object"

    missing = [field for field in _ID_FIELDS if not lineage.get(field)]
    if missing:
        return (
            PaperPathStatus.BROKEN,
            None,
            "incomplete lineage declaration; missing " + ", ".join(missing),
        )
    ids = tuple(str(lineage[field]) for field in _ID_FIELDS)
    if len(set(ids)) != len(ids):
        return PaperPathStatus.BROKEN, None, "lineage node ids must be distinct"
    return None, ids, ""


def _load_subgraph(connection: Any, node_ids: tuple[str, str, str]) -> tuple[list[LineageNode], list[LineageEdge]]:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT node_id::text, node_type, semantic_hash, schema_version,
                   quality_status, status, bytes_hash, storage_uri,
                   availability_quality, row_count, min_event_time, max_event_time
            FROM lineage.node
            WHERE node_id IN (%s::uuid, %s::uuid, %s::uuid)
            """,
            node_ids,
        )
        nodes = [LineageNode(*row) for row in cursor.fetchall()]
        cursor.execute(
            """
            SELECT source_node_id::text, target_node_id::text, edge_type
            FROM lineage.edge
            WHERE source_node_id IN (%s::uuid, %s::uuid, %s::uuid)
              AND target_node_id IN (%s::uuid, %s::uuid, %s::uuid)
            """,
            (*node_ids, *node_ids),
        )
        edges = [LineageEdge(source, target, EdgeType(kind)) for source, target, kind in cursor.fetchall()]
        return nodes, edges
    finally:
        cursor.close()


def verify_paper_path(
    connection: Any,
    ledger: Mapping[str, Any],
    *,
    strategy_id: str,
) -> PaperPathResult:
    """Verify the exact signal -> snapshot -> L0 path declared by a real ledger.

    A ledger without identifiers is ``ABSENT`` (coverage zero), not a successful
    verification. Once a declaration exists, every malformed or unprovable state is
    ``BROKEN``.
    """

    preliminary, node_ids, detail = _declared_ids(ledger, strategy_id)
    if preliminary is not None:
        return PaperPathResult(preliminary, strategy_id, detail)
    assert node_ids is not None

    try:
        nodes, edges = _load_subgraph(connection, node_ids)
        nodes_by_id = {node.node_id: node for node in nodes}
        missing = [node_id for node_id in node_ids if node_id not in nodes_by_id]
        if missing:
            raise LineagePathError("declared lineage nodes missing from persistence: " + ", ".join(missing))
        actual_types = tuple(nodes_by_id[node_id].node_type for node_id in node_ids)
        if actual_types != _EXPECTED_TYPES:
            raise LineagePathError(
                f"wrong lineage node types: expected {_EXPECTED_TYPES!r}, got {actual_types!r}"
            )
        path = resolve_unique_path(
            nodes,
            edges,
            source_node_id=node_ids[0],
            target_node_id=node_ids[2],
        )
        resolved_ids = tuple(node.node_id for node in path)
        if resolved_ids != node_ids:
            raise LineagePathError(
                f"golden path must be signal -> snapshot -> bar_l0, got {resolved_ids!r}"
            )
    except Exception as exc:  # noqa: BLE001 - DB adapter boundary must fail closed as BROKEN
        return PaperPathResult(PaperPathStatus.BROKEN, strategy_id, str(exc), node_ids)

    return PaperPathResult(
        PaperPathStatus.RESOLVED,
        strategy_id,
        "unique persisted signal -> snapshot -> bar_l0 path verified",
        node_ids,
    )
