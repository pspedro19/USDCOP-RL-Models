"""Typed lineage objects shared by ingestion and control-plane writers (BL-24)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re
from typing import Iterable

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


class EdgeType(StrEnum):
    CONSUMED = "CONSUMED"
    PRODUCED = "PRODUCED"
    DERIVED_FROM = "DERIVED_FROM"
    CORRECTED_BY = "CORRECTED_BY"
    SUPERSEDES = "SUPERSEDES"


class RevisionType(StrEnum):
    LEGITIMATE_RELEASE = "LEGITIMATE_RELEASE"
    PROVIDER_CORRECTION = "PROVIDER_CORRECTION"
    PIPELINE_ERROR = "PIPELINE_ERROR"
    SCHEMA_REINTERPRETATION = "SCHEMA_REINTERPRETATION"


class LineagePathError(ValueError):
    """Raised when a requested lineage path cannot be proved uniquely."""


@dataclass(frozen=True, slots=True)
class LineageEdge:
    source_node_id: str
    target_node_id: str
    edge_type: EdgeType

    def __post_init__(self) -> None:
        if not self.source_node_id or not self.target_node_id:
            raise ValueError("lineage edge endpoints must be non-empty")


@dataclass(frozen=True, slots=True)
class LineageNode:
    node_id: str
    node_type: str
    semantic_hash: str
    schema_version: str
    quality_status: str
    status: str = "VALID"
    bytes_hash: str | None = None
    storage_uri: str | None = None
    availability_quality: str = "UNKNOWN"
    row_count: int | None = None
    min_event_time: datetime | None = None
    max_event_time: datetime | None = None

    def __post_init__(self) -> None:
        if _SHA256.fullmatch(self.semantic_hash) is None:
            raise ValueError("semantic_hash must be sha256")
        if self.bytes_hash is not None and _SHA256.fullmatch(self.bytes_hash) is None:
            raise ValueError("bytes_hash must be sha256")
        if self.status not in {"VALID", "STALE", "INVALIDATED"}:
            raise ValueError("invalid lineage node status")
        if self.availability_quality not in {"REAL_VINTAGE", "RECONSTRUCTED", "UNKNOWN"}:
            raise ValueError("invalid availability_quality")
        if self.row_count is not None and (
            type(self.row_count) is not int or self.row_count < 0
        ):
            raise ValueError("row_count must be a non-negative integer")
        if (self.min_event_time is None) is not (self.max_event_time is None):
            raise ValueError("min_event_time and max_event_time must be provided together")
        if (
            self.min_event_time is not None
            and self.max_event_time is not None
            and self.max_event_time < self.min_event_time
        ):
            raise ValueError("max_event_time cannot precede min_event_time")


def resolve_unique_path(
    nodes: Iterable[LineageNode],
    edges: Iterable[LineageEdge],
    *,
    source_node_id: str,
    target_node_id: str,
    edge_types: Iterable[EdgeType] = (
        EdgeType.CONSUMED,
        EdgeType.PRODUCED,
        EdgeType.DERIVED_FROM,
    ),
) -> tuple[LineageNode, ...]:
    """Resolve exactly one directed dependency path, otherwise fail closed."""
    by_id: dict[str, LineageNode] = {}
    for node in nodes:
        if node.node_id in by_id:
            raise LineagePathError(f"duplicate lineage node_id: {node.node_id}")
        by_id[node.node_id] = node
    for endpoint in (source_node_id, target_node_id):
        if endpoint not in by_id:
            raise LineagePathError(f"missing lineage node: {endpoint}")

    allowed = frozenset(edge_types)
    adjacency: dict[str, list[str]] = {node_id: [] for node_id in by_id}
    for edge in edges:
        if edge.source_node_id not in by_id or edge.target_node_id not in by_id:
            raise LineagePathError(
                "lineage edge references a missing endpoint: "
                f"{edge.source_node_id}->{edge.target_node_id}"
            )
        if edge.edge_type in allowed:
            adjacency[edge.source_node_id].append(edge.target_node_id)

    paths: list[tuple[str, ...]] = []

    def visit(current: str, path: tuple[str, ...]) -> None:
        if len(paths) > 1:
            return
        if current == target_node_id:
            paths.append(path)
            return
        for next_node in sorted(set(adjacency[current])):
            if next_node not in path:
                visit(next_node, (*path, next_node))

    visit(source_node_id, (source_node_id,))
    if not paths:
        raise LineagePathError(
            f"no lineage path from {source_node_id} to {target_node_id}"
        )
    if len(paths) != 1:
        raise LineagePathError(
            f"ambiguous lineage path from {source_node_id} to {target_node_id}"
        )
    return tuple(by_id[node_id] for node_id in paths[0])
