from __future__ import annotations

import pytest

import src.lineage as lineage
from src.lineage import (
    EdgeType,
    LineageEdge,
    LineageNode,
    LineagePathError,
    resolve_unique_path,
)


def test_package_exports_complete_public_lineage_contract() -> None:
    assert set(lineage.__all__) == {
        "EdgeType",
        "LineageEdge",
        "LineageNode",
        "LineagePathError",
        "RevisionType",
        "resolve_unique_path",
    }


def _node(node_id: str) -> LineageNode:
    return LineageNode(
        node_id=node_id,
        node_type="artifact",
        semantic_hash="sha256:" + node_id[0] * 64,
        schema_version="1",
        quality_status="PASS",
    )


def test_resolves_complete_golden_path_in_order() -> None:
    nodes = [_node("a-signal"), _node("b-snapshot"), _node("c-bar")]
    edges = [
        LineageEdge("a-signal", "b-snapshot", EdgeType.DERIVED_FROM),
        LineageEdge("b-snapshot", "c-bar", EdgeType.CONSUMED),
    ]
    path = resolve_unique_path(
        nodes, edges, source_node_id="a-signal", target_node_id="c-bar"
    )
    assert [node.node_id for node in path] == ["a-signal", "b-snapshot", "c-bar"]


def test_missing_intermediate_edge_fails_closed() -> None:
    nodes = [_node("a-signal"), _node("b-snapshot"), _node("c-bar")]
    edges = [LineageEdge("a-signal", "b-snapshot", EdgeType.DERIVED_FROM)]
    with pytest.raises(LineagePathError, match="no lineage path"):
        resolve_unique_path(
            nodes, edges, source_node_id="a-signal", target_node_id="c-bar"
        )


def test_cycle_does_not_create_a_false_second_path() -> None:
    nodes = [_node("a-signal"), _node("b-snapshot"), _node("c-bar")]
    edges = [
        LineageEdge("a-signal", "b-snapshot", EdgeType.DERIVED_FROM),
        LineageEdge("b-snapshot", "a-signal", EdgeType.DERIVED_FROM),
        LineageEdge("b-snapshot", "c-bar", EdgeType.CONSUMED),
    ]
    path = resolve_unique_path(
        nodes, edges, source_node_id="a-signal", target_node_id="c-bar"
    )
    assert [node.node_id for node in path] == ["a-signal", "b-snapshot", "c-bar"]


def test_multiple_routes_are_rejected_as_ambiguous() -> None:
    nodes = [_node("a-source"), _node("b-left"), _node("c-right"), _node("d-target")]
    edges = [
        LineageEdge("a-source", "b-left", EdgeType.DERIVED_FROM),
        LineageEdge("b-left", "d-target", EdgeType.CONSUMED),
        LineageEdge("a-source", "c-right", EdgeType.DERIVED_FROM),
        LineageEdge("c-right", "d-target", EdgeType.CONSUMED),
    ]
    with pytest.raises(LineagePathError, match="ambiguous lineage path"):
        resolve_unique_path(
            nodes, edges, source_node_id="a-source", target_node_id="d-target"
        )


def test_edge_with_unknown_endpoint_is_rejected_even_if_not_on_route() -> None:
    nodes = [_node("a-source"), _node("b-target")]
    edges = [
        LineageEdge("a-source", "b-target", EdgeType.DERIVED_FROM),
        LineageEdge("missing", "b-target", EdgeType.CORRECTED_BY),
    ]
    with pytest.raises(LineagePathError, match="missing endpoint"):
        resolve_unique_path(
            nodes, edges, source_node_id="a-source", target_node_id="b-target"
        )
