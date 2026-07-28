"""Semantic bundle comparison used while old and FABRIC DAGs run in parallel."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.identity.canonical import canonical_json_bytes, semantic_hash as canonical_semantic_hash

DEFAULT_VOLATILE_FIELDS = frozenset(
    {
        "generated_at",
        "published_at",
        "run_id",
        "dag_run_id",
        "task_instance",
        "source_path",
    }
)


def normalize_semantics(
    value: Any,
    *,
    volatile_fields: Iterable[str] = DEFAULT_VOLATILE_FIELDS,
) -> Any:
    ignored = set(volatile_fields)
    if isinstance(value, Mapping):
        return {
            str(key): normalize_semantics(child, volatile_fields=ignored)
            for key, child in value.items()
            if str(key) not in ignored
        }
    if isinstance(value, list):
        return [normalize_semantics(child, volatile_fields=ignored) for child in value]
    return value


def semantic_hash(value: Any, *, volatile_fields: Iterable[str] = DEFAULT_VOLATILE_FIELDS) -> str:
    return canonical_semantic_hash(
        normalize_semantics(value, volatile_fields=volatile_fields)
    )


@dataclass(frozen=True)
class SemanticDiff:
    equal: bool
    left_hash: str
    right_hash: str
    first_difference: str | None


def _first_difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return f"{path}: keys {sorted(left)} != {sorted(right)}"
        for key in left:
            found = _first_difference(left[key], right[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            found = _first_difference(left_item, right_item, f"{path}[{index}]")
            if found:
                return found
        return None
    return (
        None
        if canonical_json_bytes(left) == canonical_json_bytes(right)
        else f"{path}: value differs"
    )


def compare(
    left: Any,
    right: Any,
    *,
    volatile_fields: Iterable[str] = DEFAULT_VOLATILE_FIELDS,
) -> SemanticDiff:
    normalized_left = normalize_semantics(left, volatile_fields=volatile_fields)
    normalized_right = normalize_semantics(right, volatile_fields=volatile_fields)
    left_hash = semantic_hash(normalized_left, volatile_fields=())
    right_hash = semantic_hash(normalized_right, volatile_fields=())
    return SemanticDiff(
        equal=left_hash == right_hash,
        left_hash=left_hash,
        right_hash=right_hash,
        first_difference=_first_difference(normalized_left, normalized_right),
    )


def compare_json_files(left_path: Path, right_path: Path) -> SemanticDiff:
    with left_path.open(encoding="utf-8") as handle:
        left = json.load(handle)
    with right_path.open(encoding="utf-8") as handle:
        right = json.load(handle)
    return compare(left, right)
