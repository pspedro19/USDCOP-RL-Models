"""Typed lineage objects shared by ingestion and control-plane writers (BL-24)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
import re

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
