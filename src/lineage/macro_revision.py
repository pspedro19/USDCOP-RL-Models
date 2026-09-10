"""Transactional lineage emission for macro observation upserts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Sequence

from src.identity.canonical import semantic_hash
from src.lineage.graph import RevisionType


_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class MacroRevisionResult:
    nodes_recorded: int
    revisions_recorded: int


def _safe_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe SQL identifier: {value!r}")
    return value


def _observation_payload(
    *, schema: str, table: str, observed_on: Any, variable: str, value: Any
) -> dict[str, Any]:
    if hasattr(observed_on, "to_pydatetime"):
        observed_on = observed_on.to_pydatetime()
    if isinstance(observed_on, datetime) and observed_on.tzinfo is None:
        observed_on = observed_on.date()
    if type(value).__module__.split(".", 1)[0] == "numpy":
        value = value.item()
    return {
        "schema": schema,
        "table": table,
        "observed_on": observed_on,
        "variable": variable,
        "value": value,
    }


class MacroRevisionWriter:
    """Record macro nodes and typed revisions using the caller's transaction."""

    def __init__(self, *, schema: str = "public") -> None:
        self.schema = _safe_identifier(schema)

    def _upsert_node(
        self,
        cursor: Any,
        *,
        table: str,
        observed_on: Any,
        variable: str,
        value: Any,
        verified_at: datetime,
    ) -> tuple[Any, str]:
        payload = _observation_payload(
            schema=self.schema,
            table=table,
            observed_on=observed_on,
            variable=variable,
            value=value,
        )
        digest = semantic_hash(payload)
        quality_status = "MISSING" if value is None else "PASS"
        storage_uri = f"db://{self.schema}/{table}/{observed_on}#{variable}"
        cursor.execute(
            """
            INSERT INTO lineage.node (
                node_type, semantic_hash, schema_version, min_event_time,
                max_event_time, quality_status, status, storage_uri,
                availability_quality, last_verified_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, 'VALID', %s, 'UNKNOWN', %s)
            ON CONFLICT (node_type, semantic_hash) DO UPDATE
            SET last_verified_at = GREATEST(
                lineage.node.last_verified_at,
                EXCLUDED.last_verified_at
            )
            RETURNING node_id
            """,
            (
                "macro_observation",
                digest,
                "1",
                observed_on,
                observed_on,
                quality_status,
                storage_uri,
                verified_at,
            ),
        )
        row = cursor.fetchone()
        if not row:
            raise RuntimeError("lineage.node upsert returned no node_id")
        return row[0], digest

    def record_before_upsert(
        self,
        cursor: Any,
        *,
        table: str,
        date_column: str,
        rows: Iterable[Sequence[Any]],
        columns: Sequence[str],
        revision_type: RevisionType,
        actor: str,
        run_id: str,
        event_time: datetime,
    ) -> MacroRevisionResult:
        """Lock prior values, then emit nodes/revisions without committing.

        The caller must execute the data upsert with this same cursor and commit
        only after both lineage and data writes succeed.
        """

        table = _safe_identifier(table)
        date_column = _safe_identifier(date_column)
        safe_columns = [_safe_identifier(column) for column in columns]
        if not actor.strip() or not run_id.strip():
            raise ValueError("actor and run_id must be non-empty")
        if event_time.tzinfo is None or event_time.utcoffset() is None:
            raise ValueError("event_time must be timezone-aware")
        if not isinstance(revision_type, RevisionType):
            raise ValueError("revision_type must be a RevisionType")

        materialized_rows = [tuple(row) for row in rows]
        expected_width = len(safe_columns) + 1
        if any(len(row) != expected_width for row in materialized_rows):
            raise ValueError(f"each row must contain {expected_width} values")
        if not materialized_rows or not safe_columns:
            return MacroRevisionResult(nodes_recorded=0, revisions_recorded=0)

        observation_dates = list(dict.fromkeys(row[0] for row in materialized_rows))
        selected = ", ".join([date_column, *safe_columns])
        cursor.execute(
            f"""
            SELECT {selected}
            FROM {self.schema}.{table}
            WHERE {date_column} = ANY(%s)
            FOR UPDATE
            """,
            (observation_dates,),
        )
        prior_by_date = {row[0]: tuple(row[1:]) for row in cursor.fetchall()}

        node_hashes: set[str] = set()
        revisions = 0
        for row in materialized_rows:
            observed_on = row[0]
            previous_values = prior_by_date.get(observed_on)
            for offset, variable in enumerate(safe_columns, start=1):
                new_value = row[offset]
                revised_id, revised_hash = self._upsert_node(
                    cursor,
                    table=table,
                    observed_on=observed_on,
                    variable=variable,
                    value=new_value,
                    verified_at=event_time,
                )
                node_hashes.add(revised_hash)
                if previous_values is None:
                    continue

                old_value = previous_values[offset - 1]
                old_payload = _observation_payload(
                    schema=self.schema,
                    table=table,
                    observed_on=observed_on,
                    variable=variable,
                    value=old_value,
                )
                if semantic_hash(old_payload) == revised_hash:
                    continue

                original_id, original_hash = self._upsert_node(
                    cursor,
                    table=table,
                    observed_on=observed_on,
                    variable=variable,
                    value=old_value,
                    verified_at=event_time,
                )
                node_hashes.add(original_hash)
                edge_type = (
                    "SUPERSEDES"
                    if revision_type
                    in {RevisionType.LEGITIMATE_RELEASE, RevisionType.SCHEMA_REINTERPRETATION}
                    else "CORRECTED_BY"
                )
                cursor.execute(
                    """
                    INSERT INTO lineage.edge (
                        source_node_id, target_node_id, edge_type, run_id
                    )
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (source_node_id, target_node_id, edge_type) DO NOTHING
                    """,
                    (original_id, revised_id, edge_type, run_id),
                )
                details = json.dumps(
                    {
                        "date": str(observed_on),
                        "new_semantic_hash": revised_hash,
                        "old_semantic_hash": original_hash,
                        "run_id": run_id,
                        "table": table,
                        "variable": variable,
                    },
                    sort_keys=True,
                )
                cursor.execute(
                    """
                    INSERT INTO lineage.revision_event (
                        original_node_id, revised_node_id, revision_type, branch,
                        reason, event_time, actor, details
                    )
                    SELECT %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    WHERE NOT EXISTS (
                        SELECT 1 FROM lineage.revision_event
                        WHERE original_node_id = %s
                          AND revised_node_id = %s
                          AND revision_type = %s
                          AND branch = %s
                          AND details->>'run_id' = %s
                    )
                    """,
                    (
                        original_id,
                        revised_id,
                        revision_type.value,
                        "latest_revised",
                        f"macro observation changed: {self.schema}.{table}.{variable}",
                        event_time,
                        actor,
                        details,
                        original_id,
                        revised_id,
                        revision_type.value,
                        "latest_revised",
                        run_id,
                    ),
                )
                revisions += 1

        return MacroRevisionResult(
            nodes_recorded=len(node_hashes), revisions_recorded=revisions
        )
