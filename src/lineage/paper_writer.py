"""Transactional writer for a paper signal's persisted golden lineage path."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Mapping, Sequence

import pandas as pd

from src.forecasting.dataset_loader import DatasetProvenance, _frame_semantic_hash
from src.identity.canonical import semantic_hash


@dataclass(frozen=True, slots=True)
class PaperLineageDeclaration:
    timestamp: str
    signal_node_id: str
    snapshot_node_id: str
    bar_l0_node_id: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def _upsert_node(
    cursor: Any,
    *,
    node_type: str,
    payload: Mapping[str, Any],
    row_count: int,
    min_event_time: str,
    max_event_time: str,
    storage_uri: str,
    availability_quality: str,
    verified_at: datetime,
) -> str:
    digest = semantic_hash(payload)
    cursor.execute(
        """
        INSERT INTO lineage.node (
            node_type, semantic_hash, bytes_hash, schema_version, row_count,
            min_event_time, max_event_time, quality_status, status, storage_uri,
            availability_quality, last_verified_at
        ) VALUES (
            %s, %s, %s, '1', %s, %s, %s, 'PASS', 'VALID', %s, %s, %s
        )
        ON CONFLICT (node_type, semantic_hash) DO UPDATE
        SET last_verified_at = GREATEST(
            lineage.node.last_verified_at, EXCLUDED.last_verified_at
        )
        RETURNING node_id::text
        """,
        (
            node_type,
            digest,
            digest,
            row_count,
            min_event_time,
            max_event_time,
            storage_uri,
            availability_quality,
            verified_at,
        ),
    )
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"lineage node upsert returned no id for {node_type}")
    return str(row[0])


def _edge(cursor: Any, source: str, target: str, edge_type: str, run_id: str) -> None:
    cursor.execute(
        """
        INSERT INTO lineage.edge (source_node_id, target_node_id, edge_type, run_id)
        VALUES (%s::uuid, %s::uuid, %s, %s)
        ON CONFLICT (source_node_id, target_node_id, edge_type) DO NOTHING
        """,
        (source, target, edge_type, run_id),
    )


def _strategy_node(cursor: Any, strategy_id: str, node_id: str, role: str) -> None:
    cursor.execute(
        """
        INSERT INTO lineage.strategy_node (strategy_id, node_id, role)
        VALUES (%s, %s::uuid, %s)
        ON CONFLICT (strategy_id, node_id, role) DO NOTHING
        """,
        (strategy_id, node_id, role),
    )


def persist_paper_lineage(
    cursor: Any,
    *,
    strategy_id: str,
    trade: Mapping[str, Any],
    dataset: pd.DataFrame,
    provenance: DatasetProvenance,
    run_id: str,
    verified_at: datetime,
) -> PaperLineageDeclaration:
    """Persist signal -> consumed snapshot -> exact entry bar without committing."""

    timestamp = str(trade.get("timestamp", ""))
    side = str(trade.get("side", ""))
    if not strategy_id or not timestamp or not side or not run_id:
        raise ValueError("strategy_id, trade timestamp/side and run_id are required")
    if verified_at.tzinfo is None or verified_at.utcoffset() is None:
        raise ValueError("verified_at must be timezone-aware")

    columns = list(provenance.snapshot_columns)
    missing = [column for column in columns if column not in dataset.columns]
    if missing:
        raise ValueError(f"dataset no longer contains provenance columns: {missing}")
    observed_snapshot_hash = _frame_semantic_hash(dataset, columns)
    if observed_snapshot_hash != provenance.snapshot_semantic_hash:
        raise ValueError("dataset content does not match declared snapshot provenance")

    signal_day = pd.Timestamp(timestamp).tz_localize(None).normalize()
    dates = pd.to_datetime(dataset["date"]).dt.tz_localize(None).dt.normalize()
    matching = dataset.loc[dates == signal_day]
    if len(matching) != 1:
        raise ValueError(
            f"paper signal timestamp must resolve to exactly one L0 bar; got {len(matching)}"
        )
    bar_columns = [column for column in ("date", "open", "high", "low", "close") if column in dataset]
    if bar_columns != ["date", "open", "high", "low", "close"]:
        raise ValueError("dataset lacks the complete daily OHLC bar")
    bar_payload = {
        "asset_id": "usdcop",
        "interval_id": "1d",
        "bar": json.loads(
            json.dumps(
                {
                    key: (
                        pd.Timestamp(value).isoformat()
                        if key == "date"
                        else float(value)
                    )
                    for key, value in zip(bar_columns, matching.iloc[0][bar_columns], strict=True)
                }
            )
        ),
    }
    snapshot_payload = {
        "asset_id": "usdcop",
        "contract": "CTR-FORECAST-DATA-LOADER-001",
        "snapshot_semantic_hash": provenance.snapshot_semantic_hash,
        "snapshot_columns": list(provenance.snapshot_columns),
        "row_count": provenance.row_count,
        "min_event_time": provenance.min_event_time,
        "max_event_time": provenance.max_event_time,
        "sources": {"ohlcv": asdict(provenance.ohlcv), "macro": asdict(provenance.macro)},
    }
    signal_payload = {"strategy_id": strategy_id, "timestamp": timestamp, "side": side}

    signal_id = _upsert_node(
        cursor,
        node_type="paper_signal",
        payload=signal_payload,
        row_count=1,
        min_event_time=timestamp,
        max_event_time=timestamp,
        storage_uri=f"paper-ledger://{strategy_id}/{timestamp}",
        availability_quality="RECONSTRUCTED",
        verified_at=verified_at,
    )
    snapshot_id = _upsert_node(
        cursor,
        node_type="data_snapshot",
        payload=snapshot_payload,
        row_count=provenance.row_count,
        min_event_time=provenance.min_event_time,
        max_event_time=provenance.max_event_time,
        storage_uri="dataset://forecasting/usdcop/" + provenance.snapshot_semantic_hash,
        availability_quality="RECONSTRUCTED",
        verified_at=verified_at,
    )
    bar_id = _upsert_node(
        cursor,
        node_type="bar_l0",
        payload=bar_payload,
        row_count=1,
        min_event_time=timestamp,
        max_event_time=timestamp,
        storage_uri=f"dataset://forecasting/usdcop/bar/{signal_day.date().isoformat()}",
        availability_quality="RECONSTRUCTED",
        verified_at=verified_at,
    )
    _edge(cursor, signal_id, snapshot_id, "DERIVED_FROM", run_id)
    _edge(cursor, snapshot_id, bar_id, "CONSUMED", run_id)
    _strategy_node(cursor, strategy_id, signal_id, "SIGNAL")
    _strategy_node(cursor, strategy_id, snapshot_id, "INPUT")
    _strategy_node(cursor, strategy_id, bar_id, "INPUT")
    return PaperLineageDeclaration(timestamp, signal_id, snapshot_id, bar_id)
