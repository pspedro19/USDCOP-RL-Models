"""Atomic Fabric publication boundary for normalized provider OHLCV observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping

from src.data_quality.ingest_guard import (
    record_quarantine,
    resolved_instrument_id,
    ruleset_from_spine,
)
from src.data_quality.rules import QualityDecision
from src.identity.canonical import semantic_hash


@dataclass(frozen=True, slots=True)
class MarketPublicationResult:
    accepted: tuple[Mapping[str, Any], ...]
    raw_count: int
    canonical_count: int
    quarantine_count: int


def _timestamp(value: object) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("market observation time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _json_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Decimal):
        return str(value)
    return value


def _safe_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_scalar(value) for key, value in row.items()}


def _structurally_representable(row: Mapping[str, Any]) -> bool:
    try:
        values = {name: Decimal(str(row[name])) for name in ("open", "high", "low", "close")}
        if any(not value.is_finite() for value in values.values()):
            return False
        if values["high"] < max(values.values()) or values["low"] > min(values.values()):
            return False
        volume = row.get("volume")
        if volume is not None:
            parsed_volume = Decimal(str(volume))
            if not parsed_volume.is_finite() or parsed_volume < 0:
                return False
        _timestamp(row.get("time"))
    except (KeyError, TypeError, ValueError, InvalidOperation):
        return False
    return True


def _quality_observed_at(row: Mapping[str, Any], *, fallback: datetime) -> datetime:
    """Return the economic observation time used by provider/date-scoped rules.

    ``fallback`` is only for malformed timestamps that will be quarantined by another
    rule.  A valid historical row must never inherit the retrieval time of its batch.
    """
    try:
        return _timestamp(row.get("time"))
    except (TypeError, ValueError):
        return fallback


def _source_hash(
    row: Mapping[str, Any], *, provider_id: str, provider_symbol: str, interval_id: str
) -> str:
    return semantic_hash(
        {
            "provider_id": provider_id,
            "provider_symbol": provider_symbol,
            "interval_id": interval_id,
            "row": _safe_row(row),
        }
    )


def _insert_raw(
    conn,
    *,
    instrument_id: str,
    provider_id: str,
    provider_symbol: str,
    interval_id: str,
    row: Mapping[str, Any],
    observed_at: datetime,
    source_uri: str | None,
) -> str:
    event_time = _timestamp(row["time"])
    payload_hash = _source_hash(
        row,
        provider_id=provider_id,
        provider_symbol=provider_symbol,
        interval_id=interval_id,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO market.raw_bar (
                instrument_id, provider_id, provider_symbol, interval_id, event_time,
                available_at, retrieved_at, open, high, low, close, volume,
                source_payload_hash, source_uri, metadata
            ) VALUES (
                %s::uuid, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s::jsonb
            )
            ON CONFLICT (
                instrument_id, provider_id, interval_id, event_time, source_payload_hash
            ) DO NOTHING
            RETURNING raw_bar_id::text
            """,
            (
                instrument_id,
                provider_id,
                provider_symbol,
                interval_id,
                event_time,
                observed_at,
                observed_at,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row.get("volume"),
                payload_hash,
                source_uri,
                '{"publication_contract":"C025"}',
            ),
        )
        inserted = cur.fetchone()
        if inserted:
            return str(inserted[0])
        cur.execute(
            """
            SELECT raw_bar_id::text FROM market.raw_bar
            WHERE instrument_id=%s::uuid AND provider_id=%s AND interval_id=%s
              AND event_time=%s AND source_payload_hash=%s
            """,
            (instrument_id, provider_id, interval_id, event_time, payload_hash),
        )
        stored = cur.fetchone()
    if not stored:
        raise RuntimeError("raw_bar conflict returned no durable identity")
    return str(stored[0])


def _insert_canonical(
    conn,
    *,
    instrument_id: str,
    interval_id: str,
    raw_bar_id: str,
    row: Mapping[str, Any],
    observed_at: datetime,
) -> None:
    event_time = _timestamp(row["time"])
    digest = semantic_hash(
        {
            "instrument_id": instrument_id,
            "interval_id": interval_id,
            "event_time": event_time,
            "bar_method": "provider_official",
            "canonical_version": 1,
            "source_raw_bar_ids": [raw_bar_id],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row.get("volume"),
        }
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO market.canonical_bar (
                instrument_id, interval_id, event_time, canonical_version,
                available_at, bar_method, source_raw_bar_ids,
                open, high, low, close, volume, semantic_hash, quality_status
            ) VALUES (
                %s::uuid, %s, %s, 1, %s, 'provider_official', ARRAY[%s::uuid],
                %s, %s, %s, %s, %s, %s, 'VALID'
            )
            ON CONFLICT (semantic_hash) DO NOTHING
            RETURNING canonical_bar_id::text
            """,
            (
                instrument_id,
                interval_id,
                event_time,
                observed_at,
                raw_bar_id,
                row["open"],
                row["high"],
                row["low"],
                row["close"],
                row.get("volume"),
                digest,
            ),
        )
        inserted = cur.fetchone()
        if inserted:
            canonical_id = str(inserted[0])
        else:
            cur.execute(
                "SELECT canonical_bar_id::text FROM market.canonical_bar "
                "WHERE semantic_hash=%s",
                (digest,),
            )
            stored = cur.fetchone()
            if not stored:
                raise RuntimeError("canonical_bar conflict returned no durable identity")
            canonical_id = str(stored[0])
        cur.execute(
            """
            INSERT INTO market.canonical_bar_source (
                canonical_bar_id, raw_bar_id, source_order
            ) VALUES (%s::uuid, %s::uuid, 0)
            ON CONFLICT DO NOTHING
            """,
            (canonical_id, raw_bar_id),
        )


def publish_provider_rows(
    conn,
    *,
    provider_id: str,
    provider_symbol: str,
    interval_id: str,
    rows: Iterable[Mapping[str, Any]],
    source_uri: str | None = None,
    observed_at: datetime | None = None,
    quality_observed_at: datetime | None = None,
) -> MarketPublicationResult:
    """Publish without committing; the caller owns one transaction with legacy writes.

    ``observed_at`` is retrieval/availability time. ``quality_observed_at`` is reserved
    for governed correction replay, where the original rule instant must be preserved.
    Ordinary ingestion evaluates every row at its own economic event time.
    """

    retrieval_instant = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    explicit_quality_instant = (
        quality_observed_at.astimezone(timezone.utc)
        if quality_observed_at is not None
        else None
    )
    ruleset = ruleset_from_spine(conn)
    instrument_id = resolved_instrument_id(ruleset, provider_id, provider_symbol)
    accepted: list[Mapping[str, Any]] = []
    raw_count = canonical_count = quarantine_count = 0

    for original in rows:
        row = dict(original)
        quality_instant = explicit_quality_instant or _quality_observed_at(
            row, fallback=retrieval_instant
        )
        if instrument_id is None or not _structurally_representable(row):
            safe = _safe_row(row)
            decision = ruleset.evaluate_provider_bar(
                provider_id, provider_symbol, safe, observed_at=quality_instant
            )
            if decision.accepted:
                decision = QualityDecision(
                    False,
                    "QUARANTINED",
                    "bar.raw_schema",
                    safe,
                    "observation cannot be represented by market.raw_bar",
                )
            record_quarantine(
                conn,
                provider_id=provider_id,
                provider_symbol=provider_symbol,
                row=safe,
                decision=decision,
                rule_version=ruleset.version,
                instrument_id=instrument_id,
            )
            quarantine_count += 1
            continue

        raw_bar_id = _insert_raw(
            conn,
            instrument_id=instrument_id,
            provider_id=provider_id,
            provider_symbol=provider_symbol,
            interval_id=interval_id,
            row=row,
            observed_at=retrieval_instant,
            source_uri=source_uri,
        )
        raw_count += 1
        decision = ruleset.evaluate_provider_bar(
            provider_id, provider_symbol, row, observed_at=quality_instant
        )
        if not decision.accepted:
            record_quarantine(
                conn,
                provider_id=provider_id,
                provider_symbol=provider_symbol,
                row=row,
                decision=decision,
                rule_version=ruleset.version,
                instrument_id=instrument_id,
            )
            quarantine_count += 1
            continue
        _insert_canonical(
            conn,
            instrument_id=instrument_id,
            interval_id=interval_id,
            raw_bar_id=raw_bar_id,
            row=row,
            observed_at=retrieval_instant,
        )
        canonical_count += 1
        accepted.append(original)

    return MarketPublicationResult(
        accepted=tuple(accepted),
        raw_count=raw_count,
        canonical_count=canonical_count,
        quarantine_count=quarantine_count,
    )
