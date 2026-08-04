"""Governed, atomic resolution of market-data quarantine events (C027 / BL-40)."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from src.identity.canonical import semantic_hash
from src.market.publication import publish_provider_rows


CORRECTION_NAMESPACE = uuid.UUID("eb76c13c-3d57-5af7-b3d7-a780bf936fc1")
REVISION_TYPES = frozenset(
    {"PROVIDER_CORRECTION", "PIPELINE_ERROR", "SCHEMA_REINTERPRETATION"}
)


class MarketCorrectionError(RuntimeError):
    """A quarantine cannot be resolved without violating the correction contract."""


@dataclass(frozen=True, slots=True)
class CorrectionRequest:
    revision_type: str
    corrected_record: Mapping[str, Any]
    reason: str
    corrected_by: str
    compared_provider_id: str | None = None

    def validate(self) -> None:
        if self.revision_type not in REVISION_TYPES:
            raise MarketCorrectionError(
                f"revision_type must be one of {sorted(REVISION_TYPES)}"
            )
        if not isinstance(self.corrected_record, Mapping):
            raise MarketCorrectionError("corrected_record must be a mapping")
        missing = [
            field
            for field in ("time", "open", "high", "low", "close")
            if self.corrected_record.get(field) is None
        ]
        if missing:
            raise MarketCorrectionError(f"corrected_record missing fields: {missing}")
        if not self.reason.strip() or not self.corrected_by.strip():
            raise MarketCorrectionError("reason and corrected_by are required")
        if self.revision_type == "PROVIDER_CORRECTION" and not (
            self.compared_provider_id and self.compared_provider_id.strip()
        ):
            raise MarketCorrectionError(
                "PROVIDER_CORRECTION requires compared_provider_id evidence"
            )


@dataclass(frozen=True, slots=True)
class QuarantineContext:
    quarantine_id: str
    status: str
    correction_event_id: str | None
    provider_id: str | None
    provider_symbol: str | None
    interval_id: str | None
    observed_at: datetime | None
    source_uri: str | None
    source_record: Mapping[str, Any]
    context_version: int | None

    def require_replayable(self) -> None:
        values = (
            self.provider_id,
            self.provider_symbol,
            self.interval_id,
            self.observed_at,
            self.source_uri,
        )
        if self.context_version != 1 or any(value is None or value == "" for value in values):
            raise MarketCorrectionError(
                f"quarantine {self.quarantine_id} has no typed replay context"
            )
        if self.observed_at is None or self.observed_at.tzinfo is None:
            raise MarketCorrectionError("quarantine observed_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    correction_event_id: str
    canonical_count: int
    idempotent: bool


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _normalized_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(record)
    time_value = normalized.get("time")
    if isinstance(time_value, str):
        try:
            time_value = datetime.fromisoformat(time_value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MarketCorrectionError("corrected_record.time must be ISO-8601") from exc
        normalized["time"] = time_value
    if not isinstance(time_value, datetime) or time_value.tzinfo is None:
        raise MarketCorrectionError("corrected_record.time must be timezone-aware")
    normalized["time"] = time_value.astimezone(timezone.utc)
    return normalized


def _correction_id(quarantine_id: str, request: CorrectionRequest) -> str:
    digest = semantic_hash(
        {
            "quarantine_id": quarantine_id,
            "revision_type": request.revision_type,
            "corrected_record": _jsonable(request.corrected_record),
            "reason": request.reason.strip(),
            "corrected_by": request.corrected_by.strip(),
            "compared_provider_id": (
                request.compared_provider_id.strip()
                if request.compared_provider_id
                else None
            ),
        }
    )
    return str(uuid.uuid5(CORRECTION_NAMESPACE, digest))


def _lock_quarantine(conn, quarantine_id: str) -> QuarantineContext:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT quarantine_id::text, status, correction_event_id::text,
                   provider_id, provider_symbol, interval_id, observed_at,
                   source_uri, source_record, context_version
            FROM quality.quarantine_event
            WHERE quarantine_id=%s::uuid
            FOR UPDATE
            """,
            (quarantine_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise MarketCorrectionError(f"quarantine {quarantine_id} not found")
    return QuarantineContext(*row)


def _insert_correction_event(
    conn,
    *,
    correction_id: str,
    context: QuarantineContext,
    request: CorrectionRequest,
    corrected_record: Mapping[str, Any],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO quality.correction_event (
                correction_event_id, quarantine_id, revision_type,
                compared_provider_id, old_record, corrected_record,
                reason, corrected_by
            ) VALUES (
                %s::uuid, %s::uuid, %s, %s, %s::jsonb, %s::jsonb, %s, %s
            )
            """,
            (
                correction_id,
                context.quarantine_id,
                request.revision_type,
                request.compared_provider_id,
                json.dumps(_jsonable(context.source_record), sort_keys=True),
                json.dumps(_jsonable(corrected_record), sort_keys=True),
                request.reason.strip(),
                request.corrected_by.strip(),
            ),
        )


def _mark_corrected(
    conn, *, context: QuarantineContext, correction_id: str, request: CorrectionRequest
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE quality.quarantine_event
            SET status='CORRECTED', correction_event_id=%s::uuid, resolution=%s
            WHERE quarantine_id=%s::uuid AND status='OPEN'
            """,
            (
                correction_id,
                f"{request.revision_type}: {request.reason.strip()}",
                context.quarantine_id,
            ),
        )
        if cur.rowcount != 1:
            raise MarketCorrectionError("quarantine state changed while correcting")


def _savepoint(conn, statement: str) -> None:
    with conn.cursor() as cur:
        cur.execute(statement)


def apply_market_correction(
    conn, *, quarantine_id: str, request: CorrectionRequest
) -> CorrectionResult:
    """Resolve one quarantine without committing; caller owns the outer transaction."""
    request.validate()
    corrected_record = _normalized_record(request.corrected_record)
    correction_id = _correction_id(quarantine_id, request)
    _savepoint(conn, "SAVEPOINT c027_market_correction")
    try:
        context = _lock_quarantine(conn, quarantine_id)
        context.require_replayable()
        if context.status == "CORRECTED":
            if context.correction_event_id == correction_id:
                _savepoint(conn, "RELEASE SAVEPOINT c027_market_correction")
                return CorrectionResult(correction_id, canonical_count=0, idempotent=True)
            raise MarketCorrectionError(
                "quarantine already resolved by a different correction"
            )
        if context.status != "OPEN":
            raise MarketCorrectionError(
                f"quarantine status {context.status!r} cannot be corrected"
            )

        publication = publish_provider_rows(
            conn,
            provider_id=str(context.provider_id),
            provider_symbol=str(context.provider_symbol),
            interval_id=str(context.interval_id),
            rows=[corrected_record],
            source_uri=f"correction://{context.quarantine_id}",
            observed_at=datetime.now(timezone.utc),
            quality_observed_at=context.observed_at,
        )
        if (
            len(publication.accepted) != 1
            or publication.canonical_count != 1
            or publication.quarantine_count != 0
        ):
            raise MarketCorrectionError(
                "corrected record did not produce exactly one accepted canonical bar"
            )
        _insert_correction_event(
            conn,
            correction_id=correction_id,
            context=context,
            request=request,
            corrected_record=corrected_record,
        )
        _mark_corrected(
            conn, context=context, correction_id=correction_id, request=request
        )
    except Exception:
        _savepoint(conn, "ROLLBACK TO SAVEPOINT c027_market_correction")
        _savepoint(conn, "RELEASE SAVEPOINT c027_market_correction")
        raise
    _savepoint(conn, "RELEASE SAVEPOINT c027_market_correction")
    return CorrectionResult(
        correction_event_id=correction_id,
        canonical_count=publication.canonical_count,
        idempotent=False,
    )


__all__ = [
    "CorrectionRequest",
    "CorrectionResult",
    "MarketCorrectionError",
    "QuarantineContext",
    "apply_market_correction",
]
