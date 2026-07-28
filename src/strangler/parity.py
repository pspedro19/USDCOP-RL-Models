"""Parity harness and append-only ledger for the strangler migration (BL-31, §8.2/§29.1).

The parity rule is the one FABRIC states: the artefacts produced by the artisanal
(legacy) path and by the candidate path must hash to the same value. Where WE own the
writer (our JSON ledgers, signals, targets, bundles) the comparison is the canonical
`semantic_hash` and byte parity holds by construction. Where we do NOT own the writer
(Parquet, broker payloads) only a physical `bytes_hash` is available here — semantic
normalisation for those formats is deliberately NOT implemented, and a layer whose plan
requires `canonical_json` refuses to accept a `bytes` observation (fail-closed).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from src.identity.canonical import CanonicalizationError, canonical_json_bytes, semantic_hash
from src.strangler.contracts import (
    AcceptanceAttestation,
    CriterionStatus,
    ExecutionReadiness,
    HashKind,
    LayerState,
    LayerTransition,
    MigrationLayer,
    ParityObservation,
    ParityVerdict,
    RecordType,
    StranglerContractError,
    parse_enum,
    parse_layer,
)

#: Extensions whose writer we control -> canonical JSON semantics.
CANONICAL_SUFFIXES = {".json", ".jsonl"}


class ParityError(ValueError):
    """The parity comparison could not be performed."""


def _read_json_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def hash_artifact(path: str | Path) -> tuple[str, HashKind]:
    """Return `(hash, kind)` for one artifact.

    `.json`/`.jsonl` are re-serialised canonically, so two writers that disagree only on
    key order or whitespace still hash equal (that is semantic parity). Anything else is
    hashed byte-wise and flagged as such — never silently promoted to semantic parity.
    """

    p = Path(path)
    if not p.is_file():
        raise ParityError(f"artifact does not exist: {p}")
    if p.suffix.lower() in CANONICAL_SUFFIXES:
        try:
            payload = _read_json_payload(p)
        except json.JSONDecodeError as exc:
            raise ParityError(f"{p} is not valid JSON: {exc}") from exc
        try:
            return semantic_hash(payload), HashKind.CANONICAL_JSON
        except CanonicalizationError as exc:
            # NaN/Infinity or an unsupported type: the artifact itself violates the repo's
            # JSON-safety rule. Refuse rather than fall back to a byte hash that would
            # look like parity evidence.
            raise ParityError(f"{p} is not canonicalizable: {exc}") from exc
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    return f"sha256:{digest}", HashKind.BYTES


def observe_parity(
    *,
    layer: MigrationLayer | str,
    artifact_id: str,
    legacy_path: str | Path,
    candidate_path: str | Path,
    required_hash_kind: HashKind,
    observed_at: datetime | None = None,
) -> ParityObservation:
    """Compare one legacy artifact against its candidate twin.

    Any failure (missing file, unreadable JSON, weaker hash kind than the layer requires,
    disagreeing hash kinds) produces an `INVALID` observation, which the gate treats
    exactly like a mismatch: it breaks the green streak.
    """

    lyr = parse_layer(layer)
    when = observed_at or datetime.now(timezone.utc)

    def _invalid(note: str) -> ParityObservation:
        return ParityObservation(
            layer=lyr,
            artifact_id=artifact_id,
            observed_at=when,
            legacy_hash=None,
            candidate_hash=None,
            hash_kind=required_hash_kind,
            verdict=ParityVerdict.INVALID,
            note=note,
        )

    try:
        legacy_hash, legacy_kind = hash_artifact(legacy_path)
    except ParityError as exc:
        return _invalid(f"legacy: {exc}")
    try:
        candidate_hash, candidate_kind = hash_artifact(candidate_path)
    except ParityError as exc:
        return _invalid(f"candidate: {exc}")

    if legacy_kind is not candidate_kind:
        return _invalid(
            f"hash kinds disagree: legacy={legacy_kind}, candidate={candidate_kind}"
        )
    if legacy_kind is not required_hash_kind:
        return _invalid(
            f"layer requires {required_hash_kind} parity but the artifacts only support "
            f"{legacy_kind} (§8.2: no byte parity over formats we do not write)"
        )

    return ParityObservation(
        layer=lyr,
        artifact_id=artifact_id,
        observed_at=when,
        legacy_hash=legacy_hash,
        candidate_hash=candidate_hash,
        hash_kind=legacy_kind,
        verdict=(
            ParityVerdict.MATCH if legacy_hash == candidate_hash else ParityVerdict.MISMATCH
        ),
    )


# --------------------------------------------------------------------------- ledger


class ParityLedger:
    """Append-only JSONL ledger of migration events, written canonically.

    Records are never rewritten: the migration state of every layer is *derived* by
    replaying the file, so a rollback is an event, not an edit.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: list[dict[str, Any]] = []
        if self.path.is_file():
            self._records = list(self._read(self.path))

    # -- io -------------------------------------------------------------------

    @staticmethod
    def _read(path: Path) -> Iterator[dict[str, Any]]:
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StranglerContractError(
                    f"{path}:{lineno} is not valid JSON: {exc}"
                ) from exc
            if not isinstance(record, dict) or "record_type" not in record:
                raise StranglerContractError(f"{path}:{lineno} is not a ledger record")
            yield record

    def append(self, event: Any) -> dict[str, Any]:
        """Append one contract object (observation / transition / attestation)."""

        if not hasattr(event, "to_record"):
            raise StranglerContractError(
                f"{type(event).__name__} is not a ledger event (no to_record)"
            )
        record = event.to_record()
        line = canonical_json_bytes(record)  # rejects NaN/Infinity by construction
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as fh:
            fh.write(line + b"\n")
        self._records.append(json.loads(line.decode("utf-8")))
        return self._records[-1]

    # -- projections ----------------------------------------------------------

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._records)

    def _of_type(self, record_type: RecordType) -> list[dict[str, Any]]:
        return [r for r in self._records if r.get("record_type") == str(record_type)]

    @staticmethod
    def _parse_dt(value: Any, what: str) -> datetime:
        if not isinstance(value, str):
            raise StranglerContractError(f"{what} must be an ISO-8601 string")
        raw = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise StranglerContractError(f"{what} is not ISO-8601: {value!r}") from exc
        if parsed.tzinfo is None:
            raise StranglerContractError(f"{what} must be timezone-aware")
        return parsed.astimezone(timezone.utc)

    def observations(
        self, layer: MigrationLayer | str | None = None
    ) -> tuple[ParityObservation, ...]:
        wanted = parse_layer(layer) if layer is not None else None
        out: list[ParityObservation] = []
        for record in self._of_type(RecordType.PARITY_OBSERVATION):
            lyr = parse_layer(record["layer"])
            if wanted is not None and lyr is not wanted:
                continue
            out.append(
                ParityObservation(
                    layer=lyr,
                    artifact_id=record["artifact_id"],
                    observed_at=self._parse_dt(record["observed_at"], "observed_at"),
                    legacy_hash=record.get("legacy_hash"),
                    candidate_hash=record.get("candidate_hash"),
                    hash_kind=parse_enum(HashKind, record["hash_kind"], "hash_kind"),
                    verdict=parse_enum(ParityVerdict, record["verdict"], "verdict"),
                    note=record.get("note") or "",
                )
            )
        return tuple(sorted(out, key=lambda o: o.observed_at))

    def transitions(self) -> tuple[LayerTransition, ...]:
        out = [
            LayerTransition(
                layer=parse_layer(r["layer"]),
                state=parse_enum(LayerState, r["state"], "state"),
                at=self._parse_dt(r["at"], "at"),
                reason=r["reason"],
            )
            for r in self._of_type(RecordType.LAYER_TRANSITION)
        ]
        return tuple(sorted(out, key=lambda t: t.at))

    def acceptance(self) -> dict[int, AcceptanceAttestation]:
        latest: dict[int, AcceptanceAttestation] = {}
        for r in self._of_type(RecordType.ACCEPTANCE_ATTESTATION):
            att = AcceptanceAttestation(
                criterion_id=int(r["criterion_id"]),
                status=parse_enum(CriterionStatus, r["status"], "status"),
                at=self._parse_dt(r["at"], "at"),
                evidence=r.get("evidence") or "",
            )
            current = latest.get(att.criterion_id)
            if current is None or att.at >= current.at:
                latest[att.criterion_id] = att
        return latest

    def execution_readiness(self) -> ExecutionReadiness | None:
        records = self._of_type(RecordType.EXECUTION_READINESS)
        if not records:
            return None
        latest = max(records, key=lambda r: self._parse_dt(r["at"], "at"))
        return ExecutionReadiness(
            at=self._parse_dt(latest["at"], "at"),
            canary_flow=latest.get("canary_flow") or "",
            service_outside_airflow=bool(latest.get("service_outside_airflow")),
            idempotency_proven=bool(latest.get("idempotency_proven")),
            pretrade_enforced=bool(latest.get("pretrade_enforced")),
            kill_switch_with_airflow_down=bool(latest.get("kill_switch_with_airflow_down")),
            reconciliation_pre_intra_eod=bool(latest.get("reconciliation_pre_intra_eod")),
            attested_by=latest.get("attested_by") or "",
        )


def green_streak(
    observations: Sequence[ParityObservation] | Iterable[ParityObservation],
) -> tuple[int, datetime | None, datetime | None]:
    """Return `(count, since, until)` for the trailing run of MATCH observations.

    Any MISMATCH or INVALID resets the streak: "paridad verde SOSTENIDA" (§29.2) means
    uninterrupted, not "mostly green".
    """

    ordered = sorted(observations, key=lambda o: o.observed_at)
    count = 0
    since: datetime | None = None
    until: datetime | None = None
    for obs in ordered:
        if obs.verdict is ParityVerdict.MATCH:
            count += 1
            since = obs.observed_at if count == 1 else since
            until = obs.observed_at
        else:
            count = 0
            since = None
            until = None
    return count, since, until
