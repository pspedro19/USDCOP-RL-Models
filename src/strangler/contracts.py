"""Types of the USD/COP strangler migration control plane (BL-31, FABRIC §29).

The migration replaces the artisanal COP chain layer by layer, running both
implementations in parallel and demanding parity before each step:

    ingest -> canon -> verify -> features -> dataset -> train -> gate -> signal -> execute

Everything here is *engineering* governance: it decides when a layer is allowed to
advance, never what a model should predict. No parameter in this module is fitted,
searched or tuned against results (0 trials).

Backend-only on purpose: BL-31 declares "Impacto frontend: Ninguno", so there is no
TypeScript mirror. If the Control Tower (BL-32) ever renders the parity table, the
mirror becomes mandatory and must be proposed as a contract change.

Contract: CTR-STRANGLER-COP-001 · Version: 1.0.0
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence

CONTRACT_ID = "CTR-STRANGLER-COP-001"
CONTRACT_VERSION = "1.0.0"


class StranglerContractError(ValueError):
    """A declaration violates the strangler contract (fail-closed)."""


class MigrationLayer(StrEnum):
    """The nine layers of §29, in migration order."""

    INGEST = "ingest"
    CANON = "canon"
    VERIFY = "verify"
    FEATURES = "features"
    DATASET = "dataset"
    TRAIN = "train"
    GATE = "gate"
    SIGNAL = "signal"
    EXECUTE = "execute"


#: §29 order. `EXECUTE` (L7) is last by construction, not by convention.
LAYER_ORDER: tuple[MigrationLayer, ...] = (
    MigrationLayer.INGEST,
    MigrationLayer.CANON,
    MigrationLayer.VERIFY,
    MigrationLayer.FEATURES,
    MigrationLayer.DATASET,
    MigrationLayer.TRAIN,
    MigrationLayer.GATE,
    MigrationLayer.SIGNAL,
    MigrationLayer.EXECUTE,
)

#: The layer that carries real money. Migrates last, with the double net of §29.4.
MONEY_LAYER = MigrationLayer.EXECUTE


class LayerState(StrEnum):
    NOT_STARTED = "NOT_STARTED"      # only the artisanal path exists
    PARALLEL = "PARALLEL"            # both paths run; parity not yet sustained
    PARITY_GREEN = "PARITY_GREEN"    # sustained parity window met, not switched yet
    MIGRATED = "MIGRATED"            # new path is authoritative
    ROLLED_BACK = "ROLLED_BACK"      # new path disabled, artisanal re-enabled (§29.6)


class HashKind(StrEnum):
    """§8.2: byte parity only where WE control the writer."""

    CANONICAL_JSON = "canonical_json"  # semantic_hash == bytes_hash by construction
    BYTES = "bytes"                    # physical integrity only, NOT semantic parity


class ParityVerdict(StrEnum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    INVALID = "INVALID"  # could not be compared under the required hash kind


class CriterionStatus(StrEnum):
    PENDING = "PENDING"
    PASS = "PASS"
    FAIL = "FAIL"


class RecordType(StrEnum):
    PARITY_OBSERVATION = "parity_observation"
    LAYER_TRANSITION = "layer_transition"
    ACCEPTANCE_ATTESTATION = "acceptance_attestation"
    EXECUTION_READINESS = "execution_readiness"


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_ASSET_URI = re.compile(r"^asset://[A-Za-z0-9_\-./{}]+$")


def _require_aware(value: datetime, what: str) -> datetime:
    if not isinstance(value, datetime):
        raise StranglerContractError(f"{what} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.utcoffset() is None:
        raise StranglerContractError(f"{what} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _require_text(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StranglerContractError(f"{what} must be a non-empty string")
    return value


def _require_hash(value: Any, what: str) -> str:
    _require_text(value, what)
    if not _SHA256.fullmatch(value):
        raise StranglerContractError(f"{what} must look like sha256:<64 hex>, got {value!r}")
    return value


def _tuple_of_text(values: Any, what: str, *, minimum: int = 0) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise StranglerContractError(f"{what} must be a list of strings")
    out = tuple(_require_text(v, f"{what}[]") for v in values)
    if len(out) < minimum:
        raise StranglerContractError(f"{what} needs at least {minimum} entr(y|ies)")
    return out


# --------------------------------------------------------------------------- plan


@dataclass(frozen=True, slots=True)
class SensorMigration:
    """§29.3: an `ExternalTaskSensor` becomes an Airflow Asset when its layer migrates.

    Declared per layer so the swap is auditable instead of incidental.
    """

    sensor_ref: str            # "<file>::<task_id>"
    external_dag_id: str       # the DAG the sensor waits on today
    replacement_asset: str     # "asset://usdcop/<layer>"

    def __post_init__(self) -> None:
        _require_text(self.sensor_ref, "sensor_ref")
        if "::" not in self.sensor_ref:
            raise StranglerContractError("sensor_ref must be '<file>::<task_id>'")
        _require_text(self.external_dag_id, "external_dag_id")
        if not _ASSET_URI.fullmatch(self.replacement_asset or ""):
            raise StranglerContractError(
                f"replacement_asset must be an asset:// URI, got {self.replacement_asset!r}"
            )


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """§29.6: rollback declared per layer, without data loss."""

    trigger: str
    steps: tuple[str, ...]
    data_loss: bool = False
    rehearsed: bool = False

    def __post_init__(self) -> None:
        _require_text(self.trigger, "rollback.trigger")
        if len(self.steps) < 2:
            raise StranglerContractError("rollback.steps must declare at least 2 steps")
        for step in self.steps:
            _require_text(step, "rollback.steps[]")
        if self.data_loss:
            raise StranglerContractError(
                "§29.6 forbids a rollback with data loss — both paths write immutable artifacts"
            )


@dataclass(frozen=True, slots=True)
class LayerPlan:
    layer: MigrationLayer
    legacy_anchors: tuple[str, ...]
    artifacts: tuple[str, ...]
    required_hash_kind: HashKind
    rollback: RollbackPlan
    candidate_generator: str | None = None      # BL-28 output; None = not built yet
    min_parallel_days: int = 14                 # §29.1 "≥ 2 semanas"
    min_observations: int = 2
    sensor_migrations: tuple[SensorMigration, ...] = ()
    caveat: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.layer, MigrationLayer):
            raise StranglerContractError(f"unknown layer {self.layer!r}")
        _tuple_of_text(self.legacy_anchors, f"{self.layer}.legacy_anchors", minimum=1)
        _tuple_of_text(self.artifacts, f"{self.layer}.artifacts", minimum=1)
        if not isinstance(self.required_hash_kind, HashKind):
            raise StranglerContractError(f"{self.layer}.required_hash_kind is not a HashKind")
        if self.candidate_generator is not None:
            _require_text(self.candidate_generator, f"{self.layer}.candidate_generator")
        if self.min_parallel_days < 14:
            raise StranglerContractError(
                f"{self.layer}.min_parallel_days={self.min_parallel_days} violates §29.1 (>= 14)"
            )
        if self.min_observations < 2:
            raise StranglerContractError(
                f"{self.layer}.min_observations must be >= 2 (a single sample is not a window)"
            )

    @property
    def generator_declared(self) -> bool:
        return self.candidate_generator is not None


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    """One of the 15 production acceptance criteria of §30."""

    id: int
    text: str
    owner: str  # the BL that must supply the evidence

    def __post_init__(self) -> None:
        if not 1 <= self.id <= 15:
            raise StranglerContractError(f"acceptance criterion id out of range: {self.id}")
        _require_text(self.text, "acceptance.text")
        _require_text(self.owner, "acceptance.owner")


@dataclass(frozen=True, slots=True)
class StranglerPlan:
    asset: str
    contract_id: str
    version: str
    layers: tuple[LayerPlan, ...]
    acceptance_criteria: tuple[AcceptanceCriterion, ...]
    #: §29.4 — the rest of the chain must hold parity for a month before L7 moves.
    execute_requires_days_after_last_migration: int = 30
    ab_cohort: tuple[str, ...] = ()  # §29.5 — untouched during the migration

    def __post_init__(self) -> None:
        _require_text(self.asset, "asset")
        if self.contract_id != CONTRACT_ID:
            raise StranglerContractError(
                f"plan contract_id {self.contract_id!r} != {CONTRACT_ID!r}"
            )
        if not re.fullmatch(r"\d+\.\d+\.\d+", str(self.version or "")):
            raise StranglerContractError(f"plan version must be semver, got {self.version!r}")
        declared = tuple(lp.layer for lp in self.layers)
        if declared != LAYER_ORDER:
            raise StranglerContractError(
                "plan must declare the nine §29 layers in order; got "
                f"{[str(x) for x in declared]}"
            )
        ids = sorted(c.id for c in self.acceptance_criteria)
        if ids != list(range(1, 16)):
            raise StranglerContractError(
                f"§30 requires exactly the 15 acceptance criteria; got ids {ids}"
            )
        if self.execute_requires_days_after_last_migration < 30:
            raise StranglerContractError(
                "§29.4 requires >= 30 days of green chain before the money layer moves"
            )

    def layer_plan(self, layer: MigrationLayer) -> LayerPlan:
        for lp in self.layers:
            if lp.layer is layer:
                return lp
        raise StranglerContractError(f"layer {layer} not declared in plan")

    def previous_layers(self, layer: MigrationLayer) -> tuple[MigrationLayer, ...]:
        idx = LAYER_ORDER.index(layer)
        return LAYER_ORDER[:idx]


# ------------------------------------------------------------------- ledger records


@dataclass(frozen=True, slots=True)
class ParityObservation:
    """One comparison of a legacy artifact against its candidate twin."""

    layer: MigrationLayer
    artifact_id: str
    observed_at: datetime
    legacy_hash: str | None
    candidate_hash: str | None
    hash_kind: HashKind
    verdict: ParityVerdict
    note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _require_aware(self.observed_at, "observed_at"))
        _require_text(self.artifact_id, "artifact_id")
        if self.verdict is ParityVerdict.INVALID:
            if not self.note:
                raise StranglerContractError("an INVALID observation must carry a note")
        else:
            _require_hash(self.legacy_hash, "legacy_hash")
            _require_hash(self.candidate_hash, "candidate_hash")
            equal = self.legacy_hash == self.candidate_hash
            if equal is not (self.verdict is ParityVerdict.MATCH):
                raise StranglerContractError(
                    "verdict contradicts the hashes — MATCH iff legacy_hash == candidate_hash"
                )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": str(RecordType.PARITY_OBSERVATION),
            "layer": str(self.layer),
            "artifact_id": self.artifact_id,
            "observed_at": self.observed_at,
            "legacy_hash": self.legacy_hash,
            "candidate_hash": self.candidate_hash,
            "hash_kind": str(self.hash_kind),
            "verdict": str(self.verdict),
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class LayerTransition:
    layer: MigrationLayer
    state: LayerState
    at: datetime
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _require_aware(self.at, "at"))
        _require_text(self.reason, "reason")

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": str(RecordType.LAYER_TRANSITION),
            "layer": str(self.layer),
            "state": str(self.state),
            "at": self.at,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class AcceptanceAttestation:
    criterion_id: int
    status: CriterionStatus
    at: datetime
    evidence: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _require_aware(self.at, "at"))
        if not 1 <= self.criterion_id <= 15:
            raise StranglerContractError(f"criterion_id out of range: {self.criterion_id}")
        if self.status is CriterionStatus.PASS:
            _require_text(self.evidence, "evidence")

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": str(RecordType.ACCEPTANCE_ATTESTATION),
            "criterion_id": self.criterion_id,
            "status": str(self.status),
            "at": self.at,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class ExecutionReadiness:
    """Interface to BL-30 (external execution service). NOT implemented here.

    BL-31 only consumes this attestation; producing it truthfully is BL-30's job.
    Every flag defaults to False so an absent attestation blocks L7 (fail-closed).
    """

    at: datetime
    canary_flow: str = ""
    service_outside_airflow: bool = False
    idempotency_proven: bool = False
    pretrade_enforced: bool = False
    kill_switch_with_airflow_down: bool = False
    reconciliation_pre_intra_eod: bool = False
    attested_by: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "at", _require_aware(self.at, "at"))
        if self.complete and not self.canary_flow:
            raise StranglerContractError(
                "a complete execution readiness must name the canary flow it was proven on (§29.4)"
            )

    @property
    def missing(self) -> tuple[str, ...]:
        flags = {
            "service_outside_airflow": self.service_outside_airflow,
            "idempotency_proven": self.idempotency_proven,
            "pretrade_enforced": self.pretrade_enforced,
            "kill_switch_with_airflow_down": self.kill_switch_with_airflow_down,
            "reconciliation_pre_intra_eod": self.reconciliation_pre_intra_eod,
        }
        return tuple(sorted(name for name, ok in flags.items() if not ok))

    @property
    def complete(self) -> bool:
        return not self.missing

    def to_record(self) -> dict[str, Any]:
        return {
            "record_type": str(RecordType.EXECUTION_READINESS),
            "at": self.at,
            "canary_flow": self.canary_flow,
            "service_outside_airflow": self.service_outside_airflow,
            "idempotency_proven": self.idempotency_proven,
            "pretrade_enforced": self.pretrade_enforced,
            "kill_switch_with_airflow_down": self.kill_switch_with_airflow_down,
            "reconciliation_pre_intra_eod": self.reconciliation_pre_intra_eod,
            "attested_by": self.attested_by,
        }


@dataclass(frozen=True, slots=True)
class LayerStatus:
    """Derived view: what the ledger says about one layer right now."""

    layer: MigrationLayer
    state: LayerState
    observations: int
    green_streak: int
    green_since: datetime | None
    last_observation_at: datetime | None
    green_days: float
    blockers: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class GateDecision:
    layer: MigrationLayer
    allowed: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.allowed and self.reasons:
            raise StranglerContractError("an allowed decision must carry no blocking reasons")
        if not self.allowed and not self.reasons:
            raise StranglerContractError("a blocked decision must say why")


def parse_layer(value: str | MigrationLayer) -> MigrationLayer:
    if isinstance(value, MigrationLayer):
        return value
    try:
        return MigrationLayer(str(value).strip().lower())
    except ValueError as exc:
        raise StranglerContractError(
            f"unknown layer {value!r}; expected one of {[str(x) for x in LAYER_ORDER]}"
        ) from exc


def parse_enum(enum_cls: type[StrEnum], value: Any, what: str) -> Any:
    try:
        return enum_cls(str(value))
    except ValueError as exc:
        raise StranglerContractError(f"unknown {what}: {value!r}") from exc


def as_mapping(value: Any, what: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StranglerContractError(f"{what} must be a mapping, got {type(value).__name__}")
    return value
