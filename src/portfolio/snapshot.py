"""Explicit cutoff barrier for book construction (BL-26)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Any, Mapping
from types import MappingProxyType


class SnapshotError(ValueError):
    pass


class MissingPolicy(StrEnum):
    FLAT = "FLAT"
    KEEP_POSITION_UNTIL_EXPIRY = "KEEP_POSITION_UNTIL_EXPIRY"
    EXIT_ONLY = "EXIT_ONLY"
    USE_LAST_VALID_WITH_MAX_AGE = "USE_LAST_VALID_WITH_MAX_AGE"


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise SnapshotError("signal payload keys must be strings")
        return MappingProxyType(
            {key: _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise SnapshotError("signal payload must use ordered JSON collections")
    return value


@dataclass(frozen=True, slots=True)
class AcceptedSignal:
    signal_id: str
    sleeve_id: str
    as_of: datetime
    available_at: datetime
    valid_until: datetime
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("signal_id", "sleeve_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SnapshotError(f"{name} must be a non-empty string")
        for name in ("as_of", "available_at", "valid_until"):
            value = getattr(self, name)
            if not isinstance(value, datetime):
                raise SnapshotError(f"{self.signal_id}: {name} must be a datetime")
            if value.tzinfo is None or value.utcoffset() is None:
                raise SnapshotError(
                    f"{self.signal_id}: {name} must be timezone-aware"
                )
        if self.available_at < self.as_of:
            raise SnapshotError(
                f"{self.signal_id}: available_at cannot precede as_of"
            )
        if self.valid_until < self.as_of:
            raise SnapshotError(
                f"{self.signal_id}: valid_until cannot precede as_of"
            )
        if not isinstance(self.payload, Mapping):
            raise SnapshotError(f"{self.signal_id}: payload must be a mapping")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))


@dataclass(frozen=True, slots=True)
class PositionState:
    """Position evidence required by KEEP/EXIT fallbacks."""

    sleeve_id: str
    instrument_id: str
    currency: str
    as_of: datetime
    available_at: datetime
    valid_until: datetime
    signed_weight: Decimal

    def __post_init__(self) -> None:
        for name in ("sleeve_id", "instrument_id", "currency"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise SnapshotError(f"position {name} must be a non-empty string")
        for name in ("as_of", "available_at", "valid_until"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise SnapshotError(f"position {name} must be timezone-aware")
        if self.available_at < self.as_of or self.valid_until < self.as_of:
            raise SnapshotError("position timestamps are inconsistent")
        if not isinstance(self.signed_weight, Decimal) or not self.signed_weight.is_finite():
            raise SnapshotError("position signed_weight must be a finite Decimal")


@dataclass(frozen=True, slots=True)
class MaterializedInput:
    """The exact per-sleeve input consumed by the allocator."""

    sleeve_id: str
    resolution: str | MissingPolicy
    source_signal_id: str | None
    as_of: datetime | None
    available_at: datetime | None
    valid_until: datetime | None
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.sleeve_id, str) or not self.sleeve_id.strip():
            raise SnapshotError("materialized sleeve_id is required")
        if self.resolution != "ACCEPTED":
            try:
                MissingPolicy(self.resolution)
            except (TypeError, ValueError) as exc:
                raise SnapshotError(f"invalid snapshot resolution {self.resolution!r}") from exc
        for name in ("as_of", "available_at", "valid_until"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() is None
            ):
                raise SnapshotError(f"materialized {name} must be timezone-aware")
        if not isinstance(self.payload, Mapping):
            raise SnapshotError("materialized payload must be a mapping")
        object.__setattr__(self, "payload", _deep_freeze(self.payload))


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    snapshot_id: str
    cutoff_time: datetime
    required_sleeves: tuple[str, ...]
    accepted_signals: tuple[AcceptedSignal, ...]
    stale_signals: tuple[str, ...]
    missing_signals: tuple[str, ...]
    fallback_applied: Mapping[str, str]
    max_age_by_sleeve: Mapping[str, timedelta]
    missing_policy_by_sleeve: Mapping[str, MissingPolicy]
    materialized_inputs: Mapping[str, MaterializedInput]
    semantic_hash: str

    def __post_init__(self) -> None:
        from src.identity.canonical import semantic_hash

        if (
            not isinstance(self.cutoff_time, datetime)
            or self.cutoff_time.tzinfo is None
            or self.cutoff_time.utcoffset() is None
        ):
            raise SnapshotError("cutoff_time must be timezone-aware")
        if (
            not self.required_sleeves
            or tuple(sorted(self.required_sleeves)) != self.required_sleeves
            or len(set(self.required_sleeves)) != len(self.required_sleeves)
        ):
            raise SnapshotError("required_sleeves must be sorted, non-empty and unique")
        required = set(self.required_sleeves)
        if set(self.max_age_by_sleeve) != required:
            raise SnapshotError("max_age_by_sleeve keys must exactly match required_sleeves")
        if set(self.missing_policy_by_sleeve) != required:
            raise SnapshotError(
                "missing_policy_by_sleeve keys must exactly match required_sleeves"
            )
        if set(self.materialized_inputs) != required:
            raise SnapshotError(
                "materialized_inputs keys must exactly match required_sleeves"
            )
        if not set(self.fallback_applied).issubset(required):
            raise SnapshotError("fallback_applied contains an undeclared sleeve")
        for sleeve in self.required_sleeves:
            max_age = self.max_age_by_sleeve[sleeve]
            if not isinstance(max_age, timedelta) or max_age <= timedelta(0):
                raise SnapshotError(f"{sleeve}: positive max_age is required")
            try:
                MissingPolicy(self.missing_policy_by_sleeve[sleeve])
            except (TypeError, ValueError) as exc:
                raise SnapshotError(f"{sleeve}: invalid missing policy") from exc
            materialized = self.materialized_inputs[sleeve]
            if (
                not isinstance(materialized, MaterializedInput)
                or materialized.sleeve_id != sleeve
            ):
                raise SnapshotError(f"{sleeve}: invalid materialized input")
        if any(not isinstance(signal, AcceptedSignal) for signal in self.accepted_signals):
            raise SnapshotError("accepted_signals must contain AcceptedSignal records")

        object.__setattr__(
            self, "cutoff_time", self.cutoff_time.astimezone(timezone.utc)
        )
        object.__setattr__(
            self, "fallback_applied", MappingProxyType(dict(self.fallback_applied))
        )
        object.__setattr__(
            self,
            "max_age_by_sleeve",
            MappingProxyType(dict(self.max_age_by_sleeve)),
        )
        object.__setattr__(
            self,
            "missing_policy_by_sleeve",
            MappingProxyType(
                {
                    sleeve: MissingPolicy(self.missing_policy_by_sleeve[sleeve])
                    for sleeve in self.required_sleeves
                }
            ),
        )
        object.__setattr__(
            self,
            "materialized_inputs",
            MappingProxyType(dict(self.materialized_inputs)),
        )
        self._validate_causal_contract()
        expected_hash = semantic_hash(self.identity_payload())
        if self.semantic_hash != expected_hash:
            raise SnapshotError("semantic_hash mismatch")
        expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, expected_hash))
        if self.snapshot_id != expected_id:
            raise SnapshotError("snapshot_id mismatch")

    def _validate_causal_contract(self) -> None:
        cutoff = self.cutoff_time
        required = set(self.required_sleeves)
        if (
            tuple(sorted(self.stale_signals)) != self.stale_signals
            or len(set(self.stale_signals)) != len(self.stale_signals)
            or any(
                not isinstance(signal_id, str) or not signal_id.strip()
                for signal_id in self.stale_signals
            )
        ):
            raise SnapshotError("stale_signals must be sorted, unique signal ids")
        if (
            tuple(sorted(self.missing_signals)) != self.missing_signals
            or len(set(self.missing_signals)) != len(self.missing_signals)
            or not set(self.missing_signals).issubset(required)
        ):
            raise SnapshotError(
                "missing_signals must be sorted, unique required sleeve ids"
            )

        accepted_by_sleeve: dict[str, AcceptedSignal] = {}
        accepted_ids: set[str] = set()
        for signal in self.accepted_signals:
            if signal.signal_id in accepted_ids:
                raise SnapshotError(f"duplicate accepted signal_id: {signal.signal_id}")
            if signal.sleeve_id not in required:
                raise SnapshotError(
                    f"{signal.signal_id}: accepted signal has undeclared sleeve"
                )
            if signal.sleeve_id in accepted_by_sleeve:
                raise SnapshotError(
                    f"{signal.sleeve_id}: more than one accepted signal"
                )
            if signal.as_of > cutoff:
                raise SnapshotError(f"{signal.signal_id}: as_of exceeds cutoff")
            if signal.available_at > cutoff:
                raise SnapshotError(f"{signal.signal_id}: available_at exceeds cutoff")
            if cutoff - signal.as_of > self.max_age_by_sleeve[signal.sleeve_id]:
                raise SnapshotError(f"{signal.signal_id}: accepted signal exceeds max_age")
            if signal.valid_until < cutoff:
                raise SnapshotError(
                    f"{signal.signal_id}: accepted signal expired before cutoff"
                )
            materialized = self.materialized_inputs[signal.sleeve_id]
            if materialized.resolution != "ACCEPTED":
                raise SnapshotError(
                    f"{signal.sleeve_id}: accepted signal has fallback materialization"
                )
            if (
                materialized.source_signal_id != signal.signal_id
                or materialized.as_of != signal.as_of
                or materialized.available_at != signal.available_at
                or materialized.valid_until != signal.valid_until
                or materialized.payload != signal.payload
            ):
                raise SnapshotError(
                    f"{signal.sleeve_id}: accepted signal and materialized input differ"
                )
            accepted_ids.add(signal.signal_id)
            accepted_by_sleeve[signal.sleeve_id] = signal

        if tuple(accepted_by_sleeve) != tuple(sorted(accepted_by_sleeve)):
            raise SnapshotError("accepted_signals must be ordered by sleeve_id")
        stale_ids = set(self.stale_signals)
        if accepted_ids & stale_ids:
            raise SnapshotError("a signal cannot be both accepted and stale")

        fallback_sleeves = required - set(accepted_by_sleeve)
        if set(self.fallback_applied) != fallback_sleeves:
            raise SnapshotError(
                "fallback_applied keys must exactly match non-accepted sleeves"
            )
        if not set(self.missing_signals).issubset(fallback_sleeves):
            raise SnapshotError("an accepted sleeve cannot also be missing")
        if (
            len(self.stale_signals) + len(self.missing_signals)
            != len(fallback_sleeves)
        ):
            raise SnapshotError(
                "accepted/stale/missing classifications must partition required_sleeves"
            )

        for sleeve in fallback_sleeves:
            materialized = self.materialized_inputs[sleeve]
            try:
                resolution = MissingPolicy(materialized.resolution)
                applied = MissingPolicy(self.fallback_applied[sleeve])
            except (TypeError, ValueError) as exc:
                raise SnapshotError(f"{sleeve}: invalid fallback resolution") from exc
            declared = self.missing_policy_by_sleeve[sleeve]
            if resolution is not declared or applied is not declared:
                raise SnapshotError(
                    f"{sleeve}: fallback resolution differs from declared policy"
                )
            if materialized.as_of is not None and materialized.as_of > cutoff:
                raise SnapshotError(f"{sleeve}: fallback as_of exceeds cutoff")
            if (
                materialized.available_at is not None
                and materialized.available_at > cutoff
            ):
                raise SnapshotError(f"{sleeve}: fallback available_at exceeds cutoff")
            if resolution is MissingPolicy.USE_LAST_VALID_WITH_MAX_AGE:
                if (
                    not materialized.source_signal_id
                    or materialized.as_of is None
                    or materialized.available_at is None
                    or materialized.valid_until is None
                ):
                    raise SnapshotError(
                        f"{sleeve}: last-valid fallback lacks signal evidence"
                    )
                if materialized.valid_until < cutoff:
                    raise SnapshotError(
                        f"{sleeve}: last-valid fallback expired before cutoff"
                    )
                if cutoff - materialized.as_of > self.max_age_by_sleeve[sleeve]:
                    raise SnapshotError(
                        f"{sleeve}: last-valid fallback exceeds max_age"
                    )
            elif (
                resolution is MissingPolicy.KEEP_POSITION_UNTIL_EXPIRY
                and (
                    materialized.valid_until is None
                    or materialized.valid_until < cutoff
                )
            ):
                raise SnapshotError(
                    f"{sleeve}: kept position expired before cutoff"
                )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "cutoff_time": self.cutoff_time,
            "required_sleeves": list(self.required_sleeves),
            "accepted_signals": [
                {
                    "signal_id": item.signal_id,
                    "sleeve_id": item.sleeve_id,
                    "as_of": item.as_of,
                    "available_at": item.available_at,
                    "valid_until": item.valid_until,
                    "payload": item.payload,
                }
                for item in self.accepted_signals
            ],
            "stale_signals": list(self.stale_signals),
            "missing_signals": list(self.missing_signals),
            "fallback_applied": dict(self.fallback_applied),
            "max_age_by_sleeve_seconds": {
                sleeve: str(self.max_age_by_sleeve[sleeve].total_seconds())
                for sleeve in self.required_sleeves
            },
            "missing_policy_by_sleeve": {
                sleeve: MissingPolicy(
                    self.missing_policy_by_sleeve[sleeve]
                ).value
                for sleeve in self.required_sleeves
            },
            "materialized_inputs": {
                sleeve: {
                    "resolution": str(self.materialized_inputs[sleeve].resolution),
                    "source_signal_id": self.materialized_inputs[
                        sleeve
                    ].source_signal_id,
                    "as_of": self.materialized_inputs[sleeve].as_of,
                    "available_at": self.materialized_inputs[sleeve].available_at,
                    "valid_until": self.materialized_inputs[sleeve].valid_until,
                    "payload": self.materialized_inputs[sleeve].payload,
                }
                for sleeve in self.required_sleeves
            },
        }


class SnapshotBuilder:
    def build(
        self,
        *,
        cutoff_time: datetime,
        required_sleeves: list[str],
        signals: list[AcceptedSignal],
        max_age_by_sleeve: Mapping[str, timedelta],
        missing_policy_by_sleeve: Mapping[str, MissingPolicy],
        position_by_sleeve: Mapping[str, PositionState] | None = None,
        last_valid_signal_by_sleeve: Mapping[str, AcceptedSignal] | None = None,
    ) -> PortfolioSnapshot:
        from src.identity.canonical import semantic_hash

        if cutoff_time.tzinfo is None or cutoff_time.utcoffset() is None:
            raise SnapshotError("cutoff_time must be timezone-aware")
        if not required_sleeves or len(set(required_sleeves)) != len(required_sleeves):
            raise SnapshotError("required_sleeves must be non-empty and unique")
        if any(
            not isinstance(sleeve, str) or not sleeve.strip()
            for sleeve in required_sleeves
        ):
            raise SnapshotError("required_sleeves entries must be non-empty strings")
        required = tuple(sorted(required_sleeves))
        required_set = set(required)
        if set(max_age_by_sleeve) != required_set:
            raise SnapshotError("max_age_by_sleeve keys must exactly match required_sleeves")
        if set(missing_policy_by_sleeve) != required_set:
            raise SnapshotError(
                "missing_policy_by_sleeve keys must exactly match required_sleeves"
            )
        positions = dict(position_by_sleeve or {})
        last_valid = dict(last_valid_signal_by_sleeve or {})
        if not set(positions).issubset(required_set):
            raise SnapshotError("position_by_sleeve contains an undeclared sleeve")
        if not set(last_valid).issubset(required_set):
            raise SnapshotError("last_valid_signal_by_sleeve contains an undeclared sleeve")
        cutoff = cutoff_time.astimezone(timezone.utc)
        accepted: list[AcceptedSignal] = []
        stale: list[str] = []
        missing: list[str] = []
        fallbacks: dict[str, str] = {}
        materialized: dict[str, MaterializedInput] = {}
        by_sleeve: dict[str, list[AcceptedSignal]] = {}
        seen_signal_ids: set[str] = set()
        for signal in signals:
            if not isinstance(signal, AcceptedSignal):
                raise SnapshotError("signals must contain AcceptedSignal records")
            if signal.signal_id in seen_signal_ids:
                raise SnapshotError(f"duplicate signal_id: {signal.signal_id}")
            seen_signal_ids.add(signal.signal_id)
            if signal.sleeve_id not in required_set:
                raise SnapshotError(
                    f"{signal.signal_id}: undeclared sleeve {signal.sleeve_id}"
                )
            if signal.as_of > cutoff or signal.available_at > cutoff:
                continue
            by_sleeve.setdefault(signal.sleeve_id, []).append(signal)
        for sleeve in required:
            max_age = max_age_by_sleeve.get(sleeve)
            policy = missing_policy_by_sleeve.get(sleeve)
            if max_age is None or max_age <= timedelta(0):
                raise SnapshotError(f"{sleeve}: positive max_age is required")
            if policy is None:
                raise SnapshotError(f"{sleeve}: missing-signal policy is required")
            candidates = sorted(
                by_sleeve.get(sleeve, []),
                key=lambda item: (item.as_of, item.available_at, item.signal_id),
            )
            signal = candidates[-1] if candidates else None
            if signal is None:
                missing.append(sleeve)
                fallbacks[sleeve] = policy.value
                self._materialize_fallback(
                    materialized=materialized,
                    sleeve=sleeve,
                    policy=policy,
                    cutoff=cutoff,
                    max_age=max_age,
                    position=positions.get(sleeve),
                    fallback_signal=last_valid.get(sleeve),
                )
                continue
            if signal.valid_until < cutoff or cutoff - signal.as_of > max_age:
                stale.append(signal.signal_id)
                fallbacks[sleeve] = policy.value
                fallback_signal = signal
                external_last = last_valid.get(sleeve)
                if (
                    external_last is not None
                    and external_last.as_of > fallback_signal.as_of
                ):
                    fallback_signal = external_last
                self._materialize_fallback(
                    materialized=materialized,
                    sleeve=sleeve,
                    policy=policy,
                    cutoff=cutoff,
                    max_age=max_age,
                    position=positions.get(sleeve),
                    fallback_signal=fallback_signal,
                )
                continue
            accepted.append(signal)
            materialized[sleeve] = MaterializedInput(
                sleeve_id=sleeve,
                resolution="ACCEPTED",
                source_signal_id=signal.signal_id,
                as_of=signal.as_of,
                available_at=signal.available_at,
                valid_until=signal.valid_until,
                payload=signal.payload,
            )
        payload = {
            "cutoff_time": cutoff,
            "required_sleeves": list(required),
            "accepted_signals": [
                {
                    "signal_id": item.signal_id,
                    "sleeve_id": item.sleeve_id,
                    "as_of": item.as_of,
                    "available_at": item.available_at,
                    "valid_until": item.valid_until,
                    "payload": item.payload,
                }
                for item in accepted
            ],
            "stale_signals": sorted(stale),
            "missing_signals": sorted(missing),
            "fallback_applied": fallbacks,
            "max_age_by_sleeve_seconds": {
                sleeve: str(max_age_by_sleeve[sleeve].total_seconds())
                for sleeve in required
            },
            "missing_policy_by_sleeve": {
                sleeve: missing_policy_by_sleeve[sleeve].value
                for sleeve in required
            },
            "materialized_inputs": {
                sleeve: {
                    "resolution": str(materialized[sleeve].resolution),
                    "source_signal_id": materialized[sleeve].source_signal_id,
                    "as_of": materialized[sleeve].as_of,
                    "available_at": materialized[sleeve].available_at,
                    "valid_until": materialized[sleeve].valid_until,
                    "payload": materialized[sleeve].payload,
                }
                for sleeve in required
            },
        }
        digest = semantic_hash(payload)
        return PortfolioSnapshot(
            snapshot_id=str(uuid.uuid5(uuid.NAMESPACE_URL, digest)),
            cutoff_time=cutoff,
            required_sleeves=required,
            accepted_signals=tuple(accepted),
            stale_signals=tuple(sorted(stale)),
            missing_signals=tuple(sorted(missing)),
            fallback_applied=MappingProxyType(dict(fallbacks)),
            max_age_by_sleeve=MappingProxyType(
                {sleeve: max_age_by_sleeve[sleeve] for sleeve in required}
            ),
            missing_policy_by_sleeve=MappingProxyType(
                {
                    sleeve: missing_policy_by_sleeve[sleeve]
                    for sleeve in required
                }
            ),
            materialized_inputs=MappingProxyType(
                {sleeve: materialized[sleeve] for sleeve in required}
            ),
            semantic_hash=digest,
        )

    @staticmethod
    def _materialize_fallback(
        *,
        materialized: dict[str, MaterializedInput],
        sleeve: str,
        policy: MissingPolicy,
        cutoff: datetime,
        max_age: timedelta,
        position: PositionState | None,
        fallback_signal: AcceptedSignal | None,
    ) -> None:
        if policy is MissingPolicy.FLAT:
            materialized[sleeve] = MaterializedInput(
                sleeve_id=sleeve,
                resolution=policy,
                source_signal_id=None,
                as_of=cutoff,
                available_at=cutoff,
                valid_until=cutoff,
                payload={
                    "instruction": "TARGET",
                    "side": 0,
                    "target_weight": Decimal("0"),
                    "reduce_only": True,
                },
            )
            return

        if policy is MissingPolicy.USE_LAST_VALID_WITH_MAX_AGE:
            if fallback_signal is None:
                raise SnapshotError(f"{sleeve}: last valid signal is required")
            if fallback_signal.sleeve_id != sleeve:
                raise SnapshotError(f"{sleeve}: last valid signal belongs to another sleeve")
            if (
                fallback_signal.as_of > cutoff
                or fallback_signal.available_at > cutoff
                or fallback_signal.valid_until < cutoff
                or cutoff - fallback_signal.as_of > max_age
            ):
                raise SnapshotError(f"{sleeve}: last valid signal exceeds cutoff/max_age")
            materialized[sleeve] = MaterializedInput(
                sleeve_id=sleeve,
                resolution=policy,
                source_signal_id=fallback_signal.signal_id,
                as_of=fallback_signal.as_of,
                available_at=fallback_signal.available_at,
                valid_until=fallback_signal.valid_until,
                payload=fallback_signal.payload,
            )
            return

        if position is None:
            raise SnapshotError(f"{sleeve}: position state is required for {policy.value}")
        if position.sleeve_id != sleeve:
            raise SnapshotError(f"{sleeve}: position state belongs to another sleeve")
        if position.available_at > cutoff or position.as_of > cutoff:
            raise SnapshotError(f"{sleeve}: position state was unavailable at cutoff")
        if policy is MissingPolicy.KEEP_POSITION_UNTIL_EXPIRY:
            if position.valid_until < cutoff:
                raise SnapshotError(f"{sleeve}: position state has expired")
            target_weight = position.signed_weight
            valid_until = position.valid_until
            instruction = "KEEP_POSITION_UNTIL_EXPIRY"
        else:
            target_weight = Decimal("0")
            valid_until = cutoff
            instruction = "EXIT_ONLY"
        materialized[sleeve] = MaterializedInput(
            sleeve_id=sleeve,
            resolution=policy,
            source_signal_id=None,
            as_of=position.as_of,
            available_at=position.available_at,
            valid_until=valid_until,
            payload={
                "instruction": instruction,
                "instrument_id": position.instrument_id,
                "currency": position.currency,
                "side": (
                    1
                    if target_weight > 0
                    else -1
                    if target_weight < 0
                    else 0
                ),
                "target_weight": target_weight,
                "reduce_only": policy is MissingPolicy.EXIT_ONLY,
            },
        )
