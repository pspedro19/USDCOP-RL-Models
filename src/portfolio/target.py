"""Canonical aggregate portfolio target consumed by execution (BL-26/27/30)."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from src.execution.events import ExecutionEnvironment
from src.identity.canonical import semantic_hash


class TargetError(ValueError):
    pass


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TargetError("target mapping keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        raise TargetError("target collections must be ordered")
    return value


def _aware(value: datetime, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TargetError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True, slots=True)
class TargetExposure:
    allocation_id: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    instrument: str
    side: str
    risk_budget: Decimal
    target_weight: Decimal
    currency: str

    def __post_init__(self) -> None:
        for name in (
            "allocation_id",
            "strategy_id",
            "sleeve_id",
            "instrument_id",
            "instrument",
            "currency",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TargetError(f"exposure {name} must be a non-empty string")
        if self.side not in {"LONG", "SHORT", "FLAT"}:
            raise TargetError("exposure side must be LONG, SHORT or FLAT")
        for name in ("risk_budget", "target_weight"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise TargetError(f"exposure {name} must be a finite Decimal")
        if self.risk_budget < 0 or abs(self.target_weight) != self.risk_budget:
            raise TargetError("exposure target_weight must equal signed risk_budget")
        expected_side = (
            "LONG"
            if self.target_weight > 0
            else "SHORT"
            if self.target_weight < 0
            else "FLAT"
        )
        if self.side != expected_side:
            raise TargetError("exposure side contradicts target_weight")

    def canonical_payload(self) -> Mapping[str, Any]:
        return {
            "allocation_id": self.allocation_id,
            "strategy_id": self.strategy_id,
            "sleeve_id": self.sleeve_id,
            "instrument_id": self.instrument_id,
            "instrument": self.instrument,
            "side": self.side,
            "risk_budget": self.risk_budget,
            "target_weight": self.target_weight,
            "currency": self.currency,
        }


@dataclass(frozen=True, slots=True)
class PortfolioTarget:
    target_id: str
    target_version: str
    snapshot_id: str
    account_id: str
    environment: ExecutionEnvironment
    allocator_version: str
    valid_from: datetime
    valid_until: datetime
    rebalance_cutoff: datetime
    decision_fingerprint: str
    constraints_snapshot: Mapping[str, Any]
    infeasibility_fallback: int | None
    fallback_incident_id: str | None
    exposures: tuple[TargetExposure, ...]
    semantic_hash: str

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "target_version",
            "snapshot_id",
            "account_id",
            "allocator_version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TargetError(f"{name} must be a non-empty string")
        try:
            environment = ExecutionEnvironment(self.environment)
        except (TypeError, ValueError) as exc:
            raise TargetError(f"invalid execution environment {self.environment!r}") from exc
        object.__setattr__(self, "environment", environment)
        for name in ("valid_from", "valid_until", "rebalance_cutoff"):
            object.__setattr__(self, name, _aware(getattr(self, name), name))
        if self.valid_until <= self.valid_from:
            raise TargetError("target valid_until must be after valid_from")
        if self.rebalance_cutoff > self.valid_from:
            raise TargetError("target rebalance_cutoff cannot exceed valid_from")
        if _SHA256.fullmatch(self.decision_fingerprint) is None:
            raise TargetError("decision_fingerprint must be sha256")
        if _SHA256.fullmatch(self.semantic_hash) is None:
            raise TargetError("semantic_hash must be sha256")
        if not isinstance(self.constraints_snapshot, Mapping) or not self.constraints_snapshot:
            raise TargetError("constraints_snapshot must be a non-empty mapping")
        object.__setattr__(
            self, "constraints_snapshot", _freeze(self.constraints_snapshot)
        )
        if not self.exposures or any(
            not isinstance(item, TargetExposure) for item in self.exposures
        ):
            raise TargetError("target requires TargetExposure records")
        sleeves = [item.sleeve_id for item in self.exposures]
        if len(sleeves) != len(set(sleeves)):
            raise TargetError("duplicate sleeve in target exposures")
        if self.infeasibility_fallback is None:
            if self.fallback_incident_id is not None:
                raise TargetError("fallback incident requires fallback level")
        elif (
            type(self.infeasibility_fallback) is not int
            or self.infeasibility_fallback not in {1, 2, 3, 4}
            or not isinstance(self.fallback_incident_id, str)
            or not self.fallback_incident_id.strip()
        ):
            raise TargetError("fallback level 1..4 requires an incident id")
        identity_payload = {
            "target_version": self.target_version,
            "snapshot_id": self.snapshot_id,
            "account_id": self.account_id,
            "environment": self.environment.value,
            "allocator_version": self.allocator_version,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "rebalance_cutoff": self.rebalance_cutoff,
            "decision_fingerprint": self.decision_fingerprint,
            "constraints_snapshot": self.constraints_snapshot,
            "infeasibility_fallback": self.infeasibility_fallback,
            "fallback_incident_id": self.fallback_incident_id,
            "exposures": [item.canonical_payload() for item in self.exposures],
        }
        expected_hash = semantic_hash(identity_payload)
        if self.semantic_hash != expected_hash:
            raise TargetError("semantic_hash does not match canonical target content")
        expected_id = str(uuid.uuid5(uuid.NAMESPACE_URL, expected_hash))
        if self.target_id != expected_id:
            raise TargetError("target_id does not match canonical target identity")


@dataclass(frozen=True, slots=True)
class ExecutableExposure:
    """Read-only projection combining one aggregate header and one exposure."""

    target: PortfolioTarget
    exposure: TargetExposure

    @property
    def target_id(self) -> str:
        return self.target.target_id

    @property
    def target_version(self) -> str:
        return self.target.target_version

    @property
    def snapshot_id(self) -> str:
        return self.target.snapshot_id

    @property
    def allocation_id(self) -> str:
        return self.exposure.allocation_id

    @property
    def account_id(self) -> str:
        return self.target.account_id

    @property
    def strategy_id(self) -> str:
        return self.exposure.strategy_id

    @property
    def sleeve_id(self) -> str:
        return self.exposure.sleeve_id

    @property
    def instrument_id(self) -> str:
        return self.exposure.instrument_id

    @property
    def instrument(self) -> str:
        return self.exposure.instrument

    @property
    def settlement_currency(self) -> str:
        return self.exposure.currency

    @property
    def target_weight(self) -> Decimal:
        return self.exposure.target_weight

    @property
    def environment(self) -> ExecutionEnvironment:
        return self.target.environment

    @property
    def valid_from(self) -> datetime:
        return self.target.valid_from

    @property
    def valid_until(self) -> datetime:
        return self.target.valid_until

    @property
    def rebalance_cutoff(self) -> datetime:
        return self.target.rebalance_cutoff

    @property
    def decision_fingerprint(self) -> str:
        return self.target.decision_fingerprint


class TargetBuilder:
    def build(
        self,
        *,
        target_version: str,
        snapshot_id: str,
        account_id: str,
        environment: ExecutionEnvironment | str,
        allocator_version: str,
        valid_from: datetime,
        valid_until: datetime,
        rebalance_cutoff: datetime,
        decision_fingerprint: str,
        constraints_snapshot: Mapping[str, Any],
        exposures: Sequence[TargetExposure],
        infeasibility_fallback: int | None = None,
        fallback_incident_id: str | None = None,
    ) -> PortfolioTarget:
        ordered = tuple(
            sorted(
                exposures,
                key=lambda item: (
                    item.sleeve_id,
                    item.instrument_id,
                    item.allocation_id,
                ),
            )
        )
        payload = {
            "target_version": target_version,
            "snapshot_id": snapshot_id,
            "account_id": account_id,
            "environment": ExecutionEnvironment(environment).value,
            "allocator_version": allocator_version,
            "valid_from": _aware(valid_from, "valid_from"),
            "valid_until": _aware(valid_until, "valid_until"),
            "rebalance_cutoff": _aware(rebalance_cutoff, "rebalance_cutoff"),
            "decision_fingerprint": decision_fingerprint,
            "constraints_snapshot": constraints_snapshot,
            "infeasibility_fallback": infeasibility_fallback,
            "fallback_incident_id": fallback_incident_id,
            "exposures": [item.canonical_payload() for item in ordered],
        }
        digest = semantic_hash(payload)
        return PortfolioTarget(
            target_id=str(uuid.uuid5(uuid.NAMESPACE_URL, digest)),
            semantic_hash=digest,
            exposures=ordered,
            target_version=target_version,
            snapshot_id=snapshot_id,
            account_id=account_id,
            environment=ExecutionEnvironment(environment),
            allocator_version=allocator_version,
            valid_from=valid_from,
            valid_until=valid_until,
            rebalance_cutoff=rebalance_cutoff,
            decision_fingerprint=decision_fingerprint,
            constraints_snapshot=constraints_snapshot,
            infeasibility_fallback=infeasibility_fallback,
            fallback_incident_id=fallback_incident_id,
        )


__all__ = [
    "ExecutableExposure",
    "PortfolioTarget",
    "TargetBuilder",
    "TargetError",
    "TargetExposure",
]
