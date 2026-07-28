"""Shared order/fill event contracts and idempotency (BL-21)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from src.identity.canonical import canonical_json_bytes


class ExecutionContractError(ValueError):
    pass


class ExecutionEnvironment(StrEnum):
    REPLAY = "replay"
    PAPER = "paper"
    CANARY = "canary"
    LIVE = "live"


class OrderStatus(StrEnum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    SUBMITTED = "SUBMITTED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCEL_PENDING = "CANCEL_PENDING"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    SIMULATED = "SIMULATED"
    EXPIRED = "EXPIRED"
    QUARANTINED = "QUARANTINED"


_SHA256 = re.compile(r"sha256:[0-9a-f]{64}")


def _aware(value: datetime, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ExecutionContractError(f"{field} must be timezone-aware")


def _positive_finite(value: Decimal, field: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ExecutionContractError(f"{field} must be a finite Decimal")
    if value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ExecutionContractError(f"{field} must be {qualifier}")


def order_idempotency_key(
    *,
    account_id: str,
    instrument: str,
    target_version: str,
    decision_fingerprint: str,
    environment: ExecutionEnvironment | str,
    rebalance_cutoff: datetime,
) -> str:
    if rebalance_cutoff.tzinfo is None or rebalance_cutoff.utcoffset() is None:
        raise ExecutionContractError("rebalance_cutoff must be timezone-aware")
    try:
        canonical_environment = ExecutionEnvironment(environment)
    except (TypeError, ValueError) as exc:
        raise ExecutionContractError(f"invalid execution environment {environment!r}") from exc
    payload = {
        "account_id": account_id,
        "instrument": instrument,
        "target_version": target_version,
        "decision_fingerprint": decision_fingerprint,
        "environment": canonical_environment.value,
        "rebalance_cutoff": rebalance_cutoff.astimezone(timezone.utc),
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class OrderHeader:
    order_id: str
    client_order_id: str
    idempotency_key: str
    account_id: str
    env: ExecutionEnvironment
    executor_type: str
    strategy_id: str
    sleeve_id: str
    target_version: str
    instrument: str
    currency: str
    side: str
    qty: Decimal
    order_type: str
    decision_fingerprint: str
    execution_fingerprint: str
    rebalance_cutoff: datetime
    submitted_at: datetime

    def __post_init__(self) -> None:
        _positive_finite(self.qty, "qty")
        if self.side not in {"BUY", "SELL"}:
            raise ExecutionContractError("side must be BUY or SELL")
        if self.executor_type not in {"deterministic_simulator", "broker"}:
            raise ExecutionContractError("executor_type is not registered")
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ExecutionContractError("currency is required")
        if _SHA256.fullmatch(self.idempotency_key) is None:
            raise ExecutionContractError("idempotency_key must be a sha256 fingerprint")
        if _SHA256.fullmatch(self.decision_fingerprint) is None:
            raise ExecutionContractError("decision_fingerprint must be sha256")
        if _SHA256.fullmatch(self.execution_fingerprint) is None:
            raise ExecutionContractError("execution_fingerprint must be sha256")
        _aware(self.rebalance_cutoff, "rebalance_cutoff")
        _aware(self.submitted_at, "submitted_at")


@dataclass(frozen=True, slots=True)
class OrderStatusEvent:
    event_id: str
    order_id: str
    event_time: datetime
    status: OrderStatus
    actor: str
    reason_code: str | None = None
    broker_order_id: str | None = None

    def __post_init__(self) -> None:
        _aware(self.event_time, "event_time")
        if not self.event_id or not self.order_id or not self.actor:
            raise ExecutionContractError("status event identity and actor are required")


@dataclass(frozen=True, slots=True)
class FillEvent:
    fill_id: str
    order_id: str
    fill_time: datetime
    qty: Decimal
    price: Decimal
    commission: Decimal = Decimal("0")
    venue: str | None = None
    broker_fill_id: str = ""
    currency: str | None = None

    def __post_init__(self) -> None:
        _aware(self.fill_time, "fill_time")
        _positive_finite(self.qty, "qty")
        _positive_finite(self.price, "price")
        _positive_finite(self.commission, "commission", allow_zero=True)
        if not self.fill_id or not self.order_id:
            raise ExecutionContractError("fill identity is required")
        if not isinstance(self.broker_fill_id, str) or not self.broker_fill_id.strip():
            raise ExecutionContractError(
                "broker_fill_id is required; simulators must supply a synthetic stable id"
            )

    @property
    def fill_fingerprint(self) -> str:
        payload = {
            "order_id": self.order_id,
            "broker_fill_id": self.broker_fill_id,
            "fill_time": self.fill_time.astimezone(timezone.utc),
            "qty": self.qty,
            "price": self.price,
            "venue": self.venue,
        }
        return "sha256:" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class FillCorrectionEvent:
    correction_id: str
    fill_id: str
    event_time: datetime
    field: str
    old_value: str | None
    new_value: str | None
    reason: str
    actor: str

    ALLOWED_FIELDS = frozenset(
        {"qty", "price", "commission", "venue", "broker_fill_id", "currency"}
    )

    def __post_init__(self) -> None:
        _aware(self.event_time, "event_time")
        if self.field not in self.ALLOWED_FIELDS:
            raise ExecutionContractError(f"uncorrectable fill field {self.field!r}")
        if not self.correction_id or not self.fill_id or not self.reason or not self.actor:
            raise ExecutionContractError("correction identity, reason and actor are required")
