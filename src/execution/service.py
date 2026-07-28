"""Airflow-independent execution coordinator (BL-30).

The only accepted economic input is an approved immutable portfolio target.
Signals and forecasts are intentionally absent from every public interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import IntEnum, StrEnum
from typing import Any, Mapping, Protocol, Sequence

from src.execution.events import ExecutionEnvironment, order_idempotency_key
from src.identity.fingerprints import execution_fingerprint
from src.portfolio.target import ExecutableExposure, PortfolioTarget


class ExecutionRejected(RuntimeError):
    pass


class BrokerRejected(RuntimeError):
    """The venue explicitly and authoritatively rejected the order."""


class BrokerSubmissionUnknown(RuntimeError):
    """The transport failed and venue acceptance is unknown."""


class KillSwitchActionUnknown(RuntimeError):
    """A kill-switch broker effect may have happened and needs reconciliation."""


class KillSwitchLevel(IntEnum):
    CLEAR = 0
    BLOCK_NEW = 1
    CANCEL_OPEN = 2
    EXIT_ALL = 3
    ACCOUNT_FREEZE = 4


class ReconciliationStatus(StrEnum):
    RECONCILED = "RECONCILED"
    MISMATCH = "MISMATCH"
    QUARANTINED = "QUARANTINED"


@dataclass(frozen=True)
class AccountState:
    nav: Decimal
    current_qty: Decimal
    mark_price: Decimal
    currency: str
    broker_position_qty: Decimal
    open_order_ids: tuple[str, ...] = ()
    daily_loss_fraction: Decimal = Decimal("0")
    drawdown_fraction: Decimal = Decimal("0")
    gross_exposure_fraction: Decimal = Decimal("0")
    net_exposure_fraction: Decimal = Decimal("0")
    leverage: Decimal = Decimal("0")
    available_cash: Decimal = Decimal("0")
    cash_currency: str | None = None
    daily_trade_count: int = 0


@dataclass(frozen=True)
class RiskLimits:
    max_order_notional: Decimal
    max_position_notional: Decimal
    max_gross_exposure_fraction: Decimal
    max_daily_loss_fraction: Decimal
    allowed_instruments: frozenset[str]
    currency: str
    max_daily_trades: int
    min_order_notional: Decimal = Decimal("0")
    max_net_exposure_fraction: Decimal = Decimal("1")
    max_leverage: Decimal = Decimal("1")
    max_drawdown_fraction: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        for name in (
            "max_order_notional",
            "max_position_notional",
            "max_gross_exposure_fraction",
            "max_daily_loss_fraction",
            "min_order_notional",
            "max_net_exposure_fraction",
            "max_leverage",
            "max_drawdown_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
                raise ExecutionRejected(
                    f"{name.upper()}_MUST_BE_NON_NEGATIVE_FINITE_DECIMAL"
                )
        if not isinstance(self.currency, str) or not self.currency.strip():
            raise ExecutionRejected("RISK_LIMIT_CURRENCY_REQUIRED")
        if type(self.max_daily_trades) is not int or self.max_daily_trades <= 0:
            raise ExecutionRejected("MAX_DAILY_TRADES_MUST_BE_POSITIVE_INTEGER")


@dataclass(frozen=True)
class ExecutionControls:
    health_nominal: bool
    operational_nominal: bool
    price_within_collar: bool
    liquidity_sufficient: bool
    market_session_open: bool
    currency_settled: bool
    account_enabled: bool
    user_trading_enabled: bool
    trading_mode: ExecutionEnvironment

    def __post_init__(self) -> None:
        try:
            mode = ExecutionEnvironment(self.trading_mode)
        except (TypeError, ValueError) as exc:
            raise ExecutionRejected(f"INVALID_TRADING_MODE:{self.trading_mode!r}") from exc
        object.__setattr__(self, "trading_mode", mode)


@dataclass(frozen=True)
class EffectiveKillSwitch:
    event_id: str
    level: KillSwitchLevel


@dataclass(frozen=True)
class Reconciliation:
    reconciliation_id: str
    status: ReconciliationStatus
    internal_qty: Decimal
    broker_qty: Decimal
    observed_at: datetime
    details: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderIntent:
    target_id: str
    allocation_id: str
    account_id: str
    strategy_id: str
    sleeve_id: str
    instrument_id: str
    instrument: str
    currency: str
    side: str
    quantity: Decimal
    notional: Decimal
    target_version: str
    decision_fingerprint: str
    execution_fingerprint: str
    idempotency_key: str
    broker_id: str
    environment: str
    rebalance_cutoff: datetime
    reconciliation_id: str
    reduce_only: bool


@dataclass(frozen=True)
class PreTradeDecision:
    allowed: bool
    checks: Mapping[str, bool]
    reason_codes: tuple[str, ...]
    intent: OrderIntent | None
    kill_switch_level: KillSwitchLevel
    reconciliation_id: str


class TargetRepository(Protocol):
    async def get_target(self, target_id: str) -> PortfolioTarget | None: ...


class AccountRepository(Protocol):
    async def state_for(self, target: ExecutableExposure) -> AccountState: ...


class RiskRepository(Protocol):
    async def limits_for(self, target: ExecutableExposure) -> RiskLimits: ...


class KillSwitchRepository(Protocol):
    async def effective_state(self, account_id: str, at: datetime) -> EffectiveKillSwitch: ...


class ControlRepository(Protocol):
    async def for_target(
        self, target: ExecutableExposure, at: datetime
    ) -> ExecutionControls: ...


class Reconciler(Protocol):
    async def pre_operation(
        self, target: ExecutableExposure, account: AccountState
    ) -> Reconciliation: ...

    async def intraday(self, account_id: str) -> Reconciliation: ...

    async def end_of_day(self, account_id: str) -> Reconciliation: ...


class EventLedger(Protocol):
    async def append_pretrade(
        self, target: ExecutableExposure, decision: PreTradeDecision
    ) -> None: ...

    async def claim_order(
        self, intent: OrderIntent
    ) -> tuple[Mapping[str, Any], bool]:
        """Atomically insert by idempotency key and return ``(header, inserted)``."""
        ...

    async def claim_order_dispatch(
        self,
        *,
        order_id: str,
        execution_fingerprint: str,
        claimed_at: datetime,
        lease_seconds: int,
    ) -> str | None:
        """Fence one dispatch attempt; return ``None`` when complete/in-flight."""
        ...

    async def finalize_order_dispatch(
        self,
        *,
        order_id: str,
        claim_token: str,
        event_time: datetime,
        status: str,
        reason_code: str | None = None,
        broker_order_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Atomically append the terminal dispatch status and complete the claim."""
        ...

    async def require_order_dispatch_reconciliation(
        self,
        *,
        order_id: str,
        claim_token: str,
        event_time: datetime,
        reason_code: str,
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Atomically append SUBMIT_UNKNOWN and fence the uncertain attempt."""
        ...

    async def claim_kill_switch_action(
        self,
        *,
        action_key: str,
        kill_switch_event_id: str,
        account_id: str,
        action: str,
        claimed_at: datetime,
        lease_seconds: int,
    ) -> str | None:
        """Return a fencing token for a new/retryable/expired claim, else ``None``.

        A transport failure must transition the claim to
        ``RECONCILIATION_REQUIRED`` so the stable broker idempotency key can be
        retried.  A worker crash leaves ``CLAIMED`` until its lease expires.
        """
        ...

    async def complete_kill_switch_action(
        self,
        *,
        action_key: str,
        claim_token: str,
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Fence and mark the exact claimed attempt ``COMPLETED``."""
        ...

    async def require_kill_switch_reconciliation(
        self,
        *,
        action_key: str,
        claim_token: str,
        reason_code: str,
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """Fence an uncertain attempt as retryable after reconciliation."""
        ...

    async def append_status(
        self,
        order_id: str,
        status: str,
        reason_code: str | None = None,
        broker_order_id: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None: ...


class Broker(Protocol):
    async def submit(self, intent: OrderIntent) -> Mapping[str, Any]: ...

    async def cancel_open(
        self, account_id: str, *, idempotency_key: str
    ) -> Sequence[str]: ...

    async def exit_all(
        self, account_id: str, *, idempotency_key: str
    ) -> Sequence[Mapping[str, Any]]: ...


class ExecutionService:
    """Coordinates one target without any dependency on Airflow availability."""

    def __init__(
        self,
        *,
        targets: TargetRepository,
        accounts: AccountRepository,
        risks: RiskRepository,
        kill_switch: KillSwitchRepository,
        reconciler: Reconciler,
        ledger: EventLedger,
        broker: Broker,
        broker_id: str,
        order_policy_hash: str,
        controls: ControlRepository | None = None,
    ):
        self.targets = targets
        self.accounts = accounts
        self.risks = risks
        self.kill_switch = kill_switch
        self.reconciler = reconciler
        self.ledger = ledger
        self.broker = broker
        self.broker_id = broker_id
        self.order_policy_hash = order_policy_hash
        self.controls = controls

    async def execute_target(
        self, target_id: str, *, now: datetime | None = None
    ) -> Mapping[str, Any]:
        supplied_now = now or datetime.now(timezone.utc)
        if supplied_now.tzinfo is None or supplied_now.utcoffset() is None:
            raise ExecutionRejected("NOW_MUST_BE_TIMEZONE_AWARE")
        at = supplied_now.astimezone(timezone.utc)
        target = await self.targets.get_target(target_id)
        if target is None:
            raise ExecutionRejected("TARGET_NOT_FOUND")
        if not isinstance(target, PortfolioTarget):
            raise ExecutionRejected("INVALID_TARGET_CONTRACT")
        if not target.valid_from <= at <= target.valid_until:
            raise ExecutionRejected("TARGET_OUTSIDE_VALIDITY_WINDOW")

        switch = await self.kill_switch.effective_state(target.account_id, at)
        if not isinstance(switch, EffectiveKillSwitch) or not switch.event_id:
            raise ExecutionRejected("INVALID_KILL_SWITCH_STATE")
        await self._enforce_switch_side_effects(target.account_id, switch, now=at)

        prepared: list[tuple[ExecutableExposure, PreTradeDecision]] = []
        for exposure in target.exposures:
            executable = ExecutableExposure(target=target, exposure=exposure)
            account = await self.accounts.state_for(executable)
            reconciliation = await self.reconciler.pre_operation(executable, account)
            limits = await self.risks.limits_for(executable)
            controls = (
                await self.controls.for_target(executable, at)
                if self.controls is not None
                else ExecutionControls(
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    ExecutionEnvironment.PAPER,
                )
            )
            decision = self._pretrade(
                target=executable,
                account=account,
                limits=limits,
                reconciliation=reconciliation,
                kill_level=switch.level,
                controls=controls,
                now=at,
            )
            await self.ledger.append_pretrade(executable, decision)
            prepared.append((executable, decision))

        rejected = [
            f"{item.sleeve_id}:{','.join(decision.reason_codes)}"
            for item, decision in prepared
            if not decision.allowed or decision.intent is None
        ]
        if rejected:
            raise ExecutionRejected(
                "PRETRADE_REJECTED:" + ";".join(rejected)
            )

        orders = [
            await self._dispatch_intent(
                executable,
                decision.intent,
                now=at,
            )
            for executable, decision in prepared
            if decision.intent is not None
        ]
        statuses = {str(item["status"]) for item in orders}
        overall = (
            next(iter(statuses))
            if len(statuses) == 1
            else "TARGET_DISPATCH_MIXED"
        )
        response: dict[str, Any] = {
            "status": overall,
            "target_id": target.target_id,
            "orders": tuple(orders),
        }
        if len(orders) == 1:
            response.update(orders[0])
            response["status"] = overall
        return response

    async def _dispatch_intent(
        self,
        target: ExecutableExposure,
        intent: OrderIntent,
        *,
        now: datetime,
    ) -> Mapping[str, Any]:
        header, inserted = await self.ledger.claim_order(intent)
        order_id = str(header["order_id"])
        claim_token = await self.ledger.claim_order_dispatch(
            order_id=order_id,
            execution_fingerprint=intent.execution_fingerprint,
            claimed_at=now,
            lease_seconds=30,
        )
        if claim_token is None:
            return {
                "status": "IDEMPOTENT_REPLAY",
                "order": header,
                "inserted": inserted,
            }

        if target.environment in {
            ExecutionEnvironment.REPLAY,
            ExecutionEnvironment.PAPER,
        }:
            finalized = await self.ledger.finalize_order_dispatch(
                order_id=order_id,
                claim_token=claim_token,
                event_time=now,
                status="SIMULATED",
                details={"environment": target.environment.value},
            )
            if not finalized:
                raise ExecutionRejected("ORDER_DISPATCH_FENCING_FAILED")
            return {"status": "SIMULATED", "order": header}

        try:
            broker_response = await self.broker.submit(intent)
        except BrokerRejected:
            finalized = await self.ledger.finalize_order_dispatch(
                order_id=order_id,
                claim_token=claim_token,
                event_time=now,
                status="REJECTED",
                reason_code="BROKER_EXPLICIT_REJECTION",
            )
            if not finalized:
                raise ExecutionRejected("ORDER_DISPATCH_FENCING_FAILED")
            raise
        except Exception as exc:
            recorded = await self.ledger.require_order_dispatch_reconciliation(
                order_id=order_id,
                claim_token=claim_token,
                event_time=now,
                reason_code="BROKER_SUBMIT_UNCERTAIN",
                details={"error_type": type(exc).__name__},
            )
            if not recorded:
                raise ExecutionRejected("ORDER_DISPATCH_FENCING_FAILED") from exc
            raise BrokerSubmissionUnknown(
                "broker submission requires reconciliation"
            ) from exc

        finalized = await self.ledger.finalize_order_dispatch(
            order_id=order_id,
            claim_token=claim_token,
            event_time=now,
            status="SUBMITTED",
            broker_order_id=(
                str(broker_response.get("broker_order_id") or "") or None
            ),
            details={"broker_response_uri": broker_response.get("raw_response_uri")},
        )
        if not finalized:
            raise ExecutionRejected("ORDER_DISPATCH_FENCING_FAILED")
        return {
            "status": "SUBMITTED",
            "order": header,
            "broker": broker_response,
        }

    def _pretrade(
        self,
        *,
        target: ExecutableExposure,
        account: AccountState,
        limits: RiskLimits,
        reconciliation: Reconciliation,
        kill_level: KillSwitchLevel,
        controls: ExecutionControls | None = None,
        now: datetime,
    ) -> PreTradeDecision:
        controls = controls or ExecutionControls(
            health_nominal=True,
            operational_nominal=True,
            price_within_collar=True,
            liquidity_sufficient=True,
            market_session_open=True,
            currency_settled=True,
            account_enabled=True,
            user_trading_enabled=True,
            trading_mode=target.environment,
        )
        if now.tzinfo is None or now.utcoffset() is None:
            raise ExecutionRejected("NOW_MUST_BE_TIMEZONE_AWARE")
        decimal_fields = {
            "target_weight": target.target_weight,
            "nav": account.nav,
            "current_qty": account.current_qty,
            "mark_price": account.mark_price,
            "daily_loss_fraction": account.daily_loss_fraction,
            "drawdown_fraction": account.drawdown_fraction,
            "gross_exposure_fraction": account.gross_exposure_fraction,
            "net_exposure_fraction": account.net_exposure_fraction,
            "leverage": account.leverage,
            "available_cash": account.available_cash,
        }
        if any(not isinstance(value, Decimal) or not value.is_finite() for value in decimal_fields.values()):
            raise ExecutionRejected("NONFINITE_PRETRADE_INPUT")
        target_notional = account.nav * target.target_weight
        current_notional = account.current_qty * account.mark_price
        delta_notional = target_notional - current_notional
        side = "BUY" if delta_notional > 0 else "SELL"
        quantity = abs(delta_notional / account.mark_price) if account.mark_price > 0 else Decimal("0")
        same_direction = (
            current_notional == 0
            or target_notional == 0
            or (current_notional > 0) == (target_notional > 0)
        )
        reduce_only = (
            target_notional == 0
            or (
                same_direction
                and current_notional != 0
                and abs(target_notional) <= abs(current_notional)
            )
        )
        kill_allows = (
            kill_level is KillSwitchLevel.CLEAR
            or (
                kill_level in {KillSwitchLevel.BLOCK_NEW, KillSwitchLevel.CANCEL_OPEN}
                and reduce_only
            )
        )
        projected_gross = max(
            Decimal("0"),
            account.gross_exposure_fraction
            - (abs(current_notional) / max(account.nav, Decimal("1")))
            + (abs(target_notional) / max(account.nav, Decimal("1"))),
        )
        projected_net = (
            account.net_exposure_fraction
            - (current_notional / max(account.nav, Decimal("1")))
            + (target_notional / max(account.nav, Decimal("1")))
        )
        checks = {
            "published_immutable_target": (
                target.target.semantic_hash.startswith("sha256:")
                and bool(target.target.exposures)
            ),
            "target_time_valid": target.valid_from <= now <= target.valid_until,
            "snapshot_and_allocation_linked": bool(target.snapshot_id and target.allocation_id),
            "decision_fingerprint_present": target.decision_fingerprint.startswith("sha256:"),
            "reconciliation_clean": reconciliation.status is ReconciliationStatus.RECONCILED,
            "broker_internal_position_match": reconciliation.internal_qty == reconciliation.broker_qty,
            "health_nominal": controls.health_nominal,
            "operational_nominal": controls.operational_nominal,
            "kill_switch_allows": kill_allows,
            "instrument_allowed": target.instrument_id in limits.allowed_instruments,
            "account_enabled": controls.account_enabled,
            "user_trading_enabled": controls.user_trading_enabled,
            "trading_mode_matches_environment": controls.trading_mode is target.environment,
            "daily_trade_cap": (
                type(account.daily_trade_count) is int
                and type(limits.max_daily_trades) is int
                and 0 <= account.daily_trade_count < limits.max_daily_trades
            ),
            "positive_nav": account.nav > 0,
            "positive_mark": account.mark_price > 0,
            "quantity_positive": quantity > 0,
            "minimum_notional": abs(delta_notional) >= limits.min_order_notional,
            "order_notional_cap": abs(delta_notional) <= limits.max_order_notional,
            "position_notional_cap": abs(target_notional) <= limits.max_position_notional,
            "gross_exposure_cap": projected_gross <= limits.max_gross_exposure_fraction,
            "net_exposure_cap": abs(projected_net) <= limits.max_net_exposure_fraction,
            "leverage_cap": account.leverage <= limits.max_leverage,
            "daily_loss_cap": account.daily_loss_fraction <= limits.max_daily_loss_fraction,
            "drawdown_cap": account.drawdown_fraction <= limits.max_drawdown_fraction,
            "price_collar": controls.price_within_collar,
            "liquidity": controls.liquidity_sufficient,
            "market_session": controls.market_session_open,
            "cash_and_currency": (
                controls.currency_settled
                and account.cash_currency == account.currency
                and limits.currency == account.currency
                and target.settlement_currency == account.currency
                and (reduce_only or account.available_cash >= abs(delta_notional))
            ),
        }
        reasons = tuple(key.upper() for key, passed in checks.items() if not passed)
        intent: OrderIntent | None = None
        if all(checks.values()):
            fingerprint = execution_fingerprint(
                decision_fingerprint_value=target.decision_fingerprint,
                env=target.environment,
                account_id=target.account_id,
                broker_id=self.broker_id,
                order_policy_hash=self.order_policy_hash,
            )
            idempotency = order_idempotency_key(
                account_id=target.account_id,
                instrument=target.instrument,
                target_version=target.target_version,
                decision_fingerprint=target.decision_fingerprint,
                environment=target.environment,
                rebalance_cutoff=target.rebalance_cutoff,
            )
            intent = OrderIntent(
                target_id=target.target_id,
                allocation_id=target.allocation_id,
                account_id=target.account_id,
                strategy_id=target.strategy_id,
                sleeve_id=target.sleeve_id,
                instrument_id=target.instrument_id,
                instrument=target.instrument,
                currency=target.settlement_currency,
                side=side,
                quantity=quantity,
                notional=abs(delta_notional),
                target_version=target.target_version,
                decision_fingerprint=target.decision_fingerprint,
                execution_fingerprint=fingerprint,
                idempotency_key=idempotency,
                broker_id=self.broker_id,
                environment=target.environment.value,
                rebalance_cutoff=target.rebalance_cutoff,
                reconciliation_id=reconciliation.reconciliation_id,
                reduce_only=reduce_only,
            )
        return PreTradeDecision(
            allowed=intent is not None,
            checks=checks,
            reason_codes=reasons,
            intent=intent,
            kill_switch_level=kill_level,
            reconciliation_id=reconciliation.reconciliation_id,
        )

    async def _enforce_switch_side_effects(
        self,
        account_id: str,
        switch: EffectiveKillSwitch,
        *,
        now: datetime | None = None,
    ) -> None:
        claimed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        actions: tuple[str, ...]
        if switch.level is KillSwitchLevel.CANCEL_OPEN:
            actions = ("cancel_open",)
        elif switch.level is KillSwitchLevel.EXIT_ALL:
            actions = ("cancel_open", "exit_all")
        elif switch.level is KillSwitchLevel.ACCOUNT_FREEZE:
            actions = ("cancel_open",)
        else:
            actions = ()
        for action in actions:
            action_key = f"{switch.event_id}:{account_id}:{action}"
            claim_token = await self.ledger.claim_kill_switch_action(
                action_key=action_key,
                kill_switch_event_id=switch.event_id,
                account_id=account_id,
                action=action,
                claimed_at=claimed_at,
                lease_seconds=30,
            )
            if claim_token is None:
                continue
            try:
                if action == "cancel_open":
                    result = await self.broker.cancel_open(
                        account_id, idempotency_key=action_key
                    )
                else:
                    result = await self.broker.exit_all(
                        account_id, idempotency_key=action_key
                    )
            except Exception as exc:
                recorded = await self.ledger.require_kill_switch_reconciliation(
                    action_key=action_key,
                    claim_token=claim_token,
                    reason_code="BROKER_EFFECT_UNCERTAIN",
                    details={"error_type": type(exc).__name__},
                )
                if not recorded:
                    raise ExecutionRejected(
                        "KILL_SWITCH_ACTION_FENCING_FAILED"
                    ) from exc
                raise KillSwitchActionUnknown(
                    f"{action} requires reconciliation before safe retry"
                ) from exc
            completed = await self.ledger.complete_kill_switch_action(
                action_key=action_key,
                claim_token=claim_token,
                details={"affected_count": len(result)},
            )
            if not completed:
                raise ExecutionRejected("KILL_SWITCH_ACTION_FENCING_FAILED")


__all__ = [
    "AccountState",
    "BrokerRejected",
    "BrokerSubmissionUnknown",
    "EffectiveKillSwitch",
    "ExecutableExposure",
    "ExecutionRejected",
    "ExecutionControls",
    "ExecutionService",
    "KillSwitchActionUnknown",
    "KillSwitchLevel",
    "OrderIntent",
    "PortfolioTarget",
    "PreTradeDecision",
    "Reconciliation",
    "ReconciliationStatus",
    "RiskLimits",
]
