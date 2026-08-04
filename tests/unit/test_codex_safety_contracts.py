"""IA-R001 safety contracts reproduced from Claude's adversarial review."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
import sqlite3
import sys
import types

import pytest

if "src" not in sys.modules:
    package = types.ModuleType("src")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src")]
    sys.modules["src"] = package


def test_canonical_numbers_are_typed_stable_and_line_ending_independent() -> None:
    from src.identity.canonical import canonical_json_bytes, semantic_hash

    assert semantic_hash({"x": 1}) == semantic_hash({"x": 1.0})
    assert semantic_hash({"x": 1.0}) != semantic_hash({"x": "1"})
    assert semantic_hash({"x": 1e-13}) != semantic_hash({"x": 0.0})
    assert semantic_hash({"x": "a\r\nb"}) == semantic_hash({"x": "a\nb"})
    assert canonical_json_bytes({"x": 1.0}) == b'{"x":1}'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_nonfinite_numbers(value: float) -> None:
    from src.identity.canonical import CanonicalizationError, canonical_json_bytes

    with pytest.raises(CanonicalizationError, match="NaN and Infinity are forbidden"):
        canonical_json_bytes({"value": value})


def test_schema_quantum_applies_inside_arrays_and_unmatched_paths_fail() -> None:
    from src.identity.canonical import CanonicalizationError, canonical_json_bytes

    assert canonical_json_bytes(
        {"legs": [{"qty": Decimal("1.234")}]},
        field_quantums={"/legs/qty": "0.01"},
    ) == b'{"legs":[{"qty":1.23}]}'
    with pytest.raises(CanonicalizationError, match="matched no numeric"):
        canonical_json_bytes({"legs": []}, field_quantums={"/legs/qty": "0.01"})


def test_illegal_governance_object_cannot_exist() -> None:
    from src.governance.declaration import (
        CapitalTier,
        GovernanceDeclaration,
        OperationalState,
        ResearchState,
    )

    with pytest.raises(ValueError, match="PAPER cannot use FULL"):
        GovernanceDeclaration(
            ResearchState.PAPER,
            CapitalTier.FULL,
            OperationalState.NOMINAL,
        )


def test_market_alias_registry_is_bijective_and_quality_is_fail_closed() -> None:
    from src.data_quality.rules import QualityRuleSet
    from src.market.identity import IdentityError, ProviderSymbol, ProviderSymbolRegistry

    with pytest.raises(IdentityError, match="maps to both"):
        ProviderSymbolRegistry(
            [
                ProviderSymbol("yahoo", "MXN=X", "usdmxn"),
                ProviderSymbol(" YAHOO ", "mxn=x", "other"),
            ]
        )
    registry = ProviderSymbolRegistry(
        [
            ProviderSymbol("twelvedata", "USD/MXN", "usdmxn"),
            ProviderSymbol("yahoo", "MXN=X", "usdmxn"),
            ProviderSymbol("internal", "USD_MXN", "usdmxn"),
        ]
    )
    rules = QualityRuleSet.from_yaml(
        "config/quality/market_price_ranges.yaml", identity_registry=registry
    )
    good = {"open": 19, "high": 20, "low": 18, "close": 19.5}
    modern_start = datetime(1993, 1, 1, tzinfo=timezone.utc)
    assert rules.evaluate_provider_bar(
        "TWELVEDATA", "usd/mxn", good, observed_at=modern_start
    ).accepted
    historical = {"open": 2.72, "high": 2.75, "low": 2.7, "close": 2.712}
    assert rules.evaluate_provider_bar(
        "twelvedata", "USD/MXN", historical, observed_at=modern_start
    ).accepted
    below_declared_range = {"open": 2.49, "high": 2.49, "low": 2.49, "close": 2.49}
    assert not rules.evaluate_provider_bar(
        "twelvedata", "USD/MXN", below_declared_range, observed_at=modern_start
    ).accepted
    assert not rules.evaluate_provider_bar(
        "yahoo", "MXN=X", good, observed_at=modern_start
    ).accepted
    assert not rules.evaluate_provider_bar("twelvedata", "USD/MXN", good).accepted
    assert not rules.evaluate_provider_bar(
        "twelvedata", "USD/MXN", good, observed_at=datetime(1993, 1, 1)
    ).accepted
    assert not rules.evaluate_provider_bar(
        "twelvedata",
        "USD/MXN",
        good,
        observed_at=modern_start - timedelta(microseconds=1),
    ).accepted
    assert not rules.evaluate_bar("usdmxn", good).accepted
    assert not rules.evaluate_bar("unknown", good).accepted
    assert not rules.evaluate_bar(
        "usdmxn", {"open": -1, "high": 20, "low": -2, "close": 19}
    ).accepted
    assert not rules.evaluate_bar(
        "usdmxn",
        {"open": "Infinity", "high": "Infinity", "low": 18, "close": 19},
    ).accepted


def test_scoped_quality_range_config_is_closed_world(tmp_path: Path) -> None:
    from src.data_quality.rules import QualityRuleSet

    config = tmp_path / "ranges.yaml"
    config.write_text(
        """version: '1'\nprice_ranges:\n  usdmxn:\n    - provider_id: twelvedata\n      valid_from: '1993-01-01T00:00:00Z'\n      bounds: [2.5, 100]\n      invented: true\n""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown fields"):
        QualityRuleSet.from_yaml(config)


def test_scoped_quality_range_uses_latest_effective_regime(tmp_path: Path) -> None:
    from src.data_quality.rules import QualityRuleSet
    from src.market.identity import ProviderSymbol, ProviderSymbolRegistry

    registry = ProviderSymbolRegistry(
        [ProviderSymbol("twelvedata", "USD/MXN", "usdmxn")]
    )
    old_regime_only = {"open": 5, "high": 5, "low": 5, "close": 5}
    regimes = [
        """    - provider_id: twelvedata
      valid_from: '1993-01-01T00:00:00Z'
      bounds: [2.5, 100]
""",
        """    - provider_id: twelvedata
      valid_from: '2010-01-01T00:00:00Z'
      bounds: [10, 40]
""",
    ]

    for order, ordered_regimes in (
        ("ascending", regimes),
        ("descending", list(reversed(regimes))),
    ):
        config = tmp_path / f"ranges-{order}.yaml"
        config.write_text(
            "version: '1'\nprice_ranges:\n  usdmxn:\n" + "".join(ordered_regimes),
            encoding="utf-8",
        )
        rules = QualityRuleSet.from_yaml(config, identity_registry=registry)

        assert rules.evaluate_provider_bar(
            "twelvedata",
            "USD/MXN",
            old_regime_only,
            observed_at=datetime(2009, 12, 31, tzinfo=timezone.utc),
        ).accepted, order
        decision = rules.evaluate_provider_bar(
            "twelvedata",
            "USD/MXN",
            old_regime_only,
            observed_at=datetime(2010, 1, 1, tzinfo=timezone.utc),
        )
        assert not decision.accepted, order
        assert decision.rule_id == "bar.range.usdmxn", order


def test_scoped_quality_range_rejects_duplicate_provider_cutoff(tmp_path: Path) -> None:
    from src.data_quality.rules import QualityRuleSet

    config = tmp_path / "ranges.yaml"
    config.write_text(
        """version: '1'
price_ranges:
  usdmxn:
    - provider_id: twelvedata
      valid_from: '2010-01-01T00:00:00Z'
      bounds: [10, 40]
    - provider_id: TWELVEDATA
      valid_from: '2010-01-01T00:00:00Z'
      bounds: [11, 41]
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate scoped price range"):
        QualityRuleSet.from_yaml(config)


def _target(environment: str = "paper", currency: str = "USD"):
    from src.portfolio.target import TargetBuilder, TargetExposure

    now = datetime(2026, 1, 5, tzinfo=timezone.utc)
    return TargetBuilder().build(
        target_version="v1",
        snapshot_id="snapshot-1",
        account_id="account-1",
        allocator_version="allocator-v1",
        constraints_snapshot={"gross_cap": Decimal("1")},
        exposures=[
            TargetExposure(
                allocation_id="allocation-1",
                strategy_id="strategy-1",
                sleeve_id="sleeve-1",
                instrument_id="instrument-1",
                instrument="USD/COP",
                currency=currency,
                side="LONG",
                risk_budget=Decimal("0.1"),
                target_weight=Decimal("0.1"),
            )
        ],
        valid_from=now,
        valid_until=now + timedelta(hours=1),
        decision_fingerprint="sha256:" + "a" * 64,
        environment=environment,
        rebalance_cutoff=now,
    )


def _executable(environment: str = "paper", currency: str = "USD"):
    from src.portfolio.target import ExecutableExposure

    target = _target(environment, currency)
    return ExecutableExposure(target, target.exposures[0])


def test_execution_environment_is_closed_and_is_part_of_idempotency() -> None:
    from src.execution.events import order_idempotency_key
    from src.portfolio.target import TargetError

    with pytest.raises((TargetError, ValueError), match="lve"):
        _target("lve")
    cutoff = datetime(2026, 1, 5, tzinfo=timezone.utc)
    common = dict(
        account_id="a",
        instrument="USD/COP",
        target_version="v1",
        decision_fingerprint="sha256:" + "a" * 64,
        rebalance_cutoff=cutoff,
    )
    assert order_idempotency_key(environment="paper", **common) != order_idempotency_key(
        environment="live", **common
    )


def test_pretrade_rejects_currency_and_global_mode_mismatch() -> None:
    from src.execution.service import (
        AccountState,
        ExecutionControls,
        ExecutionService,
        KillSwitchLevel,
        Reconciliation,
        ReconciliationStatus,
        RiskLimits,
    )

    now = datetime(2026, 1, 5, tzinfo=timezone.utc)
    service = ExecutionService(
        targets=None,
        accounts=None,
        risks=None,
        kill_switch=None,
        reconciler=None,
        ledger=None,
        broker=None,
        broker_id="broker",
        order_policy_hash="sha256:" + "b" * 64,
    )
    decision = service._pretrade(
        target=_executable("live", "COP"),
        account=AccountState(
            nav=Decimal("1000"),
            current_qty=Decimal("0"),
            mark_price=Decimal("10"),
            currency="USD",
            broker_position_qty=Decimal("0"),
            available_cash=Decimal("1000"),
            cash_currency="USD",
        ),
        limits=RiskLimits(
            max_order_notional=Decimal("1000"),
            max_position_notional=Decimal("1000"),
            max_gross_exposure_fraction=Decimal("2"),
            max_daily_loss_fraction=Decimal("1"),
            allowed_instruments=frozenset({"instrument-1"}),
            currency="USD",
            max_daily_trades=10,
        ),
        reconciliation=Reconciliation(
            "r", ReconciliationStatus.RECONCILED, Decimal("0"), Decimal("0"), now
        ),
        kill_level=KillSwitchLevel.CLEAR,
        controls=ExecutionControls(
            True, True, True, True, True, True, True, True, "paper"
        ),
        now=now,
    )
    assert not decision.allowed
    assert "TRADING_MODE_MATCHES_ENVIRONMENT" in decision.reason_codes
    assert "CASH_AND_CURRENCY" in decision.reason_codes


class _Repo:
    def __init__(self, value):
        self.value = value

    async def get_target(self, target_id):
        return self.value

    async def state_for(self, target):
        return self.value

    async def limits_for(self, target):
        return self.value

    async def for_target(self, target, at):
        return self.value


class _KillRepo:
    def __init__(self, state):
        self.state = state

    async def effective_state(self, account_id, at):
        return self.state


class _Reconciler:
    def __init__(self, value):
        self.value = value

    async def pre_operation(self, target, account):
        return self.value


class _Ledger:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.orders = {}
        self.statuses = []
        self.dispatches = {}
        self.dispatch_attempts = {}
        self.actions = {}
        self.action_attempts = {}
        self.kill_switch_event_ids = []

    async def append_pretrade(self, target, decision):
        return None

    async def claim_order(self, intent):
        async with self.lock:
            if intent.idempotency_key in self.orders:
                return self.orders[intent.idempotency_key], False
            order = {
                "order_id": f"order-{len(self.orders) + 1}",
                "key": intent.idempotency_key,
            }
            self.orders[intent.idempotency_key] = order
            return order, True

    async def claim_order_dispatch(
        self,
        *,
        order_id,
        execution_fingerprint,
        claimed_at,
        lease_seconds,
    ):
        async with self.lock:
            state = self.dispatches.get(order_id)
            if state in {"CLAIMED", "COMPLETED"}:
                return None
            attempt = self.dispatch_attempts.get(order_id, 0) + 1
            self.dispatch_attempts[order_id] = attempt
            token = f"{order_id}:dispatch:{attempt}"
            self.dispatches[order_id] = "CLAIMED"
            return token

    async def finalize_order_dispatch(
        self,
        *,
        order_id,
        claim_token,
        event_time,
        status,
        reason_code=None,
        broker_order_id=None,
        details=None,
    ):
        async with self.lock:
            if self.dispatches.get(order_id) != "CLAIMED":
                return False
            self.statuses.append((status, reason_code))
            self.dispatches[order_id] = "COMPLETED"
            return True

    async def require_order_dispatch_reconciliation(
        self,
        *,
        order_id,
        claim_token,
        event_time,
        reason_code,
        details=None,
    ):
        async with self.lock:
            if self.dispatches.get(order_id) != "CLAIMED":
                return False
            self.statuses.append(("SUBMIT_UNKNOWN", reason_code))
            self.dispatches[order_id] = "RECONCILIATION_REQUIRED"
            return True

    async def claim_kill_switch_action(
        self,
        *,
        action_key,
        kill_switch_event_id,
        account_id,
        action,
        claimed_at=None,
        lease_seconds=30,
    ):
        async with self.lock:
            self.kill_switch_event_ids.append(kill_switch_event_id)
            state = self.actions.get(action_key)
            if state in {"CLAIMED", "COMPLETED"}:
                return None
            attempt = self.action_attempts.get(action_key, 0) + 1
            self.action_attempts[action_key] = attempt
            token = f"{action_key}:attempt:{attempt}"
            self.actions[action_key] = "CLAIMED"
            return token

    async def complete_kill_switch_action(self, *, action_key, claim_token, details=None):
        async with self.lock:
            if self.actions.get(action_key) != "CLAIMED":
                return False
            self.actions[action_key] = "COMPLETED"
            return True

    async def require_kill_switch_reconciliation(
        self, *, action_key, claim_token, reason_code, details=None
    ):
        async with self.lock:
            if self.actions.get(action_key) != "CLAIMED":
                return False
            self.actions[action_key] = "RECONCILIATION_REQUIRED"
            return True

    async def append_status(
        self, order_id, status, reason_code=None, broker_order_id=None, details=None
    ):
        self.statuses.append((status, reason_code))


class _Broker:
    def __init__(self, error=None):
        self.error = error
        self.kill_error = None
        self.submits = 0
        self.cancels = 0
        self.exits = 0

    async def submit(self, intent):
        self.submits += 1
        if self.error:
            raise self.error
        return {"broker_order_id": "broker-1"}

    async def cancel_open(self, account_id, *, idempotency_key):
        self.cancels += 1
        if self.kill_error is not None:
            error, self.kill_error = self.kill_error, None
            raise error
        return []

    async def exit_all(self, account_id, *, idempotency_key):
        self.exits += 1
        return []


def _service(environment="paper", *, broker_error=None):
    from src.execution.service import (
        AccountState,
        EffectiveKillSwitch,
        ExecutionControls,
        ExecutionService,
        KillSwitchLevel,
        Reconciliation,
        ReconciliationStatus,
        RiskLimits,
    )

    now = datetime(2026, 1, 5, tzinfo=timezone.utc)
    account = AccountState(
        nav=Decimal("1000"),
        current_qty=Decimal("0"),
        mark_price=Decimal("10"),
        currency="USD",
        broker_position_qty=Decimal("0"),
        available_cash=Decimal("1000"),
        cash_currency="USD",
    )
    limits = RiskLimits(
        max_order_notional=Decimal("1000"),
        max_position_notional=Decimal("1000"),
        max_gross_exposure_fraction=Decimal("2"),
        max_daily_loss_fraction=Decimal("1"),
        allowed_instruments=frozenset({"instrument-1"}),
        currency="USD",
        max_daily_trades=10,
    )
    reconciliation = Reconciliation(
        "r", ReconciliationStatus.RECONCILED, Decimal("0"), Decimal("0"), now
    )
    controls = ExecutionControls(
        True, True, True, True, True, True, True, True, environment
    )
    ledger = _Ledger()
    broker = _Broker(broker_error)
    service = ExecutionService(
        targets=_Repo(_target(environment)),
        accounts=_Repo(account),
        risks=_Repo(limits),
        kill_switch=_KillRepo(EffectiveKillSwitch("switch-clear", KillSwitchLevel.CLEAR)),
        reconciler=_Reconciler(reconciliation),
        ledger=ledger,
        broker=broker,
        broker_id="broker",
        order_policy_hash="sha256:" + "b" * 64,
        controls=_Repo(controls),
    )
    return service, ledger, broker, now


def test_paper_never_reaches_broker_and_atomic_claim_prevents_duplicates() -> None:
    async def scenario():
        service, _, broker, now = _service("paper")
        results = await asyncio.gather(
            service.execute_target("target-1", now=now),
            service.execute_target("target-1", now=now),
        )
        assert {result["status"] for result in results} == {
            "SIMULATED",
            "IDEMPOTENT_REPLAY",
        }
        assert broker.submits == 0

    asyncio.run(scenario())


def test_broker_timeout_is_unknown_and_kill_actions_are_one_shot() -> None:
    from src.execution.service import (
        BrokerSubmissionUnknown,
        EffectiveKillSwitch,
        KillSwitchLevel,
    )

    async def scenario():
        service, ledger, broker, now = _service("live", broker_error=TimeoutError())
        with pytest.raises(BrokerSubmissionUnknown):
            await service.execute_target("target-1", now=now)
        assert ledger.statuses[-1][0] == "SUBMIT_UNKNOWN"
        switch = EffectiveKillSwitch("switch-exit", KillSwitchLevel.EXIT_ALL)
        await service._enforce_switch_side_effects("account-1", switch)
        await service._enforce_switch_side_effects("account-1", switch)
        assert (broker.cancels, broker.exits) == (1, 1)

    asyncio.run(scenario())


def test_unknown_order_dispatch_retries_with_same_broker_idempotency_key() -> None:
    from src.execution.service import BrokerSubmissionUnknown

    async def scenario():
        service, ledger, broker, now = _service("live", broker_error=TimeoutError())
        with pytest.raises(BrokerSubmissionUnknown):
            await service.execute_target("target-1", now=now)
        first_key = next(iter(ledger.orders))
        broker.error = None
        result = await service.execute_target(
            "target-1", now=now + timedelta(seconds=1)
        )
        assert result["status"] == "SUBMITTED"
        assert next(iter(ledger.orders)) == first_key
        assert len(ledger.orders) == 1
        assert broker.submits == 2
        assert ledger.dispatch_attempts["order-1"] == 2

    asyncio.run(scenario())


def test_kill_action_transport_failure_is_recorded_and_safely_retried() -> None:
    from src.execution.service import (
        EffectiveKillSwitch,
        KillSwitchActionUnknown,
        KillSwitchLevel,
    )

    async def scenario():
        service, ledger, broker, _ = _service("live")
        broker.kill_error = TimeoutError("acceptance unknown")
        switch = EffectiveKillSwitch("switch-cancel", KillSwitchLevel.CANCEL_OPEN)

        with pytest.raises(KillSwitchActionUnknown):
            await service._enforce_switch_side_effects("account-1", switch)
        assert ledger.actions["switch-cancel:account-1:cancel_open"] == (
            "RECONCILIATION_REQUIRED"
        )

        await service._enforce_switch_side_effects("account-1", switch)
        assert broker.cancels == 2
        assert ledger.actions["switch-cancel:account-1:cancel_open"] == "COMPLETED"
        assert ledger.kill_switch_event_ids == ["switch-cancel", "switch-cancel"]

        await service._enforce_switch_side_effects("account-1", switch)
        assert broker.cancels == 2
        assert ledger.kill_switch_event_ids == [
            "switch-cancel",
            "switch-cancel",
            "switch-cancel",
        ]

    asyncio.run(scenario())


def test_kill_action_sql_uses_fencing_leases_and_immutable_attempt_events() -> None:
    sql = (
        Path("database/migrations/077_portfolio_control.sql")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "claim_token uuid not null" in sql
    assert "lease_expires_at timestamptz not null" in sql
    assert "create or replace function portfolio.claim_kill_switch_action" in sql
    assert "create or replace function portfolio.complete_kill_switch_action" in sql
    assert "create or replace function portfolio.require_kill_switch_reconciliation" in sql
    assert "create table if not exists portfolio.kill_switch_action_event" in sql
    assert "trg_kill_switch_action_event_immutable" in sql
    assert "revoke all on function portfolio.claim_kill_switch_action" in sql


def test_pnl_identity_requires_complete_single_currency_lineage_and_safe_tolerance() -> None:
    sql = (
        Path("database/migrations/075_fact_position_pnl.sql")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "gross_pnl_count" in sql
    assert "reported_residual_count" in sql
    assert "currency_value_count" in sql
    assert "lineage_value_count" in sql
    assert "p_tolerance is null" in sql
    assert "'nan'::numeric" in sql
    assert "p_tolerance > 1" in sql


def test_pnl_identity_error_applies_abs_after_subtracting_reported_residual() -> None:
    """Execute the production expression where the two parenthesizations diverge."""
    sql = Path("database/migrations/075_fact_position_pnl.sql").read_text(
        encoding="utf-8"
    )
    calculated_marker = ") AS calculated_residual,"
    calculated_end = sql.index(calculated_marker)
    expression_start = calculated_end + len(calculated_marker)
    expression_end = sql.index(" AS identity_error", expression_start)
    identity_error_expression = sql[expression_start:expression_end].strip()

    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE pnl (pnl_component TEXT NOT NULL, amount NUMERIC NOT NULL)"
        )

        def identity_error(reported_residual: int) -> int:
            connection.execute("DELETE FROM pnl")
            connection.executemany(
                "INSERT INTO pnl (pnl_component, amount) VALUES (?, ?)",
                [
                    ("gross_pnl", 10),
                    ("pnl_beta", 15),
                    ("pnl_residual", reported_residual),
                ],
            )
            row = connection.execute(
                f"SELECT {identity_error_expression} FROM pnl"
            ).fetchone()
            assert row is not None
            return int(row[0])

        # calculated_residual = -5. Correct: ABS(calculated - reported).
        assert identity_error(-3) == 2
        assert identity_error(3) == 8


def test_execution_sql_fences_status_transitions_and_correction_chain() -> None:
    sql = (
        Path("database/migrations/074_exec_event_sourcing.sql")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "create or replace function exec.validate_order_status_transition" in sql
    assert "for update" in sql
    assert "invalid order status transition" in sql
    assert "correction_fingerprint text not null unique" in sql
    assert "old_value text not null" in sql
    assert "effective_fill_fingerprint" in sql
    assert "broker_fill_id cannot collide" in sql
    assert "fill currency must match order currency" in sql
    assert "create table if not exists exec.order_dispatch" in sql
    assert "create table if not exists exec.order_dispatch_event" in sql
    assert "create or replace function exec.claim_order_dispatch" in sql
    assert "create or replace function exec.finalize_order_dispatch" in sql
    assert "create or replace function exec.require_order_dispatch_reconciliation" in sql
    assert "trg_order_dispatch_event_immutable" in sql


def test_reconciliation_cannot_claim_green_with_missing_or_contradictory_facts() -> None:
    sql = (
        Path("database/migrations/078_exec_reconciliation.sql")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "currency text not null" in sql
    assert "internal_qty is not null and broker_qty is not null" in sql
    assert "internal_cash is not null and broker_cash is not null" in sql
    assert "reconciled quantities disagree" in sql
    assert "reconciled cash balances disagree" in sql
    assert "reconciled status cannot carry discrepancies" in sql
    assert "mismatch status requires discrepancies" in sql
    assert "reconciliation observed_at is in the future" in sql
    assert "'reconciliation_guard',\n            'broker reconciliation discrepancy:" in sql
    assert "new.reconciliation_id, now()" in sql


def test_portfolio_sql_uses_materialized_sleeves_and_aggregate_target_artifact() -> None:
    sql = (
        Path("database/migrations/077_portfolio_control.sql")
        .read_text(encoding="utf-8")
        .lower()
    )
    reconciliation_sql = (
        Path("database/migrations/078_exec_reconciliation.sql")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "signal_id text" in sql
    assert "resolution text not null" in sql
    assert "materialized_payload jsonb not null" in sql
    assert "create table if not exists portfolio.target_exposure" in sql
    assert "constraints_snapshot jsonb not null" in sql
    assert "infeasibility_fallback integer" in sql
    assert "approval_status" not in sql
    assert "trg_target_exposure_immutable" in sql
    assert "fk_exec_order_portfolio_exposure" in reconciliation_sql


def _metric_engine():
    from src.metrics.engine import MetricCatalog, MetricEngine

    return MetricEngine.from_asset_registry(
        MetricCatalog.load("config/metrics/catalog.yaml"),
        assets_dir="config/assets",
    )


def _metric_context(n_trades=20, returns=None):
    end = datetime(2026, 1, 5, tzinfo=timezone.utc)
    return end, {
        "returns": returns or ([0.01, -0.005, 0.003, -0.001] * 5),
        "return_interval": "P1W",
        "n_trades": n_trades,
        "window_start": end - timedelta(weeks=26),
        "window_end": end,
    }


def test_metrics_suppress_small_samples_and_undefined_ratios() -> None:
    from src.metrics.formulas import calmar_ratio

    end, context = _metric_context(n_trades=5)
    event = _metric_engine().compute(
        entity_type="strategy",
        entity_id="s",
        metric="strategy.sharpe",
        window="26w",
        env="backtest",
        as_of=end,
        asset_id="usdcop",
        context=context,
    )
    assert event.metric_value is None
    assert event.status == "INSUFFICIENT_SAMPLE"
    assert calmar_ratio(__import__("numpy").array([0.01] * 30), 52) is None


def test_metric_identity_commits_inputs_and_window_label_is_enforced() -> None:
    from src.metrics.engine import MetricContractError

    engine = _metric_engine()
    end, first_context = _metric_context()
    _, second_context = _metric_context(
        returns=[0.02, -0.005, 0.003, -0.001] * 5
    )
    kwargs = dict(
        entity_type="strategy",
        entity_id="s",
        metric="strategy.sharpe",
        window="26w",
        env="backtest",
        as_of=end,
        asset_id="usdcop",
    )
    assert engine.compute(context=first_context, **kwargs).metric_event_id != engine.compute(
        context=second_context, **kwargs
    ).metric_event_id
    first_context["window_start"] = end - timedelta(weeks=25)
    with pytest.raises(MetricContractError, match="label"):
        engine.compute(context=first_context, **kwargs)


def test_fabric_migration_plan_is_explicit_and_review_gated() -> None:
    import importlib.util

    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_contract", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names = [item.name for item in module.get_migration_files("fabric-v1")]
    assert names == [f"{number:03d}_" + suffix for number, suffix in (
        (70, "fabric_control_plane.sql"),
        (71, "forecast_schema_roles.sql"),
        (72, "reference_identity.sql"),
        (73, "market_quality.sql"),
        (74, "exec_event_sourcing.sql"),
        (75, "fact_position_pnl.sql"),
        (76, "lineage_graph.sql"),
        (77, "portfolio_control.sql"),
        (78, "exec_reconciliation.sql"),
        (79, "fabric_integrity_remediation.sql"),
        (80, "market_physical_profile.sql"),
        (81, "synthetic_demo_isolation.sql"),
    )]
    assert not module.plan_is_authorized("fabric-v1", None)
    digest = module.get_plan_digest("fabric-v1")
    assert digest == module.PINNED_PLAN_DIGESTS["fabric-v1"]
    assert module.plan_is_authorized("fabric-v1", digest)


def test_platform_bootstrap_plan_is_explicit_minimal_and_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.util

    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_bootstrap", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = [item.name for item in module.get_migration_files("platform-bootstrap-v1")]
    assert names == [
        "045_newsengine_initial.sql",
        "046_weekly_analysis_tables.sql",
        "050_consolidated_h5_ddl.sql",
        "051_asset_daily_ohlcv.sql",
        "053_sb_user_approval.sql",
        "054_h5_subtrades_unique.sql",
        "055_rbac_monetization.sql",
    ]
    assert module.PLAN_PREREQUISITE_TABLES["platform-bootstrap-v1"] == (
        "public.sb_users",
        "public.usdcop_m5_ohlcv",
        "public.macro_indicators_daily",
    )
    assert "platform-bootstrap-v1" in module.REVIEW_GATED_PLANS
    assert not module.plan_is_authorized("platform-bootstrap-v1", None)
    reviewed_digest = module.get_plan_digest("platform-bootstrap-v1")
    assert reviewed_digest == module.PINNED_PLAN_DIGESTS["platform-bootstrap-v1"]
    assert module.plan_is_authorized("platform-bootstrap-v1", reviewed_digest)

    original_files = module.MIGRATION_PLANS["platform-bootstrap-v1"]
    changed_migration = tmp_path / original_files[0].name
    original_bytes = original_files[0].read_bytes()
    changed_migration.write_bytes(original_bytes + b"\n-- unauthorized byte change\n")
    monkeypatch.setitem(
        module.MIGRATION_PLANS,
        "platform-bootstrap-v1",
        (changed_migration, *original_files[1:]),
    )

    changed_digest = module.get_plan_digest("platform-bootstrap-v1")
    assert changed_digest != reviewed_digest
    assert not module.plan_is_authorized("platform-bootstrap-v1", reviewed_digest)
    assert not module.plan_is_authorized("platform-bootstrap-v1", changed_digest)


def test_platform_bootstrap_excludes_superseded_and_optional_migrations() -> None:
    import importlib.util

    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_exclusions", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = {item.name for item in module.get_migration_files("platform-bootstrap-v1")}
    forbidden = {
        "043_forecast_h5_tables.sql",
        "044_smart_simple_columns.sql",
        "047_pgvector_embeddings.sql",
        "048_reconciliation_tables.sql",
        "049_regime_gate_columns.sql",
        "052_crypto_native_data.sql",
    }
    assert names.isdisjoint(forbidden)


def test_platform_bootstrap_orders_user_role_before_rbac_expansion() -> None:
    import importlib.util

    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_role_order", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    names = [item.name for item in module.get_migration_files("platform-bootstrap-v1")]
    assert names.index("053_sb_user_approval.sql") < names.index(
        "055_rbac_monetization.sql"
    )


def test_migrator_prefers_database_url_and_requires_explicit_fallback_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util

    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_connection", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls: list[dict[str, object]] = []

    async def connect(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setitem(sys.modules, "asyncpg", types.SimpleNamespace(connect=connect))
    monkeypatch.setenv("DATABASE_URL", "postgresql://configured.example/db")
    monkeypatch.setenv("POSTGRES_PASSWORD", "ignored-fallback")
    asyncio.run(module.get_connection())
    assert calls == [{"dsn": "postgresql://configured.example/db"}]

    calls.clear()
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.delenv("POSTGRES_PASSWORD")
    with pytest.raises(RuntimeError, match="DATABASE_URL or POSTGRES_PASSWORD"):
        asyncio.run(module.get_connection())
    assert calls == []


def test_platform_bootstrap_prerequisites_fail_closed() -> None:
    import importlib.util

    path = Path("scripts/ops/db_migrate.py")
    spec = importlib.util.spec_from_file_location("db_migrate_prerequisites", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Connection:
        def __init__(self, existing: set[tuple[str, str]]) -> None:
            self.existing = existing

        async def fetchval(
            self, _query: str, schema: str, table: str
        ) -> bool:
            return (schema, table) in self.existing

    incomplete = Connection(
        {("public", "sb_users"), ("public", "usdcop_m5_ohlcv")}
    )
    assert not asyncio.run(
        module.validate_plan_prerequisites(incomplete, "platform-bootstrap-v1")
    )

    complete = Connection(
        {
            ("public", "sb_users"),
            ("public", "usdcop_m5_ohlcv"),
            ("public", "macro_indicators_daily"),
        }
    )
    assert asyncio.run(
        module.validate_plan_prerequisites(complete, "platform-bootstrap-v1")
    )
