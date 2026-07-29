"""IA-R001 adversarial contracts for Codex FABRIC implementation."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path
import sys
import types

import pytest

# The repository's historical ``src.__init__`` eagerly imports the full trading
# stack (including optional runtime dependencies).  These contract tests isolate
# the new packages exactly as services do in their slim images.
if "src" not in sys.modules:
    package = types.ModuleType("src")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "src")]
    sys.modules["src"] = package


def test_governance_matrix_has_exactly_26_legal_state_combinations() -> None:
    from src.governance.declaration import (
        CapitalTier,
        GovernanceDeclaration,
        OperationalState,
        ResearchState,
        validate_declaration,
    )

    legal = 0
    for research in ResearchState:
        for capital in CapitalTier:
            for operational in OperationalState:
                try:
                    declaration = GovernanceDeclaration(
                        research,
                        capital,
                        operational,
                        exit_checklist="PASS" if research is ResearchState.WITHDRAWN else None,
                        dag_declared=research.value
                        in {"FROZEN", "PAPER", "CHAMPION", "RETIRING"},
                    )
                    validate_declaration(declaration)
                except ValueError:
                    continue
                legal += 1
    assert legal == 26


def test_canonical_mapping_rejects_non_string_key_with_domain_error() -> None:
    from src.identity.canonical import CanonicalizationError, canonical_json_bytes

    with pytest.raises(CanonicalizationError, match="keys must be strings"):
        canonical_json_bytes({"ok": 1, 2: "bad"})


def test_spec_fingerprint_commits_data_and_feature_snapshots() -> None:
    from src.identity.fingerprints import spec_fingerprint

    common = {
        "strategy_spec": {"id": "s1"},
        "feature_snapshot_id": "feature:1",
        "config_hash": "sha256:" + "1" * 64,
        "model_or_policy_hash": "sha256:" + "2" * 64,
        "calendar_hash": "sha256:" + "3" * 64,
        "cost_model_hash": "sha256:" + "4" * 64,
        "dependency_lock_hash": "sha256:" + "5" * 64,
        "container_image_digest": "sha256:" + "6" * 64,
    }
    first = spec_fingerprint(data_snapshot_id="data:1", **common)
    second = spec_fingerprint(data_snapshot_id="data:2", **common)
    assert first != second


def test_fabric_factory_config_builds_and_preserves_action_diagnostic_wall() -> None:
    import yaml

    from src.orchestration.factories import FactoryKind, build_all_specs

    raw = yaml.safe_load(open("config/assets/fabric_factories.yaml", encoding="utf-8"))
    specs = build_all_specs(raw)
    assert specs
    assert {spec.kind for spec in specs} == {
        FactoryKind.DATA,
        FactoryKind.STRATEGY,
        FactoryKind.FORECAST,
    }
    assert all(
        not uri.startswith("portfolio://")
        for spec in specs
        if spec.kind is FactoryKind.FORECAST
        for uri in spec.produces
    )


def test_semantic_diff_ignores_only_declared_volatile_fields() -> None:
    from src.orchestration.semantic_diff import compare

    result = compare(
        {"value": Decimal("1.20"), "generated_at": "old"},
        {"value": Decimal("1.2"), "generated_at": "new"},
    )
    assert result.equal
    assert result.first_difference is None


def test_snapshot_rejects_signal_not_available_at_cutoff() -> None:
    from src.portfolio.snapshot import AcceptedSignal, MissingPolicy, SnapshotBuilder

    cutoff = datetime(2026, 1, 5, tzinfo=timezone.utc)
    future = AcceptedSignal(
        signal_id="signal-1",
        sleeve_id="sleeve-1",
        as_of=cutoff - timedelta(days=1),
        available_at=cutoff + timedelta(seconds=1),
        valid_until=cutoff + timedelta(days=1),
        payload={"decision_fingerprint": "sha256:" + "a" * 64},
    )
    snapshot = SnapshotBuilder().build(
        cutoff_time=cutoff,
        required_sleeves=["sleeve-1"],
        signals=[future],
        max_age_by_sleeve={"sleeve-1": timedelta(days=3)},
        missing_policy_by_sleeve={"sleeve-1": MissingPolicy.FLAT},
    )
    assert snapshot.accepted_signals == ()
    assert snapshot.missing_signals == ("sleeve-1",)


def test_snapshot_hash_commits_policy_max_age_and_signal_payload() -> None:
    from src.portfolio.snapshot import AcceptedSignal, MissingPolicy, SnapshotBuilder

    cutoff = datetime(2026, 1, 5, tzinfo=timezone.utc)

    def build(max_age: timedelta, payload_value: str) -> str:
        signal = AcceptedSignal(
            signal_id="signal-1",
            sleeve_id="sleeve-1",
            as_of=cutoff,
            available_at=cutoff,
            valid_until=cutoff + timedelta(days=1),
            payload={"reason": payload_value},
        )
        return SnapshotBuilder().build(
            cutoff_time=cutoff,
            required_sleeves=["sleeve-1"],
            signals=[signal],
            max_age_by_sleeve={"sleeve-1": max_age},
            missing_policy_by_sleeve={"sleeve-1": MissingPolicy.USE_LAST_VALID_WITH_MAX_AGE},
        ).semantic_hash

    assert build(timedelta(days=1), "a") != build(timedelta(days=2), "a")
    assert build(timedelta(days=1), "a") != build(timedelta(days=1), "b")


def test_snapshot_is_deeply_immutable_and_retry_identity_is_deterministic() -> None:
    from src.portfolio.snapshot import (
        AcceptedSignal,
        MissingPolicy,
        SnapshotBuilder,
        SnapshotError,
    )

    cutoff = datetime(2026, 1, 5, tzinfo=timezone.utc)
    mutable_payload = {"nested": {"values": [1, 2]}}
    signal = AcceptedSignal(
        signal_id="signal-1",
        sleeve_id="sleeve-1",
        as_of=cutoff,
        available_at=cutoff,
        valid_until=cutoff + timedelta(days=1),
        payload=mutable_payload,
    )
    common = dict(
        cutoff_time=cutoff,
        required_sleeves=["sleeve-1"],
        signals=[signal],
        max_age_by_sleeve={"sleeve-1": timedelta(days=2)},
        missing_policy_by_sleeve={"sleeve-1": MissingPolicy.FLAT},
    )
    first = SnapshotBuilder().build(**common)
    second = SnapshotBuilder().build(**common)
    mutable_payload["nested"]["values"].append(3)

    assert first.snapshot_id == second.snapshot_id
    assert first.accepted_signals[0].payload["nested"]["values"] == (1, 2)
    with pytest.raises(TypeError):
        first.fallback_applied["sleeve-1"] = "EXIT_ONLY"
    with pytest.raises(
        SnapshotError,
        match=r"snapshot_id mismatch: expected .+, got whatever-i-want",
    ):
        replace(first, snapshot_id="whatever-i-want")
    with pytest.raises(SnapshotError, match="semantic_hash mismatch"):
        replace(first, semantic_hash="sha256:" + "0" * 64)
    with pytest.raises(SnapshotError, match="cutoff_time must be timezone-aware"):
        replace(first, cutoff_time=cutoff.replace(tzinfo=None))


def test_snapshot_materializes_fallbacks_instead_of_only_labelling_them() -> None:
    from src.portfolio.snapshot import (
        AcceptedSignal,
        MissingPolicy,
        SnapshotBuilder,
        SnapshotError,
    )

    cutoff = datetime(2026, 1, 5, tzinfo=timezone.utc)
    common = dict(
        cutoff_time=cutoff,
        required_sleeves=["sleeve-1"],
        signals=[],
        max_age_by_sleeve={"sleeve-1": timedelta(days=2)},
    )
    flat = SnapshotBuilder().build(
        **common,
        missing_policy_by_sleeve={"sleeve-1": MissingPolicy.FLAT},
    )
    materialized = flat.materialized_inputs["sleeve-1"]
    assert materialized.resolution is MissingPolicy.FLAT
    assert materialized.payload["side"] == 0
    assert materialized.payload["target_weight"] == Decimal("0")

    with pytest.raises(SnapshotError, match="position state"):
        SnapshotBuilder().build(
            **common,
            missing_policy_by_sleeve={
                "sleeve-1": MissingPolicy.KEEP_POSITION_UNTIL_EXPIRY
            },
        )
    with pytest.raises(SnapshotError, match="last valid signal"):
        SnapshotBuilder().build(
            **common,
            missing_policy_by_sleeve={
                "sleeve-1": MissingPolicy.USE_LAST_VALID_WITH_MAX_AGE
            },
        )
    expired = AcceptedSignal(
        signal_id="expired-signal",
        sleeve_id="sleeve-1",
        as_of=cutoff - timedelta(days=1),
        available_at=cutoff - timedelta(days=1),
        valid_until=cutoff - timedelta(seconds=1),
        payload={"decision_fingerprint": "sha256:" + "a" * 64},
    )
    with pytest.raises(SnapshotError, match="last valid signal exceeds cutoff/max_age"):
        SnapshotBuilder().build(
            **common,
            missing_policy_by_sleeve={
                "sleeve-1": MissingPolicy.USE_LAST_VALID_WITH_MAX_AGE
            },
            last_valid_signal_by_sleeve={"sleeve-1": expired},
        )


def _neutral_multipliers(*sleeves: str) -> dict[str, dict[str, float]]:
    return {
        sleeve: {
            "forward": 1.0,
            "liquidity": 1.0,
            "diversification": 1.0,
            "operations": 1.0,
            "drawdown": 1.0,
        }
        for sleeve in sleeves
    }


def test_inverse_volatility_baseline_is_normalized_before_caps() -> None:
    from src.portfolio.allocator import AllocatorV1

    result = AllocatorV1(
        sleeve_caps={"slow": 1.0, "fast": 1.0},
        gross_cap=1.0,
        asset_caps={"a": 1.0, "b": 1.0},
    ).allocate(
        volatility={"slow": 0.10, "fast": 0.20},
        side={"slow": 1, "fast": 1},
        sleeve_asset={"slow": "a", "fast": "b"},
        multipliers=_neutral_multipliers("slow", "fast"),
    )
    assert result.risk_budgets["slow"] == pytest.approx(2 / 3)
    assert result.risk_budgets["fast"] == pytest.approx(1 / 3)


def test_allocator_multiplier_reduces_without_renormalizing_survivors() -> None:
    from src.portfolio.allocator import AllocatorV1

    allocator = AllocatorV1(
        sleeve_caps={"a1": 1.0, "b1": 1.0},
        gross_cap=1.0,
        asset_caps={"a": 1.0, "b": 1.0},
    )
    baseline = allocator.allocate(
        volatility={"a1": 0.1, "b1": 0.1},
        side={"a1": 1, "b1": 1},
        sleeve_asset={"a1": "a", "b1": "b"},
        multipliers=_neutral_multipliers("a1", "b1"),
    )
    reduced = allocator.allocate(
        volatility={"a1": 0.1, "b1": 0.1},
        side={"a1": 1, "b1": 1},
        sleeve_asset={"a1": "a", "b1": "b"},
        multipliers={
            **_neutral_multipliers("a1", "b1"),
            "a1": {
                **_neutral_multipliers("a1")["a1"],
                "forward": 0.0,
            },
        },
    )
    assert reduced.risk_budgets["a1"] == 0
    assert reduced.risk_budgets["b1"] == baseline.risk_budgets["b1"]


def test_allocator_fails_closed_on_implicit_defaults_and_invalid_volatility() -> None:
    from src.portfolio.allocator import AllocationError, AllocatorV1

    allocator = AllocatorV1(
        sleeve_caps={"a1": 1.0},
        gross_cap=1.0,
        asset_caps={"a": 1.0},
    )
    common = dict(
        side={"a1": 1},
        sleeve_asset={"a1": "a"},
        multipliers=_neutral_multipliers("a1"),
    )
    with pytest.raises(AllocationError, match="volatility"):
        allocator.allocate(volatility={"a1": 0.0}, **common)
    with pytest.raises(AllocationError, match="multipliers"):
        allocator.allocate(
            volatility={"a1": 0.1},
            side={"a1": 1},
            sleeve_asset={"a1": "a"},
            multipliers={},
        )
    with pytest.raises(AllocationError, match="sleeve cap"):
        AllocatorV1(
            sleeve_caps={},
            gross_cap=1.0,
            asset_caps={"a": 1.0},
        ).allocate(volatility={"a1": 0.1}, **common)


def test_allocator_flat_sleeve_has_zero_budget_and_sql_compatible_weight() -> None:
    from src.portfolio.allocator import AllocatorV1

    result = AllocatorV1(
        sleeve_caps={"flat": 1.0, "active": 1.0},
        gross_cap=1.0,
        asset_caps={"a": 1.0, "b": 1.0},
    ).allocate(
        volatility={"flat": 0.1, "active": 0.1},
        side={"flat": 0, "active": 1},
        sleeve_asset={"flat": "a", "active": "b"},
        multipliers=_neutral_multipliers("flat", "active"),
    )
    assert result.risk_budgets["flat"] == 0
    assert result.signed_weights["flat"] == 0


def test_allocator_diversification_range_matches_registered_fabric_contract() -> None:
    from src.portfolio.allocator import AllocationError, AllocatorV1

    allocator = AllocatorV1(
        sleeve_caps={"a1": 1.0, "b1": 0.1},
        gross_cap=1.0,
        asset_caps={"a": 1.0, "b": 0.1},
    )
    common = dict(
        volatility={"a1": 0.1, "b1": 0.1},
        side={"a1": 1, "b1": 1},
        sleeve_asset={"a1": "a", "b1": "b"},
    )
    increased = allocator.allocate(
        **common,
        multipliers={
            **_neutral_multipliers("a1", "b1"),
            "a1": {
                **_neutral_multipliers("a1", "b1")["a1"],
                "diversification": 1.1,
            }
        },
    )
    assert increased.risk_budgets["a1"] == pytest.approx(0.55)
    assert increased.risk_budgets["b1"] == pytest.approx(0.1)
    with pytest.raises(AllocationError, match="forward"):
        allocator.allocate(
            **common,
            multipliers={
                **_neutral_multipliers("a1", "b1"),
                "a1": {
                    **_neutral_multipliers("a1", "b1")["a1"],
                    "forward": 1.01,
                }
            },
        )


class _ScriptedBudgetOptimizer:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []

    def solve(self, request):
        from src.portfolio.allocator import InfeasibleAllocation

        self.requests.append(request)
        if not self.outputs:
            raise InfeasibleAllocation("script exhausted")
        output = self.outputs.pop(0)
        if output is None:
            raise InfeasibleAllocation("scripted infeasibility")
        return output


def test_constrained_allocator_relaxes_only_to_registered_turnover_limit() -> None:
    from src.portfolio.allocator import AllocatorV1

    optimizer = _ScriptedBudgetOptimizer([None, {"a1": 0.4}])
    result = AllocatorV1(
        sleeve_caps={"a1": 1.0},
        gross_cap=1.0,
        asset_caps={"asset-a": 1.0},
        optimizer=optimizer,
    ).allocate_constrained(
        volatility={"a1": 0.1},
        side={"a1": 1},
        sleeve_asset={"a1": "asset-a"},
        multipliers=_neutral_multipliers("a1"),
        previous_budgets={"a1": 0.3},
        covariance=[[0.01]],
        target_vol=0.2,
        turnover_budget=0.05,
        turnover_relaxation_limit=0.1,
    )

    assert result.risk_budgets == {"a1": 0.4}
    assert result.fallback_level == 1
    assert result.incident == "ALLOCATOR_FALLBACK_1_TURNOVER_RELAXED"
    assert [request.turnover_budget for request in optimizer.requests] == [
        0.05,
        0.1,
    ]


def test_constrained_allocator_rejects_dishonest_solver_and_targets_zero() -> None:
    from src.portfolio.allocator import AllocatorV1

    optimizer = _ScriptedBudgetOptimizer([{"a1": 2.0}] * 7)
    result = AllocatorV1(
        sleeve_caps={"a1": 0.5},
        gross_cap=0.5,
        asset_caps={"asset-a": 0.5},
        optimizer=optimizer,
    ).allocate_constrained(
        volatility={"a1": 0.1},
        side={"a1": -1},
        sleeve_asset={"a1": "asset-a"},
        multipliers=_neutral_multipliers("a1"),
        previous_budgets={"a1": 0.0},
        covariance=[[0.01]],
        target_vol=0.2,
        turnover_budget=0.1,
        turnover_relaxation_limit=0.2,
    )

    assert len(optimizer.requests) == 7
    assert result.fallback_level == 4
    assert result.risk_budgets == {"a1": 0.0}
    assert result.signed_weights == {"a1": 0.0}
    assert result.incidents[-1].severity == "CRITICAL"


def test_constrained_allocator_fails_closed_on_non_psd_covariance() -> None:
    from src.portfolio.allocator import AllocationError, AllocatorV1

    optimizer = _ScriptedBudgetOptimizer([])
    with pytest.raises(AllocationError, match="positive semidefinite"):
        AllocatorV1(
            sleeve_caps={"a1": 1.0, "b1": 1.0},
            gross_cap=1.0,
            asset_caps={"asset-a": 1.0, "asset-b": 1.0},
            optimizer=optimizer,
        ).allocate_constrained(
            volatility={"a1": 0.1, "b1": 0.1},
            side={"a1": 1, "b1": 1},
            sleeve_asset={"a1": "asset-a", "b1": "asset-b"},
            multipliers=_neutral_multipliers("a1", "b1"),
            previous_budgets={"a1": 0.0, "b1": 0.0},
            covariance=[[1.0, 2.0], [2.0, 1.0]],
            target_vol=0.2,
            turnover_budget=0.1,
            turnover_relaxation_limit=0.2,
        )
    assert optimizer.requests == []


def test_portfolio_target_is_one_deterministic_aggregate_with_immutable_exposures() -> None:
    from src.portfolio.target import TargetBuilder, TargetError, TargetExposure

    cutoff = datetime(2026, 1, 5, tzinfo=timezone.utc)
    exposures = [
        TargetExposure(
            allocation_id="allocation-b",
            strategy_id="strategy-b",
            sleeve_id="sleeve-b",
            instrument_id="instrument-b",
            instrument="XAUUSD",
            side="SHORT",
            risk_budget=Decimal("0.2"),
            target_weight=Decimal("-0.2"),
            currency="USD",
        ),
        TargetExposure(
            allocation_id="allocation-a",
            strategy_id="strategy-a",
            sleeve_id="sleeve-a",
            instrument_id="instrument-a",
            instrument="USDCOP",
            side="LONG",
            risk_budget=Decimal("0.1"),
            target_weight=Decimal("0.1"),
            currency="COP",
        ),
    ]
    common = dict(
        target_version="184",
        snapshot_id="snapshot-1",
        account_id="account-1",
        environment="paper",
        allocator_version="allocator_v1",
        valid_from=cutoff,
        valid_until=cutoff + timedelta(days=1),
        rebalance_cutoff=cutoff,
        decision_fingerprint="sha256:" + "a" * 64,
        constraints_snapshot={"gross_cap": Decimal("1.0")},
        exposures=exposures,
    )
    first = TargetBuilder().build(**common)
    second = TargetBuilder().build(**{**common, "exposures": list(reversed(exposures))})

    assert first.target_id == second.target_id
    assert first.semantic_hash == second.semantic_hash
    assert tuple(item.sleeve_id for item in first.exposures) == (
        "sleeve-a",
        "sleeve-b",
    )
    with pytest.raises(TypeError):
        first.constraints_snapshot["gross_cap"] = Decimal("2")
    with pytest.raises(TargetError, match="duplicate sleeve"):
        TargetBuilder().build(**{**common, "exposures": [exposures[0], exposures[0]]})


def test_quality_rule_quarantines_decimal_nan_instead_of_crashing() -> None:
    from src.data_quality.rules import QualityRuleSet

    decision = QualityRuleSet().evaluate_bar(
        "USDMXN", {"open": "NaN", "high": "20", "low": "19", "close": "19.5"}
    )
    assert not decision.accepted
    assert decision.rule_id == "bar.numeric"


def test_qlab_idempotency_rejects_same_trial_id_with_changed_result(tmp_path) -> None:
    from src.research.qlab import QLabError, TrialCharge, TrialLedger

    ledger = TrialLedger(tmp_path / "ledger.jsonl")
    first = TrialCharge(
        trial_id="FT-9999",
        family="f",
        asset="a",
        cluster="c",
        kind="forecast",
        variant="v",
        cutoff="2025-01-01T00:00:00Z",
        result="reject",
    )
    ledger.charge(first)
    with pytest.raises(QLabError, match="different payload"):
        ledger.charge(TrialCharge(**{**first.__dict__, "result": "promote"}))


def test_execution_flip_is_opening_and_exit_all_never_submits_target() -> None:
    from src.execution.service import (
        AccountState,
        ExecutionService,
        KillSwitchLevel,
        Reconciliation,
        ReconciliationStatus,
        RiskLimits,
    )
    from src.portfolio.target import ExecutableExposure, TargetBuilder, TargetExposure

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
    aggregate = TargetBuilder().build(
        target_version="1",
        snapshot_id="s",
        account_id="acct",
        allocator_version="allocator-v1",
        constraints_snapshot={"gross_cap": Decimal("1")},
        exposures=[
            TargetExposure(
                allocation_id="a",
                strategy_id="strategy",
                sleeve_id="sleeve",
                instrument_id="instrument",
                instrument="USDCOP",
                currency="USD",
                side="SHORT",
                risk_budget=Decimal("0.10"),
                target_weight=Decimal("-0.10"),
            )
        ],
        valid_from=now,
        valid_until=now + timedelta(minutes=1),
        decision_fingerprint="sha256:" + "a" * 64,
        environment="paper",
        rebalance_cutoff=now,
    )
    target = ExecutableExposure(aggregate, aggregate.exposures[0])
    decision = service._pretrade(
        target=target,
        account=AccountState(
            nav=Decimal("1000"),
            current_qty=Decimal("10"),
            mark_price=Decimal("10"),
            currency="USD",
            broker_position_qty=Decimal("10"),
            available_cash=Decimal("1000"),
            cash_currency="USD",
        ),
        limits=RiskLimits(
            max_order_notional=Decimal("1000"),
            max_position_notional=Decimal("1000"),
            max_gross_exposure_fraction=Decimal("2"),
            max_daily_loss_fraction=Decimal("1"),
            allowed_instruments=frozenset({"instrument"}),
            currency="USD",
            max_daily_trades=100,
        ),
        reconciliation=Reconciliation(
            "r", ReconciliationStatus.RECONCILED, Decimal("10"), Decimal("10"), now
        ),
        kill_level=KillSwitchLevel.EXIT_ALL,
        now=now,
    )
    assert not decision.allowed
    assert "KILL_SWITCH_ALLOWS" in decision.reason_codes


def test_metric_event_identity_is_deterministic_and_annualization_is_finite() -> None:
    from src.metrics.engine import MetricCatalog, MetricContractError, MetricEngine

    catalog = MetricCatalog.load("config/metrics/catalog.yaml")
    engine = MetricEngine(catalog, annualization_by_asset={"usdcop": 52})
    as_of = datetime(2026, 1, 5, tzinfo=timezone.utc)
    kwargs = {
        "entity_type": "strategy",
        "entity_id": "s1",
        "metric": "strategy.sharpe",
        "window": "26w",
        "env": "backtest",
        "as_of": as_of,
        "asset_id": "usdcop",
        "context": {
            "returns": [0.01, -0.005, 0.003] * 7,
            "n_trades": 21,
            "window_start": as_of - timedelta(weeks=26),
            "window_end": as_of,
        },
        "run_id": "run-1",
    }
    assert engine.compute(**kwargs).metric_event_id == engine.compute(**kwargs).metric_event_id
    bad_engine = MetricEngine(
        catalog, annualization_by_asset={"usdcop": float("inf")}
    )
    with pytest.raises(MetricContractError, match="annualization"):
        bad_engine.compute(**kwargs)


def test_canonical_artifact_refuses_divergent_overwrite(tmp_path) -> None:
    from src.identity.canonical import CanonicalArtifact, CanonicalizationError

    path = tmp_path / "artifact.json"
    CanonicalArtifact.build({"a": 1}).write(path)
    CanonicalArtifact.build({"a": 1}).write(path)
    with pytest.raises(CanonicalizationError, match="divergent"):
        CanonicalArtifact.build({"a": 2}).write(path)


def test_lineage_rejects_bad_hash_negative_rows_and_inverted_range() -> None:
    from src.lineage.graph import LineageNode

    with pytest.raises(ValueError, match="semantic_hash"):
        LineageNode("n", "raw", "sha256:abc", "1", "VALID")
    with pytest.raises(ValueError, match="row_count"):
        LineageNode("n", "raw", "sha256:" + "a" * 64, "1", "VALID", row_count=-1)
    with pytest.raises(ValueError, match="cannot precede"):
        LineageNode(
            "n",
            "raw",
            "sha256:" + "a" * 64,
            "1",
            "VALID",
            min_event_time=datetime(2026, 1, 2, tzinfo=timezone.utc),
            max_event_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )


def test_fill_events_reject_nonfinite_values_and_expose_stable_fingerprint() -> None:
    from src.execution.events import ExecutionContractError, FillEvent

    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ExecutionContractError, match="finite"):
        FillEvent("f", "o", at, Decimal("NaN"), Decimal("1"))
    event = FillEvent("f", "o", at, Decimal("2"), Decimal("3"), broker_fill_id="b")
    assert event.fill_fingerprint == FillEvent(
        "another-id", "o", at, Decimal("2"), Decimal("3"), broker_fill_id="b"
    ).fill_fingerprint


def test_dataset_wall_rejects_forecast_to_action() -> None:
    from src.orchestration.dataset_uri import DatasetContractError, validate_dataset_edges

    with pytest.raises(DatasetContractError, match="forbidden"):
        validate_dataset_edges(
            [
                {
                    "source": "forecast://zoo/prediction/v1",
                    "target": "action://sleeve/strategy_output/v2",
                }
            ]
        )


def test_bar_interval_seed_is_bidirectionally_equal_to_python_enum() -> None:
    import re

    from src.market.identity import BarInterval

    sql = (
        Path("database") / "migrations" / "072_reference_identity.sql"
    ).read_text(encoding="utf-8")
    insert = re.search(
        r"INSERT\s+INTO\s+reference\.bar_interval.*?\bVALUES\b"
        r"(?P<rows>.*?)\bON\s+CONFLICT\b",
        sql,
        flags=re.IGNORECASE | re.DOTALL,
    )
    assert insert is not None, "reference.bar_interval seed INSERT is missing"
    rows = re.findall(
        r"\('([^']+)'\s*,\s*(NULL|\d+)\s*,\s*(TRUE|FALSE)\s*\)",
        insert.group("rows"),
        flags=re.IGNORECASE,
    )
    ddl_intervals = {interval_id for interval_id, _seconds, _aware in rows}
    python_intervals = {interval.value for interval in BarInterval}

    assert ddl_intervals == python_intervals
    ddl_metadata = {
        interval_id: (seconds.upper(), calendar_aware.upper())
        for interval_id, seconds, calendar_aware in rows
    }
    assert ddl_metadata[BarInterval.M1.value] == ("60", "FALSE")


def _load_asset_pipeline_factory_with_airflow_stubs(monkeypatch):
    """Import the real DAG module without requiring the Airflow distribution."""

    import importlib.util

    airflow_module = types.ModuleType("airflow")
    airflow_module.__path__ = []
    airflow_module.DAG = object
    operators_module = types.ModuleType("airflow.operators")
    operators_module.__path__ = []
    python_module = types.ModuleType("airflow.operators.python")
    python_module.PythonOperator = object
    utils_module = types.ModuleType("airflow.utils")
    utils_module.__path__ = []
    dates_module = types.ModuleType("airflow.utils.dates")
    dates_module.days_ago = lambda _days: datetime(2026, 1, 1, tzinfo=timezone.utc)
    trigger_module = types.ModuleType("airflow.utils.trigger_rule")
    trigger_module.TriggerRule = types.SimpleNamespace(
        ALL_DONE="all_done",
        ALL_SUCCESS="all_success",
    )
    for name, module in {
        "airflow": airflow_module,
        "airflow.operators": operators_module,
        "airflow.operators.python": python_module,
        "airflow.utils": utils_module,
        "airflow.utils.dates": dates_module,
        "airflow.utils.trigger_rule": trigger_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    module_name = "_bl35_asset_pipeline_factory"
    module_path = (
        Path(__file__).resolve().parents[2]
        / "airflow"
        / "dags"
        / "asset_pipeline_factory.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_asset_pipeline_config_fails_parse_on_forbidden_dataset_edge(
    tmp_path, monkeypatch
) -> None:
    import yaml

    from src.orchestration.dataset_uri import DatasetContractError, validate_dataset_edges

    production = yaml.safe_load(
        (Path("config") / "assets" / "pipelines.yaml").read_text(encoding="utf-8")
    )
    assert production["dataset_edges"]
    validate_dataset_edges(production["dataset_edges"])

    module = _load_asset_pipeline_factory_with_airflow_stubs(monkeypatch)
    config_path = tmp_path / "pipelines.yaml"
    module.CONFIG_PATH = config_path
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset_edges": [
                    {
                        "source": "forecast://zoo/prediction/v1",
                        "target": "exec://broker/orders/v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DatasetContractError, match="forbidden"):
        module._load_config()

    no_edges = {"registry_root": "public/data", "assets": {}}
    config_path.write_text(yaml.safe_dump(no_edges), encoding="utf-8")
    assert module._load_config() == no_edges


def test_family_projection_repairs_after_ledger_only_retry(tmp_path) -> None:
    from src.research.qlab import FamilyStore, TrialCharge, TrialLedger

    families = FamilyStore(tmp_path / "families")
    families.declare(
        {
            "family_id": "family_a",
            "kind": "forecast",
            "cluster_id": "cluster_a",
            "asset": "asset_a",
            "question": "q",
            "bar": "b",
        }
    )
    row = TrialLedger(tmp_path / "ledger.jsonl").charge(
        TrialCharge(
            trial_id="FT-9998",
            family="family_a",
            asset="asset_a",
            cluster="cluster_a",
            kind="forecast",
            variant="v1",
            cutoff="2025-01-01T00:00:00Z",
        )
    )
    families.record_trial("family_a", row)
    families.record_trial("family_a", row)
    family = families.load("family_a")
    assert family["trials_charged"] == 1
    assert [cell["trial_id"] for cell in family["cells"]] == ["FT-9998"]


def test_catalog_backfill_inventory_includes_archived_baseline_and_trades(
    tmp_path, monkeypatch
) -> None:
    import scripts.data.backfill_catalog_facts as backfill

    public = tmp_path / "public"
    strategy_root = public / "strategies" / "archived_s"
    (strategy_root / "backtests" / "1.0.0").mkdir(parents=True)
    registry = {
        "strategies": [
            {
                "strategy_id": "archived_s",
                "asset_id": "asset_a",
                "status": "archived",
                "manifest": "strategies/archived_s/manifest.json",
            }
        ]
    }
    manifest = {
        "backtests": [
            {
                "model_version": "1.0.0",
                "year": 2025,
                "summary": "strategies/archived_s/backtests/1.0.0/summary_2025.json",
                "trades": "strategies/archived_s/backtests/1.0.0/trades_2025.json",
            }
        ]
    }
    summary = {
        "strategies": {
            "archived_s": {"sharpe": 0.5, "total_return_pct": 3.0, "n_trades": 1},
            "buy_and_hold": {"total_return_pct": 2.0},
        }
    }
    trades = {
        "trades": [
            {
                "trade_id": 1,
                "timestamp": "2025-01-01T00:00:00Z",
                "exit_timestamp": "2025-01-02T00:00:00Z",
                "side": "LONG",
                "entry_price": 10,
                "exit_price": 11,
                "pnl_usd": 1,
                "equity_at_entry": 100,
                "leverage": 1,
            }
        ]
    }
    (public / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
    (strategy_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    base = strategy_root / "backtests" / "1.0.0"
    (base / "summary_2025.json").write_text(json.dumps(summary), encoding="utf-8")
    (base / "trades_2025.json").write_text(json.dumps(trades), encoding="utf-8")
    monkeypatch.setattr(backfill, "PUBLIC", public)

    metrics, trade_facts, missing, population = backfill.inventory(public / "registry.json")
    assert not missing
    assert population == ["archived_s"]
    assert any(fact.status == "archived" for fact in metrics)
    assert any("::baseline::buy_and_hold" in fact.strategy_id for fact in metrics)
    assert len(trade_facts) == 1


def test_sql_contracts_have_no_invalid_float_isfinite_and_are_append_only() -> None:
    migrations = Path("database/migrations")
    text = "\n".join(
        (migrations / name).read_text(encoding="utf-8")
        for name in (
            "070_fabric_control_plane.sql",
            "071_forecast_schema_roles.sql",
            "073_market_quality.sql",
            "074_exec_event_sourcing.sql",
            "075_fact_position_pnl.sql",
            "076_lineage_graph.sql",
            "078_exec_reconciliation.sql",
        )
    )
    assert "isfinite(" not in text
    assert "fill_fingerprint TEXT NOT NULL UNIQUE" in text
    assert "trg_canonical_bar_immutable" in text
    assert "trg_fact_pnl_immutable" in text
    assert "trg_exec_reconciliation_immutable" in text
