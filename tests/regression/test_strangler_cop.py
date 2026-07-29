"""Regression: the USD/COP strangler migration control plane (BL-31, FABRIC §29 + §30).

Contract: CTR-STRANGLER-COP-001

What is being locked down here is *the order and the brakes*, not a model. Three classes
of failure would be invisible without these tests:

1. A layer switching while its predecessor is still artisanal (§29.2).
2. "Parity" declared on a window shorter than two weeks, or on a streak that a mismatch
   already broke (§29.1) — "mostly green" is not sustained green.
3. L7 (real money) moving on parity alone, without the month of green chain, the external
   execution service and the 15 acceptance criteria of §30 (§29.4).

Every gate is fail-closed: absent evidence blocks, it never waves through.

FALSE-GREEN WARNING (INTEGRATION-CONTRACT.md F-01): this module imports
``src.strangler.parity``, which imports ``src.identity.canonical`` — a module
owned by CODEX (BL-17) that is **not committed**. Everything below therefore
passes only because that untracked directory exists on this disk. Two brakes
make that impossible to report as a clean green:

* on a clean clone the import raises loudly (guard in ``src/strangler/parity.py``),
  so this file is a collection ERROR, never a silent skip;
* in this working copy :func:`test_strangler_dependency_is_committed` below fails,
  so the suite can never be quoted as "N/N passed" while the dependency is
  untracked.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.strangler import (
    LAYER_ORDER,
    MONEY_LAYER,
    AcceptanceAttestation,
    CriterionStatus,
    ExecutionReadiness,
    HashKind,
    LayerState,
    LayerTransition,
    MigrationLayer,
    ParityLedger,
    ParityVerdict,
    StranglerContractError,
    derive_states,
    evaluate_advance,
    green_streak,
    hash_artifact,
    layer_status,
    load_plan,
    observe_parity,
    parity_table,
    parse_plan,
    summarize,
)
from src.strangler.contracts import RollbackPlan, SensorMigration
from src.strangler.plan import DEFAULT_PLAN_PATH

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


# ------------------------------------------------------- anti-false-green brake


def test_strangler_dependency_is_committed() -> None:
    """This suite's result is only meaningful if its dependencies are in git.

    EXPECTED RED until CODEX commits ``src/identity/`` (INTEGRATION-CONTRACT F-01,
    TDD-GAPS G-01). Do NOT delete or skip this: without it, the 38 green tests
    below can be quoted as evidence that code compiles in a clean clone, which
    is false. The gate logic lives in one place —
    ``tests/regression/test_repo_self_contained.py`` — this is the local brake.
    """
    from tests.regression.test_repo_self_contained import orphan_src_imports

    strangler_orphans = [
        (path, lineno, module)
        for path, lineno, module in orphan_src_imports()
        if path.startswith("src/strangler/")
    ]
    assert not strangler_orphans, (
        "the strangler package imports modules that are NOT in git, so every "
        f"green below is a false green: {strangler_orphans}"
    )


# --------------------------------------------------------------------------- helpers


def _plan():
    return load_plan(DEFAULT_PLAN_PATH)


def _ledger(tmp_path: Path) -> ParityLedger:
    return ParityLedger(tmp_path / "parity_ledger.jsonl")


def _write_json(path: Path, payload) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _green_run(
    ledger: ParityLedger,
    layer: MigrationLayer,
    *,
    days: int,
    count: int,
    start: datetime = T0,
) -> None:
    """Append `count` MATCH observations spread over `days` days."""
    step = timedelta(days=days / max(count - 1, 1))
    for i in range(count):
        ledger.append(
            observe_parity(
                layer=layer,
                artifact_id=f"probe-{i}",
                legacy_path=_LEGACY,
                candidate_path=_CANDIDATE,
                required_hash_kind=HashKind.CANONICAL_JSON,
                observed_at=start + step * i,
            )
        )


def _migrate_through(
    ledger: ParityLedger, upto: MigrationLayer, *, at: datetime = T0
) -> None:
    """Record MIGRATED transitions for every layer strictly before `upto`."""
    for i, layer in enumerate(LAYER_ORDER):
        if layer is upto:
            break
        ledger.append(
            LayerTransition(
                layer=layer,
                state=LayerState.MIGRATED,
                at=at + timedelta(minutes=i),
                reason="test fixture",
            )
        )


@pytest.fixture(autouse=True)
def _artifacts(tmp_path_factory):
    """Two byte-different but semantically identical JSON artifacts, plus a divergent one."""
    global _LEGACY, _CANDIDATE, _DIVERGENT
    base = tmp_path_factory.mktemp("artifacts")
    _LEGACY = _write_json(base / "legacy.json", {"b": 2, "a": 1, "n": 1.5})
    # same content, different key order and whitespace: canonical parity must still hold
    (base / "candidate.json").write_text(
        '{\n  "a": 1,\n  "n": 1.50,\n  "b": 2\n}\n', encoding="utf-8"
    )
    _CANDIDATE = base / "candidate.json"
    _DIVERGENT = _write_json(base / "divergent.json", {"a": 1, "b": 3, "n": 1.5})
    yield


# ------------------------------------------------------------------- plan invariants


def test_plan_declares_the_nine_layers_in_order():
    plan = _plan()
    assert tuple(lp.layer for lp in plan.layers) == LAYER_ORDER
    assert LAYER_ORDER[-1] is MONEY_LAYER, "L7 must be last by construction (§29.4)"


def test_plan_layer_anchors_point_at_real_files():
    """An anchor to code that no longer exists means the plan describes a dead system."""
    dead = []
    for layer_plan in _plan().layers:
        for anchor in layer_plan.legacy_anchors:
            path = ROOT / anchor.split("::", 1)[0]
            if not path.exists():
                dead.append(f"{layer_plan.layer}: {anchor}")
    assert not dead, f"strangler plan anchors to paths that do not exist: {dead}"


def test_every_layer_declares_a_rollback_without_data_loss():
    for layer_plan in _plan().layers:
        assert layer_plan.rollback.steps, f"{layer_plan.layer} has no rollback (§29.6)"
        assert layer_plan.rollback.data_loss is False


def test_no_layer_claims_byte_parity_over_external_formats():
    """§31 explicitly rejects byte parity over formats we do not write."""
    offenders = [
        str(lp.layer)
        for lp in _plan().layers
        if lp.required_hash_kind is not HashKind.CANONICAL_JSON
    ]
    assert not offenders, f"layers requiring non-canonical parity: {offenders}"


def test_plan_has_no_candidate_generator_yet_and_says_so():
    """Honest as-built: BL-28 has not delivered the generators, so nothing may advance."""
    plan = _plan()
    assert all(lp.candidate_generator is None for lp in plan.layers)
    ledger = ParityLedger(ROOT / "does-not-exist" / "empty.jsonl")
    for layer_plan in plan.layers:
        decision = evaluate_advance(plan, ledger, layer_plan.layer, now=T0)
        assert not decision.allowed
    first = evaluate_advance(plan, ledger, MigrationLayer.INGEST, now=T0)
    assert any("BL-28" in reason for reason in first.reasons)


def test_signal_layer_declares_the_real_sensor_to_asset_swap():
    """§29.3 — the only ExternalTaskSensor in the COP H5 chain, swapped when SIGNAL moves."""
    layer_plan = _plan().layer_plan(MigrationLayer.SIGNAL)
    assert layer_plan.sensor_migrations, "SIGNAL must declare the sensor -> Asset swap"
    sensor = layer_plan.sensor_migrations[0]
    source = (ROOT / sensor.sensor_ref.split("::", 1)[0]).read_text(encoding="utf-8")
    assert "ExternalTaskSensor" in source
    assert sensor.external_dag_id in source
    assert sensor.replacement_asset.startswith("asset://")


def test_plan_lists_the_fifteen_acceptance_criteria():
    ids = sorted(c.id for c in _plan().acceptance_criteria)
    assert ids == list(range(1, 16))


def test_plan_keeps_the_live_ab_cohort_untouched():
    """§29.5 — the A/B ledgers are the reference pattern, not migration targets."""
    assert _plan().ab_cohort, "the A/B cohort must be declared so it is visibly off-limits"


@pytest.mark.parametrize(
    "mutation, expected",
    [
        ({"contract_id": "CTR-WRONG-001"}, "contract_id"),
        ({"version": "1.0"}, "semver"),
        ({"execute_requires_days_after_last_migration": 7}, "30 days"),
        ({"unknown_key": 1}, "unknown key"),
    ],
)
def test_plan_loader_is_fail_closed(mutation, expected):
    import yaml

    payload = yaml.safe_load(DEFAULT_PLAN_PATH.read_text(encoding="utf-8"))
    payload.update(mutation)
    with pytest.raises(StranglerContractError):
        parse_plan(payload)


def test_plan_rejects_a_reordered_chain():
    import yaml

    payload = yaml.safe_load(DEFAULT_PLAN_PATH.read_text(encoding="utf-8"))
    payload["layers"][0], payload["layers"][1] = payload["layers"][1], payload["layers"][0]
    with pytest.raises(StranglerContractError, match="in order"):
        parse_plan(payload)


def test_plan_rejects_a_parallel_window_under_two_weeks():
    import yaml

    payload = yaml.safe_load(DEFAULT_PLAN_PATH.read_text(encoding="utf-8"))
    payload["layers"][0]["min_parallel_days"] = 3
    with pytest.raises(StranglerContractError, match="29.1"):
        parse_plan(payload)


def test_rollback_with_data_loss_is_unrepresentable():
    with pytest.raises(StranglerContractError, match="data loss"):
        RollbackPlan(trigger="x", steps=("a", "b"), data_loss=True)


def test_sensor_migration_requires_an_asset_uri():
    with pytest.raises(StranglerContractError, match="asset://"):
        SensorMigration(
            sensor_ref="airflow/dags/x.py::wait",
            external_dag_id="dag",
            replacement_asset="dataset:usdcop/train",
        )


# ------------------------------------------------------------------- parity harness


def test_canonical_parity_ignores_formatting_but_not_content():
    legacy_hash, kind = hash_artifact(_LEGACY)
    candidate_hash, _ = hash_artifact(_CANDIDATE)
    divergent_hash, _ = hash_artifact(_DIVERGENT)
    assert kind is HashKind.CANONICAL_JSON
    assert legacy_hash == candidate_hash, "key order/whitespace must not break parity"
    assert legacy_hash != divergent_hash


def test_non_json_artifact_cannot_satisfy_a_canonical_layer(tmp_path):
    parquet_like = tmp_path / "bars.parquet"
    parquet_like.write_bytes(b"PAR1\x00binary")
    observation = observe_parity(
        layer=MigrationLayer.CANON,
        artifact_id="bars",
        legacy_path=parquet_like,
        candidate_path=parquet_like,
        required_hash_kind=HashKind.CANONICAL_JSON,
        observed_at=T0,
    )
    assert observation.verdict is ParityVerdict.INVALID
    assert "8.2" in observation.note


def test_missing_artifact_is_invalid_not_green(tmp_path):
    observation = observe_parity(
        layer=MigrationLayer.SIGNAL,
        artifact_id="week=2026-W01",
        legacy_path=_LEGACY,
        candidate_path=tmp_path / "nope.json",
        required_hash_kind=HashKind.CANONICAL_JSON,
        observed_at=T0,
    )
    assert observation.verdict is ParityVerdict.INVALID


def test_non_finite_json_is_refused_rather_than_hashed(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"pf": Infinity}', encoding="utf-8")
    observation = observe_parity(
        layer=MigrationLayer.GATE,
        artifact_id="summary",
        legacy_path=bad,
        candidate_path=bad,
        required_hash_kind=HashKind.CANONICAL_JSON,
        observed_at=T0,
    )
    assert observation.verdict is ParityVerdict.INVALID
    assert "canonicaliz" in observation.note


def test_observation_cannot_lie_about_its_own_hashes():
    from src.strangler.contracts import ParityObservation

    with pytest.raises(StranglerContractError, match="contradicts"):
        ParityObservation(
            layer=MigrationLayer.SIGNAL,
            artifact_id="x",
            observed_at=T0,
            legacy_hash="sha256:" + "a" * 64,
            candidate_hash="sha256:" + "b" * 64,
            hash_kind=HashKind.CANONICAL_JSON,
            verdict=ParityVerdict.MATCH,
        )


def test_ledger_is_append_only_and_json_safe(tmp_path):
    ledger = _ledger(tmp_path)
    _green_run(ledger, MigrationLayer.INGEST, days=1, count=2)
    raw = (tmp_path / "parity_ledger.jsonl").read_text(encoding="utf-8")
    assert raw.count("\n") == 2
    for token in ("NaN", "Infinity", "-Infinity"):
        assert token not in raw
    # reopening sees the same records; appending never rewrites
    reopened = ParityLedger(tmp_path / "parity_ledger.jsonl")
    assert len(reopened.observations(MigrationLayer.INGEST)) == 2


def test_a_mismatch_resets_the_green_streak(tmp_path):
    ledger = _ledger(tmp_path)
    _green_run(ledger, MigrationLayer.TRAIN, days=20, count=5)
    assert green_streak(ledger.observations(MigrationLayer.TRAIN))[0] == 5
    ledger.append(
        observe_parity(
            layer=MigrationLayer.TRAIN,
            artifact_id="probe-late",
            legacy_path=_LEGACY,
            candidate_path=_DIVERGENT,
            required_hash_kind=HashKind.CANONICAL_JSON,
            observed_at=T0 + timedelta(days=21),
        )
    )
    assert green_streak(ledger.observations(MigrationLayer.TRAIN))[0] == 0


def test_an_invalid_observation_resets_the_streak_as_hard_as_a_mismatch(tmp_path):
    """§29.1 — an INVALID compared NOTHING (missing artifact, non-canonicalizable
    JSON, wrong hash kind). Letting it ride the streak would let a layer reach
    PARITY_GREEN on observations where no comparison ever happened.

    RED con: src/strangler/parity.py::green_streak
    `if obs.verdict is ParityVerdict.MATCH:` -> `if obs.verdict is not ParityVerdict.MISMATCH:`
    """
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    _migrate_through(ledger, MigrationLayer.CANON)
    # 4 green probes spread over the full 14-day window (days 0, 5, 10, 15)...
    _green_run(ledger, MigrationLayer.CANON, days=15, count=4)
    # ...with one observation that compared nothing dropped in the MIDDLE (day 8).
    invalid = observe_parity(
        layer=MigrationLayer.CANON,
        artifact_id="probe-vanished",
        legacy_path=_LEGACY,
        candidate_path=tmp_path / "candidate-was-never-written.json",
        required_hash_kind=HashKind.CANONICAL_JSON,
        observed_at=T0 + timedelta(days=8),
    )
    assert invalid.verdict is ParityVerdict.INVALID
    ledger.append(invalid)

    observations = ledger.observations(MigrationLayer.CANON)
    assert len(observations) == 5
    # Only the 2 probes AFTER the INVALID survive; the run before it is void.
    assert green_streak(observations)[0] == 2

    # And the same INVALID at the END leaves nothing standing at all.
    ledger.append(
        observe_parity(
            layer=MigrationLayer.CANON,
            artifact_id="probe-vanished-late",
            legacy_path=_LEGACY,
            candidate_path=tmp_path / "still-not-written.json",
            required_hash_kind=HashKind.CANONICAL_JSON,
            observed_at=T0 + timedelta(days=16),
        )
    )
    assert green_streak(ledger.observations(MigrationLayer.CANON))[0] == 0


def test_a_layer_cannot_reach_parity_green_over_invalid_observations(tmp_path):
    """The gate-level consequence: with an INVALID inside the window the layer
    must stay PARALLEL and refuse to advance — declaring parity without having
    compared anything is the exact false-green this harness exists to prevent.

    RED con: src/strangler/parity.py::green_streak
    `if obs.verdict is ParityVerdict.MATCH:` -> `if obs.verdict is not ParityVerdict.MISMATCH:`
    """
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    _migrate_through(ledger, MigrationLayer.CANON)
    _green_run(ledger, MigrationLayer.CANON, days=15, count=4)
    ledger.append(
        observe_parity(
            layer=MigrationLayer.CANON,
            artifact_id="probe-vanished",
            legacy_path=_LEGACY,
            candidate_path=tmp_path / "candidate-was-never-written.json",
            required_hash_kind=HashKind.CANONICAL_JSON,
            observed_at=T0 + timedelta(days=8),
        )
    )
    now = T0 + timedelta(days=16)
    status = layer_status(plan, ledger, MigrationLayer.CANON, now=now)
    assert status.state is not LayerState.PARITY_GREEN, (
        "a layer reached PARITY_GREEN over an observation that compared nothing"
    )
    assert status.state is LayerState.PARALLEL
    decision = evaluate_advance(plan, ledger, MigrationLayer.CANON, now=now)
    assert not decision.allowed, decision.reasons


# ------------------------------------------------------------------------- the gates


def _unlock_generators(plan):
    """Simulate BL-28 having delivered the generators (it has not)."""
    from dataclasses import replace

    layers = tuple(
        replace(lp, candidate_generator=f"generated__usdcop__{lp.layer}") for lp in plan.layers
    )
    return replace(plan, layers=layers)


def test_predecessor_must_be_migrated_first(tmp_path):
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    _green_run(ledger, MigrationLayer.FEATURES, days=30, count=6)
    decision = evaluate_advance(plan, ledger, MigrationLayer.FEATURES, now=T0 + timedelta(days=31))
    assert not decision.allowed
    assert any("predecessor" in reason for reason in decision.reasons)


def test_parity_window_shorter_than_two_weeks_blocks(tmp_path):
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    _migrate_through(ledger, MigrationLayer.CANON)
    _green_run(ledger, MigrationLayer.CANON, days=10, count=6)
    decision = evaluate_advance(plan, ledger, MigrationLayer.CANON, now=T0 + timedelta(days=11))
    assert not decision.allowed
    assert any("29.1" in reason for reason in decision.reasons)


def test_a_layer_with_green_parity_and_migrated_predecessors_may_advance(tmp_path):
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    _migrate_through(ledger, MigrationLayer.CANON)
    _green_run(ledger, MigrationLayer.CANON, days=15, count=4)
    decision = evaluate_advance(plan, ledger, MigrationLayer.CANON, now=T0 + timedelta(days=16))
    assert decision.allowed, decision.reasons
    assert layer_status(plan, ledger, MigrationLayer.CANON, now=T0 + timedelta(days=16)).state is (
        LayerState.PARITY_GREEN
    )


def test_signal_layer_needs_its_sensor_swap_declared(tmp_path):
    """Remove the declared sensor->Asset replacement: the gate must notice (§29.3)."""
    from dataclasses import replace

    plan = _unlock_generators(_plan())
    layers = tuple(
        replace(lp, sensor_migrations=()) if lp.layer is MigrationLayer.SIGNAL else lp
        for lp in plan.layers
    )
    plan = replace(plan, layers=layers)
    ledger = _ledger(tmp_path)
    _migrate_through(ledger, MigrationLayer.SIGNAL)
    _green_run(ledger, MigrationLayer.SIGNAL, days=15, count=4)
    decision = evaluate_advance(plan, ledger, MigrationLayer.SIGNAL, now=T0 + timedelta(days=16))
    assert not decision.allowed
    assert any("29.3" in reason for reason in decision.reasons)


def _fully_green_chain(tmp_path, *, chain_age_days: int = 40):
    """Everything except L7 migrated, L7 itself parity-green. Money gate still decides."""
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    _migrate_through(ledger, MONEY_LAYER)
    _green_run(ledger, MONEY_LAYER, days=15, count=4)
    now = T0 + timedelta(days=chain_age_days)
    return plan, ledger, now


def test_money_layer_blocked_without_execution_service(tmp_path):
    """The whole chain green is NOT enough: BL-30 must attest first (§29.4)."""
    plan, ledger, now = _fully_green_chain(tmp_path)
    decision = evaluate_advance(plan, ledger, MONEY_LAYER, now=now)
    assert not decision.allowed
    assert any("BL-30" in reason for reason in decision.reasons)
    assert any("acceptance criteria" in reason for reason in decision.reasons)


def test_money_layer_blocked_while_the_chain_is_younger_than_a_month(tmp_path):
    plan, ledger, _ = _fully_green_chain(tmp_path)
    decision = evaluate_advance(plan, ledger, MONEY_LAYER, now=T0 + timedelta(days=20))
    assert not decision.allowed
    assert any("29.4" in reason and "<" in reason for reason in decision.reasons)


def test_money_layer_blocked_by_a_partial_execution_readiness(tmp_path):
    plan, ledger, now = _fully_green_chain(tmp_path)
    ledger.append(
        ExecutionReadiness(
            at=now,
            canary_flow="xauusd_canary",
            service_outside_airflow=True,
            idempotency_proven=True,
            pretrade_enforced=True,
            kill_switch_with_airflow_down=False,  # the one that matters most
            reconciliation_pre_intra_eod=True,
            attested_by="test",
        )
    )
    decision = evaluate_advance(plan, ledger, MONEY_LAYER, now=now)
    assert not decision.allowed
    assert any("kill_switch_with_airflow_down" in reason for reason in decision.reasons)


def test_money_layer_blocked_by_a_single_pending_acceptance_criterion(tmp_path):
    plan, ledger, now = _fully_green_chain(tmp_path)
    ledger.append(
        ExecutionReadiness(
            at=now,
            canary_flow="xauusd_canary",
            service_outside_airflow=True,
            idempotency_proven=True,
            pretrade_enforced=True,
            kill_switch_with_airflow_down=True,
            reconciliation_pre_intra_eod=True,
            attested_by="test",
        )
    )
    for criterion_id in range(1, 15):  # 14 of 15
        ledger.append(
            AcceptanceAttestation(
                criterion_id=criterion_id,
                status=CriterionStatus.PASS,
                at=now,
                evidence=f"evidence/{criterion_id}",
            )
        )
    decision = evaluate_advance(plan, ledger, MONEY_LAYER, now=now)
    assert not decision.allowed
    assert any("#15" in reason for reason in decision.reasons)


def test_money_layer_blocked_until_its_rollback_is_rehearsed(tmp_path):
    """All 15 criteria PASS, service attested — the unrehearsed rollback is the last brake."""
    plan, ledger, now = _fully_green_chain(tmp_path)
    ledger.append(
        ExecutionReadiness(
            at=now,
            canary_flow="xauusd_canary",
            service_outside_airflow=True,
            idempotency_proven=True,
            pretrade_enforced=True,
            kill_switch_with_airflow_down=True,
            reconciliation_pre_intra_eod=True,
            attested_by="test",
        )
    )
    for criterion_id in range(1, 16):
        ledger.append(
            AcceptanceAttestation(
                criterion_id=criterion_id,
                status=CriterionStatus.PASS,
                at=now,
                evidence=f"evidence/{criterion_id}",
            )
        )
    decision = evaluate_advance(plan, ledger, MONEY_LAYER, now=now)
    assert not decision.allowed
    assert any("rehearsed" in reason for reason in decision.reasons)

    # ...and with the rehearsal recorded, the money layer finally opens.
    from dataclasses import replace

    money = plan.layer_plan(MONEY_LAYER)
    rehearsed = replace(money, rollback=replace(money.rollback, rehearsed=True))
    plan = replace(
        plan,
        layers=tuple(rehearsed if lp.layer is MONEY_LAYER else lp for lp in plan.layers),
    )
    assert evaluate_advance(plan, ledger, MONEY_LAYER, now=now).allowed


def test_a_failed_criterion_blocks_as_hard_as_a_pending_one(tmp_path):
    plan, ledger, now = _fully_green_chain(tmp_path)
    ledger.append(
        AcceptanceAttestation(
            criterion_id=10,
            status=CriterionStatus.FAIL,
            at=now,
            evidence="kill switch survived Airflow but not the broker timeout",
        )
    )
    decision = evaluate_advance(plan, ledger, MONEY_LAYER, now=now)
    assert any("#10 [FAIL]" in reason for reason in decision.reasons)


def test_rollback_is_an_event_that_the_state_projection_honours(tmp_path):
    plan = _unlock_generators(_plan())
    ledger = _ledger(tmp_path)
    ledger.append(
        LayerTransition(
            layer=MigrationLayer.INGEST, state=LayerState.MIGRATED, at=T0, reason="switch"
        )
    )
    assert derive_states(plan, ledger)[MigrationLayer.INGEST] is LayerState.MIGRATED
    ledger.append(
        LayerTransition(
            layer=MigrationLayer.INGEST,
            state=LayerState.ROLLED_BACK,
            at=T0 + timedelta(days=1),
            reason="window mismatch on 2026-01-06",
        )
    )
    states = derive_states(plan, ledger)
    assert states[MigrationLayer.INGEST] is LayerState.ROLLED_BACK
    # and the next layer is blocked again — rollback really un-migrates
    decision = evaluate_advance(plan, ledger, MigrationLayer.CANON, now=T0 + timedelta(days=2))
    assert any("predecessor" in reason for reason in decision.reasons)


# ------------------------------------------------------------------------- evidence


def test_parity_table_covers_every_layer_and_is_json_safe(tmp_path):
    plan = _plan()
    ledger = _ledger(tmp_path)
    rows = parity_table(plan, ledger, now=T0)
    assert [row["layer"] for row in rows] == [str(x) for x in LAYER_ORDER]
    assert all(row["may_advance"] is False for row in rows)
    payload = summarize(plan, ledger, now=T0)
    dumped = json.dumps(payload, allow_nan=False)  # raises on NaN/Infinity
    assert "NaN" not in dumped and "Infinity" not in dumped
    assert payload["layers_migrated"] == 0
    assert len(payload["acceptance_pending"]) == 15
    assert payload["execution_readiness"]["attested"] is False


def test_cli_status_reports_red_and_writes_evidence(tmp_path):
    import scripts.validation.check_strangler_parity as cli

    out_dir = tmp_path / "evidence"
    code = cli.main(
        [
            "--ledger",
            str(tmp_path / "ledger.jsonl"),
            "status",
            "--write-evidence",
            "--evidence-dir",
            str(out_dir),
        ]
    )
    assert code == 1, "with no candidate path the migration must report red"
    payload = json.loads((out_dir / "parity_table.json").read_text(encoding="utf-8"))
    assert len(payload["layers"]) == len(LAYER_ORDER)
    md = (out_dir / "parity_table.md").read_text(encoding="utf-8")
    assert md.startswith("---\n"), "evidence markdown must carry knowledge front matter"
    assert "kind: audit" in md


def test_cli_gate_exit_codes(tmp_path):
    import scripts.validation.check_strangler_parity as cli

    assert cli.main(["--ledger", str(tmp_path / "l.jsonl"), "gate", "--layer", "execute"]) == 1
    assert cli.main(["--ledger", str(tmp_path / "l.jsonl"), "gate", "--layer", "nope"]) == 2


def test_cli_refuses_to_record_a_migration_the_gate_blocks(tmp_path):
    import scripts.validation.check_strangler_parity as cli

    ledger_path = tmp_path / "l.jsonl"
    code = cli.main(
        [
            "--ledger",
            str(ledger_path),
            "transition",
            "--layer",
            "ingest",
            "--state",
            "MIGRATED",
            "--reason",
            "wishful thinking",
        ]
    )
    assert code == 1
    assert not ledger_path.exists(), "a refused transition must not be written"
