"""Advance gates of the strangler migration (BL-31, FABRIC §29 + §30).

One question, answered fail-closed: *may this layer switch from the artisanal path to
the candidate path right now?* Every blocker is reported (not just the first one), so an
operator sees the whole distance to green instead of peeling it one error at a time.

Nothing here is tuned. The three numbers that appear (14 days of parallel running,
30 days of green chain before the money layer, 15 acceptance criteria) are quoted from
§29.1, §29.4 and §30 respectively — they are governance, not hyperparameters.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.strangler.contracts import (
    MONEY_LAYER,
    AcceptanceCriterion,
    CriterionStatus,
    GateDecision,
    LayerState,
    LayerStatus,
    LayerTransition,
    MigrationLayer,
    ParityVerdict,
    StranglerContractError,
    StranglerPlan,
    parse_layer,
)
from src.strangler.parity import ParityLedger, green_streak


def _now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        raise StranglerContractError("`now` must be timezone-aware")
    return value.astimezone(timezone.utc)


def derive_states(plan: StranglerPlan, ledger: ParityLedger) -> dict[MigrationLayer, LayerState]:
    """Replay the ledger: last transition wins; layers without one are NOT_STARTED."""

    states = {lp.layer: LayerState.NOT_STARTED for lp in plan.layers}
    for transition in ledger.transitions():  # already sorted by time
        if transition.layer not in states:
            raise StranglerContractError(
                f"ledger references layer {transition.layer} which the plan does not declare"
            )
        states[transition.layer] = transition.state
    return states


def migrated_at(ledger: ParityLedger, layer: MigrationLayer) -> datetime | None:
    """Timestamp of the transition that left `layer` in MIGRATED, if it still is."""

    last: LayerTransition | None = None
    for transition in ledger.transitions():
        if transition.layer is layer:
            last = transition
    if last is None or last.state is not LayerState.MIGRATED:
        return None
    return last.at


def layer_status(
    plan: StranglerPlan,
    ledger: ParityLedger,
    layer: MigrationLayer | str,
    *,
    now: datetime | None = None,
) -> LayerStatus:
    lyr = parse_layer(layer)
    layer_plan = plan.layer_plan(lyr)
    when = _now(now)
    observations = ledger.observations(lyr)
    streak, since, until = green_streak(observations)
    green_days = ((until - since).total_seconds() / 86400.0) if (since and until) else 0.0

    declared_state = derive_states(plan, ledger)[lyr]
    state = declared_state
    if declared_state in (LayerState.NOT_STARTED, LayerState.PARALLEL, LayerState.PARITY_GREEN):
        if not observations:
            state = LayerState.NOT_STARTED
        elif (
            streak >= layer_plan.min_observations
            and green_days >= layer_plan.min_parallel_days
        ):
            state = LayerState.PARITY_GREEN
        else:
            state = LayerState.PARALLEL

    blockers: list[str] = []
    if not layer_plan.generator_declared:
        blockers.append(
            "no candidate generator declared (BL-28 owns it; until then there is no "
            "second path to compare against)"
        )
    if observations and streak == 0:
        last = observations[-1]
        blockers.append(
            f"green streak broken by {last.verdict} on {last.artifact_id} "
            f"at {last.observed_at.isoformat()}"
            + (f" — {last.note}" if last.note else "")
        )
    if streak and green_days < layer_plan.min_parallel_days:
        blockers.append(
            f"parallel window {green_days:.2f}d < required {layer_plan.min_parallel_days}d (§29.1)"
        )
    if streak and streak < layer_plan.min_observations:
        blockers.append(
            f"{streak} green observation(s) < required {layer_plan.min_observations}"
        )

    return LayerStatus(
        layer=lyr,
        state=state,
        observations=len(observations),
        green_streak=streak,
        green_since=since,
        last_observation_at=observations[-1].observed_at if observations else None,
        green_days=round(green_days, 4),
        blockers=tuple(blockers),
    )


def acceptance_gaps(
    plan: StranglerPlan, ledger: ParityLedger
) -> tuple[tuple[AcceptanceCriterion, CriterionStatus], ...]:
    """The §30 criteria that are not PASS yet, with their current status."""

    attested = ledger.acceptance()
    gaps: list[tuple[AcceptanceCriterion, CriterionStatus]] = []
    for criterion in sorted(plan.acceptance_criteria, key=lambda c: c.id):
        status = attested.get(criterion.id)
        current = status.status if status else CriterionStatus.PENDING
        if current is not CriterionStatus.PASS:
            gaps.append((criterion, current))
    return tuple(gaps)


def evaluate_advance(
    plan: StranglerPlan,
    ledger: ParityLedger,
    layer: MigrationLayer | str,
    *,
    now: datetime | None = None,
) -> GateDecision:
    """May `layer` be switched to the candidate path (state -> MIGRATED) right now?"""

    lyr = parse_layer(layer)
    layer_plan = plan.layer_plan(lyr)
    when = _now(now)
    states = derive_states(plan, ledger)
    reasons: list[str] = []

    if states[lyr] is LayerState.MIGRATED:
        return GateDecision(layer=lyr, allowed=False, reasons=("layer is already MIGRATED",))

    # §29.2 — nothing advances over a predecessor that has not migrated.
    for previous in plan.previous_layers(lyr):
        if states[previous] is not LayerState.MIGRATED:
            reasons.append(
                f"predecessor '{previous}' is {states[previous]}, not MIGRATED (§29.2)"
            )

    status = layer_status(plan, ledger, lyr, now=when)
    if status.state is not LayerState.PARITY_GREEN:
        reasons.extend(status.blockers)
        if not status.blockers:
            reasons.append(f"layer is {status.state}, parity window not satisfied (§29.1)")

    # §29.3 — the sensor swap must be declared before the layer moves, not after.
    if lyr in _LAYERS_WITH_LEGACY_SENSORS and not layer_plan.sensor_migrations:
        reasons.append(
            f"layer '{lyr}' is fed by an ExternalTaskSensor but declares no "
            "sensor -> Asset replacement (§29.3)"
        )

    # §29.6 — a layer without a declared rollback cannot be switched. (The contract
    # already refuses to build a RollbackPlan with data loss.)
    if not layer_plan.rollback.steps:
        reasons.append("no rollback declared for this layer (§29.6)")

    if lyr is MONEY_LAYER:
        reasons.extend(_money_layer_reasons(plan, ledger, when, states))

    return GateDecision(layer=lyr, allowed=not reasons, reasons=tuple(reasons))


#: Layers whose legacy implementation is chained by an ExternalTaskSensor today
#: (`forecast_h5_l5_weekly_signal.py::wait_for_h5_l3_training`). Verified 2026-07-28.
_LAYERS_WITH_LEGACY_SENSORS = frozenset({MigrationLayer.SIGNAL})


def _money_layer_reasons(
    plan: StranglerPlan,
    ledger: ParityLedger,
    when: datetime,
    states: dict[MigrationLayer, LayerState],
) -> list[str]:
    """§29.4 + §30: the double net demanded before real money changes hands."""

    reasons: list[str] = []
    others = [lp.layer for lp in plan.layers if lp.layer is not MONEY_LAYER]

    stamps = [migrated_at(ledger, other) for other in others]
    if any(stamp is None for stamp in stamps):
        pending = [str(o) for o, s in zip(others, stamps) if s is None]
        reasons.append(f"chain not fully migrated yet: {pending} (§29.4)")
    else:
        youngest = max(s for s in stamps if s is not None)
        age_days = (when - youngest).total_seconds() / 86400.0
        required = plan.execute_requires_days_after_last_migration
        if age_days < required:
            reasons.append(
                f"chain green for {age_days:.2f}d < required {required}d before L7 (§29.4)"
            )

    readiness = ledger.execution_readiness()
    if readiness is None:
        reasons.append(
            "no execution-readiness attestation on file — BL-30 (external execution "
            "service, pre-trade, idempotency, kill switch) must attest first (§29.4)"
        )
    else:
        if readiness.missing:
            reasons.append(
                "execution readiness incomplete: " + ", ".join(readiness.missing) + " (§29.4)"
            )
        if not readiness.canary_flow:
            reasons.append("execution readiness does not name the canary flow (§29.4)")

    gaps = acceptance_gaps(plan, ledger)
    if gaps:
        reasons.append(
            "§30 acceptance criteria not PASS: "
            + ", ".join(f"#{c.id} [{status}] (owner {c.owner})" for c, status in gaps)
        )

    money_plan = plan.layer_plan(MONEY_LAYER)
    if not money_plan.rollback.rehearsed:
        reasons.append(
            "the L7 rollback has not been rehearsed — the money layer migrates with a "
            "double net (§29.4 + §30.15)"
        )
    return reasons


def parity_table(
    plan: StranglerPlan, ledger: ParityLedger, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """The per-layer parity table BL-31 asks for as evidence."""

    when = _now(now)
    rows: list[dict[str, Any]] = []
    for layer_plan in plan.layers:
        status = layer_status(plan, ledger, layer_plan.layer, now=when)
        decision = evaluate_advance(plan, ledger, layer_plan.layer, now=when)
        rows.append(
            {
                "layer": str(layer_plan.layer),
                "state": str(status.state),
                "candidate_generator": layer_plan.candidate_generator,
                "required_hash_kind": str(layer_plan.required_hash_kind),
                "observations": status.observations,
                "green_streak": status.green_streak,
                "green_days": status.green_days,
                "min_parallel_days": layer_plan.min_parallel_days,
                "last_observation_at": (
                    status.last_observation_at.isoformat()
                    if status.last_observation_at
                    else None
                ),
                "may_advance": decision.allowed,
                "blockers": list(decision.reasons),
                "rollback_declared": bool(layer_plan.rollback.steps),
                "rollback_rehearsed": layer_plan.rollback.rehearsed,
                "sensor_migrations": [
                    {
                        "sensor_ref": sm.sensor_ref,
                        "external_dag_id": sm.external_dag_id,
                        "replacement_asset": sm.replacement_asset,
                    }
                    for sm in layer_plan.sensor_migrations
                ],
            }
        )
    return rows


def summarize(plan: StranglerPlan, ledger: ParityLedger, *, now: datetime | None = None) -> dict:
    when = _now(now)
    rows = parity_table(plan, ledger, now=when)
    gaps = acceptance_gaps(plan, ledger)
    readiness = ledger.execution_readiness()
    return {
        "asset": plan.asset,
        "contract_id": plan.contract_id,
        "plan_version": plan.version,
        "generated_at": when.isoformat().replace("+00:00", "Z"),
        "ab_cohort_untouched": list(plan.ab_cohort),
        "layers": rows,
        "layers_migrated": sum(1 for r in rows if r["state"] == str(LayerState.MIGRATED)),
        "acceptance_pending": [
            {"id": c.id, "status": str(status), "owner": c.owner} for c, status in gaps
        ],
        "execution_readiness": (
            {"attested": False, "missing": None}
            if readiness is None
            else {"attested": True, "missing": list(readiness.missing)}
        ),
    }


__all__ = [
    "acceptance_gaps",
    "derive_states",
    "evaluate_advance",
    "layer_status",
    "migrated_at",
    "parity_table",
    "summarize",
    "ParityVerdict",
]
