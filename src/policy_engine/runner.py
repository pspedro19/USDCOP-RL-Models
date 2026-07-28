"""
Policy runner — the single evaluation entry point (BL-46 R4).

Why this module exists: the spec (§4, §8) requires that Airflow, the backtest,
paper and live all call the SAME ``evaluate(snapshot, context)``. Anything
that publishes a signal goes through :func:`evaluate_policy` +
:func:`publish_signal` here, so a future factory task
(``resolve_feature_snapshot -> validate_policy_inputs -> evaluate_policy ->
publish_strategy_signal``) is wiring, not a second engine.

Declared fallbacks are enforced HERE (invariant 9: no default, no freeze):

===================  ==========================================================
``FAIL_CLOSED``      invalid/missing inputs raise — nothing is published
``FLAT``             invalid/missing inputs produce an explicit FLAT decision
                     with a reason code, never a silent guess
===================  ==========================================================

Spec: .claude/specs/planes/05-rule-based-strategies.md §3.2, §4, §7, §8
Rule: .claude/rules/strategy-engines.md (invariants 4, 5, 6, 9)
Contract: CTR-POLICY-BACKEND-001
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.contracts.policy import (
    FALLBACK_MODES,
    EngineRef,
    Policy,
    PolicyContext,
    StrategyDecision,
)
from src.contracts.policy_version import (
    DECISION_SCHEMA_VERSIONS,
    PolicyVersionRecord,
    StrategySignalRecord,
)
from src.contracts.rule_trace import RuleTrace, RuleTraceEntry, ensure_json_safe
from src.strategies.policies.loader import build_policy as _loader_build_policy

#: Bumped whenever the evaluation semantics change (travels in the index).
POLICY_ENGINE_VERSION = "1.0.0"

# ``FALLBACK_MODES`` is IMPORTED above, not redefined here. It used to be a
# local tuple, and ``loader.STALE_INPUT_POLICIES`` was a second one that also
# accepted ``HOLD``: a spec declaring it passed validation and then died with
# ValueError inside ``evaluate_policy`` — a contract that validates what the
# engine cannot execute (K-034). Now
# ``policy_engine.FALLBACK_MODES is loader.STALE_INPUT_POLICIES is
# src.contracts.policy.FALLBACK_MODES`` — ONE object (same idiom as
# ``build_policy`` below), so the two gates cannot drift.
# Mirrored as FALLBACK_MODES in policy-version.contract.ts.

#: The coded_policy import allowlist is NOT redefined here: it is
#: ``loader.ALLOWED_MODULE_PREFIX`` (``src.strategies.policies.``), the narrow
#: one. A second allowlist is a second policy (INTEGRATION-CONTRACT F-09).

REASON_INPUT_MISSING = "INPUT_MISSING"
REASON_INPUT_STALE = "INPUT_STALE"


# ---------------------------------------------------------------------------
# Policy construction — DELEGATED, not reimplemented
# ---------------------------------------------------------------------------
#
# There used to be TWO ``build_policy`` functions with different semantics
# (INTEGRATION-CONTRACT.md F-09 / TDD-GAPS.md G-13): this one, and
# ``src/strategies/policies/loader.py::build_policy``. Invariant 4 of
# `.claude/rules/strategy-engines.md` — "un solo motor de evaluación; dos
# implementaciones = el backtest miente" — makes that a defect, not a
# duplication of convenience, because the two disagreed on what is even
# buildable.
#
# The loader is the SSOT. The choice is technical, not alphabetical:
#
#   1. It HONOURS ``migration.status``. A spec marked ``SPEC_ONLY`` (documento
#      de record, no runnable policy) is fail-closed there; this function
#      ignored ``migration`` entirely and happily built one.
#   2. It VERIFIES THE FREEZE: ``governance.policy_hash`` must equal the hash
#      derived from the spec's economic content, which is what makes "congelar
#      la receta ES congelar la estrategia" checkable. This function never
#      looked at it.
#   3. Its import allowlist is the NARROW one (``src.strategies.policies.``);
#      this one accepted all of ``src.strategies.`` plus ``strategies.policies.``.
#   4. It runs the full structural validation, including building the
#      declarative AST against the operator whitelist (invariant 6).
#   5. It is the one with real callers — ``scripts/validation/
#      {validate_policy_specs,check_policy_parity}.py`` and the spec test suite
#      all import it. This one had ZERO callers and was exported as the public
#      API anyway, so the LAX gate was the advertised one.
#
# Nothing is wrapped: rebinding the name keeps ONE function object, so
# ``policy_engine.build_policy is loader.build_policy`` is literally true and a
# future divergence cannot hide behind a thin adapter.

#: THE constructor. Re-exported, never reimplemented — see the note above.
build_policy = _loader_build_policy


# ---------------------------------------------------------------------------
# Evaluation (with declared fallbacks)
# ---------------------------------------------------------------------------

def _flat_decision(
    policy: Policy,
    context: PolicyContext,
    reason_code: str,
    detail: str,
) -> StrategyDecision:
    """
    Explicit FLAT decision for the ``FLAT`` fallback. Requires the policy to
    expose its identity (sleeve_id/version/policy_hash); if it does not, the
    runner FAILS CLOSED instead of inventing an attribution.
    """
    sleeve_id = getattr(policy, "sleeve_id", None)
    version = getattr(policy, "version", None)
    policy_hash = getattr(policy, "policy_hash", None)
    if not (sleeve_id and version and policy_hash):
        raise ValueError(
            "FLAT fallback needs sleeve_id/version/policy_hash on the policy; "
            f"failing closed instead of publishing an unattributable decision ({detail})"
        )
    trace = RuleTrace(
        rules=(
            RuleTraceEntry(
                rule_id="input_gate",
                label="Inputs válidos",
                observed={},
                result=False,
                reason_code=reason_code,
            ),
        ),
        fallback_applied=True,
    )
    return StrategyDecision(
        sleeve_id=sleeve_id,
        strategy_version=version,
        engine_ref=EngineRef(
            type="rule_based",
            policy_version_id=getattr(policy, "policy_version_id", None),
            policy_hash=policy_hash,
        ),
        as_of=context.as_of,
        direction="FLAT",
        target_exposure=0.0,
        reason_codes=(reason_code,),
        decision_components={"fallback_detail": detail},
        rule_trace=trace,
    )


def evaluate_policy(
    policy: Policy,
    snapshot: Mapping[str, Any],
    context: PolicyContext,
    *,
    missing_input_policy: str = "FAIL_CLOSED",
    stale_input_policy: str = "FAIL_CLOSED",
) -> StrategyDecision:
    """
    Evaluate ANY engine's policy on an EXPLICIT snapshot.

    The snapshot is passed in — this function never queries "the latest data"
    (invariant 5). Staleness is a fact the caller resolves and declares in
    ``context.extras['snapshot_is_stale']`` (a real bool or nothing).
    """
    for name, mode in (
        ("missing_input_policy", missing_input_policy),
        ("stale_input_policy", stale_input_policy),
    ):
        if mode not in FALLBACK_MODES:
            raise ValueError(f"{name} must be one of {FALLBACK_MODES}, got {mode!r}")
    if not isinstance(context, PolicyContext):
        raise ValueError(
            f"context must be a PolicyContext, got {type(context).__name__}"
        )
    if not context.as_of:
        raise ValueError("context.as_of is required (decisions never carry an empty as_of)")

    stale = context.extras.get("snapshot_is_stale", False)
    if not isinstance(stale, bool):
        raise ValueError(
            f"context.extras['snapshot_is_stale'] must be a bool, got {stale!r}"
        )
    if stale:
        if stale_input_policy == "FAIL_CLOSED":
            raise ValueError(
                "snapshot is stale and stale_input_policy=FAIL_CLOSED — not publishing"
            )
        return _flat_decision(policy, context, REASON_INPUT_STALE, "snapshot marked stale")

    errors = policy.validate_inputs(snapshot)
    if errors:
        if missing_input_policy == "FAIL_CLOSED":
            raise ValueError(f"invalid policy inputs: {'; '.join(errors)}")
        return _flat_decision(
            policy, context, REASON_INPUT_MISSING, "; ".join(errors)[:200]
        )

    return policy.evaluate(snapshot, context)


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

def publish_signal(
    decision: StrategyDecision,
    *,
    policy_version_id: str,
    instrument_id: str,
    valid_from: str,
    valid_until: str,
    created_at: str,
    rule_trace_uri: str | None = None,
    decision_schema_version: str = DECISION_SCHEMA_VERSIONS[0],
) -> StrategySignalRecord:
    """Normalize a decision into the common ``action.strategy_signal`` row."""
    return StrategySignalRecord.from_decision(
        decision,
        policy_version_id=policy_version_id,
        instrument_id=instrument_id,
        valid_from=valid_from,
        valid_until=valid_until,
        created_at=created_at,
        rule_trace_uri=rule_trace_uri,
        decision_schema_version=decision_schema_version,
    )


def write_policy_version_index(
    records: Iterable[PolicyVersionRecord],
    path: str | Path,
    *,
    generated_at: str,
) -> Path:
    """
    Write the file-based ``control.policy_version`` projection the dashboard
    API reads (``GET /api/strategies``). Lives OUTSIDE ``public/`` — the only
    access path is the RBAC-gated API (C-006 precedent).

    Fail-closed: every record is a validated :class:`PolicyVersionRecord`,
    ``policy_version_id`` is unique, and the JSON is written strictly (no
    NaN/Infinity, no ``default=`` fallback).
    """
    items = list(records)
    for i, record in enumerate(items):
        if not isinstance(record, PolicyVersionRecord):
            raise ValueError(
                f"records[{i}] must be a PolicyVersionRecord, got {type(record).__name__}"
            )
    ids = [r.policy_version_id for r in items]
    if len(set(ids)) != len(ids):
        raise ValueError("policy_version_id must be unique in the index")
    payload = {
        "schema": "policy_version_index_v1",
        "policy_engine_version": POLICY_ENGINE_VERSION,
        "generated_at": generated_at,
        "versions": [r.to_dict() for r in items],
    }
    ensure_json_safe(payload, "policy_version_index")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return out
