"""
Policy Contract (single evaluation engine, all strategy engines)
================================================================
Every strategy — rule_based, ml or composite — implements the SAME
interface and emits the SAME ``StrategyDecision``. Execution, facts, BI
and frontend never branch on the engine (invariant 1 of
`.claude/rules/strategy-engines.md`).

Interface (spec §4)::

    Policy.required_features()               -> list[str]
    Policy.validate_inputs(snapshot)         -> list[str]   # error messages, [] = OK
    Policy.evaluate(snapshot, context)       -> StrategyDecision

Design decisions sealed here (spec §15):
- §15.1: the declarative mode uses a whitelist-AST DSL
  (`src/contracts/policy_dsl.py`), NEVER eval/exec/SQL from YAML.
- §15.2: policies MAY be stateful — ``PolicyContext.state`` is a mutable
  per-strategy dict that persists between evaluations (gold_dynamic_exit
  requires it: trailing state, streak counters). Pure policies simply
  ignore it.

Parity rule: the same ``evaluate(snapshot, context)`` runs in backtest,
paper and live; only the SOURCE of the feature snapshot changes.

Spec: .claude/specs/planes/05-rule-based-strategies.md §4, §15
Rule: .claude/rules/strategy-engines.md
Contract: CTR-POLICY-001 (BL-45 R1)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from src.contracts.rule_trace import RuleTrace

# ---------------------------------------------------------------------------
# Engine discriminator
# ---------------------------------------------------------------------------

ENGINE_TYPES = ("rule_based", "ml", "composite")

VALID_DIRECTIONS = ("LONG", "SHORT", "FLAT")


@dataclass(frozen=True)
class EngineRef:
    """
    Discriminated engine reference (spec §4).

    - ``rule_based``: policy_hash required, model snapshots forbidden.
    - ``ml``: model_snapshot_id required.
    - ``composite``: policy_hash + model_snapshot_ids (predictor components).
    """

    type: str
    policy_version_id: str | None = None
    policy_hash: str | None = None
    model_snapshot_id: str | None = None
    model_snapshot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in ENGINE_TYPES:
            raise ValueError(
                f"engine_ref.type must be one of {ENGINE_TYPES}, got {self.type!r}"
            )
        if self.type == "rule_based":
            if not self.policy_hash:
                raise ValueError("rule_based engine_ref requires policy_hash")
            if self.model_snapshot_id or self.model_snapshot_ids:
                raise ValueError(
                    "rule_based engine_ref must NOT carry model snapshots "
                    "(a rules policy has no trained weights)"
                )
        elif self.type == "ml":
            if not self.model_snapshot_id:
                raise ValueError("ml engine_ref requires model_snapshot_id")
        elif self.type == "composite":
            if not self.policy_hash:
                raise ValueError("composite engine_ref requires policy_hash")

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type}
        if self.policy_version_id:
            out["policy_version_id"] = self.policy_version_id
        if self.policy_hash:
            out["policy_hash"] = self.policy_hash
        if self.model_snapshot_id:
            out["model_snapshot_id"] = self.model_snapshot_id
        if self.model_snapshot_ids:
            out["model_snapshot_ids"] = list(self.model_snapshot_ids)
        return out


# ---------------------------------------------------------------------------
# Evaluation context (stateful policies — decision §15.2: WITH state)
# ---------------------------------------------------------------------------

@dataclass
class PolicyContext:
    """
    Context handed to every ``Policy.evaluate`` call.

    ``state`` is the per-strategy mutable store that PERSISTS between
    evaluations (trailing stops, "3 closes above the MA" counters, the
    gold_dynamic_exit case). The runner owns persistence; the policy
    reads/writes the dict. Stateless policies ignore it.

    ``previous_snapshot`` enables crossing operators
    (crosses_above/crosses_below) without the policy issuing queries.
    """

    as_of: str | None = None
    mode: str = "DECISION"  # DECISION | FREEZE | REVALIDATE | BACKFILL
    state: dict[str, Any] = field(default_factory=dict)
    previous_snapshot: Mapping[str, Any] | None = None
    extras: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# StrategyDecision — the universal output of every engine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StrategyDecision:
    """
    Common decision emitted by ALL engines (spec §4 ``strategy_decision``).

    Deterministic: ``signal_id`` and ``decision_fingerprint`` are derived
    from the economic content, so same inputs + same policy => identical
    decision (CI validation §11).
    """

    sleeve_id: str
    strategy_version: str
    engine_ref: EngineRef
    as_of: str
    direction: str                      # LONG | SHORT | FLAT
    target_exposure: float
    reason_codes: tuple[str, ...] = ()
    decision_components: dict[str, Any] = field(default_factory=dict)
    rule_trace: RuleTrace | None = None
    feature_snapshot_id: str | None = None
    signal_id: str = ""                 # derived if empty
    decision_fingerprint: str = ""      # derived if empty

    def __post_init__(self) -> None:
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got {self.direction!r}"
            )
        if not self.decision_fingerprint:
            object.__setattr__(self, "decision_fingerprint", self._fingerprint())
        if not self.signal_id:
            object.__setattr__(
                self,
                "signal_id",
                f"{self.sleeve_id}:{self.as_of}:{self.decision_fingerprint[:16]}",
            )

    def _fingerprint(self) -> str:
        payload = {
            "sleeve_id": self.sleeve_id,
            "strategy_version": self.strategy_version,
            "engine_ref": self.engine_ref.to_dict(),
            "as_of": self.as_of,
            "direction": self.direction,
            "target_exposure": self.target_exposure,
            "reason_codes": list(self.reason_codes),
            "decision_components": self.decision_components,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "sleeve_id": self.sleeve_id,
            "strategy_version": self.strategy_version,
            "engine_ref": self.engine_ref.to_dict(),
            "as_of": self.as_of,
            "direction": self.direction,
            "target_exposure": self.target_exposure,
            "reason_codes": list(self.reason_codes),
            "decision_components": dict(self.decision_components),
            "rule_trace": self.rule_trace.to_dict() if self.rule_trace else None,
            "feature_snapshot_id": self.feature_snapshot_id,
            "decision_fingerprint": self.decision_fingerprint,
        }


# ---------------------------------------------------------------------------
# Policy protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Policy(Protocol):
    """
    Common interface every policy implements — coded_policy (Python class
    in Git) or declarative (DSL spec compiled by ``policy_dsl``).
    """

    def required_features(self) -> list[str]:
        """Feature names the snapshot must contain."""
        ...

    def validate_inputs(self, snapshot: Mapping[str, Any]) -> list[str]:
        """Return a list of human-readable errors; empty list = valid."""
        ...

    def evaluate(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        """Evaluate the policy on an explicit feature snapshot."""
        ...
