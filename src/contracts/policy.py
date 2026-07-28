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

import datetime as _dt
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from src.contracts.rule_trace import RuleTrace, ensure_json_safe

# ---------------------------------------------------------------------------
# Engine discriminator
# ---------------------------------------------------------------------------

ENGINE_TYPES = ("rule_based", "ml", "composite")

VALID_DIRECTIONS = ("LONG", "SHORT", "FLAT")

#: Evaluation modes a PolicyContext can run under (fail-closed whitelist —
#: mirrored as POLICY_MODES in policy.contract.ts).
POLICY_MODES = ("DECISION", "FREEZE", "REVALIDATE", "BACKFILL")


# ---------------------------------------------------------------------------
# Strict form validators (C-004 remedy-3 finding 4: policy_hash=True,
# sleeve_id=None, as_of=None must be TYPED errors, never truthiness passes).
# Mirrored as HASH_PATTERN / ID_PATTERN / ISO_TIMESTAMP_PATTERN in
# policy.contract.ts — change BOTH sides.
# ---------------------------------------------------------------------------

#: ``sha256:<lowercase-hex>`` (8..64 hex chars; full fingerprints use 64).
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{8,64}$")

#: Identifier form for sleeve_id / signal_id / snapshot ids / versions.
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

#: ISO-8601 date or datetime (the only accepted ``as_of`` form).
ISO_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2}(\.\d{1,6})?)?(Z|[+-]\d{2}:\d{2})?)?$"
)


def is_real_number(value: Any) -> bool:
    """True only for genuine int/float — bool is explicitly NOT a number."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def require_id(field_name: str, value: Any) -> str:
    """Non-empty identifier string; bool/None/any-other-type => ValueError."""
    if not isinstance(value, str) or not ID_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a non-empty identifier string "
            f"(pattern {ID_PATTERN.pattern!r}), got {value!r}"
        )
    return value


def require_hash(field_name: str, value: Any) -> str:
    """'sha256:<hex>' string; bool/None/malformed => ValueError."""
    if not isinstance(value, str) or not HASH_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a 'sha256:<hex>' string "
            f"(pattern {HASH_PATTERN.pattern!r}), got {value!r}"
        )
    return value


def require_iso_timestamp(field_name: str, value: Any) -> str:
    """ISO-8601 date/datetime string; bool/None/malformed => ValueError."""
    if not isinstance(value, str) or not ISO_TIMESTAMP_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be an ISO-8601 date/datetime string, got {value!r}"
        )
    try:
        _dt.date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(
            f"{field_name} has an invalid calendar date: {value!r}"
        ) from exc
    return value


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
        # Type-strict field forms (C-004 remedy-3 finding 4): policy_hash=True
        # or model_snapshot_id=True must raise, never pass a truthiness check.
        if self.policy_version_id is not None:
            require_id("engine_ref.policy_version_id", self.policy_version_id)
        if self.policy_hash is not None:
            require_hash("engine_ref.policy_hash", self.policy_hash)
        if self.model_snapshot_id is not None:
            require_id("engine_ref.model_snapshot_id", self.model_snapshot_id)
        if not isinstance(self.model_snapshot_ids, (tuple, list)):
            raise ValueError(
                f"engine_ref.model_snapshot_ids must be a sequence of id strings, "
                f"got {self.model_snapshot_ids!r}"
            )
        for i, snapshot_id in enumerate(self.model_snapshot_ids):
            require_id(f"engine_ref.model_snapshot_ids[{i}]", snapshot_id)
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

    def __post_init__(self) -> None:
        # Fail-closed: an unknown mode is a typed error at construction,
        # never a silently-accepted string (C-004 remedy 2).
        if self.mode not in POLICY_MODES:
            raise ValueError(
                f"context.mode must be one of {POLICY_MODES}, got {self.mode!r}"
            )
        # as_of, when provided, must already be a well-formed ISO timestamp —
        # a bool/None/garbage as_of would otherwise propagate into signal_ids
        # (C-004 remedy-3 finding 4).
        if self.as_of is not None:
            require_iso_timestamp("context.as_of", self.as_of)


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
        # Type-strict identifiers (C-004 remedy-3 finding 4): sleeve_id=None,
        # as_of=None, policy_hash=True-style values are typed errors here.
        require_id("sleeve_id", self.sleeve_id)
        require_id("strategy_version", self.strategy_version)
        if not isinstance(self.engine_ref, EngineRef):
            raise ValueError(
                f"engine_ref must be an EngineRef, got {type(self.engine_ref).__name__}"
            )
        require_iso_timestamp("as_of", self.as_of)
        if self.direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {VALID_DIRECTIONS}, got {self.direction!r}"
            )
        # Fail-closed finiteness: NaN/±Infinity never enter a decision (and
        # therefore never a JSON export — strategy-contract rule, C-004 remedy 4).
        # isinstance(bool) FIRST: bool is an int subclass and must not coerce
        # (C-004 remedy-3 finding 1).
        if isinstance(self.target_exposure, bool) or not isinstance(
            self.target_exposure, (int, float)
        ):
            raise ValueError(
                f"target_exposure must be a number, got {self.target_exposure!r}"
            )
        if not math.isfinite(self.target_exposure):
            raise ValueError(
                f"target_exposure must be finite, got {self.target_exposure!r} "
                "(NaN/Infinity forbidden — strategy-contract invariant 2)"
            )
        if not isinstance(self.reason_codes, (tuple, list)) or not all(
            isinstance(code, str) for code in self.reason_codes
        ):
            raise ValueError(
                f"reason_codes must be a sequence of strings, got {self.reason_codes!r}"
            )
        # No NaN/Infinity anywhere in the serializable payload (finding 5).
        ensure_json_safe(self.decision_components, "decision_components")
        if self.rule_trace is not None and not isinstance(self.rule_trace, RuleTrace):
            raise ValueError(
                f"rule_trace must be a RuleTrace or None, got {type(self.rule_trace).__name__}"
            )
        if self.feature_snapshot_id is not None:
            require_id("feature_snapshot_id", self.feature_snapshot_id)
        if not isinstance(self.decision_fingerprint, str):
            raise ValueError(
                f"decision_fingerprint must be a string, got {self.decision_fingerprint!r}"
            )
        if self.decision_fingerprint:
            require_hash("decision_fingerprint", self.decision_fingerprint)
        else:
            object.__setattr__(self, "decision_fingerprint", self._fingerprint())
        if not isinstance(self.signal_id, str):
            raise ValueError(f"signal_id must be a string, got {self.signal_id!r}")
        if self.signal_id:
            require_id("signal_id", self.signal_id)
        else:
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
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False
        )
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

    def to_json(self) -> str:
        """
        Strict JSON serialization: ``allow_nan=False`` means an Infinity/NaN
        that somehow reached the payload RAISES instead of emitting invalid
        JSON (repo rule: never Infinity/NaN in JSON — C-004 remedy-3 finding 5).
        """
        return json.dumps(self.to_dict(), allow_nan=False, default=str)


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
