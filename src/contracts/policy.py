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

#: The FOUR engines of invariant 1 (`.claude/rules/strategy-engines.md`):
#: rule_based | ml | rl | composite. ``rl`` was missing in the R1 cut while the
#: always-loaded rule and spec §2/§12 (PPO USD/COP) list it, and BL-46 R5
#: requires an RLPolicyPanel — an engine the contract rejects can never reach
#: the renderer. Added bilaterally (mirrored in policy.contract.ts).
ENGINE_TYPES = ("rule_based", "ml", "rl", "composite")

VALID_DIRECTIONS = ("LONG", "SHORT", "FLAT")

#: Evaluation modes a PolicyContext can run under (fail-closed whitelist —
#: mirrored as POLICY_MODES in policy.contract.ts).
POLICY_MODES = ("DECISION", "FREEZE", "REVALIDATE", "BACKFILL")


# ---------------------------------------------------------------------------
# Strict form validators (C-004 remedy-3 finding 4: policy_hash=True,
# sleeve_id=None, as_of=None must be TYPED errors, never truthiness passes;
# remedy-4 divergence 2: re.fullmatch ONLY — ``$`` with re.match accepts a
# trailing '\n' — and a REAL calendar/clock, not just the textual form:
# 2026-02-30, 25:00 and +25:00 are impossible values, never accepted).
# Mirrored as HASH_PATTERN / ID_PATTERN / ISO_TIMESTAMP_PATTERN in
# policy.contract.ts — change BOTH sides.
# ---------------------------------------------------------------------------

#: ``sha256:<lowercase-hex>`` (8..64 hex chars; full fingerprints use 64).
HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{8,64}$")

#: Identifier form for sleeve_id / snapshot ids / versions.
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")

#: ISO-8601 date or datetime FORM (the only accepted ``as_of`` shape).
#: Groups: 1=year 2=month 3=day 4=hour 5=minute 6=second 7=offset.
#: The form is necessary but NOT sufficient — require_iso_timestamp also
#: validates the real calendar/clock/offset ranges.
ISO_TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})"
    r"(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.\d{1,6})?)?"
    r"(Z|[+-]\d{2}:\d{2})?)?$"
)


def is_real_number(value: Any) -> bool:
    """True only for genuine int/float — bool is explicitly NOT a number."""
    return not isinstance(value, bool) and isinstance(value, (int, float))


def require_id(field_name: str, value: Any) -> str:
    """Non-empty identifier string; bool/None/any-other-type => ValueError."""
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a non-empty identifier string "
            f"(pattern {ID_PATTERN.pattern!r}), got {value!r}"
        )
    return value


def require_hash(field_name: str, value: Any) -> str:
    """'sha256:<hex>' string; bool/None/malformed => ValueError."""
    if not isinstance(value, str) or not HASH_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a 'sha256:<hex>' string "
            f"(pattern {HASH_PATTERN.pattern!r}), got {value!r}"
        )
    return value


def require_iso_timestamp(field_name: str, value: Any) -> str:
    """
    ISO-8601 date/datetime string validated against the REAL calendar and
    clock (C-004 remedy-4 divergence 2): 2026-02-30, hour 25, minute 61 and
    UTC offset +25:00 are ValueErrors even though they match the textual
    form. bool/None/malformed => ValueError.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be an ISO-8601 date/datetime string, got {value!r}"
        )
    m = ISO_TIMESTAMP_PATTERN.fullmatch(value)
    if m is None:
        raise ValueError(
            f"{field_name} must be an ISO-8601 date/datetime string, got {value!r}"
        )
    year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        _dt.date(year, month, day)  # real calendar incl. leap years
    except ValueError as exc:
        raise ValueError(
            f"{field_name} has an impossible calendar date: {value!r}"
        ) from exc
    if m.group(4) is not None:
        hour, minute = int(m.group(4)), int(m.group(5))
        second = int(m.group(6)) if m.group(6) is not None else 0
        if hour > 23 or minute > 59 or second > 59:
            raise ValueError(
                f"{field_name} has an impossible time of day: {value!r}"
            )
    offset = m.group(7)
    if offset is not None and offset != "Z":
        offset_hour, offset_minute = int(offset[1:3]), int(offset[4:6])
        if offset_hour > 23 or offset_minute > 59:
            raise ValueError(
                f"{field_name} has an impossible UTC offset: {value!r}"
            )
    return value


@dataclass(frozen=True)
class EngineRef:
    """
    Discriminated engine reference (spec §4).

    - ``rule_based``: policy_hash required, model snapshots forbidden.
    - ``ml``: model_snapshot_id required.
    - ``rl``: model_snapshot_id required (the learned policy artifact has
      trained weights, exactly like ``ml``); policy_hash optional.
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
        elif self.type in ("ml", "rl"):
            if not self.model_snapshot_id:
                raise ValueError(
                    f"{self.type} engine_ref requires model_snapshot_id"
                )
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
        # Derived id (C-004 remedy-4 divergence 2d): the composite form is
        # '<sleeve_id>:<as_of>:<fingerprint-hex16>'. Every part is already
        # validated above (sleeve_id identifier, as_of real ISO incl. offset,
        # fingerprint sha256:<hex>), so a supplied signal_id must EQUAL its
        # derivation — an id whose embedded timestamp carries an impossible
        # offset, a foreign sleeve or a foreign fingerprint prefix is rejected.
        derived_signal_id = (
            f"{self.sleeve_id}:{self.as_of}:"
            f"{self.decision_fingerprint.removeprefix('sha256:')[:16]}"
        )
        if self.signal_id:
            if self.signal_id != derived_signal_id:
                raise ValueError(
                    "signal_id must equal its derivation "
                    f"'<sleeve_id>:<as_of>:<fingerprint-hex16>' = "
                    f"{derived_signal_id!r}, got {self.signal_id!r}"
                )
        else:
            object.__setattr__(self, "signal_id", derived_signal_id)

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
        # No ``default=`` fallback (C-004 remedy-4 divergence 4): a non-JSON
        # type reaching the fingerprint payload is a TypeError, never a
        # silently-stringified value.
        canonical = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
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
        No ``default=`` fallback (remedy-4 divergence 4): numpy.inf or
        Decimal('NaN') raise (TypeError/ValueError) — they are NEVER
        serialized as text like '"inf"'.
        """
        payload = self.to_dict()
        ensure_json_safe(payload, "strategy_decision")
        return json.dumps(payload, allow_nan=False)


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
