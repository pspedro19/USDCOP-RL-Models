"""
Rule Trace Contract (rule_trace_v1)
===================================
Structured explanation of a rule-based / composite policy evaluation.

The backend produces the trace; the frontend ONLY renders it
(invariant 7 of `.claude/rules/strategy-engines.md`: the frontend never
re-evaluates conditions).

Schema (rule_trace_v1)::

    {
      "trace_schema": "rule_trace_v1",
      "rules": [
        { "rule_id": "close_above_ma200",
          "label": "Precio sobre MA200",
          "observed": { "close": 6412.8, "ma_200": 5984.2 },
          "result": true,
          "reason_code": "CLOSE_ABOVE_MA200" }
      ]
    }

Spec: .claude/specs/planes/05-rule-based-strategies.md §9
Rule: .claude/rules/strategy-engines.md
Contract: CTR-POLICY-001 (BL-45 R1)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

RULE_TRACE_SCHEMA_V1 = "rule_trace_v1"

#: Exact whitelist of supported trace schemas (fail-closed — mirrored as
#: SUPPORTED_TRACE_SCHEMAS in policy.contract.ts). Adding a v2 means adding
#: it HERE and in the TS mirror, never accepting unknown strings.
SUPPORTED_TRACE_SCHEMAS = (RULE_TRACE_SCHEMA_V1,)


def ensure_json_safe(value: Any, path: str = "value") -> None:
    """
    Recursive CLOSED-WORLD JSON check (C-004 remedy-4 divergence 4). The
    allowed types are EXACTLY: dict (str keys) / list / tuple / str / int /
    finite float / bool / None. Everything else — numpy scalars, Decimal,
    datetime, set, bytes, ... — is a typed ValueError, INCLUDING non-finites
    that arrive as numpy/Decimal (numpy.float64 inf is caught by the
    finiteness branch because it subclasses float; numpy.float32 and
    Decimal('NaN') are caught by the closed type set). Repo rule: JSON
    exports NEVER contain Infinity/NaN, and non-JSON types are never
    silently stringified (no ``default=`` fallback anywhere in this contract).
    Mirrored as ``collectNonFinite`` in policy.contract.ts.
    """
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, float):  # includes numpy.float64 (float subclass)
        if not math.isfinite(value):
            raise ValueError(
                f"{path} contains a non-finite number ({value!r}) — "
                "NaN/Infinity forbidden in JSON exports"
            )
        return
    if isinstance(value, int):  # bool already handled above
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(
                    f"{path} has a non-string key {key!r} — JSON objects "
                    "require string keys"
                )
            ensure_json_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            ensure_json_safe(item, f"{path}[{i}]")
        return
    raise ValueError(
        f"{path} has a non-JSON-serializable type {type(value).__name__} "
        f"({value!r}) — allowed types are exactly dict/list/str/int/"
        "finite-float/bool/None (convert numpy/Decimal/datetime/set/bytes "
        "at the producer)"
    )


@dataclass(frozen=True)
class RuleTraceEntry:
    """One evaluated condition inside a policy run."""

    rule_id: str
    label: str
    observed: dict[str, Any] = field(default_factory=dict)
    result: bool = False
    reason_code: str = ""

    def __post_init__(self) -> None:
        # Type-strict: str()/bool() coercions are forbidden (C-004 remedy-3).
        if not isinstance(self.rule_id, str) or not self.rule_id:
            raise ValueError(
                f"rule_trace entry requires a non-empty string rule_id, got {self.rule_id!r}"
            )
        if not isinstance(self.label, str):
            raise ValueError(f"rule_trace entry label must be a string, got {self.label!r}")
        if not isinstance(self.observed, Mapping):
            raise ValueError(
                f"rule_trace entry observed must be a mapping, got {self.observed!r}"
            )
        ensure_json_safe(self.observed, f"rule_trace[{self.rule_id}].observed")
        if not isinstance(self.result, bool):
            raise ValueError(
                f"rule_trace entry result must be a bool, got {self.result!r}"
            )
        if not isinstance(self.reason_code, str):
            raise ValueError(
                f"rule_trace entry reason_code must be a string, got {self.reason_code!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "label": self.label,
            "observed": dict(self.observed),
            "result": self.result,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class RuleTrace:
    """Full trace of one policy evaluation (rule_trace_v1)."""

    rules: tuple[RuleTraceEntry, ...] = ()
    trace_schema: str = RULE_TRACE_SCHEMA_V1

    def __post_init__(self) -> None:
        # Fail-closed: an unsupported schema is a typed error at construction
        # (C-004 remedy 3) — not only in from_dict.
        if self.trace_schema not in SUPPORTED_TRACE_SCHEMAS:
            raise ValueError(
                f"Unsupported trace_schema: {self.trace_schema!r} "
                f"(supported: {SUPPORTED_TRACE_SCHEMAS})"
            )
        if not isinstance(self.rules, (tuple, list)) or not all(
            isinstance(r, RuleTraceEntry) for r in self.rules
        ):
            raise ValueError("rules must be a sequence of RuleTraceEntry")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema": self.trace_schema,
            "rules": [r.to_dict() for r in self.rules],
        }

    def to_json(self) -> str:
        """
        Strict JSON: an Infinity/NaN raises instead of emitting invalid JSON.
        No ``default=`` fallback (C-004 remedy-4 divergence 4): a non-JSON
        type (numpy scalar, Decimal, ...) raises instead of serializing as text.
        """
        payload = self.to_dict()
        ensure_json_safe(payload, "rule_trace")
        return json.dumps(payload, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RuleTrace":
        if not isinstance(payload, Mapping):
            raise ValueError(f"rule_trace payload must be a mapping, got {payload!r}")
        # trace_schema is REQUIRED — a payload without it is rejected on BOTH
        # sides (TS already rejected it; Python aligned in C-004 remedy-3
        # finding 7). Never defaulted.
        if "trace_schema" not in payload:
            raise ValueError(
                "rule_trace payload requires an explicit trace_schema "
                f"(supported: {SUPPORTED_TRACE_SCHEMAS}) — implicit defaults are forbidden"
            )
        schema = payload["trace_schema"]
        if schema not in SUPPORTED_TRACE_SCHEMAS:
            raise ValueError(f"Unsupported trace_schema: {schema!r}")
        rules_raw = payload.get("rules")
        if not isinstance(rules_raw, (list, tuple)):
            raise ValueError(
                f"rule_trace payload requires a 'rules' list, got {rules_raw!r}"
            )
        entries = []
        for i, r in enumerate(rules_raw):
            if not isinstance(r, Mapping) or "rule_id" not in r:
                raise ValueError(f"rule_trace rules[{i}] must be a mapping with rule_id")
            entries.append(
                RuleTraceEntry(
                    rule_id=r["rule_id"],
                    label=r.get("label", r["rule_id"]),
                    observed=dict(r.get("observed", {})),
                    result=r.get("result", False),
                    reason_code=r.get("reason_code", ""),
                )
            )
        return cls(rules=tuple(entries))
