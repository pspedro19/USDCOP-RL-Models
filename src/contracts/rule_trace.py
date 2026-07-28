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

from dataclasses import dataclass, field
from typing import Any

RULE_TRACE_SCHEMA_V1 = "rule_trace_v1"


@dataclass(frozen=True)
class RuleTraceEntry:
    """One evaluated condition inside a policy run."""

    rule_id: str
    label: str
    observed: dict[str, Any] = field(default_factory=dict)
    result: bool = False
    reason_code: str = ""

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_schema": self.trace_schema,
            "rules": [r.to_dict() for r in self.rules],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RuleTrace":
        schema = payload.get("trace_schema", RULE_TRACE_SCHEMA_V1)
        if schema != RULE_TRACE_SCHEMA_V1:
            raise ValueError(f"Unsupported trace_schema: {schema!r}")
        entries = tuple(
            RuleTraceEntry(
                rule_id=str(r["rule_id"]),
                label=str(r.get("label", r["rule_id"])),
                observed=dict(r.get("observed", {})),
                result=bool(r.get("result", False)),
                reason_code=str(r.get("reason_code", "")),
            )
            for r in payload.get("rules", [])
        )
        return cls(rules=entries)
