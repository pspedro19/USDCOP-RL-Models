"""Base class for coded policies (spec §3.2 mode A).

Gives every coded policy the SAME strictness the declarative engine already
has (``src/contracts/policy_dsl.py``): required features present, every
snapshot value a real finite number, explicit fallback, deterministic
``StrategyDecision``. Subclasses implement one method — ``decide`` — and get
validation, fallbacks, tracing and fingerprinting for free.

Why coded and not declarative: the whitelist DSL has comparison/logic
operators only, by design. Vol targeting is arithmetic
(``exposure = target_vol / realized_vol``, clipped), so the sized policies
(Gold, BTC) MUST be mode A. That is the escape hatch §3.2 sanctions; the
alternative — adding arithmetic operators to the DSL — would widen the
attack surface of the YAML for no gain.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from src.contracts.policy import (
    EngineRef,
    PolicyContext,
    StrategyDecision,
)
from src.contracts.rule_trace import RuleTrace, RuleTraceEntry


class CodedPolicy:
    """Implements the ``Policy`` protocol; subclasses provide ``decide``."""

    #: Features the subclass reads and that MUST be in the snapshot.
    required: tuple[str, ...] = ()
    #: Features read with a declared default when absent (mirrors ``df.get``).
    optional_defaults: dict[str, float] = {}

    def __init__(self, spec: Mapping[str, Any]):
        from src.strategies.policies.loader import canonical_policy_hash

        self.spec = dict(spec)
        self.sleeve_id = str(spec["id"])
        self.version = str(spec["version"])
        self.engine_type = str(spec["engine"]["type"])
        self.params: dict[str, Any] = dict(spec["policy"].get("params", {}))
        resolution = spec["policy"]["resolution"]
        self.default_exposure = float(resolution["default_target_exposure"])
        self.default_direction = str(resolution.get("default_direction", "FLAT"))
        self.default_reason_code = str(
            resolution.get("default_reason_code", "NO_RULE_MATCHED")
        )
        self.missing_input_policy = str(spec["policy"]["missing_input_policy"])
        self.stale_input_policy = str(spec["policy"]["stale_input_policy"])
        self.policy_hash = canonical_policy_hash(spec)
        self.policy_version_id = (spec.get("governance") or {}).get("policy_version_id")
        declared_required = list(spec["inputs"].get("required_features", []))
        if declared_required and tuple(declared_required) != tuple(self.required):
            raise ValueError(
                f"{self.sleeve_id}: inputs.required_features del spec "
                f"{declared_required} no coincide con las que la política lee "
                f"{list(self.required)} — el spec y el código deben decir lo mismo"
            )
        declared_optional = list(spec["inputs"].get("optional_features", []))
        if sorted(declared_optional) != sorted(self.optional_defaults):
            raise ValueError(
                f"{self.sleeve_id}: inputs.optional_features {declared_optional} no "
                f"coincide con {sorted(self.optional_defaults)}"
            )

    # --- Policy protocol ---------------------------------------------------

    def required_features(self) -> list[str]:
        return list(self.required)

    def validate_inputs(self, snapshot: Mapping[str, Any]) -> list[str]:
        if not isinstance(snapshot, Mapping):
            return [f"feature_snapshot must be a mapping, got {type(snapshot).__name__}"]
        errors: list[str] = []
        for feature in self.required:
            if feature not in snapshot:
                errors.append(f"Missing required feature: {feature}")
        for feature, value in snapshot.items():
            if value is None:
                errors.append(f"Feature {feature} is null")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"Feature {feature} must be a real number, "
                    f"got {type(value).__name__} (bool/string coercion forbidden)"
                )
            elif not math.isfinite(value):
                errors.append(
                    f"Feature {feature} is not finite ({value!r}) — NaN/Infinity forbidden"
                )
        return errors

    def evaluate(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        if not context.as_of:
            raise ValueError(
                f"context.as_of is required to evaluate {self.sleeve_id} "
                "(fail-closed: decisions never carry an empty as_of)"
            )
        errors = self.validate_inputs(snapshot)
        if errors:
            if self.missing_input_policy == "FAIL_CLOSED":
                raise ValueError(
                    f"Invalid inputs for {self.sleeve_id}: {'; '.join(errors)}"
                )
            # FLAT: la política declara que un input ausente/no finito se
            # resuelve como exposición cero, nunca como una posición inventada.
            return self._decision(
                context,
                direction="FLAT",
                exposure=0.0,
                reason_codes=("MISSING_INPUT_FLAT",),
                components={},
                trace=RuleTrace(rules=(), fallback_applied=True),
            )
        return self.decide(snapshot, context)

    # --- subclass hook -----------------------------------------------------

    def decide(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- helpers -----------------------------------------------------------

    def _get(self, snapshot: Mapping[str, Any], feature: str) -> float:
        if feature in snapshot:
            return float(snapshot[feature])
        if feature in self.optional_defaults:
            return float(self.optional_defaults[feature])
        raise ValueError(f"Feature {feature!r} missing from snapshot")

    def _decision(
        self,
        context: PolicyContext,
        *,
        direction: str,
        exposure: float,
        reason_codes: Sequence[str],
        components: Mapping[str, Any],
        trace: RuleTrace | None,
    ) -> StrategyDecision:
        return StrategyDecision(
            sleeve_id=self.sleeve_id,
            strategy_version=self.version,
            engine_ref=EngineRef(
                type=self.engine_type,
                policy_version_id=self.policy_version_id,
                policy_hash=self.policy_hash,
            ),
            as_of=context.as_of,
            direction=direction,
            target_exposure=float(exposure),
            reason_codes=tuple(reason_codes),
            decision_components=dict(components),
            rule_trace=trace,
        )

    @staticmethod
    def _entry(
        rule_id: str,
        label: str,
        observed: Mapping[str, Any],
        result: bool,
        reason_code: str,
        threshold: Mapping[str, Any],
    ) -> RuleTraceEntry:
        return RuleTraceEntry(
            rule_id=rule_id,
            label=label,
            observed=dict(observed),
            result=bool(result),
            reason_code=reason_code,
            threshold=dict(threshold),
        )
