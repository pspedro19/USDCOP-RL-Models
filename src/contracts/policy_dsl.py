"""
Declarative Policy DSL (qlab_policy_v1)
=======================================
Whitelist-AST evaluator for declarative rule specs (YAML mode B, spec §3.2).

SECURITY — non-negotiable (invariant 6 of `.claude/rules/strategy-engines.md`):
conditions are data (nested dicts), evaluated by walking a restricted AST.
The ONLY allowed operators are:

    greater_than · less_than · equal · all · any · not
    · crosses_above · crosses_below · between

There is NO eval/exec/SQL/inline-Python path — a spec carrying anything
else (an unknown operator, a raw string condition like ``"eval(...)"``)
raises ``ValueError`` at parse/evaluation time. Complex indicators live
as registered features (code_reference + code_hash); the rule only
references the feature id.

Operand grammar:
- ``"feature.<name>"``   -> value of ``snapshot["<name>"]``
- ``{"feature": "<name>"}`` -> same
- int/float literal      -> itself

Crossing operators need the previous bar: they read
``PolicyContext.previous_snapshot`` (never issue queries — invariant 5).

Spec: .claude/specs/planes/05-rule-based-strategies.md §3.2, §4
Rule: .claude/rules/strategy-engines.md
Contract: CTR-POLICY-001 (BL-45 R1)
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from src.contracts.policy import (
    VALID_DIRECTIONS,
    EngineRef,
    PolicyContext,
    StrategyDecision,
    policy_canonical_hash,
    require_hash,
)
from src.contracts.rule_trace import RuleTrace, RuleTraceEntry

# ---------------------------------------------------------------------------
# Whitelist — the ONLY operators a declarative spec may use
# ---------------------------------------------------------------------------

COMPARISON_OPERATORS = frozenset(
    {"greater_than", "less_than", "equal", "crosses_above", "crosses_below"}
)
LOGICAL_OPERATORS = frozenset({"all", "any", "not"})
RANGE_OPERATORS = frozenset({"between"})

ALLOWED_OPERATORS = COMPARISON_OPERATORS | LOGICAL_OPERATORS | RANGE_OPERATORS

_FEATURE_PREFIX = "feature."


# ---------------------------------------------------------------------------
# Operand resolution
# ---------------------------------------------------------------------------

def _feature_name(operand: Any) -> str | None:
    """Return the feature name if the operand is a feature reference."""
    if isinstance(operand, str):
        if operand.startswith(_FEATURE_PREFIX):
            name = operand[len(_FEATURE_PREFIX):]
            if not name:
                raise ValueError(
                    "Invalid operand 'feature.': the feature name must be "
                    "non-empty (C-004 remedy-3 finding 2)"
                )
            return name
        raise ValueError(
            f"Invalid operand {operand!r}: strings must be 'feature.<name>' "
            "references — arbitrary expressions/code are forbidden"
        )
    if isinstance(operand, Mapping):
        if set(operand.keys()) == {"feature"}:
            name = operand["feature"]
            # Type-strict: {"feature": true}/None/dict/number is a typed
            # error — no str() coercion (C-004 remedy-3 finding 2).
            if isinstance(name, bool) or not isinstance(name, str) or not name:
                raise ValueError(
                    f"Invalid operand mapping {operand!r}: 'feature' must be "
                    f"a non-empty string, got {name!r}"
                )
            return name
        raise ValueError(f"Invalid operand mapping {operand!r}")
    return None


def _valid_literal(operand: Any) -> bool:
    """A numeric literal operand: int/float (not bool) and FINITE (no NaN/Inf)."""
    return (
        not isinstance(operand, bool)
        and isinstance(operand, (int, float))
        and math.isfinite(operand)
    )


def _strict_exposure(value: Any, where: str) -> float:
    """
    Type-strict exposure: bool (isinstance FIRST — bool subclasses int) and
    numeric strings are typed errors, never float()-coerced
    (C-004 remedy-3 finding 1). Must also be finite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where} must be a real number "
            f"(bool/string coercion forbidden), got {value!r}"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"{where} must be finite (NaN/Infinity forbidden), got {value!r}"
        )
    return float(value)


def _canonical_policy_hash(spec: Mapping[str, Any]) -> str:
    """
    Deterministic sha256 of the canonical spec JSON.

    Delegates to the family SSOT ``policy.policy_canonical_hash`` — one logical
    object, one hash (INTEGRATION-CONTRACT.md F-02). Strict on BOTH axes:
    ``allow_nan=False`` and no ``default=`` fallback, so a spec carrying a
    non-JSON type raises instead of hashing a silently-stringified value.
    """
    return policy_canonical_hash(spec)


def _snapshot_value(name: str, snapshot: Mapping[str, Any]) -> float:
    """
    Strict snapshot read: the value must be a REAL finite number — bool,
    numeric strings, None, NaN and ±Infinity are typed errors, never
    float()-coerced (C-004 remedy-3 findings 1/3).
    """
    if name not in snapshot:
        raise ValueError(f"Feature {name!r} missing from snapshot")
    value = snapshot[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Feature {name!r} must be a real number, got {value!r} "
            "(bool/string coercion forbidden)"
        )
    if not math.isfinite(value):
        raise ValueError(
            f"Feature {name!r} is not finite ({value!r}) — NaN/Infinity forbidden"
        )
    return float(value)


def _resolve(operand: Any, snapshot: Mapping[str, Any]) -> float:
    name = _feature_name(operand)
    if name is not None:
        return _snapshot_value(name, snapshot)
    if not _valid_literal(operand):
        raise ValueError(
            f"Invalid operand {operand!r}: must be a feature reference or a "
            "finite numeric literal (NaN/Infinity forbidden)"
        )
    return float(operand)


def threshold_values(
    node: Any, snapshot: Mapping[str, Any]
) -> dict[str, float]:
    """
    Collect the RIGHT-hand (threshold) values a condition compares against —
    the "Umbral" column of the §9 table. Emitted by the BACKEND so the
    frontend never has to infer which observed value is the threshold
    (invariant 7: the UI renders, it does not re-evaluate).

    Keys: the feature name when the threshold is a feature reference,
    ``limit`` for a numeric literal, ``lower``/``upper`` for ``between``.
    Collisions inside ``all``/``any`` are suffixed (``key#2``) so the map is
    lossless and deterministic.
    """
    out: dict[str, float] = {}
    _collect_thresholds(node, snapshot, out)
    return out


def _put_threshold(out: dict[str, float], key: str, value: float) -> None:
    if key not in out:
        out[key] = value
        return
    i = 2
    while f"{key}#{i}" in out:
        i += 1
    out[f"{key}#{i}"] = value


def _threshold_key(operand: Any, fallback: str) -> str:
    name = _feature_name(operand)
    return name if name is not None else fallback


def _collect_thresholds(
    node: Any, snapshot: Mapping[str, Any], out: dict[str, float]
) -> None:
    validate_condition(node)
    operator = node["operator"]
    if operator in COMPARISON_OPERATORS:
        _put_threshold(
            out,
            _threshold_key(node["right"], "limit"),
            _resolve(node["right"], snapshot),
        )
        return
    if operator == "between":
        _put_threshold(out, "lower", _resolve(node["lower"], snapshot))
        _put_threshold(out, "upper", _resolve(node["upper"], snapshot))
        return
    if operator == "not":
        _collect_thresholds(node["condition"], snapshot, out)
        return
    if operator in ("all", "any"):
        for child in node["conditions"]:
            _collect_thresholds(child, snapshot, out)


def referenced_features(node: Any) -> set[str]:
    """Collect every feature name referenced inside a condition AST."""
    features: set[str] = set()
    _collect_features(node, features)
    return features


def _collect_features(node: Any, out: set[str]) -> None:
    # Lenient walk: only collects feature refs; structural validation is
    # validate_condition's job (operator names etc. are plain strings here).
    if isinstance(node, str):
        if node.startswith(_FEATURE_PREFIX) and node[len(_FEATURE_PREFIX):]:
            out.add(node[len(_FEATURE_PREFIX):])
        return
    if isinstance(node, Mapping):
        if set(node.keys()) == {"feature"}:
            # Only collect well-formed refs; malformed ones ({"feature": true})
            # are validate_condition's job to reject.
            name = node["feature"]
            if isinstance(name, str) and name and not isinstance(name, bool):
                out.add(name)
            return
        for key, value in node.items():
            if key != "operator":
                _collect_features(value, out)
        return
    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_features(item, out)


# ---------------------------------------------------------------------------
# AST evaluation
# ---------------------------------------------------------------------------

def validate_condition(node: Any) -> None:
    """
    Structurally validate a condition AST without evaluating it.
    Raises ValueError on any operator outside the whitelist or on any
    attempt at code execution (raw strings, eval, SQL...).
    """
    if not isinstance(node, Mapping):
        raise ValueError(
            f"Condition must be a mapping with an 'operator' key, got {node!r} "
            "— raw string/code conditions are forbidden"
        )
    operator = node.get("operator")
    if operator not in ALLOWED_OPERATORS:
        raise ValueError(
            f"Operator {operator!r} is not in the whitelist {sorted(ALLOWED_OPERATORS)}"
        )
    if operator in COMPARISON_OPERATORS:
        for key in ("left", "right"):
            if key not in node:
                raise ValueError(f"Operator {operator!r} requires {key!r}")
            _validate_operand(node[key], operator)
    elif operator == "not":
        if "condition" not in node:
            raise ValueError("Operator 'not' requires 'condition'")
        validate_condition(node["condition"])
    elif operator in ("all", "any"):
        conditions = node.get("conditions")
        if not isinstance(conditions, (list, tuple)) or not conditions:
            raise ValueError(
                f"Operator {operator!r} requires a non-empty 'conditions' list"
            )
        for child in conditions:
            validate_condition(child)
    elif operator == "between":
        for key in ("value", "lower", "upper"):
            if key not in node:
                raise ValueError(f"Operator 'between' requires {key!r}")
            _validate_operand(node[key], operator)


def _validate_operand(operand: Any, operator: Any) -> None:
    """Reject any operand that is not a feature ref or finite numeric literal."""
    # _feature_name raises on invalid string/mapping forms (e.g. "eval(...)")
    if _feature_name(operand) is None and not _valid_literal(operand):
        raise ValueError(
            f"Invalid operand {operand!r} for operator {operator!r} "
            "(feature reference or finite numeric literal only)"
        )


def evaluate_condition(
    node: Any,
    snapshot: Mapping[str, Any],
    previous_snapshot: Mapping[str, Any] | None = None,
) -> bool:
    """
    Evaluate a whitelisted condition AST against a feature snapshot.
    NEVER executes code from the spec — pure data interpretation.
    """
    validate_condition(node)
    operator = node["operator"]

    if operator == "greater_than":
        return _resolve(node["left"], snapshot) > _resolve(node["right"], snapshot)
    if operator == "less_than":
        return _resolve(node["left"], snapshot) < _resolve(node["right"], snapshot)
    if operator == "equal":
        return _resolve(node["left"], snapshot) == _resolve(node["right"], snapshot)
    if operator == "between":
        value = _resolve(node["value"], snapshot)
        lower = _resolve(node["lower"], snapshot)
        upper = _resolve(node["upper"], snapshot)
        return lower <= value <= upper
    if operator == "not":
        return not evaluate_condition(node["condition"], snapshot, previous_snapshot)
    if operator == "all":
        return all(
            evaluate_condition(c, snapshot, previous_snapshot)
            for c in node["conditions"]
        )
    if operator == "any":
        return any(
            evaluate_condition(c, snapshot, previous_snapshot)
            for c in node["conditions"]
        )
    if operator in ("crosses_above", "crosses_below"):
        if previous_snapshot is None:
            raise ValueError(
                f"Operator {operator!r} requires context.previous_snapshot"
            )
        left_now = _resolve(node["left"], snapshot)
        right_now = _resolve(node["right"], snapshot)
        left_prev = _resolve(node["left"], previous_snapshot)
        right_prev = _resolve(node["right"], previous_snapshot)
        if operator == "crosses_above":
            return left_prev <= right_prev and left_now > right_now
        return left_prev >= right_prev and left_now < right_now

    raise ValueError(f"Operator {operator!r} is not in the whitelist")  # pragma: no cover


# ---------------------------------------------------------------------------
# Declarative policy (implements the Policy protocol)
# ---------------------------------------------------------------------------

class DeclarativePolicy:
    """
    Compiles a declarative spec (YAML mode B, schema qlab_policy_v1) into a
    ``Policy``. Resolution: ``first_match`` by descending priority; if no
    rule fires, the declared default applies (invariant 9: explicit
    fallback or no freeze).

    Spec shape (already parsed from YAML)::

        {
          "id": "spx500_daily_ma200_v1",
          "version": "2.0.0",
          "policy_hash": "sha256:...",
          "resolution": {"mode": "first_match",
                         "default_target_exposure": 0.0,
                         "default_direction": "FLAT",
                         "default_reason_code": "NO_RULE_MATCHED"},
          "rules": [
            {"id": "trend_on", "label": "...", "priority": 100,
             "when": {"operator": "greater_than",
                      "left": "feature.close", "right": "feature.ma_200"},
             "output": {"direction": "LONG", "target_exposure": 1.0,
                        "reason_code": "CLOSE_ABOVE_MA200"}},
            ...
          ]
        }
    """

    def __init__(self, spec: Mapping[str, Any]):
        sleeve_id = spec.get("id")
        if isinstance(sleeve_id, bool) or not isinstance(sleeve_id, str) or not sleeve_id:
            raise ValueError(
                f"Declarative policy spec requires a non-empty string 'id', got {sleeve_id!r}"
            )
        self.sleeve_id = sleeve_id
        version = spec.get("version", "1.0.0")
        if not isinstance(version, str) or not version:
            raise ValueError(f"spec 'version' must be a non-empty string, got {version!r}")
        self.version = version
        # policy_hash: explicit hash validated for form, or the canonical hash
        # of the spec itself — never a str()-coerced placeholder (C-004
        # remedy-3 finding 4).
        policy_hash = spec.get("policy_hash")
        if policy_hash is None:
            policy_hash = _canonical_policy_hash(spec)
        require_hash("policy_hash", policy_hash)
        self.policy_hash = policy_hash
        self.policy_version_id = spec.get("policy_version_id")

        resolution = spec.get("resolution", {})
        mode = resolution.get("mode", "first_match")
        if mode != "first_match":
            raise ValueError(f"Unsupported resolution mode: {mode!r}")
        if "default_target_exposure" not in resolution:
            raise ValueError(
                "Declarative policy requires an explicit "
                "resolution.default_target_exposure (no implicit fallback)"
            )
        self.default_exposure = _strict_exposure(
            resolution["default_target_exposure"],
            "resolution.default_target_exposure",
        )
        self.default_direction = str(resolution.get("default_direction", "FLAT"))
        self.default_reason_code = str(
            resolution.get("default_reason_code", "NO_RULE_MATCHED")
        )
        if self.default_direction not in VALID_DIRECTIONS:
            raise ValueError(
                f"default_direction must be one of {VALID_DIRECTIONS}"
            )

        # Fallback DECLARADO por feature ausente (invariante 9 de `strategy-engines.md`,
        # decisión CXD-462). Existe por un caso concreto: la MA de 200 sesiones no está
        # definida en las primeras 199 barras, y el `coded_policy` vivo convierte ese
        # hueco en `0.0` silenciosamente — "no hay dato" acaba indistinguible de "la
        # política dice estar plano". Declararlo aquí conserva esa conducta, pero como
        # afirmación auditable en vez de accidente aritmético.
        #
        # Lo que este mecanismo NO hace, a propósito:
        #  * no relaja el rechazo global de NaN: sin fallback declarado para ESA feature,
        #    el `NaN` sigue siendo un error duro;
        #  * no cubre un valor corrupto. `"abc"` o `True` no son "dato ausente", son un
        #    contrato roto, y ahí no hay fallback que valga.
        self.feature_fallbacks: dict[str, dict[str, Any]] = {}
        for nombre, declarado in (resolution.get("feature_fallbacks") or {}).items():
            if not isinstance(declarado, Mapping):
                raise ValueError(
                    f"feature_fallbacks[{nombre!r}] must be a mapping with "
                    "direction/target_exposure/reason_code"
                )
            direccion = str(declarado.get("direction", "FLAT"))
            if direccion not in VALID_DIRECTIONS:
                raise ValueError(
                    f"feature_fallbacks[{nombre!r}].direction must be one of "
                    f"{VALID_DIRECTIONS}"
                )
            if "target_exposure" not in declarado:
                raise ValueError(
                    f"feature_fallbacks[{nombre!r}] requires an explicit "
                    "target_exposure (no implicit fallback)"
                )
            if not declarado.get("reason_code"):
                raise ValueError(
                    f"feature_fallbacks[{nombre!r}] requires a reason_code: una decisión "
                    "tomada por ausencia de dato debe ser legible en la traza"
                )
            self.feature_fallbacks[str(nombre)] = {
                "direction": direccion,
                "target_exposure": _strict_exposure(
                    declarado["target_exposure"],
                    f"feature_fallbacks[{nombre!r}].target_exposure",
                ),
                "reason_code": str(declarado["reason_code"]),
            }

        rules = spec.get("rules")
        if not rules:
            raise ValueError("Declarative policy requires at least one rule")
        self.rules: list[dict[str, Any]] = []
        for rule in rules:
            if "id" not in rule or "when" not in rule or "output" not in rule:
                raise ValueError(f"Rule missing id/when/output: {rule!r}")
            validate_condition(rule["when"])
            output = rule["output"]
            direction = output.get("direction")
            if direction not in VALID_DIRECTIONS:
                raise ValueError(
                    f"Rule {rule['id']!r} output.direction must be one of "
                    f"{VALID_DIRECTIONS}, got {direction!r}"
                )
            if "target_exposure" not in output:
                raise ValueError(
                    f"Rule {rule['id']!r} output requires target_exposure"
                )
            _strict_exposure(
                output["target_exposure"],
                f"Rule {rule['id']!r} output.target_exposure",
            )
            self.rules.append(dict(rule))
        # first_match by descending priority, stable on declaration order
        self.rules.sort(key=lambda r: -int(r.get("priority", 0)))

        self._required = sorted(
            set().union(*(referenced_features(r["when"]) for r in self.rules))
        )

    # --- Policy protocol -------------------------------------------------

    def required_features(self) -> list[str]:
        return list(self._required)

    def validate_inputs(self, snapshot: Mapping[str, Any]) -> list[str]:
        """
        Validate the COMPLETE snapshot, symmetric with the TS mirror's
        ``validateFeatureSnapshot`` (C-004 remedy-4 divergence 3): required
        features must be present, and EVERY entry — required or not — must
        be a real finite number. ``{"close": 1.0, "unused": inf}`` fails on
        both sides identically.
        """
        if not isinstance(snapshot, Mapping):
            return [f"feature_snapshot must be a mapping, got {type(snapshot).__name__}"]
        errors: list[str] = []
        for name in self._required:
            if name not in snapshot:
                errors.append(f"Missing required feature: {name}")
        for name, value in snapshot.items():
            if value is None:
                errors.append(f"Feature {name} is null")
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(
                    f"Feature {name} must be a real number, "
                    f"got {type(value).__name__} (bool/string coercion forbidden)"
                )
            elif not math.isfinite(value):
                # NaN AND ±Infinity are typed errors (C-004 remedy-3 finding 3)
                errors.append(
                    f"Feature {name} is not finite ({value!r}) — "
                    "NaN/Infinity forbidden"
                )
        return errors

    def _declared_absence(self, snapshot: Mapping[str, Any]) -> str | None:
        """Primera feature requerida **ausente** que tenga fallback declarado.

        "Ausente" es sólo dos cosas: no estar en el snapshot, o ser un `NaN`/±Inf. Un
        `"abc"`, un `True` o un `None` NO son ausencia: son un contrato roto, y siguen
        muriendo en `validate_inputs`. La distinción es el punto — un fallback que se
        tragara valores corruptos volvería a fabricar decisiones sobre datos basura.
        """
        if not isinstance(snapshot, Mapping) or not self.feature_fallbacks:
            return None
        for nombre in self._required:
            if nombre not in self.feature_fallbacks:
                continue
            if nombre not in snapshot:
                return nombre
            valor = snapshot[nombre]
            if isinstance(valor, float) and not math.isfinite(valor):
                return nombre
        return None

    def _fallback_decision(
        self, feature: str, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        """Decisión tomada por ausencia declarada, marcada como tal en la traza."""
        declarado = self.feature_fallbacks[feature]
        # La ausencia se registra como `null`, nunca como el `NaN` crudo: los contratos
        # de export prohíben `NaN`/`Infinity` en JSON, y además `null` **es** el hecho —
        # el valor no existía. Colar el NaN aquí habría reintroducido por la traza justo
        # lo que el DSL rechaza por la entrada.
        entrada = RuleTraceEntry(
            rule_id=f"fallback.{feature}",
            label=f"Feature '{feature}' no disponible",
            observed={feature: None},
            result=True,
            reason_code=declarado["reason_code"],
            threshold={},
        )
        return StrategyDecision(
            sleeve_id=self.sleeve_id,
            strategy_version=self.version,
            engine_ref=EngineRef(
                type="rule_based",
                policy_version_id=self.policy_version_id,
                policy_hash=self.policy_hash,
            ),
            as_of=context.as_of,
            direction=declarado["direction"],
            target_exposure=declarado["target_exposure"],
            reason_codes=(declarado["reason_code"],),
            decision_components={feature: None},
            rule_trace=RuleTrace(
                rules=(entrada,),
                winning_rule_id=None,
                # `fallback_applied=True`: la decisión NO salió de ninguna regla. Quien
                # lea la traza debe poder distinguir "ninguna regla disparó" de "faltaba
                # el dato", y el `reason_code` declarado lo dice.
                fallback_applied=True,
            ),
        )

    def evaluate(
        self, snapshot: Mapping[str, Any], context: PolicyContext
    ) -> StrategyDecision:
        if not context.as_of:
            raise ValueError(
                f"context.as_of is required to evaluate {self.sleeve_id} "
                "(fail-closed: decisions never carry an empty as_of)"
            )
        ausente = self._declared_absence(snapshot)
        if ausente is not None:
            return self._fallback_decision(ausente, snapshot, context)

        errors = self.validate_inputs(snapshot)
        if errors:
            raise ValueError(
                f"Invalid inputs for {self.sleeve_id}: {'; '.join(errors)}"
            )

        trace_entries: list[RuleTraceEntry] = []
        winner: dict[str, Any] | None = None
        for rule in self.rules:
            fired = evaluate_condition(
                rule["when"], snapshot, context.previous_snapshot
            )
            observed = {
                name: snapshot[name]
                for name in sorted(referenced_features(rule["when"]))
            }
            trace_entries.append(
                RuleTraceEntry(
                    rule_id=str(rule["id"]),
                    label=str(rule.get("label", rule["id"])),
                    observed=observed,
                    result=fired,
                    reason_code=str(rule["output"].get("reason_code", "")),
                    threshold=dict(threshold_values(rule["when"], snapshot)),
                )
            )
            if fired and winner is None:
                winner = rule  # first_match; keep tracing remaining rules

        if winner is not None:
            output = winner["output"]
            direction = str(output["direction"])
            exposure = float(output["target_exposure"])
            reason_codes = (str(output.get("reason_code", winner["id"])),)
        else:
            direction = self.default_direction
            exposure = self.default_exposure
            reason_codes = (self.default_reason_code,)

        components = {name: snapshot[name] for name in self._required}
        return StrategyDecision(
            sleeve_id=self.sleeve_id,
            strategy_version=self.version,
            engine_ref=EngineRef(
                type="rule_based",
                policy_version_id=self.policy_version_id,
                policy_hash=self.policy_hash,
            ),
            as_of=context.as_of,
            direction=direction,
            target_exposure=exposure,
            reason_codes=reason_codes,
            decision_components=components,
            # Resolution facts stated by the ENGINE (BL-46 R5): which rule won
            # under first_match, or that the declared fallback applied. The
            # frontend renders these; it never re-derives them.
            rule_trace=RuleTrace(
                rules=tuple(trace_entries),
                winning_rule_id=str(winner["id"]) if winner is not None else None,
                fallback_applied=winner is None,
            ),
        )
