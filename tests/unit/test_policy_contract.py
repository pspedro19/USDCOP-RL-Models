"""
Unit tests — Policy contract + declarative DSL (BL-45 R1, CTR-POLICY-001).

Covers:
- MA200 declarative example evaluates correctly (LONG above, default FLAT below)
- Determinism: same input + same policy => identical decision (CI §11)
- Operator outside the whitelist => ValueError
- eval/code in spec => rejected (never executed)
- EngineRef discriminated invariants
- Stateful context (decision §15.2) and crossing operators
- TS mirror parity (policy.contract.ts — fields + whitelist identical)
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from src.contracts.policy import (
    ENGINE_TYPES,
    POLICY_MODES,
    VALID_DIRECTIONS,
    EngineRef,
    Policy,
    PolicyContext,
    StrategyDecision,
)
from src.contracts.policy_dsl import (
    ALLOWED_OPERATORS,
    COMPARISON_OPERATORS,
    LOGICAL_OPERATORS,
    RANGE_OPERATORS,
    DeclarativePolicy,
    evaluate_condition,
    validate_condition,
)
from src.contracts.rule_trace import (
    RULE_TRACE_SCHEMA_V1,
    SUPPORTED_TRACE_SCHEMAS,
    RuleTrace,
    RuleTraceEntry,
    ensure_json_safe,
)

ROOT = Path(__file__).resolve().parents[2]
TS_MIRROR = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts" / "policy.contract.ts"


def ma200_spec() -> dict:
    """Declarative MA200 spec (mode B, spec §3.2) as parsed from YAML."""
    return {
        "id": "spx500_daily_ma200_v1",
        "version": "2.0.0",
        "policy_hash": "sha256:deadbeef",
        "resolution": {
            "mode": "first_match",
            "default_target_exposure": 0.0,
            "default_direction": "FLAT",
            "default_reason_code": "CLOSE_BELOW_MA200",
        },
        "rules": [
            {
                "id": "trend_on",
                "label": "Precio sobre MA200",
                "priority": 100,
                "when": {
                    "operator": "greater_than",
                    "left": "feature.close",
                    "right": "feature.ma_200",
                },
                "output": {
                    "direction": "LONG",
                    "target_exposure": 1.0,
                    "reason_code": "CLOSE_ABOVE_MA200",
                },
            },
        ],
    }


class TestMA200Declarative:
    def test_long_when_close_above_ma(self):
        policy = DeclarativePolicy(ma200_spec())
        snapshot = {"close": 6412.8, "ma_200": 5984.2}
        decision = policy.evaluate(snapshot, PolicyContext(as_of="2026-07-27"))

        assert decision.direction == "LONG"
        assert decision.target_exposure == 1.0
        assert decision.reason_codes == ("CLOSE_ABOVE_MA200",)
        assert decision.engine_ref.type == "rule_based"
        assert decision.engine_ref.policy_hash == "sha256:deadbeef"
        assert decision.decision_components == {"close": 6412.8, "ma_200": 5984.2}

    def test_flat_default_when_close_below_ma(self):
        policy = DeclarativePolicy(ma200_spec())
        snapshot = {"close": 5500.0, "ma_200": 5984.2}
        decision = policy.evaluate(snapshot, PolicyContext(as_of="2026-07-27"))

        assert decision.direction == "FLAT"
        assert decision.target_exposure == 0.0
        assert decision.reason_codes == ("CLOSE_BELOW_MA200",)

    def test_rule_trace_v1_emitted(self):
        policy = DeclarativePolicy(ma200_spec())
        decision = policy.evaluate(
            {"close": 6412.8, "ma_200": 5984.2}, PolicyContext(as_of="2026-07-27")
        )
        trace = decision.rule_trace
        assert isinstance(trace, RuleTrace)
        payload = trace.to_dict()
        assert payload["trace_schema"] == RULE_TRACE_SCHEMA_V1
        assert payload["rules"][0]["rule_id"] == "trend_on"
        assert payload["rules"][0]["result"] is True
        assert payload["rules"][0]["observed"] == {"close": 6412.8, "ma_200": 5984.2}
        assert payload["rules"][0]["reason_code"] == "CLOSE_ABOVE_MA200"
        # round-trip
        assert RuleTrace.from_dict(payload).to_dict() == payload

    def test_required_features_and_validate_inputs(self):
        policy = DeclarativePolicy(ma200_spec())
        assert policy.required_features() == ["close", "ma_200"]
        assert policy.validate_inputs({"close": 1.0, "ma_200": 2.0}) == []
        errors = policy.validate_inputs({"close": 1.0})
        assert errors and "ma_200" in errors[0]
        assert policy.validate_inputs({"close": 1.0, "ma_200": float("nan")})

    def test_satisfies_policy_protocol(self):
        assert isinstance(DeclarativePolicy(ma200_spec()), Policy)


class TestDeterminism:
    def test_same_input_same_decision(self):
        """CI §11: mismos inputs + misma policy => misma decisión."""
        snapshot = {"close": 6412.8, "ma_200": 5984.2}
        d1 = DeclarativePolicy(ma200_spec()).evaluate(
            snapshot, PolicyContext(as_of="2026-07-27")
        )
        d2 = DeclarativePolicy(ma200_spec()).evaluate(
            snapshot, PolicyContext(as_of="2026-07-27")
        )
        assert d1 == d2
        assert d1.decision_fingerprint == d2.decision_fingerprint
        assert d1.signal_id == d2.signal_id

    def test_different_input_different_fingerprint(self):
        p = DeclarativePolicy(ma200_spec())
        d1 = p.evaluate({"close": 6412.8, "ma_200": 5984.2}, PolicyContext(as_of="2026-07-27"))
        d2 = p.evaluate({"close": 6000.0, "ma_200": 5984.2}, PolicyContext(as_of="2026-07-27"))
        assert d1.decision_fingerprint != d2.decision_fingerprint


class TestWhitelistSecurity:
    def test_operator_outside_whitelist_raises(self):
        spec = ma200_spec()
        spec["rules"][0]["when"] = {
            "operator": "python_exec",
            "left": "feature.close",
            "right": "feature.ma_200",
        }
        with pytest.raises(ValueError, match="whitelist"):
            DeclarativePolicy(spec)

    def test_eval_operator_rejected(self):
        with pytest.raises(ValueError, match="whitelist"):
            validate_condition({"operator": "eval", "code": "close > ma_200"})

    def test_raw_string_condition_rejected(self):
        """`when: "eval(close > ma200)"` — code from YAML is NEVER executed."""
        spec = ma200_spec()
        spec["rules"][0]["when"] = "eval(close > ma_200 and custom_python())"
        with pytest.raises(ValueError):
            DeclarativePolicy(spec)

    def test_code_string_operand_rejected(self):
        with pytest.raises(ValueError):
            evaluate_condition(
                {
                    "operator": "greater_than",
                    "left": "__import__('os').system('id')",
                    "right": 0,
                },
                {"close": 1.0},
            )

    def test_whitelist_is_exactly_the_spec_set(self):
        assert ALLOWED_OPERATORS == {
            "greater_than", "less_than", "equal", "all", "any", "not",
            "crosses_above", "crosses_below", "between",
        }


class TestOperators:
    SNAP = {"close": 10.0, "ma": 8.0, "vix": 15.0}

    def test_less_than_equal_between(self):
        assert evaluate_condition(
            {"operator": "less_than", "left": "feature.ma", "right": "feature.close"},
            self.SNAP,
        )
        assert evaluate_condition(
            {"operator": "equal", "left": "feature.vix", "right": 15.0}, self.SNAP
        )
        assert evaluate_condition(
            {"operator": "between", "value": "feature.vix", "lower": 10, "upper": 20},
            self.SNAP,
        )

    def test_logical_all_any_not(self):
        gt = {"operator": "greater_than", "left": "feature.close", "right": "feature.ma"}
        lt = {"operator": "less_than", "left": "feature.close", "right": "feature.ma"}
        assert evaluate_condition({"operator": "all", "conditions": [gt]}, self.SNAP)
        assert evaluate_condition({"operator": "any", "conditions": [lt, gt]}, self.SNAP)
        assert evaluate_condition({"operator": "not", "condition": lt}, self.SNAP)

    def test_crosses_above_uses_previous_snapshot(self):
        cond = {"operator": "crosses_above", "left": "feature.close", "right": "feature.ma"}
        now = {"close": 10.0, "ma": 9.0}
        prev = {"close": 8.5, "ma": 9.0}
        assert evaluate_condition(cond, now, previous_snapshot=prev)
        assert not evaluate_condition(cond, now, previous_snapshot=now)
        with pytest.raises(ValueError, match="previous_snapshot"):
            evaluate_condition(cond, now)


class TestEngineRef:
    def test_rule_based_requires_policy_hash(self):
        with pytest.raises(ValueError, match="policy_hash"):
            EngineRef(type="rule_based")

    def test_rule_based_forbids_model_snapshot(self):
        with pytest.raises(ValueError, match="model"):
            EngineRef(type="rule_based", policy_hash="sha256:deadbeef", model_snapshot_id="m1")

    def test_ml_requires_model_snapshot(self):
        with pytest.raises(ValueError, match="model_snapshot_id"):
            EngineRef(type="ml")
        assert EngineRef(type="ml", model_snapshot_id="m1").to_dict() == {
            "type": "ml", "model_snapshot_id": "m1",
        }

    def test_composite_carries_policy_and_models(self):
        ref = EngineRef(
            type="composite", policy_hash="sha256:deadbeef", model_snapshot_ids=("m1", "m2")
        )
        assert ref.to_dict()["model_snapshot_ids"] == ["m1", "m2"]

    def test_unknown_engine_type_rejected(self):
        with pytest.raises(ValueError, match="engine_ref.type"):
            EngineRef(type="quantum")


class TestStatefulContext:
    def test_context_state_persists_across_evaluations(self):
        """Decision §15.2: context carries mutable per-strategy state."""

        class StreakPolicy:
            """Coded policy: LONG only after 3 consecutive closes above MA."""

            def required_features(self):
                return ["close", "ma_200"]

            def validate_inputs(self, snapshot):
                return [f for f in self.required_features() if f not in snapshot]

            def evaluate(self, snapshot, context):
                streak = context.state.get("streak", 0)
                streak = streak + 1 if snapshot["close"] > snapshot["ma_200"] else 0
                context.state["streak"] = streak
                direction = "LONG" if streak >= 3 else "FLAT"
                return StrategyDecision(
                    sleeve_id="streak_v1",
                    strategy_version="1.0.0",
                    engine_ref=EngineRef(type="rule_based", policy_hash="sha256:deadbeef"),
                    as_of=context.as_of or "",
                    direction=direction,
                    target_exposure=1.0 if direction == "LONG" else 0.0,
                    reason_codes=("STREAK",),
                    decision_components={"streak": streak},
                )

        policy = StreakPolicy()
        assert isinstance(policy, Policy)
        ctx = PolicyContext(as_of="2026-07-27")
        snap = {"close": 10.0, "ma_200": 9.0}
        assert policy.evaluate(snap, ctx).direction == "FLAT"
        assert policy.evaluate(snap, ctx).direction == "FLAT"
        assert policy.evaluate(snap, ctx).direction == "LONG"
        assert ctx.state["streak"] == 3


class TestDecisionInvariants:
    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError, match="direction"):
            StrategyDecision(
                sleeve_id="s",
                strategy_version="1",
                engine_ref=EngineRef(type="rule_based", policy_hash="sha256:deadbeef"),
                as_of="2026-07-27",
                direction="UP",
                target_exposure=1.0,
            )

    def test_to_dict_shape(self):
        d = StrategyDecision(
            sleeve_id="s",
            strategy_version="1",
            engine_ref=EngineRef(type="rule_based", policy_hash="sha256:deadbeef"),
            as_of="2026-07-27",
            direction="FLAT",
            target_exposure=0.0,
        ).to_dict()
        assert d["engine_ref"] == {"type": "rule_based", "policy_hash": "sha256:deadbeef"}
        assert d["decision_fingerprint"].startswith("sha256:")
        assert d["signal_id"].startswith("s:2026-07-27:")

    def test_missing_default_exposure_rejected(self):
        spec = ma200_spec()
        del spec["resolution"]["default_target_exposure"]
        with pytest.raises(ValueError, match="default_target_exposure"):
            DeclarativePolicy(spec)


# ---------------------------------------------------------------------------
# Fail-closed construction (C-004 second remedy — Codex findings 2/3/4):
# invalid enum/literal/finiteness values RAISE at construction, never pass.
# ---------------------------------------------------------------------------

def _decision(**overrides) -> StrategyDecision:
    kwargs = dict(
        sleeve_id="s",
        strategy_version="1",
        engine_ref=EngineRef(type="rule_based", policy_hash="sha256:deadbeef"),
        as_of="2026-07-27",
        direction="FLAT",
        target_exposure=0.0,
    )
    kwargs.update(overrides)
    return StrategyDecision(**kwargs)


class TestFailClosedMode:
    def test_invalid_mode_rejected(self):
        """Codex finding 2: mode='DROP_TABLE' must raise, not pass silently."""
        with pytest.raises(ValueError, match="context.mode"):
            PolicyContext(mode="DROP_TABLE")

    def test_all_whitelisted_modes_construct(self):
        assert POLICY_MODES == ("DECISION", "FREEZE", "REVALIDATE", "BACKFILL")
        for mode in POLICY_MODES:
            assert PolicyContext(mode=mode).mode == mode

    def test_default_mode_is_decision(self):
        assert PolicyContext().mode == "DECISION"


class TestFailClosedTraceSchema:
    def test_unsupported_schema_rejected_at_construction(self):
        """Codex finding 3: trace_schema='rule_trace_v2' must raise."""
        with pytest.raises(ValueError, match="trace_schema"):
            RuleTrace(trace_schema="rule_trace_v2")

    def test_unsupported_schema_rejected_in_from_dict(self):
        with pytest.raises(ValueError, match="trace_schema"):
            RuleTrace.from_dict({"trace_schema": "rule_trace_v2", "rules": []})

    def test_supported_schema_is_exactly_v1(self):
        assert SUPPORTED_TRACE_SCHEMAS == (RULE_TRACE_SCHEMA_V1,)
        assert RuleTrace().trace_schema == RULE_TRACE_SCHEMA_V1


class TestFailClosedExposure:
    """Codex finding 4: NaN/±Infinity target_exposure must raise
    (strategy-contract invariant 2: no NaN/Infinity, ever)."""

    @pytest.mark.parametrize(
        "bad", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"]
    )
    def test_non_finite_exposure_rejected(self, bad):
        with pytest.raises(ValueError, match="finite"):
            _decision(target_exposure=bad)

    @pytest.mark.parametrize("bad", ["1.0", None, True], ids=["str", "none", "bool"])
    def test_non_numeric_exposure_rejected(self, bad):
        with pytest.raises(ValueError, match="target_exposure"):
            _decision(target_exposure=bad)

    def test_finite_exposure_accepted(self):
        assert _decision(target_exposure=1.5).target_exposure == 1.5

    def test_declarative_default_exposure_nan_rejected(self):
        spec = ma200_spec()
        spec["resolution"]["default_target_exposure"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            DeclarativePolicy(spec)

    def test_declarative_rule_exposure_inf_rejected(self):
        spec = ma200_spec()
        spec["rules"][0]["output"]["target_exposure"] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            DeclarativePolicy(spec)


class TestFailClosedOperands:
    def test_nan_literal_operand_rejected(self):
        with pytest.raises(ValueError, match="[Oo]perand"):
            validate_condition(
                {"operator": "greater_than", "left": "feature.x", "right": float("nan")}
            )

    def test_between_invalid_operand_rejected(self):
        with pytest.raises(ValueError, match="[Oo]perand"):
            validate_condition(
                {
                    "operator": "between",
                    "value": "DROP TABLE trades",
                    "lower": 0,
                    "upper": 1,
                }
            )


# ---------------------------------------------------------------------------
# TS mirror parity (contract-change skill step 6 — same pattern as
# test_forecast_output_contract.py: read the .ts as text, assert identical
# fields and whitelist, not string vibes)
# ---------------------------------------------------------------------------

def _ts_text() -> str:
    return TS_MIRROR.read_text(encoding="utf-8")


def _ts_const_literals(text: str, name: str, pattern: str = r"'([a-z_]+)'") -> set[str]:
    """Extract the string literals of an `export const NAME = [ ... ] as const`."""
    m = re.search(rf"export const {name} = \[(.*?)\]", text, re.S)
    assert m, f"const {name} missing from TS mirror"
    return set(re.findall(pattern, m.group(1)))


def _ts_interface(text: str, name: str) -> str:
    m = re.search(rf"export interface {name} \{{(.*?)\n\}}", text, re.S)
    assert m, f"interface {name} missing from TS mirror"
    return m.group(1)


class TestTsMirrorParity:
    def test_mirror_exists(self):
        assert TS_MIRROR.is_file(), "TS mirror missing — contracts change in pairs"

    # --- Whitelist: IDENTICAL sets, per operator family -------------------

    def test_operator_whitelist_identical(self):
        text = _ts_text()
        assert _ts_const_literals(text, "COMPARISON_OPERATORS") == set(COMPARISON_OPERATORS)
        assert _ts_const_literals(text, "LOGICAL_OPERATORS") == set(LOGICAL_OPERATORS)
        assert _ts_const_literals(text, "RANGE_OPERATORS") == set(RANGE_OPERATORS)

    def test_allowed_operators_is_union_of_families(self):
        text = _ts_text()
        m = re.search(r"export const ALLOWED_OPERATORS = \[(.*?)\]", text, re.S)
        assert m, "ALLOWED_OPERATORS missing from TS mirror"
        body = m.group(1)
        for family in ("COMPARISON_OPERATORS", "LOGICAL_OPERATORS", "RANGE_OPERATORS"):
            assert f"...{family}" in body, f"ALLOWED_OPERATORS must spread {family}"
        # no extra literal smuggled into the union
        assert not re.findall(r"'([a-z_]+)'", body)
        union = (
            _ts_const_literals(text, "COMPARISON_OPERATORS")
            | _ts_const_literals(text, "LOGICAL_OPERATORS")
            | _ts_const_literals(text, "RANGE_OPERATORS")
        )
        assert union == set(ALLOWED_OPERATORS)

    # --- Discriminator constants ------------------------------------------

    def test_engine_types_and_directions_identical(self):
        text = _ts_text()
        assert _ts_const_literals(text, "ENGINE_TYPES") == set(ENGINE_TYPES)
        assert _ts_const_literals(
            text, "VALID_DIRECTIONS", pattern=r"'([A-Z_]+)'"
        ) == set(VALID_DIRECTIONS)

    def test_policy_modes_match_python_context(self):
        text = _ts_text()
        assert _ts_const_literals(
            text, "POLICY_MODES", pattern=r"'([A-Z_]+)'"
        ) == set(POLICY_MODES)
        assert PolicyContext().mode == "DECISION"

    # --- EngineRef invariants encoded as TYPES ----------------------------

    def test_engine_ref_is_a_discriminated_union(self):
        text = _ts_text()
        assert re.search(
            # BL-46: the union carries the FOUR engines of invariant 1
            # (rl added bilaterally with ENGINE_TYPES).
            r"export type EngineRef =\s*\|?\s*RuleBasedEngineRef\s*\|\s*MlEngineRef"
            r"\s*\|\s*RlEngineRef\s*\|\s*CompositeEngineRef",
            text,
        ), "EngineRef must be the discriminated union of the three engine refs"

    def test_rule_based_requires_hash_and_forbids_snapshots(self):
        block = _ts_interface(_ts_text(), "RuleBasedEngineRef")
        assert "type: 'rule_based';" in block
        assert re.search(r"policy_hash: string;", block), "policy_hash must be required"
        assert "model_snapshot_id?: never;" in block, (
            "rule_based must forbid model_snapshot_id at the type level"
        )
        assert "model_snapshot_ids?: never;" in block, (
            "rule_based must forbid model_snapshot_ids at the type level"
        )

    def test_ml_requires_model_snapshot_id(self):
        block = _ts_interface(_ts_text(), "MlEngineRef")
        assert "type: 'ml';" in block
        assert re.search(r"model_snapshot_id: string;", block), (
            "ml must require model_snapshot_id (no `?`)"
        )

    def test_composite_requires_hash_and_carries_model_ids(self):
        block = _ts_interface(_ts_text(), "CompositeEngineRef")
        assert "type: 'composite';" in block
        assert re.search(r"policy_hash: string;", block), "policy_hash must be required"
        assert "model_snapshot_ids" in block

    def test_every_engine_ref_field_declared_in_ts(self):
        text = _ts_text()
        for f in fields(EngineRef):
            assert f.name in text, f"EngineRef field {f.name!r} missing from TS mirror"

    # --- StrategyDecision / PolicyContext / RuleTrace fields --------------

    def test_every_strategy_decision_field_declared_in_ts(self):
        block = _ts_interface(_ts_text(), "StrategyDecision")
        for f in fields(StrategyDecision):
            assert f.name in block, (
                f"StrategyDecision field {f.name!r} missing from TS mirror"
            )

    def test_every_policy_context_field_declared_in_ts(self):
        block = _ts_interface(_ts_text(), "PolicyContext")
        for f in fields(PolicyContext):
            assert f.name in block, (
                f"PolicyContext field {f.name!r} missing from TS mirror"
            )

    def test_rule_trace_v1_schema_and_fields(self):
        text = _ts_text()
        assert f"RULE_TRACE_SCHEMA_V1 = '{RULE_TRACE_SCHEMA_V1}'" in text, (
            "TS mirror must pin the rule_trace_v1 schema literal"
        )
        trace_block = _ts_interface(text, "RuleTrace")
        assert "trace_schema: typeof RULE_TRACE_SCHEMA_V1;" in trace_block, (
            "trace_schema must be the LITERAL 'rule_trace_v1' type, not string"
        )
        for f in fields(RuleTrace):
            assert f.name in trace_block, (
                f"RuleTrace field {f.name!r} missing from TS mirror"
            )
        entry_block = _ts_interface(text, "RuleTraceEntry")
        for f in fields(RuleTraceEntry):
            assert f.name in entry_block, (
                f"RuleTraceEntry field {f.name!r} missing from TS mirror"
            )


# ---------------------------------------------------------------------------
# TS mirror parity — Policy protocol + DSL AST/operand types + fail-closed
# runtime validators (C-004 second remedy — Codex findings 1 and 6).
# Bilateral SEMANTIC parity: same literals, same whitelist, same rejections.
# ---------------------------------------------------------------------------

class TestTsMirrorPolicyProtocol:
    def test_policy_interface_mirrors_protocol_methods(self):
        """Codex finding 1: the TS mirror must declare `Policy`."""
        block = _ts_interface(_ts_text(), "Policy")
        # exactly the three Protocol methods, with the protocol signatures
        assert re.search(r"required_features\(\): string\[\];", block)
        assert re.search(
            r"validate_inputs\(snapshot: FeatureSnapshot\): string\[\];", block
        )
        assert re.search(
            r"evaluate\(snapshot: FeatureSnapshot, context: PolicyContext\): "
            r"StrategyDecision;",
            block,
        )

    def test_protocol_method_names_all_present(self):
        block = _ts_interface(_ts_text(), "Policy")
        for name in ("required_features", "validate_inputs", "evaluate"):
            assert name in block, f"Policy protocol method {name!r} missing from TS"


class TestTsMirrorDslAst:
    def test_operator_family_types_declared(self):
        text = _ts_text()
        for name in ("ComparisonOperator", "LogicalOperator", "RangeOperator"):
            family = re.sub(r"(?<!^)([A-Z])", r"_\1", name).upper() + "S"
            assert re.search(
                rf"export type {name} = \(typeof {family}\)\[number\];", text
            ), f"{name} must derive from the {family} whitelist const"

    def test_operand_grammar_mirrored(self):
        text = _ts_text()
        # "feature.<name>" string form as a template-literal type
        assert re.search(
            r"export type FeatureRefString = `feature\.\$\{string\}`;", text
        ), "FeatureRefString must be the `feature.${string}` template literal"
        assert "export interface FeatureRefObject" in text
        assert re.search(
            r"export type FeatureRef = FeatureRefString \| FeatureRefObject;", text
        )
        # operand = feature ref | numeric literal (finiteness enforced runtime)
        assert re.search(r"export type Operand = FeatureRef \| number;", text)

    def test_condition_node_union_is_complete(self):
        text = _ts_text()
        m = re.search(r"export type ConditionNode =\s*(.*?);", text, re.S)
        assert m, "ConditionNode union missing from TS mirror"
        union = m.group(1)
        for node in (
            "ComparisonCondition", "AllAnyCondition", "NotCondition",
            "BetweenCondition",
        ):
            assert node in union, f"ConditionNode must include {node}"

    def test_condition_node_shapes_mirror_python_validate_condition(self):
        text = _ts_text()
        cmp_block = _ts_interface(text, "ComparisonCondition")
        assert "operator: ComparisonOperator;" in cmp_block
        assert "left: Operand;" in cmp_block and "right: Operand;" in cmp_block

        allany_block = _ts_interface(text, "AllAnyCondition")
        assert "operator: 'all' | 'any';" in allany_block
        assert "conditions: ConditionNode[];" in allany_block

        not_block = _ts_interface(text, "NotCondition")
        assert "operator: 'not';" in not_block
        assert "condition: ConditionNode;" in not_block

        between_block = _ts_interface(text, "BetweenCondition")
        assert "operator: 'between';" in between_block
        for key in ("value", "lower", "upper"):
            assert f"{key}: Operand;" in between_block


class TestTsMirrorFailClosedValidators:
    """The TS runtime validators must reject EXACTLY what Python rejects
    (same pattern as forecast-output.contract.ts::validateForecastOutput)."""

    def test_supported_trace_schemas_mirrored(self):
        text = _ts_text()
        assert re.search(
            r"export const SUPPORTED_TRACE_SCHEMAS = \[RULE_TRACE_SCHEMA_V1\] as const;",
            text,
        ), "TS must pin the same fail-closed schema whitelist"
        assert set(SUPPORTED_TRACE_SCHEMAS) == {RULE_TRACE_SCHEMA_V1}

    def _fn(self, text: str, name: str) -> str:
        m = re.search(
            rf"export function {name}\(raw: unknown\).*?\n\}}", text, re.S
        )
        assert m, f"runtime validator {name} missing from TS mirror"
        return m.group(0)

    def test_mode_validator_fails_closed(self):
        body = self._fn(_ts_text(), "validatePolicyContext")
        assert "POLICY_MODES.includes" in body, (
            "validatePolicyContext must whitelist-check mode (DROP_TABLE rejected)"
        )

    def test_trace_schema_validator_fails_closed(self):
        body = self._fn(_ts_text(), "validateRuleTrace")
        assert "SUPPORTED_TRACE_SCHEMAS.includes" in body, (
            "validateRuleTrace must whitelist-check trace_schema (v2 rejected)"
        )

    def test_decision_validator_rejects_non_finite_exposure(self):
        body = self._fn(_ts_text(), "validateStrategyDecision")
        assert "finite(o.target_exposure)" in body, (
            "validateStrategyDecision must reject NaN/Infinity target_exposure"
        )
        assert "VALID_DIRECTIONS.includes" in body
        assert "validateEngineRef" in body and "validateRuleTrace" in body

    def test_condition_validator_whitelists_operators(self):
        body = self._fn(_ts_text(), "validateConditionNode")
        assert "ALLOWED_OPERATORS.includes" in body, (
            "validateConditionNode must reject any operator outside the whitelist"
        )
        assert "validateOperand" in body

    def test_operand_validator_rejects_nan_and_code_strings(self):
        body = self._fn(_ts_text(), "validateOperand")
        assert "Number.isFinite" in body, "NaN/Infinity literals must be rejected"
        assert "startsWith('feature.')" in body, (
            "strings must be 'feature.<name>' refs; code/SQL strings rejected"
        )

    def test_engine_ref_validator_mirrors_invariants(self):
        body = self._fn(_ts_text(), "validateEngineRef")
        assert "ENGINE_TYPES.includes" in body
        assert "requires policy_hash" in body
        assert "must NOT carry model snapshots" in body
        assert "requires model_snapshot_id" in body


# ---------------------------------------------------------------------------
# C-004 remedy-3 (Codex rejection of 57ee451) — type-strict validation.
# Findings: (1) bool/string exposure coercion, (2) {feature: true} + empty
# feature name, (3) snapshot Infinity, (4) policy_hash=True / sleeve_id=None /
# as_of=None, (5) Infinity serializable to JSON, (7) trace dict without
# trace_schema accepted by from_dict.
# ---------------------------------------------------------------------------

FULL_HASH = "sha256:" + "deadbeef" * 8  # 64 lowercase hex chars


class TestRemedy3StrictExposures:
    """Finding 1: exposures must be REAL numbers — no float() coercion."""

    def test_declarative_default_exposure_bool_rejected(self):
        spec = ma200_spec()
        spec["resolution"]["default_target_exposure"] = True
        with pytest.raises(ValueError, match="real number"):
            DeclarativePolicy(spec)

    def test_declarative_default_exposure_numeric_string_rejected(self):
        spec = ma200_spec()
        spec["resolution"]["default_target_exposure"] = "1.0"
        with pytest.raises(ValueError, match="real number"):
            DeclarativePolicy(spec)

    def test_declarative_rule_exposure_bool_rejected(self):
        spec = ma200_spec()
        spec["rules"][0]["output"]["target_exposure"] = True
        with pytest.raises(ValueError, match="real number"):
            DeclarativePolicy(spec)

    def test_declarative_rule_exposure_numeric_string_rejected(self):
        spec = ma200_spec()
        spec["rules"][0]["output"]["target_exposure"] = "1.0"
        with pytest.raises(ValueError, match="real number"):
            DeclarativePolicy(spec)


class TestRemedy3StrictOperands:
    """Finding 2: {feature: true} and 'feature.' (empty name) are typed errors."""

    @pytest.mark.parametrize(
        "bad", [True, None, 1.5, {"nested": "x"}], ids=["bool", "none", "number", "dict"]
    )
    def test_mapping_feature_non_string_rejected(self, bad):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_condition(
                {"operator": "greater_than", "left": {"feature": bad}, "right": 1.0}
            )

    def test_mapping_feature_empty_string_rejected(self):
        with pytest.raises(ValueError, match="non-empty string"):
            validate_condition(
                {"operator": "greater_than", "left": {"feature": ""}, "right": 1.0}
            )

    def test_empty_feature_name_string_form_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            validate_condition(
                {"operator": "greater_than", "left": "feature.", "right": 1.0}
            )

    def test_referenced_features_ignores_malformed_refs(self):
        from src.contracts.policy_dsl import referenced_features

        # lenient walk must not str()-coerce True into a feature name
        assert referenced_features({"feature": True}) == set()
        assert referenced_features("feature.") == set()


class TestRemedy3SnapshotFiniteness:
    """Finding 3: every numeric snapshot value must be finite at validate_inputs."""

    def _policy(self):
        return DeclarativePolicy(ma200_spec())

    @pytest.mark.parametrize(
        "bad", [float("inf"), float("-inf"), float("nan")], ids=["inf", "-inf", "nan"]
    )
    def test_non_finite_snapshot_value_errors(self, bad):
        errors = self._policy().validate_inputs({"close": bad, "ma_200": 1.0})
        assert errors and "close" in errors[0]

    @pytest.mark.parametrize("bad", [True, "5", None], ids=["bool", "str", "none"])
    def test_non_numeric_snapshot_value_errors(self, bad):
        errors = self._policy().validate_inputs({"close": bad, "ma_200": 1.0})
        assert errors and "close" in errors[0]

    def test_evaluate_raises_on_infinite_snapshot(self):
        with pytest.raises(ValueError, match="[Ff]inite|[Ii]nvalid inputs"):
            self._policy().evaluate(
                {"close": float("inf"), "ma_200": 1.0},
                PolicyContext(as_of="2026-07-27"),
            )

    def test_resolve_rejects_bool_snapshot_value(self):
        with pytest.raises(ValueError, match="real number"):
            evaluate_condition(
                {"operator": "greater_than", "left": "feature.close", "right": 0.5},
                {"close": True},
            )


class TestRemedy3StrictIdsAndHashes:
    """Finding 4 + red-team: bool/None never pass truthiness for ids/hashes."""

    def test_engine_ref_policy_hash_bool_rejected(self):
        with pytest.raises(ValueError, match="policy_hash"):
            EngineRef(type="rule_based", policy_hash=True)

    def test_engine_ref_policy_hash_non_hex_rejected(self):
        with pytest.raises(ValueError, match="sha256"):
            EngineRef(type="rule_based", policy_hash="sha256:NOT-HEX!")

    def test_engine_ref_model_snapshot_id_bool_rejected(self):
        with pytest.raises(ValueError, match="model_snapshot_id"):
            EngineRef(type="ml", model_snapshot_id=True)

    def test_decision_sleeve_id_none_rejected(self):
        with pytest.raises(ValueError, match="sleeve_id"):
            _decision(sleeve_id=None)

    def test_decision_as_of_none_rejected(self):
        with pytest.raises(ValueError, match="as_of"):
            _decision(as_of=None)

    def test_decision_as_of_non_iso_rejected(self):
        with pytest.raises(ValueError, match="as_of"):
            _decision(as_of="not-a-date")

    def test_decision_signal_id_bool_rejected(self):
        with pytest.raises(ValueError, match="signal_id"):
            _decision(signal_id=True)

    def test_decision_fingerprint_bool_rejected(self):
        with pytest.raises(ValueError, match="decision_fingerprint"):
            _decision(decision_fingerprint=True)

    def test_context_as_of_bool_rejected(self):
        with pytest.raises(ValueError, match="as_of"):
            PolicyContext(as_of=True)

    def test_declarative_policy_hash_computed_when_absent(self):
        spec = ma200_spec()
        del spec["policy_hash"]
        policy = DeclarativePolicy(spec)
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", policy.policy_hash)


class TestRemedy3JsonStrict:
    """Finding 5: an Infinity can NEVER be serialized — it raises."""

    def test_components_with_infinity_rejected_at_construction(self):
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            _decision(decision_components={"x": float("inf")})

    def test_nested_components_with_nan_rejected(self):
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            _decision(decision_components={"a": {"b": [1.0, float("nan")]}})

    def test_to_json_raises_if_infinity_injected_post_construction(self):
        d = _decision()
        d.decision_components["x"] = float("inf")  # dict is mutable — belt+braces
        with pytest.raises(ValueError):
            d.to_json()

    def test_to_json_emits_valid_json(self):
        import json as _json

        payload = _json.loads(_decision().to_json())
        assert payload["direction"] == "FLAT"

    def test_trace_observed_infinity_rejected(self):
        with pytest.raises(ValueError, match="[Nn]on-finite"):
            RuleTraceEntry(rule_id="r1", label="x", observed={"close": float("inf")})


class TestRemedy3TraceFromDictStrict:
    """Finding 7 (red-team): from_dict without explicit trace_schema is rejected."""

    def test_missing_trace_schema_rejected(self):
        with pytest.raises(ValueError, match="trace_schema"):
            RuleTrace.from_dict({"rules": []})

    def test_missing_rules_rejected(self):
        with pytest.raises(ValueError, match="rules"):
            RuleTrace.from_dict({"trace_schema": RULE_TRACE_SCHEMA_V1})

    def test_non_bool_result_rejected(self):
        with pytest.raises(ValueError, match="result"):
            RuleTrace.from_dict(
                {
                    "trace_schema": RULE_TRACE_SCHEMA_V1,
                    "rules": [{"rule_id": "r1", "result": "true"}],
                }
            )

    def test_non_string_rule_id_rejected(self):
        with pytest.raises(ValueError, match="rule_id"):
            RuleTrace.from_dict(
                {
                    "trace_schema": RULE_TRACE_SCHEMA_V1,
                    "rules": [{"rule_id": 42}],
                }
            )


# ---------------------------------------------------------------------------
# SHARED FIXTURE (C-004 remedio-4 divergence 1) — ONE versioned JSON case
# table (tests/fixtures/policy_contract_cases.v1.json) LOADED by both runners
# (this file and usdcop-trading-dashboard/tests/unit/contracts/
# policy-contract-parity.test.ts). Nothing is duplicated literally: each
# runner recomputes the fixture's content SHA-256 before running and goes RED
# on drift. Non-JSON values travel as sentinels ($nonfinite / $pytype) and
# are decoded natively per runtime.
# ---------------------------------------------------------------------------

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "policy_contract_cases.v1.json"

_SHA_FIELD = re.compile(r'"content_sha256":\s*"([0-9a-f]{64})"')


def _decode_fixture_value(value):
    """Decode fixture sentinels into native Python values/types."""
    if isinstance(value, dict):
        keys = set(value)
        if keys == {"$nonfinite"}:
            return {
                "NaN": float("nan"),
                "Infinity": float("inf"),
                "-Infinity": float("-inf"),
            }[value["$nonfinite"]]
        if keys == {"$pytype", "value"}:
            kind, v = value["$pytype"], value["value"]
            if kind == "numpy.float32":
                import numpy

                return numpy.float32(v)
            if kind == "numpy.float64":
                import numpy

                return numpy.float64(v)
            if kind == "decimal":
                from decimal import Decimal

                return Decimal(v)
            if kind == "datetime":
                return _dt.datetime.fromisoformat(v)
            if kind == "set":
                return set(v)
            if kind == "bytes":
                return v.encode("utf-8")
            raise AssertionError(f"unknown $pytype {kind!r} in fixture")
        return {k: _decode_fixture_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_fixture_value(v) for v in value]
    return value


def load_parity_fixture() -> tuple[dict, list[tuple[str, str, object, str]]]:
    """Load + SHA-verify the shared fixture. Drift => AssertionError (RED)."""
    raw = FIXTURE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = _SHA_FIELD.search(raw)
    if not m:
        raise AssertionError("fixture must declare a 64-hex content_sha256")
    declared = m.group(1)
    actual = hashlib.sha256(raw.replace(declared, "", 1).encode("utf-8")).hexdigest()
    if actual != declared:
        raise AssertionError(
            f"FIXTURE DRIFT: declared content_sha256 {declared} != recomputed "
            f"{actual} — the case table changed without regenerating the pin "
            "(both runners refuse to run)"
        )
    doc = json.loads(raw)
    if doc.get("fixture") != "policy_contract_cases" or doc.get("version") != "v1":
        raise AssertionError("unexpected fixture identity/version")
    cases = [
        (c["id"], c["target"], _decode_fixture_value(c["payload"]), c["expect"])
        for c in doc["cases"]
    ]
    ids = [c[0] for c in cases]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate case ids in fixture")
    return doc, cases


_FIXTURE_DOC, PARITY_CASES = load_parity_fixture()


def _run_python_case(target: str, payload):
    """Execute one table case against the REAL Python validators/constructors."""
    try:
        if target == "decision":
            kwargs = dict(payload)
            kwargs["engine_ref"] = EngineRef(**kwargs["engine_ref"])
            kwargs["reason_codes"] = tuple(kwargs.get("reason_codes", ()))
            StrategyDecision(**kwargs)
        elif target == "engine_ref":
            EngineRef(**payload)
        elif target == "condition":
            validate_condition(payload)
        elif target == "trace":
            RuleTrace.from_dict(payload)
        elif target == "context":
            PolicyContext(**payload)
        elif target == "snapshot":
            errors = DeclarativePolicy(ma200_spec()).validate_inputs(payload)
            if errors:
                raise ValueError("; ".join(errors))
        else:  # pragma: no cover
            raise AssertionError(f"unknown target {target!r}")
        return "valid"
    except (ValueError, TypeError, KeyError):
        return "invalid"


class TestSharedCaseTable:
    """Divergence 1: ONE fixture, loaded (not duplicated) and EXECUTED here
    against the real constructors/validators; the Vitest twin loads THE SAME
    file and executes it against the TS runtime validators."""

    @pytest.mark.parametrize(
        "case_id,target,payload,expect",
        PARITY_CASES,
        ids=[c[0] for c in PARITY_CASES],
    )
    def test_case(self, case_id, target, payload, expect):
        assert _run_python_case(target, payload) == expect, (
            f"case {case_id!r}: Python verdict diverges from the fixture"
        )

    def test_fixture_sha_verified_and_ids_unique(self):
        doc, cases = load_parity_fixture()  # raises on drift
        assert len(cases) >= 35, "fixture lost cases — never shrink the table"
        assert len({c[0] for c in cases}) == len(cases)

    def test_fixture_is_the_only_case_source(self):
        """No literal case table may reappear in the TS runner (drift vector)."""
        ts_twin = (
            ROOT / "usdcop-trading-dashboard" / "tests" / "unit" / "contracts"
            / "policy-contract-parity.test.ts"
        ).read_text(encoding="utf-8")
        assert "policy_contract_cases.v1.json" in ts_twin, (
            "the Vitest twin must load the shared fixture file"
        )
        assert "decision_direction_drop_table" not in ts_twin, (
            "the Vitest twin must not duplicate case literals"
        )


# ---------------------------------------------------------------------------
# C-004 remedio-4 divergence 2 — ISO with a REAL calendar (2026-02-30, 25:00,
# +25:00 impossible), re.fullmatch (no trailing-newline pass), and derived-id
# validation (composed signal_id: parts + embedded timestamp offset).
# ---------------------------------------------------------------------------


class TestRemedy4IsoRealCalendar:
    @pytest.mark.parametrize(
        "bad",
        [
            "2026-02-30",              # day does not exist
            "2026-02-29",              # 2026 is not a leap year
            "2026-13-01",              # month 13
            "2026-00-10",              # month 0
            "2026-07-27T25:00",        # hour 25
            "2026-07-27T10:61",        # minute 61
            "2026-07-27T10:00:61",     # second 61
            "2026-07-27T10:00+25:00",  # offset hour 25
            "2026-07-27T10:00+05:61",  # offset minute 61
            "2026-07-27\n",            # trailing newline (re.match hole)
            "2026-07-27T10:00Z\n",
        ],
    )
    def test_impossible_as_of_rejected(self, bad):
        with pytest.raises(ValueError, match="as_of"):
            PolicyContext(as_of=bad)
        with pytest.raises(ValueError, match="as_of"):
            _decision(as_of=bad)

    @pytest.mark.parametrize(
        "good",
        [
            "2026-07-27",
            "2028-02-29",                    # 2028 IS a leap year
            "2026-07-27T10:30",
            "2026-07-27 10:30:59",
            "2026-07-27T10:30:00.123456Z",
            "2026-07-27T10:30:00-05:00",
            "2026-12-31T23:59:59+23:59",
        ],
    )
    def test_real_timestamps_accepted(self, good):
        assert PolicyContext(as_of=good).as_of == good
        assert _decision(as_of=good).as_of == good


class TestRemedy4FullmatchIds:
    def test_sleeve_id_trailing_newline_rejected(self):
        with pytest.raises(ValueError, match="sleeve_id"):
            _decision(sleeve_id="s1\n")

    def test_hash_trailing_newline_rejected(self):
        with pytest.raises(ValueError, match="policy_hash"):
            EngineRef(type="rule_based", policy_hash=FULL_HASH + "\n")

    def test_model_snapshot_id_trailing_newline_rejected(self):
        with pytest.raises(ValueError, match="model_snapshot_id"):
            EngineRef(type="ml", model_snapshot_id="m1\n")

    def test_no_bare_re_match_left_in_policy_validators(self):
        """$ + re.match accepts a trailing '\\n'; only fullmatch is allowed."""
        src = (ROOT / "src" / "contracts" / "policy.py").read_text(encoding="utf-8")
        assert ".match(" not in src, (
            "policy.py must use re.fullmatch — .match($) accepts a trailing newline"
        )


class TestRemedy4SignalIdComposite:
    def test_derived_signal_id_is_sleeve_asof_hex16(self):
        d = _decision()
        hex16 = d.decision_fingerprint.removeprefix("sha256:")[:16]
        assert re.fullmatch(r"[0-9a-f]{16}", hex16)
        assert d.signal_id == f"s:2026-07-27:{hex16}"

    def test_supplied_matching_signal_id_accepted(self):
        base = _decision()
        assert _decision(signal_id=base.signal_id).signal_id == base.signal_id

    def test_supplied_signal_id_wrong_sleeve_rejected(self):
        with pytest.raises(ValueError, match="signal_id"):
            _decision(signal_id="other:2026-07-27:deadbeefdeadbeef")

    def test_supplied_signal_id_wrong_fingerprint_prefix_rejected(self):
        with pytest.raises(ValueError, match="signal_id"):
            _decision(signal_id="s:2026-07-27:0123456789abcdef")

    def test_supplied_signal_id_impossible_embedded_offset_rejected(self):
        with pytest.raises(ValueError, match="signal_id"):
            _decision(signal_id="s:2026-07-27T10:00+25:00:deadbeefdeadbeef")


# ---------------------------------------------------------------------------
# C-004 remedio-4 divergence 3 — the WHOLE snapshot is validated (required
# AND extra entries), symmetric with TS validateFeatureSnapshot.
# ---------------------------------------------------------------------------


class TestRemedy4SnapshotSymmetric:
    def _policy(self):
        return DeclarativePolicy(ma200_spec())

    def test_codex_probe_close_plus_unused_infinity(self):
        """{close:1.0, unused:Infinity} must fail on BOTH sides identically."""
        errors = self._policy().validate_inputs(
            {"close": 1.0, "unused": float("inf")}
        )
        assert any("ma_200" in e for e in errors)  # required still enforced
        assert any("unused" in e for e in errors)  # extra entry validated too

    def test_extra_key_infinity_fails(self):
        errors = self._policy().validate_inputs(
            {"close": 1.0, "ma_200": 1.0, "unused": float("inf")}
        )
        assert errors and "unused" in errors[0]

    def test_extra_key_string_fails(self):
        errors = self._policy().validate_inputs(
            {"close": 1.0, "ma_200": 1.0, "note": "hello"}
        )
        assert errors and "note" in errors[0]

    def test_evaluate_raises_on_extra_infinity(self):
        with pytest.raises(ValueError, match="[Ii]nvalid inputs"):
            self._policy().evaluate(
                {"close": 1.0, "ma_200": 1.0, "unused": float("inf")},
                PolicyContext(as_of="2026-07-27"),
            )

    def test_non_mapping_snapshot_rejected(self):
        assert self._policy().validate_inputs([1.0, 2.0])
        assert self._policy().validate_inputs(None)


# ---------------------------------------------------------------------------
# C-004 remedio-4 divergence 4 — closed JSON: the allowed types are EXACTLY
# dict/list/str/int/finite-float/bool/None. numpy scalars, Decimal, datetime,
# set, bytes RAISE (non-finites included), and default=str is banned so
# numpy.inf / Decimal('NaN') can never serialize as text.
# ---------------------------------------------------------------------------


def _np():
    import numpy

    return numpy


class TestRemedy4ClosedJson:
    def test_numpy_float32_inf_raises(self):
        with pytest.raises(ValueError, match="non-JSON|non-finite"):
            ensure_json_safe({"x": _np().float32("inf")})

    def test_numpy_float64_nan_raises(self):
        # np.float64 subclasses float — caught by the finiteness branch
        with pytest.raises(ValueError, match="non-finite"):
            ensure_json_safe({"x": _np().float64("nan")})

    def test_decimal_nan_raises(self):
        from decimal import Decimal

        with pytest.raises(ValueError, match="non-JSON"):
            ensure_json_safe({"x": Decimal("NaN")})

    def test_decimal_finite_raises_closed_types(self):
        from decimal import Decimal

        with pytest.raises(ValueError, match="non-JSON"):
            ensure_json_safe({"x": Decimal("1.5")})

    def test_datetime_raises(self):
        with pytest.raises(ValueError, match="non-JSON"):
            ensure_json_safe({"x": _dt.datetime(2026, 7, 27, 10, 0)})

    def test_set_raises(self):
        with pytest.raises(ValueError, match="non-JSON"):
            ensure_json_safe({"x": {1, 2}})

    def test_bytes_raises(self):
        with pytest.raises(ValueError, match="non-JSON"):
            ensure_json_safe({"x": b"abc"})

    def test_numpy_int_raises_closed_types(self):
        with pytest.raises(ValueError, match="non-JSON"):
            ensure_json_safe({"x": _np().int64(5)})

    def test_non_string_key_raises(self):
        with pytest.raises(ValueError, match="key"):
            ensure_json_safe({1: "x"})

    def test_plain_finite_types_accepted(self):
        ensure_json_safe(
            {"a": 1, "b": 1.5, "c": "x", "d": True, "e": None, "f": [1, {"g": 2.0}]}
        )

    def test_components_numpy_inf_rejected_at_construction(self):
        with pytest.raises(ValueError, match="non-JSON|non-finite"):
            _decision(decision_components={"x": _np().float32("inf")})

    def test_components_decimal_nan_rejected_at_construction(self):
        from decimal import Decimal

        with pytest.raises(ValueError, match="non-JSON"):
            _decision(decision_components={"x": Decimal("NaN")})

    def test_to_json_raises_on_injected_numpy_inf_never_text(self):
        d = _decision()
        d.decision_components["x"] = _np().float32("inf")  # mutable dict
        with pytest.raises((TypeError, ValueError)):
            d.to_json()

    def test_trace_to_json_raises_on_injected_decimal_nan(self):
        from decimal import Decimal

        trace = RuleTrace(
            rules=(RuleTraceEntry(rule_id="r1", label="R1", observed={"c": 1.0}),)
        )
        trace.rules[0].observed["c"] = Decimal("NaN")  # mutable dict
        with pytest.raises((TypeError, ValueError)):
            trace.to_json()

    def test_default_str_banned_from_contract_sources(self):
        """default=str turned numpy.inf/Decimal('NaN') into '"inf"'/'"NaN"' text."""
        for rel in ("policy.py", "policy_dsl.py", "rule_trace.py"):
            src = (ROOT / "src" / "contracts" / rel).read_text(encoding="utf-8")
            assert "default=str" not in src, (
                f"src/contracts/{rel}: default=str would serialize "
                "numpy.inf/Decimal('NaN') as text — forbidden"
            )
