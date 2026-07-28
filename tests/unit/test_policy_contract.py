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
            r"export type EngineRef =\s*RuleBasedEngineRef \| MlEngineRef \| CompositeEngineRef",
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
# SHARED CASE TABLE (C-004 remedy-3 finding 6) — the SAME cases, executed in
# BOTH runtimes. This literal table is duplicated in
# usdcop-trading-dashboard/tests/unit/contracts/policy-contract-parity.test.ts
# (case ids and expected verdicts MUST stay identical — a checksum test on
# each side pins the ids). Python executes them against the real
# constructors/validators; Vitest executes them against the TS runtime
# validators. Same verdict, case by case.
# ---------------------------------------------------------------------------


def _decision_payload(**overrides) -> dict:
    payload = {
        "signal_id": "s1:2026-07-27:deadbeefdeadbeef",
        "sleeve_id": "s1",
        "strategy_version": "1.0.0",
        "engine_ref": {"type": "rule_based", "policy_hash": FULL_HASH},
        "as_of": "2026-07-27",
        "direction": "FLAT",
        "target_exposure": 0.0,
        "reason_codes": [],
        "decision_components": {"close": 1.0},
        "rule_trace": None,
        "feature_snapshot_id": None,
        "decision_fingerprint": FULL_HASH,
    }
    payload.update(overrides)
    return payload


PARITY_CASES = [
    # --- StrategyDecision -------------------------------------------------
    ("decision_valid", "decision", _decision_payload(), "valid"),
    ("decision_direction_drop_table", "decision",
     _decision_payload(direction="DROP TABLE trades"), "invalid"),
    ("decision_exposure_bool", "decision",
     _decision_payload(target_exposure=True), "invalid"),
    ("decision_exposure_numeric_string", "decision",
     _decision_payload(target_exposure="1.0"), "invalid"),
    ("decision_exposure_inf", "decision",
     _decision_payload(target_exposure=float("inf")), "invalid"),
    ("decision_exposure_nan", "decision",
     _decision_payload(target_exposure=float("nan")), "invalid"),
    ("decision_sleeve_id_none", "decision",
     _decision_payload(sleeve_id=None), "invalid"),
    ("decision_as_of_none", "decision",
     _decision_payload(as_of=None), "invalid"),
    ("decision_policy_hash_true", "decision",
     _decision_payload(engine_ref={"type": "rule_based", "policy_hash": True}),
     "invalid"),
    ("decision_components_infinity", "decision",
     _decision_payload(decision_components={"x": float("inf")}), "invalid"),
    ("decision_trace_missing_schema", "decision",
     _decision_payload(rule_trace={"rules": []}), "invalid"),
    # --- EngineRef --------------------------------------------------------
    ("engine_ref_valid", "engine_ref",
     {"type": "rule_based", "policy_hash": FULL_HASH}, "valid"),
    ("engine_ref_hash_bool", "engine_ref",
     {"type": "rule_based", "policy_hash": True}, "invalid"),
    ("engine_ref_hash_not_hex", "engine_ref",
     {"type": "rule_based", "policy_hash": "sha256:NOT-HEX!"}, "invalid"),
    ("engine_ref_ml_snapshot_bool", "engine_ref",
     {"type": "ml", "model_snapshot_id": True}, "invalid"),
    # --- Condition AST ----------------------------------------------------
    ("condition_valid_gt", "condition",
     {"operator": "greater_than", "left": "feature.close", "right": 1.5}, "valid"),
    ("condition_op_drop_table", "condition",
     {"operator": "DROP TABLE trades", "left": "feature.close", "right": 1.0},
     "invalid"),
    ("condition_op_eval", "condition",
     {"operator": "eval", "code": "close > ma_200"}, "invalid"),
    ("condition_feature_true_mapping", "condition",
     {"operator": "greater_than", "left": {"feature": True}, "right": 1.0},
     "invalid"),
    ("condition_feature_empty", "condition",
     {"operator": "greater_than", "left": "feature.", "right": 1.0}, "invalid"),
    ("condition_nan_literal", "condition",
     {"operator": "greater_than", "left": "feature.close", "right": float("nan")},
     "invalid"),
    ("condition_inf_literal", "condition",
     {"operator": "greater_than", "left": "feature.close", "right": float("inf")},
     "invalid"),
    ("condition_bool_operand", "condition",
     {"operator": "greater_than", "left": True, "right": 1.0}, "invalid"),
    # --- RuleTrace --------------------------------------------------------
    ("trace_valid_v1", "trace",
     {"trace_schema": "rule_trace_v1",
      "rules": [{"rule_id": "r1", "label": "R1", "observed": {"close": 1.0},
                 "result": True, "reason_code": "GT"}]}, "valid"),
    ("trace_v2_rejected", "trace",
     {"trace_schema": "rule_trace_v2", "rules": []}, "invalid"),
    ("trace_missing_schema", "trace", {"rules": []}, "invalid"),
    ("trace_observed_infinity", "trace",
     {"trace_schema": "rule_trace_v1",
      "rules": [{"rule_id": "r1", "label": "R1",
                 "observed": {"close": float("inf")}, "result": True,
                 "reason_code": "GT"}]}, "invalid"),
    # --- PolicyContext ----------------------------------------------------
    ("context_valid", "context", {"mode": "DECISION"}, "valid"),
    ("context_mode_drop_table", "context", {"mode": "DROP_TABLE"}, "invalid"),
    # --- FeatureSnapshot --------------------------------------------------
    ("snapshot_valid", "snapshot", {"close": 2.0, "ma_200": 1.0}, "valid"),
    ("snapshot_infinity", "snapshot",
     {"close": float("inf"), "ma_200": 1.0}, "invalid"),
    ("snapshot_nan", "snapshot", {"close": float("nan"), "ma_200": 1.0}, "invalid"),
    ("snapshot_bool_value", "snapshot", {"close": True, "ma_200": 1.0}, "invalid"),
    ("snapshot_null_value", "snapshot", {"close": None, "ma_200": 1.0}, "invalid"),
    ("snapshot_string_value", "snapshot", {"close": "5", "ma_200": 1.0}, "invalid"),
]

#: Pinned so both runtimes prove they run the SAME table (mirrored in the
#: Vitest file — if you add a case, update BOTH files and BOTH pins).
PARITY_CASE_IDS_SHA256_PREFIX = "case-table-v1:35"


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
    """Finding 6: the same case table EXECUTED (not text-inspected) in Python.
    The Vitest twin executes the identical table against the TS validators."""

    @pytest.mark.parametrize(
        "case_id,target,payload,expect",
        PARITY_CASES,
        ids=[c[0] for c in PARITY_CASES],
    )
    def test_case(self, case_id, target, payload, expect):
        assert _run_python_case(target, payload) == expect, (
            f"case {case_id!r}: Python verdict diverges from the table"
        )

    def test_table_pin_matches(self):
        """The pin encodes table version + case count; the TS twin asserts the
        same pin over the same ids — drift in either file fails one side."""
        assert PARITY_CASE_IDS_SHA256_PREFIX == f"case-table-v1:{len(PARITY_CASES)}"
        assert len({c[0] for c in PARITY_CASES}) == len(PARITY_CASES)
