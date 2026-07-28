"""
Unit tests — Policy contract + declarative DSL (BL-45 R1, CTR-POLICY-001).

Covers:
- MA200 declarative example evaluates correctly (LONG above, default FLAT below)
- Determinism: same input + same policy => identical decision (CI §11)
- Operator outside the whitelist => ValueError
- eval/code in spec => rejected (never executed)
- EngineRef discriminated invariants
- Stateful context (decision §15.2) and crossing operators
"""

from __future__ import annotations

import pytest

from src.contracts.policy import EngineRef, Policy, PolicyContext, StrategyDecision
from src.contracts.policy_dsl import (
    ALLOWED_OPERATORS,
    DeclarativePolicy,
    evaluate_condition,
    validate_condition,
)
from src.contracts.rule_trace import RULE_TRACE_SCHEMA_V1, RuleTrace


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
        d1 = p.evaluate({"close": 6412.8, "ma_200": 5984.2}, PolicyContext(as_of="x"))
        d2 = p.evaluate({"close": 6000.0, "ma_200": 5984.2}, PolicyContext(as_of="x"))
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
            EngineRef(type="rule_based", policy_hash="sha256:x", model_snapshot_id="m1")

    def test_ml_requires_model_snapshot(self):
        with pytest.raises(ValueError, match="model_snapshot_id"):
            EngineRef(type="ml")
        assert EngineRef(type="ml", model_snapshot_id="m1").to_dict() == {
            "type": "ml", "model_snapshot_id": "m1",
        }

    def test_composite_carries_policy_and_models(self):
        ref = EngineRef(
            type="composite", policy_hash="sha256:x", model_snapshot_ids=("m1", "m2")
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
                    engine_ref=EngineRef(type="rule_based", policy_hash="sha256:s"),
                    as_of=context.as_of or "",
                    direction=direction,
                    target_exposure=1.0 if direction == "LONG" else 0.0,
                    reason_codes=("STREAK",),
                    decision_components={"streak": streak},
                )

        policy = StreakPolicy()
        assert isinstance(policy, Policy)
        ctx = PolicyContext(as_of="t")
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
                engine_ref=EngineRef(type="rule_based", policy_hash="sha256:x"),
                as_of="t",
                direction="UP",
                target_exposure=1.0,
            )

    def test_to_dict_shape(self):
        d = StrategyDecision(
            sleeve_id="s",
            strategy_version="1",
            engine_ref=EngineRef(type="rule_based", policy_hash="sha256:x"),
            as_of="t",
            direction="FLAT",
            target_exposure=0.0,
        ).to_dict()
        assert d["engine_ref"] == {"type": "rule_based", "policy_hash": "sha256:x"}
        assert d["decision_fingerprint"].startswith("sha256:")
        assert d["signal_id"].startswith("s:t:")

    def test_missing_default_exposure_rejected(self):
        spec = ma200_spec()
        del spec["resolution"]["default_target_exposure"]
        with pytest.raises(ValueError, match="default_target_exposure"):
            DeclarativePolicy(spec)
