"""
Unit tests — policy BACKEND contract (BL-46 R4/R5, CTR-POLICY-BACKEND-001).

Same discipline as C-004 remedio-4: ONE shared fixture
(``tests/fixtures/policy_backend_cases.v1.json``) loaded (never duplicated),
SHA-verified before running, and EXECUTED here against the real Python
constructors. The Vitest twin
(``usdcop-trading-dashboard/tests/unit/contracts/policy-backend-parity.test.ts``)
loads THE SAME file and executes it against the TS validators — same verdict,
case by case.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path

import pytest

from src.contracts.policy import EngineRef, PolicyContext, StrategyDecision
from src.contracts.policy_dsl import DeclarativePolicy
from src.contracts.policy_version import (
    CONFIG_FIELD_TYPES,
    DECISION_SCHEMA_VERSIONS,
    IMPLEMENTATION_MODES,
    PRESENTATION_FORMATS,
    URI_SCHEMES,
    ConfigFieldSpec,
    ConfigSchema,
    PolicyVersionRecord,
    PresentationComponent,
    PresentationSpec,
    StrategySignalRecord,
    build_config_schema,
    instant_epoch_seconds,
)
from src.contracts.rule_trace import RuleTrace
from src.policy_engine import evaluate_policy, publish_signal

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests" / "fixtures" / "policy_backend_cases.v1.json"
TS_MIRROR = (
    ROOT / "usdcop-trading-dashboard" / "lib" / "contracts" / "policy-version.contract.ts"
)
_SHA_FIELD = re.compile(r'"content_sha256":\s*"([0-9a-f]{64})"')


def _decode(value):
    if isinstance(value, dict):
        keys = set(value)
        if keys == {"$nonfinite"}:
            return {"NaN": float("nan"), "Infinity": float("inf"),
                    "-Infinity": float("-inf")}[value["$nonfinite"]]
        if keys == {"$pytype", "value"}:
            if value["$pytype"] == "decimal":
                return Decimal(value["value"])
            raise AssertionError(f"unknown $pytype {value['$pytype']!r}")
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def load_fixture():
    raw = FIXTURE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
    m = _SHA_FIELD.search(raw)
    if not m:
        raise AssertionError("fixture must declare a 64-hex content_sha256")
    declared = m.group(1)
    actual = hashlib.sha256(raw.replace(declared, "", 1).encode("utf-8")).hexdigest()
    if actual != declared:
        raise AssertionError(
            f"FIXTURE DRIFT: declared {declared} != recomputed {actual}"
        )
    doc = json.loads(raw)
    if doc.get("fixture") != "policy_backend_cases" or doc.get("version") != "v1":
        raise AssertionError("unexpected fixture identity/version")
    cases = [
        (c["id"], c["target"], _decode(c["payload"]), c["expect"]) for c in doc["cases"]
    ]
    ids = [c[0] for c in cases]
    if len(set(ids)) != len(ids):
        raise AssertionError("duplicate case ids")
    return doc, cases


FIXTURE_DOC, CASES = load_fixture()


def _run(target, payload):
    try:
        if target == "policy_version":
            PolicyVersionRecord.from_dict(payload)
        elif target == "strategy_signal":
            kwargs = dict(payload)
            if "reason_codes" in kwargs and isinstance(kwargs["reason_codes"], list):
                kwargs["reason_codes"] = tuple(kwargs["reason_codes"])
            StrategySignalRecord(**kwargs)
        elif target == "trace":
            RuleTrace.from_dict(payload)
        elif target == "engine_ref":
            EngineRef(**payload)
        elif target == "presentation":
            components = payload.get("components", [])
            if not isinstance(components, list):
                raise ValueError("components must be a list")
            PresentationSpec(
                engine_label=payload.get("engine_label"),
                description=payload.get("description"),
                components=tuple(PresentationComponent(**c) for c in components),
            )
        elif target == "config_field":
            build_config_schema([payload])
        elif target == "config_values":
            schema = build_config_schema(payload["fields"])
            errors = schema.validate_values(payload["values"])
            if errors:
                raise ValueError("; ".join(errors))
        else:  # pragma: no cover
            raise AssertionError(f"unknown target {target!r}")
        return "valid"
    except (ValueError, TypeError, KeyError):
        return "invalid"


class TestSharedBackendCaseTable:
    @pytest.mark.parametrize(
        "case_id,target,payload,expect", CASES, ids=[c[0] for c in CASES]
    )
    def test_case(self, case_id, target, payload, expect):
        assert _run(target, payload) == expect, (
            f"case {case_id!r}: Python verdict diverges from the fixture"
        )

    def test_fixture_sha_and_unique_ids(self):
        _doc, cases = load_fixture()
        assert len(cases) >= 90, "never shrink the table"

    def test_ts_twin_loads_the_same_fixture(self):
        twin = (
            ROOT / "usdcop-trading-dashboard" / "tests" / "unit" / "contracts"
            / "policy-backend-parity.test.ts"
        ).read_text(encoding="utf-8")
        assert "policy_backend_cases.v1.json" in twin
        assert "pv_manifest_uri_relative" not in twin, "no duplicated case literals"


class TestMirrorSets:
    """The TS mirror must declare the SAME whitelists (set equality)."""

    def _literals(self, name: str) -> set[str]:
        text = TS_MIRROR.read_text(encoding="utf-8")
        m = re.search(rf"{name}\s*=\s*\[(.*?)\]\s*as const", text, re.S)
        assert m, f"{name} not found in the TS mirror"
        return set(re.findall(r"'([^']+)'", m.group(1)))

    def test_implementation_modes(self):
        assert self._literals("IMPLEMENTATION_MODES") == set(IMPLEMENTATION_MODES)

    def test_decision_schema_versions(self):
        assert self._literals("DECISION_SCHEMA_VERSIONS") == set(DECISION_SCHEMA_VERSIONS)

    def test_presentation_formats(self):
        assert self._literals("PRESENTATION_FORMATS") == set(PRESENTATION_FORMATS)

    def test_config_field_types(self):
        assert self._literals("CONFIG_FIELD_TYPES") == set(CONFIG_FIELD_TYPES)

    def test_uri_schemes(self):
        assert self._literals("URI_SCHEMES") == set(URI_SCHEMES)


class TestInstantArithmetic:
    def test_offsets_are_resolved_not_string_compared(self):
        assert instant_epoch_seconds("2026-07-28T10:00:00-05:00") == (
            instant_epoch_seconds("2026-07-28T15:00:00Z")
        )

    def test_leap_day_is_real(self):
        assert instant_epoch_seconds("2028-03-01T00:00:00Z") - instant_epoch_seconds(
            "2028-02-28T00:00:00Z"
        ) == 2 * 86400


class TestPresentationIsNotEconomic:
    """§9.1: renaming a label changes presentation_hash and NOTHING else."""

    def _spec(self, label: str) -> PresentationSpec:
        return PresentationSpec(
            engine_label="Reglas MA200",
            description="Exposicion sobre la media de 200 sesiones",
            components=(PresentationComponent(key="close", label=label, format="price"),),
        )

    def test_label_change_changes_presentation_hash(self):
        assert self._spec("Cierre").presentation_hash() != self._spec(
            "Precio de cierre"
        ).presentation_hash()

    def test_policy_hash_is_untouched_by_presentation(self):
        policy = DeclarativePolicy(_ma200_spec())
        before = policy.policy_hash
        self._spec("Cierre").presentation_hash()
        assert DeclarativePolicy(_ma200_spec()).policy_hash == before


def _ma200_spec() -> dict:
    return {
        "id": "spx500_daily_ma200_v1",
        "version": "2.0.0",
        "policy_hash": "sha256:" + "ab" * 32,
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
                "when": {"operator": "greater_than", "left": "feature.close",
                         "right": "feature.ma_200"},
                "output": {"direction": "LONG", "target_exposure": 1.0,
                           "reason_code": "CLOSE_ABOVE_MA200"},
            }
        ],
    }


class TestEngineEmitsRenderableFacts:
    """The renderer needs facts, not inference (invariant 7)."""

    def test_trace_carries_threshold_and_winner(self):
        policy = DeclarativePolicy(_ma200_spec())
        decision = policy.evaluate(
            {"close": 6412.8, "ma_200": 5984.2}, PolicyContext(as_of="2026-07-27")
        )
        trace = decision.rule_trace
        assert trace is not None
        assert trace.winning_rule_id == "trend_on"
        assert trace.fallback_applied is False
        assert trace.rules[0].threshold == {"ma_200": 5984.2}

    def test_fallback_is_declared_when_no_rule_fires(self):
        policy = DeclarativePolicy(_ma200_spec())
        decision = policy.evaluate(
            {"close": 100.0, "ma_200": 5984.2}, PolicyContext(as_of="2026-07-27")
        )
        assert decision.direction == "FLAT"
        assert decision.rule_trace.fallback_applied is True
        assert decision.rule_trace.winning_rule_id is None


class TestSingleEvaluationLibrary:
    """Backtest/paper/live publish through the same runner (invariant 4)."""

    def test_evaluate_and_publish_roundtrip(self):
        policy = DeclarativePolicy(_ma200_spec())
        decision = evaluate_policy(
            policy, {"close": 6412.8, "ma_200": 5984.2},
            PolicyContext(as_of="2026-07-27"),
        )
        record = publish_signal(
            decision,
            policy_version_id="pv_ma200_001",
            instrument_id="spx500_index",
            valid_from="2026-07-28T13:30:00Z",
            valid_until="2026-07-28T20:00:00Z",
            created_at="2026-07-27T20:05:00Z",
        )
        assert record.signal_id == decision.signal_id
        assert record.direction == "LONG"
        assert json.loads(record.to_json())["decision_schema_version"] == (
            DECISION_SCHEMA_VERSIONS[0]
        )

    def test_missing_input_fail_closed_raises(self):
        policy = DeclarativePolicy(_ma200_spec())
        with pytest.raises(ValueError):
            evaluate_policy(policy, {"close": 1.0}, PolicyContext(as_of="2026-07-27"))

    def test_missing_input_flat_fallback_is_explicit(self):
        policy = DeclarativePolicy(_ma200_spec())
        decision = evaluate_policy(
            policy, {"close": 1.0}, PolicyContext(as_of="2026-07-27"),
            missing_input_policy="FLAT",
        )
        assert decision.direction == "FLAT"
        assert decision.reason_codes == ("INPUT_MISSING",)
        assert decision.rule_trace.fallback_applied is True

    def test_stale_snapshot_fail_closed(self):
        policy = DeclarativePolicy(_ma200_spec())
        ctx = PolicyContext(as_of="2026-07-27", extras={"snapshot_is_stale": True})
        with pytest.raises(ValueError):
            evaluate_policy(policy, {"close": 6412.8, "ma_200": 5984.2}, ctx)

    def test_unknown_fallback_mode_rejected(self):
        policy = DeclarativePolicy(_ma200_spec())
        with pytest.raises(ValueError):
            evaluate_policy(
                policy, {"close": 6412.8, "ma_200": 5984.2},
                PolicyContext(as_of="2026-07-27"), missing_input_policy="IGNORE",
            )


class TestSignalFromDecision:
    def test_from_decision_requires_a_decision(self):
        with pytest.raises(ValueError):
            StrategySignalRecord.from_decision(
                {"direction": "LONG"},  # type: ignore[arg-type]
                policy_version_id="pv", instrument_id="i",
                valid_from="2026-07-28T13:30:00Z",
                valid_until="2026-07-28T20:00:00Z",
                created_at="2026-07-27T20:05:00Z",
            )

    def test_signal_id_travels_from_the_decision(self):
        decision = StrategyDecision(
            sleeve_id="s1",
            strategy_version="1.0.0",
            engine_ref=EngineRef(type="rule_based", policy_hash="sha256:" + "ab" * 32),
            as_of="2026-07-27",
            direction="FLAT",
            target_exposure=0.0,
        )
        record = StrategySignalRecord.from_decision(
            decision, policy_version_id="pv1", instrument_id="spx500_index",
            valid_from="2026-07-28T13:30:00Z", valid_until="2026-07-28T20:00:00Z",
            created_at="2026-07-27T20:05:00Z",
        )
        assert record.signal_id == decision.signal_id
        assert record.decision_fingerprint == decision.decision_fingerprint


class TestConfigSchema:
    def test_duplicate_keys_rejected(self):
        f = ConfigFieldSpec(key="a", label="A", type="boolean")
        with pytest.raises(ValueError):
            ConfigSchema(fields=(f, f))

    def test_unknown_key_is_an_error_not_a_silent_drop(self):
        schema = build_config_schema(
            [{"key": "a", "label": "A", "type": "boolean"}]
        )
        assert schema.validate_values({"b": True}) == ["unknown config key: b"]
