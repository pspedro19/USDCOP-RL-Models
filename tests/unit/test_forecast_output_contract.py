"""ForecastOutput contract (CTR-FORECAST-OUTPUT-001, FABRIC §15.2 / BL-15).

Guards the DIAGNOSTIC wall:
- happy path validates and round-trips;
- NaN/Inf are rejected physically (validate raises);
- ``diagnostic_only`` is forced True no matter what the caller passes;
- a forecast_output dict is REJECTED BY TYPE where a StrategyTrade is expected
  (the allocator/book only accepts strategy records — rejection is physical,
  not convention);
- the TS mirror declares the same fields (parity, not string vibes).
"""
from __future__ import annotations

import math
from dataclasses import fields
from pathlib import Path

import pytest

from src.contracts.forecast_output import (
    PREDICTION_TYPES,
    DirectionProbability,
    ForecastOutput,
    ForecastOutputError,
    ForecastPrediction,
)
from src.contracts.strategy_schema import StrategyTrade

ROOT = Path(__file__).resolve().parents[2]
TS_MIRROR = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts" / "forecast-output.contract.ts"


def make_forecast(**overrides) -> ForecastOutput:
    base = dict(
        forecast_id="fo-2026-07-27-ridge-h5",
        forecast_spec_id="usdcop_forecast_zoo_v3",
        asset="usdcop",
        model_id="ridge_v2",
        horizon="5d",
        as_of="2026-07-27T00:00:00Z",
        available_at="2026-07-27T00:05:00Z",
        target_time="2026-08-03T00:00:00Z",
        prediction=ForecastPrediction(type="return", point=0.0062, lower=-0.0110, upper=0.0240),
        direction_probability=DirectionProbability(up=0.58),
        model_fingerprint="sha256:deadbeef",
        data_snapshot_id="snap-0001",
    )
    base.update(overrides)
    return ForecastOutput(**base)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_valid_record_passes_and_chains(self):
        fo = make_forecast()
        assert fo.validate() is fo

    def test_round_trip_dict(self):
        fo = make_forecast()
        d = fo.to_dict()
        again = ForecastOutput.from_dict(d)
        again.validate()
        assert again == fo

    def test_from_dict_accepts_nested_dicts(self):
        d = make_forecast().to_dict()
        d["prediction"] = {"type": "return", "point": 0.01, "lower": -0.02, "upper": 0.03}
        d["direction_probability"] = {"up": 0.5}
        fo = ForecastOutput.from_dict(d)
        fo.validate()
        assert fo.prediction == ForecastPrediction(type="return", point=0.01, lower=-0.02, upper=0.03)
        assert fo.direction_probability == DirectionProbability(up=0.5)

    def test_from_dict_rejects_unknown_keys(self):
        """BL-15 remedio: silently dropping unknown keys was the laundering hole
        (a StrategyTrade-shaped payload lost its actionable fields and passed)."""
        d = make_forecast().to_dict()
        d["not_a_field"] = "ignored-before-the-remedio"
        with pytest.raises(ForecastOutputError, match="not_a_field"):
            ForecastOutput.from_dict(d)

    def test_from_dict_rejects_unknown_nested_keys(self):
        d = make_forecast().to_dict()
        d["prediction"] = {"type": "return", "point": 0.01, "junk": 1}
        with pytest.raises(ForecastOutputError, match="junk"):
            ForecastOutput.from_dict(d)

    def test_direction_probability_is_optional(self):
        make_forecast(direction_probability=None).validate()

    def test_interval_is_optional(self):
        make_forecast(
            prediction=ForecastPrediction(type="price", point=4123.5)
        ).validate()


# ---------------------------------------------------------------------------
# NaN / Inf rejection
# ---------------------------------------------------------------------------

class TestNanInfRejection:
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_point_rejected(self, bad):
        fo = make_forecast(prediction=ForecastPrediction(type="return", point=bad))
        with pytest.raises(ForecastOutputError, match="prediction.point"):
            fo.validate()

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_bounds_rejected(self, bad):
        fo = make_forecast(
            prediction=ForecastPrediction(type="return", point=0.0, lower=-0.1, upper=bad)
        )
        with pytest.raises(ForecastOutputError, match="prediction.upper"):
            fo.validate()

    @pytest.mark.parametrize("bad", [math.nan, math.inf])
    def test_direction_probability_rejected(self, bad):
        fo = make_forecast(direction_probability=DirectionProbability(up=bad))
        with pytest.raises(ForecastOutputError, match="direction_probability.up"):
            fo.validate()


# ---------------------------------------------------------------------------
# Other contract violations
# ---------------------------------------------------------------------------

class TestContractViolations:
    def test_empty_required_string(self):
        with pytest.raises(ForecastOutputError, match="model_fingerprint"):
            make_forecast(model_fingerprint="").validate()

    def test_bad_prediction_type(self):
        fo = make_forecast(prediction=ForecastPrediction(type="vibes", point=0.0))
        with pytest.raises(ForecastOutputError, match="prediction.type"):
            fo.validate()

    def test_probability_out_of_range(self):
        fo = make_forecast(direction_probability=DirectionProbability(up=1.3))
        with pytest.raises(ForecastOutputError, match=r"\[0, 1\]"):
            fo.validate()

    def test_look_ahead_available_before_as_of(self):
        fo = make_forecast(available_at="2026-07-26T23:59:00Z")
        with pytest.raises(ForecastOutputError, match="anti-look-ahead"):
            fo.validate()

    def test_target_time_must_be_after_as_of(self):
        fo = make_forecast(target_time="2026-07-27T00:00:00Z")
        with pytest.raises(ForecastOutputError, match="target_time"):
            fo.validate()

    def test_point_outside_interval(self):
        fo = make_forecast(
            prediction=ForecastPrediction(type="return", point=0.5, lower=-0.01, upper=0.02)
        )
        with pytest.raises(ForecastOutputError, match="within"):
            fo.validate()

    def test_bad_timestamp(self):
        with pytest.raises(ForecastOutputError, match="as_of"):
            make_forecast(as_of="viernes").validate()


# ---------------------------------------------------------------------------
# diagnostic_only is forced True (the wall)
# ---------------------------------------------------------------------------

class TestDiagnosticOnlyForced:
    def test_constructor_cannot_unset_it(self):
        fo = make_forecast(diagnostic_only=False)
        assert fo.diagnostic_only is True
        fo.validate()

    def test_from_dict_rejects_an_actionable_claim(self):
        """The ingest wall REJECTS instead of coercing: silently flipping
        ``diagnostic_only`` to True laundered an actionable claim into a
        clean-looking forecast (BL-15 remedio, point 4)."""
        d = make_forecast().to_dict()
        d["diagnostic_only"] = False
        with pytest.raises(ForecastOutputError, match="DIAGNOSTIC"):
            ForecastOutput.from_dict(d)

    def test_serialized_record_carries_true(self):
        assert make_forecast().to_dict()["diagnostic_only"] is True


# ---------------------------------------------------------------------------
# Physical rejection where a strategy record is expected (allocator/book side)
# ---------------------------------------------------------------------------

class TestTypeRejection:
    def test_forecast_dict_rejected_by_strategy_trade_typecheck(self):
        """The book/ledger constructs StrategyTrade from records it accepts.

        A forecast_output dict shares no fields with StrategyTrade, so the
        type-check itself rejects it — no convention involved.
        """
        fo_dict = make_forecast().to_dict()
        with pytest.raises(TypeError):
            StrategyTrade(**fo_dict)

    def test_forecast_is_not_a_strategy_trade(self):
        assert not isinstance(make_forecast(), StrategyTrade)

    def test_no_field_overlap_with_strategy_trade(self):
        """If someone adds a shared field, the TypeError above could silently
        weaken. Pin zero overlap so the physical rejection stays physical."""
        fo_fields = {f.name for f in fields(ForecastOutput)}
        trade_fields = {f.name for f in fields(StrategyTrade)}
        assert not (fo_fields & trade_fields)


# ---------------------------------------------------------------------------
# TS mirror parity (contract-change skill step 6)
# ---------------------------------------------------------------------------

class TestTsMirrorParity:
    def test_mirror_exists(self):
        assert TS_MIRROR.is_file(), "TS mirror missing — contracts change in pairs"

    def test_every_python_field_is_declared_in_ts(self):
        text = TS_MIRROR.read_text(encoding="utf-8")
        for f in fields(ForecastOutput):
            assert f"{f.name}" in text, f"field {f.name!r} missing from TS mirror"

    def test_ts_diagnostic_only_is_literal_true(self):
        text = TS_MIRROR.read_text(encoding="utf-8")
        assert "diagnostic_only: true;" in text, (
            "TS mirror must type diagnostic_only as the literal `true`"
        )

    def test_prediction_types_match(self):
        text = TS_MIRROR.read_text(encoding="utf-8")
        for t in PREDICTION_TYPES:
            assert f"'{t}'" in text, f"prediction type {t!r} missing from TS mirror"


# ---------------------------------------------------------------------------
# BL-15 REMEDIO (CODEX rejection vs 91fe7b6)
# ---------------------------------------------------------------------------

class TestMixedTimezoneIsTypedRejection:
    """(1) Comparing naive vs aware datetimes raises a raw TypeError in Python.
    A contract must reject it as a CONTRACT error, never leak TypeError."""

    def test_mixed_aware_as_of_naive_available_at(self):
        fo = make_forecast(
            as_of="2026-07-27T00:00:00Z",
            available_at="2026-07-27T00:05:00",     # naive
            target_time="2026-08-03T00:00:00Z",
        )
        with pytest.raises(ForecastOutputError, match="naive"):
            fo.validate()

    def test_mixed_naive_as_of_aware_target(self):
        fo = make_forecast(
            as_of="2026-07-27T00:00:00",            # naive
            available_at="2026-07-27T00:05:00",
            target_time="2026-08-03T00:00:00+00:00",
        )
        with pytest.raises(ForecastOutputError, match="naive"):
            fo.validate()


class TestIngestWall:
    """(4) A payload that breaks the contract must NOT be able to enter."""

    def test_actionable_payload_is_not_laundered_into_a_forecast(self):
        from src.contracts.forecast_output import ingest_forecast_output
        payload = make_forecast().to_dict()
        payload["diagnostic_only"] = False        # claims to be actionable
        payload["pnl"] = 1234.5                   # StrategyTrade smell
        payload["exit_reason"] = "TP"
        with pytest.raises(ForecastOutputError):
            ingest_forecast_output(payload)

    def test_unknown_keys_are_rejected_not_silently_dropped(self):
        from src.contracts.forecast_output import ingest_forecast_output
        payload = make_forecast().to_dict()
        payload["not_a_field"] = "smuggled"
        with pytest.raises(ForecastOutputError, match="not_a_field"):
            ingest_forecast_output(payload)


# ---------------------------------------------------------------------------
# (2) SHARED CASE TABLE — bilateral parity Py <-> TS
# ---------------------------------------------------------------------------
# The SAME fixture is executed by this runner and by the Vitest twin
# (usdcop-trading-dashboard/tests/unit/contracts/forecast-output-parity.test.ts).
# Nothing is duplicated literally; both recompute the fixture's content SHA-256
# and refuse to run on drift.

import hashlib
import json
import re

from src.contracts.forecast_output import validate_forecast_payload

FIXTURE_PATH = ROOT / "tests" / "fixtures" / "forecast_output_cases.v1.json"
_SHA_FIELD = re.compile(r'"content_sha256":\s*"([0-9a-f]{64})"')


def _decode(value):
    """Decode fixture sentinels into native Python values."""
    if isinstance(value, dict):
        if set(value) == {"$nonfinite"}:
            return {
                "NaN": math.nan,
                "Infinity": math.inf,
                "-Infinity": -math.inf,
            }[value["$nonfinite"]]
        return {k: _decode(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v) for v in value]
    return value


def _load_cases():
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
    if doc.get("fixture") != "forecast_output_cases" or doc.get("version") != "v1":
        raise AssertionError("unexpected fixture identity/version")
    cases = [(c["id"], _decode(c["payload"]), c["expect"]) for c in doc["cases"]]
    ids = [c[0] for c in cases]
    assert len(set(ids)) == len(ids), "duplicate case ids in fixture"
    assert len(cases) == doc["case_count"], "case_count pin does not match"
    return cases


CASES = _load_cases()


class TestSharedCaseTable:
    def test_fixture_pin(self):
        """Anti-drift: the case table cannot grow/shrink silently."""
        assert len(CASES) == 96, (
            "case-table-v1:96 — update BOTH runners and the pin deliberately"
        )

    @pytest.mark.parametrize("case_id,payload,expect", CASES,
                             ids=[c[0] for c in CASES])
    def test_case(self, case_id, payload, expect):
        errors = validate_forecast_payload(payload)
        verdict = "invalid" if errors else "valid"
        assert verdict == expect, (
            f"{case_id}: expected {expect}, got {verdict} (errors={errors})"
        )
