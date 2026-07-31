from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
import yaml

import scripts.pipeline.generate_usdcop_h1_daily_shadow_v1 as daily
from scripts.validation.evaluate_usdcop_h1_daily_shadow_v1 import (
    anytime_directional_evidence,
)


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = yaml.safe_load(
    (ROOT / "config/forecast_experiments/usdcop_h1_daily_shadow_v1.yaml")
    .read_text(encoding="utf-8")
)


@pytest.fixture(scope="module")
def baseline_frame() -> pd.DataFrame:
    frame, _ = daily.build_live_frame(CONTRACT, pd.Timestamp("2026-07-22"))
    return frame


def _prediction(frame: pd.DataFrame, origin: str) -> dict:
    matches = frame.index[pd.to_datetime(frame["date"]).eq(pd.Timestamp(origin))]
    assert len(matches) == 1
    return daily.make_daily_prediction(frame, int(matches[0]), CONTRACT)


def test_weekly_model_is_fixed_within_week_and_refreshes_next_week(baseline_frame):
    monday = _prediction(baseline_frame, "2024-12-23")
    friday = _prediction(baseline_frame, "2024-12-27")
    next_monday = _prediction(baseline_frame, "2024-12-30")
    assert monday["weekly_training_anchor"] == "2024-12-20"
    assert friday["weekly_training_anchor"] == "2024-12-20"
    assert monday["weekly_model_sha256"] == friday["weekly_model_sha256"]
    assert next_monday["weekly_training_anchor"] == "2024-12-27"
    assert next_monday["weekly_model_sha256"] != monday["weekly_model_sha256"]
    assert monday["model_variant_key"] == "long_price|C=0.1|balanced|hl=1260"


def test_daily_commit_window_and_prelaunch_guard(baseline_frame):
    bogota = ZoneInfo("America/Bogota")
    assert not daily.commit_window_open(
        datetime(2026, 7, 27, 15, 29, tzinfo=bogota), CONTRACT
    )
    assert daily.commit_window_open(
        datetime(2026, 7, 27, 15, 30, tzinfo=bogota), CONTRACT
    )
    assert not daily.commit_window_open(
        datetime(2026, 7, 26, 16, 0, tzinfo=bogota), CONTRACT
    )
    with pytest.raises(RuntimeError, match="precedes prospective launch"):
        daily.validate_current_commit(
            baseline_frame,
            datetime(2026, 7, 22, 16, 0, tzinfo=bogota),
            CONTRACT,
        )


def test_current_date_rule_rejects_stale_origin():
    bogota = ZoneInfo("America/Bogota")
    stale = pd.DataFrame({"date": pd.to_datetime(["2026-07-24"]), "close": [1.0]})
    with pytest.raises(RuntimeError, match="stale daily origin"):
        daily.validate_current_commit(
            stale,
            datetime(2026, 7, 27, 15, 30, tzinfo=bogota),
            CONTRACT,
        )


def test_daily_hash_chain_allows_multiple_origins_in_same_iso_week():
    records = []
    previous = daily.GENESIS_HASH
    for origin in ("2026-07-27", "2026-07-28"):
        record = {
            "origin_date": origin,
            "iso_week": "2026-W31",
            "value": len(records),
            "previous_prediction_sha256": previous,
        }
        record["prediction_record_sha256"] = daily.canonical_hash(
            record, "prediction_record_sha256"
        )
        records.append(record)
        previous = record["prediction_record_sha256"]
    assert daily.validate_daily_hash_chain(
        records,
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    ) == previous
    tampered = deepcopy(records)
    tampered[1]["value"] = 99
    with pytest.raises(ValueError, match="Tampered"):
        daily.validate_daily_hash_chain(
            tampered,
            hash_field="prediction_record_sha256",
            previous_field="previous_prediction_sha256",
        )


def test_cli_rejects_manual_clock_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_usdcop_h1_daily_shadow_v1.py", "--as-of-datetime", "2026-07-27T15:30:00-05:00"],
    )
    with pytest.raises(SystemExit) as exc:
        daily.parse_args()
    assert exc.value.code == 2


def test_anytime_evidence_is_fail_closed_and_detects_extreme_skill():
    empty = anytime_directional_evidence(np.asarray([], dtype=int))
    assert empty["lower_confidence_bound_95"] is None
    noise = anytime_directional_evidence(np.asarray([0, 1] * 100, dtype=int))
    assert noise["lower_confidence_bound_95"] <= 0.50
    perfect = anytime_directional_evidence(np.ones(100, dtype=int))
    assert perfect["e_value_at_0_50"] >= 20.0
    assert perfect["lower_confidence_bound_95"] > 0.50
