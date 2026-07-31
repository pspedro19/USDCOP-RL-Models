from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import yaml

import scripts.pipeline.generate_usdcop_h1_regime_shadow_v2 as shadow


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = yaml.safe_load(
    (ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v2.yaml")
    .read_text(encoding="utf-8")
)


def test_live_model_exactly_reproduces_registered_research_probability():
    frame, _ = shadow.build_live_frame(CONTRACT, pd.Timestamp("2026-07-21"))
    origin = pd.Timestamp("2024-12-27")
    origin_index = int(frame.index[pd.to_datetime(frame["date"]).eq(origin)][0])
    live = shadow.make_prediction(frame, origin_index, CONTRACT)
    research = pd.read_csv(
        ROOT / "reports/usdcop_long_history_directional_tournament_predictions.csv"
    )
    expected = research[
        research["horizon_days"].eq(1)
        & research["origin_date"].str.startswith("2024-12-27")
    ].iloc[0]
    assert live["model_variant_key"] == "long_price|C=0.1|balanced|hl=1260"
    assert live["probability_up"] == float(expected["probability_up"])


def test_commit_window_is_current_friday_after_registered_time_only():
    bogota = ZoneInfo("America/Bogota")
    assert not shadow.commit_window_open(
        datetime(2026, 7, 31, 15, 29, tzinfo=bogota), CONTRACT
    )
    assert shadow.commit_window_open(
        datetime(2026, 7, 31, 15, 30, tzinfo=bogota), CONTRACT
    )
    assert not shadow.commit_window_open(
        datetime(2026, 7, 30, 16, 0, tzinfo=bogota), CONTRACT
    )


def test_cli_rejects_manual_clock_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["generate_usdcop_h1_regime_shadow_v2.py", "--as-of-datetime", "2026-07-31T15:30:00-05:00"],
    )
    with pytest.raises(SystemExit) as exc:
        shadow.parse_args()
    assert exc.value.code == 2


def test_current_week_rule_rejects_stale_or_future_origin():
    bogota = ZoneInfo("America/Bogota")
    now = datetime(2026, 7, 31, 15, 30, tzinfo=bogota)
    stale = pd.DataFrame({"date": pd.to_datetime(["2026-07-24"]), "close": [1.0]})
    with pytest.raises(RuntimeError, match="stale origin"):
        shadow.validate_current_commit(stale, now, CONTRACT)

    matured = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-31", "2026-08-03"]),
        "close": [1.0, 1.1],
    })
    with pytest.raises(RuntimeError, match="stale origin"):
        shadow.validate_current_commit(matured, now, CONTRACT)


def test_hash_chain_detects_record_tampering():
    first = {
        "iso_week": "2026-W31",
        "value": 1,
        "previous_prediction_sha256": shadow.GENESIS_HASH,
    }
    first["prediction_record_sha256"] = shadow.canonical_hash(
        first, "prediction_record_sha256"
    )
    second = {
        "iso_week": "2026-W32",
        "value": 2,
        "previous_prediction_sha256": first["prediction_record_sha256"],
    }
    second["prediction_record_sha256"] = shadow.canonical_hash(
        second, "prediction_record_sha256"
    )
    assert shadow.validate_hash_chain(
        [first, second],
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    ) == second["prediction_record_sha256"]
    tampered = deepcopy(second)
    tampered["value"] = 999
    with pytest.raises(ValueError, match="Tampered"):
        shadow.validate_hash_chain(
            [first, tampered],
            hash_field="prediction_record_sha256",
            previous_field="previous_prediction_sha256",
        )


def test_live_revisions_on_or_before_baseline_are_ignored(tmp_path):
    contract = deepcopy(CONTRACT)
    baseline_frame, _ = shadow.build_live_frame(
        contract, pd.Timestamp("2026-07-21")
    )
    expected_close = float(
        baseline_frame.loc[
            pd.to_datetime(baseline_frame["date"]).eq(pd.Timestamp("2026-07-21")),
            "close",
        ].iloc[0]
    )
    live = pd.DataFrame([
        {
            "time": pd.Timestamp("2026-07-21", tz="UTC"),
            "open": 9999.0, "high": 9999.0, "low": 9999.0, "close": 9999.0,
        },
        {
            "time": pd.Timestamp("2026-07-22", tz="UTC"),
            "open": 4100.0, "high": 4110.0, "low": 4090.0, "close": 4105.0,
        },
    ])
    live_path = tmp_path / "live.parquet"
    live.to_parquet(live_path, index=False)
    contract["data"]["live_current_file"] = str(live_path)
    frame, _ = shadow.build_live_frame(contract, pd.Timestamp("2026-07-22"))
    cutoff_close = float(
        frame.loc[
            pd.to_datetime(frame["date"]).eq(pd.Timestamp("2026-07-21")), "close"
        ].iloc[0]
    )
    assert cutoff_close == expected_close
    assert float(frame.iloc[-1]["close"]) == 4105.0
