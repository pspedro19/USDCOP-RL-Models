from __future__ import annotations

import numpy as np
import pytest

from scripts.validation.evaluate_usdcop_h1_shadow_v2 import (
    execution_summary,
    pesaran_timmermann,
    risk_coverage_curve,
    wilson_interval,
)


def test_pesaran_timmermann_detects_perfect_balanced_directions() -> None:
    actual = np.asarray(([0, 1] * 60), dtype=int)
    result = pesaran_timmermann(actual, actual.copy())
    assert result["valid"] is True
    assert result["statistic"] > 5.0
    assert result["one_sided_p"] < 0.001


def test_pesaran_timmermann_is_fail_closed_for_degenerate_series() -> None:
    actual = np.ones(120, dtype=int)
    predicted = np.ones(120, dtype=int)
    result = pesaran_timmermann(actual, predicted)
    assert result["valid"] is False
    assert result["one_sided_p"] is None


def test_wilson_interval_contains_observed_rate() -> None:
    interval = wilson_interval(60, 100)
    assert interval["lower"] < 0.60 < interval["upper"]
    assert wilson_interval(0, 0) == {"lower": None, "upper": None}


def test_risk_coverage_excludes_flat_and_applies_confidence() -> None:
    matured = [
        (
            {"decision": "UP", "probability_confidence": 0.20},
            {"actual": 1},
        ),
        (
            {"decision": "DOWN", "probability_confidence": 0.08},
            {"actual": 0},
        ),
        (
            {"decision": "FLAT", "probability_confidence": 0.90},
            {"actual": 1},
        ),
    ]
    curve = risk_coverage_curve(matured)
    at_zero = next(point for point in curve if point["minimum_probability_confidence"] == 0)
    at_ten = next(point for point in curve if point["minimum_probability_confidence"] == 0.10)
    assert at_zero["signals"] == 2
    assert at_zero["coverage_of_matured_weeks"] == 2 / 3
    assert at_zero["directional_accuracy"] == 1.0
    assert at_ten["signals"] == 1


def test_execution_summary_never_self_authorizes_cost_gate() -> None:
    signals = [
        (
            {"decision": "UP", "base_price": 4_000.0},
            {"actual_price": 4_040.0},
        )
    ]
    result = execution_summary(signals)
    assert result["gross_mean_return_per_signal"] == pytest.approx(0.01)
    assert result["break_even_round_trip_cost_bps"] == pytest.approx(100.0)
    assert result["registered_cost_assumption_bps"] is None
    assert result["positive_net_value_after_costs"] is False
