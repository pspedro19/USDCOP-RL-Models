"""Analytic controls for PPO training; no market trials are charged."""

from __future__ import annotations

import numpy as np

from src.research.synthetic_sessions import (
    Fixture, make_sessions, oracle_result, oracle_weights,
)


def test_synthetic_fixture_is_deterministic_and_has_known_dimensions():
    a = make_sessions(Fixture.SIGNAL_ABOVE_COST, n=3, seed=42)
    b = make_sessions(Fixture.SIGNAL_ABOVE_COST, n=3, seed=42)
    for left, right in zip(a, b, strict=True):
        assert np.array_equal(left.close, right.close)
        assert np.array_equal(left.market, right.market)


def test_noise_and_subcost_signal_optima_are_flat():
    for fixture in (Fixture.NOISE_WITH_COST, Fixture.SIGNAL_BELOW_COST):
        for spec in make_sessions(fixture, n=20, seed=7):
            result = oracle_result(fixture, spec)
            assert np.all(oracle_weights(fixture, spec) == 0.0)
            assert result.daily_return == 0.0


def test_above_cost_signal_oracle_is_profitable_and_beats_flat():
    specs = make_sessions(Fixture.SIGNAL_ABOVE_COST, n=20, seed=7)
    values = [oracle_result(Fixture.SIGNAL_ABOVE_COST, spec).daily_return for spec in specs]
    assert min(values) > 0.0

