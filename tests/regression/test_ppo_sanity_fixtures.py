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


def test_sanity_protocol_stops_at_first_recipe(monkeypatch):
    import scripts.analysis.thesis_ppo_sanity as sanity

    calls = []

    def fake_run(fixture, seeds, timesteps, probe):
        calls.append((fixture, probe))
        return {"passed": probe == "ent_coef_zero"}

    monkeypatch.setattr(sanity, "run", fake_run)
    report = sanity.run_protocol(seeds=(1,), timesteps=10)
    assert report["selected_probe"] == "ent_coef_zero"
    assert [p for _, p in calls] == ["baseline"] * 4 + ["ent_coef_zero"] * 4


def test_rule_baselines_have_fixed_causal_lengths():
    from scripts.analysis.thesis_baselines import (
        mean_reversion_policy, momentum_policy, opening_range_policy,
        regime_rules_policy,
    )

    close = np.linspace(4000.0, 4100.0, 60)
    for policy in (momentum_policy, mean_reversion_policy, opening_range_policy):
        weights = policy(close)
        assert weights.shape == (59,)
        assert np.isfinite(weights).all()
    assert np.all(regime_rules_policy(3) == -1.0)
    assert np.all(regime_rules_policy(2) == 1.0)
    assert np.all(regime_rules_policy(0) == 0.0)
