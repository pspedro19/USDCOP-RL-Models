"""Analytic controls for PPO training; no market trials are charged."""

from __future__ import annotations

import json

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


def test_sanity_protocol_runs_only_the_frozen_recipe_without_search(monkeypatch, tmp_path):
    import scripts.analysis.thesis_ppo_sanity as sanity

    calls = []

    def fake_run(fixture, seeds, timesteps, probe, checkpoint_dir):
        calls.append((fixture, probe))
        return {"fixture": fixture.value, "passed": fixture != Fixture.NOISE_WITH_COST,
                "passed_seeds": []}

    monkeypatch.setattr(sanity, "run", fake_run)
    monkeypatch.setattr(sanity, "aggregate_fixture_reports", lambda paths:
                        {"selected_probe": "flat_init_no_turn", "passed": False,
                         "n_fixture_reports": len(paths)})
    report = sanity.run_protocol(timesteps=100_000, checkpoint_dir=tmp_path)
    assert report["selected_probe"] == "flat_init_no_turn"
    assert report["passed"] is False  # a failed control is recorded, not tuned away
    assert report["n_fixture_reports"] == 4
    assert [p for _, p in calls] == ["flat_init_no_turn"] * 4
    assert sanity.PROBES == ("flat_init_no_turn",)


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


def test_sanity_aggregator_requires_all_protocol_seeds(tmp_path):
    from scripts.analysis.aggregate_sanity_fixture import aggregate

    base = {"fixture": "S2", "probe": "flat_init_no_turn", "synthetic_only": True,
            "market_trials_charged": 0, "rows": [{"mean_net": 0.01, "mean_abs_exposure": 1.0}]}
    for seed in (42, 123, 456, 789, 1337):
        payload = {**base, "rows": [{**base["rows"][0], "seed": seed}]}
        (tmp_path / f"S2_flat_init_no_turn_seed{seed}.json").write_text(
            json.dumps(payload), encoding="utf-8")
    report = aggregate(tmp_path, "S2", "flat_init_no_turn")
    assert report["passed"] is True
    assert report["passed_seeds"] == [42, 123, 456, 789, 1337]


def test_protocol_aggregator_requires_one_probe_for_all_fixtures(tmp_path):
    from scripts.analysis.aggregate_sanity_protocol import aggregate_protocol

    for fixture in ("S1", "S2", "S3", "S4"):
        payload = {
            "fixture": fixture,
            "probe": "flat_init_no_turn",
            "synthetic_only": True,
            "market_trials_charged": 0,
            "seeds": [42, 123, 456, 789, 1337],
            "rows": [{"seed": seed} for seed in (42, 123, 456, 789, 1337)],
            "passed": True,
            "passed_seeds": [42, 123, 456, 789, 1337],
        }
        (tmp_path / f"sanity_{fixture}_flat_init_no_turn.json").write_text(
            json.dumps(payload), encoding="utf-8")
    report = aggregate_protocol(tmp_path / "sanity", "flat_init_no_turn")
    assert report["passed"] is True
    assert report["market_evidence"] is False
