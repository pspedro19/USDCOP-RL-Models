"""Controls for evidence identity, zero fees and shared training (no PPO training)."""
from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.research.cost_model import CostParameters, bar_cost, realized_vol_pips
from src.research.ppo_recipe import (
    FROZEN_SANITY_PROBE, build_ppo, canonical_sha256, effective_recipe,
    sessions_sha256, write_immutable_json,
)
from src.research.session_env import EXPOSURE_LEVELS, run_session, simple_returns
from src.research.session_gym import SessionTradingEnv
from src.research.synthetic_sessions import (
    Fixture, make_sessions, oracle_result, oracle_weights, sanity_protocol_manifest,
    sanity_session_splits, validate_sanity_manifest,
)

ROOT = Path(__file__).resolve().parents[2]


def test_s2_is_zero_total_cost_not_merely_zero_spread():
    for spec in make_sessions(Fixture.SIGNAL_PLANTED, n=5, seed=19):
        result = oracle_result(Fixture.SIGNAL_PLANTED, spec)
        assert spec.spread_pips == 0.0
        assert spec.cost_parameters == CostParameters(0.0, 0.0)
        assert result.total_cost == 0.0
        assert result.terminal_cost == 0.0
        assert result.daily_return == result.gross_return
        assert all(b.spread_component_pips == b.slippage_component_pips == 0.0
                   for b in result.breakdown)


def test_s3_has_same_signal_and_real_market_costs():
    s2 = make_sessions("S2", n=3, seed=19)
    s3 = make_sessions("S3", n=3, seed=19)
    for left, right in zip(s2, s3, strict=True):
        np.testing.assert_array_equal(left.close, right.close)
        np.testing.assert_array_equal(left.market, right.market)
        assert right.cost_parameters is None
        free = oracle_result("S2", left)
        paid = oracle_result("S3", right)
        assert free.gross_return == paid.gross_return
        assert paid.total_cost > 0.0
        assert paid.daily_return < free.daily_return


@pytest.mark.parametrize("fixture", ["S1", "S2", "S3", "S4"])
def test_zero_override_and_market_costs_keep_gym_accounting_parity(fixture):
    spec = make_sessions(fixture, n=1, seed=13)[0]
    env = SessionTradingEnv([spec], shuffle=False)
    env.reset()
    actions = np.random.default_rng(17).integers(0, len(EXPOSURE_LEVELS), 59)
    rewards = []
    for action in actions:
        _, reward, _, _, _ = env.step(int(action))
        rewards.append(reward)
    weights = np.array([EXPOSURE_LEVELS[a] for a in actions])
    result = run_session(spec.close, weights, spec.spread_pips,
                         cost_parameters=spec.cost_parameters)
    assert sum(rewards) / env.reward_scale == pytest.approx(result.daily_return, abs=1e-12)
    assert env.last_result.daily_return == result.daily_return


def test_default_market_arithmetic_is_unchanged():
    spec = make_sessions("S3", n=1, seed=3)[0]
    weights = oracle_weights("S3", spec)
    sigma = realized_vol_pips(spec.close)
    changes = np.diff(np.r_[0.0, weights, 0.0])
    expected_cost = np.sum((np.abs(changes) * (spec.spread_pips / 2 + 0.5)
                           + 0.1 * np.abs(changes) * sigma) / spec.close)
    result = run_session(spec.close, weights, spec.spread_pips)
    expected_gross = np.sum(weights * simple_returns(spec.close))
    assert result.daily_return == pytest.approx(expected_gross - expected_cost, abs=1e-12)
    default = bar_cost(1.5, 3.0, 2.1, 4000.0)
    explicit = bar_cost(1.5, 3.0, 2.1, 4000.0, cost_parameters=CostParameters())
    assert default == explicit


def test_unseen_sessions_use_a_distinct_generation_seed_and_prices():
    train, unseen = sanity_session_splits("S2", 42, n_train=4, n_eval=4)
    assert {s.date for s in train}.isdisjoint({s.date for s in unseen})
    assert not np.array_equal(train[0].close, unseen[0].close)
    assert sessions_sha256(train) != sessions_sha256(unseen)
    repeat_train, repeat_unseen = sanity_session_splits("S2", 42, n_train=4, n_eval=4)
    assert sessions_sha256(train) == sessions_sha256(repeat_train)
    assert sessions_sha256(unseen) == sessions_sha256(repeat_unseen)


def test_effective_recipe_is_shared_in_both_callers():
    for name in ("thesis_ppo_sanity.py", "thesis_train_ppo.py"):
        tree = ast.parse((ROOT / "scripts/analysis" / name).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert any(isinstance(n.func, ast.Name) and n.func.id == "build_ppo" for n in calls)
        assert not any(isinstance(n.func, ast.Name) and n.func.id == "PPO" for n in calls)
    config = effective_recipe(FROZEN_SANITY_PROBE)
    assert config["policy_kwargs"]["net_arch"] == {"pi": [256, 256], "vf": [256, 256]}
    assert config["environment"]["reward_scale"] == 100.0


def test_shared_factory_applies_frozen_architecture_and_flat_initialization():
    pytest.importorskip("stable_baselines3")
    import torch

    sessions = make_sessions("S2", n=1, seed=0)
    model, norm, _ = build_ppo(sessions, seed=42, probe=FROZEN_SANITY_PROBE)
    try:
        assert model.n_steps == 4096
        assert model.policy.net_arch == {"pi": [256, 256], "vf": [256, 256]}
        assert norm.norm_obs is False
        assert norm.norm_reward is True
        assert norm.venv.envs[0].reward_scale == 100.0
        np.testing.assert_array_equal(model.policy.action_net.bias.detach().numpy(), [0, 0, 3, 0, 0])
        with torch.no_grad():
            observation = torch.zeros((1, model.observation_space.shape[0]))
            probs = model.policy.get_distribution(observation).distribution.probs[0]
        assert float(probs[2]) == pytest.approx(np.exp(3) / (np.exp(3) + 4), abs=1e-6)
        assert model.num_timesteps == 0  # fixture construction is not an experiment
    finally:
        norm.close()


def test_old_boolean_gate_cannot_pass_a_research_grade_manifest():
    with pytest.raises(ValueError, match="lacks"):
        validate_sanity_manifest({"passed": True, "probe": FROZEN_SANITY_PROBE})


def test_stale_fixture_or_code_hash_invalidates_green_evidence():
    manifest = sanity_protocol_manifest()
    report = {"schema_version": "research-grade-sanity-v1", "manifest": manifest,
              "fingerprint": canonical_sha256(manifest), "passed": True}
    assert validate_sanity_manifest(report) == manifest
    tampered = deepcopy(report)
    tampered["manifest"]["fixtures"]["S2"]["fees"]["commission_per_side"] = 0.5
    # Rehashing a stale manifest must not make it valid for current code.
    tampered["fingerprint"] = canonical_sha256(tampered["manifest"])
    with pytest.raises(ValueError, match="stale"):
        validate_sanity_manifest(tampered)
    tampered = deepcopy(report)
    tampered["manifest"]["source_sha256"]["src/research/cost_model.py"] = "0" * 64
    tampered["fingerprint"] = canonical_sha256(tampered["manifest"])
    with pytest.raises(ValueError, match="stale"):
        validate_sanity_manifest(tampered)


def test_search_and_s2_spread_override_are_rejected():
    with pytest.raises(ValueError, match="no search"):
        sanity_protocol_manifest("ent_coef_zero")
    with pytest.raises(ValueError, match="zero-cost"):
        make_sessions("S2", spread_pips=3.0)


def test_evidence_writer_is_exclusive(tmp_path):
    target = tmp_path / "result.json"
    write_immutable_json(target, {"passed": False})
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_immutable_json(target, {"passed": True})
    assert target.read_bytes() == before


def test_single_seed_never_counts_as_five_seed_protocol(monkeypatch):
    from scripts.analysis import thesis_ppo_sanity as module
    monkeypatch.setattr(module, "_train_one", lambda fixture, seed, *args, **kwargs:
                        {"seed": seed, "passed": True})
    report = module.run("S2", seeds=(42,))
    assert report["passed"] is False
    assert report["complete"] is False
    assert report["passed_seeds"] == [42]


def test_s2_requires_seventy_percent_of_same_path_oracle():
    from scripts.analysis.thesis_ppo_sanity import _passes
    stats = {"mean_net": 0.069, "mean_cost": 0.0, "oracle_mean_net": 0.1}
    assert _passes(Fixture.SIGNAL_PLANTED, stats) is False
    assert _passes(Fixture.SIGNAL_PLANTED, dict(stats, mean_net=0.071)) is True
    assert _passes(Fixture.SIGNAL_PLANTED, dict(stats, mean_net=0.08, mean_cost=0.001)) is False


def test_protocol_aggregate_follows_fixture_artifacts_not_just_pass_flags(tmp_path):
    from src.research.ppo_recipe import file_sha256
    from scripts.analysis.thesis_ppo_sanity import aggregate_fixture_reports
    from src.research.synthetic_sessions import SANITY_SEEDS, UNSEEN_SEED_OFFSET

    manifest = sanity_protocol_manifest()
    fingerprint = canonical_sha256(manifest)
    fixtures = []
    for fixture in Fixture:
        rows, evidence = [], []
        for seed in SANITY_SEEDS:
            tag = f"{fixture.value}-{seed}"
            run_manifest = {"schema_version": "research-grade-sanity-v1", "manifest": manifest,
                            "fingerprint": fingerprint, "fixture": fixture.value, "seed": seed,
                            "training_dataset_sha256": "1" * 64, "unseen_dataset_sha256": "2" * 64,
                            "train_generation_seed": seed,
                            "unseen_generation_seed": seed + UNSEEN_SEED_OFFSET}
            frozen = tmp_path / f"{tag}-manifest.json"
            write_immutable_json(frozen, run_manifest)
            checkpoint = tmp_path / f"{tag}.zip"
            normalizer = tmp_path / f"{tag}.pkl"
            # Deliberately inert fixture files: this unit test never fits a PPO.
            checkpoint.write_bytes(b"unit-test-checkpoint")
            normalizer.write_bytes(b"unit-test-normalizer")
            stats = {"mean_net": 0.1, "mean_abs_exposure": 0.0, "mean_cost": 0.0,
                     "oracle_mean_net": 0.1, "n_eval_sessions": 100}
            row = {k: v for k, v in run_manifest.items() if k not in ("manifest", "schema_version")}
            row.update(train=stats, unseen=stats, identity_unchanged=True, passed=True,
                       timesteps_requested=100_000, timesteps_effective=102_400,
                       artifacts={name: {"path": str(path), "sha256": file_sha256(path)}
                                  for name, path in (("checkpoint", checkpoint),
                                                     ("vecnormalize", normalizer), ("manifest", frozen))})
            result = tmp_path / f"{tag}-result.json"
            write_immutable_json(result, row)
            evidence.append({"path": str(result), "sha256": file_sha256(result)})
            rows.append(row)
        fixture_path = tmp_path / f"{fixture.value}.json"
        write_immutable_json(fixture_path, {"schema_version": "research-grade-sanity-v1",
                              "manifest": manifest, "fingerprint": fingerprint,
                              "fixture": fixture.value, "synthetic_only": True,
                              "market_evidence": False, "rows": rows, "input_evidence": evidence,
                              "passed": True})
        fixtures.append(fixture_path)
    report = aggregate_fixture_reports(fixtures)
    assert report["passed"] is True
    assert {Path(e["path"]).name for e in report["input_evidence"]} == {"S1.json", "S2.json", "S3.json", "S4.json"}
    (tmp_path / "S2-42.zip").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="artifact was changed"):
        aggregate_fixture_reports(fixtures)
