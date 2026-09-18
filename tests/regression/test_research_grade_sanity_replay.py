"""Unit controls for the independent policy evaluator; no learning/market data."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from scripts.diagnostics.verify_sanity_policy_replay import compare_replay, replay_sessions
from src.research.session_env import EXPOSURE_LEVELS
from src.research.synthetic_sessions import make_sessions, oracle_result


class IdentityNormalizer:
    training = False
    norm_reward = False
    norm_obs = False

    def normalize_obs(self, observation):
        return observation.copy()


class AnalyticPolicy:
    def __init__(self, *, flat):
        self.flat = flat

    def predict(self, observation, *, deterministic):
        assert deterministic is True
        weight = 0.0 if self.flat else float(np.sign(observation[0]))
        return np.asarray(EXPOSURE_LEVELS.index(weight)), None


@pytest.mark.parametrize("fixture,flat", [("S1", True), ("S2", False), ("S3", False), ("S4", True)])
def test_own_loop_reproduces_known_policies(fixture, flat):
    sessions = make_sessions(fixture, n=3, seed=42)
    actual = replay_sessions(AnalyticPolicy(flat=flat), IdentityNormalizer(), sessions, fixture)
    expected = [oracle_result(fixture, spec).daily_return for spec in sessions]
    np.testing.assert_allclose(actual["daily_returns"], expected, atol=1e-12, rtol=0)
    assert actual["n_eval_sessions"] == 3
    if fixture == "S2":
        assert actual["mean_cost"] == 0.0
    assert compare_replay(actual, actual)["passed"] is True


def test_replay_rejects_unfrozen_normalizer():
    norm = IdentityNormalizer()
    norm.training = True
    with pytest.raises(ValueError, match="frozen"):
        replay_sessions(AnalyticPolicy(flat=True), norm, make_sessions("S1", n=1), "S1")


def test_replay_rejects_silent_observation_transform():
    class ChangedNormalizer(IdentityNormalizer):
        def normalize_obs(self, observation):
            return observation + 0.1
    with pytest.raises(ValueError, match="changed the observation"):
        replay_sessions(AnalyticPolicy(flat=True), ChangedNormalizer(), make_sessions("S1", n=1), "S1")


def test_replay_rejects_s2_fee_override():
    sessions = [replace(make_sessions("S2", n=1)[0], cost_parameters=None)]
    with pytest.raises(ValueError, match="fees differ"):
        replay_sessions(AnalyticPolicy(flat=False), IdentityNormalizer(), sessions, "S2")


def test_equal_mean_does_not_hide_reordered_daily_evidence():
    sessions = make_sessions("S2", n=3, seed=42)
    actual = replay_sessions(AnalyticPolicy(flat=False), IdentityNormalizer(), sessions, "S2")
    recorded = deepcopy(actual)
    recorded["daily_returns"] = list(reversed(recorded["daily_returns"]))
    assert recorded["mean_net"] == actual["mean_net"]
    assert compare_replay(actual, recorded)["passed"] is False


def test_replay_cannot_train_or_reuse_the_producing_evaluator():
    import ast
    from pathlib import Path
    import scripts.diagnostics.verify_sanity_policy_replay as module
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert not any(isinstance(node, ast.Attribute) and node.attr == "learn" for node in ast.walk(tree))
    assert "thesis_ppo_sanity" not in source


def test_replay_requires_finite_daily_evidence():
    row = {"daily_returns": [float("nan")]}
    with pytest.raises(ValueError, match="not comparable"):
        compare_replay(row, row)
