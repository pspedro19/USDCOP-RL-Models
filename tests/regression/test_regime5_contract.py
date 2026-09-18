"""Engineering fixtures, not market performance evidence."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.research.features import FEATURE_ORDER, SCHEMA
from src.research.observation_contract import (
    LEGACY_VERSION,
    REGIME5_VERSION,
    observation_contract,
    require_model_contract,
)
from src.research.session_env import run_session
from src.research.session_gym import SessionSpec, SessionTradingEnv


def spec(version=REGIME5_VERSION):
    slots = observation_contract(version).regime_slots
    return SessionSpec(
        "engineering-fixture",
        np.linspace(4000, 4050, 60),
        np.zeros((60, 25)),
        np.r_[np.zeros(3), np.ones(slots) / slots],
        3.0,
        observation_version=version,
    )


def test_new_contract_appends_without_mutating_legacy():
    old = SCHEMA.sha256
    new = observation_contract(REGIME5_VERSION)
    assert new.order[:-1] == tuple(FEATURE_ORDER)
    assert new.order[-1] == "p_regime_4" and len(new.order) == 38
    assert old == SCHEMA.sha256
    assert new.sha256 != observation_contract().sha256


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_k_padding_preserves_every_probability(k):
    raw = np.arange(1, k + 1, dtype=float)
    raw /= raw.sum()
    padded = observation_contract(REGIME5_VERSION).posterior(raw, k)
    np.testing.assert_array_equal(padded[:k], raw)
    np.testing.assert_array_equal(padded[k:], 0)


@pytest.mark.parametrize("k", [True, False, 2.0, "3", 0, 1, 6])
def test_invalid_k_is_never_coerced(k):
    with pytest.raises(ValueError):
        observation_contract(REGIME5_VERSION).posterior([0.5, 0.5], k)


@pytest.mark.parametrize(
    "value",
    [
        [True, False],
        [np.nan, 0],
        [0.2, 0.2],
        [-0.1, 1.1],
        ["0.5", "0.5"],
        np.array([0.5 + 0j, 0.5]),
        [[0.5, 0.5]],
        [0.3, 0.3, 0.4],
        np.ma.array([0.5, 0.5], mask=[True, False]),
    ],
)
def test_invalid_posterior_is_not_repaired(value):
    with pytest.raises(ValueError):
        observation_contract(REGIME5_VERSION).posterior(value, 2)


def test_env_checks_every_session_not_only_first():
    with pytest.raises(ValueError, match="mixed observation"):
        SessionTradingEnv([spec(), spec(LEGACY_VERSION)])
    with pytest.raises(ValueError):
        SessionTradingEnv(
            [spec(LEGACY_VERSION), replace(spec(), observation_version=LEGACY_VERSION)]
        )


def test_version_cannot_be_guessed_from_dimensions():
    with pytest.raises(ValueError):
        SessionTradingEnv([replace(spec(), observation_version=LEGACY_VERSION)])
    with pytest.raises(ValueError):
        observation_contract("research38")


@pytest.mark.parametrize("version", [LEGACY_VERSION, REGIME5_VERSION])
def test_env_reward_and_terminal_accounting_unchanged(version):
    session = spec(version)
    env = SessionTradingEnv([session], shuffle=False)
    obs, _ = env.reset()
    assert obs.shape == (len(observation_contract(version).order),)
    actions = np.random.default_rng(42).integers(0, 5, 59)
    reward = 0
    for i, a in enumerate(actions):
        _, r, done, _, _ = env.step(int(a))
        reward += r / 100
        assert done == (i == 58)
    expected = run_session(session.close, np.array([-1, -0.5, 0, 0.5, 1])[actions], 3)
    assert reward == pytest.approx(expected.daily_return, abs=1e-12)
    assert env.last_result.terminal_cost == expected.terminal_cost


def test_model_requires_identity_not_only_shape():
    model = SimpleNamespace(observation_space=SimpleNamespace(shape=(38,)))
    with pytest.raises(ValueError):
        require_model_contract(model, REGIME5_VERSION)
    model.research_observation_sha256 = observation_contract(REGIME5_VERSION).sha256
    require_model_contract(model, REGIME5_VERSION)
    model.observation_space.shape = (37,)
    with pytest.raises(ValueError):
        require_model_contract(model, REGIME5_VERSION)
