"""
Regression: el `gym.Env` y la contabilidad puntúan exactamente el mismo juego.

Contract: CTR-RESEARCH-SESSIONGYM-001 · Date: 2026-08-25

## El fallo que este test existe para impedir

Un `Env` que reimplementa los costos "casi igual" que el motor de backtest produce un agente
que optimiza un juego distinto del que se le puntúa. No hay traza: el entrenamiento converge,
las tablas salen, y ambos números son plausibles por separado. La única defensa es comprobar
la igualdad, senda por senda, con tolerancia numérica.

Se prueba con **sendas aleatorias**, no con una política fija: un flip `+1 -> -1` y un cierre
parcial ejercitan ramas del costo que mantener una posición nunca toca.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("gymnasium")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.features import FEATURE_ORDER, GROUPS  # noqa: E402
from src.research.session_env import (BARS_PER_SESSION, EXPOSURE_LEVELS,  # noqa: E402
                                      OPERABLE_RETURNS)
from src.research.session_gym import (OBS_CLIP, SessionSpec, SessionTradingEnv,  # noqa: E402
                                      replay_weights)

N_MARKET = sum(len(GROUPS[g]) for g in ("precio", "volatilidad", "tendencia", "temporal"))
N_CONTEXT = len(GROUPS["macro"]) + len(GROUPS["regimen"])


def make_spec(seed: int = 0, spread: float = 3.0) -> SessionSpec:
    rng = np.random.default_rng(seed)
    close = 4000.0 + np.cumsum(rng.normal(0, 2.0, BARS_PER_SESSION))
    return SessionSpec(
        date=f"2024-01-{(seed % 28) + 1:02d}",
        close=close,
        market=rng.normal(0, 1, (BARS_PER_SESSION, N_MARKET)).astype(np.float32),
        context=rng.normal(0, 1, N_CONTEXT).astype(np.float32),
        spread_pips=spread,
    )


def run_path(env: SessionTradingEnv, actions):
    env.reset()
    for a in actions:
        _, _, term, _, info = env.step(int(a))
    assert term, "el episodio debe terminar tras 59 decisiones"
    return info


# ---------------------------------------------------------------------------
# Paridad
# ---------------------------------------------------------------------------

def test_gym_matches_run_session_on_random_paths():
    """Para 40 sendas aleatorias, `daily_return` del Env == el de `run_session`."""
    rng = np.random.default_rng(11)
    for k in range(40):
        spec = make_spec(seed=k, spread=float(rng.choice([2.0, 3.0, 6.0])))
        env = SessionTradingEnv([spec], seed=0, shuffle=False)
        actions = rng.integers(0, len(EXPOSURE_LEVELS), OPERABLE_RETURNS)
        info = run_path(env, actions)

        weights = [EXPOSURE_LEVELS[int(a)] for a in actions]
        offline = replay_weights(spec, weights)

        assert info["daily_return"] == pytest.approx(offline.daily_return, rel=1e-12), (
            f"senda {k}: el Env y run_session difieren — el agente entrenaría contra "
            f"un juego distinto del que se le puntúa"
        )
        assert info["n_changes"] == offline.n_changes


def test_episode_reward_equals_daily_return_including_terminal_cost():
    """El reward de entrenamiento y la liquidación reportada tienen la misma contabilidad."""
    spec = make_spec(seed=3)
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    rng = np.random.default_rng(5)
    actions = rng.integers(0, len(EXPOSURE_LEVELS), OPERABLE_RETURNS)

    env.reset()
    total = 0.0
    for a in actions:
        _, r, term, _, info = env.step(int(a))
        total += r / env.reward_scale

    res = env.last_result
    assert total == pytest.approx(res.daily_return, abs=1e-12)
    assert res.terminal_cost >= 0.0


def test_always_flat_earns_exactly_zero():
    """No operar no cuesta. Es el listón que la Fase E dejó fijado."""
    spec = make_spec(seed=9, spread=6.0)
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    flat = EXPOSURE_LEVELS.index(0.0)
    info = run_path(env, [flat] * OPERABLE_RETURNS)
    assert info["daily_return"] == pytest.approx(0.0, abs=1e-15)
    assert info["terminal_cost"] == pytest.approx(0.0)


def test_reward_shaping_defaults_to_identity():
    """The diagnostic shaping knobs must not change the frozen objective by default."""
    env = SessionTradingEnv([make_spec(seed=7)], seed=7, shuffle=False)
    _obs, _ = env.reset()
    total = 0.0
    terminated = False
    while not terminated:
        _obs, reward, terminated, _truncated, _info = env.step(2)  # flat action
        total += reward
    assert total == pytest.approx(env.last_result.daily_return * env.reward_scale)


# ---------------------------------------------------------------------------
# Cronología y espacios
# ---------------------------------------------------------------------------

def test_there_are_exactly_59_steps_per_episode():
    spec = make_spec()
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    env.reset()
    for b in range(OPERABLE_RETURNS - 1):
        _, _, term, _, _ = env.step(2)
        assert not term, f"terminó en el paso {b}, antes de las {OPERABLE_RETURNS} decisiones"
    _, _, term, _, _ = env.step(2)
    assert term


def test_action_space_maps_to_the_frozen_exposure_levels():
    spec = make_spec()
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    assert env.action_space.n == len(EXPOSURE_LEVELS) == 5
    assert EXPOSURE_LEVELS == (-1.0, -0.5, 0.0, 0.5, 1.0)


def test_observation_dimension_matches_the_feature_schema():
    """El Env y `feature_schema.json` tienen que contar las mismas features."""
    spec = make_spec()
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    assert env.observation_space.shape == (len(FEATURE_ORDER),)
    obs, _ = env.reset()
    assert obs.shape == (len(FEATURE_ORDER),)
    assert obs.dtype == np.float32


def test_observations_stay_within_the_declared_clip():
    """§6.6 fija clipping a ±5; el Box lo declara y el Env debe respetarlo."""
    spec = make_spec(seed=4)
    env = SessionTradingEnv([spec], seed=0, shuffle=False)
    obs, _ = env.reset()
    rng = np.random.default_rng(2)
    for _ in range(OPERABLE_RETURNS - 1):
        obs, _, _, _, _ = env.step(int(rng.integers(0, 5)))
        assert np.all(np.abs(obs) <= OBS_CLIP + 1e-6)


def test_position_state_resets_between_episodes():
    """§9.1 fija `w_{-1}=0`: arrastrar la posición de ayer sería otro entorno."""
    specs = [make_spec(seed=1), make_spec(seed=2)]
    env = SessionTradingEnv(specs, seed=0, shuffle=False)
    long_ = EXPOSURE_LEVELS.index(1.0)
    env.reset()
    for _ in range(OPERABLE_RETURNS):
        env.step(long_)

    obs, _ = env.reset()
    n_mkt = N_MARKET
    pos = obs[n_mkt:n_mkt + len(GROUPS["posicion"])]
    assert np.allclose(pos, 0.0), (
        f"el estado de posición no se reinició: {pos}"
    )


def test_a_spec_with_the_wrong_number_of_bars_is_rejected():
    with pytest.raises(ValueError, match="barras"):
        SessionSpec(date="x", close=np.zeros(59), market=np.zeros((59, N_MARKET)),
                    context=np.zeros(N_CONTEXT), spread_pips=2.0)
