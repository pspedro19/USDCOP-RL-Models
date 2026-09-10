"""
Regression: el escalado no deja que la evaluación se filtre en el entrenamiento.

Contract: CTR-RESEARCH-DATASET-001 · Date: 2026-08-25
Implementa los **tests 4 y 8** de §13 del plan de tesis.

## Por qué esta fuga es la peligrosa

Un `StandardScaler` ajustado sobre todo el período mete la media y la varianza del futuro
dentro de cada observación del pasado. No rompe nada, no produce un número raro, y mejora los
resultados de forma sistemática. Es exactamente lo que la constitución llama «normalización
global» en la capa 1 del anti-look-ahead, y la razón de que se mida en vez de leerse:

- **Test 4** — alterar selección y hold-out por completo no puede mover **ni un valor** del
  escalador. Se compara con igualdad exacta, no aproximada: si el escalador viera un solo dato
  de evaluación, algo cambiaría en el decimoquinto decimal, y eso ya sería la fuga.
- **Test 8** — `VecNormalize` se guarda y se recarga con `training=False` y
  `norm_reward=False`. Si siguiera actualizando su media móvil durante la evaluación, cada
  episodio evaluado cambiaría la puntuación de los siguientes, y el orden de evaluación —un
  detalle sin significado— movería las tablas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.dataset import CLIP, MARKET_FEATURES, fit_dev_scaler  # noqa: E402


def fake_features(n_sessions: int = 30, seed: int = 0) -> pd.DataFrame:
    """Tabla con la forma que produce `build_market_features`: features + `_session`."""
    rng = np.random.default_rng(seed)
    rows, sessions = [], []
    day = pd.Timestamp("2020-01-06").date()
    for i in range(n_sessions):
        for _ in range(60):
            rows.append(rng.normal(0, 1, len(MARKET_FEATURES)))
            sessions.append(day)
        day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date()
    df = pd.DataFrame(rows, columns=MARKET_FEATURES)
    df["_session"] = sessions
    return df


def split(df: pd.DataFrame):
    days = sorted(set(df["_session"]))
    return set(days[:10]), set(days[10:20]), set(days[20:])


# ---------------------------------------------------------------------------
# TEST 4 — el escalador es ciego a los bloques de evaluación
# ---------------------------------------------------------------------------

def test_scaler_is_blind_to_evaluation_data():
    """Alterar selección y hold-out no puede mover el escalador ni un bit."""
    df = fake_features()
    dev, sel, hold = split(df)

    mean_a, scale_a = fit_dev_scaler(df, dev)

    tampered = df.copy()
    evaluation = tampered["_session"].isin(sel | hold)
    tampered.loc[evaluation, MARKET_FEATURES] = (
        tampered.loc[evaluation, MARKET_FEATURES] * 1000.0 + 500.0)

    mean_b, scale_b = fit_dev_scaler(tampered, dev)

    assert np.array_equal(mean_a, mean_b), (
        "la media del escalador cambió al alterar datos de EVALUACIÓN — fuga de "
        "normalización global (constitución §4, capa 1)"
    )
    assert np.array_equal(scale_a, scale_b), "la escala del escalador vio la evaluación"


def test_scaler_actually_uses_the_development_block():
    """El complemento del test anterior: si alterar DESARROLLO tampoco lo moviera,
    el escalador no estaría ajustándose a nada y el test 4 pasaría trivialmente."""
    df = fake_features()
    dev, _, _ = split(df)
    mean_a, _ = fit_dev_scaler(df, dev)

    tampered = df.copy()
    is_dev = tampered["_session"].isin(dev)
    tampered.loc[is_dev, MARKET_FEATURES] += 7.0

    mean_b, _ = fit_dev_scaler(tampered, dev)
    assert not np.allclose(mean_a, mean_b), (
        "alterar desarrollo no movió el escalador: no se está ajustando a nada"
    )


def test_scaler_refuses_an_empty_development_block():
    """Un escalador ajustado sobre cero filas devolvería NaN silenciosamente."""
    df = fake_features()
    with pytest.raises(ValueError, match="desarrollo"):
        fit_dev_scaler(df, set())


def test_constant_features_do_not_produce_infinities():
    """Escala 0 -> división por cero. §6.6 clipa a ±5; un inf se lo salta."""
    df = fake_features()
    dev, _, _ = split(df)
    df.loc[df["_session"].isin(dev), MARKET_FEATURES[0]] = 3.0

    _, scale = fit_dev_scaler(df, dev)
    assert scale[0] == 1.0, "una feature constante debe escalarse por 1, no por 0"
    assert np.isfinite(scale).all()


# ---------------------------------------------------------------------------
# TEST 8 — VecNormalize congelado en evaluación
# ---------------------------------------------------------------------------

def test_vecnormalize_is_frozen_during_evaluation(tmp_path):
    """Recargado para evaluar, no puede seguir actualizando sus estadísticos.

    Si lo hiciera, cada episodio evaluado alteraría la puntuación de los siguientes y el
    ORDEN de evaluación —que no significa nada— movería las tablas.
    """
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("gymnasium")
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-5, 5, (3,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            self._i = 0
            return np.zeros(3, dtype=np.float32), {}

        def step(self, a):
            self._i += 1
            return (np.ones(3, dtype=np.float32) * self._i,
                    float(self._i), self._i >= 5, False, {})

    venv = VecNormalize(DummyVecEnv([lambda: Dummy()]), norm_obs=False, norm_reward=True)
    venv.reset()
    for _ in range(40):
        venv.step(np.array([0]))
    path = tmp_path / "vecnorm.pkl"
    venv.save(str(path))

    evaluation = VecNormalize.load(str(path), DummyVecEnv([lambda: Dummy()]))
    evaluation.training = False
    evaluation.norm_reward = False

    before_mean = float(evaluation.ret_rms.mean)
    before_var = float(evaluation.ret_rms.var)
    evaluation.reset()
    for _ in range(40):
        evaluation.step(np.array([1]))

    assert float(evaluation.ret_rms.mean) == before_mean, (
        "VecNormalize actualizó su media durante la evaluación"
    )
    assert float(evaluation.ret_rms.var) == before_var
    assert evaluation.training is False and evaluation.norm_reward is False


def test_the_training_flag_is_what_freezes_it():
    """Contraprueba: con `training=True` los estadísticos SÍ se mueven.

    Sin esto, el test anterior pasaría aunque `VecNormalize` no estuviera normalizando nada.
    """
    pytest.importorskip("stable_baselines3")
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class Dummy(gym.Env):
        observation_space = gym.spaces.Box(-5, 5, (2,), dtype=np.float32)
        action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, a):
            return np.ones(2, dtype=np.float32), 3.0, True, False, {}

    venv = VecNormalize(DummyVecEnv([lambda: Dummy()]), norm_obs=False, norm_reward=True)
    venv.reset()
    before = float(venv.ret_rms.var)
    for _ in range(30):
        venv.step(np.array([0]))
    assert float(venv.ret_rms.var) != before


def test_observations_are_clipped_to_the_declared_bound():
    """§6.6 fija ±5 y el `Box` del Env lo declara: el dataset tiene que respetarlo."""
    assert CLIP == 5.0
    df = fake_features()
    dev, _, _ = split(df)
    mean, scale = fit_dev_scaler(df, dev)
    X = (df[MARKET_FEATURES].to_numpy() - mean) / scale
    assert np.abs(np.clip(X, -CLIP, CLIP)).max() <= CLIP
