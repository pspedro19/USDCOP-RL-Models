"""`gym.Env` de una sesión, montado SOBRE la contabilidad de `session_env`.

Contract: CTR-RESEARCH-SESSIONGYM-001 · Date: 2026-08-25

## Por qué un wrapper y no otro entorno

`session_env.py` son funciones puras: dada una senda de exposición, devuelve el resultado del
día. PPO necesita `reset()`/`step()`. La tentación evidente es reimplementar la aritmética
dentro del `Env` — y es exactamente el error que hace irreproducible una tesis: el agente
optimizaría un juego y las tablas puntuarían otro, sin que ninguna prueba lo delatara, porque
ambos números serían plausibles por separado.

Aquí el `Env` **acumula** la senda de exposición y, al terminar el episodio, llama a
`run_session` — la MISMA función que usan los baselines y que verifican los tests 9, 10 y 13.
`SessionTradingEnv.last_result` es un `SessionResult` idéntico al que produciría evaluar esa
senda offline. `test_gym_matches_run_session` lo comprueba sobre sendas aleatorias.

## Reward ≠ contabilidad (§9.6, principio 5)

El reward que entrena es el retorno neto de la barra, escalado. Lo que se REPORTA es
`daily_return` de `run_session`. Son cosas distintas a propósito: el reward puede llevar
`VecNormalize`, escala o penalizaciones sin que eso toque una sola cifra de las tablas.

## Cronología (§9.1, test 10)

En el paso `b` el agente observa información hasta el **cierre de b**, elige `w_b`, y cobra
`r_{b+1}`. Hay 60 barras y **59 decisiones operables**: la última barra no tiene un retorno
siguiente que capturar dentro de la sesión. Al cerrar el episodio se fuerza `w = 0` y su costo
se cobra — no es una acción evitable.

## Estado de posición

Las cinco features endógenas de §6.5 (`w_prev`, PnL no realizado, barras en posición,
drawdown de sesión, número de cambios) se calculan aquí, dentro del bucle, porque dependen de
la trayectoria del agente y no del mercado. Se reinician en cada `reset()`: §9.1 fija
`w_{-1} = 0`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:                                              # gymnasium es opcional para importar
    import gymnasium as gym                       # el resto del modulo (tests de paridad)
    from gymnasium import spaces
    _GYM = True
except ImportError:                               # pragma: no cover
    gym, spaces, _GYM = object, None, False

from src.research.cost_model import realized_vol_pips
from src.research.features import FEATURE_ORDER, GROUPS
from src.research.session_env import (BARS_PER_SESSION, EXPOSURE_LEVELS, OPERABLE_RETURNS,
                                      run_session, simple_returns)

N_FEATURES = len(FEATURE_ORDER)
POSITION_FEATURES = GROUPS["posicion"]
OBS_CLIP = 5.0                                    # §6.6


@dataclass(frozen=True)
class SessionSpec:
    """Todo lo que define una sesión para el entorno. Inmutable a propósito."""

    date: object
    close: np.ndarray                             # 60 cierres
    market: np.ndarray                            # (60, n_market) ya escalado
    context: np.ndarray                           # macro + régimen, constante en el día
    spread_pips: float

    def __post_init__(self) -> None:
        if len(self.close) != BARS_PER_SESSION:
            raise ValueError(f"{self.date}: {len(self.close)} barras, se esperan "
                             f"{BARS_PER_SESSION}")


def position_state(w_prev: float, bars_in_pos: int, unrealized: float,
                   drawdown: float, n_changes: int) -> np.ndarray:
    """Las cinco features endógenas, normalizadas a magnitudes comparables."""
    return np.array([
        w_prev,
        np.clip(unrealized * 100.0, -OBS_CLIP, OBS_CLIP),
        bars_in_pos / OPERABLE_RETURNS,
        np.clip(drawdown * 100.0, -OBS_CLIP, OBS_CLIP),
        n_changes / OPERABLE_RETURNS,
    ], dtype=np.float32)


class SessionTradingEnv(gym.Env if _GYM else object):
    """Un episodio = una sesión de 59 decisiones. Acción discreta de 5 niveles (§2, dec. 4)."""

    metadata = {"render_modes": []}

    def __init__(self, sessions: list[SessionSpec], seed: int | None = None,
                 reward_scale: float = 100.0, shuffle: bool = True):
        if not _GYM:                                          # pragma: no cover
            raise ImportError("gymnasium no está instalado")
        if not sessions:
            raise ValueError("no hay sesiones")
        self.sessions = sessions
        self.reward_scale = float(reward_scale)
        self.shuffle = shuffle
        self._rng = np.random.default_rng(seed)
        self._order = np.arange(len(sessions))
        self._cursor = 0

        n_ctx = len(sessions[0].context)
        n_mkt = sessions[0].market.shape[1]
        self._obs_dim = n_mkt + len(POSITION_FEATURES) + n_ctx
        if self._obs_dim != N_FEATURES:
            raise ValueError(
                f"dimensión de observación {self._obs_dim} != {N_FEATURES} del esquema "
                f"(mercado {n_mkt} + posición {len(POSITION_FEATURES)} + contexto {n_ctx})"
            )

        self.action_space = spaces.Discrete(len(EXPOSURE_LEVELS))
        self.observation_space = spaces.Box(low=-OBS_CLIP, high=OBS_CLIP,
                                            shape=(self._obs_dim,), dtype=np.float32)
        self.last_result = None

    # -- ciclo -------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._cursor >= len(self._order):
            self._cursor = 0
            if self.shuffle:
                self._rng.shuffle(self._order)
        self._spec = self.sessions[self._order[self._cursor]]
        self._cursor += 1

        self._b = 0
        self._weights: list[float] = []
        self._w_prev = 0.0                       # §9.1: w_{-1} = 0
        self._bars_in_pos = 0
        self._n_changes = 0
        self._entry_price = None
        self._peak = 0.0
        self._cum = 0.0
        self._returns = simple_returns(self._spec.close)
        self._sigma = realized_vol_pips(self._spec.close)
        return self._observe(), {}

    def step(self, action: int):
        w = float(EXPOSURE_LEVELS[int(action)])
        dw = w - self._w_prev

        # Costo de ESTE cambio, con la misma fórmula que `cost_model` (§9.3).
        c = self._spec.close[self._b]
        cost = abs(dw) * (self._spec.spread_pips / 2.0 + 0.5)
        cost += 0.1 * abs(dw) * self._sigma[self._b]
        cost_ret = cost / c

        gross = w * self._returns[self._b]        # decidir en b captura r_{b+1}
        net = gross - cost_ret

        self._cum += net
        self._peak = max(self._peak, self._cum)
        if dw != 0.0:
            self._n_changes += 1
            self._entry_price = c if w != 0.0 else None
            self._bars_in_pos = 0
        if w != 0.0:
            self._bars_in_pos += 1

        self._weights.append(w)
        self._w_prev = w
        self._b += 1

        terminated = self._b >= OPERABLE_RETURNS
        terminal_cost_ret = 0.0
        if terminated:
            # The terminal liquidation is part of the economic objective.  Keep
            # run_session as the accounting authority and charge exactly the same
            # terminal cost in the final reward step.
            self.last_result = run_session(self._spec.close, np.asarray(self._weights),
                                           self._spec.spread_pips, date=self._spec.date)
            terminal_cost_ret = self.last_result.terminal_cost
            obs = np.zeros(self._obs_dim, dtype=np.float32)
            info = {"daily_return": self.last_result.daily_return,
                    "n_changes": self.last_result.n_changes,
                    "terminal_cost": self.last_result.terminal_cost,
                    "date": self._spec.date}
        else:
            obs, info = self._observe(), {}

        return obs, float((net - terminal_cost_ret) * self.reward_scale), terminated, False, info

    # -- observación -------------------------------------------------------
    def _observe(self) -> np.ndarray:
        unreal = 0.0
        if self._entry_price and self._w_prev != 0.0:
            unreal = self._w_prev * (self._spec.close[self._b] / self._entry_price - 1.0)
        pos = position_state(self._w_prev, self._bars_in_pos, unreal,
                             self._cum - self._peak, self._n_changes)
        obs = np.concatenate([self._spec.market[self._b], pos, self._spec.context])
        return np.clip(obs, -OBS_CLIP, OBS_CLIP).astype(np.float32)


def replay_weights(spec: SessionSpec, weights) -> object:
    """Puntúa una senda fuera del `Env` — el lado offline del test de paridad."""
    return run_session(spec.close, np.asarray(weights, dtype=float),
                       spec.spread_pips, date=spec.date)
