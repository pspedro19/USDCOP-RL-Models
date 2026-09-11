"""Entorno de una sesión: 60 barras, 59 retornos operables, cierre terminal cobrado.

Contract: CTR-RESEARCH-SESSIONENV-001 · Date: 2026-08-24

Implementa §9.1-§9.2 y §9.6 del plan de tesis, y es el sujeto de los tests 9, 10, 13 y 15.

## Por qué NO se reutiliza `src/training/environments/trading_env.py`

Aquel entorno es correcto para producción y contradice cuatro decisiones **congeladas** de §2:

| §2 / §9.1 congela | El entorno de producción hace |
|---|---|
| Episodio = una sesión, barras 0-59 | 2.400 barras (~40 días) |
| Acción = exposición `{-1, -0.5, 0, +0.5, +1}` (decisión 4) | 4 discretas: HOLD/BUY/SELL/CLOSE |
| Cierre terminal forzado **y cobrado** (decisión 5) | stop-loss, take-profit, trailing, circuit breaker |
| Costo en pips dependiente del régimen (decisión 8) | `transaction_cost_bps: 2.5` plano |

Reutilizarlo habría cambiado la tesis sin que nadie lo escribiera en ningún sitio. Este
entorno es más pequeño precisamente porque el diseño de la tesis es más simple: no hay stops,
no hay re-entradas, no hay curriculum — hay exposición y su costo.

## La cronología, tal cual §9.1

```
cierre de barra b, b = 0..58
  -> observar x_b con informacion <= cierre_b
  -> decidir w_b
  -> pagar costo por dw_b = w_b - w_{b-1}
  -> mantener w_b durante la barra b+1
  -> recibir r_{b+1}

cierre de barra 59  (paso terminal, SIN retorno)
  -> dw_close = 0 - w_58
  -> cobrar costo terminal
  -> fin del episodio
```

`w_{-1} = 0`. **59 retornos operables**, no 60: la decisión de la barra 59 no tiene barra
siguiente que capturar. Contar 60 sería el error clásico de dejar que la última decisión
cobre un retorno que nunca ocurrió.

## Reward != contabilidad económica

§9.6 y el principio 5: el reward que entrena al agente y la serie de retornos que se reporta
son **cosas distintas**. `step()` devuelve el reward; `economic_returns()` devuelve la serie
con la que se calculan Sharpe, Calmar y drawdown. Mezclarlas hace que mejorar el reward
parezca ganar dinero.

## Los baselines viven aquí dentro

B1 es `w=+1` siempre, NULL-A es `w=-1`, always-flat es `w=0`. Al ser políticas del MISMO
entorno comparten motor de costos, cronología y contabilidad con el PPO **por construcción**,
no por acuerdo entre dos implementaciones que podrían divergir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from src.research.cost_model import session_costs

# §2, decisión 4: espacio de acción congelado.
EXPOSURE_LEVELS: tuple[float, ...] = (-1.0, -0.5, 0.0, 0.5, 1.0)
BARS_PER_SESSION = 60
OPERABLE_RETURNS = BARS_PER_SESSION - 1     # 59 (§9.1)


@dataclass
class SessionResult:
    """Lo que produce una sesión. `daily_return` es la unidad de la contabilidad."""

    date: object
    weights: np.ndarray                     # w_0 .. w_58 (59 decisiones operables)
    bar_returns: np.ndarray                 # r_1 .. r_59
    costs: np.ndarray                       # incluye el paso terminal
    gross_return: float
    total_cost: float
    daily_return: float
    terminal_cost: float
    n_changes: int
    mean_abs_exposure: float
    spread_pips: float
    breakdown: list = field(default_factory=list)


def simple_returns(close: np.ndarray) -> np.ndarray:
    """Retornos SIMPLES (§9.2), no logarítmicos.

    El plan lo corrige explícitamente: mezclar log-retornos con costos en retornos simples
    es incoherente, y la equity se compone con simples. Aquí no hay logaritmos.
    """
    c = np.asarray(close, dtype=float)
    return c[1:] / c[:-1] - 1.0


def run_session(close: np.ndarray, weights: Sequence[float], spread_pips: float,
                date=None) -> SessionResult:
    """Ejecuta una sesión con una senda de exposición dada.

    `weights` son las 59 decisiones operables `w_0..w_58`. El cierre terminal lo añade el
    entorno; no es una acción que el agente pueda evitar.
    """
    c = np.asarray(close, dtype=float)
    if len(c) != BARS_PER_SESSION:
        raise ValueError(f"la sesión debe tener {BARS_PER_SESSION} barras, tiene {len(c)}")
    w = np.asarray(weights, dtype=float)
    if len(w) != OPERABLE_RETURNS:
        raise ValueError(
            f"se esperan {OPERABLE_RETURNS} decisiones operables (§9.1), llegan {len(w)}")

    r = simple_returns(c)                         # r_1 .. r_59, longitud 59
    costs, breakdown = session_costs(w, c, spread_pips,
                                     include_terminal=True)

    gross = float(np.sum(w * r))
    total_cost = float(np.sum(costs))
    terminal = float(costs[-1])
    changes = int(np.sum(np.abs(np.diff(np.concatenate([[0.0], w]))) > 1e-12))

    return SessionResult(
        date=date, weights=w, bar_returns=r, costs=costs,
        gross_return=gross, total_cost=total_cost,
        daily_return=gross - total_cost, terminal_cost=terminal,
        n_changes=changes, mean_abs_exposure=float(np.mean(np.abs(w))),
        spread_pips=float(spread_pips), breakdown=breakdown,
    )


# ---------------------------------------------------------------------------
# Políticas fijas: los baselines de §10.6 / §10.8
# ---------------------------------------------------------------------------

def constant_policy(level: float) -> Callable[[np.ndarray], np.ndarray]:
    """B1 (`+1`), NULL-A (`-1`), always-flat (`0`). Un solo cambio al abrir y otro al cerrar."""
    if level not in EXPOSURE_LEVELS:
        raise ValueError(f"exposición {level} fuera del espacio congelado {EXPOSURE_LEVELS}")
    return lambda close: np.full(OPERABLE_RETURNS, float(level))


def random_policy(seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """Control aleatorio (§10.8). Semilla fija: un control irreproducible no controla nada."""
    def policy(close: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.choice(np.asarray(EXPOSURE_LEVELS), size=OPERABLE_RETURNS)
    return policy


def run_policy(sessions: dict, spreads: dict, policy: Callable[[np.ndarray], np.ndarray]
               ) -> list[SessionResult]:
    """Aplica una política a todas las sesiones de la máscara.

    `sessions[fecha] -> close (60 barras)`; `spreads[fecha] -> spread_pips`. Una fecha sin
    spread se OMITE en vez de costearse con un valor por defecto: inventar el costo de un día
    del que no sabemos el régimen es exactamente el tipo de relleno que la máscara existe para
    impedir.
    """
    out = []
    for date, close in sessions.items():
        sp = spreads.get(date)
        if sp is None or not np.isfinite(sp):
            continue
        out.append(run_session(close, policy(close), sp, date=date))
    return out


def daily_series(results: Sequence[SessionResult]) -> np.ndarray:
    """La serie diaria con la que se calculan TODAS las métricas económicas."""
    return np.asarray([r.daily_return for r in results], dtype=float)


def equity_curve(results: Sequence[SessionResult], initial: float = 10_000.0) -> np.ndarray:
    """Equity COMPUESTA (§9.2): `E_{d+1} = E_d · (1 + r_d)`, no suma acumulada."""
    return initial * np.cumprod(1.0 + daily_series(results))
