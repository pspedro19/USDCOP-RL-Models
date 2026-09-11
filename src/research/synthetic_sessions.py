"""Deterministic synthetic sessions with known economic optima.

These fixtures are controls for the PPO recipe, never evidence about USD/COP or
gold.  The planted direction is exposed in market feature zero and predicts the
*next* bar only, matching the session environment chronology.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from src.research.features import GROUPS
from src.research.session_env import BARS_PER_SESSION, OPERABLE_RETURNS, run_session
from src.research.session_gym import SessionSpec

N_MARKET = sum(len(GROUPS[g]) for g in ("precio", "volatilidad", "tendencia", "temporal"))
N_CONTEXT = len(GROUPS["macro"]) + len(GROUPS["regimen"])


class Fixture(StrEnum):
    NOISE_WITH_COST = "S1"
    SIGNAL_PLANTED = "S2"
    SIGNAL_ABOVE_COST = "S3"
    SIGNAL_BELOW_COST = "S4"


def _alpha(fixture: Fixture) -> float:
    return {
        Fixture.NOISE_WITH_COST: 0.0,
        Fixture.SIGNAL_PLANTED: 0.002,
        Fixture.SIGNAL_ABOVE_COST: 0.002,
        Fixture.SIGNAL_BELOW_COST: 0.00015,
    }[fixture]


def _spread_pips(fixture: Fixture) -> float:
    """Coste por fixture. **S2 no cobra coste; S3 si.**

    Sin esto, S2 y S3 eran la MISMA fixture byte a byte: ambas declaran `alpha = 0.002` y
    ambas heredaban `spread_pips = 3.0`, asi que generaban cierres y features identicos con la
    misma semilla. La bateria decia probar cuatro condiciones y probaba tres, con una contada
    dos veces. Medido el 2026-09-11.

    Lo que cada una debe aislar, segun el diseno:
      * **S2** senal plantada **sin coste** -> ¿la receta aprende la senal siquiera?
      * **S3** la misma senal **pagando coste** -> ¿opera cuando el alfa lo supera?

    Con el mismo coste en ambas, S2 no puede responder su pregunta: un fallo de S2 seria
    indistinguible de un fallo de S3, y un aprobado de S3 hacia redundante a S2.
    """
    return 0.0 if fixture is Fixture.SIGNAL_PLANTED else 3.0


def make_sessions(fixture: Fixture | str, n: int = 500, seed: int = 0,
                  spread_pips: float | None = None) -> list[SessionSpec]:
    """Generate reproducible ``SessionSpec`` objects for one control fixture."""
    fixture = Fixture(fixture)
    if spread_pips is None:
        spread_pips = _spread_pips(fixture)
    rng = np.random.default_rng(seed)
    alpha = _alpha(fixture)
    sessions: list[SessionSpec] = []
    for i in range(n):
        direction = rng.choice(np.array([-1.0, 1.0]), size=OPERABLE_RETURNS)
        noise = rng.normal(0.0, 0.0004, size=OPERABLE_RETURNS)
        returns = direction * alpha + noise
        close = np.empty(BARS_PER_SESSION, dtype=float)
        close[0] = 4000.0
        close[1:] = close[0] * np.cumprod(1.0 + returns)
        market = rng.normal(0.0, 1.0, size=(BARS_PER_SESSION, N_MARKET)).astype(np.float32)
        # The first feature is available at b and predicts r[b+1].  S1 has no
        # predictive relation because its return alpha is zero.
        market[:OPERABLE_RETURNS, 0] = direction.astype(np.float32)
        market[-1, 0] = 0.0
        context = np.zeros(N_CONTEXT, dtype=np.float32)
        sessions.append(SessionSpec(
            date=f"synthetic-{i:04d}", close=close, market=market,
            context=context, spread_pips=spread_pips,
        ))
    return sessions


def oracle_weights(fixture: Fixture | str, spec: SessionSpec) -> np.ndarray:
    """Analytic policy for fixture assertions, not a learned policy."""
    fixture = Fixture(fixture)
    if fixture in (Fixture.NOISE_WITH_COST, Fixture.SIGNAL_BELOW_COST):
        return np.zeros(OPERABLE_RETURNS, dtype=float)
    return np.sign(spec.market[:OPERABLE_RETURNS, 0]).astype(float)


def oracle_result(fixture: Fixture | str, spec: SessionSpec):
    return run_session(spec.close, oracle_weights(fixture, spec), spec.spread_pips,
                       date=spec.date)

