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


def make_sessions(fixture: Fixture | str, n: int = 500, seed: int = 0,
                  spread_pips: float = 3.0) -> list[SessionSpec]:
    """Generate reproducible ``SessionSpec`` objects for one control fixture."""
    fixture = Fixture(fixture)
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

