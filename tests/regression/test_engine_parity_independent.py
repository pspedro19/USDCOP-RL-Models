"""
Regression: el motor de contabilidad contra una implementación independiente.

Contract: CTR-RESEARCH-SESSIONENV-001 · Date: 2026-08-25
Implementa el **test 7** de §13 del plan de tesis (paridad de motor, tolerancia 1e-6).

## Por qué NO se usa `vectorbt`

§13 nombraba `vectorbt` como segundo motor. No está instalado, y añadir una dependencia
pesada para una comprobación de aritmética empeora la reproducibilidad del trabajo sin
mejorar lo que el test mide. Lo que el test mide es: **¿la contabilidad es correcta, o solo
autoconsistente?**

Para eso sirve cualquier implementación construida sobre otra formulación. Aquí la referencia
lleva **caja y unidades** —compra y vende nocional, marca a mercado, acumula efectivo— mientras
`run_session` trabaja con **retornos y pesos**. Son dos álgebras distintas para el mismo hecho
económico: si coinciden a 1e-6 sobre sendas aleatorias, el resultado no depende de la
formulación.

Un motor comparado consigo mismo no prueba nada; `test_session_gym_parity.py` ya cubre la
consistencia interna (`gym.Env` ↔ `run_session`). Este fichero cubre lo otro.

## Qué NO cubre

La referencia reimplementa la MISMA fórmula de costos de §9.3 — no podría ser de otro modo,
porque el contrato de costos es una decisión, no un hecho verificable. Lo que se valida es la
**mecánica**: apertura, mantenimiento, flip, cierre terminal y composición.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.cost_model import (COMMISSION_PIPS_PER_SIDE,  # noqa: E402
                                     SLIPPAGE_COEF, realized_vol_pips)
from src.research.session_env import (BARS_PER_SESSION, EXPOSURE_LEVELS,  # noqa: E402
                                      OPERABLE_RETURNS, run_session)

TOL = 1e-6


def reference_session_pnl(close: np.ndarray, weights: np.ndarray,
                          spread_pips: float) -> float:
    """Motor de referencia: CAJA y UNIDADES, no pesos y retornos.

    Formulación deliberadamente distinta de la de `session_env`:

    - Se mantiene un nocional constante de 1 unidad de capital. La exposición `w` se traduce
      en `units = w / precio`, o sea el número de USD que se tienen (positivo) o se deben
      (negativo).
    - Cada cambio de exposición compra o vende unidades **al precio de la barra** y descuenta
      el costo en efectivo, en la misma moneda.
    - Al final se liquida todo y se devuelve el efectivo acumulado sobre el nocional inicial.

    Nadie escribiría esto para un backtest —es más lento y más torpe—, y esa es exactamente la
    virtud: no comparte una sola línea de álgebra con la implementación que valida.
    """
    c = np.asarray(close, dtype=float)
    w = np.asarray(weights, dtype=float)
    sigma = realized_vol_pips(c)

    cash = 0.0          # efectivo acumulado, en unidades de capital inicial
    units = 0.0         # USD en posición (signo = dirección)
    prev_w = 0.0

    for b in range(len(w)):
        # El precio al que se opera es el cierre de la barra b (mid; el spread se cobra aparte).
        price = c[b]
        target_units = w[b] / price
        delta_units = target_units - units

        # Comprar unidades cuesta efectivo; venderlas lo aporta.
        cash -= delta_units * price

        # Costo de transaccion, en unidades de capital: |dw| pips / precio.
        dw = w[b] - prev_w
        cost_pips = abs(dw) * (spread_pips / 2.0 + COMMISSION_PIPS_PER_SIDE)
        cost_pips += SLIPPAGE_COEF * abs(dw) * sigma[b]
        cash -= cost_pips / price

        units = target_units
        prev_w = w[b]

    # Cierre terminal: se liquida al ultimo precio observado y se cobra el cambio.
    last_price = c[len(w)]          # la barra siguiente a la ultima decision
    cash += units * last_price

    terminal_pips = abs(0.0 - prev_w) * (spread_pips / 2.0 + COMMISSION_PIPS_PER_SIDE)
    terminal_pips += SLIPPAGE_COEF * abs(prev_w) * sigma[len(w) - 1]
    cash -= terminal_pips / c[len(w) - 1]

    return cash


def random_close(seed: int, n: int = BARS_PER_SESSION) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 4000.0 + np.cumsum(rng.normal(0, 2.5, n))


# ---------------------------------------------------------------------------
# TEST 7 — paridad de motor
# ---------------------------------------------------------------------------

def test_t7_independent_engine_agrees_on_random_paths():
    """60 sendas aleatorias, dos formulaciones, tolerancia 1e-6."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in range(60):
        close = random_close(k)
        spread = float(rng.choice([2.0, 3.0, 6.0]))
        w = np.array([EXPOSURE_LEVELS[i]
                      for i in rng.integers(0, len(EXPOSURE_LEVELS), OPERABLE_RETURNS)])

        ours = run_session(close, w, spread).daily_return
        ref = reference_session_pnl(close, w, spread)
        worst = max(worst, abs(ours - ref))
        assert abs(ours - ref) < TOL, (
            f"senda {k}: motor {ours:.10f} vs referencia {ref:.10f} "
            f"(delta {abs(ours - ref):.2e}, spread {spread})"
        )
    assert worst < TOL


def test_t7_agreement_holds_for_the_edge_policies():
    """Los casos límite: no operar, mantener largo, mantener corto, y flip cada barra."""
    close = random_close(99)
    policies = {
        "flat": np.zeros(OPERABLE_RETURNS),
        "long": np.ones(OPERABLE_RETURNS),
        "short": -np.ones(OPERABLE_RETURNS),
        "flip": np.array([1.0 if b % 2 == 0 else -1.0 for b in range(OPERABLE_RETURNS)]),
        "half": np.full(OPERABLE_RETURNS, 0.5),
    }
    for name, w in policies.items():
        for spread in (2.0, 6.0):
            ours = run_session(close, w, spread).daily_return
            ref = reference_session_pnl(close, w, spread)
            assert abs(ours - ref) < TOL, (
                f"{name} @ spread {spread}: {ours:.10f} vs {ref:.10f}"
            )


def test_t7_the_reference_is_not_a_copy():
    """Contraprueba: la referencia tiene que DISCREPAR de un motor deliberadamente roto.

    Sin esto, un test de paridad pasa aunque ambas implementaciones compartan el mismo error
    — que es la forma en que una comprobación de paridad se vuelve decorativa.
    """
    close = random_close(7)
    w = np.ones(OPERABLE_RETURNS)

    ref = reference_session_pnl(close, w, spread_pips=3.0)
    # Motor roto: se "olvida" del cierre terminal, el error clasico que §9.1 previene.
    broken = run_session(close, w, 3.0)
    broken_value = broken.daily_return + broken.terminal_cost

    assert abs(broken_value - ref) > TOL, (
        "la referencia coincide con un motor al que le falta el costo terminal: "
        "no está comprobando nada"
    )


def test_t7_flat_policy_is_exactly_zero_in_both_engines():
    close = random_close(3)
    w = np.zeros(OPERABLE_RETURNS)
    assert abs(reference_session_pnl(close, w, 6.0)) < 1e-15
    assert abs(run_session(close, w, 6.0).daily_return) < 1e-15
