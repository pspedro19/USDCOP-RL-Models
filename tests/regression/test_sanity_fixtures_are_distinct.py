"""Las cuatro fixtures de sanidad tienen que probar cuatro condiciones distintas.

La bateria S1-S4 solo vale si cada fixture aisla una pregunta:

    S1  ruido + coste            -> ¿se abstiene cuando no hay nada que ganar?
    S2  senal SIN coste          -> ¿aprende la senal siquiera?
    S3  la misma senal CON coste -> ¿opera cuando el alfa supera el coste?
    S4  senal por debajo del coste -> ¿se abstiene cuando no lo supera?

El 2026-09-11 se midio que **S2 y S3 eran la misma fixture byte a byte**: ambas declaraban
`alpha = 0.002` y ambas heredaban el `spread_pips = 3.0` por defecto, asi que con la misma
semilla generaban cierres y features identicos. El agregado publicado decia "las cuatro
fixtures pasan" cuando en realidad se habian verificado **tres condiciones, una contada dos
veces**, y sus netos coincidian hasta el quinto decimal en cuatro de las cinco semillas.

Eso no invalidaba el patron discriminante -- plano en S1/S4, operando en S2/S3 -- pero si la
afirmacion de cobertura, que es lo que una compuerta vende.

Este test fija las dos propiedades que faltaban: que ninguna pareja de fixtures sea
indistinguible, y que la diferencia entre S2 y S3 sea exactamente el coste.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

pytest.importorskip("pandas")

from src.research.synthetic_sessions import (
    Fixture,
    make_sessions,
    oracle_result,
)


def _fingerprint(fixture: Fixture, seed: int = 42, n: int = 6):
    sessions = make_sessions(fixture, n=n, seed=seed)
    closes = np.concatenate([s.close for s in sessions])
    spreads = np.array([s.spread_pips for s in sessions], dtype=float)
    return closes, spreads


@pytest.mark.parametrize("left,right", list(itertools.combinations(list(Fixture), 2)))
def test_no_two_fixtures_are_identical(left: Fixture, right: Fixture) -> None:
    """Dos fixtures indistinguibles no son dos pruebas: son una contada dos veces."""
    left_closes, left_spreads = _fingerprint(left)
    right_closes, right_spreads = _fingerprint(right)
    identical = (np.array_equal(left_closes, right_closes)
                 and np.array_equal(left_spreads, right_spreads))
    assert not identical, (
        f"{left.value} y {right.value} generan exactamente los mismos datos con la misma "
        "semilla. La bateria afirmaria cubrir dos condiciones y cubriria una."
    )


def test_s2_is_free_of_cost_and_s3_is_not() -> None:
    """La diferencia entre S2 y S3 es el coste, y solo el coste."""
    planted = make_sessions(Fixture.SIGNAL_PLANTED, n=4, seed=7)
    above = make_sessions(Fixture.SIGNAL_ABOVE_COST, n=4, seed=7)

    assert all(s.spread_pips == 0.0 for s in planted), (
        "S2 debe ser SIN coste: es la unica fixture que puede responder '¿aprende la senal "
        "siquiera?' sin que un fallo se confunda con no poder pagar el spread."
    )
    assert all(s.spread_pips > 0.0 for s in above), "S3 debe cobrar coste"

    # Mismo proceso generador: la senal y los precios son los mismos, cambia el peaje.
    for left, right in zip(planted, above, strict=False):
        np.testing.assert_array_equal(left.close, right.close)
        np.testing.assert_array_equal(left.market, right.market)


def test_s3_oracle_strictly_pays_the_declared_cost() -> None:
    """The same known-good actions must have lower net return under S3's toll."""
    free = make_sessions(Fixture.SIGNAL_PLANTED, n=4, seed=19)
    charged = make_sessions(Fixture.SIGNAL_ABOVE_COST, n=4, seed=19)
    free_net = np.mean([oracle_result(Fixture.SIGNAL_PLANTED, s).daily_return for s in free])
    charged_net = np.mean([oracle_result(Fixture.SIGNAL_ABOVE_COST, s).daily_return for s in charged])
    assert charged_net < free_net


def test_s2_pays_no_spread_commission_or_slippage() -> None:
    """S2 is now genuinely free; historical partial-cost evidence is obsolete."""
    sessions = make_sessions(Fixture.SIGNAL_PLANTED, n=3, seed=5)
    results = [oracle_result(Fixture.SIGNAL_PLANTED, s) for s in sessions]
    assert all(s.spread_pips == 0.0 for s in sessions)
    assert all(r.total_cost == 0.0 for r in results)
    assert all(r.daily_return == r.gross_return for r in results)


def test_every_fixture_declares_a_spread() -> None:
    for fixture in Fixture:
        sessions = make_sessions(fixture, n=2, seed=1)
        assert all(np.isfinite(s.spread_pips) for s in sessions)
        assert all(s.spread_pips >= 0.0 for s in sessions)
