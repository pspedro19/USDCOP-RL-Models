"""
Regression: cronología, costos y contabilidad del entorno de sesión.

Contract: CTR-RESEARCH-SESSIONENV-001 / CTR-RESEARCH-COST-001
Implementa los tests **9, 10 y 13** de §13 del plan de tesis.

Estos tres son la base de todo lo que venga después: si la cronología deja que una decisión
capture su propio retorno, o si el costo terminal no se cobra, **todas** las métricas de la
tesis quedan infladas y ninguna tabla lo delataría — los números seguirían siendo plausibles.

- **Test 9**  — apertura, reducción, flip y cierre cobran los lados correctos, sin doble conteo.
- **Test 10** — una acción al cierre de `b` solo captura `r_{b+1}`.
- **Test 13** — la serie diaria coincide con la variación de equity **e incluye el costo del
  cierre terminal**; el Sharpe no se calcula sobre barras.

Sin Postgres, sin Airflow, sin datos de mercado: sesiones sintéticas donde la respuesta
correcta se conoce a mano.
"""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.cost_model import (  # noqa: E402
    COMMISSION_PIPS_PER_SIDE, bar_cost, min_round_trip_pips, session_costs)
from src.research.session_env import (  # noqa: E402
    BARS_PER_SESSION, EXPOSURE_LEVELS, OPERABLE_RETURNS, constant_policy,
    equity_curve, run_session, simple_returns)


def flat_close(price: float = 4000.0) -> np.ndarray:
    """Sesión de precio constante: los retornos son 0, así el costo queda aislado."""
    return np.full(BARS_PER_SESSION, price, dtype=float)


def drifting_close(start: float = 4000.0, step: float = 1.0) -> np.ndarray:
    return start + step * np.arange(BARS_PER_SESSION, dtype=float)


# ---------------------------------------------------------------------------
# TEST 9 — los costos cobran los lados correctos, sin doble conteo
# ---------------------------------------------------------------------------

def test_t9_round_trip_matches_the_declared_pips():
    """`2·(spread/2 + 0.5)` debe dar 3/4/7 pips para spread 2/3/6 (§9.3)."""
    assert [min_round_trip_pips(s) for s in (2.0, 3.0, 6.0)] == [3.0, 4.0, 7.0]


def test_t9_opening_charges_one_side():
    """Abrir 1× cobra medio spread + una comisión, nada más."""
    bd = bar_cost(dw=1.0, spread_pips=2.0, sigma12_pips=0.0, close=4000.0)
    assert bd.spread_component_pips == pytest.approx(2.0 / 2 + COMMISSION_PIPS_PER_SIDE)
    assert bd.slippage_component_pips == 0.0
    assert bd.cost_pips == pytest.approx(1.5)


def test_t9_flip_charges_two_sides_exactly_once():
    """`+1 -> -1` es `|dw|=2`: dos lados, UNA vez. Multiplicar por 2 sería doble conteo."""
    opening = bar_cost(1.0, spread_pips=3.0, sigma12_pips=0.0, close=4000.0)
    flip = bar_cost(-2.0, spread_pips=3.0, sigma12_pips=0.0, close=4000.0)
    assert flip.cost_pips == pytest.approx(2.0 * opening.cost_pips), (
        "un flip debe costar exactamente dos aperturas, ni más ni menos"
    )


def test_t9_partial_reduction_charges_proportionally():
    """Reducir de 1.0 a 0.5 cuesta la mitad que abrir 1.0: el costo escala con |dw|."""
    full = bar_cost(1.0, 3.0, 0.0, 4000.0)
    half = bar_cost(-0.5, 3.0, 0.0, 4000.0)
    assert half.cost_pips == pytest.approx(0.5 * full.cost_pips)


def test_t9_no_change_costs_nothing():
    assert bar_cost(0.0, 6.0, 100.0, 4000.0).cost_pips == 0.0


def test_t9_slippage_scales_with_volatility_and_size():
    """`0.1 · |dw| · sigma12`: doblar la vol o el tamaño dobla el slippage."""
    base = bar_cost(1.0, 2.0, sigma12_pips=10.0, close=4000.0)
    dbl_vol = bar_cost(1.0, 2.0, sigma12_pips=20.0, close=4000.0)
    dbl_size = bar_cost(2.0, 2.0, sigma12_pips=10.0, close=4000.0)
    assert dbl_vol.slippage_component_pips == pytest.approx(
        2 * base.slippage_component_pips)
    assert dbl_size.slippage_component_pips == pytest.approx(
        2 * base.slippage_component_pips)
    assert base.slippage_component_pips == pytest.approx(0.1 * 1.0 * 10.0)


def test_t9_cost_ret_is_pips_over_price():
    bd = bar_cost(1.0, 2.0, 0.0, close=4000.0)
    assert bd.cost_ret == pytest.approx(bd.cost_pips / 4000.0)


def test_t9_holding_all_session_costs_exactly_one_round_trip():
    """B1 mantiene `+1` todo el día: un cambio al abrir, otro al cerrar. Nada más."""
    close = flat_close()
    w = constant_policy(1.0)(close)
    costs, breakdown = session_costs(w, close, spread_pips=2.0)
    nonzero = [b for b in breakdown if b.cost_pips > 0]
    assert len(nonzero) == 2, (
        f"B1 debería cobrar 2 veces (apertura + cierre), cobra {len(nonzero)}"
    )
    total_pips = sum(b.cost_pips for b in breakdown)
    assert total_pips == pytest.approx(min_round_trip_pips(2.0)), (
        "el total de una posición mantenida debe ser el round-trip mínimo"
    )


# ---------------------------------------------------------------------------
# TEST 10 — cronología: la decisión en `b` captura `r_{b+1}`, nunca `r_b`
# ---------------------------------------------------------------------------

def test_t10_there_are_exactly_59_operable_returns():
    """60 barras, 59 retornos. Contar 60 dejaría a la última decisión cobrar un retorno
    que no existe."""
    assert OPERABLE_RETURNS == 59
    assert len(simple_returns(flat_close())) == 59
    with pytest.raises(ValueError, match="decisiones operables"):
        run_session(flat_close(), np.zeros(BARS_PER_SESSION), 2.0)


def test_t10_action_at_bar_b_captures_only_the_next_bar():
    """Con exposición en UNA sola barra, el retorno bruto es exactamente `r_{b+1}`."""
    close = drifting_close()
    r = simple_returns(close)
    for b in (0, 17, 58):
        w = np.zeros(OPERABLE_RETURNS)
        w[b] = 1.0
        res = run_session(close, w, spread_pips=2.0)
        assert res.gross_return == pytest.approx(r[b]), (
            f"la decisión en la barra {b} debería capturar r[{b}] (= r_{{{b}+1}} del plan)"
        )


def test_t10_a_position_taken_after_the_move_earns_nothing_from_it():
    """El caso que delata el look-ahead: el precio salta en `r_5`; posicionarse en `b=5`
    (o sea, decidir tras ver ese salto) NO puede cobrarlo."""
    close = flat_close()
    close[6:] += 40.0                      # el salto ocurre entre la barra 5 y la 6 => r_5
    r = simple_returns(close)
    assert r[5] > 0 and np.allclose(np.delete(r, 5), 0.0)

    w_before = np.zeros(OPERABLE_RETURNS); w_before[5] = 1.0
    w_after = np.zeros(OPERABLE_RETURNS); w_after[6] = 1.0
    assert run_session(close, w_before, 2.0).gross_return == pytest.approx(r[5])
    assert run_session(close, w_after, 2.0).gross_return == pytest.approx(0.0), (
        "posicionarse DESPUES del movimiento no puede capturarlo"
    )


# ---------------------------------------------------------------------------
# TEST 13 — contabilidad: serie diaria == variación de equity, con costo terminal
# ---------------------------------------------------------------------------

def test_t13_terminal_cost_is_charged_and_visible():
    """§9.1: el cierre terminal es una regla del entorno, no una acción evitable."""
    res = run_session(flat_close(), constant_policy(1.0)(flat_close()), spread_pips=2.0)
    assert res.terminal_cost > 0, "el cierre terminal no se cobró"
    assert len(res.costs) == OPERABLE_RETURNS + 1, (
        "debe haber un paso de costo MAS que decisiones: el terminal tiene costo y no retorno"
    )
    assert res.costs[-1] == pytest.approx(res.terminal_cost)


def test_terminal_close_uses_bar_59_price():
    """La liquidación usa el cierre final, no el cierre de la decisión 58."""
    close = np.full(BARS_PER_SESSION, 4000.0)
    close[-1] = 4400.0
    res = run_session(close, constant_policy(1.0)(close), spread_pips=3.0)
    # The last jump contributes to terminal slippage as well as spread/commission.
    assert res.terminal_cost == pytest.approx(0.003205913352872407)


def test_t13_daily_return_is_gross_minus_all_costs_including_terminal():
    res = run_session(drifting_close(), constant_policy(1.0)(drifting_close()), 3.0)
    assert res.daily_return == pytest.approx(res.gross_return - res.total_cost)
    assert res.total_cost >= res.terminal_cost > 0


def test_t13_flat_session_loses_exactly_the_round_trip():
    """Precio plano y posición mantenida: el resultado del día es -round_trip/precio."""
    close = flat_close(4000.0)
    res = run_session(close, constant_policy(1.0)(close), spread_pips=2.0)
    assert res.gross_return == pytest.approx(0.0)
    assert res.daily_return == pytest.approx(-min_round_trip_pips(2.0) / 4000.0)


def test_t13_always_flat_costs_nothing_at_all():
    """No operar no puede costar. Si cuesta, el cierre terminal está cobrando de más."""
    close = drifting_close()
    res = run_session(close, constant_policy(0.0)(close), spread_pips=6.0)
    assert res.total_cost == pytest.approx(0.0)
    assert res.daily_return == pytest.approx(0.0)


def test_t13_equity_is_compounded_not_summed():
    """§9.2: `E_{d+1} = E_d·(1+r_d)`. Una suma acumulada daría otro número."""
    close = flat_close()
    results = [run_session(close, constant_policy(1.0)(close), 2.0) for _ in range(5)]
    eq = equity_curve(results, initial=10_000.0)
    r = results[0].daily_return
    assert eq[-1] == pytest.approx(10_000.0 * (1.0 + r) ** 5)
    assert eq[-1] != pytest.approx(10_000.0 * (1.0 + 5 * r)), (
        "la equity está sumando en vez de componer"
    )


def test_t13_metrics_are_computed_on_daily_returns_not_bars():
    """El Sharpe de la tesis vive sobre `retorno_diario_d`, no sobre barras de 5 minutos."""
    close = drifting_close()
    res = run_session(close, constant_policy(1.0)(close), 2.0)
    assert len(res.bar_returns) == OPERABLE_RETURNS
    assert isinstance(res.daily_return, float), (
        "una sesión aporta UN dato a la serie diaria, no 59"
    )


# ---------------------------------------------------------------------------
# Espacio de acción congelado (§2, decisión 4)
# ---------------------------------------------------------------------------

def test_action_space_is_the_frozen_five_levels():
    assert EXPOSURE_LEVELS == (-1.0, -0.5, 0.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="fuera del espacio congelado"):
        constant_policy(0.75)


# ---------------------------------------------------------------------------
# La vectorizacion de `realized_vol_pips` no puede cambiar un solo costo
# ---------------------------------------------------------------------------

def _realized_vol_loop(close, window=12):
    """La implementación original, en bucle. Se conserva SOLO como referencia del test."""
    c = np.asarray(close, dtype=float)
    log_ret = np.diff(np.log(c), prepend=np.log(c[0]))
    out = np.zeros_like(c)
    for i in range(len(c)):
        seg = log_ret[max(0, i - window + 1): i + 1]
        out[i] = c[i] * (np.std(seg, ddof=1) if len(seg) > 1 else 0.0)
    return out


def test_realized_vol_vectorization_is_exact():
    """`realized_vol_pips` se vectorizó por velocidad; tiene que dar lo MISMO.

    Se llama una vez por `run_session`, y el re-scoring de frecuencia la invoca 29.200 veces
    sobre los mismos cierres — en bucle tardaba más que todo el resto del análisis junto.

    Pero es la función que fija el término de slippage de **todos** los costos de la tesis:
    si la versión rápida divergiera de la lenta, cambiarían todas las tablas y **ninguna lo
    delataría**, porque los números seguirían siendo plausibles. De ahí que la referencia en
    bucle viva dentro del test.
    """
    from src.research.cost_model import realized_vol_pips

    rng = np.random.default_rng(0)
    worst = 0.0
    for seed in range(30):
        close = 4000.0 + np.cumsum(np.random.default_rng(seed).normal(0, 2.5, 60))
        worst = max(worst, float(np.abs(realized_vol_pips(close)
                                        - _realized_vol_loop(close)).max()))
    assert worst < 1e-9, f"la version vectorizada diverge del bucle: {worst:.3e}"

    # Casos limite donde una formula por sumas acumuladas se rompe facil.
    flat = np.full(60, 4000.0)
    assert np.allclose(realized_vol_pips(flat), 0.0), "precio constante => vol cero"
    assert np.all(np.isfinite(realized_vol_pips(flat)))
    assert realized_vol_pips(np.array([4000.0]))[0] == 0.0, "una barra no tiene dispersion"
    assert realized_vol_pips(flat)[0] == 0.0, "la barra 0 nunca tiene ventana"
