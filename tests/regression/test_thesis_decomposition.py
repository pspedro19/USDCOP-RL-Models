"""
Regression: la descomposición bruto/costo y el break-even.

Contract: CTR-RESEARCH-DECOMP-001 · Date: 2026-08-25

## Qué se está protegiendo

El rechazo de la tesis (PPO no bate a `always_flat`) no depende de este módulo y no cambia.
Lo que sí depende es la **explicación**: que el agente tiene señal (bruto positivo en 10/10) y
que el costo de su frecuencia la anula.

Esa explicación descansa sobre dos cálculos que es fácil hacer mal y difícil ver mal:

1. **El re-scoring por frecuencia.** Con `k = 1` tiene que reproducir EXACTAMENTE el resultado
   publicado. Si no, la curva de frecuencia entera está comparando contra algo que no es el
   resultado de la tesis, y nadie lo notaría: los números seguirían pareciendo razonables.
2. **El break-even en forma cerrada.** Se despeja algebraicamente; se comprueba contra una
   búsqueda numérica sobre el motor real. Un despeje con un signo cambiado da un `s*`
   plausible y equivocado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.cost_model import COMMISSION_PIPS_PER_SIDE, realized_vol_pips  # noqa: E402
from src.research.decomposition import (break_even_spread, flat_paradox,  # noqa: E402
                                        hold_k_bars,
                                        production_cost_per_side_pips)
from src.research.session_env import (BARS_PER_SESSION, EXPOSURE_LEVELS,  # noqa: E402
                                      OPERABLE_RETURNS, run_session)


def synthetic_close(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 4000.0 + np.cumsum(rng.normal(0, 2.5, BARS_PER_SESSION))


def random_weights(seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([EXPOSURE_LEVELS[i]
                     for i in rng.integers(0, len(EXPOSURE_LEVELS), OPERABLE_RETURNS)])


# ---------------------------------------------------------------------------
# Re-scoring por frecuencia
# ---------------------------------------------------------------------------

def test_k1_is_the_identity():
    """`k=1` tiene que devolver la senda original, bit a bit.

    Es el ancla de toda la curva de frecuencia: si `k=1` no reproduce el resultado publicado,
    los demás `k` se comparan contra un punto de partida que no existe.
    """
    for s in range(5):
        w = random_weights(s)
        assert np.array_equal(hold_k_bars(w, 1), w)
        assert np.array_equal(hold_k_bars(w, 0), w)


def test_k1_reproduces_the_published_session_result():
    """La identidad, comprobada a través del motor y no solo del array."""
    close = synthetic_close(3)
    w = random_weights(4)
    original = run_session(close, w, 3.0)
    rescored = run_session(close, hold_k_bars(w, 1), 3.0)
    assert rescored.daily_return == pytest.approx(original.daily_return, rel=1e-15)
    assert rescored.n_changes == original.n_changes


def test_larger_k_never_increases_the_number_of_decisions():
    """Decidir cada `k` barras no puede producir MÁS puntos de decisión distintos."""
    w = random_weights(7)
    prev = len(set(hold_k_bars(w, 1).tolist()))
    for k in (5, 15, 30, 59):
        held = hold_k_bars(w, k)
        distinct_blocks = len(range(0, len(w), k))
        assert distinct_blocks <= len(w)
        # El turnover no puede crecer al mantener mas tiempo la posicion.
        assert np.abs(np.diff(held)).sum() <= np.abs(np.diff(w)).sum() + 1e-12
        prev = distinct_blocks


def test_k_equal_to_the_session_holds_one_position_all_day():
    w = random_weights(9)
    held = hold_k_bars(w, OPERABLE_RETURNS)
    assert len(set(held.tolist())) == 1
    assert held[0] == w[0]


def test_holding_longer_reduces_cost_on_a_churning_path():
    """El punto entero del análisis: menos decisiones, menos costo.

    Se usa una senda que alterna en cada barra —el peor caso de turnover— porque es donde el
    efecto tiene que verse sin ambigüedad.
    """
    close = synthetic_close(11)
    churn = np.array([1.0 if b % 2 == 0 else -1.0 for b in range(OPERABLE_RETURNS)])

    costs = [run_session(close, hold_k_bars(churn, k), 3.0).total_cost
             for k in (1, 5, 15, OPERABLE_RETURNS)]
    assert costs == sorted(costs, reverse=True), f"el costo no cae con k: {costs}"
    assert costs[-1] < costs[0] / 10, "mantener todo el dia debería costar un orden menos"


# ---------------------------------------------------------------------------
# Break-even
# ---------------------------------------------------------------------------

def _numeric_break_even(close, w, lo=-5.0, hi=60.0, tol=1e-7) -> float:
    """Búsqueda binaria sobre el motor REAL: la referencia contra la que se valida el despeje."""
    res0 = run_session(close, w, 0.0)
    gross = res0.gross_return

    def net_at(s):
        return gross - run_session(close, w, s).total_cost

    # `net_at` decrece en `s`; se busca el cero.
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if net_at(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return (lo + hi) / 2.0


def test_closed_form_break_even_matches_a_numeric_search():
    """El despeje algebraico contra la búsqueda numérica sobre el motor real.

    Un signo cambiado en el despeje produce un `s*` perfectamente plausible y equivocado;
    esta es la única forma de cazarlo.
    """
    for seed in range(6):
        close = synthetic_close(seed)
        w = random_weights(seed + 20)
        res = run_session(close, w, 3.0)

        sigma = realized_vol_pips(close)
        prev = np.concatenate([[0.0], w])
        sum_dw = float(np.abs(np.diff(np.concatenate([[0.0], w, [0.0]]))).sum())
        sum_dw_sigma = float(
            sum(abs(w[b] - prev[b]) * sigma[b] for b in range(len(w)))
            + abs(0.0 - w[-1]) * sigma[len(w) - 1])

        be = break_even_spread(sum_gross=res.gross_return, sum_abs_dw=sum_dw,
                               sum_abs_dw_sigma=sum_dw_sigma,
                               mean_close=float(np.mean(close)), spread_assumed=3.0)
        numeric = _numeric_break_even(close, w)

        assert be.spread_star_pips == pytest.approx(numeric, abs=0.05), (
            f"seed {seed}: forma cerrada {be.spread_star_pips:.4f} vs "
            f"numerica {numeric:.4f}"
        )


def test_a_policy_with_no_turnover_has_no_break_even():
    """Sin operaciones no hay costo que igualar; `s*` no está definido."""
    be = break_even_spread(sum_gross=0.0, sum_abs_dw=0.0, sum_abs_dw_sigma=0.0,
                           mean_close=4000.0, spread_assumed=3.0)
    assert be.spread_star_pips is None and be.viable_at_assumed is False


def test_negative_break_even_means_not_even_a_zero_spread_would_do():
    """Si el alfa no cubre ni la comisión, `s*` sale negativo — y eso es informativo."""
    be = break_even_spread(sum_gross=1e-6, sum_abs_dw=1000.0, sum_abs_dw_sigma=0.0,
                           mean_close=4000.0, spread_assumed=3.0)
    assert be.spread_star_pips < 0
    assert be.viable_at_assumed is False
    # Con spread 0 el costo por unidad de |dw| es la comision de 0,5 pips por lado.
    assert be.spread_star_pips == pytest.approx(-2 * COMMISSION_PIPS_PER_SIDE, abs=0.01)


# ---------------------------------------------------------------------------
# El contraste con producción
# ---------------------------------------------------------------------------

def test_production_cost_is_materially_lower_than_the_thesis_assumption():
    """La tesis asume ~2,5× el costo del track de producción de este mismo repo.

    Es el contraste que un tribunal va a pedir, y que sube la constante de spread —declarada
    en §8.4 y **nunca medida**— a amenaza principal a la validez.
    """
    price = 4000.0
    prod = production_cost_per_side_pips(price)
    thesis = 3.48 / 2.0 + COMMISSION_PIPS_PER_SIDE      # spread medio observado del HMM

    assert 0.8 < prod < 1.0, f"produccion deberia rondar 0,9 pips por lado, da {prod:.3f}"
    assert thesis / prod > 2.0, (
        f"la tesis asume {thesis:.2f} pips/lado y produccion {prod:.2f}: "
        "el contraste tiene que ser material para que merezca reportarse"
    )


# ---------------------------------------------------------------------------
# La paradoja del always-flat
# ---------------------------------------------------------------------------

def test_flat_paradox_counts_the_action_distribution():
    sessions = [{"weights": [1.0] * OPERABLE_RETURNS, "n_changes": 1, "date": "d1"},
                {"weights": [0.0] * OPERABLE_RETURNS, "n_changes": 0, "date": "d2"}]
    out = flat_paradox(sessions)
    assert out["fraction_bars_flat"] == pytest.approx(0.5)
    assert out["fully_flat_sessions"] == 1
    assert out["mean_abs_exposure"] == pytest.approx(0.5)


def test_zero_is_actually_in_the_frozen_action_space():
    """Todo el hallazgo depende de esto: la política óptima era ALCANZABLE.

    Si `0.0` no estuviera en `EXPOSURE_LEVELS`, «el agente no encontró always-flat» sería
    trivialmente cierto y no diría nada sobre el agente.
    """
    assert 0.0 in EXPOSURE_LEVELS
    close = synthetic_close(1)
    res = run_session(close, np.zeros(OPERABLE_RETURNS), 6.0)
    assert res.daily_return == pytest.approx(0.0, abs=1e-15)
    assert res.total_cost == pytest.approx(0.0)
