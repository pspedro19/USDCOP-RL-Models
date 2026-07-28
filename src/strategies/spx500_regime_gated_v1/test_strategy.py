"""Tests runnable de la capa de estrategia (espejo de los invariantes del repo).

    cd strategy && python -m pytest test_strategy.py -q

A diferencia de la suite `sp500.*` (roja hasta que exista el paquete), estos tests
corren HOY porque la capa de estrategia es autocontenida y reusa los kernels reales.
Verifican las mismas propiedades duras: next-open, costo monótono, régimen PIT,
y que el DSR con N=989 no aprueba ruido.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import datagen
from benchmarks import build_benchmark
from costs import CostModel
from engine import BacktestConfig, BacktestEngine
from kernels import deflated_sharpe
from metrics import compute_metrics
from policies import POLICIES, STRATEGY_IDS
from regime import label_regimes


@pytest.fixture(scope="module")
def df():
    return datagen.generate(end="2010-12-31")   # tramo corto para tests rápidos


# ---------------------------------------------------------------- semántica temporal
def test_signal_executes_next_open(df):
    """w_t (info hasta el cierre de t) afecta r_{t+1}, nunca r_t."""
    w = pd.Series(0.0, index=df.index)
    w.iloc[10] = 1.0
    res = BacktestEngine().run(w, df)
    # el peso puesto en el índice 10 sólo puede rendir a partir del 11
    assert res.returns_gross.iloc[10] == 0.0
    assert res.returns_gross.iloc[11] != 0.0


def test_weights_are_lagged_exactly_once(df):
    """Un doble shift cambiaría el Sharpe: el rezago es exactamente uno."""
    w = POLICIES["spx_trend_b2"](df)
    once = BacktestEngine().run(w, df).returns_net
    twice = BacktestEngine().run(w.shift(1).fillna(0.0), df).returns_net
    assert not np.isclose(once.sum(), twice.sum())


# ---------------------------------------------------------------- costos (G6)
def test_zero_turnover_zero_cost(df):
    w = pd.Series(0.7, index=df.index)          # peso constante ⇒ turnover 0
    res = BacktestEngine().run(w, df)
    assert res.cost.iloc[1:].sum() == pytest.approx(0.0, abs=1e-12)


def test_costs_are_monotone_in_turnover():
    cm = CostModel(2.0)
    lo = cm.apply(pd.Series([0.1, 0.1]))
    hi = cm.apply(pd.Series([0.5, 0.5]))
    assert (hi > lo).all()


# ---------------------------------------------------------------- régimen (G5)
def test_regime_labels_are_pit(df):
    """Etiquetar con datos hasta t no puede cambiar al anexar el futuro (no look-ahead)."""
    full = label_regimes(df)
    cut = len(df) // 2
    partial = label_regimes(df.iloc[:cut])
    # las etiquetas del pasado no cambian al ver menos futuro
    common = full.index[:cut]
    pd.testing.assert_frame_equal(full.loc[common], partial.loc[common])


def test_every_day_can_have_multiple_regimes(df):
    reg = label_regimes(df)
    assert set(reg.columns) == {"bull", "bear", "high_vol", "sideways"}
    assert reg.sum(axis=1).max() >= 1


# ---------------------------------------------------------------- políticas
def test_policies_never_short_or_over_lever(df):
    for sid in STRATEGY_IDS:
        w = POLICIES[sid](df)
        assert w.min() >= 0.0, f"{sid} tomó corto"
        assert w.max() <= 1.5 + 1e-9, f"{sid} superó el apalancamiento máximo"


def test_gated_is_bounded_by_trend(df):
    """S3 sólo RECORTA la exposición de B2 (overlay que nunca apalanca sobre la tendencia)."""
    trend = POLICIES["spx_trend_b2"](df)
    gated = POLICIES["spx_regime_gated_v1"](df)
    assert (gated <= trend + 1e-9).all()


# ---------------------------------------------------------------- DSR tiene dientes (G4)
def test_dsr_with_989_trials_rejects_a_sharpe_of_1():
    """R-47: un Sharpe de ~1.0 con 989 trials NO es significativo."""
    dsr = deflated_sharpe(
        sr=1.0, sr_variance=0.25, n_trials=989, t=2520,
        skew=0.0, kurt=3.0, periods_per_year=252,
    )
    assert dsr < 0.95, f"el DSR debería rechazar (dio {dsr:.3f})"


def test_benchmarks_all_present(df):
    for name in ("SPY_TR", "SIXTY_FORTY", "VOL_TARGET_10", "MA200"):
        w = build_benchmark(name, df)
        assert len(w) == len(df)
