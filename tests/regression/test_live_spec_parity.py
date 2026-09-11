"""
Regression: el spec vivo es el MISMO que el del batch.

Contract: CTR-RESEARCH-FORWARD-001 · Date: 2026-08-25

## Por qué este test decide si la rama forward vale algo

El brazo RL forward carga una política congelada y le da de comer un `SessionSpec` construido
por `build_live_spec`. Si ese spec difiere del que produjo `build_research_data`, la política
está viendo un espacio de observación distinto del que aprendió — **y sigue emitiendo
decisiones**. Los números saldrían, serían plausibles, y no medirían lo que dicen medir.

## El fallo que ya cazó, y por qué hizo falta comparar varias fechas

La primera versión no aplicaba la **máscara de evaluación** al histórico. El batch calcula las
features sobre la serie enmascarada —solo sesiones válidas, concatenadas—, así que sus ventanas
largas (`rv_78`, `EMA_72`, `zscore_60`) saltan festivos e incompletas.

Comparando **una sola** fecha el delta era ~1e-7 y parecía ruido de `float32`. Comparando seis,
`2024-01-02` salió con **4,27** sobre una feature acotada a ±5: una observación completamente
distinta, justo en la primera sesión del hold-out.

De ahí que este test recorra fechas de los tres bloques y de las dos fronteras, no una cómoda.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tolerancias. `market` y `context` deben ser EXACTOS: los dos salen de la misma aritmetica
# sobre los mismos datos, asi que cualquier diferencia es un pipeline distinto, no ruido.
# `spread_pips` admite 1e-9 porque el posterior lo calculan dos implementaciones distintas
# del HMM (hmmlearn en el batch, la recursion alfa portable en el vivo).
EXACT = 0.0
SPREAD_TOL = 1e-9


@pytest.fixture(scope="module")
def data():
    from src.research.dataset import CACHE, load_or_build

    if not CACHE.is_file():
        pytest.skip("sin cache del dataset; construirla ajusta el HMM y tarda minutos")
    return load_or_build(verbose=False)


@pytest.fixture(scope="module")
def artifacts_exist():
    from src.research.live_spec import SCALER_PATH
    from src.research.regime_portable import DEFAULT_PATH

    for p in (SCALER_PATH, DEFAULT_PATH):
        if not p.is_file():
            pytest.skip(f"falta {p.name}: exportalo antes de correr el carril forward")
    return True


def _sample_dates(data):
    """Fechas de los tres bloques y de las dos fronteras entre ellos.

    Las fronteras son donde el enmascarado muerde: la primera sesión de un bloque arrastra
    ventanas que cruzan el hueco de fin de año.
    """
    picks = []
    for block in (data.holdout, data.selection, data.development):
        if not block:
            continue
        picks.extend([block[0], block[-1]])
        if len(block) > 200:
            picks.append(block[len(block) // 2])
    return picks


def test_live_spec_reproduces_the_batch_spec(data, artifacts_exist):
    """Elemento a elemento, sobre fechas de los tres bloques y sus fronteras."""
    from src.research.live_spec import build_live_spec

    worst_market = worst_ctx = worst_spread = 0.0
    offender = None

    for ref in _sample_dates(data):
        live = build_live_spec(ref.date)

        assert live.date == ref.date
        assert live.market.shape == ref.market.shape
        assert live.context.shape == ref.context.shape

        m = float(np.abs(live.market - ref.market).max())
        c = float(np.abs(live.context - ref.context).max())
        s = abs(live.spread_pips - ref.spread_pips)
        if m > worst_market:
            worst_market, offender = m, ref.date
        worst_ctx = max(worst_ctx, c)
        worst_spread = max(worst_spread, s)

        assert np.array_equal(live.close, ref.close), f"{ref.date}: los cierres difieren"

    assert worst_market <= EXACT, (
        f"las features de mercado difieren (peor {worst_market:.3e} en {offender}). "
        "La politica congelada estaria viendo otro espacio de observacion."
    )
    assert worst_ctx <= EXACT, f"el contexto difiere: {worst_ctx:.3e}"
    assert worst_spread < SPREAD_TOL, f"el spread difiere: {worst_spread:.3e}"


def test_a_masked_out_session_is_refused(data, artifacts_exist):
    """Una fecha excluida por la máscara no puede producir spec.

    El batch no genera spec para festivos ni sesiones incompletas. Si el vivo inventara uno,
    el forward operaría días que el entrenamiento nunca vio, y la serie diaria tendría filas
    que ninguna tabla del hold-out tiene.
    """
    from src.research.evaluation_mask import build_mask
    from src.research.live_spec import build_live_spec

    mask = build_mask()
    excluded = [d for group in mask.excluded.values() for d in group]
    if not excluded:
        pytest.skip("la mascara no excluye ninguna sesion")

    valid = set(mask.valid)
    target = next((d for d in sorted(excluded, reverse=True) if d not in valid), None)
    if target is None:
        pytest.skip("sin fecha excluida utilizable")

    with pytest.raises(ValueError, match="mascara|máscara"):
        build_live_spec(target)


def test_an_incomplete_session_is_refused(artifacts_exist):
    """Menos de 60 barras produce un spec distinto del que vio el entrenamiento."""
    import pandas as pd

    from src.research.dataset import SEED_M5
    from src.research.live_spec import build_live_spec
    from src.research.evaluation_mask import build_mask

    if not SEED_M5.is_file():
        pytest.skip("falta el seed de 5 minutos")

    m5 = pd.read_parquet(SEED_M5)
    target = sorted(build_mask().valid)[-1]
    t = pd.to_datetime(m5["time"])
    # Se amputan barras de la sesion objetivo; el resto del historico queda intacto.
    day = t.dt.date == target
    truncated = pd.concat([m5[~day], m5[day].head(30)])

    with pytest.raises(ValueError, match="barras"):
        build_live_spec(target, m5=truncated)


def test_partial_live_spec_is_prefix_causal(artifacts_exist):
    """Las observaciones disponibles no cambian al agregar barras posteriores."""
    import pandas as pd

    from src.research.dataset import SEED_M5
    from src.research.evaluation_mask import build_mask
    from src.research.live_spec import build_live_spec, build_live_spec_partial

    if not SEED_M5.is_file():
        pytest.skip("falta el seed de 5 minutos")
    target = sorted(build_mask().valid)[-1]
    m5 = pd.read_parquet(SEED_M5)
    day = m5[pd.to_datetime(m5["time"]).dt.date == target]
    assert len(day) == 60
    full = build_live_spec(target, m5=m5)
    for n in (1, 11, 59):
        partial = build_live_spec_partial(target, day.iloc[:n])
        assert partial.bars_received == n
        np.testing.assert_allclose(partial.market, full.market[:n], atol=1e-5)
        np.testing.assert_allclose(partial.context, full.context, atol=1e-5)
        np.testing.assert_array_equal(partial.closes, full.close[:n])
