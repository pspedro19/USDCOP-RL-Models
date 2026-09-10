"""
Regression: el HMM portable calcula lo MISMO que hmmlearn.

Contract: CTR-RESEARCH-REGIME-PORTABLE-001 · Date: 2026-08-25

## Qué protege

El brazo RL forward construye la observación de cada sesión viva, y esa observación incluye
los 4 posteriores de régimen. El contenedor de Airflow no tiene `hmmlearn`, así que el
posterior se calcula con una recursión alfa en numpy sobre parámetros exportados.

Si esa recursión divergiera de `hmmlearn`, **el brazo forward correría con posteriores
distintos de los que produjeron el resultado de la tesis**. La comparación RL-vs-LLM seguiría
dando números, seguirían siendo plausibles, y no medirían lo que dicen medir.

Por eso el test no comprueba «devuelve algo razonable» sino igualdad numérica contra la
implementación de referencia, sobre las observaciones reales del carril.

## El caso límite que importa

La recursión alfa **sin normalizar** cae por debajo del mínimo de `float64` en unas decenas de
pasos y devuelve `nan`. Las ventanas aquí son de cientos de sesiones, así que ese fallo sería
sistemático, no anecdótico — y `nan` propagado a una observación es un modelo prediciendo sobre
basura. Hay un test dedicado.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.regime_portable import (DEFAULT_PATH,  # noqa: E402
                                          PortableRegimeModel)


@pytest.fixture(scope="module")
def portable() -> PortableRegimeModel:
    if not DEFAULT_PATH.is_file():
        pytest.skip(f"falta {DEFAULT_PATH.name}: exportalo con export_from_frozen()")
    return PortableRegimeModel.load()


def test_the_frozen_export_exists_and_is_coherent(portable):
    """Los parámetros tienen que ser un HMM válido, no un JSON con forma de HMM."""
    k = portable.k
    assert k >= 2
    assert portable.startprob.shape == (k,)
    assert portable.transmat.shape == (k, k)
    assert portable.means.shape[0] == k
    assert portable.covars.shape[:1] == (k,)
    assert len(portable.labels) == k
    assert len(portable.vol_order) == k
    assert sorted(portable.vol_order) == list(range(k)), "vol_order debe ser una permutación"

    assert portable.startprob.sum() == pytest.approx(1.0, abs=1e-9)
    for row in portable.transmat:
        assert row.sum() == pytest.approx(1.0, abs=1e-9), "las filas de transición suman 1"

    d = portable.means.shape[1]
    assert portable.std_means.shape == (d,)
    assert portable.std_scales.shape == (d,)
    assert np.all(portable.std_scales > 0), "escalar por cero daría inf"


def test_posterior_is_a_probability_distribution(portable):
    rng = np.random.default_rng(0)
    d = portable.means.shape[1]
    for n in (1, 5, 60, 400):
        window = rng.normal(0, 1, (n, d)) * portable.std_scales + portable.std_means
        p = portable.filtered_posterior(window)
        assert p.shape == (portable.k,)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)
        assert np.all(p >= 0.0)
        assert np.all(np.isfinite(p))


def test_long_windows_do_not_underflow(portable):
    """Sin normalizar por paso, `alpha` muere en decenas de pasos. Las ventanas son de cientos.

    Este es el fallo que haría al brazo forward predecir sobre `nan` sin que nada avise.
    """
    rng = np.random.default_rng(1)
    d = portable.means.shape[1]
    window = rng.normal(0, 1, (2000, d)) * portable.std_scales + portable.std_means
    p = portable.filtered_posterior(window)
    assert np.all(np.isfinite(p)), "underflow: la recursión no está renormalizando"
    assert p.sum() == pytest.approx(1.0, abs=1e-9)


def test_extreme_outlier_does_not_produce_nan(portable):
    """Una observación imposible bajo todos los estados no puede devolver `nan` en silencio."""
    d = portable.means.shape[1]
    window = np.full((10, d), 1e9)
    p = portable.filtered_posterior(window)
    assert np.all(np.isfinite(p)) and p.sum() == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# La comparación que da sentido a todo lo demás
# ---------------------------------------------------------------------------

def test_portable_matches_hmmlearn_on_real_sessions(portable):
    """Igualdad numérica contra la implementación de referencia, sobre datos reales."""
    pytest.importorskip("hmmlearn")
    pd = pytest.importorskip("pandas")

    from src.research.dataset import CACHE, load_or_build
    from src.research.evaluation_mask import build_mask
    from src.research.regime_hmm import build_regime_observations

    if not CACHE.is_file():
        pytest.skip("sin cache del dataset; el ajuste del HMM tarda minutos")

    data = load_or_build(verbose=False)
    frozen = data.regime_model
    if isinstance(frozen, dict):
        pytest.skip("la cache trae la ficha del HMM, no el modelo ajustado")

    seed = ROOT / "seeds" / "latest" / "usdcop_m5_ohlcv.parquet"
    if not seed.is_file():
        pytest.skip("falta el seed de 5 minutos")

    obs = build_regime_observations(pd.read_parquet(seed),
                                    valid_sessions=set(build_mask().valid)).dropna()
    arr = obs.to_numpy(dtype=float)

    worst = 0.0
    for end in (60, 200, 500, 900, len(arr) - 1):
        a = frozen.filtered_posterior(arr[: end + 1])
        b = portable.filtered_posterior(arr[: end + 1])
        worst = max(worst, float(np.abs(a - b).max()))

    assert worst < 1e-9, (
        f"el HMM portable diverge de hmmlearn en {worst:.3e}. El brazo forward correría con "
        "posteriores distintos de los que produjeron el resultado de la tesis."
    )
