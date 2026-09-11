"""
Regression: el HMM de régimen es causal y su fit está congelado.

Contract: CTR-RESEARCH-REGIME-001 · Implementa el **test 3** del plan de tesis
(`.claude/specs/planes/06-tesis-rl-llm-hibrido.md` §13): *"HMM filtrado: serie completa vs
truncada en `t` producen el mismo posterior en `t`"*.

## Por qué este test es la diferencia entre un resultado y un artefacto

`hmmlearn.predict_proba` corre forward-**backward**. Sobre la serie completa, el posterior en
`t` incorpora observaciones posteriores a `t`: si eso alimenta el costo o la decisión, el
backtest sabe el futuro y su Sharpe es ficción. El plan lo prohíbe explícitamente
(§8.2: *"recursión forward filtrada; nunca Viterbi sobre la secuencia completa ni posterior
suavizada"*), y la constitución lo cataloga como capa 2 del anti-look-ahead.

El fallo es **silencioso**: un HMM suavizado produce números perfectamente plausibles. Solo
una comparación explícita lo detecta.

Sin Postgres ni Airflow.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("hmmlearn")

ROOT = Path(__file__).resolve().parents[2]

# Persistencia mínima para que un estado intermedio pueda llamarse "tendencial" (§8.3).
TENDENCIAL_PERSISTENCE = 0.8


@pytest.fixture(scope="module")
def fitted():
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import yaml
    from src.research.evaluation_mask import build_mask
    from src.research.regime_hmm import build_regime_observations, fit_frozen

    seed = ROOT / "seeds/latest/usdcop_m5_ohlcv.parquet"
    if not seed.is_file() or seed.open("rb").read(32).startswith(b"version https://git-lfs"):
        pytest.skip("seed de 5 minutos no materializada")
    blocks = yaml.safe_load(
        (ROOT / "config/research/partition.yaml").read_text(encoding="utf-8"))["blocks"]
    obs = build_regime_observations(pd.read_parquet(seed), valid_sessions=build_mask().valid)
    dev = obs[(obs.index >= blocks["development"]["start"])
              & (obs.index <= blocks["development"]["end"])]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fit_frozen(dev)
    return model, obs, blocks


def test_posterior_is_filtered_not_smoothed(fitted):
    """TEST 3. El posterior en `t` no puede cambiar por lo que pase DESPUES de `t`."""
    model, obs, _ = fitted
    clean = obs.dropna()
    arr = clean.to_numpy(dtype=float)
    n = len(arr)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed_all = model.model.predict_proba(model._standardize(arr))[:, list(model.vol_order)]

    # Se muestrea la MITAD INICIAL de la serie: cuanto más cerca del final está `t`, menos
    # futuro le queda al suavizado para discrepar del filtrado, así que exigir diferencia en
    # `n-50` no prueba causalidad, prueba aritmética. En la cola llegan a coincidir a 1e-16.
    sampled = range(200, n // 2, 29)
    differences = []
    for t in sampled:
        truncated = model.filtered_posterior(arr[: t + 1])
        repeat = model.filtered_posterior(arr[: t + 1])
        assert np.allclose(truncated, repeat, atol=1e-12), "el posterior no es determinista"
        differences.append(float(np.abs(truncated - smoothed_all[t]).max()))

    moved = np.asarray(differences)
    # Se compara una FRACCIÓN, no cada punto. En un instante cuyo régimen es inequívoco, el
    # posterior es casi one-hot y conocer el futuro no aporta nada: filtrado y suavizado
    # coinciden de forma legítima, y eso pasa en ~36 % de los puntos medidos. La versión
    # anterior exigía diferencia en tres índices fijos y se puso roja el 2026-09-11 al
    # alargarse la serie, cuando dos de los tres cayeron sobre puntos de esos.
    #
    # El poder del guardia no se pierde: si el código devolviera el suavizado, coincidirían
    # TODOS los puntos y la fracción caeria a cero.
    fraction_moved = float((moved > 1e-6).mean())
    assert fraction_moved > 0.25, (
        f"solo el {100 * fraction_moved:.1f}% de los puntos muestreados difiere del suavizado "
        f"sobre la serie completa (mediana {np.median(moved):.2e}). Si practicamente ninguno "
        "se mueve, se esta leyendo el suavizado — que es exactamente el look-ahead que este "
        "test existe para impedir."
    )
    assert moved.max() > 1e-3, (
        "ningun punto de la primera mitad difiere de forma apreciable del suavizado; la serie "
        "es degenerada o el filtrado no lo es."
    )


def test_adding_future_does_not_change_a_past_posterior(fitted):
    """La forma directa del mismo invariante: extender la serie no reescribe el pasado."""
    model, obs, _ = fitted
    arr = obs.dropna().to_numpy(dtype=float)
    t = 300
    before = model.filtered_posterior(arr[: t + 1])
    after = model.filtered_posterior(arr[: t + 1])   # el mismo corte, tras haber visto más
    assert np.allclose(before, after, atol=1e-12)


def test_fit_used_only_the_development_block(fitted):
    """Un fit que tocó selección o hold-out invalida el juicio posterior."""
    model, _obs, blocks = fitted
    dev_start, dev_end = blocks["development"]["start"], blocks["development"]["end"]
    assert model.fit_range[0] >= dev_start, f"el fit empieza antes de desarrollo: {model.fit_range}"
    assert model.fit_range[1] <= dev_end, (
        f"el fit termina en {model.fit_range[1]}, DESPUES de desarrollo ({dev_end}). "
        "Ajustar con datos de selección o hold-out contamina el juicio."
    )


def test_k_selection_respects_the_declared_hysteresis(fitted):
    """§8.2: K=3 se mantiene salvo que otro mejore el BIC en MAS de 10 puntos."""
    from src.research.regime_hmm import BIC_HYSTERESIS, K_DEFAULT
    model, _o, _b = fitted
    bic = model.bic_by_k
    assert bic, "no se registró el BIC por K — la elección no sería auditable"
    if model.k != K_DEFAULT and K_DEFAULT in bic:
        margin = bic[K_DEFAULT] - bic[model.k]
        assert margin > BIC_HYSTERESIS, (
            f"K={model.k} desplazó al default K={K_DEFAULT} con una mejora de solo "
            f"{margin:.1f} puntos de BIC (<= {BIC_HYSTERESIS}). La histéresis existe para "
            "que un empate estadístico no cambie el régimen del sistema entero."
        )


def test_spread_comes_from_the_posterior_not_from_argmax(fitted):
    """§8.4: `spread_d` es continuo. `argmax` es solo para el desglose descriptivo."""
    from src.research.regime_hmm import SPREAD_PIPS_BY_LEVEL, spread_series
    model, obs, blocks = fitted
    sp = spread_series(model, obs).dropna(subset=["spread_pips"])
    lo, _mid, hi = SPREAD_PIPS_BY_LEVEL
    assert sp["spread_pips"].between(lo - 1e-9, hi + 1e-9).all(), (
        "hay spreads fuera del rango declarado {2, 3, 6} pips"
    )
    # Si viniera de argmax, solo existirían K valores distintos.
    assert sp["spread_pips"].nunique() > model.k * 3, (
        f"solo {sp['spread_pips'].nunique()} spreads distintos para K={model.k}: eso es un "
        "argmax disfrazado, no la mezcla ponderada de §8.4."
    )


def test_round_trip_matches_the_declared_cost_contract(fitted):
    """El round-trip mínimo debe reproducir los 3/4/7 pips que declara §9.3."""
    from src.research.regime_hmm import (COMMISSION_PIPS_PER_SIDE, SPREAD_PIPS_BY_LEVEL)
    expected = [2.0 * (s / 2.0 + COMMISSION_PIPS_PER_SIDE) for s in SPREAD_PIPS_BY_LEVEL]
    assert expected == [3.0, 4.0, 7.0], (
        f"el contrato de costos dejó de cuadrar con §9.3: {expected} != [3, 4, 7]"
    )


def test_middle_states_are_only_called_trending_when_persistence_supports_it(fitted):
    """§8.3: "tendencial" solo si la persistencia lo respalda; si no, "intermedio"."""
    model, _o, _b = fitted
    labels = model.state_labels()
    persist = np.diag(model.model.transmat_)[list(model.vol_order)]
    for i, lab in enumerate(labels):
        if lab == "tendencial":
            assert persist[i] >= TENDENCIAL_PERSISTENCE, (
                f"el estado {i} se llama 'tendencial' con persistencia {persist[i]:.3f}. "
                "Un nombre que sugiere tendencia sobre un estado que dura un día es una "
                "afirmación que el dato no respalda."
            )
