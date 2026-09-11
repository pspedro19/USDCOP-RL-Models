"""
Regression: el contraste pareado y la Regla B.

Contract: CTR-RESEARCH-INFERENCE-001 · Date: 2026-08-25

## Qué protege cada bloque

1. **El remuestreo es un remuestreo.** Un bootstrap mal vectorizado que devuelva
   `arange(n)` produce intervalos absurdamente estrechos y todo sale «significativo». Se
   comprueban las propiedades: índices en rango, bloques contiguos, y que un bloque largo
   produzca más contigüidad que uno corto.

2. **El pareo sirve para algo.** Se mide contra el remuestreo independiente: si el IC pareado
   no fuese más estrecho sobre series correlacionadas, el pareo estaría roto.

3. **La Regla B bloquea.** El gate del hold-out se comprueba como comportamiento, no como
   intención — es la única defensa contra abrirlo un martes por la tarde sin darse cuenta.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.research.inference import (WHITE_SPA_OMISSION, bootstrap_sharpe_ci,  # noqa: E402
                                    paired_sharpe_test, sharpe,
                                    stationary_bootstrap_indices)


# ---------------------------------------------------------------------------
# El remuestreo
# ---------------------------------------------------------------------------

def test_indices_stay_in_range_and_have_the_right_length():
    rng = np.random.default_rng(0)
    for n in (10, 100, 584):
        idx = stationary_bootstrap_indices(n, 10, rng)
        assert len(idx) == n
        assert idx.min() >= 0 and idx.max() < n


def test_it_actually_resamples():
    """Si devolviera la serie original, cada IC sería de anchura cero."""
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap_indices(200, 10, rng)
    assert not np.array_equal(idx, np.arange(200)), "no está remuestreando nada"
    assert len(set(idx.tolist())) < 200, "sin repeticiones no es un bootstrap"


def test_longer_blocks_preserve_more_contiguity():
    """La propiedad que define el bootstrap de bloque: preservar dependencia temporal.

    Con L grande hay más pasos donde `idx[t+1] == idx[t]+1`. Si el parámetro no cambiara
    nada, el bootstrap trataría la serie como independiente y los IC saldrían demasiado
    estrechos justo donde importa: sobre retornos autocorrelacionados.
    """
    rng = np.random.default_rng(2)
    n = 2000

    def contiguity(L: int) -> float:
        idx = stationary_bootstrap_indices(n, L, rng)
        return float(np.mean(np.diff(idx) == 1))

    assert contiguity(50) > contiguity(2) + 0.3


def test_every_observation_can_appear():
    """El envolvimiento circular existe para que la cola de la serie no quede infra-muestreada."""
    rng = np.random.default_rng(3)
    n = 50
    seen = set()
    for _ in range(200):
        seen.update(stationary_bootstrap_indices(n, 5, rng).tolist())
    assert seen == set(range(n)), f"nunca aparecen: {sorted(set(range(n)) - seen)}"


# ---------------------------------------------------------------------------
# El contraste pareado
# ---------------------------------------------------------------------------

def test_a_series_against_itself_has_exactly_zero_difference():
    r = np.random.default_rng(4).normal(0.0005, 0.01, 300)
    t = paired_sharpe_test(r, r, "A", "A")
    assert t.diff == 0.0
    assert not t.decisive, "una serie contra sí misma no puede ser 'decidible'"
    assert t.correlation == pytest.approx(1.0)


def test_pairing_narrows_the_interval_on_correlated_series():
    """La razón de ser del pareo, medida."""
    rng = np.random.default_rng(5)
    common = rng.normal(0, 0.01, 400)
    a = common + rng.normal(0.0004, 0.003, 400)
    b = common + rng.normal(0, 0.003, 400)

    paired = paired_sharpe_test(a, b, n_boot=4000)
    width_paired = paired.ci_high - paired.ci_low

    rng2 = np.random.default_rng(6)
    diffs = []
    for _ in range(4000):
        ia = stationary_bootstrap_indices(len(a), 10, rng2)
        ib = stationary_bootstrap_indices(len(b), 10, rng2)
        diffs.append(sharpe(a[ia]) - sharpe(b[ib]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    assert width_paired < (hi - lo) * 0.75, (
        f"el pareo no está estrechando el IC ({width_paired:.3f} vs {hi - lo:.3f}): "
        "probablemente no comparte índices entre las dos series"
    )
    assert paired.correlation > 0.8


def test_misaligned_series_are_rejected():
    """Comparar series de distinta longitud sería comparar días distintos."""
    with pytest.raises(ValueError, match="no alineadas"):
        paired_sharpe_test(np.zeros(10), np.zeros(11))


def test_a_large_real_difference_is_declared_decisive():
    """Contraprueba del test de indecidibilidad: con un efecto grande tiene que decidir."""
    rng = np.random.default_rng(7)
    common = rng.normal(0, 0.008, 500)
    a = common + rng.normal(0.003, 0.002, 500)
    b = common - rng.normal(0.003, 0.002, 500)
    t = paired_sharpe_test(a, b, "fuerte", "flojo", n_boot=4000)
    assert t.decisive and t.diff > 0
    assert "DECIDIBLE" in t.verdict()


def test_the_indecisive_verdict_does_not_say_no_difference():
    """§11.1: un contraste indecidible se reporta como tal, no como 'sin diferencias'.

    Es la distinción que más se pierde al redactar: 'no significativo' se lee como 'iguales',
    cuando lo que dicen los datos es que no alcanzan para distinguirlos.
    """
    rng = np.random.default_rng(8)
    common = rng.normal(0, 0.01, 120)
    t = paired_sharpe_test(common + rng.normal(0, 0.004, 120),
                           common + rng.normal(0, 0.004, 120), n_boot=2000)
    assert not t.decisive
    v = t.verdict()
    assert "INDECIDIBLE" in v and "no bastan" in v


def test_pvalue_and_ci_agree():
    """Un p<0.05 con un IC que incluye el cero sería una contradicción publicable."""
    rng = np.random.default_rng(9)
    for k in range(6):
        r = np.random.default_rng(20 + k)
        common = r.normal(0, 0.01, 300)
        a = common + r.normal(0.0015 * k, 0.003, 300)
        b = common + r.normal(0, 0.003, 300)
        t = paired_sharpe_test(a, b, n_boot=3000, seed=k)
        if t.decisive:
            assert t.p_value < 0.10, f"IC excluye 0 pero p={t.p_value:.3f}"


def test_bootstrap_ci_brackets_the_point_estimate():
    r = np.random.default_rng(10).normal(0.0006, 0.009, 400)
    ci = bootstrap_sharpe_ci(r, n_boot=4000)
    assert ci["ci_low"] < ci["sharpe"] < ci["ci_high"]


def test_zero_variance_series_does_not_explode():
    """`always_flat` es exactamente esto: 584 ceros. Es un baseline real, no un caso raro."""
    assert sharpe(np.zeros(100)) == 0.0
    t = paired_sharpe_test(np.zeros(50), np.zeros(50))
    assert t.diff == 0.0 and np.isfinite(t.ci_low)


def test_pvalue_exposes_finite_bootstrap_resolution():
    rng = np.random.default_rng(44)
    a = rng.normal(0.02, 0.001, 120)
    b = rng.normal(0.0, 0.001, 120)
    t = paired_sharpe_test(a, b, n_boot=200, blocks=(5,))
    assert t.n_exceedances >= 0
    assert t.p_value > 0.0


# ---------------------------------------------------------------------------
# Regla B
# ---------------------------------------------------------------------------

def test_holdout_is_blocked_while_the_preregistration_is_unsigned():
    """Comportamiento, no intención: se ejecuta el script y se mira el código de salida."""
    script = ROOT / "scripts" / "analysis" / "thesis_statistics.py"
    sys.path.insert(0, str(ROOT))
    from scripts.analysis.thesis_statistics import preregistration_is_signed

    signed, status = preregistration_is_signed()
    if signed:
        pytest.skip(f"el pre-registro ya está firmado ({status}); el hold-out está abierto")

    proc = subprocess.run([sys.executable, str(script), "--block", "holdout"],
                          capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    assert proc.returncode == 2, (
        f"la Regla B no bloqueó: exit={proc.returncode}\n{proc.stdout[-800:]}"
    )
    assert "Regla B" in proc.stdout


def test_white_and_spa_omission_is_documented_not_silent():
    """Omitir un contraste sin decirlo se lee como no haberlo pensado."""
    assert "White" in WHITE_SPA_OMISSION and "SPA" in WHITE_SPA_OMISSION
    assert "DOS" in WHITE_SPA_OMISSION or "dos" in WHITE_SPA_OMISSION
    assert len(WHITE_SPA_OMISSION) > 200
