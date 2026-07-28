"""Deflated Sharpe Ratio (SDD-007 §2, Gate G4).

Bailey, D. y López de Prado, M. (2014). "The Deflated Sharpe Ratio: Correcting
for Selection Bias, Backtest Overfitting and Non-Normality".
Journal of Portfolio Management 40(5):94-107.

CONVENCIÓN DE UNIDADES — leer antes de tocar nada
-------------------------------------------------
La fórmula del PSR combina `SR` con `sqrt(T-1)`, donde `T` es el número de
OBSERVACIONES. Por lo tanto `SR` debe estar expresado **por observación**, no
anualizado. Mezclar un Sharpe anualizado con `T` diario infla el estadístico en
un factor `sqrt(periods_per_year)` y produce DSR ≈ 1.0 para cualquier cosa.

Este módulo acepta entradas anualizadas por comodidad (`periods_per_year=252`)
y las de-anualiza internamente:

    sr_pp  = sr_ann / sqrt(ppy)
    var_pp = sr_variance_ann / ppy

`skew` y `kurt` son siempre momentos de los retornos por observación.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = [
    "EULER_MASCHERONI",
    "deflated_sharpe",
    "deflated_sharpe_from_registry",
    "expected_max_sharpe",
    "min_track_record_length",
    "probabilistic_sharpe",
]

EULER_MASCHERONI = 0.5772156649015329


def _deannualize(sr: float, sr_variance: float, ppy: int) -> tuple[float, float]:
    if ppy < 1:
        raise ValueError("periods_per_year debe ser >= 1")
    if ppy == 1:
        return float(sr), float(sr_variance)
    return float(sr) / np.sqrt(ppy), float(sr_variance) / ppy


def expected_max_sharpe(
    sr_variance: float,
    n_trials: int,
    *,
    periods_per_year: int = 1,
) -> float:
    """`SR*`: Sharpe esperado del MEJOR de `n_trials` bajo la nula de cero habilidad.

    Devuelve el valor en las mismas unidades que `sr_variance` (si se pasa una
    varianza anualizada, `SR*` sale anualizado).

    SR* = sqrt(Var(SR)) · [ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e)) ]
    """
    if n_trials < 2:
        raise ValueError("El DSR requiere N >= 2 trials")
    if sr_variance < 0:
        raise ValueError("sr_variance no puede ser negativa")

    a = norm.ppf(1.0 - 1.0 / n_trials)
    b = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_star = np.sqrt(sr_variance) * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b)

    if periods_per_year > 1:  # sr_variance venía anualizada; SR* también
        return float(sr_star)
    return float(sr_star)


def probabilistic_sharpe(
    sr: float,
    sr_star: float,
    t: int,
    skew: float,
    kurt: float,
    *,
    periods_per_year: int = 1,
) -> float:
    """PSR(SR*) = Φ[ (SR - SR*)·sqrt(T-1) / sqrt(1 - γ₃·SR + ((γ₄-1)/4)·SR²) ].

    `kurt` es la kurtosis NO centrada (3.0 = normal).
    """
    if t < 2:
        raise ValueError("Se requieren al menos 2 observaciones")

    ppy = periods_per_year
    sr_pp = sr / np.sqrt(ppy) if ppy > 1 else sr
    sr_star_pp = sr_star / np.sqrt(ppy) if ppy > 1 else sr_star

    variance = 1.0 - skew * sr_pp + ((kurt - 1.0) / 4.0) * sr_pp**2
    if variance <= 0:
        # Skew/kurtosis extremos: el estimador del Sharpe no tiene varianza finita.
        raise ValueError(
            f"Varianza del estimador de SR no positiva ({variance:.4f}): "
            "la distribución de retornos es demasiado patológica para el PSR"
        )

    z = (sr_pp - sr_star_pp) * np.sqrt(t - 1.0) / np.sqrt(variance)
    return float(norm.cdf(z))


def deflated_sharpe(
    sr: float,
    sr_variance: float,
    n_trials: int,
    t: int,
    skew: float,
    kurt: float,
    *,
    periods_per_year: int = 1,
) -> float:
    """DSR = PSR(SR*), con SR* estimado a partir de `n_trials`.

    Gate G4 (SDD-000 §3) exige DSR > 0.95 con N = TrialRegistry.count().
    """
    sr_star = expected_max_sharpe(sr_variance, n_trials, periods_per_year=periods_per_year)
    return probabilistic_sharpe(
        sr, sr_star, t, skew, kurt, periods_per_year=periods_per_year
    )


def min_track_record_length(
    sr: float,
    sr_star: float,
    alpha: float,
    skew: float,
    kurt: float,
    *,
    periods_per_year: int = 1,
) -> float:
    """Observaciones mínimas para afirmar SR > SR* con confianza 1-α.

    MinTRL = 1 + [1 - γ₃·SR + ((γ₄-1)/4)·SR²] · (Z_α / (SR - SR*))²
    """
    ppy = periods_per_year
    sr_pp = sr / np.sqrt(ppy) if ppy > 1 else sr
    sr_star_pp = sr_star / np.sqrt(ppy) if ppy > 1 else sr_star

    if sr_pp <= sr_star_pp:
        raise ValueError(
            f"MinTRL indefinido: SR ({sr}) no supera a SR* ({sr_star}). "
            "No hay nada que afirmar."
        )

    z = norm.ppf(1.0 - alpha)
    variance = 1.0 - skew * sr_pp + ((kurt - 1.0) / 4.0) * sr_pp**2
    return float(1.0 + variance * (z / (sr_pp - sr_star_pp)) ** 2)


def deflated_sharpe_from_registry(
    registry,
    candidate_family: str,
    t: int,
    skew: float,
    kurt: float,
    *,
    periods_per_year: int = 252,
) -> float:
    """DSR del mejor candidato de una familia, con N tomado del registry verificado.

    ADR-009: el DSR NO acepta un `n_trials` arbitrario. Lo lee de la cadena de
    hashes, y solo tras verificarla. Ese es todo el punto.
    """
    registry.verify()  # lanza RegistryIntegrityError si la cadena está rota

    sharpes = np.array(
        [r.sharpe_oos for r in registry.records if r.sharpe_oos is not None],
        dtype=float,
    )
    if sharpes.size < 2:
        raise ValueError("Se requieren al menos 2 trials con Sharpe registrado")

    candidates = [
        r.sharpe_oos
        for r in registry.records
        if r.family == candidate_family and r.sharpe_oos is not None
    ]
    if not candidates:
        raise ValueError(f"Sin trials completados para la familia '{candidate_family}'")

    return deflated_sharpe(
        sr=max(candidates),
        sr_variance=float(sharpes.var(ddof=1)),
        n_trials=registry.count(),  # ⟵ TODOS los trials, no los reportados
        t=t,
        skew=skew,
        kurt=kurt,
        periods_per_year=periods_per_year,
    )
