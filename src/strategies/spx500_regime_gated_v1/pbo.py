"""Probability of Backtest Overfitting vía CSCV (SDD-007 §3, Gate G4).

Bailey, Borwein, López de Prado y Zhu (2017). "The Probability of Backtest
Overfitting". Journal of Computational Finance 20(4):39-70.

Es el único test que mide EL PROCESO DE SELECCIÓN, no un modelo. Responde:
"si elijo la mejor configuración in-sample, ¿queda por encima de la mediana
out-of-sample con más probabilidad que una moneda?"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["CSCVResult", "Degradation", "cscv", "performance_degradation"]


@dataclass(frozen=True, slots=True)
class CSCVResult:
    pbo: float
    prob_oos_loss: float
    n_combinations: int
    n_partitions: int
    n_configs: int
    sr_is_star: np.ndarray   # Sharpe IS de la config elegida, por combinación
    sr_oos_star: np.ndarray  # su Sharpe OOS
    logits: np.ndarray       # λ_c


@dataclass(frozen=True, slots=True)
class Degradation:
    slope: float
    intercept: float
    r_value: float
    p_value: float


def _sharpe_from_moments(s1: np.ndarray, s2: np.ndarray, n: int) -> np.ndarray:
    """Sharpe por observación a partir de Σx y Σx²  (sin anualizar: el ranking no cambia)."""
    mean = s1 / n
    var = np.maximum(s2 / n - mean**2, 1e-24)
    return mean / np.sqrt(var)


def cscv(returns: pd.DataFrame, n_partitions: int = 16) -> CSCVResult:
    """Combinatorially Symmetric Cross-Validation.

    Parameters
    ----------
    returns : (T x N_configs) matriz de retornos, una columna por configuración.
    n_partitions : S, debe ser par. C(S, S/2) combinaciones.
    """
    if n_partitions % 2 != 0:
        raise ValueError(f"n_partitions debe ser par (recibido {n_partitions})")
    if n_partitions < 4:
        raise ValueError("n_partitions >= 4")

    m = np.ascontiguousarray(returns.to_numpy(dtype=float))
    t, n_configs = m.shape
    if n_configs < 2:
        raise ValueError("Se requieren al menos 2 configuraciones para rankear")

    block = t // n_partitions
    if block < 2:
        raise ValueError("Demasiadas particiones para el largo de la serie")
    m = m[: block * n_partitions]

    # Momentos por partición: (S, N)
    blocks = m.reshape(n_partitions, block, n_configs)
    s1 = blocks.sum(axis=1)
    s2 = (blocks**2).sum(axis=1)
    tot1, tot2 = s1.sum(axis=0), s2.sum(axis=0)

    half = n_partitions // 2
    combos = list(combinations(range(n_partitions), half))
    n_comb = len(combos)

    # Máscara (C, S) -> agregados IS por combinación sin recorrer los datos
    mask = np.zeros((n_comb, n_partitions), dtype=float)
    for i, c in enumerate(combos):
        mask[i, list(c)] = 1.0

    is1, is2 = mask @ s1, mask @ s2            # (C, N)
    oos1, oos2 = tot1 - is1, tot2 - is2

    n_is = half * block          # IS y OOS tienen el mismo largo (CSCV es simétrico)
    sr_is = _sharpe_from_moments(is1, is2, n_is)
    sr_oos = _sharpe_from_moments(oos1, oos2, n_is)

    # Config elegida in-sample, por combinación
    star = np.argmax(sr_is, axis=1)
    rows = np.arange(n_comb)
    sr_is_star = sr_is[rows, star]
    sr_oos_star = sr_oos[rows, star]

    # Rango relativo del elegido dentro de la distribución OOS
    ranks = (sr_oos <= sr_oos_star[:, None]).sum(axis=1)          # 1..N
    omega = ranks / (n_configs + 1.0)
    omega = np.clip(omega, 1e-12, 1 - 1e-12)
    logits = np.log(omega / (1.0 - omega))

    return CSCVResult(
        pbo=float((logits < 0).mean()),
        prob_oos_loss=float((sr_oos_star < 0).mean()),
        n_combinations=n_comb,
        n_partitions=n_partitions,
        n_configs=n_configs,
        sr_is_star=sr_is_star,
        sr_oos_star=sr_oos_star,
        logits=logits,
    )


def performance_degradation(result: CSCVResult) -> Degradation:
    """Regresión SR_OOS ~ SR_IS. Pendiente <= 0 es la firma del overfitting."""
    reg = stats.linregress(result.sr_is_star, result.sr_oos_star)
    return Degradation(
        slope=float(reg.slope),
        intercept=float(reg.intercept),
        r_value=float(reg.rvalue),
        p_value=float(reg.pvalue),
    )


def expected_combinations(n_partitions: int) -> int:
    return math.comb(n_partitions, n_partitions // 2)
