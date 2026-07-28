"""Modelo de costos de transacción (SDD-006 §2) y break-even (Gate G6 / K3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from kernels import sharpe

__all__ = ["CostModel", "STRESS_SCENARIOS", "break_even_cost"]

# SDD-006 §2: cinco escenarios obligatorios en el reporte (c_roundtrip en pb).
STRESS_SCENARIOS: dict[str, float] = {
    "optimista": 1.0,
    "base": 2.0,
    "retail": 5.0,
    "pesimista": 10.0,
    "estres": 20.0,
}


@dataclass(frozen=True, slots=True)
class CostModel:
    """cost_t = turnover_t · c_roundtrip. Impacto ≈ 0 al AUM base (SDD-006 §2)."""

    cost_bps_roundtrip: float = 2.0

    def apply(self, turnover: pd.Series) -> pd.Series:
        return turnover * (self.cost_bps_roundtrip * 1e-4)


def break_even_cost(returns_gross: pd.Series, turnover: pd.Series) -> float:
    """Costo por unidad de turnover (en pb) que lleva el Sharpe neto a cero.

    G6 exige break_even ≥ 3 × 2 pb = 6 pb (SDD-006 §2.1). Un alfa que muere a 3 pb
    es ruido con un modelo de costos optimista.
    """
    g = returns_gross.to_numpy(dtype=float)
    tvr = turnover.to_numpy(dtype=float)

    def net_sharpe(c_bps: float) -> float:
        return sharpe(g - c_bps * 1e-4 * tvr)

    lo, hi = 0.0, 100.0  # 0 a 100 pb
    if net_sharpe(lo) <= 0:
        return 0.0
    if net_sharpe(hi) > 0:
        return float(hi)   # sobrevive incluso a 100 pb: alfa muy robusto (o irreal)
    return float(brentq(net_sharpe, lo, hi))
