"""Métricas del backtest (SDD-006 §4) + puentes a DSR/PBO (Gate G4) y régimen (G5)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

if __package__:
    from .kernels import PERIODS_PER_YEAR, cer_gain_bps, cscv, deflated_sharpe, sharpe
else:  # soporte del runner standalone documentado
    from kernels import PERIODS_PER_YEAR, cer_gain_bps, cscv, deflated_sharpe, sharpe

__all__ = ["PerfMetrics", "compute_metrics", "dsr_from_family", "pbo_from_family",
           "regime_robustness"]

PPY = PERIODS_PER_YEAR


@dataclass(frozen=True, slots=True)
class PerfMetrics:
    sharpe: float
    cagr: float
    ann_vol: float
    max_drawdown: float
    calmar: float
    sortino: float
    turnover_ann: float
    skew: float
    kurt: float          # NO centrada (3.0 = normal), la que consume el DSR
    n_obs: int


def _max_drawdown(returns: pd.Series) -> float:
    curve = (1.0 + returns).cumprod()
    dd = curve / curve.cummax() - 1.0
    return float(dd.min())


def compute_metrics(returns_net: pd.Series, turnover: pd.Series) -> PerfMetrics:
    r = returns_net.dropna()
    if r.std(ddof=1) == 0 or len(r) < 2:
        return PerfMetrics(0, 0, 0, 0, 0, 0, 0, 0, 3.0, len(r))

    ann_vol = float(r.std(ddof=1) * np.sqrt(PPY))
    cagr = float((1.0 + r).prod() ** (PPY / len(r)) - 1.0)
    mdd = _max_drawdown(r)
    downside = r[r < 0].std(ddof=1)
    sortino = float(r.mean() / downside * np.sqrt(PPY)) if downside > 0 else 0.0
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0

    return PerfMetrics(
        sharpe=sharpe(r),
        cagr=cagr,
        ann_vol=ann_vol,
        max_drawdown=mdd,
        calmar=calmar,
        sortino=sortino,
        turnover_ann=float(turnover.mean() * PPY),
        skew=float(stats.skew(r)),
        kurt=float(stats.kurtosis(r, fisher=False)),   # non-central
        n_obs=len(r),
    )


def dsr_from_family(candidate: PerfMetrics, family_sharpes: np.ndarray,
                    n_trials: int) -> float:
    """DSR del candidato con Var(SR) tomada de la familia y N del estudio.

    n_trials = conteo total gobernado por el HYPOTHESIS-REGISTRY, no solo los
    reportados ni el presupuesto maximo. Con N grande, SR* sube y un Sharpe de
    1.0 no prueba nada.
    """
    sr_var = float(np.var(family_sharpes, ddof=1)) if family_sharpes.size >= 2 else 0.25
    return deflated_sharpe(
        sr=candidate.sharpe,
        sr_variance=sr_var,
        n_trials=n_trials,
        t=candidate.n_obs,
        skew=candidate.skew,
        kurt=candidate.kurt,
        periods_per_year=PPY,
    )


def pbo_from_family(returns_matrix: pd.DataFrame, n_partitions: int = 16) -> float:
    """PBO vía CSCV sobre la matriz de retornos de la familia de configs (SDD-007 §3)."""
    return float(cscv(returns_matrix, n_partitions=n_partitions).pbo)


def regime_robustness(
    strat_net: pd.Series, bench_net: pd.Series, regimes: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Sharpe de la estrategia vs benchmark dentro de cada régimen (Gate G5).

    G5 pasa si la estrategia bate al benchmark en ≥3 de 4 regímenes.
    """
    out: dict[str, dict[str, float]] = {}
    strat, bench = strat_net.align(bench_net, join="inner")
    reg = regimes.reindex(strat.index).fillna(False)
    for name in reg.columns:
        mask = reg[name].to_numpy(dtype=bool)
        s_strat = strat[mask]
        s_bench = bench[mask]
        out[name] = {
            "n_days": int(mask.sum()),
            "sharpe_strat": sharpe(s_strat) if len(s_strat) > 1 else 0.0,
            "sharpe_bench": sharpe(s_bench) if len(s_bench) > 1 else 0.0,
        }
    return out
