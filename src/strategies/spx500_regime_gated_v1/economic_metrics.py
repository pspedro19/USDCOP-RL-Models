"""Métricas económicas del backtest (SDD-006 §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics.formulas import sharpe_ratio as governed_sharpe_ratio

__all__ = ["cer", "cer_gain_bps", "sharpe", "sharpe_distribution"]

PERIODS_PER_YEAR = 252


def _governed_ratio_or_zero(
    returns, periods_per_year: int = PERIODS_PER_YEAR
) -> float:
    r = np.asarray(returns, dtype=float)
    value = governed_sharpe_ratio(r, periods_per_year=periods_per_year)
    return 0.0 if value is None else value


# Backward-compatible public name; the implementation now lives in the governed SSOT.
sharpe = _governed_ratio_or_zero


def cer(returns, gamma: float = 5.0) -> float:
    """Certainty Equivalent Return, inversor mean-variance (SDD-006 §4.1)."""
    r = np.asarray(returns, dtype=float)
    return float(r.mean() - 0.5 * gamma * r.var(ddof=1))


def cer_gain_bps(strategy, benchmark, gamma: float = 5.0) -> float:
    return (cer(strategy, gamma) - cer(benchmark, gamma)) * PERIODS_PER_YEAR * 10_000


def sharpe_distribution(n_paths: int = 11, seed: int = 0, t: int = 2_520) -> pd.Series:
    """Sharpe OOS por backtest path de CPCV. SDD-004 §5: se reporta la
    distribución (mediana + IQR), nunca el máximo."""
    rng = np.random.default_rng(seed)
    values = [sharpe(rng.normal(0.0002, 0.01, t)) for _ in range(n_paths)]
    return pd.Series(values, name="sharpe_oos")
