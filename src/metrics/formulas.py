"""Dependency-light governed metric formulas.

This module intentionally imports neither service bootstrap code nor databases,
so the metric engine remains usable by Airflow, APIs, CLI backfills and CI.
"""

from __future__ import annotations

import math

import numpy as np


def sharpe_ratio(returns: np.ndarray, periods_per_year: int) -> float | None:
    if len(returns) < 2:
        return None
    standard_deviation = float(np.std(returns, ddof=1))
    if standard_deviation <= np.finfo(float).eps:
        return None
    return float(np.mean(returns) / standard_deviation * math.sqrt(periods_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) < 2:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float(np.max((peak - equity) / peak))


def calmar_ratio(returns: np.ndarray, periods_per_year: int) -> float | None:
    if len(returns) < 2:
        return None
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    drawdown = max_drawdown(equity)
    if drawdown == 0 or equity[-1] <= 0:
        return None
    years = len(returns) / periods_per_year
    cagr = float(equity[-1] ** (1.0 / years) - 1.0)
    return cagr / drawdown


def _norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _norm_ppf(probability: float) -> float:
    """Acklam inverse-normal approximation used by the DSR definition."""
    if probability <= 0:
        return -math.inf
    if probability >= 1:
        return math.inf
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.3577518672690,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    )
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (
        0.007784695709041462,
        0.3224671290700398,
        2.445134137142996,
        3.754408661907416,
    )
    low, high = 0.02425, 0.97575
    if probability < low:
        q = math.sqrt(-2 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if probability > high:
        q = math.sqrt(-2 * math.log(1 - probability))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = probability - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def expected_max_sharpe_null(n_trials: int, trials_sharpe_std: float) -> float:
    if n_trials <= 1 or trials_sharpe_std <= 0:
        return 0.0
    gamma = 0.5772156649015329
    return float(
        trials_sharpe_std
        * (
            (1 - gamma) * _norm_ppf(1 - 1 / n_trials)
            + gamma * _norm_ppf(1 - 1 / (n_trials * math.e))
        )
    )


def deflated_sharpe_probability(
    *,
    sharpe_per_period: float,
    n_obs: int,
    n_trials: int,
    trials_sharpe_std: float,
    skew: float,
    kurtosis: float,
) -> float:
    benchmark = expected_max_sharpe_null(n_trials, trials_sharpe_std)
    denominator = (
        1.0
        - skew * sharpe_per_period
        + ((kurtosis - 1.0) / 4.0) * sharpe_per_period**2
    )
    if n_obs < 3 or denominator <= 0:
        return 0.0
    z_score = (
        (sharpe_per_period - benchmark)
        * math.sqrt(n_obs - 1)
        / math.sqrt(denominator)
    )
    return float(_norm_cdf(z_score))
