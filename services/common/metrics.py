"""
Financial Metrics Calculations
==============================
Shared financial metric calculations for all services.

DRY: Centralizes Sharpe, Sortino, CAGR, VaR calculations.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(
    returns: np.ndarray | list[float],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Array of returns (daily, 5-min, etc.)
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods in a year (252 for daily, 60*5*252 for 5-min)

    Returns:
        Annualized Sharpe ratio
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)

    if std_excess == 0 or np.isnan(std_excess):
        return 0.0

    sharpe = (mean_excess / std_excess) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray | list[float],
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized Sortino ratio.

    Args:
        returns: Array of returns
        target_return: Target return (default: 0)
        periods_per_year: Number of periods in a year

    Returns:
        Annualized Sortino ratio
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (target_return / periods_per_year)
    downside_returns = np.minimum(excess_returns, 0)
    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    mean_excess = np.mean(excess_returns)
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    return float(sortino)


def calculate_cagr(
    prices: np.ndarray | list[float],
    periods_per_year: int = 252
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        prices: Array of prices (first and last used)
        periods_per_year: Number of periods in a year

    Returns:
        CAGR as decimal (0.12 = 12%)
    """
    prices = np.asarray(prices)
    if len(prices) < 2 or prices[0] <= 0 or prices[-1] <= 0:
        return 0.0

    total_return = prices[-1] / prices[0]
    n_periods = len(prices) - 1
    years = n_periods / periods_per_year

    if years <= 0:
        return 0.0

    cagr = (total_return ** (1 / years)) - 1
    return float(cagr)


def calculate_max_drawdown(prices: np.ndarray | list[float]) -> float:
    """
    Calculate maximum drawdown.

    Args:
        prices: Array of prices or equity curve

    Returns:
        Maximum drawdown as positive decimal (0.15 = 15% drawdown)
    """
    prices = np.asarray(prices)
    if len(prices) < 2:
        return 0.0

    peak = np.maximum.accumulate(prices)
    drawdown = (peak - prices) / peak
    max_dd = np.max(drawdown)

    return float(max_dd) if not np.isnan(max_dd) else 0.0


def calculate_calmar_ratio(
    returns: np.ndarray | list[float],
    prices: np.ndarray | list[float] | None = None,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).

    Args:
        returns: Array of returns
        prices: Array of prices (optional, computed from returns if not provided)
        periods_per_year: Number of periods in a year

    Returns:
        Calmar ratio
    """
    returns = np.asarray(returns)

    if prices is None:
        # Compute prices from returns
        prices = np.cumprod(1 + returns)
        prices = np.insert(prices, 0, 1.0)  # Start at 1.0

    prices = np.asarray(prices)

    cagr = calculate_cagr(prices, periods_per_year)
    max_dd = calculate_max_drawdown(prices)

    if max_dd == 0:
        return 0.0

    return float(cagr / max_dd)


def calculate_var(
    returns: np.ndarray | list[float],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Value at Risk (VaR) using historical method.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 0.95 = 95%)

    Returns:
        VaR as positive decimal (loss at risk)
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    percentile = (1 - confidence_level) * 100
    var = -np.percentile(returns, percentile)

    return float(var) if not np.isnan(var) else 0.0


def calculate_expected_shortfall(
    returns: np.ndarray | list[float],
    confidence_level: float = 0.95
) -> float:
    """
    Calculate Expected Shortfall (CVaR).

    Args:
        returns: Array of returns
        confidence_level: Confidence level (default: 0.95)

    Returns:
        Expected Shortfall as positive decimal
    """
    returns = np.asarray(returns)
    if len(returns) < 2:
        return 0.0

    var = calculate_var(returns, confidence_level)
    tail_returns = returns[returns <= -var]

    if len(tail_returns) == 0:
        return var

    es = -np.mean(tail_returns)
    return float(es) if not np.isnan(es) else var


def calculate_win_rate(returns: np.ndarray | list[float]) -> float:
    """
    Calculate win rate (percentage of positive returns).

    Args:
        returns: Array of returns

    Returns:
        Win rate as decimal (0.55 = 55%)
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    wins = np.sum(returns > 0)
    total = len(returns)

    return float(wins / total)


def calculate_profit_factor(returns: np.ndarray | list[float]) -> float:
    """
    Calculate profit factor (gross profit / gross loss).

    Args:
        returns: Array of returns

    Returns:
        Profit factor (>1 is profitable)
    """
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = -np.sum(returns[returns < 0])

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    return float(gross_profit / gross_loss)


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf (no scipy dependency)."""
    import math
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse standard-normal CDF (Acklam's rational approximation). Used for DSR's SR0."""
    import math
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def probabilistic_sharpe_ratio(
    sharpe_per_period: float, n_obs: int, skew: float = 0.0, kurtosis: float = 3.0,
    sr_benchmark: float = 0.0,
) -> float:
    """Probabilistic Sharpe Ratio (Bailey & López de Prado 2012).

    Probability that the true (per-period, non-annualized) Sharpe exceeds ``sr_benchmark``, given the
    estimate's standard error inflated by non-normality (skew/kurtosis). ``kurtosis`` is the full
    (non-excess) kurtosis (normal = 3). Returns a probability in [0, 1].
    """
    if n_obs is None or n_obs < 3:
        return 0.0
    sr = float(sharpe_per_period)
    denom = 1.0 - skew * sr + ((kurtosis - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return 0.0
    import math
    z = (sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom)
    return float(_norm_cdf(z))


def expected_max_sharpe_null(n_trials: int, trials_sharpe_std: float) -> float:
    """Expected maximum per-period Sharpe under the null across ``n_trials`` independent strategies.

    SR0 = σ · [ (1−γ)·Z(1−1/N) + γ·Z(1−1/(N·e)) ]  (Bailey & López de Prado 2014, Deflated Sharpe).
    ``trials_sharpe_std`` = std of the per-period Sharpe estimates across the trials.
    """
    import math
    n = max(int(n_trials), 1)
    if n == 1 or trials_sharpe_std <= 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler–Mascheroni
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return float(trials_sharpe_std * ((1.0 - gamma) * z1 + gamma * z2))


def deflated_sharpe_ratio(
    sharpe_per_period: float, n_obs: int, n_trials: int, trials_sharpe_std: float,
    skew: float = 0.0, kurtosis: float = 3.0,
) -> dict:
    """Deflated Sharpe Ratio: PSR evaluated against the null-expected MAX Sharpe over N trials.

    Corrects the single-test PSR for multiple testing (selection bias from trying ``n_trials``
    strategies). Returns ``{sr0, dsr, significant}`` where ``dsr`` is the probability the strategy's
    true Sharpe beats the best-of-N under the null; ``significant`` = dsr > 0.95.
    """
    sr0 = expected_max_sharpe_null(n_trials, trials_sharpe_std)
    dsr = probabilistic_sharpe_ratio(sharpe_per_period, n_obs, skew, kurtosis, sr_benchmark=sr0)
    return {"sr0": round(sr0, 4), "dsr": round(dsr, 4), "significant": bool(dsr > 0.95)}


def trial_aware_moments(returns: np.ndarray | list[float], sr_benchmark: float = 0.0) -> dict:
    """Per-period Sharpe + higher moments + PSR for one return series.

    The building block for trial-aware (Deflated Sharpe) evaluation: the caller collects
    ``sharpe_per_period`` across N strategies to estimate the trial dispersion, then calls
    :func:`deflated_sharpe_ratio`. ``kurtosis`` is full (non-excess, normal = 3).
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return {"sharpe_per_period": 0.0, "skew": 0.0, "kurtosis": 3.0, "n_obs": int(n), "psr": 0.0}
    mean, std = float(r.mean()), float(r.std(ddof=1))
    sr_pp = mean / std
    # sample skewness / kurtosis (full, non-excess)
    z = (r - mean) / std
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    psr = probabilistic_sharpe_ratio(sr_pp, n, skew, kurt, sr_benchmark)
    return {"sharpe_per_period": round(sr_pp, 6), "skew": round(skew, 4),
            "kurtosis": round(kurt, 4), "n_obs": int(n), "psr": round(psr, 4)}


def calculate_all_metrics(
    returns: np.ndarray | list[float],
    prices: np.ndarray | list[float] | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0
) -> dict:
    """
    Calculate all financial metrics at once.

    Args:
        returns: Array of returns
        prices: Array of prices (optional)
        periods_per_year: Number of periods in a year
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with all metrics
    """
    returns = np.asarray(returns)

    if prices is None and len(returns) > 0:
        prices = np.cumprod(1 + returns)
        prices = np.insert(prices, 0, 1.0)

    return {
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, 0.0, periods_per_year),
        'cagr': calculate_cagr(prices, periods_per_year) if prices is not None else 0.0,
        'max_drawdown': calculate_max_drawdown(prices) if prices is not None else 0.0,
        'calmar_ratio': calculate_calmar_ratio(returns, prices, periods_per_year),
        'var_95': calculate_var(returns, 0.95),
        'expected_shortfall_95': calculate_expected_shortfall(returns, 0.95),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'total_return': float(np.prod(1 + returns) - 1) if len(returns) > 0 else 0.0,
        'volatility': float(np.std(returns, ddof=1) * np.sqrt(periods_per_year)) if len(returns) > 1 else 0.0,
        'n_observations': len(returns),
    }


# ---------------------------------------------------------------------------
# Honest-gate extensions (master plan OLA 2, CTR-QUANT-CONSTITUTION-001 §3):
# B1' paired-exposure baseline + transaction-cost stress. ONE implementation
# consumed by src/gold_rl/backtest.py and src/btc_strategy/backtest.py.
# ---------------------------------------------------------------------------

def _ann_return_dd_calmar(returns: np.ndarray, ann_factor: float) -> dict:
    """Annualized return, max drawdown and Calmar for a daily-return series."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return {"ann_return_pct": 0.0, "max_dd_pct": 0.0, "calmar": 0.0, "sharpe": 0.0}
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    years = len(r) / ann_factor
    ann = float(eq[-1] ** (1.0 / years) - 1.0) if years > 0 and eq[-1] > 0 else 0.0
    sd = r.std(ddof=1)
    sharpe = float(r.mean() / sd * np.sqrt(ann_factor)) if sd > 0 else 0.0
    return {"ann_return_pct": round(ann * 100, 2), "max_dd_pct": round(mdd * 100, 2),
            "calmar": round(ann / abs(mdd), 3) if mdd < 0 else 0.0,
            "sharpe": round(sharpe, 3)}


def paired_exposure_baseline(position: np.ndarray, asset_ret: np.ndarray,
                             ann_factor: float) -> dict:
    """B1' — constant exposure equal to the strategy's mean realized |exposure|.

    Separates genuine timing from simply-less-beta (audit II-1): a vol-targeted sleeve
    whose typical exposure is ~0.44x must beat constant-0.44x, not HODL 1.0x. Costless by
    construction (a constant position has ~zero turnover), which makes the bar HARDER —
    honest in the right direction.
    """
    pos = np.asarray(position, dtype=float)
    ret = np.asarray(asset_ret, dtype=float)
    mean_exp = float(np.nanmean(np.abs(pos)))
    stats = _ann_return_dd_calmar(mean_exp * ret, ann_factor)
    return {"mean_exposure": round(mean_exp, 4), **stats}


def cost_stress(position: np.ndarray, asset_ret: np.ndarray, cost: np.ndarray,
                swap: np.ndarray | None, ann_factor: float) -> dict:
    """Re-price the SAME positions at x1/x2/x3 transaction costs (audit II-4).

    REJECT rule (quant-constitution §3.4): a strategy that dies with costs doubled has no
    real edge — the assumed bps (gold 2bps, COP retail spread) are optimistic.
    """
    pos = np.asarray(position, dtype=float)
    ret = np.asarray(asset_ret, dtype=float)
    c = np.asarray(cost, dtype=float)
    sw = np.asarray(swap, dtype=float) if swap is not None else np.zeros_like(c)
    out: dict = {}
    for mult in (1, 2, 3):
        stats = _ann_return_dd_calmar(pos * ret - mult * c - sw, ann_factor)
        out[f"x{mult}"] = stats
    out["survives_2x"] = bool(out["x2"]["ann_return_pct"] > 0 and out["x2"]["calmar"] > 0)
    out["survives_3x"] = bool(out["x3"]["ann_return_pct"] > 0)
    return out


# ---------------------------------------------------------------------------
# Selection-bias validation (ported 2026-07-20 from the xasset-alpha-engine skill).
#
# `quant-constitution.md` requires trial-aware evidence, but the repo only had the
# Deflated Sharpe half of it. These three close the gap:
#   * purged_kfold_splits - the repo's walk-forward has a gap but no embargo, so serial
#     correlation still carried test information into training.
#   * pbo_cscv - answers "does my in-sample winner stay above median out-of-sample?".
#     There was no implementation at all; DSR deflates a Sharpe, PBO indicts the
#     selection procedure itself.
#   * sharpe_ratio_stderr - autocorrelation-adjusted, so error bars stop being
#     optimistic on overlapping or trending returns.
#
# They live HERE, not in the skill, because a skill is advisory and this is the
# module the constitution names for edge claims.
# ---------------------------------------------------------------------------


def sharpe_ratio_stderr(
    returns: np.ndarray | list[float], adjust_autocorr: bool = True
) -> float:
    """Standard error of a per-period Sharpe estimate.

    With ``adjust_autocorr`` the SE is inflated for first-order serial correlation
    (Lo 2002). Overlapping windows and trend-following returns are positively
    autocorrelated, and ignoring that makes every confidence interval too narrow.
    """
    import math

    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return float("inf")
    sd = r.std(ddof=1)
    if sd <= 0:
        return float("inf")
    sr = r.mean() / sd
    var = (1.0 + 0.5 * sr * sr) / (n - 1)
    se = math.sqrt(var)
    if adjust_autocorr and n > 3:
        rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        if np.isfinite(rho) and abs(rho) < 0.999:
            se *= math.sqrt((1.0 + rho) / (1.0 - rho))
    return float(se)


def purged_kfold_splits(t1, n_splits: int = 5, embargo_pct: float = 0.01):
    """K-fold splits with purging AND embargo (López de Prado, AFML ch. 7).

    Plain K-fold leaks whenever label windows straddle a fold boundary: the training set
    then contains information about the test period and OOS Sharpe comes back inflated.
    Every multi-day-hold strategy in this repo has that exposure.

    Purging drops training samples whose label window overlaps the test window; the embargo
    additionally drops samples immediately AFTER the test window, where serial correlation
    still carries test information.

    Args:
        t1: ``pd.Series`` of label END times, indexed by label start time, sorted ascending.
        n_splits: number of folds (>= 2).
        embargo_pct: fraction of the sample embargoed after each test fold.

    Yields:
        ``(train_indices, test_indices)`` positional integer arrays.
    """
    import pandas as pd

    if not isinstance(t1, pd.Series):
        raise TypeError("t1 must be a pd.Series of label end times")
    if not t1.index.is_monotonic_increasing:
        raise ValueError("t1 index must be sorted ascending")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if not 0.0 <= embargo_pct < 1.0:
        raise ValueError("embargo_pct must be in [0, 1)")

    n = len(t1)
    indices = np.arange(n)
    embargo = int(n * embargo_pct)

    for test_idx in np.array_split(indices, n_splits):
        test_start_time = t1.index[test_idx[0]]
        test_end_time = t1.iloc[test_idx].max()
        before = indices[(t1.values < test_start_time)]
        after_pos = int(t1.index.searchsorted(test_end_time, side="right"))
        after = indices[min(after_pos + embargo, n):]
        train_idx = np.concatenate([before, after])
        if train_idx.size == 0:
            raise ValueError(
                "Purging removed the entire training set for a fold. Label horizon is too "
                f"long relative to sample length (n={n}, n_splits={n_splits})."
            )
        yield train_idx, test_idx


def pbo_cscv(returns_matrix, n_blocks: int = 16) -> dict:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    DSR deflates a Sharpe for how many trials you ran. PBO asks the complementary and
    arguably harsher question: **when I pick the best variant in-sample, does it stay above
    median out-of-sample?** PBO is the fraction of splits where it does not.

    ``pbo > 0.5`` means the selection procedure is worse than random — the in-sample winner
    tends to be an out-of-sample loser. That is a REJECT regardless of Sharpe.

    Args:
        returns_matrix: ``(T observations x N strategies)``, N >= 2.
        n_blocks: even, <= 20. C(n_blocks, n_blocks/2) splits are evaluated.

    Returns:
        ``{pbo, n_combinations, n_strategies}``.
    """
    import math
    from itertools import combinations

    from scipy import stats

    m = np.asarray(returns_matrix, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns_matrix must be 2-D (T x N)")
    t, n_strat = m.shape
    if n_strat < 2:
        raise ValueError("PBO needs >= 2 strategies to rank")
    if n_blocks % 2 != 0:
        raise ValueError("n_blocks must be even")
    if n_blocks > 20:
        raise ValueError(f"n_blocks={n_blocks} gives too many combinations")
    if t < n_blocks * 2:
        raise ValueError(f"need >= {n_blocks * 2} observations for {n_blocks} blocks")

    blocks = np.array_split(np.arange(t), n_blocks)
    logits: list[float] = []
    for combo in combinations(range(n_blocks), n_blocks // 2):
        is_idx = np.concatenate([blocks[i] for i in combo])
        oos_idx = np.concatenate([blocks[i] for i in range(n_blocks) if i not in combo])
        with np.errstate(invalid="ignore", divide="ignore"):
            sd_is = m[is_idx].std(axis=0, ddof=1)
            sd_oos = m[oos_idx].std(axis=0, ddof=1)
            sr_is = np.where(sd_is > 0, m[is_idx].mean(axis=0) / sd_is, -np.inf)
            sr_oos = np.where(sd_oos > 0, m[oos_idx].mean(axis=0) / sd_oos, -np.inf)
        if not np.isfinite(sr_is).any():
            continue
        best = int(np.nanargmax(sr_is))
        rank = float(stats.rankdata(sr_oos)[best]) / (n_strat + 1)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        raise ValueError("no valid splits - check for degenerate return series")
    arr = np.asarray(logits)
    return {
        "pbo": round(float((arr <= 0).mean()), 4),
        "n_combinations": int(arr.size),
        "n_strategies": int(n_strat),
    }
