"""Backtest validation that can say no.

The audit of this collection found `evaluate_backtest.py` will return
"Score: 94/100 - Verdict: Deploy" for a purely in-sample curve fit, because
out-of-sample performance is not one of its inputs. This module implements the
measurements that make that impossible:

- `purged_kfold_splits`  - CV that does not leak across overlapping labels
- `probabilistic_sharpe_ratio` - Sharpe as a probability, not a point estimate
- `deflated_sharpe_ratio` - Sharpe haircut for the number of trials you ran
- `pbo_cscv`             - probability your backtest selection is overfit
- `sharpe_ratio_stderr`  - CI, autocorrelation-adjusted

References:
  Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio", J. Portfolio Mgmt
  Bailey, Borwein, Lopez de Prado & Zhu (2016), "The Probability of Backtest
    Overfitting", J. Computational Finance
  Lopez de Prado (2018), "Advances in Financial Machine Learning", ch. 7
  Lo (2002), "The Statistics of Sharpe Ratios", Financial Analysts Journal
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "purged_kfold_splits",
    "sharpe_ratio",
    "sharpe_ratio_stderr",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "pbo_cscv",
    "PBOResult",
    "ValidationVerdict",
    "validate",
]

_EULER_GAMMA = 0.5772156649015329


# --------------------------------------------------------------------------
# Cross-validation
# --------------------------------------------------------------------------

def purged_kfold_splits(
    t1: pd.Series,
    n_splits: int = 5,
    embargo_pct: float = 0.01,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """K-fold splits with purging and embargo (AFML ch. 7).

    Plain K-fold leaks whenever labels overlap in time. If a label spans
    t0->t0+20d and the fold boundary falls inside that window, the training set
    contains information about the test period, and out-of-sample Sharpe comes
    back inflated. Every multi-day-hold strategy has this problem.

    Purging drops training samples whose label window overlaps the test window.
    The embargo additionally drops training samples immediately *after* the test
    window, where serial correlation still carries test information.

    Args:
        t1: label end times, indexed by label start time, sorted ascending.
            For a 20-day holding period this is `starts + 20 days`.
        n_splits: number of folds.
        embargo_pct: fraction of the sample embargoed after each test fold.

    Yields:
        (train_indices, test_indices) as positional integer arrays.
    """
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

        # Purge: keep training labels that finished before the test began.
        before = indices[(t1.values < test_start_time)]

        # Embargo: resume training only after the test's last label ends,
        # plus the embargo buffer.
        after_pos = int(t1.index.searchsorted(test_end_time, side="right"))
        after = indices[min(after_pos + embargo, n):]

        train_idx = np.concatenate([before, after])
        if train_idx.size == 0:
            raise ValueError(
                f"Purging removed the entire training set for a fold. Label "
                f"horizon is too long relative to sample length "
                f"(n={n}, n_splits={n_splits})."
            )
        yield train_idx, test_idx


# --------------------------------------------------------------------------
# Sharpe with error bars
# --------------------------------------------------------------------------

def _clean(returns: Sequence[float] | np.ndarray | pd.Series) -> np.ndarray:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        raise ValueError(f"need at least 3 finite returns, got {r.size}")
    if np.std(r, ddof=1) == 0:
        raise ValueError("returns have zero variance - Sharpe is undefined")
    return r


def sharpe_ratio(
    returns, periods_per_year: int | None = None, rf_per_period: float = 0.0
) -> float:
    """Sharpe ratio. Non-annualised unless `periods_per_year` is given.

    The probability functions below all expect the NON-annualised value.
    """
    r = _clean(returns) - rf_per_period
    sr = r.mean() / r.std(ddof=1)
    if periods_per_year:
        sr *= math.sqrt(periods_per_year)
    return float(sr)


def sharpe_ratio_stderr(returns, adjust_autocorr: bool = True) -> float:
    """Standard error of the non-annualised Sharpe ratio.

    Uses the Lo (2002) / Mertens expansion, which accounts for skew and
    kurtosis - the normal-iid formula understates the error badly for the
    fat-tailed, negatively-skewed returns typical of carry strategies.

    With `adjust_autocorr`, applies a first-order autocorrelation correction.
    Monthly-rebalanced strategies on daily returns are autocorrelated, and
    ignoring it inflates the apparent significance.
    """
    r = _clean(returns)
    n = r.size
    sr = r.mean() / r.std(ddof=1)
    g3 = float(stats.skew(r, bias=False))
    g4 = float(stats.kurtosis(r, fisher=False, bias=False))  # raw, normal = 3

    var = (1.0 - g3 * sr + 0.25 * (g4 - 1.0) * sr**2) / (n - 1)
    var = max(var, 1e-12)
    se = math.sqrt(var)

    if adjust_autocorr and n > 3:
        rho = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        if np.isfinite(rho) and abs(rho) < 0.999:
            # Inflate SE when returns are positively autocorrelated.
            se *= math.sqrt((1.0 + rho) / (1.0 - rho))
    return se


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(true Sharpe > sr_benchmark), given observed returns.

    `sr_benchmark` is non-annualised. PSR > 0.95 is the usual bar for claiming
    a Sharpe is distinguishable from the benchmark.
    """
    r = _clean(returns)
    sr = r.mean() / r.std(ddof=1)
    se = sharpe_ratio_stderr(r, adjust_autocorr=False)
    return float(stats.norm.cdf((sr - sr_benchmark) / se))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum Sharpe from `n_trials` strategies with zero true edge.

    This is the null you must beat. Run 100 backtests on noise and the best one
    looks good - this quantifies how good, so you can subtract it.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if sr_variance < 0:
        raise ValueError("sr_variance must be >= 0")
    if n_trials == 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1.0 - _EULER_GAMMA) * z1 + _EULER_GAMMA * z2))


def deflated_sharpe_ratio(returns, trial_sharpes: Sequence[float]) -> float:
    """Probability the strategy's true Sharpe exceeds what trial-selection alone explains.

    Pass EVERY non-annualised trial Sharpe you computed while searching -
    including the ones you discarded. That count is the whole point: DSR
    deflates by how hard you looked. Reporting only the winner is the
    misconduct this metric exists to detect.

    Returns a probability. Below 0.95, the result is not distinguishable from
    the best of N lucky draws.
    """
    trials = np.asarray(list(trial_sharpes), dtype=float)
    trials = trials[np.isfinite(trials)]
    if trials.size < 2:
        raise ValueError(
            "need >= 2 trial Sharpes to estimate their variance. If you truly "
            "ran one untuned strategy, use probabilistic_sharpe_ratio instead."
        )
    sr0 = expected_max_sharpe(float(np.var(trials, ddof=1)), trials.size)
    return probabilistic_sharpe_ratio(returns, sr_benchmark=sr0)


# --------------------------------------------------------------------------
# Probability of Backtest Overfitting
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PBOResult:
    pbo: float
    n_combinations: int
    n_strategies: int
    logits: np.ndarray

    @property
    def verdict(self) -> str:
        if self.pbo <= 0.10:
            return "LOW - in-sample selection generalises"
        if self.pbo <= 0.25:
            return "MODERATE - some selection luck"
        if self.pbo <= 0.50:
            return "HIGH - selection is mostly luck"
        return "SEVERE - selection is anti-predictive"


def pbo_cscv(returns_matrix, n_blocks: int = 16) -> PBOResult:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    Answers the question that matters when you have tried many variants:
    *when I pick the best strategy in-sample, does it stay above median
    out-of-sample?* PBO is the fraction of splits where it does not.

    A PBO above 0.5 means your selection procedure is worse than random - the
    in-sample winner tends to be an out-of-sample loser.

    Args:
        returns_matrix: (T observations x N strategies). N is the number of
            variants you compared, and must be >= 2.
        n_blocks: even number of time blocks. C(n_blocks, n_blocks/2) splits
            are evaluated, so 16 gives 12,870 - keep it at 16 or below.
    """
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

        # Relative rank of the IS winner among OOS results, in (0, 1).
        rank = float(stats.rankdata(sr_oos)[best]) / (n_strat + 1)
        logits.append(math.log(rank / (1.0 - rank)))

    if not logits:
        raise ValueError("no valid splits - check for degenerate return series")

    arr = np.asarray(logits)
    return PBOResult(
        pbo=float((arr <= 0).mean()),
        n_combinations=arr.size,
        n_strategies=n_strat,
        logits=arr,
    )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

@dataclass
class ValidationVerdict:
    passed: bool
    reasons: list[str]
    metrics: dict[str, float]

    def report(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        lines = [f"Validation: {head}", ""]
        for k, v in self.metrics.items():
            lines.append(f"  {k:.<34} {v: .4f}")
        if self.reasons:
            lines.append("")
            lines.append("  Blocking:" if not self.passed else "  Notes:")
            lines.extend(f"    - {r}" for r in self.reasons)
        return "\n".join(lines)


def validate(
    oos_returns,
    trial_sharpes: Sequence[float],
    returns_matrix=None,
    periods_per_year: int = 252,
    min_dsr: float = 0.95,
    max_pbo: float = 0.25,
) -> ValidationVerdict:
    """Gate a strategy on out-of-sample evidence.

    Unlike the existing `evaluate_backtest.py`, out-of-sample returns and the
    trial count are REQUIRED arguments. There is no code path that produces a
    passing verdict without them.
    """
    reasons: list[str] = []
    r = _clean(oos_returns)

    sr = sharpe_ratio(r)
    sr_ann = sr * math.sqrt(periods_per_year)
    se = sharpe_ratio_stderr(r)
    psr = probabilistic_sharpe_ratio(r, 0.0)

    metrics = {
        "oos_sharpe_annualised": sr_ann,
        "oos_sharpe_per_period": sr,
        "sharpe_stderr": se,
        "sharpe_ci95_low_ann": (sr - 1.96 * se) * math.sqrt(periods_per_year),
        "sharpe_ci95_high_ann": (sr + 1.96 * se) * math.sqrt(periods_per_year),
        "psr_vs_zero": psr,
        "n_observations": float(r.size),
        "n_trials": float(len(list(trial_sharpes))),
    }

    try:
        dsr = deflated_sharpe_ratio(r, trial_sharpes)
        metrics["deflated_sharpe"] = dsr
        if dsr < min_dsr:
            reasons.append(
                f"Deflated Sharpe {dsr:.3f} < {min_dsr}: not distinguishable "
                f"from the best of {len(list(trial_sharpes))} trials."
            )
    except ValueError as exc:
        reasons.append(f"Deflated Sharpe unavailable: {exc}")

    if metrics["sharpe_ci95_low_ann"] <= 0:
        reasons.append(
            f"95% CI for annualised Sharpe includes zero "
            f"({metrics['sharpe_ci95_low_ann']:.2f} to "
            f"{metrics['sharpe_ci95_high_ann']:.2f})."
        )

    if returns_matrix is not None:
        res = pbo_cscv(returns_matrix)
        metrics["pbo"] = res.pbo
        if res.pbo > max_pbo:
            reasons.append(f"PBO {res.pbo:.3f} > {max_pbo} - {res.verdict}.")

    return ValidationVerdict(passed=not reasons, reasons=reasons, metrics=metrics)
