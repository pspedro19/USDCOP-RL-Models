"""Tests for the validation layer.

The behavioural tests matter more than the algebraic ones here: the point of
this module is that it says NO to overfit backtests, so the tests assert it
rejects things the existing evaluate_backtest.py accepts.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from xasset.validation import (  # noqa: E402
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo_cscv,
    probabilistic_sharpe_ratio,
    purged_kfold_splits,
    sharpe_ratio,
    sharpe_ratio_stderr,
    validate,
)

RNG = np.random.default_rng(20260720)


# --------------------------------------------------------------------------
# Purged K-fold
# --------------------------------------------------------------------------

def _t1(n: int, horizon: int) -> pd.Series:
    starts = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(starts + pd.Timedelta(days=horizon), index=starts)


def test_purged_kfold_covers_every_sample_exactly_once_in_test():
    t1 = _t1(200, horizon=5)
    tested = np.concatenate([te for _, te in purged_kfold_splits(t1, n_splits=5)])
    assert np.array_equal(np.sort(tested), np.arange(200))


def test_purging_removes_overlapping_labels():
    """A label that spans the fold boundary must not remain in train."""
    horizon = 10
    t1 = _t1(200, horizon=horizon)
    for train, test in purged_kfold_splits(t1, n_splits=5, embargo_pct=0.0):
        test_start = t1.index[test[0]]
        # No training label may still be open when the test period begins.
        assert (t1.iloc[train].values < test_start).sum() + (
            t1.index[train] > t1.iloc[test].max()
        ).sum() == len(train)


def test_longer_horizon_purges_more():
    short = _t1(300, horizon=1)
    long_ = _t1(300, horizon=30)
    n_short = sum(len(tr) for tr, _ in purged_kfold_splits(short, n_splits=5))
    n_long = sum(len(tr) for tr, _ in purged_kfold_splits(long_, n_splits=5))
    assert n_long < n_short


def test_embargo_shrinks_training_set():
    t1 = _t1(300, horizon=2)
    none = sum(len(tr) for tr, _ in purged_kfold_splits(t1, 5, embargo_pct=0.0))
    some = sum(len(tr) for tr, _ in purged_kfold_splits(t1, 5, embargo_pct=0.05))
    assert some < none


def test_unsorted_index_rejected():
    t1 = _t1(50, 2).iloc[::-1]
    with pytest.raises(ValueError, match="sorted ascending"):
        list(purged_kfold_splits(t1))


# --------------------------------------------------------------------------
# Sharpe statistics
# --------------------------------------------------------------------------

def test_sharpe_annualisation():
    r = RNG.normal(0.001, 0.01, 2000)
    assert sharpe_ratio(r, periods_per_year=252) == pytest.approx(
        sharpe_ratio(r) * math.sqrt(252), rel=1e-12
    )


def test_zero_variance_rejected():
    with pytest.raises(ValueError, match="zero variance"):
        sharpe_ratio(np.ones(100))


def test_stderr_shrinks_with_sample_size():
    small = sharpe_ratio_stderr(RNG.normal(0, 0.01, 100), adjust_autocorr=False)
    large = sharpe_ratio_stderr(RNG.normal(0, 0.01, 10_000), adjust_autocorr=False)
    assert large < small


def test_negative_skew_raises_stderr():
    """Carry-like returns (many small gains, rare large loss) are less certain."""
    n = 4000
    sym = RNG.normal(0.0005, 0.01, n)
    skewed = RNG.normal(0.0015, 0.005, n)
    skewed[RNG.choice(n, 40, replace=False)] -= 0.09  # crash tail
    # Match Sharpe so only the shape differs.
    se_sym = sharpe_ratio_stderr(sym / sym.std() * 0.01, adjust_autocorr=False)
    se_skw = sharpe_ratio_stderr(skewed / skewed.std() * 0.01, adjust_autocorr=False)
    assert se_skw > se_sym


def test_psr_high_for_strong_edge_low_for_noise():
    strong = RNG.normal(0.0015, 0.005, 2000)
    noise = RNG.normal(0.0, 0.01, 2000)
    assert probabilistic_sharpe_ratio(strong) > 0.99
    assert probabilistic_sharpe_ratio(noise) < 0.95


# --------------------------------------------------------------------------
# Deflated Sharpe
# --------------------------------------------------------------------------

def test_expected_max_sharpe_grows_with_trials():
    v = 0.01
    assert expected_max_sharpe(v, 1) == 0.0
    seq = [expected_max_sharpe(v, n) for n in (2, 10, 100, 1000)]
    assert seq == sorted(seq)


def test_dsr_rejects_best_of_many_noise_trials():
    """The core guarantee: mine 200 noise strategies, keep the winner, get rejected."""
    trials = [sharpe_ratio(RNG.normal(0, 0.01, 1000)) for _ in range(200)]
    best = int(np.argmax(trials))
    # Regenerate a series with the winning Sharpe, same seed logic as the draw.
    winner = RNG.normal(0, 0.01, 1000)
    winner = winner - winner.mean() + trials[best] * winner.std(ddof=1)
    dsr = deflated_sharpe_ratio(winner, trials)
    assert dsr < 0.95, f"DSR {dsr:.3f} let a pure-noise winner through"


def test_dsr_below_psr_when_many_trials():
    r = RNG.normal(0.0012, 0.01, 2000)
    trials = list(RNG.normal(0.0, 0.03, 150))
    assert deflated_sharpe_ratio(r, trials) < probabilistic_sharpe_ratio(r)


def test_dsr_requires_multiple_trials():
    with pytest.raises(ValueError, match="need >= 2 trial"):
        deflated_sharpe_ratio(RNG.normal(0, 0.01, 500), [0.1])


# --------------------------------------------------------------------------
# PBO
# --------------------------------------------------------------------------

def test_pbo_near_half_for_pure_noise():
    """With no real edge, the in-sample winner is a coin flip out-of-sample.

    PBO on a SINGLE noise matrix is extremely dispersed - measured across seeds
    it spans roughly 0.01 to 0.82 - so this averages over independent matrices.
    A tight assertion on one draw would be flaky, not rigorous.
    """
    pbos = [
        pbo_cscv(np.random.default_rng(s).normal(0, 0.01, (1200, 12)), n_blocks=10).pbo
        for s in range(12)
    ]
    assert 0.35 < float(np.mean(pbos)) < 0.65
    assert pbo_cscv(RNG.normal(0, 0.01, (1200, 12)), n_blocks=10).n_strategies == 12


def test_pbo_low_when_one_strategy_genuinely_dominates():
    m = RNG.normal(0, 0.01, (1200, 12))
    m[:, 3] += 0.0025  # a persistent, real edge
    res = pbo_cscv(m, n_blocks=10)
    assert res.pbo < 0.10


def test_pbo_rejects_single_strategy():
    with pytest.raises(ValueError, match=">= 2 strategies"):
        pbo_cscv(RNG.normal(0, 0.01, (500, 1)))


def test_pbo_rejects_odd_blocks():
    with pytest.raises(ValueError, match="must be even"):
        pbo_cscv(RNG.normal(0, 0.01, (500, 4)), n_blocks=7)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------

def test_validate_fails_the_curve_fit_scenario():
    """The exact failure mode found in evaluate_backtest.py.

    A great-looking equity curve, selected from 300 trials, must not pass.
    """
    r = RNG.normal(0.0008, 0.008, 900)
    trials = list(RNG.normal(0.0, 0.05, 300))
    v = validate(r, trial_sharpes=trials)
    assert not v.passed
    assert any("Deflated Sharpe" in x for x in v.reasons)


def test_validate_passes_strong_edge_few_trials():
    r = RNG.normal(0.0016, 0.006, 2500)
    trials = list(RNG.normal(0.0, 0.01, 5))
    v = validate(r, trial_sharpes=trials)
    assert v.passed, v.report()


def test_validate_flags_ci_crossing_zero():
    r = RNG.normal(0.00015, 0.02, 400)
    v = validate(r, trial_sharpes=list(RNG.normal(0, 0.01, 3)))
    assert not v.passed
    assert any("includes zero" in x for x in v.reasons)


def test_validate_requires_trials_argument():
    with pytest.raises(TypeError):
        validate(RNG.normal(0.001, 0.01, 500))  # type: ignore[call-arg]


def test_verdict_report_renders():
    v = validate(RNG.normal(0.001, 0.01, 800), list(RNG.normal(0, 0.02, 20)))
    assert "Validation:" in v.report()
    assert "oos_sharpe_annualised" in v.report()
