"""Unit tests for the trial-aware metrics (PSR / Deflated Sharpe) in services/common/metrics.py.

Locks in the Bailey & López de Prado math wired into the gold/btc rule-based runners.
"""
from __future__ import annotations

import numpy as np

from services.common.metrics import (
    deflated_sharpe_ratio,
    expected_max_sharpe_null,
    probabilistic_sharpe_ratio,
    trial_aware_moments,
    _norm_cdf,
    _norm_ppf,
)


def test_norm_inverse_roundtrip():
    assert abs(_norm_ppf(0.975) - 1.959964) < 1e-3
    assert abs(_norm_cdf(1.959964) - 0.975) < 1e-4


def test_psr_monotonic_in_sample_size():
    # same Sharpe, more observations -> more confident it beats 0
    weak = probabilistic_sharpe_ratio(0.05, 100)
    strong = probabilistic_sharpe_ratio(0.05, 5000)
    assert 0.0 <= weak <= strong <= 1.0
    assert strong > 0.99


def test_expected_max_sharpe_grows_with_trials():
    # trying more strategies raises the null-expected best Sharpe (selection bias)
    sr0_3 = expected_max_sharpe_null(3, 0.02)
    sr0_50 = expected_max_sharpe_null(50, 0.02)
    assert 0.0 < sr0_3 < sr0_50
    # single trial => no deflation
    assert expected_max_sharpe_null(1, 0.02) == 0.0


def test_deflated_sharpe_penalizes_multiple_trials():
    # more trials at the same dispersion -> lower DSR for the same strategy
    kw = dict(sharpe_per_period=0.05, n_obs=3000, skew=-0.3, kurtosis=5.0, trials_sharpe_std=0.02)
    d1 = deflated_sharpe_ratio(n_trials=1, **kw)["dsr"]
    d10 = deflated_sharpe_ratio(n_trials=10, **kw)["dsr"]
    assert d1 >= d10
    assert 0.0 <= d10 <= 1.0


def test_trial_aware_moments_shape():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, size=2000)
    m = trial_aware_moments(r)
    assert set(m) == {"sharpe_per_period", "skew", "kurtosis", "n_obs", "psr"}
    assert m["n_obs"] == 2000
    assert 2.0 < m["kurtosis"] < 4.0  # ~normal
