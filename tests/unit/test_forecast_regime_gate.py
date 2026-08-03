import numpy as np

from scripts.pipeline.generate_weekly_forecasts import compute_regime_shift_score


def test_regime_shift_score_stable_distribution_is_small():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 3))
    score = compute_regime_shift_score(x)
    assert score < 1.5


def test_regime_shift_score_detects_large_recent_shift():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(400, 3))
    x[-60:] += 4.0
    score = compute_regime_shift_score(x)
    assert score >= 1.5
