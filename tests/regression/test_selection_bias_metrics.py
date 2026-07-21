"""Selection-bias validation ported into the constitutional SSOT.

Contract: CTR-QUANT-CONSTITUTION-001

`quant-constitution.md` demands trial-aware evidence, but the repo only had the Deflated
Sharpe half. `purged_kfold_splits` (embargo, not just a gap), `pbo_cscv` (does the in-sample
winner survive out-of-sample?) and an autocorrelation-adjusted `sharpe_ratio_stderr` were
ported from the xasset-alpha-engine skill into `services/common/metrics.py` so production
pipelines can use them — a skill is advisory, this module is what the constitution names.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from services.common.metrics import (
    deflated_sharpe_ratio,
    pbo_cscv,
    purged_kfold_splits,
    sharpe_ratio_stderr,
)


# ------------------------------------------------------------------ stderr

def test_stderr_inflates_under_positive_autocorrelation():
    """The whole point: overlapping/trending returns must widen the error bars."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, 600)
    ar = np.zeros(600)
    for i in range(1, 600):
        ar[i] = 0.6 * ar[i - 1] + noise[i]

    adjusted = sharpe_ratio_stderr(ar, adjust_autocorr=True)
    naive = sharpe_ratio_stderr(ar, adjust_autocorr=False)
    assert adjusted > naive, "autocorrelation adjustment must widen, never narrow, the SE"


def test_stderr_degenerate_inputs():
    assert sharpe_ratio_stderr([0.01, 0.01]) == float("inf")      # n < 3
    assert sharpe_ratio_stderr([0.01] * 50) == float("inf")       # zero variance


# ------------------------------------------------- purged k-fold + embargo

def _labels(n: int, horizon: int = 5) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(idx + pd.Timedelta(days=horizon), index=idx)


def test_purged_splits_cover_every_sample_once_in_test():
    t1 = _labels(200)
    test_sets = [set(te.tolist()) for _, te in purged_kfold_splits(t1, n_splits=5)]
    union: set[int] = set().union(*test_sets)
    assert union == set(range(200))
    assert sum(len(s) for s in test_sets) == 200, "test folds must not overlap"


def test_purge_removes_overlapping_labels_from_training():
    """A training label whose window straddles the test start is leakage."""
    t1 = _labels(200, horizon=10)
    for train, test in purged_kfold_splits(t1, n_splits=4, embargo_pct=0.0):
        test_start = t1.index[test[0]]
        assert (t1.values[train] < test_start).sum() == (t1.values[train] < test_start).sum()
        # No training label may still be open when the test window opens...
        before = train[train < test[0]]
        assert all(t1.values[i] < test_start for i in before), "purge failed"


def test_embargo_drops_samples_after_the_test_window():
    t1 = _labels(300)
    no_emb = [len(tr) for tr, _ in purged_kfold_splits(t1, n_splits=5, embargo_pct=0.0)]
    with_emb = [len(tr) for tr, _ in purged_kfold_splits(t1, n_splits=5, embargo_pct=0.10)]
    assert sum(with_emb) < sum(no_emb), "embargo must remove post-test training samples"


def test_purged_splits_reject_bad_arguments():
    t1 = _labels(50)
    with pytest.raises(TypeError):
        list(purged_kfold_splits([1, 2, 3]))
    with pytest.raises(ValueError):
        list(purged_kfold_splits(t1, n_splits=1))
    with pytest.raises(ValueError):
        list(purged_kfold_splits(t1, embargo_pct=1.0))


# ------------------------------------------------------------------- PBO

def test_pbo_is_high_when_strategies_are_pure_noise():
    """Selecting the best of N noise series should NOT survive out-of-sample."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.01, (400, 12))
    result = pbo_cscv(noise, n_blocks=10)
    assert 0.0 <= result["pbo"] <= 1.0
    assert result["n_strategies"] == 12
    assert result["pbo"] > 0.3, (
        f"pure noise produced pbo={result['pbo']}, implying the in-sample winner "
        "reliably survives OOS — that would mean the estimator is broken"
    )


def test_pbo_is_low_when_one_strategy_has_real_edge():
    rng = np.random.default_rng(11)
    m = rng.normal(0, 0.01, (400, 8))
    m[:, 3] += 0.004  # a genuine, persistent edge
    assert pbo_cscv(m, n_blocks=10)["pbo"] < 0.5


def test_pbo_rejects_bad_arguments():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(100, 1)))          # needs >= 2 strategies
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(100, 4)), n_blocks=7)   # must be even
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(10, 4)), n_blocks=16)   # too few observations


# --------------------------------------------- the skill delegates, not duplicates

def test_skill_wrapper_returns_the_identical_ssot_result():
    """xasset-alpha-engine must not be a second implementation of the release gate."""
    import sys
    from pathlib import Path

    from scipy import stats

    root = Path(__file__).resolve().parents[2]
    skill = root / ".claude" / "skills" / "xasset-alpha-engine" / "scripts"
    if not skill.is_dir():
        pytest.skip("xasset-alpha-engine not promoted")
    sys.path.insert(0, str(skill))
    from xasset import validation as v  # noqa: PLC0415

    rng = np.random.default_rng(42)
    r = rng.normal(0.001, 0.01, 500)
    trials = list(rng.normal(0.05, 0.03, 20))

    assert v.deflated_sharpe_ratio(r, trials) == deflated_sharpe_ratio(
        sharpe_per_period=float(r.mean() / r.std(ddof=1)),
        n_obs=len(r),
        n_trials=len(trials),
        trials_sharpe_std=float(np.std(trials, ddof=1)),
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
    )


def test_vote1_includes_dsr_gate():
    """STAT-001: Vote 1 must be unable to PROMOTE without a trial-aware DSR.

    The original five gates accepted a candidate at -14% return and Sharpe 0.01 -- Vote 1
    filtered nothing. Gate 6 implements the constitution's mandate, with the trial count read
    from the registry front-matter (never a literal) and the sigma-unit ambiguity resolved in
    the CONSERVATIVE direction: a gate that passes on the favorable reading of an ambiguity is
    a gate someone will eventually argue past.
    """
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "scripts" / "pipeline"
           / "train_and_export_smart_simple.py").read_text(encoding="utf-8", errors="replace")
    assert "_dsr_gate(m)" in src, "gate 6 (deflated_sharpe) missing from export_approval_state"
    assert "n_trials_total" in src, "the DSR gate must read its trial count from the registry"
    assert "min(" in src.split("def _dsr_gate")[1].split("def export_approval_state")[0], (
        "the DSR gate must take the minimum over both sigma-unit readings"
    )


def test_approval_backtest_model_set_matches_live_serving():
    """The Vote-2 numbers must come from the model set that actually trades.

    An independent agent reported use_xgboost: true and a live(Ridge+BR) vs
    approved(Ridge+BR+XGB) divergence. Verification showed the config was ALREADY reconciled
    (use_xgboost: false, with a comment saying why) -- the agent misread. But the check
    surfaced two real hazards, both fixed and pinned here: the CODE defaults were True (a
    missing key would silently resurrect XGBoost and diverge approval from serving), and
    CLAUDE.md still advertised Ridge+BR+XGBoost as the production track.
    """
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parents[2]

    exec_cfg = yaml.safe_load((root / "config/execution/smart_simple_v1.yaml")
                              .read_text(encoding="utf-8"))
    assert exec_cfg["models"]["use_xgboost"] is False, (
        "approval export includes XGBoost while live serving is Ridge+BR -- Vote 2 would "
        "approve numbers from an ensemble that does not trade"
    )

    fss = yaml.safe_load((root / "config/forecasting_ssot.yaml").read_text(encoding="utf-8"))
    h5_models = fss["tracks"]["h5"]["models"]
    assert "xgboost" not in h5_models, (
        f"live H5 model list {h5_models} gained xgboost without promoting it through its own "
        "immutable bundle"
    )
    assert set(h5_models) == {"ridge", "bayesian_ridge"}, (
        f"live H5 serving set changed to {h5_models}; the approval export and CLAUDE.md must "
        "move in the same commit or Vote 2 approves an ensemble that does not trade"
    )

    src = (root / "scripts/pipeline/train_and_export_smart_simple.py").read_text(encoding="utf-8")
    assert 'get("use_xgboost", True)' not in src, (
        "a True default means a missing config key silently resurrects the divergence"
    )
