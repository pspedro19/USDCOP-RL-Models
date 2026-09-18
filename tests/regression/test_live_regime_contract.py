"""C045: live builders must not hide a posterior/schema mismatch.

These are isolated contract fixtures, not market experiments or trading results.
No fitting, model exports, API requests, or access to frozen data is performed.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.research import live_spec as live


@pytest.fixture
def inputs(monkeypatch):
    target = pd.Timestamp("2023-01-04").date()
    bars = pd.DataFrame({
        "time": pd.date_range("2023-01-04 08:00", periods=60, freq="5min", tz="America/Bogota"),
        "close": np.full(60, 4000.0),
        "open": np.full(60, 4000.0),
        "high": np.full(60, 4000.0),
        "low": np.full(60, 4000.0),
    })
    observations = pd.DataFrame({"feature": np.zeros(60)},
                                index=pd.bdate_range(end="2023-01-03", periods=60))
    calls = []
    levels = live._levels_for_k

    def features(frame, **kwargs):
        calls.append("features")
        result = pd.DataFrame(0.0, index=range(len(frame)), columns=live.MARKET_FEATURES)
        result["_session"] = target
        return result

    monkeypatch.setattr(live.pd, "read_parquet", lambda *a, **k: bars.copy())
    monkeypatch.setattr(live, "_mask_from_frame", lambda *a, **k: SimpleNamespace(valid=[target]))
    monkeypatch.setattr(live, "build_market_features", features)
    monkeypatch.setattr(live, "build_regime_observations", lambda *a, **k: observations.copy())

    def macro(dates):
        calls.append("macro")
        return pd.DataFrame(0.0, index=pd.to_datetime(dates), columns=live.MACRO_FEATURES)

    def spread_levels(k):
        calls.append("spread")
        return levels(k)

    monkeypatch.setattr(live, "attach_macro_features", macro)
    monkeypatch.setattr(live, "_levels_for_k", spread_levels)
    scaler = live.FrozenScaler(np.zeros(len(live.MARKET_FEATURES)),
                               np.ones(len(live.MARKET_FEATURES)), tuple(live.MARKET_FEATURES))

    def build(route, k, posterior):
        def filtered(prior):
            calls.append("posterior")
            return posterior

        model = SimpleNamespace(k=k, filtered_posterior=filtered)
        if route == "complete":
            return live.build_live_spec(target, m5=bars, scaler=scaler, regime=model)
        return live.build_live_spec_partial(target, bars.iloc[:11], scaler=scaler, regime=model)

    return build, calls


@pytest.mark.parametrize("route", ["complete", "partial"])
@pytest.mark.parametrize("k", [5, np.int64(5), 0, -1, True, False, np.bool_(True),
                              2.5, 4.0, np.float64(4), "3", None])
def test_invalid_width_rejected_before_features_and_filter(inputs, route, k):
    build, calls = inputs
    posterior = np.full(5, 0.2) if k in (5, np.int64(5)) else np.full(4, 0.25)
    with pytest.raises(ValueError, match="HMM|regime"):
        build(route, k, posterior)
    assert calls == [], "width rejection must precede features and posterior computation"


@pytest.mark.parametrize("route", ["complete", "partial"])
@pytest.mark.parametrize("posterior", [
    [], [[0.25] * 4], [0.5, 0.5], [0.2] * 5,
    [np.nan, 0, 0, 1], [np.inf, 0, 0, 1], [-0.1, 0.1, 0, 1],
    [1.1, 0, 0, 0], [0, 0, 0, 0], [0.2] * 4,
    [1 + 2e-9, 0, 0, 0], [0.5, 0.5, 2e-9, 0],
    [True, False, False, False], ["1", "0", "0", "0"], [1j, 0, 0, 1],
    [-1e-12, 0, 0, 1], [0.5, 0.5 - 2e-9, 0, 0], 1.0,
    np.array([1, 0, 0, 0], dtype=object),
])
def test_invalid_posterior_is_rejected_not_truncated_or_normalized(inputs, route, posterior):
    build, calls = inputs
    with pytest.raises(ValueError, match="HMM|regime"):
        build(route, 4, posterior)
    assert "spread" not in calls and "macro" not in calls


@pytest.mark.parametrize("route", ["complete", "partial"])
@pytest.mark.parametrize("posterior", [
    [1.0], [0.25, 0.75], [0.25, 0.5, 0.25], [0.125, 0.25, 0.375, 0.25],
    [0.5, 0.5, 5e-10, 0],
])
def test_valid_posterior_and_spread_remain_exact(inputs, route, posterior):
    build, _ = inputs
    result = build(route, np.int64(len(posterior)), posterior)
    original = np.asarray(posterior, dtype=float)
    expected = np.pad(original, (0, live.N_REGIMES - len(original)))
    np.testing.assert_array_equal(result.context[-live.N_REGIMES:], expected.astype(np.float32))
    assert result.spread_pips == float(np.dot(original, live._levels_for_k(len(original))))


@pytest.mark.parametrize("route", ["complete", "partial"])
def test_shared_readonly_posterior_is_never_mutated(inputs, route):
    build, _ = inputs
    original = np.array([0.5, 0.5, 5e-10, 0.0])
    expected = original.copy()
    original.flags.writeable = False
    build(route, 4, original)
    np.testing.assert_array_equal(original, expected)
    assert original.flags.writeable is False


@pytest.mark.parametrize("route", ["complete", "partial"])
@pytest.mark.parametrize("posterior", [
    [True, 0, 0, 0],
    np.ma.array([1.0, 0, 0, 0], mask=[True, False, False, False]),
])
def test_mixed_boolean_or_masked_probability_is_not_silently_coerced(inputs, route, posterior):
    build, calls = inputs
    with pytest.raises(ValueError, match="HMM|regime"):
        build(route, 4, posterior)
    assert "spread" not in calls and "macro" not in calls
