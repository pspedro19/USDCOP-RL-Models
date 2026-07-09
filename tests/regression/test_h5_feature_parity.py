"""
Regression: H5 Smart Simple feature parity (audit A3-02).

Locks in the fix that made live weekly training/inference use the SAME feature
set as the approved +25.63% backtest. Guards:
  1. enhance_features_v2 is a single shared SSOT (src.forecasting.enhance_v2).
  2. It deterministically produces base + 2 regime features (vol_regime_ratio,
     trend_slope_60d) and introduces no NaN in the new columns.
  3. The export script imports the shared function (not a private copy).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.forecasting.enhance_v2 import enhance_features_v2


def _synthetic_df(n=200):
    """Minimal frame with the columns enhance_features_v2 needs."""
    idx = pd.date_range("2021-01-01", periods=n, freq="B")
    rng = np.arange(n, dtype=float)
    close = 4000 + np.cumsum(np.sin(rng / 7.0)) * 5
    return pd.DataFrame(
        {
            "date": idx,
            "close": close,
            "volatility_5d": np.abs(np.sin(rng / 5.0)) * 0.01 + 1e-4,
            "volatility_20d": np.abs(np.cos(rng / 11.0)) * 0.01 + 1e-4,
        }
    )


def test_enhance_v2_adds_exactly_the_two_regime_features():
    df = _synthetic_df()
    base = ["close", "volatility_5d", "volatility_20d"]
    out, cols = enhance_features_v2(df, base)
    assert cols[: len(base)] == base, "base features must be preserved in order"
    assert cols[len(base):] == ["vol_regime_ratio", "trend_slope_60d"], (
        "must append exactly the 2 v2.0 regime features (audit A3-02)"
    )
    assert len(cols) == len(base) + 2


def test_enhance_v2_no_nan_in_new_columns():
    df = _synthetic_df()
    out, cols = enhance_features_v2(df, ["close", "volatility_5d", "volatility_20d"])
    for c in ("vol_regime_ratio", "trend_slope_60d"):
        assert not out[c].isna().any(), f"{c} must be fully filled (no NaN)"


def test_enhance_v2_is_deterministic():
    df = _synthetic_df()
    base = ["close", "volatility_5d", "volatility_20d"]
    out1, c1 = enhance_features_v2(df.copy(), base)
    out2, c2 = enhance_features_v2(df.copy(), base)
    assert c1 == c2
    pd.testing.assert_series_equal(out1["vol_regime_ratio"], out2["vol_regime_ratio"])
    pd.testing.assert_series_equal(out1["trend_slope_60d"], out2["trend_slope_60d"])


def test_export_script_imports_shared_enhance_v2():
    """The export script must import the shared SSOT, not redefine it."""
    src = Path("scripts/pipeline/train_and_export_smart_simple.py").read_text(encoding="utf-8")
    assert "from src.forecasting.enhance_v2 import enhance_features_v2" in src
    assert "def enhance_features_v2(" not in src, (
        "export script must NOT carry a private copy of enhance_features_v2 (audit A3-02)"
    )
