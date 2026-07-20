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


def test_enhance_v2_appends_the_v2_features_in_a_stable_order():
    """Order and membership must be deterministic — that is what parity depends on.

    This used to assert "exactly 2" features, which quietly encoded a BUG as the contract:
    the macro merge read `macro["fecha"]` (the DatetimeIndex, not a column) and looked for
    lowercase names MACRO_DAILY_CLEAN does not use, so the macro block always failed and
    only the 2 regime features survived.

    With the merge fixed there are 4. The macro pair is conditional on the parquet being
    present (the DAGs pass their own project_root), so assert the regime pair strictly and
    the macro pair as an ordered optional suffix — pinning a hard count is what let the
    silent no-op live here in the first place.
    """
    df = _synthetic_df()
    base = ["close", "volatility_5d", "volatility_20d"]
    out, cols = enhance_features_v2(df, base)

    assert cols[: len(base)] == base, "base features must be preserved in order"

    added = cols[len(base):]
    assert added[:2] == ["vol_regime_ratio", "trend_slope_60d"], (
        "the 2 v2.0 regime features must come first, in this order (audit A3-02)"
    )
    assert added[2:] in ([], ["rate_diff_ibr_ust2y", "term_spread"]), (
        f"unexpected macro feature tail: {added[2:]}. Expected either none (no macro "
        "parquet) or the ordered pair — anything else breaks train/backtest/infer parity."
    )
    for col in added:
        assert col in out.columns, f"{col} is in feature_cols but not in the dataframe"


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
