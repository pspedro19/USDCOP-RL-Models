"""The v2 macro features must actually carry signal, not be silently zero.

Contract: CTR-H5-FEATURES-001

`enhance_v2.py` read `macro["fecha"]` (it is the DatetimeIndex, not a column) and looked for
lowercase column names (`ibr_overnight`, `fedfunds_rate`, `ust10y_close`, `ust2y_close`) that
do not exist — MACRO_DAILY_CLEAN uses the UPPERCASE SSOT names. The KeyError was caught,
printed, and then the fill-NaN block set both features to 0.0.

So `carry_diff` and `term_spread` sat in `feature_cols` as **constant zero** in every training
window since v2 shipped: the contract advertised 2 features the model never received.

A zero-variance feature is invisible in accuracy metrics — it just quietly does nothing — so
only an explicit variance assertion catches it.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
MACRO = ROOT / "data" / "pipeline" / "04_cleaning" / "output" / "MACRO_DAILY_CLEAN.parquet"

# The SSOT column names enhance_v2 depends on.
REQUIRED = [
    "FINC_RATE_IBR_OVERNIGHT_COL_D_IBR",
    "FINC_BOND_YIELD10Y_USA_D_UST10Y",
    "FINC_BOND_YIELD2Y_USA_D_DGS2",
]

pytestmark = pytest.mark.skipif(not MACRO.is_file(), reason="MACRO_DAILY_CLEAN not present")


def test_macro_clean_exposes_the_columns_enhance_v2_needs():
    macro = pd.read_parquet(MACRO)
    missing = [c for c in REQUIRED if c not in macro.columns]
    assert not missing, (
        f"MACRO_DAILY_CLEAN lost SSOT columns {missing}; enhance_v2 will silently "
        f"fall back to zero-filled macro features. Available: {sorted(macro.columns)[:6]}"
    )


def test_fecha_is_the_index_not_a_column():
    """Pins the shape that broke the original code, so a future regen is caught."""
    macro = pd.read_parquet(MACRO)
    assert macro.index.name == "fecha", (
        f"expected `fecha` as the DatetimeIndex name, got {macro.index.name!r}. "
        "enhance_v2 does reset_index() based on this contract."
    )


def test_carry_diff_and_term_spread_have_variance():
    """The actual regression: these must not be constant."""
    import sys

    sys.path.insert(0, str(ROOT))
    from src.forecasting.enhance_v2 import enhance_features_v2  # noqa: PLC0415

    try:
        from src.data.dataset_loader import DatasetLoader  # noqa: PLC0415
    except ImportError:
        pytest.skip("dataset loader unavailable in this environment")

    loader = DatasetLoader()
    df, cols = loader.load_dataset(target_horizon=5)
    df, cols = enhance_features_v2(df, cols)

    for feat in ("rate_diff_ibr_ust2y", "term_spread"):
        assert feat in df.columns, f"{feat} missing entirely"
        series = df[feat]
        assert series.std() > 1e-9, (
            f"{feat} is constant (std={series.std()}). It is listed in feature_cols, so the "
            "model is training on a dead column — this is exactly the bug that shipped."
        )
        assert (series != 0).sum() > 0.5 * len(series), (
            f"{feat} is mostly zeros ({(series == 0).sum()}/{len(series)}), which means the "
            "merge_asof did not match — check the date alignment."
        )
