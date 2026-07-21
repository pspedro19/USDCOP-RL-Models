"""FX series in MACRO_DAILY_CLEAN must be scale-consistent.

Contract: CTR-DQ-MACRO-001

Found 2026-07-21 while running H-COP-XLEAD-01: USDMXN and USDCLP in the CLEAN parquet jump
x10,000 on 2026-01-27 (MXN 17.36 -> 171,335) -- quotes arriving without a decimal separator
after a source change, spliced into the series untouched by the cleaning stage. A single such
row inflates the return std ~45x and poisons every z-score, ratio or return computed across
the splice. The hypothesis verdict survived only because its training windows end before 2026.

The check is on RETURNS, not levels: a >100% single-day move in a major FX pair is not a
market event, it is a decimal bug wearing one.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CLEAN = Path(__file__).resolve().parents[2] / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet"
FX_COLS = ("FXRT_SPOT_USDMXN_MEX_D_USDMXN", "FXRT_SPOT_USDCLP_CHL_D_USDCLP")


# xfail lifted 2026-07-22: root cause was investing.com es-locale (comma-decimal) quotes;
# the source now 403s anyway. Fixed by promoting twelvedata to primary in the SSOT, repairing
# the DB rows (56 MXN + 55 CLP) and the CLEAN parquet from the repaired DB. This is now a
# HARD guard: a new scale splice must fail CI, not be waved through.
def test_clean_fx_series_have_no_scale_splices():
    if not CLEAN.is_file():
        pytest.skip("CLEAN parquet absent")
    m = pd.read_parquet(CLEAN).reset_index()
    dc = "fecha" if "fecha" in m.columns else m.columns[0]
    offenders = {}
    for col in FX_COLS:
        if col not in m.columns:
            continue
        s = m[[dc, col]].dropna().sort_values(dc)
        r = np.log(s[col] / s[col].shift(1)).abs()
        bad = s.loc[r > 0.7, dc]  # |log-ret| > 0.7 == a >100% day: decimal bug, not a market
        if len(bad):
            offenders[col] = [str(b)[:10] for b in bad.tolist()[:3]]
    assert not offenders, f"scale splices in CLEAN fx series: {offenders}"
