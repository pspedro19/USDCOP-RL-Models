"""Regression tests for the OHLCV seed validator (CTR-DQ-OHLCV-001).

These lock in the gate that would have caught the Gold daily calendar bug (weekday day-shift that
piled ~20% of bars onto Sunday). See src/data_quality/ohlcv_validators.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.contracts.asset_profile import (
    AssetProfile,
    DataSourceSpec,
    SessionSpec,
)
from src.data_quality.ohlcv_validators import (
    OHLCVValidationError,
    validate_ohlcv_seed,
)


def _profile(days, tz="UTC", mode="metals", daily_close_tz="America/New_York"):
    return AssetProfile(
        asset_id="test",
        symbol="XAU/USD",
        chart_symbol="XAUUSD",
        display_name="Test",
        asset_class="commodity",
        quote_ccy="USD",
        base_ccy="XAU",
        price_range=(100.0, 10000.0),
        session=SessionSpec(mode=mode, timezone=tz, days=tuple(days),
                            trading_days_per_year=250),
        data_source=DataSourceSpec(provider="twelvedata", provider_symbol="XAU/USD"),
        raw={"session": {"daily_close_tz": daily_close_tz}},
    )


def _daily_frame(dates_utc, close=1000.0):
    t = pd.to_datetime(list(dates_utc)).tz_localize("UTC")
    n = len(t)
    return pd.DataFrame({
        "time": t, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 0.0,
    })


def test_clean_weekday_seed_passes():
    # Mon-Fri bars at 17:00 ET (= 22:00 UTC EST) → all in-session
    days = pd.bdate_range("2024-01-01", "2024-03-01", tz="America/New_York")
    t = (days + pd.Timedelta(hours=17)).tz_convert("UTC")
    df = pd.DataFrame({"time": t, "open": 1000.0, "high": 1001.0, "low": 999.0,
                       "close": 1000.0, "volume": 0.0})
    rep = validate_ohlcv_seed(df, _profile(days=[0, 1, 2, 3, 4]), "daily")
    assert rep.ok, [str(i) for i in rep.errors]


def test_sunday_pileup_fails_weekday_coverage():
    # Simulate the Gold bug: bars land on Sunday (dow=6) for a Mon-Fri asset.
    sundays = pd.date_range("2024-01-07", periods=30, freq="7D")  # all Sundays
    df = _daily_frame([d.strftime("%Y-%m-%d") for d in sundays])
    prof = _profile(days=[0, 1, 2, 3, 4])
    rep = validate_ohlcv_seed(df, prof, "daily")
    assert not rep.ok
    assert any(i.check == "weekday_coverage" for i in rep.errors)
    with pytest.raises(OHLCVValidationError):
        rep.raise_if_failed()


def test_btc_24x7_allows_weekends():
    # BTC session.days = 0..6 → weekend bars are expected, not an error.
    t = pd.date_range("2024-01-01", periods=60, freq="1D", tz="UTC")
    df = pd.DataFrame({"time": t, "open": 40000.0, "high": 40100.0, "low": 39900.0,
                       "close": 40000.0, "volume": 1.0})
    prof = _profile(days=[0, 1, 2, 3, 4, 5, 6], mode="24x7", daily_close_tz=None)
    rep = validate_ohlcv_seed(df, prof, "daily")
    assert rep.ok, [str(i) for i in rep.errors]


def test_duplicate_and_nan_are_errors():
    t = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]).tz_localize("UTC")  # dup
    df = pd.DataFrame({"time": t, "open": [1000, 1000, np.nan], "high": 1001.0,
                       "low": 999.0, "close": 1000.0, "volume": 0.0})
    prof = _profile(days=[0, 1, 2, 3, 4, 5, 6], mode="24x7", daily_close_tz=None)
    rep = validate_ohlcv_seed(df, prof, "daily")
    checks = {i.check for i in rep.errors}
    assert "duplicate_time" in checks
    assert "nan_ohlc" in checks
