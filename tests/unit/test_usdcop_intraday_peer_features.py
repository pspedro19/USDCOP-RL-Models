from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.usdcop_intraday_peer_features import (
    FEATURES,
    attach_intraday_peer_features,
)


def _hourly_fixture() -> pd.DataFrame:
    rows = []
    for symbol, base in (("USD/MXN", 17.0), ("USD/BRL", 5.0)):
        for day_index, day in enumerate(("2026-01-05", "2026-01-06")):
            for hour in range(9, 19):
                opened = pd.Timestamp(f"{day} {hour:02d}:00:00", tz="UTC")
                close = base + day_index * 0.1 + hour * 0.001
                rows.append({
                    "time": opened,
                    "symbol": symbol,
                    "tf": "1h",
                    "open": close - 0.0002,
                    "high": close + 0.0003,
                    "low": close - 0.0003,
                    "close": close,
                    "source": "twelvedata_1h_backfill",
                    "available_at": opened + pd.Timedelta(minutes=66),
                })
    return pd.DataFrame(rows)


def _prices() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-06"]),
        "close": [4300.0, 4310.0],
    })


def test_intraday_features_use_last_completed_hourly_bar():
    result, features, provenance = attach_intraday_peer_features(
        _prices(), hourly_frame=_hourly_fixture()
    )
    assert features == FEATURES
    assert result.loc[1, "audit_mxn_preopen_bar_open"] == pd.Timestamp(
        "2026-01-06 11:00:00", tz="UTC"
    )
    assert result.loc[1, "audit_mxn_decision_bar_open"] == pd.Timestamp(
        "2026-01-06 17:00:00", tz="UTC"
    )
    assert result.loc[1, FEATURES].notna().all()
    assert provenance["partial_capture_rows_excluded"] == 0
    assert provenance["promotion_eligible"] is False


def test_future_open_hour_cannot_change_same_day_features():
    hourly = _hourly_fixture()
    base, _, _ = attach_intraday_peer_features(
        _prices(), hourly_frame=hourly
    )
    future = hourly["time"].eq(pd.Timestamp("2026-01-06 18:00:00", tz="UTC"))
    mutated = hourly.copy()
    mutated.loc[future, "open"] *= 2
    mutated.loc[future, "high"] *= 2
    mutated.loc[future, "low"] *= 2
    mutated.loc[future, "close"] *= 2
    changed, _, _ = attach_intraday_peer_features(
        _prices(), hourly_frame=mutated
    )
    np.testing.assert_allclose(
        base.loc[1, FEATURES].astype(float),
        changed.loc[1, FEATURES].astype(float),
        rtol=0,
        atol=0,
    )


def test_provably_partial_decision_bar_is_excluded():
    hourly = _hourly_fixture()
    target = (
        hourly["symbol"].eq("USD/MXN")
        & hourly["time"].eq(pd.Timestamp("2026-01-06 17:00:00", tz="UTC"))
    )
    hourly.loc[target, "available_at"] = pd.Timestamp(
        "2026-01-06 17:20:00", tz="UTC"
    )
    result, _, provenance = attach_intraday_peer_features(
        _prices(), hourly_frame=hourly
    )
    assert result.loc[1, "audit_mxn_decision_bar_open"] == pd.Timestamp(
        "2026-01-06 16:00:00", tz="UTC"
    )
    assert provenance["partial_capture_rows_excluded"] == 1
