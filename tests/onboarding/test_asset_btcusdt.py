"""TDD onboarding test-contract for the Bitcoin (BTC/USDT) asset.

Binds the acceptance tests from _onboarding-playbook.md §7 / SPEC-13 to the concrete BTC
artifacts (AssetProfile + ingested seeds). A new asset MUST pass these before it can reach
modeling/paper trading. This is the CRYPTO mirror of test_asset_xauusd.py — the differences
are exactly the 24/7 invariants (the hard break vs COP 5h session and Gold 23h metals):

    A1  profile loads and validates (crypto class, BTC/USDT, chart 'BTCUSDT')
    A2  seed loads, closes in price_range, no NaN, high>=low (5-min + daily)
    A3  measured median bars/day == profile.session.bars_per_day (288 — catches 24/7 mismatch)
    A4  timestamps tz-aware & UTC-native; daily aligned to the UTC 00:00 close instant
    C1  24/7 structural: is_24x7, weekend NOT flat, weekend bars actually present in the seed
    B1  every macro_driver is a BTC driver; no COP-only driver (EMBI/IBR/TPM) leaked

Runs standalone (yaml + pandas only) — no ML stack, no DB.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
_ASSET = "btcusdt"


def _load_ap():
    if "asset_profile" in sys.modules:
        return sys.modules["asset_profile"]
    p = _REPO / "src" / "contracts" / "asset_profile.py"
    spec = importlib.util.spec_from_file_location("asset_profile", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["asset_profile"] = mod
    spec.loader.exec_module(mod)
    return mod


ap = _load_ap()
profile = ap.load_asset_profile(_ASSET)


# --------------------------------------------------------------------------- A1
def test_a1_profile_loads_and_validates():
    assert profile.asset_id == "btcusdt"
    assert profile.symbol == "BTC/USDT"
    assert profile.chart_symbol == "BTCUSDT"  # derived, never "USDCOP"
    assert profile.asset_class == "crypto"
    assert profile.validate() == []
    lo, hi = profile.price_range
    assert lo < hi


def _seed(rel: str) -> pd.DataFrame:
    path = _REPO / rel
    if not path.exists():
        pytest.skip(f"seed not ingested yet: {rel} (run scripts/data/ingest_asset_ohlcv.py --asset btcusdt)")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- A2
@pytest.mark.parametrize("rel", [
    "seeds/latest/btcusdt_m5_ohlcv.parquet",
    "seeds/latest/btcusdt_daily_ohlcv.parquet",
])
def test_a2_seed_integrity_and_range(rel):
    df = _seed(rel)
    assert len(df) > 0
    assert set(["time", "symbol", "open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert df[["open", "high", "low", "close"]].isna().sum().sum() == 0
    assert (df["high"] >= df["low"]).all()
    assert (df["symbol"] == "BTC/USDT").all()
    lo, hi = profile.price_range
    assert df["close"].between(lo, hi).mean() > 0.99  # >99% in range


# --------------------------------------------------------------------------- A3
def test_a3_bars_per_day_matches_measured():
    df = _seed("seeds/latest/btcusdt_m5_ohlcv.parquet")
    from zoneinfo import ZoneInfo
    local_day = df["time"].dt.tz_convert(ZoneInfo(profile.session.timezone)).dt.date
    median = int(df.groupby(local_day).size().median())
    # profile declares 288 (24h * 12). 24/7 => EVERY day (incl. weekends) must be full.
    assert profile.session.bars_per_day == median, (
        f"profile.bars_per_day={profile.session.bars_per_day} != measured median {median}"
    )


# --------------------------------------------------------------------------- A4
def test_a4_timezone_aware_and_daily_utc_close():
    m5 = _seed("seeds/latest/btcusdt_m5_ohlcv.parquet")
    assert m5["time"].dt.tz is not None  # tz-aware
    dly = _seed("seeds/latest/btcusdt_daily_ohlcv.parquet")
    utc_hours = dly["time"].dt.tz_convert("UTC").dt.hour.unique().tolist()
    assert utc_hours == [0], f"daily crypto bars must close at 00:00 UTC, got hours {utc_hours}"


# --------------------------------------------------------------------------- C1 (crypto-only)
def test_c1_is_24x7_and_weekends_present():
    # structural profile invariants
    assert profile.session.is_24x7, "BTC session must be 24/7 (mode 24x7, all 7 days)"
    assert profile.session.weekend_flat is False, "BTC holds over weekends (weekend_flat=false)"
    assert str(profile.session.forced_close).lower() in ("none", "null", ""), \
        "crypto has no forced weekly close"
    # and the seed must actually contain Sat/Sun bars (proves 24/7 ingestion, not a weekday copy)
    df = _seed("seeds/latest/btcusdt_m5_ohlcv.parquet")
    dows = set(df["time"].dt.dayofweek.unique().tolist())
    assert {5, 6}.issubset(dows), f"weekend bars missing from BTC seed (dows present: {sorted(dows)})"


# --------------------------------------------------------------------------- B1
def test_b1_no_cop_only_driver_leaked():
    ids = {d.series_id.lower() for d in profile.macro_drivers}
    cop_only = {"embi_col", "embi", "ibr", "tpm", "colcap", "col10y"}
    leaked = ids & cop_only
    assert not leaked, f"COP-only drivers leaked into BTC profile: {leaked}"
    # BTC's real drivers present: at least one crypto-native + real-yields anchor
    assert any(("funding" in i or "mvrv" in i or "nupl" in i or "etf" in i) for i in ids), \
        "no crypto-native driver (funding/on-chain/ETF) in BTC profile"
    assert any("dfii" in i for i in ids), "missing real-yields (DFII10) anchor"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
