"""TDD onboarding test-contract for the Gold (XAU/USD) asset.

Binds the acceptance tests from sdd-multi-asset-onboarding.md §7 / SPEC-12 to the concrete
Gold artifacts produced by the data layer (AssetProfile + ingested seeds). A new asset MUST
pass these before it can reach modeling/paper trading.

    A1  profile loads and validates
    A2  seed loads, closes in price_range, no NaN, high>=low (5-min + daily)
    A3  measured median bars/day == profile.session.bars_per_day (catches 24/7 vs weekday mismatch)
    A4  timestamps tz-aware; daily aligned to the NY-close instant (DST-correct)
    B1  every macro_driver is a Gold driver; no COP-only driver (EMBI/IBR/TPM) leaked

Runs standalone (yaml + pandas only) — no ML stack, no DB.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
_ASSET = "xauusd"


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
    assert profile.asset_id == "xauusd"
    assert profile.symbol == "XAU/USD"
    assert profile.chart_symbol == "XAUUSD"  # derived, never "USDCOP"
    assert profile.asset_class == "commodity"
    assert profile.validate() == []
    lo, hi = profile.price_range
    assert lo < hi


def _seed(rel: str) -> pd.DataFrame:
    path = _REPO / rel
    if not path.exists():
        pytest.skip(f"seed not ingested yet: {rel} (run scripts/ingest_asset_ohlcv.py --asset xauusd)")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- A2
@pytest.mark.parametrize("rel", [
    "seeds/latest/xauusd_m5_ohlcv.parquet",
    "seeds/latest/xauusd_daily_ohlcv.parquet",
])
def test_a2_seed_integrity_and_range(rel):
    df = _seed(rel)
    assert len(df) > 0
    assert set(["time", "symbol", "open", "high", "low", "close", "volume"]).issubset(df.columns)
    assert df[["open", "high", "low", "close"]].isna().sum().sum() == 0
    assert (df["high"] >= df["low"]).all()
    assert (df["symbol"] == "XAU/USD").all()
    lo, hi = profile.price_range
    assert df["close"].between(lo, hi).mean() > 0.99  # >99% in range


# --------------------------------------------------------------------------- A3
def test_a3_bars_per_day_matches_measured():
    df = _seed("seeds/latest/xauusd_m5_ohlcv.parquet")
    from zoneinfo import ZoneInfo
    local_day = df["time"].dt.tz_convert(ZoneInfo(profile.session.timezone)).dt.date
    median = int(df.groupby(local_day).size().median())
    # profile declares 288 (24h metals); measured must match (guards 24/7 vs weekday assumptions)
    assert profile.session.bars_per_day == median, (
        f"profile.bars_per_day={profile.session.bars_per_day} != measured median {median}"
    )


# --------------------------------------------------------------------------- A4
def test_a4_timezone_aware_and_daily_nyclose():
    m5 = _seed("seeds/latest/xauusd_m5_ohlcv.parquet")
    assert m5["time"].dt.tz is not None  # tz-aware
    dly = _seed("seeds/latest/xauusd_daily_ohlcv.parquet")
    ny_hours = dly["time"].dt.tz_convert("America/New_York").dt.hour.unique().tolist()
    assert ny_hours == [17], f"daily bars must close at 17:00 ET, got hours {ny_hours}"


# --------------------------------------------------------------------------- B1
def test_b1_no_cop_only_driver_leaked():
    ids = {d.series_id.lower() for d in profile.macro_drivers}
    cop_only = {"embi_col", "embi", "ibr", "tpm", "wti", "colcap", "col10y"}
    leaked = ids & cop_only
    assert not leaked, f"COP-only drivers leaked into Gold profile: {leaked}"
    # Gold's real drivers present
    assert any("dxy" in i or "dtwex" in i for i in ids)
    assert any("dfii" in i for i in ids)  # real yields


# --------------------------------------------------------------------------- D1 (audit A10-01/02/03)
def test_d1_regime_thresholds_consumed_not_copied():
    """The classifier must HONOR profile hurst thresholds (parametrized, not hardcoded),
    and any fitted values must never be the COP pair (0.52/0.42)."""
    import numpy as np
    import yaml as _yaml

    from src.gold_rl.indicators import classify_regime

    raw = _yaml.safe_load(
        (Path(__file__).resolve().parents[2] / "config" / "assets" / "xauusd.yaml")
        .read_text(encoding="utf-8"))
    rg = raw.get("regime_gate") or {}
    ht, hm = rg.get("hurst_trending"), rg.get("hurst_mean_rev")
    # If fitted, they must not be the copied COP thresholds (the playbook forbids it).
    if ht is not None or hm is not None:
        assert (ht, hm) != (0.52, 0.42), "Gold thresholds must be re-fit, not copied from COP"

    # Synthetic frame: strong-ADX, mid-Hurst (0.55) rows — classification must FLIP when
    # the trending pivot moves across 0.55, proving the parameter is actually consumed.
    n = 30
    df = pd.DataFrame({
        "realized_vol_20": np.full(n, 0.20),
        "adx_14": np.full(n, 30.0),
        "hurst_smooth": np.full(n, 0.55),
        "z_sma50": np.zeros(n),
    })
    from src.gold_rl.indicators import TREND

    lo = classify_regime(df, dwell=1, hurst_trending=0.50, hurst_mean_rev=0.50)
    hi = classify_regime(df, dwell=1, hurst_trending=0.60, hurst_mean_rev=0.60)
    assert (lo["regime"] == TREND).all(), "pivot 0.50 with Hurst 0.55 must classify TREND"
    assert not (hi["regime"] == TREND).any(), "pivot 0.60 with Hurst 0.55 must NOT be TREND"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
