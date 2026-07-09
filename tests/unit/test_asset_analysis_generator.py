"""
Unit tests for the multi-asset weekly analysis generator (Gold/BTC).

Validates that the generator:
  - loads its SSOT profiles (config/analysis/analysis_assets.yaml)
  - produces a schema-valid WeeklyViewData payload from REAL OHLCV + strategy trades
  - emits JSON with NO NaN/Inf (dashboard JSON-safety invariant)
  - writes namespaced files + a newest-first per-asset index

News fetching is skipped (skip_news=True) so the suite is offline/fast and does
not depend on GDELT availability.

Contract mirrors: usdcop-trading-dashboard/lib/contracts/weekly-analysis.contract.ts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.asset_analysis_generator import AssetAnalysisGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Assets whose OHLCV seeds must be present for these tests to be meaningful.
_ASSETS = ["xauusd", "btcusdt"]


def _seed_exists(gen: AssetAnalysisGenerator, asset: str) -> bool:
    p = gen.profile(asset)
    return (PROJECT_ROOT / p.ohlcv_seed).exists()


@pytest.fixture(scope="module")
def gen() -> AssetAnalysisGenerator:
    return AssetAnalysisGenerator()


@pytest.mark.parametrize("asset", _ASSETS)
def test_profile_loads_from_ssot(gen: AssetAnalysisGenerator, asset: str):
    p = gen.profile(asset)
    assert p.asset_id == asset
    assert "/" in p.symbol
    assert p.chart_symbol and "/" not in p.chart_symbol
    assert p.strategy_id
    assert p.annualization_days in (252, 365)


def test_unknown_asset_raises(gen: AssetAnalysisGenerator):
    with pytest.raises(KeyError):
        gen.profile("doesnotexist")


@pytest.mark.parametrize("asset", _ASSETS)
def test_generate_week_is_schema_valid_and_json_safe(gen: AssetAnalysisGenerator, asset: str):
    if not _seed_exists(gen, asset):
        pytest.skip(f"{asset} OHLCV seed not present")

    # Pick a week that exists in the seed (mid-year 2026 is covered by both).
    view = gen.generate_week(asset, 2026, 20, skip_news=True)
    assert view is not None, "expected bars for 2026-W20"

    # Required top-level sections the /analysis page consumes
    for key in (
        "weekly_summary", "daily_entries", "signals",
        "technical_analysis", "news_intelligence", "macro_regime",
    ):
        assert key in view, f"missing section {key}"

    # weekly_summary.ohlcv must be numeric + complete
    ohlcv = view["weekly_summary"]["ohlcv"]
    for f in ("open", "high", "low", "close", "change_pct"):
        assert isinstance(ohlcv[f], (int, float)), f"{f} not numeric"
    assert ohlcv["high"] >= ohlcv["low"]

    # signals.h5 present with a valid direction
    assert view["signals"]["h5"]["direction"] in ("LONG", "SHORT", "HOLD")

    # daily entries carry per-day close/change
    assert len(view["daily_entries"]) >= 1
    de = view["daily_entries"][0]
    for f in ("analysis_date", "usdcop_close", "usdcop_change_pct", "day_of_week"):
        assert f in de

    # technicals computed from real price
    ind = view["technical_analysis"]["indicators"]
    assert "rsi_14" in ind and "macd_line" in ind
    sr = view["technical_analysis"]["support_resistance"]
    assert sr["resistance"] >= sr["support"]

    # JSON-safety invariant: no NaN / Infinity anywhere
    dumped = json.dumps(view)
    assert "NaN" not in dumped and "Infinity" not in dumped


@pytest.mark.parametrize("asset", _ASSETS)
def test_write_and_index_roundtrip(gen: AssetAnalysisGenerator, asset: str, tmp_path):
    if not _seed_exists(gen, asset):
        pytest.skip(f"{asset} OHLCV seed not present")

    # Redirect output to a temp dir so the test never mutates real dashboard data.
    gen.output_root = tmp_path
    view = gen.generate_week(asset, 2026, 20, skip_news=True)
    assert view is not None
    out = gen.write_week(asset, view, generated_at="2026-01-01T00:00:00+00:00")
    assert out.exists()

    idx_path = gen.rebuild_index(asset)
    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    assert idx["asset_id"] == asset
    assert idx["weeks"], "index should list the written week"
    # newest-first ordering (frontend relies on weeks[0] == most recent)
    years_weeks = [(w["year"], w["week"]) for w in idx["weeks"]]
    assert years_weeks == sorted(years_weeks, reverse=True)

    # events file created so the calendar route 200s
    assert (tmp_path / asset / "upcoming_events.json").exists()
