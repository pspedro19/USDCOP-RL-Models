"""Contract/regression tests for the dynamic strategy registry (CTR-STRAT-REGISTRY-001).

Guards the data-science lifecycle invariants that let a NEW model version appear as a
selectable, replayable entry in the frontend WITHOUT breaking existing consumptions:

  - a training run publishes an immutable, versioned backtest bundle
  - a second version COEXISTS with the first (never overwrites it)
  - publishing NEVER touches the legacy production/*.json the current frontend consumes
  - manifest/registry are JSON-safe (no Infinity/NaN)
  - the chart symbol is derived (no hardcoded "USDCOP")

The registry contract is loaded as a standalone leaf module (no ML stack) so these tests
run fast and in isolation.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _REPO / "src" / "contracts" / "strategy_manifest.py"


def _load_contract():
    if "strategy_manifest" in sys.modules:
        return sys.modules["strategy_manifest"]
    spec = importlib.util.spec_from_file_location("strategy_manifest", _MANIFEST_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["strategy_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


sm = _load_contract()


# --------------------------------------------------------------------------- helpers
def make_summary(sid: str, year: int, ret: float, extra: dict | None = None) -> dict:
    s = {
        "strategy_id": sid,
        "year": year,
        "strategies": {sid: {"total_return_pct": ret, "sharpe": 3.0}},
        "statistical_tests": {"p_value": 0.01},
    }
    if extra:
        s.update(extra)
    return s


def make_trades(sid: str) -> dict:
    return {"strategy_id": sid, "trades": [{"trade_id": 1, "side": "SHORT", "pnl_pct": 1.0}]}


def publish(pub, sid="s1", asset="usdcop", symbol="USD/COP", version="1.0.0", year=2025, ret=25.0):
    return pub.publish(
        strategy_id=sid, asset_id=asset, symbol=symbol, display_name="Strat 1",
        pipeline_type="ml_forecasting", timeframe="weekly", version=version, year=year,
        summary=make_summary(sid, year, ret), trades=make_trades(sid),
        gates={"passed": 5, "of": 5, "recommendation": "PROMOTE"},
        headline={"return_pct": ret, "sharpe": 3.0, "p_value": 0.01},
    )


# --------------------------------------------------------------------------- tests
def test_publish_creates_versioned_immutable_bundle(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    r = publish(pub)
    assert r["wrote_new_files"] is True
    assert (tmp_path / "strategies/s1/backtests/1.0.0/summary_2025.json").exists()
    assert (tmp_path / "strategies/s1/backtests/1.0.0/trades_2025.json").exists()
    assert (tmp_path / "strategies/s1/manifest.json").exists()
    assert (tmp_path / "registry.json").exists()


def test_republish_same_version_is_immutable(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    publish(pub, ret=25.0)
    # Attempt to re-publish the SAME (version, year) with a different return -> must NOT overwrite.
    r2 = publish(pub, ret=999.0)
    assert r2["immutable_hit"] is True
    data = json.loads((tmp_path / "strategies/s1/backtests/1.0.0/summary_2025.json").read_text())
    assert data["strategies"]["s1"]["total_return_pct"] == 25.0  # original preserved


def test_second_version_coexists_and_v1_untouched(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    publish(pub, version="1.0.0", ret=25.0)
    v1_bytes = (tmp_path / "strategies/s1/backtests/1.0.0/summary_2025.json").read_bytes()
    publish(pub, version="2.0.0", ret=18.0)  # new version = changed hyperparameters/features
    # v1 files are byte-identical (immutable)
    assert (tmp_path / "strategies/s1/backtests/1.0.0/summary_2025.json").read_bytes() == v1_bytes
    # both versions coexist in the manifest (drives the frontend version dropdown)
    manifest = json.loads((tmp_path / "strategies/s1/manifest.json").read_text())
    versions = {b["model_version"] for b in manifest["backtests"]}
    assert versions == {"1.0.0", "2.0.0"}
    mv = {m["version"] for m in manifest["model_versions"]}
    assert mv == {"1.0.0", "2.0.0"}
    # the just-published version is the active one
    active = [m["version"] for m in manifest["model_versions"] if m["active"]]
    assert active == ["2.0.0"]


def test_both_versions_are_replayable(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    publish(pub, version="1.0.0")
    publish(pub, version="2.0.0")
    manifest = json.loads((tmp_path / "strategies/s1/manifest.json").read_text())
    assert all(b["replayable"] for b in manifest["backtests"])
    assert manifest["capabilities"]["replay"] is True


def test_registry_discovers_published_strategy(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    publish(pub, version="1.0.0")
    publish(pub, version="2.0.0")
    registry = json.loads((tmp_path / "registry.json").read_text())
    sids = [s["strategy_id"] for s in registry["strategies"]]
    assert "s1" in sids
    assert registry["default"]["strategy_id"] in sids


def test_publish_does_not_touch_legacy_files(tmp_path):
    # Simulate the files the CURRENT frontend consumes.
    prod = tmp_path / "production"
    prod.mkdir()
    (prod / "summary_2025.json").write_text('{"legacy":"do-not-touch"}', encoding="utf-8")
    (prod / "strategies.json").write_text('{"strategies":[],"default_strategy":"x"}', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in prod.iterdir()}

    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    publish(pub, version="1.0.0")

    after = {p.name: p.read_bytes() for p in prod.iterdir()}
    assert before == after  # legacy consumption contracts are byte-identical


def test_json_safety_infinity_and_nan_become_null(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    bad = make_summary("s1", 2025, 25.0, extra={"profit_factor": float("inf"), "ratio": float("nan")})
    pub.publish(
        strategy_id="s1", asset_id="usdcop", symbol="USD/COP", display_name="S",
        pipeline_type="ml_forecasting", timeframe="weekly", version="1.0.0", year=2025,
        summary=bad, trades=make_trades("s1"),
    )
    raw = (tmp_path / "strategies/s1/backtests/1.0.0/summary_2025.json").read_text()
    assert "Infinity" not in raw and "NaN" not in raw
    data = json.loads(raw)  # would raise if invalid JSON
    assert data["profit_factor"] is None and data["ratio"] is None


def test_chart_symbol_derived_not_hardcoded(tmp_path):
    pub = sm.BundlePublisher(tmp_path, generated_at="t0")
    pub.publish(
        strategy_id="gold_strat", asset_id="xauusd", symbol="XAU/USD", display_name="Gold",
        pipeline_type="ml_forecasting", timeframe="weekly", version="1.0.0", year=2025,
        summary=make_summary("gold_strat", 2025, 12.0), trades=make_trades("gold_strat"),
    )
    manifest = json.loads((tmp_path / "strategies/gold_strat/manifest.json").read_text())
    assert manifest["symbol"] == "XAU/USD"
    assert manifest["chart_symbol"] == "XAUUSD"  # derived, never "USDCOP"
    registry = json.loads((tmp_path / "registry.json").read_text())
    assert any(a["asset_id"] == "xauusd" for a in registry["assets"])


if __name__ == "__main__":  # allow running without pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
