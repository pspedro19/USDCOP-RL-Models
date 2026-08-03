"""Contract gates for the SP500 asset onboarding."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_spx500_asset_profile_is_session_and_feature_specific():
    cfg = yaml.safe_load((ROOT / "config/assets/spx500.yaml").read_text())
    assert cfg["asset_class"] == "equity_index"
    assert cfg["session"]["timezone"] == "America/New_York"
    assert cfg["session"]["open"] == "09:30"
    assert cfg["session"]["close"] == "16:00"
    assert cfg["feature_profile"]["causal_lag_bars"] >= 1
    assert "breadth_ma200" in cfg["feature_profile"]["features"]
    assert "log_ret_5m" not in cfg["feature_profile"]["features"]


def test_spx500_analysis_and_registry_are_discoverable():
    analysis = yaml.safe_load((ROOT / "config/analysis/analysis_assets.yaml").read_text())
    assert analysis["assets"]["spx500"]["strategy_id"] == "spx500_regime_gated_v1"
    registry = json.loads((ROOT / "usdcop-trading-dashboard/public/data/registry.json").read_text())
    spx_assets = [a for a in registry["assets"] if a["asset_id"] == "spx500"]
    assert spx_assets and spx_assets[0]["asset_class"] == "equity_index"
    assert any(s["strategy_id"] == "spx500_regime_gated_v1" for s in registry["strategies"])


def test_catalog_backend_uses_registry_symbols_without_fx_allowlist():
    route = (ROOT / "usdcop-trading-dashboard/app/api/catalog/route.ts").read_text(encoding="utf-8")
    # New assets must become price probes by registry publication alone.
    assert "LIVE_FX_SYMBOLS" not in route
    assert "new Set(symbols)" in route


def test_spx500_scaffold_does_not_use_forward_macro_shift():
    source = (ROOT / "src/strategies/spx500_regime_gated_v1/datagen.py").read_text(encoding="utf-8", errors="replace")
    assert "shift(-" not in source
    assert "SYNTHETIC_WARNING" in source


def test_spx500_manifest_blocks_production_until_real_oos():
    manifest = json.loads((ROOT / "usdcop-trading-dashboard/public/data/strategies/spx500_regime_gated_v1/manifest.json").read_text())
    assert manifest["status"] == "experimental"
    assert manifest["capabilities"]["live"] is False
    assert manifest["approval"]["status"] == "PENDING_REAL_DATA_AND_OOS"
