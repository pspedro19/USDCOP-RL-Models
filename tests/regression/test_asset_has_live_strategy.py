"""Every asset must have a visible strategy, and leverage must be an explicit bound.

Contract: CTR-QUANT-EVIDENCE-001 / CTR-FORWARD-TRACK-001

Two guards, both written after the failure they prevent actually happened.

1. Publishing `gold_trend_simple` archived SPX500's only strategy and left that asset with ZERO
   visible tools -- and the publisher exited 0. The champion set was a flat
   {"smart_simple_v11", "btc_trend_b2", SID} that simply never mentioned spx500. A flat set
   cannot express "one per asset" because it does not know which assets exist.

2. There was no MAX_LEVERAGE constant. Leverage was bounded only implicitly by
   MAX_POSITION_SIZE = 1.0, which is one refactor away from not being a bound. No asset here
   has an OOS-validated edge that would justify levering it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "usdcop-trading-dashboard" / "public" / "data" / "registry.json"


def _registry() -> list[dict]:
    if not REGISTRY.is_file():
        pytest.skip("registry.json absent")
    return json.loads(REGISTRY.read_text(encoding="utf-8"))["strategies"]


def test_every_asset_has_a_visible_strategy():
    rows = _registry()
    assets = {s.get("asset_id") for s in rows if s.get("asset_id")}
    assert assets, "registry declares no assets"

    orphaned = [
        a for a in sorted(assets)
        if not [s for s in rows if s.get("asset_id") == a and s.get("status") != "archived"]
    ]
    assert not orphaned, (
        f"assets with every strategy archived: {orphaned}. The dashboard renders these as empty "
        "with no error. Publishing one asset's champion must never silently retire another's."
    )


def test_at_most_one_non_archived_per_asset():
    """One champion per asset keeps the comparison honest.

    Two live strategies for the same asset means the dashboard picks one, and which one it
    picks becomes an implicit, unrecorded selection decision.
    """
    rows = _registry()
    by_asset: dict[str, list[str]] = {}
    for s in rows:
        if s.get("status") != "archived" and s.get("asset_id"):
            by_asset.setdefault(s["asset_id"], []).append(s["strategy_id"])
    multi = {a: ids for a, ids in by_asset.items() if len(ids) > 1}
    assert not multi, (
        f"more than one live strategy per asset: {multi}. Archive the challengers or record "
        "the champion explicitly."
    )


def test_max_leverage_is_declared_and_is_one():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "core_constants", ROOT / "src" / "core" / "constants.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "MAX_LEVERAGE"), (
        "src/core/constants.py has no MAX_LEVERAGE. Leverage bounded only implicitly by "
        "MAX_POSITION_SIZE is not a control."
    )
    assert mod.MAX_LEVERAGE == 1.0, (
        f"MAX_LEVERAGE is {mod.MAX_LEVERAGE}, not 1.0 (spot only). Raising it requires an ADR "
        "and a signed withdrawal protocol whose drawdown trigger accounts for the new exposure. "
        "No asset in this system has an OOS-validated edge that justifies leverage."
    )
