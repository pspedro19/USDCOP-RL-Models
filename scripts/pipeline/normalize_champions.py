#!/usr/bin/env python3
"""Champion-per-asset normalization — one idempotent pass, callable from every pipeline.

Contract: CTR-STRAT-REGISTRY-001 (champion invariant)

Why this exists as a STANDALONE script: the champion set used to live inside one publisher
(`publish_gold_trend_simple.py`), so only the pipeline that happened to call that publisher
enforced it. The weekly asset DAGs each publish their own strategy family — gold's
`run_gold_pipeline` iterates the OLD family (gold_trend_simple is not in its STRATEGIES dict),
and BTC's `run_btc_pipeline` publishes the b2 family — so every Sunday the cycle could quietly
re-label a dethroned strategy as visible and leave the actual champion without a weekly
refresh. A rule enforced in one caller is a rule enforced by coincidence.

Champions are chosen by OOS evidence (constitution 3: "si la estrategia no lo bate, el
baseline ES la estrategia"), recorded here as the single writable authority, and the guard
refuses to finish successfully if any asset would end up with zero visible strategies —
publishing one asset's champion must never silently retire another asset's only tool.

Run:  python scripts/pipeline/normalize_champions.py           # enforce
      python scripts/pipeline/normalize_champions.py --check   # verify only, exit 1 on drift
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PUBLIC_DATA = REPO / "usdcop-trading-dashboard" / "public" / "data"

# The single writable champion authority. Changing a champion is an evidence decision:
# update this mapping in the same commit as the evidence artifact that justifies it.
#   usdcop  smart_simple_v11        frozen; forward running since 2026-03-18
#   xauusd  gold_trend_simple       only strategy passing exposure-matching in BOTH windows
#   btcusdt btc_hodl_b1             +4.70% in the held-out year vs -1.37% for every variant
#   spx500  spx500_regime_gated_v1  only thing measured on real data; experimental
CHAMPION_BY_ASSET: dict[str, str] = {
    "usdcop": "smart_simple_v11",
    "xauusd": "gold_trend_simple",
    "btcusdt": "btc_hodl_b1",
    "spx500": "spx500_regime_gated_v1",
}

# Statuses a champion may hold; anything else visible gets archived.
_CHAMPION_KEEP = {"experimental", "paper", "production"}


def normalize(check_only: bool = False) -> int:
    strat_root = PUBLIC_DATA / "strategies"
    champions = set(CHAMPION_BY_ASSET.values())
    changed, drift = [], []

    for man_path in sorted(strat_root.glob("*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        sid, status = man.get("strategy_id"), man.get("status")
        want = status if sid in champions and status in _CHAMPION_KEEP else (
            "experimental" if sid in champions else "archived")
        if status != want:
            drift.append(f"{sid}: {status} -> {want}")
            if not check_only:
                man["status"] = want
                man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
                changed.append(sid)

    # Refresh the registry from manifests so the dashboard sees the normalized truth.
    if changed:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "strategy_manifest", REPO / "src" / "contracts" / "strategy_manifest.py")
            sm = importlib.util.module_from_spec(spec)
            sys.modules["strategy_manifest"] = sm
            spec.loader.exec_module(sm)
            sm.BundlePublisher(PUBLIC_DATA).refresh_registry()
            print(f"[champions] normalizados: {drift}")
        except Exception as e:  # noqa: BLE001
            print(f"[champions] manifests escritos pero registry NO refrescado: {e}")
            return 1
    elif drift:
        print(f"[champions] DRIFT detectado (check-only): {drift}")
        return 1
    else:
        print("[champions] sin cambios")

    # THE GUARD: no asset may end with zero visible strategies. This exact failure happened
    # on 2026-07-21 -- publishing gold archived SPX500's only strategy and the publisher
    # exited 0. A normalizer that can empty an asset must refuse to finish well.
    reg = json.loads((PUBLIC_DATA / "registry.json").read_text(encoding="utf-8"))
    orphaned = [a for a in CHAMPION_BY_ASSET
                if not [s for s in reg["strategies"]
                        if s.get("asset_id") == a and s.get("status") != "archived"]]
    if orphaned:
        print(f"[champions] ERROR: activos sin estrategia visible: {orphaned}")
        return 1
    print(f"[champions] OK: {len(CHAMPION_BY_ASSET)} activos con campeona visible")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on drift")
    return normalize(check_only=ap.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
