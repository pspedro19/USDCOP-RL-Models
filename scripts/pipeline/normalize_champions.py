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
FROZEN_MANIFESTS = REPO / "config" / "strategy_manifests"

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

# BL-13/C-005 surface discriminator. Mirrors src/contracts/strategy_manifest.SURFACES
# (this script loads that module by path only when refreshing the registry, so the
# constant is repeated here — keep the two in lockstep).
_SURFACES = ("action", "diagnostic")


def _frozen_surfaces() -> dict[str, str]:
    """strategy_id -> `surface` declared by its frozen YAML manifest (BL-13).

    The frozen manifests are the surface authority for the strategies they cover;
    diagnostic surfaces exist to be looked at, never traded: they can NEVER be visible
    as champion nor hold an experimental/paper/production status.

    Fail-closed (remedio-2 C-005): a frozen YAML that DECLARES a surface outside
    _SURFACES raises ValueError — normalize() turns that into exit 1 in BOTH modes.
    Absence keeps legacy semantics (the YAML simply is not surface-authoritative).
    """
    import yaml
    out: dict[str, str] = {}
    for p in sorted(FROZEN_MANIFESTS.glob("*.yaml")):
        m = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        sid, surf = m.get("strategy_id"), m.get("surface")
        if surf is not None and surf not in _SURFACES:
            raise ValueError(
                f"{p.name}: surface {surf!r} fuera de {_SURFACES} — un manifiesto "
                "congelado con surface invalido es ERROR duro (fail-closed BL-13/C-005); "
                "corrige el YAML, no se ignora ni se coacciona")
        if sid and surf in _SURFACES:
            out[sid] = surf
    return out


def _diagnostic_ids() -> set[str]:
    """strategy_ids whose frozen manifest declares `surface: diagnostic` (BL-13)."""
    return {sid for sid, surf in _frozen_surfaces().items() if surf == "diagnostic"}


def _load_strategy_manifest_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "strategy_manifest", REPO / "src" / "contracts" / "strategy_manifest.py")
    sm = importlib.util.module_from_spec(spec)
    sys.modules["strategy_manifest"] = sm
    spec.loader.exec_module(sm)
    return sm


def _registry_missing_surface() -> list[str]:
    """Registry entries whose `surface` is absent/invalid (C-005 divergence guard)."""
    try:
        reg = json.loads((PUBLIC_DATA / "registry.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["<registry.json ilegible>"]
    return [str(s.get("strategy_id")) for s in reg.get("strategies", [])
            if s.get("surface") not in _SURFACES]


def normalize(check_only: bool = False) -> int:
    strat_root = PUBLIC_DATA / "strategies"
    champions = set(CHAMPION_BY_ASSET.values())
    # Fail-closed (remedio-2): an invalid frozen surface aborts BOTH modes with exit 1
    # before anything is normalized — a broken authority must not drive rewrites.
    try:
        frozen_surface = _frozen_surfaces()
    except ValueError as e:
        print(f"[champions] ERROR: {e}")
        return 1
    changed, drift, surface_errors = [], [], []

    for man_path in sorted(strat_root.glob("*/manifest.json")):
        man = json.loads(man_path.read_text(encoding="utf-8"))
        sid, status = man.get("strategy_id"), man.get("status")
        # C-005: every bundle manifest carries surface. The frozen YAML is authoritative
        # where one exists; otherwise the bundle's own declaration; ABSENCE -> "action"
        # (every published bundle was a tradeable candidate — diagnostic is opt-in).
        declared = man.get("surface")
        if declared is not None and declared not in _SURFACES and sid not in frozen_surface:
            # remedio-2: an unknown DECLARED surface without frozen authority to repair
            # it is an ERROR — never coerced to 'action', never normalized in silence.
            surface_errors.append(
                f"{sid}: bundle manifest declara surface {declared!r} fuera de "
                f"{_SURFACES} — fail-closed, no se coacciona (BL-13/C-005)")
            continue
        want_surface = frozen_surface.get(
            sid, declared if declared in _SURFACES else "action")
        if want_surface == "diagnostic":
            # BL-13: a diagnostic surface is forced to archived, no matter what the
            # champion authority or its bundle claims — and that contradiction is an error.
            want = "archived"
            if sid in champions:
                surface_errors.append(
                    f"{sid}: surface=diagnostic pero figura en CHAMPION_BY_ASSET — "
                    "una superficie diagnostica JAMAS puede ser campeona")
            elif status in _CHAMPION_KEEP:
                surface_errors.append(
                    f"{sid}: surface=diagnostic con status visible {status!r} — forzado a archived")
        else:
            want = status if sid in champions and status in _CHAMPION_KEEP else (
                "experimental" if sid in champions else "archived")
        if status != want:
            drift.append(f"{sid}: {status} -> {want}")
        if man.get("surface") != want_surface:
            drift.append(f"{sid}: surface {man.get('surface')} -> {want_surface}")
        if (status != want or man.get("surface") != want_surface) and not check_only:
            man["status"] = want
            man["surface"] = want_surface
            man_path.write_text(json.dumps(man, indent=2, ensure_ascii=False),
                                encoding="utf-8")
            changed.append(sid)

    for err in surface_errors:
        print(f"[champions] ERROR: {err}")

    # Refresh the registry from manifests so the dashboard sees the normalized truth —
    # also when the manifests were already right but registry.json predates C-005
    # (surface_present=0 was exactly the shipped-but-not-served gap Codex rejected).
    registry_stale = bool(_registry_missing_surface())
    if not check_only and (changed or registry_stale):
        try:
            sm = _load_strategy_manifest_module()
            from datetime import datetime, timezone
            builder = sm.RegistryBuilder(
                PUBLIC_DATA, generated_at=datetime.now(timezone.utc).isoformat())
            builder.write(builder.build(write_manifests=False))
            print(f"[champions] normalizados: {drift or ['registry: surface backfill']}")
        except Exception as e:  # noqa: BLE001
            print(f"[champions] manifests escritos pero registry NO refrescado: {e}")
            return 1
    elif drift or (check_only and registry_stale):
        print(f"[champions] DRIFT detectado (check-only): "
              f"{drift or ['registry.json sin surface — corre normalize']}")
        return 1
    else:
        print("[champions] sin cambios")

    # THE GUARD: no asset may end with zero visible strategies. This exact failure happened
    # on 2026-07-21 -- publishing gold archived SPX500's only strategy and the publisher
    # exited 0. A normalizer that can empty an asset must refuse to finish well.
    reg = json.loads((PUBLIC_DATA / "registry.json").read_text(encoding="utf-8"))
    # C-005 fail-closed per row: the registry the dashboard serves must carry surface,
    # and a diagnostic row may never be visible. Checked on the ARTIFACT, not the YAMLs.
    no_surface = [str(s.get("strategy_id")) for s in reg.get("strategies", [])
                  if s.get("surface") not in _SURFACES]
    if no_surface:
        print(f"[champions] ERROR: entradas de registry sin surface valido: {no_surface}")
        return 1
    visible_diag = [s["strategy_id"] for s in reg["strategies"]
                    if s.get("surface") == "diagnostic" and s.get("status") != "archived"]
    if visible_diag:
        print(f"[champions] ERROR: superficies diagnostic visibles en registry: {visible_diag}")
        return 1
    orphaned = [a for a in CHAMPION_BY_ASSET
                if not [s for s in reg["strategies"]
                        if s.get("asset_id") == a and s.get("status") != "archived"]]
    if orphaned:
        print(f"[champions] ERROR: activos sin estrategia visible: {orphaned}")
        return 1
    if surface_errors:
        return 1
    print(f"[champions] OK: {len(CHAMPION_BY_ASSET)} activos con campeona visible")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify only; exit 1 on drift")
    return normalize(check_only=ap.parse_args().check)


if __name__ == "__main__":
    raise SystemExit(main())
