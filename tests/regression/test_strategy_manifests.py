"""Every champion has a frozen manifest, and the freeze is enforced by hash.

Contract: CTR-STRAT-MANIFEST-001 (Fase 0, plan 2026-07-21 — Codex step 1 + run-experiment)

A manifest is only immutable if something breaks when its subject changes. The teeth here:
`code_hash_sha256_16` is the hash of the strategy's source files at freeze time. Editing any
of those files without re-freezing the manifest turns this test red — which converts "we
changed the strategy" from a silent event into a conscious versioning decision, exactly what
the run-experiment skill demands of frozen SSOT configs ("config congelado al arrancar; si
hace falta ajustar, se ABORTA y se abre un experimento nuevo").

The clock is asserted per asset because the library carries two incompatible annualization
conventions (onboard-asset: BTC sqrt365; xasset engine: 252 after intersect) and a metric
cited without its clock is unfalsifiable.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS = ROOT / "config" / "strategy_manifests"

EXPECTED_CLOCKS = {"usdcop": 52, "xauusd": 252, "btcusdt": 365, "spx500": 252}


def _champions() -> dict[str, str]:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "normalize_champions", ROOT / "scripts" / "pipeline" / "normalize_champions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CHAMPION_BY_ASSET


def _manifest(asset: str) -> dict:
    p = MANIFESTS / f"{asset}.yaml"
    assert p.is_file(), f"champion asset {asset} has no frozen manifest at {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def test_every_champion_has_a_manifest_that_names_it():
    for asset, sid in _champions().items():
        m = _manifest(asset)
        assert m["strategy_id"] == sid, (
            f"{asset}: manifest freezes {m['strategy_id']!r} but the champion authority says "
            f"{sid!r}. A champion change requires a new manifest in the same commit."
        )


def test_clock_is_declared_and_correct():
    for asset, expect in EXPECTED_CLOCKS.items():
        m = _manifest(asset)
        got = m["clock"]["periods_per_year"]
        assert got == expect, f"{asset}: clock {got} != {expect}"


def test_code_hash_detects_strategy_drift():
    """Editing a frozen strategy's source without re-freezing must fail CI."""
    for asset in _champions():
        m = _manifest(asset)
        digest = hashlib.sha256()
        for f in m["files"]:
            digest.update((ROOT / f).read_bytes())
        current = digest.hexdigest()[:16]
        assert current == m["code_hash_sha256_16"], (
            f"{asset}: strategy source drifted from its frozen manifest "
            f"(manifest={m['code_hash_sha256_16']}, current={current}). Either revert the "
            "source change, or consciously re-freeze: bump the manifest version, update the "
            "hash, and count the look as a trial if any result was observed."
        )


def test_manifests_declare_action_surface():
    """BL-13: every frozen manifest declares its surface, and diagnostic never champions.

    `surface` is the discriminator between tradeable strategies (action) and
    look-only research surfaces (diagnostic). A diagnostic surface must never be
    the champion the registry serves — normalize_champions enforces it at runtime;
    this test enforces it at freeze time.
    """
    champions = set(_champions().values())
    manifests = sorted(MANIFESTS.glob("*.yaml"))
    assert manifests, f"no frozen manifests found under {MANIFESTS}"
    for p in manifests:
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        surface = m.get("surface")
        assert surface in {"action", "diagnostic"}, (
            f"{p.name}: surface is {surface!r} — every frozen manifest must declare "
            "surface: action|diagnostic (BL-13)"
        )
        if surface == "diagnostic":
            assert m["strategy_id"] not in champions, (
                f"{p.name}: {m['strategy_id']!r} declares surface=diagnostic but is a "
                "champion in CHAMPION_BY_ASSET — diagnostic surfaces can never be champions"
            )


def test_composite_declares_components():
    """BL-14 / FABRIC §16: composite strategies declare their frozen-RECIPE predictor.

    Any manifest whose model.kind is ml_ensemble (or that declares a components
    block) must expose components[0] with role=decision_input and recipe_frozen
    true. What is frozen is the RECIPE (features, hyperparams, weekly expanding
    retrain, train-only scaler) — never the weights: each Sunday retrain produces
    a registered model_snapshot under the same spec. Rule-based manifests without
    an ML model are exempt.
    """
    for p in sorted(MANIFESTS.glob("*.yaml")):
        m = yaml.safe_load(p.read_text(encoding="utf-8"))
        model = m.get("model") or {}
        is_composite = model.get("kind") == "ml_ensemble" or "components" in m
        if not is_composite:
            continue  # rule-based (no ML model): not required to declare components
        comps = m.get("components")
        assert isinstance(comps, list) and comps, (
            f"{p.name}: model.kind=ml_ensemble but no components block — declare the "
            "predictor as a frozen RECIPE (BL-14)"
        )
        first = comps[0]
        assert first.get("role") == "decision_input", (
            f"{p.name}: components[0].role is {first.get('role')!r}, expected "
            "'decision_input' (BL-14)"
        )
        assert first.get("recipe_frozen") is True, (
            f"{p.name}: components[0].recipe_frozen must be true — the recipe is frozen, "
            "snapshots are registered; never claim immutable weights (FABRIC §16)"
        )


def test_registry_carries_surface_and_diagnostic_never_visible():
    """BL-13 / C-005: `surface` must reach the registry the dashboard serves.

    The frozen YAMLs declaring surface is necessary but not sufficient — the registry
    (public/data/registry.json) is what the frontend actually reads, and a wall that
    exists only in files the frontend never opens is a wall by convention (the exact
    gap Codex rejected: strategies=18, surface_present=0). Every registry entry must
    declare surface, and a diagnostic entry may never be visible nor champion.
    """
    reg = json.loads((ROOT / "usdcop-trading-dashboard/public/data/registry.json")
                     .read_text(encoding="utf-8"))
    champions = set(_champions().values())
    assert reg["strategies"], "registry lists no strategies"
    for s in reg["strategies"]:
        sid = s.get("strategy_id")
        assert s.get("surface") in {"action", "diagnostic"}, (
            f"registry entry {sid!r}: surface is {s.get('surface')!r} — every registry "
            "entry must declare surface: action|diagnostic (BL-13/C-005). "
            "Run scripts/pipeline/normalize_champions.py to stamp+refresh."
        )
        if s["surface"] == "diagnostic":
            assert s.get("status") == "archived", (
                f"registry entry {sid!r}: surface=diagnostic with visible status "
                f"{s.get('status')!r} — diagnostic surfaces exist to be looked at, never traded"
            )
            assert sid not in champions, (
                f"registry entry {sid!r}: surface=diagnostic but champion in "
                "CHAMPION_BY_ASSET — a diagnostic surface can never be the champion served"
            )


def test_diagnostic_champion_forces_red_exit_and_archival(tmp_path, monkeypatch):
    """BL-13 verificación: entrada diagnostic con status CHAMPION => exit rojo, REAL.

    Runs normalize_champions end-to-end in a sandbox where the champion authority
    names a strategy whose frozen manifest declares surface: diagnostic. Both modes
    must exit red, the bundle manifest must be forced to archived, and the refreshed
    registry must carry surface=diagnostic + status=archived — i.e. the contradiction
    is not just printed, it is neutralized in the artifact the dashboard serves.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "normalize_champions_sandbox",
        ROOT / "scripts" / "pipeline" / "normalize_champions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "aaa.yaml").write_text(
        yaml.safe_dump({"strategy_id": "diag_x", "surface": "diagnostic"}),
        encoding="utf-8")

    public = tmp_path / "public"
    bundle = public / "strategies" / "diag_x"
    bundle.mkdir(parents=True)
    manifest = {
        "strategy_id": "diag_x", "asset_id": "aaa", "symbol": "AAA/USD",
        "chart_symbol": "AAAUSD", "display_name": "Diagnostic X",
        "pipeline_type": "rule_based", "timeframe": "weekly",
        "status": "experimental", "backtests": [], "model_versions": [],
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (public / "registry.json").write_text(json.dumps({
        "generated_at": "sandbox", "assets": [],
        "strategies": [{"strategy_id": "diag_x", "asset_id": "aaa",
                        "status": "experimental"}],
        "default": {"asset_id": "aaa", "strategy_id": "diag_x"},
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "FROZEN_MANIFESTS", frozen)
    monkeypatch.setattr(mod, "PUBLIC_DATA", public)
    monkeypatch.setattr(mod, "CHAMPION_BY_ASSET", {"aaa": "diag_x"})

    assert mod.normalize(check_only=True) != 0, (
        "check mode must exit red when a diagnostic surface is champion")
    assert mod.normalize(check_only=False) != 0, (
        "enforce mode must still exit red: a diagnostic champion is a contradiction "
        "to surface, not a state to normalize into silence")

    after = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    assert after["status"] == "archived", "diagnostic surface must be forced to archived"
    assert after.get("surface") == "diagnostic", (
        "enforce must stamp the frozen surface into the bundle manifest")

    reg = json.loads((public / "registry.json").read_text(encoding="utf-8"))
    entry = next(s for s in reg["strategies"] if s["strategy_id"] == "diag_x")
    assert entry.get("surface") == "diagnostic" and entry.get("status") == "archived", (
        f"registry must be refreshed with the neutralized truth, got {entry!r} — "
        "the dashboard reads registry.json, not the frozen YAMLs"
    )


def test_surface_contract_is_mirrored_in_typescript():
    """C-005 mirror rule: the TS contracts must carry the same optional surface field.

    Mirror map (contract-change skill): strategy_schema.py ↔ strategy.contract.ts and
    strategy_manifest.py ↔ strategy-manifest.contract.ts — same commit, both sides.
    """
    import re

    dash = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts"
    union = re.compile(r"StrategySurface\s*=\s*'action'\s*\|\s*'diagnostic'")
    field = re.compile(r"^\s*surface\?\s*:\s*StrategySurface", re.M)
    for name in ("strategy.contract.ts", "strategy-manifest.contract.ts"):
        src = (dash / name).read_text(encoding="utf-8")
        assert union.search(src), (
            f"{name}: missing `type StrategySurface = 'action' | 'diagnostic'` (C-005)")
        assert field.search(src), (
            f"{name}: missing optional `surface?: StrategySurface` field (C-005)")


def test_registry_champion_matches_manifest():
    reg = json.loads((ROOT / "usdcop-trading-dashboard/public/data/registry.json")
                     .read_text(encoding="utf-8"))
    live = {s["asset_id"]: s["strategy_id"] for s in reg["strategies"]
            if s.get("status") != "archived"}
    for asset, sid in _champions().items():
        if asset in live:
            assert live[asset] == sid, (
                f"{asset}: registry serves {live[asset]!r} but manifest/authority freeze {sid!r}"
            )
