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
