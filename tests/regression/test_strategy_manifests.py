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
        # BL-14 remediacion (rechazo funcional Codex): linaje por componente
        for key in ("spec_fingerprint_sha256_16", "code_reference",
                    "code_hash_sha256_16", "feature_set", "current_model_snapshot"):
            assert key in first, (
                f"{p.name}: components[0] missing lineage field {key!r} (BL-14)")


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


def _load_contract_module():
    """Load src/contracts/strategy_manifest.py by path (the module is a JSON-only leaf;
    importing it via `src.contracts` would eager-import the ML stack)."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "strategy_manifest_under_test", ROOT / "src" / "contracts" / "strategy_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass processing requires the module registered
    spec.loader.exec_module(mod)
    return mod


def _load_normalize_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "normalize_champions_sandbox2",
        ROOT / "scripts" / "pipeline" / "normalize_champions.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dataclasses_reject_unknown_surface_fail_closed():
    """BL-13/C-005 remedio-2 (Codex hallazgo 1): construction is the wall.

    An unknown surface must raise ValueError at CONSTRUCTION time in both
    StrategyBundleManifest and RegistryStrategyEntry; absence/None keeps the
    ACKed legacy semantics (-> "action"). 'banana' -> 'action' must be impossible.
    """
    sm = _load_contract_module()
    base = dict(
        strategy_id="s1", asset_id="usdcop", symbol="USD/COP", chart_symbol="USDCOP",
        display_name="S1", pipeline_type="rule_based", timeframe="weekly", status="paper",
    )
    with pytest.raises(ValueError, match="surface"):
        sm.StrategyBundleManifest(**base, surface="unknown_surface")
    with pytest.raises(ValueError, match="surface"):
        sm.RegistryStrategyEntry(
            strategy_id="s1", asset_id="usdcop", status="paper", display_name="S1",
            pipeline_type="rule_based", timeframe="weekly",
            manifest="strategies/s1/manifest.json", surface="unknown_surface")
    with pytest.raises(ValueError, match="surface"):
        sm.StrategyBundleManifest.from_dict({**base, "surface": "banana"})

    # ACKed legacy semantics: absence (or None) -> "action", still constructs.
    assert sm.StrategyBundleManifest(**base).surface == "action"
    assert sm.StrategyBundleManifest(**base, surface=None).surface == "action"
    assert sm.RegistryStrategyEntry(
        strategy_id="s1", asset_id="usdcop", status="paper", display_name="S1",
        pipeline_type="rule_based", timeframe="weekly",
        manifest="strategies/s1/manifest.json").surface == "action"


def test_registry_builder_raises_on_invalid_surface_never_coerces(tmp_path):
    """BL-13/C-005 remedio-2 (Codex hallazgo 1): RegistryBuilder must RAISE on an
    invalid manifest surface — never silently coerce it to 'action' and serve the
    strategy as tradeable."""
    sm = _load_contract_module()
    bundle = tmp_path / "strategies" / "bad_surface"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text(json.dumps({
        "strategy_id": "bad_surface", "asset_id": "usdcop", "symbol": "USD/COP",
        "chart_symbol": "USDCOP", "display_name": "Bad", "pipeline_type": "rule_based",
        "timeframe": "weekly", "status": "paper", "surface": "unknown_surface",
        "backtests": [], "model_versions": [],
    }), encoding="utf-8")

    builder = sm.RegistryBuilder(tmp_path, generated_at="2026-07-27T00:00:00Z")
    with pytest.raises(ValueError, match="surface"):
        builder.build(write_manifests=False)

    # Sanity: a valid manifest still builds and carries its surface through.
    (bundle / "manifest.json").write_text(json.dumps({
        "strategy_id": "bad_surface", "asset_id": "usdcop", "symbol": "USD/COP",
        "chart_symbol": "USDCOP", "display_name": "Bad", "pipeline_type": "rule_based",
        "timeframe": "weekly", "status": "archived", "surface": "diagnostic",
        "backtests": [], "model_versions": [],
    }), encoding="utf-8")
    idx = builder.build(write_manifests=False)
    assert idx.strategies[0].surface == "diagnostic"


def test_frozen_yaml_invalid_surface_exits_red_in_both_modes(tmp_path, monkeypatch):
    """BL-13/C-005 remedio-2 (Codex hallazgo 1): a frozen YAML whose surface is
    outside {action, diagnostic} is a hard ERROR (exit 1) in BOTH modes — not a
    row _frozen_surfaces silently skips. Enforce mode must not write anything."""
    mod = _load_normalize_module()

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    (frozen / "bad.yaml").write_text(
        yaml.safe_dump({"strategy_id": "x_strat", "surface": "unknown_surface"}),
        encoding="utf-8")

    public = tmp_path / "public"
    bundle = public / "strategies" / "x_strat"
    bundle.mkdir(parents=True)
    manifest_body = json.dumps({
        "strategy_id": "x_strat", "asset_id": "aaa", "symbol": "AAA/USD",
        "chart_symbol": "AAAUSD", "display_name": "X", "pipeline_type": "rule_based",
        "timeframe": "weekly", "status": "experimental", "surface": "action",
        "backtests": [], "model_versions": [],
    })
    (bundle / "manifest.json").write_text(manifest_body, encoding="utf-8")
    (public / "registry.json").write_text(json.dumps({
        "generated_at": "sandbox", "assets": [],
        "strategies": [{"strategy_id": "x_strat", "asset_id": "aaa",
                        "status": "experimental", "surface": "action"}],
        "default": {"asset_id": "aaa", "strategy_id": "x_strat"},
    }), encoding="utf-8")

    monkeypatch.setattr(mod, "FROZEN_MANIFESTS", frozen)
    monkeypatch.setattr(mod, "PUBLIC_DATA", public)
    monkeypatch.setattr(mod, "CHAMPION_BY_ASSET", {"aaa": "x_strat"})

    assert mod.normalize(check_only=True) != 0, (
        "--check must exit red when a frozen YAML declares an unknown surface")
    assert mod.normalize(check_only=False) != 0, (
        "enforce must exit red too: an invalid frozen surface is fixed at the "
        "source, never normalized into silence")
    assert (bundle / "manifest.json").read_text(encoding="utf-8") == manifest_body, (
        "enforce must not rewrite bundles while the frozen authority is invalid")

    # Same sandbox with a VALID frozen surface: both modes go green again.
    (frozen / "bad.yaml").write_text(
        yaml.safe_dump({"strategy_id": "x_strat", "surface": "action"}),
        encoding="utf-8")
    assert mod.normalize(check_only=True) == 0
    assert mod.normalize(check_only=False) == 0


def test_ts_runtime_surface_validator_mirrors_python_whitelist():
    """K-024 remedio-2 (Codex hallazgo 2): parity is SEMANTIC, not just a closed
    union type. The TS contract must expose a RUNTIME validator (pattern of
    policy.contract.ts / forecast-output.contract.ts) whose whitelist is the SAME
    tuple as Python strategy_manifest.SURFACES, rejecting unknown values."""
    import re

    sm = _load_contract_module()
    dash = ROOT / "usdcop-trading-dashboard" / "lib" / "contracts"
    src = (dash / "strategy-manifest.contract.ts").read_text(encoding="utf-8")

    m = re.search(r"STRATEGY_SURFACES\s*=\s*\[([^\]]*)\]\s*as\s*const", src)
    assert m, (
        "strategy-manifest.contract.ts: missing runtime whitelist "
        "`export const STRATEGY_SURFACES = [...] as const` (K-024)")
    ts_values = tuple(re.findall(r"'([^']+)'", m.group(1)))
    assert ts_values == tuple(sm.SURFACES), (
        f"TS runtime whitelist {ts_values} != Python SURFACES {tuple(sm.SURFACES)} — "
        "same values, same order, both sides")

    fn = re.search(
        r"export function validateStrategySurface\s*\(raw: unknown\): string\[\]"
        r"(.*?)\n\}", src, re.S)
    assert fn, (
        "strategy-manifest.contract.ts: missing runtime validator "
        "`export function validateStrategySurface(raw: unknown): string[]` (K-024)")
    assert "STRATEGY_SURFACES" in fn.group(1), (
        "validateStrategySurface must consult the STRATEGY_SURFACES whitelist, "
        "not a re-typed copy")
    assert re.search(r"export function assertStrategySurface", src), (
        "strategy-manifest.contract.ts: missing fail-closed accessor "
        "assertStrategySurface (absence -> 'action', unknown -> throw)")

    # strategy.contract.ts re-exports the SAME validator (single runtime source).
    src2 = (dash / "strategy.contract.ts").read_text(encoding="utf-8")
    assert "validateStrategySurface" in src2 and "strategy-manifest.contract" in src2, (
        "strategy.contract.ts must re-export the runtime surface validator from "
        "strategy-manifest.contract (one whitelist, two entry points)")


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
