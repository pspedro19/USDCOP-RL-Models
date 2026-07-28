"""BL-39 — Feature contracts por estrategia-versión + normalización al artefacto.

Contract: CTR-FEATURE-CATALOG-001 (config/features/**)

What this file enforces (BL-39 Verificación + DATA-STRATEGY §40-48 + Plan Consolidado §1.2-1.4):

1. CI gate (fail-closed): a catalog feature without `causality_policy` or without
   `sign_prior` is REJECTED (Anexo A.4 — "sin prior, la feature no entra al store").
   `ambiguous` priors require an explicit note (legacy ex-post documentation, 0 trials).
   BL-39-r2 (CXD-041): the schema is STRICT — empty strings, non-bool is_active,
   malformed/negative lookback, zero lookback on windowed transforms, and invalid
   code_reference sub-objects (wrong types, unknown keys) are all rejected.
2. Normalization constants NEVER live in the catalog (§1.3 problema 1 — leakage
   silencioso): mean/std belong to a versioned normalization_snapshot artifact.
3. The §41-47 matrix is a CI fixture, split into TWO explicit contracts (CXD-041):
   `usdcop_smart_simple_v11_recipe25` (the recipe, 25) and
   `usdcop_smart_simple_v11_dag_legacy23` (what the H5-L3 DAG persists today, 23).
   The divergence is DECLARED between the two and LOCKED — not resolved (BL-14
   drift_note in config/strategy_manifests/usdcop.yaml). Rule-based champions
   declare their minimal input set (MA200 solo close).
4. The v11 legacy artifacts (scaler/models/feature_cols as-of 2026-07-06) are
   registered bit-identical as `legacy_v1`; when present on disk they must match the
   frozen hashes byte-for-byte (PROTOCOL §6: reproducción EXACTA antes y después).
5. Bit-check: the v11 signal of the last artifact week is reproduced from the
   DECLARED feature_set + normalization_snapshot and must be bit-identical to the
   as-built H5-L5b pipeline path (scripts/validation/bitcheck_v11_signal.py).

HASH METHOD (CXD-041/043): hashes over SOURCE files use the canonical LF form
(CRLF->LF normalized bytes == git blob under .gitattributes text eol=lf), so a
clean checkout on any OS reproduces them. Hashes over binary/data ARTIFACTS
(outputs/*.pkl, feature_cols_h5.json) stay raw — bit-identity, no normalization.

CI vs LOCAL-ONLY (CXD-041 punto 7, declared in the catalog's `ci_gates`): all
declaration-level tests here run in CI; the bit-identity tests skip cleanly
when outputs/** (gitignored) is absent and only verify where the frozen
pipeline lives. The bitcheck CLI declares the same: --require-artifacts exits
2 with "artefactos locales requeridos; verificación local-only".

NOTE (frozen wall): v11 is FROZEN. This BL adds declarations only — zero runtime
changes, zero trials. Reconciling 25 vs 23 requires its own BL + conscious re-freeze.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = ROOT / "config" / "features"
CATALOG_PATH = FEATURES_DIR / "feature_catalog.yaml"
SETS_DIR = FEATURES_DIR / "feature_sets"
SNAPS_DIR = FEATURES_DIR / "normalization_snapshots"
MANIFEST_PATH = ROOT / "config" / "strategy_manifests" / "usdcop.yaml"
ARTIFACTS_DIR = ROOT / "outputs" / "forecasting" / "h5_weekly_models" / "latest"

# ── Frozen fixture: the §41-47 matrix, pinned (BL-39) ────────────────────────
# 21 base (dataset_loader SSOT) — order is part of the freeze.
BASE_21 = [
    "close", "open", "high", "low",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
    "day_of_week", "month", "is_month_end",
    "dxy_close_lag1", "oil_close_lag1", "vix_close_lag1", "embi_close_lag1",
]
# What the H5-L3 DAG actually persisted (feature_cols_h5.json, as-of 2026-07-06).
SNAPSHOT_23 = BASE_21 + ["vol_regime_ratio", "trend_slope_60d"]
# What the export/backtest recipe builds (enhance_features_v2, macro merge OK).
RECIPE_25 = SNAPSHOT_23 + ["rate_diff_ibr_ust2y", "term_spread"]

# Bit-identity anchors (== manifest components[0].current_model_snapshot as-of 2026-07-06).
FEATURE_COLS_JSON_SHA16 = "c3393242ef998896"
SCALER_SHA16 = "3302221e2b9dee39"

RULE_BASED_MINIMAL = {
    # asset -> (strategy_id, minimal ordered input features)  — §45-47: "MA200 solo close"
    "xauusd": ("gold_trend_simple", ["close"]),
    "btcusdt": ("btc_hodl_b1", ["close"]),
    "spx500": ("spx500_regime_gated_v1", ["close"]),
}


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_feature_catalog",
        ROOT / "scripts" / "validation" / "validate_feature_catalog.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _catalog() -> dict:
    assert CATALOG_PATH.is_file(), f"missing feature catalog at {CATALOG_PATH} (BL-39)"
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))


def _feature_set(name: str) -> dict:
    p = SETS_DIR / f"{name}.yaml"
    assert p.is_file(), f"missing feature set {p} (BL-39)"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def _sha16_raw(path: Path) -> str:
    """Raw-byte hash — ONLY for binary/data artifacts (bit-identity)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _sha16_lf(path: Path) -> str:
    """Hash canónico LF for SOURCE files: CRLF->LF normalized bytes, equal to
    hashing the git blob (`git show :<path>`) on any OS (CXD-041/043)."""
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════
# 1. CI gate: fail-closed catalog validation (BL-39 Verificación / Anexo A.4)
# ═══════════════════════════════════════════════════════════════════════════

def _valid_entry(**overrides) -> dict:
    entry = {
        "feature_id": "synthetic_ok",
        "unit": "decimal",
        "feature_group": "returns",
        "causality_policy": "same_bar",
        "source_contract": "market.canonical_bar",
        "transformation": "pct_change",
        "lookback": "P1D",
        "compute_location": "python",
        "code_reference": None,
        "sign_prior": "positive",
        "sign_prior_note": "synthetic",
        "is_active": True,
    }
    entry.update(overrides)
    return entry


def test_gate_rejects_feature_without_causality_policy():
    v = _load_validator()
    entry = _valid_entry()
    del entry["causality_policy"]
    errors = v.validate_entries([entry])
    assert errors and any("causality_policy" in e for e in errors), (
        "a feature without causality_policy must be rejected (BL-39/Anexo A.4), "
        f"got errors={errors}")


def test_gate_rejects_unknown_causality_policy():
    v = _load_validator()
    errors = v.validate_entries([_valid_entry(causality_policy="whenever")])
    assert errors and any("causality_policy" in e for e in errors)


def test_gate_rejects_feature_without_sign_prior():
    v = _load_validator()
    entry = _valid_entry()
    del entry["sign_prior"]
    errors = v.validate_entries([entry])
    assert errors and any("sign_prior" in e for e in errors), (
        "sin prior de signo la feature no entra al store (Anexo A.4), "
        f"got errors={errors}")


def test_gate_rejects_ambiguous_prior_without_note():
    v = _load_validator()
    errors = v.validate_entries(
        [_valid_entry(sign_prior="ambiguous", sign_prior_note=None)])
    assert errors and any("sign_prior_note" in e for e in errors), (
        "an 'ambiguous' prior without an explicit note makes the gate toothless")


def test_gate_rejects_normalization_constants_in_catalog():
    """§1.3 problema 1: mean/std in the catalog = silent leakage on retrain."""
    v = _load_validator()
    for bad_key in ("normalization_mean", "normalization_std"):
        errors = v.validate_entries([_valid_entry(**{bad_key: 21.16})])
        assert errors and any("normalization" in e for e in errors), (
            f"catalog entry carrying {bad_key} must be rejected — normalization "
            "belongs to the versioned snapshot artifact, never the catalog")


def test_real_catalog_passes_the_gate():
    v = _load_validator()
    cat = _catalog()
    errors = v.validate_entries(cat["features"])
    assert errors == [], f"feature catalog has violations: {errors}"


# ── BL-39-r2 (CXD-041): strict schema — the validator must reject garbage ───

def test_gate_rejects_empty_strings():
    """Fail-first CXD-041: an all-empty-strings entry used to raise only ONE
    violation (missing code_reference). Empty is not present."""
    v = _load_validator()
    entry = _valid_entry(feature_id="", unit="", feature_group="",
                         source_contract="", transformation="", lookback="",
                         compute_location="")
    errors = v.validate_entries([entry])
    for field in ("feature_id", "unit", "feature_group", "source_contract",
                  "transformation", "lookback", "compute_location"):
        assert any(f"{field!r}" in e and "non-empty" in e for e in errors), (
            f"empty string for {field} must be rejected, got errors={errors}")


def test_gate_rejects_negative_and_malformed_lookback():
    v = _load_validator()
    for bad in ("P-5D", "-P5D", "5D", "P5", "P5W", "PXD"):
        errors = v.validate_entries([_valid_entry(lookback=bad)])
        assert any("lookback" in e for e in errors), (
            f"lookback {bad!r} must be rejected, got errors={errors}")
    # ints are not ISO-8601 durations either
    errors = v.validate_entries([_valid_entry(lookback=-5)])
    assert any("lookback" in e for e in errors)


def test_gate_rejects_zero_lookback_on_windowed_transform():
    v = _load_validator()
    errors = v.validate_entries(
        [_valid_entry(transformation="rolling_std(return_1d)", lookback="P0D")])
    assert any("P0D" in e for e in errors), (
        f"a windowed transformation with zero lookback must be rejected, got {errors}")
    # ...but P0D stays legal for transformations that need no history.
    ok = v.validate_entries(
        [_valid_entry(transformation="identity", lookback="P0D",
                      code_reference=None)])
    assert ok == [], f"identity + P0D must pass, got {ok}"


def test_gate_rejects_wrong_types():
    v = _load_validator()
    errors = v.validate_entries([_valid_entry(is_active="yes")])
    assert any("is_active" in e and "bool" in e for e in errors), (
        f"is_active must be a real bool, got errors={errors}")
    errors = v.validate_entries([_valid_entry(unit=42)])
    assert any("'unit'" in e for e in errors)


def test_gate_validates_code_reference_recursively():
    """CXD-041: sub-objects get schema'd too — not just presence of keys."""
    v = _load_validator()
    good_file = "src/forecasting/dataset_loader.py"
    # int sha256_16
    errors = v.validate_entries([_valid_entry(
        code_reference={"file": good_file, "sha256_16": 12345})])
    assert any("sha256_16" in e for e in errors)
    # malformed sha (wrong length / non-hex)
    errors = v.validate_entries([_valid_entry(
        code_reference={"file": good_file, "sha256_16": "XYZ"})])
    assert any("sha256_16" in e for e in errors)
    # unknown keys are rejected
    errors = v.validate_entries([_valid_entry(
        code_reference={"file": good_file, "sha256_16": "937624aba7cd4f18",
                        "extra_key": "??"})])
    assert any("unknown keys" in e for e in errors)
    # empty file
    errors = v.validate_entries([_valid_entry(
        code_reference={"file": "", "sha256_16": "937624aba7cd4f18"})])
    assert any("file" in e for e in errors)
    # non-dict code_reference
    errors = v.validate_entries([_valid_entry(code_reference="a string")])
    assert any("mapping" in e for e in errors)


def test_code_hash_method_is_line_ending_invariant(tmp_path, monkeypatch):
    """CXD-041/043 regression: the validator's source hash must be identical for
    a CRLF checkout (Windows) and an LF checkout (Linux / git blob).

    Red with the old raw-byte method; green with canonical CRLF->LF."""
    v = _load_validator()
    content_lf = b"def f():\n    return 1\n"
    f_lf = tmp_path / "lf.py"
    f_crlf = tmp_path / "crlf.py"
    f_lf.write_bytes(content_lf)
    f_crlf.write_bytes(content_lf.replace(b"\n", b"\r\n"))
    # Old method (raw) is OS-dependent — the CXD-041 bug:
    assert (hashlib.sha256(f_lf.read_bytes()).hexdigest()
            != hashlib.sha256(f_crlf.read_bytes()).hexdigest())
    # Canonical method is EOL-invariant and equals the git-blob (LF) hash:
    assert v._sha16(f_crlf) == v._sha16(f_lf) == hashlib.sha256(
        content_lf).hexdigest()[:16]


def test_validator_cli_is_a_ci_gate():
    """CXD-041 punto 7: what IS verifiable without gitignored artifacts runs as
    a normal test — the CLI gate must exit 0 on the real catalog."""
    v = _load_validator()
    assert v.main() == 0, "validate_feature_catalog CLI gate must pass in CI"


def test_bitcheck_declares_local_only_and_exits_distinctly(tmp_path):
    """CXD-041 punto 7: the bit-check needs gitignored outputs/** so it is
    NO-CI. With --require-artifacts and no artifacts it must fail CLEANLY with
    a DISTINCT exit code (2) and the declared message; without the flag it
    declares a SKIP (0). This test runs anywhere (uses an empty root)."""
    import subprocess
    import sys as _sys
    script = ROOT / "scripts" / "validation" / "bitcheck_v11_signal.py"
    r = subprocess.run(
        [_sys.executable, str(script), "--require-artifacts", "--root", str(tmp_path)],
        capture_output=True, text=True)
    assert r.returncode == 2, (
        f"missing artifacts + --require-artifacts must exit 2 (distinct), got "
        f"{r.returncode}: {r.stdout} {r.stderr}")
    assert "local-only" in r.stdout or "verificación local-only" in r.stdout
    r = subprocess.run(
        [_sys.executable, str(script), "--root", str(tmp_path)],
        capture_output=True, text=True)
    assert r.returncode == 0 and "[SKIP]" in r.stdout, (
        f"missing artifacts without the flag must SKIP with exit 0, got "
        f"{r.returncode}: {r.stdout} {r.stderr}")


def test_catalog_code_references_exist_and_hashes_match():
    """§40.1: code_reference + code_hash — editing the computing code without
    re-registering the catalog is drift and must go red."""
    cat = _catalog()
    seen: dict[str, str] = {}
    for f in cat["features"]:
        ref = f.get("code_reference")
        if not ref:
            assert f.get("transformation") == "identity", (
                f"{f['feature_id']}: only identity features may omit code_reference")
            continue
        path, recorded = ref["file"], ref["sha256_16"]
        assert (ROOT / path).is_file(), f"{f['feature_id']}: missing code file {path}"
        current = seen.setdefault(path, _sha16_lf(ROOT / path))
        assert current == recorded, (
            f"{f['feature_id']}: {path} drifted from the registered catalog hash "
            f"(registered={recorded}, current={current}). Re-register consciously: "
            "update the hash + note; if the SEMANTICS changed, that is a new feature "
            "version, not an edit (BL-39)")


# ═══════════════════════════════════════════════════════════════════════════
# 2. §41-47 matrix as CI fixture — recipe25 vs dag_legacy23, drift DECLARED
#    between TWO explicit contracts (CXD-041 split)
# ═══════════════════════════════════════════════════════════════════════════

def test_v11_recipe25_contract():
    fs = _feature_set("usdcop_smart_simple_v11_recipe25")
    assert fs["feature_set_id"] == "usdcop_smart_simple_v11_recipe25"
    ordered = [f["feature_id"] for f in fs["ordered_features"]]
    assert ordered == RECIPE_25, (
        "v11 recipe feature set must be EXACTLY the frozen 25 (order included); "
        f"got {ordered}")
    orders = [f["order"] for f in fs["ordered_features"]]
    assert orders == list(range(25)), "order must be dense 0..24"
    assert all(f["required"] is True for f in fs["ordered_features"])

    cat_ids = {f["feature_id"] for f in _catalog()["features"]}
    missing = [f for f in ordered if f not in cat_ids]
    assert not missing, f"feature_set references features absent from catalog: {missing}"

    # §42.5: v12/v14 share v11's decision input — declared, so comparison is paired.
    assert set(fs.get("shared_by", [])) == {"smart_simple_v12", "smart_simple_v14"}, (
        "v12/v14 must reference the same decision-input feature set as v11 (§42.5)")

    # The recipe has NO registered normalization snapshot (the production scaler
    # was fit on the DAG's 23 — it belongs to the dag_legacy23 contract).
    assert fs["normalization_snapshot_id"] is None
    assert "usdcop_h5_scaler_legacy_v1" in str(fs.get("normalization_note", "")), (
        "the recipe must point at where the persisted scaler actually lives")

    # Divergence declared toward the sibling contract.
    div = fs["divergence"]
    assert div["status"] == "declared_not_resolved"
    assert div["counterpart_feature_set_id"] == "usdcop_smart_simple_v11_dag_legacy23"
    assert div["missing_in_counterpart"] == ["rate_diff_ibr_ust2y", "term_spread"]


def test_v11_dag_legacy23_contract_locked_not_resolved():
    """BL-14 declared a REAL divergence (recipe 25 vs DAG snapshot 23). CXD-041:
    the divergence lives BETWEEN two explicit contracts, not inside one. This
    test LOCKS the dag_legacy23 side — resolving it silently is prohibited
    (0 trials)."""
    fs = _feature_set("usdcop_smart_simple_v11_dag_legacy23")
    assert fs["feature_set_id"] == "usdcop_smart_simple_v11_dag_legacy23"
    ordered = [f["feature_id"] for f in fs["ordered_features"]]
    assert ordered == SNAPSHOT_23, (
        "dag_legacy23 must be the exact 23-column list of feature_cols_h5.json")
    orders = [f["order"] for f in fs["ordered_features"]]
    assert orders == list(range(23)), "order must be dense 0..22"
    assert fs["file_sha256_16"] == FEATURE_COLS_JSON_SHA16

    drift = fs["drift"]
    assert drift["status"] == "declared_not_resolved"
    assert drift["counterpart_feature_set_id"] == "usdcop_smart_simple_v11_recipe25"
    assert drift["missing_vs_recipe"] == ["rate_diff_ibr_ust2y", "term_spread"]
    assert "BL" in str(drift["resolution"]), (
        "resolution must defer to its own BL + conscious re-freeze")

    # The scaler contract hangs off THIS side (fit on the 23).
    assert fs["normalization_snapshot_id"] == "usdcop_h5_scaler_legacy_v1"

    # Cross-lock with the frozen manifest (BL-14): ids, numbers and hash agree.
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    comp_fs = manifest["components"][0]["feature_set"]
    assert comp_fs["recipe_feature_set_id"] == "usdcop_smart_simple_v11_recipe25"
    assert comp_fs["dag_snapshot_feature_set_id"] == "usdcop_smart_simple_v11_dag_legacy23"
    assert comp_fs["recipe_n_features"] == len(RECIPE_25) == 25
    assert comp_fs["snapshot_n_features"] == len(SNAPSHOT_23) == 23
    assert "drift_note" in comp_fs, "manifest drift_note is the SSOT declaration"
    manifest_hash = (manifest["components"][0]["current_model_snapshot"]
                     ["artifacts_sha256_16"]["feature_cols_h5.json"])
    assert manifest_hash == FEATURE_COLS_JSON_SHA16, (
        "feature_set pin and manifest as-of hash must be the same bytes")

    # Consistency: snapshot is a strict prefix of the recipe (enhance_v2 append order).
    assert RECIPE_25[:23] == SNAPSHOT_23


def test_rule_based_champions_declare_minimal_sets():
    """§45-47 fixture: rule-based strategies declare their minimal input set
    (MA200 solo close); indicators are derived inside frozen policy code."""
    for asset, (sid, minimal) in RULE_BASED_MINIMAL.items():
        fs = _feature_set(sid)
        assert fs["strategy_id"] == sid
        assert fs["asset_id"] == asset
        ordered = [f["feature_id"] for f in fs["ordered_features"]]
        assert ordered == minimal, (
            f"{sid}: minimal input set must be {minimal}, got {ordered}")
        assert fs.get("normalization_snapshot_id") is None, (
            f"{sid}: rule-based, retrain never — no normalization snapshot")
        assert fs.get("derived_in_policy"), (
            f"{sid}: must declare which indicators are derived inside the frozen "
            "policy code (SMA votes / MA200 / vol targeting)")
        # The declared strategy must be the frozen manifest's champion strategy.
        m = yaml.safe_load(
            (ROOT / "config" / "strategy_manifests" / f"{asset}.yaml")
            .read_text(encoding="utf-8"))
        assert m["strategy_id"] == sid


# ═══════════════════════════════════════════════════════════════════════════
# 3. Normalization → versioned snapshot artifact (never the catalog)
# ═══════════════════════════════════════════════════════════════════════════

def test_v11_normalization_snapshot_registered_as_legacy_v1():
    fs = _feature_set("usdcop_smart_simple_v11_dag_legacy23")
    snap_id = fs["normalization_snapshot_id"]
    assert snap_id == "usdcop_h5_scaler_legacy_v1"
    p = SNAPS_DIR / f"{snap_id}.yaml"
    assert p.is_file(), f"missing normalization snapshot registration {p}"
    snap = yaml.safe_load(p.read_text(encoding="utf-8"))

    assert snap["normalization_snapshot_id"] == snap_id
    # CXD-041 split: the snapshot references the DAG contract (scaler fit on 23),
    # never the recipe25 one.
    assert snap["feature_set_id"] == "usdcop_smart_simple_v11_dag_legacy23"
    for key in ("artifact", "training", "ordered_feature_hash_sha256_16",
                "semantic_hash_sha256_16"):
        assert key in snap, f"normalization snapshot missing {key}"
    assert snap["artifact"]["sha256_16"] == SCALER_SHA16, (
        "legacy_v1 must register the AS-IS scaler artifact bit-identical")
    assert snap["ordered_feature_hash_sha256_16"] == FEATURE_COLS_JSON_SHA16
    assert snap["n_features"] == 23, (
        "the scaler was fit on the DAG snapshot (23), not the recipe (25) — that IS "
        "the declared drift; do not 'fix' this number here")
    # Rotation is legitimate (recipe frozen, weights rotate): the record must say so.
    assert "rotat" in str(snap.get("rotation_policy", "")).lower(), (
        "snapshot must declare the rotating-pointer policy (each L3 run = NEW snapshot id)")


def test_v11_normalization_snapshot_declares_immutability_honestly():
    """CXD-041 punto 5: the artifact pointer `latest/` is MUTABLE. An immutable
    run URI must be declared if one exists; since only `latest/` exists locally
    (no per-run outputs dir, no resolvable MLflow run_id in this repo), the
    honest declaration is run_uri: null + an explicit mutable_pointer_caveat
    whose identity is the registered hashes — never an invented URI."""
    snap = yaml.safe_load(
        (SNAPS_DIR / "usdcop_h5_scaler_legacy_v1.yaml").read_text(encoding="utf-8"))
    art = snap["artifact"]
    assert "run_uri" in art, (
        "snapshot must declare run_uri (immutable run) or null with a caveat")
    if art["run_uri"] is None:
        caveat = art.get("mutable_pointer_caveat", "")
        assert caveat, "run_uri null requires an explicit mutable_pointer_caveat"
        # The caveat must anchor identity to the registered hashes.
        assert art["sha256_16"] in caveat and snap["semantic_hash_sha256_16"] in caveat, (
            "the caveat must state that the HASHES are the identity of the snapshot")
    else:
        assert str(art["run_uri"]).strip(), "run_uri must be a non-empty URI"


def test_catalog_has_no_normalization_constants_anywhere():
    raw = CATALOG_PATH.read_text(encoding="utf-8")
    for token in ("normalization_mean", "normalization_std", "zscore_fixed"):
        assert token not in raw, (
            f"catalog contains {token!r} — normalization lives in the snapshot "
            "artifact, never the catalog (§1.3, BL-39 razón de ser)")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Bit-identity with the frozen pipeline (local artifacts; PROTOCOL §6)
# ═══════════════════════════════════════════════════════════════════════════

def _require_artifacts():
    if not (ARTIFACTS_DIR / "feature_cols_h5.json").is_file():
        pytest.skip("H5 model artifacts not present (outputs/ is gitignored) — "
                    "bit-identity checks run where the frozen pipeline lives")


def test_disk_artifacts_are_bit_identical_to_frozen_registration():
    _require_artifacts()
    disk_cols = json.loads(
        (ARTIFACTS_DIR / "feature_cols_h5.json").read_text(encoding="utf-8"))
    assert disk_cols == SNAPSHOT_23, (
        "feature_cols_h5.json on disk no longer matches the registered legacy_v1 "
        "snapshot — if H5-L3 reconciled to 25 features, that is a conscious "
        "re-freeze event with its own BL, not a silent pass")
    assert _sha16_raw(ARTIFACTS_DIR / "feature_cols_h5.json") == FEATURE_COLS_JSON_SHA16
    assert _sha16_raw(ARTIFACTS_DIR / "scaler_h5.pkl") == SCALER_SHA16
    manifest = yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))
    expected = manifest["components"][0]["current_model_snapshot"]["artifacts_sha256_16"]
    for fname in ("ridge_h5.pkl", "bayesian_ridge_h5.pkl"):
        assert _sha16_raw(ARTIFACTS_DIR / fname) == expected[fname], (
            f"{fname} drifted from the manifest as-of hashes — models rotated: "
            "register a NEW normalization/model snapshot instead of mutating legacy_v1")


def test_bitcheck_v11_signal_from_feature_set_plus_snapshot():
    """BL-39 Verificación: reproduce the v11 signal of the last (artifact) week from
    feature_set + normalization snapshot == bit-check vs the as-built L5b path."""
    _require_artifacts()
    pred_path = ROOT / "outputs" / "forecasting" / "h5_l5a_pred_features_temp.parquet"
    if not pred_path.is_file():
        pytest.skip("h5_l5a_pred_features_temp.parquet not present locally")

    spec = importlib.util.spec_from_file_location(
        "bitcheck_v11_signal",
        ROOT / "scripts" / "validation" / "bitcheck_v11_signal.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    report = mod.run_bitcheck(ROOT)
    assert report["ok"], f"bit-check FAILED: {report['errors']}"
    # Bit-exact, not approximately equal.
    assert report["feature_row_bit_identical"] is True
    assert report["predictions_bit_identical"] is True
    assert report["ensemble_return_bit_identical"] is True
