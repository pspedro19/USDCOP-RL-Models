"""Feature catalog gate — CTR-FEATURE-CATALOG-001 (BL-39).

Fail-closed validator for config/features/feature_catalog.yaml:

- `causality_policy` is MANDATORY and must be one of the declared policies
  (point_in_time | same_bar | lagged_1). A feature whose availability rule is
  undeclared cannot be audited for look-ahead (quant-constitution §4).
- `sign_prior` is MANDATORY (Anexo A.4: "sin prior, la feature no entra al
  store"). Allowed: positive | negative | ambiguous; `ambiguous` REQUIRES a
  sign_prior_note (otherwise the gate is toothless).
- Normalization constants (`normalization_mean`/`normalization_std`, or the
  `zscore_fixed` pattern) are PROHIBITED in the catalog — they belong to a
  versioned normalization snapshot artifact (§40.3, Plan Consolidado §1.3).
- `code_reference` (file + sha256_16) is mandatory except for `identity`
  passthrough features; referenced files must exist.

Used by tests/regression/test_feature_contracts.py and runnable as a CLI gate:

    python scripts/validation/validate_feature_catalog.py
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "config" / "features" / "feature_catalog.yaml"

CAUSALITY_POLICIES = ("point_in_time", "same_bar", "lagged_1")
SIGN_PRIORS = ("positive", "negative", "ambiguous")
PROHIBITED_KEYS = ("normalization_mean", "normalization_std", "zscore_fixed")
REQUIRED_KEYS = (
    "feature_id", "unit", "feature_group", "causality_policy",
    "source_contract", "transformation", "lookback", "compute_location",
    "sign_prior", "is_active",
)


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def validate_entries(features: list[dict]) -> list[str]:
    """Validate catalog entries. Returns a list of violations ([] = pass)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, f in enumerate(features):
        fid = f.get("feature_id") or f"<entry #{i}>"
        if fid in seen_ids:
            errors.append(f"{fid}: duplicate feature_id")
        seen_ids.add(fid)

        for key in REQUIRED_KEYS:
            if key not in f or f[key] is None:
                errors.append(f"{fid}: missing mandatory field {key!r}")

        cp = f.get("causality_policy")
        if cp is not None and cp not in CAUSALITY_POLICIES:
            errors.append(
                f"{fid}: unknown causality_policy {cp!r} "
                f"(allowed: {CAUSALITY_POLICIES})")

        sp = f.get("sign_prior")
        if sp is not None and sp not in SIGN_PRIORS:
            errors.append(f"{fid}: unknown sign_prior {sp!r} (allowed: {SIGN_PRIORS})")
        if sp == "ambiguous" and not f.get("sign_prior_note"):
            errors.append(
                f"{fid}: sign_prior 'ambiguous' requires an explicit "
                "sign_prior_note (Anexo A.4)")

        for bad in PROHIBITED_KEYS:
            if bad in f or bad in str(f.get("transformation", "")):
                errors.append(
                    f"{fid}: normalization constant/pattern {bad!r} in catalog — "
                    "normalization belongs to the versioned snapshot artifact "
                    "(§40.3), never the catalog")

        ref = f.get("code_reference")
        if ref is None:
            if f.get("transformation") != "identity":
                errors.append(
                    f"{fid}: code_reference is mandatory for non-identity "
                    "transformations (§40.1)")
        else:
            if not isinstance(ref, dict) or "file" not in ref or "sha256_16" not in ref:
                errors.append(f"{fid}: code_reference must declare file + sha256_16")
            elif not (ROOT / ref["file"]).is_file():
                errors.append(f"{fid}: code_reference file not found: {ref['file']}")
    return errors


def validate_code_hashes(features: list[dict]) -> list[str]:
    """Verify recorded code hashes against the working tree (drift detection)."""
    errors: list[str] = []
    cache: dict[str, str] = {}
    for f in features:
        ref = f.get("code_reference")
        if not isinstance(ref, dict) or "file" not in ref:
            continue
        path = ref["file"]
        target = ROOT / path
        if not target.is_file():
            continue  # already reported by validate_entries
        current = cache.setdefault(path, _sha16(target))
        if current != ref.get("sha256_16"):
            errors.append(
                f"{f.get('feature_id')}: {path} drifted "
                f"(registered={ref.get('sha256_16')}, current={current}) — "
                "re-register consciously; a SEMANTIC change is a new feature version")
    return errors


def main() -> int:
    if not CATALOG_PATH.is_file():
        print(f"[FAIL] missing catalog: {CATALOG_PATH}")
        return 1
    cat = yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))
    features = cat.get("features") or []
    errors = validate_entries(features) + validate_code_hashes(features)
    if errors:
        print(f"[FAIL] feature catalog: {len(errors)} violation(s)")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"[OK] feature catalog: {len(features)} features, 0 violations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
