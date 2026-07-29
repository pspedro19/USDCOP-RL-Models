"""Feature catalog gate — CTR-FEATURE-CATALOG-001 (BL-39, endurecido BL-39-r2 / CXD-041).

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
- STRICT SCHEMA (CXD-041): required string fields must be NON-EMPTY strings
  (not just present), `is_active` must be a real bool, `lookback` must be an
  ISO-8601 day duration `P<n>D` with n >= 0 — negative is unrepresentable and
  rejected as a format error, and n == 0 is only legal for transformations
  that genuinely need no history (identity / calendar_extract). Sub-objects
  are validated recursively: `code_reference` must be a dict whose `file` is
  a non-empty string, whose `sha256_16` matches ^[0-9a-f]{16}$ (a string, not
  an int), whose optional `symbol` is a non-empty string, and which carries
  no unknown keys.

HASHING (CXD-041/043 root cause): every `sha256_16` over a SOURCE FILE is the
hash canónico LF — the file bytes with CRLF normalized to LF
(`data.replace(b"\r\n", b"\n")`), equivalent to the git blob under
`.gitattributes` `* text=auto eol=lf`. Reproducible from `git show :<path>`
on any OS. Hashing raw working-tree bytes (the old method) baked Windows CRLF
into the declared hashes, so a clean Linux checkout could not reproduce them.

CI vs local-only: THIS validator and tests/regression/test_feature_contracts.py
are the CI gate (declaration-level, no gitignored inputs needed). The v11
bit-check (scripts/validation/bitcheck_v11_signal.py) needs outputs/*.pkl
(gitignored) and is LOCAL-ONLY by construction — see `ci_gates` in the catalog.

Used by tests/regression/test_feature_contracts.py and runnable as a CLI gate:

    python scripts/validation/validate_feature_catalog.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
# Runnable as a standalone CLI (`python scripts/validation/validate_feature_catalog.py`):
# sys.path[0] is then this script's directory, so `src.` would not resolve without this.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.identity.source_hash import canonical_lf, file_code_hash  # noqa: E402

CATALOG_PATH = ROOT / "config" / "features" / "feature_catalog.yaml"

CAUSALITY_POLICIES = ("point_in_time", "same_bar", "lagged_1")
SIGN_PRIORS = ("positive", "negative", "ambiguous")
PROHIBITED_KEYS = ("normalization_mean", "normalization_std", "zscore_fixed")
REQUIRED_KEYS = (
    "feature_id", "unit", "feature_group", "causality_policy",
    "source_contract", "transformation", "lookback", "compute_location",
    "sign_prior", "is_active",
)
# Required keys whose value must be a NON-EMPTY string.
REQUIRED_STR_KEYS = (
    "feature_id", "unit", "feature_group", "causality_policy",
    "source_contract", "transformation", "lookback", "compute_location",
    "sign_prior",
)
# code_reference sub-schema (recursive validation, CXD-041).
CODE_REF_REQUIRED = ("file", "sha256_16")
CODE_REF_OPTIONAL = ("symbol",)
SHA16_RE = re.compile(r"^[0-9a-f]{16}$")
LOOKBACK_RE = re.compile(r"^P(\d+)D$")
# Transformations that legitimately need zero bars of history.
ZERO_LOOKBACK_TRANSFORMS = ("identity", "calendar_extract")


# El método de hashing NO se re-implementa aquí: vive en src/identity/source_hash.py,
# la única implementación de producción (BL-13 red-team — dos implementaciones del mismo
# concepto significan que una miente). Estos alias conservan los nombres históricos que
# ya consumen tests/regression/test_feature_contracts.py y el CLI.
_canonical_lf = canonical_lf


def _sha16(path: Path) -> str:
    """sha256[:16] del fichero con bytes CRLF->LF normalizados (hash canónico LF).

    Delegado a `src.identity.source_hash.file_code_hash`: reproducible desde el
    blob git en cualquier OS (equivalente a `git show :<path> | sha256sum` con
    .gitattributes text eol=lf). CXD-041/043: hashear los bytes crudos del working
    tree fijaba los CRLF de Windows en el hash declarado y un checkout limpio en
    Linux no lo reproducía.
    """
    return file_code_hash(path)


def _validate_code_reference(fid: str, ref: object, errors: list[str]) -> None:
    """Recursive sub-object schema for code_reference (CXD-041 strictness)."""
    if not isinstance(ref, dict):
        errors.append(f"{fid}: code_reference must be a mapping, got {type(ref).__name__}")
        return
    for key in CODE_REF_REQUIRED:
        if key not in ref:
            errors.append(f"{fid}: code_reference missing required key {key!r}")
    unknown = set(ref) - set(CODE_REF_REQUIRED) - set(CODE_REF_OPTIONAL)
    if unknown:
        errors.append(
            f"{fid}: code_reference has unknown keys {sorted(unknown)} "
            f"(allowed: {CODE_REF_REQUIRED + CODE_REF_OPTIONAL})")
    file_ = ref.get("file")
    if "file" in ref and (not isinstance(file_, str) or not file_.strip()):
        errors.append(f"{fid}: code_reference.file must be a non-empty string")
    elif isinstance(file_, str) and file_.strip() and not (ROOT / file_).is_file():
        errors.append(f"{fid}: code_reference file not found: {file_}")
    sha = ref.get("sha256_16")
    if "sha256_16" in ref and (not isinstance(sha, str) or not SHA16_RE.match(sha)):
        errors.append(
            f"{fid}: code_reference.sha256_16 must be a 16-char lowercase hex "
            f"STRING, got {sha!r}")
    sym = ref.get("symbol")
    if "symbol" in ref and (not isinstance(sym, str) or not sym.strip()):
        errors.append(f"{fid}: code_reference.symbol must be a non-empty string")


def validate_entries(features: list[dict]) -> list[str]:
    """Validate catalog entries. Returns a list of violations ([] = pass)."""
    errors: list[str] = []
    seen_ids: set[str] = set()
    for i, f in enumerate(features):
        fid = f.get("feature_id") or f"<entry #{i}>"
        if not isinstance(f, dict):
            errors.append(f"<entry #{i}>: entry must be a mapping")
            continue
        if fid in seen_ids:
            errors.append(f"{fid}: duplicate feature_id")
        seen_ids.add(fid)

        for key in REQUIRED_KEYS:
            if key not in f or f[key] is None:
                errors.append(f"{fid}: missing mandatory field {key!r}")

        # Strict types: required strings must be non-empty strings (CXD-041).
        for key in REQUIRED_STR_KEYS:
            val = f.get(key)
            if val is None:
                continue  # already reported as missing
            if not isinstance(val, str) or not val.strip():
                errors.append(
                    f"{fid}: field {key!r} must be a non-empty string, got {val!r}")

        ia = f.get("is_active")
        if ia is not None and not isinstance(ia, bool):
            errors.append(f"{fid}: is_active must be a bool, got {ia!r}")

        cp = f.get("causality_policy")
        if isinstance(cp, str) and cp and cp not in CAUSALITY_POLICIES:
            errors.append(
                f"{fid}: unknown causality_policy {cp!r} "
                f"(allowed: {CAUSALITY_POLICIES})")

        sp = f.get("sign_prior")
        if isinstance(sp, str) and sp and sp not in SIGN_PRIORS:
            errors.append(f"{fid}: unknown sign_prior {sp!r} (allowed: {SIGN_PRIORS})")
        if sp == "ambiguous" and not f.get("sign_prior_note"):
            errors.append(
                f"{fid}: sign_prior 'ambiguous' requires an explicit "
                "sign_prior_note (Anexo A.4)")

        # Lookback: ISO-8601 day duration, n >= 0; negative/malformed rejected;
        # zero only for transformations that need no history (CXD-041).
        lb = f.get("lookback")
        if isinstance(lb, str) and lb:
            m = LOOKBACK_RE.match(lb)
            if not m:
                errors.append(
                    f"{fid}: lookback {lb!r} is not a valid ISO-8601 day duration "
                    "P<n>D with n >= 0 (negative/zero-padded/malformed rejected)")
            elif int(m.group(1)) == 0 and f.get("transformation") not in ZERO_LOOKBACK_TRANSFORMS:
                errors.append(
                    f"{fid}: lookback P0D with transformation "
                    f"{f.get('transformation')!r} — a windowed transformation "
                    "cannot have zero lookback (declare the real window)")
        elif lb is not None and not isinstance(lb, str):
            errors.append(f"{fid}: lookback must be a string, got {lb!r}")

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
            _validate_code_reference(fid, ref, errors)
    return errors


def validate_code_hashes(features: list[dict]) -> list[str]:
    """Verify recorded code hashes against the working tree (drift detection).

    Uses the canonical LF hash (`_sha16`) so the same declared value verifies
    on Windows (CRLF working tree) and on a clean Linux checkout (LF blobs).
    """
    errors: list[str] = []
    cache: dict[str, str] = {}
    for f in features:
        ref = f.get("code_reference")
        if not isinstance(ref, dict) or "file" not in ref:
            continue
        path = ref["file"]
        if not isinstance(path, str) or not path.strip():
            continue  # already reported by validate_entries
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
