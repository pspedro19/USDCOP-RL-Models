#!/usr/bin/env python3
"""Fail-closed audit of research source identity and frequency lineage.

This is a provenance gate, not a downloader and not a strategy evaluation.  It
deliberately treats Investing/TwelveData as providers that must be identified
and cross-checked, rather than silently promoting a fallback into a canonical
instrument.  A diagnostic report may be produced while a primary source is
pending, but confirmatory jobs must require the gate to pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "config/research/source_lineage.yaml"


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_provenance(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def audit(registry_path: Path = REGISTRY, *, root: Path = ROOT) -> dict[str, Any]:
    """Return a deterministic source-lineage report without network access."""
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise ValueError("source lineage must be a mapping")
    policy = registry.get("policy", {})
    sources = registry.get("sources", {})
    series = registry.get("series", {})
    errors: list[str] = []
    warnings: list[str] = []
    checked: dict[str, Any] = {}

    required_series = {"usdcop_m5", "brent", "dgs2", "ibr", "dxy"}
    missing = sorted(required_series - set(series))
    errors.extend(f"missing_series:{name}" for name in missing)
    for name, spec in series.items():
        if not isinstance(spec, dict):
            errors.append(f"invalid_series:{name}")
            continue
        for field in ("frequency", "instrument", "unit", "primary_source", "availability"):
            if not spec.get(field):
                errors.append(f"{name}:missing_{field}")
        primary = spec.get("primary_source")
        if primary not in sources:
            errors.append(f"{name}:unknown_primary_source:{primary}")
        validators = spec.get("validators", [])
        if not isinstance(validators, list) or not validators:
            errors.append(f"{name}:missing_validators")
        artifact = spec.get("artifact")
        item: dict[str, Any] = {
            "primary_source": primary,
            "frequency": spec.get("frequency"),
            "instrument": spec.get("instrument"),
            "unit": spec.get("unit"),
            "status": spec.get("status"),
            "artifact": artifact,
        }
        if artifact:
            artifact_path = root / artifact
            item["artifact_exists"] = artifact_path.is_file()
            item["artifact_sha256"] = _sha256(artifact_path)
            if not artifact_path.is_file():
                warnings.append(f"{name}:artifact_missing:{artifact}")
            provenance_path = artifact_path.with_suffix(".provenance.json")
            provenance = _load_provenance(provenance_path)
            item["provenance_exists"] = bool(provenance)
            if provenance:
                item["provenance_artifact_sha256_matches"] = (
                    provenance.get("artifact_sha256") == item["artifact_sha256"]
                )
                item["verified_series"] = sorted(
                    provenance.get("series_verified_against_declared_source", [])
                )
            else:
                item["provenance_artifact_sha256_matches"] = False
                item["verified_series"] = []
        crosscheck = spec.get("crosscheck_report")
        if crosscheck:
            crosscheck_path = root / crosscheck
            item["crosscheck_report"] = crosscheck
            item["crosscheck_exists"] = crosscheck_path.is_file()
            if crosscheck_path.is_file():
                try:
                    crosscheck_data = json.loads(crosscheck_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    crosscheck_data = {}
                item["crosscheck_ok"] = crosscheck_data.get("agreement_flag") == "OK"
                item["crosscheck_report_sha256"] = _sha256(crosscheck_path)
                if not item["crosscheck_ok"]:
                    errors.append(f"{name}:crosscheck_not_ok")
            else:
                item["crosscheck_ok"] = False
                warnings.append(f"{name}:crosscheck_missing")
        if "pending" in str(spec.get("status", "")) or str(spec.get("status", "")).startswith("validation_only"):
            warnings.append(f"{name}:primary_pending")
        checked[name] = item

    required_policy = {
        "primary_required_for_confirmatory": True,
        "secondary_is_validation_only": True,
        "silent_fallback": "forbidden",
    }
    for key, expected in required_policy.items():
        if policy.get(key) != expected:
            errors.append(f"policy:{key} must equal {expected!r}")

    # The macro provenance names series by semantic name; compare it with the
    # registry without assuming that a copied/unverified value is acceptable.
    macro_provenance = root / "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.provenance.json"
    prov = _load_provenance(macro_provenance)
    verified_names = set(prov.get("series_verified_against_declared_source", []))
    semantic_to_declared = {"brent": "brent", "dgs2": "dgs2", "ibr": "ibr", "dxy": "dxy"}
    for semantic, declared_name in semantic_to_declared.items():
        if declared_name not in verified_names:
            checked.setdefault(semantic, {})["provenance_verified"] = False
            warnings.append(f"{semantic}:not_verified_in_macro_provenance")
        else:
            checked.setdefault(semantic, {})["provenance_verified"] = True

    confirmatory_ready = not errors and not any(
        ("pending" in str(item.get("status", "")) or str(item.get("status", "")).startswith("validation_only"))
        or ("crosscheck_report" in item and item.get("crosscheck_ok") is not True)
        or item.get("provenance_verified") is False
        for item in checked.values()
    )
    return {
        "contract": "CTR-RESEARCH-SOURCE-LINEAGE-001",
        "registry_sha256": _sha256(registry_path),
        "network_called": False,
        "secrets_read": False,
        "series": checked,
        "errors": sorted(errors),
        "warnings": sorted(set(warnings)),
        "confirmatory_ready": bool(confirmatory_ready),
        "diagnostic_allowed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-diagnostic", action="store_true")
    args = parser.parse_args()
    report = audit(args.registry.resolve(), root=ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"confirmatory_ready": report["confirmatory_ready"],
                      "errors": len(report["errors"]), "warnings": len(report["warnings"])}, ensure_ascii=False))
    if report["errors"] or (not report["confirmatory_ready"] and not args.allow_diagnostic):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
