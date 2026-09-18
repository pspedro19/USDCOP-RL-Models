#!/usr/bin/env python
"""Reconcile macro identity numerically from archived original source responses.

Default execution fetches and archives references, but never repairs the parquet.
--reference-manifest replays a builder/verifier manifest without network access.
This verifies reproducibility, not source independence or historical availability.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.build_research_macro import _banrep_ibr, _fred, _investing_dxy
from src.research.macro_evidence import (
    SCHEMA_VERSION, canonical_json, compare_series, immutable_write,
    replay_payloads, require_macro_evidence, sha256_file,
)

AVAILABILITY = ROOT / "config/research/macro_availability.yaml"
CLEAN = ROOT / "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet"
PROVENANCE = CLEAN.with_suffix(".provenance.json")
FRED_SOURCES = {"FRED_DCOILBRENTEU": "DCOILBRENTEU", "FRED_DGS2": "DGS2"}
TOLERANCE = 0.01
_digest = sha256_file


def _load_local_provenance(clean_path: Path) -> dict:
    """Provenance alone cannot satisfy the numerical verification gate."""
    path = clean_path.with_suffix(".provenance.json")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) and value.get("artifact_sha256") == _digest(clean_path) else {}


def _local_reference_with_column(path, value_column):
    """Legacy explicit CSV diagnostic; not an automatic identity certificate."""
    import pandas as pd

    raw = Path(path).read_bytes()
    frame = pd.read_csv(io.BytesIO(raw))
    date_col = next((c for c in frame if str(c).lower() in
                     {"date", "datetime", "time", "fecha"}), None)
    if date_col is None or value_column not in frame or date_col == value_column:
        raise ValueError("referencia local DXY: fecha/columna numérica inexistente")
    dates = pd.to_datetime(frame[date_col], errors="raise")
    values = pd.to_numeric(frame[value_column], errors="raise")
    result = pd.Series(values.to_numpy(dtype=float), index=dates).dropna()
    if result.empty or result.index.has_duplicates:
        raise ValueError("referencia local DXY: vacía o fechas duplicadas")
    return result.sort_index(), hashlib.sha256(raw).hexdigest()


def _local_reference(path):
    import pandas as pd

    frame = pd.read_csv(path)
    date_col = next((c for c in frame if str(c).lower() in
                     {"date", "datetime", "time", "fecha"}), None)
    numeric = [c for c in frame if c != date_col and
               pd.to_numeric(frame[c], errors="coerce").notna().any()]
    if len(numeric) != 1:
        raise ValueError("referencia local DXY: se esperaba exactamente una columna numérica")
    return _local_reference_with_column(path, numeric[0])


def build_report(*, clean_path: Path, availability: Path, evidence_dir: Path,
                 reference_manifest: Path | None = None) -> dict:
    """Produce measured statistics and preserve failed sources explicitly."""
    import pandas as pd
    import yaml

    clean_raw, availability_raw = clean_path.read_bytes(), availability.read_bytes()
    clean = pd.read_parquet(io.BytesIO(clean_raw))
    declared = yaml.safe_load(availability_raw)["series"]
    reference_meta = None
    if reference_manifest is not None:
        reference_meta = json.loads(reference_manifest.read_text(encoding="utf-8"))
        evidence_dir = Path(reference_meta["evidence_root"]).resolve()
    report = {}
    for name, spec in declared.items():
        source, column = spec["source"], spec["column"]
        entry = {"column": column, "declared_source": source, "unit": spec["unit"],
                 "fallback": spec.get("fallback"), "honoured": False,
                 "reference_relationship": "declared_source_reproduction_not_independent"}
        try:
            ours = clean[column]
            nonnull = ours.dropna()
            if nonnull.empty:
                raise ValueError("empty local source column")
            records = []
            if reference_meta is not None:
                original = reference_meta["series"][name]
                if original.get("declared_source") != source:
                    raise ValueError("offline source declaration mismatch")
                records = original["reference_payloads"]
                reference = replay_payloads(records, source=source, evidence_root=evidence_dir)
            else:
                kwargs = {"evidence_dir": evidence_dir, "records": records}
                if source in FRED_SOURCES:
                    reference, _ = _fred(FRED_SOURCES[source], **kwargs)
                elif source == "BANREP_IBR":
                    reference, _ = _banrep_ibr(**kwargs)
                elif source == "INVESTING_DXY":
                    reference, _ = _investing_dxy(nonnull.index.min().date(),
                                                nonnull.index.max().date(), **kwargs)
                else:
                    raise ValueError("source has no archived parser; not certifiable here")
            entry["reference_payloads"] = records
            entry.update(compare_series(ours, reference, tolerance=TOLERANCE))
            entry["honoured"] = (entry["n_missing_reference"] == 0 and entry["n_missing_local"] == 0
                                  and entry["n_discrepancies"] == 0)
            entry["status"] = "COINCIDE" if entry["honoured"] else "NO COINCIDE"
        except Exception as exc:
            # No exception text: provider URLs can contain credentials.
            entry.update(status=f"NO VERIFICADO ({type(exc).__name__})", honoured=False)
        report[name] = entry
    output = {
        "schema_version": SCHEMA_VERSION,
        "contract": "CTR-RESEARCH-MACRO-AVAILABILITY-001",
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {"availability_sha256": hashlib.sha256(availability_raw).hexdigest(),
                   "clean_sha256": hashlib.sha256(clean_raw).hexdigest(),
                   "availability_path": str(availability.resolve()),
                   "clean_path": str(clean_path.resolve())},
        "verifier_sha256": _digest(Path(__file__)),
        "evidence_root": str(evidence_dir.resolve()), "tolerance": TOLERANCE,
        "tolerance_note": "absolute source units; all common observations must match; no >99% shortcut",
        "series": report,
        "all_declared_identities_honoured": bool(report) and all(e["honoured"] for e in report.values()),
        "source_independence_verified": False, "historical_availability_verified": False,
        "mode": "offline_raw_replay" if reference_manifest else "fresh_capture_and_replay",
    }
    try:
        output["strict_verification"] = require_macro_evidence(
            output, clean=clean_path, availability=availability)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        output["all_declared_identities_honoured"] = False
        output["strict_verification"] = {"numerical_identity_verified": False,
                                         "failure_type": type(exc).__name__}
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True,
                        help="new report path; existing output is rejected")
    parser.add_argument("--clean-path", type=Path, default=CLEAN)
    parser.add_argument("--availability", type=Path, default=AVAILABILITY)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--reference-manifest", type=Path,
                        help="offline builder provenance or report with archived reference payloads")
    args = parser.parse_args()
    if args.output.exists():
        print("ABORTA: report output already exists; choose a new path", file=sys.stderr)
        return 2
    evidence_dir = args.evidence_dir or args.output.parent / "macro_evidence"
    report = build_report(clean_path=args.clean_path.resolve(),
                          availability=args.availability.resolve(),
                          evidence_dir=evidence_dir.resolve(),
                          reference_manifest=args.reference_manifest)
    raw = canonical_json(report)
    immutable_write(args.output, raw)
    immutable_write(evidence_dir / "reports" / f"{hashlib.sha256(raw).hexdigest()}.json", raw)
    for name, entry in report["series"].items():
        print(f"{name}: {entry['status']} n_common={entry.get('n_common', 0)}")
    print(f"numerical_identity_verified={report['all_declared_identities_honoured']}")
    return 0 if report["all_declared_identities_honoured"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
