"""Revalidate an ALFRED capture from bytes for the retrospective E2E auditor.

This gate does not certify market-time availability, fill provenance or an edge.
It never calls a provider and never changes the frozen capture/parser/training data.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from scripts.diagnostics import audit_research_macro_vintages as source
from src.research import alfred_vintages as vintage


def _safe(path: Path) -> Path:
    path = Path(path)
    for candidate in (path, path.resolve()):
        if any(
            part.lower().startswith((".env", "credentials", "service-account"))
            or part.lower() == "secrets"
            or part.lower().endswith((".pem", ".key"))
            for part in candidate.parts
        ):
            raise ValueError("secret evidence path refused before reading")
    return source.allowed(path)


def _json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON evidence key")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError("nonfinite JSON evidence value")

    result = json.loads(
        _safe(path).read_bytes(), object_pairs_hook=pairs, parse_constant=invalid_constant
    )
    if not isinstance(result, dict):
        raise ValueError("evidence document must be an object")
    return result


def _hash(path: Path, expected: str) -> None:
    if not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
        raise ValueError("complete externally declared SHA256 digest required")
    if vintage.digest(_safe(path)) != expected:
        raise ValueError("evidence identity/hash differs from declared digest")


def _inputs(bundle: Path, expected_sha: str):
    """Reject ambiguous JSON before using the frozen historical input loader."""
    _hash(bundle / "manifest.json", expected_sha)
    manifest = _json(bundle / "manifest.json")
    _hash(bundle / "daily_series.json", manifest["artifacts_sha256"]["daily_series.json"])
    _json(bundle / "daily_series.json")
    snapshot = _safe(Path(manifest["snapshot_manifest"]))
    _hash(snapshot, manifest["snapshot_sha256"])
    _json(snapshot)
    return source.load_inputs(bundle, expected_sha)


def verify_vintage_capture(
    *, capture: Path, expected_capture_sha: str, bundle: Path, expected_bundle_sha: str
) -> dict:
    """Read and recompute exact-cohort evidence; a rehashed fake PASS still fails.

    A newer figure-only bundle may refer to the capture if its cohort, macro,
    availability contract and archival source identity are exactly unchanged.
    The original capture bundle is verified too; no most-recent-file discovery.
    """
    capture, bundle = _safe(capture), _safe(bundle)
    manifest_path = capture / "manifest.json"
    _hash(manifest_path, expected_capture_sha)
    manifest = _json(manifest_path)
    if _safe(Path(manifest["raw_evidence_root"])) != capture:
        raise ValueError("expected original capture directory, not a report-only replay")
    _hash(capture / "audit_freeze.json", manifest["freeze_sha256"])
    _hash(capture / "report.json", manifest["report_sha256"])
    freeze, reported = _json(capture / "audit_freeze.json"), _json(capture / "report.json")
    if (
        freeze["version"] != vintage.VERSION
        or freeze["series"] != sorted(vintage.SERIES_COLUMNS)
        or type(freeze["window_calendar_days"]) is not int
        or freeze["window_calendar_days"] != 60
        or freeze["coverage_is_entire_selection"] is not True
    ):
        raise ValueError("capture protocol or complete-cohort declaration mismatch")
    for filename, expected, current in (
        ("alfred_vintages.py", freeze["source_sha256"], Path(vintage.__file__)),
        ("audit_research_macro_vintages.py", freeze["runner_sha256"], Path(source.__file__)),
    ):
        _hash(capture / "source" / filename, expected)
        if vintage.digest(current) != expected:
            raise ValueError("current parser/runner differs from frozen capture source")

    original_manifest = _safe(Path(freeze["inputs"]["bundle_manifest"]))
    if original_manifest.name != "manifest.json":
        raise ValueError("original bundle manifest filename mismatch")
    _, original_dates, original_inputs = _inputs(
        original_manifest.parent, freeze["inputs"]["bundle_sha256"]
    )
    frame, dates, inputs = _inputs(bundle, expected_bundle_sha)
    if original_inputs != freeze["inputs"]:
        raise ValueError("capture input identities differ from original archived bundle")
    location_fields = {"bundle_manifest", "bundle_sha256"}
    if (
        {k: v for k, v in inputs.items() if k not in location_fields}
        != {k: v for k, v in original_inputs.items() if k not in location_fields}
        or dates != original_dates
        or dates != freeze["dates"]
    ):
        raise ValueError("capture and current bundle do not identify the same full cohort/inputs")

    expected_records = {
        f"captures/{series}_{(date.fromisoformat(day) - timedelta(days=1)).isoformat()}.json"
        for day in dates
        for series in vintage.SERIES_COLUMNS
    }
    records_by_path = {}
    for name, digest in manifest["capture_records_sha256"].items():
        canonical = name.replace("\\", "/")
        if canonical in records_by_path:
            raise ValueError("duplicate normalized capture record path")
        records_by_path[canonical] = digest
    if set(records_by_path) != expected_records:
        raise ValueError("manifest capture records do not cover exact cohort")
    records = []
    for name in sorted(expected_records):
        record_path = _safe(capture / name)
        if not record_path.is_relative_to(capture):
            raise ValueError("capture record path escape")
        _hash(record_path, records_by_path[name])
        record = _json(record_path)
        expected_start = (
            date.fromisoformat(record["vintage_date"])
            - timedelta(days=freeze["window_calendar_days"])
        ).isoformat()
        if record["start_date"] != expected_start:
            raise ValueError("capture record window differs from frozen window")
        if record.get("raw_path") is not None:
            _safe(capture / record["raw_path"])
        records.append(record)
    recomputed = source.summarize(
        frame, dates, records, capture, inputs["max_staleness_business_days"]
    )
    expected_report = {
        **recomputed,
        "inputs": original_inputs,
        "coverage_is_entire_selection": True,
        "network_called": True,
        "raw_evidence_root": str(capture),
        "freeze_sha256": manifest["freeze_sha256"],
    }
    reported_at = datetime.fromisoformat(reported["generated_at_utc"])
    if reported_at.tzinfo is None or reported_at > datetime.now(UTC):
        raise ValueError("report creation timestamp is not a real past instant")

    def without_clock(value):
        return {k: v for k, v in value.items() if k != "generated_at_utc"}

    if without_clock(reported) != without_clock(expected_report):
        raise ValueError("report claims differ from recomputed raw evidence")
    mismatches = sum(
        row["status"] in {"NOT_IN_PRIOR_DATE_VINTAGE", "VALUE_DIFFERS_FROM_PRIOR_DATE_VINTAGE"}
        for row in recomputed["rows"]
    )
    status = "DIAGNOSTIC_REPRODUCED_WITH_MISMATCHES" if mismatches else "DIAGNOSTIC_REPRODUCED"
    if recomputed["fetch_or_parse_errors"]:
        status = "DIAGNOSTIC_PARTIAL"
    return {
        "status": status,
        "capture_manifest_sha256": expected_capture_sha,
        "current_bundle_manifest_sha256": expected_bundle_sha,
        "original_bundle_manifest_sha256": original_inputs["bundle_sha256"],
        "snapshot_sha256": inputs["snapshot_sha256"],
        "macro_sha256": inputs["macro_sha256"],
        "source_sha256": freeze["source_sha256"],
        "runner_sha256": freeze["runner_sha256"],
        "verifier_sha256": vintage.digest(Path(__file__)),
        "scope": recomputed["scope"],
        "series": recomputed["series"],
        "requested_snapshots": recomputed["requested_snapshots"],
        "capture_coverage": recomputed["capture_coverage"],
        "fetch_or_parse_errors": recomputed["fetch_or_parse_errors"],
        "n_sessions": len(dates),
        "mismatch_rows": mismatches,
        "network_called": False,
        "opening_availability_verified": False,
        "historical_collector_receipt_verified": False,
        "dxy_ibr_vintages_verified": False,
        "scientific_closure_ready": False,
        "hmm_transformed_input_path_audited": False,
        "archived_macro_to_historical_checkpoints_lineage_verified": False,
        "proves": [
            "archived_bytes_match_externally_supplied_manifest_hashes",
            "cohort_and_macro_inputs_match_the_requested_bundle",
            "saved_report_matches_current_offline_recomputation_from_raw_csv",
        ],
        "does_not_prove": [
            "hora_08",
            "recibo_historico",
            "DXY",
            "IBR",
            "linaje_de_entrenamiento",
            "HMM_input_transformation_path",
            "independent_source_authentication",
            "executable_prices_or_costs",
            "strategy_edge",
            "confirmatory_replication",
        ],
    }
