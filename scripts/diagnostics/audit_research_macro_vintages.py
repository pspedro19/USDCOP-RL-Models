"""Capture/replay public ALFRED snapshots for the frozen selection cohort.

No .env, API keys, strategy replay, training or mutation of market data. The
output is date-resolution data evidence, not a pre-open availability certificate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.research.alfred_vintages import (  # noqa: E402
    SERIES_COLUMNS,
    VERSION,
    capture_snapshot,
    compare_snapshot,
    digest,
    json_bytes,
    replay_snapshot,
    validate_daily_labels,
    write_new,
)


def allowed(path: Path) -> Path:
    target = path.resolve()
    if not target.is_relative_to((ROOT / "outputs/thesis-repair").resolve()):
        raise ValueError("diagnostic paths must stay within outputs/thesis-repair")
    if any(p.lower().startswith(".env") or p.lower() == "secrets"
           or p.lower().endswith((".pem", ".key"))
           or (p.lower().startswith(("credentials", "service-account")) and p.lower().endswith(".json"))
           for p in target.parts):
        raise ValueError("secret path forbidden")
    return target


def load_inputs(bundle: Path, expected_sha: str) -> tuple[pd.DataFrame, list[str], dict]:
    bundle = allowed(bundle)
    manifest_path = bundle / "manifest.json"
    if digest(manifest_path) != expected_sha:
        raise ValueError("bundle manifest identity mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    series_path = bundle / "daily_series.json"
    if digest(series_path) != manifest["artifacts_sha256"]["daily_series.json"]:
        raise ValueError("cohort series hash mismatch")
    daily = json.loads(series_path.read_text(encoding="utf-8"))
    dates = validate_daily_labels([row["date"] for row in daily["always_flat"]])
    if not dates or dates != sorted(dates):
        raise ValueError("nonempty ordered cohort required")
    if any([row["date"] for row in rows] != dates for rows in daily.values()):
        raise ValueError("retrospective arms do not share the same cohort")
    snapshot_path = allowed(Path(manifest["snapshot_manifest"]))
    if digest(snapshot_path) != manifest["snapshot_sha256"]:
        raise ValueError("snapshot manifest hash mismatch")
    archive = json.loads(snapshot_path.read_text(encoding="utf-8"))
    candidates = [row for row in archive["files"] if row["path"] ==
                  "data/pipeline/04_cleaning/output/MACRO_RESEARCH_v2.parquet"]
    if len(candidates) != 1:
        raise ValueError("one exact archived macro input required")
    record = candidates[0]
    expected_object = "objects/" + record["sha256"]
    if record["object"] != expected_object:
        raise ValueError("invalid archived object name")
    macro_path = allowed(snapshot_path.parent / expected_object)
    if (not macro_path.is_relative_to(snapshot_path.parent)
            or digest(macro_path) != record["sha256"]):
        raise ValueError("macro object hash mismatch or path escape")
    frame = pd.read_parquet(macro_path)
    if not set(SERIES_COLUMNS.values()) <= set(frame):
        raise ValueError("missing declared macro columns")
    availability = [r for r in archive["files"] if r["path"] == "config/research/macro_availability.yaml"]
    if len(availability) != 1 or availability[0]["object"] != "objects/" + availability[0]["sha256"]:
        raise ValueError("archived availability contract required")
    availability_path = allowed(snapshot_path.parent / availability[0]["object"])
    if digest(availability_path) != availability[0]["sha256"]:
        raise ValueError("archived availability hash mismatch")
    stale_days = yaml.safe_load(availability_path.read_text(encoding="utf-8"))["max_staleness_business_days"]
    if type(stale_days) is not int or stale_days < 0:
        raise ValueError("invalid archived freshness limit")
    return frame, dates, {
        "bundle_manifest": str(manifest_path), "bundle_sha256": expected_sha,
        "daily_series_sha256": digest(series_path),
        "snapshot_manifest": str(snapshot_path), "snapshot_sha256": digest(snapshot_path),
        "macro_sha256": record["sha256"], "cohort_size": len(dates),
        "availability_sha256": availability[0]["sha256"],
        "max_staleness_business_days": stale_days,
    }


def summarize(frame: pd.DataFrame, dates: list[str], records: list[dict], root: Path,
              max_staleness_business_days: int) -> dict:
    expected = {(series, day) for series in SERIES_COLUMNS for day in dates}
    keys = [(r["series"], r["session_date"]) for r in records]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("capture records must cover the exact declared cohort, without duplicates")
    rows = []
    for record in sorted(records, key=lambda r: (r["series"], r["session_date"])):
        if record["network_status"] == "FETCH_OR_PARSE_ERROR":
            rows.append({"series": record["series"], "session_date": record["session_date"],
                         "status": "FETCH_OR_PARSE_ERROR", "error_type": record["error_type"]})
            continue
        vintage = replay_snapshot(root, record)
        row = compare_snapshot(frame[SERIES_COLUMNS[record["series"]]], vintage,
                               series=record["series"], session_date=record["session_date"],
                               vintage_date=record["vintage_date"], capture_start_date=record["start_date"],
                               max_staleness_business_days=max_staleness_business_days)
        row.update(raw_sha256=record["raw_sha256"], retrieved_at_utc=record["retrieved_at_utc"])
        rows.append(row)
    groups = {}
    for series in SERIES_COLUMNS:
        selected = [r for r in rows if r["series"] == series]
        groups[series] = {
            "counts": dict(Counter(r["status"] for r in selected)),
            "sessions": len(selected),
        }
    errors = sum(row["status"] == "FETCH_OR_PARSE_ERROR" for row in rows)
    return {
        "version": VERSION, "scope": "DIRECT_MACRO_FEATURE_INPUT_DIAGNOSTIC_NO_STRATEGY_EVALUATION",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "capture_coverage": "COMPLETE" if errors == 0 else "PARTIAL",
        "requested_snapshots": len(expected), "fetch_or_parse_errors": errors,
        "opening_availability_verified": False, "historical_collector_receipt_verified": False,
        "dxy_ibr_vintages_verified": False, "scientific_closure_ready": False,
        "hmm_transformed_input_path_audited": False,
        "archived_macro_to_historical_checkpoints_lineage_verified": False,
        "conclusions_allowed": [
            "Only value/period membership in the requested prior-calendar-date ALFRED vintage.",
            "Absence does not prove unavailability at 08:00 COT through another source or same-day release.",
            "Vintage date is neither an exact publication timestamp nor a historical collector receipt.",
            "This audit does not replace macro data, select lags, recompute PnL or authorize training.",
            "The macro was preserved with the experiment; missing historical checkpoint lineage is not restored by this audit.",
        ],
        "sources": ["https://alfred.stlouisfed.org/help",
                    "https://fred.stlouisfed.org/docs/api/fred/series_observations.html",
                    "https://www.federalreserve.gov/feeds/h15.html"],
        "series": groups, "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--expected-bundle-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--capture", action="store_true")
    mode.add_argument("--replay-run", type=Path)
    parser.add_argument("--expected-capture-sha")
    parser.add_argument("--max-sessions", type=int)
    args = parser.parse_args()
    if bool(args.replay_run) != bool(args.expected_capture_sha):
        raise ValueError("offline replay requires expected capture manifest SHA; capture does not")
    output = allowed(args.output)
    if output.exists():
        raise ValueError("output directory already exists; no evidence overwrite")
    frame, full_dates, identities = load_inputs(args.bundle, args.expected_bundle_sha)
    dates = full_dates
    if args.max_sessions is not None:
        if not 1 <= args.max_sessions <= len(full_dates):
            raise ValueError("invalid requested prefix size")
        dates = full_dates[:args.max_sessions]
    freeze = {"version": VERSION, "inputs": identities, "dates": dates,
              "series": sorted(SERIES_COLUMNS), "window_calendar_days": 60,
              "source_sha256": digest(ROOT / "src/research/alfred_vintages.py"),
              "runner_sha256": digest(Path(__file__)),
              "coverage_is_entire_selection": dates == full_dates}
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "audit_freeze.json", json_bytes(freeze))
    write_new(output / "source/alfred_vintages.py", (ROOT / "src/research/alfred_vintages.py").read_bytes())
    write_new(output / "source/audit_research_macro_vintages.py", Path(__file__).read_bytes())
    if args.capture:
        records = []
        # Bounded serial public requests: no parallel source load, no API keys,
        # no automatic retry that could silently replace failed evidence.
        for i, day in enumerate(dates):
            for series in SERIES_COLUMNS:
                records.append(capture_snapshot(output, series, day))
                print(json.dumps({"session": i + 1, "of": len(dates), "series": series,
                                  "status": records[-1]["network_status"]}), flush=True)
                time.sleep(0.5)
        raw_root = output
    else:
        raw_root = allowed(args.replay_run)
        capture_manifest_path = raw_root / "manifest.json"
        if digest(capture_manifest_path) != args.expected_capture_sha:
            raise ValueError("capture manifest differs from externally expected digest")
        capture_manifest = json.loads(capture_manifest_path.read_text(encoding="utf-8"))
        if digest(raw_root / "audit_freeze.json") != capture_manifest["freeze_sha256"]:
            raise ValueError("capture freeze hash mismatch")
        original = json.loads((raw_root / "audit_freeze.json").read_text(encoding="utf-8"))
        if original != freeze:
            raise ValueError("offline replay requires identical inputs, cohort and source code")
        records = []
        for day in dates:
            vintage = (pd.Timestamp(day) - pd.Timedelta(days=1)).date().isoformat()
            for series in SERIES_COLUMNS:
                record_path = raw_root / "captures" / f"{series}_{vintage}.json"
                if digest(record_path) != capture_manifest["capture_records_sha256"][str(record_path.relative_to(raw_root))]:
                    raise ValueError("capture record hash mismatch")
                records.append(json.loads(record_path.read_text(encoding="utf-8")))
    report = summarize(frame, dates, records, raw_root, identities["max_staleness_business_days"])
    report.update(inputs=identities, coverage_is_entire_selection=dates == full_dates,
                  network_called=bool(args.capture), raw_evidence_root=str(raw_root),
                  freeze_sha256=digest(output / "audit_freeze.json"))
    write_new(output / "report.json", json_bytes(report))
    capture_hashes = {str(path.relative_to(raw_root)): digest(path)
                      for path in sorted((raw_root / "captures").glob("*.json"))}
    write_new(output / "manifest.json", json_bytes({
        "report_sha256": digest(output / "report.json"),
        "freeze_sha256": digest(output / "audit_freeze.json"),
        "raw_evidence_root": str(raw_root), "capture_records_sha256": capture_hashes,
    }))
    print(json.dumps({"report": str(output / "report.json"), "series": report["series"],
                      "capture_coverage": report["capture_coverage"]}), flush=True)
    return 0 if report["capture_coverage"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
