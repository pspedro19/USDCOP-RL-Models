#!/usr/bin/env python
"""Build the immutable five-seed S1--S4 synthetic protocol verdict.

This report is an optimizer/environment gate only.  It deliberately refuses mixed probes,
partial seed sets, market data, or charged trials, so a passing synthetic control cannot be
mistaken for evidence of a tradable edge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1337)
FIXTURES = ("S1", "S2", "S3", "S4")


def aggregate_protocol(directory: Path, probe: str) -> dict:
    fixture_reports: dict[str, dict] = {}
    evidence: list[dict] = []
    for fixture in FIXTURES:
        path = directory.parent / f"sanity_{fixture}_{probe}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing aggregate: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fixture") != fixture or payload.get("probe") != probe:
            raise ValueError(f"mixed fixture/probe in {path}")
        if payload.get("synthetic_only") is not True or payload.get("market_trials_charged") != 0:
            raise ValueError(f"non-synthetic or charged aggregate: {path}")
        if tuple(payload.get("seeds", ())) != SEEDS or len(payload.get("rows", ())) != 5:
            raise ValueError(f"incomplete five-seed aggregate: {path}")
        if payload.get("passed") is not True:
            raise ValueError(f"fixture failed its pre-registered criterion: {path}")
        fixture_reports[fixture] = {
            "passed": True,
            "passed_seeds": payload.get("passed_seeds", []),
            "rows": payload["rows"],
        }
        evidence.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return {
        "protocol": "S1-S4",
        "probe": probe,
        "seeds": list(SEEDS),
        "fixtures": fixture_reports,
        "synthetic_only": True,
        "market_trials_charged": 0,
        "market_evidence": False,
        "passed": True,
        "interpretation": (
            "Optimizer/environment controls pass with one frozen probe; "
            "this is not evidence of market profitability."
        ),
        "input_evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("outputs/thesis-repair/sanity"))
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate_protocol(args.directory, args.probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
