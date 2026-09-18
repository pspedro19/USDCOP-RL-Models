#!/usr/bin/env python
"""Aggregate one synthetic PPO fixture without charging market trials.

The training runner deliberately executes one seed per process.  This command makes the
five-seed verdict reproducible and refuses partial or mixed-probe evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SEEDS = (42, 123, 456, 789, 1337)


def aggregate(directory: Path, fixture: str, probe: str) -> dict:
    rows: list[dict] = []
    evidence: list[dict] = []
    for seed in SEEDS:
        path = directory / f"{fixture}_{probe}_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing synthetic evidence: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("synthetic_only") is not True or payload.get("market_trials_charged") != 0:
            raise ValueError(f"non-synthetic or charged evidence: {path}")
        if payload.get("fixture") != fixture or payload.get("probe") != probe:
            raise ValueError(f"mixed fixture/probe in {path}")
        items = payload.get("rows", [])
        if len(items) != 1 or items[0].get("seed") != seed:
            raise ValueError(f"unexpected seed row in {path}")
        rows.append(items[0])
        evidence.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    if fixture in {"S1", "S4"}:
        passed_seeds = [r for r in rows if r["mean_abs_exposure"] < 0.1 and r["mean_net"] > -0.005]
    else:
        passed_seeds = [r for r in rows if r["mean_net"] > 0.0]
    return {
        "fixture": fixture,
        "probe": probe,
        "synthetic_only": True,
        "market_trials_charged": 0,
        "seeds": list(SEEDS),
        "rows": rows,
        "passed_seeds": [r["seed"] for r in passed_seeds],
        "passed": len(passed_seeds) >= 4,
        "pass_rule": "4/5 seeds satisfy the pre-registered fixture criterion",
        "input_evidence": evidence,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path("outputs/thesis-repair/sanity"))
    parser.add_argument("--fixture", choices=("S1", "S2", "S3", "S4"), required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.directory, args.fixture, args.probe)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
