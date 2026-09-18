#!/usr/bin/env python
"""Summarize immutable PPO run JSONs without selecting a winner.

The report is deliberately descriptive: it preserves every seed/configuration and
marks the experiment incomplete until the preregistered 5x2 matrix is present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_CONFIGS = ("ppo_regime", "ppo_backbone")
EXPECTED_SEEDS = (42, 123, 456, 789, 1337)


def _metric(block: dict, key: str) -> float | int | None:
    value = block.get(key)
    return value if isinstance(value, int | float) else None


def summarize(directory: Path) -> dict:
    rows: list[dict] = []
    invalid: list[str] = []
    for path in sorted(directory.glob("ppo_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            invalid.append(str(path))
            continue
        if not isinstance(payload, dict):
            invalid.append(str(path))
            continue
        if not {"config", "seed", "timesteps", "development", "selection"} <= payload.keys():
            invalid.append(str(path))
            continue
        rows.append({
            "artifact": str(path),
            "config": payload["config"],
            "seed": payload["seed"],
            "timesteps": payload["timesteps"],
            "refit": bool(payload.get("refit", False)),
            "development": {
                "total_return": _metric(payload["development"], "total_return"),
                "sharpe": _metric(payload["development"], "sharpe"),
                "n_ops": _metric(payload["development"], "n_ops"),
            },
            "selection": {
                "total_return": _metric(payload["selection"], "total_return"),
                "sharpe": _metric(payload["selection"], "sharpe"),
                "n_ops": _metric(payload["selection"], "n_ops"),
            },
        })
    keys = {(row["config"], row["seed"]) for row in rows}
    expected = {(config, seed) for config in EXPECTED_CONFIGS for seed in EXPECTED_SEEDS}
    return {
        "contract": "CTR-THESIS-PPO-DIAGNOSTIC-SUMMARY-001",
        "directory": str(directory),
        "classification": "diagnostic_retrospective",
        "confirmatory": False,
        "expected_runs": len(expected),
        "valid_runs": len(rows),
        "missing_runs": [
            {"config": config, "seed": seed}
            for config, seed in sorted(expected - keys)
        ],
        "invalid_artifacts": invalid,
        "complete_matrix": keys == expected and not invalid,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.directory)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "valid_runs": report["valid_runs"],
        "complete_matrix": report["complete_matrix"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
