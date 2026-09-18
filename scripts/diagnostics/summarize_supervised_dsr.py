#!/usr/bin/env python3
"""Persist DSR for the fixed supervised v2 diagnostic arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from services.common.metrics import dsr_report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--n-trials", type=int, default=115)
    args = p.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    values = np.asarray(source.get("daily_returns", []), dtype=float)
    sd = float(np.std(values, ddof=1))
    sharpe_pp = float(np.mean(values) / sd) if sd > 0 else 0.0
    report = dsr_report(sharpe_pp, len(values), args.n_trials, periods_per_year=221)
    result = {
        "contract": "CTR-THESIS-SUPERVISED-DSR-001",
        "source": str(args.input),
        "evaluation_block": source.get("evaluation_block"),
        "n_trials": args.n_trials,
        "report": report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "headline_dsr": report["headline_dsr"], "passes": report["passes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
