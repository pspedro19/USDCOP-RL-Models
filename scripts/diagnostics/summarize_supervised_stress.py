#!/usr/bin/env python3
"""Cost stress for the fixed supervised diagnostic arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    gross = np.asarray(source["daily_gross_returns"], dtype=float)
    costs = np.asarray(source["daily_costs"], dtype=float)
    results = {}
    for factor in (1, 2, 3):
        net = gross - factor * costs
        equity = np.cumprod(1.0 + net)
        results[f"x{factor}"] = {
            "compound_return_pct": float((equity[-1] - 1.0) * 100.0),
            "mean_daily_net_pct": float(np.mean(net) * 100.0),
            "survives": bool(equity[-1] > 1.0),
        }
    payload = {
        "contract": "CTR-THESIS-SUPERVISED-STRESS-001",
        "source": str(args.input),
        "evaluation_block": source.get("evaluation_block"),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
