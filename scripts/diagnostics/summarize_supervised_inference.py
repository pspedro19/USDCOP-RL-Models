#!/usr/bin/env python3
"""Stationary paired bootstrap for the fixed supervised diagnostic arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.research.inference import paired_sharpe_test


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    values = np.asarray(source["daily_returns"], dtype=float)
    result = paired_sharpe_test(values, np.zeros_like(values), "supervised_v2", "always_flat")
    result_dict = result.to_dict()
    if not np.isfinite(result_dict.get("correlation", 0.0)):
        result_dict["correlation"] = None
    payload = {
        "contract": "CTR-THESIS-SUPERVISED-INFERENCE-001",
        "source": str(args.input),
        "evaluation_block": source.get("evaluation_block"),
        "result": result_dict,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decisive": result.decisive, "diff": result.diff, "p_value": result.p_value}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
