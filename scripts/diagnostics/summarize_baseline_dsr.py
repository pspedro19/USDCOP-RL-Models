#!/usr/bin/env python3
"""Compute DSR for baseline return vectors already published by thesis_baselines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from services.common.metrics import dsr_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, default=115)
    parser.add_argument("--periods-per-year", type=int, default=221)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    reports: dict[str, dict] = {}
    for row in payload.get("rows", []):
        if int(row.get("n_traded", 0)) < 20:
            reports[row["baseline"]] = {
                "status": "NOT_COMPUTABLE",
                "reason": "constitution_suppresses_ratios_below_20_trades",
                "n_traded": int(row.get("n_traded", 0)),
            }
            continue
        values = np.asarray(row.get("daily_returns", []), dtype=float)
        if values.size < 2 or not np.isfinite(values).all():
            reports[row["baseline"]] = {
                "status": "NOT_COMPUTABLE",
                "reason": "missing_or_invalid_daily_returns",
            }
            continue
        sd = float(np.std(values, ddof=1))
        sharpe_per_period = float(np.mean(values) / sd) if sd > 0 else 0.0
        reports[row["baseline"]] = dsr_report(
            sharpe_per_period,
            int(values.size),
            args.n_trials,
            periods_per_year=args.periods_per_year,
        )

    result = {
        "contract": "CTR-THESIS-BASELINE-DSR-001",
        "source": str(args.input),
        "n_trials": args.n_trials,
        "periods_per_year": args.periods_per_year,
        "reports": reports,
        "all_pass": all(
            report.get("passes") is True for report in reports.values()
            if report.get("status") != "NOT_COMPUTABLE"
        ) and bool(reports),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "strategies": len(reports), "all_pass": result["all_pass"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
