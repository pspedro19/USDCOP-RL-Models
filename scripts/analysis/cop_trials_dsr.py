"""Honest (trial-aware) Deflated Sharpe for the COP H5 track (smart_simple_v11).

Context (audit 2026-07, G2 of the master plan): the celebrated p=0.0097 / p=0.006 for
smart_simple v1.1/v2.0 came from iterating on the SAME OOS-2025 window (42-cell 2D grid in
FC-H5-SIMPLE-001, "current config ranks #8 of 42"; FC-SIZE-001 picked "best Sharpe-to-MaxDD"
among its cells). That p-value is the winning cell's p-value, not an unbiased estimate.
This script computes the **Deflated Sharpe Ratio** of the published 2025 backtest under a
retroactively reconstructed trial count, using the same Bailey & López de Prado machinery
already used for Gold/BTC (`services/common/metrics.py`).

Because the per-cell Sharpes of the historical grids were not persisted, the trial
dispersion (sigma of per-period Sharpe across trials) cannot be recovered exactly — we
report DSR across a declared, documented range (sensitivity), never picking the friendly
cell. The exact recomputation belongs to the COP-NULL suite (OLA 4), which reruns the
walk-forward.

Outputs a markdown-ready table + JSON to stdout. Read-only; touches nothing.

Usage: python -m scripts.analysis.cop_trials_dsr
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from services.common.metrics import (  # noqa: E402
    deflated_sharpe_ratio,
    trial_aware_moments,
)

TRADES = REPO / "usdcop-trading-dashboard/public/data/production/trades/smart_simple_v11_2025.json"

# ----- Trial count reconstruction (documented sources, conservative lower bounds) -----
# Sources: .claude/experiments/EXPERIMENT_LOG.md
#   FC-H5-SIMPLE-001: v1.0 eval (1) + 2D grid hs/tp "ranks #8 of 42" (42) + v1.1 re-eval (1) = 44
#   FC-SIZE-001 (same-family sizing grid looked at same OOS): baseline + ~5 cells       = 6
#   v1.1 -> v2.0 development (regime gate on/off, effective HS, XGB add, dyn leverage,
#     weekly-vs-monthly retrain fix — each looked at 2025/2026)                          >= 5
#   smart_simple_aggr A/B variant                                                        >= 1
TRIALS_SCENARIOS = {
    "N=44 (solo FC-H5-SIMPLE-001)": 44,
    "N=56 (cota inferior documentada)": 56,
    "N=70 (holgado: versiones no registradas)": 70,
}
# Trial dispersion of PER-PERIOD (weekly) Sharpe across grid cells. Documented anchors:
# v1.0 ann Sharpe 2.87 -> weekly 0.398 ; v1.1 3.52 -> 0.488 ; grid spanned worse cells too.
SIGMA_SCENARIOS = {"σ_trials=0.05/sem (bajo)": 0.05, "σ_trials=0.10/sem (base)": 0.10,
                   "σ_trials=0.15/sem (alto)": 0.15}


def weekly_returns_from_trades(trades: list[dict], year: int = 2025) -> np.ndarray:
    """Weekly portfolio return series INCLUDING flat weeks (the strategy is weekly; a week
    without a trade is a genuine 0% observation, not missing data)."""
    idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="W-MON")
    ret = pd.Series(0.0, index=idx)
    for t in trades:
        ts = pd.Timestamp(t["timestamp"]).tz_localize(None)
        # bucket to the Monday of the trade's week
        monday = ts - pd.Timedelta(days=ts.weekday())
        key = idx[idx.get_indexer([monday], method="nearest")[0]]
        # pnl_pct is on portfolio equity for that week's trade
        ret[key] += float(t.get("pnl_pct", 0.0))
    return (ret / 100.0).values


def main() -> int:
    data = json.loads(TRADES.read_text(encoding="utf-8"))
    trades = data.get("trades", data if isinstance(data, list) else [])
    r = weekly_returns_from_trades(trades)
    m = trial_aware_moments(r)
    ann = m["sharpe_per_period"] * math.sqrt(52)
    print(f"smart_simple_v11 2025 — {len(trades)} trades, {m['n_obs']} semanas")
    print(f"  Sharpe semanal={m['sharpe_per_period']:.4f} (anualizado≈{ann:.2f}), "
          f"skew={m['skew']}, kurt={m['kurtosis']}, PSR(1 trial)={m['psr']}")
    print()
    print("| Escenario de trials | σ_trials (sem) | SR0 (sem) | DSR | ¿>0.95? |")
    print("|---|---|---|---|---|")
    results = {}
    for tn, n in TRIALS_SCENARIOS.items():
        for sn, sig in SIGMA_SCENARIOS.items():
            d = deflated_sharpe_ratio(m["sharpe_per_period"], m["n_obs"], n, sig,
                                      m["skew"], m["kurtosis"])
            print(f"| {tn} | {sig} | {d['sr0']} | {d['dsr']} | "
                  f"{'SI' if d['significant'] else 'NO'} |")
            results[f"{tn}|{sn}"] = d
    print()
    print(json.dumps({"moments": m, "sharpe_ann": round(ann, 3), "scenarios": results},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
