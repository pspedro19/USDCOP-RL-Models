"""Forward-vs-backtest tracker — the organ that decides whether any of this works.

Contract: CTR-FORWARD-TRACK-001

Nothing in this repo accumulated forward performance. `run_btc_pipeline.py --phase production`
writes a snapshot and OVERWRITES it on the next run, so there was no time series at all. That
makes graduation criterion #4 of `WITHDRAWAL-PROTOCOL-BTC.md:48` -- *"divergencia
|paper − backtest-replay| de retorno semanal < 2pp promedio"* -- literally unevaluable, and the
26-week window pointless: you would arrive at the end with nothing to judge.

The existing `forecast_h5_paper_trading` table cannot serve: it is a singleton keyed on
`signal_date` with no `strategy_id`, and it is shaped for weekly DIRECTIONAL forecasting
(`running_da_pct`) rather than continuous exposure.

## What this does

Appends one immutable row per (strategy, week). Never rewrites history: a corrected week is a
new row with a later `recorded_at`, so a revision can be seen rather than silently applied.
That matters because the whole point is to compare what we PREDICTED against what HAPPENED, and
a store that can be edited retroactively cannot support that comparison.

For each week it records the paper return, the backtest-replay return for the same week, and
their divergence. Divergence is the honest early-warning signal: a strategy can look fine on
absolute return while quietly diverging from its own model, which means the model no longer
describes it -- and that shows up before the PnL does.

## What it deliberately does NOT do

- It does not decide. Thresholds live in the withdrawal protocols, signed ex-ante.
- It does not fill gaps. A missing week stays missing; interpolating forward evidence would be
  fabricating the only clean data this system has.

Run: python -m scripts.analysis.forward_tracker --record   (weekly, after signals)
     python -m scripts.analysis.forward_tracker --report
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from services.common.metrics import _ann_return_dd_calmar  # noqa: E402

STORE = REPO / ".claude" / "evidence" / "forward" / "forward_ledger.jsonl"
REGISTRY = REPO / "usdcop-trading-dashboard" / "public" / "data" / "registry.json"

# From WITHDRAWAL-PROTOCOL-BTC.md:48. Declared here as a REPORTING threshold only -- this
# module surfaces breaches, it never acts on them.
MAX_WEEKLY_DIVERGENCE_PP = 2.0


def _champions() -> list[dict]:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    return [s for s in reg["strategies"] if s.get("status") != "archived"]


def _iso_week(ts: pd.Timestamp) -> str:
    y, w, _ = ts.isocalendar()
    return f"{y}-W{int(w):02d}"


def record(week: str | None = None) -> int:
    """Append this week's row per live strategy. Idempotent per (strategy, week, source)."""
    STORE.parent.mkdir(parents=True, exist_ok=True)
    now = pd.Timestamp.now(tz=timezone.utc)
    wk = week or _iso_week(now)

    existing = set()
    if STORE.exists():
        for line in STORE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            existing.add((r["strategy_id"], r["week"]))

    from scripts.analysis.profitability_adapters import ADAPTERS

    written = 0
    for champ in _champions():
        sid, asset = champ["strategy_id"], champ.get("asset_id")
        if (sid, wk) in existing:
            print(f"  {sid:26} {wk}  ya registrado (no se reescribe)")
            continue

        # Backtest-replay return for this same week, from the adapter that produced the bundle.
        replay_ret = None
        try:
            if asset in ADAPTERS:
                s = ADAPTERS[asset]()
                idx = pd.to_datetime(pd.Index(s.index))
                mask = np.asarray([_iso_week(t) == wk for t in idx], dtype=bool)
                if mask.any():
                    replay_ret = float((np.prod(1 + s.strat_ret[mask]) - 1) * 100)
        except Exception as e:  # noqa: BLE001
            print(f"  {sid:26} replay no disponible: {e}")

        # Paper return comes from the live production bundle. Absent = absent, not zero: a week
        # with no signal is not a flat week, and recording 0.0 would quietly invent evidence.
        paper_ret = None
        prod = REPO / "usdcop-trading-dashboard/public/data/production" / f"summary_{sid}.json"
        if prod.is_file():
            try:
                d = json.loads(prod.read_text(encoding="utf-8"))
                paper_ret = (d.get("strategies", {}).get(sid, {})
                             .get("total_return_pct"))
            except Exception as e:  # noqa: BLE001
                print(f"  {sid:26} paper no legible: {e}")

        div = (abs(paper_ret - replay_ret)
               if paper_ret is not None and replay_ret is not None else None)
        row = {
            "recorded_at": now.isoformat(), "week": wk,
            "strategy_id": sid, "asset_id": asset, "status": champ.get("status"),
            "paper_return_pct": paper_ret, "replay_return_pct": replay_ret,
            "divergence_pp": round(div, 4) if div is not None else None,
            "divergence_breach": bool(div > MAX_WEEKLY_DIVERGENCE_PP) if div is not None else None,
            "evidence_class": "research_only",
        }
        with STORE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        written += 1
        print(f"  {sid:26} {wk}  paper={paper_ret}  replay={replay_ret}  div={row['divergence_pp']}")

    print(f"\n{written} fila(s) escritas -> {STORE}")
    return 0


def report() -> int:
    if not STORE.exists():
        print("Sin ledger todavia. Corre --record semanalmente tras publicar senales.")
        print("El reloj del protocolo no puede evaluarse sin esto.")
        return 0

    rows = [json.loads(l) for l in STORE.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # A corrected week appears as a later row; the report reads the newest per (strategy, week)
    # while the ledger keeps both.
    df = df.sort_values("recorded_at").drop_duplicates(["strategy_id", "week"], keep="last")

    print("=" * 78)
    print("FORWARD LEDGER")
    print("=" * 78)
    for sid, g in df.groupby("strategy_id"):
        g = g.sort_values("week")
        paper = g["paper_return_pct"].dropna().to_numpy(float) / 100.0
        divs = g["divergence_pp"].dropna().to_numpy(float)
        weeks = len(g)
        print(f"\n{sid}  ({g['asset_id'].iloc[0]})  semanas registradas={weeks}")
        if weeks < 4:
            # The protocols require 16-26 weeks. Printing a Sharpe off 2 points would be exactly
            # the small-sample theatre the constitution forbids.
            print(f"  N={weeks}: solo se reporta conteo. Se necesitan >=16 semanas "
                  f"(protocolo de retiro) antes de cualquier metrica.")
            continue
        st = _ann_return_dd_calmar(paper, 52)
        print(f"  ann={st['ann_return_pct']}%  MaxDD={st['max_dd_pct']}%  Calmar={st['calmar']}")
        if divs.size:
            print(f"  divergencia media={divs.mean():.3f}pp  max={divs.max():.3f}pp  "
                  f"(umbral {MAX_WEEKLY_DIVERGENCE_PP}pp)  brechas={int((divs > MAX_WEEKLY_DIVERGENCE_PP).sum())}")
    print(f"\nledger: {STORE}  ({len(rows)} filas totales, historial completo)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--week", default=None, help="ISO week, e.g. 2026-W30")
    a = ap.parse_args()
    if a.record:
        return record(a.week)
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
