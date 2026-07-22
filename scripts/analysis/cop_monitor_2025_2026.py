"""Monitoreo descriptivo v11/v12/v14 — OOS-2025 (celdas ya pagadas) + replay 2026 YTD.

PRE-DECLARACIÓN (antes de correr):
- 2025 para las tres configs = RE-LECTURA de celdas ya medidas (0 trials nuevos).
- 2026 YTD para v12/v14 = celdas NUEVAS abiertas por monitoreo → +2 trials (N 69→71).
  Se corre para RESPONDER "¿cómo va el año?", NO para seleccionar entre configs.
  El juez sellado de v12/v14 sigue siendo el forward POST-freeze (paper 2026-07-27);
  el tramo Ene–Jul 2026 queda marcado como MIRADO y jamás podrá reclamarse como forward
  limpio de v12/v14. Para v11, 2026 es su forward real ya reportado (+3.36%).
- Prohibido promover/elegir nada con base en este replay (quant-constitution §1).
"""
from __future__ import annotations

import json
import shutil
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

CONFIGS = {
    "v11_produccion": {},
    "v12_cap15": {"vt_max": 1.5},
    "v14_cap15_ladder": {"vt_max": 1.5, "ladder_enabled": True, "ladder_k2": 2.0},
}


def monthly_table(trades):
    by_m = defaultdict(list)
    for t in trades:
        m = str(t.get("timestamp", ""))[:7]
        by_m[m].append(t["pnl_pct"])
    eq = 10000.0
    rows = []
    for m in sorted(by_m):
        ret_m = float(np.prod([1 + p / 100 for p in by_m[m]]) - 1) * 100
        eq *= 1 + ret_m / 100
        rows.append({"mes": m, "trades": len(by_m[m]), "ret_pct": round(ret_m, 2),
                     "equity": round(eq, 0)})
    return rows


def main() -> int:
    from train_and_export_smart_simple import load_config, load_data, run_production_backtest
    from src.forecasting.enhance_v2 import enhance_features_v2

    cfg0 = load_config()
    df, feats = load_data()
    df, feats = enhance_features_v2(df, feats)

    out = {"note": "2025=re-lectura celdas pagadas; 2026 v12/v14=replay descriptivo (+2 trials); "
                   "juez v12/v14 = forward post-freeze 2026-07-27",
           "results": {}}
    for name, over in CONFIGS.items():
        out["results"][name] = {}
        for year in (2025, 2026):
            c = dict(cfg0)
            c.update(over)
            r = run_production_backtest(df, feats, c, year)
            trades = r["trades"]
            met = r.get("metrics", {})
            eq = [10000.0]
            for t in trades:
                eq.append(eq[-1] * (1 + t["pnl_pct"] / 100))
            eqa = np.array(eq)
            dd = float(np.min(eqa / np.maximum.accumulate(eqa) - 1)) * 100
            hs = sum(1 for t in trades if "hard_stop" in str(t.get("exit_reason", "")))
            summ = {"ret_pct": round((eq[-1] / 10000 - 1) * 100, 2),
                    "maxdd_pct": round(dd, 2), "n_trades": len(trades),
                    "hard_stops": hs,
                    "wr_pct": round(100 * sum(1 for t in trades if t["pnl_pct"] > 0) / max(len(trades), 1), 1),
                    "bh_pct": met.get("bh_return"),
                    "monthly": monthly_table(trades),
                    "trades": [{k: t.get(k) for k in ("timestamp", "side", "entry_price",
                                                      "exit_price", "exit_reason", "leverage", "pnl_pct")}
                               for t in trades]}
            out["results"][name][str(year)] = summ
            print(f"{name} {year}: ret={summ['ret_pct']}% dd={summ['maxdd_pct']} "
                  f"trades={summ['n_trades']} HS={hs} WR={summ['wr_pct']}%", flush=True)

    o = REPO / ".claude/evidence/cop_monitor_2025_2026" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "monitor.json").write_text(json.dumps(out, indent=2, default=str))
    shutil.copy(__file__, o / "generator_script.py")
    print(f"artefacto -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
