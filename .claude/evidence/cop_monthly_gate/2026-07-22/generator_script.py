"""H-MONTHLY-01 etapa (i) — gate predictivo del reloj mensual (ABRE 2 celdas: +2 trials).

Pre-registro: HYPOTHESIS-REGISTRY.md "H-RISK-FAM-02 + H-MONTHLY-01 (2026-07-22)".
DOS celdas pre-declaradas, sin seleccion posterior:
  m1 = devaluacion implicita BanRep (macro_banrep_forwards_monthly, tenor '91 a 180',
       published_at <= fin de mes de decision — regla PIT +60d ya en tabla)
  m2 = pit_eme_curve_12m_pct = eme_12m_mean / eme_near_mean - 1 (available_at <= eom)
Target: log-retorno del MES SIGUIENTE de USD/COP (seed diario). SOLO diseno <=2024.
Gate por celda: IC (Spearman) con IC95 block-bootstrap (b=3 meses, 2000, seed 42)
que excluya 0. Las etapas (ii)-(iv) siguen selladas: nada economico se abre aqui.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from src.data.usdcop_forward_macro import DEFAULT_OUTPUT  # noqa: E402

END_DESIGN = pd.Timestamp("2024-12-31")
SEED, N_BOOT, BLOCK = 42, 2000, 3


def block_boot_spearman(x, y, n_boot=N_BOOT, block=BLOCK, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(x)
    stats = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=int(np.ceil(n / block)))
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel() % n
        idx = idx[:n]
        r = spearmanr(x[idx], y[idx]).statistic
        if np.isfinite(r):
            stats.append(r)
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main() -> int:
    d = pd.read_parquet(REPO / "seeds/latest/usdcop_daily_ohlcv.parquet")
    d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None)
    d = d.sort_values("time")
    mclose = d.set_index("time")["close"].resample("ME").last().dropna()
    next_ret = np.log(mclose.shift(-1) / mclose)          # retorno del mes siguiente

    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("POSTGRES_HOST", "localhost"),
                            dbname="usdcop_trading", user="admin",
                            password=os.environ.get("POSTGRES_PASSWORD", ""))
    dev = pd.read_sql("""SELECT month, implied_dev, published_at
                         FROM macro_banrep_forwards_monthly
                         WHERE tenor='91 a 180' ORDER BY month""", conn).dropna()
    dev["published_at"] = pd.to_datetime(dev["published_at"])

    pit = pd.read_parquet(DEFAULT_OUTPUT)

    def pit_series(sid):
        g = pit[pit["series_id"] == sid].copy()
        g["available_at"] = pd.to_datetime(g["available_at"]).dt.tz_localize(None)
        return (g.sort_values("available_at")
                 .drop_duplicates("available_at", keep="last")[["available_at", "value"]])

    eme12, emen = pit_series("br_eme_usdcop_12m_mean"), pit_series("br_eme_usdcop_near_mean")

    rows = []
    for eom in mclose.index:
        if eom > END_DESIGN or eom not in next_ret.index or not np.isfinite(next_ret.loc[eom]):
            continue
        dv = dev[dev["published_at"] <= eom]
        m1 = float(dv["implied_dev"].iloc[-1]) if len(dv) else np.nan
        e12 = eme12[eme12["available_at"] <= eom]
        enr = emen[emen["available_at"] <= eom]
        m2 = (float(e12["value"].iloc[-1]) / float(enr["value"].iloc[-1]) - 1.0) \
            if len(e12) and len(enr) and float(enr["value"].iloc[-1]) > 0 else np.nan
        rows.append({"eom": eom, "next_ret": float(next_ret.loc[eom]), "m1": m1, "m2": m2})
    D = pd.DataFrame(rows)

    out = {"hypothesis": "H-MONTHLY-01 etapa (i)", "design_end": str(END_DESIGN.date()),
           "cells": {}, "trials": "+2 (una por celda)"}
    for cell in ("m1", "m2"):
        sub = D.dropna(subset=[cell, "next_ret"])
        n = len(sub)
        if n < 36:
            out["cells"][cell] = {"n": n, "verdict": "INSUFICIENTE (<36 meses)"}
            print(f"{cell}: n={n} INSUFICIENTE")
            continue
        ic = float(spearmanr(sub[cell], sub["next_ret"]).statistic)
        ci = block_boot_spearman(sub[cell].to_numpy(), sub["next_ret"].to_numpy())
        win = ci[0] > 0 or ci[1] < 0
        out["cells"][cell] = {"n": n, "ic_spearman": round(ic, 4),
                              "ci95_block3": [round(c, 4) for c in ci],
                              "first": str(sub.eom.min().date()),
                              "last": str(sub.eom.max().date()),
                              "WIN": bool(win)}
        print(f"{cell}: n={n} ({sub.eom.min().date()}->{sub.eom.max().date()}) "
              f"IC={ic:.4f} CI95={ci} WIN={win}")
    wins = [c for c, r in out["cells"].items() if r.get("WIN")]
    out["verdict"] = ("etapa (ii) habilitada para: " + ",".join(wins)) if wins else \
        "NO_RECHAZA en ambas celdas — etapa (ii) NO se abre; reloj mensual cerrado hasta nuevo pre-registro"
    o = REPO / ".claude/evidence/cop_monthly_gate" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "monthly_gate.json").write_text(json.dumps(out, indent=2, default=str))
    shutil.copy(__file__, o / "generator_script.py")
    print(f"\nVEREDICTO: {out['verdict']}")
    print(f"artefacto -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
