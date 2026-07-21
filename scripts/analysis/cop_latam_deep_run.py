"""H-LATAM-02 con historia profunda — la hipótesis DESBLOQUEADA por el backfill (+1 trial).

Contract: CTR-QUANT-CONSTITUTION-001 · Pre-registro original en el registry COP:
"H-LATAM-02: ΔCalmar(TSMOM basket, B1′ basket)" sobre {COP, MXN, BRL}, votos TSMOM
4/8/13 semanas, posición causal shift(1) — BLOCKED_DATA con 17 semanas de MXN/BRL.
El backfill máximo (2026-07-21) entregó: COP 1989→, MXN 1990→, BRL 1994→ (diario).

Ventanas pre-declaradas AQUÍ, antes de correr (este archivo es el generador persistido):
- DISEÑO: ≤2024-12-31 (décadas completas; se reporta por década además del agregado).
- OOS: 2025 completo, UN disparo. Juez del trial: ΔCalmar(basket, B1′) en OOS-2025 con
  IC95 block-bootstrap (b=4 semanas) que excluya 0. 2026 NO se toca (forward futuro).
- Mecánica EXACTA del pre-registro: votos = Σ sign(ret_w) para w∈{4,8,13} / 3;
  pos = votes.shift(1); sin vol-targeting, sin carry (H-LATAM-01 sigue gated), sin CLP
  (universo registrado). Cambiar cualquier cosa = otra hipótesis.

Costo: +1 trial al abrir el OOS (N 64→65 en el registry COP).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO))
from services.common.metrics import circular_block_bootstrap  # noqa: E402

WEEKS = 52
PAIRS = {"COP": "USD/COP", "MXN": "USD/MXN", "BRL": "USD/BRL"}


def ann_dd_calmar(x, ppy=52):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 10:
        return {"n": int(len(x))}
    eq = np.cumprod(1 + x)
    peak = np.maximum.accumulate(eq)
    dd = float(np.min(eq / peak - 1))
    ann = eq[-1] ** (ppy / len(x)) - 1
    return {"ann_pct": round(ann * 100, 2), "maxdd_pct": round(dd * 100, 2),
            "calmar": round(ann / abs(dd), 3) if dd < 0 else None, "n": int(len(x))}


def load_weekly_closes():
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("POSTGRES_HOST", "localhost"),
                            dbname="usdcop_trading", user="admin",
                            password=os.environ.get("POSTGRES_PASSWORD", ""))
    out = {}
    for k, sym in PAIRS.items():
        d = pd.read_sql("SELECT time, close FROM asset_daily_ohlcv WHERE symbol=%s ORDER BY time",
                        conn, params=(sym,))
        d["time"] = pd.to_datetime(d["time"], utc=True).dt.tz_localize(None)
        s = d.set_index("time")["close"].astype(float)
        # pre-1991 COP es tasa de referencia plana — se usa igual (los votos TSMOM sobre
        # una serie administrada dan pos≈constante; se reporta por década, no se oculta)
        out[k] = s.resample("W-FRI").last().dropna()
    return out


def tsmom(wclose):
    wk = wclose.pct_change()
    votes = sum(np.sign(wclose.pct_change(w).fillna(0.0)) for w in (4, 8, 13)) / 3.0
    pos = votes.shift(1)
    return (pos * wk).dropna(), (pos.abs().mean() * wk).dropna()


def main():
    closes = load_weekly_closes()
    strat, b1p = {}, {}
    for k, s in closes.items():
        strat[k], b1p[k] = tsmom(s)
        print(f"{k}: semanas {len(strat[k])} ({strat[k].index.min().date()} -> {strat[k].index.max().date()})")

    basket = pd.concat(strat, axis=1).mean(axis=1).dropna()
    basket_b1p = pd.concat(b1p, axis=1).mean(axis=1).reindex(basket.index).fillna(0.0)

    res = {"universe": list(PAIRS), "mechanics": "votes(4/8/13w)/3, shift(1), sin vol-target/carry/CLP",
           "design": {}, "oos_2025": {}, "per_decade": {}}
    dz = basket[basket.index <= "2024-12-31"]
    dzp = basket_b1p.reindex(dz.index)
    res["design"]["basket"] = ann_dd_calmar(dz.values)
    res["design"]["b1prime"] = ann_dd_calmar(dzp.values)
    for dec in range(1990, 2030, 10):
        seg = dz[(dz.index.year >= dec) & (dz.index.year < dec + 10)]
        if len(seg) > 50:
            res["per_decade"][f"{dec}s"] = ann_dd_calmar(seg.values)
    print("\nDISEÑO ≤2024:", res["design"]["basket"], "| B1':", res["design"]["b1prime"])
    for d, v in res["per_decade"].items():
        print(f"  {d}: {v}")

    # ── UN disparo OOS-2025 (abre el trial) ──
    oz = basket[(basket.index >= "2025-01-01") & (basket.index <= "2025-12-31")]
    ozp = basket_b1p.reindex(oz.index)
    res["oos_2025"]["basket"] = ann_dd_calmar(oz.values)
    res["oos_2025"]["b1prime"] = ann_dd_calmar(ozp.values)

    def calmar_stat(x):
        eq = np.cumprod(1 + x); peak = np.maximum.accumulate(eq)
        dd = float(np.min(eq / peak - 1))
        ann = eq[-1] ** (52 / len(x)) - 1
        return ann / abs(dd) if dd < 0 else 0.0

    delta = (oz - ozp).dropna().values
    bb = circular_block_bootstrap(delta, np.mean, block=4)
    # ΔCalmar pareado por bootstrap de indices comunes
    rng = np.random.default_rng(42)
    oz_v, ozp_v = oz.values, ozp.reindex(oz.index).fillna(0.0).values
    n = len(oz_v)
    dels = []
    for _ in range(2000):
        starts = rng.integers(0, n, size=int(np.ceil(n / 4)))
        idx = (starts[:, None] + np.arange(4)[None, :]).ravel() % n
        idx = idx[:n]
        dels.append(calmar_stat(oz_v[idx]) - calmar_stat(ozp_v[idx]))
    ci = [float(np.percentile(dels, 2.5)), float(np.percentile(dels, 97.5))]
    dc = calmar_stat(oz_v) - calmar_stat(ozp_v)
    res["oos_2025"]["delta_calmar"] = round(dc, 3)
    res["oos_2025"]["delta_calmar_ci95_block4"] = [round(c, 3) for c in ci]
    res["oos_2025"]["delta_mean_weekly_ci95"] = bb["ci95"]
    win = ci[0] > 0
    res["verdict"] = ("RECHAZA H0 — la cesta TSMOM bate a B1' en OOS-2025" if win
                      else "NO_RECHAZA — cesta no distinguible de exposición pasiva emparejada en 2025")
    res["trials"] = "+1 (N 64->65)"
    print(f"\nOOS-2025 basket: {res['oos_2025']['basket']} | B1': {res['oos_2025']['b1prime']}")
    print(f"ΔCalmar = {dc:.3f}, IC95 block-4 = {ci} -> {'WIN' if win else 'NO_RECHAZA'}")

    o = REPO / ".claude/evidence/cop_latam_deep" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "h_latam_02_deep.json").write_text(json.dumps(res, indent=2, default=str))
    shutil.copy(__file__, o / "generator_script.py")
    print(f"artefacto -> {o}")


if __name__ == "__main__":
    main()
