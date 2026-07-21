"""Recompute sigma_trials REAL de las 42 celdas pagadas — COP-NULL OLA 4 (0 trials nuevos).

Contract: CTR-QUANT-CONSTITUTION-001 §2 · deuda del panel 4/4 (2026-07-21):
"la tabla DSR usa un grid ASUMIDO [0.05,0.10,0.15]; sigma_trials nunca se persistió".

Qué hace (re-medición de celdas YA CONTAMINADAS, prohibido elegir nada):
1. UNA pasada del motor actual (purgado + fills open-aware, datos reparados) sobre 2025
   con collect_week_data → señales/leverage/entradas por semana.
2. Para cada una de las 42 celdas históricas (tp_r × hs_m — la MISMA rejilla de
   FC-H5-SIMPLE-001, diagnose_smart_simple_v1.py:531-532) se re-simulan SOLO las salidas.
3. σ_trials = std del Sharpe semanal entre celdas; N_eff por clustering de correlación
   entre las series de retornos de las celdas (López de Prado).
4. Recalcula la tabla DSR del registry con σ medida + N_eff, vía dsr_report (SSOT).

Salida: .claude/evidence/cop_sigma_trials/<fecha>/sigma_trials.json + este script es el
generador persistido (deuda de reproducibilidad del panel).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

TP_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
HS_MULTS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
SLIPPAGE = 0.0001


def main() -> int:
    from train_and_export_smart_simple import (load_config, load_data,
                                               run_production_backtest, simulate_week)
    from src.forecasting.enhance_v2 import enhance_features_v2
    from src.forecasting.adaptive_stops import AdaptiveStopsConfig, compute_adaptive_stops
    from services.common.metrics import dsr_report

    cfg = load_config()
    df, feats = load_data()
    df, feats = enhance_features_v2(df, feats)
    # collect_week_data=True via el wrapper de produccion (misma señal para toda celda)
    res = run_production_backtest(df, feats, cfg, 2025)
    weeks = [w for w in res.get("week_data", []) if w.get("trade")]
    print(f"semanas ejecutadas 2025 (motor purgado): {len(weeks)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    def bars_for(monday):
        m = pd.Timestamp(monday)
        rows = df[(df["date"] > m) & (df["date"] <= m + pd.offsets.BDay(4))]
        return [{"open": float(r.open), "high": float(r.high),
                 "low": float(r.low), "close": float(r.close)}
                for r in rows.itertuples()]

    cell_rets = {}
    for tp_r in TP_RATIOS:
        for hs_m in HS_MULTS:
            sc = AdaptiveStopsConfig(vol_multiplier=hs_m, tp_ratio=tp_r)
            rets = []
            for w in weeks:
                stops = compute_adaptive_stops(w["rv_ann"], sc)
                bars = bars_for(w["signal_date"])
                if not bars:
                    continue
                exit_p, reason, _ = simulate_week(
                    w["direction"], w["entry_price"], bars,
                    stops.hard_stop_pct, stops.take_profit_pct)
                lev = w["final_lev"]
                pnl = w["direction"] * (exit_p - w["entry_price"]) / w["entry_price"] * lev
                cost = SLIPPAGE * lev if reason == "week_end" else 0.0
                rets.append(pnl - cost)
            cell_rets[(tp_r, hs_m)] = np.array(rets)

    sharpes = {}
    for k, r in cell_rets.items():
        sharpes[k] = float(np.mean(r) / np.std(r, ddof=1)) if len(r) > 2 and np.std(r) > 0 else 0.0
    sh = np.array(list(sharpes.values()))
    sigma_measured = float(np.std(sh, ddof=1))

    # N_eff por clustering de correlacion (umbral 0.95 = celdas casi identicas)
    keys = list(cell_rets.keys())
    n = min(len(cell_rets[k]) for k in keys)
    M = np.column_stack([cell_rets[k][:n] for k in keys])
    C = np.corrcoef(M.T)
    assigned, clusters = set(), 0
    for i in range(len(keys)):
        if i in assigned:
            continue
        clusters += 1
        for j in range(i, len(keys)):
            if C[i, j] > 0.95:
                assigned.add(j)
    print(f"sigma_trials MEDIDA (Sharpe semanal entre 42 celdas): {sigma_measured:.4f}")
    print(f"N_eff por clusters de correlacion >0.95: {clusters} (de 42 celdas)")

    # DSR de v11 con sigma medida — Sharpe semanal honesto actual (+7.35% serie)
    trades = json.loads((REPO / "usdcop-trading-dashboard/public/data/production/trades/"
                         "smart_simple_v11_2025.json").read_text(encoding="utf-8"))["trades"]
    wk = np.array([t["pnl_pct"] for t in trades], dtype=float) / 100
    sr_weekly = float(np.mean(wk) / np.std(wk, ddof=1))
    out = {"grid": {"tp_ratios": TP_RATIOS, "hs_mults": HS_MULTS},
           "n_weeks_engine": len(weeks),
           "sigma_trials_measured": sigma_measured,
           "n_eff_corr_clusters": clusters,
           "sharpe_weekly_v11_honesto": sr_weekly,
           "cell_sharpes": {f"tp{k[0]}_hs{k[1]}": v for k, v in sharpes.items()},
           "dsr": {}}
    for label, n_tr in (("N=59_nominal", 59), (f"N_eff={clusters}", clusters),
                        (f"N_eff+17_no_grid", clusters + 17)):
        rep = dsr_report(sr_weekly, len(wk), n_tr,
                         sigma_grid=(sigma_measured,), periods_per_year=52,
                         skew=float(pd.Series(wk).skew()),
                         kurtosis=float(pd.Series(wk).kurt()) + 3)
        out["dsr"][label] = rep
        print(f"DSR ({label}, sigma medida {sigma_measured:.3f}): {rep}")

    o = REPO / ".claude/evidence/cop_sigma_trials" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "sigma_trials.json").write_text(json.dumps(out, indent=2, default=str))
    import shutil
    shutil.copy(__file__, o / "generator_script.py")   # reproducibilidad exigida por el panel
    print(f"artefacto -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
