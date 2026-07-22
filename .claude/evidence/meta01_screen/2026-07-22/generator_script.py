"""H-META-01 SCREENING — c1 (consenso temporal 21 votos) y c2 (dispersión) vs null endurecido.

ABRE la tabla: +2 trials (N 86→88). Pre-registro sellado + enmienda #1 (registry
2026-07-22) — implementación EXACTA:
- B = v12 + techo v13 (flag ON, cutoff diseño), trades 2020-2024, una fila por semana.
- c1(t) = media igual-peso de 21 votos: 7 modelos × orígenes t, t−1, t−2; voto = 1 si
  sign(pred)==lado H5 actual, 0 si opuesto; pred cero/ausente = 0.5. Sin backfill.
- c2(t) = −std transversal de los 7 preds del origen t, estandarizados por la escala
  expanding PROPIA de cada modelo (std de sus preds estrictamente < t, mín 20 obs).
  Regla determinista declarada: si hay <5 de 7 preds estandarizables, c2 = NaN y la
  fila se excluye SOLO de la celda c2 (no de c1).
- Target = PnL a exposición UNITARIA: dir·(exit−entry)/entry (dirección/entrada/salida
  de B congeladas; el leverage incumbente NO entra al target).
- Null = OLS[1, lado, leverage_pre_meta]; candidata añade SOLO c1 o SOLO c2.
- CV: 5 folds EXPANDING contiguos con purga de 1 semana; preprocessing con train only.
- Bar: upperIC95(ΔMSE OOF, pareado, block-bootstrap circular b=4 FIJO) < 0 Y signo
  económico correcto (coef>0 en ambas celdas por construcción).
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "pipeline"))

from services.common.metrics import circular_block_bootstrap  # noqa: E402

LEDGER = REPO / "data/pipeline/meta01/zoo_ledger.parquet"
V12_CONFIG = REPO / "config/execution/smart_simple_v12_lev_cap.yaml"
DESIGN_YEARS = (2020, 2021, 2022, 2023, 2024)
DESIGN_CUTOFF = pd.Timestamp("2024-12-31")
MODELS7 = ["ard", "xgboost_pure", "lightgbm_pure", "catboost_pure",
           "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost"]


def collect_b_trades():
    import copy
    from train_and_export_smart_simple import (load_config, load_data,
                                               run_walkforward_backtest,
                                               compute_v13_leverage_ceilings)
    from src.forecasting.enhance_v2 import enhance_features_v2
    cfg = load_config(config_path=str(V12_CONFIG), version_override="13.0.0",
                      strategy_id="smart_simple_v13_qrisk")
    cfg["v13_ceiling_enabled"] = True
    df, feats = load_data()
    df, feats = enhance_features_v2(df, feats)
    ceilings, meta = compute_v13_leverage_ceilings(df, cutoff=DESIGN_CUTOFF)
    cfg["_v13_ceiling_cache"], cfg["_v13_ceiling_meta"] = ceilings, meta
    rows = []
    for year in DESIGN_YEARS:
        r = run_walkforward_backtest(df, feats, cfg, year)
        for t in r["trades"]:
            d = +1 if t["side"] == "LONG" else -1
            rows.append({"monday": pd.Timestamp(str(t["timestamp"])[:10]).normalize(),
                         "side": d, "leverage": float(t["leverage"]),
                         "unit_pnl": d * (t["exit_price"] - t["entry_price"]) / t["entry_price"]})
        print(f"  B {year}: {len(r['trades'])} trades", flush=True)
    B = pd.DataFrame(rows).sort_values("monday").reset_index(drop=True)
    # normaliza monday al inicio de semana ISO (los trades llevan la fecha del lunes real)
    B["week_monday"] = B["monday"].dt.to_period("W-SUN").dt.start_time
    return B


def build_cells(B, L):
    L = L.copy()
    L["week_monday"] = pd.to_datetime(L["monday"]).dt.to_period("W-SUN").dt.start_time
    piv = L.pivot_table(index="week_monday", columns="model_id", values="pred_h5",
                        aggfunc="last").sort_index()
    piv = piv.reindex(columns=MODELS7)
    # escala expanding causal por modelo (std de preds < t, min 20)
    z = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float)
    for m in MODELS7:
        s = piv[m]
        sd = s.expanding(20).std().shift(1)
        z[m] = s / sd.replace(0, np.nan)

    weeks = list(piv.index)
    wk_pos = {w: i for i, w in enumerate(weeks)}
    c1_list, c2_list = [], []
    for _, r in B.iterrows():
        w = r["week_monday"]
        votes = []
        for lag in (0, 1, 2):
            if w in wk_pos and wk_pos[w] - lag >= 0:
                wl = weeks[wk_pos[w] - lag]
                preds = piv.loc[wl]
            else:
                preds = pd.Series(np.nan, index=MODELS7)
            for m in MODELS7:
                p = preds[m]
                if pd.isna(p) or p == 0:
                    votes.append(0.5)               # regla 5c
                else:
                    votes.append(1.0 if np.sign(p) == r["side"] else 0.0)
        c1_list.append(float(np.mean(votes)))
        if w in wk_pos:
            zz = z.loc[w].dropna()
            c2_list.append(float(-zz.std(ddof=1)) if len(zz) >= 5 else np.nan)
        else:
            c2_list.append(np.nan)
    B = B.copy()
    B["c1"], B["c2"] = c1_list, c2_list
    return B


def expanding_folds(n, k=5, purge=1):
    # 5 bloques de test contiguos tras un train inicial; train = todo lo ANTERIOR - purga
    edges = np.linspace(n // 3, n, k + 1).astype(int)
    for i in range(k):
        te = np.arange(edges[i], edges[i + 1])
        tr = np.arange(0, max(0, edges[i] - purge))
        if len(tr) >= 20 and len(te) > 0:
            yield tr, te


def oof_mse(B, extra_col=None):
    cols = ["side", "leverage"] + ([extra_col] if extra_col else [])
    d = B.dropna(subset=cols + ["unit_pnl"]).reset_index(drop=True)
    y = d["unit_pnl"].to_numpy()
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    err = np.full(len(d), np.nan)
    coefs = []
    for tr, te in expanding_folds(len(d)):
        beta, *_ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
        err[te] = (y[te] - X[te] @ beta) ** 2
        coefs.append(beta[-1] if extra_col else np.nan)
    m = np.isfinite(err)
    return d.index[m], err[m], (float(np.nanmean(coefs)) if extra_col else None), d


def main() -> int:
    L = pd.read_parquet(LEDGER)
    print(f"ledger: {len(L)} filas, {L['monday'].nunique()} semanas")
    B = collect_b_trades()
    print(f"trades B (diseño 2020-24): {len(B)}")
    B = build_cells(B, L)

    out = {"hypothesis": "H-META-01 screening (enmienda #1)", "n_trades": len(B),
           "trials": "+2 (N 86->88)", "cells": {}}
    for cell in ("c1", "c2"):
        idx_n, err_null, _, d_null = oof_mse(B.dropna(subset=[cell]))
        idx_c, err_cell, coef, d_cell = oof_mse(B.dropna(subset=[cell]), extra_col=cell)
        # pareado sobre el solape de indices OOF
        common = sorted(set(idx_n) & set(idx_c))
        en = pd.Series(err_null, index=idx_n).loc[common].to_numpy()
        ec = pd.Series(err_cell, index=idx_c).loc[common].to_numpy()
        delta = ec - en                                  # <0 = la celda mejora
        bb = circular_block_bootstrap(delta, np.mean, block=4)
        ci = bb["ci95"]
        sign_ok = coef is not None and coef > 0
        win = ci[1] is not None and ci[1] < 0 and sign_ok
        out["cells"][cell] = {
            "n_oof": len(common),
            "mse_null": float(np.mean(en)), "mse_cell": float(np.mean(ec)),
            "delta_mse_mean": float(np.mean(delta)),
            "delta_ci95_block4": ci, "coef_oof_mean": coef,
            "sign_economico_ok": bool(sign_ok), "WIN": bool(win)}
        print(f"{cell}: n={len(common)} MSE {np.mean(ec):.6f} vs null {np.mean(en):.6f} "
              f"dCI95={ci} coef={coef:.4f} WIN={win}")
    wins = [c for c, r in out["cells"].items() if r["WIN"]]
    out["verdict"] = (f"WIN: {wins} -> variante económica habilitada (+1 trial)"
                      if wins else
                      "NO_RECHAZA ambas — el consenso del zoo NO añade sobre lado+leverage; "
                      "familia CERRADA; la combinada óptima sigue siendo v13")
    o = REPO / ".claude/evidence/meta01_screen" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "screen.json").write_text(json.dumps(out, indent=2, default=str))
    B.to_csv(o / "design_table.csv", index=False)
    shutil.copy(__file__, o / "generator_script.py")
    print(f"\nVEREDICTO: {out['verdict']}")
    print(f"artefacto -> {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
