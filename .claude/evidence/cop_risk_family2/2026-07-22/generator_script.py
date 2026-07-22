"""H-RISK-FAM-02 — screening familia de riesgo EME/SFC/forward (ABRE resultados: +5 trials).

Contract: CTR-QUANT-CONSTITUTION-001 · Pre-registro: HYPOTHESIS-REGISTRY.md
"H-RISK-FAM-02 + H-MONTHLY-01 (PRE-REGISTRO 2026-07-22)" — protocolo CLONADO de
H-RISK-FAM-01 (cop_risk_family_screen.py) sin desviaciones: mismo universo <=2024,
mismos targets (gap_week p90 rolling-252 + q90 rango semanal), mismo purged K-fold,
mismo bar (Brier vs frecuencia base; pinball vs NULL intercepto-solo en el mismo CV),
mismo IC95 block-bootstrap b=4. Solo cambia la familia de features (g1..g5).

PIT: features EME/SFC desde USDCOP_FORWARD_MACRO_PIT.parquet via available_at <= asof
(viernes previo a la semana operada); g5 desde macro_banrep_forwards_monthly con
published_at <= asof (regla +60d ya en tabla). ADVERTENCIA declarada en el pre-registro:
estas features llegan pre-contaminadas por la seleccion del audit externo (evento #3);
el screening <=2024 mitiga, no elimina — cualquier uso economico se juzga en forward.
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from services.common.metrics import circular_block_bootstrap  # noqa: E402
from src.data.usdcop_forward_macro import DEFAULT_OUTPUT  # noqa: E402

END_DESIGN = pd.Timestamp("2024-12-31")


def pit_series(pit: pd.DataFrame, series_id: str) -> pd.DataFrame:
    g = pit[pit["series_id"] == series_id].copy()
    g["available_at"] = pd.to_datetime(g["available_at"]).dt.tz_localize(None)
    g = g.sort_values(["available_at"]).drop_duplicates("available_at", keep="last")
    return g[["available_at", "value"]].reset_index(drop=True)


def asof_value(g: pd.DataFrame, ts: pd.Timestamp):
    s = g[g["available_at"] <= ts]
    return float(s["value"].iloc[-1]) if len(s) else np.nan


def asof_delta(g: pd.DataFrame, ts: pd.Timestamp):
    s = g[g["available_at"] <= ts]
    if len(s) < 2:
        return np.nan
    return float(s["value"].iloc[-1] - s["value"].iloc[-2])


def asof_relchg(g: pd.DataFrame, ts: pd.Timestamp):
    s = g[g["available_at"] <= ts]
    if len(s) < 2 or abs(s["value"].iloc[-2]) < 1e-12:
        return np.nan
    return float(abs(s["value"].iloc[-1] / s["value"].iloc[-2] - 1.0))


def load_weekly():
    d = pd.read_parquet(REPO / "seeds/latest/usdcop_daily_ohlcv.parquet")
    d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None)
    d = d[d["time"] <= END_DESIGN].sort_values("time").reset_index(drop=True)
    d["ret"] = np.log(d["close"] / d["close"].shift(1))
    d["gap"] = (d["open"] / d["close"].shift(1) - 1).abs()
    d["dow"] = d["time"].dt.dayofweek
    d["week"] = d["time"].dt.to_period("W-SUN")

    pit = pd.read_parquet(DEFAULT_OUTPUT)
    near_mean = pit_series(pit, "br_eme_usdcop_near_mean")
    near_std = pit_series(pit, "br_eme_usdcop_near_std")
    near_med = pit_series(pit, "br_eme_usdcop_near_median")
    sfc_net = pit_series(pit, "sfc_pension_deriv_usd_net_m")
    sfc_gross = pit_series(pit, "sfc_pension_deriv_usd_gross_m")

    # g5: implied dev mensual desde la tabla PIT propia (published_at = eom+60d)
    import psycopg2
    conn = psycopg2.connect(host=os.environ.get("POSTGRES_HOST", "localhost"),
                            dbname="usdcop_trading", user="admin",
                            password=os.environ.get("POSTGRES_PASSWORD", ""))
    dev = pd.read_sql("""SELECT month, implied_dev, published_at
                         FROM macro_banrep_forwards_monthly
                         WHERE tenor = '91 a 180' ORDER BY month""", conn)
    dev["available_at"] = pd.to_datetime(dev["published_at"])
    dev = dev.rename(columns={"implied_dev": "value"})[["available_at", "value"]].dropna()

    rows = []
    for wk, g in d.groupby("week"):
        if len(g) < 3:
            continue
        monday = g.iloc[0]
        prev = d[d["time"] < g["time"].iloc[0]]
        if len(prev) < 260:
            continue
        prev_close = prev["close"].iloc[-1]
        rng_pct = (g["high"].max() - g["low"].min()) / prev_close * 100
        gap_bps = monday["gap"] * 1e4
        p90_gap = prev.loc[prev["dow"] == 0, "gap"].tail(252).quantile(0.90) * 1e4
        asof = prev["time"].iloc[-1]

        nm, ns = asof_value(near_mean, asof), asof_value(near_std, asof)
        g1 = ns / abs(nm) if (np.isfinite(ns) and np.isfinite(nm) and abs(nm) > 0) else np.nan
        g2 = asof_relchg(near_med, asof)
        g3 = asof_delta(sfc_net, asof)
        net, gross = asof_value(sfc_net, asof), asof_value(sfc_gross, asof)
        g4 = net / abs(gross) if (np.isfinite(net) and np.isfinite(gross) and abs(gross) > 0) else np.nan
        g5 = asof_delta(dev, asof)

        rows.append({"monday": monday["time"], "gap_bps": gap_bps,
                     "gap_week": int(gap_bps > p90_gap), "range_pct": rng_pct,
                     "g1_eme_disp": g1, "g2_eme_rev": g2, "g3_sfc_flow": g3,
                     "g4_sfc_netgross": g4, "g5_dev_chg": g5,
                     "base_rate": float(prev.loc[prev["dow"] == 0, "gap"].tail(252)
                                        .gt(prev.loc[prev["dow"] == 0, "gap"].tail(252)
                                            .quantile(0.90)).mean())})
    return pd.DataFrame(rows)


def purged_kfold_idx(n, k=5, purge=1):
    fold = np.array_split(np.arange(n), k)
    for i in range(k):
        test = fold[i]
        tr = np.setdiff1d(np.arange(n), test)
        tr = tr[(tr < test.min() - purge) | (tr > test.max() + purge)]
        yield tr, test


def screen():
    from sklearn.linear_model import LogisticRegression, QuantileRegressor
    W = load_weekly()
    print(f"semanas de disenio: {len(W)} ({W.monday.min().date()} -> {W.monday.max().date()})")
    print(f"tasa base gap_week: {W.gap_week.mean():.3f}")
    cells = ["g1_eme_disp", "g2_eme_rev", "g3_sfc_flow", "g4_sfc_netgross", "g5_dev_chg"]
    results = {}
    for cell in cells:
        d = W.dropna(subset=[cell, "gap_week", "range_pct"]).reset_index(drop=True)
        n = len(d)
        if n < 60:
            results[cell] = {"n": n, "verdict": "INSUFICIENTE (<60 semanas)"}
            print(f"{cell}: n={n} INSUFICIENTE")
            continue
        x = d[cell].to_numpy().reshape(-1, 1)
        p_model = np.full(n, np.nan)
        for tr, te in purged_kfold_idx(n):
            if len(np.unique(d.gap_week.iloc[tr])) < 2:
                p_model[te] = d.gap_week.iloc[tr].mean()
                continue
            m = LogisticRegression().fit(x[tr], d.gap_week.iloc[tr])
            p_model[te] = m.predict_proba(x[te])[:, 1]
        brier_m = (p_model - d.gap_week) ** 2
        brier_b = (d.base_rate - d.gap_week) ** 2
        db = circular_block_bootstrap((brier_b - brier_m).to_numpy(), np.mean, block=4)

        q_model = np.full(n, np.nan)
        for tr, te in purged_kfold_idx(n):
            qr = QuantileRegressor(quantile=0.90, alpha=1e-4, solver="highs").fit(
                x[tr], d.range_pct.iloc[tr])
            q_model[te] = qr.predict(x[te])

        def pinball(y, q, tau=0.90):
            e = y - q
            return np.maximum(tau * e, (tau - 1) * e)
        pin_m = pinball(d.range_pct.to_numpy(), q_model)
        q_null = np.full(n, np.nan)
        for tr, te in purged_kfold_idx(n):
            q_null[te] = np.quantile(d.range_pct.iloc[tr], 0.90)
        pin_null = pinball(d.range_pct.to_numpy(), q_null)
        dp = circular_block_bootstrap(pin_null - pin_m, np.mean, block=4)
        win_brier = db["ci95"][0] is not None and db["ci95"][0] > 0
        win_pin = dp["ci95"][0] is not None and dp["ci95"][0] > 0
        results[cell] = {
            "n": n,
            "brier_model": float(np.nanmean(brier_m)), "brier_base": float(np.nanmean(brier_b)),
            "delta_brier_ci95": db["ci95"], "WIN_brier": bool(win_brier),
            "pinball_model": float(np.nanmean(pin_m)),
            "pinball_null_intercepto": float(np.nanmean(pin_null)),
            "delta_pinball_vs_null_ci95": dp["ci95"], "WIN_pinball": bool(win_pin),
        }
        print(f"{cell}: n={n} | Brier {np.nanmean(brier_m):.4f} vs base "
              f"{np.nanmean(brier_b):.4f} (WIN={win_brier}) | pinball "
              f"{np.nanmean(pin_m):.3f} vs NULL {np.nanmean(pin_null):.3f} (WIN={win_pin})")
    wins = [c for c, r in results.items() if r.get("WIN_brier") or r.get("WIN_pinball")]
    verdict = {"family": "H-RISK-FAM-02", "cells": results, "wins": wins,
               "trials_spent": 5, "consequence": (
                   "candidata v15 (features ganadoras a la QR del techo v13, 1 variante, "
                   "design-run pareado +1 trial, juez forward)"
                   if wins else "familia CERRADA por escrito")}
    o = REPO / ".claude/evidence/cop_risk_family2" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "screen_results.json").write_text(json.dumps(verdict, indent=2, default=str))
    shutil.copy(__file__, o / "generator_script.py")
    print(f"\nWINS: {wins if wins else 'NINGUNA — familia cerrada'}")
    print(f"artefacto -> {o}")
    return verdict


if __name__ == "__main__":
    screen()
