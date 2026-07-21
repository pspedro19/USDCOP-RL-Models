"""H-RISK-FAM-01 — screening de la familia de riesgo pre-registrada (ABRE resultados: +5 trials).

Contract: CTR-QUANT-CONSTITUTION-001 · Pre-registro: HYPOTHESIS-REGISTRY.md
"H-RISK-FAM-01 (PRE-REGISTRO 2026-07-21)" — familia CERRADA de 5 celdas, protocolo
sellado ANTES de este script. Este archivo ES el generador persistido (deuda del panel).

Protocolo (del pre-registro, sin desviaciones):
- Universo: SOLO <= 2024-12-31, semanas lunes-viernes del seed COP reparado.
- Targets: gap_week = 1{|gap de apertura lunes| > p90 rolling-252d} · q90 del rango
  semanal (max high - min low, en % del cierre previo).
- Por celda: modelo UNIVARIADO (logistic -> Brier; QuantileRegressor q90 -> pinball),
  purged K-Fold K=5 con purga 1 semana + embargo = lookback de la celda.
- Bar: batir al BASELINE DE PERSISTENCIA (Brier: frecuencia base rolling; pinball:
  q90 = 1.645 * EWMA-vol * sqrt(5), lambda 0.94). Win = IC95 block-bootstrap (b=4) de
  la DIFERENCIA de perdida excluye 0 a favor de la celda.
- Se publica la tabla ENTERA. Consecuencia pre-firmada: >=1 win -> 1 trial economico
  (sizing P(gap), candidata v13, juez forward). 0 wins -> familia CERRADA.

Aproximaciones DECLARADAS (debilitan, no fabrican):
- F1 calendario: FOMC = fechas historicas publicas hardcodeadas; BanRep = ultimo viernes
  habil del mes (aprox de su calendario de decision); IPC DANE = dia 5 +/- (SSOT
  typical_day). Un dummy mal fechado REDUCE el poder de F1 — sesgo contra nosotros.
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

from services.common.metrics import circular_block_bootstrap, ewma_volatility  # noqa: E402

RNG = np.random.default_rng(42)
END_DESIGN = pd.Timestamp("2024-12-31")

# FOMC decision dates 2020-2024 (public record)
FOMC = pd.to_datetime([
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10","2020-07-29",
    "2020-09-16","2020-11-05","2020-12-16","2021-01-27","2021-03-17","2021-04-28",
    "2021-06-16","2021-07-28","2021-09-22","2021-11-03","2021-12-15","2022-01-26",
    "2022-03-16","2022-05-04","2022-06-15","2022-07-27","2022-09-21","2022-11-02",
    "2022-12-14","2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13","2024-01-31","2024-03-20","2024-05-01",
    "2024-06-12","2024-07-31","2024-09-18","2024-11-07","2024-12-18"])


def load_weekly():
    d = pd.read_parquet(REPO / "seeds/latest/usdcop_daily_ohlcv.parquet")
    d["time"] = pd.to_datetime(d["time"]).dt.tz_localize(None)
    d = d[d["time"] <= END_DESIGN].sort_values("time").reset_index(drop=True)
    d["ret"] = np.log(d["close"] / d["close"].shift(1))
    d["gap"] = (d["open"] / d["close"].shift(1) - 1).abs()
    d["dow"] = d["time"].dt.dayofweek
    d["week"] = d["time"].dt.to_period("W-SUN")

    macro = pd.read_parquet(REPO / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet")
    macro.index = pd.to_datetime(macro.index)
    cols = {c.upper(): c for c in macro.columns}
    vix = macro[cols["VOLT_VIX_USA_D_VIX"]].rename("vix")
    embi = macro[cols["CRSK_SPREAD_EMBI_COL_D_EMBI"]].rename("embi")

    monthly = pd.read_parquet(REPO / "data/pipeline/04_cleaning/output/MACRO_MONTHLY_CLEAN.parquet")
    monthly.index = pd.to_datetime(monthly.index)
    resint = monthly[[c for c in monthly.columns if "RESINT" in c.upper()][0]].rename("resint")

    rows = []
    vol20 = d["ret"].rolling(20).std() * np.sqrt(252)
    d["vol20"] = vol20
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
        # features t-1 (todas conocibles el viernes previo)
        vv = prev["vol20"].tail(20).diff().std() * np.sqrt(252)          # F2 vol-of-vol
        vix_t1 = float(vix.loc[:asof].iloc[-1]) if len(vix.loc[:asof]) else np.nan
        embi_ser = embi.loc[:asof].dropna()
        embi_acc = float((embi_ser.diff(5).iloc[-1] - embi_ser.diff(5).tail(252).mean())
                         / (embi_ser.diff(5).tail(252).std() + 1e-9)) if len(embi_ser) > 260 else np.nan
        res_ser = resint.loc[:asof - pd.Timedelta(days=45)].dropna()     # lag publicacion
        res_z = float((res_ser.iloc[-1] - res_ser.tail(12).mean())
                      / (res_ser.tail(12).std() + 1e-9)) if len(res_ser) > 12 else np.nan
        # F3: prob condicional de gap-cola dado nivel vix (kernel binario simple en terciles)
        prev_mon = prev[prev["dow"] == 0].tail(252)
        if len(prev_mon) > 30 and not np.isnan(vix_t1):
            vix_al = vix.reindex(prev_mon["time"], method="ffill")
            ter = pd.qcut(vix_al, 3, labels=False, duplicates="drop")
            my_ter = 1
            if len(pd.unique(ter.dropna())) == 3:
                _cut = pd.cut([vix_t1], bins=pd.qcut(vix_al, 3, retbins=True,
                              duplicates="drop")[1], labels=False, include_lowest=True)[0]
                # vix_t1 fuera del rango historico -> clip al tercil extremo
                if pd.isna(_cut):
                    _cut = 0 if vix_t1 < float(vix_al.min()) else 2
                my_ter = int(_cut)
            mask = (ter == my_ter).to_numpy()
            thr = prev_mon["gap"].quantile(0.90)
            f3 = float((prev_mon["gap"].to_numpy()[mask] > thr).mean()) if mask.sum() > 5 else np.nan
        else:
            f3 = np.nan
        # F1 calendario: evento en la semana operada (conocible ex-ante)
        wk_start, wk_end = g["time"].iloc[0], g["time"].iloc[-1]
        fomc_wk = int(((FOMC >= wk_start) & (FOMC <= wk_end)).any())
        last_bday = pd.date_range(wk_start.replace(day=1),
                                  wk_start + pd.offsets.MonthEnd(0), freq="B")[-1]
        banrep_wk = int(wk_start <= last_bday <= wk_end)
        ipc_wk = int(any((t.day in (4, 5, 6, 7, 8)) for t in g["time"]))
        f1 = int(fomc_wk or banrep_wk or ipc_wk)
        # EWMA vol para el baseline de pinball
        _ew = ewma_volatility(prev["ret"].dropna().to_numpy()[-260:],
                              lam=0.94, periods_per_year=252)
        sig_ewma = float(np.asarray(_ew).ravel()[-1])   # la serie EWMA: tomar el ultimo valor
        rows.append({"monday": monday["time"], "gap_bps": gap_bps,
                     "gap_week": int(gap_bps > p90_gap), "range_pct": rng_pct,
                     "f1_event": f1, "f2_volofvol": vv, "f3_gaptail": f3,
                     "f4_embiacc": embi_acc, "f5_resintz": res_z,
                     "base_rate": float(prev.loc[prev["dow"] == 0, "gap"].tail(252)
                                        .gt(prev.loc[prev["dow"] == 0, "gap"].tail(252)
                                            .quantile(0.90)).mean()),
                     "sigma_ewma": sig_ewma})
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
    cells = ["f1_event", "f2_volofvol", "f3_gaptail", "f4_embiacc", "f5_resintz"]
    results = {}
    for cell in cells:
        d = W.dropna(subset=[cell, "gap_week", "range_pct", "sigma_ewma"]).reset_index(drop=True)
        n = len(d)
        x = d[cell].to_numpy().reshape(-1, 1)
        # --- Brier: modelo univariado vs frecuencia base rolling ---
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
        # --- pinball q90 del rango vs baseline persistencia EWMA ---
        q_model = np.full(n, np.nan)
        for tr, te in purged_kfold_idx(n):
            qr = QuantileRegressor(quantile=0.90, alpha=1e-4, solver="highs").fit(
                x[tr], d.range_pct.iloc[tr])
            q_model[te] = qr.predict(x[te])
        q_base = 1.645 * d.sigma_ewma / np.sqrt(52) * np.sqrt(5) / np.sqrt(5) * 100 * np.sqrt(5) / np.sqrt(252/5)
        # baseline correcto: q90 semanal = 1.645 * sigma_ewma_anual * sqrt(5/252) * 100
        q_base = 1.645 * d.sigma_ewma * np.sqrt(5 / 252) * 100
        def pinball(y, q, tau=0.90):
            e = y - q
            return np.maximum(tau * e, (tau - 1) * e)
        pin_m = pinball(d.range_pct.to_numpy(), q_model)
        pin_b = pinball(d.range_pct.to_numpy(), q_base.to_numpy())
        # NULL JUSTO (correccion contra nosotros mismos, 2026-07-21): el baseline
        # 1.645*sigma*sqrt(5) es el q90 de un RETORNO, no del RANGO -> subestima y
        # cualquier modelo gana por el intercepto. El null que aisla la FEATURE es
        # intercepto-solo en el MISMO purged CV.
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
            "pinball_model": float(np.nanmean(pin_m)), "pinball_base_ewma": float(np.nanmean(pin_b)),
            "pinball_null_intercepto": float(np.nanmean(pin_null)),
            "delta_pinball_vs_null_ci95": dp["ci95"], "WIN_pinball": bool(win_pin),
        }
        print(f"{cell}: n={n} | Brier {np.nanmean(brier_m):.4f} vs base "
              f"{np.nanmean(brier_b):.4f} (WIN={win_brier}) | pinball "
              f"{np.nanmean(pin_m):.3f} vs NULL-intercepto {np.nanmean(pin_null):.3f} "
              f"(ewma mal-especificado: {np.nanmean(pin_b):.3f}) (WIN={win_pin})")
    wins = [c for c, r in results.items() if r["WIN_brier"] or r["WIN_pinball"]]
    verdict = {"family": "H-RISK-FAM-01", "cells": results, "wins": wins,
               "trials_spent": 5, "consequence": (
                   "1 trial economico adicional (sizing P(gap), v13, juez forward)"
                   if wins else "familia CERRADA por escrito")}
    o = REPO / ".claude/evidence/cop_risk_family" / date.today().isoformat()
    o.mkdir(parents=True, exist_ok=True)
    (o / "screen_results.json").write_text(json.dumps(verdict, indent=2, default=str))
    shutil.copy(__file__, o / "generator_script.py")
    print(f"\nWINS: {wins if wins else 'NINGUNA — familia cerrada'}")
    print(f"artefacto -> {o}")
    return verdict


if __name__ == "__main__":
    screen()
