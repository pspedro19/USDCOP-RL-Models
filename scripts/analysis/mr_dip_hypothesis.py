#!/usr/bin/env python3
"""H-BTC-MR-01 / H-XAU-MR-01 — mean-reversion de dips (pre-registrada 2026-07-07).

PRIORS CONGELADOS (ver `.claude/specs/assets/_strategy-history.md §6`, escritos ANTES de correr):
  trigger  = ret_3d < -Z * sigma3   con sigma3 = rvol20_diaria * sqrt(3), evaluado a t-1 (causal)
  filtro   = close > SMA200 a t-1 (no comprar cuchillos)
  hold     = H dias (re-arma con nueva señal)
  prior    = (Z=1.5, H=5) · sensibilidades one-at-a-time Z∈{1.0,2.0}, H∈{3,10}
Variantes: (a) standalone (informativa: WR/PF) · (b) combinada con la campeona del activo.
Bar de adopción (ex-ante): combinada > campeona en Calmar Y Sharpe en diseño ≤2024,
confirma OOS-2025, sobrevive costos ×2, DSR trial-aware > 0.95. TODAS las celdas se reportan.

Usage: python scripts/analysis/mr_dip_hypothesis.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from services.common.metrics import deflated_sharpe_ratio  # noqa: E402

# Trials acumulados del sistema tras esta ronda (registro §5: ~30 previos + 7 ronda pasada + 12 aquí)
TRIALS_TOTAL = 49

CELLS = [("prior", 1.5, 5), ("z1.0", 1.0, 5), ("z2.0", 2.0, 5), ("h3", 1.5, 3), ("h10", 1.5, 10)]


def dip_window(feat: pd.DataFrame, z: float, hold: int, ann_days: int) -> pd.Series:
    """Ventana MR causal: señal calculada con info hasta t-1, activa H días."""
    close = feat["close"]
    ret3 = close.pct_change(3)
    sigma3 = (feat["realized_vol_20"] / np.sqrt(ann_days)) * np.sqrt(3)  # vol 3d en unidades diarias
    trigger = (ret3 < -z * sigma3) & (close > close.rolling(200).mean())
    trig_lag = trigger.shift(1).fillna(False)  # causal: la decisión de hoy usa el cierre de ayer
    active = trig_lag.rolling(hold, min_periods=1).max().fillna(0.0)
    return active.astype(float)


def seg_trades(d: pd.DataFrame) -> tuple[int, float, float | None]:
    """(n_trades, win_rate_pct, profit_factor) segmentando posición≠0."""
    pos, ret = d["position"].values, d["strat_ret"].values
    pnls, i, n = [], 0, len(d)
    while i < n:
        if abs(pos[i]) < 1e-9:
            i += 1
            continue
        j, g = i, 1.0
        while j < n and abs(pos[j]) > 1e-9:
            g *= 1.0 + ret[j]
            j += 1
        pnls.append(g - 1.0)
        i = j
    if not pnls:
        return 0, 0.0, None
    wins = [p for p in pnls if p > 0]
    gl = abs(sum(p for p in pnls if p < 0))
    pf = round(sum(wins) / gl, 2) if gl > 0 else None
    return len(pnls), round(100 * len(wins) / len(pnls), 1), pf


def run_asset(asset: str) -> None:
    if asset == "btcusdt":
        from src.btc_strategy import backtest as bt
        from src.btc_strategy.indicators import build_daily_features, classify_regime
        from src.btc_strategy.strategies import build_positions, intent_trend
        seed, ann, champ_fn, champ = "btcusdt_daily_ohlcv.parquet", 365, intent_trend, "btc_trend_b2"
        design = ("2018-01-01", "2024-12-31")
    else:
        from src.gold_rl import backtest as bt
        from src.gold_rl.indicators import build_daily_features, classify_regime
        from src.gold_rl.strategies import build_positions, direction_long_only
        seed, ann, champ_fn, champ = "xauusd_daily_ohlcv.parquet", 252, direction_long_only, "gold_long_only_b1"
        design = ("2004-01-01", "2024-12-31")

    df = pd.read_parquet(REPO / "seeds" / "latest" / seed).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = classify_regime(build_daily_features(df))

    def full_and_slices(pos_df: pd.DataFrame) -> dict:
        d = bt.compute_returns(pos_df)
        out = {}
        for label, lo, hi in [("design", design[0], design[1]), ("oos2025", "2025-01-01", "2025-12-31"),
                              ("full", None, None)]:
            s = d if lo is None else d[(d["time"] >= lo) & (d["time"] <= hi)]
            if len(s) < 30:
                continue
            s = s.copy()
            s["equity"] = (1.0 + s["strat_ret"]).cumprod()
            m = bt.metrics(s)
            n_tr, wr, pf = seg_trades(s)
            out[label] = {**m, "trades": n_tr, "wr": wr, "pf": pf,
                          "sr_daily": s["strat_ret"].values}
        return out

    def cost_x2(pos_df: pd.DataFrame, lo: str, hi: str) -> float:
        d = bt.compute_returns(pos_df)
        s = d[(d["time"] >= lo) & (d["time"] <= hi)].copy()
        s["strat_ret"] = s["position"] * s["ret"] - 2 * s["cost"] - 2 * s["swap"]
        s["equity"] = (1.0 + s["strat_ret"]).cumprod()
        return bt.metrics(s)["calmar"]

    champ_pos = build_positions(feat, champ_fn)
    champ_res = full_and_slices(champ_pos)
    print(f"\n{'='*100}\n{asset.upper()} — campeona {champ}")
    hdr = f"{'celda':<22}{'ventana':<9}{'ret%':>8}{'Sharpe':>8}{'Calmar':>8}{'MaxDD%':>8}{'trades':>7}{'WR%':>7}{'PF':>7}"
    print(hdr)
    for w, r in champ_res.items():
        print(f"{champ:<22}{w:<9}{r['total_return_pct']:>8.1f}{r['sharpe']:>8.3f}{r['calmar']:>8.3f}"
              f"{r['max_dd']:>8.1f}{r['trades']:>7}{r['wr']:>7.1f}{str(r['pf']):>7}")

    for name, z, hold in CELLS:
        win = dip_window(feat, z, hold, ann)
        # (a) standalone
        sa = build_positions(feat, lambda f, w=win: w)
        # (b) combinada / overlay
        if asset == "btcusdt":
            comb_fn = lambda f, w=win: np.maximum(champ_fn(f), w)  # noqa: E731
        else:
            comb_fn = lambda f, w=win: (1.0 + 0.5 * w).clip(upper=1.5)  # noqa: E731
        cb = build_positions(feat, comb_fn)
        if asset == "xauusd":
            cb["position"] = cb["position"].clip(upper=1.5)
        for tag, res in [("MR-standalone", full_and_slices(sa)), ("MR-combinada", full_and_slices(cb))]:
            for w, r in res.items():
                print(f"{name+' '+tag:<22}{w:<9}{r['total_return_pct']:>8.1f}{r['sharpe']:>8.3f}"
                      f"{r['calmar']:>8.3f}{r['max_dd']:>8.1f}{r['trades']:>7}{r['wr']:>7.1f}{str(r['pf']):>7}")
            if tag == "MR-combinada" and name == "prior":
                dsg, ch_dsg = res.get("design"), champ_res.get("design")
                beats = (dsg and ch_dsg and dsg["calmar"] > ch_dsg["calmar"]
                         and dsg["sharpe"] > ch_dsg["sharpe"])
                cx2 = cost_x2(cb, design[0], design[1])
                from scipy import stats as sps
                sr = dsg["sr_daily"]
                sr_daily = float(np.mean(sr) / np.std(sr)) if np.std(sr) > 0 else 0.0
                dsr = deflated_sharpe_ratio(sr_daily, len(sr), TRIALS_TOTAL, 0.5 / np.sqrt(ann),
                                            skew=float(sps.skew(sr)),
                                            kurtosis=float(sps.kurtosis(sr, fisher=False)))
                dsr_v = round(dsr.get("dsr", 0.0), 4) if isinstance(dsr, dict) else dsr
                print(f"  → GATE prior combinada: bate campeona en diseño (Calmar Y Sharpe)? "
                      f"{'SÍ' if beats else 'NO'} · Calmar costos×2={cx2:.3f} · DSR({TRIALS_TOTAL} trials)={dsr_v}")


if __name__ == "__main__":
    for a in ("btcusdt", "xauusd"):
        run_asset(a)
