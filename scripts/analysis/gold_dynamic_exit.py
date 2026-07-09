#!/usr/bin/env python3
"""H-XAU-DYNEXIT-01 — Oro con salidas DINÁMICAS decididas por la estrategia (pre-reg §8.7).

Sin calendario: entra cuando la señal enciende (votos SMA{63,126,252} ≥ 2/3), sale cuando
(a) la señal muere o (b) toca el Chandelier trailing (max close desde entrada − M×ATR14,
prior M=3.0, sens {2.0, 4.0}). Hold variable: días→meses, decidido por el precio.
Selección en diseño ≤2024; OOS-2025 confirma. Todas las celdas se reportan (+3 trials).

Usage: python scripts/analysis/gold_dynamic_exit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.gold_rl import backtest as bt  # noqa: E402
from src.gold_rl.indicators import build_daily_features  # noqa: E402

COST = 2.0 / 1e4
SWAP_D = 0.025 / 252
CELLS = [("prior M=3.0", 3.0), ("sens M=2.0", 2.0), ("sens M=4.0", 4.0)]


def simulate(feat: pd.DataFrame, trail_mult: float, cost_mult: float = 1.0) -> pd.DataFrame:
    d = feat.reset_index(drop=True).copy()
    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    d["sig"] = (votes >= 2).astype(int)
    d["size_raw"] = (0.10 / d["realized_vol_20"].clip(lower=0.06)).clip(upper=1.5)

    n = len(d)
    strat, pos = np.zeros(n), np.zeros(n)
    trades = []
    cost = cost_mult * COST
    swap = cost_mult * SWAP_D
    in_trade, entry_i, size, hi_close, trail_px, prev_px = False, -1, 0.0, 0.0, 0.0, 0.0
    open_col = "open" if "open" in d else "close"
    for i in range(1, n):
        row = d.iloc[i]
        sig_y = int(d["sig"].iloc[i - 1])            # señal al cierre de AYER (causal)
        if not in_trade:
            if sig_y == 1 and np.isfinite(d["size_raw"].iloc[i - 1]) and np.isfinite(d["atr_14"].iloc[i - 1]):
                in_trade, entry_i = True, i
                size = float(d["size_raw"].iloc[i - 1])
                entry_px = float(row[open_col])
                hi_close = entry_px
                trail_px = entry_px - trail_mult * float(d["atr_14"].iloc[i - 1])
                prev_px = entry_px
                r = size * (float(row["close"]) / prev_px - 1.0) - swap * size - cost * size
                strat[i] += r
                pos[i] = size
                hi_close = max(hi_close, float(row["close"]))
                trail_px = max(trail_px, hi_close - trail_mult * float(row["atr_14"]))
                prev_px = float(row["close"])
            continue
        # en trade: primero chequear salidas
        exited, reason, exit_px = False, "", 0.0
        if sig_y == 0:                                # señal murió ayer → sale hoy al open
            exited, reason, exit_px = True, "signal_off", float(row[open_col])
        elif float(row["low"]) <= trail_px:           # trailing intra-día
            exited, reason, exit_px = True, "trailing_stop", trail_px
        px_end = exit_px if exited else float(row["close"])
        r = size * (px_end / prev_px - 1.0) - swap * size
        if exited:
            r -= cost * size
        strat[i] += r
        pos[i] = size
        prev_px = px_end
        if exited:
            entry_px_t = float(d.iloc[entry_i][open_col])
            trades.append({"entry_time": d["time"].iloc[entry_i], "exit_time": row["time"],
                           "days": i - entry_i, "entry_px": entry_px_t, "exit_px": px_end,
                           "size": size,
                           "pnl_pct": size * (px_end / entry_px_t - 1.0) * 100,
                           "reason": reason})
            in_trade = False
        else:
            hi_close = max(hi_close, float(row["close"]))
            trail_px = max(trail_px, hi_close - trail_mult * float(row["atr_14"]))
    d["strat_ret"] = strat
    d["position"] = pos
    d["ret"] = d["close"].pct_change().fillna(0.0)
    d.attrs["trades"] = trades
    return d


def main() -> None:
    from services.common.metrics import paired_exposure_baseline

    df = pd.read_parquet(REPO / "seeds/latest/xauusd_daily_ohlcv.parquet").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    print(f"{'celda':<14}{'ventana':<11}{'ret%':>8}{'Sharpe':>8}{'Calmar':>8}{'MaxDD%':>8}{'p-val':>7}"
          f"{'trades':>7}{'WR%':>7}{'PF':>7}{'B1pCal':>8}{'x2Cal':>7}{'hold d (med/max)':>18}  salidas")
    for name, m_ in CELLS:
        d = simulate(feat, m_)
        for wlab, lo, hi in [("diseño≤24", "2004-01-01", "2024-12-31"),
                             ("OOS2025", "2025-01-01", "2025-12-31")]:
            s = d[(d["time"] >= lo) & (d["time"] <= hi)].copy()
            s["equity"] = (1.0 + s["strat_ret"]).cumprod()
            met = bt.metrics(s)
            boot = bt.block_bootstrap_pvalue(s["strat_ret"])
            tw = [t for t in d.attrs["trades"]
                  if str(t["exit_time"])[:4] >= lo[:4] and str(t["entry_time"])[:4] <= hi[:4]
                  and lo <= str(t["exit_time"])[:10] <= hi]
            wins = [t for t in tw if t["pnl_pct"] > 0]
            gl = abs(sum(t["pnl_pct"] for t in tw if t["pnl_pct"] < 0))
            pf = round(sum(t["pnl_pct"] for t in wins) / gl, 2) if gl > 0 else None
            wr = round(100 * len(wins) / len(tw), 1) if tw else 0.0
            durs = sorted(t["days"] for t in tw)
            dur = f"{durs[len(durs)//2]}/{durs[-1]}" if durs else "—"
            b1p = paired_exposure_baseline(s["position"].values, s["ret"].values, 252)
            d2 = simulate(feat, m_, cost_mult=2.0)
            s2 = d2[(d2["time"] >= lo) & (d2["time"] <= hi)].copy()
            s2["equity"] = (1.0 + s2["strat_ret"]).cumprod()
            reasons: dict[str, int] = {}
            for t in tw:
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            print(f"{name:<14}{wlab:<11}{met['total_return_pct']:>8.1f}{met['sharpe']:>8.3f}"
                  f"{met['calmar']:>8.3f}{met['max_dd']:>8.1f}{boot['p_value']:>7.3f}{len(tw):>7}"
                  f"{wr:>7.1f}{str(pf):>7}{b1p['calmar']:>8.3f}{bt.metrics(s2)['calmar']:>7.3f}"
                  f"{dur:>18}  {reasons}")


if __name__ == "__main__":
    main()
