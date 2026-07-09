#!/usr/bin/env python3
"""H-WKTPHS-HORIZON-01 — mecánica TP/HS con horizonte flexible N∈{1,2,3,4,8} semanas (§8.6).

Selección honesta: la celda ganadora se elige por Calmar en DISEÑO ≤2024; el OOS-2025 solo
confirma. TP=1.0×ATR_N / HS=2.0×ATR_N fijos (priors §8.3, no se re-abren). Todas las celdas
se reportan y cuentan como trials. Usage: python scripts/analysis/wktphs_horizon_grid.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

HORIZONS = [1, 2, 3, 4, 8]  # semanas ISO por bloque
TP_MULT, HS_MULT = 1.0, 2.0


def simulate(feat: pd.DataFrame, n_weeks: int, *, week_days: int, cost_side: float,
             swap_daily: float, size_expr, cost_mult: float = 1.0):
    d = feat.reset_index(drop=True).copy()
    d["iso_week"] = d["time"].dt.strftime("%G-W%V")
    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    d["signal_raw"] = (votes >= 2).astype(float)
    d["size_raw"] = size_expr(d)

    weeks = list(d.groupby("iso_week", sort=False))
    blocks = [weeks[i:i + n_weeks] for i in range(1, len(weeks), n_weeks)]
    strat, pos = np.zeros(len(d)), np.zeros(len(d))
    trades = []
    cost = cost_mult * cost_side
    swap = cost_mult * swap_daily
    prev_week = weeks[0][1]
    for blk in blocks:
        days = pd.concat([w[1] for w in blk])
        sig = prev_week.iloc[-1]
        prev_week = blk[-1][1]
        if sig["signal_raw"] < 1 or len(days) == 0:
            continue
        size = float(sig["size_raw"])
        atr_n = float(sig["atr_14"]) * np.sqrt(week_days * n_weeks)
        if not np.isfinite(atr_n) or atr_n <= 0 or not np.isfinite(size) or size <= 0:
            continue
        entry = float(days.iloc[0]["open"]) if "open" in days else float(days.iloc[0]["close"])
        tp_px, hs_px = entry + TP_MULT * atr_n, entry - HS_MULT * atr_n
        prev_px, exited, reason, exit_px = entry, False, "block_end", float(days.iloc[-1]["close"])
        for k, (_, row) in enumerate(days.iterrows()):
            if exited:
                break
            de = None
            if float(row["low"]) <= hs_px:
                de, reason = hs_px, "hard_stop"
            elif float(row["high"]) >= tp_px:
                de, reason = tp_px, "take_profit"
            px = de if de is not None else float(row["close"])
            r = size * (px / prev_px - 1.0) - swap * size
            if k == 0:
                r -= cost * size
            if de is not None or k == len(days) - 1:
                r -= cost * size
            strat[row.name] += r
            pos[row.name] = size
            prev_px = px
            if de is not None:
                exited, exit_px = True, de
        trades.append({"week": blk[0][0], "pnl_pct": size * (exit_px / entry - 1.0) * 100,
                       "reason": reason})
    d["strat_ret"] = strat
    d["position"] = pos
    d["ret"] = d["close"].pct_change().fillna(0.0)
    d.attrs["trades"] = trades
    return d


def run(asset: str) -> None:
    if asset == "btcusdt":
        from src.btc_strategy import backtest as bt
        from src.btc_strategy.indicators import build_daily_features
        seed, week_days, ann = "btcusdt_daily_ohlcv.parquet", 7, 365
        cost_side, swap_daily = 13.0 / 1e4, 0.0
        size_expr = lambda d: (0.30 / d["realized_vol_20"].clip(lower=0.30)).clip(upper=1.0)  # noqa: E731
        champ = "btc_trend_b2 (bar diseño: Sharpe 1.581 / Calmar 2.839)"
        design = ("2018-01-01", "2024-12-31")
    else:
        from src.gold_rl import backtest as bt
        from src.gold_rl.indicators import build_daily_features
        seed, week_days, ann = "xauusd_daily_ohlcv.parquet", 5, 252
        cost_side, swap_daily = 2.0 / 1e4, 0.025 / 252
        size_expr = lambda d: (0.10 / d["realized_vol_20"].clip(lower=0.06)).clip(upper=1.5)  # noqa: E731
        champ = "gold_long_only_b1 (bar diseño: Calmar 0.085)"
        design = ("2004-01-01", "2024-12-31")

    from services.common.metrics import paired_exposure_baseline

    df = pd.read_parquet(REPO / "seeds/latest" / seed).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    print(f"\n{'='*118}\n{asset.upper()} — {champ}")
    print(f"{'N sem':<7}{'ventana':<12}{'ret%':>8}{'Sharpe':>8}{'Calmar':>8}{'MaxDD%':>8}"
          f"{'p-val':>7}{'trades':>7}{'WR%':>7}{'PF':>7}{'B1pCal':>8}{'x2Cal':>7}  salidas")
    for n in HORIZONS:
        d = simulate(feat, n, week_days=week_days, cost_side=cost_side,
                     swap_daily=swap_daily, size_expr=size_expr)
        for wlab, lo, hi in [("diseño≤24", design[0], design[1]),
                             ("OOS2025", "2025-01-01", "2025-12-31")]:
            s = d[(d["time"] >= lo) & (d["time"] <= hi)].copy()
            s["equity"] = (1.0 + s["strat_ret"]).cumprod()
            m = bt.metrics(s)
            boot = bt.block_bootstrap_pvalue(s["strat_ret"])
            tw = [t for t in d.attrs["trades"] if lo[:4] + "-W01" <= t["week"] <= hi[:4] + "-W53"]
            wins = [t for t in tw if t["pnl_pct"] > 0]
            gl = abs(sum(t["pnl_pct"] for t in tw if t["pnl_pct"] < 0))
            pf = round(sum(t["pnl_pct"] for t in wins) / gl, 2) if gl > 0 else None
            wr = round(100 * len(wins) / len(tw), 1) if tw else 0.0
            b1p = paired_exposure_baseline(s["position"].values, s["ret"].values, ann)
            d2 = simulate(feat, n, week_days=week_days, cost_side=cost_side,
                          swap_daily=swap_daily, size_expr=size_expr, cost_mult=2.0)
            s2 = d2[(d2["time"] >= lo) & (d2["time"] <= hi)].copy()
            s2["equity"] = (1.0 + s2["strat_ret"]).cumprod()
            reasons: dict[str, int] = {}
            for t in tw:
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            print(f"N={n:<5}{wlab:<12}{m['total_return_pct']:>8.1f}{m['sharpe']:>8.3f}"
                  f"{m['calmar']:>8.3f}{m['max_dd']:>8.1f}{boot['p_value']:>7.3f}{len(tw):>7}"
                  f"{wr:>7.1f}{str(pf):>7}{b1p['calmar']:>8.3f}{bt.metrics(s2)['calmar']:>7.3f}  {reasons}")


if __name__ == "__main__":
    for a in ("btcusdt", "xauusd"):
        run(a)
