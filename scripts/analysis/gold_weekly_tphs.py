#!/usr/bin/env python3
"""H-XAU-WKTPHS-01 — Oro ACTIVO con mecánica semanal COP-v11 (pre-registrada §8.3, ex-ante).

Trasplanta la MECÁNICA de smart_simple_v11 (cadencia semanal + TP/HS + re-entrada) a Oro.
Priors congelados ANTES de correr: señal = votos SMA{63,126,252} ≥ 2/3 al cierre del viernes;
entrada lunes open; TP=+1.0×ATR_w, HS=−2.0×ATR_w (ATR_w = ATR14d×√5, HS 2.0 = piso v11);
si ambos tocan el mismo día gana el HS (conservador); si nada toca ⇒ cierre viernes (week_end).
Sizing vol-target 10%/cap 1.5. Costos motor Oro: 2 bps/lado + swap 2.5%/año sobre exposición.
Sensibilidades pre-registradas TP∈{0.75, 1.5} — se reportan TODAS las celdas.

Usage: python scripts/analysis/gold_weekly_tphs.py
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

COST = 2.0 / 1e4          # por lado, sobre notional expuesto
SWAP_D = 0.025 / 252      # drag diario sobre exposición
TP_CELLS = [("prior TP=1.0", 1.0), ("sens TP=0.75", 0.75), ("sens TP=1.5", 1.5)]
HS_MULT = 2.0


def simulate(feat: pd.DataFrame, tp_mult: float) -> pd.DataFrame:
    """Devuelve un DataFrame diario con strat_ret + registro de trades."""
    d = feat.reset_index(drop=True).copy()
    d["iso_week"] = d["time"].dt.strftime("%G-W%V")
    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    d["signal_raw"] = (votes >= 2).astype(float)
    d["size_raw"] = (0.10 / d["realized_vol_20"].clip(lower=0.06)).clip(upper=1.5)

    strat_ret = np.zeros(len(d))
    trades = []
    weeks = list(d.groupby("iso_week", sort=False))
    for wi in range(1, len(weeks)):
        prev_days = weeks[wi - 1][1]
        days = weeks[wi][1]
        sig_row = prev_days.iloc[-1]                     # cierre del viernes anterior (causal)
        if sig_row["signal_raw"] < 1 or len(days) == 0:
            continue
        size = float(sig_row["size_raw"])
        atr_w = float(sig_row["atr_14"]) * np.sqrt(5)
        if not np.isfinite(atr_w) or atr_w <= 0 or not np.isfinite(size):
            continue
        entry = float(days.iloc[0]["open"]) if "open" in days else float(days.iloc[0]["close"])
        tp_px = entry + tp_mult * atr_w
        hs_px = entry - HS_MULT * atr_w
        prev_close, exited, reason, exit_px = entry, False, "week_end", float(days.iloc[-1]["close"])
        for k, (_, row) in enumerate(days.iterrows()):
            if exited:
                break
            lo, hi, cl = float(row["low"]), float(row["high"]), float(row["close"])
            day_exit = None
            if lo <= hs_px:
                day_exit, reason = hs_px, "hard_stop"       # conservador: HS primero
            elif hi >= tp_px:
                day_exit, reason = tp_px, "take_profit"
            px_end = day_exit if day_exit is not None else cl
            r = size * (px_end / prev_close - 1.0) - SWAP_D * size
            if k == 0:
                r -= COST * size                             # entrada
            if day_exit is not None or k == len(days) - 1:
                r -= COST * size                             # salida
            strat_ret[row.name] += r
            prev_close = px_end
            if day_exit is not None:
                exited, exit_px = True, day_exit
        pnl = size * (exit_px / entry - 1.0)
        trades.append({"week": weeks[wi][0], "entry": entry, "exit": exit_px,
                       "pnl_pct": pnl * 100, "reason": reason, "size": size})
    d["strat_ret"] = strat_ret
    d["ret"] = d["close"].pct_change().fillna(0.0)
    d["position"] = 0.0  # no usado por metrics
    d["equity"] = (1.0 + d["strat_ret"]).cumprod()
    d.attrs["trades"] = trades
    return d


def window(d: pd.DataFrame, lo: str | None, hi: str | None) -> pd.DataFrame:
    s = d if lo is None else d[(d["time"] >= lo) & (d["time"] <= hi)]
    s = s.copy()
    s["equity"] = (1.0 + s["strat_ret"]).cumprod()
    return s


def trade_stats(trades: list[dict], lo: str, hi: str) -> tuple[int, float, float | None, dict]:
    tw = [t for t in trades if lo <= t["week"] <= hi]
    if not tw:
        return 0, 0.0, None, {}
    wins = [t for t in tw if t["pnl_pct"] > 0]
    gl = abs(sum(t["pnl_pct"] for t in tw if t["pnl_pct"] < 0))
    pf = round(sum(t["pnl_pct"] for t in wins) / gl, 2) if gl > 0 else None
    reasons: dict[str, int] = {}
    for t in tw:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
    return len(tw), round(100 * len(wins) / len(tw), 1), pf, reasons


def main() -> None:
    df = pd.read_parquet(REPO / "seeds/latest/xauusd_daily_ohlcv.parquet").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    print(f"{'celda':<16}{'ventana':<10}{'ret%':>8}{'Sharpe':>8}{'Calmar':>8}{'MaxDD%':>8}"
          f"{'p-val':>7}{'trades':>7}{'WR%':>7}{'PF':>7}  salidas")
    for name, tp in TP_CELLS:
        d = simulate(feat, tp)
        for wlab, lo, hi in [("diseño≤24", "2004-01-01", "2024-12-31"),
                             ("OOS2025", "2025-01-01", "2025-12-31")]:
            s = window(d, lo, hi)
            m = bt.metrics(s)
            boot = bt.block_bootstrap_pvalue(s["strat_ret"])
            n, wr, pf, reasons = trade_stats(d.attrs["trades"], lo[:4] + "-W01", hi[:4] + "-W53")
            print(f"{name:<16}{wlab:<10}{m['total_return_pct']:>8.1f}{m['sharpe']:>8.3f}"
                  f"{m['calmar']:>8.3f}{m['max_dd']:>8.1f}{boot['p_value']:>7.3f}{n:>7}"
                  f"{wr:>7.1f}{str(pf):>7}  {reasons}")


if __name__ == "__main__":
    main()
