#!/usr/bin/env python3
"""H-BTC-WKTPHS-01 — trasplante de la mecánica semanal COP-v11 a BTC (pre-reg §8.5, ex-ante).

Priors idénticos a Oro (§8.3), cero fit sobre BTC: señal viernes = votos SMA{63,126,252} ≥ 2/3;
entrada lunes open; TP=+1.0×ATR_w / HS=−2.0×ATR_w (ATR_w = ATR14d×√5... BTC es 24/7 ⇒ √7 días
calendario por semana ISO); capa de riesgo BTC: vol-target 30%, floor 30%, spot-only (exposición
≤1.0); costos motor BTC (13 bps/lado, sin swap). Sens TP∈{0.75, 1.5}. Todas las celdas se reportan.

Usage: python scripts/analysis/btc_weekly_tphs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.btc_strategy import backtest as bt  # noqa: E402
from src.btc_strategy.indicators import build_daily_features  # noqa: E402

COST = 13.0 / 1e4         # por lado (taker+spread+slippage, motor BTC)
WEEK_DAYS = 7             # BTC 24/7: la semana ISO tiene 7 barras diarias
TP_CELLS = [("prior TP=1.0", 1.0), ("sens TP=0.75", 0.75), ("sens TP=1.5", 1.5)]
HS_MULT = 2.0


def simulate(feat: pd.DataFrame, tp_mult: float, cost_mult: float = 1.0) -> pd.DataFrame:
    d = feat.reset_index(drop=True).copy()
    d["iso_week"] = d["time"].dt.strftime("%G-W%V")
    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    d["signal_raw"] = (votes >= 2).astype(float)
    d["size_raw"] = (0.30 / d["realized_vol_20"].clip(lower=0.30)).clip(upper=1.0)  # spot-only

    strat = np.zeros(len(d))
    pos = np.zeros(len(d))
    trades = []
    cost = cost_mult * COST
    weeks = list(d.groupby("iso_week", sort=False))
    for wi in range(1, len(weeks)):
        sig = weeks[wi - 1][1].iloc[-1]       # último día de la semana anterior (causal)
        days = weeks[wi][1]
        if sig["signal_raw"] < 1 or len(days) == 0:
            continue
        size = float(sig["size_raw"])
        atr_w = float(sig["atr_14"]) * np.sqrt(WEEK_DAYS)
        if not np.isfinite(atr_w) or atr_w <= 0 or not np.isfinite(size) or size <= 0:
            continue
        entry = float(days.iloc[0]["open"]) if "open" in days else float(days.iloc[0]["close"])
        tp_px, hs_px = entry + tp_mult * atr_w, entry - HS_MULT * atr_w
        prev, exited, reason, exit_px = entry, False, "week_end", float(days.iloc[-1]["close"])
        for k, (_, row) in enumerate(days.iterrows()):
            if exited:
                break
            de = None
            if float(row["low"]) <= hs_px:
                de, reason = hs_px, "hard_stop"
            elif float(row["high"]) >= tp_px:
                de, reason = tp_px, "take_profit"
            px = de if de is not None else float(row["close"])
            r = size * (px / prev - 1.0)
            if k == 0:
                r -= cost * size
            if de is not None or k == len(days) - 1:
                r -= cost * size
            strat[row.name] += r
            pos[row.name] = size
            prev = px
            if de is not None:
                exited, exit_px = True, de
        trades.append({"week": weeks[wi][0], "pnl_pct": size * (exit_px / entry - 1.0) * 100,
                       "reason": reason})
    d["strat_ret"] = strat
    d["position"] = pos
    d["ret"] = d["close"].pct_change().fillna(0.0)
    d.attrs["trades"] = trades
    return d


def main() -> None:
    from services.common.metrics import paired_exposure_baseline

    df = pd.read_parquet(REPO / "seeds/latest/btcusdt_daily_ohlcv.parquet").sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    print(f"{'celda':<16}{'ventana':<11}{'ret%':>8}{'Sharpe':>8}{'Calmar':>8}{'MaxDD%':>8}"
          f"{'p-val':>7}{'trades':>7}{'WR%':>7}{'PF':>7}  salidas")
    for name, tp in TP_CELLS:
        d = simulate(feat, tp)
        for wlab, lo, hi in [("diseño18-24", "2018-01-01", "2024-12-31"),
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
            reasons: dict[str, int] = {}
            for t in tw:
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            extra = ""
            if name.startswith("prior"):
                b1p = paired_exposure_baseline(s["position"].values, s["ret"].values, 365)
                s2 = simulate(feat, tp, cost_mult=2.0)
                s2 = s2[(s2["time"] >= lo) & (s2["time"] <= hi)].copy()
                s2["equity"] = (1.0 + s2["strat_ret"]).cumprod()
                extra = f"  B1'Calmar={b1p['calmar']:.3f} · costos×2 Calmar={bt.metrics(s2)['calmar']:.3f}"
            print(f"{name:<16}{wlab:<11}{m['total_return_pct']:>8.1f}{m['sharpe']:>8.3f}"
                  f"{m['calmar']:>8.3f}{m['max_dd']:>8.1f}{boot['p_value']:>7.3f}{len(tw):>7}"
                  f"{wr:>7.1f}{str(pf):>7}  {reasons}{extra}")


if __name__ == "__main__":
    main()
