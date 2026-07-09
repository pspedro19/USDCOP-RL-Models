#!/usr/bin/env python3
"""Reporte OOS-2025 ÚNICO (metodología COP: diseño/entrenamiento ≤2024, juez = TODO 2025).

Evalúa cada estrategia SOLO sobre 2025 con el motor de costos real del activo:
retorno, Sharpe, p-value (block-bootstrap sobre retornos diarios), WR/PF por trade,
MaxDD, $10K→, y baseline buy-and-hold del activo. N<20 trades ⇒ el p-value se reporta
del bootstrap diario (365/252 obs), pero el conteo de trades se muestra siempre
(quant-constitution §6). Usage: python scripts/analysis/oos2025_report.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.mr_dip_hypothesis import dip_window, seg_trades  # noqa: E402


def report(asset: str) -> None:
    if asset == "btcusdt":
        from src.btc_strategy import backtest as bt
        from src.btc_strategy.indicators import build_daily_features, classify_regime
        from src.btc_strategy.strategies import build_positions, intent_hodl, intent_trend
        seed, ann = "btcusdt_daily_ohlcv.parquet", 365
        base = {"btc_hodl_b1 (baseline)": intent_hodl, "btc_trend_b2 (campeona)": intent_trend}
        mr_champ = intent_trend
        combine = lambda f, w: np.maximum(mr_champ(f), w)  # noqa: E731
    else:
        from src.gold_rl import backtest as bt
        from src.gold_rl.indicators import build_daily_features, classify_regime
        from src.gold_rl.strategies import (build_positions, direction_long_only,
                                            direction_trend_ensemble, direction_trend_follower)
        seed, ann = "xauusd_daily_ohlcv.parquet", 252
        base = {"gold_long_only_b1 (campeona)": direction_long_only,
                "gold_trend_b2": direction_trend_follower,
                "gold_trend_ens": direction_trend_ensemble}
        combine = None

    df = pd.read_parquet(REPO / "seeds" / "latest" / seed).sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = classify_regime(build_daily_features(df))

    def oos_row(pos_df: pd.DataFrame) -> dict | None:
        d = bt.compute_returns(pos_df)
        s = d[(d["time"] >= "2025-01-01") & (d["time"] <= "2025-12-31")].copy()
        if len(s) < 30:
            return None
        s["equity"] = (1.0 + s["strat_ret"]).cumprod()
        m = bt.metrics(s)
        boot = bt.block_bootstrap_pvalue(s["strat_ret"])
        n_tr, wr, pf = seg_trades(s)
        return {"ret": m["total_return_pct"], "sharpe": m["sharpe"], "p": boot["p_value"],
                "ci": [boot["ci_low"], boot["ci_high"]], "wr": wr, "pf": pf,
                "dd": m["max_dd"], "trades": n_tr, "k10": round(10000 * (1 + m["total_return_pct"] / 100))}

    rows: dict[str, dict] = {}
    for name, fn in base.items():
        r = oos_row(build_positions(feat, fn))
        if r:
            rows[name] = r
    # MR prior (celda pre-registrada 1.5σ/5d — la única del prior, sin picking)
    win = dip_window(feat, 1.5, 5, ann)
    r = oos_row(build_positions(feat, lambda f, w=win: w))
    if r:
        rows[("btc" if asset == "btcusdt" else "gold") + "_dip_mr (standalone, prior)"] = r
    if combine is not None:
        r = oos_row(build_positions(feat, lambda f, w=win: combine(f, w)))
        if r:
            rows["MR combinada (prior)"] = r

    y = df[(df["time"] >= "2025-01-01") & (df["time"] <= "2025-12-31")]
    bh = float(y["close"].iloc[-1] / y["close"].iloc[0] - 1.0) * 100

    print(f"\n{'='*112}\n{asset.upper()} — OOS 2025 ÚNICO (diseño ≤2024) · buy&hold 2025 = {bh:+.1f}%")
    print(f"{'estrategia':<34}{'Ret%':>7}{'Sharpe':>8}{'p-val':>7}{'CI95 ann':>18}{'WR%':>7}{'PF':>7}{'MaxDD%':>8}{'Trades':>7}{'$10K→':>9}")
    for n, r in rows.items():
        ci = f"[{r['ci'][0]:+.0f},{r['ci'][1]:+.0f}]"
        print(f"{n:<34}{r['ret']:>7.1f}{r['sharpe']:>8.3f}{r['p']:>7.3f}{ci:>18}{r['wr']:>7.1f}"
              f"{str(r['pf']):>7}{r['dd']:>8.1f}{r['trades']:>7}{r['k10']:>9,}")


if __name__ == "__main__":
    for a in ("xauusd", "btcusdt"):
        report(a)
