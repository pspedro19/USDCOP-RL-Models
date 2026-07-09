#!/usr/bin/env python3
"""Forward-forecast visual para Oro/BTC con el MISMO lenguaje que USD/COP (imagen + horizontes).

Honestidad primero: Oro/BTC no tienen zoo ML (y el zoo COP tiene DA~50% — no se replica ruido).
Lo que SÍ es riguroso y visualizable:
  • Fan-chart de bandas de volatilidad realizada (random-walk, drift 0) a horizontes {1,5,10,20,30}d.
  • El POSICIONAMIENTO actual de la estrategia campeona (dirección + exposición vol-target).
  • DA REAL por horizonte: walk-forward 2025 — ¿el signo de la posición predijo el signo del
    retorno a H días? (métrica medida, no inventada).

Salidas por asset (servidas por la ruta gateada /api/forecasting/<asset>/...):
  public/forecasting/<asset>/forward_<asset>_<YYYY>_W<WW>.png   (fan chart, tema oscuro)
  public/forecasting/<asset>/forward.json                        (horizontes + DA + imagen)

Usage: python scripts/pipeline/generate_asset_forward_charts.py [--asset xauusd|btcusdt|all]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT_BASE = REPO / "usdcop-trading-dashboard" / "public" / "forecasting"
HORIZONS = [1, 5, 10, 20, 30]

ASSETS = {
    "xauusd": {
        "seed": "seeds/latest/xauusd_daily_ohlcv.parquet",
        "display": "Oro (XAU/USD)",
        "champion": "gold_long_only_b1",
        "ann": 252,
    },
    "btcusdt": {
        "seed": "seeds/latest/btcusdt_daily_ohlcv.parquet",
        "display": "Bitcoin (BTC/USDT)",
        "champion": "btc_trend_b2",
        "ann": 365,
    },
}


def champion_direction(asset: str, df: pd.DataFrame) -> pd.Series:
    """Dirección diaria de la campeona (causal — sin mirar el futuro)."""
    if asset == "xauusd":
        return pd.Series(1.0, index=df.index)  # B1: siempre long
    sma = df["close"].rolling(100).mean()
    # ADX Wilder simplificado suficiente para dirección B2 (mismo criterio del pipeline)
    from src.btc_strategy.indicators import build_daily_features
    feat = build_daily_features(df)
    long = ((feat["close"] > feat["sma_100"]) & (feat["adx_14"] > 25)).astype(float)
    return long


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="all", choices=["all", *ASSETS])
    a = ap.parse_args()
    targets = list(ASSETS) if a.asset == "all" else [a.asset]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for asset in targets:
        cfg = ASSETS[asset]
        df = pd.read_parquet(REPO / cfg["seed"]).sort_values("time").reset_index(drop=True)
        close = df["close"].astype(float)
        logret = np.log(close / close.shift(1))
        sigma_d = float(logret.tail(60).std())  # vol realizada 60d (diaria)
        last = float(close.iloc[-1])
        last_dt = pd.Timestamp(df["time"].iloc[-1])

        direction = champion_direction(asset, df)
        cur_dir = float(direction.iloc[-1])
        # exposición vol-target aproximada (informativa)
        target_vol = 0.10 if asset == "xauusd" else 0.30
        ann = cfg["ann"]
        exposure = min(1.0, target_vol / max(sigma_d * np.sqrt(ann), 1e-9)) * (1.0 if cur_dir > 0 else 0.0)

        # DA walk-forward 2025 por horizonte: signo(posición_t) vs signo(ret t→t+H)
        d25 = df[df["time"].dt.year == 2025].index
        horizons_out = []
        for h in HORIZONS:
            hits = tot = 0
            for i in d25:
                if i + h >= len(close):
                    continue
                sgn_pos = direction.iloc[i]
                if sgn_pos == 0:
                    continue
                fwd = close.iloc[i + h] - close.iloc[i]
                tot += 1
                if (fwd > 0 and sgn_pos > 0) or (fwd < 0 and sgn_pos < 0):
                    hits += 1
            move = sigma_d * np.sqrt(h)
            horizons_out.append({
                "h_days": h,
                "exp_move_pct": round(move * 100, 2),          # ±1σ del RW
                "ci68_pct": [round(-move * 100, 2), round(move * 100, 2)],
                "ci95_pct": [round(-1.96 * move * 100, 2), round(1.96 * move * 100, 2)],
                "da_2025_pct": round(100 * hits / tot, 1) if tot else None,
                "n_2025": tot,
            })

        # ── PNG fan chart (tema oscuro, mismo lenguaje visual que COP) ──
        hist = df.tail(90)
        fut_days = np.arange(1, 31)
        fut_dates = [last_dt + pd.Timedelta(days=int(k)) for k in fut_days]
        med = np.full(len(fut_days), last)
        b68 = last * np.exp(sigma_d * np.sqrt(fut_days))
        l68 = last * np.exp(-sigma_d * np.sqrt(fut_days))
        b95 = last * np.exp(1.96 * sigma_d * np.sqrt(fut_days))
        l95 = last * np.exp(-1.96 * sigma_d * np.sqrt(fut_days))

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0b1220")
        ax.set_facecolor("#0b1220")
        ax.plot(hist["time"], hist["close"], color="#e2e8f0", lw=1.6, label="Histórico (90d)")
        ax.fill_between(fut_dates, l95, b95, color="#8b5cf6", alpha=0.18, label="Banda 95% (vol realizada)")
        ax.fill_between(fut_dates, l68, b68, color="#22d3ee", alpha=0.25, label="Banda 68%")
        ax.plot(fut_dates, med, color="#a78bfa", lw=1.8, ls="--", label="Mediana RW (drift 0)")
        dir_txt = "LONG" if cur_dir > 0 else "FLAT"
        ax.axvline(last_dt, color="#f59e0b", lw=1, ls=":")
        ax.set_title(f"{cfg['display']} — Forward {HORIZONS[-1]}d · Estrategia {cfg['champion']}: "
                     f"{dir_txt} · exposición {exposure:.2f}x", color="white", fontsize=12)
        for s in ax.spines.values():
            s.set_color("#334155")
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#0b1220", labelcolor="#cbd5e1", edgecolor="#334155", fontsize=8)
        ax.grid(color="#1e293b", lw=0.5)

        iso = datetime.now(timezone.utc).isocalendar()
        img_name = f"forward_{asset}_{iso[0]}_W{iso[1]:02d}.png"
        out_dir = OUT_BASE / asset
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / img_name, dpi=110, bbox_inches="tight", facecolor="#0b1220")
        plt.close(fig)

        doc = {
            "asset": asset,
            "display": cfg["display"],
            "champion": cfg["champion"],
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "last_close": last,
            "last_bar": str(last_dt),
            "direction": dir_txt,
            "exposure": round(exposure, 2),
            "vol_daily_pct": round(sigma_d * 100, 2),
            "image": f"forecasting/{asset}/{img_name}",
            "horizons": horizons_out,
            "methodology": ("Bandas de volatilidad realizada 60d (random-walk, drift 0) + "
                            "posicionamiento actual de la estrategia campeona. DA por horizonte "
                            "= walk-forward 2025 medido. NO es una predicción ML."),
        }
        with open(out_dir / "forward.json", "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"[{asset}] {img_name} + forward.json  dir={dir_txt} exp={exposure:.2f} "
              f"σd={sigma_d*100:.2f}%  DA1d2025={horizons_out[0]['da_2025_pct']}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
