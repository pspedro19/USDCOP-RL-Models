"""Portfolio layer + LATAM TSMOM (master plan OLA 7 / audit II-3, II-5, III-4).

  P2  Equal-risk mix of the three sleeves (COP v11 weekly, gold_trend_ens, btc_trend_b2)
      with rolling-|rho| monitor and an aggregate DD breaker (12% -> x0.5, 18% -> flat;
      priors ex-ante, ADR-0012 style: never relaxed in drawdown).
      H-PORT-01: block-bootstrap ΔCalmar(mix, best sleeve).
  P1  LATAM TSMOM 4/8/13w on {COP, MXN, BRL} daily closes (resampled from the m5 seeds
      already in aux_pairs; CLP has no seed -> documented exclusion, not silent).
      H-LATAM-02: ΔCalmar(TSMOM basket, B1' basket). Carry leg (H-LATAM-01) requires
      measured policy rates + broker swaps -> stays gated on H-COP-CARRY-00 (honest).

Reuses: `services.common.metrics._ann_return_dd_calmar`, block bootstrap from
`scripts.analysis.cop_null_suite`. Read-only. Run: python -m scripts.analysis.portfolio_layer
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from services.common.metrics import _ann_return_dd_calmar  # noqa: E402
from scripts.analysis.cop_null_suite import (  # noqa: E402
    block_bootstrap_delta_calmar, v11_weekly_returns_2025,
)

PUB = REPO / "usdcop-trading-dashboard/public/data/strategies"
WEEKS = 52


def sleeve_weekly(strategy_id: str, version: str, year: int = 2026) -> pd.Series:
    """Weekly returns of a published sleeve from its immutable bundle trades (2025 slice)."""
    f = PUB / strategy_id / "backtests" / version / f"trades_{year}.json"
    data = json.loads(f.read_text(encoding="utf-8"))
    rows = []
    for t in data.get("trades", []):
        ts = pd.Timestamp(str(t.get("exit_timestamp") or t.get("timestamp"))).tz_localize(None)
        rows.append((ts, float(t.get("pnl_pct", 0.0)) / 100.0))
    s = pd.Series(dict(rows)).sort_index()
    s = s[(s.index >= "2025-01-01") & (s.index <= "2025-12-31")]
    return s.resample("W-MON").sum().reindex(
        pd.date_range("2025-01-06", "2025-12-29", freq="W-MON"), fill_value=0.0)


def equal_risk_mix(sleeves: dict[str, pd.Series]) -> pd.Series:
    df = pd.DataFrame(sleeves).fillna(0.0)
    vol = df.rolling(13, min_periods=4).std().shift(1)  # causal inverse-vol weights
    w = (1.0 / vol.replace(0, np.nan))
    w = w.div(w.sum(axis=1), axis=0).fillna(1.0 / df.shape[1])
    mix = (df * w).sum(axis=1)
    # aggregate DD breaker (priors ex-ante: 12% -> x0.5, 18% -> flat)
    eq = (1 + mix).cumprod()
    dd = eq / eq.cummax() - 1.0
    scale = pd.Series(1.0, index=mix.index)
    scale[dd.shift(1) < -0.12] = 0.5
    scale[dd.shift(1) < -0.18] = 0.0
    return mix * scale


def latam_tsmom() -> dict:
    """TSMOM 4/8/13w votes on daily closes resampled from the m5 seeds (COP, MXN, BRL)."""
    out = {}
    basket, b1p = [], []
    for sym, seed in [("COP", "usdcop_m5_ohlcv.parquet"), ("MXN", "usdmxn_m5_ohlcv.parquet"),
                      ("BRL", "usdbrl_m5_ohlcv.parquet")]:
        px = pd.read_parquet(REPO / "seeds/latest" / seed)
        px["time"] = pd.to_datetime(px["time"]).dt.tz_localize(None)
        px["close"] = px["close"].astype(float)  # seeds store Decimal
        daily = px.set_index("time")["close"].resample("1D").last().dropna()
        wclose = daily.resample("W-FRI").last()
        wk = wclose.pct_change().dropna()
        votes = sum(np.sign(wclose.pct_change(w).fillna(0.0)) for w in (4, 8, 13)) / 3.0
        pos = votes.shift(1).reindex(wk.index).fillna(0.0)  # causal
        strat = pos * wk
        basket.append(strat)
        b1p.append(float(pos.abs().mean()) * wk)  # paired-exposure per pair
        out[sym] = _ann_return_dd_calmar(strat.dropna().values, WEEKS)
    bk = pd.concat(basket, axis=1).mean(axis=1).dropna()
    bp = pd.concat(b1p, axis=1).mean(axis=1).reindex(bk.index).fillna(0.0)
    out["basket"] = _ann_return_dd_calmar(bk.values, WEEKS)
    out["basket_b1prime"] = _ann_return_dd_calmar(bp.values, WEEKS)
    out["H-LATAM-02"] = block_bootstrap_delta_calmar(bk.values, bp.values)
    return out


def main() -> int:
    print("=" * 70)
    print("PORTFOLIO LAYER + LATAM TSMOM (OLA 7)")
    print("=" * 70)

    v11, _ = v11_weekly_returns_2025()
    sleeves = {"cop_v11": v11,
               "gold_ens": sleeve_weekly("gold_trend_ens", "1.3.0"),
               "btc_b2": sleeve_weekly("btc_trend_b2", "1.2.1")}
    for k, s in sleeves.items():
        st = _ann_return_dd_calmar(s.values, WEEKS)
        print(f"  sleeve {k:9s}: ann={st['ann_return_pct']}%  MaxDD={st['max_dd_pct']}%  "
              f"Calmar={st['calmar']}")
    corr = pd.DataFrame(sleeves).corr()
    print(f"  |rho| máx entre sleeves 2025: {corr.abs().where(~np.eye(3, dtype=bool)).max().max():.2f}")

    mix = equal_risk_mix(sleeves)
    sm = _ann_return_dd_calmar(mix.values, WEEKS)
    best_id = max(sleeves, key=lambda k: _ann_return_dd_calmar(sleeves[k].values, WEEKS)["calmar"])
    best = sleeves[best_id]
    res = block_bootstrap_delta_calmar(mix.values, best.reindex(mix.index).fillna(0).values)
    print(f"\n[P2] mix equal-risk: ann={sm['ann_return_pct']}%  MaxDD={sm['max_dd_pct']}%  "
          f"Calmar={sm['calmar']}")
    print(f"[H-PORT-01] ΔCalmar(mix, mejor sleeve={best_id}) = {res['delta_calmar_mean']} "
          f"IC95={res['ci95']} excluye0={res['excludes_zero']}")

    print("\n[P1] LATAM TSMOM 4/8/13w (COP/MXN/BRL; CLP excluido: sin seed — exclusión declarada)")
    lat = latam_tsmom()
    for k in ("COP", "MXN", "BRL", "basket", "basket_b1prime"):
        s = lat[k]
        print(f"  {k:15s}: ann={s['ann_return_pct']}%  MaxDD={s['max_dd_pct']}%  Calmar={s['calmar']}")
    r = lat["H-LATAM-02"]
    print(f"[H-LATAM-02] ΔCalmar(basket, B1') = {r['delta_calmar_mean']} IC95={r['ci95']} "
          f"excluye0={r['excludes_zero']}")
    print("[H-LATAM-01 carry] GATED: requiere tasas medidas + swaps del broker (H-COP-CARRY-00)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
