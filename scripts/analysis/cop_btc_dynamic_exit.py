#!/usr/bin/env python3
"""H-DYNEXIT-XASSET-01 — familia dinámica (señal-off + Chandelier) en USDCOP y BTC (§8.8).

Priors idénticos a §8.7 (M=3.0, sens {2,4}), cero re-tuning. BTC long-flat spot-only;
USDCOP simétrica long/short (trailing espejo). Todas las celdas se reportan (+6 trials).

Usage: python scripts/analysis/cop_btc_dynamic_exit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CELLS = [("prior M=3.0", 3.0), ("sens M=2.0", 2.0), ("sens M=4.0", 4.0)]


def wilder_atr(h: pd.Series, l: pd.Series, c: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def prep(df: pd.DataFrame, ann: int, target_vol: float, vol_floor: float, cap: float) -> pd.DataFrame:
    d = df.sort_values("time").reset_index(drop=True).copy()
    d["atr_14"] = wilder_atr(d["high"], d["low"], d["close"])
    d["rvol"] = d["close"].pct_change().rolling(20).std() * np.sqrt(ann)
    d["size_raw"] = (target_vol / d["rvol"].clip(lower=vol_floor)).clip(upper=cap)
    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    d["votes"] = votes
    return d


def simulate(d: pd.DataFrame, m_trail: float, *, allow_short: bool, cost_side: float,
             cost_mult: float = 1.0) -> pd.DataFrame:
    n = len(d)
    strat, pos = np.zeros(n), np.zeros(n)
    trades = []
    cost = cost_mult * cost_side
    open_col = "open" if "open" in d else "close"
    in_tr, side, size, entry_i, entry_px, ext_px, trail, prev = False, 0, 0.0, -1, 0.0, 0.0, 0.0, 0.0

    def sig_dir(i: int) -> int:
        v = int(d["votes"].iloc[i])
        if v >= 2:
            return 1
        return -1 if allow_short and v <= 1 else 0

    for i in range(1, n):
        row = d.iloc[i]
        s_y = sig_dir(i - 1)  # señal al cierre de AYER (causal)
        atr_y = float(d["atr_14"].iloc[i - 1])
        if not in_tr:
            if s_y != 0 and np.isfinite(atr_y) and atr_y > 0 and np.isfinite(d["size_raw"].iloc[i - 1]):
                in_tr, side, entry_i = True, s_y, i
                size = float(d["size_raw"].iloc[i - 1])
                entry_px = float(row[open_col])
                ext_px = entry_px  # extremo favorable (max close long / min close short)
                trail = entry_px - side * m_trail * atr_y
                r = side * size * (float(row["close"]) / entry_px - 1.0) - cost * size
                strat[i] += r
                pos[i] = side * size
                ext_px = max(ext_px, float(row["close"])) if side > 0 else min(ext_px, float(row["close"]))
                trail = max(trail, ext_px - m_trail * float(row["atr_14"])) if side > 0 \
                    else min(trail, ext_px + m_trail * float(row["atr_14"]))
                prev = float(row["close"])
            continue
        exited, reason, exit_px = False, "", 0.0
        if s_y != side:                                   # señal murió o giró → sale al open
            exited, reason, exit_px = True, "signal_off", float(row[open_col])
        elif side > 0 and float(row["low"]) <= trail:
            exited, reason, exit_px = True, "trailing_stop", trail
        elif side < 0 and float(row["high"]) >= trail:
            exited, reason, exit_px = True, "trailing_stop", trail
        px_end = exit_px if exited else float(row["close"])
        r = side * size * (px_end / prev - 1.0)
        if exited:
            r -= cost * size
        strat[i] += r
        pos[i] = side * size
        prev = px_end
        if exited:
            trades.append({"exit_time": row["time"], "days": i - entry_i, "side": side,
                           "pnl_pct": side * size * (px_end / entry_px - 1.0) * 100, "reason": reason})
            in_tr = False
        else:
            ext_px = max(ext_px, float(row["close"])) if side > 0 else min(ext_px, float(row["close"]))
            trail = max(trail, ext_px - m_trail * float(row["atr_14"])) if side > 0 \
                else min(trail, ext_px + m_trail * float(row["atr_14"]))
    d = d.copy()
    d["strat_ret"] = strat
    d["position"] = pos
    d["ret"] = d["close"].pct_change().fillna(0.0)
    d.attrs["trades"] = trades
    return d


def metrics(s: pd.DataFrame, ann: int) -> dict:
    r = s["strat_ret"].dropna()
    eq = (1.0 + r).cumprod()
    total = float(eq.iloc[-1] - 1.0) if len(eq) else 0.0
    years = len(r) / ann
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and 1 + total > 0 else 0.0
    sharpe = float(r.mean() / r.std() * np.sqrt(ann)) if r.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    return {"ret": round(total * 100, 1), "sharpe": round(sharpe, 3),
            "calmar": round(cagr / abs(mdd), 3) if mdd < 0 else 0.0, "dd": round(mdd * 100, 1)}


def boot_p(r: pd.Series, ann: int) -> float:
    r = r.dropna().values
    if len(r) < 60:
        return 1.0
    rng = np.random.default_rng(42)
    n, block = len(r), 20
    means = np.empty(5000)
    for b in range(5000):
        starts = rng.integers(0, n - block + 1, size=n // block + 1)
        means[b] = np.concatenate([r[s:s + block] for s in starts])[:n].mean()
    return round(float((means <= 0).mean()), 4)


def run(asset: str) -> None:
    if asset == "btcusdt":
        df = pd.read_parquet(REPO / "seeds/latest/btcusdt_daily_ohlcv.parquet")
        d0 = prep(df, 365, 0.30, 0.30, 1.0)
        allow_short, cost_side, ann = False, 13.0 / 1e4, 365
        design = ("2018-01-01", "2024-12-31")
        bar = "bar B2 diseño: Sharpe 1.581 / Calmar 2.839"
    else:
        df = pd.read_parquet(REPO / "seeds/latest/usdcop_daily_ohlcv.parquet")
        d0 = prep(df, 252, 0.10, 0.04, 1.5)
        allow_short, cost_side, ann = True, 5.0 / 1e4, 252
        design = ("2021-01-01", "2024-12-31")
        bar = "contexto: v11 2025 +25.6% (34 tr) · NULL-A 2025 Calmar 1.52"

    print(f"\n{'='*112}\n{asset.upper()} — dinámica señal-off + Chandelier · {bar}")
    print(f"{'celda':<14}{'ventana':<11}{'ret%':>7}{'Sharpe':>8}{'Calmar':>8}{'MaxDD%':>8}{'p-val':>7}"
          f"{'trades':>7}{'WR%':>6}{'L/S':>8}{'hold med/max':>13}  salidas")
    for name, m_ in CELLS:
        d = simulate(d0, m_, allow_short=allow_short, cost_side=cost_side)
        for wlab, lo, hi in [("diseño", design[0], design[1]), ("OOS2025", "2025-01-01", "2025-12-31")]:
            s = d[(d["time"] >= lo) & (d["time"] <= hi)]
            met = metrics(s, ann)
            tw = [t for t in d.attrs["trades"] if lo <= str(t["exit_time"])[:10] <= hi]
            wins = [t for t in tw if t["pnl_pct"] > 0]
            wr = round(100 * len(wins) / len(tw), 1) if tw else 0.0
            ls = f"{sum(1 for t in tw if t['side'] > 0)}/{sum(1 for t in tw if t['side'] < 0)}"
            durs = sorted(t["days"] for t in tw)
            dur = f"{durs[len(durs)//2]}/{durs[-1]}" if durs else "—"
            reasons: dict[str, int] = {}
            for t in tw:
                reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1
            print(f"{name:<14}{wlab:<11}{met['ret']:>7.1f}{met['sharpe']:>8.3f}{met['calmar']:>8.3f}"
                  f"{met['dd']:>8.1f}{boot_p(s['strat_ret'], ann):>7.3f}{len(tw):>7}{wr:>6.1f}"
                  f"{ls:>8}{dur:>13}  {reasons}")


if __name__ == "__main__":
    for a in ("usdcop", "btcusdt"):
        run(a)
