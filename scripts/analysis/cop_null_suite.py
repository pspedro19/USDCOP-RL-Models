"""COP-NULL suite (master plan OLA 4 / audit II-2, III-1.3) — the pending statistical
audit of the production strategy.

  NULL-A       always-SHORT 1x every week, SAME frozen v11 TP/HS mechanics
               (src/forecasting/adaptive_stops.py). "Does the ML/regime layer add
               anything over just being short?"
  NULL-B       constant short exposure == v11's realized mean weekly |leverage|
               (incl. flat weeks), no exits. "Timing, or just less beta?"
  DECOMP       v11 2025 PnL split: spot direction vs exit-mechanics contribution.
               (Carry leg NOT measurable without broker swaps — H-COP-CARRY-00 first.)
  STRESS-2122  NULL-A walked through 2021-2022 (the violent COP depreciation):
               measures the structural damage of the short bias.
  H-COP-V11-01 block-bootstrap ΔCalmar(v11, NULL-A) on 2025 weekly returns.
               If the CI does not exclude 0 -> NULL-A IS the strategy (honest gate).

Reuses (never re-derives): diagnostic loader + week simulator
(`scripts.diagnostics.diagnose_smart_simple_v1`), frozen stops config
(`AdaptiveStopsConfig()` defaults = v11), shared Calmar math (`services.common.metrics`).
Read-only; prints a registry-ready report. Run:
    python -m scripts.analysis.cop_null_suite
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.diagnostics.diagnose_smart_simple_v1 import (  # noqa: E402
    STOPS_CONFIG, _get_bars, load_data, simulate_week,
)
from src.forecasting.adaptive_stops import compute_adaptive_stops  # noqa: E402
from services.common.metrics import _ann_return_dd_calmar  # noqa: E402

TRADES_2025 = (REPO / "usdcop-trading-dashboard/public/data/production/trades/"
               "smart_simple_v11_2025.json")
WEEKS_PER_YEAR = 52


def weekly_short_returns(df: pd.DataFrame, start: str, end: str, *,
                         use_stops: bool, leverage: float = 1.0) -> pd.Series:
    """Always-short weekly returns over [start, end]. use_stops=True replays the frozen
    v11 TP/HS mechanics bar-by-bar; False holds Monday-open -> Friday-close (pure beta)."""
    mondays = pd.date_range(start, end, freq="W-MON")
    out = {}
    vol = df.set_index("date")["return_1d"].rolling(20).std()
    closes = df.set_index("date")["close"]
    for monday in mondays:
        bars = _get_bars(df, monday)
        if not bars:
            continue
        # diagnostic convention: signal on Monday, ENTER at Monday close, exits on Tue-Fri bars
        entry_val = closes.asof(monday)
        if pd.isna(entry_val):
            continue
        entry = float(entry_val)
        v = float(vol.asof(monday - pd.Timedelta(days=1)) or 0.008)
        if use_stops:
            stops = compute_adaptive_stops(v, STOPS_CONFIG)
            exit_px, _reason, _i = simulate_week(-1, entry, bars,
                                                 stops.hard_stop_pct, stops.take_profit_pct)
        else:
            exit_px = bars[-1]["close"]
        out[monday] = leverage * (entry - exit_px) / entry  # SHORT pnl
    return pd.Series(out)


def v11_weekly_returns_2025() -> pd.Series:
    data = json.loads(TRADES_2025.read_text(encoding="utf-8"))
    idx = pd.date_range("2025-01-01", "2025-12-31", freq="W-MON")
    ret = pd.Series(0.0, index=idx)
    levs = pd.Series(0.0, index=idx)
    for t in data.get("trades", []):
        ts = pd.Timestamp(t["timestamp"]).tz_localize(None)
        monday = (ts - pd.Timedelta(days=ts.weekday())).normalize()
        key = idx[idx.get_indexer([monday], method="nearest")[0]]
        ret[key] += float(t.get("pnl_pct", 0.0)) / 100.0
        levs[key] = abs(float(t.get("leverage", 1.0)))
    return ret, float(levs.mean())


def block_bootstrap_delta_calmar(a: np.ndarray, b: np.ndarray, *, block: int = 6,
                                 n_boot: int = 5000, seed: int = 42) -> dict:
    """Paired moving-block bootstrap of Calmar(a) − Calmar(b) on aligned weekly returns.
    Blocks preserve autocorrelation (quant-constitution: iid bootstrap inflates
    significance on autocorrelated weekly series)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, size=(n // block) + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        ca = _ann_return_dd_calmar(a[idx], WEEKS_PER_YEAR)["calmar"]
        cb = _ann_return_dd_calmar(b[idx], WEEKS_PER_YEAR)["calmar"]
        deltas[i] = ca - cb
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"delta_calmar_mean": round(float(deltas.mean()), 3),
            "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def main() -> int:
    df, _cols = load_data()
    print("=" * 72)
    print("COP-NULL SUITE — auditoría estadística de smart_simple_v11 (OLA 4)")
    print("=" * 72)

    # ---- NULL-A / NULL-B on 2025 ------------------------------------------------
    null_a = weekly_short_returns(df, "2025-01-01", "2025-12-31", use_stops=True)
    v11, mean_lev = v11_weekly_returns_2025()
    null_b = weekly_short_returns(df, "2025-01-01", "2025-12-31",
                                  use_stops=False, leverage=mean_lev * (34 / 52))
    # NULL-B exposure = mean leverage of traded weeks scaled by participation (flat weeks = 0)

    sa = _ann_return_dd_calmar(null_a.values, WEEKS_PER_YEAR)
    sb = _ann_return_dd_calmar(null_b.values, WEEKS_PER_YEAR)
    sv = _ann_return_dd_calmar(v11.values, WEEKS_PER_YEAR)
    print(f"\n[2025] v11      : ann={sv['ann_return_pct']}%  MaxDD={sv['max_dd_pct']}%  "
          f"Calmar={sv['calmar']}  Sharpe={sv['sharpe']}")
    print(f"[2025] NULL-A   : ann={sa['ann_return_pct']}%  MaxDD={sa['max_dd_pct']}%  "
          f"Calmar={sa['calmar']}  Sharpe={sa['sharpe']}  (siempre-short 1x + TP/HS v11)")
    print(f"[2025] NULL-B   : ann={sb['ann_return_pct']}%  MaxDD={sb['max_dd_pct']}%  "
          f"Calmar={sb['calmar']}  (exposición constante ~{mean_lev * (34/52):.2f}x)")

    # ---- H-COP-V11-01 -------------------------------------------------------------
    common = v11.index.intersection(null_a.index)
    res = block_bootstrap_delta_calmar(v11[common].values, null_a[common].values)
    verdict = ("v11 SUPERA a NULL-A (IC excluye 0)" if res["excludes_zero"]
               and res["delta_calmar_mean"] > 0 else
               "NO se puede afirmar que v11 supere a NULL-A -> NULL-A ES la estrategia "
               "(honest gate) salvo que el forward 2026 lo rescate")
    print(f"\n[H-COP-V11-01] ΔCalmar(v11, NULL-A) = {res['delta_calmar_mean']} "
          f"IC95={res['ci95']} -> {verdict}")

    # ---- DECOMP (spot vs mecánica de salidas) --------------------------------------
    hold = weekly_short_returns(df, "2025-01-01", "2025-12-31", use_stops=False)
    common2 = null_a.index.intersection(hold.index)
    mech = (null_a[common2] - hold[common2])
    print(f"\n[DECOMP 2025, siempre-short] spot(hold-to-Friday) ann="
          f"{_ann_return_dd_calmar(hold.values, WEEKS_PER_YEAR)['ann_return_pct']}%  "
          f"aporte de la mecánica TP/HS = {mech.sum()*100:+.2f}pp acumulados")
    print("[DECOMP] pata carry: NO medible sin swaps del broker -> H-COP-CARRY-00 primero")

    # ---- STRESS-2122 ----------------------------------------------------------------
    stress = weekly_short_returns(df, "2021-01-01", "2022-12-31", use_stops=True)
    ss = _ann_return_dd_calmar(stress.values, WEEKS_PER_YEAR)
    cum = float((1 + stress).prod() - 1) * 100
    print(f"\n[STRESS-2122] siempre-short + TP/HS por la depreciación 2021-22: "
          f"acumulado={cum:+.1f}%  ann={ss['ann_return_pct']}%  MaxDD={ss['max_dd_pct']}%")
    print("  -> este es el daño estructural del sesgo corto que el OOS 2023-25 nunca pisó")

    print("\n[registro] pegar estos resultados en specs/assets/usdcop/HYPOTHESIS-REGISTRY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
