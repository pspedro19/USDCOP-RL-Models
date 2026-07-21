"""Per-strategy metrics split by the methodology's own windows.

Contract: CTR-QUANT-EVIDENCE-001

`CLAUDE.md` states the methodology plainly: **trained <= Dec-2024, 2025 = backtest (OOS),
2026 = production**. Every report I produced before this one used FULL history -- gold from
2004, BTC from 2017 -- which blends the design period with the evaluation period and makes a
strategy look like whatever its longest, most favourable stretch was.

`btc_trend_b2` is the clearest case: 27.78% annualized over 2017-2026 and **-1.37% in 2025**,
the only truly held-out year. Quoting the first number without the second is the single most
misleading thing this repo can do, and I did it.

Three windows, reported separately, never merged:

  DESIGN   <= 2024-12-31   the strategy was built looking at this. Not evidence.
  OOS      2025            held out. This is the backtest verdict.
  LIVE     2026 YTD        forward. Nothing about 2026 informed any design decision.

A strategy is only interesting if OOS and LIVE agree with DESIGN. When they disagree, DESIGN is
the number to distrust -- it is the one that had the opportunity to be fitted.

Run: python -m scripts.analysis.window_report
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from services.common.metrics import _ann_return_dd_calmar, omega_ratio  # noqa: E402

WINDOWS = {
    "DESIGN(<=2024)": (None, "2024-12-31"),
    "OOS(2025)": ("2025-01-01", "2025-12-31"),
    "LIVE(2026 YTD)": ("2026-01-01", None),
}

STRATEGIES = ("xauusd", "xauusd_simple", "btcusdt", "spx500", "spx500_simple", "usdcop")


def _slice(idx: pd.DatetimeIndex, lo: str | None, hi: str | None) -> np.ndarray:
    """Boolean window mask.

    `np.asarray`, not `.to_numpy()`: comparing an Index to a scalar already returns a numpy
    array, and calling `.to_numpy()` on it raises AttributeError. Same trap that silently
    blanked the OOS column in exposure_matched.py — there it was swallowed by a bare except and
    printed 'n/a'; here it surfaced as an empty table, which is at least visible.
    """
    m = np.ones(len(idx), dtype=bool)
    if lo:
        m &= np.asarray(idx >= pd.Timestamp(lo), dtype=bool)
    if hi:
        m &= np.asarray(idx <= pd.Timestamp(hi), dtype=bool)
    return m


def analyse(name: str) -> dict | None:
    from scripts.analysis.profitability_adapters import ADAPTERS

    s = ADAPTERS[name]()
    idx = pd.to_datetime(pd.Index(s.index)).tz_localize(None)
    out = {"strategy_id": s.strategy_id, "asset": s.asset, "clock": s.clock_label,
           "windows": {}}

    for label, (lo, hi) in WINDOWS.items():
        m = _slice(idx, lo, hi)
        n = int(m.sum())
        if n < 5:
            out["windows"][label] = {"n_periods": n, "status": "sin datos suficientes"}
            continue
        r = s.strat_ret[m]
        a = s.asset_ret[m]
        st = _ann_return_dd_calmar(r, s.clock)
        # Trades are counted as position changes inside the window, not inherited from the
        # full-history total -- a strategy with 250 lifetime trades may have 3 in 2026.
        pos = s.position[m]
        n_tr = int((np.abs(np.diff(pos, prepend=pos[0] if len(pos) else 0.0)) > 1e-9).sum())
        out["windows"][label] = {
            "n_periods": n, "n_trades": n_tr,
            "start": str(idx[m][0].date()), "end": str(idx[m][-1].date()),
            "net_return_pct": round(float((np.prod(1 + r) - 1) * 100), 3),
            "ann_return_pct": st["ann_return_pct"], "max_dd_pct": st["max_dd_pct"],
            "calmar": st["calmar"], "omega": omega_ratio(r),
            "buy_hold_pct": round(float((np.prod(1 + a) - 1) * 100), 3),
            "mean_exposure": round(float(np.nanmean(np.abs(pos))), 4),
        }
    return out


def main() -> int:
    rows = []
    for name in STRATEGIES:
        try:
            r = analyse(name)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:22} ERROR: {e}")
            continue
        if r:
            rows.append(r)

    print("=" * 104)
    print("METRICAS POR VENTANA — metodologia CLAUDE.md: entrenado <=2024 | 2025 OOS | 2026 produccion")
    print("=" * 104)
    for w in WINDOWS:
        print(f"\n### {w}")
        print(f"{'estrategia':24}{'N':>5}{'trades':>8}{'ret%':>10}{'ann%':>9}{'MaxDD%':>9}"
              f"{'Calmar':>8}{'B&H%':>10}{'expo':>7}")
        print("-" * 90)
        for r in rows:
            d = r["windows"].get(w, {})
            if "net_return_pct" not in d:
                print(f"{r['strategy_id']:24}{d.get('n_periods', 0):>5}{'':>8}"
                      f"{d.get('status', 'n/d'):>10}")
                continue
            print(f"{r['strategy_id']:24}{d['n_periods']:>5}{d['n_trades']:>8}"
                  f"{d['net_return_pct']:>10}{str(d['ann_return_pct']):>9}"
                  f"{str(d['max_dd_pct']):>9}{str(d['calmar']):>8}{d['buy_hold_pct']:>10}"
                  f"{d['mean_exposure']:>7}")

    out = REPO / ".claude" / "evidence" / "windows" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "window_report.json").write_text(json.dumps({
        "methodology": ("CLAUDE.md: trained <= Dec-2024, 2025 = backtest OOS, 2026 = production. "
                        "DESIGN is not evidence: the strategy was built looking at it."),
        "windows": {k: {"from": v[0], "to": v[1]} for k, v in WINDOWS.items()},
        "strategies": rows, "evidence_class": "research_only",
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nartefacto -> {out / 'window_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
