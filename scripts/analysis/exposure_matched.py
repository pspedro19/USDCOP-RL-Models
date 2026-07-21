"""Exposure-matched comparison — separates timing from simply holding less.

Contract: CTR-QUANT-EXPMATCH-001

A strategy that is flat most of the time will show a better Sharpe and a smaller drawdown than
buy-and-hold, and neither fact says anything about skill: it is holding less. `B1'` in the
constitution exists to catch exactly that, by comparing against a CONSTANT exposure equal to
the strategy's realized average.

This module asks the mirror-image question, which is the one that matters for deciding whether
to trade something: scaled up to the SAME average exposure as buy-and-hold, does the strategy
still win? If yes, the timing is real. If the advantage evaporates, it was only ever less beta.

Scaling multiplies costs and swap by the same factor, so the comparison stays honest -- a
levered version pays levered costs.

WHAT THIS IS NOT: a trading recommendation. BTC's realized exposure is ~0.16, so matching
buy-and-hold means ~6.2x leverage and an ~82% drawdown. Nobody should trade that. The scaled
figures answer "is the timing real", not "size it this way". Sizing is governed by
`bet-sizing` (quarter Kelly, capped) and the risk rules, not by this comparison.

Run: python -m scripts.analysis.exposure_matched
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

from services.common.metrics import (  # noqa: E402
    _ann_return_dd_calmar, capture_ratios, omega_ratio, sharpe_ratio_stderr,
)

ASSETS = ("btcusdt", "xauusd", "xauusd_simple", "spx500", "spx500_simple")


OOS_YEAR = 2025   # the one truly held-out year; declared, not chosen after seeing results


def _oos_mask(index) -> np.ndarray:
    """Boolean mask for the held-out year.

    `Index.year == 2025` already returns a numpy ndarray, not an Index, so calling `.to_numpy()`
    on it raises AttributeError. The first version wrapped that in a bare `except` and returned
    an all-False mask, which made every asset silently report OOS `n/a` -- a swallowed exception
    turning the most important column in the table into a blank. Raising here instead: if the
    index cannot be interpreted as dates, that is a broken adapter, not a missing window.
    """
    idx = pd.to_datetime(pd.Index(index))
    return np.asarray(idx.year == OOS_YEAR, dtype=bool)


def analyse(name: str) -> dict | None:
    from scripts.analysis.profitability_adapters import ADAPTERS

    s = ADAPTERS[name]()
    exp = float(np.nanmean(np.abs(s.position)))
    if exp <= 1e-6:
        return None

    k = 1.0 / exp                       # scale to average exposure 1.0 == B1's exposure
    scaled = k * s.position * s.asset_ret - k * s.cost - k * s.swap

    # THE CORRECTION THAT MATTERS (2026-07-21). The first version of this module reported only
    # full-history figures, and I read BTC's "172% vs 36%, Calmar 2.101" as proof of timing
    # skill. It is not: that window includes 2018-2024, where the strategy was fitted. In the
    # single held-out year every BTC rung was NEGATIVE while plain HODL made +4.70%.
    #
    # An edge that appears only before the hold-out is not an edge, it is a memory. Reporting
    # the two windows side by side is the whole point -- a full-history number alone is the
    # most persuasive misleading statistic this codebase can produce.
    m = _oos_mask(s.index)
    oos = None
    if m.sum() >= 20:
        oos = {
            "year": OOS_YEAR, "n_obs": int(m.sum()),
            "as_is": _ann_return_dd_calmar(s.strat_ret[m], s.clock),
            "exposure_matched": _ann_return_dd_calmar(scaled[m], s.clock),
            "b1_buy_and_hold": _ann_return_dd_calmar(s.asset_ret[m], s.clock),
        }
        o_m, o_b = oos["exposure_matched"], oos["b1_buy_and_hold"]
        oos["timing_is_real_oos"] = bool(o_m["calmar"] > o_b["calmar"]
                                         and o_m["ann_return_pct"] > o_b["ann_return_pct"])

    out = {
        "oos_window": oos,
        "asset": s.asset, "strategy_id": s.strategy_id, "clock": s.clock,
        "mean_realized_exposure": round(exp, 4), "scale_factor": round(k, 3),
        "as_is": _ann_return_dd_calmar(s.strat_ret, s.clock),
        "exposure_matched": _ann_return_dd_calmar(scaled, s.clock),
        "b1_buy_and_hold": _ann_return_dd_calmar(s.asset_ret, s.clock),
        "omega_matched": omega_ratio(scaled),
        "omega_b1": omega_ratio(s.asset_ret),
        "capture_matched": capture_ratios(scaled, s.asset_ret),
        "sharpe_stderr_matched": round(float(sharpe_ratio_stderr(scaled)), 5),
        "leverage_warning": (
            f"matching B1 requires {k:.2f}x leverage and a "
            f"{_ann_return_dd_calmar(scaled, s.clock)['max_dd_pct']}% drawdown -- this is a "
            "diagnostic of timing, NOT a position size"
        ),
    }
    m, b = out["exposure_matched"], out["b1_buy_and_hold"]
    out["verdict"] = {
        "calmar_beats_b1": bool(m["calmar"] > b["calmar"]),
        "return_beats_b1": bool(m["ann_return_pct"] > b["ann_return_pct"]),
        "dd_no_worse_than_b1": bool(m["max_dd_pct"] >= b["max_dd_pct"]),  # both negative
        "timing_is_real_full_history": bool(m["calmar"] > b["calmar"]
                                            and m["ann_return_pct"] > b["ann_return_pct"]),
    }
    # The verdict that counts is the OOS one. When the two disagree, the full-history number is
    # the one to distrust: it is the window the strategy was built on.
    if out["oos_window"] is not None:
        out["verdict"]["timing_is_real_OOS"] = out["oos_window"]["timing_is_real_oos"]
        out["verdict"]["disagrees_with_full_history"] = bool(
            out["oos_window"]["timing_is_real_oos"] != out["verdict"]["timing_is_real_full_history"])
    return out


def main() -> int:
    print("=" * 92)
    print("COMPARACION A EXPOSICION EMPAREJADA (¿timing real o solo menos beta?)")
    print("=" * 92)
    print(f"{'estrategia':26} {'exp':>6} {'k':>6} | {'ann% B1':>9} {'ann% match':>11} "
          f"{'Calmar B1':>10} {'Calmar match':>13} {'hist':>8} {'OOS-2025':>8}")

    results = []
    for a in ASSETS:
        try:
            r = analyse(a)
        except Exception as e:  # noqa: BLE001
            print(f"  {a:24} ERROR: {e}")
            continue
        if r is None:
            continue
        results.append(r)
        m, b = r["exposure_matched"], r["b1_buy_and_hold"]
        print(f"{r['strategy_id']:26} {r['mean_realized_exposure']:>6} {r['scale_factor']:>6} | "
              f"{b['ann_return_pct']:>9} {m['ann_return_pct']:>11} {b['calmar']:>10} "
              f"{m['calmar']:>13} {'SI' if r['verdict']['timing_is_real_full_history'] else 'no':>8}"
              f" {('SI' if r['verdict'].get('timing_is_real_OOS') else ('no' if r['oos_window'] else 'n/a')):>8}")

    out = REPO / ".claude" / "evidence" / "exposure_matched" / date.today().isoformat()
    out.mkdir(parents=True, exist_ok=True)
    (out / "exposure_matched.json").write_text(json.dumps({
        "results": results, "evidence_class": "research_only", "promotion_eligible": False,
        "note": ("Scaled figures diagnose whether timing is real at equal beta. They are NOT "
                 "position sizes: see leverage_warning per asset."),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nartefacto -> {out / 'exposure_matched.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
