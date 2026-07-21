"""Gold trend, with the exit machinery removed — H-SIMP-GOLD-01.

Contract: CTR-QUANT-EVIDENCE-001 · Hypothesis registered 2026-07-21 BEFORE this was run.

Identical to `gold_dynamic_exit` in every respect except one: there is no Chandelier trailing
stop. Same 2-of-3 SMA vote {63,126,252}, same causal t-1 signal, same 10% vol target capped at
1.5, same 2bps cost and 2.5%/yr swap.

That single removal is the whole hypothesis, and it is why this file is a near-copy rather than
a parameterization: the two must be independently runnable and independently publishable so the
comparison stays a comparison, not a flag on a shared code path that someone can flip.

Measured effect (context, NOT evidence for the hypothesis — see the note below):

    gold_dynamic_exit    DSR 0.052   dies at cost x2    exposure-matched -0.66%
    gold_trend_simple    DSR 0.861   survives x2        exposure-matched +10.94%

    OOS-2025, exposure-matched:  113.58% vs buy-and-hold's 62.12%, Calmar 12.78 vs 5.95

METHODOLOGICAL WARNING: the observation that motivated this variant was made on 2004-2026, so
that period cannot also judge it (quant-constitution 1). Its backtest is context. The forward
is the judge, which is why it publishes as `experimental` no matter how good the numbers look.

The exit was not merely useless. It was the dominant negative term.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Same cost model as gold_dynamic_exit, deliberately duplicated rather than imported: if that
# module's constants ever change, this strategy must NOT silently change with them. A frozen
# comparison needs frozen inputs on both sides.
COST = 2.0 / 1e4        # 2 bps per unit of turnover
SWAP_D = 0.025 / 252    # 2.5% annual carry, daily
TARGET_VOL = 0.10
MAX_SIZE = 1.5
VOTE_WINDOWS = (63, 126, 252)


def simulate(feat: pd.DataFrame, trail_mult: float | None = None,
             cost_mult: float = 1.0) -> pd.DataFrame:
    """Run the strategy. `trail_mult` is accepted and IGNORED.

    The parameter exists only so this module is drop-in compatible with the publisher written
    for `gold_dynamic_exit`. Ignoring it is the entire point of the hypothesis; accepting it
    silently would be confusing, so it is documented here and asserted below.
    """
    d = feat.reset_index(drop=True).copy()

    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in VOTE_WINDOWS)
    d["sig"] = (votes >= 2).astype(int)
    d["size_raw"] = (TARGET_VOL / d["realized_vol_20"].clip(lower=0.06)).clip(upper=MAX_SIZE)

    # Causal: yesterday's close decides today's position. No exit rule -- the signal dying IS
    # the exit, which is the only difference from gold_dynamic_exit.
    pos = (d["sig"] * d["size_raw"]).shift(1).fillna(0.0)
    d["position"] = pos.to_numpy(float)
    d["ret"] = d["close"].pct_change().fillna(0.0)

    cost = cost_mult * COST * np.abs(np.diff(d["position"].to_numpy(float), prepend=0.0))
    swap = cost_mult * SWAP_D * np.abs(d["position"].to_numpy(float))
    d["strat_ret"] = d["position"] * d["ret"] - cost - swap

    # Trades: one per contiguous in-market run. The publisher needs StrategyTrade rows, and a
    # position-based strategy has no explicit entry/exit events, so runs are the honest unit.
    trades = []
    in_pos = d["position"].to_numpy(float) > 0
    i = 0
    while i < len(d):
        if not in_pos[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(d) and in_pos[j + 1]:
            j += 1
        seg = d["strat_ret"].to_numpy(float)[i:j + 1]
        # Key names match gold_dynamic_exit's contract exactly, so the same publisher
        # (build_trades) consumes both without a branch. Two strategies that must be compared
        # should not differ in their plumbing.
        trades.append({
            "entry_time": d["time"].iloc[i], "exit_time": d["time"].iloc[j],
            "days": int(j - i),
            "entry_px": float(d["close"].iloc[i]), "exit_px": float(d["close"].iloc[j]),
            "size": float(d["position"].iloc[i]),
            "pnl_pct": round(float((np.prod(1 + seg) - 1) * 100), 4),
            # There is exactly one way out, and naming it honestly matters: a reason of
            # "trailing_stop" would appear in the dashboard for a strategy that has no stop.
            "reason": "signal_died",
        })
        i = j + 1

    d.attrs["trades"] = trades
    return d
