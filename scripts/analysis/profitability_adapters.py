"""Asset adapters for the profitability harness — thin wrappers, zero re-derivation.

Contract: CTR-QUANT-EVIDENCE-001

Each adapter's only job is to hand back a `Sleeve` built from the SAME code that produced the
published bundles. If an adapter reimplements a strategy, the evidence stops describing what
the system actually does, so every one of these imports rather than copies.

Each adapter is pinned to its PRE-REGISTERED parameters — gold to its M=3.0 prior, never the
best of {2.0, 3.0, 4.0}. Picking the winning sensitivity cell here would be a grid search over
the test set with extra steps.

spx500 has no adapter on purpose: that track belongs to another agent.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.profitability_evidence import Sleeve  # noqa: E402


def _load_script(path: Path, name: str):
    """Import a `scripts/` entrypoint as a module (the pattern run_btc_pipeline.py already uses)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# XAU/USD — gold_dynamic_exit at its prior cell
# ---------------------------------------------------------------------------

def gold() -> Sleeve:
    from src.gold_rl.indicators import build_daily_features

    mod = _load_script(ROOT / "scripts/analysis/gold_dynamic_exit.py", "gold_dynamic_exit")
    df = pd.read_parquet(ROOT / "seeds/latest/xauusd_daily_ohlcv.parquet")
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    d = mod.simulate(feat, 3.0)          # PRIOR cell. Not tuned. Not the best of the sweep.
    pos = d["position"].to_numpy(float)
    ret = d["ret"].to_numpy(float)
    # simulate() nets costs inside strat_ret; recover them so cost_stress can re-price x2/x3.
    cost = pos * ret - d["strat_ret"].to_numpy(float)

    # Dumb baseline for a trend track: the same 2-of-3 SMA vote, always on, no exit logic.
    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    dumb = (votes >= 2).astype(float).to_numpy()

    return Sleeve(
        asset="xauusd", strategy_id="gold_dynamic_exit",
        index=d["time"] if "time" in d else d.index,
        position=pos, asset_ret=ret, cost=np.abs(cost), swap=None,
        n_trades=len(d.attrs.get("trades", [])),
        clock=252, clock_label="daily/252",
        dumb_name="sma_vote_always_on", dumb_position=dumb,
    )


# ---------------------------------------------------------------------------
# BTC/USDT — btc_trend_b2
# ---------------------------------------------------------------------------

def btc() -> Sleeve:
    from src.btc_strategy.backtest import compute_returns, extract_trades
    from src.btc_strategy.indicators import build_daily_features
    from src.btc_strategy.strategies import STRATEGIES, build_positions

    df = pd.read_parquet(ROOT / "seeds/latest/btcusdt_daily_ohlcv.parquet")
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    _, intent_fn, _ = STRATEGIES["btc_trend_b2"]
    d = compute_returns(build_positions(feat, intent_fn))

    pos = d["position"].to_numpy(float)
    ret = d["ret"].to_numpy(float)
    cost = d["cost"].to_numpy(float)
    swap = d["swap"].to_numpy(float)
    n_trades = len(extract_trades(d))

    return Sleeve(
        asset="btcusdt", strategy_id="btc_trend_b2",
        index=d["time"] if "time" in d else d.index,
        position=pos, asset_ret=ret, cost=cost, swap=swap,
        n_trades=n_trades, clock=365, clock_label="daily/365",
        dumb_name="hodl_vol_targeted",
        dumb_position=np.full_like(pos, float(np.nanmean(np.abs(pos)))),
    )


# ---------------------------------------------------------------------------
# USD/COP — smart_simple_v11, weekly clock
# ---------------------------------------------------------------------------

def cop() -> Sleeve:
    """COP runs on a WEEKLY clock (52), not 261.

    `portfolio_layer.py:33` and `cop_trials_dsr.py` both annualize COP with 52 because the
    strategy takes at most one position per week. Using the asset profile's 261 trading days
    here would inflate every annualized number by ~sqrt(5).
    """
    from scripts.analysis.portfolio_layer import sleeve_weekly

    # Use the LATEST published version, resolved from disk — not a literal, which would go
    # stale silently the next time a bundle is published.
    vdir = ROOT / "usdcop-trading-dashboard/public/data/strategies/smart_simple_v11/backtests"
    versions = sorted(p.name for p in vdir.iterdir() if p.is_dir()) if vdir.is_dir() else []
    if not versions:
        raise RuntimeError(
            f"COP: no published bundle under {vdir}. The harness will not fabricate a series."
        )
    version = versions[-1]
    s = sleeve_weekly("smart_simple_v11", version, 2025)   # sleeve_weekly slices OOS-2025
    if s is None or not len(s):
        raise RuntimeError(f"COP: bundle {version} produced an empty weekly sleeve.")
    ret = s.to_numpy(float)

    # v11 is flat unless the regime gate opens; a non-zero week is a trade.
    pos = (ret != 0).astype(float)
    n_trades = int((ret != 0).sum())

    # HONESTY CAVEAT, surfaced rather than hidden:
    # The published bundle stores the STRATEGY's net weekly P&L, not the underlying USD/COP
    # price series. So `asset_ret` here is already the strategy's own return, and with
    # position=1 on every active week, B1 "buy & hold" collapses to the strategy itself
    # (identical numbers) and B1' is only a rescaling of it. Those two baselines are therefore
    # NOT VALID for COP from this data source and must be read as such.
    # Computing them anyway and reporting the tie as "matches buy & hold" would be a fabricated
    # comparison. Fixing it properly requires joining the weekly COP price series — tracked as
    # a follow-up, not silently patched here.

    return Sleeve(
        asset="usdcop", strategy_id="smart_simple_v11",
        index=s.index, position=pos, asset_ret=ret,
        cost=np.zeros_like(ret),   # bundle returns are already net of costs
        swap=None, n_trades=n_trades,
        clock=52, clock_label="weekly/52",
        dumb_name="always_short_1x", dumb_position=np.full_like(ret, -1.0),
        invalid_baselines=("B1_buy_and_hold", "B1_prime_exposure_matched"),
    )


ADAPTERS = {"usdcop": cop, "xauusd": gold, "btcusdt": btc}
