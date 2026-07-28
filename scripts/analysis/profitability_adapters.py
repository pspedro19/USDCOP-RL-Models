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

from scripts.analysis.profitability_types import Sleeve  # noqa: E402


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

    # ACTIVE version from the registry, not the last one alphabetically. Sorting directory
    # names picked "3.0.0-B", which has no trades_2026.json, so COP silently lost its entire
    # LIVE window while the registry said active_version = 2.0.0 with backtest_years
    # [2025, 2026]. "Newest name" and "the one we actually run" are different questions.
    import json as _j
    reg = _j.loads((ROOT / "usdcop-trading-dashboard/public/data/registry.json")
                   .read_text(encoding="utf-8"))
    version = next((r.get("active_version") for r in reg["strategies"]
                    if r.get("strategy_id") == "smart_simple_v11"), None)
    if not version:
        raise RuntimeError("COP: registry declares no active_version for smart_simple_v11.")
    # Concatenate every published year so COP has the same DESIGN/OOS/LIVE windows as the
    # other assets. Previously this loaded 2025 only, so COP was absent from every 2026 report
    # while its production bundle showed +1.77% -- present in one place, missing in another.
    parts = []
    for yr in (2025, 2026):
        y = sleeve_weekly("smart_simple_v11", version, yr)
        if y is not None and len(y):
            parts.append(y)
    if not parts:
        raise RuntimeError(f"COP: bundle {version} produced no weekly sleeve for 2025 or 2026.")
    s = pd.concat(parts).sort_index()
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


# ---------------------------------------------------------------------------
# SPX500 — regime-gated, on REAL SPY total-return data
# ---------------------------------------------------------------------------

def spx500() -> Sleeve:
    """Runs the engine on the real SPY snapshot, not `datagen.generate()`.

    The strategy shipped wired to a synthetic series whose own docstring says
    "NO evidencia de alfa", so its published metrics measured the wiring. `load_real` supplies
    the same column contract from the actual snapshot, leaving regimes/policies/gates untouched.

    Pinned to the CENTRAL prior of the declared family (vol target 0.10, MA 200) — never the
    best of the 4x3 sweep in `_config_family`. That sweep exists to feed PBO, which is exactly
    the measure of what picking its winner would cost.
    """
    # The SPX500 engine was vendored wholesale and its modules import each other flatly
    # (`import costs`, not `from . import costs`), so its own directory must be on sys.path.
    pkg = ROOT / "src" / "strategies" / "spx500_regime_gated_v1"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    from src.strategies.spx500_regime_gated_v1.engine import BacktestConfig, BacktestEngine
    from src.strategies.spx500_regime_gated_v1.load_real import load_real
    from src.strategies.spx500_regime_gated_v1.policies import spx_regime_gated_v1

    df = load_real()
    close = df["close"].astype(float)

    # Use the ACTUAL gated policy. The first version of this adapter built
    # `trend_on * vol_target_weights` by hand, which is the SIMPLE strategy -- it never
    # invoked regime.py at all. It then scored identically to the deliberately-simplified
    # variant (Calmar 0.577 vs 0.578), which is what exposed the mistake. Measuring a
    # strategy that is not the strategy is worse than not measuring it: every conclusion
    # drawn about "the regime gate" would have been about code that never ran.
    w = spx_regime_gated_v1(df)
    trend_on = (close > close.rolling(200, min_periods=200).mean()).astype(float)

    res = BacktestEngine(BacktestConfig(cost_bps_roundtrip=3.0)).run(w, df)

    pos = res.weights_exec.to_numpy(float)
    ret = df["open_to_open_return"].to_numpy(float)[: len(pos)]
    cost = res.cost.to_numpy(float)
    n_trades = int((np.abs(np.diff(pos, prepend=0.0)) > 1e-9).sum())

    return Sleeve(
        asset="spx500", strategy_id="spx500_regime_gated_v1",
        index=df["timestamp"].iloc[: len(pos)],
        position=pos, asset_ret=ret, cost=cost, swap=None,
        n_trades=n_trades, clock=252, clock_label="daily/252",
        # An equity index trends up over almost any long window, so the honest dumb baseline
        # is the trend filter with no vol targeting and no gating at all.
        dumb_name="ma200_always_on", dumb_position=trend_on.to_numpy(float)[: len(pos)],
    )


ADAPTERS = {"usdcop": cop, "xauusd": gold, "btcusdt": btc, "spx500": spx500}


# ---------------------------------------------------------------------------
# Simplification hypotheses — H-SIMP-GOLD-01 / H-SIMP-SPX-01
# ---------------------------------------------------------------------------
# These are the strategies with the machinery REMOVED. Registered in the hypothesis
# registries on 2026-07-21 BEFORE being run. Their historical numbers are context only:
# the observation that motivated them was made on the same history, so that history cannot
# also judge them (quant-constitution 1). The forward is the judge.

def gold_simple() -> Sleeve:
    """H-SIMP-GOLD-01: same 2-of-3 SMA vote, vol-targeted, NO trailing exit."""
    from src.gold_rl.indicators import build_daily_features

    df = pd.read_parquet(ROOT / "seeds/latest/xauusd_daily_ohlcv.parquet")
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    d = build_daily_features(df).reset_index(drop=True)

    votes = sum((d["close"] > d["close"].rolling(w).mean()).astype(int) for w in (63, 126, 252))
    sig = (votes >= 2).astype(float)
    size = (0.10 / d["realized_vol_20"].clip(lower=0.06)).clip(upper=1.5)
    pos = (sig * size).shift(1).fillna(0.0).to_numpy(float)   # causal: yesterday's close

    ret = d["close"].pct_change().fillna(0.0).to_numpy(float)
    COST, SWAP_D = 2.0 / 1e4, 0.025 / 252                      # identical to gold_dynamic_exit
    cost = np.abs(np.diff(pos, prepend=0.0)) * COST
    swap = np.abs(pos) * SWAP_D
    n_trades = int((np.abs(np.diff((pos > 0).astype(float), prepend=0.0)) > 0).sum())

    return Sleeve(
        asset="xauusd", strategy_id="gold_trend_simple",
        index=d["time"], position=pos, asset_ret=ret, cost=cost, swap=swap,
        n_trades=n_trades, clock=252, clock_label="daily/252",
        dumb_name="sma_vote_unsized", dumb_position=sig.shift(1).fillna(0.0).to_numpy(float),
    )


def spx500_simple() -> Sleeve:
    """H-SIMP-SPX-01: MA200 + vol target, NO regime gate."""
    pkg = ROOT / "src" / "strategies" / "spx500_regime_gated_v1"
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    from src.strategies.spx500_regime_gated_v1.engine import BacktestConfig, BacktestEngine
    from src.strategies.spx500_regime_gated_v1.load_real import load_real
    from src.strategies.spx500_regime_gated_v1.policies import vol_target_weights

    df = load_real()
    close = df["close"].astype(float)
    trend_on = (close > close.rolling(200, min_periods=200).mean()).astype(float)
    w = (trend_on * vol_target_weights(close, target=0.10)).clip(upper=1.5)

    res = BacktestEngine(BacktestConfig(cost_bps_roundtrip=3.0)).run(w, df)
    pos = res.weights_exec.to_numpy(float)
    ret = df["open_to_open_return"].to_numpy(float)[: len(pos)]

    return Sleeve(
        asset="spx500", strategy_id="spx500_trend_simple",
        index=df["timestamp"].iloc[: len(pos)],
        position=pos, asset_ret=ret, cost=res.cost.to_numpy(float), swap=None,
        n_trades=int((np.abs(np.diff(pos, prepend=0.0)) > 1e-9).sum()),
        clock=252, clock_label="daily/252",
        dumb_name="ma200_always_on", dumb_position=trend_on.to_numpy(float)[: len(pos)],
    )


ADAPTERS["xauusd_simple"] = gold_simple
ADAPTERS["spx500_simple"] = spx500_simple


def btc_hodl() -> Sleeve:
    """btc_hodl_b1 — the OOS champion (constitution: the baseline IS the strategy).

    Exists because the forward tracker resolves a replay series for each CHAMPION, and the
    asset-keyed `btc()` adapter returns btc_trend_b2 — the strategy the champion replaced.
    Computing divergence between hodl's paper and trend_b2's replay would compare two
    different strategies and call the difference "tracking error".
    """
    from src.btc_strategy.backtest import compute_returns, extract_trades
    from src.btc_strategy.indicators import build_daily_features
    from src.btc_strategy.strategies import STRATEGIES, build_positions

    df = pd.read_parquet(ROOT / "seeds/latest/btcusdt_daily_ohlcv.parquet")
    df = df.sort_values("time").reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    feat = build_daily_features(df)

    _, intent_fn, _ = STRATEGIES["btc_hodl_b1"]
    d = compute_returns(build_positions(feat, intent_fn))
    pos = d["position"].to_numpy(float)

    return Sleeve(
        asset="btcusdt", strategy_id="btc_hodl_b1",
        index=d["time"] if "time" in d else d.index,
        position=pos, asset_ret=d["ret"].to_numpy(float),
        cost=d["cost"].to_numpy(float), swap=d["swap"].to_numpy(float),
        n_trades=len(extract_trades(d)), clock=365, clock_label="daily/365",
        dumb_name="flat", dumb_position=np.zeros_like(pos),
    )


# Strategy-keyed entries: the tracker resolves by strategy_id FIRST, so a champion change
# never silently points the replay at the strategy it replaced.
ADAPTERS["btc_hodl_b1"] = btc_hodl
ADAPTERS["gold_trend_simple"] = gold_simple
ADAPTERS["spx500_regime_gated_v1"] = spx500
