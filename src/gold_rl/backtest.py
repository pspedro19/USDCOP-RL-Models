"""SPEC-09 backtest: metrics, trades, gates, regime attribution, bootstrap CIs.

Daily backtest with a realistic cost model (transaction cost on turnover + overnight swap on held
exposure, STRATEGY §8). Produces dashboard-ready summary + trades dicts for the registry publisher,
and honest statistics (block-bootstrap p-value, DSR-style trial awareness left to the caller).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Shared, trial-aware stats (single home — never re-implement PSR/DSR here; see
# services/common/metrics.py, the same "define once, import everywhere" pattern as safe_json_dump).
try:
    from services.common.metrics import trial_aware_moments
except Exception:  # pragma: no cover - keeps the module importable if services/ isn't on path
    trial_aware_moments = None

TRADING_DAYS = 252
COST_BPS = 2.0          # transaction cost per unit turnover (spread+slippage), bps of notional
SWAP_ANNUAL = 0.025     # overnight financing drag on held exposure (~2.5%/yr, STRATEGY §8)


# --------------------------------------------------------------------------- returns / equity
def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ret"] = d["close"].pct_change().fillna(0.0)
    turnover = d["position"].diff().abs().fillna(d["position"].abs())
    d["cost"] = turnover * (COST_BPS / 1e4)
    d["swap"] = d["position"].abs() * (SWAP_ANNUAL / TRADING_DAYS)
    d["strat_ret"] = d["position"] * d["ret"] - d["cost"] - d["swap"]
    d["equity"] = (1.0 + d["strat_ret"]).cumprod()
    return d


# --------------------------------------------------------------------------- metrics
def _max_dd(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min())


def metrics(d: pd.DataFrame) -> dict:
    r = d["strat_ret"].dropna()
    if r.empty or r.std() == 0:
        return {"cagr": 0, "sharpe": 0, "sortino": 0, "max_dd": 0, "calmar": 0,
                "total_return_pct": 0, "vol_ann": 0}
    n = len(r)
    total = float(d["equity"].iloc[-1] / d["equity"].iloc[0] - 1.0) if len(d) else 0.0
    years = n / TRADING_DAYS
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and (1 + total) > 0 else 0.0
    sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))
    downside = r[r < 0].std()
    sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
    mdd = _max_dd(d["equity"])
    calmar = float(cagr / abs(mdd)) if mdd < 0 else 0.0
    return {"cagr": round(cagr * 100, 2), "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
            "max_dd": round(mdd * 100, 2), "calmar": round(calmar, 3),
            "total_return_pct": round(total * 100, 2),
            "vol_ann": round(float(r.std() * np.sqrt(TRADING_DAYS)) * 100, 2)}


def block_bootstrap_pvalue(r: pd.Series, *, block: int = 20, n_boot: int = 5000,
                           seed: int = 42) -> dict:
    """Moving-block bootstrap on daily returns -> 95% CI of annualized return + p-value.

    Reproducible via a FIXED numpy RNG seed (deterministic across runs). The p-value is the
    bootstrap probability that the mean daily return is <= 0 (one-sided H0: no positive edge);
    block=20 preserves ~monthly autocorrelation so the CI isn't falsely tight.
    """
    r = r.dropna().values
    n = len(r)
    if n < block * 3:
        return {"p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0, "significant": False}
    rng = np.random.default_rng(seed)
    n_blocks = n // block + 1
    max_start = n - block
    means = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([r[s:s + block] for s in starts])[:n]
        means[b] = sample.mean()
    ann = means * TRADING_DAYS
    p = float((means <= 0).mean())
    return {"p_value": round(p, 4), "ci_low": round(float(np.percentile(ann, 2.5)) * 100, 2),
            "ci_high": round(float(np.percentile(ann, 97.5)) * 100, 2),
            "significant": bool(p < 0.05)}


# --------------------------------------------------------------------------- trades segmentation
def extract_trades(d: pd.DataFrame, *, initial_capital: float = 10000.0) -> list[dict]:
    """Segment the position series into discrete trades (entry when it becomes non-zero / flips,
    exit when it returns to zero / flips). Produces the universal StrategyTrade schema."""
    pos = d["position"].values
    trades = []
    equity = initial_capital
    i, n, tid = 0, len(d), 0
    while i < n:
        if abs(pos[i]) < 1e-9:
            i += 1
            continue
        sign = np.sign(pos[i])
        j = i
        seg_ret = 1.0
        while j < n and np.sign(pos[j]) == sign and abs(pos[j]) > 1e-9:
            seg_ret *= (1.0 + d["strat_ret"].iloc[j])
            j += 1
        pnl_pct = (seg_ret - 1.0) * 100
        entry = d.iloc[i]
        exit_ = d.iloc[j - 1]
        eq_in = equity
        equity *= seg_ret
        tid += 1
        trades.append({
            "trade_id": tid,
            "timestamp": entry["time"].isoformat(),
            "exit_timestamp": exit_["time"].isoformat(),
            "side": "LONG" if sign > 0 else "SHORT",
            "entry_price": round(float(entry["close"]), 2),
            "exit_price": round(float(exit_["close"]), 2),
            "pnl_pct": round(float(pnl_pct), 4),
            "pnl_usd": round(float(eq_in * (seg_ret - 1.0)), 2),
            "equity_at_entry": round(float(eq_in), 2),
            "equity_at_exit": round(float(equity), 2),
            "leverage": round(float(abs(entry["position"])), 3),
            "exit_reason": "regime_flip" if j < n else "week_end",
            "regime": str(entry.get("regime", "")),
        })
        i = j
    return trades


def regime_attribution(d: pd.DataFrame) -> dict:
    """PnL / Sharpe / share per regime — the single most informative diagnostic (SPEC-09)."""
    out = {}
    for reg, g in d.groupby("regime"):
        r = g["strat_ret"].dropna()
        if r.empty:
            continue
        out[str(reg)] = {
            "days": int(len(g)),
            "pnl_pct": round(float(((1 + r).prod() - 1) * 100), 2),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)), 3) if r.std() else 0.0,
            "share_pct": round(100 * len(g) / len(d), 1),
        }
    return out


# --------------------------------------------------------------------------- gates
def evaluate_gates(m: dict, n_trades: int, boot: dict) -> list[dict]:
    """The 5 approval gates (mirror sdd-approval-spec) bound to the OOS backtest = Vote 1/2."""
    return [
        {"gate": "min_return_pct", "label": "Retorno Minimo", "value": m["total_return_pct"],
         "threshold": -15.0, "passed": m["total_return_pct"] > -15.0},
        {"gate": "min_sharpe_ratio", "label": "Sharpe Minimo", "value": m["sharpe"],
         "threshold": 0.0, "passed": m["sharpe"] > 0.0},
        {"gate": "max_drawdown_pct", "label": "Max Drawdown", "value": abs(m["max_dd"]),
         "threshold": 20.0, "passed": abs(m["max_dd"]) < 20.0},
        {"gate": "min_trades", "label": "Trades Minimos", "value": n_trades,
         "threshold": 10, "passed": n_trades >= 10},
        {"gate": "statistical_significance", "label": "Significancia (p<0.05)",
         "value": boot["p_value"], "threshold": 0.05, "passed": boot["significant"]},
    ]


def recommendation(gates: list[dict]) -> tuple[str, float]:
    passed = sum(g["passed"] for g in gates)
    conf = passed / len(gates)
    critical_fail = not gates[0]["passed"] or not gates[2]["passed"]  # return or drawdown
    if passed == len(gates):
        return "PROMOTE", conf
    if critical_fail:
        return "REJECT", conf
    return "REVIEW", conf


# --------------------------------------------------------------------------- full run
def _oos_slice_stats(d: pd.DataFrame, oos_year: int) -> dict | None:
    """Metrics + bootstrap on a single held-out calendar year (true OOS gate).

    The rule constants are fixed a-priori, but the honest split still isolates the year that was NOT
    used to eyeball thresholds. Reports the OOS year on its own so the gates aren't computed on the
    same span used to design the strategy (metrics-agent gap: gold/btc were full-history in-sample).
    """
    yr = d[d["time"].dt.year == oos_year]
    if len(yr) < 20:
        return None
    mo = metrics(yr)
    bo = block_bootstrap_pvalue(yr["strat_ret"])
    out = {"year": oos_year, "n_trading_days": int(len(yr)), "metrics": mo,
           "statistical_tests": {"p_value": bo["p_value"], "significant": bo["significant"],
                                 "bootstrap_95ci_ann": [bo["ci_low"], bo["ci_high"]]}}
    if trial_aware_moments is not None:
        out["statistical_tests"].update(trial_aware_moments(yr["strat_ret"].values))
    return out


def run_backtest(df_positions: pd.DataFrame, strategy_id: str, strategy_name: str, *,
                 year: int, bh_df: pd.DataFrame | None = None,
                 initial_capital: float = 10000.0, oos_year: int | None = None) -> dict:
    """Full backtest -> {summary, trades, metrics, attribution, gates} ready to publish.

    ``oos_year`` (e.g. 2025) additionally reports metrics/stats on that held-out year alone.
    """
    d = compute_returns(df_positions)
    m = metrics(d)
    boot = block_bootstrap_pvalue(d["strat_ret"])
    trades = extract_trades(d, initial_capital=initial_capital)
    attrib = regime_attribution(d)
    gates = evaluate_gates(m, len(trades), boot)
    rec, conf = recommendation(gates)

    n_long = sum(1 for t in trades if t["side"] == "LONG")
    n_short = sum(1 for t in trades if t["side"] == "SHORT")
    wins = [t for t in trades if t["pnl_pct"] > 0]
    gross_win = sum(t["pnl_pct"] for t in wins)
    gross_loss = abs(sum(t["pnl_pct"] for t in trades if t["pnl_pct"] < 0))
    pf = round(gross_win / gross_loss, 3) if gross_loss > 0 else None
    final_eq = initial_capital * float(d["equity"].iloc[-1])

    strat_stats = {
        "final_equity": round(final_eq, 2), "total_return_pct": m["total_return_pct"],
        "sharpe": m["sharpe"], "sortino": m["sortino"], "calmar": m["calmar"],
        "max_dd_pct": m["max_dd"], "win_rate_pct": round(100 * len(wins) / len(trades), 1) if trades else 0.0,
        "profit_factor": pf, "n_long": n_long, "n_short": n_short,
        "trading_days": int(len(d)),
        "exit_reasons": {r: sum(1 for t in trades if t["exit_reason"] == r)
                         for r in {t["exit_reason"] for t in trades}},
        "regime_attribution": attrib,
    }
    strategies = {strategy_id: strat_stats}
    if bh_df is not None and not bh_df.empty:
        bh_total = float(bh_df["close"].iloc[-1] / bh_df["close"].iloc[0] - 1.0)
        strategies["buy_and_hold"] = {
            "final_equity": round(initial_capital * (1 + bh_total), 2),
            "total_return_pct": round(bh_total * 100, 2),
        }

    summary = {
        "strategy_id": strategy_id, "strategy_name": strategy_name, "year": year,
        "initial_capital": initial_capital, "asset": "XAU/USD",
        "n_trading_days": int(len(d)),
        "strategies": strategies,
        "statistical_tests": {"p_value": boot["p_value"], "significant": boot["significant"],
                              "bootstrap_95ci_ann": [boot["ci_low"], boot["ci_high"]]},
        "backtest_recommendation": rec, "backtest_confidence": conf,
        "gates": gates,
    }
    # trial-aware moments (per-period Sharpe + skew/kurt + single-test PSR) for Deflated-Sharpe
    # deflation in the runner (which knows the full trial set); JSON-safe finite floats.
    if trial_aware_moments is not None:
        summary["statistical_tests"].update(trial_aware_moments(d["strat_ret"].values))
    if oos_year is not None:
        oos = _oos_slice_stats(d, oos_year)
        if oos is not None:
            summary["oos"] = oos
    # Honest-gate extensions (OLA 2): B1' paired-exposure baseline + cost stress x1/x2/x3.
    # Shared implementation in services/common/metrics.py — do not re-derive here.
    try:
        from services.common.metrics import cost_stress, paired_exposure_baseline
        b1p = paired_exposure_baseline(d["position"].values, d["ret"].values, TRADING_DAYS)
        stress = cost_stress(d["position"].values, d["ret"].values, d["cost"].values,
                             d["swap"].values, TRADING_DAYS)
        summary["honest_gate"] = {
            "b1_prime": b1p,
            "beats_b1_prime_calmar": bool(m["calmar"] > b1p["calmar"]),
            "cost_stress": stress,
        }
    except Exception:
        pass  # extensions are additive; never break the core backtest
    trades_file = {"strategy_id": strategy_id, "strategy_name": strategy_name,
                   "initial_capital": initial_capital,
                   "date_range": {"start": d["time"].iloc[0].date().isoformat(),
                                  "end": d["time"].iloc[-1].date().isoformat()},
                   "trades": trades,
                   "summary": {"total_trades": len(trades), "win_rate": strat_stats["win_rate_pct"],
                               "total_return_pct": m["total_return_pct"], "sharpe_ratio": m["sharpe"],
                               "max_drawdown_pct": m["max_dd"], "profit_factor": pf,
                               "p_value": boot["p_value"], "n_long": n_long, "n_short": n_short}}
    return {"summary": summary, "trades": trades_file, "metrics": m, "attribution": attrib,
            "gates": gates, "recommendation": rec, "bootstrap": boot,
            "headline": {"return_pct": m["total_return_pct"], "sharpe": m["sharpe"],
                         "p_value": boot["p_value"]}, "equity_curve": d[["time", "equity"]]}
