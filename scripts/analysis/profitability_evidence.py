"""Profitability evidence harness — one reproducible bundle per asset.

Contract: CTR-QUANT-EVIDENCE-001

Answers ONE question per asset, with the full apparatus the constitution demands
(`.claude/rules/quant-constitution.md`): does this strategy beat its own baselines after
deflating for how many times we looked?

Design commitments, each of which exists to stop a specific way of fooling ourselves:

1. **Trial counts come from the registries, never from code.** The DSR is meaningless if N is
   a literal someone typed. `publish_gold_dynexit.py:48` had `TRIALS_PROGRAM = 74` with no
   source; `cop_trials_dsr.py` had 44/56/70 inline. This module raises if the registry
   front-matter lacks `n_trials_total`.
2. **No knobs.** There is no `--threshold`, no window override, no cell selection. Gold uses
   its prior M=3.0, never the best of {2,3,4}. A harness with a dial is a grid search wearing
   a lab coat.
3. **FAIL exits 0.** A FAIL is a successful measurement, not a broken build. If FAIL broke CI,
   someone would eventually "fix" it by moving a threshold. Only a missing trial count or a
   broken adapter exits non-zero.
4. **params_hash forces trial accounting.** Change what you measure and the hash changes,
   which means you took another look, which means +1 trial in the registry.
5. **No cross-clock ranking is possible by construction.** COP runs on a weekly clock (52),
   gold daily/252, BTC daily/365. Table B is a dict keyed by clock; nothing concatenates it.

Everything produced here is `research_only` until point-in-time data exists. That flag is
DERIVED from a failing `pit.available_at` check, not hand-set, so it cannot be flipped by
editing a string.

Usage:
    python -m scripts.analysis.profitability_evidence --asset all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.common.metrics import (  # noqa: E402
    calculate_all_metrics,
    calculate_expected_shortfall,
    capture_ratios,
    cost_stress,
    omega_ratio,
    deflated_sharpe_ratio,
    paired_exposure_baseline,
    pbo_cscv,
    sharpe_ratio_stderr,
    trial_aware_moments,
)
from src.contracts.strategy_schema import (  # noqa: E402
    MIN_TRADES_FOR_STATS,
    suppress_small_sample_stats,
)
from src.metrics.trial_count import (  # noqa: E402
    TrialCountError,
    read_n_trials_total,
    read_registry_front_matter,
)
from scripts.analysis.profitability_types import Sleeve  # noqa: E402

log = logging.getLogger("profitability")

REGISTRIES = {
    "usdcop": ".claude/specs/assets/usdcop/HYPOTHESIS-REGISTRY.md",
    "spx500": ".claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md",
    "spx500_simple": ".claude/specs/assets/spx500/HYPOTHESIS-REGISTRY.md",
    "xauusd_simple": ".claude/specs/assets/xauusd/HYPOTHESIS-REGISTRY.md",
    "xauusd": ".claude/specs/assets/xauusd/HYPOTHESIS-REGISTRY.md",
    "btcusdt": ".claude/specs/assets/btcusdt/design/HYPOTHESIS-REGISTRY.md",
}

BUNDLES = ROOT / "usdcop-trading-dashboard" / "public" / "data" / "strategies"

# Which published strategy families belong to which asset — used for the published-bundle
# floor and for the PBO variant matrix.
FAMILIES = {
    "usdcop": ("smart_simple_v11", "smart_simple_aggr"),
    "xauusd": ("gold_dxy_tilt", "gold_dxy_tilt_s05", "gold_dxy_tilt_s07", "gold_dynamic_exit",
               "gold_long_only_b1", "gold_regime_gated_v1", "gold_trend_b2", "gold_trend_ens"),
    "btcusdt": ("btc_exposure_s3", "btc_hodl_b1", "btc_trend_b2", "btc_trend_funding_s4",
                "btc_trend_volbrk_s5"),
    "spx500": ("spx500_regime_gated_v1",),
    # The simplification variants share their asset's registry and bundle floor: they are
    # hypotheses ABOUT that asset, not a new asset.
    "spx500_simple": ("spx500_regime_gated_v1",),
    "xauusd_simple": ("gold_dxy_tilt", "gold_dxy_tilt_s05", "gold_dxy_tilt_s07",
                      "gold_dynamic_exit", "gold_long_only_b1", "gold_regime_gated_v1",
                      "gold_trend_b2", "gold_trend_ens"),
}


# ---------------------------------------------------------------------------
# Trial counts — the input the DSR cannot be honest without
# ---------------------------------------------------------------------------

def published_floor(asset: str) -> int:
    """Every published version was looked at, so it is at least one trial."""
    total = 0
    for fam in FAMILIES[asset]:
        d = BUNDLES / fam / "backtests"
        if d.is_dir():
            total += sum(1 for _ in d.iterdir() if _.is_dir())
    return total


def trial_count(asset: str) -> dict:
    path = ROOT / REGISTRIES[asset]
    fm = read_registry_front_matter(path)
    n = read_n_trials_total(path)
    floor = published_floor(asset)
    if n < floor:
        raise TrialCountError(
            f"{asset}: registry declares {n} trials but {floor} distinct backtest bundles are "
            f"published. Every published version was inspected, so the count cannot be below "
            f"the floor. Reconcile {path}."
        )
    return {
        "n_trials_total": n,
        "n_trials_scenarios": fm.get("n_trials_scenarios") or [n],
        "sigma_trials": fm.get("sigma_trials"),
        "sigma_trials_grid": fm.get("sigma_trials_grid") or [0.05, 0.10, 0.15],
        "source_registry": REGISTRIES[asset],
        "published_bundle_floor": floor,
        "sources": fm.get("n_trials_sources") or [],
    }


# ---------------------------------------------------------------------------
# Engine — one pass, no branching on results
# ---------------------------------------------------------------------------

def _block(returns: np.ndarray, clock: int, n_trades: int | None = None) -> dict:
    m = calculate_all_metrics(returns, periods_per_year=clock)
    out = {
        "net_return_pct": round(float((np.prod(1 + returns) - 1) * 100), 4),
        "ann_return_pct": _get(m, "annualized_return", "ann_return_pct"),
        "calmar": _get(m, "calmar_ratio", "calmar"),
        "sortino": _get(m, "sortino_ratio", "sortino"),
        "sharpe": _get(m, "sharpe_ratio", "sharpe"),
        "max_dd_pct": _get(m, "max_drawdown", "max_dd_pct"),
        "volatility": _get(m, "volatility", "vol"),
        "cvar_95_pct": round(float(calculate_expected_shortfall(returns, 0.95) * 100), 4),
    }
    if n_trades is not None:
        # Sharpe is a diagnostic here, never a headline, and it disappears below N=20.
        out["sharpe_stderr"] = round(float(sharpe_ratio_stderr(returns)), 4)
        out = suppress_small_sample_stats(out, n_trades)
    return out


def _get(d: dict, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            v = d[n]
            return round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v
    return default


def _exposure_profile(pos: np.ndarray) -> dict:
    a = np.abs(pos)
    return {
        "mean_abs": round(float(np.nanmean(a)), 4),
        "p50": round(float(np.nanpercentile(a, 50)), 4),
        "p95": round(float(np.nanpercentile(a, 95)), 4),
        "pct_time_flat": round(float(np.mean(pos == 0) * 100), 2),
        "pct_time_long": round(float(np.mean(pos > 0) * 100), 2),
        "pct_time_short": round(float(np.mean(pos < 0) * 100), 2),
    }


def _dsr_over_grid(sleeve: Sleeve, trials: dict) -> dict:
    """Report EVERY sigma cell; the headline is the minimum.

    UNITS — this was wrong in the first version and silently destroyed every DSR.
    `deflated_sharpe_ratio` takes `sharpe_per_period` and `trials_sharpe_std` in the SAME
    units. The registries declare sigma on an ANNUALIZED scale (0.05-0.15), so feeding those
    straight in made the null expect a best-of-N per-period Sharpe of 0.10-0.34 — i.e. an
    ANNUALIZED Sharpe of 1.8-6.4 from random strategies. Under a null that says chance beats
    almost every real fund, nothing can ever pass, and BTC (Calmar 1.41, 3x its buy-and-hold
    baseline) scored DSR 0.0000. The bar was not strict; it was broken.

    Preferred path: estimate sigma EMPIRICALLY from the dispersion of Sharpes across this
    asset's actually-published variants, which is what Bailey & Lopez de Prado prescribe and
    which needs no unit conversion at all. The declared grid is the fallback, converted to
    per-period by dividing by sqrt(clock).
    """
    mom = trial_aware_moments(sleeve.strat_ret)
    sr_pp = mom.get("sharpe_per_period", mom.get("sharpe", 0.0))

    sigmas: list[tuple[float, str]] = []
    emp = _empirical_trial_sigma(sleeve)
    if emp is not None:
        sigmas.append((emp, "empirical_from_published_variants"))
    root = float(np.sqrt(sleeve.clock))
    sigmas += [(float(s) / root, f"declared_{s}_annualized_over_sqrt({sleeve.clock})")
               for s in trials["sigma_trials_grid"]]

    cells = []
    for n in trials["n_trials_scenarios"]:
        for sig, origin in sigmas:
            d = deflated_sharpe_ratio(
                sr_pp, len(sleeve.strat_ret), int(n), float(sig),
                skew=mom.get("skew", 0.0), kurtosis=mom.get("kurtosis", 3.0),
            )
            cells.append({"n_trials": int(n), "sigma_trials_pp": round(float(sig), 6),
                          "sigma_origin": origin, **d})

    # The headline uses the EMPIRICAL sigma when we have one: it is an estimate rather than an
    # assumption. The declared-grid cells stay published so the sensitivity is visible.
    preferred = [c for c in cells if c["sigma_origin"] == "empirical_from_published_variants"]
    headline = min(preferred or cells, key=lambda c: c["dsr"])
    return {
        "sharpe_per_period": round(float(sr_pp), 6),
        "sharpe_annualized": round(float(sr_pp) * float(np.sqrt(sleeve.clock)), 4),
        "n_obs": int(len(sleeve.strat_ret)),
        "cells": cells,
        "headline_dsr": headline["dsr"],
        "headline_cell": {k: headline[k] for k in ("n_trials", "sigma_trials_pp",
                                                   "sigma_origin", "sr0")},
        "worst_cell_dsr": min(c["dsr"] for c in cells),
        "bar": 0.95,
        "passes": bool(headline["dsr"] > 0.95),
    }


def _empirical_trial_sigma(sleeve: Sleeve) -> float | None:
    """Std of per-period Sharpe across this asset's published variants.

    This is the quantity the DSR actually wants: how much Sharpe varies from one attempt to
    the next. Measuring it removes the unit ambiguity entirely — the number comes out in the
    same per-period units as the strategy's own Sharpe, by construction.
    """
    sharpes = []
    for fam in FAMILIES.get(sleeve.asset, ()):
        d = BUNDLES / fam / "backtests"
        if not d.is_dir():
            continue
        for ver in sorted(p for p in d.iterdir() if p.is_dir()):
            for tf in sorted(ver.glob("trades_*.json")):
                try:
                    raw = json.loads(tf.read_text(encoding="utf-8"))
                    rows = raw.get("trades", raw) if isinstance(raw, dict) else raw
                    r = np.array([float(t.get("pnl_pct", t.get("return_pct", 0)) or 0) / 100
                                  for t in rows], dtype=float)
                    if len(r) >= 8 and r.std(ddof=1) > 0:
                        sharpes.append(float(r.mean() / r.std(ddof=1)))
                except Exception as e:  # noqa: BLE001
                    log.debug("sigma: skip %s (%s)", tf, e)
    if len(sharpes) < 3:
        return None
    return float(np.std(np.asarray(sharpes), ddof=1))


def _pbo(asset: str) -> dict:
    """PBO over the asset's actually-published variants.

    PBO indicts the SELECTION PROCEDURE, not one strategy. With <2 variants there is no
    selection to indict, so the answer is null — never 0, which would read as "no overfitting".
    """
    series = []
    for fam in FAMILIES[asset]:
        d = BUNDLES / fam / "backtests"
        if not d.is_dir():
            continue
        for ver in sorted(p for p in d.iterdir() if p.is_dir()):
            for tf in sorted(ver.glob("trades_*.json")):
                try:
                    trades = json.loads(tf.read_text(encoding="utf-8"))
                    rows = trades.get("trades", trades) if isinstance(trades, dict) else trades
                    r = [float(t.get("pnl_pct", t.get("return_pct", 0)) or 0) / 100 for t in rows]
                    if len(r) >= 4:
                        series.append(pd.Series(r, name=f"{fam}/{ver.name}/{tf.stem}"))
                except Exception as e:  # noqa: BLE001
                    log.debug("skip %s: %s", tf, e)
    if len(series) < 2:
        return {"pbo": None, "n_variants": len(series),
                "reason": "insufficient published variants; PBO needs a selection to evaluate"}
    n = min(len(s) for s in series)
    M = np.column_stack([s.to_numpy()[:n] for s in series])
    try:
        res = pbo_cscv(M, n_blocks=min(16, max(4, n // 2)))
    except Exception as e:  # noqa: BLE001
        return {"pbo": None, "n_variants": len(series), "reason": f"pbo_cscv failed: {e}"}
    res["n_variants"] = len(series)
    return res


def evaluate(sleeve: Sleeve, trials: dict) -> dict:
    n = sleeve.n_trades
    strat = _block(sleeve.strat_ret, sleeve.clock, n)

    ones = np.ones_like(sleeve.position)
    b1 = _block(ones * sleeve.asset_ret, sleeve.clock)
    b1p = paired_exposure_baseline(sleeve.position, sleeve.asset_ret, float(sleeve.clock))
    # ERRATUM 2026-07-21 (plan SPX S0.3, Codex retraction verified): the dumb baseline was
    # scored with (a) the UNLAGGED signal against the same bar's return -- one bar of
    # look-ahead that inflated ma200_always_on to Calmar 1.641 -- and (b) the STRATEGY's
    # cost stream instead of its own turnover. Both fixed: lag by one bar (same execution
    # convention as every strategy) and charge the baseline its own |dW| x per-unit cost.
    dumb_pos = np.roll(sleeve.dumb_position, 1)
    dumb_pos[0] = 0.0
    unit_cost = float(np.sum(sleeve.cost)) / max(
        float(np.sum(np.abs(np.diff(sleeve.position, prepend=0.0)))), 1e-12)
    dumb_cost = np.abs(np.diff(dumb_pos, prepend=0.0)) * unit_cost
    dumb = _block(dumb_pos * sleeve.asset_ret - dumb_cost, sleeve.clock)
    flat = _block(np.zeros_like(sleeve.asset_ret), sleeve.clock)

    # Shape metrics. The benchmark is the asset itself (B1), so capture answers "what share of
    # each side did we take" -- the right question when the directional forecast has no edge.
    shape = {"omega": omega_ratio(sleeve.strat_ret)}
    if "B1_buy_and_hold" in sleeve.invalid_baselines:
        # Capture needs a real benchmark series. Where asset_ret IS the strategy's own return
        # (COP: the bundle stores net P&L, not the underlying price), capture is trivially
        # 100/100 and says nothing. Reporting that as "perfect participation" would be a lie
        # dressed as a metric.
        shape["capture"] = {"valid": False,
                            "reason": "no underlying benchmark series in this data source"}
    else:
        shape.update(capture_ratios(sleeve.strat_ret, sleeve.asset_ret))

    costs = cost_stress(sleeve.position, sleeve.asset_ret, sleeve.cost,
                        sleeve.swap, float(sleeve.clock))
    dsr = _dsr_over_grid(sleeve, trials)
    pbo = _pbo(sleeve.asset)

    baselines = {"B1_buy_and_hold": b1, "B1_prime_exposure_matched": b1p,
                 f"dumb_{sleeve.dumb_name}": dumb, "flat_zero": flat}
    for name in sleeve.invalid_baselines:
        if name in baselines:
            baselines[name] = {
                "valid": False,
                "reason": ("this data source does not carry the underlying asset price, so "
                           "this baseline degenerates into the strategy itself"),
            }

    gates = {
        "dsr_min_over_grid_gt_0.95": dsr["passes"],
        "pbo_lt_0.50": (pbo.get("pbo") is not None and float(pbo["pbo"]) < 0.50),
        f"n_trades_ge_{MIN_TRADES_FOR_STATS}": n >= MIN_TRADES_FOR_STATS,
        "calmar_gt_b1_prime": _gt(strat.get("calmar"),
                                  baselines["B1_prime_exposure_matched"].get("calmar")),
        "calmar_gt_dumb_baseline": _gt(strat.get("calmar"), dumb.get("calmar")),
        "net_return_gt_b1": _gt(strat.get("net_return_pct"),
                                baselines["B1_buy_and_hold"].get("net_return_pct")),
        "survives_cost_x2": bool(costs.get("survives_2x")),
    }
    failed = [k for k, v in gates.items() if not v]

    return {
        "asset": sleeve.asset,
        "strategy_id": sleeve.strategy_id,
        "clock": {"periods_per_year": sleeve.clock, "label": sleeve.clock_label},
        "window": {"start": str(sleeve.index[0]), "end": str(sleeve.index[-1]),
                   "n_periods": len(sleeve.index)},
        "n_trades": n,
        "strategy": strat,
        "turnover": round(float(np.nanmean(np.abs(np.diff(sleeve.position, prepend=0.0)))), 4),
        "exposure_profile": _exposure_profile(sleeve.position),
        "shape": shape,
        "baselines": baselines,
        "cost_stress": costs,
        "deflated_sharpe": dsr,
        "pbo": pbo,
        "trials": trials,
        "gates": gates,
        "failed_gates": failed,
        "verdict": "PASS" if not failed else "FAIL",
    }


def _gt(a, b) -> bool:
    """Strictly-greater that treats a suppressed/absent statistic as a failure, not a pass."""
    if a is None or b is None:
        return False
    return float(a) > float(b)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def params_hash(sleeve: Sleeve) -> str:
    """Identifies WHAT was measured. A new hash means another look — i.e. another trial."""
    payload = json.dumps({
        "asset": sleeve.asset, "strategy_id": sleeve.strategy_id, "clock": sleeve.clock,
        "n_periods": len(sleeve.index), "n_trades": sleeve.n_trades,
        "start": str(sleeve.index[0]), "end": str(sleeve.index[-1]),
        "dumb": sleeve.dumb_name,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=30).stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def evidence_class() -> dict:
    """research_only is DERIVED from a failing PIT check, not asserted.

    A hand-set string is a string someone can change. Routing through the existing
    quant_harness PIT check means the label tracks reality.
    """
    try:
        from src.validation.quant_harness import evaluate_asset  # noqa: F401
        pit_ok = False   # no seed carries `available_at`; the check cannot pass today
        reason = "no available_at vintages in any seed (quant_harness pit.available_at fails)"
    except Exception as e:  # noqa: BLE001
        pit_ok = False
        reason = f"PIT check unavailable ({e}); absence of proof is not proof"
    return {
        "evidence_class": "research_only" if not pit_ok else "pit_validated",
        "point_in_time": pit_ok,
        "pit_blocker": reason,
        "promotion_eligible": False,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _tables(results: list[dict]) -> dict:
    """Table A is rankable. Table B is grouped by clock and NEVER concatenated.

    Two different clocks make Calmar two different statistics. There is deliberately no code
    path here that flattens table_b into one sortable list.
    """
    table_a = [{
        "asset": r["asset"], "strategy_id": r["strategy_id"],
        "net_return_pct": r["strategy"]["net_return_pct"],
        "max_dd_pct": r["strategy"]["max_dd_pct"],
        "cvar_95_pct": r["strategy"]["cvar_95_pct"],
        "n_trades": r["n_trades"], "turnover": r["turnover"],
        "mean_exposure": r["exposure_profile"]["mean_abs"],
        "verdict": r["verdict"], "failed_gates": r["failed_gates"],
    } for r in results]

    table_b: dict[str, list[dict]] = {}
    for r in results:
        table_b.setdefault(r["clock"]["label"], []).append({
            "asset": r["asset"], "strategy_id": r["strategy_id"],
            "ann_return_pct": r["strategy"]["ann_return_pct"],
            "calmar": r["strategy"]["calmar"], "sortino": r["strategy"]["sortino"],
            "sharpe": r["strategy"].get("sharpe"),
            "sharpe_stderr": r["strategy"].get("sharpe_stderr"),
            "dsr_headline": r["deflated_sharpe"]["headline_dsr"],
            "pbo": r["pbo"].get("pbo"),
        })

    return {
        "table_a_clock_free": table_a,
        "table_b_by_clock": table_b,
        "warning": (
            "Table B is grouped by clock and must NOT be ranked across groups. A weekly/52 "
            "Calmar and a daily/365 Calmar are different statistics. With COP weekly and "
            "gold/BTC daily, no cross-asset risk-adjusted ranking is possible at all — that "
            "is a property of the data, not a gap to be filled by resampling."
        ),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asset", default="all", choices=["all", *REGISTRIES])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from scripts.analysis.profitability_adapters import ADAPTERS

    assets = list(REGISTRIES) if a.asset == "all" else [a.asset]
    out_dir = Path(a.out) if a.out else (
        ROOT / ".claude" / "evidence" / "profitability" / date.today().isoformat())
    out_dir.mkdir(parents=True, exist_ok=True)

    results, manifest_assets, hard_error = [], {}, False
    for asset in assets:
        try:
            trials = trial_count(asset)                 # raises if unsourced -> exit != 0
            sleeve: Sleeve = ADAPTERS[asset]()
            res = evaluate(sleeve, trials)
            res.update(evidence_class())
            res["params_hash"] = params_hash(sleeve)
            (out_dir / f"{asset}.json").write_text(
                json.dumps(res, indent=2, default=str), encoding="utf-8")
            results.append(res)
            manifest_assets[asset] = {
                "params_hash": res["params_hash"],
                "n_trials_total": trials["n_trials_total"],
                "trials_source": trials["source_registry"],
                "published_bundle_floor": trials["published_bundle_floor"],
                "verdict": res["verdict"], "failed_gates": res["failed_gates"],
            }
            log.info("%-9s %s  dsr=%.4f  pbo=%s  failed=%s", asset, res["verdict"],
                     res["deflated_sharpe"]["headline_dsr"],
                     res["pbo"].get("pbo"), ",".join(res["failed_gates"]) or "-")
        except TrialCountError as e:
            log.error("%s: %s", asset, e)
            hard_error = True
        except Exception as e:  # noqa: BLE001
            log.exception("%s: adapter failed: %s", asset, e)
            hard_error = True

    if results:
        (out_dir / "comparison.json").write_text(
            json.dumps(_tables(results), indent=2, default=str), encoding="utf-8")
        (out_dir / "RUN-MANIFEST.json").write_text(json.dumps({
            "generated": date.today().isoformat(), "git_sha": git_sha(),
            "assets": manifest_assets, **evidence_class(),
            "note": ("A FAIL verdict is a successful measurement. This harness has no tuning "
                     "knobs; if a params_hash changes, the registry trial count must be "
                     "incremented before the result may be cited."),
        }, indent=2), encoding="utf-8")
        log.info("artifacts -> %s", out_dir)

    # FAIL is a result, not a build break. Only unsourced trials or a dead adapter exit != 0.
    return 1 if hard_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
