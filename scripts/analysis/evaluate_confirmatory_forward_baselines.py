"""Evaluate frozen, parameter-free baseline policies on forward SessionSpecs."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from src.research.session_env import run_session


def momentum(close: np.ndarray) -> np.ndarray:
    out = np.zeros(59, dtype=float)
    for bar in range(3, 59):
        out[bar] = np.sign(close[bar] / close[bar - 3] - 1.0)
    return out


def mean_reversion(close: np.ndarray) -> np.ndarray:
    out = np.zeros(59, dtype=float)
    for bar in range(12, 59):
        window = np.log(close[bar - 11:bar + 1] / close[bar - 12:bar])
        sd = float(np.std(window, ddof=1))
        if sd > 0:
            z = float((window[-1] - np.mean(window)) / sd)
            out[bar] = -1.0 if z > 1.0 else (1.0 if z < -1.0 else 0.0)
    return out


def opening_range(close: np.ndarray) -> np.ndarray:
    out = np.zeros(59, dtype=float)
    high, low = float(np.max(close[:6])), float(np.min(close[:6]))
    out[6:] = np.where(close[6:59] > high, 1.0, np.where(close[6:59] < low, -1.0, 0.0))
    return out


def summarize(results: list, name: str, annualization: float) -> dict:
    daily = np.asarray([float(x.daily_return) for x in results])
    equity = np.cumprod(1.0 + daily)
    sd = float(np.std(daily, ddof=1)) if len(daily) > 1 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    n_changes = int(sum(x.n_changes for x in results))
    return {
        "baseline": name,
        "n_sessions": len(results),
        "n_traded_sessions": int(sum(x.n_changes > 0 for x in results)),
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "sharpe": float(np.mean(daily) / sd * np.sqrt(annualization)) if sd > 0 else 0.0,
        "max_drawdown": float(np.min(dd)) if len(dd) else 0.0,
        "total_cost": float(sum(x.total_cost for x in results)),
        "total_changes": n_changes,
        "daily_returns": daily.tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=Path, required=True)
    parser.add_argument("--portable-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bundle = pickle.loads(args.specs.read_bytes())
    manifest = bundle["manifest"]
    if manifest.get("portable_sha256") != args.portable_sha256:
        raise ValueError("forward specs and portable hash differ")
    specs = bundle["sessions"]
    if not specs:
        raise ValueError("no forward sessions")
    dates = [s.date for s in specs]
    years = max((max(dates) - min(dates)).days / 365.25, 1 / 365.25)
    annualization = len(specs) / years
    policies = {
        "always_flat": lambda _s: np.zeros(59),
        "always_long_1x": lambda _s: np.ones(59),
        "always_short_1x": lambda _s: -np.ones(59),
        "momentum_3bar": lambda s: momentum(s.close),
        "mean_reversion_12bar": lambda s: mean_reversion(s.close),
        "opening_range_6bar": lambda s: opening_range(s.close),
        "random_seed42": None,
        "regime_two_rules": None,
    }
    rng = np.random.default_rng(42)
    rows = []
    for name, policy in policies.items():
        results = []
        for spec in specs:
            if name == "random_seed42":
                weights = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=59)
            elif name == "regime_two_rules":
                state = int(np.argmax(spec.context[3:]))
                level = -1.0 if state == 3 else (1.0 if state == 2 else 0.0)
                weights = np.full(59, level)
            else:
                weights = policy(spec)
            results.append(run_session(spec.close, weights, spec.spread_pips, date=spec.date))
        rows.append(summarize(results, name, annualization))
    payload = {
        "schema_version": "confirmatory-forward-baselines-v1",
        "scope": "post_freeze_forward_2026_partial",
        "specs_sha256": __import__("hashlib").sha256(args.specs.read_bytes()).hexdigest(),
        "portable_sha256": args.portable_sha256,
        "n_sessions": len(specs),
        "annualization_sessions_per_year": annualization,
        "rows": rows,
        "parameter_selection": "none; all policies frozen before evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({r["baseline"]: {"return": r["total_return"], "sharpe": r["sharpe"]} for r in rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
