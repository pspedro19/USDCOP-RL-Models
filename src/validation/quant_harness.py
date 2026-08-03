"""Deterministic quantitative assurance harness for all supported assets.

The harness is deliberately dependency-light and returns machine-readable evidence.
It never promotes a model; callers must apply their own production gate.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Iterable
import math
import numpy as np

ASSETS = ("usdcop", "xauusd", "btcusdt", "spx500")

@dataclass
class Check:
    name: str
    passed: bool
    value: float | None = None
    detail: str = ""

@dataclass
class AssetEvidence:
    asset: str
    checks: list[Check]
    metrics: dict[str, float]
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

def _check(name: str, ok: bool, value: float | None = None, detail: str = "") -> Check:
    return Check(name, bool(ok), value, detail)

def _arr(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)

def evaluate_asset(asset: str, frame: Any, *, prediction: Iterable[float] | None = None,
                   actual: Iterable[float] | None = None, returns: Iterable[float] | None = None,
                   baseline: Iterable[float] | None = None, costs_bps: float = 10.0,
                   trial_count: int | None = None, dsr: float | None = None,
                   pbo: float | None = None, benchmark_return: float | None = None) -> AssetEvidence:
    """Evaluate one asset. ``frame`` may be a pandas DataFrame or mapping of columns."""
    if asset not in ASSETS:
        raise ValueError(f"unsupported asset: {asset}")
    checks: list[Check] = []
    columns = set(getattr(frame, "columns", frame.keys() if hasattr(frame, "keys") else ()))
    checks.append(_check("schema.asset", asset in ASSETS))
    checks.append(_check("schema.timestamp", "timestamp" in columns or "ts" in columns))
    if hasattr(frame, "__len__"):
        checks.append(_check("data.non_empty", len(frame) > 0, float(len(frame))))
    ts = frame["timestamp"] if "timestamp" in columns else (frame["ts"] if "ts" in columns else None)
    if ts is not None:
        try:
            vals = list(ts)
            monotonic = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
            checks.append(_check("data.monotonic_unique", monotonic))
        except Exception:
            checks.append(_check("data.monotonic_unique", False))
    else:
        checks.append(_check("data.monotonic_unique", False, detail="timestamp missing"))
    # PIT and leakage contracts: available_at must precede observation timestamp.
    if "available_at" in columns and ts is not None:
        try:
            checks.append(_check("pit.available_at", all(a <= t for a, t in zip(frame["available_at"], ts))))
        except Exception:
            checks.append(_check("pit.available_at", False))
    elif "available_at" not in columns:
        checks.append(_check("pit.available_at", False, detail="required for point-in-time evidence"))
    if prediction is not None and actual is not None:
        p, y = _arr(prediction), _arr(actual)
        n = min(len(p), len(y)); p, y = p[:n], y[:n]
        err = p - y
        mae = float(np.mean(np.abs(err))) if n else math.nan
        rmse = float(np.sqrt(np.mean(err ** 2))) if n else math.nan
        direction = float(np.mean(np.sign(p[1:] - p[:-1]) == np.sign(y[1:] - y[:-1]))) if n > 1 else math.nan
        checks += [_check("forecast.finite", bool(np.isfinite(p).all() and np.isfinite(y).all())),
                   _check("forecast.sample_size", n >= 30, float(n))]
        metrics = {"mae": mae, "rmse": rmse, "directional_accuracy": direction}
        if baseline is not None:
            b = _arr(baseline)[:n]; bmae = float(np.mean(np.abs(b-y)))
            metrics["baseline_mae"] = bmae
            checks.append(_check("forecast.beats_baseline", mae < bmae, mae - bmae))
    else:
        metrics = {}
    if returns is not None:
        r = _arr(returns); net = r - costs_bps / 10000.0
        mean, vol = float(np.mean(net)), float(np.std(net, ddof=1)) if len(net) > 1 else math.nan
        sharpe = mean / vol * math.sqrt(252) if vol and np.isfinite(vol) else math.nan
        eq = np.cumprod(1 + net) if len(net) else np.array([])
        dd = float(np.min(eq / np.maximum.accumulate(eq) - 1)) if len(eq) else math.nan
        turnover = float(np.mean(np.abs(np.diff(r)))) if len(r) > 1 else 0.0
        metrics.update({"net_return": float(np.prod(1 + net) - 1) if len(net) else math.nan,
                        "sharpe": sharpe, "max_drawdown": dd, "turnover": turnover,
                        "costs_bps": costs_bps})
        checks += [_check("strategy.finite", bool(np.isfinite(r).all())),
                   _check("strategy.sample_size", len(r) >= 30, float(len(r))),
                   _check("strategy.costs_declared", costs_bps > 0, costs_bps),
                   _check("strategy.max_drawdown_bound", dd > -0.8, dd)]
    # Evidence fields expected by downstream DSR/PBO gates.  Values are supplied
    # by the experiment manifest (never inferred from synthetic returns).
    stats_ok = (trial_count is not None and trial_count > 0 and dsr is not None
                and 0.0 <= dsr <= 1.0 and pbo is not None and 0.0 <= pbo <= 1.0)
    checks.append(_check("statistics.dsr_pbo_evidence", stats_ok,
                         dsr, detail="requires trial_count, dsr and pbo from experiment manifest"))
    if trial_count is not None:
        metrics["trial_count"] = float(trial_count)
    if dsr is not None:
        metrics["dsr"] = float(dsr)
    if pbo is not None:
        metrics["pbo"] = float(pbo)
    if benchmark_return is not None:
        metrics["benchmark_return"] = float(benchmark_return)
        if returns is not None and "net_return" in metrics:
            checks.append(_check("strategy.beats_benchmark", metrics["net_return"] > benchmark_return,
                                 metrics["net_return"] - benchmark_return))
    return AssetEvidence(asset, checks, metrics)

def run_harness(datasets: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    evidence = [evaluate_asset(a, datasets[a], **kwargs.get(a, {})) for a in ASSETS if a in datasets]
    return {"schema_version": 1, "assets": [asdict(e) | {"passed": e.passed} for e in evidence],
            "passed": bool(evidence) and all(e.passed for e in evidence)}
