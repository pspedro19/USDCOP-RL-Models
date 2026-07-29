"""Production promotion gate for the SP500 strategy.

The gate is deliberately strict: scaffold/synthetic data can exercise code but
can never produce a promotable OOS report.  Evidence is represented as a plain
mapping so it can be loaded from JSON manifests produced by the backtest DAG.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


REQUIRED_EVIDENCE = ("data_pit", "leakage", "purged_cv", "pbo", "dsr", "costs", "benchmark")


@dataclass(frozen=True)
class SP500GateResult:
    passed: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def validate_sp500_oos_evidence(evidence: Mapping[str, Any]) -> SP500GateResult:
    """Validate an OOS evidence manifest and return deterministic pass/fail.

    Required fields: real PIT data lineage, costs, benchmarks and statistical
    evidence.  Numeric thresholds match ``config/forecast_experiments``.
    """
    reasons: list[str] = []
    source = str(evidence.get("data_source_status", "")).lower()
    if source in {"", "synthetic", "synthetic_scaffold_until_real_feed", "scaffold"}:
        reasons.append("synthetic_or_unidentified_data")
    if evidence.get("point_in_time") is not True or evidence.get("available_at_required") is not True:
        reasons.append("pit_lineage_missing")
    for key in REQUIRED_EVIDENCE:
        value = evidence.get(key)
        # pbo/dsr may be supplied as their numeric statistic (the preferred
        # representation) instead of a separate boolean gate flag.
        if key in {"pbo", "dsr"} and isinstance(value, (int, float)) and not isinstance(value, bool):
            continue
        if value is not True:
            reasons.append(f"gate_{key}_missing")
    if not evidence.get("cost_model"):
        reasons.append("cost_model_missing")
    benchmarks = evidence.get("benchmarks")
    if not isinstance(benchmarks, (list, tuple)) or len(benchmarks) < 4:
        reasons.append("benchmarks_incomplete")
    if evidence.get("oos_sharpe", float("-inf")) < 0.5:
        reasons.append("oos_sharpe_below_threshold")
    if evidence.get("oos_return_pct", float("-inf")) < 0:
        reasons.append("oos_return_below_threshold")
    if evidence.get("max_drawdown_pct", float("inf")) > 25:
        reasons.append("drawdown_above_threshold")
    if evidence.get("pbo", 1.0) >= 0.5:
        reasons.append("pbo_too_high")
    if evidence.get("dsr", 0.0) <= 0.95:
        reasons.append("dsr_below_threshold")
    return SP500GateResult(not reasons, tuple(reasons))


__all__ = ["SP500GateResult", "validate_sp500_oos_evidence"]
