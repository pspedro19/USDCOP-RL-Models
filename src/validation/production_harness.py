"""Production assurance harness for multi-asset model lifecycle.

Pure, deterministic checks suitable for CI and scheduled runs.  It does not
promote artifacts implicitly: a failed gate always returns ``go=False``.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json
from typing import Any, Mapping

@dataclass(frozen=True)
class HarnessThresholds:
    max_drift_psi: float = 0.20
    max_error_rate: float = 0.01
    max_latency_p95_ms: float = 750.0
    min_uptime: float = 0.995
    min_sharpe: float = 0.0
    max_drawdown: float = 0.35

@dataclass(frozen=True)
class HarnessResult:
    go: bool
    gates: dict[str, bool]
    reasons: list[str]
    artifact_id: str
    timestamp: str

def artifact_id(manifest: Mapping[str, Any]) -> str:
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]

def verify_artifact_manifest(manifest: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Fail-closed verification of release metadata and content hashes."""
    errors: list[str] = []
    for key in ("model_version", "dataset_hash"):
        if not manifest.get(key):
            errors.append(f"missing {key}")
    declared = manifest.get("artifact_hash") or manifest.get("artifact_sha256")
    if declared:
        body = {k: v for k, v in manifest.items() if k not in {"artifact_hash", "artifact_sha256"}}
        actual = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if declared != actual and declared != actual[:16]:
            errors.append("artifact hash mismatch")
    return (not errors, errors)

def _gate(name: str, condition: bool, reasons: list[str], msg: str) -> bool:
    if not condition: reasons.append(msg)
    return condition

def evaluate(candidate: Mapping[str, Any], champion: Mapping[str, Any] | None = None,
             thresholds: HarnessThresholds = HarnessThresholds(), kill_switch: bool = False) -> HarnessResult:
    reasons: list[str] = []
    metrics = candidate.get("metrics", {})
    gates = {
        "kill_switch": _gate("kill_switch", not kill_switch, reasons, "kill switch active"),
        "artifact": _gate("artifact", bool(candidate.get("model_version")) and bool(candidate.get("dataset_hash")), reasons, "model_version and dataset_hash required"),
        "drift": _gate("drift", float(metrics.get("psi", 0)) <= thresholds.max_drift_psi, reasons, "feature drift PSI above threshold"),
        "slo_latency": _gate("slo_latency", float(metrics.get("latency_p95_ms", 0)) <= thresholds.max_latency_p95_ms, reasons, "latency p95 SLO breached"),
        "slo_errors": _gate("slo_errors", float(metrics.get("error_rate", 0)) <= thresholds.max_error_rate, reasons, "error-rate SLO breached"),
        "uptime": _gate("uptime", float(metrics.get("uptime", 1)) >= thresholds.min_uptime, reasons, "uptime SLO breached"),
        "risk": _gate("risk", float(metrics.get("sharpe", 0)) >= thresholds.min_sharpe and abs(float(metrics.get("max_drawdown", 0))) <= thresholds.max_drawdown, reasons, "risk metrics outside limits"),
    }
    if champion:
        # Challenger must not materially regress risk-adjusted performance.
        gates["challenger"] = _gate("challenger", float(metrics.get("sharpe", 0)) >= float(champion.get("metrics", {}).get("sharpe", 0)) - 0.10, reasons, "challenger Sharpe regresses > 0.10")
    else: gates["challenger"] = True
    return HarnessResult(all(gates.values()), gates, reasons, artifact_id(candidate), datetime.now(timezone.utc).isoformat())

def write_evidence(result: HarnessResult, path: str | Path) -> Path:
    out = Path(path); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    return out

def rollback_state(active: str, previous: str, result: HarnessResult) -> dict[str, str]:
    return {"active": active if result.go else previous, "previous": previous, "action": "promote" if result.go else "rollback"}

def rehearse_rollback(active: str, previous: str, failed_candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Exercise rollback decision without mutating deployment state."""
    result = evaluate(failed_candidate)
    state = rollback_state(active, previous, result)
    return {"ok": (not result.go and state["active"] == previous and state["action"] == "rollback"),
            "result": asdict(result), "state": state}

def evaluate_observability(metrics: Mapping[str, Any], thresholds: HarnessThresholds = HarnessThresholds()) -> dict[str, bool]:
    """Check required telemetry is present and within SLO budgets."""
    required = ("latency_p95_ms", "error_rate", "uptime", "psi")
    present = all(k in metrics for k in required)
    return {"telemetry_complete": present,
            "latency_slo": present and float(metrics.get("latency_p95_ms", 1e9)) <= thresholds.max_latency_p95_ms,
            "error_slo": present and float(metrics.get("error_rate", 1e9)) <= thresholds.max_error_rate,
            "uptime_slo": present and float(metrics.get("uptime", 0)) >= thresholds.min_uptime,
            "drift_slo": present and float(metrics.get("psi", 1e9)) <= thresholds.max_drift_psi}
