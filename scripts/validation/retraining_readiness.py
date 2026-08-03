"""Fail-closed readiness check for the 2026 retraining cycle."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

def evaluate() -> dict:
    stats_path = ROOT / ".claude/codex/evidence/market-data-statistics.json"
    evidence_path = ROOT / "config/quant_evidence/assets.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    raw_assets = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
    assets = raw_assets.get("assets", {}) if isinstance(raw_assets, dict) else {}
    checks = {
        "statistics_audit": stats.get("decision") == "PASS",
        "pit_vintages": all(v.get("data_pit") is True for v in assets.values()) if isinstance(assets, dict) else False,
        "oos_evidence": all(bool(v.get("oos_manifest")) and int(v.get("trial_count", 0)) > 0 for v in assets.values()) if isinstance(assets, dict) else False,
        "promotion_eligible": all(v.get("status") == "approved" for v in assets.values()) if isinstance(assets, dict) else False,
    }
    return {"schema_version": 1, "cycle": "2026", "checks": checks,
            "ready": all(checks.values()),
            "decision": "GO" if all(checks.values()) else "NO-GO",
            "reason": "No se puede afirmar rentabilidad ni reentrenamiento listo sin PIT, OOS y promoción por activo."}

if __name__ == "__main__":
    out = ROOT / ".claude/codex/evidence/retraining-readiness-2026.json"
    result = evaluate()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["ready"] else 1)
