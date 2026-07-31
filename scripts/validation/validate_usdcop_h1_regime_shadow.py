"""Validate and hash the prospective USD/COP H1 regime-shadow contract."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v1.yaml"
OUTPUT = ROOT / "reports/usdcop_h1_regime_shadow_registration.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    if contract["_meta"]["signal_authorized"] is not False:
        errors.append("signal_authorized must remain false")
    if contract["_meta"]["capital_authorized"] is not False:
        errors.append("capital_authorized must remain false")
    protocol = contract["prospective_protocol"]
    if protocol["first_eligible_iso_week"] <= protocol["latest_research_audit_week"]:
        errors.append("prospective week must be strictly after the research audit")
    if protocol["allow_retrospective_overwrite"] is not False:
        errors.append("retrospective overwrite must be disabled")
    if int(protocol["minimum_matured_signals_before_promotion_review"]) < 100:
        errors.append("promotion review requires at least 100 matured signals")
    if int(protocol["minimum_calendar_weeks_before_promotion_review"]) < 104:
        errors.append("promotion review requires at least 104 calendar weeks")
    if contract["regime_gate"].get("fallback_direction") is not None:
        errors.append("disallowed regimes cannot force a fallback direction")
    if contract["governance"]["current_model_can_be_promoted_without_new_evidence"] is not False:
        errors.append("current evidence cannot authorize promotion")
    source_hashes: dict[str, str] = {}
    for key in ("deep_history_file", "current_history_file"):
        path = ROOT / contract["data"][key]
        if not path.exists():
            errors.append(f"missing data source: {path}")
        else:
            source_hashes[key] = sha256(path)
    if errors:
        raise SystemExit("Shadow registration failed:\n- " + "\n- ".join(errors))
    registration = {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(CONFIG.relative_to(ROOT)),
        "contract_sha256": sha256(CONFIG),
        "source_sha256": source_hashes,
        "first_eligible_iso_week": protocol["first_eligible_iso_week"],
        "signal_authorized": False,
        "capital_authorized": False,
        "registration_valid": True,
    }
    OUTPUT.write_text(json.dumps(registration, indent=2), encoding="utf-8")
    print(json.dumps(registration, indent=2))


if __name__ == "__main__":
    main()
