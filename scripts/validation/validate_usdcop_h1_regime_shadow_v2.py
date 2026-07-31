"""Register H1 regime shadow v2 only after integrity and parity checks pass."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys

import pandas as pd
import sklearn
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline.generate_usdcop_h1_regime_shadow_v2 import (
    build_live_frame,
    make_prediction,
    output_paths,
)


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v2.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = output_paths(contract)
    errors: list[str] = []
    if contract["_meta"].get("signal_authorized") is not False:
        errors.append("signal_authorized must remain false")
    if contract["_meta"].get("capital_authorized") is not False:
        errors.append("capital_authorized must remain false")
    protocol = contract["prospective_protocol"]
    if protocol["first_eligible_iso_week"] <= protocol["latest_research_audit_week"]:
        errors.append("first prospective week must follow latest research audit")
    if protocol.get("commit_current_iso_week_only") is not True:
        errors.append("current-week-only commit is mandatory")
    if protocol.get("missed_week_policy") != "MISS_PERMANENTLY_NEVER_BACKFILL":
        errors.append("missed weeks must never be backfilled")
    if protocol.get("require_unmatured_target_at_commit") is not True:
        errors.append("target must be unresolved at commit")
    if contract["model"].get("class_weight") != "balanced":
        errors.append("live model must be the selected balanced logit")
    if contract["regime_gate"].get("fallback_direction") is not None:
        errors.append("disallowed regimes must remain FLAT")
    if int(protocol["minimum_calendar_weeks_before_promotion_review"]) < 104:
        errors.append("promotion review requires at least 104 calendar weeks")
    if int(protocol["minimum_matured_signals_before_promotion_review"]) < 100:
        errors.append("promotion review requires at least 100 matured signals")
    for ledger_name in ("predictions", "outcomes"):
        ledger = paths[ledger_name]
        if ledger.exists() and ledger.stat().st_size:
            errors.append(f"cannot register after {ledger_name} ledger has records")

    code_paths = [
        str(contract["implementation_integrity"]["generator"]),
        *map(str, contract["implementation_integrity"]["hashed_dependencies"]),
    ]
    code_hashes: dict[str, str] = {}
    for relative in code_paths:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing registered code: {relative}")
        else:
            code_hashes[relative] = sha256(path)
    baseline_paths = [
        str(contract["data"]["baseline_deep_file"]),
        str(contract["data"]["baseline_current_file"]),
        str(contract["data"]["baseline_manifest"]),
    ]
    baseline_hashes: dict[str, str] = {}
    for relative in baseline_paths:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing immutable baseline: {relative}")
        else:
            baseline_hashes[relative] = sha256(path)
    if baseline_hashes.get(str(contract["data"]["baseline_deep_file"])) != contract["data"]["baseline_deep_sha256"]:
        errors.append("deep baseline hash does not match contract")
    if baseline_hashes.get(str(contract["data"]["baseline_current_file"])) != contract["data"]["baseline_current_sha256"]:
        errors.append("current baseline hash does not match contract")

    parity: dict[str, object] = {}
    try:
        frame, _ = build_live_frame(
            contract, pd.Timestamp(contract["data"]["baseline_cutoff"])
        )
        parity_spec = contract["implementation_integrity"]["parity_reference"]
        parity_date = pd.Timestamp(parity_spec["origin_date"])
        matches = frame.index[pd.to_datetime(frame["date"]).eq(parity_date)]
        if len(matches) != 1:
            raise RuntimeError("parity origin missing from immutable baseline")
        live = make_prediction(frame, int(matches[0]), contract)
        artifact = ROOT / parity_spec["research_artifact"]
        if artifact.exists():
            if sha256(artifact) != parity_spec["research_artifact_sha256"]:
                raise RuntimeError("research parity artifact hash drift")
            research = pd.read_csv(artifact)
            expected_rows = research[
                research["horizon_days"].eq(1)
                & research["origin_date"].str.startswith(str(parity_spec["origin_date"]))
            ]
            if len(expected_rows) != 1:
                raise RuntimeError("research parity row is not unique")
            expected = float(expected_rows.iloc[0]["probability_up"])
        else:
            expected = float(parity_spec["probability_up"])
        delta = abs(float(live["probability_up"]) - expected)
        parity = {
            "origin_date": parity_date.date().isoformat(),
            "model_variant_key": live["model_variant_key"],
            "live_probability_up": live["probability_up"],
            "research_probability_up": expected,
            "absolute_delta": delta,
            "exact_match": delta == 0.0,
            "reference_artifact_sha256": parity_spec["research_artifact_sha256"],
            "reference_artifact_available_in_runtime": artifact.exists(),
        }
        if live["model_variant_key"] != parity_spec["model_variant_key"]:
            errors.append("live model variant key does not match parity reference")
        if delta != 0.0:
            errors.append(f"live/research probability parity failed: {delta}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"parity check failed: {exc}")

    if errors:
        raise SystemExit("V2 registration failed:\n- " + "\n- ".join(errors))
    registration = {
        "schema_version": "2.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": sha256(CONFIG),
        "code_sha256": code_hashes,
        "baseline_data_sha256": baseline_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "parity_check": parity,
        "first_eligible_iso_week": protocol["first_eligible_iso_week"],
        "prediction_records_at_registration": 0,
        "outcome_records_at_registration": 0,
        "signal_authorized": False,
        "capital_authorized": False,
        "registration_valid": True,
    }
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["registration"].write_text(
        json.dumps(registration, indent=2), encoding="utf-8"
    )
    print(json.dumps(registration, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
