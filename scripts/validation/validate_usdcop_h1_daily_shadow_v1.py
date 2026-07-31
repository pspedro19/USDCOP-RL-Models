"""Register the daily H1 shadow after causality and implementation checks."""
from __future__ import annotations

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

from scripts.pipeline.generate_usdcop_h1_daily_shadow_v1 import (  # noqa: E402
    build_live_frame,
    make_daily_prediction,
    output_paths,
    sha256_file,
)


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_daily_shadow_v1.yaml"
PARENT_CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v2.yaml"


def main() -> int:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    parent = yaml.safe_load(PARENT_CONFIG.read_text(encoding="utf-8"))
    paths = output_paths(contract)
    errors: list[str] = []
    if contract["_meta"].get("signal_authorized") is not False:
        errors.append("signal_authorized must remain false")
    if contract["_meta"].get("capital_authorized") is not False:
        errors.append("capital_authorized must remain false")
    protocol = contract["prospective_protocol"]
    if protocol.get("missed_session_policy") != "MISS_PERMANENTLY_NEVER_BACKFILL":
        errors.append("missed daily sessions must never be backfilled")
    if protocol.get("commit_current_local_date_only") is not True:
        errors.append("current-local-date-only commit is mandatory")
    if protocol.get("model_retraining_frequency") != "once_per_iso_week":
        errors.append("daily experiment must retain weekly retraining")
    if int(protocol["minimum_committed_sessions_before_review"]) < 252:
        errors.append("review requires at least 252 committed sessions")
    if int(protocol["minimum_matured_selective_signals_before_review"]) < 100:
        errors.append("review requires at least 100 selective signals")
    if int(protocol["minimum_elapsed_calendar_months_before_review"]) < 12:
        errors.append("review requires at least 12 elapsed months")
    if contract["pre_registration_evidence"].get("exact_daily_origin_rule_historically_evaluated") is not False:
        errors.append("daily-origin historical performance must remain unopened")
    if contract["trial_accounting"].get("new_primary_trials") != 1:
        errors.append("daily transport must account for exactly one new primary trial")

    exact_model_fields = (
        "family", "feature_group", "c", "class_weight", "max_iter",
        "random_state", "half_life_sessions", "imputer", "scaler",
        "probability_threshold", "implementation", "features",
    )
    for field in exact_model_fields:
        if contract["model"].get(field) != parent["model"].get(field):
            errors.append(f"daily model drifted from weekly v2: {field}")
    if contract["regime_gate"] != parent["regime_gate"]:
        errors.append("daily regime gate drifted from weekly v2")

    for ledger_name in ("predictions", "outcomes"):
        ledger = paths[ledger_name]
        if ledger.exists() and ledger.stat().st_size:
            errors.append(f"cannot register after {ledger_name} ledger has records")

    integrity = contract["implementation_integrity"]
    code_paths = [
        str(integrity["generator"]),
        str(integrity["evaluator"]),
        str(integrity["validator"]),
        *map(str, integrity["hashed_dependencies"]),
    ]
    code_hashes: dict[str, str] = {}
    for relative in code_paths:
        path = ROOT / relative
        if not path.exists():
            errors.append(f"missing registered code: {relative}")
        else:
            code_hashes[relative] = sha256_file(path)
    generator_source = (ROOT / integrity["generator"]).read_text(encoding="utf-8")
    if "--as-of" in generator_source or "as_of_datetime" in generator_source:
        errors.append("manual clock override exists in production generator")

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
            baseline_hashes[relative] = sha256_file(path)
    expected_hashes = {
        str(contract["data"]["baseline_deep_file"]): contract["data"]["baseline_deep_sha256"],
        str(contract["data"]["baseline_current_file"]): contract["data"]["baseline_current_sha256"],
        str(contract["data"]["baseline_manifest"]): contract["data"]["baseline_manifest_sha256"],
    }
    for relative, expected in expected_hashes.items():
        if baseline_hashes.get(relative) != expected:
            errors.append(f"baseline hash mismatch: {relative}")

    weekly_retraining_check: dict[str, object] = {}
    try:
        frame, provenance = build_live_frame(
            contract, pd.Timestamp(contract["data"]["baseline_cutoff"])
        )
        predictions: dict[str, dict[str, object]] = {}
        for origin_text in ("2024-12-23", "2024-12-27", "2024-12-30"):
            matches = frame.index[pd.to_datetime(frame["date"]).eq(pd.Timestamp(origin_text))]
            if len(matches) != 1:
                raise RuntimeError(f"mechanical parity origin missing: {origin_text}")
            prediction = make_daily_prediction(frame, int(matches[0]), contract)
            predictions[origin_text] = {
                "probability_up": prediction["probability_up"],
                "weekly_training_anchor": prediction["weekly_training_anchor"],
                "weekly_model_sha256": prediction["weekly_model_sha256"],
                "train_label_end": prediction["train_label_end"],
                "model_variant_key": prediction["model_variant_key"],
            }
        same_week_hash = predictions["2024-12-23"]["weekly_model_sha256"]
        if predictions["2024-12-27"]["weekly_model_sha256"] != same_week_hash:
            errors.append("model fingerprint changed within the same ISO week")
        if predictions["2024-12-30"]["weekly_model_sha256"] == same_week_hash:
            errors.append("model fingerprint did not refresh for the next ISO week")
        weekly_retraining_check = {
            "exact_parent_model_fields": list(exact_model_fields),
            "same_week_model_stable": predictions["2024-12-27"]["weekly_model_sha256"] == same_week_hash,
            "next_week_model_refreshed": predictions["2024-12-30"]["weekly_model_sha256"] != same_week_hash,
            "mechanical_predictions_only_no_accuracy_opened": predictions,
            "baseline_frame_end": provenance["frame_end"],
        }
    except Exception as exc:  # noqa: BLE001
        errors.append(f"weekly retraining mechanical check failed: {exc}")

    if errors:
        raise SystemExit("Daily H1 registration failed:\n- " + "\n- ".join(errors))
    registration = {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(CONFIG.relative_to(ROOT)).replace("\\", "/"),
        "contract_sha256": sha256_file(CONFIG),
        "code_sha256": code_hashes,
        "baseline_data_sha256": baseline_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "weekly_retraining_mechanical_check": weekly_retraining_check,
        "historical_daily_accuracy_opened_at_registration": False,
        "first_eligible_date": protocol["first_eligible_date"],
        "prediction_records_at_registration": 0,
        "outcome_records_at_registration": 0,
        "directional_trials_after_registration": contract["trial_accounting"]["directional_trials_after_registration"],
        "global_trials_after_registration": contract["trial_accounting"]["global_trials_after_registration"],
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
