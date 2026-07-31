"""Fail-closed prospective USD/COP H1 regime-shadow v2.

Predictions are committed only for the current Friday, while the H1 target is
still unresolved.  Missed weeks are never reconstructed.  Every prediction is
bound to an immutable processed-frame snapshot, registered code hashes and an
append-only SHA-256 chain.  Outcomes live in a separate append-only chain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, time
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import sklearn
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_causal_regime_gate import build_states
from scripts.analysis.usdcop_directional_edge_tournament import (
    Variant,
    eligible_train_indices,
    fit_model,
    make_direction_target,
)
from scripts.analysis.usdcop_long_history_directional_tournament import (
    build_long_history_frame,
    build_price_features,
)


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v2.yaml"
GENESIS_HASH = "0" * 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_hash(record: dict[str, Any], hash_field: str) -> str:
    payload = {key: value for key, value in record.items() if key != hash_field}
    raw = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def validate_hash_chain(
    records: list[dict[str, Any]],
    *,
    hash_field: str,
    previous_field: str,
) -> str:
    previous = GENESIS_HASH
    seen_weeks: set[str] = set()
    for position, record in enumerate(records):
        week = str(record.get("iso_week"))
        if week in seen_weeks:
            raise ValueError(f"Duplicate ISO week in hash chain: {week}")
        seen_weeks.add(week)
        if record.get(previous_field) != previous:
            raise ValueError(f"Broken previous hash at chain position {position}")
        expected = canonical_hash(record, hash_field)
        if record.get(hash_field) != expected:
            raise ValueError(f"Tampered record at chain position {position}")
        previous = expected
    return previous


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(json_safe(record), sort_keys=True, ensure_ascii=False)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def output_paths(contract: dict[str, Any]) -> dict[str, Path]:
    output = contract["outputs"]
    directory = ROOT / output["directory"]
    return {
        "directory": directory,
        "registration": ROOT / contract["implementation_integrity"]["registration_file"],
        "predictions": directory / output["prediction_ledger"],
        "outcomes": directory / output["outcome_ledger"],
        "snapshots": directory / output["snapshot_directory"],
        "index": ROOT / output["index"],
    }


def load_registered_contract() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = output_paths(contract)
    if not paths["registration"].exists():
        raise FileNotFoundError("V2 registration is missing")
    registration = json.loads(paths["registration"].read_text(encoding="utf-8"))
    contract_hash = sha256_file(CONFIG)
    if registration.get("contract_sha256") != contract_hash:
        raise RuntimeError("Contract hash drift; create a new experiment id")
    if registration.get("registration_valid") is not True:
        raise RuntimeError("Registration is not valid")
    if contract["_meta"].get("signal_authorized") is not False:
        raise RuntimeError("Shadow contract must keep signal_authorized=false")
    if contract["_meta"].get("capital_authorized") is not False:
        raise RuntimeError("Shadow contract must keep capital_authorized=false")
    for relative, expected in registration["code_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Registered code drift: {relative}")
    for relative, expected in registration["baseline_data_sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Registered baseline drift: {relative}")
    runtime = registration["runtime_versions"]
    if runtime.get("python") != f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}":
        raise RuntimeError("Registered Python version drift")
    if runtime.get("pandas") != pd.__version__:
        raise RuntimeError("Registered pandas version drift")
    if runtime.get("scikit_learn") != sklearn.__version__:
        raise RuntimeError("Registered scikit-learn version drift")
    code_bundle_hash = hashlib.sha256(
        json.dumps(registration["code_sha256"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    return contract, registration, contract_hash, code_bundle_hash


def _validate_ohlc(frame: pd.DataFrame) -> None:
    columns = ["open", "high", "low", "close"]
    if frame[columns].isna().any().any() or frame[columns].le(0).any().any():
        raise ValueError("Daily frame contains null or non-positive OHLC")
    if not (
        frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()
        and frame["low"].le(frame[["open", "close"]].min(axis=1)).all()
    ):
        raise ValueError("Daily frame violates OHLC ordering")


def build_live_frame(
    contract: dict[str, Any], as_of_date: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = contract["data"]
    deep_path = ROOT / data["baseline_deep_file"]
    current_path = ROOT / data["baseline_current_file"]
    baseline, baseline_provenance = build_long_history_frame(deep_path, current_path)
    cutoff = pd.Timestamp(data["baseline_cutoff"])
    baseline = baseline[pd.to_datetime(baseline["date"]).le(cutoff)].copy()
    if pd.Timestamp(baseline["date"].max()) != cutoff:
        raise RuntimeError("Immutable baseline does not end at registered cutoff")

    live_path = ROOT / data["live_current_file"]
    live = pd.read_parquet(live_path).copy()
    live["date"] = pd.to_datetime(
        live["time"], utc=True,
    ).dt.tz_localize(None).dt.normalize()
    live = live[
        live["date"].gt(cutoff)
        & live["date"].le(pd.Timestamp(as_of_date).normalize())
    ].copy()
    for column in ("open", "high", "low", "close"):
        live[column] = pd.to_numeric(live[column], errors="coerce")
    live["research_source"] = "live_current_first_seen_snapshot"
    live = live.sort_values("date")
    if live["date"].duplicated().any():
        raise ValueError("Live daily source has duplicate post-baseline dates")
    columns = ["date", "open", "high", "low", "close", "research_source"]
    raw = pd.concat([baseline[columns], live[columns]], ignore_index=True)
    raw = raw.sort_values("date").reset_index(drop=True)
    if raw["date"].duplicated().any():
        raise ValueError("Assembled shadow history has duplicate dates")
    _validate_ohlc(raw)
    frame = build_price_features(raw)
    provenance = {
        "baseline": baseline_provenance,
        "baseline_cutoff": cutoff.date().isoformat(),
        "live_source_path": str(live_path.relative_to(ROOT)).replace("\\", "/"),
        "live_source_sha256": sha256_file(live_path),
        "live_rows_after_cutoff": int(len(live)),
        "frame_rows": int(len(frame)),
        "frame_start": pd.Timestamp(frame["date"].min()).date().isoformat(),
        "frame_end": pd.Timestamp(frame["date"].max()).date().isoformat(),
    }
    return frame, provenance


def iso_week(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%G-W%V")


def commit_window_open(now: datetime, contract: dict[str, Any]) -> bool:
    protocol = contract["prospective_protocol"]
    hour, minute = map(int, str(protocol["commit_not_before_local_time"]).split(":"))
    return now.weekday() == int(protocol["commit_weekday"]) and now.time() >= time(hour, minute)


def validate_current_commit(
    frame: pd.DataFrame,
    now: datetime,
    contract: dict[str, Any],
) -> tuple[int, str]:
    origin_index = int(frame.index[-1])
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    current_week = iso_week(pd.Timestamp(now.date()))
    origin_week = iso_week(origin_date)
    if origin_week != current_week:
        raise RuntimeError("Current-week-only rule rejected stale origin data")
    if bool(contract["prospective_protocol"]["require_origin_date_equals_local_commit_date"]):
        if origin_date.date() != now.date():
            raise RuntimeError("Origin date is not the local commit date")
    first_week = str(contract["prospective_protocol"]["first_eligible_iso_week"])
    if current_week < first_week:
        raise RuntimeError("Current week precedes prospective registration")
    horizon = int(contract["prospective_protocol"]["target_horizon_sessions"])
    if origin_index + horizon < len(frame):
        raise RuntimeError("Target already matured; retroactive commit forbidden")
    return origin_index, current_week


def ensure_snapshot(
    frame: pd.DataFrame,
    provenance: dict[str, Any],
    week: str,
    paths: dict[str, Path],
    contract_hash: str,
    code_bundle_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = paths["snapshots"] / week
    frame_path = directory / "processed_frame.parquet"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() or frame_path.exists():
        if not (manifest_path.exists() and frame_path.exists()):
            raise RuntimeError(f"Incomplete immutable snapshot for {week}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("frame_sha256") != sha256_file(frame_path):
            raise RuntimeError(f"Snapshot hash mismatch for {week}")
        if manifest.get("contract_sha256") != contract_hash:
            raise RuntimeError(f"Snapshot contract mismatch for {week}")
        return pd.read_parquet(frame_path), manifest
    directory.mkdir(parents=True, exist_ok=False)
    temporary = directory / "processed_frame.parquet.tmp"
    frame.to_parquet(temporary, index=False)
    temporary.replace(frame_path)
    manifest = {
        "schema_version": "1.0.0",
        "iso_week": week,
        "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
        "frame_sha256": sha256_file(frame_path),
        "frame_rows": int(len(frame)),
        "frame_end": pd.Timestamp(frame["date"].max()).date().isoformat(),
        "contract_sha256": contract_hash,
        "code_bundle_sha256": code_bundle_hash,
        "source_provenance": provenance,
    }
    temp_manifest = directory / "manifest.json.tmp"
    temp_manifest.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    temp_manifest.replace(manifest_path)
    return frame, manifest


def make_prediction(
    frame: pd.DataFrame,
    origin_index: int,
    contract: dict[str, Any],
) -> dict[str, Any]:
    model_spec = contract["model"]
    horizon = int(contract["prospective_protocol"]["target_horizon_sessions"])
    target = make_direction_target(frame, horizon)
    train = eligible_train_indices(target, origin_index, horizon)
    variant = Variant(
        str(model_spec["feature_group"]),
        float(model_spec["c"]),
        str(model_spec["class_weight"]),
        int(model_spec["half_life_sessions"]),
    )
    features = [str(value) for value in model_spec["features"]]
    fitted = fit_model(frame, target, features, train, variant)
    logistic = fitted.named_steps["logisticregression"]
    if logistic.class_weight != "balanced":
        raise RuntimeError("Live model does not reproduce registered balanced logit")
    probability = float(
        fitted.predict_proba(frame.loc[[origin_index], features])[0, 1]
    )
    threshold = float(model_spec["probability_threshold"])
    base_direction = "UP" if probability >= threshold else "DOWN"
    states = build_states(frame)
    regime = str(states.loc[origin_index, "regime"])
    allowed = regime in set(map(str, contract["regime_gate"]["allowed_states"]))
    decision = base_direction if allowed else "FLAT"
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    label_end_index = int(train[-1]) + horizon
    train_up_rate = float(target.iloc[train].mean())
    return {
        "iso_week": iso_week(origin_date),
        "origin_date": origin_date.date().isoformat(),
        "target_date_estimated": (origin_date + pd.offsets.BDay(horizon)).date().isoformat(),
        "base_price": float(frame.loc[origin_index, "close"]),
        "probability_up": probability,
        "probability_confidence": abs(probability - 0.5) * 2.0,
        "threshold": threshold,
        "base_direction": base_direction,
        "regime": regime,
        "trend_z": float(states.loc[origin_index, "trend_z"]),
        "regime_allowed": allowed,
        "decision": decision,
        "train_up_rate": train_up_rate,
        "causal_majority_direction": "UP" if train_up_rate >= 0.5 else "DOWN",
        "train_count": int(len(train)),
        "train_label_end": pd.Timestamp(frame.loc[label_end_index, "date"]).date().isoformat(),
        "model_variant_key": variant.key,
        "signal_authorized": False,
        "capital_authorized": False,
    }


def append_prediction_record(
    base_record: dict[str, Any],
    predictions: list[dict[str, Any]],
    path: Path,
) -> dict[str, Any]:
    previous = validate_hash_chain(
        predictions,
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    )
    if any(str(item["iso_week"]) == str(base_record["iso_week"]) for item in predictions):
        raise RuntimeError(f"Prediction already committed for {base_record['iso_week']}")
    record = {**base_record, "previous_prediction_sha256": previous}
    record["prediction_record_sha256"] = canonical_hash(
        record, "prediction_record_sha256"
    )
    append_jsonl(path, record)
    predictions.append(record)
    return record


def append_new_outcomes(
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    frame: pd.DataFrame,
    path: Path,
    observed_at: datetime,
    live_source_sha256: str,
) -> list[dict[str, Any]]:
    previous = validate_hash_chain(
        outcomes,
        hash_field="outcome_record_sha256",
        previous_field="previous_outcome_sha256",
    )
    completed = {str(item["prediction_record_sha256"]) for item in outcomes}
    dates = pd.to_datetime(frame["date"])
    date_to_index = {value.date(): int(index) for index, value in enumerate(dates)}
    for prediction in predictions:
        prediction_hash = str(prediction["prediction_record_sha256"])
        if prediction_hash in completed:
            continue
        origin_index = date_to_index.get(pd.Timestamp(prediction["origin_date"]).date())
        if origin_index is None or origin_index + 1 >= len(frame):
            continue
        target_price = float(frame.loc[origin_index + 1, "close"])
        actual = int(target_price > float(prediction["base_price"]))
        decision = str(prediction["decision"])
        majority = str(prediction["causal_majority_direction"])
        outcome = {
            "iso_week": prediction["iso_week"],
            "prediction_record_sha256": prediction_hash,
            "target_date": pd.Timestamp(frame.loc[origin_index + 1, "date"]).date().isoformat(),
            "actual_price": target_price,
            "actual": actual,
            "hit": None if decision == "FLAT" else bool((decision == "UP") == bool(actual)),
            "causal_majority_hit": bool((majority == "UP") == bool(actual)),
            "observed_at": observed_at.isoformat(),
            "outcome_source_sha256": live_source_sha256,
            "previous_outcome_sha256": previous,
        }
        outcome["outcome_record_sha256"] = canonical_hash(
            outcome, "outcome_record_sha256"
        )
        append_jsonl(path, outcome)
        outcomes.append(outcome)
        previous = str(outcome["outcome_record_sha256"])
        completed.add(prediction_hash)
    return outcomes


def metric_summary(
    predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    outcome_by_hash = {
        str(item["prediction_record_sha256"]): item for item in outcomes
    }
    matured = [
        (prediction, outcome_by_hash[str(prediction["prediction_record_sha256"])])
        for prediction in predictions
        if str(prediction["prediction_record_sha256"]) in outcome_by_hash
    ]
    signals = [pair for pair in matured if pair[0]["decision"] != "FLAT"]
    summary: dict[str, Any] = {
        "committed_weeks": len(predictions),
        "matured_weeks": len(matured),
        "matured_signals": len(signals),
        "coverage": len(signals) / len(matured) if matured else None,
        "directional_accuracy": None,
        "selective_risk": None,
        "balanced_accuracy": None,
        "up_recall": None,
        "down_recall": None,
        "prediction_up_rate": None,
        "lift_vs_causal_majority": None,
        "brier_all_committed_predictions": None,
    }
    if matured:
        probabilities = np.asarray([pair[0]["probability_up"] for pair in matured], dtype=float)
        actual_all = np.asarray([pair[1]["actual"] for pair in matured], dtype=int)
        summary["brier_all_committed_predictions"] = float(
            np.mean((probabilities - actual_all) ** 2)
        )
    if not signals:
        return summary
    actual = np.asarray([pair[1]["actual"] for pair in signals], dtype=int)
    predicted = np.asarray([pair[0]["decision"] == "UP" for pair in signals], dtype=int)
    hits = predicted == actual
    up = actual == 1
    down = actual == 0
    up_recall = float(predicted[up].mean()) if up.any() else None
    down_recall = float((1 - predicted[down]).mean()) if down.any() else None
    majority_hits = np.asarray(
        [pair[1]["causal_majority_hit"] for pair in signals], dtype=bool
    )
    da = float(hits.mean())
    summary.update({
        "directional_accuracy": da,
        "selective_risk": 1.0 - da,
        "balanced_accuracy": (
            0.5 * (up_recall + down_recall)
            if up_recall is not None and down_recall is not None else None
        ),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "prediction_up_rate": float(predicted.mean()),
        "lift_vs_causal_majority": da - float(majority_hits.mean()),
    })
    return summary


def build_index(
    contract: dict[str, Any],
    registration: dict[str, Any],
    contract_hash: str,
    code_bundle_hash: str,
    predictions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    provenance: dict[str, Any],
    generated_at: datetime,
) -> dict[str, Any]:
    protocol = contract["prospective_protocol"]
    summary = metric_summary(predictions, outcomes)
    sample_review_eligible = bool(
        summary["committed_weeks"] >= int(protocol["minimum_calendar_weeks_before_promotion_review"])
        and summary["matured_signals"] >= int(protocol["minimum_matured_signals_before_promotion_review"])
    )
    outcome_by_hash = {
        str(item["prediction_record_sha256"]): item for item in outcomes
    }
    records = []
    for prediction in predictions:
        records.append({
            **prediction,
            "outcome": outcome_by_hash.get(str(prediction["prediction_record_sha256"])),
        })
    return {
        "schema_version": "2.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "supersedes": contract["_meta"]["supersedes"],
        "generated_at": generated_at.isoformat(),
        "data_cutoff": provenance["frame_end"],
        "first_eligible_iso_week": protocol["first_eligible_iso_week"],
        "status": "awaiting_first_commit" if not predictions else "collecting_prospective_evidence",
        "registration_valid": bool(registration["registration_valid"]),
        "contract_sha256": contract_hash,
        "code_bundle_sha256": code_bundle_hash,
        "prediction_chain_head": (
            predictions[-1]["prediction_record_sha256"] if predictions else GENESIS_HASH
        ),
        "outcome_chain_head": (
            outcomes[-1]["outcome_record_sha256"] if outcomes else GENESIS_HASH
        ),
        "signal_authorized": False,
        "capital_authorized": False,
        "sample_review_eligible": sample_review_eligible,
        "promotion_review_eligible": False,
        "summary": summary,
        "provenance": provenance,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    contract, registration, contract_hash, code_bundle_hash = load_registered_contract()
    timezone = ZoneInfo(str(contract["asset"]["timezone"]))
    now = datetime.now(timezone)
    frame, provenance = build_live_frame(contract, pd.Timestamp(now.date()))
    paths = output_paths(contract)
    predictions = load_jsonl(paths["predictions"])
    outcomes = load_jsonl(paths["outcomes"])
    validate_hash_chain(
        predictions,
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    )
    validate_hash_chain(
        outcomes,
        hash_field="outcome_record_sha256",
        previous_field="previous_outcome_sha256",
    )
    if args.validate_only:
        print(json.dumps({
            "registration_valid": True,
            "contract_sha256": contract_hash,
            "code_bundle_sha256": code_bundle_hash,
            "prediction_chain_records": len(predictions),
            "outcome_chain_records": len(outcomes),
        }, indent=2))
        return 0

    outcomes = append_new_outcomes(
        predictions,
        outcomes,
        frame,
        paths["outcomes"],
        now,
        provenance["live_source_sha256"],
    )
    current_week = iso_week(pd.Timestamp(now.date()))
    already_committed = any(
        str(record["iso_week"]) == current_week for record in predictions
    )
    if commit_window_open(now, contract) and not already_committed:
        origin_index, week = validate_current_commit(frame, now, contract)
        snapshot_frame, snapshot_manifest = ensure_snapshot(
            frame,
            provenance,
            week,
            paths,
            contract_hash,
            code_bundle_hash,
        )
        origin_index = int(snapshot_frame.index[-1])
        base_record = make_prediction(snapshot_frame, origin_index, contract)
        base_record.update({
            "experiment_id": contract["_meta"]["experiment_id"],
            "committed_at": now.isoformat(),
            "contract_sha256": contract_hash,
            "code_bundle_sha256": code_bundle_hash,
            "input_snapshot_sha256": snapshot_manifest["frame_sha256"],
        })
        append_prediction_record(base_record, predictions, paths["predictions"])

    index = build_index(
        contract,
        registration,
        contract_hash,
        code_bundle_hash,
        predictions,
        outcomes,
        provenance,
        now,
    )
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["index"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(index), indent=2), encoding="utf-8")
    temporary.replace(paths["index"])
    print(json.dumps({
        "experiment_id": index["experiment_id"],
        "status": index["status"],
        "data_cutoff": index["data_cutoff"],
        "committed_weeks": index["summary"]["committed_weeks"],
        "matured_signals": index["summary"]["matured_signals"],
        "prediction_chain_head": index["prediction_chain_head"],
        "capital_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
