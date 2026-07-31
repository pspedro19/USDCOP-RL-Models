"""Prospective daily-origin USD/COP H1 shadow with weekly retraining.

The exact H1 v2 estimator is refit once per ISO week using the last session of
the prior week as its training anchor.  A prediction can only be committed for
the current completed session.  Missed sessions are never reconstructed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, time
import hashlib
import json
import math
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

from scripts.analysis.usdcop_causal_regime_gate import build_states  # noqa: E402
from scripts.analysis.usdcop_directional_edge_tournament import (  # noqa: E402
    Variant,
    eligible_train_indices,
    fit_model,
    make_direction_target,
)
from scripts.pipeline.generate_usdcop_h1_regime_shadow_v2 import (  # noqa: E402
    append_jsonl,
    build_live_frame,
    canonical_hash,
    json_safe,
    load_jsonl,
    sha256_file,
)


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_daily_shadow_v1.yaml"
GENESIS_HASH = "0" * 64


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


def validate_daily_hash_chain(
    records: list[dict[str, Any]], *, hash_field: str, previous_field: str,
) -> str:
    previous = GENESIS_HASH
    seen: set[str] = set()
    last_date: str | None = None
    for position, record in enumerate(records):
        origin = str(record.get("origin_date"))
        if not origin or origin == "None":
            raise ValueError(f"Missing origin_date at chain position {position}")
        if origin in seen:
            raise ValueError(f"Duplicate origin_date in hash chain: {origin}")
        if last_date is not None and origin <= last_date:
            raise ValueError("Daily hash chain is not strictly chronological")
        if record.get(previous_field) != previous:
            raise ValueError(f"Broken previous hash at chain position {position}")
        expected = canonical_hash(record, hash_field)
        if record.get(hash_field) != expected:
            raise ValueError(f"Tampered record at chain position {position}")
        seen.add(origin)
        last_date = origin
        previous = expected
    return previous


def load_registered_contract() -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = output_paths(contract)
    if not paths["registration"].exists():
        raise FileNotFoundError("Daily H1 shadow registration is missing")
    registration = json.loads(paths["registration"].read_text(encoding="utf-8"))
    contract_hash = sha256_file(CONFIG)
    if registration.get("contract_sha256") != contract_hash:
        raise RuntimeError("Contract hash drift; create a new experiment id")
    if registration.get("registration_valid") is not True:
        raise RuntimeError("Registration is invalid")
    if contract["_meta"].get("signal_authorized") is not False:
        raise RuntimeError("Daily shadow must keep signal_authorized=false")
    if contract["_meta"].get("capital_authorized") is not False:
        raise RuntimeError("Daily shadow must keep capital_authorized=false")
    for relative, expected in registration["code_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"Registered code drift: {relative}")
    for relative, expected in registration["baseline_data_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
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


def iso_week(value: pd.Timestamp | date) -> str:
    return pd.Timestamp(value).strftime("%G-W%V")


def commit_window_open(now: datetime, contract: dict[str, Any]) -> bool:
    protocol = contract["prospective_protocol"]
    hour, minute = map(int, protocol["commit_not_before_local_time"].split(":"))
    return now.weekday() in set(map(int, protocol["commit_weekdays"])) and now.time() >= time(hour, minute)


def weekly_training_anchor(frame: pd.DataFrame, origin_index: int) -> int:
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    origin_week = iso_week(origin_date)
    candidates = frame.index[
        pd.to_datetime(frame["date"]).map(iso_week).lt(origin_week)
    ]
    if len(candidates) == 0:
        raise RuntimeError("No prior ISO-week session is available as training anchor")
    anchor = int(candidates[-1])
    if anchor >= origin_index:
        raise RuntimeError("Weekly training anchor is not prior to origin")
    return anchor


def _variant(contract: dict[str, Any]) -> Variant:
    spec = contract["model"]
    return Variant(
        str(spec["feature_group"]),
        float(spec["c"]),
        str(spec["class_weight"]),
        int(spec["half_life_sessions"]),
    )


def model_fingerprint(
    fitted: Any,
    train_indices: np.ndarray,
    frame: pd.DataFrame,
    anchor_index: int,
    variant: Variant,
) -> str:
    logistic = fitted.named_steps["logisticregression"]
    imputer = fitted.named_steps["simpleimputer"]
    scaler = fitted.named_steps["standardscaler"]
    payload = {
        "variant": variant.key,
        "training_anchor": pd.Timestamp(frame.loc[anchor_index, "date"]).date().isoformat(),
        "train_count": int(len(train_indices)),
        "train_first": int(train_indices[0]),
        "train_last": int(train_indices[-1]),
        "classes": logistic.classes_.tolist(),
        "coef": logistic.coef_.tolist(),
        "intercept": logistic.intercept_.tolist(),
        "imputer_statistics": imputer.statistics_.tolist(),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    raw = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def make_daily_prediction(
    frame: pd.DataFrame, origin_index: int, contract: dict[str, Any],
) -> dict[str, Any]:
    horizon = int(contract["prospective_protocol"]["target_horizon_sessions"])
    target = make_direction_target(frame, horizon)
    anchor = weekly_training_anchor(frame, origin_index)
    train = eligible_train_indices(target, anchor, horizon)
    variant = _variant(contract)
    features = list(map(str, contract["model"]["features"]))
    fitted = fit_model(frame, target, features, train, variant)
    logistic = fitted.named_steps["logisticregression"]
    if logistic.class_weight != "balanced":
        raise RuntimeError("Daily live model is not the registered balanced logit")
    probability = float(fitted.predict_proba(frame.loc[[origin_index], features])[0, 1])
    threshold = float(contract["model"]["probability_threshold"])
    base_direction = "UP" if probability >= threshold else "DOWN"
    states = build_states(frame)
    regime = str(states.loc[origin_index, "regime"])
    allowed = regime in set(map(str, contract["regime_gate"]["allowed_states"]))
    decision = base_direction if allowed else "FLAT"
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    anchor_date = pd.Timestamp(frame.loc[anchor, "date"])
    label_end_index = int(train[-1]) + horizon
    train_up_rate = float(target.iloc[train].mean())
    return {
        "origin_date": origin_date.date().isoformat(),
        "iso_week": iso_week(origin_date),
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
        "weekly_training_anchor": anchor_date.date().isoformat(),
        "weekly_training_anchor_iso_week": iso_week(anchor_date),
        "weekly_model_sha256": model_fingerprint(fitted, train, frame, anchor, variant),
        "train_up_rate": train_up_rate,
        "causal_majority_direction": "UP" if train_up_rate >= 0.5 else "DOWN",
        "train_count": int(len(train)),
        "train_label_end": pd.Timestamp(frame.loc[label_end_index, "date"]).date().isoformat(),
        "model_variant_key": variant.key,
        "signal_authorized": False,
        "capital_authorized": False,
    }


def validate_current_commit(frame: pd.DataFrame, now: datetime, contract: dict[str, Any]) -> int:
    origin_index = int(frame.index[-1])
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    if origin_date.date() != now.date():
        raise RuntimeError("Current-date-only rule rejected stale daily origin")
    first_date = pd.Timestamp(contract["prospective_protocol"]["first_eligible_date"]).date()
    if now.date() < first_date:
        raise RuntimeError("Current date precedes prospective launch")
    horizon = int(contract["prospective_protocol"]["target_horizon_sessions"])
    if origin_index + horizon < len(frame):
        raise RuntimeError("Target already matured; retroactive daily commit forbidden")
    return origin_index


def load_current_m5_session(local_date: date, contract: dict[str, Any]) -> pd.DataFrame:
    import psycopg2

    password = os.environ.get("POSTGRES_PASSWORD")
    if not password:
        raise RuntimeError("POSTGRES_PASSWORD is required for M5 decision snapshot")
    connection = psycopg2.connect(
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        dbname=os.environ.get("POSTGRES_DB", "usdcop_trading"),
        user=os.environ.get("POSTGRES_USER", "admin"),
        password=password,
    )
    try:
        query = """
            SELECT time, symbol, open, high, low, close, volume
            FROM usdcop_m5_ohlcv
            WHERE symbol = %s
              AND (time AT TIME ZONE %s)::date = %s
            ORDER BY time
        """
        frame = pd.read_sql_query(
            query,
            connection,
            params=(contract["data"]["m5_symbol"], contract["asset"]["timezone"], local_date),
        )
    finally:
        connection.close()
    if frame.empty:
        raise RuntimeError("No current-session M5 bars are available")
    timestamps = pd.to_datetime(frame["time"], utc=True)
    if timestamps.duplicated().any():
        raise RuntimeError("Current-session M5 snapshot has duplicate timestamps")
    minimum = int(contract["data"]["minimum_m5_bars"])
    expected = int(contract["data"]["expected_m5_bars"])
    if len(frame) < minimum:
        raise RuntimeError(f"Current M5 session is incomplete: {len(frame)} < {minimum}")
    if len(frame) != expected:
        raise RuntimeError(f"Current M5 session has {len(frame)} bars; expected exactly {expected}")
    timezone = ZoneInfo(contract["asset"]["timezone"])
    local_times = timestamps.dt.tz_convert(timezone)
    expected_last = time.fromisoformat(contract["data"]["required_last_local_bar_time"])
    if local_times.iloc[-1].time().replace(tzinfo=None) < expected_last:
        raise RuntimeError("Current M5 session does not reach the registered final bar")
    return frame


def ensure_snapshot(
    frame: pd.DataFrame,
    m5_frame: pd.DataFrame,
    provenance: dict[str, Any],
    origin_date: str,
    paths: dict[str, Path],
    contract_hash: str,
    code_bundle_hash: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    directory = paths["snapshots"] / origin_date
    frame_path = directory / "processed_frame.parquet"
    m5_path = directory / "decision_session_m5.parquet"
    manifest_path = directory / "manifest.json"
    if any(path.exists() for path in (frame_path, m5_path, manifest_path)):
        if not all(path.exists() for path in (frame_path, m5_path, manifest_path)):
            raise RuntimeError(f"Incomplete immutable daily snapshot for {origin_date}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("processed_frame_sha256") != sha256_file(frame_path):
            raise RuntimeError("Daily processed-frame snapshot hash mismatch")
        if manifest.get("decision_session_m5_sha256") != sha256_file(m5_path):
            raise RuntimeError("Daily M5 snapshot hash mismatch")
        if manifest.get("contract_sha256") != contract_hash:
            raise RuntimeError("Daily snapshot contract hash mismatch")
        return pd.read_parquet(frame_path), manifest
    directory.mkdir(parents=True, exist_ok=False)
    frame_tmp = frame_path.with_suffix(".parquet.tmp")
    m5_tmp = m5_path.with_suffix(".parquet.tmp")
    frame.to_parquet(frame_tmp, index=False)
    m5_frame.to_parquet(m5_tmp, index=False)
    frame_tmp.replace(frame_path)
    m5_tmp.replace(m5_path)
    manifest = {
        "schema_version": "1.0.0",
        "experiment_id": "usdcop_h1_daily_shadow_v1",
        "origin_date": origin_date,
        "created_at_utc": datetime.now(ZoneInfo("UTC")).isoformat(),
        "processed_frame_sha256": sha256_file(frame_path),
        "decision_session_m5_sha256": sha256_file(m5_path),
        "frame_rows": int(len(frame)),
        "frame_end": pd.Timestamp(frame["date"].max()).date().isoformat(),
        "m5_rows": int(len(m5_frame)),
        "m5_first": pd.to_datetime(m5_frame["time"], utc=True).min().isoformat(),
        "m5_last": pd.to_datetime(m5_frame["time"], utc=True).max().isoformat(),
        "contract_sha256": contract_hash,
        "code_bundle_sha256": code_bundle_hash,
        "source_provenance": provenance,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    return frame, manifest


def append_prediction(
    base_record: dict[str, Any], predictions: list[dict[str, Any]], path: Path,
) -> dict[str, Any]:
    previous = validate_daily_hash_chain(
        predictions,
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    )
    if any(item["origin_date"] == base_record["origin_date"] for item in predictions):
        raise RuntimeError(f"Prediction already committed for {base_record['origin_date']}")
    record = {**base_record, "previous_prediction_sha256": previous}
    record["prediction_record_sha256"] = canonical_hash(record, "prediction_record_sha256")
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
    previous = validate_daily_hash_chain(
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
        actual_return = target_price / float(prediction["base_price"]) - 1.0
        actual = int(actual_return > 0.0)
        base_up = prediction["base_direction"] == "UP"
        majority_up = prediction["causal_majority_direction"] == "UP"
        decision = str(prediction["decision"])
        outcome = {
            "origin_date": prediction["origin_date"],
            "iso_week": prediction["iso_week"],
            "prediction_record_sha256": prediction_hash,
            "target_date": pd.Timestamp(frame.loc[origin_index + 1, "date"]).date().isoformat(),
            "actual_price": target_price,
            "actual_return": actual_return,
            "actual": actual,
            "base_hit": bool(base_up == bool(actual)),
            "selective_hit": None if decision == "FLAT" else bool((decision == "UP") == bool(actual)),
            "causal_majority_hit": bool(majority_up == bool(actual)),
            "observed_at": observed_at.isoformat(),
            "outcome_source_sha256": live_source_sha256,
            "previous_outcome_sha256": previous,
        }
        outcome["outcome_record_sha256"] = canonical_hash(outcome, "outcome_record_sha256")
        append_jsonl(path, outcome)
        outcomes.append(outcome)
        previous = str(outcome["outcome_record_sha256"])
        completed.add(prediction_hash)
    return outcomes


def _classification_summary(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    if len(actual) == 0:
        return {
            "n": 0, "directional_accuracy": None, "balanced_accuracy": None,
            "up_recall": None, "down_recall": None, "prediction_up_rate": None,
        }
    hits = predicted == actual
    up = actual == 1
    down = actual == 0
    up_recall = float(predicted[up].mean()) if up.any() else None
    down_recall = float((1 - predicted[down]).mean()) if down.any() else None
    return {
        "n": int(len(actual)),
        "directional_accuracy": float(hits.mean()),
        "balanced_accuracy": 0.5 * (up_recall + down_recall)
        if up_recall is not None and down_recall is not None else None,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "prediction_up_rate": float(predicted.mean()),
    }


def metric_summary(
    predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]],
) -> dict[str, Any]:
    outcome_by_hash = {str(item["prediction_record_sha256"]): item for item in outcomes}
    matured = [
        (prediction, outcome_by_hash[str(prediction["prediction_record_sha256"])])
        for prediction in predictions
        if str(prediction["prediction_record_sha256"]) in outcome_by_hash
    ]
    actual = np.asarray([outcome["actual"] for _, outcome in matured], dtype=int)
    base_predicted = np.asarray(
        [prediction["base_direction"] == "UP" for prediction, _ in matured], dtype=int
    )
    base = _classification_summary(actual, base_predicted)
    selective_pairs = [pair for pair in matured if pair[0]["decision"] != "FLAT"]
    selective_actual = np.asarray([outcome["actual"] for _, outcome in selective_pairs], dtype=int)
    selective_predicted = np.asarray(
        [prediction["decision"] == "UP" for prediction, _ in selective_pairs], dtype=int
    )
    selective = _classification_summary(selective_actual, selective_predicted)
    probabilities = np.asarray([prediction["probability_up"] for prediction, _ in matured], dtype=float)
    priors = np.asarray([prediction["train_up_rate"] for prediction, _ in matured], dtype=float)
    base_hits = np.asarray([outcome["base_hit"] for _, outcome in matured], dtype=float)
    majority_hits = np.asarray([outcome["causal_majority_hit"] for _, outcome in matured], dtype=float)
    return {
        "committed_sessions": len(predictions),
        "matured_base_predictions": len(matured),
        "matured_selective_signals": len(selective_pairs),
        "selective_coverage": len(selective_pairs) / len(matured) if matured else None,
        "base": base,
        "selective": selective,
        "base_lift_vs_causal_majority": float((base_hits - majority_hits).mean()) if matured else None,
        "brier_model": float(np.mean((probabilities - actual) ** 2)) if matured else None,
        "brier_causal_prior": float(np.mean((priors - actual) ** 2)) if matured else None,
    }


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
    summary = metric_summary(predictions, outcomes)
    outcome_by_hash = {str(item["prediction_record_sha256"]): item for item in outcomes}
    records = [
        {**prediction, "outcome": outcome_by_hash.get(str(prediction["prediction_record_sha256"]))}
        for prediction in predictions
    ]
    return {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "parent_rule": contract["_meta"]["parent_rule"],
        "generated_at": generated_at.isoformat(),
        "data_cutoff": provenance["frame_end"],
        "first_eligible_date": contract["prospective_protocol"]["first_eligible_date"],
        "status": "awaiting_first_commit" if not predictions else "collecting_prospective_evidence",
        "registration_valid": bool(registration["registration_valid"]),
        "contract_sha256": contract_hash,
        "code_bundle_sha256": code_bundle_hash,
        "prediction_chain_head": predictions[-1]["prediction_record_sha256"] if predictions else GENESIS_HASH,
        "outcome_chain_head": outcomes[-1]["outcome_record_sha256"] if outcomes else GENESIS_HASH,
        "signal_authorized": False,
        "capital_authorized": False,
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
    timezone = ZoneInfo(contract["asset"]["timezone"])
    now = datetime.now(timezone)
    frame, provenance = build_live_frame(contract, pd.Timestamp(now.date()))
    paths = output_paths(contract)
    predictions = load_jsonl(paths["predictions"])
    outcomes = load_jsonl(paths["outcomes"])
    prediction_head = validate_daily_hash_chain(
        predictions,
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    )
    outcome_head = validate_daily_hash_chain(
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
            "prediction_chain_head": prediction_head,
            "outcome_chain_head": outcome_head,
        }, indent=2))
        return 0

    outcomes = append_new_outcomes(
        predictions, outcomes, frame, paths["outcomes"], now, provenance["live_source_sha256"]
    )
    already_committed = any(item["origin_date"] == now.date().isoformat() for item in predictions)
    first_date = pd.Timestamp(contract["prospective_protocol"]["first_eligible_date"]).date()
    if commit_window_open(now, contract) and now.date() >= first_date and not already_committed:
        origin_index = validate_current_commit(frame, now, contract)
        m5_frame = load_current_m5_session(now.date(), contract)
        snapshot_frame, snapshot_manifest = ensure_snapshot(
            frame,
            m5_frame,
            provenance,
            now.date().isoformat(),
            paths,
            contract_hash,
            code_bundle_hash,
        )
        origin_index = int(snapshot_frame.index[-1])
        base_record = make_daily_prediction(snapshot_frame, origin_index, contract)
        base_record.update({
            "experiment_id": contract["_meta"]["experiment_id"],
            "committed_at": now.isoformat(),
            "contract_sha256": contract_hash,
            "code_bundle_sha256": code_bundle_hash,
            "input_snapshot_sha256": snapshot_manifest["processed_frame_sha256"],
            "decision_session_m5_sha256": snapshot_manifest["decision_session_m5_sha256"],
        })
        append_prediction(base_record, predictions, paths["predictions"])

    index = build_index(
        contract, registration, contract_hash, code_bundle_hash,
        predictions, outcomes, provenance, now,
    )
    paths["index"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["index"].with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_safe(index), indent=2), encoding="utf-8")
    temporary.replace(paths["index"])
    print(json.dumps({
        "experiment_id": index["experiment_id"],
        "status": index["status"],
        "data_cutoff": index["data_cutoff"],
        "committed_sessions": index["summary"]["committed_sessions"],
        "matured_base_predictions": index["summary"]["matured_base_predictions"],
        "capital_authorized": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
