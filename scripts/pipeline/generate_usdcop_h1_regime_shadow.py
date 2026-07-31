"""Commit immutable prospective H1 regime-shadow predictions for USD/COP.

The job is intended to run Friday after the USD/COP close. Predictions are
appended once per ISO week and never recomputed in the ledger. Later runs may
only attach the realized next-session outcome. A contract-hash mismatch fails
closed and requires a new experiment registration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_causal_regime_gate import build_states
from scripts.analysis.usdcop_long_history_directional_tournament import (
    build_long_history_frame,
)
from src.forecasting.directional_replay import walk_forward_probabilities


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v1.yaml"
REGISTRATION = ROOT / "reports/usdcop_h1_regime_shadow_registration.json"
LEDGER = ROOT / "reports/usdcop_h1_regime_shadow_ledger.csv"
INDEX = (
    ROOT
    / "usdcop-trading-dashboard/public/forecasting/usdcop/h1_regime_shadow_index.json"
)
LEDGER_COLUMNS = [
    "iso_week", "origin_date", "target_date", "target_date_estimated",
    "base_price", "probability_up", "threshold", "base_direction",
    "regime", "regime_allowed", "decision", "train_up_rate",
    "causal_majority_direction", "train_count", "train_label_end",
    "training_mode", "actual", "actual_price", "hit",
    "causal_majority_hit", "committed_at", "contract_sha256",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_week(value: str) -> tuple[int, int]:
    year, week = value.split("-W")
    return int(year), int(week)


def current_week(as_of: pd.Timestamp) -> tuple[int, int]:
    iso = as_of.isocalendar()
    return int(iso.year), int(iso.week)


def week_can_commit(
    iso_week: str, as_of: pd.Timestamp, commit_weekday: int,
) -> bool:
    origin_week = parse_week(iso_week)
    now_week = current_week(as_of)
    return origin_week < now_week or (
        origin_week == now_week and int(as_of.dayofweek) >= commit_weekday
    )


def load_contract() -> tuple[dict[str, Any], dict[str, Any], str]:
    if not CONFIG.exists() or not REGISTRATION.exists():
        raise FileNotFoundError("Shadow contract or registration is missing")
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registration = json.loads(REGISTRATION.read_text(encoding="utf-8"))
    contract_hash = sha256(CONFIG)
    if registration.get("contract_sha256") != contract_hash:
        raise RuntimeError(
            "Shadow contract hash changed after registration; create a new registration"
        )
    if contract["_meta"]["signal_authorized"] is not False:
        raise RuntimeError("Shadow contract must remain signal_authorized=false")
    if registration.get("signal_authorized") is not False:
        raise RuntimeError("Registration must remain signal_authorized=false")
    return contract, registration, contract_hash


def load_ledger() -> pd.DataFrame:
    if not LEDGER.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    ledger = pd.read_csv(LEDGER)
    missing = [column for column in LEDGER_COLUMNS if column not in ledger.columns]
    if missing:
        raise ValueError(f"Shadow ledger is missing columns: {missing}")
    if ledger["iso_week"].duplicated().any():
        raise ValueError("Shadow ledger contains duplicate ISO weeks")
    return ledger[LEDGER_COLUMNS].copy()


def mature_direction_target(frame: pd.DataFrame) -> pd.Series:
    returns = np.log(frame["close"].shift(-1) / frame["close"])
    return (returns > 0).where(returns.notna()).astype(float)


def build_candidates(
    frame: pd.DataFrame,
    contract: dict[str, Any],
) -> pd.DataFrame:
    first_week = str(contract["prospective_protocol"]["first_eligible_iso_week"])
    first_year, first_number = parse_week(first_week)
    first_date = pd.Timestamp.fromisocalendar(first_year, first_number, 1)
    if frame["date"].max() < first_date:
        return pd.DataFrame()
    model = contract["model"]
    horizon = int(contract["prospective_protocol"]["target_horizon_sessions"])
    candidates = walk_forward_probabilities(
        frame=frame,
        features=[str(value) for value in model["features"]],
        horizon=horizon,
        half_life=int(model["half_life_sessions"]),
        origin_start=first_date.date().isoformat(),
        origin_end=frame["date"].max().date().isoformat(),
        c_value=float(model["c"]),
        max_iter=int(model["max_iter"]),
    )
    if candidates.empty:
        return candidates
    states = build_states(frame).rename(columns={"date": "origin_date"})
    candidates = candidates.merge(
        states[["origin_date", "regime", "trend_z", "volatility_reference"]],
        on="origin_date",
        how="left",
        validate="many_to_one",
    )
    return candidates


def causal_train_prior(
    frame: pd.DataFrame, origin_date: pd.Timestamp,
) -> tuple[float, str]:
    dates = pd.to_datetime(frame["date"])
    matches = np.flatnonzero(dates.eq(origin_date).to_numpy())
    if len(matches) != 1:
        raise ValueError(f"Origin date not found exactly once: {origin_date}")
    origin_index = int(matches[0])
    target = mature_direction_target(frame)
    train = np.arange(0, origin_index)
    train = train[target.iloc[train].notna().to_numpy()]
    rate = float(target.iloc[train].mean())
    return rate, "UP" if rate >= 0.5 else "DOWN"


def append_new_predictions(
    ledger: pd.DataFrame,
    candidates: pd.DataFrame,
    frame: pd.DataFrame,
    contract: dict[str, Any],
    contract_hash: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    if candidates.empty:
        return ledger
    existing_weeks = set(ledger["iso_week"].astype(str))
    protocol = contract["prospective_protocol"]
    commit_weekday = int(protocol["commit_weekday"])
    first_week = parse_week(str(protocol["first_eligible_iso_week"]))
    allowed_states = set(map(str, contract["regime_gate"]["allowed_states"]))
    threshold = float(contract["model"]["probability_threshold"])
    committed_at = datetime.now(ZoneInfo(contract["asset"]["timezone"])).isoformat()
    rows: list[dict[str, Any]] = []
    for row in candidates.sort_values("origin_date").itertuples(index=False):
        iso_week = str(row.iso_week)
        if parse_week(iso_week) < first_week or iso_week in existing_weeks:
            continue
        if not week_can_commit(iso_week, as_of, commit_weekday):
            continue
        base_direction = "UP" if float(row.probability_up) >= threshold else "DOWN"
        regime_allowed = str(row.regime) in allowed_states
        decision = base_direction if regime_allowed else "FLAT"
        train_up_rate, majority_direction = causal_train_prior(
            frame, pd.Timestamp(row.origin_date),
        )
        actual = None if pd.isna(row.actual) else int(row.actual)
        rows.append({
            "iso_week": iso_week,
            "origin_date": pd.Timestamp(row.origin_date).date().isoformat(),
            "target_date": pd.Timestamp(row.target_date).date().isoformat(),
            "target_date_estimated": bool(row.target_date_estimated),
            "base_price": float(row.base_price),
            "probability_up": float(row.probability_up),
            "threshold": threshold,
            "base_direction": base_direction,
            "regime": str(row.regime),
            "regime_allowed": regime_allowed,
            "decision": decision,
            "train_up_rate": train_up_rate,
            "causal_majority_direction": majority_direction,
            "train_count": int(row.train_count),
            "train_label_end": (
                None if pd.isna(row.train_label_end)
                else pd.Timestamp(row.train_label_end).date().isoformat()
            ),
            "training_mode": str(contract["model"]["training_mode"]),
            "actual": actual,
            "actual_price": None,
            "hit": (
                None if actual is None or decision == "FLAT"
                else bool((decision == "UP") == bool(actual))
            ),
            "causal_majority_hit": (
                None if actual is None
                else bool((majority_direction == "UP") == bool(actual))
            ),
            "committed_at": committed_at,
            "contract_sha256": contract_hash,
        })
        existing_weeks.add(iso_week)
    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
    return ledger.sort_values("iso_week").reset_index(drop=True)


def update_outcomes(ledger: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    result = ledger.copy()
    dates = pd.to_datetime(frame["date"])
    date_to_index = {date.date(): index for index, date in enumerate(dates)}
    for index, row in result.iterrows():
        if pd.notna(row["actual"]):
            continue
        origin = pd.Timestamp(row["origin_date"]).date()
        origin_index = date_to_index.get(origin)
        if origin_index is None or origin_index + 1 >= len(frame):
            continue
        actual_price = float(frame.loc[origin_index + 1, "close"])
        actual = int(actual_price > float(row["base_price"]))
        result.at[index, "target_date"] = pd.Timestamp(
            frame.loc[origin_index + 1, "date"]
        ).date().isoformat()
        result.at[index, "target_date_estimated"] = False
        result.at[index, "actual"] = actual
        result.at[index, "actual_price"] = actual_price
        if row["decision"] != "FLAT":
            result.at[index, "hit"] = bool(
                (row["decision"] == "UP") == bool(actual)
            )
        result.at[index, "causal_majority_hit"] = bool(
            (row["causal_majority_direction"] == "UP") == bool(actual)
        )
    return result


def metric_summary(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {
            "committed_weeks": 0, "matured_weeks": 0, "matured_signals": 0,
            "coverage": None, "directional_accuracy": None,
            "balanced_accuracy": None, "up_recall": None, "down_recall": None,
            "lift_vs_causal_majority": None,
        }
    matured = ledger[ledger["actual"].notna()].copy()
    signals = matured[matured["decision"].ne("FLAT")].copy()
    summary: dict[str, Any] = {
        "committed_weeks": int(len(ledger)),
        "matured_weeks": int(len(matured)),
        "matured_signals": int(len(signals)),
        "coverage": float(len(signals) / len(matured)) if len(matured) else None,
        "directional_accuracy": None,
        "balanced_accuracy": None,
        "up_recall": None,
        "down_recall": None,
        "lift_vs_causal_majority": None,
    }
    if signals.empty:
        return summary
    actual = signals["actual"].astype(int)
    prediction = signals["decision"].eq("UP").astype(int)
    up = actual.eq(1)
    down = actual.eq(0)
    up_recall = float(prediction[up].mean()) if up.any() else None
    down_recall = float((1 - prediction[down]).mean()) if down.any() else None
    baseline_accuracy = float(
        signals["causal_majority_hit"].astype(bool).mean()
    )
    da = float((prediction == actual).mean())
    summary.update({
        "directional_accuracy": da,
        "balanced_accuracy": (
            0.5 * (up_recall + down_recall)
            if up_recall is not None and down_recall is not None else None
        ),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "lift_vs_causal_majority": da - baseline_accuracy,
    })
    return summary


def json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--as-of-date",
        help="Override America/Bogota commit date (YYYY-MM-DD) for deterministic tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract, registration, contract_hash = load_contract()
    timezone = ZoneInfo(str(contract["asset"]["timezone"]))
    as_of = (
        pd.Timestamp(args.as_of_date)
        if args.as_of_date else pd.Timestamp(datetime.now(timezone).date())
    )
    frame, provenance = build_long_history_frame()
    candidates = build_candidates(frame, contract)
    ledger = load_ledger()
    ledger = append_new_predictions(
        ledger, candidates, frame, contract, contract_hash, as_of,
    )
    ledger = update_outcomes(ledger, frame)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(LEDGER, index=False, columns=LEDGER_COLUMNS)

    summary = metric_summary(ledger)
    protocol = contract["prospective_protocol"]
    enough_evidence = bool(
        summary["committed_weeks"]
        >= int(protocol["minimum_calendar_weeks_before_promotion_review"])
        and summary["matured_signals"]
        >= int(protocol["minimum_matured_signals_before_promotion_review"])
    )
    document = {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "generated_at": datetime.now(timezone).isoformat(),
        "data_cutoff": frame["date"].max().date().isoformat(),
        "first_eligible_iso_week": protocol["first_eligible_iso_week"],
        "status": (
            "awaiting_first_commit" if ledger.empty
            else "collecting_prospective_evidence"
        ),
        "contract_sha256": contract_hash,
        "registration_valid": bool(registration["registration_valid"]),
        "signal_authorized": False,
        "capital_authorized": False,
        "promotion_review_eligible": enough_evidence,
        "summary": summary,
        "provenance": provenance,
        "records": [
            json_value(record)
            for record in ledger.to_dict(orient="records")
        ],
    }
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": document["status"],
        "data_cutoff": document["data_cutoff"],
        "committed_weeks": summary["committed_weeks"],
        "matured_signals": summary["matured_signals"],
        "promotion_review_eligible": enough_evidence,
        "index": str(INDEX.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
