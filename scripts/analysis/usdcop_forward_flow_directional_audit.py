"""Preregistered paired audit of BanRep forward flows for USD/COP direction.

The experiment toggles exactly one information family: official consolidated
forward positioning, balances and implied-devaluation curves.  Everything else
(weekly origins, labels, estimator, threshold and training samples) is shared
between price-only and price-plus-flow models.

The initial BanRep workbook backfill is reconstructed from a mutable historical
file.  Consequently every result produced here is research-only.  A passing
historical gate can only nominate a candidate for prospective shadow tracking;
it cannot authorize capital.
"""
from __future__ import annotations

import argparse
from copy import copy
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_long_history_directional_tournament import (
    build_long_history_frame,
)
from src.data.usdcop_forward_macro import attach_forward_macro_features


DEFAULT_CONFIG = (
    ROOT / "config" / "forecast_experiments"
    / "usdcop_forward_flow_direction_v1.yaml"
)
DEFAULT_REGISTRATION = (
    ROOT / "config" / "forecast_experiments" / "preregistrations"
    / "usdcop_forward_flow_direction_v1.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def load_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    experiment_id = str(config["_meta"]["experiment_id"])
    registration_path = (
        config_path.parent / "preregistrations" / f"{experiment_id}.json"
    )
    with registration_path.open("r", encoding="utf-8") as handle:
        registration = json.load(handle)
    actual_hash = sha256_file(config_path)
    expected_hash = registration["config_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "Preregistered protocol changed before execution: "
            f"expected {expected_hash}, got {actual_hash}"
        )
    for key, relative in (
        ("deep_price", config["data"]["price_sources"]["deep_path"]),
        ("current_price", config["data"]["price_sources"]["current_path"]),
        ("forward_macro_pit", config["data"]["flow_path"]),
    ):
        actual = sha256_file(ROOT / relative)
        expected = registration["data_sha256"][key]
        if actual != expected:
            raise ValueError(
                f"Registered data drift for {key}: expected {expected}, got {actual}"
            )
    return config, registration, registration_path


def build_research_frame(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    frame, price_provenance = build_long_history_frame()
    frame = frame[pd.to_datetime(frame["date"]) <= pd.Timestamp(
        config["data"]["evaluation_asof"]
    )].copy()
    frame, attached = attach_forward_macro_features(
        frame,
        ROOT / config["data"]["flow_path"],
        decision_hour_bogota=int(config["data"]["decision_hour"]),
        promotion_only=False,
    )
    required = list(config["features"]["price"]) + list(
        config["features"]["forward_flow"]
    )
    missing = [feature for feature in required if feature not in frame.columns]
    if missing:
        raise ValueError(f"Registered features missing from research frame: {missing}")
    attached_set = set(attached)
    unattached = [
        feature for feature in config["features"]["forward_flow"]
        if feature not in attached_set
    ]
    if unattached:
        raise ValueError(f"Registered flow features were not causally attached: {unattached}")
    frame = frame.sort_values("date").reset_index(drop=True)
    coverage: list[dict[str, Any]] = []
    for feature in required:
        valid = frame.loc[frame[feature].notna(), "date"]
        coverage.append({
            "feature": feature,
            "family": "price" if feature in config["features"]["price"] else "forward_flow",
            "non_null_rows": int(len(valid)),
            "first_available": valid.min() if len(valid) else None,
            "last_available": valid.max() if len(valid) else None,
        })
    provenance = {
        **price_provenance,
        "research_frame_rows": int(len(frame)),
        "research_frame_end": pd.Timestamp(frame["date"].max()),
        "attached_pit_feature_count": int(len(attached)),
        "registered_flow_feature_count": int(len(config["features"]["forward_flow"])),
        "historical_flow_is_reconstructed": True,
    }
    return frame, provenance, coverage


def make_target(frame: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    future_return = np.log(frame["close"].shift(-horizon) / frame["close"])
    target = (future_return > 0).where(future_return.notna()).astype(float)
    return target, future_return


def weekly_origin_indices(
    frame: pd.DataFrame,
    start: str,
    end: str,
    *,
    asof: str,
    exclude_incomplete_current_week: bool,
) -> list[int]:
    dates = pd.to_datetime(frame["date"])
    iso = dates.dt.strftime("%G-W%V")
    origins = frame.assign(_iso_week=iso).groupby("_iso_week").tail(1)
    origins = origins[origins["date"].between(pd.Timestamp(start), pd.Timestamp(end))]
    if exclude_incomplete_current_week:
        asof_week = pd.Timestamp(asof).strftime("%G-W%V")
        origins = origins[origins["_iso_week"] != asof_week]
    return [int(index) for index in origins.index]


def eligible_train_indices(
    frame: pd.DataFrame,
    target: pd.Series,
    horizon: int,
    label_known_through_index: int,
    train_start: str,
) -> np.ndarray:
    dates = pd.to_datetime(frame["date"])
    start_index = int(np.searchsorted(dates.to_numpy(), np.datetime64(train_start)))
    final_origin_index = label_known_through_index - horizon
    if final_origin_index < start_index:
        return np.array([], dtype=int)
    indices = np.arange(start_index, final_origin_index + 1, dtype=int)
    return indices[target.iloc[indices].notna().to_numpy()]


def new_model(config: dict[str, Any]) -> Pipeline:
    spec = config["model"]
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=float(spec["c"]),
            penalty=str(spec["penalty"]),
            class_weight=str(spec["class_weight"]),
            max_iter=int(spec["max_iter"]),
            random_state=int(spec["random_state"]),
        ),
    )


def fit_model(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    train: np.ndarray,
    config: dict[str, Any],
) -> Pipeline:
    minimum = int(config["inference"]["minimum_mature_training_labels"])
    if len(train) < minimum:
        raise ValueError(f"Only {len(train)} mature labels; protocol requires {minimum}")
    classes = target.iloc[train].dropna().astype(int).unique()
    if len(classes) != 2:
        raise ValueError(f"Training target contains {len(classes)} class(es), expected 2")
    model = new_model(config)
    model.fit(frame.loc[train, features], target.iloc[train].astype(int))
    return model


def forecast_record(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    future_return: pd.Series,
    horizon: int,
    origin_index: int,
    model_name: str,
    model: Pipeline,
    features: list[str],
    train: np.ndarray,
    period_name: str,
    period_role: str,
    training_mode: str,
    threshold: float,
) -> dict[str, Any]:
    probability = float(model.predict_proba(frame.loc[[origin_index], features])[0, 1])
    prediction = int(probability >= threshold)
    actual_value = target.iloc[origin_index]
    actual = None if pd.isna(actual_value) else int(actual_value)
    actual_return = future_return.iloc[origin_index]
    target_index = origin_index + horizon
    if target_index < len(frame):
        target_date = pd.Timestamp(frame.loc[target_index, "date"])
    else:
        target_date = pd.Timestamp(frame.loc[origin_index, "date"]) + pd.offsets.BDay(horizon)
    train_up_rate = float(target.iloc[train].mean())
    majority_prediction = int(train_up_rate >= 0.5)
    last_train_target_index = int(train[-1]) + horizon
    return {
        "period": period_name,
        "period_role": period_role,
        "iso_week": pd.Timestamp(frame.loc[origin_index, "date"]).strftime("%G-W%V"),
        "origin_date": pd.Timestamp(frame.loc[origin_index, "date"]),
        "target_date": target_date,
        "horizon_days": int(horizon),
        "model": model_name,
        "feature_count": int(len(features)),
        "probability_up": probability,
        "threshold": threshold,
        "prediction": prediction,
        "predicted_direction": "UP" if prediction else "DOWN",
        "actual": actual,
        "actual_direction": None if actual is None else ("UP" if actual else "DOWN"),
        "correct": None if actual is None else int(prediction == actual),
        "actual_log_return": None if pd.isna(actual_return) else float(actual_return),
        "train_count": int(len(train)),
        "train_up_rate": train_up_rate,
        "majority_prediction": majority_prediction,
        "majority_correct": None if actual is None else int(majority_prediction == actual),
        "train_label_end": pd.Timestamp(frame.loc[last_train_target_index, "date"]),
        "training_mode": training_mode,
        "promotion_eligible": False,
    }


def run_replay(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    price_features = list(config["features"]["price"])
    flow_features = list(config["features"]["forward_flow"])
    feature_sets = {
        "price_only": price_features,
        "price_plus_forward_flow": price_features + flow_features,
    }
    horizons = [int(value) for value in config["protocol"]["horizons_trading_days"]]
    threshold = float(config["protocol"]["threshold"])
    train_start = str(config["data"]["price_start"])
    dates = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target, future_return = make_target(frame, horizon)
        print(f"[H{horizon}] paired weekly replay", flush=True)
        for period_name, period in config["protocol"]["periods"].items():
            origin_indices = weekly_origin_indices(
                frame,
                str(period["start"]),
                str(period["end"]),
                asof=str(config["data"]["evaluation_asof"]),
                exclude_incomplete_current_week=bool(
                    config["data"]["exclude_incomplete_current_iso_week"]
                ),
            )
            training_mode = str(period["training_mode"])
            role = str(period["role"])
            frozen_models: dict[str, tuple[Pipeline, np.ndarray]] = {}
            if training_mode.startswith("frozen_pre_"):
                frozen_year = int(training_mode.rsplit("_", 1)[-1])
                pre_year = np.flatnonzero(dates.lt(pd.Timestamp(f"{frozen_year}-01-01")))
                if not len(pre_year):
                    raise ValueError(f"No observations before frozen year {frozen_year}")
                train = eligible_train_indices(
                    frame, target, horizon, int(pre_year[-1]), train_start
                )
                for model_name, features in feature_sets.items():
                    frozen_models[model_name] = (
                        fit_model(frame, target, features, train, config), train
                    )
            for origin_index in origin_indices:
                if training_mode.startswith("frozen_pre_"):
                    models = frozen_models
                else:
                    train = eligible_train_indices(
                        frame, target, horizon, origin_index, train_start
                    )
                    models = {
                        model_name: (
                            fit_model(frame, target, features, train, config), train
                        )
                        for model_name, features in feature_sets.items()
                    }
                for model_name, features in feature_sets.items():
                    model, train = models[model_name]
                    rows.append(forecast_record(
                        frame=frame,
                        target=target,
                        future_return=future_return,
                        horizon=horizon,
                        origin_index=origin_index,
                        model_name=model_name,
                        model=model,
                        features=features,
                        train=train,
                        period_name=period_name,
                        period_role=role,
                        training_mode=training_mode,
                        threshold=threshold,
                    ))
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("Preregistered replay produced no forecasts")
    return result.sort_values(
        ["horizon_days", "origin_date", "model"]
    ).reset_index(drop=True)


def safe_auc(actual: pd.Series, probability: pd.Series) -> float | None:
    if actual.nunique() < 2:
        return None
    actual_values = actual.astype(int).to_numpy()
    probabilities = probability.astype(float).to_numpy()
    positives = actual_values == 1
    negatives = ~positives
    ranks = pd.Series(probabilities).rank(method="average").to_numpy()
    rank_sum = float(ranks[positives].sum())
    return (
        rank_sum - positives.sum() * (positives.sum() + 1) / 2
    ) / (positives.sum() * negatives.sum())


def metric_row(
    frame: pd.DataFrame,
    *,
    scope: str,
    period: str,
    horizon: int,
    model: str,
) -> dict[str, Any]:
    matured = frame[frame["actual"].notna()].copy()
    if matured.empty:
        return {
            "scope": scope, "period": period, "horizon_days": horizon,
            "model": model, "n_forecasts": int(len(frame)), "n_matured": 0,
        }
    actual = matured["actual"].astype(int)
    prediction = matured["prediction"].astype(int)
    probability = matured["probability_up"].astype(float)
    up = actual.eq(1)
    down = actual.eq(0)
    up_recall = float(prediction[up].mean()) if up.any() else None
    down_recall = float((1 - prediction[down]).mean()) if down.any() else None
    balanced = (
        0.5 * (up_recall + down_recall)
        if up_recall is not None and down_recall is not None else None
    )
    minimum_recall = (
        min(up_recall, down_recall)
        if up_recall is not None and down_recall is not None else None
    )
    da = float((actual == prediction).mean())
    majority_da = float(matured["majority_correct"].astype(int).mean())
    actual_up_rate = float(actual.mean())
    oracle_constant = max(actual_up_rate, 1.0 - actual_up_rate)
    brier = float(np.mean((probability.to_numpy() - actual.to_numpy()) ** 2))
    clipped = probability.clip(1e-12, 1 - 1e-12)
    log_loss = float(-np.mean(
        actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)
    ))
    return {
        "scope": scope,
        "period": period,
        "horizon_days": int(horizon),
        "model": model,
        "n_forecasts": int(len(frame)),
        "n_matured": int(len(matured)),
        "directional_accuracy": da,
        "balanced_accuracy": balanced,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "minimum_class_recall": minimum_recall,
        "prediction_up_rate": float(prediction.mean()),
        "actual_up_rate": actual_up_rate,
        "causal_majority_accuracy": majority_da,
        "lift_vs_causal_majority": da - majority_da,
        "oracle_constant_accuracy": oracle_constant,
        "lift_vs_oracle_constant": da - oracle_constant,
        "auc": safe_auc(actual, probability),
        "brier_score": brier,
        "log_loss": log_loss,
    }


def summarize(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (period, horizon, model), group in predictions.groupby(
        ["period", "horizon_days", "model"], sort=False
    ):
        rows.append(metric_row(
            group,
            scope="protocol_period",
            period=str(period),
            horizon=int(horizon),
            model=str(model),
        ))
    year_frame = predictions.copy()
    year_frame["calendar_year"] = pd.to_datetime(year_frame["origin_date"]).dt.year
    for (year, horizon, model), group in year_frame.groupby(
        ["calendar_year", "horizon_days", "model"], sort=False
    ):
        rows.append(metric_row(
            group,
            scope="calendar_year",
            period=str(int(year)),
            horizon=int(horizon),
            model=str(model),
        ))
    return pd.DataFrame(rows).sort_values(
        ["scope", "period", "horizon_days", "model"]
    ).reset_index(drop=True)


def exact_mcnemar(price_hits: np.ndarray, flow_hits: np.ndarray) -> tuple[int, int, float]:
    flow_only = int(np.sum(flow_hits & ~price_hits))
    price_only = int(np.sum(price_hits & ~flow_hits))
    discordant = flow_only + price_only
    p_value = (
        float(binomtest(flow_only, discordant, 0.5).pvalue)
        if discordant else 1.0
    )
    return flow_only, price_only, p_value


def moving_block_bootstrap(
    paired_hit_delta: np.ndarray,
    block_length: int,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    n = len(paired_hit_delta)
    if not n:
        return None, None, None
    rng = np.random.default_rng(seed)
    blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=(samples, blocks))
    offsets = np.arange(block_length, dtype=int)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(samples, -1)[:, :n]
    means = paired_hit_delta[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper), float(np.mean(means <= 0.0))


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_tests(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = int(config["statistics"]["bootstrap_samples"])
    base_seed = int(config["statistics"]["bootstrap_seed"])
    period_names = list(config["protocol"]["periods"])
    for period_index, period in enumerate(period_names):
        subset = predictions[predictions["period"].eq(period)]
        for horizon, group in subset.groupby("horizon_days"):
            mature = group[group["actual"].notna()].copy()
            paired = mature.pivot(
                index="origin_date", columns="model", values="correct"
            ).dropna()
            paired = paired.sort_index()
            price_hits = paired["price_only"].astype(bool).to_numpy()
            flow_hits = paired["price_plus_forward_flow"].astype(bool).to_numpy()
            flow_only, price_only, p_value = exact_mcnemar(price_hits, flow_hits)
            delta = flow_hits.astype(float) - price_hits.astype(float)
            block_length = int(math.ceil(int(horizon) / 5))
            ci_low, ci_high, bootstrap_p_nonpositive = moving_block_bootstrap(
                delta,
                block_length,
                samples,
                base_seed + period_index * 100 + int(horizon),
            )
            nonoverlap = paired.iloc[::block_length]
            non_price = nonoverlap["price_only"].astype(float)
            non_flow = nonoverlap["price_plus_forward_flow"].astype(float)
            treatment_metric = summary[
                summary["scope"].eq("protocol_period")
                & summary["period"].eq(period)
                & summary["horizon_days"].eq(int(horizon))
                & summary["model"].eq("price_plus_forward_flow")
            ].iloc[0]
            rows.append({
                "period": period,
                "horizon_days": int(horizon),
                "n_pairs": int(len(paired)),
                "price_only_da": float(price_hits.mean()) if len(paired) else None,
                "price_plus_flow_da": float(flow_hits.mean()) if len(paired) else None,
                "paired_da_delta": float(delta.mean()) if len(paired) else None,
                "discordant_flow_right": flow_only,
                "discordant_price_right": price_only,
                "mcnemar_p_raw": p_value,
                "block_length_weeks": block_length,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_p_delta_nonpositive": bootstrap_p_nonpositive,
                "nonoverlap_n": int(len(nonoverlap)),
                "nonoverlap_price_da": float(non_price.mean()) if len(nonoverlap) else None,
                "nonoverlap_flow_da": float(non_flow.mean()) if len(nonoverlap) else None,
                "nonoverlap_da_delta": (
                    float((non_flow - non_price).mean()) if len(nonoverlap) else None
                ),
                "treatment_balanced_accuracy": treatment_metric.get("balanced_accuracy"),
                "treatment_minimum_class_recall": treatment_metric.get(
                    "minimum_class_recall"
                ),
            })
    result = pd.DataFrame(rows)
    adjusted_parts = []
    for _, part in result.groupby("period", sort=False):
        part = part.copy()
        part["mcnemar_p_holm"] = holm_adjust(part["mcnemar_p_raw"])
        family_size = int(
            config["statistics"].get("confirmatory_family_size", len(part))
        )
        part["mcnemar_p_family_bonferroni"] = (
            part["mcnemar_p_raw"] * family_size
        ).clip(upper=1.0)
        part["mcnemar_p_gate"] = part[[
            "mcnemar_p_holm", "mcnemar_p_family_bonferroni"
        ]].max(axis=1)
        adjusted_parts.append(part)
    result = pd.concat(adjusted_parts, ignore_index=True)
    gates = config["gates"]["incremental_primary_per_horizon"]
    primary = str(config["hypothesis"]["primary_period"])
    result["primary_incremental_gate_pass"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    mask = result["period"].eq(primary)
    result.loc[mask, "primary_incremental_gate_pass"] = (
        result.loc[mask, "paired_da_delta"].gt(float(gates["paired_da_delta_gt"]))
        & result.loc[mask, "mcnemar_p_gate"].lt(float(gates["holm_mcnemar_p_lt"]))
        & result.loc[mask, "bootstrap_ci_low"].gt(
            float(gates["block_bootstrap_ci_low_gt"])
        )
        & result.loc[mask, "treatment_balanced_accuracy"].ge(
            float(gates["treatment_balanced_accuracy_gte"])
        )
        & result.loc[mask, "treatment_minimum_class_recall"].ge(
            float(gates["treatment_minimum_class_recall_gte"])
        )
        & result.loc[mask, "nonoverlap_da_delta"].ge(
            float(gates["nonoverlap_da_delta_gte"])
        )
    )
    return result.sort_values(["period", "horizon_days"]).reset_index(drop=True)


def horizon_decisions(
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    primary_name = str(config["hypothesis"]["primary_period"])
    replay_name = "frozen_replay_2025"
    forward_name = "weekly_retrain_2026"
    confirmation = config["gates"]["temporal_confirmation"]
    rows: list[dict[str, Any]] = []
    for horizon in config["protocol"]["horizons_trading_days"]:
        horizon = int(horizon)
        primary = tests[
            tests["period"].eq(primary_name)
            & tests["horizon_days"].eq(horizon)
        ].iloc[0]
        replay = tests[
            tests["period"].eq(replay_name)
            & tests["horizon_days"].eq(horizon)
        ].iloc[0]
        forward = tests[
            tests["period"].eq(forward_name)
            & tests["horizon_days"].eq(horizon)
        ].iloc[0]
        replay_metric = summary[
            summary["scope"].eq("protocol_period")
            & summary["period"].eq(replay_name)
            & summary["horizon_days"].eq(horizon)
            & summary["model"].eq("price_plus_forward_flow")
        ].iloc[0]
        forward_metric = summary[
            summary["scope"].eq("protocol_period")
            & summary["period"].eq(forward_name)
            & summary["horizon_days"].eq(horizon)
            & summary["model"].eq("price_plus_forward_flow")
        ].iloc[0]
        replay_confirmation = bool(
            replay["paired_da_delta"]
            >= float(confirmation["frozen_replay_2025_da_delta_gte"])
            and replay_metric["balanced_accuracy"]
            >= float(confirmation["frozen_replay_2025_balanced_accuracy_gte"])
        )
        primary_pass = bool(primary["primary_incremental_gate_pass"])
        candidate = primary_pass and replay_confirmation
        if not primary_pass:
            reason = "FAIL_PRIMARY_INCREMENTAL_GATE"
        elif not replay_confirmation:
            reason = "FAIL_2025_TEMPORAL_CONFIRMATION"
        else:
            reason = "PROSPECTIVE_SHADOW_CANDIDATE_ONLY"
        rows.append({
            "horizon_days": horizon,
            "primary_n_pairs": int(primary["n_pairs"]),
            "primary_price_da": primary["price_only_da"],
            "primary_flow_da": primary["price_plus_flow_da"],
            "primary_da_delta": primary["paired_da_delta"],
            "primary_flow_bda": primary["treatment_balanced_accuracy"],
            "primary_min_recall": primary["treatment_minimum_class_recall"],
            "primary_holm_p": primary["mcnemar_p_holm"],
            "primary_family_adjusted_p": primary["mcnemar_p_gate"],
            "primary_bootstrap_ci_low": primary["bootstrap_ci_low"],
            "primary_bootstrap_ci_high": primary["bootstrap_ci_high"],
            "primary_nonoverlap_delta": primary["nonoverlap_da_delta"],
            "primary_gate_pass": primary_pass,
            "replay_2025_n_pairs": int(replay["n_pairs"]),
            "replay_2025_da_delta": replay["paired_da_delta"],
            "replay_2025_flow_bda": replay_metric["balanced_accuracy"],
            "replay_2025_confirmation": replay_confirmation,
            "forward_2026_n_matured": int(forward_metric["n_matured"]),
            "forward_2026_da_delta": forward["paired_da_delta"],
            "forward_2026_flow_da": forward["price_plus_flow_da"],
            "forward_2026_flow_bda": forward_metric["balanced_accuracy"],
            "forward_2026_diagnostic_mature_enough": bool(
                forward_metric["n_matured"]
                >= int(confirmation["weekly_retrain_2026_is_diagnostic_until_matured_weeks"])
            ),
            "prospective_shadow_candidate": candidate,
            "capital_authorized": False,
            "decision": reason,
        })
    return pd.DataFrame(rows)


def write_workbook(
    path: Path,
    overview: pd.DataFrame,
    decisions: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    predictions: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        overview.to_excel(writer, sheet_name="overview", index=False)
        decisions.to_excel(writer, sheet_name="horizon_decisions", index=False)
        summary.to_excel(writer, sheet_name="metrics", index=False)
        tests.to_excel(writer, sheet_name="paired_tests", index=False)
        predictions.to_excel(writer, sheet_name="weekly_predictions", index=False)
        features.to_excel(writer, sheet_name="features", index=False)
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                cell.font = font
            for column_cells in worksheet.columns:
                values = [str(cell.value) if cell.value is not None else "" for cell in column_cells[:200]]
                width = min(55, max(10, max(map(len, values), default=10) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config, registration, registration_path = load_protocol(config_path)
    print(
        "Protocol verified: "
        f"{registration['experiment_id']} @ {registration['config_sha256'][:12]}",
        flush=True,
    )
    frame, provenance, coverage = build_research_frame(config)
    predictions = run_replay(frame, config)
    summary = summarize(predictions)
    tests = paired_tests(predictions, summary, config)
    decisions = horizon_decisions(tests, summary, config)

    output_dir = ROOT / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / config["outputs"]["predictions"]
    summary_path = output_dir / config["outputs"]["summary"]
    tests_path = output_dir / config["outputs"]["paired_tests"]
    workbook_path = output_dir / config["outputs"]["workbook"]
    predictions.to_csv(predictions_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    decisions.to_csv(output_dir / "horizon_decisions.csv", index=False)

    manifest = {
        "experiment_id": registration["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": sha256_file(config_path),
        "registration_sha256": sha256_file(registration_path),
        "script_sha256": sha256_file(Path(__file__)),
        "data_sha256": registration["data_sha256"],
        "provenance": provenance,
        "prediction_rows": int(len(predictions)),
        "matured_prediction_rows": int(predictions["actual"].notna().sum()),
        "horizons_tested": [int(value) for value in config["protocol"]["horizons_trading_days"]],
        "primary_gate_pass_count": int(decisions["primary_gate_pass"].sum()),
        "prospective_shadow_candidate_count": int(
            decisions["prospective_shadow_candidate"].sum()
        ),
        "capital_authorized": False,
        "evidence_class": config["_meta"]["evidence_class"],
        "warning": (
            "Initial BanRep consolidated history is reconstructed from a mutable "
            "workbook. Results are research-only and cannot establish genuine OOS proof."
        ),
    }
    manifest_path = output_dir / config["outputs"]["manifest"]
    manifest_path.write_text(
        json.dumps(safe_json(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    overview = pd.DataFrame([
        {"field": "experiment_id", "value": registration["experiment_id"]},
        {"field": "evidence_class", "value": config["_meta"]["evidence_class"]},
        {"field": "primary_period", "value": config["hypothesis"]["primary_period"]},
        {"field": "all_7_horizons_tested", "value": True},
        {"field": "primary_gate_pass_count", "value": int(decisions["primary_gate_pass"].sum())},
        {"field": "shadow_candidate_count", "value": int(decisions["prospective_shadow_candidate"].sum())},
        {"field": "capital_authorized", "value": False},
        {"field": "config_sha256", "value": sha256_file(config_path)},
        {"field": "warning", "value": manifest["warning"]},
    ])
    write_workbook(
        workbook_path,
        overview,
        decisions,
        summary,
        tests,
        predictions,
        pd.DataFrame(coverage),
    )

    primary = decisions[[
        "horizon_days", "primary_price_da", "primary_flow_da",
        "primary_da_delta", "primary_flow_bda", "primary_holm_p",
        "primary_gate_pass", "replay_2025_da_delta",
        "forward_2026_flow_da", "forward_2026_da_delta",
        "prospective_shadow_candidate",
    ]]
    print(primary.to_string(index=False), flush=True)
    print(f"Workbook: {workbook_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
