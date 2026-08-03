"""Causal USD/COP directional-model tournament with selective decisions.

The tournament is deliberately separated from the production replay.  It tests
model *rules* using weekly walk-forward predictions from 2022-2024, freezes the
winning rule before 2025, evaluates a model frozen throughout 2025, and then
re-trains weekly in 2026 using only labels that have matured at each origin.

No 2025/2026 outcome participates in model, feature-family, memory, class-weight
or decision-threshold selection.  Outputs are research evidence and never
authorize a trading signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.weekly_forecasting_oos_canonical import build_frame


HORIZONS = (1, 5, 10, 15, 20, 25, 30)
VALIDATION_YEARS = (2022, 2023, 2024)
CONFIDENCE_CENTERS = (0.45, 0.50, 0.55)
CONFIDENCE_MARGINS = (0.00, 0.05, 0.10)
MIN_VALIDATION_COVERAGE = 0.25
MIN_VALIDATION_SIGNALS_PER_YEAR = 12
MIN_VALIDATION_PREDICTION_CLASS_RATE = 0.10
MIN_VALIDATION_CLASS_RECALL = 0.20
MIN_VALIDATION_YEAR_BALANCED_ACCURACY = 0.48
BOOTSTRAP_SAMPLES = 5_000
RANDOM_SEED = 2307


@dataclass(frozen=True)
class Variant:
    feature_group: str
    c_value: float
    class_weight: str
    half_life: int | None

    @property
    def key(self) -> str:
        memory = "expanding" if self.half_life is None else str(self.half_life)
        return f"{self.feature_group}|C={self.c_value:g}|{self.class_weight}|hl={memory}"


def make_return_target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    return np.log(frame["close"].shift(-horizon) / frame["close"])


def make_direction_target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    returns = make_return_target(frame, horizon)
    return (returns > 0).where(returns.notna()).astype(float)


def weekly_origins(frame: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    origins = frame.assign(
        iso_week=dates.dt.strftime("%G-W%V")
    ).groupby("iso_week").tail(1)
    return origins[
        origins["date"].between(pd.Timestamp(start), pd.Timestamp(end))
    ].copy()


def eligible_train_indices(
    target: pd.Series, origin_index: int, horizon: int,
) -> np.ndarray:
    # A label at i is mature only when i + horizon <= origin_index.
    indices = np.arange(0, max(0, origin_index - horizon + 1))
    return indices[target.iloc[indices].notna().to_numpy()]


def sample_weights(indices: np.ndarray, half_life: int | None) -> np.ndarray | None:
    if half_life is None or not len(indices):
        return None
    ages = indices[-1] - indices
    return np.exp(-np.log(2.0) * ages / half_life)


def new_model(variant: Variant) -> Pipeline:
    class_weight: str | None = None if variant.class_weight == "unweighted" else "balanced"
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=variant.c_value,
            class_weight=class_weight,
            max_iter=2_000,
            random_state=23,
        ),
    )


def fit_model(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    indices: np.ndarray,
    variant: Variant,
) -> Pipeline:
    if len(indices) < 300:
        raise ValueError(f"Insufficient mature labels: {len(indices)}")
    model = new_model(variant)
    weights = sample_weights(indices, variant.half_life)
    fit_kwargs: dict[str, Any] = {}
    if weights is not None:
        fit_kwargs["logisticregression__sample_weight"] = weights
    model.fit(
        frame.loc[indices, features],
        target.iloc[indices].astype(int),
        **fit_kwargs,
    )
    return model


def target_date(frame: pd.DataFrame, origin_index: int, horizon: int) -> pd.Timestamp:
    index = origin_index + horizon
    if index < len(frame):
        return pd.Timestamp(frame.loc[index, "date"])
    return pd.Timestamp(frame.loc[origin_index, "date"]) + pd.offsets.BDay(horizon)


def prediction_row(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    returns: pd.Series,
    origin_index: int,
    horizon: int,
    model: Pipeline,
    features: list[str],
    train: np.ndarray,
    variant: Variant,
    training_mode: str,
) -> dict[str, Any]:
    probability = float(model.predict_proba(frame.loc[[origin_index], features])[0, 1])
    actual = target.iloc[origin_index]
    actual_return = returns.iloc[origin_index]
    last_label_end = None
    if len(train):
        end_index = int(train[-1]) + horizon
        if end_index < len(frame):
            last_label_end = pd.Timestamp(frame.loc[end_index, "date"])
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    return {
        "iso_week": origin_date.strftime("%G-W%V"),
        "origin_date": origin_date,
        "target_date": target_date(frame, origin_index, horizon),
        "horizon_days": horizon,
        "variant_key": variant.key,
        "feature_group": variant.feature_group,
        "c_value": variant.c_value,
        "class_weight": variant.class_weight,
        "half_life": "expanding" if variant.half_life is None else str(variant.half_life),
        "probability_up": probability,
        "actual": None if pd.isna(actual) else int(actual),
        "actual_log_return": None if pd.isna(actual_return) else float(actual_return),
        "train_up_rate": float(target.iloc[train].mean()),
        "train_count": int(len(train)),
        "train_label_end": last_label_end,
        "training_mode": training_mode,
    }


def walk_forward_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    variant: Variant,
    start: str,
    end: str,
) -> pd.DataFrame:
    target = make_direction_target(frame, horizon)
    returns = make_return_target(frame, horizon)
    rows: list[dict[str, Any]] = []
    for origin_index in weekly_origins(frame, start, end).index:
        train = eligible_train_indices(target, int(origin_index), horizon)
        if len(train) < 300:
            continue
        model = fit_model(frame, target, features, train, variant)
        rows.append(prediction_row(
            frame=frame,
            target=target,
            returns=returns,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            features=features,
            train=train,
            variant=variant,
            training_mode="weekly_expanding_matured_labels",
        ))
    return pd.DataFrame(rows)


def frozen_year_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    variant: Variant,
    year: int,
) -> pd.DataFrame:
    target = make_direction_target(frame, horizon)
    returns = make_return_target(frame, horizon)
    dates = pd.to_datetime(frame["date"])
    cutoff = int(np.searchsorted(dates, pd.Timestamp(f"{year}-01-01")))
    train = np.arange(0, max(0, cutoff - horizon))
    train = train[target.iloc[train].notna().to_numpy()]
    model = fit_model(frame, target, features, train, variant)
    origins = weekly_origins(frame, f"{year}-01-01", f"{year}-12-31")
    return pd.DataFrame([
        prediction_row(
            frame=frame,
            target=target,
            returns=returns,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            features=features,
            train=train,
            variant=variant,
            training_mode=f"frozen_pre_{year}",
        )
        for origin_index in origins.index
    ])


def feature_groups(
    frame: pd.DataFrame,
    price_features: list[str],
    all_features: list[str],
    cutoff: str = "2022-01-01",
) -> dict[str, list[str]]:
    pit = set(frame.attrs.get("forward_pit_features", []))
    classic = [feature for feature in all_features if feature not in pit]
    groups = {
        "price": list(price_features),
        "classic_macro": classic,
        "enriched_pit": list(all_features),
    }
    dates = pd.to_datetime(frame["date"])
    eligible = dates < pd.Timestamp(cutoff)
    result: dict[str, list[str]] = {}
    for name, features in groups.items():
        # Availability is frozen before validation; no validation target is used.
        result[name] = [
            feature for feature in features
            if feature in frame.columns and frame.loc[eligible, feature].notna().sum() >= 250
        ]
        if not result[name]:
            raise ValueError(f"Feature group {name} has no causally available columns")
    return result


def selective_predictions(
    frame: pd.DataFrame, lower: float, upper: float,
) -> pd.DataFrame:
    matured = frame[frame["actual"].notna()].copy()
    prediction = pd.Series(pd.NA, index=matured.index, dtype="Int64")
    prediction.loc[matured["probability_up"] <= lower] = 0
    prediction.loc[matured["probability_up"] >= upper] = 1
    matured["prediction"] = prediction
    return matured[matured["prediction"].notna()].copy()


def safe_auc(actual: pd.Series, probability: pd.Series) -> float | None:
    if actual.nunique() < 2:
        return None
    return float(roc_auc_score(actual.astype(int), probability.astype(float)))


def metrics(frame: pd.DataFrame, lower: float, upper: float) -> dict[str, Any]:
    matured = frame[frame["actual"].notna()].copy()
    selected = selective_predictions(matured, lower, upper)
    if selected.empty:
        return {
            "n_total": int(len(matured)), "n_signals": 0, "coverage": 0.0,
            "directional_accuracy": None, "balanced_accuracy": None,
            "up_recall": None, "down_recall": None, "minimum_class_recall": None,
            "prediction_up_rate": None, "actual_up_rate": None,
            "causal_majority_accuracy": None, "lift_vs_causal_majority": None,
            "oracle_constant_accuracy": None, "lift_vs_oracle_constant": None,
            "auc_all": safe_auc(matured["actual"], matured["probability_up"]),
        }
    actual = selected["actual"].astype(int)
    prediction = selected["prediction"].astype(int)
    up = actual.eq(1)
    down = actual.eq(0)
    up_recall = float(prediction[up].mean()) if up.any() else None
    down_recall = float((1 - prediction[down]).mean()) if down.any() else None
    balanced = (
        0.5 * (up_recall + down_recall)
        if up_recall is not None and down_recall is not None else None
    )
    minimum = (
        min(up_recall, down_recall)
        if up_recall is not None and down_recall is not None else None
    )
    causal_majority = (selected["train_up_rate"] >= 0.5).astype(int)
    directional_accuracy = float((prediction == actual).mean())
    causal_accuracy = float((causal_majority == actual).mean())
    actual_up_rate = float(actual.mean())
    oracle_constant = max(actual_up_rate, 1.0 - actual_up_rate)
    return {
        "n_total": int(len(matured)),
        "n_signals": int(len(selected)),
        "coverage": float(len(selected) / len(matured)) if len(matured) else 0.0,
        "directional_accuracy": directional_accuracy,
        "balanced_accuracy": balanced,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "minimum_class_recall": minimum,
        "prediction_up_rate": float(prediction.mean()),
        "actual_up_rate": actual_up_rate,
        "causal_majority_accuracy": causal_accuracy,
        "lift_vs_causal_majority": directional_accuracy - causal_accuracy,
        "oracle_constant_accuracy": oracle_constant,
        "lift_vs_oracle_constant": directional_accuracy - oracle_constant,
        "auc_all": safe_auc(matured["actual"], matured["probability_up"]),
    }


def candidate_is_eligible(year_metrics: list[dict[str, Any]]) -> bool:
    return all(
        item["n_signals"] >= MIN_VALIDATION_SIGNALS_PER_YEAR
        and item["coverage"] >= MIN_VALIDATION_COVERAGE
        and item["balanced_accuracy"] is not None
        and item["balanced_accuracy"] >= MIN_VALIDATION_YEAR_BALANCED_ACCURACY
        and item["minimum_class_recall"] is not None
        and item["minimum_class_recall"] >= MIN_VALIDATION_CLASS_RECALL
        and item["prediction_up_rate"] is not None
        and MIN_VALIDATION_PREDICTION_CLASS_RATE
        <= item["prediction_up_rate"]
        <= 1.0 - MIN_VALIDATION_PREDICTION_CLASS_RATE
        for item in year_metrics
    )


def candidate_score(year_metrics: list[dict[str, Any]]) -> float:
    values = [item["balanced_accuracy"] for item in year_metrics]
    recalls = [item["minimum_class_recall"] for item in year_metrics]
    accuracies = [item["directional_accuracy"] for item in year_metrics]
    coverages = [item["coverage"] for item in year_metrics]
    if any(value is None for value in values + recalls + accuracies):
        return float("-inf")
    return float(
        0.35 * np.mean(values)
        + 0.30 * np.min(values)
        + 0.15 * np.mean(recalls)
        + 0.15 * np.mean(accuracies)
        + 0.05 * np.min(coverages)
    )


def threshold_grid() -> Iterable[tuple[float, float, float, float]]:
    for center in CONFIDENCE_CENTERS:
        for margin in CONFIDENCE_MARGINS:
            lower = round(center - margin, 3)
            upper = round(center + margin, 3)
            if 0.0 < lower <= upper < 1.0:
                yield center, margin, lower, upper


def select_candidate(
    validation_by_variant: dict[str, pd.DataFrame],
    variants_by_key: dict[str, Variant],
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for key, frame in validation_by_variant.items():
        for center, margin, lower, upper in threshold_grid():
            annual = []
            for year in VALIDATION_YEARS:
                part = frame[frame["origin_date"].dt.year.eq(year)]
                annual.append(metrics(part, lower, upper))
            pooled = metrics(frame, lower, upper)
            row: dict[str, Any] = {
                "horizon_days": horizon,
                "variant_key": key,
                **asdict(variants_by_key[key]),
                "center": center,
                "margin": margin,
                "lower_threshold": lower,
                "upper_threshold": upper,
                "eligible": candidate_is_eligible(annual),
                "selection_score": candidate_score(annual),
                "pooled_balanced_accuracy": pooled["balanced_accuracy"],
                "pooled_directional_accuracy": pooled["directional_accuracy"],
                "pooled_coverage": pooled["coverage"],
                "pooled_auc": pooled["auc_all"],
                "minimum_year_balanced_accuracy": min(
                    item["balanced_accuracy"]
                    if item["balanced_accuracy"] is not None else -np.inf
                    for item in annual
                ),
            }
            for year, item in zip(VALIDATION_YEARS, annual, strict=True):
                for name in (
                    "n_signals", "coverage", "directional_accuracy",
                    "balanced_accuracy", "minimum_class_recall",
                    "lift_vs_causal_majority", "auc_all",
                ):
                    row[f"{year}_{name}"] = item[name]
            rows.append(row)
    candidates = pd.DataFrame(rows)
    pool = candidates[candidates["eligible"]]
    if pool.empty:
        pool = candidates
    best_index = pool.sort_values(
        [
            "selection_score", "minimum_year_balanced_accuracy",
            "pooled_balanced_accuracy", "pooled_coverage", "margin",
        ],
        ascending=[False, False, False, False, True],
    ).index[0]
    best = candidates.loc[best_index].to_dict()
    best["eligible"] = bool(best["eligible"])
    best["candidate_count"] = int(len(candidates))
    best["eligible_candidate_count"] = int(candidates["eligible"].sum())
    return best, candidates


def moving_block_bootstrap(
    values: np.ndarray,
    block_length: int,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, float | None]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < max(8, 2 * block_length):
        return {"mean": float(np.mean(values)) if n else None, "ci_low": None,
                "ci_high": None, "one_sided_p": None}
    rng = np.random.default_rng(seed)
    starts = np.arange(n)
    width = int(math.ceil(n / block_length))
    sampled_means = np.empty(samples, dtype=float)
    centered = values - values.mean()
    for sample in range(samples):
        chosen = rng.choice(starts, size=width, replace=True)
        indices = np.concatenate([
            (np.arange(start, start + block_length) % n) for start in chosen
        ])[:n]
        sampled_means[sample] = values[indices].mean()
    centered_means = sampled_means - values.mean()
    observed = float(values.mean())
    return {
        "mean": observed,
        "ci_low": float(np.quantile(sampled_means, 0.025)),
        "ci_high": float(np.quantile(sampled_means, 0.975)),
        "one_sided_p": float((1 + np.sum(centered_means >= observed)) / (samples + 1)),
    }


def non_overlapping_sensitivity(
    selected: pd.DataFrame, horizon: int,
) -> dict[str, float | int | None]:
    step = max(1, int(math.ceil(horizon / 5)))
    if selected.empty:
        return {"step_weeks": step, "offset_count": 0, "minimum_da": None,
                "mean_da": None, "minimum_balanced_accuracy": None,
                "mean_balanced_accuracy": None}
    da_values: list[float] = []
    balanced_values: list[float] = []
    for offset in range(step):
        part = selected.iloc[offset::step]
        if part.empty:
            continue
        actual = part["actual"].astype(int)
        prediction = part["prediction"].astype(int)
        da_values.append(float((actual == prediction).mean()))
        if actual.nunique() > 1:
            up = actual.eq(1)
            down = actual.eq(0)
            balanced_values.append(float(
                0.5 * (prediction[up].mean() + (1 - prediction[down]).mean())
            ))
    return {
        "step_weeks": step,
        "offset_count": len(da_values),
        "minimum_da": min(da_values) if da_values else None,
        "mean_da": float(np.mean(da_values)) if da_values else None,
        "minimum_balanced_accuracy": min(balanced_values) if balanced_values else None,
        "mean_balanced_accuracy": (
            float(np.mean(balanced_values)) if balanced_values else None
        ),
    }


def evaluation_row(
    frame: pd.DataFrame,
    horizon: int,
    period: str,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    result = metrics(frame, lower, upper)
    selected = selective_predictions(frame, lower, upper)
    if selected.empty:
        differences = np.array([], dtype=float)
    else:
        actual = selected["actual"].astype(int)
        prediction = selected["prediction"].astype(int)
        baseline = (selected["train_up_rate"] >= 0.5).astype(int)
        differences = (
            (prediction == actual).astype(float) - (baseline == actual).astype(float)
        ).to_numpy()
    block_length = max(1, int(math.ceil(horizon / 5)))
    bootstrap = moving_block_bootstrap(
        differences, block_length, seed=RANDOM_SEED + horizon,
    )
    sensitivity = non_overlapping_sensitivity(selected, horizon)
    return {
        "period": period,
        "horizon_days": horizon,
        "lower_threshold": lower,
        "upper_threshold": upper,
        **result,
        "overlap_block_weeks": block_length,
        "lift_bootstrap_ci_low": bootstrap["ci_low"],
        "lift_bootstrap_ci_high": bootstrap["ci_high"],
        "lift_bootstrap_one_sided_p": bootstrap["one_sided_p"],
        **{f"nonoverlap_{key}": value for key, value in sensitivity.items()},
    }


def promotion_gate(row: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = {
        "coverage>=25%": row["coverage"] is not None and row["coverage"] >= 0.25,
        "signals>=12": row["n_signals"] >= 12,
        "DA>=55%": (
            row["directional_accuracy"] is not None
            and row["directional_accuracy"] >= 0.55
        ),
        "BDA>=55%": (
            row["balanced_accuracy"] is not None
            and row["balanced_accuracy"] >= 0.55
        ),
        "min_recall>=35%": (
            row["minimum_class_recall"] is not None
            and row["minimum_class_recall"] >= 0.35
        ),
        "lift_causal_majority>0": (
            row["lift_vs_causal_majority"] is not None
            and row["lift_vs_causal_majority"] > 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return not failed, failed


def safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a smaller diagnostic grid (C=0.1, half-lives expanding/252).",
    )
    parser.add_argument(
        "--output-prefix", default="usdcop_directional_edge_tournament",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, price, all_candidates = build_frame(
        include_forward_pit=True, promotion_only=False,
    )
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])
    groups = feature_groups(frame, price, all_candidates)

    c_values = (0.1,) if args.quick else (0.03, 0.1)
    half_lives = (None, 252) if args.quick else (None, 252, 504)
    variants = [
        Variant(group, c_value, class_weight, half_life)
        for group in groups
        for c_value in c_values
        for class_weight in ("unweighted", "balanced")
        for half_life in half_lives
    ]
    variants_by_key = {variant.key: variant for variant in variants}

    report_dir = ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    all_candidate_frames: list[pd.DataFrame] = []
    selection_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        print(f"[H{horizon}] validation tournament: {len(variants)} model variants", flush=True)
        validation_by_variant: dict[str, pd.DataFrame] = {}
        for index, variant in enumerate(variants, start=1):
            predictions = walk_forward_probabilities(
                frame,
                groups[variant.feature_group],
                horizon,
                variant,
                "2022-01-01",
                "2024-12-31",
            )
            validation_by_variant[variant.key] = predictions
            if index % max(1, len(variants) // 3) == 0:
                print(f"[H{horizon}] fitted {index}/{len(variants)} variants", flush=True)

        best, candidates = select_candidate(
            validation_by_variant, variants_by_key, horizon,
        )
        all_candidate_frames.append(candidates)
        variant = variants_by_key[str(best["variant_key"])]
        lower = float(best["lower_threshold"])
        upper = float(best["upper_threshold"])
        features = groups[variant.feature_group]
        feature_hash = hashlib.sha256("|".join(features).encode()).hexdigest()[:16]
        best["feature_count"] = len(features)
        best["feature_hash"] = feature_hash
        best["selected_features"] = "|".join(features)
        selection_rows.append(best)
        print(
            f"[H{horizon}] selected {variant.key}, thresholds={lower:.2f}/{upper:.2f}, "
            f"eligible={best['eligible']}",
            flush=True,
        )

        validation = validation_by_variant[variant.key].copy()
        frozen_2025 = frozen_year_probabilities(
            frame, features, horizon, variant, 2025,
        )
        expanding_2026 = walk_forward_probabilities(
            frame, features, horizon, variant, "2026-01-01", "2026-12-31",
        )
        for period, predictions in (
            ("2022_2024_VALIDATION", validation),
            ("2025_FROZEN_OOS", frozen_2025),
            ("2026_EXPANDING_YTD", expanding_2026),
        ):
            predictions = predictions.copy()
            predictions["period"] = period
            predictions["lower_threshold"] = lower
            predictions["upper_threshold"] = upper
            selected = selective_predictions(predictions, lower, upper)
            signal_by_index = selected["prediction"].astype(int)
            predictions["prediction"] = pd.Series(pd.NA, index=predictions.index, dtype="Int64")
            predictions.loc[signal_by_index.index, "prediction"] = signal_by_index
            predictions["decision"] = predictions["prediction"].map({0: "DOWN", 1: "UP"}).fillna("FLAT")
            prediction_frames.append(predictions)
            row = evaluation_row(predictions, horizon, period, lower, upper)
            passed, failed = promotion_gate(row)
            row["research_metric_gate_passed"] = passed
            row["failed_metric_gates"] = "|".join(failed)
            evaluation_rows.append(row)

    candidates = pd.concat(all_candidate_frames, ignore_index=True)
    selections = pd.DataFrame(selection_rows)
    evaluations = pd.DataFrame(evaluation_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)

    candidates_path = report_dir / f"{args.output_prefix}_candidates.csv"
    selections_path = report_dir / f"{args.output_prefix}_selections.csv"
    evaluations_path = report_dir / f"{args.output_prefix}_metrics.csv"
    predictions_path = report_dir / f"{args.output_prefix}_predictions.csv"
    manifest_path = report_dir / f"{args.output_prefix}_manifest.json"
    candidates.to_csv(candidates_path, index=False)
    selections.to_csv(selections_path, index=False)
    evaluations.to_csv(evaluations_path, index=False)
    predictions.to_csv(predictions_path, index=False)

    generalization: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        rows = evaluations[
            evaluations["horizon_days"].eq(horizon)
            & evaluations["period"].isin(("2025_FROZEN_OOS", "2026_EXPANDING_YTD"))
        ]
        passed = bool(len(rows) == 2 and rows["research_metric_gate_passed"].all())
        generalization.append({
            "horizon_days": horizon,
            "passes_2025_and_2026_metric_gates": passed,
            "institutional_promotion_authorized": False,
            "reason": (
                "Metric gates passed, but fresh prospective evidence and multiple-testing "
                "control are still required."
                if passed else "One or both replay periods fail the declared metric gates."
            ),
        })
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "USD/COP",
        "signal_authorized": False,
        "selection_protocol": {
            "feature_availability_cutoff": "2022-01-01",
            "model_and_threshold_validation_years": list(VALIDATION_YEARS),
            "frozen_oos_year": 2025,
            "weekly_expanding_year": 2026,
            "label_maturity": "origin_index_minus_horizon",
            "weekly_origin": "last_available_session_per_iso_week",
            "candidate_count_per_horizon": int(len(variants) * len(list(threshold_grid()))),
            "feature_groups": {key: len(value) for key, value in groups.items()},
            "metric_gates": {
                "minimum_coverage": 0.25,
                "minimum_signals": 12,
                "minimum_directional_accuracy": 0.55,
                "minimum_balanced_accuracy": 0.55,
                "minimum_class_recall": 0.35,
                "positive_lift_vs_causal_majority": True,
            },
        },
        "important_limitations": [
            "The research team has already inspected 2025 and 2026 outcomes; neither is a fresh prospective holdout.",
            "A candidate search requires explicit multiple-testing control before capital promotion.",
            "Overlapping horizon labels reduce effective sample size; block bootstrap and non-overlap sensitivity are reported.",
            "Economic value after spreads, slippage and position sizing is outside this directional-only tournament.",
        ],
        "generalization": generalization,
        "outputs": {
            "candidates": str(candidates_path.relative_to(ROOT)),
            "selections": str(selections_path.relative_to(ROOT)),
            "metrics": str(evaluations_path.relative_to(ROOT)),
            "predictions": str(predictions_path.relative_to(ROOT)),
        },
    }
    manifest_path.write_text(
        json.dumps(safe_json(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    display_columns = [
        "period", "horizon_days", "n_signals", "coverage",
        "directional_accuracy", "balanced_accuracy", "up_recall", "down_recall",
        "lift_vs_causal_majority", "auc_all", "research_metric_gate_passed",
    ]
    print(evaluations[display_columns].to_string(index=False), flush=True)
    print(f"Wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
