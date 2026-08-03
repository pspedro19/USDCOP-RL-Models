"""Two-stage event-then-direction USD/COP tournament.

Stage 1 predicts whether the forward move exceeds a causal volatility-scaled
noise band. Stage 2 predicts UP/DOWN conditional on historical event labels.
The live decision is FLAT unless both event and directional confidence gates
pass. Rule selection is confined to 2015-2019; 2020 onward is not selected on.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_directional_edge_tournament import (
    HORIZONS,
    candidate_score,
    evaluation_row,
    metrics,
    promotion_gate,
    safe_json,
    sample_weights,
    selective_predictions,
    weekly_origins,
)
from scripts.analysis.usdcop_long_history_directional_tournament import (
    FEATURES,
    ROBUSTNESS_YEARS,
    SELECTION_YEARS,
    build_long_history_frame,
)


NOISE_FLOOR = 0.001
EVENT_SCALES = (0.35, 0.75)
HALF_LIVES = (252, 1260)
EVENT_THRESHOLDS = (0.45, 0.50, 0.55, 0.60)
DIRECTION_MARGINS = (0.025, 0.05, 0.10)


@dataclass(frozen=True)
class EventVariant:
    event_scale: float
    half_life: int

    @property
    def key(self) -> str:
        return f"event_scale={self.event_scale:g}|hl={self.half_life}"


def targets(
    frame: pd.DataFrame, horizon: int, event_scale: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    forward_return = np.log(frame["close"].shift(-horizon) / frame["close"])
    direction = (forward_return > 0).where(forward_return.notna()).astype(float)
    ex_ante_vol = frame["volatility_20d"].astype(float) * math.sqrt(horizon)
    barrier = pd.Series(
        np.maximum(NOISE_FLOOR, event_scale * ex_ante_vol),
        index=frame.index,
        dtype=float,
    )
    event = (forward_return.abs() > barrier).where(
        forward_return.notna() & barrier.notna()
    ).astype(float)
    event_direction = direction.where(event.eq(1.0))
    return forward_return, direction, event, event_direction


def mature_indices(
    target: pd.Series, origin_index: int, horizon: int,
) -> np.ndarray:
    indices = np.arange(0, max(0, origin_index - horizon + 1))
    return indices[target.iloc[indices].notna().to_numpy()]


def new_classifier() -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2_000,
            random_state=23,
        ),
    )


def fit_classifier(
    frame: pd.DataFrame,
    target: pd.Series,
    train: np.ndarray,
    half_life: int,
) -> Pipeline:
    if len(train) < 200:
        raise ValueError(f"Insufficient mature labels: {len(train)}")
    y = target.iloc[train].astype(int)
    if y.nunique() < 2 or y.value_counts().min() < 20:
        raise ValueError("Insufficient observations in one classifier class")
    model = new_classifier()
    weights = sample_weights(train, half_life)
    model.fit(
        frame.loc[train, FEATURES],
        y,
        logisticregression__sample_weight=weights,
    )
    return model


def prediction_row(
    *,
    frame: pd.DataFrame,
    origin_index: int,
    horizon: int,
    variant: EventVariant,
    forward_return: pd.Series,
    direction: pd.Series,
    event: pd.Series,
    barrier: pd.Series,
    event_model: Pipeline,
    direction_model: Pipeline,
    event_train: np.ndarray,
    direction_train: np.ndarray,
    training_mode: str,
) -> dict[str, Any]:
    origin_date = pd.Timestamp(frame.loc[origin_index, "date"])
    target_index = origin_index + horizon
    target_date = (
        pd.Timestamp(frame.loc[target_index, "date"])
        if target_index < len(frame)
        else origin_date + pd.offsets.BDay(horizon)
    )
    last_label_end = None
    if len(event_train):
        end_index = int(event_train[-1]) + horizon
        if end_index < len(frame):
            last_label_end = pd.Timestamp(frame.loc[end_index, "date"])
    actual_direction = direction.iloc[origin_index]
    actual_event = event.iloc[origin_index]
    actual_return = forward_return.iloc[origin_index]
    probability_event = float(
        event_model.predict_proba(frame.loc[[origin_index], FEATURES])[0, 1]
    )
    probability_up = float(
        direction_model.predict_proba(frame.loc[[origin_index], FEATURES])[0, 1]
    )
    all_direction_train = mature_indices(direction, origin_index, horizon)
    return {
        "iso_week": origin_date.strftime("%G-W%V"),
        "origin_date": origin_date,
        "target_date": target_date,
        "horizon_days": horizon,
        "variant_key": variant.key,
        "event_scale": variant.event_scale,
        "half_life": variant.half_life,
        "event_barrier_log_return": float(barrier.iloc[origin_index]),
        "probability_event": probability_event,
        "probability_up_raw": probability_up,
        "probability_up": probability_up,
        "actual": None if pd.isna(actual_direction) else int(actual_direction),
        "actual_event": None if pd.isna(actual_event) else int(actual_event),
        "actual_log_return": None if pd.isna(actual_return) else float(actual_return),
        "train_up_rate": float(direction.iloc[all_direction_train].mean()),
        "event_train_count": int(len(event_train)),
        "direction_train_count": int(len(direction_train)),
        "train_label_end": last_label_end,
        "training_mode": training_mode,
    }


def fit_pair(
    frame: pd.DataFrame,
    origin_index: int,
    horizon: int,
    variant: EventVariant,
    event: pd.Series,
    event_direction: pd.Series,
) -> tuple[Pipeline, Pipeline, np.ndarray, np.ndarray]:
    event_train = mature_indices(event, origin_index, horizon)
    direction_train = mature_indices(event_direction, origin_index, horizon)
    event_model = fit_classifier(frame, event, event_train, variant.half_life)
    direction_model = fit_classifier(
        frame, event_direction, direction_train, variant.half_life,
    )
    return event_model, direction_model, event_train, direction_train


def walk_forward(
    frame: pd.DataFrame,
    horizon: int,
    variant: EventVariant,
    start: str,
    end: str,
) -> pd.DataFrame:
    forward_return, direction, event, event_direction = targets(
        frame, horizon, variant.event_scale,
    )
    ex_ante_vol = frame["volatility_20d"].astype(float) * math.sqrt(horizon)
    barrier = pd.Series(
        np.maximum(NOISE_FLOOR, variant.event_scale * ex_ante_vol),
        index=frame.index,
    )
    rows: list[dict[str, Any]] = []
    for origin_index in weekly_origins(frame, start, end).index:
        origin_index = int(origin_index)
        try:
            event_model, direction_model, event_train, direction_train = fit_pair(
                frame, origin_index, horizon, variant, event, event_direction,
            )
        except ValueError:
            continue
        rows.append(prediction_row(
            frame=frame,
            origin_index=origin_index,
            horizon=horizon,
            variant=variant,
            forward_return=forward_return,
            direction=direction,
            event=event,
            barrier=barrier,
            event_model=event_model,
            direction_model=direction_model,
            event_train=event_train,
            direction_train=direction_train,
            training_mode="weekly_expanding_matured_labels",
        ))
    return pd.DataFrame(rows)


def frozen_year(
    frame: pd.DataFrame,
    horizon: int,
    variant: EventVariant,
    year: int,
) -> pd.DataFrame:
    forward_return, direction, event, event_direction = targets(
        frame, horizon, variant.event_scale,
    )
    ex_ante_vol = frame["volatility_20d"].astype(float) * math.sqrt(horizon)
    barrier = pd.Series(
        np.maximum(NOISE_FLOOR, variant.event_scale * ex_ante_vol),
        index=frame.index,
    )
    dates = pd.to_datetime(frame["date"])
    cutoff = int(np.searchsorted(dates, pd.Timestamp(f"{year}-01-01")))
    pseudo_origin = cutoff - 1
    event_model, direction_model, event_train, direction_train = fit_pair(
        frame, pseudo_origin, horizon, variant, event, event_direction,
    )
    origins = weekly_origins(frame, f"{year}-01-01", f"{year}-12-31")
    return pd.DataFrame([
        prediction_row(
            frame=frame,
            origin_index=int(origin_index),
            horizon=horizon,
            variant=variant,
            forward_return=forward_return,
            direction=direction,
            event=event,
            barrier=barrier,
            event_model=event_model,
            direction_model=direction_model,
            event_train=event_train,
            direction_train=direction_train,
            training_mode=f"frozen_pre_{year}",
        )
        for origin_index in origins.index
    ])


def apply_policy(
    frame: pd.DataFrame,
    event_threshold: float,
    direction_margin: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["probability_up"] = result["probability_up_raw"]
    event_failed = result["probability_event"] < event_threshold
    # With a strictly positive margin, 0.5 is guaranteed to abstain.
    result.loc[event_failed, "probability_up"] = 0.5
    result["event_threshold"] = event_threshold
    result["direction_margin"] = direction_margin
    result["lower_threshold"] = 0.5 - direction_margin
    result["upper_threshold"] = 0.5 + direction_margin
    result["prediction"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    signals = selective_predictions(
        result, 0.5 - direction_margin, 0.5 + direction_margin,
    )
    result.loc[signals.index, "prediction"] = signals["prediction"].astype(int)
    result["decision"] = result["prediction"].map({0: "DOWN", 1: "UP"}).fillna("FLAT")
    return result


def policy_metrics(
    frame: pd.DataFrame,
    event_threshold: float,
    direction_margin: float,
) -> dict[str, Any]:
    policy = apply_policy(frame, event_threshold, direction_margin)
    lower = 0.5 - direction_margin
    upper = 0.5 + direction_margin
    result = metrics(policy, lower, upper)
    selected = selective_predictions(policy, lower, upper)
    result["event_precision"] = (
        float(selected["actual_event"].astype(float).mean())
        if len(selected) else None
    )
    result["mean_absolute_realized_return"] = (
        float(selected["actual_log_return"].abs().mean())
        if len(selected) else None
    )
    return result


def selection_eligible(annual: list[dict[str, Any]]) -> bool:
    return all(
        item["n_signals"] >= 8
        and item["coverage"] >= 0.15
        and item["balanced_accuracy"] is not None
        and item["balanced_accuracy"] >= 0.48
        and item["minimum_class_recall"] is not None
        and item["minimum_class_recall"] >= 0.20
        and item["prediction_up_rate"] is not None
        and 0.10 <= item["prediction_up_rate"] <= 0.90
        for item in annual
    )


def select_policy(
    predictions_by_variant: dict[str, pd.DataFrame],
    variants: dict[str, EventVariant],
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for key, predictions in predictions_by_variant.items():
        for event_threshold in EVENT_THRESHOLDS:
            for direction_margin in DIRECTION_MARGINS:
                annual = [
                    policy_metrics(
                        predictions[predictions["origin_date"].dt.year.eq(year)],
                        event_threshold,
                        direction_margin,
                    )
                    for year in SELECTION_YEARS
                ]
                pooled = policy_metrics(
                    predictions, event_threshold, direction_margin,
                )
                row = {
                    "horizon_days": horizon,
                    "variant_key": key,
                    "event_scale": variants[key].event_scale,
                    "half_life": variants[key].half_life,
                    "event_threshold": event_threshold,
                    "direction_margin": direction_margin,
                    "lower_threshold": 0.5 - direction_margin,
                    "upper_threshold": 0.5 + direction_margin,
                    "eligible": selection_eligible(annual),
                    "selection_score": candidate_score(annual),
                    "pooled_da": pooled["directional_accuracy"],
                    "pooled_bda": pooled["balanced_accuracy"],
                    "pooled_coverage": pooled["coverage"],
                    "pooled_event_precision": pooled["event_precision"],
                    "minimum_year_bda": min(
                        item["balanced_accuracy"]
                        if item["balanced_accuracy"] is not None else -np.inf
                        for item in annual
                    ),
                }
                for year, item in zip(SELECTION_YEARS, annual, strict=True):
                    row[f"{year}_da"] = item["directional_accuracy"]
                    row[f"{year}_bda"] = item["balanced_accuracy"]
                    row[f"{year}_coverage"] = item["coverage"]
                    row[f"{year}_min_recall"] = item["minimum_class_recall"]
                rows.append(row)
    candidates = pd.DataFrame(rows)
    pool = candidates[candidates["eligible"]]
    if pool.empty:
        pool = candidates
    best = pool.sort_values(
        ["selection_score", "minimum_year_bda", "pooled_bda", "pooled_coverage", "direction_margin"],
        ascending=[False, False, False, False, True],
    ).iloc[0].to_dict()
    best["eligible"] = bool(best["eligible"])
    best["candidate_count"] = int(len(candidates))
    best["eligible_candidate_count"] = int(candidates["eligible"].sum())
    return best, candidates


def evaluate_period(
    predictions: pd.DataFrame,
    period: str,
    horizon: int,
    event_threshold: float,
    direction_margin: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    policy = apply_policy(predictions, event_threshold, direction_margin)
    policy["period"] = period
    lower = 0.5 - direction_margin
    upper = 0.5 + direction_margin
    row = evaluation_row(policy, horizon, period, lower, upper)
    extra = policy_metrics(predictions, event_threshold, direction_margin)
    row["event_precision"] = extra["event_precision"]
    row["mean_absolute_realized_return"] = extra["mean_absolute_realized_return"]
    passed, failed = promotion_gate(row)
    row["research_metric_gate_passed"] = passed
    row["failed_metric_gates"] = "|".join(failed)
    return policy, row


def main() -> None:
    frame, provenance = build_long_history_frame()
    variants_list = [
        EventVariant(event_scale, half_life)
        for event_scale in EVENT_SCALES
        for half_life in HALF_LIVES
    ]
    variants = {variant.key: variant for variant in variants_list}
    candidates_all: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    metrics_all: list[dict[str, Any]] = []
    predictions_all: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        print(f"[H{horizon}] event-direction selection", flush=True)
        validation_by_variant = {
            variant.key: walk_forward(
                frame, horizon, variant, "2015-01-01", "2019-12-31",
            )
            for variant in variants_list
        }
        best, candidates = select_policy(
            validation_by_variant, variants, horizon,
        )
        candidates_all.append(candidates)
        variant = variants[str(best["variant_key"])]
        event_threshold = float(best["event_threshold"])
        direction_margin = float(best["direction_margin"])
        selections.append(best)

        blocks = (
            ("2015_2019_SELECTION", validation_by_variant[variant.key]),
            (
                "2020_2024_UNTOUCHED_ROBUSTNESS",
                walk_forward(
                    frame, horizon, variant, "2020-01-01", "2024-12-31",
                ),
            ),
            ("2025_FROZEN_OOS", frozen_year(frame, horizon, variant, 2025)),
            (
                "2026_EXPANDING_YTD",
                walk_forward(
                    frame, horizon, variant, "2026-01-01", "2026-12-31",
                ),
            ),
        )
        for period, predictions in blocks:
            policy, row = evaluate_period(
                predictions,
                period,
                horizon,
                event_threshold,
                direction_margin,
            )
            predictions_all.append(policy)
            metrics_all.append(row)
            if period == "2020_2024_UNTOUCHED_ROBUSTNESS":
                for year in ROBUSTNESS_YEARS:
                    annual = predictions[
                        predictions["origin_date"].dt.year.eq(year)
                    ]
                    _, annual_row = evaluate_period(
                        annual,
                        f"{year}_UNTOUCHED_ROBUSTNESS",
                        horizon,
                        event_threshold,
                        direction_margin,
                    )
                    metrics_all.append(annual_row)
        print(
            f"[H{horizon}] scale={variant.event_scale:g}, hl={variant.half_life}, "
            f"p_event>={event_threshold:.2f}, margin={direction_margin:.3f}, "
            f"eligible={best['eligible']}",
            flush=True,
        )

    report = ROOT / "reports"
    prefix = "usdcop_event_direction_tournament"
    candidate_frame = pd.concat(candidates_all, ignore_index=True)
    selection_frame = pd.DataFrame(selections)
    metric_frame = pd.DataFrame(metrics_all)
    prediction_frame = pd.concat(predictions_all, ignore_index=True)
    candidate_frame.to_csv(report / f"{prefix}_candidates.csv", index=False)
    selection_frame.to_csv(report / f"{prefix}_selections.csv", index=False)
    metric_frame.to_csv(report / f"{prefix}_metrics.csv", index=False)
    prediction_frame.to_csv(report / f"{prefix}_predictions.csv", index=False)

    required = (
        "2020_2024_UNTOUCHED_ROBUSTNESS", "2025_FROZEN_OOS", "2026_EXPANDING_YTD",
    )
    generalization = []
    for horizon in HORIZONS:
        rows = metric_frame[
            metric_frame["horizon_days"].eq(horizon)
            & metric_frame["period"].isin(required)
        ]
        generalization.append({
            "horizon_days": horizon,
            "passes_all_research_metric_gates": bool(
                len(rows) == len(required)
                and rows["research_metric_gate_passed"].all()
            ),
            "institutional_promotion_authorized": False,
        })
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "USD/COP",
        "signal_authorized": False,
        "architecture": "event_then_direction_with_abstention",
        "selection_years": list(SELECTION_YEARS),
        "untouched_robustness_years": list(ROBUSTNESS_YEARS),
        "noise_floor_log_return": NOISE_FLOOR,
        "event_scales": list(EVENT_SCALES),
        "half_lives": list(HALF_LIVES),
        "feature_count": len(FEATURES),
        "provenance": provenance,
        "generalization": generalization,
        "limitations": [
            "This compact tournament intentionally tests only four architecture variants to limit multiple testing.",
            "2025 and 2026 are audit OOS because earlier research has already inspected their outcomes.",
            "Prospective shadow evidence remains mandatory before capital promotion.",
        ],
    }
    (report / f"{prefix}_manifest.json").write_text(
        json.dumps(safe_json(manifest), indent=2), encoding="utf-8",
    )
    columns = [
        "period", "horizon_days", "n_signals", "coverage",
        "directional_accuracy", "balanced_accuracy", "up_recall", "down_recall",
        "lift_vs_causal_majority", "event_precision",
        "research_metric_gate_passed",
    ]
    print(metric_frame[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
