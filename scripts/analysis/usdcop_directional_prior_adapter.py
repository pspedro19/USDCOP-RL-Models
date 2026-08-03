"""Causal prior-shift adapter for selected USD/COP directional classifiers.

The adapter changes only the intercept of an existing probability using the
recent, fully-matured UP/DOWN base rate.  Its window, shrinkage, strength and
UP/DOWN abstention thresholds are selected on 2022-2024.  The 2025 prior is
frozen at the start of that year; the 2026 prior expands causally each week.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_directional_edge_tournament import (
    HORIZONS,
    VALIDATION_YEARS,
    candidate_is_eligible,
    candidate_score,
    evaluation_row,
    make_direction_target,
    metrics,
    promotion_gate,
    safe_json,
    selective_predictions,
    threshold_grid,
)
from scripts.analysis.weekly_forecasting_oos_canonical import build_frame


PRIOR_WINDOWS = (20, 60, 120, 252)
PRIOR_SHRINKAGES = (10.0, 30.0, 60.0)
PRIOR_STRENGTHS = (0.5, 1.0)


def logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-5, 1.0 - 1e-5))
    return float(np.log(clipped / (1.0 - clipped)))


def expit(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0))))


def mature_indices(origin_index: int, horizon: int) -> np.ndarray:
    return np.arange(0, max(0, origin_index - horizon + 1))


def attach_prior_adjustments(
    predictions: pd.DataFrame,
    frame: pd.DataFrame,
    horizon: int,
    window: int,
    shrinkage: float,
    strength: float,
) -> pd.DataFrame:
    adjusted = predictions.copy()
    target = make_direction_target(frame, horizon)
    dates = pd.to_datetime(frame["date"])
    index_by_date = {date.date(): index for index, date in enumerate(dates)}
    frozen_cutoff = int(np.searchsorted(dates, pd.Timestamp("2025-01-01")))
    values: list[float] = []
    recent_priors: list[float] = []
    for row in adjusted.itertuples(index=False):
        origin_index = index_by_date[pd.Timestamp(row.origin_date).date()]
        if str(row.period) == "2025_FROZEN_OOS":
            prior_origin = frozen_cutoff - 1
        else:
            prior_origin = origin_index
        eligible = mature_indices(prior_origin, horizon)
        eligible = eligible[target.iloc[eligible].notna().to_numpy()]
        recent = target.iloc[eligible[-window:]].dropna()
        full_prior = float(row.train_up_rate)
        recent_prior = float(
            (recent.sum() + shrinkage * full_prior) / (len(recent) + shrinkage)
        )
        probability = expit(
            logit(float(row.probability_up))
            + strength * (logit(recent_prior) - logit(full_prior))
        )
        values.append(probability)
        recent_priors.append(recent_prior)
    adjusted["probability_up_raw"] = adjusted["probability_up"]
    adjusted["probability_up"] = values
    adjusted["recent_up_prior"] = recent_priors
    adjusted["prior_window"] = window
    adjusted["prior_shrinkage"] = shrinkage
    adjusted["prior_strength"] = strength
    return adjusted


def annual_candidate(
    frame: pd.DataFrame,
    lower: float,
    upper: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annual = [
        metrics(frame[frame["origin_date"].dt.year.eq(year)], lower, upper)
        for year in VALIDATION_YEARS
    ]
    return annual, metrics(frame, lower, upper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-prefix", default="usdcop_directional_edge_tournament_quick",
    )
    parser.add_argument(
        "--output-prefix", default="usdcop_directional_prior_adapter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_dir = ROOT / "reports"
    source_path = report_dir / f"{args.input_prefix}_predictions.csv"
    source = pd.read_csv(source_path, parse_dates=["origin_date", "target_date"])
    frame, _, _ = build_frame(include_forward_pit=True, promotion_only=False)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])

    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    selected_predictions: list[pd.DataFrame] = []
    candidate_rows: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        raw = source[source["horizon_days"].eq(horizon)].copy()
        validation_raw = raw[raw["period"].eq("2022_2024_VALIDATION")]
        trials: list[dict[str, Any]] = []

        # The zero-strength control ensures the adapter must beat no adjustment.
        adapter_rules = [(0, 0.0, 0.0)] + [
            (window, shrinkage, strength)
            for window in PRIOR_WINDOWS
            for shrinkage in PRIOR_SHRINKAGES
            for strength in PRIOR_STRENGTHS
        ]
        adjusted_cache: dict[tuple[int, float, float], pd.DataFrame] = {}
        for window, shrinkage, strength in adapter_rules:
            if strength == 0.0:
                adjusted = validation_raw.copy()
                adjusted["probability_up_raw"] = adjusted["probability_up"]
                adjusted["recent_up_prior"] = adjusted["train_up_rate"]
                adjusted["prior_window"] = window
                adjusted["prior_shrinkage"] = shrinkage
                adjusted["prior_strength"] = strength
            else:
                adjusted = attach_prior_adjustments(
                    validation_raw, frame, horizon, window, shrinkage, strength,
                )
            adjusted_cache[(window, shrinkage, strength)] = adjusted
            for center, margin, lower, upper in threshold_grid():
                annual, pooled = annual_candidate(adjusted, lower, upper)
                trial = {
                    "horizon_days": horizon,
                    "prior_window": window,
                    "prior_shrinkage": shrinkage,
                    "prior_strength": strength,
                    "center": center,
                    "margin": margin,
                    "lower_threshold": lower,
                    "upper_threshold": upper,
                    "eligible": candidate_is_eligible(annual),
                    "selection_score": candidate_score(annual),
                    "pooled_da": pooled["directional_accuracy"],
                    "pooled_bda": pooled["balanced_accuracy"],
                    "pooled_coverage": pooled["coverage"],
                    "minimum_year_bda": min(
                        item["balanced_accuracy"]
                        if item["balanced_accuracy"] is not None else -np.inf
                        for item in annual
                    ),
                }
                for year, item in zip(VALIDATION_YEARS, annual, strict=True):
                    trial[f"{year}_da"] = item["directional_accuracy"]
                    trial[f"{year}_bda"] = item["balanced_accuracy"]
                    trial[f"{year}_coverage"] = item["coverage"]
                    trial[f"{year}_min_recall"] = item["minimum_class_recall"]
                trials.append(trial)

        trial_frame = pd.DataFrame(trials)
        candidate_rows.extend(trial_frame.to_dict(orient="records"))
        pool = trial_frame[trial_frame["eligible"]]
        if pool.empty:
            pool = trial_frame
        best = pool.sort_values(
            ["selection_score", "minimum_year_bda", "pooled_bda", "pooled_coverage", "margin"],
            ascending=[False, False, False, False, True],
        ).iloc[0].to_dict()
        best["eligible"] = bool(best["eligible"])
        best["candidate_count"] = int(len(trial_frame))
        best["eligible_candidate_count"] = int(trial_frame["eligible"].sum())
        selection_rows.append(best)

        window = int(best["prior_window"])
        shrinkage = float(best["prior_shrinkage"])
        strength = float(best["prior_strength"])
        lower = float(best["lower_threshold"])
        upper = float(best["upper_threshold"])
        if strength == 0.0:
            chosen = raw.copy()
            chosen["probability_up_raw"] = chosen["probability_up"]
            chosen["recent_up_prior"] = chosen["train_up_rate"]
            chosen["prior_window"] = window
            chosen["prior_shrinkage"] = shrinkage
            chosen["prior_strength"] = strength
        else:
            chosen = attach_prior_adjustments(
                raw, frame, horizon, window, shrinkage, strength,
            )
        chosen["lower_threshold"] = lower
        chosen["upper_threshold"] = upper
        chosen["prediction"] = pd.Series(pd.NA, index=chosen.index, dtype="Int64")
        signals = selective_predictions(chosen, lower, upper)
        chosen.loc[signals.index, "prediction"] = signals["prediction"].astype(int)
        chosen["decision"] = chosen["prediction"].map({0: "DOWN", 1: "UP"}).fillna("FLAT")
        selected_predictions.append(chosen)

        for period, part in chosen.groupby("period", sort=False):
            row = evaluation_row(part, horizon, str(period), lower, upper)
            passed, failed = promotion_gate(row)
            row["research_metric_gate_passed"] = passed
            row["failed_metric_gates"] = "|".join(failed)
            metric_rows.append(row)
        print(
            f"[H{horizon}] prior window={window}, shrink={shrinkage:g}, "
            f"strength={strength:g}, thresholds={lower:.2f}/{upper:.2f}",
            flush=True,
        )

    candidates = pd.DataFrame(candidate_rows)
    selections = pd.DataFrame(selection_rows)
    metrics_frame = pd.DataFrame(metric_rows)
    predictions = pd.concat(selected_predictions, ignore_index=True)
    candidates.to_csv(report_dir / f"{args.output_prefix}_candidates.csv", index=False)
    selections.to_csv(report_dir / f"{args.output_prefix}_selections.csv", index=False)
    metrics_frame.to_csv(report_dir / f"{args.output_prefix}_metrics.csv", index=False)
    predictions.to_csv(report_dir / f"{args.output_prefix}_predictions.csv", index=False)

    replay = metrics_frame[metrics_frame["period"].isin((
        "2025_FROZEN_OOS", "2026_EXPANDING_YTD",
    ))]
    generalized = []
    for horizon in HORIZONS:
        part = replay[replay["horizon_days"].eq(horizon)]
        generalized.append({
            "horizon_days": horizon,
            "passes_both_metric_gates": bool(
                len(part) == 2 and part["research_metric_gate_passed"].all()
            ),
            "institutional_promotion_authorized": False,
        })
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "USD/COP",
        "signal_authorized": False,
        "input_predictions": str(source_path.relative_to(ROOT)),
        "selection_years": list(VALIDATION_YEARS),
        "frozen_prior_year": 2025,
        "expanding_prior_year": 2026,
        "generalization": generalized,
        "note": (
            "Prior adaptation addresses label/base-rate shift only; it cannot repair "
            "a feature ranking whose AUC reverses out of sample."
        ),
    }
    (report_dir / f"{args.output_prefix}_manifest.json").write_text(
        json.dumps(safe_json(manifest), indent=2), encoding="utf-8",
    )
    columns = [
        "period", "horizon_days", "n_signals", "coverage",
        "directional_accuracy", "balanced_accuracy", "up_recall", "down_recall",
        "lift_vs_causal_majority", "auc_all", "research_metric_gate_passed",
    ]
    print(metrics_frame[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
