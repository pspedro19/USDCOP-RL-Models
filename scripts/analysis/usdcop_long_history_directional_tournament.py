"""Long-history, price-only USD/COP directional tournament.

Purpose
-------
Test whether the short 2019-present training history is a root cause of unstable
directional forecasts.  The research frame joins the vetted TwelveData deep
history through 2019-12-17 with the current daily seed from 2019-12-18 onward.
Only the post-band floating regime (2000 onward) is admitted.

Protocol
--------
* 2000-2014: initial estimation history;
* 2015-2019: weekly walk-forward rule/threshold selection;
* 2020-2024: untouched pre-OOS robustness audit (never used in selection);
* 2025: model frozen before the year;
* 2026: weekly expanding re-training with mature labels only.

This is a research tournament.  It does not alter the production SSOT and does
not authorize a signal.
"""
from __future__ import annotations

import hashlib
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
    Variant,
    candidate_score,
    evaluation_row,
    frozen_year_probabilities,
    metrics,
    promotion_gate,
    safe_json,
    selective_predictions,
    threshold_grid,
    walk_forward_probabilities,
)


SELECTION_YEARS = (2015, 2016, 2017, 2018, 2019)
ROBUSTNESS_YEARS = (2020, 2021, 2022, 2023, 2024)
HISTORY_START = pd.Timestamp("2000-01-01")
DEEP_PATH = ROOT / "data/backups/features/asset_daily_ohlcv.parquet"
CURRENT_PATH = ROOT / "seeds/latest/usdcop_daily_ohlcv.parquet"
FEATURES = [
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values("date").reset_index(drop=True).copy()
    for column in ("open", "high", "low", "close"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["return_1d"] = result["close"].pct_change(1)
    for lag in (5, 10, 20):
        result[f"return_{lag}d"] = result["close"].pct_change(lag)
    for window in (5, 10, 20):
        result[f"volatility_{window}d"] = result["return_1d"].rolling(window).std()
    delta = result["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    result["rsi_14d"] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0, np.nan))
    result["ma_ratio_20d"] = result["close"] / result["close"].rolling(20).mean()
    result["ma_ratio_50d"] = result["close"] / result["close"].rolling(50).mean()
    return result


def build_long_history_frame(
    deep_path: Path = DEEP_PATH,
    current_path: Path = CURRENT_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the price frame, optionally from immutable experiment snapshots."""
    deep = pd.read_parquet(deep_path)
    deep = deep[
        deep["symbol"].eq("USD/COP")
        & deep["source"].eq("twelvedata_daily_deep")
    ].copy()
    deep["date"] = pd.to_datetime(deep["time"], utc=True).dt.tz_localize(None).dt.normalize()
    current = pd.read_parquet(current_path).copy()
    current["date"] = pd.to_datetime(
        current["time"], utc=True,
    ).dt.tz_localize(None).dt.normalize()
    current_start = current["date"].min()
    deep = deep[deep["date"] < current_start]
    deep["research_source"] = "twelvedata_daily_deep"
    current["research_source"] = "current_daily_seed"
    columns = ["date", "open", "high", "low", "close", "research_source"]
    joined = pd.concat([deep[columns], current[columns]], ignore_index=True)
    joined = joined[joined["date"] >= HISTORY_START]
    joined = joined.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    if joined["date"].duplicated().any():
        raise ValueError("Long-history frame contains duplicate dates")
    if joined[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("Long-history frame contains null OHLC prices")
    if (joined[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Long-history frame contains non-positive OHLC prices")
    if not (
        joined["high"].ge(joined[["open", "close"]].max(axis=1)).all()
        and joined["low"].le(joined[["open", "close"]].min(axis=1)).all()
    ):
        raise ValueError("Long-history frame violates OHLC ordering")
    returns = np.log(joined["close"] / joined["close"].shift(1))
    provenance = {
        "rows": int(len(joined)),
        "start": joined["date"].min().date().isoformat(),
        "end": joined["date"].max().date().isoformat(),
        "source_rows": {
            str(key): int(value)
            for key, value in joined["research_source"].value_counts().items()
        },
        "duplicate_dates": int(joined["date"].duplicated().sum()),
        "maximum_absolute_daily_log_return": float(returns.abs().max()),
        "deep_sha256": sha256_file(deep_path),
        "current_sha256": sha256_file(current_path),
        "join_date": current_start.date().isoformat(),
    }
    return build_price_features(joined), provenance


def selection_candidate_is_eligible(annual: list[dict[str, Any]]) -> bool:
    return all(
        item["n_signals"] >= 12
        and item["coverage"] >= 0.25
        and item["balanced_accuracy"] is not None
        and item["balanced_accuracy"] >= 0.48
        and item["minimum_class_recall"] is not None
        and item["minimum_class_recall"] >= 0.20
        and item["prediction_up_rate"] is not None
        and 0.10 <= item["prediction_up_rate"] <= 0.90
        for item in annual
    )


def select_rule(
    validation_by_variant: dict[str, pd.DataFrame],
    variants: dict[str, Variant],
    horizon: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for key, predictions in validation_by_variant.items():
        for center, margin, lower, upper in threshold_grid():
            annual = [
                metrics(
                    predictions[predictions["origin_date"].dt.year.eq(year)],
                    lower,
                    upper,
                )
                for year in SELECTION_YEARS
            ]
            pooled = metrics(predictions, lower, upper)
            row = {
                "horizon_days": horizon,
                "variant_key": key,
                "half_life": (
                    "expanding" if variants[key].half_life is None
                    else variants[key].half_life
                ),
                "center": center,
                "margin": margin,
                "lower_threshold": lower,
                "upper_threshold": upper,
                "eligible": selection_candidate_is_eligible(annual),
                "selection_score": candidate_score(annual),
                "pooled_da": pooled["directional_accuracy"],
                "pooled_bda": pooled["balanced_accuracy"],
                "pooled_coverage": pooled["coverage"],
                "pooled_auc": pooled["auc_all"],
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
        ["selection_score", "minimum_year_bda", "pooled_bda", "pooled_coverage", "margin"],
        ascending=[False, False, False, False, True],
    ).iloc[0].to_dict()
    best["eligible"] = bool(best["eligible"])
    best["candidate_count"] = int(len(candidates))
    best["eligible_candidate_count"] = int(candidates["eligible"].sum())
    return best, candidates


def attach_decisions(
    predictions: pd.DataFrame, lower: float, upper: float,
) -> pd.DataFrame:
    result = predictions.copy()
    result["lower_threshold"] = lower
    result["upper_threshold"] = upper
    result["prediction"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    signals = selective_predictions(result, lower, upper)
    result.loc[signals.index, "prediction"] = signals["prediction"].astype(int)
    result["decision"] = result["prediction"].map({0: "DOWN", 1: "UP"}).fillna("FLAT")
    return result


def main() -> None:
    frame, provenance = build_long_history_frame()
    variants_list = [
        Variant("long_price", 0.1, "balanced", half_life)
        for half_life in (None, 252, 504, 1260)
    ]
    variants = {variant.key: variant for variant in variants_list}
    all_candidates: list[pd.DataFrame] = []
    selections: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    all_predictions: list[pd.DataFrame] = []

    for horizon in HORIZONS:
        print(f"[H{horizon}] long-history selection tournament", flush=True)
        validation_by_variant = {
            variant.key: walk_forward_probabilities(
                frame,
                FEATURES,
                horizon,
                variant,
                "2015-01-01",
                "2019-12-31",
            )
            for variant in variants_list
        }
        best, candidates = select_rule(validation_by_variant, variants, horizon)
        all_candidates.append(candidates)
        variant = variants[str(best["variant_key"])]
        lower = float(best["lower_threshold"])
        upper = float(best["upper_threshold"])
        best["feature_count"] = len(FEATURES)
        best["selected_features"] = "|".join(FEATURES)
        selections.append(best)

        blocks = (
            (
                "2015_2019_SELECTION",
                validation_by_variant[variant.key],
            ),
            (
                "2020_2024_UNTOUCHED_ROBUSTNESS",
                walk_forward_probabilities(
                    frame, FEATURES, horizon, variant, "2020-01-01", "2024-12-31",
                ),
            ),
            (
                "2025_FROZEN_OOS",
                frozen_year_probabilities(frame, FEATURES, horizon, variant, 2025),
            ),
            (
                "2026_EXPANDING_YTD",
                walk_forward_probabilities(
                    frame, FEATURES, horizon, variant, "2026-01-01", "2026-12-31",
                ),
            ),
        )
        for period, predictions in blocks:
            predictions = attach_decisions(predictions, lower, upper)
            predictions["period"] = period
            all_predictions.append(predictions)
            row = evaluation_row(predictions, horizon, period, lower, upper)
            passed, failed = promotion_gate(row)
            row["research_metric_gate_passed"] = passed
            row["failed_metric_gates"] = "|".join(failed)
            all_metrics.append(row)
            if period == "2020_2024_UNTOUCHED_ROBUSTNESS":
                for year in ROBUSTNESS_YEARS:
                    annual = predictions[
                        predictions["origin_date"].dt.year.eq(year)
                    ]
                    annual_period = f"{year}_UNTOUCHED_ROBUSTNESS"
                    annual_row = evaluation_row(
                        annual, horizon, annual_period, lower, upper,
                    )
                    annual_passed, annual_failed = promotion_gate(annual_row)
                    annual_row["research_metric_gate_passed"] = annual_passed
                    annual_row["failed_metric_gates"] = "|".join(annual_failed)
                    all_metrics.append(annual_row)
        print(
            f"[H{horizon}] selected hl={best['half_life']}, "
            f"thresholds={lower:.2f}/{upper:.2f}, eligible={best['eligible']}",
            flush=True,
        )

    report = ROOT / "reports"
    prefix = "usdcop_long_history_directional_tournament"
    candidate_frame = pd.concat(all_candidates, ignore_index=True)
    selection_frame = pd.DataFrame(selections)
    metric_frame = pd.DataFrame(all_metrics)
    prediction_frame = pd.concat(all_predictions, ignore_index=True)
    candidate_frame.to_csv(report / f"{prefix}_candidates.csv", index=False)
    selection_frame.to_csv(report / f"{prefix}_selections.csv", index=False)
    metric_frame.to_csv(report / f"{prefix}_metrics.csv", index=False)
    prediction_frame.to_csv(report / f"{prefix}_predictions.csv", index=False)

    generalization = []
    required_periods = (
        "2020_2024_UNTOUCHED_ROBUSTNESS", "2025_FROZEN_OOS", "2026_EXPANDING_YTD",
    )
    for horizon in HORIZONS:
        rows = metric_frame[
            metric_frame["horizon_days"].eq(horizon)
            & metric_frame["period"].isin(required_periods)
        ]
        generalization.append({
            "horizon_days": horizon,
            "passes_all_research_metric_gates": bool(
                len(rows) == len(required_periods)
                and rows["research_metric_gate_passed"].all()
            ),
            "institutional_promotion_authorized": False,
        })
    manifest = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "asset": "USD/COP",
        "signal_authorized": False,
        "history_regime": "post_exchange_band_float",
        "history_start": HISTORY_START.date().isoformat(),
        "protocol": {
            "initial_estimation": "2000-2014",
            "selection_years": list(SELECTION_YEARS),
            "untouched_robustness_years": list(ROBUSTNESS_YEARS),
            "frozen_oos_year": 2025,
            "expanding_weekly_year": 2026,
            "features": FEATURES,
            "model_family": "balanced_logistic_regression",
            "candidate_count_per_horizon": len(variants_list) * len(list(threshold_grid())),
        },
        "provenance": provenance,
        "generalization": generalization,
        "limitations": [
            "The 2020-2024 block is untouched by the algorithmic selection in this run, but the research team has broad prior knowledge of market history.",
            "2025 and 2026 have already been inspected in earlier research and are audit OOS, not fresh prospective holdouts.",
            "Price-only history cannot replace point-in-time macro releases; it tests sample-size stability only.",
        ],
    }
    (report / f"{prefix}_manifest.json").write_text(
        json.dumps(safe_json(manifest), indent=2), encoding="utf-8",
    )
    columns = [
        "period", "horizon_days", "n_signals", "coverage",
        "directional_accuracy", "balanced_accuracy", "up_recall", "down_recall",
        "lift_vs_causal_majority", "auc_all", "research_metric_gate_passed",
    ]
    print(metric_frame[columns].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
