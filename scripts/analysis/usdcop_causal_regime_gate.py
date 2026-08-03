"""Causal trend/volatility gate for long-history USD/COP predictions.

The base model and its threshold were selected in 2015-2019 by the long-history
tournament.  This second layer admits only fixed, observable trend/volatility
states whose selection-period predictions had balanced class behavior.  The
gate is then audited on 2020-2024, 2025 and 2026 without further tuning.
"""
from __future__ import annotations

import json
import math
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
    promotion_gate,
    safe_json,
)
from scripts.analysis.usdcop_long_history_directional_tournament import (
    build_long_history_frame,
)


INPUT = ROOT / "reports/usdcop_long_history_directional_tournament_predictions.csv"
PREFIX = "usdcop_causal_regime_gate"
TREND_Z_THRESHOLD = 0.50
MIN_STATE_OBSERVATIONS = 20


def build_states(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["date", "return_20d", "volatility_20d"]].copy()
    denominator = result["volatility_20d"] * math.sqrt(20)
    result["trend_z"] = result["return_20d"] / denominator.replace(0, np.nan)
    result["trend_state"] = np.select(
        [
            result["trend_z"] > TREND_Z_THRESHOLD,
            result["trend_z"] < -TREND_Z_THRESHOLD,
        ],
        ["trend_up", "trend_down"],
        default="range",
    )
    # The reference uses only information strictly before the origin session.
    result["volatility_reference"] = (
        result["volatility_20d"].rolling(252, min_periods=120).median().shift(1)
    )
    result["volatility_state"] = np.where(
        result["volatility_20d"] > result["volatility_reference"],
        "high_vol",
        "low_vol",
    )
    result["regime"] = result["trend_state"] + "__" + result["volatility_state"]
    return result


def selected_metrics(frame: pd.DataFrame, total: int | None = None) -> dict[str, Any]:
    matured = frame[frame["actual"].notna() & frame["prediction"].notna()].copy()
    denominator = total if total is not None else len(matured)
    if matured.empty:
        return {
            "n_total": int(denominator), "n_signals": 0, "coverage": 0.0,
            "directional_accuracy": None, "balanced_accuracy": None,
            "up_recall": None, "down_recall": None, "minimum_class_recall": None,
            "prediction_up_rate": None, "actual_up_rate": None,
            "causal_majority_accuracy": None, "lift_vs_causal_majority": None,
        }
    actual = matured["actual"].astype(int)
    prediction = matured["prediction"].astype(int)
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
    baseline = (matured["train_up_rate"] >= 0.5).astype(int)
    da = float((prediction == actual).mean())
    baseline_da = float((baseline == actual).mean())
    return {
        "n_total": int(denominator),
        "n_signals": int(len(matured)),
        "coverage": float(len(matured) / denominator) if denominator else 0.0,
        "directional_accuracy": da,
        "balanced_accuracy": balanced,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "minimum_class_recall": minimum,
        "prediction_up_rate": float(prediction.mean()),
        "actual_up_rate": float(actual.mean()),
        "causal_majority_accuracy": baseline_da,
        "lift_vs_causal_majority": da - baseline_da,
    }


def state_is_eligible(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics["n_signals"] >= MIN_STATE_OBSERVATIONS
        and metrics["balanced_accuracy"] is not None
        and metrics["balanced_accuracy"] >= 0.55
        and metrics["directional_accuracy"] is not None
        and metrics["directional_accuracy"] >= 0.52
        and metrics["minimum_class_recall"] is not None
        and metrics["minimum_class_recall"] >= 0.25
        and metrics["prediction_up_rate"] is not None
        and 0.10 <= metrics["prediction_up_rate"] <= 0.90
    )


def main() -> None:
    price, provenance = build_long_history_frame()
    states = build_states(price)
    predictions = pd.read_csv(
        INPUT,
        parse_dates=["origin_date", "target_date", "train_label_end"],
    )
    predictions = predictions.merge(
        states.rename(columns={"date": "origin_date"}),
        on="origin_date",
        how="left",
        validate="many_to_one",
    )
    if predictions["regime"].isna().any():
        raise ValueError("Regime merge left prediction rows without a causal state")

    state_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    gated_frames: list[pd.DataFrame] = []
    core_periods = (
        "2015_2019_SELECTION",
        "2020_2024_UNTOUCHED_ROBUSTNESS",
        "2025_FROZEN_OOS",
        "2026_EXPANDING_YTD",
    )

    for horizon in HORIZONS:
        horizon_frame = predictions[predictions["horizon_days"].eq(horizon)].copy()
        selection = horizon_frame[
            horizon_frame["period"].eq("2015_2019_SELECTION")
        ]
        allowed: list[str] = []
        for regime, state_frame in selection.groupby("regime"):
            state_metric = selected_metrics(state_frame)
            eligible = state_is_eligible(state_metric)
            state_rows.append({
                "horizon_days": horizon,
                "regime": regime,
                "eligible": eligible,
                **state_metric,
            })
            if eligible:
                allowed.append(str(regime))
        selection_rows.append({
            "horizon_days": horizon,
            "allowed_regimes": "|".join(sorted(allowed)),
            "allowed_regime_count": len(allowed),
            "candidate_regime_count": int(selection["regime"].nunique()),
            "selection_observations": int(len(selection)),
        })

        for period in core_periods:
            part = horizon_frame[horizon_frame["period"].eq(period)].copy()
            total = int(part["actual"].notna().sum())
            gated = part[part["regime"].isin(allowed)].copy()
            gated["regime_gate_passed"] = True
            gated["gated_period"] = period
            gated_frames.append(gated)
            row = {
                "period": period,
                "horizon_days": horizon,
                "allowed_regime_count": len(allowed),
                **selected_metrics(gated, total=total),
            }
            passed, failed = promotion_gate(row)
            row["research_metric_gate_passed"] = passed
            row["failed_metric_gates"] = "|".join(failed)
            metric_rows.append(row)
            if period == "2020_2024_UNTOUCHED_ROBUSTNESS":
                for year in range(2020, 2025):
                    annual = part[part["origin_date"].dt.year.eq(year)]
                    annual_gated = annual[annual["regime"].isin(allowed)]
                    annual_row = {
                        "period": f"{year}_UNTOUCHED_ROBUSTNESS",
                        "horizon_days": horizon,
                        "allowed_regime_count": len(allowed),
                        **selected_metrics(
                            annual_gated,
                            total=int(annual["actual"].notna().sum()),
                        ),
                    }
                    annual_passed, annual_failed = promotion_gate(annual_row)
                    annual_row["research_metric_gate_passed"] = annual_passed
                    annual_row["failed_metric_gates"] = "|".join(annual_failed)
                    metric_rows.append(annual_row)

    report = ROOT / "reports"
    state_frame = pd.DataFrame(state_rows)
    selection_frame = pd.DataFrame(selection_rows)
    metric_frame = pd.DataFrame(metric_rows)
    gated_frame = (
        pd.concat(gated_frames, ignore_index=True)
        if gated_frames else pd.DataFrame()
    )
    state_frame.to_csv(report / f"{PREFIX}_state_selection.csv", index=False)
    selection_frame.to_csv(report / f"{PREFIX}_selections.csv", index=False)
    metric_frame.to_csv(report / f"{PREFIX}_metrics.csv", index=False)
    gated_frame.to_csv(report / f"{PREFIX}_predictions.csv", index=False)

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
        "base_predictions": str(INPUT.relative_to(ROOT)),
        "regime_definition": {
            "trend_z_threshold": TREND_Z_THRESHOLD,
            "trend_z": "return_20d / (volatility_20d * sqrt(20))",
            "volatility_reference": "lagged rolling 252-session median",
            "states": "trend_up|range|trend_down x high_vol|low_vol",
        },
        "state_selection_gate": {
            "minimum_observations": MIN_STATE_OBSERVATIONS,
            "minimum_da": 0.52,
            "minimum_bda": 0.55,
            "minimum_class_recall": 0.25,
        },
        "provenance": provenance,
        "generalization": generalization,
        "limitations": [
            "The regime gate is a second selection layer on 2015-2019; 2020-2024 is therefore the first clean audit for the combined rule.",
            "A regime with no robust selection evidence is FLAT; no fallback direction is forced.",
            "2025/2026 remain audit OOS rather than fresh prospective holdouts.",
        ],
    }
    (report / f"{PREFIX}_manifest.json").write_text(
        json.dumps(safe_json(manifest), indent=2), encoding="utf-8",
    )
    columns = [
        "period", "horizon_days", "allowed_regime_count", "n_signals", "coverage",
        "directional_accuracy", "balanced_accuracy", "up_recall", "down_recall",
        "lift_vs_causal_majority", "research_metric_gate_passed",
    ]
    print(selection_frame.to_string(index=False))
    print(metric_frame[metric_frame["period"].isin(core_periods)][columns].to_string(index=False))


if __name__ == "__main__":
    main()
