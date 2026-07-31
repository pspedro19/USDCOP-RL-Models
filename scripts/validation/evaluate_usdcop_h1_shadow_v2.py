"""Evaluate the immutable USD/COP H1 shadow v2 evidence without promoting it.

The prediction and outcome ledgers are the source of truth.  This report is a
rebuildable view and is deliberately fail-closed: it can make an experiment
eligible for an independent review, but it never authorizes a signal or capital.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
from scipy import stats
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_directional_edge_tournament import (  # noqa: E402
    moving_block_bootstrap,
)
from scripts.pipeline.generate_usdcop_h1_regime_shadow_v2 import (  # noqa: E402
    load_jsonl,
    load_registered_contract,
    metric_summary,
    output_paths,
    validate_hash_chain,
)


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_regime_shadow_v2.yaml"
REPORT = ROOT / "reports/usdcop_h1_regime_shadow_v2/prospective_evaluation.json"
BOOTSTRAP_BLOCK_SIGNALS = 4
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 23
RISK_COVERAGE_THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def wilson_interval(successes: int, observations: int) -> dict[str, float | None]:
    if observations <= 0:
        return {"lower": None, "upper": None}
    z = 1.959963984540054
    p = successes / observations
    denominator = 1.0 + z * z / observations
    center = (p + z * z / (2.0 * observations)) / denominator
    margin = (
        z
        * math.sqrt(
            p * (1.0 - p) / observations + z * z / (4.0 * observations**2)
        )
        / denominator
    )
    return {"lower": center - margin, "upper": center + margin}


def pesaran_timmermann(actual: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """One-sided Pesaran-Timmermann directional-accuracy test.

    H0 is directional independence.  A result is undefined when either series
    is degenerate or the asymptotic variance is non-positive.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted lengths differ")
    n = len(actual)
    if n < 8:
        return {"n": n, "statistic": None, "one_sided_p": None, "valid": False}
    p_actual = float(actual.mean())
    p_predicted = float(predicted.mean())
    p_correct = float((actual == predicted).mean())
    p_independent = (
        p_actual * p_predicted + (1.0 - p_actual) * (1.0 - p_predicted)
    )
    variance_correct = p_independent * (1.0 - p_independent) / n
    variance_actual = p_actual * (1.0 - p_actual) / n
    variance_predicted = p_predicted * (1.0 - p_predicted) / n
    variance_independent = (
        (2.0 * p_actual - 1.0) ** 2 * variance_predicted
        + (2.0 * p_predicted - 1.0) ** 2 * variance_actual
        + 4.0 * variance_actual * variance_predicted
    )
    variance = variance_correct - variance_independent
    if variance <= 0.0 or not np.isfinite(variance):
        return {
            "n": n,
            "p_correct": p_correct,
            "p_under_independence": p_independent,
            "statistic": None,
            "one_sided_p": None,
            "valid": False,
        }
    statistic = (p_correct - p_independent) / math.sqrt(variance)
    return {
        "n": n,
        "p_correct": p_correct,
        "p_under_independence": p_independent,
        "statistic": statistic,
        "one_sided_p": float(stats.norm.sf(statistic)),
        "valid": True,
    }


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    actual = np.asarray(actual, dtype=int)
    predicted = np.asarray(predicted, dtype=int)
    up = actual == 1
    down = actual == 0
    if not up.any() or not down.any():
        return None
    return 0.5 * (float(predicted[up].mean()) + float((1 - predicted[down]).mean()))


def joined_records(
    predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    outcome_by_prediction = {
        str(item["prediction_record_sha256"]): item for item in outcomes
    }
    return [
        (prediction, outcome_by_prediction[str(prediction["prediction_record_sha256"])])
        for prediction in predictions
        if str(prediction["prediction_record_sha256"]) in outcome_by_prediction
    ]


def risk_coverage_curve(
    matured: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    total = len(matured)
    for threshold in RISK_COVERAGE_THRESHOLDS:
        selected = [
            pair
            for pair in matured
            if pair[0]["decision"] != "FLAT"
            and float(pair[0]["probability_confidence"]) >= threshold
        ]
        actual = np.asarray([pair[1]["actual"] for pair in selected], dtype=int)
        predicted = np.asarray(
            [pair[0]["decision"] == "UP" for pair in selected], dtype=int
        )
        hits = predicted == actual
        points.append({
            "minimum_probability_confidence": threshold,
            "signals": len(selected),
            "coverage_of_matured_weeks": len(selected) / total if total else None,
            "directional_accuracy": float(hits.mean()) if len(selected) else None,
            "balanced_accuracy": balanced_accuracy(actual, predicted)
            if len(selected) else None,
        })
    return points


def execution_summary(
    signals: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    if not signals:
        return {
            "gross_mean_return_per_signal": None,
            "gross_compound_return": None,
            "break_even_round_trip_cost_bps": None,
            "registered_cost_assumption_bps": None,
            "net_return_after_registered_costs": None,
            "positive_net_value_after_costs": False,
            "gate_reason": "No matured signals and no preregistered execution-cost schedule.",
        }
    gross = np.asarray([
        (1.0 if prediction["decision"] == "UP" else -1.0)
        * (float(outcome["actual_price"]) / float(prediction["base_price"]) - 1.0)
        for prediction, outcome in signals
    ])
    return {
        "gross_mean_return_per_signal": float(gross.mean()),
        "gross_compound_return": float(np.prod(1.0 + gross) - 1.0),
        "break_even_round_trip_cost_bps": max(0.0, float(gross.mean()) * 10_000.0),
        "registered_cost_assumption_bps": None,
        "net_return_after_registered_costs": None,
        "positive_net_value_after_costs": False,
        "gate_reason": (
            "Direction-only shadow has no preregistered executable fill and cost "
            "schedule; capital review remains closed."
        ),
    }


def evaluate() -> dict[str, Any]:
    contract, registration, contract_hash, code_bundle_hash = load_registered_contract()
    paths = output_paths(contract)
    predictions = load_jsonl(paths["predictions"])
    outcomes = load_jsonl(paths["outcomes"])
    prediction_head = validate_hash_chain(
        predictions,
        hash_field="prediction_record_sha256",
        previous_field="previous_prediction_sha256",
    )
    outcome_head = validate_hash_chain(
        outcomes,
        hash_field="outcome_record_sha256",
        previous_field="previous_outcome_sha256",
    )
    matured = joined_records(predictions, outcomes)
    signals = [pair for pair in matured if pair[0]["decision"] != "FLAT"]
    summary = metric_summary(predictions, outcomes)
    actual = np.asarray([pair[1]["actual"] for pair in signals], dtype=int)
    predicted = np.asarray(
        [pair[0]["decision"] == "UP" for pair in signals], dtype=int
    )
    majority = np.asarray(
        [pair[0]["causal_majority_direction"] == "UP" for pair in signals],
        dtype=int,
    )
    hits = predicted == actual
    majority_hits = majority == actual
    lift = hits.astype(float) - majority_hits.astype(float)
    bootstrap = moving_block_bootstrap(
        lift,
        block_length=BOOTSTRAP_BLOCK_SIGNALS,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    pt = pesaran_timmermann(actual, predicted)
    binomial = (
        {
            "successes": int(hits.sum()),
            "n": len(hits),
            "one_sided_p_vs_50pct": float(
                stats.binomtest(int(hits.sum()), len(hits), 0.5, alternative="greater").pvalue
            ),
            "wilson_95": wilson_interval(int(hits.sum()), len(hits)),
        }
        if len(hits)
        else {
            "successes": 0,
            "n": 0,
            "one_sided_p_vs_50pct": None,
            "wilson_95": wilson_interval(0, 0),
        }
    )
    protocol = contract["prospective_protocol"]
    thresholds = contract["promotion_gates"]
    sample_sufficient = bool(
        len(predictions) >= int(protocol["minimum_calendar_weeks_before_promotion_review"])
        and len(signals) >= int(protocol["minimum_matured_signals_before_promotion_review"])
    )

    def at_least(value: float | None, minimum: float) -> bool:
        return bool(value is not None and value >= minimum)

    def at_most(value: float | None, maximum: float) -> bool:
        return bool(value is not None and value <= maximum)

    class_rate = summary["prediction_up_rate"]
    statistical_gates = {
        "minimum_directional_accuracy": at_least(
            summary["directional_accuracy"], float(thresholds["minimum_directional_accuracy"])
        ),
        "minimum_balanced_accuracy": at_least(
            summary["balanced_accuracy"], float(thresholds["minimum_balanced_accuracy"])
        ),
        "minimum_up_recall": at_least(
            summary["up_recall"], float(thresholds["minimum_up_recall"])
        ),
        "minimum_down_recall": at_least(
            summary["down_recall"], float(thresholds["minimum_down_recall"])
        ),
        "positive_lift_vs_causal_majority": at_least(
            summary["lift_vs_causal_majority"],
            float(thresholds["minimum_lift_vs_causal_majority"]),
        ),
        "lift_block_bootstrap_one_sided_p": at_most(
            bootstrap["one_sided_p"],
            float(thresholds["lift_block_bootstrap_one_sided_p_max"]),
        ),
        "lift_block_bootstrap_ci95_lower": at_least(
            bootstrap["ci_low"],
            float(thresholds["lift_block_bootstrap_ci95_lower_min"]),
        ),
        "pesaran_timmermann_one_sided_p": at_most(
            pt["one_sided_p"], float(thresholds["pesaran_timmermann_p_max"])
        ),
        "prediction_class_balance": bool(
            class_rate is not None
            and float(thresholds["minimum_prediction_class_rate"])
            <= class_rate
            <= float(thresholds["maximum_prediction_class_rate"])
        ),
        "maximum_brier_score": at_most(
            summary["brier_all_committed_predictions"],
            float(thresholds["maximum_brier_score"]),
        ),
    }
    execution = execution_summary(signals)
    review_gates = {
        "minimum_sample": sample_sufficient,
        "all_statistical_gates": all(statistical_gates.values()),
        "single_preregistered_primary_rule_no_prospective_selection": True,
        "registration_and_hash_chains_valid": True,
        "positive_net_value_after_costs": execution["positive_net_value_after_costs"],
    }
    review_eligible = bool(all(review_gates.values()))
    if not sample_sufficient:
        status = "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    elif not all(statistical_gates.values()):
        status = "STATISTICAL_PROMOTION_GATES_FAILED"
    elif not execution["positive_net_value_after_costs"]:
        status = "EXECUTION_AND_COST_REVIEW_REQUIRED"
    else:
        status = "ELIGIBLE_FOR_INDEPENDENT_PROMOTION_REVIEW"
    return {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "interpretation": (
            "Derived prospective audit only. It cannot authorize signals, capital, "
            "or an automatic model promotion."
        ),
        "signal_authorized": False,
        "capital_authorized": False,
        "automatic_promotion": False,
        "promotion_review_eligible": review_eligible,
        "integrity": {
            "registration_valid": registration["registration_valid"],
            "contract_sha256": contract_hash,
            "code_bundle_sha256": code_bundle_hash,
            "prediction_chain_head": prediction_head,
            "outcome_chain_head": outcome_head,
            "python": sys.version.split()[0],
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "sample": {
            "committed_weeks": len(predictions),
            "matured_weeks": len(matured),
            "matured_signals": len(signals),
            "minimum_committed_weeks": int(
                protocol["minimum_calendar_weeks_before_promotion_review"]
            ),
            "minimum_matured_signals": int(
                protocol["minimum_matured_signals_before_promotion_review"]
            ),
            "sufficient": sample_sufficient,
        },
        "metrics": summary,
        "uncertainty": {
            "directional_accuracy_binomial": binomial,
            "pesaran_timmermann": pt,
            "paired_lift_block_bootstrap": {
                **bootstrap,
                "block_length_signals": BOOTSTRAP_BLOCK_SIGNALS,
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
                "estimand": "model_hit_minus_causal_majority_hit",
            },
        },
        "risk_coverage": risk_coverage_curve(matured),
        "execution": execution,
        "statistical_gates": statistical_gates,
        "review_gates": review_gates,
        "methodology": {
            "primary_unit": "prospectively committed weekly signal",
            "baseline": "causal expanding-window majority direction at each origin",
            "multiplicity": (
                "One frozen primary rule. Prospective data cannot select a replacement; "
                "any model, feature, threshold, gate, or data change starts a new experiment."
            ),
            "interim_rule": (
                "Interim reports are descriptive and can never authorize capital."
            ),
        },
    }


def main() -> int:
    report = evaluate()
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    temporary = REPORT.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(REPORT)
    print(json.dumps({
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "committed_weeks": report["sample"]["committed_weeks"],
        "matured_signals": report["sample"]["matured_signals"],
        "promotion_review_eligible": report["promotion_review_eligible"],
        "capital_authorized": report["capital_authorized"],
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
