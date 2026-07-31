"""Fail-closed evaluation of the prospective daily H1 shadow ledgers."""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_directional_edge_tournament import (  # noqa: E402
    moving_block_bootstrap,
)
from scripts.pipeline.generate_usdcop_h1_daily_shadow_v1 import (  # noqa: E402
    json_safe,
    load_jsonl,
    load_registered_contract,
    metric_summary,
    output_paths,
    validate_daily_hash_chain,
)
from scripts.validation.evaluate_usdcop_h1_shadow_v2 import (  # noqa: E402
    pesaran_timmermann,
    wilson_interval,
)


REPORT = ROOT / "reports/usdcop_h1_daily_shadow_v1/prospective_evaluation.json"
BLOCK_LENGTH = 5
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 31
ALPHA = 0.05
BETTING_FRACTION_CAP = 1.0
RISK_COVERAGE_THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)


def joined_records(
    predictions: list[dict[str, Any]], outcomes: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    by_prediction = {str(item["prediction_record_sha256"]): item for item in outcomes}
    return [
        (prediction, by_prediction[str(prediction["prediction_record_sha256"])])
        for prediction in predictions
        if str(prediction["prediction_record_sha256"]) in by_prediction
    ]


def predictable_betting_log_e(hits: np.ndarray, null_mean: float) -> float:
    """Anytime-valid e-process for bounded Bernoulli hits under mean <= null.

    The bet at time t uses only earlier hits.  For lambda in [0, 1/p0],
    1 + lambda * (X_t - p0) is nonnegative and has conditional expectation
    <= 1 under the null.  We conservatively cap lambda at one.
    """
    if not 0.0 < null_mean < 1.0:
        raise ValueError("null_mean must be in (0, 1)")
    successes = 0.0
    log_e = 0.0
    for prior_count, value in enumerate(np.asarray(hits, dtype=float)):
        estimate = (successes + 0.5) / (prior_count + 1.0)
        raw_bet = (estimate - null_mean) / (null_mean * (1.0 - null_mean))
        bet = min(max(raw_bet, 0.0), BETTING_FRACTION_CAP, 0.999 / null_mean)
        factor = 1.0 + bet * (value - null_mean)
        if factor <= 0.0:
            raise RuntimeError("Non-positive e-process factor")
        log_e += math.log(factor)
        successes += value
    return log_e


def anytime_directional_evidence(hits: np.ndarray) -> dict[str, Any]:
    hits = np.asarray(hits, dtype=float)
    if len(hits) == 0:
        return {
            "n": 0,
            "e_value_at_0_50": 1.0,
            "anytime_p_at_0_50": 1.0,
            "lower_confidence_bound_95": None,
            "method": "predictable plug-in betting e-process",
        }
    log_e_50 = predictable_betting_log_e(hits, 0.50)
    threshold = math.log(1.0 / ALPHA)
    rejected: list[float] = []
    for null in np.linspace(0.01, 0.99, 981):
        if predictable_betting_log_e(hits, float(null)) >= threshold:
            rejected.append(float(null))
    lower = max(rejected) if rejected else 0.0
    e_value = float(math.exp(min(log_e_50, 700.0)))
    return {
        "n": len(hits),
        "e_value_at_0_50": e_value,
        "anytime_p_at_0_50": min(1.0, 1.0 / e_value),
        "lower_confidence_bound_95": lower,
        "alpha": ALPHA,
        "betting_fraction_cap": BETTING_FRACTION_CAP,
        "method": "predictable plug-in betting e-process inverted on a 0.001 grid",
        "monitoring_validity": "time-uniform under conditional mean null for bounded hits",
    }


def risk_coverage_curve(
    matured: list[tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for threshold in RISK_COVERAGE_THRESHOLDS:
        selected = [
            pair for pair in matured
            if pair[0]["decision"] != "FLAT"
            and float(pair[0]["probability_confidence"]) >= threshold
        ]
        actual = np.asarray([outcome["actual"] for _, outcome in selected], dtype=int)
        predicted = np.asarray(
            [prediction["decision"] == "UP" for prediction, _ in selected], dtype=int
        )
        up = actual == 1
        down = actual == 0
        up_recall = float(predicted[up].mean()) if up.any() else None
        down_recall = float((1 - predicted[down]).mean()) if down.any() else None
        points.append({
            "minimum_probability_confidence": threshold,
            "signals": len(selected),
            "coverage_of_matured_origins": len(selected) / len(matured) if matured else None,
            "directional_accuracy": float((predicted == actual).mean()) if len(selected) else None,
            "balanced_accuracy": 0.5 * (up_recall + down_recall)
            if up_recall is not None and down_recall is not None else None,
        })
    return points


def execution_summary(
    selective: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    if not selective:
        gross = np.asarray([], dtype=float)
    else:
        gross = np.asarray([
            (1.0 if prediction["decision"] == "UP" else -1.0)
            * float(outcome["actual_return"])
            for prediction, outcome in selective
        ])
    return {
        "signals": len(gross),
        "gross_mean_return_per_signal": float(gross.mean()) if len(gross) else None,
        "gross_compound_return": float(np.prod(1.0 + gross) - 1.0) if len(gross) else None,
        "break_even_round_trip_cost_bps": max(0.0, float(gross.mean()) * 10_000.0)
        if len(gross) else None,
        "registered_executable_quote_and_cost_schedule": False,
        "net_value_after_costs": None,
        "positive_net_value_after_costs": False,
        "gate_reason": (
            "Daily close direction is not an executable fill. Timestamped bid/ask, "
            "slippage, fees and financing remain required."
        ),
    }


def evaluate() -> dict[str, Any]:
    contract, registration, contract_hash, code_bundle_hash = load_registered_contract()
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
    matured = joined_records(predictions, outcomes)
    selective = [pair for pair in matured if pair[0]["decision"] != "FLAT"]
    summary = metric_summary(predictions, outcomes)
    actual = np.asarray([outcome["actual"] for _, outcome in matured], dtype=int)
    base_predicted = np.asarray(
        [prediction["base_direction"] == "UP" for prediction, _ in matured], dtype=int
    )
    base_hits = base_predicted == actual
    majority_hits = np.asarray(
        [outcome["causal_majority_hit"] for _, outcome in matured], dtype=bool
    )
    lift = base_hits.astype(float) - majority_hits.astype(float)
    lift_bootstrap = moving_block_bootstrap(
        lift, block_length=BLOCK_LENGTH, samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )
    probabilities = np.asarray(
        [prediction["probability_up"] for prediction, _ in matured], dtype=float
    )
    priors = np.asarray([prediction["train_up_rate"] for prediction, _ in matured], dtype=float)
    brier_improvement = (priors - actual) ** 2 - (probabilities - actual) ** 2
    brier_bootstrap = moving_block_bootstrap(
        brier_improvement,
        block_length=BLOCK_LENGTH,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED + 1,
    )
    pt_base = pesaran_timmermann(actual, base_predicted)
    anytime = anytime_directional_evidence(base_hits)
    binomial = {
        "successes": int(base_hits.sum()),
        "n": len(base_hits),
        "one_sided_p_vs_50pct": float(
            stats.binomtest(int(base_hits.sum()), len(base_hits), 0.5, alternative="greater").pvalue
        ) if len(base_hits) else None,
        "wilson_95": wilson_interval(int(base_hits.sum()), len(base_hits)),
        "fixed_sample_only": True,
    }
    selective_actual = np.asarray([outcome["actual"] for _, outcome in selective], dtype=int)
    selective_predicted = np.asarray(
        [prediction["decision"] == "UP" for prediction, _ in selective], dtype=int
    )
    pt_selective = pesaran_timmermann(selective_actual, selective_predicted)
    execution = execution_summary(selective)

    protocol = contract["prospective_protocol"]
    thresholds = contract["promotion_gates"]
    elapsed_days = 0
    if predictions:
        elapsed_days = (
            datetime.now(timezone.utc).date()
            - pd.Timestamp(predictions[0]["origin_date"]).date()
        ).days
    sample_gates = {
        "minimum_elapsed_calendar_months": elapsed_days >= 365,
        "minimum_committed_sessions": len(predictions) >= int(
            protocol["minimum_committed_sessions_before_review"]
        ),
        "minimum_matured_base_predictions": len(matured) >= int(
            protocol["minimum_matured_base_predictions_before_review"]
        ),
        "minimum_matured_selective_signals": len(selective) >= int(
            protocol["minimum_matured_selective_signals_before_review"]
        ),
    }

    def at_least(value: float | None, threshold: float) -> bool:
        return bool(value is not None and value >= threshold)

    def at_most(value: float | None, threshold: float) -> bool:
        return bool(value is not None and value <= threshold)

    base = summary["base"]
    selective_metrics = summary["selective"]
    class_rate = base["prediction_up_rate"]
    statistical_gates = {
        "minimum_base_directional_accuracy": at_least(
            base["directional_accuracy"], float(thresholds["minimum_base_directional_accuracy"])
        ),
        "minimum_base_balanced_accuracy": at_least(
            base["balanced_accuracy"], float(thresholds["minimum_base_balanced_accuracy"])
        ),
        "minimum_selective_directional_accuracy": at_least(
            selective_metrics["directional_accuracy"],
            float(thresholds["minimum_selective_directional_accuracy"]),
        ),
        "minimum_selective_balanced_accuracy": at_least(
            selective_metrics["balanced_accuracy"],
            float(thresholds["minimum_selective_balanced_accuracy"]),
        ),
        "minimum_base_up_recall": at_least(base["up_recall"], float(thresholds["minimum_up_recall"])),
        "minimum_base_down_recall": at_least(base["down_recall"], float(thresholds["minimum_down_recall"])),
        "positive_paired_lift": at_least(
            summary["base_lift_vs_causal_majority"], float(thresholds["minimum_lift_vs_causal_majority"])
        ),
        "lift_block_bootstrap_p": at_most(
            lift_bootstrap["one_sided_p"], float(thresholds["lift_block_bootstrap_one_sided_p_max"])
        ),
        "lift_block_bootstrap_ci_lower_strict": bool(
            lift_bootstrap["ci_low"] is not None
            and lift_bootstrap["ci_low"]
            > float(thresholds["lift_block_bootstrap_ci95_lower_min_exclusive"])
        ),
        "pesaran_timmermann_p": at_most(
            pt_base["one_sided_p"], float(thresholds["pesaran_timmermann_p_max"])
        ),
        "prediction_class_balance": bool(
            class_rate is not None
            and float(thresholds["minimum_prediction_class_rate"])
            <= class_rate
            <= float(thresholds["maximum_prediction_class_rate"])
        ),
        "brier_better_than_causal_prior_with_ci": bool(
            brier_bootstrap["ci_low"] is not None and brier_bootstrap["ci_low"] > 0.0
        ),
        "time_uniform_lower_da_bound": bool(
            anytime["lower_confidence_bound_95"] is not None
            and anytime["lower_confidence_bound_95"]
            > float(thresholds["require_time_uniform_lower_da_bound_gt"])
        ),
    }
    review_gates = {
        "all_sample_gates": all(sample_gates.values()),
        "all_statistical_gates": all(statistical_gates.values()),
        "registration_and_hash_chains_valid": True,
        "no_prospective_model_selection": True,
        "positive_net_value_after_costs": execution["positive_net_value_after_costs"],
        "independent_validation_signoff": False,
    }
    review_eligible = all(review_gates.values())
    if not all(sample_gates.values()):
        status = "INSUFFICIENT_PROSPECTIVE_EVIDENCE"
    elif not all(statistical_gates.values()):
        status = "STATISTICAL_PROMOTION_GATES_FAILED"
    elif not execution["positive_net_value_after_costs"]:
        status = "EXECUTION_AND_COST_REVIEW_REQUIRED"
    else:
        status = "INDEPENDENT_VALIDATION_REQUIRED"
    return {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
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
        },
        "sample": {
            "elapsed_days": elapsed_days,
            "committed_sessions": len(predictions),
            "matured_base_predictions": len(matured),
            "matured_selective_signals": len(selective),
            "gates": sample_gates,
        },
        "metrics": summary,
        "uncertainty": {
            "base_direction_binomial": binomial,
            "base_direction_pesaran_timmermann": pt_base,
            "selective_direction_pesaran_timmermann": pt_selective,
            "base_direction_anytime_evidence": anytime,
            "paired_lift_block_bootstrap": {
                **lift_bootstrap,
                "block_length_sessions": BLOCK_LENGTH,
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED,
            },
            "brier_improvement_block_bootstrap": {
                **brier_bootstrap,
                "estimand": "causal_prior_squared_error_minus_model_squared_error",
                "block_length_sessions": BLOCK_LENGTH,
                "samples": BOOTSTRAP_SAMPLES,
                "seed": BOOTSTRAP_SEED + 1,
            },
        },
        "risk_coverage": risk_coverage_curve(matured),
        "execution": execution,
        "statistical_gates": statistical_gates,
        "review_gates": review_gates,
        "interpretation": (
            "Prospective evidence monitor only. Historical daily outcomes cannot be "
            "inserted, and no report can authorize signals or capital."
        ),
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
        "committed_sessions": report["sample"]["committed_sessions"],
        "matured_base_predictions": report["sample"]["matured_base_predictions"],
        "promotion_review_eligible": report["promotion_review_eligible"],
        "capital_authorized": report["capital_authorized"],
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
