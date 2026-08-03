"""Preregistered paired audit of intraday MXN/BRL features for USD/COP.

This is a research-only replay over a frozen reconstructed hourly snapshot.
It compares the same weekly origins, mature labels and estimator with and
without exactly six predecision LatAm peer features.  Historical results may
nominate a prospective shadow candidate but can never authorize capital.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_forward_flow_directional_audit import (
    eligible_train_indices,
    exact_mcnemar,
    fit_model,
    forecast_record,
    holm_adjust,
    make_target,
    moving_block_bootstrap,
    safe_json,
    sha256_file,
    summarize,
    weekly_origin_indices,
    write_workbook,
)
from scripts.analysis.usdcop_long_history_directional_tournament import (
    build_long_history_frame,
)
from src.data.usdcop_intraday_peer_features import (
    attach_intraday_peer_features,
)


DEFAULT_CONFIG = (
    ROOT / "config" / "forecast_experiments"
    / "usdcop_intraday_latam_lead_v1.yaml"
)
TREATMENT = "price_plus_intraday_latam"


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
    checks = {
        "config": config_path,
        "deep_price": ROOT / config["data"]["price_sources"]["deep_path"],
        "current_price": ROOT / config["data"]["price_sources"]["current_path"],
        "peer_hourly": ROOT / config["data"]["peer_hourly_path"],
        "feature_builder": ROOT / "src/data/usdcop_intraday_peer_features.py",
        "audit_script": Path(__file__),
    }
    expected = {
        "config": registration["config_sha256"],
        **registration["data_sha256"],
        **registration["code_sha256"],
    }
    for key, path in checks.items():
        actual_hash = sha256_file(path)
        if actual_hash != expected[key]:
            raise ValueError(
                f"Preregistered {key} drift: expected {expected[key]}, "
                f"got {actual_hash}"
            )
    return config, registration, registration_path


def build_research_frame(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    deep = ROOT / config["data"]["price_sources"]["deep_path"]
    current = ROOT / config["data"]["price_sources"]["current_path"]
    frame, price_provenance = build_long_history_frame(deep, current)
    frame = frame[
        pd.to_datetime(frame["date"]).le(
            pd.Timestamp(config["data"]["evaluation_asof"])
        )
    ].copy()
    preopen_hour, preopen_minute = map(
        int, str(config["data"]["preopen_time"]).split(":")
    )
    decision_hour, decision_minute = map(
        int, str(config["data"]["decision_time"]).split(":")
    )
    frame, attached, peer_provenance = attach_intraday_peer_features(
        frame,
        ROOT / config["data"]["peer_hourly_path"],
        preopen_hour_bogota=preopen_hour,
        preopen_minute_bogota=preopen_minute,
        decision_hour_bogota=decision_hour,
        decision_minute_bogota=decision_minute,
        maximum_snapshot_staleness_hours=int(
            config["data"]["maximum_snapshot_staleness_hours"]
        ),
    )
    registered = list(config["features"]["intraday_latam"])
    if attached != registered:
        raise ValueError(
            f"Feature contract drift: attached={attached}, registered={registered}"
        )
    required = list(config["features"]["price"]) + registered
    missing = [feature for feature in required if feature not in frame.columns]
    if missing:
        raise ValueError(f"Registered features missing: {missing}")
    frame = frame.sort_values("date").reset_index(drop=True)
    coverage: list[dict[str, Any]] = []
    for feature in required:
        valid = frame.loc[frame[feature].notna(), "date"]
        coverage.append({
            "feature": feature,
            "family": (
                "price" if feature in config["features"]["price"]
                else "intraday_latam"
            ),
            "non_null_rows": int(len(valid)),
            "coverage": float(len(valid) / len(frame)),
            "first_available": valid.min() if len(valid) else None,
            "last_available": valid.max() if len(valid) else None,
        })
    provenance = {
        "price": price_provenance,
        "intraday_peers": peer_provenance,
        "research_frame_rows": int(len(frame)),
        "research_frame_end": pd.Timestamp(frame["date"].max()),
        "historical_peer_availability_is_reconstructed": True,
        "promotion_eligible": False,
    }
    return frame, provenance, coverage


def run_replay(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    price_features = list(config["features"]["price"])
    peer_features = list(config["features"]["intraday_latam"])
    feature_sets = {
        "price_only": price_features,
        TREATMENT: price_features + peer_features,
    }
    horizons = [int(value) for value in config["protocol"]["horizons_trading_days"]]
    threshold = float(config["protocol"]["threshold"])
    train_start = str(config["data"]["price_start"])
    dates = pd.to_datetime(frame["date"])
    audit_columns = [column for column in frame if column.startswith("audit_")]
    rows: list[dict[str, Any]] = []
    for horizon in horizons:
        target, future_return = make_target(frame, horizon)
        print(f"[H{horizon}] paired weekly replay", flush=True)
        for period_name, period in config["protocol"]["periods"].items():
            origins = weekly_origin_indices(
                frame,
                str(period["start"]),
                str(period["end"]),
                asof=str(config["data"]["evaluation_asof"]),
                exclude_incomplete_current_week=bool(
                    config["data"]["exclude_incomplete_current_iso_week"]
                ),
            )
            mode = str(period["training_mode"])
            role = str(period["role"])
            frozen: dict[str, tuple[Any, np.ndarray]] = {}
            if mode.startswith("frozen_pre_"):
                year = int(mode.rsplit("_", 1)[-1])
                before = np.flatnonzero(dates.lt(pd.Timestamp(f"{year}-01-01")))
                train = eligible_train_indices(
                    frame, target, horizon, int(before[-1]), train_start
                )
                frozen = {
                    name: (fit_model(frame, target, features, train, config), train)
                    for name, features in feature_sets.items()
                }
            for origin_index in origins:
                if frozen:
                    models = frozen
                else:
                    train = eligible_train_indices(
                        frame, target, horizon, origin_index, train_start
                    )
                    models = {
                        name: (
                            fit_model(frame, target, features, train, config), train
                        )
                        for name, features in feature_sets.items()
                    }
                for model_name, features in feature_sets.items():
                    model, train = models[model_name]
                    record = forecast_record(
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
                        training_mode=mode,
                        threshold=threshold,
                    )
                    record["origin_close"] = float(frame.loc[origin_index, "close"])
                    target_index = origin_index + horizon
                    record["actual_target_close"] = (
                        float(frame.loc[target_index, "close"])
                        if target_index < len(frame) else None
                    )
                    record["decision_time_bogota"] = config["data"]["decision_time"]
                    for column in audit_columns:
                        record[column] = frame.loc[origin_index, column]
                    rows.append(record)
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("Preregistered replay produced no forecasts")
    return result.sort_values(
        ["horizon_days", "origin_date", "model"]
    ).reset_index(drop=True)


def paired_tests(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    samples = int(config["statistics"]["bootstrap_samples"])
    seed = int(config["statistics"]["bootstrap_seed"])
    for period_index, period in enumerate(config["protocol"]["periods"]):
        subset = predictions[predictions["period"].eq(period)]
        for horizon, group in subset.groupby("horizon_days"):
            mature = group[group["actual"].notna()]
            paired = mature.pivot(
                index="origin_date", columns="model", values="correct"
            ).dropna().sort_index()
            price_hits = paired["price_only"].astype(bool).to_numpy()
            treatment_hits = paired[TREATMENT].astype(bool).to_numpy()
            treatment_only, price_only, raw_p = exact_mcnemar(
                price_hits, treatment_hits
            )
            delta = treatment_hits.astype(float) - price_hits.astype(float)
            block_length = int(math.ceil(int(horizon) / 5))
            ci_low, ci_high, p_nonpositive = moving_block_bootstrap(
                delta,
                block_length,
                samples,
                seed + period_index * 100 + int(horizon),
            )
            nonoverlap = paired.iloc[::block_length]
            treatment_metric = summary[
                summary["scope"].eq("protocol_period")
                & summary["period"].eq(period)
                & summary["horizon_days"].eq(int(horizon))
                & summary["model"].eq(TREATMENT)
            ].iloc[0]
            rows.append({
                "period": period,
                "horizon_days": int(horizon),
                "n_pairs": int(len(paired)),
                "price_only_da": float(price_hits.mean()) if len(paired) else None,
                "treatment_da": (
                    float(treatment_hits.mean()) if len(paired) else None
                ),
                "paired_da_delta": float(delta.mean()) if len(paired) else None,
                "discordant_treatment_right": treatment_only,
                "discordant_price_right": price_only,
                "mcnemar_p_raw": raw_p,
                "block_length_weeks": block_length,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "bootstrap_p_delta_nonpositive": p_nonpositive,
                "nonoverlap_n": int(len(nonoverlap)),
                "nonoverlap_price_da": (
                    float(nonoverlap["price_only"].mean())
                    if len(nonoverlap) else None
                ),
                "nonoverlap_treatment_da": (
                    float(nonoverlap[TREATMENT].mean())
                    if len(nonoverlap) else None
                ),
                "nonoverlap_da_delta": (
                    float(
                        (nonoverlap[TREATMENT] - nonoverlap["price_only"]).mean()
                    ) if len(nonoverlap) else None
                ),
                "treatment_balanced_accuracy": treatment_metric.get(
                    "balanced_accuracy"
                ),
                "treatment_minimum_class_recall": treatment_metric.get(
                    "minimum_class_recall"
                ),
            })
    result = pd.DataFrame(rows)
    adjusted = []
    for _, part in result.groupby("period", sort=False):
        part = part.copy()
        part["mcnemar_p_holm"] = holm_adjust(part["mcnemar_p_raw"])
        family_size = int(config["statistics"]["confirmatory_family_size"])
        part["mcnemar_p_family_bonferroni"] = (
            part["mcnemar_p_raw"] * family_size
        ).clip(upper=1.0)
        part["mcnemar_p_gate"] = part[[
            "mcnemar_p_holm", "mcnemar_p_family_bonferroni"
        ]].max(axis=1)
        adjusted.append(part)
    result = pd.concat(adjusted, ignore_index=True)
    gates = config["gates"]["incremental_primary_per_horizon"]
    primary = str(config["hypothesis"]["primary_period"])
    result["primary_incremental_gate_pass"] = pd.Series(
        pd.NA, index=result.index, dtype="boolean"
    )
    mask = result["period"].eq(primary)
    result.loc[mask, "primary_incremental_gate_pass"] = (
        result.loc[mask, "paired_da_delta"].gt(float(gates["paired_da_delta_gt"]))
        & result.loc[mask, "mcnemar_p_gate"].lt(
            float(gates["family_adjusted_mcnemar_p_lt"])
        )
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
    confirmation = config["gates"]["temporal_confirmation"]
    rows = []

    def test_row(period: str, horizon: int) -> pd.Series:
        return tests[
            tests["period"].eq(period) & tests["horizon_days"].eq(horizon)
        ].iloc[0]

    def metric_row(period: str, horizon: int) -> pd.Series:
        return summary[
            summary["scope"].eq("protocol_period")
            & summary["period"].eq(period)
            & summary["horizon_days"].eq(horizon)
            & summary["model"].eq(TREATMENT)
        ].iloc[0]

    for value in config["protocol"]["horizons_trading_days"]:
        horizon = int(value)
        primary = test_row(primary_name, horizon)
        replay = test_row("frozen_replay_2025", horizon)
        forward = test_row("weekly_retrain_2026", horizon)
        primary_metric = metric_row(primary_name, horizon)
        replay_metric = metric_row("frozen_replay_2025", horizon)
        forward_metric = metric_row("weekly_retrain_2026", horizon)
        replay_confirmation = bool(
            replay["paired_da_delta"]
            >= float(confirmation["frozen_replay_2025_da_delta_gte"])
            and replay_metric["directional_accuracy"]
            >= float(confirmation["frozen_replay_2025_directional_accuracy_gte"])
            and replay_metric["balanced_accuracy"]
            >= float(confirmation["frozen_replay_2025_balanced_accuracy_gte"])
            and replay_metric["lift_vs_causal_majority"]
            >= float(confirmation["frozen_replay_2025_lift_vs_causal_majority_gte"])
        )
        primary_pass = bool(primary["primary_incremental_gate_pass"])
        candidate = primary_pass and replay_confirmation
        reason = (
            "PROSPECTIVE_SHADOW_CANDIDATE_ONLY" if candidate
            else "FAIL_2025_TEMPORAL_CONFIRMATION" if primary_pass
            else "FAIL_PRIMARY_INCREMENTAL_GATE"
        )
        rows.append({
            "horizon_days": horizon,
            "primary_n_pairs": int(primary["n_pairs"]),
            "primary_price_da": primary["price_only_da"],
            "primary_treatment_da": primary["treatment_da"],
            "primary_da_delta": primary["paired_da_delta"],
            "primary_treatment_bda": primary_metric["balanced_accuracy"],
            "primary_up_recall": primary_metric["up_recall"],
            "primary_down_recall": primary_metric["down_recall"],
            "primary_family_adjusted_p": primary["mcnemar_p_gate"],
            "primary_bootstrap_ci_low": primary["bootstrap_ci_low"],
            "primary_bootstrap_ci_high": primary["bootstrap_ci_high"],
            "primary_nonoverlap_delta": primary["nonoverlap_da_delta"],
            "primary_gate_pass": primary_pass,
            "replay_2025_n_pairs": int(replay["n_pairs"]),
            "replay_2025_treatment_da": replay_metric["directional_accuracy"],
            "replay_2025_treatment_bda": replay_metric["balanced_accuracy"],
            "replay_2025_da_delta": replay["paired_da_delta"],
            "replay_2025_lift_vs_causal_majority": replay_metric[
                "lift_vs_causal_majority"
            ],
            "replay_2025_confirmation": replay_confirmation,
            "forward_2026_n_matured": int(forward_metric["n_matured"]),
            "forward_2026_treatment_da": forward_metric["directional_accuracy"],
            "forward_2026_treatment_bda": forward_metric["balanced_accuracy"],
            "forward_2026_da_delta": forward["paired_da_delta"],
            "forward_2026_lift_vs_causal_majority": forward_metric[
                "lift_vs_causal_majority"
            ],
            "forward_2026_diagnostic_mature_enough": bool(
                forward_metric["n_matured"] >= int(
                    confirmation[
                        "weekly_retrain_2026_is_diagnostic_until_matured_weeks"
                    ]
                )
            ),
            "prospective_shadow_candidate": candidate,
            "capital_authorized": False,
            "decision": reason,
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    config, registration, registration_path = load_protocol(config_path)
    print(
        f"Protocol verified: {registration['experiment_id']} "
        f"@ {registration['config_sha256'][:12]}",
        flush=True,
    )
    frame, provenance, coverage = build_research_frame(config)
    predictions = run_replay(frame, config)
    summary = summarize(predictions)
    tests = paired_tests(predictions, summary, config)
    decisions = horizon_decisions(tests, summary, config)

    output_dir = ROOT / config["outputs"]["directory"]
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / config["outputs"]["predictions"], index=False)
    summary.to_csv(output_dir / config["outputs"]["summary"], index=False)
    tests.to_csv(output_dir / config["outputs"]["paired_tests"], index=False)
    decisions.to_csv(output_dir / "horizon_decisions.csv", index=False)
    manifest = {
        "experiment_id": registration["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": sha256_file(config_path),
        "registration_sha256": sha256_file(registration_path),
        "script_sha256": sha256_file(Path(__file__)),
        "feature_builder_sha256": sha256_file(
            ROOT / "src/data/usdcop_intraday_peer_features.py"
        ),
        "data_sha256": registration["data_sha256"],
        "provenance": provenance,
        "prediction_rows": int(len(predictions)),
        "matured_prediction_rows": int(predictions["actual"].notna().sum()),
        "horizons_tested": [
            int(value) for value in config["protocol"]["horizons_trading_days"]
        ],
        "directional_trials_opened": 7,
        "directional_trials_total_after_run": 48,
        "global_trials_total_after_run": 109,
        "primary_gate_pass_count": int(decisions["primary_gate_pass"].sum()),
        "prospective_shadow_candidate_count": int(
            decisions["prospective_shadow_candidate"].sum()
        ),
        "capital_authorized": False,
        "evidence_class": config["_meta"]["evidence_class"],
        "warning": (
            "Hourly history has reconstructed, not first-seen, availability and "
            "2025-2026 were previously inspected. Historical output is research-only."
        ),
    }
    (output_dir / config["outputs"]["manifest"]).write_text(
        json.dumps(safe_json(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    overview = pd.DataFrame([
        {"field": "experiment_id", "value": registration["experiment_id"]},
        {"field": "evidence_class", "value": config["_meta"]["evidence_class"]},
        {"field": "all_7_horizons_tested", "value": True},
        {"field": "trials_opened", "value": 7},
        {"field": "primary_gate_pass_count", "value": int(decisions["primary_gate_pass"].sum())},
        {"field": "shadow_candidate_count", "value": int(decisions["prospective_shadow_candidate"].sum())},
        {"field": "capital_authorized", "value": False},
        {"field": "config_sha256", "value": sha256_file(config_path)},
        {"field": "warning", "value": manifest["warning"]},
    ])
    write_workbook(
        output_dir / config["outputs"]["workbook"],
        overview,
        decisions,
        summary,
        tests,
        predictions,
        pd.DataFrame(coverage),
    )
    print(decisions[[
        "horizon_days", "primary_price_da", "primary_treatment_da",
        "primary_da_delta", "primary_treatment_bda",
        "primary_family_adjusted_p", "primary_gate_pass",
        "replay_2025_treatment_da", "replay_2025_da_delta",
        "forward_2026_treatment_da", "forward_2026_da_delta",
        "prospective_shadow_candidate",
    ]].to_string(index=False), flush=True)
    print(
        f"Workbook: {output_dir / config['outputs']['workbook']}", flush=True
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
