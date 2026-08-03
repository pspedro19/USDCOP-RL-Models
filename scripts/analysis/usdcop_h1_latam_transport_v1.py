"""Locked external transport test for the frozen USD/COP H1 rule.

Use ``--register`` before the first result is computed.  Registration hashes
the full protocol, code and source snapshot without opening directional
outcomes.  The normal run refuses any drift.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analysis.usdcop_causal_regime_gate import build_states  # noqa: E402
from scripts.analysis.usdcop_directional_edge_tournament import (  # noqa: E402
    Variant,
    walk_forward_probabilities,
)
from scripts.analysis.usdcop_long_history_directional_tournament import (  # noqa: E402
    build_price_features,
)
from scripts.validation.evaluate_usdcop_h1_shadow_v2 import (  # noqa: E402
    pesaran_timmermann,
    wilson_interval,
)


CONFIG = ROOT / "config/forecast_experiments/usdcop_h1_latam_transport_v1.yaml"
RUNTIME_PACKAGES = ("numpy", "pandas", "scikit-learn", "scipy", "PyYAML", "pyarrow")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        **{
            package: importlib.metadata.version(package)
            for package in RUNTIME_PACKAGES
        },
    }


def paths(contract: dict[str, Any]) -> dict[str, Path]:
    output = contract["outputs"]
    directory = ROOT / output["directory"]
    return {
        "registration": ROOT / contract["implementation_integrity"]["registration_file"],
        "directory": directory,
        "predictions": directory / output["predictions"],
        "metrics": directory / output["metrics"],
        "annual": directory / output["annual_metrics"],
        "result": directory / output["result"],
    }


def registered_files(contract: dict[str, Any]) -> list[str]:
    integrity = contract["implementation_integrity"]
    return [str(integrity["audit_script"]), *map(str, integrity["hashed_dependencies"])]


def register(contract: dict[str, Any]) -> int:
    output = paths(contract)
    if output["registration"].exists():
        raise SystemExit(f"Registration is immutable and already exists: {output['registration']}")
    existing_results = [
        path for name, path in output.items()
        if name not in {"registration", "directory"} and path.exists()
    ]
    if existing_results:
        raise SystemExit(f"Cannot register after outputs exist: {existing_results}")
    source = ROOT / contract["data"]["source_file"]
    if sha256(source) != contract["data"]["source_sha256"]:
        raise SystemExit("Source snapshot hash does not match protocol")
    code_hashes = {relative: sha256(ROOT / relative) for relative in registered_files(contract)}
    registration = {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_sha256": sha256(CONFIG),
        "source_sha256": sha256(source),
        "code_sha256": code_hashes,
        "runtime_versions": runtime_versions(),
        "primary_period": [contract["scope"]["primary_start"], contract["scope"]["primary_end"]],
        "assets": contract["scope"]["assets"],
        "historical_transport_metrics_opened_at_registration": False,
        "new_primary_trials": contract["trial_accounting"]["new_primary_trials"],
        "directional_trials_after": contract["trial_accounting"]["directional_trials_after"],
        "global_trials_after": contract["trial_accounting"]["global_trials_after"],
        "signal_authorized": False,
        "capital_authorized": False,
        "registration_valid": True,
    }
    output["registration"].parent.mkdir(parents=True, exist_ok=True)
    output["registration"].write_text(json.dumps(registration, indent=2), encoding="utf-8")
    print(json.dumps(registration, indent=2))
    return 0


def load_registered() -> tuple[dict[str, Any], dict[str, Any]]:
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    registration_path = paths(contract)["registration"]
    if not registration_path.exists():
        raise RuntimeError("Transport test is not preregistered")
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    if registration.get("registration_valid") is not True:
        raise RuntimeError("Transport registration is invalid")
    if registration["contract_sha256"] != sha256(CONFIG):
        raise RuntimeError("Transport contract drift")
    source = ROOT / contract["data"]["source_file"]
    if registration["source_sha256"] != sha256(source):
        raise RuntimeError("Transport source drift")
    for relative, expected in registration["code_sha256"].items():
        if sha256(ROOT / relative) != expected:
            raise RuntimeError(f"Transport code drift: {relative}")
    if registration.get("runtime_versions") != runtime_versions():
        raise RuntimeError(
            "Transport runtime drift: "
            f"registered={registration.get('runtime_versions')} current={runtime_versions()}"
        )
    return contract, registration


def build_asset_frame(contract: dict[str, Any], symbol: str) -> pd.DataFrame:
    source = pd.read_parquet(ROOT / contract["data"]["source_file"])
    frame = source[
        source["symbol"].eq(symbol)
        & source["source"].eq(contract["data"]["required_source"])
    ].copy()
    frame["date"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None).dt.normalize()
    frame = frame[
        frame["date"].between(
            pd.Timestamp(contract["scope"]["history_start"]),
            pd.Timestamp(contract["scope"]["diagnostic_end"]),
        )
    ].copy()
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    columns = ["date", "open", "high", "low", "close"]
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[columns].isna().any().any():
        raise RuntimeError(f"{symbol} contains null OHLC")
    if frame[columns[1:]].le(0).any().any():
        raise RuntimeError(f"{symbol} contains non-positive OHLC")
    if not (
        frame["high"].ge(frame[["open", "close"]].max(axis=1)).all()
        and frame["low"].le(frame[["open", "close"]].min(axis=1)).all()
    ):
        raise RuntimeError(f"{symbol} violates OHLC ordering")
    if frame["date"].min() > pd.Timestamp("2001-01-01"):
        raise RuntimeError(f"{symbol} has insufficient estimation history")
    return build_price_features(frame)


def transported_predictions(
    contract: dict[str, Any], symbol: str,
) -> pd.DataFrame:
    frame = build_asset_frame(contract, symbol)
    model = contract["model"]
    variant = Variant(
        str(model["feature_group"]), float(model["c"]),
        str(model["class_weight"]), int(model["half_life_sessions"]),
    )
    predictions = walk_forward_probabilities(
        frame,
        list(map(str, model["features"])),
        int(contract["scope"]["horizon_sessions"]),
        variant,
        contract["scope"]["primary_start"],
        contract["scope"]["diagnostic_end"],
    )
    states = build_states(frame)
    regime_by_date = dict(zip(pd.to_datetime(frame["date"]), states["regime"].astype(str)))
    trend_by_date = dict(zip(pd.to_datetime(frame["date"]), states["trend_z"].astype(float)))
    predictions["asset"] = symbol
    predictions["regime"] = pd.to_datetime(predictions["origin_date"]).map(regime_by_date)
    predictions["trend_z"] = pd.to_datetime(predictions["origin_date"]).map(trend_by_date)
    allowed = set(map(str, contract["regime_gate"]["allowed_states"]))
    predictions["regime_allowed"] = predictions["regime"].isin(allowed)
    predictions["base_prediction"] = (predictions["probability_up"] >= 0.50).astype(int)
    predictions["prediction"] = predictions["base_prediction"].where(
        predictions["regime_allowed"], pd.NA
    ).astype("Int64")
    predictions["decision"] = predictions["prediction"].map({0: "DOWN", 1: "UP"}).fillna("FLAT")
    predictions["period"] = np.where(
        pd.to_datetime(predictions["origin_date"]).le(pd.Timestamp(contract["scope"]["primary_end"])),
        "primary_2020_2025", "diagnostic_2026",
    )
    return predictions


def classification_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    matured = frame[frame["actual"].notna()].copy()
    signals = matured[matured["prediction"].notna()].copy()
    result: dict[str, Any] = {
        "n_total": len(matured),
        "n_signals": len(signals),
        "coverage": len(signals) / len(matured) if len(matured) else 0.0,
        "directional_accuracy": None,
        "balanced_accuracy": None,
        "up_recall": None,
        "down_recall": None,
        "minimum_class_recall": None,
        "prediction_up_rate": None,
        "actual_up_rate": None,
        "causal_majority_accuracy": None,
        "lift_vs_causal_majority": None,
        "pesaran_timmermann_p": None,
        "wilson_95_low": None,
        "wilson_95_high": None,
    }
    if signals.empty:
        return result
    actual = signals["actual"].astype(int).to_numpy()
    predicted = signals["prediction"].astype(int).to_numpy()
    majority = (signals["train_up_rate"].astype(float).to_numpy() >= 0.5).astype(int)
    hits = predicted == actual
    up = actual == 1
    down = actual == 0
    up_recall = float(predicted[up].mean()) if up.any() else None
    down_recall = float((1 - predicted[down]).mean()) if down.any() else None
    pt = pesaran_timmermann(actual, predicted)
    interval = wilson_interval(int(hits.sum()), len(hits))
    result.update({
        "directional_accuracy": float(hits.mean()),
        "balanced_accuracy": 0.5 * (up_recall + down_recall)
        if up_recall is not None and down_recall is not None else None,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "minimum_class_recall": min(up_recall, down_recall)
        if up_recall is not None and down_recall is not None else None,
        "prediction_up_rate": float(predicted.mean()),
        "actual_up_rate": float(actual.mean()),
        "causal_majority_accuracy": float((majority == actual).mean()),
        "lift_vs_causal_majority": float((hits.astype(float) - (majority == actual)).mean()),
        "pesaran_timmermann_p": pt["one_sided_p"],
        "wilson_95_low": interval["lower"],
        "wilson_95_high": interval["upper"],
    })
    return result


def combined_bootstrap(frame: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    """Block bootstrap the equal-asset-week lift while clustering by ISO week.

    Zero-signal weeks remain in the ordered cluster sequence. Resampling sums
    cluster numerators and denominators separately, so a week with one signal
    never receives the same asset-week weight as a week with two signals.
    """
    matured = frame[frame["actual"].notna()].copy()
    if matured.duplicated(["iso_week", "asset"]).any():
        raise RuntimeError("Duplicate asset-week rows in transport estimand")
    signal = matured["prediction"].notna()
    predicted = matured["prediction"].fillna(0).astype(int)
    actual = matured["actual"].astype(int)
    majority = matured["train_up_rate"].astype(float).ge(0.5).astype(int)
    matured["signal_count"] = signal.astype(int)
    matured["lift_sum"] = np.where(
        signal,
        predicted.eq(actual).astype(float) - majority.eq(actual).astype(float),
        0.0,
    )
    clusters = matured.groupby("iso_week", sort=True).agg(
        lift_sum=("lift_sum", "sum"),
        signal_count=("signal_count", "sum"),
    )
    sums = clusters["lift_sum"].to_numpy(dtype=float)
    counts = clusters["signal_count"].to_numpy(dtype=float)
    n_weeks = len(clusters)
    n_signals = int(counts.sum())
    if n_signals == 0:
        return {
            "mean": None, "ci_low": None, "ci_high": None,
            "one_sided_p": None, "n_weeks": n_weeks,
            "n_signal_asset_weeks": 0,
        }
    spec = contract["primary_estimand"]
    block_length = int(spec["block_length_weeks"])
    samples = int(spec["bootstrap_samples"])
    if n_weeks < max(8, 2 * block_length):
        return {
            "mean": float(sums.sum() / counts.sum()),
            "ci_low": None, "ci_high": None, "one_sided_p": None,
            "n_weeks": n_weeks, "n_signal_asset_weeks": n_signals,
        }
    observed = float(sums.sum() / counts.sum())
    centered_sums = sums - observed * counts
    rng = np.random.default_rng(int(spec["bootstrap_seed"]))
    starts = np.arange(n_weeks)
    width = int(math.ceil(n_weeks / block_length))
    sampled_means = np.empty(samples, dtype=float)
    null_means = np.empty(samples, dtype=float)
    for sample in range(samples):
        chosen = rng.choice(starts, size=width, replace=True)
        indices = np.concatenate([
            (np.arange(start, start + block_length) % n_weeks)
            for start in chosen
        ])[:n_weeks]
        denominator = counts[indices].sum()
        if denominator <= 0:
            sampled_means[sample] = np.nan
            null_means[sample] = np.nan
            continue
        sampled_means[sample] = sums[indices].sum() / denominator
        null_means[sample] = centered_sums[indices].sum() / denominator
    sampled_means = sampled_means[np.isfinite(sampled_means)]
    null_means = null_means[np.isfinite(null_means)]
    if not len(sampled_means) or not len(null_means):
        raise RuntimeError("Week-cluster bootstrap produced no valid draws")
    return {
        "mean": observed,
        "ci_low": float(np.quantile(sampled_means, 0.025)),
        "ci_high": float(np.quantile(sampled_means, 0.975)),
        "one_sided_p": float((1 + np.sum(null_means >= observed)) / (len(null_means) + 1)),
        "n_weeks": n_weeks,
        "n_signal_asset_weeks": n_signals,
        "valid_draws": int(len(sampled_means)),
        "block_length_weeks": block_length,
    }


def evaluate_gates(
    combined: dict[str, Any],
    by_asset: dict[str, dict[str, Any]],
    annual: list[dict[str, Any]],
    bootstrap: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, bool]:
    gates = contract["pass_gates"]
    asset_constraints = []
    for metrics in by_asset.values():
        rate = metrics["prediction_up_rate"]
        asset_constraints.append(bool(
            metrics["directional_accuracy"] is not None
            and metrics["directional_accuracy"] >= float(gates["minimum_each_asset_directional_accuracy"])
            and metrics["balanced_accuracy"] is not None
            and metrics["balanced_accuracy"] >= float(gates["minimum_each_asset_balanced_accuracy"])
            and metrics["minimum_class_recall"] is not None
            and metrics["minimum_class_recall"] >= float(gates["minimum_each_asset_class_recall"])
            and rate is not None
            and float(gates["minimum_prediction_class_rate"]) <= rate <= float(gates["maximum_prediction_class_rate"])
        ))
    required_years = set(map(int, gates["annual_stability_years"]))
    annual_map = {int(row["year"]): row for row in annual}
    annual_stable = required_years.issubset(annual_map) and all(
        annual_map[year]["n_signals"] >= int(gates["minimum_annual_combined_signals"])
        and annual_map[year]["balanced_accuracy"] is not None
        and annual_map[year]["balanced_accuracy"] >= float(gates["minimum_annual_combined_balanced_accuracy"])
        for year in required_years
    )
    return {
        "minimum_combined_signals": combined["n_signals"] >= int(gates["minimum_combined_signals"]),
        "minimum_combined_coverage": combined["coverage"] >= float(gates["minimum_combined_coverage"]),
        "minimum_combined_directional_accuracy": combined["directional_accuracy"] is not None
        and combined["directional_accuracy"] >= float(gates["minimum_combined_directional_accuracy"]),
        "minimum_combined_balanced_accuracy": combined["balanced_accuracy"] is not None
        and combined["balanced_accuracy"] >= float(gates["minimum_combined_balanced_accuracy"]),
        "all_asset_constraints": all(asset_constraints),
        "positive_combined_lift": combined["lift_vs_causal_majority"] is not None
        and combined["lift_vs_causal_majority"] > float(gates["minimum_combined_lift_vs_causal_majority"]),
        "lift_bootstrap_ci_lower_strict": bootstrap["ci_low"] is not None
        and bootstrap["ci_low"] > float(gates["lift_block_bootstrap_ci95_lower_strictly_gt"]),
        "lift_bootstrap_p": bootstrap["one_sided_p"] is not None
        and bootstrap["one_sided_p"] <= float(gates["lift_block_bootstrap_one_sided_p_max"]),
        "annual_stability": annual_stable,
    }


def main() -> int:
    args = parse_args()
    contract = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.register:
        return register(contract)
    contract, registration = load_registered()
    output = paths(contract)
    existing_results = [
        path for name, path in output.items()
        if name not in {"registration", "directory"} and path.exists()
    ]
    if existing_results:
        raise RuntimeError(f"Locked one-shot result already exists: {existing_results}")
    frames = [transported_predictions(contract, symbol) for symbol in contract["scope"]["assets"]]
    predictions = pd.concat(frames, ignore_index=True).sort_values(["origin_date", "asset"])
    primary = predictions[predictions["period"].eq("primary_2020_2025")].copy()
    diagnostic = predictions[predictions["period"].eq("diagnostic_2026")].copy()

    primary_combined = classification_metrics(primary)
    primary_by_asset = {
        symbol: classification_metrics(primary[primary["asset"].eq(symbol)])
        for symbol in contract["scope"]["assets"]
    }
    diagnostic_combined = classification_metrics(diagnostic)
    diagnostic_by_asset = {
        symbol: classification_metrics(diagnostic[diagnostic["asset"].eq(symbol)])
        for symbol in contract["scope"]["assets"]
    }
    annual = []
    for year, group in primary.groupby(pd.to_datetime(primary["origin_date"]).dt.year):
        annual.append({"year": int(year), **classification_metrics(group)})
    bootstrap = combined_bootstrap(primary, contract)
    if not np.isclose(
        float(bootstrap["mean"]),
        float(primary_combined["lift_vs_causal_majority"]),
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Bootstrap estimand drift: clustered mean does not equal pooled asset-week lift"
        )
    gates = evaluate_gates(primary_combined, primary_by_asset, annual, bootstrap, contract)
    passed = all(gates.values())

    metric_rows = [
        {"period": "primary_2020_2025", "asset": "COMBINED", **primary_combined},
        *[
            {"period": "primary_2020_2025", "asset": asset, **metrics}
            for asset, metrics in primary_by_asset.items()
        ],
        {"period": "diagnostic_2026", "asset": "COMBINED", **diagnostic_combined},
        *[
            {"period": "diagnostic_2026", "asset": asset, **metrics}
            for asset, metrics in diagnostic_by_asset.items()
        ],
    ]
    output["directory"].mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output["predictions"], index=False)
    pd.DataFrame(metric_rows).to_csv(output["metrics"], index=False)
    pd.DataFrame(annual).to_csv(output["annual"], index=False)
    result = {
        "schema_version": "1.0.0",
        "experiment_id": contract["_meta"]["experiment_id"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "registration_sha256": sha256(output["registration"]),
        "source_sha256": registration["source_sha256"],
        "primary_period": "2020-2025",
        "primary_combined": primary_combined,
        "primary_by_asset": primary_by_asset,
        "primary_annual": annual,
        "paired_week_cluster_bootstrap": bootstrap,
        "pesaran_timmermann_role": "descriptive_not_an_inferential_gate",
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "primary_pass": passed,
        "verdict": "PASS_EXTERNAL_TRANSPORT_SUPPORT" if passed else "FAIL_CLOSE_TRANSPORT_FAMILY",
        "diagnostic_2026": {
            "combined": diagnostic_combined,
            "by_asset": diagnostic_by_asset,
            "can_rescue_primary_failure": False,
        },
        "signal_authorized": False,
        "capital_authorized": False,
        "interpretation": contract["governance"]["interpretation_if_pass" if passed else "interpretation_if_fail"],
    }
    output["result"].write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    result["output_sha256"] = {
        name: sha256(output[name]) for name in ("predictions", "metrics", "annual")
    }
    output["result"].write_text(
        json.dumps(json_safe(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({
        "experiment_id": result["experiment_id"],
        "primary_pass": passed,
        "verdict": result["verdict"],
        "combined_signals": primary_combined["n_signals"],
        "combined_da": primary_combined["directional_accuracy"],
        "combined_bda": primary_combined["balanced_accuracy"],
        "failed_gates": result["failed_gates"],
        "capital_authorized": False,
    }, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
