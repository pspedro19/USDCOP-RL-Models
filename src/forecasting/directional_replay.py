"""Causal weekly directional replay for USD/COP.

This module turns the research-grade PIT macro classifier into an immutable
weekly publication contract.  It deliberately keeps model selection, feature
selection, label maturity and horizon selection separate:

* features are selected with labels that mature before the configured cutoff;
* model memory/thresholds are selected only in the configured validation years;
* 2025 is inferred with models frozen at the end of 2024;
* 2026 is refit at every weekly origin with only labels mature at that origin;
* a horizon can enter the weekly consensus only from already-mature forecasts.

The resulting direction is a shadow/research decision.  Execution authorization
is intentionally outside this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.analysis.weekly_forecasting_oos_canonical import build_frame


SCHEMA_VERSION = "1.1.0"
MODEL_FAMILY = "pit_macro_logistic"
POINT_MODEL_FAMILY = "ridge_log_return"


@dataclass(frozen=True)
class ReplayPaths:
    dashboard_dir: Path
    index_file: Path
    ledger_file: Path
    report_ledger: Path
    report_summary: Path
    report_workbook: Path


def load_replay_config(path: Path) -> dict[str, Any]:
    """Load and minimally validate the replay configuration."""
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    required = ("_meta", "asset", "protocol", "horizons", "selection", "outputs")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"Directional replay config missing sections: {missing}")
    horizons = tuple(int(value) for value in cfg["horizons"]["values"])
    if horizons != (1, 5, 10, 15, 20, 25, 30):
        raise ValueError(f"Unexpected USD/COP horizon contract: {horizons}")
    if cfg["_meta"].get("signal_authorized") is not False:
        raise ValueError("Directional replay must remain signal_authorized=false")
    return cfg


def resolve_paths(root: Path, cfg: dict[str, Any]) -> ReplayPaths:
    outputs = cfg["outputs"]
    dashboard = root / outputs["dashboard_dir"]
    return ReplayPaths(
        dashboard_dir=dashboard,
        index_file=dashboard / outputs["index_file"],
        ledger_file=dashboard / outputs["ledger_file"],
        report_ledger=root / outputs["report_ledger"],
        report_summary=root / outputs["report_summary"],
        report_workbook=root / outputs["report_workbook"],
    )


def _safe_float(value: Any, digits: int = 6) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return round(number, digits)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _make_target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    future_return = _make_return_target(frame, horizon)
    return (future_return > 0).where(future_return.notna()).astype(float)


def _make_return_target(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward log return whose label matures exactly ``horizon`` sessions later."""
    return np.log(frame["close"].shift(-horizon) / frame["close"])


def _weekly_origins(frame: pd.DataFrame, start: str) -> pd.DataFrame:
    dates = pd.to_datetime(frame["date"])
    origins = frame.assign(iso_week=dates.dt.strftime("%G-W%V")).groupby("iso_week").tail(1)
    return origins[origins["date"] >= pd.Timestamp(start)].copy()


def _eligible_train_indices(
    target: pd.Series,
    origin_index: int,
    horizon: int,
) -> np.ndarray:
    # The last included label ends at ``origin_index``.  Nothing crossing the
    # prediction origin is admitted.
    indices = np.arange(0, max(0, origin_index - horizon + 1))
    return indices[target.iloc[indices].notna().to_numpy()]


def _sample_weights(indices: np.ndarray, half_life: int | None) -> np.ndarray | None:
    if half_life is None or not len(indices):
        return None
    ages = indices[-1] - indices
    return np.exp(-np.log(2.0) * ages / half_life)


def _new_model(c_value: float, max_iter: int) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=c_value, max_iter=max_iter, random_state=23),
    )


def _fit_model(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    train_indices: np.ndarray,
    half_life: int | None,
    c_value: float,
    max_iter: int,
) -> Pipeline:
    if len(train_indices) < 300:
        raise ValueError(f"Insufficient mature labels: {len(train_indices)}")
    model = _new_model(c_value, max_iter)
    weights = _sample_weights(train_indices, half_life)
    kwargs: dict[str, Any] = {}
    if weights is not None:
        kwargs["logisticregression__sample_weight"] = weights
    model.fit(
        frame.loc[train_indices, features],
        target.iloc[train_indices].astype(int),
        **kwargs,
    )
    return model


def _new_point_model(alpha: float) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha),
    )


def _fit_point_model(
    frame: pd.DataFrame,
    target: pd.Series,
    features: list[str],
    train_indices: np.ndarray,
    half_life: int | None,
    alpha: float,
    clip_quantiles: tuple[float, float],
) -> tuple[Pipeline, tuple[float, float]]:
    """Fit a causal log-return point model and causal anti-extrapolation bounds."""
    if len(train_indices) < 300:
        raise ValueError(f"Insufficient mature point labels: {len(train_indices)}")
    model = _new_point_model(alpha)
    weights = _sample_weights(train_indices, half_life)
    kwargs: dict[str, Any] = {}
    if weights is not None:
        kwargs["ridge__sample_weight"] = weights
    y_train = target.iloc[train_indices].astype(float)
    model.fit(frame.loc[train_indices, features], y_train, **kwargs)
    lower, upper = np.quantile(y_train.to_numpy(), clip_quantiles)
    return model, (float(lower), float(upper))


def _point_prediction_row(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    origin_index: int,
    horizon: int,
    model: Pipeline,
    clip_bounds: tuple[float, float],
    features: list[str],
    train_indices: np.ndarray,
) -> dict[str, Any]:
    origin = frame.loc[origin_index]
    base_price = float(origin["close"])
    raw_return = float(model.predict(frame.loc[[origin_index], features])[0])
    forecast_return = float(np.clip(raw_return, clip_bounds[0], clip_bounds[1]))
    forecast_price = float(base_price * np.exp(forecast_return))
    actual_return = target.iloc[origin_index]
    actual_return = None if pd.isna(actual_return) else float(actual_return)
    actual_price = None if actual_return is None else float(base_price * np.exp(actual_return))
    label_end = None
    if len(train_indices):
        label_index = int(train_indices[-1]) + horizon
        if label_index < len(frame):
            label_end = pd.Timestamp(frame.loc[label_index, "date"])
    return {
        "iso_week": pd.Timestamp(origin["date"]).strftime("%G-W%V"),
        "origin_date": pd.Timestamp(origin["date"]),
        "horizon_days": horizon,
        "forecast_log_return": forecast_return,
        "forecast_return_pct": float((np.exp(forecast_return) - 1.0) * 100.0),
        "forecast_price": forecast_price,
        "forecast_price_change": forecast_price - base_price,
        "point_forecast_direction": "UP" if forecast_return >= 0 else "DOWN",
        "point_forecast_clipped": bool(not np.isclose(raw_return, forecast_return)),
        "actual_log_return": actual_return,
        "actual_price": actual_price,
        "point_abs_error_price": (
            None if actual_price is None else abs(forecast_price - actual_price)
        ),
        "point_abs_error_pct": (
            None if actual_price is None
            else abs(forecast_price - actual_price) / actual_price * 100.0
        ),
        "point_train_count": int(len(train_indices)),
        "point_train_label_end": label_end,
    }


def walk_forward_point_forecasts(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    half_life: int | None,
    origin_start: str,
    origin_end: str,
    alpha: float,
    clip_quantiles: tuple[float, float],
) -> pd.DataFrame:
    """Causal weekly point forecasts for validation/model selection."""
    target = _make_return_target(frame, horizon)
    origins = _weekly_origins(frame, origin_start)
    origins = origins[origins["date"] <= pd.Timestamp(origin_end)]
    rows: list[dict[str, Any]] = []
    for origin_index in origins.index:
        train = _eligible_train_indices(target, int(origin_index), horizon)
        if len(train) < 300:
            continue
        model, bounds = _fit_point_model(
            frame, target, features, train, half_life, alpha, clip_quantiles
        )
        rows.append(_point_prediction_row(
            frame=frame,
            target=target,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            clip_bounds=bounds,
            features=features,
            train_indices=train,
        ))
    return pd.DataFrame(rows)


def frozen_year_point_forecasts(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    half_life: int | None,
    year: int,
    alpha: float,
    clip_quantiles: tuple[float, float],
) -> pd.DataFrame:
    """Fit the numeric return model once before the frozen OOS year."""
    target = _make_return_target(frame, horizon)
    dates = pd.to_datetime(frame["date"])
    cutoff_index = int(np.searchsorted(dates, pd.Timestamp(f"{year}-01-01")))
    train = np.arange(0, max(0, cutoff_index - horizon))
    train = train[target.iloc[train].notna().to_numpy()]
    model, bounds = _fit_point_model(
        frame, target, features, train, half_life, alpha, clip_quantiles
    )
    origins = _weekly_origins(frame, f"{year}-01-01")
    origins = origins[origins["date"] < pd.Timestamp(f"{year + 1}-01-01")]
    return pd.DataFrame([
        _point_prediction_row(
            frame=frame,
            target=target,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            clip_bounds=bounds,
            features=features,
            train_indices=train,
        )
        for origin_index in origins.index
    ])


def expanding_year_point_forecasts(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    half_life: int | None,
    year: int,
    alpha: float,
    clip_quantiles: tuple[float, float],
) -> pd.DataFrame:
    """Refit the numeric return model weekly with mature labels only."""
    target = _make_return_target(frame, horizon)
    origins = _weekly_origins(frame, f"{year}-01-01")
    origins = origins[origins["date"] < pd.Timestamp(f"{year + 1}-01-01")]
    rows: list[dict[str, Any]] = []
    for origin_index in origins.index:
        train = _eligible_train_indices(target, int(origin_index), horizon)
        if len(train) < 300:
            continue
        model, bounds = _fit_point_model(
            frame, target, features, train, half_life, alpha, clip_quantiles
        )
        rows.append(_point_prediction_row(
            frame=frame,
            target=target,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            clip_bounds=bounds,
            features=features,
            train_indices=train,
        ))
    return pd.DataFrame(rows)


def select_frozen_features(
    frame: pd.DataFrame,
    candidates: list[str],
    horizon: int,
    cutoff: str,
    feature_count: int,
) -> list[str]:
    """Select features using labels whose target ends strictly before cutoff."""
    target = _make_target(frame, horizon)
    dates = pd.to_datetime(frame["date"])
    cutoff_index = int(np.searchsorted(dates, pd.Timestamp(cutoff)))
    eligible = np.arange(0, max(0, cutoff_index - horizon))
    eligible = eligible[target.iloc[eligible].notna().to_numpy()]
    usable = [name for name in candidates if frame.loc[eligible, name].notna().sum() >= 250]
    if not usable:
        raise ValueError(f"No eligible features for H{horizon} before {cutoff}")
    values = SimpleImputer(strategy="median").fit_transform(frame.loc[eligible, usable])
    scores = mutual_info_classif(
        values,
        target.iloc[eligible].astype(int).to_numpy(),
        random_state=23,
    )
    order = np.argsort(scores)[::-1][: min(feature_count, len(usable))]
    return [usable[index] for index in order]


def _target_metadata(
    frame: pd.DataFrame,
    origin_index: int,
    horizon: int,
    target: pd.Series,
) -> tuple[pd.Timestamp, bool, int | None]:
    target_index = origin_index + horizon
    if target_index < len(frame):
        target_date = pd.Timestamp(frame.loc[target_index, "date"])
        actual = None if pd.isna(target.iloc[origin_index]) else int(target.iloc[origin_index])
        return target_date, False, actual
    target_date = pd.Timestamp(frame.loc[origin_index, "date"]) + pd.offsets.BDay(horizon)
    return target_date, True, None


def _prediction_row(
    *,
    frame: pd.DataFrame,
    target: pd.Series,
    origin_index: int,
    horizon: int,
    model: Pipeline,
    features: list[str],
    train_indices: np.ndarray,
    half_life: int | None,
    training_mode: str,
) -> dict[str, Any]:
    origin = frame.loc[origin_index]
    probability = float(model.predict_proba(frame.loc[[origin_index], features])[0, 1])
    target_date, estimated, actual = _target_metadata(frame, origin_index, horizon, target)
    label_end = None
    if len(train_indices):
        label_index = int(train_indices[-1]) + horizon
        if label_index < len(frame):
            label_end = pd.Timestamp(frame.loc[label_index, "date"])
    return {
        "iso_week": pd.Timestamp(origin["date"]).strftime("%G-W%V"),
        "origin_date": pd.Timestamp(origin["date"]),
        "base_price": float(origin["close"]),
        "horizon_days": horizon,
        "target_date": target_date,
        "target_date_estimated": bool(estimated),
        "probability_up": probability,
        "actual": actual,
        "train_count": int(len(train_indices)),
        "train_label_end": label_end,
        "half_life": "expanding" if half_life is None else str(half_life),
        "training_mode": training_mode,
    }


def walk_forward_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    half_life: int | None,
    origin_start: str,
    origin_end: str,
    c_value: float,
    max_iter: int,
) -> pd.DataFrame:
    """Generate causal weekly probabilities, including unresolved final targets."""
    target = _make_target(frame, horizon)
    origins = _weekly_origins(frame, origin_start)
    origins = origins[origins["date"] <= pd.Timestamp(origin_end)]
    rows: list[dict[str, Any]] = []
    for origin_index in origins.index:
        train = _eligible_train_indices(target, int(origin_index), horizon)
        if len(train) < 300:
            continue
        model = _fit_model(
            frame, target, features, train, half_life, c_value, max_iter
        )
        rows.append(_prediction_row(
            frame=frame,
            target=target,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            features=features,
            train_indices=train,
            half_life=half_life,
            training_mode="weekly_expanding_validation",
        ))
    return pd.DataFrame(rows)


def frozen_year_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    half_life: int | None,
    year: int,
    c_value: float,
    max_iter: int,
) -> pd.DataFrame:
    """Fit once before ``year`` and infer every weekly origin without refitting."""
    target = _make_target(frame, horizon)
    dates = pd.to_datetime(frame["date"])
    cutoff_index = int(np.searchsorted(dates, pd.Timestamp(f"{year}-01-01")))
    train = np.arange(0, max(0, cutoff_index - horizon))
    train = train[target.iloc[train].notna().to_numpy()]
    model = _fit_model(frame, target, features, train, half_life, c_value, max_iter)
    origins = _weekly_origins(frame, f"{year}-01-01")
    origins = origins[origins["date"] < pd.Timestamp(f"{year + 1}-01-01")]
    rows = [
        _prediction_row(
            frame=frame,
            target=target,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            features=features,
            train_indices=train,
            half_life=half_life,
            training_mode=f"frozen_pre_{year}",
        )
        for origin_index in origins.index
    ]
    return pd.DataFrame(rows)


def expanding_year_probabilities(
    frame: pd.DataFrame,
    features: list[str],
    horizon: int,
    half_life: int | None,
    year: int,
    c_value: float,
    max_iter: int,
) -> pd.DataFrame:
    """Refit at each origin from labels mature at that exact origin."""
    target = _make_target(frame, horizon)
    origins = _weekly_origins(frame, f"{year}-01-01")
    origins = origins[origins["date"] < pd.Timestamp(f"{year + 1}-01-01")]
    rows: list[dict[str, Any]] = []
    for origin_index in origins.index:
        train = _eligible_train_indices(target, int(origin_index), horizon)
        if len(train) < 300:
            continue
        model = _fit_model(frame, target, features, train, half_life, c_value, max_iter)
        rows.append(_prediction_row(
            frame=frame,
            target=target,
            origin_index=int(origin_index),
            horizon=horizon,
            model=model,
            features=features,
            train_indices=train,
            half_life=half_life,
            training_mode="weekly_expanding_matured_labels",
        ))
    return pd.DataFrame(rows)


def directional_metrics(frame: pd.DataFrame, threshold: float) -> dict[str, Any]:
    matured = frame[frame["actual"].notna()].copy()
    if matured.empty:
        return {
            "n": 0, "directional_accuracy": None, "balanced_accuracy": None,
            "up_recall": None, "down_recall": None, "minimum_class_recall": None,
            "brier": None, "prediction_up_rate": None, "actual_up_rate": None,
            "up_count": 0, "down_count": 0,
        }
    actual = matured["actual"].astype(int)
    prediction = (matured["probability_up"] >= threshold).astype(int)
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
    return {
        "n": int(len(matured)),
        "directional_accuracy": float((prediction == actual).mean()),
        "balanced_accuracy": balanced,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "minimum_class_recall": minimum,
        "brier": float(np.mean((matured["probability_up"] - actual) ** 2)),
        "prediction_up_rate": float(prediction.mean()),
        "actual_up_rate": float(actual.mean()),
        "up_count": int(up.sum()),
        "down_count": int(down.sum()),
    }


def point_forecast_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Evaluate numeric prices against the honest no-change/spot baseline."""
    matured = frame[frame["actual_log_return"].notna()].copy()
    if matured.empty:
        return {
            "n": 0,
            "mae_log_return": None,
            "rmse_log_return": None,
            "naive_mae_log_return": None,
            "naive_rmse_log_return": None,
            "mae_skill_vs_spot": None,
            "rmse_skill_vs_spot": None,
            "mae_price": None,
            "mape_price": None,
            "directional_accuracy": None,
            "balanced_accuracy": None,
            "up_recall": None,
            "down_recall": None,
            "minimum_class_recall": None,
            "bias_log_return": None,
        }
    actual = matured["actual_log_return"].astype(float).to_numpy()
    forecast = matured["forecast_log_return"].astype(float).to_numpy()
    error = forecast - actual
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    naive_mae = float(np.mean(np.abs(actual)))
    naive_rmse = float(np.sqrt(np.mean(np.square(actual))))
    direction_frame = pd.DataFrame({
        "probability_up": (forecast >= 0).astype(float),
        "actual": (actual >= 0).astype(int),
    })
    direction = directional_metrics(direction_frame, 0.5)
    actual_price = matured["actual_price"].astype(float).to_numpy()
    forecast_price = matured["forecast_price"].astype(float).to_numpy()
    return {
        "n": int(len(matured)),
        "mae_log_return": mae,
        "rmse_log_return": rmse,
        "naive_mae_log_return": naive_mae,
        "naive_rmse_log_return": naive_rmse,
        "mae_skill_vs_spot": None if naive_mae <= 0 else float(1.0 - mae / naive_mae),
        "rmse_skill_vs_spot": None if naive_rmse <= 0 else float(1.0 - rmse / naive_rmse),
        "mae_price": float(np.mean(np.abs(forecast_price - actual_price))),
        "mape_price": float(np.mean(np.abs(forecast_price - actual_price) / actual_price)),
        "directional_accuracy": direction["directional_accuracy"],
        "balanced_accuracy": direction["balanced_accuracy"],
        "up_recall": direction["up_recall"],
        "down_recall": direction["down_recall"],
        "minimum_class_recall": direction["minimum_class_recall"],
        "bias_log_return": float(np.mean(error)),
    }


def select_point_model_card(
    variants: dict[float, pd.DataFrame],
    point_cfg: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select Ridge alpha by validation MAE skill and calibrate an 80% interval."""
    candidates: list[dict[str, Any]] = []
    for alpha, predictions in variants.items():
        metrics = point_forecast_metrics(predictions)
        skill = metrics.get("mae_skill_vs_spot")
        candidates.append({
            "alpha": float(alpha),
            "metrics": metrics,
            "score": float(skill) if skill is not None and np.isfinite(skill) else -np.inf,
        })
    best = max(
        candidates,
        key=lambda item: (
            item["score"],
            -(item["metrics"].get("rmse_log_return") or np.inf),
            -item["alpha"],
        ),
    )
    selected = variants[best["alpha"]].copy()
    matured = selected[selected["actual_log_return"].notna()]
    residuals = np.abs(
        matured["actual_log_return"].astype(float)
        - matured["forecast_log_return"].astype(float)
    )
    interval_level = float(point_cfg["interval_level"])
    residual_quantile = float(residuals.quantile(interval_level)) if len(residuals) else 0.0
    metrics = dict(best["metrics"])
    metrics["interval_coverage"] = (
        float((residuals <= residual_quantile).mean()) if len(residuals) else None
    )
    minimum_skill = float(point_cfg["minimum_validation_mae_skill"])
    eligible = bool(best["score"] >= minimum_skill)
    return {
        "model_family": POINT_MODEL_FAMILY,
        "alpha": _safe_float(best["alpha"], 3),
        "target": str(point_cfg["target"]),
        "selection_metric": str(point_cfg["selection_metric"]),
        "validation_eligible": eligible,
        "validation_metrics": _json_metrics(metrics),
        "interval_level": _safe_float(interval_level),
        "interval_abs_log_return": _safe_float(residual_quantile),
        "clip_quantiles": [float(value) for value in point_cfg["prediction_clip_quantiles"]],
        "candidate_count": len(candidates),
    }, selected


def robust_score(metrics: dict[str, Any], weights: dict[str, float]) -> float:
    values = (
        metrics.get("directional_accuracy"),
        metrics.get("balanced_accuracy"),
        metrics.get("minimum_class_recall"),
    )
    if any(value is None or not np.isfinite(value) for value in values):
        return float("-inf")
    return float(
        weights["directional_accuracy"] * values[0]
        + weights["balanced_accuracy"] * values[1]
        + weights["minimum_class_recall"] * values[2]
    )


def validation_candidate_is_eligible(metrics: dict[str, Any], selection: dict[str, Any]) -> bool:
    rate = metrics.get("prediction_up_rate")
    recall = metrics.get("minimum_class_recall")
    balanced = metrics.get("balanced_accuracy")
    return bool(
        balanced is not None
        and balanced >= 0.50
        and recall is not None
        and recall >= float(selection["validation_minimum_class_recall"])
        and rate is not None
        and float(selection["validation_minimum_prediction_class_rate"])
        <= rate
        <= float(selection["validation_maximum_prediction_class_rate"])
    )


def select_model_card(
    variants: dict[int | None, pd.DataFrame],
    features: list[str],
    horizon: int,
    validation_years: Iterable[int],
    thresholds: Iterable[float],
    selection: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    years = {int(year) for year in validation_years}
    weights = selection["score_weights"]
    candidates: list[dict[str, Any]] = []
    for half_life, predictions in variants.items():
        validation = predictions[predictions["origin_date"].dt.year.isin(years)]
        for threshold in thresholds:
            metrics = directional_metrics(validation, float(threshold))
            candidates.append({
                "half_life_raw": half_life,
                "half_life": "expanding" if half_life is None else str(half_life),
                "threshold": float(threshold),
                "score": robust_score(metrics, weights),
                "eligible": validation_candidate_is_eligible(metrics, selection),
                "metrics": metrics,
            })
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    pool = eligible or candidates
    best = max(
        pool,
        key=lambda item: (
            item["score"],
            -(item["metrics"].get("brier") or 1.0),
            -abs(item["threshold"] - 0.5),
        ),
    )
    selected = variants[best["half_life_raw"]].copy()
    selected["threshold"] = best["threshold"]
    selected["prediction"] = (selected["probability_up"] >= best["threshold"]).astype(int)
    feature_hash = hashlib.sha256("|".join(features).encode()).hexdigest()[:16]
    card = {
        "horizon_days": horizon,
        "model_family": MODEL_FAMILY,
        "half_life": best["half_life"],
        "threshold": _safe_float(best["threshold"], 3),
        "validation_years": sorted(years),
        "validation_score": _safe_float(best["score"]),
        "validation_eligible": bool(best["eligible"]),
        "validation_metrics": _json_metrics(best["metrics"]),
        "selected_features": features,
        "feature_hash": feature_hash,
        "candidate_count": len(candidates),
    }
    return card, selected


def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: int(value) if key in {"n", "up_count", "down_count"} and value is not None
        else _safe_float(value)
        for key, value in metrics.items()
    }


def horizon_evidence(
    history: pd.DataFrame,
    threshold: float,
    origin_date: pd.Timestamp,
    selection: dict[str, Any],
    validation_eligible: bool,
) -> dict[str, Any]:
    matured = history[
        history["actual"].notna()
        & (history["target_date"] <= origin_date)
    ].sort_values("origin_date")
    lookback = int(selection["evidence_lookback_weeks"])
    matured = matured.tail(lookback)
    metrics = directional_metrics(matured, threshold)
    score = robust_score(metrics, selection["score_weights"])
    n = int(metrics["n"])
    shrinkage = int(selection["shrinkage_weeks"])
    shrunk_score = (
        0.5 + (score - 0.5) * n / (n + shrinkage)
        if np.isfinite(score) else float("-inf")
    )
    rate = metrics.get("prediction_up_rate")
    recall = metrics.get("minimum_class_recall")
    balanced = metrics.get("balanced_accuracy")
    eligible = bool(
        validation_eligible
        and n >= int(selection["minimum_matured_weeks"])
        and metrics["up_count"] >= int(selection["minimum_class_observations"])
        and metrics["down_count"] >= int(selection["minimum_class_observations"])
        and balanced is not None
        and balanced >= float(selection["minimum_balanced_accuracy"])
        and recall is not None
        and recall >= float(selection["minimum_class_recall"])
        and rate is not None
        and float(selection["minimum_prediction_class_rate"])
        <= rate
        <= float(selection["maximum_prediction_class_rate"])
    )
    return {
        **_json_metrics(metrics),
        "raw_score": _safe_float(score),
        "shrunk_score": _safe_float(shrunk_score),
        "eligible": eligible,
    }


def _price_regime(frame: pd.DataFrame, origin_date: pd.Timestamp) -> dict[str, Any]:
    history = frame[frame["date"] <= origin_date].copy()
    returns = np.log(history["close"] / history["close"].shift(1)).dropna()
    if len(returns) < 120:
        return {"state": "transition", "direction_shift_z": 0.0}
    recent = returns.tail(20)
    reference = returns.tail(120).head(100)
    recent_down = float((recent < 0).mean())
    reference_down = float((reference < 0).mean())
    standard_error = np.sqrt(max(reference_down * (1 - reference_down) / len(recent), 1e-6))
    z_value = (recent_down - reference_down) / standard_error
    if z_value > 1.5 and float(recent.mean()) < float(reference.mean()):
        state = "risk_off"
    elif z_value < -1.5 and float(recent.mean()) > float(reference.mean()):
        state = "risk_on"
    else:
        state = "transition"
    return {"state": state, "direction_shift_z": _safe_float(z_value)}


def _horizon_role(horizon: int, roles: dict[str, list[int]]) -> str:
    for role, values in roles.items():
        if horizon in [int(value) for value in values]:
            return role
    return "unassigned"


def build_week_contract(
    *,
    frame: pd.DataFrame,
    week_rows: pd.DataFrame,
    histories: dict[int, pd.DataFrame],
    model_cards: dict[int, dict[str, Any]],
    cfg: dict[str, Any],
    latest_week: str,
) -> dict[str, Any]:
    origin_date = pd.Timestamp(week_rows["origin_date"].iloc[0])
    iso_week = str(week_rows["iso_week"].iloc[0])
    year = int(origin_date.year)
    roles = cfg["horizons"]["roles"]
    selection = cfg["selection"]
    freeze_evidence = year == int(cfg["protocol"]["frozen_replay_year"])
    evidence_date = pd.Timestamp(f"{year}-01-01") if freeze_evidence else origin_date

    horizon_outputs: list[dict[str, Any]] = []
    for _, row in week_rows.sort_values("horizon_days").iterrows():
        horizon = int(row["horizon_days"])
        card = model_cards[horizon]
        point_card = card["point_forecast"]
        threshold = float(card["threshold"])
        evidence = horizon_evidence(
            histories[horizon], threshold, evidence_date, selection,
            bool(card["validation_eligible"]),
        )
        prediction = "UP" if float(row["probability_up"]) >= threshold else "DOWN"
        actual = None if pd.isna(row["actual"]) else int(row["actual"])
        role = _horizon_role(horizon, roles)
        direction_eligible = bool(evidence["eligible"] and role != "execution")
        forecast_log_return = float(row["forecast_log_return"])
        forecast_price = float(row["forecast_price"])
        interval_width = float(point_card["interval_abs_log_return"])
        interval_lower = float(row["base_price"] * np.exp(forecast_log_return - interval_width))
        interval_upper = float(row["base_price"] * np.exp(forecast_log_return + interval_width))
        point_direction = str(row["point_forecast_direction"])
        actual_log_return = (
            None if pd.isna(row["actual_log_return"]) else float(row["actual_log_return"])
        )
        actual_price = None if pd.isna(row["actual_price"]) else float(row["actual_price"])
        horizon_outputs.append({
            "horizon_days": horizon,
            "role": role,
            "target_date": pd.Timestamp(row["target_date"]).date().isoformat(),
            "target_date_estimated": bool(row["target_date_estimated"]),
            "probability_up": _safe_float(row["probability_up"]),
            "threshold": _safe_float(threshold, 3),
            "prediction": prediction,
            "forecast_log_return": _safe_float(forecast_log_return),
            "forecast_return_pct": _safe_float(row["forecast_return_pct"], 4),
            "forecast_price": _safe_float(forecast_price, 2),
            "forecast_price_change": _safe_float(row["forecast_price_change"], 2),
            "forecast_interval_lower": _safe_float(interval_lower, 2),
            "forecast_interval_upper": _safe_float(interval_upper, 2),
            "forecast_interval_level": _safe_float(point_card["interval_level"]),
            "point_forecast_direction": point_direction,
            "direction_price_agree": bool(point_direction == prediction),
            "point_forecast_clipped": bool(row["point_forecast_clipped"]),
            "point_forecast_validation_eligible": bool(point_card["validation_eligible"]),
            "actual": actual,
            "actual_log_return": _safe_float(actual_log_return),
            "actual_price": _safe_float(actual_price, 2),
            "point_abs_error_price": _safe_float(row["point_abs_error_price"], 2),
            "point_abs_error_pct": _safe_float(row["point_abs_error_pct"], 4),
            "hit": None if actual is None else bool(actual == (1 if prediction == "UP" else 0)),
            "train_count": int(row["train_count"]),
            "train_label_end": (
                pd.Timestamp(row["train_label_end"]).date().isoformat()
                if pd.notna(row["train_label_end"]) else None
            ),
            "point_train_label_end": (
                pd.Timestamp(row["point_train_label_end"]).date().isoformat()
                if pd.notna(row["point_train_label_end"]) else None
            ),
            "training_mode": str(row["training_mode"]),
            "model_family": MODEL_FAMILY,
            "feature_hash": card["feature_hash"],
            "validation_eligible": bool(card["validation_eligible"]),
            "evidence": evidence,
            "eligible_for_direction": direction_eligible,
            "selected": False,
            "selection_rank": None,
        })

    ranked = sorted(
        horizon_outputs,
        key=lambda item: item["evidence"].get("shrunk_score")
        if item["evidence"].get("shrunk_score") is not None else -1.0,
        reverse=True,
    )
    for rank, item in enumerate(ranked, start=1):
        item["selection_rank"] = rank

    selected: list[dict[str, Any]] = []
    for sleeve in ("tactical", "swing"):
        candidates = [
            item for item in ranked
            if item["role"] == sleeve
        ]
        if candidates:
            candidates[0]["selected"] = True
            selected.append(candidates[0])

    required = int(selection["required_confirming_sleeves"])
    agreement = len(selected) >= required and len({item["prediction"] for item in selected}) == 1
    promotion_gate_passed = bool(
        agreement and all(item["eligible_for_direction"] for item in selected)
    )
    direction = selected[0]["prediction"] if agreement else "FLAT"
    primary = max(
        selected,
        key=lambda item: item["evidence"].get("shrunk_score") or -1.0,
        default=None,
    )
    confirmation = [item["horizon_days"] for item in selected if item is not primary]
    margins = [
        min(1.0, abs(float(item["probability_up"]) - float(item["threshold"])) / 0.20)
        for item in selected
    ]
    confidence = float(np.mean(margins)) if agreement and margins else 0.0
    action = {
        "UP": "LONG_USD_SHORT_COP",
        "DOWN": "SHORT_USD_LONG_COP",
        "FLAT": "FLAT",
    }[direction]
    primary_actual = primary["actual"] if primary is not None else None
    decision_hit = (
        None if direction == "FLAT" or primary_actual is None
        else bool(primary_actual == (1 if direction == "UP" else 0))
    )
    if not selected:
        rationale = "No horizon candidate was available in both sleeves."
    elif len(selected) < required:
        rationale = "Only one horizon sleeve passed; confirmation is required."
    elif not agreement:
        rationale = "Tactical and swing horizons disagree; the policy abstains."
    elif not promotion_gate_passed:
        rationale = "Best causal candidates agree, but one or more promotion gates remain unmet."
    else:
        rationale = "Tactical and swing horizons passed causal gates and agree."

    latest_partial = iso_week == latest_week and origin_date.dayofweek < 4
    image_name = cfg["outputs"]["image_template"].format(week=iso_week.replace("-", "_"))
    return {
        "iso_week": iso_week,
        "year": year,
        "origin_date": origin_date.date().isoformat(),
        "origin_is_partial_week": bool(latest_partial),
        "base_price": _safe_float(week_rows["base_price"].iloc[0], 2),
        "training_mode": str(week_rows["training_mode"].iloc[0]),
        "regime": _price_regime(frame, origin_date),
        "decision": {
            "direction": direction,
            "action": action,
            "status": "SHADOW" if direction != "FLAT" else "ABSTAIN",
            "signal_authorized": False,
            "promotion_gate_passed": promotion_gate_passed,
            "primary_horizon": None if primary is None else primary["horizon_days"],
            "confirmation_horizons": confirmation,
            "selected_horizons": [item["horizon_days"] for item in selected],
            "confidence_proxy": _safe_float(confidence),
            "target_date": None if primary is None else primary["target_date"],
            "forecast_price": None if primary is None else primary["forecast_price"],
            "forecast_return_pct": None if primary is None else primary["forecast_return_pct"],
            "forecast_interval_lower": (
                None if primary is None else primary["forecast_interval_lower"]
            ),
            "forecast_interval_upper": (
                None if primary is None else primary["forecast_interval_upper"]
            ),
            "actual": primary_actual,
            "hit": decision_hit,
            "rationale": rationale,
        },
        "horizons": sorted(horizon_outputs, key=lambda item: item["horizon_days"]),
        "image_path": f"usdcop/{image_name}",
    }


def _summary_for_records(records: list[dict[str, Any]], year: int) -> dict[str, Any]:
    weeks = [record for record in records if record["year"] == year]
    decisions = [record for record in weeks if record["decision"]["direction"] != "FLAT"]
    matured = [record for record in decisions if record["decision"]["actual"] is not None]
    decision_frame = pd.DataFrame([
        {
            "probability_up": 1.0 if record["decision"]["direction"] == "UP" else 0.0,
            "actual": record["decision"]["actual"],
        }
        for record in matured
    ])
    decision_metrics = directional_metrics(decision_frame, 0.5) if matured else directional_metrics(pd.DataFrame(columns=["probability_up", "actual"]), 0.5)
    horizon_metrics = []
    for horizon in (1, 5, 10, 15, 20, 25, 30):
        rows = []
        point_rows = []
        threshold = None
        for record in weeks:
            item = next(value for value in record["horizons"] if value["horizon_days"] == horizon)
            threshold = float(item["threshold"])
            rows.append({"probability_up": item["probability_up"], "actual": item["actual"]})
            point_rows.append({
                "forecast_log_return": item["forecast_log_return"],
                "actual_log_return": item["actual_log_return"],
                "forecast_price": item["forecast_price"],
                "actual_price": item["actual_price"],
            })
        metrics = directional_metrics(pd.DataFrame(rows), float(threshold))
        point_metrics = point_forecast_metrics(pd.DataFrame(point_rows))
        horizon_metrics.append({
            "horizon_days": horizon,
            **_json_metrics(metrics),
            "point_forecast": _json_metrics(point_metrics),
        })
    return {
        "year": year,
        "weeks_total": len(weeks),
        "shadow_decisions": len(decisions),
        "abstentions": len(weeks) - len(decisions),
        "coverage": _safe_float(len(decisions) / len(weeks) if weeks else None),
        "matured_decisions": len(matured),
        "pending_decisions": len(decisions) - len(matured),
        "decision_metrics": _json_metrics(decision_metrics),
        "horizon_metrics": horizon_metrics,
    }


def build_directional_replay(root: Path, cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build the complete 2025 frozen + 2026 expanding replay document."""
    protocol = cfg["protocol"]
    model_cfg = protocol["model"]
    point_cfg = protocol["point_forecast"]
    horizons = [int(value) for value in cfg["horizons"]["values"]]
    validation_years = [int(value) for value in protocol["validation_years"]]
    validation_start = f"{min(validation_years)}-01-01"
    validation_end = f"{max(validation_years)}-12-31"
    half_lives = [None if value is None else int(value) for value in protocol["half_lives_days"]]
    thresholds = [float(value) for value in protocol["probability_thresholds"]]
    point_alphas = [float(value) for value in point_cfg["alphas"]]
    clip_quantiles = tuple(float(value) for value in point_cfg["prediction_clip_quantiles"])
    if len(clip_quantiles) != 2 or not 0 <= clip_quantiles[0] < clip_quantiles[1] <= 1:
        raise ValueError(f"Invalid point forecast clip quantiles: {clip_quantiles}")

    frame, _, candidates = build_frame(include_forward_pit=True, promotion_only=False)
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"])

    model_cards: dict[int, dict[str, Any]] = {}
    validation_history: dict[int, pd.DataFrame] = {}
    replay_by_horizon: dict[int, pd.DataFrame] = {}
    for horizon in horizons:
        features = select_frozen_features(
            frame, candidates, horizon,
            str(protocol["feature_selection_cutoff"]),
            int(protocol["feature_count"]),
        )
        variants = {
            half_life: walk_forward_probabilities(
                frame, features, horizon, half_life,
                validation_start, validation_end,
                float(model_cfg["c"]), int(model_cfg["max_iter"]),
            )
            for half_life in half_lives
        }
        card, selected_validation = select_model_card(
            variants, features, horizon, validation_years, thresholds, cfg["selection"]
        )
        selected_half_life = None if card["half_life"] == "expanding" else int(card["half_life"])
        point_variants = {
            alpha: walk_forward_point_forecasts(
                frame, features, horizon, selected_half_life,
                validation_start, validation_end, alpha, clip_quantiles,
            )
            for alpha in point_alphas
        }
        point_card, _ = select_point_model_card(point_variants, point_cfg)
        selected_alpha = float(point_card["alpha"])
        frozen = frozen_year_probabilities(
            frame, features, horizon, selected_half_life,
            int(protocol["frozen_replay_year"]),
            float(model_cfg["c"]), int(model_cfg["max_iter"]),
        )
        expanding = expanding_year_probabilities(
            frame, features, horizon, selected_half_life,
            int(protocol["expanding_retrain_year"]),
            float(model_cfg["c"]), int(model_cfg["max_iter"]),
        )
        replay = pd.concat([frozen, expanding], ignore_index=True)
        point_frozen = frozen_year_point_forecasts(
            frame, features, horizon, selected_half_life,
            int(protocol["frozen_replay_year"]), selected_alpha, clip_quantiles,
        )
        point_expanding = expanding_year_point_forecasts(
            frame, features, horizon, selected_half_life,
            int(protocol["expanding_retrain_year"]), selected_alpha, clip_quantiles,
        )
        point_replay = pd.concat([point_frozen, point_expanding], ignore_index=True)
        point_columns = [
            column for column in point_replay.columns
            if column not in {"iso_week"}
        ]
        replay = replay.merge(
            point_replay[point_columns],
            on=["origin_date", "horizon_days"],
            how="left",
            validate="one_to_one",
        )
        replay["threshold"] = float(card["threshold"])
        replay["prediction"] = (replay["probability_up"] >= float(card["threshold"])).astype(int)
        card["point_forecast"] = point_card
        model_cards[horizon] = card
        validation_history[horizon] = selected_validation
        replay_by_horizon[horizon] = replay

    replay_frame = pd.concat(replay_by_horizon.values(), ignore_index=True)
    latest_week = str(replay_frame.sort_values("origin_date")["iso_week"].iloc[-1])
    histories = {
        horizon: pd.concat(
            [validation_history[horizon], replay_by_horizon[horizon]],
            ignore_index=True,
        ).sort_values("origin_date")
        for horizon in horizons
    }
    records = [
        build_week_contract(
            frame=frame,
            week_rows=week_rows,
            histories=histories,
            model_cards=model_cards,
            cfg=cfg,
            latest_week=latest_week,
        )
        for _, week_rows in replay_frame.groupby("iso_week", sort=True)
    ]

    config_path = root / "config/forecast_experiments/usdcop_directional_macro_replay_v1.yaml"
    lineage_paths = {
        "ohlcv": root / "seeds/latest/usdcop_daily_ohlcv.parquet",
        "classic_macro": root / "data/pipeline/04_cleaning/output/MACRO_DAILY_CLEAN.parquet",
        "forward_macro_pit": root / "data/pipeline/04_cleaning/output/USDCOP_FORWARD_MACRO_PIT.parquet",
    }
    contract_basis = {
        "config_sha256": _sha256_file(config_path),
        "models": model_cards,
        "data_cutoff": frame["date"].max().date().isoformat(),
    }
    contract_hash = hashlib.sha256(
        json.dumps(contract_basis, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    replay_years = [int(value) for value in protocol["replay_years"]]
    document = {
        "schema_version": SCHEMA_VERSION,
        "contract_hash": contract_hash,
        "asset_id": cfg["asset"]["id"],
        "symbol": cfg["asset"]["symbol"],
        "chart_symbol": cfg["asset"]["chart_symbol"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_cutoff": frame["date"].max().date().isoformat(),
        "latest_week": latest_week,
        "years": replay_years,
        "methodology": {
            "experiment_id": cfg["_meta"]["experiment_id"],
            "status": cfg["_meta"]["status"],
            "signal_authorized": False,
            "feature_selection_cutoff": str(protocol["feature_selection_cutoff"]),
            "validation_years": validation_years,
            "frozen_replay_year": int(protocol["frozen_replay_year"]),
            "expanding_retrain_year": int(protocol["expanding_retrain_year"]),
            "label_maturity_rule": str(protocol["label_maturity_rule"]),
            "horizon_selection": "matured_only_two_sleeve_consensus",
        },
        "lineage": {
            key: {"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": _sha256_file(path)}
            for key, path in lineage_paths.items()
        },
        "models": {str(horizon): model_cards[horizon] for horizon in horizons},
        "summaries": [_summary_for_records(records, year) for year in replay_years],
        "weeks": records,
    }
    return document, replay_frame


def flatten_ledger(document: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for week in document["weeks"]:
        decision = week["decision"]
        for horizon in week["horizons"]:
            evidence = horizon["evidence"]
            rows.append({
                "iso_week": week["iso_week"],
                "origin_date": week["origin_date"],
                "origin_is_partial_week": week["origin_is_partial_week"],
                "base_price": week["base_price"],
                "regime": week["regime"]["state"],
                "training_mode": horizon["training_mode"],
                "horizon_days": horizon["horizon_days"],
                "role": horizon["role"],
                "target_date": horizon["target_date"],
                "target_date_estimated": horizon["target_date_estimated"],
                "probability_up": horizon["probability_up"],
                "threshold": horizon["threshold"],
                "prediction": horizon["prediction"],
                "forecast_log_return": horizon["forecast_log_return"],
                "forecast_return_pct": horizon["forecast_return_pct"],
                "forecast_price": horizon["forecast_price"],
                "forecast_price_change": horizon["forecast_price_change"],
                "forecast_interval_lower": horizon["forecast_interval_lower"],
                "forecast_interval_upper": horizon["forecast_interval_upper"],
                "forecast_interval_level": horizon["forecast_interval_level"],
                "point_forecast_direction": horizon["point_forecast_direction"],
                "direction_price_agree": horizon["direction_price_agree"],
                "point_forecast_clipped": horizon["point_forecast_clipped"],
                "point_forecast_validation_eligible": horizon["point_forecast_validation_eligible"],
                "actual": horizon["actual"],
                "actual_log_return": horizon["actual_log_return"],
                "actual_price": horizon["actual_price"],
                "point_abs_error_price": horizon["point_abs_error_price"],
                "point_abs_error_pct": horizon["point_abs_error_pct"],
                "hit": horizon["hit"],
                "train_count": horizon["train_count"],
                "train_label_end": horizon["train_label_end"],
                "point_train_label_end": horizon["point_train_label_end"],
                "feature_hash": horizon["feature_hash"],
                "validation_eligible": horizon["validation_eligible"],
                "evidence_n": evidence["n"],
                "evidence_da": evidence["directional_accuracy"],
                "evidence_balanced_da": evidence["balanced_accuracy"],
                "evidence_min_recall": evidence["minimum_class_recall"],
                "evidence_brier": evidence["brier"],
                "evidence_score": evidence["shrunk_score"],
                "eligible_for_direction": horizon["eligible_for_direction"],
                "selected": horizon["selected"],
                "selection_rank": horizon["selection_rank"],
                "decision_direction": decision["direction"],
                "decision_action": decision["action"],
                "decision_status": decision["status"],
                "promotion_gate_passed": decision["promotion_gate_passed"],
                "primary_horizon": decision["primary_horizon"],
                "selected_horizons": "|".join(map(str, decision["selected_horizons"])),
                "decision_confidence_proxy": decision["confidence_proxy"],
                "decision_forecast_price": decision["forecast_price"],
                "decision_forecast_return_pct": decision["forecast_return_pct"],
                "decision_forecast_interval_lower": decision["forecast_interval_lower"],
                "decision_forecast_interval_upper": decision["forecast_interval_upper"],
                "decision_actual": decision["actual"],
                "decision_hit": decision["hit"],
                "image_path": week["image_path"],
            })
    return pd.DataFrame(rows)


def summary_frame(document: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for summary in document["summaries"]:
        decision = summary["decision_metrics"]
        rows.append({
            "year": summary["year"],
            "scope": "selected_strategy",
            "horizon_days": None,
            "n_weeks": summary["weeks_total"],
            "n_matured": summary["matured_decisions"],
            "coverage": summary["coverage"],
            "directional_accuracy": decision["directional_accuracy"],
            "balanced_accuracy": decision["balanced_accuracy"],
            "up_recall": decision["up_recall"],
            "down_recall": decision["down_recall"],
            "brier": decision["brier"],
        })
        for metrics in summary["horizon_metrics"]:
            point = metrics.get("point_forecast", {})
            rows.append({
                "year": summary["year"],
                "scope": "horizon",
                "horizon_days": metrics["horizon_days"],
                "n_weeks": summary["weeks_total"],
                "n_matured": metrics["n"],
                "coverage": None,
                "directional_accuracy": metrics["directional_accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "up_recall": metrics["up_recall"],
                "down_recall": metrics["down_recall"],
                "brier": metrics["brier"],
                "point_mae_log_return": point.get("mae_log_return"),
                "point_rmse_log_return": point.get("rmse_log_return"),
                "point_mae_skill_vs_spot": point.get("mae_skill_vs_spot"),
                "point_rmse_skill_vs_spot": point.get("rmse_skill_vs_spot"),
                "point_mae_price": point.get("mae_price"),
                "point_mape_price": point.get("mape_price"),
                "point_directional_accuracy": point.get("directional_accuracy"),
                "point_balanced_accuracy": point.get("balanced_accuracy"),
            })
    return pd.DataFrame(rows)


def validate_replay_document(
    document: dict[str, Any],
    root: Path,
    *,
    require_images: bool = False,
) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if document.get("asset_id") != "usdcop":
        errors.append("asset_id must be usdcop")
    weeks = document.get("weeks") or []
    labels = [week.get("iso_week") for week in weeks]
    if len(labels) != len(set(labels)):
        errors.append("duplicate iso_week records")
    if labels != sorted(labels):
        errors.append("weeks are not sorted")
    expected_horizons = [1, 5, 10, 15, 20, 25, 30]
    for week in weeks:
        origin = pd.Timestamp(week["origin_date"])
        horizons = week.get("horizons") or []
        actual_horizons = [int(item["horizon_days"]) for item in horizons]
        if actual_horizons != expected_horizons:
            errors.append(f"{week['iso_week']}: invalid horizons {actual_horizons}")
        selected = [item for item in horizons if item.get("selected")]
        decision = week["decision"]
        if sorted(item["horizon_days"] for item in selected) != sorted(decision["selected_horizons"]):
            errors.append(f"{week['iso_week']}: selected horizon mismatch")
        if decision["direction"] != "FLAT":
            if len(selected) < 2 or len({item["prediction"] for item in selected}) != 1:
                errors.append(f"{week['iso_week']}: directional decision lacks two-sleeve agreement")
        for item in horizons:
            label_end = item.get("train_label_end")
            point_label_end = item.get("point_train_label_end")
            if label_end and pd.Timestamp(label_end) > origin:
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: immature training label")
            if point_label_end and pd.Timestamp(point_label_end) > origin:
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: immature point label")
            if week["year"] == 2025 and label_end and pd.Timestamp(label_end) >= pd.Timestamp("2025-01-01"):
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: frozen replay used 2025 label")
            if (
                week["year"] == 2025
                and point_label_end
                and pd.Timestamp(point_label_end) >= pd.Timestamp("2025-01-01")
            ):
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: point replay used 2025 label")
            if item["actual"] is None and item["hit"] is not None:
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: hit without actual")
            price = item.get("forecast_price")
            log_return = item.get("forecast_log_return")
            lower = item.get("forecast_interval_lower")
            upper = item.get("forecast_interval_upper")
            if any(value is None for value in (price, log_return, lower, upper)):
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: point forecast missing")
            else:
                implied = float(week["base_price"]) * np.exp(float(log_return))
                if abs(float(price) - implied) > 0.02:
                    errors.append(f"{week['iso_week']} H{item['horizon_days']}: inconsistent point price")
                if not (0 < float(lower) <= float(price) <= float(upper)):
                    errors.append(f"{week['iso_week']} H{item['horizon_days']}: invalid price interval")
            if item["actual"] is None and (
                item.get("actual_log_return") is not None or item.get("actual_price") is not None
            ):
                errors.append(f"{week['iso_week']} H{item['horizon_days']}: unresolved point has actual")
        if require_images:
            image = root / "usdcop-trading-dashboard/public/forecasting" / week["image_path"]
            if not image.exists() or image.stat().st_size == 0:
                errors.append(f"{week['iso_week']}: image missing")
    expected_years = set(document.get("years") or [])
    actual_years = {int(week["year"]) for week in weeks}
    if expected_years != actual_years:
        errors.append(f"year coverage mismatch: expected={expected_years}, actual={actual_years}")
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        errors.append(f"document is not strict JSON: {exc}")
    return errors


__all__ = [
    "SCHEMA_VERSION",
    "ReplayPaths",
    "build_directional_replay",
    "directional_metrics",
    "point_forecast_metrics",
    "flatten_ledger",
    "load_replay_config",
    "resolve_paths",
    "summary_frame",
    "validate_replay_document",
]
