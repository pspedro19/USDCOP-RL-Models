"""Single governed metric engine backed by ``config/metrics/catalog.yaml`` (BL-18)."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import yaml

from src.identity.canonical import CanonicalizationError, canonical_json_bytes
from src.metrics.annualization import AnnualizationRegistry
from src.metrics.errors import MetricContractError

from src.metrics.formulas import (
    calmar_ratio,
    deflated_sharpe_probability,
    max_drawdown,
    sharpe_ratio,
)


class MetricEnvironment(StrEnum):
    BACKTEST = "backtest"
    HELD_OUT = "held_out"
    PAPER = "paper"
    CANARY = "canary"
    LIVE = "live"


MetricFormula = Callable[[Mapping[str, Any], float], float | None]


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    namespace: str
    name: str
    formula_version: str
    source: str
    unit: str
    annualization: str | int | float | None = None
    windows: tuple[str, ...] = ()
    warning: float | None = None
    critical: float | None = None
    required_inputs: tuple[str, ...] = ()
    higher_is_better: bool = True
    min_trades: int | None = None
    min_obs: int | None = None
    trials_sharpe_std_min: float | None = None
    plausible_min: float | None = None
    plausible_max: float | None = None

    @property
    def metric_id(self) -> str:
        return f"{self.namespace}.{self.name}"


@dataclass(frozen=True, slots=True)
class MetricEvent:
    metric_event_id: str
    event_time: str
    catalog_version: str
    formula_version: str
    entity_type: str
    entity_id: str
    metric_namespace: str
    metric_name: str
    metric_value: float | None
    metric_unit: str
    status: str
    threshold_warning: float | None
    threshold_critical: float | None
    strategy_id: str | None = None
    asset_id: str | None = None
    run_id: str | None = None
    environment: str | None = None
    dimensions: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


class MetricCatalog:
    def __init__(self, version: str, definitions: Mapping[str, MetricDefinition]) -> None:
        self.version = version
        self._definitions = dict(definitions)

    @classmethod
    def load(cls, path: str | Path) -> "MetricCatalog":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        version = str(raw.get("catalog_version") or "")
        if not version:
            raise MetricContractError("catalog_version is required")
        metrics = raw.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise MetricContractError("metrics must be a non-empty mapping")
        definitions: dict[str, MetricDefinition] = {}
        for metric_id, body in metrics.items():
            if not isinstance(metric_id, str) or metric_id.count(".") != 1:
                raise MetricContractError(f"invalid metric id: {metric_id!r}")
            if not isinstance(body, dict):
                raise MetricContractError(f"{metric_id}: definition must be an object")
            namespace, name = metric_id.split(".", 1)
            higher_is_better = body.get("higher_is_better", True)
            if type(higher_is_better) is not bool:
                raise MetricContractError(f"{metric_id}: higher_is_better must be boolean")
            for threshold_name in (
                "warning",
                "critical",
                "plausible_min",
                "plausible_max",
            ):
                threshold = body.get(threshold_name)
                if (
                    threshold is not None
                    and (
                        isinstance(threshold, bool)
                        or not isinstance(threshold, (int, float))
                        or not math.isfinite(float(threshold))
                    )
                ):
                    raise MetricContractError(
                        f"{metric_id}: {threshold_name} must be finite numeric or null"
                    )
            min_trades = body.get("min_trades")
            if min_trades is not None and (
                type(min_trades) is not int or min_trades <= 0
            ):
                raise MetricContractError(
                    f"{metric_id}: min_trades must be a positive integer or null"
                )
            min_obs = body.get("min_obs")
            if min_obs is not None and (
                type(min_obs) is not int or min_obs <= 0
            ):
                raise MetricContractError(
                    f"{metric_id}: min_obs must be a positive integer or null"
                )
            std_floor = body.get("trials_sharpe_std_min")
            if std_floor is not None and (
                isinstance(std_floor, bool)
                or not isinstance(std_floor, (int, float))
                or not math.isfinite(float(std_floor))
                or float(std_floor) <= 0
            ):
                raise MetricContractError(
                    f"{metric_id}: trials_sharpe_std_min must be positive finite"
                )
            definition = MetricDefinition(
                namespace=namespace,
                name=name,
                formula_version=str(body.get("formula_version") or ""),
                source=str(body.get("source") or ""),
                unit=str(body.get("unit") or ""),
                annualization=body.get("annualization"),
                windows=tuple(body.get("windows") or ()),
                warning=body.get("warning"),
                critical=body.get("critical"),
                required_inputs=tuple(body.get("required_inputs") or ()),
                higher_is_better=higher_is_better,
                min_trades=min_trades,
                min_obs=min_obs,
                trials_sharpe_std_min=(
                    None if std_floor is None else float(std_floor)
                ),
                plausible_min=body.get("plausible_min"),
                plausible_max=body.get("plausible_max"),
            )
            if not definition.formula_version or not definition.source or not definition.unit:
                raise MetricContractError(
                    f"{metric_id}: formula_version, source and unit are required"
                )
            if (
                definition.plausible_min is not None
                and definition.plausible_max is not None
                and definition.plausible_min >= definition.plausible_max
            ):
                raise MetricContractError(
                    f"{metric_id}: plausible_min must be below plausible_max"
                )
            definitions[metric_id] = definition
        return cls(version, definitions)

    def get(self, metric_id: str) -> MetricDefinition:
        try:
            return self._definitions[metric_id]
        except KeyError as exc:
            raise MetricContractError(f"metric {metric_id!r} is not catalogued") from exc

    def __contains__(self, metric_id: str) -> bool:
        return metric_id in self._definitions

    @property
    def metric_ids(self) -> frozenset[str]:
        return frozenset(self._definitions)


def _finite_array(context: Mapping[str, Any], name: str) -> np.ndarray:
    if name not in context:
        raise MetricContractError(f"missing metric input {name!r}")
    try:
        values = np.asarray(context[name], dtype=float)
    except (TypeError, ValueError) as exc:
        raise MetricContractError(f"{name} must contain numeric values") from exc
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise MetricContractError(f"{name} must be a non-empty finite 1-D array")
    return values


def _sharpe(context: Mapping[str, Any], annualization: float) -> float | None:
    return sharpe_ratio(
        _finite_array(context, "returns"), periods_per_year=int(annualization)
    )


def _calmar(context: Mapping[str, Any], annualization: float) -> float | None:
    return calmar_ratio(
        _finite_array(context, "returns"), periods_per_year=int(annualization)
    )


def _max_drawdown(context: Mapping[str, Any], annualization: float) -> float | None:
    del annualization
    equity = _finite_array(context, "equity")
    if np.any(equity <= 0):
        raise MetricContractError("equity must be strictly positive")
    return float(max_drawdown(equity))


def _dsr(context: Mapping[str, Any], annualization: float) -> float | None:
    del annualization
    returns = _finite_array(context, "returns")
    if len(returns) < 3 or float(np.std(returns, ddof=1)) == 0:
        raise MetricContractError("DSR requires at least 3 non-constant returns")
    n_trials = context.get("n_trials")
    if type(n_trials) is not int or n_trials < 1:
        raise MetricContractError("n_trials must be a positive integer")
    trials_sharpe_std = context.get("trials_sharpe_std")
    if (
        isinstance(trials_sharpe_std, bool)
        or not isinstance(trials_sharpe_std, (int, float))
        or not math.isfinite(float(trials_sharpe_std))
        or float(trials_sharpe_std) <= 0
    ):
        raise MetricContractError("trials_sharpe_std must be a finite positive number")
    centered = returns - float(np.mean(returns))
    population_std = float(np.std(returns, ddof=0))
    if population_std <= np.finfo(float).eps:
        raise MetricContractError("DSR requires non-constant returns")
    moments = {
        "skew": float(np.mean(centered**3) / population_std**3),
        "kurtosis": float(np.mean(centered**4) / population_std**4),
    }
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1))
    return deflated_sharpe_probability(
        sharpe_per_period=sharpe,
        n_trials=n_trials,
        n_obs=len(returns),
        trials_sharpe_std=float(trials_sharpe_std),
        skew=moments["skew"],
        kurtosis=moments["kurtosis"],
    )


def _timing_ratio(context: Mapping[str, Any], annualization: float) -> float | None:
    del annualization
    weights = _finite_array(context, "weights")
    returns = _finite_array(context, "asset_returns")
    if len(weights) != len(returns):
        raise MetricContractError("weights and asset_returns must have equal length")
    if len(weights) < 2:
        raise MetricContractError("timing_ratio requires at least two observations")
    denominator = float(np.var(returns, ddof=1))
    if denominator <= 0:
        return None
    beta = float(np.cov(weights, returns, ddof=1)[0, 1] / denominator)
    timing_pnl = weights * returns - beta * returns
    scale = float(np.std(returns, ddof=1))
    return None if scale == 0 else float(np.mean(timing_pnl) / scale)


FORMULAS: dict[str, MetricFormula] = {
    "strategy.sharpe": _sharpe,
    "strategy.calmar": _calmar,
    "strategy.max_drawdown": _max_drawdown,
    "research.dsr": _dsr,
    "strategy.timing_ratio": _timing_ratio,
}


class MetricEngine:
    """The only application entry point for governed metric calculations."""

    def __init__(
        self,
        catalog: MetricCatalog,
        *,
        formulas: Mapping[str, MetricFormula] | None = None,
        annualization_registry: AnnualizationRegistry | None = None,
    ) -> None:
        self.catalog = catalog
        self.formulas = dict(FORMULAS if formulas is None else formulas)
        self.annualization_registry = annualization_registry
        missing_formulas = sorted(catalog.metric_ids - set(self.formulas))
        if missing_formulas:
            raise MetricContractError(
                f"catalogued metrics have no registered formula: {missing_formulas}"
            )

    @classmethod
    def from_asset_registry(
        cls,
        catalog: MetricCatalog,
        *,
        assets_dir: str | Path,
        formulas: Mapping[str, MetricFormula] | None = None,
    ) -> "MetricEngine":
        """Build the application engine from the AssetProfile SSOT."""
        return cls(
            catalog,
            formulas=formulas,
            annualization_registry=AnnualizationRegistry.load(assets_dir),
        )

    def compute(
        self,
        *,
        entity_type: str,
        entity_id: str,
        metric: str,
        window: str,
        env: str,
        as_of: datetime,
        context: Mapping[str, Any],
        strategy_id: str | None = None,
        asset_id: str | None = None,
        run_id: str | None = None,
        lineage: Mapping[str, Any] | None = None,
    ) -> MetricEvent:
        try:
            canonical_environment = MetricEnvironment(env).value
        except (TypeError, ValueError) as exc:
            raise MetricContractError(
                f"invalid metric environment {env!r}"
            ) from exc
        definition = self.catalog.get(metric)
        if definition.windows and window not in definition.windows:
            raise MetricContractError(
                f"{metric}: window {window!r} not in {definition.windows}"
            )
        missing = sorted(set(definition.required_inputs) - set(context))
        if missing:
            raise MetricContractError(f"{metric}: missing inputs {missing}")
        try:
            formula = self.formulas[metric]
        except KeyError as exc:
            raise MetricContractError(f"{metric}: no registered formula") from exc
        self._validate_window(window=window, as_of=as_of, context=context)
        annualization = self._annualization(definition, asset_id, context)
        observation_count = self._observation_count(context)
        n_trades = context.get("n_trades")
        insufficient_sample = False
        if definition.min_trades is not None:
            if type(n_trades) is not int or n_trades < 0:
                raise MetricContractError(
                    f"{metric}: n_trades must be a non-negative integer"
                )
            insufficient_sample = n_trades < definition.min_trades
        if definition.min_obs is not None:
            if observation_count is None:
                raise MetricContractError(
                    f"{metric}: no observation series available for min_obs"
                )
            insufficient_sample = (
                insufficient_sample
                or observation_count < definition.min_obs
            )
        if definition.trials_sharpe_std_min is not None:
            trial_std = context.get("trials_sharpe_std")
            if (
                isinstance(trial_std, bool)
                or not isinstance(trial_std, (int, float))
                or not math.isfinite(float(trial_std))
                or float(trial_std) < definition.trials_sharpe_std_min
            ):
                raise MetricContractError(
                    f"{metric}: trials_sharpe_std must be >= "
                    f"{definition.trials_sharpe_std_min:g}"
                )
        value = None if insufficient_sample else formula(context, annualization)
        if value is not None and not math.isfinite(value):
            raise MetricContractError(f"{metric}: formula returned a non-finite value")
        plausibility_violation = value is not None and (
            (
                definition.plausible_min is not None
                and value < definition.plausible_min
            )
            or (
                definition.plausible_max is not None
                and value > definition.plausible_max
            )
        )
        status = (
            "INSUFFICIENT_SAMPLE"
            if insufficient_sample
            else "CRITICAL"
            if plausibility_violation
            else self._status(value, definition)
        )
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise MetricContractError("as_of must be timezone-aware")
        identity_context = self._identity_context(context)
        context_hash = "sha256:" + hashlib.sha256(
            canonical_json_bytes(identity_context)
        ).hexdigest()
        event_identity = canonical_json_bytes(
            {
                "catalog_version": self.catalog.version,
                "formula_version": definition.formula_version,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "metric": metric,
                "window": window,
                "environment": canonical_environment,
                "as_of": as_of,
                "run_id": run_id,
                "strategy_id": strategy_id,
                "asset_id": asset_id,
                "annualization_periods": annualization,
                "context_hash": context_hash,
            }
        ).decode("utf-8")
        safe_lineage = self._validated_metadata("lineage", lineage or {})
        return MetricEvent(
            metric_event_id=str(uuid.uuid5(uuid.NAMESPACE_URL, event_identity)),
            event_time=as_of.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            catalog_version=self.catalog.version,
            formula_version=definition.formula_version,
            entity_type=entity_type,
            entity_id=entity_id,
            strategy_id=strategy_id,
            asset_id=asset_id,
            run_id=run_id,
            environment=canonical_environment,
            metric_namespace=definition.namespace,
            metric_name=definition.name,
            metric_value=None if plausibility_violation else value,
            metric_unit=definition.unit,
            status=status,
            threshold_warning=definition.warning,
            threshold_critical=definition.critical,
            dimensions={
                "window": window,
                "context_hash": context_hash,
                "n_trades": n_trades,
                "n_observations": observation_count,
                "plausibility_violation": plausibility_violation,
                "return_interval": context.get("return_interval"),
                "annualization_periods": (
                    int(annualization)
                    if definition.annualization is not None
                    else None
                ),
                "n_trials": context.get("n_trials"),
                "trials_sharpe_std": context.get("trials_sharpe_std"),
            },
            lineage=safe_lineage,
        )

    def _annualization(
        self,
        definition: MetricDefinition,
        asset_id: str | None,
        context: Mapping[str, Any],
    ) -> float:
        if definition.annualization == "from_asset_registry":
            if not asset_id:
                raise MetricContractError(
                    "asset_id is required for registry-backed annualization"
                )
            interval = context.get("return_interval")
            if not isinstance(interval, str) or not interval:
                raise MetricContractError(
                    "return_interval is required for registry-backed annualization"
                )
            if self.annualization_registry is None:
                raise MetricContractError(
                    "AssetProfile annualization registry is not configured"
                )
            try:
                value = self.annualization_registry.periods_per_year(
                    asset_id, interval
                )
            except ValueError as exc:
                raise MetricContractError(str(exc)) from exc
        elif definition.annualization is None:
            value = 1.0
        else:
            value = definition.annualization
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
        ):
            raise MetricContractError("annualization must be a positive number")
        return float(value)

    @staticmethod
    def _validate_window(
        *, window: str, as_of: datetime, context: Mapping[str, Any]
    ) -> None:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise MetricContractError("as_of must be timezone-aware")
        start = context.get("window_start")
        end = context.get("window_end")
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise MetricContractError(
                "window_start and window_end datetimes are required"
            )
        if (
            start.tzinfo is None
            or start.utcoffset() is None
            or end.tzinfo is None
            or end.utcoffset() is None
        ):
            raise MetricContractError("metric window timestamps must be timezone-aware")
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        as_of_utc = as_of.astimezone(timezone.utc)
        if start_utc > end_utc or end_utc != as_of_utc:
            raise MetricContractError("metric window must end exactly at as_of")
        if window.endswith("w") and window[:-1].isdigit():
            expected = timedelta(weeks=int(window[:-1]))
            if end_utc - start_utc != expected:
                raise MetricContractError(
                    f"{window} label does not match evaluated time range"
                )

    @staticmethod
    def _identity_context(context: Mapping[str, Any]) -> dict[str, Any]:
        def normalize(value: Any) -> Any:
            if isinstance(value, np.ndarray):
                return [normalize(item) for item in value.tolist()]
            if isinstance(value, np.generic):
                return normalize(value.item())
            if isinstance(value, Mapping):
                return {key: normalize(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            return value

        return {name: normalize(value) for name, value in context.items()}

    @staticmethod
    def _observation_count(context: Mapping[str, Any]) -> int | None:
        for name in ("returns", "equity", "weights", "asset_returns"):
            value = context.get(name)
            if isinstance(value, np.ndarray) and value.ndim == 1:
                return int(value.size)
            if isinstance(value, (list, tuple)):
                return len(value)
        return None

    @staticmethod
    def _validated_metadata(
        name: str, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise MetricContractError(f"{name} must be a mapping")
        copied = dict(value)
        try:
            canonical_json_bytes(copied)
        except CanonicalizationError as exc:
            raise MetricContractError(f"{name} is not canonical JSON: {exc}") from exc
        return copied

    @staticmethod
    def _status(value: float | None, definition: MetricDefinition) -> str:
        if value is None:
            return "N_A"
        if definition.higher_is_better:
            if definition.critical is not None and value <= definition.critical:
                return "CRITICAL"
            if definition.warning is not None and value <= definition.warning:
                return "WARNING"
        else:
            if definition.critical is not None and value >= definition.critical:
                return "CRITICAL"
            if definition.warning is not None and value >= definition.warning:
                return "WARNING"
        return "OK"
