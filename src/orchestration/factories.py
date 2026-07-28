"""Pure DAG specifications for the FABRIC data, ACTION and DIAGNOSTIC lanes.

This module deliberately has no Airflow import.  The Airflow adapter consumes
these immutable specifications, which keeps parse-time policy checks usable in
CI and by future orchestrators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping

from .dataset_uri import DatasetURI, validate_dataset_edges


class FactoryKind(StrEnum):
    DATA = "data"
    STRATEGY = "strategy"
    FORECAST = "forecast"
    BACKFILL = "backfill"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    callable_path: str
    args: tuple[str, ...] = ()
    pool: str | None = None
    timeout_minutes: int = 45
    retries: int = 1


@dataclass(frozen=True)
class DagSpec:
    dag_id: str
    kind: FactoryKind
    owner_id: str
    schedule: str | None
    tasks: tuple[TaskSpec, ...]
    consumes: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    as_of: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def assert_constitutional(self) -> None:
        if not self.tasks:
            raise ValueError(f"{self.dag_id}: at least one task is required")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError(f"{self.dag_id}: duplicate task_id")
        if self.kind == FactoryKind.BACKFILL and not self.as_of:
            raise ValueError(f"{self.dag_id}: a backfill requires an explicit as_of")
        edges = [
            {"source": source, "target": target}
            for source in self.consumes
            for target in self.produces
        ]
        validate_dataset_edges(edges)
        if self.kind == FactoryKind.FORECAST:
            for uri in self.produces:
                parsed = DatasetURI.parse(uri)
                if parsed.scheme != "forecast":
                    raise ValueError(
                        f"{self.dag_id}: DIAGNOSTIC factory may publish only forecast:// datasets"
                    )
        if self.kind == FactoryKind.STRATEGY:
            for uri in self.produces:
                parsed = DatasetURI.parse(uri)
                if parsed.scheme not in {"strategy", "action", "artifact"}:
                    raise ValueError(
                        f"{self.dag_id}: ACTION factory emitted forbidden namespace {parsed.scheme}"
                    )


def _task(raw: Mapping[str, Any], default_pool: str | None = None) -> TaskSpec:
    return TaskSpec(
        task_id=str(raw["id"]),
        callable_path=str(raw["script"]),
        args=tuple(str(value) for value in raw.get("args", ())),
        pool=raw.get("pool", default_pool),
        timeout_minutes=int(raw.get("timeout_minutes", 45)),
        retries=int(raw.get("retries", 1)),
    )


def build_data_specs(config: Mapping[str, Any]) -> list[DagSpec]:
    specs: list[DagSpec] = []
    for asset_id, raw in (config.get("data") or {}).items():
        if not raw.get("enabled", True):
            continue
        tasks = tuple(_task(task, raw.get("external_api_pool")) for task in raw["tasks"])
        spec = DagSpec(
            dag_id=f"asset__{asset_id}__data",
            kind=FactoryKind.DATA,
            owner_id=asset_id,
            schedule=raw.get("schedule"),
            tasks=tasks,
            consumes=tuple(raw.get("consumes", ())),
            produces=tuple(raw.get("produces", ())),
            tags=("fabric", "data", asset_id),
        )
        spec.assert_constitutional()
        specs.append(spec)
    return specs


def build_strategy_specs(config: Mapping[str, Any]) -> list[DagSpec]:
    specs: list[DagSpec] = []
    for sleeve_id, raw in (config.get("strategies") or {}).items():
        if not raw.get("enabled", True):
            continue
        spec = DagSpec(
            dag_id=f"strat__{sleeve_id}",
            kind=FactoryKind.STRATEGY,
            owner_id=sleeve_id,
            schedule=raw.get("schedule"),
            tasks=tuple(_task(task) for task in raw["tasks"]),
            consumes=tuple(raw.get("consumes", ())),
            produces=tuple(raw.get("produces", ())),
            tags=("fabric", "action", sleeve_id),
        )
        spec.assert_constitutional()
        specs.append(spec)
    return specs


def build_forecast_specs(config: Mapping[str, Any]) -> list[DagSpec]:
    specs: list[DagSpec] = []
    for forecast_id, raw in (config.get("forecasts") or {}).items():
        if not raw.get("enabled", True):
            continue
        spec = DagSpec(
            dag_id=f"forecast__{forecast_id}",
            kind=FactoryKind.FORECAST,
            owner_id=forecast_id,
            schedule=raw.get("schedule"),
            tasks=tuple(_task(task) for task in raw["tasks"]),
            consumes=tuple(raw.get("consumes", ())),
            produces=tuple(raw.get("produces", ())),
            tags=("fabric", "diagnostic", forecast_id),
        )
        spec.assert_constitutional()
        specs.append(spec)
    return specs


def build_backfill_spec(
    *,
    asset_id: str,
    as_of: str,
    tasks: Iterable[TaskSpec],
    consumes: Iterable[str] = (),
    produces: Iterable[str] = (),
) -> DagSpec:
    """Build a deliberately unscheduled, cutoff-pinned backfill DAG."""
    spec = DagSpec(
        dag_id=f"backfill__{asset_id}__{as_of.replace(':', '').replace('-', '')}",
        kind=FactoryKind.BACKFILL,
        owner_id=asset_id,
        schedule=None,
        tasks=tuple(tasks),
        consumes=tuple(consumes),
        produces=tuple(produces),
        tags=("fabric", "backfill", asset_id),
        as_of=as_of,
    )
    spec.assert_constitutional()
    return spec


def build_all_specs(config: Mapping[str, Any]) -> list[DagSpec]:
    """Return all regular factories after validating cross-DAG dataset edges."""
    specs = [
        *build_data_specs(config),
        *build_strategy_specs(config),
        *build_forecast_specs(config),
    ]
    validate_dataset_edges(
        [
            {"source": source, "target": target}
            for spec in specs
            for source in spec.consumes
            for target in spec.produces
        ]
    )
    dag_ids = [spec.dag_id for spec in specs]
    if len(set(dag_ids)) != len(dag_ids):
        raise ValueError("factory generated duplicate dag_id")
    return specs
