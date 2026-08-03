"""Governed metric APIs with dependency-light lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.metrics.engine import (
        MetricCatalog,
        MetricDefinition,
        MetricEngine,
        MetricEnvironment,
        MetricEvent,
    )
    from src.metrics.persistence import PersistMetricEventResult, persist_metric_event


_EXPORT_MODULES = {
    "MetricCatalog": "src.metrics.engine",
    "MetricDefinition": "src.metrics.engine",
    "MetricEngine": "src.metrics.engine",
    "MetricEnvironment": "src.metrics.engine",
    "MetricEvent": "src.metrics.engine",
    "PersistMetricEventResult": "src.metrics.persistence",
    "persist_metric_event": "src.metrics.persistence",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    try:
        module_name = _EXPORT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value
