"""
Pydantic schemas for the USD/COP Forecasting API.

This module exports all schema classes for use in FastAPI endpoints.
"""
from .health import (
    ServiceStatus,
    ServiceDetails,
    HealthCheck,
    ReadinessCheck,
    LivenessCheck,
    IndividualServiceCheck,
)

from .forecasts import (
    ForecastItem,
    ForecastResponse,
    ForecastFilters,
    LatestForecastItem,
    LatestForecastResponse,
    ConsensusItem,
    ConsensusResponse,
    WeekForecastResponse,
    HorizonForecastResponse,
    DashboardResponse,
    MinioFileInfo,
)

from .models import (
    ModelMetric,
    ModelInfo,
    ModelResponse,
    ModelDetailMetric,
    ModelDetailResponse,
    ModelComparisonItem,
    ModelComparison,
)

__all__ = [
    # Health schemas
    "ServiceStatus",
    "ServiceDetails",
    "HealthCheck",
    "ReadinessCheck",
    "LivenessCheck",
    "IndividualServiceCheck",
    # Forecast schemas
    "ForecastItem",
    "ForecastResponse",
    "ForecastFilters",
    "LatestForecastItem",
    "LatestForecastResponse",
    "ConsensusItem",
    "ConsensusResponse",
    "WeekForecastResponse",
    "HorizonForecastResponse",
    "DashboardResponse",
    "MinioFileInfo",
    # Model schemas
    "ModelMetric",
    "ModelInfo",
    "ModelResponse",
    "ModelDetailMetric",
    "ModelDetailResponse",
    "ModelComparisonItem",
    "ModelComparison",
]
