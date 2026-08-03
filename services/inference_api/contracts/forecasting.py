"""
Forecasting API Contracts
=========================

Pydantic schemas for forecasting endpoints.
These contracts are the SSOT for API request/response types.

Frontend Zod schemas must match these definitions.

@version 1.0.0
"""

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# ============================================================================
# ENUMS
# ============================================================================

class ForecastDirection(str, Enum):
    """Direction of price movement."""
    UP = "UP"
    DOWN = "DOWN"


class HorizonCategory(str, Enum):
    """Horizon classification."""
    SHORT = "short"   # 1-5 days
    MEDIUM = "medium"  # 10-20 days
    LONG = "long"     # 25-30 days


class ModelType(str, Enum):
    """Model category."""
    LINEAR = "linear"
    BOOSTING = "boosting"
    HYBRID = "hybrid"


# ============================================================================
# MODEL SCHEMAS
# ============================================================================

class ModelInfo(BaseModel):
    """Basic model information."""
    model_id: str = Field(..., description="Unique model identifier")
    model_name: str = Field(..., description="Display name")
    model_type: ModelType = Field(..., description="Model category")
    requires_scaling: bool = Field(False, description="Needs feature scaling")
    supports_early_stopping: bool = Field(False, description="Can early stop")
    is_active: bool = Field(True, description="Currently active")


class ModelMetrics(BaseModel):
    """Model performance metrics for a specific horizon."""
    model_id: str
    horizon_id: int
    direction_accuracy: float = Field(..., ge=0, le=100, description="DA percentage")
    rmse: float = Field(..., ge=0, description="Root Mean Squared Error")
    mae: float | None = Field(None, ge=0, description="Mean Absolute Error")
    mape: float | None = Field(None, description="Mean Absolute Percentage Error")
    r2: float | None = Field(None, description="R-squared")
    sample_count: int = Field(..., ge=0, description="Number of samples")


class ModelComparison(BaseModel):
    """Model comparison result."""
    model_id: str
    is_selected: bool
    avg_direction_accuracy: float
    avg_rmse: float
    rank: int


class ModelListResponse(BaseModel):
    """Response for listing all models."""
    models: list[ModelInfo]
    count: int


class ModelDetailResponse(BaseModel):
    """Detailed model information."""
    model: ModelInfo
    metrics_by_horizon: list[ModelMetrics]
    total_forecasts: int


class ModelRankingResponse(BaseModel):
    """Model ranking by metric."""
    metric: str
    horizon: int | None
    rankings: list[ModelComparison]


# ============================================================================
# FORECAST SCHEMAS
# ============================================================================

class Forecast(BaseModel):
    """Single forecast record."""
    id: str | None = None
    inference_date: date
    inference_week: int = Field(..., ge=1, le=53)
    inference_year: int = Field(..., ge=2020, le=2100)
    target_date: date
    model_id: str
    horizon_id: int
    base_price: float = Field(..., gt=0, description="Starting price")
    predicted_price: float = Field(..., gt=0, description="Forecasted price")
    predicted_return_pct: float | None = None
    price_change: float | None = None
    direction: ForecastDirection
    signal: int = Field(..., ge=-1, le=1, description="-1=SELL, 0=HOLD, 1=BUY")
    confidence: float | None = Field(None, ge=0, le=1)
    actual_price: float | None = None
    direction_correct: bool | None = None


class ForecastListResponse(BaseModel):
    """Response for forecast list."""
    source: str = Field(..., description="Data source: postgresql, csv, none")
    count: int = Field(..., ge=0)
    data: list[Forecast]


class LatestForecastResponse(BaseModel):
    """Latest forecasts per model/horizon."""
    source: str
    count: int
    data: list[Forecast]


# ============================================================================
# CONSENSUS SCHEMAS
# ============================================================================

class Consensus(BaseModel):
    """Consensus forecast for a horizon."""
    inference_date: date
    horizon_id: int
    horizon_label: str | None = None
    avg_predicted_price: float
    median_predicted_price: float | None = None
    std_predicted_price: float | None = None
    min_predicted_price: float | None = None
    max_predicted_price: float | None = None
    consensus_direction: ForecastDirection
    bullish_count: int = Field(..., ge=0)
    bearish_count: int = Field(..., ge=0)
    total_models: int = Field(..., ge=0)
    agreement_pct: float | None = None


class ConsensusResponse(BaseModel):
    """Response for consensus forecasts."""
    source: str
    count: int
    data: list[Consensus]


# ============================================================================
# DASHBOARD SCHEMAS
# ============================================================================

class DashboardResponse(BaseModel):
    """Complete dashboard data in one response."""
    source: str
    forecasts: list[dict[str, Any]]
    consensus: list[dict[str, Any]]
    metrics: list[dict[str, Any]]
    last_update: str


# ============================================================================
# USD/COP CAUSAL DIRECTIONAL REPLAY
# ============================================================================

class DirectionalEvidence(BaseModel):
    n: int = Field(..., ge=0)
    directional_accuracy: float | None = Field(None, ge=0, le=1)
    balanced_accuracy: float | None = Field(None, ge=0, le=1)
    up_recall: float | None = Field(None, ge=0, le=1)
    down_recall: float | None = Field(None, ge=0, le=1)
    minimum_class_recall: float | None = Field(None, ge=0, le=1)
    brier: float | None = Field(None, ge=0, le=1)
    prediction_up_rate: float | None = Field(None, ge=0, le=1)
    actual_up_rate: float | None = Field(None, ge=0, le=1)
    up_count: int = Field(..., ge=0)
    down_count: int = Field(..., ge=0)
    raw_score: float | None = Field(None, ge=0, le=1)
    shrunk_score: float | None = Field(None, ge=0, le=1)
    eligible: bool


class DirectionalHorizonForecast(BaseModel):
    horizon_days: int = Field(..., gt=0)
    role: str
    target_date: date
    target_date_estimated: bool
    probability_up: float = Field(..., ge=0, le=1)
    threshold: float = Field(..., ge=0, le=1)
    prediction: Literal["UP", "DOWN"]
    forecast_log_return: float
    forecast_return_pct: float
    forecast_price: float = Field(..., gt=0)
    forecast_price_change: float
    forecast_interval_lower: float = Field(..., gt=0)
    forecast_interval_upper: float = Field(..., gt=0)
    forecast_interval_level: float = Field(..., ge=0, le=1)
    point_forecast_direction: Literal["UP", "DOWN"]
    direction_price_agree: bool
    point_forecast_clipped: bool
    point_forecast_validation_eligible: bool
    actual: Literal[0, 1] | None
    actual_log_return: float | None
    actual_price: float | None = Field(None, gt=0)
    point_abs_error_price: float | None = Field(None, ge=0)
    point_abs_error_pct: float | None = Field(None, ge=0)
    hit: bool | None
    train_count: int = Field(..., ge=0)
    train_label_end: date | None
    point_train_label_end: date | None
    training_mode: str
    model_family: str
    feature_hash: str
    validation_eligible: bool
    evidence: DirectionalEvidence
    eligible_for_direction: bool
    selected: bool
    selection_rank: int = Field(..., gt=0)


class DirectionalReplayDecision(BaseModel):
    direction: Literal["UP", "DOWN", "FLAT"]
    action: Literal["LONG_USD_SHORT_COP", "SHORT_USD_LONG_COP", "FLAT"]
    status: Literal["SHADOW", "ABSTAIN"]
    signal_authorized: Literal[False]
    promotion_gate_passed: bool
    primary_horizon: int | None
    confirmation_horizons: list[int]
    selected_horizons: list[int]
    confidence_proxy: float = Field(..., ge=0, le=1)
    target_date: date | None
    forecast_price: float | None = Field(None, gt=0)
    forecast_return_pct: float | None
    forecast_interval_lower: float | None = Field(None, gt=0)
    forecast_interval_upper: float | None = Field(None, gt=0)
    actual: Literal[0, 1] | None
    hit: bool | None
    rationale: str


class DirectionalReplayWeek(BaseModel):
    iso_week: str
    year: int
    origin_date: date
    origin_is_partial_week: bool
    base_price: float = Field(..., gt=0)
    training_mode: str
    regime: dict[str, Any]
    decision: DirectionalReplayDecision
    horizons: list[DirectionalHorizonForecast]
    image_path: str


class DirectionalReplayIndex(BaseModel):
    schema_version: str
    contract_hash: str
    asset_id: Literal["usdcop"]
    symbol: Literal["USD/COP"]
    chart_symbol: Literal["USDCOP"]
    generated_at: str
    data_cutoff: date
    latest_week: str
    years: list[int]
    methodology: dict[str, Any]
    lineage: dict[str, dict[str, Any]]
    models: dict[str, dict[str, Any]]
    summaries: list[dict[str, Any]]
    weeks: list[DirectionalReplayWeek]


class WeekForecastResponse(BaseModel):
    """Forecasts for a specific week."""
    year: int
    week: int
    forecasts: list[dict[str, Any]]
    minio_files: list[dict[str, Any]]
    minio_path: str | None = None
    source: str
    count: int


class HorizonForecastResponse(BaseModel):
    """Forecasts for a specific horizon."""
    horizon: int
    source: str
    count: int
    data: list[dict[str, Any]]


# ============================================================================
# IMAGE SCHEMAS
# ============================================================================

class ImageMetadata(BaseModel):
    """Metadata for a forecast image."""
    image_type: str  # backtest, forecast, heatmap
    model_id: str
    horizon_id: int | None = None
    filename: str
    url: str
    size: int | None = None
    last_modified: str | None = None


class ImageListResponse(BaseModel):
    """List of available images."""
    images: list[ImageMetadata]
    count: int


# ============================================================================
# QUERY PARAMETERS
# ============================================================================

class ForecastQueryParams(BaseModel):
    """Query parameters for forecast filtering."""
    model: str | None = None
    horizon: int | None = Field(None, ge=1, le=60)
    week: int | None = Field(None, ge=1, le=53)
    year: int | None = Field(None, ge=2020, le=2100)
    limit: int = Field(100, ge=1, le=1000)
