"""
Forecasting API Contracts
=========================

Pydantic schemas for forecasting endpoints.
These contracts are the SSOT for API request/response types.

Frontend Zod schemas must match these definitions.

@version 1.0.0
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import date, datetime
from enum import Enum


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
    mae: Optional[float] = Field(None, ge=0, description="Mean Absolute Error")
    mape: Optional[float] = Field(None, description="Mean Absolute Percentage Error")
    r2: Optional[float] = Field(None, description="R-squared")
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
    models: List[ModelInfo]
    count: int


class ModelDetailResponse(BaseModel):
    """Detailed model information."""
    model: ModelInfo
    metrics_by_horizon: List[ModelMetrics]
    total_forecasts: int


class ModelRankingResponse(BaseModel):
    """Model ranking by metric."""
    metric: str
    horizon: Optional[int]
    rankings: List[ModelComparison]


# ============================================================================
# FORECAST SCHEMAS
# ============================================================================

class Forecast(BaseModel):
    """Single forecast record."""
    id: Optional[str] = None
    inference_date: date
    inference_week: int = Field(..., ge=1, le=53)
    inference_year: int = Field(..., ge=2020, le=2100)
    target_date: date
    model_id: str
    horizon_id: int
    base_price: float = Field(..., gt=0, description="Starting price")
    predicted_price: float = Field(..., gt=0, description="Forecasted price")
    predicted_return_pct: Optional[float] = None
    price_change: Optional[float] = None
    direction: ForecastDirection
    signal: int = Field(..., ge=-1, le=1, description="-1=SELL, 0=HOLD, 1=BUY")
    confidence: Optional[float] = Field(None, ge=0, le=1)
    actual_price: Optional[float] = None
    direction_correct: Optional[bool] = None


class ForecastListResponse(BaseModel):
    """Response for forecast list."""
    source: str = Field(..., description="Data source: postgresql, csv, none")
    count: int = Field(..., ge=0)
    data: List[Forecast]


class LatestForecastResponse(BaseModel):
    """Latest forecasts per model/horizon."""
    source: str
    count: int
    data: List[Forecast]


# ============================================================================
# CONSENSUS SCHEMAS
# ============================================================================

class Consensus(BaseModel):
    """Consensus forecast for a horizon."""
    inference_date: date
    horizon_id: int
    horizon_label: Optional[str] = None
    avg_predicted_price: float
    median_predicted_price: Optional[float] = None
    std_predicted_price: Optional[float] = None
    min_predicted_price: Optional[float] = None
    max_predicted_price: Optional[float] = None
    consensus_direction: ForecastDirection
    bullish_count: int = Field(..., ge=0)
    bearish_count: int = Field(..., ge=0)
    total_models: int = Field(..., ge=0)
    agreement_pct: Optional[float] = None


class ConsensusResponse(BaseModel):
    """Response for consensus forecasts."""
    source: str
    count: int
    data: List[Consensus]


# ============================================================================
# DASHBOARD SCHEMAS
# ============================================================================

class DashboardResponse(BaseModel):
    """Complete dashboard data in one response."""
    source: str
    forecasts: List[Dict[str, Any]]
    consensus: List[Dict[str, Any]]
    metrics: List[Dict[str, Any]]
    last_update: str


class WeekForecastResponse(BaseModel):
    """Forecasts for a specific week."""
    year: int
    week: int
    forecasts: List[Dict[str, Any]]
    minio_files: List[Dict[str, Any]]
    minio_path: Optional[str] = None
    source: str
    count: int


class HorizonForecastResponse(BaseModel):
    """Forecasts for a specific horizon."""
    horizon: int
    source: str
    count: int
    data: List[Dict[str, Any]]


# ============================================================================
# IMAGE SCHEMAS
# ============================================================================

class ImageMetadata(BaseModel):
    """Metadata for a forecast image."""
    image_type: str  # backtest, forecast, heatmap
    model_id: str
    horizon_id: Optional[int] = None
    filename: str
    url: str
    size: Optional[int] = None
    last_modified: Optional[str] = None


class ImageListResponse(BaseModel):
    """List of available images."""
    images: List[ImageMetadata]
    count: int


# ============================================================================
# QUERY PARAMETERS
# ============================================================================

class ForecastQueryParams(BaseModel):
    """Query parameters for forecast filtering."""
    model: Optional[str] = None
    horizon: Optional[int] = Field(None, ge=1, le=60)
    week: Optional[int] = Field(None, ge=1, le=53)
    year: Optional[int] = Field(None, ge=2020, le=2100)
    limit: int = Field(100, ge=1, le=1000)
