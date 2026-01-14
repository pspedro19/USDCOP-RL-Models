"""
API Schema Definitions
======================

Request/Response schemas for inference API.
Used to generate OpenAPI spec and TypeScript types.

Contract: CTR-SHARED-API-001
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import Field, field_validator, model_validator

from .core import BaseSchema, DataSource
from .features import OBSERVATION_DIM
from .trading import TradeSchema, TradeSummarySchema


# =============================================================================
# TYPE VARIABLES
# =============================================================================


T = TypeVar("T")


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================


class BacktestRequestSchema(BaseSchema):
    """Request for backtest endpoint.

    POST /v1/backtest
    """

    start_date: str = Field(
        ...,
        description="Start date for backtest (YYYY-MM-DD)",
        examples=["2025-01-01"],
        json_schema_extra={"format": "date"}
    )
    end_date: str = Field(
        ...,
        description="End date for backtest (YYYY-MM-DD)",
        examples=["2025-06-30"],
        json_schema_extra={"format": "date"}
    )
    model_id: str = Field(
        default="ppo",
        description="Model ID to use for inference",
        examples=["ppo", "ppo_ensemble"]
    )
    force_regenerate: bool = Field(
        default=False,
        description="Force regeneration even if cached trades exist"
    )

    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        """Validate date format is YYYY-MM-DD."""
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Use YYYY-MM-DD")
        return v

    @model_validator(mode="after")
    def validate_date_range(self) -> "BacktestRequestSchema":
        """Validate end_date > start_date and range <= 365 days."""
        start = date.fromisoformat(self.start_date)
        end = date.fromisoformat(self.end_date)

        if end <= start:
            raise ValueError("end_date must be after start_date")

        if (end - start).days > 365:
            raise ValueError("Date range cannot exceed 365 days")

        return self


class InferenceRequestSchema(BaseSchema):
    """Request for single inference endpoint.

    POST /v1/inference
    """

    observation: List[float] = Field(
        ...,
        min_length=OBSERVATION_DIM,
        max_length=OBSERVATION_DIM,
        description=f"Observation vector ({OBSERVATION_DIM} dimensions)",
        examples=[[0.001, 0.005, 0.01, 50.0, 0.5, 25.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]]
    )
    model_id: str = Field(
        default="ppo",
        description="Model ID to use"
    )


class ReplayLoadRequestSchema(BaseSchema):
    """Request to load replay data.

    POST /api/replay/load
    """

    start_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Start date (YYYY-MM-DD)"
    )
    end_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="End date (YYYY-MM-DD)"
    )
    model_id: str = Field(..., description="Model ID for replay")
    force_regenerate: bool = Field(
        default=False,
        description="Force regenerate trades"
    )


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================


class ApiMetadataSchema(BaseSchema):
    """Metadata for API responses."""

    data_source: DataSource = Field(..., description="Source of the data")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Response timestamp (UTC)"
    )
    is_real_data: bool = Field(..., description="Whether data is real or mock")
    latency_ms: Optional[float] = Field(
        default=None, ge=0, description="Request latency in milliseconds"
    )
    cache_hit: Optional[bool] = Field(default=None, description="Whether cache was hit")
    request_id: Optional[str] = Field(default=None, description="Request tracking ID")


class ApiResponseSchema(BaseSchema, Generic[T]):
    """Generic API response wrapper.

    Use create_api_response() for type-safe instantiation.
    """
    model_config = BaseSchema.model_config.copy()
    model_config["extra"] = "allow"

    success: bool = Field(..., description="Whether request succeeded")
    data: Optional[Any] = Field(default=None, description="Response data")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    message: Optional[str] = Field(default=None, description="Additional message")
    metadata: ApiMetadataSchema = Field(..., description="Response metadata")


class BacktestResponseSchema(BaseSchema):
    """Response for backtest endpoint.

    POST /v1/backtest
    """

    success: bool = Field(default=True, description="Whether backtest succeeded")
    source: str = Field(
        ...,
        description="'database' if cached, 'generated' if newly computed"
    )
    trade_count: int = Field(..., ge=0, description="Number of trades generated")
    trades: List[TradeSchema] = Field(..., description="List of trades")
    summary: Optional[TradeSummarySchema] = Field(
        default=None, description="Trade summary statistics"
    )
    processing_time_ms: Optional[float] = Field(
        default=None, ge=0, description="Processing time in milliseconds"
    )
    date_range: Optional[Dict[str, str]] = Field(
        default=None, description="Actual date range of data"
    )


class HealthResponseSchema(BaseSchema):
    """Response for health check endpoint.

    GET /v1/health
    """

    status: str = Field(
        default="healthy",
        description="Service health status",
        examples=["healthy", "unhealthy", "degraded"]
    )
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    database_connected: bool = Field(..., description="Whether database is connected")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Health check timestamp"
    )


class ErrorResponseSchema(BaseSchema):
    """Error response for failed requests."""

    success: bool = Field(default=False)
    error: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(
        default=None,
        description="Machine-readable error code",
        examples=["INVALID_DATE_RANGE", "MODEL_NOT_FOUND", "DATABASE_ERROR"]
    )
    details: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional error details"
    )


class ProgressUpdateSchema(BaseSchema):
    """Progress update for streaming endpoints."""

    progress: float = Field(..., ge=0, le=1, description="Progress (0-1)")
    current_bar: int = Field(..., ge=0, description="Current bar index")
    total_bars: int = Field(..., ge=0, description="Total bars to process")
    trades_generated: int = Field(..., ge=0, description="Trades generated so far")
    status: str = Field(
        ...,
        description="Status",
        examples=["running", "completed", "error"]
    )
    message: Optional[str] = Field(default=None, description="Status message")


class ReplayLoadResponseSchema(BaseSchema):
    """Response for replay load endpoint."""

    trades: List[TradeSchema] = Field(..., description="Loaded trades")
    total: int = Field(..., ge=0, description="Total trade count")
    summary: Optional[TradeSummarySchema] = Field(default=None)
    source: str = Field(..., description="Data source")
    date_range: Dict[str, str] = Field(..., description="Date range of data")
    processing_time_ms: Optional[float] = Field(default=None, ge=0)


class ModelInfoSchema(BaseSchema):
    """Model information response."""
    model_config = BaseSchema.model_config.copy()
    model_config["protected_namespaces"] = ()

    model_id: str = Field(..., description="Model identifier")
    display_name: str = Field(..., description="Display name")
    version: Optional[str] = Field(default=None, description="Model version")
    status: str = Field(
        ...,
        description="Model status",
        examples=["available", "loaded", "not_found"]
    )
    observation_dim: int = Field(
        default=OBSERVATION_DIM,
        description="Observation dimension"
    )
    description: Optional[str] = Field(default=None, description="Model description")


class ModelsResponseSchema(BaseSchema):
    """Response for models listing endpoint."""

    models: List[ModelInfoSchema] = Field(..., description="Available models")
    default_model: str = Field(..., description="Default model ID")
    total: int = Field(..., ge=0, description="Total model count")


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_api_response(
    data: Optional[T] = None,
    success: bool = True,
    error: Optional[str] = None,
    message: Optional[str] = None,
    data_source: DataSource = DataSource.POSTGRES,
    is_real_data: bool = True,
    latency_ms: Optional[float] = None,
    cache_hit: Optional[bool] = None,
    request_id: Optional[str] = None,
) -> ApiResponseSchema:
    """Create a type-safe API response.

    Args:
        data: Response data
        success: Whether request succeeded
        error: Error message if failed
        message: Additional message
        data_source: Source of data
        is_real_data: Whether data is real
        latency_ms: Request latency
        cache_hit: Whether cache was hit
        request_id: Request tracking ID

    Returns:
        ApiResponseSchema instance
    """
    metadata = ApiMetadataSchema(
        data_source=data_source,
        is_real_data=is_real_data,
        latency_ms=latency_ms,
        cache_hit=cache_hit,
        request_id=request_id,
    )

    return ApiResponseSchema(
        success=success,
        data=data,
        error=error,
        message=message,
        metadata=metadata,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # Request schemas
    "BacktestRequestSchema",
    "InferenceRequestSchema",
    "ReplayLoadRequestSchema",
    # Response schemas
    "ApiMetadataSchema",
    "ApiResponseSchema",
    "BacktestResponseSchema",
    "HealthResponseSchema",
    "ErrorResponseSchema",
    "ProgressUpdateSchema",
    "ReplayLoadResponseSchema",
    "ModelInfoSchema",
    "ModelsResponseSchema",
    # Factory
    "create_api_response",
]
