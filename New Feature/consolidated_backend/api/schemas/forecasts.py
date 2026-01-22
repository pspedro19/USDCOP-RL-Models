"""
Forecast Pydantic schemas.

This module defines the schema models for forecast endpoints,
including individual forecasts, responses, and filtering options.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict
from datetime import date, datetime
from enum import Enum


class DirectionEnum(str, Enum):
    """Price movement direction."""
    UP = "UP"
    DOWN = "DOWN"
    NEUTRAL = "NEUTRAL"


class SignalEnum(str, Enum):
    """Trading signal."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class DataSourceEnum(str, Enum):
    """Data source type."""
    POSTGRESQL = "postgresql"
    CSV = "csv"
    MINIO = "minio"
    NONE = "none"


class ForecastFilters(BaseModel):
    """
    Query filters for forecast endpoints.

    Attributes:
        model: Filter by model name (e.g., 'ridge', 'xgboost')
        horizon: Filter by forecast horizon in days (e.g., 5, 10, 20)
        week: Filter by inference week number (1-52)
        year: Filter by year
        limit: Maximum number of results to return
    """
    model: Optional[str] = Field(
        None,
        description="Filter by model name",
        example="ridge"
    )
    horizon: Optional[int] = Field(
        None,
        description="Filter by forecast horizon in days",
        example=5,
        ge=1,
        le=60
    )
    week: Optional[int] = Field(
        None,
        description="Filter by inference week number",
        example=25,
        ge=1,
        le=53
    )
    year: Optional[int] = Field(
        None,
        description="Filter by year",
        example=2024,
        ge=2020,
        le=2030
    )
    limit: int = Field(
        100,
        description="Maximum number of results",
        ge=1,
        le=1000
    )


class ForecastItem(BaseModel):
    """
    Individual forecast item.

    Attributes:
        inference_date: Date when forecast was generated
        inference_week: Week number of inference
        inference_year: Year of inference
        target_date: Target date for the prediction
        model_name: Name of the ML model
        horizon: Forecast horizon in days
        base_price: Base USD/COP price at inference time
        predicted_price: Predicted USD/COP price
        predicted_return_pct: Predicted return percentage
        price_change: Absolute price change
        direction: Predicted price direction (UP/DOWN)
        signal: Trading signal (BUY/SELL/HOLD)
        actual_price: Actual price (if known)
        direction_correct: Whether direction prediction was correct
        minio_week_path: Path to artifacts in MinIO
        model_display_name: Human-readable model name
        model_type: Type of model (linear, tree-based, etc.)
        horizon_label: Human-readable horizon label
        horizon_category: Horizon category (short/medium/long)
    """
    inference_date: Optional[str] = Field(None, description="Date when forecast was generated")
    inference_week: Optional[int] = Field(None, description="Week number of inference")
    inference_year: Optional[int] = Field(None, description="Year of inference")
    target_date: Optional[str] = Field(None, description="Target date for the prediction")
    model_name: Optional[str] = Field(None, description="Name of the ML model", example="ridge")
    horizon: Optional[int] = Field(None, description="Forecast horizon in days", example=5)
    base_price: Optional[float] = Field(None, description="Base USD/COP price", example=4250.50)
    predicted_price: Optional[float] = Field(None, description="Predicted USD/COP price", example=4275.25)
    predicted_return_pct: Optional[float] = Field(None, description="Predicted return percentage", example=0.58)
    price_change: Optional[float] = Field(None, description="Absolute price change", example=24.75)
    direction: Optional[str] = Field(None, description="Predicted direction (UP/DOWN)", example="UP")
    signal: Optional[str] = Field(None, description="Trading signal", example="BUY")
    actual_price: Optional[float] = Field(None, description="Actual price if known")
    direction_correct: Optional[bool] = Field(None, description="Whether direction was correct")
    minio_week_path: Optional[str] = Field(None, description="Path in MinIO")
    model_display_name: Optional[str] = Field(None, description="Human-readable model name")
    model_type: Optional[str] = Field(None, description="Model type", example="linear")
    horizon_label: Optional[str] = Field(None, description="Horizon label", example="5 days")
    horizon_category: Optional[str] = Field(None, description="Horizon category", example="short")

    class Config:
        json_schema_extra = {
            "example": {
                "inference_date": "2024-01-15",
                "inference_week": 3,
                "inference_year": 2024,
                "target_date": "2024-01-22",
                "model_name": "ridge",
                "horizon": 5,
                "base_price": 4250.50,
                "predicted_price": 4275.25,
                "predicted_return_pct": 0.58,
                "price_change": 24.75,
                "direction": "UP",
                "signal": "BUY"
            }
        }


class ForecastResponse(BaseModel):
    """
    Response for forecast list endpoints.

    Attributes:
        source: Data source (postgresql, csv, minio, none)
        count: Number of forecasts returned
        data: List of forecast items
        message: Optional message (e.g., when no data)
    """
    source: str = Field(
        ...,
        description="Data source used",
        example="postgresql"
    )
    count: int = Field(
        ...,
        description="Number of forecasts returned",
        example=50
    )
    data: List[Dict[str, Any]] = Field(
        ...,
        description="List of forecast items"
    )
    message: Optional[str] = Field(
        None,
        description="Optional message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "source": "postgresql",
                "count": 2,
                "data": [
                    {
                        "inference_date": "2024-01-15",
                        "model_name": "ridge",
                        "horizon": 5,
                        "predicted_price": 4275.25,
                        "direction": "UP"
                    },
                    {
                        "inference_date": "2024-01-15",
                        "model_name": "xgboost",
                        "horizon": 5,
                        "predicted_price": 4280.10,
                        "direction": "UP"
                    }
                ]
            }
        }


class LatestForecastItem(BaseModel):
    """
    Latest forecast item with additional metadata.

    Contains the same fields as ForecastItem plus model metrics.
    """
    inference_date: Optional[str] = None
    inference_week: Optional[int] = None
    inference_year: Optional[int] = None
    model_id: Optional[str] = None
    horizon_id: Optional[int] = None
    base_price: Optional[float] = None
    predicted_price: Optional[float] = None
    predicted_return_pct: Optional[float] = None
    direction: Optional[str] = None
    signal: Optional[str] = None
    model_display_name: Optional[str] = None
    model_type: Optional[str] = None
    horizon_label: Optional[str] = None


class LatestForecastResponse(BaseModel):
    """
    Response for latest forecasts endpoint.

    Attributes:
        source: Data source used
        count: Number of forecasts
        data: List of latest forecast items
    """
    source: str = Field(..., description="Data source", example="postgresql")
    count: int = Field(..., description="Number of forecasts", example=25)
    data: List[Dict[str, Any]] = Field(..., description="Forecast data")


class ConsensusItem(BaseModel):
    """
    Consensus forecast item (aggregated across models).

    Attributes:
        inference_date: Date of consensus calculation
        horizon_id: Forecast horizon in days
        consensus_price: Consensus predicted price
        consensus_direction: Consensus direction
        model_agreement_pct: Percentage of models agreeing
        horizon_label: Human-readable horizon
        horizon_category: Horizon category
    """
    inference_date: Optional[str] = Field(None, description="Consensus calculation date")
    horizon_id: Optional[int] = Field(None, description="Forecast horizon", example=5)
    consensus_price: Optional[float] = Field(None, description="Consensus price", example=4270.50)
    consensus_direction: Optional[str] = Field(None, description="Consensus direction", example="UP")
    model_agreement_pct: Optional[float] = Field(None, description="Model agreement percentage", example=80.0)
    horizon_label: Optional[str] = Field(None, description="Horizon label", example="5 days")
    horizon_category: Optional[str] = Field(None, description="Horizon category", example="short")


class ConsensusResponse(BaseModel):
    """
    Response for consensus endpoint.

    Attributes:
        source: Data source
        count: Number of consensus items
        data: Consensus data by horizon
    """
    source: str = Field(..., description="Data source", example="postgresql")
    count: int = Field(..., description="Number of items", example=4)
    data: List[Dict[str, Any]] = Field(..., description="Consensus data")


class MinioFileInfo(BaseModel):
    """
    Information about a file in MinIO.

    Attributes:
        name: File name
        size: File size in bytes
        last_modified: Last modification timestamp
        url: URL to access the file
    """
    name: str = Field(..., description="File name", example="forward_forecast_ridge.png")
    size: Optional[int] = Field(None, description="File size in bytes", example=125430)
    last_modified: Optional[str] = Field(None, description="Last modified timestamp")
    url: str = Field(..., description="File access URL")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "forward_forecast_ridge.png",
                "size": 125430,
                "last_modified": "2024-01-15T10:30:00.000000",
                "url": "http://localhost:9000/forecasts/2024/week03/figures/forward_forecast_ridge.png"
            }
        }


class WeekForecastResponse(BaseModel):
    """
    Response for forecasts by week endpoint.

    Attributes:
        year: Year of forecasts
        week: Week number
        forecasts: List of forecasts
        minio_files: List of associated files in MinIO
        minio_path: Base path in MinIO
        source: Data source
        count: Number of forecasts
    """
    year: int = Field(..., description="Year", example=2024)
    week: int = Field(..., description="Week number", example=3)
    forecasts: List[Dict[str, Any]] = Field(..., description="Forecasts for the week")
    minio_files: List[Dict[str, Any]] = Field(default=[], description="Associated MinIO files")
    minio_path: Optional[str] = Field(None, description="MinIO base path")
    source: str = Field(..., description="Data source", example="postgresql")
    count: Optional[int] = Field(None, description="Forecast count")

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2024,
                "week": 3,
                "forecasts": [
                    {"model_name": "ridge", "horizon": 5, "predicted_price": 4275.25}
                ],
                "minio_files": [
                    {"name": "forward_forecast_ridge.png", "size": 125430}
                ],
                "minio_path": "forecasts/2024/week03/",
                "source": "postgresql",
                "count": 25
            }
        }


class HorizonForecastResponse(BaseModel):
    """
    Response for forecasts by horizon endpoint.

    Attributes:
        horizon: Forecast horizon in days
        source: Data source
        count: Number of forecasts
        data: Forecast data
    """
    horizon: int = Field(..., description="Forecast horizon in days", example=5)
    source: str = Field(..., description="Data source", example="postgresql")
    count: int = Field(..., description="Number of forecasts", example=50)
    data: List[Dict[str, Any]] = Field(..., description="Forecast data")


class MetricItem(BaseModel):
    """
    Model performance metric.

    Attributes:
        model_id: Model identifier
        horizon_id: Horizon in days
        direction_accuracy: Direction accuracy percentage
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared score
    """
    model_id: Optional[str] = Field(None, description="Model identifier")
    horizon_id: Optional[int] = Field(None, description="Horizon in days")
    direction_accuracy: Optional[float] = Field(None, description="Direction accuracy %")
    rmse: Optional[float] = Field(None, description="RMSE")
    mae: Optional[float] = Field(None, description="MAE")
    r2: Optional[float] = Field(None, description="R-squared")


class DashboardResponse(BaseModel):
    """
    Complete dashboard data response.

    Attributes:
        source: Data source
        forecasts: Latest forecasts by model/horizon
        consensus: Consensus forecasts by horizon
        metrics: Model performance metrics
        last_update: Timestamp of last data update
        error: Error message if any
    """
    source: str = Field(..., description="Data source", example="postgresql")
    forecasts: List[Dict[str, Any]] = Field(..., description="Latest forecasts")
    consensus: List[Dict[str, Any]] = Field(default=[], description="Consensus data")
    metrics: List[Dict[str, Any]] = Field(default=[], description="Model metrics")
    last_update: str = Field(..., description="Last update timestamp")
    error: Optional[str] = Field(None, description="Error message if any")

    class Config:
        json_schema_extra = {
            "example": {
                "source": "postgresql",
                "forecasts": [
                    {
                        "model_name": "ridge",
                        "horizon_days": 5,
                        "current_price": 4250.50,
                        "predicted_price": 4275.25,
                        "direction": "UP",
                        "signal": "BUY"
                    }
                ],
                "consensus": [
                    {
                        "horizon_id": 5,
                        "consensus_price": 4270.50,
                        "consensus_direction": "UP"
                    }
                ],
                "metrics": [
                    {
                        "model_id": "ridge",
                        "horizon_id": 5,
                        "direction_accuracy": 0.72,
                        "rmse": 45.2
                    }
                ],
                "last_update": "2024-01-15T10:30:00.000000"
            }
        }
