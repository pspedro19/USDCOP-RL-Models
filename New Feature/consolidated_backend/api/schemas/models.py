"""
Model Pydantic schemas.

This module defines the schema models for model endpoints,
including model information, metrics, and comparisons.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class ModelTypeEnum(str, Enum):
    """Type of machine learning model."""
    LINEAR = "linear"
    TREE_BASED = "tree_based"
    BAYESIAN = "bayesian"
    ENSEMBLE = "ensemble"
    NEURAL_NETWORK = "neural_network"


class ModelMetric(BaseModel):
    """
    Performance metric for a model at a specific horizon.

    Attributes:
        horizon: Forecast horizon in days
        direction_accuracy: Percentage of correct direction predictions
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared (coefficient of determination)
        mape: Mean Absolute Percentage Error
        sharpe_ratio: Risk-adjusted return metric
    """
    horizon: int = Field(
        ...,
        description="Forecast horizon in days",
        example=5,
        ge=1,
        le=60
    )
    direction_accuracy: Optional[float] = Field(
        None,
        description="Direction accuracy percentage (0-100)",
        example=72.5,
        ge=0,
        le=100
    )
    rmse: Optional[float] = Field(
        None,
        description="Root Mean Squared Error",
        example=45.23
    )
    mae: Optional[float] = Field(
        None,
        description="Mean Absolute Error",
        example=35.12
    )
    r2: Optional[float] = Field(
        None,
        description="R-squared coefficient",
        example=0.85
    )
    mape: Optional[float] = Field(
        None,
        description="Mean Absolute Percentage Error",
        example=1.2
    )
    sharpe_ratio: Optional[float] = Field(
        None,
        description="Sharpe ratio for risk-adjusted returns",
        example=1.45
    )

    class Config:
        json_schema_extra = {
            "example": {
                "horizon": 5,
                "direction_accuracy": 72.5,
                "rmse": 45.23,
                "mae": 35.12,
                "r2": 0.85
            }
        }


class ModelInfo(BaseModel):
    """
    Basic model information with average metrics.

    Attributes:
        name: Model name/identifier
        horizons: List of supported forecast horizons
        avg_direction_accuracy: Average direction accuracy across horizons
        avg_rmse: Average RMSE across horizons
        model_type: Type of model (linear, tree-based, etc.)
        description: Model description
    """
    name: str = Field(
        ...,
        description="Model name/identifier",
        example="ridge"
    )
    horizons: List[int] = Field(
        ...,
        description="Supported forecast horizons in days",
        example=[5, 10, 20, 40]
    )
    avg_direction_accuracy: Optional[float] = Field(
        None,
        description="Average direction accuracy percentage",
        example=68.5
    )
    avg_rmse: Optional[float] = Field(
        None,
        description="Average RMSE across horizons",
        example=52.3
    )
    model_type: Optional[str] = Field(
        None,
        description="Type of machine learning model",
        example="linear"
    )
    description: Optional[str] = Field(
        None,
        description="Model description"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "ridge",
                "horizons": [5, 10, 20, 40],
                "avg_direction_accuracy": 68.5,
                "avg_rmse": 52.3,
                "model_type": "linear"
            }
        }


class ModelResponse(BaseModel):
    """
    Response for model list endpoint.

    Attributes:
        models: List of available models with their info
        count: Total number of models
    """
    models: List[ModelInfo] = Field(
        ...,
        description="List of available models"
    )
    count: Optional[int] = Field(
        None,
        description="Total number of models"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "models": [
                    {
                        "name": "ridge",
                        "horizons": [5, 10, 20, 40],
                        "avg_direction_accuracy": 68.5,
                        "avg_rmse": 52.3
                    },
                    {
                        "name": "xgboost",
                        "horizons": [5, 10, 20, 40],
                        "avg_direction_accuracy": 71.2,
                        "avg_rmse": 48.7
                    }
                ],
                "count": 2
            }
        }


class ModelDetailMetric(BaseModel):
    """
    Detailed metric for a specific horizon.

    Attributes:
        horizon: Forecast horizon in days
        direction_accuracy: Direction accuracy percentage
        rmse: Root Mean Squared Error
        mae: Mean Absolute Error
        r2: R-squared
        sample_count: Number of samples used
    """
    horizon: int = Field(..., description="Horizon in days", example=5)
    direction_accuracy: Optional[float] = Field(None, description="Direction accuracy %")
    rmse: Optional[float] = Field(None, description="RMSE")
    mae: Optional[float] = Field(None, description="MAE")
    r2: Optional[float] = Field(None, description="R-squared")
    sample_count: Optional[int] = Field(None, description="Sample count")


class ModelDetailResponse(BaseModel):
    """
    Detailed model information response.

    Attributes:
        name: Model name
        metrics_by_horizon: Performance metrics for each horizon
        total_forecasts: Total number of forecasts made
        model_type: Type of model
        training_date: Last training date
        description: Model description
    """
    name: str = Field(
        ...,
        description="Model name",
        example="ridge"
    )
    metrics_by_horizon: List[ModelDetailMetric] = Field(
        ...,
        description="Metrics for each supported horizon"
    )
    total_forecasts: int = Field(
        ...,
        description="Total number of forecasts made",
        example=1250
    )
    model_type: Optional[str] = Field(
        None,
        description="Model type",
        example="linear"
    )
    training_date: Optional[str] = Field(
        None,
        description="Last training date"
    )
    description: Optional[str] = Field(
        None,
        description="Model description"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "name": "ridge",
                "metrics_by_horizon": [
                    {"horizon": 5, "direction_accuracy": 72.5, "rmse": 42.3},
                    {"horizon": 10, "direction_accuracy": 68.2, "rmse": 55.1},
                    {"horizon": 20, "direction_accuracy": 64.8, "rmse": 72.4},
                    {"horizon": 40, "direction_accuracy": 58.3, "rmse": 95.2}
                ],
                "total_forecasts": 1250,
                "model_type": "linear"
            }
        }


class ModelComparisonItem(BaseModel):
    """
    Single model entry in a comparison.

    Attributes:
        model: Model name
        is_selected: Whether this is the selected model for comparison
        avg_da: Average direction accuracy
        avg_rmse: Average RMSE
        rank: Rank among all models (by direction accuracy)
        performance_vs_baseline: Performance compared to baseline
    """
    model: str = Field(
        ...,
        description="Model name",
        example="ridge"
    )
    is_selected: bool = Field(
        ...,
        description="Whether this is the selected model",
        example=False
    )
    avg_da: Optional[float] = Field(
        None,
        description="Average direction accuracy",
        example=68.5
    )
    avg_rmse: Optional[float] = Field(
        None,
        description="Average RMSE",
        example=52.3
    )
    rank: Optional[int] = Field(
        None,
        description="Rank by direction accuracy",
        example=3
    )
    performance_vs_baseline: Optional[float] = Field(
        None,
        description="Performance vs baseline (%)",
        example=15.2
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "ridge",
                "is_selected": True,
                "avg_da": 68.5,
                "avg_rmse": 52.3,
                "rank": 3
            }
        }


class ModelComparison(BaseModel):
    """
    Response for model comparison endpoint.

    Attributes:
        selected_model: Name of the model being compared
        comparison: List of all models with their metrics for comparison
        best_model: Name of the best performing model
        model_count: Total number of models compared
    """
    selected_model: str = Field(
        ...,
        description="Selected model for comparison",
        example="ridge"
    )
    comparison: List[ModelComparisonItem] = Field(
        ...,
        description="All models sorted by performance"
    )
    best_model: Optional[str] = Field(
        None,
        description="Best performing model name",
        example="xgboost"
    )
    model_count: Optional[int] = Field(
        None,
        description="Total number of models",
        example=5
    )

    class Config:
        json_schema_extra = {
            "example": {
                "selected_model": "ridge",
                "comparison": [
                    {"model": "xgboost", "is_selected": False, "avg_da": 71.2, "avg_rmse": 48.7, "rank": 1},
                    {"model": "lightgbm", "is_selected": False, "avg_da": 70.1, "avg_rmse": 49.2, "rank": 2},
                    {"model": "ridge", "is_selected": True, "avg_da": 68.5, "avg_rmse": 52.3, "rank": 3},
                    {"model": "catboost", "is_selected": False, "avg_da": 67.8, "avg_rmse": 53.1, "rank": 4},
                    {"model": "bayesian_ridge", "is_selected": False, "avg_da": 66.2, "avg_rmse": 55.4, "rank": 5}
                ],
                "best_model": "xgboost",
                "model_count": 5
            }
        }


class ModelMetricsHistory(BaseModel):
    """
    Historical performance metrics for a model.

    Attributes:
        model: Model name
        horizon: Forecast horizon
        history: List of historical metric records
    """
    model: str = Field(..., description="Model name")
    horizon: int = Field(..., description="Forecast horizon in days")
    history: List[Dict[str, Any]] = Field(..., description="Historical metrics")


class ModelRanking(BaseModel):
    """
    Model ranking by a specific metric.

    Attributes:
        metric: Metric used for ranking
        horizon: Horizon for ranking (optional, null for overall)
        rankings: Ordered list of models with their scores
    """
    metric: str = Field(
        ...,
        description="Metric used for ranking",
        example="direction_accuracy"
    )
    horizon: Optional[int] = Field(
        None,
        description="Specific horizon or null for overall"
    )
    rankings: List[Dict[str, Any]] = Field(
        ...,
        description="Models ordered by rank"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "metric": "direction_accuracy",
                "horizon": 5,
                "rankings": [
                    {"rank": 1, "model": "xgboost", "score": 73.5},
                    {"rank": 2, "model": "lightgbm", "score": 72.1},
                    {"rank": 3, "model": "ridge", "score": 70.8}
                ]
            }
        }
