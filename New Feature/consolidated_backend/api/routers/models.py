"""
Model endpoints - Access ML model information and performance metrics.

This module provides REST API endpoints for accessing machine learning model
information, performance metrics, and model comparisons for the USD/COP
forecasting pipeline.

All endpoints require authentication.
"""
from fastapi import APIRouter, HTTPException, Depends, Path, Query
from typing import Optional, List, Dict, Any
import pandas as pd
import os
import logging

from ..auth.dependencies import get_current_user, User
from ..schemas.models import (
    ModelResponse,
    ModelDetailResponse,
    ModelComparison,
    ModelInfo,
    ModelDetailMetric,
    ModelComparisonItem,
)
from ..config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


def get_csv_data() -> Optional[pd.DataFrame]:
    """
    Load the dashboard CSV data.

    Returns:
        DataFrame with model data or None if not available
    """
    csv_path = os.path.join(settings.OUTPUTS_PATH, "bi_dashboard", "bi_dashboard_unified.csv")

    if not os.path.exists(csv_path):
        csv_path = os.path.join(settings.OUTPUTS_PATH, "bi", "bi_dashboard_unified.csv")

    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)

    return None


@router.get(
    "/",
    response_model=ModelResponse,
    summary="Get list of available models",
    description="""
    Retrieve a list of all available machine learning models with their average metrics.

    For each model, returns:
    - **name**: Model identifier (e.g., 'ridge', 'xgboost')
    - **horizons**: Supported forecast horizons in days
    - **avg_direction_accuracy**: Average direction prediction accuracy (0-100%)
    - **avg_rmse**: Average Root Mean Squared Error

    **Requires authentication.**
    """,
    response_description="List of models with summary metrics",
    responses={
        200: {
            "description": "Models retrieved successfully",
            "content": {
                "application/json": {
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
                        ]
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated",
            "content": {
                "application/json": {
                    "example": {"detail": "Not authenticated"}
                }
            }
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {"detail": "Error loading model data"}
                }
            }
        }
    }
)
async def get_models(current_user: User = Depends(get_current_user)):
    """
    Get list of available models with their metrics.

    Aggregates metrics across all horizons to provide summary statistics
    for each model.

    Args:
        current_user: Authenticated user (injected by dependency)

    Returns:
        ModelResponse with list of models and their metrics

    Raises:
        HTTPException: On data loading errors
    """
    try:
        df = get_csv_data()

        if df is not None:
            # Aggregate metrics by model
            models = []
            for model_name in df["model_name"].unique():
                model_df = df[df["model_name"] == model_name]

                model_info = {
                    "name": model_name,
                    "horizons": sorted(model_df["horizon"].unique().tolist()),
                    "avg_direction_accuracy": None,
                    "avg_rmse": None,
                }

                # Add metrics if available
                if "direction_accuracy" in model_df.columns:
                    model_info["avg_direction_accuracy"] = float(model_df["direction_accuracy"].mean())
                if "rmse" in model_df.columns:
                    model_info["avg_rmse"] = float(model_df["rmse"].mean())

                models.append(model_info)

            return {"models": models}
        else:
            return {"models": []}

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{model_name}",
    response_model=ModelDetailResponse,
    summary="Get detailed model information",
    description="""
    Retrieve detailed information about a specific model, including
    performance metrics broken down by forecast horizon.

    Returns:
    - **name**: Model identifier
    - **metrics_by_horizon**: Performance metrics for each horizon
    - **total_forecasts**: Total number of forecasts made by this model

    **Requires authentication.**
    """,
    response_description="Detailed model information with per-horizon metrics",
    responses={
        200: {
            "description": "Model details retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "name": "ridge",
                        "metrics_by_horizon": [
                            {"horizon": 5, "direction_accuracy": 72.5, "rmse": 42.3},
                            {"horizon": 10, "direction_accuracy": 68.2, "rmse": 55.1}
                        ],
                        "total_forecasts": 1250
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Model 'unknown_model' not found"}
                }
            }
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def get_model_details(
    model_name: str = Path(
        ...,
        description="Model name/identifier",
        example="ridge"
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed information about a specific model.

    Retrieves performance metrics broken down by each supported
    forecast horizon.

    Args:
        model_name: Model identifier to look up
        current_user: Authenticated user (injected by dependency)

    Returns:
        ModelDetailResponse with detailed metrics

    Raises:
        HTTPException: 404 if model not found, 500 on errors
    """
    try:
        df = get_csv_data()

        if df is not None:
            model_df = df[df["model_name"] == model_name]

            if model_df.empty:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            # Metrics by horizon
            metrics_by_horizon = []
            for horizon in sorted(model_df["horizon"].unique()):
                h_df = model_df[model_df["horizon"] == horizon]

                metrics = {"horizon": int(horizon)}

                if "direction_accuracy" in h_df.columns:
                    metrics["direction_accuracy"] = float(h_df["direction_accuracy"].mean())
                if "rmse" in h_df.columns:
                    metrics["rmse"] = float(h_df["rmse"].mean())
                if "mae" in h_df.columns:
                    metrics["mae"] = float(h_df["mae"].mean())
                if "r2" in h_df.columns:
                    metrics["r2"] = float(h_df["r2"].mean())

                metrics_by_horizon.append(metrics)

            return {
                "name": model_name,
                "metrics_by_horizon": metrics_by_horizon,
                "total_forecasts": len(model_df),
            }
        else:
            raise HTTPException(status_code=404, detail="No data available")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model details: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{model_name}/comparison",
    response_model=ModelComparison,
    summary="Compare model against all others",
    description="""
    Compare a specific model against all other available models.

    Returns a ranked list of all models sorted by direction accuracy,
    showing how the selected model performs relative to others.

    For each model in the comparison:
    - **model**: Model name
    - **is_selected**: Whether this is the requested model
    - **avg_da**: Average direction accuracy (0-100%)
    - **avg_rmse**: Average RMSE
    - **rank**: Position in the ranking (1 = best)

    **Requires authentication.**
    """,
    response_description="Model comparison with rankings",
    responses={
        200: {
            "description": "Comparison retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "selected_model": "ridge",
                        "comparison": [
                            {"model": "xgboost", "is_selected": False, "avg_da": 71.2, "avg_rmse": 48.7, "rank": 1},
                            {"model": "ridge", "is_selected": True, "avg_da": 68.5, "avg_rmse": 52.3, "rank": 2}
                        ],
                        "best_model": "xgboost",
                        "model_count": 5
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "Model not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Model 'unknown_model' not found"}
                }
            }
        },
        500: {
            "description": "Internal server error"
        }
    }
)
async def compare_model(
    model_name: str = Path(
        ...,
        description="Model name to compare",
        example="ridge"
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Compare a model against all other models.

    Creates a ranked comparison of all models sorted by direction
    accuracy, highlighting the selected model's position.

    Args:
        model_name: Model to compare
        current_user: Authenticated user (injected by dependency)

    Returns:
        ModelComparison with ranked model list

    Raises:
        HTTPException: 404 if model not found, 500 on errors
    """
    try:
        df = get_csv_data()

        if df is not None:
            if model_name not in df["model_name"].unique():
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

            # Calculate average DA for each model
            comparison = []
            for m in df["model_name"].unique():
                m_df = df[df["model_name"] == m]

                comp = {
                    "model": m,
                    "is_selected": m == model_name,
                }

                if "direction_accuracy" in m_df.columns:
                    comp["avg_da"] = float(m_df["direction_accuracy"].mean())
                if "rmse" in m_df.columns:
                    comp["avg_rmse"] = float(m_df["rmse"].mean())

                comparison.append(comp)

            # Sort by DA and add ranks
            if comparison and "avg_da" in comparison[0]:
                comparison.sort(key=lambda x: x.get("avg_da", 0), reverse=True)
                for i, comp in enumerate(comparison):
                    comp["rank"] = i + 1

            best_model = comparison[0]["model"] if comparison else None

            return {
                "selected_model": model_name,
                "comparison": comparison,
                "best_model": best_model,
                "model_count": len(comparison),
            }
        else:
            raise HTTPException(status_code=404, detail="No data available")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/{model_name}/metrics/{horizon}",
    summary="Get model metrics for a specific horizon",
    description="""
    Retrieve detailed metrics for a specific model and horizon combination.

    Returns performance metrics including:
    - Direction accuracy
    - RMSE (Root Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R-squared

    **Requires authentication.**
    """,
    response_description="Model metrics for the specified horizon",
    responses={
        200: {
            "description": "Metrics retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "model": "ridge",
                        "horizon": 5,
                        "direction_accuracy": 72.5,
                        "rmse": 42.3,
                        "mae": 35.1,
                        "r2": 0.85,
                        "sample_count": 250
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "Model or horizon not found"
        }
    }
)
async def get_model_horizon_metrics(
    model_name: str = Path(
        ...,
        description="Model name",
        example="ridge"
    ),
    horizon: int = Path(
        ...,
        description="Forecast horizon in days",
        example=5,
        ge=1,
        le=60
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get metrics for a specific model and horizon.

    Args:
        model_name: Model identifier
        horizon: Forecast horizon in days
        current_user: Authenticated user (injected by dependency)

    Returns:
        Dictionary with detailed metrics

    Raises:
        HTTPException: 404 if model/horizon not found
    """
    try:
        df = get_csv_data()

        if df is None:
            raise HTTPException(status_code=404, detail="No data available")

        # Filter for model and horizon
        filtered_df = df[(df["model_name"] == model_name) & (df["horizon"] == horizon)]

        if filtered_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for model '{model_name}' with horizon {horizon}"
            )

        result = {
            "model": model_name,
            "horizon": horizon,
            "sample_count": len(filtered_df),
        }

        # Add available metrics
        metric_columns = ["direction_accuracy", "rmse", "mae", "r2", "mape"]
        for col in metric_columns:
            if col in filtered_df.columns:
                result[col] = float(filtered_df[col].mean())

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting horizon metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/ranking",
    summary="Get model rankings",
    description="""
    Get a ranked list of all models by a specific metric.

    Supported metrics:
    - **direction_accuracy** (default): Percentage of correct direction predictions
    - **rmse**: Root Mean Squared Error (lower is better)
    - **mae**: Mean Absolute Error (lower is better)

    Optionally filter by a specific horizon.

    **Requires authentication.**
    """,
    response_description="Ranked list of models",
    responses={
        200: {
            "description": "Ranking retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "metric": "direction_accuracy",
                        "horizon": None,
                        "rankings": [
                            {"rank": 1, "model": "xgboost", "score": 71.2},
                            {"rank": 2, "model": "lightgbm", "score": 70.1},
                            {"rank": 3, "model": "ridge", "score": 68.5}
                        ]
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        }
    }
)
async def get_model_rankings(
    metric: str = Query(
        "direction_accuracy",
        description="Metric to rank by",
        example="direction_accuracy"
    ),
    horizon: Optional[int] = Query(
        None,
        description="Filter by specific horizon (optional)",
        example=5,
        ge=1,
        le=60
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get model rankings by a specific metric.

    Args:
        metric: Metric to use for ranking
        horizon: Optional horizon filter
        current_user: Authenticated user (injected by dependency)

    Returns:
        Dictionary with rankings
    """
    try:
        df = get_csv_data()

        if df is None:
            return {
                "metric": metric,
                "horizon": horizon,
                "rankings": []
            }

        # Filter by horizon if specified
        if horizon is not None:
            df = df[df["horizon"] == horizon]

        if df.empty or metric not in df.columns:
            return {
                "metric": metric,
                "horizon": horizon,
                "rankings": []
            }

        # Calculate average metric per model
        model_scores = []
        for model_name in df["model_name"].unique():
            model_df = df[df["model_name"] == model_name]
            score = float(model_df[metric].mean())
            model_scores.append({"model": model_name, "score": score})

        # Sort (reverse for accuracy, normal for error metrics)
        reverse = metric in ["direction_accuracy", "r2"]
        model_scores.sort(key=lambda x: x["score"], reverse=reverse)

        # Add ranks
        rankings = []
        for i, item in enumerate(model_scores):
            rankings.append({
                "rank": i + 1,
                "model": item["model"],
                "score": round(item["score"], 4)
            })

        return {
            "metric": metric,
            "horizon": horizon,
            "rankings": rankings
        }

    except Exception as e:
        logger.error(f"Error getting rankings: {e}")
        raise HTTPException(status_code=500, detail=str(e))
