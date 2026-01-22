"""
Forecast endpoints - Read forecasts from PostgreSQL and MinIO.

This module provides REST API endpoints for accessing USD/COP exchange rate
forecasts, including filtering, latest predictions, consensus views, and
historical data by week or horizon.

All endpoints require authentication except where noted.
"""
from fastapi import APIRouter, Query, HTTPException, Depends, Path
from fastapi.responses import JSONResponse
from typing import Optional, List, Dict, Any
from datetime import date, datetime
import pandas as pd
import json
import os
import logging

from ..auth.dependencies import get_current_user, User
from ..schemas.forecasts import (
    ForecastResponse,
    LatestForecastResponse,
    ConsensusResponse,
    WeekForecastResponse,
    HorizonForecastResponse,
    DashboardResponse,
)
from ..config import settings, get_db_connection, get_minio_client

router = APIRouter()
logger = logging.getLogger(__name__)


async def get_forecasts_from_csv(
    model: Optional[str],
    horizon: Optional[int],
    limit: int
) -> Dict[str, Any]:
    """
    Fallback: read forecasts from CSV files.

    Args:
        model: Filter by model name
        horizon: Filter by horizon
        limit: Maximum results

    Returns:
        Dictionary with source, count, and data
    """
    csv_path = os.path.join(settings.OUTPUTS_PATH, "bi", "bi_dashboard_unified.csv")

    if not os.path.exists(csv_path):
        # Try alternative path
        csv_path = os.path.join(settings.OUTPUTS_PATH, "bi_dashboard", "bi_dashboard_unified.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if model:
            df = df[df["model_name"] == model]
        if horizon:
            df = df[df["horizon"] == horizon]

        df = df.head(limit)

        return {
            "source": "csv",
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    return {"source": "none", "count": 0, "data": [], "message": "No forecast data available"}


@router.get(
    "/",
    response_model=ForecastResponse,
    summary="Get forecasts with optional filtering",
    description="""
    Retrieve USD/COP exchange rate forecasts with optional filtering.

    Supports filtering by:
    - **model**: Filter by specific ML model (ridge, xgboost, lightgbm, catboost, bayesian_ridge)
    - **horizon**: Filter by forecast horizon in days (5, 10, 20, 40)
    - **week**: Filter by inference week number (1-52)
    - **year**: Filter by year

    Returns forecasts ordered by inference date (most recent first).
    Falls back to CSV data if PostgreSQL is unavailable.

    **Requires authentication.**
    """,
    response_description="List of forecasts matching the filter criteria",
    responses={
        200: {
            "description": "Forecasts retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "source": "postgresql",
                        "count": 50,
                        "data": [
                            {
                                "inference_date": "2024-01-15",
                                "model_name": "ridge",
                                "horizon": 5,
                                "predicted_price": 4275.25,
                                "direction": "UP"
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
                    "example": {"detail": "Database connection failed"}
                }
            }
        }
    }
)
async def get_forecasts(
    model: Optional[str] = Query(
        None,
        description="Filter by model name (e.g., 'ridge', 'xgboost', 'lightgbm', 'catboost', 'bayesian_ridge')",
        example="ridge"
    ),
    horizon: Optional[int] = Query(
        None,
        description="Filter by forecast horizon in days",
        example=5,
        ge=1,
        le=60
    ),
    week: Optional[int] = Query(
        None,
        description="Filter by inference week number (1-52)",
        example=25,
        ge=1,
        le=53
    ),
    year: Optional[int] = Query(
        None,
        description="Filter by year",
        example=2024,
        ge=2020,
        le=2030
    ),
    limit: int = Query(
        100,
        description="Maximum number of results to return",
        example=100,
        ge=1,
        le=1000
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get forecasts from PostgreSQL with optional filtering.

    Queries the bi.fact_forecasts table with optional filters and joins
    with dimension tables for additional metadata. Falls back to CSV
    files if the database is unavailable.

    Args:
        model: Filter by model name
        horizon: Filter by forecast horizon in days
        week: Filter by inference week number
        year: Filter by year
        limit: Maximum results to return
        current_user: Authenticated user (injected by dependency)

    Returns:
        ForecastResponse with source, count, and data list
    """
    try:
        conn = get_db_connection()

        # Build query with optional filters
        query = """
            SELECT
                f.inference_date,
                f.inference_week,
                f.inference_year,
                f.target_date,
                f.model_id as model_name,
                f.horizon_id as horizon,
                f.base_price,
                f.predicted_price,
                f.predicted_return_pct,
                f.price_change,
                f.direction,
                f.signal,
                f.actual_price,
                f.direction_correct,
                f.minio_week_path,
                m.model_name as model_display_name,
                m.model_type,
                h.horizon_label,
                h.horizon_category
            FROM bi.fact_forecasts f
            LEFT JOIN bi.dim_models m ON f.model_id = m.model_id
            LEFT JOIN bi.dim_horizons h ON f.horizon_id = h.horizon_id
            WHERE 1=1
        """
        params = []

        if model:
            query += " AND f.model_id = %s"
            params.append(model)
        if horizon:
            query += " AND f.horizon_id = %s"
            params.append(horizon)
        if week:
            query += " AND f.inference_week = %s"
            params.append(week)
        if year:
            query += " AND f.inference_year = %s"
            params.append(year)

        query += " ORDER BY f.inference_date DESC, f.model_id, f.horizon_id"
        query += f" LIMIT {limit}"

        df = pd.read_sql(query, conn, params=params if params else None)
        conn.close()

        if df.empty:
            # Fallback to CSV
            return await get_forecasts_from_csv(model, horizon, limit)

        return {
            "source": "postgresql",
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        logger.warning(f"PostgreSQL error: {e}, falling back to CSV")
        return await get_forecasts_from_csv(model, horizon, limit)


@router.get(
    "/latest",
    response_model=LatestForecastResponse,
    summary="Get the most recent forecasts",
    description="""
    Retrieve the most recent forecast for each model and horizon combination.

    Returns one forecast per model-horizon pair, representing the latest
    prediction available for each combination.

    **Requires authentication.**
    """,
    response_description="Latest forecasts for all model/horizon combinations",
    responses={
        200: {
            "description": "Latest forecasts retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "source": "postgresql",
                        "count": 20,
                        "data": [
                            {
                                "model_name": "ridge",
                                "horizon": 5,
                                "predicted_price": 4275.25,
                                "direction": "UP"
                            }
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
async def get_latest_forecasts(current_user: User = Depends(get_current_user)):
    """
    Get the most recent forecasts for all models and horizons.

    Uses DISTINCT ON to select only the latest forecast for each
    model-horizon combination.

    Args:
        current_user: Authenticated user (injected by dependency)

    Returns:
        LatestForecastResponse with latest forecasts
    """
    try:
        conn = get_db_connection()

        query = """
            SELECT DISTINCT ON (model_id, horizon_id)
                f.*,
                m.model_name as model_display_name,
                m.model_type,
                h.horizon_label
            FROM bi.fact_forecasts f
            LEFT JOIN bi.dim_models m ON f.model_id = m.model_id
            LEFT JOIN bi.dim_horizons h ON f.horizon_id = h.horizon_id
            ORDER BY model_id, horizon_id, inference_date DESC
        """

        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            # Fallback
            return await get_forecasts_from_csv(None, None, 100)

        return {
            "source": "postgresql",
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        logger.warning(f"Error: {e}")
        return await get_forecasts_from_csv(None, None, 100)


@router.get(
    "/consensus",
    response_model=ConsensusResponse,
    summary="Get consensus forecasts by horizon",
    description="""
    Retrieve consensus forecasts aggregated across all models for each horizon.

    Consensus represents the average prediction across all models, providing
    a more robust forecast estimate.

    **Requires authentication.**
    """,
    response_description="Consensus forecasts by horizon",
    responses={
        200: {
            "description": "Consensus forecasts retrieved successfully"
        },
        401: {
            "description": "Not authenticated"
        },
        500: {
            "description": "Database error"
        }
    }
)
async def get_consensus(current_user: User = Depends(get_current_user)):
    """
    Get consensus forecasts by horizon.

    Queries the bi.fact_consensus table for aggregated model predictions.

    Args:
        current_user: Authenticated user (injected by dependency)

    Returns:
        ConsensusResponse with consensus data
    """
    try:
        conn = get_db_connection()

        query = """
            SELECT
                c.*,
                h.horizon_label,
                h.horizon_category
            FROM bi.fact_consensus c
            LEFT JOIN bi.dim_horizons h ON c.horizon_id = h.horizon_id
            ORDER BY c.inference_date DESC, c.horizon_id
            LIMIT 100
        """

        df = pd.read_sql(query, conn)
        conn.close()

        return {
            "source": "postgresql",
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        logger.error(f"Consensus error: {e}")
        return {"source": "none", "count": 0, "data": []}


@router.get(
    "/by-week/{year}/{week}",
    response_model=WeekForecastResponse,
    summary="Get all forecasts for a specific week",
    description="""
    Retrieve all forecasts generated during a specific week, along with
    associated MinIO artifacts (figures, data files).

    Combines data from PostgreSQL (forecast records) and MinIO (associated files).

    **Requires authentication.**
    """,
    response_description="Forecasts and associated files for the specified week",
    responses={
        200: {
            "description": "Week forecasts retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "year": 2024,
                        "week": 3,
                        "forecasts": [],
                        "minio_files": [],
                        "source": "postgresql",
                        "count": 20
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        },
        500: {
            "description": "Error retrieving data",
            "content": {
                "application/json": {
                    "example": {"detail": "Database error"}
                }
            }
        }
    }
)
async def get_forecasts_by_week(
    year: int = Path(
        ...,
        description="Year of the forecasts",
        example=2024,
        ge=2020,
        le=2030
    ),
    week: int = Path(
        ...,
        description="Week number (1-52)",
        example=3,
        ge=1,
        le=53
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get all forecasts for a specific week from PostgreSQL and MinIO.

    Retrieves forecast records from the database and lists associated
    artifact files from MinIO storage.

    Args:
        year: Year of the forecasts
        week: Week number (1-52)
        current_user: Authenticated user (injected by dependency)

    Returns:
        WeekForecastResponse with forecasts and MinIO files

    Raises:
        HTTPException: On database or storage errors
    """
    result = {
        "year": year,
        "week": week,
        "forecasts": [],
        "minio_files": [],
        "source": "postgresql"
    }

    try:
        # 1. Get from PostgreSQL
        conn = get_db_connection()

        query = """
            SELECT
                f.*,
                m.model_name as model_display_name,
                h.horizon_label
            FROM bi.fact_forecasts f
            LEFT JOIN bi.dim_models m ON f.model_id = m.model_id
            LEFT JOIN bi.dim_horizons h ON f.horizon_id = h.horizon_id
            WHERE f.inference_year = %s AND f.inference_week = %s
            ORDER BY f.model_id, f.horizon_id
        """

        df = pd.read_sql(query, conn, params=[year, week])
        conn.close()

        result["forecasts"] = df.to_dict(orient="records")
        result["count"] = len(df)

        # 2. Get files from MinIO
        client = get_minio_client()
        if client:
            try:
                bucket = "forecasts"
                prefix = f"{year}/week{week:02d}/"

                objects = client.list_objects(bucket, prefix=prefix, recursive=True)
                minio_files = []

                for obj in objects:
                    minio_files.append({
                        "name": obj.object_name,
                        "size": obj.size,
                        "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
                        "url": f"http://{settings.MINIO_ENDPOINT}/{bucket}/{obj.object_name}"
                    })

                result["minio_files"] = minio_files
                result["minio_path"] = f"{bucket}/{prefix}"

            except Exception as e:
                logger.warning(f"MinIO error: {e}")

        return result

    except Exception as e:
        logger.error(f"Error getting week data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/by-horizon/{horizon}",
    response_model=HorizonForecastResponse,
    summary="Get all forecasts for a specific horizon",
    description="""
    Retrieve all forecasts for a specific forecast horizon (e.g., 5, 10, 20, 40 days).

    Returns forecasts from all models for the specified horizon, ordered by
    inference date (most recent first).

    **Requires authentication.**
    """,
    response_description="Forecasts for the specified horizon",
    responses={
        200: {
            "description": "Horizon forecasts retrieved successfully"
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "No forecasts found for this horizon"
        }
    }
)
async def get_forecasts_by_horizon(
    horizon: int = Path(
        ...,
        description="Forecast horizon in days (e.g., 5, 10, 20, 40)",
        example=5,
        ge=1,
        le=60
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Get all forecasts for a specific horizon.

    Args:
        horizon: Forecast horizon in days
        current_user: Authenticated user (injected by dependency)

    Returns:
        HorizonForecastResponse with forecasts for the horizon
    """
    try:
        conn = get_db_connection()

        query = """
            SELECT
                f.*,
                m.model_name as model_display_name
            FROM bi.fact_forecasts f
            LEFT JOIN bi.dim_models m ON f.model_id = m.model_id
            WHERE f.horizon_id = %s
            ORDER BY f.inference_date DESC, f.model_id
            LIMIT 100
        """

        df = pd.read_sql(query, conn, params=[horizon])
        conn.close()

        return {
            "horizon": horizon,
            "source": "postgresql",
            "count": len(df),
            "data": df.to_dict(orient="records"),
        }

    except Exception as e:
        logger.warning(f"Error: {e}")
        # Fallback to CSV
        csv_result = await get_forecasts_from_csv(None, horizon, 100)
        return {
            "horizon": horizon,
            "source": csv_result.get("source", "none"),
            "count": csv_result.get("count", 0),
            "data": csv_result.get("data", []),
        }


@router.get(
    "/minio/{year}/{week}/{path:path}",
    summary="Get a specific file from MinIO",
    description="""
    Retrieve a specific file from MinIO storage by path.

    Supports JSON files (returned as parsed JSON) and binary files
    (returned with metadata).

    **Requires authentication.**
    """,
    response_description="File content or metadata",
    responses={
        200: {
            "description": "File retrieved successfully"
        },
        401: {
            "description": "Not authenticated"
        },
        404: {
            "description": "File not found",
            "content": {
                "application/json": {
                    "example": {"detail": "File not found: filename.json"}
                }
            }
        },
        503: {
            "description": "MinIO not available",
            "content": {
                "application/json": {
                    "example": {"detail": "MinIO not available"}
                }
            }
        }
    }
)
async def get_minio_file(
    year: int = Path(..., description="Year", example=2024),
    week: int = Path(..., description="Week number", example=3),
    path: str = Path(..., description="File path within the week folder"),
    current_user: User = Depends(get_current_user),
):
    """
    Get a specific file from MinIO.

    Args:
        year: Year of the file
        week: Week number
        path: File path within the week folder
        current_user: Authenticated user (injected by dependency)

    Returns:
        JSON content or file metadata

    Raises:
        HTTPException: 404 if file not found, 503 if MinIO unavailable
    """
    client = get_minio_client()
    if not client:
        raise HTTPException(status_code=503, detail="MinIO not available")

    try:
        bucket = "forecasts"
        object_name = f"{year}/week{week:02d}/{path}"

        # Get object data
        response = client.get_object(bucket, object_name)
        data = response.read()
        response.close()

        # If JSON, parse and return
        if path.endswith('.json'):
            return json.loads(data.decode('utf-8'))

        # Otherwise return as bytes info
        return {
            "bucket": bucket,
            "object": object_name,
            "size": len(data),
            "content_type": "application/octet-stream"
        }

    except Exception as e:
        logger.error(f"MinIO get error: {e}")
        raise HTTPException(status_code=404, detail=f"File not found: {path}")


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Get complete dashboard data",
    description="""
    Retrieve all data needed for the dashboard view in a single request.

    Includes:
    - **Latest forecasts**: Most recent forecast per model/horizon
    - **Consensus**: Aggregated predictions across models
    - **Metrics**: Model performance metrics

    Reads from CSV file as the primary source (bi_dashboard_unified.csv).

    **Requires authentication.**
    """,
    response_description="Complete dashboard data including forecasts, consensus, and metrics",
    responses={
        200: {
            "description": "Dashboard data retrieved successfully",
            "content": {
                "application/json": {
                    "example": {
                        "source": "csv",
                        "forecasts": [],
                        "consensus": [],
                        "metrics": [],
                        "last_update": "2024-01-15T10:30:00.000000"
                    }
                }
            }
        },
        401: {
            "description": "Not authenticated"
        }
    }
)
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """
    Get complete dashboard data from CSV file.

    Reads directly from bi_dashboard_unified.csv which contains all
    backtest and forward forecast data with proper horizon combinations.

    Args:
        current_user: Authenticated user (injected by dependency)

    Returns:
        DashboardResponse with all dashboard components
    """
    # Read from CSV as primary source (contains all horizons correctly)
    csv_path = os.path.join(settings.OUTPUTS_PATH, "bi", "bi_dashboard_unified.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(settings.OUTPUTS_PATH, "bi_dashboard", "bi_dashboard_unified.csv")

    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Dashboard loaded {len(df)} rows from CSV: {csv_path}")
            return {
                "source": "csv",
                "forecasts": df.to_dict(orient="records"),
                "consensus": [],
                "metrics": [],
                "last_update": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")

    # Fallback to PostgreSQL if CSV not available
    try:
        conn = get_db_connection()

        # Latest forecasts per model/horizon
        forecasts_query = """
            SELECT DISTINCT ON (model_id, horizon_id)
                f.inference_date,
                f.inference_week,
                f.inference_year,
                f.model_id as model_name,
                f.horizon_id as horizon_days,
                f.base_price as current_price,
                f.predicted_price,
                f.predicted_return_pct as price_change_pct,
                f.direction,
                f.signal,
                f.minio_week_path as image_forecast,
                m.avg_direction_accuracy as model_avg_direction_accuracy,
                m.avg_rmse as model_avg_rmse
            FROM bi.fact_forecasts f
            LEFT JOIN bi.dim_models m ON f.model_id = m.model_id
            ORDER BY model_id, horizon_id, inference_date DESC
        """

        df_forecasts = pd.read_sql(forecasts_query, conn)

        # Latest consensus
        consensus_query = """
            SELECT DISTINCT ON (horizon_id)
                c.*
            FROM bi.fact_consensus c
            ORDER BY horizon_id, inference_date DESC
        """

        df_consensus = pd.read_sql(consensus_query, conn)

        # Model metrics
        metrics_query = """
            SELECT
                model_id,
                horizon_id,
                direction_accuracy,
                rmse,
                mae,
                r2
            FROM bi.fact_model_metrics
            WHERE training_date = (SELECT MAX(training_date) FROM bi.fact_model_metrics)
        """

        df_metrics = pd.read_sql(metrics_query, conn)
        conn.close()

        return {
            "source": "postgresql",
            "forecasts": df_forecasts.to_dict(orient="records"),
            "consensus": df_consensus.to_dict(orient="records"),
            "metrics": df_metrics.to_dict(orient="records"),
            "last_update": datetime.now().isoformat()
        }

    except Exception as e:
        logger.warning(f"Dashboard error: {e}")
        return {
            "source": "none",
            "forecasts": [],
            "consensus": [],
            "metrics": [],
            "error": str(e),
            "last_update": datetime.now().isoformat()
        }
