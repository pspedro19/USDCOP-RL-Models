"""
Forecasts Router
================

Endpoints for accessing forecast data.

@version 1.0.0
"""

from fastapi import APIRouter, Query, HTTPException, Depends, Path
from typing import Optional, List, Dict, Any
from datetime import date
import pandas as pd
import os
import logging

from services.inference_api.contracts.forecasting import (
    ForecastListResponse,
    LatestForecastResponse,
    ConsensusResponse,
    WeekForecastResponse,
    HorizonForecastResponse,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Data paths - will be configured via environment
DATA_PATH = os.environ.get('FORECASTING_DATA_PATH', 'data/processed')
CSV_FILENAME = 'bi_dashboard_unified.csv'


def get_csv_path() -> Optional[str]:
    """Find the dashboard CSV file."""
    possible_paths = [
        os.path.join(DATA_PATH, 'bi', CSV_FILENAME),
        os.path.join(DATA_PATH, 'bi_dashboard', CSV_FILENAME),
        os.path.join(DATA_PATH, CSV_FILENAME),
        # Fallback to public folder for development
        os.path.join('public', 'forecasting', CSV_FILENAME),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None


def load_forecasts_from_csv(
    model: Optional[str] = None,
    horizon: Optional[int] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Load forecasts from CSV file."""
    csv_path = get_csv_path()

    if not csv_path:
        logger.warning("Forecast CSV not found")
        return {"source": "none", "count": 0, "data": []}

    try:
        df = pd.read_csv(csv_path)

        # Apply filters
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
    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return {"source": "none", "count": 0, "data": [], "error": str(e)}


@router.get(
    "/",
    response_model=ForecastListResponse,
    summary="Get forecasts with optional filtering",
    description="Retrieve USD/COP forecasts with optional model/horizon/week filters.",
)
async def get_forecasts(
    model: Optional[str] = Query(None, description="Filter by model name"),
    horizon: Optional[int] = Query(None, ge=1, le=60, description="Filter by horizon"),
    week: Optional[int] = Query(None, ge=1, le=53, description="Filter by week"),
    year: Optional[int] = Query(None, ge=2020, le=2100, description="Filter by year"),
    limit: int = Query(100, ge=1, le=1000, description="Max results"),
):
    """Get forecasts from database or CSV fallback."""
    # Try database first (would be implemented with proper DB connection)
    # For now, use CSV fallback

    result = load_forecasts_from_csv(model, horizon, limit)

    # Apply week/year filters if data supports it
    if result["data"] and (week or year):
        df = pd.DataFrame(result["data"])
        if "inference_week" in df.columns and week:
            df = df[df["inference_week"] == week]
        if "inference_year" in df.columns and year:
            df = df[df["inference_year"] == year]
        result["data"] = df.head(limit).to_dict(orient="records")
        result["count"] = len(result["data"])

    return result


@router.get(
    "/latest",
    response_model=LatestForecastResponse,
    summary="Get latest forecasts per model/horizon",
)
async def get_latest_forecasts():
    """Get the most recent forecast for each model/horizon combination."""
    result = load_forecasts_from_csv(limit=1000)

    if not result["data"]:
        return result

    df = pd.DataFrame(result["data"])

    # Get latest per model/horizon
    if "model_name" in df.columns and "horizon" in df.columns:
        if "inference_date" in df.columns:
            df["inference_date"] = pd.to_datetime(df["inference_date"])
            df = df.sort_values("inference_date", ascending=False)

        df = df.drop_duplicates(subset=["model_name", "horizon"], keep="first")

    return {
        "source": result["source"],
        "count": len(df),
        "data": df.to_dict(orient="records"),
    }


@router.get(
    "/consensus",
    response_model=ConsensusResponse,
    summary="Get consensus forecasts by horizon",
)
async def get_consensus():
    """Get aggregated consensus forecasts across all models."""
    result = load_forecasts_from_csv(limit=1000)

    if not result["data"]:
        return {"source": "none", "count": 0, "data": []}

    df = pd.DataFrame(result["data"])

    # Calculate consensus by horizon
    consensus_data = []

    if "horizon" in df.columns and "direction" in df.columns:
        for horizon in sorted(df["horizon"].unique()):
            h_df = df[df["horizon"] == horizon]

            bullish = len(h_df[h_df["direction"] == "UP"])
            bearish = len(h_df[h_df["direction"] == "DOWN"])
            total = len(h_df)

            consensus = {
                "horizon_id": int(horizon),
                "horizon_label": f"{int(horizon)} days",
                "bullish_count": bullish,
                "bearish_count": bearish,
                "total_models": total,
                "consensus_direction": "UP" if bullish > bearish else "DOWN",
                "agreement_pct": max(bullish, bearish) / total * 100 if total > 0 else 0,
            }

            if "predicted_price" in h_df.columns:
                consensus["avg_predicted_price"] = float(h_df["predicted_price"].mean())
                consensus["median_predicted_price"] = float(h_df["predicted_price"].median())
                consensus["std_predicted_price"] = float(h_df["predicted_price"].std())

            consensus_data.append(consensus)

    return {
        "source": result["source"],
        "count": len(consensus_data),
        "data": consensus_data,
    }


@router.get(
    "/by-week/{year}/{week}",
    response_model=WeekForecastResponse,
    summary="Get forecasts for a specific week",
)
async def get_forecasts_by_week(
    year: int = Path(..., ge=2020, le=2100),
    week: int = Path(..., ge=1, le=53),
):
    """Get all forecasts for a specific week."""
    result = load_forecasts_from_csv(limit=1000)

    data = result["data"]
    if data:
        df = pd.DataFrame(data)
        if "inference_week" in df.columns and "inference_year" in df.columns:
            df = df[(df["inference_week"] == week) & (df["inference_year"] == year)]
            data = df.to_dict(orient="records")

    return {
        "year": year,
        "week": week,
        "forecasts": data,
        "minio_files": [],  # Would be populated from MinIO
        "minio_path": None,
        "source": result["source"],
        "count": len(data),
    }


@router.get(
    "/by-horizon/{horizon}",
    response_model=HorizonForecastResponse,
    summary="Get forecasts for a specific horizon",
)
async def get_forecasts_by_horizon(
    horizon: int = Path(..., ge=1, le=60),
):
    """Get all forecasts for a specific horizon."""
    result = load_forecasts_from_csv(horizon=horizon, limit=500)

    return {
        "horizon": horizon,
        "source": result["source"],
        "count": result["count"],
        "data": result["data"],
    }
