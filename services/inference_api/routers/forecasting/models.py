"""
Models Router
=============

Endpoints for ML model information and metrics.

@version 1.0.0
"""

from fastapi import APIRouter, Query, HTTPException, Path
from typing import Optional, List, Dict, Any
import pandas as pd
import os
import logging

from services.inference_api.contracts.forecasting import (
    ModelListResponse,
    ModelDetailResponse,
    ModelRankingResponse,
    ModelInfo,
    ModelType,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Model definitions (SSOT)
MODEL_DEFINITIONS = {
    "ridge": ModelInfo(
        model_id="ridge",
        model_name="Ridge Regression",
        model_type=ModelType.LINEAR,
        requires_scaling=True,
        supports_early_stopping=False,
    ),
    "bayesian_ridge": ModelInfo(
        model_id="bayesian_ridge",
        model_name="Bayesian Ridge",
        model_type=ModelType.LINEAR,
        requires_scaling=True,
        supports_early_stopping=False,
    ),
    "ard": ModelInfo(
        model_id="ard",
        model_name="ARD Regression",
        model_type=ModelType.LINEAR,
        requires_scaling=True,
        supports_early_stopping=False,
    ),
    "xgboost_pure": ModelInfo(
        model_id="xgboost_pure",
        model_name="XGBoost",
        model_type=ModelType.BOOSTING,
        requires_scaling=False,
        supports_early_stopping=True,
    ),
    "lightgbm_pure": ModelInfo(
        model_id="lightgbm_pure",
        model_name="LightGBM",
        model_type=ModelType.BOOSTING,
        requires_scaling=False,
        supports_early_stopping=True,
    ),
    "catboost_pure": ModelInfo(
        model_id="catboost_pure",
        model_name="CatBoost",
        model_type=ModelType.BOOSTING,
        requires_scaling=False,
        supports_early_stopping=True,
    ),
    "hybrid_xgboost": ModelInfo(
        model_id="hybrid_xgboost",
        model_name="XGBoost Hybrid",
        model_type=ModelType.HYBRID,
        requires_scaling=True,
        supports_early_stopping=True,
    ),
    "hybrid_lightgbm": ModelInfo(
        model_id="hybrid_lightgbm",
        model_name="LightGBM Hybrid",
        model_type=ModelType.HYBRID,
        requires_scaling=True,
        supports_early_stopping=True,
    ),
    "hybrid_catboost": ModelInfo(
        model_id="hybrid_catboost",
        model_name="CatBoost Hybrid",
        model_type=ModelType.HYBRID,
        requires_scaling=True,
        supports_early_stopping=True,
    ),
}

DATA_PATH = os.environ.get('FORECASTING_DATA_PATH', 'data/processed')


def get_csv_data() -> Optional[pd.DataFrame]:
    """Load CSV data."""
    possible_paths = [
        os.path.join(DATA_PATH, 'bi', 'bi_dashboard_unified.csv'),
        os.path.join(DATA_PATH, 'bi_dashboard', 'bi_dashboard_unified.csv'),
        os.path.join('public', 'forecasting', 'bi_dashboard_unified.csv'),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return pd.read_csv(path)
    return None


@router.get(
    "/",
    response_model=ModelListResponse,
    summary="Get all available models",
)
async def get_models():
    """List all available forecasting models with metrics."""
    df = get_csv_data()

    models = []
    for model_id, model_info in MODEL_DEFINITIONS.items():
        model_dict = model_info.model_dump()

        # Add metrics from CSV if available
        if df is not None and "model_name" in df.columns:
            model_df = df[df["model_name"] == model_id]
            if not model_df.empty:
                if "direction_accuracy" in model_df.columns:
                    model_dict["avg_direction_accuracy"] = float(model_df["direction_accuracy"].mean())
                if "rmse" in model_df.columns:
                    model_dict["avg_rmse"] = float(model_df["rmse"].mean())
                if "horizon" in model_df.columns:
                    model_dict["horizons"] = sorted(model_df["horizon"].unique().tolist())

        models.append(model_dict)

    return {"models": models, "count": len(models)}


@router.get(
    "/{model_id}",
    summary="Get detailed model information",
)
async def get_model_detail(
    model_id: str = Path(..., description="Model identifier"),
):
    """Get detailed information about a specific model."""
    if model_id not in MODEL_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    model_info = MODEL_DEFINITIONS[model_id]

    df = get_csv_data()
    metrics_by_horizon = []
    total_forecasts = 0

    if df is not None and "model_name" in df.columns:
        model_df = df[df["model_name"] == model_id]
        total_forecasts = len(model_df)

        if "horizon" in model_df.columns:
            for horizon in sorted(model_df["horizon"].unique()):
                h_df = model_df[model_df["horizon"] == horizon]

                metrics = {
                    "model_id": model_id,
                    "horizon_id": int(horizon),
                    "sample_count": len(h_df),
                }

                metric_cols = ["direction_accuracy", "rmse", "mae", "mape", "r2"]
                for col in metric_cols:
                    if col in h_df.columns:
                        metrics[col] = float(h_df[col].mean())

                metrics_by_horizon.append(metrics)

    return {
        "model": model_info,
        "metrics_by_horizon": metrics_by_horizon,
        "total_forecasts": total_forecasts,
    }


@router.get(
    "/{model_id}/comparison",
    summary="Compare model against all others",
)
async def compare_model(
    model_id: str = Path(..., description="Model to compare"),
):
    """Get comparison of model against all others."""
    if model_id not in MODEL_DEFINITIONS:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

    df = get_csv_data()

    if df is None or "model_name" not in df.columns:
        return {
            "selected_model": model_id,
            "comparison": [],
            "best_model": None,
            "model_count": 0,
        }

    # Calculate metrics per model
    comparison = []
    for mid in df["model_name"].unique():
        m_df = df[df["model_name"] == mid]

        comp = {
            "model_id": mid,
            "is_selected": mid == model_id,
        }

        if "direction_accuracy" in m_df.columns:
            comp["avg_direction_accuracy"] = float(m_df["direction_accuracy"].mean())
        if "rmse" in m_df.columns:
            comp["avg_rmse"] = float(m_df["rmse"].mean())

        comparison.append(comp)

    # Sort by DA and add ranks
    if comparison and "avg_direction_accuracy" in comparison[0]:
        comparison.sort(key=lambda x: x.get("avg_direction_accuracy", 0), reverse=True)
        for i, c in enumerate(comparison):
            c["rank"] = i + 1

    best_model = comparison[0]["model_id"] if comparison else None

    return {
        "selected_model": model_id,
        "comparison": comparison,
        "best_model": best_model,
        "model_count": len(comparison),
    }


@router.get(
    "/ranking",
    response_model=ModelRankingResponse,
    summary="Get model rankings by metric",
)
async def get_model_rankings(
    metric: str = Query("direction_accuracy", description="Metric to rank by"),
    horizon: Optional[int] = Query(None, ge=1, le=60, description="Filter by horizon"),
):
    """Get ranked list of models by specified metric."""
    df = get_csv_data()

    if df is None or metric not in df.columns:
        return {"metric": metric, "horizon": horizon, "rankings": []}

    if horizon is not None:
        df = df[df["horizon"] == horizon]

    if df.empty:
        return {"metric": metric, "horizon": horizon, "rankings": []}

    # Calculate average metric per model
    rankings = []
    for model_id in df["model_name"].unique():
        model_df = df[df["model_name"] == model_id]
        score = float(model_df[metric].mean())
        rankings.append({"model_id": model_id, "score": score})

    # Sort (descending for accuracy, ascending for errors)
    reverse = metric in ["direction_accuracy", "r2"]
    rankings.sort(key=lambda x: x["score"], reverse=reverse)

    # Add ranks
    result = []
    for i, item in enumerate(rankings):
        result.append({
            "model_id": item["model_id"],
            "is_selected": False,
            "avg_direction_accuracy": item["score"] if metric == "direction_accuracy" else 0,
            "avg_rmse": item["score"] if metric == "rmse" else 0,
            "rank": i + 1,
        })

    return {"metric": metric, "horizon": horizon, "rankings": result}
