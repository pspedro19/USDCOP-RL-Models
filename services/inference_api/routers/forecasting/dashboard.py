"""
Dashboard Router
================

Aggregate endpoint for dashboard data.

@version 1.0.0
"""

from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import pandas as pd
import os
import logging

from services.inference_api.contracts.forecasting import DashboardResponse

router = APIRouter()
logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get('FORECASTING_DATA_PATH', 'data/processed')


def get_dashboard_data() -> Dict[str, Any]:
    """
    Load all dashboard data in one call.

    Returns forecasts, consensus, and metrics in a single response
    to minimize frontend API calls.
    """
    # Find CSV
    possible_paths = [
        os.path.join(DATA_PATH, 'bi', 'bi_dashboard_unified.csv'),
        os.path.join(DATA_PATH, 'bi_dashboard', 'bi_dashboard_unified.csv'),
        os.path.join('public', 'forecasting', 'bi_dashboard_unified.csv'),
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            break

    if not csv_path:
        return {
            "source": "none",
            "forecasts": [],
            "consensus": [],
            "metrics": [],
            "last_update": datetime.now().isoformat(),
        }

    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Dashboard loaded {len(df)} rows from {csv_path}")

        # Get forecasts (all data)
        forecasts = df.to_dict(orient="records")

        # Calculate consensus by horizon
        consensus = []
        if "horizon" in df.columns and "direction" in df.columns:
            for horizon in sorted(df["horizon"].unique()):
                h_df = df[df["horizon"] == horizon]
                bullish = len(h_df[h_df["direction"] == "UP"])
                bearish = len(h_df[h_df["direction"] == "DOWN"])
                total = len(h_df)

                c = {
                    "horizon_id": int(horizon),
                    "bullish_count": bullish,
                    "bearish_count": bearish,
                    "total_models": total,
                    "consensus_direction": "UP" if bullish > bearish else "DOWN",
                    "agreement_pct": max(bullish, bearish) / total * 100 if total > 0 else 0,
                }

                if "predicted_price" in h_df.columns:
                    c["avg_predicted_price"] = float(h_df["predicted_price"].mean())

                consensus.append(c)

        # Calculate metrics summary by model
        metrics = []
        if "model_name" in df.columns:
            for model in df["model_name"].unique():
                m_df = df[df["model_name"] == model]

                m = {
                    "model_id": model,
                    "sample_count": len(m_df),
                }

                metric_cols = ["direction_accuracy", "rmse", "mae", "r2"]
                for col in metric_cols:
                    if col in m_df.columns:
                        m[f"avg_{col}"] = float(m_df[col].mean())

                metrics.append(m)

        return {
            "source": "csv",
            "forecasts": forecasts,
            "consensus": consensus,
            "metrics": metrics,
            "last_update": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error loading dashboard data: {e}")
        return {
            "source": "none",
            "forecasts": [],
            "consensus": [],
            "metrics": [],
            "error": str(e),
            "last_update": datetime.now().isoformat(),
        }


@router.get(
    "/dashboard",
    response_model=DashboardResponse,
    summary="Get complete dashboard data",
    description="""
    Retrieve all data needed for the forecasting dashboard in a single request.

    Includes:
    - Latest forecasts for all models/horizons
    - Consensus direction by horizon
    - Model performance metrics summary

    This endpoint is optimized to minimize frontend API calls.
    """,
)
async def get_dashboard():
    """Get all dashboard data in one response."""
    return get_dashboard_data()


@router.get(
    "/health",
    summary="Forecasting service health check",
)
async def health_check():
    """Check forecasting service health."""
    csv_path = None
    for path in [
        os.path.join(DATA_PATH, 'bi', 'bi_dashboard_unified.csv'),
        os.path.join(DATA_PATH, 'bi_dashboard', 'bi_dashboard_unified.csv'),
        os.path.join('public', 'forecasting', 'bi_dashboard_unified.csv'),
    ]:
        if os.path.exists(path):
            csv_path = path
            break

    return {
        "status": "healthy" if csv_path else "degraded",
        "data_source": csv_path or "not_found",
        "timestamp": datetime.now().isoformat(),
    }
