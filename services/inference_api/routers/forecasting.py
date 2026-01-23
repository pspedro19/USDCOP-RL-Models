"""
Forecasting API Router
======================

Endpoints for accessing forecasting predictions and metrics.

Architecture:
    PostgreSQL (Primary) → CSV (Fallback)

    The router first attempts to query PostgreSQL for data.
    If PostgreSQL is unavailable or returns no data, it falls
    back to reading from CSV files.

Endpoints:
    GET /api/v1/forecasting/dashboard       - Dashboard data (latest forecasts)
    GET /api/v1/forecasting/forecasts       - Historical forecasts
    GET /api/v1/forecasting/consensus       - Consensus by horizon
    GET /api/v1/forecasting/models          - Available models
    GET /api/v1/forecasting/metrics         - Model performance metrics

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
Contract: CTR-FORECASTING-001
"""

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# =============================================================================
# SSOT IMPORTS
# =============================================================================

try:
    from src.forecasting.contracts import (
        HORIZONS,
        MODEL_IDS,
        HORIZON_LABELS,
        ForecastDirection,
        EnsembleType,
        FORECASTING_CONTRACT_VERSION,
        FORECASTING_CONTRACT_HASH,
    )
    CONTRACTS_AVAILABLE = True
except ImportError:
    CONTRACTS_AVAILABLE = False
    HORIZONS = (1, 5, 10, 15, 20, 25, 30)
    MODEL_IDS = ("ridge", "bayesian_ridge", "ard", "xgboost_pure", "lightgbm_pure",
                 "catboost_pure", "hybrid_xgboost", "hybrid_lightgbm", "hybrid_catboost")
    HORIZON_LABELS = {1: "1 day", 5: "5 days", 10: "10 days", 15: "15 days",
                      20: "20 days", 25: "25 days", 30: "30 days"}
    FORECASTING_CONTRACT_VERSION = "unknown"
    FORECASTING_CONTRACT_HASH = "unknown"


router = APIRouter(
    prefix="/forecasting",
    tags=["forecasting"],
)

logger = logging.getLogger(__name__)


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ForecastItem(BaseModel):
    """Single forecast prediction."""
    model_id: str
    horizon: int
    inference_date: str
    target_date: Optional[str] = None
    base_price: Optional[float] = None
    predicted_price: Optional[float] = None
    predicted_return_pct: float
    direction: str
    signal: int
    confidence: Optional[float] = None


class ConsensusItem(BaseModel):
    """Consensus for a single horizon."""
    horizon: int
    horizon_label: str
    bullish_count: int
    bearish_count: int
    consensus_direction: str
    consensus_strength: float
    avg_predicted_return: float


class DashboardResponse(BaseModel):
    """Dashboard response with all forecasting data."""
    inference_date: str
    inference_week: int
    inference_year: int
    current_price: Optional[float] = None
    forecasts: List[ForecastItem]
    consensus: List[ConsensusItem]
    ensembles: Dict[str, List[ForecastItem]] = Field(default_factory=dict)
    data_source: str  # "postgresql" or "csv"
    contract_version: str = FORECASTING_CONTRACT_VERSION


class ModelInfo(BaseModel):
    """Model information."""
    model_id: str
    model_name: str
    model_type: str
    description: Optional[str] = None


class ModelMetrics(BaseModel):
    """Model performance metrics."""
    model_id: str
    horizon: int
    direction_accuracy: float
    rmse: Optional[float] = None
    mae: Optional[float] = None
    sample_count: int


# =============================================================================
# DATABASE HELPERS
# =============================================================================

def get_db_connection():
    """Get database connection."""
    import psycopg2
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not configured")
    return psycopg2.connect(database_url)


def query_postgresql_forecasts(
    inference_date: Optional[str] = None,
    limit: int = 100,
) -> Optional[List[Dict[str, Any]]]:
    """
    Query forecasts from PostgreSQL.

    Returns None if query fails (triggers CSV fallback).
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if inference_date:
            cur.execute("""
                SELECT
                    model_id, horizon, inference_date, base_price,
                    predicted_return_pct, direction, signal, confidence
                FROM bi.fact_forecasts
                WHERE inference_date = %s::date
                ORDER BY model_id, horizon
            """, (inference_date,))
        else:
            cur.execute("""
                SELECT
                    model_id, horizon, inference_date, base_price,
                    predicted_return_pct, direction, signal, confidence
                FROM bi.fact_forecasts
                WHERE inference_date = (
                    SELECT MAX(inference_date) FROM bi.fact_forecasts
                )
                ORDER BY model_id, horizon
            """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return None

        return [
            {
                "model_id": row[0],
                "horizon": row[1],
                "inference_date": str(row[2]),
                "base_price": float(row[3]) if row[3] else None,
                "predicted_return_pct": float(row[4]) if row[4] else 0,
                "direction": row[5] or "NEUTRAL",
                "signal": row[6] or 0,
                "confidence": float(row[7]) if row[7] else None,
            }
            for row in rows
        ]

    except Exception as e:
        logger.warning(f"PostgreSQL query failed: {e}")
        return None


def query_postgresql_consensus(inference_date: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
    """Query consensus from PostgreSQL."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        if inference_date:
            cur.execute("""
                SELECT
                    horizon, bullish_count, bearish_count,
                    consensus_direction, consensus_strength, avg_predicted_return
                FROM bi.fact_consensus
                WHERE inference_date = %s::date
                ORDER BY horizon
            """, (inference_date,))
        else:
            cur.execute("""
                SELECT
                    horizon, bullish_count, bearish_count,
                    consensus_direction, consensus_strength, avg_predicted_return
                FROM bi.fact_consensus
                WHERE inference_date = (
                    SELECT MAX(inference_date) FROM bi.fact_consensus
                )
                ORDER BY horizon
            """)

        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows:
            return None

        return [
            {
                "horizon": row[0],
                "horizon_label": HORIZON_LABELS.get(row[0], f"{row[0]} days"),
                "bullish_count": row[1] or 0,
                "bearish_count": row[2] or 0,
                "consensus_direction": row[3] or "NEUTRAL",
                "consensus_strength": float(row[4]) if row[4] else 0.5,
                "avg_predicted_return": float(row[5]) if row[5] else 0,
            }
            for row in rows
        ]

    except Exception as e:
        logger.warning(f"PostgreSQL consensus query failed: {e}")
        return None


# =============================================================================
# CSV FALLBACK HELPERS
# =============================================================================

def load_csv_forecasts() -> Optional[List[Dict[str, Any]]]:
    """
    Load forecasts from CSV fallback.

    Searches for bi_dashboard_unified.csv in multiple locations.
    """
    csv_paths = [
        Path("/opt/airflow/data/forecasting/bi_dashboard_unified.csv"),
        Path("data/forecasting/bi_dashboard_unified.csv"),
        Path("NewFeature/consolidated_backend/outputs/bi_dashboard_unified.csv"),
    ]

    for csv_path in csv_paths:
        if csv_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(csv_path)

                forecasts = []
                for _, row in df.iterrows():
                    forecasts.append({
                        "model_id": row.get("model_id", "unknown"),
                        "horizon": int(row.get("horizon", 0)),
                        "inference_date": str(row.get("inference_date", "")),
                        "base_price": float(row["base_price"]) if "base_price" in row and pd.notna(row["base_price"]) else None,
                        "predicted_return_pct": float(row.get("predicted_return_pct", 0)),
                        "direction": row.get("direction", "NEUTRAL"),
                        "signal": int(row.get("signal", 0)),
                        "confidence": float(row["confidence"]) if "confidence" in row and pd.notna(row["confidence"]) else None,
                    })

                logger.info(f"Loaded {len(forecasts)} forecasts from CSV: {csv_path}")
                return forecasts

            except Exception as e:
                logger.warning(f"CSV load failed for {csv_path}: {e}")
                continue

    return None


def compute_consensus_from_forecasts(forecasts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute consensus from individual forecasts."""
    from collections import defaultdict

    by_horizon = defaultdict(list)
    for f in forecasts:
        by_horizon[f["horizon"]].append(f)

    consensus = []
    for horizon in sorted(by_horizon.keys()):
        horizon_forecasts = by_horizon[horizon]
        bullish = sum(1 for f in horizon_forecasts if f["direction"] == "UP")
        bearish = sum(1 for f in horizon_forecasts if f["direction"] == "DOWN")
        total = len(horizon_forecasts)

        if bullish > bearish:
            direction = "BULLISH"
            strength = bullish / total if total > 0 else 0.5
        elif bearish > bullish:
            direction = "BEARISH"
            strength = bearish / total if total > 0 else 0.5
        else:
            direction = "NEUTRAL"
            strength = 0.5

        avg_return = sum(f["predicted_return_pct"] for f in horizon_forecasts) / total if total > 0 else 0

        consensus.append({
            "horizon": horizon,
            "horizon_label": HORIZON_LABELS.get(horizon, f"{horizon} days"),
            "bullish_count": bullish,
            "bearish_count": bearish,
            "consensus_direction": direction,
            "consensus_strength": strength,
            "avg_predicted_return": avg_return,
        })

    return consensus


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    inference_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
):
    """
    Get forecasting dashboard data.

    Returns latest forecasts, consensus, and ensemble predictions.
    Uses PostgreSQL as primary source with CSV fallback.
    """
    data_source = "postgresql"

    # Try PostgreSQL first
    forecasts = query_postgresql_forecasts(inference_date)
    consensus = query_postgresql_consensus(inference_date)

    # Fallback to CSV if PostgreSQL fails
    if forecasts is None:
        forecasts = load_csv_forecasts()
        data_source = "csv"

        if forecasts is None:
            raise HTTPException(
                status_code=503,
                detail="No forecast data available from PostgreSQL or CSV"
            )

    # Compute consensus if not from DB
    if consensus is None:
        consensus = compute_consensus_from_forecasts(forecasts)

    # Extract metadata
    if forecasts:
        sample = forecasts[0]
        inf_date = sample.get("inference_date", str(date.today()))
        # Parse date to get week/year
        try:
            dt = datetime.strptime(inf_date, "%Y-%m-%d")
            inf_week = dt.isocalendar()[1]
            inf_year = dt.year
        except:
            inf_week = 0
            inf_year = datetime.now().year
    else:
        inf_date = str(date.today())
        inf_week = 0
        inf_year = datetime.now().year

    # Get current price
    current_price = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT close FROM usdcop_m5_ohlcv ORDER BY time DESC LIMIT 1")
        result = cur.fetchone()
        cur.close()
        conn.close()
        if result:
            current_price = float(result[0])
    except:
        pass

    return DashboardResponse(
        inference_date=inf_date,
        inference_week=inf_week,
        inference_year=inf_year,
        current_price=current_price,
        forecasts=[ForecastItem(**f) for f in forecasts],
        consensus=[ConsensusItem(**c) for c in consensus],
        ensembles={},  # TODO: Add ensemble predictions
        data_source=data_source,
        contract_version=FORECASTING_CONTRACT_VERSION,
    )


@router.get("/forecasts", response_model=List[ForecastItem])
async def get_forecasts(
    inference_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    horizon: Optional[int] = Query(None, description="Filter by horizon"),
    limit: int = Query(100, ge=1, le=1000),
):
    """
    Get historical forecasts.

    Supports filtering by date, model, and horizon.
    """
    forecasts = query_postgresql_forecasts(inference_date, limit)

    if forecasts is None:
        forecasts = load_csv_forecasts()

    if forecasts is None:
        return []

    # Apply filters
    if model_id:
        forecasts = [f for f in forecasts if f["model_id"] == model_id]
    if horizon:
        forecasts = [f for f in forecasts if f["horizon"] == horizon]

    return [ForecastItem(**f) for f in forecasts[:limit]]


@router.get("/consensus", response_model=List[ConsensusItem])
async def get_consensus(
    inference_date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
):
    """
    Get consensus by horizon.

    Returns bullish/bearish counts and consensus strength for each horizon.
    """
    consensus = query_postgresql_consensus(inference_date)

    if consensus is None:
        # Try to compute from forecasts
        forecasts = query_postgresql_forecasts(inference_date)
        if forecasts is None:
            forecasts = load_csv_forecasts()
        if forecasts:
            consensus = compute_consensus_from_forecasts(forecasts)

    if consensus is None:
        return []

    return [ConsensusItem(**c) for c in consensus]


@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """
    Get list of available forecasting models.

    Returns model metadata from SSOT contracts.
    """
    try:
        from src.forecasting.contracts import MODEL_DEFINITIONS
        return [
            ModelInfo(
                model_id=model_id,
                model_name=info.get("name", model_id),
                model_type=info.get("type", "unknown").value if hasattr(info.get("type"), "value") else str(info.get("type", "unknown")),
                description=f"Requires scaling: {info.get('requires_scaling', False)}, Early stopping: {info.get('supports_early_stopping', False)}",
            )
            for model_id, info in MODEL_DEFINITIONS.items()
        ]
    except ImportError:
        # Fallback to basic model list
        return [
            ModelInfo(model_id=mid, model_name=mid, model_type="unknown")
            for mid in MODEL_IDS
        ]


@router.get("/metrics", response_model=List[ModelMetrics])
async def get_metrics(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    horizon: Optional[int] = Query(None, description="Filter by horizon"),
):
    """
    Get model performance metrics.

    Returns direction accuracy, RMSE, MAE for each model/horizon combination.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        query = """
            SELECT model_id, horizon, direction_accuracy, rmse, mae, sample_count
            FROM bi.fact_model_metrics
            WHERE 1=1
        """
        params = []

        if model_id:
            query += " AND model_id LIKE %s"
            params.append(f"%{model_id}%")
        if horizon:
            query += " AND horizon = %s"
            params.append(horizon)

        query += " ORDER BY model_id, horizon"

        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        conn.close()

        return [
            ModelMetrics(
                model_id=row[0],
                horizon=row[1],
                direction_accuracy=float(row[2]) if row[2] else 0,
                rmse=float(row[3]) if row[3] else None,
                mae=float(row[4]) if row[4] else None,
                sample_count=row[5] or 0,
            )
            for row in rows
        ]

    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        return []


@router.get("/health")
async def health_check():
    """
    Forecasting service health check.

    Returns availability of PostgreSQL and CSV data sources.
    """
    health = {
        "status": "healthy",
        "postgresql": False,
        "csv": False,
        "contract_version": FORECASTING_CONTRACT_VERSION,
        "contract_hash": FORECASTING_CONTRACT_HASH,
    }

    # Check PostgreSQL
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM bi.fact_forecasts LIMIT 1")
        cur.close()
        conn.close()
        health["postgresql"] = True
    except:
        pass

    # Check CSV
    csv_paths = [
        Path("/opt/airflow/data/forecasting/bi_dashboard_unified.csv"),
        Path("data/forecasting/bi_dashboard_unified.csv"),
    ]
    for path in csv_paths:
        if path.exists():
            health["csv"] = True
            break

    if not health["postgresql"] and not health["csv"]:
        health["status"] = "degraded"

    return health
