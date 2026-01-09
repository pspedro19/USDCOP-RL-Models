"""
API Routes for ML Analytics Service
====================================
FastAPI routes for model performance monitoring.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from services.metrics_calculator import MetricsCalculator
from services.drift_detector import DriftDetector
from services.prediction_tracker import PredictionTracker
from services.performance_analyzer import PerformanceAnalyzer
from database.postgres_client import PostgresClient

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Initialize services (will be set in main.py)
db_client: Optional[PostgresClient] = None
metrics_calc: Optional[MetricsCalculator] = None
drift_detector: Optional[DriftDetector] = None
prediction_tracker: Optional[PredictionTracker] = None
performance_analyzer: Optional[PerformanceAnalyzer] = None


def init_services(db: PostgresClient):
    """Initialize services with database client"""
    global db_client, metrics_calc, drift_detector, prediction_tracker, performance_analyzer
    db_client = db
    metrics_calc = MetricsCalculator(db)
    drift_detector = DriftDetector(db)
    prediction_tracker = PredictionTracker(db)
    performance_analyzer = PerformanceAnalyzer(db)


# ============================================================================
# METRICS ENDPOINTS
# ============================================================================

@router.get("/api/metrics/summary")
async def get_metrics_summary():
    """
    Get overall metrics summary for all models.

    Returns:
        Summary of metrics across all models
    """
    try:
        summary = metrics_calc.get_metrics_summary()
        return {
            'success': True,
            'data': summary,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting metrics summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metrics/rolling")
async def get_rolling_metrics(
    model_id: str = Query(..., description="Model identifier"),
    window: str = Query('24h', description="Time window: 1h, 24h, 7d, 30d")
):
    """
    Get rolling window metrics for a specific model.

    Args:
        model_id: Model identifier
        window: Time window (1h, 24h, 7d, 30d)

    Returns:
        Rolling metrics for the specified window
    """
    try:
        if window not in ['1h', '24h', '7d', '30d']:
            raise HTTPException(
                status_code=400,
                detail="Invalid window. Must be one of: 1h, 24h, 7d, 30d"
            )

        metrics = metrics_calc.calculate_rolling_metrics(model_id, window)
        return {
            'success': True,
            'data': metrics,
            'timestamp': datetime.utcnow()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating rolling metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/metrics/model/{model_id}")
async def get_model_metrics(
    model_id: str,
    window: str = Query('24h', description="Time window")
):
    """
    Get detailed metrics for a specific model.

    Args:
        model_id: Model identifier
        window: Time window

    Returns:
        Detailed model metrics
    """
    try:
        metrics = metrics_calc.calculate_rolling_metrics(model_id, window)
        drift = drift_detector.detect_drift(model_id)
        accuracy = prediction_tracker.get_prediction_accuracy(model_id, window)

        return {
            'success': True,
            'data': {
                'model_id': model_id,
                'window': window,
                'metrics': metrics,
                'drift': drift,
                'accuracy': accuracy
            },
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# PREDICTION ENDPOINTS
# ============================================================================

@router.get("/api/predictions/accuracy")
async def get_prediction_accuracy(
    model_id: str = Query(..., description="Model identifier"),
    window: str = Query('24h', description="Time window")
):
    """
    Get prediction accuracy metrics.

    Args:
        model_id: Model identifier
        window: Time window

    Returns:
        Prediction accuracy metrics
    """
    try:
        accuracy = prediction_tracker.get_prediction_accuracy(model_id, window)
        return {
            'success': True,
            'data': accuracy,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting prediction accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predictions/history")
async def get_prediction_history(
    model_id: str = Query(..., description="Model identifier"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(100, ge=1, le=1000, description="Results per page")
):
    """
    Get prediction history with outcomes.

    Args:
        model_id: Model identifier
        page: Page number
        page_size: Results per page

    Returns:
        Historical predictions
    """
    try:
        history = prediction_tracker.get_prediction_history(
            model_id=model_id,
            page=page,
            page_size=page_size
        )
        return {
            'success': True,
            'data': history,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/predictions/comparison")
async def get_predictions_comparison(
    model_id: str = Query(..., description="Model identifier"),
    limit: int = Query(100, ge=1, le=1000, description="Number of predictions")
):
    """
    Compare predictions vs actual outcomes.

    Args:
        model_id: Model identifier
        limit: Number of predictions

    Returns:
        Prediction comparisons
    """
    try:
        comparisons = prediction_tracker.compare_predictions_vs_actuals(
            model_id=model_id,
            limit=limit
        )
        return {
            'success': True,
            'data': {
                'model_id': model_id,
                'comparisons': comparisons,
                'total': len(comparisons)
            },
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error comparing predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# DRIFT DETECTION ENDPOINTS
# ============================================================================

@router.get("/api/drift/status")
async def get_drift_status(
    model_id: str = Query(..., description="Model identifier"),
    window_hours: int = Query(24, ge=1, le=168, description="Recent window in hours"),
    baseline_days: int = Query(7, ge=1, le=30, description="Baseline period in days")
):
    """
    Get drift detection status.

    Args:
        model_id: Model identifier
        window_hours: Recent window to analyze (hours)
        baseline_days: Baseline period for comparison (days)

    Returns:
        Drift detection status
    """
    try:
        drift = drift_detector.detect_drift(
            model_id=model_id,
            window_hours=window_hours,
            baseline_days=baseline_days
        )
        return {
            'success': True,
            'data': drift,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error detecting drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/drift/features")
async def get_drift_by_features(
    model_id: str = Query(..., description="Model identifier")
):
    """
    Get drift detection results grouped by feature.

    Args:
        model_id: Model identifier

    Returns:
        Drift metrics by feature
    """
    try:
        drift = drift_detector.get_drift_by_feature(model_id)
        return {
            'success': True,
            'data': drift,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting drift by features: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HEALTH & PERFORMANCE ENDPOINTS
# ============================================================================

@router.get("/api/health/models")
async def get_models_health():
    """
    Get health status for all models.

    Returns:
        Health status for all models
    """
    try:
        health = performance_analyzer.get_models_health_status()
        return {
            'success': True,
            'data': health,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting models health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/health/model/{model_id}")
async def get_model_health(model_id: str):
    """
    Get health status for a specific model.

    Args:
        model_id: Model identifier

    Returns:
        Model health status
    """
    try:
        # Get model stats
        query = """
            SELECT
                model_id,
                model_version,
                COUNT(*) as prediction_count,
                AVG(CASE WHEN reward > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                AVG(latency_ms) as avg_latency_ms,
                MAX(timestamp_utc) as last_prediction,
                MIN(timestamp_utc) as first_prediction
            FROM dw.fact_rl_inference
            WHERE model_id = %s
                AND timestamp_utc >= NOW() - INTERVAL '24 hours'
            GROUP BY model_id, model_version
        """
        result = db_client.execute_single(query, (model_id,))

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found or has no recent activity"
            )

        health = performance_analyzer._analyze_model_health(result)

        return {
            'success': True,
            'data': health,
            'timestamp': datetime.utcnow()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/performance/trends/{model_id}")
async def get_performance_trends(
    model_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days")
):
    """
    Get performance trends over time.

    Args:
        model_id: Model identifier
        days: Number of days to analyze

    Returns:
        Performance trends
    """
    try:
        trends = performance_analyzer.get_performance_trends(model_id, days)
        return {
            'success': True,
            'data': trends,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error getting performance trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/performance/comparison")
async def get_model_comparison(
    window: str = Query('24h', description="Time window")
):
    """
    Compare performance across all models.

    Args:
        window: Time window for comparison

    Returns:
        Model comparison metrics
    """
    try:
        comparison = performance_analyzer.get_model_comparison(window)
        return {
            'success': True,
            'data': comparison,
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Error comparing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# SERVICE HEALTH ENDPOINT
# ============================================================================

@router.get("/health")
async def health_check():
    """
    Service health check.

    Returns:
        Service health status
    """
    try:
        # Test database connection
        db_healthy = db_client.test_connection()

        return {
            'success': True,
            'service': 'ML Analytics Service',
            'status': 'healthy' if db_healthy else 'degraded',
            'database': 'connected' if db_healthy else 'disconnected',
            'timestamp': datetime.utcnow()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            'success': False,
            'service': 'ML Analytics Service',
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow()
        }
