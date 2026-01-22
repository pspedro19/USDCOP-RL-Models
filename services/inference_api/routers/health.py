"""
Health Router - Health check and status endpoints

P2-1 Remediation: Added /metrics endpoint for Prometheus scraping
"""

from fastapi import APIRouter, Request
from fastapi.responses import Response
from datetime import datetime
import asyncpg
import logging
from ..models.responses import HealthResponse
from ..config import get_settings
from .. import __version__
from ..core.metrics import (
    get_metrics,
    get_content_type,
    update_active_models,
    set_api_info,
    PROMETHEUS_AVAILABLE,
)

router = APIRouter(tags=["health"])
logger = logging.getLogger(__name__)
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check(req: Request):
    """
    Health check endpoint.

    Returns service status, version, and component health.
    """
    # Check model loaded
    model_loaded = False
    try:
        model_loaded = req.app.state.inference_engine.is_loaded("ppo_primary")
    except Exception:
        pass

    # Check database connection
    db_connected = False
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        db_connected = True
    except Exception as e:
        logger.warning(f"Database connection check failed: {e}")

    status = "healthy" if (model_loaded and db_connected) else "degraded"

    return HealthResponse(
        status=status,
        version=__version__,
        model_loaded=model_loaded,
        database_connected=db_connected,
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/ready")
async def readiness_check(req: Request):
    """
    Kubernetes readiness probe endpoint.
    """
    try:
        # Check if model is loaded
        if not req.app.state.inference_engine.is_loaded("ppo_primary"):
            return {"ready": False, "reason": "Model not loaded"}

        return {"ready": True}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


@router.get("/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint.
    """
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}


@router.get("/consistency/{model_id}")
async def check_model_consistency(model_id: str, verify_hashes: bool = False):
    """
    Check feature store consistency for a model.

    This endpoint validates:
    1. Model is registered in ModelRegistry
    2. Norm stats file exists (CRITICAL - fail-fast)
    3. Builder type matches observation dimension
    4. Norm stats are not wrong hardcoded defaults
    5. Builder can be instantiated
    6. Hash verification (optional)

    Args:
        model_id: Model identifier (e.g., "ppo_primary")
        verify_hashes: Whether to verify file hashes (slower)

    Returns:
        ConsistencyReport with validation results
    """
    try:
        from ..services.consistency_validator import validate_model_consistency

        report = validate_model_consistency(
            model_id=model_id,
            project_root=settings.project_root,
            verify_hashes=verify_hashes,
        )

        return report.to_dict()

    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        return {
            "model_id": model_id,
            "overall_status": "error",
            "passed": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


@router.get("/consistency")
async def check_all_consistency(verify_hashes: bool = False):
    """
    Check feature store consistency for all registered models.

    Returns consistency reports for all models in ModelRegistry.
    """
    try:
        from ..services.consistency_validator import ConsistencyValidatorService
        from ..contracts.model_contract import ModelRegistry

        validator = ConsistencyValidatorService(settings.project_root)
        reports = validator.validate_all_models(verify_hashes)

        # Summarize
        all_passed = all(r.passed for r in reports.values())
        any_warnings = any(r.has_warnings for r in reports.values())

        return {
            "overall_status": "passed" if all_passed else ("warning" if not any(not r.passed for r in reports.values()) else "failed"),
            "all_passed": all_passed,
            "has_warnings": any_warnings,
            "models_checked": len(reports),
            "reports": {k: v.to_dict() for k, v in reports.items()},
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Consistency check failed: {e}")
        return {
            "overall_status": "error",
            "all_passed": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


# =============================================================================
# PROMETHEUS METRICS ENDPOINT (P2-1 remediation)
# =============================================================================

@router.get("/metrics")
async def metrics(req: Request):
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.

    Metrics exposed:
    - inference_requests_total: Total inference requests
    - inference_latency_seconds: Request latency histogram
    - active_models_count: Number of loaded models
    - inference_errors_total: Error counts by type
    - feature_nan_ratio: Feature quality gauge
    - feast_hit_ratio: Feast vs fallback ratio
    - model_agreement_ratio: Shadow mode agreement
    """
    # Update dynamic gauges
    try:
        engine = req.app.state.inference_engine
        if hasattr(engine, 'models'):
            update_active_models(len(engine.models))

        # Set API info
        set_api_info(__version__)
    except Exception as e:
        logger.debug(f"Error updating dynamic metrics: {e}")

    # Return metrics
    return Response(
        content=get_metrics(),
        media_type=get_content_type()
    )


@router.get("/metrics/status")
async def metrics_status():
    """
    Get metrics system status.

    Returns information about Prometheus metrics availability.
    """
    return {
        "prometheus_available": PROMETHEUS_AVAILABLE,
        "metrics_endpoint": "/api/v1/health/metrics",
        "scrape_interval_recommended": "15s",
        "timestamp": datetime.utcnow().isoformat(),
    }


# =============================================================================
# DATABASE SCHEMA HEALTH (Schema drift prevention)
# =============================================================================

# Required tables for the inference service to work correctly
REQUIRED_TABLES = [
    ("public", "usdcop_m5_ohlcv", "OHLCV price data"),
    ("public", "macro_indicators_daily", "Macro indicators"),
    ("public", "trades_history", "Trade history for backtest"),
    ("public", "trading_state", "Trading state per model"),
    ("config", "models", "Model configurations"),
]


@router.get("/schema")
async def schema_health():
    """
    Check database schema health.

    Validates that all required tables exist.
    Use this to detect schema drift before it causes runtime errors.

    Returns:
        - status: "healthy", "degraded", or "unhealthy"
        - tables: list of table checks with status
        - missing_count: number of missing tables
        - action: recommended action if issues found
    """
    try:
        conn = await asyncpg.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            database=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )

        table_checks = []
        missing_count = 0

        for schema, table, description in REQUIRED_TABLES:
            exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_schema = $1 AND table_name = $2
                )
            """, schema, table)

            table_checks.append({
                "schema": schema,
                "table": table,
                "description": description,
                "exists": exists,
                "status": "ok" if exists else "missing",
            })

            if not exists:
                missing_count += 1

        await conn.close()

        # Determine overall status
        if missing_count == 0:
            status = "healthy"
            action = None
        elif missing_count <= 2:
            status = "degraded"
            action = "Run: python scripts/db_migrate.py"
        else:
            status = "unhealthy"
            action = "Run: python scripts/db_migrate.py --validate"

        return {
            "status": status,
            "tables": table_checks,
            "total_tables": len(REQUIRED_TABLES),
            "missing_count": missing_count,
            "action": action,
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Schema health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "action": "Check database connection",
            "timestamp": datetime.utcnow().isoformat(),
        }
