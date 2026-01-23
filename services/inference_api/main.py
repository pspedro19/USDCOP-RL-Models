"""
USDCOP Inference API Service
============================

FastAPI service for on-demand backtesting and trade generation.
Runs PPO model inference on historical data to generate trades.

Contract: CTR-API-001

Security Features:
    - API Key / JWT Authentication (P0-3)
    - Rate limiting (Token Bucket algorithm)
    - Correlation ID tracking (X-Request-ID)
    - Structured request logging

Usage:
    uvicorn services.inference_api.main:app --host 0.0.0.0 --port 8000 --reload

Or from the project root:
    python -m uvicorn services.inference_api.main:app --host 0.0.0.0 --port 8000

OpenAPI:
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc
    - OpenAPI JSON: http://localhost:8000/openapi.json

Environment Variables:
    - ENABLE_AUTH: Enable authentication (default: true)
    - JWT_SECRET: Secret key for JWT tokens
    - API_KEY: Fallback API key when database unavailable
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .routers import (
    backtest_router,
    config_router,
    forecasting_router,
    health_router,
    lineage_router,
    models_router,
    monitoring_router,
    operations_router,
    replay_router,
    ssot_router,
    trades_router,
    websocket_router,
)
from .core.inference_engine import InferenceEngine
from .middleware import setup_middleware, setup_exception_handlers
from .middleware.auth import AuthMiddleware, API_KEY_HEADER
from .middleware.security_headers import SecurityHeadersMiddleware, RequestSizeLimitMiddleware
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Global auth middleware instance (initialized in lifespan)
auth_middleware: Optional[AuthMiddleware] = None


async def get_db_pool():
    """Get database connection pool.

    Returns None if not configured, allowing auth to fall back
    to environment variable validation.
    """
    # Try to import and get pool from a connection manager
    try:
        import asyncpg
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            pool = await asyncpg.create_pool(db_url, min_size=1, max_size=5)
            return pool
    except Exception as e:
        logger.warning(f"Database pool not available: {e}")
    return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads model on startup, cleans up on shutdown.
    """
    global auth_middleware

    logger.info("Starting USDCOP Inference Service...")

    # Initialize database pool (optional)
    db_pool = await get_db_pool()
    app.state.db_pool = db_pool

    # Verify required database tables exist (prevents schema drift issues)
    # Note: In Docker, migrations run via entrypoint.sh before this
    try:
        from scripts.db_migrate import validate_tables
        tables_ok = await validate_tables()
        if not tables_ok:
            logger.warning("Some database tables may be missing - run: python scripts/db_migrate.py")
    except ImportError:
        # Normal when running in Docker (migrations handled by entrypoint)
        logger.debug("Migration script not in path (handled by Docker entrypoint)")
    except Exception as e:
        logger.warning(f"Could not verify database tables: {e}")

    # Initialize auth middleware
    jwt_secret = os.environ.get("JWT_SECRET", "change-me-in-production")
    if jwt_secret == "change-me-in-production":
        logger.warning(
            "Using default JWT secret. Set JWT_SECRET env var in production!"
        )

    auth_middleware = AuthMiddleware(
        db_pool=db_pool,
        jwt_secret=jwt_secret,
        enabled=os.environ.get("ENABLE_AUTH", "true").lower() == "true"
    )
    app.state.auth_middleware = auth_middleware

    # Initialize inference engine
    inference_engine = InferenceEngine()

    # Load default model
    logger.info("Loading PPO model...")
    inference_engine.load_model("ppo_primary")

    # Store in app state for access in routes
    app.state.inference_engine = inference_engine

    # Initialize drift detectors (Sprint 3: COMP-88)
    try:
        from src.monitoring.drift_detector import (
            FeatureDriftDetector,
            MultivariateDriftDetector,
            create_drift_detector,
            create_multivariate_drift_detector,
        )
        from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM

        # Path to reference statistics
        project_root = Path(__file__).parent.parent.parent
        norm_stats_path = project_root / "config" / "norm_stats.json"

        # Initialize univariate drift detector
        if norm_stats_path.exists():
            drift_detector = create_drift_detector(
                reference_stats_path=str(norm_stats_path),
                p_value_threshold=0.01,
                window_size=1000,
            )
            app.state.drift_detector = drift_detector
            logger.info(f"Drift detector initialized with {len(drift_detector.reference_manager.feature_names)} features")
        else:
            logger.warning(f"Norm stats not found at {norm_stats_path}, drift detection using demo data")
            app.state.drift_detector = None

        # Initialize multivariate drift detector
        multivariate_detector = create_multivariate_drift_detector(
            n_features=OBSERVATION_DIM,
            window_size=500,
            min_samples=100,
        )
        app.state.multivariate_drift_detector = multivariate_detector
        logger.info(f"Multivariate drift detector initialized for {OBSERVATION_DIM} features")

        # Initialize drift observation service
        from .services.drift_observation_service import (
            DriftObservationService,
            create_drift_observation_hook,
        )

        drift_service = DriftObservationService(
            drift_detector=app.state.drift_detector,
            multivariate_detector=multivariate_detector,
            feature_order=list(FEATURE_ORDER),
        )
        app.state.drift_observation_service = drift_service

        # Create hook for inference engine
        drift_hook = create_drift_observation_hook(drift_service)
        inference_engine.post_predict_hook = drift_hook
        logger.info("Drift observation service initialized and hooked to inference engine")

        # Load multivariate reference data if available
        drift_ref_path = project_root / "config" / "drift_reference.json"
        if drift_ref_path.exists():
            try:
                import json
                with open(drift_ref_path, 'r') as f:
                    ref_data = json.load(f)
                if "observations" in ref_data:
                    import numpy as np
                    ref_array = np.array(ref_data["observations"])
                    multivariate_detector.set_reference_data(ref_array)
                    logger.info(f"Loaded multivariate reference: {len(ref_array)} samples")
            except Exception as ref_e:
                logger.warning(f"Could not load drift reference: {ref_e}")

        # Initialize drift persistence service
        from .services.drift_persistence_service import DriftPersistenceService
        drift_persistence = DriftPersistenceService(db_pool=db_pool)
        app.state.drift_persistence_service = drift_persistence
        logger.info(f"Drift persistence service initialized: enabled={drift_persistence._enabled}")

    except Exception as e:
        logger.warning(f"Could not initialize drift detectors: {e}")
        app.state.drift_detector = None
        app.state.multivariate_drift_detector = None
        app.state.drift_observation_service = None
        app.state.drift_persistence_service = None

    logger.info(f"USDCOP Inference Service v{__version__} ready!")
    logger.info(f"Authentication: {'enabled' if auth_middleware.enabled else 'disabled'}")

    yield

    # Cleanup
    logger.info("Shutting down USDCOP Inference Service...")
    if db_pool:
        await db_pool.close()


# OpenAPI Tags for grouping endpoints
OPENAPI_TAGS = [
    {
        "name": "backtest",
        "description": "Backtest operations - run model inference on historical data",
    },
    {
        "name": "config",
        "description": "Configuration management - feature flags, trading config, kill switch",
    },
    {
        "name": "health",
        "description": "Health check and service status endpoints",
    },
    {
        "name": "lineage",
        "description": "Lineage tracking - trade and model audit trails",
    },
    {
        "name": "models",
        "description": "Model management - list, load, and inspect models",
    },
    {
        "name": "operations",
        "description": "Emergency controls - kill switch, pause, resume trading operations",
    },
    {
        "name": "replay",
        "description": "Feature replay - retrieve historical features for debugging",
    },
    {
        "name": "trades",
        "description": "Trade history - list trades, performance summary, latest trade",
    },
    {
        "name": "websocket",
        "description": "WebSocket - real-time predictions and updates",
    },
    {
        "name": "ssot",
        "description": "Single Source of Truth - contract values and validation",
    },
    {
        "name": "forecasting",
        "description": "Forecasting - ML model predictions, consensus, and backtest metrics",
    },
]


def custom_openapi():
    """Custom OpenAPI schema with extended metadata."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="USDCOP Inference Service",
        version=__version__,
        description="""
## USD/COP Trading Inference API

On-demand backtesting and trade generation service using PPO Reinforcement Learning.

### Features
- **Backtest**: Run model inference on historical data
- **Caching**: Results cached in PostgreSQL for fast retrieval
- **Streaming**: Real-time progress updates via Server-Sent Events
- **Multi-Model**: Support for multiple registered models

### Contract Version
- Feature Contract: 15 dimensions
- API Contract: CTR-API-001

### Signal Mapping
| Backend | Frontend |
|---------|----------|
| LONG    | BUY      |
| SHORT   | SELL     |
| HOLD    | HOLD     |

### Authentication
API requests require authentication via one of:
- **API Key**: `X-API-Key` header with valid key
- **JWT Token**: `Authorization: Bearer <token>` header

Public endpoints (health, docs) do not require authentication.
        """,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "Trading Team",
        "email": "trading@example.com",
    }
    openapi_schema["info"]["license"] = {
        "name": "Proprietary",
    }

    # Add servers
    openapi_schema["servers"] = [
        {"url": "http://localhost:8000", "description": "Local development"},
        {"url": "http://192.168.1.7:8000", "description": "Local network"},
    ]

    # Add security schemes
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        },
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "JWT Bearer token"
        }
    }

    # Apply security globally (except excluded paths)
    openapi_schema["security"] = [
        {"ApiKeyAuth": []},
        {"BearerAuth": []}
    ]

    # Add x-contract-version to schemas
    if "schemas" in openapi_schema["components"]:
        for schema_name, schema in openapi_schema["components"]["schemas"].items():
            schema["x-contract-version"] = "1.0.0"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Determine if running in production (P0 Security - protect Swagger/ReDoc)
_is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

# Create FastAPI app with conditional docs (disabled in production)
app = FastAPI(
    title="USDCOP Inference Service",
    description="""
    On-demand backtesting and trade generation service for USD/COP trading.

    ## Features
    - Run backtests on historical data
    - Generate trades using PPO RL model
    - Cache results in PostgreSQL for fast retrieval
    - Real-time progress updates via SSE

    ## Endpoints
    - `POST /v1/backtest` - Run backtest and get trades
    - `POST /v1/backtest/stream` - Run backtest with progress updates
    - `GET /v1/health` - Health check

    ## Contract
    - Feature Contract: 15 dimensions
    - API Contract: CTR-API-001
    """,
    version=__version__,
    lifespan=lifespan,
    # P0 Security: Disable Swagger/ReDoc/OpenAPI in production
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
    openapi_url=None if _is_production else "/openapi.json",
    openapi_tags=OPENAPI_TAGS,
)

# Add CORS middleware with restricted origins (P0 Security Fix)
# SECURITY: In production, only allow specific trusted origins
_cors_origins = [
    "http://localhost:3000",           # Dashboard local development
    "http://localhost:5000",           # Dashboard Docker
    "http://dashboard:3000",           # Dashboard Docker internal
    os.getenv("DASHBOARD_URL", "").strip(),  # Dashboard production URL
]
# Filter out empty strings
_cors_origins = [origin for origin in _cors_origins if origin]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add Security Headers Middleware (P0 Security)
app.add_middleware(SecurityHeadersMiddleware)

# Add Request Size Limit Middleware (P1 Security - DoS protection)
# Limits request body to 1MB to prevent large payload attacks
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_content_length=1_000_000  # 1MB
)

# Setup security middleware (rate limiting, correlation ID, logging)
setup_middleware(
    app,
    enable_rate_limiting=True,
    enable_request_logging=True,
    enable_correlation_id=True,
    rate_limit_requests_per_minute=100,
    rate_limit_burst_size=20,
)

# Setup exception handlers (unified error responses)
setup_exception_handlers(app)

# Include routers with /api/v1 prefix for proper versioning
app.include_router(health_router, prefix="/api/v1")
app.include_router(backtest_router, prefix="/api/v1")
app.include_router(config_router, prefix="/api/v1")
app.include_router(models_router, prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1/monitoring")
app.include_router(operations_router, prefix="/api/v1")
app.include_router(replay_router, prefix="/api/v1")
app.include_router(lineage_router, prefix="/api/v1")
app.include_router(trades_router, prefix="/api/v1")
app.include_router(websocket_router, prefix="/api/v1")
app.include_router(ssot_router, prefix="/api/v1")

# Optional forecasting router (may not be available in all deployments)
if forecasting_router:
    app.include_router(forecasting_router, prefix="/api/v1")

# Also maintain legacy /v1 routes for backward compatibility
app.include_router(health_router, prefix="/v1", include_in_schema=False)
app.include_router(backtest_router, prefix="/v1", include_in_schema=False)
app.include_router(config_router, prefix="/v1", include_in_schema=False)
app.include_router(models_router, prefix="/v1", include_in_schema=False)
app.include_router(monitoring_router, prefix="/v1/monitoring", include_in_schema=False)
app.include_router(operations_router, prefix="/v1", include_in_schema=False)
app.include_router(replay_router, prefix="/v1", include_in_schema=False)
app.include_router(lineage_router, prefix="/v1", include_in_schema=False)
app.include_router(trades_router, prefix="/v1", include_in_schema=False)
app.include_router(websocket_router, prefix="/v1", include_in_schema=False)
app.include_router(ssot_router, prefix="/v1", include_in_schema=False)
if forecasting_router:
    app.include_router(forecasting_router, prefix="/v1", include_in_schema=False)


# Override OpenAPI schema with custom version
app.openapi = custom_openapi


# =============================================================================
# Authentication Dependency
# =============================================================================

async def verify_auth(request: Request) -> str:
    """Dependency to verify authentication for protected routes.

    Returns:
        User ID from authentication

    Raises:
        HTTPException: 401 if not authenticated
    """
    auth = request.app.state.auth_middleware
    user_id = await auth.verify_request(request)
    request.state.user_id = user_id
    return user_id


@app.get("/", tags=["health"])
async def root(request: Request):
    """Root endpoint with service info and API documentation links."""
    auth_enabled = getattr(request.app.state, 'auth_middleware', None)
    auth_status = "enabled" if auth_enabled and auth_enabled.enabled else "disabled"

    return {
        "service": "USDCOP Inference Service",
        "version": __version__,
        "api_version": "v1",
        "contract": "CTR-API-001",
        "feature_contract": "15 dimensions",
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json",
        },
        "endpoints": {
            "health": "/api/v1/health",
            "backtest": "/api/v1/backtest",
            "config": "/api/v1/config",
            "forecasting": "/api/v1/forecasting",
            "models": "/api/v1/models",
            "operations": "/api/v1/operations",
            "replay": "/api/v1/replay",
            "lineage": "/api/v1/lineage",
            "trades": "/api/v1/trades",
            "websocket": "/api/v1/ws/predictions",
            "ssot": "/api/v1/ssot",
        },
        "security": {
            "authentication": auth_status,
            "auth_methods": ["X-API-Key header", "Bearer JWT token"],
            "rate_limiting": "100 requests/minute",
            "correlation_id": "X-Request-ID header",
        }
    }


@app.get("/openapi-export", tags=["health"])
async def export_openapi():
    """Export OpenAPI specification as JSON.

    Use this endpoint to download the OpenAPI spec for:
    - Generating TypeScript types
    - API documentation
    - Client SDK generation
    """
    return app.openapi()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "services.inference_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
