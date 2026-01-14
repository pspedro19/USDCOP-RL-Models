"""
USDCOP Inference API Service
============================

FastAPI service for on-demand backtesting and trade generation.
Runs PPO model inference on historical data to generate trades.

Contract: CTR-API-001

Security Features:
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
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from .routers import backtest_router, health_router, models_router
from .core.inference_engine import InferenceEngine
from .middleware import setup_middleware, setup_exception_handlers
from . import __version__

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Loads model on startup, cleans up on shutdown.
    """
    logger.info("Starting USDCOP Inference Service...")

    # Initialize inference engine
    inference_engine = InferenceEngine()

    # Load default model
    logger.info("Loading PPO model...")
    inference_engine.load_model("ppo_primary")

    # Store in app state for access in routes
    app.state.inference_engine = inference_engine

    logger.info(f"USDCOP Inference Service v{__version__} ready!")

    yield

    # Cleanup
    logger.info("Shutting down USDCOP Inference Service...")


# OpenAPI Tags for grouping endpoints
OPENAPI_TAGS = [
    {
        "name": "backtest",
        "description": "Backtest operations - run model inference on historical data",
    },
    {
        "name": "health",
        "description": "Health check and service status endpoints",
    },
    {
        "name": "models",
        "description": "Model management - list, load, and inspect models",
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
Currently no authentication required (development mode).
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

    # Add x-contract-version to schemas
    if "components" in openapi_schema and "schemas" in openapi_schema["components"]:
        for schema_name, schema in openapi_schema["components"]["schemas"].items():
            schema["x-contract-version"] = "1.0.0"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI app
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
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=OPENAPI_TAGS,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
app.include_router(models_router, prefix="/api/v1")

# Also maintain legacy /v1 routes for backward compatibility
app.include_router(health_router, prefix="/v1", include_in_schema=False)
app.include_router(backtest_router, prefix="/v1", include_in_schema=False)
app.include_router(models_router, prefix="/v1", include_in_schema=False)


# Override OpenAPI schema with custom version
app.openapi = custom_openapi


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with service info and API documentation links."""
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
            "models": "/api/v1/models",
        },
        "security": {
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
