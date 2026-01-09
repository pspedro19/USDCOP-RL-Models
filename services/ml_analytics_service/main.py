"""
ML Analytics Service - Main Application
========================================
FastAPI backend service for monitoring RL model performance.

Features:
- Rolling metrics calculation (1h, 24h, 7d, 30d windows)
- Data/concept drift detection
- Prediction tracking vs actuals
- Model health monitoring
- Performance analysis and comparison

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
Port: 8004
"""

import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import SERVICE_CONFIG
from database.postgres_client import PostgresClient
from api.routes import router, init_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Global database client
db_client: PostgresClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    global db_client
    logger.info("Starting ML Analytics Service...")

    try:
        # Initialize database client
        db_client = PostgresClient(min_connections=2, max_connections=10)
        logger.info("Database connection pool initialized")

        # Test connection
        if db_client.test_connection():
            logger.info("Database connection test successful")
        else:
            logger.warning("Database connection test failed")

        # Initialize services
        init_services(db_client)
        logger.info("Services initialized")

        logger.info(f"ML Analytics Service started on port {SERVICE_CONFIG.port}")

    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down ML Analytics Service...")
    if db_client:
        db_client.close()
        logger.info("Database connections closed")


# Create FastAPI application
app = FastAPI(
    title="ML Analytics Service",
    description=(
        "Professional ML Analytics Backend for monitoring RL model performance. "
        "Provides real-time metrics, drift detection, prediction tracking, and health monitoring."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with service information.
    """
    return {
        "service": "ML Analytics Service",
        "version": "1.0.0",
        "description": "Professional ML Analytics Backend for RL model monitoring",
        "status": "operational",
        "endpoints": {
            "metrics": {
                "summary": "/api/metrics/summary",
                "rolling": "/api/metrics/rolling?model_id=X&window=24h",
                "model_detail": "/api/metrics/model/{model_id}"
            },
            "predictions": {
                "accuracy": "/api/predictions/accuracy?model_id=X",
                "history": "/api/predictions/history?model_id=X",
                "comparison": "/api/predictions/comparison?model_id=X"
            },
            "drift": {
                "status": "/api/drift/status?model_id=X",
                "features": "/api/drift/features?model_id=X"
            },
            "health": {
                "all_models": "/api/health/models",
                "single_model": "/api/health/model/{model_id}",
                "service": "/health"
            },
            "performance": {
                "trends": "/api/performance/trends/{model_id}",
                "comparison": "/api/performance/comparison"
            }
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        }
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    Global exception handler for unhandled errors.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting server on {SERVICE_CONFIG.host}:{SERVICE_CONFIG.port}")

    uvicorn.run(
        "main:app",
        host=SERVICE_CONFIG.host,
        port=SERVICE_CONFIG.port,
        reload=SERVICE_CONFIG.reload,
        log_level="info"
    )
