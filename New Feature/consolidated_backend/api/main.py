"""
FastAPI REST API - USD/COP Forecasting Pipeline

This API provides endpoints for:
- Authentication (JWT-based)
- Forecasts data access
- Model management
- Image serving
- Health checks

All sensitive endpoints require authentication.
Health check endpoints are public.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from contextlib import asynccontextmanager
import os

from .routers import forecasts, models, health, images, auth
from .database import get_connection_pool, close_connection_pool


# =============================================================================
# OpenAPI Tags Metadata
# =============================================================================
tags_metadata = [
    {
        "name": "Health",
        "description": """
Health check endpoints for service monitoring and readiness verification.

**Endpoints:**
- `/health` - Basic liveness check
- `/health/ready` - Comprehensive readiness check with dependency verification
- `/health/live` - Kubernetes liveness probe
- `/health/{service}` - Individual service health check

These endpoints are **public** and do not require authentication.
        """,
    },
    {
        "name": "Authentication",
        "description": """
JWT-based authentication endpoints.

**Endpoints:**
- `POST /auth/login` - Authenticate and receive JWT token
- `POST /auth/logout` - Invalidate current token
- `GET /auth/me` - Get current user information
- `POST /auth/refresh` - Refresh access token

**Token Usage:**
Include the token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```
        """,
    },
    {
        "name": "Forecasts",
        "description": """
USD/COP exchange rate forecast endpoints.

**Available Forecasts:**
- Multiple ML models: Ridge, XGBoost, LightGBM, CatBoost, Bayesian Ridge
- Multiple horizons: 5, 10, 20, 40 days
- Consensus forecasts aggregated across models

**Data Sources:**
- Primary: PostgreSQL database (bi.fact_forecasts)
- Fallback: CSV files
- Artifacts: MinIO object storage

All endpoints require **authentication**.
        """,
    },
    {
        "name": "Models",
        "description": """
Machine learning model information and performance metrics.

**Available Information:**
- List of available models with average metrics
- Detailed per-horizon metrics for each model
- Model comparisons and rankings
- Historical performance data

**Metrics Provided:**
- Direction Accuracy (%)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R-squared

All endpoints require **authentication**.
        """,
    },
    {
        "name": "Images",
        "description": """
Visualization image serving endpoints.

**Image Types:**
- `forward_forecast` - Future price prediction plots
- `backtest` - Historical validation results
- `heatmap` - Direction accuracy by model/horizon
- `ranking` - Model performance rankings
- `comparison` - Side-by-side model comparisons

**Data Sources:**
- Primary: MinIO object storage
- Fallback: Local file system

All endpoints require **authentication**.
        """,
    },
]


# =============================================================================
# Application Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.

    Handles startup and shutdown procedures including database connection pool.
    """
    # Startup
    print("=" * 60)
    print("Starting USD/COP Forecasting API...")
    print("=" * 60)
    print(f"Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"Database: {os.getenv('DATABASE_URL', 'Not configured')[:50]}...")
    print(f"MLflow: {os.getenv('MLFLOW_TRACKING_URI', 'Not configured')}")
    print(f"MinIO: {os.getenv('MINIO_ENDPOINT', 'Not configured')}")
    print("=" * 60)

    # Initialize database connection pool
    try:
        get_connection_pool()
        print("Database connection pool initialized")
    except Exception as e:
        print(f"Warning: Could not initialize database pool: {e}")
        print("Authentication will fail until database is available")

    yield

    # Shutdown
    print("Shutting down USD/COP Forecasting API...")

    # Close database connection pool
    try:
        close_connection_pool()
        print("Database connection pool closed")
    except Exception as e:
        print(f"Error closing database pool: {e}")


# =============================================================================
# FastAPI Application
# =============================================================================
app = FastAPI(
    title="USD/COP Forecasting API",
    description="""
## USD/COP Exchange Rate Forecasting REST API

This API provides access to machine learning forecasts for the USD/COP exchange rate,
including model performance metrics and visualization artifacts.

### Features

- **Multi-model Forecasting**: Ridge, XGBoost, LightGBM, CatBoost, Bayesian Ridge
- **Multiple Horizons**: 5, 10, 20, and 40 day forecasts
- **Consensus Forecasts**: Aggregated predictions across all models
- **Real-time Metrics**: Direction accuracy, RMSE, MAE, R-squared
- **Visualizations**: Forecast plots, backtests, heatmaps, rankings

### Authentication

Most endpoints require JWT authentication. Obtain a token via `/auth/login`
and include it in the `Authorization` header:

```
Authorization: Bearer <your_jwt_token>
```

### Data Sources

- **PostgreSQL**: Primary storage for forecasts and metrics
- **MinIO**: Object storage for artifacts and images
- **MLflow**: Model tracking and registry
- **CSV Fallback**: Local file fallback when services unavailable

### Quick Links

- [API Documentation (Swagger UI)](/docs)
- [API Documentation (ReDoc)](/redoc)
- [Health Check](/health)
- [Readiness Check](/health/ready)
    """,
    version="1.0.0",
    terms_of_service="https://example.com/terms/",
    contact={
        "name": "USD/COP Forecasting Team",
        "url": "https://github.com/your-org/usdcop-forecasting",
        "email": "forecasting-team@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=tags_metadata,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# CORS Configuration
# =============================================================================
# Default allowed origins
CORS_ORIGINS = [
    "http://localhost:3000",      # React default
    "http://localhost:5173",      # Vite default
    "http://localhost:8080",      # Alternative frontend
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

# Add custom frontend URL from environment
frontend_url = os.getenv("FRONTEND_URL")
if frontend_url and frontend_url not in CORS_ORIGINS:
    CORS_ORIGINS.append(frontend_url)

# Add additional origins from environment (comma-separated)
additional_origins = os.getenv("CORS_ORIGINS", "")
if additional_origins:
    for origin in additional_origins.split(","):
        origin = origin.strip()
        if origin and origin not in CORS_ORIGINS:
            CORS_ORIGINS.append(origin)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
    ],
    expose_headers=[
        "X-Source",
        "X-File-Path",
    ],
    max_age=600,  # Cache preflight requests for 10 minutes
)


# =============================================================================
# Custom OpenAPI Schema
# =============================================================================
def custom_openapi():
    """
    Generate custom OpenAPI schema with additional metadata.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
        tags=tags_metadata,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter your JWT token obtained from /auth/login"
        }
    }

    # Add servers
    openapi_schema["servers"] = [
        {
            "url": "/",
            "description": "Current server"
        },
        {
            "url": "http://localhost:8000",
            "description": "Local development server"
        },
        {
            "url": "https://api.forecasting.example.com",
            "description": "Production server"
        }
    ]

    # Add external documentation
    openapi_schema["externalDocs"] = {
        "description": "Find more information about the USD/COP Forecasting Pipeline",
        "url": "https://github.com/your-org/usdcop-forecasting/wiki"
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# =============================================================================
# Include Routers
# =============================================================================
# Authentication (public endpoints for login)
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)

# Health checks (public - no auth required)
app.include_router(
    health.router,
    tags=["Health"]
)

# Protected endpoints (require authentication)
app.include_router(
    forecasts.router,
    prefix="/api/forecasts",
    tags=["Forecasts"]
)

app.include_router(
    models.router,
    prefix="/api/models",
    tags=["Models"]
)

app.include_router(
    images.router,
    prefix="/api/images",
    tags=["Images"]
)


# =============================================================================
# Root Endpoint
# =============================================================================
@app.get(
    "/",
    summary="API Root",
    description="Returns basic API information and useful links.",
    response_description="API metadata",
    responses={
        200: {
            "description": "API information",
            "content": {
                "application/json": {
                    "example": {
                        "name": "USD/COP Forecasting API",
                        "version": "1.0.0",
                        "description": "REST API for USD/COP exchange rate forecasting",
                        "docs": "/docs",
                        "redoc": "/redoc",
                        "openapi": "/openapi.json",
                        "health": "/health",
                        "ready": "/health/ready"
                    }
                }
            }
        }
    }
)
async def root():
    """
    Root endpoint providing API metadata and navigation links.

    Returns:
        Dictionary with API information and useful endpoint URLs
    """
    return {
        "name": "USD/COP Forecasting API",
        "version": "1.0.0",
        "description": "REST API for USD/COP exchange rate forecasting",
        "docs": "/docs",
        "redoc": "/redoc",
        "openapi": "/openapi.json",
        "health": "/health",
        "ready": "/health/ready",
        "endpoints": {
            "auth": "/auth",
            "forecasts": "/api/forecasts",
            "models": "/api/models",
            "images": "/api/images"
        }
    }


@app.get(
    "/api",
    summary="API Version Info",
    description="Returns API version and available endpoints.",
    response_description="API version information",
    tags=["Health"]
)
async def api_info():
    """
    API version endpoint.

    Returns:
        Dictionary with version info and available endpoints
    """
    return {
        "api_version": "v1",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            {"path": "/api/forecasts", "description": "Forecast data"},
            {"path": "/api/models", "description": "Model information"},
            {"path": "/api/images", "description": "Visualization images"},
            {"path": "/auth", "description": "Authentication"},
            {"path": "/health", "description": "Health checks"},
        ]
    }
