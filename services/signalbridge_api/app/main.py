"""
SignalBridge API - Main Application Entry Point
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from app.core.config import settings
from app.core.database import init_db, close_db
from app.core.exceptions import SignalBridgeException
from app.api import api_router
from app.middleware.rate_limit import RateLimitMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting SignalBridge API", env=settings.app_env)

    if settings.is_development:
        # Initialize database tables in development
        await init_db()
        logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("Shutting down SignalBridge API")
    await close_db()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Trading Signal Bridge API - Connect signals to multiple exchanges",
    version="1.0.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    openapi_url="/openapi.json" if settings.is_development else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting middleware (disabled in development for easier testing)
if not settings.is_development:
    app.add_middleware(RateLimitMiddleware)


# Global exception handler for SignalBridgeException
@app.exception_handler(SignalBridgeException)
async def signalbridge_exception_handler(
    request: Request,
    exc: SignalBridgeException,
):
    """Handle custom SignalBridge exceptions."""
    logger.error(
        "SignalBridge error",
        error_code=exc.error_code.value,
        message=exc.message,
        path=request.url.path,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
    )


# Include API routes
app.include_router(api_router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "env": settings.app_env,
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs" if settings.is_development else None,
        "health": "/health",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.is_development,
    )
