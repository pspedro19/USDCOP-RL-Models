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

    # Skip auto table creation - tables are managed via SQL migrations
    # (init-scripts/20-signalbridge-schema.sql)
    logger.info("Database tables managed via SQL migrations")

    # Initialize bridges to consume signals from:
    # 1. WebSocket (for backtests and dashboard)
    # 2. Redis Streams (for live trading from L5 DAG)
    from app.services.websocket_bridge import WebSocketBridgeManager
    from app.services.redis_streams_bridge import RedisStreamsBridgeManager
    from app.services.signal_bridge_orchestrator import SignalBridgeOrchestrator
    from app.core.database import get_db_context

    async def handle_signal(signal):
        """Handle incoming signal from Inference API or L5 DAG."""
        try:
            async with get_db_context() as db:
                orchestrator = SignalBridgeOrchestrator(db_session=db)
                # Get default user ID for automated trading
                from uuid import UUID
                default_user_id = UUID("00000001-0000-4000-8000-000000000001")
                result = await orchestrator.process_signal(
                    signal=signal,
                    user_id=default_user_id,
                    credential_id=signal.credential_id,
                )
                logger.info(
                    "Signal processed",
                    signal_id=str(signal.signal_id),
                    action=signal.action,
                    status=result.status if result else "unknown",
                )
        except Exception as e:
            logger.error("Failed to process signal", error=str(e), signal_id=str(signal.signal_id))

    # Create and start WebSocket bridge (for backtests/dashboard)
    ws_bridge = WebSocketBridgeManager.create_instance(
        on_signal_received=handle_signal,
    )
    await ws_bridge.start()
    logger.info(
        "WebSocket bridge started",
        url=settings.inference_ws_url,
    )

    # Create and start Redis Streams bridge (for live trading from L5 DAG)
    redis_bridge = RedisStreamsBridgeManager.create_instance(
        redis_url=settings.redis_url,
        stream_name="signals:ppo_primary:stream",
        consumer_group="signalbridge",
        on_signal_received=handle_signal,
    )
    await redis_bridge.start()
    logger.info(
        "Redis Streams bridge started",
        stream="signals:ppo_primary:stream",
    )

    # Store bridge references in app state for health checks
    app.state.websocket_bridge = ws_bridge
    app.state.redis_streams_bridge = redis_bridge

    yield

    # Shutdown
    logger.info("Shutting down SignalBridge API")

    # Stop WebSocket bridge
    await WebSocketBridgeManager.stop()
    logger.info("WebSocket bridge stopped")

    # Stop Redis Streams bridge
    await RedisStreamsBridgeManager.stop()
    logger.info("Redis Streams bridge stopped")

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
    ws_bridge = getattr(app.state, "websocket_bridge", None)
    redis_bridge = getattr(app.state, "redis_streams_bridge", None)

    # Overall status is healthy if at least one bridge is connected
    ws_connected = ws_bridge.is_connected if ws_bridge else False
    redis_connected = redis_bridge.is_connected if redis_bridge else False
    is_healthy = ws_connected or redis_connected

    return {
        "status": "healthy" if is_healthy else "degraded",
        "app": settings.app_name,
        "env": settings.app_env,
        "bridges": {
            "websocket": {
                "connected": ws_connected,
                "url": settings.inference_ws_url,
            },
            "redis_streams": {
                "connected": redis_connected,
                "stream": "signals:ppo_primary:stream",
            },
        },
        "trading_mode": settings.trading_mode,
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
