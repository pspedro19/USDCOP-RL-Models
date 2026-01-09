#!/usr/bin/env python3
"""
Trading Signals Service
========================
Real-time trading signals backend service for USDCOP trading system.

Features:
- PPO-LSTM model inference
- Real-time signal generation
- WebSocket broadcasting
- Position tracking
- Risk management

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17

Usage:
    python main.py
    uvicorn main:app --host 0.0.0.0 --port 8003
"""

import logging
import sys
from datetime import datetime
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading_signals.log')
    ]
)
logger = logging.getLogger(__name__)

# Import modules
from config import get_config
from models.model_loader import ONNXModelLoader
from services.inference_service import InferenceService
from services.signal_generator import SignalGenerator
from services.position_manager import PositionManager
from api.routes import router, set_services
from api.websocket import (
    ConnectionManager,
    SignalBroadcaster,
    websocket_endpoint,
    heartbeat_loop
)

# Global service instances
inference_service: InferenceService = None
signal_generator: SignalGenerator = None
position_manager: PositionManager = None
connection_manager: ConnectionManager = None
signal_broadcaster: SignalBroadcaster = None
heartbeat_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Trading Signals Service")
    logger.info("=" * 60)

    global inference_service, signal_generator, position_manager
    global connection_manager, signal_broadcaster, heartbeat_task

    config = get_config()

    try:
        # Initialize services
        logger.info("Initializing services...")

        # 1. Model Loader & Inference Service
        logger.info(f"Loading model: {config.model_path}")
        model_loader = ONNXModelLoader(
            model_path=config.model_path,
            model_version=config.model_version,
            use_gpu=False
        )
        inference_service = InferenceService(model_loader=model_loader)
        inference_service.initialize()

        # 2. Signal Generator
        logger.info("Initializing signal generator...")
        signal_generator = SignalGenerator(inference_service=inference_service)

        # 3. Position Manager
        logger.info("Initializing position manager...")
        position_manager = PositionManager()

        # 4. WebSocket Manager
        logger.info("Initializing WebSocket broadcaster...")
        connection_manager = ConnectionManager()
        signal_broadcaster = SignalBroadcaster(connection_manager=connection_manager)

        # Set services in routes module
        set_services(inference_service, signal_generator, position_manager)

        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            heartbeat_loop(signal_broadcaster, interval=config.ws_heartbeat_interval)
        )

        logger.info("=" * 60)
        logger.info(f"Service: {config.service_name} v{config.service_version}")
        logger.info(f"Model: {config.model_version} ({config.model_type})")
        logger.info(f"Port: {config.port}")
        logger.info(f"Model loaded: {inference_service.model_loader.is_loaded}")
        logger.info("=" * 60)
        logger.info("Trading Signals Service ready!")
        logger.info("=" * 60)

        yield

    except Exception as e:
        logger.error(f"Error during startup: {e}", exc_info=True)
        raise

    # Shutdown
    logger.info("=" * 60)
    logger.info("Shutting down Trading Signals Service")
    logger.info("=" * 60)

    # Cancel heartbeat task
    if heartbeat_task:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass

    # Shutdown services
    if inference_service:
        inference_service.shutdown()

    logger.info("Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Trading Signals Service",
    description="Real-time trading signals backend for USDCOP trading system using PPO-LSTM model",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "trading-signals-service",
        "version": "1.0.0",
        "status": "active",
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "signals": "/api/signals",
            "websocket": "/ws/signals"
        },
        "model": {
            "version": config.model_version,
            "type": config.model_type,
            "loaded": inference_service.model_loader.is_loaded if inference_service else False
        }
    }


# Health check endpoint (top-level)
@app.get("/health")
async def health():
    """Simple health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "trading-signals-service",
        "version": "1.0.0"
    }


# Include API routes
app.include_router(router)


# WebSocket endpoint
@app.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    """WebSocket endpoint for real-time signals"""
    global connection_manager, signal_broadcaster

    if connection_manager is None or signal_broadcaster is None:
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await websocket_endpoint(websocket, connection_manager, signal_broadcaster)


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={
            "status": "error",
            "error": "Not Found",
            "detail": "The requested resource was not found",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


def main():
    """Main entry point"""
    config = get_config()

    logger.info(f"Starting server on {config.host}:{config.port}")

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    main()
