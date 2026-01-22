"""
Main API router that combines all route modules.
"""

from fastapi import APIRouter

from app.api.routes import (
    auth_router,
    users_router,
    exchanges_router,
    trading_router,
    signals_router,
    executions_router,
    webhooks_router,
    signal_bridge_router,
    ws_notifications_router,
)

api_router = APIRouter(prefix="/api")

# Include all route modules
api_router.include_router(auth_router)
api_router.include_router(users_router)
api_router.include_router(exchanges_router)
api_router.include_router(trading_router)
api_router.include_router(signals_router)
api_router.include_router(executions_router)
api_router.include_router(webhooks_router)

# Signal Bridge routes
api_router.include_router(signal_bridge_router)
api_router.include_router(ws_notifications_router)
