"""
Main API router that combines all route modules.
"""

from fastapi import APIRouter

from app.api.routes import (
    admin_router,
    auth_router,
    exchanges_router,
    executions_router,
    signal_bridge_router,
    signals_router,
    trading_router,
    users_router,
    webhooks_router,
    ws_notifications_router,
)

api_router = APIRouter(prefix="/api")

# Include all route modules
api_router.include_router(auth_router)
api_router.include_router(admin_router)
api_router.include_router(users_router)
api_router.include_router(exchanges_router)
api_router.include_router(trading_router)
api_router.include_router(signals_router)
api_router.include_router(executions_router)
api_router.include_router(webhooks_router)

# Signal Bridge routes
api_router.include_router(signal_bridge_router)
api_router.include_router(ws_notifications_router)

# Multi-tenant SignalBridge (CTR-RBAC-001 R5 + S4 fan-out)
from app.api.routes.tenant import router as tenant_router  # noqa: E402
api_router.include_router(tenant_router)
