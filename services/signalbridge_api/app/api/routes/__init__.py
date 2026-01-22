"""
API Routes module.
"""

from .auth import router as auth_router
from .users import router as users_router
from .exchanges import router as exchanges_router
from .trading import router as trading_router
from .signals import router as signals_router
from .executions import router as executions_router
from .webhooks import router as webhooks_router
from .signal_bridge import router as signal_bridge_router
from .ws_notifications import router as ws_notifications_router

__all__ = [
    "auth_router",
    "users_router",
    "exchanges_router",
    "trading_router",
    "signals_router",
    "executions_router",
    "webhooks_router",
    "signal_bridge_router",
    "ws_notifications_router",
]
