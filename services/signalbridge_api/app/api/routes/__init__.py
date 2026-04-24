"""
API Routes module.
"""

from .auth import router as auth_router
from .exchanges import router as exchanges_router
from .executions import router as executions_router
from .signal_bridge import router as signal_bridge_router
from .signals import router as signals_router
from .trading import router as trading_router
from .users import router as users_router
from .webhooks import router as webhooks_router
from .ws_notifications import router as ws_notifications_router

__all__ = [
    "auth_router",
    "exchanges_router",
    "executions_router",
    "signal_bridge_router",
    "signals_router",
    "trading_router",
    "users_router",
    "webhooks_router",
    "ws_notifications_router",
]
