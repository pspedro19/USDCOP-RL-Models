"""
FastAPI routers
"""

from .backtest import router as backtest_router
from .health import router as health_router
from .lineage import router as lineage_router
from .models import router as models_router
from .monitoring import router as monitoring_router
from .operations import router as operations_router
from .replay import router as replay_router
from .ssot import router as ssot_router
from .trades import router as trades_router
from .websocket import router as websocket_router

# Optional config router (may not exist in all deployments)
try:
    from .config import router as config_router
except ImportError:
    config_router = None

__all__ = [
    "backtest_router",
    "config_router",
    "health_router",
    "lineage_router",
    "models_router",
    "monitoring_router",
    "operations_router",
    "replay_router",
    "ssot_router",
    "trades_router",
    "websocket_router",
]
