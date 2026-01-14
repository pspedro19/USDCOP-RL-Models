"""
FastAPI routers
"""

from .backtest import router as backtest_router
from .health import router as health_router
from .models import router as models_router

__all__ = ["backtest_router", "health_router", "models_router"]
