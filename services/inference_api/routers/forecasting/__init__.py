"""
Forecasting Routers
===================

API endpoints for USD/COP forecasting.

Routes:
- /api/v1/forecasting/forecasts - Forecast data
- /api/v1/forecasting/models - Model information
- /api/v1/forecasting/images - Visualization images
- /api/v1/forecasting/dashboard - Dashboard aggregate

@version 1.0.0
"""

from fastapi import APIRouter

from services.inference_api.routers.forecasting.forecasts import router as forecasts_router
from services.inference_api.routers.forecasting.models import router as models_router
from services.inference_api.routers.forecasting.images import router as images_router
from services.inference_api.routers.forecasting.dashboard import router as dashboard_router

# Main forecasting router
router = APIRouter(prefix="/forecasting", tags=["forecasting"])

# Include sub-routers
router.include_router(forecasts_router, prefix="/forecasts")
router.include_router(models_router, prefix="/models")
router.include_router(images_router, prefix="/images")
router.include_router(dashboard_router)

__all__ = ['router']
