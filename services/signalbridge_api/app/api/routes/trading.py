"""
Trading configuration routes.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.auth import get_current_active_user
from app.models import User
from app.contracts.trading import (
    TradingConfig,
    TradingConfigUpdate,
    TradingStatus,
)
from app.services.trading import TradingService

router = APIRouter(prefix="/trading", tags=["Trading"])


@router.get("/config", response_model=TradingConfig)
async def get_trading_config(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get trading configuration.
    """
    trading_service = TradingService(db)
    config = await trading_service.get_or_create_config(current_user.id)
    return trading_service.to_response(config)


@router.patch("/config", response_model=TradingConfig)
async def update_trading_config(
    data: TradingConfigUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Update trading configuration.
    """
    trading_service = TradingService(db)
    config = await trading_service.update_config(current_user.id, data)
    return trading_service.to_response(config)


@router.get("/status", response_model=TradingStatus)
async def get_trading_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get current trading status.
    """
    trading_service = TradingService(db)
    return await trading_service.get_status(current_user.id)


@router.post("/toggle", response_model=TradingConfig)
async def toggle_trading(
    enabled: bool,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Enable or disable trading.
    """
    trading_service = TradingService(db)
    config = await trading_service.toggle_trading(current_user.id, enabled)
    return trading_service.to_response(config)
