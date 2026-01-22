"""
Signal routes.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.auth import get_current_active_user
from app.models import User
from app.contracts.signal import (
    TradingSignal,
    SignalCreate,
    SignalAction,
    SignalStats,
    SignalFilter,
)
from app.contracts.common import PaginatedResponse
from app.services.signal import SignalService

router = APIRouter(prefix="/signals", tags=["Signals"])


@router.get("", response_model=PaginatedResponse[TradingSignal])
async def list_signals(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    action: Optional[int] = None,
    symbol: Optional[str] = None,
    is_processed: Optional[bool] = None,
    since: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List trading signals with pagination and filtering.
    """
    signal_service = SignalService(db)

    filters = SignalFilter(
        action=SignalAction(action) if action is not None else None,
        symbol=symbol,
        is_processed=is_processed,
        since=since,
    )

    signals, total = await signal_service.get_signals(
        user_id=current_user.id,
        filters=filters,
        page=page,
        limit=limit,
    )

    return PaginatedResponse(
        items=[signal_service.to_response(s) for s in signals],
        total=total,
        page=page,
        limit=limit,
        has_more=(page * limit) < total,
    )


@router.post("", response_model=TradingSignal, status_code=201)
async def create_signal(
    data: SignalCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new trading signal.
    """
    signal_service = SignalService(db)
    signal = await signal_service.create_signal(
        user_id=current_user.id,
        data=data,
    )

    return signal_service.to_response(signal)


@router.get("/recent", response_model=List[TradingSignal])
async def get_recent_signals(
    limit: int = Query(5, ge=1, le=20),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get most recent signals.
    """
    signal_service = SignalService(db)
    signals = await signal_service.get_recent_signals(
        user_id=current_user.id,
        limit=limit,
    )

    return [signal_service.to_response(s) for s in signals]


@router.get("/stats", response_model=SignalStats)
async def get_signal_stats(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get signal statistics.
    """
    signal_service = SignalService(db)
    return await signal_service.get_stats(
        user_id=current_user.id,
        days=days,
    )


@router.get("/{signal_id}", response_model=TradingSignal)
async def get_signal(
    signal_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific signal.
    """
    signal_service = SignalService(db)
    signal = await signal_service.get_signal(
        signal_id=signal_id,
        user_id=current_user.id,
    )

    if not signal:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(
            message="Signal not found",
            resource_type="Signal",
            resource_id=str(signal_id),
        ).to_http_exception()

    return signal_service.to_response(signal)
