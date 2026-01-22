"""
Execution routes.
"""

from typing import Optional
from uuid import UUID
from datetime import datetime
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.middleware.auth import get_current_active_user
from app.models import User
from app.contracts.execution import (
    ExecutionResult,
    ExecutionCreate,
    ExecutionSummary,
    ExecutionStats,
    ExecutionStatus,
    ExecutionFilter,
    OrderSide,
    TodayStats,
)
from app.contracts.exchange import SupportedExchange
from app.contracts.common import PaginatedResponse, SuccessResponse
from app.services.execution import ExecutionService

router = APIRouter(prefix="/executions", tags=["Executions"])


@router.get("", response_model=PaginatedResponse[ExecutionSummary])
async def list_executions(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    exchange: Optional[SupportedExchange] = None,
    symbol: Optional[str] = None,
    status: Optional[ExecutionStatus] = None,
    side: Optional[OrderSide] = None,
    since: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    List executions with pagination and filtering.
    """
    execution_service = ExecutionService(db)

    filters = ExecutionFilter(
        exchange=exchange,
        symbol=symbol,
        status=status,
        side=side,
        since=since,
    )

    executions, total = await execution_service.get_executions(
        user_id=current_user.id,
        filters=filters,
        page=page,
        limit=limit,
    )

    return PaginatedResponse(
        items=[execution_service.to_summary(e) for e in executions],
        total=total,
        page=page,
        limit=limit,
        has_more=(page * limit) < total,
    )


@router.post("", response_model=ExecutionResult, status_code=201)
async def create_execution(
    data: ExecutionCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new execution request.
    """
    execution_service = ExecutionService(db)
    execution = await execution_service.create_execution(
        user_id=current_user.id,
        data=data,
    )

    return execution_service.to_response(execution)


@router.get("/stats", response_model=ExecutionStats)
async def get_execution_stats(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get execution statistics.
    """
    execution_service = ExecutionService(db)
    return await execution_service.get_stats(
        user_id=current_user.id,
        days=days,
    )


@router.get("/today", response_model=TodayStats)
async def get_today_stats(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get today's trading statistics.
    """
    execution_service = ExecutionService(db)
    return await execution_service.get_today_stats(current_user.id)


@router.get("/{execution_id}", response_model=ExecutionResult)
async def get_execution(
    execution_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific execution.
    """
    execution_service = ExecutionService(db)
    execution = await execution_service.get_execution(
        execution_id=execution_id,
        user_id=current_user.id,
    )

    if not execution:
        from app.core.exceptions import NotFoundError
        raise NotFoundError(
            message="Execution not found",
            resource_type="Execution",
            resource_id=str(execution_id),
        ).to_http_exception()

    return execution_service.to_response(execution)


@router.post("/{execution_id}/execute", response_model=ExecutionResult)
async def execute_order(
    execution_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Execute a pending order on the exchange.
    """
    execution_service = ExecutionService(db)
    execution = await execution_service.execute_order(
        execution_id=execution_id,
        user_id=current_user.id,
    )

    return execution_service.to_response(execution)


@router.post("/{execution_id}/cancel", response_model=ExecutionResult)
async def cancel_execution(
    execution_id: UUID,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel an open order.
    """
    execution_service = ExecutionService(db)
    execution = await execution_service.cancel_execution(
        execution_id=execution_id,
        user_id=current_user.id,
    )

    return execution_service.to_response(execution)
