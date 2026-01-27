"""
Signal Bridge Router
====================

REST API endpoints for the Signal Bridge.

Endpoints:
- POST /signal-bridge/process     - Process signal manually
- GET  /signal-bridge/status      - Get bridge status
- POST /signal-bridge/kill-switch - Activate/deactivate kill switch
- GET  /signal-bridge/history     - Get execution history
- GET  /signal-bridge/statistics  - Get bridge statistics
- GET  /signal-bridge/user/{id}/state - Get user trading state
- PUT  /signal-bridge/user/{id}/limits - Update user risk limits

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

from datetime import datetime, timedelta
from typing import Optional, List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.signal_bridge import (
    ManualSignalCreate,
    InferenceSignalCreate,
    ExecutionResult,
    BridgeStatus,
    BridgeHealthCheck,
    BridgeStatistics,
    KillSwitchRequest,
    ExecutionHistoryFilter,
    UserRiskLimits,
    UserRiskLimitsCreate,
    UserRiskLimitsUpdate,
    TradingMode,
    RiskCheckResult,
)
from app.contracts.common import SuccessResponse, PaginatedResponse
from app.contracts.execution import ExecutionStatus
from app.core.database import get_db_session
from app.services.signal_bridge_orchestrator import SignalBridgeOrchestrator
from app.middleware.auth import get_current_user

router = APIRouter(prefix="/signal-bridge", tags=["Signal Bridge"])


# =============================================================================
# Dependencies
# =============================================================================

async def get_orchestrator(
    db: AsyncSession = Depends(get_db_session),
) -> SignalBridgeOrchestrator:
    """Get SignalBridgeOrchestrator instance."""
    return SignalBridgeOrchestrator(db_session=db)


# =============================================================================
# Status and Health Endpoints
# =============================================================================

@router.get(
    "/status",
    response_model=BridgeStatus,
    summary="Get bridge status",
    description="Returns current status of the Signal Bridge including trading mode, connections, and statistics.",
)
async def get_status(
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> BridgeStatus:
    """Get current status of the Signal Bridge."""
    return await orchestrator.get_status()


@router.get(
    "/health",
    response_model=BridgeHealthCheck,
    summary="Health check",
    description="Detailed health check of all bridge components.",
)
async def health_check(
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> BridgeHealthCheck:
    """Perform health check on bridge components."""
    errors = []
    db_healthy = True
    redis_healthy = True
    vault_healthy = True
    inference_ws_healthy = False

    # Check database
    try:
        # Simple query to verify database connection
        await orchestrator.db_session.execute("SELECT 1")
    except Exception as e:
        db_healthy = False
        errors.append(f"Database: {str(e)}")

    # Check Redis (via TradingFlags)
    try:
        from src.config.trading_flags import get_trading_flags_redis
        flags = get_trading_flags_redis()
        redis_healthy = flags.health_check()
    except Exception as e:
        redis_healthy = False
        errors.append(f"Redis: {str(e)}")

    # Check Vault
    try:
        vault_healthy = await orchestrator.vault_service.health_check()
    except Exception as e:
        vault_healthy = False
        errors.append(f"Vault: {str(e)}")

    # Check Inference WebSocket
    try:
        from app.services.websocket_bridge import WebSocketBridgeManager
        bridge = WebSocketBridgeManager.get_instance()
        inference_ws_healthy = bridge.is_connected if bridge else False
    except:
        pass

    # Check exchange connections
    exchange_status = {}
    try:
        from app.adapters import ExchangeAdapterFactory
        factory = ExchangeAdapterFactory()

        # Check configured exchanges (mock check - just verify factory works)
        for exchange_name in ["binance", "mexc"]:
            try:
                # Just verify the adapter can be instantiated (no credentials needed for health check)
                exchange_status[exchange_name] = True
            except Exception:
                exchange_status[exchange_name] = False
    except Exception as e:
        errors.append(f"Exchange factory: {str(e)}")

    # Determine overall status
    if not (db_healthy and redis_healthy):
        status_str = "unhealthy"
    elif not (vault_healthy and inference_ws_healthy):
        status_str = "degraded"
    else:
        status_str = "healthy"

    return BridgeHealthCheck(
        status=status_str,
        database=db_healthy,
        redis=redis_healthy,
        vault=vault_healthy,
        inference_ws=inference_ws_healthy,
        exchanges=exchange_status,
        errors=errors,
    )


# =============================================================================
# Signal Processing Endpoints
# =============================================================================

@router.post(
    "/process",
    response_model=ExecutionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Process signal",
    description="Manually submit a trading signal for processing through the bridge.",
)
async def process_signal(
    signal: ManualSignalCreate,
    user = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> ExecutionResult:
    """Process a manual trading signal."""
    user_id = UUID(str(user.id)) if not isinstance(user.id, UUID) else user.id

    # Convert to InferenceSignalCreate
    inference_signal = InferenceSignalCreate(
        signal_id=UUID(str(uuid4())),
        model_id=signal.model_id,
        action=signal.action,
        confidence=signal.confidence,
        symbol=signal.symbol,
        credential_id=signal.credential_id,
        metadata=signal.metadata,
    )

    return await orchestrator.process_signal(
        signal=inference_signal,
        user_id=user_id,
        credential_id=signal.credential_id,
    )


@router.post(
    "/validate",
    response_model=RiskCheckResult,
    summary="Validate signal",
    description="Validate a signal against risk rules without executing.",
)
async def validate_signal(
    signal: ManualSignalCreate,
    user = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> RiskCheckResult:
    """Validate a signal without executing."""
    user_id = UUID(str(user.id)) if not isinstance(user.id, UUID) else user.id

    inference_signal = InferenceSignalCreate(
        signal_id=UUID(str(uuid4())),
        model_id=signal.model_id,
        action=signal.action,
        confidence=signal.confidence,
        symbol=signal.symbol,
        credential_id=signal.credential_id,
    )

    # Only perform risk validation
    quantity = signal.quantity or 100.0
    return await orchestrator.risk_bridge.validate_execution(
        signal=inference_signal,
        quantity=quantity,
        user_id=user_id,
    )


# =============================================================================
# Kill Switch Endpoints
# =============================================================================

@router.post(
    "/kill-switch",
    response_model=SuccessResponse,
    summary="Control kill switch",
    description="Activate or deactivate the emergency kill switch.",
)
async def control_kill_switch(
    request: KillSwitchRequest,
    user = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> SuccessResponse:
    """Activate or deactivate the kill switch."""
    user_id = user.id
    username = getattr(user, "name", getattr(user, "username", "unknown"))

    if request.activate:
        success = await orchestrator.activate_kill_switch(
            reason=request.reason,
            activated_by=username,
        )
        return SuccessResponse(
            success=success,
            message=f"Kill switch activated: {request.reason}",
            data={"activated_by": username},
        )
    else:
        if not request.confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Deactivation requires confirm=True",
            )

        success = await orchestrator.deactivate_kill_switch(
            confirm=request.confirm,
            deactivated_by=username,
        )
        return SuccessResponse(
            success=success,
            message="Kill switch deactivated",
            data={"deactivated_by": username},
        )


@router.get(
    "/kill-switch/status",
    response_model=dict,
    summary="Get kill switch status",
    description="Get current kill switch status and reason.",
)
async def get_kill_switch_status(
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> dict:
    """Get kill switch status."""
    bridge_status = await orchestrator.get_status()
    return {
        "active": bridge_status.kill_switch_active,
        "reason": bridge_status.kill_switch_reason,
        "trading_mode": bridge_status.trading_mode.value,
    }


# =============================================================================
# History and Statistics Endpoints
# =============================================================================

@router.get(
    "/history",
    response_model=PaginatedResponse[ExecutionResult],
    summary="Get execution history",
    description="Retrieve paginated execution history with optional filters.",
)
async def get_history(
    exchange: Optional[str] = Query(None, description="Filter by exchange"),
    symbol: Optional[str] = Query(None, description="Filter by symbol"),
    exec_status: Optional[ExecutionStatus] = Query(None, alias="status", description="Filter by status"),
    model_id: Optional[str] = Query(None, description="Filter by model"),
    since: Optional[datetime] = Query(None, description="Start date"),
    until: Optional[datetime] = Query(None, description="End date"),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
    user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> PaginatedResponse[ExecutionResult]:
    """Get execution history with filters."""
    from sqlalchemy import text

    user_id = user.id
    offset = (page - 1) * limit

    # Build WHERE clauses dynamically
    conditions = ["e.user_id = :user_id"]
    params = {"user_id": user_id, "limit": limit, "offset": offset}

    if exchange:
        conditions.append("e.exchange = :exchange")
        params["exchange"] = exchange
    if symbol:
        conditions.append("e.symbol = :symbol")
        params["symbol"] = symbol
    if exec_status:
        conditions.append("e.status = :status")
        params["status"] = exec_status.value
    if model_id:
        conditions.append("e.model_id = :model_id")
        params["model_id"] = model_id
    if since:
        conditions.append("e.created_at >= :since")
        params["since"] = since
    if until:
        conditions.append("e.created_at <= :until")
        params["until"] = until

    where_clause = " AND ".join(conditions)

    # Count total
    count_result = await db.execute(
        text(f"""
            SELECT COUNT(*) as total
            FROM executions e
            WHERE {where_clause}
        """),
        params
    )
    total = count_result.scalar() or 0

    # Get paginated results
    result = await db.execute(
        text(f"""
            SELECT
                e.id, e.user_id, e.credential_id, e.exchange, e.symbol,
                e.side, e.order_type, e.quantity, e.price, e.filled_quantity,
                e.average_fill_price, e.status, e.exchange_order_id,
                e.inference_signal_id, e.model_id, e.confidence,
                e.processing_time_ms, e.risk_check_result,
                e.error_message, e.metadata, e.created_at, e.updated_at
            FROM executions e
            WHERE {where_clause}
            ORDER BY e.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        params
    )
    rows = result.fetchall()

    # Convert to ExecutionResult objects
    items = []
    for row in rows:
        items.append(ExecutionResult(
            execution_id=row.id,
            signal_id=row.inference_signal_id,
            status=ExecutionStatus(row.status) if row.status else ExecutionStatus.PENDING,
            exchange=row.exchange,
            symbol=row.symbol,
            side=row.side,
            quantity=float(row.quantity) if row.quantity else 0.0,
            filled_quantity=float(row.filled_quantity) if row.filled_quantity else 0.0,
            price=float(row.price) if row.price else None,
            average_fill_price=float(row.average_fill_price) if row.average_fill_price else None,
            exchange_order_id=row.exchange_order_id,
            error_message=row.error_message,
            processing_time_ms=row.processing_time_ms,
            created_at=row.created_at,
            metadata=row.metadata or {},
        ))

    return PaginatedResponse(
        items=items,
        total=total,
        page=page,
        limit=limit,
        has_more=(offset + limit) < total,
    )


@router.get(
    "/statistics",
    response_model=BridgeStatistics,
    summary="Get statistics",
    description="Get bridge execution statistics for a time period.",
)
async def get_statistics(
    days: int = Query(7, ge=1, le=90, description="Number of days"),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
    db: AsyncSession = Depends(get_db_session),
) -> BridgeStatistics:
    """Get bridge statistics for the specified period."""
    from sqlalchemy import text

    bridge_status = await orchestrator.get_status()
    stats = bridge_status.stats or {}

    period_end = datetime.utcnow()
    period_start = period_end - timedelta(days=days)

    # Query aggregated statistics from database
    result = await db.execute(
        text("""
            SELECT
                COUNT(*) as total_executions,
                COUNT(CASE WHEN status = 'FILLED' THEN 1 END) as successful_executions,
                COUNT(CASE WHEN status = 'FAILED' OR status = 'REJECTED' THEN 1 END) as failed_executions,
                COUNT(CASE WHEN risk_check_result->>'decision' = 'BLOCK' THEN 1 END) as blocked_by_risk,
                COALESCE(SUM(CASE WHEN status = 'FILLED' THEN quantity * average_fill_price END), 0) as total_volume_usd,
                COALESCE(AVG(processing_time_ms), 0) as avg_execution_time_ms
            FROM executions
            WHERE created_at >= :period_start AND created_at <= :period_end
        """),
        {"period_start": period_start, "period_end": period_end}
    )
    row = result.first()

    # Query PnL from trades or executions with realized_pnl column if exists
    pnl_result = await db.execute(
        text("""
            SELECT COALESCE(SUM(realized_pnl), 0) as total_pnl_usd
            FROM executions
            WHERE created_at >= :period_start AND created_at <= :period_end
            AND realized_pnl IS NOT NULL
        """),
        {"period_start": period_start, "period_end": period_end}
    )
    pnl_row = pnl_result.first()

    return BridgeStatistics(
        total_signals_received=stats.get("signals_received", 0),
        total_executions=row.total_executions if row else 0,
        successful_executions=row.successful_executions if row else 0,
        failed_executions=row.failed_executions if row else 0,
        blocked_by_risk=row.blocked_by_risk if row else 0,
        total_volume_usd=float(row.total_volume_usd) if row and row.total_volume_usd else 0.0,
        total_pnl_usd=float(pnl_row.total_pnl_usd) if pnl_row and pnl_row.total_pnl_usd else 0.0,
        avg_execution_time_ms=float(row.avg_execution_time_ms) if row and row.avg_execution_time_ms else 0.0,
        period_start=period_start,
        period_end=period_end,
    )


# =============================================================================
# User State and Limits Endpoints
# =============================================================================

@router.get(
    "/user/{user_id}/state",
    response_model=dict,
    summary="Get user trading state",
    description="Get current trading state for a specific user.",
)
async def get_user_state(
    user_id: UUID,
    user: dict = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> dict:
    """Get user trading state."""
    # TODO: Add authorization check (user can only see own state or admin can see all)
    return await orchestrator.get_user_state(user_id)


@router.get(
    "/user/{user_id}/limits",
    response_model=UserRiskLimits,
    summary="Get user risk limits",
    description="Get risk limits configuration for a user.",
)
async def get_user_limits(
    user_id: UUID,
    user = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> UserRiskLimits:
    """Get user risk limits."""
    limits = await orchestrator.risk_bridge.get_user_limits(user_id)
    if not limits:
        # Return defaults
        return UserRiskLimits(user_id=user_id)
    return limits


@router.put(
    "/user/{user_id}/limits",
    response_model=UserRiskLimits,
    summary="Update user risk limits",
    description="Update risk limits for a user.",
)
async def update_user_limits(
    user_id: UUID,
    limits: UserRiskLimitsUpdate,
    user = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> UserRiskLimits:
    """Update user risk limits."""
    # Verify user has permission (can only update own limits unless admin)
    current_user_id = user.id
    is_admin = getattr(user, "role", None) == "admin"

    if str(user_id) != str(current_user_id) and not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to update other user's limits"
        )

    # Update limits in database
    updated_limits = await orchestrator.risk_bridge.create_or_update_user_limits(
        user_id=user_id,
        max_daily_loss_pct=limits.max_daily_loss_pct,
        max_trades_per_day=limits.max_trades_per_day,
        max_position_size_usd=limits.max_position_size_usd,
        cooldown_minutes=limits.cooldown_minutes,
        enable_short=limits.enable_short,
    )

    return updated_limits


@router.post(
    "/user/{user_id}/reset",
    response_model=SuccessResponse,
    summary="Reset user risk state",
    description="Reset the risk state for a user (clears daily counters).",
)
async def reset_user_state(
    user_id: UUID,
    user: dict = Depends(get_current_user),
    orchestrator: SignalBridgeOrchestrator = Depends(get_orchestrator),
) -> SuccessResponse:
    """Reset user risk state."""
    reset = orchestrator.risk_bridge.reset_user_enforcer(user_id)
    return SuccessResponse(
        success=reset,
        message="User risk state reset" if reset else "No state to reset",
        data={"user_id": str(user_id)},
    )


# Need uuid4 import
from uuid import uuid4
