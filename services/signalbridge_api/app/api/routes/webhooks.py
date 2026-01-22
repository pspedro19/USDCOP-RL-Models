"""
Webhook routes for external signal sources (TradingView, etc).
"""

from typing import Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Header, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import settings
from app.models import User
from app.contracts.signal import SignalCreate, SignalWebhook, SignalAction
from app.services.signal import SignalService

router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


@router.post("/tradingview/{user_id}")
async def tradingview_webhook(
    user_id: UUID,
    data: SignalWebhook,
    db: AsyncSession = Depends(get_db),
):
    """
    Receive TradingView webhook signals.

    TradingView sends signals to this endpoint when alerts are triggered.
    The signal is validated and stored for processing.

    URL format: /api/webhooks/tradingview/{user_id}

    Expected payload:
    {
        "symbol": "BTCUSDT",
        "action": "buy|sell|close",
        "price": 50000.00,
        "quantity": 0.001,
        "stop_loss": 49000.00,
        "take_profit": 52000.00,
        "passphrase": "your-secret-passphrase",
        "comment": "Optional comment"
    }
    """
    # Verify user exists
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or inactive",
        )

    # Optional: Validate passphrase if configured
    # This would require storing webhook passphrases per user

    # Convert webhook to signal
    signal_service = SignalService(db)

    signal_data = SignalCreate(
        symbol=data.symbol.upper(),
        action=data.to_signal_action(),
        price=data.price,
        quantity=data.quantity,
        stop_loss=data.stop_loss,
        take_profit=data.take_profit,
        source="tradingview",
        metadata={
            "comment": data.comment,
            "raw_action": data.action,
        },
    )

    signal = await signal_service.create_signal(
        user_id=user_id,
        data=signal_data,
    )

    # Optionally trigger async processing here
    # await process_signal.delay(str(signal.id))

    return {
        "success": True,
        "signal_id": str(signal.id),
        "message": "Signal received and queued for processing",
    }


@router.post("/custom/{user_id}")
async def custom_webhook(
    user_id: UUID,
    data: SignalCreate,
    x_api_key: Optional[str] = Header(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Receive custom webhook signals from any source.

    This endpoint accepts signals in the standard SignalCreate format.
    Authentication is done via X-API-Key header.

    Note: For production, implement proper API key validation per user.
    """
    # Verify user exists
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found or inactive",
        )

    # TODO: Implement proper API key validation
    # For now, just accept the signal

    signal_service = SignalService(db)

    # Override source to track origin
    data.source = data.source or "custom_webhook"

    signal = await signal_service.create_signal(
        user_id=user_id,
        data=data,
    )

    return {
        "success": True,
        "signal_id": str(signal.id),
        "message": "Signal received and queued for processing",
    }
