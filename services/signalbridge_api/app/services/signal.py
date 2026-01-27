"""
Signal service for managing trading signals.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from uuid import UUID
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Signal
from app.contracts.signal import (
    TradingSignal,
    SignalCreate,
    SignalAction,
    SignalStats,
    SignalFilter,
)
from app.core.exceptions import NotFoundError


class SignalService:
    """Service for signal management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_signal(
        self,
        signal_id: UUID,
        user_id: UUID,
    ) -> Optional[Signal]:
        """Get a specific signal."""
        result = await self.db.execute(
            select(Signal).where(
                and_(
                    Signal.id == signal_id,
                    Signal.user_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_signals(
        self,
        user_id: UUID,
        filters: Optional[SignalFilter] = None,
        page: int = 1,
        limit: int = 20,
    ) -> tuple[List[Signal], int]:
        """
        Get signals for a user with filtering and pagination.

        Args:
            user_id: User ID
            filters: Optional filters
            page: Page number
            limit: Items per page

        Returns:
            Tuple of (signals, total_count)
        """
        query = select(Signal).where(Signal.user_id == user_id)

        if filters:
            if filters.action is not None:
                query = query.where(Signal.action == filters.action)
            if filters.symbol:
                query = query.where(Signal.symbol == filters.symbol.upper())
            if filters.source:
                query = query.where(Signal.source == filters.source)
            if filters.is_processed is not None:
                query = query.where(Signal.is_processed == filters.is_processed)
            if filters.since:
                query = query.where(Signal.created_at >= filters.since)
            if filters.until:
                query = query.where(Signal.created_at <= filters.until)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        query = query.order_by(desc(Signal.created_at))
        query = query.offset((page - 1) * limit).limit(limit)

        result = await self.db.execute(query)
        signals = list(result.scalars().all())

        return signals, total

    async def create_signal(
        self,
        user_id: UUID,
        data: SignalCreate,
    ) -> Signal:
        """
        Create a new trading signal.

        Args:
            user_id: User ID
            data: Signal data

        Returns:
            Created signal
        """
        signal = Signal(
            user_id=user_id,
            symbol=data.symbol.upper(),
            action=data.action,
            price=data.price,
            quantity=data.quantity,
            stop_loss=data.stop_loss,
            take_profit=data.take_profit,
            source=data.source,
            metadata=data.metadata,
            is_processed=False,
        )

        self.db.add(signal)
        await self.db.commit()
        await self.db.refresh(signal)

        return signal

    async def mark_processed(
        self,
        signal_id: UUID,
        execution_id: Optional[UUID] = None,
    ) -> Signal:
        """
        Mark a signal as processed.

        Args:
            signal_id: Signal ID
            execution_id: Optional execution ID

        Returns:
            Updated signal
        """
        result = await self.db.execute(
            select(Signal).where(Signal.id == signal_id)
        )
        signal = result.scalar_one_or_none()

        if not signal:
            raise NotFoundError(
                message="Signal not found",
                resource_type="Signal",
                resource_id=str(signal_id),
            )

        signal.is_processed = True
        signal.processed_at = datetime.utcnow()
        signal.execution_id = execution_id

        await self.db.commit()
        await self.db.refresh(signal)

        return signal

    async def get_recent_signals(
        self,
        user_id: UUID,
        limit: int = 5,
    ) -> List[Signal]:
        """Get most recent signals."""
        result = await self.db.execute(
            select(Signal)
            .where(Signal.user_id == user_id)
            .order_by(desc(Signal.created_at))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_pending_signals(
        self,
        user_id: Optional[UUID] = None,
    ) -> List[Signal]:
        """Get unprocessed signals."""
        query = select(Signal).where(Signal.is_processed == False)

        if user_id:
            query = query.where(Signal.user_id == user_id)

        query = query.order_by(Signal.created_at)

        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def get_stats(
        self,
        user_id: UUID,
        days: int = 7,
    ) -> SignalStats:
        """
        Get signal statistics.

        Args:
            user_id: User ID
            days: Number of days to include

        Returns:
            Signal statistics
        """
        since = datetime.utcnow() - timedelta(days=days)

        # Total signals
        total_result = await self.db.execute(
            select(func.count(Signal.id)).where(
                and_(
                    Signal.user_id == user_id,
                    Signal.created_at >= since,
                )
            )
        )
        total = total_result.scalar() or 0

        # By action
        buy_result = await self.db.execute(
            select(func.count(Signal.id)).where(
                and_(
                    Signal.user_id == user_id,
                    Signal.action == SignalAction.BUY,
                    Signal.created_at >= since,
                )
            )
        )
        buy_count = buy_result.scalar() or 0

        sell_result = await self.db.execute(
            select(func.count(Signal.id)).where(
                and_(
                    Signal.user_id == user_id,
                    Signal.action == SignalAction.SELL,
                    Signal.created_at >= since,
                )
            )
        )
        sell_count = sell_result.scalar() or 0

        close_result = await self.db.execute(
            select(func.count(Signal.id)).where(
                and_(
                    Signal.user_id == user_id,
                    Signal.action == SignalAction.CLOSE,
                    Signal.created_at >= since,
                )
            )
        )
        close_count = close_result.scalar() or 0

        # Processed
        processed_result = await self.db.execute(
            select(func.count(Signal.id)).where(
                and_(
                    Signal.user_id == user_id,
                    Signal.is_processed == True,
                    Signal.created_at >= since,
                )
            )
        )
        processed_count = processed_result.scalar() or 0

        pending_count = total - processed_count
        success_rate = (processed_count / total * 100) if total > 0 else 0

        return SignalStats(
            total_signals=total,
            buy_signals=buy_count,
            sell_signals=sell_count,
            close_signals=close_count,
            processed_signals=processed_count,
            pending_signals=pending_count,
            success_rate=round(success_rate, 2),
            period_days=days,
        )

    def to_response(self, signal: Signal) -> TradingSignal:
        """Convert signal model to response."""
        return TradingSignal(
            id=signal.id,
            user_id=signal.user_id,
            symbol=signal.symbol,
            action=SignalAction(signal.action),
            price=signal.price,
            quantity=signal.quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            source=signal.source,
            metadata=signal.signal_metadata or {},
            created_at=signal.created_at,
            processed_at=signal.processed_at,
            is_processed=signal.is_processed,
            execution_id=signal.execution_id,
        )
