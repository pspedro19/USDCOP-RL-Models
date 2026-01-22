"""
Execution service for managing trade executions.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from uuid import UUID
from sqlalchemy import select, func, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Execution, ExchangeCredential
from app.contracts.execution import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionCreate,
    ExecutionSummary,
    ExecutionStats,
    ExecutionStatus,
    ExecutionFilter,
    OrderType,
    OrderSide,
    TodayStats,
)
from app.contracts.exchange import SupportedExchange
from app.adapters import get_exchange_adapter
from app.services.vault import vault_service
from app.core.exceptions import NotFoundError, ExchangeError, ValidationError


class ExecutionService:
    """Service for execution management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_execution(
        self,
        execution_id: UUID,
        user_id: UUID,
    ) -> Optional[Execution]:
        """Get a specific execution."""
        result = await self.db.execute(
            select(Execution).where(
                and_(
                    Execution.id == execution_id,
                    Execution.user_id == user_id,
                )
            )
        )
        return result.scalar_one_or_none()

    async def get_executions(
        self,
        user_id: UUID,
        filters: Optional[ExecutionFilter] = None,
        page: int = 1,
        limit: int = 20,
    ) -> Tuple[List[Execution], int]:
        """
        Get executions for a user with filtering and pagination.

        Args:
            user_id: User ID
            filters: Optional filters
            page: Page number
            limit: Items per page

        Returns:
            Tuple of (executions, total_count)
        """
        query = select(Execution).where(Execution.user_id == user_id)

        if filters:
            if filters.exchange:
                query = query.where(Execution.exchange == filters.exchange.value)
            if filters.symbol:
                query = query.where(Execution.symbol == filters.symbol.upper())
            if filters.side:
                query = query.where(Execution.side == filters.side.value)
            if filters.status:
                query = query.where(Execution.status == filters.status.value)
            if filters.since:
                query = query.where(Execution.created_at >= filters.since)
            if filters.until:
                query = query.where(Execution.created_at <= filters.until)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0

        # Apply pagination
        query = query.order_by(desc(Execution.created_at))
        query = query.offset((page - 1) * limit).limit(limit)

        result = await self.db.execute(query)
        executions = list(result.scalars().all())

        return executions, total

    async def create_execution(
        self,
        user_id: UUID,
        data: ExecutionCreate,
    ) -> Execution:
        """
        Create a new execution request.

        Args:
            user_id: User ID
            data: Execution data

        Returns:
            Created execution
        """
        # Verify credential exists and belongs to user
        cred_result = await self.db.execute(
            select(ExchangeCredential).where(
                and_(
                    ExchangeCredential.id == data.credential_id,
                    ExchangeCredential.user_id == user_id,
                )
            )
        )
        credential = cred_result.scalar_one_or_none()

        if not credential:
            raise NotFoundError(
                message="Exchange credential not found",
                resource_type="ExchangeCredential",
                resource_id=str(data.credential_id),
            )

        execution = Execution(
            user_id=user_id,
            signal_id=data.signal_id,
            exchange=data.exchange.value,
            credential_id=data.credential_id,
            symbol=data.symbol.upper(),
            side=data.side.value,
            order_type=data.order_type.value,
            quantity=data.quantity,
            price=data.price,
            stop_loss=data.stop_loss,
            take_profit=data.take_profit,
            status=ExecutionStatus.PENDING.value,
            metadata=data.metadata,
        )

        self.db.add(execution)
        await self.db.commit()
        await self.db.refresh(execution)

        return execution

    async def execute_order(
        self,
        execution_id: UUID,
        user_id: UUID,
    ) -> Execution:
        """
        Execute a pending order on the exchange.

        Args:
            execution_id: Execution ID
            user_id: User ID

        Returns:
            Updated execution with result
        """
        execution = await self.get_execution(execution_id, user_id)

        if not execution:
            raise NotFoundError(
                message="Execution not found",
                resource_type="Execution",
                resource_id=str(execution_id),
            )

        if execution.status != ExecutionStatus.PENDING.value:
            raise ValidationError(
                message=f"Execution is not pending (status: {execution.status})"
            )

        # Get credential
        cred_result = await self.db.execute(
            select(ExchangeCredential).where(
                ExchangeCredential.id == execution.credential_id
            )
        )
        credential = cred_result.scalar_one_or_none()

        if not credential:
            execution.status = ExecutionStatus.FAILED.value
            execution.error_message = "Credential not found"
            await self.db.commit()
            raise NotFoundError(
                message="Exchange credential not found",
                resource_type="ExchangeCredential",
            )

        # Decrypt credentials
        api_key = vault_service.decrypt(
            credential.encrypted_api_key,
            credential.key_version,
        )
        api_secret = vault_service.decrypt(
            credential.encrypted_api_secret,
            credential.key_version,
        )

        passphrase = None
        if credential.encrypted_passphrase:
            passphrase = vault_service.decrypt(
                credential.encrypted_passphrase,
                credential.key_version,
            )

        # Create adapter and execute
        adapter = get_exchange_adapter(
            exchange=SupportedExchange(execution.exchange),
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=credential.is_testnet,
        )

        try:
            if execution.order_type == OrderType.MARKET.value:
                result = await adapter.place_market_order(
                    symbol=execution.symbol,
                    side=OrderSide(execution.side),
                    quantity=execution.quantity,
                )
            else:
                result = await adapter.place_limit_order(
                    symbol=execution.symbol,
                    side=OrderSide(execution.side),
                    quantity=execution.quantity,
                    price=execution.price,
                )

            # Update execution with result
            execution.exchange_order_id = result.order_id
            execution.status = result.status.value
            execution.filled_quantity = result.filled_quantity
            execution.average_price = result.average_price
            execution.commission = result.commission
            execution.commission_asset = result.commission_asset
            execution.executed_at = result.executed_at or datetime.utcnow()
            execution.error_message = result.error_message
            execution.raw_response = result.raw_response

            # Update credential last_used
            credential.last_used = datetime.utcnow()

            await self.db.commit()
            await self.db.refresh(execution)

            return execution

        except Exception as e:
            execution.status = ExecutionStatus.FAILED.value
            execution.error_message = str(e)
            await self.db.commit()
            raise ExchangeError(message=str(e))
        finally:
            await adapter.close()

    async def cancel_execution(
        self,
        execution_id: UUID,
        user_id: UUID,
    ) -> Execution:
        """
        Cancel an open order.

        Args:
            execution_id: Execution ID
            user_id: User ID

        Returns:
            Updated execution
        """
        execution = await self.get_execution(execution_id, user_id)

        if not execution:
            raise NotFoundError(
                message="Execution not found",
                resource_type="Execution",
                resource_id=str(execution_id),
            )

        if execution.status not in [
            ExecutionStatus.SUBMITTED.value,
            ExecutionStatus.PARTIAL.value,
        ]:
            raise ValidationError(
                message=f"Cannot cancel execution with status: {execution.status}"
            )

        if not execution.exchange_order_id:
            raise ValidationError(message="No exchange order ID to cancel")

        # Get credential
        cred_result = await self.db.execute(
            select(ExchangeCredential).where(
                ExchangeCredential.id == execution.credential_id
            )
        )
        credential = cred_result.scalar_one_or_none()

        if not credential:
            raise NotFoundError(
                message="Exchange credential not found",
                resource_type="ExchangeCredential",
            )

        # Decrypt credentials
        api_key = vault_service.decrypt(
            credential.encrypted_api_key,
            credential.key_version,
        )
        api_secret = vault_service.decrypt(
            credential.encrypted_api_secret,
            credential.key_version,
        )

        passphrase = None
        if credential.encrypted_passphrase:
            passphrase = vault_service.decrypt(
                credential.encrypted_passphrase,
                credential.key_version,
            )

        # Create adapter and cancel
        adapter = get_exchange_adapter(
            exchange=SupportedExchange(execution.exchange),
            api_key=api_key,
            api_secret=api_secret,
            passphrase=passphrase,
            testnet=credential.is_testnet,
        )

        try:
            result = await adapter.cancel_order(
                symbol=execution.symbol,
                order_id=execution.exchange_order_id,
            )

            execution.status = result.status.value
            execution.error_message = result.error_message

            await self.db.commit()
            await self.db.refresh(execution)

            return execution

        finally:
            await adapter.close()

    async def get_stats(
        self,
        user_id: UUID,
        days: int = 7,
    ) -> ExecutionStats:
        """
        Get execution statistics.

        Args:
            user_id: User ID
            days: Number of days to include

        Returns:
            Execution statistics
        """
        since = datetime.utcnow() - timedelta(days=days)

        base_filter = and_(
            Execution.user_id == user_id,
            Execution.created_at >= since,
        )

        # Total executions
        total_result = await self.db.execute(
            select(func.count(Execution.id)).where(base_filter)
        )
        total = total_result.scalar() or 0

        # Successful executions
        success_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    base_filter,
                    Execution.status == ExecutionStatus.FILLED.value,
                )
            )
        )
        successful = success_result.scalar() or 0

        # Failed executions
        failed_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    base_filter,
                    Execution.status.in_([
                        ExecutionStatus.FAILED.value,
                        ExecutionStatus.REJECTED.value,
                    ]),
                )
            )
        )
        failed = failed_result.scalar() or 0

        # Pending executions
        pending_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    base_filter,
                    Execution.status.in_([
                        ExecutionStatus.PENDING.value,
                        ExecutionStatus.SUBMITTED.value,
                    ]),
                )
            )
        )
        pending = pending_result.scalar() or 0

        # Total volume
        volume_result = await self.db.execute(
            select(func.sum(Execution.filled_quantity * Execution.average_price)).where(
                and_(
                    base_filter,
                    Execution.status == ExecutionStatus.FILLED.value,
                )
            )
        )
        total_volume = volume_result.scalar() or 0

        # Total commission
        commission_result = await self.db.execute(
            select(func.sum(Execution.commission)).where(base_filter)
        )
        total_commission = commission_result.scalar() or 0

        win_rate = (successful / total * 100) if total > 0 else 0

        return ExecutionStats(
            total_executions=total,
            successful_executions=successful,
            failed_executions=failed,
            pending_executions=pending,
            total_volume=float(total_volume),
            total_commission=float(total_commission),
            win_rate=round(win_rate, 2),
            period_days=days,
        )

    async def get_today_stats(self, user_id: UUID) -> TodayStats:
        """Get today's trading statistics."""
        today_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        base_filter = and_(
            Execution.user_id == user_id,
            Execution.created_at >= today_start,
        )

        # Total trades today
        total_result = await self.db.execute(
            select(func.count(Execution.id)).where(base_filter)
        )
        total = total_result.scalar() or 0

        # Successful trades
        success_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    base_filter,
                    Execution.status == ExecutionStatus.FILLED.value,
                )
            )
        )
        successful = success_result.scalar() or 0

        # Failed trades
        failed_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    base_filter,
                    Execution.status.in_([
                        ExecutionStatus.FAILED.value,
                        ExecutionStatus.REJECTED.value,
                    ]),
                )
            )
        )
        failed = failed_result.scalar() or 0

        # Total volume
        volume_result = await self.db.execute(
            select(func.sum(Execution.filled_quantity * Execution.average_price)).where(
                and_(
                    base_filter,
                    Execution.status == ExecutionStatus.FILLED.value,
                )
            )
        )
        total_volume = volume_result.scalar() or 0

        win_rate = (successful / total * 100) if total > 0 else 0

        return TodayStats(
            total_trades=total,
            successful_trades=successful,
            failed_trades=failed,
            total_volume=float(total_volume),
            total_pnl=0.0,  # Would need position tracking for accurate PnL
            win_rate=round(win_rate, 2),
        )

    def to_response(self, execution: Execution) -> ExecutionResult:
        """Convert execution model to response."""
        return ExecutionResult(
            id=execution.id,
            request_id=execution.id,
            exchange_order_id=execution.exchange_order_id,
            status=ExecutionStatus(execution.status),
            filled_quantity=execution.filled_quantity,
            average_price=execution.average_price,
            commission=execution.commission,
            commission_asset=execution.commission_asset,
            executed_at=execution.executed_at,
            error_message=execution.error_message,
            raw_response=execution.raw_response,
        )

    def to_summary(self, execution: Execution) -> ExecutionSummary:
        """Convert execution model to summary response."""
        return ExecutionSummary(
            id=execution.id,
            exchange=SupportedExchange(execution.exchange),
            symbol=execution.symbol,
            side=OrderSide(execution.side),
            order_type=OrderType(execution.order_type),
            quantity=execution.quantity,
            status=ExecutionStatus(execution.status),
            filled_quantity=execution.filled_quantity,
            average_price=execution.average_price,
            created_at=execution.created_at,
            executed_at=execution.executed_at,
        )
