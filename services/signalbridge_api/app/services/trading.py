"""
Trading configuration service.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from uuid import UUID
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import TradingConfig, Execution, ExchangeCredential
from app.contracts.trading import (
    TradingConfig as TradingConfigContract,
    TradingConfigUpdate,
    TradingStatus,
)
from app.contracts.exchange import SupportedExchange
from app.core.exceptions import NotFoundError


class TradingService:
    """Service for trading configuration management."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_config(self, user_id: UUID) -> Optional[TradingConfig]:
        """Get trading config for a user."""
        result = await self.db.execute(
            select(TradingConfig).where(TradingConfig.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_or_create_config(self, user_id: UUID) -> TradingConfig:
        """Get or create trading config for a user."""
        config = await self.get_config(user_id)

        if not config:
            config = TradingConfig(
                user_id=user_id,
                trading_enabled=False,
            )
            self.db.add(config)
            await self.db.commit()
            await self.db.refresh(config)

        return config

    async def update_config(
        self,
        user_id: UUID,
        data: TradingConfigUpdate,
    ) -> TradingConfig:
        """
        Update trading configuration.

        Args:
            user_id: User ID
            data: Update data

        Returns:
            Updated config
        """
        config = await self.get_or_create_config(user_id)

        update_data = data.model_dump(exclude_unset=True)

        for key, value in update_data.items():
            if value is not None:
                setattr(config, key, value)

        config.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(config)

        return config

    async def toggle_trading(
        self,
        user_id: UUID,
        enabled: bool,
    ) -> TradingConfig:
        """
        Enable or disable trading.

        Args:
            user_id: User ID
            enabled: Whether to enable trading

        Returns:
            Updated config
        """
        config = await self.get_or_create_config(user_id)
        config.trading_enabled = enabled
        config.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(config)

        return config

    async def get_status(self, user_id: UUID) -> TradingStatus:
        """
        Get current trading status.

        Args:
            user_id: User ID

        Returns:
            Trading status
        """
        config = await self.get_or_create_config(user_id)

        # Count today's trades
        today_start = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        daily_count_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    Execution.user_id == user_id,
                    Execution.created_at >= today_start,
                )
            )
        )
        daily_trades = daily_count_result.scalar() or 0

        # Count active positions (open orders)
        active_result = await self.db.execute(
            select(func.count(Execution.id)).where(
                and_(
                    Execution.user_id == user_id,
                    Execution.status.in_(["submitted", "partial"]),
                )
            )
        )
        active_positions = active_result.scalar() or 0

        # Get last trade timestamp
        last_trade_result = await self.db.execute(
            select(Execution.created_at)
            .where(Execution.user_id == user_id)
            .order_by(Execution.created_at.desc())
            .limit(1)
        )
        last_trade = last_trade_result.scalar_one_or_none()

        # Get exchange connection status
        credentials_result = await self.db.execute(
            select(ExchangeCredential).where(
                and_(
                    ExchangeCredential.user_id == user_id,
                    ExchangeCredential.is_active == True,
                )
            )
        )
        credentials = credentials_result.scalars().all()

        exchange_connections = {
            cred.exchange: cred.is_valid for cred in credentials
        }

        return TradingStatus(
            trading_enabled=config.trading_enabled,
            active_positions=active_positions,
            daily_trades_count=daily_trades,
            daily_trades_limit=config.max_daily_trades,
            last_trade_at=last_trade,
            exchange_connections=exchange_connections,
        )

    async def can_trade(self, user_id: UUID) -> tuple[bool, Optional[str]]:
        """
        Check if user can trade.

        Args:
            user_id: User ID

        Returns:
            Tuple of (can_trade, reason)
        """
        config = await self.get_config(user_id)

        if not config:
            return False, "Trading not configured"

        if not config.trading_enabled:
            return False, "Trading is disabled"

        # Check daily limit
        status = await self.get_status(user_id)

        if status.daily_trades_count >= config.max_daily_trades:
            return False, "Daily trade limit reached"

        if status.active_positions >= config.max_concurrent_positions:
            return False, "Maximum concurrent positions reached"

        return True, None

    def to_response(self, config: TradingConfig) -> TradingConfigContract:
        """Convert config model to response."""
        return TradingConfigContract(
            id=config.id,
            user_id=config.user_id,
            trading_enabled=config.trading_enabled,
            default_exchange=SupportedExchange(config.default_exchange)
            if config.default_exchange
            else None,
            max_position_size=config.max_position_size,
            stop_loss_percent=config.stop_loss_percent,
            take_profit_percent=config.take_profit_percent,
            use_trailing_stop=config.use_trailing_stop,
            trailing_stop_percent=config.trailing_stop_percent,
            allowed_symbols=config.allowed_symbols or [],
            blocked_symbols=config.blocked_symbols or [],
            max_daily_trades=config.max_daily_trades,
            max_concurrent_positions=config.max_concurrent_positions,
            created_at=config.created_at,
            updated_at=config.updated_at,
        )
