"""
Signal processing Celery tasks.
"""

import asyncio
from uuid import UUID
from celery import shared_task
from sqlalchemy import select, and_

from app.tasks.celery_app import celery_app
from app.core.database import SyncSessionLocal
from app.models import Signal, TradingConfig, ExchangeCredential, Execution
from app.contracts.signal import SignalAction
from app.contracts.execution import OrderType, OrderSide, ExecutionStatus
from app.contracts.exchange import SupportedExchange
from app.adapters import get_exchange_adapter
from app.services.vault import vault_service
import structlog

logger = structlog.get_logger()


def run_async(coro):
    """Run async function in sync context."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3)
def process_signal(self, signal_id: str):
    """
    Process a single trading signal.

    This task:
    1. Loads the signal from database
    2. Validates trading conditions (enabled, limits, etc.)
    3. Creates an execution request
    4. Executes the order on the exchange
    5. Updates signal and execution status
    """
    logger.info("Processing signal", signal_id=signal_id)

    with SyncSessionLocal() as db:
        try:
            # Load signal
            signal = db.query(Signal).filter(Signal.id == UUID(signal_id)).first()

            if not signal:
                logger.error("Signal not found", signal_id=signal_id)
                return {"success": False, "error": "Signal not found"}

            if signal.is_processed:
                logger.info("Signal already processed", signal_id=signal_id)
                return {"success": True, "message": "Already processed"}

            # Load trading config
            config = db.query(TradingConfig).filter(
                TradingConfig.user_id == signal.user_id
            ).first()

            if not config or not config.trading_enabled:
                logger.info("Trading disabled", user_id=str(signal.user_id))
                return {"success": False, "error": "Trading disabled"}

            # Check symbol filters
            if config.blocked_symbols and signal.symbol in config.blocked_symbols:
                logger.info("Symbol blocked", symbol=signal.symbol)
                return {"success": False, "error": "Symbol blocked"}

            if config.allowed_symbols and signal.symbol not in config.allowed_symbols:
                logger.info("Symbol not allowed", symbol=signal.symbol)
                return {"success": False, "error": "Symbol not in allowed list"}

            # Get exchange credentials
            exchange = SupportedExchange(config.default_exchange) if config.default_exchange else None

            if not exchange:
                logger.error("No default exchange configured")
                return {"success": False, "error": "No default exchange"}

            credential = db.query(ExchangeCredential).filter(
                and_(
                    ExchangeCredential.user_id == signal.user_id,
                    ExchangeCredential.exchange == exchange.value,
                    ExchangeCredential.is_active == True,
                    ExchangeCredential.is_valid == True,
                )
            ).first()

            if not credential:
                logger.error("No valid credentials for exchange", exchange=exchange.value)
                return {"success": False, "error": "No valid credentials"}

            # Determine order side
            if signal.action == SignalAction.BUY:
                side = OrderSide.BUY
            elif signal.action == SignalAction.SELL:
                side = OrderSide.SELL
            elif signal.action == SignalAction.CLOSE:
                # For close, we would need to check current position
                # For now, assume it's a sell
                side = OrderSide.SELL
            else:
                logger.info("HOLD signal, no action", signal_id=signal_id)
                signal.is_processed = True
                db.commit()
                return {"success": True, "message": "HOLD signal, no action needed"}

            # Create execution
            execution = Execution(
                user_id=signal.user_id,
                signal_id=signal.id,
                exchange=exchange.value,
                credential_id=credential.id,
                symbol=signal.symbol,
                side=side.value,
                order_type=OrderType.MARKET.value,
                quantity=signal.quantity or 0.001,  # Default quantity if not specified
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                status=ExecutionStatus.PENDING.value,
            )
            db.add(execution)
            db.flush()

            # Execute order
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

            # Execute order via adapter
            async def _execute():
                adapter = get_exchange_adapter(
                    exchange=exchange,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                    testnet=credential.is_testnet,
                )
                try:
                    result = await adapter.place_market_order(
                        symbol=signal.symbol,
                        side=side,
                        quantity=execution.quantity,
                    )
                    return result
                finally:
                    await adapter.close()

            result = run_async(_execute())

            # Update execution with result
            execution.exchange_order_id = result.order_id
            execution.status = result.status.value
            execution.filled_quantity = result.filled_quantity
            execution.average_price = result.average_price
            execution.commission = result.commission
            execution.commission_asset = result.commission_asset
            execution.executed_at = result.executed_at
            execution.error_message = result.error_message
            execution.raw_response = result.raw_response

            # Mark signal as processed
            signal.is_processed = True
            signal.execution_id = execution.id

            db.commit()

            logger.info(
                "Signal processed successfully",
                signal_id=signal_id,
                execution_id=str(execution.id),
                status=result.status.value,
            )

            return {
                "success": result.success,
                "execution_id": str(execution.id),
                "status": result.status.value,
            }

        except Exception as e:
            logger.error("Error processing signal", signal_id=signal_id, error=str(e))
            db.rollback()

            # Retry with exponential backoff
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task
def process_pending_signals():
    """
    Process all pending signals.
    Called periodically by Celery Beat.
    """
    logger.info("Processing pending signals")

    with SyncSessionLocal() as db:
        # Get unprocessed signals
        signals = db.query(Signal).filter(
            Signal.is_processed == False
        ).order_by(Signal.created_at).limit(10).all()

        processed = 0
        for signal in signals:
            # Queue signal for processing
            process_signal.delay(str(signal.id))
            processed += 1

        logger.info("Queued signals for processing", count=processed)

        return {"queued": processed}
