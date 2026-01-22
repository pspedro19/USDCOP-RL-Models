"""
Execution Celery tasks.
"""

import asyncio
from datetime import datetime, timedelta
from uuid import UUID
from celery import shared_task
from sqlalchemy import select, and_

from app.tasks.celery_app import celery_app
from app.core.database import SyncSessionLocal
from app.models import Execution, ExchangeCredential
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
def execute_order(self, execution_id: str):
    """
    Execute an order on the exchange.

    This task:
    1. Loads the execution from database
    2. Gets exchange credentials
    3. Executes the order via the exchange adapter
    4. Updates execution with result
    """
    logger.info("Executing order", execution_id=execution_id)

    with SyncSessionLocal() as db:
        try:
            # Load execution
            execution = db.query(Execution).filter(
                Execution.id == UUID(execution_id)
            ).first()

            if not execution:
                logger.error("Execution not found", execution_id=execution_id)
                return {"success": False, "error": "Execution not found"}

            if execution.status != ExecutionStatus.PENDING.value:
                logger.info(
                    "Execution not pending",
                    execution_id=execution_id,
                    status=execution.status,
                )
                return {"success": False, "error": f"Status is {execution.status}"}

            # Get credential
            credential = db.query(ExchangeCredential).filter(
                ExchangeCredential.id == execution.credential_id
            ).first()

            if not credential:
                execution.status = ExecutionStatus.FAILED.value
                execution.error_message = "Credential not found"
                db.commit()
                return {"success": False, "error": "Credential not found"}

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

            # Execute order
            async def _execute():
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
                    return result
                finally:
                    await adapter.close()

            result = run_async(_execute())

            # Update execution
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

            db.commit()

            logger.info(
                "Order executed",
                execution_id=execution_id,
                status=result.status.value,
                filled=result.filled_quantity,
            )

            return {
                "success": result.success,
                "status": result.status.value,
                "order_id": result.order_id,
            }

        except Exception as e:
            logger.error(
                "Error executing order",
                execution_id=execution_id,
                error=str(e),
            )

            # Update execution status
            execution = db.query(Execution).filter(
                Execution.id == UUID(execution_id)
            ).first()
            if execution:
                execution.status = ExecutionStatus.FAILED.value
                execution.error_message = str(e)
                db.commit()

            # Retry with exponential backoff
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task
def retry_failed_executions():
    """
    Retry failed executions that are retryable.
    Called periodically by Celery Beat.
    """
    logger.info("Checking for retryable executions")

    with SyncSessionLocal() as db:
        # Get recently failed executions (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)

        executions = db.query(Execution).filter(
            and_(
                Execution.status == ExecutionStatus.FAILED.value,
                Execution.created_at >= cutoff,
                # Only retry if error seems transient
                Execution.error_message.like("%timeout%")
                | Execution.error_message.like("%connection%")
                | Execution.error_message.like("%rate limit%"),
            )
        ).limit(10).all()

        retried = 0
        for execution in executions:
            # Reset status and retry
            execution.status = ExecutionStatus.PENDING.value
            execution.error_message = None
            db.commit()

            # Queue for execution
            execute_order.delay(str(execution.id))
            retried += 1

        logger.info("Retried executions", count=retried)

        return {"retried": retried}


@celery_app.task
def check_order_status(execution_id: str):
    """
    Check and update status of a submitted order.
    """
    logger.info("Checking order status", execution_id=execution_id)

    with SyncSessionLocal() as db:
        execution = db.query(Execution).filter(
            Execution.id == UUID(execution_id)
        ).first()

        if not execution or not execution.exchange_order_id:
            return {"success": False, "error": "Invalid execution"}

        if execution.status not in [
            ExecutionStatus.SUBMITTED.value,
            ExecutionStatus.PARTIAL.value,
        ]:
            return {"success": True, "message": "Order not open"}

        # Get credential
        credential = db.query(ExchangeCredential).filter(
            ExchangeCredential.id == execution.credential_id
        ).first()

        if not credential:
            return {"success": False, "error": "Credential not found"}

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

        # Check order status
        async def _check():
            adapter = get_exchange_adapter(
                exchange=SupportedExchange(execution.exchange),
                api_key=api_key,
                api_secret=api_secret,
                passphrase=passphrase,
                testnet=credential.is_testnet,
            )
            try:
                result = await adapter.get_order_status(
                    symbol=execution.symbol,
                    order_id=execution.exchange_order_id,
                )
                return result
            finally:
                await adapter.close()

        result = run_async(_check())

        # Update execution
        execution.status = result.status.value
        execution.filled_quantity = result.filled_quantity
        execution.average_price = result.average_price
        execution.commission = result.commission
        execution.commission_asset = result.commission_asset

        db.commit()

        logger.info(
            "Order status updated",
            execution_id=execution_id,
            status=result.status.value,
        )

        return {
            "success": True,
            "status": result.status.value,
        }
