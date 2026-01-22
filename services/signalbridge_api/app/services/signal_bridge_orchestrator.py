"""
Signal Bridge Orchestrator
==========================

Central orchestrator that manages the flow:
Signal -> Validation -> Execution -> Persistence

Coordinates between:
- RiskBridgeService (risk validation)
- ExchangeAdapterFactory (exchange execution)
- VaultService (credentials)
- TradingFlagsRedis (trading mode control)
- WebSocketBridge (signal consumption)

Design Pattern: Orchestrator/Coordinator
- Single point of coordination for signal processing
- Manages transaction-like flow with proper error handling

Author: Trading Team
Version: 1.0.0
Date: 2026-01-22
"""

import asyncio
import logging
import os
from datetime import datetime
from time import perf_counter
from typing import Optional, Dict, Any, List
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.signal_bridge import (
    InferenceSignalCreate,
    ManualSignalCreate,
    ExecutionResult,
    RiskCheckResult,
    RiskDecision,
    RiskReason,
    BridgeStatus,
    BridgeHealthCheck,
    BridgeEventType,
    BridgeStatistics,
    TradingMode,
    KillSwitchRequest,
    ExecutionAuditCreate,
    UserTradingState,
)
from app.contracts.execution import ExecutionStatus, OrderSide, OrderType
from app.contracts.exchange import SupportedExchange
from app.adapters.factory import ExchangeAdapterFactory
from app.adapters.base import OrderResult
from app.services.risk_bridge import RiskBridgeService
from app.services.vault import VaultService
from app.core.config import settings

# Import TradingFlagsRedis if available
try:
    from src.config.trading_flags import (
        TradingFlagsRedis,
        get_trading_flags_redis,
        TradingMode as CoreTradingMode,
    )
    TRADING_FLAGS_AVAILABLE = True
except ImportError:
    TRADING_FLAGS_AVAILABLE = False
    TradingFlagsRedis = None
    get_trading_flags_redis = None

logger = logging.getLogger(__name__)


class SignalBridgeOrchestrator:
    """
    Orchestrates the signal-to-execution flow.

    Responsibilities:
    1. Receive signals (from WebSocket or API)
    2. Validate against risk rules
    3. Execute on exchange
    4. Persist results
    5. Notify subscribers

    Usage:
        orchestrator = SignalBridgeOrchestrator(db_session)

        # Process a signal
        result = await orchestrator.process_signal(signal, user_id, credential_id)

        # Get current state
        state = await orchestrator.get_user_state(user_id)

        # Activate kill switch
        await orchestrator.activate_kill_switch(reason="Emergency")
    """

    def __init__(
        self,
        db_session: AsyncSession,
        vault_service: Optional[VaultService] = None,
        risk_bridge: Optional[RiskBridgeService] = None,
        trading_mode: Optional[TradingMode] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            db_session: Database session
            vault_service: Vault service for credentials
            risk_bridge: Risk bridge service
            trading_mode: Override trading mode
        """
        self.db_session = db_session
        self.vault_service = vault_service or VaultService()
        self.risk_bridge = risk_bridge or RiskBridgeService(db_session)

        # Trading mode
        self._trading_mode_override = trading_mode
        self._trading_flags = None
        if TRADING_FLAGS_AVAILABLE and get_trading_flags_redis:
            try:
                self._trading_flags = get_trading_flags_redis()
            except Exception as e:
                logger.warning(f"Could not initialize TradingFlagsRedis: {e}")

        # Statistics
        self._start_time = datetime.utcnow()
        self._signals_received = 0
        self._executions_total = 0
        self._executions_success = 0
        self._executions_failed = 0
        self._blocked_by_risk = 0
        self._last_signal_at: Optional[datetime] = None
        self._last_execution_at: Optional[datetime] = None

        # Event listeners for notifications
        self._listeners: List[callable] = []

        logger.info(f"SignalBridgeOrchestrator initialized, mode={self.trading_mode}")

    @property
    def trading_mode(self) -> TradingMode:
        """Get current trading mode."""
        if self._trading_mode_override:
            return self._trading_mode_override

        if self._trading_flags:
            if self._trading_flags.kill_switch:
                return TradingMode.KILLED
            if self._trading_flags.maintenance_mode:
                return TradingMode.DISABLED
            if self._trading_flags.paper_trading:
                return TradingMode.PAPER
            return TradingMode.LIVE

        # Default from settings
        mode_str = getattr(settings, "trading_mode", "PAPER").upper()
        return TradingMode(mode_str)

    @property
    def is_active(self) -> bool:
        """Check if bridge is active and processing signals."""
        return self.trading_mode not in (TradingMode.KILLED, TradingMode.DISABLED)

    async def process_signal(
        self,
        signal: InferenceSignalCreate,
        user_id: UUID,
        credential_id: Optional[UUID] = None,
    ) -> ExecutionResult:
        """
        Process a signal through the full pipeline.

        Flow:
        1. Validate trading mode
        2. Check risk rules
        3. Get exchange credentials
        4. Execute order
        5. Record result
        6. Notify listeners

        Args:
            signal: Inference signal to process
            user_id: User identifier
            credential_id: Exchange credential ID (optional, uses signal's if None)

        Returns:
            ExecutionResult with outcome
        """
        start_time = perf_counter()
        self._signals_received += 1
        self._last_signal_at = datetime.utcnow()

        cred_id = credential_id or signal.credential_id
        execution_id = uuid4()

        logger.info(
            f"Processing signal {signal.signal_id}: "
            f"{signal.action_name} {signal.symbol} (confidence={signal.confidence:.3f})"
        )

        # Audit: Signal received
        await self._audit_event(
            execution_id,
            BridgeEventType.SIGNAL_RECEIVED,
            {"signal": signal.dict(), "user_id": str(user_id)},
        )

        # Step 1: Check trading mode
        if self.trading_mode == TradingMode.KILLED:
            return self._blocked_result(
                execution_id,
                signal,
                RiskReason.KILL_SWITCH,
                "Kill switch is active",
            )

        if self.trading_mode == TradingMode.DISABLED:
            return self._blocked_result(
                execution_id,
                signal,
                RiskReason.KILL_SWITCH,
                "Trading is disabled",
            )

        # Step 2: Risk validation
        try:
            quantity = await self._calculate_position_size(signal, user_id)
            current_price = await self._get_current_price(signal.symbol)

            await self._audit_event(
                execution_id,
                BridgeEventType.RISK_CHECK_STARTED,
                {"quantity": quantity, "price": current_price},
            )

            risk_result = await self.risk_bridge.validate_execution(
                signal=signal,
                quantity=quantity,
                user_id=user_id,
                current_price=current_price,
            )

            if risk_result.is_blocked:
                self._blocked_by_risk += 1
                await self._audit_event(
                    execution_id,
                    BridgeEventType.RISK_CHECK_FAILED,
                    risk_result.metadata,
                )
                return self._blocked_result(
                    execution_id,
                    signal,
                    risk_result.reason,
                    risk_result.message,
                    risk_result,
                )

            # Use adjusted size if reduced
            if risk_result.adjusted_size:
                quantity = risk_result.adjusted_size

            await self._audit_event(
                execution_id,
                BridgeEventType.RISK_CHECK_PASSED,
                {"quantity": quantity, "adjusted": risk_result.adjusted_size is not None},
            )

        except Exception as e:
            logger.error(f"Risk validation error: {e}")
            return self._error_result(execution_id, signal, f"Risk validation failed: {e}")

        # Step 3: Handle based on trading mode
        if self.trading_mode == TradingMode.SHADOW:
            # Log only, no execution
            return ExecutionResult(
                success=True,
                execution_id=execution_id,
                signal_id=signal.signal_id,
                status=ExecutionStatus.FILLED,  # Simulated success
                symbol=signal.symbol,
                side=self._action_to_side(signal.action),
                requested_quantity=quantity,
                filled_quantity=quantity,
                filled_price=current_price or 0,
                risk_check=risk_result,
                processing_time_ms=(perf_counter() - start_time) * 1000,
                metadata={"mode": "shadow", "simulated": True},
            )

        if self.trading_mode == TradingMode.PAPER:
            # Paper trading simulation
            return await self._execute_paper_trade(
                execution_id, signal, quantity, current_price, risk_result, start_time
            )

        # Step 4: Live/Staging execution
        return await self._execute_live_trade(
            execution_id, signal, user_id, cred_id, quantity, risk_result, start_time
        )

    async def _execute_paper_trade(
        self,
        execution_id: UUID,
        signal: InferenceSignalCreate,
        quantity: float,
        price: Optional[float],
        risk_result: RiskCheckResult,
        start_time: float,
    ) -> ExecutionResult:
        """Execute a paper trade (simulation)."""
        await self._audit_event(
            execution_id,
            BridgeEventType.EXECUTION_STARTED,
            {"mode": "paper"},
        )

        # Simulate execution
        fill_price = price or 100.0
        commission = quantity * fill_price * 0.001  # 0.1% commission

        self._executions_total += 1
        self._executions_success += 1
        self._last_execution_at = datetime.utcnow()

        await self._audit_event(
            execution_id,
            BridgeEventType.EXECUTION_FILLED,
            {"fill_price": fill_price, "commission": commission},
        )

        result = ExecutionResult(
            success=True,
            execution_id=execution_id,
            signal_id=signal.signal_id,
            status=ExecutionStatus.FILLED,
            symbol=signal.symbol,
            side=self._action_to_side(signal.action),
            requested_quantity=quantity,
            filled_quantity=quantity,
            filled_price=fill_price,
            commission=commission,
            risk_check=risk_result,
            executed_at=datetime.utcnow(),
            processing_time_ms=(perf_counter() - start_time) * 1000,
            metadata={"mode": "paper", "simulated": True},
        )

        # Notify listeners
        await self._notify_execution(result)

        return result

    async def _execute_live_trade(
        self,
        execution_id: UUID,
        signal: InferenceSignalCreate,
        user_id: UUID,
        credential_id: UUID,
        quantity: float,
        risk_result: RiskCheckResult,
        start_time: float,
    ) -> ExecutionResult:
        """Execute a live trade on the exchange."""
        await self._audit_event(
            execution_id,
            BridgeEventType.EXECUTION_STARTED,
            {"mode": "live", "credential_id": str(credential_id)},
        )

        try:
            # Get exchange credentials from Vault
            credentials = await self._get_credentials(credential_id)
            if not credentials:
                return self._error_result(
                    execution_id, signal, "Failed to retrieve exchange credentials"
                )

            # Create exchange adapter
            adapter = ExchangeAdapterFactory.create(
                exchange=credentials["exchange"],
                api_key=credentials["api_key"],
                api_secret=credentials["api_secret"],
                passphrase=credentials.get("passphrase"),
                testnet=credentials.get("testnet", False),
            )

            # Place order
            side = self._action_to_side(signal.action)
            symbol = adapter.normalize_symbol(signal.symbol)

            await self._audit_event(
                execution_id,
                BridgeEventType.EXECUTION_SUBMITTED,
                {"symbol": symbol, "side": side.value, "quantity": quantity},
            )

            order_result: OrderResult = await adapter.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
            )

            await adapter.close()

            if not order_result.success:
                self._executions_total += 1
                self._executions_failed += 1

                await self._audit_event(
                    execution_id,
                    BridgeEventType.EXECUTION_FAILED,
                    {"error": order_result.error_message},
                )

                return ExecutionResult(
                    success=False,
                    execution_id=execution_id,
                    signal_id=signal.signal_id,
                    status=order_result.status,
                    exchange=credentials["exchange"],
                    symbol=signal.symbol,
                    side=side,
                    requested_quantity=quantity,
                    error_message=order_result.error_message,
                    risk_check=risk_result,
                    processing_time_ms=(perf_counter() - start_time) * 1000,
                )

            # Success
            self._executions_total += 1
            self._executions_success += 1
            self._last_execution_at = datetime.utcnow()

            await self._audit_event(
                execution_id,
                BridgeEventType.EXECUTION_FILLED,
                {
                    "order_id": order_result.order_id,
                    "fill_price": order_result.average_price,
                    "filled_quantity": order_result.filled_quantity,
                    "commission": order_result.commission,
                },
            )

            result = ExecutionResult(
                success=True,
                execution_id=execution_id,
                signal_id=signal.signal_id,
                status=order_result.status,
                exchange=credentials["exchange"],
                symbol=signal.symbol,
                side=side,
                requested_quantity=quantity,
                filled_quantity=order_result.filled_quantity,
                filled_price=order_result.average_price,
                commission=order_result.commission,
                risk_check=risk_result,
                executed_at=order_result.executed_at,
                processing_time_ms=(perf_counter() - start_time) * 1000,
                metadata={"order_id": order_result.order_id},
            )

            # Record trade result for risk tracking
            pnl_pct = 0.0  # Calculate based on position entry/exit
            await self.risk_bridge.record_trade_result(pnl_pct, signal, user_id)

            # Notify listeners
            await self._notify_execution(result)

            return result

        except Exception as e:
            logger.error(f"Live execution error: {e}", exc_info=True)
            self._executions_total += 1
            self._executions_failed += 1

            await self._audit_event(
                execution_id,
                BridgeEventType.EXECUTION_FAILED,
                {"error": str(e)},
            )

            return self._error_result(execution_id, signal, str(e))

    async def get_user_state(self, user_id: UUID) -> Dict[str, Any]:
        """
        Get current trading state for a user.

        Args:
            user_id: User identifier

        Returns:
            Dict with user trading state
        """
        risk_status = await self.risk_bridge.get_user_risk_status(user_id)

        return {
            "user_id": str(user_id),
            "bridge_active": self.is_active,
            "trading_mode": self.trading_mode.value,
            "risk_status": risk_status,
        }

    async def get_status(self) -> BridgeStatus:
        """Get current bridge status."""
        uptime = (datetime.utcnow() - self._start_time).total_seconds()

        # Get connected exchanges
        exchanges = list(ExchangeAdapterFactory.get_supported_exchanges())

        # Check inference WS connection
        ws_connected = False
        try:
            from app.services.websocket_bridge import WebSocketBridgeManager
            bridge = WebSocketBridgeManager.get_instance()
            ws_connected = bridge.is_connected if bridge else False
        except:
            pass

        return BridgeStatus(
            is_active=self.is_active,
            kill_switch_active=self.trading_mode == TradingMode.KILLED,
            kill_switch_reason=self._get_kill_switch_reason(),
            trading_mode=self.trading_mode,
            connected_exchanges=exchanges,
            pending_executions=0,  # TODO: Track pending
            last_signal_at=self._last_signal_at,
            last_execution_at=self._last_execution_at,
            inference_ws_connected=ws_connected,
            uptime_seconds=uptime,
            stats={
                "signals_received": self._signals_received,
                "executions_total": self._executions_total,
                "executions_success": self._executions_success,
                "executions_failed": self._executions_failed,
                "blocked_by_risk": self._blocked_by_risk,
            },
        )

    async def activate_kill_switch(self, reason: str, activated_by: str = "system") -> bool:
        """
        Activate the kill switch.

        Args:
            reason: Reason for activation
            activated_by: Who activated it

        Returns:
            True if activated successfully
        """
        logger.critical(f"KILL SWITCH ACTIVATED by {activated_by}: {reason}")

        if self._trading_flags:
            self._trading_flags.set_kill_switch(True, reason=reason, updated_by=activated_by)

        # Force mode override
        self._trading_mode_override = TradingMode.KILLED

        # Audit
        await self._audit_event(
            uuid4(),
            BridgeEventType.KILL_SWITCH_ACTIVATED,
            {"reason": reason, "activated_by": activated_by},
        )

        # Notify listeners
        for listener in self._listeners:
            try:
                await listener({
                    "type": "kill_switch",
                    "active": True,
                    "reason": reason,
                    "activated_by": activated_by,
                })
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")

        return True

    async def deactivate_kill_switch(self, confirm: bool, deactivated_by: str = "system") -> bool:
        """
        Deactivate the kill switch.

        Args:
            confirm: Must be True to confirm deactivation
            deactivated_by: Who deactivated it

        Returns:
            True if deactivated successfully
        """
        if not confirm:
            logger.warning("Kill switch deactivation requires confirm=True")
            return False

        logger.warning(f"Kill switch DEACTIVATED by {deactivated_by}")

        if self._trading_flags:
            self._trading_flags.set_kill_switch(False, reason="Manual deactivation", updated_by=deactivated_by)

        # Clear mode override
        self._trading_mode_override = None

        # Audit
        await self._audit_event(
            uuid4(),
            BridgeEventType.KILL_SWITCH_DEACTIVATED,
            {"deactivated_by": deactivated_by},
        )

        return True

    def add_listener(self, listener: callable) -> None:
        """Add an event listener for execution notifications."""
        self._listeners.append(listener)

    def remove_listener(self, listener: callable) -> None:
        """Remove an event listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _action_to_side(self, action: int) -> OrderSide:
        """Convert action int to OrderSide."""
        # 0=SELL, 1=HOLD, 2=BUY
        if action == 2:
            return OrderSide.BUY
        return OrderSide.SELL

    async def _calculate_position_size(
        self,
        signal: InferenceSignalCreate,
        user_id: UUID,
        current_price: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on user settings and portfolio.

        Uses fractional position sizing:
        - Gets user's risk limits from database
        - Calculates based on max_position_size_usd
        - Scales by confidence if applicable

        Args:
            signal: Inference signal
            user_id: User identifier
            current_price: Current market price

        Returns:
            Position size in quote currency (USD)
        """
        # Get user limits
        user_limits = await self.risk_bridge.get_user_limits(user_id)

        # Base position size from settings or user limits
        if user_limits:
            base_size = float(user_limits.max_position_size_usd)
        else:
            base_size = settings.default_position_size_usd

        # Scale by confidence (optional strategy)
        # Higher confidence = larger position (within limits)
        confidence_scale = 0.5 + (signal.confidence * 0.5)  # 0.5 to 1.0 multiplier
        position_size = base_size * confidence_scale

        # Ensure within global limits
        max_allowed = settings.max_position_size_usd
        position_size = min(position_size, max_allowed)

        logger.debug(
            f"Position size calculated: base={base_size}, "
            f"confidence={signal.confidence:.2f}, final={position_size:.2f}"
        )

        return position_size

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get current market price for a symbol.

        Tries multiple sources:
        1. Exchange adapter (if credentials available)
        2. Inference API price endpoint
        3. Cached price from Redis

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price or None if unavailable
        """
        import httpx

        # Try inference API first (has cached prices)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{settings.inference_api_url}/api/market/price",
                    params={"symbol": symbol}
                )
                if response.status_code == 200:
                    data = response.json()
                    price = data.get("price")
                    if price:
                        logger.debug(f"Price for {symbol} from inference API: {price}")
                        return float(price)
        except Exception as e:
            logger.warning(f"Could not fetch price from inference API: {e}")

        # Try Redis cache
        try:
            import redis
            import json
            r = redis.Redis(
                host=os.environ.get("REDIS_HOST", "redis"),
                port=int(os.environ.get("REDIS_PORT", "6379")),
                decode_responses=True,
            )
            cached = r.get(f"price:{symbol}")
            if cached:
                data = json.loads(cached)
                price = data.get("price")
                if price:
                    logger.debug(f"Price for {symbol} from Redis cache: {price}")
                    return float(price)
        except Exception as e:
            logger.warning(f"Could not fetch price from Redis: {e}")

        # Fallback: try to get from any connected exchange adapter
        # This would require having default credentials set up
        logger.warning(f"No price available for {symbol}")
        return None

    async def _get_credentials(self, credential_id: UUID) -> Optional[Dict[str, Any]]:
        """Get exchange credentials from Vault."""
        try:
            return await self.vault_service.get_exchange_credentials(
                credential_id,
                db_session=self.db_session,
            )
        except Exception as e:
            logger.error(f"Error fetching credentials: {e}")
            return None

    def _get_kill_switch_reason(self) -> Optional[str]:
        """Get kill switch reason if active."""
        if self._trading_flags and self._trading_flags.kill_switch:
            state = self._trading_flags._get_flag_state("kill_switch")
            return state.get("reason") if state else None
        return None

    def _blocked_result(
        self,
        execution_id: UUID,
        signal: InferenceSignalCreate,
        reason: RiskReason,
        message: str,
        risk_result: Optional[RiskCheckResult] = None,
    ) -> ExecutionResult:
        """Create a blocked execution result."""
        return ExecutionResult(
            success=False,
            execution_id=execution_id,
            signal_id=signal.signal_id,
            status=ExecutionStatus.REJECTED,
            symbol=signal.symbol,
            risk_check=risk_result or RiskCheckResult(
                decision=RiskDecision.BLOCK,
                reason=reason,
                message=message,
            ),
            error_message=message,
        )

    def _error_result(
        self,
        execution_id: UUID,
        signal: InferenceSignalCreate,
        error: str,
    ) -> ExecutionResult:
        """Create an error execution result."""
        self._executions_total += 1
        self._executions_failed += 1

        return ExecutionResult(
            success=False,
            execution_id=execution_id,
            signal_id=signal.signal_id,
            status=ExecutionStatus.FAILED,
            symbol=signal.symbol,
            error_message=error,
        )

    async def _audit_event(
        self,
        execution_id: UUID,
        event_type: BridgeEventType,
        event_data: Dict[str, Any],
    ) -> None:
        """Record an audit event."""
        try:
            # TODO: Persist to execution_audit table
            logger.debug(f"Audit: {event_type.value} for {execution_id}: {event_data}")
        except Exception as e:
            logger.error(f"Error recording audit event: {e}")

    async def _notify_execution(self, result: ExecutionResult) -> None:
        """Notify listeners of execution result."""
        for listener in self._listeners:
            try:
                await listener({
                    "type": "execution_update",
                    "execution_id": str(result.execution_id),
                    "status": result.status.value,
                    "filled_quantity": result.filled_quantity,
                    "filled_price": result.filled_price,
                })
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")


# Missing UserTradingState class - add to contracts if needed
class UserTradingState:
    """User trading state placeholder."""
    pass
