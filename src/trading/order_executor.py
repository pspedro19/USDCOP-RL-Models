"""
OrderExecutor - Order Execution Component
==========================================

Single Responsibility: Execute trading orders and manage order lifecycle.
Split from the PaperTrader God Class to follow SOLID principles.

This component is responsible for:
- Processing trading signals into orders
- Managing order state (pending, filled, cancelled)
- Calculating position sizes
- Coordinating with PositionTracker and RiskEnforcer

Design Patterns:
- Single Responsibility Principle: Only handles order execution
- Strategy Pattern: Different execution strategies (paper, live)
- Dependency Injection: Receives dependencies via constructor

Author: Trading Team
Version: 1.0.0
Date: 2025-01-16
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
)
import logging
import uuid

from .position_tracker import (
    Position,
    PositionDirection,
    PositionTracker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class OrderType(str, Enum):
    """Type of order."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(str, Enum):
    """Side of order."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Status of an order."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class SignalType(str, Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"
    HOLD = "HOLD"


@dataclass
class Order:
    """
    Represents a trading order.

    Attributes:
        order_id: Unique identifier
        model_id: Model that generated the order
        side: BUY or SELL
        order_type: MARKET, LIMIT, etc.
        size: Order size in units
        price: Limit/stop price (None for market orders)
        status: Current order status
        created_at: Order creation timestamp
        filled_at: Order fill timestamp
        filled_price: Actual fill price
        filled_size: Actual filled size
        commission: Commission paid
        slippage: Price slippage
        metadata: Additional order data
    """
    order_id: str
    model_id: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_price: Optional[float] = None
    filled_size: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            "order_id": self.order_id,
            "model_id": self.model_id,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "size": self.size,
            "price": self.price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "filled_at": self.filled_at.isoformat() if self.filled_at else None,
            "filled_price": self.filled_price,
            "filled_size": self.filled_size,
            "commission": self.commission,
            "slippage": self.slippage,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    """
    Result of order execution.

    Attributes:
        success: Whether execution was successful
        order: The executed order
        position: Resulting position (if any)
        message: Execution message
        pnl: P&L from the execution (if closing position)
    """
    success: bool
    order: Optional[Order] = None
    position: Optional[Position] = None
    message: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "order": self.order.to_dict() if self.order else None,
            "position": self.position.to_dict() if self.position else None,
            "message": self.message,
            "pnl": self.pnl,
            "pnl_pct": self.pnl_pct,
        }


@dataclass
class ExecutorConfig:
    """
    Configuration for order executor.

    Attributes:
        initial_capital: Starting capital
        position_size_pct: Position size as percentage of capital
        enable_short: Whether short positions are allowed
        commission_pct: Commission percentage per trade
        max_slippage_pct: Maximum allowed slippage
        min_order_size: Minimum order size
    """
    initial_capital: float = 10000.0
    position_size_pct: float = 0.1
    enable_short: bool = True
    commission_pct: float = 0.0001
    max_slippage_pct: float = 0.001
    min_order_size: float = 0.01


# =============================================================================
# Risk Validator Protocol
# =============================================================================

class IRiskValidator(Protocol):
    """Protocol for risk validation."""

    def validate_signal(
        self,
        signal: str,
        current_drawdown_pct: float
    ) -> Tuple[bool, str]:
        """Validate if a signal should be executed."""
        ...

    def record_trade_result(self, pnl_pct: float, signal: str) -> None:
        """Record the result of a trade."""
        ...


# =============================================================================
# OrderExecutor Implementation
# =============================================================================

class OrderExecutor:
    """
    Executes trading orders and manages order lifecycle.

    Single Responsibility: Handle order execution and coordination.

    This class coordinates between:
    - PositionTracker: For position state management
    - RiskValidator: For risk checks before execution
    - Broker API: For actual order submission (in live mode)

    Features:
    - Signal-to-order translation
    - Position sizing calculation
    - Commission and slippage handling
    - Order history tracking
    - P&L calculation

    Usage:
        # Initialize with dependencies
        position_tracker = PositionTracker()
        config = ExecutorConfig(initial_capital=50000)

        executor = OrderExecutor(
            position_tracker=position_tracker,
            config=config
        )

        # Execute a signal
        result = executor.execute_signal(
            model_id="ppo_v1",
            signal=SignalType.LONG,
            current_price=4250.50
        )

        if result.success:
            print(f"Order filled: {result.order}")
    """

    def __init__(
        self,
        position_tracker: PositionTracker,
        config: Optional[ExecutorConfig] = None,
        risk_validator: Optional[IRiskValidator] = None,
    ) -> None:
        """
        Initialize the order executor.

        Args:
            position_tracker: Position tracking component
            config: Executor configuration
            risk_validator: Optional risk validation component
        """
        self._position_tracker = position_tracker
        self._config = config or ExecutorConfig()
        self._risk_validator = risk_validator

        # Capital tracking
        self._current_capital = self._config.initial_capital

        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._order_history: List[Order] = []

        # Statistics
        self._total_commission: float = 0.0
        self._total_slippage: float = 0.0

        logger.info(
            f"OrderExecutor initialized: capital=${self._config.initial_capital:.2f}, "
            f"position_size={self._config.position_size_pct*100:.1f}%"
        )

    # =========================================================================
    # Signal Execution
    # =========================================================================

    def execute_signal(
        self,
        model_id: str,
        signal: SignalType,
        current_price: float,
        timestamp: Optional[datetime] = None,
        current_drawdown_pct: float = 0.0,
    ) -> ExecutionResult:
        """
        Execute a trading signal.

        This is the main entry point for signal execution. It:
        1. Validates the signal with risk manager (if available)
        2. Determines required action based on current position
        3. Creates and executes orders
        4. Updates position tracker

        Args:
            model_id: Model generating the signal
            signal: Trading signal (LONG, SHORT, CLOSE, HOLD)
            current_price: Current market price
            timestamp: Execution timestamp
            current_drawdown_pct: Current portfolio drawdown

        Returns:
            ExecutionResult with order and position details
        """
        timestamp = timestamp or datetime.now()

        # HOLD signal - no action
        if signal == SignalType.HOLD:
            return ExecutionResult(
                success=True,
                message="HOLD signal - no action taken"
            )

        # Risk validation (if validator available)
        if self._risk_validator:
            allowed, reason = self._risk_validator.validate_signal(
                signal.value, current_drawdown_pct
            )
            if not allowed:
                logger.warning(f"Signal blocked by risk manager: {reason}")
                return ExecutionResult(
                    success=False,
                    message=f"Blocked by risk manager: {reason}"
                )

        # Get current position state
        has_position = self._position_tracker.has_position(model_id)
        current_direction = self._position_tracker.get_position_direction(model_id)

        # CLOSE signal
        if signal == SignalType.CLOSE:
            if not has_position:
                return ExecutionResult(
                    success=True,
                    message="CLOSE signal but no position to close"
                )
            return self._close_position(model_id, current_price, timestamp)

        # LONG signal
        if signal == SignalType.LONG:
            if has_position:
                if current_direction == PositionDirection.LONG:
                    return ExecutionResult(
                        success=True,
                        message="Already LONG - holding position"
                    )
                # Close SHORT before opening LONG
                close_result = self._close_position(model_id, current_price, timestamp)
                if not close_result.success:
                    return close_result

            return self._open_position(
                model_id, PositionDirection.LONG, current_price, timestamp
            )

        # SHORT signal
        if signal == SignalType.SHORT:
            if not self._config.enable_short:
                return ExecutionResult(
                    success=False,
                    message="SHORT trading is disabled"
                )

            if has_position:
                if current_direction == PositionDirection.SHORT:
                    return ExecutionResult(
                        success=True,
                        message="Already SHORT - holding position"
                    )
                # Close LONG before opening SHORT
                close_result = self._close_position(model_id, current_price, timestamp)
                if not close_result.success:
                    return close_result

            return self._open_position(
                model_id, PositionDirection.SHORT, current_price, timestamp
            )

        return ExecutionResult(
            success=False,
            message=f"Unknown signal: {signal}"
        )

    # =========================================================================
    # Order Creation and Execution
    # =========================================================================

    def _open_position(
        self,
        model_id: str,
        direction: PositionDirection,
        price: float,
        timestamp: datetime,
    ) -> ExecutionResult:
        """
        Open a new position.

        Args:
            model_id: Model identifier
            direction: Position direction
            price: Entry price
            timestamp: Entry timestamp

        Returns:
            ExecutionResult
        """
        # Calculate position size
        size = self._calculate_position_size(price)

        if size < self._config.min_order_size:
            return ExecutionResult(
                success=False,
                message=f"Calculated size ({size:.4f}) below minimum ({self._config.min_order_size})"
            )

        # Create order
        side = OrderSide.BUY if direction == PositionDirection.LONG else OrderSide.SELL
        order = Order(
            order_id=self._generate_order_id(),
            model_id=model_id,
            side=side,
            order_type=OrderType.MARKET,
            size=size,
            created_at=timestamp,
        )

        # Simulate fill (paper trading)
        fill_price = self._apply_slippage(price, side)
        commission = self._calculate_commission(fill_price, size)

        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = fill_price
        order.filled_size = size
        order.commission = commission
        order.slippage = abs(fill_price - price)

        # Update statistics
        self._total_commission += commission
        self._total_slippage += order.slippage * size

        # Record order
        self._order_history.append(order)

        # Open position in tracker
        try:
            position = self._position_tracker.open_position(
                model_id=model_id,
                direction=direction,
                size=size,
                price=fill_price,
                timestamp=timestamp,
                metadata={"order_id": order.order_id},
            )
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return ExecutionResult(
                success=False,
                order=order,
                message=f"Failed to open position: {e}"
            )

        logger.info(
            f"Opened {direction.value}: model={model_id}, price={fill_price:.4f}, "
            f"size={size:.4f}, commission={commission:.4f}"
        )

        return ExecutionResult(
            success=True,
            order=order,
            position=position,
            message=f"Opened {direction.value} position"
        )

    def _close_position(
        self,
        model_id: str,
        price: float,
        timestamp: datetime,
    ) -> ExecutionResult:
        """
        Close an existing position.

        Args:
            model_id: Model identifier
            price: Exit price
            timestamp: Exit timestamp

        Returns:
            ExecutionResult with P&L
        """
        # Get current position
        current_position = self._position_tracker.get_position(model_id)
        if current_position is None:
            return ExecutionResult(
                success=False,
                message=f"No open position for model {model_id}"
            )

        # Create close order
        side = (
            OrderSide.SELL
            if current_position.direction == PositionDirection.LONG
            else OrderSide.BUY
        )
        order = Order(
            order_id=self._generate_order_id(),
            model_id=model_id,
            side=side,
            order_type=OrderType.MARKET,
            size=current_position.size,
            created_at=timestamp,
        )

        # Simulate fill
        fill_price = self._apply_slippage(price, side)
        commission = self._calculate_commission(fill_price, current_position.size)

        order.status = OrderStatus.FILLED
        order.filled_at = timestamp
        order.filled_price = fill_price
        order.filled_size = current_position.size
        order.commission = commission
        order.slippage = abs(fill_price - price)

        # Update statistics
        self._total_commission += commission
        self._total_slippage += order.slippage * current_position.size

        # Record order
        self._order_history.append(order)

        # Close position in tracker
        try:
            closed_position = self._position_tracker.close_position(
                model_id=model_id,
                price=fill_price,
                timestamp=timestamp,
            )
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return ExecutionResult(
                success=False,
                order=order,
                message=f"Failed to close position: {e}"
            )

        # Calculate net P&L (after commissions)
        gross_pnl = closed_position.pnl
        entry_commission = self._calculate_commission(
            current_position.entry_price, current_position.size
        )
        net_pnl = gross_pnl - commission - entry_commission

        # Update capital
        self._current_capital += net_pnl

        # Record with risk validator if available
        if self._risk_validator:
            position_value = current_position.entry_price * current_position.size
            pnl_pct = (net_pnl / position_value * 100) if position_value > 0 else 0.0
            self._risk_validator.record_trade_result(pnl_pct, current_position.direction.value)

        logger.info(
            f"Closed {current_position.direction.value}: model={model_id}, "
            f"entry={current_position.entry_price:.4f}, exit={fill_price:.4f}, "
            f"gross_pnl={gross_pnl:.2f}, net_pnl={net_pnl:.2f}"
        )

        return ExecutionResult(
            success=True,
            order=order,
            position=closed_position,
            message=f"Closed {current_position.direction.value} position",
            pnl=net_pnl,
            pnl_pct=closed_position.pnl_pct,
        )

    def close_all_positions(
        self,
        price: float,
        timestamp: Optional[datetime] = None,
    ) -> List[ExecutionResult]:
        """
        Close all open positions.

        Args:
            price: Exit price
            timestamp: Exit timestamp

        Returns:
            List of execution results
        """
        timestamp = timestamp or datetime.now()
        results = []

        positions = self._position_tracker.get_all_positions()
        for position in positions:
            result = self._close_position(position.model_id, price, timestamp)
            results.append(result)

        return results

    # =========================================================================
    # Calculation Helpers
    # =========================================================================

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        return str(uuid.uuid4())[:8].upper()

    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on current capital.

        Args:
            price: Current price

        Returns:
            Position size in units
        """
        capital_per_trade = self._current_capital * self._config.position_size_pct
        return capital_per_trade / price

    def _calculate_commission(self, price: float, size: float) -> float:
        """
        Calculate commission for a trade.

        Args:
            price: Execution price
            size: Trade size

        Returns:
            Commission amount
        """
        notional = price * size
        return notional * self._config.commission_pct

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """
        Apply slippage to price.

        Args:
            price: Base price
            side: Order side (BUY/SELL)

        Returns:
            Price with slippage applied
        """
        slippage = price * self._config.max_slippage_pct

        # Buyers pay more, sellers receive less
        if side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage

    # =========================================================================
    # Query Methods
    # =========================================================================

    @property
    def current_capital(self) -> float:
        """Get current capital."""
        return self._current_capital

    @property
    def initial_capital(self) -> float:
        """Get initial capital."""
        return self._config.initial_capital

    def get_total_pnl(self) -> float:
        """Get total realized P&L."""
        return self._position_tracker.get_total_pnl()

    def get_return_pct(self) -> float:
        """Get total return percentage."""
        return (
            (self._current_capital - self._config.initial_capital)
            / self._config.initial_capital * 100
        )

    def get_order_history(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[Order]:
        """
        Get order history.

        Args:
            model_id: Optional filter by model
            limit: Optional limit

        Returns:
            List of orders
        """
        history = self._order_history

        if model_id:
            history = [o for o in history if o.model_id == model_id]

        if limit:
            history = history[-limit:]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        position_stats = self._position_tracker.get_statistics()

        return {
            **position_stats,
            "current_capital": round(self._current_capital, 2),
            "initial_capital": self._config.initial_capital,
            "return_pct": round(self.get_return_pct(), 2),
            "total_commission": round(self._total_commission, 4),
            "total_slippage": round(self._total_slippage, 4),
            "total_orders": len(self._order_history),
            "pending_orders": len(self._pending_orders),
        }

    # =========================================================================
    # State Management
    # =========================================================================

    def reset(self) -> None:
        """Reset executor to initial state."""
        self._current_capital = self._config.initial_capital
        self._pending_orders.clear()
        self._order_history.clear()
        self._total_commission = 0.0
        self._total_slippage = 0.0
        self._position_tracker.reset()

        logger.info(
            f"OrderExecutor reset: capital=${self._config.initial_capital:.2f}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize executor state."""
        return {
            "current_capital": self._current_capital,
            "config": {
                "initial_capital": self._config.initial_capital,
                "position_size_pct": self._config.position_size_pct,
                "enable_short": self._config.enable_short,
                "commission_pct": self._config.commission_pct,
                "max_slippage_pct": self._config.max_slippage_pct,
            },
            "pending_orders": {k: v.to_dict() for k, v in self._pending_orders.items()},
            "order_history": [o.to_dict() for o in self._order_history],
            "total_commission": self._total_commission,
            "total_slippage": self._total_slippage,
            "positions": self._position_tracker.to_dict(),
            "statistics": self.get_statistics(),
        }

    def __repr__(self) -> str:
        return (
            f"OrderExecutor(capital=${self._current_capital:.2f}, "
            f"orders={len(self._order_history)})"
        )
