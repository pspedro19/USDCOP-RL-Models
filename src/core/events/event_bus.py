"""
Event Bus Pattern - Publisher-Subscriber for System Events
===========================================================

This module implements an Event Bus for decoupled communication between
system components. It supports both synchronous and asynchronous event
handling, enabling loose coupling between publishers and subscribers.

Event Types:
- TradeEvent: Trade lifecycle events (opened, closed, stop_loss, take_profit)
- RiskEvent: Risk management events (circuit_breaker, cooldown, max_drawdown)
- SystemEvent: General system events (startup, shutdown, error)

Usage:
    from src.core.events import EventBus, TradeEvent, RiskEvent

    # Create event bus instance
    bus = EventBus()

    # Subscribe to events
    def on_trade_closed(event: TradeEvent):
        print(f"Trade {event.trade_id} closed with PnL: {event.pnl}")

    bus.subscribe("trade.closed", on_trade_closed)

    # Publish events
    bus.publish(TradeEvent(
        event_type="closed",
        trade_id="T001",
        signal="LONG",
        pnl=150.50,
        timestamp=datetime.now()
    ))

Author: USD/COP Trading System
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Set,
    Union,
    runtime_checkable
)
from collections import defaultdict
import asyncio
import logging
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priority levels for event handlers."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class BaseEvent:
    """Base class for all events."""
    timestamp: datetime = field(default_factory=datetime.now)

    def get_event_key(self) -> str:
        """Get the routing key for this event."""
        return f"{self.__class__.__name__.lower()}"


@dataclass
class TradeEvent(BaseEvent):
    """
    Event for trade lifecycle changes.

    Attributes:
        event_type: Type of event (opened, closed, stop_loss, take_profit)
        trade_id: Unique identifier for the trade
        signal: Trading signal that initiated the trade
        pnl: Profit/Loss value (0 for opened events)
        timestamp: When the event occurred
        metadata: Additional context data
    """
    event_type: str = ""  # "opened", "closed", "stop_loss", "take_profit"
    trade_id: str = ""
    signal: str = ""
    pnl: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_event_key(self) -> str:
        """Get routing key: trade.{event_type}"""
        return f"trade.{self.event_type}"


@dataclass
class RiskEvent(BaseEvent):
    """
    Event for risk management triggers.

    Attributes:
        event_type: Type of risk event (circuit_breaker, cooldown, max_drawdown)
        reason: Human-readable explanation
        timestamp: When the event occurred
        details: Additional context data
    """
    event_type: str = ""  # "circuit_breaker", "cooldown", "max_drawdown"
    reason: str = ""
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def get_event_key(self) -> str:
        """Get routing key: risk.{event_type}"""
        return f"risk.{self.event_type}"


@dataclass
class SystemEvent(BaseEvent):
    """
    Event for system-level notifications.

    Attributes:
        event_type: Type of system event (startup, shutdown, error, warning)
        component: Component that generated the event
        message: Event message
        details: Additional context data
    """
    event_type: str = ""  # "startup", "shutdown", "error", "warning"
    component: str = ""
    message: str = ""
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

    def get_event_key(self) -> str:
        """Get routing key: system.{event_type}"""
        return f"system.{self.event_type}"


# Type alias for events
Event = Union[TradeEvent, RiskEvent, SystemEvent, BaseEvent]

# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Can be coroutine


@dataclass
class Subscription:
    """Represents a subscription to events."""
    handler: EventHandler
    priority: EventPriority = EventPriority.NORMAL
    is_async: bool = False
    weak_ref: bool = False  # Use weak reference for handler

    def __hash__(self):
        return id(self.handler)


@runtime_checkable
class IEventBus(Protocol):
    """Interface for Event Bus implementations."""

    def subscribe(
        self,
        event_type: str,
        listener: EventHandler,
        priority: EventPriority = EventPriority.NORMAL
    ) -> None:
        """Subscribe to events of a specific type."""
        ...

    def unsubscribe(
        self,
        event_type: str,
        listener: EventHandler
    ) -> None:
        """Unsubscribe from events."""
        ...

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        ...


class EventBus(IEventBus):
    """
    Publisher-Subscriber Event Bus for system events.

    Provides decoupled communication between components through
    event publishing and subscription. Supports:
    - Wildcard subscriptions (e.g., "trade.*" matches all trade events)
    - Priority-based handler execution
    - Synchronous and asynchronous handlers
    - Thread-safe operations
    - Event history for debugging

    Example:
        bus = EventBus()

        # Subscribe with specific event type
        bus.subscribe("trade.closed", handle_trade_closed)

        # Subscribe to all trade events
        bus.subscribe("trade.*", handle_any_trade)

        # Publish event
        bus.publish(TradeEvent(event_type="closed", ...))
    """

    _instance: Optional["EventBus"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern - ensure single event bus instance."""
        if kwargs.get("singleton", True):
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
                return cls._instance
        return super().__new__(cls)

    def __init__(
        self,
        max_history: int = 1000,
        enable_async: bool = True,
        singleton: bool = True
    ):
        """
        Initialize the Event Bus.

        Args:
            max_history: Maximum number of events to keep in history
            enable_async: Enable async handler support
            singleton: Use singleton pattern (default True)
        """
        if getattr(self, "_initialized", False):
            return

        self._subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self._wildcard_subscribers: Dict[str, List[Subscription]] = defaultdict(list)
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._enable_async = enable_async
        self._lock = threading.RLock()

        # Thread pool for async handlers
        if enable_async:
            self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="event_bus_")
        else:
            self._executor = None

        # Async event loop reference
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None

        self._initialized = True
        logger.info(
            f"EventBus initialized (max_history={max_history}, "
            f"async={enable_async})"
        )

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (useful for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.clear()
                cls._instance = None

    def subscribe(
        self,
        event_type: str,
        listener: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        is_async: bool = False
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Event type to subscribe to (e.g., "trade.closed", "risk.*")
            listener: Callback function to invoke on event
            priority: Handler priority (higher priority runs first)
            is_async: Whether the handler is asynchronous
        """
        subscription = Subscription(
            handler=listener,
            priority=priority,
            is_async=is_async
        )

        with self._lock:
            # Check for wildcard subscription
            if "*" in event_type:
                self._wildcard_subscribers[event_type].append(subscription)
                # Sort by priority (descending)
                self._wildcard_subscribers[event_type].sort(
                    key=lambda s: s.priority.value, reverse=True
                )
                logger.debug(f"Subscribed to wildcard '{event_type}'")
            else:
                self._subscribers[event_type].append(subscription)
                # Sort by priority (descending)
                self._subscribers[event_type].sort(
                    key=lambda s: s.priority.value, reverse=True
                )
                logger.debug(f"Subscribed to '{event_type}'")

    def unsubscribe(
        self,
        event_type: str,
        listener: EventHandler
    ) -> None:
        """
        Unsubscribe a listener from events.

        Args:
            event_type: Event type to unsubscribe from
            listener: The handler function to remove
        """
        with self._lock:
            if "*" in event_type:
                subscribers = self._wildcard_subscribers.get(event_type, [])
            else:
                subscribers = self._subscribers.get(event_type, [])

            # Remove matching subscriptions
            for sub in subscribers[:]:  # Copy to allow modification
                if sub.handler == listener:
                    subscribers.remove(sub)
                    logger.debug(f"Unsubscribed from '{event_type}'")
                    break

    def publish(self, event: Event) -> None:
        """
        Publish an event to all subscribers.

        Dispatches the event to:
        1. Exact match subscribers
        2. Wildcard subscribers that match

        Args:
            event: Event to publish
        """
        event_key = event.get_event_key()

        # Record in history
        self._record_event(event)

        # Get matching subscribers
        handlers = self._get_matching_handlers(event_key)

        if not handlers:
            logger.debug(f"No subscribers for event '{event_key}'")
            return

        logger.debug(
            f"Publishing '{event_key}' to {len(handlers)} handlers"
        )

        # Dispatch to handlers
        for subscription in handlers:
            try:
                if subscription.is_async and self._enable_async:
                    self._dispatch_async(subscription.handler, event)
                else:
                    subscription.handler(event)
            except Exception as e:
                logger.error(
                    f"Error in event handler for '{event_key}': {e}",
                    exc_info=True
                )

    async def publish_async(self, event: Event) -> None:
        """
        Publish an event asynchronously.

        Awaits all async handlers and runs sync handlers in executor.

        Args:
            event: Event to publish
        """
        event_key = event.get_event_key()

        # Record in history
        self._record_event(event)

        # Get matching subscribers
        handlers = self._get_matching_handlers(event_key)

        if not handlers:
            logger.debug(f"No subscribers for event '{event_key}'")
            return

        logger.debug(
            f"Publishing '{event_key}' async to {len(handlers)} handlers"
        )

        # Create tasks for all handlers
        tasks = []
        for subscription in handlers:
            try:
                if subscription.is_async:
                    # Async handler - await directly
                    result = subscription.handler(event)
                    if asyncio.iscoroutine(result):
                        tasks.append(asyncio.create_task(result))
                else:
                    # Sync handler - run in executor
                    loop = asyncio.get_event_loop()
                    tasks.append(
                        loop.run_in_executor(
                            self._executor,
                            subscription.handler,
                            event
                        )
                    )
            except Exception as e:
                logger.error(
                    f"Error creating task for '{event_key}': {e}",
                    exc_info=True
                )

        # Wait for all handlers to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"Handler {i} failed for '{event_key}': {result}"
                    )

    def _get_matching_handlers(
        self,
        event_key: str
    ) -> List[Subscription]:
        """
        Get all handlers matching an event key.

        Matches exact subscriptions and wildcard patterns.

        Args:
            event_key: The event routing key

        Returns:
            List of matching subscriptions sorted by priority
        """
        handlers: List[Subscription] = []

        with self._lock:
            # Exact match subscribers
            handlers.extend(self._subscribers.get(event_key, []))

            # Wildcard match subscribers
            for pattern, subs in self._wildcard_subscribers.items():
                if self._matches_pattern(event_key, pattern):
                    handlers.extend(subs)

        # Sort by priority (descending)
        handlers.sort(key=lambda s: s.priority.value, reverse=True)
        return handlers

    def _matches_pattern(self, event_key: str, pattern: str) -> bool:
        """
        Check if event key matches a wildcard pattern.

        Supports:
        - "trade.*" matches "trade.opened", "trade.closed", etc.
        - "*" matches everything

        Args:
            event_key: The event routing key
            pattern: The pattern to match against

        Returns:
            True if pattern matches
        """
        if pattern == "*":
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_key.startswith(prefix + ".")

        return event_key == pattern

    def _dispatch_async(
        self,
        handler: EventHandler,
        event: Event
    ) -> None:
        """
        Dispatch handler asynchronously.

        Args:
            handler: Handler function
            event: Event to pass to handler
        """
        if self._executor:
            self._executor.submit(self._safe_call, handler, event)
        else:
            # Fallback to sync
            self._safe_call(handler, event)

    def _safe_call(
        self,
        handler: EventHandler,
        event: Event
    ) -> None:
        """
        Safely call a handler with error handling.

        Args:
            handler: Handler function
            event: Event to pass
        """
        try:
            handler(event)
        except Exception as e:
            logger.error(
                f"Error in async handler: {e}",
                exc_info=True
            )

    def _record_event(self, event: Event) -> None:
        """
        Record event in history.

        Args:
            event: Event to record
        """
        with self._lock:
            self._event_history.append(event)

            # Trim history if needed
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]

    def get_history(
        self,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Event]:
        """
        Get event history, optionally filtered by type.

        Args:
            event_type: Optional filter (e.g., "trade", "risk")
            limit: Maximum events to return

        Returns:
            List of events (most recent first)
        """
        with self._lock:
            history = self._event_history[:]

        if event_type:
            history = [
                e for e in history
                if e.get_event_key().startswith(event_type)
            ]

        return history[-limit:][::-1]  # Most recent first

    def get_subscriber_count(self, event_type: Optional[str] = None) -> int:
        """
        Get count of subscribers.

        Args:
            event_type: Optional specific event type to count

        Returns:
            Number of subscribers
        """
        with self._lock:
            if event_type:
                exact = len(self._subscribers.get(event_type, []))
                wildcard = sum(
                    len(subs) for pattern, subs in self._wildcard_subscribers.items()
                    if self._matches_pattern(event_type, pattern)
                )
                return exact + wildcard
            else:
                exact = sum(len(subs) for subs in self._subscribers.values())
                wildcard = sum(len(subs) for subs in self._wildcard_subscribers.values())
                return exact + wildcard

    def clear(self) -> None:
        """Clear all subscribers and history."""
        with self._lock:
            self._subscribers.clear()
            self._wildcard_subscribers.clear()
            self._event_history.clear()
        logger.info("EventBus cleared")

    def shutdown(self) -> None:
        """Shutdown the event bus and clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None
        self.clear()
        logger.info("EventBus shutdown complete")

    def __repr__(self) -> str:
        return (
            f"EventBus(subscribers={self.get_subscriber_count()}, "
            f"history={len(self._event_history)})"
        )


# Convenience function for getting singleton instance
def get_event_bus() -> EventBus:
    """
    Get the singleton EventBus instance.

    Returns:
        The global EventBus instance
    """
    return EventBus()


# Decorator for subscribing methods to events
def subscribe_to(
    event_type: str,
    priority: EventPriority = EventPriority.NORMAL,
    bus: Optional[EventBus] = None
):
    """
    Decorator for subscribing a function to events.

    Usage:
        @subscribe_to("trade.closed")
        def handle_trade_closed(event: TradeEvent):
            print(f"Trade closed: {event.trade_id}")

    Args:
        event_type: Event type to subscribe to
        priority: Handler priority
        bus: Optional EventBus instance (uses singleton if None)

    Returns:
        Decorator function
    """
    def decorator(func: EventHandler) -> EventHandler:
        target_bus = bus or get_event_bus()
        target_bus.subscribe(event_type, func, priority)
        return func
    return decorator
