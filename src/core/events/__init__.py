"""
Events Module - Event Bus Pattern for System Communication
==========================================================

Provides decoupled communication between system components through
an event-driven architecture using the Publisher-Subscriber pattern.

Components:
- EventBus: Central hub for event publishing and subscription
- TradeEvent: Events for trade lifecycle
- RiskEvent: Events for risk management
- SystemEvent: General system events

Usage:
    from src.core.events import EventBus, TradeEvent, RiskEvent

    bus = EventBus()

    # Subscribe to events
    bus.subscribe("trade.closed", handle_closed)
    bus.subscribe("trade.*", handle_any_trade)

    # Publish events
    bus.publish(TradeEvent(event_type="closed", trade_id="T001", ...))

Author: USD/COP Trading System
Version: 1.0.0
"""

from .event_bus import (
    # Core classes
    EventBus,
    BaseEvent,
    TradeEvent,
    RiskEvent,
    SystemEvent,

    # Types
    Event,
    EventHandler,
    AsyncEventHandler,
    Subscription,
    EventPriority,

    # Protocol
    IEventBus,

    # Utilities
    get_event_bus,
    subscribe_to,
)

__all__ = [
    # Core classes
    "EventBus",
    "BaseEvent",
    "TradeEvent",
    "RiskEvent",
    "SystemEvent",

    # Types
    "Event",
    "EventHandler",
    "AsyncEventHandler",
    "Subscription",
    "EventPriority",

    # Protocol
    "IEventBus",

    # Utilities
    "get_event_bus",
    "subscribe_to",
]
