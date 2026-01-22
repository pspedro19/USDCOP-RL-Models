"""
Unit Tests for Event Bus Pattern
================================

Tests for the Event Bus implementation in src/core/events/event_bus.py

Author: USD/COP Trading System
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock, call
import time

from src.core.events.event_bus import (
    EventBus,
    TradeEvent,
    RiskEvent,
    SystemEvent,
    BaseEvent,
    EventPriority,
    get_event_bus,
    subscribe_to,
)


@pytest.fixture
def event_bus():
    """Create a fresh event bus for each test."""
    # Reset singleton for isolated testing
    EventBus.reset_instance()
    bus = EventBus(singleton=False)
    yield bus
    bus.shutdown()


@pytest.fixture
def mock_handler():
    """Create a mock event handler."""
    return Mock()


class TestTradeEvent:
    """Tests for TradeEvent dataclass."""

    def test_create_trade_event(self):
        """Test creating a trade event."""
        event = TradeEvent(
            event_type="opened",
            trade_id="T001",
            signal="LONG",
            pnl=0.0
        )

        assert event.event_type == "opened"
        assert event.trade_id == "T001"
        assert event.signal == "LONG"
        assert event.pnl == 0.0
        assert isinstance(event.timestamp, datetime)

    def test_event_key(self):
        """Test event routing key generation."""
        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)

        assert event.get_event_key() == "trade.closed"

    def test_metadata_default(self):
        """Test that metadata defaults to empty dict."""
        event = TradeEvent(event_type="opened", trade_id="T001", signal="LONG", pnl=0.0)

        assert event.metadata == {}


class TestRiskEvent:
    """Tests for RiskEvent dataclass."""

    def test_create_risk_event(self):
        """Test creating a risk event."""
        event = RiskEvent(
            event_type="circuit_breaker",
            reason="5 consecutive losses"
        )

        assert event.event_type == "circuit_breaker"
        assert event.reason == "5 consecutive losses"

    def test_event_key(self):
        """Test event routing key generation."""
        event = RiskEvent(event_type="max_drawdown", reason="Exceeded limit")

        assert event.get_event_key() == "risk.max_drawdown"


class TestSystemEvent:
    """Tests for SystemEvent dataclass."""

    def test_create_system_event(self):
        """Test creating a system event."""
        event = SystemEvent(
            event_type="startup",
            component="trading_engine",
            message="Engine started successfully"
        )

        assert event.event_type == "startup"
        assert event.component == "trading_engine"

    def test_event_key(self):
        """Test event routing key generation."""
        event = SystemEvent(event_type="error", component="api", message="Connection failed")

        assert event.get_event_key() == "system.error"


class TestEventBusSubscription:
    """Tests for event subscription."""

    def test_subscribe_exact_match(self, event_bus, mock_handler):
        """Test subscribing to exact event type."""
        event_bus.subscribe("trade.closed", mock_handler)

        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)
        event_bus.publish(event)

        mock_handler.assert_called_once_with(event)

    def test_subscribe_wildcard(self, event_bus, mock_handler):
        """Test subscribing to wildcard pattern."""
        event_bus.subscribe("trade.*", mock_handler)

        event1 = TradeEvent(event_type="opened", trade_id="T001", signal="LONG", pnl=0.0)
        event2 = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)

        event_bus.publish(event1)
        event_bus.publish(event2)

        assert mock_handler.call_count == 2

    def test_no_match(self, event_bus, mock_handler):
        """Test that non-matching events don't trigger handler."""
        event_bus.subscribe("trade.closed", mock_handler)

        event = TradeEvent(event_type="opened", trade_id="T001", signal="LONG", pnl=0.0)
        event_bus.publish(event)

        mock_handler.assert_not_called()

    def test_unsubscribe(self, event_bus, mock_handler):
        """Test unsubscribing from events."""
        event_bus.subscribe("trade.closed", mock_handler)
        event_bus.unsubscribe("trade.closed", mock_handler)

        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)
        event_bus.publish(event)

        mock_handler.assert_not_called()

    def test_multiple_subscribers(self, event_bus):
        """Test multiple subscribers receive events."""
        handler1 = Mock()
        handler2 = Mock()

        event_bus.subscribe("trade.closed", handler1)
        event_bus.subscribe("trade.closed", handler2)

        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)
        event_bus.publish(event)

        handler1.assert_called_once_with(event)
        handler2.assert_called_once_with(event)


class TestEventBusPriority:
    """Tests for priority-based handler execution."""

    def test_priority_order(self, event_bus):
        """Test handlers are called in priority order."""
        call_order = []

        def low_priority(event):
            call_order.append("low")

        def normal_priority(event):
            call_order.append("normal")

        def high_priority(event):
            call_order.append("high")

        event_bus.subscribe("test", low_priority, priority=EventPriority.LOW)
        event_bus.subscribe("test", normal_priority, priority=EventPriority.NORMAL)
        event_bus.subscribe("test", high_priority, priority=EventPriority.HIGH)

        event = BaseEvent()
        # Manually set event key
        event.get_event_key = lambda: "test"
        event_bus.publish(event)

        # High priority should be first
        assert call_order == ["high", "normal", "low"]


class TestEventBusPublish:
    """Tests for event publishing."""

    def test_publish_records_history(self, event_bus):
        """Test that publishing records events in history."""
        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)
        event_bus.publish(event)

        history = event_bus.get_history()

        assert len(history) == 1
        assert history[0].trade_id == "T001"

    def test_history_limit(self, event_bus):
        """Test that history respects limit."""
        for i in range(10):
            event = TradeEvent(event_type="closed", trade_id=f"T{i:03d}", signal="LONG", pnl=0.0)
            event_bus.publish(event)

        history = event_bus.get_history(limit=5)

        assert len(history) == 5

    def test_history_filter_by_type(self, event_bus):
        """Test filtering history by event type."""
        event_bus.publish(TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=0.0))
        event_bus.publish(RiskEvent(event_type="circuit_breaker", reason="test"))
        event_bus.publish(TradeEvent(event_type="opened", trade_id="T002", signal="SHORT", pnl=0.0))

        trade_history = event_bus.get_history(event_type="trade")
        risk_history = event_bus.get_history(event_type="risk")

        assert len(trade_history) == 2
        assert len(risk_history) == 1

    def test_handler_error_doesnt_stop_others(self, event_bus):
        """Test that a handler error doesn't stop other handlers."""
        handler1 = Mock()
        handler2 = Mock(side_effect=Exception("Handler error"))
        handler3 = Mock()

        event_bus.subscribe("test.event", handler1, priority=EventPriority.HIGH)
        event_bus.subscribe("test.event", handler2, priority=EventPriority.NORMAL)
        event_bus.subscribe("test.event", handler3, priority=EventPriority.LOW)

        event = BaseEvent()
        event.get_event_key = lambda: "test.event"
        event_bus.publish(event)

        # All handlers should be attempted
        handler1.assert_called_once()
        handler2.assert_called_once()
        handler3.assert_called_once()


class TestEventBusStatistics:
    """Tests for event bus statistics."""

    def test_subscriber_count(self, event_bus, mock_handler):
        """Test getting subscriber count."""
        assert event_bus.get_subscriber_count() == 0

        event_bus.subscribe("trade.closed", mock_handler)
        assert event_bus.get_subscriber_count() == 1

        event_bus.subscribe("trade.opened", mock_handler)
        assert event_bus.get_subscriber_count() == 2

    def test_subscriber_count_for_type(self, event_bus):
        """Test getting subscriber count for specific type."""
        handler1 = Mock()
        handler2 = Mock()

        event_bus.subscribe("trade.closed", handler1)
        event_bus.subscribe("trade.closed", handler2)
        event_bus.subscribe("risk.circuit_breaker", handler1)

        assert event_bus.get_subscriber_count("trade.closed") == 2
        assert event_bus.get_subscriber_count("risk.circuit_breaker") == 1


class TestEventBusClear:
    """Tests for clearing event bus."""

    def test_clear(self, event_bus, mock_handler):
        """Test clearing subscribers and history."""
        event_bus.subscribe("trade.closed", mock_handler)
        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=0.0)
        event_bus.publish(event)

        event_bus.clear()

        assert event_bus.get_subscriber_count() == 0
        assert len(event_bus.get_history()) == 0


class TestEventBusSingleton:
    """Tests for singleton behavior."""

    def test_singleton_instance(self):
        """Test that EventBus uses singleton pattern."""
        EventBus.reset_instance()

        bus1 = EventBus(singleton=True)
        bus2 = EventBus(singleton=True)

        assert bus1 is bus2

        EventBus.reset_instance()

    def test_get_event_bus_helper(self):
        """Test get_event_bus convenience function."""
        EventBus.reset_instance()

        bus1 = get_event_bus()
        bus2 = get_event_bus()

        assert bus1 is bus2

        EventBus.reset_instance()


class TestWildcardMatching:
    """Tests for wildcard pattern matching."""

    def test_matches_all_wildcard(self, event_bus, mock_handler):
        """Test * matches everything."""
        event_bus.subscribe("*", mock_handler)

        event_bus.publish(TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=0.0))
        event_bus.publish(RiskEvent(event_type="circuit_breaker", reason="test"))

        assert mock_handler.call_count == 2

    def test_prefix_wildcard(self, event_bus, mock_handler):
        """Test prefix.* matches prefix events."""
        event_bus.subscribe("trade.*", mock_handler)

        event_bus.publish(TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=0.0))
        event_bus.publish(TradeEvent(event_type="opened", trade_id="T002", signal="LONG", pnl=0.0))
        event_bus.publish(RiskEvent(event_type="circuit_breaker", reason="test"))

        assert mock_handler.call_count == 2  # Only trade events


class TestSubscribeDecorator:
    """Tests for subscribe_to decorator."""

    def test_decorator_subscribes(self):
        """Test that decorator subscribes function to events."""
        EventBus.reset_instance()
        bus = get_event_bus()

        call_count = [0]

        @subscribe_to("trade.closed")
        def handle_trade(event):
            call_count[0] += 1

        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=0.0)
        bus.publish(event)

        assert call_count[0] == 1

        EventBus.reset_instance()


class TestAsyncEventHandling:
    """Tests for async event handling."""

    @pytest.mark.asyncio
    async def test_async_publish(self, event_bus):
        """Test async event publishing."""
        results = []

        async def async_handler(event):
            await asyncio.sleep(0.01)
            results.append(event.trade_id)

        event_bus.subscribe("trade.closed", async_handler, is_async=True)

        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)
        await event_bus.publish_async(event)

        assert "T001" in results

    @pytest.mark.asyncio
    async def test_mixed_sync_async_handlers(self, event_bus):
        """Test mixing sync and async handlers."""
        results = []

        def sync_handler(event):
            results.append(f"sync:{event.trade_id}")

        async def async_handler(event):
            await asyncio.sleep(0.01)
            results.append(f"async:{event.trade_id}")

        event_bus.subscribe("trade.closed", sync_handler, is_async=False)
        event_bus.subscribe("trade.closed", async_handler, is_async=True)

        event = TradeEvent(event_type="closed", trade_id="T001", signal="LONG", pnl=100.0)
        await event_bus.publish_async(event)

        assert len(results) == 2
        assert any("sync:T001" in r for r in results)
        assert any("async:T001" in r for r in results)
