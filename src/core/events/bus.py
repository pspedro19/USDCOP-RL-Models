"""
Event Bus Implementation with Redis Backend
Central event management for USDCOP trading system
"""

import os
import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

# Environment configuration
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)
EVENT_CHANNEL = os.getenv("EVENT_CHANNEL", "trading.events")

class EventType(Enum):
    """Standard event types for USDCOP system"""
    # Data source events
    SOURCE_CHANGED = "SourceChanged"
    STREAM_STALE = "StreamStale"
    CIRCUIT_OPEN = "CircuitOpen"
    CIRCUIT_HALF_OPEN = "CircuitHalfOpen"
    CIRCUIT_CLOSED = "CircuitClosed"
    
    # Pipeline events
    PIPELINE_STARTED = "PipelineStarted"
    PIPELINE_STAGE_PROGRESS = "PipelineStageProgress"
    PIPELINE_COMPLETE = "PipelineComplete"
    PIPELINE_ERROR = "PipelineError"
    
    # Trading events
    TRADE_SIGNAL = "TradeSignal"
    TRADE_EXECUTED = "TradeExecuted"
    POSITION_OPENED = "PositionOpened"
    POSITION_CLOSED = "PositionClosed"
    
    # Database events
    DATA_SAVED = "DataSaved"
    DATA_RETRIEVED = "DataRetrieved"
    
    # System events
    SYSTEM_STARTUP = "SystemStartup"
    SYSTEM_SHUTDOWN = "SystemShutdown"
    HEALTH_CHECK = "HealthCheck"
    ERROR_OCCURRED = "ErrorOccurred"

@dataclass
class Event:
    """Standard event structure"""
    event: str  # EventType.value
    source: str
    ts: str
    correlation_id: Optional[str] = None
    payload: Dict[str, Any] = None
    metadata: Optional[Dict[str, Any]] = None

class EventBus(ABC):
    """Abstract event bus interface"""
    
    @abstractmethod
    def publish(self, event: Event) -> None:
        """Publish an event"""
        pass
    
    @abstractmethod
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe to specific event type"""
        pass
    
    @abstractmethod
    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        """Subscribe to all events"""
        pass

class InMemoryEventBus(EventBus):
    """In-memory event bus for testing/development"""
    
    def __init__(self):
        self.handlers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.global_handlers: List[Callable] = []
        self.logger = logging.getLogger(__name__)
    
    def publish(self, event: Event) -> None:
        """Publish event to all subscribers"""
        try:
            # Notify specific handlers
            event_type = EventType(event.event)
            for handler in self.handlers.get(event_type, []):
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Handler error for {event.event}: {e}")
            
            # Notify global handlers
            for handler in self.global_handlers:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Global handler error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Event publishing error: {e}")
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe to specific event type"""
        self.handlers[event_type].append(handler)
        self.logger.debug(f"Subscribed to {event_type.value}")
    
    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        """Subscribe to all events"""
        self.global_handlers.append(handler)
        self.logger.debug("Subscribed to all events")

class RedisEventBus(EventBus):
    """Redis-based event bus for production"""
    
    def __init__(self):
        try:
            import redis
            self.redis = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                password=REDIS_PASSWORD,
                decode_responses=True
            )
            self.pubsub = self.redis.pubsub(ignore_subscribe_messages=True)
            self.handlers: Dict[EventType, List[Callable]] = defaultdict(list)
            self.global_handlers: List[Callable] = []
            self.subscription_thread = None
            self.running = False
            self.logger = logging.getLogger(__name__)
            
        except ImportError:
            raise ImportError("Redis package not installed. Install with: pip install redis")
    
    def start(self):
        """Start Redis event bus"""
        if self.running:
            return
        
        self.running = True
        self.subscription_thread = threading.Thread(target=self._subscription_loop, daemon=True)
        self.subscription_thread.start()
        self.logger.info("Redis event bus started")
    
    def stop(self):
        """Stop Redis event bus"""
        if not self.running:
            return
        
        self.running = False
        if self.subscription_thread:
            self.subscription_thread.join(timeout=5)
        
        try:
            self.pubsub.close()
            self.redis.close()
        except Exception as e:
            self.logger.error(f"Error closing Redis connections: {e}")
        
        self.logger.info("Redis event bus stopped")
    
    def publish(self, event: Event) -> None:
        """Publish event to Redis channel"""
        try:
            event_data = json.dumps(asdict(event))
            self.redis.publish(EVENT_CHANNEL, event_data)
            self.logger.debug(f"Published event: {event.event}")
            
        except Exception as e:
            self.logger.error(f"Failed to publish event: {e}")
    
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None:
        """Subscribe to specific event type"""
        self.handlers[event_type].append(handler)
        self.logger.debug(f"Subscribed to {event_type.value}")
    
    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        """Subscribe to all events"""
        self.global_handlers.append(handler)
        self.logger.debug("Subscribed to all events")
    
    def _subscription_loop(self):
        """Background loop for Redis subscriptions"""
        try:
            self.pubsub.subscribe(EVENT_CHANNEL)
            
            for message in self.pubsub.listen():
                if not self.running:
                    break
                
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        event = Event(**event_data)
                        
                        # Notify specific handlers
                        event_type = EventType(event.event)
                        for handler in self.handlers.get(event_type, []):
                            try:
                                handler(event)
                            except Exception as e:
                                self.logger.error(f"Handler error for {event.event}: {e}")
                        
                        # Notify global handlers
                        for handler in self.global_handlers:
                            try:
                                handler(event)
                            except Exception as e:
                                self.logger.error(f"Global handler error: {e}")
                                
                    except Exception as e:
                        self.logger.error(f"Event processing error: {e}")
                        
        except Exception as e:
            self.logger.error(f"Redis subscription error: {e}")
        finally:
            self.pubsub.close()

def get_event_bus() -> EventBus:
    """Get event bus instance based on environment"""
    impl = os.getenv("EVENT_BUS_IMPL", "redis")
    
    if impl == "redis":
        try:
            bus = RedisEventBus()
            bus.start()
            return bus
        except Exception as e:
            logger.warning(f"Redis event bus failed, falling back to in-memory: {e}")
            return InMemoryEventBus()
    
    return InMemoryEventBus()

# Global event bus instance
event_bus = get_event_bus()
