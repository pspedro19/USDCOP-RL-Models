"""
Redis Streams Bridge Service
============================

Consumes prediction signals from L5 DAG via Redis Streams.
Connects to: signals:ppo_primary:stream (published by l5_multi_model_inference.py)

This is the production signal source for live trading.
L5 DAG runs during trading hours and publishes predictions to Redis Streams.

Design Pattern: Observer/Consumer
- Subscribes to Redis Streams for model predictions
- Forwards signals to SignalBridgeOrchestrator

Author: Trading Team
Version: 1.0.0
Date: 2026-01-23
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Optional, Callable, Any, Dict, List
from uuid import UUID, uuid4

import redis.asyncio as aioredis
from redis.exceptions import ConnectionError, TimeoutError, ResponseError

from app.contracts.signal_bridge import InferenceSignalCreate
from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisStreamsBridge:
    """
    Consumes signals from L5 DAG via Redis Streams.

    Features:
    - Automatic reconnection with backoff
    - Consumer group support for reliability
    - Message acknowledgment
    - Health monitoring

    Usage:
        bridge = RedisStreamsBridge(
            redis_url="redis://redis:6379/0",
            stream_name="signals:ppo_primary:stream",
            on_signal_received=handle_signal,
        )

        # Start consuming in background
        await bridge.start()

        # Stop when done
        await bridge.stop()
    """

    # Connection state
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"

    def __init__(
        self,
        redis_url: Optional[str] = None,
        stream_name: str = "signals:ppo_primary:stream",
        consumer_group: str = "signalbridge",
        consumer_name: Optional[str] = None,
        on_signal_received: Optional[Callable[[InferenceSignalCreate], Any]] = None,
        default_credential_id: Optional[UUID] = None,
        reconnect_delay_base: float = 1.0,
        reconnect_delay_max: float = 60.0,
        block_timeout_ms: int = 5000,
        batch_size: int = 10,
    ):
        """
        Initialize Redis Streams Bridge.

        Args:
            redis_url: Redis connection URL
            stream_name: Redis stream to consume from
            consumer_group: Consumer group name
            consumer_name: Unique consumer name (auto-generated if None)
            on_signal_received: Callback when signal is received
            default_credential_id: Default exchange credential to use
            reconnect_delay_base: Base delay for reconnection (seconds)
            reconnect_delay_max: Maximum reconnection delay (seconds)
            block_timeout_ms: XREADGROUP block timeout
            batch_size: Max messages to read per batch
        """
        self.redis_url = redis_url or getattr(settings, "redis_url", "redis://redis:6379/0")
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = consumer_name or f"signalbridge-{uuid4().hex[:8]}"
        self.on_signal_received = on_signal_received
        self.default_credential_id = default_credential_id

        # Connection settings
        self.reconnect_delay_base = reconnect_delay_base
        self.reconnect_delay_max = reconnect_delay_max
        self.reconnect_attempt = 0
        self.block_timeout_ms = block_timeout_ms
        self.batch_size = batch_size

        # State
        self._connection_state = self.DISCONNECTED
        self._redis: Optional[aioredis.Redis] = None
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None
        self._last_message_at: Optional[datetime] = None
        self._messages_received = 0
        self._signals_processed = 0
        self._errors_count = 0

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connection_state == self.CONNECTED

    @property
    def state(self) -> str:
        """Get current connection state."""
        return self._connection_state

    @property
    def stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "state": self._connection_state,
            "is_connected": self.is_connected,
            "messages_received": self._messages_received,
            "signals_processed": self._signals_processed,
            "errors_count": self._errors_count,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "reconnect_attempts": self.reconnect_attempt,
            "stream_name": self.stream_name,
            "consumer_group": self.consumer_group,
            "consumer_name": self.consumer_name,
        }

    async def start(self) -> None:
        """
        Start the Redis Streams bridge.

        Begins consuming messages from the L5 DAG stream.
        Runs until stop() is called.
        """
        if self._running:
            logger.warning("Redis Streams bridge already running")
            return

        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info(
            f"Redis Streams bridge started, consuming from {self.stream_name}"
        )

    async def stop(self) -> None:
        """
        Stop the Redis Streams bridge gracefully.
        """
        logger.info("Stopping Redis Streams bridge...")
        self._running = False

        # Cancel consumer task
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        # Close Redis connection
        if self._redis:
            await self._redis.close()
            self._redis = None

        self._connection_state = self.DISCONNECTED
        logger.info("Redis Streams bridge stopped")

    async def _connect(self) -> None:
        """Establish connection to Redis."""
        self._connection_state = self.CONNECTING
        logger.info(f"Connecting to Redis: {self.redis_url}")

        try:
            self._redis = aioredis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_timeout=10.0,
            )
            # Test connection
            await self._redis.ping()

            # Create consumer group if it doesn't exist
            await self._ensure_consumer_group()

            self._connection_state = self.CONNECTED
            self.reconnect_attempt = 0
            logger.info("Connected to Redis")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connection_state = self.DISCONNECTED
            raise

    async def _ensure_consumer_group(self) -> None:
        """Ensure consumer group exists for the stream."""
        try:
            await self._redis.xgroup_create(
                self.stream_name,
                self.consumer_group,
                id="$",  # Start from new messages
                mkstream=True,  # Create stream if doesn't exist
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {self.consumer_group} already exists")
            else:
                raise

    async def _consume_loop(self) -> None:
        """
        Main consumption loop with automatic reconnection.
        """
        while self._running:
            try:
                if self._connection_state != self.CONNECTED:
                    await self._connect()

                await self._read_and_process()

            except asyncio.CancelledError:
                break
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Redis connection error: {e}")
                self._connection_state = self.DISCONNECTED
                self._errors_count += 1
            except Exception as e:
                logger.error(f"Redis consume loop error: {e}")
                self._errors_count += 1

            if self._running and self._connection_state != self.CONNECTED:
                # Reconnect with exponential backoff
                await self._reconnect_delay()

    async def _read_and_process(self) -> None:
        """Read messages from stream and process them."""
        try:
            # Read new messages from consumer group
            response = await self._redis.xreadgroup(
                self.consumer_group,
                self.consumer_name,
                {self.stream_name: ">"},  # Only new messages
                count=self.batch_size,
                block=self.block_timeout_ms,
            )

            if not response:
                return

            for stream_name, messages in response:
                for message_id, data in messages:
                    await self._handle_message(message_id, data)
                    # Acknowledge message
                    await self._redis.xack(
                        self.stream_name, self.consumer_group, message_id
                    )

        except ResponseError as e:
            if "NOGROUP" in str(e):
                await self._ensure_consumer_group()
            else:
                raise

    async def _handle_message(self, message_id: str, data: Dict[str, str]) -> None:
        """
        Handle incoming Redis Stream message.

        Args:
            message_id: Redis message ID
            data: Message data (all values are strings from Redis)
        """
        self._messages_received += 1
        self._last_message_at = datetime.utcnow()

        try:
            # Skip init messages
            if data.get("_init"):
                return

            # Parse signal from L5 DAG format
            # L5 DAG publishes: signal_type, confidence, model_id, timestamp, etc.
            signal_type = data.get("signal_type", data.get("action_type", "HOLD"))
            confidence_str = data.get("confidence", "0.0")
            model_id = data.get("model_id", "ppo_primary")
            timestamp_str = data.get("timestamp") or data.get("_timestamp")
            symbol = data.get("symbol", "USDCOP")

            # Parse confidence (might be JSON or float string)
            try:
                if confidence_str.startswith("{"):
                    confidence_data = json.loads(confidence_str)
                    confidence = float(confidence_data.get("value", 0.0))
                else:
                    confidence = float(confidence_str)
            except (json.JSONDecodeError, ValueError):
                confidence = 0.0

            # Convert signal_type to action integer
            # L5 DAG uses: HOLD, BUY, SELL
            signal_map = {"SELL": 0, "HOLD": 1, "BUY": 2}
            action = signal_map.get(signal_type.upper(), 1)

            # Skip HOLD signals
            if action == 1:
                logger.debug(f"Skipping HOLD signal from {model_id}")
                return

            # Parse timestamp
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(
                        timestamp_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    timestamp = datetime.utcnow()
            else:
                timestamp = datetime.utcnow()

            # Build signal
            signal = InferenceSignalCreate(
                signal_id=uuid4(),
                model_id=model_id,
                action=action,
                confidence=confidence,
                symbol=symbol,
                credential_id=self.default_credential_id or uuid4(),
                timestamp=timestamp,
                metadata={
                    "source": "redis_streams_bridge",
                    "stream": self.stream_name,
                    "message_id": message_id,
                    "raw_data": data,
                },
            )

            # Invoke callback
            if self.on_signal_received:
                self._signals_processed += 1
                logger.info(
                    f"Signal from Redis: {signal.action_name} {symbol} "
                    f"(confidence={confidence:.3f}, model={model_id})"
                )
                await self.on_signal_received(signal)

        except Exception as e:
            logger.error(f"Error processing Redis message {message_id}: {e}", exc_info=True)
            self._errors_count += 1

    async def _reconnect_delay(self) -> None:
        """Wait before reconnecting with exponential backoff."""
        self.reconnect_attempt += 1
        self._connection_state = self.RECONNECTING

        # Calculate delay with exponential backoff
        delay = min(
            self.reconnect_delay_base * (2 ** (self.reconnect_attempt - 1)),
            self.reconnect_delay_max,
        )

        logger.info(
            f"Redis reconnecting in {delay:.1f}s (attempt {self.reconnect_attempt})"
        )
        await asyncio.sleep(delay)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the Redis connection.

        Returns:
            Dict with health status
        """
        is_healthy = self.is_connected and self._running

        # Check for stale connection
        if self._last_message_at:
            seconds_since_message = (datetime.utcnow() - self._last_message_at).total_seconds()
            # During trading hours, expect messages every 5 minutes from L5 DAG
            if seconds_since_message > 600:  # 10 minutes without messages
                is_healthy = False

        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "connected": self.is_connected,
            "state": self._connection_state,
            "last_message_at": self._last_message_at.isoformat() if self._last_message_at else None,
            "messages_received": self._messages_received,
            "signals_processed": self._signals_processed,
            "errors_count": self._errors_count,
            "stream_name": self.stream_name,
        }


class RedisStreamsBridgeManager:
    """
    Manager for Redis Streams bridge lifecycle.

    Provides a singleton-like interface for the bridge.
    """

    _instance: Optional[RedisStreamsBridge] = None

    @classmethod
    def get_instance(cls) -> Optional[RedisStreamsBridge]:
        """Get the current bridge instance."""
        return cls._instance

    @classmethod
    def create_instance(
        cls,
        on_signal_received: Optional[Callable[[InferenceSignalCreate], Any]] = None,
        **kwargs,
    ) -> RedisStreamsBridge:
        """
        Create a new bridge instance.

        Args:
            on_signal_received: Callback for received signals
            **kwargs: Additional arguments for RedisStreamsBridge

        Returns:
            New RedisStreamsBridge instance
        """
        if cls._instance is not None:
            logger.warning("Redis Streams bridge instance already exists - stopping old instance")
            asyncio.create_task(cls._instance.stop())

        cls._instance = RedisStreamsBridge(
            on_signal_received=on_signal_received,
            **kwargs,
        )
        return cls._instance

    @classmethod
    async def start(cls) -> bool:
        """Start the bridge instance."""
        if cls._instance is None:
            logger.error("No Redis Streams bridge instance to start")
            return False

        await cls._instance.start()
        return True

    @classmethod
    async def stop(cls) -> None:
        """Stop the bridge instance."""
        if cls._instance:
            await cls._instance.stop()
            cls._instance = None
