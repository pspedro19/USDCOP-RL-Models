"""
Redis Streams Manager for Multi-Model Trading
==============================================
Comprehensive manager for Redis Streams with producer, consumer, and SSE support.

Features:
- Stream management with retention policies
- Producer with auto-serialization and timestamps
- Consumer with consumer groups and acknowledgment
- SSE adapter for FastAPI Server-Sent Events
- Utilities for monitoring and maintenance

Compatible with Redis 7.x and uses redis-py with async support.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-26
"""

import asyncio
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

import redis.asyncio as aioredis
import yaml
from redis.asyncio import Redis
from redis.exceptions import ConnectionError, ResponseError, TimeoutError

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class StreamConfig:
    """Configuration for a single stream."""
    name: str
    pattern: Optional[str] = None
    retention_ms: int = 86400000  # 24 hours
    maxlen: int = 10000
    approximate_maxlen: bool = True
    consumer_groups: List[Dict[str, Any]] = field(default_factory=list)
    message_schema: Dict[str, str] = field(default_factory=dict)


@dataclass
class RedisStreamsConfig:
    """Configuration for Redis Streams manager."""
    host: str = "redis"
    port: int = 6379
    password: str = ""
    db: int = 0
    streams: Dict[str, StreamConfig] = field(default_factory=dict)

    # Consumer settings
    block_timeout_ms: int = 5000
    batch_size: int = 10
    dead_letter_after_retries: int = 3
    claim_min_idle_ms: int = 300000

    # Producer settings
    max_retries: int = 3
    retry_delay_ms: int = 1000
    add_timestamp: bool = True
    add_uuid: bool = True

    # SSE settings
    keepalive_interval_ms: int = 15000
    max_connection_time_ms: int = 3600000
    retry_interval_ms: int = 5000

    @classmethod
    def from_yaml(cls, config_path: Optional[str] = None) -> "RedisStreamsConfig":
        """Load configuration from YAML file."""
        if config_path is None:
            # Default: Look for config relative to services directory
            services_dir = Path(__file__).parent.parent
            config_path = services_dir.parent / "config" / "redis_streams.yaml"

        config_path = Path(config_path)
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Parse connection settings
        conn = data.get("connection", {})
        host = os.environ.get("REDIS_HOST", conn.get("host", "redis"))
        port = int(os.environ.get("REDIS_PORT", conn.get("port", 6379)))
        password = os.environ.get("REDIS_PASSWORD", conn.get("password", ""))
        db = conn.get("db", 0)

        # Parse stream configs
        streams = {}
        for stream_name, stream_data in data.get("streams", {}).items():
            streams[stream_name] = StreamConfig(
                name=stream_data.get("name", ""),
                pattern=stream_data.get("pattern"),
                retention_ms=stream_data.get("retention_ms", 86400000),
                maxlen=stream_data.get("maxlen", 10000),
                approximate_maxlen=stream_data.get("approximate_maxlen", True),
                consumer_groups=stream_data.get("consumer_groups", []),
                message_schema=stream_data.get("message_schema", {}),
            )

        # Parse consumer settings
        consumers = data.get("consumers", {})
        producers = data.get("producers", {})
        sse = data.get("sse", {})

        return cls(
            host=host,
            port=port,
            password=password,
            db=db,
            streams=streams,
            block_timeout_ms=consumers.get("block_timeout_ms", 5000),
            batch_size=consumers.get("batch_size", 10),
            dead_letter_after_retries=consumers.get("dead_letter_after_retries", 3),
            claim_min_idle_ms=consumers.get("claim_min_idle_ms", 300000),
            max_retries=producers.get("max_retries", 3),
            retry_delay_ms=producers.get("retry_delay_ms", 1000),
            add_timestamp=producers.get("add_timestamp", True),
            add_uuid=producers.get("add_uuid", True),
            keepalive_interval_ms=sse.get("keepalive_interval_ms", 15000),
            max_connection_time_ms=sse.get("max_connection_time_ms", 3600000),
            retry_interval_ms=sse.get("retry_interval_ms", 5000),
        )


# =============================================================================
# Stream Manager
# =============================================================================

class RedisStreamsManager:
    """
    Manages Redis Streams connections and provides factory methods
    for producers and consumers.
    """

    def __init__(self, config: Optional[RedisStreamsConfig] = None):
        """
        Initialize the Redis Streams Manager.

        Args:
            config: Configuration object. If None, loads from YAML.
        """
        self.config = config or RedisStreamsConfig.from_yaml()
        self._redis: Optional[Redis] = None
        self._connection_pool: Optional[aioredis.ConnectionPool] = None
        self._is_connected = False

    async def connect(self) -> None:
        """Establish connection to Redis."""
        if self._is_connected:
            return

        try:
            self._connection_pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                password=self.config.password or None,
                db=self.config.db,
                decode_responses=True,
                max_connections=100,
            )
            self._redis = aioredis.Redis(connection_pool=self._connection_pool)

            # Test connection
            await self._redis.ping()
            self._is_connected = True
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
        if self._connection_pool:
            await self._connection_pool.disconnect()
        self._is_connected = False
        logger.info("Disconnected from Redis")

    async def ensure_connected(self) -> Redis:
        """Ensure connection is established and return Redis client."""
        if not self._is_connected:
            await self.connect()
        return self._redis

    async def create_stream(
        self,
        stream_name: str,
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> bool:
        """
        Create a stream with initial entry if it doesn't exist.

        Args:
            stream_name: Name of the stream
            maxlen: Maximum length (uses config default if None)
            approximate: Use approximate trimming (~)

        Returns:
            True if stream was created, False if already exists
        """
        redis = await self.ensure_connected()

        try:
            # Check if stream exists
            exists = await redis.exists(stream_name)
            if exists:
                logger.debug(f"Stream {stream_name} already exists")
                return False

            # Create stream with initial message
            await redis.xadd(
                stream_name,
                {"_init": "stream_created", "_timestamp": datetime.now(timezone.utc).isoformat()},
                maxlen=maxlen or self.config.streams.get("signals", StreamConfig("")).maxlen,
                approximate=approximate,
            )
            logger.info(f"Created stream: {stream_name}")
            return True

        except Exception as e:
            logger.error(f"Error creating stream {stream_name}: {e}")
            raise

    async def create_consumer_group(
        self,
        stream_name: str,
        group_name: str,
        start_id: str = "$",
        mkstream: bool = True,
    ) -> bool:
        """
        Create a consumer group for a stream.

        Args:
            stream_name: Name of the stream
            group_name: Name of the consumer group
            start_id: Starting message ID ("$" for new, "0" for all)
            mkstream: Create stream if it doesn't exist

        Returns:
            True if group was created, False if already exists
        """
        redis = await self.ensure_connected()

        try:
            await redis.xgroup_create(
                stream_name,
                group_name,
                id=start_id,
                mkstream=mkstream,
            )
            logger.info(f"Created consumer group {group_name} for stream {stream_name}")
            return True

        except ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {group_name} already exists for {stream_name}")
                return False
            raise

    async def setup_streams_from_config(self) -> Dict[str, List[str]]:
        """
        Create all streams and consumer groups defined in config.

        Returns:
            Dict mapping stream names to list of created consumer groups
        """
        results = {}

        for stream_key, stream_config in self.config.streams.items():
            stream_name = stream_config.name or stream_config.pattern
            if not stream_name:
                continue

            # For pattern-based streams, skip (they're created dynamically)
            if stream_config.pattern and "{" in stream_config.pattern:
                continue

            await self.create_stream(
                stream_name,
                maxlen=stream_config.maxlen,
                approximate=stream_config.approximate_maxlen,
            )

            results[stream_name] = []
            for group_config in stream_config.consumer_groups:
                group_name = group_config.get("name")
                start_id = group_config.get("start_id", "$")
                if group_name:
                    created = await self.create_consumer_group(
                        stream_name,
                        group_name,
                        start_id=start_id,
                    )
                    if created:
                        results[stream_name].append(group_name)

        return results

    def get_stream_name(self, stream_key: str, model_id: Optional[str] = None) -> str:
        """
        Get the actual stream name from config key.

        Args:
            stream_key: Config key (e.g., "signals", "market")
            model_id: Model ID for pattern-based streams

        Returns:
            Actual stream name
        """
        stream_config = self.config.streams.get(stream_key)
        if not stream_config:
            raise ValueError(f"Unknown stream key: {stream_key}")

        if stream_config.pattern and model_id:
            return stream_config.pattern.format(model_id=model_id)
        return stream_config.name

    def get_producer(self) -> "StreamProducer":
        """Get a producer instance."""
        return StreamProducer(self)

    def get_consumer(
        self,
        group_name: str,
        consumer_name: Optional[str] = None,
    ) -> "StreamConsumer":
        """
        Get a consumer instance.

        Args:
            group_name: Consumer group name
            consumer_name: Unique consumer name (auto-generated if None)
        """
        return StreamConsumer(self, group_name, consumer_name)

    def get_sse_adapter(self) -> "SSEAdapter":
        """Get an SSE adapter instance."""
        return SSEAdapter(self)


# =============================================================================
# Producer
# =============================================================================

class StreamProducer:
    """
    Producer for publishing messages to Redis Streams.

    Features:
    - Auto-serialization of complex types
    - Automatic timestamp and UUID addition
    - Retry logic for reliability
    - Batch publishing support
    """

    def __init__(self, manager: RedisStreamsManager):
        """
        Initialize the producer.

        Args:
            manager: RedisStreamsManager instance
        """
        self.manager = manager
        self.config = manager.config

    def _serialize_value(self, value: Any) -> str:
        """Serialize a value for Redis."""
        if isinstance(value, (dict, list)):
            return json.dumps(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, bool):
            return "1" if value else "0"
        return str(value)

    def _prepare_message(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Prepare message for publishing."""
        message = {}

        # Add UUID if configured
        if self.config.add_uuid and "_id" not in data:
            message["_id"] = str(uuid.uuid4())

        # Add timestamp if configured
        if self.config.add_timestamp and "_timestamp" not in data:
            message["_timestamp"] = datetime.now(timezone.utc).isoformat()

        # Serialize all values
        for key, value in data.items():
            message[key] = self._serialize_value(value)

        return message

    async def publish(
        self,
        stream_name: str,
        data: Dict[str, Any],
        maxlen: Optional[int] = None,
        approximate: bool = True,
    ) -> str:
        """
        Publish a message to a stream.

        Args:
            stream_name: Target stream name
            data: Message data
            maxlen: Override maxlen from config
            approximate: Use approximate trimming

        Returns:
            Message ID assigned by Redis
        """
        redis = await self.manager.ensure_connected()
        message = self._prepare_message(data)

        for attempt in range(self.config.max_retries):
            try:
                message_id = await redis.xadd(
                    stream_name,
                    message,
                    maxlen=maxlen,
                    approximate=approximate,
                )
                logger.debug(f"Published message {message_id} to {stream_name}")
                return message_id

            except (ConnectionError, TimeoutError) as e:
                if attempt == self.config.max_retries - 1:
                    logger.error(f"Failed to publish after {self.config.max_retries} attempts: {e}")
                    raise
                await asyncio.sleep(self.config.retry_delay_ms / 1000)

    async def publish_signal(
        self,
        model_id: str,
        signal_data: Dict[str, Any],
    ) -> str:
        """
        Publish a signal to a model-specific stream.

        Args:
            model_id: Model identifier (e.g., "ppo_v1", "xgboost_ensemble")
            signal_data: Signal data including type, confidence, etc.

        Returns:
            Message ID
        """
        # Add model_id to data
        data = {"model_id": model_id, **signal_data}

        # Get stream name from pattern
        stream_config = self.config.streams.get("signals")
        if stream_config and stream_config.pattern:
            stream_name = stream_config.pattern.format(model_id=model_id)
        else:
            stream_name = f"signals:{model_id}:stream"

        # Ensure stream and consumer groups exist
        redis = await self.manager.ensure_connected()
        if not await redis.exists(stream_name):
            await self.manager.create_stream(
                stream_name,
                maxlen=stream_config.maxlen if stream_config else 10000,
            )
            # Create consumer groups
            if stream_config:
                for group_config in stream_config.consumer_groups:
                    group_name = group_config.get("name")
                    if group_name:
                        await self.manager.create_consumer_group(
                            stream_name,
                            group_name,
                            start_id=group_config.get("start_id", "$"),
                        )

        return await self.publish(
            stream_name,
            data,
            maxlen=stream_config.maxlen if stream_config else None,
        )

    async def publish_all_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
    ) -> Dict[str, str]:
        """
        Publish signals from multiple models.

        Also publishes aggregated data to the signals_all stream.

        Args:
            signals: Dict mapping model_id to signal_data

        Returns:
            Dict mapping model_id to message_id
        """
        results = {}
        aggregated = []

        for model_id, signal_data in signals.items():
            # Publish to individual stream
            message_id = await self.publish_signal(model_id, signal_data)
            results[model_id] = message_id

            # Prepare for aggregated stream
            aggregated.append({
                "model_id": model_id,
                "signal_type": signal_data.get("signal_type"),
                "confidence": signal_data.get("confidence"),
                "source_stream": f"signals:{model_id}:stream",
            })

        # Publish to aggregated stream
        all_stream = self.manager.get_stream_name("signals_all")
        for agg_signal in aggregated:
            await self.publish(all_stream, agg_signal)

        return results

    async def publish_event(
        self,
        event_type: str,
        payload: Dict[str, Any],
        severity: str = "INFO",
        model_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> str:
        """
        Publish an event to the events/audit stream.

        Args:
            event_type: Type of event (SIGNAL, TRADE, ERROR, ALERT, AUDIT)
            payload: Event payload
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
            model_id: Associated model ID if applicable
            correlation_id: For tracking related events

        Returns:
            Message ID
        """
        stream_name = self.manager.get_stream_name("events")

        data = {
            "event_type": event_type,
            "severity": severity,
            "payload": payload,
            "correlation_id": correlation_id or str(uuid.uuid4()),
        }
        if model_id:
            data["model_id"] = model_id

        return await self.publish(stream_name, data)

    async def publish_market_data(
        self,
        ohlcv: Dict[str, Any],
        bar_number: Optional[int] = None,
    ) -> str:
        """
        Publish market OHLCV data.

        Args:
            ohlcv: OHLCV data with timestamp, open, high, low, close, volume
            bar_number: Trading session bar number (1-59)

        Returns:
            Message ID
        """
        stream_name = self.manager.get_stream_name("market")

        data = {**ohlcv}
        if bar_number is not None:
            data["bar_number"] = bar_number

        return await self.publish(stream_name, data)


# =============================================================================
# Consumer
# =============================================================================

class StreamConsumer:
    """
    Consumer for reading messages from Redis Streams.

    Features:
    - Consumer group support for load distribution
    - Message acknowledgment
    - Pending message handling
    - Dead letter queue support
    - Reconnection logic
    """

    def __init__(
        self,
        manager: RedisStreamsManager,
        group_name: str,
        consumer_name: Optional[str] = None,
    ):
        """
        Initialize the consumer.

        Args:
            manager: RedisStreamsManager instance
            group_name: Consumer group name
            consumer_name: Unique consumer name (auto-generated if None)
        """
        self.manager = manager
        self.config = manager.config
        self.group_name = group_name
        self.consumer_name = consumer_name or f"consumer-{uuid.uuid4().hex[:8]}"
        self._subscriptions: Dict[str, Callable] = {}
        self._running = False

    def _deserialize_message(self, message: Dict[str, str]) -> Dict[str, Any]:
        """Deserialize message values from Redis."""
        result = {}
        for key, value in message.items():
            # Try to parse as JSON
            if value.startswith(("{", "[")):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass

            # Try to parse as number
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
                continue
            except ValueError:
                pass

            # Try to parse as datetime
            if "T" in value and ("Z" in value or "+" in value or value.count("-") >= 2):
                try:
                    result[key] = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    continue
                except ValueError:
                    pass

            # Keep as string
            result[key] = value

        return result

    async def read(
        self,
        stream_name: str,
        count: Optional[int] = None,
        block_ms: Optional[int] = None,
    ) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Read messages from a stream using consumer group.

        Args:
            stream_name: Stream to read from
            count: Max messages to read
            block_ms: Blocking timeout (uses config default if None)

        Returns:
            List of (stream_name, message_id, data) tuples
        """
        redis = await self.manager.ensure_connected()

        try:
            # Read new messages
            response = await redis.xreadgroup(
                self.group_name,
                self.consumer_name,
                {stream_name: ">"},
                count=count or self.config.batch_size,
                block=block_ms or self.config.block_timeout_ms,
            )

            if not response:
                return []

            results = []
            for stream, messages in response:
                for message_id, data in messages:
                    results.append((
                        stream,
                        message_id,
                        self._deserialize_message(data),
                    ))

            return results

        except ResponseError as e:
            if "NOGROUP" in str(e):
                # Create group if it doesn't exist
                await self.manager.create_consumer_group(stream_name, self.group_name)
                return []
            raise

    async def ack(self, stream_name: str, message_id: str) -> bool:
        """
        Acknowledge a message.

        Args:
            stream_name: Stream name
            message_id: Message ID to acknowledge

        Returns:
            True if acknowledged successfully
        """
        redis = await self.manager.ensure_connected()
        result = await redis.xack(stream_name, self.group_name, message_id)
        return result > 0

    async def ack_batch(
        self,
        stream_name: str,
        message_ids: List[str],
    ) -> int:
        """
        Acknowledge multiple messages.

        Args:
            stream_name: Stream name
            message_ids: List of message IDs

        Returns:
            Number of messages acknowledged
        """
        if not message_ids:
            return 0
        redis = await self.manager.ensure_connected()
        return await redis.xack(stream_name, self.group_name, *message_ids)

    async def get_pending(
        self,
        stream_name: str,
        count: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get pending (unacknowledged) messages for this consumer.

        Args:
            stream_name: Stream name
            count: Max messages to return

        Returns:
            List of pending message info
        """
        redis = await self.manager.ensure_connected()

        try:
            result = await redis.xpending_range(
                stream_name,
                self.group_name,
                min="-",
                max="+",
                count=count,
                consumername=self.consumer_name,
            )
            return result

        except ResponseError:
            return []

    async def claim_pending(
        self,
        stream_name: str,
        min_idle_ms: Optional[int] = None,
        count: int = 10,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Claim pending messages from other consumers.

        Args:
            stream_name: Stream name
            min_idle_ms: Minimum idle time (uses config default if None)
            count: Max messages to claim

        Returns:
            List of (message_id, data) tuples
        """
        redis = await self.manager.ensure_connected()
        min_idle = min_idle_ms or self.config.claim_min_idle_ms

        try:
            # First get pending messages info
            pending = await redis.xpending_range(
                stream_name,
                self.group_name,
                min="-",
                max="+",
                count=count,
            )

            if not pending:
                return []

            # Filter by idle time and claim
            message_ids = [
                p["message_id"] for p in pending
                if p.get("time_since_delivered", 0) >= min_idle
            ]

            if not message_ids:
                return []

            result = await redis.xclaim(
                stream_name,
                self.group_name,
                self.consumer_name,
                min_idle,
                message_ids,
            )

            return [
                (msg_id, self._deserialize_message(data))
                for msg_id, data in result
            ]

        except ResponseError as e:
            logger.warning(f"Error claiming pending messages: {e}")
            return []

    async def subscribe_model(
        self,
        model_id: str,
        callback: Callable[[str, str, Dict[str, Any]], Any],
    ) -> None:
        """
        Subscribe to a model's signal stream.

        Args:
            model_id: Model identifier
            callback: Async function called with (stream_name, message_id, data)
        """
        stream_config = self.config.streams.get("signals")
        if stream_config and stream_config.pattern:
            stream_name = stream_config.pattern.format(model_id=model_id)
        else:
            stream_name = f"signals:{model_id}:stream"

        # Ensure consumer group exists
        await self.manager.create_consumer_group(
            stream_name,
            self.group_name,
            mkstream=True,
        )

        self._subscriptions[stream_name] = callback

    async def subscribe_all(
        self,
        callback: Callable[[str, str, Dict[str, Any]], Any],
    ) -> None:
        """
        Subscribe to the aggregated signals stream.

        Args:
            callback: Async function called with (stream_name, message_id, data)
        """
        stream_name = self.manager.get_stream_name("signals_all")

        # Ensure consumer group exists
        await self.manager.create_consumer_group(
            stream_name,
            self.group_name,
            mkstream=True,
        )

        self._subscriptions[stream_name] = callback

    async def run(self) -> None:
        """
        Start consuming messages from subscribed streams.

        This is a blocking operation that runs until stop() is called.
        """
        if not self._subscriptions:
            raise ValueError("No streams subscribed. Call subscribe_* first.")

        self._running = True
        logger.info(
            f"Consumer {self.consumer_name} starting for streams: "
            f"{list(self._subscriptions.keys())}"
        )

        while self._running:
            try:
                redis = await self.manager.ensure_connected()

                # Build streams dict for xreadgroup
                streams_dict = {
                    stream: ">" for stream in self._subscriptions.keys()
                }

                # Read from all subscribed streams
                response = await redis.xreadgroup(
                    self.group_name,
                    self.consumer_name,
                    streams_dict,
                    count=self.config.batch_size,
                    block=self.config.block_timeout_ms,
                )

                if not response:
                    continue

                # Process messages
                for stream_name, messages in response:
                    callback = self._subscriptions.get(stream_name)
                    if not callback:
                        continue

                    for message_id, data in messages:
                        try:
                            deserialized = self._deserialize_message(data)
                            await callback(stream_name, message_id, deserialized)

                            # Auto-acknowledge
                            await self.ack(stream_name, message_id)

                        except Exception as e:
                            logger.error(
                                f"Error processing message {message_id}: {e}"
                            )
                            # Could implement dead letter queue here

            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Connection error, reconnecting: {e}")
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Unexpected error in consumer loop: {e}")
                await asyncio.sleep(1)

        logger.info(f"Consumer {self.consumer_name} stopped")

    def stop(self) -> None:
        """Stop the consumer loop."""
        self._running = False


# =============================================================================
# SSE Adapter
# =============================================================================

class SSEAdapter:
    """
    Server-Sent Events adapter for streaming Redis messages to HTTP clients.

    Compatible with FastAPI's StreamingResponse.
    """

    def __init__(self, manager: RedisStreamsManager):
        """
        Initialize the SSE adapter.

        Args:
            manager: RedisStreamsManager instance
        """
        self.manager = manager
        self.config = manager.config

    def _format_sse(
        self,
        data: Any,
        event: Optional[str] = None,
        id: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> str:
        """Format data as SSE message."""
        lines = []

        if id:
            lines.append(f"id: {id}")
        if event:
            lines.append(f"event: {event}")
        if retry:
            lines.append(f"retry: {retry}")

        # Data can be multiline
        if isinstance(data, (dict, list)):
            data = json.dumps(data)

        for line in str(data).split("\n"):
            lines.append(f"data: {line}")

        lines.append("")  # Empty line to end message
        return "\n".join(lines) + "\n"

    async def sse_generator(
        self,
        model_id: Optional[str] = None,
        stream_type: str = "signals",
    ) -> AsyncGenerator[str, None]:
        """
        Async generator for SSE streaming.

        Args:
            model_id: Specific model to stream (None for all)
            stream_type: Type of stream ("signals", "market", "events")

        Yields:
            SSE-formatted strings
        """
        redis = await self.manager.ensure_connected()

        # Determine stream name
        if model_id:
            stream_config = self.config.streams.get(stream_type)
            if stream_config and stream_config.pattern:
                stream_name = stream_config.pattern.format(model_id=model_id)
            else:
                stream_name = f"{stream_type}:{model_id}:stream"
        else:
            stream_name = self.manager.get_stream_name(
                f"{stream_type}_all" if stream_type == "signals" else stream_type
            )

        # Send initial connection message
        yield self._format_sse(
            {"status": "connected", "stream": stream_name},
            event="connection",
            retry=self.config.retry_interval_ms,
        )

        last_id = "$"  # Only new messages
        last_keepalive = asyncio.get_event_loop().time()
        start_time = asyncio.get_event_loop().time()
        keepalive_interval = self.config.keepalive_interval_ms / 1000
        max_time = self.config.max_connection_time_ms / 1000

        try:
            while True:
                current_time = asyncio.get_event_loop().time()

                # Check max connection time
                if current_time - start_time > max_time:
                    yield self._format_sse(
                        {"reason": "max_connection_time"},
                        event="disconnect",
                    )
                    break

                # Read messages
                try:
                    response = await redis.xread(
                        {stream_name: last_id},
                        count=10,
                        block=min(
                            int(keepalive_interval * 1000),
                            self.config.block_timeout_ms,
                        ),
                    )

                    if response:
                        for stream, messages in response:
                            for message_id, data in messages:
                                last_id = message_id

                                # Deserialize and send
                                deserialized = {}
                                for key, value in data.items():
                                    if value.startswith(("{", "[")):
                                        try:
                                            deserialized[key] = json.loads(value)
                                        except json.JSONDecodeError:
                                            deserialized[key] = value
                                    else:
                                        deserialized[key] = value

                                yield self._format_sse(
                                    deserialized,
                                    event=stream_type,
                                    id=message_id,
                                )
                                last_keepalive = current_time

                except TimeoutError:
                    pass

                # Send keepalive if needed
                if current_time - last_keepalive >= keepalive_interval:
                    yield self._format_sse(
                        {"type": "keepalive"},
                        event="keepalive",
                    )
                    last_keepalive = current_time

        except asyncio.CancelledError:
            yield self._format_sse(
                {"reason": "client_disconnect"},
                event="disconnect",
            )
        except Exception as e:
            logger.error(f"SSE generator error: {e}")
            yield self._format_sse(
                {"error": str(e)},
                event="error",
            )

    async def signals_sse(
        self,
        model_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        SSE generator specifically for trading signals.

        Args:
            model_id: Specific model or None for all models
        """
        async for event in self.sse_generator(model_id, "signals"):
            yield event

    async def market_sse(self) -> AsyncGenerator[str, None]:
        """SSE generator for market data."""
        async for event in self.sse_generator(None, "market"):
            yield event

    async def events_sse(self) -> AsyncGenerator[str, None]:
        """SSE generator for trading events."""
        async for event in self.sse_generator(None, "events"):
            yield event


# =============================================================================
# Utilities
# =============================================================================

class StreamUtilities:
    """Utility functions for stream monitoring and maintenance."""

    def __init__(self, manager: RedisStreamsManager):
        """Initialize utilities."""
        self.manager = manager

    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a stream.

        Args:
            stream_name: Stream name

        Returns:
            Dict with stream info including length, groups, etc.
        """
        redis = await self.manager.ensure_connected()

        try:
            info = await redis.xinfo_stream(stream_name)

            # Get consumer groups info
            groups = []
            try:
                groups_info = await redis.xinfo_groups(stream_name)
                groups = [
                    {
                        "name": g.get("name"),
                        "consumers": g.get("consumers"),
                        "pending": g.get("pending"),
                        "last_delivered_id": g.get("last-delivered-id"),
                    }
                    for g in groups_info
                ]
            except ResponseError:
                pass

            return {
                "length": info.get("length", 0),
                "radix_tree_keys": info.get("radix-tree-keys", 0),
                "radix_tree_nodes": info.get("radix-tree-nodes", 0),
                "first_entry": info.get("first-entry"),
                "last_entry": info.get("last-entry"),
                "groups": groups,
            }

        except ResponseError as e:
            if "no such key" in str(e).lower():
                return {"error": "Stream does not exist"}
            raise

    async def get_pending_messages(
        self,
        stream_name: str,
        group_name: str,
    ) -> Dict[str, Any]:
        """
        Get pending messages summary for a consumer group.

        Args:
            stream_name: Stream name
            group_name: Consumer group name

        Returns:
            Dict with pending messages summary
        """
        redis = await self.manager.ensure_connected()

        try:
            pending = await redis.xpending(stream_name, group_name)

            return {
                "count": pending.get("pending", 0),
                "min_id": pending.get("min"),
                "max_id": pending.get("max"),
                "consumers": pending.get("consumers", []),
            }

        except ResponseError as e:
            if "NOGROUP" in str(e):
                return {"error": "Consumer group does not exist"}
            raise

    async def get_all_streams_info(self) -> Dict[str, Dict[str, Any]]:
        """Get info for all configured streams."""
        results = {}

        for stream_key, stream_config in self.manager.config.streams.items():
            stream_name = stream_config.name
            if not stream_name or "{" in (stream_config.pattern or ""):
                continue

            info = await self.get_stream_info(stream_name)
            results[stream_name] = info

        return results

    async def cleanup_old_messages(
        self,
        stream_name: str,
        max_age_ms: Optional[int] = None,
    ) -> int:
        """
        Remove messages older than max_age.

        Note: Redis doesn't support time-based trimming directly.
        This uses MINID to trim by message ID timestamp.

        Args:
            stream_name: Stream name
            max_age_ms: Max message age in milliseconds

        Returns:
            Number of messages removed (approximate)
        """
        redis = await self.manager.ensure_connected()

        if max_age_ms is None:
            # Get from config
            for cfg in self.manager.config.streams.values():
                if cfg.name == stream_name:
                    max_age_ms = cfg.retention_ms
                    break

        if not max_age_ms:
            return 0

        # Calculate MINID based on timestamp
        min_timestamp = int(
            (datetime.now(timezone.utc).timestamp() * 1000) - max_age_ms
        )
        min_id = f"{min_timestamp}-0"

        try:
            # Get length before
            info_before = await redis.xinfo_stream(stream_name)
            len_before = info_before.get("length", 0)

            # Trim using MINID
            await redis.xtrim(stream_name, minid=min_id)

            # Get length after
            info_after = await redis.xinfo_stream(stream_name)
            len_after = info_after.get("length", 0)

            removed = len_before - len_after
            if removed > 0:
                logger.info(f"Cleaned up {removed} messages from {stream_name}")

            return removed

        except ResponseError as e:
            logger.warning(f"Error cleaning up stream {stream_name}: {e}")
            return 0

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Redis Streams health.

        Returns:
            Dict with health status
        """
        try:
            redis = await self.manager.ensure_connected()

            # Basic ping
            await redis.ping()

            # Get all streams info
            streams_info = await self.get_all_streams_info()

            # Check for issues
            issues = []
            for stream_name, info in streams_info.items():
                if "error" in info:
                    issues.append(f"{stream_name}: {info['error']}")
                elif info.get("length", 0) == 0:
                    issues.append(f"{stream_name}: empty")

            return {
                "status": "healthy" if not issues else "degraded",
                "connected": True,
                "streams_count": len(streams_info),
                "total_messages": sum(
                    s.get("length", 0) for s in streams_info.values()
                    if "error" not in s
                ),
                "issues": issues if issues else None,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }


# =============================================================================
# Factory Functions
# =============================================================================

# Singleton manager instance
_manager: Optional[RedisStreamsManager] = None


def get_redis_streams_manager(
    config: Optional[RedisStreamsConfig] = None,
) -> RedisStreamsManager:
    """
    Get or create the Redis Streams Manager singleton.

    Args:
        config: Optional configuration (uses YAML if None)

    Returns:
        RedisStreamsManager instance
    """
    global _manager
    if _manager is None:
        _manager = RedisStreamsManager(config)
    return _manager


async def get_producer() -> StreamProducer:
    """Get a producer instance (convenience function)."""
    manager = get_redis_streams_manager()
    await manager.connect()
    return manager.get_producer()


async def get_consumer(
    group_name: str,
    consumer_name: Optional[str] = None,
) -> StreamConsumer:
    """Get a consumer instance (convenience function)."""
    manager = get_redis_streams_manager()
    await manager.connect()
    return manager.get_consumer(group_name, consumer_name)


async def get_sse_adapter() -> SSEAdapter:
    """Get an SSE adapter instance (convenience function)."""
    manager = get_redis_streams_manager()
    await manager.connect()
    return manager.get_sse_adapter()


async def get_utilities() -> StreamUtilities:
    """Get utilities instance (convenience function)."""
    manager = get_redis_streams_manager()
    await manager.connect()
    return StreamUtilities(manager)
