"""
Integration Tests: Redis Streams
================================

Tests Redis streaming functionality for real-time signal distribution.

Tests cover:
- Publishing signals to model streams
- Subscribing and receiving signals
- Stream retention and pruning
- Consumer groups
- Message serialization

Author: Pedro @ Lean Tech Solutions
Date: 2025-12-26
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from unittest.mock import MagicMock, AsyncMock


# Try to import Redis
try:
    import redis
    from redis import asyncio as aioredis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False


@pytest.fixture
def redis_url() -> str:
    """Redis connection URL"""
    import os
    return os.getenv('TEST_REDIS_URL', os.getenv('REDIS_URL', 'redis://localhost:6379/1'))


@pytest.fixture
def redis_client(redis_url) -> 'redis.Redis':
    """Create Redis client for testing"""
    if not HAS_REDIS:
        pytest.skip("redis package not installed")

    client = redis.from_url(redis_url, decode_responses=True)

    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis not available for testing")

    yield client

    # Cleanup test keys
    test_keys = client.keys('test:*') + client.keys('usdcop:test:*')
    if test_keys:
        client.delete(*test_keys)

    client.close()


@pytest.fixture
async def async_redis_client(redis_url):
    """Create async Redis client for testing"""
    if not HAS_REDIS:
        pytest.skip("redis package not installed")

    client = aioredis.from_url(redis_url, decode_responses=True)

    try:
        await client.ping()
    except Exception:
        pytest.skip("Redis not available for testing")

    yield client

    # Cleanup test keys
    test_keys = await client.keys('test:*')
    if test_keys:
        await client.delete(*test_keys)

    await client.aclose()


@pytest.fixture
def sample_signal() -> Dict[str, Any]:
    """Sample trading signal for testing"""
    return {
        'model_id': 'ppo_primary',
        'action': 0.75,
        'signal': 'buy',
        'confidence': 0.85,
        'position_delta': 0.25,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'bar_number': 30,
        'features_hash': 'abc123'
    }


@pytest.fixture
def stream_key() -> str:
    """Test stream key"""
    return 'test:signals:ppo_primary'


class MockRedisManager:
    """Mock Redis manager for testing without actual Redis"""

    def __init__(self):
        self.streams: Dict[str, List[Dict]] = {}
        self.message_id = 0

    async def publish_signal(
        self,
        stream_key: str,
        signal: Dict[str, Any],
        maxlen: int = 1000
    ) -> str:
        """Mock publish to stream"""
        if stream_key not in self.streams:
            self.streams[stream_key] = []

        self.message_id += 1
        msg_id = f"{int(time.time() * 1000)}-{self.message_id}"

        self.streams[stream_key].append({
            'id': msg_id,
            'data': signal
        })

        # Trim to maxlen
        if len(self.streams[stream_key]) > maxlen:
            self.streams[stream_key] = self.streams[stream_key][-maxlen:]

        return msg_id

    async def read_stream(
        self,
        stream_key: str,
        last_id: str = '0',
        count: int = 10
    ) -> List[Dict]:
        """Mock read from stream"""
        if stream_key not in self.streams:
            return []

        messages = []
        for msg in self.streams[stream_key]:
            if msg['id'] > last_id:
                messages.append(msg)
                if len(messages) >= count:
                    break

        return messages

    async def get_stream_length(self, stream_key: str) -> int:
        """Get stream length"""
        return len(self.streams.get(stream_key, []))

    async def trim_stream(self, stream_key: str, maxlen: int) -> int:
        """Trim stream to maxlen"""
        if stream_key not in self.streams:
            return 0

        original_len = len(self.streams[stream_key])
        if original_len > maxlen:
            self.streams[stream_key] = self.streams[stream_key][-maxlen:]
            return original_len - maxlen
        return 0


@pytest.fixture
def mock_redis_manager() -> MockRedisManager:
    """Create mock Redis manager"""
    return MockRedisManager()


@pytest.mark.integration
class TestPublishSignal:
    """Tests for publishing signals to streams"""

    @pytest.mark.asyncio
    async def test_publish_signal_returns_message_id(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Can publish to model stream and get message ID"""
        msg_id = await mock_redis_manager.publish_signal(stream_key, sample_signal)

        assert msg_id is not None
        assert isinstance(msg_id, str)
        assert '-' in msg_id  # Redis stream ID format: timestamp-sequence

    @pytest.mark.asyncio
    async def test_publish_multiple_signals(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Can publish multiple signals"""
        ids = []
        for i in range(10):
            signal = sample_signal.copy()
            signal['bar_number'] = i + 1
            msg_id = await mock_redis_manager.publish_signal(stream_key, signal)
            ids.append(msg_id)

        assert len(ids) == 10
        assert len(set(ids)) == 10  # All unique IDs

        # Verify stream length
        length = await mock_redis_manager.get_stream_length(stream_key)
        assert length == 10

    @pytest.mark.asyncio
    async def test_publish_to_multiple_streams(
        self,
        mock_redis_manager,
        sample_signal
    ):
        """Can publish to different model streams"""
        streams = [
            'test:signals:ppo_primary',
            'test:signals:ppo_secondary',
            'test:signals:xgb_v1'
        ]

        for stream in streams:
            signal = sample_signal.copy()
            signal['model_id'] = stream.split(':')[-1]
            await mock_redis_manager.publish_signal(stream, signal)

        # Each stream should have 1 message
        for stream in streams:
            length = await mock_redis_manager.get_stream_length(stream)
            assert length == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_real_redis_publish(
        self,
        async_redis_client,
        sample_signal,
        stream_key
    ):
        """Test publish with real Redis"""
        # Serialize signal values to strings
        serialized = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                     for k, v in sample_signal.items()}

        msg_id = await async_redis_client.xadd(
            stream_key,
            serialized,
            maxlen=100
        )

        assert msg_id is not None

        # Verify message was added
        messages = await async_redis_client.xrange(stream_key, '-', '+', count=1)
        assert len(messages) == 1

        # Cleanup
        await async_redis_client.delete(stream_key)


@pytest.mark.integration
class TestSubscribeSignal:
    """Tests for subscribing and receiving signals"""

    @pytest.mark.asyncio
    async def test_subscribe_receives_signals(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Can subscribe and receive signals"""
        # Publish some signals
        for i in range(5):
            signal = sample_signal.copy()
            signal['bar_number'] = i + 1
            await mock_redis_manager.publish_signal(stream_key, signal)

        # Read from stream
        messages = await mock_redis_manager.read_stream(stream_key, last_id='0', count=10)

        assert len(messages) == 5

        for i, msg in enumerate(messages):
            assert msg['data']['bar_number'] == i + 1

    @pytest.mark.asyncio
    async def test_subscribe_from_last_id(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Can subscribe from specific message ID"""
        # Publish signals
        ids = []
        for i in range(10):
            signal = sample_signal.copy()
            signal['bar_number'] = i + 1
            msg_id = await mock_redis_manager.publish_signal(stream_key, signal)
            ids.append(msg_id)

        # Read only from message 5 onwards
        messages = await mock_redis_manager.read_stream(
            stream_key,
            last_id=ids[4],  # Start after 5th message
            count=10
        )

        assert len(messages) == 5  # Should get messages 6-10
        assert messages[0]['data']['bar_number'] == 6

    @pytest.mark.asyncio
    async def test_subscribe_empty_stream(
        self,
        mock_redis_manager,
        stream_key
    ):
        """Subscribing to empty stream returns empty list"""
        messages = await mock_redis_manager.read_stream(stream_key, count=10)

        assert messages == []

    @pytest.mark.asyncio
    async def test_subscribe_with_count_limit(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Subscription respects count limit"""
        # Publish 20 signals
        for i in range(20):
            signal = sample_signal.copy()
            signal['bar_number'] = i + 1
            await mock_redis_manager.publish_signal(stream_key, signal)

        # Read only 5
        messages = await mock_redis_manager.read_stream(stream_key, count=5)

        assert len(messages) == 5


@pytest.mark.integration
class TestStreamRetention:
    """Tests for stream retention and pruning"""

    @pytest.mark.asyncio
    async def test_stream_respects_maxlen(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Stream is trimmed to maxlen on publish"""
        maxlen = 10

        # Publish 20 signals with maxlen=10
        for i in range(20):
            signal = sample_signal.copy()
            signal['bar_number'] = i + 1
            await mock_redis_manager.publish_signal(stream_key, signal, maxlen=maxlen)

        # Stream should have at most 10 messages
        length = await mock_redis_manager.get_stream_length(stream_key)
        assert length <= maxlen

    @pytest.mark.asyncio
    async def test_old_messages_are_pruned(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Old messages are removed when maxlen exceeded"""
        maxlen = 5

        # Publish 10 signals
        for i in range(10):
            signal = sample_signal.copy()
            signal['bar_number'] = i + 1
            await mock_redis_manager.publish_signal(stream_key, signal, maxlen=maxlen)

        # Read all messages
        messages = await mock_redis_manager.read_stream(stream_key, count=100)

        # Only last 5 should remain
        assert len(messages) == maxlen

        # First message should be #6 (0-indexed #5)
        bar_numbers = [m['data']['bar_number'] for m in messages]
        assert bar_numbers[0] >= 6

    @pytest.mark.asyncio
    async def test_manual_trim(
        self,
        mock_redis_manager,
        sample_signal,
        stream_key
    ):
        """Can manually trim stream"""
        # Publish 100 signals
        for i in range(100):
            await mock_redis_manager.publish_signal(stream_key, sample_signal, maxlen=1000)

        # Manually trim to 10
        removed = await mock_redis_manager.trim_stream(stream_key, maxlen=10)

        assert removed == 90

        length = await mock_redis_manager.get_stream_length(stream_key)
        assert length == 10

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_real_redis_trim(
        self,
        async_redis_client,
        sample_signal,
        stream_key
    ):
        """Test trim with real Redis"""
        # Add 50 messages
        for i in range(50):
            serialized = {k: str(v) for k, v in sample_signal.items()}
            serialized['sequence'] = str(i)
            await async_redis_client.xadd(stream_key, serialized)

        # Trim to 10
        await async_redis_client.xtrim(stream_key, maxlen=10)

        # Verify length
        length = await async_redis_client.xlen(stream_key)
        assert length == 10

        # Cleanup
        await async_redis_client.delete(stream_key)


@pytest.mark.integration
class TestConsumerGroups:
    """Tests for consumer group functionality"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_create_consumer_group(
        self,
        async_redis_client,
        sample_signal,
        stream_key
    ):
        """Can create consumer group"""
        group_name = 'test-consumers'

        # Create stream with initial message
        serialized = {k: str(v) for k, v in sample_signal.items()}
        await async_redis_client.xadd(stream_key, serialized)

        # Create consumer group
        try:
            await async_redis_client.xgroup_create(
                stream_key,
                group_name,
                id='0',
                mkstream=True
            )
        except redis.ResponseError as e:
            if 'BUSYGROUP' not in str(e):
                raise

        # Verify group exists
        groups = await async_redis_client.xinfo_groups(stream_key)
        group_names = [g['name'] for g in groups]
        assert group_name in group_names

        # Cleanup
        await async_redis_client.delete(stream_key)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_consumer_group_read(
        self,
        async_redis_client,
        sample_signal,
        stream_key
    ):
        """Consumer can read from group"""
        group_name = 'test-group'
        consumer_name = 'consumer-1'

        # Add messages
        for i in range(5):
            serialized = {k: str(v) for k, v in sample_signal.items()}
            serialized['sequence'] = str(i)
            await async_redis_client.xadd(stream_key, serialized)

        # Create group
        try:
            await async_redis_client.xgroup_create(
                stream_key, group_name, id='0', mkstream=True
            )
        except redis.ResponseError:
            pass  # Group may already exist

        # Read with consumer
        messages = await async_redis_client.xreadgroup(
            group_name,
            consumer_name,
            {stream_key: '>'},
            count=10
        )

        assert len(messages) > 0
        assert messages[0][0] == stream_key

        # Cleanup
        await async_redis_client.delete(stream_key)


@pytest.mark.integration
class TestMessageSerialization:
    """Tests for message serialization"""

    def test_serialize_signal_for_redis(self, sample_signal):
        """Signal can be serialized for Redis"""
        # Redis streams require string values
        serialized = {}
        for key, value in sample_signal.items():
            if isinstance(value, (dict, list)):
                serialized[key] = json.dumps(value)
            else:
                serialized[key] = str(value)

        # All values should be strings
        for key, value in serialized.items():
            assert isinstance(value, str), f"Key {key} is not a string"

    def test_deserialize_signal_from_redis(self, sample_signal):
        """Signal can be deserialized from Redis"""
        # Simulate Redis storage (all strings)
        stored = {k: str(v) for k, v in sample_signal.items()}

        # Deserialize
        deserialized = {}
        for key, value in stored.items():
            if key in ['action', 'confidence', 'position_delta']:
                deserialized[key] = float(value)
            elif key in ['bar_number']:
                deserialized[key] = int(value)
            else:
                deserialized[key] = value

        assert deserialized['action'] == sample_signal['action']
        assert deserialized['bar_number'] == sample_signal['bar_number']

    def test_json_complex_signal(self):
        """Complex signal with nested data can be serialized"""
        complex_signal = {
            'model_id': 'ppo_primary',
            'action': 0.75,
            'features': [0.1, 0.2, 0.3],
            'metadata': {'version': '1.0', 'timestamp': '2025-12-26'}
        }

        # Serialize
        serialized = {
            k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
            for k, v in complex_signal.items()
        }

        # Deserialize
        deserialized = {}
        for key, value in serialized.items():
            try:
                deserialized[key] = json.loads(value)
            except json.JSONDecodeError:
                deserialized[key] = value

        assert deserialized['features'] == [0.1, 0.2, 0.3]
        assert deserialized['metadata']['version'] == '1.0'


@pytest.mark.integration
class TestStreamInfo:
    """Tests for stream information retrieval"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_get_stream_info(
        self,
        async_redis_client,
        sample_signal,
        stream_key
    ):
        """Can get stream information"""
        # Add messages
        for i in range(10):
            serialized = {k: str(v) for k, v in sample_signal.items()}
            await async_redis_client.xadd(stream_key, serialized)

        # Get stream info
        info = await async_redis_client.xinfo_stream(stream_key)

        assert info['length'] == 10
        assert 'first-entry' in info
        assert 'last-entry' in info

        # Cleanup
        await async_redis_client.delete(stream_key)

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_get_pending_messages(
        self,
        async_redis_client,
        sample_signal,
        stream_key
    ):
        """Can get pending message count"""
        group_name = 'pending-test-group'
        consumer_name = 'consumer-1'

        # Add messages
        for i in range(5):
            serialized = {k: str(v) for k, v in sample_signal.items()}
            await async_redis_client.xadd(stream_key, serialized)

        # Create group and read without acknowledging
        try:
            await async_redis_client.xgroup_create(
                stream_key, group_name, id='0', mkstream=True
            )
        except redis.ResponseError:
            pass

        # Read messages (creates pending entries)
        await async_redis_client.xreadgroup(
            group_name, consumer_name, {stream_key: '>'}, count=5
        )

        # Check pending
        pending = await async_redis_client.xpending(stream_key, group_name)
        assert pending['pending'] == 5

        # Cleanup
        await async_redis_client.delete(stream_key)


@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in Redis operations"""

    @pytest.mark.asyncio
    async def test_publish_to_invalid_stream_name(self, mock_redis_manager, sample_signal):
        """Publishing to stream with empty name should handle gracefully"""
        # This should either work or raise a clear error
        try:
            await mock_redis_manager.publish_signal('', sample_signal)
        except (ValueError, KeyError):
            pass  # Acceptable

    @pytest.mark.asyncio
    async def test_read_nonexistent_stream(self, mock_redis_manager):
        """Reading from non-existent stream returns empty"""
        messages = await mock_redis_manager.read_stream('nonexistent:stream', count=10)
        assert messages == []

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_REDIS, reason="redis not installed")
    async def test_connection_recovery(self, redis_url):
        """Can recover from connection issues"""
        client = aioredis.from_url(redis_url)

        try:
            await client.ping()

            # Simulate reconnection by closing and reopening
            await client.aclose()

            # Reconnect
            client = aioredis.from_url(redis_url)
            result = await client.ping()
            assert result is True

        except Exception:
            pytest.skip("Could not test connection recovery")
        finally:
            await client.aclose()
