"""
Integration Tests for Real-time to Database Pipeline
======================================================

Tests the complete flow from API → DB → Redis → WebSocket:
- Real database operations (with testcontainers or test DB)
- ON CONFLICT UPSERT behavior
- Gap detection and backfill
- Redis caching and pub/sub
- Data consistency checks

Uses testcontainers for isolated testing or connects to test database.

Target Coverage: >80%
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict, Any
import pytz
import asyncpg
import redis

# Mark all tests as integration tests
pytestmark = pytest.mark.integration

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')


# ============================================================================
# Test: Complete API → DB → Redis Flow
# ============================================================================

class TestCompleteDataFlow:
    """Test complete data flow from ingestion to storage"""

    @pytest.mark.asyncio
    async def test_api_to_database_insertion(self, db_pool, clean_db, sample_ohlcv_data):
        """Test complete flow: API data → Database"""
        async with db_pool.acquire() as conn:
            # Insert sample data
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                sample_ohlcv_data['symbol'],
                sample_ohlcv_data['datetime'],
                sample_ohlcv_data['open'],
                sample_ohlcv_data['high'],
                sample_ohlcv_data['low'],
                sample_ohlcv_data['close'],
                sample_ohlcv_data['volume'],
                'test_source',
                sample_ohlcv_data['timeframe']
            )

            # Verify insertion
            result = await conn.fetchrow("""
                SELECT * FROM market_data
                WHERE symbol = $1 AND source = 'test_source'
                ORDER BY datetime DESC LIMIT 1
            """, sample_ohlcv_data['symbol'])

            assert result is not None
            assert result['symbol'] == sample_ohlcv_data['symbol']
            assert result['close'] == sample_ohlcv_data['close']

    @pytest.mark.asyncio
    async def test_database_to_redis_cache(self, db_pool, redis_client, clean_db, clean_redis, sample_ohlcv_data):
        """Test flow: Database → Redis cache"""
        # Insert to database
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                sample_ohlcv_data['symbol'],
                sample_ohlcv_data['datetime'],
                sample_ohlcv_data['open'],
                sample_ohlcv_data['high'],
                sample_ohlcv_data['low'],
                sample_ohlcv_data['close'],
                sample_ohlcv_data['volume'],
                'test_source',
                sample_ohlcv_data['timeframe']
            )

            # Fetch and cache in Redis
            latest_data = await conn.fetchrow("""
                SELECT * FROM market_data
                WHERE symbol = $1 AND source = 'test_source'
                ORDER BY datetime DESC LIMIT 1
            """, sample_ohlcv_data['symbol'])

        # Cache in Redis
        cache_key = f"test:usdcop:latest"
        cache_data = {
            'symbol': latest_data['symbol'],
            'price': float(latest_data['close']),
            'timestamp': latest_data['datetime'].isoformat()
        }

        redis_client.setex(cache_key, 300, json.dumps(cache_data, default=str))

        # Verify cache
        cached = redis_client.get(cache_key)
        assert cached is not None

        cached_obj = json.loads(cached)
        assert cached_obj['symbol'] == sample_ohlcv_data['symbol']
        assert cached_obj['price'] == float(sample_ohlcv_data['close'])


# ============================================================================
# Test: ON CONFLICT Behavior
# ============================================================================

class TestOnConflictBehavior:
    """Test UPSERT and ON CONFLICT behavior"""

    @pytest.mark.asyncio
    async def test_upsert_on_duplicate_key(self, db_pool, clean_db):
        """Test ON CONFLICT behavior when inserting duplicate data"""
        test_datetime = datetime.now(COT_TZ).replace(microsecond=0)

        async with db_pool.acquire() as conn:
            # First insert
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                'USDCOP',
                test_datetime,
                Decimal('4250.00'),
                Decimal('4255.00'),
                Decimal('4248.00'),
                Decimal('4252.00'),
                5000,
                'test_source',
                '5min'
            )

            # Second insert with ON CONFLICT (should update)
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (symbol, datetime, timeframe, source)
                DO UPDATE SET
                    high = GREATEST(market_data.high, EXCLUDED.high),
                    low = LEAST(market_data.low, EXCLUDED.low),
                    close = EXCLUDED.close,
                    volume = market_data.volume + EXCLUDED.volume,
                    updated_at = NOW()
            """,
                'USDCOP',
                test_datetime,
                Decimal('4250.00'),
                Decimal('4260.00'),  # Higher high
                Decimal('4245.00'),  # Lower low
                Decimal('4258.00'),  # New close
                3000,  # Additional volume
                'test_source',
                '5min'
            )

            # Verify UPSERT behavior
            result = await conn.fetchrow("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP' AND datetime = $1 AND source = 'test_source'
            """, test_datetime)

            assert result is not None
            assert result['high'] == Decimal('4260.00')  # Updated to max
            assert result['low'] == Decimal('4245.00')   # Updated to min
            assert result['close'] == Decimal('4258.00')  # Updated to latest
            assert result['volume'] == 8000  # Summed volumes

    @pytest.mark.asyncio
    async def test_realtime_table_insertion(self, db_pool, clean_db):
        """Test insertion into realtime_market_data table"""
        test_time = datetime.now(COT_TZ)

        async with db_pool.acquire() as conn:
            # Insert real-time tick
            await conn.execute("""
                INSERT INTO realtime_market_data
                (symbol, time, bid, ask, last, volume, spread, session_date, trading_session, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                'USDCOP',
                test_time,
                Decimal('4250.00'),
                Decimal('4251.00'),
                Decimal('4250.50'),
                1000,
                Decimal('1.00'),
                test_time.date(),
                True,
                'test_websocket'
            )

            # Verify insertion
            result = await conn.fetchrow("""
                SELECT * FROM realtime_market_data
                WHERE symbol = 'USDCOP' AND source = 'test_websocket'
                ORDER BY time DESC LIMIT 1
            """)

            assert result is not None
            assert result['symbol'] == 'USDCOP'
            assert result['last'] == Decimal('4250.50')
            assert result['spread'] == Decimal('1.00')


# ============================================================================
# Test: Gap Detection and Backfill
# ============================================================================

class TestGapDetection:
    """Test gap detection and backfill logic"""

    @pytest.mark.asyncio
    async def test_detect_data_gap(self, db_pool, clean_db):
        """Test detection of data gaps"""
        async with db_pool.acquire() as conn:
            # Insert data with gaps
            base_time = datetime(2025, 10, 22, 10, 0, 0, tzinfo=COT_TZ)

            # Insert bars at 10:00, 10:05, 10:15 (missing 10:10)
            for i in [0, 5, 15]:
                bar_time = base_time + timedelta(minutes=i)
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    'USDCOP',
                    bar_time,
                    Decimal('4250.00'),
                    Decimal('4255.00'),
                    Decimal('4248.00'),
                    Decimal('4252.00'),
                    5000,
                    'test_source',
                    '5min'
                )

            # Detect gap
            gap_query = """
                WITH time_series AS (
                    SELECT datetime,
                           LAG(datetime) OVER (ORDER BY datetime) as prev_datetime
                    FROM market_data
                    WHERE symbol = $1 AND source = 'test_source'
                    AND datetime >= $2
                    ORDER BY datetime
                )
                SELECT *
                FROM time_series
                WHERE EXTRACT(EPOCH FROM (datetime - prev_datetime)) > 360
            """

            gaps = await conn.fetch(gap_query, 'USDCOP', base_time)

            assert len(gaps) > 0  # Should detect the 10-minute gap

    @pytest.mark.asyncio
    async def test_backfill_missing_data(self, db_pool, clean_db):
        """Test backfilling missing data"""
        async with db_pool.acquire() as conn:
            # Create initial data with gap
            base_time = datetime(2025, 10, 22, 10, 0, 0, tzinfo=COT_TZ)

            # Insert at 10:00 and 10:10 (missing 10:05)
            for i in [0, 10]:
                bar_time = base_time + timedelta(minutes=i)
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    'USDCOP',
                    bar_time,
                    Decimal('4250.00'),
                    Decimal('4255.00'),
                    Decimal('4248.00'),
                    Decimal('4252.00'),
                    5000,
                    'test_source',
                    '5min'
                )

            # Backfill missing bar at 10:05
            backfill_time = base_time + timedelta(minutes=5)
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                'USDCOP',
                backfill_time,
                Decimal('4250.00'),
                Decimal('4253.00'),
                Decimal('4249.00'),
                Decimal('4251.00'),
                3000,
                'test_backfill',
                '5min'
            )

            # Verify backfill
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime BETWEEN $1 AND $2
            """, base_time, base_time + timedelta(minutes=10))

            assert count == 3  # All three bars should exist


# ============================================================================
# Test: Data Consistency
# ============================================================================

class TestDataConsistency:
    """Test data consistency across operations"""

    @pytest.mark.asyncio
    async def test_timestamp_consistency(self, db_pool, clean_db):
        """Test timestamp handling and consistency"""
        test_time = datetime(2025, 10, 22, 10, 30, 0, tzinfo=COT_TZ)

        async with db_pool.acquire() as conn:
            # Insert with specific timestamp
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                'USDCOP',
                test_time,
                Decimal('4250.00'),
                Decimal('4255.00'),
                Decimal('4248.00'),
                Decimal('4252.00'),
                5000,
                'test_source',
                '5min'
            )

            # Retrieve and verify timestamp
            result = await conn.fetchrow("""
                SELECT datetime FROM market_data
                WHERE symbol = 'USDCOP' AND source = 'test_source'
                ORDER BY datetime DESC LIMIT 1
            """)

            assert result is not None
            # Convert to COT timezone for comparison
            stored_time = result['datetime'].astimezone(COT_TZ)
            assert stored_time.replace(tzinfo=None) == test_time.replace(tzinfo=None)

    @pytest.mark.asyncio
    async def test_ohlcv_integrity(self, db_pool, clean_db):
        """Test OHLCV data integrity constraints"""
        test_time = datetime.now(COT_TZ).replace(microsecond=0)

        async with db_pool.acquire() as conn:
            # Insert valid OHLCV data
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                'USDCOP',
                test_time,
                Decimal('4250.00'),
                Decimal('4255.00'),  # high >= open, close
                Decimal('4248.00'),  # low <= open, close
                Decimal('4252.00'),
                5000,
                'test_source',
                '5min'
            )

            # Verify OHLC relationships
            result = await conn.fetchrow("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP' AND datetime = $1 AND source = 'test_source'
            """, test_time)

            assert result['high'] >= result['open']
            assert result['high'] >= result['close']
            assert result['low'] <= result['open']
            assert result['low'] <= result['close']
            assert result['volume'] >= 0

    @pytest.mark.asyncio
    async def test_concurrent_insertions(self, db_pool, clean_db):
        """Test handling of concurrent insertions"""
        test_time = datetime.now(COT_TZ).replace(microsecond=0)

        # Simulate concurrent insertions
        async def insert_data(conn, price_offset):
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                ON CONFLICT (symbol, datetime, timeframe, source)
                DO UPDATE SET
                    high = GREATEST(market_data.high, EXCLUDED.high),
                    low = LEAST(market_data.low, EXCLUDED.low),
                    close = EXCLUDED.close,
                    volume = market_data.volume + EXCLUDED.volume,
                    updated_at = NOW()
            """,
                'USDCOP',
                test_time,
                Decimal(str(4250.00 + price_offset)),
                Decimal(str(4255.00 + price_offset)),
                Decimal(str(4248.00 + price_offset)),
                Decimal(str(4252.00 + price_offset)),
                1000,
                'test_concurrent',
                '5min'
            )

        async with db_pool.acquire() as conn:
            # Run multiple concurrent inserts
            await asyncio.gather(
                insert_data(conn, 0),
                insert_data(conn, 1),
                insert_data(conn, 2)
            )

            # Verify final state
            result = await conn.fetchrow("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP' AND datetime = $1 AND source = 'test_concurrent'
            """, test_time)

            assert result is not None
            assert result['volume'] == 3000  # All volumes should be summed


# ============================================================================
# Test: Redis Pub/Sub Integration
# ============================================================================

class TestRedisPubSub:
    """Test Redis pub/sub integration"""

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self, redis_client, clean_redis):
        """Test Redis pub/sub for real-time updates"""
        channel = 'test:market_data:realtime'

        # Create subscriber
        pubsub = redis_client.pubsub()
        pubsub.subscribe(channel)

        # Publish message
        test_message = {
            'type': 'price_update',
            'symbol': 'USDCOP',
            'price': 4250.50,
            'timestamp': datetime.now(COT_TZ).isoformat()
        }

        redis_client.publish(channel, json.dumps(test_message, default=str))

        # Wait a bit for message propagation
        await asyncio.sleep(0.1)

        # Receive message
        message = pubsub.get_message(timeout=1)
        if message and message['type'] == 'message':
            received_data = json.loads(message['data'])
            assert received_data['symbol'] == 'USDCOP'
            assert received_data['price'] == 4250.50

        pubsub.close()

    def test_cache_expiration(self, redis_client, clean_redis):
        """Test Redis cache expiration"""
        test_key = 'test:usdcop:cache'
        test_data = {'symbol': 'USDCOP', 'price': 4250.50}

        # Set with 1 second TTL
        redis_client.setex(test_key, 1, json.dumps(test_data))

        # Verify exists
        assert redis_client.exists(test_key) == 1

        # Wait for expiration
        import time
        time.sleep(1.5)

        # Verify expired
        assert redis_client.exists(test_key) == 0


# ============================================================================
# Test: Performance and Scalability
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_bulk_insert_performance(self, db_pool, clean_db, generate_market_ticks):
        """Test bulk insert performance"""
        ticks = generate_market_ticks(count=100)
        base_time = datetime.now(COT_TZ).replace(microsecond=0)

        async with db_pool.acquire() as conn:
            start_time = datetime.now()

            # Bulk insert
            for i, tick in enumerate(ticks):
                bar_time = base_time + timedelta(minutes=i * 5)
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    'USDCOP',
                    bar_time,
                    Decimal(str(tick['price'])),
                    Decimal(str(tick['ask'])),
                    Decimal(str(tick['bid'])),
                    Decimal(str(tick['price'])),
                    tick['volume'],
                    'test_bulk',
                    '5min'
                )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            # Verify all inserted
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM market_data
                WHERE source = 'test_bulk'
            """)

            assert count == 100
            # Should complete in reasonable time (< 5 seconds for 100 records)
            assert duration < 5.0

    @pytest.mark.asyncio
    async def test_query_performance(self, db_pool, clean_db, generate_market_ticks):
        """Test query performance with indexes"""
        # Insert test data
        ticks = generate_market_ticks(count=50)
        base_time = datetime.now(COT_TZ).replace(microsecond=0)

        async with db_pool.acquire() as conn:
            for i, tick in enumerate(ticks):
                bar_time = base_time + timedelta(minutes=i * 5)
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    'USDCOP',
                    bar_time,
                    Decimal(str(tick['price'])),
                    Decimal(str(tick['ask'])),
                    Decimal(str(tick['bid'])),
                    Decimal(str(tick['price'])),
                    tick['volume'],
                    'test_query',
                    '5min'
                )

            # Measure query performance
            start_time = datetime.now()

            results = await conn.fetch("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP' AND source = 'test_query'
                AND datetime BETWEEN $1 AND $2
                ORDER BY datetime DESC
            """, base_time, base_time + timedelta(hours=5))

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            assert len(results) > 0
            # Query should be fast with proper indexes (< 100ms)
            assert duration < 0.1
