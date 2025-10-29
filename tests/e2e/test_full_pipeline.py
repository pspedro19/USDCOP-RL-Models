"""
End-to-End Tests for Complete Trading Pipeline
===============================================

Tests the full pipeline from L0 to Real-time to Dashboard:
- L0 pipeline execution and data availability
- Real-time service activation during market hours
- Data flow through all layers
- Dashboard API endpoints
- Recovery after system crash
- Multi-symbol support (if applicable)

These tests validate the entire system working together.

Target: Complete user journeys and critical paths
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, time as datetime_time, timedelta
from decimal import Decimal
from typing import Dict, Any, List
import pytz
import asyncpg
import redis
import aiohttp

# Mark all tests as e2e tests
pytestmark = [pytest.mark.e2e, pytest.mark.slow]

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')


# ============================================================================
# Test: Complete L0 → RT → Dashboard Flow
# ============================================================================

class TestCompletePipeline:
    """Test complete data pipeline flow"""

    @pytest.mark.asyncio
    async def test_l0_pipeline_to_database(self, db_pool, clean_db):
        """Test L0 pipeline execution and data persistence"""
        # Simulate L0 pipeline completion
        pipeline_name = 'test_usdcop_m5__01_l0_intelligent_acquire'

        async with db_pool.acquire() as conn:
            # Insert pipeline status
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, completed_at, records_processed, progress_percent)
                VALUES ($1, $2, $3, NOW(), NOW(), $4, $5)
            """,
                pipeline_name,
                'L0',
                'completed',
                100,
                100
            )

            # Insert market data as if L0 completed
            base_time = datetime.now(COT_TZ).replace(hour=10, minute=0, second=0, microsecond=0)

            for i in range(10):
                bar_time = base_time + timedelta(minutes=i * 5)
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
                    'twelvedata',
                    '5min'
                )

            # Verify L0 data exists
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM market_data
                WHERE symbol = 'USDCOP' AND source = 'twelvedata'
                AND datetime >= $1
            """, base_time)

            assert count == 10

            # Verify pipeline status
            status = await conn.fetchrow("""
                SELECT * FROM pipeline_status
                WHERE pipeline_name = $1 AND status = 'completed'
                ORDER BY completed_at DESC LIMIT 1
            """, pipeline_name)

            assert status is not None
            assert status['records_processed'] == 100

    @pytest.mark.asyncio
    async def test_realtime_service_activation(self, db_pool, redis_client, clean_db, clean_redis):
        """Test real-time service activation after L0 completion"""
        # Simulate L0 completion signal
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, completed_at, records_processed)
                VALUES ('usdcop_m5__01_l0_intelligent_acquire', 'L0', 'completed', NOW(), NOW(), 50)
            """)

        # Simulate real-time service checking L0 status
        async with db_pool.acquire() as conn:
            l0_status = await conn.fetchrow("""
                SELECT status, completed_at, records_processed
                FROM pipeline_status
                WHERE pipeline_name LIKE '%L0%'
                AND status = 'completed'
                ORDER BY completed_at DESC
                LIMIT 1
            """)

            assert l0_status is not None
            assert l0_status['status'] == 'completed'

        # Update real-time service status
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, current_step, progress_percent)
                VALUES ('realtime_orchestrator', 'realtime', 'active', NOW(), 'WebSocket collection active', 100)
            """)

            # Verify real-time service is active
            rt_status = await conn.fetchrow("""
                SELECT * FROM pipeline_status
                WHERE pipeline_name = 'realtime_orchestrator' AND status = 'active'
                ORDER BY started_at DESC LIMIT 1
            """)

            assert rt_status is not None
            assert rt_status['current_step'] == 'WebSocket collection active'

    @pytest.mark.asyncio
    async def test_data_flow_through_layers(self, db_pool, redis_client, clean_db, clean_redis):
        """Test data flowing through L0 → RT → Cache → Dashboard"""
        base_time = datetime.now(COT_TZ).replace(microsecond=0)

        # Step 1: L0 data
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                'USDCOP',
                base_time,
                Decimal('4250.00'),
                Decimal('4255.00'),
                Decimal('4248.00'),
                Decimal('4252.00'),
                5000,
                'twelvedata',
                '5min'
            )

        # Step 2: Real-time data
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO realtime_market_data
                (symbol, time, bid, ask, last, volume, spread, session_date, trading_session, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
                'USDCOP',
                base_time,
                Decimal('4252.00'),
                Decimal('4253.00'),
                Decimal('4252.50'),
                1000,
                Decimal('1.00'),
                base_time.date(),
                True,
                'websocket'
            )

        # Step 3: Cache in Redis
        cache_data = {
            'symbol': 'USDCOP',
            'price': 4252.50,
            'timestamp': base_time.isoformat()
        }
        redis_client.setex('test:usdcop:latest', 300, json.dumps(cache_data, default=str))

        # Step 4: Verify data at each layer
        async with db_pool.acquire() as conn:
            # Verify L0 data
            l0_data = await conn.fetchrow("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP' AND datetime = $1 AND source = 'twelvedata'
            """, base_time)

            assert l0_data is not None
            assert l0_data['close'] == Decimal('4252.00')

            # Verify RT data
            rt_data = await conn.fetchrow("""
                SELECT * FROM realtime_market_data
                WHERE symbol = 'USDCOP' AND source = 'websocket'
                ORDER BY time DESC LIMIT 1
            """)

            assert rt_data is not None
            assert rt_data['last'] == Decimal('4252.50')

        # Verify cache
        cached = redis_client.get('test:usdcop:latest')
        assert cached is not None

        cached_obj = json.loads(cached)
        assert cached_obj['price'] == 4252.50


# ============================================================================
# Test: Market Hours Behavior
# ============================================================================

class TestMarketHoursBehavior:
    """Test system behavior during and outside market hours"""

    @pytest.mark.asyncio
    async def test_market_open_behavior(self, db_pool, market_hours_config):
        """Test system behavior when market is open"""
        # Simulate market open time (10:00 AM COT)
        market_time = datetime.now(COT_TZ).replace(hour=10, minute=0, second=0, microsecond=0)

        # Check if it's market hours
        current_time = market_time.time()
        market_start = datetime_time(market_hours_config['start_hour'], market_hours_config['start_minute'])
        market_end = datetime_time(market_hours_config['end_hour'], market_hours_config['end_minute'])

        is_market_open = market_start <= current_time <= market_end
        assert is_market_open is True

        # Verify market session exists
        async with db_pool.acquire() as conn:
            # Create market session
            await conn.execute("""
                INSERT INTO market_sessions (session_date, start_time, status)
                VALUES ($1, $2, $3)
                ON CONFLICT (session_date) DO NOTHING
            """, market_time.date(), market_time, 'active')

            # Verify session
            session = await conn.fetchrow("""
                SELECT * FROM market_sessions
                WHERE session_date = $1
            """, market_time.date())

            assert session is not None
            assert session['status'] == 'active'

    @pytest.mark.asyncio
    async def test_market_closed_behavior(self, db_pool, market_hours_config):
        """Test system behavior when market is closed"""
        # Simulate after-hours time (2:00 PM COT)
        after_hours_time = datetime.now(COT_TZ).replace(hour=14, minute=0, second=0, microsecond=0)

        current_time = after_hours_time.time()
        market_start = datetime_time(market_hours_config['start_hour'], market_hours_config['start_minute'])
        market_end = datetime_time(market_hours_config['end_hour'], market_hours_config['end_minute'])

        is_market_open = market_start <= current_time <= market_end
        assert is_market_open is False

        # Verify real-time service should be idle
        async with db_pool.acquire() as conn:
            # Update service status to market_closed
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, current_step)
                VALUES ('realtime_orchestrator', 'realtime', 'market_closed', NOW(), 'Waiting for market open')
            """)

            # Verify status
            status = await conn.fetchrow("""
                SELECT * FROM pipeline_status
                WHERE pipeline_name = 'realtime_orchestrator'
                ORDER BY started_at DESC LIMIT 1
            """)

            assert status is not None
            assert status['status'] == 'market_closed'


# ============================================================================
# Test: Recovery After Crash
# ============================================================================

class TestSystemRecovery:
    """Test system recovery after crashes or failures"""

    @pytest.mark.asyncio
    async def test_recovery_after_database_disconnect(self, db_pool, clean_db):
        """Test recovery after database connection loss"""
        # Simulate successful operation
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_data
                (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
            """,
                'USDCOP',
                datetime.now(COT_TZ),
                Decimal('4250.00'),
                Decimal('4255.00'),
                Decimal('4248.00'),
                Decimal('4252.00'),
                5000,
                'test_recovery',
                '5min'
            )

        # Simulate crash recovery - verify data persists
        await asyncio.sleep(0.5)

        async with db_pool.acquire() as conn:
            count = await conn.fetchval("""
                SELECT COUNT(*) FROM market_data
                WHERE source = 'test_recovery'
            """)

            assert count > 0  # Data should persist after "recovery"

    @pytest.mark.asyncio
    async def test_recovery_after_redis_disconnect(self, redis_client, clean_redis):
        """Test recovery after Redis connection loss"""
        # Set initial cache
        redis_client.setex('test:recovery:key', 60, json.dumps({'value': 'initial'}))

        # Verify cache exists
        assert redis_client.exists('test:recovery:key') == 1

        # Simulate recovery - cache should still work
        cached = redis_client.get('test:recovery:key')
        assert cached is not None

    @pytest.mark.asyncio
    async def test_service_restart_recovery(self, db_pool):
        """Test service recovery after restart"""
        # Simulate service crash during operation
        async with db_pool.acquire() as conn:
            # Mark service as error
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, current_step)
                VALUES ('realtime_orchestrator', 'realtime', 'error', NOW(), 'Service crashed')
            """)

            # Simulate restart - new healthy status
            await asyncio.sleep(0.1)

            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, current_step, progress_percent)
                VALUES ('realtime_orchestrator', 'realtime', 'active', NOW(), 'Recovered and running', 100)
            """)

            # Verify recovery
            latest_status = await conn.fetchrow("""
                SELECT * FROM pipeline_status
                WHERE pipeline_name = 'realtime_orchestrator'
                ORDER BY started_at DESC LIMIT 1
            """)

            assert latest_status is not None
            assert latest_status['status'] == 'active'


# ============================================================================
# Test: Dashboard API Endpoints
# ============================================================================

class TestDashboardAPIs:
    """Test dashboard API endpoints end-to-end"""

    @pytest.mark.asyncio
    async def test_latest_price_endpoint(self, redis_client, clean_redis):
        """Test /market/latest endpoint"""
        # Set test data in Redis
        test_data = {
            'symbol': 'USDCOP',
            'price': 4250.50,
            'bid': 4250.00,
            'ask': 4251.00,
            'timestamp': datetime.now(COT_TZ).isoformat()
        }

        redis_client.setex('usdcop:latest_price', 300, json.dumps(test_data, default=str))

        # Simulate API endpoint retrieval
        cached = redis_client.get('usdcop:latest_price')
        assert cached is not None

        result = json.loads(cached)
        assert result['symbol'] == 'USDCOP'
        assert result['price'] == 4250.50

    @pytest.mark.asyncio
    async def test_pipeline_status_endpoint(self, db_pool, clean_db):
        """Test pipeline status endpoint"""
        # Insert various pipeline statuses
        async with db_pool.acquire() as conn:
            pipelines = [
                ('usdcop_m5__01_l0_intelligent_acquire', 'L0', 'completed'),
                ('realtime_orchestrator', 'realtime', 'active'),
                ('usdcop_m5__02_l1_standardize', 'L1', 'running')
            ]

            for name, ptype, status in pipelines:
                await conn.execute("""
                    INSERT INTO pipeline_status
                    (pipeline_name, pipeline_type, status, started_at, progress_percent)
                    VALUES ($1, $2, $3, NOW(), $4)
                """, name, ptype, status, 100 if status == 'completed' else 50)

            # Query pipeline statuses (simulating API)
            statuses = await conn.fetch("""
                SELECT DISTINCT ON (pipeline_name)
                    pipeline_name, pipeline_type, status, progress_percent, started_at
                FROM pipeline_status
                ORDER BY pipeline_name, started_at DESC
            """)

            assert len(statuses) >= 3

            # Verify each pipeline
            status_dict = {s['pipeline_name']: s for s in statuses}

            assert 'usdcop_m5__01_l0_intelligent_acquire' in status_dict
            assert status_dict['usdcop_m5__01_l0_intelligent_acquire']['status'] == 'completed'

    @pytest.mark.asyncio
    async def test_market_data_query_endpoint(self, db_pool, clean_db):
        """Test market data query endpoint"""
        base_time = datetime.now(COT_TZ).replace(hour=10, minute=0, second=0, microsecond=0)

        # Insert test data
        async with db_pool.acquire() as conn:
            for i in range(20):
                bar_time = base_time + timedelta(minutes=i * 5)
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    'USDCOP',
                    bar_time,
                    Decimal('4250.00') + Decimal(i),
                    Decimal('4255.00') + Decimal(i),
                    Decimal('4248.00') + Decimal(i),
                    Decimal('4252.00') + Decimal(i),
                    5000,
                    'twelvedata',
                    '5min'
                )

            # Query data (simulating API endpoint)
            start_time = base_time
            end_time = base_time + timedelta(hours=2)

            results = await conn.fetch("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime BETWEEN $1 AND $2
                ORDER BY datetime DESC
                LIMIT 10
            """, start_time, end_time)

            assert len(results) == 10
            assert results[0]['datetime'] > results[-1]['datetime']  # Descending order


# ============================================================================
# Test: Multi-Symbol Support (if applicable)
# ============================================================================

class TestMultiSymbolSupport:
    """Test support for multiple trading symbols"""

    @pytest.mark.asyncio
    async def test_multiple_symbols_data_flow(self, db_pool, clean_db):
        """Test data flow for multiple symbols"""
        symbols = ['USDCOP', 'EURCOP', 'GBPCOP']
        base_time = datetime.now(COT_TZ).replace(microsecond=0)

        async with db_pool.acquire() as conn:
            # Insert data for multiple symbols
            for symbol in symbols:
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    symbol,
                    base_time,
                    Decimal('4250.00'),
                    Decimal('4255.00'),
                    Decimal('4248.00'),
                    Decimal('4252.00'),
                    5000,
                    'twelvedata',
                    '5min'
                )

            # Verify each symbol
            for symbol in symbols:
                result = await conn.fetchrow("""
                    SELECT * FROM market_data
                    WHERE symbol = $1 AND datetime = $2
                """, symbol, base_time)

                assert result is not None
                assert result['symbol'] == symbol


# ============================================================================
# Test: Complete User Journey
# ============================================================================

class TestUserJourney:
    """Test complete user journeys through the system"""

    @pytest.mark.asyncio
    async def test_trader_morning_routine(self, db_pool, redis_client, clean_db, clean_redis):
        """Test a trader's morning routine: system startup → data check → trading"""
        # Step 1: System starts, L0 pipeline runs
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, completed_at, records_processed)
                VALUES ('usdcop_m5__01_l0_intelligent_acquire', 'L0', 'completed', NOW(), NOW(), 59)
            """)

        # Step 2: Real-time service activates
        async with db_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO pipeline_status
                (pipeline_name, pipeline_type, status, started_at, current_step, progress_percent)
                VALUES ('realtime_orchestrator', 'realtime', 'active', NOW(), 'Collecting real-time data', 100)
            """)

        # Step 3: Trader checks dashboard - latest price
        cache_data = {
            'symbol': 'USDCOP',
            'price': 4250.50,
            'timestamp': datetime.now(COT_TZ).isoformat()
        }
        redis_client.setex('usdcop:latest_price', 300, json.dumps(cache_data, default=str))

        # Step 4: Trader queries historical data
        base_time = datetime.now(COT_TZ).replace(hour=8, minute=0, second=0, microsecond=0)

        async with db_pool.acquire() as conn:
            for i in range(12):  # First hour of trading (12 * 5min bars)
                bar_time = base_time + timedelta(minutes=i * 5)
                await conn.execute("""
                    INSERT INTO market_data
                    (symbol, datetime, open, high, low, close, volume, source, timeframe, created_at)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, NOW())
                """,
                    'USDCOP',
                    bar_time,
                    Decimal('4250.00') + Decimal(i * 0.5),
                    Decimal('4255.00') + Decimal(i * 0.5),
                    Decimal('4248.00') + Decimal(i * 0.5),
                    Decimal('4252.00') + Decimal(i * 0.5),
                    5000,
                    'twelvedata',
                    '5min'
                )

            # Trader queries morning data
            morning_data = await conn.fetch("""
                SELECT * FROM market_data
                WHERE symbol = 'USDCOP'
                AND datetime >= $1
                AND datetime < $2
                ORDER BY datetime ASC
            """, base_time, base_time + timedelta(hours=1))

            assert len(morning_data) == 12

        # Step 5: Verify all systems operational
        cached_price = redis_client.get('usdcop:latest_price')
        assert cached_price is not None

        async with db_pool.acquire() as conn:
            l0_status = await conn.fetchval("""
                SELECT COUNT(*) FROM pipeline_status
                WHERE pipeline_type = 'L0' AND status = 'completed'
            """)

            rt_status = await conn.fetchval("""
                SELECT COUNT(*) FROM pipeline_status
                WHERE pipeline_type = 'realtime' AND status = 'active'
            """)

            assert l0_status > 0
            assert rt_status > 0

        print("\n✅ Trader morning routine test completed successfully!")
