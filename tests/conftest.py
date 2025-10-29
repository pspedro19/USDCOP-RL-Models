"""
Pytest Configuration and Shared Fixtures
==========================================

Provides reusable fixtures for all test suites.
"""

import asyncio
import os
import pytest
import redis
import asyncpg
from datetime import datetime, time
from decimal import Decimal
from typing import Dict, Any, AsyncGenerator
import pytz

# Set test environment
os.environ['LOG_LEVEL'] = 'DEBUG'
os.environ['TESTING'] = 'true'

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

# ============================================================================
# Pytest Configuration
# ============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Mock Data Fixtures
# ============================================================================

@pytest.fixture
def sample_market_data() -> Dict[str, Any]:
    """Sample market data from TwelveData API"""
    return {
        'symbol': 'USDCOP',
        'price': 4250.50,
        'bid': 4250.00,
        'ask': 4251.00,
        'volume': 1000,
        'timestamp': '2025-10-22T10:30:00-05:00'
    }


@pytest.fixture
def sample_ohlcv_data() -> Dict[str, Any]:
    """Sample OHLCV bar data"""
    return {
        'symbol': 'USDCOP',
        'datetime': datetime.now(COT_TZ),
        'open': Decimal('4250.00'),
        'high': Decimal('4255.00'),
        'low': Decimal('4248.00'),
        'close': Decimal('4252.50'),
        'volume': 5000,
        'source': 'twelvedata',
        'timeframe': '5min'
    }


@pytest.fixture
def mock_websocket_message():
    """Mock WebSocket message from TwelveData"""
    return {
        'event': 'price',
        'symbol': 'USD/COP',
        'price': 4250.50,
        'bid': 4250.00,
        'ask': 4251.00,
        'volume': 1000,
        'timestamp': '2025-10-22T10:30:00Z'
    }


@pytest.fixture
def market_hours_config():
    """Market hours configuration for testing"""
    return {
        'timezone': 'America/Bogota',
        'start_hour': 8,
        'start_minute': 0,
        'end_hour': 12,
        'end_minute': 55,
        'weekdays_only': True
    }


# ============================================================================
# Database Fixtures
# ============================================================================

@pytest.fixture
async def db_pool(event_loop) -> AsyncGenerator[asyncpg.Pool, None]:
    """Create test database connection pool"""
    database_url = os.getenv('TEST_DATABASE_URL', os.getenv('DATABASE_URL'))

    if not database_url:
        pytest.skip("Database URL not configured for testing")

    pool = await asyncpg.create_pool(
        database_url,
        min_size=2,
        max_size=5,
        command_timeout=30
    )

    yield pool

    await pool.close()


@pytest.fixture
async def clean_db(db_pool):
    """Clean test database tables before each test"""
    async with db_pool.acquire() as conn:
        # Clean test data
        await conn.execute("DELETE FROM realtime_market_data WHERE source LIKE '%test%'")
        await conn.execute("DELETE FROM market_data WHERE source LIKE '%test%'")
        await conn.execute("DELETE FROM pipeline_status WHERE pipeline_name LIKE '%test%'")

    yield

    # Cleanup after test
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM realtime_market_data WHERE source LIKE '%test%'")
        await conn.execute("DELETE FROM market_data WHERE source LIKE '%test%'")
        await conn.execute("DELETE FROM pipeline_status WHERE pipeline_name LIKE '%test%'")


# ============================================================================
# Redis Fixtures
# ============================================================================

@pytest.fixture
def redis_client() -> redis.Redis:
    """Create Redis client for testing"""
    redis_url = os.getenv('TEST_REDIS_URL', os.getenv('REDIS_URL', 'redis://localhost:6379/1'))

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
def clean_redis(redis_client):
    """Clean Redis test keys before each test"""
    test_keys = redis_client.keys('test:*') + redis_client.keys('usdcop:test:*')
    if test_keys:
        redis_client.delete(*test_keys)

    yield

    # Cleanup after test
    test_keys = redis_client.keys('test:*') + redis_client.keys('usdcop:test:*')
    if test_keys:
        redis_client.delete(*test_keys)


# ============================================================================
# Mock API Fixtures
# ============================================================================

@pytest.fixture
def mock_twelvedata_api(mocker):
    """Mock TwelveData API responses"""
    mock_response = {
        'meta': {
            'symbol': 'USD/COP',
            'interval': '5min',
            'currency': 'COP'
        },
        'values': [
            {
                'datetime': '2025-10-22 10:30:00',
                'open': '4250.00',
                'high': '4255.00',
                'low': '4248.00',
                'close': '4252.50',
                'volume': '5000'
            }
        ],
        'status': 'ok'
    }

    return mocker.Mock(return_value=mock_response)


# ============================================================================
# Circuit Breaker Fixtures
# ============================================================================

@pytest.fixture
def circuit_breaker_states():
    """Circuit breaker state configurations"""
    return {
        'closed': {'failures': 0, 'state': 'closed', 'last_failure': None},
        'open': {'failures': 5, 'state': 'open', 'last_failure': datetime.now(COT_TZ)},
        'half_open': {'failures': 3, 'state': 'half_open', 'last_failure': datetime.now(COT_TZ)}
    }


# ============================================================================
# Leader Election Fixtures
# ============================================================================

@pytest.fixture
def leader_election_config():
    """Leader election configuration"""
    return {
        'lease_duration': 30,
        'renewal_interval': 10,
        'node_id': 'test-node-1',
        'election_key': 'test:leader:election'
    }


# ============================================================================
# Performance Metrics Fixtures
# ============================================================================

@pytest.fixture
def performance_thresholds():
    """Performance testing thresholds"""
    return {
        'max_latency_ms': 100,  # Max latency for data processing
        'min_throughput': 100,   # Min messages per second
        'max_memory_mb': 512,    # Max memory usage
        'max_cpu_percent': 80    # Max CPU usage
    }


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def generate_market_ticks():
    """Generate multiple market ticks for testing"""
    def _generate(count: int = 10, base_price: float = 4250.00):
        ticks = []
        for i in range(count):
            price = base_price + (i * 0.5)
            ticks.append({
                'symbol': 'USDCOP',
                'price': price,
                'bid': price - 0.5,
                'ask': price + 0.5,
                'volume': 1000 + (i * 100),
                'timestamp': datetime.now(COT_TZ).isoformat()
            })
        return ticks

    return _generate


# ============================================================================
# Marker Registration
# ============================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests with mocks"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests requiring DB/Redis"
    )
    config.addinivalue_line(
        "markers", "load: Load and performance tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end pipeline tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow running tests"
    )
