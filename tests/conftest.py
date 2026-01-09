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
# Feature Builder Testing Fixtures
# ============================================================================

@pytest.fixture
def feature_calculator():
    """Feature calculator instance for testing"""
    import sys
    from pathlib import Path

    # Add src to path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))

    from core.services.feature_builder import FeatureBuilder
    return FeatureBuilder()


@pytest.fixture
def sample_ohlcv_df():
    """Sample OHLCV dataframe with 100 bars from fixtures"""
    import pandas as pd
    from pathlib import Path

    fixtures_path = Path(__file__).parent / 'fixtures' / 'sample_ohlcv.csv'
    if fixtures_path.exists():
        return pd.read_csv(fixtures_path, parse_dates=['time'])

    # Fallback: create minimal sample data
    return pd.DataFrame({
        'time': pd.date_range('2020-03-02 13:00:00', periods=100, freq='5min'),
        'open': [3520.0 + i * 0.5 for i in range(100)],
        'high': [3521.0 + i * 0.5 for i in range(100)],
        'low': [3519.0 + i * 0.5 for i in range(100)],
        'close': [3520.5 + i * 0.5 for i in range(100)],
    })


@pytest.fixture
def sample_macro_df():
    """Sample macro dataframe with 10 days from fixtures"""
    import pandas as pd
    from pathlib import Path

    fixtures_path = Path(__file__).parent / 'fixtures' / 'sample_macro.csv'
    if fixtures_path.exists():
        return pd.read_csv(fixtures_path, parse_dates=['date'])

    # Fallback: create minimal sample data
    return pd.DataFrame({
        'date': pd.date_range('2020-03-02', periods=10, freq='D'),
        'dxy': [103.0] * 10,
        'vix': [20.0] * 10,
        'embi': [300.0] * 10,
        'brent': [50.0] * 10,
        'treasury_2y': [1.0] * 10,
        'treasury_10y': [1.5] * 10,
        'usdmxn': [19.5] * 10,
    })


@pytest.fixture
def legacy_features_df():
    """Legacy features from RL_DS3_MACRO_CORE.csv for parity testing"""
    import pandas as pd
    from pathlib import Path

    legacy_path = Path(__file__).parent.parent / 'data' / 'archive' / 'PASS' / 'OUTPUT_RL' / 'RL_DS3_MACRO_CORE.csv'
    if legacy_path.exists():
        return pd.read_csv(legacy_path, parse_dates=['timestamp'], nrows=1000)

    pytest.skip("Legacy dataset not available for parity testing")


@pytest.fixture
def feature_config():
    """Feature configuration from feature_config.json"""
    import json
    from pathlib import Path

    config_path = Path(__file__).parent.parent / 'config' / 'feature_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Marker Registration
# ============================================================================

# ============================================================================
# API Client Fixtures
# ============================================================================

@pytest.fixture
def trading_api_base_url():
    """Trading API base URL"""
    return os.getenv('TRADING_API_URL', 'http://localhost:8001')


@pytest.fixture
def pipeline_api_base_url():
    """Pipeline Data API base URL"""
    return os.getenv('PIPELINE_API_URL', 'http://localhost:8002')


@pytest.fixture
def analytics_api_base_url():
    """Trading Analytics API base URL"""
    return os.getenv('ANALYTICS_API_URL', 'http://localhost:8003')


@pytest.fixture
def api_timeout():
    """Default API timeout in seconds"""
    return 30


@pytest.fixture
def api_headers():
    """Default API headers"""
    return {
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    }


# ============================================================================
# Model Fixtures
# ============================================================================

@pytest.fixture
def model_path():
    """Path to trained PPO model"""
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / 'models'

    # Try v14 first, fallback to v11
    for version in ['v14', 'v11']:
        model_file = models_dir / f'ppo_usdcop_{version}_fold0.zip'
        if model_file.exists():
            return str(model_file)

    return None


@pytest.fixture
def loaded_model(model_path):
    """Load PPO model for testing"""
    if model_path is None:
        pytest.skip("No model file available")

    try:
        from stable_baselines3 import PPO
        return PPO.load(model_path)
    except ImportError:
        pytest.skip("stable-baselines3 not installed")


@pytest.fixture
def observation_space_dim():
    """Expected observation space dimension"""
    return 15


@pytest.fixture
def action_space_dim():
    """Expected action space dimension"""
    return 1


# ============================================================================
# Trading Configuration Fixtures
# ============================================================================

@pytest.fixture
def trading_config():
    """Trading configuration for tests"""
    return {
        'symbol': 'USD/COP',
        'timeframe': '5min',
        'cost_per_trade': 0.0015,
        'weak_signal_threshold': 0.3,
        'market_hours': {
            'start': 8,
            'end': 12,
            'end_minute': 55,
            'timezone': 'America/Bogota'
        },
        'bars_per_session': 60
    }


@pytest.fixture
def production_gates():
    """Production deployment gate thresholds"""
    return {
        'sharpe_ratio': 0.0,
        'win_rate': 0.35,
        'max_drawdown': -0.20,
        'profit_factor': 1.0,
        'calmar_ratio': 0.0
    }


@pytest.fixture
def norm_stats():
    """Feature normalization statistics"""
    return {
        'log_ret_5m': {'mean': 2.0e-06, 'std': 0.001138},
        'log_ret_1h': {'mean': 2.3e-05, 'std': 0.003776},
        'log_ret_4h': {'mean': 5.2e-05, 'std': 0.007768},
        'rsi_9': {'mean': 49.27, 'std': 23.07},
        'atr_pct': {'mean': 0.062, 'std': 0.0446},
        'adx_14': {'mean': 32.01, 'std': 16.36},
        'dxy_z': {'mean': 103.0, 'std': 5.0},
        'vix_z': {'mean': 20.0, 'std': 10.0},
        'embi_z': {'mean': 300.0, 'std': 100.0},
        'rate_spread': {'mean': -0.0326, 'std': 1.400}
    }


# ============================================================================
# Sample Data Fixtures for API Testing
# ============================================================================

@pytest.fixture
def sample_inference_request():
    """Sample inference request payload"""
    return {
        'observation': [0.1, 0.2, -0.1, 0.5, 0.3, 0.4, -0.2, 0.1,
                       0.6, -0.3, 0.2, 0.1, -0.1, 0.5, 0.5],
        'position': 0.5,
        'step': 30
    }


@pytest.fixture
def sample_candlestick_request():
    """Sample candlestick data request"""
    return {
        'symbol': 'USDCOP',
        'timeframe': '5min',
        'start_date': '2025-12-16',
        'end_date': '2025-12-17',
        'limit': 100
    }


@pytest.fixture
def sample_backtest_config():
    """Sample backtest configuration"""
    return {
        'model_id': 'ppo_usdcop_v14_fold0',
        'start_date': '2025-11-01',
        'end_date': '2025-12-17',
        'initial_capital': 100000,
        'position_size': 0.1,
        'cost_per_trade': 0.0015
    }


# ============================================================================
# Mock API Response Fixtures
# ============================================================================

@pytest.fixture
def mock_health_response():
    """Mock health check response"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now(COT_TZ).isoformat(),
        'version': '3.0.0',
        'database': 'connected'
    }


@pytest.fixture
def mock_market_status():
    """Mock market status response"""
    return {
        'is_open': True,
        'current_time': datetime.now(COT_TZ).isoformat(),
        'timezone': 'America/Bogota',
        'trading_hours': '08:00 - 12:55 COT',
        'trading_days': 'Monday - Friday'
    }


@pytest.fixture
def mock_inference_response():
    """Mock inference API response"""
    return {
        'action': 0.75,
        'position_delta': 0.25,
        'confidence': 0.85,
        'signal_strength': 'strong',
        'timestamp': datetime.now(COT_TZ).isoformat()
    }


# ============================================================================
# Multi-Model API Testing Fixtures
# ============================================================================

@pytest.fixture
def multi_model_api_base_url():
    """Multi-model trading API base URL"""
    return os.getenv('MULTI_MODEL_API_URL', 'http://localhost:8006')


@pytest.fixture
async def async_http_client():
    """Async HTTP client for API testing"""
    try:
        from httpx import AsyncClient
        async with AsyncClient(timeout=30.0) as client:
            yield client
    except ImportError:
        pytest.skip("httpx not installed")


@pytest.fixture
def model_registry():
    """Registry of available models for testing"""
    from pathlib import Path
    models_dir = Path(__file__).parent.parent / 'models'

    return {
        'ppo_v19': {
            'model_id': 'ppo_v19',
            'model_name': 'PPO USD/COP V19',
            'model_type': 'RL',
            'framework': 'stable-baselines3',
            'algorithm': 'PPO',
            'version': 'v19',
            'file_path': str(models_dir / 'ppo_usdcop_v19_fold0.zip'),
            'observation_dim': 15,
            'action_space': 'Box(-1, 1)',
            'is_active': True
        },
        'ppo_v15_fold3': {
            'model_id': 'ppo_v15_fold3',
            'model_name': 'PPO USD/COP V15 Fold3',
            'model_type': 'RL',
            'framework': 'stable-baselines3',
            'algorithm': 'PPO',
            'version': 'v15',
            'file_path': str(models_dir / 'ppo_usdcop_v15_fold3.zip'),
            'observation_dim': 15,
            'action_space': 'Box(-1, 1)',
            'is_active': True
        }
    }


@pytest.fixture
def sample_observation():
    """Sample 15-dim observation vector for inference testing"""
    import numpy as np
    return np.array([
        0.1,    # log_ret_5m
        0.2,    # log_ret_1h
        0.15,   # log_ret_4h
        0.3,    # rsi_9 (normalized)
        0.1,    # atr_pct (normalized)
        0.2,    # adx_14 (normalized)
        -0.5,   # dxy_z
        0.01,   # dxy_change_1d
        0.3,    # vix_z
        0.1,    # embi_z
        0.02,   # brent_change_1d
        -0.5,   # rate_spread (normalized)
        0.003,  # usdmxn_change_1d
        0.5,    # position
        0.5     # time_normalized
    ], dtype=np.float32)


@pytest.fixture
def feature_order():
    """Expected feature order for observations"""
    return [
        'log_ret_5m',
        'log_ret_1h',
        'log_ret_4h',
        'rsi_9',
        'atr_pct',
        'adx_14',
        'dxy_z',
        'dxy_change_1d',
        'vix_z',
        'embi_z',
        'brent_change_1d',
        'rate_spread',
        'usdmxn_change_1d'
    ]


@pytest.fixture
def sample_trading_signal():
    """Sample trading signal for stream testing"""
    from datetime import timezone
    return {
        'model_id': 'ppo_v19',
        'action': 0.75,
        'signal': 'buy',
        'confidence': 0.85,
        'position_delta': 0.25,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'bar_number': 30,
        'features_hash': 'abc123def456'
    }


@pytest.fixture
def action_thresholds():
    """Action discretization thresholds"""
    return {
        'strong_buy': 0.6,
        'buy': 0.3,
        'hold_min': -0.3,
        'hold_max': 0.3,
        'sell': -0.3,
        'strong_sell': -0.6
    }


# ============================================================================
# Database Schema Testing Fixtures
# ============================================================================

@pytest.fixture
def db_config():
    """Database connection configuration"""
    return {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': int(os.getenv('POSTGRES_PORT', '5432')),
        'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
        'user': os.getenv('POSTGRES_USER', 'admin'),
        'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
    }


@pytest.fixture
def sync_db_connection(db_config):
    """Synchronous database connection for testing"""
    try:
        import psycopg2
        conn = psycopg2.connect(**db_config)
        yield conn
        conn.close()
    except Exception:
        pytest.skip("Database not available for testing")


@pytest.fixture
def expected_dw_tables():
    """Expected tables in dw schema"""
    return [
        'dim_strategy',
        'fact_strategy_signals',
        'fact_equity_curve',
        'fact_strategy_positions',
        'fact_strategy_performance'
    ]


# ============================================================================
# Redis Stream Testing Fixtures
# ============================================================================

@pytest.fixture
def redis_stream_config():
    """Redis stream configuration for testing"""
    return {
        'maxlen': 1000,
        'consumer_group': 'test-consumers',
        'stream_prefix': 'test:signals:'
    }


@pytest.fixture
async def async_redis_client():
    """Async Redis client for testing"""
    redis_url = os.getenv('TEST_REDIS_URL', os.getenv('REDIS_URL', 'redis://localhost:6379/1'))

    try:
        from redis import asyncio as aioredis
        client = aioredis.from_url(redis_url, decode_responses=True)
        await client.ping()
        yield client
        await client.aclose()
    except Exception:
        pytest.skip("Redis not available for testing")


# ============================================================================
# Feature Builder Testing Fixtures (Extended)
# ============================================================================

@pytest.fixture
def feature_builder():
    """Feature builder instance for testing"""
    import sys
    from pathlib import Path

    # Add src to path
    src_path = Path(__file__).parent.parent / 'src'
    sys.path.insert(0, str(src_path))

    try:
        from core.services.feature_builder import FeatureBuilder
        return FeatureBuilder()
    except ImportError:
        pytest.skip("FeatureBuilder not available")


@pytest.fixture
def sample_features_dict(feature_order):
    """Sample features dictionary for observation building"""
    return {feat: 0.0 for feat in feature_order}


@pytest.fixture
def sample_ohlcv_100bars():
    """Generate 100 bars of OHLCV data"""
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    base_price = 4250.0
    n_bars = 100

    returns = np.random.normal(0, 0.001, n_bars)
    close_prices = base_price * np.exp(np.cumsum(returns))

    high_prices = close_prices * (1 + np.abs(np.random.normal(0, 0.0005, n_bars)))
    low_prices = close_prices * (1 - np.abs(np.random.normal(0, 0.0005, n_bars)))
    open_prices = close_prices * (1 + np.random.normal(0, 0.0003, n_bars))

    return pd.DataFrame({
        'time': pd.date_range('2025-12-01 08:00:00', periods=n_bars, freq='5min'),
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': np.random.randint(100, 10000, n_bars)
    })


# ============================================================================
# Pytest Configuration and Markers
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
    config.addinivalue_line(
        "markers", "api: API endpoint tests"
    )
    config.addinivalue_line(
        "markers", "asyncio: Async tests"
    )
