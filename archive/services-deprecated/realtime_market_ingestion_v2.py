"""
USDCOP Real-time Market Ingestion Service V2
============================================

Production-ready service with:
- Circuit breaker for fault tolerance
- Async Redis with connection pooling
- Leader election using Redis locks
- Intelligent ON CONFLICT resolution for OHLCV data
- Prometheus metrics
- Structured logging
- Graceful shutdown

Author: USDCOP Trading System
Version: 2.0.0
"""

import asyncio
import json
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional

import pytz

from redis import asyncio as aioredis
import asyncpg
import httpx
from circuitbreaker import CircuitBreaker, CircuitBreakerError
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

# ===========================
# Configuration & Logging
# ===========================

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================

class Config:
    """Service configuration from environment variables"""

    # Database
    DATABASE_URL = os.getenv(
        'DATABASE_URL',
        'postgresql://admin:admin123@postgres:5432/usdcop_trading'
    )

    # Redis
    REDIS_URL = os.getenv('REDIS_URL', 'redis://:redis123@redis:6379/0')

    # TwelveData API
    TWELVEDATA_API_KEY = os.getenv('TWELVEDATA_API_KEY_1', '')
    TWELVEDATA_BASE_URL = os.getenv(
        'TWELVEDATA_BASE_URL',
        'https://api.twelvedata.com'
    )

    # Service Configuration
    SYMBOL = os.getenv('SYMBOL', 'USD/COP')
    INTERVAL = os.getenv('INTERVAL', '5min')
    POLLING_INTERVAL = int(os.getenv('POLLING_INTERVAL', '300'))  # 300 seconds (5 minutes)
    TIMEZONE = os.getenv('TIMEZONE', 'America/Bogota')

    # Circuit Breaker
    FAILURE_THRESHOLD = int(os.getenv('FAILURE_THRESHOLD', '5'))
    RECOVERY_TIMEOUT = int(os.getenv('RECOVERY_TIMEOUT', '300'))  # 5 minutes
    EXPECTED_EXCEPTION = Exception

    # Leader Election
    LEADER_LOCK_KEY = 'realtime_ingestion:leader_lock'
    LEADER_LOCK_TTL = int(os.getenv('LEADER_LOCK_TTL', '360'))  # 360 seconds (6 minutes)
    LEADER_RENEWAL_INTERVAL = int(os.getenv('LEADER_RENEWAL_INTERVAL', '120'))  # 120 seconds (2 minutes)

    # Redis Cache
    CACHE_TTL = int(os.getenv('CACHE_TTL', '300'))  # 5 minutes

    # HTTP Client
    HTTP_TIMEOUT = int(os.getenv('HTTP_TIMEOUT', '30'))

    # Service
    PORT = int(os.getenv('PORT', '8087'))


# ===========================
# Prometheus Metrics
# ===========================

# Counters
fetch_counter = Counter(
    'market_data_fetch_total',
    'Total number of market data fetch attempts',
    ['status']
)
error_counter = Counter(
    'market_data_errors_total',
    'Total number of errors',
    ['error_type']
)
insert_counter = Counter(
    'market_data_inserts_total',
    'Total number of database inserts',
    ['operation']
)

# Histograms
fetch_latency = Histogram(
    'market_data_fetch_duration_seconds',
    'Time spent fetching market data'
)
db_latency = Histogram(
    'database_operation_duration_seconds',
    'Time spent on database operations',
    ['operation']
)

# Gauges
is_leader = Gauge(
    'service_is_leader',
    'Whether this instance is the leader (1) or follower (0)'
)
circuit_breaker_state = Gauge(
    'circuit_breaker_state',
    'Circuit breaker state: 0=closed, 1=open, 2=half-open'
)
last_successful_fetch = Gauge(
    'last_successful_fetch_timestamp',
    'Unix timestamp of last successful data fetch'
)


# ===========================
# Data Models
# ===========================

class OHLCVData(BaseModel):
    """OHLCV data model"""
    time: datetime
    symbol: str = Field(default='USD/COP')
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Optional[int] = 0
    source: str = Field(default='twelvedata')


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    timestamp: str
    is_leader: bool
    circuit_breaker_state: str
    database_connected: bool
    redis_connected: bool
    last_fetch_time: Optional[str] = None


# ===========================
# Market Data Fetcher
# ===========================

class MarketDataFetcher:
    """Fetches market data from TwelveData API with circuit breaker"""

    def __init__(self, config: Config):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.FAILURE_THRESHOLD,
            recovery_timeout=config.RECOVERY_TIMEOUT,
            expected_exception=config.EXPECTED_EXCEPTION
        )

    async def initialize(self):
        """Initialize HTTP client"""
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.HTTP_TIMEOUT),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        logger.info("Market data fetcher initialized")

    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            logger.info("Market data fetcher closed")

    @property
    def circuit_state(self) -> str:
        """Get circuit breaker state"""
        if self._circuit_breaker._state == 0:
            return "closed"
        elif self._circuit_breaker._state == 1:
            return "open"
        else:
            return "half_open"

    async def fetch_latest_data(self) -> Optional[OHLCVData]:
        """
        Fetch latest market data from TwelveData API
        Uses circuit breaker pattern for fault tolerance
        """
        try:
            return await self._fetch_with_circuit_breaker()
        except CircuitBreakerError:
            logger.warning("Circuit breaker is OPEN - using cached data")
            error_counter.labels(error_type='circuit_breaker_open').inc()
            circuit_breaker_state.set(1)
            return None
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            error_counter.labels(error_type='fetch_error').inc()
            return None

    @CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=300,
        expected_exception=Exception
    )
    async def _fetch_with_circuit_breaker(self) -> Optional[OHLCVData]:
        """Internal method wrapped with circuit breaker"""

        # Update circuit breaker state metric
        state_map = {'closed': 0, 'open': 1, 'half_open': 2}
        circuit_breaker_state.set(state_map.get(self.circuit_state, 0))

        start_time = datetime.now()

        url = f"{self.config.TWELVEDATA_BASE_URL}/time_series"
        params = {
            'symbol': self.config.SYMBOL,
            'interval': self.config.INTERVAL,
            'outputsize': 1,
            'apikey': self.config.TWELVEDATA_API_KEY,
            'format': 'JSON',
            'timezone': self.config.TIMEZONE
        }

        logger.debug(f"Fetching market data: {url} with params {params}")

        response = await self.client.get(url, params=params)

        # Track latency
        latency = (datetime.now() - start_time).total_seconds()
        fetch_latency.observe(latency)

        if response.status_code != 200:
            fetch_counter.labels(status='error').inc()
            raise Exception(f"API returned status {response.status_code}: {response.text}")

        data = response.json()

        # Handle API errors
        if 'status' in data and data['status'] == 'error':
            fetch_counter.labels(status='api_error').inc()
            raise Exception(f"API error: {data.get('message', 'Unknown error')}")

        if 'values' not in data or not data['values']:
            fetch_counter.labels(status='no_data').inc()
            raise Exception("No data returned from API")

        # Parse OHLCV data
        latest = data['values'][0]

        # Parse datetime and ensure it's timezone-aware in COT
        dt_str = latest['datetime']
        dt_parsed = datetime.fromisoformat(dt_str)

        # If datetime is naive (no timezone), localize to COT
        if dt_parsed.tzinfo is None:
            cot_tz = pytz.timezone(self.config.TIMEZONE)
            dt_aware = cot_tz.localize(dt_parsed)
        else:
            # If it has timezone, convert to COT
            cot_tz = pytz.timezone(self.config.TIMEZONE)
            dt_aware = dt_parsed.astimezone(cot_tz)

        ohlcv = OHLCVData(
            time=dt_aware,
            symbol=self.config.SYMBOL,
            open=Decimal(str(latest['open'])),
            high=Decimal(str(latest['high'])),
            low=Decimal(str(latest['low'])),
            close=Decimal(str(latest['close'])),
            volume=int(latest.get('volume', 0)),
            source='twelvedata'
        )

        fetch_counter.labels(status='success').inc()
        last_successful_fetch.set(datetime.now().timestamp())

        logger.info(f"Fetched OHLCV: {ohlcv.symbol} @ {ohlcv.time} | C: {ohlcv.close}")

        return ohlcv


# ===========================
# Database Manager
# ===========================

class DatabaseManager:
    """Manages PostgreSQL connection and operations"""

    def __init__(self, config: Config):
        self.config = config
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            self.config.DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )

        # Ensure OHLCV table exists
        await self._create_ohlcv_table()

        logger.info("Database connection pool initialized")

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")

    async def _create_ohlcv_table(self):
        """Create OHLCV table if it doesn't exist"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS usdcop_m5_ohlcv (
                    time TIMESTAMPTZ NOT NULL,
                    symbol TEXT NOT NULL,
                    open NUMERIC(12,6) NOT NULL,
                    high NUMERIC(12,6) NOT NULL,
                    low NUMERIC(12,6) NOT NULL,
                    close NUMERIC(12,6) NOT NULL,
                    volume BIGINT DEFAULT 0,
                    source TEXT DEFAULT 'twelvedata',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (time, symbol)
                );

                -- Create hypertable if TimescaleDB is available
                SELECT create_hypertable('usdcop_m5_ohlcv', 'time',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );

                -- Create index on symbol and time
                CREATE INDEX IF NOT EXISTS idx_usdcop_m5_ohlcv_symbol_time
                ON usdcop_m5_ohlcv (symbol, time DESC);
            """)
            logger.info("OHLCV table schema verified")

    async def insert_ohlcv(self, ohlcv: OHLCVData) -> bool:
        """
        Insert OHLCV data with intelligent conflict resolution

        ON CONFLICT strategy:
        - GREATEST for high (keep maximum)
        - LEAST for low (keep minimum)
        - ALWAYS update close (latest close)
        - SUM volume (aggregate)
        - Preserve source if 'twelvedata'
        """
        start_time = datetime.now()

        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO usdcop_m5_ohlcv (
                        time, symbol, open, high, low, close, volume, source
                    )
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    ON CONFLICT (time, symbol)
                    DO UPDATE SET
                        high = GREATEST(usdcop_m5_ohlcv.high, EXCLUDED.high),
                        low = LEAST(usdcop_m5_ohlcv.low, EXCLUDED.low),
                        close = EXCLUDED.close,
                        volume = usdcop_m5_ohlcv.volume + EXCLUDED.volume,
                        source = CASE
                            WHEN EXCLUDED.source = 'twelvedata' THEN EXCLUDED.source
                            ELSE usdcop_m5_ohlcv.source
                        END,
                        updated_at = NOW()
                """,
                    ohlcv.time,
                    ohlcv.symbol,
                    ohlcv.open,
                    ohlcv.high,
                    ohlcv.low,
                    ohlcv.close,
                    ohlcv.volume,
                    ohlcv.source
                )

            # Track metrics
            latency = (datetime.now() - start_time).total_seconds()
            db_latency.labels(operation='insert_ohlcv').observe(latency)
            insert_counter.labels(operation='upsert').inc()

            logger.debug(f"Inserted/Updated OHLCV for {ohlcv.symbol} @ {ohlcv.time}")
            return True

        except Exception as e:
            logger.error(f"Database error inserting OHLCV: {e}")
            error_counter.labels(error_type='database_error').inc()
            return False

    async def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False


# ===========================
# Redis Manager (Async)
# ===========================

class RedisManager:
    """Manages async Redis connection, caching, and pub/sub"""

    def __init__(self, config: Config):
        self.config = config
        self.redis: Optional[aioredis.Redis] = None
        self.leader_lock_token: Optional[str] = None

    async def initialize(self):
        """Initialize Redis connection pool"""
        self.redis = await aioredis.from_url(
            self.config.REDIS_URL,
            encoding='utf-8',
            decode_responses=True,
            max_connections=10
        )
        logger.info("Redis connection pool initialized")

    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
            logger.info("Redis connection closed")

    async def health_check(self) -> bool:
        """Check Redis connectivity"""
        try:
            await self.redis.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    async def cache_latest_data(self, ohlcv: OHLCVData):
        """Cache latest OHLCV data in Redis"""
        try:
            cache_key = f"market_data:latest:{ohlcv.symbol}"
            cache_data = {
                'time': ohlcv.time.isoformat(),
                'symbol': ohlcv.symbol,
                'open': str(ohlcv.open),
                'high': str(ohlcv.high),
                'low': str(ohlcv.low),
                'close': str(ohlcv.close),
                'volume': ohlcv.volume,
                'source': ohlcv.source
            }

            await self.redis.setex(
                cache_key,
                self.config.CACHE_TTL,
                json.dumps(cache_data)
            )

            logger.debug(f"Cached latest data for {ohlcv.symbol}")

        except Exception as e:
            logger.error(f"Error caching data: {e}")
            error_counter.labels(error_type='cache_error').inc()

    async def publish_to_stream(self, ohlcv: OHLCVData):
        """Publish OHLCV data to Redis stream (not pub/sub)"""
        try:
            stream_key = f"market_data:stream:{ohlcv.symbol}"

            message = {
                'time': ohlcv.time.isoformat(),
                'symbol': ohlcv.symbol,
                'open': str(ohlcv.open),
                'high': str(ohlcv.high),
                'low': str(ohlcv.low),
                'close': str(ohlcv.close),
                'volume': str(ohlcv.volume),
                'source': ohlcv.source
            }

            await self.redis.xadd(stream_key, message)

            logger.debug(f"Published to Redis stream: {stream_key}")

        except Exception as e:
            logger.error(f"Error publishing to stream: {e}")
            error_counter.labels(error_type='publish_error').inc()

    async def get_cached_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached OHLCV data from Redis"""
        try:
            cache_key = f"market_data:latest:{symbol}"
            cached = await self.redis.get(cache_key)

            if cached:
                return json.loads(cached)
            return None

        except Exception as e:
            logger.error(f"Error getting cached data: {e}")
            return None

    async def acquire_leader_lock(self) -> bool:
        """
        Acquire leader lock using Redis SET NX
        Returns True if lock acquired, False otherwise
        """
        try:
            import uuid

            # Generate unique token for this instance
            if not self.leader_lock_token:
                self.leader_lock_token = str(uuid.uuid4())

            # Try to acquire lock with SET NX (set if not exists)
            acquired = await self.redis.set(
                self.config.LEADER_LOCK_KEY,
                self.leader_lock_token,
                nx=True,  # Only set if key doesn't exist
                ex=self.config.LEADER_LOCK_TTL  # Expire after TTL seconds
            )

            if acquired:
                is_leader.set(1)
                logger.info(f"✓ Leader lock ACQUIRED (token: {self.leader_lock_token[:8]}...)")
                return True
            else:
                is_leader.set(0)
                return False

        except Exception as e:
            logger.error(f"Error acquiring leader lock: {e}")
            is_leader.set(0)
            return False

    async def renew_leader_lock(self) -> bool:
        """
        Renew leader lock if still held by this instance
        """
        try:
            if not self.leader_lock_token:
                return False

            # Check if we still own the lock
            current_token = await self.redis.get(self.config.LEADER_LOCK_KEY)

            if current_token == self.leader_lock_token:
                # Renew the lock
                await self.redis.expire(
                    self.config.LEADER_LOCK_KEY,
                    self.config.LEADER_LOCK_TTL
                )
                logger.debug("Leader lock renewed")
                is_leader.set(1)
                return True
            else:
                # We lost the lock
                logger.warning("Lost leader lock - another instance took over")
                is_leader.set(0)
                self.leader_lock_token = None
                return False

        except Exception as e:
            logger.error(f"Error renewing leader lock: {e}")
            is_leader.set(0)
            return False

    async def release_leader_lock(self):
        """Release leader lock if held by this instance"""
        try:
            if not self.leader_lock_token:
                return

            # Only delete if we still own it
            current_token = await self.redis.get(self.config.LEADER_LOCK_KEY)

            if current_token == self.leader_lock_token:
                await self.redis.delete(self.config.LEADER_LOCK_KEY)
                logger.info("Leader lock released")

            self.leader_lock_token = None
            is_leader.set(0)

        except Exception as e:
            logger.error(f"Error releasing leader lock: {e}")


# ===========================
# Ingestion Orchestrator
# ===========================

class IngestionOrchestrator:
    """Orchestrates the entire ingestion process"""

    def __init__(self, config: Config):
        self.config = config
        self.fetcher = MarketDataFetcher(config)
        self.db_manager = DatabaseManager(config)
        self.redis_manager = RedisManager(config)
        self.running = False
        self.is_leader_instance = False

    async def initialize(self):
        """Initialize all components"""
        await self.fetcher.initialize()
        await self.db_manager.initialize()
        await self.redis_manager.initialize()
        logger.info("Ingestion orchestrator initialized")

    async def close(self):
        """Close all components"""
        await self.redis_manager.release_leader_lock()
        await self.fetcher.close()
        await self.db_manager.close()
        await self.redis_manager.close()
        logger.info("Ingestion orchestrator closed")

    async def health_status(self) -> HealthResponse:
        """Get detailed health status"""
        db_healthy = await self.db_manager.health_check()
        redis_healthy = await self.redis_manager.health_check()

        last_fetch = last_successful_fetch._value._value
        last_fetch_time = None
        if last_fetch > 0:
            last_fetch_time = datetime.fromtimestamp(last_fetch).isoformat()

        overall_status = "healthy" if (db_healthy and redis_healthy) else "degraded"

        return HealthResponse(
            status=overall_status,
            service="realtime-market-ingestion-v2",
            timestamp=datetime.now().isoformat(),
            is_leader=self.is_leader_instance,
            circuit_breaker_state=self.fetcher.circuit_state,
            database_connected=db_healthy,
            redis_connected=redis_healthy,
            last_fetch_time=last_fetch_time
        )

    async def leader_election_loop(self):
        """Background task for leader election and lock renewal"""
        while self.running:
            try:
                if self.is_leader_instance:
                    # Renew existing lock
                    renewed = await self.redis_manager.renew_leader_lock()
                    if not renewed:
                        logger.warning("Failed to renew leader lock - lost leadership")
                        self.is_leader_instance = False
                else:
                    # Try to acquire lock
                    acquired = await self.redis_manager.acquire_leader_lock()
                    if acquired:
                        self.is_leader_instance = True

                # Wait before next renewal/acquisition attempt
                await asyncio.sleep(self.config.LEADER_RENEWAL_INTERVAL)

            except Exception as e:
                logger.error(f"Error in leader election loop: {e}")
                await asyncio.sleep(10)

    def is_market_open(self) -> bool:
        """
        Check if Colombian market is open for trading

        Trading hours: Monday-Friday, 8:00 AM - 12:55 PM COT (America/Bogota)
        """
        tz = pytz.timezone(self.config.TIMEZONE)
        now = datetime.now(tz)

        # Check if it's a weekend
        if now.weekday() > 4:  # Saturday (5) or Sunday (6)
            logger.debug(f"Market closed - Weekend (day {now.weekday()})")
            return False

        # Check trading hours (8:00 - 12:55 COT)
        hour = now.hour
        minute = now.minute

        # Before 8:00 AM
        if hour < 8:
            logger.debug(f"Market closed - Before opening (current: {hour:02d}:{minute:02d})")
            return False

        # After 12:55 PM
        if hour > 12:
            logger.debug(f"Market closed - After closing (current: {hour:02d}:{minute:02d})")
            return False

        if hour == 12 and minute > 55:
            logger.debug(f"Market closed - After 12:55 PM (current: {hour:02d}:{minute:02d})")
            return False

        logger.debug(f"Market open - Trading hours (current: {hour:02d}:{minute:02d})")
        return True

    def calculate_next_5min_interval(self) -> float:
        """
        Calculate seconds to wait until next 5-minute interval (aligned to :00, :05, :10, etc.)

        Examples:
        - Current: 10:03:27 → Next: 10:05:00 → Wait: 93 seconds
        - Current: 10:05:00 → Next: 10:10:00 → Wait: 300 seconds
        - Current: 10:57:30 → Next: 11:00:00 → Wait: 150 seconds
        """
        # Get current time in market timezone
        tz = pytz.timezone(self.config.TIMEZONE)
        now = datetime.now(tz)

        # Calculate minutes to next 5-min interval
        current_minute = now.minute
        next_interval_minute = ((current_minute // 5) + 1) * 5

        # If we're at minute 55-59, next interval is next hour at :00
        if next_interval_minute >= 60:
            next_time = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            next_time = now.replace(minute=next_interval_minute, second=0, microsecond=0)

        # Calculate wait time in seconds
        wait_seconds = (next_time - now).total_seconds()

        logger.info(f"⏰ Current: {now.strftime('%H:%M:%S')} → Next fetch: {next_time.strftime('%H:%M:%S')} (wait {wait_seconds:.0f}s)")

        return max(1, wait_seconds)  # Minimum 1 second

    async def ingestion_loop(self):
        """Main ingestion loop - aligned to 5-minute intervals (:00, :05, :10, etc.)"""
        logger.info("Starting ingestion loop (5-minute aligned)")
        logger.info("Trading hours: Monday-Friday, 8:00 AM - 12:55 PM COT")

        while self.running:
            try:
                # Only ingest if we're the leader
                if not self.is_leader_instance:
                    logger.debug("Not leader - standby mode")
                    await asyncio.sleep(60)  # Check every minute if still not leader
                    continue

                # Check if market is open
                if not self.is_market_open():
                    logger.info("📴 Market is CLOSED - Pausing ingestion (outside 8:00-12:55 COT)")
                    await asyncio.sleep(300)  # Check again in 5 minutes
                    continue

                logger.info("=== Starting data ingestion cycle (LEADER) ===")
                logger.info("📊 Market is OPEN - Trading hours")

                # Fetch latest market data
                ohlcv = await self.fetcher.fetch_latest_data()

                if ohlcv:
                    # Store in database
                    success = await self.db_manager.insert_ohlcv(ohlcv)

                    if success:
                        # Cache in Redis
                        await self.redis_manager.cache_latest_data(ohlcv)

                        # Publish to Redis stream
                        await self.redis_manager.publish_to_stream(ohlcv)

                        logger.info(f"✓ Ingestion successful: {ohlcv.symbol} @ {ohlcv.close}")
                    else:
                        logger.error("Failed to store data in database")
                else:
                    logger.warning("No data fetched - using cached fallback")

                # Wait until next 5-minute interval (aligned: :00, :05, :10, etc.)
                logger.info("Calculating next 5-minute interval...")
                wait_seconds = self.calculate_next_5min_interval()
                logger.info(f"Waiting {int(wait_seconds)} seconds until next 5-min interval...")
                await asyncio.sleep(wait_seconds)

            except Exception as e:
                logger.error(f"Error in ingestion loop: {e}", exc_info=True)
                error_counter.labels(error_type='ingestion_error').inc()
                await asyncio.sleep(60)

    async def start(self):
        """Start the orchestrator"""
        self.running = True

        # Start leader election and ingestion loops
        await asyncio.gather(
            self.leader_election_loop(),
            self.ingestion_loop()
        )

    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping ingestion orchestrator...")
        self.running = False
        await self.close()


# ===========================
# FastAPI Application
# ===========================

# Global orchestrator instance
orchestrator: Optional[IngestionOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global orchestrator

    # Startup
    logger.info("=== Starting Real-time Market Ingestion Service V2 ===")

    config = Config()
    orchestrator = IngestionOrchestrator(config)
    await orchestrator.initialize()

    # Start orchestrator in background
    asyncio.create_task(orchestrator.start())

    yield

    # Shutdown
    logger.info("=== Shutting down Real-time Market Ingestion Service V2 ===")
    if orchestrator:
        await orchestrator.stop()


app = FastAPI(
    title="USDCOP Real-time Market Ingestion V2",
    description="Production-ready market data ingestion with circuit breaker, leader election, and async Redis",
    version="2.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns detailed service health status
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return await orchestrator.health_status()


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/status")
async def status():
    """
    Service status endpoint
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    return JSONResponse({
        "service": "realtime-market-ingestion-v2",
        "version": "2.0.0",
        "is_leader": orchestrator.is_leader_instance,
        "circuit_breaker_state": orchestrator.fetcher.circuit_state,
        "running": orchestrator.running,
        "timestamp": datetime.now().isoformat()
    })


@app.get("/latest/{symbol}")
async def get_latest_data(symbol: str = "USD/COP"):
    """
    Get latest cached market data for symbol
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    cached_data = await orchestrator.redis_manager.get_cached_data(symbol)

    if not cached_data:
        raise HTTPException(status_code=404, detail="No cached data available")

    return JSONResponse(cached_data)


@app.post("/force-fetch")
async def force_fetch():
    """
    Force an immediate data fetch (leader only)
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    if not orchestrator.is_leader_instance:
        raise HTTPException(status_code=403, detail="Only leader can force fetch")

    ohlcv = await orchestrator.fetcher.fetch_latest_data()

    if not ohlcv:
        raise HTTPException(status_code=500, detail="Failed to fetch data")

    success = await orchestrator.db_manager.insert_ohlcv(ohlcv)

    if success:
        await orchestrator.redis_manager.cache_latest_data(ohlcv)
        await orchestrator.redis_manager.publish_to_stream(ohlcv)

        return JSONResponse({
            "status": "success",
            "data": {
                "time": ohlcv.time.isoformat(),
                "symbol": ohlcv.symbol,
                "close": str(ohlcv.close)
            }
        })
    else:
        raise HTTPException(status_code=500, detail="Failed to store data")


# ===========================
# Graceful Shutdown
# ===========================

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum} - initiating graceful shutdown")
    if orchestrator:
        asyncio.create_task(orchestrator.stop())
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ===========================
# Main Entry Point
# ===========================

if __name__ == "__main__":
    import uvicorn

    config = Config()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=True
    )
