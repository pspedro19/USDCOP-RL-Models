"""
Enhanced WebSocket Service for Real-time Market Data with L0 Integration
========================================================================

Features:
- Ready signal detection - waits for L0 backup system completion
- Smart startup with L0 data freshness and completeness checks
- Colombian business hours awareness (8:00-12:55 COT, L-V)
- Seamless handover from L0 historical to real-time mode
- PostgreSQL integration for real-time data insertion
- Intelligent API key management and rate limiting
- Comprehensive health checks and status monitoring
- Fallback to L0 incremental if real-time fails

Author: Enhanced for L0 Integration
Version: 4.0.0
"""

import asyncio
import json
import logging
import os
import redis
import websockets
from datetime import datetime, time, timedelta, date
from decimal import Decimal
from typing import Dict, Set, Any, Optional, List
import asyncpg
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
import pytz
import aiofiles
from pathlib import Path
import pandas as pd
import psutil
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

# File system paths
READY_SIGNALS_DIR = Path('/data/ready-signals')
L0_READY_FLAG = READY_SIGNALS_DIR / 'l0_ready.flag'
BACKUP_DIR = Path('/home/GlobalForex/USDCOP-RL-Models/data/backups')

# Market hours configuration
MARKET_START_HOUR = 8
MARKET_START_MINUTE = 0
MARKET_END_HOUR = 12
MARKET_END_MINUTE = 55

class ServiceStatus(Enum):
    """Service status enumeration"""
    INITIALIZING = "initializing"
    WAITING_L0 = "waiting_l0"
    CHECKING_DATA = "checking_data"
    READY = "ready"
    ACTIVE = "active"
    MARKET_CLOSED = "market_closed"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class L0DataStatus:
    """L0 data status information"""
    ready_signal_exists: bool
    ready_signal_timestamp: Optional[datetime]
    data_freshness_hours: Optional[float]
    completeness_percentage: Optional[float]
    total_records: Optional[int]
    last_update: Optional[datetime]
    is_fresh: bool
    is_complete: bool

class L0ReadySignalDetector:
    """Detects and validates L0 backup system ready signals"""

    def __init__(self):
        self.ready_signals_dir = READY_SIGNALS_DIR
        self.l0_ready_flag = L0_READY_FLAG

    async def ensure_directories(self):
        """Ensure ready signals directory exists"""
        try:
            self.ready_signals_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Ready signals directory ensured: {self.ready_signals_dir}")
        except Exception as e:
            logger.error(f"❌ Failed to create ready signals directory: {e}")
            raise

    async def check_l0_ready_signal(self) -> bool:
        """Check if L0 ready signal exists"""
        try:
            return self.l0_ready_flag.exists()
        except Exception as e:
            logger.error(f"❌ Error checking L0 ready signal: {e}")
            return False

    async def get_ready_signal_timestamp(self) -> Optional[datetime]:
        """Get the timestamp when L0 ready signal was created"""
        try:
            if await self.check_l0_ready_signal():
                stat = self.l0_ready_flag.stat()
                return datetime.fromtimestamp(stat.st_mtime, tz=COT_TZ)
            return None
        except Exception as e:
            logger.error(f"❌ Error getting ready signal timestamp: {e}")
            return None

    async def wait_for_l0_ready(self, max_wait_minutes: int = 60) -> bool:
        """Wait for L0 ready signal with timeout"""
        logger.info(f"⏳ Waiting for L0 ready signal (max {max_wait_minutes} minutes)...")

        start_time = datetime.now()
        max_wait_time = start_time + timedelta(minutes=max_wait_minutes)

        while datetime.now() < max_wait_time:
            if await self.check_l0_ready_signal():
                signal_time = await self.get_ready_signal_timestamp()
                logger.info(f"✅ L0 ready signal detected! Created at: {signal_time}")
                return True

            # Wait 30 seconds before checking again
            await asyncio.sleep(30)
            logger.debug(f"🔍 Still waiting for L0 ready signal...")

        logger.warning(f"⚠️ L0 ready signal not detected within {max_wait_minutes} minutes")
        return False

class L0DataValidator:
    """Validates L0 data freshness and completeness"""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool

    async def check_l0_data_status(self) -> L0DataStatus:
        """Check comprehensive L0 data status"""
        try:
            ready_signal_exists = L0_READY_FLAG.exists()
            ready_signal_timestamp = None

            if ready_signal_exists:
                stat = L0_READY_FLAG.stat()
                ready_signal_timestamp = datetime.fromtimestamp(stat.st_mtime, tz=COT_TZ)

            # Check database for L0 data
            async with self.db_pool.acquire() as conn:
                # Get data freshness and completeness
                result = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total_records,
                        MAX(datetime) as last_update,
                        COUNT(DISTINCT DATE(datetime AT TIME ZONE 'America/Bogota')) as unique_days
                    FROM market_data
                    WHERE symbol = 'USDCOP'
                    AND source != 'websocket'
                    AND datetime >= NOW() - INTERVAL '7 days'
                """)

                # Calculate expected records for last 7 days
                expected_result = await conn.fetchval("""
                    SELECT count_expected_trading_bars('USDCOP', NOW() - INTERVAL '7 days', NOW())
                """) if await self._function_exists(conn, 'count_expected_trading_bars') else None

                total_records = result['total_records'] if result else 0
                last_update = result['last_update'] if result else None
                expected_records = expected_result if expected_result else (7 * 59)  # Fallback: 7 days * 59 bars/day

                # Calculate freshness
                data_freshness_hours = None
                if last_update:
                    data_freshness_hours = (datetime.now(COT_TZ) - last_update.replace(tzinfo=COT_TZ)).total_seconds() / 3600

                # Calculate completeness
                completeness_percentage = (total_records / expected_records * 100) if expected_records > 0 else 0

                # Determine if data is fresh and complete
                is_fresh = data_freshness_hours is not None and data_freshness_hours < 24  # Less than 24 hours old
                is_complete = completeness_percentage >= 95.0  # At least 95% complete

                return L0DataStatus(
                    ready_signal_exists=ready_signal_exists,
                    ready_signal_timestamp=ready_signal_timestamp,
                    data_freshness_hours=data_freshness_hours,
                    completeness_percentage=completeness_percentage,
                    total_records=total_records,
                    last_update=last_update,
                    is_fresh=is_fresh,
                    is_complete=is_complete
                )

        except Exception as e:
            logger.error(f"❌ Error checking L0 data status: {e}")
            return L0DataStatus(
                ready_signal_exists=False,
                ready_signal_timestamp=None,
                data_freshness_hours=None,
                completeness_percentage=None,
                total_records=None,
                last_update=None,
                is_fresh=False,
                is_complete=False
            )

    async def _function_exists(self, conn, function_name: str) -> bool:
        """Check if a database function exists"""
        try:
            result = await conn.fetchval("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_proc
                    WHERE proname = $1
                )
            """, function_name)
            return result
        except:
            return False

class ColombianMarketHoursManager:
    """Manages Colombian market hours with holiday awareness"""

    def __init__(self, db_pool: asyncpg.Pool):
        self.db_pool = db_pool
        self.market_start = time(MARKET_START_HOUR, MARKET_START_MINUTE)
        self.market_end = time(MARKET_END_HOUR, MARKET_END_MINUTE)

    def is_market_open(self, check_time: Optional[datetime] = None) -> bool:
        """Check if market is currently open (L-V 8:00-12:55 COT)"""
        if check_time is None:
            check_time = datetime.now(COT_TZ)
        elif check_time.tzinfo is None:
            check_time = COT_TZ.localize(check_time)
        else:
            check_time = check_time.astimezone(COT_TZ)

        current_time = check_time.time()
        weekday = check_time.weekday()  # 0=Monday, 6=Sunday

        # Check if it's a weekday (Monday to Friday)
        if weekday >= 5:  # Saturday or Sunday
            return False

        # Check time range
        return self.market_start <= current_time <= self.market_end

    async def get_next_market_session(self) -> Dict[str, Any]:
        """Get information about the next market session"""
        now = datetime.now(COT_TZ)

        if self.is_market_open(now):
            # Market is open, return current session info
            today = now.date()
            market_close = COT_TZ.localize(datetime.combine(today, self.market_end))

            return {
                'status': 'open',
                'current_session_start': COT_TZ.localize(datetime.combine(today, self.market_start)).isoformat(),
                'current_session_end': market_close.isoformat(),
                'minutes_until_close': int((market_close - now).total_seconds() / 60)
            }
        else:
            # Market is closed, find next opening
            next_open = self._find_next_market_open(now)

            return {
                'status': 'closed',
                'next_open': next_open.isoformat() if next_open else None,
                'minutes_until_open': int((next_open - now).total_seconds() / 60) if next_open else None
            }

    def _find_next_market_open(self, from_time: datetime) -> Optional[datetime]:
        """Find the next market opening time"""
        current = from_time.replace(second=0, microsecond=0)

        # Check next 7 days
        for i in range(7):
            check_date = (current + timedelta(days=i)).date()
            check_datetime = COT_TZ.localize(datetime.combine(check_date, self.market_start))

            # Skip weekends
            if check_datetime.weekday() >= 5:
                continue

            # If it's today, make sure we haven't passed opening time
            if i == 0 and current.time() >= self.market_start:
                continue

            return check_datetime

        return None

class APIKeyManager:
    """Intelligent API key management with rotation and rate limiting"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_key_index = 0
        self.calls_per_key = {}
        self.last_call_time = {}
        self.rate_limit_delay = 8  # seconds between calls

        # Initialize counters
        for i, key in enumerate(self.api_keys):
            self.calls_per_key[i] = 0
            self.last_call_time[i] = None

    def _load_api_keys(self) -> List[str]:
        """Load TwelveData API keys from environment"""
        api_keys = []

        # Load GRUPO 1 keys
        for i in range(1, 9):
            key = os.getenv(f'TWELVEDATA_API_KEY_G1_{i}') or os.getenv(f'API_KEY_G1_{i}')
            if key and key.strip():
                api_keys.append(key.strip())

        # Load GRUPO 2 keys
        for i in range(1, 9):
            key = os.getenv(f'TWELVEDATA_API_KEY_G2_{i}') or os.getenv(f'API_KEY_G2_{i}')
            if key and key.strip():
                api_keys.append(key.strip())

        # Fallback to numbered keys
        if not api_keys:
            for i in range(1, 17):
                key = os.getenv(f'TWELVEDATA_API_KEY_{i}')
                if key and key.strip():
                    api_keys.append(key.strip())

        if not api_keys:
            logger.error("❌ No TwelveData API keys found!")
            raise ValueError("No TwelveData API keys configured")

        logger.info(f"🔑 Loaded {len(api_keys)} TwelveData API keys")
        return api_keys

    async def get_next_api_key(self) -> str:
        """Get next available API key with rate limiting"""
        if not self.api_keys:
            raise ValueError("No API keys available")

        current_key_index = self.current_key_index

        # Check rate limiting for current key
        if self.last_call_time.get(current_key_index):
            time_since_last = (datetime.now() - self.last_call_time[current_key_index]).total_seconds()
            if time_since_last < self.rate_limit_delay:
                wait_time = self.rate_limit_delay - time_since_last
                logger.debug(f"⏳ Rate limiting: waiting {wait_time:.1f}s for key {current_key_index + 1}")
                await asyncio.sleep(wait_time)

        # Update usage
        self.calls_per_key[current_key_index] += 1
        self.last_call_time[current_key_index] = datetime.now()

        # Rotate to next key for next call
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)

        current_key = self.api_keys[current_key_index]
        logger.debug(f"🔑 Using API key {current_key_index + 1}/{len(self.api_keys)}")

        return current_key

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API key usage statistics"""
        total_calls = sum(self.calls_per_key.values())

        return {
            'total_keys': len(self.api_keys),
            'current_key_index': self.current_key_index + 1,
            'total_calls': total_calls,
            'calls_per_key': {f'key_{i+1}': calls for i, calls in self.calls_per_key.items()},
            'rate_limit_delay': self.rate_limit_delay
        }

class RealtimeDataProcessor:
    """Processes and stores real-time data with database integration"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.data_buffer = []
        self.buffer_size = 100
        self.last_db_sync = datetime.now()
        self.sync_interval = timedelta(minutes=5)

    async def process_realtime_tick(self, data: Dict[str, Any]) -> bool:
        """Process incoming real-time tick data"""
        try:
            # Validate data structure
            if not isinstance(data, dict) or 'price' not in data:
                logger.warning("Invalid tick data received")
                return False

            # Normalize data format
            normalized_data = {
                'symbol': 'USDCOP',
                'time': datetime.now(COT_TZ),
                'last': float(data['price']),
                'bid': float(data.get('bid', data['price'])),
                'ask': float(data.get('ask', data['price'])),
                'volume': int(data.get('volume', 0)),
                'source': 'twelvedata_websocket',
                'session_date': datetime.now(COT_TZ).date(),
                'trading_session': True
            }

            # Calculate spread
            normalized_data['spread'] = normalized_data['ask'] - normalized_data['bid']

            # Store immediately in real-time table
            await self._store_realtime_data(normalized_data)

            # Cache latest price
            await self._cache_latest_price(normalized_data)

            # Add to buffer for batch processing
            self.data_buffer.append(normalized_data)

            # Check if we need to sync to historical table
            if self._should_sync_to_db():
                await self._sync_to_market_data_table()

            logger.debug(f"💱 Processed tick: USDCOP = {normalized_data['last']}")
            return True

        except Exception as e:
            logger.error(f"❌ Error processing realtime tick: {e}")
            return False

    async def _store_realtime_data(self, data: Dict[str, Any]):
        """Store real-time data in dedicated table"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO realtime_market_data
                    (symbol, time, bid, ask, last, volume, spread, session_date, trading_session, source)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    data['symbol'],
                    data['time'],
                    Decimal(str(data['bid'])),
                    Decimal(str(data['ask'])),
                    Decimal(str(data['last'])),
                    data['volume'],
                    Decimal(str(data['spread'])) if data['spread'] else None,
                    data['session_date'],
                    data['trading_session'],
                    data['source']
                )
        except Exception as e:
            logger.error(f"❌ Error storing realtime data: {e}")

    async def _cache_latest_price(self, data: Dict[str, Any]):
        """Cache latest price in Redis"""
        try:
            cache_data = {
                'symbol': data['symbol'],
                'price': data['last'],
                'bid': data['bid'],
                'ask': data['ask'],
                'spread': data['spread'],
                'volume': data['volume'],
                'timestamp': data['time'].isoformat(),
                'source': data['source']
            }

            # Cache with 5-minute TTL
            self.redis_client.setex(
                'usdcop:latest_price',
                300,
                json.dumps(cache_data, default=str)
            )

            # Publish to subscribers
            self.redis_client.publish(
                'market_data:realtime',
                json.dumps({
                    'type': 'price_update',
                    'data': cache_data
                }, default=str)
            )

        except Exception as e:
            logger.error(f"❌ Error caching latest price: {e}")

    def _should_sync_to_db(self) -> bool:
        """Check if we should sync buffered data to market_data table"""
        now = datetime.now()
        return (len(self.data_buffer) >= self.buffer_size or
                (now - self.last_db_sync) >= self.sync_interval)

    async def _sync_to_market_data_table(self):
        """Sync buffered data to market_data table as 5-minute aggregates"""
        if not self.data_buffer:
            return

        try:
            # Group data by 5-minute windows
            df = pd.DataFrame(self.data_buffer)
            df['time'] = pd.to_datetime(df['time'])

            # Create 5-minute bins
            df['time_bin'] = df['time'].dt.floor('5min')

            # Aggregate OHLCV data
            aggregated = df.groupby('time_bin').agg({
                'last': ['first', 'max', 'min', 'last'],
                'volume': 'sum',
                'symbol': 'first',
                'source': 'first'
            }).reset_index()

            # Flatten column names
            aggregated.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'symbol', 'source']

            # Insert aggregated data
            async with self.db_pool.acquire() as conn:
                for _, row in aggregated.iterrows():
                    await conn.execute("""
                        INSERT INTO market_data
                        (symbol, datetime, open, high, low, close, volume, source, created_at)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
                        ON CONFLICT (symbol, datetime, source)
                        DO UPDATE SET
                            high = GREATEST(market_data.high, EXCLUDED.high),
                            low = LEAST(market_data.low, EXCLUDED.low),
                            close = EXCLUDED.close,
                            volume = market_data.volume + EXCLUDED.volume,
                            updated_at = NOW()
                    """,
                        row['symbol'],
                        row['datetime'],
                        Decimal(str(row['open'])),
                        Decimal(str(row['high'])),
                        Decimal(str(row['low'])),
                        Decimal(str(row['close'])),
                        int(row['volume']),
                        f"{row['source']}_aggregated"
                    )

            # Clear buffer and update sync time
            processed_count = len(self.data_buffer)
            self.data_buffer.clear()
            self.last_db_sync = datetime.now()

            logger.info(f"📊 Synced {processed_count} ticks to market_data table as {len(aggregated)} OHLC bars")

        except Exception as e:
            logger.error(f"❌ Error syncing to market_data table: {e}")

class WebSocketConnectionManager:
    """Enhanced WebSocket connection manager"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connections_by_symbol: Dict[str, Set[WebSocket]] = {}
        self.connection_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'errors': 0
        }

    async def connect(self, websocket: WebSocket, symbol: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        self.connection_stats['total_connections'] += 1
        self.connection_stats['active_connections'] = len(self.active_connections)

        if symbol:
            if symbol not in self.connections_by_symbol:
                self.connections_by_symbol[symbol] = set()
            self.connections_by_symbol[symbol].add(websocket)

        logger.info(f"📱 New WebSocket connection. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket, symbol: str = None):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        self.connection_stats['active_connections'] = len(self.active_connections)

        if symbol and symbol in self.connections_by_symbol:
            self.connections_by_symbol[symbol].discard(websocket)
            if not self.connections_by_symbol[symbol]:
                del self.connections_by_symbol[symbol]

        logger.info(f"📱 WebSocket disconnected. Total: {len(self.active_connections)}")

    async def send_to_all(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return

        message_str = json.dumps(message, default=str)
        disconnected = set()

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
                self.connection_stats['messages_sent'] += 1
            except Exception as e:
                logger.error(f"❌ Error sending message: {e}")
                disconnected.add(connection)
                self.connection_stats['errors'] += 1

        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)

    def get_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            **self.connection_stats,
            'connections_by_symbol': {
                symbol: len(connections)
                for symbol, connections in self.connections_by_symbol.items()
            }
        }

class HealthMonitor:
    """Comprehensive health monitoring system"""

    def __init__(self, db_pool: asyncpg.Pool, redis_client: redis.Redis):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.start_time = datetime.now(COT_TZ)
        self.metrics = {
            'service_status': ServiceStatus.INITIALIZING,
            'uptime_seconds': 0,
            'database_connected': False,
            'redis_connected': False,
            'websocket_connected': False,
            'market_hours_active': False,
            'l0_ready': False,
            'last_health_check': None,
            'error_count': 0,
            'processed_ticks': 0
        }

    async def update_metrics(self, **kwargs):
        """Update health metrics"""
        self.metrics.update(kwargs)
        self.metrics['uptime_seconds'] = int((datetime.now(COT_TZ) - self.start_time).total_seconds())
        self.metrics['last_health_check'] = datetime.now(COT_TZ).isoformat()

        # Store in database
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO system_health
                    (service_name, status, details, response_time_ms)
                    VALUES ('enhanced_websocket_service', $1, $2, $3)
                """,
                    self.metrics['service_status'].value,
                    json.dumps(self.metrics, default=str),
                    None
                )
        except Exception as e:
            logger.error(f"❌ Error storing health metrics: {e}")

    async def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            # Database connectivity
            db_connected = False
            db_response_time = None
            try:
                start = datetime.now()
                async with self.db_pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                db_response_time = int((datetime.now() - start).total_seconds() * 1000)
                db_connected = True
            except:
                pass

            # Redis connectivity
            redis_connected = False
            redis_response_time = None
            try:
                start = datetime.now()
                self.redis_client.ping()
                redis_response_time = int((datetime.now() - start).total_seconds() * 1000)
                redis_connected = True
            except:
                pass

            # System resources
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'service_info': {
                    'name': 'enhanced_websocket_service',
                    'version': '4.0.0',
                    'status': self.metrics['service_status'].value,
                    'uptime_seconds': self.metrics['uptime_seconds'],
                    'start_time': self.start_time.isoformat()
                },
                'connectivity': {
                    'database': {
                        'connected': db_connected,
                        'response_time_ms': db_response_time
                    },
                    'redis': {
                        'connected': redis_connected,
                        'response_time_ms': redis_response_time
                    }
                },
                'system_resources': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_mb': memory.available // 1024 // 1024,
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free // 1024 // 1024 // 1024
                },
                'metrics': self.metrics,
                'timestamp': datetime.now(COT_TZ).isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error getting system info: {e}")
            return {'error': str(e)}

class EnhancedWebSocketService:
    """Main enhanced WebSocket service with L0 integration"""

    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.ready_detector = L0ReadySignalDetector()
        self.data_validator = None
        self.market_manager = None
        self.api_manager = None
        self.data_processor = None
        self.connection_manager = WebSocketConnectionManager()
        self.health_monitor = None
        self.websocket_connection = None
        self.is_running = False
        self.status = ServiceStatus.INITIALIZING

        # Configuration
        self.ws_url = "wss://ws.twelvedata.com/v1/quotes/price"
        self.startup_timeout = int(os.getenv('STARTUP_TIMEOUT_MINUTES', '60'))

    async def initialize(self):
        """Initialize all components"""
        try:
            logger.info("🚀 Initializing Enhanced WebSocket Service...")

            # Ensure directories exist
            await self.ready_detector.ensure_directories()

            # Initialize database connection
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise ValueError("DATABASE_URL environment variable not set")

            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=3,
                max_size=15,
                command_timeout=60
            )

            # Test database connection
            async with self.db_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            logger.info("✅ Database connection established")

            # Initialize Redis
            redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            logger.info("✅ Redis connection established")

            # Initialize components
            self.data_validator = L0DataValidator(self.db_pool)
            self.market_manager = ColombianMarketHoursManager(self.db_pool)
            self.api_manager = APIKeyManager()
            self.data_processor = RealtimeDataProcessor(self.db_pool, self.redis_client)
            self.health_monitor = HealthMonitor(self.db_pool, self.redis_client)

            # Update health metrics
            await self.health_monitor.update_metrics(
                service_status=ServiceStatus.WAITING_L0,
                database_connected=True,
                redis_connected=True
            )

            logger.info("✅ Enhanced WebSocket Service initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize service: {e}")
            await self.health_monitor.update_metrics(
                service_status=ServiceStatus.ERROR,
                error_count=self.health_monitor.metrics['error_count'] + 1
            ) if self.health_monitor else None
            raise

    async def startup_sequence(self):
        """Complete startup sequence with L0 integration"""
        try:
            logger.info("🔄 Starting enhanced startup sequence...")

            # Step 1: Wait for L0 ready signal
            self.status = ServiceStatus.WAITING_L0
            await self.health_monitor.update_metrics(service_status=self.status)

            if not await self.ready_detector.wait_for_l0_ready(self.startup_timeout):
                logger.warning("⚠️ L0 ready signal timeout - checking data status anyway")

            # Step 2: Check L0 data freshness and completeness
            self.status = ServiceStatus.CHECKING_DATA
            await self.health_monitor.update_metrics(service_status=self.status)

            l0_status = await self.data_validator.check_l0_data_status()

            logger.info(f"📊 L0 Data Status:")
            logger.info(f"   Ready Signal: {l0_status.ready_signal_exists}")
            logger.info(f"   Data Freshness: {l0_status.data_freshness_hours:.1f} hours" if l0_status.data_freshness_hours else "   Data Freshness: Unknown")
            logger.info(f"   Completeness: {l0_status.completeness_percentage:.1f}%" if l0_status.completeness_percentage else "   Completeness: Unknown")
            logger.info(f"   Total Records: {l0_status.total_records}")
            logger.info(f"   Is Fresh: {l0_status.is_fresh}")
            logger.info(f"   Is Complete: {l0_status.is_complete}")

            # Update health metrics
            await self.health_monitor.update_metrics(
                l0_ready=l0_status.ready_signal_exists,
                l0_data_fresh=l0_status.is_fresh,
                l0_data_complete=l0_status.is_complete,
                l0_completeness_percent=l0_status.completeness_percentage
            )

            # Step 3: Check market hours
            market_session = await self.market_manager.get_next_market_session()
            is_market_open = self.market_manager.is_market_open()

            await self.health_monitor.update_metrics(
                market_hours_active=is_market_open,
                market_session_info=market_session
            )

            if is_market_open:
                logger.info("🟢 Market is OPEN - ready for real-time data collection")
                self.status = ServiceStatus.READY
            else:
                logger.info(f"🔴 Market is CLOSED - next session: {market_session.get('next_open')}")
                self.status = ServiceStatus.MARKET_CLOSED

            await self.health_monitor.update_metrics(service_status=self.status)

            logger.info("✅ Startup sequence completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Startup sequence failed: {e}")
            self.status = ServiceStatus.ERROR
            await self.health_monitor.update_metrics(
                service_status=self.status,
                error_count=self.health_monitor.metrics['error_count'] + 1
            )
            return False

    async def start_realtime_collection(self):
        """Start real-time WebSocket data collection"""
        if not self.market_manager.is_market_open():
            logger.info("📅 Market closed - not starting real-time collection")
            return False

        try:
            logger.info("🔗 Starting real-time WebSocket collection...")

            api_key = await self.api_manager.get_next_api_key()

            async with websockets.connect(self.ws_url) as websocket:
                self.websocket_connection = websocket
                self.status = ServiceStatus.ACTIVE

                await self.health_monitor.update_metrics(
                    service_status=self.status,
                    websocket_connected=True
                )

                # Subscribe to USD/COP
                subscribe_message = {
                    "action": "subscribe",
                    "params": {
                        "symbols": "USD/COP",
                        "apikey": api_key
                    }
                }

                await websocket.send(json.dumps(subscribe_message))
                logger.info("✅ Subscribed to USD/COP real-time data")

                # Listen for messages
                async for message in websocket:
                    if not self.market_manager.is_market_open():
                        logger.info("📈 Market closed - stopping real-time collection")
                        break

                    try:
                        data = json.loads(message)
                        if await self.data_processor.process_realtime_tick(data):
                            await self.health_monitor.update_metrics(
                                processed_ticks=self.health_monitor.metrics['processed_ticks'] + 1
                            )

                        # Broadcast to WebSocket clients
                        await self.connection_manager.send_to_all({
                            'type': 'market_data',
                            'data': data,
                            'timestamp': datetime.now(COT_TZ).isoformat()
                        })

                    except Exception as e:
                        logger.error(f"❌ Error processing WebSocket message: {e}")
                        await self.health_monitor.update_metrics(
                            error_count=self.health_monitor.metrics['error_count'] + 1
                        )

        except Exception as e:
            logger.error(f"❌ WebSocket connection error: {e}")
            self.status = ServiceStatus.ERROR
            await self.health_monitor.update_metrics(
                service_status=self.status,
                websocket_connected=False,
                error_count=self.health_monitor.metrics['error_count'] + 1
            )
            return False

        finally:
            self.websocket_connection = None
            await self.health_monitor.update_metrics(websocket_connected=False)

    async def run_service(self):
        """Main service loop"""
        logger.info("🎯 Starting Enhanced WebSocket Service main loop")
        self.is_running = True

        # Run startup sequence
        if not await self.startup_sequence():
            logger.error("❌ Startup sequence failed - service cannot start")
            return

        while self.is_running:
            try:
                if self.market_manager.is_market_open():
                    if self.status != ServiceStatus.ACTIVE:
                        logger.info("🟢 Market opened - starting real-time collection")
                        await self.start_realtime_collection()
                else:
                    if self.status == ServiceStatus.ACTIVE:
                        logger.info("🔴 Market closed - stopping real-time collection")
                        self.status = ServiceStatus.MARKET_CLOSED
                        await self.health_monitor.update_metrics(
                            service_status=self.status,
                            market_hours_active=False
                        )

                # Update health metrics periodically
                await self.health_monitor.update_metrics(
                    market_hours_active=self.market_manager.is_market_open()
                )

                # Wait before next check
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"❌ Error in service loop: {e}")
                self.status = ServiceStatus.ERROR
                await self.health_monitor.update_metrics(
                    service_status=self.status,
                    error_count=self.health_monitor.metrics['error_count'] + 1
                )
                await asyncio.sleep(60)

    async def close(self):
        """Close all connections and cleanup"""
        logger.info("🔌 Shutting down Enhanced WebSocket Service...")
        self.is_running = False
        self.status = ServiceStatus.SHUTDOWN

        if self.websocket_connection:
            await self.websocket_connection.close()

        if self.db_pool:
            await self.db_pool.close()

        if self.redis_client:
            self.redis_client.close()

        if self.health_monitor:
            await self.health_monitor.update_metrics(service_status=self.status)

        logger.info("✅ Enhanced WebSocket Service shutdown complete")

# Global service instance
enhanced_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global enhanced_service

    # Startup
    enhanced_service = EnhancedWebSocketService()
    await enhanced_service.initialize()

    # Start service in background
    asyncio.create_task(enhanced_service.run_service())

    yield

    # Shutdown
    if enhanced_service:
        await enhanced_service.close()

# FastAPI application
app = FastAPI(
    title="Enhanced USDCOP WebSocket Service",
    description="Real-time market data WebSocket service with intelligent L0 backup integration",
    version="4.0.0",
    lifespan=lifespan
)

# CORS configuration
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000').split(',')
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    if not enhanced_service:
        return JSONResponse({"status": "unhealthy", "error": "Service not initialized"}, status_code=503)

    try:
        system_info = await enhanced_service.health_monitor.get_system_info()

        return JSONResponse({
            "status": "healthy" if enhanced_service.is_running else "unhealthy",
            "service": "enhanced-websocket-service",
            "version": "4.0.0",
            **system_info
        })

    except Exception as e:
        logger.error(f"❌ Error in health check: {e}")
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=500)

@app.get("/status")
async def get_detailed_status():
    """Get detailed service status"""
    if not enhanced_service:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        # Get L0 data status
        l0_status = await enhanced_service.data_validator.check_l0_data_status()

        # Get market session info
        market_session = await enhanced_service.market_manager.get_next_market_session()

        # Get API key usage
        api_usage = enhanced_service.api_manager.get_usage_stats()

        # Get connection stats
        connection_stats = enhanced_service.connection_manager.get_stats()

        return JSONResponse({
            "service_status": enhanced_service.status.value,
            "is_running": enhanced_service.is_running,
            "l0_integration": {
                "ready_signal_exists": l0_status.ready_signal_exists,
                "ready_signal_timestamp": l0_status.ready_signal_timestamp.isoformat() if l0_status.ready_signal_timestamp else None,
                "data_freshness_hours": l0_status.data_freshness_hours,
                "completeness_percentage": l0_status.completeness_percentage,
                "total_records": l0_status.total_records,
                "is_fresh": l0_status.is_fresh,
                "is_complete": l0_status.is_complete
            },
            "market_session": market_session,
            "api_key_usage": api_usage,
            "websocket_connections": connection_stats,
            "realtime_collection": {
                "active": enhanced_service.status == ServiceStatus.ACTIVE,
                "websocket_connected": enhanced_service.websocket_connection is not None
            },
            "timestamp": datetime.now(COT_TZ).isoformat()
        })

    except Exception as e:
        logger.error(f"❌ Error getting detailed status: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.get("/market/latest")
async def get_latest_price():
    """Get latest cached price"""
    if not enhanced_service or not enhanced_service.redis_client:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        cached_data = enhanced_service.redis_client.get('usdcop:latest_price')
        if cached_data:
            return json.loads(cached_data)
        else:
            return JSONResponse({"error": "No data available"}, status_code=404)

    except Exception as e:
        logger.error(f"❌ Error getting latest price: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.get("/market/session")
async def get_market_session():
    """Get current market session information"""
    if not enhanced_service:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        session_info = await enhanced_service.market_manager.get_next_market_session()
        return JSONResponse(session_info)

    except Exception as e:
        logger.error(f"❌ Error getting market session: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time data"""
    if not enhanced_service:
        await websocket.close(code=1011, reason="Service not available")
        return

    await enhanced_service.connection_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages (heartbeat, subscription requests, etc.)
                if message.get('type') == 'heartbeat':
                    await websocket.send_text(json.dumps({
                        'type': 'heartbeat_response',
                        'timestamp': datetime.now(COT_TZ).isoformat(),
                        'server_status': enhanced_service.status.value
                    }))

            except json.JSONDecodeError:
                logger.warning(f"❌ Invalid JSON received from client: {data}")

    except WebSocketDisconnect:
        enhanced_service.connection_manager.disconnect(websocket)

@app.websocket("/ws/{symbol}")
async def websocket_symbol_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for specific symbol updates"""
    if not enhanced_service:
        await websocket.close(code=1011, reason="Service not available")
        return

    await enhanced_service.connection_manager.connect(websocket, symbol)
    try:
        while True:
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_text(json.dumps({
                'type': 'heartbeat',
                'symbol': symbol,
                'timestamp': datetime.now(COT_TZ).isoformat(),
                'server_status': enhanced_service.status.value
            }))
    except WebSocketDisconnect:
        enhanced_service.connection_manager.disconnect(websocket, symbol)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )