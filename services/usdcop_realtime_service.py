"""
Enhanced USDCOP Real-time Service with TwelveData WebSocket Integration
Optimized for ultra-low latency data processing during market hours (8:00-12:55 COT)
"""

import asyncio
import json
import logging
import os
import redis
import websockets
from datetime import datetime, time, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional
import asyncpg
from contextlib import asynccontextmanager
import pytz

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

# Redis key constants
REDIS_KEYS = {
    "latest_price": "usdcop:latest",
    "5min_buffer": "usdcop:5min_buffer",
    "market_status": "usdcop:market_status",
    "session_stats": "usdcop:session_stats"
}

class RealtimeCache:
    """High-performance Redis cache manager for real-time data"""

    def __init__(self, redis_client):
        self.redis = redis_client

    def cache_latest_price(self, price_data: Dict[str, Any]):
        """Cache the latest price with 5-minute TTL"""
        try:
            self.redis.setex(
                REDIS_KEYS["latest_price"],
                300,  # 5 minutes TTL
                json.dumps(price_data, default=str)
            )
        except Exception as e:
            logger.error(f"Error caching latest price: {e}")

    def buffer_5min_data(self, data: Dict[str, Any]):
        """Buffer data for 5-minute batch processing"""
        try:
            # Add to buffer list
            self.redis.lpush(REDIS_KEYS["5min_buffer"], json.dumps(data, default=str))
            # Set expiration for safety
            self.redis.expire(REDIS_KEYS["5min_buffer"], 600)  # 10 minutes

            # Keep buffer size manageable (max 100 entries)
            self.redis.ltrim(REDIS_KEYS["5min_buffer"], 0, 99)
        except Exception as e:
            logger.error(f"Error buffering data: {e}")

    def get_buffer_data(self) -> list:
        """Get all buffered data and clear buffer"""
        try:
            # Get all data from buffer
            data = self.redis.lrange(REDIS_KEYS["5min_buffer"], 0, -1)
            # Clear the buffer
            self.redis.delete(REDIS_KEYS["5min_buffer"])

            return [json.loads(item) for item in data]
        except Exception as e:
            logger.error(f"Error getting buffer data: {e}")
            return []

    def update_market_status(self, status: str):
        """Update market status in cache"""
        try:
            status_data = {
                "status": status,
                "timestamp": datetime.now(COT_TZ).isoformat(),
                "market_hours": {
                    "start": "08:00",
                    "end": "12:55",
                    "timezone": "America/Bogota"
                }
            }
            self.redis.setex(
                REDIS_KEYS["market_status"],
                60,  # 1 minute TTL
                json.dumps(status_data)
            )
        except Exception as e:
            logger.error(f"Error updating market status: {e}")

class DatabaseManager:
    """Enhanced database manager with UPSERT operations"""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=3,
            max_size=15,
            command_timeout=60
        )

        # Test connection
        async with self.pool.acquire() as conn:
            await conn.fetchval('SELECT 1')

        logger.info("Database pool initialized successfully")

    async def close(self):
        """Close database pool"""
        if self.pool:
            await self.pool.close()

    async def upsert_market_data(self, data_batch: list):
        """Batch UPSERT market data to avoid duplicates"""
        if not data_batch:
            return 0

        try:
            async with self.pool.acquire() as conn:
                # Prepare UPSERT query
                upsert_query = """
                INSERT INTO market_data (symbol, datetime, open, high, low, close, volume, source, trading_session, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'websocket', true, NOW())
                ON CONFLICT (symbol, datetime, timeframe, source)
                DO UPDATE SET
                    close = EXCLUDED.close,
                    high = GREATEST(market_data.high, EXCLUDED.high),
                    low = LEAST(market_data.low, EXCLUDED.low),
                    volume = market_data.volume + EXCLUDED.volume,
                    updated_at = NOW()
                RETURNING id
                """

                inserted_count = 0
                for data in data_batch:
                    dt = datetime.fromisoformat(data['timestamp'].replace('Z', ''))

                    result = await conn.fetchval(
                        upsert_query,
                        data['symbol'],
                        dt,
                        Decimal(str(data.get('open', data['last']))),
                        Decimal(str(data.get('high', data['last']))),
                        Decimal(str(data.get('low', data['last']))),
                        Decimal(str(data['last'])),
                        data.get('volume', 0),
                    )

                    if result:
                        inserted_count += 1

                logger.info(f"UPSERT completed: {inserted_count} records processed")
                return inserted_count

        except Exception as e:
            logger.error(f"Error in UPSERT operation: {e}")
            return 0

class USDCOPRealtimeService:
    """Enhanced real-time service for USDCOP data"""

    def __init__(self):
        self.db = None
        self.redis_client = None
        self.cache = None
        self.is_running = False
        self.websocket_connection = None

        # Configuration
        self.market_start_hour = int(os.getenv('MARKET_START_HOUR', 8))
        self.market_end_hour = int(os.getenv('MARKET_END_HOUR', 12))
        self.market_end_minute = int(os.getenv('MARKET_END_MINUTE', 55))
        self.sync_interval_minutes = int(os.getenv('SYNC_INTERVAL_MINUTES', 5))

        # TwelveData WebSocket URL
        self.ws_url = "wss://ws.twelvedata.com/v1/quotes/price"
        self.api_key = os.getenv('TWELVEDATA_API_KEY_1')

    async def initialize(self):
        """Initialize all connections"""
        try:
            # Initialize database
            database_url = os.getenv('DATABASE_URL')
            self.db = DatabaseManager(database_url)
            await self.db.initialize()

            # Initialize Redis
            redis_url = os.getenv('REDIS_URL')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()

            # Initialize cache manager
            self.cache = RealtimeCache(self.redis_client)

            logger.info("USDCOPRealtimeService initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service: {e}")
            raise

    async def close(self):
        """Close all connections"""
        self.is_running = False

        if self.websocket_connection:
            await self.websocket_connection.close()

        if self.db:
            await self.db.close()

        if self.redis_client:
            self.redis_client.close()

        logger.info("USDCOPRealtimeService closed")

    def is_market_open(self) -> bool:
        """Check if market is currently open (8:00-12:55 COT, Mon-Fri)"""
        now = datetime.now(COT_TZ)
        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Check if it's a weekday
        if weekday >= 5:  # Saturday or Sunday
            return False

        # Check time range
        market_start = time(self.market_start_hour, 0)
        market_end = time(self.market_end_hour, self.market_end_minute)

        return market_start <= current_time <= market_end

    async def market_hours_websocket(self):
        """Main WebSocket connection active only during market hours"""
        if not self.is_market_open():
            logger.info("Market is closed, WebSocket will not connect")
            self.cache.update_market_status("closed")
            return

        try:
            self.cache.update_market_status("connecting")

            async with websockets.connect(self.ws_url) as websocket:
                self.websocket_connection = websocket
                self.cache.update_market_status("connected")

                # Subscribe to USD/COP
                subscribe_message = {
                    "action": "subscribe",
                    "params": {
                        "symbols": "USD/COP"
                    }
                }

                if self.api_key:
                    subscribe_message["params"]["apikey"] = self.api_key

                await websocket.send(json.dumps(subscribe_message))
                logger.info("Subscribed to USD/COP real-time data")

                # Start listening to messages
                async for message in websocket:
                    if not self.is_market_open():
                        logger.info("Market closed, stopping WebSocket")
                        break

                    try:
                        data = json.loads(message)
                        await self.process_realtime_data(data)
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.cache.update_market_status("error")

    async def process_realtime_data(self, data: Dict[str, Any]):
        """Process incoming real-time data with ultra-fast caching"""
        try:
            # Validate data structure
            if not isinstance(data, dict) or 'price' not in data:
                return

            # Normalize data format
            normalized_data = {
                'symbol': 'USDCOP',
                'timestamp': datetime.now(COT_TZ).isoformat(),
                'last': float(data['price']),
                'bid': float(data.get('bid', data['price'])),
                'ask': float(data.get('ask', data['price'])),
                'volume': int(data.get('volume', 0)),
                'source': 'twelvedata_ws'
            }

            # 1. Cache immediately to Redis (ultra-fast)
            self.cache.cache_latest_price(normalized_data)

            # 2. Buffer for 5-minute batch processing
            self.cache.buffer_5min_data(normalized_data)

            # 3. Publish to WebSocket subscribers
            await self.broadcast_to_clients(normalized_data)

            logger.debug(f"Processed real-time data: {normalized_data['symbol']} = {normalized_data['last']}")

        except Exception as e:
            logger.error(f"Error processing real-time data: {e}")

    async def broadcast_to_clients(self, data: Dict[str, Any]):
        """Broadcast data to WebSocket clients via Redis pub/sub"""
        try:
            message = {
                'type': 'market_data',
                'data': data,
                'timestamp': datetime.now(COT_TZ).isoformat()
            }

            self.redis_client.publish('market_data:realtime', json.dumps(message, default=str))

        except Exception as e:
            logger.error(f"Error broadcasting to clients: {e}")

    async def should_persist_to_db(self) -> bool:
        """Check if it's time to persist buffered data to database"""
        try:
            last_sync_key = "usdcop:last_sync"
            last_sync = self.redis_client.get(last_sync_key)

            if not last_sync:
                return True

            last_sync_time = datetime.fromisoformat(last_sync)
            now = datetime.now(COT_TZ)

            # Sync every 5 minutes
            return (now - last_sync_time).total_seconds() >= (self.sync_interval_minutes * 60)

        except Exception as e:
            logger.error(f"Error checking sync timing: {e}")
            return False

    async def sync_to_postgresql(self):
        """Sync buffered data to PostgreSQL"""
        try:
            if not await self.should_persist_to_db():
                return

            # Get buffered data
            buffer_data = self.cache.get_buffer_data()

            if not buffer_data:
                logger.info("No data in buffer to sync")
                return

            # Batch insert to PostgreSQL
            inserted_count = await self.db.upsert_market_data(buffer_data)

            # Update last sync time
            self.redis_client.set(
                "usdcop:last_sync",
                datetime.now(COT_TZ).isoformat()
            )

            logger.info(f"Synced {inserted_count} records to PostgreSQL")

        except Exception as e:
            logger.error(f"Error syncing to PostgreSQL: {e}")

    async def run_service(self):
        """Main service loop"""
        logger.info("Starting USDCOP Real-time Service")
        self.is_running = True

        while self.is_running:
            try:
                if self.is_market_open():
                    # Run WebSocket connection during market hours
                    await self.market_hours_websocket()
                else:
                    # Market closed - update status and wait
                    self.cache.update_market_status("closed")
                    logger.info("Market closed, waiting for next session...")
                    await asyncio.sleep(300)  # Wait 5 minutes

                # Always try to sync any remaining buffered data
                await self.sync_to_postgresql()

            except Exception as e:
                logger.error(f"Error in service loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry

# FastAPI integration
from fastapi import FastAPI
from fastapi.responses import JSONResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global realtime_service

    # Startup
    realtime_service = USDCOPRealtimeService()
    await realtime_service.initialize()

    # Start service in background
    asyncio.create_task(realtime_service.run_service())

    yield

    # Shutdown
    if realtime_service:
        await realtime_service.close()

app = FastAPI(
    title="USDCOP Enhanced Real-time Service",
    description="Ultra-low latency USDCOP data processing with TwelveData WebSocket",
    version="2.0.0",
    lifespan=lifespan
)

realtime_service = None

@app.get("/health")
async def health_check():
    """Enhanced health check"""
    return JSONResponse({
        "status": "healthy" if realtime_service and realtime_service.is_running else "unhealthy",
        "service": "usdcop-realtime-service",
        "version": "2.0.0",
        "timestamp": datetime.now(COT_TZ).isoformat(),
        "market_open": realtime_service.is_market_open() if realtime_service else False,
        "websocket_connected": realtime_service.websocket_connection is not None if realtime_service else False
    })

@app.get("/market/latest")
async def get_latest_price():
    """Get latest cached price"""
    if not realtime_service or not realtime_service.redis_client:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        cached_data = realtime_service.redis_client.get(REDIS_KEYS["latest_price"])
        if cached_data:
            return json.loads(cached_data)
        else:
            return JSONResponse({"error": "No data available"}, status_code=404)
    except Exception as e:
        logger.error(f"Error getting latest price: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.get("/market/status")
async def market_status():
    """Get detailed market status"""
    if not realtime_service:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        cached_status = realtime_service.redis_client.get(REDIS_KEYS["market_status"])
        status_data = json.loads(cached_status) if cached_status else {}

        return JSONResponse({
            "is_open": realtime_service.is_market_open(),
            "current_time": datetime.now(COT_TZ).isoformat(),
            "websocket_status": "connected" if realtime_service.websocket_connection else "disconnected",
            **status_data
        })
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )