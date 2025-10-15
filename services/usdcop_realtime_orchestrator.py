"""
USDCOP Real-time Orchestrator Service
Manages the complete real-time data flow with L0 pipeline dependency management
Only operates during market hours (L-V 8:00 AM - 12:55 PM COT)
"""

import asyncio
import json
import logging
import os
import redis
import websockets
from datetime import datetime, time, timezone, timedelta
from decimal import Decimal
from typing import Dict, Any, Optional, List
import asyncpg
from contextlib import asynccontextmanager
import pytz
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

class PipelineDependencyManager:
    """Manages dependencies between L0 pipeline and real-time services"""

    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.l0_pipeline_completed = False
        self.realtime_enabled = False

    async def check_l0_pipeline_status(self) -> bool:
        """Check if L0 pipeline has completed successfully for today"""
        try:
            today = datetime.now(COT_TZ).date()

            async with self.db_pool.acquire() as conn:
                # Check if L0 pipeline completed today
                result = await conn.fetchrow("""
                    SELECT status, completed_at, records_processed
                    FROM pipeline_status
                    WHERE pipeline_name LIKE '%L0%'
                    AND pipeline_type = 'L0'
                    AND DATE(started_at AT TIME ZONE 'America/Bogota') = $1
                    AND status = 'completed'
                    ORDER BY completed_at DESC
                    LIMIT 1
                """, today)

                if result:
                    logger.info(f"L0 pipeline completed at {result['completed_at']} with {result['records_processed']} records")
                    return True

                # Check if there's historical data available (fallback)
                historical_count = await conn.fetchval("""
                    SELECT COUNT(*)
                    FROM market_data
                    WHERE symbol = 'USDCOP'
                    AND datetime >= $1
                    AND source != 'websocket'
                """, today - timedelta(days=1))

                if historical_count > 0:
                    logger.info(f"Found {historical_count} historical records, L0 pipeline data available")
                    return True

                logger.info("L0 pipeline not completed yet, waiting...")
                return False

        except Exception as e:
            logger.error(f"Error checking L0 pipeline status: {e}")
            return False

    async def wait_for_l0_completion(self, max_wait_minutes: int = 30):
        """Wait for L0 pipeline to complete before starting real-time collection"""
        logger.info(f"Waiting for L0 pipeline completion (max {max_wait_minutes} minutes)...")

        start_time = datetime.now()
        max_wait_time = start_time + timedelta(minutes=max_wait_minutes)

        while datetime.now() < max_wait_time:
            if await self.check_l0_pipeline_status():
                self.l0_pipeline_completed = True
                logger.info("✅ L0 pipeline completed! Starting real-time data collection...")
                return True

            # Wait 60 seconds before checking again
            await asyncio.sleep(60)

        logger.warning(f"⚠️ L0 pipeline did not complete within {max_wait_minutes} minutes")
        return False

    async def update_pipeline_status(self, pipeline_name: str, status: str, **kwargs):
        """Update pipeline status in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO pipeline_status
                    (pipeline_name, pipeline_type, status, started_at, progress_percent, current_step, records_processed)
                    VALUES ($1, 'realtime', $2, NOW(), $3, $4, $5)
                    ON CONFLICT (pipeline_name, status)
                    DO UPDATE SET
                        updated_at = NOW(),
                        progress_percent = EXCLUDED.progress_percent,
                        current_step = EXCLUDED.current_step,
                        records_processed = EXCLUDED.records_processed
                """,
                    pipeline_name,
                    status,
                    kwargs.get('progress_percent', 0),
                    kwargs.get('current_step', 'Starting'),
                    kwargs.get('records_processed', 0)
                )
        except Exception as e:
            logger.error(f"Error updating pipeline status: {e}")

class MarketHoursManager:
    """Manages market hours validation for Colombia (COT)"""

    def __init__(self, db_pool):
        self.db_pool = db_pool
        self.market_start_hour = 8
        self.market_start_minute = 0
        self.market_end_hour = 12
        self.market_end_minute = 55

    async def is_market_open(self) -> bool:
        """Check if market is currently open using database function"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchval("SELECT is_market_open()")
                return result
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            # Fallback to local calculation
            return self._local_market_check()

    def _local_market_check(self) -> bool:
        """Local fallback for market hours check"""
        now = datetime.now(COT_TZ)
        current_time = now.time()
        weekday = now.weekday()  # 0=Monday, 6=Sunday

        # Check if it's a weekday
        if weekday >= 5:  # Saturday or Sunday
            return False

        # Check time range
        market_start = time(self.market_start_hour, self.market_start_minute)
        market_end = time(self.market_end_hour, self.market_end_minute)

        return market_start <= current_time <= market_end

    async def get_session_info(self) -> Dict[str, Any]:
        """Get current market session information"""
        try:
            async with self.db_pool.acquire() as conn:
                result = await conn.fetchrow("""
                    SELECT * FROM get_current_market_session()
                """)

                if result:
                    return {
                        'session_date': result['session_date'].isoformat(),
                        'is_open': result['is_open'],
                        'current_time': result['current_time'].isoformat(),
                        'next_open': result['next_open'].isoformat() if result['next_open'] else None,
                        'next_close': result['next_close'].isoformat() if result['next_close'] else None
                    }

        except Exception as e:
            logger.error(f"Error getting session info: {e}")

        # Fallback response
        now = datetime.now(COT_TZ)
        return {
            'session_date': now.date().isoformat(),
            'is_open': await self.is_market_open(),
            'current_time': now.isoformat(),
            'next_open': None,
            'next_close': None
        }

    async def ensure_session_exists(self):
        """Ensure current market session exists in database"""
        try:
            today = datetime.now(COT_TZ).date()

            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_sessions (session_date, start_time, status)
                    VALUES ($1, $2, 'active')
                    ON CONFLICT (session_date)
                    DO UPDATE SET updated_at = NOW()
                """, today, datetime.now(COT_TZ))

                logger.debug(f"Market session ensured for {today}")

        except Exception as e:
            logger.error(f"Error ensuring market session: {e}")

class RealtimeDataManager:
    """Manages real-time data collection with WebSocket integration"""

    def __init__(self, db_pool, redis_client):
        self.db_pool = db_pool
        self.redis_client = redis_client
        self.websocket_connection = None
        self.is_collecting = False

        # Configuration
        self.ws_url = "wss://ws.twelvedata.com/v1/quotes/price"
        self.api_key = os.getenv('TWELVEDATA_API_KEY_1')

    async def start_websocket_collection(self):
        """Start WebSocket data collection"""
        if not self.api_key:
            logger.error("No TwelveData API key available")
            return

        try:
            logger.info("Connecting to TwelveData WebSocket...")

            async with websockets.connect(self.ws_url) as websocket:
                self.websocket_connection = websocket
                self.is_collecting = True

                # Subscribe to USD/COP
                subscribe_message = {
                    "action": "subscribe",
                    "params": {
                        "symbols": "USD/COP",
                        "apikey": self.api_key
                    }
                }

                await websocket.send(json.dumps(subscribe_message))
                logger.info("✅ Subscribed to USD/COP real-time data")

                # Update status
                await self._update_status("websocket_connected")

                # Listen for messages
                async for message in websocket:
                    try:
                        data = json.loads(message)
                        await self.process_realtime_data(data)
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")

        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self.is_collecting = False
            await self._update_status("websocket_error", error_details=str(e))

    async def process_realtime_data(self, data: Dict[str, Any]):
        """Process incoming real-time data"""
        try:
            # Validate data structure
            if not isinstance(data, dict) or 'price' not in data:
                return

            # Normalize data format
            normalized_data = {
                'symbol': 'USDCOP',
                'time': datetime.now(COT_TZ),
                'last': float(data['price']),
                'bid': float(data.get('bid', data['price'])),
                'ask': float(data.get('ask', data['price'])),
                'volume': int(data.get('volume', 0)),
                'source': 'twelvedata_ws',
                'session_date': datetime.now(COT_TZ).date(),
                'trading_session': True
            }

            # Calculate spread
            normalized_data['spread'] = normalized_data['ask'] - normalized_data['bid']

            # Store in real-time table
            await self._store_realtime_data(normalized_data)

            # Cache in Redis
            await self._cache_latest_data(normalized_data)

            # Publish to subscribers
            await self._publish_to_subscribers(normalized_data)

            logger.debug(f"Processed: USDCOP = {normalized_data['last']}")

        except Exception as e:
            logger.error(f"Error processing real-time data: {e}")

    async def _store_realtime_data(self, data: Dict[str, Any]):
        """Store real-time data in database"""
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
            logger.error(f"Error storing real-time data: {e}")

    async def _cache_latest_data(self, data: Dict[str, Any]):
        """Cache latest data in Redis"""
        try:
            cache_key = f"usdcop:latest"
            cache_data = {
                'symbol': data['symbol'],
                'price': data['last'],
                'bid': data['bid'],
                'ask': data['ask'],
                'spread': data['spread'],
                'timestamp': data['time'].isoformat(),
                'source': data['source']
            }

            self.redis_client.setex(
                cache_key,
                300,  # 5 minutes TTL
                json.dumps(cache_data, default=str)
            )

        except Exception as e:
            logger.error(f"Error caching data: {e}")

    async def _publish_to_subscribers(self, data: Dict[str, Any]):
        """Publish data to Redis pub/sub for WebSocket clients"""
        try:
            message = {
                'type': 'realtime_price',
                'symbol': data['symbol'],
                'price': data['last'],
                'bid': data['bid'],
                'ask': data['ask'],
                'volume': data['volume'],
                'timestamp': data['time'].isoformat(),
                'source': data['source']
            }

            self.redis_client.publish(
                'market_data:realtime',
                json.dumps(message, default=str)
            )

        except Exception as e:
            logger.error(f"Error publishing to subscribers: {e}")

    async def _update_status(self, status: str, **kwargs):
        """Update service status"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO system_health
                    (service_name, status, details, error_message)
                    VALUES ('realtime_orchestrator', $1, $2, $3)
                """,
                    status,
                    json.dumps(kwargs) if kwargs else None,
                    kwargs.get('error_details')
                )

        except Exception as e:
            logger.error(f"Error updating status: {e}")

class USDCOPRealtimeOrchestrator:
    """Main orchestrator for USDCOP real-time data system"""

    def __init__(self):
        self.db_pool = None
        self.redis_client = None
        self.dependency_manager = None
        self.market_hours_manager = None
        self.realtime_manager = None
        self.is_running = False

    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize database connection
            database_url = os.getenv('DATABASE_URL')
            self.db_pool = await asyncpg.create_pool(
                database_url,
                min_size=3,
                max_size=15,
                command_timeout=60
            )

            # Test database connection
            async with self.db_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')

            # Initialize Redis
            redis_url = os.getenv('REDIS_URL')
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()

            # Initialize managers
            self.dependency_manager = PipelineDependencyManager(self.db_pool)
            self.market_hours_manager = MarketHoursManager(self.db_pool)
            self.realtime_manager = RealtimeDataManager(self.db_pool, self.redis_client)

            logger.info("🚀 USDCOP Realtime Orchestrator initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise

    async def close(self):
        """Close all connections"""
        self.is_running = False

        if self.realtime_manager and self.realtime_manager.websocket_connection:
            await self.realtime_manager.websocket_connection.close()

        if self.db_pool:
            await self.db_pool.close()

        if self.redis_client:
            self.redis_client.close()

        logger.info("🔌 USDCOP Realtime Orchestrator closed")

    async def run_orchestrator(self):
        """Main orchestration loop"""
        logger.info("🎯 Starting USDCOP Real-time Orchestrator")
        self.is_running = True

        while self.is_running:
            try:
                # Check if market is open
                if not await self.market_hours_manager.is_market_open():
                    logger.info("📅 Market closed - waiting for next session...")
                    await self.dependency_manager.update_pipeline_status(
                        'realtime_orchestrator', 'market_closed'
                    )
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue

                # Ensure market session exists
                await self.market_hours_manager.ensure_session_exists()

                # Check L0 pipeline dependency
                if not self.dependency_manager.l0_pipeline_completed:
                    logger.info("⏳ Waiting for L0 pipeline to complete...")
                    await self.dependency_manager.update_pipeline_status(
                        'realtime_orchestrator', 'waiting_l0_pipeline'
                    )

                    if not await self.dependency_manager.wait_for_l0_completion():
                        logger.warning("⚠️ L0 pipeline timeout - starting with available data")

                # Start real-time data collection
                if not self.realtime_manager.is_collecting:
                    await self.dependency_manager.update_pipeline_status(
                        'realtime_orchestrator', 'active', current_step='Starting WebSocket collection'
                    )

                    await self.realtime_manager.start_websocket_collection()

                # Keep running while market is open
                while await self.market_hours_manager.is_market_open() and self.is_running:
                    await asyncio.sleep(30)  # Check every 30 seconds

                # Market closed - stop collection
                if self.realtime_manager.is_collecting:
                    self.realtime_manager.is_collecting = False
                    logger.info("📈 Market closed - stopping real-time collection")

            except Exception as e:
                logger.error(f"Error in orchestrator loop: {e}")
                await self.dependency_manager.update_pipeline_status(
                    'realtime_orchestrator', 'error', current_step=f'Error: {str(e)}'
                )
                await asyncio.sleep(60)  # Wait 1 minute before retry

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global orchestrator

    # Startup
    orchestrator = USDCOPRealtimeOrchestrator()
    await orchestrator.initialize()

    # Start orchestrator in background
    asyncio.create_task(orchestrator.run_orchestrator())

    yield

    # Shutdown
    if orchestrator:
        await orchestrator.close()

app = FastAPI(
    title="USDCOP Real-time Orchestrator",
    description="Complete orchestration service for USDCOP real-time trading data",
    version="3.0.0",
    lifespan=lifespan
)

orchestrator = None

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    if not orchestrator:
        return JSONResponse({"status": "unhealthy", "error": "Orchestrator not initialized"}, status_code=503)

    market_info = await orchestrator.market_hours_manager.get_session_info()

    return JSONResponse({
        "status": "healthy" if orchestrator.is_running else "unhealthy",
        "service": "usdcop-realtime-orchestrator",
        "version": "3.0.0",
        "timestamp": datetime.now(COT_TZ).isoformat(),
        "l0_pipeline_completed": orchestrator.dependency_manager.l0_pipeline_completed,
        "realtime_collecting": orchestrator.realtime_manager.is_collecting if orchestrator.realtime_manager else False,
        "market_session": market_info
    })

@app.get("/status")
async def get_status():
    """Get detailed orchestrator status"""
    if not orchestrator:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        async with orchestrator.db_pool.acquire() as conn:
            # Get latest pipeline status
            pipeline_status = await conn.fetchrow("""
                SELECT * FROM pipeline_status
                WHERE pipeline_name = 'realtime_orchestrator'
                ORDER BY updated_at DESC
                LIMIT 1
            """)

            # Get today's market session
            session_info = await conn.fetchrow("""
                SELECT * FROM market_sessions
                WHERE session_date = CURRENT_DATE
                LIMIT 1
            """)

        return JSONResponse({
            "orchestrator_status": orchestrator.is_running,
            "l0_dependency": {
                "completed": orchestrator.dependency_manager.l0_pipeline_completed,
                "last_check": datetime.now(COT_TZ).isoformat()
            },
            "realtime_collection": {
                "active": orchestrator.realtime_manager.is_collecting if orchestrator.realtime_manager else False,
                "websocket_connected": orchestrator.realtime_manager.websocket_connection is not None if orchestrator.realtime_manager else False
            },
            "pipeline_status": dict(pipeline_status) if pipeline_status else None,
            "session_info": dict(session_info) if session_info else None,
            "timestamp": datetime.now(COT_TZ).isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.get("/market/latest")
async def get_latest_price():
    """Get latest cached price from Redis"""
    if not orchestrator or not orchestrator.redis_client:
        return JSONResponse({"error": "Service not available"}, status_code=503)

    try:
        cached_data = orchestrator.redis_client.get("usdcop:latest")
        if cached_data:
            return json.loads(cached_data)
        else:
            return JSONResponse({"error": "No data available"}, status_code=404)

    except Exception as e:
        logger.error(f"Error getting latest price: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )