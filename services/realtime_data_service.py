"""
Real-time Data Service for USDCOP Trading System
Handles data updates during market hours (8:00 AM - 12:55 PM)
Updates every 5 minutes and publishes to Redis
"""

import asyncio
import json
import logging
import os
from datetime import datetime, time, date
from decimal import Decimal
import asyncpg
import redis
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketDataManager:
    """Manages market data collection and storage"""
    
    def __init__(self, database_url: str, redis_url: str):
        self.database_url = database_url
        self.redis_url = redis_url
        self.db_pool = None
        self.redis_client = None
        self.is_running = False
        self.current_session_date = None
        
        # Market hours configuration
        self.market_start = time(8, 0)  # 8:00 AM
        self.market_end = time(12, 55)  # 12:55 PM
        self.update_interval = int(os.getenv('UPDATE_INTERVAL', 300))  # 5 minutes
        
    async def initialize(self):
        """Initialize database and Redis connections"""
        try:
            # Initialize database pool
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=60
            )
            
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test connections
            async with self.db_pool.acquire() as conn:
                await conn.fetchval('SELECT 1')
            
            self.redis_client.ping()
            
            logger.info("Database and Redis connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    async def close(self):
        """Close all connections"""
        if self.db_pool:
            await self.db_pool.close()
        if self.redis_client:
            self.redis_client.close()
        logger.info("All connections closed")
    
    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now().time()
        today = datetime.now().weekday()  # 0=Monday, 6=Sunday
        
        # Check if it's a weekday (Monday to Friday)
        if today >= 5:  # Saturday or Sunday
            return False
            
        return self.market_start <= now <= self.market_end
    
    async def ensure_session_exists(self):
        """Ensure current session exists in database"""
        today = date.today()
        
        if self.current_session_date != today:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO market_sessions (session_date, start_time, status)
                    VALUES ($1, $2, 'active')
                    ON CONFLICT (session_date) 
                    DO UPDATE SET updated_at = NOW()
                """, today, datetime.now())
                
                self.current_session_date = today
                logger.info(f"Market session initialized for {today}")
    
    async def get_latest_market_data(self) -> Optional[Dict[str, Any]]:
        """Simulate getting latest market data (replace with actual data source)"""
        try:
            # This is a simulation - replace with actual MT5 or other data source
            import random
            
            # Get last known price from database
            async with self.db_pool.acquire() as conn:
                last_price = await conn.fetchrow("""
                    SELECT close, time 
                    FROM historical_market_data 
                    WHERE symbol = 'USDCOP'
                    ORDER BY time DESC 
                    LIMIT 1
                """)
                
                base_price = float(last_price['close']) if last_price else 4200.0
            
            # Simulate realistic price movement
            change_percent = random.uniform(-0.005, 0.005)  # -0.5% to +0.5%
            new_price = base_price * (1 + change_percent)
            spread = random.uniform(0.5, 2.0)
            
            market_data = {
                'symbol': 'USDCOP',
                'timestamp': datetime.now().isoformat(),
                'bid': round(new_price - spread/2, 6),
                'ask': round(new_price + spread/2, 6),
                'last': round(new_price, 6),
                'volume': random.randint(100000, 1000000),
                'change_percent': round(change_percent * 100, 4),
                'spread': round(spread, 6)
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    async def store_realtime_data(self, data: Dict[str, Any]):
        """Store real-time data in database"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO realtime_market_data 
                    (time, symbol, bid, ask, last, volume, change_percent, session_date)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                    datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')),
                    data['symbol'],
                    Decimal(str(data['bid'])),
                    Decimal(str(data['ask'])),
                    Decimal(str(data['last'])),
                    data['volume'],
                    Decimal(str(data['change_percent'])) if data['change_percent'] else None,
                    date.today()
                )
                
            logger.debug(f"Stored real-time data for {data['symbol']}")
            
        except Exception as e:
            logger.error(f"Error storing real-time data: {e}")
    
    async def store_historical_data(self, data: Dict[str, Any]):
        """Store data as historical 5-minute candle"""
        try:
            # Get OHLC data for the current 5-minute window
            now = datetime.now()
            window_start = now.replace(minute=(now.minute // 5) * 5, second=0, microsecond=0)
            
            async with self.db_pool.acquire() as conn:
                # Check if candle already exists for this time window
                existing = await conn.fetchrow("""
                    SELECT time, open, high, low, close, volume
                    FROM historical_market_data
                    WHERE symbol = $1 AND time = $2
                """, data['symbol'], window_start)
                
                if existing:
                    # Update existing candle (high, low, close, volume)
                    await conn.execute("""
                        UPDATE historical_market_data
                        SET high = GREATEST(high, $3),
                            low = LEAST(low, $4),
                            close = $5,
                            volume = volume + $6,
                            tick_count = tick_count + 1,
                            updated_at = NOW()
                        WHERE symbol = $1 AND time = $2
                    """, 
                        data['symbol'], 
                        window_start,
                        Decimal(str(data['last'])),
                        Decimal(str(data['last'])),
                        Decimal(str(data['last'])),
                        data['volume']
                    )
                else:
                    # Create new candle
                    await conn.execute("""
                        INSERT INTO historical_market_data
                        (time, symbol, open, high, low, close, volume, spread, tick_count)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, 1)
                    """, 
                        window_start,
                        data['symbol'],
                        Decimal(str(data['last'])),
                        Decimal(str(data['last'])),
                        Decimal(str(data['last'])),
                        Decimal(str(data['last'])),
                        data['volume'],
                        Decimal(str(data['spread'])) if data.get('spread') else None
                    )
                    
            logger.debug(f"Updated historical data for {data['symbol']} at {window_start}")
            
        except Exception as e:
            logger.error(f"Error storing historical data: {e}")
    
    async def publish_to_redis(self, channel: str, data: Dict[str, Any]):
        """Publish data to Redis channel"""
        try:
            message = json.dumps(data, default=str)
            self.redis_client.publish(channel, message)
            logger.debug(f"Published to Redis channel {channel}")
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")
    
    async def cache_latest_data(self, data: Dict[str, Any]):
        """Cache latest data in Redis for fast access"""
        try:
            cache_key = f"latest_market_data:{data['symbol']}"
            cache_data = json.dumps(data, default=str)
            self.redis_client.setex(cache_key, 600, cache_data)  # 10 minutes TTL
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    async def update_system_health(self, status: str, response_time: int = None):
        """Update system health metrics"""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO system_health 
                    (time, service_name, status, response_time_ms)
                    VALUES ($1, $2, $3, $4)
                """, 
                    datetime.now(),
                    'realtime-data-service',
                    status,
                    response_time
                )
        except Exception as e:
            logger.error(f"Error updating system health: {e}")
    
    async def data_collection_loop(self):
        """Main data collection loop"""
        logger.info("Starting data collection loop")
        
        while self.is_running:
            try:
                if self.is_market_open():
                    start_time = datetime.now()
                    
                    # Ensure session exists
                    await self.ensure_session_exists()
                    
                    # Get latest market data
                    market_data = await self.get_latest_market_data()
                    
                    if market_data:
                        # Store in both real-time and historical tables
                        await self.store_realtime_data(market_data)
                        await self.store_historical_data(market_data)
                        
                        # Cache for fast access
                        await self.cache_latest_data(market_data)
                        
                        # Publish to WebSocket subscribers
                        await self.publish_to_redis('market_data:realtime', market_data)
                        
                        # Calculate response time
                        response_time = int((datetime.now() - start_time).total_seconds() * 1000)
                        await self.update_system_health('healthy', response_time)
                        
                        logger.info(f"Market data updated: {market_data['symbol']} = {market_data['last']}")
                    
                    else:
                        await self.update_system_health('error')
                        logger.warning("Failed to get market data")
                
                else:
                    # Market is closed
                    logger.info("Market is closed, waiting...")
                    await self.update_system_health('idle')
                    
                # Wait for next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in data collection loop: {e}")
                await self.update_system_health('error')
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    async def start(self):
        """Start the data collection service"""
        await self.initialize()
        self.is_running = True
        await self.data_collection_loop()
    
    async def stop(self):
        """Stop the data collection service"""
        self.is_running = False
        await self.close()

# Global instance
data_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global data_manager
    
    # Startup
    database_url = os.getenv('DATABASE_URL')
    redis_url = os.getenv('REDIS_URL')
    
    data_manager = MarketDataManager(database_url, redis_url)
    
    # Start data collection in background
    asyncio.create_task(data_manager.start())
    
    yield
    
    # Shutdown
    if data_manager:
        await data_manager.stop()

# FastAPI application
app = FastAPI(
    title="USDCOP Real-time Data Service",
    description="Collects and stores real-time market data",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = data_manager and data_manager.is_running
    
    return JSONResponse({
        "status": "healthy" if is_healthy else "unhealthy",
        "service": "realtime-data-service",
        "timestamp": datetime.now().isoformat(),
        "market_open": data_manager.is_market_open() if data_manager else False,
        "session_date": str(data_manager.current_session_date) if data_manager and data_manager.current_session_date else None
    })

@app.get("/market/latest/{symbol}")
async def get_latest_market_data(symbol: str = "USDCOP"):
    """Get latest market data for symbol"""
    if not data_manager or not data_manager.redis_client:
        return JSONResponse({"error": "Service not available"}, status_code=503)
    
    try:
        cache_key = f"latest_market_data:{symbol}"
        cached_data = data_manager.redis_client.get(cache_key)
        
        if cached_data:
            return json.loads(cached_data)
        else:
            return JSONResponse({"error": "No data available"}, status_code=404)
            
    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)

@app.get("/market/status")
async def market_status():
    """Get market status"""
    return JSONResponse({
        "is_open": data_manager.is_market_open() if data_manager else False,
        "current_time": datetime.now().isoformat(),
        "market_hours": {
            "start": "08:00",
            "end": "12:55",
            "timezone": "America/Bogota"
        },
        "session_date": str(data_manager.current_session_date) if data_manager and data_manager.current_session_date else None
    })

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )