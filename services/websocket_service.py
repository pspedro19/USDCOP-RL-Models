"""
WebSocket Service for Real-time Market Data
Handles WebSocket connections for live trading data updates
"""

import asyncio
import json
import logging
from datetime import datetime, time
from typing import Dict, Set, Any
import redis
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.connections_by_symbol: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, symbol: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        if symbol:
            if symbol not in self.connections_by_symbol:
                self.connections_by_symbol[symbol] = set()
            self.connections_by_symbol[symbol].add(websocket)
            
        logger.info(f"New WebSocket connection. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket, symbol: str = None):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        
        if symbol and symbol in self.connections_by_symbol:
            self.connections_by_symbol[symbol].discard(websocket)
            if not self.connections_by_symbol[symbol]:
                del self.connections_by_symbol[symbol]
                
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_to_all(self, message: dict):
        """Send message to all connected clients"""
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending message to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_symbol_subscribers(self, symbol: str, message: dict):
        """Send message to clients subscribed to specific symbol"""
        if symbol not in self.connections_by_symbol:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.connections_by_symbol[symbol]:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error sending message to connection: {e}")
                disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn, symbol)

class RedisSubscriber:
    """Handles Redis pub/sub for real-time data"""
    
    def __init__(self, redis_url: str, connection_manager: ConnectionManager):
        self.redis_url = redis_url
        self.connection_manager = connection_manager
        self.redis_client = None
        self.pubsub = None
        self.is_running = False
        
    async def start_subscriber(self):
        """Start listening to Redis pub/sub"""
        try:
            # Initialize Redis client
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test connection first
            try:
                self.redis_client.ping()
                logger.info("Redis connection successful")
            except Exception as e:
                logger.error(f"Redis connection failed: {e}")
                return
                
            self.pubsub = self.redis_client.pubsub()
            # Subscribe to different channels
            self.pubsub.subscribe(
                'market_data:realtime',
                'market_data:historical',
                'system:health',
                'trading:alerts'
            )
            
            self.is_running = True
            logger.info("Redis subscriber started")
            
            # Run subscriber in background without blocking
            while self.is_running:
                try:
                    message = self.pubsub.get_message(timeout=0.1)  # Non-blocking
                    if message and message['type'] == 'message':
                        await self.handle_redis_message(message)
                    await asyncio.sleep(0.1)  # Small delay to prevent busy loop
                except Exception as e:
                    logger.error(f"Error processing Redis message: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"Error starting Redis subscriber: {e}")
            
    async def stop_subscriber(self):
        """Stop the Redis subscriber"""
        self.is_running = False
        if self.pubsub:
            try:
                self.pubsub.unsubscribe()
                self.pubsub.close()
            except:
                pass
        if self.redis_client:
            try:
                self.redis_client.close()
            except:
                pass
        logger.info("Redis subscriber stopped")
        
    async def handle_redis_message(self, message):
        """Process incoming Redis messages and forward to WebSocket clients"""
        try:
            channel = message['channel']
            data = json.loads(message['data'])
            
            # Add metadata
            websocket_message = {
                'channel': channel,
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            # Route messages based on channel
            if channel == 'market_data:realtime':
                symbol = data.get('symbol', 'USDCOP')
                await self.connection_manager.send_to_symbol_subscribers(symbol, websocket_message)
            else:
                # Send to all connections for system health and alerts
                await self.connection_manager.send_to_all(websocket_message)
                
        except Exception as e:
            logger.error(f"Error handling Redis message: {e}")

# Global instances
connection_manager = ConnectionManager()
redis_subscriber = None

async def start_redis_subscriber():
    """Start Redis subscriber in background"""
    global redis_subscriber
    try:
        if redis_subscriber:
            await redis_subscriber.start_subscriber()
    except Exception as e:
        logger.error(f"Failed to start Redis subscriber: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global redis_subscriber
    
    # Startup
    redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    redis_subscriber = RedisSubscriber(redis_url, connection_manager)
    
    # Start Redis subscriber in background task
    asyncio.create_task(start_redis_subscriber())
    
    # Give it a moment to initialize
    await asyncio.sleep(0.1)
    
    yield
    
    # Shutdown
    if redis_subscriber:
        await redis_subscriber.stop_subscriber()

# FastAPI application
app = FastAPI(
    title="USDCOP WebSocket Service",
    description="Real-time market data WebSocket service",
    version="1.0.0",
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
    """Health check endpoint"""
    return JSONResponse({
        "status": "healthy",
        "service": "websocket-service",
        "timestamp": datetime.now().isoformat(),
        "connections": len(connection_manager.active_connections),
        "redis_connected": redis_subscriber.is_running if redis_subscriber else False
    })

@app.get("/metrics")
async def metrics():
    """Service metrics endpoint"""
    return JSONResponse({
        "total_connections": len(connection_manager.active_connections),
        "connections_by_symbol": {
            symbol: len(connections) 
            for symbol, connections in connection_manager.connections_by_symbol.items()
        },
        "redis_status": "connected" if redis_subscriber and redis_subscriber.is_running else "disconnected",
        "timestamp": datetime.now().isoformat()
    })

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for general updates"""
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client requests (like subscription requests)
                if message.get('type') == 'subscribe':
                    symbol = message.get('symbol', 'USDCOP')
                    if symbol not in connection_manager.connections_by_symbol:
                        connection_manager.connections_by_symbol[symbol] = set()
                    connection_manager.connections_by_symbol[symbol].add(websocket)
                    
                    await websocket.send_text(json.dumps({
                        'type': 'subscription_confirmed',
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received from client: {data}")
                
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

@app.websocket("/ws/{symbol}")
async def websocket_symbol_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for specific symbol updates"""
    await connection_manager.connect(websocket, symbol)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for heartbeat
            await websocket.send_text(json.dumps({
                'type': 'heartbeat',
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }))
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket, symbol)

def is_market_hours() -> bool:
    """Check if current time is within market hours (8:00 AM - 12:55 PM)"""
    now = datetime.now().time()
    market_start = time(8, 0)  # 8:00 AM
    market_end = time(12, 55)  # 12:55 PM
    
    return market_start <= now <= market_end

@app.get("/market/status")
async def market_status():
    """Get current market status"""
    return JSONResponse({
        "is_open": is_market_hours(),
        "current_time": datetime.now().isoformat(),
        "market_hours": {
            "start": "08:00",
            "end": "12:55",
            "timezone": "America/Bogota"
        }
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