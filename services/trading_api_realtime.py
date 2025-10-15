#!/usr/bin/env python3
"""
USDCOP Trading API - Real-Time with WebSocket
============================================

API REST + WebSocket para datos reales de trading USD/COP
- REST API para datos históricos
- WebSocket para datos en tiempo real cada 5 minutos
- Solo datos reales de la base de datos
"""

from fastapi import FastAPI, WebSocket, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
import logging
from pydantic import BaseModel
import uvicorn
from contextlib import asynccontextmanager
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from environment variables
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'usdcop-postgres-timescale'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# Trading hours configuration (Colombian time)
COT_TIMEZONE = pytz.timezone('America/Bogota')
TRADING_START_HOUR = 8   # 8:00 AM COT
TRADING_END_HOUR = 12    # 12:55 PM COT
TRADING_END_MINUTE = 55
TRADING_DAYS = [0, 1, 2, 3, 4]  # Monday=0 to Friday=4

# Global variables for WebSocket management
websocket_connections: List[WebSocket] = []
last_sent_timestamp = None

# Data models
class CandlestickData(BaseModel):
    time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    rsi: Optional[float] = None

class MarketData(BaseModel):
    symbol: str
    price: float
    timestamp: str
    volume: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    source: str

class CandlestickResponse(BaseModel):
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    count: int
    data: List[CandlestickData]

class RealTimeUpdate(BaseModel):
    type: str = "price_update"
    symbol: str
    price: float
    timestamp: str
    volume: float
    change: Optional[float] = None
    change_percent: Optional[float] = None
    source: str

def is_market_open() -> bool:
    """Check if market is currently open (8AM-12:55PM COT, Mon-Fri)"""
    now_cot = datetime.now(COT_TIMEZONE)

    # Check if it's a trading day (Monday=0 to Friday=4)
    if now_cot.weekday() not in TRADING_DAYS:
        return False

    # Check if it's within trading hours
    trading_start = now_cot.replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)
    trading_end = now_cot.replace(hour=TRADING_END_HOUR, minute=TRADING_END_MINUTE, second=0, microsecond=0)

    return trading_start <= now_cot <= trading_end

def get_market_status() -> dict:
    """Get detailed market status information"""
    now_cot = datetime.now(COT_TIMEZONE)
    is_open = is_market_open()

    # Calculate next market open/close
    if is_open:
        # Market is open, calculate when it closes
        next_close = now_cot.replace(hour=TRADING_END_HOUR, minute=TRADING_END_MINUTE, second=0, microsecond=0)
        if now_cot > next_close:
            # Already past closing, next event is tomorrow's open
            next_event = (now_cot + timedelta(days=1)).replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)
            next_event_type = "market_open"
        else:
            next_event = next_close
            next_event_type = "market_close"
    else:
        # Market is closed, calculate when it opens
        next_open = now_cot.replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)

        # If it's weekend or after hours today, move to next trading day
        if now_cot.weekday() > 4 or (now_cot.weekday() == 4 and now_cot.hour >= TRADING_END_HOUR):
            # Move to next Monday
            days_ahead = 7 - now_cot.weekday()
            next_open = (now_cot + timedelta(days=days_ahead)).replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)
        elif now_cot.hour >= TRADING_END_HOUR:
            # After hours today, move to tomorrow
            next_open = (now_cot + timedelta(days=1)).replace(hour=TRADING_START_HOUR, minute=0, second=0, microsecond=0)

        next_event = next_open
        next_event_type = "market_open"

    return {
        "is_open": is_open,
        "current_time": now_cot.isoformat(),
        "timezone": "America/Bogota",
        "trading_hours": f"{TRADING_START_HOUR:02d}:00 - {TRADING_END_HOUR:02d}:{TRADING_END_MINUTE:02d} COT",
        "trading_days": "Monday - Friday",
        "next_event": next_event.isoformat(),
        "next_event_type": next_event_type,
        "time_to_next_event": str(next_event - now_cot).split(".")[0]
    }

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def calculate_technical_indicators(df):
    """Calculate technical indicators using pandas"""
    if len(df) < 50:
        logger.warning(f"Not enough data for indicators: {len(df)} rows")
        return df

    try:
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    except Exception as e:
        logger.warning(f"Error calculating indicators: {e}")

    return df

def create_candlesticks_from_ticks(df, timeframe='5m'):
    """Convert tick data to candlesticks"""
    if df.empty:
        return df

    # Parse timeframe
    timeframe_map = {
        '1m': '1T', '5m': '5T', '15m': '15T', '30m': '30T',
        '1h': '1H', '4h': '4H', '1d': '1D'
    }
    pandas_freq = timeframe_map.get(timeframe, '5T')

    # Convert to candlesticks
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Resample to create candlesticks
    candlesticks = df.resample(pandas_freq).agg({
        'price': ['first', 'max', 'min', 'last'],
        'volume': 'sum'
    }).dropna()

    # Flatten column names
    candlesticks.columns = ['open', 'high', 'low', 'close', 'volume']
    candlesticks = candlesticks.reset_index()

    return candlesticks

async def get_latest_data():
    """Get latest market data from database"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get latest record
        query = """
        SELECT symbol, price, timestamp, COALESCE(volume, 0) as volume,
               bid, ask, source
        FROM market_data
        WHERE symbol = 'USDCOP'
        ORDER BY timestamp DESC
        LIMIT 1
        """

        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            # Get previous price for change calculation
            cursor.execute("""
                SELECT price FROM market_data
                WHERE symbol = 'USDCOP' AND timestamp < %s
                ORDER BY timestamp DESC LIMIT 1
            """, (result[2],))
            prev_result = cursor.fetchone()
            prev_price = prev_result[0] if prev_result else result[1]

            change = float(result[1]) - float(prev_price)
            change_percent = (change / float(prev_price)) * 100 if prev_price > 0 else 0

            update = RealTimeUpdate(
                symbol=result[0],
                price=float(result[1]),
                timestamp=result[2].isoformat(),
                volume=float(result[3]),
                change=change,
                change_percent=change_percent,
                source=result[6]
            )

            conn.close()
            return update

        conn.close()
        return None

    except Exception as e:
        logger.error(f"Error getting latest data: {e}")
        return None

async def broadcast_to_websockets(data: dict):
    """Broadcast data to all connected WebSocket clients"""
    if not websocket_connections:
        return

    message = json.dumps(data)
    disconnected = []

    for websocket in websocket_connections:
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.warning(f"WebSocket send failed: {e}")
            disconnected.append(websocket)

    # Remove disconnected clients
    for ws in disconnected:
        websocket_connections.remove(ws)

async def real_time_data_monitor():
    """Background task to monitor for new data every 5 minutes (ONLY during market hours)"""
    global last_sent_timestamp

    while True:
        try:
            # Check for new data every 5 minutes
            await asyncio.sleep(300)  # 5 minutes = 300 seconds

            # Check if market is open - ONLY broadcast during trading hours
            if not is_market_open():
                logger.info("Market is closed - WebSocket monitor paused")
                # Send market closed message to connected clients
                if websocket_connections:
                    market_status = get_market_status()
                    closed_message = {
                        "type": "market_closed",
                        "message": "Market is closed",
                        "market_status": market_status
                    }
                    await broadcast_to_websockets(closed_message)
                continue

            if not websocket_connections:
                continue

            latest_data = await get_latest_data()
            if latest_data:
                # Only send if this is newer data
                current_timestamp = latest_data.timestamp
                if last_sent_timestamp != current_timestamp:
                    logger.info(f"Broadcasting new data: {latest_data.price} at {current_timestamp}")
                    await broadcast_to_websockets(latest_data.dict())
                    last_sent_timestamp = current_timestamp

        except Exception as e:
            logger.error(f"Error in real-time monitor: {e}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying

# Create FastAPI app with lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting real-time data monitor...")
    task = asyncio.create_task(real_time_data_monitor())

    yield

    # Shutdown
    logger.info("Shutting down real-time data monitor...")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="USDCOP Trading API - Real-Time",
    description="API REST + WebSocket para datos REALES de trading USD/COP con actualizaciones cada 5 minutos",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "USDCOP Trading API - Real-Time with WebSocket",
        "status": "active",
        "version": "1.0.0",
        "data_source": "real_only",
        "features": ["REST API", "WebSocket", "5min updates"],
        "websocket_endpoint": "/ws",
        "connected_clients": len(websocket_connections)
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Check data availability
        cursor.execute("SELECT COUNT(*) FROM market_data")
        count = cursor.fetchone()[0]

        cursor.execute("SELECT MAX(timestamp) FROM market_data")
        latest = cursor.fetchone()[0]

        conn.close()

        # Get market status
        market_status = get_market_status()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "total_records": count,
            "latest_data": latest.isoformat() if latest else None,
            "websocket_clients": len(websocket_connections),
            "real_time_monitor": "active" if market_status["is_open"] else "paused",
            "market_status": market_status
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "disconnected",
            "error": str(e)
        }

@app.get("/api/latest/{symbol}")
async def get_latest_price(symbol: str = "USDCOP"):
    """Get latest REAL price for symbol (only during market hours)"""
    # Check if market is open
    market_status = get_market_status()
    if not market_status["is_open"]:
        raise HTTPException(
            status_code=425,  # Too Early
            detail={
                "error": "Market is closed",
                "message": "Real-time data is only available during trading hours: 8:00 AM - 12:55 PM COT (Monday-Friday)",
                "market_status": market_status
            }
        )

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        SELECT symbol, price, timestamp, COALESCE(volume, 0) as volume,
               bid, ask, source
        FROM market_data
        WHERE symbol = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """

        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        conn.close()

        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No real data found for symbol {symbol}"
            )

        return MarketData(
            symbol=result[0],
            price=float(result[1]),
            timestamp=result[2].isoformat(),
            volume=float(result[3]),
            bid=float(result[4]) if result[4] else None,
            ask=float(result[5]) if result[5] else None,
            source=result[6]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest price: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/candlesticks/{symbol}")
async def get_candlesticks(
    symbol: str = "USDCOP",
    timeframe: str = Query("5m", description="Timeframe (1m, 5m, 15m, 30m, 1h, 4h, 1d)"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, description="Maximum number of records"),
    include_indicators: bool = Query(True, description="Include technical indicators")
):
    """Get REAL candlestick data for charts"""
    try:
        conn = get_db_connection()

        # Default date range (last 7 days for performance)
        if not end_date:
            end_date = datetime.utcnow().isoformat()
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=7)).isoformat()

        # Get raw tick data from market_data table
        query = """
        SELECT timestamp, symbol, price, COALESCE(volume, 0) as volume
        FROM market_data
        WHERE symbol = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        LIMIT %s
        """

        df = pd.read_sql_query(
            query,
            conn,
            params=(symbol, start_date, end_date, limit)
        )
        conn.close()

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No real data found for {symbol} between {start_date} and {end_date}"
            )

        # Convert tick data to candlesticks
        candlesticks_df = create_candlesticks_from_ticks(df, timeframe)

        if candlesticks_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No candlestick data could be generated for {symbol}"
            )

        # Calculate technical indicators if requested and enough data
        if include_indicators and len(candlesticks_df) >= 50:
            candlesticks_df = calculate_technical_indicators(candlesticks_df)

        # Convert to response format
        candlesticks = []
        for _, row in candlesticks_df.iterrows():
            candlestick = CandlestickData(
                time=int(row['timestamp'].timestamp() * 1000),
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row['volume'])
            )

            # Add indicators if available
            if include_indicators:
                if 'ema_20' in row and pd.notna(row['ema_20']):
                    candlestick.ema_20 = float(row['ema_20'])
                if 'ema_50' in row and pd.notna(row['ema_50']):
                    candlestick.ema_50 = float(row['ema_50'])
                if 'bb_upper' in row and pd.notna(row['bb_upper']):
                    candlestick.bb_upper = float(row['bb_upper'])
                if 'bb_middle' in row and pd.notna(row['bb_middle']):
                    candlestick.bb_middle = float(row['bb_middle'])
                if 'bb_lower' in row and pd.notna(row['bb_lower']):
                    candlestick.bb_lower = float(row['bb_lower'])
                if 'rsi' in row and pd.notna(row['rsi']):
                    candlestick.rsi = float(row['rsi'])

            candlesticks.append(candlestick)

        return CandlestickResponse(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            count=len(candlesticks),
            data=candlesticks
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching candlesticks: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time data"""
    await websocket.accept()
    websocket_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total clients: {len(websocket_connections)}")

    try:
        # Send initial data
        initial_data = await get_latest_data()
        if initial_data:
            await websocket.send_text(json.dumps(initial_data.dict()))

        # Keep connection alive
        while True:
            # Wait for client messages (ping/pong)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_text(json.dumps({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()}))

    except Exception as e:
        logger.warning(f"WebSocket client disconnected: {e}")
    finally:
        if websocket in websocket_connections:
            websocket_connections.remove(websocket)
        logger.info(f"WebSocket client removed. Total clients: {len(websocket_connections)}")

@app.get("/api/stats/{symbol}")
async def get_symbol_stats(symbol: str = "USDCOP"):
    """Get REAL symbol statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get 24h statistics from real data
        query = """
        SELECT
            COUNT(*) as count,
            MIN(price) as low_24h,
            MAX(price) as max_24h,
            AVG(price) as avg_24h,
            SUM(COALESCE(volume, 0)) as volume_24h,
            MAX(timestamp) as latest_timestamp
        FROM market_data
        WHERE symbol = %s
        AND timestamp >= NOW() - INTERVAL '24 hours'
        """

        cursor.execute(query, (symbol,))
        result = cursor.fetchone()
        conn.close()

        if not result or result[0] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No real data found for {symbol} in the last 24 hours"
            )

        return {
            "symbol": symbol,
            "count_24h": result[0],
            "low_24h": float(result[1]),
            "high_24h": float(result[2]),
            "avg_24h": float(result[3]),
            "volume_24h": float(result[4]),
            "latest_timestamp": result[5].isoformat() if result[5] else None,
            "data_source": "real",
            "websocket_clients": len(websocket_connections)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/market-status")
async def get_market_status_endpoint():
    """Get current market status and trading hours"""
    return get_market_status()

if __name__ == "__main__":
    uvicorn.run(
        "trading_api_realtime:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )