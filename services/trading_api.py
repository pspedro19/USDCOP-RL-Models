#!/usr/bin/env python3
"""
USDCOP Trading API - Professional Data Service
==============================================

API REST/WebSocket para servir datos de trading al frontend TradingView
Mejores prácticas:
- API REST para datos históricos
- WebSocket para datos en tiempo real
- Caché Redis para performance
- Indicadores técnicos calculados server-side
"""

from fastapi import FastAPI, WebSocket, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
import pandas as pd
import numpy as np
import redis
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
# import talib  # Commented out due to build issues
import pyti
from contextlib import asynccontextmanager
import uvicorn
import logging
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
POSTGRES_CONFIG = {
    'host': 'usdcop-postgres-timescale',
    'port': 5432,
    'database': 'usdcop_trading',
    'user': 'admin',
    'password': 'admin123'
}

REDIS_CONFIG = {
    'host': 'usdcop-redis',
    'port': 6379,
    'password': 'redis123',
    'decode_responses': True
}

# Pydantic models
class CandlestickData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: int

class TechnicalIndicators(BaseModel):
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    rsi: Optional[float] = None

class TradingViewCandle(BaseModel):
    time: int  # Unix timestamp
    open: float
    high: float
    low: float
    close: float
    volume: int
    indicators: TechnicalIndicators

# Global connections
db_pool = None
redis_client = None
active_websockets = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejo del ciclo de vida de la aplicación"""
    global db_pool, redis_client

    # Startup
    logger.info("🚀 Starting Trading API...")

    try:
        # Initialize Redis connection
        redis_client = redis.Redis(**REDIS_CONFIG)
        redis_client.ping()
        logger.info("✅ Redis connected")

        # Test PostgreSQL connection
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        conn.close()
        logger.info("✅ PostgreSQL connected")

    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

    yield

    # Shutdown
    logger.info("🛑 Shutting down Trading API...")
    if redis_client:
        redis_client.close()

# Initialize FastAPI
app = FastAPI(
    title="USDCOP Trading API",
    description="Professional trading data API for TradingView integration",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        return psycopg2.connect(**POSTGRES_CONFIG)
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators using TA-Lib"""
    if len(df) < 200:  # Need enough data for indicators
        return df

    try:
        # Convert to numpy arrays
        close = df['close'].values.astype(float)
        high = df['high'].values.astype(float)
        low = df['low'].values.astype(float)
        volume = df['volume'].values.astype(float)

        # EMAs
        df['ema_20'] = talib.EMA(close, timeperiod=20)
        df['ema_50'] = talib.EMA(close, timeperiod=50)
        df['ema_200'] = talib.EMA(close, timeperiod=200)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower

        # RSI
        df['rsi'] = talib.RSI(close, timeperiod=14)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(close)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_hist

        # Volume indicators
        df['volume_sma'] = talib.SMA(volume, timeperiod=20)

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")

    return df

def cache_key(symbol: str, timeframe: str, start: str, end: str) -> str:
    """Generate cache key"""
    return f"trading_data:{symbol}:{timeframe}:{start}:{end}"

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "ok", "service": "USDCOP Trading API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Comprehensive health check"""
    health = {"status": "ok", "checks": {}}

    # Check Redis
    try:
        redis_client.ping()
        health["checks"]["redis"] = "ok"
    except:
        health["checks"]["redis"] = "error"
        health["status"] = "degraded"

    # Check PostgreSQL
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM market_data WHERE symbol = 'USDCOP'")
            count = cursor.fetchone()[0]
            health["checks"]["postgres"] = "ok"
            health["checks"]["data_records"] = count
        conn.close()
    except:
        health["checks"]["postgres"] = "error"
        health["status"] = "error"

    return health

@app.get("/api/candlesticks/{symbol}")
async def get_candlesticks(
    symbol: str,
    timeframe: str = Query("5m", description="Timeframe: 5m, 15m, 1h, 4h, 1d"),
    start_date: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(1000, description="Maximum number of candles"),
    include_indicators: bool = Query(True, description="Include technical indicators")
):
    """
    Get candlestick data with technical indicators
    Optimized for TradingView integration
    """

    # Validate symbol
    if symbol.upper() != "USDCOP":
        raise HTTPException(status_code=404, detail="Symbol not found")

    # Default date range (last 30 days if not specified)
    if not end_date:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if not start_date:
        start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=30)
        start_date = start_dt.strftime('%Y-%m-%d')

    # Check cache first
    cache_key_str = cache_key(symbol, timeframe, start_date, end_date)
    try:
        cached_data = redis_client.get(cache_key_str)
        if cached_data:
            logger.info(f"📋 Cache hit for {symbol} {timeframe}")
            return json.loads(cached_data)
    except:
        pass  # Cache miss, continue with DB query

    try:
        conn = get_db_connection()

        # Build query based on timeframe
        if timeframe == "5m":
            # Raw 5-minute data
            query = """
            SELECT
                timestamp,
                price as close,
                bid as low,
                ask as high,
                price as open,
                volume
            FROM market_data
            WHERE symbol = %s
            AND DATE(timestamp) BETWEEN %s AND %s
            ORDER BY timestamp
            LIMIT %s
            """
        else:
            # Aggregated data for other timeframes
            interval_map = {
                "15m": "15 minutes",
                "1h": "1 hour",
                "4h": "4 hours",
                "1d": "1 day"
            }

            if timeframe not in interval_map:
                raise HTTPException(status_code=400, detail="Invalid timeframe")

            query = f"""
            SELECT
                date_trunc('{interval_map[timeframe]}', timestamp) as timestamp,
                FIRST(price, timestamp) as open,
                MAX(ask) as high,
                MIN(bid) as low,
                LAST(price, timestamp) as close,
                SUM(volume) as volume
            FROM market_data
            WHERE symbol = %s
            AND DATE(timestamp) BETWEEN %s AND %s
            GROUP BY date_trunc('{interval_map[timeframe]}', timestamp)
            ORDER BY timestamp
            LIMIT %s
            """

        # Execute query
        df = pd.read_sql_query(query, conn, params=[symbol.upper(), start_date, end_date, limit])
        conn.close()

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for the specified range")

        # Calculate technical indicators if requested
        if include_indicators:
            df = calculate_technical_indicators(df)

        # Convert to TradingView format
        candlesticks = []
        for _, row in df.iterrows():
            # Convert timestamp to Unix timestamp (milliseconds)
            if isinstance(row['timestamp'], str):
                ts = pd.to_datetime(row['timestamp'])
            else:
                ts = row['timestamp']

            unix_timestamp = int(ts.timestamp() * 1000)

            indicators = TechnicalIndicators()
            if include_indicators:
                indicators.ema_20 = float(row.get('ema_20', 0)) if pd.notna(row.get('ema_20')) else None
                indicators.ema_50 = float(row.get('ema_50', 0)) if pd.notna(row.get('ema_50')) else None
                indicators.ema_200 = float(row.get('ema_200', 0)) if pd.notna(row.get('ema_200')) else None
                indicators.bb_upper = float(row.get('bb_upper', 0)) if pd.notna(row.get('bb_upper')) else None
                indicators.bb_middle = float(row.get('bb_middle', 0)) if pd.notna(row.get('bb_middle')) else None
                indicators.bb_lower = float(row.get('bb_lower', 0)) if pd.notna(row.get('bb_lower')) else None
                indicators.rsi = float(row.get('rsi', 0)) if pd.notna(row.get('rsi')) else None

            candle = TradingViewCandle(
                time=unix_timestamp,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=int(row['volume']),
                indicators=indicators
            )
            candlesticks.append(candle.dict())

        # Prepare response
        response = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "count": len(candlesticks),
            "data": candlesticks
        }

        # Cache the response (5 minutes for recent data, 1 hour for historical)
        cache_ttl = 300 if end_date == datetime.now().strftime('%Y-%m-%d') else 3600
        try:
            redis_client.setex(cache_key_str, cache_ttl, json.dumps(response))
        except:
            pass  # Cache write failed, continue

        logger.info(f"📊 Served {len(candlesticks)} candles for {symbol} {timeframe}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching candlesticks: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/latest/{symbol}")
async def get_latest_price(symbol: str):
    """Get latest price data for a symbol"""

    if symbol.upper() != "USDCOP":
        raise HTTPException(status_code=404, detail="Symbol not found")

    try:
        conn = get_db_connection()

        query = """
        SELECT
            timestamp,
            price as close,
            bid,
            ask,
            volume,
            source
        FROM market_data
        WHERE symbol = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """

        with conn.cursor() as cursor:
            cursor.execute(query, [symbol.upper()])
            result = cursor.fetchone()

            if not result:
                raise HTTPException(status_code=404, detail="No data found")

            latest = {
                "symbol": symbol.upper(),
                "timestamp": result[0].isoformat(),
                "price": float(result[1]),
                "bid": float(result[2]),
                "ask": float(result[3]),
                "volume": int(result[4]),
                "source": result[5],
                "last_updated": datetime.now().isoformat()
            }

        conn.close()
        return latest

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest price: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """
    WebSocket endpoint for real-time data
    Sends updates when new data is available
    """

    if symbol.upper() != "USDCOP":
        await websocket.close(code=4004, reason="Symbol not found")
        return

    await websocket.accept()
    active_websockets.add(websocket)

    try:
        logger.info(f"📡 WebSocket connected for {symbol}")

        # Send initial latest data
        latest_data = await get_latest_price(symbol)
        await websocket.send_json({
            "type": "initial",
            "data": latest_data
        })

        # Keep connection alive and send updates
        last_timestamp = None

        while True:
            await asyncio.sleep(5)  # Check every 5 seconds

            # Check for new data
            try:
                current_latest = await get_latest_price(symbol)
                current_timestamp = current_latest["timestamp"]

                if last_timestamp != current_timestamp:
                    await websocket.send_json({
                        "type": "update",
                        "data": current_latest
                    })
                    last_timestamp = current_timestamp
                    logger.info(f"📈 Sent update for {symbol}")

            except Exception as e:
                logger.error(f"Error in WebSocket update: {e}")
                break

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        active_websockets.discard(websocket)
        logger.info(f"📡 WebSocket disconnected for {symbol}")

@app.get("/api/symbols")
async def get_available_symbols():
    """Get list of available trading symbols"""
    return {
        "symbols": [
            {
                "symbol": "USDCOP",
                "name": "US Dollar / Colombian Peso",
                "type": "forex",
                "exchange": "Colombian Market",
                "active": True
            }
        ]
    }

@app.get("/api/stats/{symbol}")
async def get_symbol_stats(symbol: str):
    """Get comprehensive statistics for a symbol"""

    if symbol.upper() != "USDCOP":
        raise HTTPException(status_code=404, detail="Symbol not found")

    try:
        conn = get_db_connection()

        query = """
        SELECT
            COUNT(*) as total_records,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price,
            STDDEV(price) as price_std
        FROM market_data
        WHERE symbol = %s
        """

        with conn.cursor() as cursor:
            cursor.execute(query, [symbol.upper()])
            result = cursor.fetchone()

            stats = {
                "symbol": symbol.upper(),
                "total_records": result[0],
                "first_date": result[1].isoformat() if result[1] else None,
                "last_date": result[2].isoformat() if result[2] else None,
                "min_price": float(result[3]) if result[3] else None,
                "max_price": float(result[4]) if result[4] else None,
                "avg_price": float(result[5]) if result[5] else None,
                "price_std": float(result[6]) if result[6] else None,
                "generated_at": datetime.now().isoformat()
            }

        conn.close()
        return stats

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    # For development
    uvicorn.run(
        "trading_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )