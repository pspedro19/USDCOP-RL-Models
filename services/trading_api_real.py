#!/usr/bin/env python3
"""
USDCOP Trading API - REAL DATA ONLY
==================================

API REST para servir SOLO datos reales de la base de datos
NO genera datos sintéticos ni mock
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
import logging
from pydantic import BaseModel

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

# FastAPI app
app = FastAPI(
    title="USDCOP Trading API - Real Data Only",
    description="API para datos REALES de trading USD/COP (sin datos sintéticos)",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")

def calculate_technical_indicators(df):
    """Calculate technical indicators using pandas ONLY if enough data"""
    if len(df) < 50:  # Need minimum data for indicators
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

@app.get("/")
async def root():
    return {
        "message": "USDCOP Trading API - Real Data Only",
        "status": "active",
        "version": "1.0.0",
        "data_source": "real_only"
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

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected",
            "total_records": count,
            "latest_data": latest.isoformat() if latest else None
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
    """Get latest REAL price for symbol"""
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
            "data_source": "real"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/data-info")
async def get_data_info():
    """Get information about available real data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get data range and statistics
        cursor.execute("""
            SELECT
                COUNT(*) as total_records,
                MIN(timestamp) as earliest_data,
                MAX(timestamp) as latest_data,
                COUNT(DISTINCT symbol) as symbols_count,
                COUNT(DISTINCT source) as sources_count
            FROM market_data
        """)

        result = cursor.fetchone()

        # Get symbol breakdown
        cursor.execute("""
            SELECT symbol, COUNT(*) as count
            FROM market_data
            GROUP BY symbol
            ORDER BY count DESC
        """)

        symbols = cursor.fetchall()

        conn.close()

        return {
            "total_records": result[0],
            "earliest_data": result[1].isoformat() if result[1] else None,
            "latest_data": result[2].isoformat() if result[2] else None,
            "symbols_count": result[3],
            "sources_count": result[4],
            "symbols": [{"symbol": s[0], "count": s[1]} for s in symbols],
            "data_type": "real_only"
        }

    except Exception as e:
        logger.error(f"Error fetching data info: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "trading_api_real:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )