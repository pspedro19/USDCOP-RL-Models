#!/usr/bin/env python3
"""
USDCOP Trading API - Simplified Version
======================================

API REST simple para servir datos de trading al frontend
Sin dependencias complejas como TA-Lib
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
    source: str = "database"

class CandlestickResponse(BaseModel):
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    count: int
    data: List[CandlestickData]

# FastAPI app
app = FastAPI(
    title="USDCOP Trading API",
    description="API para datos de trading USD/COP",
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

def calculate_simple_indicators(df):
    """Calculate simple technical indicators using pandas"""
    try:
        # EMAs
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()

        # Bollinger Bands (simplified)
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (2 * bb_std)
        df['bb_lower'] = df['bb_middle'] - (2 * bb_std)

        # RSI (simplified)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    except Exception as e:
        logger.warning(f"Error calculating indicators: {e}")
        # Fill with NaN if calculation fails
        for col in ['ema_20', 'ema_50', 'bb_upper', 'bb_middle', 'bb_lower', 'rsi']:
            if col not in df.columns:
                df[col] = np.nan

    return df

@app.get("/")
async def root():
    return {"message": "USDCOP Trading API", "status": "active", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        conn = get_db_connection()
        conn.close()
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "connected"
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
    """Get latest price for symbol"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Query latest price from l0_raw_data table
        query = """
        SELECT symbol, price, timestamp, volume, source
        FROM l0_raw_data
        WHERE symbol = %s
        ORDER BY timestamp DESC
        LIMIT 1
        """

        cursor.execute(query, (symbol,))
        result = cursor.fetchone()

        if not result:
            # Return mock data if no real data found
            return MarketData(
                symbol=symbol,
                price=4200.0,
                timestamp=datetime.utcnow().isoformat(),
                volume=1000.0,
                source="mock"
            )

        return MarketData(
            symbol=result[0],
            price=float(result[1]),
            timestamp=result[2].isoformat() if result[2] else datetime.utcnow().isoformat(),
            volume=float(result[3]) if result[3] else 0.0,
            source=result[4] if result[4] else "database"
        )

    except Exception as e:
        logger.error(f"Error fetching latest price: {e}")
        # Return mock data on error
        return MarketData(
            symbol=symbol,
            price=4200.0,
            timestamp=datetime.utcnow().isoformat(),
            volume=1000.0,
            source="fallback"
        )
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/api/candlesticks/{symbol}")
async def get_candlesticks(
    symbol: str = "USDCOP",
    timeframe: str = Query("5m", description="Timeframe (5m, 15m, 1h, 4h, 1d)"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    limit: int = Query(1000, description="Maximum number of records"),
    include_indicators: bool = Query(True, description="Include technical indicators")
):
    """Get candlestick data for charts"""
    try:
        conn = get_db_connection()

        # Default date range (last 30 days)
        if not end_date:
            end_date = datetime.utcnow().isoformat()
        if not start_date:
            start_date = (datetime.utcnow() - timedelta(days=30)).isoformat()

        # Build query based on available tables
        # First try l5_serving_data (processed data)
        query = """
        SELECT timestamp, open, high, low, close, volume
        FROM l5_serving_data
        WHERE symbol = %s
        AND timestamp BETWEEN %s AND %s
        ORDER BY timestamp ASC
        LIMIT %s
        """

        df = None
        try:
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol, start_date, end_date, limit)
            )
        except Exception as e1:
            # Fallback to l0_raw_data and create candlesticks
            logger.warning(f"l5_serving_data not available: {e1}, trying l0_raw_data")
            try:
                query = """
                SELECT timestamp, price as close, price as open, price as high, price as low,
                       COALESCE(volume, 1000) as volume
                FROM l0_raw_data
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
            except Exception as e2:
                logger.warning(f"l0_raw_data also not available: {e2}, generating mock data")
                df = None

        if df.empty:
            # Generate mock data if no data found
            logger.warning("No data found, generating mock data")
            end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            start_time = datetime.fromisoformat(start_date.replace('Z', '+00:00'))

            # Generate 100 mock candlesticks
            time_diff = (end_time - start_time) / 100
            mock_data = []
            base_price = 4200.0

            for i in range(100):
                timestamp = start_time + (time_diff * i)
                # Add some random price movement
                price_change = np.random.normal(0, 10)
                price = base_price + price_change
                base_price = price  # Trend continuation

                mock_data.append({
                    'timestamp': timestamp,
                    'open': price - np.random.uniform(-5, 5),
                    'high': price + np.random.uniform(0, 10),
                    'low': price - np.random.uniform(0, 10),
                    'close': price,
                    'volume': np.random.uniform(500, 2000)
                })

            df = pd.DataFrame(mock_data)

        # Ensure proper column types
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate technical indicators if requested
        if include_indicators and len(df) > 20:
            df = calculate_simple_indicators(df)

        # Convert to response format
        candlesticks = []
        for _, row in df.iterrows():
            candlestick = CandlestickData(
                time=int(row['timestamp'].timestamp() * 1000),  # Convert to milliseconds
                open=float(row['open']) if pd.notna(row['open']) else 4200.0,
                high=float(row['high']) if pd.notna(row['high']) else 4200.0,
                low=float(row['low']) if pd.notna(row['low']) else 4200.0,
                close=float(row['close']) if pd.notna(row['close']) else 4200.0,
                volume=float(row['volume']) if pd.notna(row['volume']) else 1000.0
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

    except Exception as e:
        logger.error(f"Error fetching candlesticks: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

@app.get("/api/stats/{symbol}")
async def get_symbol_stats(symbol: str = "USDCOP"):
    """Get symbol statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Get 24h statistics
        query = """
        SELECT
            COUNT(*) as count,
            MIN(price) as low_24h,
            MAX(price) as high_24h,
            AVG(price) as avg_24h,
            SUM(COALESCE(volume, 0)) as volume_24h
        FROM l0_raw_data
        WHERE symbol = %s
        AND timestamp >= NOW() - INTERVAL '24 hours'
        """

        cursor.execute(query, (symbol,))
        result = cursor.fetchone()

        if result and result[0] > 0:
            return {
                "symbol": symbol,
                "count_24h": result[0],
                "low_24h": float(result[1]) if result[1] else 0,
                "high_24h": float(result[2]) if result[2] else 0,
                "avg_24h": float(result[3]) if result[3] else 0,
                "volume_24h": float(result[4]) if result[4] else 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Return mock stats
            return {
                "symbol": symbol,
                "count_24h": 288,  # Every 5 minutes for 24h
                "low_24h": 4180.0,
                "high_24h": 4220.0,
                "avg_24h": 4200.0,
                "volume_24h": 50000.0,
                "timestamp": datetime.utcnow().isoformat()
            }

    except Exception as e:
        logger.error(f"Error fetching stats: {e}")
        # Return mock stats on error
        return {
            "symbol": symbol,
            "count_24h": 288,
            "low_24h": 4180.0,
            "high_24h": 4220.0,
            "avg_24h": 4200.0,
            "volume_24h": 50000.0,
            "timestamp": datetime.utcnow().isoformat(),
            "error": "Using fallback data"
        }
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "trading_api_simple:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )