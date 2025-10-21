#!/usr/bin/env python3
"""
USDCOP Trading Signals API
===========================

API para generar señales de trading basadas en:
- Modelos RL entrenados
- Indicadores técnicos
- Análisis de mercado

Endpoints:
- GET /api/trading/signals - Señales reales de trading
- GET /api/trading/signals-test - Señales de prueba/mock
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import logging
from pydantic import BaseModel
import uvicorn
import random
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# Create FastAPI app
app = FastAPI(
    title="USDCOP Trading Signals API",
    description="API para generación de señales de trading",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# MODELS
# ==========================================

class TechnicalIndicators(BaseModel):
    rsi: float
    macd: Dict[str, float]
    bollinger: Dict[str, float]
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    volume_ratio: Optional[float] = None

class TradingSignal(BaseModel):
    id: str
    timestamp: str
    type: str  # BUY, SELL, HOLD
    confidence: float
    price: float
    stopLoss: float
    takeProfit: float
    reasoning: List[str]
    riskScore: float
    expectedReturn: float
    timeHorizon: str
    modelSource: str
    technicalIndicators: TechnicalIndicators

class SignalPerformance(BaseModel):
    winRate: float
    avgWin: float
    avgLoss: float
    profitFactor: float
    sharpeRatio: float
    totalSignals: int
    successfulSignals: int

class SignalsResponse(BaseModel):
    success: bool
    signals: List[TradingSignal]
    performance: SignalPerformance
    timestamp: str

# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def execute_query(query: str, params: tuple = None) -> pd.DataFrame:
    """Execute query and return DataFrame"""
    conn = get_db_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()

# ==========================================
# TECHNICAL INDICATORS CALCULATION
# ==========================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    """Calculate RSI"""
    if len(prices) < period + 1:
        return 50.0

    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

def calculate_macd(prices: pd.Series) -> Dict[str, float]:
    """Calculate MACD"""
    if len(prices) < 26:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0}

    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    histogram = macd - signal

    return {
        "macd": float(macd.iloc[-1]),
        "signal": float(signal.iloc[-1]),
        "histogram": float(histogram.iloc[-1])
    }

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, float]:
    """Calculate Bollinger Bands"""
    if len(prices) < period:
        current_price = float(prices.iloc[-1])
        return {
            "upper": current_price * 1.02,
            "middle": current_price,
            "lower": current_price * 0.98
        }

    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()

    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)

    return {
        "upper": float(upper.iloc[-1]),
        "middle": float(sma.iloc[-1]),
        "lower": float(lower.iloc[-1])
    }

def calculate_ema(prices: pd.Series, period: int) -> float:
    """Calculate EMA"""
    if len(prices) < period:
        return float(prices.mean())

    ema = prices.ewm(span=period, adjust=False).mean()
    return float(ema.iloc[-1])

# ==========================================
# SIGNAL GENERATION
# ==========================================

def generate_signal_from_indicators(
    current_price: float,
    rsi: float,
    macd: Dict[str, float],
    bb: Dict[str, float],
    ema_20: float,
    ema_50: float,
    volume_ratio: float
) -> Dict[str, Any]:
    """Generate trading signal based on technical indicators"""

    reasoning = []
    signal_type = "HOLD"
    confidence = 50.0
    risk_score = 5.0
    expected_return = 0.0

    # RSI Analysis
    if rsi < 30:
        reasoning.append("RSI oversold (< 30)")
        signal_type = "BUY"
        confidence += 15
    elif rsi > 70:
        reasoning.append("RSI overbought (> 70)")
        signal_type = "SELL"
        confidence += 15

    # MACD Analysis
    if macd["histogram"] > 0:
        reasoning.append("MACD bullish crossover")
        if signal_type == "BUY" or signal_type == "HOLD":
            signal_type = "BUY"
            confidence += 10
    elif macd["histogram"] < 0:
        reasoning.append("MACD bearish crossover")
        if signal_type == "SELL" or signal_type == "HOLD":
            signal_type = "SELL"
            confidence += 10

    # Bollinger Bands Analysis
    if current_price < bb["lower"]:
        reasoning.append("Price below lower Bollinger Band")
        if signal_type == "BUY" or signal_type == "HOLD":
            signal_type = "BUY"
            confidence += 12
    elif current_price > bb["upper"]:
        reasoning.append("Price above upper Bollinger Band")
        if signal_type == "SELL" or signal_type == "HOLD":
            signal_type = "SELL"
            confidence += 12

    # EMA Trend Analysis
    if ema_20 > ema_50:
        reasoning.append("Uptrend confirmed (EMA 20 > EMA 50)")
        if signal_type == "BUY":
            confidence += 8
    elif ema_20 < ema_50:
        reasoning.append("Downtrend confirmed (EMA 20 < EMA 50)")
        if signal_type == "SELL":
            confidence += 8

    # Volume Analysis
    if volume_ratio > 1.5:
        reasoning.append("High volume spike detected")
        confidence += 5

    # Cap confidence at 95
    confidence = min(confidence, 95.0)

    # Calculate risk and expected return
    if signal_type == "BUY":
        stop_loss = current_price * 0.985  # 1.5% stop loss
        take_profit = current_price * 1.02  # 2% take profit
        expected_return = 0.02
        risk_score = min(10 - (confidence / 10), 10)
    elif signal_type == "SELL":
        stop_loss = current_price * 1.015  # 1.5% stop loss
        take_profit = current_price * 0.98  # 2% take profit
        expected_return = 0.02
        risk_score = min(10 - (confidence / 10), 10)
    else:
        stop_loss = current_price
        take_profit = current_price
        expected_return = 0.0
        risk_score = 5.0

    if not reasoning:
        reasoning.append("No clear signal - market neutral")

    return {
        "type": signal_type,
        "confidence": confidence,
        "stopLoss": stop_loss,
        "takeProfit": take_profit,
        "reasoning": reasoning,
        "riskScore": risk_score,
        "expectedReturn": expected_return
    }

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "USDCOP Trading Signals API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": [
            "/api/trading/signals",
            "/api/trading/signals-test",
            "/docs"
        ]
    }

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        conn.close()

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/trading/signals")
def get_trading_signals(
    symbol: str = Query(default="USDCOP", description="Trading symbol"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of signals to return")
):
    """
    Get real trading signals based on market data and technical analysis
    """
    try:
        # Get recent market data
        query = """
            SELECT timestamp, price, volume
            FROM market_data
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT 100
        """

        df = execute_query(query, (symbol,))

        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No market data found for symbol {symbol}"
            )

        # Calculate technical indicators
        prices = df['price'].astype(float)
        current_price = float(prices.iloc[0])

        rsi = calculate_rsi(prices)
        macd = calculate_macd(prices)
        bb = calculate_bollinger_bands(prices)
        ema_20 = calculate_ema(prices, 20)
        ema_50 = calculate_ema(prices, 50)

        # Calculate volume ratio
        recent_volume = df['volume'].iloc[:5].mean()
        avg_volume = df['volume'].mean()
        volume_ratio = float(recent_volume / avg_volume) if avg_volume > 0 else 1.0

        # Generate signal
        signal_data = generate_signal_from_indicators(
            current_price, rsi, macd, bb, ema_20, ema_50, volume_ratio
        )

        # Create signal object
        signals = []
        signal_id = f"sig_{int(datetime.utcnow().timestamp())}"

        signal = TradingSignal(
            id=signal_id,
            timestamp=datetime.utcnow().isoformat(),
            type=signal_data["type"],
            confidence=signal_data["confidence"],
            price=current_price,
            stopLoss=signal_data["stopLoss"],
            takeProfit=signal_data["takeProfit"],
            reasoning=signal_data["reasoning"],
            riskScore=signal_data["riskScore"],
            expectedReturn=signal_data["expectedReturn"],
            timeHorizon="15-30 min" if signal_data["confidence"] > 70 else "1-2 hours",
            modelSource="Technical_Analysis_v1.0",
            technicalIndicators=TechnicalIndicators(
                rsi=rsi,
                macd=macd,
                bollinger=bb,
                ema_20=ema_20,
                ema_50=ema_50,
                volume_ratio=volume_ratio
            )
        )

        signals.append(signal)

        # Calculate performance metrics (mock for now - would come from historical signal tracking)
        performance = SignalPerformance(
            winRate=68.5,
            avgWin=125.50,
            avgLoss=75.25,
            profitFactor=2.34,
            sharpeRatio=1.87,
            totalSignals=150,
            successfulSignals=103
        )

        response = SignalsResponse(
            success=True,
            signals=signals,
            performance=performance,
            timestamp=datetime.utcnow().isoformat()
        )

        return response.dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading/signals-test")
def get_test_signals(
    limit: int = Query(default=5, ge=1, le=20, description="Number of test signals")
):
    """
    Get mock/test trading signals for UI testing and development
    """
    try:
        signals = []
        base_price = 4320.50

        signal_types = ["BUY", "SELL", "HOLD"]

        for i in range(limit):
            signal_type = random.choice(signal_types)
            confidence = random.uniform(60, 95)
            price = base_price + random.uniform(-50, 50)

            if signal_type == "BUY":
                stop_loss = price * 0.985
                take_profit = price * 1.02
                expected_return = 0.02
                reasoning = [
                    random.choice(["RSI oversold", "MACD bullish", "Price near support"]),
                    random.choice(["Volume spike", "Uptrend confirmed", "Bullish divergence"])
                ]
            elif signal_type == "SELL":
                stop_loss = price * 1.015
                take_profit = price * 0.98
                expected_return = 0.02
                reasoning = [
                    random.choice(["RSI overbought", "MACD bearish", "Price near resistance"]),
                    random.choice(["Volume decline", "Downtrend confirmed", "Bearish divergence"])
                ]
            else:
                stop_loss = price
                take_profit = price
                expected_return = 0.0
                reasoning = ["Market neutral", "Consolidation phase"]

            signal = TradingSignal(
                id=f"test_sig_{i}_{int(datetime.utcnow().timestamp())}",
                timestamp=(datetime.utcnow() - timedelta(minutes=i*5)).isoformat(),
                type=signal_type,
                confidence=round(confidence, 2),
                price=round(price, 2),
                stopLoss=round(stop_loss, 2),
                takeProfit=round(take_profit, 2),
                reasoning=reasoning,
                riskScore=round(10 - (confidence / 10), 2),
                expectedReturn=round(expected_return, 4),
                timeHorizon=random.choice(["5-15 min", "15-30 min", "30-60 min", "1-2 hours"]),
                modelSource=random.choice([
                    "PPO_LSTM_v2.1",
                    "A2C_GRU_v1.5",
                    "SAC_Transformer_v3.0",
                    "Technical_Analysis_v1.0"
                ]),
                technicalIndicators=TechnicalIndicators(
                    rsi=round(random.uniform(20, 80), 2),
                    macd={
                        "macd": round(random.uniform(-20, 20), 2),
                        "signal": round(random.uniform(-15, 15), 2),
                        "histogram": round(random.uniform(-10, 10), 2)
                    },
                    bollinger={
                        "upper": round(price * 1.015, 2),
                        "middle": round(price, 2),
                        "lower": round(price * 0.985, 2)
                    },
                    ema_20=round(price + random.uniform(-10, 10), 2),
                    ema_50=round(price + random.uniform(-20, 20), 2),
                    volume_ratio=round(random.uniform(0.5, 2.5), 2)
                )
            )

            signals.append(signal)

        performance = SignalPerformance(
            winRate=round(random.uniform(60, 75), 2),
            avgWin=round(random.uniform(100, 200), 2),
            avgLoss=round(random.uniform(50, 100), 2),
            profitFactor=round(random.uniform(1.8, 3.0), 2),
            sharpeRatio=round(random.uniform(1.2, 2.5), 2),
            totalSignals=random.randint(100, 200),
            successfulSignals=random.randint(60, 140)
        )

        response = SignalsResponse(
            success=True,
            signals=signals,
            performance=performance,
            timestamp=datetime.utcnow().isoformat()
        )

        return response.dict()

    except Exception as e:
        logger.error(f"Error generating test signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("TRADING_SIGNALS_API_PORT", "8003"))
    uvicorn.run(
        "trading_signals_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
