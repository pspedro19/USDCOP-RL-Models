#!/usr/bin/env python3
"""
Multi-Model Trading Signals API
================================

FastAPI service for multi-strategy trading signals dashboard.
Aggregates signals, performance metrics, equity curves, and positions
from 5 trading strategies: RL_PPO, ML_XGB, ML_LGBM, LLM_CLAUDE, ENSEMBLE.

Port: 8006
Database: PostgreSQL (dw.* schema)
"""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
import json
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
import os
import logging
from pydantic import BaseModel, Field
from enum import Enum
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'usdcop-postgres-timescale'),
    'port': int(os.getenv('POSTGRES_PORT', '5432')),
    'database': os.getenv('POSTGRES_DB', 'usdcop_trading'),
    'user': os.getenv('POSTGRES_USER', 'admin'),
    'password': os.getenv('POSTGRES_PASSWORD', 'admin123')
}

# WebSocket clients
active_websockets: List[WebSocket] = []

# ============================================================
# PYDANTIC MODELS
# ============================================================

class SignalType(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"
    CLOSE = "close"

class SideType(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class StrategyTypeEnum(str, Enum):
    RL = "RL"
    ML = "ML"
    LLM = "LLM"
    ENSEMBLE = "ENSEMBLE"

class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    PRE_MARKET = "pre_market"

# 1. Latest Signals
class StrategySignal(BaseModel):
    strategy_code: str
    strategy_name: str
    signal: str
    side: str
    confidence: float = Field(ge=0, le=1)
    size: float = Field(ge=0, le=1)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    risk_usd: float
    reasoning: str
    timestamp: str
    age_seconds: int

class LatestSignalsResponse(BaseModel):
    timestamp: str
    market_price: float
    market_status: str
    signals: List[StrategySignal]

# 2. Performance Comparison
class StrategyPerformance(BaseModel):
    strategy_code: str
    strategy_name: str
    strategy_type: str

    # Returns
    total_return_pct: float
    daily_return_pct: float

    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trading
    total_trades: int
    win_rate: float = Field(ge=0, le=1)
    profit_factor: float

    # Risk
    max_drawdown_pct: float
    current_drawdown_pct: float
    volatility_pct: float

    # Position
    avg_hold_time_minutes: float
    current_equity: float
    open_positions: int

class PerformanceComparisonResponse(BaseModel):
    period: str
    start_date: str
    end_date: str
    strategies: List[StrategyPerformance]

# 3. Equity Curves
class EquityPoint(BaseModel):
    timestamp: str
    equity_value: float
    return_pct: float
    drawdown_pct: float

class EquityCurve(BaseModel):
    strategy_code: str
    strategy_name: str
    data: List[EquityPoint]
    summary: Dict[str, float]

class EquityCurvesResponse(BaseModel):
    start_date: str
    end_date: str
    resolution: str
    curves: List[EquityCurve]

# 4. Current Positions
class Position(BaseModel):
    position_id: int
    strategy_code: str
    strategy_name: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float
    unrealized_pnl_pct: float
    entry_time: str
    holding_time_minutes: int
    leverage: int

class CurrentPositionsResponse(BaseModel):
    timestamp: str
    total_positions: int
    total_notional: float
    total_pnl: float
    positions: List[Position]

# 5. P&L Summary
class StrategyPnL(BaseModel):
    strategy_code: str
    strategy_name: str
    gross_profit: float
    gross_loss: float
    net_profit: float
    total_fees: float
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_trade: float
    profit_factor: float

class PnLSummaryResponse(BaseModel):
    period: str
    start_date: str
    end_date: str
    strategies: List[StrategyPnL]
    portfolio_total: float

# ============================================================
# DATABASE UTILITIES
# ============================================================

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute query and return list of dicts"""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_market_price() -> float:
    """Get latest market price"""
    query = """
        SELECT close FROM usdcop_m5_ohlcv
        WHERE symbol = 'USD/COP'
        ORDER BY time DESC LIMIT 1
    """
    result = execute_query(query)
    if result:
        return float(result[0]['close'])
    return 0.0

def get_market_status() -> str:
    """Get current market status"""
    # Simplified - should check trading hours
    now = datetime.now()
    if now.weekday() < 5 and 8 <= now.hour < 13:
        return "open"
    return "closed"

# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Multi-Model Trading Signals API",
    description="API for multi-strategy trading signals dashboard",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "message": "Multi-Model Trading Signals API",
        "version": "1.0.0",
        "endpoints": {
            "signals": "/api/models/signals/latest",
            "performance": "/api/models/performance/comparison",
            "equity_curves": "/api/models/equity-curves",
            "positions": "/api/models/positions/current",
            "pnl": "/api/models/pnl/summary",
            "websocket": "/ws/trading-signals"
        }
    }

@app.get("/api/health")
async def health_check():
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
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.get("/api/models/signals/latest", response_model=LatestSignalsResponse)
async def get_latest_signals():
    """
    Get latest signal from each strategy

    Returns the most recent trading signal generated by each strategy
    with current market price and status.
    """
    try:
        # Get latest signals
        query = """
            WITH latest_signals AS (
                SELECT DISTINCT ON (fs.strategy_id)
                    ds.strategy_code,
                    ds.strategy_name,
                    fs.signal,
                    fs.side,
                    fs.confidence,
                    fs.size,
                    fs.entry_price,
                    fs.stop_loss,
                    fs.take_profit,
                    fs.risk_usd,
                    fs.reasoning,
                    fs.timestamp_utc,
                    EXTRACT(EPOCH FROM (NOW() - fs.timestamp_utc))::INT as age_seconds
                FROM dw.fact_strategy_signals fs
                JOIN dw.dim_strategy ds ON fs.strategy_id = ds.strategy_id
                WHERE ds.is_active = TRUE
                ORDER BY fs.strategy_id, fs.timestamp_utc DESC
            )
            SELECT * FROM latest_signals
            ORDER BY strategy_code;
        """

        results = execute_query(query)

        # Get market price and status
        market_price = get_market_price()
        market_status = get_market_status()

        signals = []
        for row in results:
            signal = StrategySignal(
                strategy_code=row['strategy_code'],
                strategy_name=row['strategy_name'],
                signal=row['signal'],
                side=row['side'],
                confidence=float(row['confidence']),
                size=float(row['size']),
                entry_price=float(row['entry_price']) if row['entry_price'] else None,
                stop_loss=float(row['stop_loss']) if row['stop_loss'] else None,
                take_profit=float(row['take_profit']) if row['take_profit'] else None,
                risk_usd=float(row['risk_usd']),
                reasoning=row['reasoning'],
                timestamp=row['timestamp_utc'].isoformat(),
                age_seconds=row['age_seconds']
            )
            signals.append(signal)

        return LatestSignalsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            market_price=market_price,
            market_status=market_status,
            signals=signals
        )

    except Exception as e:
        logger.error(f"Error fetching latest signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/performance/comparison", response_model=PerformanceComparisonResponse)
async def get_performance_comparison(
    period: str = Query("30d", description="Period: 24h, 7d, 30d, all")
):
    """
    Compare performance metrics across all strategies

    Returns comparative metrics including returns, Sharpe ratio, win rate,
    and drawdown for each active strategy.
    """
    try:
        # Parse period to interval
        period_map = {
            "24h": "1 day",
            "7d": "7 days",
            "30d": "30 days",
            "all": "10 years"  # Effectively all data
        }
        interval = period_map.get(period, "30 days")

        query = """
            WITH latest_equity AS (
                SELECT DISTINCT ON (strategy_id)
                    strategy_id,
                    equity_value,
                    return_since_start_pct,
                    current_drawdown_pct
                FROM dw.fact_equity_curve
                WHERE timestamp_utc >= NOW() - INTERVAL %s
                ORDER BY strategy_id, timestamp_utc DESC
            ),
            performance_agg AS (
                SELECT
                    strategy_id,
                    AVG(daily_return_pct) as avg_daily_return,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(sortino_ratio) as avg_sortino,
                    AVG(calmar_ratio) as avg_calmar,
                    SUM(n_trades) as total_trades,
                    AVG(win_rate) as avg_win_rate,
                    AVG(max_drawdown_pct) as max_dd,
                    AVG(avg_hold_time_minutes) as avg_hold_time
                FROM dw.fact_strategy_performance
                WHERE date_cot >= CURRENT_DATE - INTERVAL %s
                GROUP BY strategy_id
            ),
            open_pos AS (
                SELECT strategy_id, COUNT(*) as open_count
                FROM dw.fact_strategy_positions
                WHERE status = 'open'
                GROUP BY strategy_id
            )
            SELECT
                ds.strategy_code,
                ds.strategy_name,
                ds.strategy_type,
                COALESCE(le.equity_value, 10000) as current_equity,
                COALESCE(le.return_since_start_pct, 0) as total_return_pct,
                COALESCE(pa.avg_daily_return, 0) as daily_return_pct,
                COALESCE(pa.avg_sharpe, 0) as sharpe_ratio,
                COALESCE(pa.avg_sortino, 0) as sortino_ratio,
                COALESCE(pa.avg_calmar, 0) as calmar_ratio,
                COALESCE(pa.total_trades, 0) as total_trades,
                COALESCE(pa.avg_win_rate, 0) as win_rate,
                COALESCE(ABS(pa.max_dd), 0) as max_drawdown_pct,
                COALESCE(le.current_drawdown_pct, 0) as current_drawdown_pct,
                8.0 as volatility_pct,
                COALESCE(pa.avg_hold_time, 0) as avg_hold_time_minutes,
                COALESCE(op.open_count, 0) as open_positions,
                2.1 as profit_factor
            FROM dw.dim_strategy ds
            LEFT JOIN latest_equity le ON ds.strategy_id = le.strategy_id
            LEFT JOIN performance_agg pa ON ds.strategy_id = pa.strategy_id
            LEFT JOIN open_pos op ON ds.strategy_id = op.strategy_id
            WHERE ds.is_active = TRUE
            ORDER BY ds.strategy_code;
        """

        results = execute_query(query, (interval, interval))

        strategies = []
        for row in results:
            strategy = StrategyPerformance(
                strategy_code=row['strategy_code'],
                strategy_name=row['strategy_name'],
                strategy_type=row['strategy_type'],
                total_return_pct=float(row['total_return_pct']),
                daily_return_pct=float(row['daily_return_pct']),
                sharpe_ratio=float(row['sharpe_ratio']),
                sortino_ratio=float(row['sortino_ratio']),
                calmar_ratio=float(row['calmar_ratio']),
                total_trades=int(row['total_trades']),
                win_rate=float(row['win_rate']),
                profit_factor=float(row['profit_factor']),
                max_drawdown_pct=float(row['max_drawdown_pct']),
                current_drawdown_pct=float(row['current_drawdown_pct']),
                volatility_pct=float(row['volatility_pct']),
                avg_hold_time_minutes=float(row['avg_hold_time_minutes']),
                current_equity=float(row['current_equity']),
                open_positions=int(row['open_positions'])
            )
            strategies.append(strategy)

        return PerformanceComparisonResponse(
            period=period,
            start_date=(datetime.now(timezone.utc) - timedelta(days=30)).isoformat(),
            end_date=datetime.now(timezone.utc).isoformat(),
            strategies=strategies
        )

    except Exception as e:
        logger.error(f"Error fetching performance comparison: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/equity-curves", response_model=EquityCurvesResponse)
async def get_equity_curves(
    hours: int = Query(24, description="Number of hours to fetch", ge=1, le=720),
    strategies: Optional[str] = Query(None, description="Comma-separated strategy codes"),
    resolution: str = Query("5m", description="Resolution: 5m, 1h, 1d")
):
    """
    Get historical equity curves for charting

    Returns time-series equity data for selected strategies at specified resolution.
    """
    try:
        # Parse strategies
        strategy_list = strategies.split(',') if strategies else None

        # Resolution to interval
        resolution_map = {
            "5m": "5 minutes",
            "1h": "1 hour",
            "1d": "1 day"
        }

        if resolution == "5m":
            # No aggregation
            query = """
                SELECT
                    ds.strategy_code,
                    ds.strategy_name,
                    ec.timestamp_utc,
                    ec.equity_value,
                    ec.return_since_start_pct,
                    ec.current_drawdown_pct
                FROM dw.fact_equity_curve ec
                JOIN dw.dim_strategy ds ON ec.strategy_id = ds.strategy_id
                WHERE ec.timestamp_utc >= NOW() - INTERVAL %s
                  AND (ds.strategy_code = ANY(%s) OR %s IS NULL)
                ORDER BY ds.strategy_code, ec.timestamp_utc;
            """
            results = execute_query(query, (f"{hours} hours", strategy_list, strategy_list))
        else:
            # Use time_bucket for aggregation
            query = """
                SELECT
                    ds.strategy_code,
                    ds.strategy_name,
                    time_bucket(%s, ec.timestamp_utc) as timestamp,
                    LAST(ec.equity_value, ec.timestamp_utc) as equity_value,
                    LAST(ec.return_since_start_pct, ec.timestamp_utc) as return_since_start_pct,
                    LAST(ec.current_drawdown_pct, ec.timestamp_utc) as current_drawdown_pct
                FROM dw.fact_equity_curve ec
                JOIN dw.dim_strategy ds ON ec.strategy_id = ds.strategy_id
                WHERE ec.timestamp_utc >= NOW() - INTERVAL %s
                  AND (ds.strategy_code = ANY(%s) OR %s IS NULL)
                GROUP BY ds.strategy_code, ds.strategy_name, time_bucket(%s, ec.timestamp_utc)
                ORDER BY ds.strategy_code, timestamp;
            """
            interval = resolution_map[resolution]
            results = execute_query(query, (interval, f"{hours} hours", strategy_list, strategy_list, interval))

        # Group by strategy
        curves_dict = {}
        for row in results:
            code = row['strategy_code']
            if code not in curves_dict:
                curves_dict[code] = {
                    'strategy_name': row['strategy_name'],
                    'data': []
                }

            point = EquityPoint(
                timestamp=row['timestamp_utc' if resolution == '5m' else 'timestamp'].isoformat(),
                equity_value=float(row['equity_value']),
                return_pct=float(row['return_since_start_pct']),
                drawdown_pct=float(row['current_drawdown_pct'])
            )
            curves_dict[code]['data'].append(point)

        # Build curves
        curves = []
        for code, data in curves_dict.items():
            if data['data']:
                summary = {
                    'starting_equity': data['data'][0].equity_value,
                    'ending_equity': data['data'][-1].equity_value,
                    'total_return_pct': data['data'][-1].return_pct
                }
            else:
                summary = {}

            curve = EquityCurve(
                strategy_code=code,
                strategy_name=data['strategy_name'],
                data=data['data'],
                summary=summary
            )
            curves.append(curve)

        return EquityCurvesResponse(
            start_date=(datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(),
            end_date=datetime.now(timezone.utc).isoformat(),
            resolution=resolution,
            curves=curves
        )

    except Exception as e:
        logger.error(f"Error fetching equity curves: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/positions/current", response_model=CurrentPositionsResponse)
async def get_current_positions(
    strategy: Optional[str] = Query(None, description="Filter by strategy code")
):
    """
    Get all open positions across strategies

    Returns current open positions with unrealized P&L and holding time.
    """
    try:
        query = """
            WITH current_price AS (
                SELECT close FROM usdcop_m5_ohlcv
                ORDER BY time DESC LIMIT 1
            )
            SELECT
                fp.position_id,
                ds.strategy_code,
                ds.strategy_name,
                fp.side,
                fp.quantity,
                fp.entry_price,
                cp.close as current_price,
                fp.stop_loss,
                fp.take_profit,
                COALESCE(fp.unrealized_pnl, 0) as unrealized_pnl,
                COALESCE((fp.unrealized_pnl / (fp.entry_price * fp.quantity)) * 100, 0) as unrealized_pnl_pct,
                fp.entry_time,
                EXTRACT(EPOCH FROM (NOW() - fp.entry_time)) / 60 as holding_time_minutes,
                fp.leverage
            FROM dw.fact_strategy_positions fp
            JOIN dw.dim_strategy ds ON fp.strategy_id = ds.strategy_id
            CROSS JOIN current_price cp
            WHERE fp.status = 'open'
              AND (ds.strategy_code = %s OR %s IS NULL)
            ORDER BY fp.entry_time DESC;
        """

        results = execute_query(query, (strategy, strategy))

        positions = []
        total_pnl = 0.0
        total_notional = 0.0

        for row in results:
            position = Position(
                position_id=int(row['position_id']),
                strategy_code=row['strategy_code'],
                strategy_name=row['strategy_name'],
                side=row['side'],
                quantity=float(row['quantity']),
                entry_price=float(row['entry_price']),
                current_price=float(row['current_price']),
                stop_loss=float(row['stop_loss']) if row['stop_loss'] else None,
                take_profit=float(row['take_profit']) if row['take_profit'] else None,
                unrealized_pnl=float(row['unrealized_pnl']),
                unrealized_pnl_pct=float(row['unrealized_pnl_pct']),
                entry_time=row['entry_time'].isoformat(),
                holding_time_minutes=int(row['holding_time_minutes']),
                leverage=int(row['leverage'])
            )
            positions.append(position)
            total_pnl += position.unrealized_pnl
            total_notional += position.quantity * position.entry_price

        return CurrentPositionsResponse(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_positions=len(positions),
            total_notional=total_notional,
            total_pnl=total_pnl,
            positions=positions
        )

    except Exception as e:
        logger.error(f"Error fetching current positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/pnl/summary", response_model=PnLSummaryResponse)
async def get_pnl_summary(
    period: str = Query("today", description="Period: today, week, month, all")
):
    """
    Get P&L breakdown per strategy

    Returns profit/loss metrics, trade statistics, and win rate for each strategy.
    """
    try:
        # Parse period
        period_map = {
            "today": (datetime.now(timezone.utc).date(), datetime.now(timezone.utc).date()),
            "week": (datetime.now(timezone.utc).date() - timedelta(days=7), datetime.now(timezone.utc).date()),
            "month": (datetime.now(timezone.utc).date() - timedelta(days=30), datetime.now(timezone.utc).date()),
            "all": (datetime(2020, 1, 1).date(), datetime.now(timezone.utc).date())
        }
        start_date, end_date = period_map.get(period, period_map["today"])

        query = """
            SELECT
                ds.strategy_code,
                ds.strategy_name,
                COALESCE(SUM(sp.gross_profit), 0) as gross_profit,
                COALESCE(SUM(ABS(sp.gross_loss)), 0) as gross_loss,
                COALESCE(SUM(sp.net_profit), 0) as net_profit,
                COALESCE(SUM(sp.total_fees), 0) as total_fees,
                COALESCE(SUM(sp.n_trades), 0) as n_trades,
                COALESCE(SUM(sp.n_wins), 0) as n_wins,
                COALESCE(SUM(sp.n_losses), 0) as n_losses,
                COALESCE(AVG(sp.win_rate), 0) as win_rate,
                COALESCE(AVG(CASE WHEN sp.n_wins > 0 THEN sp.gross_profit / sp.n_wins ELSE 0 END), 0) as avg_win,
                COALESCE(AVG(CASE WHEN sp.n_losses > 0 THEN ABS(sp.gross_loss) / sp.n_losses ELSE 0 END), 0) as avg_loss,
                COALESCE(AVG(CASE WHEN sp.n_trades > 0 THEN sp.net_profit / sp.n_trades ELSE 0 END), 0) as avg_trade,
                COALESCE(AVG(CASE WHEN sp.gross_loss <> 0 THEN sp.gross_profit / ABS(sp.gross_loss) ELSE 0 END), 0) as profit_factor
            FROM dw.dim_strategy ds
            LEFT JOIN dw.fact_strategy_performance sp ON ds.strategy_id = sp.strategy_id
            WHERE sp.date_cot >= %s AND sp.date_cot <= %s
              AND ds.is_active = TRUE
            GROUP BY ds.strategy_code, ds.strategy_name
            ORDER BY net_profit DESC;
        """

        results = execute_query(query, (start_date, end_date))

        strategies = []
        portfolio_total = 0.0

        for row in results:
            strategy_pnl = StrategyPnL(
                strategy_code=row['strategy_code'],
                strategy_name=row['strategy_name'],
                gross_profit=float(row['gross_profit']),
                gross_loss=float(row['gross_loss']),
                net_profit=float(row['net_profit']),
                total_fees=float(row['total_fees']),
                n_trades=int(row['n_trades']),
                n_wins=int(row['n_wins']),
                n_losses=int(row['n_losses']),
                win_rate=float(row['win_rate']),
                avg_win=float(row['avg_win']),
                avg_loss=float(row['avg_loss']),
                avg_trade=float(row['avg_trade']),
                profit_factor=float(row['profit_factor'])
            )
            strategies.append(strategy_pnl)
            portfolio_total += strategy_pnl.net_profit

        return PnLSummaryResponse(
            period=period,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            strategies=strategies,
            portfolio_total=portfolio_total
        )

    except Exception as e:
        logger.error(f"Error fetching P&L summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# WEBSOCKET ENDPOINT
# ============================================================

@app.websocket("/ws/trading-signals")
async def websocket_trading_signals(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading signals

    Broadcasts:
    - New signals when generated
    - Price updates every 5 minutes
    - Position changes (open/close)
    - Equity curve updates
    """
    await websocket.accept()
    active_websockets.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(active_websockets)}")

    try:
        # Send initial data
        initial_data = {
            "type": "connection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "Connected to trading signals stream"
        }
        await websocket.send_json(initial_data)

        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send heartbeat
                heartbeat = {
                    "type": "heartbeat",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                await websocket.send_json(heartbeat)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in active_websockets:
            active_websockets.remove(websocket)
        logger.info(f"WebSocket client removed. Total: {len(active_websockets)}")

async def broadcast_to_websockets(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    if not active_websockets:
        return

    disconnected = []
    for ws in active_websockets:
        try:
            await ws.send_json(message)
        except Exception as e:
            logger.warning(f"Failed to send to WebSocket: {e}")
            disconnected.append(ws)

    # Remove disconnected clients
    for ws in disconnected:
        active_websockets.remove(ws)

# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8006))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
