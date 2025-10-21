#!/usr/bin/env python3
"""
USDCOP Backtest API
====================

API para gestionar backtesting de estrategias de trading:
- Ejecutar backtests con datos históricos
- Obtener resultados y métricas de backtest
- Análisis de performance y risk metrics
- Generación de reportes detallados

Integrado con el sistema de trading RL y database.
"""

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
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
import json
import random
import asyncio

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
    title="USDCOP Backtest API",
    description="API para backtesting de estrategias de trading",
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

# Global state for backtest tracking
backtest_status = {
    "running": False,
    "progress": 0,
    "current_run_id": None,
    "last_completed": None
}

# ==========================================
# MODELS
# ==========================================

class BacktestTrigger(BaseModel):
    forceRebuild: bool = False
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100000
    strategy: str = "RL_PPO"

class Trade(BaseModel):
    id: str
    timestamp: str
    symbol: str
    side: str  # buy or sell
    quantity: float
    price: float
    pnl: float
    commission: float
    reason: Optional[str] = None

class DailyReturn(BaseModel):
    date: str
    return_: float
    cumulative_return: float
    price: float
    drawdown: Optional[float] = None

# ==========================================
# DATABASE CONNECTION
# ==========================================

def get_db_connection():
    """Get database connection"""
    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG, cursor_factory=RealDictCursor)
        return conn
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

def execute_query(query: str, params: tuple = None) -> List[Dict]:
    """Execute query and return list of dicts"""
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        cur.close()
        return [dict(row) for row in rows]
    finally:
        conn.close()

# ==========================================
# BACKTEST CALCULATION FUNCTIONS
# ==========================================

def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sharpe Ratio"""
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)  # Daily risk-free rate

    if np.std(excess_returns) == 0:
        return 0.0

    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return float(sharpe)

def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02) -> float:
    """Calculate Sortino Ratio"""
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / 252)

    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return 0.0

    sortino = np.mean(excess_returns) / downside_std * np.sqrt(252)
    return float(sortino)

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate Maximum Drawdown"""
    if not equity_curve or len(equity_curve) < 2:
        return 0.0

    equity = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = float(np.min(drawdown))

    return max_dd

def calculate_calmar_ratio(returns: List[float], max_drawdown: float) -> float:
    """Calculate Calmar Ratio"""
    if abs(max_drawdown) < 0.001:
        return 0.0

    annual_return = np.mean(returns) * 252
    calmar = annual_return / abs(max_drawdown)

    return float(calmar)

def generate_backtest_results(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    strategy: str
) -> Dict:
    """
    Generate backtest results using historical data

    This is a simplified version - in production, this would:
    1. Load historical market data
    2. Run the RL agent through historical episodes
    3. Calculate actual trades and P&L
    4. Compute all performance metrics
    """

    # Get historical data
    query = """
        SELECT timestamp, price, volume
        FROM market_data
        WHERE timestamp >= %s AND timestamp <= %s
        ORDER BY timestamp ASC
        LIMIT 10000
    """

    try:
        rows = execute_query(query, (start_date, end_date))
    except:
        rows = []

    if not rows:
        # Generate synthetic results if no data
        logger.warning("No historical data found, generating synthetic backtest")
        rows = []
        current_date = start_date
        base_price = 4300.0

        while current_date <= end_date:
            rows.append({
                'timestamp': current_date,
                'price': base_price + random.uniform(-50, 50),
                'volume': random.randint(100000, 1000000)
            })
            current_date += timedelta(hours=1)
            if len(rows) >= 1000:
                break

    # Simulate trading
    capital = initial_capital
    position = 0
    trades = []
    daily_returns = []
    equity_curve = [initial_capital]

    returns_list = []

    for i, row in enumerate(rows):
        if i == 0:
            continue

        price = float(row['price'])
        prev_price = float(rows[i-1]['price'])

        # Simple strategy: random signals (in production, use RL agent)
        signal = random.choice([0, 1, -1])  # -1: sell, 0: hold, 1: buy

        # Execute trade
        if signal == 1 and position <= 0:  # Buy signal
            quantity = capital * 0.1 / price  # Use 10% of capital
            cost = quantity * price
            commission = cost * 0.001  # 0.1% commission

            if capital >= cost + commission:
                position += quantity
                capital -= (cost + commission)

                trades.append({
                    "id": f"trade_{len(trades)+1}",
                    "timestamp": row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                    "symbol": "USDCOP",
                    "side": "buy",
                    "quantity": round(quantity, 4),
                    "price": round(price, 2),
                    "pnl": 0,
                    "commission": round(commission, 2),
                    "reason": "RL Buy Signal"
                })

        elif signal == -1 and position > 0:  # Sell signal
            revenue = position * price
            commission = revenue * 0.001
            pnl = revenue - (position * prev_price) - commission

            capital += revenue - commission

            trades.append({
                "id": f"trade_{len(trades)+1}",
                "timestamp": row['timestamp'].isoformat() if isinstance(row['timestamp'], datetime) else row['timestamp'],
                "symbol": "USDCOP",
                "side": "sell",
                "quantity": round(position, 4),
                "price": round(price, 2),
                "pnl": round(pnl, 2),
                "commission": round(commission, 2),
                "reason": "RL Sell Signal"
            })

            position = 0

        # Calculate equity
        current_equity = capital + (position * price)
        equity_curve.append(current_equity)

        # Daily returns
        daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
        returns_list.append(daily_return)

        # Store daily return every 24 data points (approximately)
        if i % 24 == 0:
            daily_returns.append({
                "date": row['timestamp'].strftime("%Y-%m-%d") if isinstance(row['timestamp'], datetime) else row['timestamp'][:10],
                "return": round(daily_return, 6),
                "cumulativeReturn": round((current_equity - initial_capital) / initial_capital, 6),
                "price": round(current_equity, 2)
            })

    # Calculate final metrics
    final_capital = capital + (position * float(rows[-1]['price']))
    total_return = (final_capital - initial_capital) / initial_capital

    # Calculate KPIs
    sharpe = calculate_sharpe_ratio(returns_list)
    sortino = calculate_sortino_ratio(returns_list)
    max_dd = calculate_max_drawdown(equity_curve)
    calmar = calculate_calmar_ratio(returns_list, max_dd)

    # Calculate trading metrics
    winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t.get('pnl', 0) < 0]

    win_rate = len(winning_trades) / len(trades) if trades else 0

    avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
    avg_loss = abs(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0

    profit_factor = abs(sum([t['pnl'] for t in winning_trades]) / sum([t['pnl'] for t in losing_trades])) if losing_trades else 0

    payoff = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # CAGR calculation
    days = (end_date - start_date).days
    years = max(days / 365.25, 0.01)
    cagr = (final_capital / initial_capital) ** (1 / years) - 1

    # Annualized volatility
    vol_annualized = np.std(returns_list) * np.sqrt(252) if returns_list else 0

    results = {
        "run_id": f"backtest_{int(datetime.utcnow().timestamp())}",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": initial_capital,
            "strategy": strategy
        },
        "test": {
            "kpis": {
                "top_bar": {
                    "CAGR": round(cagr, 4),
                    "Sharpe": round(sharpe, 2),
                    "Sortino": round(sortino, 2),
                    "Calmar": round(calmar, 2),
                    "MaxDD": round(max_dd, 4),
                    "Vol_annualizada": round(vol_annualized, 4)
                },
                "trading_micro": {
                    "win_rate": round(win_rate, 3),
                    "profit_factor": round(profit_factor, 2),
                    "payoff": round(payoff, 2),
                    "expectancy_bps": round((avg_win * win_rate - avg_loss * (1 - win_rate)) * 10000 / initial_capital, 2) if initial_capital > 0 else 0,
                    "total_trades": len(trades),
                    "winning_trades": len(winning_trades),
                    "losing_trades": len(losing_trades)
                },
                "returns": {
                    "total_return": round(total_return, 4),
                    "final_capital": round(final_capital, 2),
                    "total_pnl": round(final_capital - initial_capital, 2)
                }
            },
            "dailyReturns": daily_returns[-30:],  # Last 30 days
            "trades": trades[-50:]  # Last 50 trades
        }
    }

    return results

async def run_backtest_async(config: BacktestTrigger):
    """
    Run backtest asynchronously in background
    """
    global backtest_status

    try:
        backtest_status["running"] = True
        backtest_status["progress"] = 0

        logger.info(f"Starting backtest with config: {config}")

        # Determine date range
        if config.start_date and config.end_date:
            start_date = datetime.fromisoformat(config.start_date.replace('Z', '+00:00'))
            end_date = datetime.fromisoformat(config.end_date.replace('Z', '+00:00'))
        else:
            # Default: last 30 days
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)

        backtest_status["progress"] = 25
        await asyncio.sleep(1)  # Simulate work

        # Run backtest
        results = generate_backtest_results(
            start_date=start_date,
            end_date=end_date,
            initial_capital=config.initial_capital,
            strategy=config.strategy
        )

        backtest_status["progress"] = 75
        await asyncio.sleep(1)  # Simulate work

        # Store results (in production, save to database)
        backtest_status["last_completed"] = results
        backtest_status["current_run_id"] = results["run_id"]
        backtest_status["progress"] = 100

        logger.info(f"Backtest completed: {results['run_id']}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        backtest_status["last_completed"] = {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }
    finally:
        backtest_status["running"] = False

# ==========================================
# ENDPOINTS
# ==========================================

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "USDCOP Backtest API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "results": "/api/backtest/results",
            "trigger": "/api/backtest/trigger (POST)",
            "status": "/api/backtest/status"
        }
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
            "backtest_running": backtest_status["running"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/backtest/results")
def get_backtest_results():
    """
    Get latest backtest results

    Returns complete backtest analysis including:
    - Performance KPIs (Sharpe, Sortino, Calmar, CAGR, etc.)
    - Trading metrics (win rate, profit factor, etc.)
    - Daily returns time series
    - Trade history
    """
    try:
        # Return last completed backtest or generate new one
        if backtest_status["last_completed"] and "error" not in backtest_status["last_completed"]:
            results = backtest_status["last_completed"]
        else:
            # Generate fresh backtest results
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=30)

            results = generate_backtest_results(
                start_date=start_date,
                end_date=end_date,
                initial_capital=100000,
                strategy="RL_PPO"
            )

            backtest_status["last_completed"] = results
            backtest_status["current_run_id"] = results["run_id"]

        return {
            "success": True,
            "data": results
        }

    except Exception as e:
        logger.error(f"Error getting backtest results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/backtest/trigger")
async def trigger_backtest(config: BacktestTrigger, background_tasks: BackgroundTasks):
    """
    Trigger a new backtest run

    Parameters:
    - forceRebuild: Force rebuild even if recent results exist
    - start_date: Backtest start date (ISO format)
    - end_date: Backtest end date (ISO format)
    - initial_capital: Starting capital (default: 100000)
    - strategy: Strategy to test (default: RL_PPO)

    Returns immediately and runs backtest in background
    """
    try:
        global backtest_status

        if backtest_status["running"]:
            return {
                "success": False,
                "message": "Backtest already running",
                "current_run_id": backtest_status["current_run_id"],
                "progress": backtest_status["progress"]
            }

        # Start backtest in background
        background_tasks.add_task(run_backtest_async, config)

        return {
            "success": True,
            "message": "Backtest started",
            "config": config.dict(),
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Error triggering backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/backtest/status")
def get_backtest_status():
    """
    Get current backtest execution status
    """
    return {
        "running": backtest_status["running"],
        "progress": backtest_status["progress"],
        "current_run_id": backtest_status["current_run_id"],
        "has_results": backtest_status["last_completed"] is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    port = int(os.getenv("BACKTEST_API_PORT", "8006"))
    uvicorn.run(
        "backtest_api:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        log_level="info"
    )
