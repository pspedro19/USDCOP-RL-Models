"""
Optimized L0 Validator FastAPI Service
=====================================

FastAPI wrapper for the optimized L0 validator with health endpoints
and real-time status monitoring.

Author: USDCOP Trading Team
Version: 3.0.0
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional

import asyncpg
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import uvicorn
import pytz

from optimized_l0_validator import (
    OptimizedL0Validator,
    RealtimeWebSocketManager,
    APIKeyManager,
    MarketHoursValidator,
    ValidationSeverity
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Colombia timezone
COT_TZ = pytz.timezone('America/Bogota')

# Global service instances
validator: Optional[OptimizedL0Validator] = None
websocket_manager: Optional[RealtimeWebSocketManager] = None
db_pool: Optional[asyncpg.Pool] = None
redis_client: Optional[redis.Redis] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown"""
    global validator, websocket_manager, db_pool, redis_client

    logger.info("🚀 Starting Optimized L0 Validator Service")

    try:
        # Initialize database connection
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required")

        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )

        # Test database connection
        async with db_pool.acquire() as conn:
            await conn.fetchval('SELECT 1')
        logger.info("✅ Database connection established")

        # Initialize Redis
        redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        redis_client = redis.from_url(redis_url, decode_responses=True)
        redis_client.ping()
        logger.info("✅ Redis connection established")

        # Initialize validator
        validator = OptimizedL0Validator(db_pool, redis_client)
        await validator.initialize()
        logger.info("✅ L0 Validator initialized")

        # Initialize WebSocket manager
        websocket_manager = RealtimeWebSocketManager(db_pool, redis_client, validator.api_manager)
        logger.info("✅ WebSocket Manager initialized")

        # Start background tasks
        asyncio.create_task(background_market_monitor())
        logger.info("✅ Background tasks started")

        yield

    except Exception as e:
        logger.error(f"❌ Failed to initialize service: {e}")
        raise
    finally:
        # Cleanup
        logger.info("🔌 Shutting down Optimized L0 Validator Service")

        if validator:
            await validator.close()

        if db_pool:
            await db_pool.close()

        if redis_client:
            redis_client.close()

app = FastAPI(
    title="Optimized L0 Validator API",
    description="High-performance L0 data validation service with real-time WebSocket integration",
    version="3.0.0",
    lifespan=lifespan
)

async def background_market_monitor():
    """Background task to monitor market hours and manage real-time collection"""
    market_validator = MarketHoursValidator()

    while True:
        try:
            current_time = datetime.now(COT_TZ)
            market_open = market_validator.is_market_open(current_time)

            # Update market status in Redis
            redis_client.setex(
                "market:status",
                60,  # 1 minute TTL
                json.dumps({
                    'is_open': market_open,
                    'timestamp': current_time.isoformat(),
                    'next_check': (current_time + timedelta(minutes=5)).isoformat()
                })
            )

            # Manage WebSocket collection
            if market_open and websocket_manager and not websocket_manager.is_collecting:
                logger.info("🔴 Market opened - starting real-time collection")
                asyncio.create_task(websocket_manager.start_realtime_collection())
            elif not market_open and websocket_manager and websocket_manager.is_collecting:
                logger.info("🔘 Market closed - stopping real-time collection")
                websocket_manager.is_collecting = False

            # Wait 5 minutes before next check
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Error in background market monitor: {e}")
            await asyncio.sleep(60)  # Wait 1 minute on error

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        health_status = {
            "status": "healthy",
            "service": "optimized-l0-validator",
            "version": "3.0.0",
            "timestamp": datetime.now(COT_TZ).isoformat()
        }

        # Check database connection
        if db_pool:
            try:
                async with db_pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')
                health_status["database"] = "connected"
            except Exception as e:
                health_status["database"] = f"error: {str(e)}"
                health_status["status"] = "unhealthy"
        else:
            health_status["database"] = "not_initialized"
            health_status["status"] = "unhealthy"

        # Check Redis connection
        if redis_client:
            try:
                redis_client.ping()
                health_status["redis"] = "connected"
            except Exception as e:
                health_status["redis"] = f"error: {str(e)}"
                health_status["status"] = "unhealthy"
        else:
            health_status["redis"] = "not_initialized"
            health_status["status"] = "unhealthy"

        # Check API key availability
        if validator and validator.api_manager:
            api_stats = validator.api_manager.get_usage_stats()
            health_status["api_keys"] = {
                "total_keys": sum(len(group.keys) for group in validator.api_manager.groups),
                "active_group": api_stats.get("active_group"),
                "total_calls": api_stats.get("total_calls", 0)
            }
        else:
            health_status["api_keys"] = "not_initialized"

        # Check market status
        market_validator = MarketHoursValidator()
        current_time = datetime.now(COT_TZ)
        health_status["market"] = {
            "is_open": market_validator.is_market_open(current_time),
            "current_time_cot": current_time.isoformat()
        }

        # Check WebSocket status
        if websocket_manager:
            health_status["websocket"] = {
                "collecting": websocket_manager.is_collecting,
                "connected": websocket_manager.websocket_connection is not None
            }
        else:
            health_status["websocket"] = "not_initialized"

        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(COT_TZ).isoformat()
            },
            status_code=503
        )

@app.get("/status")
async def get_detailed_status():
    """Get detailed service status"""
    if not validator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Get API usage statistics
        api_stats = validator.api_manager.get_usage_stats()

        # Get recent validation results
        latest_validation = redis_client.get("l0_validation:latest")
        validation_data = json.loads(latest_validation) if latest_validation else None

        # Get market status
        market_status = redis_client.get("market:status")
        market_data = json.loads(market_status) if market_status else None

        # Get WebSocket status
        ws_status = redis_client.get("websocket:usdcop:status")
        ws_data = json.loads(ws_status) if ws_status else None

        return {
            "service_status": "active",
            "timestamp": datetime.now(COT_TZ).isoformat(),
            "api_usage": api_stats,
            "latest_validation": validation_data,
            "market_status": market_data,
            "websocket_status": ws_data,
            "memory_usage": {
                "db_pool_size": db_pool.get_size() if db_pool else 0,
                "db_pool_idle": db_pool.get_idle_size() if db_pool else 0
            }
        }

    except Exception as e:
        logger.error(f"Error getting detailed status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/validate/historical")
async def run_historical_validation(
    background_tasks: BackgroundTasks,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Run historical data validation"""
    if not validator:
        raise HTTPException(status_code=503, detail="Validator not initialized")

    try:
        # Parse dates or use defaults
        if start_date:
            start_dt = date.fromisoformat(start_date)
        else:
            start_dt = date(2020, 1, 1)

        if end_date:
            end_dt = date.fromisoformat(end_date)
        else:
            end_dt = date(2025, 12, 31)

        # Run validation in background
        background_tasks.add_task(
            run_validation_task,
            start_dt,
            end_dt
        )

        return {
            "message": "Historical validation started",
            "date_range": {
                "start": start_dt.isoformat(),
                "end": end_dt.isoformat()
            },
            "status": "running",
            "timestamp": datetime.now(COT_TZ).isoformat()
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {str(e)}")
    except Exception as e:
        logger.error(f"Error starting historical validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_validation_task(start_date: date, end_date: date):
    """Background task to run validation"""
    try:
        logger.info(f"Starting historical validation: {start_date} to {end_date}")

        # Update status in Redis
        redis_client.setex(
            "validation:status",
            3600,  # 1 hour TTL
            json.dumps({
                "status": "running",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "started_at": datetime.now(COT_TZ).isoformat()
            })
        )

        # Run validation
        report = await validator.validate_historical_data(start_date, end_date)

        # Update completion status
        redis_client.setex(
            "validation:status",
            3600,
            json.dumps({
                "status": "completed",
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "started_at": report.get("validation_start"),
                "completed_at": report.get("validation_end"),
                "results": report.get("data_quality", {})
            })
        )

        logger.info(f"Historical validation completed: {report['data_quality']['completeness_percentage']:.1f}% complete")

    except Exception as e:
        logger.error(f"Validation task error: {e}")

        # Update error status
        redis_client.setex(
            "validation:status",
            3600,
            json.dumps({
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(COT_TZ).isoformat()
            })
        )

@app.get("/validate/status")
async def get_validation_status():
    """Get current validation status"""
    try:
        status_data = redis_client.get("validation:status")
        if status_data:
            return json.loads(status_data)
        else:
            return {
                "status": "idle",
                "message": "No validation running"
            }
    except Exception as e:
        logger.error(f"Error getting validation status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/latest")
async def get_latest_market_data():
    """Get latest cached market data"""
    try:
        latest_data = redis_client.get("usdcop:latest_price")
        if latest_data:
            return json.loads(latest_data)
        else:
            # Fallback to database query
            if db_pool:
                async with db_pool.acquire() as conn:
                    result = await conn.fetchrow("""
                        SELECT symbol, time, bid, ask, last, volume, source
                        FROM realtime_market_data_optimized
                        WHERE symbol = 'USDCOP'
                        ORDER BY time DESC
                        LIMIT 1
                    """)

                    if result:
                        return {
                            "symbol": result["symbol"],
                            "price": float(result["last"]),
                            "bid": float(result["bid"]) if result["bid"] else None,
                            "ask": float(result["ask"]) if result["ask"] else None,
                            "volume": result["volume"],
                            "timestamp": result["time"].isoformat(),
                            "source": result["source"]
                        }

            raise HTTPException(status_code=404, detail="No market data available")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest market data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/market/session")
async def get_market_session_info():
    """Get current market session information"""
    try:
        market_validator = MarketHoursValidator()
        current_time = datetime.now(COT_TZ)

        return {
            "current_time": current_time.isoformat(),
            "is_market_open": market_validator.is_market_open(current_time),
            "market_hours": {
                "start": f"{os.getenv('MARKET_START_HOUR', 8):02d}:{os.getenv('MARKET_START_MINUTE', 0):02d}",
                "end": f"{os.getenv('MARKET_END_HOUR', 12):02d}:{os.getenv('MARKET_END_MINUTE', 55):02d}"
            },
            "timezone": "America/Bogota",
            "today_session": current_time.date().isoformat(),
            "is_trading_day": current_time.weekday() < 5  # Monday=0 to Friday=4
        }

    except Exception as e:
        logger.error(f"Error getting market session info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/usage")
async def get_api_usage_stats():
    """Get API key usage statistics"""
    if not validator or not validator.api_manager:
        raise HTTPException(status_code=503, detail="API Manager not initialized")

    try:
        stats = validator.api_manager.get_usage_stats()
        return {
            "api_usage": stats,
            "timestamp": datetime.now(COT_TZ).isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting API usage stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/quality/summary")
async def get_data_quality_summary():
    """Get data quality summary for recent days"""
    if not db_pool:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        async with db_pool.acquire() as conn:
            # Get quality summary for last 30 days
            results = await conn.fetch("""
                SELECT * FROM daily_data_quality_summary
                WHERE trading_date >= CURRENT_DATE - INTERVAL '30 days'
                ORDER BY trading_date DESC
                LIMIT 30
            """)

            quality_data = []
            for row in results:
                quality_data.append({
                    "trading_date": row["trading_date"].isoformat(),
                    "total_bars": row["total_bars"],
                    "expected_bars": row["expected_bars"],
                    "completeness_percent": float(row["completeness_percent"]),
                    "quality_status": row["quality_status"],
                    "first_bar": row["first_bar"].isoformat() if row["first_bar"] else None,
                    "last_bar": row["last_bar"].isoformat() if row["last_bar"] else None,
                    "data_sources": row["data_sources"]
                })

            return {
                "summary": quality_data,
                "total_days": len(quality_data),
                "timestamp": datetime.now(COT_TZ).isoformat()
            }

    except Exception as e:
        logger.error(f"Error getting data quality summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/websocket/start")
async def start_websocket_collection():
    """Manually start WebSocket data collection"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")

    try:
        market_validator = MarketHoursValidator()
        if not market_validator.is_market_open(datetime.now(COT_TZ)):
            raise HTTPException(status_code=400, detail="Market is currently closed")

        if websocket_manager.is_collecting:
            return {"message": "WebSocket collection already running"}

        # Start collection in background
        asyncio.create_task(websocket_manager.start_realtime_collection())

        return {
            "message": "WebSocket collection started",
            "timestamp": datetime.now(COT_TZ).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting WebSocket collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/websocket/stop")
async def stop_websocket_collection():
    """Manually stop WebSocket data collection"""
    if not websocket_manager:
        raise HTTPException(status_code=503, detail="WebSocket manager not initialized")

    try:
        if not websocket_manager.is_collecting:
            return {"message": "WebSocket collection not running"}

        websocket_manager.is_collecting = False

        return {
            "message": "WebSocket collection stopped",
            "timestamp": datetime.now(COT_TZ).isoformat()
        }

    except Exception as e:
        logger.error(f"Error stopping WebSocket collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8086))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info"
    )