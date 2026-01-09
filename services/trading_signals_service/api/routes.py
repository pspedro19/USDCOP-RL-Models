"""
API Routes
===========
FastAPI routes for trading signals service.

Author: Pedro @ Lean Tech Solutions
Created: 2025-12-17
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import JSONResponse
import psycopg2
import pandas as pd

from ..models.signal_schema import (
    TradingSignal,
    SignalResponse,
    SignalHistoryResponse,
    GenerateSignalRequest,
    SignalAction,
    HealthCheckResponse,
    ErrorResponse
)
from ..config import get_config
from ..utils.technical_indicators import TechnicalIndicators

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/signals", tags=["signals"])

# Global service instances (will be injected by main.py)
_inference_service = None
_signal_generator = None
_position_manager = None
_service_start_time = datetime.utcnow()


def set_services(inference_service, signal_generator, position_manager):
    """Set global service instances (called from main.py)"""
    global _inference_service, _signal_generator, _position_manager
    _inference_service = inference_service
    _signal_generator = signal_generator
    _position_manager = position_manager


def get_db_connection():
    """Get database connection"""
    config = get_config()
    try:
        return psycopg2.connect(**config.get_db_config())
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection failed")


@router.get("/latest", response_model=SignalResponse)
async def get_latest_signal():
    """Get the latest trading signal"""
    try:
        if _position_manager is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        latest_signal = _position_manager.get_latest_signal()

        if latest_signal is None:
            raise HTTPException(status_code=404, detail="No signals available")

        return SignalResponse(
            status="success",
            signal=latest_signal,
            message="Latest signal retrieved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest signal: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=SignalHistoryResponse)
async def get_signal_history(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of signals"),
    action: Optional[SignalAction] = Query(None, description="Filter by action type"),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)")
):
    """Get signal history with optional filtering"""
    try:
        if _position_manager is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Get signals
        signals = _position_manager.get_signal_history(
            limit=limit,
            action_filter=action
        )

        # Apply date filtering if specified
        if start_date or end_date:
            start_dt = datetime.fromisoformat(start_date) if start_date else datetime.min
            end_dt = datetime.fromisoformat(end_date) if end_date else datetime.max

            signals = [
                s for s in signals
                if start_dt <= s.timestamp <= end_dt
            ]

        return SignalHistoryResponse(
            status="success",
            signals=signals,
            count=len(signals),
            start_date=datetime.fromisoformat(start_date) if start_date else None,
            end_date=datetime.fromisoformat(end_date) if end_date else None,
            filters={'action': action.value if action else None, 'limit': limit}
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")
    except Exception as e:
        logger.error(f"Error getting signal history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate", response_model=SignalResponse)
async def generate_signal(request: GenerateSignalRequest):
    """Generate a new trading signal from provided market data"""
    try:
        if _signal_generator is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Prepare market data
        market_data = {
            'symbol': request.symbol,
            'close': request.close_price,
            'open': request.open_price or request.close_price,
            'high': request.high_price or request.close_price,
            'low': request.low_price or request.close_price,
            'volume': request.volume or 0.0
        }

        # Add pre-calculated indicators if provided
        technical_indicators = {}
        if request.rsi is not None:
            technical_indicators['rsi'] = request.rsi
        if request.macd is not None:
            technical_indicators['macd'] = request.macd
        if request.macd_signal is not None:
            technical_indicators['macd_signal'] = request.macd_signal
        if request.bb_upper is not None:
            technical_indicators['bb_upper'] = request.bb_upper
        if request.bb_lower is not None:
            technical_indicators['bb_lower'] = request.bb_lower

        # Generate signal
        signal = _signal_generator.generate_signal(
            market_data=market_data,
            technical_indicators=technical_indicators if technical_indicators else None
        )

        if signal is None:
            raise HTTPException(status_code=500, detail="Failed to generate signal")

        # Override for testing if specified
        if request.force_action:
            signal.action = request.force_action
        if request.override_confidence is not None:
            signal.confidence = request.override_confidence

        # Add to position manager
        _position_manager.add_signal(signal)

        return SignalResponse(
            status="success",
            signal=signal,
            message="Signal generated successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-from-db", response_model=SignalResponse)
async def generate_signal_from_db(
    symbol: str = Query("USDCOP", description="Trading symbol"),
    lookback_bars: int = Query(100, ge=50, le=500, description="Number of historical bars for indicators")
):
    """Generate signal using latest data from database"""
    try:
        if _signal_generator is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        # Fetch recent data from database
        conn = get_db_connection()
        db_symbol = 'USD/COP' if symbol == 'USDCOP' else symbol

        query = """
        SELECT time, open, high, low, close, COALESCE(volume, 0) as volume
        FROM usdcop_m5_ohlcv
        WHERE symbol = %s
        ORDER BY time DESC
        LIMIT %s
        """

        df = pd.read_sql_query(query, conn, params=(db_symbol, lookback_bars))
        conn.close()

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")

        # Reverse to chronological order
        df = df.iloc[::-1].reset_index(drop=True)

        # Calculate technical indicators
        indicators = TechnicalIndicators.calculate_all(df)

        # Prepare market data
        latest = df.iloc[-1]
        market_data = {
            'symbol': symbol,
            'close': float(latest['close']),
            'open': float(latest['open']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'volume': float(latest['volume'])
        }

        # Generate signal
        signal = _signal_generator.generate_signal(
            market_data=market_data,
            technical_indicators=indicators
        )

        if signal is None:
            raise HTTPException(status_code=500, detail="Failed to generate signal")

        # Add to position manager
        _position_manager.add_signal(signal)

        return SignalResponse(
            status="success",
            signal=signal,
            message="Signal generated from database successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating signal from DB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/active")
async def get_active_positions():
    """Get all active positions"""
    try:
        if _position_manager is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        positions = _position_manager.get_active_positions()

        return {
            "status": "success",
            "count": len(positions),
            "positions": positions
        }

    except Exception as e:
        logger.error(f"Error getting active positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions/closed")
async def get_closed_positions(
    limit: int = Query(50, ge=1, le=500, description="Maximum number of positions")
):
    """Get closed positions history"""
    try:
        if _position_manager is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        positions = _position_manager.get_closed_positions(limit=limit)

        return {
            "status": "success",
            "count": len(positions),
            "positions": positions
        }

    except Exception as e:
        logger.error(f"Error getting closed positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics")
async def get_statistics():
    """Get signal and position statistics"""
    try:
        if _signal_generator is None or _position_manager is None or _inference_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        stats = {
            "inference": _inference_service.get_stats(),
            "signals": _signal_generator.get_stats(),
            "positions": _position_manager.get_statistics()
        }

        return {
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "statistics": stats
        }

    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """Get model information"""
    try:
        if _inference_service is None:
            raise HTTPException(status_code=503, detail="Service not initialized")

        info = _inference_service.get_model_info()

        return {
            "status": "success",
            "model": info
        }

    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database
        db_connected = False
        try:
            conn = get_db_connection()
            conn.close()
            db_connected = True
        except:
            pass

        # Check Redis (optional - not implemented yet)
        redis_connected = True  # Placeholder

        # Check model
        model_loaded = _inference_service.model_loader.is_loaded if _inference_service else False

        # Calculate uptime
        uptime = (datetime.utcnow() - _service_start_time).total_seconds()

        # Get last signal
        last_signal = _position_manager.get_latest_signal() if _position_manager else None

        config = get_config()

        return HealthCheckResponse(
            status="healthy" if all([db_connected, model_loaded]) else "degraded",
            timestamp=datetime.utcnow(),
            service=config.service_name,
            version=config.service_version,
            model_loaded=model_loaded,
            database_connected=db_connected,
            redis_connected=redis_connected,
            uptime_seconds=uptime,
            total_signals_generated=_position_manager.total_signals if _position_manager else 0,
            last_signal_timestamp=last_signal.timestamp if last_signal else None
        )

    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            service="trading-signals-service",
            version="1.0.0",
            model_loaded=False,
            database_connected=False,
            redis_connected=False,
            uptime_seconds=0.0,
            total_signals_generated=0
        )
