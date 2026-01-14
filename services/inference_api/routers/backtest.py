"""
Backtest Router - Endpoints for running backtests

Supports INVESTOR_MODE for demo presentations with realistic metrics.
Set INVESTOR_MODE=true environment variable to enable.
"""

import os
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import logging
from ..models.requests import BacktestRequest
from ..models.responses import BacktestResponse, ErrorResponse
from ..orchestrator.backtest_orchestrator import BacktestOrchestrator

# Import demo mode (isolated module)
import sys
from pathlib import Path

# Add paths for demo_mode import (works in Docker and local dev)
_paths_to_try = [
    Path(__file__).parent.parent.parent,  # services directory (local dev)
    Path("/app"),  # Docker container path
    Path(__file__).parent.parent.parent.parent,  # project root
]
for _path in _paths_to_try:
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from demo_mode import DemoTradeGenerator, is_investor_mode
    DEMO_MODE_AVAILABLE = True
    logging.getLogger(__name__).info(f"Demo mode module loaded. INVESTOR_MODE={is_investor_mode()}")
except ImportError as e:
    DEMO_MODE_AVAILABLE = False
    logging.getLogger(__name__).warning(f"Demo mode not available: {e}")
    def is_investor_mode():
        return False

router = APIRouter(prefix="/backtest", tags=["backtest"])
logger = logging.getLogger(__name__)


@router.post("", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest, req: Request):
    """
    Run backtest for a date range.

    If trades already exist for the date range and model, returns cached results.
    Otherwise, runs simulation and saves results to database.

    - **start_date**: Start date in YYYY-MM-DD format
    - **end_date**: End date in YYYY-MM-DD format
    - **model_id**: Model ID to use (default: ppo_primary)
    - **force_regenerate**: Force regeneration even if trades exist

    **INVESTOR_MODE**: When enabled, returns optimized demo results for presentations.
    """
    try:
        # Check for investor/demo mode
        if DEMO_MODE_AVAILABLE and is_investor_mode():
            logger.info(f"[INVESTOR MODE] Running demo backtest: {request.start_date} to {request.end_date}")
            result = await _run_demo_backtest(request, req)
            return result

        # Normal mode - use real orchestrator
        orchestrator = BacktestOrchestrator(
            inference_engine=req.app.state.inference_engine
        )

        result = await orchestrator.run(
            start_date=request.start_date,
            end_date=request.end_date,
            model_id=request.model_id,
            force_regenerate=request.force_regenerate
        )

        await orchestrator.cleanup()

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Backtest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def _run_demo_backtest(request: BacktestRequest, req: Request):
    """
    Run demo backtest with optimized results for investor presentations.
    Completely isolated from production code.
    """
    import asyncpg
    from datetime import datetime as dt

    # Convert string dates to date objects for asyncpg
    start_dt = dt.strptime(request.start_date, "%Y-%m-%d").date()
    end_dt = dt.strptime(request.end_date, "%Y-%m-%d").date()

    # Fetch real price data for realistic trade generation
    db_url = os.getenv("DATABASE_URL", "postgresql://admin:admin123@localhost:5432/usdcop_trading")

    try:
        conn = await asyncpg.connect(db_url)

        # Fetch OHLCV data for the date range
        query = """
            SELECT time, open, high, low, close, volume
            FROM usdcop_m5_ohlcv
            WHERE time >= $1::date AND time < ($2::date + interval '1 day')
            ORDER BY time
        """
        rows = await conn.fetch(query, start_dt, end_dt)
        await conn.close()

        price_data = [
            {
                "time": row["time"].isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]) if row["volume"] else 0,
            }
            for row in rows
        ]

        logger.info(f"[INVESTOR MODE] Loaded {len(price_data)} candles for demo generation")

    except Exception as e:
        logger.error(f"[INVESTOR MODE] Failed to fetch price data: {e}")
        price_data = []

    # Generate demo trades
    generator = DemoTradeGenerator()
    result = generator.generate_trades(
        start_date=request.start_date,
        end_date=request.end_date,
        price_data=price_data,
        initial_capital=getattr(request, 'initial_capital', 100000.0)
    )

    logger.info(f"[INVESTOR MODE] Generated {result['trade_count']} demo trades")

    return result


@router.post("/stream")
async def run_backtest_stream(request: BacktestRequest, req: Request):
    """
    Run backtest with Server-Sent Events for real-time progress updates.

    Returns a stream of SSE events with progress updates and final result.
    """
    try:
        orchestrator = BacktestOrchestrator(
            inference_engine=req.app.state.inference_engine
        )

        async def event_generator():
            try:
                async for event in orchestrator.run_with_progress(
                    start_date=request.start_date,
                    end_date=request.end_date,
                    model_id=request.model_id
                ):
                    yield event
            finally:
                await orchestrator.cleanup()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logger.error(f"Stream error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{model_id}")
async def get_backtest_status(model_id: str, start_date: str, end_date: str, req: Request):
    """
    Check if backtest data exists for a date range.

    Returns count of existing trades.
    """
    try:
        orchestrator = BacktestOrchestrator(
            inference_engine=req.app.state.inference_engine
        )

        count = await orchestrator.trade_persister.get_trade_count(
            start_date, end_date, model_id
        )

        await orchestrator.cleanup()

        return {
            "model_id": model_id,
            "start_date": start_date,
            "end_date": end_date,
            "trade_count": count,
            "has_data": count > 0
        }

    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
