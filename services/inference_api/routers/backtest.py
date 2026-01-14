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
# Try multiple import paths for Docker and local dev compatibility
DEMO_MODE_AVAILABLE = False

try:
    # Docker container path (PYTHONPATH=/app, module at /app/services/demo_mode)
    from services.demo_mode import DemoTradeGenerator, is_investor_mode
    DEMO_MODE_AVAILABLE = True
    logging.getLogger(__name__).info(f"Demo mode loaded (services.demo_mode). INVESTOR_MODE={is_investor_mode()}")
except ImportError:
    try:
        # Local dev fallback (when running from services directory)
        from demo_mode import DemoTradeGenerator, is_investor_mode
        DEMO_MODE_AVAILABLE = True
        logging.getLogger(__name__).info(f"Demo mode loaded (demo_mode). INVESTOR_MODE={is_investor_mode()}")
    except ImportError as e:
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
        initial_capital=getattr(request, 'initial_capital', 10000.0)
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
        # Check for investor/demo mode
        if DEMO_MODE_AVAILABLE and is_investor_mode():
            logger.info(f"[INVESTOR MODE] Running demo stream backtest: {request.start_date} to {request.end_date}")
            return StreamingResponse(
                _demo_stream_generator(request, req),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )

        # Normal mode
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


async def _demo_stream_generator(request: BacktestRequest, req: Request):
    """Generate SSE events for demo mode with simulated progress."""
    import asyncio
    import json
    import time

    start_time = time.time()

    # Get the demo result first
    result = await _run_demo_backtest(request, req)
    trades = result.get("trades", [])
    total_trades = len(trades)

    # Simulate progress updates
    for i in range(0, 101, 10):
        progress_event = {
            "type": "progress",
            "data": {
                "progress": i / 100,
                "current_bar": int(i * 2.24),  # ~224 total
                "total_bars": 224,
                "trades_generated": int(total_trades * i / 100),
                "status": "running" if i < 100 else "completed",
                "message": f"Processing... {i}%"
            }
        }
        yield f"data: {json.dumps(progress_event)}\n\n"
        await asyncio.sleep(0.05)  # Small delay for visual effect

    # Fix result format to match dashboard schema
    processing_time = (time.time() - start_time) * 1000

    # Fix summary: max_drawdown_pct must be positive (absolute value)
    summary = result.get("summary", {})
    if summary and "max_drawdown_pct" in summary:
        summary["max_drawdown_pct"] = abs(summary["max_drawdown_pct"])

    final_result = {
        "success": True,
        "source": "generated",  # Must be 'database', 'generated', or 'error'
        "trade_count": total_trades,
        "trades": trades,
        "summary": summary,
        "processing_time_ms": processing_time,
        "date_range": {
            "start": request.start_date,
            "end": request.end_date
        }
    }

    # Send final result with type 'result' (not 'complete')
    final_event = {
        "type": "result",
        "data": final_result
    }
    yield f"data: {json.dumps(final_event)}\n\n"
    logger.info(f"[INVESTOR MODE] Stream completed with {total_trades} trades")


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
