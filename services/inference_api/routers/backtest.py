"""
Backtest Router - Endpoints for running backtests

Supports two modes based on model_id selection:
- investor_demo: Synthetic trades for investor presentations
- ppo_primary: Real PPO model backtesting
"""

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
import logging
from ..models.requests import BacktestRequest
from ..models.responses import BacktestResponse, ErrorResponse
from ..orchestrator.backtest_orchestrator import BacktestOrchestrator

# Import demo mode (isolated module)
# Demo mode is selected by model_id, not environment variable
DEMO_MODE_AVAILABLE = False
is_demo_model = lambda model_id: model_id == "investor_demo"  # Default fallback

try:
    # Docker container path (PYTHONPATH=/app, module at /app/services/demo_mode)
    from services.demo_mode import DemoTradeGenerator, is_demo_model
    DEMO_MODE_AVAILABLE = True
    logging.getLogger(__name__).info("Demo mode loaded (services.demo_mode)")
except ImportError:
    try:
        # Local dev fallback (when running from services directory)
        from demo_mode import DemoTradeGenerator, is_demo_model
        DEMO_MODE_AVAILABLE = True
        logging.getLogger(__name__).info("Demo mode loaded (demo_mode)")
    except ImportError as e:
        logging.getLogger(__name__).warning(f"Demo mode not available: {e}")

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
    - **model_id**: Model ID to use (investor_demo for demo, ppo_primary for real)
    - **force_regenerate**: Force regeneration even if trades exist
    """
    try:
        # Demo mode determined ONLY by model_id (investor_demo vs ppo_primary)
        use_demo = is_demo_model(request.model_id)
        if use_demo:
            logger.info(f"[DEMO MODE] Running demo backtest for model '{request.model_id}': {request.start_date} to {request.end_date}")
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
    # Use environment variables or fallback to local defaults
    from ..config import get_settings
    settings = get_settings()
    db_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

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
        # Demo mode determined ONLY by model_id (investor_demo vs ppo_primary)
        use_demo = is_demo_model(request.model_id)
        if use_demo:
            logger.info(f"[DEMO MODE] Running demo stream backtest for model '{request.model_id}': {request.start_date} to {request.end_date}")
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
                    model_id=request.model_id,
                    force_regenerate=request.force_regenerate
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
    """
    Generate SSE events for demo mode with REALISTIC trade-by-trade streaming.

    This matches the real backtest streaming behavior:
    1. Initial progress events
    2. Individual trade events with current_equity (for real-time equity curve)
    3. Final result event

    The delay between trades creates the same dynamic visual effect as real backtests.
    """
    import asyncio
    import json
    import time

    start_time = time.time()

    # Initial progress: connecting
    yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 0.0, 'current_bar': 0, 'total_bars': 0, 'trades_generated': 0, 'status': 'starting', 'message': 'Checking for existing trades...'}})}\n\n"
    await asyncio.sleep(0.1)

    # Get the demo result (generates all trades)
    result = await _run_demo_backtest(request, req)
    trades = result.get("trades", [])
    total_trades = len(trades)

    # Loading progress
    yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 0.05, 'current_bar': 0, 'total_bars': 0, 'trades_generated': 0, 'status': 'loading', 'message': 'Loading historical data...'}})}\n\n"
    await asyncio.sleep(0.15)

    # Running progress - now stream trades one by one
    yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 0.1, 'current_bar': 0, 'total_bars': total_trades * 10, 'trades_generated': 0, 'status': 'running', 'message': f'Loaded data, starting simulation...'}})}\n\n"
    await asyncio.sleep(0.1)

    # Calculate delay per trade to create smooth animation
    # Target: ~15-30 seconds for full backtest visualization
    # Adjust based on number of trades
    if total_trades > 0:
        # For ~100 trades: 0.15s each = 15s total
        # For ~50 trades: 0.3s each = 15s total
        # For ~200 trades: 0.1s each = 20s total
        delay_per_trade = max(0.08, min(0.4, 20.0 / total_trades))
    else:
        delay_per_trade = 0.1

    logger.info(f"[INVESTOR MODE] Streaming {total_trades} trades with {delay_per_trade:.2f}s delay each")

    # Stream each trade individually (just like real backtest)
    for idx, trade in enumerate(trades):
        # Send trade event with current_equity for real-time equity curve updates
        trade_event = {
            "type": "trade",
            "data": {
                **trade,
                "current_equity": trade.get("equity_at_exit", trade.get("equity_at_entry", 10000))
            }
        }
        yield f"data: {json.dumps(trade_event)}\n\n"

        # Send progress update after each trade
        progress = 0.1 + (0.8 * (idx + 1) / total_trades)
        progress_event = {
            "type": "progress",
            "data": {
                "progress": progress,
                "current_bar": (idx + 1) * 10,
                "total_bars": total_trades * 10,
                "trades_generated": idx + 1,
                "status": "running",
                "message": f"Trade #{idx + 1} generated..."
            }
        }
        yield f"data: {json.dumps(progress_event)}\n\n"

        # Delay between trades for visual effect
        await asyncio.sleep(delay_per_trade)

    # Saving progress
    yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 0.9, 'current_bar': total_trades * 10, 'total_bars': total_trades * 10, 'trades_generated': total_trades, 'status': 'saving', 'message': f'Generated {total_trades} trades, saving...'}})}\n\n"
    await asyncio.sleep(0.1)

    # Fix result format to match dashboard schema
    processing_time = (time.time() - start_time) * 1000

    # Fix summary: max_drawdown_pct must be positive (absolute value)
    summary = result.get("summary", {})
    if summary and "max_drawdown_pct" in summary:
        summary["max_drawdown_pct"] = abs(summary["max_drawdown_pct"])

    # Completed progress
    yield f"data: {json.dumps({'type': 'progress', 'data': {'progress': 1.0, 'current_bar': total_trades * 10, 'total_bars': total_trades * 10, 'trades_generated': total_trades, 'status': 'completed', 'message': 'Backtest complete!'}})}\n\n"

    final_result = {
        "success": True,
        "source": "generated",
        "trade_count": total_trades,
        "trades": trades,
        "summary": summary,
        "processing_time_ms": processing_time,
        "date_range": {
            "start": request.start_date,
            "end": request.end_date
        }
    }

    # Send final result
    yield f"data: {json.dumps({'type': 'result', 'data': final_result})}\n\n"
    logger.info(f"[INVESTOR MODE] Stream completed with {total_trades} trades in {processing_time:.0f}ms")


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
