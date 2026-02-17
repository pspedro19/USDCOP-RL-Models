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


@router.post("/real")
async def run_real_backtest(
    proposal_id: str,
    start_date: str,
    end_date: str,
    req: Request
):
    """
    Run REAL backtest using the same BacktestEngine as L4 validation.

    This ensures metrics match between L4 validation and dashboard replay.
    Uses the model from the proposal's lineage.

    - **proposal_id**: The proposal ID to get model lineage from
    - **start_date**: Start date (YYYY-MM-DD)
    - **end_date**: End date (YYYY-MM-DD)
    """
    import json
    import asyncpg
    import pandas as pd
    import numpy as np
    from pathlib import Path

    logger.info(f"[REAL BACKTEST] Running for proposal {proposal_id}: {start_date} to {end_date}")

    try:
        # Get database connection
        from ..config import get_settings
        settings = get_settings()
        db_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

        conn = await asyncpg.connect(db_url)

        # 1. Get proposal lineage
        row = await conn.fetchrow("""
            SELECT model_id, lineage, metrics
            FROM promotion_proposals
            WHERE proposal_id = $1
        """, proposal_id)

        if not row:
            await conn.close()
            raise HTTPException(status_code=404, detail=f"Proposal {proposal_id} not found")

        model_id = row['model_id']
        lineage = row['lineage'] if isinstance(row['lineage'], dict) else json.loads(row['lineage'])

        logger.info(f"[REAL BACKTEST] Model: {model_id}, Lineage: {lineage}")

        await conn.close()

        # 2. Load model and norm_stats
        from stable_baselines3 import PPO

        model_path = lineage.get('modelPath') or lineage.get('model_path')
        norm_stats_path = lineage.get('normStatsPath') or lineage.get('norm_stats_path')

        if not model_path:
            model_path = f"models/{model_id}/final_model.zip"
        if not norm_stats_path:
            norm_stats_path = f"models/{model_id}/norm_stats.json"

        # Resolve paths (inside container)
        # Models are mounted at /models, data at /app/data
        if model_path.startswith("models/"):
            model_path = Path("/") / model_path  # /models/xxx
        else:
            model_path = Path("/app") / model_path

        if norm_stats_path.startswith("models/"):
            norm_stats_path = Path("/") / norm_stats_path
        else:
            norm_stats_path = Path("/app") / norm_stats_path

        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
        if not norm_stats_path.exists():
            raise HTTPException(status_code=404, detail=f"Norm stats not found: {norm_stats_path}")

        model = PPO.load(str(model_path), device='cpu')
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)

        logger.info(f"[REAL BACKTEST] Model loaded from {model_path}")

        # 3. Load val + test data (combined for complete backtest coverage)
        # First try to get dataset_path from lineage
        dataset_prefix = lineage.get('datasetPath') or lineage.get('dataset_path')

        df = None
        loaded_from = None

        if dataset_prefix:
            # Use lineage-specified dataset path
            # e.g., "data/pipeline/07_output/5min/DS_production"
            if dataset_prefix.startswith("data/"):
                base_path = Path("/app") / dataset_prefix
            else:
                base_path = Path(dataset_prefix)

            # Load val + test and combine
            dfs = []
            for split in ['val', 'test']:
                split_path = Path(str(base_path) + f"_{split}.parquet")
                if split_path.exists():
                    split_df = pd.read_parquet(split_path)
                    dfs.append(split_df)
                    logger.info(f"[REAL BACKTEST] Loaded {split}: {len(split_df)} rows from {split_path}")

            if dfs:
                df = pd.concat(dfs).sort_index()
                df = df[~df.index.duplicated(keep='first')]  # Remove duplicates
                loaded_from = str(base_path) + "_{val,test}.parquet"

        # Fallback to hardcoded paths if lineage not available
        if df is None:
            dataset_prefixes = [
                Path("/app/data/pipeline/07_output/5min/DS_production"),
                Path("/app/data/pipeline/07_output/5min/DS_v3_close_only"),
            ]

            for base_path in dataset_prefixes:
                dfs = []
                for split in ['val', 'test']:
                    split_path = Path(str(base_path) + f"_{split}.parquet")
                    if split_path.exists():
                        split_df = pd.read_parquet(split_path)
                        dfs.append(split_df)
                        logger.info(f"[REAL BACKTEST] Loaded {split}: {len(split_df)} rows")

                if dfs:
                    df = pd.concat(dfs).sort_index()
                    df = df[~df.index.duplicated(keep='first')]
                    loaded_from = str(base_path) + "_{val,test}.parquet"
                    break

        if df is None:
            raise HTTPException(status_code=404, detail="No val/test data found")

        logger.info(f"[REAL BACKTEST] Combined dataset: {len(df)} rows ({df.index.min()} to {df.index.max()})")

        # Filter by date range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)]

        logger.info(f"[REAL BACKTEST] Filtered to {len(df_filtered)} rows: {df_filtered.index.min()} to {df_filtered.index.max()}")

        if len(df_filtered) == 0:
            raise HTTPException(status_code=400, detail="No data in specified date range")

        # 4. Run backtest using REAL BacktestEngine
        from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig
        from src.config.experiment_loader import get_feature_order

        # Get feature columns
        try:
            feature_cols = list(get_feature_order()[:18])
        except Exception:
            feature_cols = [
                'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
                'rsi_9', 'rsi_21', 'volatility_pct', 'trend_z',
                'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
                'brent_change_1d', 'gold_change_1d', 'rate_spread_z',
                'rate_spread_change', 'usdmxn_change_1d', 'yield_curve_z'
            ]

        config = BacktestConfig.from_ssot()
        engine = BacktestEngine(config, norm_stats, feature_cols)

        logger.info(f"[REAL BACKTEST] Running engine on {len(df_filtered)} bars...")

        for i in range(1, len(df_filtered)):
            row = df_filtered.iloc[i]
            prev_row = df_filtered.iloc[i - 1]
            obs = engine.build_observation(row)
            action, _ = model.predict(obs, deterministic=True)
            action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
            engine.step(action_val, row, prev_row)

        result = engine.get_result(df_filtered)

        logger.info(f"[REAL BACKTEST] Complete: {result.total_trades} trades, {result.win_rate_pct:.1f}% WR, {result.total_return_pct:.2f}% return")

        # 5. Build response
        trades = []
        cumulative_pnl = 0.0

        for i, trade in enumerate(result.trades):
            cumulative_pnl += trade.pnl
            trades.append({
                'trade_id': i + 1,
                'entry_time': trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time),
                'exit_time': trade.exit_time.isoformat() if hasattr(trade.exit_time, 'isoformat') else str(trade.exit_time),
                'side': trade.direction,
                'entry_price': float(trade.entry_price),
                'exit_price': float(trade.exit_price),
                'pnl': float(trade.pnl),
                'pnl_usd': float(trade.pnl),
                'pnl_percent': float(trade.pnl_pct),
                'status': 'closed',
                'duration_minutes': int(trade.bars_held * 5),
                'exit_reason': 'signal',
                'equity_at_entry': float(config.initial_capital + cumulative_pnl - trade.pnl),
                'equity_at_exit': float(config.initial_capital + cumulative_pnl),
            })

        return {
            'success': True,
            'source': 'real_backtest',
            'proposal_id': proposal_id,
            'model_id': model_id,
            'trade_count': result.total_trades,
            'trades': trades,
            'summary': {
                'total_trades': result.total_trades,
                'winning_trades': int(result.win_rate_pct * result.total_trades / 100) if result.total_trades > 0 else 0,
                'losing_trades': result.total_trades - int(result.win_rate_pct * result.total_trades / 100) if result.total_trades > 0 else 0,
                'win_rate': round(result.win_rate_pct, 2),
                'total_pnl': round(result.total_return_pct * config.initial_capital / 100, 2),
                'total_return_pct': round(result.total_return_pct, 2),
                'max_drawdown_pct': round(result.max_drawdown_pct, 2),
                'sharpe_ratio': round(result.sharpe_annual, 2),
            },
            'config': {
                'threshold_long': config.threshold_long,
                'threshold_short': config.threshold_short,
                'stop_loss_pct': config.stop_loss_pct,
                'take_profit_pct': config.take_profit_pct,
                'trailing_stop_enabled': config.trailing_stop_enabled,
                'spread_bps': config.spread_bps,
                'slippage_bps': config.slippage_bps,
            },
            'date_range': {
                'start': start_date,
                'end': end_date
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[REAL BACKTEST] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/real/stream")
async def run_real_backtest_stream(
    proposal_id: str,
    start_date: str,
    end_date: str,
    req: Request
):
    """
    Run REAL backtest with SSE streaming for progressive updates.

    Emits events:
    - progress: {percent, bars_processed, total_bars}
    - trade: {trade_id, side, entry_price, exit_price, pnl, ...}
    - complete: {summary, config, date_range}
    """
    import json
    import asyncpg
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import asyncio

    async def generate_stream():
        try:
            logger.info(f"[REAL BACKTEST SSE] Starting for proposal {proposal_id}: {start_date} to {end_date}")

            # Emit initial status
            yield f"data: {json.dumps({'type': 'status', 'message': 'Connecting to database...'})}\n\n"

            # Get database connection
            from ..config import get_settings
            settings = get_settings()
            db_url = f"postgresql://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}"

            conn = await asyncpg.connect(db_url)

            # 1. Get proposal lineage
            row = await conn.fetchrow("""
                SELECT model_id, lineage, metrics
                FROM promotion_proposals
                WHERE proposal_id = $1
            """, proposal_id)

            if not row:
                await conn.close()
                yield f"data: {json.dumps({'type': 'error', 'message': f'Proposal {proposal_id} not found'})}\n\n"
                return

            model_id = row['model_id']
            lineage = row['lineage'] if isinstance(row['lineage'], dict) else json.loads(row['lineage'])
            await conn.close()

            yield f"data: {json.dumps({'type': 'status', 'message': 'Loading model...'})}\n\n"

            # 2. Load model and norm_stats
            from stable_baselines3 import PPO

            model_path = lineage.get('modelPath') or lineage.get('model_path')
            norm_stats_path = lineage.get('normStatsPath') or lineage.get('norm_stats_path')

            if not model_path:
                model_path = f"models/{model_id}/final_model.zip"
            if not norm_stats_path:
                norm_stats_path = f"models/{model_id}/norm_stats.json"

            # Resolve paths
            if model_path.startswith("models/"):
                model_path = Path("/") / model_path
            else:
                model_path = Path("/app") / model_path

            if norm_stats_path.startswith("models/"):
                norm_stats_path = Path("/") / norm_stats_path
            else:
                norm_stats_path = Path("/app") / norm_stats_path

            if not model_path.exists():
                yield f"data: {json.dumps({'type': 'error', 'message': f'Model not found: {model_path}'})}\n\n"
                return

            model = PPO.load(str(model_path), device='cpu')
            with open(norm_stats_path) as f:
                norm_stats = json.load(f)

            yield f"data: {json.dumps({'type': 'status', 'message': 'Loading market data (val + test combined)...'})}\n\n"

            # 3. Load val + test data combined for complete backtest coverage
            dataset_prefix = lineage.get('datasetPath') or lineage.get('dataset_path')

            df = None
            loaded_from = None

            if dataset_prefix:
                # Use lineage-specified dataset path
                if dataset_prefix.startswith("data/"):
                    base_path = Path("/app") / dataset_prefix
                else:
                    base_path = Path(dataset_prefix)

                # Load val + test and combine
                dfs = []
                for split in ['val', 'test']:
                    split_path = Path(str(base_path) + f"_{split}.parquet")
                    if split_path.exists():
                        split_df = pd.read_parquet(split_path)
                        dfs.append(split_df)
                        logger.info(f"[REAL BACKTEST SSE] Loaded {split}: {len(split_df)} rows")

                if dfs:
                    df = pd.concat(dfs).sort_index()
                    df = df[~df.index.duplicated(keep='first')]
                    loaded_from = str(base_path)

            # Fallback to hardcoded paths
            if df is None:
                dataset_prefixes = [
                    Path("/app/data/pipeline/07_output/5min/DS_production"),
                    Path("/app/data/pipeline/07_output/5min/DS_v3_close_only"),
                ]

                for base_path in dataset_prefixes:
                    dfs = []
                    for split in ['val', 'test']:
                        split_path = Path(str(base_path) + f"_{split}.parquet")
                        if split_path.exists():
                            split_df = pd.read_parquet(split_path)
                            dfs.append(split_df)
                            logger.info(f"[REAL BACKTEST SSE] Loaded {split}: {len(split_df)} rows")

                    if dfs:
                        df = pd.concat(dfs).sort_index()
                        df = df[~df.index.duplicated(keep='first')]
                        loaded_from = str(base_path)
                        break

            if df is None:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No val/test data found'})}\n\n"
                return

            logger.info(f"[REAL BACKTEST SSE] Combined dataset: {len(df)} rows ({df.index.min()} to {df.index.max()})")

            # Filter by date range
            start_dt = pd.Timestamp(start_date)
            end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            df_filtered = df[(df.index >= start_dt) & (df.index < end_dt)]

            total_bars = len(df_filtered)
            if total_bars == 0:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No data in specified date range'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'status', 'message': f'Running backtest on {total_bars} bars...'})}\n\n"

            # 4. Run backtest with streaming
            from src.evaluation.backtest_engine import BacktestEngine, BacktestConfig
            from src.config.experiment_loader import get_feature_order

            try:
                feature_cols = list(get_feature_order()[:18])
            except Exception:
                feature_cols = [
                    'log_ret_5m', 'log_ret_1h', 'log_ret_4h', 'log_ret_1d',
                    'rsi_9', 'rsi_21', 'volatility_pct', 'trend_z',
                    'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
                    'brent_change_1d', 'gold_change_1d', 'rate_spread_z',
                    'rate_spread_change', 'usdmxn_change_1d', 'yield_curve_z'
                ]

            config = BacktestConfig.from_ssot()
            engine = BacktestEngine(config, norm_stats, feature_cols)

            # Track trades for streaming
            last_trade_count = 0
            cumulative_pnl = 0.0
            trades_emitted = []

            # Process bars and emit progress
            progress_interval = max(1, total_bars // 100)  # Emit ~100 progress updates

            for i in range(1, total_bars):
                row = df_filtered.iloc[i]
                prev_row = df_filtered.iloc[i - 1]
                obs = engine.build_observation(row)
                action, _ = model.predict(obs, deterministic=True)
                action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
                engine.step(action_val, row, prev_row)

                # Check for new closed trades
                current_trades = len(engine.trades)
                if current_trades > last_trade_count:
                    # New trade closed - emit it
                    for j in range(last_trade_count, current_trades):
                        trade = engine.trades[j]
                        cumulative_pnl += trade.pnl
                        trade_data = {
                            'trade_id': j + 1,
                            'entry_time': trade.entry_time.isoformat() if hasattr(trade.entry_time, 'isoformat') else str(trade.entry_time),
                            'exit_time': trade.exit_time.isoformat() if hasattr(trade.exit_time, 'isoformat') else str(trade.exit_time),
                            'side': trade.direction,
                            'entry_price': float(trade.entry_price),
                            'exit_price': float(trade.exit_price),
                            'pnl': float(trade.pnl),
                            'pnl_usd': float(trade.pnl),
                            'pnl_percent': float(trade.pnl_pct),
                            'status': 'closed',
                            'duration_minutes': int(trade.bars_held * 5),
                            'exit_reason': 'signal',
                            'equity_at_entry': float(config.initial_capital + cumulative_pnl - trade.pnl),
                            'equity_at_exit': float(config.initial_capital + cumulative_pnl),
                        }
                        trades_emitted.append(trade_data)
                        yield f"data: {json.dumps({'type': 'trade', 'data': trade_data})}\n\n"

                    last_trade_count = current_trades

                # Emit progress every N bars
                if i % progress_interval == 0 or i == total_bars - 1:
                    percent = round((i / total_bars) * 100, 1)
                    yield f"data: {json.dumps({'type': 'progress', 'percent': percent, 'bars_processed': i, 'total_bars': total_bars, 'trades_so_far': len(trades_emitted)})}\n\n"
                    await asyncio.sleep(0)  # Allow other tasks to run

            # Get final result
            result = engine.get_result(df_filtered)

            # Emit complete event with summary
            complete_data = {
                'type': 'complete',
                'success': True,
                'source': 'real_backtest_stream',
                'proposal_id': proposal_id,
                'model_id': model_id,
                'trade_count': result.total_trades,
                'summary': {
                    'total_trades': result.total_trades,
                    'winning_trades': int(result.win_rate_pct * result.total_trades / 100) if result.total_trades > 0 else 0,
                    'losing_trades': result.total_trades - int(result.win_rate_pct * result.total_trades / 100) if result.total_trades > 0 else 0,
                    'win_rate': round(result.win_rate_pct, 2),
                    'total_pnl': round(result.total_return_pct * config.initial_capital / 100, 2),
                    'total_return_pct': round(result.total_return_pct, 2),
                    'max_drawdown_pct': round(result.max_drawdown_pct, 2),
                    'sharpe_ratio': round(result.sharpe_annual, 2),
                },
                'config': {
                    'threshold_long': config.threshold_long,
                    'threshold_short': config.threshold_short,
                    'stop_loss_pct': config.stop_loss_pct,
                    'take_profit_pct': config.take_profit_pct,
                },
                'date_range': {'start': start_date, 'end': end_date}
            }

            yield f"data: {json.dumps(complete_data)}\n\n"
            logger.info(f"[REAL BACKTEST SSE] Complete: {result.total_trades} trades, {result.win_rate_pct:.1f}% WR")

        except Exception as e:
            logger.error(f"[REAL BACKTEST SSE] Error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


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
