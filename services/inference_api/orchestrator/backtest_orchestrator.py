"""
Backtest Orchestrator
Coordinates the entire backtest pipeline
"""

import time
import asyncio
import json
import queue
import threading
from typing import List, Dict, Any, Optional, AsyncGenerator
import logging
from ..core.data_loader import DataLoader
from ..core.inference_engine import InferenceEngine
from ..core.observation_builder import ObservationBuilder
from ..core import get_observation_builder
from ..core.trade_simulator import TradeSimulator, Trade
from ..core.trade_persister import TradePersister
from ..models.responses import BacktestResponse, TradeResponse, BacktestSummary, ProgressUpdate

logger = logging.getLogger(__name__)


class BacktestOrchestrator:
    """
    Orchestrates the entire backtest pipeline:
    1. Check for existing trades
    2. Load historical data
    3. Run model inference
    4. Simulate trades
    5. Persist results
    """

    def __init__(self, inference_engine: InferenceEngine):
        self.inference_engine = inference_engine
        self.data_loader = DataLoader()
        self.observation_builder = ObservationBuilder()  # Default 15-dim builder
        self.observation_builders = {}  # Cache for model-specific builders
        self.trade_simulator = TradeSimulator()
        self.trade_persister = TradePersister()

    def _get_observation_builder(self, model_id: str):
        """Get the appropriate observation builder for a model."""
        if model_id not in self.observation_builders:
            self.observation_builders[model_id] = get_observation_builder(model_id)
        return self.observation_builders[model_id]

    async def run(
        self,
        start_date: str,
        end_date: str,
        model_id: str = "ppo_primary",
        force_regenerate: bool = False
    ) -> BacktestResponse:
        """
        Run backtest for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            model_id: Model to use for inference
            force_regenerate: Force regeneration even if trades exist

        Returns:
            BacktestResponse with trades and summary
        """
        start_time = time.time()

        logger.info(f"Starting backtest: {start_date} to {end_date} (model={model_id})")

        # 1. Check for existing trades (unless force regenerate)
        if not force_regenerate:
            existing_trades = await self.trade_persister.get_trades(
                start_date, end_date, model_id
            )

            if existing_trades:
                logger.info(f"Found {len(existing_trades)} existing trades")

                # Calculate summary from existing trades
                summary = self._calculate_summary_from_dicts(existing_trades)

                return BacktestResponse(
                    success=True,
                    source="database",
                    trade_count=len(existing_trades),
                    trades=[TradeResponse(**t) for t in existing_trades],
                    summary=summary,
                    processing_time_ms=(time.time() - start_time) * 1000,
                    date_range={"start": start_date, "end": end_date}
                )

        # 2. Delete existing trades if force regenerate
        if force_regenerate:
            deleted = await self.trade_persister.delete_trades(
                start_date, end_date, model_id
            )
            logger.info(f"Deleted {deleted} existing trades for regeneration")

        # 3. Ensure model is loaded
        if not self.inference_engine.is_loaded(model_id):
            self.inference_engine.load_model(model_id)

        # 4. Load historical data
        logger.info("Loading historical data...")
        df = await self.data_loader.load_combined_data(start_date, end_date)
        logger.info(f"Loaded {len(df)} bars of data")

        if df.empty:
            return BacktestResponse(
                success=False,
                source="error",
                trade_count=0,
                trades=[],
                summary=None,
                processing_time_ms=(time.time() - start_time) * 1000,
                date_range={"start": start_date, "end": end_date}
            )

        # 5. Run simulation (use model-specific observation builder)
        logger.info(f"Running simulation with model {model_id}...")
        observation_builder = self._get_observation_builder(model_id)
        logger.info(f"Using observation builder: {type(observation_builder).__name__} (dim={getattr(observation_builder, 'observation_dim', 15)})")
        trades = self.trade_simulator.run_simulation(
            df=df,
            inference_engine=self.inference_engine,
            observation_builder=observation_builder,
            model_id=model_id
        )
        logger.info(f"Generated {len(trades)} trades")

        # 6. Calculate summary
        summary_dict = self.trade_simulator.calculate_summary(trades)
        summary = BacktestSummary(**summary_dict)

        # 7. Save to database
        logger.info("Saving trades to database...")
        saved_count = await self.trade_persister.save_trades(trades)
        logger.info(f"Saved {saved_count} trades")

        # 8. Convert trades to response format
        trade_responses = [self._trade_to_response(t) for t in trades]

        processing_time = (time.time() - start_time) * 1000

        return BacktestResponse(
            success=True,
            source="generated",
            trade_count=len(trades),
            trades=trade_responses,
            summary=summary,
            processing_time_ms=processing_time,
            date_range={"start": start_date, "end": end_date}
        )

    async def run_with_progress(
        self,
        start_date: str,
        end_date: str,
        model_id: str = "ppo_primary",
        force_regenerate: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        Run backtest with Server-Sent Events for progress updates.

        Yields SSE-formatted progress updates.
        """
        start_time = time.time()
        logger.info(f"[STREAM] run_with_progress called: {start_date} to {end_date}, model={model_id}, force={force_regenerate}")

        # Initial progress
        yield self._sse_event(ProgressUpdate(
            progress=0.0,
            current_bar=0,
            total_bars=0,
            trades_generated=0,
            status="starting",
            message="Checking for existing trades..."
        ))

        # Delete existing trades if force regenerate
        if force_regenerate:
            deleted = await self.trade_persister.delete_trades(
                start_date, end_date, model_id
            )
            if deleted > 0:
                logger.info(f"Deleted {deleted} existing trades for regeneration")
            existing_trades = []
        else:
            # Check existing trades
            existing_trades = await self.trade_persister.get_trades(
                start_date, end_date, model_id
            )

        if existing_trades:
            yield self._sse_event(ProgressUpdate(
                progress=1.0,
                current_bar=0,
                total_bars=0,
                trades_generated=len(existing_trades),
                status="completed",
                message=f"Found {len(existing_trades)} existing trades"
            ))

            # Yield final result
            summary = self._calculate_summary_from_dicts(existing_trades)
            result = BacktestResponse(
                success=True,
                source="database",
                trade_count=len(existing_trades),
                trades=[TradeResponse(**t) for t in existing_trades],
                summary=summary,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            yield f"data: {json.dumps({'type': 'result', 'data': result.model_dump()})}\n\n"
            return

        # Load data
        yield self._sse_event(ProgressUpdate(
            progress=0.05,
            current_bar=0,
            total_bars=0,
            trades_generated=0,
            status="loading",
            message="Loading historical data..."
        ))

        # Ensure model loaded
        if not self.inference_engine.is_loaded(model_id):
            self.inference_engine.load_model(model_id)

        df = await self.data_loader.load_combined_data(start_date, end_date)
        total_bars = len(df)

        yield self._sse_event(ProgressUpdate(
            progress=0.1,
            current_bar=0,
            total_bars=total_bars,
            trades_generated=0,
            status="running",
            message=f"Loaded {total_bars} bars, starting simulation..."
        ))

        # Run simulation with real-time trade streaming
        trades: List[Trade] = []
        trade_queue: queue.Queue = queue.Queue()
        simulation_done = threading.Event()
        current_equity = 10000.0  # Initial equity
        logger.info(f"[STREAM] Starting simulation with {total_bars} bars, model={model_id}")

        def trade_callback(trade: Trade, equity: float):
            """Called when a trade is closed - puts trade in queue for streaming"""
            nonlocal current_equity
            current_equity = equity
            trade_queue.put(trade)
            logger.info(f"[STREAM] Trade #{trade.trade_id} queued: {trade.side} @ {trade.entry_price:.2f}, equity={equity:.2f}")

        def progress_callback(progress: float, bar_idx: int, total: int):
            pass  # Progress updates handled separately

        def run_simulation():
            """Run simulation in separate thread"""
            nonlocal trades
            observation_builder = self._get_observation_builder(model_id)
            trades = self.trade_simulator.run_simulation(
                df=df,
                inference_engine=self.inference_engine,
                observation_builder=observation_builder,
                model_id=model_id,
                progress_callback=progress_callback,
                trade_callback=trade_callback
            )
            simulation_done.set()

        # Start simulation in background thread
        sim_thread = threading.Thread(target=run_simulation)
        sim_thread.start()

        # Stream trades as they are generated
        trades_streamed = 0
        logger.info(f"[STREAM] Entering trade streaming loop, simulation_done={simulation_done.is_set()}")
        while not simulation_done.is_set() or not trade_queue.empty():
            try:
                # Get trade from queue with timeout
                trade = trade_queue.get(timeout=0.1)
                trades_streamed += 1
                logger.info(f"[STREAM] Yielding trade #{trades_streamed}: {trade.side} @ {trade.entry_price:.2f}")

                # Send trade event with equity for equity curve update
                trade_data = self._trade_to_response(trade).model_dump()
                trade_data["current_equity"] = current_equity
                yield f"data: {json.dumps({'type': 'trade', 'data': trade_data})}\n\n"

                # Send progress update
                yield self._sse_event(ProgressUpdate(
                    progress=0.1 + (0.8 * trades_streamed / max(1, trades_streamed + 5)),
                    current_bar=0,
                    total_bars=total_bars,
                    trades_generated=trades_streamed,
                    status="running",
                    message=f"Trade #{trades_streamed} generated..."
                ))

            except queue.Empty:
                # No trade ready, continue waiting
                await asyncio.sleep(0.05)

        # Wait for simulation thread to complete
        sim_thread.join()
        logger.info(f"[STREAM] Simulation complete. Total trades: {len(trades)}, Streamed: {trades_streamed}")

        yield self._sse_event(ProgressUpdate(
            progress=0.9,
            current_bar=total_bars,
            total_bars=total_bars,
            trades_generated=len(trades),
            status="saving",
            message=f"Generated {len(trades)} trades, saving..."
        ))

        # Save trades
        await self.trade_persister.save_trades(trades)

        # Final result
        summary_dict = self.trade_simulator.calculate_summary(trades)
        summary = BacktestSummary(**summary_dict)

        yield self._sse_event(ProgressUpdate(
            progress=1.0,
            current_bar=total_bars,
            total_bars=total_bars,
            trades_generated=len(trades),
            status="completed",
            message="Backtest complete!"
        ))

        result = BacktestResponse(
            success=True,
            source="generated",
            trade_count=len(trades),
            trades=[self._trade_to_response(t) for t in trades],
            summary=summary,
            processing_time_ms=(time.time() - start_time) * 1000
        )

        yield f"data: {json.dumps({'type': 'result', 'data': result.model_dump()})}\n\n"

    def _sse_event(self, update: ProgressUpdate) -> str:
        """Format progress update as SSE event"""
        return f"data: {json.dumps({'type': 'progress', 'data': update.model_dump()})}\n\n"

    def _trade_to_response(self, trade: Trade) -> TradeResponse:
        """Convert Trade object to TradeResponse"""
        return TradeResponse(
            trade_id=trade.trade_id,
            model_id=trade.model_id,
            timestamp=trade.entry_time.isoformat() if trade.entry_time else "",
            entry_time=trade.entry_time.isoformat() if trade.entry_time else "",
            exit_time=trade.exit_time.isoformat() if trade.exit_time else None,
            side=trade.side,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            pnl=trade.pnl_usd,
            pnl_usd=trade.pnl_usd,
            pnl_percent=trade.pnl_pct,
            pnl_pct=trade.pnl_pct,
            status="closed" if trade.exit_time else "open",
            duration_minutes=trade.duration_bars * 5,
            exit_reason=trade.exit_reason,
            equity_at_entry=trade.equity_at_entry,
            equity_at_exit=trade.equity_at_exit,
            entry_confidence=trade.entry_confidence,
            exit_confidence=trade.exit_confidence
        )

    def _calculate_summary_from_dicts(
        self,
        trades: List[Dict[str, Any]]
    ) -> BacktestSummary:
        """Calculate summary from trade dictionaries"""
        if not trades:
            return BacktestSummary(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0
            )

        pnls = [t.get("pnl_usd", t.get("pnl", 0)) for t in trades]
        winning = len([p for p in pnls if p > 0])
        losing = len([p for p in pnls if p < 0])

        total_pnl = sum(pnls)
        initial = 10000.0
        total_return = (total_pnl / initial) * 100

        # Calculate drawdown
        equity = initial
        peak = initial
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        durations = [t.get("duration_minutes", 0) for t in trades]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return BacktestSummary(
            total_trades=len(trades),
            winning_trades=winning,
            losing_trades=losing,
            win_rate=(winning / len(trades) * 100) if trades else 0.0,
            total_pnl=total_pnl,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd * 100,
            avg_trade_duration_minutes=avg_duration
        )

    async def cleanup(self):
        """Cleanup resources"""
        await self.data_loader.close()
        await self.trade_persister.close()
