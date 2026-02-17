"""
Backtest Factory - Factory Pattern Implementation
=================================================
Creates backtest runners and orchestrators with proper configuration.

SOLID Principles:
- Single Responsibility: Factory only creates objects
- Open/Closed: New runners via registration
- Dependency Inversion: Depends on contracts, not implementations

Design Patterns:
- Factory Pattern: Creates configured backtest runners
- Builder Pattern: Fluent configuration API
- Registry Pattern: Stores runner type registrations

Author: Trading Team
Version: 1.0.0
Created: 2025-01-12
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Type

from contracts.backtest_contracts import (
    BacktestConfig,
    BacktestRequest,
    BacktestResult,
    BacktestStatus,
    BacktestMetrics,
    BacktestPeriodType,
    TradeRecord,
)

logger = logging.getLogger(__name__)


# =============================================================================
# PROTOCOLS - Interface Definitions
# =============================================================================

class BacktestRunner(Protocol):
    """Protocol defining backtest runner interface"""

    def run(self, request: BacktestRequest) -> BacktestResult:
        """Run backtest and return result"""
        ...


class MetricsCalculator(Protocol):
    """Protocol for metrics calculation"""

    def calculate(self, trades: List[TradeRecord], start_date: date, end_date: date) -> BacktestMetrics:
        """Calculate metrics from trades"""
        ...


# =============================================================================
# ABSTRACT RUNNER BASE
# =============================================================================

class AbstractBacktestRunner(ABC):
    """
    Abstract base class for backtest runners.

    Implements Template Method pattern:
    - run() is the template method
    - _load_data(), _run_simulation(), _calculate_metrics() are hooks
    """

    def __init__(self, config: BacktestConfig, project_root: Path):
        self.config = config
        self.project_root = project_root

    def run(self, request: BacktestRequest) -> BacktestResult:
        """
        Template method for running backtest.

        Steps:
        1. Check cache
        2. Load data
        3. Run simulation
        4. Calculate metrics
        5. Persist results
        """
        import time
        start_time = time.time()

        try:
            # Step 1: Check cache
            cached = self._check_cache(request)
            if cached and not request.force_regenerate:
                logger.info(f"Using cached backtest for {request.model_id}")
                return cached

            # Step 2: Load data
            logger.info(f"Loading data for {request.start_date} to {request.end_date}")
            data = self._load_data(request)

            # Step 3: Run simulation
            logger.info(f"Running simulation with model {request.model_id}")
            trades = self._run_simulation(data, request)

            # Step 4: Calculate metrics
            logger.info(f"Calculating metrics for {len(trades)} trades")
            metrics = self._calculate_metrics(trades, request.start_date, request.end_date)

            # Step 5: Persist
            self._persist_results(trades, request)

            processing_time = (time.time() - start_time) * 1000

            return BacktestResult(
                model_id=request.model_id,
                period_type=request.period_type,
                status=BacktestStatus.COMPLETED,
                source="generated",
                processing_time_ms=processing_time,
                trades=trades,
                metrics=metrics,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"Backtest failed: {e}")

            return BacktestResult(
                model_id=request.model_id,
                period_type=request.period_type,
                status=BacktestStatus.FAILED,
                source="generated",
                processing_time_ms=processing_time,
                error_message=str(e),
            )

    @abstractmethod
    def _check_cache(self, request: BacktestRequest) -> Optional[BacktestResult]:
        """Check if cached result exists"""
        pass

    @abstractmethod
    def _load_data(self, request: BacktestRequest) -> Any:
        """Load data for simulation"""
        pass

    @abstractmethod
    def _run_simulation(self, data: Any, request: BacktestRequest) -> List[TradeRecord]:
        """Run trading simulation"""
        pass

    @abstractmethod
    def _calculate_metrics(
        self, trades: List[TradeRecord], start_date: date, end_date: date
    ) -> BacktestMetrics:
        """Calculate metrics from trades"""
        pass

    @abstractmethod
    def _persist_results(self, trades: List[TradeRecord], request: BacktestRequest) -> None:
        """Persist results to storage"""
        pass


# =============================================================================
# CONCRETE RUNNERS
# =============================================================================

class OrchestratorBacktestRunner(AbstractBacktestRunner):
    """
    Backtest runner using existing BacktestOrchestrator.

    Wraps the existing inference API orchestrator for DAG use.
    """

    def __init__(self, config: BacktestConfig, project_root: Path, db_connection: Any = None):
        super().__init__(config, project_root)
        self.db_connection = db_connection
        self._orchestrator = None

    def _get_orchestrator(self):
        """Lazy-load orchestrator"""
        if self._orchestrator is None:
            import sys
            src_path = str(self.project_root / "services" / "inference_api")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)

            try:
                from orchestrator.backtest_orchestrator import BacktestOrchestrator
                self._orchestrator = BacktestOrchestrator(
                    project_root=self.project_root,
                    db_connection=self.db_connection,
                )
            except ImportError as e:
                logger.warning(f"Could not import BacktestOrchestrator: {e}")
                self._orchestrator = None

        return self._orchestrator

    def _check_cache(self, request: BacktestRequest) -> Optional[BacktestResult]:
        """Check database for cached trades"""
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            return None

        try:
            cached_trades = orchestrator.get_cached_trades(
                model_id=request.model_id,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            if cached_trades:
                # Convert to TradeRecords
                trades = self._convert_trades(cached_trades)
                metrics = self._calculate_metrics(trades, request.start_date, request.end_date)

                return BacktestResult(
                    model_id=request.model_id,
                    period_type=request.period_type,
                    status=BacktestStatus.CACHED,
                    source="database",
                    processing_time_ms=0,
                    trades=trades,
                    metrics=metrics,
                )
        except Exception as e:
            logger.warning(f"Cache check failed: {e}")

        return None

    def _load_data(self, request: BacktestRequest) -> Any:
        """Load data via orchestrator"""
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            raise RuntimeError("BacktestOrchestrator not available")

        return orchestrator.load_data(request.start_date, request.end_date)

    def _run_simulation(self, data: Any, request: BacktestRequest) -> List[TradeRecord]:
        """Run simulation via orchestrator"""
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            raise RuntimeError("BacktestOrchestrator not available")

        raw_trades = orchestrator.run_simulation(
            data=data,
            model_id=request.model_id,
            config={
                "initial_capital": self.config.initial_capital,
                "transaction_cost_bps": self.config.transaction_cost_bps,
                "long_threshold": self.config.long_entry_threshold,
                "short_threshold": self.config.short_entry_threshold,
                "exit_threshold": self.config.exit_threshold,
            }
        )

        return self._convert_trades(raw_trades)

    def _calculate_metrics(
        self, trades: List[TradeRecord], start_date: date, end_date: date
    ) -> BacktestMetrics:
        """Calculate metrics from trades"""
        if not trades:
            return BacktestMetrics(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                total_pnl_usd=0,
                total_return_pct=0,
                max_drawdown_pct=0,
                max_drawdown_usd=0,
                win_rate=0,
                start_date=start_date,
                end_date=end_date,
                trading_days=0,
            )

        # Calculate metrics
        winners = [t for t in trades if t.pnl_usd and t.pnl_usd > 0]
        losers = [t for t in trades if t.pnl_usd and t.pnl_usd < 0]

        total_pnl = sum(t.pnl_usd or 0 for t in trades)
        gross_profit = sum(t.pnl_usd for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl_usd for t in losers)) if losers else 0

        initial_equity = trades[0].equity_at_entry if trades else self.config.initial_capital
        final_equity = trades[-1].equity_at_exit or trades[-1].equity_at_entry if trades else initial_equity

        # Calculate drawdown
        peak = initial_equity
        max_dd_usd = 0
        for trade in trades:
            equity = trade.equity_at_exit or trade.equity_at_entry
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd_usd:
                max_dd_usd = dd

        max_dd_pct = max_dd_usd / peak if peak > 0 else 0

        # Calculate consecutive wins/losses
        max_cons_wins = max_cons_losses = 0
        cons_wins = cons_losses = 0
        for trade in trades:
            if trade.pnl_usd and trade.pnl_usd > 0:
                cons_wins += 1
                cons_losses = 0
                max_cons_wins = max(max_cons_wins, cons_wins)
            elif trade.pnl_usd and trade.pnl_usd < 0:
                cons_losses += 1
                cons_wins = 0
                max_cons_losses = max(max_cons_losses, cons_losses)

        # Calculate Sharpe (simplified daily)
        returns = []
        for trade in trades:
            if trade.pnl_pct:
                returns.append(trade.pnl_pct)

        sharpe = None
        if len(returns) > 1:
            import statistics
            mean_ret = statistics.mean(returns)
            std_ret = statistics.stdev(returns)
            if std_ret > 0:
                sharpe = (mean_ret / std_ret) * (252 ** 0.5)  # Annualized

        return BacktestMetrics(
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            total_pnl_usd=total_pnl,
            total_return_pct=(final_equity - initial_equity) / initial_equity if initial_equity > 0 else 0,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usd=max_dd_usd,
            win_rate=len(winners) / len(trades) if trades else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else None,
            avg_win_usd=statistics.mean([t.pnl_usd for t in winners]) if winners else None,
            avg_loss_usd=statistics.mean([t.pnl_usd for t in losers]) if losers else None,
            avg_trade_duration_minutes=statistics.mean([t.duration_minutes for t in trades if t.duration_minutes]) if trades else None,
            max_consecutive_wins=max_cons_wins,
            max_consecutive_losses=max_cons_losses,
            start_date=start_date,
            end_date=end_date,
            trading_days=(end_date - start_date).days,
        )

    def _persist_results(self, trades: List[TradeRecord], request: BacktestRequest) -> None:
        """Persist trades to database"""
        orchestrator = self._get_orchestrator()
        if orchestrator is None:
            return

        try:
            orchestrator.persist_trades(
                trades=[t.model_dump() for t in trades],
                model_id=request.model_id,
            )
        except Exception as e:
            logger.warning(f"Failed to persist trades: {e}")

    def _convert_trades(self, raw_trades: List[Dict]) -> List[TradeRecord]:
        """Convert raw trade dicts to TradeRecord"""
        records = []
        for t in raw_trades:
            try:
                records.append(TradeRecord(
                    trade_id=t.get("trade_id"),
                    entry_time=t.get("entry_time") or t.get("timestamp"),
                    exit_time=t.get("exit_time"),
                    side=t.get("side", "LONG"),
                    entry_price=t.get("entry_price", 0),
                    exit_price=t.get("exit_price"),
                    pnl_usd=t.get("pnl_usd") or t.get("pnl"),
                    pnl_pct=t.get("pnl_pct"),
                    equity_at_entry=t.get("equity_at_entry", self.config.initial_capital),
                    equity_at_exit=t.get("equity_at_exit"),
                    entry_confidence=t.get("entry_confidence") or t.get("confidence"),
                    exit_reason=t.get("exit_reason"),
                ))
            except Exception as e:
                logger.warning(f"Failed to convert trade: {e}")
                continue

        return records


class EvaluationBacktestRunner(AbstractBacktestRunner):
    """
    Backtest runner using src/evaluation/backtest_engine.py.

    This is the primary runner for L4 validation, using the unified
    BacktestEngine that ensures exact parity with L3 training environment.
    """

    def __init__(self, config: BacktestConfig, project_root: Path, **kwargs):
        super().__init__(config, project_root)
        self._model = None
        self._norm_stats = None
        self._test_df = None
        self._model_path = kwargs.get('model_path')
        self._norm_stats_path = kwargs.get('norm_stats_path')
        self._dataset_path = kwargs.get('dataset_path')

    def _check_cache(self, request: BacktestRequest) -> Optional[BacktestResult]:
        """No caching for evaluation runner"""
        return None

    def _load_model(self, model_id: str):
        """Load model from standard paths"""
        import sys
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from stable_baselines3 import PPO

            # Try multiple model paths
            model_paths = [
                self._model_path,
                self.project_root / "models" / f"{model_id}" / "final_model.zip",
                self.project_root / "models" / f"ppo_{model_id}_production" / "final_model.zip",
                self.project_root / "models" / f"{model_id}_production" / "final_model.zip",
            ]

            for path in model_paths:
                if path and Path(path).exists():
                    self._model = PPO.load(str(path))
                    logger.info(f"Loaded model from {path}")
                    return

            raise FileNotFoundError(f"Model not found for {model_id}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _load_norm_stats(self, model_id: str) -> Dict:
        """Load normalization statistics"""
        import json

        norm_paths = [
            self._norm_stats_path,
            self.project_root / "models" / f"{model_id}" / "norm_stats.json",
            self.project_root / "models" / f"ppo_{model_id}_production" / "norm_stats.json",
            self.project_root / "models" / f"{model_id}_production" / "norm_stats.json",
        ]

        for path in norm_paths:
            if path and Path(path).exists():
                with open(path) as f:
                    self._norm_stats = json.load(f)
                logger.info(f"Loaded norm_stats from {path}")
                return self._norm_stats

        raise FileNotFoundError(f"norm_stats.json not found for {model_id}")

    def _load_data(self, request: BacktestRequest) -> Any:
        """Load val + test datasets combined for complete backtest coverage"""
        import pandas as pd

        # First, try to load combined val + test from dataset prefix
        dataset_prefixes = []

        # If dataset_path is provided, check if it's a specific file or a prefix
        if self._dataset_path:
            ds_path = Path(self._dataset_path)
            if ds_path.exists() and ds_path.suffix == '.parquet':
                # Single file - try to extract prefix for combined loading
                path_str = str(ds_path)
                for split in ['_train.parquet', '_val.parquet', '_test.parquet']:
                    if path_str.endswith(split):
                        # Extract prefix (e.g., "DS_v3_close_only")
                        prefix_path = path_str[:-len(split)]
                        dataset_prefixes.append(Path(prefix_path))
                        break
                else:
                    # Not a split file, use as-is
                    dataset_prefixes.append(ds_path)
            else:
                # Assume it's a prefix path
                dataset_prefixes.append(ds_path)

        # Add fallback paths
        dataset_prefixes.extend([
            self.project_root / "data" / "pipeline" / "07_output" / "5min" / "DS_v3_close_only",
            self.project_root / "data" / "pipeline" / "07_output" / "5min" / "DS_production",
        ])

        # Try to load combined val + test
        for base_path in dataset_prefixes:
            dfs = []

            # Check if it's a full path to a single file
            if base_path and base_path.suffix == '.parquet' and base_path.exists():
                self._test_df = pd.read_parquet(base_path)
                logger.info(f"Loaded single dataset from {base_path}: {len(self._test_df)} rows")
                return self._filter_by_date(request)

            # Try to load val + test splits
            for split in ['val', 'test']:
                split_path = Path(str(base_path) + f"_{split}.parquet")
                if split_path.exists():
                    split_df = pd.read_parquet(split_path)
                    dfs.append(split_df)
                    logger.info(f"Loaded {split}: {len(split_df)} rows from {split_path}")

            if dfs:
                # Combine and deduplicate
                self._test_df = pd.concat(dfs).sort_index()
                self._test_df = self._test_df[~self._test_df.index.duplicated(keep='first')]
                logger.info(f"Combined val+test dataset: {len(self._test_df)} rows ({self._test_df.index.min()} to {self._test_df.index.max()})")
                return self._filter_by_date(request)

        raise FileNotFoundError("Val/Test datasets not found")

    def _filter_by_date(self, request: BacktestRequest) -> Any:
        """Filter loaded dataset by date range"""
        import pandas as pd

        # Filter by date range
        if 'timestamp' in self._test_df.columns:
            ts_col = 'timestamp'
        elif 'datetime' in self._test_df.columns:
            ts_col = 'datetime'
        elif isinstance(self._test_df.index, pd.DatetimeIndex):
            self._test_df['timestamp'] = self._test_df.index
            ts_col = 'timestamp'
        else:
            ts_col = None

        if ts_col:
            self._test_df[ts_col] = pd.to_datetime(self._test_df[ts_col])
            start = pd.to_datetime(request.start_date)
            end = pd.to_datetime(request.end_date)
            if self._test_df[ts_col].dt.tz is not None:
                start = start.tz_localize('UTC')
                end = end.tz_localize('UTC')
            self._test_df = self._test_df[
                (self._test_df[ts_col] >= start) &
                (self._test_df[ts_col] <= end)
            ]
            logger.info(f"Filtered to date range: {len(self._test_df)} rows")

        return self._test_df

    def _run_simulation(self, data: Any, request: BacktestRequest) -> List[TradeRecord]:
        """Run simulation using BacktestEngine"""
        import sys
        src_path = str(self.project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        try:
            from evaluation.backtest_engine import (
                BacktestEngine,
                BacktestConfig as EvalBacktestConfig,
            )
        except ImportError:
            logger.warning("Could not import BacktestEngine from src/evaluation")
            raise RuntimeError("BacktestEngine not available")

        # Load model and norm_stats
        if self._model is None:
            self._load_model(request.model_id)

        if self._norm_stats is None:
            self._load_norm_stats(request.model_id)

        # Get feature columns from norm_stats
        if "features" in self._norm_stats:
            feature_stats = self._norm_stats["features"]
        else:
            feature_stats = {k: v for k, v in self._norm_stats.items() if not k.startswith("_")}

        feature_cols = list(feature_stats.keys())
        logger.info(f"Using {len(feature_cols)} features: {feature_cols[:5]}...")

        # Create backtest config
        eval_config = EvalBacktestConfig(
            initial_capital=self.config.initial_capital,
            spread_bps=self.config.transaction_cost_bps,
            slippage_bps=self.config.slippage_bps,
            threshold_long=self.config.long_entry_threshold,
            threshold_short=self.config.short_entry_threshold,
        )

        # Initialize engine
        engine = BacktestEngine(eval_config, feature_stats, feature_cols)

        # Run backtest
        df = self._test_df
        trades_internal = []

        for i in range(1, len(df)):
            obs = engine.build_observation(df.iloc[i])
            action, _ = self._model.predict(obs, deterministic=True)
            action_val = float(action[0]) if hasattr(action, '__len__') else float(action)
            engine.step(action_val, df.iloc[i], df.iloc[i-1])

        # Close final position
        engine.close_position(df.iloc[-1])

        # Convert internal trades to TradeRecord format
        trades = []
        from datetime import datetime as dt
        import pandas as pd

        for t in engine.trades:
            # Parse entry_time (could be datetime string or index name)
            entry_time_val = getattr(t, 'entry_time', None)
            if entry_time_val:
                try:
                    if isinstance(entry_time_val, str) and entry_time_val.strip():
                        entry_time_val = pd.to_datetime(entry_time_val)
                    elif not isinstance(entry_time_val, dt):
                        entry_time_val = dt.now()  # Fallback
                except:
                    entry_time_val = dt.now()
            else:
                entry_time_val = dt.now()

            # Parse exit_time
            exit_time_val = getattr(t, 'exit_time', None)
            if exit_time_val:
                try:
                    if isinstance(exit_time_val, str) and exit_time_val.strip():
                        exit_time_val = pd.to_datetime(exit_time_val)
                    elif not isinstance(exit_time_val, dt):
                        exit_time_val = None
                except:
                    exit_time_val = None

            # Ensure entry_price > 0 (required by contract)
            entry_price = getattr(t, 'entry_price', 1.0)
            if entry_price <= 0:
                entry_price = 4200.0  # Default USDCOP price

            exit_price = getattr(t, 'exit_price', None)
            if exit_price is not None and exit_price <= 0:
                exit_price = None

            trades.append(TradeRecord(
                trade_id=len(trades),
                entry_time=entry_time_val,
                exit_time=exit_time_val,
                side=getattr(t, 'direction', 'LONG'),
                entry_price=entry_price,
                exit_price=exit_price,
                pnl_usd=getattr(t, 'pnl', 0),
                pnl_pct=getattr(t, 'pnl_pct', None) if hasattr(t, 'pnl_pct') else None,
                equity_at_entry=self.config.initial_capital,
                equity_at_exit=engine.capital,
            ))

        # Store engine for metrics calculation
        self._engine = engine

        logger.info(f"Simulation complete: {len(trades)} trades")
        return trades

    def _calculate_metrics(
        self, trades: List[TradeRecord], start_date: date, end_date: date
    ) -> BacktestMetrics:
        """Calculate metrics from engine results"""
        if hasattr(self, '_engine'):
            engine = self._engine
            result = engine.get_result(self._test_df)

            return BacktestMetrics(
                total_trades=result.total_trades,
                winning_trades=result.winning_trades,
                losing_trades=result.losing_trades,
                total_pnl_usd=result.total_pnl,
                total_return_pct=result.total_return_pct / 100,  # Convert to decimal
                sharpe_ratio=result.sharpe_annual,
                max_drawdown_pct=result.max_drawdown_pct / 100,  # Convert to decimal
                max_drawdown_usd=result.max_drawdown_pct / 100 * self.config.initial_capital,
                win_rate=result.win_rate_pct / 100,  # Convert to decimal
                profit_factor=result.profit_factor if result.profit_factor < 999 else None,
                avg_win_usd=result.avg_pnl if result.avg_pnl > 0 else None,
                avg_loss_usd=result.avg_pnl if result.avg_pnl < 0 else None,
                avg_trade_duration_minutes=result.avg_duration_hours * 60 if result.avg_duration_hours else None,
                start_date=start_date,
                end_date=end_date,
                trading_days=int(result.trading_days),
            )

        # Fallback to parent calculation
        return OrchestratorBacktestRunner._calculate_metrics(self, trades, start_date, end_date)

    def _persist_results(self, trades: List[TradeRecord], request: BacktestRequest) -> None:
        """Persist results to file"""
        import json
        from datetime import datetime

        output_dir = self.project_root / "results" / "backtests"
        output_dir.mkdir(parents=True, exist_ok=True)

        result_file = output_dir / f"{request.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(result_file, 'w') as f:
            json.dump({
                "model_id": request.model_id,
                "start_date": str(request.start_date),
                "end_date": str(request.end_date),
                "trade_count": len(trades),
                "trades": [t.model_dump() if hasattr(t, 'model_dump') else vars(t) for t in trades[:100]],
            }, f, indent=2, default=str)

        logger.info(f"Results saved to {result_file}")


class MockBacktestRunner(AbstractBacktestRunner):
    """
    Mock backtest runner for testing.

    Returns deterministic results without hitting external services.
    """

    def _check_cache(self, request: BacktestRequest) -> Optional[BacktestResult]:
        return None

    def _load_data(self, request: BacktestRequest) -> Any:
        return {"bars": 1000}  # Mock data

    def _run_simulation(self, data: Any, request: BacktestRequest) -> List[TradeRecord]:
        """Generate mock trades"""
        import random
        trades = []
        equity = self.config.initial_capital

        for i in range(50):  # 50 mock trades
            side = random.choice(["LONG", "SHORT"])
            pnl = random.gauss(10, 50)  # Mean $10, std $50
            equity += pnl

            trades.append(TradeRecord(
                trade_id=i,
                entry_time=datetime(2025, 1, 1, 9 + (i % 8), 0),
                exit_time=datetime(2025, 1, 1, 10 + (i % 8), 0),
                side=side,
                entry_price=4200.0,
                exit_price=4200.0 + (pnl / 10),
                pnl_usd=pnl,
                pnl_pct=pnl / equity,
                equity_at_entry=equity - pnl,
                equity_at_exit=equity,
            ))

        return trades

    def _calculate_metrics(
        self, trades: List[TradeRecord], start_date: date, end_date: date
    ) -> BacktestMetrics:
        # Delegate to parent
        return OrchestratorBacktestRunner._calculate_metrics(self, trades, start_date, end_date)

    def _persist_results(self, trades: List[TradeRecord], request: BacktestRequest) -> None:
        logger.info(f"Mock: Would persist {len(trades)} trades")


# =============================================================================
# FACTORY
# =============================================================================

class BacktestRunnerFactory:
    """
    Factory for creating backtest runners.

    Implements Factory Pattern with Registry.
    """

    _runners: Dict[str, Type[AbstractBacktestRunner]] = {
        "evaluation": EvaluationBacktestRunner,  # Primary runner using BacktestEngine
        "orchestrator": OrchestratorBacktestRunner,  # Legacy - requires external module
        "mock": MockBacktestRunner,
    }

    @classmethod
    def register(cls, name: str, runner_class: Type[AbstractBacktestRunner]) -> None:
        """Register a new runner type"""
        cls._runners[name] = runner_class
        logger.info(f"Registered backtest runner: {name}")

    @classmethod
    def create(
        cls,
        runner_type: str,
        config: BacktestConfig,
        project_root: Path,
        **kwargs
    ) -> AbstractBacktestRunner:
        """
        Create a backtest runner.

        Args:
            runner_type: Type of runner ("orchestrator", "mock")
            config: Backtest configuration
            project_root: Project root path
            **kwargs: Additional args for specific runners

        Returns:
            Configured BacktestRunner

        Raises:
            ValueError: If runner_type unknown
        """
        if runner_type not in cls._runners:
            raise ValueError(
                f"Unknown runner type: {runner_type}. "
                f"Available: {list(cls._runners.keys())}"
            )

        runner_class = cls._runners[runner_type]
        return runner_class(config, project_root, **kwargs)

    @classmethod
    def list_runners(cls) -> List[str]:
        """List available runner types"""
        return list(cls._runners.keys())


# =============================================================================
# BUILDER - Fluent Configuration
# =============================================================================

class BacktestConfigBuilder:
    """
    Builder for BacktestConfig with fluent API.

    Usage:
        config = (BacktestConfigBuilder()
            .with_model("ppo_model")
            .with_capital(10000)
            .with_thresholds(long=0.33, short=-0.33)
            .build())
    """

    def __init__(self):
        # Import SSOT config for canonical values
        try:
            from src.config.backtest_ssot import BACKTEST_SSOT
            ssot = BACKTEST_SSOT
        except ImportError:
            # Fallback if SSOT not available - use canonical values directly
            ssot = None

        self._model_id: str = "ppo_latest"
        self._model_path: Optional[str] = None
        self._norm_stats_path: Optional[str] = None
        # SSOT: Use canonical config values for consistency with L4
        self._initial_capital: float = ssot.initial_capital if ssot else 10_000.0
        self._transaction_cost_bps: float = ssot.spread_bps if ssot else 2.5  # SSOT: 2.5 bps
        self._slippage_bps: float = ssot.slippage_bps if ssot else 1.0  # SSOT: 1.0 bps
        self._long_entry: float = ssot.threshold_long if ssot else 0.50  # SSOT: matches training
        self._short_entry: float = ssot.threshold_short if ssot else -0.50  # SSOT: matches training
        self._exit: float = ssot.exit_threshold if ssot else 0.0  # SSOT: neutral
        self._stop_loss: float = ssot.stop_loss_pct if ssot else 0.025
        self._take_profit: float = ssot.take_profit_pct if ssot else 0.03
        self._max_bars: int = ssot.max_position_bars if ssot else 576

    def with_model(self, model_id: str, model_path: Optional[str] = None) -> "BacktestConfigBuilder":
        self._model_id = model_id
        self._model_path = model_path
        return self

    def with_norm_stats(self, path: str) -> "BacktestConfigBuilder":
        self._norm_stats_path = path
        return self

    def with_capital(self, amount: float) -> "BacktestConfigBuilder":
        self._initial_capital = amount
        return self

    def with_costs(self, transaction_bps: float, slippage_bps: float = 1.0) -> "BacktestConfigBuilder":
        self._transaction_cost_bps = transaction_bps
        self._slippage_bps = slippage_bps
        return self

    def with_thresholds(
        self,
        long: float = 0.50,  # SSOT: matches training
        short: float = -0.50,  # SSOT: matches training
        exit: float = 0.0  # SSOT: neutral zone
    ) -> "BacktestConfigBuilder":
        self._long_entry = long
        self._short_entry = short
        self._exit = exit
        return self

    def with_risk(
        self,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.03,
        max_bars: int = 20
    ) -> "BacktestConfigBuilder":
        self._stop_loss = stop_loss_pct
        self._take_profit = take_profit_pct
        self._max_bars = max_bars
        return self

    def build(self) -> BacktestConfig:
        """Build the configuration"""
        return BacktestConfig(
            model_id=self._model_id,
            model_path=self._model_path,
            norm_stats_path=self._norm_stats_path,
            initial_capital=self._initial_capital,
            transaction_cost_bps=self._transaction_cost_bps,
            slippage_bps=self._slippage_bps,
            long_entry_threshold=self._long_entry,
            short_entry_threshold=self._short_entry,
            exit_threshold=self._exit,
            stop_loss_pct=self._stop_loss,
            take_profit_pct=self._take_profit,
            max_position_bars=self._max_bars,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_backtest_runner(
    model_id: str = "ppo_latest",
    project_root: Optional[Path] = None,
    runner_type: str = "orchestrator",
    **config_kwargs
) -> AbstractBacktestRunner:
    """
    Convenience function to create a backtest runner.

    Args:
        model_id: Model identifier
        project_root: Project root path
        runner_type: Type of runner
        **config_kwargs: Additional config options

    Returns:
        Configured BacktestRunner
    """
    if project_root is None:
        project_root = Path("/opt/airflow")

    # SSOT: Use canonical config values for L4 consistency
    config = (BacktestConfigBuilder()
        .with_model(model_id)
        .with_capital(config_kwargs.get("initial_capital", 10_000.0))
        .with_costs(config_kwargs.get("transaction_cost_bps", 2.5))  # SSOT: 2.5 bps
        .build())

    return BacktestRunnerFactory.create(
        runner_type=runner_type,
        config=config,
        project_root=project_root,
        **config_kwargs
    )
