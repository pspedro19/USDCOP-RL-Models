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
        "orchestrator": OrchestratorBacktestRunner,
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
        self._model_id: str = "ppo_latest"
        self._model_path: Optional[str] = None
        self._norm_stats_path: Optional[str] = None
        self._initial_capital: float = 10_000.0
        self._transaction_cost_bps: float = 75.0  # realistic USDCOP spread
        self._slippage_bps: float = 15.0  # realistic slippage
        self._long_entry: float = 0.33  # matches training
        self._short_entry: float = -0.33  # matches training
        self._exit: float = 0.15  # exit threshold
        self._stop_loss: float = 0.02
        self._take_profit: float = 0.03
        self._max_bars: int = 20

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
        long: float = 0.33,  # matches training
        short: float = -0.33,  # matches training
        exit: float = 0.15  # adjusted
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

    config = (BacktestConfigBuilder()
        .with_model(model_id)
        .with_capital(config_kwargs.get("initial_capital", 10_000.0))
        .with_costs(config_kwargs.get("transaction_cost_bps", 75.0))  # 75 bps
        .build())

    return BacktestRunnerFactory.create(
        runner_type=runner_type,
        config=config,
        project_root=project_root,
        **config_kwargs
    )
