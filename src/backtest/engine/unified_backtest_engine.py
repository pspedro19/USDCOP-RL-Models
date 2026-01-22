"""
UnifiedBacktestEngine - SINGLE SOURCE OF TRUTH for All Backtesting
===================================================================

This module implements the canonical backtest engine that MUST be used for all
backtesting operations in the USDCOP trading system.

Design Principles:
- Single Source of Truth: ONE backtest implementation used everywhere
- No Look-Ahead Bias: Strict bar-by-bar processing with data isolation
- Realistic Execution: Transaction costs and slippage modeling
- Reproducible: Same config + data = identical results
- Auditable: Full trade history with feature snapshots

CRITICAL INVARIANTS:
- Bar processing is strictly sequential (no peeking)
- All prices used for execution are adjusted for slippage
- Transaction costs are applied on every position change
- Position size is normalized to [-1, 1]
- Feature calculation uses CanonicalFeatureBuilder (SSOT)

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    UnifiedBacktestEngine                        │
    │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
    │  │BacktestConfig│  │ ModelLoader  │  │CanonicalFeature-   │    │
    │  │  (immutable) │  │   (ONNX)     │  │    Builder         │    │
    │  └──────────────┘  └──────────────┘  └────────────────────┘    │
    │                            │                                    │
    │                            ▼                                    │
    │  for bar in data:                                               │
    │    1. _build_observation(bar)  ← NO look-ahead                  │
    │    2. _get_signal(model_output) → TradeDirection                │
    │    3. _execute_signal(direction) → Apply slippage + costs       │
    │    4. _update_equity()                                          │
    │                            │                                    │
    │                            ▼                                    │
    │  ┌────────────────────────────────────────────────────────┐    │
    │  │  BacktestResult: metrics, trades[], equity_curve       │    │
    │  └────────────────────────────────────────────────────────┘    │
    └─────────────────────────────────────────────────────────────────┘

Author: Trading Team
Version: 1.0.0
Created: 2025-01-17
Contract: CTR-BACKTEST-001
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Protocol, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# TRADE DIRECTION ENUM
# =============================================================================

class TradeDirection(IntEnum):
    """
    Trade direction enum.

    Uses integer values for easy arithmetic:
    - LONG = 1: Long position (buy)
    - SHORT = -1: Short position (sell)
    - FLAT = 0: No position

    Example:
        pnl = direction.value * (exit_price - entry_price) / entry_price
    """
    LONG = 1
    SHORT = -1
    FLAT = 0

    def __str__(self) -> str:
        return self.name

    @classmethod
    def from_action(cls, action: int) -> "TradeDirection":
        """
        Convert model action to TradeDirection.

        Assumes action mapping:
            0 = HOLD (FLAT)
            1 = BUY (LONG)
            2 = SELL (SHORT)

        Args:
            action: Model action index

        Returns:
            Corresponding TradeDirection
        """
        mapping = {0: cls.FLAT, 1: cls.LONG, 2: cls.SHORT}
        return mapping.get(action, cls.FLAT)


# =============================================================================
# CONFIGURATION DATACLASS (IMMUTABLE)
# =============================================================================

@dataclass(frozen=True)
class BacktestConfig:
    """
    Immutable backtest configuration.

    frozen=True ensures configuration cannot be modified after creation,
    preventing accidental state changes during backtest execution.

    Attributes:
        start_date: Backtest start datetime
        end_date: Backtest end datetime
        model_uri: Path or URI to model artifact (ONNX file)
        norm_stats_path: Path to normalization statistics JSON
        transaction_cost_bps: Transaction cost in basis points (default: 75 = 0.75%)
        slippage_bps: Slippage in basis points (default: 15 = 0.15%)
        initial_capital: Starting capital in USD (default: 100,000)
        position_size: Position size multiplier [0, 1] (default: 1.0)
        threshold_long: Signal threshold for long entry (default: 0.33)
        threshold_short: Signal threshold for short entry (default: -0.33)
        allow_short: Whether short positions are allowed (default: True)
        max_position_hold_bars: Maximum bars to hold position (None = no limit)

    Example:
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_uri="models/ppo_v1.onnx",
            norm_stats_path="models/norm_stats.json",
            transaction_cost_bps=75.0,
            slippage_bps=15.0,
        )
    """
    start_date: datetime
    end_date: datetime
    model_uri: str
    norm_stats_path: str
    transaction_cost_bps: float = 75.0
    slippage_bps: float = 15.0
    initial_capital: float = 100_000.0
    position_size: float = 1.0
    threshold_long: float = 0.33
    threshold_short: float = -0.33
    allow_short: bool = True
    max_position_hold_bars: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validation is tricky with frozen dataclass, but we can still raise
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < self.position_size <= 1:
            raise ValueError("position_size must be in (0, 1]")
        if self.transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps cannot be negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps cannot be negative")

    @property
    def transaction_cost_decimal(self) -> float:
        """Get transaction cost as decimal (bps / 10000)."""
        return self.transaction_cost_bps / 10000.0

    @property
    def slippage_decimal(self) -> float:
        """Get slippage as decimal (bps / 10000)."""
        return self.slippage_bps / 10000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "model_uri": self.model_uri,
            "norm_stats_path": self.norm_stats_path,
            "transaction_cost_bps": self.transaction_cost_bps,
            "slippage_bps": self.slippage_bps,
            "initial_capital": self.initial_capital,
            "position_size": self.position_size,
            "threshold_long": self.threshold_long,
            "threshold_short": self.threshold_short,
            "allow_short": self.allow_short,
            "max_position_hold_bars": self.max_position_hold_bars,
        }

    def get_config_hash(self) -> str:
        """Get deterministic hash of configuration for caching."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


# =============================================================================
# TRADE RECORD DATACLASS
# =============================================================================

@dataclass
class Trade:
    """
    Record of a single trade (position entry to exit).

    Captures complete trade lifecycle including:
    - Entry/exit timing and prices
    - Direction and P&L
    - Feature snapshot for audit/analysis
    - Execution metadata

    Attributes:
        trade_id: Unique trade identifier
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        entry_price: Entry price (after slippage)
        exit_price: Exit price (after slippage)
        direction: TradeDirection (LONG or SHORT)
        position_size: Normalized position size
        pnl_absolute: Absolute P&L in currency
        pnl_percent: P&L as percentage
        bars_held: Number of bars position was held
        entry_signal_confidence: Model confidence at entry
        exit_reason: Reason for exit (signal, stop, timeout, etc.)
        features_snapshot: Feature values at entry (for audit)
        transaction_costs: Total transaction costs paid
    """
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: TradeDirection
    position_size: float
    pnl_absolute: float
    pnl_percent: float
    bars_held: int
    entry_signal_confidence: float = 0.0
    exit_reason: str = "signal"
    features_snapshot: Optional[Dict[str, float]] = None
    transaction_costs: float = 0.0

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl_absolute > 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "direction": self.direction.name,
            "position_size": self.position_size,
            "pnl_absolute": self.pnl_absolute,
            "pnl_percent": self.pnl_percent,
            "bars_held": self.bars_held,
            "entry_signal_confidence": self.entry_signal_confidence,
            "exit_reason": self.exit_reason,
            "features_snapshot": self.features_snapshot,
            "transaction_costs": self.transaction_costs,
        }


# =============================================================================
# BACKTEST METRICS DATACLASS
# =============================================================================

@dataclass
class BacktestMetrics:
    """
    Comprehensive backtest performance metrics.

    All metrics are calculated from daily returns to ensure
    comparability with standard financial metrics.

    Attributes:
        total_return: Total return as decimal (0.25 = 25%)
        sharpe_ratio: Annualized Sharpe ratio (risk-adjusted return)
        sortino_ratio: Annualized Sortino ratio (downside risk-adjusted)
        max_drawdown: Maximum drawdown as decimal (0.15 = 15%)
        win_rate: Percentage of winning trades (0-100)
        profit_factor: Gross profit / Gross loss
        calmar_ratio: CAGR / Max Drawdown
        total_trades: Total number of completed trades
        avg_trade_pnl: Average P&L per trade
        avg_bars_held: Average bars held per trade
    """
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    total_trades: int
    avg_trade_pnl: float
    avg_bars_held: float

    # Additional metrics for detailed analysis
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        return {
            "total_return": round(self.total_return, 6),
            "total_return_pct": round(self.total_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate": round(self.win_rate, 2),
            "profit_factor": round(self.profit_factor, 4),
            "calmar_ratio": round(self.calmar_ratio, 4),
            "total_trades": self.total_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "avg_bars_held": round(self.avg_bars_held, 2),
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "gross_profit": round(self.gross_profit, 2),
            "gross_loss": round(self.gross_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "avg_win": round(self.avg_win, 4),
            "avg_loss": round(self.avg_loss, 4),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"BacktestMetrics(\n"
            f"  Total Return: {self.total_return*100:.2f}%\n"
            f"  Sharpe Ratio: {self.sharpe_ratio:.3f}\n"
            f"  Sortino Ratio: {self.sortino_ratio:.3f}\n"
            f"  Max Drawdown: {self.max_drawdown*100:.2f}%\n"
            f"  Win Rate: {self.win_rate:.1f}%\n"
            f"  Profit Factor: {self.profit_factor:.2f}\n"
            f"  Calmar Ratio: {self.calmar_ratio:.3f}\n"
            f"  Total Trades: {self.total_trades}\n"
            f"  Avg Trade P&L: {self.avg_trade_pnl*100:.3f}%\n"
            f"  Avg Bars Held: {self.avg_bars_held:.1f}\n"
            f")"
        )


# =============================================================================
# BACKTEST RESULT DATACLASS
# =============================================================================

@dataclass
class BacktestResult:
    """
    Complete backtest output.

    Contains all information needed to analyze and reproduce a backtest:
    - Configuration used
    - Performance metrics
    - Full trade history
    - Equity curve time series
    - Daily returns series

    Attributes:
        config: BacktestConfig used
        metrics: BacktestMetrics calculated
        trades: List of Trade objects
        equity_curve: DataFrame with timestamp, equity, drawdown columns
        daily_returns: Series of daily returns
        execution_time_seconds: Time taken to run backtest
        bars_processed: Number of data bars processed
        start_timestamp: Actual first bar timestamp
        end_timestamp: Actual last bar timestamp
    """
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: pd.DataFrame
    daily_returns: pd.Series
    execution_time_seconds: float = 0.0
    bars_processed: int = 0
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "trades": [t.to_dict() for t in self.trades],
            "execution_time_seconds": round(self.execution_time_seconds, 2),
            "bars_processed": self.bars_processed,
            "start_timestamp": self.start_timestamp.isoformat() if self.start_timestamp else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save result to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        result_dict = self.to_dict()
        result_dict["equity_curve"] = self.equity_curve.to_dict(orient="records")
        result_dict["daily_returns"] = self.daily_returns.to_dict()

        with open(path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2, default=str)

        logger.info(f"Backtest result saved to {path}")

    def summary(self) -> str:
        """Get human-readable summary."""
        return (
            f"\n{'='*60}\n"
            f"BACKTEST RESULT SUMMARY\n"
            f"{'='*60}\n"
            f"Period: {self.start_timestamp} to {self.end_timestamp}\n"
            f"Bars Processed: {self.bars_processed:,}\n"
            f"Execution Time: {self.execution_time_seconds:.2f}s\n"
            f"{'='*60}\n"
            f"{self.metrics}\n"
            f"{'='*60}\n"
        )


# =============================================================================
# MODEL LOADER PROTOCOL (STUB)
# =============================================================================

class IBacktestModelLoader(Protocol):
    """Protocol for model loaders used in backtesting."""

    def load(self, path: str) -> bool:
        """Load model from path."""
        ...

    def predict(self, observation: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Run prediction.

        Returns:
            Tuple of (action_index, confidence, action_probabilities)
        """
        ...

    def is_loaded(self) -> bool:
        """Check if model is ready."""
        ...


class StubModelLoader:
    """
    Stub model loader for when ONNX runtime is not available.

    Always returns HOLD signal with 0.5 confidence.
    Use this for testing the backtest infrastructure without models.
    """

    def __init__(self) -> None:
        self._loaded = False
        self._model_path: Optional[str] = None

    def load(self, path: str) -> bool:
        """Simulate loading a model."""
        self._model_path = path
        self._loaded = True
        logger.warning(f"StubModelLoader: Simulating model load from {path}")
        return True

    def predict(self, observation: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Return dummy prediction (HOLD with 0.5 confidence).

        For real backtests, use ONNXModelLoader instead.
        """
        action = 0  # HOLD
        confidence = 0.5
        probs = np.array([0.34, 0.33, 0.33], dtype=np.float32)
        return action, confidence, probs

    def is_loaded(self) -> bool:
        return self._loaded


# =============================================================================
# FEATURE BUILDER PROTOCOL (STUB)
# =============================================================================

class IBacktestFeatureBuilder(Protocol):
    """Protocol for feature builders used in backtesting."""

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame],
        position: float,
        bar_idx: int,
    ) -> np.ndarray:
        """Build observation vector for model input."""
        ...

    def get_observation_dim(self) -> int:
        """Get expected observation dimension."""
        ...


class StubFeatureBuilder:
    """
    Stub feature builder for when CanonicalFeatureBuilder is not available.

    Returns random observations of correct dimension.
    Use this for testing the backtest infrastructure without features.
    """

    OBSERVATION_DIM: Final[int] = 15

    def __init__(self) -> None:
        logger.warning("StubFeatureBuilder: Using stub implementation")

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: Optional[pd.DataFrame],
        position: float,
        bar_idx: int,
    ) -> np.ndarray:
        """Return dummy observation (zeros)."""
        obs = np.zeros(self.OBSERVATION_DIM, dtype=np.float32)
        obs[-2] = position  # Position slot
        obs[-1] = 0.5  # Time normalized
        return obs

    def get_observation_dim(self) -> int:
        return self.OBSERVATION_DIM


# =============================================================================
# UNIFIED BACKTEST ENGINE - SINGLE SOURCE OF TRUTH
# =============================================================================

class UnifiedBacktestEngine:
    """
    UNIFIED Backtest Engine - Single Source of Truth for ALL backtesting.

    This class is the ONLY authorized implementation for running backtests.
    All other backtest code should delegate to this class.

    Key Features:
    - No look-ahead bias: Strict bar-by-bar processing
    - Realistic execution: Slippage and transaction costs
    - Position management: Entry, exit, max hold duration
    - Feature parity: Uses CanonicalFeatureBuilder (SSOT)
    - Audit trail: Full trade history with feature snapshots

    Usage:
        config = BacktestConfig(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_uri="models/ppo_v1.onnx",
            norm_stats_path="models/norm_stats.json",
        )

        engine = UnifiedBacktestEngine(config)
        result = engine.run(ohlcv_df, macro_df)

        print(result.summary())
        result.save("backtest_results/run_001.json")

    Invariants:
    - _process_bar() only sees data up to current bar index
    - Position changes trigger slippage and cost calculation
    - Equity is updated at end of each bar
    - All trades are recorded with complete metadata
    """

    # WARMUP_BARS matches CanonicalFeatureBuilder requirement
    WARMUP_BARS: Final[int] = 60

    def __init__(
        self,
        config: BacktestConfig,
        model_loader: Optional[IBacktestModelLoader] = None,
        feature_builder: Optional[IBacktestFeatureBuilder] = None,
    ):
        """
        Initialize UnifiedBacktestEngine.

        Args:
            config: Immutable backtest configuration
            model_loader: Model loader (uses stub if None)
            feature_builder: Feature builder (uses stub if None)
        """
        self._config = config

        # Initialize model loader (try real, fallback to stub)
        if model_loader is not None:
            self._model_loader = model_loader
        else:
            self._model_loader = self._create_model_loader()

        # Initialize feature builder (try real, fallback to stub)
        if feature_builder is not None:
            self._feature_builder = feature_builder
        else:
            self._feature_builder = self._create_feature_builder()

        # State variables (reset on each run)
        self._equity: float = config.initial_capital
        self._position: TradeDirection = TradeDirection.FLAT
        self._position_entry_price: float = 0.0
        self._position_entry_time: Optional[datetime] = None
        self._position_entry_bar: int = 0
        self._position_entry_confidence: float = 0.0
        self._position_features: Optional[Dict[str, float]] = None

        # Trade tracking
        self._trades: List[Trade] = []
        self._trade_counter: int = 0
        self._equity_history: List[Tuple[datetime, float]] = []

        logger.info(f"UnifiedBacktestEngine initialized with config hash: {config.get_config_hash()}")

    def _create_model_loader(self) -> IBacktestModelLoader:
        """Create model loader, with fallback to stub."""
        try:
            from src.inference.model_loader import ONNXModelLoader

            loader = ONNXModelLoader(name="backtest_model")
            if loader.load(self._config.model_uri):
                logger.info(f"Loaded ONNX model from {self._config.model_uri}")
                return _ONNXModelLoaderAdapter(loader)
            else:
                logger.warning("Failed to load ONNX model, using stub")
                return StubModelLoader()
        except ImportError:
            logger.warning("ONNXModelLoader not available, using stub")
            return StubModelLoader()
        except Exception as e:
            logger.warning(f"Error creating model loader: {e}, using stub")
            return StubModelLoader()

    def _create_feature_builder(self) -> IBacktestFeatureBuilder:
        """Create feature builder, with fallback to stub."""
        try:
            from src.feature_store.builders import (
                CanonicalFeatureBuilder,
                BuilderContext,
            )

            builder = CanonicalFeatureBuilder.for_backtest(
                norm_stats_path=self._config.norm_stats_path
            )
            logger.info(f"Using CanonicalFeatureBuilder with hash: {builder.get_norm_stats_hash()[:12]}...")
            return builder
        except ImportError:
            logger.warning("CanonicalFeatureBuilder not available, using stub")
            return StubFeatureBuilder()
        except Exception as e:
            logger.warning(f"Error creating feature builder: {e}, using stub")
            return StubFeatureBuilder()

    @property
    def config(self) -> BacktestConfig:
        """Get backtest configuration."""
        return self._config

    def run(
        self,
        ohlcv_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame] = None,
    ) -> BacktestResult:
        """
        Run backtest on provided data.

        CRITICAL: This method processes bars sequentially to prevent look-ahead bias.
        Each bar only sees historical data up to that point.

        Args:
            ohlcv_df: DataFrame with OHLCV columns (open, high, low, close, volume)
                     Must have DatetimeIndex or 'timestamp' column
            macro_df: Optional DataFrame with macro indicators
                     (dxy, vix, embi, brent, usdmxn, treasury_10y)

        Returns:
            BacktestResult with metrics, trades, and equity curve

        Raises:
            ValueError: If data is insufficient or malformed
        """
        start_time = time.perf_counter()

        # Validate and prepare data
        ohlcv_df = self._validate_ohlcv(ohlcv_df)
        macro_df = self._validate_macro(macro_df, ohlcv_df) if macro_df is not None else None

        # Filter to date range
        ohlcv_df = self._filter_date_range(ohlcv_df)
        if macro_df is not None:
            macro_df = self._filter_date_range(macro_df)

        if len(ohlcv_df) < self.WARMUP_BARS + 1:
            raise ValueError(
                f"Insufficient data: {len(ohlcv_df)} bars, need at least {self.WARMUP_BARS + 1}"
            )

        # Reset state
        self._reset_state()

        # Load model if not already loaded
        if not self._model_loader.is_loaded():
            if not self._model_loader.load(self._config.model_uri):
                raise RuntimeError(f"Failed to load model from {self._config.model_uri}")

        # Record initial equity
        first_timestamp = ohlcv_df.index[self.WARMUP_BARS]
        self._equity_history.append((first_timestamp, self._equity))

        # Process bars sequentially (CRITICAL: no look-ahead)
        n_bars = len(ohlcv_df)
        for bar_idx in range(self.WARMUP_BARS, n_bars):
            self._process_bar(ohlcv_df, macro_df, bar_idx)

        # Close any remaining position at end
        if self._position != TradeDirection.FLAT:
            final_bar = n_bars - 1
            final_price = float(ohlcv_df["close"].iloc[final_bar])
            final_time = ohlcv_df.index[final_bar]
            self._close_position(final_price, final_time, final_bar, exit_reason="end_of_data")

        # Build results
        equity_df = self._build_equity_curve(ohlcv_df)
        daily_returns = self._calculate_daily_returns(equity_df)
        metrics = self._calculate_metrics(daily_returns)

        execution_time = time.perf_counter() - start_time

        result = BacktestResult(
            config=self._config,
            metrics=metrics,
            trades=self._trades.copy(),
            equity_curve=equity_df,
            daily_returns=daily_returns,
            execution_time_seconds=execution_time,
            bars_processed=n_bars - self.WARMUP_BARS,
            start_timestamp=ohlcv_df.index[self.WARMUP_BARS],
            end_timestamp=ohlcv_df.index[-1],
        )

        logger.info(
            f"Backtest complete: {result.bars_processed} bars, "
            f"{len(self._trades)} trades, "
            f"{metrics.total_return*100:.2f}% return, "
            f"{execution_time:.2f}s"
        )

        return result

    def _reset_state(self) -> None:
        """Reset all state variables for new backtest run."""
        self._equity = self._config.initial_capital
        self._position = TradeDirection.FLAT
        self._position_entry_price = 0.0
        self._position_entry_time = None
        self._position_entry_bar = 0
        self._position_entry_confidence = 0.0
        self._position_features = None
        self._trades = []
        self._trade_counter = 0
        self._equity_history = []

    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize OHLCV DataFrame."""
        required_cols = ["open", "high", "low", "close"]

        # Normalize column names
        df = df.copy()
        df.columns = df.columns.str.lower()

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Ensure datetime index
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        df = df.sort_index()

        # Check for NaN in critical columns
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            logger.warning(f"NaN values in OHLCV data: {nan_counts.to_dict()}")
            df = df.ffill()

        return df

    def _validate_macro(
        self, macro_df: pd.DataFrame, ohlcv_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Validate and align macro DataFrame."""
        macro_df = macro_df.copy()
        macro_df.columns = macro_df.columns.str.lower()

        # Ensure datetime index
        if "timestamp" in macro_df.columns:
            macro_df["timestamp"] = pd.to_datetime(macro_df["timestamp"])
            macro_df = macro_df.set_index("timestamp")
        elif not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df.index = pd.to_datetime(macro_df.index)

        macro_df = macro_df.sort_index()

        # Forward fill to align with OHLCV (macro data is typically less frequent)
        macro_df = macro_df.reindex(ohlcv_df.index, method="ffill")

        return macro_df

    def _filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to configured date range."""
        mask = (df.index >= self._config.start_date) & (df.index <= self._config.end_date)
        return df.loc[mask].copy()

    def _process_bar(
        self,
        ohlcv_df: pd.DataFrame,
        macro_df: Optional[pd.DataFrame],
        bar_idx: int,
    ) -> None:
        """
        Process single bar with NO look-ahead.

        CRITICAL: This method ONLY sees data up to bar_idx (inclusive).
        Future bars are never accessed.

        Args:
            ohlcv_df: Full OHLCV DataFrame
            macro_df: Full macro DataFrame (optional)
            bar_idx: Current bar index
        """
        # Get current bar data (NO look-ahead)
        current_time = ohlcv_df.index[bar_idx]
        current_price = float(ohlcv_df["close"].iloc[bar_idx])

        # Check max hold duration
        if (
            self._position != TradeDirection.FLAT
            and self._config.max_position_hold_bars is not None
        ):
            bars_held = bar_idx - self._position_entry_bar
            if bars_held >= self._config.max_position_hold_bars:
                self._close_position(
                    current_price, current_time, bar_idx, exit_reason="max_hold_duration"
                )

        # Build observation (only using data up to bar_idx)
        # Slice data to prevent any possible look-ahead
        ohlcv_slice = ohlcv_df.iloc[: bar_idx + 1]
        macro_slice = macro_df.iloc[: bar_idx + 1] if macro_df is not None else None

        try:
            observation = self._feature_builder.build_observation(
                ohlcv=ohlcv_slice,
                macro=macro_slice,
                position=float(self._position.value),
                bar_idx=len(ohlcv_slice) - 1,  # Last bar in slice
            )
        except Exception as e:
            logger.debug(f"Feature build failed at bar {bar_idx}: {e}")
            observation = np.zeros(
                self._feature_builder.get_observation_dim(), dtype=np.float32
            )

        # Get model prediction
        action, confidence, probs = self._model_loader.predict(observation)

        # Convert to signal
        signal = self._get_signal(action, confidence, probs)

        # Execute signal
        self._execute_signal(
            signal=signal,
            current_price=current_price,
            current_time=current_time,
            bar_idx=bar_idx,
            confidence=confidence,
            observation=observation,
        )

        # Update equity with current position mark-to-market
        self._update_equity(current_price, current_time)

    def _get_signal(
        self,
        action: int,
        confidence: float,
        probs: np.ndarray,
    ) -> TradeDirection:
        """
        Convert model output to TradeDirection using thresholds.

        Signal logic:
        - Action 1 (BUY) + confidence >= threshold_long -> LONG
        - Action 2 (SELL) + confidence >= abs(threshold_short) -> SHORT
        - Otherwise -> FLAT (no position change)

        Args:
            action: Model action index (0=HOLD, 1=BUY, 2=SELL)
            confidence: Model confidence [0, 1]
            probs: Action probabilities

        Returns:
            TradeDirection signal
        """
        # Calculate directional confidence
        # Positive = bullish, Negative = bearish
        if len(probs) >= 3:
            directional_confidence = probs[1] - probs[2]  # BUY prob - SELL prob
        else:
            directional_confidence = 0.0

        # Apply thresholds
        if action == 1 and directional_confidence >= self._config.threshold_long:
            return TradeDirection.LONG
        elif action == 2 and directional_confidence <= self._config.threshold_short:
            if self._config.allow_short:
                return TradeDirection.SHORT
            else:
                return TradeDirection.FLAT
        else:
            return TradeDirection.FLAT

    def _execute_signal(
        self,
        signal: TradeDirection,
        current_price: float,
        current_time: datetime,
        bar_idx: int,
        confidence: float,
        observation: np.ndarray,
    ) -> None:
        """
        Execute trading signal with position management.

        State machine:
        - FLAT + LONG signal -> Open long
        - FLAT + SHORT signal -> Open short
        - LONG + FLAT/SHORT signal -> Close long
        - SHORT + FLAT/LONG signal -> Close short
        - Same direction -> Hold (no action)

        Args:
            signal: Desired position direction
            current_price: Current close price
            current_time: Current timestamp
            bar_idx: Current bar index
            confidence: Model confidence
            observation: Feature observation
        """
        # No change needed
        if signal == self._position:
            return

        # Close existing position if any
        if self._position != TradeDirection.FLAT:
            self._close_position(current_price, current_time, bar_idx)

        # Open new position if signal is not FLAT
        if signal != TradeDirection.FLAT:
            self._open_position(
                direction=signal,
                price=current_price,
                time=current_time,
                bar_idx=bar_idx,
                confidence=confidence,
                observation=observation,
            )

    def _open_position(
        self,
        direction: TradeDirection,
        price: float,
        time: datetime,
        bar_idx: int,
        confidence: float,
        observation: np.ndarray,
    ) -> None:
        """
        Open new position with slippage.

        Slippage is adverse: pay more for longs, receive less for shorts.

        Args:
            direction: LONG or SHORT
            price: Market price
            time: Entry timestamp
            bar_idx: Entry bar index
            confidence: Model confidence at entry
            observation: Feature observation for audit
        """
        # Apply slippage (adverse direction)
        slippage = self._config.slippage_decimal
        if direction == TradeDirection.LONG:
            entry_price = price * (1 + slippage)
        else:  # SHORT
            entry_price = price * (1 - slippage)

        # Apply transaction cost
        self._equity *= (1 - self._config.transaction_cost_decimal)

        # Record position
        self._position = direction
        self._position_entry_price = entry_price
        self._position_entry_time = time
        self._position_entry_bar = bar_idx
        self._position_entry_confidence = confidence

        # Store features for audit (optional)
        self._position_features = {
            f"feature_{i}": float(v) for i, v in enumerate(observation)
        }

        logger.debug(
            f"Opened {direction.name} at {entry_price:.4f} "
            f"(market: {price:.4f}, confidence: {confidence:.3f})"
        )

    def _close_position(
        self,
        price: float,
        time: datetime,
        bar_idx: int,
        exit_reason: str = "signal",
    ) -> None:
        """
        Close existing position with slippage and cost calculation.

        Args:
            price: Market price at exit
            time: Exit timestamp
            bar_idx: Exit bar index
            exit_reason: Reason for closing (signal, stop, timeout, etc.)
        """
        if self._position == TradeDirection.FLAT:
            return

        # Apply slippage (adverse direction for exit)
        slippage = self._config.slippage_decimal
        if self._position == TradeDirection.LONG:
            exit_price = price * (1 - slippage)  # Sell lower
        else:  # SHORT
            exit_price = price * (1 + slippage)  # Buy back higher

        # Calculate P&L
        if self._position == TradeDirection.LONG:
            pnl_pct = (exit_price - self._position_entry_price) / self._position_entry_price
        else:  # SHORT
            pnl_pct = (self._position_entry_price - exit_price) / self._position_entry_price

        # Apply position size
        pnl_pct *= self._config.position_size

        # Transaction cost on exit
        exit_cost = self._config.transaction_cost_decimal
        total_cost = self._config.transaction_cost_decimal * 2  # Entry + exit

        # Apply P&L and exit cost to equity
        position_equity = self._equity * (1 + pnl_pct) * (1 - exit_cost)
        pnl_absolute = position_equity - self._equity
        self._equity = position_equity

        # Calculate bars held
        bars_held = bar_idx - self._position_entry_bar

        # Record trade
        self._trade_counter += 1
        trade = Trade(
            trade_id=self._trade_counter,
            entry_time=self._position_entry_time,
            exit_time=time,
            entry_price=self._position_entry_price,
            exit_price=exit_price,
            direction=self._position,
            position_size=self._config.position_size,
            pnl_absolute=pnl_absolute,
            pnl_percent=pnl_pct,
            bars_held=bars_held,
            entry_signal_confidence=self._position_entry_confidence,
            exit_reason=exit_reason,
            features_snapshot=self._position_features,
            transaction_costs=total_cost * self._equity,
        )
        self._trades.append(trade)

        logger.debug(
            f"Closed {self._position.name} at {exit_price:.4f} "
            f"(P&L: {pnl_pct*100:.3f}%, reason: {exit_reason})"
        )

        # Reset position state
        self._position = TradeDirection.FLAT
        self._position_entry_price = 0.0
        self._position_entry_time = None
        self._position_entry_bar = 0
        self._position_entry_confidence = 0.0
        self._position_features = None

    def _update_equity(self, current_price: float, current_time: datetime) -> None:
        """
        Update equity with mark-to-market for open position.

        Args:
            current_price: Current market price
            current_time: Current timestamp
        """
        if self._position != TradeDirection.FLAT:
            # Mark-to-market (unrealized P&L)
            if self._position == TradeDirection.LONG:
                unrealized_pnl_pct = (current_price - self._position_entry_price) / self._position_entry_price
            else:  # SHORT
                unrealized_pnl_pct = (self._position_entry_price - current_price) / self._position_entry_price

            unrealized_pnl_pct *= self._config.position_size

            # Record equity including unrealized
            mtm_equity = self._equity * (1 + unrealized_pnl_pct)
            self._equity_history.append((current_time, mtm_equity))
        else:
            self._equity_history.append((current_time, self._equity))

    def _build_equity_curve(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build equity curve DataFrame from history.

        Args:
            ohlcv_df: OHLCV DataFrame for reference

        Returns:
            DataFrame with timestamp index, equity, and drawdown columns
        """
        if not self._equity_history:
            return pd.DataFrame(columns=["equity", "drawdown"])

        # Create DataFrame from history
        timestamps, equities = zip(*self._equity_history)
        equity_df = pd.DataFrame({
            "equity": equities
        }, index=pd.DatetimeIndex(timestamps))

        # Calculate drawdown
        equity_df["peak"] = equity_df["equity"].cummax()
        equity_df["drawdown"] = (equity_df["peak"] - equity_df["equity"]) / equity_df["peak"]
        equity_df = equity_df.drop(columns=["peak"])

        return equity_df

    def _calculate_daily_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """
        Calculate daily returns from equity curve.

        Args:
            equity_df: Equity curve DataFrame

        Returns:
            Series of daily returns
        """
        if equity_df.empty:
            return pd.Series(dtype=float)

        # Resample to daily
        daily_equity = equity_df["equity"].resample("D").last().dropna()

        # Calculate returns
        daily_returns = daily_equity.pct_change().dropna()

        return daily_returns

    def _calculate_metrics(self, daily_returns: pd.Series) -> BacktestMetrics:
        """
        Calculate comprehensive backtest metrics.

        Args:
            daily_returns: Series of daily returns

        Returns:
            BacktestMetrics with all performance statistics
        """
        # Handle empty case
        if daily_returns.empty or len(self._trades) == 0:
            return BacktestMetrics(
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                calmar_ratio=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                avg_bars_held=0.0,
            )

        # Total return
        total_return = (self._equity / self._config.initial_capital) - 1

        # Sharpe ratio (annualized, assuming 252 trading days)
        if daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = (daily_returns.mean() / downside_returns.std()) * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio  # No downside = same as Sharpe

        # Max drawdown
        equity_curve = (1 + daily_returns).cumprod()
        peak = equity_curve.cummax()
        drawdown = (peak - equity_curve) / peak
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0

        # Trade statistics
        trade_pnls = [t.pnl_absolute for t in self._trades]
        trade_pcts = [t.pnl_percent for t in self._trades]
        trade_bars = [t.bars_held for t in self._trades]

        winning_trades = [p for p in trade_pnls if p > 0]
        losing_trades = [p for p in trade_pnls if p < 0]

        total_trades = len(self._trades)
        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0.0

        gross_profit = sum(winning_trades) if winning_trades else 0.0
        gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0

        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        if profit_factor == float("inf"):
            profit_factor = 10.0  # Cap for display

        # Calmar ratio (CAGR / Max Drawdown)
        if max_drawdown > 0:
            # Approximate CAGR
            n_years = len(daily_returns) / 252
            cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1 if n_years > 0 else 0
            calmar_ratio = cagr / max_drawdown
        else:
            calmar_ratio = 0.0

        avg_trade_pnl = np.mean(trade_pcts) if trade_pcts else 0.0
        avg_bars_held = np.mean(trade_bars) if trade_bars else 0.0

        # Consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(trade_pnls, positive=True)
        max_consecutive_losses = self._calculate_max_consecutive(trade_pnls, positive=False)

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            avg_trade_pnl=avg_trade_pnl,
            avg_bars_held=avg_bars_held,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            largest_win=max(winning_trades) if winning_trades else 0.0,
            largest_loss=min(losing_trades) if losing_trades else 0.0,
            avg_win=np.mean(winning_trades) if winning_trades else 0.0,
            avg_loss=np.mean(losing_trades) if losing_trades else 0.0,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
        )

    def _calculate_max_consecutive(
        self, pnls: List[float], positive: bool
    ) -> int:
        """Calculate maximum consecutive wins or losses."""
        max_streak = 0
        current_streak = 0

        for pnl in pnls:
            if (positive and pnl > 0) or (not positive and pnl < 0):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak


# =============================================================================
# ADAPTER FOR ONNX MODEL LOADER
# =============================================================================

class _ONNXModelLoaderAdapter:
    """Adapter to make ONNXModelLoader compatible with IBacktestModelLoader."""

    def __init__(self, loader) -> None:
        self._loader = loader

    def load(self, path: str) -> bool:
        return self._loader.is_loaded() or self._loader.load(path)

    def predict(self, observation: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """Run prediction and return (action, confidence, probs)."""
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        observation = observation.astype(np.float32)

        session = self._loader.session
        outputs = session.run(None, {self._loader.input_name: observation})

        action_probs = outputs[0][0]
        action = int(np.argmax(action_probs))
        confidence = float(np.max(action_probs))

        return action, confidence, action_probs

    def is_loaded(self) -> bool:
        return self._loader.is_loaded()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_backtest_engine(
    start_date: datetime,
    end_date: datetime,
    model_uri: str,
    norm_stats_path: str,
    transaction_cost_bps: float = 75.0,
    slippage_bps: float = 15.0,
    initial_capital: float = 100_000.0,
    position_size: float = 1.0,
    threshold_long: float = 0.33,
    threshold_short: float = -0.33,
    allow_short: bool = True,
    max_position_hold_bars: Optional[int] = None,
) -> UnifiedBacktestEngine:
    """
    Factory function to create UnifiedBacktestEngine.

    Convenience function that creates BacktestConfig and engine in one call.

    Args:
        start_date: Backtest start datetime
        end_date: Backtest end datetime
        model_uri: Path to model artifact
        norm_stats_path: Path to normalization statistics
        transaction_cost_bps: Transaction cost in basis points
        slippage_bps: Slippage in basis points
        initial_capital: Starting capital
        position_size: Position size multiplier
        threshold_long: Long entry threshold
        threshold_short: Short entry threshold
        allow_short: Whether to allow short positions
        max_position_hold_bars: Maximum hold duration

    Returns:
        Configured UnifiedBacktestEngine instance

    Example:
        engine = create_backtest_engine(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
            model_uri="models/ppo_v1.onnx",
            norm_stats_path="models/norm_stats.json",
        )
        result = engine.run(ohlcv_df, macro_df)
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        model_uri=model_uri,
        norm_stats_path=norm_stats_path,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        initial_capital=initial_capital,
        position_size=position_size,
        threshold_long=threshold_long,
        threshold_short=threshold_short,
        allow_short=allow_short,
        max_position_hold_bars=max_position_hold_bars,
    )

    return UnifiedBacktestEngine(config)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    # Quick test with synthetic data
    import random

    logging.basicConfig(level=logging.INFO)

    # Generate synthetic OHLCV data
    n_bars = 500
    base_price = 4200.0

    timestamps = pd.date_range(
        start="2024-01-01", periods=n_bars, freq="5min"
    )

    random.seed(42)
    np.random.seed(42)

    prices = [base_price]
    for _ in range(n_bars - 1):
        change = random.gauss(0, 0.001)
        prices.append(prices[-1] * (1 + change))

    ohlcv_df = pd.DataFrame({
        "open": prices,
        "high": [p * 1.001 for p in prices],
        "low": [p * 0.999 for p in prices],
        "close": prices,
        "volume": [random.randint(1000, 10000) for _ in prices],
    }, index=timestamps)

    # Create engine with stub model
    config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 12, 31),
        model_uri="models/ppo_v1.onnx",
        norm_stats_path="models/norm_stats.json",
        transaction_cost_bps=75.0,
        slippage_bps=15.0,
    )

    engine = UnifiedBacktestEngine(config)

    print("Running backtest with synthetic data...")
    result = engine.run(ohlcv_df)

    print(result.summary())
    print(f"Trade count: {len(result.trades)}")

    print("\nUnifiedBacktestEngine test complete!")
