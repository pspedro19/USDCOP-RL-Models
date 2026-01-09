"""
Paper Trader - Simulated Trading Execution System
==================================================

Provides paper trading functionality for validating trading strategies
without executing real orders. Tracks positions, calculates PnL,
and persists trade data to PostgreSQL for dashboard visualization.

Features:
- Multi-model position tracking (independent positions per model)
- Accurate PnL calculation for LONG and SHORT positions
- Equity curve tracking
- Comprehensive statistics
- PostgreSQL persistence to trading_metrics table
- Detailed logging for debugging and auditing

Table Schema (trading_metrics):
- timestamp: TIMESTAMPTZ
- metric_name: TEXT (uses 'paper_trade_pnl')
- metric_value: NUMERIC (the PnL value)
- metric_type: TEXT (uses 'paper_trading')
- strategy_name: TEXT (model_id)
- metadata: JSONB (complete trade details)

Author: USD/COP Trading System
Version: 1.0.0
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging
import json
import copy

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Direction of a trade position."""
    LONG = "LONG"
    SHORT = "SHORT"
    CLOSE = "CLOSE"


@dataclass
class PaperTrade:
    """
    Represents a simulated paper trade.

    Attributes:
        trade_id: Unique identifier for the trade
        model_id: ID of the model that generated the signal
        signal: Original signal type (LONG, SHORT, CLOSE)
        side: Execution side (buy, sell)
        entry_price: Price at which position was opened
        entry_time: Timestamp when position was opened
        exit_price: Price at which position was closed (if closed)
        exit_time: Timestamp when position was closed (if closed)
        pnl: Profit/Loss in absolute terms
        pnl_pct: Profit/Loss as percentage
        status: Current status (open, closed)
        size: Position size in units
        direction: LONG or SHORT direction
    """
    trade_id: int
    model_id: str
    signal: str
    side: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "open"
    size: float = 1.0
    direction: str = "LONG"

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for serialization."""
        result = asdict(self)
        # Convert datetime objects to ISO format strings
        if self.entry_time:
            result['entry_time'] = self.entry_time.isoformat()
        if self.exit_time:
            result['exit_time'] = self.exit_time.isoformat()
        return result

    def to_json(self) -> str:
        """Convert trade to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PaperTrade':
        """Create PaperTrade from dictionary."""
        if isinstance(data.get('entry_time'), str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if isinstance(data.get('exit_time'), str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        return cls(**data)


class PaperTrader:
    """
    Simulates trade execution without placing real orders.

    Manages positions for multiple models independently, calculates
    PnL accurately for both LONG and SHORT positions, and persists
    trade data to PostgreSQL for dashboard visualization.

    Attributes:
        initial_capital: Starting capital for paper trading
        current_capital: Current capital after trades
        db_connection: Optional PostgreSQL connection for persistence
        positions: Dict mapping model_id to open positions
        trade_history: List of all completed trades
        equity_curve: Historical equity values
    """

    METRIC_NAME = "paper_trade_pnl"
    METRIC_TYPE = "paper_trading"

    def __init__(
        self,
        initial_capital: float = 10000.0,
        db_connection: Any = None,
        position_size_pct: float = 0.1,
        enable_short: bool = True
    ):
        """
        Initialize the PaperTrader.

        Args:
            initial_capital: Starting capital for paper trading
            db_connection: Optional PostgreSQL connection for persistence
            position_size_pct: Percentage of capital to use per trade (0.1 = 10%)
            enable_short: Whether to allow short positions
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.db_connection = db_connection
        self.position_size_pct = position_size_pct
        self.enable_short = enable_short

        # Position tracking by model
        self._positions: Dict[str, PaperTrade] = {}

        # Trade history
        self._trade_history: List[PaperTrade] = []

        # Trade ID counter
        self._trade_counter = 0

        # Equity curve (capital after each trade)
        self._equity_curve: List[Dict[str, Any]] = [
            {"timestamp": datetime.now(), "equity": initial_capital}
        ]

        # Model-specific equity tracking
        self._model_equity: Dict[str, float] = {}

        logger.info(
            f"PaperTrader initialized with capital=${initial_capital:.2f}, "
            f"position_size={position_size_pct*100:.1f}%, "
            f"short_enabled={enable_short}"
        )

    def _generate_trade_id(self) -> int:
        """Generate unique trade ID."""
        self._trade_counter += 1
        return self._trade_counter

    def _calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on current capital.

        Args:
            price: Current market price

        Returns:
            Number of units to trade
        """
        capital_per_trade = self.current_capital * self.position_size_pct
        return capital_per_trade / price

    def execute_signal(
        self,
        model_id: str,
        signal: str,
        current_price: float,
        timestamp: Optional[datetime] = None
    ) -> Optional[PaperTrade]:
        """
        Execute a trading signal from a model.

        Handles LONG, SHORT, and CLOSE signals. If a position exists
        and the new signal is opposite direction, closes the existing
        position first.

        Args:
            model_id: Identifier of the model generating the signal
            signal: Trading signal (LONG, SHORT, CLOSE, HOLD)
            current_price: Current market price
            timestamp: Optional timestamp (defaults to now)

        Returns:
            PaperTrade object if a trade was executed, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.now()

        signal_upper = signal.upper().strip()

        logger.debug(
            f"Executing signal: model={model_id}, signal={signal_upper}, "
            f"price={current_price:.4f}"
        )

        # HOLD signal - no action
        if signal_upper == "HOLD":
            logger.debug(f"Model {model_id}: HOLD signal, no action taken")
            return None

        # Check for existing position
        has_position = model_id in self._positions
        current_position = self._positions.get(model_id)

        # CLOSE signal
        if signal_upper == "CLOSE":
            if has_position:
                return self._close_position(model_id, current_price, timestamp)
            else:
                logger.warning(
                    f"Model {model_id}: CLOSE signal but no open position"
                )
                return None

        # LONG signal
        if signal_upper == "LONG":
            if has_position:
                if current_position.direction == "LONG":
                    # Already long, hold position
                    logger.debug(
                        f"Model {model_id}: Already LONG, holding position"
                    )
                    return None
                else:
                    # Close SHORT and open LONG
                    logger.info(
                        f"Model {model_id}: Closing SHORT to open LONG"
                    )
                    self._close_position(model_id, current_price, timestamp)

            return self._open_position(
                model_id, "LONG", current_price, timestamp
            )

        # SHORT signal
        if signal_upper == "SHORT":
            if not self.enable_short:
                logger.warning(
                    f"Model {model_id}: SHORT signal but short trading disabled"
                )
                return None

            if has_position:
                if current_position.direction == "SHORT":
                    # Already short, hold position
                    logger.debug(
                        f"Model {model_id}: Already SHORT, holding position"
                    )
                    return None
                else:
                    # Close LONG and open SHORT
                    logger.info(
                        f"Model {model_id}: Closing LONG to open SHORT"
                    )
                    self._close_position(model_id, current_price, timestamp)

            return self._open_position(
                model_id, "SHORT", current_price, timestamp
            )

        logger.warning(f"Unknown signal: {signal_upper}")
        return None

    def _open_position(
        self,
        model_id: str,
        direction: str,
        price: float,
        timestamp: datetime
    ) -> PaperTrade:
        """
        Open a new position.

        Args:
            model_id: Identifier of the model
            direction: LONG or SHORT
            price: Entry price
            timestamp: Entry timestamp

        Returns:
            Created PaperTrade
        """
        trade_id = self._generate_trade_id()
        size = self._calculate_position_size(price)

        # Side is "buy" for LONG, "sell" for SHORT (opening direction)
        side = "buy" if direction == "LONG" else "sell"

        trade = PaperTrade(
            trade_id=trade_id,
            model_id=model_id,
            signal=direction,
            side=side,
            entry_price=price,
            entry_time=timestamp,
            status="open",
            size=size,
            direction=direction
        )

        self._positions[model_id] = trade

        logger.info(
            f"OPEN {direction}: model={model_id}, trade_id={trade_id}, "
            f"price={price:.4f}, size={size:.4f}"
        )

        return trade

    def _close_position(
        self,
        model_id: str,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> PaperTrade:
        """
        Close an existing position.

        Calculates PnL based on direction:
        - LONG: PnL = (exit_price - entry_price) * size
        - SHORT: PnL = (entry_price - exit_price) * size

        Args:
            model_id: Identifier of the model
            price: Exit price
            timestamp: Exit timestamp

        Returns:
            Closed PaperTrade with calculated PnL
        """
        if timestamp is None:
            timestamp = datetime.now()

        if model_id not in self._positions:
            raise ValueError(f"No open position for model {model_id}")

        trade = self._positions.pop(model_id)

        # Calculate PnL based on direction
        if trade.direction == "LONG":
            # LONG: profit when price goes up
            pnl = (price - trade.entry_price) * trade.size
        else:
            # SHORT: profit when price goes down
            pnl = (trade.entry_price - price) * trade.size

        # Calculate PnL percentage
        position_value = trade.entry_price * trade.size
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0.0

        # Update trade
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.pnl_pct = pnl_pct
        trade.status = "closed"
        trade.side = "sell" if trade.direction == "LONG" else "buy"

        # Update capital
        self.current_capital += pnl

        # Record in history
        self._trade_history.append(trade)

        # Update equity curve
        self._equity_curve.append({
            "timestamp": timestamp,
            "equity": self.current_capital,
            "trade_id": trade.trade_id,
            "model_id": model_id
        })

        # Update model-specific equity
        if model_id not in self._model_equity:
            self._model_equity[model_id] = self.initial_capital
        self._model_equity[model_id] += pnl

        logger.info(
            f"CLOSE {trade.direction}: model={model_id}, "
            f"trade_id={trade.trade_id}, "
            f"entry={trade.entry_price:.4f}, exit={price:.4f}, "
            f"PnL=${pnl:.2f} ({pnl_pct:+.2f}%)"
        )

        # Persist to database
        self._persist_trade(trade)

        return trade

    def _persist_trade(self, trade: PaperTrade) -> bool:
        """
        Persist trade to trading_metrics table for dashboard.

        INSERT INTO trading_metrics (
            timestamp, metric_name, metric_value,
            metric_type, strategy_name, metadata
        )

        Args:
            trade: Completed PaperTrade to persist

        Returns:
            True if successful, False otherwise
        """
        if self.db_connection is None:
            logger.debug("No database connection, skipping persistence")
            return False

        try:
            # Prepare metadata as JSONB
            metadata = {
                "trade_id": trade.trade_id,
                "signal": trade.signal,
                "side": trade.side,
                "direction": trade.direction,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "entry_time": trade.entry_time.isoformat() if trade.entry_time else None,
                "exit_time": trade.exit_time.isoformat() if trade.exit_time else None,
                "size": trade.size,
                "pnl_pct": trade.pnl_pct,
                "status": trade.status,
                "current_capital": self.current_capital
            }

            query = """
                INSERT INTO trading_metrics (
                    timestamp, metric_name, metric_value,
                    metric_type, strategy_name, metadata
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
            """

            cursor = self.db_connection.cursor()
            cursor.execute(
                query,
                (
                    trade.exit_time or datetime.now(),
                    self.METRIC_NAME,
                    trade.pnl,
                    self.METRIC_TYPE,
                    trade.model_id,
                    json.dumps(metadata)
                )
            )
            self.db_connection.commit()
            cursor.close()

            logger.debug(
                f"Persisted trade {trade.trade_id} to trading_metrics"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to persist trade: {e}")
            try:
                self.db_connection.rollback()
            except Exception:
                pass
            return False

    def persist_trade_async(self, trade: PaperTrade) -> None:
        """
        Persist trade asynchronously (for psycopg async or asyncpg).

        Args:
            trade: Completed PaperTrade to persist
        """
        # For future async implementation
        self._persist_trade(trade)

    def get_open_positions(self) -> List[PaperTrade]:
        """
        Get all currently open positions.

        Returns:
            List of open PaperTrade objects
        """
        return list(self._positions.values())

    def get_open_position(self, model_id: str) -> Optional[PaperTrade]:
        """
        Get open position for a specific model.

        Args:
            model_id: Identifier of the model

        Returns:
            PaperTrade if position exists, None otherwise
        """
        return self._positions.get(model_id)

    def has_open_position(self, model_id: str) -> bool:
        """Check if model has an open position."""
        return model_id in self._positions

    def get_trade_history(
        self,
        model_id: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[PaperTrade]:
        """
        Get completed trade history.

        Args:
            model_id: Optional filter by model
            limit: Optional limit on number of trades

        Returns:
            List of completed PaperTrade objects
        """
        history = self._trade_history

        if model_id:
            history = [t for t in history if t.model_id == model_id]

        if limit:
            history = history[-limit:]

        return history

    def get_equity_curve(self) -> List[float]:
        """
        Get equity curve values.

        Returns:
            List of equity values over time
        """
        return [point["equity"] for point in self._equity_curve]

    def get_equity_curve_with_timestamps(self) -> List[Dict[str, Any]]:
        """
        Get equity curve with timestamps.

        Returns:
            List of dicts with timestamp and equity values
        """
        return copy.deepcopy(self._equity_curve)

    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate unrealized PnL for all open positions.

        Args:
            current_prices: Dict mapping model_id or symbol to current price

        Returns:
            Total unrealized PnL
        """
        unrealized = 0.0

        for model_id, trade in self._positions.items():
            # Try to get price by model_id or use a default key
            price = current_prices.get(model_id, current_prices.get("default"))

            if price is None:
                logger.warning(
                    f"No current price for model {model_id}, "
                    "skipping unrealized PnL"
                )
                continue

            if trade.direction == "LONG":
                unrealized += (price - trade.entry_price) * trade.size
            else:
                unrealized += (trade.entry_price - price) * trade.size

        return unrealized

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate comprehensive trading statistics.

        Returns:
            Dictionary with performance metrics
        """
        history = self._trade_history

        if not history:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "total_pnl_pct": 0.0,
                "average_pnl": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "current_capital": self.current_capital,
                "initial_capital": self.initial_capital,
                "return_pct": 0.0,
                "open_positions": len(self._positions)
            }

        # Basic counts
        total_trades = len(history)
        winning_trades = [t for t in history if t.pnl > 0]
        losing_trades = [t for t in history if t.pnl < 0]

        num_winners = len(winning_trades)
        num_losers = len(losing_trades)

        # Win rate
        win_rate = (num_winners / total_trades * 100) if total_trades > 0 else 0.0

        # PnL calculations
        total_pnl = sum(t.pnl for t in history)
        total_pnl_pct = sum(t.pnl_pct for t in history)
        average_pnl = total_pnl / total_trades if total_trades > 0 else 0.0

        # Average win/loss
        average_win = (
            sum(t.pnl for t in winning_trades) / num_winners
            if num_winners > 0 else 0.0
        )
        average_loss = (
            sum(t.pnl for t in losing_trades) / num_losers
            if num_losers > 0 else 0.0
        )

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else float('inf')
        )

        # Max drawdown
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio (simplified, daily returns)
        sharpe_ratio = self._calculate_sharpe_ratio()

        # Return percentage
        return_pct = (
            (self.current_capital - self.initial_capital) /
            self.initial_capital * 100
        )

        # Long vs Short breakdown
        long_trades = [t for t in history if t.direction == "LONG"]
        short_trades = [t for t in history if t.direction == "SHORT"]

        return {
            "total_trades": total_trades,
            "winning_trades": num_winners,
            "losing_trades": num_losers,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "average_pnl": round(average_pnl, 2),
            "average_win": round(average_win, 2),
            "average_loss": round(average_loss, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else None,
            "max_drawdown": round(max_drawdown, 2),
            "max_drawdown_pct": round(max_drawdown / self.initial_capital * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "current_capital": round(self.current_capital, 2),
            "initial_capital": self.initial_capital,
            "return_pct": round(return_pct, 2),
            "open_positions": len(self._positions),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl": round(sum(t.pnl for t in long_trades), 2),
            "short_pnl": round(sum(t.pnl for t in short_trades), 2)
        }

    def get_model_statistics(self, model_id: str) -> Dict[str, Any]:
        """
        Get statistics for a specific model.

        Args:
            model_id: Identifier of the model

        Returns:
            Dictionary with model-specific metrics
        """
        model_trades = [
            t for t in self._trade_history if t.model_id == model_id
        ]

        if not model_trades:
            return {
                "model_id": model_id,
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "has_open_position": self.has_open_position(model_id)
            }

        winners = [t for t in model_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in model_trades)
        win_rate = len(winners) / len(model_trades) * 100

        return {
            "model_id": model_id,
            "total_trades": len(model_trades),
            "winning_trades": len(winners),
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "average_pnl": round(total_pnl / len(model_trades), 2),
            "has_open_position": self.has_open_position(model_id),
            "model_equity": round(
                self._model_equity.get(model_id, self.initial_capital), 2
            )
        }

    def get_all_model_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all models that have traded.

        Returns:
            Dictionary mapping model_id to statistics
        """
        model_ids = set(t.model_id for t in self._trade_history)
        model_ids.update(self._positions.keys())

        return {
            model_id: self.get_model_statistics(model_id)
            for model_id in model_ids
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        if len(self._equity_curve) < 2:
            return 0.0

        equity_values = [p["equity"] for p in self._equity_curve]
        peak = equity_values[0]
        max_dd = 0.0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio from trade returns.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Annualized Sharpe ratio
        """
        if len(self._trade_history) < 2:
            return 0.0

        returns = [t.pnl_pct / 100 for t in self._trade_history]

        import statistics
        try:
            mean_return = statistics.mean(returns)
            std_return = statistics.stdev(returns)

            if std_return == 0:
                return 0.0

            # Annualize assuming ~252 trading days, average 5 trades per day
            trades_per_year = 252 * 5
            annualized_return = mean_return * trades_per_year
            annualized_std = std_return * (trades_per_year ** 0.5)

            sharpe = (annualized_return - risk_free_rate) / annualized_std
            return sharpe

        except statistics.StatisticsError:
            return 0.0

    def close_all_positions(
        self,
        current_price: float,
        timestamp: Optional[datetime] = None
    ) -> List[PaperTrade]:
        """
        Close all open positions.

        Args:
            current_price: Price to use for closing
            timestamp: Optional timestamp

        Returns:
            List of closed trades
        """
        if timestamp is None:
            timestamp = datetime.now()

        closed = []
        model_ids = list(self._positions.keys())

        for model_id in model_ids:
            try:
                trade = self._close_position(model_id, current_price, timestamp)
                closed.append(trade)
            except Exception as e:
                logger.error(f"Failed to close position for {model_id}: {e}")

        logger.info(f"Closed {len(closed)} positions at price {current_price}")
        return closed

    def reset(self) -> None:
        """Reset paper trader to initial state."""
        self._positions.clear()
        self._trade_history.clear()
        self._trade_counter = 0
        self.current_capital = self.initial_capital
        self._equity_curve = [
            {"timestamp": datetime.now(), "equity": self.initial_capital}
        ]
        self._model_equity.clear()

        logger.info(
            f"PaperTrader reset to initial capital ${self.initial_capital}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize paper trader state to dictionary."""
        return {
            "initial_capital": self.initial_capital,
            "current_capital": self.current_capital,
            "position_size_pct": self.position_size_pct,
            "enable_short": self.enable_short,
            "trade_counter": self._trade_counter,
            "open_positions": {
                k: v.to_dict() for k, v in self._positions.items()
            },
            "trade_history": [t.to_dict() for t in self._trade_history],
            "equity_curve": [
                {
                    "timestamp": p["timestamp"].isoformat()
                    if isinstance(p["timestamp"], datetime)
                    else p["timestamp"],
                    "equity": p["equity"]
                }
                for p in self._equity_curve
            ],
            "statistics": self.get_statistics()
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        db_connection: Any = None
    ) -> 'PaperTrader':
        """
        Restore paper trader from dictionary.

        Args:
            data: Serialized state dictionary
            db_connection: Optional database connection

        Returns:
            Restored PaperTrader instance
        """
        trader = cls(
            initial_capital=data["initial_capital"],
            db_connection=db_connection,
            position_size_pct=data.get("position_size_pct", 0.1),
            enable_short=data.get("enable_short", True)
        )

        trader.current_capital = data["current_capital"]
        trader._trade_counter = data.get("trade_counter", 0)

        # Restore positions
        for model_id, trade_data in data.get("open_positions", {}).items():
            trader._positions[model_id] = PaperTrade.from_dict(trade_data)

        # Restore history
        for trade_data in data.get("trade_history", []):
            trader._trade_history.append(PaperTrade.from_dict(trade_data))

        # Restore equity curve
        trader._equity_curve = []
        for point in data.get("equity_curve", []):
            ts = point["timestamp"]
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts)
            trader._equity_curve.append({
                "timestamp": ts,
                "equity": point["equity"]
            })

        return trader

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"PaperTrader(capital=${self.current_capital:.2f}, "
            f"trades={len(self._trade_history)}, "
            f"open_positions={len(self._positions)})"
        )
