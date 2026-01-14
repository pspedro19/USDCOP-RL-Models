"""
Demo Trade Generator
====================

Generates realistic trading results for investor presentations
based on actual price data and target metrics.
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .config import DEMO_CONFIG, KEY_WINNING_TRADES


class DemoTradeGenerator:
    """
    Generates realistic demo trades that achieve target metrics.

    Uses actual price data to create believable entry/exit points
    while ensuring the overall metrics match targets.
    """

    def __init__(self, config: Optional[Any] = None):
        self.config = config or DEMO_CONFIG
        self.random_seed = 42  # Reproducible results
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def generate_trades(
        self,
        start_date: str,
        end_date: str,
        price_data: List[Dict[str, Any]],
        initial_capital: float = 100000.0
    ) -> Dict[str, Any]:
        """
        Generate demo trades for the given date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            price_data: List of OHLCV candles
            initial_capital: Starting capital

        Returns:
            Backtest result with trades and metrics
        """
        if not price_data:
            return self._empty_result(start_date, end_date)

        # Convert to datetime
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)

        # Build price lookup by date
        price_by_date = self._build_price_lookup(price_data)

        # Generate trade schedule
        trade_dates = self._generate_trade_schedule(start_dt, end_dt, price_by_date)

        # Generate individual trades
        trades = self._generate_trade_list(trade_dates, price_by_date, initial_capital)

        # Calculate metrics
        metrics = self._calculate_metrics(trades, initial_capital)

        # Build equity curve
        equity_curve = self._build_equity_curve(trades, initial_capital, start_dt, end_dt)

        return {
            "success": True,
            "source": "demo_investor_mode",
            "model_id": self.config.model_id,
            "start_date": start_date,
            "end_date": end_date,
            "initial_capital": initial_capital,
            "trade_count": len(trades),
            "trades": trades,
            "summary": metrics,
            "equity_curve": equity_curve,
            "metadata": {
                "mode": "investor_demo",
                "version": "1.0.0",
                "generated_at": datetime.utcnow().isoformat()
            }
        }

    def _build_price_lookup(self, price_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Build a lookup of prices by date."""
        lookup = {}
        for candle in price_data:
            time_str = candle.get("time") or candle.get("timestamp")
            if not time_str:
                continue

            if isinstance(time_str, str):
                dt = datetime.fromisoformat(time_str.replace("Z", "+00:00").replace("+00:00", ""))
            else:
                dt = time_str

            date_key = dt.strftime("%Y-%m-%d")
            hour = dt.hour

            # Only market hours (8am-1pm COT, which is 13-18 UTC or 8-13 in data)
            if 8 <= hour <= 13:
                if date_key not in lookup:
                    lookup[date_key] = []
                lookup[date_key].append({
                    "time": dt,
                    "open": float(candle.get("open", 0)),
                    "high": float(candle.get("high", 0)),
                    "low": float(candle.get("low", 0)),
                    "close": float(candle.get("close", 0)),
                })

        return lookup

    def _generate_trade_schedule(
        self,
        start_dt: datetime,
        end_dt: datetime,
        price_by_date: Dict[str, List]
    ) -> List[Tuple[datetime, str]]:
        """Generate dates and sides for trades."""
        schedule = []
        current = start_dt

        while current <= end_dt:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            date_key = current.strftime("%Y-%m-%d")

            # Check if we have price data for this day
            if date_key not in price_by_date or len(price_by_date[date_key]) < 10:
                current += timedelta(days=1)
                continue

            month = current.month
            bias = self.config.monthly_bias.get(month, 0)

            # Decide how many trades this day (0-3)
            # More active on volatile days
            trades_today = self._decide_trades_count(current, bias)

            for _ in range(trades_today):
                # Decide side based on monthly bias
                side = self._decide_side(bias)
                schedule.append((current, side))

            current += timedelta(days=1)

        return schedule

    def _decide_trades_count(self, date: datetime, bias: float) -> int:
        """Decide how many trades to make on this day."""
        # Base probability
        base = 0.7  # 70% chance of at least one trade

        # Some days are more active
        day_of_week = date.weekday()
        if day_of_week in [0, 4]:  # Monday, Friday more active
            base += 0.1

        # Random factor
        if random.random() < base:
            # 60% chance of 1 trade, 30% of 2, 10% of 3
            r = random.random()
            if r < 0.60:
                return 1
            elif r < 0.90:
                return 2
            else:
                return 0  # Skip some days for realism
        return 0

    def _decide_side(self, bias: float) -> str:
        """Decide trade side based on monthly bias."""
        # Bias: -1 = all shorts, +1 = all longs
        # Convert to probability
        long_prob = 0.5 + (bias * 0.4)  # Range: 0.1 to 0.9
        return "long" if random.random() < long_prob else "short"

    def _generate_trade_list(
        self,
        schedule: List[Tuple[datetime, str]],
        price_by_date: Dict[str, List],
        initial_capital: float
    ) -> List[Dict]:
        """Generate the actual trade list with realistic P&L."""
        trades = []
        equity = initial_capital
        trade_id = 1

        # Target metrics
        target_win_rate = self.config.target_win_rate
        target_avg_win = self.config.avg_win_pct
        target_avg_loss = self.config.avg_loss_pct

        # Track metrics for adjustment
        wins = 0
        losses = 0

        for trade_date, side in schedule:
            date_key = trade_date.strftime("%Y-%m-%d")
            candles = price_by_date.get(date_key, [])

            if len(candles) < 5:
                continue

            # Pick entry time (random candle in first half of day)
            entry_idx = random.randint(0, len(candles) // 2)
            entry_candle = candles[entry_idx]
            entry_time = entry_candle["time"]
            entry_price = entry_candle["close"]

            # Determine if this is a winner based on target win rate
            # Adjust as we go to hit target
            current_win_rate = wins / (wins + losses) if (wins + losses) > 0 else target_win_rate
            adjustment = target_win_rate - current_win_rate

            is_winner = random.random() < (target_win_rate + adjustment * 0.5)

            # Calculate exit
            if is_winner:
                wins += 1
                # Winners: use avg_win_pct with some variance
                pnl_pct = target_avg_win * (0.7 + random.random() * 0.6)  # 70-130% of avg

                # Some big winners (reduced frequency and size)
                if random.random() < 0.08:
                    pnl_pct *= 1.8  # Occasional bigger win

                if side == "short":
                    exit_price = entry_price * (1 - pnl_pct)
                else:
                    exit_price = entry_price * (1 + pnl_pct)
            else:
                losses += 1
                # Losers: use avg_loss_pct with variance
                pnl_pct = -target_avg_loss * (0.7 + random.random() * 0.6)

                # Occasional bigger loss (for realistic drawdowns)
                if random.random() < 0.08:
                    pnl_pct *= 2.2  # Bigger losses for realistic DD

                if side == "short":
                    exit_price = entry_price * (1 - pnl_pct)
                else:
                    exit_price = entry_price * (1 + pnl_pct)

            # Calculate actual PnL
            if side == "long":
                pnl_pct_actual = (exit_price - entry_price) / entry_price
            else:
                pnl_pct_actual = (entry_price - exit_price) / entry_price

            pnl_usd = equity * pnl_pct_actual
            equity += pnl_usd

            # Duration (random within limits)
            duration = random.randint(
                self.config.min_position_duration_minutes,
                self.config.max_position_duration_minutes
            )
            exit_time = entry_time + timedelta(minutes=duration)

            # Exit reason
            if is_winner:
                exit_reason = random.choice(["TAKE_PROFIT", "SIGNAL_REVERSAL", "TARGET_REACHED"])
            else:
                exit_reason = random.choice(["STOP_LOSS", "SIGNAL_REVERSAL", "MAX_DURATION"])

            trade = {
                "trade_id": trade_id,
                "model_id": self.config.model_id,
                "timestamp": entry_time.isoformat() + "+00:00",
                "entry_time": entry_time.isoformat() + "+00:00",
                "exit_time": exit_time.isoformat() + "+00:00",
                "side": side,
                "entry_price": round(entry_price, 2),
                "exit_price": round(exit_price, 2),
                "pnl": round(pnl_usd, 2),
                "pnl_usd": round(pnl_usd, 2),
                "pnl_percent": round(pnl_pct_actual * 100, 4),
                "pnl_pct": round(pnl_pct_actual * 100, 4),
                "status": "closed",
                "duration_minutes": duration,
                "exit_reason": exit_reason,
                "equity_at_entry": round(equity - pnl_usd, 2),
                "equity_at_exit": round(equity, 2),
                "entry_confidence": round(0.6 + random.random() * 0.35, 3),
                "exit_confidence": round(0.5 + random.random() * 0.4, 3),
            }

            trades.append(trade)
            trade_id += 1

        # Sort by timestamp
        trades.sort(key=lambda t: t["timestamp"])

        # Re-number trade IDs
        for i, trade in enumerate(trades):
            trade["trade_id"] = i + 1

        return trades

    def _calculate_metrics(self, trades: List[Dict], initial_capital: float) -> Dict:
        """Calculate summary metrics from trades."""
        if not trades:
            return self._empty_metrics()

        pnls = [t["pnl"] for t in trades]
        pnl_pcts = [t["pnl_percent"] for t in trades]

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        total_pnl = sum(pnls)
        total_return = total_pnl / initial_capital

        # Win rate
        win_rate = len(wins) / len(trades) if trades else 0

        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Max drawdown calculation
        equity_curve = [initial_capital]
        for pnl in pnls:
            equity_curve.append(equity_curve[-1] + pnl)

        peak = equity_curve[0]
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        if len(pnl_pcts) > 1:
            returns_std = np.std(pnl_pcts)
            avg_return = np.mean(pnl_pcts)
            # Annualized (assuming ~250 trading days)
            sharpe = (avg_return * np.sqrt(250)) / returns_std if returns_std > 0 else 0
        else:
            sharpe = 0

        # Adjust sharpe to be more realistic
        sharpe = min(max(sharpe, 0), 3.0)  # Cap at 3.0

        return {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return * 100, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "largest_win": round(max(wins), 2) if wins else 0,
            "largest_loss": round(min(losses), 2) if losses else 0,
            "avg_trade_duration_minutes": round(np.mean([t["duration_minutes"] for t in trades]), 1),
            "final_equity": round(initial_capital + total_pnl, 2),
        }

    def _build_equity_curve(
        self,
        trades: List[Dict],
        initial_capital: float,
        start_dt: datetime,
        end_dt: datetime
    ) -> List[Dict]:
        """Build equity curve for charting."""
        curve = [{
            "timestamp": start_dt.isoformat() + "+00:00",
            "equity": initial_capital,
            "drawdown": 0,
        }]

        equity = initial_capital
        peak = initial_capital

        for trade in trades:
            equity += trade["pnl"]
            if equity > peak:
                peak = equity
            dd = (equity - peak) / peak if peak > 0 else 0

            curve.append({
                "timestamp": trade["exit_time"],
                "equity": round(equity, 2),
                "drawdown": round(dd * 100, 2),
                "trade_id": trade["trade_id"],
            })

        return curve

    def _empty_result(self, start_date: str, end_date: str) -> Dict:
        """Return empty result structure."""
        return {
            "success": True,
            "source": "demo_investor_mode",
            "model_id": self.config.model_id,
            "start_date": start_date,
            "end_date": end_date,
            "trade_count": 0,
            "trades": [],
            "summary": self._empty_metrics(),
            "equity_curve": [],
        }

    def _empty_metrics(self) -> Dict:
        """Return empty metrics structure."""
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "total_return_pct": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
            "avg_trade_duration_minutes": 0,
        }
