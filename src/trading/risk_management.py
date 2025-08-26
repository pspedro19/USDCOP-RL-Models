"""
Risk Management Module
======================
Comprehensive risk management tools including position sizing, stop-loss, drawdown control, and VaR.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration"""
    max_risk_per_trade: float = 0.02  # 2%
    max_daily_loss: float = 0.05      # 5%
    max_drawdown: float = 0.10        # 10%
    max_positions: int = 3
    max_correlation: float = 0.7
    var_confidence: float = 0.95      # 95% VaR


class RiskManager:
    """
    Comprehensive risk management system.
    
    Handles position sizing, stop-loss calculation, take-profit levels,
    and overall portfolio risk monitoring.
    """
    
    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize risk manager.
        
        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self.positions = {}
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.equity_curve = []
        
    def calculate_stop_loss(
        self,
        entry_price: float,
        position_type: str,
        atr: float,
        multiplier: float = 2.0
    ) -> float:
        """
        Calculate stop-loss price based on ATR.
        
        Args:
            entry_price: Entry price for the position
            position_type: 'LONG' or 'SHORT'
            atr: Average True Range value
            multiplier: ATR multiplier for stop distance
            
        Returns:
            Stop-loss price
        """
        stop_distance = atr * multiplier
        
        if position_type.upper() == 'LONG':
            return entry_price - stop_distance
        elif position_type.upper() == 'SHORT':
            return entry_price + stop_distance
        else:
            raise ValueError(f"Invalid position type: {position_type}")
    
    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        risk_reward_ratio: float = 2.0
    ) -> float:
        """
        Calculate take-profit price based on risk-reward ratio.
        
        Args:
            entry_price: Entry price for the position
            stop_loss: Stop-loss price
            risk_reward_ratio: Target risk-reward ratio
            
        Returns:
            Take-profit price
        """
        risk = abs(entry_price - stop_loss)
        reward = risk * risk_reward_ratio
        
        if stop_loss < entry_price:  # Long position
            return entry_price + reward
        else:  # Short position
            return entry_price - reward
    
    def validate_position(
        self,
        symbol: str,
        volume: float,
        entry_price: float,
        stop_loss: float,
        account_balance: float
    ) -> Tuple[bool, str]:
        """
        Validate if position meets risk requirements.
        
        Args:
            symbol: Trading symbol
            volume: Position size
            entry_price: Entry price
            stop_loss: Stop-loss price
            account_balance: Current account balance
            
        Returns:
            Tuple of (is_valid, reason)
        """
        # Calculate position risk
        risk_per_share = abs(entry_price - stop_loss)
        total_risk = risk_per_share * volume
        risk_percentage = total_risk / account_balance
        
        # Check risk per trade limit
        if risk_percentage > self.limits.max_risk_per_trade:
            return False, f"Risk {risk_percentage:.3f} exceeds limit {self.limits.max_risk_per_trade}"
        
        # Check maximum positions
        if len(self.positions) >= self.limits.max_positions:
            return False, f"Maximum positions {self.limits.max_positions} reached"
        
        # Check daily loss limit
        if self.daily_pnl < -self.limits.max_daily_loss * account_balance:
            return False, f"Daily loss limit {self.limits.max_daily_loss} exceeded"
        
        return True, "Position approved"
    
    def update_equity(self, current_equity: float):
        """Update equity curve and check drawdown"""
        self.equity_curve.append(current_equity)
        
        # Update peak equity
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        # Keep only last 10000 values
        if len(self.equity_curve) > 10000:
            self.equity_curve = self.equity_curve[-10000:]
    
    def get_current_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if not self.equity_curve or self.peak_equity == 0:
            return 0.0
            
        current_equity = self.equity_curve[-1]
        return (self.peak_equity - current_equity) / self.peak_equity
    
    def check_drawdown_limit(self) -> bool:
        """Check if current drawdown exceeds limit"""
        current_drawdown = self.get_current_drawdown()
        return current_drawdown <= self.limits.max_drawdown


class PositionSizer:
    """
    Position sizing using various methods.
    """
    
    def __init__(self, method: str = "fixed_percent"):
        """
        Initialize position sizer.
        
        Args:
            method: Sizing method ('fixed_percent', 'kelly', 'var')
        """
        self.method = method
        
    def calculate_position_size(
        self,
        account_balance: float,
        risk_per_trade: float = 0.02,
        entry_price: float = None,
        stop_loss: float = None,
        win_rate: float = None,
        avg_win_loss_ratio: float = None,
        max_risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size based on selected method.
        
        Args:
            account_balance: Current account balance
            risk_per_trade: Risk percentage per trade
            entry_price: Entry price
            stop_loss: Stop-loss price
            win_rate: Historical win rate (for Kelly)
            avg_win_loss_ratio: Average win/loss ratio (for Kelly)
            max_risk_per_trade: Maximum risk cap
            
        Returns:
            Position size (volume)
        """
        if self.method == "fixed_percent":
            return self._fixed_percent_sizing(account_balance, risk_per_trade)
            
        elif self.method == "kelly":
            if win_rate is None or avg_win_loss_ratio is None:
                logger.warning("Kelly criterion requires win_rate and avg_win_loss_ratio")
                return self._fixed_percent_sizing(account_balance, risk_per_trade)
            return self._kelly_sizing(
                account_balance, win_rate, avg_win_loss_ratio, max_risk_per_trade
            )
            
        else:
            logger.warning(f"Unknown sizing method: {self.method}")
            return self._fixed_percent_sizing(account_balance, risk_per_trade)
    
    def _fixed_percent_sizing(self, balance: float, risk_percent: float) -> float:
        """Fixed percentage risk sizing"""
        return balance * risk_percent
    
    def _kelly_sizing(
        self,
        balance: float,
        win_rate: float,
        avg_win_loss_ratio: float,
        max_risk: float
    ) -> float:
        """Kelly Criterion sizing with cap"""
        # Kelly formula: f = (p*b - q)/b
        # where p = win_rate, q = 1-p, b = avg_win_loss_ratio
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio
        
        if b <= 0:
            return balance * max_risk
            
        kelly_fraction = (p * b - q) / b
        
        # Cap Kelly fraction to max risk
        kelly_fraction = min(kelly_fraction, max_risk)
        kelly_fraction = max(kelly_fraction, 0)  # No negative sizing
        
        return balance * kelly_fraction


class DrawdownMonitor:
    """
    Monitor and control maximum drawdown.
    """
    
    def __init__(self, max_drawdown: float = 0.10):
        """
        Initialize drawdown monitor.
        
        Args:
            max_drawdown: Maximum allowed drawdown percentage
        """
        self.max_drawdown = max_drawdown
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.drawdown_history = deque(maxlen=1000)
        
    def update(self, current_value: float) -> bool:
        """
        Update with current portfolio value.
        
        Args:
            current_value: Current portfolio value
            
        Returns:
            True if should stop trading due to excessive drawdown
        """
        # Update peak
        if current_value > self.peak_value:
            self.peak_value = current_value
            
        # Calculate current drawdown
        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0
            
        # Store in history
        self.drawdown_history.append(self.current_drawdown)
        
        # Check if exceeds limit
        return self.current_drawdown > self.max_drawdown
    
    def get_max_historical_drawdown(self) -> float:
        """Get maximum historical drawdown"""
        return max(self.drawdown_history) if self.drawdown_history else 0.0


class DailyRiskLimiter:
    """
    Monitor and enforce daily risk limits.
    """
    
    def __init__(self, daily_loss_limit: float, initial_balance: float):
        """
        Initialize daily risk limiter.
        
        Args:
            daily_loss_limit: Maximum daily loss (absolute value)
            initial_balance: Starting balance for the day
        """
        self.daily_loss_limit = daily_loss_limit
        self.initial_balance = initial_balance
        self.daily_pnl = 0.0
        self.trades_today = []
        self.current_date = datetime.now().date()
        
    def add_trade_result(self, pnl: float):
        """Add trade result to daily tracking"""
        current_date = datetime.now().date()
        
        # Reset if new day
        if current_date != self.current_date:
            self.daily_pnl = 0.0
            self.trades_today.clear()
            self.current_date = current_date
            
        self.daily_pnl += pnl
        self.trades_today.append(pnl)
        
    def can_trade(self) -> bool:
        """Check if trading is allowed based on daily limits"""
        return abs(self.daily_pnl) < self.daily_loss_limit
    
    def get_remaining_risk(self) -> float:
        """Get remaining risk capacity for the day"""
        return max(0, self.daily_loss_limit - abs(self.daily_pnl))


class CorrelationRiskChecker:
    """
    Check correlation risk between positions.
    """
    
    def __init__(self, max_correlation: float = 0.7):
        """
        Initialize correlation checker.
        
        Args:
            max_correlation: Maximum allowed correlation between positions
        """
        self.max_correlation = max_correlation
        self.correlation_matrix = {}
        
    def can_add_position(
        self,
        new_symbol: str,
        new_direction: str,
        existing_positions: List[Dict]
    ) -> bool:
        """
        Check if new position can be added based on correlation limits.
        
        Args:
            new_symbol: Symbol of new position
            new_direction: Direction ('LONG' or 'SHORT')
            existing_positions: List of existing positions
            
        Returns:
            True if position can be added
        """
        for position in existing_positions:
            existing_symbol = position['symbol']
            existing_direction = position['direction']
            
            # Get correlation
            correlation = self.get_correlation(new_symbol, existing_symbol)
            
            # Adjust correlation based on direction
            if new_direction != existing_direction:
                correlation = -correlation
                
            if abs(correlation) > self.max_correlation:
                logger.warning(
                    f"High correlation {correlation:.3f} between {new_symbol} and {existing_symbol}"
                )
                return False
                
        return True
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two symbols.
        
        This is a placeholder implementation.
        In practice, you would calculate this from historical data.
        """
        # Mock correlation data
        correlations = {
            ('USDCOP', 'USDMXN'): 0.85,
            ('USDCOP', 'USDBRL'): 0.75,
            ('EURUSD', 'GBPUSD'): 0.65,
            ('EURUSD', 'USDCHF'): -0.70,
        }
        
        key1 = (symbol1, symbol2)
        key2 = (symbol2, symbol1)
        
        return correlations.get(key1, correlations.get(key2, 0.0))


class VaRCalculator:
    """
    Value at Risk (VaR) calculator.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize VaR calculator.
        
        Args:
            confidence_level: Confidence level for VaR calculation
        """
        self.confidence_level = confidence_level
        
    def calculate_var(
        self,
        returns: np.ndarray,
        portfolio_value: float,
        method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk.
        
        Args:
            returns: Historical returns array
            portfolio_value: Current portfolio value
            method: Calculation method ('historical', 'parametric')
            
        Returns:
            VaR value (positive number representing potential loss)
        """
        if len(returns) == 0:
            return 0.0
            
        if method == "historical":
            return self._historical_var(returns, portfolio_value)
        elif method == "parametric":
            return self._parametric_var(returns, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _historical_var(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Calculate historical VaR"""
        percentile = (1 - self.confidence_level) * 100
        var_return = np.percentile(returns, percentile)
        return abs(var_return * portfolio_value)
    
    def _parametric_var(self, returns: np.ndarray, portfolio_value: float) -> float:
        """Calculate parametric VaR assuming normal distribution"""
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Z-score for confidence level
        z_scores = {0.90: 1.28, 0.95: 1.65, 0.99: 2.33}
        z_score = z_scores.get(self.confidence_level, 1.65)
        
        var_return = mean_return - z_score * std_return
        return abs(var_return * portfolio_value)


class EmergencyStop:
    """
    Emergency stop mechanism for critical situations.
    """
    
    def __init__(self, max_loss: float, executor=None):
        """
        Initialize emergency stop.
        
        Args:
            max_loss: Maximum loss before emergency stop
            executor: Order executor for closing positions
        """
        self.max_loss = max_loss
        self.executor = executor
        self.triggered = False
        
    def check_and_trigger(self, current_loss: float) -> bool:
        """
        Check if emergency stop should be triggered.
        
        Args:
            current_loss: Current portfolio loss
            
        Returns:
            True if emergency stop was triggered
        """
        if current_loss >= self.max_loss and not self.triggered:
            self.triggered = True
            logger.critical(f"EMERGENCY STOP TRIGGERED: Loss {current_loss} exceeds limit {self.max_loss}")
            
            if self.executor:
                try:
                    result = self.executor.close_all_positions()
                    logger.info(f"Emergency close result: {result}")
                except Exception as e:
                    logger.error(f"Failed to execute emergency stop: {e}")
                    
            return True
            
        return False
    
    def reset(self):
        """Reset emergency stop state"""
        self.triggered = False