"""
Risk Management Unit Tests
==========================
Tests for position sizing, stop-loss, take-profit, and drawdown control.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch


class TestRiskManagement:
    """Test risk management functionality"""
    
    def test_position_sizing_kelly_criterion(self):
        """Test position sizing using Kelly Criterion"""
        from src.trading.risk_management import PositionSizer
        
        sizer = PositionSizer(method='kelly')
        
        # Win rate = 60%, avg win/loss ratio = 1.5
        position_size = sizer.calculate_position_size(
            account_balance=10000,
            win_rate=0.6,
            avg_win_loss_ratio=1.5,
            max_risk_per_trade=0.02
        )
        
        # Kelly formula: f = (p*b - q)/b where p=win_rate, q=1-p, b=win/loss ratio
        # f = (0.6*1.5 - 0.4)/1.5 = 0.5/1.5 = 0.333
        # But should be capped by max_risk_per_trade (200 = 10000 * 0.02)
        assert position_size <= 200  # Max risk per trade in dollars
        assert position_size > 0
    
    def test_stop_loss_calculation(self):
        """Test stop-loss price calculation"""
        from src.trading.risk_management import RiskManager
        
        risk_manager = RiskManager()
        
        # Test for long position
        stop_loss = risk_manager.calculate_stop_loss(
            entry_price=4000,
            position_type='LONG',
            atr=20,
            multiplier=2
        )
        
        assert stop_loss == 3960  # 4000 - (20 * 2)
        
        # Test for short position
        stop_loss = risk_manager.calculate_stop_loss(
            entry_price=4000,
            position_type='SHORT',
            atr=20,
            multiplier=2
        )
        
        assert stop_loss == 4040  # 4000 + (20 * 2)
    
    def test_take_profit_calculation(self):
        """Test take-profit price calculation"""
        from src.trading.risk_management import RiskManager
        
        risk_manager = RiskManager()
        
        # Test for long position with 2:1 risk-reward ratio
        take_profit = risk_manager.calculate_take_profit(
            entry_price=4000,
            stop_loss=3960,
            risk_reward_ratio=2
        )
        
        assert take_profit == 4080  # 4000 + (40 * 2)
        
        # Test for short position
        take_profit = risk_manager.calculate_take_profit(
            entry_price=4000,
            stop_loss=4040,
            risk_reward_ratio=2
        )
        
        assert take_profit == 3920  # 4000 - (40 * 2)
    
    def test_max_drawdown_check(self):
        """Test maximum drawdown monitoring"""
        from src.trading.risk_management import DrawdownMonitor
        
        monitor = DrawdownMonitor(max_drawdown=0.1)  # 10% max drawdown
        
        # Simulate equity curve
        equity_curve = [10000, 10500, 10200, 9500, 9800, 10100]
        
        for equity in equity_curve:
            should_stop = monitor.update(equity)
            
            if equity == 9500:  # Drawdown of 9.5% from peak
                assert should_stop == False
            
        # Check final drawdown
        assert monitor.current_drawdown < 0.1
    
    def test_daily_loss_limit(self):
        """Test daily loss limit enforcement"""
        from src.trading.risk_management import DailyRiskLimiter
        
        limiter = DailyRiskLimiter(
            daily_loss_limit=500,
            initial_balance=10000
        )
        
        # Add losing trades
        limiter.add_trade_result(-200)
        assert limiter.can_trade() == True
        
        limiter.add_trade_result(-250)
        assert limiter.can_trade() == True
        
        limiter.add_trade_result(-100)
        assert limiter.can_trade() == False  # Exceeded daily limit
    
    def test_correlation_risk_check(self):
        """Test correlation risk between positions"""
        from src.trading.risk_management import CorrelationRiskChecker
        
        checker = CorrelationRiskChecker(max_correlation=0.7)
        
        # Create sample positions
        positions = [
            {'symbol': 'USDCOP', 'direction': 'LONG'},
            {'symbol': 'USDMXN', 'direction': 'LONG'}
        ]
        
        # Mock correlation matrix
        with patch.object(checker, 'get_correlation') as mock_corr:
            mock_corr.return_value = 0.85  # High correlation
            
            can_add = checker.can_add_position(
                new_symbol='USDBRL',
                new_direction='LONG',
                existing_positions=positions
            )
            
            assert can_add == False  # Too correlated
    
    def test_var_calculation(self):
        """Test Value at Risk (VaR) calculation"""
        from src.trading.risk_management import VaRCalculator
        
        calculator = VaRCalculator(confidence_level=0.95)
        
        # Historical returns
        returns = np.random.normal(0, 0.01, 1000)  # 1% daily volatility
        
        var_95 = calculator.calculate_var(
            returns=returns,
            portfolio_value=10000
        )
        
        # VaR should be around 1.65 * std * portfolio_value
        expected_var = 1.65 * 0.01 * 10000
        assert abs(var_95 - expected_var) < 50  # Within $50 tolerance