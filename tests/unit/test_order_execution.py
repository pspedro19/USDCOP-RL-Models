"""
Order Execution Unit Tests
==========================
Tests for market orders, limit orders, and order management.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock


class TestOrderExecution:
    """Test order execution functionality"""
    
    def test_market_order_execution(self):
        """Test market order execution"""
        from src.trading.order_executor import OrderExecutor
        
        # Test in simulation mode (default)
        executor = OrderExecutor(simulation_mode=True)
        result = executor.execute_market_order(
            symbol='USDCOP.r',
            volume=0.1,
            order_type='BUY'
        )
        
        assert result['success'] == True
        assert 'order_id' in result
        assert result['executed_volume'] == 0.1
        assert 'executed_price' in result
        assert 'slippage' in result
    
    def test_limit_order_execution(self):
        """Test limit order execution"""
        from src.trading.order_executor import OrderExecutor
        
        # Test in simulation mode (default)
        executor = OrderExecutor(simulation_mode=True)
        result = executor.execute_limit_order(
            symbol='USDCOP.r',
            volume=0.1,
            order_type='BUY',
            price=3995
        )
        
        assert result['success'] == True
        assert 'order_id' in result
        assert result['status'] == 'PENDING'
    
    def test_order_modification(self):
        """Test order modification (price/volume)"""
        from src.trading.order_executor import OrderExecutor
        
        # Test in simulation mode (default)
        executor = OrderExecutor(simulation_mode=True)
        
        # First create an order to modify
        order_result = executor.execute_limit_order(
            symbol='USDCOP.r',
            volume=0.1,
            order_type='BUY',
            price=4000
        )
        order_id = order_result['order_id']
        
        # Now modify it
        result = executor.modify_order(
            order_id=order_id,
            new_price=4005,
            new_sl=3990,
            new_tp=4020
        )
        
        assert result['success'] == True
    
    def test_order_cancellation(self):
        """Test order cancellation"""
        from src.trading.order_executor import OrderExecutor
        
        # Test in simulation mode (default)
        executor = OrderExecutor(simulation_mode=True)
        
        # First create an order to cancel
        order_result = executor.execute_limit_order(
            symbol='USDCOP.r',
            volume=0.1,
            order_type='BUY',
            price=4000
        )
        order_id = order_result['order_id']
        
        # Now cancel it
        result = executor.cancel_order(order_id=order_id)
        
        assert result['success'] == True
    
    def test_partial_fill_handling(self):
        """Test handling of partial order fills"""
        from src.trading.order_executor import OrderExecutor
        
        # Test in simulation mode (default)
        executor = OrderExecutor(simulation_mode=True)
        
        # Create an order first
        order_result = executor.execute_limit_order(
            symbol='USDCOP.r',
            volume=1.0,
            order_type='BUY',
            price=4000
        )
        order_id = order_result['order_id']
        
        # Check order status
        status = executor.check_order_status(order_id=order_id)
        
        assert 'state' in status
        assert 'filled_volume' in status or 'executed_volume' in status
        assert status['state'] in ['FILLED', 'PENDING', 'PARTIAL']
    
    def test_slippage_tracking(self):
        """Test slippage tracking for executed orders"""
        from src.trading.order_executor import SlippageTracker
        
        tracker = SlippageTracker()
        
        # Track order with slippage
        tracker.track_execution(
            expected_price=4000,
            executed_price=4002,
            volume=0.1,
            order_type='BUY'
        )
        
        stats = tracker.get_statistics()
        
        assert stats['average_slippage'] == 2  # 4002 - 4000
        assert stats['total_slippage_cost'] == 0.2  # 2 * 0.1