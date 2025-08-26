"""
Live Trading E2E Tests
======================
End-to-end tests for live trading scenarios.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio


class TestLiveTrading:
    """Test live trading scenarios"""
    
    @pytest.mark.asyncio
    async def test_live_trading_session(self):
        """Test complete live trading session"""
        from src.trading.live_trader import LiveTrader
        from src.core.connectors.mt5_connector import RobustMT5Connector
        
        with patch.object(RobustMT5Connector, 'get_tick') as mock_tick, \
             patch('src.trading.order_executor.OrderExecutor.execute_market_order') as mock_order:
            
            # Mock tick data stream
            mock_tick.side_effect = [
                {'time': datetime.now(), 'bid': 4000, 'ask': 4001},
                {'time': datetime.now(), 'bid': 4002, 'ask': 4003},
                {'time': datetime.now(), 'bid': 4005, 'ask': 4006},
            ]
            
            # Mock order execution
            mock_order.return_value = {'success': True, 'order_id': '12345'}
            
            trader = LiveTrader(
                symbol='USDCOP',
                model_path='models/best_model.pkl',
                risk_per_trade=0.01
            )
            
            # Run for 3 ticks
            results = []
            for _ in range(3):
                result = await trader.process_tick()
                results.append(result)
            
            assert len(results) == 3
            assert any(r.get('action') != 'HOLD' for r in results)
    
    def test_connection_recovery(self):
        """Test recovery from connection loss"""
        from src.trading.live_trader import LiveTrader
        from src.core.connectors.mt5_connector import ConnectionError
        
        with patch('src.core.connectors.mt5_connector.RobustMT5Connector') as MockConnector:
            mock_connector = MockConnector.return_value
            
            # Simulate connection loss and recovery
            mock_connector.get_tick.side_effect = [
                ConnectionError("Connection lost"),
                ConnectionError("Still disconnected"),
                {'time': datetime.now(), 'bid': 4000, 'ask': 4001}  # Recovered
            ]
            
            trader = LiveTrader(
                symbol='USDCOP',
                connector=mock_connector,
                max_reconnect_attempts=3
            )
            
            # Should recover after 2 failures
            tick = trader.get_tick_with_retry()
            assert tick is not None
            assert tick['bid'] == 4000
    
    def test_emergency_stop(self):
        """Test emergency stop mechanism"""
        from src.trading.live_trader import LiveTrader
        from src.trading.risk_management import EmergencyStop
        
        with patch('src.trading.order_executor.OrderExecutor') as MockExecutor:
            mock_executor = MockExecutor.return_value
            mock_executor.close_all_positions.return_value = {'closed': 3}
            
            emergency_stop = EmergencyStop(
                max_loss=1000,
                executor=mock_executor
            )
            
            # Trigger emergency stop
            emergency_stop.check_and_trigger(current_loss=1100)
            
            # Should close all positions
            mock_executor.close_all_positions.assert_called_once()
    
    def test_multi_timeframe_analysis(self):
        """Test multi-timeframe analysis in live trading"""
        from src.trading.live_trader import MultiTimeframeTrader
        
        with patch('src.core.connectors.mt5_connector.RobustMT5Connector') as MockConnector:
            mock_connector = MockConnector.return_value
            
            # Mock data for different timeframes
            mock_connector.get_rates_range.side_effect = [
                pd.DataFrame({'close': np.random.randn(100) + 4000}),  # M5
                pd.DataFrame({'close': np.random.randn(100) + 4000}),  # M15
                pd.DataFrame({'close': np.random.randn(100) + 4000}),  # H1
            ]
            
            trader = MultiTimeframeTrader(
                symbol='USDCOP',
                timeframes=['M5', 'M15', 'H1'],
                connector=mock_connector
            )
            
            analysis = trader.analyze()
            
            assert 'M5' in analysis
            assert 'M15' in analysis
            assert 'H1' in analysis
            assert analysis['consensus'] in ['BUY', 'SELL', 'HOLD']
    
    def test_news_event_handling(self):
        """Test handling of news events during live trading"""
        from src.trading.live_trader import NewsAwareTrader
        from src.data.news_monitor import NewsMonitor
        
        with patch.object(NewsMonitor, 'get_upcoming_events') as mock_news:
            # Mock high-impact news event
            mock_news.return_value = [
                {
                    'time': datetime.now() + timedelta(minutes=5),
                    'impact': 'HIGH',
                    'currency': 'USD',
                    'event': 'NFP Release'
                }
            ]
            
            trader = NewsAwareTrader(
                symbol='USDCOP',
                news_monitor=NewsMonitor()
            )
            
            # Should pause trading before high-impact news
            should_trade = trader.check_trading_conditions()
            assert should_trade == False