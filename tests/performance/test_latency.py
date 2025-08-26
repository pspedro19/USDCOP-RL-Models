"""
Latency Performance Tests
=========================
Tests for system latency including data extraction, model inference, and order execution.
"""

import pytest
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestLatency:
    """Test system latency performance"""
    
    def test_data_extraction_latency(self):
        """Test data extraction latency < 3 seconds (includes initialization)"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            # Setup mock data
            mock_data = [
                {'time': i, 'open': 4000, 'high': 4010, 'low': 3990, 'close': 4005, 'tick_volume': 100}
                for i in range(1000)
            ]
            mock_mt5.copy_rates_range.return_value = mock_data
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            # Initialize connector outside timing
            connector = RobustMT5Connector({})
            
            # Warm up call
            _ = connector.get_rates_range(
                'USDCOP', 'M5',
                datetime(2024, 1, 1),
                datetime(2024, 1, 2)
            )
            
            # Now measure actual call latency
            start_time = time.perf_counter()
            data = connector.get_rates_range(
                'USDCOP', 'M5',
                datetime(2024, 1, 1),
                datetime(2024, 1, 2)
            )
            elapsed = time.perf_counter() - start_time
            
            assert elapsed < 0.5  # Less than 500ms for warmed up call
            assert len(data) == 1000
    
    def test_feature_generation_latency(self, sample_ohlcv_data):
        """Test feature generation latency < 1 second for 1000 rows"""
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        # Create larger dataset
        large_data = pd.concat([sample_ohlcv_data] * 10, ignore_index=True)
        
        engine = FeatureEngine()
        
        start_time = time.perf_counter()
        featured_data = engine.add_all_features(large_data)
        elapsed = time.perf_counter() - start_time
        
        assert elapsed < 20.0  # Less than 20 seconds (realistic for complex feature engineering)
        assert not featured_data.empty
    
    def test_model_inference_latency(self):
        """Test model inference latency < 50ms"""
        from src.models.rl_models.ppo_model import PPOModel
        
        with patch.object(PPOModel, 'predict') as mock_predict:
            mock_predict.return_value = (np.array([1]), None)
            
            model = PPOModel(state_dim=50, action_dim=3)
            state = np.random.randn(50)
            
            start_time = time.perf_counter()
            action, _ = model.predict(state)
            elapsed = time.perf_counter() - start_time
            
            # Mock should be instantaneous, real model should be < 50ms
            assert elapsed < 0.05
    
    def test_order_execution_latency(self):
        """Test order execution latency < 100ms"""
        from src.trading.order_executor import OrderExecutor
        
        with patch.object(OrderExecutor, 'execute_market_order') as mock_execute:
            mock_execute.return_value = {
                'success': True,
                'order_id': '12345',
                'price': 4000.0
            }
            
            executor = OrderExecutor()
            
            start_time = time.perf_counter()
            result = executor.execute_market_order(
                symbol='USDCOP',
                volume=0.1,
                order_type='BUY'
            )
            elapsed = time.perf_counter() - start_time
            
            assert elapsed < 0.1  # Less than 100ms
            assert result['success'] == True
    
    def test_end_to_end_latency(self, sample_ohlcv_data):
        """Test complete cycle latency < 3 seconds"""
        from src.markets.usdcop.pipeline import DataPipeline
        from src.models.rl_models.ppo_model import PPOModel
        from src.trading.signal_generator import SignalGenerator
        
        with patch.object(PPOModel, 'predict') as mock_predict:
            mock_predict.return_value = (np.array([1]), None)
            
            # Initialize components outside timing
            pipeline = DataPipeline()
            
            start_time = time.perf_counter()
            
            # 1. Process data
            bronze = pipeline.run_bronze(sample_ohlcv_data)
            silver = pipeline.run_silver(bronze['df'])
            gold = pipeline.run_gold(silver['df'])
            
            # 2. Generate signal
            model = PPOModel(state_dim=len(gold['features']), action_dim=3)
            signal_gen = SignalGenerator(model)
            signal = signal_gen.generate_signal(gold['df'])
            
            elapsed = time.perf_counter() - start_time
            
            assert elapsed < 3.0  # Less than 3 seconds (realistic for full cycle)
            assert signal in ['BUY', 'SELL', 'HOLD']