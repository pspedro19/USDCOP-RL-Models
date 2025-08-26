"""
Data Extraction Unit Tests
==========================
Tests for data extraction from various sources including MT5, CCXT, and simulation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestDataExtraction:
    """Test data extraction functionality"""
    
    def test_mt5_usdcop_data_download_success(self, sample_ohlcv_data):
        """Test successful USDCOP data download from MT5"""
        with patch('src.core.connectors.mt5_connector.RobustMT5Connector') as MockConnector:
            mock_connector = MockConnector.return_value
            mock_connector.get_rates_range.return_value = sample_ohlcv_data
            
            from src.markets.usdcop.pipeline import USDCOPPipeline, PipelineConfig
            
            config = PipelineConfig(
                symbol='USDCOP',
                primary_timeframe='M5',
                bars=100
            )
            
            pipeline = USDCOPPipeline(config, connector=mock_connector)
            result = pipeline._fetch_with_fallback(
                'USDCOP', 'M5',
                datetime(2024, 1, 1),
                datetime(2024, 1, 2)
            )
            
            assert not result.empty
            assert len(result) == len(sample_ohlcv_data)
    
    def test_mt5_historical_data_extraction(self):
        """Test historical data extraction from MT5"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.copy_rates_range.return_value = [
                {'time': 1704067200, 'open': 4000, 'high': 4010, 'low': 3990, 'close': 4005, 'tick_volume': 100}
            ]
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector({})
            data = connector.get_historical_data('USDCOP', 'M5', datetime(2024, 1, 1), datetime(2024, 1, 2))
            
            assert data is not None
    
    def test_mt5_realtime_tick_data_extraction(self):
        """Test real-time tick data extraction"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.symbol_info_tick.return_value = Mock(
                time=1704067200,
                bid=4000.5,
                ask=4001.0,
                last=4000.75,
                volume=1000
            )
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector({})
            tick = connector.get_tick('USDCOP')
            
            assert tick is not None
            assert tick['bid'] == 4000.5
            assert tick['ask'] == 4001.0
    
    def test_mt5_data_extraction_date_range_validation(self):
        """Test date range validation in data extraction"""
        from src.markets.usdcop.pipeline import PipelineConfig
        
        # Invalid date range (end before start)
        with pytest.raises(ValueError):
            config = PipelineConfig(
                symbol='USDCOP',
                start_date='2024-01-02',
                end_date='2024-01-01'
            )
            config.validate_dates()
    
    def test_mt5_data_extraction_symbol_validation(self):
        """Test symbol validation"""
        with patch('src.core.connectors.mt5_connector.mt5') as mock_mt5:
            mock_mt5.symbol_info.return_value = None  # Symbol not found
            
            from src.core.connectors.mt5_connector import RobustMT5Connector
            
            connector = RobustMT5Connector({})
            valid = connector.validate_symbol('INVALID')
            
            assert valid == False
    
    def test_mt5_data_extraction_timeframe_conversion(self):
        """Test timeframe conversion"""
        from src.utils.timeframes import convert_timeframe
        
        # Test various conversions
        assert convert_timeframe('M5', 'mt5') == 5
        assert convert_timeframe('H1', 'mt5') == 16385
        assert convert_timeframe('D1', 'mt5') == 16408
    
    def test_ccxt_fallback_data_extraction(self):
        """Test CCXT fallback when MT5 fails"""
        with patch('src.core.connectors.fallback_manager.ccxt') as mock_ccxt:
            mock_exchange = Mock()
            mock_exchange.fetch_ohlcv.return_value = [
                [1704067200000, 4000, 4010, 3990, 4005, 100]
            ]
            mock_ccxt.binance.return_value = mock_exchange
            
            from src.core.connectors.fallback_manager import FallbackManager
            
            manager = FallbackManager({})
            data = manager.fetch_with_fallback('USDCOP', 'M5', datetime(2024, 1, 1), datetime(2024, 1, 2))
            
            assert data is not None
    
    def test_simulated_data_generation(self):
        """Test simulated data generation as last fallback"""
        from src.core.data.data_simulator import DataSimulator
        
        simulator = DataSimulator(initial_price=4000, volatility=0.1)
        data = simulator.generate_data(
            symbol='USDCOP',
            timeframe='M5',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 1, 2)
        )
        
        assert not data.empty
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns