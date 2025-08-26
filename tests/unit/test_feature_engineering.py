"""
Feature Engineering Unit Tests
===============================
Tests for technical indicators, market microstructure, and custom features.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_rsi_calculation(self):
        """Test RSI (Relative Strength Index) calculation"""
        # Create simple price data
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113]
        data = pd.DataFrame({'close': prices})
        
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_rsi(data, period=14)
        
        assert 'rsi_14' in result.columns
        # RSI should be between 0 and 100
        assert result['rsi_14'].dropna().min() >= 0
        assert result['rsi_14'].dropna().max() <= 100
    
    def test_macd_indicator(self):
        """Test MACD indicator calculation"""
        # Generate price data
        np.random.seed(42)
        prices = 4000 + np.cumsum(np.random.randn(100) * 10)
        data = pd.DataFrame({'close': prices})
        
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_macd(data)
        
        assert 'macd' in result.columns
        assert 'macd_signal' in result.columns
        assert 'macd_diff' in result.columns
        
        # MACD should have fewer NaN values at the end
        assert result['macd'].notna().sum() > 70
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation"""
        # Generate price data with trend
        prices = 4000 + np.arange(100) * 2 + np.random.randn(100) * 5
        data = pd.DataFrame({'close': prices})
        
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_bollinger_bands(data, period=20)
        
        assert 'bb_upper' in result.columns
        assert 'bb_middle' in result.columns
        assert 'bb_lower' in result.columns
        assert 'bb_width' in result.columns
        
        # Upper band should be above lower band
        valid_rows = result[['bb_upper', 'bb_lower']].dropna()
        assert (valid_rows['bb_upper'] > valid_rows['bb_lower']).all()
    
    def test_volume_indicators(self, sample_ohlcv_data):
        """Test volume-based indicators"""
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_volume_indicators(sample_ohlcv_data)
        
        assert 'volume_sma' in result.columns
        assert 'volume_ratio' in result.columns
        assert 'obv' in result.columns  # On-Balance Volume
        
        # Volume ratio should be positive
        assert (result['volume_ratio'].dropna() > 0).all()
    
    def test_price_patterns(self, sample_ohlcv_data):
        """Test price pattern detection"""
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_price_patterns(sample_ohlcv_data)
        
        # Check for candlestick patterns
        assert 'is_doji' in result.columns
        assert 'is_hammer' in result.columns
        assert 'is_engulfing' in result.columns
        
        # Pattern columns should be boolean
        assert result['is_doji'].dtype == bool
    
    def test_market_microstructure_features(self, sample_ohlcv_data):
        """Test market microstructure features"""
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_microstructure_features(sample_ohlcv_data)
        
        assert 'bid_ask_spread' in result.columns
        assert 'log_returns' in result.columns
        assert 'realized_volatility' in result.columns
        assert 'volume_imbalance' in result.columns
        
        # Volatility should be positive
        assert (result['realized_volatility'].dropna() >= 0).all()
    
    def test_time_based_features(self, sample_ohlcv_data):
        """Test time-based features"""
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_time_features(sample_ohlcv_data)
        
        assert 'hour' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_london_session' in result.columns
        assert 'is_ny_session' in result.columns
        
        # Hour should be 0-23
        assert result['hour'].min() >= 0
        assert result['hour'].max() <= 23
    
    def test_custom_usdcop_features(self, sample_ohlcv_data):
        """Test custom USDCOP-specific features"""
        from src.markets.usdcop.feature_engine import FeatureEngine
        
        engine = FeatureEngine()
        result = engine.add_usdcop_features(sample_ohlcv_data)
        
        # USDCOP-specific features
        assert 'trm_deviation' in result.columns  # Deviation from TRM
        assert 'intervention_risk' in result.columns  # Central bank intervention risk
        assert 'volatility_regime' in result.columns  # Current volatility regime
        
        # Intervention risk should be between 0 and 1
        assert result['intervention_risk'].dropna().min() >= 0
        assert result['intervention_risk'].dropna().max() <= 1