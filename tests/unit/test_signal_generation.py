"""
Signal Generation Unit Tests
============================
Tests for trading signal generation from model predictions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch


class TestSignalGeneration:
    """Test signal generation functionality"""
    
    def test_buy_signal_generation(self):
        """Test buy signal generation from model output"""
        from src.trading.signal_generator import SignalGenerator
        
        # Mock model that predicts buy
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([0]), None)  # 0 = Buy
        
        generator = SignalGenerator(model=mock_model)
        
        # Create sample data
        data = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        
        signal = generator.generate_signal(data)
        assert signal == 'BUY'
    
    def test_sell_signal_generation(self):
        """Test sell signal generation from model output"""
        from src.trading.signal_generator import SignalGenerator
        
        # Mock model that predicts sell
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([1]), None)  # 1 = Sell
        
        generator = SignalGenerator(model=mock_model)
        
        data = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        
        signal = generator.generate_signal(data)
        assert signal == 'SELL'
    
    def test_hold_signal_generation(self):
        """Test hold signal generation from model output"""
        from src.trading.signal_generator import SignalGenerator
        
        # Mock model that predicts hold
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([2]), None)  # 2 = Hold
        
        generator = SignalGenerator(model=mock_model)
        
        data = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        
        signal = generator.generate_signal(data)
        assert signal == 'HOLD'
    
    def test_signal_confidence_threshold(self):
        """Test signal generation with confidence threshold"""
        from src.trading.signal_generator import SignalGenerator
        
        mock_model = Mock()
        # Return action with confidence
        mock_model.predict.return_value = (np.array([0]), {'confidence': 0.3})
        
        generator = SignalGenerator(
            model=mock_model,
            confidence_threshold=0.7
        )
        
        data = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        
        # Should return HOLD due to low confidence
        signal = generator.generate_signal(data)
        assert signal == 'HOLD'
    
    def test_ensemble_signal_generation(self):
        """Test ensemble signal from multiple models"""
        from src.trading.signal_generator import EnsembleSignalGenerator
        
        # Create multiple mock models
        models = []
        for prediction in [0, 0, 1, 0, 2]:  # 3 buys, 1 sell, 1 hold
            model = Mock()
            model.predict.return_value = (np.array([prediction]), None)
            models.append(model)
        
        generator = EnsembleSignalGenerator(models=models)
        
        data = pd.DataFrame({'feature1': [1], 'feature2': [2]})
        
        # Majority vote should be BUY
        signal = generator.generate_signal(data)
        assert signal == 'BUY'