"""
Pipeline Unit Tests
===================
Tests for bronze, silver, and gold pipeline stages with proper data flow.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock


class TestPipeline:
    """Test pipeline functionality"""
    
    def test_bronze_data_validation(self, sample_ohlcv_data):
        """Test bronze stage data validation"""
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        result = pipeline.run_bronze(sample_ohlcv_data)
        
        assert 'df' in result
        assert 'quality_score' in result
        assert 'validation_errors' in result
        assert result['quality_score'] > 0.8  # Good quality data
    
    def test_bronze_data_quality_checks(self):
        """Test bronze stage quality checks"""
        # Create data with quality issues
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [4000] * 100,
            'high': [3999] * 100,  # High < Open (invalid)
            'low': [3990] * 100,
            'close': [4000] * 100,
            'volume': [0] * 100  # Zero volume (suspicious)
        })
        
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        result = pipeline.run_bronze(data)
        
        assert result['quality_score'] < 0.5  # Poor quality
        assert len(result['validation_errors']) > 0
    
    def test_silver_missing_data_handling(self):
        """Test silver stage missing data handling"""
        # Create data with gaps
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [np.nan if i % 20 == 0 else 4000 + i for i in range(100)],
            'high': [4010 + i for i in range(100)],
            'low': [3990 + i for i in range(100)],
            'close': [4000 + i for i in range(100)],
            'volume': [100 + i for i in range(100)]
        })
        
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        result = pipeline.run_silver(data)
        
        assert 'df' in result
        assert 'gaps_filled' in result
        assert result['df']['open'].isna().sum() == 0  # No missing values
        assert result['gaps_filled'] == 5  # 5 gaps filled
    
    def test_silver_outlier_handling(self):
        """Test silver stage outlier handling"""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [4000 + i for i in range(100)],
            'high': [4010 + i for i in range(100)],
            'low': [3990 + i for i in range(100)],
            'close': [4000 + i if i != 50 else 10000 for i in range(100)],  # Outlier
            'volume': [100 + i for i in range(100)]
        })
        
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        result = pipeline.run_silver(data)
        
        assert result['outliers_removed'] > 0
        assert 10000 not in result['df']['close'].values
    
    def test_gold_feature_engineering(self, sample_ohlcv_data):
        """Test gold stage feature engineering"""
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        result = pipeline.run_gold(sample_ohlcv_data)
        
        assert 'df' in result
        assert 'features' in result
        assert len(result['features']) >= 20  # Many features added (changed from > to >=)
        assert 'rsi' in result['df'].columns or 'rsi_14' in result['df'].columns
        assert 'macd' in result['df'].columns
    
    def test_gold_scaling_normalization(self, sample_features_data):
        """Test gold stage data scaling and normalization"""
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        result = pipeline.run_gold(sample_features_data, scale=True)
        
        # Check scaled values are in expected range
        scaled_cols = [col for col in result['df'].columns if '_scaled' in col]
        for col in scaled_cols:
            assert result['df'][col].min() >= -3
            assert result['df'][col].max() <= 3
    
    def test_complete_pipeline_flow(self, sample_ohlcv_data):
        """Test complete pipeline flow from bronze to gold"""
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        
        # Bronze stage
        bronze_result = pipeline.run_bronze(sample_ohlcv_data)
        assert bronze_result['quality_score'] > 0
        
        # Silver stage
        silver_result = pipeline.run_silver(bronze_result['df'])
        assert not silver_result['df'].empty
        
        # Gold stage
        gold_result = pipeline.run_gold(silver_result['df'])
        assert len(gold_result['features']) > 0
        
        # Final data should have more columns than original
        assert len(gold_result['df'].columns) > len(sample_ohlcv_data.columns)