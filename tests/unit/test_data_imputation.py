"""
Data Imputation Unit Tests
==========================
Tests for handling missing data, forward fill, interpolation, and outlier detection.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestDataImputation:
    """Test data imputation functionality"""
    
    def test_missing_data_detection(self):
        """Test detection of missing data in OHLCV"""
        # Create data with missing values
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': [np.nan if i % 10 == 0 else 4000 + i for i in range(100)],
            'high': [4010 + i for i in range(100)],
            'low': [3990 + i for i in range(100)],
            'close': [4000 + i for i in range(100)],
            'volume': [100 + i for i in range(100)]
        })
        
        from src.core.data.imputation import DataQualityChecker
        
        checker = DataQualityChecker()
        missing_info = checker.detect_missing(data)
        
        assert missing_info['open'] == 10  # 10 missing values
        assert missing_info['high'] == 0
    
    def test_forward_fill_imputation(self):
        """Test forward fill imputation method"""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100, np.nan, np.nan, 103, np.nan, 105, 106, np.nan, 108, 109]
        })
        
        from src.core.data.imputation import Imputer
        
        imputer = Imputer(method='forward_fill')
        filled_data = imputer.impute(data)
        
        assert filled_data['close'].isna().sum() == 0
        assert filled_data['close'].iloc[1] == 100  # Forward filled
        assert filled_data['close'].iloc[2] == 100  # Forward filled
    
    def test_linear_interpolation(self):
        """Test linear interpolation for missing values"""
        data = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=10, freq='5min'),
            'close': [100, np.nan, np.nan, 106, np.nan, 110, 112, np.nan, 116, 118]
        })
        
        from src.core.data.imputation import Imputer
        
        imputer = Imputer(method='linear')
        filled_data = imputer.impute(data)
        
        assert filled_data['close'].isna().sum() == 0
        assert filled_data['close'].iloc[1] == 102  # Interpolated
        assert filled_data['close'].iloc[2] == 104  # Interpolated
        assert filled_data['close'].iloc[4] == 108  # Interpolated
    
    def test_outlier_detection_zscore(self):
        """Test outlier detection using Z-score method"""
        np.random.seed(42)
        data = pd.DataFrame({
            'close': np.random.normal(4000, 10, 100)
        })
        # Add outliers
        data.loc[10, 'close'] = 4200  # Outlier
        data.loc[50, 'close'] = 3800  # Outlier
        
        from src.core.data.outlier_detection import OutlierDetector
        
        detector = OutlierDetector(method='zscore', threshold=3)
        outliers = detector.detect(data['close'])
        
        assert 10 in outliers
        assert 50 in outliers
    
    def test_outlier_removal_iqr(self):
        """Test outlier removal using IQR method"""
        data = pd.DataFrame({
            'close': [4000, 4001, 4002, 5000, 4003, 4004, 3000, 4005, 4006, 4007]
        })
        
        from src.core.data.outlier_detection import OutlierDetector
        
        detector = OutlierDetector(method='iqr')
        cleaned_data = detector.remove_outliers(data)
        
        assert 5000 not in cleaned_data['close'].values
        assert 3000 not in cleaned_data['close'].values
    
    def test_gap_filling_business_hours(self):
        """Test gap filling for business hours only"""
        # Create data with weekend gap
        dates = pd.date_range('2024-01-05 21:00', periods=10, freq='H')  # Friday evening
        data = pd.DataFrame({
            'time': dates,
            'close': range(4000, 4010)
        })
        
        from src.core.data.gap_filler import GapFiller
        
        filler = GapFiller(business_hours_only=True)
        filled_data = filler.fill_gaps(data)
        
        # Should not fill weekend gaps
        assert len(filled_data) == len(data)
    
    def test_multivariate_imputation(self):
        """Test multivariate imputation using correlations"""
        # Create correlated data with missing values
        np.random.seed(42)
        size = 100
        data = pd.DataFrame({
            'open': np.random.normal(4000, 10, size),
            'close': np.nan,  # Will be imputed
            'high': np.random.normal(4010, 10, size),
            'low': np.random.normal(3990, 10, size)
        })
        # Close is correlated with open
        data['close'] = data['open'] + np.random.normal(0, 2, size)
        # Add missing values
        data.loc[10:20, 'close'] = np.nan
        
        from src.core.data.imputation import MultivarImputer
        
        imputer = MultivarImputer()
        filled_data = imputer.impute(data)
        
        assert filled_data['close'].isna().sum() == 0
        # Imputed values should be close to open values
        imputed_values = filled_data.loc[10:20, 'close']
        open_values = filled_data.loc[10:20, 'open']
        assert np.abs(imputed_values - open_values).mean() < 10
    
    def test_seasonal_imputation(self):
        """Test seasonal pattern-aware imputation"""
        # Create data with seasonal pattern
        dates = pd.date_range('2024-01-01', periods=288, freq='5min')  # One day
        hours = dates.hour
        data = pd.DataFrame({
            'time': dates,
            'volume': 1000 + 500 * np.sin(2 * np.pi * hours / 24)  # Daily pattern
        })
        # Add missing values
        data.loc[50:60, 'volume'] = np.nan
        
        from src.core.data.imputation import SeasonalImputer
        
        imputer = SeasonalImputer(period=288)  # Daily period
        filled_data = imputer.impute(data)
        
        assert filled_data['volume'].isna().sum() == 0
        # Imputed values should follow seasonal pattern
        imputed_mean = filled_data.loc[50:60, 'volume'].mean()
        assert 900 < imputed_mean < 1100  # Should be around base value