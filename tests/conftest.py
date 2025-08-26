"""
Pytest Configuration and Shared Fixtures
=========================================
Central configuration for all tests with shared fixtures and utilities.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import os

# Set test environment
os.environ['ENVIRONMENT'] = 'test'
os.environ['TESTING'] = 'true'


@pytest.fixture(scope='session')
def test_data_dir():
    """Create temporary data directory for tests"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing"""
    size = 100
    dates = pd.date_range('2024-01-01', periods=size, freq='5min')
    
    np.random.seed(42)
    base_price = 4000
    returns = np.random.randn(size) * 0.001
    prices = base_price * np.exp(np.cumsum(returns))
    
    return pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.randn(size) * 0.0001),
        'high': prices * (1 + np.abs(np.random.randn(size)) * 0.0002),
        'low': prices * (1 - np.abs(np.random.randn(size)) * 0.0002),
        'close': prices,
        'volume': np.random.randint(100, 1000, size),
        'tick_volume': np.random.randint(10, 100, size),
        'spread': np.random.uniform(1, 5, size)
    })


@pytest.fixture
def mock_mt5_config():
    """Mock MT5 configuration"""
    return {
        'server': 'test-server',
        'login': 12345,
        'password': 'test-pass',
        'timeout': 60000
    }


@pytest.fixture
def sample_features_data(sample_ohlcv_data):
    """Generate sample data with features"""
    df = sample_ohlcv_data.copy()
    
    # Add simple technical indicators
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['rsi'] = 50 + np.random.randn(len(df)) * 10
    df['macd'] = np.random.randn(len(df)) * 0.001
    
    return df.dropna()


@pytest.fixture
def mock_database(monkeypatch):
    """Mock database for testing"""
    class MockDB:
        def __init__(self):
            self.data = {}
        
        def save(self, table, data):
            self.data[table] = data
            return True
        
        def load(self, table):
            return self.data.get(table, pd.DataFrame())
    
    mock_db = MockDB()
    monkeypatch.setattr('src.core.database.db_integration.db_integration', mock_db)
    return mock_db