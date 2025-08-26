"""
Comprehensive Integration Tests
===============================
Tests the integration between all system components to ensure
seamless data flow and proper error handling.
"""

import pytest
import pandas as pd
import numpy as np
import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import system components
from src.core.config.unified_config import config_loader, get_config, get_system_config
from src.core.database.db_integration import db_integration
from src.core.errors.handlers import TradingSystemError, handle_errors, error_handler
from src.core.monitoring.health_checks import health_checker, get_health_status
from src.markets.usdcop.feature_engine import FeatureEngine
from src.markets.usdcop.pipeline import USDCOPPipeline, EnhancedFeatureEngine, DataPipeline


# ===========================
# Test Fixtures
# ===========================

@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing"""
    size = 500
    dates = pd.date_range('2024-01-01', periods=size, freq='5min')
    
    # Generate realistic price data
    np.random.seed(42)
    base_price = 4000
    returns = np.random.randn(size) * 0.001
    prices = base_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.randn(size) * 0.0001),
        'high': prices * (1 + np.abs(np.random.randn(size)) * 0.0002),
        'low': prices * (1 - np.abs(np.random.randn(size)) * 0.0002),
        'close': prices,
        'volume': np.random.randint(100, 1000, size),
        'tick_volume': np.random.randint(10, 100, size),
        'spread': np.random.randint(1, 5, size)
    })
    
    # Ensure OHLC relationships
    data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
    data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
    
    return data


@pytest.fixture
def temp_database(tmp_path):
    """Create temporary database for testing"""
    db_path = tmp_path / "test_trading.db"
    
    # Patch the database path in config
    original_db_path = get_system_config().db_path
    get_system_config().db_path = str(db_path)
    
    yield db_path
    
    # Restore original path
    get_system_config().db_path = original_db_path


# ===========================
# Configuration Tests
# ===========================

class TestConfigurationIntegration:
    """Test configuration system integration"""
    
    def test_config_loader_initialization(self):
        """Test configuration loader initializes properly"""
        assert config_loader.initialized
        assert config_loader.get_system_config() is not None
    
    def test_config_loading(self):
        """Test loading configuration files"""
        # MT5 config should be loaded
        mt5_config = get_config('mt5')
        assert isinstance(mt5_config, dict)
        
        # USDCOP config should be loaded
        usdcop_config = get_config('usdcop')
        assert isinstance(usdcop_config, dict)
    
    def test_env_var_replacement(self):
        """Test environment variable replacement in configs"""
        import os
        os.environ['TEST_VAR'] = 'test_value'
        
        test_config = {'key': '${TEST_VAR}'}
        replaced = config_loader._replace_env_vars(test_config)
        
        assert replaced['key'] == 'test_value'
        
        del os.environ['TEST_VAR']
    
    def test_nested_config_access(self):
        """Test accessing nested configuration values"""
        # This should work if mt5_config.yaml has connection.timeout
        value = config_loader.get_nested_config('mt5.connection.timeout', default=60000)
        assert value is not None


# ===========================
# Database Integration Tests
# ===========================

class TestDatabaseIntegration:
    """Test database integration across components"""
    
    def test_database_initialization(self, temp_database):
        """Test database initializes properly"""
        assert db_integration.initialized
        assert db_integration.db_manager is not None
    
    def test_save_and_retrieve_market_data(self, temp_database, sample_ohlcv_data):
        """Test saving and retrieving market data"""
        # Save data
        success = db_integration.save_market_data(
            sample_ohlcv_data,
            symbol="USDCOP",
            timeframe="M5"
        )
        assert success
        
        # Retrieve data
        retrieved = db_integration.get_latest_data("USDCOP", "M5", limit=100)
        assert not retrieved.empty
        assert len(retrieved) <= 100
    
    def test_save_model_metrics(self, temp_database):
        """Test saving model metrics"""
        metrics = {
            'accuracy': 0.85,
            'loss': 0.15,
            'sharpe_ratio': 1.5
        }
        
        success = db_integration.save_model_metrics(metrics, "test_model_001")
        assert success
        
        # Retrieve metrics
        retrieved = db_integration.get_model_metrics("test_model_001")
        assert not retrieved.empty
    
    def test_save_trades(self, temp_database):
        """Test saving trade data"""
        import uuid
        # Use unique strategy_id to avoid conflicts
        strategy_id = f"test_strategy_{uuid.uuid4().hex[:8]}"
        
        trades = [
            {'symbol': 'USDCOP', 'action': 'BUY', 'price': 4000, 'volume': 0.1},
            {'symbol': 'USDCOP', 'action': 'SELL', 'price': 4010, 'volume': 0.1}
        ]
        
        success = db_integration.save_trades(trades, strategy_id)
        assert success
        
        # Retrieve trades
        retrieved = db_integration.get_trades(strategy_id)
        assert not retrieved.empty
        assert len(retrieved) == 2
    
    def test_database_transaction_rollback(self, temp_database):
        """Test database transaction rollback on error"""
        with pytest.raises(Exception):
            with db_integration.transaction() as conn:
                if conn:
                    # This should cause an error
                    conn.execute("INVALID SQL")


# ===========================
# Feature Engineering Tests
# ===========================

class TestFeatureEngineIntegration:
    """Test feature engineering integration"""
    
    def test_feature_engine_initialization(self):
        """Test feature engine initializes properly"""
        engine = FeatureEngine()
        assert engine is not None
        assert hasattr(engine, 'add_all_features')
    
    def test_feature_generation(self, sample_ohlcv_data):
        """Test feature generation on sample data"""
        engine = FeatureEngine()
        featured_data = engine.add_all_features(sample_ohlcv_data)
        
        # Should have more columns than original
        assert len(featured_data.columns) > len(sample_ohlcv_data.columns)
        
        # Check for specific features
        assert 'rsi_14' in featured_data.columns
        assert 'sma_20' in featured_data.columns
        assert 'macd' in featured_data.columns
    
    def test_enhanced_feature_engine(self, sample_ohlcv_data):
        """Test enhanced feature engine from pipeline"""
        engine = EnhancedFeatureEngine(use_ray=False)
        featured_data = engine.add_features_parallel(sample_ohlcv_data)
        
        assert not featured_data.empty
        assert len(featured_data.columns) > len(sample_ohlcv_data.columns)


# ===========================
# Pipeline Integration Tests
# ===========================

class TestPipelineIntegration:
    """Test data pipeline integration"""
    
    @pytest.fixture
    def mock_connector(self):
        """Create mock connector for testing"""
        connector = MagicMock()
        connector.get_latest_rates.return_value = pd.DataFrame({
            'time': pd.date_range('2024-01-01', periods=100, freq='5min'),
            'open': np.random.randn(100) + 4000,
            'high': np.random.randn(100) + 4010,
            'low': np.random.randn(100) + 3990,
            'close': np.random.randn(100) + 4000,
            'volume': np.random.randint(100, 1000, 100)
        })
        return connector
    
    def test_pipeline_initialization(self, mock_connector):
        """Test pipeline initializes properly"""
        pipeline = DataPipeline(connector=mock_connector)
        assert pipeline is not None
        assert pipeline.feature_engine is not None
    
    def test_bronze_to_silver_flow(self, sample_ohlcv_data):
        """Test data flow from bronze to silver stage"""
        pipeline = DataPipeline()
        
        # Process bronze stage
        bronze_result = pipeline.run_bronze(sample_ohlcv_data)
        assert 'df' in bronze_result
        assert bronze_result['quality_score'] > 0
        
        # Process silver stage
        silver_result = pipeline.run_silver(bronze_result['df'])
        assert 'df' in silver_result
        assert not silver_result['df'].empty
    
    def test_complete_pipeline_flow(self, sample_ohlcv_data, temp_database):
        """Test complete pipeline flow from bronze to gold"""
        pipeline = DataPipeline()
        
        # Run bronze stage
        bronze_result = pipeline.run_bronze(sample_ohlcv_data)
        assert bronze_result['quality_score'] > 0
        
        # Run silver stage
        silver_result = pipeline.run_silver(bronze_result['df'])
        assert silver_result['gaps_filled'] >= 0
        
        # Run gold stage
        gold_result = pipeline.run_gold(silver_result['df'])
        assert 'features' in gold_result
        assert len(gold_result['features']) > 0


# ===========================
# Error Handling Tests
# ===========================

class TestErrorHandlingIntegration:
    """Test error handling integration"""
    
    def test_error_handler_initialization(self):
        """Test error handler initializes properly"""
        assert error_handler is not None
        assert hasattr(error_handler, 'handle')
    
    def test_handle_errors_decorator(self):
        """Test handle_errors decorator"""
        
        @handle_errors(
            component="test_component",
            fallback_value=None,
            reraise=False
        )
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result is None
    
    def test_error_tracking(self):
        """Test error tracking and statistics"""
        # Clear previous errors
        error_handler.reset_stats()
        
        # Generate some errors
        for i in range(5):
            error_handler.handle(
                "test_component",
                ValueError(f"Test error {i}"),
                context={'iteration': i}
            )
        
        # Check statistics
        stats = error_handler.get_error_stats()
        assert stats['total_errors'] == 5
        assert 'test_component' in stats['last_errors']
    
    def test_custom_exceptions(self):
        """Test custom exception classes"""
        error = TradingSystemError(
            "Test error",
            code="TEST001",
            details={'component': 'test'}
        )
        
        assert error.code == "TEST001"
        assert error.details['component'] == 'test'
        assert error.timestamp is not None


# ===========================
# Health Check Tests
# ===========================

class TestHealthCheckIntegration:
    """Test health check integration"""
    
    def test_health_checker_initialization(self):
        """Test health checker initializes properly"""
        assert health_checker is not None
        assert len(health_checker.checks) > 0
    
    def test_run_health_checks(self):
        """Test running health checks"""
        results = health_checker.run_checks()
        
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check specific components
        if 'database' in results:
            assert results['database'].component == 'database'
            assert results['database'].status in ['healthy', 'degraded', 'unhealthy', 'unknown']
    
    def test_system_health_status(self):
        """Test getting system health status"""
        status = get_health_status()
        
        assert 'overall_status' in status
        assert 'components' in status
        assert 'system_metrics' in status
        assert 'uptime' in status
    
    def test_health_trends(self):
        """Test health trend analysis"""
        # Run checks multiple times
        for _ in range(3):
            health_checker.run_checks()
            time.sleep(0.1)
        
        trends = health_checker.get_health_trends()
        assert isinstance(trends, dict)


# ===========================
# End-to-End Integration Tests
# ===========================

class TestEndToEndIntegration:
    """Test complete system integration"""
    
    def test_complete_data_flow(self, sample_ohlcv_data, temp_database):
        """Test complete data flow through the system"""
        
        # 1. Initialize configuration
        assert config_loader.initialized
        
        # 2. Process data through pipeline
        pipeline = DataPipeline()
        bronze_result = pipeline.run_bronze(sample_ohlcv_data)
        silver_result = pipeline.run_silver(bronze_result['df'])
        gold_result = pipeline.run_gold(silver_result['df'])
        
        # 3. Save to database
        success = db_integration.save_market_data(
            gold_result['df'],
            symbol="USDCOP",
            timeframe="M5"
        )
        assert success
        
        # 4. Retrieve from database
        retrieved = db_integration.get_latest_data("USDCOP", "M5")
        assert not retrieved.empty
        
        # 5. Check health
        health = get_health_status()
        assert health['overall_status'] in ['healthy', 'degraded', 'unhealthy', 'unknown']
    
    def test_error_recovery(self):
        """Test system recovers from errors"""
        
        @handle_errors(
            component="test",
            fallback_value="recovered",
            reraise=False,
            max_retries=2
        )
        def flaky_function():
            if not hasattr(flaky_function, 'call_count'):
                flaky_function.call_count = 0
            flaky_function.call_count += 1
            
            if flaky_function.call_count < 3:
                raise ConnectionError("Temporary failure")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert flaky_function.call_count == 3
    
    def test_concurrent_operations(self, sample_ohlcv_data, temp_database):
        """Test concurrent database operations"""
        import threading
        
        results = []
        
        def save_data(thread_id):
            try:
                success = db_integration.save_market_data(
                    sample_ohlcv_data,
                    symbol=f"TEST_{thread_id}",
                    timeframe="M5"
                )
                results.append(success)
            except Exception as e:
                results.append(False)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=save_data, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for all threads
        for t in threads:
            t.join()
        
        # Check results
        assert len(results) == 5
        assert any(results)  # At least some should succeed


# ===========================
# Performance Tests
# ===========================

class TestPerformanceIntegration:
    """Test system performance"""
    
    def test_feature_generation_performance(self, sample_ohlcv_data):
        """Test feature generation performance"""
        engine = FeatureEngine()
        
        start_time = time.time()
        featured_data = engine.add_all_features(sample_ohlcv_data)
        elapsed = time.time() - start_time
        
        # Should process 500 rows in under 5 seconds
        assert elapsed < 5.0
        assert not featured_data.empty
    
    def test_database_write_performance(self, sample_ohlcv_data, temp_database):
        """Test database write performance"""
        start_time = time.time()
        
        success = db_integration.save_market_data(
            sample_ohlcv_data,
            symbol="PERF_TEST",
            timeframe="M5"
        )
        
        elapsed = time.time() - start_time
        
        # Should save 500 rows in under 1 second
        assert success
        assert elapsed < 1.0
    
    def test_health_check_performance(self):
        """Test health check performance"""
        start_time = time.time()
        health = get_health_status()
        elapsed = time.time() - start_time
        
        # Health check should complete in under 2 seconds
        assert elapsed < 2.0
        assert health is not None


# ===========================
# Run Tests
# ===========================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])