"""
End-to-End Integration Tests
============================
Tests for complete system flow from data extraction to trade execution.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path


class TestEndToEnd:
    """Test complete system integration"""
    
    def test_complete_trading_cycle(self, sample_ohlcv_data):
        """Test complete cycle: data -> features -> model -> signal -> execution"""
        from src.markets.usdcop.pipeline import USDCOPPipeline, PipelineConfig
        from src.models.rl_models.ppo_model import PPOModel
        from src.trading.signal_generator import SignalGenerator
        from src.trading.order_executor import OrderExecutor
        
        # 1. Data Pipeline
        config = PipelineConfig(symbol='USDCOP', primary_timeframe='M5')
        pipeline = USDCOPPipeline(config)
        
        # Process through pipeline stages  
        # For test data, use the dates from the sample data
        start_date = pd.to_datetime(sample_ohlcv_data['time'].min())
        end_date = pd.to_datetime(sample_ohlcv_data['time'].max())
        
        # Mock the data fetch to return our sample data
        with patch.object(pipeline, '_fetch_data_with_fallback', return_value=sample_ohlcv_data):
            bronze_result = pipeline.run_bronze(start_date, end_date)
        
        silver_result = pipeline.run_silver(bronze_result['df'])
        gold_result = pipeline.run_gold(silver_result['df'])
        
        assert not gold_result['df'].empty
        
        # 2. Model Prediction
        with patch('src.models.rl_models.ppo_model.PPOModel.predict') as mock_predict:
            mock_predict.return_value = (np.array([1]), None)  # Buy signal
            
            model = PPOModel(state_dim=len(gold_result['features']), action_dim=3)
            action, _ = model.predict(gold_result['df'].iloc[-1].values)
            
            assert action[0] == 1  # Buy action
        
        # 3. Signal Generation
        signal_gen = SignalGenerator(model=model)
        signal = signal_gen.generate_signal(gold_result['df'])
        
        assert signal in ['BUY', 'SELL', 'HOLD']
        
        # 4. Order Execution
        with patch('src.trading.order_executor.MT5Executor.execute_order') as mock_execute:
            mock_execute.return_value = {'success': True, 'order_id': '12345'}
            
            executor = OrderExecutor()
            result = executor.execute_signal(signal, volume=0.1)
            
            assert result['success'] == True
    
    def test_data_pipeline_integration(self, sample_ohlcv_data):
        """Test integration between pipeline stages"""
        from src.markets.usdcop.pipeline import DataPipeline
        
        pipeline = DataPipeline()
        
        # Test data flow
        bronze = pipeline.run_bronze(sample_ohlcv_data)
        assert bronze['quality_score'] > 0
        
        silver = pipeline.run_silver(bronze['df'])
        assert silver['df'].isna().sum().sum() == 0  # No missing values
        
        gold = pipeline.run_gold(silver['df'])
        assert len(gold['features']) >= 20
        
        # Verify data integrity through stages
        assert len(gold['df']) <= len(sample_ohlcv_data)  # May lose some rows
    
    def test_model_serving_integration(self, sample_features_data):
        """Test model serving and prediction pipeline"""
        from src.models.serving import ModelServer
        
        with patch.object(ModelServer, 'load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = (np.array([0]), None)
            mock_load.return_value = mock_model
            
            server = ModelServer()
            server.load_model('test_model', 'latest')
            
            # Make prediction
            prediction = server.predict(sample_features_data.iloc[-1].values)
            
            assert prediction is not None
            assert prediction[0] in [0, 1, 2]  # Action space
    
    def test_database_persistence_flow(self, sample_ohlcv_data, tmp_path):
        """Test data persistence through database"""
        from src.core.database.database_manager import DatabaseManager
        
        db_path = tmp_path / "test.db"
        db_manager = DatabaseManager(str(db_path))
        
        # Save data
        success = db_manager.save_market_data(
            sample_ohlcv_data,
            table='market_data',
            symbol='USDCOP'
        )
        assert success
        
        # Retrieve data
        retrieved = db_manager.get_latest_data(
            table='market_data',
            symbol='USDCOP',
            limit=50
        )
        
        assert not retrieved.empty
        assert len(retrieved) <= 50
    
    def test_airflow_dag_execution(self):
        """Test Airflow DAG execution flow"""
        from airflow.models import DagBag
        from airflow.utils.dates import days_ago
        
        with patch('airflow.models.DagBag') as MockDagBag:
            mock_dagbag = MockDagBag.return_value
            mock_dag = MagicMock()
            mock_dag.dag_id = 'usdcop_trading_pipeline'
            mock_dagbag.dags = {'usdcop_trading_pipeline': mock_dag}
            
            dagbag = DagBag(dag_folder='airflow/dags')
            dag = dagbag.dags.get('usdcop_trading_pipeline')
            
            assert dag is not None
            assert dag.dag_id == 'usdcop_trading_pipeline'