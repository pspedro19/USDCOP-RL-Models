"""
Forecasting Pipeline E2E Tests
==============================

End-to-end tests for the forecasting pipeline, validating:
1. Data contracts alignment
2. Feature engineering
3. Model training
4. Inference pipeline
5. SSOT consistency

@version 1.0.0
"""

import pytest
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDataContracts:
    """Test SSOT data contracts."""

    def test_feature_columns_count(self):
        """Verify FEATURE_COLUMNS has exactly 19 features."""
        from src.forecasting.data_contracts import FEATURE_COLUMNS, NUM_FEATURES

        assert len(FEATURE_COLUMNS) == 19, f"Expected 19, got {len(FEATURE_COLUMNS)}"
        assert NUM_FEATURES == 19, f"NUM_FEATURES mismatch: {NUM_FEATURES}"

    def test_feature_columns_order(self):
        """Verify feature columns are in expected order."""
        from src.forecasting.data_contracts import FEATURE_COLUMNS

        expected_order = [
            "close", "open", "high", "low",
            "return_1d", "return_5d", "return_10d", "return_20d",
            "volatility_5d", "volatility_10d", "volatility_20d",
            "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
            "day_of_week", "month", "is_month_end",
            "dxy_close_lag1", "oil_close_lag1",
        ]

        for i, (expected, actual) in enumerate(zip(expected_order, FEATURE_COLUMNS)):
            assert expected == actual, f"Position {i}: expected '{expected}', got '{actual}'"

    def test_target_horizons(self):
        """Verify TARGET_HORIZONS are correct."""
        from src.forecasting.data_contracts import TARGET_HORIZONS

        expected = (1, 5, 10, 15, 20, 25, 30)
        assert TARGET_HORIZONS == expected, f"Expected {expected}, got {TARGET_HORIZONS}"

    def test_contract_hash_deterministic(self):
        """Verify contract hash is deterministic."""
        from src.forecasting.data_contracts import compute_data_contract_hash

        hash1 = compute_data_contract_hash()
        hash2 = compute_data_contract_hash()

        assert hash1 == hash2, "Contract hash should be deterministic"
        assert len(hash1) == 16, f"Hash should be 16 chars, got {len(hash1)}"


class TestConfigSSoT:
    """Test configuration SSOT."""

    def test_config_singleton(self):
        """Verify config singleton works."""
        from src.forecasting.config import get_config

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2, "Config should be singleton"

    def test_config_features_match_contracts(self):
        """Verify config features match data_contracts."""
        from src.forecasting.config import get_config
        from src.forecasting.data_contracts import FEATURE_COLUMNS

        config = get_config()

        assert config.features.feature_columns == FEATURE_COLUMNS, "Config features should match contracts"

    def test_config_horizons_match_contracts(self):
        """Verify config horizons match data_contracts."""
        from src.forecasting.config import get_config
        from src.forecasting.data_contracts import TARGET_HORIZONS

        config = get_config()

        assert config.features.target_horizons == TARGET_HORIZONS, "Config horizons should match contracts"

    def test_config_hash_computation(self):
        """Verify config hash is computed correctly."""
        from src.forecasting.config import get_config

        config = get_config()
        hash_value = config.compute_hash()

        assert len(hash_value) == 16, f"Hash should be 16 chars, got {len(hash_value)}"


class TestEngineValidation:
    """Test engine validation functionality."""

    def test_feature_validation_success(self):
        """Test validation passes with correct features."""
        from src.forecasting.engine import ForecastingEngine
        from src.forecasting.data_contracts import FEATURE_COLUMNS

        # Create mock DataFrame with all features
        n_rows = 100
        data = {col: np.random.randn(n_rows) for col in FEATURE_COLUMNS}
        data['date'] = pd.date_range('2020-01-01', periods=n_rows)
        df = pd.DataFrame(data)

        engine = ForecastingEngine()
        is_valid, errors = engine._validate_features(df)

        assert is_valid, f"Validation should pass: {errors}"
        assert len(errors) == 0, "No errors expected"

    def test_feature_validation_missing_columns(self):
        """Test validation fails with missing columns."""
        from src.forecasting.engine import ForecastingEngine
        from src.forecasting.data_contracts import FEATURE_COLUMNS

        # Create DataFrame missing some features
        n_rows = 100
        data = {col: np.random.randn(n_rows) for col in FEATURE_COLUMNS[:10]}  # Only first 10
        data['date'] = pd.date_range('2020-01-01', periods=n_rows)
        df = pd.DataFrame(data)

        engine = ForecastingEngine()
        is_valid, errors = engine._validate_features(df)

        assert not is_valid, "Validation should fail"
        assert len(errors) > 0, "Errors expected"
        assert any("Missing" in e for e in errors), "Should report missing features"


class TestModelsContract:
    """Test model contracts."""

    def test_model_ids_count(self):
        """Verify correct number of models."""
        from src.forecasting.contracts import MODEL_IDS

        assert len(MODEL_IDS) == 9, f"Expected 9 models, got {len(MODEL_IDS)}"

    def test_model_factory_all_models(self):
        """Verify factory can create all models."""
        from src.forecasting.models.factory import ModelFactory
        from src.forecasting.contracts import MODEL_IDS

        for model_id in MODEL_IDS:
            try:
                model = ModelFactory.create(model_id)
                assert model is not None, f"Model {model_id} should be created"
            except ImportError as e:
                # Some models may need optional dependencies
                pytest.skip(f"Model {model_id} requires optional dependency: {e}")

    def test_horizons_count(self):
        """Verify correct number of horizons."""
        from src.forecasting.contracts import HORIZONS

        assert len(HORIZONS) == 7, f"Expected 7 horizons, got {len(HORIZONS)}"


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset with all required features."""
        from src.forecasting.data_contracts import FEATURE_COLUMNS, TARGET_HORIZONS

        n_rows = 500
        np.random.seed(42)

        # Generate realistic OHLCV data
        base_price = 4000
        returns = np.random.randn(n_rows) * 0.01
        close = base_price * np.cumprod(1 + returns)

        data = {
            'date': pd.date_range('2020-01-01', periods=n_rows, freq='B'),
            'close': close,
            'open': close * (1 + np.random.randn(n_rows) * 0.005),
            'high': close * (1 + np.abs(np.random.randn(n_rows) * 0.01)),
            'low': close * (1 - np.abs(np.random.randn(n_rows) * 0.01)),
        }

        # Add returns
        for window in [1, 5, 10, 20]:
            data[f'return_{window}d'] = pd.Series(close).pct_change(window).values

        # Add volatility
        for window in [5, 10, 20]:
            data[f'volatility_{window}d'] = pd.Series(data['return_1d']).rolling(window).std().values

        # Add technical
        data['rsi_14d'] = 50 + np.random.randn(n_rows) * 15  # Mock RSI
        data['ma_ratio_20d'] = 1 + np.random.randn(n_rows) * 0.02
        data['ma_ratio_50d'] = 1 + np.random.randn(n_rows) * 0.03

        # Add calendar
        dates = pd.to_datetime(data['date'])
        data['day_of_week'] = dates.dayofweek.astype(float)
        data['month'] = dates.month.astype(float)
        data['is_month_end'] = dates.is_month_end.astype(float)

        # Add macro
        data['dxy_close_lag1'] = 100 + np.random.randn(n_rows) * 5
        data['oil_close_lag1'] = 70 + np.random.randn(n_rows) * 10

        # Add targets
        for h in TARGET_HORIZONS:
            data[f'target_{h}d'] = np.roll(close, -h)
            data[f'target_return_{h}d'] = np.log(np.roll(close, -h) / close)

        df = pd.DataFrame(data)

        return df

    def test_dataset_feature_alignment(self, sample_dataset):
        """Test that sample dataset has all required features."""
        from src.forecasting.data_contracts import FEATURE_COLUMNS

        for col in FEATURE_COLUMNS:
            assert col in sample_dataset.columns, f"Missing feature: {col}"

    def test_engine_prepare_features(self, sample_dataset):
        """Test engine can prepare features from dataset."""
        from src.forecasting.engine import ForecastingEngine

        engine = ForecastingEngine()

        # Prepare features
        X, feature_cols = engine._prepare_features(sample_dataset)

        # Verify shape
        assert X.shape[1] == 19, f"Expected 19 features, got {X.shape[1]}"
        assert len(feature_cols) == 19, f"Expected 19 feature names"

    def test_engine_create_targets(self, sample_dataset):
        """Test engine can create targets for all horizons."""
        from src.forecasting.engine import ForecastingEngine
        from src.forecasting.contracts import HORIZONS

        engine = ForecastingEngine()

        targets = engine._create_targets(sample_dataset, list(HORIZONS))

        # Verify all horizons have targets
        for h in HORIZONS:
            assert h in targets, f"Missing target for horizon {h}"
            assert len(targets[h]) == len(sample_dataset), f"Target length mismatch for H={h}"


class TestParamsYAML:
    """Test params.yaml configuration."""

    def test_params_yaml_exists(self):
        """Verify params.yaml exists."""
        params_path = PROJECT_ROOT / "params.yaml"
        assert params_path.exists(), "params.yaml should exist"

    def test_params_yaml_has_forecasting_section(self):
        """Verify params.yaml has forecasting section."""
        import yaml

        params_path = PROJECT_ROOT / "params.yaml"

        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert "forecasting" in params, "params.yaml should have forecasting section"

    def test_params_yaml_features_count(self):
        """Verify params.yaml has correct feature count."""
        import yaml

        params_path = PROJECT_ROOT / "params.yaml"

        with open(params_path) as f:
            params = yaml.safe_load(f)

        forecasting = params.get("forecasting", {})
        features = forecasting.get("features", {})
        num_features = features.get("num_features", 0)

        assert num_features == 19, f"Expected 19 features in params.yaml, got {num_features}"


class TestDVCStages:
    """Test DVC pipeline configuration."""

    def test_dvc_yaml_exists(self):
        """Verify dvc.yaml exists."""
        dvc_path = PROJECT_ROOT / "dvc.yaml"
        assert dvc_path.exists(), "dvc.yaml should exist"

    def test_dvc_yaml_has_forecasting_stages(self):
        """Verify dvc.yaml has forecasting stages."""
        import yaml

        dvc_path = PROJECT_ROOT / "dvc.yaml"

        with open(dvc_path) as f:
            dvc_config = yaml.safe_load(f)

        stages = dvc_config.get("stages", {})

        forecasting_stages = [
            "forecast_prepare_data",
            "forecast_train",
            "forecast_evaluate",
        ]

        for stage in forecasting_stages:
            assert stage in stages, f"Missing DVC stage: {stage}"


class TestMLflowIntegration:
    """Test MLflow integration for forecasting."""

    def test_config_has_mlflow_section(self):
        """Verify ForecastingConfig has MLflow configuration."""
        from src.forecasting.config import get_config

        config = get_config()

        assert hasattr(config, 'mlflow'), "Config should have mlflow attribute"
        assert config.mlflow.enabled is True, "MLflow should be enabled by default"
        assert config.mlflow.experiment_name == "forecasting-training"

    def test_params_yaml_has_mlflow_section(self):
        """Verify params.yaml has MLflow configuration for forecasting."""
        import yaml

        params_path = PROJECT_ROOT / "params.yaml"

        with open(params_path) as f:
            params = yaml.safe_load(f)

        forecasting = params.get("forecasting", {})
        mlflow_config = forecasting.get("mlflow", {})

        assert "enabled" in mlflow_config, "MLflow should have enabled flag"
        assert "experiment_name" in mlflow_config, "MLflow should have experiment_name"
        assert "tracking_uri" in mlflow_config, "MLflow should have tracking_uri"

    def test_engine_has_mlflow_methods(self):
        """Verify ForecastingEngine has MLflow methods."""
        from src.forecasting.engine import ForecastingEngine

        engine = ForecastingEngine()

        # Check MLflow-related methods exist
        assert hasattr(engine, '_log_to_mlflow'), "Should have _log_to_mlflow method"
        assert hasattr(engine, '_log_training_summary_to_mlflow'), "Should have _log_training_summary_to_mlflow"
        assert hasattr(engine, '_register_model_to_mlflow'), "Should have _register_model_to_mlflow"

    def test_mlflow_logging_method_signature(self):
        """Verify _log_to_mlflow has correct parameters."""
        from src.forecasting.engine import ForecastingEngine
        import inspect

        engine = ForecastingEngine()
        sig = inspect.signature(engine._log_to_mlflow)

        expected_params = ['model_id', 'horizon', 'wf_result', 'experiment_name', 'model_path', 'params']

        actual_params = list(sig.parameters.keys())

        # Skip 'self'
        actual_params = [p for p in actual_params if p != 'self']

        for param in expected_params:
            assert param in actual_params, f"Method should have parameter: {param}"

    def test_engine_logs_data_contract_info(self):
        """Verify engine imports data contract constants for MLflow logging."""
        from src.forecasting.engine import (
            DATA_CONTRACT_VERSION,
            DATA_CONTRACT_HASH,
            NUM_FEATURES,
            FEATURE_COLUMNS,
        )

        assert DATA_CONTRACT_VERSION is not None, "Should have DATA_CONTRACT_VERSION"
        assert DATA_CONTRACT_HASH is not None, "Should have DATA_CONTRACT_HASH"
        assert NUM_FEATURES == 19, "Should have correct NUM_FEATURES"
        assert len(FEATURE_COLUMNS) == 19, "Should have correct FEATURE_COLUMNS count"


class TestMLflowConfig:
    """Test MLflow configuration class."""

    def test_mlflow_config_defaults(self):
        """Test MLflowConfig default values."""
        from src.forecasting.config import MLflowConfig

        config = MLflowConfig()

        assert config.enabled is True
        assert config.tracking_uri == "http://localhost:5000"
        assert config.experiment_name == "forecasting-training"
        assert config.registry_enabled is True
        assert config.model_name_prefix == "forecasting"
        assert config.log_models is True
        assert config.log_artifacts is True
        assert config.log_params is True
        assert config.log_metrics is True

    def test_mlflow_config_in_forecasting_config(self):
        """Test MLflowConfig is part of ForecastingConfig."""
        from src.forecasting.config import ForecastingConfig, MLflowConfig

        config = ForecastingConfig()

        assert isinstance(config.mlflow, MLflowConfig)
        assert config.mlflow.project_tag == "usdcop-forecasting"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
