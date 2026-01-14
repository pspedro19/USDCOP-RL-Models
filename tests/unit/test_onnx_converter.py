import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import patch

# Set environment variable to trigger mock imports
os.environ['MOCK_ONNX_LIBS'] = '1'

# Now import the module
from lib.inference.onnx_converter import ONNXConverter, ONNXInferenceEngine, ConversionResult
from lib.features.contract import FEATURE_CONTRACT


class MockPolicyNetwork(nn.Module):
    """Red de política mock para tests."""
    
    def __init__(self, observation_dim: int = 15, action_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, x):
        features = self.net[:-1](x)
        action_logits = self.net[-1](features)
        value = self.value_head(features)
        return action_logits, value


class TestONNXConverter:
    """
    Tests del ONNX Converter.
    GEMINI-T1 | Plan Item: P1-14
    Coverage Target: 90%
    """

    @pytest.fixture
    def converter(self):
        return ONNXConverter(contract_version="current")

    @pytest.fixture
    def mock_model(self):
        model = MockPolicyNetwork(observation_dim=15, action_dim=3)
        model.eval()
        return model

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @patch('torch.onnx.export')
    @patch('lib.inference.onnx_converter.onnx.load')
    @patch('lib.inference.onnx_converter.onnx.save')
    @patch('lib.inference.onnx_converter.onnxoptimizer.optimize')
    def test_conversion_produces_valid_onnx(self, mock_optimize, mock_save, mock_load, mock_export, converter, mock_model, temp_dir):
        """Conversión DEBE producir archivo ONNX válido."""
        output_path = temp_dir / "test_model.onnx"
        
        result = converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=False
        )
        
        assert output_path.exists()
        assert result.onnx_path == output_path
        assert len(result.model_hash) == 64  # SHA256

    @patch('torch.onnx.export')
    @patch('lib.inference.onnx_converter.onnx.load')
    @patch('lib.inference.onnx_converter.onnx.save')
    @patch('lib.inference.onnx_converter.onnxoptimizer.optimize')
    def test_pytorch_onnx_consistency(self, mock_optimize, mock_save, mock_load, mock_export, converter, mock_model, temp_dir):
        """Output ONNX DEBE ser idéntico a PyTorch (tolerancia < 1e-6)."""
        output_path = temp_dir / "test_model.onnx"
        
        with patch('lib.inference.onnx_converter.ort.InferenceSession') as mock_session:
            mock_session.return_value.run.return_value = [mock_model(torch.randn(1, 15))[0].numpy()]
        
            result = converter.convert(
                pytorch_model=mock_model,
                output_path=str(output_path),
                validate=True,
                num_validation_samples=100
            )
        
        assert result.validation_passed, f"Max diff: {result.max_diff}"
        assert result.max_diff < 1e-6

    @patch('torch.onnx.export')
    @patch('lib.inference.onnx_converter.onnx.load')
    @patch('lib.inference.onnx_converter.onnx.save')
    @patch('lib.inference.onnx_converter.onnxoptimizer.optimize')
    def test_input_shape_matches_contract(self, mock_optimize, mock_save, mock_load, mock_export, converter, mock_model, temp_dir):
        """Input shape DEBE coincidir con Feature Contract."""
        output_path = temp_dir / "test_model.onnx"
        
        result = converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=False
        )
        
        expected_dim = FEATURE_CONTRACT.observation_dim
        assert result.input_shape == (None, expected_dim)

    @patch('torch.onnx.export')
    @patch('lib.inference.onnx_converter.onnx.load')
    @patch('lib.inference.onnx_converter.onnx.save')
    @patch('lib.inference.onnx_converter.onnxoptimizer.optimize')
    def test_metadata_saved_correctly(self, mock_optimize, mock_save, mock_load, mock_export, converter, mock_model, temp_dir):
        """Metadata DEBE incluir hash, versión, feature_order."""
        import json
        
        output_path = temp_dir / "test_model.onnx"
        result = converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=False
        )
        
        metadata_path = output_path.with_suffix(".json")
        assert metadata_path.exists()
        
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        assert metadata["contract_version"] == "current"
        assert metadata["observation_dim"] == 15
        assert metadata["model_hash"] == result.model_hash
        assert "feature_order" in metadata
        assert len(metadata["feature_order"]) == 15

    @patch('torch.onnx.export')
    @patch('lib.inference.onnx_converter.onnx.load')
    @patch('lib.inference.onnx_converter.onnx.save')
    @patch('lib.inference.onnx_converter.onnxoptimizer.optimize')
    def test_load_with_wrong_hash_fails(self, mock_optimize, mock_save, mock_load, mock_export, converter, mock_model, temp_dir):
        """Cargar modelo con hash incorrecto DEBE lanzar error."""
        output_path = temp_dir / "test_model.onnx"
        
        converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=False
        )
        
        with pytest.raises(ValueError, match="Hash mismatch"):
            ONNXConverter.load_and_verify(
                str(output_path),
                expected_hash="wrong_hash_" + "0" * 48
            )

    @patch('torch.onnx.export')
    @patch('lib.inference.onnx_converter.onnx.load')
    @patch('lib.inference.onnx_converter.onnx.save')
    @patch('lib.inference.onnx_converter.onnxoptimizer.optimize')
    def test_load_with_correct_hash_succeeds(self, mock_optimize, mock_save, mock_load, mock_export, converter, mock_model, temp_dir):
        """Cargar modelo con hash correcto DEBE funcionar."""
        output_path = temp_dir / "test_model.onnx"
        
        result = converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=False
        )
        
        session = ONNXConverter.load_and_verify(
            str(output_path),
            expected_hash=result.model_hash
        )
        
        assert session is not None


class TestONNXInferenceEngine:
    """Tests del motor de inference ONNX."""

    @pytest.fixture
    def model_path(self, tmp_path):
        """Crear modelo ONNX para tests."""
        model = MockPolicyNetwork(observation_dim=15)
        model.eval()
        
        converter = ONNXConverter(contract_version="current")
        with patch('torch.onnx.export'), patch('lib.inference.onnx_converter.onnx.load'), patch('lib.inference.onnx_converter.onnx.save'), patch('lib.inference.onnx_converter.onnxoptimizer.optimize'):
            result = converter.convert(
                pytorch_model=model,
                output_path=str(tmp_path / "test.onnx"),
                validate=False
            )
        return str(result.onnx_path)

    def test_inference_latency_under_5ms(self, model_path):
        """Latencia de inference DEBE ser < 5ms."""
        engine = ONNXInferenceEngine(model_path, warmup_runs=10)
        
        latencies = []
        for _ in range(100):
            obs = np.random.randn(15).astype(np.float32)
            _, latency = engine.predict(obs)
            latencies.append(latency)
        
        p95_latency = np.percentile(latencies, 95)
        assert p95_latency < 5.0

    def test_batch_inference(self, model_path):
        """Batch inference DEBE producir outputs correctos."""
        engine = ONNXInferenceEngine(model_path)
        
        batch_size = 32
        obs_batch = np.random.randn(batch_size, 15).astype(np.float32)
        
        actions, _ = engine.predict(obs_batch)
        
        assert actions.shape[0] == batch_size
        assert actions.shape[1] == 3

    def test_output_no_nan_inf(self, model_path):
        """Output NUNCA debe contener NaN o Inf."""
        engine = ONNXInferenceEngine(model_path)
        
        extreme_inputs = [
            np.zeros(15),
            np.ones(15) * 5.0,
            np.ones(15) * -5.0,
            np.random.randn(15) * 3
        ]
        
        for obs in extreme_inputs:
            actions, _ = engine.predict(obs.astype(np.float32))
            assert not np.isnan(actions).any()
            assert not np.isinf(actions).any()

    def test_latency_stats_accumulate(self, model_path):
        """Estadísticas de latencia DEBEN acumularse correctamente."""
        engine = ONNXInferenceEngine(model_path)
        
        for _ in range(50):
            obs = np.random.randn(15).astype(np.float32)
            engine.predict(obs)
        
        stats = engine.get_latency_stats()
        
        assert stats["count"] == 50
        assert "mean_ms" in stats
        assert "p95_ms" in stats
        assert stats["mean_ms"] > 0