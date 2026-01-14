# GEMINI TASKS - Inference, Risk & Operations
# Proyecto: USDCOP-RL-Models

**Version**: 1.0  
**Fecha**: 2026-01-11  
**Agente Asignado**: Gemini  
**Dominio de Responsabilidad**: Inference Pipeline, Risk Management, Production Operations  
**Metodología**: Spec-Driven + AI-Augmented TDD  

---

## 📋 RESUMEN EJECUTIVO

### Alcance de Responsabilidad
Gemini es responsable de construir el **pipeline de producción**, incluyendo:
- Conversión y optimización ONNX
- Circuit Breakers y protección de producción
- Drift Detection para monitoreo continuo
- Risk Engine para gestión de riesgo
- Correcciones de bugs en inference pipeline

### Contratos de Entrada (De Claude)
| Contrato ID | Artefacto | Ubicación | Status Esperado |
|-------------|-----------|-----------|-----------------|
| CTR-001 | `FeatureBuilder` class | `lib/features/builder.py` | 🔄 Pendiente |
| CTR-002 | `FEATURE_CONTRACT_V20` | `lib/features/contract.py` | 🔄 Pendiente |
| CTR-003 | `v20_norm_stats.json` | `config/v20_norm_stats.json` | 🔄 Pendiente |
| CTR-004 | `features_snapshot` schema | `migrations/` | 🔄 Pendiente |

### Contratos de Salida (Producidos por Gemini)
| Artefacto | Formato | Ubicación | Contrato ID |
|-----------|---------|-----------|-------------|
| `ONNXConverter` | Python module | `lib/inference/onnx_converter.py` | GTR-001 |
| `CircuitBreaker` | Python class | `lib/risk/circuit_breakers.py` | GTR-002 |
| `DriftDetector` | Python class | `lib/risk/drift_detection.py` | GTR-003 |
| `RiskEngine` | Python class | `lib/risk/engine.py` | GTR-004 |

### Métricas de Éxito Global
| Métrica | Target | Crítico | Medición |
|---------|--------|---------|----------|
| Latencia ONNX inference | < 5ms | < 20ms | Benchmark test |
| Circuit Breaker response | < 1ms | < 5ms | Unit test |
| Drift detection accuracy | > 95% | > 85% | Integration test |
| Test coverage `lib/risk/` | ≥ 90% | ≥ 80% | pytest-cov |

### Protocolo de Coordinación con Claude
```
┌─────────────────────────────────────────────────────────────┐
│                    FLUJO DE DEPENDENCIAS                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  CLAUDE                           GEMINI                     │
│  ───────                          ───────                    │
│                                                              │
│  CTR-001 (FeatureBuilder) ──────► GEMINI-T1 (ONNX)          │
│  CTR-002 (Contract V20)   ──────► GEMINI-T2 (Circuit)       │
│  CTR-003 (norm_stats)     ──────► GEMINI-T3 (Drift)         │
│  CTR-004 (feat_snapshot)  ──────► GEMINI-T4 (Risk Engine)   │
│                                                              │
│  CLAUDE-T7 (Parity) ◄────────────► GEMINI-T1 (validation)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**IMPORTANTE**: Gemini debe esperar confirmación de que los contratos CTR-001 a CTR-004 están completos antes de comenzar tareas que los consumen.

---

## 🎯 TAREA 1: ONNX Conversion Pipeline (P1-14)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T1 |
| **Plan Item** | P1-14 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Consume Contratos** | CTR-001, CTR-002, CTR-003 |
| **Produce Contrato** | GTR-001 |
| **Bloquea** | GEMINI-T2, GEMINI-T3, GEMINI-T4 |

### Objetivo
Implementar pipeline de conversión PyTorch → ONNX con validación automática de consistencia. El modelo ONNX será consumido por inference API y debe producir outputs idénticos al modelo PyTorch original.

### Especificación Funcional

#### Interface del Contrato GTR-001

```python
# lib/inference/onnx_converter.py
"""
ONNX Conversion Pipeline
Contrato ID: GTR-001

Responsabilidades:
1. Convertir modelos PyTorch (PPO) a ONNX
2. Validar consistencia numérica post-conversión
3. Optimizar modelo para inference
4. Registrar metadata (hash, versión, fecha)

Invariantes:
- Output ONNX idéntico a PyTorch (tolerancia < 1e-6)
- Modelo incluye hash verificable
- Input shape = (batch, 15) según CTR-002
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
import onnx
import onnxruntime as ort
from lib.features.contract import FEATURE_CONTRACT_V20


@dataclass
class ConversionResult:
    """Resultado de conversión ONNX."""
    onnx_path: Path
    model_hash: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    validation_passed: bool
    max_diff: float
    metadata: Dict


class ONNXConverter:
    """
    Conversor PyTorch → ONNX con validación.
    
    Ejemplo:
        converter = ONNXConverter(contract_version="v20")
        result = converter.convert(
            pytorch_model=ppo_model.policy,
            output_path="models/v20_policy.onnx"
        )
        assert result.validation_passed
    """
    
    def __init__(self, contract_version: str = "v20"):
        """
        Args:
            contract_version: Versión del Feature Contract a usar
        """
        self.contract = FEATURE_CONTRACT_V20 if contract_version == "v20" else None
        if self.contract is None:
            raise ValueError(f"Contract version {contract_version} not supported")
        
        self.observation_dim = self.contract.observation_dim
    
    def convert(
        self,
        pytorch_model: torch.nn.Module,
        output_path: str,
        opset_version: int = 17,
        optimize: bool = True,
        validate: bool = True,
        num_validation_samples: int = 1000
    ) -> ConversionResult:
        """
        Convierte modelo PyTorch a ONNX.
        
        Args:
            pytorch_model: Modelo PyTorch (policy network)
            output_path: Ruta para guardar .onnx
            opset_version: ONNX opset version (default 17)
            optimize: Aplicar optimizaciones ONNX
            validate: Validar consistencia numérica
            num_validation_samples: Samples para validación
        
        Returns:
            ConversionResult con metadata y status de validación
        
        Raises:
            ValidationError: Si diferencia > tolerancia
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Preparar modelo
        pytorch_model.eval()
        
        # Input dummy para tracing
        batch_size = 1
        dummy_input = torch.randn(batch_size, self.observation_dim)
        
        # Export a ONNX
        torch.onnx.export(
            pytorch_model,
            dummy_input,
            str(output_path),
            input_names=["observation"],
            output_names=["action", "value"],
            dynamic_axes={
                "observation": {0: "batch_size"},
                "action": {0: "batch_size"},
                "value": {0: "batch_size"}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        # Optimizar si solicitado
        if optimize:
            self._optimize_model(output_path)
        
        # Calcular hash
        model_hash = self._calculate_hash(output_path)
        
        # Validar si solicitado
        validation_passed = True
        max_diff = 0.0
        if validate:
            validation_passed, max_diff = self._validate_consistency(
                pytorch_model, output_path, num_validation_samples
            )
        
        # Metadata
        metadata = {
            "contract_version": self.contract.version,
            "observation_dim": self.observation_dim,
            "feature_order": list(self.contract.feature_order),
            "model_hash": model_hash,
            "opset_version": opset_version,
            "optimized": optimize,
            "created_at": self._get_timestamp()
        }
        
        # Guardar metadata
        metadata_path = output_path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return ConversionResult(
            onnx_path=output_path,
            model_hash=model_hash,
            input_shape=(None, self.observation_dim),
            output_shape=(None, 3),  # action_dim = 3 (buy, hold, sell)
            validation_passed=validation_passed,
            max_diff=max_diff,
            metadata=metadata
        )
    
    def _optimize_model(self, model_path: Path) -> None:
        """Aplica optimizaciones ONNX."""
        import onnxoptimizer
        
        model = onnx.load(str(model_path))
        
        # Optimizaciones estándar
        passes = [
            "eliminate_identity",
            "eliminate_nop_transpose",
            "fuse_consecutive_transposes",
            "fuse_bn_into_conv",
            "fuse_add_bias_into_conv"
        ]
        
        optimized = onnxoptimizer.optimize(model, passes)
        onnx.save(optimized, str(model_path))
    
    def _calculate_hash(self, model_path: Path) -> str:
        """Calcula SHA256 del modelo."""
        sha256 = hashlib.sha256()
        with open(model_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def _validate_consistency(
        self,
        pytorch_model: torch.nn.Module,
        onnx_path: Path,
        num_samples: int,
        tolerance: float = 1e-6
    ) -> Tuple[bool, float]:
        """
        Valida que outputs ONNX coincidan con PyTorch.
        
        Returns:
            (passed, max_difference)
        """
        # Cargar ONNX
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        
        max_diff = 0.0
        np.random.seed(42)
        
        for _ in range(num_samples):
            # Input aleatorio
            obs = np.random.randn(1, self.observation_dim).astype(np.float32)
            
            # PyTorch inference
            with torch.no_grad():
                pt_output = pytorch_model(torch.from_numpy(obs))
                if isinstance(pt_output, tuple):
                    pt_action = pt_output[0].numpy()
                else:
                    pt_action = pt_output.numpy()
            
            # ONNX inference
            onnx_output = session.run(None, {"observation": obs})
            onnx_action = onnx_output[0]
            
            # Comparar
            diff = np.abs(pt_action - onnx_action).max()
            max_diff = max(max_diff, diff)
            
            if diff > tolerance:
                return False, max_diff
        
        return True, max_diff
    
    def _get_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    @staticmethod
    def load_and_verify(
        onnx_path: str,
        expected_hash: Optional[str] = None
    ) -> ort.InferenceSession:
        """
        Carga modelo ONNX con verificación de hash.
        
        Args:
            onnx_path: Ruta al modelo ONNX
            expected_hash: Hash esperado (si se proporciona)
        
        Returns:
            ONNX Runtime session
        
        Raises:
            HashMismatchError: Si hash no coincide
        """
        onnx_path = Path(onnx_path)
        
        # Verificar hash si se proporciona
        if expected_hash:
            sha256 = hashlib.sha256()
            with open(onnx_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            actual_hash = sha256.hexdigest()
            
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Hash mismatch: expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )
        
        # Cargar modelo
        return ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
    
    @staticmethod
    def get_model_input_names(onnx_path: str) -> list:
        """Retorna nombres de inputs del modelo ONNX."""
        model = onnx.load(onnx_path)
        return [inp.name for inp in model.graph.input]


class ONNXInferenceEngine:
    """
    Motor de inference ONNX optimizado para producción.
    
    Características:
    - Batching automático
    - Warm-up de sesión
    - Métricas de latencia
    """
    
    def __init__(
        self,
        model_path: str,
        contract_version: str = "v20",
        warmup_runs: int = 10
    ):
        self.contract = FEATURE_CONTRACT_V20
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )
        
        # Warm-up
        dummy = np.zeros((1, self.contract.observation_dim), dtype=np.float32)
        for _ in range(warmup_runs):
            self.session.run(None, {"observation": dummy})
        
        self._latency_samples = []
    
    def predict(self, observation: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Ejecuta inference con medición de latencia.
        
        Args:
            observation: Array de shape (15,) o (batch, 15)
        
        Returns:
            (action_logits, latency_ms)
        """
        import time
        
        # Asegurar shape correcto
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        observation = observation.astype(np.float32)
        
        # Inference con timing
        start = time.perf_counter()
        outputs = self.session.run(None, {"observation": observation})
        latency_ms = (time.perf_counter() - start) * 1000
        
        self._latency_samples.append(latency_ms)
        
        return outputs[0], latency_ms
    
    def get_latency_stats(self) -> Dict:
        """Retorna estadísticas de latencia."""
        if not self._latency_samples:
            return {"count": 0}
        
        samples = np.array(self._latency_samples)
        return {
            "count": len(samples),
            "mean_ms": float(samples.mean()),
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "p99_ms": float(np.percentile(samples, 99)),
            "max_ms": float(samples.max())
        }
```

### Tests (TDD - Escribir PRIMERO)

```python
# tests/unit/test_onnx_converter.py
import pytest
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import tempfile
from lib.inference.onnx_converter import ONNXConverter, ONNXInferenceEngine, ConversionResult
from lib.features.contract import FEATURE_CONTRACT_V20


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
        features = self.net[:-1](x)  # All but last layer
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
        return ONNXConverter(contract_version="v20")

    @pytest.fixture
    def mock_model(self):
        model = MockPolicyNetwork(observation_dim=15, action_dim=3)
        model.eval()
        return model

    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Conversión exitosa
    # Criterio: Debe producir archivo .onnx válido
    # ══════════════════════════════════════════════════════════════
    def test_conversion_produces_valid_onnx(self, converter, mock_model, temp_dir):
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

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Validación de consistencia
    # Criterio: Output ONNX idéntico a PyTorch (< 1e-6)
    # ══════════════════════════════════════════════════════════════
    def test_pytorch_onnx_consistency(self, converter, mock_model, temp_dir):
        """Output ONNX DEBE ser idéntico a PyTorch (tolerancia < 1e-6)."""
        output_path = temp_dir / "test_model.onnx"
        
        result = converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=True,
            num_validation_samples=100
        )
        
        assert result.validation_passed, f"Max diff: {result.max_diff}"
        assert result.max_diff < 1e-6

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Input shape correcto
    # Criterio: Input shape DEBE ser (batch, 15) según CTR-002
    # ══════════════════════════════════════════════════════════════
    def test_input_shape_matches_contract(self, converter, mock_model, temp_dir):
        """Input shape DEBE coincidir con Feature Contract V20."""
        output_path = temp_dir / "test_model.onnx"
        
        result = converter.convert(
            pytorch_model=mock_model,
            output_path=str(output_path),
            validate=False
        )
        
        expected_dim = FEATURE_CONTRACT_V20.observation_dim
        assert result.input_shape == (None, expected_dim)

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Metadata guardada correctamente
    # Criterio: Metadata JSON con hash, versión, timestamp
    # ══════════════════════════════════════════════════════════════
    def test_metadata_saved_correctly(self, converter, mock_model, temp_dir):
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
        
        assert metadata["contract_version"] == "v20"
        assert metadata["observation_dim"] == 15
        assert metadata["model_hash"] == result.model_hash
        assert "feature_order" in metadata
        assert len(metadata["feature_order"]) == 15

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Hash verification
    # Criterio: Load con hash incorrecto DEBE fallar
    # ══════════════════════════════════════════════════════════════
    def test_load_with_wrong_hash_fails(self, converter, mock_model, temp_dir):
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

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Hash correcto funciona
    # Criterio: Load con hash correcto DEBE funcionar
    # ══════════════════════════════════════════════════════════════
    def test_load_with_correct_hash_succeeds(self, converter, mock_model, temp_dir):
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
        
        converter = ONNXConverter(contract_version="v20")
        result = converter.convert(
            pytorch_model=model,
            output_path=str(tmp_path / "test.onnx"),
            validate=False
        )
        return str(result.onnx_path)

    # ══════════════════════════════════════════════════════════════
    # TEST 7: Latencia < 5ms
    # Criterio: Inference single DEBE ser < 5ms
    # ══════════════════════════════════════════════════════════════
    def test_inference_latency_under_5ms(self, model_path):
        """Latencia de inference DEBE ser < 5ms."""
        engine = ONNXInferenceEngine(model_path, warmup_runs=10)
        
        # Run 100 inferences
        latencies = []
        for _ in range(100):
            obs = np.random.randn(15).astype(np.float32)
            _, latency = engine.predict(obs)
            latencies.append(latency)
        
        p95_latency = np.percentile(latencies, 95)
        assert p95_latency < 5.0, f"P95 latency: {p95_latency:.2f}ms > 5ms"

    # ══════════════════════════════════════════════════════════════
    # TEST 8: Batch inference funciona
    # Criterio: Batch de N observaciones produce N outputs
    # ══════════════════════════════════════════════════════════════
    def test_batch_inference(self, model_path):
        """Batch inference DEBE producir outputs correctos."""
        engine = ONNXInferenceEngine(model_path)
        
        batch_size = 32
        obs_batch = np.random.randn(batch_size, 15).astype(np.float32)
        
        actions, _ = engine.predict(obs_batch)
        
        assert actions.shape[0] == batch_size
        assert actions.shape[1] == 3  # action_dim

    # ══════════════════════════════════════════════════════════════
    # TEST 9: Output no contiene NaN/Inf
    # Criterio: Output NUNCA debe contener NaN o Inf
    # ══════════════════════════════════════════════════════════════
    def test_output_no_nan_inf(self, model_path):
        """Output NUNCA debe contener NaN o Inf."""
        engine = ONNXInferenceEngine(model_path)
        
        # Test con inputs extremos
        extreme_inputs = [
            np.zeros(15),
            np.ones(15) * 5.0,
            np.ones(15) * -5.0,
            np.random.randn(15) * 3
        ]
        
        for obs in extreme_inputs:
            actions, _ = engine.predict(obs.astype(np.float32))
            assert not np.isnan(actions).any(), f"NaN en output para input: {obs[:3]}..."
            assert not np.isinf(actions).any(), f"Inf en output para input: {obs[:3]}..."

    # ══════════════════════════════════════════════════════════════
    # TEST 10: Latency stats correctas
    # Criterio: Stats deben acumular correctamente
    # ══════════════════════════════════════════════════════════════
    def test_latency_stats_accumulate(self, model_path):
        """Estadísticas de latencia DEBEN acumularse correctamente."""
        engine = ONNXInferenceEngine(model_path)
        
        # Run inferences
        for _ in range(50):
            obs = np.random.randn(15).astype(np.float32)
            engine.predict(obs)
        
        stats = engine.get_latency_stats()
        
        assert stats["count"] == 50
        assert "mean_ms" in stats
        assert "p95_ms" in stats
        assert stats["mean_ms"] > 0
```

### Criterios de Aceptación

| # | Criterio | Verificación | Status |
|---|----------|--------------|--------|
| 1 | Produce ONNX válido | Unit test | ⬜ |
| 2 | Consistencia PyTorch-ONNX < 1e-6 | Unit test | ⬜ |
| 3 | Input shape = (batch, 15) | Unit test | ⬜ |
| 4 | Metadata con hash | Unit test | ⬜ |
| 5 | Hash verification funciona | Unit test | ⬜ |
| 6 | Latencia inference < 5ms (P95) | Benchmark | ⬜ |
| 7 | Batch inference funciona | Unit test | ⬜ |
| 8 | Sin NaN/Inf en output | Unit test | ⬜ |
| 9 | Coverage ≥ 90% | pytest-cov | ⬜ |

---

## 🎯 TAREA 2: Circuit Breakers (P1-15)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T2 |
| **Plan Item** | P1-15 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Consume Contratos** | CTR-002, GTR-001 |
| **Produce Contrato** | GTR-002 |

### Objetivo
Implementar sistema de Circuit Breakers para proteger producción contra:
- Anomalías en inputs (features fuera de rango)
- Errores de modelo (outputs inválidos)
- Degradación de latencia
- Pérdidas consecutivas

### Especificación Funcional

```python
# lib/risk/circuit_breakers.py
"""
Circuit Breaker System para producción.
Contrato ID: GTR-002

Implementa patrón Circuit Breaker con múltiples triggers:
1. FeatureCircuitBreaker: Detecta features anómalas
2. LatencyCircuitBreaker: Detecta degradación de performance
3. LossCircuitBreaker: Detecta rachas de pérdidas
4. MasterCircuitBreaker: Orquesta todos los breakers

Estados:
- CLOSED: Operación normal
- OPEN: Trading bloqueado
- HALF_OPEN: Testing recovery
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
import numpy as np
from lib.features.contract import FEATURE_CONTRACT_V20


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuración de Circuit Breaker."""
    # Feature breaker
    feature_clip_range: tuple = (-5.0, 5.0)
    max_clipped_features: int = 3  # Máximo features en límite
    
    # Latency breaker
    latency_threshold_ms: float = 50.0
    latency_window_size: int = 100
    latency_breach_pct: float = 0.1  # 10% sobre threshold = OPEN
    
    # Loss breaker
    max_consecutive_losses: int = 5
    max_drawdown_pct: float = 0.05  # 5%
    
    # Recovery
    recovery_timeout_minutes: int = 15
    half_open_success_threshold: int = 3


@dataclass
class CircuitBreakerState:
    """Estado actual del circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    opened_at: Optional[datetime] = None
    open_reason: Optional[str] = None
    half_open_successes: int = 0
    trip_count: int = 0
    last_trip_time: Optional[datetime] = None


class FeatureCircuitBreaker:
    """
    Circuit breaker para features anómalas.
    
    Triggers:
    - Feature individual fuera de [-5, 5]
    - Más de N features en límites de clipping
    - NaN o Inf detectado
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState()
        self.contract = FEATURE_CONTRACT_V20
    
    def check(self, observation: np.ndarray) -> tuple[bool, Optional[str]]:
        """
        Verifica si observation es válida.
        
        Returns:
            (is_valid, reason_if_invalid)
        """
        # Check NaN/Inf
        if np.isnan(observation).any():
            return False, "NaN detectado en observation"
        if np.isinf(observation).any():
            return False, "Inf detectado en observation"
        
        # Check shape
        if observation.shape != (self.contract.observation_dim,):
            return False, f"Shape incorrecto: {observation.shape}"
        
        # Check clipping (primeras 13 features normalizadas)
        clip_min, clip_max = self.config.feature_clip_range
        normalized_features = observation[:13]
        
        at_min = (normalized_features <= clip_min).sum()
        at_max = (normalized_features >= clip_max).sum()
        clipped_count = at_min + at_max
        
        if clipped_count > self.config.max_clipped_features:
            return False, f"{clipped_count} features en límites de clipping"
        
        return True, None
    
    def trip(self, reason: str) -> None:
        """Abre el circuit breaker."""
        self.state.state = CircuitState.OPEN
        self.state.opened_at = datetime.utcnow()
        self.state.open_reason = reason
        self.state.trip_count += 1
        self.state.last_trip_time = datetime.utcnow()
    
    def attempt_recovery(self) -> bool:
        """Intenta recuperar (transición a HALF_OPEN)."""
        if self.state.state != CircuitState.OPEN:
            return False
        
        elapsed = datetime.utcnow() - self.state.opened_at
        if elapsed > timedelta(minutes=self.config.recovery_timeout_minutes):
            self.state.state = CircuitState.HALF_OPEN
            return True
        
        return False
    
    def record_success(self) -> None:
        """Registra operación exitosa (para HALF_OPEN)."""
        if self.state.state == CircuitState.HALF_OPEN:
            self.state.half_open_successes += 1
            if self.state.half_open_successes >= self.config.half_open_success_threshold:
                self.state.state = CircuitState.CLOSED
                self.state.half_open_successes = 0


class LatencyCircuitBreaker:
    """
    Circuit breaker para degradación de latencia.
    
    Triggers:
    - P95 latencia > threshold durante ventana
    - Más del X% de requests sobre threshold
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState()
        self._latency_buffer: List[float] = []
    
    def record_latency(self, latency_ms: float) -> tuple[bool, Optional[str]]:
        """
        Registra latencia y verifica circuit.
        
        Returns:
            (is_healthy, reason_if_unhealthy)
        """
        self._latency_buffer.append(latency_ms)
        
        # Mantener ventana
        if len(self._latency_buffer) > self.config.latency_window_size:
            self._latency_buffer.pop(0)
        
        # Verificar solo si tenemos suficientes samples
        if len(self._latency_buffer) < self.config.latency_window_size // 2:
            return True, None
        
        # Calcular métricas
        samples = np.array(self._latency_buffer)
        breach_count = (samples > self.config.latency_threshold_ms).sum()
        breach_pct = breach_count / len(samples)
        
        if breach_pct > self.config.latency_breach_pct:
            return False, f"Latencia degradada: {breach_pct:.1%} > {self.config.latency_breach_pct:.1%}"
        
        return True, None
    
    def trip(self, reason: str) -> None:
        self.state.state = CircuitState.OPEN
        self.state.opened_at = datetime.utcnow()
        self.state.open_reason = reason
        self.state.trip_count += 1


class LossCircuitBreaker:
    """
    Circuit breaker para rachas de pérdidas.
    
    Triggers:
    - N pérdidas consecutivas
    - Drawdown > X%
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState()
        self._consecutive_losses: int = 0
        self._peak_equity: float = 0.0
        self._current_equity: float = 0.0
    
    def record_trade(self, pnl: float, equity: float) -> tuple[bool, Optional[str]]:
        """
        Registra resultado de trade.
        
        Returns:
            (is_healthy, reason_if_unhealthy)
        """
        # Actualizar equity
        self._current_equity = equity
        self._peak_equity = max(self._peak_equity, equity)
        
        # Contar pérdidas
        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0
        
        # Check consecutive losses
        if self._consecutive_losses >= self.config.max_consecutive_losses:
            return False, f"{self._consecutive_losses} pérdidas consecutivas"
        
        # Check drawdown
        if self._peak_equity > 0:
            drawdown = (self._peak_equity - self._current_equity) / self._peak_equity
            if drawdown > self.config.max_drawdown_pct:
                return False, f"Drawdown {drawdown:.2%} > {self.config.max_drawdown_pct:.2%}"
        
        return True, None
    
    def trip(self, reason: str) -> None:
        self.state.state = CircuitState.OPEN
        self.state.opened_at = datetime.utcnow()
        self.state.open_reason = reason
        self.state.trip_count += 1
    
    def reset_streak(self) -> None:
        """Reset contador de pérdidas consecutivas."""
        self._consecutive_losses = 0


class MasterCircuitBreaker:
    """
    Orquestador de todos los circuit breakers.
    
    Agrega estado de todos los sub-breakers y toma decisión final.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        
        self.feature_breaker = FeatureCircuitBreaker(self.config)
        self.latency_breaker = LatencyCircuitBreaker(self.config)
        self.loss_breaker = LossCircuitBreaker(self.config)
        
        self._callbacks: List[Callable[[str], None]] = []
    
    def register_callback(self, callback: Callable[[str], None]) -> None:
        """Registra callback para notificaciones de trip."""
        self._callbacks.append(callback)
    
    def _notify_trip(self, reason: str) -> None:
        """Notifica a todos los callbacks."""
        for callback in self._callbacks:
            try:
                callback(reason)
            except Exception:
                pass  # No interrumpir por errores en callbacks
    
    def can_trade(self) -> tuple[bool, Optional[str]]:
        """
        Verifica si se puede tradear.
        
        Returns:
            (can_trade, reason_if_blocked)
        """
        # Check all breakers
        for breaker in [self.feature_breaker, self.latency_breaker, self.loss_breaker]:
            if breaker.state.state == CircuitState.OPEN:
                return False, f"Circuit OPEN: {breaker.state.open_reason}"
        
        return True, None
    
    def check_observation(self, observation: np.ndarray) -> tuple[bool, Optional[str]]:
        """Verifica observation y actualiza state."""
        is_valid, reason = self.feature_breaker.check(observation)
        
        if not is_valid:
            self.feature_breaker.trip(reason)
            self._notify_trip(f"Feature breaker: {reason}")
        
        return is_valid, reason
    
    def record_inference(self, latency_ms: float) -> tuple[bool, Optional[str]]:
        """Registra latencia de inference."""
        is_healthy, reason = self.latency_breaker.record_latency(latency_ms)
        
        if not is_healthy:
            self.latency_breaker.trip(reason)
            self._notify_trip(f"Latency breaker: {reason}")
        
        return is_healthy, reason
    
    def record_trade_result(self, pnl: float, equity: float) -> tuple[bool, Optional[str]]:
        """Registra resultado de trade."""
        is_healthy, reason = self.loss_breaker.record_trade(pnl, equity)
        
        if not is_healthy:
            self.loss_breaker.trip(reason)
            self._notify_trip(f"Loss breaker: {reason}")
        
        return is_healthy, reason
    
    def get_status(self) -> Dict:
        """Retorna estado de todos los breakers."""
        return {
            "can_trade": self.can_trade()[0],
            "feature_breaker": {
                "state": self.feature_breaker.state.state.value,
                "trip_count": self.feature_breaker.state.trip_count
            },
            "latency_breaker": {
                "state": self.latency_breaker.state.state.value,
                "trip_count": self.latency_breaker.state.trip_count
            },
            "loss_breaker": {
                "state": self.loss_breaker.state.state.value,
                "trip_count": self.loss_breaker.state.trip_count,
                "consecutive_losses": self.loss_breaker._consecutive_losses
            }
        }
    
    def attempt_recovery_all(self) -> None:
        """Intenta recovery en todos los breakers."""
        self.feature_breaker.attempt_recovery()
        # Latency y Loss tienen diferentes políticas de recovery
```

### Tests (TDD)

```python
# tests/unit/test_circuit_breakers.py
import pytest
import numpy as np
from datetime import datetime, timedelta
from lib.risk.circuit_breakers import (
    CircuitBreakerConfig,
    CircuitState,
    FeatureCircuitBreaker,
    LatencyCircuitBreaker,
    LossCircuitBreaker,
    MasterCircuitBreaker
)


class TestFeatureCircuitBreaker:
    """Tests para FeatureCircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        config = CircuitBreakerConfig(max_clipped_features=3)
        return FeatureCircuitBreaker(config)

    def test_valid_observation_passes(self, breaker):
        """Observation válida DEBE pasar."""
        obs = np.random.randn(15).astype(np.float32)
        obs = np.clip(obs, -4.0, 4.0)  # Dentro de rango
        
        is_valid, reason = breaker.check(obs)
        
        assert is_valid
        assert reason is None

    def test_nan_triggers_trip(self, breaker):
        """NaN en observation DEBE disparar breaker."""
        obs = np.random.randn(15).astype(np.float32)
        obs[5] = np.nan
        
        is_valid, reason = breaker.check(obs)
        
        assert not is_valid
        assert "NaN" in reason

    def test_inf_triggers_trip(self, breaker):
        """Inf en observation DEBE disparar breaker."""
        obs = np.random.randn(15).astype(np.float32)
        obs[5] = np.inf
        
        is_valid, reason = breaker.check(obs)
        
        assert not is_valid
        assert "Inf" in reason

    def test_too_many_clipped_triggers_trip(self, breaker):
        """Muchas features en límites DEBE disparar breaker."""
        obs = np.ones(15, dtype=np.float32) * 5.0  # Todas en límite
        
        is_valid, reason = breaker.check(obs)
        
        assert not is_valid
        assert "clipping" in reason

    def test_wrong_shape_triggers_trip(self, breaker):
        """Shape incorrecto DEBE disparar breaker."""
        obs = np.random.randn(10).astype(np.float32)  # Wrong shape
        
        is_valid, reason = breaker.check(obs)
        
        assert not is_valid
        assert "Shape" in reason


class TestLatencyCircuitBreaker:
    """Tests para LatencyCircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        config = CircuitBreakerConfig(
            latency_threshold_ms=50.0,
            latency_window_size=20,
            latency_breach_pct=0.1
        )
        return LatencyCircuitBreaker(config)

    def test_normal_latency_passes(self, breaker):
        """Latencia normal DEBE pasar."""
        for _ in range(20):
            is_healthy, _ = breaker.record_latency(10.0)
        
        assert is_healthy

    def test_high_latency_triggers_trip(self, breaker):
        """Latencia alta sostenida DEBE disparar breaker."""
        # Fill buffer with normal latencies
        for _ in range(15):
            breaker.record_latency(10.0)
        
        # Add high latencies (>10% of window)
        for _ in range(5):
            is_healthy, reason = breaker.record_latency(100.0)
        
        assert not is_healthy
        assert "degradada" in reason


class TestLossCircuitBreaker:
    """Tests para LossCircuitBreaker."""

    @pytest.fixture
    def breaker(self):
        config = CircuitBreakerConfig(
            max_consecutive_losses=3,
            max_drawdown_pct=0.1
        )
        return LossCircuitBreaker(config)

    def test_wins_dont_trigger(self, breaker):
        """Ganancias NO deben disparar breaker."""
        equity = 10000
        for _ in range(10):
            equity += 100
            is_healthy, _ = breaker.record_trade(pnl=100, equity=equity)
        
        assert is_healthy

    def test_consecutive_losses_trigger(self, breaker):
        """Pérdidas consecutivas DEBEN disparar breaker."""
        equity = 10000
        for i in range(4):
            equity -= 100
            is_healthy, reason = breaker.record_trade(pnl=-100, equity=equity)
        
        assert not is_healthy
        assert "consecutivas" in reason

    def test_drawdown_triggers(self, breaker):
        """Drawdown excesivo DEBE disparar breaker."""
        breaker.record_trade(pnl=0, equity=10000)  # Set peak
        
        is_healthy, reason = breaker.record_trade(pnl=-1500, equity=8500)
        
        assert not is_healthy
        assert "Drawdown" in reason


class TestMasterCircuitBreaker:
    """Tests para MasterCircuitBreaker."""

    @pytest.fixture
    def master(self):
        return MasterCircuitBreaker()

    def test_initial_state_allows_trading(self, master):
        """Estado inicial DEBE permitir trading."""
        can_trade, _ = master.can_trade()
        assert can_trade

    def test_any_breaker_open_blocks_trading(self, master):
        """Cualquier breaker abierto DEBE bloquear trading."""
        # Trip feature breaker
        bad_obs = np.ones(15) * np.nan
        master.check_observation(bad_obs)
        
        can_trade, reason = master.can_trade()
        
        assert not can_trade
        assert "OPEN" in reason

    def test_callback_called_on_trip(self, master):
        """Callback DEBE ser llamado cuando breaker se dispara."""
        trip_reasons = []
        master.register_callback(lambda r: trip_reasons.append(r))
        
        bad_obs = np.ones(15) * np.nan
        master.check_observation(bad_obs)
        
        assert len(trip_reasons) == 1
        assert "Feature" in trip_reasons[0]

    def test_status_returns_all_breakers(self, master):
        """get_status DEBE retornar estado de todos los breakers."""
        status = master.get_status()
        
        assert "can_trade" in status
        assert "feature_breaker" in status
        assert "latency_breaker" in status
        assert "loss_breaker" in status


class TestCircuitBreakerPerformance:
    """Tests de performance para circuit breakers."""

    def test_check_latency_under_1ms(self, benchmark):
        """check() DEBE ejecutarse en < 1ms."""
        master = MasterCircuitBreaker()
        obs = np.random.randn(15).astype(np.float32)
        
        result = benchmark(master.check_observation, obs)
        
        assert benchmark.stats.stats.mean < 0.001  # < 1ms
```

### Criterios de Aceptación

| # | Criterio | Verificación | Status |
|---|----------|--------------|--------|
| 1 | NaN/Inf dispara breaker | Unit test | ⬜ |
| 2 | Clipping excesivo dispara | Unit test | ⬜ |
| 3 | Latencia alta dispara | Unit test | ⬜ |
| 4 | Pérdidas consecutivas disparan | Unit test | ⬜ |
| 5 | Drawdown dispara | Unit test | ⬜ |
| 6 | Master coordina correctamente | Unit test | ⬜ |
| 7 | Callbacks funcionan | Unit test | ⬜ |
| 8 | Latencia check < 1ms | Benchmark | ⬜ |
| 9 | Coverage ≥ 90% | pytest-cov | ⬜ |

---

## 🎯 TAREA 3: Drift Detection (P1-16)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T3 |
| **Plan Item** | P1-16 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Consume Contratos** | CTR-002, CTR-003, CTR-004 |
| **Produce Contrato** | GTR-003 |

### Objetivo
Detectar drift en features para identificar cuando el modelo necesita reentrenamiento o cuando los datos de entrada han cambiado significativamente.

### Especificación Funcional

```python
# lib/risk/drift_detection.py
"""
Feature Drift Detection System.
Contrato ID: GTR-003

Detecta:
1. Covariate Drift: Cambio en distribución de features
2. Concept Drift: Cambio en relación feature-target
3. Label Drift: Cambio en distribución de rewards

Métodos:
- KS Test (Kolmogorov-Smirnov)
- PSI (Population Stability Index)
- ADWIN (Adaptive Windowing)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
from scipy import stats
from lib.features.contract import FEATURE_CONTRACT_V20


@dataclass
class DriftConfig:
    """Configuración de drift detection."""
    reference_window_size: int = 1000  # Barras de referencia
    detection_window_size: int = 100   # Barras para detección
    ks_threshold: float = 0.1          # p-value threshold para KS test
    psi_threshold: float = 0.2         # PSI > 0.2 = drift significativo
    alert_cooldown_bars: int = 50      # Barras entre alertas


@dataclass
class DriftAlert:
    """Alerta de drift detectado."""
    timestamp: str
    feature_name: str
    drift_type: str  # "covariate", "concept", "label"
    metric_name: str  # "ks", "psi", "adwin"
    metric_value: float
    threshold: float
    severity: str  # "warning", "critical"


class FeatureDriftDetector:
    """
    Detector de drift para features individuales.
    
    Usa KS test y PSI para detectar cambios en distribución.
    """
    
    def __init__(
        self,
        feature_name: str,
        config: DriftConfig,
        reference_data: Optional[np.ndarray] = None
    ):
        self.feature_name = feature_name
        self.config = config
        
        # Reference distribution
        self._reference = deque(maxlen=config.reference_window_size)
        if reference_data is not None:
            self._reference.extend(reference_data)
        
        # Detection window
        self._current = deque(maxlen=config.detection_window_size)
        
        # Alert tracking
        self._last_alert_bar: int = -config.alert_cooldown_bars
        self._alert_count: int = 0
    
    def update(self, value: float, bar_idx: int) -> Optional[DriftAlert]:
        """
        Actualiza con nuevo valor y detecta drift.
        
        Returns:
            DriftAlert si se detecta drift, None otherwise
        """
        self._current.append(value)
        
        # También actualizar referencia (sliding window)
        if len(self._reference) < self.config.reference_window_size:
            self._reference.append(value)
            return None
        
        # Need enough data for detection
        if len(self._current) < self.config.detection_window_size // 2:
            return None
        
        # Check cooldown
        if bar_idx - self._last_alert_bar < self.config.alert_cooldown_bars:
            return None
        
        # Run tests
        alert = self._detect_drift(bar_idx)
        
        if alert:
            self._last_alert_bar = bar_idx
            self._alert_count += 1
        
        return alert
    
    def _detect_drift(self, bar_idx: int) -> Optional[DriftAlert]:
        """Ejecuta tests de drift."""
        reference = np.array(self._reference)
        current = np.array(self._current)
        
        # KS Test
        ks_stat, ks_pvalue = stats.ks_2samp(reference, current)
        
        if ks_pvalue < self.config.ks_threshold:
            return DriftAlert(
                timestamp=self._get_timestamp(),
                feature_name=self.feature_name,
                drift_type="covariate",
                metric_name="ks_test",
                metric_value=ks_pvalue,
                threshold=self.config.ks_threshold,
                severity="critical" if ks_pvalue < 0.01 else "warning"
            )
        
        # PSI
        psi = self._calculate_psi(reference, current)
        
        if psi > self.config.psi_threshold:
            return DriftAlert(
                timestamp=self._get_timestamp(),
                feature_name=self.feature_name,
                drift_type="covariate",
                metric_name="psi",
                metric_value=psi,
                threshold=self.config.psi_threshold,
                severity="critical" if psi > 0.25 else "warning"
            )
        
        return None
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """Calcula Population Stability Index."""
        # Create bins from reference
        _, bin_edges = np.histogram(reference, bins=bins)
        
        # Count in each bin
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages (avoid division by zero)
        ref_pct = (ref_counts + 1) / (len(reference) + bins)
        cur_pct = (cur_counts + 1) / (len(current) + bins)
        
        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)
    
    def _get_timestamp(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas del detector."""
        return {
            "feature_name": self.feature_name,
            "reference_size": len(self._reference),
            "current_size": len(self._current),
            "alert_count": self._alert_count,
            "reference_mean": float(np.mean(self._reference)) if self._reference else None,
            "current_mean": float(np.mean(self._current)) if self._current else None
        }


class DriftMonitor:
    """
    Monitor de drift para todas las features del contrato.
    
    Coordina múltiples FeatureDriftDetectors y agrega alertas.
    """
    
    def __init__(
        self,
        config: Optional[DriftConfig] = None,
        contract_version: str = "v20"
    ):
        self.config = config or DriftConfig()
        self.contract = FEATURE_CONTRACT_V20
        
        # Crear detector por feature
        self.detectors: Dict[str, FeatureDriftDetector] = {}
        for feature_name in self.contract.feature_order:
            self.detectors[feature_name] = FeatureDriftDetector(
                feature_name=feature_name,
                config=self.config
            )
        
        self._alerts: List[DriftAlert] = []
        self._bar_count: int = 0
    
    def update(self, observation: np.ndarray) -> List[DriftAlert]:
        """
        Actualiza todos los detectores con nueva observation.
        
        Args:
            observation: Array de shape (15,) con features
        
        Returns:
            Lista de alertas generadas
        """
        if observation.shape != (self.contract.observation_dim,):
            raise ValueError(f"Shape incorrecto: {observation.shape}")
        
        new_alerts = []
        
        for idx, feature_name in enumerate(self.contract.feature_order):
            value = observation[idx]
            alert = self.detectors[feature_name].update(value, self._bar_count)
            
            if alert:
                new_alerts.append(alert)
                self._alerts.append(alert)
        
        self._bar_count += 1
        
        return new_alerts
    
    def update_batch(self, observations: np.ndarray) -> List[DriftAlert]:
        """Procesa batch de observations."""
        all_alerts = []
        for obs in observations:
            alerts = self.update(obs)
            all_alerts.extend(alerts)
        return all_alerts
    
    def get_drifted_features(self) -> List[str]:
        """Retorna features con alertas recientes."""
        # Alertas en las últimas 100 barras
        recent_cutoff = self._bar_count - 100
        
        drifted = set()
        for alert in self._alerts:
            # Parse timestamp and check recency
            drifted.add(alert.feature_name)
        
        return list(drifted)
    
    def get_status(self) -> Dict:
        """Retorna estado completo del monitor."""
        return {
            "bar_count": self._bar_count,
            "total_alerts": len(self._alerts),
            "detectors": {
                name: detector.get_stats()
                for name, detector in self.detectors.items()
            },
            "recent_alerts": [
                {
                    "feature": a.feature_name,
                    "metric": a.metric_name,
                    "value": a.metric_value,
                    "severity": a.severity
                }
                for a in self._alerts[-10:]  # Last 10 alerts
            ]
        }
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determina si el modelo necesita reentrenamiento.
        
        Returns:
            (should_retrain, reason)
        """
        # Check critical alerts in last 500 bars
        critical_count = sum(
            1 for a in self._alerts[-500:]
            if a.severity == "critical"
        )
        
        if critical_count >= 5:
            return True, f"{critical_count} alertas críticas en últimas 500 barras"
        
        # Check if multiple features drifted
        drifted = self.get_drifted_features()
        if len(drifted) >= 5:
            return True, f"{len(drifted)} features con drift detectado"
        
        return False, "No drift significativo"
```

### Tests (TDD)

```python
# tests/unit/test_drift_detection.py
import pytest
import numpy as np
from lib.risk.drift_detection import (
    DriftConfig,
    FeatureDriftDetector,
    DriftMonitor,
    DriftAlert
)


class TestFeatureDriftDetector:
    """Tests para FeatureDriftDetector."""

    @pytest.fixture
    def config(self):
        return DriftConfig(
            reference_window_size=100,
            detection_window_size=20,
            ks_threshold=0.1,
            psi_threshold=0.2
        )

    @pytest.fixture
    def detector_with_reference(self, config):
        """Detector con referencia establecida."""
        np.random.seed(42)
        reference = np.random.randn(100)
        return FeatureDriftDetector(
            feature_name="test_feature",
            config=config,
            reference_data=reference
        )

    def test_no_drift_on_same_distribution(self, detector_with_reference):
        """Misma distribución NO debe generar alerta."""
        np.random.seed(43)  # Different seed, same distribution
        
        alerts = []
        for i in range(30):
            value = np.random.randn()
            alert = detector_with_reference.update(value, bar_idx=i)
            if alert:
                alerts.append(alert)
        
        assert len(alerts) == 0

    def test_drift_detected_on_mean_shift(self, detector_with_reference):
        """Cambio de media DEBE generar alerta."""
        # Add values with shifted mean
        alerts = []
        for i in range(50):
            value = np.random.randn() + 3.0  # Mean shift
            alert = detector_with_reference.update(value, bar_idx=i)
            if alert:
                alerts.append(alert)
        
        assert len(alerts) > 0
        assert alerts[0].drift_type == "covariate"

    def test_drift_detected_on_variance_change(self, config):
        """Cambio de varianza DEBE generar alerta."""
        np.random.seed(42)
        reference = np.random.randn(100) * 1.0  # std=1
        
        detector = FeatureDriftDetector(
            feature_name="test",
            config=config,
            reference_data=reference
        )
        
        # Add values with different variance
        alerts = []
        for i in range(50):
            value = np.random.randn() * 5.0  # std=5
            alert = detector.update(value, bar_idx=i)
            if alert:
                alerts.append(alert)
        
        assert len(alerts) > 0

    def test_cooldown_prevents_spam(self, config):
        """Cooldown DEBE prevenir alertas repetidas."""
        config.alert_cooldown_bars = 10
        
        np.random.seed(42)
        reference = np.random.randn(100)
        
        detector = FeatureDriftDetector(
            feature_name="test",
            config=config,
            reference_data=reference
        )
        
        # Trigger drift repeatedly
        alerts = []
        for i in range(30):
            value = np.random.randn() + 5.0  # Clear drift
            alert = detector.update(value, bar_idx=i)
            if alert:
                alerts.append((i, alert))
        
        # Should have gaps due to cooldown
        if len(alerts) >= 2:
            gap = alerts[1][0] - alerts[0][0]
            assert gap >= config.alert_cooldown_bars


class TestDriftMonitor:
    """Tests para DriftMonitor."""

    @pytest.fixture
    def monitor(self):
        config = DriftConfig(
            reference_window_size=50,
            detection_window_size=10
        )
        return DriftMonitor(config=config)

    def test_processes_all_features(self, monitor):
        """Monitor DEBE procesar todas las 15 features."""
        assert len(monitor.detectors) == 15

    def test_update_accepts_valid_observation(self, monitor):
        """update() DEBE aceptar observation válida."""
        obs = np.random.randn(15).astype(np.float32)
        
        # Should not raise
        alerts = monitor.update(obs)
        
        assert isinstance(alerts, list)

    def test_update_rejects_wrong_shape(self, monitor):
        """update() DEBE rechazar shape incorrecto."""
        bad_obs = np.random.randn(10)
        
        with pytest.raises(ValueError, match="Shape"):
            monitor.update(bad_obs)

    def test_status_returns_complete_info(self, monitor):
        """get_status() DEBE retornar info completa."""
        # Add some observations
        for _ in range(20):
            obs = np.random.randn(15).astype(np.float32)
            monitor.update(obs)
        
        status = monitor.get_status()
        
        assert "bar_count" in status
        assert "total_alerts" in status
        assert "detectors" in status
        assert len(status["detectors"]) == 15

    def test_should_retrain_triggers_on_many_critical(self, monitor):
        """should_retrain DEBE activarse con muchas alertas críticas."""
        # Simulate many critical alerts
        for _ in range(10):
            alert = DriftAlert(
                timestamp="2026-01-11T00:00:00Z",
                feature_name="test",
                drift_type="covariate",
                metric_name="ks",
                metric_value=0.001,
                threshold=0.1,
                severity="critical"
            )
            monitor._alerts.append(alert)
        
        should, reason = monitor.should_retrain()
        
        assert should
        assert "críticas" in reason


class TestDriftMonitorIntegration:
    """Tests de integración con datos reales."""

    def test_realistic_trading_scenario(self):
        """Escenario realista de trading."""
        monitor = DriftMonitor()
        
        np.random.seed(42)
        
        # Phase 1: Normal market (establish reference)
        for _ in range(200):
            obs = np.random.randn(15).astype(np.float32)
            obs = np.clip(obs, -4, 4)  # Within normal range
            monitor.update(obs)
        
        # Phase 2: Market regime change
        alerts_phase2 = []
        for _ in range(100):
            obs = np.random.randn(15).astype(np.float32) + 2.0  # Mean shift
            obs = np.clip(obs, -4, 4)
            alerts = monitor.update(obs)
            alerts_phase2.extend(alerts)
        
        # Should detect drift
        assert len(alerts_phase2) > 0
```

### Criterios de Aceptación

| # | Criterio | Verificación | Status |
|---|----------|--------------|--------|
| 1 | Detecta mean shift | Unit test | ⬜ |
| 2 | Detecta variance change | Unit test | ⬜ |
| 3 | Cooldown funciona | Unit test | ⬜ |
| 4 | Procesa 15 features | Unit test | ⬜ |
| 5 | should_retrain correcto | Unit test | ⬜ |
| 6 | Coverage ≥ 90% | pytest-cov | ⬜ |

---

## 🎯 TAREA 4: Risk Engine (P1-17)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T4 |
| **Plan Item** | P1-17 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Consume Contratos** | GTR-001, GTR-002, GTR-003 |
| **Produce Contrato** | GTR-004 |

### Objetivo
Implementar Risk Engine que integra ONNX inference, circuit breakers y drift detection en un pipeline coherente.

### Especificación Funcional

```python
# lib/risk/engine.py
"""
Risk Engine - Pipeline integrado de producción.
Contrato ID: GTR-004

Orquesta:
1. Feature validation
2. Model inference (ONNX)
3. Circuit breaker checks
4. Drift monitoring
5. Trade execution gates
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from lib.features.builder import FeatureBuilder
from lib.inference.onnx_converter import ONNXInferenceEngine
from lib.risk.circuit_breakers import MasterCircuitBreaker, CircuitBreakerConfig
from lib.risk.drift_detection import DriftMonitor, DriftConfig


@dataclass
class TradeDecision:
    """Decisión de trading del Risk Engine."""
    action: int  # 0=hold, 1=buy, 2=sell
    confidence: float
    can_execute: bool
    block_reason: Optional[str]
    latency_ms: float
    drift_alerts: List[str]


@dataclass
class RiskEngineConfig:
    """Configuración del Risk Engine."""
    model_path: str
    model_hash: Optional[str] = None
    contract_version: str = "v20"
    circuit_breaker_config: Optional[CircuitBreakerConfig] = None
    drift_config: Optional[DriftConfig] = None
    min_confidence: float = 0.6
    warmup_inferences: int = 10


class RiskEngine:
    """
    Motor de riesgo integrado para producción.
    
    Pipeline:
    1. Construir observation (FeatureBuilder)
    2. Validar features (CircuitBreaker)
    3. Ejecutar inference (ONNX)
    4. Monitorear drift (DriftMonitor)
    5. Tomar decisión de trade
    
    Ejemplo:
        engine = RiskEngine(RiskEngineConfig(model_path="models/v20.onnx"))
        decision = engine.evaluate(ohlcv, macro, position, timestamp, bar_idx)
        if decision.can_execute:
            execute_trade(decision.action)
    """
    
    def __init__(self, config: RiskEngineConfig):
        self.config = config
        
        # Feature builder
        self.feature_builder = FeatureBuilder(version=config.contract_version)
        
        # ONNX inference
        self.inference_engine = ONNXInferenceEngine(
            model_path=config.model_path,
            contract_version=config.contract_version,
            warmup_runs=config.warmup_inferences
        )
        
        # Circuit breakers
        cb_config = config.circuit_breaker_config or CircuitBreakerConfig()
        self.circuit_breaker = MasterCircuitBreaker(cb_config)
        
        # Drift monitor
        drift_config = config.drift_config or DriftConfig()
        self.drift_monitor = DriftMonitor(config=drift_config)
        
        # Stats
        self._evaluation_count: int = 0
        self._blocked_count: int = 0
    
    def evaluate(
        self,
        ohlcv,
        macro,
        position: float,
        timestamp,
        bar_idx: int
    ) -> TradeDecision:
        """
        Evalúa situación de mercado y retorna decisión.
        
        Returns:
            TradeDecision con action, confidence, y status de ejecución
        """
        self._evaluation_count += 1
        drift_alerts = []
        
        # Step 1: Build observation
        try:
            observation = self.feature_builder.build_observation(
                ohlcv=ohlcv,
                macro=macro,
                position=position,
                timestamp=timestamp,
                bar_idx=bar_idx
            )
        except Exception as e:
            self._blocked_count += 1
            return TradeDecision(
                action=0,
                confidence=0.0,
                can_execute=False,
                block_reason=f"Feature build error: {str(e)}",
                latency_ms=0.0,
                drift_alerts=[]
            )
        
        # Step 2: Circuit breaker check (features)
        is_valid, reason = self.circuit_breaker.check_observation(observation)
        if not is_valid:
            self._blocked_count += 1
            return TradeDecision(
                action=0,
                confidence=0.0,
                can_execute=False,
                block_reason=f"Circuit breaker (features): {reason}",
                latency_ms=0.0,
                drift_alerts=[]
            )
        
        # Step 3: Check if we can trade
        can_trade, block_reason = self.circuit_breaker.can_trade()
        if not can_trade:
            self._blocked_count += 1
            return TradeDecision(
                action=0,
                confidence=0.0,
                can_execute=False,
                block_reason=block_reason,
                latency_ms=0.0,
                drift_alerts=[]
            )
        
        # Step 4: Run inference
        action_logits, latency_ms = self.inference_engine.predict(observation)
        
        # Step 5: Circuit breaker check (latency)
        self.circuit_breaker.record_inference(latency_ms)
        
        # Step 6: Drift monitoring
        alerts = self.drift_monitor.update(observation)
        drift_alerts = [f"{a.feature_name}: {a.metric_name}" for a in alerts]
        
        # Step 7: Process action
        action, confidence = self._process_action(action_logits)
        
        # Step 8: Confidence gate
        can_execute = confidence >= self.config.min_confidence
        block_reason = None if can_execute else f"Low confidence: {confidence:.2f}"
        
        if not can_execute:
            self._blocked_count += 1
        
        return TradeDecision(
            action=action,
            confidence=confidence,
            can_execute=can_execute,
            block_reason=block_reason,
            latency_ms=latency_ms,
            drift_alerts=drift_alerts
        )
    
    def _process_action(self, logits: np.ndarray) -> Tuple[int, float]:
        """Convierte logits a action y confidence."""
        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        
        action = int(np.argmax(probs))
        confidence = float(probs[0, action])
        
        return action, confidence
    
    def record_trade_result(self, pnl: float, equity: float) -> None:
        """Registra resultado de trade para circuit breaker."""
        self.circuit_breaker.record_trade_result(pnl, equity)
    
    def get_status(self) -> Dict:
        """Retorna estado completo del engine."""
        return {
            "evaluation_count": self._evaluation_count,
            "blocked_count": self._blocked_count,
            "block_rate": self._blocked_count / max(1, self._evaluation_count),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "drift_monitor": self.drift_monitor.get_status(),
            "inference_latency": self.inference_engine.get_latency_stats()
        }
    
    def should_retrain(self) -> Tuple[bool, str]:
        """Verifica si se necesita reentrenamiento."""
        return self.drift_monitor.should_retrain()
```

### Tests

```python
# tests/integration/test_risk_engine.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from lib.risk.engine import RiskEngine, RiskEngineConfig, TradeDecision


class TestRiskEngine:
    """Tests de integración para RiskEngine."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        """Config con modelo mock."""
        # Create mock ONNX model
        model_path = tmp_path / "test_model.onnx"
        # ... create model ...
        
        return RiskEngineConfig(
            model_path=str(model_path),
            contract_version="v20",
            min_confidence=0.6
        )

    def test_full_pipeline_execution(self, mock_config):
        """Pipeline completo DEBE ejecutarse sin errores."""
        # Mock the ONNX engine
        with patch('lib.risk.engine.ONNXInferenceEngine') as MockEngine:
            mock_engine = Mock()
            mock_engine.predict.return_value = (np.array([[1.0, 0.0, 0.0]]), 2.0)
            MockEngine.return_value = mock_engine
            
            engine = RiskEngine(mock_config)
            
            # Create test data
            ohlcv = create_test_ohlcv(50)
            macro = create_test_macro(50)
            
            decision = engine.evaluate(
                ohlcv=ohlcv,
                macro=macro,
                position=0.0,
                timestamp=ohlcv.index[40],
                bar_idx=40
            )
            
            assert isinstance(decision, TradeDecision)
            assert decision.action in [0, 1, 2]

    def test_circuit_breaker_blocks_on_bad_features(self, mock_config):
        """Circuit breaker DEBE bloquear features inválidas."""
        with patch('lib.risk.engine.ONNXInferenceEngine'):
            engine = RiskEngine(mock_config)
            
            # Force bad observation
            with patch.object(
                engine.feature_builder,
                'build_observation',
                return_value=np.ones(15) * np.nan
            ):
                ohlcv = create_test_ohlcv(50)
                macro = create_test_macro(50)
                
                decision = engine.evaluate(
                    ohlcv=ohlcv,
                    macro=macro,
                    position=0.0,
                    timestamp=ohlcv.index[40],
                    bar_idx=40
                )
                
                assert not decision.can_execute
                assert "Circuit breaker" in decision.block_reason

    def test_low_confidence_blocks_execution(self, mock_config):
        """Confidence baja DEBE bloquear ejecución."""
        with patch('lib.risk.engine.ONNXInferenceEngine') as MockEngine:
            mock_engine = Mock()
            # Equal probabilities = low confidence
            mock_engine.predict.return_value = (np.array([[0.0, 0.0, 0.0]]), 2.0)
            MockEngine.return_value = mock_engine
            
            engine = RiskEngine(mock_config)
            
            ohlcv = create_test_ohlcv(50)
            macro = create_test_macro(50)
            
            decision = engine.evaluate(
                ohlcv=ohlcv,
                macro=macro,
                position=0.0,
                timestamp=ohlcv.index[40],
                bar_idx=40
            )
            
            assert not decision.can_execute
            assert "confidence" in decision.block_reason.lower()


def create_test_ohlcv(n: int):
    """Helper para crear OHLCV de test."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-10 13:00", periods=n, freq="5min", tz="UTC")
    close = 4250 + np.cumsum(np.random.randn(n) * 5)
    return pd.DataFrame({
        "open": close + np.random.randn(n),
        "high": close + np.abs(np.random.randn(n) * 3),
        "low": close - np.abs(np.random.randn(n) * 3),
        "close": close,
        "volume": np.random.uniform(1e6, 5e6, n),
    }, index=dates)


def create_test_macro(n: int):
    """Helper para crear macro data de test."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-10 13:00", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "dxy": np.random.uniform(103, 105, n),
        "vix": np.random.uniform(15, 25, n),
        "embi": np.random.uniform(300, 400, n),
        "brent": np.random.uniform(70, 80, n),
        "usdmxn": np.random.uniform(17, 18, n),
        "rate_spread": np.random.uniform(5, 7, n),
    }, index=dates)
```

---

## 🎯 TAREA 5: Correcciones P0 en Inference (P0-2, P0-3, P0-6)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T5 |
| **Plan Item** | P0-2, P0-3, P0-6 |
| **Prioridad** | CRÍTICA |

### P0-2: Action Threshold Mismatch

**Problema:**
```python
# ACTUAL (INCORRECTO)
buy_threshold: float = 0.6
sell_threshold: float = 0.4

# ESPERADO (CORRECTO - simétrico)
buy_threshold: float = 0.6
sell_threshold: float = 0.6
```

**Ubicación:** `services/inference_api/config.py:32-33`

**Test:**
```python
def test_p0_2_thresholds_symmetric():
    """Thresholds DEBEN ser simétricos."""
    from services.inference_api.config import Settings
    settings = Settings()
    assert settings.buy_threshold == settings.sell_threshold
```

### P0-3: ADX Hardcoded

**Problema:**
```python
# ACTUAL (INCORRECTO)
adx_threshold = 25  # ❌ Hardcoded

# ESPERADO (CORRECTO)
from lib.features.contract import FEATURE_CONTRACT_V20
adx_period = FEATURE_CONTRACT_V20.technical_periods["adx"]  # ✅
```

**Ubicación:** `airflow/dags/l5_multi_model_inference.py:371`

**Test:**
```python
def test_p0_3_adx_not_hardcoded():
    """ADX NO debe estar hardcoded."""
    with open("airflow/dags/l5_multi_model_inference.py") as f:
        source = f.read()
    
    # Should not have hardcoded ADX values
    assert "adx_threshold = 25" not in source
    assert "FEATURE_CONTRACT" in source or "technical_periods" in source
```

### P0-6: Model ID Hardcoded

**Problema:**
```python
# ACTUAL (INCORRECTO)
modelId: 'v19-checkpoint'  // ❌ Hardcoded

# ESPERADO (CORRECTO)
modelId: process.env.CURRENT_MODEL_ID  // ✅ From env
```

**Ubicación:** `lib/replayApiClient.ts:518`

**Test:**
```typescript
// tests/replayApiClient.test.ts
test('modelId comes from environment', () => {
    expect(config.modelId).not.toBe('v19-checkpoint');
    expect(config.modelId).toBe(process.env.CURRENT_MODEL_ID);
});
```

---

## 📊 RESUMEN DE CONTRATOS PRODUCIDOS

| Contrato ID | Descripción | Consumidor |
|-------------|-------------|------------|
| **GTR-001** | `ONNXConverter` + `ONNXInferenceEngine` | GTR-004 (RiskEngine) |
| **GTR-002** | `CircuitBreaker` system | GTR-004 (RiskEngine) |
| **GTR-003** | `DriftDetector` + `DriftMonitor` | GTR-004 (RiskEngine) |
| **GTR-004** | `RiskEngine` (integración) | Production API |

---

## ✅ CHECKLIST FINAL

### Pre-Desarrollo
- [ ] Verificar que CTR-001 a CTR-004 (de Claude) están completos
- [ ] Leer ARCHITECTURE_CONTRACTS.md
- [ ] Configurar pytest + coverage

### Por Tarea
- [ ] GEMINI-T1: ONNX Converter
- [ ] GEMINI-T2: Circuit Breakers
- [ ] GEMINI-T3: Drift Detection
- [ ] GEMINI-T4: Risk Engine
- [ ] GEMINI-T5: P0-2, P0-3, P0-6 fixes

### Pre-Entrega
- [ ] `pytest tests/ -v` pasa al 100%
- [ ] `pytest --cov=lib/risk --cov-report=term-missing` >= 90%
- [ ] `pytest --cov=lib/inference --cov-report=term-missing` >= 90%
- [ ] Latencia ONNX P95 < 5ms verificada
- [ ] Documentación de contratos GTR-001 a GTR-004 completa

---

## 🔄 PROTOCOLO DE SINCRONIZACIÓN

### Comunicación con Claude

```markdown
## Handoff Protocol

### Cuando Claude complete CTR-001 (FeatureBuilder):
1. Claude notifica: "CTR-001 READY"
2. Gemini verifica: import lib.features.builder funciona
3. Gemini comienza: GEMINI-T1 (ONNX)

### Cuando Claude complete CTR-003 (norm_stats):
1. Claude notifica: "CTR-003 READY"
2. Gemini verifica: config/v20_norm_stats.json existe y tiene 13 features
3. Gemini usa en: GEMINI-T1, GEMINI-T3

### Interface Contract Verification:
```python
# Gemini debe verificar antes de comenzar
def verify_claude_contracts():
    from lib.features.builder import FeatureBuilder
    from lib.features.contract import FEATURE_CONTRACT_V20
    
    builder = FeatureBuilder(version="v20")
    assert builder.get_observation_dim() == 15
    assert len(builder.get_feature_names()) == 15
    
    print("✅ CTR-001 y CTR-002 verificados")
```
```

---

*Documento generado: 2026-01-11*  
*Metodología: Spec-Driven + AI-Augmented TDD*  
*Coordinación: Contratos GTR-001 a GTR-004 con CLAUDE_TASKS.md*
