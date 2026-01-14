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

## 🎯 TAREA 6: ExternalTaskSensor para DAGs (P1-1)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T6 |
| **Plan Item** | P1-1 |
| **Prioridad** | ALTA |
| **Ubicación** | `airflow/dags/` |

### Objetivo
Eliminar dependencias implícitas entre DAGs reemplazando `catchup=False` con `ExternalTaskSensor` para coordinación explícita.

### Problema
```python
# ACTUAL (INCORRECTO - dependencias implícitas)
dag = DAG(
    'l5_model_inference',
    catchup=False,  # ❌ Asume que DAG anterior terminó
)

# ESPERADO (CORRECTO - dependencia explícita)
wait_for_features = ExternalTaskSensor(
    task_id='wait_for_feature_pipeline',
    external_dag_id='l4_feature_pipeline',
    external_task_id='final_task',
    timeout=3600,
)
```

### Solución

```python
# airflow/dags/utils/dag_dependencies.py
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import timedelta


def create_dag_dependency(
    task_id: str,
    external_dag_id: str,
    external_task_id: str,
    timeout_hours: int = 1,
    execution_delta: timedelta = None
) -> ExternalTaskSensor:
    """
    Crea sensor de dependencia entre DAGs.

    Args:
        task_id: ID del task sensor
        external_dag_id: DAG del que dependemos
        external_task_id: Task específico del que dependemos
        timeout_hours: Timeout en horas
        execution_delta: Delta de tiempo para ejecución (si DAGs no están sincronizados)

    Returns:
        ExternalTaskSensor configurado
    """
    return ExternalTaskSensor(
        task_id=task_id,
        external_dag_id=external_dag_id,
        external_task_id=external_task_id,
        timeout=timeout_hours * 3600,
        execution_delta=execution_delta or timedelta(hours=0),
        mode='reschedule',  # No bloquea worker mientras espera
        poke_interval=60,
    )


# Dependencias estándar del proyecto
DAG_DEPENDENCIES = {
    'l5_model_inference': [
        {
            'task_id': 'wait_for_features',
            'external_dag_id': 'l4_feature_pipeline',
            'external_task_id': 'validate_features',
        },
        {
            'task_id': 'wait_for_macro',
            'external_dag_id': 'l3_macro_ingest',
            'external_task_id': 'final_validation',
        },
    ],
    'l4_feature_pipeline': [
        {
            'task_id': 'wait_for_ohlcv',
            'external_dag_id': 'l2_ohlcv_ingest',
            'external_task_id': 'validate_ohlcv',
        },
    ],
}
```

### Test de Validación

```python
# tests/unit/test_p1_1_dag_dependencies.py

def test_no_implicit_dag_dependencies():
    """DAGs con dependencias DEBEN usar ExternalTaskSensor."""
    from airflow.sensors.external_task import ExternalTaskSensor
    import importlib

    dags_with_dependencies = [
        'l5_multi_model_inference',
        'l4_feature_pipeline',
    ]

    for dag_module in dags_with_dependencies:
        module = importlib.import_module(f'airflow.dags.{dag_module}')
        dag = getattr(module, 'dag', None)

        if dag:
            sensor_tasks = [
                t for t in dag.tasks
                if isinstance(t, ExternalTaskSensor)
            ]
            assert len(sensor_tasks) > 0, \
                f"DAG {dag_module} debe usar ExternalTaskSensor"
```

---

## 🎯 TAREA 7: Unificar Fuente Macro (P1-7)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T7 |
| **Plan Item** | P1-7 |
| **Prioridad** | ALTA |
| **Dependencias** | CTR-001 (FeatureBuilder de Claude) |

### Objetivo
Unificar todas las fuentes de datos macro en una sola fuente de verdad, eliminando duplicación entre BD y archivos estáticos.

### Problema
```python
# ACTUAL (INCORRECTO - múltiples fuentes)
# Archivo 1: macro_data.csv (estático)
# Archivo 2: tabla macro_daily (BD)
# Archivo 3: API endpoint /macro (runtime)

# ESPERADO (CORRECTO - single source of truth)
from lib.data.macro_source import MacroDataSource
macro_df = MacroDataSource.get_macro_data(start_date, end_date)
```

### Solución

```python
# lib/data/macro_source.py
from typing import Optional
import pandas as pd
from datetime import date
from sqlalchemy import create_engine


class MacroDataSource:
    """
    Single Source of Truth para datos macro.
    Contrato: GTR-005

    Prioridad de fuentes:
    1. BD PostgreSQL (más actualizada)
    2. Cache Redis (TTL 1 hora)
    3. Fallback a CSV (histórico)
    """

    _engine = None
    _cache = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            from lib.config.security import SecuritySettings
            settings = SecuritySettings()
            cls._engine = create_engine(settings.get_postgres_url())
        return cls._engine

    @classmethod
    def get_macro_data(
        cls,
        start_date: date,
        end_date: date,
        columns: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Obtiene datos macro desde la fuente unificada.

        Args:
            start_date: Fecha inicio
            end_date: Fecha fin
            columns: Columnas específicas (default: todas)

        Returns:
            DataFrame con datos macro

        Raises:
            ValueError: Si no hay datos para el rango
        """
        # Intentar BD primero
        try:
            df = cls._fetch_from_db(start_date, end_date, columns)
            if len(df) > 0:
                return df
        except Exception as e:
            logger.warning(f"BD macro fallback: {e}")

        # Fallback a CSV histórico
        df = cls._fetch_from_csv(start_date, end_date, columns)

        if len(df) == 0:
            raise ValueError(
                f"No hay datos macro para {start_date} - {end_date}"
            )

        return df

    @classmethod
    def _fetch_from_db(
        cls,
        start_date: date,
        end_date: date,
        columns: Optional[list]
    ) -> pd.DataFrame:
        """Fetch desde PostgreSQL."""
        cols = ", ".join(columns) if columns else "*"
        query = f"""
            SELECT {cols}
            FROM macro_daily
            WHERE date >= %s AND date <= %s
            ORDER BY date
        """
        return pd.read_sql(
            query,
            cls.get_engine(),
            params=[start_date, end_date],
            parse_dates=['date'],
            index_col='date'
        )

    @classmethod
    def _fetch_from_csv(
        cls,
        start_date: date,
        end_date: date,
        columns: Optional[list]
    ) -> pd.DataFrame:
        """Fallback a CSV histórico."""
        csv_path = Path("data/macro/macro_historical.csv")
        if not csv_path.exists():
            return pd.DataFrame()

        df = pd.read_csv(csv_path, parse_dates=['date'], index_col='date')
        df = df[(df.index >= str(start_date)) & (df.index <= str(end_date))]

        if columns:
            df = df[columns]

        return df

    @classmethod
    def validate_coverage(
        cls,
        start_date: date,
        end_date: date
    ) -> dict:
        """Valida cobertura de datos para el rango."""
        df = cls.get_macro_data(start_date, end_date)

        expected_days = pd.bdate_range(start_date, end_date)  # Business days
        actual_days = df.index.unique()

        missing = set(expected_days) - set(actual_days)

        return {
            'expected': len(expected_days),
            'actual': len(actual_days),
            'coverage_pct': len(actual_days) / len(expected_days) * 100,
            'missing_dates': sorted(list(missing)),
        }
```

### Test de Validación

```python
# tests/unit/test_p1_7_macro_source.py

def test_macro_single_source_of_truth():
    """Solo debe existir una fuente de macro data."""
    from lib.data.macro_source import MacroDataSource

    # Verificar que existe y funciona
    df = MacroDataSource.get_macro_data(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31)
    )

    assert len(df) > 0
    assert 'dxy' in df.columns
    assert 'vix' in df.columns


def test_macro_coverage_validation():
    """Validación de cobertura debe funcionar."""
    from lib.data.macro_source import MacroDataSource

    coverage = MacroDataSource.validate_coverage(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31)
    )

    assert coverage['coverage_pct'] >= 95  # 95% mínimo
```

---

## 🎯 TAREA 8: Dataset Registry (P1-9)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T8 |
| **Plan Item** | P1-9 |
| **Prioridad** | ALTA |
| **Produce Contrato** | GTR-006 |

### Objetivo
Implementar registro de datasets con checksums para garantizar reproducibilidad de entrenamientos.

### Solución

```python
# lib/dataset_registry.py
import hashlib
import json
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict


@dataclass
class DatasetMetadata:
    """Metadata registrada para cada dataset."""
    dataset_name: str
    version: str
    checksum_sha256: str
    checksum_md5: str
    row_count: int
    column_count: int
    columns: list
    date_start: datetime
    date_end: datetime
    stats: Dict[str, dict]
    created_at: datetime
    git_commit: Optional[str] = None
    pipeline_version: Optional[str] = None


class DatasetRegistry:
    """
    Registro de datasets para reproducibilidad.
    Contrato: GTR-006

    Uso:
        registry = DatasetRegistry(connection)
        metadata = registry.register(df, "training_v20", "1.0")
        registry.verify(df, "training_v20", "1.0")  # Raises si checksum diferente
    """

    def __init__(self, connection):
        self.conn = connection

    def register(
        self,
        df: pd.DataFrame,
        name: str,
        version: str,
        git_commit: Optional[str] = None,
        pipeline_version: Optional[str] = None
    ) -> DatasetMetadata:
        """
        Registra dataset con checksums.

        Args:
            df: DataFrame a registrar
            name: Nombre del dataset
            version: Versión semántica
            git_commit: Commit de git (opcional)
            pipeline_version: Versión del pipeline (opcional)

        Returns:
            DatasetMetadata con todos los checksums
        """
        # Serializar para checksum (determinístico)
        content = df.to_csv(index=False).encode('utf-8')

        # Checksums
        md5 = hashlib.md5(content).hexdigest()
        sha256 = hashlib.sha256(content).hexdigest()

        # Stats por columna numérica
        stats = {}
        for col in df.select_dtypes(include='number').columns:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'nulls': int(df[col].isna().sum()),
            }

        # Determinar rango de fechas
        date_col = next(
            (c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()),
            None
        )
        date_start = df[date_col].min() if date_col else None
        date_end = df[date_col].max() if date_col else None

        metadata = DatasetMetadata(
            dataset_name=name,
            version=version,
            checksum_sha256=sha256,
            checksum_md5=md5,
            row_count=len(df),
            column_count=len(df.columns),
            columns=list(df.columns),
            date_start=date_start,
            date_end=date_end,
            stats=stats,
            created_at=datetime.utcnow(),
            git_commit=git_commit,
            pipeline_version=pipeline_version,
        )

        # Insertar en BD
        self._insert_metadata(metadata)

        return metadata

    def verify(
        self,
        df: pd.DataFrame,
        name: str,
        version: str
    ) -> bool:
        """
        Verifica que dataset coincide con checksum registrado.

        Raises:
            ValueError: Si checksum no coincide
        """
        content = df.to_csv(index=False).encode('utf-8')
        actual_sha256 = hashlib.sha256(content).hexdigest()

        cur = self.conn.cursor()
        cur.execute("""
            SELECT checksum_sha256 FROM dataset_registry
            WHERE dataset_name = %s AND version = %s
        """, (name, version))

        row = cur.fetchone()
        if not row:
            raise ValueError(f"Dataset {name} v{version} no registrado")

        expected_sha256 = row[0]

        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Dataset checksum mismatch!\n"
                f"Expected: {expected_sha256[:16]}...\n"
                f"Actual: {actual_sha256[:16]}...\n"
                f"El dataset puede haber sido modificado."
            )

        return True

    def _insert_metadata(self, metadata: DatasetMetadata):
        """Inserta metadata en BD."""
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO dataset_registry
            (dataset_name, version, checksum_sha256, checksum_md5,
             row_count, column_count, columns_json, date_start, date_end,
             stats_json, git_commit, pipeline_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (dataset_name, version) DO UPDATE SET
                checksum_sha256 = EXCLUDED.checksum_sha256,
                checksum_md5 = EXCLUDED.checksum_md5
        """, (
            metadata.dataset_name, metadata.version,
            metadata.checksum_sha256, metadata.checksum_md5,
            metadata.row_count, metadata.column_count,
            json.dumps(metadata.columns),
            metadata.date_start, metadata.date_end,
            json.dumps(metadata.stats),
            metadata.git_commit, metadata.pipeline_version,
        ))
        self.conn.commit()
```

### SQL Migration

```sql
-- migrations/V003__dataset_registry.sql

CREATE TABLE IF NOT EXISTS dataset_registry (
    id SERIAL PRIMARY KEY,
    dataset_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Checksums
    checksum_md5 VARCHAR(32) NOT NULL,
    checksum_sha256 VARCHAR(64) NOT NULL,

    -- Dimensions
    row_count INTEGER NOT NULL,
    column_count INTEGER NOT NULL,
    columns_json JSONB NOT NULL,

    -- Date ranges
    date_start TIMESTAMP,
    date_end TIMESTAMP,

    -- Source tracking
    git_commit VARCHAR(40),
    pipeline_version VARCHAR(20),

    -- Stats
    stats_json JSONB,

    UNIQUE(dataset_name, version)
);

CREATE INDEX idx_dataset_name ON dataset_registry(dataset_name);
CREATE INDEX idx_dataset_checksum ON dataset_registry(checksum_sha256);
```

### Test de Validación

```python
# tests/integration/test_p1_9_dataset_registry.py

def test_dataset_registration_and_verification():
    """Dataset registration debe funcionar end-to-end."""
    import pandas as pd
    from lib.dataset_registry import DatasetRegistry

    # Crear dataset de prueba
    df = pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=100),
        'value': range(100),
    })

    # Registrar
    registry = DatasetRegistry(get_test_connection())
    metadata = registry.register(df, "test_dataset", "1.0")

    assert metadata.checksum_sha256 is not None
    assert metadata.row_count == 100

    # Verificar (mismo df debe pasar)
    assert registry.verify(df, "test_dataset", "1.0") is True

    # Verificar (df modificado debe fallar)
    df_modified = df.copy()
    df_modified['value'] = df_modified['value'] + 1

    with pytest.raises(ValueError, match="checksum mismatch"):
        registry.verify(df_modified, "test_dataset", "1.0")
```

---

## 🎯 TAREA 9: Externalizar Config V20 (P1-10)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T9 |
| **Plan Item** | P1-10 |
| **Prioridad** | ALTA |
| **Produce Contrato** | GTR-007 |

### Objetivo
Externalizar toda la configuración hardcodeada del modelo V20 a archivos YAML, permitiendo cambios sin modificar código.

### Solución

```yaml
# config/v20_config.yaml
# Contrato: GTR-007
# NUNCA modificar durante ejecución. Crear v21_config.yaml para cambios.

model:
  name: ppo_v20
  version: "20"
  observation_dim: 15
  action_space: 3  # HOLD, BUY, SELL

training:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  ent_coef: 0.01
  clip_range: 0.2
  gamma: 0.99
  gae_lambda: 0.95

thresholds:
  long: 0.30
  short: -0.30
  confidence_min: 0.6

features:
  norm_stats_path: "config/v20_norm_stats.json"
  clip_range: [-5.0, 5.0]
  warmup_bars: 14

trading:
  initial_capital: 10000
  transaction_cost_bps: 25
  slippage_bps: 5
  max_position_size: 1.0

risk:
  max_drawdown_pct: 15.0
  daily_loss_limit_pct: 5.0
  position_limit: 1.0
  volatility_scaling: true

dates:
  training_start: "2023-01-01"
  training_end: "2024-12-31"
  validation_start: "2025-01-01"
  validation_end: "2025-06-30"
  test_start: "2025-07-01"
```

```python
# lib/config_loader.py
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field
from functools import lru_cache


class ModelConfig(BaseModel):
    name: str
    version: str
    observation_dim: int = 15
    action_space: int = 3


class TrainingConfig(BaseModel):
    learning_rate: float
    n_steps: int
    batch_size: int
    n_epochs: int
    ent_coef: float
    clip_range: float
    gamma: float = 0.99
    gae_lambda: float = 0.95


class ThresholdsConfig(BaseModel):
    long: float = 0.30
    short: float = -0.30
    confidence_min: float = 0.6


class FeaturesConfig(BaseModel):
    norm_stats_path: str
    clip_range: list = Field(default=[-5.0, 5.0])
    warmup_bars: int = 14


class TradingConfig(BaseModel):
    initial_capital: float = 10000
    transaction_cost_bps: float = 25
    slippage_bps: float = 5
    max_position_size: float = 1.0


class RiskConfig(BaseModel):
    max_drawdown_pct: float = 15.0
    daily_loss_limit_pct: float = 5.0
    position_limit: float = 1.0
    volatility_scaling: bool = True


class DatesConfig(BaseModel):
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str


class V20Config(BaseModel):
    """
    Configuración completa V20.
    Contrato: GTR-007
    """
    model: ModelConfig
    training: TrainingConfig
    thresholds: ThresholdsConfig
    features: FeaturesConfig
    trading: TradingConfig
    risk: RiskConfig
    dates: DatesConfig


@lru_cache(maxsize=1)
def load_config(version: str = 'v20') -> V20Config:
    """
    Carga configuración desde YAML.

    Args:
        version: Versión del modelo (default: v20)

    Returns:
        V20Config validada

    Raises:
        FileNotFoundError: Si archivo no existe
        ValidationError: Si config inválida
    """
    config_path = Path(f'config/{version}_config.yaml')

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config {config_path} no existe. "
            f"Disponibles: {list(Path('config').glob('*_config.yaml'))}"
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return V20Config(**data)


def get_training_params(version: str = 'v20') -> dict:
    """Convenience function para obtener params de SB3."""
    config = load_config(version)
    return {
        'learning_rate': config.training.learning_rate,
        'n_steps': config.training.n_steps,
        'batch_size': config.training.batch_size,
        'n_epochs': config.training.n_epochs,
        'ent_coef': config.training.ent_coef,
        'clip_range': config.training.clip_range,
        'gamma': config.training.gamma,
        'gae_lambda': config.training.gae_lambda,
    }
```

### Test de Validación

```python
# tests/unit/test_p1_10_config_loader.py

def test_config_loads_from_yaml():
    """Config debe cargar desde YAML, no hardcoded."""
    from lib.config_loader import load_config

    config = load_config('v20')

    assert config.model.version == "20"
    assert config.training.learning_rate == 3.0e-4
    assert config.features.observation_dim == 15


def test_no_hardcoded_training_params():
    """Training script NO debe tener params hardcoded."""
    import ast

    with open("scripts/train_v20_production.py") as f:
        source = f.read()

    tree = ast.parse(source)

    # Buscar assignments directos de learning_rate
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if hasattr(target, 'id'):
                    if target.id in ['learning_rate', 'n_steps', 'batch_size']:
                        # Debe venir de config, no hardcoded
                        assert not isinstance(node.value, (ast.Num, ast.Constant)), \
                            f"{target.id} está hardcoded"


def test_config_file_exists():
    """Config file v20_config.yaml DEBE existir."""
    from pathlib import Path

    config_path = Path("config/v20_config.yaml")
    assert config_path.exists(), "config/v20_config.yaml no existe"
```

---

## 🎯 TAREA 10: Remove Dual Equity Filtering (P1-4)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T10 |
| **Plan Item** | P1-4 |
| **Prioridad** | MEDIA |
| **Ubicación** | `usdcop-trading-dashboard/components/trading/` |

### Problema
Existe filtrado duplicado de equity en frontend que causa inconsistencias:
- Un filtro en `TradingView` component
- Otro filtro en `TradesTable` component
- Lógica duplicada que puede divergir

### Solución

```typescript
// lib/trading/equityFilter.ts

/**
 * Single source of truth para filtrado de equity.
 * Contrato: GTR-008
 */

export interface EquityFilterConfig {
    minEquity: number;
    maxDrawdownPct: number;
    excludeWarmup: boolean;
    warmupBars: number;
}

export const DEFAULT_EQUITY_FILTER: EquityFilterConfig = {
    minEquity: 0,
    maxDrawdownPct: 100,
    excludeWarmup: true,
    warmupBars: 14,  // From FEATURE_CONTRACT_V20
};

export function filterEquityData(
    data: EquityDataPoint[],
    config: EquityFilterConfig = DEFAULT_EQUITY_FILTER
): EquityDataPoint[] {
    let filtered = data;

    // Excluir warmup si configurado
    if (config.excludeWarmup) {
        filtered = filtered.slice(config.warmupBars);
    }

    // Filtrar por equity mínimo
    if (config.minEquity > 0) {
        filtered = filtered.filter(d => d.equity >= config.minEquity);
    }

    // Filtrar por drawdown máximo
    if (config.maxDrawdownPct < 100) {
        let peak = filtered[0]?.equity ?? 0;
        filtered = filtered.filter(d => {
            peak = Math.max(peak, d.equity);
            const drawdown = (peak - d.equity) / peak * 100;
            return drawdown <= config.maxDrawdownPct;
        });
    }

    return filtered;
}
```

```typescript
// Actualizar componentes para usar filtro único

// components/trading/TradingView.tsx
import { filterEquityData, DEFAULT_EQUITY_FILTER } from '@/lib/trading/equityFilter';

// ANTES (duplicado):
// const filteredData = data.filter(d => d.equity > 0).slice(14);

// DESPUÉS (unificado):
const filteredData = filterEquityData(data, {
    ...DEFAULT_EQUITY_FILTER,
    minEquity: userConfig.minEquity ?? 0,
});
```

### Test de Validación

```typescript
// tests/unit/equityFilter.test.ts

describe('Equity Filter - Single Source of Truth', () => {
    it('should filter warmup bars when configured', () => {
        const data = Array.from({ length: 100 }, (_, i) => ({
            bar: i,
            equity: 10000 + i * 10,
        }));

        const filtered = filterEquityData(data, {
            ...DEFAULT_EQUITY_FILTER,
            excludeWarmup: true,
            warmupBars: 14,
        });

        expect(filtered.length).toBe(86);  // 100 - 14
        expect(filtered[0].bar).toBe(14);
    });

    it('should apply consistent filtering across components', () => {
        // Verificar que no hay lógica duplicada
        const tradingViewSource = readFileSync('components/trading/TradingView.tsx', 'utf-8');
        const tradesTableSource = readFileSync('components/trading/TradesTable.tsx', 'utf-8');

        // No debe haber filtrado inline
        expect(tradingViewSource).not.toMatch(/\.filter\(.*equity.*\)\.slice/);
        expect(tradesTableSource).not.toMatch(/\.filter\(.*equity.*\)\.slice/);

        // Debe usar el filtro centralizado
        expect(tradingViewSource).toContain('filterEquityData');
        expect(tradesTableSource).toContain('filterEquityData');
    });
});
```

---

## 🎯 TAREA 11: Paper Trading Validation Framework (P1-8)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T11 |
| **Plan Item** | P1-8 |
| **Prioridad** | ALTA |
| **Audit Ref** | ML-02, ML-06 |
| **Produce Contrato** | GTR-009 |

### Objetivo
Crear framework para validar resultados de paper trading contra backtest, detectando discrepancias significativas.

### Solución

```python
# lib/validation/paper_trading_validator.py
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


@dataclass
class ValidationResult:
    """Resultado de validación paper vs backtest."""
    metric: str
    backtest_value: float
    paper_value: float
    difference: float
    difference_pct: float
    threshold_pct: float
    passed: bool
    notes: str = ""


@dataclass
class PaperTradingReport:
    """Reporte completo de validación."""
    start_date: datetime
    end_date: datetime
    total_days: int
    results: List[ValidationResult]
    overall_passed: bool
    warnings: List[str]
    critical_failures: List[str]


class PaperTradingValidator:
    """
    Valida resultados de paper trading contra backtest.
    Contrato: GTR-009

    Uso:
        validator = PaperTradingValidator(
            backtest_results=backtest_df,
            paper_results=paper_df,
            thresholds=VALIDATION_THRESHOLDS
        )
        report = validator.validate()

        if not report.overall_passed:
            raise ValidationError(report.critical_failures)
    """

    # Thresholds por defecto (diferencia % máxima permitida)
    DEFAULT_THRESHOLDS = {
        'total_return': 10.0,      # 10% diferencia permitida
        'sharpe_ratio': 15.0,      # 15% diferencia permitida
        'max_drawdown': 20.0,      # 20% diferencia permitida
        'win_rate': 10.0,          # 10% diferencia permitida
        'avg_trade_return': 15.0,  # 15% diferencia permitida
        'trade_count': 25.0,       # 25% diferencia permitida
        'execution_slippage': 50.0, # 50% - paper puede tener más slippage
    }

    def __init__(
        self,
        backtest_results: pd.DataFrame,
        paper_results: pd.DataFrame,
        thresholds: Optional[Dict[str, float]] = None
    ):
        self.backtest = backtest_results
        self.paper = paper_results
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

    def validate(self) -> PaperTradingReport:
        """Ejecuta validación completa."""
        results = []
        warnings = []
        critical_failures = []

        # Validar cada métrica
        for metric, threshold in self.thresholds.items():
            result = self._validate_metric(metric, threshold)
            results.append(result)

            if not result.passed:
                if threshold <= 15.0:  # Thresholds estrictos son críticos
                    critical_failures.append(
                        f"{metric}: {result.difference_pct:.1f}% diff (max {threshold}%)"
                    )
                else:
                    warnings.append(
                        f"{metric}: {result.difference_pct:.1f}% diff (threshold {threshold}%)"
                    )

        # Validaciones adicionales
        self._check_trade_alignment(warnings)
        self._check_timing_consistency(warnings)

        return PaperTradingReport(
            start_date=self.paper.index.min(),
            end_date=self.paper.index.max(),
            total_days=(self.paper.index.max() - self.paper.index.min()).days,
            results=results,
            overall_passed=len(critical_failures) == 0,
            warnings=warnings,
            critical_failures=critical_failures
        )

    def _validate_metric(self, metric: str, threshold: float) -> ValidationResult:
        """Valida una métrica específica."""
        backtest_val = self._calculate_metric(self.backtest, metric)
        paper_val = self._calculate_metric(self.paper, metric)

        if backtest_val == 0:
            diff_pct = 100.0 if paper_val != 0 else 0.0
        else:
            diff_pct = abs(paper_val - backtest_val) / abs(backtest_val) * 100

        return ValidationResult(
            metric=metric,
            backtest_value=backtest_val,
            paper_value=paper_val,
            difference=paper_val - backtest_val,
            difference_pct=diff_pct,
            threshold_pct=threshold,
            passed=diff_pct <= threshold
        )

    def _calculate_metric(self, df: pd.DataFrame, metric: str) -> float:
        """Calcula métrica del DataFrame."""
        if metric == 'total_return':
            return (df['equity'].iloc[-1] / df['equity'].iloc[0] - 1) * 100
        elif metric == 'sharpe_ratio':
            returns = df['equity'].pct_change().dropna()
            return returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        elif metric == 'max_drawdown':
            equity = df['equity']
            peak = equity.expanding().max()
            drawdown = (equity - peak) / peak
            return abs(drawdown.min()) * 100
        elif metric == 'win_rate':
            if 'pnl' in df.columns:
                return (df['pnl'] > 0).mean() * 100
            return 50.0
        elif metric == 'trade_count':
            if 'trade_id' in df.columns:
                return df['trade_id'].nunique()
            return 0
        else:
            return 0.0

    def _check_trade_alignment(self, warnings: List[str]):
        """Verifica que trades ocurren en momentos similares."""
        if 'trade_time' not in self.backtest.columns or 'trade_time' not in self.paper.columns:
            return

        bt_trades = set(self.backtest['trade_time'].dt.date.unique())
        paper_trades = set(self.paper['trade_time'].dt.date.unique())

        only_backtest = bt_trades - paper_trades
        only_paper = paper_trades - bt_trades

        if len(only_backtest) > len(bt_trades) * 0.2:
            warnings.append(
                f"20%+ trades solo en backtest ({len(only_backtest)} días)"
            )

        if len(only_paper) > len(paper_trades) * 0.2:
            warnings.append(
                f"20%+ trades solo en paper ({len(only_paper)} días)"
            )

    def _check_timing_consistency(self, warnings: List[str]):
        """Verifica consistencia temporal."""
        min_days = 60  # Mínimo 2 meses para validación

        actual_days = (self.paper.index.max() - self.paper.index.min()).days

        if actual_days < min_days:
            warnings.append(
                f"Paper trading muy corto: {actual_days} días (mínimo recomendado: {min_days})"
            )
```

### Test de Validación

```python
# tests/integration/test_p1_8_paper_validation.py

def test_paper_trading_validation_framework():
    """Framework de validación debe funcionar end-to-end."""
    from lib.validation.paper_trading_validator import PaperTradingValidator

    # Crear datos sintéticos
    backtest = create_synthetic_results(sharpe=1.5, trades=100)
    paper = create_synthetic_results(sharpe=1.4, trades=95)  # Ligeramente peor

    validator = PaperTradingValidator(backtest, paper)
    report = validator.validate()

    # Pequeñas diferencias deben pasar
    assert report.overall_passed, f"Failures: {report.critical_failures}"


def test_paper_trading_detects_significant_divergence():
    """Divergencias significativas deben ser detectadas."""
    from lib.validation.paper_trading_validator import PaperTradingValidator

    backtest = create_synthetic_results(sharpe=1.5, trades=100)
    paper = create_synthetic_results(sharpe=0.5, trades=50)  # Muy diferente

    validator = PaperTradingValidator(backtest, paper)
    report = validator.validate()

    assert not report.overall_passed
    assert len(report.critical_failures) > 0
```

---

## 🎯 TAREA 12: Cleanup - Remove Dead Code & Temp Dirs (P2-2) ⚡ URGENTE

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T12 |
| **Plan Item** | P2-2 |
| **Prioridad** | ⚡ URGENTE (promovido de P2) |
| **Impacto** | 168+ directorios temporales, ~500MB+ espacio |

### Problema
El repositorio tiene acumulación de:
- 168+ directorios `tmpclaude-*-cwd` (residuos de ejecuciones)
- Múltiples `__pycache__` anidados
- Archivos `.pyc` huérfanos
- Documentos obsoletos sin archivar
- Servicios duplicados (`paper_trading_v1`, `v2`, `v3`)

### Solución

#### Script PowerShell (Windows)

```powershell
# scripts/cleanup_project.ps1
# Ejecutar desde la raíz del proyecto
# USO: .\scripts\cleanup_project.ps1 -WhatIf  # Para preview
#      .\scripts\cleanup_project.ps1          # Para ejecutar

param(
    [switch]$WhatIf = $false
)

$ErrorActionPreference = "Stop"
$totalRemoved = 0
$totalSize = 0

Write-Host "🧹 CLEANUP DEL PROYECTO USDCOP-RL-Models" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# 1. Eliminar directorios tmpclaude-*
Write-Host "`n📁 Buscando directorios tmpclaude-*..." -ForegroundColor Yellow
$tmpDirs = Get-ChildItem -Path . -Directory -Recurse -Filter "tmpclaude-*" -ErrorAction SilentlyContinue
if ($tmpDirs) {
    $count = $tmpDirs.Count
    $size = ($tmpDirs | Get-ChildItem -Recurse -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum / 1MB
    Write-Host "  Encontrados: $count directorios (~$([math]::Round($size, 2)) MB)" -ForegroundColor Red

    if (-not $WhatIf) {
        $tmpDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Eliminados" -ForegroundColor Green
    } else {
        Write-Host "  [WhatIf] Se eliminarían $count directorios" -ForegroundColor Gray
    }
    $totalRemoved += $count
    $totalSize += $size
}

# 2. Eliminar __pycache__
Write-Host "`n📁 Buscando __pycache__..." -ForegroundColor Yellow
$pycacheDirs = Get-ChildItem -Path . -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue
if ($pycacheDirs) {
    $count = $pycacheDirs.Count
    Write-Host "  Encontrados: $count directorios" -ForegroundColor Red

    if (-not $WhatIf) {
        $pycacheDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Eliminados" -ForegroundColor Green
    }
    $totalRemoved += $count
}

# 3. Eliminar .pyc huérfanos
Write-Host "`n📄 Buscando archivos .pyc..." -ForegroundColor Yellow
$pycFiles = Get-ChildItem -Path . -Recurse -Filter "*.pyc" -ErrorAction SilentlyContinue
if ($pycFiles) {
    $count = $pycFiles.Count
    Write-Host "  Encontrados: $count archivos" -ForegroundColor Red

    if (-not $WhatIf) {
        $pycFiles | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Eliminados" -ForegroundColor Green
    }
}

# 4. Eliminar .pytest_cache
Write-Host "`n📁 Buscando .pytest_cache..." -ForegroundColor Yellow
$pytestDirs = Get-ChildItem -Path . -Directory -Recurse -Filter ".pytest_cache" -ErrorAction SilentlyContinue
if ($pytestDirs) {
    $count = $pytestDirs.Count
    Write-Host "  Encontrados: $count directorios" -ForegroundColor Red

    if (-not $WhatIf) {
        $pytestDirs | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Eliminados" -ForegroundColor Green
    }
    $totalRemoved += $count
}

# 5. Eliminar node_modules/.cache
Write-Host "`n📁 Buscando node_modules/.cache..." -ForegroundColor Yellow
$nmCache = Get-ChildItem -Path . -Directory -Recurse -Filter ".cache" -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -like "*node_modules*" }
if ($nmCache) {
    $count = $nmCache.Count
    Write-Host "  Encontrados: $count directorios" -ForegroundColor Red

    if (-not $WhatIf) {
        $nmCache | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "  ✅ Eliminados" -ForegroundColor Green
    }
}

# 6. Crear directorio archive/ si no existe
Write-Host "`n📂 Verificando directorio archive/..." -ForegroundColor Yellow
if (-not (Test-Path "archive")) {
    if (-not $WhatIf) {
        New-Item -ItemType Directory -Path "archive" | Out-Null
        Write-Host "  ✅ Creado archive/" -ForegroundColor Green
    } else {
        Write-Host "  [WhatIf] Se crearía archive/" -ForegroundColor Gray
    }
}

# 7. Mover docs obsoletos a archive/
$obsoleteDocs = @(
    "CLEANUP_COMPLETE_SUMMARY.md",
    "NEXT_STEPS_COMPLETE.md",
    "REPLAY_SYSTEM_*.md"
)

Write-Host "`n📄 Archivando documentos obsoletos..." -ForegroundColor Yellow
foreach ($pattern in $obsoleteDocs) {
    $files = Get-ChildItem -Path . -Filter $pattern -ErrorAction SilentlyContinue
    foreach ($file in $files) {
        if (-not $WhatIf) {
            Move-Item $file.FullName "archive/" -Force -ErrorAction SilentlyContinue
            Write-Host "  ✅ Archivado: $($file.Name)" -ForegroundColor Green
        } else {
            Write-Host "  [WhatIf] Se archivaría: $($file.Name)" -ForegroundColor Gray
        }
    }
}

# Resumen
Write-Host "`n=========================================" -ForegroundColor Cyan
Write-Host "📊 RESUMEN" -ForegroundColor Cyan
Write-Host "  Directorios procesados: $totalRemoved" -ForegroundColor White
Write-Host "  Espacio estimado liberado: ~$([math]::Round($totalSize, 2)) MB" -ForegroundColor White

if ($WhatIf) {
    Write-Host "`n⚠️  Ejecutado en modo WhatIf - nada fue eliminado" -ForegroundColor Yellow
    Write-Host "   Ejecutar sin -WhatIf para aplicar cambios" -ForegroundColor Yellow
}

Write-Host "`n✅ Cleanup completado" -ForegroundColor Green
```

#### Script Bash (Linux/Mac)

```bash
#!/bin/bash
# scripts/cleanup_project.sh
# Ejecutar desde la raíz del proyecto
# USO: ./scripts/cleanup_project.sh --dry-run  # Para preview
#      ./scripts/cleanup_project.sh            # Para ejecutar

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
fi

echo "🧹 CLEANUP DEL PROYECTO USDCOP-RL-Models"
echo "========================================="

# 1. Eliminar directorios tmpclaude-*
echo -e "\n📁 Buscando directorios tmpclaude-*..."
TMP_DIRS=$(find . -type d -name "tmpclaude-*" 2>/dev/null)
TMP_COUNT=$(echo "$TMP_DIRS" | grep -c "tmpclaude" || echo 0)

if [[ $TMP_COUNT -gt 0 ]]; then
    echo "  Encontrados: $TMP_COUNT directorios"
    if [[ "$DRY_RUN" == false ]]; then
        find . -type d -name "tmpclaude-*" -exec rm -rf {} + 2>/dev/null
        echo "  ✅ Eliminados"
    else
        echo "  [dry-run] Se eliminarían $TMP_COUNT directorios"
    fi
fi

# 2. Eliminar __pycache__
echo -e "\n📁 Buscando __pycache__..."
PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
if [[ $PYCACHE_COUNT -gt 0 ]]; then
    echo "  Encontrados: $PYCACHE_COUNT directorios"
    if [[ "$DRY_RUN" == false ]]; then
        find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
        echo "  ✅ Eliminados"
    fi
fi

# 3. Eliminar .pyc
echo -e "\n📄 Buscando archivos .pyc..."
if [[ "$DRY_RUN" == false ]]; then
    find . -name "*.pyc" -delete 2>/dev/null
fi

# 4. Eliminar .pytest_cache
echo -e "\n📁 Buscando .pytest_cache..."
if [[ "$DRY_RUN" == false ]]; then
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null
fi

# 5. Crear archive/
echo -e "\n📂 Verificando directorio archive/..."
if [[ ! -d "archive" ]]; then
    if [[ "$DRY_RUN" == false ]]; then
        mkdir -p archive
        echo "  ✅ Creado archive/"
    fi
fi

echo -e "\n========================================="
echo "✅ Cleanup completado"

if [[ "$DRY_RUN" == true ]]; then
    echo -e "\n⚠️  Ejecutado en modo --dry-run - nada fue eliminado"
fi
```

### Actualizar .gitignore

```gitignore
# Agregar a .gitignore
tmpclaude-*/
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/
*.egg-info/
dist/
build/
.mypy_cache/
```

### Test de Validación

```python
# tests/unit/test_cleanup.py
import os
from pathlib import Path


def test_no_tmpclaude_directories():
    """No deben existir directorios tmpclaude-* en el repo."""
    root = Path(".")
    tmp_dirs = list(root.rglob("tmpclaude-*"))

    assert len(tmp_dirs) == 0, \
        f"Encontrados {len(tmp_dirs)} directorios tmpclaude-*: {tmp_dirs[:5]}..."


def test_gitignore_has_cleanup_patterns():
    """gitignore debe incluir patrones de cleanup."""
    gitignore = Path(".gitignore").read_text()

    required_patterns = [
        "tmpclaude-*/",
        "__pycache__/",
        "*.pyc",
        ".pytest_cache/",
    ]

    for pattern in required_patterns:
        assert pattern in gitignore, f"Falta {pattern} en .gitignore"


def test_archive_directory_exists():
    """Directorio archive/ debe existir para docs obsoletos."""
    assert Path("archive").is_dir(), "Falta directorio archive/"
```

### Criterios de Aceptación

| # | Criterio | Verificación |
|---|----------|--------------|
| 1 | 0 directorios `tmpclaude-*` | `find . -name "tmpclaude-*"` vacío |
| 2 | 0 directorios `__pycache__` | `find . -name "__pycache__"` vacío |
| 3 | `.gitignore` actualizado | Contiene patrones de exclusión |
| 4 | `archive/` existe | Directorio creado |
| 5 | Tests de cleanup pasan | `pytest tests/unit/test_cleanup.py` |

---

## 📊 RESUMEN DE CONTRATOS PRODUCIDOS

| Contrato ID | Descripción | Consumidor |
|-------------|-------------|------------|
| **GTR-001** | `ONNXConverter` + `ONNXInferenceEngine` | GTR-004 (RiskEngine) |
| **GTR-002** | `CircuitBreaker` system | GTR-004 (RiskEngine) |
| **GTR-003** | `DriftDetector` + `DriftMonitor` | GTR-004 (RiskEngine) |
| **GTR-004** | `RiskEngine` (integración) | Production API |
| **GTR-005** | `MacroDataSource` (single source) | Training, Inference |
| **GTR-006** | `DatasetRegistry` (checksums) | Training, Reproducibility |
| **GTR-007** | `V20Config` (externalizado) | All components |
| **GTR-008** | `filterEquityData` (frontend) | Dashboard components |
| **GTR-009** | `PaperTradingValidator` | Pre-production validation |
| **GTR-010** | Cleanup Scripts | DevOps, Maintenance |

---

## ✅ CHECKLIST FINAL

### Pre-Desarrollo
- [ ] Verificar que CTR-001 a CTR-010 (de Claude) están completos
- [ ] Leer ARCHITECTURE_CONTRACTS.md
- [ ] Configurar pytest + coverage

### Por Tarea (Total: 12 tareas)
| Tarea | Item | Prioridad | Status |
|-------|------|-----------|--------|
| GEMINI-T1 | P1-14 ONNX Converter | CRÍTICA | ⬜ |
| GEMINI-T2 | P1-15 Circuit Breakers | CRÍTICA | ⬜ |
| GEMINI-T3 | P1-16 Drift Detection | ALTA | ⬜ |
| GEMINI-T4 | P1-17 Risk Engine | ALTA | ⬜ |
| GEMINI-T5 | P0-2, P0-3, P0-6 fixes | CRÍTICA | ✅ |
| GEMINI-T6 | P1-1 ExternalTaskSensor | ALTA | ⬜ |
| GEMINI-T7 | P1-7 Unificar Macro | ALTA | ⬜ |
| GEMINI-T8 | P1-9 Dataset Registry | ALTA | ⬜ |
| GEMINI-T9 | P1-10 Externalizar Config | ALTA | ⬜ |
| GEMINI-T10 | P1-4 Remove Dual Equity Filtering | MEDIA | ⬜ |
| GEMINI-T11 | P1-8 Paper Trading Validation | ALTA | ⬜ |
| **GEMINI-T12** | **P2-2 Cleanup (promovido)** | **⚡ URGENTE** | ✅ |

### Pre-Entrega
- [ ] `pytest tests/ -v` pasa al 100%
- [ ] `pytest --cov=lib/risk --cov-report=term-missing` >= 90%
- [ ] `pytest --cov=lib/inference --cov-report=term-missing` >= 90%
- [ ] `npm test` frontend pasa al 100%
- [ ] Latencia ONNX P95 < 5ms verificada
- [ ] Documentación de contratos GTR-001 a GTR-009 completa
- [ ] `config/v20_config.yaml` existe y validado
- [ ] Paper trading validator configurado

### Grafo de Dependencias

```
                    ┌─────────────────────────────────────────┐
                    │  DEPENDENCIAS DE CLAUDE                 │
                    ├─────────────────────────────────────────┤
CTR-001 (Claude) ───┬─→ GEMINI-T1 (ONNX) ────┐               │
                    │                        │               │
CTR-002 (Claude) ───┤                        ├─→ GEMINI-T4   │
                    │                        │   (Risk)      │
CTR-003 (Claude) ───┼─→ GEMINI-T3 (Drift) ───┘               │
                    │                                        │
CTR-004 (Claude) ───┤                                        │
                    │                                        │
CTR-007 (Claude) ───┼─→ GEMINI-T7 (Macro)                    │
                    │                                        │
CTR-010 (Claude) ───┴─→ GEMINI-T8, T11 (Registry, Validator) │
                    └─────────────────────────────────────────┘

GEMINI-T2 (Circuit Breakers) ─→ GEMINI-T4 (Risk Engine)

GEMINI-T5 (P0 fixes)  ─→ [independiente - PRIMERO]
GEMINI-T6 (DAGs)      ─→ [independiente]
GEMINI-T9 (Config)    ─→ [independiente]
GEMINI-T10 (Equity)   ─→ [independiente - frontend]
GEMINI-T11 (Paper)    ─→ CTR-010 (Model Registry)
```

---

## 🔄 PROTOCOLO DE SINCRONIZACIÓN

### Comunicación con Claude

```markdown
## Handoff Protocol

### Cuando Claude complete CTR-001 (FeatureBuilder):
1. Claude notifica: "CTR-001 READY"
2. Gemini verifica: import lib.features.builder funciona
3. Gemini comienza: GEMINI-T1 (ONNX), GEMINI-T7 (Macro)

### Cuando Claude complete CTR-003 (norm_stats):
1. Claude notifica: "CTR-003 READY"
2. Gemini verifica: config/v20_norm_stats.json existe y tiene 13 features
3. Gemini usa en: GEMINI-T1, GEMINI-T3

### Cuando Claude complete CTR-007 (SecuritySettings):
1. Claude notifica: "CTR-007 READY"
2. Gemini verifica: lib.config.security importable
3. Gemini usa en: GEMINI-T7 (MacroDataSource)

### Interface Contract Verification:
```python
# Gemini debe verificar antes de comenzar
def verify_claude_contracts():
    from lib.features.builder import FeatureBuilder
    from lib.features.contract import FEATURE_CONTRACT_V20
    from lib.config.security import SecuritySettings

    builder = FeatureBuilder(version="v20")
    assert builder.get_observation_dim() == 15
    assert len(builder.get_feature_names()) == 15

    # SecuritySettings existe pero no requiere valores en tests
    assert hasattr(SecuritySettings, 'get_postgres_url')

    print("✅ CTR-001, CTR-002 y CTR-007 verificados")
```
```

---

*Documento generado: 2026-01-11*
*Actualizado: 2026-01-11 (v1.2 - agregadas tareas T10-T11, total 11 tareas)*
*Metodología: Spec-Driven + AI-Augmented TDD*
*Coordinación: Contratos GTR-001 a GTR-009 con CLAUDE_TASKS.md*


---

## 🎯 TAREA 13: Dynamic Model API - Lista de Modelos desde BD (P1-19)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T13 |
| **Plan Item** | P1-19 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 3 SP |
| **Consume Contratos** | CTR-011 (ModelAutoRegister de Claude) |
| **Produce Contrato** | GTR-010 |
| **Bloquea** | GEMINI-T14 (Replay Integration) |

### Objetivo
Modificar GET /api/models para leer modelos dinamicamente de `config.models` en lugar de lista estatica hardcodeada. Esto permite que nuevos modelos registrados automaticamente aparezcan en el frontend.

### Estado Actual (PROBLEMA)

El endpoint actual tiene modelos hardcodeados:

```typescript
// usdcop-trading-dashboard/app/api/models/route.ts - ACTUAL
const STATIC_MODELS = [
  { id: "ppo_v19_prod", dbModelId: "ppo_v1", ... },  // HARDCODED
  { id: "ppo_v20_prod", dbModelId: "ppo_v20_macro", ... },  // HARDCODED
];
```

### Especificacion Funcional

**GET /api/models**

Query params:
- `status`: filter by status (testing, production, deprecated)
- `algorithm`: filter by algorithm (PPO, SAC, TD3)
- `include_deprecated`: boolean (default false)

Response:
```json
{
  "models": [ModelInfo],
  "source": "database",
  "total": number,
  "filters_applied": { "status": "...", "algorithm": "..." }
}
```

### Implementacion Requerida

1. Query `config.models` table
2. Map rows to ModelInfo interface
3. Apply filters (status, algorithm)
4. Fallback to static models if DB fails

### Archivos a Modificar

| Archivo | Accion |
|---------|--------|
| `usdcop-trading-dashboard/app/api/models/route.ts` | MODIFICAR - Query BD |
| `usdcop-trading-dashboard/components/trading/ModelSelector.tsx` | CREAR |
| `usdcop-trading-dashboard/tests/api/models.test.ts` | CREAR |

---

## 🎯 TAREA 14: Replay Mode con Seleccion Dinamica de Modelos (P1-20)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T14 |
| **Plan Item** | P1-20 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Dependencias** | GEMINI-T13 (Dynamic Model API) |
| **Produce Contrato** | GTR-011 |

### Objetivo
Integrar la seleccion dinamica de modelos con el replay mode, permitiendo ejecutar backtest de cualquier modelo registrado en `config.models`.

### Flujo Completo

```
1. Usuario abre Dashboard
2. GET /api/models -> Lista modelos de config.models
3. Usuario selecciona modelo (ppo_v21_20260112)
4. Usuario selecciona rango de fechas
5. Click "Iniciar Replay"
6. POST /api/replay/load-trades { modelId, startDate, endDate }
7. API llama Python Inference Service
8. Service ejecuta backtest con modelo seleccionado
9. Trades retornados al frontend
10. Replay mode reproduce trades visualmente
```

### Modificaciones Requeridas

1. **POST /api/replay/load-trades**
   - Validar que modelo existe en config.models
   - Obtener model_path desde BD
   - Pasar model_path al Python Inference Service

2. **Python Inference Service /v1/backtest**
   - Aceptar model_path dinamico
   - Cargar modelo desde path especificado
   - Ejecutar backtest

### Archivos a Modificar

| Archivo | Accion |
|---------|--------|
| `usdcop-trading-dashboard/app/api/replay/load-trades/route.ts` | MODIFICAR |
| `services/inference_api/routes/backtest.py` | MODIFICAR |
| `usdcop-trading-dashboard/hooks/useReplay.ts` | MODIFICAR |

---

## 🎯 TAREA 15: Python Inference Service Completo (P1-21)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | GEMINI-T15 |
| **Plan Item** | P1-21 |
| **Prioridad** | CRITICA |
| **Esfuerzo Estimado** | 8 SP |
| **Dependencias** | GEMINI-T13, GEMINI-T14 |
| **Produce Contrato** | GTR-012 |

### Objetivo
Implementar el Python Inference Service completo en `services/inference_api/` que:
1. Expone endpoint /v1/backtest para ejecutar backtests
2. Carga modelos dinamicamente desde model_path
3. Cachea resultados en Redis
4. Calcula metricas de performance

### Endpoints Requeridos

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| /health | GET | Health check |
| /v1/models | GET | Lista modelos disponibles |
| /v1/backtest | POST | Ejecuta backtest |
| /v1/inference | POST | Inference single observation |

### Archivos a Crear

| Archivo | Descripcion |
|---------|-------------|
| `services/inference_api/main.py` | FastAPI app |
| `services/inference_api/routes/backtest.py` | Backtest endpoint |
| `services/inference_api/routes/inference.py` | Inference endpoint |
| `services/inference_api/services/model_loader.py` | Carga modelos |
| `services/inference_api/services/backtest_engine.py` | Motor backtest |
| `services/inference_api/Dockerfile` | Container |
| `services/inference_api/requirements.txt` | Dependencies |

---

## CHECKLIST ACTUALIZADO GEMINI

### Por Tarea (Total: 15 tareas)
| Tarea | Item | Prioridad | Status |
|-------|------|-----------|--------|
| GEMINI-T1 | P1-14 ONNX Converter | CRITICA | No iniciado |
| GEMINI-T2 | P1-15 Circuit Breakers | CRITICA | No iniciado |
| GEMINI-T3 | P1-16 Drift Detection | ALTA | No iniciado |
| GEMINI-T4 | P1-17 Risk Engine | ALTA | No iniciado |
| GEMINI-T5 | P0-2, P0-3, P0-6 fixes | CRITICA | COMPLETADO |
| GEMINI-T6 | P1-1 ExternalTaskSensor | ALTA | No iniciado |
| GEMINI-T7 | P1-7 Unificar Macro | ALTA | No iniciado |
| GEMINI-T8 | P1-9 Dataset Registry | ALTA | No iniciado |
| GEMINI-T9 | P1-10 Externalizar Config | ALTA | No iniciado |
| GEMINI-T10 | P1-4 Remove Dual Equity Filtering | MEDIA | No iniciado |
| GEMINI-T11 | P1-8 Paper Trading Validation | ALTA | No iniciado |
| GEMINI-T12 | P2-2 Cleanup | URGENTE | COMPLETADO |
| **GEMINI-T13** | **P1-19 Dynamic Model API** | **ALTA** | No iniciado |
| **GEMINI-T14** | **P1-20 Replay Model Selection** | **ALTA** | No iniciado |
| **GEMINI-T15** | **P1-21 Python Inference Service** | **CRITICA** | No iniciado |

### Nuevos Contratos Producidos
| Contrato ID | Descripcion | Consumidor |
|-------------|-------------|------------|
| GTR-010 | Dynamic Model API | Frontend |
| GTR-011 | Replay Model Selection | Frontend |
| GTR-012 | Python Inference Service | API, Replay |

### Grafo de Dependencias Actualizado

```
CTR-011 (Claude Auto-Register) --> GEMINI-T13 (Dynamic Model API)
                                         |
                                         v
                                  GEMINI-T14 (Replay Selection)
                                         |
                                         v
                                  GEMINI-T15 (Inference Service)
```

---

*Actualizado: 2026-01-12 (v1.3 - agregadas tareas T13-T15 Model Registry Flow)*
