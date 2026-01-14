# CLAUDE TASKS - Core Feature Contract & Training Pipeline
# Proyecto: USDCOP-RL-Models

**Version**: 1.0  
**Fecha**: 2026-01-11  
**Agente Asignado**: Claude  
**Dominio de Responsabilidad**: Feature Engineering, Training Pipeline, Data Integrity  
**Metodología**: Spec-Driven + AI-Augmented TDD  

---

## 📋 RESUMEN EJECUTIVO

### Alcance de Responsabilidad
Claude es responsable de construir el **núcleo del Feature Contract Pattern** y **correcciones críticas de frontend/seguridad**, incluyendo:
- Implementación de `FeatureBuilder` como Single Source of Truth
- Calculators individuales (RSI, ATR, ADX, Returns, Macro)
- Corrección de bugs críticos en pipeline de datos (P0-9, P0-10, P0-11)
- Correcciones de seguridad (P0-4 passwords, P0-5 frontend)
- Correcciones de frontend (P1-6 confidence hardcode)
- Tests de paridad training/inference
- Migración de BD para `features_snapshot`

### Contratos de Salida (Para Gemini)
| Artefacto | Formato | Ubicación | Contrato ID |
|-----------|---------|-----------|-------------|
| `FeatureBuilder` class | Python module | `lib/features/builder.py` | CTR-001 |
| `FEATURE_CONTRACT_V20` | Python dict (frozen) | `lib/features/contract.py` | CTR-002 |
| `v20_norm_stats.json` | JSON file | `config/v20_norm_stats.json` | CTR-003 |
| `features_snapshot` schema | SQL migration | `migrations/` | CTR-004 |
| Calculators API | Python modules | `lib/features/calculators/` | CTR-005 |
| `safe_merge_macro` | Python function | `lib/data/safe_merge.py` | CTR-006 |
| `SecuritySettings` | Python class | `lib/config/security.py` | CTR-007 |

### Métricas de Éxito Global
| Métrica | Target | Crítico | Medición |
|---------|--------|---------|----------|
| Test Coverage `lib/features/` | ≥ 95% | ≥ 80% | `pytest --cov` |
| Latencia por observación | < 10ms | < 50ms | Benchmark test |
| Zero NaN/Inf en outputs | 100% | 100% | Unit tests |
| Paridad training/inference | 100% bitwise | 99.99% | Parity tests |
| Zero hardcoded passwords | 100% | 100% | Security scan |
| Zero data leakage | 100% | 100% | Anti-leakage tests |

---

## 🎯 TAREA 1: Feature Contract Core (P1-13)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T1 |
| **Plan Item** | P1-13 |
| **Prioridad** | CRÍTICA |
| **Esfuerzo Estimado** | 8 SP |
| **Dependencias** | P0-1 (norm_stats), P0-8 (feat_snap), P0-9 (look-ahead) |
| **Produce Contrato** | CTR-001, CTR-002 |

### Objetivo
Implementar el **Feature Contract Pattern** como Single Source of Truth para cálculo de features. Este componente será consumido por Gemini para ONNX inference y Risk Engine.

### Especificación Funcional

#### Archivo: `lib/features/contract.py`
```python
"""
Feature Contract V20 - Frozen Specification
INMUTABLE: No modificar. Crear V21 si es necesario.
Contrato ID: CTR-002
"""

from typing import Dict, List, TypedDict, Final
from dataclasses import dataclass

class TechnicalPeriods(TypedDict):
    rsi: int
    atr: int
    adx: int

class TradingHours(TypedDict):
    start: str  # HH:MM UTC
    end: str    # HH:MM UTC

@dataclass(frozen=True)
class FeatureContractV20:
    version: str = "v20"
    observation_dim: int = 15
    feature_order: tuple = (
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized"
    )
    norm_stats_path: str = "config/v20_norm_stats.json"
    clip_range: tuple = (-5.0, 5.0)
    trading_hours_utc: TradingHours = {"start": "13:00", "end": "19:00"}
    technical_periods: TechnicalPeriods = {"rsi": 9, "atr": 10, "adx": 14}
    warmup_bars: int = 14  # max(rsi, atr, adx)
    created_at: str = "2026-01-11"

FEATURE_CONTRACT_V20: Final = FeatureContractV20()

def get_contract(version: str = "v20") -> FeatureContractV20:
    """Factory para obtener contratos por versión."""
    contracts = {"v20": FEATURE_CONTRACT_V20}
    if version not in contracts:
        raise ValueError(f"Contrato {version} no existe. Disponibles: {list(contracts.keys())}")
    return contracts[version]
```

#### Archivo: `lib/features/builder.py`
```python
"""
FeatureBuilder - Single Source of Truth para observaciones
Contrato ID: CTR-001
Consumido por: Training, Inference API, Replay API, Airflow DAGs
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from .contract import get_contract, FeatureContractV20
from .calculators import returns, rsi, atr, adx, macro
from ..config.loader import load_norm_stats

class FeatureBuilder:
    """
    Constructor de observaciones siguiendo Feature Contract V20.
    
    Invariantes:
    - observation siempre tiene shape=(15,)
    - observation nunca contiene NaN o Inf
    - features normalizadas están en [-5.0, 5.0]
    - mismo input produce exactamente mismo output (determinista)
    
    Ejemplo:
        builder = FeatureBuilder(version="v20")
        obs = builder.build_observation(ohlcv, macro, position, timestamp, bar_idx)
    """
    
    def __init__(self, version: str = "v20"):
        self.contract = get_contract(version)
        self.norm_stats = load_norm_stats(self.contract.norm_stats_path)
        self._validate_norm_stats()
    
    def _validate_norm_stats(self) -> None:
        """Valida que norm_stats contenga todas las features requeridas."""
        required = set(self.contract.feature_order[:13])  # Excluyendo position, time_normalized
        available = set(self.norm_stats.keys())
        missing = required - available
        if missing:
            raise ValueError(f"norm_stats missing features: {missing}")
    
    def get_observation_dim(self) -> int:
        """Retorna dimensión del observation space."""
        return self.contract.observation_dim
    
    def get_feature_names(self) -> Tuple[str, ...]:
        """Retorna nombres de features en orden exacto del contrato."""
        return self.contract.feature_order
    
    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int
    ) -> np.ndarray:
        """
        Construye observation array siguiendo Feature Contract V20.
        
        Args:
            ohlcv: DataFrame con columns [open, high, low, close, volume], index=datetime
            macro: DataFrame con columns [dxy, vix, embi, brent, usdmxn, rate_spread], index=datetime
            position: Posición actual en [-1, 1]
            timestamp: Timestamp UTC de la barra actual
            bar_idx: Índice de la barra (para warmup validation)
        
        Returns:
            np.ndarray de shape (15,) con features normalizadas
        
        Raises:
            ValueError: Si bar_idx < warmup_bars o position fuera de rango
        """
        # Validaciones
        self._validate_inputs(ohlcv, macro, position, bar_idx)
        
        # Calcular features técnicas
        raw_features = self._calculate_raw_features(ohlcv, macro, bar_idx)
        
        # Normalizar (excepto position y time_normalized)
        normalized = self._normalize_features(raw_features)
        
        # Agregar position y time_normalized
        normalized["position"] = np.clip(position, -1.0, 1.0)
        normalized["time_normalized"] = self._compute_time_normalized(timestamp)
        
        # Construir array en orden del contrato
        observation = self._assemble_observation(normalized)
        
        # Validación final
        assert observation.shape == (self.contract.observation_dim,)
        assert not np.isnan(observation).any(), "NaN detectado en observation"
        assert not np.isinf(observation).any(), "Inf detectado en observation"
        
        return observation
    
    def _validate_inputs(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        position: float,
        bar_idx: int
    ) -> None:
        """Valida inputs antes de procesar."""
        if bar_idx < self.contract.warmup_bars:
            raise ValueError(
                f"bar_idx ({bar_idx}) < warmup_bars ({self.contract.warmup_bars})"
            )
        if not -1.0 <= position <= 1.0:
            raise ValueError(f"position ({position}) fuera de rango [-1, 1]")
        
        required_ohlcv = {"open", "high", "low", "close", "volume"}
        if not required_ohlcv.issubset(ohlcv.columns):
            raise ValueError(f"ohlcv missing columns: {required_ohlcv - set(ohlcv.columns)}")
    
    def _calculate_raw_features(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        bar_idx: int
    ) -> Dict[str, float]:
        """Calcula features crudas (sin normalizar)."""
        periods = self.contract.technical_periods
        
        return {
            # Returns
            "log_ret_5m": returns.log_return(ohlcv["close"], periods=1, bar_idx=bar_idx),
            "log_ret_1h": returns.log_return(ohlcv["close"], periods=12, bar_idx=bar_idx),
            "log_ret_4h": returns.log_return(ohlcv["close"], periods=48, bar_idx=bar_idx),
            
            # Technical indicators
            "rsi_9": rsi.calculate(ohlcv["close"], period=periods["rsi"], bar_idx=bar_idx),
            "atr_pct": atr.calculate_pct(ohlcv, period=periods["atr"], bar_idx=bar_idx),
            "adx_14": adx.calculate(ohlcv, period=periods["adx"], bar_idx=bar_idx),
            
            # Macro (z-scores calculated internally)
            "dxy_z": macro.z_score(macro["dxy"], bar_idx=bar_idx),
            "dxy_change_1d": macro.change_1d(macro["dxy"], bar_idx=bar_idx),
            "vix_z": macro.z_score(macro["vix"], bar_idx=bar_idx),
            "embi_z": macro.z_score(macro["embi"], bar_idx=bar_idx),
            "brent_change_1d": macro.change_1d(macro["brent"], bar_idx=bar_idx),
            "rate_spread": macro.get_value(macro["rate_spread"], bar_idx=bar_idx),
            "usdmxn_change_1d": macro.change_1d(macro["usdmxn"], bar_idx=bar_idx),
        }
    
    def _normalize_features(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Aplica z-score normalization y clipping."""
        normalized = {}
        clip_min, clip_max = self.contract.clip_range
        
        for name, value in raw.items():
            stats = self.norm_stats.get(name)
            if stats:
                z = (value - stats["mean"]) / stats["std"]
                normalized[name] = np.clip(z, clip_min, clip_max)
            else:
                # Features que no requieren normalización adicional
                normalized[name] = np.clip(value, clip_min, clip_max)
        
        return normalized
    
    def _compute_time_normalized(self, timestamp: pd.Timestamp) -> float:
        """Normaliza timestamp a [0, 1] dentro de trading hours."""
        hours = self.contract.trading_hours_utc
        start_hour = int(hours["start"].split(":")[0])
        end_hour = int(hours["end"].split(":")[0])
        
        current_minutes = timestamp.hour * 60 + timestamp.minute
        start_minutes = start_hour * 60
        end_minutes = end_hour * 60
        
        if end_minutes <= start_minutes:
            end_minutes += 24 * 60  # Handle overnight
        
        normalized = (current_minutes - start_minutes) / (end_minutes - start_minutes)
        return np.clip(normalized, 0.0, 1.0)
    
    def _assemble_observation(self, features: Dict[str, float]) -> np.ndarray:
        """Ensambla array en orden exacto del contrato."""
        observation = np.zeros(self.contract.observation_dim, dtype=np.float32)
        
        for idx, name in enumerate(self.contract.feature_order):
            observation[idx] = features[name]
        
        return observation
    
    def export_feature_snapshot(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int
    ) -> Dict:
        """
        Exporta snapshot completo para auditoría y BD.
        
        Returns:
            Dict JSON-serializable con:
            - raw_features: valores sin normalizar
            - normalized_features: valores normalizados
            - metadata: version, timestamp, bar_idx
        """
        raw = self._calculate_raw_features(ohlcv, macro, bar_idx)
        normalized = self._normalize_features(raw)
        normalized["position"] = np.clip(position, -1.0, 1.0)
        normalized["time_normalized"] = self._compute_time_normalized(timestamp)
        
        return {
            "version": self.contract.version,
            "timestamp": timestamp.isoformat(),
            "bar_idx": bar_idx,
            "raw_features": {k: float(v) for k, v in raw.items()},
            "normalized_features": {k: float(v) for k, v in normalized.items()},
        }
```

### Tests (TDD - Escribir PRIMERO)

```python
# tests/unit/test_feature_builder.py
import pytest
import numpy as np
import pandas as pd
import json
from lib.features.builder import FeatureBuilder
from lib.features.contract import get_contract, FEATURE_CONTRACT_V20


class TestFeatureBuilderContract:
    """
    Tests del Feature Contract Pattern.
    CLAUDE-T1 | Plan Item: P1-13
    Coverage Target: 95%
    """

    @pytest.fixture
    def builder(self):
        """FeatureBuilder con contrato V20."""
        return FeatureBuilder(version="v20")

    @pytest.fixture
    def sample_ohlcv(self):
        """100 barras de 5min de datos sintéticos reproducibles."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-10 13:00", periods=100, freq="5min", tz="UTC")
        base_price = 4250.0
        
        close = base_price + np.cumsum(np.random.randn(100) * 5)
        high = close + np.abs(np.random.randn(100) * 3)
        low = close - np.abs(np.random.randn(100) * 3)
        open_ = close + np.random.randn(100) * 2
        
        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, 100),
        }, index=dates)

    @pytest.fixture
    def sample_macro(self):
        """Datos macro alineados con ohlcv."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-10 13:00", periods=100, freq="5min", tz="UTC")
        return pd.DataFrame({
            "dxy": np.random.uniform(103, 105, 100),
            "vix": np.random.uniform(15, 25, 100),
            "embi": np.random.uniform(300, 400, 100),
            "brent": np.random.uniform(70, 80, 100),
            "usdmxn": np.random.uniform(17, 18, 100),
            "rate_spread": np.random.uniform(5, 7, 100),
        }, index=dates)

    # ══════════════════════════════════════════════════════════════
    # TEST 1: Dimensión correcta
    # Criterio: observation_dim DEBE ser exactamente 15 para V20
    # ══════════════════════════════════════════════════════════════
    def test_observation_dimension_matches_contract(self, builder):
        """observation_dim DEBE ser 15 para V20."""
        contract = get_contract("v20")
        
        assert builder.get_observation_dim() == 15
        assert builder.get_observation_dim() == contract.observation_dim

    # ══════════════════════════════════════════════════════════════
    # TEST 2: Orden de features exacto
    # Criterio: Orden DEBE coincidir byte-a-byte con contrato
    # ══════════════════════════════════════════════════════════════
    def test_feature_order_matches_contract_exactly(self, builder):
        """Orden de features DEBE coincidir exactamente con contrato."""
        expected = FEATURE_CONTRACT_V20.feature_order
        actual = builder.get_feature_names()
        
        assert actual == expected, f"Orden incorrecto:\nExpected: {expected}\nActual: {actual}"

    # ══════════════════════════════════════════════════════════════
    # TEST 3: Sin NaN ni Inf
    # Criterio: Observaciones NUNCA deben contener NaN o Inf
    # ══════════════════════════════════════════════════════════════
    def test_observation_no_nan_or_inf_across_all_bars(self, builder, sample_ohlcv, sample_macro):
        """Observaciones NUNCA deben contener NaN o Inf en ninguna barra."""
        warmup = FEATURE_CONTRACT_V20.warmup_bars
        
        for bar_idx in range(warmup, len(sample_ohlcv)):
            obs = builder.build_observation(
                ohlcv=sample_ohlcv,
                macro=sample_macro,
                position=0.0,
                timestamp=sample_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            
            assert not np.isnan(obs).any(), f"NaN encontrado en bar {bar_idx}: {obs}"
            assert not np.isinf(obs).any(), f"Inf encontrado en bar {bar_idx}: {obs}"

    # ══════════════════════════════════════════════════════════════
    # TEST 4: Clipping correcto [-5.0, 5.0]
    # Criterio: Features normalizadas (0-12) en rango [-5, 5]
    # ══════════════════════════════════════════════════════════════
    def test_normalized_features_clipped_to_range(self, builder, sample_ohlcv, sample_macro):
        """Features normalizadas DEBEN estar en [-5.0, 5.0]."""
        obs = builder.build_observation(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=0.0,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )
        
        # Features 0-12 son normalizadas
        normalized_features = obs[:13]
        
        assert np.all(normalized_features >= -5.0), \
            f"Feature < -5.0 detectada: {normalized_features[normalized_features < -5.0]}"
        assert np.all(normalized_features <= 5.0), \
            f"Feature > 5.0 detectada: {normalized_features[normalized_features > 5.0]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 5: Determinismo (mismo input → mismo output)
    # Criterio: Resultados idénticos bit-a-bit para mismo input
    # ══════════════════════════════════════════════════════════════
    def test_determinism_same_input_same_output(self, builder, sample_ohlcv, sample_macro):
        """Mismo input DEBE producir exactamente mismo output."""
        kwargs = {
            "ohlcv": sample_ohlcv,
            "macro": sample_macro,
            "position": 0.5,
            "timestamp": sample_ohlcv.index[50],
            "bar_idx": 50
        }
        
        obs1 = builder.build_observation(**kwargs)
        obs2 = builder.build_observation(**kwargs)
        obs3 = builder.build_observation(**kwargs)
        
        np.testing.assert_array_equal(obs1, obs2)
        np.testing.assert_array_equal(obs2, obs3)

    # ══════════════════════════════════════════════════════════════
    # TEST 6: Position en rango [-1, 1]
    # Criterio: Position (idx=13) DEBE reflejar input clipped
    # ══════════════════════════════════════════════════════════════
    @pytest.mark.parametrize("position,expected", [
        (-1.0, -1.0),
        (-0.5, -0.5),
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (-1.5, -1.0),  # Clipped
        (1.5, 1.0),    # Clipped
    ])
    def test_position_clipped_correctly(self, builder, sample_ohlcv, sample_macro, position, expected):
        """Position DEBE estar clipped en [-1, 1]."""
        obs = builder.build_observation(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=position,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )
        
        assert obs[13] == expected, f"Position index 13: expected {expected}, got {obs[13]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 7: time_normalized en [0, 1]
    # Criterio: time_normalized (idx=14) DEBE estar en [0, 1]
    # ══════════════════════════════════════════════════════════════
    def test_time_normalized_in_valid_range(self, builder, sample_ohlcv, sample_macro):
        """time_normalized DEBE estar en [0, 1]."""
        for bar_idx in range(14, len(sample_ohlcv)):
            obs = builder.build_observation(
                ohlcv=sample_ohlcv,
                macro=sample_macro,
                position=0.0,
                timestamp=sample_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            
            assert 0.0 <= obs[14] <= 1.0, f"time_normalized fuera de rango en bar {bar_idx}: {obs[14]}"

    # ══════════════════════════════════════════════════════════════
    # TEST 8: features_snapshot JSON serializable
    # Criterio: Snapshot DEBE ser serializable sin pérdida
    # ══════════════════════════════════════════════════════════════
    def test_features_snapshot_json_serializable(self, builder, sample_ohlcv, sample_macro):
        """features_snapshot DEBE ser serializable a JSON."""
        snapshot = builder.export_feature_snapshot(
            ohlcv=sample_ohlcv,
            macro=sample_macro,
            position=0.0,
            timestamp=sample_ohlcv.index[50],
            bar_idx=50
        )
        
        # No debe lanzar excepción
        json_str = json.dumps(snapshot)
        
        # Round-trip debe preservar datos
        recovered = json.loads(json_str)
        assert recovered["version"] == "v20"
        assert len(recovered["raw_features"]) == 13
        assert len(recovered["normalized_features"]) == 15

    # ══════════════════════════════════════════════════════════════
    # TEST 9: Warmup validation
    # Criterio: bar_idx < warmup_bars DEBE lanzar ValueError
    # ══════════════════════════════════════════════════════════════
    def test_warmup_validation_raises_on_insufficient_bars(self, builder, sample_ohlcv, sample_macro):
        """bar_idx < warmup_bars DEBE lanzar ValueError."""
        with pytest.raises(ValueError, match="warmup_bars"):
            builder.build_observation(
                ohlcv=sample_ohlcv,
                macro=sample_macro,
                position=0.0,
                timestamp=sample_ohlcv.index[5],
                bar_idx=5  # < 14 warmup bars
            )

    # ══════════════════════════════════════════════════════════════
    # TEST 10: Shape consistency
    # Criterio: Output shape SIEMPRE (15,) dtype float32
    # ══════════════════════════════════════════════════════════════
    def test_output_shape_and_dtype_consistent(self, builder, sample_ohlcv, sample_macro):
        """Output DEBE tener shape (15,) y dtype float32."""
        for bar_idx in range(14, 50):
            obs = builder.build_observation(
                ohlcv=sample_ohlcv,
                macro=sample_macro,
                position=np.random.uniform(-1, 1),
                timestamp=sample_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            
            assert obs.shape == (15,), f"Shape incorrecto: {obs.shape}"
            assert obs.dtype == np.float32, f"Dtype incorrecto: {obs.dtype}"


class TestFeatureBuilderEdgeCases:
    """Tests de edge cases y manejo de errores."""

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="v20")

    def test_missing_ohlcv_columns_raises_error(self, builder):
        """DataFrame sin columnas requeridas DEBE lanzar ValueError."""
        bad_ohlcv = pd.DataFrame({"close": [1, 2, 3]})
        bad_macro = pd.DataFrame({"dxy": [1, 2, 3]})
        
        with pytest.raises(ValueError, match="missing columns"):
            builder.build_observation(
                ohlcv=bad_ohlcv,
                macro=bad_macro,
                position=0.0,
                timestamp=pd.Timestamp("2026-01-10 13:00", tz="UTC"),
                bar_idx=20
            )

    def test_invalid_contract_version_raises_error(self):
        """Versión de contrato inválida DEBE lanzar ValueError."""
        with pytest.raises(ValueError, match="no existe"):
            FeatureBuilder(version="v99")

    def test_position_out_of_range_is_clipped(self, builder):
        """Position fuera de rango DEBE ser clipped, no error."""
        # Este test verifica el comportamiento de clipping
        # El builder debe aceptar y clipear, no rechazar
        pass  # Cubierto por test_position_clipped_correctly


class TestFeatureBuilderPerformance:
    """Tests de performance y benchmarks."""

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="v20")

    @pytest.fixture
    def large_ohlcv(self):
        """1000 barras para benchmark."""
        np.random.seed(42)
        dates = pd.date_range("2026-01-01 13:00", periods=1000, freq="5min", tz="UTC")
        base_price = 4250.0
        close = base_price + np.cumsum(np.random.randn(1000) * 5)
        
        return pd.DataFrame({
            "open": close + np.random.randn(1000),
            "high": close + np.abs(np.random.randn(1000) * 3),
            "low": close - np.abs(np.random.randn(1000) * 3),
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, 1000),
        }, index=dates)

    @pytest.fixture
    def large_macro(self):
        np.random.seed(42)
        dates = pd.date_range("2026-01-01 13:00", periods=1000, freq="5min", tz="UTC")
        return pd.DataFrame({
            "dxy": np.random.uniform(103, 105, 1000),
            "vix": np.random.uniform(15, 25, 1000),
            "embi": np.random.uniform(300, 400, 1000),
            "brent": np.random.uniform(70, 80, 1000),
            "usdmxn": np.random.uniform(17, 18, 1000),
            "rate_spread": np.random.uniform(5, 7, 1000),
        }, index=dates)

    def test_latency_under_10ms(self, builder, large_ohlcv, large_macro, benchmark):
        """Latencia por observación DEBE ser < 10ms."""
        def build_single():
            return builder.build_observation(
                ohlcv=large_ohlcv,
                macro=large_macro,
                position=0.0,
                timestamp=large_ohlcv.index[500],
                bar_idx=500
            )
        
        result = benchmark(build_single)
        # pytest-benchmark reportará el tiempo
        # Assert manual: < 10ms = 0.01s
        assert benchmark.stats.stats.mean < 0.01

    def test_batch_memory_under_100mb(self, builder, large_ohlcv, large_macro):
        """Batch de 1000 observaciones DEBE usar < 100MB."""
        import tracemalloc
        
        tracemalloc.start()
        
        observations = []
        for bar_idx in range(14, 1000):
            obs = builder.build_observation(
                ohlcv=large_ohlcv,
                macro=large_macro,
                position=0.0,
                timestamp=large_ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            observations.append(obs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        peak_mb = peak / 1024 / 1024
        assert peak_mb < 100, f"Peak memory: {peak_mb:.2f}MB > 100MB"
```

### Criterios de Aceptación

| # | Criterio | Verificación | Status |
|---|----------|--------------|--------|
| 1 | observation_dim == 15 | Unit test | ⬜ |
| 2 | Feature order exacto | Unit test | ⬜ |
| 3 | Sin NaN/Inf | Unit test | ⬜ |
| 4 | Clipping [-5, 5] | Unit test | ⬜ |
| 5 | Determinismo | Unit test | ⬜ |
| 6 | Position clipping | Unit test | ⬜ |
| 7 | time_normalized [0,1] | Unit test | ⬜ |
| 8 | JSON serializable | Unit test | ⬜ |
| 9 | Warmup validation | Unit test | ⬜ |
| 10 | Latencia < 10ms | Benchmark | ⬜ |
| 11 | Coverage ≥ 95% | pytest-cov | ⬜ |

---

## 🎯 TAREA 2: Calculators Individuales (P1-13 subtasks)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T2 |
| **Plan Item** | P1-13 (subtasks) |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Dependencias** | CLAUDE-T1 (contract.py) |
| **Produce Contrato** | CTR-005 |

### Objetivo
Implementar calculators individuales siguiendo Single Responsibility Principle. Cada calculator debe ser testeable de forma aislada y producir resultados deterministas.

### Especificación: `lib/features/calculators/returns.py`

```python
"""
Calculator para log returns.
Contrato: CTR-005a
"""

import numpy as np
import pandas as pd
from typing import Union


def log_return(
    close: pd.Series,
    periods: int,
    bar_idx: int
) -> float:
    """
    Calcula log return para un período dado.
    
    Args:
        close: Serie de precios de cierre
        periods: Número de barras hacia atrás (1=5min, 12=1h, 48=4h)
        bar_idx: Índice de la barra actual
    
    Returns:
        Log return como float. Retorna 0.0 si no hay suficientes datos.
    
    Example:
        ret = log_return(close_series, periods=12, bar_idx=50)  # 1h return
    """
    if bar_idx < periods:
        return 0.0
    
    current = close.iloc[bar_idx]
    previous = close.iloc[bar_idx - periods]
    
    if previous <= 0 or current <= 0:
        return 0.0
    
    return float(np.log(current / previous))
```

### Especificación: `lib/features/calculators/rsi.py`

```python
"""
Calculator para RSI (Relative Strength Index).
Contrato: CTR-005b
Período V20: 9
"""

import numpy as np
import pandas as pd


def calculate(
    close: pd.Series,
    period: int,
    bar_idx: int
) -> float:
    """
    Calcula RSI usando EMA para smoothing.
    
    Args:
        close: Serie de precios de cierre
        period: Período RSI (default V20: 9)
        bar_idx: Índice de la barra actual
    
    Returns:
        RSI en rango [0, 100]. Retorna 50.0 si no hay suficientes datos.
    
    Nota:
        Usa EMA smoothing (Wilder's method) para consistencia.
    """
    if bar_idx < period:
        return 50.0  # Neutral
    
    # Slice hasta bar_idx inclusive
    prices = close.iloc[max(0, bar_idx - period * 2):bar_idx + 1]
    
    # Calcular cambios
    delta = prices.diff()
    
    # Separar gains y losses
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    # EMA con span=period
    alpha = 1.0 / period
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return float(np.clip(rsi, 0.0, 100.0))
```

### Especificación: `lib/features/calculators/atr.py`

```python
"""
Calculator para ATR (Average True Range) como porcentaje.
Contrato: CTR-005c
Período V20: 10
"""

import numpy as np
import pandas as pd


def calculate_pct(
    ohlcv: pd.DataFrame,
    period: int,
    bar_idx: int
) -> float:
    """
    Calcula ATR como porcentaje del precio actual.
    
    Args:
        ohlcv: DataFrame con columns [open, high, low, close]
        period: Período ATR (default V20: 10)
        bar_idx: Índice de la barra actual
    
    Returns:
        ATR % en rango [0, inf). Retorna 0.0 si no hay suficientes datos.
    
    Formula:
        TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
        ATR = EMA(TR, period)
        ATR_pct = ATR / close * 100
    """
    if bar_idx < period:
        return 0.0
    
    # Slice necesario
    df = ohlcv.iloc[max(0, bar_idx - period * 2):bar_idx + 1].copy()
    
    # True Range
    df["prev_close"] = df["close"].shift(1)
    df["tr1"] = df["high"] - df["low"]
    df["tr2"] = (df["high"] - df["prev_close"]).abs()
    df["tr3"] = (df["low"] - df["prev_close"]).abs()
    df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
    
    # ATR con EMA
    atr = df["tr"].ewm(span=period, adjust=False).mean().iloc[-1]
    
    # Como porcentaje
    current_price = df["close"].iloc[-1]
    if current_price <= 0:
        return 0.0
    
    atr_pct = (atr / current_price) * 100
    
    return float(max(0.0, atr_pct))
```

### Especificación: `lib/features/calculators/adx.py`

```python
"""
Calculator para ADX (Average Directional Index).
Contrato: CTR-005d
Período V20: 14
"""

import numpy as np
import pandas as pd


def calculate(
    ohlcv: pd.DataFrame,
    period: int,
    bar_idx: int
) -> float:
    """
    Calcula ADX usando método estándar de Wilder.
    
    Args:
        ohlcv: DataFrame con columns [high, low, close]
        period: Período ADX (default V20: 14)
        bar_idx: Índice de la barra actual
    
    Returns:
        ADX en rango [0, 100]. Retorna 0.0 si no hay suficientes datos.
    
    Nota:
        Requiere 2*period + 1 barras para cálculo estable.
    """
    required_bars = period * 2 + 1
    if bar_idx < required_bars:
        return 0.0
    
    # Slice necesario
    df = ohlcv.iloc[max(0, bar_idx - period * 3):bar_idx + 1].copy()
    
    # +DM y -DM
    df["high_diff"] = df["high"].diff()
    df["low_diff"] = -df["low"].diff()
    
    df["+dm"] = np.where(
        (df["high_diff"] > df["low_diff"]) & (df["high_diff"] > 0),
        df["high_diff"],
        0.0
    )
    df["-dm"] = np.where(
        (df["low_diff"] > df["high_diff"]) & (df["low_diff"] > 0),
        df["low_diff"],
        0.0
    )
    
    # True Range
    df["prev_close"] = df["close"].shift(1)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["prev_close"]).abs(),
            (df["low"] - df["prev_close"]).abs()
        )
    )
    
    # Smoothed (Wilder's smoothing = EMA with alpha=1/period)
    alpha = 1.0 / period
    df["atr"] = df["tr"].ewm(alpha=alpha, adjust=False).mean()
    df["+di"] = 100 * df["+dm"].ewm(alpha=alpha, adjust=False).mean() / df["atr"]
    df["-di"] = 100 * df["-dm"].ewm(alpha=alpha, adjust=False).mean() / df["atr"]
    
    # DX
    di_sum = df["+di"] + df["-di"]
    df["dx"] = np.where(
        di_sum > 0,
        100 * (df["+di"] - df["-di"]).abs() / di_sum,
        0.0
    )
    
    # ADX = smoothed DX
    adx = df["dx"].ewm(alpha=alpha, adjust=False).mean().iloc[-1]
    
    return float(np.clip(adx, 0.0, 100.0))
```

### Tests para Calculators

```python
# tests/unit/test_calculators.py
import pytest
import numpy as np
import pandas as pd
from lib.features.calculators import returns, rsi, atr, adx


class TestReturnsCalculator:
    """Tests para returns.py"""

    @pytest.fixture
    def close_series(self):
        np.random.seed(42)
        return pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))

    def test_log_return_5min_correct(self, close_series):
        """Log return 5min (periods=1) calculado correctamente."""
        ret = returns.log_return(close_series, periods=1, bar_idx=50)
        expected = np.log(close_series.iloc[50] / close_series.iloc[49])
        assert np.isclose(ret, expected, rtol=1e-10)

    def test_log_return_1h_correct(self, close_series):
        """Log return 1h (periods=12) calculado correctamente."""
        ret = returns.log_return(close_series, periods=12, bar_idx=50)
        expected = np.log(close_series.iloc[50] / close_series.iloc[38])
        assert np.isclose(ret, expected, rtol=1e-10)

    def test_log_return_insufficient_data_returns_zero(self, close_series):
        """Sin suficientes datos retorna 0.0."""
        ret = returns.log_return(close_series, periods=12, bar_idx=5)
        assert ret == 0.0

    def test_log_return_no_nan(self, close_series):
        """Nunca retorna NaN."""
        for bar_idx in range(100):
            ret = returns.log_return(close_series, periods=1, bar_idx=bar_idx)
            assert not np.isnan(ret)


class TestRSICalculator:
    """Tests para rsi.py"""

    @pytest.fixture
    def close_series(self):
        np.random.seed(42)
        return pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))

    def test_rsi_in_valid_range(self, close_series):
        """RSI siempre en [0, 100]."""
        for bar_idx in range(9, 100):
            rsi_val = rsi.calculate(close_series, period=9, bar_idx=bar_idx)
            assert 0.0 <= rsi_val <= 100.0

    def test_rsi_neutral_on_insufficient_data(self, close_series):
        """RSI retorna 50 (neutral) sin suficientes datos."""
        rsi_val = rsi.calculate(close_series, period=9, bar_idx=5)
        assert rsi_val == 50.0

    def test_rsi_deterministic(self, close_series):
        """Mismo input produce mismo RSI."""
        r1 = rsi.calculate(close_series, period=9, bar_idx=50)
        r2 = rsi.calculate(close_series, period=9, bar_idx=50)
        assert r1 == r2


class TestATRCalculator:
    """Tests para atr.py"""

    @pytest.fixture
    def ohlcv(self):
        np.random.seed(42)
        close = pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))
        return pd.DataFrame({
            "open": close + np.random.randn(100),
            "high": close + np.abs(np.random.randn(100) * 3),
            "low": close - np.abs(np.random.randn(100) * 3),
            "close": close,
        })

    def test_atr_pct_non_negative(self, ohlcv):
        """ATR % siempre >= 0."""
        for bar_idx in range(10, 100):
            atr_val = atr.calculate_pct(ohlcv, period=10, bar_idx=bar_idx)
            assert atr_val >= 0.0

    def test_atr_pct_zero_on_insufficient_data(self, ohlcv):
        """ATR % retorna 0 sin suficientes datos."""
        atr_val = atr.calculate_pct(ohlcv, period=10, bar_idx=5)
        assert atr_val == 0.0


class TestADXCalculator:
    """Tests para adx.py"""

    @pytest.fixture
    def ohlcv(self):
        np.random.seed(42)
        close = pd.Series(4250 + np.cumsum(np.random.randn(100) * 5))
        return pd.DataFrame({
            "open": close + np.random.randn(100),
            "high": close + np.abs(np.random.randn(100) * 3),
            "low": close - np.abs(np.random.randn(100) * 3),
            "close": close,
        })

    def test_adx_in_valid_range(self, ohlcv):
        """ADX siempre en [0, 100]."""
        for bar_idx in range(30, 100):
            adx_val = adx.calculate(ohlcv, period=14, bar_idx=bar_idx)
            assert 0.0 <= adx_val <= 100.0

    def test_adx_zero_on_insufficient_data(self, ohlcv):
        """ADX retorna 0 sin suficientes datos."""
        adx_val = adx.calculate(ohlcv, period=14, bar_idx=10)
        assert adx_val == 0.0
```

### Criterios de Aceptación

| # | Criterio | Verificación |
|---|----------|--------------|
| 1 | log_return sin NaN | Unit test |
| 2 | RSI en [0, 100] | Unit test |
| 3 | ATR_pct >= 0 | Unit test |
| 4 | ADX en [0, 100] | Unit test |
| 5 | Todos deterministas | Unit test |
| 6 | Coverage >= 95% | pytest-cov |

---

## 🎯 TAREA 3: Corrección P0-1 (norm_stats v19→v20)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T3 |
| **Plan Item** | P0-1 |
| **Prioridad** | CRÍTICA |
| **Ubicación** | `services/inference_api/config.py:24` |
| **Dependencia de** | - |
| **Bloquea** | CLAUDE-T1, GEMINI-T1 |

### Problema
```python
# ACTUAL (INCORRECTO)
norm_stats_path: str = "config/v19_norm_stats.json"  # ❌ v19

# ESPERADO (CORRECTO)
norm_stats_path: str = "config/v20_norm_stats.json"  # ✅ v20
```

### Solución

```python
# services/inference_api/config.py
from pydantic import BaseSettings
from lib.features.contract import FEATURE_CONTRACT_V20


class Settings(BaseSettings):
    # Usar contrato como source of truth
    model_version: str = FEATURE_CONTRACT_V20.version
    norm_stats_path: str = FEATURE_CONTRACT_V20.norm_stats_path
    observation_dim: int = FEATURE_CONTRACT_V20.observation_dim
    
    class Config:
        env_prefix = "INFERENCE_"
```

### Test de Validación

```python
# tests/unit/test_p0_fixes.py

def test_p0_1_norm_stats_version_matches_contract():
    """P0-1: norm_stats path DEBE ser v20."""
    from services.inference_api.config import Settings
    from lib.features.contract import FEATURE_CONTRACT_V20
    
    settings = Settings()
    
    assert "v20" in settings.norm_stats_path
    assert settings.norm_stats_path == FEATURE_CONTRACT_V20.norm_stats_path
```

---

## 🎯 TAREA 4: Migración BD features_snapshot (P0-8)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T4 |
| **Plan Item** | P0-8 |
| **Prioridad** | CRÍTICA |
| **Produce Contrato** | CTR-004 |

### Especificación SQL

```sql
-- migrations/V003__add_features_snapshot.sql

-- Agregar columna features_snapshot
ALTER TABLE trades
ADD COLUMN features_snapshot JSONB;

-- Agregar columna model_hash
ALTER TABLE trades
ADD COLUMN model_hash VARCHAR(64);

-- Índice para queries de auditoría
CREATE INDEX idx_trades_model_hash ON trades(model_hash);

-- Constraint para nuevos trades
ALTER TABLE trades
ADD CONSTRAINT chk_snapshot_required
CHECK (
    created_at < '2026-01-15' OR  -- Trades antiguos exentos
    (features_snapshot IS NOT NULL AND model_hash IS NOT NULL)
);

-- Comentarios de documentación
COMMENT ON COLUMN trades.features_snapshot IS 
    'JSON snapshot de features al momento del trade. Schema: {version, timestamp, bar_idx, raw_features, normalized_features}';

COMMENT ON COLUMN trades.model_hash IS
    'SHA256 hash del modelo ONNX usado para la decisión';
```

### Schema del Snapshot (CTR-004)

```python
# lib/models/trade_snapshot.py
from pydantic import BaseModel
from typing import Dict
from datetime import datetime


class FeaturesSnapshot(BaseModel):
    """
    Schema para features_snapshot en BD.
    Contrato ID: CTR-004
    """
    version: str  # "v20"
    timestamp: datetime
    bar_idx: int
    raw_features: Dict[str, float]
    normalized_features: Dict[str, float]
    
    class Config:
        json_schema_extra = {
            "example": {
                "version": "v20",
                "timestamp": "2026-01-11T15:30:00Z",
                "bar_idx": 288,
                "raw_features": {
                    "log_ret_5m": -0.0012,
                    "rsi_9": 45.2,
                    # ...
                },
                "normalized_features": {
                    "log_ret_5m": -0.5,
                    "rsi_9": -0.3,
                    # ...
                }
            }
        }
```

### Test de Validación

```python
# tests/integration/test_features_snapshot_migration.py

def test_features_snapshot_schema_valid():
    """features_snapshot DEBE validar contra schema."""
    from lib.models.trade_snapshot import FeaturesSnapshot
    from lib.features.builder import FeatureBuilder
    
    builder = FeatureBuilder(version="v20")
    # ... crear snapshot
    
    # Debe parsear sin error
    validated = FeaturesSnapshot(**snapshot)
    assert validated.version == "v20"
```

---

## 🎯 TAREA 5: Fix Look-ahead Bias (P0-9)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T5 |
| **Plan Item** | P0-9 |
| **Prioridad** | CRÍTICA |
| **Ubicación** | `regime_detector.py` |

### Problema
```python
# ACTUAL (INCORRECTO - usa datos futuros)
regime = df['volatility'].rank(pct=True)  # ❌ rank() usa toda la serie

# ESPERADO (CORRECTO - solo datos históricos)
regime = expanding_percentile(df['volatility'], bar_idx)  # ✅
```

### Solución

```python
# lib/features/calculators/regime.py

import numpy as np
import pandas as pd


def expanding_percentile(
    series: pd.Series,
    bar_idx: int,
    min_periods: int = 20
) -> float:
    """
    Calcula percentil usando SOLO datos históricos (sin look-ahead).
    
    Args:
        series: Serie de valores
        bar_idx: Índice actual
        min_periods: Mínimo de datos para cálculo válido
    
    Returns:
        Percentil en [0, 1]. Retorna 0.5 si no hay suficientes datos.
    """
    if bar_idx < min_periods:
        return 0.5  # Neutral
    
    # Solo datos hasta bar_idx (inclusive)
    historical = series.iloc[:bar_idx + 1]
    current_value = historical.iloc[-1]
    
    # Percentil manual sin look-ahead
    count_below = (historical < current_value).sum()
    percentile = count_below / len(historical)
    
    return float(percentile)


def detect_regime(
    ohlcv: pd.DataFrame,
    bar_idx: int,
    volatility_period: int = 20
) -> str:
    """
    Detecta régimen de mercado sin look-ahead bias.
    
    Returns:
        'low_vol', 'medium_vol', 'high_vol'
    """
    # Calcular volatilidad histórica
    returns = np.log(ohlcv['close'] / ohlcv['close'].shift(1))
    volatility = returns.rolling(volatility_period).std()
    
    # Percentil sin look-ahead
    pct = expanding_percentile(volatility, bar_idx)
    
    if pct < 0.33:
        return 'low_vol'
    elif pct < 0.67:
        return 'medium_vol'
    else:
        return 'high_vol'
```

### Test Anti-Look-ahead

```python
# tests/unit/test_no_lookahead.py

def test_expanding_percentile_no_future_data():
    """expanding_percentile NUNCA debe usar datos futuros."""
    np.random.seed(42)
    series = pd.Series(np.random.randn(100))
    
    for bar_idx in range(20, 100):
        pct = expanding_percentile(series, bar_idx)
        
        # Verificar que solo usa datos hasta bar_idx
        historical = series.iloc[:bar_idx + 1]
        expected = (historical < historical.iloc[-1]).sum() / len(historical)
        
        assert pct == expected, f"Look-ahead detectado en bar {bar_idx}"


def test_regime_detector_no_lookahead():
    """Regime detector NUNCA debe usar datos futuros."""
    # Crear datos donde el futuro es muy diferente
    ohlcv = create_test_ohlcv(100)
    
    # Modificar datos futuros
    ohlcv_modified = ohlcv.copy()
    ohlcv_modified.iloc[51:, :] *= 10  # Cambio drástico
    
    # Régimen en bar 50 debe ser IGUAL con y sin modificación futura
    regime_original = detect_regime(ohlcv, bar_idx=50)
    regime_modified = detect_regime(ohlcv_modified, bar_idx=50)
    
    assert regime_original == regime_modified, "Look-ahead bias detectado!"
```

---

## 🎯 TAREA 6: Fix ffill sin límite (P0-10)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T6 |
| **Plan Item** | P0-10 |
| **Prioridad** | CRÍTICA |
| **Ubicación** | `01_build_5min_datasets.py:751` |

### Problema
```python
# ACTUAL (INCORRECTO)
macro_df.ffill()  # ❌ Sin límite - puede propagar datos obsoletos

# ESPERADO (CORRECTO)
macro_df.ffill(limit=144)  # ✅ Máximo 12 horas (144 barras de 5min)
```

### Test de Validación

```python
# tests/unit/test_data_pipeline.py

def test_ffill_has_limit():
    """ffill DEBE tener limit=144 (12 horas)."""
    import ast
    
    with open("scripts/01_build_5min_datasets.py") as f:
        source = f.read()
    
    tree = ast.parse(source)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'attr') and node.func.attr == 'ffill':
                # Verificar que tiene limit keyword
                has_limit = any(
                    kw.arg == 'limit' for kw in node.keywords
                )
                assert has_limit, f"ffill sin limit en línea {node.lineno}"
```

---

## 🎯 TAREA 7: Fix merge_asof tolerance Data Leakage (P0-11)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T7 |
| **Plan Item** | P0-11 |
| **Prioridad** | CRÍTICA |
| **Ubicación** | `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py:225-231` |

### Problema
```python
# ACTUAL (INCORRECTO - permite data leakage)
df = pd.merge_asof(
    df_ohlcv.sort_values('datetime'),
    df_macro_subset.sort_values('datetime'),
    on='datetime',
    direction='backward',
    tolerance=pd.Timedelta('1 day')  # ❌ Permite datos futuros
)

# ESPERADO (CORRECTO - sin tolerance)
df = pd.merge_asof(
    df_ohlcv.sort_values('datetime'),
    df_macro_subset.sort_values('datetime'),
    on='datetime',
    direction='backward'
    # SIN tolerance - solo datos disponibles al momento
)
```

### Solución

```python
# lib/data/safe_merge.py

def safe_merge_macro(
    df_ohlcv: pd.DataFrame,
    df_macro: pd.DataFrame,
    track_source: bool = True
) -> pd.DataFrame:
    """
    Merge macro data SIN data leakage.

    Args:
        df_ohlcv: OHLCV data con datetime index
        df_macro: Macro data con datetime index
        track_source: Si True, agrega columna macro_source_date

    Returns:
        DataFrame merged sin data leakage
    """
    # Asegurar que macro data esté al inicio del día
    df_macro_daily = df_macro.copy()
    df_macro_daily['datetime'] = pd.to_datetime(
        df_macro_daily['datetime'].dt.date
    )

    if track_source:
        df_macro_daily['macro_source_date'] = df_macro_daily['datetime']

    # Merge SIN tolerance
    df = pd.merge_asof(
        df_ohlcv.sort_values('datetime'),
        df_macro_daily.sort_values('datetime'),
        on='datetime',
        direction='backward'
        # NO tolerance - strict temporal ordering
    )

    # Validar no hay future data
    if track_source:
        leakage = df[df['macro_source_date'] > df['datetime']]
        if len(leakage) > 0:
            raise ValueError(
                f"Data leakage detectado: {len(leakage)} rows con macro del futuro"
            )

    return df
```

### Test de Validación

```python
# tests/unit/test_p0_11_merge_asof.py

def test_merge_asof_no_tolerance():
    """merge_asof NO debe tener tolerance parameter."""
    import ast

    with open("data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py") as f:
        source = f.read()

    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if hasattr(node.func, 'attr') and node.func.attr == 'merge_asof':
                # Verificar que NO tiene tolerance keyword
                has_tolerance = any(
                    kw.arg == 'tolerance' for kw in node.keywords
                )
                assert not has_tolerance, \
                    f"merge_asof tiene tolerance en línea {node.lineno} - DATA LEAKAGE RISK"


def test_no_future_macro_data():
    """Macro data NUNCA debe ser del futuro."""
    df = load_merged_dataset_sample()

    if 'macro_source_date' in df.columns:
        future_leak = df[df['macro_source_date'] > df['datetime']]
        assert len(future_leak) == 0, \
            f"Data leakage: {len(future_leak)} rows con macro del futuro"
```

---

## 🎯 TAREA 8: Fix Hardcoded Passwords (P0-4)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T8 |
| **Plan Item** | P0-4 |
| **Prioridad** | 🔴 CRÍTICA SEGURIDAD |
| **Ubicación** | Múltiples archivos de configuración |

### Problema
```python
# ACTUAL (INCORRECTO - passwords en código)
postgres_password: str = "default_password"  # ❌ SECURITY RISK
redis_password: str = "redis123"  # ❌ SECURITY RISK
```

### Solución

```python
# lib/config/security.py
import os
from pydantic import BaseSettings, SecretStr
from typing import Optional


class SecuritySettings(BaseSettings):
    """
    Configuración de seguridad - NUNCA defaults para passwords.
    """
    # Database
    postgres_password: SecretStr  # Required - no default
    postgres_user: str = "trading_user"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "trading_db"

    # Redis
    redis_password: Optional[SecretStr] = None
    redis_host: str = "localhost"
    redis_port: int = 6379

    # API Keys
    api_secret_key: SecretStr  # Required - no default

    class Config:
        env_prefix = "TRADING_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_postgres_url(self) -> str:
        """Construye URL de conexión PostgreSQL."""
        return (
            f"postgresql://{self.postgres_user}:"
            f"{self.postgres_password.get_secret_value()}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


# Uso
def get_db_connection():
    settings = SecuritySettings()
    # Si TRADING_POSTGRES_PASSWORD no está definido, lanza error
    return create_connection(settings.get_postgres_url())
```

### Test de Validación

```python
# tests/unit/test_p0_4_no_hardcoded_passwords.py
import os
import re
from pathlib import Path


def test_no_hardcoded_passwords_in_code():
    """Código NO debe contener passwords hardcoded."""
    # Patrones sospechosos
    patterns = [
        r'password\s*[=:]\s*["\'][^"\']+["\']',
        r'secret\s*[=:]\s*["\'][^"\']+["\']',
        r'api_key\s*[=:]\s*["\'][^"\']+["\']',
    ]

    # Archivos a revisar
    code_dirs = ["lib/", "services/", "airflow/"]

    violations = []
    for code_dir in code_dirs:
        for py_file in Path(code_dir).rglob("*.py"):
            content = py_file.read_text()
            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    violations.append((py_file, matches))

    assert len(violations) == 0, \
        f"Passwords hardcoded encontrados: {violations}"


def test_security_settings_require_env():
    """SecuritySettings DEBE requerir variables de entorno."""
    from lib.config.security import SecuritySettings
    import pytest

    # Limpiar variables de entorno
    os.environ.pop("TRADING_POSTGRES_PASSWORD", None)
    os.environ.pop("TRADING_API_SECRET_KEY", None)

    # Debe fallar sin las variables requeridas
    with pytest.raises(Exception):  # ValidationError
        SecuritySettings()
```

---

## 🎯 TAREA 9: Fix MIN_TICK_INTERVAL_MS (P0-5)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T9 |
| **Plan Item** | P0-5 |
| **Prioridad** | CRÍTICA |
| **Ubicación** | `usdcop-trading-dashboard/utils/replayPerformance.ts:299` |

### Problema
```typescript
// ACTUAL (INCORRECTO - constante no definida)
if (elapsed < PERF_THRESHOLDS.MIN_TICK_INTERVAL_MS) {  // ❌ undefined
    // ...
}
```

### Solución

```typescript
// utils/replayPerformance.ts

export const PERF_THRESHOLDS = {
    // Existing
    MAX_FRAME_TIME_MS: 16,      // 60fps target
    WARN_FRAME_TIME_MS: 33,     // 30fps warning

    // NEW - Fix P0-5
    MIN_TICK_INTERVAL_MS: 16,   // Minimum time between ticks (60fps)

    // Animation
    MAX_BATCH_SIZE: 100,
    DEBOUNCE_MS: 50,
};
```

### Test de Validación

```typescript
// tests/unit/replayPerformance.test.ts

describe('PERF_THRESHOLDS', () => {
    it('should have MIN_TICK_INTERVAL_MS defined', () => {
        expect(PERF_THRESHOLDS.MIN_TICK_INTERVAL_MS).toBeDefined();
        expect(PERF_THRESHOLDS.MIN_TICK_INTERVAL_MS).toBe(16);
    });

    it('should have all required thresholds', () => {
        const required = [
            'MAX_FRAME_TIME_MS',
            'WARN_FRAME_TIME_MS',
            'MIN_TICK_INTERVAL_MS',
            'MAX_BATCH_SIZE',
            'DEBOUNCE_MS'
        ];

        required.forEach(key => {
            expect(PERF_THRESHOLDS[key]).toBeDefined();
        });
    });
});
```

---

## 🎯 TAREA 10: Fix Confidence Hardcode Frontend (P1-6)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T10 |
| **Plan Item** | P1-6 |
| **Prioridad** | ALTA |
| **Ubicación** | `usdcop-trading-dashboard/components/charts/TradingChartWithSignals.tsx:168-180` |

### Problema
```typescript
// ACTUAL (INCORRECTO)
confidence: 75,  // ❌ Hardcoded!

// ESPERADO (CORRECTO)
confidence: trade.entry_confidence ?? trade.confidence ?? 75,
```

### Solución

```typescript
// components/charts/TradingChartWithSignals.tsx

// Actualizar mapeo de trades en replay mode
if (isReplayMode && replayTrades && replayTrades.length > 0) {
  return replayTrades.map((trade) => ({
    id: trade.trade_id,
    trade_id: trade.trade_id,
    timestamp: trade.timestamp || trade.entry_time || '',
    time: trade.timestamp || trade.entry_time || '',
    type: ['BUY', 'LONG'].includes((trade.side || '').toUpperCase()) ? 'BUY' : 'SELL',
    price: trade.entry_price,
    // FIX: Usar confidence real del trade
    confidence: trade.entry_confidence
      ?? trade.confidence
      ?? (trade.model_metadata?.confidence)
      ?? 75,  // Fallback solo si no hay datos
    // Preservar metadata adicional
    stopLoss: trade.stop_loss ?? null,
    takeProfit: trade.take_profit ?? null,
    modelVersion: trade.model_version ?? 'unknown',
    entropy: trade.model_metadata?.entropy ?? null,
    featuresSnapshot: trade.features_snapshot ?? null,
  }));
}

// Helper para color de marker basado en confidence
const getConfidenceColor = (confidence: number): string => {
  if (confidence >= 90) return '#00C853';  // Alto - verde brillante
  if (confidence >= 75) return '#4CAF50';  // Bueno - verde
  if (confidence >= 60) return '#FFC107';  // Medio - amarillo
  return '#FF5722';                         // Bajo - naranja
};
```

### Test de Validación

```typescript
// tests/components/TradingChartWithSignals.test.tsx

describe('Trade confidence mapping', () => {
    it('should use entry_confidence when available', () => {
        const trade = {
            trade_id: '123',
            entry_confidence: 85,
            confidence: 70,
        };

        const mapped = mapTradeToSignal(trade);
        expect(mapped.confidence).toBe(85);
    });

    it('should fallback to confidence when entry_confidence missing', () => {
        const trade = {
            trade_id: '123',
            confidence: 70,
        };

        const mapped = mapTradeToSignal(trade);
        expect(mapped.confidence).toBe(70);
    });

    it('should use model_metadata.confidence as third fallback', () => {
        const trade = {
            trade_id: '123',
            model_metadata: { confidence: 65 },
        };

        const mapped = mapTradeToSignal(trade);
        expect(mapped.confidence).toBe(65);
    });

    it('should fallback to 75 only when no confidence available', () => {
        const trade = { trade_id: '123' };

        const mapped = mapTradeToSignal(trade);
        expect(mapped.confidence).toBe(75);
    });
});
```

---

## 🎯 TAREA 11: Tests de Paridad Training/Inference (P1-12)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T11 |
| **Plan Item** | P1-12 |
| **Prioridad** | ALTA |
| **Dependencias** | CLAUDE-T1, GEMINI-T1 |

### Objetivo
Garantizar que las features calculadas durante training son IDÉNTICAS a las calculadas durante inference.

### Test Suite

```python
# tests/integration/test_training_inference_parity.py
import pytest
import numpy as np
import pandas as pd
from lib.features.builder import FeatureBuilder


class TestTrainingInferenceParity:
    """
    Tests CRÍTICOS de paridad training/inference.
    CLAUDE-T7 | Plan Item: P1-12
    
    Cualquier diferencia causa training/inference skew.
    """

    @pytest.fixture
    def builder(self):
        return FeatureBuilder(version="v20")

    @pytest.fixture
    def production_sample(self):
        """Cargar sample real de producción."""
        # TODO: Cargar desde data/test_fixtures/
        return load_production_sample()

    def test_bitwise_parity_on_production_data(self, builder, production_sample):
        """
        Features recalculadas DEBEN ser idénticas bit-a-bit
        a las guardadas durante training.
        """
        ohlcv, macro, saved_features = production_sample
        
        for bar_idx in range(14, len(ohlcv)):
            # Recalcular con FeatureBuilder
            obs = builder.build_observation(
                ohlcv=ohlcv,
                macro=macro,
                position=saved_features[bar_idx]["position"],
                timestamp=ohlcv.index[bar_idx],
                bar_idx=bar_idx
            )
            
            # Comparar bit-a-bit
            saved_obs = np.array(saved_features[bar_idx]["observation"])
            
            np.testing.assert_array_equal(
                obs, saved_obs,
                err_msg=f"Parity violation at bar {bar_idx}"
            )

    def test_norm_stats_hash_matches_training(self, builder):
        """
        Hash de norm_stats usado en inference DEBE coincidir
        con el usado durante training del modelo actual.
        """
        import hashlib
        import json
        
        # Hash de norm_stats actual
        with open(builder.contract.norm_stats_path) as f:
            current_hash = hashlib.sha256(f.read().encode()).hexdigest()
        
        # Hash registrado en model metadata
        model_metadata = load_model_metadata()  # TODO: implementar
        training_hash = model_metadata["norm_stats_hash"]
        
        assert current_hash == training_hash, \
            f"norm_stats hash mismatch: {current_hash[:8]} vs {training_hash[:8]}"

    def test_feature_order_matches_model_input(self, builder):
        """
        Orden de features DEBE coincidir con el esperado por el modelo.
        """
        from lib.inference.onnx_converter import get_model_input_names
        
        model_input_names = get_model_input_names()  # TODO: implementar
        builder_names = list(builder.get_feature_names())
        
        assert builder_names == model_input_names, \
            f"Feature order mismatch:\nBuilder: {builder_names}\nModel: {model_input_names}"

    def test_replay_api_uses_feature_builder(self):
        """Replay API DEBE usar FeatureBuilder."""
        import inspect
        from services.replay_api import ReplayService
        
        source = inspect.getsource(ReplayService)
        
        assert "FeatureBuilder" in source, \
            "Replay API no usa FeatureBuilder - posible skew"
        assert "feature_calculator_factory" not in source, \
            "Replay API usa código obsoleto"

    def test_airflow_dag_uses_feature_builder(self):
        """Airflow DAG DEBE usar FeatureBuilder."""
        with open("airflow/dags/l5_multi_model_inference.py") as f:
            source = f.read()
        
        assert "FeatureBuilder" in source, \
            "Airflow DAG no usa FeatureBuilder - posible skew"
```

---

## 🎯 TAREA 12: ML Workflow Disciplinado (P0-7) 🔴 CRÍTICO

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T12 |
| **Plan Item** | P0-7 |
| **Prioridad** | 🔴 CRÍTICA - RIESGO P-HACKING |
| **Audit Ref** | ML-02, ML-05, ML-06 |

### Problema
El workflow actual no tiene guardrails contra p-hacking/overfitting:
- No se documenta cuántas veces se miró el validation set
- No hay separación estricta Exploration → Validation → Test
- Riesgo de contaminación del test set

### Solución

```python
# config/hyperparameter_decisions.json
{
    "model_version": "v20",
    "created_at": "2026-01-11",
    "exploration_phase": {
        "start_date": "2025-10-01",
        "end_date": "2025-11-15",
        "experiments_run": 47,
        "validation_looks": 12,
        "notes": "Grid search sobre learning_rate, n_steps, ent_coef"
    },
    "final_hyperparameters": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "selection_criteria": "Mejor Sharpe en validation 2025-01-01 a 2025-06-30"
    },
    "validation_phase": {
        "single_pass_date": "2025-11-20",
        "validation_metrics": {
            "sharpe": 1.42,
            "max_drawdown": -0.12,
            "win_rate": 0.54
        },
        "passed": true
    },
    "test_phase": {
        "execution_date": null,
        "test_set_touched": false,
        "notes": "Test set reservado para evaluación final pre-producción"
    },
    "audit_trail": [
        {"date": "2025-10-15", "action": "Initial exploration", "validation_looks": 3},
        {"date": "2025-11-01", "action": "Refined learning_rate", "validation_looks": 5},
        {"date": "2025-11-15", "action": "Final tuning", "validation_looks": 4}
    ]
}
```

```python
# lib/ml_workflow/experiment_tracker.py
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, List


@dataclass
class ExperimentLog:
    """Registro de experimento para auditoría."""
    experiment_id: str
    timestamp: datetime
    phase: str  # 'exploration', 'validation', 'test'
    hyperparameters: dict
    validation_looked: bool
    metrics: Optional[dict] = None
    notes: str = ""


class MLWorkflowTracker:
    """
    Tracker para workflow ML disciplinado.
    Contrato: CTR-008

    Previene:
    - Múltiples looks al validation set sin documentar
    - Contaminación del test set
    - P-hacking por búsqueda exhaustiva
    """

    def __init__(self, config_path: Path = Path("config/hyperparameter_decisions.json")):
        self.config_path = config_path
        self.config = self._load_or_create_config()

    def _load_or_create_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return json.load(f)
        return {
            "model_version": "",
            "exploration_phase": {"validation_looks": 0},
            "validation_phase": {"single_pass_date": None},
            "test_phase": {"test_set_touched": False},
            "audit_trail": []
        }

    def log_validation_look(self, action: str, notes: str = ""):
        """Registra cada vez que se mira el validation set."""
        self.config["exploration_phase"]["validation_looks"] += 1
        self.config["audit_trail"].append({
            "date": datetime.now().isoformat(),
            "action": action,
            "validation_looks": self.config["exploration_phase"]["validation_looks"],
            "notes": notes
        })
        self._save_config()

        # Warning si muchos looks
        if self.config["exploration_phase"]["validation_looks"] > 20:
            print("⚠️ WARNING: >20 validation looks - alto riesgo de overfitting")

    def validate_test_set_untouched(self) -> bool:
        """Verifica que test set no ha sido tocado."""
        if self.config["test_phase"]["test_set_touched"]:
            raise ValueError(
                "🔴 TEST SET YA FUE TOCADO - Resultados no son confiables\n"
                f"Fecha: {self.config['test_phase'].get('execution_date')}"
            )
        return True

    def mark_test_execution(self, metrics: dict):
        """Marca ejecución única del test set."""
        if self.config["test_phase"]["test_set_touched"]:
            raise ValueError("Test set ya fue ejecutado - no se permite re-ejecución")

        self.config["test_phase"]["test_set_touched"] = True
        self.config["test_phase"]["execution_date"] = datetime.now().isoformat()
        self.config["test_phase"]["final_metrics"] = metrics
        self._save_config()

        print("✅ Test set ejecutado - métricas finales registradas")

    def get_exploration_summary(self) -> dict:
        """Resumen de fase de exploración."""
        return {
            "total_validation_looks": self.config["exploration_phase"]["validation_looks"],
            "audit_entries": len(self.config["audit_trail"]),
            "test_set_touched": self.config["test_phase"]["test_set_touched"]
        }

    def _save_config(self):
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
```

### Test de Validación

```python
# tests/unit/test_p0_7_ml_workflow.py
import pytest
from lib.ml_workflow.experiment_tracker import MLWorkflowTracker
from pathlib import Path
import tempfile


class TestMLWorkflowDiscipline:
    """Tests para workflow ML disciplinado."""

    @pytest.fixture
    def tracker(self):
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            return MLWorkflowTracker(Path(f.name))

    def test_validation_looks_are_tracked(self, tracker):
        """Cada look al validation set DEBE ser registrado."""
        assert tracker.config["exploration_phase"]["validation_looks"] == 0

        tracker.log_validation_look("Test look 1")
        tracker.log_validation_look("Test look 2")

        assert tracker.config["exploration_phase"]["validation_looks"] == 2
        assert len(tracker.config["audit_trail"]) == 2

    def test_test_set_protection(self, tracker):
        """Test set NO debe poder tocarse dos veces."""
        tracker.validate_test_set_untouched()  # Primera vez OK

        tracker.mark_test_execution({"sharpe": 1.5})

        # Segunda ejecución debe fallar
        with pytest.raises(ValueError, match="ya fue ejecutado"):
            tracker.mark_test_execution({"sharpe": 1.6})

    def test_hyperparameter_decisions_file_exists(self):
        """Archivo de decisiones de hiperparámetros DEBE existir."""
        config_path = Path("config/hyperparameter_decisions.json")
        assert config_path.exists(), \
            "config/hyperparameter_decisions.json no existe - crear con historial"

    def test_audit_trail_has_entries(self):
        """Audit trail DEBE tener registros de exploración."""
        config_path = Path("config/hyperparameter_decisions.json")
        if config_path.exists():
            import json
            with open(config_path) as f:
                config = json.load(f)

            assert len(config.get("audit_trail", [])) > 0, \
                "No hay registros de exploración - documentar proceso"
```

---

## 🎯 TAREA 13: Model Metadata + bid_ask_spread (P1-2)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T13 |
| **Plan Item** | P1-2 |
| **Prioridad** | ALTA |
| **Audit Ref** | ML-07, TL-02 |

### Objetivo
Capturar metadata completa del modelo incluyendo `bid_ask_spread` al momento de cada trade.

### Solución

```python
# lib/models/model_metadata.py
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
import hashlib


@dataclass
class ModelMetadata:
    """
    Metadata capturada con cada predicción del modelo.
    Contrato: CTR-009
    """
    model_id: str
    model_version: str
    model_hash: str
    norm_stats_hash: str
    config_hash: str

    # Features al momento de predicción
    observation: list  # 15 floats
    raw_features: dict

    # Market state
    bid_ask_spread: float
    market_volatility: float
    timestamp: datetime

    # Predicción
    action: int
    action_probabilities: list
    value_estimate: float
    entropy: float
    confidence: float

    def to_dict(self) -> dict:
        """Serializa para BD."""
        d = asdict(self)
        d['timestamp'] = self.timestamp.isoformat()
        return d


def capture_model_metadata(
    model,
    observation: np.ndarray,
    raw_features: dict,
    bid_ask_spread: float,
    market_volatility: float
) -> ModelMetadata:
    """
    Captura metadata completa de predicción.

    Args:
        model: Modelo ONNX o SB3
        observation: Vector de features
        raw_features: Features sin normalizar
        bid_ask_spread: Spread actual del mercado
        market_volatility: Volatilidad actual

    Returns:
        ModelMetadata con toda la información
    """
    # Obtener predicción con probabilidades
    if hasattr(model, 'predict_proba'):
        action, probs, value = model.predict_with_info(observation)
    else:
        action = model.predict(observation, deterministic=True)
        probs = [0.33, 0.34, 0.33]  # Fallback si no hay probs
        value = 0.0

    # Calcular entropy y confidence
    probs_array = np.array(probs)
    entropy = -np.sum(probs_array * np.log(probs_array + 1e-10))
    confidence = float(np.max(probs_array))

    return ModelMetadata(
        model_id=model.model_id,
        model_version=model.version,
        model_hash=model.model_hash,
        norm_stats_hash=model.norm_stats_hash,
        config_hash=model.config_hash,
        observation=observation.tolist(),
        raw_features=raw_features,
        bid_ask_spread=bid_ask_spread,
        market_volatility=market_volatility,
        timestamp=datetime.utcnow(),
        action=int(action),
        action_probabilities=probs,
        value_estimate=float(value),
        entropy=float(entropy),
        confidence=confidence
    )
```

### SQL Migration

```sql
-- migrations/V004__model_metadata.sql

-- Extender tabla trades para incluir metadata
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS model_metadata JSONB;

-- Agregar índices para queries de auditoría
CREATE INDEX IF NOT EXISTS idx_trades_model_hash
ON trades((model_metadata->>'model_hash'));

CREATE INDEX IF NOT EXISTS idx_trades_bid_ask
ON trades((model_metadata->>'bid_ask_spread'));

-- Constraint: nuevos trades DEBEN tener metadata
ALTER TABLE trades
ADD CONSTRAINT chk_model_metadata_required
CHECK (
    created_at < '2026-01-15' OR  -- Trades antiguos exentos
    model_metadata IS NOT NULL
);
```

---

## 🎯 TAREA 14: Feature Order Runtime Validation (P1-5)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T14 |
| **Plan Item** | P1-5 |
| **Prioridad** | MEDIA |
| **Dependencias** | CLAUDE-T1 |

### Objetivo
Validar en runtime que el orden de features coincide entre FeatureBuilder y modelo ONNX.

### Solución

```python
# lib/features/validation.py

def validate_feature_order_at_startup(
    builder: FeatureBuilder,
    model_path: Path
) -> bool:
    """
    Valida orden de features al iniciar el servicio.

    Raises:
        FeatureOrderMismatchError si hay diferencia
    """
    import onnx

    # Obtener orden del builder
    builder_order = list(builder.get_feature_names())

    # Obtener orden del modelo ONNX (si tiene metadata)
    model = onnx.load(str(model_path))
    model_order = None

    for prop in model.metadata_props:
        if prop.key == "feature_order":
            model_order = json.loads(prop.value)
            break

    if model_order is None:
        # Modelo sin metadata - asumir orden correcto pero log warning
        logger.warning(
            f"Modelo {model_path} sin feature_order metadata - "
            "no se puede validar orden"
        )
        return True

    if builder_order != model_order:
        raise FeatureOrderMismatchError(
            f"Feature order mismatch!\n"
            f"Builder: {builder_order}\n"
            f"Model: {model_order}\n"
            f"Esto causará predicciones incorrectas."
        )

    logger.info(f"✅ Feature order validado: {len(builder_order)} features")
    return True


class FeatureOrderMismatchError(Exception):
    """Error crítico: orden de features no coincide."""
    pass
```

### Test

```python
def test_feature_order_validated_at_startup():
    """Feature order DEBE validarse al iniciar servicio."""
    from lib.features.validation import validate_feature_order_at_startup
    from lib.features.builder import FeatureBuilder

    builder = FeatureBuilder(version="v20")
    model_path = Path("models/ppo_v20.onnx")

    # No debe lanzar excepción
    assert validate_feature_order_at_startup(builder, model_path)
```

---

## 🎯 TAREA 15: Model Hash Registration en BD (P1-11)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T15 |
| **Plan Item** | P1-11 |
| **Prioridad** | ALTA |
| **Produce Contrato** | CTR-010 |

### Solución

```sql
-- migrations/V005__model_registry.sql

CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) UNIQUE NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    model_path TEXT NOT NULL,

    -- Hashes de integridad
    model_hash VARCHAR(64) NOT NULL,
    norm_stats_hash VARCHAR(64) NOT NULL,
    config_hash VARCHAR(64) NOT NULL,

    -- Metadata
    observation_dim INTEGER NOT NULL,
    action_space INTEGER NOT NULL,
    feature_order JSONB NOT NULL,

    -- Training info
    training_dataset_id INTEGER REFERENCES dataset_registry(id),
    training_start_date DATE,
    training_end_date DATE,
    validation_metrics JSONB,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deployed_at TIMESTAMP,
    retired_at TIMESTAMP,

    -- Status
    status VARCHAR(20) DEFAULT 'registered'  -- registered, deployed, retired
);

CREATE INDEX idx_model_registry_hash ON model_registry(model_hash);
CREATE INDEX idx_model_registry_status ON model_registry(status);
```

```python
# lib/model_registry.py

class ModelRegistry:
    """
    Registro de modelos en BD para trazabilidad.
    Contrato: CTR-010
    """

    def register_model(
        self,
        model_path: Path,
        version: str,
        training_info: dict
    ) -> str:
        """Registra modelo con todos sus hashes."""
        model_hash = self._compute_hash(model_path)
        norm_stats_hash = self._compute_hash(
            Path(f"config/{version}_norm_stats.json")
        )
        config_hash = self._compute_hash(
            Path(f"config/{version}_config.yaml")
        )

        model_id = f"ppo_{version}_{model_hash[:8]}"

        self.conn.execute("""
            INSERT INTO model_registry
            (model_id, model_version, model_path, model_hash,
             norm_stats_hash, config_hash, observation_dim, action_space,
             feature_order, training_dataset_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            model_id, version, str(model_path), model_hash,
            norm_stats_hash, config_hash, 15, 3,
            json.dumps(list(FEATURE_CONTRACT_V20.feature_order)),
            training_info.get('dataset_id')
        ))

        return model_id

    def verify_model_integrity(self, model_id: str) -> bool:
        """Verifica integridad del modelo contra registro."""
        row = self.conn.fetchone(
            "SELECT model_path, model_hash FROM model_registry WHERE model_id = %s",
            (model_id,)
        )

        if not row:
            raise ValueError(f"Modelo {model_id} no registrado")

        current_hash = self._compute_hash(Path(row['model_path']))

        if current_hash != row['model_hash']:
            raise ValueError(
                f"Model integrity check FAILED!\n"
                f"Expected: {row['model_hash']}\n"
                f"Actual: {current_hash}"
            )

        return True
```

---

## 📊 RESUMEN DE CONTRATOS PRODUCIDOS

| Contrato ID | Descripción | Consumidor |
|-------------|-------------|------------|
| **CTR-001** | `FeatureBuilder` class | Gemini (ONNX, Risk Engine) |
| **CTR-002** | `FEATURE_CONTRACT_V20` frozen dict | Gemini (validation) |
| **CTR-003** | `v20_norm_stats.json` | Gemini (inference) |
| **CTR-004** | `features_snapshot` BD schema | Gemini (drift detection) |
| **CTR-005** | Calculators API | Internal use |
| **CTR-006** | `safe_merge_macro` | Data pipeline |
| **CTR-007** | `SecuritySettings` | All services |
| **CTR-008** | `MLWorkflowTracker` | Training pipeline |
| **CTR-009** | `ModelMetadata` | Inference, Auditing |
| **CTR-010** | `ModelRegistry` (BD) | Gemini, Production |

---

## ✅ CHECKLIST FINAL

### Pre-Desarrollo
- [ ] Leer ARCHITECTURE_CONTRACTS.md completo
- [ ] Verificar dependencias en DAG
- [ ] Configurar pytest + coverage

### Por Tarea (Total: 15 tareas)
| Tarea | Item | Prioridad | Status |
|-------|------|-----------|--------|
| CLAUDE-T1 | P1-13 Feature Contract Core | CRÍTICA | ✅ |
| CLAUDE-T2 | P1-13 Calculators | ALTA | ✅ |
| CLAUDE-T3 | P0-1 norm_stats fix | CRÍTICA | ✅ |
| CLAUDE-T4 | P0-8 features_snapshot migration | CRÍTICA | ✅ |
| CLAUDE-T5 | P0-9 look-ahead bias fix | CRÍTICA | ✅ |
| CLAUDE-T6 | P0-10 ffill limit fix | CRÍTICA | ✅ |
| CLAUDE-T7 | P0-11 merge_asof tolerance | CRÍTICA | ✅ |
| CLAUDE-T8 | P0-4 hardcoded passwords | 🔴 SEGURIDAD | ✅ |
| CLAUDE-T9 | P0-5 MIN_TICK_INTERVAL_MS | CRÍTICA | ✅ |
| CLAUDE-T10 | P1-6 confidence hardcode | ALTA | ✅ |
| CLAUDE-T11 | P1-12 parity tests | ALTA | ✅ |
| **CLAUDE-T12** | **P0-7 ML Workflow Disciplinado** | **🔴 CRÍTICA** | ✅ |
| **CLAUDE-T13** | **P1-2 Model Metadata + bid_ask** | **ALTA** | ✅ |
| **CLAUDE-T14** | **P1-5 Feature Order Validation** | **MEDIA** | ✅ |
| **CLAUDE-T15** | **P1-11 Model Hash Registration BD** | **ALTA** | ✅ |

### Pre-Entrega
- [ ] `pytest tests/ -v` pasa al 100%
- [ ] `pytest --cov=lib/features --cov-report=term-missing` >= 95%
- [ ] `npm test` frontend pasa al 100%
- [ ] No imports de `feature_calculator_factory.py`
- [ ] No passwords hardcodeados (security scan)
- [ ] `config/hyperparameter_decisions.json` existe con audit trail
- [ ] Documentación de contratos completa
- [ ] Notificar a Gemini que CTR-001 a CTR-010 están listos

### Grafo de Dependencias

```
                    ┌─────────────────────────────────────────┐
                    │  P0 CRÍTICOS (ejecutar primero)         │
                    ├─────────────────────────────────────────┤
CLAUDE-T12 (P0-7)  ─┤  ML Workflow (PRIMERO - previene p-hacking)
CLAUDE-T3 (P0-1)   ─┤                                         │
CLAUDE-T8 (P0-4)   ─┤  [independientes - paralelo]            │
CLAUDE-T5 (P0-9)   ─┤                                         │
CLAUDE-T6 (P0-10)  ─┤                                         │
CLAUDE-T7 (P0-11)  ─┤                                         │
CLAUDE-T9 (P0-5)   ─┘                                         │
                    └────────────────┬────────────────────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────────────┐
                    │  P1 CORE (después de P0)                │
                    ├─────────────────────────────────────────┤
CLAUDE-T1 (P1-13) ──┬─→ CLAUDE-T2 (Calculators)               │
                    │                                         │
CLAUDE-T4 (P0-8)  ──┼─→ CLAUDE-T13 (Model Metadata)           │
                    │                                         │
CLAUDE-T1         ──┼─→ CLAUDE-T14 (Feature Order Validation) │
                    │                                         │
CLAUDE-T1         ──┼─→ CLAUDE-T11 (Parity Tests)             │
                    │                                         │
CLAUDE-T13        ──┴─→ CLAUDE-T15 (Model Registry BD)        │
                    └─────────────────────────────────────────┘

CLAUDE-T10 (P1-6) ─→ [independiente - frontend]
```

---

*Documento generado: 2026-01-11*
*Actualizado: 2026-01-11 (v1.2 - agregadas tareas T12-T15, total 15 tareas)*
*Metodología: Spec-Driven + AI-Augmented TDD*
*Coordinación: Contratos CTR-001 a CTR-010 con GEMINI_TASKS.md*


---

## 🎯 TAREA 16: Auto-Registro de Modelos Post-Entrenamiento (P1-18)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T16 |
| **Plan Item** | P1-18 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Dependencias** | CLAUDE-T15 (Model Registry BD) |
| **Produce Contrato** | CTR-011 |
| **Bloquea** | GEMINI-T13 (Dynamic Model API) |

### Objetivo
Implementar auto-registro de modelos en config.models al completar entrenamiento. Cada nuevo modelo entrenado debe aparecer automaticamente disponible para seleccion en el frontend sin intervencion manual.

### Especificacion Funcional

#### Clase Principal: ModelAutoRegister

Ubicacion: src/ml_workflow/model_auto_register.py

Responsabilidades:
1. Registrar modelo automaticamente al completar entrenamiento
2. Calcular y almacenar model_hash SHA256
3. Ejecutar backtest y guardar metricas
4. Insertar en config.models con status=testing

Metodos principales:
- register_model(model_path, version, algorithm, training_config, run_backtest) -> ModelRegistrationResult
- calculate_model_hash(model_path) -> str (SHA256 64 chars)
- run_backtest(model_path, start_date, end_date) -> Dict[str, float]
- _generate_model_id(algorithm, version) -> str (formato: ppo_v21_20260112_143052)

#### Training Callback: ModelRegistrationCallback

Ubicacion: src/ml_workflow/training_callbacks.py

Callback SB3 que registra automaticamente al terminar entrenamiento.
Se integra con model.learn(callback=ModelRegistrationCallback(...))

### SQL Migration

Archivo: database/migrations/006_model_auto_register.sql

- ALTER TABLE config.models ADD model_hash, training_config, registered_at, registered_by
- CREATE INDEX idx_models_hash
- CREATE UNIQUE INDEX idx_models_production (solo 1 production por algoritmo)
- CREATE FUNCTION promote_model_to_production(model_id)

### Tests Requeridos

1. test_register_creates_model_id - Genera model_id unico
2. test_register_calculates_hash - SHA256 correcto
3. test_register_inserts_to_database - Insert en config.models
4. test_register_with_backtest - Ejecuta backtest y guarda metricas
5. test_promote_deprecates_old_production - Promocion depreca anterior
6. test_callback_registers_on_training_end - Callback funciona con SB3

### Criterios de Aceptacion

| # | Criterio | Verificacion |
|---|----------|--------------|
| 1 | register_model() inserta en config.models | Unit test |
| 2 | Hash SHA256 calculado correctamente | Unit test |
| 3 | Backtest ejecutado y metricas guardadas | Integration test |
| 4 | Callback integrado con SB3 training | Integration test |
| 5 | Promocion a production depreca anterior | Unit test |

### Archivos a Crear

| Archivo | Descripcion |
|---------|-------------|
| src/ml_workflow/model_auto_register.py | Clase principal |
| src/ml_workflow/training_callbacks.py | Callback SB3 |
| database/migrations/006_model_auto_register.sql | Migration BD |
| tests/unit/test_model_auto_register.py | Tests unitarios |

---

## ACTUALIZACION CHECKLIST

### Por Tarea (Total: 16 tareas)
| Tarea | Item | Prioridad | Status |
|-------|------|-----------|--------|
| CLAUDE-T16 | P1-18 Auto-Registro Modelos | ALTA | ⬜ |

### Nuevo Contrato Producido
| Contrato ID | Descripcion | Consumidor |
|-------------|-------------|------------|
| CTR-011 | ModelAutoRegister class | Training pipeline, Gemini API |

### Grafo de Dependencias Actualizado



---

*Actualizado: 2026-01-12 (v1.3 - agregada tarea T16 Auto-Registro)*


---

## 🎯 TAREA 16: Auto-Registro de Modelos Post-Entrenamiento (P1-18)

### Metadata
| Campo | Valor |
|-------|-------|
| **ID** | CLAUDE-T16 |
| **Plan Item** | P1-18 |
| **Prioridad** | ALTA |
| **Esfuerzo Estimado** | 5 SP |
| **Dependencias** | CLAUDE-T15 (Model Registry BD) |
| **Produce Contrato** | CTR-011 |
| **Bloquea** | GEMINI-T13 (Dynamic Model API) |

### Objetivo
Implementar auto-registro de modelos en `config.models` al completar entrenamiento. Cada nuevo modelo entrenado debe aparecer automáticamente disponible para selección en el frontend sin intervención manual.

### Especificación Funcional

#### Clase Principal: ModelAutoRegister

**Ubicación:** `src/ml_workflow/model_auto_register.py`

**Responsabilidades:**
1. Registrar modelo automáticamente al completar entrenamiento
2. Calcular y almacenar model_hash SHA256
3. Ejecutar backtest y guardar métricas
4. Insertar en config.models con status='testing'

**Métodos principales:**
- `register_model(model_path, version, algorithm, training_config, run_backtest)` → ModelRegistrationResult
- `calculate_model_hash(model_path)` → str (SHA256 64 chars)
- `run_backtest(model_path, start_date, end_date)` → Dict[str, float]
- `_generate_model_id(algorithm, version)` → str (formato: ppo_v21_20260112_143052)

#### Training Callback: ModelRegistrationCallback

**Ubicación:** `src/ml_workflow/training_callbacks.py`

Callback SB3 que registra automáticamente al terminar entrenamiento.
Se integra con `model.learn(callback=ModelRegistrationCallback(...))`

### SQL Migration

**Archivo:** `database/migrations/006_model_auto_register.sql`

```sql
ALTER TABLE config.models
ADD COLUMN IF NOT EXISTS model_hash VARCHAR(64),
ADD COLUMN IF NOT EXISTS training_config JSONB,
ADD COLUMN IF NOT EXISTS registered_at TIMESTAMPTZ DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS registered_by VARCHAR(100) DEFAULT 'auto_register';

CREATE INDEX IF NOT EXISTS idx_models_hash ON config.models(model_hash);

CREATE UNIQUE INDEX IF NOT EXISTS idx_models_production
ON config.models(algorithm)
WHERE status = 'production';

CREATE OR REPLACE FUNCTION promote_model_to_production(p_model_id VARCHAR)
RETURNS VOID AS $$
BEGIN
    UPDATE config.models
    SET status = 'deprecated', updated_at = NOW()
    WHERE algorithm = (SELECT algorithm FROM config.models WHERE model_id = p_model_id)
    AND status = 'production';

    UPDATE config.models
    SET status = 'production', updated_at = NOW()
    WHERE model_id = p_model_id;
END;
$$ LANGUAGE plpgsql;
```

### Tests Requeridos

1. `test_register_creates_model_id` - Genera model_id único
2. `test_register_calculates_hash` - SHA256 correcto
3. `test_register_inserts_to_database` - Insert en config.models
4. `test_register_with_backtest` - Ejecuta backtest y guarda métricas
5. `test_promote_deprecates_old_production` - Promoción depreca anterior
6. `test_callback_registers_on_training_end` - Callback funciona con SB3

### Criterios de Aceptación

| # | Criterio | Verificación |
|---|----------|--------------|
| 1 | register_model() inserta en config.models | Unit test |
| 2 | Hash SHA256 calculado correctamente | Unit test |
| 3 | Backtest ejecutado y métricas guardadas | Integration test |
| 4 | Callback integrado con SB3 training | Integration test |
| 5 | Promoción a production depreca anterior | Unit test |

### Archivos a Crear

| Archivo | Descripción |
|---------|-------------|
| `src/ml_workflow/model_auto_register.py` | Clase principal |
| `src/ml_workflow/training_callbacks.py` | Callback SB3 |
| `database/migrations/006_model_auto_register.sql` | Migration BD |
| `tests/unit/test_model_auto_register.py` | Tests unitarios |

---

## 📊 ACTUALIZACIÓN RESUMEN DE CONTRATOS

| Contrato ID | Descripción | Consumidor |
|-------------|-------------|------------|
| **CTR-011** | `ModelAutoRegister` class | Training pipeline, Gemini API |

---

## ✅ CHECKLIST ACTUALIZADO

### Por Tarea (Total: 16 tareas)
| Tarea | Item | Prioridad | Status |
|-------|------|-----------|--------|
| CLAUDE-T16 | P1-18 Auto-Registro Modelos | ALTA | ⬜ |

### Grafo de Dependencias Actualizado

```
CLAUDE-T15 (Model Registry BD) ──► CLAUDE-T16 (Auto-Registro)
                                         │
                                         ▼
                                  GEMINI-T13 (Dynamic Model API)
                                         │
                                         ▼
                                  GEMINI-T14 (Replay Integration)
```

---

*Actualizado: 2026-01-12 (v1.3 - agregada tarea T16 Auto-Registro)*
