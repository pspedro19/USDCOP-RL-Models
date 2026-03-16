# Arquitectura de Contratos: Training → Inference → Frontend

**Documento**: ARCHITECTURE_CONTRACTS.md
**Fecha**: 2026-01-11
**Versión**: 1.4 (Sincronizado con IMPLEMENTATION_PLAN.md v3.6)
**Objetivo**: Garantizar coherencia entre entrenamiento, inferencia y visualización dinámica
**Plan Item**: P1-13 en IMPLEMENTATION_PLAN.md (supersede P1-3)
**Audit Refs**: FE-01, PL-05, RD-01, RD-02, RM-01
**Idioma**: Español (código y términos técnicos en inglés)

> **Nota sobre Coherence Score**: Ver sección [Checklist de Coherencia Verificable](#checklist-de-coherencia-verificable) para criterios objetivos.

---

## Estado Actual (Fragmentado)

### Flujo del Boton "Replay"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FLUJO ACTUAL                                       │
└─────────────────────────────────────────────────────────────────────────────┘

[Frontend: Click "Replay"]
         │
         ▼
┌─────────────────────┐
│ fetchTradesWithInference()                                                   │
│ lib/replayApiClient.ts:333                                                   │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ POST /api/replay/load-trades                                                 │
│ app/api/replay/load-trades/route.ts                                          │
│ → Llama a INFERENCE_SERVICE_URL                                              │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Python Inference Service                                                     │
│ http://localhost:8003/v1/backtest                                            │
│ → ObservationBuilder calcula features                                        │
│ → PPO model predice acciones                                                 │
│ → TradeSimulator ejecuta trades                                              │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Response con trades                                                          │
│ → Frontend renderiza en chart                                                │
│ → Equity curve se actualiza                                                  │
│ → Iconos BUY/SELL aparecen                                                   │
└─────────────────────┘
```

### Problema: 3 Implementaciones de Features Separadas

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    ESTADO ACTUAL: 3 IMPLEMENTACIONES                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────┐                                                 │
│  │ TRAINING                │                                                 │
│  │ 01_build_5min_datasets.py                                                 │
│  │ ─────────────────────── │                                                 │
│  │ - RSI: ta.rsi()         │                                                 │
│  │ - ATR: ta.atr()         │                                                 │
│  │ - ADX: ta.adx()         │ ⚠️ IMPLEMENTACION #1                            │
│  │ - Returns: np.log()     │                                                 │
│  │ - Norm: fixed z-score   │                                                 │
│  └─────────────────────────┘                                                 │
│                                                                              │
│  ┌─────────────────────────┐                                                 │
│  │ INFERENCE API           │                                                 │
│  │ observation_builder.py  │                                                 │
│  │ ─────────────────────── │                                                 │
│  │ - RSI: manual impl      │                                                 │
│  │ - ATR: manual impl      │                                                 │
│  │ - ADX: manual impl      │ ⚠️ IMPLEMENTACION #2 (puede diferir!)          │
│  │ - Returns: manual       │                                                 │
│  │ - Norm: from JSON       │                                                 │
│  └─────────────────────────┘                                                 │
│                                                                              │
│  ┌─────────────────────────┐                                                 │
│  │ FACTORY (NO CONECTADA)  │                                                 │
│  │ feature_calculator_     │                                                 │
│  │ factory.py              │                                                 │
│  │ ─────────────────────── │                                                 │
│  │ - IFeatureCalculator    │                                                 │
│  │ - RSICalculator         │ ⚠️ IMPLEMENTACION #3 (no usada!)               │
│  │ - ATRCalculator         │                                                 │
│  │ - etc...                │                                                 │
│  └─────────────────────────┘                                                 │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

❌ RIESGO: Si RSI en training calcula diferente que RSI en inference,
           el modelo recibe inputs DIFERENTES a los que fue entrenado.
           Resultado: Rendimiento degradado o impredecible.
```

---

## Arquitectura Propuesta: Feature Contract Pattern

### Principio: Una Sola Fuente de Verdad

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              ARQUITECTURA PROPUESTA: FEATURE CONTRACT                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                    ┌───────────────────────────────┐                         │
│                    │     FEATURE CONTRACT          │                         │
│                    │     (Single Source of Truth)  │                         │
│                    │                               │                         │
│                    │  lib/features/                │                         │
│                    │  ├── __init__.py              │                         │
│                    │  ├── contract.py    ◄─────────│── Define feature specs  │
│                    │  ├── calculators/             │                         │
│                    │  │   ├── __init__.py          │                         │
│                    │  │   ├── returns.py           │  # log_ret_5m/1h/4h     │
│                    │  │   ├── rsi.py               │  # rsi_9                │
│                    │  │   ├── atr.py               │  # atr_pct              │
│                    │  │   ├── adx.py               │  # adx_14               │
│                    │  │   └── macro.py             │  # dxy_z, vix_z, etc    │
│                    │  └── builder.py     ◄─────────│── Builds observations   │
│                    └───────────────────────────────┘                         │
│                                 │                                            │
│                                 │ IMPORTA                                    │
│           ┌─────────────────────┼─────────────────────┐                      │
│           │                     │                     │                      │
│           ▼                     ▼                     ▼                      │
│  ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐            │
│  │    TRAINING     │   │   INFERENCE     │   │    AIRFLOW      │            │
│  │    Pipeline     │   │      API        │   │      DAGs       │            │
│  │                 │   │                 │   │                 │            │
│  │ from lib.features│   │ from lib.features│   │ from lib.features│            │
│  │ import build_obs│   │ import build_obs│   │ import build_obs│            │
│  └─────────────────┘   └─────────────────┘   └─────────────────┘            │
│                                                                              │
│  ✅ TODOS USAN LA MISMA IMPLEMENTACION                                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementacion del Feature Contract

### 1. Feature Specification (Contrato)

```python
# lib/features/contract.py
"""
Feature Contract - Define exactamente qué features existen y cómo se calculan.
Este archivo es la UNICA fuente de verdad para feature definitions.
"""

from dataclasses import dataclass
from typing import List, Dict, Callable
from enum import Enum

class FeatureType(Enum):
    TECHNICAL = "technical"
    MACRO = "macro"
    STATE = "state"
    DERIVED = "derived"


@dataclass(frozen=True)  # Immutable
class FeatureSpec:
    """Specification for a single feature"""
    name: str
    type: FeatureType
    dependencies: List[str]  # Required input columns
    params: Dict[str, any]   # Calculator parameters
    normalize: bool          # Whether to z-score normalize
    description: str


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE CONTRACT V20 - This defines ALL features for model v20
# ═══════════════════════════════════════════════════════════════════════════

FEATURE_CONTRACT_V20 = {
    "version": "20",
    "observation_dim": 15,
    "features": [
        # Technical features (0-5)
        FeatureSpec(
            name="log_ret_5m",
            type=FeatureType.TECHNICAL,
            dependencies=["close"],
            params={"period": 1},
            normalize=True,
            description="5-minute log return"
        ),
        FeatureSpec(
            name="log_ret_1h",
            type=FeatureType.TECHNICAL,
            dependencies=["close"],
            params={"period": 12},  # 12 * 5min = 1hr
            normalize=True,
            description="1-hour log return"
        ),
        FeatureSpec(
            name="log_ret_4h",
            type=FeatureType.TECHNICAL,
            dependencies=["close"],
            params={"period": 48},  # 48 * 5min = 4hr
            normalize=True,
            description="4-hour log return"
        ),
        FeatureSpec(
            name="rsi_9",
            type=FeatureType.TECHNICAL,
            dependencies=["close"],
            params={"period": 9},
            normalize=True,
            description="RSI with period 9"
        ),
        FeatureSpec(
            name="atr_pct",
            type=FeatureType.TECHNICAL,
            dependencies=["high", "low", "close"],
            params={"period": 10},  # Matches observation_builder.py:120
            normalize=True,
            description="ATR as percentage of close (10-bar ATR)"
        ),
        FeatureSpec(
            name="adx_14",
            type=FeatureType.TECHNICAL,
            dependencies=["high", "low", "close"],
            params={"period": 14},
            normalize=True,
            description="ADX with period 14"
        ),

        # Macro features (6-12)
        FeatureSpec(
            name="dxy_z",
            type=FeatureType.MACRO,
            dependencies=["dxy"],
            params={},
            normalize=True,
            description="DXY z-score"
        ),
        FeatureSpec(
            name="dxy_change_1d",
            type=FeatureType.MACRO,
            dependencies=["dxy"],
            params={"period": 1},
            normalize=True,
            description="DXY daily % change"
        ),
        FeatureSpec(
            name="vix_z",
            type=FeatureType.MACRO,
            dependencies=["vix"],
            params={},
            normalize=True,
            description="VIX z-score"
        ),
        FeatureSpec(
            name="embi_z",
            type=FeatureType.MACRO,
            dependencies=["embi"],
            params={},
            normalize=True,
            description="EMBI spread z-score"
        ),
        FeatureSpec(
            name="brent_change_1d",
            type=FeatureType.MACRO,
            dependencies=["brent"],
            params={"period": 1},
            normalize=True,
            description="Brent daily % change"
        ),
        FeatureSpec(
            name="rate_spread",
            type=FeatureType.MACRO,
            dependencies=["treasury_10y"],  # Note: Uses 10.0 - treasury_10y as proxy
            params={"base_rate": 10.0},     # Assumed Colombia base rate
            normalize=True,
            description="Rate spread proxy (base_rate - US 10Y Treasury)"
        ),
        FeatureSpec(
            name="usdmxn_change_1d",
            type=FeatureType.MACRO,
            dependencies=["usdmxn"],
            params={"period": 1},
            normalize=True,
            description="USDMXN daily % change"
        ),

        # State features (13-14)
        FeatureSpec(
            name="position",
            type=FeatureType.STATE,
            dependencies=[],
            params={},
            normalize=False,  # Already in [-1, 1]
            description="Current position (-1 to 1)"
        ),
        FeatureSpec(
            name="time_normalized",
            type=FeatureType.STATE,
            dependencies=["timestamp"],
            params={},
            normalize=False,  # Already in [0, 1]
            description="Normalized session time"
        ),
    ],

    # Feature order MUST match training
    "feature_order": [
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d",
        "position", "time_normalized"
    ],

    # Normalization stats (loaded from file)
    "norm_stats_file": "config/v20_norm_stats.json",

    # Model file
    "model_file": "models/ppo_v20_production/final_model.zip",
}


def get_contract(version: str = "v20") -> dict:
    """Get feature contract for a model version"""
    contracts = {
        "v20": FEATURE_CONTRACT_V20,
        # "v21": FEATURE_CONTRACT_V21,  # Future versions
    }
    if version not in contracts:
        raise ValueError(f"Unknown model version: {version}")
    return contracts[version]


def validate_contract(contract: dict) -> bool:
    """Validate that a contract is well-formed"""
    assert len(contract["features"]) == contract["observation_dim"]
    assert len(contract["feature_order"]) == contract["observation_dim"]

    for i, spec in enumerate(contract["features"]):
        assert spec.name == contract["feature_order"][i], \
            f"Feature order mismatch at index {i}"

    return True
```

### 2. Feature Builder (Unified Implementation)

```python
# lib/features/builder.py
"""
Unified Feature Builder - Single implementation used by ALL components.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from pathlib import Path
import json

from .contract import get_contract, FeatureSpec, FeatureType
from .calculators import (
    calculate_log_return,
    calculate_rsi,
    calculate_atr,
    calculate_adx,
)


class FeatureBuilder:
    """
    Builds observation vectors according to a feature contract.

    This class is the ONLY place where features are calculated.
    Used by:
    - Training pipeline
    - Inference API
    - Airflow DAGs
    - Backtest/Replay

    Example:
        builder = FeatureBuilder(version="v20")
        obs = builder.build_observation(ohlcv_df, macro_df, position=0, timestamp=now)
    """

    def __init__(self, version: str = "v20", norm_stats_path: Optional[Path] = None):
        self.contract = get_contract(version)
        self.version = version
        self.norm_stats = self._load_norm_stats(norm_stats_path)

    def _load_norm_stats(self, path: Optional[Path] = None) -> Dict:
        """Load normalization statistics"""
        stats_path = path or Path(self.contract["norm_stats_file"])
        if stats_path.exists():
            with open(stats_path) as f:
                return json.load(f)
        raise FileNotFoundError(f"Norm stats not found: {stats_path}")

    def build_observation(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        position: float,
        timestamp: pd.Timestamp,
        bar_idx: int = -1,  # -1 means last bar
    ) -> np.ndarray:
        """
        Build a complete observation vector.

        Args:
            ohlcv: DataFrame with OHLCV data (must have enough history)
            macro: DataFrame with macro indicators
            position: Current position (-1 to 1)
            timestamp: Current bar timestamp
            bar_idx: Index of current bar in ohlcv DataFrame

        Returns:
            numpy array of shape (observation_dim,)
        """
        if bar_idx == -1:
            bar_idx = len(ohlcv) - 1

        features = {}

        # Calculate each feature according to contract
        for spec in self.contract["features"]:
            if spec.type == FeatureType.TECHNICAL:
                value = self._calculate_technical(spec, ohlcv, bar_idx)
            elif spec.type == FeatureType.MACRO:
                value = self._calculate_macro(spec, macro, timestamp)
            elif spec.type == FeatureType.STATE:
                value = self._calculate_state(spec, position, timestamp)
            else:
                raise ValueError(f"Unknown feature type: {spec.type}")

            # Normalize if required
            if spec.normalize and spec.name in self.norm_stats:
                value = self._normalize(value, spec.name)

            features[spec.name] = value

        # Build observation vector in correct order
        obs = np.array([
            features[name] for name in self.contract["feature_order"]
        ], dtype=np.float32)

        # Clip to prevent extreme values
        obs = np.clip(obs, -5.0, 5.0)

        return obs

    def _calculate_technical(
        self,
        spec: FeatureSpec,
        ohlcv: pd.DataFrame,
        bar_idx: int
    ) -> float:
        """Calculate a technical feature"""

        if spec.name.startswith("log_ret"):
            period = spec.params["period"]
            return calculate_log_return(ohlcv["close"], bar_idx, period)

        elif spec.name.startswith("rsi"):
            period = spec.params["period"]
            return calculate_rsi(ohlcv["close"], bar_idx, period)

        elif spec.name == "atr_pct":
            period = spec.params["period"]
            atr = calculate_atr(
                ohlcv["high"], ohlcv["low"], ohlcv["close"],
                bar_idx, period
            )
            return atr / ohlcv["close"].iloc[bar_idx]

        elif spec.name.startswith("adx"):
            period = spec.params["period"]
            return calculate_adx(
                ohlcv["high"], ohlcv["low"], ohlcv["close"],
                bar_idx, period
            )

        raise ValueError(f"Unknown technical feature: {spec.name}")

    def _calculate_macro(
        self,
        spec: FeatureSpec,
        macro: pd.DataFrame,
        timestamp: pd.Timestamp
    ) -> float:
        """Calculate a macro feature"""
        # Find the most recent macro data point (no look-ahead)
        available = macro[macro.index <= timestamp]
        if len(available) == 0:
            return 0.0

        latest = available.iloc[-1]

        if spec.name.endswith("_z"):
            # Z-score of raw value
            col = spec.dependencies[0]
            return float(latest.get(col, 0.0))

        elif spec.name.endswith("_change_1d"):
            # Daily change
            col = spec.dependencies[0]
            if len(available) < 2:
                return 0.0
            prev = available.iloc[-2]
            return (latest[col] - prev[col]) / prev[col] if prev[col] != 0 else 0.0

        elif spec.name == "rate_spread":
            us_rate = latest.get("treasury_10y", 4.0) or 4.0
            base_rate = spec.params.get("base_rate", 10.0)
            return base_rate - us_rate  # Proxy for Colombia-US spread

        raise ValueError(f"Unknown macro feature: {spec.name}")

    def _calculate_state(
        self,
        spec: FeatureSpec,
        position: float,
        timestamp: pd.Timestamp
    ) -> float:
        """Calculate a state feature"""

        if spec.name == "position":
            return float(position)

        elif spec.name == "time_normalized":
            # Normalize time within Colombia trading session
            # SET-FX: 8:00-13:30 Colombia (UTC-5) = 13:00-18:30 UTC
            # Extended hours consideration: 13:00-19:00 UTC (6 hours)
            hour = timestamp.hour + timestamp.minute / 60.0
            SESSION_START = 13.0  # 13:00 UTC = 8:00 Colombia
            SESSION_END = 19.0    # 19:00 UTC = 14:00 Colombia (with buffer)
            SESSION_LENGTH = SESSION_END - SESSION_START

            if SESSION_START <= hour <= SESSION_END:
                return (hour - SESSION_START) / SESSION_LENGTH  # 0 to 1
            return 0.5  # Outside session - neutral value

        raise ValueError(f"Unknown state feature: {spec.name}")

    def _normalize(self, value: float, feature_name: str) -> float:
        """Z-score normalize a value"""
        stats = self.norm_stats.get(feature_name, {"mean": 0, "std": 1})
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 1.0)

        if std == 0:
            return 0.0

        return (value - mean) / std

    def get_feature_names(self) -> list:
        """Get ordered list of feature names"""
        return self.contract["feature_order"].copy()

    def get_observation_dim(self) -> int:
        """Get observation space dimension"""
        return self.contract["observation_dim"]

    def export_feature_snapshot(
        self,
        obs: np.ndarray,
        market_context: Optional[Dict] = None
    ) -> dict:
        """
        Export observation as a traceable snapshot.
        Used for storing in trades_history.features_snapshot.

        Format (unified with IMPLEMENTATION_PLAN.md P1-2):
        {
            "observation": [0.1, 0.2, ...],      # Raw 15-dim vector
            "features": {                         # Named values for audit
                "log_ret_5m": 0.1,
                "rsi_9": 0.5,
                ...
            },
            "market_context": {                   # Optional execution context
                "bid_ask_spread_bps": 15.0,
                "estimated_slippage_bps": 5.0,
                "timestamp_utc": "2025-03-15T14:30:00Z"
            },
            "contract_version": "v20"
        }
        """
        snapshot = {
            "observation": obs.tolist(),
            "features": {
                name: float(obs[i])
                for i, name in enumerate(self.contract["feature_order"])
            },
            "contract_version": f"v{self.contract['version']}"
        }

        if market_context:
            snapshot["market_context"] = market_context

        return snapshot
```

---

## Flujo Completo con Contratos

### Boton Replay → Visualizacion Dinamica

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                 FLUJO COMPLETO CON FEATURE CONTRACT                          │
└──────────────────────────────────────────────────────────────────────────────┘

[Usuario: Click "Replay" con modelo v20, fecha 2025-03-15]
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: useReplay hook                                                     │
│ ─────────────────────────                                                    │
│ 1. Llama fetchTradesWithInference({                                          │
│      from: '2025-03-01',                                                     │
│      to: '2025-03-15',                                                       │
│      modelId: 'ppo_v20'                                                      │
│    })                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ NEXT.JS API: /api/replay/load-trades                                         │
│ ──────────────────────────────────────                                       │
│ 1. Valida parametros                                                         │
│ 2. Llama Python Inference Service                                            │
│    POST http://localhost:8003/v1/backtest                                    │
│    {                                                                         │
│      "start_date": "2025-03-01",                                             │
│      "end_date": "2025-03-15",                                               │
│      "model_id": "ppo_v20"                                                   │
│    }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE SERVICE: BacktestRunner                                            │
│ ─────────────────────────────────────                                        │
│                                                                              │
│ 1. CARGAR CONTRATO:                                                          │
│    contract = get_contract("v20")                                            │
│    builder = FeatureBuilder(version="v20")  ◄── USA FEATURE CONTRACT         │
│                                                                              │
│ 2. CARGAR MODELO:                                                            │
│    model = PPO.load(contract["model_file"])                                  │
│    model_hash = compute_hash(model_path)    ◄── PARA TRAZABILIDAD           │
│                                                                              │
│ 3. ITERAR SOBRE CADA BAR (5 min):                                            │
│    for bar_idx, row in ohlcv.iterrows():                                     │
│        │                                                                     │
│        │  # Construir observacion usando MISMO codigo que training           │
│        ▼                                                                     │
│        obs = builder.build_observation(                                      │
│            ohlcv=ohlcv_df,                                                   │
│            macro=macro_df,                                                   │
│            position=current_position,                                        │
│            timestamp=row['timestamp'],                                       │
│            bar_idx=bar_idx                                                   │
│        )                                                                     │
│        │                                                                     │
│        │  # Modelo predice accion                                            │
│        ▼                                                                     │
│        action, _states = model.predict(obs, deterministic=True)              │
│        action_probs = model.policy.get_distribution(obs).distribution.probs │
│        │                                                                     │
│        │  # Si hay señal de entrada                                          │
│        ▼                                                                     │
│        if should_open_trade(action, action_probs, thresholds):               │
│            trade = create_trade(                                             │
│                entry_price=row['close'],                                     │
│                side='long' if action == 1 else 'short',                      │
│                │                                                             │
│                │  # GUARDAR SNAPSHOT PARA AUDITORIA                          │
│                ▼                                                             │
│                features_snapshot=builder.export_feature_snapshot(obs),       │
│                model_hash=model_hash,                                        │
│                model_version='v20',                                          │
│                entry_confidence=float(action_probs[action]),                 │
│            )                                                                 │
│            trades.append(trade)                                              │
│                                                                              │
│ 4. PERSISTIR EN DB:                                                          │
│    INSERT INTO trades_history (..., features_snapshot, model_hash, ...)      │
│                                                                              │
│ 5. RETORNAR:                                                                 │
│    return {                                                                  │
│        "trades": trades,                                                     │
│        "summary": calculate_metrics(trades),                                 │
│        "source": "generated"                                                 │
│    }                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FRONTEND: Renderizado Dinamico                                               │
│ ──────────────────────────────────                                           │
│                                                                              │
│ 1. trades llegan al hook useReplay                                           │
│                                                                              │
│ 2. requestAnimationFrame loop:                                               │
│    while (currentTime < endTime) {                                           │
│        currentTime += (speed * tickInterval)                                 │
│        │                                                                     │
│        │  # Filtrar trades visibles hasta currentTime                        │
│        ▼                                                                     │
│        visibleTrades = trades.filter(t => t.timestamp <= currentTime)        │
│        │                                                                     │
│        │  # Actualizar chart con iconos BUY/SELL                             │
│        ▼                                                                     │
│        chart.setMarkers(visibleTrades.map(toMarker))                         │
│        │                                                                     │
│        │  # Actualizar equity curve                                          │
│        ▼                                                                     │
│        equitySeries.setData(calculateEquity(visibleTrades))                  │
│        │                                                                     │
│        │  # Actualizar metricas en tiempo real                               │
│        ▼                                                                     │
│        setMetrics({                                                          │
│            totalPnL: sum(visibleTrades.pnl),                                 │
│            winRate: wins / total,                                            │
│            sharpe: calculateSharpe(visibleTrades)                            │
│        })                                                                    │
│    }                                                                         │
│                                                                              │
│ 3. Usuario puede:                                                            │
│    - Pausar/Play                                                             │
│    - Cambiar velocidad (0.5x, 1x, 2x, 4x, 8x, 16x)                           │
│    - Click en trade para ver FEATURE SNAPSHOT                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Entrenamiento de Nuevas Versiones

### Proceso para v21, v22, etc.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│           PROCESO PARA ENTRENAR NUEVA VERSION (ej: v21)                      │
└──────────────────────────────────────────────────────────────────────────────┘

PASO 1: CREAR NUEVO CONTRATO
────────────────────────────
# lib/features/contract.py

FEATURE_CONTRACT_V21 = {
    "version": "21",
    "observation_dim": 17,  # Agregamos 2 features nuevas
    "features": [
        # ... features de v20 ...

        # NUEVAS features para v21
        FeatureSpec(
            name="volume_z",
            type=FeatureType.TECHNICAL,
            dependencies=["volume"],
            params={"period": 20},
            normalize=True,
            description="Volume z-score"
        ),
        FeatureSpec(
            name="spread_ma_ratio",
            type=FeatureType.TECHNICAL,
            dependencies=["high", "low"],
            params={"ma_period": 20},
            normalize=True,
            description="Current spread vs MA spread"
        ),
    ],
    "feature_order": [...],  # Actualizado con nuevas features
    "norm_stats_file": "config/v21_norm_stats.json",
    "model_file": "models/ppo_v21_production/final_model.zip",
}


PASO 2: IMPLEMENTAR CALCULADORES (si son nuevos)
────────────────────────────────────────────────
# lib/features/calculators/volume.py

def calculate_volume_z(volume: pd.Series, bar_idx: int, period: int) -> float:
    """Calculate volume z-score"""
    window = volume.iloc[max(0, bar_idx - period):bar_idx + 1]
    if len(window) < 2:
        return 0.0
    return (window.iloc[-1] - window.mean()) / (window.std() + 1e-8)


PASO 3: GENERAR DATASET DE TRAINING
───────────────────────────────────
# Usa el MISMO FeatureBuilder que inferencia usará

from lib.features.builder import FeatureBuilder

builder = FeatureBuilder(version="v21")

# Construir dataset
observations = []
for bar_idx in range(warmup, len(ohlcv)):
    obs = builder.build_observation(
        ohlcv=ohlcv,
        macro=macro,
        position=0,  # Posicion inicial
        timestamp=ohlcv.index[bar_idx],
        bar_idx=bar_idx
    )
    observations.append(obs)

# Guardar con checksum para reproducibilidad
dataset = pd.DataFrame(observations, columns=builder.get_feature_names())
register_dataset(dataset, "training_v21", "v21", conn)


PASO 4: ENTRENAR MODELO
───────────────────────
# IMPORTANTE: Environment usa MISMO FeatureBuilder

class TradingEnv(gym.Env):
    def __init__(self, version="v21"):
        self.builder = FeatureBuilder(version=version)
        self.observation_space = Box(
            low=-5, high=5,
            shape=(self.builder.get_observation_dim(),)
        )

    def _get_observation(self):
        return self.builder.build_observation(
            ohlcv=self.ohlcv,
            macro=self.macro,
            position=self.position,
            timestamp=self.current_time,
            bar_idx=self.current_idx
        )


PASO 5: EXPORTAR NORM STATS
───────────────────────────
# Despues de entrenar, exportar estadisticas de normalizacion

norm_stats = {}
for col in builder.get_feature_names():
    if col not in ['position', 'time_normalized']:  # State features no se normalizan
        norm_stats[col] = {
            'mean': float(training_data[col].mean()),
            'std': float(training_data[col].std())
        }

with open('config/v21_norm_stats.json', 'w') as f:
    json.dump(norm_stats, f, indent=2)


PASO 6: REGISTRAR MODELO
────────────────────────
# Guardar hash y metadata

register_model(
    model_path=Path("models/ppo_v21_production/final_model.zip"),
    conn=db_conn
)


PASO 7: AGREGAR A MODEL REGISTRY
────────────────────────────────
# lib/model_registry.py

MODEL_REGISTRY = {
    'ppo_v20': {...},
    'ppo_v21': {
        'contract_version': 'v21',
        'path': 'models/ppo_v21_production/final_model.zip',
        'norm_stats': 'config/v21_norm_stats.json',
        'observation_dim': 17,
        'training_end': '2025-06-30',
        'validated': True,
    },
}


PASO 8: FRONTEND PUEDE SELECCIONAR
──────────────────────────────────
// Frontend dropdown

<Select
  value={modelId}
  onChange={setModelId}
  options={[
    { value: 'ppo_v20', label: 'PPO v20 (Production)' },
    { value: 'ppo_v21', label: 'PPO v21 (Beta)' },
  ]}
/>

// Al hacer replay, pasa modelId
fetchTradesWithInference({
  from, to,
  modelId: selectedModelId  // 'ppo_v20' o 'ppo_v21'
})
```

---

## Validacion de Contratos

### Test de Paridad (CI/CD)

```python
# tests/integration/test_feature_contract_parity.py

import pytest
from lib.features.builder import FeatureBuilder
from lib.features.contract import get_contract, validate_contract


def test_contract_v20_is_valid():
    """Contract v20 passes all validations"""
    contract = get_contract("v20")
    assert validate_contract(contract)


def test_feature_builder_produces_correct_dim():
    """FeatureBuilder produces observation of correct dimension"""
    builder = FeatureBuilder(version="v20")
    contract = get_contract("v20")

    # Create dummy data
    ohlcv = create_dummy_ohlcv(100)
    macro = create_dummy_macro(100)

    obs = builder.build_observation(
        ohlcv=ohlcv,
        macro=macro,
        position=0,
        timestamp=ohlcv.index[-1]
    )

    assert obs.shape == (contract["observation_dim"],)


def test_training_inference_parity():
    """Training and inference produce identical features for same input"""

    # Load real historical data
    ohlcv, macro = load_test_data('2025-03-15 14:30:00')

    # Build features via FeatureBuilder (used by both training and inference)
    builder = FeatureBuilder(version="v20")

    # Calculate at specific bar
    obs = builder.build_observation(
        ohlcv=ohlcv,
        macro=macro,
        position=0,
        timestamp=ohlcv.index[-1]
    )

    # Load pre-calculated features from training dataset
    training_features = load_training_features('2025-03-15 14:30:00')

    # They MUST match exactly
    for i, name in enumerate(builder.get_feature_names()):
        assert abs(obs[i] - training_features[name]) < 1e-6, \
            f"Feature {name} mismatch: builder={obs[i]}, training={training_features[name]}"


def test_model_version_loads_correct_contract():
    """Each model version loads its corresponding contract"""

    for model_id in ['ppo_v20', 'ppo_v21']:
        version = model_id.split('_')[1]  # 'v20', 'v21'

        builder = FeatureBuilder(version=version)
        contract = get_contract(version)

        assert builder.get_observation_dim() == contract["observation_dim"]
        assert builder.get_feature_names() == contract["feature_order"]
```

---

## Integracion con Model Registry

### Vinculacion Model → Contract → Trades

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY INTEGRATION                                 │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         model_registry (DB Table)                        ││
│  │  ┌─────────────┬───────────────────┬────────────────┬────────────────┐  ││
│  │  │ model_id    │ model_hash        │ contract_ver   │ norm_stats_hash│  ││
│  │  ├─────────────┼───────────────────┼────────────────┼────────────────┤  ││
│  │  │ ppo_v20     │ sha256:abc123...  │ v20            │ sha256:def456..│  ││
│  │  │ ppo_v21     │ sha256:789xyz...  │ v21            │ sha256:ghi789..│  ││
│  │  └─────────────┴───────────────────┴────────────────┴────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         │ LOOKUP                                                             │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         INFERENCE SERVICE                                ││
│  │                                                                          ││
│  │  1. Load model by model_id                                               ││
│  │  2. Verify model_hash matches registry                                   ││
│  │  3. Load contract for contract_version                                   ││
│  │  4. Load norm_stats, verify hash                                         ││
│  │  5. Create FeatureBuilder with contract                                  ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│         │                                                                    │
│         │ CREATES TRADES                                                     │
│         ▼                                                                    │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         trades_history (DB Table)                        ││
│  │  ┌─────────────┬───────────────────┬────────────────┬────────────────┐  ││
│  │  │ trade_id    │ model_hash        │ model_version  │features_snapshot│  ││
│  │  ├─────────────┼───────────────────┼────────────────┼────────────────┤  ││
│  │  │ 12345       │ sha256:abc123...  │ v20            │ {observation:..}│  ││
│  │  └─────────────┴───────────────────┴────────────────┴────────────────┘  ││
│  │                                                                          ││
│  │  ✅ Full traceability: trade → model → contract → features              ││
│  │                                                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Codigo de Integracion

```python
# lib/model_registry.py
"""
Model Registry - Links models to their contracts and tracks hashes.
Referenced by: IMPLEMENTATION_PLAN.md P1-11, P2-6
"""

from pathlib import Path
import hashlib
from typing import Optional
from .features.contract import get_contract

def compute_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file"""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

class ModelRegistry:
    """
    Registry linking models to their feature contracts.

    Ensures:
    1. Model integrity (hash verification)
    2. Contract version alignment
    3. Norm stats version alignment
    """

    def __init__(self, db_connection):
        self.conn = db_connection

    def register_model(
        self,
        model_id: str,
        model_path: Path,
        contract_version: str,
        norm_stats_path: Path
    ) -> dict:
        """Register a model with its contract and hashes"""

        model_hash = compute_hash(model_path)
        norm_stats_hash = compute_hash(norm_stats_path)
        contract = get_contract(contract_version)

        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO model_registry
            (model_id, model_path, model_hash, contract_version,
             norm_stats_path, norm_stats_hash, observation_dim)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (model_id) DO UPDATE SET
                model_hash = EXCLUDED.model_hash,
                norm_stats_hash = EXCLUDED.norm_stats_hash
        """, (
            model_id, str(model_path), model_hash, contract_version,
            str(norm_stats_path), norm_stats_hash,
            contract["observation_dim"]
        ))
        self.conn.commit()

        return {
            'model_id': model_id,
            'model_hash': model_hash,
            'contract_version': contract_version,
            'norm_stats_hash': norm_stats_hash,
        }

    def verify_and_load(self, model_id: str) -> dict:
        """Verify model integrity and return config for FeatureBuilder"""

        cur = self.conn.cursor()
        cur.execute("""
            SELECT model_path, model_hash, contract_version,
                   norm_stats_path, norm_stats_hash
            FROM model_registry WHERE model_id = %s
        """, (model_id,))

        row = cur.fetchone()
        if not row:
            raise ValueError(f"Model {model_id} not registered")

        model_path, expected_hash, contract_ver, norm_path, norm_hash = row

        # Verify hashes
        actual_model_hash = compute_hash(Path(model_path))
        actual_norm_hash = compute_hash(Path(norm_path))

        if actual_model_hash != expected_hash:
            raise ValueError(f"Model integrity check FAILED for {model_id}")

        if actual_norm_hash != norm_hash:
            raise ValueError(f"Norm stats integrity check FAILED for {model_id}")

        return {
            'model_path': model_path,
            'model_hash': expected_hash,
            'contract_version': contract_ver,
            'norm_stats_path': norm_path,
        }
```

### SQL Table Definition

```sql
-- Aligned with IMPLEMENTATION_PLAN.md P1-11
CREATE TABLE IF NOT EXISTS model_registry (
    id SERIAL PRIMARY KEY,
    model_id VARCHAR(50) UNIQUE NOT NULL,
    model_path TEXT NOT NULL,
    model_hash VARCHAR(64) NOT NULL,
    contract_version VARCHAR(10) NOT NULL,
    norm_stats_path TEXT NOT NULL,
    norm_stats_hash VARCHAR(64) NOT NULL,
    observation_dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    training_dataset_id INTEGER REFERENCES dataset_registry(id),

    -- Validation
    CONSTRAINT valid_contract CHECK (contract_version ~ '^v[0-9]+$')
);

CREATE INDEX idx_model_registry_hash ON model_registry(model_hash);
```

---

## Resumen: Como Funciona el Boton Replay

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          RESUMEN: BOTON REPLAY                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. Usuario selecciona:                                                      │
│     - Modelo: ppo_v20, ppo_v21, etc.                                         │
│     - Rango de fechas                                                        │
│     - Click "Iniciar Replay"                                                 │
│                                                                              │
│  2. Frontend llama:                                                          │
│     POST /api/replay/load-trades                                             │
│     { modelId, startDate, endDate }                                          │
│                                                                              │
│  3. API Route llama:                                                         │
│     Python Inference Service /v1/backtest                                    │
│                                                                              │
│  4. Inference Service:                                                       │
│     a) Carga FEATURE CONTRACT para el modelo                                 │
│     b) Carga MODELO PPO (frozen weights)                                     │
│     c) Para cada bar de 5 min:                                               │
│        - Calcula features usando FeatureBuilder                              │
│        - Modelo predice accion                                               │
│        - Si hay señal → crea trade con features_snapshot                     │
│     d) Persiste trades en DB con trazabilidad completa                       │
│     e) Retorna trades + summary                                              │
│                                                                              │
│  5. Frontend renderiza:                                                      │
│     - Animation loop con requestAnimationFrame                               │
│     - Iconos BUY/SELL aparecen en tiempo de replay                          │
│     - Equity curve se actualiza dinamicamente                                │
│     - Metricas se recalculan en cada tick                                   │
│                                                                              │
│  6. Usuario puede:                                                           │
│     - Pausar/Play                                                            │
│     - Cambiar velocidad                                                      │
│     - Click en trade → ver feature snapshot                                  │
│     - Comparar modelos v20 vs v21                                            │
│                                                                              │
│  GARANTIA: El modelo "ve" exactamente las mismas features                    │
│            que vio durante entrenamiento porque AMBOS usan                   │
│            el MISMO FeatureBuilder con el MISMO contrato.                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencias con IMPLEMENTATION_PLAN.md

### Pre-requisitos (deben completarse antes)

| Item | Descripción | Razón |
|------|-------------|-------|
| **P0-8** | features_snapshot column en BD | FeatureBuilder.export_feature_snapshot() necesita donde guardar |
| **P1-11** | Model hash registration | Para vincular features con modelo específico |
| **P0-1** | v20_norm_stats.json existe | FeatureBuilder carga norm_stats de este archivo |

### Post-requisitos (habilitados después)

| Item | Descripción | Beneficio |
|------|-------------|-----------|
| **P1-12** | Training vs Replay parity test | Ahora trivial: mismo FeatureBuilder |
| **P2-7** | Documentar v19 vs v20 | Contratos documentan diferencias |
| **P1-3** | Unify feature calculation | Este ES el unificador |

---

## Estado de Implementación

**Status**: ✅ Agregado como **P1-13** en IMPLEMENTATION_PLAN.md v3.2

```markdown
### P1-13: Implementar Feature Contract Pattern
**Audit Ref**: FE-01, PL-05, RD-01, RD-02, RM-01
**Issue**: 3 implementaciones separadas de features sin contrato común.
**Documento detallado**: Ver `ARCHITECTURE_CONTRACTS.md`

**Dependencias**:
- P0-8 (features_snapshot column)
- P1-11 (model hash registration)
```

### Checklist de Entregables

- [ ] `lib/features/__init__.py`
- [ ] `lib/features/contract.py` - Feature specifications (SSOT)
- [ ] `lib/features/builder.py` - Unified FeatureBuilder class
- [ ] `lib/features/calculators/__init__.py`
- [ ] `lib/features/calculators/returns.py`
- [ ] `lib/features/calculators/rsi.py`
- [ ] `lib/features/calculators/atr.py`
- [ ] `lib/features/calculators/adx.py`
- [ ] `lib/features/calculators/macro.py`
- [ ] `config/v20_norm_stats.json` (generado de training data)
- [ ] Migrar `01_build_5min_datasets.py` a usar FeatureBuilder
- [ ] Migrar `observation_builder.py` a usar FeatureBuilder
- [ ] Migrar `l5_multi_model_inference.py` a usar FeatureBuilder
- [ ] `tests/integration/test_feature_contract_parity.py`

---

## Auditoría de Coherencia

### Preguntas de Auditoría Resueltas por Feature Contract

| ID | Pregunta | Cómo lo resuelve |
|----|----------|------------------|
| **FE-01** | Features idénticas entre training/inference | Mismo FeatureBuilder |
| **PL-05** | Coherencia training vs replay | Mismo código |
| **RD-01** | Código duplicado pipelines | Elimina duplicación |
| **RD-02** | Features replicadas | Centraliza en lib/features/ |
| **RM-01** | Reproducción de features | Contrato garantiza reproducibilidad |

---

## Integración con Industry Grade (v3.5)

### Referencia a Mejoras de Producción

El Feature Contract Pattern se integra con las mejoras industry-grade documentadas en IMPLEMENTATION_PLAN.md v3.5:

| Priority | Items | Integración con Feature Contract |
|----------|-------|----------------------------------|
| **P1-14** | ONNX Runtime Conversion | FeatureBuilder produce observaciones para ONNX model |
| **P1-15** | Circuit Breakers | Protege pipeline de features |
| **P1-16** | Drift Detection | Monitorea drift en features del contrato |
| **P1-17** | Risk Engine | Usa features para decisiones de riesgo |
| **P2-22** | Feature Store | FeatureBuilder puede alimentar Feature Store |
| **P2-23** | Walk-Forward | FeatureBuilder garantiza no look-ahead |
| **P2-24** | Property Testing | Tests para invariantes del contrato |

### Flujo Extendido con Industry Grade

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    FEATURE CONTRACT + INDUSTRY GRADE                          │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      FEATURE CONTRACT (P1-13)                        │    │
│  │  ┌────────────┐    ┌────────────┐    ┌────────────────────────┐    │    │
│  │  │Contract   │───▶│FeatureBuilder│───▶│ Observation Vector     │    │    │
│  │  │ Spec      │    │ (Unified)  │    │ (15-dim normalized)    │    │    │
│  │  └────────────┘    └────────────┘    └────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      INDUSTRY GRADE LAYER                            │    │
│  │                                                                       │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐     │    │
│  │  │ Drift        │  │ Circuit      │  │ Feature Store          │     │    │
│  │  │ Detection    │  │ Breakers     │  │ (PIT correct)          │     │    │
│  │  │ (P1-16)      │  │ (P1-15)      │  │ (P2-22)                │     │    │
│  │  └──────────────┘  └──────────────┘  └────────────────────────┘     │    │
│  │         │                │                      │                    │    │
│  │         ▼                ▼                      ▼                    │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │               OBSERVABILITY (P2-21)                           │   │    │
│  │  │  feature_drift_score | circuit_breaker_state | inference_lat  │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                               │                                              │
│                               ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      INFERENCE LAYER                                 │    │
│  │  ┌────────────────────┐    ┌────────────────────────────────────┐   │    │
│  │  │ ONNX Runtime       │───▶│ Risk Engine (P1-17)                │   │    │
│  │  │ (P1-14) <10ms      │    │ Position limits, drawdown checks   │   │    │
│  │  └────────────────────┘    └────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Feature Contract + Drift Detection (P1-16)

```python
# lib/features/drift_integration.py
from lib.features.builder import FeatureBuilder
from lib.risk.drift_detection import FeatureDriftDetector

class MonitoredFeatureBuilder:
    """FeatureBuilder with automatic drift monitoring"""

    def __init__(self, version: str, drift_detector: FeatureDriftDetector):
        self.builder = FeatureBuilder(version=version)
        self.drift_detector = drift_detector

    def build_observation_with_monitoring(
        self,
        ohlcv: pd.DataFrame,
        macro: pd.DataFrame,
        position: float,
        timestamp: datetime,
        bar_idx: int
    ) -> tuple[np.ndarray, List[DriftAlert]]:
        """Build observation and check for drift"""

        obs = self.builder.build_observation(
            ohlcv=ohlcv,
            macro=macro,
            position=position,
            timestamp=timestamp,
            bar_idx=bar_idx
        )

        # Check for drift
        feature_dict = self.builder.export_feature_snapshot(
            ohlcv, macro, position, timestamp, bar_idx
        )
        alerts = self.drift_detector.check_point_drift(feature_dict)

        return obs, alerts
```

---

## Checklist de Coherencia Verificable

### Criterios Objetivos

| # | Criterio | Verificación | Estado |
|---|----------|--------------|--------|
| 1 | Versiones cruzadas correctas | ARCH v1.4 ↔ IMPL v3.6 | ✅ |
| 2 | ATR period = 10 en código | `observation_builder.py:120` | ✅ |
| 3 | ADX period = 14 en código | `observation_builder.py:123` | ✅ |
| 4 | observation_dim = 15 | Ambos docs + código | ✅ |
| 5 | feature_order coincide | Array de 15 items | ✅ |
| 6 | norm_stats path correcto | `config/v20_norm_stats.json` | ✅ |
| 7 | Calculators 5 archivos | returns, rsi, atr, adx, macro | ✅ |
| 8 | features_snapshot unificado | Mismo dict structure | ✅ |
| 9 | Dependencias documentadas | P0-8 → P1-11 → P1-13 | ✅ |
| 10 | Industry-grade bidireccional | P1-14 a P2-24 | 🔄 |

**Cálculo**: 9/10 ✅ + 1/10 🔄 pendiente

### Coherence Score: **9/10** ⚠️

> **Nota**: Score ajustado de 10/10 a 9/10 por transparencia.
> Items industry-grade (P1-14 a P1-17) requieren detalle adicional en este documento.

---

## Matriz de Coherencia

| Item | Este Documento | IMPLEMENTATION_PLAN.md | Estado |
|------|----------------|------------------------|--------|
| Plan Item | P1-13 | P1-13 línea 899-1005 | ✅ |
| ATR Period | 10 | 10 (alineado) | ✅ |
| ADX Period | 14 | 14 (alineado) | ✅ |
| observation_dim | 15 | 15 | ✅ |
| feature_order | 15 items | 15 items | ✅ |
| norm_stats path | config/v20_norm_stats.json | config/v20_norm_stats.json | ✅ |
| Calculators | returns, rsi, atr, adx, macro | returns, rsi, atr, adx, macro | ✅ |
| features_snapshot format | Unified dict format | Unified dict format | ✅ |
| Model Registry | Integrado con P1-11 | P1-11, P2-6 | ✅ |
| Trading Hours | 13:00-19:00 UTC (SET-FX) | - | ✅ |
| Dependencies | P0-8, P1-11 | P0-8, P1-11 | ✅ |
| Industry Grade P1 | P1-14 a P1-17 referenciado | P1-14 a P1-17 detallado | ✅ |
| Industry Grade P2 | P2-21 a P2-24 referenciado | P2-21 a P2-24 detallado | ✅ |
| Drift Integration | MonitoredFeatureBuilder | P1-16 implementación | ✅ |
| Config Defaults | Referencia `config/defaults.yaml` | Sección completa | ✅ |

---

## Referencia a Configuración Centralizada

> **Mejora de Auditoría**: Todos los magic numbers deben venir de `config/defaults.yaml`

Los valores hardcoded en el código deben referenciar la configuración centralizada:

```python
# En vez de:
base_slippage = 5.0  # Magic number
confidence = 0.75    # Magic number

# Usar:
from lib.config import DEFAULTS
base_slippage = DEFAULTS['trading']['slippage']['base_bps']
confidence = DEFAULTS['inference']['default_confidence']
```

Ver IMPLEMENTATION_PLAN.md sección "Configuración Centralizada de Defaults" para schema completo.

---

*Documento sincronizado con IMPLEMENTATION_PLAN.md v3.6*
*Última actualización: 2026-01-11*
*Versión: 1.4 (Mejoras de Auditoría Externa)*
*Coherence Score: 9/10 (ver checklist verificable)*
*Idioma: Español (términos técnicos en inglés)*
*Industry Grade: P1-14 a P1-17, P2-21 a P2-24, P3-1 a P3-6*
