# 🔧 PLAN DE REMEDIACIÓN COMPLETO: 100/100
## USD/COP RL Trading System - De 54.8% a 100%

**Fecha**: 2026-01-17
**Score Actual**: 54.8% (164.5/300) - Riesgo Medio-Alto
**Score Objetivo Semana 1**: 70% (P0 Cerrados)
**Score Objetivo Semana 2**: 85% (P1 Cerrados)
**Score Objetivo Semana 3**: 100% (P2 Cerrados + Polish)
**Duración Total**: 3 semanas (15 días hábiles)

---

## 📊 DIAGNÓSTICO CONSOLIDADO

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SITUACIÓN ACTUAL: 54.8%                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  FORTALEZAS (>80%):                                                        │
│  ══════════════════                                                         │
│  ✅ Feature Calculation (85%) - SSOT bien implementado                     │
│  ✅ Feature Parity (90%) - Training/Inference consistente                  │
│  ✅ Trading Hours (87%) - Calendar robusto                                 │
│  ✅ Risk Management (83%) - Límites bien configurados                      │
│  ✅ Shadow Mode (80%) - ModelRouter funcional                              │
│                                                                             │
│  DEBILIDADES (<60%):                                                       │
│  ══════════════════                                                         │
│  ❌ Training Execution (47%) - No hay SSOT, inline functions               │
│  ❌ Backtest Replay (54%) - 3 engines diferentes                           │
│  ❌ DAG L4 Backtest (52%) - No conectado con frontend                      │
│  ❌ Lineage E2E (53%) - Trazabilidad manual                                │
│  ❌ Feature Snapshots (47%) - No GIN index, queries lentas                 │
│  ❌ Environment Config (55%) - Falta TRADING_ENABLED                       │
│  ❌ Component Connections (55%) - L5 recalcula en vez de leer L1           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚨 GAPS CRÍTICOS (P0) - SEMANA 1

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  #   │ GAP                           │ RIESGO                   │ DÍA     │
├─────────────────────────────────────────────────────────────────────────────┤
│  1   │ No hay TRADING_ENABLED flag   │ Trading sin control      │ 1       │
│  2   │ L5 recalcula features         │ Divergencia vs L1        │ 2       │
│  3   │ 3 BacktestEngines diferentes  │ Resultados inconsistentes│ 3       │
│  4   │ No smoke test antes promote   │ Modelo roto deployado    │ 4       │
│  5   │ No dataset_hash validation    │ Modelo con datos malos   │ 4       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ ROADMAP DE 3 SEMANAS

```
SEMANA 1: CRITICAL FIXES (P0) - Score: 54.8% → 70%
═════════════════════════════════════════════════════════════════════════════
├── Día 1: Trading Flags + Paper Mode + Kill Switch
├── Día 2: L5 → Lee de L1 (eliminar recálculo)
├── Día 3: Unificar BacktestEngine (SSOT)
├── Día 4: Smoke Test + Dataset Validation
├── Día 5: Testing + Integration + Buffer

SEMANA 2: OPERATIONAL IMPROVEMENTS (P1) - Score: 70% → 85%
═════════════════════════════════════════════════════════════════════════════
├── Día 6-7: Training SSOT + Seeds Configurables + Virtual PnL Shadow
├── Día 8: Replay Endpoint + GIN Index + Feature Snapshots
├── Día 9: Lineage API + Slack Alerts + Approval Workflow
├── Día 10: Testing + Buffer

SEMANA 3: QUALITY & POLISH (P2) - Score: 85% → 100%
═════════════════════════════════════════════════════════════════════════════
├── Día 11-12: WebSocket + Feature Flags + Frontend Improvements
├── Día 13: Model Comparison UI + Shadow Comparison View
├── Día 14: Dead Code Cleanup + Market Regime + Overfitting Detection
├── Día 15: Re-Audit + Go-Live Prep + Documentation
```

---

## RESUMEN DE GAPS POR CATEGORÍA

| Categoría | ❌ | ⚠️ | Total | Prioridad |
|-----------|----|----|-------|-----------|
| CF-ENV (Environment) | 5 | 8 | 13 | **P0** |
| INF-L1L5 (L1→L5) | 1 | 6 | 7 | **P0** |
| BT-RP (Backtest Replay) | 8 | 7 | 15 | **P0** |
| PM-PR (Promotion) | 8 | 6 | 14 | **P0** |
| TR-EX (Training) | 7 | 7 | 14 | P1 |
| PM-SM (Shadow Mode) | 3 | 4 | 7 | P1 |
| TZ-LE (Lineage) | 2 | 10 | 12 | P1 |
| TZ-FS (Feature Snapshots) | 5 | 6 | 11 | P1 |
| BT-L4 (DAG L4) | 7 | 10 | 17 | P1 |
| FE-API (Frontend API) | 3 | 7 | 10 | P2 |
| FE-MM (Model Management) | 3 | 2 | 5 | P2 |
| FE-BT (Frontend Backtest) | 1 | 4 | 5 | P2 |
| AR-SS (Architecture) | 2 | 4 | 6 | P2 |
| AR-CN (Connections) | 2 | 5 | 7 | P2 |
| TR-FC (Feature Calc) | 1 | 4 | 5 | P2 |
| INF-RT (Inference RT) | 2 | 5 | 7 | P2 |
| INF-FP (Feature Parity) | 1 | 2 | 3 | P2 |
| CF-TH (Trading Hours) | 0 | 4 | 4 | P2 |
| CF-RM (Risk Mgmt) | 0 | 5 | 5 | P2 |
| PR-VP (Validation) | 1 | 3 | 4 | P2 |
| **TOTAL** | **62** | **99** | **161** | |

---

# ═══════════════════════════════════════════════════════════════════════════
# SEMANA 1: CRITICAL FIXES (P0)
# Score: 54.8% → 70%
# ═══════════════════════════════════════════════════════════════════════════

## DÍA 1: TRADING FLAGS + PAPER MODE + KILL SWITCH

### 🎯 Objetivo
Implementar control total sobre el trading para evitar ejecuciones no deseadas.

### Problema Actual
```python
# ACTUAL: No hay control sobre si el trading está habilitado
# El sistema puede empezar a tradear sin querer
# No hay kill switch para emergencias
```

### Solución 1A: Variables de Entorno

**Archivo**: `.env.example`
```bash
# ══════════════════════════════════════════════════════════════════════════
# TRADING CONTROL FLAGS - CRÍTICO
# ══════════════════════════════════════════════════════════════════════════

# Master switch - debe ser true para ejecutar trades reales
TRADING_ENABLED=false

# Paper trading mode - predice pero no ejecuta
PAPER_TRADING=true

# Shadow mode - modelo challenger corre en paralelo
SHADOW_MODE_ENABLED=true

# Kill switch - para todo inmediatamente
KILL_SWITCH_ACTIVE=false

# Environment identifier
ENVIRONMENT=development

# Feature flags enabled
FEATURE_FLAGS_ENABLED=false

# ══════════════════════════════════════════════════════════════════════════
# PROMOCIÓN FLAGS
# ══════════════════════════════════════════════════════════════════════════

# Requiere smoke test antes de promover
REQUIRE_SMOKE_TEST=true

# Requiere validación de dataset_hash
REQUIRE_DATASET_HASH=true

# Días mínimos en staging antes de production
MIN_STAGING_DAYS=7
```

### Solución 1B: TradingFlags Class (SSOT)

**Archivo**: `src/config/trading_flags.py`
```python
"""
Trading flags configuration - SSOT para control de trading.
"""
import os
from dataclasses import dataclass
from typing import Optional
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingFlags:
    """Flags de control de trading - inmutable."""

    # Master switches
    trading_enabled: bool
    paper_trading: bool
    shadow_mode_enabled: bool
    kill_switch_active: bool

    # Environment
    environment: str
    feature_flags_enabled: bool

    # Promotion flags
    require_smoke_test: bool
    require_dataset_hash: bool
    min_staging_days: int

    @classmethod
    def from_env(cls) -> "TradingFlags":
        """Carga flags desde environment variables."""
        return cls(
            trading_enabled=os.environ.get("TRADING_ENABLED", "false").lower() == "true",
            paper_trading=os.environ.get("PAPER_TRADING", "true").lower() == "true",
            shadow_mode_enabled=os.environ.get("SHADOW_MODE_ENABLED", "true").lower() == "true",
            kill_switch_active=os.environ.get("KILL_SWITCH_ACTIVE", "false").lower() == "true",
            environment=os.environ.get("ENVIRONMENT", "development"),
            feature_flags_enabled=os.environ.get("FEATURE_FLAGS_ENABLED", "false").lower() == "true",
            require_smoke_test=os.environ.get("REQUIRE_SMOKE_TEST", "true").lower() == "true",
            require_dataset_hash=os.environ.get("REQUIRE_DATASET_HASH", "true").lower() == "true",
            min_staging_days=int(os.environ.get("MIN_STAGING_DAYS", "7")),
        )

    def can_execute_trade(self) -> bool:
        """Verifica si se puede ejecutar un trade real."""
        if self.kill_switch_active:
            logger.warning("Kill switch is ACTIVE - blocking trade")
            return False
        if not self.trading_enabled:
            logger.info("Trading is DISABLED")
            return False
        if self.paper_trading:
            logger.info("Paper trading mode - no real execution")
            return False
        if self.environment != "production":
            logger.warning(f"Environment is {self.environment}, not production")
            return False
        return True

    def get_trading_mode(self) -> str:
        """Retorna el modo de trading actual."""
        if self.kill_switch_active:
            return "KILLED"
        if not self.trading_enabled:
            return "DISABLED"
        if self.paper_trading:
            return "PAPER"
        if self.environment != "production":
            return "STAGING"
        return "LIVE"

    def validate_for_production(self) -> tuple[bool, list[str]]:
        """Valida que la configuración es segura para producción."""
        errors = []

        if self.environment != "production":
            errors.append(f"Environment is '{self.environment}', expected 'production'")

        if not self.require_smoke_test:
            errors.append("REQUIRE_SMOKE_TEST should be true in production")

        if not self.require_dataset_hash:
            errors.append("REQUIRE_DATASET_HASH should be true in production")

        if self.min_staging_days < 7:
            errors.append(f"MIN_STAGING_DAYS is {self.min_staging_days}, should be >= 7")

        return len(errors) == 0, errors


@lru_cache()
def get_trading_flags() -> TradingFlags:
    """Singleton de TradingFlags."""
    flags = TradingFlags.from_env()
    logger.info(f"Trading mode: {flags.get_trading_mode()}")
    return flags


def reload_trading_flags() -> TradingFlags:
    """Recarga flags (útil para runtime changes)."""
    get_trading_flags.cache_clear()
    return get_trading_flags()


def activate_kill_switch(reason: str) -> None:
    """Activa el kill switch de emergencia."""
    os.environ["KILL_SWITCH_ACTIVE"] = "true"
    reload_trading_flags()
    logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    # TODO: Send alert to Slack/PagerDuty
```

### Solución 1C: Feature Flags Manager

**Archivo**: `services/shared/feature_flags.py`
```python
"""
Feature flags manager with hot reload support.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional
import json
import os
import logging

logger = logging.getLogger(__name__)


@dataclass
class FeatureFlag:
    """Individual feature flag."""
    name: str
    enabled: bool
    description: str
    created_at: datetime
    updated_at: datetime
    rollout_percentage: int = 100  # 0-100


class FeatureFlags:
    """Feature flags manager with hot reload support."""

    def __init__(self, config_path: Optional[str] = None):
        self._flags: Dict[str, FeatureFlag] = {}
        self._config_path = config_path or os.environ.get(
            "FEATURE_FLAGS_CONFIG",
            "config/feature_flags.json"
        )
        self._last_reload: Optional[datetime] = None
        self._reload_interval_seconds = 60
        self._load_flags()

    def _load_flags(self) -> None:
        """Load flags from config file."""
        try:
            if os.path.exists(self._config_path):
                with open(self._config_path) as f:
                    data = json.load(f)
                    for name, config in data.get("flags", {}).items():
                        self._flags[name] = FeatureFlag(
                            name=name,
                            enabled=config.get("enabled", False),
                            description=config.get("description", ""),
                            created_at=datetime.fromisoformat(config.get("created_at", datetime.now().isoformat())),
                            updated_at=datetime.fromisoformat(config.get("updated_at", datetime.now().isoformat())),
                            rollout_percentage=config.get("rollout_percentage", 100),
                        )
            self._last_reload = datetime.now()
            logger.info(f"Loaded {len(self._flags)} feature flags")
        except Exception as e:
            logger.error(f"Failed to load feature flags: {e}")

    def is_enabled(self, flag_name: str, default: bool = False) -> bool:
        """Check if a feature flag is enabled."""
        self._maybe_reload()
        flag = self._flags.get(flag_name)
        if flag is None:
            return default
        return flag.enabled

    def _maybe_reload(self) -> None:
        """Reload flags if enough time has passed."""
        if self._last_reload is None:
            self._load_flags()
            return

        elapsed = (datetime.now() - self._last_reload).total_seconds()
        if elapsed >= self._reload_interval_seconds:
            self._load_flags()

    def reload(self) -> None:
        """Force reload of flags."""
        self._load_flags()

    def get_all(self) -> Dict[str, bool]:
        """Get all flags status."""
        return {name: flag.enabled for name, flag in self._flags.items()}


# Singleton instance
_feature_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Get singleton FeatureFlags instance."""
    global _feature_flags
    if _feature_flags is None:
        _feature_flags = FeatureFlags()
    return _feature_flags
```

### Solución 1D: Integrar en L5 DAG

**Modificar**: `airflow/dags/l5_multi_model_inference.py`
```python
# Al inicio del DAG
from src.config.trading_flags import get_trading_flags, activate_kill_switch

def execute_inference(**context):
    """Ejecuta inferencia con validación de flags."""
    flags = get_trading_flags()

    # Validar que podemos tradear
    if flags.kill_switch_active:
        logging.warning("Kill switch active - skipping inference")
        return {"status": "KILLED", "action": None, "executed": False}

    if not flags.trading_enabled:
        logging.info("Trading disabled - skipping inference")
        return {"status": "DISABLED", "action": None, "executed": False}

    # Ejecutar inferencia
    try:
        prediction = model.predict(observation)
    except Exception as e:
        logging.error(f"Inference failed: {e}")
        activate_kill_switch(f"Inference error: {e}")
        raise

    # Decidir si ejecutar trade
    if flags.paper_trading:
        logging.info(f"Paper trading mode - would execute: {prediction.action}")
        save_paper_trade(prediction)
        return {"status": "PAPER", "action": prediction.action, "executed": False}

    # Ejecutar trade real
    if flags.can_execute_trade():
        execute_real_trade(prediction)
        return {"status": "LIVE", "action": prediction.action, "executed": True}

    return {"status": flags.get_trading_mode(), "action": prediction.action, "executed": False}
```

### Solución 1E: API Endpoints para Control

**Archivo**: `services/inference_api/routers/config.py`
```python
"""
Configuration endpoints for trading flags.
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from src.config.trading_flags import (
    get_trading_flags,
    reload_trading_flags,
    activate_kill_switch,
    TradingFlags,
)

router = APIRouter(prefix="/config", tags=["config"])


class KillSwitchRequest(BaseModel):
    reason: str
    confirmed: bool = False


class TradingFlagsResponse(BaseModel):
    trading_enabled: bool
    paper_trading: bool
    shadow_mode_enabled: bool
    kill_switch_active: bool
    environment: str
    trading_mode: str
    can_execute: bool


@router.get("/trading-flags", response_model=TradingFlagsResponse)
async def get_flags():
    """Get current trading flags status."""
    flags = get_trading_flags()
    return TradingFlagsResponse(
        trading_enabled=flags.trading_enabled,
        paper_trading=flags.paper_trading,
        shadow_mode_enabled=flags.shadow_mode_enabled,
        kill_switch_active=flags.kill_switch_active,
        environment=flags.environment,
        trading_mode=flags.get_trading_mode(),
        can_execute=flags.can_execute_trade(),
    )


@router.post("/kill-switch")
async def trigger_kill_switch(request: KillSwitchRequest):
    """Activate emergency kill switch."""
    if not request.confirmed:
        raise HTTPException(
            status_code=400,
            detail="Must confirm kill switch activation with confirmed=true"
        )

    activate_kill_switch(request.reason)

    return {
        "status": "activated",
        "reason": request.reason,
        "message": "Kill switch activated - all trading stopped"
    }


@router.post("/reload-flags")
async def reload_flags():
    """Reload trading flags from environment."""
    flags = reload_trading_flags()
    return {
        "status": "reloaded",
        "trading_mode": flags.get_trading_mode()
    }
```

### ✅ Checklist Día 1
```
□ Agregar variables a .env.example
□ Crear src/config/trading_flags.py
□ Crear services/shared/feature_flags.py
□ Crear config/feature_flags.json (template)
□ Integrar TradingFlags en L5 DAG
□ Integrar en InferenceEngine
□ Crear services/inference_api/routers/config.py
□ Agregar router a main.py
□ Test: TRADING_ENABLED=false no ejecuta trades
□ Test: PAPER_TRADING=true no ejecuta trades reales
□ Test: KILL_SWITCH_ACTIVE=true bloquea todo
□ Test: API endpoints funcionan correctamente
```

---

## DÍA 2: L5 LEE DE L1 (NO RECALCULAR)

### 🎯 Objetivo
Eliminar duplicación de cálculo de features entre L1 y L5.

### Problema Actual
```
ACTUAL:
L1 calcula features → guarda en inference_features_5m
L5 TAMBIÉN calcula features (duplicado) → usa para predicción

RIESGO: Si L1 y L5 usan código diferente, features divergen
```

### Solución 2A: FeatureReader Class

**Archivo**: `src/feature_store/readers/feature_reader.py`
```python
"""
Reader para features pre-calculados por L1.
SSOT para lectura de features en inferencia.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import numpy as np
import logging

from src.database import get_db_session
from src.feature_store.core import FEATURE_ORDER

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """Resultado de lectura de features."""
    observation: np.ndarray
    timestamp: datetime
    age_minutes: float
    norm_stats_hash: str
    raw_features: Dict[str, float]
    source: str = "l1_pipeline"


class FeatureReader:
    """Lee features pre-calculados de la DB."""

    def __init__(self):
        self._feature_order = FEATURE_ORDER

    def get_latest_features(
        self,
        symbol: str,
        timestamp: datetime,
        max_age_minutes: int = 10,
    ) -> Optional[FeatureResult]:
        """
        Obtiene las features más recientes para un símbolo.

        Args:
            symbol: Símbolo a buscar (ej: "USDCOP")
            timestamp: Timestamp de referencia
            max_age_minutes: Máxima antigüedad permitida

        Returns:
            FeatureResult si hay features válidas, None si no.
        """
        with get_db_session() as session:
            query = """
                SELECT
                    timestamp,
                    features_normalized,
                    features_raw,
                    norm_stats_hash,
                    created_at
                FROM inference_features_5m
                WHERE symbol = :symbol
                  AND timestamp <= :timestamp
                ORDER BY timestamp DESC
                LIMIT 1
            """

            result = session.execute(query, {
                "symbol": symbol,
                "timestamp": timestamp,
            }).fetchone()

            if result is None:
                logger.warning(f"No features found for {symbol} at {timestamp}")
                return None

            # Verificar edad
            feature_time = result.timestamp
            age = (timestamp - feature_time).total_seconds() / 60

            if age > max_age_minutes:
                logger.warning(
                    f"Features too old: {age:.1f} min > {max_age_minutes} min"
                )
                return None

            # Construir observación en el orden correcto
            features_norm = result.features_normalized

            if not self._validate_feature_order(features_norm):
                logger.error("Feature order validation failed")
                return None

            observation = np.array([
                features_norm[f] for f in self._feature_order
            ], dtype=np.float32)

            return FeatureResult(
                observation=observation,
                timestamp=feature_time,
                age_minutes=age,
                norm_stats_hash=result.norm_stats_hash,
                raw_features=result.features_raw,
            )

    def _validate_feature_order(self, features_dict: Dict) -> bool:
        """Valida que el dict tiene todas las features esperadas."""
        missing = [f for f in self._feature_order if f not in features_dict]
        if missing:
            logger.error(f"Missing features: {missing}")
            return False
        return True

    def get_features_history(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        limit: int = 1000,
    ) -> List[FeatureResult]:
        """Obtiene historial de features para replay."""
        with get_db_session() as session:
            query = """
                SELECT
                    timestamp,
                    features_normalized,
                    features_raw,
                    norm_stats_hash
                FROM inference_features_5m
                WHERE symbol = :symbol
                  AND timestamp BETWEEN :start_time AND :end_time
                ORDER BY timestamp ASC
                LIMIT :limit
            """

            results = session.execute(query, {
                "symbol": symbol,
                "start_time": start_time,
                "end_time": end_time,
                "limit": limit,
            }).fetchall()

            return [
                FeatureResult(
                    observation=np.array([
                        r.features_normalized[f] for f in self._feature_order
                    ], dtype=np.float32),
                    timestamp=r.timestamp,
                    age_minutes=0,
                    norm_stats_hash=r.norm_stats_hash,
                    raw_features=r.features_raw,
                )
                for r in results
            ]
```

### Solución 2B: L1 Features Sensor

**Archivo**: `airflow/dags/sensors/feature_sensor.py`
```python
"""
Sensor que espera a que L1 complete y features estén disponibles.
"""
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from datetime import datetime
import logging

from src.feature_store.readers.feature_reader import FeatureReader

logger = logging.getLogger(__name__)


class L1FeaturesSensor(BaseSensorOperator):
    """
    Sensor que espera a que L1 haya calculado features.

    Verifica:
    1. Features existen para el timestamp esperado
    2. Features no son más viejas que max_age
    3. norm_stats_hash coincide con el esperado (opcional)
    """

    @apply_defaults
    def __init__(
        self,
        symbol: str,
        expected_norm_stats_hash: str = None,
        max_age_minutes: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.symbol = symbol
        self.expected_hash = expected_norm_stats_hash
        self.max_age = max_age_minutes
        self._reader = FeatureReader()

    def poke(self, context) -> bool:
        """Verifica si features están disponibles."""
        execution_time = context['execution_date']

        result = self._reader.get_latest_features(
            symbol=self.symbol,
            timestamp=execution_time,
            max_age_minutes=self.max_age,
        )

        if result is None:
            self.log.info(f"No features available for {self.symbol} at {execution_time}")
            return False

        # Verificar hash si se especificó
        if self.expected_hash and result.norm_stats_hash != self.expected_hash:
            self.log.warning(
                f"norm_stats_hash mismatch! Expected: {self.expected_hash}, "
                f"Got: {result.norm_stats_hash}"
            )
            # Continuar con warning (no bloquear)

        self.log.info(
            f"Features available for {self.symbol}, age: {result.age_minutes:.1f} min"
        )
        return True
```

### Solución 2C: L5 Inference Task (Lee de L1)

**Archivo**: `airflow/dags/tasks/l5_inference_task.py`
```python
"""
L5 Inference Task - Lee features de L1, NO recalcula.
"""
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np
import logging

from src.feature_store.readers.feature_reader import FeatureReader
from src.feature_store.builders.canonical_feature_builder import CanonicalFeatureBuilder
from src.config.trading_flags import get_trading_flags

logger = logging.getLogger(__name__)


class L5InferenceTask:
    """Task de inferencia que LEE features de L1."""

    def __init__(self):
        self._reader = FeatureReader()
        # Builder solo para fallback de emergencia
        self._builder = CanonicalFeatureBuilder.for_inference()
        self._fallback_count = 0
        self._max_fallbacks_before_alert = 5

    def get_features_for_inference(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> Optional[np.ndarray]:
        """
        Obtiene features para inferencia.

        Priority:
        1. Leer de inference_features_5m (calculado por L1)
        2. Fallback: calcular on-the-fly con CanonicalFeatureBuilder
        """
        # Intentar leer de L1
        features = self._reader.get_latest_features(
            symbol=symbol,
            timestamp=timestamp,
            max_age_minutes=10,
        )

        if features is not None:
            logger.info(
                f"Using features from L1 (age: {features.age_minutes:.1f} min, "
                f"hash: {features.norm_stats_hash[:8]}...)"
            )
            self._fallback_count = 0  # Reset counter
            return features.observation

        # Fallback: calcular
        logger.warning("L1 features not available - calculating on-the-fly")
        self._fallback_count += 1

        if self._fallback_count >= self._max_fallbacks_before_alert:
            self._send_l1_health_alert()

        return self._calculate_fallback(symbol, timestamp)

    def _calculate_fallback(
        self,
        symbol: str,
        timestamp: datetime,
    ) -> np.ndarray:
        """Fallback: calcula features si L1 no disponible."""
        ohlcv = self._load_ohlcv(symbol, timestamp)
        macro = self._load_macro(timestamp)

        observation = self._builder.build_observation(
            ohlcv_df=ohlcv,
            macro_df=macro,
            position=self._get_current_position(symbol),
            bar_idx=-1,
        )

        return observation

    def _send_l1_health_alert(self):
        """Envía alerta si L1 falla repetidamente."""
        logger.critical(
            f"L1 features unavailable {self._fallback_count} times consecutively. "
            "Check L1 DAG health!"
        )
        # TODO: Send to Slack/PagerDuty

    def _load_ohlcv(self, symbol: str, timestamp: datetime):
        """Carga datos OHLCV para fallback."""
        # Implementation here
        pass

    def _load_macro(self, timestamp: datetime):
        """Carga datos macro para fallback."""
        # Implementation here
        pass

    def _get_current_position(self, symbol: str) -> int:
        """Obtiene posición actual."""
        # Implementation here
        return 0


def run_l5_inference(**context) -> Dict[str, Any]:
    """Entry point para Airflow task."""
    flags = get_trading_flags()

    if flags.kill_switch_active:
        return {"status": "KILLED", "skipped": True}

    task = L5InferenceTask()
    timestamp = context['execution_date']
    symbol = context.get('params', {}).get('symbol', 'USDCOP')

    features = task.get_features_for_inference(symbol, timestamp)

    if features is None:
        return {"status": "NO_FEATURES", "skipped": True}

    # Continue with inference...
    return {"status": "SUCCESS", "features_shape": features.shape}
```

### ✅ Checklist Día 2
```
□ Crear src/feature_store/readers/feature_reader.py
□ Crear airflow/dags/sensors/feature_sensor.py
□ Crear airflow/dags/tasks/l5_inference_task.py
□ Modificar L5 DAG para usar FeatureReader
□ Agregar L1FeaturesSensor al DAG L5
□ Test: L5 lee features de tabla (no calcula)
□ Test: Fallback funciona si L1 no disponible
□ Test: Sensor espera correctamente
□ Test: Alertas se envían después de N fallbacks
□ Verificar que L1 y L5 usan mismo norm_stats_hash
```

---

## DÍA 3: UNIFICAR BACKTEST ENGINE

### 🎯 Objetivo
Un solo motor de backtest para todo el sistema (SSOT).

### Problema Actual
```
ACTUAL: 3 implementaciones de backtest diferentes
├── services/inference_api/orchestrator/backtest_orchestrator.py
├── src/backtest/factory/backtest_factory.py
└── src/backtest/builder/backtest_builder.py

RIESGO: Resultados diferentes para el mismo input
```

### Solución 3A: Unified Backtest Engine (SSOT)

**Archivo**: `src/backtest/engine/unified_backtest_engine.py`
```python
"""
SSOT: Unified Backtest Engine.

Este es el ÚNICO motor de backtest. Todos los demás deben importar de aquí.
Usado por: DAG L4, API /backtest, CLI, Dashboard
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np
import logging

from src.feature_store.builders.canonical_feature_builder import CanonicalFeatureBuilder
from src.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class BacktestConfig:
    """Configuración de backtest - inmutable."""
    # Período
    start_date: datetime
    end_date: datetime

    # Modelo
    model_uri: str
    norm_stats_path: str

    # Costos (en basis points)
    transaction_cost_bps: float = 75.0
    slippage_bps: float = 15.0

    # Capital
    initial_capital: float = 100_000.0
    position_size: float = 1.0

    # Thresholds para señales
    threshold_long: float = 0.33
    threshold_short: float = -0.33

    # Comportamiento
    allow_short: bool = True
    max_position_hold_bars: Optional[int] = None


@dataclass
class Trade:
    """Representa un trade individual."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: Optional[float] = None
    direction: TradeDirection = TradeDirection.FLAT
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    features_snapshot: Dict[str, float] = field(default_factory=dict)
    model_action: int = 1
    model_confidence: float = 0.0
    bars_held: int = 0


@dataclass
class BacktestMetrics:
    """Métricas de backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    calmar_ratio: float
    total_trades: int
    avg_trade_pnl: float
    avg_bars_held: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "avg_trade_pnl": self.avg_trade_pnl,
            "avg_bars_held": self.avg_bars_held,
        }


@dataclass
class BacktestResult:
    """Resultado completo de backtest."""
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[Trade]
    equity_curve: pd.DataFrame
    daily_returns: pd.Series

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "start_date": self.config.start_date.isoformat(),
                "end_date": self.config.end_date.isoformat(),
                "model_uri": self.config.model_uri,
                "transaction_cost_bps": self.config.transaction_cost_bps,
                "slippage_bps": self.config.slippage_bps,
            },
            "metrics": self.metrics.to_dict(),
            "trade_count": len(self.trades),
            "equity_curve": self.equity_curve.to_dict(orient="records"),
        }


class UnifiedBacktestEngine:
    """
    Motor de backtest unificado - SSOT.

    Garantiza:
    - Mismas features que training/inference (CanonicalFeatureBuilder)
    - Mismos cálculos de métricas
    - Resultados reproducibles
    - No look-ahead bias
    """

    def __init__(self, config: BacktestConfig):
        self._config = config
        self._builder = CanonicalFeatureBuilder.for_backtest(
            norm_stats_path=config.norm_stats_path
        )
        self._model = ModelLoader.load(config.model_uri)
        self._trades: List[Trade] = []
        self._equity: List[float] = [config.initial_capital]
        self._position = TradeDirection.FLAT
        self._current_trade: Optional[Trade] = None
        self._bars_in_position = 0

    def run(self, ohlcv_df: pd.DataFrame, macro_df: pd.DataFrame) -> BacktestResult:
        """
        Ejecuta backtest completo.

        Args:
            ohlcv_df: DataFrame con OHLCV (timestamp, open, high, low, close, volume)
            macro_df: DataFrame con datos macro

        Returns:
            BacktestResult con métricas, trades, equity curve
        """
        # Filtrar por período
        mask = (ohlcv_df['timestamp'] >= self._config.start_date) & \
               (ohlcv_df['timestamp'] <= self._config.end_date)
        ohlcv_df = ohlcv_df[mask].copy().reset_index(drop=True)

        logger.info(f"Running backtest: {len(ohlcv_df)} bars")

        # Iterar barra por barra (sin look-ahead)
        for idx in range(len(ohlcv_df)):
            self._process_bar(ohlcv_df, macro_df, idx)

        # Cerrar posición abierta al final
        if self._position != TradeDirection.FLAT:
            self._close_position(ohlcv_df.iloc[-1])

        # Calcular métricas
        equity_df = self._build_equity_curve(ohlcv_df)
        daily_returns = self._calculate_daily_returns(equity_df)
        metrics = self._calculate_metrics(daily_returns)

        logger.info(f"Backtest complete: {len(self._trades)} trades, Sharpe: {metrics.sharpe_ratio:.2f}")

        return BacktestResult(
            config=self._config,
            metrics=metrics,
            trades=self._trades,
            equity_curve=equity_df,
            daily_returns=daily_returns,
        )

    def _process_bar(
        self,
        ohlcv_df: pd.DataFrame,
        macro_df: pd.DataFrame,
        bar_idx: int,
    ):
        """Procesa una barra individual."""
        current_bar = ohlcv_df.iloc[bar_idx]

        # Incrementar contador de barras en posición
        if self._position != TradeDirection.FLAT:
            self._bars_in_position += 1

            # Check max hold time
            if (self._config.max_position_hold_bars and
                self._bars_in_position >= self._config.max_position_hold_bars):
                self._close_position(current_bar)
                return

        # Obtener datos hasta esta barra (no futuro)
        historical = ohlcv_df.iloc[:bar_idx + 1]

        # Calcular features con datos disponibles
        try:
            observation = self._builder.build_observation(
                ohlcv_df=historical,
                macro_df=macro_df,
                position=self._position.value,
                bar_idx=-1,
            )
        except Exception as e:
            # Skip bar si no hay suficiente historia
            return

        # Obtener predicción
        action, confidence = self._model.predict(observation)

        # Guardar snapshot de features
        features_snapshot = self._builder.get_last_features()

        # Decidir acción basada en thresholds
        signal = self._get_signal(action, confidence)

        # Ejecutar lógica de trading
        self._execute_signal(
            signal=signal,
            bar=current_bar,
            features_snapshot=features_snapshot,
            model_action=action,
            model_confidence=confidence,
        )

    def _get_signal(self, action: int, confidence: float) -> TradeDirection:
        """
        Convierte action del modelo a señal de trading.

        Action mapping (PPO output):
            0 = sell
            1 = hold
            2 = buy
        """
        if action == 2 and confidence >= self._config.threshold_long:
            return TradeDirection.LONG
        elif action == 0 and confidence >= abs(self._config.threshold_short):
            if self._config.allow_short:
                return TradeDirection.SHORT
            else:
                return TradeDirection.FLAT
        else:
            return TradeDirection.FLAT

    def _execute_signal(
        self,
        signal: TradeDirection,
        bar: pd.Series,
        features_snapshot: Dict[str, float],
        model_action: int,
        model_confidence: float,
    ):
        """Ejecuta señal de trading."""
        # Si tenemos posición y señal opuesta → cerrar
        if self._position != TradeDirection.FLAT and signal != self._position:
            self._close_position(bar)

        # Si señal y no posición → abrir
        if signal != TradeDirection.FLAT and self._position == TradeDirection.FLAT:
            self._open_position(
                direction=signal,
                bar=bar,
                features_snapshot=features_snapshot,
                model_action=model_action,
                model_confidence=model_confidence,
            )

    def _open_position(
        self,
        direction: TradeDirection,
        bar: pd.Series,
        features_snapshot: Dict[str, float],
        model_action: int,
        model_confidence: float,
    ):
        """Abre una nueva posición."""
        price = bar['close']

        # Aplicar slippage
        if direction == TradeDirection.LONG:
            price *= (1 + self._config.slippage_bps / 10000)
        else:
            price *= (1 - self._config.slippage_bps / 10000)

        # Aplicar costo de transacción
        cost = self._equity[-1] * self._config.position_size * self._config.transaction_cost_bps / 10000

        self._current_trade = Trade(
            entry_time=bar['timestamp'],
            entry_price=price,
            direction=direction,
            features_snapshot=features_snapshot,
            model_action=model_action,
            model_confidence=model_confidence,
        )
        self._position = direction
        self._bars_in_position = 0
        self._equity[-1] -= cost

    def _close_position(self, bar: pd.Series):
        """Cierra la posición actual."""
        if self._position == TradeDirection.FLAT or self._current_trade is None:
            return

        price = bar['close']

        # Aplicar slippage (inverso a la apertura)
        if self._position == TradeDirection.LONG:
            price *= (1 - self._config.slippage_bps / 10000)
        else:
            price *= (1 + self._config.slippage_bps / 10000)

        # Calcular PnL
        entry_price = self._current_trade.entry_price
        position_value = self._equity[-1] * self._config.position_size

        if self._position == TradeDirection.LONG:
            pnl_pct = (price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - price) / entry_price

        pnl = position_value * pnl_pct

        # Aplicar costo de transacción
        cost = position_value * self._config.transaction_cost_bps / 10000
        pnl -= cost

        # Actualizar trade
        self._current_trade.exit_time = bar['timestamp']
        self._current_trade.exit_price = price
        self._current_trade.pnl = pnl
        self._current_trade.pnl_pct = pnl_pct
        self._current_trade.bars_held = self._bars_in_position

        # Guardar trade
        self._trades.append(self._current_trade)

        # Actualizar equity
        self._equity.append(self._equity[-1] + pnl)

        # Reset
        self._position = TradeDirection.FLAT
        self._current_trade = None
        self._bars_in_position = 0

    def _build_equity_curve(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Construye DataFrame de equity curve."""
        timestamps = ohlcv_df['timestamp'].tolist()

        # Crear equity interpolada
        equity_values = []
        trade_idx = 0
        current_equity = self._config.initial_capital

        for ts in timestamps:
            if trade_idx < len(self._trades):
                trade = self._trades[trade_idx]
                if trade.exit_time and ts >= trade.exit_time:
                    current_equity += trade.pnl
                    trade_idx += 1
            equity_values.append(current_equity)

        return pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity_values,
        })

    def _calculate_daily_returns(self, equity_df: pd.DataFrame) -> pd.Series:
        """Calcula returns diarios."""
        equity_df = equity_df.copy()
        equity_df['date'] = pd.to_datetime(equity_df['timestamp']).dt.date
        daily = equity_df.groupby('date')['equity'].last()
        return daily.pct_change().dropna()

    def _calculate_metrics(self, daily_returns: pd.Series) -> BacktestMetrics:
        """Calcula todas las métricas."""
        if len(self._trades) == 0:
            return BacktestMetrics(
                total_return=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, win_rate=0, profit_factor=0,
                calmar_ratio=0, total_trades=0, avg_trade_pnl=0, avg_bars_held=0
            )

        # Total return
        total_return = (self._equity[-1] - self._config.initial_capital) / self._config.initial_capital

        # Sharpe ratio (anualizado, 252 días)
        if daily_returns.std() > 0:
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0 and downside_returns.std() > 0:
            sortino_ratio = daily_returns.mean() / downside_returns.std() * np.sqrt(252)
        else:
            sortino_ratio = sharpe_ratio

        # Max drawdown
        equity_series = pd.Series(self._equity)
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())

        # Win rate
        winning_trades = [t for t in self._trades if t.pnl and t.pnl > 0]
        win_rate = len(winning_trades) / len(self._trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in self._trades if t.pnl and t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trades if t.pnl and t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0

        # Average trade PnL
        avg_trade_pnl = sum(t.pnl for t in self._trades if t.pnl) / len(self._trades)

        # Average bars held
        avg_bars_held = sum(t.bars_held for t in self._trades) / len(self._trades)

        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            total_trades=len(self._trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_bars_held=avg_bars_held,
        )


# ═══════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION - Punto de entrada único
# ═══════════════════════════════════════════════════════════════════════════

def create_backtest_engine(
    model_uri: str,
    norm_stats_path: str,
    start_date: datetime,
    end_date: datetime,
    **kwargs,
) -> UnifiedBacktestEngine:
    """
    Factory function para crear BacktestEngine.

    USAR SIEMPRE ESTA FUNCIÓN, no instanciar directamente.
    """
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        model_uri=model_uri,
        norm_stats_path=norm_stats_path,
        **kwargs,
    )
    return UnifiedBacktestEngine(config)
```

### ✅ Checklist Día 3
```
□ Crear src/backtest/engine/unified_backtest_engine.py
□ Actualizar DAG L4 para usar UnifiedBacktestEngine
□ Actualizar API /backtest para usar UnifiedBacktestEngine
□ Marcar implementaciones antiguas como deprecated
□ Test: DAG L4 y API producen mismos resultados
□ Test: Métricas coinciden (Sharpe, Sortino, Max DD)
□ Test: No look-ahead bias
□ Test: Features snapshot se guarda correctamente
```

---

## DÍA 4: SMOKE TEST + DATASET VALIDATION

### 🎯 Objetivo
Prevenir que modelos rotos o con datos incorrectos lleguen a producción.

### Problema Actual
```
ACTUAL:
- Se puede promover un modelo sin verificar que funciona
- No se valida que el dataset_hash coincide con el esperado
- Modelos pueden tener norm_stats incorrectas
```

### Solución 4A: Model Smoke Test

**Archivo**: `src/models/validation/smoke_test.py`
```python
"""
Smoke test para validar modelos antes de promoción.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
import time
import logging

from src.models.model_loader import ModelLoader

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Resultado de un test individual."""
    name: str
    passed: bool
    message: str
    value: Optional[Any] = None


@dataclass
class SmokeTestResult:
    """Resultado completo de smoke test."""
    passed: bool
    checks: List[TestResult]
    errors: List[str]
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "message": c.message, "value": c.value}
                for c in self.checks
            ],
            "errors": self.errors,
            "duration_ms": self.duration_ms,
        }


class ModelSmokeTest:
    """
    Smoke test para validar que un modelo es funcional.

    Verifica:
    1. Modelo carga correctamente
    2. Modelo acepta input de 15 features
    3. Modelo produce output válido (0, 1, o 2)
    4. Modelo tiene norm_stats asociadas
    5. norm_stats_hash está logueado en MLflow
    6. Modelo es determinístico
    7. Latencia está dentro de límites
    """

    def __init__(self, model_uri: str, max_latency_ms: float = 100.0):
        self._model_uri = model_uri
        self._max_latency_ms = max_latency_ms
        self._checks: List[TestResult] = []
        self._errors: List[str] = []
        self._model = None

    def run(self) -> SmokeTestResult:
        """Ejecuta todos los checks del smoke test."""
        start_time = time.time()

        self._check_model_loads()
        if self._model is not None:
            self._check_input_shape()
            self._check_output_valid()
            self._check_deterministic()
            self._check_latency()
        self._check_norm_stats_exist()
        self._check_hashes_logged()

        duration_ms = (time.time() - start_time) * 1000
        passed = len(self._errors) == 0

        logger.info(f"Smoke test {'PASSED' if passed else 'FAILED'} in {duration_ms:.1f}ms")

        return SmokeTestResult(
            passed=passed,
            checks=self._checks,
            errors=self._errors,
            duration_ms=duration_ms,
        )

    def _check_model_loads(self):
        """Verifica que el modelo carga correctamente."""
        try:
            self._model = ModelLoader.load(self._model_uri)
            self._checks.append(TestResult(
                name="model_loads",
                passed=True,
                message="Model loaded successfully",
            ))
        except Exception as e:
            self._errors.append(f"Model failed to load: {e}")
            self._checks.append(TestResult(
                name="model_loads",
                passed=False,
                message=str(e),
            ))

    def _check_input_shape(self):
        """Verifica que acepta input de 15 features."""
        try:
            sample_input = np.random.randn(15).astype(np.float32)
            _ = self._model.predict(sample_input)
            self._checks.append(TestResult(
                name="input_shape",
                passed=True,
                message="Model accepts 15-feature input",
            ))
        except Exception as e:
            self._errors.append(f"Model rejects 15-feature input: {e}")
            self._checks.append(TestResult(
                name="input_shape",
                passed=False,
                message=str(e),
            ))

    def _check_output_valid(self):
        """Verifica que output es válido (0, 1, o 2)."""
        try:
            sample_input = np.random.randn(15).astype(np.float32)
            action, confidence = self._model.predict(sample_input)

            if action not in [0, 1, 2]:
                raise ValueError(f"Invalid action: {action}")
            if not 0 <= confidence <= 1:
                raise ValueError(f"Invalid confidence: {confidence}")

            self._checks.append(TestResult(
                name="output_valid",
                passed=True,
                message=f"action={action}, confidence={confidence:.3f}",
                value={"action": action, "confidence": confidence},
            ))
        except Exception as e:
            self._errors.append(f"Model output invalid: {e}")
            self._checks.append(TestResult(
                name="output_valid",
                passed=False,
                message=str(e),
            ))

    def _check_deterministic(self):
        """Verifica que el modelo es determinístico."""
        try:
            np.random.seed(42)
            sample_input = np.random.randn(15).astype(np.float32)

            results = []
            for _ in range(5):
                action, confidence = self._model.predict(sample_input)
                results.append((action, round(confidence, 6)))

            if len(set(results)) == 1:
                self._checks.append(TestResult(
                    name="deterministic",
                    passed=True,
                    message="Model is deterministic",
                ))
            else:
                self._errors.append(f"Model is non-deterministic: {results}")
                self._checks.append(TestResult(
                    name="deterministic",
                    passed=False,
                    message=f"Different results: {results}",
                ))
        except Exception as e:
            self._errors.append(f"Determinism check failed: {e}")

    def _check_latency(self):
        """Verifica que la latencia está dentro de límites."""
        try:
            sample_input = np.random.randn(15).astype(np.float32)

            # Warmup
            for _ in range(10):
                self._model.predict(sample_input)

            # Measure
            start = time.time()
            iterations = 100
            for _ in range(iterations):
                self._model.predict(sample_input)
            avg_ms = (time.time() - start) / iterations * 1000

            passed = avg_ms < self._max_latency_ms
            if not passed:
                self._errors.append(f"Latency too high: {avg_ms:.2f}ms > {self._max_latency_ms}ms")

            self._checks.append(TestResult(
                name="latency",
                passed=passed,
                message=f"{avg_ms:.2f}ms (max: {self._max_latency_ms}ms)",
                value=avg_ms,
            ))
        except Exception as e:
            self._errors.append(f"Latency check failed: {e}")

    def _check_norm_stats_exist(self):
        """Verifica que norm_stats.json existe como artifact."""
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()

            # Parse model URI to get model name and version
            if self._model_uri.startswith("models:/"):
                parts = self._model_uri.replace("models:/", "").split("/")
                model_name = parts[0]
                version_or_alias = parts[1] if len(parts) > 1 else "latest"

                # Get model version
                if version_or_alias.isdigit():
                    mv = client.get_model_version(model_name, version_or_alias)
                else:
                    mv = client.get_model_version_by_alias(model_name, version_or_alias)

                run_id = mv.run_id
            else:
                # Assume it's a run URI
                run_id = self._model_uri.split("/")[-1]

            artifacts = client.list_artifacts(run_id)
            artifact_paths = [a.path for a in artifacts]
            has_norm_stats = any("norm_stats" in p for p in artifact_paths)

            if has_norm_stats:
                self._checks.append(TestResult(
                    name="norm_stats_exist",
                    passed=True,
                    message="norm_stats.json artifact exists",
                ))
            else:
                self._errors.append("norm_stats.json artifact missing")
                self._checks.append(TestResult(
                    name="norm_stats_exist",
                    passed=False,
                    message="norm_stats.json not found in artifacts",
                ))
        except Exception as e:
            self._errors.append(f"Could not verify norm_stats: {e}")
            self._checks.append(TestResult(
                name="norm_stats_exist",
                passed=False,
                message=str(e),
            ))

    def _check_hashes_logged(self):
        """Verifica que los hashes están logueados en MLflow."""
        try:
            import mlflow
            client = mlflow.tracking.MlflowClient()

            # Get run_id (similar logic as above)
            if self._model_uri.startswith("models:/"):
                parts = self._model_uri.replace("models:/", "").split("/")
                model_name = parts[0]
                version_or_alias = parts[1] if len(parts) > 1 else "latest"

                if version_or_alias.isdigit():
                    mv = client.get_model_version(model_name, version_or_alias)
                else:
                    mv = client.get_model_version_by_alias(model_name, version_or_alias)

                run_id = mv.run_id
            else:
                run_id = self._model_uri.split("/")[-1]

            run = client.get_run(run_id)
            params = run.data.params

            has_norm_stats_hash = "norm_stats_hash" in params
            has_dataset_hash = "dataset_hash" in params

            if has_norm_stats_hash and has_dataset_hash:
                self._checks.append(TestResult(
                    name="hashes_logged",
                    passed=True,
                    message=f"norm_stats={params['norm_stats_hash'][:8]}..., dataset={params['dataset_hash'][:8]}...",
                    value={
                        "norm_stats_hash": params["norm_stats_hash"],
                        "dataset_hash": params["dataset_hash"],
                    },
                ))
            else:
                missing = []
                if not has_norm_stats_hash:
                    missing.append("norm_stats_hash")
                if not has_dataset_hash:
                    missing.append("dataset_hash")
                self._errors.append(f"Missing hashes: {missing}")
                self._checks.append(TestResult(
                    name="hashes_logged",
                    passed=False,
                    message=f"Missing: {missing}",
                ))
        except Exception as e:
            self._errors.append(f"Could not verify hashes: {e}")
            self._checks.append(TestResult(
                name="hashes_logged",
                passed=False,
                message=str(e),
            ))


def run_smoke_test(model_uri: str, max_latency_ms: float = 100.0) -> SmokeTestResult:
    """Función de conveniencia para correr smoke test."""
    tester = ModelSmokeTest(model_uri, max_latency_ms)
    return tester.run()
```

### Solución 4B: Promote Model con Validaciones

**Archivo**: `scripts/promote_model.py`
```python
#!/usr/bin/env python
"""
Script para promover modelos con validaciones obligatorias.
"""
import argparse
import sys
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

from src.models.validation.smoke_test import run_smoke_test
from src.config.trading_flags import get_trading_flags


def get_staging_days(client: MlflowClient, model_name: str, version: int) -> int:
    """Calcula días que el modelo ha estado en Staging."""
    mv = client.get_model_version(model_name, str(version))

    # Check version history for when it entered Staging
    # For simplicity, use last_updated_timestamp
    staging_start = mv.last_updated_timestamp / 1000  # Convert ms to s
    now = datetime.now().timestamp()
    days = (now - staging_start) / 86400
    return int(days)


def promote_model(
    model_name: str,
    version: int,
    to_stage: str,
    reason: str,
    skip_smoke_test: bool = False,
    skip_dataset_hash: bool = False,
    skip_staging_time: bool = False,
) -> dict:
    """
    Promueve modelo con validaciones.

    Returns:
        dict con resultado de la promoción
    """
    flags = get_trading_flags()
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{version}"

    print(f"\n{'='*60}")
    print(f"PROMOTING MODEL: {model_name} v{version} → {to_stage}")
    print(f"{'='*60}\n")

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDACIÓN 1: Smoke Test
    # ═══════════════════════════════════════════════════════════════════════
    if flags.require_smoke_test and not skip_smoke_test:
        print("📋 Running smoke test...")
        result = run_smoke_test(model_uri)

        if not result.passed:
            print(f"\n❌ Smoke test FAILED ({result.duration_ms:.1f}ms):")
            for error in result.errors:
                print(f"   • {error}")
            return {
                "success": False,
                "stage": "smoke_test",
                "errors": result.errors,
            }

        print(f"✅ Smoke test PASSED ({len(result.checks)} checks, {result.duration_ms:.1f}ms)")
        for check in result.checks:
            status = "✓" if check.passed else "✗"
            print(f"   {status} {check.name}: {check.message}")
    else:
        print("⚠️  Smoke test SKIPPED")

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDACIÓN 2: Dataset Hash
    # ═══════════════════════════════════════════════════════════════════════
    if flags.require_dataset_hash and not skip_dataset_hash:
        print("\n📋 Validating dataset_hash...")
        mv = client.get_model_version(model_name, str(version))
        run = client.get_run(mv.run_id)

        if "dataset_hash" not in run.data.params:
            print("❌ Model missing dataset_hash - promotion blocked")
            return {
                "success": False,
                "stage": "dataset_hash",
                "errors": ["dataset_hash not logged in MLflow"],
            }

        dataset_hash = run.data.params["dataset_hash"]
        print(f"✅ Dataset hash: {dataset_hash[:16]}...")
    else:
        print("\n⚠️  Dataset hash validation SKIPPED")

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDACIÓN 3: Staging Time (solo para Production)
    # ═══════════════════════════════════════════════════════════════════════
    if to_stage == "Production" and not skip_staging_time:
        print("\n📋 Checking staging time...")
        staging_days = get_staging_days(client, model_name, version)

        if staging_days < flags.min_staging_days:
            print(f"❌ Model has been in Staging for {staging_days} days.")
            print(f"   Minimum required: {flags.min_staging_days} days")
            return {
                "success": False,
                "stage": "staging_time",
                "errors": [f"Only {staging_days} days in staging, need {flags.min_staging_days}"],
            }

        print(f"✅ Staging time: {staging_days} days (min: {flags.min_staging_days})")
    elif to_stage == "Production":
        print("\n⚠️  Staging time check SKIPPED")

    # ═══════════════════════════════════════════════════════════════════════
    # EJECUTAR PROMOCIÓN
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n🚀 Promoting {model_name} v{version} to {to_stage}...")

    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=to_stage,
        archive_existing_versions=(to_stage == "Production"),
    )

    # Log promotion event
    with mlflow.start_run(run_name=f"promotion_{model_name}_v{version}"):
        mlflow.log_params({
            "model_name": model_name,
            "model_version": version,
            "to_stage": to_stage,
            "reason": reason,
            "promoted_by": "promote_model.py",
            "promoted_at": datetime.now().isoformat(),
        })

    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: {model_name} v{version} → {to_stage}")
    print(f"{'='*60}\n")

    return {
        "success": True,
        "model_name": model_name,
        "version": version,
        "stage": to_stage,
        "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Promote ML model with validations")
    parser.add_argument("model_name", help="Name of the model in MLflow")
    parser.add_argument("version", type=int, help="Version number to promote")
    parser.add_argument("stage", choices=["Staging", "Production", "Archived"])
    parser.add_argument("--reason", required=True, help="Reason for promotion")
    parser.add_argument("--skip-smoke-test", action="store_true")
    parser.add_argument("--skip-dataset-hash", action="store_true")
    parser.add_argument("--skip-staging-time", action="store_true")

    args = parser.parse_args()

    result = promote_model(
        model_name=args.model_name,
        version=args.version,
        to_stage=args.stage,
        reason=args.reason,
        skip_smoke_test=args.skip_smoke_test,
        skip_dataset_hash=args.skip_dataset_hash,
        skip_staging_time=args.skip_staging_time,
    )

    if not result["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

### ✅ Checklist Día 4
```
□ Crear src/models/validation/smoke_test.py
□ Crear scripts/promote_model.py
□ Integrar smoke test en CI/CD pipeline
□ Agregar validación de dataset_hash
□ Agregar validación de tiempo en staging
□ Test: promoción sin smoke test falla (si flag activo)
□ Test: promoción sin dataset_hash falla (si flag activo)
□ Test: promoción a Production antes de 7 días falla
□ Test: modelo que no carga falla smoke test
□ Test: modelo no determinístico falla smoke test
```

---

## DÍA 5: TESTING + INTEGRATION + BUFFER

### 🎯 Objetivo
Validar que todos los fixes de la Semana 1 funcionan correctamente juntos.

### Actividades

```
□ Correr suite de tests completa
   pytest tests/ -v --tb=short

□ Tests de integración para nuevos componentes
   pytest tests/integration/test_trading_flags.py
   pytest tests/integration/test_feature_reader.py
   pytest tests/integration/test_unified_backtest.py
   pytest tests/integration/test_smoke_test.py

□ Verificar que L1 → L5 flujo funciona
   - Ejecutar L1 manualmente
   - Verificar que features aparecen en DB
   - Ejecutar L5 y verificar que lee de L1

□ Probar smoke test con modelo real
   python scripts/promote_model.py usdcop_ppo 1 Staging --reason "Test"

□ Documentar cambios en CHANGELOG.md
□ Crear PR con review
□ Merge a main
□ Deploy a staging environment
□ Verificar en staging que todo funciona
```

### Tests Requeridos Semana 1

**Archivo**: `tests/integration/test_week1_integration.py`
```python
"""
Integration tests for Week 1 deliverables.
"""
import pytest
import numpy as np
from datetime import datetime, timedelta

from src.config.trading_flags import TradingFlags, get_trading_flags, reload_trading_flags
from src.feature_store.readers.feature_reader import FeatureReader
from src.backtest.engine.unified_backtest_engine import create_backtest_engine, BacktestConfig
from src.models.validation.smoke_test import run_smoke_test


class TestTradingFlags:
    """Tests for trading flags."""

    def test_flags_from_env(self, monkeypatch):
        """Test loading flags from environment."""
        monkeypatch.setenv("TRADING_ENABLED", "false")
        monkeypatch.setenv("PAPER_TRADING", "true")
        monkeypatch.setenv("KILL_SWITCH_ACTIVE", "false")

        flags = reload_trading_flags()

        assert flags.trading_enabled is False
        assert flags.paper_trading is True
        assert flags.can_execute_trade() is False

    def test_kill_switch_blocks_trading(self, monkeypatch):
        """Test that kill switch blocks all trading."""
        monkeypatch.setenv("TRADING_ENABLED", "true")
        monkeypatch.setenv("PAPER_TRADING", "false")
        monkeypatch.setenv("KILL_SWITCH_ACTIVE", "true")
        monkeypatch.setenv("ENVIRONMENT", "production")

        flags = reload_trading_flags()

        assert flags.can_execute_trade() is False
        assert flags.get_trading_mode() == "KILLED"

    def test_production_validation(self, monkeypatch):
        """Test production validation."""
        monkeypatch.setenv("ENVIRONMENT", "staging")
        monkeypatch.setenv("REQUIRE_SMOKE_TEST", "false")

        flags = reload_trading_flags()
        is_valid, errors = flags.validate_for_production()

        assert is_valid is False
        assert len(errors) >= 2


class TestFeatureReader:
    """Tests for feature reader."""

    @pytest.fixture
    def reader(self):
        return FeatureReader()

    def test_get_latest_features_returns_none_when_empty(self, reader):
        """Test that reader returns None when no features available."""
        result = reader.get_latest_features(
            symbol="NONEXISTENT",
            timestamp=datetime.now(),
            max_age_minutes=10,
        )
        assert result is None

    def test_feature_order_validation(self, reader):
        """Test feature order validation."""
        valid_dict = {f: 0.0 for f in reader._feature_order}
        assert reader._validate_feature_order(valid_dict) is True

        invalid_dict = {"missing_feature": 0.0}
        assert reader._validate_feature_order(invalid_dict) is False


class TestUnifiedBacktestEngine:
    """Tests for unified backtest engine."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data."""
        import pandas as pd
        dates = pd.date_range(start="2025-01-01", periods=100, freq="5min")
        return pd.DataFrame({
            "timestamp": dates,
            "open": np.random.randn(100).cumsum() + 4000,
            "high": np.random.randn(100).cumsum() + 4010,
            "low": np.random.randn(100).cumsum() + 3990,
            "close": np.random.randn(100).cumsum() + 4000,
            "volume": np.random.randint(1000, 10000, 100),
        })

    def test_backtest_config_immutable(self):
        """Test that BacktestConfig is properly configured."""
        config = BacktestConfig(
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 31),
            model_uri="models:/test/1",
            norm_stats_path="/path/to/stats.json",
        )
        assert config.transaction_cost_bps == 75.0
        assert config.slippage_bps == 15.0

    def test_metrics_calculation(self):
        """Test metrics are calculated correctly."""
        # This would require a mock model
        pass


class TestSmokeTest:
    """Tests for smoke test."""

    def test_smoke_test_fails_for_invalid_model(self):
        """Test that smoke test fails for non-existent model."""
        result = run_smoke_test("models:/nonexistent/999")
        assert result.passed is False
        assert len(result.errors) > 0

    def test_smoke_test_reports_duration(self):
        """Test that smoke test reports duration."""
        result = run_smoke_test("models:/nonexistent/999")
        assert result.duration_ms > 0
```

---

## ✅ RESUMEN SEMANA 1

### Entregables Completados

| # | Entregable | Archivo | Impacto |
|---|------------|---------|---------|
| 1 | Trading Flags | `src/config/trading_flags.py` | Control de trading |
| 2 | Feature Flags | `services/shared/feature_flags.py` | Feature toggles |
| 3 | Config API | `services/inference_api/routers/config.py` | API control |
| 4 | Feature Reader | `src/feature_store/readers/feature_reader.py` | L5 lee de L1 |
| 5 | L1 Sensor | `airflow/dags/sensors/feature_sensor.py` | Espera L1 |
| 6 | Unified Backtest | `src/backtest/engine/unified_backtest_engine.py` | SSOT backtest |
| 7 | Smoke Test | `src/models/validation/smoke_test.py` | Validación pre-promoción |
| 8 | Promote Script | `scripts/promote_model.py` | Promoción segura |

### Score Esperado
```
ANTES SEMANA 1:  54.8%
DESPUÉS SEMANA 1: 70%+ ✅

GAPS P0 CERRADOS:
✅ TRADING_ENABLED flag (+ PAPER_TRADING, KILL_SWITCH)
✅ L5 lee de L1 (no recalcula)
✅ Un solo BacktestEngine (SSOT)
✅ Smoke test antes de promote
✅ Dataset hash validation
```

---

# ═══════════════════════════════════════════════════════════════════════════
# SEMANA 2: OPERATIONAL IMPROVEMENTS (P1)
# Score: 70% → 85%
# ═══════════════════════════════════════════════════════════════════════════

## DÍAS 6-7: TRAINING SSOT + VIRTUAL PnL

### Entregables
- Training SSOT script (`src/training/train_ssot.py`)
- Seeds configurables y reproducibilidad
- Virtual PnL para shadow mode

### Código Principal

**Archivo**: `src/training/train_ssot.py`
```python
"""
SSOT for model training - used by both CLI and Airflow DAG L3.
"""
from dataclasses import dataclass
from typing import Optional
import hashlib
import json


@dataclass
class TrainingConfig:
    """Training configuration - SSOT."""
    seed: int = 42
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01


def set_reproducible_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_dataset_hash(path: str) -> str:
    """Compute SHA256 hash of dataset."""
    import hashlib
    hasher = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_model(config: TrainingConfig, dataset_path: str) -> dict:
    """
    Main training function - SSOT.

    Used by:
    - CLI: python -m src.training.train_ssot
    - Airflow: DAG L3 imports this function
    """
    set_reproducible_seeds(config.seed)
    dataset_hash = compute_dataset_hash(dataset_path)

    # Training logic here...

    return {
        "model_path": "...",
        "dataset_hash": dataset_hash,
        "config": config.__dict__,
    }
```

**Archivo**: `src/inference/shadow_pnl.py`
```python
"""
Virtual PnL tracker for shadow model.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np


@dataclass
class VirtualTrade:
    """Virtual trade for shadow tracking."""
    signal: str
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None


@dataclass
class ShadowMetrics:
    """Metrics for shadow model."""
    virtual_pnl: float
    virtual_sharpe: float
    trade_count: int
    win_rate: float
    agreement_rate: float  # % of times shadow agrees with champion


class ShadowPnLTracker:
    """Track virtual PnL for shadow model."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.virtual_position = 0
        self.trades: List[VirtualTrade] = []
        self.champion_signals: List[str] = []
        self.shadow_signals: List[str] = []
        self._entry_price: Optional[float] = None
        self._entry_time: Optional[datetime] = None

    def on_prediction(
        self,
        shadow_signal: str,
        champion_signal: str,
        current_price: float,
        timestamp: datetime,
    ):
        """Process shadow prediction and track virtual trades."""
        self.shadow_signals.append(shadow_signal)
        self.champion_signals.append(champion_signal)

        # Handle position changes
        if shadow_signal == "LONG" and self.virtual_position <= 0:
            if self.virtual_position < 0:
                self._close_position(current_price, timestamp)
            self._open_position("LONG", current_price, timestamp)
            self.virtual_position = 1

        elif shadow_signal == "SHORT" and self.virtual_position >= 0:
            if self.virtual_position > 0:
                self._close_position(current_price, timestamp)
            self._open_position("SHORT", current_price, timestamp)
            self.virtual_position = -1

        elif shadow_signal == "HOLD" and self.virtual_position != 0:
            self._close_position(current_price, timestamp)
            self.virtual_position = 0

    def _open_position(self, signal: str, price: float, timestamp: datetime):
        self._entry_price = price
        self._entry_time = timestamp
        self.trades.append(VirtualTrade(
            signal=signal,
            entry_price=price,
            entry_time=timestamp,
        ))

    def _close_position(self, price: float, timestamp: datetime):
        if self.trades and self.trades[-1].exit_price is None:
            trade = self.trades[-1]
            trade.exit_price = price
            trade.exit_time = timestamp

            if trade.signal == "LONG":
                trade.pnl = (price - trade.entry_price) / trade.entry_price
            else:
                trade.pnl = (trade.entry_price - price) / trade.entry_price

    def get_metrics(self) -> ShadowMetrics:
        """Calculate virtual performance metrics."""
        if not self.trades:
            return ShadowMetrics(
                virtual_pnl=0, virtual_sharpe=0, trade_count=0,
                win_rate=0, agreement_rate=0
            )

        closed_trades = [t for t in self.trades if t.pnl is not None]
        pnls = [t.pnl for t in closed_trades]

        virtual_pnl = sum(pnls) if pnls else 0
        virtual_sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 else 0
        win_rate = len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0

        # Agreement rate
        agreements = sum(1 for s, c in zip(self.shadow_signals, self.champion_signals) if s == c)
        agreement_rate = agreements / len(self.shadow_signals) if self.shadow_signals else 0

        return ShadowMetrics(
            virtual_pnl=virtual_pnl,
            virtual_sharpe=virtual_sharpe,
            trade_count=len(closed_trades),
            win_rate=win_rate,
            agreement_rate=agreement_rate,
        )
```

---

## DÍA 8: REPLAY ENDPOINT + GIN INDEX

### Entregables
- Feature replay endpoint
- GIN index para features_snapshot

**Archivo**: `services/inference_api/routers/replay.py`
```python
"""
Feature replay endpoint for debugging.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime

from src.feature_store.readers.feature_reader import FeatureReader


router = APIRouter(prefix="/replay", tags=["replay"])


class ReplayRequest(BaseModel):
    timestamp: datetime
    symbol: str = "USDCOP"
    model_id: Optional[str] = None


class ReplayResponse(BaseModel):
    timestamp: datetime
    features: dict
    prediction: dict
    signal: str


@router.post("/features", response_model=ReplayResponse)
async def replay_features(request: ReplayRequest):
    """Replay historical features for debugging."""
    reader = FeatureReader()

    result = reader.get_latest_features(
        symbol=request.symbol,
        timestamp=request.timestamp,
        max_age_minutes=60,  # Allow older features for replay
    )

    if result is None:
        raise HTTPException(404, f"No features found for {request.timestamp}")

    # Get prediction
    # ... model loading and prediction logic ...

    return ReplayResponse(
        timestamp=request.timestamp,
        features=result.raw_features,
        prediction={"action": 1, "confidence": 0.5},  # Placeholder
        signal="HOLD",
    )
```

**Archivo**: `database/migrations/020_feature_snapshot_improvements.sql`
```sql
-- Add GIN index for JSONB queries on features_snapshot
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_features_snapshot_gin
ON trades USING GIN (features_snapshot);

-- Add source tracking
ALTER TABLE trades
ADD COLUMN IF NOT EXISTS features_source VARCHAR(50) DEFAULT 'l1_pipeline';

-- Create view for easy feature analysis
CREATE OR REPLACE VIEW trade_features_expanded AS
SELECT
    t.id as trade_id,
    t.timestamp,
    t.signal,
    t.pnl,
    (features_snapshot->>'timestamp')::timestamp as feature_timestamp,
    (features_snapshot->>'source') as feature_source,
    (features_snapshot->'normalized'->>'log_ret_5m')::float as log_ret_5m,
    (features_snapshot->'normalized'->>'log_ret_1h')::float as log_ret_1h,
    (features_snapshot->'normalized'->>'rsi_14')::float as rsi_14,
    (features_snapshot->'normalized'->>'macd_signal')::float as macd_signal,
    (features_snapshot->'normalized'->>'bb_position')::float as bb_position
FROM trades t
WHERE features_snapshot IS NOT NULL;
```

---

## DÍA 9: LINEAGE API + ALERTS

### Entregables
- Lineage API unificado
- Slack alerts para eventos críticos

**Archivo**: `services/inference_api/routers/lineage.py`
```python
"""
Lineage API for complete traceability.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any


router = APIRouter(prefix="/lineage", tags=["lineage"])


class TradeLineage(BaseModel):
    trade_id: int
    timestamp: str
    signal: str
    prediction: Dict[str, Any]
    features: Dict[str, Any]
    model: Dict[str, Any]
    dataset: Dict[str, Any]
    norm_stats_hash: str


@router.get("/trade/{trade_id}", response_model=TradeLineage)
async def get_trade_lineage(trade_id: int):
    """Get complete lineage for a trade."""
    # Implementation here
    pass


@router.get("/model/{model_id}")
async def get_model_lineage(model_id: str):
    """Get complete lineage for a model."""
    pass
```

---

## DÍA 10: TESTING + BUFFER

Testing y estabilización de Semana 2.

---

# ═══════════════════════════════════════════════════════════════════════════
# SEMANA 3: QUALITY & POLISH (P2)
# Score: 85% → 100%
# ═══════════════════════════════════════════════════════════════════════════

## DÍAS 11-12: WEBSOCKET + FRONTEND

- WebSocket para real-time updates
- Feature flags UI
- Frontend improvements

## DÍA 13: MODEL COMPARISON UI

- Model comparison component
- Shadow comparison view

## DÍA 14: CLEANUP + ADVANCED ANALYSIS

- Dead code cleanup
- Market regime detection
- Overfitting detection

## DÍA 15: RE-AUDIT + GO-LIVE

- Full re-audit
- Documentation
- Go-live preparation

---

# ═══════════════════════════════════════════════════════════════════════════
# ARCHIVOS NUEVOS REQUERIDOS
# ═══════════════════════════════════════════════════════════════════════════

```
SEMANA 1 (P0):
├── src/config/trading_flags.py
├── services/shared/feature_flags.py
├── config/feature_flags.json
├── services/inference_api/routers/config.py
├── src/feature_store/readers/feature_reader.py
├── airflow/dags/sensors/feature_sensor.py
├── airflow/dags/tasks/l5_inference_task.py
├── src/backtest/engine/unified_backtest_engine.py
├── src/models/validation/smoke_test.py
└── scripts/promote_model.py

SEMANA 2 (P1):
├── src/training/train_ssot.py
├── src/training/reproducibility.py
├── src/inference/shadow_pnl.py
├── services/inference_api/routers/replay.py
├── services/inference_api/routers/lineage.py
├── database/migrations/020_feature_snapshot_improvements.sql
└── services/mlops/alerts.py

SEMANA 3 (P2):
├── services/inference_api/websocket.py
├── src/backtest/overfitting.py
├── src/backtest/regime.py
├── usdcop-trading-dashboard/components/models/ModelComparison.tsx
├── usdcop-trading-dashboard/components/shadow/ShadowComparison.tsx
└── scripts/cleanup_dead_code.py
```

---

# ═══════════════════════════════════════════════════════════════════════════
# MÉTRICAS DE ÉXITO
# ═══════════════════════════════════════════════════════════════════════════

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OBJETIVOS POR SEMANA                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SEMANA    SCORE     P0 GAPS    P1 GAPS    P2 GAPS    ESTADO               │
│  ═══════════════════════════════════════════════════════════════════════   │
│  Inicio    54.8%     5          7          10         ⚠️ Riesgo Medio-Alto │
│  Sem 1     70%+      0          7          10         🟡 Mejorando         │
│  Sem 2     85%+      0          0          10         🟢 Riesgo Bajo       │
│  Sem 3     100%      0          0          0          ✅ Production Ready  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# ═══════════════════════════════════════════════════════════════════════════
# VALIDACIÓN FINAL
# ═══════════════════════════════════════════════════════════════════════════

```bash
# Día 15: Re-ejecutar auditoría completa
python scripts/run_audit.py --full --questions=300

# Expected output:
# ✅ TR-EX: 20/20 (100%)
# ✅ TR-FC: 20/20 (100%)
# ✅ INF-RT: 25/25 (100%)
# ✅ INF-FP: 20/20 (100%)
# ✅ INF-L1L5: 15/15 (100%)
# ✅ BT-RP: 25/25 (100%)
# ✅ BT-L4: 25/25 (100%)
# ✅ PM-PR: 25/25 (100%)
# ✅ PM-SM: 25/25 (100%)
# ✅ FE-API: 20/20 (100%)
# ✅ FE-MM: 15/15 (100%)
# ✅ FE-BT: 15/15 (100%)
# ✅ CF-ENV: 20/20 (100%)
# ✅ CF-TH: 15/15 (100%)
# ✅ CF-RM: 15/15 (100%)
# ✅ TZ-LE: 15/15 (100%)
# ✅ TZ-FS: 15/15 (100%)
# ✅ AR-SS: 10/10 (100%)
# ✅ AR-CN: 10/10 (100%)
# ✅ PR-VP: 10/10 (100%)
#
# TOTAL: 300/300 = 100% ✅
```

---

**Documento creado**: 2026-01-17
**Última actualización**: 2026-01-17
**Autor**: Trading Operations Team
**Próxima revisión**: Post-implementación Semana 1

---

*Este plan asegura cobertura completa de las 300 preguntas de auditoría con implementaciones detalladas y priorizadas.*
