# USD/COP RL Trading System

Sistema de trading algoritmico basado en Reinforcement Learning (PPO) para el par USD/COP, con arquitectura production-ready y Feature Contract Pattern.

**Version**: V20 (Enero 2026)
**Estado**: Production-Ready
**Coherence Score**: 100%

---

## Tabla de Contenidos

- [Arquitectura](#arquitectura)
- [Feature Contract Pattern](#feature-contract-pattern)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Modelo V20](#modelo-v20)
- [Contratos Implementados](#contratos-implementados)
- [Pipelines Airflow](#pipelines-airflow)
- [Frontend Dashboard](#frontend-dashboard)
- [API Services](#api-services)
- [Configuracion](#configuracion)
- [Testing](#testing)
- [Seguridad](#seguridad)
- [Quick Start](#quick-start)

---

## Arquitectura

```
                    +-------------------+
                    |   Landing Page    |
                    |   (Next.js 15)    |
                    +--------+----------+
                             |
                    +--------v----------+
                    |    Dashboard      |
                    | Trading Signals   |
                    | Replay Mode       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |                             |
     +--------v--------+          +--------v--------+
     |  Inference API  |          |   Replay API    |
     |   (FastAPI)     |          |   (Next.js)     |
     +--------+--------+          +--------+--------+
              |                             |
              +--------------+--------------+
                             |
                    +--------v----------+
                    |   FeatureBuilder  |  <-- Single Source of Truth
                    |   (CTR-001)       |
                    +--------+----------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v----+  +------v------+  +---v---------+
     | Calculators |  | ONNX Model  |  | Risk Engine |
     | (CTR-005)   |  | (GTR-001)   |  | (GTR-004)   |
     +-------------+  +-------------+  +-------------+
                             |
                    +--------v----------+
                    |    PostgreSQL     |
                    |   TimescaleDB     |
                    +-------------------+
```

---

## Feature Contract Pattern

El sistema usa un **Feature Contract Pattern** que garantiza consistencia entre training e inference.

### Contrato V20 - Frozen Specification

```python
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
    trading_hours_start: str = "13:00"  # UTC
    trading_hours_end: str = "17:55"    # UTC (12:55 Bogota)
    rsi_period: int = 9
    atr_period: int = 10
    adx_period: int = 14
    warmup_bars: int = 14
```

### Invariantes del Sistema

- `observation` siempre tiene `shape=(15,)`
- `observation` nunca contiene NaN o Inf
- Features normalizadas estan en `[-5.0, 5.0]`
- Mismo input produce exactamente mismo output (determinista)

---

## Estructura del Proyecto

```
USDCOP-RL-Models/
|
|-- airflow/
|   |-- dags/
|   |   |-- l0_ohlcv_realtime.py      # Ingesta OHLCV
|   |   |-- l1_feature_refresh.py      # Calculo features (usa NewOHLCVBarSensor)
|   |   |-- l5_multi_model_inference.py # Inference (usa NewFeatureBarSensor)
|   |   |-- sensors/
|   |   |   |-- new_bar_sensor.py      # Event-driven sensors
|   |   |-- utils/
|   |       |-- dag_common.py          # Utilities compartidas
|
|-- config/
|   |-- v20_norm_stats.json            # CTR-003: Estadisticas normalizacion
|   |-- v20_config.yaml                # Configuracion modelo
|   |-- hyperparameter_decisions.json  # P0-7: Tracking ML workflow
|
|-- database/
|   |-- migrations/
|       |-- 001_initial_setup.sql
|       |-- 002_add_foreign_keys.sql
|       |-- 003_add_model_hash_and_constraints.sql  # CTR-004
|       |-- 004_model_metadata.sql
|       |-- 005_model_registry.sql
|       |-- 006_model_auto_register.sql
|
|-- models/
|   |-- ppo_v20_production/            # Modelo V20 en produccion
|       |-- final_model.zip
|       |-- policy.onnx
|
|-- services/
|   |-- inference_api/                 # FastAPI inference service
|   |   |-- config.py                  # Thresholds, paths
|   |   |-- main.py
|   |-- trading_api_multi_model.py     # Multi-model trading
|
|-- src/
|   |-- config/
|   |   |-- __init__.py
|   |   |-- loader.py                  # Load norm_stats
|   |   |-- security.py                # CTR-007: SecuritySettings
|   |
|   |-- core/
|   |   |-- calculators/
|   |       |-- regime.py              # Market regime (P0-9 safe)
|   |
|   |-- data/
|   |   |-- safe_merge.py              # CTR-006: Anti-leakage merge
|   |
|   |-- features/
|   |   |-- __init__.py
|   |   |-- contract.py                # CTR-002: FEATURE_CONTRACT_V20
|   |   |-- builder.py                 # CTR-001: FeatureBuilder
|   |   |-- calculators/               # CTR-005: Individual calculators
|   |       |-- returns.py
|   |       |-- rsi.py
|   |       |-- atr.py
|   |       |-- adx.py
|   |       |-- macro.py
|   |
|   |-- lib/
|       |-- inference/
|       |   |-- onnx_converter.py      # GTR-001: ONNXConverter
|       |
|       |-- risk/
|           |-- circuit_breakers.py    # GTR-002: CircuitBreaker
|           |-- drift_detection.py     # GTR-003: DriftDetector
|           |-- engine.py              # GTR-004: RiskEngine
|
|-- tests/
|   |-- unit/
|   |   |-- test_feature_builder_v20.py
|   |   |-- test_calculators.py
|   |   |-- test_no_lookahead.py
|   |   |-- airflow/
|   |       |-- test_sensors.py
|   |-- integration/
|       |-- test_training_inference_parity.py
|
|-- usdcop-trading-dashboard/          # Frontend Next.js 15
    |-- app/
    |   |-- page.tsx                   # Landing page
    |   |-- dashboard/
    |   |   |-- page.tsx               # Main dashboard
    |   |-- api/
    |       |-- replay/                # Replay API routes
    |       |-- trading/               # Trading API routes
    |-- components/
    |   |-- charts/
    |   |-- trading/
    |-- lib/
        |-- replayApiClient.ts         # API client (P0-6 fixed)
```

---

## Modelo V20

### Especificaciones

| Parametro | Valor |
|-----------|-------|
| **Algoritmo** | PPO (Proximal Policy Optimization) |
| **Framework** | Stable Baselines 3 |
| **Observation Dim** | 15 |
| **Action Space** | Discrete(3): Hold, Long, Short |
| **Red Neuronal** | MLP [256, 256] |
| **Learning Rate** | 3e-4 |
| **N Steps** | 2048 |
| **Batch Size** | 64 |
| **Entropy Coef** | 0.01 |
| **Clip Range** | 0.2 |

### Features (15 dimensiones)

| Index | Feature | Descripcion |
|-------|---------|-------------|
| 0 | log_ret_5m | Log return 5 minutos |
| 1 | log_ret_1h | Log return 1 hora |
| 2 | log_ret_4h | Log return 4 horas |
| 3 | rsi_9 | RSI periodo 9 |
| 4 | atr_pct | ATR como % del precio |
| 5 | adx_14 | ADX periodo 14 |
| 6 | dxy_z | DXY z-score |
| 7 | dxy_change_1d | DXY cambio diario |
| 8 | vix_z | VIX z-score |
| 9 | embi_z | EMBI z-score |
| 10 | brent_change_1d | Brent cambio diario |
| 11 | rate_spread | Spread tasas UST-COL |
| 12 | usdmxn_change_1d | USDMXN cambio diario |
| 13 | position | Posicion actual [-1, 1] |
| 14 | time_normalized | Tiempo normalizado [0, 1] |

### Thresholds de Accion

```python
threshold_long: float = 0.30   # P0-2 verified
threshold_short: float = -0.30
```

---

## Contratos Implementados

### Contratos Claude (CTR)

| ID | Nombre | Ubicacion | Estado |
|----|--------|-----------|--------|
| CTR-001 | FeatureBuilder | `src/features/builder.py` | OK |
| CTR-002 | FEATURE_CONTRACT_V20 | `src/features/contract.py` | OK |
| CTR-003 | v20_norm_stats.json | `config/v20_norm_stats.json` | OK |
| CTR-004 | features_snapshot schema | `database/migrations/003_*.sql` | OK |
| CTR-005 | Calculators API | `src/features/calculators/` | OK |
| CTR-006 | safe_merge_macro | `src/data/safe_merge.py` | OK |
| CTR-007 | SecuritySettings | `src/config/security.py` | OK |

### Contratos Gemini (GTR)

| ID | Nombre | Ubicacion | Estado |
|----|--------|-----------|--------|
| GTR-001 | ONNXConverter | `src/lib/inference/onnx_converter.py` | OK |
| GTR-002 | CircuitBreaker | `src/lib/risk/circuit_breakers.py` | OK |
| GTR-003 | DriftDetector | `src/lib/risk/drift_detection.py` | OK |
| GTR-004 | RiskEngine | `src/lib/risk/engine.py` | OK |

---

## Pipelines Airflow

### Event-Driven Architecture

Los DAGs usan **sensores** en lugar de schedules fijos para evitar drift y overlapping:

```
L0 (fixed schedule) -> inserts OHLCV
       |
       v
L1 (NewOHLCVBarSensor waits) -> calculates features
       |
       v
L5 (NewFeatureBarSensor waits) -> runs inference
```

### DAGs Activos

| DAG | Funcion | Sensor |
|-----|---------|--------|
| `l0_ohlcv_realtime.py` | Ingesta OHLCV cada 5 min | Schedule |
| `l1_feature_refresh.py` | Calcula features | NewOHLCVBarSensor |
| `l5_multi_model_inference.py` | Ejecuta inference | NewFeatureBarSensor |

### Sensores Custom

```python
from sensors.new_bar_sensor import NewOHLCVBarSensor, NewFeatureBarSensor

# Espera nuevos datos OHLCV
sensor_ohlcv = NewOHLCVBarSensor(
    task_id='wait_for_ohlcv',
    max_staleness_minutes=10,
    poke_interval=30,
    timeout=300,
)

# Espera features completas
sensor_features = NewFeatureBarSensor(
    task_id='wait_for_features',
    require_complete=True,
    critical_features=['log_ret_5m', 'rsi_9', 'dxy_z', ...],
)
```

---

## Frontend Dashboard

### Stack Tecnologico

- **Framework**: Next.js 15 (App Router)
- **UI**: Tailwind CSS + shadcn/ui
- **Charts**: Lightweight Charts (TradingView)
- **State**: React hooks + Zustand
- **Testing**: Playwright E2E

### Paginas Principales

| Ruta | Descripcion |
|------|-------------|
| `/` | Landing page |
| `/dashboard` | Dashboard principal con signals |
| `/dashboard?mode=replay` | Modo replay historico |
| `/forecasting` | Predicciones y forecast |

### Replay Mode

Sistema de replay historico para analisis de trades:

```typescript
// Uso del API client
const result = await fetchInferenceTrades(
  { from: '2026-01-01', to: '2026-01-10' },
  { modelId: 'ppo_v20', forceRegenerate: false }
);
```

---

## API Services

### Inference API (FastAPI)

```
POST /api/inference
  Body: { observation: float[15] }
  Response: { action: int, confidence: float, latency_ms: float }

GET /api/models/{modelId}/status
  Response: { state: string, can_trade: bool }

GET /api/models/{modelId}/equity-curve
  Response: { points: EquityPoint[] }
```

### Replay API (Next.js)

```
POST /api/replay/load-trades
  Body: { startDate, endDate, modelId, forceRegenerate }
  Response: { trades: ReplayTrade[], summary: TradeSummary }

GET /api/replay/candles?from=&to=
  Response: { candles: OHLCV[] }
```

---

## Configuracion

### Variables de Entorno Requeridas

```bash
# Database (OBLIGATORIO - nunca hardcodear)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=usdcop_trading
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=<strong-password>  # REQUIRED

# Model
CURRENT_MODEL_ID=ppo_v20

# API
API_SECRET_KEY=<your-secret-key>
```

### Archivo .env.example

```bash
cp .env.example .env
# Editar .env con valores reales
```

---

## Testing

### Ejecutar Tests

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Con coverage
pytest --cov=src --cov-report=html

# Tests de Airflow (requiere Docker)
pytest tests/unit/airflow/ -v
```

### Tests Criticos

| Test | Verifica |
|------|----------|
| `test_feature_builder_v20.py` | Consistencia dimensional |
| `test_no_lookahead.py` | No data leakage |
| `test_training_inference_parity.py` | Paridad training/inference |
| `test_calculators.py` | Calculators individuales |

---

## Seguridad

### Practicas Implementadas (P0-4)

1. **No Hardcoded Passwords**: Usar `SecuritySettings.from_env()`
2. **Validacion de Passwords**: Minimo 8 caracteres, no passwords comunes
3. **URL Masking**: `get_postgres_url_masked()` para logging

```python
from src.config.security import SecuritySettings

# CORRECTO
settings = SecuritySettings.from_env()
url = settings.get_postgres_url()

# INCORRECTO - nunca hacer esto
url = "postgresql://user:password123@host/db"  # NO!
```

### Anti-Data Leakage (P0-10, P0-11)

```python
from src.data.safe_merge import safe_ffill, safe_merge_macro

# Forward-fill con limite (max 144 barras = 12 horas)
df = safe_ffill(df, columns=['dxy', 'vix'], limit=144)

# Merge sin tolerance para evitar data leakage
df = safe_merge_macro(df_ohlcv, df_macro, track_source=True)
```

---

## Quick Start

### 1. Clonar e Instalar

```bash
git clone <repo-url>
cd USDCOP-RL-Models

# Python environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

pip install -r requirements.txt
```

### 2. Configurar Environment

```bash
cp .env.example .env
# Editar .env con credenciales reales
```

### 3. Verificar Instalacion

```bash
python -c "
from src.features.builder import FeatureBuilder
from src.features.contract import FEATURE_CONTRACT_V20
print(f'Contract V20: {FEATURE_CONTRACT_V20.observation_dim} features')
print('Installation OK!')
"
```

### 4. Ejecutar Tests

```bash
pytest tests/unit/ -v
```

### 5. Iniciar Dashboard (Dev)

```bash
cd usdcop-trading-dashboard
npm install
npm run dev
# Abrir http://localhost:3000
```

### 6. Iniciar Airflow (Docker)

```bash
docker-compose up -d
# Abrir http://localhost:8080
```

---

## Documentacion Adicional

- [ARCHITECTURE_CONTRACTS.md](./docs/ARCHITECTURE_CONTRACTS.md) - Contratos de arquitectura
- [IMPLEMENTATION_PLAN.md](./docs/IMPLEMENTATION_PLAN.md) - Plan de implementacion P0-P2
- [CLAUDE_TASKS.md](./docs/CLAUDE_TASKS.md) - Tareas Claude (CTR-001 a CTR-007)
- [GEMINI_TASKS.md](./docs/GEMINI_TASKS.md) - Tareas Gemini (GTR-001 a GTR-004)
- [P2_PENDING_TASKS.md](./docs/P2_PENDING_TASKS.md) - Tareas pendientes P2
- [CLEANUP_RECOMMENDATIONS.md](./docs/CLEANUP_RECOMMENDATIONS.md) - Recomendaciones de limpieza

---

## Licencia

Propiedad de Lean Tech Solutions. Todos los derechos reservados.

---

*Ultima actualizacion: Enero 2026*
