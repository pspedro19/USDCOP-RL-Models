# Plan de Remediación Completo
## USDCOP Trading System - Post-Auditoría
**Fecha:** 2026-01-17
**Objetivo:** Corregir todos los issues críticos identificados en la auditoría de código

---

## Resumen Ejecutivo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PLAN DE REMEDIACIÓN COMPLETO                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  Total Issues:        14                                                    │
│  P0 (Críticos):        3  → Resolver HOY                                   │
│  P1 (Altos):           4  → Resolver esta semana                           │
│  P2 (Medios):          4  → Resolver este mes                              │
│  P3 (Bajos):           3  → Resolver cuando sea posible                    │
│                                                                             │
│  Score Actual:        72/100                                                │
│  Score Esperado:      94/100 (después de remediation)                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## FASE 1: P0 - CRÍTICOS (Resolver HOY)

### P0-1: L1 DAG Feature Parity
**Archivo:** `airflow/dags/l1_feature_refresh.py`
**Impacto:** Training/Inference drift - El modelo recibe features diferentes

#### Problema Actual
```python
# L1 DAG ACTUAL (INCORRECTO) - líneas 102-138
def calc_rsi(series: pd.Series, period: int = 9) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()  # SMA
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()  # SMA
    ...

def calc_atr(high, low, close, period=10):
    ...
    atr = tr.rolling(window=period).mean()  # SMA

def calc_adx(high, low, close, period=14):
    ...
    # También usa SMA
```

#### Solución
```python
# NUEVO: Importar calculadores SSOT
from src.feature_store.core import (
    RSICalculator,
    ATRPercentCalculator,
    ADXCalculator,
    LogReturnCalculator,
    CalculatorRegistry,
    FEATURE_CONTRACT
)

# ELIMINAR: Las funciones calc_rsi(), calc_atr(), calc_adx() locales

# REEMPLAZAR con:
def calculate_technical_features(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """Calcular features técnicos usando SSOT calculadores con Wilder's EMA."""
    registry = CalculatorRegistry.instance()

    result = pd.DataFrame(index=ohlcv_df.index)

    # Usar calculadores del registry (ya usan Wilder's EMA)
    for feature_name in ['log_ret_5m', 'log_ret_1h', 'log_ret_4h',
                         'rsi_9', 'atr_pct', 'adx_14']:
        calc = registry.get(feature_name)
        if calc:
            result[feature_name] = calc.calculate_batch(ohlcv_df)

    return result
```

#### Validación
```bash
# Después del fix, ejecutar:
pytest tests/regression/test_feature_builder_parity.py -v
pytest tests/unit/test_builder_delegation.py -v
```

---

### P0-2: Prometheus Config
**Archivo:** `docker/prometheus/prometheus.yml`
**Impacto:** Alertas NO funcionan - 40+ reglas definidas pero no cargadas

#### Problema Actual
```yaml
rule_files: []          # VACÍO
alertmanagers:
  static_configs:
    - targets: []       # VACÍO
```

#### Solución
```yaml
# Agregar al inicio del archivo:
rule_files:
  - /etc/prometheus/rules/*.yml

# Modificar alerting section:
alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
      scheme: http
      timeout: 10s

# Agregar scrape job para inference-api (faltante):
scrape_configs:
  # ... existing jobs ...

  - job_name: 'inference-api'
    static_configs:
      - targets: ['inference-api:8000']
    metrics_path: /metrics
    scrape_interval: 15s
```

#### También actualizar docker-compose.yml
```yaml
prometheus:
  volumes:
    - ./config/prometheus/rules:/etc/prometheus/rules:ro  # Agregar este mount
```

---

### P0-3: DVC Endpoint
**Archivo:** `.dvc/config`
**Impacto:** DVC no funciona en Docker

#### Problema Actual
```ini
[remote "minio"]
    url = s3://dvc-storage
    endpointurl = http://localhost:9000  # No funciona en Docker
```

#### Solución
```ini
[remote "minio"]
    url = s3://dvc-storage
    endpointurl = http://minio:9000
```

#### Comando
```bash
dvc remote modify minio endpointurl http://minio:9000
```

---

## FASE 2: P1 - ALTOS (Resolver esta semana)

### P1-1: DVC Pipeline Execution
**Impacto:** Sin reproducibilidad de datos

#### Acciones
```bash
# 1. Verificar que MinIO esté corriendo
docker-compose up -d minio

# 2. Crear bucket si no existe
mc alias set minio http://localhost:9000 $MINIO_ACCESS_KEY $MINIO_SECRET_KEY
mc mb minio/dvc-storage --ignore-existing

# 3. Ejecutar pipeline DVC
cd /path/to/project
dvc repro

# 4. Verificar dvc.lock generado
ls -la dvc.lock

# 5. Push a remote
dvc push
```

---

### P1-2: MLflow Hash Logging
**Archivo:** `airflow/dags/l3_model_training.py`
**Impacto:** Sin trazabilidad data → model

#### Agregar después de línea 560
```python
# ANTES (actual):
mlflow.log_params({
    "version": config["version"],
    "total_timesteps": config["total_timesteps"],
    # ...
})

# DESPUÉS (agregar):
# Obtener hashes de XCom
dataset_hash = ti.xcom_pull(key='dataset_hash', task_ids='validate_dataset')
norm_stats_hash = ti.xcom_pull(key='norm_stats_hash', task_ids='generate_norm_stats')

mlflow.log_params({
    "version": config["version"],
    "total_timesteps": config["total_timesteps"],
    # ... existing params ...

    # AGREGAR ESTOS:
    "dataset_hash": dataset_hash,
    "norm_stats_hash": norm_stats_hash,
    "feature_contract_version": "current",
    "observation_dim": 15,
})

# También loguear como tags para fácil búsqueda
mlflow.set_tags({
    "dataset_hash": dataset_hash[:12],  # Primeros 12 chars
    "norm_stats_hash": norm_stats_hash[:12],
})
```

---

### P1-3: Feast Integration en Inference API
**Archivos:**
- `services/inference_api/core/observation_builder.py`
- `services/inference_api/core/feature_adapter.py`

#### Modificar observation_builder.py
```python
# Agregar import
from src.feature_store.feast_service import FeastInferenceService

class ObservationBuilder:
    def __init__(self, norm_stats_path: Optional[Path] = None):
        # Existente
        self._adapter = InferenceFeatureAdapter(norm_stats_path=norm_path)

        # AGREGAR: Inicializar Feast service con fallback
        self._feast_service = None
        try:
            self._feast_service = FeastInferenceService(
                enable_fallback=True,
                fallback_builder=self._adapter._canonical_builder
            )
            logger.info("FeastInferenceService initialized for online feature serving")
        except Exception as e:
            logger.warning(f"Feast not available, using direct calculation: {e}")

    def build_observation(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        position: float,
        session_progress: float = 0.5
    ) -> np.ndarray:
        """Build observation, trying Feast first then falling back to adapter."""

        # Intentar Feast primero si está disponible
        if self._feast_service and self._feast_service.is_available():
            try:
                symbol = "USDCOP"
                bar_id = str(df.index[bar_idx])

                return self._feast_service.get_features(
                    symbol=symbol,
                    bar_id=bar_id,
                    position=position,
                    time_normalized=session_progress
                )
            except Exception as e:
                logger.warning(f"Feast retrieval failed, using adapter: {e}")

        # Fallback a adapter directo
        return self._adapter.build_observation(
            df=df,
            bar_idx=bar_idx,
            position=position,
            session_progress=session_progress,
            check_circuit_breaker=True
        )
```

---

### P1-4: Vault Integration en DAGs
**Archivos:**
- `airflow/dags/l0_ohlcv_realtime.py`
- `airflow/dags/l0_macro_unified.py`
- `airflow/dags/l3_model_training.py`
- `airflow/dags/l5_multi_model_inference.py`

#### Template para todos los DAGs
```python
# Agregar al inicio del archivo
import sys
sys.path.insert(0, '/opt/airflow')

from src.shared.secrets import get_vault_client

# Función helper para obtener secrets
def get_secret(path: str, key: str, env_fallback: str = None) -> str:
    """Get secret from Vault with env fallback."""
    try:
        client = get_vault_client()
        return client.get_secret(path, key)
    except Exception as e:
        if env_fallback:
            value = os.environ.get(env_fallback)
            if value:
                logging.warning(f"Using env fallback for {path}/{key}")
                return value
        raise

# REEMPLAZAR en l0_ohlcv_realtime.py:
# ANTES:
TWELVEDATA_API_KEY = os.environ.get('TWELVEDATA_API_KEY_1')

# DESPUÉS:
TWELVEDATA_API_KEY = get_secret(
    'trading/twelvedata',
    'api_key_1',
    env_fallback='TWELVEDATA_API_KEY_1'
)

# REEMPLAZAR en l3_model_training.py:
# ANTES:
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI')

# DESPUÉS:
MLFLOW_TRACKING_URI = get_secret(
    'trading/mlflow',
    'tracking_uri',
    env_fallback='MLFLOW_TRACKING_URI'
)
```

---

## FASE 3: P2 - MEDIOS (Resolver este mes)

### P2-1: Prometheus Business Metrics
**Archivo:** `services/inference_api/routers/metrics.py` (NUEVO)

```python
"""Prometheus metrics endpoint for inference API."""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi import APIRouter, Response

router = APIRouter(tags=["metrics"])

# Counters
TRADING_SIGNALS_TOTAL = Counter(
    'usdcop_trading_signals_total',
    'Total trading signals generated',
    ['action', 'model_id']
)

INFERENCE_REQUESTS_TOTAL = Counter(
    'usdcop_inference_requests_total',
    'Total inference requests',
    ['status']
)

FEATURE_ERRORS_TOTAL = Counter(
    'usdcop_feature_calculation_errors_total',
    'Feature calculation errors',
    ['feature_name']
)

# Histograms
INFERENCE_LATENCY = Histogram(
    'usdcop_model_inference_duration_seconds',
    'Model inference latency in seconds',
    ['model_id'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

FEATURE_LATENCY = Histogram(
    'usdcop_feature_computation_duration_seconds',
    'Feature computation latency',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1]
)

# Gauges
ACTIVE_MODELS = Gauge(
    'usdcop_active_models',
    'Number of active models'
)

SIGNAL_CONFIDENCE = Gauge(
    'usdcop_signal_confidence',
    'Current signal confidence',
    ['model_id']
)

FEATURE_DRIFT_SCORE = Gauge(
    'usdcop_feature_drift_score',
    'Feature drift score (KS statistic)',
    ['feature_name']
)

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )
```

### P2-2: norm_stats_hash Validation Enforcement
**Archivo:** `services/inference_api/core/observation_builder.py`

```python
# Modificar __init__ para requerir hash validation
def __init__(
    self,
    norm_stats_path: Optional[Path] = None,
    expected_norm_stats_hash: Optional[str] = None  # AGREGAR
):
    norm_path = str(norm_stats_path) if norm_stats_path else None

    # Obtener hash esperado de config o variable de entorno
    if expected_norm_stats_hash is None:
        expected_norm_stats_hash = os.environ.get('EXPECTED_NORM_STATS_HASH')

    try:
        self._adapter = InferenceFeatureAdapter(
            norm_stats_path=norm_path,
            expected_hash=expected_norm_stats_hash  # Pasar al adapter
        )
    except NormStatsHashMismatchError as e:
        logger.critical(f"CRITICAL: Norm stats hash mismatch! {e}")
        raise
```

### P2-3: FeatureCache Schema Update
**Archivo:** `services/mlops/feature_cache.py`

```python
# Actualizar DEFAULT_FEATURE_ORDER para usar 15-feature SSOT
from src.feature_store.core import FEATURE_ORDER

DEFAULT_FEATURE_ORDER = FEATURE_ORDER  # Usar SSOT directamente

# O explícitamente:
DEFAULT_FEATURE_ORDER = (
    "log_ret_5m", "log_ret_1h", "log_ret_4h",
    "rsi_9", "atr_pct", "adx_14",
    "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
    "brent_change_1d", "rate_spread", "usdmxn_change_1d",
    "position", "time_normalized"
)
```

### P2-4: Model Signatures en MLflow
**Archivo:** `scripts/train_with_mlflow.py`

```python
from mlflow.models import infer_signature
import numpy as np

def log_model_with_signature(self, model_path: str):
    """Log model with input/output signature."""

    # Crear sample input (15-dim observation)
    sample_input = np.zeros((1, 15), dtype=np.float32)

    # Sample output (continuous action)
    sample_output = np.array([[0.0]], dtype=np.float32)

    # Inferir signature
    signature = infer_signature(sample_input, sample_output)

    # Log model con signature
    mlflow.pytorch.log_model(
        pytorch_model=self.model,
        artifact_path="model",
        signature=signature,
        input_example=sample_input,
        registered_model_name=f"usdcop_ppo_{self.config['version']}"
    )
```

---

## FASE 4: P3 - BAJOS (Cuando sea posible)

### P3-1: Exponential Backoff en DAGs
```python
# En default_args de cada DAG
default_args = {
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
    'retry_exponential_backoff': True,  # AGREGAR
    'max_retry_delay': timedelta(minutes=30),  # AGREGAR
}
```

### P3-2: Connection Pooling en DAGs
```python
# Reemplazar get_db_connection() en utils/dag_common.py
from services.common.database import get_connection_pool

_pool = None

def get_db_connection():
    global _pool
    if _pool is None:
        _pool = get_connection_pool()
    return _pool.getconn()

def release_connection(conn):
    global _pool
    if _pool:
        _pool.putconn(conn)
```

### P3-3: Docker Secrets Full Adoption
```yaml
# En docker-compose.yml, para cada servicio:
airflow-webserver:
  secrets:
    - db_password
    - airflow_fernet_key
  environment:
    POSTGRES_PASSWORD_FILE: /run/secrets/db_password
    AIRFLOW__CORE__FERNET_KEY_FILE: /run/secrets/airflow_fernet_key
```

---

## Orden de Ejecución

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SECUENCIA DE IMPLEMENTACIÓN                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DÍA 1 (P0 - Críticos):                                                    │
│  ══════════════════════                                                     │
│  09:00  P0-1: L1 DAG Feature Parity (el más importante)                    │
│  11:00  P0-2: Prometheus Config                                            │
│  12:00  P0-3: DVC Endpoint                                                 │
│  13:00  Tests de regresión para P0                                         │
│                                                                             │
│  DÍA 2-3 (P1 - Altos):                                                     │
│  ══════════════════════                                                     │
│  DÍA 2:                                                                     │
│    09:00  P1-1: DVC Pipeline Execution                                     │
│    11:00  P1-2: MLflow Hash Logging                                        │
│  DÍA 3:                                                                     │
│    09:00  P1-3: Feast Integration                                          │
│    14:00  P1-4: Vault Integration en DAGs                                  │
│                                                                             │
│  SEMANA 2-4 (P2 - Medios):                                                 │
│  ══════════════════════════                                                 │
│  Semana 2: P2-1 Prometheus metrics, P2-2 Hash validation                   │
│  Semana 3: P2-3 FeatureCache update, P2-4 Model signatures                 │
│                                                                             │
│  CUANDO SEA POSIBLE (P3 - Bajos):                                          │
│  ═════════════════════════════════                                          │
│  P3-1, P3-2, P3-3                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Validación Post-Remediación

### Tests a Ejecutar
```bash
# Después de P0
pytest tests/regression/test_feature_builder_parity.py -v
pytest tests/unit/test_builder_delegation.py -v
pytest tests/integration/test_infrastructure.py -v

# Después de P1
dvc status  # Verificar dvc.lock existe
curl http://localhost:9090/api/v1/rules  # Verificar alert rules cargadas
curl http://localhost:6566/health  # Verificar Feast

# Verificación Manual
docker-compose logs prometheus | grep "rule"
docker-compose logs alertmanager | grep "ready"
```

### Métricas de Éxito
| Métrica | Antes | Después |
|---------|-------|---------|
| Alert rules activas | 0 | 40+ |
| Feature parity tests | ❌ | ✅ |
| dvc.lock existe | ❌ | ✅ |
| MLflow tiene hashes | ❌ | ✅ |
| Feast hit rate | 0% | >80% |
| Vault adoption DAGs | 0% | 100% |

---

## Archivos a Modificar (Resumen)

| Archivo | Fase | Cambios |
|---------|------|---------|
| `airflow/dags/l1_feature_refresh.py` | P0-1 | Reemplazar calc_* con SSOT imports |
| `docker/prometheus/prometheus.yml` | P0-2 | Agregar rule_files, alertmanagers |
| `.dvc/config` | P0-3 | Cambiar localhost a minio |
| `airflow/dags/l3_model_training.py` | P1-2 | Agregar mlflow.log_param hashes |
| `services/inference_api/core/observation_builder.py` | P1-3 | Agregar FeastInferenceService |
| `airflow/dags/l0_*.py`, `l3_*.py`, `l5_*.py` | P1-4 | Usar VaultClient |
| `services/inference_api/routers/metrics.py` | P2-1 | NUEVO archivo |
| `services/mlops/feature_cache.py` | P2-3 | Actualizar FEATURE_ORDER |
| `scripts/train_with_mlflow.py` | P2-4 | Agregar model signature |

---

*Plan generado: 2026-01-17*
*Autor: Claude Code Audit System*
