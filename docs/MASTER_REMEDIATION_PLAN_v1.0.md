# PLAN MAESTRO DE REMEDIACIÓN v1.0
## USD/COP RL Trading System - Enterprise Operations Upgrade

**Documento**: MASTER_REMEDIATION_PLAN_v1.0.md
**Versión**: 1.0.0
**Fecha**: 2026-01-17
**Autor**: Trading Operations Team
**Estado**: APROBADO PARA IMPLEMENTACIÓN

---

## RESUMEN EJECUTIVO

### Diagnóstico Consolidado (3 Auditorías)

| Auditoría | Score | Evaluación |
|-----------|-------|------------|
| **Integración de Infraestructura** | 72% (22/30) | Aceptable con gaps |
| **Workflows Operacionales** | 57% (114/200) | Alto Riesgo |
| **Código General** | 87.6% | Bueno |
| **Promedio Ponderado** | **68%** | Requiere Remediación |

### Score Objetivo

| Métrica | Actual | Objetivo | Gap |
|---------|--------|----------|-----|
| Integración | 72% | 90% | +18% |
| Workflows | 57% | 85% | +28% |
| General | 87.6% | 95% | +7.4% |
| **Total** | **68%** | **90%** | **+22%** |

### Timeline General (6 Semanas)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ SEMANA 1    │ SEMANA 2    │ SEMANA 3    │ SEMANA 4    │ SEMANA 5-6              │
├──────────────────────────────────────────────────────────────────────────────────┤
│   FASE 0    │        FASE 1        │    FASE 2    │    FASE 3    │   FASE 4    │
│  Critical   │    Infrastructure    │  Dashboard   │  Governance  │   Polish    │
│   Fixes     │       MLOps          │    Ops UI    │  & Policies  │   & Test    │
│             │                      │              │              │             │
│ L1 DAG      │ MLflow Hashes        │ Kill Switch  │ Gov Policy   │ E2E Tests   │
│ Prometheus  │ Slack Notif          │ Rollback UI  │ Model Cards  │ Game Days   │
│ Vault DAGs  │ Feast Cache          │ Promote UI   │ Post-mortem  │ DR Drills   │
│ DVC Config  │                      │ Alerts Panel │ Runbooks     │             │
└──────────────────────────────────────────────────────────────────────────────────┘
      5d            5d                    5d             5d              10d
```

---

# FASE 0: CRITICAL FIXES (Días 1-5)

**Objetivo**: Corregir issues críticos que bloquean el funcionamiento correcto del sistema.

## 0.1 L1 DAG FEATURE PARITY [P0-BLOCKER]

### Problema
El DAG L1 (`l1_feature_refresh.py`) calcula features con lógica duplicada en lugar de usar el SSOT `CanonicalFeatureBuilder`.

### Diagnóstico
- **Gap**: Calculadores duplicados de RSI, ATR, ADX
- **Riesgo**: Training/Serving skew - features diferentes entre entrenamiento e inferencia
- **Impacto**: Predicciones incorrectas del modelo

### Solución

**Archivo**: `airflow/dags/l1_feature_refresh.py`

```python
"""
L1 Feature Refresh DAG - SSOT Edition
======================================
P0-1: Uses CanonicalFeatureBuilder as Single Source of Truth

Contract: CTR-FEAT-001 - 15 features in canonical order
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import logging

# SSOT Import - This is the canonical feature calculator
from src.feature_store.canonical_builder import CanonicalFeatureBuilder
from src.feature_store.core import FEATURE_ORDER  # Contract: 15 features

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'trading_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['trading@company.com'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'l1_feature_refresh',
    default_args=default_args,
    description='Feature refresh using SSOT CanonicalFeatureBuilder',
    schedule_interval='*/5 8-16 * * 1-5',  # Every 5 min during trading hours
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['features', 'ssot', 'production'],
)


def compute_features_ssot(**context) -> dict:
    """
    Compute features using the Single Source of Truth builder.

    This ensures:
    1. Same calculations as training (Wilder's EMA for RSI/ATR/ADX)
    2. Same feature order (CTR-FEAT-001)
    3. Same normalization stats
    """
    from airflow.hooks.postgres_hook import PostgresHook

    # Initialize SSOT builder
    builder = CanonicalFeatureBuilder()

    # Fetch latest OHLCV data
    pg_hook = PostgresHook(postgres_conn_id='trading_db')

    query = """
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv_usdcop_5m
        WHERE timestamp >= NOW() - INTERVAL '7 days'
        ORDER BY timestamp ASC
    """
    df = pg_hook.get_pandas_df(query)

    if df.empty:
        raise ValueError("No OHLCV data found for feature computation")

    # Compute features using SSOT - guaranteed contract compliance
    features = builder.compute_features(df)
    latest_features = features.iloc[-1].to_dict()

    # Validate feature count matches contract
    assert len(latest_features) == len(FEATURE_ORDER), (
        f"Feature count mismatch: got {len(latest_features)}, expected {len(FEATURE_ORDER)}"
    )

    # Build feature vector in canonical order
    feature_vector = [latest_features[f] for f in FEATURE_ORDER]

    # Store in database with hash for validation
    insert_query = """
        INSERT INTO feature_cache (timestamp, features_json, feature_vector, builder_version)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (timestamp) DO UPDATE SET
            features_json = EXCLUDED.features_json,
            feature_vector = EXCLUDED.feature_vector,
            updated_at = NOW()
    """

    import json
    pg_hook.run(insert_query, parameters=(
        df['timestamp'].iloc[-1].isoformat(),
        json.dumps(latest_features),
        feature_vector,
        builder.VERSION,
    ))

    logger.info(f"SSOT features computed: {len(feature_vector)} features, builder v{builder.VERSION}")

    return {
        'timestamp': df['timestamp'].iloc[-1].isoformat(),
        'feature_count': len(feature_vector),
        'builder_version': builder.VERSION,
    }


def validate_feature_consistency(**context) -> bool:
    """
    Validate features match expected schema and ranges.

    Part of CTR-FEAT-001 contract enforcement.
    """
    from airflow.hooks.postgres_hook import PostgresHook

    pg_hook = PostgresHook(postgres_conn_id='trading_db')

    # Get latest computed features
    query = """
        SELECT features_json, builder_version
        FROM feature_cache
        ORDER BY timestamp DESC
        LIMIT 1
    """
    result = pg_hook.get_first(query)

    if not result:
        raise ValueError("No features found for validation")

    import json
    features = json.loads(result[0])
    builder_version = result[1]

    # Validate all expected features present
    missing = set(FEATURE_ORDER) - set(features.keys())
    if missing:
        raise ValueError(f"Missing features: {missing}")

    # Validate ranges
    validations = {
        'rsi_9': (0, 100),
        'atr_pct': (0, 0.1),  # 0-10%
        'adx_14': (0, 100),
        'position': (-1, 1),
        'time_normalized': (0, 1),
    }

    for feature, (min_val, max_val) in validations.items():
        if feature in features:
            val = features[feature]
            if not (min_val <= val <= max_val):
                logger.warning(f"Feature {feature}={val} outside expected range [{min_val}, {max_val}]")

    logger.info(f"Feature validation passed: {len(features)} features, builder v{builder_version}")
    return True


# Task definitions
compute_features = PythonOperator(
    task_id='compute_features_ssot',
    python_callable=compute_features_ssot,
    dag=dag,
)

validate_features = PythonOperator(
    task_id='validate_feature_consistency',
    python_callable=validate_feature_consistency,
    dag=dag,
)

# Pipeline
compute_features >> validate_features
```

### Criterios de Aceptación
- [ ] L1 DAG importa `CanonicalFeatureBuilder` directamente
- [ ] Cero calculadores duplicados en L1
- [ ] Feature order usa `FEATURE_ORDER` de core.py
- [ ] Validación de 15 features exactas
- [ ] Builder version tracked para auditoría

---

## 0.2 PROMETHEUS + ALERTMANAGER [P0-BLOCKER]

### Problema
Prometheus configurado pero sin reglas de alerta ni conexión a AlertManager.

### Solución

**Archivo**: `docker/prometheus/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

# AlertManager integration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Load rules
rule_files:
  - /etc/prometheus/rules/*.yml

scrape_configs:
  - job_name: 'trading-api'
    static_configs:
      - targets: ['trading-api:8000']
    metrics_path: /metrics

  - job_name: 'inference-api'
    static_configs:
      - targets: ['inference-api:8000']
    metrics_path: /metrics

  - job_name: 'airflow'
    static_configs:
      - targets: ['airflow-webserver:8080']
    metrics_path: /metrics

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

**Archivo**: `docker/prometheus/rules/trading_alerts.yml`

```yaml
groups:
  - name: trading_critical
    rules:
      # P0: Trading system down
      - alert: TradingAPIDown
        expr: up{job="trading-api"} == 0
        for: 1m
        labels:
          severity: critical
          team: trading
        annotations:
          summary: "Trading API is down"
          description: "Trading API has been down for more than 1 minute"
          runbook: "https://docs.internal/runbooks/trading-api-down"

      # P0: High error rate
      - alert: HighInferenceErrorRate
        expr: |
          sum(rate(inference_errors_total[5m])) /
          sum(rate(inference_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
          team: ml
        annotations:
          summary: "Inference error rate > 5%"
          description: "Error rate is {{ $value | humanizePercentage }}"
          action: "Consider automatic rollback"

      # P0: Model drift detected
      - alert: FeatureDriftDetected
        expr: feature_drift_psi > 0.2
        for: 15m
        labels:
          severity: warning
          team: ml
        annotations:
          summary: "Feature drift detected (PSI > 0.2)"
          description: "PSI value: {{ $value }}"

  - name: trading_performance
    rules:
      # P1: Low Sharpe ratio
      - alert: LowSharpeRatio
        expr: trading_sharpe_ratio_30d < 0.5
        for: 1h
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "30-day Sharpe ratio below 0.5"
          description: "Current Sharpe: {{ $value }}"

      # P1: High drawdown
      - alert: HighDailyDrawdown
        expr: trading_daily_drawdown_pct > 0.03
        for: 5m
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "Daily drawdown exceeds 3%"
          description: "Current drawdown: {{ $value | humanizePercentage }}"

      # P1: Consecutive losses
      - alert: ConsecutiveLosses
        expr: trading_consecutive_losses >= 5
        for: 0s
        labels:
          severity: warning
          team: trading
        annotations:
          summary: "5+ consecutive losing trades"
          description: "Current streak: {{ $value }} losses"

  - name: infrastructure
    rules:
      # P1: Database connection issues
      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_activity_count / pg_settings_max_connections > 0.8
        for: 5m
        labels:
          severity: warning
          team: infra
        annotations:
          summary: "Database connection pool > 80% utilized"

      # P1: Redis memory high
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
        for: 10m
        labels:
          severity: warning
          team: infra
        annotations:
          summary: "Redis memory usage > 80%"

      # P2: Airflow DAG failures
      - alert: AirflowDAGFailure
        expr: airflow_dag_run_state{state="failed"} > 0
        for: 0s
        labels:
          severity: info
          team: data
        annotations:
          summary: "Airflow DAG {{ $labels.dag_id }} failed"
```

**Archivo**: `docker/alertmanager/alertmanager.yml`

```yaml
global:
  slack_api_url: '${SLACK_WEBHOOK_URL}'
  resolve_timeout: 5m

route:
  receiver: 'slack-notifications'
  group_by: ['alertname', 'severity']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 4h

  routes:
    # P0 Critical - immediate page
    - match:
        severity: critical
      receiver: 'slack-critical'
      group_wait: 0s
      repeat_interval: 15m

    # P1 Warning
    - match:
        severity: warning
      receiver: 'slack-warnings'
      group_wait: 1m

    # P2 Info
    - match:
        severity: info
      receiver: 'slack-info'
      group_wait: 5m

receivers:
  - name: 'slack-critical'
    slack_configs:
      - channel: '#trading-alerts-p0'
        send_resolved: true
        title: '🔴 CRITICAL: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          *{{ .Annotations.summary }}*
          {{ .Annotations.description }}

          *Runbook:* {{ .Annotations.runbook }}
          {{ end }}
        actions:
          - type: button
            text: 'Runbook'
            url: '{{ (index .Alerts 0).Annotations.runbook }}'
          - type: button
            text: 'Dashboard'
            url: 'https://grafana.internal/d/trading'

  - name: 'slack-warnings'
    slack_configs:
      - channel: '#trading-alerts'
        send_resolved: true
        title: '⚠️ WARNING: {{ .GroupLabels.alertname }}'
        text: |
          {{ range .Alerts }}
          {{ .Annotations.summary }}
          {{ .Annotations.description }}
          {{ end }}

  - name: 'slack-info'
    slack_configs:
      - channel: '#trading-info'
        send_resolved: false
        title: 'ℹ️ {{ .GroupLabels.alertname }}'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'slack-notifications'
    slack_configs:
      - channel: '#trading-alerts'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname']
```

### Criterios de Aceptación
- [ ] Prometheus conectado a AlertManager
- [ ] Reglas de alerta para: API down, error rate, drift, sharpe, drawdown
- [ ] Slack channels configurados por severidad
- [ ] Runbook links en alertas
- [ ] Botón de acción en mensajes Slack

---

## 0.3 VAULT IN DAGS [P0-SECURITY]

### Problema
DAGs acceden a secretos directamente de environment variables sin abstracción segura.

### Solución

**Archivo**: `airflow/dags/common/secrets.py`

```python
"""
Secrets Manager for Airflow DAGs
================================
P0-2: Secure credential access with Vault abstraction

Supports:
1. HashiCorp Vault (production)
2. Environment variables (development)
3. Airflow Connections (fallback)
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DatabaseCredentials:
    host: str
    port: int
    database: str
    username: str
    password: str

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class APICredentials:
    api_key: str
    secret_key: Optional[str] = None
    base_url: Optional[str] = None


class SecretsManager:
    """
    Unified secrets access for Airflow DAGs.

    Priority:
    1. Vault (if VAULT_ADDR configured)
    2. Environment variables
    3. Airflow Connections
    """

    def __init__(self):
        self._vault_client = None
        self._vault_available = False
        self._init_vault()

    def _init_vault(self):
        """Initialize Vault client if available."""
        vault_addr = os.environ.get('VAULT_ADDR')
        vault_token = os.environ.get('VAULT_TOKEN')

        if vault_addr and vault_token:
            try:
                import hvac
                self._vault_client = hvac.Client(url=vault_addr, token=vault_token)
                if self._vault_client.is_authenticated():
                    self._vault_available = True
                    logger.info("Vault client initialized successfully")
                else:
                    logger.warning("Vault authentication failed, using fallback")
            except ImportError:
                logger.warning("hvac not installed, Vault unavailable")
            except Exception as e:
                logger.warning(f"Vault init failed: {e}")

    def get_database_credentials(self, db_name: str = "trading") -> DatabaseCredentials:
        """
        Get database credentials.

        Args:
            db_name: Name of database (trading, mlflow, etc.)
        """
        if self._vault_available:
            return self._get_db_from_vault(db_name)
        return self._get_db_from_env(db_name)

    def _get_db_from_vault(self, db_name: str) -> DatabaseCredentials:
        """Get database credentials from Vault."""
        secret_path = f"secret/data/databases/{db_name}"
        response = self._vault_client.secrets.kv.v2.read_secret_version(
            path=f"databases/{db_name}",
            mount_point="secret"
        )
        data = response['data']['data']

        return DatabaseCredentials(
            host=data['host'],
            port=int(data.get('port', 5432)),
            database=data['database'],
            username=data['username'],
            password=data['password'],
        )

    def _get_db_from_env(self, db_name: str) -> DatabaseCredentials:
        """Get database credentials from environment."""
        prefix = db_name.upper()

        # Try DATABASE_URL format first
        db_url = os.environ.get(f'{prefix}_DATABASE_URL') or os.environ.get('DATABASE_URL')
        if db_url:
            return self._parse_database_url(db_url)

        # Fallback to individual vars
        return DatabaseCredentials(
            host=os.environ.get(f'{prefix}_DB_HOST', 'localhost'),
            port=int(os.environ.get(f'{prefix}_DB_PORT', 5432)),
            database=os.environ.get(f'{prefix}_DB_NAME', db_name),
            username=os.environ.get(f'{prefix}_DB_USER', 'postgres'),
            password=os.environ.get(f'{prefix}_DB_PASSWORD', ''),
        )

    def _parse_database_url(self, url: str) -> DatabaseCredentials:
        """Parse DATABASE_URL format."""
        from urllib.parse import urlparse
        parsed = urlparse(url)

        return DatabaseCredentials(
            host=parsed.hostname or 'localhost',
            port=parsed.port or 5432,
            database=parsed.path.lstrip('/'),
            username=parsed.username or 'postgres',
            password=parsed.password or '',
        )

    def get_api_credentials(self, service: str) -> APICredentials:
        """
        Get API credentials for external services.

        Args:
            service: Service name (twelvedata, slack, etc.)
        """
        if self._vault_available:
            return self._get_api_from_vault(service)
        return self._get_api_from_env(service)

    def _get_api_from_vault(self, service: str) -> APICredentials:
        """Get API credentials from Vault."""
        response = self._vault_client.secrets.kv.v2.read_secret_version(
            path=f"apis/{service}",
            mount_point="secret"
        )
        data = response['data']['data']

        return APICredentials(
            api_key=data['api_key'],
            secret_key=data.get('secret_key'),
            base_url=data.get('base_url'),
        )

    def _get_api_from_env(self, service: str) -> APICredentials:
        """Get API credentials from environment."""
        prefix = service.upper()

        return APICredentials(
            api_key=os.environ.get(f'{prefix}_API_KEY', ''),
            secret_key=os.environ.get(f'{prefix}_SECRET_KEY'),
            base_url=os.environ.get(f'{prefix}_BASE_URL'),
        )

    def get_mlflow_tracking_uri(self) -> str:
        """Get MLflow tracking URI."""
        if self._vault_available:
            try:
                response = self._vault_client.secrets.kv.v2.read_secret_version(
                    path="mlflow/tracking",
                    mount_point="secret"
                )
                return response['data']['data']['tracking_uri']
            except Exception:
                pass

        return os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5000')


# Global singleton
_secrets_manager: Optional[SecretsManager] = None


def get_secrets() -> SecretsManager:
    """Get or create the global secrets manager."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


# Convenience functions for common use cases
def get_trading_db() -> DatabaseCredentials:
    """Get trading database credentials."""
    return get_secrets().get_database_credentials("trading")


def get_twelvedata_key() -> str:
    """Get TwelveData API key."""
    return get_secrets().get_api_credentials("twelvedata").api_key


def get_slack_webhook() -> str:
    """Get Slack webhook URL."""
    return get_secrets().get_api_credentials("slack").api_key
```

### Uso en DAGs

```python
# En cualquier DAG
from common.secrets import get_trading_db, get_twelvedata_key

def my_task():
    db = get_trading_db()
    conn = psycopg2.connect(db.connection_string)

    api_key = get_twelvedata_key()
    # Use securely...
```

### Criterios de Aceptación
- [ ] SecretsManager abstrae acceso a credenciales
- [ ] Soporte para Vault + fallback a env vars
- [ ] DAGs usan `get_secrets()` en lugar de `os.environ.get()`
- [ ] No hay credenciales hardcodeadas
- [ ] Logging sin exponer secretos

---

## 0.4 DVC CONFIGURATION [P0-REPRODUCIBILITY]

### Problema
DVC remote apunta a endpoint incorrecto y no tiene lock file.

### Solución

**Archivo**: `.dvc/config`

```ini
[core]
    remote = minio
    autostage = true

[remote "minio"]
    url = s3://dvc-storage
    endpointurl = http://minio:9000
    access_key_id = ${DVC_ACCESS_KEY}
    secret_access_key = ${DVC_SECRET_KEY}

[remote "backup"]
    url = s3://dvc-backup
    endpointurl = https://s3.us-east-1.amazonaws.com
    # Production backup, credentials via AWS env vars
```

**Archivo**: `.dvc/.gitignore`

```
/config.local
/tmp
/cache
# DO NOT ignore lock file - it must be tracked!
```

**Archivo**: `dvc.lock` (generado, pero debe estar en git)

```yaml
# This file is auto-generated by DVC and should be committed
schema: '2.0'
stages:
  prepare_data:
    cmd: python scripts/prepare_data.py
    deps:
      - path: data/raw/
        hash: md5
        md5: abc123...
    outs:
      - path: data/processed/
        hash: md5
        md5: def456...

  train_model:
    cmd: python scripts/train.py
    deps:
      - path: data/processed/
        hash: md5
        md5: def456...
      - path: src/models/ppo_agent.py
        hash: md5
        md5: ghi789...
    outs:
      - path: models/
        hash: md5
        md5: jkl012...
```

### Comando de Validación

```bash
# Verificar configuración
dvc remote list
# Debe mostrar: minio -> s3://dvc-storage

# Verificar lock file trackeado
git status dvc.lock
# Debe estar tracked, no ignored

# Validar reproducibilidad
dvc repro --dry-run
# Debe mostrar pipeline sin cambios si todo está actualizado
```

### Criterios de Aceptación
- [ ] DVC remote apunta a MinIO correcto
- [ ] `dvc.lock` está en git (no en .gitignore)
- [ ] Pipeline reproducible con `dvc repro`
- [ ] Backup remote configurado
- [ ] Credenciales via variables de entorno

---

# FASE 1: MLOPS INFRASTRUCTURE (Días 6-10)

## 1.1 MLFLOW HASH LOGGING [P1-HIGH]

### Problema
MLflow no registra hashes de norm_stats.json ni dataset, impidiendo validación de consistencia.

### Solución

**Archivo**: `airflow/dags/l3_model_training.py` (agregar a training task)

```python
def log_training_artifacts(run_id: str, model_path: Path, norm_stats_path: Path):
    """
    Log training artifacts with hashes for reproducibility.

    Implements CTR-HASH-001: All model artifacts must have verifiable hashes.
    """
    import mlflow
    import hashlib
    import json

    def compute_file_hash(path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def compute_json_hash(path: Path) -> str:
        """Compute hash of JSON content (sorted keys for consistency)."""
        with open(path) as f:
            data = json.load(f)
        # Sort for deterministic ordering
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()

    with mlflow.start_run(run_id=run_id):
        # Log model artifact hash
        model_hash = compute_file_hash(model_path)
        mlflow.log_param("model_hash", model_hash[:16])  # First 16 chars for display
        mlflow.set_tag("model_hash_full", model_hash)

        # Log norm_stats hash (CRITICAL for inference consistency)
        norm_stats_hash = compute_json_hash(norm_stats_path)
        mlflow.log_param("norm_stats_hash", norm_stats_hash[:16])
        mlflow.set_tag("norm_stats_hash_full", norm_stats_hash)

        # Log dataset hash from DVC
        try:
            import subprocess
            result = subprocess.run(
                ['dvc', 'hash', 'data/processed/'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                dataset_hash = result.stdout.strip()
                mlflow.log_param("dataset_hash", dataset_hash[:16])
                mlflow.set_tag("dataset_hash_full", dataset_hash)
        except Exception as e:
            logger.warning(f"Could not compute dataset hash: {e}")

        # Log feature order hash (contract validation)
        from src.feature_store.core import FEATURE_ORDER
        feature_order_str = ",".join(FEATURE_ORDER)
        feature_order_hash = hashlib.sha256(feature_order_str.encode()).hexdigest()
        mlflow.log_param("feature_order_hash", feature_order_hash[:16])

        logger.info(f"Artifacts logged with hashes: model={model_hash[:16]}, norm_stats={norm_stats_hash[:16]}")
```

### Criterios de Aceptación
- [ ] MLflow run contiene `model_hash` param
- [ ] MLflow run contiene `norm_stats_hash` param
- [ ] MLflow run contiene `dataset_hash` param (via DVC)
- [ ] MLflow run contiene `feature_order_hash` param
- [ ] Hashes validables durante inference

---

## 1.2 SLACK NOTIFICATIONS [P1-HIGH]

### Problema
No hay notificaciones de eventos críticos al equipo.

### Solución

**Archivo**: `src/shared/notifications/slack_client.py`

```python
"""
Slack Notification Client
=========================
P1-1: Real-time notifications for trading events

Events:
- Model promotions/rollbacks
- Trading alerts (drawdown, losses)
- System health issues
- Drift detection
"""

import os
import logging
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import aiohttp

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SlackMessage:
    channel: str
    text: str
    blocks: Optional[List[Dict]] = None
    thread_ts: Optional[str] = None


class SlackClient:
    """
    Async Slack client for trading notifications.
    """

    SEVERITY_EMOJI = {
        AlertSeverity.INFO: "ℹ️",
        AlertSeverity.WARNING: "⚠️",
        AlertSeverity.ERROR: "🔴",
        AlertSeverity.CRITICAL: "🚨",
    }

    SEVERITY_COLOR = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ff9800",
        AlertSeverity.ERROR: "#f44336",
        AlertSeverity.CRITICAL: "#9c27b0",
    }

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.environ.get('SLACK_WEBHOOK_URL')
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.INFO,
        fields: Optional[Dict[str, str]] = None,
        actions: Optional[List[Dict]] = None,
    ):
        """
        Send formatted alert to Slack.

        Args:
            title: Alert title
            message: Alert description
            severity: Severity level
            fields: Key-value pairs to display
            actions: Action buttons
        """
        if not self.webhook_url:
            logger.warning("Slack webhook not configured, skipping notification")
            return

        emoji = self.SEVERITY_EMOJI.get(severity, "")
        color = self.SEVERITY_COLOR.get(severity, "#808080")

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} {title}",
                    "emoji": True
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message
                }
            }
        ]

        if fields:
            field_blocks = []
            for key, value in fields.items():
                field_blocks.append({
                    "type": "mrkdwn",
                    "text": f"*{key}:*\n{value}"
                })

            # Slack allows max 10 fields, 2 per row
            for i in range(0, len(field_blocks), 2):
                blocks.append({
                    "type": "section",
                    "fields": field_blocks[i:i+2]
                })

        if actions:
            blocks.append({
                "type": "actions",
                "elements": actions
            })

        # Add timestamp footer
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} COT"
                }
            ]
        })

        payload = {
            "attachments": [
                {
                    "color": color,
                    "blocks": blocks
                }
            ]
        }

        try:
            session = await self._get_session()
            async with session.post(self.webhook_url, json=payload) as response:
                if response.status != 200:
                    logger.error(f"Slack notification failed: {response.status}")
                else:
                    logger.debug(f"Slack notification sent: {title}")
        except Exception as e:
            logger.error(f"Slack notification error: {e}")

    async def notify_model_promotion(
        self,
        model_id: str,
        from_stage: str,
        to_stage: str,
        promoted_by: str,
        metrics: Optional[Dict] = None,
    ):
        """Notify model promotion event."""
        fields = {
            "Model": model_id,
            "Transition": f"{from_stage} → {to_stage}",
            "Promoted By": promoted_by,
        }

        if metrics:
            fields["Sharpe"] = f"{metrics.get('sharpe', 'N/A'):.2f}"
            fields["Win Rate"] = f"{metrics.get('win_rate', 0) * 100:.1f}%"

        await self.send_alert(
            title=f"Model Promoted to {to_stage.upper()}",
            message=f"Model `{model_id}` has been promoted.",
            severity=AlertSeverity.INFO,
            fields=fields,
            actions=[
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View in MLflow"},
                    "url": f"http://mlflow:5000/#/models/{model_id}"
                }
            ]
        )

    async def notify_rollback(
        self,
        from_model: str,
        to_model: str,
        reason: str,
        initiated_by: str,
    ):
        """Notify model rollback event."""
        await self.send_alert(
            title="MODEL ROLLBACK",
            message=f"Production model rolled back due to: {reason}",
            severity=AlertSeverity.WARNING,
            fields={
                "Previous Model": from_model,
                "New Model": to_model,
                "Initiated By": initiated_by,
                "Reason": reason,
            }
        )

    async def notify_kill_switch(self, reason: str, activated_by: str):
        """Notify kill switch activation."""
        await self.send_alert(
            title="🔴 KILL SWITCH ACTIVATED",
            message="ALL TRADING HAS BEEN STOPPED",
            severity=AlertSeverity.CRITICAL,
            fields={
                "Reason": reason,
                "Activated By": activated_by,
                "Action": "All positions closed",
            },
            actions=[
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View Dashboard"},
                    "url": "https://dashboard.internal/operations"
                }
            ]
        )

    async def notify_drift_detected(
        self,
        feature: str,
        psi_value: float,
        threshold: float = 0.2,
    ):
        """Notify feature drift detection."""
        await self.send_alert(
            title="Feature Drift Detected",
            message=f"Feature `{feature}` shows significant drift.",
            severity=AlertSeverity.WARNING,
            fields={
                "Feature": feature,
                "PSI Value": f"{psi_value:.3f}",
                "Threshold": f"{threshold:.3f}",
                "Status": "⚠️ Above threshold" if psi_value > threshold else "✅ Within limits",
            }
        )

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()


# Global instance
_slack_client: Optional[SlackClient] = None


def get_slack_client() -> SlackClient:
    """Get or create global Slack client."""
    global _slack_client
    if _slack_client is None:
        _slack_client = SlackClient()
    return _slack_client


# Sync wrapper for non-async contexts
def send_slack_alert(title: str, message: str, severity: str = "info", **kwargs):
    """Synchronous wrapper for sending Slack alerts."""
    client = get_slack_client()
    severity_enum = AlertSeverity(severity)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(
        client.send_alert(title, message, severity_enum, **kwargs)
    )
```

### Criterios de Aceptación
- [ ] SlackClient envía notificaciones formateadas
- [ ] Soporte para diferentes severidades con colores
- [ ] Métodos específicos para: promotion, rollback, kill switch, drift
- [ ] Timestamps en timezone Colombia (COT)
- [ ] Botones de acción con links

---

## 1.3 FEAST CACHING IN INFERENCE [P1-HIGH]

### Problema
Inference API no usa Feast para features cacheados, causa latencia alta y posible inconsistencia.

### Solución

**Archivo**: `services/inference_api/core/cached_inference.py`

```python
"""
Cached Inference Engine
=======================
P1-2: Uses Feast feature store for low-latency inference

Features:
- Online feature retrieval from Redis via Feast
- Fallback to direct computation if cache miss
- Consistency validation with training features
"""

import logging
from typing import Optional, Dict, Any, Tuple
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Feast
try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    logger.warning("Feast not installed, using fallback feature retrieval")


class CachedInferenceEngine:
    """
    Inference engine with Feast-backed feature caching.

    Provides:
    1. Low-latency feature retrieval from online store
    2. Fallback computation for cache misses
    3. Feature consistency validation
    """

    # Feature view and entity configuration
    FEATURE_VIEW = "trading_features"
    ENTITY_KEY = "timestamp"

    # Expected features (must match training)
    EXPECTED_FEATURES = [
        "log_ret_5m",
        "log_ret_1h",
        "log_ret_4h",
        "rsi_9",
        "atr_pct",
        "adx_14",
        "dxy_z",
        "dxy_change_1d",
        "vix_z",
        "embi_z",
        "brent_change_1d",
        "rate_spread",
        "usdmxn_change_1d",
        "position",
        "time_normalized",
    ]

    def __init__(
        self,
        feature_store_path: str = "feature_store/",
        fallback_enabled: bool = True,
    ):
        self.fallback_enabled = fallback_enabled
        self.store: Optional[FeatureStore] = None
        self._cache_hits = 0
        self._cache_misses = 0

        if FEAST_AVAILABLE:
            try:
                self.store = FeatureStore(repo_path=feature_store_path)
                logger.info("Feast feature store initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Feast: {e}")

    def get_features(
        self,
        timestamp: str,
        position: float = 0.0,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Get features for inference.

        Args:
            timestamp: ISO format timestamp
            position: Current position (-1 to 1)

        Returns:
            Tuple of (feature_vector, metadata)
        """
        metadata = {
            "timestamp": timestamp,
            "source": "unknown",
            "latency_ms": 0,
        }

        start_time = datetime.now()

        # Try Feast online store first
        if self.store:
            try:
                features = self._get_from_feast(timestamp)
                if features is not None:
                    self._cache_hits += 1
                    metadata["source"] = "feast_online"

                    # Add runtime features
                    features["position"] = position
                    features["time_normalized"] = self._compute_time_normalized()

                    # Build vector
                    vector = self._build_feature_vector(features)

                    metadata["latency_ms"] = (datetime.now() - start_time).total_seconds() * 1000
                    return vector, metadata

            except Exception as e:
                logger.warning(f"Feast retrieval failed: {e}")

        # Fallback to direct computation
        if self.fallback_enabled:
            self._cache_misses += 1
            features = self._compute_features_direct(timestamp)
            features["position"] = position
            features["time_normalized"] = self._compute_time_normalized()

            vector = self._build_feature_vector(features)

            metadata["source"] = "computed"
            metadata["latency_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            return vector, metadata

        raise ValueError(f"Could not retrieve features for {timestamp}")

    def _get_from_feast(self, timestamp: str) -> Optional[Dict[str, float]]:
        """Retrieve features from Feast online store."""
        entity_rows = [{"timestamp": timestamp}]

        feature_refs = [
            f"{self.FEATURE_VIEW}:{f}"
            for f in self.EXPECTED_FEATURES
            if f not in ["position", "time_normalized"]  # Runtime features
        ]

        result = self.store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        # Check if we got valid features
        features = {}
        for key, values in result.items():
            if key != self.ENTITY_KEY:
                feature_name = key.split(":")[-1] if ":" in key else key
                if values and values[0] is not None:
                    features[feature_name] = float(values[0])

        if len(features) < len(self.EXPECTED_FEATURES) - 2:  # Minus runtime features
            return None

        return features

    def _compute_features_direct(self, timestamp: str) -> Dict[str, float]:
        """Compute features directly from database (fallback)."""
        from src.feature_store.canonical_builder import CanonicalFeatureBuilder

        builder = CanonicalFeatureBuilder()
        # Implementation would fetch data and compute
        # This is a simplified placeholder

        return {f: 0.0 for f in self.EXPECTED_FEATURES}

    def _build_feature_vector(self, features: Dict[str, float]) -> np.ndarray:
        """Build feature vector in canonical order."""
        vector = []
        for feature_name in self.EXPECTED_FEATURES:
            value = features.get(feature_name, 0.0)
            if value is None:
                value = 0.0
            vector.append(float(value))

        return np.array(vector, dtype=np.float32)

    def _compute_time_normalized(self) -> float:
        """Compute normalized trading time (0-1)."""
        now = datetime.now()
        # Trading hours: 8:00 - 16:00 COT
        trading_start = 8 * 60  # minutes
        trading_end = 16 * 60
        current_minutes = now.hour * 60 + now.minute

        if current_minutes < trading_start:
            return 0.0
        elif current_minutes > trading_end:
            return 1.0
        else:
            return (current_minutes - trading_start) / (trading_end - trading_start)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache hit/miss statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": self._cache_hits / total if total > 0 else 0.0,
            "feast_available": self.store is not None,
        }

    def validate_feature_consistency(
        self,
        features: Dict[str, float],
        expected_hash: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate features match expected schema and optionally hash."""
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
        }

        # Check all expected features present
        missing = set(self.EXPECTED_FEATURES) - set(features.keys())
        if missing:
            result["valid"] = False
            result["errors"].append(f"Missing features: {missing}")

        # Check for unexpected features
        extra = set(features.keys()) - set(self.EXPECTED_FEATURES)
        if extra:
            result["warnings"].append(f"Extra features ignored: {extra}")

        # Validate ranges
        validations = {
            "rsi_9": (0, 100),
            "atr_pct": (0, 0.5),
            "adx_14": (0, 100),
            "position": (-1, 1),
            "time_normalized": (0, 1),
        }

        for feature, (min_val, max_val) in validations.items():
            if feature in features:
                val = features[feature]
                if not (min_val <= val <= max_val):
                    result["warnings"].append(
                        f"{feature}={val:.4f} outside expected [{min_val}, {max_val}]"
                    )

        return result
```

### Criterios de Aceptación
- [ ] Inference usa Feast online store para features
- [ ] Fallback a cómputo directo si Feast falla
- [ ] Cache hit/miss metrics
- [ ] Validación de consistencia de features
- [ ] Latencia < 100ms para cache hit

---

# FASE 2: DASHBOARD OPERATIONS UI (Días 11-15)

## 2.1 KILL SWITCH UI [P0-CRITICAL]

(Ver código completo en WORKFLOW_REMEDIATION_PLAN - sección 1.1)

### Resumen de Implementación

| Componente | Archivo | Descripción |
|------------|---------|-------------|
| Backend API | `services/inference_api/routers/operations.py` | Endpoints kill-switch, resume, status |
| Frontend Component | `components/operations/KillSwitch.tsx` | Botón rojo con confirmación |
| Header Integration | `components/layout/DashboardHeader.tsx` | Kill switch compacto en header |
| Estado Global | Redis-backed state | Estado persistente del kill switch |

### Criterios de Aceptación
- [ ] Botón rojo visible en header
- [ ] Confirmación con razón requerida
- [ ] Cierre automático de posiciones
- [ ] Notificación Slack inmediata
- [ ] Log en audit trail
- [ ] Resume requiere código CONFIRM_RESUME

---

## 2.2 ROLLBACK & PROMOTE UI [P1-HIGH]

(Ver código completo en WORKFLOW_REMEDIATION_PLAN - secciones 1.2 y 2.1)

### Resumen de Implementación

| Componente | Archivo | Descripción |
|------------|---------|-------------|
| Rollback API | `services/inference_api/routers/models.py` | Endpoint rollback con atomic swap |
| Rollback UI | `components/models/RollbackPanel.tsx` | Panel con versiones disponibles |
| Promote UI | `components/models/PromoteButton.tsx` | Botón con checklist y validación |

### Criterios de Aceptación Rollback
- [ ] Muestra últimas 5 versiones
- [ ] Métricas de cada versión visible
- [ ] Razón requerida
- [ ] Rollback < 60 segundos
- [ ] Atomic swap en base de datos

### Criterios de Aceptación Promote
- [ ] Validación de métricas automática
- [ ] Checklist de promoción requerido
- [ ] Diferentes thresholds por stage
- [ ] Notificación al equipo

---

## 2.3 ALERTS PANEL [P1-HIGH]

### Problema
No hay visualización de alertas activas en el dashboard.

### Solución

**Archivo**: `usdcop-trading-dashboard/components/alerts/AlertsPanel.tsx`

```typescript
'use client';

import { useState, useEffect } from 'react';
import { Bell, AlertTriangle, Info, XCircle, CheckCircle } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet';

interface Alert {
  id: string;
  severity: 'critical' | 'warning' | 'info';
  title: string;
  description: string;
  timestamp: string;
  status: 'firing' | 'resolved';
  labels: Record<string, string>;
  runbook?: string;
}

export function AlertsPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    // Poll for alerts every 30 seconds
    const fetchAlerts = async () => {
      try {
        const response = await fetch('/api/v1/alerts');
        const data = await response.json();
        setAlerts(data.alerts || []);
      } catch (error) {
        console.error('Failed to fetch alerts:', error);
      }
    };

    fetchAlerts();
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, []);

  const firingAlerts = alerts.filter(a => a.status === 'firing');
  const criticalCount = firingAlerts.filter(a => a.severity === 'critical').length;
  const warningCount = firingAlerts.filter(a => a.severity === 'warning').length;

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return <XCircle className="h-4 w-4 text-red-500" />;
      case 'warning': return <AlertTriangle className="h-4 w-4 text-yellow-500" />;
      default: return <Info className="h-4 w-4 text-blue-500" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return 'bg-red-100 border-red-300';
      case 'warning': return 'bg-yellow-100 border-yellow-300';
      default: return 'bg-blue-100 border-blue-300';
    }
  };

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button variant="ghost" size="sm" className="relative">
          <Bell className="h-5 w-5" />
          {firingAlerts.length > 0 && (
            <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75" />
              <span className={`relative inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-bold text-white ${
                criticalCount > 0 ? 'bg-red-500' : 'bg-yellow-500'
              }`}>
                {firingAlerts.length}
              </span>
            </span>
          )}
        </Button>
      </SheetTrigger>
      <SheetContent className="w-[400px] sm:w-[540px]">
        <SheetHeader>
          <SheetTitle className="flex items-center gap-2">
            <Bell className="h-5 w-5" />
            Alertas del Sistema
          </SheetTitle>
          <SheetDescription>
            {criticalCount > 0 && (
              <Badge variant="destructive" className="mr-2">
                {criticalCount} Critical
              </Badge>
            )}
            {warningCount > 0 && (
              <Badge variant="outline" className="text-yellow-600">
                {warningCount} Warning
              </Badge>
            )}
            {firingAlerts.length === 0 && (
              <span className="text-green-600 flex items-center gap-1">
                <CheckCircle className="h-4 w-4" />
                No hay alertas activas
              </span>
            )}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-6 space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
          {/* Firing alerts first */}
          {firingAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`p-4 rounded-lg border ${getSeverityColor(alert.severity)}`}
            >
              <div className="flex items-start gap-3">
                {getSeverityIcon(alert.severity)}
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium">{alert.title}</h4>
                    <Badge variant={alert.status === 'firing' ? 'destructive' : 'outline'}>
                      {alert.status}
                    </Badge>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{alert.description}</p>
                  <div className="flex items-center gap-4 mt-2 text-xs text-gray-500">
                    <span>{new Date(alert.timestamp).toLocaleString()}</span>
                    {alert.labels.team && <span>Team: {alert.labels.team}</span>}
                  </div>
                  {alert.runbook && (
                    <a
                      href={alert.runbook}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-xs text-blue-600 hover:underline mt-2 inline-block"
                    >
                      Ver Runbook →
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))}

          {/* Resolved alerts */}
          {alerts.filter(a => a.status === 'resolved').slice(0, 5).map((alert) => (
            <div
              key={alert.id}
              className="p-4 rounded-lg border bg-gray-50 border-gray-200 opacity-60"
            >
              <div className="flex items-start gap-3">
                <CheckCircle className="h-4 w-4 text-green-500" />
                <div className="flex-1">
                  <div className="flex items-center justify-between">
                    <h4 className="font-medium text-gray-600">{alert.title}</h4>
                    <Badge variant="outline" className="text-green-600">resolved</Badge>
                  </div>
                  <p className="text-sm text-gray-500 mt-1">{alert.description}</p>
                  <span className="text-xs text-gray-400">
                    {new Date(alert.timestamp).toLocaleString()}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </SheetContent>
    </Sheet>
  );
}
```

### Criterios de Aceptación
- [ ] Icono de campana en header con badge de conteo
- [ ] Panel lateral con alertas activas y resueltas
- [ ] Colores por severidad
- [ ] Links a runbooks
- [ ] Auto-refresh cada 30 segundos
- [ ] Animación de ping para alertas críticas

---

# FASE 3: GOVERNANCE & POLICIES (Días 16-20)

## 3.1 MODEL GOVERNANCE POLICY [P1-HIGH]

(Ver documento completo en WORKFLOW_REMEDIATION_PLAN - sección 2.2)

**Archivo**: `docs/MODEL_GOVERNANCE_POLICY.md`

### Contenido Clave
- Definición de stages (Development → Registered → Staging → Production → Archived)
- Requisitos cuantitativos por transición
- Roles y responsabilidades (Owner, Backup, On-Call)
- Alertas y thresholds de monitoreo
- Política de retraining
- Política de retirement
- Template de Model Card

---

## 3.2 MODEL CARDS [P2-MEDIUM]

### Problema
No existe documentación estructurada por modelo.

### Solución

**Archivo**: `docs/templates/MODEL_CARD.md`

```markdown
# Model Card: {MODEL_ID}

## Basic Information
| Field | Value |
|-------|-------|
| Model ID | {model_id} |
| Version | {version} |
| Created Date | {created_date} |
| Owner | {owner} |
| Backup Owner | {backup_owner} |
| Current Stage | {stage} |

## Training Details
| Field | Value |
|-------|-------|
| Training Period | {start_date} to {end_date} |
| Dataset Version | {dataset_hash} |
| Feature Count | 15 (CTR-FEAT-001) |
| Training Time | {training_hours} hours |
| MLflow Run ID | {mlflow_run_id} |

## Artifact Hashes
| Artifact | Hash (SHA256) |
|----------|---------------|
| Model (.zip) | {model_hash} |
| norm_stats.json | {norm_stats_hash} |
| Dataset | {dataset_hash} |
| Feature Order | {feature_order_hash} |

## Performance Metrics

### Backtest Performance
| Metric | Value | Threshold |
|--------|-------|-----------|
| Sharpe Ratio | {backtest_sharpe} | ≥ 1.0 |
| Win Rate | {backtest_win_rate}% | ≥ 50% |
| Max Drawdown | {backtest_max_dd}% | ≤ 10% |
| Total Trades | {backtest_trades} | ≥ 100 |
| Profit Factor | {profit_factor} | ≥ 1.5 |

### Staging Performance (if applicable)
| Metric | Value | Threshold |
|--------|-------|-----------|
| Sharpe Ratio | {staging_sharpe} | ≥ 1.0 |
| Win Rate | {staging_win_rate}% | ≥ 50% |
| Agreement Rate | {agreement_rate}% | ≥ 85% |
| Days in Staging | {staging_days} | ≥ 7 |

## Known Limitations
- {limitation_1}
- {limitation_2}

## Risk Factors
| Risk | Mitigation |
|------|------------|
| {risk_1} | {mitigation_1} |
| {risk_2} | {mitigation_2} |

## Change History
| Date | Change | By |
|------|--------|-----|
| {date} | Initial creation | {author} |
| {date} | Promoted to staging | {author} |
| {date} | Promoted to production | {author} |

---
*Generated: {generation_date}*
*Next Review: {review_date}*
```

### Script de Generación

**Archivo**: `scripts/generate_model_card.py`

```python
"""
Generate Model Card from MLflow and Database
=============================================
Usage: python scripts/generate_model_card.py --model-id ppo_v20_20260115
"""

import argparse
import mlflow
from datetime import datetime
from pathlib import Path
import jinja2

def generate_model_card(model_id: str, output_dir: Path = Path("docs/model_cards")):
    """Generate model card from MLflow run data."""

    # Get MLflow run
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=["1"],
        filter_string=f"tags.model_id = '{model_id}'"
    )

    if not runs:
        raise ValueError(f"No MLflow run found for model_id={model_id}")

    run = runs[0]

    # Extract data
    data = {
        "model_id": model_id,
        "version": run.data.params.get("model_version", "1"),
        "created_date": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d"),
        "owner": run.data.tags.get("owner", "trading_team"),
        "backup_owner": run.data.tags.get("backup_owner", "ml_team"),
        "stage": run.data.tags.get("stage", "registered"),
        # ... more fields
        "model_hash": run.data.params.get("model_hash", "N/A"),
        "norm_stats_hash": run.data.params.get("norm_stats_hash", "N/A"),
        "backtest_sharpe": run.data.metrics.get("backtest_sharpe", 0),
        "backtest_win_rate": run.data.metrics.get("backtest_win_rate", 0) * 100,
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Load template
    template_path = Path("docs/templates/MODEL_CARD.md")
    with open(template_path) as f:
        template = jinja2.Template(f.read())

    # Render
    content = template.render(**data)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_id}.md"
    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Model card generated: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()
    generate_model_card(args.model_id)
```

---

## 3.3 POST-MORTEM TEMPLATE [P2-MEDIUM]

**Archivo**: `docs/templates/POST_MORTEM.md`

```markdown
# Post-Mortem: {INCIDENT_TITLE}

**Incident ID:** INC-{YYYYMMDD}-{NNN}
**Date:** {DATE}
**Duration:** {START_TIME} - {END_TIME} ({DURATION})
**Severity:** P{0-3}
**Status:** {Draft|Reviewed|Closed}

---

## Executive Summary
{One paragraph summary of what happened, impact, and resolution}

## Impact

| Metric | Value |
|--------|-------|
| Trading Downtime | {minutes} minutes |
| Missed Trades | {count} |
| Financial Impact | ${amount} |
| Customers Affected | {count} |

## Timeline (all times in COT)

| Time | Event |
|------|-------|
| HH:MM | First alert triggered |
| HH:MM | On-call acknowledged |
| HH:MM | Initial investigation started |
| HH:MM | Root cause identified |
| HH:MM | Mitigation applied |
| HH:MM | Service restored |
| HH:MM | Incident closed |

## Root Cause Analysis

### What happened?
{Detailed technical explanation of what went wrong}

### Why did it happen?
{Analysis of contributing factors}

### Why wasn't it caught earlier?
{Gaps in monitoring, testing, or process}

## Contributing Factors
1. {Factor 1}
2. {Factor 2}
3. {Factor 3}

## Resolution
{What was done to fix the immediate issue}

## Action Items

| ID | Action | Owner | Due Date | Status |
|----|--------|-------|----------|--------|
| 1 | {action} | {owner} | {date} | {Open/Done} |
| 2 | {action} | {owner} | {date} | {Open/Done} |
| 3 | {action} | {owner} | {date} | {Open/Done} |

## Lessons Learned

### What went well
- {item}
- {item}

### What could be improved
- {item}
- {item}

### What we will do differently
- {item}
- {item}

## Appendix

### Relevant Logs
```
{key log excerpts}
```

### Metrics/Graphs
{Links to relevant Grafana dashboards or screenshots}

### Related Incidents
- {link to related incident}

---

**Post-Mortem Author:** {name}
**Reviewers:** {names}
**Review Date:** {date}
**Next Review:** {date if follow-up needed}
```

---

# FASE 4: POLISH & TEST (Días 21-30)

## 4.1 E2E INTEGRATION TESTS [P1-HIGH]

**Archivo**: `tests/integration/test_e2e_workflow.py`

```python
"""
End-to-End Workflow Integration Tests
=====================================
P1: Validate complete trading system workflows

Workflows tested:
1. Model Training → Backtest → Register → Stage → Production
2. Feature computation → Cache → Inference → Trade
3. Kill Switch → Resume
4. Rollback → Validation
5. Drift Detection → Alert → Retrain trigger
"""

import pytest
import asyncio
from datetime import datetime
import httpx

BASE_URL = "http://localhost:8000/api/v1"


@pytest.fixture
def api_client():
    return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)


class TestModelLifecycle:
    """Test complete model lifecycle workflow."""

    @pytest.mark.asyncio
    async def test_model_promotion_workflow(self, api_client):
        """Test: Registered → Staging → Production"""

        # 1. Get registered model
        response = await api_client.get("/models?status=registered")
        assert response.status_code == 200
        models = response.json()["models"]
        assert len(models) > 0
        model_id = models[0]["model_id"]

        # 2. Promote to staging
        response = await api_client.post(
            f"/models/{model_id}/promote",
            json={
                "target_stage": "staging",
                "reason": "E2E test promotion",
                "promoted_by": "test_suite",
            }
        )
        assert response.status_code == 200

        # 3. Verify in staging
        response = await api_client.get(f"/models/{model_id}")
        assert response.json()["status"] == "staging"

        # 4. Simulate staging period (in real test, wait 7 days)
        # For E2E, we skip time validation

        # 5. Promote to production
        response = await api_client.post(
            f"/models/{model_id}/promote",
            json={
                "target_stage": "production",
                "reason": "E2E test - staging complete",
                "promoted_by": "test_suite",
                "skip_staging_time": True,  # Test override
            }
        )
        assert response.status_code == 200

        # 6. Verify in production
        response = await api_client.get(f"/models/{model_id}")
        assert response.json()["status"] == "deployed"

    @pytest.mark.asyncio
    async def test_rollback_workflow(self, api_client):
        """Test: Production → Rollback → Previous"""

        # 1. Get current production model
        response = await api_client.get("/models?status=deployed")
        assert response.status_code == 200
        current = response.json()["models"][0]

        # 2. Get rollback targets
        response = await api_client.get("/models/rollback-targets")
        assert response.status_code == 200
        targets = response.json()["available_targets"]
        assert len(targets) > 0

        # 3. Execute rollback
        response = await api_client.post(
            "/models/rollback",
            json={
                "reason": "E2E test rollback",
                "initiated_by": "test_suite",
            }
        )
        assert response.status_code == 200
        result = response.json()
        assert result["success"]
        assert result["rollback_time_ms"] < 60000  # < 60 seconds

        # 4. Verify new production
        response = await api_client.get("/models?status=deployed")
        new_current = response.json()["models"][0]
        assert new_current["model_id"] != current["model_id"]


class TestKillSwitch:
    """Test kill switch functionality."""

    @pytest.mark.asyncio
    async def test_kill_and_resume(self, api_client):
        """Test: Kill → Verify → Resume → Verify"""

        # 1. Get initial status
        response = await api_client.get("/operations/status")
        assert response.json()["mode"] == "normal"

        # 2. Activate kill switch
        response = await api_client.post(
            "/operations/kill-switch",
            json={
                "reason": "E2E test kill switch",
                "activated_by": "test_suite",
                "close_positions": False,  # Don't actually close
                "notify_team": False,  # Don't spam Slack
            }
        )
        assert response.status_code == 200
        assert response.json()["mode"] == "killed"

        # 3. Verify killed
        response = await api_client.get("/operations/status")
        assert response.json()["kill_switch_active"]

        # 4. Resume trading
        response = await api_client.post(
            "/operations/resume",
            params={
                "resumed_by": "test_suite",
                "confirmation_code": "CONFIRM_RESUME",
            }
        )
        assert response.status_code == 200

        # 5. Verify normal
        response = await api_client.get("/operations/status")
        assert response.json()["mode"] == "normal"


class TestInferencePipeline:
    """Test inference feature → prediction flow."""

    @pytest.mark.asyncio
    async def test_feature_to_prediction(self, api_client):
        """Test: Get features → Predict → Validate output"""

        # 1. Get latest features
        response = await api_client.get("/features/latest")
        assert response.status_code == 200
        features = response.json()
        assert len(features["feature_vector"]) == 15

        # 2. Run prediction
        response = await api_client.post(
            "/predict",
            json={
                "features": features["feature_vector"],
                "model_id": "ppo_primary",
            }
        )
        assert response.status_code == 200
        prediction = response.json()

        # 3. Validate output
        assert "action" in prediction
        assert "signal" in prediction
        assert prediction["signal"] in ["LONG", "SHORT", "HOLD"]
        assert -1.0 <= prediction["action"] <= 1.0
        assert prediction["latency_ms"] < 100  # Performance SLA


class TestDriftDetection:
    """Test drift detection and alerting."""

    @pytest.mark.asyncio
    async def test_drift_monitoring(self, api_client):
        """Test: Get drift metrics → Validate thresholds"""

        # 1. Get drift status
        response = await api_client.get("/models/router/drift")
        assert response.status_code == 200
        drift = response.json()

        # 2. Validate structure
        assert "psi_scores" in drift
        assert "threshold" in drift
        assert drift["threshold"] == 0.2

        # 3. Check features monitored
        for feature in ["rsi_9", "atr_pct", "adx_14", "dxy_z"]:
            assert feature in drift["psi_scores"]


# Run with: pytest tests/integration/test_e2e_workflow.py -v
```

---

## 4.2 GAME DAY CHECKLIST [P2-MEDIUM]

**Archivo**: `docs/GAME_DAY_CHECKLIST.md`

```markdown
# Game Day Checklist
## Trading System Disaster Recovery Drill

**Purpose:** Validate team's ability to respond to production incidents
**Frequency:** Monthly
**Duration:** 2-4 hours
**Participants:** On-call engineer, Trading lead, ML engineer

---

## Pre-Game Day Preparation (1 day before)

- [ ] Schedule calendar block for all participants
- [ ] Ensure staging environment mirrors production
- [ ] Prepare scenario cards (sealed until game day)
- [ ] Verify all runbooks are accessible
- [ ] Confirm Slack channels active
- [ ] Check PagerDuty integration working

---

## Scenario 1: Kill Switch Activation

**Trigger:** Simulated 10% drawdown in 5 minutes

### Steps
1. [ ] Operator notices alert in #trading-alerts
2. [ ] Operator activates kill switch from dashboard
3. [ ] Verify trading stops within 10 seconds
4. [ ] Verify Slack notification received
5. [ ] Verify positions closed (in staging)
6. [ ] Document incident start time
7. [ ] Investigate simulated cause
8. [ ] Resume trading with confirmation code
9. [ ] Verify trading resumes successfully

### Success Criteria
- [ ] Kill switch activates in < 10 seconds
- [ ] All notifications sent
- [ ] Resume requires confirmation code
- [ ] Audit log captured

### Metrics to Record
| Metric | Value |
|--------|-------|
| Time to detect | ______ |
| Time to kill switch | ______ |
| Time to investigate | ______ |
| Time to resume | ______ |
| Total downtime | ______ |

---

## Scenario 2: Model Rollback

**Trigger:** Simulated model degradation (Sharpe < 0.3)

### Steps
1. [ ] Operator receives degradation alert
2. [ ] Operator opens rollback panel
3. [ ] Operator selects previous model version
4. [ ] Operator provides rollback reason
5. [ ] Execute rollback
6. [ ] Verify new model loaded
7. [ ] Verify predictions using new model
8. [ ] Document rollback in incident log

### Success Criteria
- [ ] Rollback completes in < 60 seconds
- [ ] No trades during rollback
- [ ] Previous model metrics visible
- [ ] Notification sent

### Metrics to Record
| Metric | Value |
|--------|-------|
| Rollback execution time | ______ |
| Inference downtime | ______ |
| First prediction after rollback | ______ |

---

## Scenario 3: Data Source Outage

**Trigger:** Simulated TwelveData API failure

### Steps
1. [ ] Operator receives stale data alert
2. [ ] Verify trading auto-paused
3. [ ] Check TwelveData status page
4. [ ] Wait for recovery OR activate kill switch
5. [ ] When data resumes, verify feature quality
6. [ ] Resume trading if quality OK

### Success Criteria
- [ ] Auto-pause triggered on stale data
- [ ] Clear messaging about data status
- [ ] Trading doesn't resume with stale features

---

## Scenario 4: Database Failover

**Trigger:** Primary database becomes unavailable

### Steps
1. [ ] Database alert received
2. [ ] Verify automatic failover to replica
3. [ ] Check trading API still responsive
4. [ ] Verify write operations work
5. [ ] Document recovery time

### Success Criteria
- [ ] Failover < 30 seconds
- [ ] No data loss
- [ ] Trading continues or gracefully pauses

---

## Post-Game Day Review

### Debrief Checklist
- [ ] All scenarios completed
- [ ] Metrics recorded for each scenario
- [ ] Issues identified
- [ ] Runbooks updated if needed
- [ ] Action items assigned
- [ ] Post-mortem document created

### Action Items Template
| Issue Found | Action | Owner | Due Date |
|-------------|--------|-------|----------|
| | | | |
| | | | |

### Sign-off
| Role | Name | Date |
|------|------|------|
| On-Call Engineer | | |
| Trading Lead | | |
| ML Engineer | | |

---

*Next Game Day: {DATE}*
*Previous Game Day: {DATE}*
*Document Owner: Trading Operations*
```

---

# RESUMEN DE IMPLEMENTACIÓN

## Timeline Día por Día

### Semana 1: Critical Fixes (Fase 0)
| Día | Tarea | Entregable |
|-----|-------|------------|
| 1-2 | L1 DAG SSOT Integration | `l1_feature_refresh.py` actualizado |
| 2-3 | Prometheus + AlertManager | Reglas de alerta configuradas |
| 3-4 | Vault in DAGs | `common/secrets.py` implementado |
| 4-5 | DVC Configuration | `.dvc/config` + `dvc.lock` trackeado |

### Semana 2: MLOps Infrastructure (Fase 1)
| Día | Tarea | Entregable |
|-----|-------|------------|
| 6-7 | MLflow Hash Logging | Training con hashes registrados |
| 7-8 | Slack Notifications | `slack_client.py` funcional |
| 8-10 | Feast Caching | Inference con feature cache |

### Semana 3: Dashboard Ops (Fase 2)
| Día | Tarea | Entregable |
|-----|-------|------------|
| 11-12 | Kill Switch UI | Botón rojo en header |
| 13-14 | Rollback & Promote UI | Panels en modelo |
| 15 | Alerts Panel | Panel de alertas |

### Semana 4: Governance (Fase 3)
| Día | Tarea | Entregable |
|-----|-------|------------|
| 16-17 | Governance Policy | Documento aprobado |
| 18-19 | Model Cards | Template + script generador |
| 20 | Post-Mortem Template | Template listo |

### Semanas 5-6: Polish & Test (Fase 4)
| Día | Tarea | Entregable |
|-----|-------|------------|
| 21-25 | E2E Tests | Suite de tests pasando |
| 26-28 | Game Day Prep | Checklist + scenarios |
| 29 | Game Day Execution | Drill completado |
| 30 | Re-auditoría | Score ≥85% |

---

## Archivos a Crear (Total: 18)

### Backend (7)
```
airflow/dags/common/secrets.py
services/inference_api/routers/operations.py
services/inference_api/services/auto_rollback.py
services/inference_api/core/cached_inference.py
src/shared/notifications/slack_client.py
scripts/generate_model_card.py
tests/integration/test_e2e_workflow.py
```

### Frontend (5)
```
components/operations/KillSwitch.tsx
components/models/RollbackPanel.tsx
components/models/PromoteButton.tsx
components/alerts/AlertsPanel.tsx
app/incidents/page.tsx
```

### Config (2)
```
docker/prometheus/rules/trading_alerts.yml
docker/alertmanager/alertmanager.yml
```

### Documentation (4)
```
docs/MODEL_GOVERNANCE_POLICY.md
docs/INCIDENT_RESPONSE_PLAYBOOK.md
docs/templates/MODEL_CARD.md
docs/templates/POST_MORTEM.md
docs/GAME_DAY_CHECKLIST.md
```

## Archivos a Modificar (6)

```
airflow/dags/l1_feature_refresh.py           # SSOT integration
airflow/dags/l3_model_training.py            # Hash logging
docker/prometheus/prometheus.yml              # AlertManager connection
.dvc/config                                   # Endpoint fix
services/inference_api/routers/models.py      # Rollback endpoint
components/layout/DashboardHeader.tsx         # Kill switch + alerts
```

---

## Métricas de Éxito

| Métrica | Objetivo | Validación |
|---------|----------|------------|
| Score Auditoría | ≥85% (170/200) | Re-auditoría Día 30 |
| Kill Switch Response | <10 segundos | Game Day test |
| Rollback Time | <60 segundos | E2E test |
| Notification Latency | <30 segundos | Monitoring |
| E2E Tests | 100% passing | CI/CD |
| Game Day | All scenarios pass | Checklist signed |

---

**Documento preparado por**: Trading Operations Team
**Fecha**: 2026-01-17
**Próxima revisión**: 2026-02-28 (Post-implementación)

---

*Este documento consolida los hallazgos de 3 auditorías y establece el plan de remediación completo para alcanzar madurez operacional enterprise-grade.*
