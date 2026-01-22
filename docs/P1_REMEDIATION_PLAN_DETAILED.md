# Plan de Remediación P1 - Detallado
## USD/COP RL Trading System

**Fecha**: 2026-01-17
**Versión**: 1.0.0
**Prioridad**: P1 (8 items críticos)

---

## Resumen Ejecutivo

Este documento detalla el plan de implementación para las **8 brechas P1** identificadas en la auditoría de 1400 preguntas, más las **17 brechas P2/P3** en el backlog.

### Cronograma de Sprints

| Sprint | Duración | Items | Esfuerzo |
|--------|----------|-------|----------|
| **Sprint 1** | 1-2 semanas | CONT-45, CONT-46 | 4-6 horas |
| **Sprint 2** | 2-3 semanas | FEAST-19, REPRO-97 | 16-20 horas |
| **Sprint 3** | 3-4 semanas | EXP-15, EXP-16, COMP-88 | 24-32 horas |
| **Sprint 4** | 4+ semanas | SEC-30 | Externo |
| **Backlog** | Ongoing | P2/P3 (17 items) | Variable |

---

# SPRINT 1: Consolidación de Contratos (CONT-45, CONT-46)

## Estimado: 4-6 horas

### Problema Actual

El `FEATURE_ORDER` está definido en **3 lugares diferentes**, creando riesgo de inconsistencia:

```
1. src/core/contracts/feature_contract.py    ← SSOT (Single Source of Truth)
2. src/features/contract.py                   ← Wrapper con fallback
3. src/feature_store/core.py                  ← Importa de #1, pero tiene fallback
```

Además, hay **múltiples import paths** que generan confusión:

```python
# Variantes actuales (confusión)
from src.core.contracts.feature_contract import FEATURE_ORDER
from src.core.contracts import FEATURE_ORDER
from src.features.contract import FEATURE_ORDER
from feature_store.core import FEATURE_ORDER
```

### Solución Propuesta

#### Paso 1: Establecer Import Path Único (1 hora)

**Archivo**: `src/core/contracts/__init__.py` - Ya exporta correctamente.

**Acción**: Crear alias en `src/__init__.py` para simplificar imports:

```python
# src/__init__.py - NUEVO
"""
USD/COP RL Trading System - Main Package
========================================

Import path canónico para contratos:
    from src.core.contracts import FEATURE_ORDER, OBSERVATION_DIM, Action

NUNCA importar directamente de:
    - src.features.contract
    - src.feature_store.core
    - airflow.dags.contracts

Estos son wrappers de compatibilidad que serán deprecados.
"""

# Canonical re-exports
from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    FEATURE_ORDER_HASH,
    Action,
    ACTION_SELL,
    ACTION_HOLD,
    ACTION_BUY,
)

__all__ = [
    "FEATURE_ORDER",
    "OBSERVATION_DIM",
    "FEATURE_CONTRACT",
    "FEATURE_ORDER_HASH",
    "Action",
    "ACTION_SELL",
    "ACTION_HOLD",
    "ACTION_BUY",
]
```

#### Paso 2: Deprecar Archivos Redundantes (2 horas)

**Archivo 1**: `src/features/contract.py`

```python
# src/features/contract.py - MODIFICACIÓN
"""
DEPRECATED: Este módulo será removido en v3.0.0

Por favor use:
    from src.core.contracts import FEATURE_ORDER, FEATURE_CONTRACT

Este archivo existe solo por compatibilidad hacia atrás.
"""
import warnings
from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    FeatureContract,
)

# Emit deprecation warning on import
warnings.warn(
    "src.features.contract is deprecated. "
    "Use 'from src.core.contracts import FEATURE_ORDER' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for compatibility
__all__ = ["FEATURE_ORDER", "OBSERVATION_DIM", "FEATURE_CONTRACT", "FeatureContract"]

def get_contract(version: str = "current") -> FeatureContract:
    """DEPRECATED: Use FEATURE_CONTRACT directly."""
    warnings.warn(
        "get_contract() is deprecated. Use FEATURE_CONTRACT directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return FEATURE_CONTRACT
```

**Archivo 2**: `src/feature_store/core.py` - Eliminar fallback local:

```python
# src/feature_store/core.py - MODIFICACIÓN (líneas 59-71)
# ANTES:
try:
    from src.core.contracts import FEATURE_ORDER
    _contracts_available = True
except ImportError:
    _contracts_available = False
    SSOT_FEATURE_ORDER = None  # ELIMINAR

# DESPUÉS:
from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
)
# No fallback - contracts son requeridos
```

#### Paso 3: Script de Migración Automática (1 hora)

**Archivo**: `scripts/migrate_imports.py`

```python
#!/usr/bin/env python3
"""
Migrate deprecated contract imports to canonical path.

Usage:
    python scripts/migrate_imports.py --check     # Solo verificar
    python scripts/migrate_imports.py --fix       # Aplicar cambios
"""
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Patrones a migrar
MIGRATION_PATTERNS = [
    # (patrón viejo, reemplazo nuevo)
    (
        r"from src\.features\.contract import",
        "from src.core.contracts import"
    ),
    (
        r"from feature_store\.core import (FEATURE_ORDER|OBSERVATION_DIM)",
        "from src.core.contracts import \\1"
    ),
    (
        r"from src\.core\.contracts\.feature_contract import",
        "from src.core.contracts import"
    ),
]

EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv"}
EXCLUDE_FILES = {"migrate_imports.py"}  # No modificar este script


def find_python_files(root: Path) -> List[Path]:
    """Find all Python files in project."""
    files = []
    for path in root.rglob("*.py"):
        if not any(excluded in path.parts for excluded in EXCLUDE_DIRS):
            if path.name not in EXCLUDE_FILES:
                files.append(path)
    return files


def check_file(path: Path) -> List[Tuple[int, str, str]]:
    """Check file for deprecated imports. Returns list of (line_num, old, new)."""
    issues = []
    try:
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            for pattern, replacement in MIGRATION_PATTERNS:
                if re.search(pattern, line):
                    new_line = re.sub(pattern, replacement, line)
                    if new_line != line:
                        issues.append((i, line.strip(), new_line.strip()))
    except Exception as e:
        print(f"Error reading {path}: {e}")

    return issues


def fix_file(path: Path) -> int:
    """Fix deprecated imports in file. Returns count of fixes."""
    try:
        content = path.read_text(encoding="utf-8")
        original = content

        for pattern, replacement in MIGRATION_PATTERNS:
            content = re.sub(pattern, replacement, content)

        if content != original:
            path.write_text(content, encoding="utf-8")
            return 1
        return 0
    except Exception as e:
        print(f"Error fixing {path}: {e}")
        return 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate deprecated imports")
    parser.add_argument("--check", action="store_true", help="Only check, don't fix")
    parser.add_argument("--fix", action="store_true", help="Apply fixes")
    args = parser.parse_args()

    if not args.check and not args.fix:
        parser.print_help()
        sys.exit(1)

    root = Path(__file__).parent.parent
    files = find_python_files(root)

    total_issues = 0
    files_with_issues = []

    for path in files:
        issues = check_file(path)
        if issues:
            total_issues += len(issues)
            files_with_issues.append((path, issues))

    if args.check:
        print(f"\n{'='*60}")
        print(f"DEPRECATED IMPORTS CHECK")
        print(f"{'='*60}\n")

        for path, issues in files_with_issues:
            print(f"\n{path.relative_to(root)}:")
            for line_num, old, new in issues:
                print(f"  Line {line_num}:")
                print(f"    OLD: {old}")
                print(f"    NEW: {new}")

        print(f"\n{'='*60}")
        print(f"Total: {total_issues} issues in {len(files_with_issues)} files")
        print(f"{'='*60}")

        sys.exit(0 if total_issues == 0 else 1)

    if args.fix:
        fixed = 0
        for path, _ in files_with_issues:
            if fix_file(path):
                fixed += 1
                print(f"Fixed: {path.relative_to(root)}")

        print(f"\nFixed {fixed} files")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

#### Paso 4: Agregar CI Check (30 min)

**Archivo**: `.github/workflows/contracts-check.yml`

```yaml
name: Contracts Consistency Check

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  check-imports:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Check for deprecated imports
        run: |
          python scripts/migrate_imports.py --check

      - name: Verify FEATURE_ORDER consistency
        run: |
          python -c "
          from src.core.contracts import FEATURE_ORDER, FEATURE_ORDER_HASH
          print(f'FEATURE_ORDER: {len(FEATURE_ORDER)} features')
          print(f'FEATURE_ORDER_HASH: {FEATURE_ORDER_HASH}')
          assert len(FEATURE_ORDER) == 15, 'Must have exactly 15 features'
          print('✓ Contract consistency verified')
          "
```

#### Paso 5: Documentar Import Path Canónico (30 min)

**Actualizar**: `docs/CONTRACTS_GUIDE.md`

```markdown
# Guía de Contratos - Import Path Canónico

## IMPORTANTE: Siempre usar este import

```python
from src.core.contracts import (
    FEATURE_ORDER,
    OBSERVATION_DIM,
    FEATURE_CONTRACT,
    Action,
)
```

## Imports DEPRECADOS (no usar)

```python
# ❌ DEPRECATED - No usar
from src.features.contract import FEATURE_ORDER
from feature_store.core import FEATURE_ORDER
from src.core.contracts.feature_contract import FEATURE_ORDER
```

## Verificación

```bash
# Verificar que no hay imports deprecados
python scripts/migrate_imports.py --check

# Corregir automáticamente
python scripts/migrate_imports.py --fix
```
```

### Checklist Sprint 1

- [ ] Crear `src/__init__.py` con exports canónicos
- [ ] Deprecar `src/features/contract.py` con warning
- [ ] Eliminar fallback en `src/feature_store/core.py`
- [ ] Crear `scripts/migrate_imports.py`
- [ ] Ejecutar migración en codebase
- [ ] Agregar CI check
- [ ] Actualizar documentación
- [ ] Verificar tests pasan

---

# SPRINT 2: Feature Drift Detection + CI Reproduction (FEAST-19, REPRO-97)

## Estimado: 16-20 horas

### FEAST-19: Feature Drift Detection

El archivo `src/monitoring/drift_detector.py` ya existe con implementación básica. Necesitamos:

1. **Integrar con Inference API** (4 horas)
2. **Crear Airflow DAG para drift monitoring** (3 horas)
3. **Agregar alertas en Grafana** (2 horas)
4. **Documentar umbrales y acciones** (1 hora)

#### Implementación Detallada

**Paso 1**: Integrar drift detector en Inference API

```python
# services/inference_api/middleware/drift_monitor.py - NUEVO
"""
Feature drift monitoring middleware for inference API.
"""
import logging
from typing import Optional
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.monitoring.drift_detector import FeatureDriftDetector, DriftReport
from src.core.contracts import FEATURE_ORDER

logger = logging.getLogger(__name__)


class DriftMonitorMiddleware(BaseHTTPMiddleware):
    """
    Middleware that monitors feature drift during inference.
    Collects feature observations and periodically checks for drift.
    """

    def __init__(
        self,
        app,
        reference_stats_path: str = "config/reference_stats.json",
        check_interval: int = 100,  # Check every N requests
        p_value_threshold: float = 0.01
    ):
        super().__init__(app)
        self.check_interval = check_interval
        self.request_count = 0

        self.drift_detector = FeatureDriftDetector(
            reference_stats_path=reference_stats_path,
            p_value_threshold=p_value_threshold,
            window_size=1000,
            min_samples=100
        )

        logger.info(
            f"DriftMonitorMiddleware initialized: "
            f"check_interval={check_interval}"
        )

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Check if this was a prediction request with features
        if hasattr(request.state, "features_used"):
            self._add_observation(request.state.features_used)

        return response

    def _add_observation(self, features: dict):
        """Add observation and check drift periodically."""
        self.drift_detector.add_observation(features)
        self.request_count += 1

        if self.request_count % self.check_interval == 0:
            self._check_and_alert()

    def _check_and_alert(self):
        """Check for drift and emit alerts."""
        report = self.drift_detector.get_drift_report()

        if report.alert_active:
            drifted = [r.feature_name for r in report.drift_results if r.is_drifted]
            logger.warning(
                f"DRIFT ALERT: {report.features_drifted}/{report.features_checked} "
                f"features drifted: {drifted}"
            )
            # TODO: Send to alertmanager

    def get_drift_report(self) -> Optional[DriftReport]:
        """Get current drift report."""
        return self.drift_detector.get_drift_report()
```

**Paso 2**: Crear endpoint de drift status

```python
# services/inference_api/routers/drift.py - NUEVO
"""
Drift monitoring endpoints.
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

router = APIRouter(prefix="/drift", tags=["drift"])


class FeatureDriftStatus(BaseModel):
    feature_name: str
    ks_statistic: float
    p_value: float
    is_drifted: bool
    drift_severity: str
    reference_mean: float
    current_mean: float


class DriftReportResponse(BaseModel):
    timestamp: datetime
    features_checked: int
    features_drifted: int
    overall_drift_score: float
    alert_active: bool
    drifted_features: List[str]
    details: List[FeatureDriftStatus]


@router.get("/status", response_model=DriftReportResponse)
async def get_drift_status():
    """Get current feature drift status."""
    # Implementation uses drift_detector from app state
    pass


@router.post("/reset")
async def reset_drift_windows():
    """Reset drift detection windows (admin only)."""
    pass


@router.get("/history")
async def get_drift_history(days: int = 7):
    """Get historical drift reports."""
    pass
```

**Paso 3**: Crear Airflow DAG para drift monitoring

```python
# airflow/dags/l6_drift_monitoring.py - NUEVO
"""
L6 Drift Monitoring DAG
=======================

Runs daily drift analysis on inference data.
Compares recent feature distributions against training reference.

Schedule: Daily at 00:30 UTC
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "mlops",
    "depends_on_past": False,
    "email_on_failure": True,
    "email": ["mlops@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    dag_id="l6_drift_monitoring",
    default_args=default_args,
    description="Daily feature drift analysis",
    schedule_interval="30 0 * * *",  # Daily at 00:30 UTC
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "monitoring", "drift"],
)


def compute_daily_drift(**context):
    """Compute drift statistics for the past 24 hours."""
    from src.monitoring.drift_detector import FeatureDriftDetector
    import pandas as pd
    from sqlalchemy import create_engine
    import os

    # Load inference data from past 24 hours
    engine = create_engine(os.environ["DATABASE_URL"])

    query = """
    SELECT state_features, created_at
    FROM trading.model_inferences
    WHERE created_at >= NOW() - INTERVAL '24 hours'
    ORDER BY created_at
    """

    df = pd.read_sql(query, engine)

    if len(df) < 100:
        print(f"Only {len(df)} inferences in last 24h, skipping drift check")
        return {"skipped": True, "reason": "insufficient_data"}

    # Parse features from JSON
    import json
    features_list = [json.loads(row) for row in df["state_features"]]

    # Run drift detection
    detector = FeatureDriftDetector(
        reference_stats_path="config/reference_stats.json",
        p_value_threshold=0.01
    )

    detector.add_batch(features_list)
    report = detector.get_drift_report()

    # Store results
    result = {
        "date": context["ds"],
        "features_checked": report.features_checked,
        "features_drifted": report.features_drifted,
        "overall_drift_score": report.overall_drift_score,
        "alert_active": report.alert_active,
        "drifted_features": [r.feature_name for r in report.drift_results if r.is_drifted],
    }

    # Push to XCom for alerting
    context["ti"].xcom_push(key="drift_report", value=result)

    return result


def alert_on_drift(**context):
    """Send alert if significant drift detected."""
    ti = context["ti"]
    report = ti.xcom_pull(task_ids="compute_drift", key="drift_report")

    if report.get("skipped"):
        print("Skipped - no alert needed")
        return

    if report["alert_active"]:
        # Send to Slack/PagerDuty
        from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook

        message = f"""
:warning: *Feature Drift Alert*
Date: {report['date']}
Features Drifted: {report['features_drifted']}/{report['features_checked']}
Drift Score: {report['overall_drift_score']:.4f}
Affected Features: {', '.join(report['drifted_features'])}
        """

        try:
            hook = SlackWebhookHook(slack_webhook_conn_id="slack_alerts")
            hook.send(text=message)
        except Exception as e:
            print(f"Failed to send Slack alert: {e}")


with dag:
    start = EmptyOperator(task_id="start")

    compute_drift = PythonOperator(
        task_id="compute_drift",
        python_callable=compute_daily_drift,
    )

    alert = PythonOperator(
        task_id="alert_on_drift",
        python_callable=alert_on_drift,
    )

    end = EmptyOperator(task_id="end")

    start >> compute_drift >> alert >> end
```

**Paso 4**: Configurar alertas en Prometheus

```yaml
# config/prometheus/rules/drift_alerts.yml - NUEVO
groups:
  - name: feature_drift
    interval: 1m
    rules:
      - alert: FeatureDriftDetected
        expr: features_drifted_count > 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Feature drift detected"
          description: "{{ $value }} features showing significant drift"

      - alert: FeatureDriftCritical
        expr: features_drifted_count >= 3
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical feature drift - multiple features affected"
          description: "{{ $value }} features showing drift. Model retraining may be needed."

      - alert: SingleFeatureHighDrift
        expr: feature_drift_score > 0.3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High drift on feature {{ $labels.feature_name }}"
          description: "KS statistic {{ $value }} exceeds threshold"
```

### REPRO-97: CI Reproduction Test

**Paso 1**: Crear test de reproducibilidad en CI

```yaml
# .github/workflows/reproducibility-test.yml - NUEVO
name: Reproducibility Test

on:
  schedule:
    - cron: '0 3 * * 0'  # Weekly on Sunday at 3 AM
  workflow_dispatch:
    inputs:
      mlflow_run_id:
        description: 'MLflow run ID to reproduce'
        required: false

jobs:
  reproduce-experiment:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install dvc[s3]

      - name: Configure DVC
        run: |
          dvc remote modify minio --local access_key_id ${{ secrets.MINIO_ACCESS_KEY }}
          dvc remote modify minio --local secret_access_key ${{ secrets.MINIO_SECRET_KEY }}

      - name: Get latest production run
        id: get_run
        run: |
          RUN_ID=${{ github.event.inputs.mlflow_run_id }}
          if [ -z "$RUN_ID" ]; then
            # Get latest production model run
            RUN_ID=$(python -c "
            import mlflow
            mlflow.set_tracking_uri('${{ secrets.MLFLOW_TRACKING_URI }}')
            client = mlflow.tracking.MlflowClient()
            versions = client.search_model_versions(\"name='ppo_usdcop'\")
            prod = [v for v in versions if v.current_stage == 'Production']
            if prod:
                print(prod[0].run_id)
            ")
          fi
          echo "run_id=$RUN_ID" >> $GITHUB_OUTPUT

      - name: Reproduce experiment
        id: reproduce
        run: |
          python scripts/reproduce_from_run.py \
            --run-id ${{ steps.get_run.outputs.run_id }} \
            --verify-hash \
            --output-dir ./reproduction_test

      - name: Verify reproduction
        run: |
          python -c "
          import json
          with open('reproduction_test/verification_report.json') as f:
              report = json.load(f)

          if not report['hash_match']:
              print('FAIL: Model hash mismatch')
              print(f\"Original: {report['original_hash']}\")
              print(f\"Reproduced: {report['reproduced_hash']}\")
              exit(1)

          print('SUCCESS: Model hash matches')
          print(f\"Hash: {report['original_hash']}\")
          "

      - name: Upload verification report
        uses: actions/upload-artifact@v4
        with:
          name: reproduction-report
          path: reproduction_test/verification_report.json

      - name: Notify on failure
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          webhook: ${{ secrets.SLACK_WEBHOOK }}
          payload: |
            {
              "text": ":x: Reproducibility test failed for run ${{ steps.get_run.outputs.run_id }}"
            }
```

**Paso 2**: Crear script de reproducción

```python
# scripts/reproduce_from_run.py - NUEVO
#!/usr/bin/env python3
"""
Reproduce an MLflow run and verify hash consistency.

Usage:
    python scripts/reproduce_from_run.py --run-id abc123 --verify-hash
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime

import mlflow


def get_run_info(run_id: str) -> dict:
    """Get all metadata from an MLflow run."""
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    return {
        "run_id": run_id,
        "params": run.data.params,
        "tags": run.data.tags,
        "metrics": run.data.metrics,
        "artifact_uri": run.info.artifact_uri,
    }


def checkout_dataset(run_info: dict) -> bool:
    """Checkout the exact dataset version used in training."""
    dvc_version = run_info["tags"].get("dvc_version")
    git_commit = run_info["tags"].get("git_commit_sha")

    if git_commit:
        print(f"Checking out git commit: {git_commit}")
        subprocess.run(["git", "checkout", git_commit, "--", "dvc.lock"], check=True)

    print("Pulling DVC data...")
    subprocess.run(["dvc", "pull"], check=True)

    return True


def verify_dataset_hash(run_info: dict, dataset_path: str) -> bool:
    """Verify dataset hash matches the logged value."""
    expected_hash = run_info["tags"].get("dataset_hash_full")

    if not expected_hash:
        print("WARNING: No dataset hash logged in run")
        return True

    actual_hash = compute_file_hash(dataset_path)

    if actual_hash != expected_hash:
        print(f"Dataset hash mismatch!")
        print(f"  Expected: {expected_hash}")
        print(f"  Actual: {actual_hash}")
        return False

    print(f"Dataset hash verified: {actual_hash[:16]}...")
    return True


def compute_file_hash(path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def train_reproduction(run_info: dict, output_dir: str) -> str:
    """Train model with same parameters and return model path."""
    from src.training.train_ssot import train_model

    params = run_info["params"]

    model_path = train_model(
        total_timesteps=int(params.get("total_timesteps", 500000)),
        learning_rate=float(params.get("learning_rate", 3e-4)),
        batch_size=int(params.get("batch_size", 64)),
        seed=int(params.get("seed", 42)),
        output_dir=output_dir,
    )

    return model_path


def verify_model_hash(original_run_info: dict, reproduced_model_path: str) -> dict:
    """Verify reproduced model matches original."""
    original_hash = original_run_info["tags"].get("model_hash_full")
    reproduced_hash = compute_file_hash(reproduced_model_path)

    return {
        "original_hash": original_hash,
        "reproduced_hash": reproduced_hash,
        "hash_match": original_hash == reproduced_hash,
    }


def main():
    parser = argparse.ArgumentParser(description="Reproduce MLflow run")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--verify-hash", action="store_true", help="Verify hashes")
    parser.add_argument("--output-dir", default="./reproduction", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get run info
    print(f"Fetching run info for {args.run_id}...")
    run_info = get_run_info(args.run_id)

    # Checkout dataset
    print("Checking out dataset...")
    checkout_dataset(run_info)

    # Verify dataset hash
    if args.verify_hash:
        dataset_path = "data/pipeline/07_output/datasets_5min/RL_DS3_MACRO_CORE.csv"
        if not verify_dataset_hash(run_info, dataset_path):
            sys.exit(1)

    # Train reproduction
    print("Training reproduction model...")
    model_path = train_reproduction(run_info, str(output_dir))

    # Verify model hash
    if args.verify_hash:
        verification = verify_model_hash(run_info, model_path)

        report = {
            "run_id": args.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            **verification,
            "original_params": run_info["params"],
        }

        report_path = output_dir / "verification_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        if verification["hash_match"]:
            print("SUCCESS: Model hash matches original!")
        else:
            print("FAIL: Model hash does not match!")
            sys.exit(1)


if __name__ == "__main__":
    main()
```

### Checklist Sprint 2

**FEAST-19 - Drift Detection:**
- [ ] Integrar drift detector en inference API
- [ ] Crear endpoint `/drift/status`
- [ ] Crear DAG `l6_drift_monitoring`
- [ ] Configurar alertas Prometheus
- [ ] Agregar dashboard Grafana
- [ ] Documentar umbrales y acciones

**REPRO-97 - CI Reproduction:**
- [ ] Crear workflow `reproducibility-test.yml`
- [ ] Crear script `reproduce_from_run.py`
- [ ] Configurar ejecución semanal
- [ ] Agregar notificaciones de fallo
- [ ] Documentar proceso

---

# SPRINT 3: Experiment Workflows + A/B Dashboard (EXP-15, EXP-16, COMP-88)

## Estimado: 24-32 horas

### EXP-15 & EXP-16: Experiment Approval Workflow

**Paso 1**: Crear modelo de experimento con pre-registro

```python
# src/experiments/experiment_registry.py - NUEVO
"""
Experiment Registry with Pre-Registration Enforcement
=====================================================

All experiments must be registered BEFORE training begins.
This prevents p-hacking and ensures proper hypothesis documentation.

Usage:
    registry = ExperimentRegistry()

    # Pre-register experiment (REQUIRED before training)
    exp_id = registry.register_experiment(
        name="test_new_learning_rate",
        hypothesis="Higher learning rate will converge faster",
        success_criteria={"min_reward": 100, "max_training_time": 3600},
        planned_duration_days=7,
        author="researcher@example.com"
    )

    # After approval, start experiment
    registry.start_experiment(exp_id, approver="lead@example.com")

    # Log results
    registry.complete_experiment(exp_id, results={...})
"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import hashlib
import uuid


class ExperimentStatus(str, Enum):
    DRAFT = "draft"           # Being written
    PENDING_APPROVAL = "pending_approval"  # Submitted for review
    APPROVED = "approved"     # Approved, not started
    RUNNING = "running"       # Currently executing
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Failed during execution
    REJECTED = "rejected"     # Not approved
    CANCELLED = "cancelled"   # Cancelled before completion


@dataclass
class ExperimentRegistration:
    """Pre-registration record for an experiment."""
    experiment_id: str
    name: str
    hypothesis: str
    success_criteria: Dict[str, Any]
    planned_duration_days: int
    author: str

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    submitted_at: Optional[str] = None
    approved_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    # Approval
    status: ExperimentStatus = ExperimentStatus.DRAFT
    approver: Optional[str] = None
    approval_notes: Optional[str] = None

    # Execution
    mlflow_experiment_id: Optional[str] = None
    mlflow_run_ids: List[str] = field(default_factory=list)

    # Results
    results: Optional[Dict[str, Any]] = None
    success: Optional[bool] = None
    conclusion: Optional[str] = None

    # Integrity
    registration_hash: Optional[str] = None

    def compute_registration_hash(self) -> str:
        """Compute hash of pre-registration to prevent tampering."""
        data = {
            "name": self.name,
            "hypothesis": self.hypothesis,
            "success_criteria": self.success_criteria,
            "planned_duration_days": self.planned_duration_days,
            "author": self.author,
            "created_at": self.created_at,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentRegistration':
        data["status"] = ExperimentStatus(data["status"])
        return cls(**data)


class ExperimentRegistry:
    """
    Registry for managing experiment pre-registration and approval.
    """

    def __init__(self, storage_path: str = "experiments/registry.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._experiments: Dict[str, ExperimentRegistration] = {}
        self._load()

    def _load(self):
        """Load experiments from storage."""
        if self.storage_path.exists():
            with open(self.storage_path) as f:
                data = json.load(f)
                for exp_id, exp_data in data.items():
                    self._experiments[exp_id] = ExperimentRegistration.from_dict(exp_data)

    def _save(self):
        """Save experiments to storage."""
        data = {exp_id: exp.to_dict() for exp_id, exp in self._experiments.items()}
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register_experiment(
        self,
        name: str,
        hypothesis: str,
        success_criteria: Dict[str, Any],
        planned_duration_days: int,
        author: str,
    ) -> str:
        """
        Pre-register an experiment.

        Args:
            name: Experiment name
            hypothesis: What you're testing
            success_criteria: Dict of metric thresholds for success
            planned_duration_days: Expected duration
            author: Email of person registering

        Returns:
            experiment_id: Unique ID for this experiment
        """
        exp_id = f"exp_{datetime.now(timezone.utc).strftime('%Y%m%d')}_{uuid.uuid4().hex[:8]}"

        exp = ExperimentRegistration(
            experiment_id=exp_id,
            name=name,
            hypothesis=hypothesis,
            success_criteria=success_criteria,
            planned_duration_days=planned_duration_days,
            author=author,
        )
        exp.registration_hash = exp.compute_registration_hash()

        self._experiments[exp_id] = exp
        self._save()

        return exp_id

    def submit_for_approval(self, experiment_id: str) -> bool:
        """Submit experiment for approval."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Can only submit DRAFT experiments, current: {exp.status}")

        exp.status = ExperimentStatus.PENDING_APPROVAL
        exp.submitted_at = datetime.now(timezone.utc).isoformat()
        self._save()

        # TODO: Send notification to approvers
        return True

    def approve_experiment(
        self,
        experiment_id: str,
        approver: str,
        notes: Optional[str] = None
    ) -> bool:
        """Approve an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.PENDING_APPROVAL:
            raise ValueError(f"Can only approve PENDING experiments")

        # Verify hash hasn't changed
        current_hash = exp.compute_registration_hash()
        if current_hash != exp.registration_hash:
            raise ValueError("Experiment was modified after submission!")

        exp.status = ExperimentStatus.APPROVED
        exp.approver = approver
        exp.approval_notes = notes
        exp.approved_at = datetime.now(timezone.utc).isoformat()
        self._save()

        return True

    def reject_experiment(
        self,
        experiment_id: str,
        approver: str,
        reason: str
    ) -> bool:
        """Reject an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        exp.status = ExperimentStatus.REJECTED
        exp.approver = approver
        exp.approval_notes = reason
        self._save()

        return True

    def start_experiment(self, experiment_id: str) -> bool:
        """
        Mark experiment as started.

        IMPORTANT: This can only be called after approval!
        """
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.APPROVED:
            raise ValueError(
                f"Cannot start experiment with status {exp.status}. "
                "Must be APPROVED first!"
            )

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now(timezone.utc).isoformat()
        self._save()

        return True

    def add_mlflow_run(self, experiment_id: str, run_id: str):
        """Link an MLflow run to this experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError("Can only add runs to RUNNING experiments")

        exp.mlflow_run_ids.append(run_id)
        self._save()

    def complete_experiment(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        conclusion: str
    ) -> bool:
        """Complete experiment with results."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        if exp.status != ExperimentStatus.RUNNING:
            raise ValueError("Can only complete RUNNING experiments")

        # Evaluate success based on criteria
        success = self._evaluate_success(exp.success_criteria, results)

        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now(timezone.utc).isoformat()
        exp.results = results
        exp.success = success
        exp.conclusion = conclusion
        self._save()

        return success

    def _evaluate_success(
        self,
        criteria: Dict[str, Any],
        results: Dict[str, Any]
    ) -> bool:
        """Evaluate if results meet success criteria."""
        for metric, threshold in criteria.items():
            if metric.startswith("min_"):
                actual_metric = metric[4:]
                if results.get(actual_metric, 0) < threshold:
                    return False
            elif metric.startswith("max_"):
                actual_metric = metric[4:]
                if results.get(actual_metric, float("inf")) > threshold:
                    return False
        return True

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRegistration]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def list_experiments(
        self,
        status: Optional[ExperimentStatus] = None
    ) -> List[ExperimentRegistration]:
        """List experiments, optionally filtered by status."""
        experiments = list(self._experiments.values())
        if status:
            experiments = [e for e in experiments if e.status == status]
        return sorted(experiments, key=lambda e: e.created_at, reverse=True)

    def get_pending_approvals(self) -> List[ExperimentRegistration]:
        """Get experiments waiting for approval."""
        return self.list_experiments(ExperimentStatus.PENDING_APPROVAL)
```

### COMP-88: Mejorar A/B Dashboard

**Paso 1**: Crear componentes de dashboard mejorado

```typescript
// usdcop-trading-dashboard/components/experiments/ABTestDashboard.tsx - NUEVO
import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  CardHeader,
  Typography,
  Grid,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from 'recharts';

interface ExperimentResult {
  name: string;
  status: 'running' | 'completed' | 'failed';
  control: {
    reward: number;
    sharpe: number;
    winRate: number;
    sampleSize: number;
  };
  treatment: {
    reward: number;
    sharpe: number;
    winRate: number;
    sampleSize: number;
  };
  statistics: {
    pValue: number;
    effectSize: number;
    confidenceInterval: [number, number];
    significant: boolean;
  };
}

export const ABTestDashboard: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentResult[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await fetch('/api/experiments/ab-tests');
      const data = await response.json();
      setExperiments(data);
    } catch (error) {
      console.error('Failed to fetch experiments:', error);
    } finally {
      setLoading(false);
    }
  };

  const getStatusChip = (status: string) => {
    const colors: Record<string, 'success' | 'warning' | 'error'> = {
      completed: 'success',
      running: 'warning',
      failed: 'error',
    };
    return <Chip label={status} color={colors[status]} size="small" />;
  };

  const getSignificanceIndicator = (significant: boolean, pValue: number) => {
    if (significant) {
      return (
        <Chip
          label={`p=${pValue.toFixed(4)} ✓`}
          color="success"
          size="small"
        />
      );
    }
    return (
      <Chip
        label={`p=${pValue.toFixed(4)}`}
        color="default"
        size="small"
      />
    );
  };

  if (loading) {
    return <LinearProgress />;
  }

  return (
    <Grid container spacing={3}>
      {/* Summary Cards */}
      <Grid item xs={12}>
        <Card>
          <CardHeader title="A/B Test Summary" />
          <CardContent>
            <Grid container spacing={2}>
              <Grid item xs={3}>
                <Typography variant="h4">
                  {experiments.length}
                </Typography>
                <Typography color="textSecondary">
                  Total Experiments
                </Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="h4">
                  {experiments.filter(e => e.status === 'running').length}
                </Typography>
                <Typography color="textSecondary">
                  Running
                </Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="h4">
                  {experiments.filter(e => e.statistics.significant).length}
                </Typography>
                <Typography color="textSecondary">
                  Significant Results
                </Typography>
              </Grid>
              <Grid item xs={3}>
                <Typography variant="h4">
                  {(experiments.filter(e => e.statistics.significant).length /
                    experiments.filter(e => e.status === 'completed').length * 100 || 0).toFixed(0)}%
                </Typography>
                <Typography color="textSecondary">
                  Win Rate
                </Typography>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Grid>

      {/* Experiment Table */}
      <Grid item xs={12}>
        <Card>
          <CardHeader title="Experiments" />
          <CardContent>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell align="right">Control Reward</TableCell>
                  <TableCell align="right">Treatment Reward</TableCell>
                  <TableCell align="right">Effect Size</TableCell>
                  <TableCell align="right">Significance</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {experiments.map((exp) => (
                  <TableRow key={exp.name}>
                    <TableCell>{exp.name}</TableCell>
                    <TableCell>{getStatusChip(exp.status)}</TableCell>
                    <TableCell align="right">
                      {exp.control.reward.toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      {exp.treatment.reward.toFixed(2)}
                    </TableCell>
                    <TableCell align="right">
                      {exp.statistics.effectSize.toFixed(3)}
                    </TableCell>
                    <TableCell align="right">
                      {getSignificanceIndicator(
                        exp.statistics.significant,
                        exp.statistics.pValue
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </Grid>

      {/* Effect Size Chart */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="Effect Sizes" />
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={experiments}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar
                  dataKey="statistics.effectSize"
                  fill="#8884d8"
                  name="Cohen's d"
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>

      {/* P-Value Distribution */}
      <Grid item xs={12} md={6}>
        <Card>
          <CardHeader title="P-Value Distribution" />
          <CardContent>
            <Alert severity="info" sx={{ mb: 2 }}>
              Multiple testing correction: Bonferroni (α = 0.05/{experiments.length})
            </Alert>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart
                data={experiments}
                layout="vertical"
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis type="number" domain={[0, 1]} />
                <YAxis type="category" dataKey="name" />
                <Tooltip />
                <Bar
                  dataKey="statistics.pValue"
                  fill="#82ca9d"
                  name="p-value"
                />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
};

export default ABTestDashboard;
```

### Checklist Sprint 3

**EXP-15 - Approval Workflow:**
- [ ] Crear `ExperimentRegistry` class
- [ ] Implementar estados de experimento
- [ ] Crear CLI para registro de experimentos
- [ ] Agregar notificaciones de aprobación
- [ ] Documentar workflow

**EXP-16 - Pre-Registration:**
- [ ] Enforcement de pre-registro antes de training
- [ ] Hash de integridad para prevenir tampering
- [ ] Validación en MLflow
- [ ] Documentar proceso

**COMP-88 - A/B Dashboard:**
- [ ] Crear componente `ABTestDashboard.tsx`
- [ ] Agregar endpoint API `/api/experiments/ab-tests`
- [ ] Implementar visualizaciones de effect size
- [ ] Agregar corrección de multiple testing
- [ ] Documentar interpretación de resultados

---

# SPRINT 4: Penetration Test (SEC-30)

## Estimado: Externo (2-4 semanas)

### Preparación Pre-Pentest (Interno)

**Semana 1: Preparación**

1. **Documentar superfície de ataque**
   - Listar todos los endpoints expuestos
   - Documentar métodos de autenticación
   - Mapear flujos de datos sensibles

2. **Preparar ambiente de pruebas**
   - Clonar producción a staging
   - Configurar logging aumentado
   - Crear cuentas de test

3. **Definir scope**
   ```
   IN SCOPE:
   - Inference API (FastAPI)
   - Trading Dashboard (Next.js)
   - MLflow UI
   - Airflow UI
   - MinIO Console
   - Grafana

   OUT OF SCOPE:
   - Cloud provider infrastructure
   - Third-party APIs (FRED, TwelveData)
   - Physical security
   ```

### Proveedores Recomendados

| Proveedor | Tipo | Costo Estimado | Duración |
|-----------|------|----------------|----------|
| **HackerOne** | Bug Bounty | $5K-$20K | Ongoing |
| **Synack** | Managed Pentest | $15K-$30K | 2-4 weeks |
| **Cobalt** | Pentest-as-Service | $10K-$20K | 2-3 weeks |
| **Local** | Consultor independiente | $5K-$10K | 1-2 weeks |

### Checklist Pentest

- [ ] Definir scope y reglas de engagement
- [ ] Preparar ambiente de staging
- [ ] Seleccionar proveedor
- [ ] Ejecutar pentest
- [ ] Recibir reporte
- [ ] Remediar hallazgos críticos
- [ ] Re-test de remediaciones
- [ ] Documentar resultados

---

# BACKLOG: P2/P3 Items (17 items)

## P2 - Medio (12 items)

| ID | Item | Esfuerzo Est. |
|----|------|---------------|
| ARCH-30 | Consolidar FEATURE_ORDER (incluido en Sprint 1) | 0h |
| SCRP-26 | Ampliar fallback entre fuentes de datos | 8h |
| FEAST-32 | Implementar streaming features | 16h |
| INF-38 | Implementar hot reload de modelos | 8h |
| SCRP-64 | Documentar ownership por fuente | 2h |
| TRAIN-47 | Mejorar determinismo del ambiente | 4h |
| INF-48 | Optimizar batch inference | 8h |
| DST-EXP-30 | Implementar más técnicas de augmentation | 8h |
| HYP-44 | Integración completa con Ray Tune | 8h |
| REPRO-100 | Implementar reproducibility badge | 4h |

## P3 - Bajo (5 items)

| ID | Item | Esfuerzo Est. |
|----|------|---------------|
| SCRP-34 | Extender proxy a más fuentes | 4h |
| SCRP-64 | Documentar ownership por fuente de datos | 2h |
| INF-48 | Optimizar batch inference | 8h |

---

# MÉTRICAS DE ÉXITO

## KPIs por Sprint

| Sprint | KPI | Target | Medición |
|--------|-----|--------|----------|
| Sprint 1 | Imports deprecados | 0 | CI check |
| Sprint 2 | Drift detection coverage | 100% features | Prometheus |
| Sprint 2 | Reproducibility rate | 100% | CI weekly |
| Sprint 3 | Experiments pre-registered | 100% | Registry |
| Sprint 4 | Critical vulnerabilities | 0 | Pentest report |

## Definition of Done por Item

**CONT-45/46:**
- [ ] Zero imports from deprecated paths
- [ ] CI check passing
- [ ] Documentation updated

**FEAST-19:**
- [ ] Drift detector integrated in inference
- [ ] Alerts configured and tested
- [ ] Dashboard showing drift metrics

**REPRO-97:**
- [ ] Weekly CI job running
- [ ] Last 4 weeks green
- [ ] Notification on failure working

**EXP-15/16:**
- [ ] Registry enforces pre-registration
- [ ] Cannot start experiment without approval
- [ ] Audit trail complete

**COMP-88:**
- [ ] Dashboard shows all running experiments
- [ ] Effect sizes visualized
- [ ] P-values with correction displayed

**SEC-30:**
- [ ] Pentest completed
- [ ] All critical findings remediated
- [ ] Re-test passed

---

*Plan de Remediación P1 - Versión 1.0*
*Generado: 2026-01-17*
*Próxima revisión: Al completar Sprint 1*

