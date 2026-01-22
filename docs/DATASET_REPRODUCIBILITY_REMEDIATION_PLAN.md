# PLAN DE REMEDIACION - DATASET, DVC, FEAST, MLFLOW, REPRODUCIBILITY
## Auditoria de 200 Preguntas + 18 Preguntas Macro Scraping

**Fecha**: 2026-01-17
**Score Actual**: 47% (103/218 preguntas)
**Score Objetivo**: 100% (218/218)
**Duracion Estimada**: 9 Semanas (135 horas total)

---

# RESUMEN EJECUTIVO

## Estado Actual por Categoria

```
+------------------------+--------+---------+-----------+-------+
| CATEGORIA              | CUMPLE | PARCIAL | NO CUMPLE | SCORE |
+------------------------+--------+---------+-----------+-------+
| DST (Dataset)          |   33   |    6    |    11     |  66%  |
| DVC (Versioning)       |   30   |   14    |     6     |  60%  |
| FEAST (Feature Store)  |   13   |    8    |    29     |  26%  |
| MLF (MLflow)           |    4   |    6    |    20     |  13%  |
| REPR (Reproducibility) |    4   |    1    |    15     |  20%  |
+------------------------+--------+---------+-----------+-------+
| SUBTOTAL Original      |   84   |   35    |    81     |  42%  |
+------------------------+--------+---------+-----------+-------+
| MACRO SCRAPING (NEW)   |        |         |           |       |
+------------------------+--------+---------+-----------+-------+
| MET (Metrics/Logs)     |    0   |    1    |     3     |  6%   |
| CAL (Publication Cal)  |    6   |    0    |     0     | 100%  |
| PIT (Point-in-Time)    |    3   |    1    |     0     |  88%  |
| FF (Forward-Fill)      |    3   |    0    |     1     |  75%  |
+------------------------+--------+---------+-----------+-------+
| SUBTOTAL Macro         |   12   |    2    |     4     |  67%  |
+------------------------+--------+---------+-----------+-------+
| TOTAL CONSOLIDADO      |   96   |   37    |    85     |  47%  |
+------------------------+--------+---------+-----------+-------+
```

---

# AUDITORIA MACRO SCRAPING (18 PREGUNTAS)

## Resultados Detallados

### METRICAS Y LOGS DE CAPTURA (MET, MON, SCRP)

| ID | Pregunta | Estado | Evidencia |
|----|----------|--------|-----------|
| MET-01 | Existe metrica `macro_ingestion_success_total`? | ❌ NO CUMPLE | No Counter definido en prometheus_metrics.py |
| MET-07 | Existe `data_staleness_seconds` por variable? | ⚠️ PARCIAL | Solo `usdcop_data_freshness_seconds` agregado |
| MON-06 | Prometheus tiene metricas por source? | ❌ NO CUMPLE | No labels por FRED/TwelveData/BanRep |
| SCRP-31 | Dashboard muestra success rate por fuente? | ❌ NO CUMPLE | No existe dashboard de macro ingestion |

### FECHAS DE PUBLICACION Y RETRASOS (CAL, FRED, BR)

| ID | Pregunta | Estado | Evidencia |
|----|----------|--------|-----------|
| CAL-02 | Existe calendario economico? | ✅ CUMPLE | scraper_cpi_investing_calendar.py, scraper_ipc_col_calendar.py |
| FRED-28 | CPI delay (~12 dias) documentado? | ✅ CUMPLE | `typical_publication_delay_days=12` en l0_data_contracts.py:391 |
| FRED-29 | Fed Funds delay (~15 dias) documentado? | ✅ CUMPLE | `typical_publication_delay_days=15` en l0_data_contracts.py:383 |
| FRED-31 | Diferencia period_date vs release_date? | ✅ CUMPLE | Ambos campos en scrapers de calendario |
| BR-21 | IBR tiene delay T+1 configurado? | ✅ CUMPLE | `typical_publication_delay_days=1` en l0_data_contracts.py:490 |
| BR-26 | IPC Colombia delay ~5 dias? | ✅ CUMPLE | `typical_publication_delay_days=5` en l0_data_contracts.py:530 |

### POINT-IN-TIME CORRECTNESS (CAL, FRED, IMP)

| ID | Pregunta | Estado | Evidencia |
|----|----------|--------|-----------|
| CAL-22 | Sistema implementa PIT para macro? | ✅ CUMPLE | merge_asof direction='backward' en BacktestFeatureBuilder |
| CAL-32 | Backtest NO usa datos pre-publicacion? | ✅ CUMPLE | Bounded FFILL + slice isolation |
| FRED-32 | Indicadores mensuales usan release_date? | ⚠️ PARCIAL | Implicit via delays, no explicit release_date column |
| IMP-27 | Macro disponible ANTES de barra OHLCV? | ✅ CUMPLE | Backward merge enforced |

### FORWARD-FILL BOUNDED (FF)

| ID | Pregunta | Estado | Evidencia |
|----|----------|--------|-----------|
| FF-03 | daily_max_days = 5? | ✅ CUMPLE | FFillConfig.daily_max_days=5 en l0_data_contracts.py:600 |
| FF-05 | monthly_max_days = 35? | ✅ CUMPLE | FFillConfig.monthly_max_days=35 en l0_data_contracts.py:602 |
| FF-08 | Solo ffill, NUNCA bfill? | ❌ NO CUMPLE | **CRITICO**: bfill() encontrado en 3 archivos |
| FF-10 | Respeta schedule por indicador? | ✅ CUMPLE | get_max_days_for_schedule() implementado |

---

## 🚨 HALLAZGO CRITICO: BFILL ENCONTRADO

**Severidad**: CRITICA - Viola causalidad temporal

### Archivos con bfill():

| Archivo | Linea | Codigo | Riesgo |
|---------|-------|--------|--------|
| `data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py` | 616 | `df_ds = df_ds.ffill().bfill()` | **CRITICO** |
| `archive/airflow-dags/deprecated/usdcop_m5__04b_l3_llm_features.py` | 467 | `df.groupby(...).ffill().bfill()` | ALTO |
| `tests/chaos/test_nan_handling.py` | 57 | `df_clean = df.ffill().bfill()` | MEDIO |

### Impacto:
- **bfill()** propaga datos hacia atras en el tiempo
- Crea look-ahead bias en datasets de entrenamiento
- Viola integridad temporal requerida para modelos RL

### Remediacion Inmediata:
```python
# ANTES (INCORRECTO):
df_ds = df_ds.ffill().bfill()

# DESPUES (CORRECTO):
df_ds = df_ds.ffill()  # Solo forward-fill
```

## Progresion Planificada

| Fase | Semanas | Items | Score Acumulado | Esfuerzo |
|------|---------|-------|-----------------|----------|
| P0 - Bloqueantes | 1-2 | 26 | 55% | 25h |
| P1 - Criticos | 3-4 | 34 | 72% | 30h |
| P2 - Importantes | 5-6 | 26 | 85% | 25h |
| P3 - Mejoras | 7-8 | 30 | 100% | 40h |

---

# FASE P0: BLOQUEANTES (Semanas 1-2)

## Objetivo: Alcanzar 55% Compliance

Estas remediaciones son **bloqueadores absolutos** para produccion. Sin ellas, no se puede garantizar reproducibilidad de entrenamientos ni trazabilidad de datos.

---

## P0-01: Log Dataset Lineage en MLflow

**Estado**: NO CUMPLE
**Impacto**: MLF-01 a MLF-05 (5 preguntas)
**Tiempo**: 3h

**Problema**: `scripts/train_with_mlflow.py` NO logea dataset_hash, feature_list, train_start_date, train_end_date.

**Solucion**:

```python
# Archivo: scripts/train_with_mlflow.py
# Agregar en la clase MLflowTrainer:

import hashlib
from datetime import datetime
from src.core.contracts.feature_contract import FEATURE_ORDER, FEATURE_ORDER_HASH

def log_dataset_lineage(self, df: pd.DataFrame, data_path: str) -> str:
    """
    Log complete dataset lineage to MLflow.

    Logs:
    - dataset_hash: SHA256 of the training dataframe
    - dataset_path: Path to raw data
    - train_start_date: First timestamp in dataset
    - train_end_date: Last timestamp in dataset
    - dataset_rows: Number of rows
    - feature_list: Ordered list of features used
    - feature_order_hash: Hash of FEATURE_ORDER for integrity
    """
    # Compute dataset hash
    dataset_bytes = df.to_csv(index=False).encode('utf-8')
    dataset_hash = hashlib.sha256(dataset_bytes).hexdigest()[:16]

    # Extract date range
    if 'timestamp' in df.columns:
        train_start = df['timestamp'].min().isoformat()
        train_end = df['timestamp'].max().isoformat()
    else:
        train_start = "unknown"
        train_end = "unknown"

    # Log to MLflow
    mlflow.log_param("dataset_hash", dataset_hash)
    mlflow.log_param("dataset_path", data_path)
    mlflow.log_param("train_start_date", train_start)
    mlflow.log_param("train_end_date", train_end)
    mlflow.log_param("dataset_rows", len(df))
    mlflow.log_param("feature_order_hash", FEATURE_ORDER_HASH)

    # Log feature list as artifact
    feature_list = {
        "feature_order": list(FEATURE_ORDER),
        "observation_dim": len(FEATURE_ORDER),
        "contract_version": "2.0.0"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(feature_list, f, indent=2)
        mlflow.log_artifact(f.name, "dataset_metadata")

    return dataset_hash
```

**Test de Validacion**:

```python
# tests/unit/test_mlflow_dataset_lineage.py
import pytest
import mlflow
from unittest.mock import patch, MagicMock
import pandas as pd

def test_log_dataset_lineage_logs_all_params():
    """Verify all required dataset params are logged."""
    trainer = create_test_trainer()
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'log_ret_5m': np.random.randn(100),
        # ... otras columnas
    })

    with patch('mlflow.log_param') as mock_log:
        trainer.log_dataset_lineage(df, "data/test.parquet")

        # Verify all required params logged
        logged_params = {call[0][0] for call in mock_log.call_args_list}
        required = {
            'dataset_hash', 'dataset_path', 'train_start_date',
            'train_end_date', 'dataset_rows', 'feature_order_hash'
        }
        assert required <= logged_params

def test_dataset_hash_deterministic():
    """Same data should produce same hash."""
    trainer = create_test_trainer()
    df = pd.DataFrame({'a': [1, 2, 3]})

    hash1 = trainer.log_dataset_lineage(df, "path1")
    hash2 = trainer.log_dataset_lineage(df, "path2")

    assert hash1 == hash2
```

---

## P0-02: Feature Service Name en ModelContract

**Estado**: NO CUMPLE
**Impacto**: FEAST-15, FEAST-16, MLF-20 (3 preguntas)
**Tiempo**: 2h

**Problema**: `ModelContract` no tiene campo `feature_service_name`.

**Solucion**:

```python
# Archivo: services/inference_api/contracts/model_contract.py
# Modificar ModelContract:

@dataclass(frozen=True)
class ModelContract:
    """
    Immutable contract for a model configuration.
    """
    model_id: str
    version: str
    builder_type: BuilderType
    observation_dim: int
    norm_stats_path: str
    model_path: str
    description: str = ""
    norm_stats_hash: Optional[str] = None
    model_hash: Optional[str] = None
    # NUEVO: Link a Feature Service de Feast
    feature_service_name: Optional[str] = None
    # NUEVO: Hash del dataset de entrenamiento
    training_dataset_hash: Optional[str] = None
    # NUEVO: Rango de fechas de entrenamiento
    training_start_date: Optional[str] = None
    training_end_date: Optional[str] = None


# Actualizar registry:
@classmethod
def _initialize_defaults(cls) -> None:
    cls.register(ModelContract(
        model_id="ppo_primary",
        version="1.0.0",
        builder_type=BuilderType.CURRENT_15DIM,
        observation_dim=15,
        norm_stats_path="config/norm_stats.json",
        model_path="models/ppo_production/final_model.zip",
        description="PPO Primary Production - 15 features with macro",
        feature_service_name="ppo_production_service",  # NUEVO
    ))
```

---

## P0-03: Script reproduce_dataset_from_run.py

**Estado**: NO EXISTE
**Impacto**: REPR-01 a REPR-05 (5 preguntas)
**Tiempo**: 4h

**Solucion Completa**:

```python
#!/usr/bin/env python3
"""
scripts/reproduce_dataset_from_run.py
=====================================
Reproduce el dataset exacto usado en un run de MLflow.

Uso:
    python scripts/reproduce_dataset_from_run.py --run-id abc123 --output data/reproduced/

Este script:
1. Lee los params del run (dataset_hash, feature_list, dates)
2. Reconstruye el dataset usando DVC y Feast
3. Valida que el hash coincida
4. Guarda el dataset reproducido
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetReproducer:
    """Reproduce datasets from MLflow runs."""

    def __init__(self, mlflow_tracking_uri: str = "http://localhost:5000"):
        self.client = MlflowClient(mlflow_tracking_uri)

    def get_run_metadata(self, run_id: str) -> Dict[str, Any]:
        """Extract dataset metadata from MLflow run."""
        run = self.client.get_run(run_id)
        params = run.data.params

        required_params = [
            'dataset_hash', 'dataset_path', 'train_start_date',
            'train_end_date', 'feature_order_hash'
        ]

        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(
                f"Run {run_id} missing required params: {missing}. "
                f"This run may predate the lineage tracking system."
            )

        return {
            'dataset_hash': params['dataset_hash'],
            'dataset_path': params['dataset_path'],
            'train_start_date': params['train_start_date'],
            'train_end_date': params['train_end_date'],
            'feature_order_hash': params['feature_order_hash'],
            'dataset_rows': int(params.get('dataset_rows', 0)),
            'dvc_version': params.get('dvc_version', 'unknown'),
        }

    def checkout_dvc_data(self, dataset_path: str, dvc_version: str) -> Path:
        """Checkout specific DVC version of data."""
        if dvc_version != 'unknown':
            logger.info(f"Checking out DVC version: {dvc_version}")
            subprocess.run(
                ['dvc', 'checkout', dataset_path, '--rev', dvc_version],
                check=True
            )
        else:
            logger.warning("DVC version unknown, using current data")
            subprocess.run(['dvc', 'pull', dataset_path], check=True)

        return Path(dataset_path)

    def load_and_filter_data(
        self,
        data_path: Path,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load data and filter to training date range."""
        logger.info(f"Loading data from {data_path}")

        if data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path, parse_dates=['timestamp'])
        else:
            raise ValueError(f"Unsupported format: {data_path.suffix}")

        # Filter to date range
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        mask = (
            (df['timestamp'] >= start_date) &
            (df['timestamp'] <= end_date)
        )
        filtered = df[mask].copy()

        logger.info(f"Filtered to {len(filtered)} rows ({start_date} to {end_date})")
        return filtered

    def verify_hash(self, df: pd.DataFrame, expected_hash: str) -> bool:
        """Verify dataset hash matches expected."""
        dataset_bytes = df.to_csv(index=False).encode('utf-8')
        actual_hash = hashlib.sha256(dataset_bytes).hexdigest()[:16]

        if actual_hash != expected_hash:
            logger.error(
                f"Hash mismatch! Expected: {expected_hash}, Got: {actual_hash}. "
                f"Dataset may have been modified or DVC version incorrect."
            )
            return False

        logger.info(f"Hash verified: {actual_hash}")
        return True

    def verify_feature_order(self, feature_order_hash: str) -> bool:
        """Verify current FEATURE_ORDER matches training."""
        from src.core.contracts.feature_contract import FEATURE_ORDER_HASH

        if FEATURE_ORDER_HASH != feature_order_hash:
            logger.error(
                f"FEATURE_ORDER mismatch! Training: {feature_order_hash}, "
                f"Current: {FEATURE_ORDER_HASH}. Contract may have changed."
            )
            return False

        logger.info("FEATURE_ORDER verified")
        return True

    def reproduce(
        self,
        run_id: str,
        output_dir: Path,
        verify_hash: bool = True
    ) -> Optional[Path]:
        """
        Full reproduction workflow.

        Returns:
            Path to reproduced dataset, or None if verification failed
        """
        logger.info(f"Reproducing dataset from run: {run_id}")

        # 1. Get metadata
        metadata = self.get_run_metadata(run_id)
        logger.info(f"Metadata: {json.dumps(metadata, indent=2)}")

        # 2. Verify feature order
        if not self.verify_feature_order(metadata['feature_order_hash']):
            logger.error("FEATURE_ORDER has changed since training!")
            return None

        # 3. Checkout DVC data
        data_path = self.checkout_dvc_data(
            metadata['dataset_path'],
            metadata['dvc_version']
        )

        # 4. Load and filter
        df = self.load_and_filter_data(
            data_path,
            metadata['train_start_date'],
            metadata['train_end_date']
        )

        # 5. Verify hash
        if verify_hash:
            if not self.verify_hash(df, metadata['dataset_hash']):
                logger.error("Dataset hash verification failed!")
                return None

        # 6. Save reproduced dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"reproduced_{run_id}.parquet"
        df.to_parquet(output_path, index=False)

        # 7. Save reproduction metadata
        meta_path = output_dir / f"reproduced_{run_id}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump({
                'original_run_id': run_id,
                'reproduction_date': pd.Timestamp.now().isoformat(),
                'rows': len(df),
                'hash_verified': verify_hash,
                **metadata
            }, f, indent=2)

        logger.info(f"Dataset reproduced: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce dataset from MLflow run"
    )
    parser.add_argument('--run-id', required=True, help='MLflow run ID')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--tracking-uri', default='http://localhost:5000')
    parser.add_argument('--skip-hash-verify', action='store_true')

    args = parser.parse_args()

    reproducer = DatasetReproducer(args.tracking_uri)
    result = reproducer.reproduce(
        run_id=args.run_id,
        output_dir=Path(args.output),
        verify_hash=not args.skip_hash_verify
    )

    if result:
        print(f"\nSUCCESS: Dataset reproduced at {result}")
        return 0
    else:
        print("\nFAILED: Could not reproduce dataset")
        return 1


if __name__ == '__main__':
    sys.exit(main())
```

---

## P0-04: Documentacion docs/DATASET_CONSTRUCTION.md

**Estado**: NO EXISTE
**Impacto**: DST-01, DST-02, DST-03, DST-04 (4 preguntas)
**Tiempo**: 3h

**Contenido del Archivo**:

```markdown
# Dataset Construction Guide
## USD/COP RL Trading System

## Overview

This document describes how training datasets are constructed from raw data sources.

## Data Flow

```
Raw Data Sources
     |
     v
+----------------+     +----------------+     +----------------+
|   L0 Macro     | --> |   L1 Features  | --> |   L2 Dataset   |
| (DAG ingest)   |     | (DAG compute)  |     | (DVC versioned)|
+----------------+     +----------------+     +----------------+
     |                       |                       |
     v                       v                       v
  PostgreSQL           Feature Store            Parquet Files
  (macro_indicators)   (Feast/Redis)           (data/processed/)
```

## Feature Order (SSOT)

All datasets MUST follow the canonical feature order defined in:
`src/core/contracts/feature_contract.py`

```python
FEATURE_ORDER = (
    "log_ret_5m",       # 0: 5-minute log return
    "log_ret_1h",       # 1: 1-hour log return
    "log_ret_4h",       # 2: 4-hour log return
    "rsi_9",            # 3: RSI(9)
    "atr_pct",          # 4: ATR as % of price
    "adx_14",           # 5: ADX(14)
    "dxy_z",            # 6: DXY z-score
    "dxy_change_1d",    # 7: DXY 1-day change
    "vix_z",            # 8: VIX z-score
    "embi_z",           # 9: EMBI Colombia z-score
    "brent_change_1d",  # 10: Brent oil 1-day change
    "rate_spread",      # 11: Interest rate spread
    "usdmxn_change_1d", # 12: USD/MXN 1-day change
    "position",         # 13: Current position
    "time_normalized",  # 14: Normalized time [0,1]
)
```

## Temporal Joins

Macro data is joined to price data using `pd.merge_asof`:

```python
merged = pd.merge_asof(
    price_df.sort_values('timestamp'),
    macro_df.sort_values('date'),
    left_on='timestamp',
    right_on='date',
    direction='backward',  # Use most recent available macro
    tolerance=pd.Timedelta('3 days')
)
```

## Trading Hours Filter

Training data is filtered to Colombian trading hours:

- **Trading Days**: Monday-Friday (excluding CO/US holidays)
- **Trading Hours**: 08:00-17:00 America/Bogota
- **Holiday Calendar**: `src/utils/trading_calendar.py`

## Dataset Versioning

Datasets are versioned with DVC:

```bash
# Track new dataset
dvc add data/processed/training_dataset.parquet
git add data/processed/training_dataset.parquet.dvc
git commit -m "Dataset v1.2.0"

# Checkout specific version
dvc checkout data/processed/training_dataset.parquet --rev v1.2.0
```

## Normalization Stats

Normalization statistics are computed ONLY from training data:

```json
{
  "log_ret_5m": {"mean": 0.0001, "std": 0.002},
  "log_ret_1h": {"mean": 0.0005, "std": 0.008},
  ...
}
```

File: `config/norm_stats.json`

## Reproducibility Checklist

Before training:
1. [ ] FEATURE_ORDER hash logged to MLflow
2. [ ] Dataset hash computed and logged
3. [ ] Date range logged (train_start, train_end)
4. [ ] DVC commit SHA logged
5. [ ] Normalization stats path logged
```

---

## P0-05: Log FEATURE_ORDER como Artifact

**Estado**: NO CUMPLE
**Impacto**: DST-20, MLF-15, MLF-16 (3 preguntas)
**Tiempo**: 1h

**Solucion**:

```python
# Agregar a scripts/train_with_mlflow.py

def log_feature_contract_artifact(self):
    """Log FEATURE_ORDER as JSON artifact for reproducibility."""
    from src.core.contracts.feature_contract import (
        FEATURE_ORDER, FEATURE_SPECS, FEATURE_CONTRACT_VERSION,
        FEATURE_ORDER_HASH, OBSERVATION_DIM
    )

    contract_artifact = {
        "contract_version": FEATURE_CONTRACT_VERSION,
        "feature_order_hash": FEATURE_ORDER_HASH,
        "observation_dim": OBSERVATION_DIM,
        "features": [
            {
                "name": name,
                "index": idx,
                "type": FEATURE_SPECS[name].type.value,
                "unit": FEATURE_SPECS[name].unit.value,
                "clip_min": FEATURE_SPECS[name].clip_min,
                "clip_max": FEATURE_SPECS[name].clip_max,
                "source": FEATURE_SPECS[name].source,
            }
            for idx, name in enumerate(FEATURE_ORDER)
        ]
    }

    artifact_path = Path(tempfile.mkdtemp()) / "feature_contract.json"
    with open(artifact_path, 'w') as f:
        json.dump(contract_artifact, f, indent=2)

    mlflow.log_artifact(str(artifact_path), "contracts")
    logger.info(f"Logged feature contract artifact (hash: {FEATURE_ORDER_HASH})")
```

---

## P0-06: Implementar merge_asof para Joins Temporales

**Estado**: NO CUMPLE
**Impacto**: DST-30, DST-31, DST-32 (3 preguntas)
**Tiempo**: 3h

**Problema**: No hay `merge_asof` implementado para joins temporales correctos entre precio y macro.

**Solucion**:

```python
# Archivo: src/features/temporal_joins.py

import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def merge_price_with_macro(
    price_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    price_time_col: str = 'timestamp',
    macro_time_col: str = 'date',
    tolerance: str = '3D',
    direction: str = 'backward'
) -> pd.DataFrame:
    """
    Merge price data with macro indicators using temporal join.

    This ensures:
    1. No look-ahead bias (direction='backward')
    2. Tolerance for stale data (3 days max)
    3. NaN for missing macro data

    Args:
        price_df: DataFrame with price/technical features
        macro_df: DataFrame with macro indicators
        price_time_col: Column name for price timestamp
        macro_time_col: Column name for macro date
        tolerance: Maximum time gap for join
        direction: 'backward' to avoid look-ahead bias

    Returns:
        Merged DataFrame with both price and macro features
    """
    # Ensure datetime types
    price_df = price_df.copy()
    macro_df = macro_df.copy()

    price_df[price_time_col] = pd.to_datetime(price_df[price_time_col])
    macro_df[macro_time_col] = pd.to_datetime(macro_df[macro_time_col])

    # Sort for merge_asof requirement
    price_df = price_df.sort_values(price_time_col)
    macro_df = macro_df.sort_values(macro_time_col)

    # Perform temporal join
    merged = pd.merge_asof(
        price_df,
        macro_df,
        left_on=price_time_col,
        right_on=macro_time_col,
        direction=direction,
        tolerance=pd.Timedelta(tolerance)
    )

    # Log join statistics
    total_rows = len(merged)
    macro_cols = [c for c in macro_df.columns if c != macro_time_col]
    null_counts = merged[macro_cols].isnull().sum()

    logger.info(
        f"Temporal join complete: {total_rows} rows, "
        f"macro null rates: {(null_counts / total_rows * 100).to_dict()}"
    )

    return merged


def validate_no_lookahead(
    df: pd.DataFrame,
    price_time_col: str = 'timestamp',
    macro_time_col: str = 'macro_date'
) -> bool:
    """
    Validate that macro data does not have look-ahead bias.

    Returns True if all macro dates are <= price timestamps.
    """
    if macro_time_col not in df.columns:
        logger.warning(f"Column {macro_time_col} not found, cannot validate")
        return True

    violations = df[df[macro_time_col] > df[price_time_col]]

    if len(violations) > 0:
        logger.error(
            f"Look-ahead bias detected! {len(violations)} rows have "
            f"macro_date > timestamp"
        )
        return False

    logger.info("No look-ahead bias detected")
    return True
```

---

## P0-07: Trading Hours Filter

**Estado**: PARCIAL
**Impacto**: DST-35, DST-36 (2 preguntas)
**Tiempo**: 2h

**Solucion Completa**:

```python
# Archivo: src/features/trading_hours_filter.py

import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TradingHoursFilter:
    """
    Filter data to Colombian trading hours.

    Trading Session:
    - Days: Monday-Friday
    - Hours: 08:00-17:00 America/Bogota
    - Excludes: Colombian and US holidays
    """

    def __init__(
        self,
        timezone: str = 'America/Bogota',
        start_hour: int = 8,
        end_hour: int = 17,
        holidays_co: Optional[List[str]] = None,
        holidays_us: Optional[List[str]] = None
    ):
        self.timezone = timezone
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.holidays_co = set(holidays_co or [])
        self.holidays_us = set(holidays_us or [])

    def filter(
        self,
        df: pd.DataFrame,
        time_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Filter DataFrame to trading hours only.

        Args:
            df: Input DataFrame
            time_col: Column containing timestamps

        Returns:
            Filtered DataFrame
        """
        df = df.copy()

        # Convert to timezone-aware
        if df[time_col].dt.tz is None:
            df[time_col] = df[time_col].dt.tz_localize('UTC')

        # Convert to local timezone
        local_time = df[time_col].dt.tz_convert(self.timezone)

        # Weekday filter (0=Monday, 4=Friday)
        is_weekday = local_time.dt.dayofweek < 5

        # Hour filter
        is_trading_hour = (
            (local_time.dt.hour >= self.start_hour) &
            (local_time.dt.hour < self.end_hour)
        )

        # Holiday filter
        date_str = local_time.dt.strftime('%Y-%m-%d')
        all_holidays = self.holidays_co | self.holidays_us
        is_not_holiday = ~date_str.isin(all_holidays)

        # Combined filter
        mask = is_weekday & is_trading_hour & is_not_holiday

        filtered = df[mask].copy()

        logger.info(
            f"Trading hours filter: {len(df)} -> {len(filtered)} rows "
            f"({len(filtered)/len(df)*100:.1f}% retained)"
        )

        return filtered

    @classmethod
    def from_calendar(cls, calendar_path: str) -> 'TradingHoursFilter':
        """Load holidays from calendar config file."""
        import yaml

        with open(calendar_path) as f:
            config = yaml.safe_load(f)

        return cls(
            timezone=config.get('timezone', 'America/Bogota'),
            start_hour=config.get('start_hour', 8),
            end_hour=config.get('end_hour', 17),
            holidays_co=config.get('holidays_co', []),
            holidays_us=config.get('holidays_us', [])
        )
```

---

## P0-08: Tests para MLflow Dataset Logging

**Estado**: NO CUMPLE
**Impacto**: TEST-DST-01 a TEST-DST-05 (5 preguntas)
**Tiempo**: 4h

**Solucion**:

```python
# tests/integration/test_mlflow_dataset_tracking.py

import pytest
import mlflow
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from scripts.train_with_mlflow import MLflowTrainer


class TestMLflowDatasetLineage:
    """Integration tests for MLflow dataset lineage tracking."""

    @pytest.fixture
    def trainer(self, tmp_path):
        """Create trainer with temp MLflow tracking."""
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        config = {
            'experiment_name': 'test_experiment',
            'data_path': 'data/test.parquet'
        }
        return MLflowTrainer(config)

    @pytest.fixture
    def sample_df(self):
        """Create sample training dataframe."""
        np.random.seed(42)
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=1000, freq='5min'),
            'log_ret_5m': np.random.randn(1000) * 0.001,
            'log_ret_1h': np.random.randn(1000) * 0.005,
            'rsi_9': np.random.uniform(0, 100, 1000),
            # ... other features
        })

    def test_dataset_hash_is_logged(self, trainer, sample_df):
        """Verify dataset_hash is logged to MLflow."""
        with mlflow.start_run():
            hash_value = trainer.log_dataset_lineage(sample_df, "test_path")

            run = mlflow.active_run()
            params = mlflow.get_run(run.info.run_id).data.params

            assert 'dataset_hash' in params
            assert params['dataset_hash'] == hash_value
            assert len(hash_value) == 16  # SHA256 truncated

    def test_date_range_is_logged(self, trainer, sample_df):
        """Verify train_start_date and train_end_date are logged."""
        with mlflow.start_run():
            trainer.log_dataset_lineage(sample_df, "test_path")

            run = mlflow.active_run()
            params = mlflow.get_run(run.info.run_id).data.params

            assert 'train_start_date' in params
            assert 'train_end_date' in params
            assert params['train_start_date'] == '2024-01-01T00:00:00'

    def test_feature_order_hash_is_logged(self, trainer, sample_df):
        """Verify FEATURE_ORDER_HASH is logged for integrity."""
        from src.core.contracts.feature_contract import FEATURE_ORDER_HASH

        with mlflow.start_run():
            trainer.log_dataset_lineage(sample_df, "test_path")

            params = mlflow.get_run(
                mlflow.active_run().info.run_id
            ).data.params

            assert params['feature_order_hash'] == FEATURE_ORDER_HASH

    def test_feature_list_artifact_created(self, trainer, sample_df, tmp_path):
        """Verify feature_contract.json artifact is created."""
        with mlflow.start_run():
            trainer.log_feature_contract_artifact()

            run_id = mlflow.active_run().info.run_id
            artifact_path = tmp_path / "mlruns" / "0" / run_id / "artifacts"

            contract_file = artifact_path / "contracts" / "feature_contract.json"
            assert contract_file.exists()

            import json
            with open(contract_file) as f:
                contract = json.load(f)

            assert contract['observation_dim'] == 15
            assert len(contract['features']) == 15

    def test_hash_deterministic_for_same_data(self, trainer):
        """Same data should always produce same hash."""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})

        with mlflow.start_run():
            hash1 = trainer.log_dataset_lineage(df, "path1")

        with mlflow.start_run():
            hash2 = trainer.log_dataset_lineage(df, "path2")

        assert hash1 == hash2

    def test_hash_changes_for_different_data(self, trainer):
        """Different data should produce different hash."""
        df1 = pd.DataFrame({'a': [1, 2, 3]})
        df2 = pd.DataFrame({'a': [1, 2, 4]})  # Changed last value

        with mlflow.start_run():
            hash1 = trainer.log_dataset_lineage(df1, "path")

        with mlflow.start_run():
            hash2 = trainer.log_dataset_lineage(df2, "path")

        assert hash1 != hash2


class TestDatasetReproduction:
    """Tests for dataset reproduction from MLflow runs."""

    @pytest.fixture
    def reproducer(self, tmp_path):
        from scripts.reproduce_dataset_from_run import DatasetReproducer
        mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
        return DatasetReproducer(f"file://{tmp_path}/mlruns")

    def test_get_run_metadata_extracts_all_fields(self, reproducer):
        """Verify all required metadata fields are extracted."""
        # Create a run with all required params
        with mlflow.start_run() as run:
            mlflow.log_param('dataset_hash', 'abc123')
            mlflow.log_param('dataset_path', 'data/train.parquet')
            mlflow.log_param('train_start_date', '2024-01-01')
            mlflow.log_param('train_end_date', '2024-06-30')
            mlflow.log_param('feature_order_hash', 'xyz789')
            run_id = run.info.run_id

        metadata = reproducer.get_run_metadata(run_id)

        assert metadata['dataset_hash'] == 'abc123'
        assert metadata['train_start_date'] == '2024-01-01'
        assert metadata['feature_order_hash'] == 'xyz789'

    def test_get_run_metadata_fails_on_missing_params(self, reproducer):
        """Verify error when required params missing."""
        with mlflow.start_run() as run:
            mlflow.log_param('dataset_hash', 'abc123')
            # Missing other required params
            run_id = run.info.run_id

        with pytest.raises(ValueError, match="missing required params"):
            reproducer.get_run_metadata(run_id)
```

---

## P0-09: ELIMINAR BFILL - CRITICO

**Estado**: ❌ NO CUMPLE - BLOQUEADOR ABSOLUTO
**Impacto**: FF-08 (1 pregunta, pero CRITICA)
**Tiempo**: 1h
**Severidad**: CRITICA - Viola causalidad temporal

**Problema**: Se encontró `bfill()` en archivos de producción que viola la integridad temporal.

**Archivos a Corregir**:

### 1. Dataset Builder (CRITICO)

```python
# Archivo: data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py
# Linea: 616

# ANTES (INCORRECTO - CREA LOOK-AHEAD BIAS):
df_ds = df_ds.ffill().bfill()

# DESPUES (CORRECTO):
df_ds = df_ds.ffill()  # Solo forward-fill, nunca backward
```

### 2. LLM Features (Deprecated pero debe corregirse)

```python
# Archivo: archive/airflow-dags/deprecated/usdcop_m5__04b_l3_llm_features.py
# Linea: 467

# ANTES (INCORRECTO):
df.groupby('episode_id')[LLM_FEATURES].ffill().bfill()

# DESPUES (CORRECTO):
df.groupby('episode_id')[LLM_FEATURES].ffill()
```

### 3. Test File

```python
# Archivo: tests/chaos/test_nan_handling.py
# Linea: 57

# ANTES (INCORRECTO):
df_clean = df.ffill().bfill()

# DESPUES (CORRECTO):
df_clean = df.ffill()
```

**Test de Validacion**:

```bash
# Verificar que NO existe bfill en codigo de produccion
grep -r "\.bfill()" --include="*.py" data/ src/ airflow/ scripts/ services/ | grep -v "test" | grep -v "archive"

# Debe retornar VACIO
```

---

## P0-10: Metricas Macro Ingestion

**Estado**: ❌ NO CUMPLE
**Impacto**: MET-01, MON-06, SCRP-31 (3 preguntas)
**Tiempo**: 4h

**Problema**: No existen metricas de Prometheus para macro ingestion por fuente.

**Solucion**:

```python
# Archivo: services/common/prometheus_metrics.py
# Agregar:

from prometheus_client import Counter, Gauge, Histogram

# Metricas de ingestion macro por fuente
macro_ingestion_success = Counter(
    'macro_ingestion_success_total',
    'Total successful macro data ingestions',
    ['source', 'indicator']  # Labels: FRED, TwelveData, BanRep, Investing, BCRP
)

macro_ingestion_errors = Counter(
    'macro_ingestion_errors_total',
    'Total failed macro data ingestions',
    ['source', 'indicator', 'error_type']
)

macro_data_staleness = Gauge(
    'macro_data_staleness_seconds',
    'Age of latest macro data in seconds',
    ['source', 'indicator']
)

macro_ingestion_latency = Histogram(
    'macro_ingestion_latency_seconds',
    'Latency of macro data ingestion',
    ['source'],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120]
)
```

**Instrumentar en DAG L0**:

```python
# Archivo: airflow/dags/l0_macro_unified.py
# En cada funcion de fetch, agregar:

from services.common.prometheus_metrics import (
    macro_ingestion_success,
    macro_ingestion_errors,
    macro_data_staleness
)

def fetch_fred_indicator(indicator_id: str, **kwargs):
    start_time = time.time()
    try:
        result = fred_client.get_series(indicator_id)

        # Registrar exito
        macro_ingestion_success.labels(
            source='FRED',
            indicator=indicator_id
        ).inc()

        # Registrar staleness
        if result:
            latest_date = result[-1]['date']
            age_seconds = (datetime.now() - latest_date).total_seconds()
            macro_data_staleness.labels(
                source='FRED',
                indicator=indicator_id
            ).set(age_seconds)

        return result

    except Exception as e:
        macro_ingestion_errors.labels(
            source='FRED',
            indicator=indicator_id,
            error_type=type(e).__name__
        ).inc()
        raise
```

---

## P0-11: Dashboard Macro Ingestion

**Estado**: ❌ NO CUMPLE
**Impacto**: SCRP-31 (1 pregunta)
**Tiempo**: 3h

**Solucion**: Crear dashboard Grafana `config/grafana/dashboards/macro-ingestion.json`

```json
{
  "title": "Macro Data Ingestion",
  "panels": [
    {
      "title": "Success Rate by Source",
      "type": "gauge",
      "targets": [
        {
          "expr": "sum(rate(macro_ingestion_success_total[1h])) by (source) / (sum(rate(macro_ingestion_success_total[1h])) by (source) + sum(rate(macro_ingestion_errors_total[1h])) by (source)) * 100"
        }
      ]
    },
    {
      "title": "Data Staleness by Indicator",
      "type": "table",
      "targets": [
        {
          "expr": "macro_data_staleness_seconds"
        }
      ]
    },
    {
      "title": "Ingestion Errors (24h)",
      "type": "timeseries",
      "targets": [
        {
          "expr": "sum(increase(macro_ingestion_errors_total[1h])) by (source, error_type)"
        }
      ]
    }
  ]
}
```

---

## P0-12: Columna release_date en Database

**Estado**: ⚠️ PARCIAL
**Impacto**: FRED-32 (1 pregunta)
**Tiempo**: 2h

**Problema**: No hay columna explicita `release_date` en `macro_indicators_daily`.

**Solucion - Migration Alembic**:

```python
# database/alembic/versions/xxx_add_release_date_column.py

def upgrade():
    op.add_column(
        'macro_indicators_daily',
        sa.Column('release_date', sa.Date(), nullable=True)
    )
    op.add_column(
        'macro_indicators_daily',
        sa.Column('ffilled_from_date', sa.Date(), nullable=True)
    )

    # Index para queries PIT
    op.create_index(
        'ix_macro_release_date',
        'macro_indicators_daily',
        ['release_date']
    )

def downgrade():
    op.drop_index('ix_macro_release_date')
    op.drop_column('macro_indicators_daily', 'release_date')
    op.drop_column('macro_indicators_daily', 'ffilled_from_date')
```

---

## Checklist P0 Completado (Actualizado)

```bash
# Ejecutar para validar P0
python -c "
from scripts.train_with_mlflow import MLflowTrainer
from scripts.reproduce_dataset_from_run import DatasetReproducer
from src.features.temporal_joins import merge_price_with_macro
from src.features.trading_hours_filter import TradingHoursFilter
from services.inference_api.contracts.model_contract import ModelContract

# Verify new fields exist
assert hasattr(MLflowTrainer, 'log_dataset_lineage')
assert hasattr(MLflowTrainer, 'log_feature_contract_artifact')
assert 'feature_service_name' in ModelContract.__dataclass_fields__

print('P0 Validation PASSED')
"

# Verificar NO existe bfill
bfill_count=$(grep -r '\.bfill()' --include='*.py' data/ src/ airflow/ scripts/ services/ 2>/dev/null | grep -v test | grep -v archive | wc -l)
if [ "$bfill_count" -gt 0 ]; then
    echo "FALLO: bfill() encontrado en $bfill_count archivos"
    exit 1
fi
echo "OK: No bfill en produccion"

# Verificar metricas macro existen
python -c "
from services.common.prometheus_metrics import macro_ingestion_success
print('OK: Metricas macro definidas')
"
```

---

# FASE P1: CRITICOS (Semanas 3-4)

## Objetivo: Alcanzar 72% Compliance

---

## P1-01: Feature Services Parametrizados en Feast

**Estado**: NO CUMPLE
**Impacto**: FEAST-20 a FEAST-30 (11 preguntas)
**Tiempo**: 6h

**Problema**: Solo existe 1 Feature Service, no hay parametrizacion para experimentos.

**Solucion**:

```python
# feature_repo/feature_services.py

from feast import FeatureService, Entity, Field
from feast.types import Float32, Int64
from datetime import timedelta

from .feature_views import (
    technical_fv,
    macro_fv,
    state_fv,
)


# Base production service (all 15 features)
ppo_production_service = FeatureService(
    name="ppo_production_service",
    features=[
        technical_fv[["log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14"]],
        macro_fv[["dxy_z", "dxy_change_1d", "vix_z", "embi_z", "brent_change_1d", "rate_spread", "usdmxn_change_1d"]],
        state_fv[["position", "time_normalized"]],
    ],
    description="Production PPO model - 15 features (6 technical + 7 macro + 2 state)",
    tags={
        "version": "2.0.0",
        "observation_dim": "15",
        "model_type": "ppo",
    },
)


# Experiment: Technical only (no macro)
technical_only_service = FeatureService(
    name="technical_only_service",
    features=[
        technical_fv[["log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14"]],
        state_fv[["position", "time_normalized"]],
    ],
    description="Technical features only - 8 features",
    tags={
        "version": "1.0.0",
        "observation_dim": "8",
        "model_type": "ppo_technical",
        "experiment": "ablation_no_macro",
    },
)


# Experiment: Minimal macro (DXY + VIX only)
minimal_macro_service = FeatureService(
    name="minimal_macro_service",
    features=[
        technical_fv[["log_ret_5m", "log_ret_1h", "log_ret_4h", "rsi_9", "atr_pct", "adx_14"]],
        macro_fv[["dxy_z", "vix_z"]],
        state_fv[["position", "time_normalized"]],
    ],
    description="Minimal macro (DXY + VIX) - 10 features",
    tags={
        "version": "1.0.0",
        "observation_dim": "10",
        "model_type": "ppo_minimal_macro",
        "experiment": "ablation_minimal_macro",
    },
)


# Experiment: Full macro (7 + EMBI variants)
extended_macro_service = FeatureService(
    name="extended_macro_service",
    features=[
        technical_fv,
        macro_fv,
        state_fv,
    ],
    description="Extended macro features - 15+ features",
    tags={
        "version": "1.0.0",
        "observation_dim": "18",
        "model_type": "ppo_extended",
        "experiment": "more_macro",
    },
)


# Factory function for custom feature services
def create_experiment_service(
    name: str,
    technical_features: list[str],
    macro_features: list[str],
    state_features: list[str] = ["position", "time_normalized"],
    description: str = "",
) -> FeatureService:
    """
    Create a custom Feature Service for experiments.

    Args:
        name: Unique service name
        technical_features: List of technical feature names
        macro_features: List of macro feature names
        state_features: List of state feature names
        description: Human-readable description

    Returns:
        FeatureService configured with specified features
    """
    features = []

    if technical_features:
        features.append(technical_fv[technical_features])
    if macro_features:
        features.append(macro_fv[macro_features])
    if state_features:
        features.append(state_fv[state_features])

    obs_dim = len(technical_features) + len(macro_features) + len(state_features)

    return FeatureService(
        name=name,
        features=features,
        description=description or f"Custom service with {obs_dim} features",
        tags={
            "observation_dim": str(obs_dim),
            "technical_count": str(len(technical_features)),
            "macro_count": str(len(macro_features)),
            "state_count": str(len(state_features)),
        },
    )
```

---

## P1-02: Tests de Integridad Feature-Model

**Estado**: NO CUMPLE
**Impacto**: FEAST-35 a FEAST-40 (6 preguntas)
**Tiempo**: 4h

```python
# tests/contracts/test_feature_model_integrity.py

import pytest
from src.core.contracts.feature_contract import (
    FEATURE_ORDER, OBSERVATION_DIM, FEATURE_ORDER_HASH
)
from services.inference_api.contracts.model_contract import (
    ModelRegistry, BuilderType
)


class TestFeatureModelIntegrity:
    """Tests verifying feature contract and model contract alignment."""

    def test_model_observation_dim_matches_feature_order(self):
        """All models must have observation_dim == len(FEATURE_ORDER)."""
        for model_id, contract in ModelRegistry.list_models().items():
            if contract.builder_type == BuilderType.CURRENT_15DIM:
                assert contract.observation_dim == OBSERVATION_DIM, (
                    f"Model {model_id} has observation_dim={contract.observation_dim} "
                    f"but FEATURE_ORDER has {OBSERVATION_DIM} features"
                )

    def test_feature_service_exists_for_model(self):
        """Models with feature_service_name must have valid Feast service."""
        from feast import FeatureStore

        fs = FeatureStore(repo_path="feature_repo/")
        available_services = {fs.name for fs in fs.list_feature_services()}

        for model_id, contract in ModelRegistry.list_models().items():
            if contract.feature_service_name:
                assert contract.feature_service_name in available_services, (
                    f"Model {model_id} references feature_service "
                    f"'{contract.feature_service_name}' which does not exist. "
                    f"Available: {available_services}"
                )

    def test_feature_service_dim_matches_model(self):
        """Feature service observation_dim tag must match model."""
        from feast import FeatureStore

        fs = FeatureStore(repo_path="feature_repo/")

        for model_id, contract in ModelRegistry.list_models().items():
            if contract.feature_service_name:
                service = fs.get_feature_service(contract.feature_service_name)
                service_dim = int(service.tags.get('observation_dim', 0))

                assert service_dim == contract.observation_dim, (
                    f"Model {model_id} has observation_dim={contract.observation_dim} "
                    f"but feature_service '{contract.feature_service_name}' "
                    f"has observation_dim={service_dim}"
                )

    def test_no_hardcoded_feature_lists(self):
        """Verify no hardcoded feature lists outside SSOT."""
        import ast
        from pathlib import Path

        hardcoded_patterns = [
            'log_ret_5m', 'log_ret_1h', 'dxy_z', 'vix_z'
        ]

        violations = []

        for py_file in Path('src').rglob('*.py'):
            if 'feature_contract' in str(py_file):
                continue  # Skip SSOT file

            content = py_file.read_text()

            for pattern in hardcoded_patterns:
                if f'"{pattern}"' in content or f"'{pattern}'" in content:
                    # Check if it's an import from feature_contract
                    if 'from src.core.contracts.feature_contract import' in content:
                        continue
                    violations.append((py_file, pattern))

        assert not violations, (
            f"Found hardcoded feature names outside SSOT:\n" +
            "\n".join(f"  {f}: {p}" for f, p in violations)
        )
```

---

## P1-03: Point-in-Time Joins en Feast

**Estado**: NO CUMPLE
**Impacto**: FEAST-45 a FEAST-50 (6 preguntas)
**Tiempo**: 4h

```python
# src/feature_store/feast_pit_joins.py

from feast import FeatureStore
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class FeastPITRetriever:
    """
    Point-in-Time feature retrieval from Feast.

    Ensures:
    1. No look-ahead bias
    2. Correct temporal alignment
    3. TTL-based feature staleness
    """

    def __init__(self, repo_path: str = "feature_repo/"):
        self.fs = FeatureStore(repo_path=repo_path)

    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        feature_service_name: str,
        full_feature_names: bool = False,
    ) -> pd.DataFrame:
        """
        Retrieve historical features with point-in-time correctness.

        Args:
            entity_df: DataFrame with entity keys and event_timestamp
            feature_service_name: Name of Feature Service to use
            full_feature_names: Whether to use full feature names (fv__feature)

        Returns:
            DataFrame with features joined point-in-time
        """
        feature_service = self.fs.get_feature_service(feature_service_name)

        # Validate entity_df has required columns
        if 'event_timestamp' not in entity_df.columns:
            raise ValueError("entity_df must have 'event_timestamp' column")

        # Ensure timestamp is timezone-aware UTC
        entity_df = entity_df.copy()
        if entity_df['event_timestamp'].dt.tz is None:
            entity_df['event_timestamp'] = entity_df['event_timestamp'].dt.tz_localize('UTC')

        logger.info(
            f"Retrieving features from {feature_service_name} "
            f"for {len(entity_df)} entities"
        )

        # Get historical features with PIT correctness
        features_df = self.fs.get_historical_features(
            entity_df=entity_df,
            features=feature_service,
            full_feature_names=full_feature_names,
        ).to_df()

        logger.info(f"Retrieved {len(features_df)} rows with {len(features_df.columns)} columns")

        return features_df

    def get_online_features(
        self,
        entity_rows: List[dict],
        feature_service_name: str,
    ) -> pd.DataFrame:
        """
        Retrieve latest feature values for online inference.

        Args:
            entity_rows: List of entity dictionaries
            feature_service_name: Name of Feature Service

        Returns:
            DataFrame with latest feature values
        """
        feature_service = self.fs.get_feature_service(feature_service_name)

        response = self.fs.get_online_features(
            features=feature_service,
            entity_rows=entity_rows,
        )

        return response.to_df()

    def validate_pit_correctness(
        self,
        features_df: pd.DataFrame,
        entity_df: pd.DataFrame,
        max_staleness: timedelta = timedelta(days=3),
    ) -> dict:
        """
        Validate point-in-time correctness of retrieved features.

        Checks:
        1. No future data (look-ahead bias)
        2. Feature staleness within limits

        Returns:
            Dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
        }

        # Check for future-dated features (would indicate PIT failure)
        if '_feature_timestamp' in features_df.columns:
            future_data = features_df[
                features_df['_feature_timestamp'] > features_df['event_timestamp']
            ]
            if len(future_data) > 0:
                results['valid'] = False
                results['errors'].append(
                    f"Found {len(future_data)} rows with future-dated features (look-ahead bias)"
                )

        # Check staleness
        if '_feature_timestamp' in features_df.columns:
            staleness = features_df['event_timestamp'] - features_df['_feature_timestamp']
            stale_rows = staleness > max_staleness
            if stale_rows.any():
                results['warnings'].append(
                    f"{stale_rows.sum()} rows have features older than {max_staleness}"
                )

        return results
```

---

## P1-04: DVC Lock File Management

**Estado**: PARCIAL
**Impacto**: DVC-30 a DVC-40 (11 preguntas)
**Tiempo**: 3h

```yaml
# dvc.yaml (actualizado con outputs y deps completos)

stages:
  prepare_data:
    cmd: python scripts/prepare_training_data.py
    deps:
      - scripts/prepare_training_data.py
      - src/core/contracts/feature_contract.py
      - config/data_config.yaml
    params:
      - prepare_data
    outs:
      - data/processed/training_data.parquet:
          cache: true
          persist: true
    metrics:
      - data/processed/data_stats.json:
          cache: false

  calculate_norm_stats:
    cmd: python scripts/calculate_norm_stats.py
    deps:
      - scripts/calculate_norm_stats.py
      - data/processed/training_data.parquet
      - src/core/contracts/feature_contract.py
    outs:
      - config/norm_stats.json:
          cache: true
    plots:
      - data/processed/feature_distributions.csv:
          x: feature
          y: mean

  train:
    cmd: python scripts/train_with_mlflow.py --config config/training_config.yaml
    deps:
      - scripts/train_with_mlflow.py
      - data/processed/training_data.parquet
      - config/norm_stats.json
      - config/training_config.yaml
      - src/core/contracts/feature_contract.py
    params:
      - train
    outs:
      - models/ppo_production/:
          cache: true
          persist: true
    metrics:
      - models/ppo_production/metrics.json:
          cache: false

  evaluate:
    cmd: python scripts/evaluate_model.py
    deps:
      - scripts/evaluate_model.py
      - models/ppo_production/
      - data/processed/test_data.parquet
    metrics:
      - models/ppo_production/evaluation_metrics.json:
          cache: false
    plots:
      - models/ppo_production/equity_curve.csv:
          x: step
          y: equity

  export_onnx:
    cmd: python scripts/export_onnx.py
    deps:
      - scripts/export_onnx.py
      - models/ppo_production/final_model.zip
    outs:
      - models/ppo_production/model.onnx:
          cache: true

  backtest:
    cmd: python scripts/run_backtest.py
    deps:
      - scripts/run_backtest.py
      - models/ppo_production/final_model.zip
      - data/processed/backtest_data.parquet
    metrics:
      - models/ppo_production/backtest_metrics.json:
          cache: false
    plots:
      - models/ppo_production/backtest_equity.csv:
          x: date
          y: equity

  promote:
    cmd: python scripts/promote_model.py
    deps:
      - scripts/promote_model.py
      - models/ppo_production/final_model.zip
      - models/ppo_production/evaluation_metrics.json
      - models/ppo_production/backtest_metrics.json
```

---

## P1-05: Documentacion DVC_GUIDE.md

**Estado**: NO EXISTE
**Impacto**: DVC-45 a DVC-50 (6 preguntas)
**Tiempo**: 2h

```markdown
# DVC Guide - Dataset Versioning
## USD/COP RL Trading System

## Overview

DVC (Data Version Control) tracks:
- Training datasets (`data/processed/`)
- Normalization stats (`config/norm_stats.json`)
- Trained models (`models/`)

## Quick Start

```bash
# Pull latest data
dvc pull

# Run full pipeline
dvc repro

# Run specific stage
dvc repro train
```

## Pipeline Stages

```
prepare_data -> calculate_norm_stats -> train -> evaluate -> export_onnx -> backtest -> promote
```

## Versioning Datasets

```bash
# After creating new dataset
dvc add data/processed/training_data.parquet
git add data/processed/training_data.parquet.dvc
git commit -m "Dataset v1.2.0 - added 2024 data"
dvc push

# Tag the version
git tag -a dataset-v1.2.0 -m "Training dataset v1.2.0"
```

## Checkout Specific Version

```bash
# By git tag
git checkout dataset-v1.2.0
dvc checkout

# By commit SHA
git checkout abc123
dvc checkout
```

## Storage Backend

Remote storage: MinIO (S3-compatible)

```bash
# config/.dvc/config
[core]
    remote = minio
[remote "minio"]
    url = s3://dvc-storage
    endpointurl = http://minio:9000
```

## Params

Training parameters in `params.yaml`:

```yaml
prepare_data:
  start_date: "2023-01-01"
  end_date: "2024-12-31"

train:
  learning_rate: 0.0003
  batch_size: 64
  total_timesteps: 1000000
```

## Metrics Tracking

```bash
# View metrics
dvc metrics show

# Compare experiments
dvc metrics diff HEAD~1
```

## Troubleshooting

### Cache Issues
```bash
dvc cache gc -w  # Clean unused cache
dvc gc -w        # Garbage collect workspace
```

### Lock File Conflicts
```bash
rm dvc.lock
dvc repro --force
```
```

---

## P1-06: Tests DVC Pipeline

**Estado**: NO CUMPLE
**Impacto**: DVC-25 a DVC-29 (5 preguntas)
**Tiempo**: 3h

```python
# tests/integration/test_dvc_pipeline.py

import pytest
import subprocess
import yaml
from pathlib import Path


class TestDVCPipeline:
    """Integration tests for DVC pipeline."""

    @pytest.fixture
    def dvc_yaml(self):
        with open('dvc.yaml') as f:
            return yaml.safe_load(f)

    def test_all_stages_have_deps(self, dvc_yaml):
        """Every stage must have dependencies defined."""
        for stage_name, stage in dvc_yaml['stages'].items():
            assert 'deps' in stage, f"Stage {stage_name} missing deps"
            assert len(stage['deps']) > 0, f"Stage {stage_name} has empty deps"

    def test_feature_contract_is_dependency(self, dvc_yaml):
        """Feature contract should be dep for data-processing stages."""
        data_stages = ['prepare_data', 'calculate_norm_stats', 'train']

        for stage_name in data_stages:
            if stage_name in dvc_yaml['stages']:
                deps = dvc_yaml['stages'][stage_name]['deps']
                assert any('feature_contract' in d for d in deps), (
                    f"Stage {stage_name} should depend on feature_contract.py"
                )

    def test_dvc_dag_is_valid(self):
        """DVC DAG should be valid (no cycles)."""
        result = subprocess.run(
            ['dvc', 'dag', '--dot'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"DVC DAG invalid: {result.stderr}"

    def test_dvc_status_clean(self):
        """After dvc repro, status should be clean."""
        # This is a "should be" test - may fail if data changed
        result = subprocess.run(
            ['dvc', 'status'],
            capture_output=True,
            text=True
        )
        # Warning: this will fail if there are uncommitted changes
        # In CI, we check that the pipeline definition is valid
        assert result.returncode == 0

    def test_params_yaml_exists_and_valid(self):
        """params.yaml must exist and be valid YAML."""
        params_path = Path('params.yaml')
        assert params_path.exists(), "params.yaml not found"

        with open(params_path) as f:
            params = yaml.safe_load(f)

        assert 'prepare_data' in params or 'train' in params, (
            "params.yaml should have prepare_data or train sections"
        )

    def test_dvc_lock_exists(self):
        """dvc.lock should exist after successful repro."""
        lock_path = Path('dvc.lock')
        # Note: may not exist in fresh clone before dvc repro
        if lock_path.exists():
            with open(lock_path) as f:
                lock = yaml.safe_load(f)
            assert 'stages' in lock
```

---

## P1-07: Norm Stats Schema Validation

**Estado**: PARCIAL
**Impacto**: DST-40 a DST-45 (6 preguntas)
**Tiempo**: 2h

```python
# src/validation/norm_stats_validator.py

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

from src.core.contracts.feature_contract import FEATURE_ORDER, FEATURE_SPECS

logger = logging.getLogger(__name__)


class NormStatsValidator:
    """
    Validates normalization statistics file.

    Ensures:
    1. All required features present
    2. Each feature has mean and std
    3. std > 0 (no constant features)
    4. Values are reasonable (no NaN/Inf)
    """

    REQUIRED_KEYS = {'mean', 'std'}

    def validate(self, norm_stats_path: Path) -> Tuple[bool, List[str]]:
        """
        Validate norm_stats.json file.

        Returns:
            (is_valid, list of errors)
        """
        errors = []

        # Load file
        try:
            with open(norm_stats_path) as f:
                stats = json.load(f)
        except FileNotFoundError:
            return False, [f"File not found: {norm_stats_path}"]
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"]

        # Check required features
        normalizable = [
            name for name, spec in FEATURE_SPECS.items()
            if spec.requires_normalization
        ]

        missing = set(normalizable) - set(stats.keys())
        if missing:
            errors.append(f"Missing features: {sorted(missing)}")

        # Validate each feature
        for name, values in stats.items():
            if name not in FEATURE_SPECS:
                errors.append(f"Unknown feature in norm_stats: {name}")
                continue

            # Check required keys
            missing_keys = self.REQUIRED_KEYS - set(values.keys())
            if missing_keys:
                errors.append(f"{name}: missing keys {missing_keys}")
                continue

            # Check values
            mean = values['mean']
            std = values['std']

            if not isinstance(mean, (int, float)):
                errors.append(f"{name}: mean is not numeric")
            elif mean != mean:  # NaN check
                errors.append(f"{name}: mean is NaN")

            if not isinstance(std, (int, float)):
                errors.append(f"{name}: std is not numeric")
            elif std != std:  # NaN check
                errors.append(f"{name}: std is NaN")
            elif std <= 0:
                errors.append(f"{name}: std must be > 0, got {std}")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info(f"norm_stats validated: {len(stats)} features OK")
        else:
            logger.error(f"norm_stats validation failed: {len(errors)} errors")

        return is_valid, errors


# JSON Schema for norm_stats.json
NORM_STATS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "additionalProperties": {
        "type": "object",
        "required": ["mean", "std"],
        "properties": {
            "mean": {"type": "number"},
            "std": {"type": "number", "exclusiveMinimum": 0}
        }
    },
    "required": list(
        name for name, spec in FEATURE_SPECS.items()
        if spec.requires_normalization
    )
}
```

---

## P1-08: REPRODUCIBILITY.md Documentation

**Estado**: NO EXISTE
**Impacto**: REPR-10 a REPR-15 (6 preguntas)
**Tiempo**: 2h

```markdown
# Reproducibility Guide
## USD/COP RL Trading System

## Philosophy

Every training run should be **100% reproducible** given:
1. MLflow run ID
2. Git commit SHA
3. DVC data version

## Reproducibility Checklist

Before training:
- [ ] Commit all code changes to git
- [ ] Run `dvc status` (should be clean)
- [ ] Verify `config/norm_stats.json` matches training data

After training:
- [ ] MLflow logs `dataset_hash`
- [ ] MLflow logs `feature_order_hash`
- [ ] MLflow logs `train_start_date` and `train_end_date`
- [ ] Model artifacts saved to MLflow

## Reproduce a Training Run

```bash
# 1. Get run metadata
python scripts/reproduce_dataset_from_run.py --run-id <RUN_ID> --output data/reproduced/

# 2. Checkout code version
git checkout $(mlflow runs get --run-id <RUN_ID> | grep git_commit | awk '{print $2}')

# 3. Checkout data version
dvc checkout

# 4. Verify dataset hash
python -c "
import pandas as pd
import hashlib
df = pd.read_parquet('data/reproduced/reproduced_<RUN_ID>.parquet')
hash = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
print(f'Hash: {hash}')
"

# 5. Re-run training with same params
mlflow run . --experiment-name reproduce_<RUN_ID>
```

## Key Files for Reproducibility

| File | Purpose | Versioned By |
|------|---------|--------------|
| `src/core/contracts/feature_contract.py` | Feature order SSOT | Git |
| `config/norm_stats.json` | Normalization stats | DVC |
| `data/processed/training_data.parquet` | Training data | DVC |
| `params.yaml` | Training params | Git + DVC |
| `models/ppo_production/` | Trained model | DVC + MLflow |

## Troubleshooting

### Hash Mismatch
If dataset hash doesn't match:
1. Check DVC version: `dvc version`
2. Check data version: `dvc diff`
3. Verify no uncommitted changes to `prepare_training_data.py`

### Feature Order Changed
If `feature_order_hash` doesn't match:
1. Check git history of `feature_contract.py`
2. Retrain if feature order changed (model incompatible)
```

---

# FASE P2: IMPORTANTES (Semanas 5-6)

## Objetivo: Alcanzar 85% Compliance

---

## P2-01: Feast Materialization Jobs

**Impacto**: FEAST-01 a FEAST-10 (10 preguntas)
**Tiempo**: 5h

## P2-02: Alembic Migrations para Feature Snapshots

**Impacto**: DST-45 a DST-50 (6 preguntas)
**Tiempo**: 4h

## P2-03: MLflow Model Registry Integration

**Impacto**: MLF-25 a MLF-30 (6 preguntas)
**Tiempo**: 4h

## P2-04: Tests E2E de Reproducibility

**Impacto**: REPR-16 a REPR-20 (5 preguntas)
**Tiempo**: 6h

---

# FASE P3: MEJORAS (Semanas 7-8)

## Objetivo: Alcanzar 100% Compliance

---

## P3-01: Feature Store Monitoring Dashboard

**Impacto**: FEAST-51 a FEAST-55 (5 preguntas)
**Tiempo**: 6h

## P3-02: DVC Remote Caching Optimization

**Impacto**: DVC-46 a DVC-50 (5 preguntas)
**Tiempo**: 4h

## P3-03: MLflow Experiment Comparison UI

**Impacto**: MLF-26 a MLF-30 (5 preguntas)
**Tiempo**: 5h

## P3-04: Automated Dataset Quality Reports

**Impacto**: DST-46 a DST-50 (5 preguntas)
**Tiempo**: 6h

## P3-05: CI/CD for Feature Store

**Impacto**: FEAST-56 a FEAST-60 (5 preguntas)
**Tiempo**: 8h

## P3-06: Documentation Site Generation

**Impacto**: General documentation (5 preguntas)
**Tiempo**: 6h

---

# CRONOGRAMA DE 9 SEMANAS (Actualizado)

```
Semana 1-2 (P0 - Bloqueantes): 35h (+10h macro)
├── P0-01: MLflow dataset lineage (3h)
├── P0-02: ModelContract feature_service (2h)
├── P0-03: reproduce_dataset_from_run.py (4h)
├── P0-04: DATASET_CONSTRUCTION.md (3h)
├── P0-05: FEATURE_ORDER artifact (1h)
├── P0-06: merge_asof implementation (3h)
├── P0-07: Trading hours filter (2h)
├── P0-08: MLflow logging tests (4h)
├── P0-09: ELIMINAR BFILL [CRITICO] (1h) ⚠️
├── P0-10: Metricas macro ingestion (4h)
├── P0-11: Dashboard macro ingestion (3h)
└── P0-12: Columna release_date (2h)
    Checkpoint: 55% compliance

Semana 3-4 (P1 - Criticos): 30h
├── P1-01: Feature Services parametrizados (6h)
├── P1-02: Feature-Model integrity tests (4h)
├── P1-03: Feast PIT joins (4h)
├── P1-04: DVC lock file management (3h)
├── P1-05: DVC_GUIDE.md (2h)
├── P1-06: DVC pipeline tests (3h)
├── P1-07: Norm stats validation (2h)
└── P1-08: REPRODUCIBILITY.md (2h)
    Checkpoint: 72% compliance

Semana 5-6 (P2 - Importantes): 25h
├── P2-01: Feast materialization (5h)
├── P2-02: Alembic migrations (4h)
├── P2-03: MLflow registry integration (4h)
├── P2-04: E2E reproducibility tests (6h)
└── P2-05: Additional documentation (6h)
    Checkpoint: 85% compliance

Semana 7-9 (P3 - Mejoras): 45h
├── P3-01: Feature Store monitoring (6h)
├── P3-02: DVC caching optimization (4h)
├── P3-03: MLflow comparison UI (5h)
├── P3-04: Dataset quality reports (6h)
├── P3-05: CI/CD for Feature Store (8h)
├── P3-06: Documentation site (6h)
├── P3-07: Data staleness per-variable metrics (5h)
└── P3-08: Final testing and polish (5h)
    Final: 100% compliance
```

---

# CRITERIOS DE ACEPTACION

## P0 Complete (55%)
- [ ] `mlflow.log_param('dataset_hash')` en todos los runs
- [ ] `reproduce_dataset_from_run.py` funcional
- [ ] `docs/DATASET_CONSTRUCTION.md` existe
- [ ] Trading hours filter implementado
- [ ] Tests de MLflow logging pasan
- [ ] **NO existe bfill() en codigo de produccion** ⚠️ CRITICO
- [ ] Metricas `macro_ingestion_success_total` definidas
- [ ] Dashboard macro ingestion creado

## P1 Complete (72%)
- [ ] 4+ Feature Services en Feast
- [ ] Tests de integridad Feature-Model pasan
- [ ] `docs/DVC_GUIDE.md` existe
- [ ] `docs/REPRODUCIBILITY.md` existe
- [ ] DVC pipeline tests pasan

## P2 Complete (85%)
- [ ] Feast materialization jobs configurados
- [ ] Alembic migrations para feature snapshots
- [ ] MLflow Model Registry integrado
- [ ] E2E reproducibility test pasa
- [ ] Columna release_date en macro_indicators_daily

## P3 Complete (100%)
- [ ] Feature Store dashboard funcional
- [ ] Todos los tests pasan
- [ ] Documentacion completa
- [ ] CI/CD para Feature Store
- [ ] Data staleness metrics per-variable
- [ ] 218/218 preguntas de auditoria cumplidas

---

# COMANDOS DE VALIDACION

```bash
# Validar P0
python scripts/validate_blockers.py --phase P0

# Validar P1
python scripts/validate_blockers.py --phase P1

# Validar compliance total
python scripts/validate_100_percent.py

# Ejecutar auditoria completa
python scripts/audit_dataset_reproducibility.py --questions 218

# Verificar NO existe bfill (CRITICO)
grep -r "\.bfill()" --include="*.py" data/ src/ airflow/ scripts/ services/ | grep -v test | grep -v archive
# Debe retornar VACIO

# Verificar metricas macro
curl -s http://localhost:9090/api/v1/query?query=macro_ingestion_success_total | jq .
```

---

**Documento generado**: 2026-01-17 (actualizado)
**Auditoria base**: 218 preguntas (DST, DVC, FEAST, MLF, REPR + MACRO SCRAPING)
**Score inicial**: 47% (103/218)
**Score objetivo**: 100% (200/200)
**Esfuerzo total**: 120 horas / 8 semanas
