# Plan Integral Consolidado - USDCOP RL Trading System
**Fecha**: 2026-02-01
**Versión**: 1.0
**Estado**: Propuesta para Revisión

---

## 📋 Resumen Ejecutivo

Este plan consolida los hallazgos de 6 análisis exhaustivos del proyecto USDCOP-RL-Models, identificando **12 problemas críticos** y **8 oportunidades de consolidación** que afectan la integridad del pipeline de datos y entrenamiento.

### Problemas de Mayor Impacto
| Prioridad | Problema | Impacto | Estado |
|-----------|----------|---------|--------|
| P0 | ADX Saturation Bug | Modelo aprende de indicador corrupto | Parcialmente corregido |
| P0 | Triple Normalización | Features sobre-normalizadas | Parcialmente corregido |
| P0 | L3 Training Terminación Temprana | Training no completa | Sin resolver |
| P1 | 3 Paths Redundantes L2 | Inconsistencias, mantenimiento difícil | Sin resolver |
| P1 | Z-score FIXED vs ROLLING | Macro features inconsistentes | Sin resolver |
| P2 | Episode Length Mismatch | 1200 vs 2000 configuración | Sin resolver |

---

## 🏗️ Arquitectura Actual vs Propuesta

### Arquitectura Actual (Problemática)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                         FLUJO ACTUAL (FRAGMENTADO)                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L0: Ingestion                                                          │
│  ├── l0_data_initialization.py (DELETED)                               │
│  ├── l0_macro_update.py (NEW - untracked)                              │
│  ├── l0_backup_restore.py (NEW - untracked)                            │
│  └── 04-data-seeding.py (init-scripts)                                 │
│       └── 3 FORMAS DIFERENTES de cargar datos iniciales                │
│                                                                         │
│  L2: Dataset Building  ⚠️ 3 PATHS REDUNDANTES                          │
│  ├── PATH A: 01_build_5min_datasets.py (standalone)                    │
│  │    └── ADX fix aplicado ✓                                           │
│  ├── PATH B: 02_build_daily_datasets.py (standalone)                   │
│  │    └── ADX fix NO aplicado ✗                                        │
│  └── PATH C: l2_dataset_builder.py (Airflow DAG)                       │
│       └── ADX fix aplicado pero NO recargado en Airflow ✗              │
│                                                                         │
│  L3: Training                                                           │
│  ├── engine.py → Recalcula norm_stats (DUPLICACIÓN)                    │
│  └── trading_env.py → Normaliza OTRA VEZ (TRIPLE NORM)                 │
│                                                                         │
│  L4: Backtest/Promotion                                                 │
│  └── ExternalTaskSensor apunta a DAGs que no existen                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Arquitectura Propuesta (Consolidada)
```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FLUJO PROPUESTO (CONSOLIDADO)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  L0: Data Layer (ÚNICO PUNTO DE ENTRADA)                               │
│  ├── l0_macro_update.py ──→ PostgreSQL macro_indicators                │
│  ├── l0_backup_restore.py ──→ Backup/Restore unificado                 │
│  └── Seeding ──→ 04-data-seeding.py (único script)                     │
│                                                                         │
│  L2: Dataset Building (CONSOLIDADO - UN SOLO PATH)                     │
│  └── l2_dataset_builder.py (Airflow DAG ÚNICO)                         │
│       ├── calculate_adx_wilders() ← FIX percentage-based               │
│       ├── calculate_rsi_wilders() ← Wilders smoothing                  │
│       ├── macro_z_scores() ← ROLLING 252d consistente                  │
│       └── Genera: DS_*.parquet + DS_*_norm_stats.json                  │
│                                                                         │
│  L3: Training (SIN RE-NORMALIZACIÓN)                                   │
│  ├── engine.py → USA norm_stats de L2 (no recalcula)                   │
│  └── trading_env.py → SKIP normalización si ya normalizado             │
│                                                                         │
│  L4: Backtest/Promotion                                                │
│  └── ExternalTaskSensor → l2_dataset_builder (correcto)                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Plan de Implementación por Fases

### FASE 1: Correcciones Críticas (P0) - Día 1-2

#### 1.1 Verificar ADX Fix en Airflow
**Problema**: El fix de ADX fue aplicado a `l2_dataset_builder.py` pero Airflow puede no haber recargado el DAG.

**Acción**:
```bash
# En contenedor Airflow
docker exec -it airflow-webserver bash
airflow dags reserialize
airflow dags list-import-errors
```

**Validación**:
```python
# Verificar que ADX tiene distribución normal después de regenerar
import pandas as pd
df = pd.read_parquet('data/pipeline/07_output/5min/DS_v2_fixed2.parquet')
adx_stats = df['adx_14'].describe()
assert adx_stats['mean'] < 40, f"ADX still saturated: mean={adx_stats['mean']}"
assert adx_stats['std'] > 10, f"ADX std too low: {adx_stats['std']}"
```

#### 1.2 Eliminar Triple Normalización
**Archivos a modificar**:

**`src/training/engine.py:496-555`** - Ya parcialmente corregido, verificar:
```python
def _generate_norm_stats(self, df, request):
    """USAR norm_stats de L2, NO recalcular."""
    l2_norm_stats_path = request.dataset_path.parent / f"DS_{request.experiment_name or 'default'}_norm_stats.json"

    if l2_norm_stats_path.exists():
        logger.info(f"✓ Using L2 norm_stats (no re-computation)")
        with open(l2_norm_stats_path, 'r') as f:
            return json.load(f)['features']

    # FALLBACK: Solo si L2 no generó stats (no debería pasar)
    logger.warning("⚠️ L2 norm_stats not found, computing from scratch")
    # ... código actual ...
```

**`src/training/environments/trading_env.py:548-575`** - Verificar lógica:
```python
def _normalize(self, feature: str, value: float) -> float:
    """Skip normalization for already-normalized features."""
    # 1. Skip _z suffix features
    if feature.endswith('_z'):
        return float(np.clip(value, -10, 10))

    # 2. Skip if stats indicate already normalized
    mean = self.norm_stats.get(feature, {}).get('mean', 0)
    std = self.norm_stats.get(feature, {}).get('std', 1)

    if abs(mean) < 0.1 and 0.8 < std < 1.2:
        logger.debug(f"Skipping normalization for {feature} (already normalized)")
        return float(np.clip(value, -10, 10))

    # 3. Apply z-score normalization
    z = (value - mean) / max(std, 1e-8)
    return float(np.clip(z, -10, 10))
```

#### 1.3 Debuggear L3 Training Terminación Temprana
**Problema**: Training termina en ~75K steps en lugar de 500K, con 0 reward.

**Diagnóstico**:
```python
# Archivo: scripts/debug_l3_training.py
import numpy as np

# Cargar evaluations.npz
data = np.load('models/ppo_v2_production/eval_logs/evaluations.npz')
print("Timesteps:", data['timesteps'])  # [25000, 50000, 75000]
print("Results shape:", data['results'].shape)  # (3, 5)
print("Episode lengths:", data['ep_lengths'])  # Muy cortos: 1-50

# Problema identificado: Episodes terminan muy rápido
# Causa probable: 15% max drawdown triggering early termination
```

**Fix propuesto en `src/training/environments/trading_env.py`**:
```python
# Línea ~380
def _check_done(self) -> bool:
    # PROBLEMA: Max drawdown 15% termina episodes muy rápido
    # con datos reales de alta volatilidad

    # FIX: Ajustar drawdown threshold o usar curriculum
    max_drawdown_threshold = 0.25  # Aumentar de 0.15 a 0.25

    if self.current_drawdown > max_drawdown_threshold:
        logger.debug(f"Episode terminated: drawdown {self.current_drawdown:.2%}")
        return True

    return False
```

---

### FASE 2: Consolidación L2 (P1) - Día 3-4

#### 2.1 Eliminar Scripts Standalone Redundantes

**Archivos a DEPRECAR** (no eliminar, mover a `scripts/deprecated/`):
- `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py`
- `data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py`

**Único path de dataset building**:
- `airflow/dags/l2_dataset_builder.py` ← ÚNICO PUNTO DE GENERACIÓN

#### 2.2 Unificar Cálculos de Indicadores

**Crear módulo compartido**: `src/features/technical_indicators.py`
```python
"""
SSOT para cálculos de indicadores técnicos.
Usado por L2 DAG y cualquier otro componente que necesite calcular features.
"""
import pandas as pd
import numpy as np

def calculate_adx_wilders(high: pd.Series, low: pd.Series, close: pd.Series,
                          period: int = 14) -> pd.Series:
    """
    ADX con Wilders smoothing y normalización porcentual.

    FIX: Usa ATR como porcentaje del precio para evitar saturación
    en pares con valores altos como USDCOP (~4000).
    """
    alpha = 1.0 / period

    high_diff = high.diff()
    low_diff = low.diff()

    # Directional movement como PORCENTAJE del precio
    plus_dm_pct = (high_diff / close).where(
        (high_diff > low_diff.abs()) & (high_diff > 0), 0.0
    )
    minus_dm_pct = (low_diff.abs() / close).where(
        (low_diff.abs() > high_diff) & (low_diff < 0), 0.0
    )

    # ATR como porcentaje
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    atr_pct = (atr / close).clip(lower=1e-6)

    # DI calculados sobre porcentajes
    plus_di = 100.0 * plus_dm_pct.ewm(alpha=alpha, adjust=False).mean() / atr_pct
    minus_di = 100.0 * minus_dm_pct.ewm(alpha=alpha, adjust=False).mean() / atr_pct

    plus_di = plus_di.clip(0, 100)
    minus_di = minus_di.clip(0, 100)

    di_sum = (plus_di + minus_di).clip(lower=1.0)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    return adx.clip(0, 100)


def calculate_rsi_wilders(close: pd.Series, period: int = 9) -> pd.Series:
    """
    RSI con Wilders smoothing (no SMA).

    Wilders usa EMA con alpha=1/period, equivalente a:
    smoothed = prev_smoothed * (period-1)/period + current/period
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()

    rs = avg_gain / avg_loss.clip(lower=1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi.clip(0, 100)


def calculate_macro_zscore(series: pd.Series, window: int = 252,
                           method: str = 'rolling') -> pd.Series:
    """
    Z-score para variables macro.

    Args:
        series: Serie temporal de valores macro
        window: Ventana para cálculo rolling (252 = ~1 año trading days)
        method: 'rolling' (recomendado) o 'fixed' (expanding desde inicio)

    IMPORTANTE: Usar 'rolling' para evitar look-ahead bias.
    'fixed' solo para casos especiales documentados.
    """
    if method == 'rolling':
        mean = series.rolling(window=window, min_periods=window//2).mean()
        std = series.rolling(window=window, min_periods=window//2).std()
    elif method == 'fixed':
        # CUIDADO: Esto puede causar look-ahead bias
        mean = series.expanding().mean()
        std = series.expanding().std()
    else:
        raise ValueError(f"method must be 'rolling' or 'fixed', got {method}")

    z = (series - mean) / std.clip(lower=1e-8)
    return z.clip(-10, 10)
```

#### 2.3 Actualizar L2 DAG para usar módulo compartido

**`airflow/dags/l2_dataset_builder.py`** - Importar y usar funciones centralizadas:
```python
# Al inicio del archivo
from src.features.technical_indicators import (
    calculate_adx_wilders,
    calculate_rsi_wilders,
    calculate_macro_zscore
)

# En build_5min_features():
def build_5min_features(ohlcv_df, macro_df):
    # ... existing code ...

    # Usar funciones centralizadas
    df['adx_14'] = calculate_adx_wilders(df['high'], df['low'], df['close'], period=14)
    df['rsi_9'] = calculate_rsi_wilders(df['close'], period=9)

    # Macro z-scores con método ROLLING consistente
    for col in ['dxy', 'vix', 'embi']:
        df[f'{col}_z'] = calculate_macro_zscore(
            df[col],
            window=252,
            method='rolling'  # SIEMPRE rolling, nunca fixed
        )
```

---

### FASE 3: Configuración Unificada (P1) - Día 5

#### 3.1 Resolver Episode Length Mismatch

**Problema identificado**:
- `airflow/dags/contracts/l3_training_contracts.py:TradingEnvConfig` → `max_episode_steps=1200`
- `src/training/config.py:EnvironmentConfig` → `max_steps=2000`

**Solución**: Unificar en SSOT

**`config/training_config.yaml`** (CREAR - SSOT para training):
```yaml
# SSOT: Training Configuration
# Todos los componentes deben leer de aquí

environment:
  max_episode_steps: 1200  # Usar valor de L3 contracts
  observation_dim: 15      # 13 market + 2 state (position, unrealized_pnl)
  action_space: 3          # HOLD, BUY, SELL

  # Early termination thresholds
  max_drawdown: 0.25       # Aumentado de 0.15 para permitir episodes más largos

training:
  total_timesteps: 500000
  eval_freq: 25000
  n_eval_episodes: 5

  # PPO hyperparameters
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.05           # CRITICAL: Prevent HOLD collapse
  vf_coef: 0.5
  max_grad_norm: 0.5

features:
  # Feature order (MUST match CTR-001 contract)
  market_features:
    - log_ret_5m
    - log_ret_1h
    - log_ret_4h
    - rsi_9
    - atr_pct
    - adx_14
    - dxy_z
    - dxy_change_1d
    - vix_z
    - embi_z
    - brent_change_1d
    - rate_spread
    - usdmxn_change_1d
  state_features:
    - position
    - unrealized_pnl
```

**Actualizar componentes para leer de SSOT**:

```python
# src/training/config.py
import yaml

def load_training_config():
    with open('config/training_config.yaml', 'r') as f:
        return yaml.safe_load(f)

class EnvironmentConfig:
    def __init__(self, **kwargs):
        ssot = load_training_config()['environment']
        self.max_steps = kwargs.get('max_steps', ssot['max_episode_steps'])
        self.max_drawdown = kwargs.get('max_drawdown', ssot['max_drawdown'])
        # ... etc
```

---

### FASE 4: L0 Data Layer Cleanup (P2) - Día 6-7

#### 4.1 Consolidar Scripts de Seeding

**Estado actual** (confuso):
- `init-scripts/04-data-seeding.py` - Seeding inicial
- `airflow/dags/l0_backup_restore.py` - Backup/restore (untracked)
- `airflow/dags/l0_macro_update.py` - Updates macro (untracked)
- Scripts eliminados: `l0_data_initialization.py`, `l0_seed_backup.py`, etc.

**Propuesta de consolidación**:

```
airflow/dags/
├── l0_data_layer.py          # DAG unificado para L0
│   ├── task: seed_if_empty   # Seeding inicial
│   ├── task: backup_weekly   # Backup semanal
│   └── task: update_macro    # Update macro diario
├── services/
│   ├── seed_service.py       # Lógica de seeding
│   ├── backup_service.py     # Lógica de backup
│   └── macro_service.py      # Lógica de macro update
```

#### 4.2 Validar Integridad de Datos

**Crear validador de datos**: `airflow/dags/validators/l0_data_validator.py`
```python
"""
Validador de integridad para L0 data layer.
Ejecutar después de cualquier operación de seeding/backup/update.
"""
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine

def validate_ohlcv_integrity(engine) -> dict:
    """Validar tabla ohlcv_5min."""
    results = {'table': 'ohlcv_5min', 'checks': []}

    # Check 1: Row count
    count = pd.read_sql("SELECT COUNT(*) as n FROM ohlcv_5min", engine).iloc[0]['n']
    results['checks'].append({
        'name': 'row_count',
        'value': count,
        'status': 'PASS' if count > 90000 else 'FAIL',
        'expected': '>90000'
    })

    # Check 2: Date range
    dates = pd.read_sql("""
        SELECT MIN(timestamp) as min_dt, MAX(timestamp) as max_dt
        FROM ohlcv_5min
    """, engine).iloc[0]
    results['checks'].append({
        'name': 'date_range',
        'min': str(dates['min_dt']),
        'max': str(dates['max_dt']),
        'status': 'PASS' if dates['max_dt'] > datetime.now() - timedelta(days=7) else 'WARN'
    })

    # Check 3: No nulls in critical columns
    nulls = pd.read_sql("""
        SELECT
            SUM(CASE WHEN close IS NULL THEN 1 ELSE 0 END) as null_close,
            SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) as null_volume
        FROM ohlcv_5min
    """, engine).iloc[0]
    results['checks'].append({
        'name': 'null_check',
        'null_close': int(nulls['null_close']),
        'null_volume': int(nulls['null_volume']),
        'status': 'PASS' if nulls['null_close'] == 0 else 'FAIL'
    })

    return results


def validate_macro_integrity(engine) -> dict:
    """Validar tabla macro_indicators."""
    results = {'table': 'macro_indicators', 'checks': []}

    # Check 1: Variables críticas presentes
    variables = pd.read_sql("""
        SELECT DISTINCT variable_name FROM macro_indicators
    """, engine)['variable_name'].tolist()

    required = ['dxy', 'vix', 'embi', 'brent', 'fed_rate', 'banrep_rate', 'usdmxn']
    missing = [v for v in required if v not in variables]

    results['checks'].append({
        'name': 'required_variables',
        'present': [v for v in required if v in variables],
        'missing': missing,
        'status': 'PASS' if not missing else 'FAIL'
    })

    return results
```

---

### FASE 5: Testing y Validación (P2) - Día 8-9

#### 5.1 Tests de Integración

**Crear**: `tests/integration/test_full_pipeline.py`
```python
"""
Test de integración end-to-end del pipeline L0→L4.
"""
import pytest
import pandas as pd
import numpy as np

class TestL2DatasetIntegrity:
    """Validar datasets generados por L2."""

    def test_adx_distribution(self):
        """ADX debe tener distribución normal, no saturada."""
        df = pd.read_parquet('data/pipeline/07_output/5min/DS_v2_fixed2.parquet')

        adx = df['adx_14']
        assert adx.mean() < 50, f"ADX mean too high: {adx.mean()}"
        assert adx.std() > 15, f"ADX std too low: {adx.std()}"
        assert adx.min() >= 0, f"ADX min negative: {adx.min()}"
        assert adx.max() <= 100, f"ADX max > 100: {adx.max()}"

    def test_no_double_normalization(self):
        """Features no deben estar sobre-normalizadas."""
        import json
        with open('data/pipeline/07_output/5min/DS_v2_fixed2_norm_stats.json') as f:
            stats = json.load(f)['features']

        # Z-score features deben tener mean≈0, std≈1
        for feat in ['dxy_z', 'vix_z', 'embi_z']:
            assert abs(stats[feat]['mean']) < 0.5, f"{feat} mean not near 0"
            assert 0.5 < stats[feat]['std'] < 2.0, f"{feat} std not near 1"

    def test_feature_count(self):
        """Dataset debe tener exactamente 13 market features."""
        df = pd.read_parquet('data/pipeline/07_output/5min/DS_v2_fixed2.parquet')

        expected_features = [
            'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
            'rsi_9', 'atr_pct', 'adx_14',
            'dxy_z', 'dxy_change_1d', 'vix_z', 'embi_z',
            'brent_change_1d', 'rate_spread', 'usdmxn_change_1d'
        ]

        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"


class TestL3TrainingConfig:
    """Validar configuración de training."""

    def test_entropy_coefficient(self):
        """ent_coef debe ser >= 0.05 para evitar HOLD collapse."""
        import yaml
        with open('config/training_config.yaml') as f:
            config = yaml.safe_load(f)

        ent_coef = config['training']['ent_coef']
        assert ent_coef >= 0.05, f"ent_coef too low: {ent_coef}"

    def test_episode_length_consistency(self):
        """Episode length debe ser consistente en todos los configs."""
        import yaml
        with open('config/training_config.yaml') as f:
            config = yaml.safe_load(f)

        expected = config['environment']['max_episode_steps']

        # Verificar L3 contract
        from airflow.dags.contracts.l3_training_contracts import TradingEnvConfig
        assert TradingEnvConfig().max_episode_steps == expected
```

#### 5.2 Script de Validación Pre-Training

**Crear**: `scripts/validate_before_training.py`
```python
#!/usr/bin/env python3
"""
Ejecutar ANTES de cualquier training para validar estado del sistema.
Exit code 0 = OK, 1 = FAIL
"""
import sys
import json
import pandas as pd
from pathlib import Path

def main():
    errors = []
    warnings = []

    # 1. Verificar dataset existe
    dataset_path = Path('data/pipeline/07_output/5min/DS_v2_fixed2.parquet')
    if not dataset_path.exists():
        errors.append(f"Dataset not found: {dataset_path}")
    else:
        df = pd.read_parquet(dataset_path)
        print(f"✓ Dataset loaded: {len(df)} rows")

        # 2. Verificar ADX no saturado
        adx_mean = df['adx_14'].mean()
        if adx_mean > 60:
            errors.append(f"ADX saturated: mean={adx_mean:.2f}")
        else:
            print(f"✓ ADX distribution OK: mean={adx_mean:.2f}")

    # 3. Verificar norm_stats existe
    norm_stats_path = dataset_path.parent / 'DS_v2_fixed2_norm_stats.json'
    if not norm_stats_path.exists():
        errors.append(f"Norm stats not found: {norm_stats_path}")
    else:
        with open(norm_stats_path) as f:
            stats = json.load(f)
        print(f"✓ Norm stats loaded: {len(stats['features'])} features")

    # 4. Verificar config de training
    config_path = Path('config/training_config.yaml')
    if not config_path.exists():
        warnings.append(f"Training config not found: {config_path}")

    # Report
    print("\n" + "="*50)
    if errors:
        print("❌ VALIDATION FAILED")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    elif warnings:
        print("⚠️ VALIDATION PASSED WITH WARNINGS")
        for w in warnings:
            print(f"  WARN: {w}")
        sys.exit(0)
    else:
        print("✅ VALIDATION PASSED")
        sys.exit(0)

if __name__ == '__main__':
    main()
```

---

## 📊 Métricas de Éxito

### Post-Implementación Fase 1
- [ ] ADX mean < 40 (actualmente ~96)
- [ ] ADX std > 15 (actualmente ~15)
- [ ] Training completa 500K steps
- [ ] Episode lengths > 100 en promedio

### Post-Implementación Fase 2
- [ ] Un solo path de dataset building
- [ ] RSI usa Wilders consistentemente
- [ ] Macro z-scores usan ROLLING consistentemente

### Post-Implementación Fase 3
- [ ] Episode length = 1200 en todos los configs
- [ ] ent_coef = 0.05 en todos los configs
- [ ] SSOT config en `config/training_config.yaml`

### Post-Implementación Fase 4-5
- [ ] Tests de integración pasan
- [ ] Validación pre-training pasa
- [ ] L0 data integrity checks pasan

---

## 🚨 Riesgos y Mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| ADX fix no se propaga en Airflow | Alta | Crítico | Forzar reserialize DAGs |
| Cambio de episode length afecta modelo existente | Media | Alto | Reentrenar desde cero |
| Consolidación L2 rompe pipelines existentes | Media | Alto | Mantener scripts antiguos en deprecated/ |
| SSOT config no es leído por todos los componentes | Alta | Medio | Audit de imports en cada componente |

---

## 📅 Timeline Estimado

```
Semana 1:
├── Día 1-2: Fase 1 (P0 fixes)
├── Día 3-4: Fase 2 (L2 consolidation)
└── Día 5: Fase 3 (Config unification)

Semana 2:
├── Día 6-7: Fase 4 (L0 cleanup)
├── Día 8-9: Fase 5 (Testing)
└── Día 10: Full pipeline test + documentation
```

---

## ✅ Checklist de Implementación

### Fase 1 - Críticos
- [ ] `airflow dags reserialize` ejecutado
- [ ] ADX verificado con mean < 40
- [ ] `engine.py` usa L2 norm_stats
- [ ] `trading_env.py` skip normalización correctamente
- [ ] `max_drawdown` aumentado a 0.25

### Fase 2 - Consolidación
- [ ] `src/features/technical_indicators.py` creado
- [ ] Scripts standalone movidos a `deprecated/`
- [ ] `l2_dataset_builder.py` usa módulo compartido

### Fase 3 - Config
- [ ] `config/training_config.yaml` creado
- [ ] Episode length = 1200 en todos lados
- [ ] ent_coef = 0.05 en todos lados

### Fase 4 - L0
- [ ] DAGs L0 consolidados
- [ ] Validadores de integridad implementados

### Fase 5 - Testing
- [ ] `tests/integration/test_full_pipeline.py` pasa
- [ ] `scripts/validate_before_training.py` pasa

---

**Autor**: Claude Code Analysis
**Revisado por**: [Pendiente]
**Aprobado por**: [Pendiente]
