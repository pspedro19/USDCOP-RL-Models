# USD/COP Trading System - Source Code Package

**Version:** 1.0.0
**Author:** Pedro @ Lean Tech Solutions
**Date:** 2025-12-16

## Overview

Este paquete consolida la funcionalidad de features del sistema de trading USD/COP, eliminando duplicación de código y estableciendo un Single Source of Truth (SSOT) para:

- Cálculo de features técnicos (RSI, ATR, ADX)
- Construcción de observaciones para el modelo PPO
- Normalización de features
- Carga centralizada de configuración

## Structure

```
src/
├── __init__.py                      # Package entry point
├── core/
│   ├── __init__.py
│   └── services/
│       ├── __init__.py
│       └── feature_builder.py      # Main feature building service (wrapper sobre feature_calculator)
└── shared/
    ├── __init__.py
    ├── config_loader.py            # Configuration loader with caching
    └── exceptions.py               # Custom exceptions
```

## Key Components

### 1. FeatureBuilder (`core/services/feature_builder.py`)

High-level wrapper para construcción de features y observaciones. Consolida ~1,200 líneas de código duplicado de 7 ubicaciones.

**Features:**
- Cálculo de indicadores técnicos (RSI, ATR, ADX)
- Construcción de observaciones 15-dim para el modelo
- Procesamiento batch de datasets
- Normalización z-score
- Validación de dimensiones

**Usage:**
```python
from src import FeatureBuilder

# Initialize
builder = FeatureBuilder()

# Calculate technical indicators
rsi = builder.calc_rsi(close, period=9)
atr_pct = builder.calc_atr_pct(high, low, close, period=10)
adx = builder.calc_adx(high, low, close, period=14)

# Build single observation (15-dim)
features_dict = {
    'log_ret_5m': 0.0002,
    'log_ret_1h': 0.0005,
    'log_ret_4h': 0.0008,
    'rsi_9': 55.0,
    'atr_pct': 0.08,
    'adx_14': 35.0,
    'dxy_z': 0.5,
    'dxy_change_1d': 0.001,
    'vix_z': -0.3,
    'embi_z': 0.2,
    'brent_change_1d': -0.02,
    'rate_spread': 1.2,
    'usdmxn_ret_1h': 0.0003
}

obs = builder.build_observation(
    features_dict,
    position=0.5,      # Current position [-1, 1]
    bar_number=30      # Bar in episode [1, 60]
)
# obs shape: (15,) = 13 features + position + time_normalized

# Batch processing
df_features = builder.build_batch(ohlcv_df, macro_df)
```

### 2. ConfigLoader (`shared/config_loader.py`)

Carga centralizada de `config/feature_config.json` con caching (singleton pattern).

**Usage:**
```python
from src import get_config

config = get_config()

# Get feature metadata
features = config.get_feature_order()  # 13-item list
obs_dim = config.get_obs_dim()         # 15
norm_stats = config.get_norm_stats('rsi_9')
clip_bounds = config.get_clip_bounds('log_ret_5m')
period = config.get_technical_period('rsi_9')

# Get trading params
trading = config.get_trading_params()
market_hours = config.get_market_hours()
```

### 3. Exceptions (`shared/exceptions.py`)

Custom exceptions para manejo de errores.

**Available:**
- `FeatureBuilderError` - Base exception
- `ConfigurationError` - Config file issues
- `NormalizationError` - Feature normalization failures
- `ValidationError` - Observation validation failures

**Usage:**
```python
from src import ValidationError

try:
    obs = builder.build_observation(features, position, bar_number)
    builder.validate_observation(obs)
except ValidationError as e:
    print(f"Validation failed: {e}")
    print(f"Expected shape: {e.details['expected_shape']}")
    print(f"Actual shape: {e.details['actual_shape']}")
```

## Critical Formula: time_normalized

**IMPORTANT:** La fórmula para `time_normalized` es:

```python
time_normalized = (bar_number - 1) / episode_length
```

Donde:
- `bar_number` ∈ [1, 60] (bars en episodio)
- `episode_length = 60` (default)
- Resultado: `time_normalized` ∈ [0, 0.983] (NO [0, 1])

**Examples:**
- Bar 1: `(1-1)/60 = 0.000`
- Bar 30: `(30-1)/60 = 0.483`
- Bar 60: `(60-1)/60 = 0.983`

Esto coincide con `environment.py:117`:
```python
time_normalized = step_count / episode_length  # step_count = 0-59
```

## Feature List (13 features)

Las 13 features en orden exacto (según `feature_config.json`):

1. `log_ret_5m` - Log return 5min
2. `log_ret_1h` - Log return 1 hour (12 bars)
3. `log_ret_4h` - Log return 4 hours (48 bars)
4. `rsi_9` - RSI period 9
5. `atr_pct` - ATR % (period 10)
6. `adx_14` - ADX period 14
7. `dxy_z` - DXY z-score
8. `dxy_change_1d` - DXY daily change
9. `vix_z` - VIX z-score
10. `embi_z` - EMBI z-score
11. `brent_change_1d` - Brent daily change
12. `rate_spread` - UST 10Y - 2Y
13. `usdmxn_ret_1h` - USDMXN 1-hour return (12 bars)

Plus 2 state variables:
14. `position` - Current position [-1, 1]
15. `time_normalized` - Time in episode [0, 0.983]

**Total observation dimension: 15**

## Compute Strategy

Features se calculan en dos ubicaciones según complejidad:

### SQL-calculated (9 features)
Calculados en `inference_features_5m` (materialized view):
- `log_ret_5m`, `log_ret_1h`, `log_ret_4h`
- `dxy_z`, `vix_z`, `embi_z`
- `dxy_change_1d`, `brent_change_1d`
- `rate_spread`

### Python-calculated (4 features)
Calculados en `FeatureBuilder` (requieren cálculos iterativos):
- `rsi_9` - Requiere rolling gain/loss
- `atr_pct` - Requiere true range calculation
- `adx_14` - Requiere DI+/DI- calculation
- `usdmxn_ret_1h` - Requiere resample complejo

## Normalization

Z-score normalization con stats fijos de training:

```python
normalized = (raw_value - mean) / std
clipped = np.clip(normalized, -4.0, 4.0)
```

Stats se leen de `feature_config.json`:
```json
{
  "name": "rsi_9",
  "norm_stats": {
    "mean": 49.27,
    "std": 23.07
  }
}
```

Features que **NO** se normalizan:
- `macro_changes` (ya están en cambios %)
- Features ya bounded (RSI, ADX están 0-100 y se normalizan)

## Testing

Run validation tests:

```bash
cd USDCOP-RL-Models
python -c "
from src import FeatureBuilder
builder = FeatureBuilder()
print(f'Version: {builder.version}')
print(f'Features: {builder.feature_order}')
print(f'Obs dim: {builder.obs_dim}')
"
```

Comprehensive test:
```bash
python scripts/test_feature_builder.py
```

## Integration Points

### 1. Training Pipeline
```python
# notebooks/pipeline entrenamiento/
from src import FeatureBuilder

builder = FeatureBuilder()
df_features = builder.build_batch(ohlcv_df, macro_df)
```

### 2. Inference DAG
```python
# airflow/dags/usdcop_m5__06_l5_realtime_inference.py
from src import FeatureBuilder

builder = FeatureBuilder()
obs = builder.build_observation(features, position, bar_number)
action, _ = model.predict(obs, deterministic=True)
```

### 3. Realtime API
```python
# services/trading_api_realtime.py
from src import FeatureBuilder

builder = FeatureBuilder()
# ... calculate features from market data
obs = builder.build_observation(features, position, bar_number)
```

## Migration from Legacy Code

Este módulo **reemplaza**:

1. `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py` (lines 217-303)
2. `data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py` (feature functions)
3. `notebooks/pipeline entrenamiento/src/utils.py` (normalize functions)
4. Inline feature calculation en DAGs
5. Inline feature calculation en services

**IMPORTANT:** `services/feature_calculator.py` (380 líneas) aún existe y es usado internamente por `FeatureBuilder`. NO eliminar.

## File Sizes

- `feature_builder.py`: ~300 lines (wrapper)
- `config_loader.py`: ~200 lines
- `exceptions.py`: ~60 lines

**Total:** ~560 lines consolidadas vs ~1,200 lines duplicadas (reducción 53%)

## References

- ARQUITECTURA_INTEGRAL_V3.md (Sections 11.0.1, 11.0.3, 12.2)
- MAPEO_MIGRACION_BIDIRECCIONAL.md (Parts 1-4)
- config/feature_config.json (v3.1.0)

## Changelog

### v1.0.0 (2025-12-16)
- Initial release
- Consolidated FeatureBuilder from 7 locations
- Added ConfigLoader with caching
- Custom exceptions
- Comprehensive validation tests
- time_normalized formula corrected to (bar_number - 1) / 60
