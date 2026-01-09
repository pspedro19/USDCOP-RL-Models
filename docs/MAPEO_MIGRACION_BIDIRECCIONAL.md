# MAPEO BIDIRECCIONAL DE MIGRACION
## USD/COP Trading System: Estado Actual <-> Arquitectura Propuesta V3
### Fecha: 2025-12-16

---

## RESUMEN EJECUTIVO

| Metrica | Estado Actual | Propuesto | Delta |
|---------|---------------|-----------|-------|
| Archivos Python | 58 | ~25 | -57% |
| Lineas de codigo (features) | ~3,200 | ~800 | -75% |
| Codigo duplicado | ~35% | <5% | -30% |
| Archivos de config | 3 | 5 | +2 |
| Funciones de features duplicadas | 7 ubicaciones | 1 (SSOT) | -86% |
| DAGs Airflow | 14 | 5 | -64% |
| Tablas DB documentadas | 12 | 15 | +3 |

### Estado de Produccion

| Componente | Estado | Accion |
|------------|--------|--------|
| Pipeline L0 (OHLCV) | OK | Reusar |
| Pipeline L1-L4 | Redundante | Deprecar |
| DAG Inferencia L5 | ROTO (19 vs 15 features) | Reescribir |
| Training Pipeline | OK | Mantener como SSOT |
| Scrapers Macro | OK | Adaptar para DB |
| feature_config.json | OK pero no usado | Implementar lectura |

---

## ERRATA Y CORRECCIONES (2025-12-16)

> **Nota**: Esta sección documenta las correcciones realizadas después de la verificación profunda con agentes múltiples.

### Archivos Corregidos

| Archivo | Corrección | Estado |
|---------|-----------|--------|
| `config/feature_config.json` | usdmxn_ret_1h: periods 1→12, clip [-0.05,0.05]→[-0.1,0.1] | ✅ CORREGIDO |
| `config/feature_config.json` | Añadido `compute_strategy` section | ✅ AÑADIDO |
| `init-scripts/12-unified-inference-schema.sql` | ELIMINADOS hour_sin/hour_cos (líneas 174-175) | ✅ CORREGIDO |
| `init-scripts/12-unified-inference-schema.sql` | usdmxn_ret_1h clip: -0.05→-0.10 | ✅ CORREGIDO |
| `init-scripts/12-unified-inference-schema.sql` | Añadido header documentando feature split SQL/Python | ✅ AÑADIDO |

### Hallazgos de Verificación

| Issue | Verificación | Resultado |
|-------|-------------|-----------|
| hour_sin/hour_cos | Buscado en 15+ archivos | LEGACY - Debe eliminarse de SQL view |
| usdmxn_ret_1h formula | Comparado config vs código training | CONFIG INCORRECTO - periods=12 en código, periods=1 en config |
| atr_pct period | Verificado en 5 archivos de dataset builder | CORRECTO - Todos usan period=10 |
| RSI/ATR/ADX en SQL | Análisis de complejidad | ATR factible, RSI moderado, ADX complejo → Mantener en Python |
| time_normalized | Verificado en environment.py:117 | `step_count/episode_length` (0.0-0.983), NO `(bar-1)/59` |
| Número de DAGs | Contados en airflow/dags/ | **14 actuales** → 5 propuestos (-64%) |

### Feature Computation Strategy (Verificado)

```
┌─────────────────────────────────────────────────────────────────┐
│ CALCULADOS EN SQL (9 features):                                 │
│   log_ret_5m, log_ret_1h, log_ret_4h, dxy_z, vix_z, embi_z,    │
│   dxy_change_1d, brent_change_1d, rate_spread                   │
├─────────────────────────────────────────────────────────────────┤
│ CALCULADOS EN PYTHON SERVICE (4 features):                      │
│   rsi_9, atr_pct, adx_14, usdmxn_ret_1h                        │
├─────────────────────────────────────────────────────────────────┤
│ ELIMINADOS EN V14 (8 features):                                 │
│   hour_sin, hour_cos, bb_position, dxy_mom_5d, vix_regime,      │
│   brent_vol_5d, sma_ratio, macd_hist                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## PARTE 1: MAPEO FORWARD (Actual -> Propuesto)

### 1.1 Data Pipeline - Funciones de Features

#### data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py

| Funcion Actual | Lineas | -> Destino Propuesto | Estado |
|----------------|--------|---------------------|--------|
| `calc_log_return()` | 217-219 | `src/core/services/feature_builder.py` | Consolidar |
| `calc_rsi()` | 221-227 | `src/core/services/feature_builder.py` | Consolidar |
| `calc_atr()` | 229-235 | `src/core/services/feature_builder.py` | Consolidar |
| `calc_atr_pct()` | 237-240 | `src/core/services/feature_builder.py` | Consolidar |
| `calc_adx()` | 242-261 | `src/core/services/feature_builder.py` | Consolidar |
| `calc_bollinger_position()` | 263-270 | `src/core/services/feature_builder.py` | ELIMINAR (no en v14) |
| `calc_macd_histogram()` | 272-278 | `src/core/services/feature_builder.py` | ELIMINAR (no en v14) |
| `calc_sma_ratio()` | 280-283 | `src/core/services/feature_builder.py` | ELIMINAR (no en v14) |
| `z_score_rolling()` | 285-290 | `src/core/services/feature_builder.py` | Consolidar |
| `pct_change_safe()` | 292-294 | `src/core/services/feature_builder.py` | Consolidar |
| `rolling_volatility()` | 296-299 | `src/core/services/feature_builder.py` | ELIMINAR (no en v14) |
| `calc_momentum()` | 301-303 | `src/core/services/feature_builder.py` | ELIMINAR (no en v14) |
| `encode_cyclical()` | 313-317 | ELIMINAR | hour_sin/cos no en v14 |

**Codigo actual (referencia):**
```python
# data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py:217-219
def calc_log_return(series, periods=1):
    """Log returns"""
    return np.log(series / series.shift(periods))

# lineas 221-227
def calc_rsi(close, period=14):
    """RSI - Relative Strength Index (0-100)"""
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

# lineas 242-261
def calc_adx(high, low, close, period=14):
    """ADX - Average Directional Index (0-100)"""
    # ... 20 lineas de codigo
```

**Codigo propuesto (consolidado):**
```python
# src/core/services/feature_builder.py
class FeatureBuilder:
    def __init__(self, config_path: str = 'config/feature_config.json'):
        self.config = self._load_config(config_path)
        self.norm_stats = self.config['features']

    def _calc_log_return(self, series: pd.Series, periods: int) -> pd.Series:
        return np.log(series / series.shift(periods))

    def _calc_rsi(self, close: pd.Series, period: int = 9) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
```

---

### 1.2 Training Pipeline

#### notebooks/pipeline entrenamiento/config/settings.py

| Componente | Lineas | -> Destino Propuesto | Cambios |
|------------|--------|---------------------|---------|
| `FEATURES_FOR_MODEL` | 33-40 | `config/feature_config.json:observation_space.order` | Sin cambios logicos |
| `DATA_PATH` | 16 | `config/feature_config.json:sources` | Path relativo |
| `RAW_RETURN_COL` | 43 | `config/feature_config.json:features.returns.raw_return_column` | Ya existe |
| `COST_PER_TRADE` | 58 | `config/feature_config.json:trading.cost_per_trade` | Ya existe |
| `WEAK_SIGNAL_THRESHOLD` | 66 | `config/feature_config.json:trading.weak_signal_threshold` | Ya existe |
| `TRADE_COUNT_THRESHOLD` | 69 | `config/feature_config.json:trading.trade_count_threshold` | Ya existe |
| `MIN_COST_THRESHOLD` | 72 | `config/feature_config.json:trading.min_cost_threshold` | Agregar |
| `BARS_PER_DAY` | 94 | `config/feature_config.json:trading.bars_per_session` | Ya existe |
| `PPO_CONFIG` | 104-119 | `config/feature_config.json:model.hyperparameters` | Ya existe |
| `WALK_FORWARD_FOLDS` | 125-161 | Mantener en settings.py | No migrar |

**Valores criticos a preservar:**
```python
# notebooks/pipeline entrenamiento/config/settings.py
FEATURES_FOR_MODEL = [
    'log_ret_5m', 'log_ret_1h', 'log_ret_4h',  # 3 returns
    'rsi_9', 'atr_pct', 'adx_14',               # 3 technical
    'dxy_z', 'dxy_change_1d',                   # 2 dxy
    'vix_z', 'embi_z',                          # 2 risk
    'brent_change_1d',                          # 1 commodity
    'rate_spread', 'usdmxn_ret_1h',             # 2 macro
]  # TOTAL: 13 features

# Observation: 13 + position(1) + time_normalized(1) = 15
```

#### notebooks/pipeline entrenamiento/src/utils.py

| Funcion | Lineas | -> Destino | Estado |
|---------|--------|------------|--------|
| `calculate_norm_stats()` | 13-41 | `src/core/services/feature_builder.py` | Consolidar |
| `normalize_df_v11()` | 44-90 | `src/core/services/feature_builder.py` | Consolidar |
| `analyze_regime_raw()` | 93-135 | `src/core/services/analysis.py` | Mantener |
| `load_and_prepare_data()` | 138-172 | `src/core/services/data_loader.py` | Refactorizar |
| `calculate_wfe()` | 175-205 | `src/core/services/metrics.py` | Mantener |
| `classify_wfe()` | 208-233 | `src/core/services/metrics.py` | Mantener |

#### notebooks/pipeline entrenamiento/src/environment.py

| Componente | Lineas | -> Destino | Estado |
|------------|--------|------------|--------|
| `TradingEnvV11` class | 32-213 | `training/pipeline/src/environment.py` | Mantener in-place |
| `__init__()` | 55-89 | - | Mantener |
| `reset()` | 91-106 | - | Mantener |
| `_obs()` | 108-120 | - | Mantener |
| `_get_raw_return()` | 122-146 | - | Mantener |
| `step()` | 148-213 | - | Mantener |

**Thresholds importados de settings.py (lineas 16-29):**
```python
# Si import falla, usa fallbacks:
WEAK_SIGNAL_THRESHOLD = 0.3
TRADE_COUNT_THRESHOLD = 0.3
MIN_COST_THRESHOLD = 0.001
```

---

### 1.3 DAGs de Airflow

#### airflow/dags/ (19 archivos) -> airflow/dags/ (5 archivos)

| DAG Actual | Lineas | Estado | -> Destino | Accion |
|------------|--------|--------|------------|--------|
| `usdcop_m5__01_l0_intelligent_acquire.py` | ~400 | OK | `l0_ohlcv_realtime.py` | Reusar |
| `usdcop_m5__00b_l0_macro_scraping.py` | ~200 | Duplicado | `l0_macro_daily_update.py` | Consolidar |
| `usdcop_m5__01b_l0_macro_acquire.py` | ~300 | Duplicado | `l0_macro_daily_update.py` | Consolidar |
| `usdcop_m5__02_l1_standardize.py` | ~250 | Redundante | ELIMINAR | Deprecar |
| `usdcop_m5__03_l2_prepare.py` | ~300 | Redundante | ELIMINAR | Deprecar |
| `usdcop_m5__04_l3_feature.py` | ~350 | Redundante | ELIMINAR | Deprecar |
| `usdcop_m5__04b_l3_llm_features.py` | ~200 | Experimental | ELIMINAR | Deprecar |
| `usdcop_m5__05_l4_rlready.py` | ~300 | Redundante | ELIMINAR | Deprecar |
| `usdcop_m5__05a_l5_rl_training.py` | ~400 | Offline | Mantener offline | No produccion |
| `usdcop_m5__05b_l5_ml_training.py` | ~350 | Offline | Mantener offline | No produccion |
| `usdcop_m5__05c_l5_llm_setup.py` | ~200 | Experimental | ELIMINAR | Deprecar |
| `usdcop_m5__06_l5_realtime_inference.py` | 846 | **ROTO** | `l5_realtime_inference.py` | **REESCRIBIR** |
| `usdcop_m5__07_l6_backtest_multi_strategy.py` | ~500 | Offline | Mantener offline | No produccion |
| `usdcop_m5__99_alert_monitor.py` | ~150 | OK | `alert_monitor.py` | Mantener |
| `l2_causal_deseasonalization.py` | ~200 | Experimental | ELIMINAR | Deprecar |
| `base_pipeline.py` | ~100 | Utility | Mantener | Refactorizar |
| `base_yaml_dag.py` | ~150 | Utility | Mantener | Refactorizar |
| `dag_factory.py` | ~200 | Utility | Mantener | Refactorizar |
| `pipeline_integration_config.py` | ~100 | Config | Mantener | Refactorizar |

**DAG de inferencia ROTO - Problema critico:**
```python
# airflow/dags/usdcop_m5__06_l5_realtime_inference.py:51-61
FEATURES_CONFIG = {
    'features': [
        'log_ret_5m', 'log_ret_1h', 'log_ret_4h',
        'rsi_9', 'atr_pct', 'adx_14', 'bb_position',      # <- bb_position NO en v14
        'dxy_z', 'dxy_change_1d', 'dxy_mom_5d',           # <- dxy_mom_5d NO en v14
        'vix_z', 'vix_regime', 'embi_z',                  # <- vix_regime NO en v14
        'brent_change_1d', 'brent_vol_5d',                # <- brent_vol_5d NO en v14
        'rate_spread', 'usdmxn_ret_1h',
        'hour_sin', 'hour_cos'                            # <- hour_sin/cos NO en v14
    ],
    'obs_dim': 20,  # INCORRECTO! Modelo espera 15
}
# vs modelo v11 que espera 13 features + 2 state = 15 obs_dim
```

---

### 1.4 Services

#### services/ (6 archivos)

| Servicio | Lineas | Estado | -> Destino | Accion |
|----------|--------|--------|------------|--------|
| `trading_api_realtime.py` | ~800 | OK | Mantener | Agregar lectura SSOT |
| `realtime_market_ingestion_v2.py` | ~500 | OK | Mantener | - |
| `pipeline_data_api.py` | ~400 | OK | Mantener | - |
| `bi_api.py` | ~300 | OK | Mantener | - |
| `trading_analytics_api.py` | ~350 | OK | Mantener | - |
| `multi_model_trading_api.py` | ~250 | Experimental | Evaluar | - |

---

## PARTE 2: MAPEO REVERSE (Propuesto -> Actual)

### 2.1 Nuevos Modulos Core

#### src/core/services/feature_builder.py (PROPUESTO)

| Metodo Propuesto | Origen en Codigo Actual | Archivo:Linea |
|------------------|-------------------------|---------------|
| `FeatureBuilder.__init__()` | **NUEVO** | - |
| `FeatureBuilder._load_config()` | **NUEVO** - Lee feature_config.json | - |
| `FeatureBuilder._calc_log_return()` | `calc_log_return()` | 06_rl_dataset_builder/01_build_5min_datasets.py:217 |
| `FeatureBuilder._calc_rsi()` | `calc_rsi()` | 06_rl_dataset_builder/01_build_5min_datasets.py:221 |
| `FeatureBuilder._calc_atr()` | `calc_atr()` | 06_rl_dataset_builder/01_build_5min_datasets.py:229 |
| `FeatureBuilder._calc_atr_pct()` | `calc_atr_pct()` | 06_rl_dataset_builder/01_build_5min_datasets.py:237 |
| `FeatureBuilder._calc_adx()` | `calc_adx()` | 06_rl_dataset_builder/01_build_5min_datasets.py:242 |
| `FeatureBuilder._z_score()` | `z_score_rolling()` | 06_rl_dataset_builder/01_build_5min_datasets.py:285 |
| `FeatureBuilder._pct_change()` | `pct_change_safe()` | 06_rl_dataset_builder/01_build_5min_datasets.py:292 |
| `FeatureBuilder.normalize()` | `normalize_df_v11()` | notebooks/.../src/utils.py:44 |
| `FeatureBuilder.calculate_norm_stats()` | `calculate_norm_stats()` | notebooks/.../src/utils.py:13 |
| `FeatureBuilder.build_observation()` | Inline en DAG | usdcop_m5__06_l5_realtime_inference.py:334 |
| `FeatureBuilder.build_batch()` | `create_dataset()` | 06_rl_dataset_builder/01_build_5min_datasets.py:918 |

**Consolidacion de 7 ubicaciones:**
```
Este modulo consolida codigo de:
+-- data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py (funciones features)
+-- data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py (duplicado)
+-- data/pipeline/03_processing/scripts/03_create_rl_datasets.py (duplicado)
+-- notebooks/pipeline entrenamiento/src/utils.py (normalizacion)
+-- airflow/dags/usdcop_m5__06_l5_realtime_inference.py (inline)
+-- services/trading_api_realtime.py (inline)
+-- data/archive/PASS/03_create_rl_datasets.py (backup)
```

---

#### config/feature_config.json (EXISTENTE - No implementado)

| Campo Propuesto | Origen Actual | Archivo:Linea | Valor |
|-----------------|---------------|---------------|-------|
| `observation_space.dimension` | `obs_dim` | environment.py:79 | 15 |
| `observation_space.order` | `FEATURES_FOR_MODEL` | settings.py:33-40 | 13 features |
| `observation_space.additional_in_env` | Hardcoded | environment.py:115-117 | [position, time_normalized] |
| `features.returns.*.norm_stats` | `NORM_STATS` | dag_inference.py:72-92 | mean/std |
| `features.technical.*.norm_stats` | `NORM_STATS` | dag_inference.py:72-92 | mean/std |
| `features.macro_zscore.*.norm_stats` | Hardcoded | dag_inference.py:80-85 | Fixed params |
| `trading.cost_per_trade` | `COST_PER_TRADE` | settings.py:58 | 0.0015 |
| `trading.weak_signal_threshold` | `WEAK_SIGNAL_THRESHOLD` | settings.py:66 | 0.3 |
| `trading.episode_length` | `CONFIG['episode_length']` | settings.py:87 | 60 |
| `trading.market_hours` | Hardcoded en DAGs | dag_inference.py:67-68 | 8:00-12:55 COT |
| `model.path` | `MODEL_CONFIG['model_path']` | dag_inference.py:43 | models/ppo_usdcop_v14_fold0.zip |
| `model.hyperparameters` | `PPO_CONFIG` | settings.py:104-119 | dict |

**Archivo feature_config.json ya existe pero NO se lee en codigo:**
```python
# Lo que deberia hacer el codigo:
with open('config/feature_config.json') as f:
    config = json.load(f)
FEATURES = config['observation_space']['order']  # 13 features
OBS_DIM = config['observation_space']['total_obs_dim']  # 15

# Lo que hace actualmente el DAG de inferencia:
FEATURES_CONFIG = {
    'features': [...],  # 19 features HARDCODED (INCORRECTO)
    'obs_dim': 20,       # HARDCODED (INCORRECTO)
}
```

---

### 2.2 Tablas de Base de Datos

#### Propuestas en Seccion 21 de ARQUITECTURA_INTEGRAL_V3.md

| Tabla Propuesta | Existe | Origen/Cambios |
|-----------------|--------|----------------|
| `usdcop_m5_ohlcv` | SI | Ya existe, reusar |
| `macro_indicators_daily` | NO | **CREAR** - 37 columnas del diccionario |
| `inference_features_5m` | NO | **CREAR** - 13 features transformadas |
| `dw.fact_rl_inference` | SI | Ya existe |
| `dw.fact_agent_actions` | SI | Ya existe |
| `dw.fact_equity_curve_realtime` | SI | Ya existe |

---

## PARTE 3: TABLA DE FEATURES (CRITICO)

### 3.1 Features del Modelo v11 (15 dimensiones)

| # | Feature | Formula | Archivo Origen | Linea | Norm Stats |
|---|---------|---------|----------------|-------|------------|
| 0 | `log_ret_5m` | `log(close/close[-1])` | 01_build_5min_datasets.py | 329 | mean=2e-06, std=0.001138 |
| 1 | `log_ret_1h` | `log(close/close[-12])` | 01_build_5min_datasets.py | 331 | mean=2.3e-05, std=0.003776 |
| 2 | `log_ret_4h` | `log(close/close[-48])` | 01_build_5min_datasets.py | 332 | mean=5.2e-05, std=0.007768 |
| 3 | `rsi_9` | `RSI(close, 9)` | 01_build_5min_datasets.py | 343 | mean=49.27, std=23.07 |
| 4 | `atr_pct` | `ATR(10)/close*100` | 01_build_5min_datasets.py | 345 | mean=0.062, std=0.0446 |
| 5 | `adx_14` | `ADX(h,l,c,14)` | 01_build_5min_datasets.py | 346 | mean=32.01, std=16.36 |
| 6 | `dxy_z` | `(dxy-103)/5` | 01_build_5min_datasets.py | 387 | Fixed: mean=103, std=5 |
| 7 | `dxy_change_1d` | `pct_change(dxy,288)` | 01_build_5min_datasets.py | 388 | clip=[-0.03, 0.03] |
| 8 | `vix_z` | `(vix-20)/10` | 01_build_5min_datasets.py | 397 | Fixed: mean=20, std=10 |
| 9 | `embi_z` | `(embi-300)/100` | 01_build_5min_datasets.py | 418 | Fixed: mean=300, std=100 |
| 10 | `brent_change_1d` | `pct_change(brent,288)` | 01_build_5min_datasets.py | 410 | clip=[-0.10, 0.10] |
| 11 | `rate_spread` | `treasury_10y - treasury_2y` | 01_build_5min_datasets.py | 426 | mean=-0.0326, std=1.400 |
| 12 | `usdmxn_ret_1h` | `pct_change(usdmxn,12)` | 01_build_5min_datasets.py | 441 | clip=[-0.1, 0.1] |
| 13 | `position` | Estado del env | environment.py | 115 | [-1, 1] |
| 14 | `time_normalized` | `step_count/60` | environment.py | 117 | [0, 0.983] |

### 3.2 Features ELIMINADOS en v14 (NO migrar)

| Feature | Razon | Archivo donde estaba | Linea |
|---------|-------|----------------------|-------|
| `bb_position` | Redundante con rsi_9 | 01_build_5min_datasets.py | 347 |
| `dxy_mom_5d` | Redundante con dxy_change_1d | 01_build_5min_datasets.py | 389 |
| `vix_regime` | Redundante con vix_z | 01_build_5min_datasets.py | 399-403 |
| `brent_vol_5d` | Correlacionado con atr_pct | 01_build_5min_datasets.py | 411 |
| `hour_sin` | Bajo valor predictivo | 01_build_5min_datasets.py | 665 |
| `hour_cos` | Bajo valor predictivo | 01_build_5min_datasets.py | 665 |
| `sma_ratio` | No seleccionado | 01_build_5min_datasets.py | 348 |
| `macd_hist` | No seleccionado | 01_build_5min_datasets.py | 349 |

---

## PARTE 4: CODIGO DUPLICADO IDENTIFICADO

### 4.1 Funciones de Features (CRITICO)

| Funcion | Ubicacion 1 | Ubicacion 2 | Ubicacion 3 | Consolidar en |
|---------|-------------|-------------|-------------|---------------|
| `calc_log_return()` | 06_builder/01_build_5min.py:217 | 06_builder/02_build_daily.py:153 | 03_processing/03_create_rl.py:206 | feature_builder.py |
| `calc_rsi()` | 06_builder/01_build_5min.py:221 | 06_builder/02_build_daily.py:158 | 03_processing/03_create_rl.py:211 | feature_builder.py |
| `calc_atr()` | 06_builder/01_build_5min.py:229 | 06_builder/02_build_daily.py:166 | dag_inference.py:264 | feature_builder.py |
| `calc_atr_pct()` | 06_builder/01_build_5min.py:237 | 06_builder/02_build_daily.py:174 | - | feature_builder.py |
| `calc_adx()` | 06_builder/01_build_5min.py:242 | 06_builder/02_build_daily.py:179 | 03_processing/03_create_rl.py:225 | feature_builder.py |
| `z_score_rolling()` | 06_builder/01_build_5min.py:285 | 06_builder/02_build_daily.py:222 | dag_inference.py:350 (inline) | feature_builder.py |
| `normalize_features()` | utils.py:44 | dag_inference.py:334 | trading_api_realtime.py:~200 | feature_builder.py |

**Lineas duplicadas estimadas: ~1,200 lineas**

### 4.2 Conexiones a Base de Datos

| Patron | Ubicaciones | Consolidar |
|--------|-------------|------------|
| `psycopg2.connect()` | dag_inference.py:107-116, services/*.py | `db_connection.py` |
| `get_db_connection()` | 5+ archivos | Singleton pattern |
| Credenciales hardcoded | Multiple | Environment variables |

### 4.3 Logica de Horario de Trading

| Patron | Ubicaciones | Valor |
|--------|-------------|-------|
| `market_hours_start = 8` | dag_inference.py:67, services/* | 8 COT |
| `market_hours_end = 13` | dag_inference.py:68, services/* | 13 COT |
| `utc_to_cot()` | dag_inference.py:119-124 | UTC-5 |
| `get_bar_number()` | dag_inference.py:127-132 | 1-60 |

---

## PARTE 5: VALORES HARDCODEADOS A MIGRAR

### 5.1 Trading Parameters

| Valor | Archivo Actual | Linea | -> Config Propuesto | Urgencia |
|-------|----------------|-------|---------------------|----------|
| `0.0015` | settings.py | 58 | feature_config.json:trading.cost_per_trade | P0 |
| `60` | settings.py | 87 | feature_config.json:trading.bars_per_session | P0 |
| `0.3` | settings.py | 66 | feature_config.json:trading.weak_signal_threshold | P0 |
| `0.3` | settings.py | 69 | feature_config.json:trading.trade_count_threshold | P1 |
| `0.001` | settings.py | 72 | feature_config.json:trading.min_cost_threshold | P1 |
| `10000.0` | dag_inference.py | 64 | feature_config.json:trading.initial_equity | P2 |

### 5.2 Normalization Stats (CRITICO)

| Parametro | Valor | Archivo | Linea | Urgencia |
|-----------|-------|---------|-------|----------|
| `log_ret_5m.mean` | 2e-06 | dag_inference.py | 73 | P0 |
| `log_ret_5m.std` | 0.001138 | dag_inference.py | 73 | P0 |
| `rsi_9.mean` | 49.27 | dag_inference.py | 76 | P0 |
| `rsi_9.std` | 23.07 | dag_inference.py | 76 | P0 |
| `dxy_z.mean` | 103.0 | dag_inference.py | 80 | P0 |
| `dxy_z.std` | 5.0 | dag_inference.py | 80 | P0 |
| `vix_z.mean` | 20.0 | dag_inference.py | 83 | P0 |
| `vix_z.std` | 10.0 | dag_inference.py | 83 | P0 |
| `embi_z.mean` | 300.0 | dag_inference.py | 85 | P0 |
| `embi_z.std` | 100.0 | dag_inference.py | 85 | P0 |

**Estos valores YA estan en feature_config.json pero el DAG NO lo lee!**

### 5.3 Paths Absolutos a Eliminar

| Path | Archivo | Linea | Solucion |
|------|---------|-------|----------|
| `C:\Users\pedro\...` | settings.py | 16 | Path relativo o env var |
| `/opt/airflow/models/` | dag_inference.py | 43 | Config file |
| `C:\Users\pedro\OneDrive\...` | 01_build_5min_datasets.py | 64-68 | Path relativo |

---

## PARTE 6: ARCHIVOS A CREAR/MODIFICAR/ELIMINAR

### CREAR (no existen hoy)

| Archivo | Proposito | Prioridad | Lineas Est. |
|---------|-----------|-----------|-------------|
| `src/core/services/feature_builder.py` | Consolidacion de features | P0 | ~400 |
| `src/core/services/data_loader.py` | Carga de datos unificada | P1 | ~150 |
| `src/shared/config_loader.py` | Lectura de feature_config.json | P0 | ~50 |
| `src/shared/trading_calendar.py` | Horarios y festivos | P1 | ~100 |
| `src/shared/db_connection.py` | Conexion DB singleton | P1 | ~80 |
| `airflow/dags/l0_macro_daily_update.py` | DAG consolidado macro | P1 | ~200 |
| `airflow/dags/l1_feature_transform.py` | Transformacion features | P2 | ~150 |
| `config/trading_calendar.json` | Festivos Colombia 2025 | P1 | ~50 |
| `init-scripts/02-macro-data-schema.sql` | Schema macro_indicators_daily | P0 | ~100 |
| `init-scripts/03-inference-features-schema.sql` | Schema inference_features_5m | P0 | ~80 |
| `tests/unit/test_feature_builder.py` | Tests unitarios | P2 | ~200 |

### MODIFICAR (requieren cambios)

| Archivo | Cambio Requerido | Prioridad |
|---------|------------------|-----------|
| `airflow/dags/usdcop_m5__06_l5_realtime_inference.py` | Leer feature_config.json, 13 features | P0 |
| `services/trading_api_realtime.py` | Importar FeatureBuilder | P1 |
| `config/feature_config.json` | Agregar min_cost_threshold | P1 |
| `docker-compose.yml` | Agregar volumen para config/ | P2 |

### ELIMINAR/DEPRECAR

| Archivo | Razon | Backup? |
|---------|-------|---------|
| `airflow/dags/usdcop_m5__02_l1_standardize.py` | Redundante con L0 | No |
| `airflow/dags/usdcop_m5__03_l2_prepare.py` | Redundante | No |
| `airflow/dags/usdcop_m5__04_l3_feature.py` | Redundante | No |
| `airflow/dags/usdcop_m5__04b_l3_llm_features.py` | Experimental | No |
| `airflow/dags/usdcop_m5__05_l4_rlready.py` | Redundante | No |
| `airflow/dags/usdcop_m5__05c_l5_llm_setup.py` | Experimental | No |
| `airflow/dags/l2_causal_deseasonalization.py` | Experimental | No |
| `data/pipeline/02_update_scrapers/` | Duplicado de 02_scrapers/ | No |
| `data/pipeline/03_processing/scripts/03b_*.py` | Duplicado | Si |

---

## PARTE 7: CHECKLIST DE ROLLBACK

### Si falla: FeatureBuilder nuevo

```bash
# Revertir a funciones originales dispersas
git checkout HEAD~1 -- data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py
git checkout HEAD~1 -- notebooks/pipeline\ entrenamiento/src/utils.py

# Verificar que entrenamiento sigue funcionando
cd "notebooks/pipeline entrenamiento"
python run.py --quick-test
```

### Si falla: DAG de inferencia corregido

```bash
# Reactivar DAG antiguo (con 19 features - SEGUIRA FALLANDO pero al menos corre)
airflow dags unpause usdcop_m5__06_l5_realtime_inference_old
airflow dags pause usdcop_m5__06_l5_realtime_inference

# Verificar logs
airflow tasks logs usdcop_m5__06_l5_realtime_inference_old run_inference $(date +%Y-%m-%d)

# NOTA: El modelo seguira fallando con dimension mismatch
# Solo usar para debugging
```

### Si falla: feature_config.json lectura

```bash
# El training pipeline tiene valores hardcoded como fallback
# Solo afecta inferencia, no entrenamiento
# Revertir cambios:
git checkout HEAD~1 -- airflow/dags/usdcop_m5__06_l5_realtime_inference.py

# El DAG volvera a usar NORM_STATS hardcoded (19 features, ROTO)
```

### Si falla: Nuevas tablas SQL

```bash
# Rollback de tablas
psql -h localhost -U admin -d usdcop_trading << EOF
DROP TABLE IF EXISTS inference_features_5m CASCADE;
DROP TABLE IF EXISTS macro_indicators_daily CASCADE;
EOF

# La inferencia seguira usando fact_macro_realtime (si existe)
```

---

## PARTE 8: VALIDACION POST-MIGRACION

### Test 1: Consistencia de Features

```python
# Verificar que FeatureBuilder produce mismos valores que codigo legacy
import pandas as pd
import numpy as np

# Cargar datos de referencia (legacy)
df_legacy = pd.read_csv('data/pipeline/07_output/RL_DS3_MACRO_CORE.csv')

# Calcular con FeatureBuilder nuevo
from src.core.services.feature_builder import FeatureBuilder
fb = FeatureBuilder()
df_new = fb.build_batch(df_raw)

# Comparar
for col in ['log_ret_5m', 'rsi_9', 'atr_pct', 'adx_14']:
    diff = (df_legacy[col] - df_new[col]).abs().max()
    assert diff < 1e-6, f"Mismatch en {col}: {diff}"
    print(f"{col}: OK (diff max = {diff:.2e})")
```

### Test 2: Dimensiones del Modelo

```python
# Verificar observation space
from stable_baselines3 import PPO
import json

model = PPO.load("models/ppo_usdcop_v14_fold0.zip")
with open('config/feature_config.json') as f:
    config = json.load(f)

expected_dim = config['observation_space']['total_obs_dim']
actual_dim = model.observation_space.shape[0]

assert actual_dim == expected_dim == 15, \
    f"Dimension mismatch! Model: {actual_dim}, Config: {expected_dim}, Expected: 15"
print(f"Observation space: {actual_dim} features (OK)")
```

### Test 3: Inferencia End-to-End

```python
# Comparar senal del servicio corregido vs legacy
import requests
import json

# Datos de prueba
test_data = {
    'close': 4200.50,
    'log_ret_5m': 0.0002,
    # ... otros features
}

# Legacy (si aun corre)
try:
    r_legacy = requests.post('http://localhost:8001/predict', json=test_data)
    action_legacy = r_legacy.json()['action']
except:
    action_legacy = None

# Nuevo
r_new = requests.post('http://localhost:8002/predict', json=test_data)
action_new = r_new.json()['action']

if action_legacy:
    print(f"Legacy: {action_legacy:.4f}")
print(f"Nuevo: {action_new:.4f}")
```

---

## APENDICE A: ESTRUCTURA DE CARPETAS

### Actual (58 archivos .py relevantes)
```
USDCOP-RL-Models/
+-- airflow/
|   +-- dags/                         # 19 DAGs (muchos redundantes)
+-- data/
|   +-- pipeline/
|       +-- 00_config/               # Configuracion
|       +-- 01_fusion/               # 1 script
|       +-- 02_scrapers/             # 12 scripts
|       +-- 02_update_scrapers/      # DUPLICADO de 02_scrapers
|       +-- 03_processing/           # 4 scripts
|       +-- 04_cleaning/             # 1 script
|       +-- 05_resampling/           # 1 script
|       +-- 06_rl_dataset_builder/   # 3 scripts (features aqui)
|       +-- 07_output/               # CSVs generados
+-- notebooks/
|   +-- pipeline entrenamiento/      # SSOT para training
|       +-- config/                  # settings.py
|       +-- src/                     # environment.py, utils.py
|       +-- models/                  # .zip models
+-- services/                        # 6 APIs
+-- config/
|   +-- feature_config.json          # SSOT (NO USADO!)
+-- models/                          # Models de produccion
```

### Propuesto (~25 archivos .py)
```
USDCOP-RL-Models/
+-- config/                          # SSOT centralizado
|   +-- feature_config.json          # Features, norm_stats, trading params
|   +-- trading_calendar.json        # Horarios, festivos
|   +-- database.yaml                # Conexiones
+-- src/                             # Codigo fuente principal
|   +-- core/
|   |   +-- services/
|   |       +-- feature_builder.py   # UNICA ubicacion de features
|   |       +-- data_loader.py
|   |       +-- metrics.py
|   +-- shared/
|       +-- config_loader.py
|       +-- trading_calendar.py
|       +-- db_connection.py
+-- airflow/
|   +-- dags/                        # 5 DAGs (limpios)
|       +-- l0_ohlcv_realtime.py
|       +-- l0_macro_daily_update.py
|       +-- l1_feature_transform.py
|       +-- l5_realtime_inference.py
|       +-- alert_monitor.py
+-- training/
|   +-- pipeline/                    # Renombrado de notebooks/...
|       +-- config/settings.py       # Importa de feature_config.json
|       +-- src/environment.py
|       +-- src/utils.py
+-- services/                        # APIs (importan FeatureBuilder)
+-- data/
|   +-- pipeline/                    # Solo scrapers y output
+-- tests/
+-- docs/
    +-- ARQUITECTURA_INTEGRAL_V3.md
    +-- MAPEO_MIGRACION_BIDIRECCIONAL.md  # Este documento
```

---

## APENDICE B: COMANDOS DE MIGRACION

### Paso 1: Crear estructura de directorios

```bash
cd "C:\Users\pedro\OneDrive\Documents\ALGO TRADING\USDCOP\USDCOP-RL-Models"

# Crear directorios
mkdir -p src/core/services
mkdir -p src/shared
mkdir -p tests/unit
mkdir -p training/pipeline
```

### Paso 2: Mover training pipeline

```bash
# Copiar (no mover aun) para preservar estructura legacy
cp -r "notebooks/pipeline entrenamiento/config" training/pipeline/
cp -r "notebooks/pipeline entrenamiento/src" training/pipeline/
```

### Paso 3: Crear feature_builder.py

```bash
# Crear archivo consolidado (ver contenido en seccion 2.1)
touch src/core/services/feature_builder.py
touch src/core/services/__init__.py
touch src/shared/config_loader.py
touch src/shared/__init__.py
```

### Paso 4: Actualizar DAG de inferencia

```bash
# Backup del DAG actual
cp airflow/dags/usdcop_m5__06_l5_realtime_inference.py \
   airflow/dags/usdcop_m5__06_l5_realtime_inference_BACKUP_$(date +%Y%m%d).py

# Editar para usar feature_config.json
# (ver codigo corregido en ARQUITECTURA_INTEGRAL_V3.md seccion 20.2)
```

### Paso 5: Crear tablas SQL

```bash
# Ejecutar en PostgreSQL
psql -h usdcop-postgres-timescale -U admin -d usdcop_trading \
    -f init-scripts/02-macro-data-schema.sql
psql -h usdcop-postgres-timescale -U admin -d usdcop_trading \
    -f init-scripts/03-inference-features-schema.sql
```

---

*Documento generado el 2025-12-16*
*Basado en analisis de 58 archivos Python del proyecto actual*
*Para uso en migracion a arquitectura v3.0*
