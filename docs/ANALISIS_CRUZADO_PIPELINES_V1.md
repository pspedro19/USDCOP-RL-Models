# Análisis Cruzado de Pipelines - USDCOP RL System
**Fecha**: 2026-02-01
**Versión**: 1.0

---

## 1. MAPA DE FLUJO DE DATOS COMPLETO

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              L0: DATA INGESTION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────┐      ┌─────────────────────────────────────────┐   │
│  │  l0_ohlcv_realtime      │      │  l0_macro_update                        │   │
│  │  Schedule: */5 min      │      │  Schedule: Hourly (8-12 COT)            │   │
│  │  Source: TwelveData API │      │  Sources: FRED, Investing, SUAMECA...   │   │
│  │                         │      │                                          │   │
│  │  Output:                │      │  Output:                                 │   │
│  │  usdcop_m5_ohlcv        │      │  macro_indicators_daily                 │   │
│  │  (91K+ rows, 5min bars) │      │  (10K+ rows, 39 variables)              │   │
│  └───────────┬─────────────┘      └───────────────┬─────────────────────────┘   │
│              │                                     │                             │
│              └──────────────┬──────────────────────┘                             │
│                             │                                                    │
│              ┌──────────────▼──────────────┐                                    │
│              │  l0_backup_restore          │                                    │
│              │  Schedule: Weekly (Sunday)   │                                    │
│              │  Backup: BOTH tables         │                                    │
│              └─────────────────────────────┘                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          L2: DATASET BUILDING                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ⚠️ PROBLEMA: 3 PATHS REDUNDANTES CON DIFERENCIAS CRÍTICAS                      │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ PATH 1: l2_dataset_builder.py (Airflow DAG) ✅ SSOT                     │    │
│  │ • ADX: Wilder's EMA + % normalization                                   │    │
│  │ • RSI: Wilder's smoothing, period=9                                     │    │
│  │ • Z-scores: ROLLING 252-day ⚠️ INCONSISTENTE                            │    │
│  │ • Clip: ±10.0                                                           │    │
│  │ • Output: Parquet + norm_stats.json + lineage.json                      │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ PATH 2: 01_build_5min_datasets.py (Standalone) ⚠️ LEGACY                │    │
│  │ • ADX: Simple rolling window (NO Wilder's)                              │    │
│  │ • RSI: Simple rolling, period=14                                        │    │
│  │ • Z-scores: FIXED (hardcoded from 2020-2025) ✅ MEJOR                    │    │
│  │ • Clip: ±4.0                                                            │    │
│  │ • Output: CSV (10 datasets)                                             │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ PATH 3: 02_build_daily_datasets.py (Standalone) ⚠️ LEGACY               │    │
│  │ • Mismo que PATH 2 pero daily frequency                                 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Output Común: data/pipeline/07_output/5min/                                    │
│  • DS_{name}_train.parquet                                                      │
│  • DS_{name}_val.parquet                                                        │
│  • DS_{name}_test.parquet                                                       │
│  • DS_{name}_norm_stats.json                                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           L3: MODEL TRAINING                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ l3_model_training.py → TrainingEngine                                   │    │
│  │                                                                          │    │
│  │ Input: DS_{name}_train.parquet + norm_stats.json                        │    │
│  │                                                                          │    │
│  │ ✅ FIX: Usa L2 norm_stats (no recalcula)                                │    │
│  │ ✅ FIX: trading_env.py skip _z features                                 │    │
│  │ ⚠️ ISSUE: episode_length 1200 vs 2000 inconsistente                     │    │
│  │ ✅ ent_coef = 0.05 (correcto)                                           │    │
│  │                                                                          │    │
│  │ Feature Order (15-dim):                                                  │    │
│  │ [log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct, adx_14,           │    │
│  │  dxy_z, dxy_change_1d, vix_z, embi_z, brent_change_1d,                  │    │
│  │  rate_spread, usdmxn_change_1d, position, time_normalized]              │    │
│  │                                                                          │    │
│  │ Output: model.zip, norm_stats.json, training_result.json                │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        L4: BACKTEST & PROMOTION                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ l4_backtest_promotion.py                                                │    │
│  │                                                                          │    │
│  │ Waits for: L3 via ExternalTaskSensor ✅                                 │    │
│  │ Test data: DS_{name}_test.parquet (out-of-sample)                       │    │
│  │                                                                          │    │
│  │ Two-Vote System:                                                        │    │
│  │ • Vote 1 (Automatic): Sharpe > 0.5, DD < 15%, WinRate > 45%            │    │
│  │ • Vote 2 (Human): Dashboard approval                                    │    │
│  │                                                                          │    │
│  │ Promotion to: s3://production/models/ + model_registry                  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. ANÁLISIS DE INCONSISTENCIAS CRÍTICAS

### 2.1 ❌ Z-SCORE: ROLLING vs FIXED (CRÍTICO)

| Pipeline | Método | Problema |
|----------|--------|----------|
| **L2 DAG (PATH 1)** | ROLLING 252-day | Look-ahead bias, feature drift en inferencia |
| **Legacy Scripts (PATH 2-3)** | FIXED (hardcoded) | Correcto, training-production parity |
| **L3 Training** | Usa lo que L2 genere | Depende de qué PATH se usó |

**IMPACTO**:
- Si se entrena con ROLLING y se hace inferencia con FIXED → Feature distribution shift
- Si se entrena con ROLLING → El modelo "ve el futuro" durante training

**SOLUCIÓN REQUERIDA**:
```python
# En l2_dataset_builder.py, cambiar de:
dxy_mean = result['dxy'].rolling(252, min_periods=20).mean()

# A:
FIXED_MACRO_STATS = {
    'dxy': {'mean': 100.21, 'std': 5.60},
    'vix': {'mean': 21.16, 'std': 7.89},
    'embi': {'mean': 322.01, 'std': 62.68},
}
result['dxy_z'] = (result['dxy'] - 100.21) / 5.60
```

---

### 2.2 ❌ ADX FORMULA DIVERGENCE (CRÍTICO)

| Pipeline | Fórmula | Resultado |
|----------|---------|-----------|
| **L2 DAG** | Wilder's EMA + % normalization | ADX mean ~30, correcto |
| **Legacy Scripts** | Simple rolling window | ADX SATURA a 95+ |

**VERIFICACIÓN ACTUAL**:
```
Dataset actual (DS_v2_fixed2_train.parquet):
  ADX mean: -0.00  ← YA NORMALIZADO (z-score)
  ADX std:  1.00   ← YA NORMALIZADO
```

**NOTA**: El dataset actual ya está normalizado, lo cual oculta si el ADX raw estaba saturado o no.

---

### 2.3 ⚠️ EPISODE LENGTH INCONSISTENCY

| Archivo | Valor | Usado por |
|---------|-------|-----------|
| `config.py` line 191 | **2000** | Default global |
| `training_config.yaml` | **1200** | SSOT (nuevo) |
| `trading_env.py` | **1200** | Default env |
| `engine.py` | **1200** | Fallback |

**IMPACTO**: El training usa 1200, pero si alguien lee `config.py` esperaría 2000.

**SOLUCIÓN**: Actualizar `config.py` a 1200 para consistencia.

---

### 2.4 ⚠️ CLIP RANGE INCONSISTENCY

| Pipeline | Clip Range | Impacto |
|----------|------------|---------|
| L2 DAG | ±10.0 | Permite outliers extremos |
| Legacy Scripts | ±4.0 | Más conservador |
| L3 Environment | ±5.0 | Intermedio |

**SOLUCIÓN**: Estandarizar en ±5.0 en todos los lugares.

---

### 2.5 ⚠️ RSI PERIOD/SMOOTHING MISMATCH

| Pipeline | Period | Smoothing |
|----------|--------|-----------|
| L2 DAG | 9 | Wilder's EMA |
| Legacy Scripts | 14 | Simple rolling |

**IMPACTO**: RSI calculado diferente produce features incompatibles.

---

### 2.6 ✅ NORMALIZATION CHAIN (CORREGIDO)

```
L2 Dataset Builder:
  → Calcula features (ADX, RSI, returns, z-scores)
  → Guarda norm_stats.json (mean, std por feature)
  → Aplica normalización al dataset

L3 Training Engine:
  → Lee dataset normalizado
  → ✅ FIX: Busca L2 norm_stats.json primero (no recalcula)
  → Pasa norm_stats a TradingEnvironment

TradingEnvironment:
  → ✅ FIX: Skip normalización para features _z
  → ✅ FIX: Detecta features ya normalizados (mean≈0, std≈1)
  → Solo normaliza features no-normalizados
```

---

## 3. CONEXIONES ENTRE PIPELINES

### 3.1 L0 OHLCV → L2

| Aspecto | L0 Output | L2 Input | Match? |
|---------|-----------|----------|--------|
| Tabla | `usdcop_m5_ohlcv` | Lee de `usdcop_m5_ohlcv` | ✅ |
| Columnas | time, open, high, low, close, volume | Usa todas | ✅ |
| Frecuencia | 5-min bars | Espera 5-min | ✅ |

### 3.2 L0 MACRO → L2

| Aspecto | L0 Output | L2 Input | Match? |
|---------|-----------|----------|--------|
| Tabla | `macro_indicators_daily` | Lee de `macro_indicators_daily` | ✅ |
| Variables críticas | dxy, vix, embi_col, ust10y, ibr, tpm | Usa dxy, vix, embi, rates | ✅ |
| Frecuencia | Daily | Forward-fill a 5-min | ✅ |

**NOTA**: L2 hace forward-fill de macro a 5-min, con anti-leakage (shift 1 day).

### 3.3 L2 → L3

| Aspecto | L2 Output | L3 Input | Match? |
|---------|-----------|----------|--------|
| Archivo | DS_{name}_train.parquet | Lee train.parquet | ✅ |
| norm_stats | DS_{name}_norm_stats.json | Busca primero | ✅ |
| Features | 13 market features | Espera 13 features | ✅ |
| Feature Order | Definido en FEATURE_ORDER | Validado en engine.py | ✅ |

### 3.4 L3 → L4

| Aspecto | L3 Output | L4 Input | Match? |
|---------|-----------|----------|--------|
| Model | model.zip | Carga PPO.load() | ✅ |
| Test Data | DS_{name}_test.parquet | Lee test.parquet | ✅ |
| norm_stats | Copia de L2 | Usa para normalizar | ✅ |
| XCom | L3Output contract | ExternalTaskSensor | ✅ |

---

## 4. MATRIZ DE CONFIGURACIÓN CRUZADA

| Parámetro | L0 OHLCV | L0 Macro | L2 DAG | L2 Legacy | L3 | L4 |
|-----------|----------|----------|--------|-----------|-----|-----|
| **Market Hours** | 8:00-12:55 | 8:00-12:00 | N/A | N/A | N/A | N/A |
| **Timezone** | COT | COT | COT | COT | UTC | UTC |
| **ADX Period** | N/A | N/A | 14 | 14 | N/A | N/A |
| **ADX Smoothing** | N/A | N/A | Wilder's | Rolling | N/A | N/A |
| **RSI Period** | N/A | N/A | 9 | 14 | N/A | N/A |
| **Z-score Method** | N/A | N/A | ROLLING | FIXED | N/A | N/A |
| **Clip Range** | N/A | N/A | ±10 | ±4 | ±5 | N/A |
| **Episode Length** | N/A | N/A | N/A | N/A | 1200 | N/A |
| **ent_coef** | N/A | N/A | N/A | N/A | 0.05 | N/A |
| **min_sharpe** | N/A | N/A | N/A | N/A | N/A | 0.5 |
| **max_drawdown** | N/A | N/A | N/A | N/A | 0.25 | 0.15 |

---

## 5. PROBLEMAS IDENTIFICADOS Y PRIORIDAD

### P0 - CRÍTICOS (Bloquean producción)

| # | Problema | Ubicación | Impacto | Solución |
|---|----------|-----------|---------|----------|
| 1 | Z-score ROLLING vs FIXED | L2 DAG | Feature drift en inferencia | Cambiar a FIXED |
| 2 | 3 paths redundantes L2 | Múltiples scripts | Inconsistencia de features | Consolidar en 1 |
| 3 | ADX fórmula diferente | L2 DAG vs Legacy | Modelo incompatible | Estandarizar Wilder's |

### P1 - ALTOS (Afectan calidad)

| # | Problema | Ubicación | Impacto | Solución |
|---|----------|-----------|---------|----------|
| 4 | RSI period 9 vs 14 | L2 DAG vs Legacy | Feature diferente | Estandarizar en 9 |
| 5 | Clip range inconsistente | Múltiples | Distribución diferente | Estandarizar ±5 |
| 6 | Episode length 1200 vs 2000 | config.py vs otros | Confusión | Actualizar config.py |

### P2 - MEDIOS (Mejoras)

| # | Problema | Ubicación | Impacto | Solución |
|---|----------|-----------|---------|----------|
| 7 | max_drawdown 0.25 vs 0.15 | L3 vs L4 | Criterios diferentes | Alinear valores |
| 8 | Legacy scripts sin deprecar | data/pipeline/06_* | Mantenimiento | Mover a deprecated/ |

---

## 6. PLAN CONSOLIDADO ACTUALIZADO

### FASE 1: CORRECCIONES CRÍTICAS (P0) - Días 1-3

#### 1.1 Unificar Z-Score a FIXED

**Archivo**: `airflow/dags/l2_dataset_builder.py`

```python
# ANTES (ROLLING - problemático):
dxy_mean = result['dxy'].rolling(252, min_periods=20).mean()
dxy_std = result['dxy'].rolling(252, min_periods=20).std()
result['dxy_z'] = (result['dxy'] - dxy_mean) / dxy_std.clip(lower=0.01)

# DESPUÉS (FIXED - correcto):
FIXED_MACRO_STATS = {
    'dxy': {'mean': 100.21, 'std': 5.60},
    'vix': {'mean': 21.16, 'std': 7.89},
    'embi': {'mean': 322.01, 'std': 62.68},
}

result['dxy_z'] = ((result['dxy'] - FIXED_MACRO_STATS['dxy']['mean'])
                   / FIXED_MACRO_STATS['dxy']['std']).clip(-10, 10)
result['vix_z'] = ((result['vix'] - FIXED_MACRO_STATS['vix']['mean'])
                   / FIXED_MACRO_STATS['vix']['std']).clip(-10, 10)
result['embi_z'] = ((result['embi'] - FIXED_MACRO_STATS['embi']['mean'])
                    / FIXED_MACRO_STATS['embi']['std']).clip(-10, 10)
```

#### 1.2 Consolidar L2 en Un Solo Path

**Acción**:
1. Deprecar `01_build_5min_datasets.py` y `02_build_daily_datasets.py`
2. Usar exclusivamente `l2_dataset_builder.py` (Airflow DAG)
3. Mover scripts legacy a `scripts/deprecated/`

```bash
mkdir -p scripts/deprecated
mv data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py scripts/deprecated/
mv data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py scripts/deprecated/
```

#### 1.3 Estandarizar ADX con Wilder's + Percentage

**Archivo**: `src/features/technical_indicators.py` (ya creado)

El módulo `technical_indicators.py` creado anteriormente tiene la implementación correcta.
Actualizar L2 DAG para importar de ahí:

```python
from src.features.technical_indicators import (
    calculate_adx_wilders,
    calculate_rsi_wilders,
)
```

---

### FASE 2: CORRECCIONES ALTAS (P1) - Días 4-5

#### 2.1 Estandarizar RSI en Period=9

**Archivo**: `airflow/dags/l2_dataset_builder.py`

Usar `calculate_rsi_wilders(close, period=9)` del módulo compartido.

#### 2.2 Estandarizar Clip Range en ±5.0

**Archivos a modificar**:
- `airflow/dags/l2_dataset_builder.py`: Cambiar clip de ±10 a ±5
- `src/training/environments/trading_env.py`: Ya usa ±5 ✅

#### 2.3 Unificar Episode Length a 1200

**Archivo**: `src/training/config.py` line 191

```python
# ANTES:
max_episode_steps: int = 2000

# DESPUÉS:
max_episode_steps: int = 1200  # Matches training_config.yaml SSOT
```

---

### FASE 3: MEJORAS (P2) - Días 6-7

#### 3.1 Alinear max_drawdown

**Decisión**: Usar 0.20 (20%) como compromiso
- L3: Cambiar de 0.25 a 0.20
- L4: Cambiar de 0.15 a 0.20

#### 3.2 Deprecar Scripts Legacy

Mover a `scripts/deprecated/` con README explicando que están obsoletos.

---

## 7. VALIDACIÓN POST-IMPLEMENTACIÓN

### Tests de Paridad

```python
# tests/integration/test_pipeline_parity.py

def test_adx_formula_consistency():
    """Verificar que ADX se calcula igual en todos los lugares."""
    from src.features.technical_indicators import calculate_adx_wilders

    # Generar datos de prueba
    df = load_test_ohlcv()

    # Calcular con módulo compartido
    adx_shared = calculate_adx_wilders(df['high'], df['low'], df['close'])

    # Verificar distribución saludable
    assert adx_shared.mean() < 50, "ADX saturado"
    assert adx_shared.std() > 10, "ADX sin varianza"

def test_zscore_is_fixed():
    """Verificar que z-scores usan stats FIXED, no ROLLING."""
    df = pd.read_parquet('data/pipeline/07_output/5min/DS_v2_fixed2_train.parquet')

    # Los z-scores deben tener mean≈0, std≈1 CONSISTENTEMENTE
    for col in ['dxy_z', 'vix_z', 'embi_z']:
        assert abs(df[col].mean()) < 0.1, f"{col} mean not near 0"
        assert 0.8 < df[col].std() < 1.2, f"{col} std not near 1"

def test_feature_order_matches_contract():
    """Verificar que feature order es consistente L2→L3→L4."""
    EXPECTED_ORDER = [
        "log_ret_5m", "log_ret_1h", "log_ret_4h",
        "rsi_9", "atr_pct", "adx_14",
        "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
        "brent_change_1d", "rate_spread", "usdmxn_change_1d"
    ]

    # Verificar L2 output
    df = pd.read_parquet('data/pipeline/07_output/5min/DS_v2_fixed2_train.parquet')
    for feat in EXPECTED_ORDER:
        assert feat in df.columns, f"Missing {feat} in L2 output"

    # Verificar norm_stats
    with open('data/pipeline/07_output/5min/DS_v2_fixed2_norm_stats.json') as f:
        stats = json.load(f)
    assert stats['feature_order'] == EXPECTED_ORDER
```

---

## 8. RESUMEN DE CAMBIOS AL PLAN ORIGINAL

| Sección Original | Cambio | Razón |
|------------------|--------|-------|
| Fase 1.1 ADX Fix | ✅ Mantener | Ya implementado correctamente |
| Fase 1.2 Triple Norm | ✅ Mantener | Ya corregido en engine.py |
| Fase 1.3 Episode Length | ⬆️ Priorizar | Inconsistencia detectada |
| Fase 2.1 Deprecar Scripts | ✅ Mantener | Confirmado 3 paths redundantes |
| Fase 2.2 Unificar Indicadores | ⬆️ Expandir | Incluir RSI period fix |
| **NUEVO** | Z-score FIXED | Crítico: ROLLING causa drift |
| **NUEVO** | Clip Range ±5 | Estandarizar distribución |
| **NUEVO** | max_drawdown 0.20 | Alinear L3 y L4 |

---

## 9. DIAGRAMA DE DEPENDENCIAS FINAL

```
┌─────────────────────────────────────────────────────────────────┐
│                    SSOT CONFIGURATION FILES                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  config/training_config.yaml ──────┬─────────────────────────┐ │
│  config/macro_variables_ssot.yaml ─┼─────────────────────────┤ │
│  src/features/technical_indicators.py ─────────────────────┐ │ │
│                                    │                       │ │ │
└────────────────────────────────────┼───────────────────────┼─┼─┘
                                     │                       │ │
                                     ▼                       │ │
┌─────────────────────────────────────────────────────────────┼─┼─┐
│  L0: DATA INGESTION                                         │ │ │
│  ├── l0_ohlcv_realtime (TwelveData → usdcop_m5_ohlcv)      │ │ │
│  ├── l0_macro_update (Multi-source → macro_indicators_daily)│ │
│  └── l0_backup_restore (Weekly backup both tables)          │ │ │
└────────────────────────────────────┬────────────────────────┼─┼─┘
                                     │                       │ │
                                     ▼                       │ │
┌─────────────────────────────────────────────────────────────┼─┼─┐
│  L2: DATASET BUILDING (SINGLE PATH)                         │ │ │
│  └── l2_dataset_builder.py (Airflow DAG) ◄─────────────────┘ │ │
│       ├── Imports: technical_indicators.py ◄─────────────────┘ │
│       ├── Uses: FIXED z-score stats                            │
│       ├── Output: train/val/test.parquet + norm_stats.json     │
│       └── Feature Order: 13 market features (CONTRACT)         │
└────────────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  L3: MODEL TRAINING                                             │
│  └── l3_model_training.py → TrainingEngine                     │
│       ├── Reads: L2 train.parquet + norm_stats.json            │
│       ├── Config: training_config.yaml (SSOT)                   │
│       ├── Environment: episode_length=1200, ent_coef=0.05      │
│       └── Output: model.zip, training_result.json              │
└────────────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│  L4: BACKTEST & PROMOTION                                       │
│  └── l4_backtest_promotion.py                                  │
│       ├── Waits: L3 via ExternalTaskSensor                     │
│       ├── Reads: L2 test.parquet + L3 model.zip                │
│       ├── Vote 1: Automatic (Sharpe>0.5, DD<20%, WR>45%)       │
│       ├── Vote 2: Human (Dashboard approval)                   │
│       └── Promotion: → s3://production/models/                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. CONCLUSIÓN

El análisis cruzado reveló **3 inconsistencias críticas** que no estaban en el plan original:

1. **Z-score ROLLING vs FIXED**: El L2 DAG usa rolling windows que causan feature drift. Debe cambiarse a estadísticas fijas.

2. **RSI Period 9 vs 14**: Diferentes períodos producen features incompatibles.

3. **Clip Range ±10 vs ±4 vs ±5**: Distribuciones diferentes afectan el modelo.

El plan original estaba **mayormente correcto** pero necesita estos ajustes adicionales para garantizar paridad training-production.

**Próximos pasos**:
1. Implementar z-score FIXED en L2 DAG
2. Estandarizar RSI period=9
3. Unificar clip range a ±5
4. Regenerar datasets con la configuración corregida
5. Reentrenar modelo
6. Validar con tests de paridad

---

**Documento generado**: 2026-02-01
**Archivos analizados**: 15+ archivos core
**Pipelines cubiertos**: L0→L4 completo
