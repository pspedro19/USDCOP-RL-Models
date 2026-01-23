# Plan Maestro: Unificación de Datos RL + Forecasting

## Versión 1.0.0 | 2026-01-22

---

## Resumen Ejecutivo

Este plan unifica las arquitecturas de datos de RL (5-min) y Forecasting (daily) bajo el principio **SSOT + DRY**, donde ambos pipelines comparten la misma fuente de macro data de PostgreSQL, con RL haciendo resample + ffill para adaptarlo a 5-min.

---

## Arquitectura Actual (PROBLEMÁTICA)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESTADO ACTUAL (FRAGMENTADO)                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐                                    ┌──────────────┐
│  RL Training │                                    │ Forecasting  │
│   (5-min)    │                                    │   (daily)    │
└──────┬───────┘                                    └──────┬───────┘
       │                                                   │
       ▼                                                   ▼
┌──────────────┐                                    ┌──────────────┐
│ CSV Backups  │                                    │   yfinance   │
│ (offline)    │                                    │   (API)      │
└──────────────┘                                    └──────────────┘
       │                                                   │
       │  MACRO_5MIN_CONSOLIDATED.csv                      │  yf.Ticker("DX-Y.NYB")
       │  (pre-procesado offline)                          │  yf.Ticker("CL=F")
       │                                                   │
       └───────────────────┬───────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────────┐
                │ PROBLEMAS:              │
                │ • 2 fuentes de macro    │
                │ • Sin SSOT              │
                │ • Código duplicado      │
                │ • No DRY                │
                └─────────────────────────┘
```

---

## Arquitectura Propuesta (UNIFICADA)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA PROPUESTA (SSOT + DRY)                       │
└─────────────────────────────────────────────────────────────────────────────┘

                      ┌─────────────────────────────┐
                      │    PostgreSQL (SSOT)        │
                      ├─────────────────────────────┤
                      │ • usdcop_m5_ohlcv  (5-min)  │
                      │ • macro_indicators_daily    │
                      │   (37 columnas macro)       │
                      └──────────────┬──────────────┘
                                     │
                      ┌──────────────┴──────────────┐
                      │   UnifiedMacroLoader        │
                      │   (src/data/macro_loader.py)│
                      └──────────────┬──────────────┘
                                     │
                 ┌───────────────────┼───────────────────┐
                 │                   │                   │
                 ▼                   ▼                   ▼
       ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
       │   load_daily()  │ │ resample_to_5m()│ │   load_ohlcv()  │
       │   (Forecasting) │ │ (RL + ffill)    │ │   (RL 5-min)    │
       └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
                │                   │                   │
                ▼                   ▼                   ▼
       ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
       │ Forecasting     │ │ RL Training     │ │ RL Inference    │
       │ Training (daily)│ │ (5-min)         │ │ (5-min)         │
       └─────────────────┘ └─────────────────┘ └─────────────────┘
```

---

## Comparación de Pipelines

| Aspecto | RL Pipeline | Forecasting Pipeline |
|---------|-------------|----------------------|
| **Frecuencia OHLCV** | 5-min | Daily (resample de 5-min) |
| **Fuente OHLCV** | `usdcop_m5_ohlcv` (DB) | `usdcop_m5_ohlcv` → resample daily |
| **Fuente Macro** | `macro_indicators_daily` (DB) | `macro_indicators_daily` (DB) |
| **Resample Macro** | Daily → 5-min (ffill) | N/A (ya es daily) |
| **Features** | 15-dim (13 core + 2 state) | 19-dim |
| **Normalización** | Z-score (fixed stats) | Raw/% returns |

---

## Columnas Macro Compartidas (SSOT)

### De `macro_indicators_daily`:

```sql
-- FOREX
fxrt_index_dxy_usa_d_dxy         → DXY (Dollar Index)
fxrt_spot_usdmxn_mex_d_usdmxn    → USD/MXN
fxrt_spot_usdclp_chl_d_usdclp    → USD/CLP

-- COMMODITIES
comm_oil_brent_glb_d_brent       → Brent Oil
comm_oil_wti_glb_d_wti           → WTI Oil (para Forecasting)

-- VOLATILITY
volt_vix_usa_d_vix               → VIX

-- CREDIT RISK
crsk_spread_embi_col_d_embi      → EMBI Colombia

-- INTEREST RATES
finc_bond_yield10y_usa_d_ust10y  → US 10Y Treasury
finc_bond_yield2y_usa_d_dgs2     → US 2Y Treasury
finc_bond_yield10y_col_d_col10y  → Colombia 10Y Bond
```

---

## Plan de Implementación

### Fase 1: Core Data Infrastructure (CREAR)

#### 1.1 Unified Macro Loader
**Archivo NUEVO**: `src/data/macro_loader.py`

```python
"""
SSOT: Carga macro desde PostgreSQL para RL y Forecasting.
"""
class UnifiedMacroLoader:
    def load_daily(self) -> pd.DataFrame:
        """Carga macro_indicators_daily desde DB."""

    def resample_to_5min(self, macro_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Resample daily → 5-min con ffill.
        - Crea grid 5-min (8:00am-12:55pm COT)
        - ffill con límite (24 bars = 2 horas)
        - Evita lookahead con merge_asof
        """

    def load_5min(self) -> pd.DataFrame:
        """Carga y resamplea en un solo paso."""
```

#### 1.2 Unified OHLCV Loader
**Archivo NUEVO**: `src/data/ohlcv_loader.py`

```python
"""
SSOT: Carga OHLCV desde PostgreSQL o backup CSV.
"""
class UnifiedOHLCVLoader:
    def load_5min(self, fallback_csv: bool = True) -> pd.DataFrame:
        """Carga usdcop_m5_ohlcv desde DB o CSV backup."""

    def resample_to_daily(self, ohlcv_5min: pd.DataFrame) -> pd.DataFrame:
        """Resample 5-min → daily para Forecasting."""
```

#### 1.3 Data Module Init
**Archivo NUEVO**: `src/data/__init__.py`

```python
from .macro_loader import UnifiedMacroLoader
from .ohlcv_loader import UnifiedOHLCVLoader

__all__ = ["UnifiedMacroLoader", "UnifiedOHLCVLoader"]
```

---

### Fase 2: Feature Builders Refactorizados (MODIFICAR)

#### 2.1 RL Feature Builder
**Archivo MODIFICAR**: `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py`

**Cambios**:
```python
# ANTES (hardcoded CSV paths)
macro_path = "data/pipeline/05_resampling/output/MACRO_5MIN_CONSOLIDATED.csv"
macro_df = pd.read_csv(macro_path)

# DESPUÉS (SSOT desde DB)
from src.data import UnifiedMacroLoader

loader = UnifiedMacroLoader()
macro_df = loader.load_5min()  # Ya incluye resample + ffill
```

#### 2.2 Forecasting Dataset Builder
**Archivo MODIFICAR**: `scripts/build_forecasting_dataset_aligned.py`

**Cambios**:
```python
# ANTES (yfinance API calls)
dxy = yf.Ticker("DX-Y.NYB").history(...)
wti = yf.Ticker("CL=F").history(...)

# DESPUÉS (SSOT desde DB)
from src.data import UnifiedMacroLoader, UnifiedOHLCVLoader

ohlcv_loader = UnifiedOHLCVLoader()
macro_loader = UnifiedMacroLoader()

# OHLCV daily (resample de 5-min)
ohlcv_daily = ohlcv_loader.load_5min()
ohlcv_daily = ohlcv_loader.resample_to_daily(ohlcv_daily)

# Macro daily (directo de DB)
macro_daily = macro_loader.load_daily()
macro_daily = macro_daily[['fecha', 'fxrt_index_dxy_usa_d_dxy', 'comm_oil_wti_glb_d_wti']]
macro_daily = macro_daily.rename(columns={
    'fxrt_index_dxy_usa_d_dxy': 'dxy',
    'comm_oil_wti_glb_d_wti': 'wti'
})
```

---

### Fase 3: DAGs Actualizados (MODIFICAR)

#### 3.1 L2 Preprocessing DAG
**Archivo MODIFICAR**: `airflow/dags/l2_preprocessing_pipeline.py`

**Cambios**: Usar `UnifiedMacroLoader` en lugar de leer CSV directo.

#### 3.2 L3 Model Training DAG
**Archivo MODIFICAR**: `airflow/dags/l3_model_training.py`

**Cambios**: Dataset ahora viene de DB via loaders unificados.

#### 3.3 L3b Forecasting Training DAG
**Archivo**: `airflow/dags/l3b_forecasting_training.py` (ya existe)

**Cambios**: Usar loaders unificados para macro data.

---

### Fase 4: Archivos a ELIMINAR (Deprecar)

| Archivo | Razón |
|---------|-------|
| `data/pipeline/05_resampling/run_resample.py` | Lógica movida a `UnifiedMacroLoader.resample_to_5min()` |
| `data/pipeline/05_resampling/output/*.csv` | Generados on-demand por loader |

**NOTA**: No eliminar físicamente hasta validar que el nuevo sistema funciona.

---

### Fase 5: Migraciones de Base de Datos (CREAR)

#### 5.1 View para Features Compartidas
**Archivo NUEVO**: `database/migrations/026_shared_macro_view.sql`

```sql
-- View: Macro data normalizada para ambos pipelines
CREATE OR REPLACE VIEW public.v_macro_unified AS
SELECT
    fecha,
    -- FOREX
    fxrt_index_dxy_usa_d_dxy AS dxy,
    fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn,
    fxrt_spot_usdclp_chl_d_usdclp AS usdclp,
    -- COMMODITIES
    comm_oil_brent_glb_d_brent AS brent,
    comm_oil_wti_glb_d_wti AS wti,
    -- VOLATILITY
    volt_vix_usa_d_vix AS vix,
    -- CREDIT
    crsk_spread_embi_col_d_embi AS embi,
    -- RATES
    finc_bond_yield10y_usa_d_ust10y AS ust10y,
    finc_bond_yield2y_usa_d_dgs2 AS ust2y,
    finc_bond_yield10y_col_d_col10y AS col10y,
    -- CALCULATED
    finc_bond_yield10y_col_d_col10y - finc_bond_yield10y_usa_d_ust10y AS rate_spread
FROM macro_indicators_daily
WHERE fecha >= '2015-01-01'
ORDER BY fecha;

COMMENT ON VIEW public.v_macro_unified IS 'SSOT: Macro data para RL y Forecasting';
```

---

## Flujo de Datos Unificado

### RL Training (5-min)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RL TRAINING DATA FLOW                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

PostgreSQL                           Python Loaders                  Output
┌──────────────────┐               ┌───────────────────┐          ┌──────────┐
│ usdcop_m5_ohlcv  │──────────────▶│ UnifiedOHLCVLoader│─────────▶│ OHLCV    │
│ (5-min bars)     │               │ .load_5min()      │          │ (5-min)  │
└──────────────────┘               └───────────────────┘          └────┬─────┘
                                                                       │
┌──────────────────┐               ┌───────────────────┐          ┌────▼─────┐
│macro_indicators_ │──────────────▶│ UnifiedMacroLoader│─────────▶│ MACRO    │
│daily (daily)     │               │ .load_5min()      │          │ (5-min   │
└──────────────────┘               │ (resample+ffill)  │          │ ffilled) │
                                   └───────────────────┘          └────┬─────┘
                                                                       │
                                                                       ▼
                                                                 ┌──────────┐
                                                                 │ MERGE    │
                                                                 │ on time  │
                                                                 └────┬─────┘
                                                                      │
                                                                      ▼
                                                                ┌───────────┐
                                                                │ RL Dataset│
                                                                │ (15-dim)  │
                                                                └───────────┘
```

### Forecasting Training (daily)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FORECASTING TRAINING DATA FLOW                                               │
└─────────────────────────────────────────────────────────────────────────────┘

PostgreSQL                           Python Loaders                  Output
┌──────────────────┐               ┌───────────────────┐          ┌──────────┐
│ usdcop_m5_ohlcv  │──────────────▶│ UnifiedOHLCVLoader│─────────▶│ OHLCV    │
│ (5-min bars)     │               │ .load_5min()      │          │ (daily   │
│                  │               │ .resample_to_daily│          │resample) │
└──────────────────┘               └───────────────────┘          └────┬─────┘
                                                                       │
┌──────────────────┐               ┌───────────────────┐          ┌────▼─────┐
│macro_indicators_ │──────────────▶│ UnifiedMacroLoader│─────────▶│ MACRO    │
│daily (daily)     │               │ .load_daily()     │          │ (daily   │
└──────────────────┘               │ (NO resample)     │          │ directo) │
                                   └───────────────────┘          └────┬─────┘
                                                                       │
                                                                       ▼
                                                                 ┌──────────┐
                                                                 │ MERGE    │
                                                                 │ on fecha │
                                                                 └────┬─────┘
                                                                      │
                                                                      ▼
                                                              ┌─────────────┐
                                                              │ Forecast DS │
                                                              │ (19-dim)    │
                                                              └─────────────┘
```

---

## Resumen de Archivos

### CREAR (6 archivos)

| Archivo | Descripción |
|---------|-------------|
| `src/data/__init__.py` | Init del módulo data |
| `src/data/macro_loader.py` | UnifiedMacroLoader |
| `src/data/ohlcv_loader.py` | UnifiedOHLCVLoader |
| `src/data/calendar.py` | Calendario de trading (festivos) |
| `database/migrations/026_shared_macro_view.sql` | View v_macro_unified |
| `tests/unit/test_data_loaders.py` | Tests de loaders |

### MODIFICAR (6 archivos)

| Archivo | Cambio |
|---------|--------|
| `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py` | Usar UnifiedMacroLoader |
| `scripts/build_forecasting_dataset_aligned.py` | Usar loaders unificados |
| `airflow/dags/l2_preprocessing_pipeline.py` | Integrar loaders |
| `airflow/dags/l3_model_training.py` | Dataset desde loaders |
| `airflow/dags/l3b_forecasting_training.py` | Usar loaders |
| `src/forecasting/engine.py` | Adaptar data loading |

### DEPRECAR (marcar como deprecated, no eliminar aún)

| Archivo | Razón |
|---------|-------|
| `data/pipeline/05_resampling/run_resample.py` | Lógica en UnifiedMacroLoader |
| `data/pipeline/05_resampling/output/MACRO_5MIN_CONSOLIDATED.csv` | Generado on-demand |
| `data/pipeline/05_resampling/output/MACRO_DAILY_CONSOLIDATED.csv` | Leer de DB |

### NO TOCAR (sin cambios)

| Archivo | Razón |
|---------|-------|
| `src/core/contracts/feature_contract.py` | SSOT de RL features (15-dim) |
| `src/forecasting/config.py` | SSOT de Forecasting features (19-dim) |
| `src/training/environments/trading_env.py` | Consume dataset, no lo genera |
| `config/norm_stats.json` | Stats de normalización RL |

---

## Validación

### Tests Requeridos

```bash
# Unit tests para loaders
pytest tests/unit/test_data_loaders.py -v

# Integration test: RL dataset build
pytest tests/integration/test_rl_dataset_builder.py -v

# Integration test: Forecasting dataset build
pytest tests/integration/test_forecasting_dataset_builder.py -v

# E2E: Verificar que ambos datasets usan mismos macro values
pytest tests/e2e/test_macro_consistency.py -v
```

### Checklist de Validación

- [ ] UnifiedMacroLoader carga desde DB correctamente
- [ ] Resample 5-min produce mismos valores que CSV legacy
- [ ] Forecasting dataset tiene DXY/WTI de la misma fuente que RL
- [ ] No hay lookahead bias en merge (merge_asof backward)
- [ ] ffill tiene límite para evitar datos stale
- [ ] Tests E2E pasan para ambos pipelines

---

## Orden de Ejecución

```
SEMANA 1: Core Infrastructure
├─ Día 1-2: Crear src/data/macro_loader.py
├─ Día 3-4: Crear src/data/ohlcv_loader.py
└─ Día 5: Tests unitarios

SEMANA 2: Integración RL
├─ Día 1-2: Modificar 01_build_5min_datasets.py
├─ Día 3: Modificar l2_preprocessing_pipeline.py
└─ Día 4-5: Tests integración RL

SEMANA 3: Integración Forecasting
├─ Día 1-2: Modificar build_forecasting_dataset_aligned.py
├─ Día 3: Modificar l3b_forecasting_training.py
└─ Día 4-5: Tests integración Forecasting

SEMANA 4: Validación E2E
├─ Día 1-2: Tests E2E macro consistency
├─ Día 3: Documentación
└─ Día 4-5: Code review + merge
```

---

## Beneficios Esperados

| Beneficio | Impacto |
|-----------|---------|
| **SSOT** | Una sola fuente de macro data (DB) |
| **DRY** | Elimina ~40% de código duplicado |
| **Consistencia** | RL y Forecasting usan exactamente los mismos valores macro |
| **Mantenibilidad** | Un solo lugar para actualizar lógica de data loading |
| **Trazabilidad** | Lineage claro desde DB → Dataset → Model |
| **Testing** | Más fácil de testear con mocks |

---

## Riesgos y Mitigaciones

| Riesgo | Mitigación |
|--------|------------|
| Diferencias numéricas vs legacy | Comparar output con CSV legacy, threshold 1e-6 |
| Performance de DB queries | Cachear en memoria, usar connection pooling |
| Downtime durante migración | Mantener CSV como fallback |
| Lookahead bias | Usar merge_asof con direction='backward' |

---

## Apéndice A: Mapping de Columnas

### RL Features ← macro_indicators_daily

| RL Feature | DB Column | Transformación |
|------------|-----------|----------------|
| `dxy` | `fxrt_index_dxy_usa_d_dxy` | Directo |
| `dxy_z` | `fxrt_index_dxy_usa_d_dxy` | Z-score (fixed stats) |
| `dxy_change_1d` | `fxrt_index_dxy_usa_d_dxy` | pct_change(1) |
| `vix` | `volt_vix_usa_d_vix` | Directo |
| `vix_z` | `volt_vix_usa_d_vix` | Z-score |
| `embi` | `crsk_spread_embi_col_d_embi` | Directo |
| `embi_z` | `crsk_spread_embi_col_d_embi` | Z-score |
| `brent` | `comm_oil_brent_glb_d_brent` | Directo |
| `brent_change_1d` | `comm_oil_brent_glb_d_brent` | pct_change(1) |
| `rate_spread` | `col10y - ust10y` | Calculado |
| `usdmxn_change_1d` | `fxrt_spot_usdmxn_mex_d_usdmxn` | pct_change(1) |

### Forecasting Features ← macro_indicators_daily

| Forecast Feature | DB Column | Transformación |
|------------------|-----------|----------------|
| `dxy_close_lag1` | `fxrt_index_dxy_usa_d_dxy` | shift(1) |
| `oil_close_lag1` | `comm_oil_wti_glb_d_wti` | shift(1) |

---

## Apéndice B: Estructura de Directorios Final

```
src/
├── data/                          # NUEVO: Unified data loading
│   ├── __init__.py
│   ├── macro_loader.py            # UnifiedMacroLoader
│   ├── ohlcv_loader.py            # UnifiedOHLCVLoader
│   └── calendar.py                # Trading calendar
│
├── feature_store/                 # EXISTENTE: Feature builders
│   ├── builders/
│   │   ├── canonical_feature_builder.py
│   │   ├── rl_feature_builder.py      # USA UnifiedMacroLoader
│   │   └── forecast_feature_builder.py # USA UnifiedMacroLoader
│   └── ...
│
├── forecasting/                   # EXISTENTE: Forecasting pipeline
│   ├── engine.py                  # MODIFICADO: usar loaders
│   └── config.py                  # SIN CAMBIOS: 19-dim contract
│
├── training/                      # EXISTENTE: RL pipeline
│   └── environments/
│       └── trading_env.py         # SIN CAMBIOS: consume dataset
│
└── core/
    └── contracts/
        └── feature_contract.py    # SIN CAMBIOS: 15-dim contract
```

---

**Autor**: Claude Code
**Fecha**: 2026-01-22
**Principio**: SSOT + DRY para Data Loading
