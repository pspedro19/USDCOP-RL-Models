# Plan Maestro: Arquitectura de Datos Unificada SSOT/DRY

## Versión 3.0.0 | 2026-01-22 | ARCHITECTURE 10/10

---

## Resumen Ejecutivo

Este plan unifica las arquitecturas de datos de **RL (5-min)** y **Forecasting (daily)** bajo los principios **SSOT (Single Source of Truth)** y **DRY (Don't Repeat Yourself)**.

### CRÍTICO: Separación Clara de Pipelines

| Pipeline | OHLCV Source | Table | Loader Method | Frecuencia |
|----------|--------------|-------|---------------|------------|
| **RL** | TwelveData API | `usdcop_m5_ohlcv` | `load_5min()` | 5-min bars |
| **Forecasting** | **Investing.com (OFICIAL)** | `bi.dim_daily_usdcop` | `load_daily()` | Daily close |

### Por qué Investing.com para Forecasting?

1. **Valores OFICIALES de cierre diario** - No resampling de 5-min
2. **Mismo precio que ven los traders** - Consistencia con mercado
3. **Sin aproximaciones** - El close de 5-min puede diferir del oficial
4. **Mejor para backtesting** - Resultados replicables

---

## Arquitectura 10/10

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA 10/10 (SSOT + DRY)                           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA INGESTION LAYER                                │
├───────────────────────────────┬─────────────────────────────────────────────┤
│     TwelveData API            │       Investing.com Scraper                 │
│     (5-minute bars)           │       (Official Daily Close)                │
│             │                 │              │                              │
│     ┌───────▼───────┐         │     ┌────────▼────────┐                     │
│     │ l0_ohlcv_     │         │     │ forecast_l0_    │                     │
│     │ realtime.py   │         │     │ daily_data.py   │                     │
│     └───────┬───────┘         │     └────────┬────────┘                     │
│             │                 │              │                              │
│     ┌───────▼───────┐         │     ┌────────▼────────┐                     │
│     │ usdcop_m5_    │         │     │ bi.dim_daily_   │                     │
│     │ ohlcv         │         │     │ usdcop          │                     │
│     │ (TimescaleDB) │         │     │ (PostgreSQL bi) │                     │
│     └───────────────┘         │     └─────────────────┘                     │
└───────────────────────────────┴─────────────────────────────────────────────┘
             │                             │
             │                             │
┌────────────▼─────────────────────────────▼──────────────────────────────────┐
│                        UNIFIED DATA LOADER                                   │
│                                                                              │
│  ┌──────────────────────────┐    ┌──────────────────────────┐               │
│  │    UnifiedOHLCVLoader    │    │    UnifiedMacroLoader    │               │
│  │                          │    │                          │               │
│  │  • load_5min()  → RL     │    │  • load_daily()  → Both  │               │
│  │  • load_daily() → Fore.  │    │  • load_5min()   → RL    │               │
│  │                          │    │    (with ffill)          │               │
│  └──────────────────────────┘    └──────────────────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
             │                             │
             │                             │
┌────────────▼─────────────────────────────▼──────────────────────────────────┐
│                          PIPELINE CONSUMERS                                  │
│                                                                              │
│  ┌────────────────────────────┐  ┌────────────────────────────┐             │
│  │       RL PIPELINE          │  │    FORECASTING PIPELINE    │             │
│  │                            │  │                            │             │
│  │  Source: TwelveData        │  │  Source: Investing.com     │             │
│  │  Table: usdcop_m5_ohlcv    │  │  Table: bi.dim_daily_usdcop│             │
│  │  Method: load_5min()       │  │  Method: load_daily()      │             │
│  │  Features: 15              │  │  Features: 19              │             │
│  │  RSI: 9-bar                │  │  RSI: 14-day               │             │
│  │                            │  │                            │             │
│  │  Use Case:                 │  │  Use Case:                 │             │
│  │  - Intraday trading        │  │  - Daily forecasting       │             │
│  │  - Pattern recognition     │  │  - Official close prices   │             │
│  │  - RL environment          │  │  - Model training          │             │
│  └────────────────────────────┘  └────────────────────────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tabla Comparativa Final

| Aspecto | RL Pipeline | Forecasting Pipeline |
|---------|-------------|---------------------|
| **OHLCV Source** | TwelveData API | Investing.com (OFICIAL) |
| **OHLCV Table** | `usdcop_m5_ohlcv` | `bi.dim_daily_usdcop` |
| **Loader Method** | `load_5min()` | `load_daily()` |
| **Frequency** | 5-minute bars | Daily OHLCV |
| **Macro Table** | `macro_indicators_daily` | `macro_indicators_daily` |
| **Macro Expansion** | ffill to 5-min grid | Direct (already daily) |
| **Features Count** | 15 | 19 |
| **RSI Period** | 9 bars (intraday) | 14 days (daily) |
| **Use Case** | Intraday RL trading | Daily price forecasting |

---

## Archivos Clave Actualizados

| Archivo | Cambio | Versión |
|---------|--------|---------|
| `src/data/ohlcv_loader.py` | Added `load_daily()` method | 3.0.0 |
| `src/data/contracts.py` | Added table names, lineage info | 3.0.0 |
| `src/data/__init__.py` | Updated exports | 3.0.0 |
| `src/forecasting/engine.py` | Uses `load_daily()` now | 3.0.0 |
| `tests/unit/test_data_loaders.py` | 33 tests (all passing) | 3.0.0 |

---

## Validación

```bash
# Run all tests
pytest tests/unit/test_data_loaders.py -v

# Verify architecture
python -c "from src.data.contracts import get_data_lineage_info; print(get_data_lineage_info())"
```

## DEPRECATED

⚠️ `resample_to_daily()` está DEPRECATED para producción.
Use `load_daily()` para obtener valores oficiales de Investing.com.

```python
# DEPRECATED (solo para fallback/validación)
df_5min = loader.load_5min("2024-01-01", "2024-12-31")
df_daily = loader.resample_to_daily(df_5min)  # ⚠️ Warning raised

# CORRECTO (producción)
df_daily = loader.load_daily("2024-01-01", "2024-12-31")  # ✅ Official values
```
           ┌───────────┴───────────┐     ┌───────────┴───────────┐
           │                       │     │                       │
           ▼                       ▼     ▼                       ▼
    ┌─────────────┐         ┌─────────────┐              ┌─────────────┐
    │ RL Training │         │ Forecasting │              │ Inference   │
    │  (5-min)    │         │   (daily)   │              │   (both)    │
    └─────────────┘         └─────────────┘              └─────────────┘
```

---

## Plan de Implementación

### Fase 1: Core Infrastructure (CREAR)

#### 1.1 Estructura de Directorios

```
src/data/
├── __init__.py                 # Exports públicos
├── ohlcv_loader.py             # UnifiedOHLCVLoader
├── macro_loader.py             # UnifiedMacroLoader
├── calendar.py                 # TradingCalendar (horarios/festivos)
└── contracts.py                # Mappings de columnas SSOT
```

#### 1.2 Archivos a Crear

| Archivo | Propósito | LOC Est. |
|---------|-----------|----------|
| `src/data/__init__.py` | Exports del módulo | ~20 |
| `src/data/contracts.py` | Mapping de columnas DB → nombres amigables | ~80 |
| `src/data/calendar.py` | Horarios de trading COT, festivos | ~100 |
| `src/data/ohlcv_loader.py` | Carga OHLCV desde DB/CSV | ~200 |
| `src/data/macro_loader.py` | Carga macro + resample 5-min | ~250 |
| `database/migrations/026_v_macro_unified.sql` | View SQL unificada | ~50 |
| `tests/unit/test_data_loaders.py` | Tests unitarios | ~200 |

**Total:** ~900 LOC nuevas

---

### Fase 2: View SQL (CREAR)

#### 2.1 `v_macro_unified`

View que normaliza nombres de columnas para ambos pipelines:

```sql
CREATE OR REPLACE VIEW public.v_macro_unified AS
SELECT
    fecha,
    -- FOREX (friendly names)
    fxrt_index_dxy_usa_d_dxy AS dxy,
    fxrt_spot_usdmxn_mex_d_usdmxn AS usdmxn,
    fxrt_spot_usdclp_chl_d_usdclp AS usdclp,
    -- COMMODITIES
    comm_oil_wti_glb_d_wti AS wti,
    comm_oil_brent_glb_d_brent AS brent,
    comm_metal_gold_glb_d_gold AS gold,
    comm_agri_coffee_glb_d_coffee AS coffee,
    -- VOLATILITY
    volt_vix_usa_d_vix AS vix,
    -- CREDIT RISK
    crsk_spread_embi_col_d_embi AS embi,
    -- INTEREST RATES
    finc_bond_yield10y_usa_d_ust10y AS ust10y,
    finc_bond_yield2y_usa_d_dgs2 AS ust2y,
    finc_bond_yield10y_col_d_col10y AS col10y,
    finc_rate_ibr_overnight_col_d_ibr AS ibr,
    -- POLICY RATES
    polr_policy_rate_col_d_tpm AS tpm,
    polr_fed_funds_usa_m_fedfunds AS fedfunds,
    -- EQUITY
    eqty_index_colcap_col_d_colcap AS colcap,
    -- CALCULATED
    finc_bond_yield10y_col_d_col10y - finc_bond_yield10y_usa_d_ust10y AS rate_spread
FROM macro_indicators_daily
WHERE fecha >= '2015-01-01'
ORDER BY fecha;
```

---

### Fase 3: Modificar Scripts Existentes

#### 3.1 `scripts/build_forecasting_dataset_aligned.py`

**Cambios:**
- Eliminar imports de yfinance
- Usar `UnifiedOHLCVLoader` y `UnifiedMacroLoader`
- Mantener lógica de 19 features SSOT

```python
# ANTES (líneas 74-144):
import yfinance as yf
def fetch_usdcop_ohlcv(start_date, end_date):
    ticker = yf.Ticker("COP=X")
    ...

# DESPUÉS:
from src.data import UnifiedOHLCVLoader, UnifiedMacroLoader

def fetch_data(start_date: str, end_date: str) -> pd.DataFrame:
    ohlcv_loader = UnifiedOHLCVLoader()
    macro_loader = UnifiedMacroLoader()

    # OHLCV daily (resample de 5-min)
    ohlcv_5min = ohlcv_loader.load_5min(start_date, end_date)
    ohlcv_daily = ohlcv_loader.resample_to_daily(ohlcv_5min)

    # Macro daily (directo de DB)
    macro_daily = macro_loader.load_daily(start_date, end_date)

    return ohlcv_daily.merge(macro_daily, on='date', how='left')
```

#### 3.2 `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py`

**Cambios:**
- Eliminar lectura de CSV comprimido
- Usar `UnifiedOHLCVLoader` y `UnifiedMacroLoader`

```python
# ANTES (líneas 140-157):
OHLCV_BACKUP = Path("data/backups/usdcop_m5_ohlcv_*.csv.gz")
df_ohlcv = pd.read_csv(gzip.open(OHLCV_BACKUP))
df_macro = pd.read_csv("05_resampling/output/MACRO_5MIN_CONSOLIDATED.csv")

# DESPUÉS:
from src.data import UnifiedOHLCVLoader, UnifiedMacroLoader

ohlcv_loader = UnifiedOHLCVLoader()
macro_loader = UnifiedMacroLoader()

df_ohlcv = ohlcv_loader.load_5min(START_DATE, END_DATE)
df_macro = macro_loader.load_5min(START_DATE, END_DATE)  # Ya incluye resample+ffill
```

#### 3.3 `src/forecasting/engine.py`

**Cambios:**
- Agregar opción `use_db: bool = True`
- Usar loaders unificados cuando `use_db=True`

---

### Fase 4: Modificar DAGs

#### 4.1 `airflow/dags/forecast_l0_daily_data.py`

**Cambios:**
- Eliminar fetch desde TwelveData/yfinance para macro
- Usar datos existentes en `usdcop_m5_ohlcv` (resampleados)

#### 4.2 `airflow/dags/l3b_forecasting_training.py`

**Cambios:**
- Usar loaders unificados para cargar dataset

---

## Contracts SSOT

### Macro Columns Mapping

```python
# src/data/contracts.py

MACRO_DB_TO_FRIENDLY = {
    # FOREX
    "fxrt_index_dxy_usa_d_dxy": "dxy",
    "fxrt_spot_usdmxn_mex_d_usdmxn": "usdmxn",
    "fxrt_spot_usdclp_chl_d_usdclp": "usdclp",
    # COMMODITIES
    "comm_oil_wti_glb_d_wti": "wti",
    "comm_oil_brent_glb_d_brent": "brent",
    "comm_metal_gold_glb_d_gold": "gold",
    # VOLATILITY
    "volt_vix_usa_d_vix": "vix",
    # CREDIT
    "crsk_spread_embi_col_d_embi": "embi",
    # RATES
    "finc_bond_yield10y_usa_d_ust10y": "ust10y",
    "finc_bond_yield10y_col_d_col10y": "col10y",
}

# Columnas usadas por RL (subset)
RL_MACRO_COLUMNS = ["dxy", "vix", "embi", "brent", "ust10y", "usdmxn", "rate_spread"]

# Columnas usadas por Forecasting (subset)
FORECASTING_MACRO_COLUMNS = ["dxy", "wti"]

# Features SSOT Forecasting (19 en orden)
FORECASTING_FEATURES = (
    "close", "open", "high", "low",
    "return_1d", "return_5d", "return_10d", "return_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "rsi_14d", "ma_ratio_20d", "ma_ratio_50d",
    "day_of_week", "month", "is_month_end",
    "dxy_close_lag1", "oil_close_lag1",
)
```

---

## Flujos de Datos

### RL Training (5-min)

```
PostgreSQL                    Python Loaders                  Output
┌──────────────────┐        ┌───────────────────┐          ┌──────────┐
│ usdcop_m5_ohlcv  │───────▶│ UnifiedOHLCVLoader│─────────▶│ OHLCV    │
│ (5-min bars)     │        │ .load_5min()      │          │ (5-min)  │
└──────────────────┘        └───────────────────┘          └────┬─────┘
                                                                │
┌──────────────────┐        ┌───────────────────┐          ┌────▼─────┐
│ v_macro_unified  │───────▶│ UnifiedMacroLoader│─────────▶│ MACRO    │
│ (daily view)     │        │ .load_5min()      │          │ (5-min   │
└──────────────────┘        │ (resample+ffill)  │          │ ffilled) │
                            └───────────────────┘          └────┬─────┘
                                                                │
                                                           ┌────▼─────┐
                                                           │ MERGE    │
                                                           │merge_asof│
                                                           └────┬─────┘
                                                                │
                                                           ┌────▼─────┐
                                                           │ RL DS    │
                                                           │(21 feat) │
                                                           └──────────┘
```

### Forecasting Training (daily)

```
PostgreSQL                    Python Loaders                  Output
┌──────────────────┐        ┌───────────────────┐          ┌──────────┐
│ usdcop_m5_ohlcv  │───────▶│ UnifiedOHLCVLoader│─────────▶│ OHLCV    │
│ (5-min bars)     │        │ .load_5min()      │          │ (daily   │
│                  │        │ .resample_daily() │          │resample) │
└──────────────────┘        └───────────────────┘          └────┬─────┘
                                                                │
┌──────────────────┐        ┌───────────────────┐          ┌────▼─────┐
│ v_macro_unified  │───────▶│ UnifiedMacroLoader│─────────▶│ MACRO    │
│ (daily view)     │        │ .load_daily()     │          │ (daily)  │
└──────────────────┘        └───────────────────┘          └────┬─────┘
                                                                │
                                                           ┌────▼─────┐
                                                           │ MERGE    │
                                                           │ on date  │
                                                           └────┬─────┘
                                                                │
                                                           ┌────▼─────┐
                                                           │Forecast  │
                                                           │(19 feat) │
                                                           └──────────┘
```

---

## Resumen de Archivos

### CREAR (7 archivos)

| Archivo | Propósito |
|---------|-----------|
| `src/data/__init__.py` | Exports del módulo |
| `src/data/contracts.py` | Mappings SSOT de columnas |
| `src/data/calendar.py` | Trading calendar COT |
| `src/data/ohlcv_loader.py` | UnifiedOHLCVLoader |
| `src/data/macro_loader.py` | UnifiedMacroLoader |
| `database/migrations/026_v_macro_unified.sql` | View SQL |
| `tests/unit/test_data_loaders.py` | Tests unitarios |

### MODIFICAR (5 archivos)

| Archivo | Cambio |
|---------|--------|
| `scripts/build_forecasting_dataset_aligned.py` | Usar loaders unificados |
| `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py` | Usar loaders unificados |
| `src/forecasting/engine.py` | Agregar opción use_db |
| `airflow/dags/forecast_l0_daily_data.py` | Usar resample de 5-min |
| `airflow/dags/l3b_forecasting_training.py` | Usar loaders |

### DEPRECAR (no eliminar)

| Archivo | Razón |
|---------|-------|
| `data/pipeline/05_resampling/output/*.csv` | Generados on-demand por loader |

---

## Validación

### Checklist

- [ ] UnifiedOHLCVLoader carga desde DB correctamente
- [ ] UnifiedMacroLoader carga desde v_macro_unified
- [ ] Resample 5-min produce mismos valores que CSV legacy (±1e-6)
- [ ] Forecasting dataset tiene DXY/WTI de misma fuente que RL
- [ ] No hay lookahead bias (merge_asof direction='backward')
- [ ] ffill tiene límite (144 bars = 12 horas max)
- [ ] Tests unitarios pasan
- [ ] Tests E2E para ambos pipelines pasan

### Tests de Consistencia

```python
# Verificar que ambos pipelines usan mismos valores macro
def test_macro_consistency():
    ohlcv = UnifiedOHLCVLoader()
    macro = UnifiedMacroLoader()

    # RL: macro a 5-min
    macro_5min = macro.load_5min("2024-01-01", "2024-12-31")

    # Forecasting: macro daily
    macro_daily = macro.load_daily("2024-01-01", "2024-12-31")

    # Verificar que los valores de un día específico coinciden
    date = "2024-06-15"
    dxy_daily = macro_daily[macro_daily['date'] == date]['dxy'].iloc[0]
    dxy_5min = macro_5min[macro_5min['date'] == date]['dxy'].iloc[0]

    assert abs(dxy_daily - dxy_5min) < 1e-6
```

---

## Beneficios Esperados

| Beneficio | Impacto |
|-----------|---------|
| **SSOT** | Una sola fuente de macro data (PostgreSQL) |
| **DRY** | Elimina ~40% de código duplicado |
| **Consistencia** | RL y Forecasting usan exactamente los mismos valores macro |
| **Mantenibilidad** | Un solo lugar para actualizar lógica de data loading |
| **Trazabilidad** | Lineage claro desde DB → Dataset → Model |
| **Fallback** | CSV backup cuando DB no disponible |

---

**Autor**: Claude Code
**Fecha**: 2026-01-22
**Principio**: SSOT + DRY para Data Loading
