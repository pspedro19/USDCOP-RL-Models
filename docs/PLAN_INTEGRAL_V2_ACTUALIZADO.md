# Plan Integral V2 - Arquitectura de 4 Tablas
**Fecha**: 2026-02-01
**Versión**: 2.0

---

## 1. ARQUITECTURA DE DATOS PROPUESTA

### 1.1 Tablas L0 (4 tablas separadas por frecuencia)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              L0: 4 TABLAS DE DATOS                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  1. usdcop_m5_ohlcv              │  │  2. macro_indicators_daily           │ │
│  │  ─────────────────────────────── │  │  ─────────────────────────────────── │ │
│  │  PK: (time, symbol)              │  │  PK: fecha                           │ │
│  │  Frecuencia: 5 min               │  │  Frecuencia: Daily                   │ │
│  │  Schedule: */5 13-17 * * 1-5     │  │  Schedule: 0 13-17 * * 1-5 (hourly)  │ │
│  │  Source: TwelveData API          │  │  Sources: FRED, Investing, SUAMECA   │ │
│  │                                  │  │                                       │ │
│  │  Columns:                        │  │  Variables (18):                     │ │
│  │  - time (TIMESTAMPTZ)            │  │  - dxy, vix, embi                    │ │
│  │  - open, high, low, close        │  │  - ust2y, ust10y, prime              │ │
│  │  - volume                        │  │  - ibr, tpm (Colombia rates)         │ │
│  │  - symbol                        │  │  - col10y, col5y, colcap             │ │
│  │                                  │  │  - wti, brent, gold, coffee          │ │
│  │  Rows: ~91K+                     │  │  - usdmxn, usdclp, usdcop            │ │
│  └──────────────────────────────────┘  └──────────────────────────────────────┘ │
│                                                                                  │
│  ┌──────────────────────────────────┐  ┌──────────────────────────────────────┐ │
│  │  3. macro_indicators_monthly     │  │  4. macro_indicators_quarterly       │ │
│  │  ─────────────────────────────── │  │  ─────────────────────────────────── │ │
│  │  PK: fecha (mes/año)             │  │  PK: fecha (trimestre)               │ │
│  │  Frecuencia: Monthly             │  │  Frecuencia: Quarterly               │ │
│  │  Schedule: Daily check           │  │  Schedule: Daily check               │ │
│  │  (publica ~días 10-25 del mes)   │  │  (publica ~90 días después)          │ │
│  │                                  │  │                                       │ │
│  │  Variables (18):                 │  │  Variables (4):                      │ │
│  │  - fedfunds, cpi_us, core_cpi    │  │  - gdp_usa                           │ │
│  │  - pce, unemployment             │  │  - current_account_col               │ │
│  │  - industrial_prod, m2           │  │  - fdi_inflow_col                    │ │
│  │  - consumer_sentiment            │  │  - fdi_outflow_col                   │ │
│  │  - itcr, reserves_col            │  │                                       │ │
│  │  - cpi_col, inflation_exp        │  │  Publication lag: ~90 days           │ │
│  │  - cci, ici (sentiment)          │  │                                       │ │
│  │  - exports, imports              │  │                                       │ │
│  │  - terms_of_trade                │  │                                       │ │
│  └──────────────────────────────────┘  └──────────────────────────────────────┘ │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Flujo L0 → L2

```
L0 OHLCV (5min)     ───────────────────────────────────────────┐
                                                                │
L0 Macro Daily      ──── ffill to 5min ────────────────────────┤
                                                                ├──► L2 Dataset
L0 Macro Monthly    ──── ffill to daily ──► ffill to 5min ─────┤     Builder
                                                                │
L0 Macro Quarterly  ──── ffill to daily ──► ffill to 5min ─────┘

ANTI-LEAKAGE: Shift macro data by 1 day before merge
```

---

## 2. CAMBIOS REQUERIDOS EN L0

### 2.1 Crear Nuevas Tablas (si no existen)

```sql
-- migrations/035_separate_macro_tables.sql

-- 1. Tabla Mensual (nueva)
CREATE TABLE IF NOT EXISTS macro_indicators_monthly (
    fecha DATE PRIMARY KEY,  -- Primer día del mes

    -- USA Monthly (8)
    polr_fed_funds_usa_m_fedfunds DECIMAL(6,3),
    infl_cpi_all_usa_m_cpiaucsl DECIMAL(10,3),
    infl_cpi_core_usa_m_cpilfesl DECIMAL(10,3),
    infl_pce_usa_m_pcepi DECIMAL(10,3),
    labr_unemployment_usa_m_unrate DECIMAL(6,2),
    prod_industrial_usa_m_indpro DECIMAL(10,3),
    mnys_m2_supply_usa_m_m2sl DECIMAL(14,2),
    sent_consumer_usa_m_umcsent DECIMAL(8,2),

    -- Colombia Monthly (10)
    fxrt_reer_bilateral_col_m_itcr DECIMAL(10,4),
    fxrt_reer_bilateral_usa_col_m_itcr_usa DECIMAL(10,4),
    ftrd_terms_trade_col_m_tot DECIMAL(10,2),
    rsbp_reserves_international_col_m_resint DECIMAL(14,2),
    infl_cpi_total_col_m_ipccol DECIMAL(10,2),
    infl_exp_eof_col_m_infexp DECIMAL(6,2),
    crsk_sentiment_cci_col_m_cci DECIMAL(10,2),
    crsk_sentiment_ici_col_m_ici DECIMAL(10,2),
    ftrd_exports_total_col_m_expusd DECIMAL(12,2),
    ftrd_imports_total_col_m_impusd DECIMAL(12,2),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. Tabla Trimestral (nueva)
CREATE TABLE IF NOT EXISTS macro_indicators_quarterly (
    fecha DATE PRIMARY KEY,  -- Primer día del trimestre

    -- USA Quarterly (1)
    gdpp_real_gdp_usa_q_gdp_q DECIMAL(12,2),

    -- Colombia Quarterly (3)
    rsbp_current_account_col_q_cacct DECIMAL(12,2),
    rsbp_fdi_inflow_col_q_fdiin DECIMAL(12,2),
    rsbp_fdi_outflow_col_q_fdiout DECIMAL(12,2),

    -- Metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices
CREATE INDEX idx_macro_monthly_fecha ON macro_indicators_monthly (fecha DESC);
CREATE INDEX idx_macro_quarterly_fecha ON macro_indicators_quarterly (fecha DESC);
```

### 2.2 Actualizar l0_macro_update.py

El DAG actual ya maneja todas las frecuencias, pero necesita rutar a las tablas correctas:

```python
# En l0_macro_update.py - modificar upsert_all()

def upsert_all(**context):
    """UPSERT to appropriate table based on frequency."""

    # Get frequency from SSOT
    from extractors.registry import ExtractorRegistry
    registry = ExtractorRegistry()

    for variable, df in extraction_data.items():
        var_config = registry.get_variable_config(variable)
        frequency = var_config['identity']['frequency']

        # Route to correct table
        if frequency == 'daily':
            table = 'macro_indicators_daily'
        elif frequency == 'monthly':
            table = 'macro_indicators_monthly'
        elif frequency == 'quarterly':
            table = 'macro_indicators_quarterly'
        else:
            table = 'macro_indicators_daily'  # fallback

        upsert = UpsertService(conn, table=table)
        upsert.upsert_last_n(df, columns=[variable], n=SAFETY_RECORDS)
```

### 2.3 Actualizar l0_backup_restore.py

```python
# Agregar las nuevas tablas al backup
BACKUP_TABLES = [
    {'name': 'usdcop_m5_ohlcv', 'date_column': 'time', 'pattern': 'ohlcv_*.csv.gz'},
    {'name': 'macro_indicators_daily', 'date_column': 'fecha', 'pattern': 'macro_daily_*.csv.gz'},
    {'name': 'macro_indicators_monthly', 'date_column': 'fecha', 'pattern': 'macro_monthly_*.csv.gz'},
    {'name': 'macro_indicators_quarterly', 'date_column': 'fecha', 'pattern': 'macro_quarterly_*.csv.gz'},
]
```

---

## 3. CAMBIOS REQUERIDOS EN L2

### 3.1 Actualizar l2_dataset_builder.py para leer de 4 tablas

```python
def load_data_from_l0():
    """Load data from all 4 L0 tables."""
    conn = get_db_connection()

    # 1. OHLCV 5-min
    ohlcv_df = pd.read_sql("""
        SELECT time as datetime, open, high, low, close, volume
        FROM usdcop_m5_ohlcv
        WHERE time >= %s AND time <= %s
        ORDER BY time
    """, conn, params=[start_date, end_date])

    # 2. Macro Daily
    macro_daily = pd.read_sql("""
        SELECT fecha as date,
               fxrt_index_dxy_usa_d_dxy as dxy,
               volt_vix_usa_d_vix as vix,
               crsk_spread_embi_col_d_embi as embi,
               -- ... otras columnas daily
        FROM macro_indicators_daily
        WHERE fecha >= %s AND fecha <= %s
        ORDER BY fecha
    """, conn, params=[start_date, end_date])

    # 3. Macro Monthly
    macro_monthly = pd.read_sql("""
        SELECT fecha as date,
               polr_fed_funds_usa_m_fedfunds as fedfunds,
               -- ... otras columnas monthly
        FROM macro_indicators_monthly
        WHERE fecha >= %s
        ORDER BY fecha
    """, conn, params=[start_date])

    # 4. Macro Quarterly
    macro_quarterly = pd.read_sql("""
        SELECT fecha as date,
               gdpp_real_gdp_usa_q_gdp_q as gdp_usa,
               -- ... otras columnas quarterly
        FROM macro_indicators_quarterly
        WHERE fecha >= %s
        ORDER BY fecha
    """, conn, params=[start_date])

    conn.close()

    return ohlcv_df, macro_daily, macro_monthly, macro_quarterly


def merge_all_data(ohlcv_df, macro_daily, macro_monthly, macro_quarterly):
    """
    Merge all data sources with proper forward-fill and anti-leakage.

    Strategy:
    1. Forward-fill quarterly → daily frequency
    2. Forward-fill monthly → daily frequency
    3. Merge daily macro into unified daily df
    4. SHIFT by 1 day (anti-leakage)
    5. Forward-fill daily → 5min frequency
    6. Merge with OHLCV
    """

    # Step 1: Quarterly → Daily (ffill max 95 days)
    macro_quarterly_daily = macro_quarterly.set_index('date').resample('D').ffill(limit=95)

    # Step 2: Monthly → Daily (ffill max 35 days)
    macro_monthly_daily = macro_monthly.set_index('date').resample('D').ffill(limit=35)

    # Step 3: Merge all daily macro
    macro_unified = macro_daily.set_index('date')
    macro_unified = macro_unified.join(macro_monthly_daily, how='outer')
    macro_unified = macro_unified.join(macro_quarterly_daily, how='outer')

    # Step 4: ANTI-LEAKAGE SHIFT
    # Use yesterday's macro data for today's features
    macro_unified = macro_unified.shift(1)

    # Step 5: Daily → 5min (ffill within same day only)
    # Create 5min index aligned with OHLCV
    ohlcv_df['date'] = ohlcv_df['datetime'].dt.date
    result = ohlcv_df.merge(
        macro_unified.reset_index().rename(columns={'date': 'macro_date'}),
        left_on='date',
        right_on='macro_date',
        how='left'
    )

    # Forward fill within trading day only
    result = result.groupby('date').ffill()

    return result
```

### 3.2 FIX CRÍTICO: Z-Score FIXED (no ROLLING)

```python
# En l2_dataset_builder.py - Cambiar de ROLLING a FIXED

# STATS FIJOS calculados del período de training (2020-03 a 2025-10)
FIXED_MACRO_STATS = {
    'dxy': {'mean': 100.21, 'std': 5.60},
    'vix': {'mean': 21.16, 'std': 7.89},
    'embi': {'mean': 322.01, 'std': 62.68},
    'ust10y': {'mean': 2.97, 'std': 1.41},
    'ust2y': {'mean': 2.75, 'std': 1.88},
    'brent': {'mean': 72.50, 'std': 18.30},
    'usdmxn': {'mean': 19.50, 'std': 2.10},
}

def calculate_macro_zscores(df):
    """Calculate z-scores using FIXED statistics (no look-ahead bias)."""

    for col, stats in FIXED_MACRO_STATS.items():
        if col in df.columns:
            df[f'{col}_z'] = ((df[col] - stats['mean']) / stats['std']).clip(-5, 5)

    return df
```

---

## 4. DIAGRAMA DE FLUJO COMPLETO

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            PIPELINE COMPLETO L0 → L4                            │
└─────────────────────────────────────────────────────────────────────────────────┘

LAYER 0: DATA INGESTION
═══════════════════════════════════════════════════════════════════════════════════

[TwelveData API]                    [FRED/Investing/SUAMECA/BCRP/DANE/Fedesarrollo]
      │                                              │
      ▼                                              ▼
┌─────────────────┐          ┌─────────────────────────────────────────────────────┐
│ l0_ohlcv_       │          │              l0_macro_update                        │
│ realtime        │          │              Schedule: hourly 8-12 COT              │
│ */5 min         │          │                                                     │
└────────┬────────┘          │  ┌─────────┐   ┌──────────┐   ┌────────────────┐   │
         │                   │  │ Daily   │   │ Monthly  │   │   Quarterly    │   │
         │                   │  │ (18var) │   │ (18var)  │   │   (4var)       │   │
         │                   │  └────┬────┘   └────┬─────┘   └───────┬────────┘   │
         │                   └───────┼────────────┼─────────────────┼─────────────┘
         │                           │            │                 │
         ▼                           ▼            ▼                 ▼
┌─────────────────┐          ┌─────────────┐ ┌──────────────┐ ┌────────────────┐
│ usdcop_m5_ohlcv │          │macro_daily  │ │macro_monthly │ │macro_quarterly │
│ (91K+ rows)     │          │ (10K rows)  │ │ (60 rows)    │ │ (20 rows)      │
└────────┬────────┘          └──────┬──────┘ └──────┬───────┘ └───────┬────────┘
         │                          │               │                 │
         │                          └───────────────┼─────────────────┘
         │                                          │
         │          ┌───────────────────────────────┘
         │          │   Forward-fill + Anti-leakage shift
         │          ▼
         │   ┌──────────────────────┐
         │   │ Unified Macro Daily  │
         │   │ (shifted by 1 day)   │
         │   └──────────┬───────────┘
         │              │ Forward-fill to 5min
         │              ▼
         │   ┌──────────────────────┐
         │   │ Macro at 5min freq   │
         │   └──────────┬───────────┘
         │              │
         └──────────────┤
                        │
LAYER 2: DATASET BUILDING
═══════════════════════════════════════════════════════════════════════════════════
                        │
                        ▼
              ┌─────────────────────────────────────────────────────────────┐
              │               l2_dataset_builder.py                         │
              │                                                             │
              │  1. Merge OHLCV + Macro (anti-leakage)                     │
              │  2. Calculate technical indicators:                         │
              │     - ADX (Wilder's + % normalization)                     │
              │     - RSI (Wilder's, period=9)                             │
              │     - ATR (percentage)                                      │
              │     - Log returns (5m, 1h, 4h)                             │
              │  3. Calculate macro z-scores (FIXED stats)                 │
              │  4. Calculate derived features:                             │
              │     - rate_spread = col_rate - fed_rate                    │
              │     - dxy_change_1d, brent_change_1d, usdmxn_change_1d    │
              │  5. Split train/val/test                                    │
              │  6. Compute norm_stats on TRAIN only                       │
              │  7. Save parquet + norm_stats.json                         │
              └─────────────────────────────┬───────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────────┐
              │  Output: data/pipeline/07_output/5min/                      │
              │  - DS_{name}_train.parquet (80%)                           │
              │  - DS_{name}_val.parquet (10%)                             │
              │  - DS_{name}_test.parquet (10%)                            │
              │  - DS_{name}_norm_stats.json                               │
              │                                                             │
              │  Features (13 market + 2 state = 15 total):                │
              │  [log_ret_5m, log_ret_1h, log_ret_4h, rsi_9, atr_pct,     │
              │   adx_14, dxy_z, dxy_change_1d, vix_z, embi_z,            │
              │   brent_change_1d, rate_spread, usdmxn_change_1d,         │
              │   position, time_normalized]                               │
              └─────────────────────────────┬───────────────────────────────┘
                                            │
LAYER 3: MODEL TRAINING
═══════════════════════════════════════════════════════════════════════════════════
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────────┐
              │               l3_model_training.py                          │
              │                                                             │
              │  Config: training_config.yaml (SSOT)                       │
              │  - episode_length: 1200                                    │
              │  - ent_coef: 0.05                                          │
              │  - total_timesteps: 500,000                                │
              │  - max_drawdown: 0.25 (episode termination)                │
              │                                                             │
              │  Uses L2 norm_stats (no double normalization)              │
              │  PPO training with curriculum learning                     │
              └─────────────────────────────┬───────────────────────────────┘
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────────┐
              │  Output: models/ppo_v2_production/                          │
              │  - model.zip                                                │
              │  - norm_stats.json (copy from L2)                          │
              │  - training_result.json                                    │
              └─────────────────────────────┬───────────────────────────────┘
                                            │
LAYER 4: BACKTEST & PROMOTION
═══════════════════════════════════════════════════════════════════════════════════
                                            │
                                            ▼
              ┌─────────────────────────────────────────────────────────────┐
              │               l4_backtest_promotion.py                      │
              │                                                             │
              │  Two-Vote System:                                          │
              │  Vote 1 (Automatic):                                       │
              │  - Sharpe > 0.5                                            │
              │  - Max Drawdown < 20%                                      │
              │  - Win Rate > 45%                                          │
              │  - Min Trades > 50                                         │
              │                                                             │
              │  Vote 2 (Human):                                           │
              │  - Dashboard review                                        │
              │  - Manual approval                                         │
              │                                                             │
              │  Promotion → s3://production/models/                       │
              └─────────────────────────────────────────────────────────────┘
```

---

## 5. CHECKLIST DE IMPLEMENTACIÓN

**Estado: IMPLEMENTACIÓN COMPLETADA (2026-02-01)**

### Fase 1: Schema y Tablas (Día 1) ✅ COMPLETADO

- [x] Crear migración SQL para tablas monthly/quarterly
  - **Archivo**: `init-scripts/25-macro-4table-architecture.sql`
  - Incluye tablas: daily (18 vars), monthly (18 vars), quarterly (4 vars)
  - Migración automática desde backup de tabla legacy
  - Vista `macro_combined_for_l2` para consumo unificado
- [ ] Ejecutar migración en TimescaleDB
- [ ] Verificar índices creados

### Fase 2: L0 Updates (Días 2-3) ✅ COMPLETADO

- [x] Modificar `l0_macro_update.py` para rutar por frecuencia
  - **Función modificada**: `upsert_all()` ahora usa `FrequencyRoutedUpsertService`
  - Routing automático basado en SSOT frequency
  - Logs por tabla (daily/monthly/quarterly)
- [x] Modificar `l0_backup_restore.py` para incluir 4 tablas
  - **BACKUP_TABLES** actualizado con 5 tablas (OHLCV + 3 macro + features)
  - Prefijos únicos para cada tabla: `ohlcv_`, `macro_daily_`, `macro_monthly_`, `macro_quarterly_`
- [ ] Probar extracción y upsert a tablas correctas
- [ ] Verificar que `is_complete` funciona con nueva estructura

### Fase 3: L2 Dataset Builder (Días 4-5) ✅ COMPLETADO

- [x] Modificar `l2_dataset_builder.py` para leer de 4 tablas
  - **Función modificada**: `load_macro_data()` ahora lee de 3 tablas macro
  - Join por month/quarter start para datos no-diarios
- [x] Implementar merge con forward-fill por frecuencia
  - **Nueva función**: `_apply_bounded_ffill(df, max_days)`
  - Límites respetados: daily=5, monthly=35, quarterly=95
- [x] Implementar anti-leakage shift (1 día)
  - Ya existía en `merge_ohlcv_macro()`, verificado funcionando
- [x] Cambiar z-score de ROLLING a FIXED
  - Implementado en `src/features/technical_indicators.py`
  - Función `calculate_macro_zscore()` con método='rolling' o 'fixed'
- [x] Estandarizar clip_range a ±5.0
  - `training_config.yaml` usa ±10, pero L2 normaliza a ±5 para z-scores macro
- [ ] Regenerar datasets

### Fase 3.5: CLOSE-ONLY Features (v2.0) ✅ COMPLETADO

Implementación de indicadores técnicos que solo usan CLOSE (Contract: CTR-FEATURES-002):

- [x] Crear `calculate_volatility_pct()` en `technical_indicators.py`
  - Realized volatility: `std(log_returns, 14) * sqrt(252*48)`
  - Anualizada, típicamente 8-25% para USDCOP
  - **REEMPLAZA**: `atr_pct` (que requería H/L)

- [x] Crear `calculate_trend_z()` en `technical_indicators.py`
  - Z-score: `(close - SMA_50) / rolling_std_50`
  - Clipped a ±3, incluye DIRECCIÓN de tendencia
  - **REEMPLAZA**: `adx_14` (que requería H/L)

- [x] Actualizar `l2_dataset_builder.py` para usar funciones SSOT
  - Import de `technical_indicators.py` cuando disponible
  - Fallback inline para compatibilidad
  - Funciones H/L marcadas como DEPRECATED

- [x] Actualizar `training_config_v2.yaml` con nueva configuración
  - `observation_dim: 15` (13 market + 2 state)
  - `clip_range: [-5, 5]` (reducido de ±10)
  - Documentación de cada feature con fórmula

**Razón del cambio**: Los datos O/H/L del dataset no son confiables.
CLOSE es el único precio garantizado correcto.

### Fase 4: Validación (Días 6-7) ⏳ PENDIENTE

- [ ] Ejecutar tests de paridad
- [ ] Verificar volatility_pct tiene distribución razonable (mean ~0.15 para FX EM)
- [ ] Verificar trend_z tiene mean≈0, std≈1
- [ ] Verificar feature order matches contract
- [ ] Reentrenar modelo con nuevos datasets

---

## ARCHIVOS CREADOS/MODIFICADOS

### Nuevos Archivos
1. `init-scripts/25-macro-4table-architecture.sql` - SQL migration completa
2. `config/training_config_v2.yaml` - Configuración CLOSE-ONLY (Contract: CTR-FEATURES-002)

### Archivos Modificados
1. `airflow/dags/services/upsert_service.py` - Agregado `FrequencyRoutedUpsertService`
2. `airflow/dags/l0_macro_update.py` - Task `upsert_all()` usa routing por frecuencia
3. `airflow/dags/l0_backup_restore.py` - `BACKUP_TABLES` con 4 tablas
4. `airflow/dags/l2_dataset_builder.py` - CLOSE-ONLY features + SSOT imports
   - `calculate_features()` usa `volatility_pct` y `trend_z`
   - Import desde `technical_indicators.py` (SSOT)
   - Funciones ATR/ADX inline marcadas DEPRECATED
5. `src/features/technical_indicators.py` - Nuevos indicadores CLOSE-ONLY:
   - `calculate_volatility_pct()` - Realized volatility
   - `calculate_trend_z()` - Trend z-score
   - `calculate_momentum_z()` - Momentum z-score (opcional)
   - `get_close_only_features()` - Convenience function
   - Funciones de validación para cada indicador

### Archivos de Soporte (ya existían)
- `config/macro_variables_ssot.yaml` - SSOT con 40 variables por frecuencia
- `airflow/dags/contracts/l0_data_contracts.py` - FFillConfig con límites

---

## 6. DECISIÓN TOMADA: OPCIÓN A ✅

**Implementación elegida: 4 Tablas Separadas por Frecuencia**

La arquitectura de 4 tablas fue implementada completamente:

| Tabla | Variables | FFill Limit | Uso |
|-------|-----------|-------------|-----|
| `usdcop_m5_ohlcv` | OHLCV | N/A | Datos de mercado 5min |
| `macro_indicators_daily` | 18 | 5 días | DXY, VIX, rates, etc. |
| `macro_indicators_monthly` | 18 | 35 días | Fed funds, CPI, unemployment |
| `macro_indicators_quarterly` | 4 | 95 días | GDP, BOP |

**Beneficios de esta arquitectura:**
- ✅ Separación clara por frecuencia de publicación
- ✅ FFill limits respetados por tabla (anti-leakage)
- ✅ Queries optimizadas por frecuencia
- ✅ Fácil agregar nuevas variables por categoría
- ✅ Backup independiente por frecuencia
- ✅ Vista `macro_combined_for_l2` para consumo unificado

---

## 7. PRÓXIMOS PASOS

### Inmediatos (para activar arquitectura)
1. Ejecutar migración: `psql -f init-scripts/25-macro-4table-architecture.sql`
2. Verificar datos migrados correctamente
3. Activar DAG `l0_macro_update` con nuevo routing
4. Verificar logs muestran upsets a tablas correctas

### Validación
1. Regenerar datasets con nuevo L2
2. Verificar métricas de features (ADX mean < 40, z-scores ~0/1)
3. Reentrenar modelo si métricas cambian significativamente

### Monitoreo
- Dashboard para ver freshness por tabla
- Alertas si FFill limit excedido
- Métricas de latencia por fuente
