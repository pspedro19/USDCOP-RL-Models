# USD/COP RL Trading System - Database Schema

**Version: 3.1 (Migration V14)**

Este directorio contiene los scripts SQL completos y organizados para el sistema de inferencia en tiempo real del modelo PPO.

---

## Documentos de Referencia (LA BIBLIA)

| Documento | Ubicación | Secciones Relevantes |
|-----------|-----------|---------------------|
| **ARQUITECTURA INTEGRAL V3** | `docs/ARQUITECTURA_INTEGRAL_V3.md` | Sección 4 (líneas 297-389), Sección 21 (2800+), Sección 11.2 |
| **MAPEO MIGRACIÓN** | `docs/MAPEO_MIGRACION_BIDIRECCIONAL.md` | ERRATA (42-44), Parte 2.2 (334-346), Features (61-66) |
| **Feature Config (SSOT)** | `config/feature_config.json` | Normalization stats, feature order |

---

## Estructura de Directorios

```
database/
├── schemas/
│   ├── 01_core_tables.sql       # DDL: macro_indicators_daily + dw.fact_rl_inference
│   └── 02_inference_view.sql    # Vista materializada con 9 features SQL
├── migrations/
│   └── 001_add_macro_table.sql  # Migración con datos de macro_ohlcv
└── README.md                    # Este archivo
```

---

## Arquitectura: 3 Tablas + 1 Vista

```
┌─────────────────────────┐     ┌────────────────────────────┐
│   usdcop_m5_ohlcv       │     │   macro_indicators_daily   │
│   (TimescaleDB)         │     │   (Regular Table)          │
│   5-min OHLCV bars      │     │   37 macro variables       │
│   SSOT: init-scripts/   │     │   Update: 3x/día           │
└───────────┬─────────────┘     └──────────────┬─────────────┘
            │                                   │
            │         INNER JOIN                │
            │    (NOT LEFT JOIN - ERRATA)       │
            └──────────────┬────────────────────┘
                           │
                           ▼
            ┌──────────────────────────────┐
            │   inference_features_5m      │
            │   (Materialized View)        │
            │   9 features SQL + 4 NULL    │
            │   Refresh: cada 5 min        │
            └──────────────┬───────────────┘
                           │
                           │  Python Service
                           │  (+4 features: RSI, ATR, ADX, USDMXN)
                           ▼
            ┌──────────────────────────────┐
            │   dw.fact_rl_inference       │
            │   (TimescaleDB)              │
            │   Log de inferencias PPO     │
            │   13 features + 2 estado     │
            └──────────────────────────────┘
```

---

## Tablas del Sistema

### Tabla 1: usdcop_m5_ohlcv (YA EXISTE)

**Ubicación**: `init-scripts/01-essential-usdcop-init.sql`

| Columna | Tipo | Descripción |
|---------|------|-------------|
| time | TIMESTAMPTZ | PK, partición hypertable |
| symbol | TEXT | 'USD/COP' |
| open, high, low, close | DECIMAL(12,6) | Precios OHLC |
| volume | BIGINT | Volumen |
| source | TEXT | 'twelvedata' |

**Tipo**: TimescaleDB Hypertable
**Granularidad**: 5 minutos
**Horario**: Lunes-Viernes, 08:00-12:55 COT

---

### Tabla 2: macro_indicators_daily (37 columnas macro + 4 metadata)

**Script**: `database/schemas/01_core_tables.sql`

#### Columnas por Grupo (37 variables macro)

| # | Columna | Tipo | Grupo | Uso en Features |
|---|---------|------|-------|-----------------|
| 1 | **dxy** | NUMERIC(10,4) | Dollar Index | dxy_z, dxy_change_1d |
| 2 | **vix** | NUMERIC(10,4) | Volatility | vix_z |
| 3 | **embi** | NUMERIC(10,4) | Country Risk | embi_z |
| 4 | **brent** | NUMERIC(10,4) | Commodities | brent_change_1d |
| 5 | wti | NUMERIC(10,4) | Commodities | - |
| 6 | gold | NUMERIC(10,4) | Commodities | - |
| 7 | coffee | NUMERIC(10,4) | Commodities | - |
| 8 | fed_funds | NUMERIC(8,4) | Policy Rates | - |
| 9 | **treasury_2y** | NUMERIC(8,4) | Fixed Income | rate_spread |
| 10 | **treasury_10y** | NUMERIC(8,4) | Fixed Income | rate_spread |
| 11 | prime_rate | NUMERIC(8,4) | Fixed Income | - |
| 12 | tpm_colombia | NUMERIC(8,4) | Policy Rates | - |
| 13 | ibr_overnight | NUMERIC(8,4) | Money Market | - |
| 14 | **usdmxn** | NUMERIC(10,4) | Exchange Rates | usdmxn_ret_1h |
| 15 | usdclp | NUMERIC(10,4) | Exchange Rates | - |
| 16 | bond_yield5y_col | NUMERIC(8,4) | Fixed Income | - |
| 17 | bond_yield10y_col | NUMERIC(8,4) | Fixed Income | - |
| 18 | colcap | NUMERIC(10,4) | Equity | - |
| 19 | itcr | NUMERIC(10,4) | Exchange Rates | - |
| 20 | cci_colombia | NUMERIC(10,4) | Sentiment | - |
| 21 | ici_colombia | NUMERIC(10,4) | Sentiment | - |
| 22 | ipc_colombia | NUMERIC(10,4) | Inflation | - |
| 23 | cpi_usa | NUMERIC(10,4) | Inflation | - |
| 24 | pce_usa | NUMERIC(10,4) | Inflation | - |
| 25 | exports_col | NUMERIC(14,2) | Trade | - |
| 26 | imports_col | NUMERIC(14,2) | Trade | - |
| 27 | terms_of_trade | NUMERIC(10,4) | Trade | - |
| 28 | ied_inflow | NUMERIC(14,2) | Balance of Payments | - |
| 29 | ied_outflow | NUMERIC(14,2) | Balance of Payments | - |
| 30 | current_account | NUMERIC(14,2) | Balance of Payments | - |
| 31 | reserves_intl | NUMERIC(14,2) | Balance of Payments | - |
| 32 | unemployment_usa | NUMERIC(8,4) | Labor | - |
| 33 | industrial_prod_usa | NUMERIC(10,4) | Production | - |
| 34 | m2_supply_usa | NUMERIC(14,2) | Monetary | - |
| 35 | consumer_sentiment | NUMERIC(10,4) | Sentiment | - |
| 36 | gdp_usa | NUMERIC(14,2) | GDP | - |
| 37 | usdcop_spot | NUMERIC(10,4) | Exchange Rates | - |

#### Metadata (4 columnas)

| Columna | Tipo | Descripción |
|---------|------|-------------|
| source | VARCHAR(100) | Fuente principal del update |
| is_complete | BOOLEAN | TRUE si campos críticos están llenos |
| created_at | TIMESTAMPTZ | Timestamp creación |
| updated_at | TIMESTAMPTZ | Timestamp actualización (auto-update) |

**Actualización**: 3 veces/día (07:55, 10:30, 12:00 COT)
**is_complete**: TRUE cuando dxy, vix, embi, brent, treasury_10y, treasury_2y están poblados

---

### Tabla 3: dw.fact_rl_inference

**Script**: `database/schemas/01_core_tables.sql`

#### Estructura de Columnas

| Sección | Columnas | Descripción |
|---------|----------|-------------|
| **Timing** (2) | timestamp_utc, timestamp_cot | Timestamps UTC y Colombia |
| **Model ID** (3) | model_id, model_version, fold_id | Identificación del modelo |
| **Features** (13) | log_ret_5m...usdmxn_ret_1h | 13 features del modelo |
| **State** (2) | position, time_normalized | Features de estado |
| **Output** (4) | action_raw, action_discretized, confidence, q_values | Salida del modelo |
| **Market** (4) | symbol, close_price, raw_return_5m, spread_bps | Contexto de mercado |
| **Portfolio Before** (3) | position_before, portfolio_value_before, log_portfolio_before | Estado previo |
| **Portfolio After** (3) | position_after, portfolio_value_after, log_portfolio_after | Estado posterior |
| **Costs** (3) | position_change, transaction_cost_bps, transaction_cost_usd | Costos de transacción |
| **Performance** (2) | reward, cumulative_reward | Métricas de rendimiento |
| **Metadata** (4) | latency_ms, inference_source, dag_run_id, created_at | Metadata |

**Total**: 43 columnas

**Tipo**: TimescaleDB Hypertable (chunk = 1 día)
**Granularidad**: Una fila por inferencia (cada 5 min durante mercado)

---

## Vista Materializada: inference_features_5m

**Script**: `database/schemas/02_inference_view.sql`

### Features: 13 totales (9 SQL + 4 Python)

| # | Feature | Calculado en | Fórmula | Clipping |
|---|---------|-------------|---------|----------|
| 1 | log_ret_5m | **SQL** | LN(close/LAG(close,1)) | [-0.05, 0.05] |
| 2 | log_ret_1h | **SQL** | LN(close/LAG(close,12)) | [-0.05, 0.05] |
| 3 | log_ret_4h | **SQL** | LN(close/LAG(close,48)) | [-0.05, 0.05] |
| 4 | **rsi_9** | Python | RSI(close, 9) | [0, 100] |
| 5 | **atr_pct** | Python | (ATR/close)*100 | - |
| 6 | **adx_14** | Python | ADX(high, low, close, 14) | [0, 100] |
| 7 | dxy_z | **SQL** | (dxy - 103) / 5 | [-4, 4] |
| 8 | dxy_change_1d | **SQL** | pct_change(dxy) | [-0.03, 0.03] |
| 9 | vix_z | **SQL** | (vix - 20) / 10 | [-4, 4] |
| 10 | embi_z | **SQL** | (embi - 300) / 100 | [-4, 4] |
| 11 | brent_change_1d | **SQL** | pct_change(brent) | [-0.10, 0.10] |
| 12 | rate_spread | **SQL** | treasury_10y - treasury_2y | sin clipping |
| 13 | **usdmxn_ret_1h** | Python | pct_change(usdmxn, 12) | [-0.10, 0.10] |

### Correcciones V14 (de MAPEO ERRATA)

| Corrección | Detalle | Línea MAPEO |
|------------|---------|-------------|
| hour_sin, hour_cos | **ELIMINADOS** (bajo valor predictivo) | 42 |
| JOIN type | **INNER JOIN** (no LEFT JOIN) | ARQUITECTURA 11.2 |
| usdmxn_ret_1h clip | Corregido a **[-0.10, 0.10]** | 43 |

### Columnas Auxiliares para Python

La vista incluye datos adicionales para calcular features técnicos en Python:
- `open`, `high`, `low`, `prev_close`, `prev_high`, `prev_low` (para ATR, ADX)
- `prev_close_9` (para RSI período 9)
- `usdmxn_raw`, `usdmxn_prev` (para usdmxn_ret_1h)
- `range_hl`, `sma_20`, `std_20` (auxiliares)

---

## Quick Start

### Instalación Nueva

```bash
# 1. Ejecutar schemas en orden
psql -h localhost -U postgres -d usdcop_trading \
  -f database/schemas/01_core_tables.sql

psql -h localhost -U postgres -d usdcop_trading \
  -f database/schemas/02_inference_view.sql

# 2. Verificar instalación
psql -h localhost -U postgres -d usdcop_trading -c "
  SELECT table_schema, table_name FROM information_schema.tables
  WHERE table_name IN ('macro_indicators_daily', 'fact_rl_inference');
"
```

### Migración desde Sistema Existente

```bash
# 1. Ejecutar migración (migra datos desde macro_ohlcv si existe)
psql -h localhost -U postgres -d usdcop_trading \
  -f database/migrations/001_add_macro_table.sql

# 2. Crear vista
psql -h localhost -U postgres -d usdcop_trading \
  -f database/schemas/02_inference_view.sql

# 3. Refresh inicial
psql -h localhost -U postgres -d usdcop_trading -c "
  SELECT refresh_inference_features();
"
```

---

## Funciones Auxiliares

| Función | Descripción | Ejemplo |
|---------|-------------|---------|
| `is_market_open()` | TRUE si mercado abierto | `SELECT is_market_open();` |
| `get_bar_number(ts)` | Número de barra (1-60) | `SELECT get_bar_number(NOW());` |
| `is_colombia_holiday(date)` | TRUE si festivo Colombia | `SELECT is_colombia_holiday('2025-12-25');` |
| `should_run_inference()` | Check completo | `SELECT * FROM should_run_inference();` |
| `refresh_inference_features()` | Refresh vista | `SELECT refresh_inference_features();` |

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. RAW DATA INGESTION                                           │
├─────────────────────────────────────────────────────────────────┤
│ TwelveData API → usdcop_m5_ohlcv (every 5min)                   │
│ TwelveData/FRED/BCRP → macro_indicators_daily (3x daily)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. FEATURE TRANSFORMATION (SQL - 9 features)                    │
├─────────────────────────────────────────────────────────────────┤
│ refresh_inference_features() → inference_features_5m            │
│ - Calculates log returns (5m, 1h, 4h)                           │
│ - Computes macro z-scores (dxy, vix, embi)                      │
│ - Applies clipping + INNER JOIN                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. FEATURE AUGMENTATION (Python - 4 features)                   │
├─────────────────────────────────────────────────────────────────┤
│ FeatureBuilder service adds:                                     │
│ - rsi_9, atr_pct, adx_14 (technical indicators)                 │
│ - usdmxn_ret_1h (FX correlation, clip [-0.10, 0.10])            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. MODEL INFERENCE                                               │
├─────────────────────────────────────────────────────────────────┤
│ PPO Model (15 dimensions = 13 features + 2 state)               │
│ → action [-1, 1] → discretize → LONG/SHORT/HOLD                 │
│ → Log to dw.fact_rl_inference                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Observation Vector (15 dimensiones)

El modelo PPO v11/v14 espera un vector de 15 dimensiones:

```python
observation = [
    # 13 Features del mercado
    log_ret_5m,      # [0] Return 5min
    log_ret_1h,      # [1] Return 1h
    log_ret_4h,      # [2] Return 4h
    rsi_9,           # [3] RSI (Python)
    atr_pct,         # [4] ATR% (Python)
    adx_14,          # [5] ADX (Python)
    dxy_z,           # [6] DXY z-score
    dxy_change_1d,   # [7] DXY change
    vix_z,           # [8] VIX z-score
    embi_z,          # [9] EMBI z-score
    brent_change_1d, # [10] Brent change
    rate_spread,     # [11] Yield curve
    usdmxn_ret_1h,   # [12] USDMXN return (Python)

    # 2 Features de estado (agregados por el agente)
    position,        # [13] Posición actual [-1, 1]
    time_normalized  # [14] Tiempo normalizado en sesión
]
```

**NOTA**: hour_sin y hour_cos fueron **ELIMINADOS** en V14 (MAPEO línea 42).

---

## Troubleshooting

### Vista retorna 0 filas

```sql
-- Verificar datos en tablas fuente
SELECT COUNT(*) FROM usdcop_m5_ohlcv WHERE time >= NOW() - INTERVAL '7 days';
SELECT COUNT(*) FROM macro_indicators_daily WHERE date >= CURRENT_DATE - 7;

-- Verificar is_complete
SELECT date, is_complete, dxy, vix, embi
FROM macro_indicators_daily
WHERE date >= CURRENT_DATE - 7
ORDER BY date DESC;
```

### Features tienen NULL

```sql
-- La vista usa INNER JOIN, no debería haber NULLs en features SQL
-- Si hay NULLs, revisar is_complete en macro
SELECT date, is_complete,
       CASE WHEN dxy IS NULL THEN 'dxy' END AS missing
FROM macro_indicators_daily
WHERE date >= CURRENT_DATE - 7 AND NOT is_complete;
```

### Market hours check falla

```sql
SELECT
    NOW() AT TIME ZONE 'America/Bogota' AS current_time_cot,
    is_market_open() AS is_open,
    * FROM should_run_inference();
```

---

## Compatibilidad

| Componente | Versión Mínima |
|------------|---------------|
| PostgreSQL | 14+ |
| TimescaleDB | 2.8+ |
| Python | 3.9+ |
| Airflow | 2.5+ |

---

## Version History

| Versión | Fecha | Cambios |
|---------|-------|---------|
| **3.1** | 2025-12-16 | Migración V14 completa |
| | | - macro_indicators_daily con 37 columnas |
| | | - dw.fact_rl_inference con 13 features individuales |
| | | - INNER JOIN en vista (ERRATA corregida) |
| | | - hour_sin/hour_cos ELIMINADOS |
| | | - usdmxn_ret_1h clip [-0.10, 0.10] |
| | | - Columnas auxiliares para Python |
| | | - Documentación completa inline |

---

## Índices Creados

### macro_indicators_daily

| Índice | Columnas | Tipo |
|--------|----------|------|
| idx_macro_date | date DESC | B-tree |
| idx_macro_complete | is_complete, date DESC | B-tree |
| idx_macro_model_features | date, dxy, vix, embi, brent, usdmxn, treasury_10y, treasury_2y | Composite |

### dw.fact_rl_inference

| Índice | Columnas | Tipo |
|--------|----------|------|
| idx_rl_inference_model_time | model_id, timestamp_utc DESC | B-tree |
| idx_rl_inference_action | action_discretized | B-tree |
| idx_rl_inference_symbol_time | symbol, timestamp_utc DESC | B-tree |
| idx_rl_inference_dag_run | dag_run_id | B-tree |

### inference_features_5m

| Índice | Columnas | Tipo |
|--------|----------|------|
| idx_inf_features_ts | timestamp | UNIQUE (required for CONCURRENTLY) |
| idx_inf_features_date | trading_date DESC | B-tree |
| idx_inf_features_hour | hour_cot, trading_date | B-tree |
| idx_inf_features_bar | bar_number, trading_date | B-tree |

---

## Roles y Permisos

### trading_app (READ-ONLY)

```sql
GRANT SELECT ON macro_indicators_daily TO trading_app;
GRANT SELECT ON dw.fact_rl_inference TO trading_app;
GRANT SELECT ON inference_features_5m TO trading_app;
```

### airflow (READ-WRITE)

```sql
GRANT ALL ON macro_indicators_daily TO airflow;
GRANT ALL ON dw.fact_rl_inference TO airflow;
GRANT USAGE ON SCHEMA dw TO airflow;
GRANT EXECUTE ON FUNCTION refresh_inference_features() TO airflow;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO airflow;
```

---

## Referencias

- **ARQUITECTURA INTEGRAL V3**: `docs/ARQUITECTURA_INTEGRAL_V3.md`
- **MAPEO MIGRACIÓN**: `docs/MAPEO_MIGRACION_BIDIRECCIONAL.md`
- **Feature Config (SSOT)**: `config/feature_config.json`
- **Schema Original**: `init-scripts/12-unified-inference-schema.sql`
- **Realtime Inference**: `init-scripts/11-realtime-inference-tables.sql`

---

## Soporte y Contacto

Para preguntas sobre este schema:

1. Revisar ARQUITECTURA_INTEGRAL_V3.md (sección 4, 11.2, 21)
2. Revisar MAPEO_MIGRACION_BIDIRECCIONAL.md (ERRATA líneas 42-44)
3. Verificar feature_config.json para parámetros de normalización

**IMPORTANTE**: Este schema es la implementación final de Migration V14. NO modificar sin consultar los documentos de referencia.
