# ARQUITECTURA INTEGRAL USD/COP Trading System v3.0

**Autor**: Pedro @ Lean Tech Solutions
**Fecha**: 2025-12-16
**Objetivo**: Propuesta definitiva que reutiliza módulos existentes y elimina redundancias

---

## 1. DIAGNÓSTICO ACTUAL: REDUNDANCIAS IDENTIFICADAS

### Carpetas Analizadas
```
USDCOP-RL-Models/
├── notebooks/pipeline entrenamiento/    # Pipeline de entrenamiento RL (V11)
│   ├── config/settings.py               # 13 features para modelo
│   ├── src/environment.py               # TradingEnvV11
│   ├── models/                          # Modelos entrenados .zip
│   └── run.py                           # Ejecución de entrenamiento
│
├── data/pipeline/                       # Pipeline de preparación de datos (V4.0)
│   ├── 00_config → 07_output/           # 7 pasos ETL
│   ├── 06_rl_dataset_builder/           # Genera 10 datasets
│   └── run_pipeline.py                  # Orquestador
│
├── init-scripts/                        # 14 scripts SQL (REDUNDANTES)
│   ├── 01-essential-usdcop-init.sql     # Tabla usdcop_m5_ohlcv (USAR)
│   ├── 02-macro-data-schema.sql         # Macro tables
│   ├── 11-realtime-inference-tables.sql # Inference tables
│   └── ... (muchas tablas sin usar)
│
└── airflow/dags/                        # 17+ DAGs (REDUNDANTES)
    ├── usdcop_m5__00_l0_ohlcv_acquire   # TwelveData
    ├── usdcop_m5__00b_l0_macro_scraping # Scraping #1
    ├── usdcop_m5__01b_l0_macro_acquire  # Scraping #2 (DUPLICADO)
    └── ...
```

### Problemas Detectados
| Problema | Impacto | Solución |
|----------|---------|----------|
| 4 DAGs de macro scraping | Redundancia, confusión | Consolidar en 1 DAG |
| 14 scripts SQL | Tablas sin usar | Reducir a 3 tablas |
| 2 pipelines separados | No hay SSOT | JSON centralizado |
| Features hardcodeados | Desincronización train/inference | `feature_config.json` |
| No hay tabla de inferencia | Recalcular features cada vez | Vista materializada |

---

## 2. ARQUITECTURA PROPUESTA: 3 TABLAS + 1 JSON

### Diagrama de Alto Nivel

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    ARQUITECTURA SIMPLIFICADA v3.0                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                   ║
║  FUENTES EXTERNAS                                                                ║
║  ────────────────                                                                ║
║  ┌─────────────────┐              ┌─────────────────────────────────────┐        ║
║  │   TwelveData    │              │         Web Scraping               │        ║
║  │   USD/COP 5M    │              │  DXY, VIX, EMBI, Brent, Treasury   │        ║
║  └────────┬────────┘              └──────────────────┬──────────────────┘        ║
║           │                                          │                           ║
║           │ */5 min (8-13 UTC)                      │ 3x/día (7:55, 10:30, 12:00)║
║           ▼                                          ▼                           ║
║  ╔═══════════════════════════════════════════════════════════════════════════╗  ║
║  ║                       CAPA DE DATOS (PostgreSQL + TimescaleDB)             ║  ║
║  ║                                                                            ║  ║
║  ║  ┌──────────────────────────┐    ┌──────────────────────────┐             ║  ║
║  ║  │   usdcop_m5_ohlcv        │    │   macro_indicators_daily │             ║  ║
║  ║  │   (ya existe)            │    │   (crear)                │             ║  ║
║  ║  ├──────────────────────────┤    ├──────────────────────────┤             ║  ║
║  ║  │ time (PK)                │    │ date (PK)                │             ║  ║
║  ║  │ open, high, low, close   │    │ dxy, vix, embi, brent    │             ║  ║
║  ║  │ volume, source           │    │ treasury_2y, treasury_10y│             ║  ║
║  ║  │ ~60 rows/día × 5 años    │    │ usdmxn, fed_funds, etc.  │             ║  ║
║  ║  │ = 75,600 rows            │    │ ~1,200 rows              │             ║  ║
║  ║  └──────────────────────────┘    └──────────────────────────┘             ║  ║
║  ║                         │                       │                          ║  ║
║  ║                         └───────────┬───────────┘                          ║  ║
║  ║                                     │ JOIN + CÁLCULOS                      ║  ║
║  ║                                     ▼                                      ║  ║
║  ║  ┌────────────────────────────────────────────────────────────────────┐   ║  ║
║  ║  │                 inference_features_5m (VISTA MATERIALIZADA)         │   ║  ║
║  ║  │                 Definida por: config/feature_config.json            │   ║  ║
║  ║  ├────────────────────────────────────────────────────────────────────┤   ║  ║
║  ║  │ timestamp, close                                                    │   ║  ║
║  ║  │ log_ret_5m, log_ret_1h, log_ret_4h    ← Calculados desde OHLCV     │   ║  ║
║  ║  │ rsi_9, atr_pct, adx_14                ← Calculados desde OHLCV     │   ║  ║
║  ║  │ dxy_z, vix_z, embi_z                  ← Z-score de macro (ffill)   │   ║  ║
║  ║  │ brent_change_1d, rate_spread          ← Calculados desde macro     │   ║  ║
║  ║  │ usdmxn_ret_1h                         ← Calculados desde macro     │   ║  ║
║  ║  │ hour_sin, hour_cos                    ← Calculados temporales      │   ║  ║
║  ║  │ _raw_ret_5m                           ← Para reward (NO normalizado)│   ║  ║
║  ║  └────────────────────────────────────────────────────────────────────┘   ║  ║
║  ╚═══════════════════════════════════════════════════════════════════════════╝  ║
║                                     │                                            ║
║                                     │ REFRESH cada 5 min                         ║
║                                     ▼                                            ║
║  ╔═══════════════════════════════════════════════════════════════════════════╗  ║
║  ║                              INFERENCIA                                    ║  ║
║  ║  ┌────────────────────────────────────────────────────────────────────┐   ║  ║
║  ║  │  1. Leer última fila de inference_features_5m                       │   ║  ║
║  ║  │  2. Normalizar usando norm_stats de feature_config.json             │   ║  ║
║  ║  │  3. Ejecutar modelo PPO (ppo_usdcop_v14_fold0.zip)                  │   ║  ║
║  ║  │  4. Guardar en fact_rl_inference                                    │   ║  ║
║  ║  └────────────────────────────────────────────────────────────────────┘   ║  ║
║  ╚═══════════════════════════════════════════════════════════════════════════╝  ║
║                                                                                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. SSOT: feature_config.json

Este archivo es la **ÚNICA FUENTE DE VERDAD** para:
- Qué features usar en el modelo
- Cómo calcularlos
- Cómo normalizarlos
- Orden exacto del observation_space

```json
{
  "_meta": {
    "version": "3.0.0",
    "model_id": "ppo_usdcop_v14",
    "created_at": "2025-12-16",
    "description": "SSOT para features de entrenamiento e inferencia USD/COP"
  },

  "observation_space": {
    "dimension": 15,
    "order": [
      "log_ret_5m", "log_ret_1h", "log_ret_4h",
      "rsi_9", "atr_pct", "adx_14",
      "dxy_z", "dxy_change_1d",
      "vix_z", "embi_z",
      "brent_change_1d",
      "rate_spread", "usdmxn_ret_1h"
    ],
    "comment": "13 features + position + time_in_session en environment.py"
  },

  "sources": {
    "ohlcv": {
      "table": "usdcop_m5_ohlcv",
      "columns": ["time", "open", "high", "low", "close"],
      "granularity": "5min",
      "update_schedule": "*/5 13-17 * * 1-5"
    },
    "macro": {
      "table": "macro_indicators_daily",
      "resample_to": "5min",
      "resample_method": "ffill",
      "update_schedule": "55 12,30 15,0 17 * * 1-5"
    }
  },

  "features": {
    "returns": {
      "source": "ohlcv",
      "items": [
        {
          "name": "log_ret_5m",
          "formula": "ln(close / close[-1])",
          "lookback": 1,
          "norm_stats": {"mean": 2.0e-06, "std": 0.001138},
          "clip": [-0.05, 0.05]
        },
        {
          "name": "log_ret_1h",
          "formula": "ln(close / close[-12])",
          "lookback": 12,
          "norm_stats": {"mean": 2.3e-05, "std": 0.003776},
          "clip": [-0.05, 0.05]
        },
        {
          "name": "log_ret_4h",
          "formula": "ln(close / close[-48])",
          "lookback": 48,
          "norm_stats": {"mean": 5.2e-05, "std": 0.007768},
          "clip": [-0.05, 0.05]
        }
      ]
    },

    "technical": {
      "source": "ohlcv",
      "items": [
        {
          "name": "rsi_9",
          "indicator": "RSI",
          "period": 9,
          "norm_stats": {"mean": 49.27, "std": 23.07},
          "range": [0, 100]
        },
        {
          "name": "atr_pct",
          "indicator": "ATR_PCT",
          "period": 10,
          "formula": "(ATR / close) * 100",
          "norm_stats": {"mean": 0.062, "std": 0.0446}
        },
        {
          "name": "adx_14",
          "indicator": "ADX",
          "period": 14,
          "norm_stats": {"mean": 32.01, "std": 16.36},
          "range": [0, 100]
        }
      ]
    },

    "macro": {
      "source": "macro_indicators_daily",
      "resample": "ffill",
      "items": [
        {
          "name": "dxy_z",
          "raw_column": "dxy",
          "transform": "zscore_rolling",
          "window": 50,
          "norm_stats": {"mean": 103.0, "std": 5.0},
          "clip": [-4, 4]
        },
        {
          "name": "dxy_change_1d",
          "raw_column": "dxy",
          "transform": "pct_change",
          "periods": 1,
          "clip": [-0.03, 0.03]
        },
        {
          "name": "vix_z",
          "raw_column": "vix",
          "transform": "zscore_rolling",
          "window": 50,
          "norm_stats": {"mean": 20.0, "std": 10.0},
          "clip": [-4, 4]
        },
        {
          "name": "embi_z",
          "raw_column": "embi",
          "transform": "zscore_rolling",
          "window": 50,
          "norm_stats": {"mean": 300.0, "std": 100.0},
          "clip": [-4, 4]
        },
        {
          "name": "brent_change_1d",
          "raw_column": "brent",
          "transform": "pct_change",
          "periods": 1,
          "clip": [-0.10, 0.10]
        },
        {
          "name": "rate_spread",
          "formula": "treasury_10y - treasury_2y",
          "norm_stats": {"mean": -0.0326, "std": 1.400}
        },
        {
          "name": "usdmxn_ret_1h",
          "raw_column": "usdmxn",
          "transform": "pct_change",
          "periods": 1,
          "clip": [-0.05, 0.05]
        }
      ]
    }
  },

  "model": {
    "path": "models/ppo_usdcop_v14_fold0.zip",
    "framework": "stable-baselines3",
    "algorithm": "PPO",
    "observation_includes": ["features", "position", "time_normalized"]
  },

  "trading": {
    "market_hours": {
      "start_utc": "13:00",
      "end_utc": "17:55",
      "timezone": "America/Bogota",
      "local_start": "08:00",
      "local_end": "12:55"
    },
    "cost_per_trade": 0.0015,
    "weak_signal_threshold": 0.3
  }
}
```

---

## 4. TABLAS DE BASE DE DATOS

### Tabla 1: `usdcop_m5_ohlcv` (YA EXISTE)
```sql
-- Ya existe en init-scripts/01-essential-usdcop-init.sql
-- NO MODIFICAR, solo usar
```

### Tabla 2: `macro_indicators_daily` (CREAR)
```sql
CREATE TABLE IF NOT EXISTS macro_indicators_daily (
    date            DATE PRIMARY KEY,

    -- Índices principales
    dxy             NUMERIC(10, 4),      -- US Dollar Index
    vix             NUMERIC(10, 4),      -- Volatility Index
    embi            NUMERIC(10, 4),      -- EMBI Colombia

    -- Commodities
    brent           NUMERIC(10, 4),
    wti             NUMERIC(10, 4),
    gold            NUMERIC(10, 4),

    -- Tasas USA
    fed_funds       NUMERIC(8, 4),
    treasury_2y     NUMERIC(8, 4),
    treasury_10y    NUMERIC(8, 4),

    -- FX pairs
    usdmxn          NUMERIC(10, 4),
    usdclp          NUMERIC(10, 4),

    -- Metadata
    source          VARCHAR(100),
    is_complete     BOOLEAN DEFAULT FALSE,
    updated_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_macro_date ON macro_indicators_daily (date DESC);
```

### Vista 3: `inference_features_5m` (CREAR)
```sql
CREATE MATERIALIZED VIEW inference_features_5m AS
WITH
ohlcv_base AS (
    SELECT
        time AS timestamp,
        close,
        LN(close / LAG(close, 1) OVER w) AS log_ret_5m,
        LN(close / LAG(close, 12) OVER w) AS log_ret_1h,
        LN(close / LAG(close, 48) OVER w) AS log_ret_4h,
        close / LAG(close, 1) OVER w - 1 AS _raw_ret_5m
    FROM usdcop_m5_ohlcv
    WHERE time >= NOW() - INTERVAL '30 days'
    WINDOW w AS (ORDER BY time)
),
macro_ffill AS (
    SELECT
        date,
        dxy, vix, embi, brent,
        treasury_2y, treasury_10y, usdmxn,
        (dxy - LAG(dxy) OVER (ORDER BY date)) / NULLIF(LAG(dxy) OVER (ORDER BY date), 0) AS dxy_change_1d,
        (brent - LAG(brent) OVER (ORDER BY date)) / NULLIF(LAG(brent) OVER (ORDER BY date), 0) AS brent_change_1d,
        treasury_10y - treasury_2y AS rate_spread,
        (usdmxn - LAG(usdmxn) OVER (ORDER BY date)) / NULLIF(LAG(usdmxn) OVER (ORDER BY date), 0) AS usdmxn_ret_1h
    FROM macro_indicators_daily
    WHERE date >= CURRENT_DATE - INTERVAL '60 days'
)
SELECT
    o.timestamp,
    o.close,
    o.log_ret_5m,
    o.log_ret_1h,
    o.log_ret_4h,
    o._raw_ret_5m,
    -- Z-scores macro
    (m.dxy - 103.0) / 5.0 AS dxy_z,
    LEAST(GREATEST(m.dxy_change_1d, -0.03), 0.03) AS dxy_change_1d,
    (m.vix - 20.0) / 10.0 AS vix_z,
    (m.embi - 300.0) / 100.0 AS embi_z,
    LEAST(GREATEST(m.brent_change_1d, -0.10), 0.10) AS brent_change_1d,
    m.rate_spread,
    LEAST(GREATEST(m.usdmxn_ret_1h, -0.05), 0.05) AS usdmxn_ret_1h,
    -- Temporal
    SIN(2 * PI() * EXTRACT(HOUR FROM o.timestamp AT TIME ZONE 'America/Bogota') / 24) AS hour_sin,
    COS(2 * PI() * EXTRACT(HOUR FROM o.timestamp AT TIME ZONE 'America/Bogota') / 24) AS hour_cos
FROM ohlcv_base o
LEFT JOIN macro_ffill m ON DATE(o.timestamp AT TIME ZONE 'America/Bogota') = m.date
ORDER BY o.timestamp DESC;

CREATE UNIQUE INDEX idx_inf_features_ts ON inference_features_5m (timestamp);
```

---

## 5. REUTILIZACIÓN DE MÓDULOS EXISTENTES

### Mapeo de Carpetas → Uso

```
MÓDULO EXISTENTE                              │ ACCIÓN                │ DÓNDE SE USA
──────────────────────────────────────────────┼───────────────────────┼────────────────────────
notebooks/pipeline entrenamiento/             │ CONSERVAR             │ Entrenamiento manual
├── config/settings.py                        │ → Migrar a JSON       │ feature_config.json
├── src/environment.py                        │ CONSERVAR             │ TradingEnvV11
├── src/callbacks.py                          │ CONSERVAR             │ EntropyScheduler
├── src/backtest_report.py                    │ CONSERVAR             │ Métricas de evaluación
├── models/*.zip                              │ → Copiar a models/    │ Inferencia en producción
└── run.py                                    │ CONSERVAR             │ Ejecutar entrenamiento
                                              │                       │
data/pipeline/                                │ CONSERVAR             │ Regenerar datasets offline
├── 06_rl_dataset_builder/                    │ REUTILIZAR            │ Funciones de cálculo
│   └── 01_build_5min_datasets.py             │   calc_rsi()          │ → SQL o Python service
│                                             │   calc_atr()          │
│                                             │   z_score_rolling()   │
├── 07_output/datasets_5min/                  │ CONSERVAR             │ Datasets para entrenamiento
│   └── RL_DS3_MACRO_CORE.csv                 │                       │
└── run_pipeline.py                           │ CONSERVAR             │ Regeneración de datos
                                              │                       │
airflow/dags/                                 │ CONSOLIDAR            │ 4 DAGs principales
├── usdcop_m5__00_l0_ohlcv_acquire           │ CONSERVAR             │ TwelveData API
├── usdcop_m5__00b_l0_macro_scraping         │ RENOMBRAR             │ → usdcop_macro_unified
├── usdcop_m5__01b_l0_macro_acquire          │ ELIMINAR (duplicado)  │
├── usdcop_m5__05_l4_rlready.py              │ ELIMINAR              │ → Vista materializada SQL
└── usdcop_m5__06_l5_realtime_inference      │ SIMPLIFICAR           │ Solo leer vista + inferir
```

### Código Reutilizable de `data/pipeline/06_rl_dataset_builder/`

```python
# Estas funciones se pueden importar directamente en el servicio de inferencia
# O convertir a SQL para la vista materializada

from data.pipeline.06_rl_dataset_builder.01_build_5min_datasets import (
    calc_log_return,    # → SQL: LN(close / LAG(close, N))
    calc_rsi,           # → Python: mantener como está
    calc_atr,           # → Python: mantener como está
    calc_adx,           # → Python: mantener como está
    z_score_rolling,    # → SQL: (x - mean) / std
    pct_change_safe,    # → SQL: (x - LAG(x)) / LAG(x)
)
```

---

## 6. FLUJO DE EJECUCIÓN SIMPLIFICADO

### Timeline Diario

```
HORA (COT) │ HORA (UTC) │ ACCIÓN                                    │ COMPONENTE
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
07:55      │ 12:55      │ MACRO SCRAPING #1 (pre-apertura)          │ DAG: macro_unified
           │            │ - DXY, VIX, EMBI, Brent, etc.             │
           │            │ - Upsert en macro_indicators_daily        │
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
08:00      │ 13:00      │ 🔔 APERTURA DEL MERCADO                   │
           │            │ - OHLCV Acquire #1                        │ DAG: ohlcv_acquire
           │            │ - REFRESH inference_features_5m           │ DAG: refresh_features
           │            │ - Inferencia #1                           │ DAG: realtime_inference
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
08:05-     │ 13:05-     │ CICLO CADA 5 MINUTOS                      │
12:55      │ 17:55      │ - OHLCV Acquire                           │
           │            │ - REFRESH features                        │
           │            │ - Inferencia                              │
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
10:30      │ 15:30      │ MACRO SCRAPING #2 (mid-morning)           │ DAG: macro_unified
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
12:00      │ 17:00      │ MACRO SCRAPING #3 (cierre)                │ DAG: macro_unified
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
12:55      │ 17:55      │ 🔔 ÚLTIMA BARRA DEL DÍA                   │
───────────┼────────────┼───────────────────────────────────────────┼────────────────────
20:00      │ 01:00+1    │ BACKUP DIARIO                             │ DAG: backup_daily
```

### DAGs Finales (4 en lugar de 17)

```
1. usdcop_ohlcv_acquire          │ */5 13-17 * * 1-5  │ TwelveData API
2. usdcop_macro_unified          │ 55 12, 30 15, 0 17 │ Web scraping consolidado
3. usdcop_refresh_features       │ */5 13-17 * * 1-5  │ REFRESH MATERIALIZED VIEW
4. usdcop_realtime_inference     │ */5 13-17 * * 1-5  │ Leer vista → PPO → Guardar
```

---

## 7. PROCESO DE ENTRENAMIENTO (FUERA DEL PIPELINE)

El entrenamiento **NO es parte del pipeline automatizado**. Se ejecuta manualmente:

```bash
# 1. Regenerar datasets históricos (ocasional)
cd data/pipeline
python run_pipeline.py --from 3  # Pasos 3-6

# 2. Verificar dataset generado
ls 07_output/datasets_5min/RL_DS3_MACRO_CORE.csv

# 3. Ejecutar entrenamiento (manual)
cd ../../notebooks/pipeline\ entrenamiento
python run.py                    # 5 folds completos (~4 horas)
python run.py --quick            # Test rápido (50k steps)

# 4. Copiar modelo entrenado a producción
cp models/ppo_usdcop_v14_fold0.zip ../../models/
```

### Consistencia Train/Inference

```
ENTRENAMIENTO (settings.py)          │ INFERENCIA (feature_config.json)
─────────────────────────────────────┼────────────────────────────────────
FEATURES_FOR_MODEL = [               │ "observation_space": {
  'log_ret_5m', 'log_ret_1h',        │   "order": [
  'log_ret_4h', 'rsi_9', 'atr_pct',  │     "log_ret_5m", "log_ret_1h",
  'adx_14', 'dxy_z', 'dxy_change_1d',│     "log_ret_4h", "rsi_9", "atr_pct",
  'vix_z', 'embi_z',                 │     "adx_14", "dxy_z", "dxy_change_1d",
  'brent_change_1d', 'rate_spread',  │     "vix_z", "embi_z",
  'usdmxn_ret_1h'                    │     "brent_change_1d", "rate_spread",
]                                     │     "usdmxn_ret_1h"
                                     │   ]
COST_PER_TRADE = 0.0015              │ "cost_per_trade": 0.0015
WEAK_SIGNAL_THRESHOLD = 0.3          │ "weak_signal_threshold": 0.3
```

---

## 8. ESTRUCTURA DE CARPETAS FINAL

```
USDCOP-RL-Models/
│
├── config/                              # CONFIGURACIÓN CENTRALIZADA (NUEVO)
│   ├── feature_config.json              # SSOT de features
│   ├── trading_calendar.json            # Horarios y festivos
│   └── database.yaml                    # Conexiones BD
│
├── models/                              # MODELOS EN PRODUCCIÓN
│   ├── ppo_usdcop_v14_fold0.zip        # Modelo activo
│   ├── ppo_usdcop_v14_fold1.zip        # Backup
│   └── norm_stats_v11.json             # Estadísticas de normalización
│
├── src/                                 # CÓDIGO COMPARTIDO (NUEVO)
│   ├── __init__.py
│   ├── feature_builder.py               # Construcción de observaciones
│   ├── model_inference.py               # Wrapper del modelo PPO
│   └── trading_calendar.py              # Calendario de mercado
│
├── airflow/dags/                        # DAGs SIMPLIFICADOS
│   ├── usdcop_ohlcv_acquire.py
│   ├── usdcop_macro_unified.py
│   ├── usdcop_refresh_features.py
│   └── usdcop_realtime_inference.py
│
├── init-scripts/                        # SQL SIMPLIFICADO
│   ├── 01-essential-usdcop-init.sql    # (ya existe) usdcop_m5_ohlcv
│   ├── 02-macro-data-schema.sql        # macro_indicators_daily
│   └── 03-inference-features-view.sql  # Vista materializada
│
├── training/                            # ENTRENAMIENTO (RENOMBRADO)
│   ├── run.py                          # Script principal
│   ├── config/settings.py              # Configuración PPO
│   ├── src/                            # environment.py, callbacks.py, etc.
│   ├── models/                         # Modelos entrenados
│   └── outputs/                        # Resultados
│
├── data/pipeline/                       # PIPELINE DE DATOS HISTÓRICOS
│   ├── run_pipeline.py                 # Orquestador
│   ├── 06_rl_dataset_builder/          # Generación de datasets
│   └── 07_output/                      # Datasets generados
│
└── docs/
    └── ARQUITECTURA_INTEGRAL_V3.md     # Este documento
```

---

## 9. PASOS DE MIGRACIÓN

### Fase 1: Crear Tabla Macro (1 día)
```sql
-- Ejecutar en PostgreSQL
\i init-scripts/02-macro-data-schema.sql
```

### Fase 2: Migrar Datos Históricos (1 día)
```python
# Script de migración desde CSV existente
python scripts/migrate_macro_to_db.py
```

### Fase 3: Crear Vista Materializada (1 día)
```sql
\i init-scripts/03-inference-features-view.sql
```

### Fase 4: Crear feature_config.json (1 día)
```bash
cp config/feature_config.json.example config/feature_config.json
# Verificar que coincide con settings.py del entrenamiento
```

### Fase 5: Consolidar DAGs (2 días)
```bash
# Pausar DAGs antiguos
# Activar nuevos DAGs
# Monitorear 48h
```

### Fase 6: Mover Carpeta de Entrenamiento (1 día)
```bash
mv "notebooks/pipeline entrenamiento" training
# Actualizar paths en settings.py
```

---

## 10. RESUMEN EJECUTIVO

| Métrica | Antes | Después | Reducción |
|---------|-------|---------|-----------|
| DAGs Airflow | 17+ | 4 | -76% |
| Tablas/Vistas BD | ~15 | 3 | -80% |
| Scripts SQL | 14 | 3 | -79% |
| Archivos config | Dispersos | 1 JSON SSOT | Centralizado |
| Líneas en DAGs | ~5,000 | ~500 | -90% |

### Beneficios
1. **SSOT**: Un solo JSON define los features para train e inference
2. **Reutilización**: Funciones de cálculo compartidas
3. **Simplicidad**: 4 DAGs en lugar de 17
4. **Consistencia**: Train y inference usan exactamente los mismos features
5. **Mantenibilidad**: Cambiar un feature = cambiar 1 archivo

---

---

## 11. ESPECIFICACIÓN TÉCNICA CONFIRMADA (Fuente de Verdad)

**Fuente**: `notebooks/pipeline entrenamiento/` (pipeline funcional verificado)

### 11.0.1 Observation Space del Modelo V11

```
┌─────────────────────────────────────────────────────────────────┐
│              OBSERVATION SPACE: 15 DIMENSIONES                  │
├─────────────────────────────────────────────────────────────────┤
│  ÍNDICE │ FEATURE          │ FUENTE           │ NORMALIZACIÓN   │
├─────────┼──────────────────┼──────────────────┼─────────────────┤
│    0    │ log_ret_5m       │ OHLCV            │ z-score + clip  │
│    1    │ log_ret_1h       │ OHLCV            │ z-score + clip  │
│    2    │ log_ret_4h       │ OHLCV            │ z-score + clip  │
│    3    │ rsi_9            │ OHLCV            │ z-score         │
│    4    │ atr_pct          │ OHLCV            │ z-score         │
│    5    │ adx_14           │ OHLCV            │ z-score         │
│    6    │ dxy_z            │ macro_daily      │ fixed z-score   │
│    7    │ dxy_change_1d    │ macro_daily      │ clip only       │
│    8    │ vix_z            │ macro_daily      │ fixed z-score   │
│    9    │ embi_z           │ macro_daily      │ fixed z-score   │
│   10    │ brent_change_1d  │ macro_daily      │ clip only       │
│   11    │ rate_spread      │ macro_daily      │ z-score         │
│   12    │ usdmxn_ret_1h    │ macro_daily      │ clip only       │
├─────────┼──────────────────┼──────────────────┼─────────────────┤
│   13    │ position         │ estado agente    │ [-1, +1]        │
│   14    │ time_normalized  │ step/59          │ [0.0, 1.0]      │
└─────────────────────────────────────────────────────────────────┘
```

### 11.0.2 Features Excluidos en V14 (Definitivo)

| Feature Eliminado | Razón Técnica | Archivo Referencia |
|-------------------|---------------|-------------------|
| `bb_position` | Redundante con `rsi_9` | settings.py:28 |
| `dxy_mom_5d` | Redundante con `dxy_change_1d` | settings.py:29 |
| `vix_regime` | Redundante con `vix_z` | settings.py:30 |
| `brent_vol_5d` | Correlacionado con `atr_pct` | settings.py:31 |
| `hour_sin` | Bajo valor predictivo para FX | settings.py:32 |
| `hour_cos` | Bajo valor predictivo para FX | settings.py:32 |

### 11.0.3 Norm Stats para los 13 Features

```python
# Fuente: feature_config.json (derivado de training)
NORM_STATS = {
    'log_ret_5m':     {'mean': 2.0e-06,  'std': 0.001138, 'clip': [-0.05, 0.05]},
    'log_ret_1h':     {'mean': 2.3e-05,  'std': 0.003776, 'clip': [-0.05, 0.05]},
    'log_ret_4h':     {'mean': 5.2e-05,  'std': 0.007768, 'clip': [-0.05, 0.05]},
    'rsi_9':          {'mean': 49.27,    'std': 23.07},
    'atr_pct':        {'mean': 0.062,    'std': 0.0446},
    'adx_14':         {'mean': 32.01,    'std': 16.36},
    'dxy_z':          {'mean': 103.0,    'std': 5.0,   'clip': [-4, 4]},  # fixed
    'dxy_change_1d':  {'clip': [-0.03, 0.03]},  # solo clip
    'vix_z':          {'mean': 20.0,     'std': 10.0,  'clip': [-4, 4]},  # fixed
    'embi_z':         {'mean': 300.0,    'std': 100.0, 'clip': [-4, 4]},  # fixed
    'brent_change_1d':{'clip': [-0.10, 0.10]},  # solo clip
    'rate_spread':    {'mean': -0.0326,  'std': 1.400},
    'usdmxn_ret_1h':  {'clip': [-0.05, 0.05]},  # solo clip
}
```

### 11.0.4 Estado Actual: Training vs Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                    DIAGNÓSTICO DE CONSISTENCIA                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TRAINING PIPELINE (✅ CORRECTO - Fuente de Verdad)             │
│  ─────────────────────────────────────────────────              │
│  Ubicación: notebooks/pipeline entrenamiento/                   │
│  Features: 13 (FEATURES_FOR_MODEL en settings.py)               │
│  Obs dim: 15 (13 features + position + time_normalized)         │
│  Modelo: ppo_usdcop_v14_fold0.zip → shape=[15] ✅               │
│                                                                  │
│  ──────────────────────────────────────────────────────────────  │
│                                                                  │
│  INFERENCE DAG (❌ ROTO - Requiere Corrección)                  │
│  ─────────────────────────────────────────────                  │
│  Ubicación: airflow/dags/usdcop_m5__06_l5_realtime_inference.py │
│  Features: 19 (hardcoded, incluye eliminados en V14)            │
│  Obs dim: 20 (19 features + 1 ???)                              │
│  Estado: DESINCRONIZADO con modelo v11                          │
│                                                                  │
│  ──────────────────────────────────────────────────────────────  │
│                                                                  │
│  ACCIÓN REQUERIDA:                                               │
│  Actualizar DAG para usar feature_config.json                   │
│  y generar observaciones de 15 dimensiones                      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 11.0.5 Cálculo de time_normalized para Inferencia

```python
# En el environment de entrenamiento (environment.py:117):
time_normalized = self.step_count / self.episode_length

# Equivalente para inferencia en producción:
def calculate_time_normalized(bar_number: int) -> float:
    """
    Calcula time_normalized para inferencia.

    Args:
        bar_number: Número de barra del día (1-60)

    Returns:
        float: Valor entre 0.0 y 0.983 (nunca llega a 1.0)
    """
    # episode_length = 60, step_count va de 0 a 59
    # bar_number va de 1 a 60
    # CORREGIDO: Debe ser /60 para coincidir con environment.py:117
    return (bar_number - 1) / 60  # 0.0 a 0.983 (bar 60 = 59/60 = 0.983)
```

---

## 12. VALIDACIÓN DE LA PROPUESTA (6 Agentes de Análisis)

**Fecha de validación**: 2025-12-16
**Método**: 6 agentes especializados analizaron independientemente diferentes aspectos

### 11.1 RESUMEN DE VALIDACIÓN

| Componente | Estado | Agente | Hallazgos Críticos |
|------------|--------|--------|-------------------|
| feature_config.json | ⚠️ REQUIERE CORRECCIÓN | ac58151 | Desincronización con DAG de inferencia (13 vs 19 features) |
| Frontend Impact | ✅ VIABLE | a333dec | 43 endpoints, migración sin downtime posible |
| SQL Schema | ✅ APROBADO CON CORRECCIONES | ab0d179 | Conflicto con tabla existente, passwords hardcoded |
| Plan de Migración | ⚠️ SUBESTIMADO | a35ad50 | 7 días → 40 días reales (5.7x), fases faltantes |
| Reutilización de Código | ⚠️ DEUDA TÉCNICA | a4739ba | 1,200 líneas duplicadas, 0% test coverage |
| Servicios | ✅ CORRECTOS | aca1b98 | 4 servicios críticos identificados |

---

### 11.2 HALLAZGOS CRÍTICOS POR AGENTE

#### 🔴 Agente 1: Validación de feature_config.json (CRÍTICO)

**Inconsistencia detectada entre archivos:**

| Archivo | Features | Estado |
|---------|----------|--------|
| `config/feature_config.json` | 13 | ✅ Correcto |
| `notebooks/pipeline entrenamiento/config/settings.py` | 13 | ✅ Coincide |
| `airflow/dags/usdcop_m5__06_l5_realtime_inference.py` | **19** | ❌ **DESINCRONIZADO** |
| `init-scripts/12-unified-inference-schema.sql` | 15 (incluye hour_sin/cos) | ⚠️ Incluye extras |

**Features faltantes en config pero presentes en DAG:**
- `bb_position`
- `dxy_mom_5d`
- `vix_regime`
- `brent_vol_5d`
- `hour_sin`, `hour_cos`

**Acción requerida:**
```python
# En usdcop_m5__06_l5_realtime_inference.py
# CAMBIAR de hardcoded a:
import json
with open('/opt/airflow/config/feature_config.json') as f:
    CONFIG = json.load(f)
FEATURES = CONFIG['observation_space']['order']
```

---

#### 🟡 Agente 2: Impacto en Frontend

**Endpoints críticos identificados (43 total):**

| Categoría | Endpoints | Criticidad |
|-----------|-----------|------------|
| Market Data | `/api/market/realtime`, `/api/candlesticks/*` | 🔴 CRÍTICA |
| Trading Signals | `/api/trading/signals` | 🔴 CRÍTICA |
| Pipeline Status | `/api/pipeline/l0-l6/*` | 🟡 IMPORTANTE |
| Analytics | `/api/analytics/*` | 🟡 IMPORTANTE |

**WebSocket (4 conexiones - CONSOLIDAR):**
- `ws://localhost:8000/ws` (Trading API)
- `ws://localhost:8082/ws` (MarketDataService)
- `ws://localhost:3001` (useRealtimeData hook)
- `/api/proxy/ws` (fallback)

**Estrategia de migración sin downtime:**
```
FASE 1: Dual Backend (T-24h)
├─ Old API: http://localhost:8000-8003
└─ New API: https://new-backend-url/

FASE 2: Cutover (5 min durante mercado cerrado 12:55-13:00)
├─ Cambiar URLs en environment variables
└─ Forzar reconexión WebSocket

FASE 3: Rollback Ready (si latencia > 2000ms)
└─ Script automático de revert
```

---

#### 🟡 Agente 3: Validación Schema SQL

**Conflictos detectados:**

| Problema | Severidad | Solución |
|----------|-----------|----------|
| `fact_rl_inference_log` duplica `dw.fact_rl_inference` | 🔴 ALTA | Unificar en una sola tabla |
| LEFT JOIN macro permite NULL → features = 0 | 🟡 MEDIA | Cambiar a INNER JOIN |
| Passwords hardcoded (líneas 359, 362) | 🟡 MEDIA | Usar variables de entorno |
| `hour_sin/hour_cos` no en observation_space | 🔴 ALTA | Remover o documentar |

**Corrección recomendada para vista materializada:**
```sql
-- ANTES (LEFT JOIN permite NULLs)
LEFT JOIN macro_processed m
    ON DATE(owr.timestamp AT TIME ZONE 'America/Bogota') = m.date

-- DESPUÉS (solo datos completos)
INNER JOIN macro_processed m
    ON DATE(owr.timestamp AT TIME ZONE 'America/Bogota') = m.date
    AND m.is_complete = TRUE
```

---

#### 🔴 Agente 4: Plan de Migración (SUBESTIMADO)

**Timeline original vs realista:**

| Fase | Original | Realista | Razón |
|------|----------|----------|-------|
| Fase 0 (Pre-migración) | - | **3 días** | No documentada |
| Fase 1 (Tabla Macro) | 1 día | 2 días | Dependencias |
| Fase 1.5 (Indicadores SQL) | - | **5 días** | No implementados |
| Fase 2 (Migrar datos) | 1 día | 3 días | Validación |
| Fase 3 (Vista materializada) | 1 día | 4 días | Sin RSI/ATR/ADX |
| Fase 4 (feature_config) | 1 día | 1 día | ✅ Ya existe |
| Fase 5 (Consolidar DAGs) | 2 días | 5 días | Despliegue gradual |
| Fase 6 (Mover training) | 1 día | 2 días | Paths |
| Fase 7 (Testing) | - | **5 días** | No documentada |
| **TOTAL** | **7 días** | **30-40 días** | **~5x** |

**Fases faltantes críticas:**
1. **Fase 0**: Análisis de dependencias y baseline de métricas
2. **Fase 1.5**: Implementar RSI, ATR, ADX en PL/pgSQL o Python service
3. **Fase 7**: Testing integral, validación de feature parity

**Plan de rollback faltante:**
```bash
# rollback.sh (CREAR)
#!/bin/bash
PHASE=$1
case $PHASE in
  "phase1") pg_restore -d usdcop_trading backups/pre_phase1.backup ;;
  "phase5") airflow dags unpause usdcop_m5__00b_l0_macro_scraping ;;
  *) echo "Usage: ./rollback.sh [phase1|phase2|phase3|phase5]" ;;
esac
```

---

#### 🟡 Agente 5: Reutilización de Código

**Código duplicado identificado:**

| Función | Archivos donde aparece | Líneas duplicadas |
|---------|------------------------|-------------------|
| `calc_rsi()` | 4 archivos en data/pipeline/ | ~160 líneas |
| `calc_atr()` | 4 archivos | ~120 líneas |
| `calc_adx()` | 4 archivos | ~200 líneas |
| `z_score_rolling()` | 4 archivos | ~80 líneas |
| `calc_log_return()` | 4 archivos | ~40 líneas |
| **TOTAL** | - | **~1,200 líneas** |

**Archivos con duplicación:**
1. `data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py`
2. `data/pipeline/06_rl_dataset_builder/02_build_daily_datasets.py`
3. `data/pipeline/03_processing/scripts/03_create_rl_datasets.py`
4. `data/pipeline/03_processing/scripts/03b_create_rl_datasets_daily.py`

**Solución propuesta - Librería compartida:**
```
usdcop_common/                    # NUEVO
├── __init__.py
├── technical/
│   ├── __init__.py
│   ├── indicators.py            # calc_rsi, calc_atr, calc_adx
│   └── normalizers.py           # z_score_rolling, normalize_df
├── constants.py                 # VIX_THRESHOLDS, ZSCORE_WINDOW
└── validators.py                # Data quality checks
```

**Test coverage actual:** 0% para funciones técnicas ⚠️

---

#### ✅ Agente 6: Análisis de Servicios

**Servicios críticos (TIER 1 - No pueden removerse):**

| Puerto | Servicio | Función | Estado |
|--------|----------|---------|--------|
| 8000 | Trading API | REST + WebSocket datos reales | ✅ ACTIVO |
| 8001 | Trading Analytics API | KPIs, Sortino, VaR | ✅ ACTIVO |
| 8006 | Multi-Model Trading API | Agregación 5 estrategias | ✅ ACTIVO |
| 8087 | Real-time Ingestion V2 | **ÚNICO** punto de ingesta | ✅ ACTIVO |

**Servicios importantes (TIER 2):**

| Puerto | Servicio | Problema |
|--------|----------|----------|
| 8004 | Pipeline Data API | ⚠️ Falta `minio_manifest_reader.py` |
| 8007 | BI API | ⚠️ Comentado en docker-compose.yml |

**Matriz de dependencias:**
```
TwelveData API
     │
     ▼
Real-time Ingestion V2 (8087)
     │ INSERT usdcop_m5_ohlcv
     ▼
PostgreSQL + Redis
     │
     ├──────────────┬────────────────┐
     ▼              ▼                ▼
Trading API   Analytics API   Multi-Model API
  (8000)         (8001)           (8006)
     │              │                │
     └──────────────┴────────────────┘
                    │
                    ▼
              Dashboard (NextJS)
```

---

### 11.3 PLAN DE ACCIÓN CONSOLIDADO

#### 🔴 CRÍTICO (Antes de iniciar migración)

| # | Acción | Responsable | Tiempo |
|---|--------|-------------|--------|
| 1 | Sincronizar DAG inference con feature_config.json | Backend | 2 días |
| 2 | Implementar RSI/ATR/ADX en Python service | Quant | 5 días |
| 3 | Crear scripts de rollback para cada fase | DevOps | 2 días |
| 4 | Pausar DAG duplicado `usdcop_m5__01b_l0_macro_acquire` | DevOps | 30 min |

#### 🟡 IMPORTANTE (Durante migración)

| # | Acción | Responsable | Tiempo |
|---|--------|-------------|--------|
| 5 | Consolidar 4 conexiones WebSocket en 1 | Frontend | 3 días |
| 6 | Crear librería `usdcop_common/` | Backend | 4 días |
| 7 | Cambiar LEFT JOIN a INNER JOIN en vista SQL | DBA | 1 día |
| 8 | Remover passwords hardcoded | Security | 1 día |

#### 🟢 DESPUÉS (Post-migración)

| # | Acción | Responsable | Tiempo |
|---|--------|-------------|--------|
| 9 | Crear tests unitarios para funciones técnicas | QA | 3 días |
| 10 | Implementar monitoring dashboard (Grafana) | DevOps | 2 días |
| 11 | Documentar runbooks de operación | All | 2 días |

---

### 11.4 CRITERIOS DE ÉXITO

**La migración se considera exitosa si:**

| Métrica | Umbral | Medición |
|---------|--------|----------|
| Latencia de inferencia P95 | < 1000ms | Prometheus |
| Alertas CRITICAL | 0 durante 1 semana | PagerDuty |
| Feature parity | 100% (legacy vs nuevo) | Test suite |
| Uptime en horario de mercado | > 99.5% | Monitoring |
| Rollback time | < 10 minutos | Runbook |

**Criterios de rollback automático:**
- Latencia > 2000ms por 3 barras consecutivas
- > 5% features con NaN
- Equity drawdown > 10% intraday
- > 3 errores de DB en 1 hora

---

### 11.5 ARCHIVOS A MODIFICAR (RESUMEN)

```
MODIFICAR:
├── airflow/dags/usdcop_m5__06_l5_realtime_inference.py  # Leer de feature_config.json
├── init-scripts/12-unified-inference-schema.sql         # INNER JOIN, remover extras
├── docker-compose.yml                                   # Remover passwords, descomentar BI API
└── config/feature_config.json                           # Validar norm_stats

CREAR:
├── usdcop_common/                                       # Librería compartida
├── scripts/rollback.sh                                  # Scripts de rollback
├── tests/unit/test_technical_indicators.py             # Test coverage
└── docs/RUNBOOK_MIGRACION.md                           # Procedimientos operacionales

ELIMINAR (después de validación):
├── airflow/dags/usdcop_m5__01b_l0_macro_acquire.py     # Duplicado
└── Código duplicado en data/pipeline/                   # Usar usdcop_common
```

---

*Validación realizada el 2025-12-16 por 6 agentes de análisis especializados*

---

## 12. ESPECIFICACIÓN DEL DAG DE INFERENCIA CORREGIDO

**Problema actual**: El DAG `usdcop_m5__06_l5_realtime_inference.py` usa 19 features hardcodeados + obs_dim=20, pero el modelo v11 espera exactamente 15 dimensiones.

### 12.1 Arquitectura del DAG Corregido

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    DAG: usdcop_realtime_inference (CORREGIDO)                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  INICIO                                                                       │
│     │                                                                         │
│     ▼                                                                         │
│  ┌─────────────────────┐                                                     │
│  │ 1. CARGAR CONFIG    │  ◄── config/feature_config.json (SSOT)              │
│  │    desde JSON       │      - 13 features en orden exacto                  │
│  │                     │      - norm_stats para cada feature                 │
│  └──────────┬──────────┘      - trading config (thresholds)                  │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 2. CHECK MARKET     │  8:00-12:55 COT (Lun-Vie)                           │
│  │    HOURS            │  Validar contra holidays_2025_colombia              │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 3. GET CURRENT      │  Leer de dw.fact_agent_actions:                     │
│  │    STATE            │  - position_after (última posición)                 │
│  │                     │  - equity_after (último equity)                     │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 4. FETCH DATA       │  Desde inference_features_5m (vista)                │
│  │    (13 features)    │  - log_ret_5m, log_ret_1h, log_ret_4h              │
│  │                     │  - rsi_9, atr_pct, adx_14                           │
│  │                     │  - dxy_z, dxy_change_1d, vix_z, embi_z              │
│  │                     │  - brent_change_1d, rate_spread, usdmxn_ret_1h      │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 5. NORMALIZE        │  Aplicar z-score usando norm_stats del JSON:        │
│  │    FEATURES         │  normalized = (raw - mean) / std                    │
│  │                     │  clip(-4, 4) después de normalizar                  │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 6. BUILD            │  observation[15] = [                                │
│  │    OBSERVATION      │      features[0:13],     # 13 features normalizados │
│  │    (15 dims)        │      position,           # posición actual [-1,+1]  │
│  │                     │      time_normalized     # (bar_number-1)/59        │
│  │                     │  ]                                                  │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 7. RUN PPO          │  model.predict(observation, deterministic=True)     │
│  │    INFERENCE        │  action ∈ [-1, +1]                                  │
│  │                     │  Aplicar WEAK_SIGNAL_THRESHOLD = 0.3                │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│  ┌─────────────────────┐                                                     │
│  │ 8. STORE RESULTS    │  - dw.fact_rl_inference (detalle)                   │
│  │                     │  - dw.fact_agent_actions (acciones)                 │
│  │                     │  - dw.fact_equity_curve_realtime (equity)           │
│  └──────────┬──────────┘                                                     │
│             │                                                                 │
│             ▼                                                                 │
│         FIN                                                                   │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 12.2 Observation Vector: 15 Dimensiones Exactas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OBSERVATION ARRAY [15 elementos]                        │
├───────┬────────────────────┬─────────────────────────────────────────────────┤
│ INDEX │ FEATURE            │ VALOR / FUENTE                                  │
├───────┼────────────────────┼─────────────────────────────────────────────────┤
│   0   │ log_ret_5m         │ (raw - 2e-06) / 0.001138, clip(-4,4)           │
│   1   │ log_ret_1h         │ (raw - 2.3e-05) / 0.003776, clip(-4,4)         │
│   2   │ log_ret_4h         │ (raw - 5.2e-05) / 0.007768, clip(-4,4)         │
│   3   │ rsi_9              │ (raw - 49.27) / 23.07, clip(-4,4)              │
│   4   │ atr_pct            │ (raw - 0.062) / 0.0446, clip(-4,4)             │
│   5   │ adx_14             │ (raw - 32.01) / 16.36, clip(-4,4)              │
│   6   │ dxy_z              │ (dxy - 103.0) / 5.0, clip(-4,4)                │
│   7   │ dxy_change_1d      │ raw clipped to [-0.03, 0.03]                   │
│   8   │ vix_z              │ (vix - 20.0) / 10.0, clip(-4,4)                │
│   9   │ embi_z             │ (embi - 300.0) / 100.0, clip(-4,4)             │
│  10   │ brent_change_1d    │ raw clipped to [-0.10, 0.10]                   │
│  11   │ rate_spread        │ (raw - (-0.0326)) / 1.400, clip(-4,4)          │
│  12   │ usdmxn_ret_1h      │ raw clipped to [-0.05, 0.05]                   │
├───────┼────────────────────┼─────────────────────────────────────────────────┤
│  13   │ position           │ Posición actual del agente: [-1.0, +1.0]       │
│  14   │ time_normalized    │ (bar_number - 1) / 59 = [0.0, 1.0]             │
└───────┴────────────────────┴─────────────────────────────────────────────────┘
```

### 12.3 Cálculo de time_normalized

```
BAR_NUMBER    │ HORA COT    │ HORA UTC    │ time_normalized
──────────────┼─────────────┼─────────────┼─────────────────
      1       │   08:00     │   13:00     │ (1-1)/59 = 0.000
      2       │   08:05     │   13:05     │ (2-1)/59 = 0.017
      3       │   08:10     │   13:10     │ (3-1)/59 = 0.034
     ...      │    ...      │    ...      │      ...
     30       │   10:25     │   15:25     │ (30-1)/59 = 0.492
     ...      │    ...      │    ...      │      ...
     59       │   12:50     │   17:50     │ (59-1)/59 = 0.983
     60       │   12:55     │   17:55     │ (60-1)/59 = 1.000
```

**Fórmula desde hora COT:**
```
bar_number = ((hora - 8) * 60 + minuto) / 5 + 1
time_normalized = (bar_number - 1) / 59
```

### 12.4 Elementos a Eliminar del DAG Actual

| Elemento | Líneas en DAG actual | Acción |
|----------|---------------------|--------|
| `bb_position` | 54, 79, 307 | ELIMINAR |
| `dxy_mom_5d` | 54, 82, 312 | ELIMINAR |
| `vix_regime` | 55, 84, 314 | ELIMINAR |
| `brent_vol_5d` | 56, 87, 317 | ELIMINAR |
| `hour_sin` | 58, 90, 322 | ELIMINAR |
| `hour_cos` | 58, 91, 323 | ELIMINAR |
| `obs_dim: 20` | 60 | CAMBIAR a 15 |
| NORM_STATS hardcoded | 72-92 | LEER de JSON |

### 12.5 Configuración que Debe Leer del JSON

```
DESDE feature_config.json:
├── observation_space.order[]      → Lista de 13 features en orden
├── observation_space.dimension    → 15 (validación)
├── features.*.norm_stats          → mean, std para cada feature
├── features.*.clip                → Rangos de clipping
├── trading.weak_signal_threshold  → 0.3
├── trading.cost_per_trade         → 0.0015
├── trading.market_hours           → Horarios de mercado
└── holidays_2025_colombia[]       → Festivos (skip inference)
```

### 12.6 Validaciones Requeridas

El DAG debe validar antes de ejecutar inferencia:

| Validación | Criterio | Acción si falla |
|------------|----------|-----------------|
| Config cargado | `len(features) == 13` | ERROR, no ejecutar |
| Datos completos | Ningún feature NULL | WARN, usar default 0.0 |
| Obs dimension | `len(observation) == 15` | ERROR, no ejecutar |
| Market hours | Dentro de 8:00-12:55 COT | SKIP, log reason |
| Holiday check | No en holidays_2025_colombia | SKIP, log reason |
| Model loaded | Model file exists | ERROR, fallback momentum |

### 12.7 Diferencias Clave: DAG Actual vs Corregido

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    COMPARACIÓN DAG ACTUAL vs CORREGIDO                      │
├────────────────────────────────────┬───────────────────────────────────────┤
│         DAG ACTUAL (ROTO)          │         DAG CORREGIDO                 │
├────────────────────────────────────┼───────────────────────────────────────┤
│ Features hardcoded (19)            │ Features desde JSON (13)              │
│ obs_dim = 20                       │ obs_dim = 15                          │
│ NORM_STATS hardcoded               │ NORM_STATS desde JSON                 │
│ Incluye hour_sin/hour_cos          │ NO incluye (eliminados V14)           │
│ Incluye bb_position, etc.          │ NO incluye (redundantes)              │
│ time_normalized = bar/60           │ time_normalized = (bar-1)/59          │
│ No valida holidays                 │ Valida holidays_2025_colombia         │
│ No lee config externo              │ Lee feature_config.json               │
│                                    │                                       │
│ RESULTADO: Error de dimensión      │ RESULTADO: Compatible con modelo v11  │
│ al cargar modelo PPO               │                                       │
└────────────────────────────────────┴───────────────────────────────────────┘
```

---

## 13. ARQUITECTURA FRONTEND Y PLAN DE MIGRACIÓN

**Análisis realizado por**: Agente de Frontend (a3ef5bb)
**Framework**: Next.js 15.5 + React 19
**Ubicación**: `usdcop-trading-dashboard/`

### 13.1 Estructura Actual del Frontend

```
usdcop-trading-dashboard/
├── app/                           # App Router (Next.js 15)
│   ├── layout.tsx                 # Root layout
│   ├── page.tsx                   # Home redirect
│   ├── dashboard/                 # Main dashboard
│   ├── market/                    # Market data views
│   ├── trading/                   # Trading interface
│   ├── analytics/                 # Analytics views
│   ├── pipeline/                  # Pipeline monitoring
│   ├── agent-trading/             # Agent trading view (NEW)
│   └── api/                       # API routes (proxy)
│       └── agent/                 # Agent endpoints (NEW)
│
├── components/
│   ├── charts/                    # Chart components
│   │   ├── CandlestickChart.tsx
│   │   ├── ChartWithPositions.tsx # NEW
│   │   ├── EquityCurveChart.tsx
│   │   └── ...
│   ├── trading/
│   │   ├── AgentActionsTable.tsx  # NEW
│   │   ├── SignalDisplay.tsx
│   │   └── ...
│   └── ui/                        # Shadcn components
│
├── hooks/                         # 12 custom hooks
│   ├── useRealtimeData.ts         # WebSocket connection
│   ├── useMarketData.ts
│   ├── useTradingSignals.ts
│   ├── usePipelineStatus.ts
│   └── ...
│
├── lib/
│   └── api.ts                     # API client
│
└── public/                        # Static assets
```

### 13.2 Vistas Habilitadas (11 Total)

| # | Vista | Ruta | Estado | Componentes Clave |
|---|-------|------|--------|-------------------|
| 1 | Dashboard | `/dashboard` | ✅ Activo | Overview, KPIs |
| 2 | Market Data | `/market` | ✅ Activo | CandlestickChart, OrderBook |
| 3 | Trading | `/trading` | ✅ Activo | SignalDisplay, TradeHistory |
| 4 | Analytics | `/analytics` | ✅ Activo | EquityCurveChart, Metrics |
| 5 | Pipeline L0 | `/pipeline/l0` | ✅ Activo | OHLCV ingestion status |
| 6 | Pipeline L1 | `/pipeline/l1` | ✅ Activo | Processing status |
| 7 | Pipeline L2-L3 | `/pipeline/l2-l3` | ✅ Activo | Feature engineering |
| 8 | Pipeline L4-L5 | `/pipeline/l4-l5` | ✅ Activo | RL ready & inference |
| 9 | Pipeline L6 | `/pipeline/l6` | ✅ Activo | Trading execution |
| 10 | Agent Trading | `/agent-trading` | 🆕 NEW | ChartWithPositions, AgentActionsTable |
| 11 | Settings | `/settings` | ✅ Activo | Configuration |

### 13.3 Endpoints API Consumidos (40+ Total)

#### Categoría: Market Data (CRÍTICO)
```
GET  /api/market/realtime              → useRealtimeData.ts
GET  /api/candlesticks/{symbol}        → CandlestickChart.tsx
GET  /api/market/orderbook             → OrderBook.tsx
WS   ws://localhost:8082/ws            → MarketDataService
WS   ws://localhost:3001               → useRealtimeData (fallback)
```

#### Categoría: Trading Signals (CRÍTICO)
```
GET  /api/trading/signals              → SignalDisplay.tsx
GET  /api/trading/positions            → PositionTracker.tsx
POST /api/trading/execute              → TradeExecutor.tsx
GET  /api/agent/actions                → AgentActionsTable.tsx (NEW)
GET  /api/agent/equity                 → EquityCurveChart.tsx (NEW)
```

#### Categoría: Pipeline Status
```
GET  /api/pipeline/status              → PipelineOverview.tsx
GET  /api/pipeline/l0/status           → L0StatusCard.tsx
GET  /api/pipeline/l1/metrics          → L1MetricsCard.tsx
GET  /api/pipeline/l2/features         → L2FeatureCard.tsx
GET  /api/pipeline/l3/processed        → L3ProcessedCard.tsx
GET  /api/pipeline/l4/rl-ready         → L4ReadyCard.tsx
GET  /api/pipeline/l5/inference        → L5InferenceCard.tsx
GET  /api/pipeline/l6/trading          → L6TradingCard.tsx
```

#### Categoría: Analytics
```
GET  /api/analytics/kpis               → KPIDisplay.tsx
GET  /api/analytics/sharpe             → SharpeCard.tsx
GET  /api/analytics/sortino            → SortinoCard.tsx
GET  /api/analytics/drawdown           → DrawdownChart.tsx
GET  /api/analytics/var                → VaRCard.tsx
```

### 13.4 Conexiones WebSocket (4 → Consolidar a 1)

| Conexión Actual | Puerto | Hook/Componente | Acción |
|-----------------|--------|-----------------|--------|
| Trading API WS | 8000 | useRealtimeData | ✅ MANTENER (primario) |
| MarketDataService | 8082 | CandlestickChart | ⚠️ ELIMINAR (redundante) |
| Real-time fallback | 3001 | useRealtimeData | ⚠️ ELIMINAR (redundante) |
| Proxy WS | `/api/proxy/ws` | Varios | ⚠️ UNIFICAR |

**Solución propuesta:**
```typescript
// hooks/useUnifiedWebSocket.ts (NUEVO)
const WEBSOCKET_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

export function useUnifiedWebSocket() {
  // Single connection for all real-time data
  // Channels: market, signals, pipeline, equity
}
```

### 13.5 Plan de Migración Frontend (6 Semanas)

```
SEMANA 1-2: Preparación
├─ Crear useUnifiedWebSocket hook
├─ Añadir feature flags para v3.0
├─ Actualizar types para 15-dimension observation
└─ Crear tests E2E base

SEMANA 3: Migración Componentes
├─ Actualizar CandlestickChart para nueva API
├─ Migrar SignalDisplay a feature_config.json
├─ Crear ChartWithPositions (ya existe)
└─ Implementar AgentActionsTable (ya existe)

SEMANA 4: Migración Hooks
├─ Consolidar 4 WS connections → 1
├─ Actualizar useTradingSignals
├─ Actualizar usePipelineStatus
└─ Deprecar hooks obsoletos

SEMANA 5: Testing
├─ E2E tests para flujos críticos
├─ Load testing WebSocket
├─ Verificar latencia < 100ms
└─ Test de reconexión WS

SEMANA 6: Cutover
├─ Deploy a staging
├─ Pruebas con mercado cerrado
├─ Cutover durante 12:55-13:00 COT
└─ Monitoreo 48h post-deploy
```

### 13.6 Impacto en Componentes por Cambio

| Cambio en Backend | Componentes Afectados | Severidad | Acción |
|-------------------|----------------------|-----------|--------|
| 15-dim observation | SignalDisplay, FeatureViewer | 🔴 ALTA | Actualizar types |
| Nueva tabla fact_agent_actions | AgentActionsTable | 🟡 MEDIA | Ya compatible |
| Endpoint /api/agent/* | AgentView | 🟢 BAJA | Ya implementado |
| WS unificado | useRealtimeData, 5+ componentes | 🔴 ALTA | Refactor |
| Remover hour_sin/cos | FeatureViewer (si existe) | 🟢 BAJA | Remover display |

---

## 14. ARQUITECTURA DOCKER Y SERVICIOS

**Análisis realizado por**: Agente Docker (afeee95)
**Ubicación**: `docker-compose.yml`, `docker/`

### 14.1 Inventario de Servicios (14 Running + 2 One-off = 16 Total)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DOCKER: 4 CAPAS (Actualizado 2025-12-16)     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  CAPA 4: MONITOREO Y OBSERVABILIDAD                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │   Grafana    │  │  Prometheus  │  │   MLflow     │                       │
│  │    :3002     │  │    :9090     │  │    :5001     │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  CAPA 3: APIs Y SERVICIOS DE APLICACIÓN (4 activos)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ Trading API  │  │Analytics API │  │Multi-Model   │  │RT Ingestion  │     │
│  │    :8000     │  │    :8001     │  │    :8006     │  │    :8087     │     │
│  │  REST + WS   │  │ KPIs,Sharpe  │  │ Multi-strat  │  │ **CRÍTICO**  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  CAPA 2: ORQUESTACIÓN + FRONTEND                                             │
│  ┌──────────────────────────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │    AIRFLOW (Scheduler + Web)     │  │  Dashboard   │  │   pgAdmin    │   │
│  │         :8080 (UI)               │  │    :5000     │  │    :5050     │   │
│  │    LocalExecutor (no worker)     │  │   NextJS     │  │   DB Admin   │   │
│  └──────────────────────────────────┘  └──────────────┘  └──────────────┘   │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  CAPA 1: INFRAESTRUCTURA                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │  PostgreSQL  │  │    Redis     │  │    MinIO     │                       │
│  │  (Timescale) │  │   (Cache)    │  │  (Artifacts) │                       │
│  │    :5432     │  │    :6379     │  │ :9000/:9001  │                       │
│  └──────────────┘  └──────────────┘  └──────────────┘                       │
│                                                                              │
│  + minio-init (one-off) + airflow-init (one-off)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 14.2 Detalle de Servicios y Puertos (Actualizado 2025-12-16)

**INVENTARIO REAL desde docker-compose.yml:**

#### Servicios Activos (14 running + 2 one-off = 16 total)

| # | Servicio | Puerto | Imagen/Build | Healthcheck | Estado |
|---|----------|--------|--------------|-------------|--------|
| 1 | postgres | 5432 | timescale/timescaledb:latest-pg15 | pg_isready | ✅ ACTIVO |
| 2 | redis | 6379 | redis:7-alpine | redis-cli ping | ✅ ACTIVO |
| 3 | minio | 9000, 9001 | minio/minio:latest | curl /minio/health/live | ✅ ACTIVO |
| 4 | minio-init | - | minio/mc:latest | - | 🔄 ONE-OFF |
| 5 | airflow-init | - | Dockerfile.airflow-ml | - | 🔄 ONE-OFF |
| 6 | airflow-scheduler | - | Dockerfile.airflow-ml | airflow jobs check | ✅ ACTIVO |
| 7 | airflow-webserver | 8080 | Dockerfile.airflow-ml | curl /health | ✅ ACTIVO |
| 8 | pgadmin | 5050 | dpage/pgadmin4:latest | wget /misc/ping | ✅ ACTIVO |
| 9 | dashboard | 5000:3000 | Dockerfile.prod (NextJS) | wget /api/health | ✅ ACTIVO |
| 10 | prometheus | 9090 | prom/prometheus:latest | wget /-/healthy | ✅ ACTIVO |
| 11 | grafana | 3002:3000 | grafana/grafana:latest | wget /api/health | ✅ ACTIVO |
| 12 | trading-api | 8000 | Dockerfile.api | curl /api/health | ✅ ACTIVO |
| 13 | analytics-api | 8001 | Dockerfile.api | Python healthcheck | ✅ ACTIVO |
| 14 | multi-model-api | 8006 | Dockerfile.api | curl /api/health | ✅ ACTIVO |
| 15 | mlflow | 5001:5000 | ghcr.io/mlflow/mlflow:v2.10.2 | curl /health | ✅ ACTIVO |
| 16 | realtime-ingestion-v2 | 8087 | Dockerfile.api | curl /health | ✅ **CRÍTICO** |

#### Servicios Removidos/Comentados (NO activos)

| Servicio | Puerto Original | Razón de Remoción |
|----------|-----------------|-------------------|
| airflow-worker | - | LocalExecutor no requiere worker separado |
| usdcop-realtime-orchestrator | 8085 | Reemplazado por realtime-ingestion-v2 |
| usdcop-realtime-service | 8084 | Reemplazado por realtime-ingestion-v2 |
| realtime-data-service | - | Legacy, reemplazado |
| optimized-l0-validator | 8086 | Integrado en Airflow L0 pipeline |
| health-monitor | 8083 | Prometheus/Grafana lo reemplazan |
| websocket-service | 8082 | Integrado en trading-api |
| compliance-api | 8003 | No requerido actualmente |
| pipeline-data-api | 8002 | Falta módulo minio_manifest_reader |
| bi-api | 8007 | Errores de importación de módulos |
| l0-contracts-api | 8088 | Servido por pipeline-data-api |
| alpha-arena-api | 8007 | Sin DAGs que poblen datos |
| nginx | 80, 443 | Acceso directo a puertos suficiente |
| selenium | 4444 | No definido en docker-compose actual |

### 14.3 Volúmenes y Persistencia

```yaml
volumes:
  postgres_data:          # Datos PostgreSQL
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./data/postgres

  minio_data:            # Artifacts de pipeline
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ./data/minio

  airflow_logs:          # Logs de Airflow
    driver: local

  grafana_data:          # Dashboards Grafana
    driver: local

  redis_data:            # Cache Redis (opcional)
    driver: local
```

### 14.4 Networks

```yaml
networks:
  usdcop-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
```

### 14.5 Dependencias de Inicio (startup order)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORDEN DE INICIO (docker-compose up)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FASE 1 (0-30s): Infraestructura Base                                        │
│  ├── postgres       ──► wait-for-it.sh :5432                                │
│  └── redis          ──► wait-for-it.sh :6379                                │
│                                                                              │
│  FASE 2 (30-60s): Storage & Scraping                                         │
│  ├── minio          ──► depends_on: postgres                                │
│  └── selenium       ──► depends_on: (none)                                  │
│                                                                              │
│  FASE 3 (60-120s): Orquestación                                              │
│  ├── airflow-init   ──► one-off: airflow db init                            │
│  ├── airflow-scheduler ──► depends_on: postgres, redis, airflow-init        │
│  ├── airflow-worker    ──► depends_on: scheduler                            │
│  └── airflow-webserver ──► depends_on: scheduler                            │
│                                                                              │
│  FASE 4 (120-180s): APIs                                                     │
│  ├── trading-api       ──► depends_on: postgres, redis                      │
│  ├── analytics-api     ──► depends_on: postgres                             │
│  ├── pipeline-api      ──► depends_on: postgres, minio                      │
│  ├── multi-model-api   ──► depends_on: postgres                             │
│  └── rt-ingestion      ──► depends_on: postgres, redis  (CRÍTICO)           │
│                                                                              │
│  FASE 5 (180-240s): Monitoreo                                                │
│  ├── prometheus        ──► depends_on: all APIs                             │
│  └── grafana           ──► depends_on: prometheus                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 14.6 Plan de Migración Docker (7 Fases, ~11 Semanas)

```
FASE 1 (Semana 1-2): Preparación
├─ Backup completo de volúmenes
├─ Documentar configuraciones actuales
├─ Crear docker-compose.v3.yml
└─ Setup staging environment

FASE 2 (Semana 3): Infraestructura
├─ Actualizar postgres con nuevos init-scripts
├─ Crear nuevas tablas (macro_indicators_daily)
├─ Crear vista materializada (inference_features_5m)
└─ Validar migración de datos

FASE 3 (Semana 4-5): Servicios Core
├─ Consolidar trading-api + analytics-api → unified-api
├─ Actualizar rt-ingestion para v3.0
├─ Remover servicios obsoletos
└─ Actualizar healthchecks

FASE 4 (Semana 6): Airflow
├─ Deploy nuevos DAGs (4 en lugar de 17)
├─ Pausar DAGs obsoletos
├─ Validar schedule cron
└─ Test refresh de vista materializada

FASE 5 (Semana 7-8): APIs v3.0
├─ Deploy unified-api con feature_config.json
├─ Actualizar endpoints para 15-dim observation
├─ Validar compatibilidad con frontend
└─ Load testing

FASE 6 (Semana 9-10): Monitoreo
├─ Actualizar dashboards Grafana
├─ Crear alertas para nueva arquitectura
├─ Configurar métricas de inferencia
└─ Setup runbooks

FASE 7 (Semana 11): Cutover
├─ Deploy a producción durante mercado cerrado
├─ Smoke tests
├─ Monitoreo 48h
└─ Cleanup containers obsoletos
```

---

## 15. ESQUEMAS DE BASE DE DATOS E INIT-SCRIPTS

**Análisis realizado por**: Agente Database (aaccc03)
**Ubicación**: `init-scripts/`

### 15.1 Init-Scripts Existentes (15 Archivos)

| # | Script | Tablas/Objetos Creados | Estado v3.0 |
|---|--------|------------------------|-------------|
| 01 | 01-essential-usdcop-init.sql | users, usdcop_m5_ohlcv, trading_metrics, trading_sessions | ✅ MANTENER |
| 02a | 02-create-dwh-schema.sql | schemas stg/dw/dm, audit_log, funciones utility | ✅ MANTENER |
| 02b | 02-macro-data-schema.sql | macro_indicators_daily | ✅ MANTENER |
| 03 | 03-create-dimensions.sql | 10 dim_* tables (Kimball) | ✅ MANTENER |
| 04 | 04-seed-dimensions.sql | Seed data for dimensions | ✅ MANTENER |
| 05 | 05-create-facts.sql | 16 fact_* tables (L0-L6) | ✅ MANTENER |
| 05b | 05b-create-fact-indicator-5m.sql | fact_indicator_5m partitions | ✅ MANTENER |
| 06 | 06-create-data-marts.sql | dm.* materialized views | ✅ MANTENER |
| 07 | 07-create-multi-strategy-tables.sql | dim_strategy, fact_strategy_* (4 tables) | ✅ MANTENER |
| 08a | 08-add-xgb-ensemble-strategies.sql | Additional strategy entries | ⚠️ REVISAR |
| 08b | 08-seed-multi-strategy-data.sql | Seed data for strategies | ✅ MANTENER |
| 09 | 09-create-alpha-arena-tables.sql | 8 Alpha Arena tables (signals + equity) | ⚠️ OPCIONAL |
| 10 | 10-add-transparent-logging-columns.sql | Logging columns | ✅ MANTENER |
| 11 | 11-realtime-inference-tables.sql | 6 inference tables + 4 views + triggers | ✅ MANTENER |
| 12 | 12-unified-inference-schema.sql | inference_features_5m (MV) | ✅ ACTUALIZAR |

### 15.2 Inventario Completo de Tablas (~60 Total)

#### Schema: `public` (Datos Operacionales - 4 tablas)
```sql
-- 01-essential-usdcop-init.sql
users (id, username, email, password_hash, is_admin, ...)        -- Autenticación
usdcop_m5_ohlcv (time, symbol, open, high, low, close, volume)  -- Hypertable PRINCIPAL
trading_metrics (timestamp, metric_name, metric_value, ...)      -- Hypertable métricas
trading_sessions (id, user_id, session_start, strategy_name, ...)-- Sesiones de trading
```

#### Schema: `stg` (Staging - ETL)
```sql
-- Tablas temporales para procesos ETL (creadas dinámicamente)
```

#### Schema: `dw` (Data Warehouse - ~50 tablas)

**Dimensiones (10 tablas):**
```sql
-- 03-create-dimensions.sql
dim_symbol (symbol_id, symbol_code, base_currency, quote_currency)
dim_source (source_id, source_name, source_type, api_endpoint)
dim_time_5m (time_id, ts_utc, ts_cot, is_trading_hour, ...)      -- Pre-populated 2020-2030
dim_model (model_sk, model_id, algorithm, hyperparams, ...)       -- SCD Type 2
dim_feature (feature_id, feature_name, calculation_formula, ...)
dim_indicator (indicator_id, indicator_name, indicator_family, ...)
dim_reward_spec (reward_spec_sk, reward_function, params, ...)    -- SCD Type 2
dim_cost_model (cost_model_sk, spread_p95_bps, slippage_bps, ...)-- SCD Type 2
dim_episode (episode_sk, episode_id, split, date_cot, ...)
dim_backtest_run (run_sk, run_id, model_sk, split, ...)
dim_strategy (strategy_id, strategy_code, strategy_type, ...)     -- 07-create-multi-strategy
```

**Facts L0-L6 (16 tablas):**
```sql
-- 05-create-facts.sql
-- L0: Raw Ingestion
fact_bar_5m (symbol_id, time_id, ts_utc, open, high, low, close) -- Hypertable
fact_l0_acquisition (run_id, rows_fetched, coverage_pct, ...)

-- L1: Standardization
fact_l1_quality (date_cot, symbol_id, coverage_pct, status_passed)

-- L2: Technical Indicators
fact_indicator_5m (symbol_id, time_id, indicator_id, indicator_value) -- Partitioned
fact_winsorization (date_cot, winsor_rate_pct, outliers_clipped)
fact_hod_baseline (hhmm_cot, median_ret_log_5m, ...)

-- L3: Feature Engineering
fact_forward_ic (feature_id, date_cot, ic, pval, is_significant)
fact_leakage_tests (feature_id, date_cot, status_passed)
fact_feature_corr (feature_i_id, feature_j_id, correlation)

-- L4: RL-Ready
fact_rl_obs_stats (feature_id, split, clip_rate, abs_max, ...)
fact_cost_model_stats (cost_model_sk, spread_p50_bps, ...)
fact_episode (episode_sk, reward_sum, reward_mean, ...)

-- L5: Model Serving
fact_signal_5m (model_sk, symbol_id, ts_utc, action, confidence) -- Hypertable
fact_inference_latency (model_sk, date_cot, latency_p95_ms, ...)

-- L6: Backtesting
fact_trade (run_sk, trade_id, side, entry_px, exit_px, pnl)
fact_perf_daily (run_sk, date_cot, daily_return, equity, drawdown)
fact_perf_summary (run_sk, split, sharpe_ratio, max_drawdown, ...)
```

**Multi-Strategy Tables (5 tablas):**
```sql
-- 07-create-multi-strategy-tables.sql
fact_strategy_signals (signal_id, timestamp_utc, strategy_id, signal, confidence) -- Hypertable
fact_strategy_positions (position_id, strategy_id, side, entry_price, pnl)
fact_strategy_performance (perf_id, date_cot, strategy_id, daily_return_pct, ...)
fact_equity_curve (equity_id, timestamp_utc, strategy_id, equity_value, ...)      -- Hypertable
```

**Alpha Arena Tables (8 tablas):**
```sql
-- 09-create-alpha-arena-tables.sql
fact_signals_rl_ppo (signal_id, timestamp, signal, confidence, features)
fact_signals_ml_lgbm (signal_id, timestamp, signal, confidence, features)
fact_signals_llm_deepseek (signal_id, timestamp, signal, reasoning, model_used)
fact_signals_llm_claude (signal_id, timestamp, signal, reasoning)
fact_equity_rl_ppo (equity_id, timestamp, balance, equity, sharpe_ratio, ...)
fact_equity_ml_lgbm (equity_id, timestamp, balance, equity, ...)
fact_equity_llm_deepseek (equity_id, timestamp, balance, equity, ...)
fact_equity_llm_claude (equity_id, timestamp, balance, equity, ...)
```

**Realtime Inference Tables (6 tablas):**
```sql
-- 11-realtime-inference-tables.sql
fact_rl_inference (inference_id, timestamp_utc, model_id, observation, action_raw, ...)  -- Hypertable
fact_agent_actions (action_id, timestamp_utc, action_type, position_before/after, pnl)   -- Hypertable
fact_session_performance (session_id, session_date, total_trades, win_rate, sharpe, ...)
fact_equity_curve_realtime (equity_id, timestamp_utc, equity_value, drawdown, ...)       -- Hypertable
fact_macro_realtime (macro_id, timestamp_utc, dxy, vix, embi, brent, ...)               -- Hypertable
fact_inference_alerts (alert_id, timestamp_utc, alert_type, severity, message)
```

**Audit & Utility:**
```sql
-- 02-create-dwh-schema.sql
audit_log (audit_id, schema_name, table_name, operation, dag_id, ...)
```

#### Schema: `dm` (Data Marts)
```sql
-- Vistas materializadas para BI (definidas en 06-create-data-marts.sql)
```

### 15.3 Vistas Importantes

```sql
-- public schema
latest_ohlcv                    -- Última barra por símbolo
daily_ohlcv_summary             -- Resumen diario
metrics_summary                 -- Métricas agregadas
active_sessions                 -- Sesiones activas

-- dw schema
dw.health_check                 -- Estado del DWH
dw.vw_portfolio_summary         -- Resumen de portfolio multi-strategy
dw.vw_alpha_arena_leaderboard   -- Ranking de estrategias
dw.vw_strategy_comparison       -- Comparación de estrategias
dw.vw_latest_decisions          -- Últimas decisiones por estrategia
dw.v_latest_agent_actions       -- Acciones recientes del agente
dw.v_session_performance_summary-- Performance de sesiones
dw.v_equity_curve_today         -- Equity curve de hoy
dw.v_active_alerts              -- Alertas activas

-- Vista materializada para inferencia
inference_features_5m           -- 13 features para modelo RL v11
```

### 15.4 Tablas Críticas para v3.0 (Mínimo Viable)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TABLAS CRÍTICAS PARA OPERACIÓN v3.0                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TIER 1: ABSOLUTAMENTE REQUERIDAS (No puede funcionar sin estas)            │
│  ─────────────────────────────────────────────────────────────              │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ public.users                     → Autenticación                    │     │
│  │ public.usdcop_m5_ohlcv          → Datos de mercado (Hypertable)    │     │
│  │ public.macro_indicators_daily   → Datos macro (o crear nueva)      │     │
│  │ dw.fact_rl_inference            → Log de inferencias               │     │
│  │ dw.fact_agent_actions           → Acciones para frontend           │     │
│  │ dw.fact_equity_curve_realtime   → Equity curve                     │     │
│  │ inference_features_5m (MV)      → Features calculados              │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  TIER 2: REQUERIDAS PARA ANALYTICS                                          │
│  ─────────────────────────────────────────────────────────                  │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ dw.fact_session_performance     → Métricas diarias                 │     │
│  │ dw.fact_inference_alerts        → Alertas del sistema              │     │
│  │ dw.audit_log                    → Trazabilidad                     │     │
│  │ public.trading_sessions         → Sesiones de usuario              │     │
│  │ public.trading_metrics          → Métricas generales               │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
│  TIER 3: OPCIONALES (Multi-Strategy / Alpha Arena)                          │
│  ─────────────────────────────────────────────────────                      │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ dw.dim_strategy                 → Si se usa multi-strategy         │     │
│  │ dw.fact_strategy_*              → Si se usa multi-strategy         │     │
│  │ dw.fact_signals_*               → Si se usa Alpha Arena            │     │
│  │ dw.fact_equity_*                → Si se usa Alpha Arena            │     │
│  │ dw.dim_* (otras dimensiones)    → Si se usa DWH completo           │     │
│  │ dw.fact_* (L0-L6)               → Si se usa pipeline completo      │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 15.5 Correcciones Requeridas en SQL

| Archivo | Línea | Problema | Corrección |
|---------|-------|----------|------------|
| 11-realtime-inference-tables.sql | 25 | `observation FLOAT[]` dice 20 features | Cambiar comentario a 15 |
| 12-unified-inference-schema.sql | 45 | LEFT JOIN permite NULLs | Cambiar a INNER JOIN |
| 12-unified-inference-schema.sql | 78-79 | Incluye hour_sin/hour_cos | Eliminar (no en v11) |

---

## 16. INTEGRACIÓN DE SERVICIOS BACKEND

**Análisis realizado por**: Agente Backend (ae3dd62)
**Ubicación**: `services/`

### 16.1 Inventario de Servicios Python (6 Total)

| Puerto | Archivo | LOC | Framework | Función Principal |
|--------|---------|-----|-----------|-------------------|
| 8000 | trading_api_realtime.py | ~800 | FastAPI | REST + WebSocket datos mercado |
| 8001 | trading_analytics_api.py | ~600 | FastAPI | KPIs, Sharpe, Sortino, VaR |
| 8004 | pipeline_data_api.py | ~400 | FastAPI | Status pipeline, MinIO artifacts |
| 8006 | multi_model_trading_api.py | ~700 | FastAPI | Agregación 5 estrategias |
| 8007 | bi_api.py | ~500 | FastAPI | Business Intelligence |
| 8087 | realtime_market_ingestion_v2.py | ~600 | Standalone | Ingesta TwelveData |

**Total LOC**: ~3,600 (servicios únicos)
**LOC duplicado**: ~5,500 (funciones repetidas entre servicios)

### 16.2 Código Duplicado Identificado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DUPLICACIÓN DE CÓDIGO (~5,500 LOC)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  FUNCIÓN calc_returns() - 4 implementaciones                                 │
│  ├── services/trading_api_realtime.py:234-267                               │
│  ├── services/trading_analytics_api.py:89-122                               │
│  ├── services/multi_model_trading_api.py:156-189                            │
│  └── airflow/dags/utils/feature_utils.py:45-78                              │
│  LOC duplicado: ~130                                                         │
│                                                                              │
│  FUNCIÓN normalize_features() - 5 implementaciones                           │
│  ├── services/trading_api_realtime.py:312-356                               │
│  ├── services/multi_model_trading_api.py:234-278                            │
│  ├── airflow/dags/usdcop_m5__06_l5_realtime_inference.py:156-200           │
│  ├── notebooks/pipeline entrenamiento/src/data_loader.py:89-133            │
│  └── data/pipeline/06_rl_dataset_builder/01_build_5min_datasets.py:267-311 │
│  LOC duplicado: ~220                                                         │
│                                                                              │
│  CLASE DatabaseConnection - 6 implementaciones                               │
│  ├── services/*.py (cada archivo tiene su propia implementación)            │
│  LOC duplicado: ~300                                                         │
│                                                                              │
│  CONFIGURACIÓN FEATURES hardcoded - 4 lugares                                │
│  ├── settings.py → FEATURES_FOR_MODEL                                       │
│  ├── trading_api_realtime.py → FEATURES                                     │
│  ├── multi_model_trading_api.py → FEATURE_LIST                             │
│  ├── inference DAG → FEATURES                                               │
│  LOC duplicado: ~200 (+ riesgo de desincronización)                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 16.3 Oportunidades de Consolidación

```
ANTES (6 servicios):                      DESPUÉS (3 servicios):
├── trading_api_realtime.py ─────┐
│                                 ├──► unified_trading_api.py (8000)
├── trading_analytics_api.py ────┘     - Market data + WebSocket
│                                      - Analytics (KPIs, Sharpe, etc.)
├── bi_api.py ───────────────────┐     - Trading signals
│                                 ├──► unified_bi_api.py (8007)
├── multi_model_trading_api.py ──┘     - BI dashboards
│                                      - Multi-model aggregation
├── pipeline_data_api.py ────────────► pipeline_api.py (8004)
│                                      - Pipeline status
│                                      - MinIO artifacts
└── realtime_market_ingestion_v2.py ─► rt_ingestion.py (8087)
                                       - TwelveData ingestion
                                       - SIN CAMBIOS (crítico)
```

### 16.4 Librería Compartida Propuesta

```python
# usdcop_common/__init__.py
from .technical import calc_rsi, calc_atr, calc_adx
from .normalizers import normalize_features, z_score
from .database import get_db_connection, execute_query
from .config import load_feature_config, get_norm_stats
from .validation import validate_observation_dim, validate_features

# Estructura de directorios:
usdcop_common/
├── __init__.py
├── technical/
│   ├── __init__.py
│   ├── indicators.py      # calc_rsi, calc_atr, calc_adx, calc_returns
│   └── normalizers.py     # z_score, normalize_features
├── database/
│   ├── __init__.py
│   ├── connection.py      # DatabaseConnection class
│   └── queries.py         # Common SQL queries
├── config/
│   ├── __init__.py
│   └── loader.py          # load_feature_config, get_norm_stats
├── validation/
│   ├── __init__.py
│   └── validators.py      # validate_observation_dim, validate_features
└── constants.py           # FEATURE_ORDER, NORM_STATS, MARKET_HOURS
```

### 16.5 Plan de Migración Backend (7 Semanas)

```
SEMANA 1-2: Crear librería compartida
├─ Extraer código duplicado a usdcop_common
├─ Crear tests unitarios (pytest)
├─ Documentar API de la librería
└─ Publicar como paquete interno

SEMANA 3: Migrar servicios críticos
├─ Actualizar rt-ingestion para usar usdcop_common
├─ Validar ingesta sin interrupciones
└─ Test de carga

SEMANA 4-5: Consolidar APIs
├─ Merge trading_api + analytics_api
├─ Merge bi_api + multi_model_api
├─ Actualizar endpoints
└─ Actualizar docker-compose.yml

SEMANA 6: Actualizar DAGs
├─ Migrar DAGs para usar usdcop_common
├─ Remover código duplicado de dags/utils/
└─ Validar pipeline completo

SEMANA 7: Testing y deploy
├─ Integration tests
├─ Load testing
├─ Deploy a staging
└─ Cutover a producción
```

---

## 17. FLUJO DE DATOS Y DEPENDENCIAS

**Análisis realizado por**: Agente Data Flow (a0e7dd3)

### 17.1 Pipeline L0 → L6 Completo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FLUJO DE DATOS: L0 → L6                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  EXTERNAS                                                                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                      │
│  │ TwelveData  │    │ Investing   │    │ Selenium    │                      │
│  │   (OHLCV)   │    │   (Macro)   │    │  (Scraper)  │                      │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                      │
│         │                   │                   │                            │
│  ═══════╪═══════════════════╪═══════════════════╪════════════════════════   │
│         │                   │                   │                            │
│  L0: ACQUIRE                ▼                   ▼                            │
│         │           ┌─────────────────────────────────┐                      │
│         │           │        Web Scraping              │                      │
│         │           │  DAG: usdcop_macro_unified       │                      │
│         │           │  Schedule: 3x/día                │                      │
│         │           └──────────────┬──────────────────┘                      │
│         │                          │                                         │
│         ▼                          ▼                                         │
│  ┌─────────────────┐    ┌─────────────────┐                                 │
│  │ usdcop_m5_ohlcv │    │ macro_indicators│                                 │
│  │   (Hypertable)  │    │     _daily      │                                 │
│  └────────┬────────┘    └────────┬────────┘                                 │
│           │                      │                                           │
│  ═════════╪══════════════════════╪═══════════════════════════════════════   │
│           │                      │                                           │
│  L1: LANDING (No processing in v3.0 - direct to DB)                         │
│           │                      │                                           │
│  ═════════╪══════════════════════╪═══════════════════════════════════════   │
│           │                      │                                           │
│  L2-L3: FEATURE ENGINEERING                                                  │
│           │                      │                                           │
│           └──────────┬───────────┘                                           │
│                      │ JOIN + CÁLCULOS                                       │
│                      ▼                                                       │
│           ┌─────────────────────┐                                            │
│           │ inference_features  │  13 features:                              │
│           │       _5m (MV)      │  - log_ret_5m/1h/4h (OHLCV)               │
│           │                     │  - rsi_9, atr_pct, adx_14 (OHLCV)         │
│           │ REFRESH: */5 min    │  - dxy_z, vix_z, embi_z (macro, z-score)  │
│           │                     │  - dxy_change_1d, brent_change_1d (macro)  │
│           │                     │  - rate_spread, usdmxn_ret_1h (macro)      │
│           └─────────┬───────────┘                                            │
│                     │                                                        │
│  ═══════════════════╪════════════════════════════════════════════════════   │
│                     │                                                        │
│  L4: RL-READY       │                                                        │
│                     ▼                                                        │
│           ┌─────────────────────┐                                            │
│           │   BUILD OBSERVATION │  observation[15]:                          │
│           │     (15 dims)       │  - features[0:13] (normalized)             │
│           │                     │  - position (from last action)             │
│           │                     │  - time_normalized ((bar-1)/59)            │
│           └─────────┬───────────┘                                            │
│                     │                                                        │
│  ═══════════════════╪════════════════════════════════════════════════════   │
│                     │                                                        │
│  L5: INFERENCE      │                                                        │
│                     ▼                                                        │
│           ┌─────────────────────┐    ┌─────────────────────┐                │
│           │    PPO MODEL        │    │  ppo_usdcop_v14     │                │
│           │   .predict()        │◄───│    _fold0.zip       │                │
│           │                     │    │                     │                │
│           │ action ∈ [-1, +1]   │    │ (Stable-Baselines3) │                │
│           └─────────┬───────────┘    └─────────────────────┘                │
│                     │                                                        │
│                     │ weak_signal_threshold = 0.3                            │
│                     │ if |action| < 0.3: action = 0                          │
│                     ▼                                                        │
│           ┌─────────────────────┐                                            │
│           │  STORE RESULTS      │                                            │
│           │  - fact_rl_inference│                                            │
│           │  - fact_agent_actions│                                           │
│           │  - fact_equity_curve│                                            │
│           └─────────┬───────────┘                                            │
│                     │                                                        │
│  ═══════════════════╪════════════════════════════════════════════════════   │
│                     │                                                        │
│  L6: EXECUTION      │                                                        │
│                     ▼                                                        │
│           ┌─────────────────────┐                                            │
│           │  FRONTEND DISPLAY   │                                            │
│           │  - ChartWithPositions                                            │
│           │  - AgentActionsTable│                                            │
│           │  - EquityCurveChart │                                            │
│           └─────────────────────┘                                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 17.2 Puntos Únicos de Fallo (Single Points of Failure)

| Componente | Impacto si Falla | Mitigación |
|------------|------------------|------------|
| PostgreSQL | 🔴 TODO el sistema | Backup cada 6h, replica |
| rt-ingestion (8087) | 🔴 No hay datos OHLCV | Alerting, auto-restart |
| TwelveData API | 🟡 No hay nuevos datos | Cache último valor |
| Airflow Scheduler | 🟡 No se ejecutan DAGs | Worker puede continuar |
| Redis | 🟡 Sin cache, más latencia | Fallback a DB |
| MinIO | 🟢 Sin artifacts históricos | No afecta inferencia |

### 17.3 Orden de Inicialización del Sistema

```
PASO │ COMPONENTE            │ VALIDACIÓN                    │ TIMEOUT
─────┼───────────────────────┼───────────────────────────────┼─────────
  1  │ PostgreSQL            │ pg_isready -h localhost       │ 60s
  2  │ Redis                 │ redis-cli ping                │ 30s
  3  │ Ejecutar init-scripts │ psql -f 01-essential-*.sql   │ 120s
  4  │ Airflow DB Init       │ airflow db check              │ 60s
  5  │ Airflow Scheduler     │ airflow jobs check            │ 30s
  6  │ MinIO                 │ mc alias set local ...        │ 30s
  7  │ rt-ingestion          │ curl localhost:8087/health    │ 30s
  8  │ Trading API           │ curl localhost:8000/health    │ 30s
  9  │ Airflow Webserver     │ curl localhost:8080/health    │ 30s
 10  │ Grafana               │ curl localhost:3000/api/health│ 30s
```

### 17.4 Matriz de Impacto por Componente

```
                            │ OHLCV  │ MACRO  │ FEATURES │ INFERENCE │ DISPLAY
────────────────────────────┼────────┼────────┼──────────┼───────────┼─────────
PostgreSQL caído            │   ❌   │   ❌   │    ❌    │    ❌     │   ❌
Redis caído                 │   ⚠️   │   ✅   │    ✅    │    ⚠️     │   ⚠️
rt-ingestion caído          │   ❌   │   ✅   │    ⚠️    │    ⚠️     │   ⚠️
Airflow caído               │   ⚠️   │   ❌   │    ❌    │    ❌     │   ⚠️
Trading API caído           │   ✅   │   ✅   │    ✅    │    ✅     │   ❌
TwelveData timeout          │   ⚠️   │   ✅   │    ⚠️    │    ⚠️     │   ⚠️
Selenium error              │   ✅   │   ❌   │    ⚠️    │    ⚠️     │   ⚠️

✅ = Sin impacto   ⚠️ = Degradado   ❌ = No funciona
```

### 17.5 Estrategia de Migración por Capas (3 Fases)

```
FASE 1: Infraestructura (Semanas 1-3)
├─ L0: Actualizar init-scripts
│      - Crear macro_indicators_daily
│      - Crear inference_features_5m (MV)
├─ L1: Sin cambios (datos van directo a DB)
└─ Validación: Datos históricos migrados correctamente

FASE 2: Pipeline (Semanas 4-7)
├─ L2-L3: Actualizar vista materializada
│         - 13 features exactos
│         - INNER JOIN (no LEFT JOIN)
├─ L4: Actualizar build_observation
│      - 15 dimensiones
│      - time_normalized = (bar-1)/59
├─ L5: Actualizar DAG de inferencia
│      - Leer feature_config.json
│      - Remover features obsoletos
└─ Validación: Feature parity con training

FASE 3: Presentación (Semanas 8-10)
├─ L6: Actualizar frontend
│      - Componentes para 15-dim
│      - WebSocket unificado
├─ APIs: Consolidar servicios
└─ Validación: E2E tests pass
```

---

## 18. PLAN DE MIGRACIÓN CONSOLIDADO

### 18.1 Timeline Total: 12 Semanas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TIMELINE DE MIGRACIÓN CONSOLIDADO                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SEMANA 1-2: PREPARACIÓN                                                     │
│  ══════════════════════════════════════════════════════════════════════════  │
│  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  │                                                                           │
│  ├── Backup completo de base de datos                                       │
│  ├── Crear docker-compose.v3.yml                                            │
│  ├── Documentar estado actual                                               │
│  ├── Setup staging environment                                              │
│  └── Crear scripts de rollback                                              │
│                                                                              │
│  SEMANA 3-4: INFRAESTRUCTURA                                                 │
│  ══════════════════════════════════════════════════════════════════════════  │
│  ░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │
│  │                                                                           │
│  ├── Ejecutar nuevos init-scripts                                           │
│  ├── Crear tabla macro_indicators_daily                                     │
│  ├── Crear vista materializada inference_features_5m                        │
│  ├── Migrar datos históricos macro                                          │
│  └── Validar integridad de datos                                            │
│                                                                              │
│  SEMANA 5-6: LIBRERÍA COMPARTIDA                                             │
│  ══════════════════════════════════════════════════════════════════════════  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░  │
│  │                                                                           │
│  ├── Crear usdcop_common/ package                                           │
│  ├── Extraer código duplicado                                               │
│  ├── Escribir tests unitarios                                               │
│  └── Documentar API                                                         │
│                                                                              │
│  SEMANA 7-8: SERVICIOS BACKEND                                               │
│  ══════════════════════════════════════════════════════════════════════════  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓  │
│  │                                                                           │
│  ├── Consolidar trading_api + analytics_api                                 │
│  ├── Actualizar rt-ingestion                                                │
│  ├── Actualizar DAGs (17 → 4)                                               │
│  └── Load testing                                                           │
│                                                                              │
│  SEMANA 9-10: FRONTEND                                                       │
│  ══════════════════════════════════════════════════════════════════════════  │
│  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░▓▓│
│  │                                                                           │
│  ├── Consolidar WebSocket (4 → 1)                                           │
│  ├── Actualizar componentes para 15-dim                                     │
│  ├── E2E tests                                                              │
│  └── Deploy a staging                                                       │
│                                                                              │
│  SEMANA 11: CUTOVER                                                          │
│  ══════════════════════════════════════════════════════════════════════════  │
│  │                                                                           │
│  ├── Deploy durante mercado cerrado (12:55-13:00 COT)                       │
│  ├── Smoke tests                                                            │
│  └── Activar monitoreo                                                      │
│                                                                              │
│  SEMANA 12: ESTABILIZACIÓN                                                   │
│  ══════════════════════════════════════════════════════════════════════════  │
│  │                                                                           │
│  ├── Monitoreo 24/7                                                         │
│  ├── Ajustes post-deploy                                                    │
│  ├── Cleanup contenedores obsoletos                                         │
│  └── Documentación final                                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 18.2 Checklist de Migración

#### Pre-Migración (Semana 0)
- [ ] Backup completo PostgreSQL
- [ ] Export de todos los volúmenes Docker
- [ ] Documentar versiones actuales
- [ ] Crear branch `feature/v3-migration`
- [ ] Setup ambiente staging

#### Fase 1: Base de Datos (Semana 1-4)
- [ ] Crear `macro_indicators_daily`
- [ ] Migrar datos macro históricos
- [ ] Crear vista `inference_features_5m`
- [ ] Validar: 13 features exactos
- [ ] Validar: INNER JOIN (no NULLs)
- [ ] Eliminar tablas Alpha Arena
- [ ] Actualizar indexes

#### Fase 2: Backend (Semana 5-8)
- [ ] Crear package `usdcop_common/`
- [ ] Tests unitarios > 80% coverage
- [ ] Consolidar APIs (6 → 3)
- [ ] Actualizar DAGs (17 → 4)
- [ ] Load test: < 100ms latencia
- [ ] Health checks actualizados

#### Fase 3: Frontend (Semana 9-10)
- [ ] Unificar WebSocket
- [ ] Actualizar types (15-dim)
- [ ] E2E tests passing
- [ ] Deploy staging

#### Cutover (Semana 11)
- [ ] Deploy a producción
- [ ] Smoke tests passing
- [ ] Alertas configuradas
- [ ] Runbook validado

#### Post-Migración (Semana 12)
- [ ] Monitoreo 48h sin alertas
- [ ] Cleanup containers
- [ ] Documentación actualizada
- [ ] Retrospectiva

### 18.3 Criterios de Rollback

| Condición | Umbral | Acción |
|-----------|--------|--------|
| Latencia P95 | > 2000ms por 3 barras | Rollback automático |
| Errores 5xx | > 10/minuto | Rollback automático |
| Features NULL | > 5% | Rollback manual |
| Equity drawdown | > 10% intraday | Pausar trading |
| DB connections | > 90% pool | Alert + investigar |

### 18.4 Contactos de Escalación

| Nivel | Condición | Contacto |
|-------|-----------|----------|
| L1 | Alertas no críticas | On-call engineer |
| L2 | Servicio degradado | Tech lead |
| L3 | Sistema caído | Pedro (owner) |

---

## 19. AUDITORÍA CRÍTICA Y ARQUITECTURA SIMPLIFICADA

**Fecha de auditoría**: 2025-12-16
**Realizada por**: 6 agentes de validación en paralelo
**Resultado**: Score promedio **47/100** - NO LISTO PARA PRODUCCIÓN

### 19.1 Resultados de Validación por Componente

| Componente | Score | Estado | Hallazgo Principal |
|------------|-------|--------|-------------------|
| Airflow DAGs | 35/100 | 🔴 CRÍTICO | 14 DAGs existen vs 4 documentados |
| Database Schema | 71/100 | 🟡 CORRECCIONES | inference_features_5m tiene bugs |
| Docker Services | 42/100 | 🔴 CRÍTICO | Network name/subnet incorrectos |
| Frontend | 48/100 | 🔴 CRÍTICO | 7 vistas existen vs 11 documentadas |
| Backend Services | 62/100 | 🟡 CORRECCIONES | LOC subestimados 73.9% |
| Feature Config SSOT | 25/100 | 🔴 CRÍTICO | **SSOT aspiracional, no implementado** |

### 19.2 Hallazgo Crítico: Mismatch de Features

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ⚠️ MISMATCH CRÍTICO DE FEATURES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TRAINING (settings.py):     13 features → obs_dim = 15 ✅                  │
│  MODELO ENTRENADO:           Espera obs_dim = 15 ✅                          │
│  DAG L5 INFERENCE (ACTUAL):  18 features → obs_dim = 20 ❌                  │
│  feature_config.json:        13 features → obs_dim = 15 ✅ (pero no usado)  │
│                                                                              │
│  RESULTADO: Si se ejecuta el DAG actual → ERROR de shape mismatch           │
│             Model expects (1, 15), receives (1, 20)                         │
│                                                                              │
│  FEATURES EXTRAS EN DAG (A ELIMINAR):                                       │
│  - bb_position (redundante con rsi_9)                                       │
│  - dxy_mom_5d (redundante con dxy_change_1d)                                │
│  - vix_regime (redundante con vix_z)                                        │
│  - brent_vol_5d (correlacionada con atr_pct)                                │
│  - hour_sin, hour_cos (bajo valor predictivo)                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 19.3 Decisión Arquitectónica: Simplificación

**CONTEXTO**: Dado que los pipelines de preprocesamiento y entrenamiento actuales en Airflow tienen problemas de sincronización, se decide:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA SIMPLIFICADA v3.1                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LO QUE SE MANTIENE ✅                            │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  TRAINING PIPELINE (FUENTE DE VERDAD)                               │   │
│  │  └── notebooks/pipeline entrenamiento/                              │   │
│  │      ├── config/settings.py → 13 FEATURES_FOR_MODEL                 │   │
│  │      ├── src/environment.py → TradingEnvV11 (obs_dim=15)            │   │
│  │      ├── src/utils.py → normalize_df_v11()                          │   │
│  │      └── models/ppo_usdcop_v14_fold*.zip                            │   │
│  │                                                                      │   │
│  │  DATA PIPELINE (BASE PARA PREPROCESAMIENTO)                         │   │
│  │  └── data/pipeline/                                                 │   │
│  │      ├── 02_scrapers/ → Actualización diaria (REUTILIZAR)           │   │
│  │      ├── 04_cleaning/ → Limpieza de datos                           │   │
│  │      ├── 05_resampling/ → Resampleo a 5min                          │   │
│  │      └── 07_output/RL_DS3_MACRO_CORE.csv → Dataset principal        │   │
│  │                                                                      │   │
│  │  SERVICIOS BACKEND                                                  │   │
│  │  └── services/*.py → Funcionan correctamente                        │   │
│  │                                                                      │   │
│  │  INFRAESTRUCTURA DOCKER                                             │   │
│  │  └── docker-compose.yml → 11 servicios activos                      │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LO QUE SE IGNORA 🔴                              │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  DAGs DE PREPROCESAMIENTO (L0-L4)                                   │   │
│  │  └── Replantear usando data/pipeline/ como base                     │   │
│  │                                                                      │   │
│  │  DAG DE TRAINING (L5)                                               │   │
│  │  └── Usar notebooks/pipeline entrenamiento/ directamente            │   │
│  │                                                                      │   │
│  │  FEATURE SELECTION                                                  │   │
│  │  └── No necesario - 13 features ya seleccionadas cuidadosamente     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     LO QUE SE CREA DESDE 0 🔵                        │   │
│  ├─────────────────────────────────────────────────────────────────────┤   │
│  │                                                                      │   │
│  │  PIPELINE DE INFERENCIA EN TIEMPO REAL                              │   │
│  │  └── Nuevo servicio que:                                            │   │
│  │      ├── Use exactamente 13 features de settings.py                 │   │
│  │      ├── Normalice igual que training (normalize_df_v11)            │   │
│  │      ├── obs_dim = 15 (13 + position + time)                        │   │
│  │      └── Cargue ppo_usdcop_v14_fold1.zip (mejor WFE)                │   │
│  │                                                                      │   │
│  │  SERVICIO DE ACTUALIZACIÓN DIARIA                                   │   │
│  │  └── Usando data/pipeline/02_scrapers/actualizador_hpc_v3.py        │   │
│  │      ├── TwelveData: DXY, VIX, USDMXN, Brent                        │   │
│  │      ├── FRED: FEDFUNDS, Treasury 2Y/10Y                            │   │
│  │      ├── BCRP: EMBI Colombia                                        │   │
│  │      └── Inserta en PostgreSQL macro_indicators_daily               │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 19.4 Features Definitivas (13 - Seleccionadas)

```python
# Fuente de verdad: notebooks/pipeline entrenamiento/config/settings.py

FEATURES_FOR_MODEL = [
    # Retornos Multi-Timeframe (3)
    'log_ret_5m',       # Return 5-minutos (base)
    'log_ret_1h',       # Return 1-hora (tendencia intradiaria)
    'log_ret_4h',       # Return 4-horas (tendencia media)

    # Indicadores Técnicos (3)
    'rsi_9',            # RSI 9 períodos (momentum)
    'atr_pct',          # ATR % (volatilidad)
    'adx_14',           # ADX 14 períodos (fuerza tendencia)

    # Variables Macroeconómicas (7)
    'dxy_z',            # DXY z-score (fortaleza USD)
    'dxy_change_1d',    # Cambio DXY diario
    'vix_z',            # VIX z-score (volatilidad global)
    'embi_z',           # EMBI z-score (riesgo emergentes)
    'brent_change_1d',  # Cambio Brent diario (petróleo)
    'rate_spread',      # Spread tasas (USD vs COP)
    'usdmxn_ret_1h',    # Return USD/MXN (correlación regional)
]

# Observación total: 13 features + position + time_normalized = 15 dims
```

### 19.5 Modelos Disponibles (5 Folds)

| Fold | Archivo | Train Return | Test Return | WFE | Recomendación |
|------|---------|--------------|-------------|-----|---------------|
| 0 | ppo_usdcop_v14_fold0.zip | +50.71% | -3.68% | 0.0% | ❌ Overfitting |
| 1 | ppo_usdcop_v14_fold1.zip | +45.01% | +38.71% | 100% | ✅ **RECOMENDADO** |
| 2 | ppo_usdcop_v14_fold2.zip | +142.74% | +5.29% | 25.8% | ⚠️ Moderado |
| 3 | ppo_usdcop_v14_fold3.zip | +79.57% | +19.55% | 100% | ✅ Excelente |
| 4 | ppo_usdcop_v14_fold4.zip | +180.07% | +18.55% | 60.8% | ✅ Bueno |

**Walk-Forward Efficiency (WFE)** = Test Sharpe / Train Sharpe (capped 100%)

### 19.6 Scrapers Disponibles para Actualización Diaria

**Ubicación**: `data/pipeline/02_scrapers/`

| Scraper | Fuente | Variables | Frecuencia |
|---------|--------|-----------|-----------|
| `actualizador_hpc_v3.py` | Orquestador | Todos | Diaria |
| `scraper_investing.py` | Investing.com | WTI, Brent, Gold, DXY, USDMXN | Diaria |
| `scraper_embi_bcrp.py` | BCRP Perú | EMBI Colombia | Diaria |
| `scraper_dane_balanza.py` | DANE | Exportaciones/Importaciones | Mensual |
| `TwelveDataClient` | TwelveData API | DXY, VIX, Treasury | Diaria |

### 19.7 Datasets de Backup para BD

```
data/pipeline/07_output/
├── RL_DS3_MACRO_CORE.csv (30 MB)      ← DATASET PRINCIPAL PARA TRAINING
├── datasets_5min/ (249 MB total)       ← 10 variantes de datasets
└── datasets_daily/ (5 MB)

data/pipeline/01_fusion/output/
├── DATASET_MACRO_DAILY.csv (171 KB)    ← Histórico macro 2020-2025
├── DATASET_MACRO_MONTHLY.csv (10 KB)
└── DATASET_MACRO_QUARTERLY.csv (892 B)
```

### 19.8 Plan de Acción Simplificado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PLAN DE ACCIÓN (3 SEMANAS)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  SEMANA 1: SINCRONIZACIÓN Y SSOT                                            │
│  ═══════════════════════════════════════════════════════════════════════    │
│  │                                                                          │
│  ├── [P0] Actualizar feature_config.json con 13 features exactas           │
│  │        - Copiar FEATURES_FOR_MODEL de settings.py                        │
│  │        - obs_dim = 15                                                    │
│  │        - norm_stats de training                                          │
│  │                                                                          │
│  ├── [P0] Corregir documentación                                           │
│  │        - Network: usdcop-trading-network (no usdcop-network)            │
│  │        - Subnet: 172.29.0.0/16 (no 172.28.0.0/16)                       │
│  │        - DAGs: 14 reales (no 4)                                         │
│  │                                                                          │
│  └── [P1] Restaurar históricos en BD desde data/pipeline/07_output/        │
│                                                                              │
│  SEMANA 2: SERVICIO DE INFERENCIA                                           │
│  ═══════════════════════════════════════════════════════════════════════    │
│  │                                                                          │
│  ├── [P0] Crear nuevo servicio de inferencia                               │
│  │        - Leer feature_config.json                                        │
│  │        - Normalizar con normalize_df_v11()                              │
│  │        - Cargar ppo_usdcop_v14_fold1.zip                                │
│  │        - Validar obs_dim = 15 antes de predict                          │
│  │                                                                          │
│  ├── [P1] Integrar con trading_api_realtime.py                             │
│  │                                                                          │
│  └── [P1] Tests end-to-end                                                 │
│                                                                              │
│  SEMANA 3: ACTUALIZACIÓN DIARIA                                             │
│  ═══════════════════════════════════════════════════════════════════════    │
│  │                                                                          │
│  ├── [P1] Configurar actualizador_hpc_v3.py como cron job                  │
│  │        - Ejecutar 07:55, 10:30, 12:00 COT                               │
│  │        - Insertar en macro_indicators_daily                             │
│  │                                                                          │
│  ├── [P2] Monitoreo de actualización                                       │
│  │                                                                          │
│  └── [P2] Alertas de datos faltantes                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 19.9 Resumen de Decisiones

| Decisión | Justificación |
|----------|---------------|
| Ignorar DAGs L0-L5 | Desincronizados, replantear con data/pipeline/ |
| Usar notebooks/pipeline entrenamiento/ | Pipeline probado con WFE 56.5% promedio |
| No hacer feature selection | 13 features ya optimizadas en V14 |
| Crear inferencia desde 0 | DAG actual usa 18 features vs 13 requeridas |
| Usar fold 1 o 3 | Mejor WFE (100%), menor overfitting |
| Scraping con actualizador_hpc_v3.py | Ya tiene lógica de paralelización y caché |

---

## 20. APÉNDICE: CORRECCIONES INMEDIATAS

### 20.1 feature_config.json Corregido

```json
{
  "_meta": {
    "version": "3.1.0",
    "model_id": "ppo_usdcop_v14",
    "created_at": "2025-12-16",
    "description": "SSOT sincronizado con notebooks/pipeline entrenamiento/config/settings.py"
  },

  "observation_space": {
    "dimension": 15,
    "features_count": 13,
    "order": [
      "log_ret_5m", "log_ret_1h", "log_ret_4h",
      "rsi_9", "atr_pct", "adx_14",
      "dxy_z", "dxy_change_1d", "vix_z", "embi_z",
      "brent_change_1d", "rate_spread", "usdmxn_ret_1h"
    ],
    "additional_in_env": ["position", "time_normalized"],
    "total_obs_dim": 15
  },

  "model": {
    "recommended_fold": 1,
    "path": "models/ppo_usdcop_v14_fold1.zip",
    "wfe": 1.0
  }
}
```

### 20.2 Código de Inferencia Correcto

```python
# Pseudo-código para nuevo servicio de inferencia
import json
from stable_baselines3 import PPO
from src.utils import normalize_df_v11, calculate_norm_stats

# 1. Cargar configuración
with open('config/feature_config.json') as f:
    config = json.load(f)

FEATURES = config['observation_space']['order']  # 13 features
OBS_DIM = config['observation_space']['total_obs_dim']  # 15

# 2. Cargar modelo
model = PPO.load(config['model']['path'])

# 3. Para cada barra de 5min:
def predict(df_raw, position_current, time_step):
    # Normalizar igual que training
    df_norm = normalize_df_v11(df_raw, norm_stats, FEATURES)

    # Construir observación
    features = df_norm[FEATURES].iloc[-1].values  # 13 valores
    obs = np.concatenate([
        features,
        [position_current],           # +1
        [time_step / 60]              # +1 (normalizado)
    ])  # Total: 15

    # Validar dimensión
    assert len(obs) == OBS_DIM, f"Expected {OBS_DIM}, got {len(obs)}"

    # Predecir
    action, _ = model.predict(obs, deterministic=True)
    return action[0]  # [-1, +1]
```

---

*Sección 19-20 añadida el 2025-12-16 tras auditoría crítica de 6 agentes*
*Arquitectura simplificada para enfocarse en inferencia desde 0*

---

## 21. ARQUITECTURA DE DATOS: PROPUESTA INTEGRAL

### 21.1 Visión General

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DE DATOS USD/COP v3.0                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐    │
│  │  FUENTES DATOS   │     │  TABLAS RAW      │     │  TABLA FEATURES  │    │
│  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘    │
│           │                        │                        │               │
│  ┌────────▼─────────┐     ┌────────▼─────────┐     ┌────────▼─────────┐    │
│  │ TwelveData WS    │────▶│ usdcop_m5_ohlcv  │────▶│inference_features│    │
│  │ (Realtime 5min)  │     │ (Hypertable)     │     │    _5m           │    │
│  └──────────────────┘     └──────────────────┘     │ (13 features)    │    │
│                                                     └──────────────────┘    │
│  ┌──────────────────┐     ┌──────────────────┐              ▲              │
│  │ FRED, Investing  │────▶│ macro_indicators │              │              │
│  │ BCRP, DANE       │     │    _daily        │──────────────┘              │
│  │ (Daily/M/Q)      │     │ (37 variables)   │     JOIN + Transform        │
│  └──────────────────┘     └──────────────────┘                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 21.2 Estructura de Tablas

#### TABLA 1: `usdcop_m5_ohlcv` (Existente - Reusar L0)

```sql
-- Tabla de OHLCV en tiempo real - Ya existe, reusar
CREATE TABLE IF NOT EXISTS usdcop_m5_ohlcv (
    time           TIMESTAMPTZ NOT NULL,
    open           NUMERIC(12,4) NOT NULL,
    high           NUMERIC(12,4) NOT NULL,
    low            NUMERIC(12,4) NOT NULL,
    close          NUMERIC(12,4) NOT NULL,
    volume         NUMERIC(20,2) DEFAULT 0,
    source         VARCHAR(50) DEFAULT 'twelvedata',
    created_at     TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (time)
);

-- Hypertable para optimización TimescaleDB
SELECT create_hypertable('usdcop_m5_ohlcv', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Índices para queries de features
CREATE INDEX idx_ohlcv_close ON usdcop_m5_ohlcv(time DESC, close);
```

**Actualización:** Cada 5 minutos durante horario de mercado (13:00-17:55 UTC)
**Fuente:** TwelveData API / WebSocket
**Servicio:** `usdcop_m5__01_l0_intelligent_acquire.py` (reusar)

---

#### TABLA 2: `macro_indicators_daily` (Nueva - 37 Variables)

```sql
-- Tabla de indicadores macroeconómicos consolidados
CREATE TABLE IF NOT EXISTS macro_indicators_daily (
    date                  DATE PRIMARY KEY,

    -- === FIXED INCOME (7 variables) ===
    bond_yield5y_col      NUMERIC(8,4),   -- Bono Colombia 5Y
    bond_yield10y_col     NUMERIC(8,4),   -- Bono Colombia 10Y
    treasury_10y          NUMERIC(8,4),   -- UST 10Y (DGS10)
    treasury_2y           NUMERIC(8,4),   -- UST 2Y (DGS2)
    ibr_overnight         NUMERIC(8,4),   -- IBR Colombia
    prime_rate            NUMERIC(8,4),   -- Prime Rate USA
    rate_spread           NUMERIC(8,4),   -- CALCULATED: treasury_10y - treasury_2y

    -- === POLICY RATES (2 variables) ===
    fedfunds              NUMERIC(8,4),   -- Fed Funds Rate
    tpm_colombia          NUMERIC(8,4),   -- Tasa Política Monetaria BanRep

    -- === COMMODITIES (5 variables) ===
    wti                   NUMERIC(10,4),  -- WTI Crude Oil
    brent                 NUMERIC(10,4),  -- Brent Crude Oil
    gold                  NUMERIC(10,4),  -- Gold Futures
    coffee                NUMERIC(10,4),  -- Coffee Arabica

    -- === EXCHANGE RATES (5 variables) ===
    dxy                   NUMERIC(10,4),  -- Dollar Index
    usdmxn                NUMERIC(10,4),  -- USD/MXN
    usdclp                NUMERIC(10,4),  -- USD/CLP
    usdcop_spot           NUMERIC(10,4),  -- USD/COP Spot (cierre diario)
    itcr                  NUMERIC(10,4),  -- Índice Tasa Cambio Real

    -- === COUNTRY RISK (3 variables) ===
    embi                  NUMERIC(10,4),  -- EMBI Colombia
    cci_colombia          NUMERIC(10,4),  -- Índice Confianza Consumidor
    ici_colombia          NUMERIC(10,4),  -- Índice Confianza Industrial

    -- === VOLATILITY (1 variable) ===
    vix                   NUMERIC(10,4),  -- CBOE VIX

    -- === INFLATION (3 variables) ===
    ipc_colombia          NUMERIC(10,4),  -- IPC Colombia (mensual, ffill)
    cpi_usa               NUMERIC(10,4),  -- CPI USA (mensual, ffill)
    pce_usa               NUMERIC(10,4),  -- PCE USA (mensual, ffill)

    -- === FOREIGN TRADE (4 variables) ===
    exports_col           NUMERIC(14,2),  -- Exportaciones Colombia (mensual, ffill)
    imports_col           NUMERIC(14,2),  -- Importaciones Colombia (mensual, ffill)
    terms_of_trade        NUMERIC(10,4),  -- Términos de Intercambio (mensual, ffill)

    -- === BALANCE OF PAYMENTS (4 variables) ===
    ied_inflow            NUMERIC(14,2),  -- IED Entrante (trimestral, ffill)
    ied_outflow           NUMERIC(14,2),  -- IED Saliente (trimestral, ffill)
    current_account       NUMERIC(14,2),  -- Cuenta Corriente BP (trimestral, ffill)
    reserves_intl         NUMERIC(14,2),  -- Reservas Internacionales (mensual, ffill)

    -- === LABOR & PRODUCTION (3 variables) ===
    unemployment_usa      NUMERIC(8,4),   -- Desempleo USA (mensual, ffill)
    industrial_prod_usa   NUMERIC(10,4),  -- Producción Industrial USA (mensual, ffill)
    m2_supply_usa         NUMERIC(14,2),  -- M2 USA (mensual, ffill)

    -- === SENTIMENT & GDP (3 variables) ===
    consumer_sentiment    NUMERIC(10,4),  -- Michigan Sentiment (mensual, ffill)
    colcap                NUMERIC(10,4),  -- Índice COLCAP
    gdp_usa               NUMERIC(14,2),  -- GDP USA Real (trimestral, ffill)

    -- === METADATA ===
    updated_at            TIMESTAMPTZ DEFAULT NOW(),
    source_versions       JSONB DEFAULT '{}'
);

-- Índices
CREATE INDEX idx_macro_date ON macro_indicators_daily(date DESC);
CREATE INDEX idx_macro_model_features ON macro_indicators_daily(date, dxy, vix, embi, brent, usdmxn, treasury_10y, treasury_2y);
```

**Frecuencias de Actualización:**
| Frecuencia | Variables | Tratamiento |
|------------|-----------|-------------|
| Diaria (D) | 18 vars | Insert directo |
| Mensual (M) | 12 vars | Forward-fill hasta siguiente dato |
| Trimestral (Q) | 7 vars | Forward-fill hasta siguiente dato |

---

#### TABLA 3: `inference_features_5m` (Nueva - 13 Features)

```sql
-- Tabla de features transformadas listas para inferencia
CREATE TABLE IF NOT EXISTS inference_features_5m (
    time              TIMESTAMPTZ NOT NULL PRIMARY KEY,

    -- === RETURNS (3 features) ===
    log_ret_5m        NUMERIC(10,6) NOT NULL,  -- ln(close/close[-1])
    log_ret_1h        NUMERIC(10,6),           -- ln(close/close[-12])
    log_ret_4h        NUMERIC(10,6),           -- ln(close/close[-48])

    -- === TECHNICAL (3 features) ===
    rsi_9             NUMERIC(10,6),           -- RSI(9), normalizado
    atr_pct           NUMERIC(10,6),           -- ATR%: (ATR/close)*100, normalizado
    adx_14            NUMERIC(10,6),           -- ADX(14), normalizado

    -- === MACRO Z-SCORE (3 features) ===
    dxy_z             NUMERIC(10,6),           -- (dxy - 103.0) / 5.0
    vix_z             NUMERIC(10,6),           -- (vix - 20.0) / 10.0
    embi_z            NUMERIC(10,6),           -- (embi - 300.0) / 100.0

    -- === MACRO CHANGES (3 features) ===
    dxy_change_1d     NUMERIC(10,6),           -- dxy pct_change(1), clip[-0.03, 0.03]
    brent_change_1d   NUMERIC(10,6),           -- brent pct_change(1), clip[-0.10, 0.10]
    usdmxn_ret_1h     NUMERIC(10,6),           -- usdmxn pct_change(1), clip[-0.05, 0.05]

    -- === MACRO DERIVED (1 feature) ===
    rate_spread       NUMERIC(10,6),           -- treasury_10y - treasury_2y, normalizado

    -- === RAW RETURN (para reward calculation) ===
    _raw_ret_5m       NUMERIC(10,6),           -- (close/close[-1]) - 1 (sin normalizar)

    -- === METADATA ===
    close_price       NUMERIC(12,4),           -- Precio close original
    macro_date        DATE,                    -- Fecha de datos macro usados
    is_valid          BOOLEAN DEFAULT TRUE,    -- Flag de calidad
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Hypertable
SELECT create_hypertable('inference_features_5m', 'time',
    chunk_time_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- Índice para queries de inferencia (últimas N barras)
CREATE INDEX idx_features_time_valid ON inference_features_5m(time DESC)
    WHERE is_valid = TRUE;
```

---

### 21.3 Pipeline de Adquisición de Datos

#### 21.3.1 Fuentes y Frecuencias

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MATRIZ DE FUENTES DE DATOS                          │
├──────────────────┬─────────────┬───────────────┬────────────────────────────┤
│ FUENTE           │ FRECUENCIA  │ VARIABLES     │ MÉTODO DE ADQUISICIÓN      │
├──────────────────┼─────────────┼───────────────┼────────────────────────────┤
│ TwelveData       │ 5min RT     │ OHLCV         │ WebSocket/REST API         │
│ Investing.com    │ Diaria      │ 12 vars       │ Selenium scraping          │
│ FRED             │ D/M/Q       │ 10 vars       │ fredapi Python lib         │
│ BanRep/SUAMECA   │ D/M         │ 8 vars        │ Selenium scraping          │
│ BCRP             │ Diaria      │ 1 var (EMBI)  │ HTTP request               │
│ DANE             │ Mensual     │ 3 vars        │ Excel download             │
│ Fedesarrollo     │ Mensual     │ 2 vars        │ Excel download             │
└──────────────────┴─────────────┴───────────────┴────────────────────────────┘
```

#### 21.3.2 Servicio de Scraping Reutilizable

**Ubicación:** `data/pipeline/02_scrapers/actualizador_hpc_v3.py`

```python
# Arquitectura del actualizador HPC existente
class MacroUpdater:
    """
    Reusar para actualización diaria de macro_indicators_daily

    Capacidades existentes:
    - Paralelización con ThreadPoolExecutor
    - Caché local para evitar re-scraping
    - Manejo de errores y reintentos
    - Logging estructurado
    """

    def update_daily_indicators(self):
        """
        Flujo de actualización:
        1. Identificar fuentes que necesitan actualización
        2. Scrape en paralelo (max 5 threads)
        3. Validar datos descargados
        4. Insert/Update en macro_indicators_daily
        5. Trigger recalculation de inference_features_5m
        """
        pass
```

**Adaptaciones necesarias:**
1. Agregar función `insert_to_postgres()` para escribir a `macro_indicators_daily`
2. Agregar flag `--update-db` para modo producción vs modo archivo
3. Crear endpoint REST para trigger manual

---

### 21.4 Pipeline de Transformación

#### 21.4.1 Cálculo de Features desde Raw

```python
# Pseudo-código para transformación OHLCV + Macro → Features

def calculate_inference_features(ohlcv_df, macro_df):
    """
    Input:
        - ohlcv_df: últimas 100 barras de usdcop_m5_ohlcv
        - macro_df: último registro de macro_indicators_daily

    Output:
        - row para inference_features_5m (13 features + metadata)
    """

    # === 1. RETURNS ===
    log_ret_5m = np.log(close / close.shift(1))
    log_ret_1h = np.log(close / close.shift(12))
    log_ret_4h = np.log(close / close.shift(48))

    # === 2. TECHNICAL ===
    rsi_9 = calc_rsi(close, period=9)
    atr = calc_atr(high, low, close, period=10)
    atr_pct = (atr / close) * 100
    adx_14 = calc_adx(high, low, close, period=14)

    # === 3. MACRO Z-SCORE (con stats fijos del training) ===
    dxy_z = (macro_df['dxy'] - 103.0) / 5.0
    vix_z = (macro_df['vix'] - 20.0) / 10.0
    embi_z = (macro_df['embi'] - 300.0) / 100.0

    # === 4. MACRO CHANGES ===
    dxy_change_1d = np.clip(macro_df['dxy_pct_change'], -0.03, 0.03)
    brent_change_1d = np.clip(macro_df['brent_pct_change'], -0.10, 0.10)
    usdmxn_ret_1h = np.clip(macro_df['usdmxn_pct_change'], -0.05, 0.05)

    # === 5. DERIVED ===
    rate_spread = macro_df['treasury_10y'] - macro_df['treasury_2y']

    # === 6. NORMALIZATION (z-score con stats de training) ===
    features = normalize_with_training_stats({
        'log_ret_5m': log_ret_5m,
        'log_ret_1h': log_ret_1h,
        'log_ret_4h': log_ret_4h,
        'rsi_9': rsi_9,
        'atr_pct': atr_pct,
        'adx_14': adx_14,
        # ... macro features ya normalizadas
    })

    # === 7. CLIP FINAL ===
    features = np.clip(features, -4.0, 4.0)

    return features
```

#### 21.4.2 Estadísticas de Normalización (SSOT)

```json
// Extraído de feature_config.json - NO MODIFICAR
{
  "normalization_stats": {
    "log_ret_5m": {"mean": 2.0e-06, "std": 0.001138},
    "log_ret_1h": {"mean": 2.3e-05, "std": 0.003776},
    "log_ret_4h": {"mean": 5.2e-05, "std": 0.007768},
    "rsi_9": {"mean": 49.27, "std": 23.07},
    "atr_pct": {"mean": 0.062, "std": 0.0446},
    "adx_14": {"mean": 32.01, "std": 16.36},
    "dxy_z": {"mean": 103.0, "std": 5.0},
    "vix_z": {"mean": 20.0, "std": 10.0},
    "embi_z": {"mean": 300.0, "std": 100.0},
    "rate_spread": {"mean": -0.0326, "std": 1.400}
  }
}
```

---

### 21.5 Schedules de Actualización

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          SCHEDULE DE ACTUALIZACIÓN                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  HORARIO DE MERCADO COLOMBIA: 08:00 - 12:55 COT (13:00 - 17:55 UTC)        │
│  ═══════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  07:55 COT ┌────────────────────────────────────────┐                      │
│  (Pre-mkt) │ DAG: macro_daily_update                │                      │
│            │ - Actualizar macro_indicators_daily    │                      │
│            │ - Forward-fill datos M/Q               │                      │
│            │ - Calcular pct_changes                 │                      │
│            └────────────────────────────────────────┘                      │
│                                                                             │
│  08:00 COT ┌────────────────────────────────────────┐                      │
│  (Market   │ Servicio: realtime_market_ingestion    │                      │
│   Open)    │ - WebSocket TwelveData conectado       │                      │
│            │ - Insert usdcop_m5_ohlcv cada 5min     │                      │
│            └────────────────────────────────────────┘                      │
│                         │                                                   │
│                         ▼                                                   │
│  */5 min   ┌────────────────────────────────────────┐                      │
│  (Market   │ Trigger: on_new_ohlcv_bar              │                      │
│   Hours)   │ - JOIN OHLCV + macro_daily             │                      │
│            │ - Calculate 13 features                │                      │
│            │ - Insert inference_features_5m         │                      │
│            │ - Invoke RL model prediction           │                      │
│            └────────────────────────────────────────┘                      │
│                                                                             │
│  13:00 COT ┌────────────────────────────────────────┐                      │
│  (Mkt      │ Servicio: close market session         │                      │
│   Close)   │ - WebSocket disconnect                 │                      │
│            │ - Log resumen del día                  │                      │
│            └────────────────────────────────────────┘                      │
│                                                                             │
│  15:30 COT ┌────────────────────────────────────────┐                      │
│  (Post     │ DAG: macro_daily_update (2nd run)      │                      │
│   NYSE)    │ - Actualizar con datos cierre USA      │                      │
│            │ - VIX, DXY, Treasury final del día     │                      │
│            └────────────────────────────────────────┘                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 21.6 Airflow DAGs Propuestos

#### DAG 1: `l0_ohlcv_realtime` (Reusar existente)

```python
# Reutilizar: usdcop_m5__01_l0_intelligent_acquire.py
# Schedule: */5 8-13 * * 1-5 (cada 5 min, 8AM-1PM COT, Lun-Vie)
# Función: Insert OHLCV a usdcop_m5_ohlcv
```

#### DAG 2: `l0_macro_daily_update` (Nuevo)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'usdcop',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'l0_macro_daily_update',
    default_args=default_args,
    description='Actualización diaria de indicadores macroeconómicos',
    schedule_interval='55 7,15 * * 1-5',  # 07:55 y 15:30 COT
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['l0', 'macro', 'production']
) as dag:

    update_daily_sources = PythonOperator(
        task_id='update_daily_sources',
        python_callable=update_daily_macro_sources,
        # Investing.com, FRED (daily vars), BCRP
    )

    update_monthly_sources = PythonOperator(
        task_id='update_monthly_sources',
        python_callable=update_monthly_macro_sources,
        # FRED (monthly), DANE, Fedesarrollo
        # Solo ejecutar si es día 1-5 del mes
    )

    forward_fill_gaps = PythonOperator(
        task_id='forward_fill_gaps',
        python_callable=forward_fill_macro_indicators,
    )

    calculate_derived = PythonOperator(
        task_id='calculate_derived',
        python_callable=calculate_derived_fields,
        # rate_spread, pct_changes
    )

    validate_data = PythonOperator(
        task_id='validate_data',
        python_callable=validate_macro_data_quality,
    )

    update_daily_sources >> update_monthly_sources >> forward_fill_gaps >> calculate_derived >> validate_data
```

#### DAG 3: `l1_feature_transform` (Nuevo)

```python
with DAG(
    'l1_feature_transform',
    description='Transformación de OHLCV + Macro a Features de Inferencia',
    schedule_interval=None,  # Triggered by l0_ohlcv_realtime
    tags=['l1', 'features', 'production']
) as dag:

    @task
    def transform_to_inference_features(ohlcv_time: str):
        """
        Triggered cuando llega nueva barra OHLCV.
        1. Lee últimas 100 barras de usdcop_m5_ohlcv
        2. Lee último registro de macro_indicators_daily
        3. Calcula 13 features
        4. Inserta en inference_features_5m
        """
        # Implementation...
        pass
```

---

### 21.7 Diccionario de 37 Variables Macro

| # | Variable | Descripción | Frecuencia | Fuente | Impacto USD/COP |
|---|----------|-------------|------------|--------|-----------------|
| 1 | bond_yield5y_col | Bono Colombia 5Y | D | Investing | Positivo |
| 2 | bond_yield10y_col | Bono Colombia 10Y | D | Investing | Positivo |
| 3 | treasury_10y | US Treasury 10Y | D | FRED | Positivo |
| 4 | treasury_2y | US Treasury 2Y | D | FRED | Positivo |
| 5 | ibr_overnight | IBR Colombia | D | BanRep | Negativo |
| 6 | prime_rate | Prime Rate USA | D | BanRep | Positivo |
| 7 | fedfunds | Fed Funds Rate | M | FRED | Positivo |
| 8 | tpm_colombia | TPM BanRep | D | BanRep | Negativo |
| 9 | wti | WTI Crude Oil | D | Investing | Positivo |
| 10 | brent | Brent Crude Oil | D | Investing | Positivo |
| 11 | gold | Gold Futures | D | Investing | Mixto |
| 12 | coffee | Coffee Arabica | D | Investing | Positivo |
| 13 | dxy | Dollar Index | D | Investing | Positivo |
| 14 | usdmxn | USD/MXN | D | Investing | Positivo |
| 15 | usdclp | USD/CLP | D | Investing | Positivo |
| 16 | itcr | Índice Tasa Cambio Real | M | BanRep | Positivo |
| 17 | embi | EMBI Colombia | D | BCRP | Positivo |
| 18 | cci_colombia | Confianza Consumidor | M | Fedesarrollo | Negativo |
| 19 | ici_colombia | Confianza Industrial | M | Fedesarrollo | Negativo |
| 20 | vix | CBOE VIX | D | Investing | Positivo |
| 21 | ipc_colombia | IPC Colombia | M | BanRep | Positivo |
| 22 | cpi_usa | CPI USA | M | FRED | Positivo |
| 23 | pce_usa | PCE USA | M | FRED | Positivo |
| 24 | exports_col | Exportaciones COL | M | DANE | Negativo |
| 25 | imports_col | Importaciones COL | M | DANE | Positivo |
| 26 | terms_of_trade | Términos Intercambio | M | BanRep | Negativo |
| 27 | ied_inflow | IED Entrante | Q | BanRep | Negativo |
| 28 | ied_outflow | IED Saliente | Q | BanRep | Positivo |
| 29 | current_account | Cuenta Corriente BP | Q | BanRep | Positivo |
| 30 | reserves_intl | Reservas Internacionales | M | BanRep | Negativo |
| 31 | unemployment_usa | Desempleo USA | M | FRED | Negativo |
| 32 | industrial_prod_usa | Producción Industrial | M | FRED | Positivo |
| 33 | m2_supply_usa | M2 USA | M | FRED | Negativo |
| 34 | consumer_sentiment | Michigan Sentiment | M | FRED | Positivo |
| 35 | colcap | Índice COLCAP | D | Investing | Negativo |
| 36 | gdp_usa | GDP USA Real | Q | FRED | Positivo |
| 37 | rate_spread | Treasury 10Y - 2Y | D | Calculado | Variable |

---

### 21.8 Mapeo de 37 Variables → 13 Features del Modelo

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TRANSFORMACIÓN: RAW → MODEL FEATURES                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  37 Variables Raw                    13 Features Modelo                     │
│  ════════════════                    ═══════════════════                    │
│                                                                             │
│  usdcop_m5_ohlcv.close ──────────┬─► log_ret_5m   (returns)                │
│                                  ├─► log_ret_1h   (returns)                │
│                                  ├─► log_ret_4h   (returns)                │
│                                  │                                         │
│  usdcop_m5_ohlcv.high/low/close ─┼─► rsi_9        (technical)              │
│                                  ├─► atr_pct      (technical)              │
│                                  └─► adx_14       (technical)              │
│                                                                             │
│  macro.dxy ─────────────────────────► dxy_z        (zscore)                │
│                                  └──► dxy_change_1d (pct_change)           │
│                                                                             │
│  macro.vix ─────────────────────────► vix_z        (zscore)                │
│                                                                             │
│  macro.embi ────────────────────────► embi_z       (zscore)                │
│                                                                             │
│  macro.brent ───────────────────────► brent_change_1d (pct_change)         │
│                                                                             │
│  macro.treasury_10y ─┬                                                      │
│  macro.treasury_2y ──┴──────────────► rate_spread  (derived)               │
│                                                                             │
│  macro.usdmxn ──────────────────────► usdmxn_ret_1h (pct_change)           │
│                                                                             │
│  ══════════════════════════════════════════════════════════════════════    │
│  NOTA: Solo 7 de las 37 variables macro se usan directamente               │
│        Las demás están disponibles para futuras versiones del modelo       │
│  ══════════════════════════════════════════════════════════════════════    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 21.9 Validación de Datos

```sql
-- Vista de monitoreo de calidad de datos
CREATE OR REPLACE VIEW v_data_quality_monitor AS
SELECT
    'ohlcv' as table_name,
    COUNT(*) as total_rows,
    MAX(time) as latest_timestamp,
    EXTRACT(EPOCH FROM (NOW() - MAX(time)))/60 as minutes_since_update,
    COUNT(*) FILTER (WHERE close IS NULL) as null_values
FROM usdcop_m5_ohlcv
WHERE time > NOW() - INTERVAL '1 day'

UNION ALL

SELECT
    'macro_daily',
    COUNT(*),
    MAX(date)::timestamp,
    EXTRACT(EPOCH FROM (NOW() - MAX(date)::timestamp))/60/60/24,
    COUNT(*) FILTER (WHERE dxy IS NULL OR vix IS NULL OR embi IS NULL)
FROM macro_indicators_daily
WHERE date > CURRENT_DATE - 7

UNION ALL

SELECT
    'inference_features',
    COUNT(*),
    MAX(time),
    EXTRACT(EPOCH FROM (NOW() - MAX(time)))/60,
    COUNT(*) FILTER (WHERE is_valid = FALSE)
FROM inference_features_5m
WHERE time > NOW() - INTERVAL '1 day';
```

---

### 21.10 Resumen de Implementación

| Prioridad | Tarea | Esfuerzo | Dependencia |
|-----------|-------|----------|-------------|
| P0 | Crear tabla `macro_indicators_daily` | 2h | SQL Schema |
| P0 | Crear tabla `inference_features_5m` | 2h | SQL Schema |
| P1 | Adaptar `actualizador_hpc_v3.py` para Postgres | 4h | Tablas creadas |
| P1 | Crear DAG `l0_macro_daily_update` | 3h | Actualizador adaptado |
| P1 | Crear función `calculate_inference_features()` | 4h | feature_config.json |
| P2 | Crear DAG `l1_feature_transform` | 3h | Función de features |
| P2 | Integrar trigger con servicio de inferencia | 3h | DAG L1 |
| P3 | Vista de monitoreo y alertas | 2h | Todas las tablas |

**Total estimado: 23 horas de desarrollo**

---

*Sección 21 añadida el 2025-12-16 - Propuesta integral de arquitectura de datos*
*Basada en análisis de data/pipeline/, feature_config.json y DICCIONARIO_MACROECONOMICOS_FINAL.csv*
